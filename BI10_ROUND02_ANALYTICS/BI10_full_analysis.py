from __future__ import annotations
# %% [markdown]
# # BI10 Round 2 — Team ShootingStar: full analysis in one file
# Flow: DATA PROCESSING -> SETUP -> 0. EDA -> 1. Challenge 1 (FHS) -> 2. Challenge 2 (CES) ->
# 3. Customer output and health x engagement view -> 4. Figures -> 5. Written report.
#
# Input: the five source CSVs in `BI10_ROUND02_DATASET/`. One run rebuilds everything (about 6 minutes).
# Sections are numbered by task. Run top to bottom (python BI10_full_analysis.py) or cell by cell.

# %% [markdown]
# # DATA PROCESSING — Phase 1: audit the source CSVs and build the data model
# Check the five source CSVs, then write two layers of Parquet. Raw files stay unchanged.
#
# * `data_model/*.parquet` — event level, typed, deduplicated. Every score in Phase 2 is computed
#   from these, so features can be rebuilt "as of" any date (no future information).
# * `data_clean.parquet` — one row per customer (customer 360) for EDA and the dashboard.
#
# Runs first (about 1 minute). The only inputs are the five source CSVs
# in `./BI10_ROUND02_DATASET/`; nothing outside this repository and no database file is read.
#
# Every CSV is checked against the SHA-256 of the file as distributed. A copy that differs (e.g. a
# truncated download that keeps its file size) stops the run instead of silently losing rows.
# %%
import hashlib
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent   # repository root (this file lives in BI10_ROUND02_ANALYTICS/)
SOURCE = ROOT / 'BI10_ROUND02_DATASET'
EXPECTED_SHA256 = {   # files as distributed by the organiser
    'CALL': '161aee727226e4b4a6a173b4ee5ee64c77eb0e456e2a671f6c64aaafc16fc2b5',
    'CONTRACT': 'b86f54e8190822d9e46754fff14fd73b01b0de75d88f7f3f208bd61cc6b5d8ff',
    'CUSTOMER': 'be283a0c0a5107239bf74a88bc83e4b1d02e1fdc1b679bba9d7b6df861b87e1d',
    'LEADS': 'b23d96520d40ffbd518864be89997dfbff3746f4d4d57c294d64cfc475179af3',
    'PAST_BEHAVIOR': '0f7e64d6183ef2f867cbabcf04b576ff92669b61ae557cd5e0976a937112ccd9',
}
MODEL = ROOT / 'data_model'
MODEL.mkdir(exist_ok=True)
OUTPUT = ROOT / 'data_clean.parquet'
TABLES = ('CUSTOMER', 'CONTRACT', 'CALL', 'LEADS', 'PAST_BEHAVIOR')
CUTOFF = '2025-06-05'   # day after the last contract / lead / behaviour record (exclusive)
SIGNED = "('Active','Finished','Paid off','Signed','Written off')"

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='3GB'")
def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

audit = {'cutoff_exclusive': CUTOFF, 'source_files': {}}
for table in TABLES:
    csv = SOURCE / f'{table}.csv'
    if not csv.exists():
        raise FileNotFoundError(f'Put the distributed {table}.csv in {SOURCE.relative_to(ROOT)}/')
    digest = sha256(csv)
    if digest != EXPECTED_SHA256[table]:
        raise ValueError(f'{table}.csv differs from the distributed file (sha256 {digest}); re-download it')
    path = csv.as_posix().replace("'", "''")
    con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_csv('{path}', header=true, all_varchar=true, delim=',', quote='\"', escape='\"')")
    audit['source_files'][table] = {'read_from': csv.relative_to(ROOT).as_posix(), 'sha256': digest}
print('Source:', json.dumps(audit['source_files'], indent=1), flush=True)

def write(sql, name):
    """Write a query to data_model/<name>.parquet and return its row count."""
    path = MODEL / f'{name}.parquet'
    con.execute(f"COPY ({sql}) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    return con.execute(f'SELECT count(*) FROM {name}').fetchone()[0]

# %% [markdown]
# ## D.1 Source checks
# Keys, missingness, orphan rows, conflicts and event dates.
# %%
source_counts = {}
for table in TABLES:
    rows, clients, missing_keys = con.execute(
        f"SELECT count(*), count(DISTINCT SKP_CLIENT), count(*) - count(SKP_CLIENT) FROM {table}"
    ).fetchone()
    source_counts[table] = rows
    assert rows > 0 and missing_keys == 0, (table, rows, missing_keys)
    if table == 'CUSTOMER':
        assert rows == clients, 'Duplicate customer key'
    else:
        orphans = con.execute(
            f'SELECT count(*) FROM {table} t ANTI JOIN CUSTOMER c USING (SKP_CLIENT)'
        ).fetchone()[0]
        assert orphans == 0, (table, orphans)
    columns = [field[0] for field in con.execute(f'DESCRIBE {table}').fetchall()]
    checks = [f"count(*) FILTER (WHERE \"{c}\" IS NULL OR trim(\"{c}\") = '')" for c in columns]
    missing = con.execute('SELECT ' + ', '.join(checks) + f' FROM {table}').fetchone()
    print(f'{table}: {rows:,} rows; {clients:,} customers; null/blank by column: {dict(zip(columns, missing))}', flush=True)
audit['source_rows'] = source_counts

quality = {
    'call_key_duplicates': '''SELECT count(*) FROM (SELECT SKP_CLIENT, CALL_ID, TIME_OF_CALL_START
        FROM CALL GROUP BY 1,2,3 HAVING count(*) > 1)''',
    'credit_cases_spanning_customers': '''SELECT count(*) FROM
        (SELECT SKP_CREDIT_CASE FROM CONTRACT GROUP BY 1 HAVING count(DISTINCT SKP_CLIENT) > 1)''',
    'unparseable_activity_timestamps': '''SELECT
        (SELECT count(*) FROM CALL WHERE try_cast(TIME_OF_CALL_START AS TIMESTAMP) IS NULL)
      + (SELECT count(*) FROM LEADS WHERE try_cast(DTIME_CREATED AS TIMESTAMP) IS NULL)
      + (SELECT count(*) FROM PAST_BEHAVIOR WHERE try_cast(DTIME_EVENT AS TIMESTAMP) IS NULL)''',
    # repeated rows of one credit case must agree on everything except income/profession fields
    'cases_with_conflicting_core_fields': '''SELECT count(*) FROM (SELECT SKP_CREDIT_CASE FROM CONTRACT GROUP BY 1
        HAVING count(DISTINCT (NAME_CREDIT_STATUS, PRODUCT, AMT_CREDIT, PAYMENT_NUM, ANNUITY, INIT_PAY,
               APPLY_CONTRACT_TIME, APPROVE_CONTRACT_TIME, SIGN_CONTRACT_TIME,
               FIRST_DUE, SECOND_DUE, THIRD_DUE, FOURTH_DUE)) > 1)''',
}
for label, sql in quality.items():
    value = con.execute(sql).fetchone()[0]
    audit[label] = value
    print(label, value, flush=True)
    assert value == 0, (label, value)
for label, sql in {
    'signed_cases_without_sign_time': f'''SELECT count(DISTINCT SKP_CREDIT_CASE) FROM CONTRACT
        WHERE NAME_CREDIT_STATUS IN {SIGNED} AND (SIGN_CONTRACT_TIME IS NULL OR trim(SIGN_CONTRACT_TIME) = '')''',
    'lead_accepted_sentinel_3000': "SELECT count(*) FROM LEADS WHERE DATE_ACCEPTED LIKE '3000%'",
    'lead_exact_duplicate_extra_rows': 'SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM LEADS)) FROM LEADS',
    'household_income_zero_rows': "SELECT count(*) FROM CONTRACT WHERE try_cast(AMT_INCOME_HOUSEHOLD AS DOUBLE) = 0",
}.items():
    audit[label] = con.execute(sql).fetchone()[0]
    print(label, audit[label], flush=True)

# %% [markdown]
# ## D.2 Contract cases
# `CONTRACT` repeats a credit case 2 or 8 times. Copies agree on every field except income, income
# type and profession, where one copy carries the value and the others carry `0` / `XNA`.
# The zero is a placeholder of the copy, not a customer without income, so a case keeps its
# positive income. Incomes outside 1M–200M VND per month (e.g. 9,000,000,000) are set to missing.
# A *genuine* conflict is two different positive incomes on one case.
# %%
n_cases = write(f"""
SELECT CAST(SKP_CREDIT_CASE AS BIGINT) SKP_CREDIT_CASE,
       CAST(any_value(SKP_CLIENT) AS BIGINT) SKP_CLIENT,
       count(*)::INT SOURCE_ROWS,
       any_value(NAME_CREDIT_STATUS) STATUS,
       nullif(any_value(PRODUCT), '') PRODUCT,
       any_value(NAME_CREDIT_STATUS) IN {SIGNED} IS_SIGNED,
       try_cast(any_value(AMT_CREDIT) AS DOUBLE) CREDIT_AMOUNT,
       try_cast(any_value(PAYMENT_NUM) AS DOUBLE) TENOR,
       try_cast(any_value(INIT_PAY) AS DOUBLE) INITIAL_PAYMENT,
       try_cast(any_value(ANNUITY) AS DOUBLE) ANNUITY,
       try_cast(max(AMT_BILLING) AS DOUBLE) BILLING_AMOUNT,
       try_cast(max(FLAG_INS) AS INT) INSURED,
       max(nullif(TRANSAC, '')) TRANSAC,
       try_cast(any_value(APPLY_CONTRACT_TIME) AS TIMESTAMP) APPLY_TIME,
       try_cast(any_value(APPROVE_CONTRACT_TIME) AS TIMESTAMP) APPROVE_TIME,
       try_cast(any_value(SIGN_CONTRACT_TIME) AS TIMESTAMP) SIGN_TIME,
       try_cast(any_value(FIRST_DUE) AS INT) D1, try_cast(any_value(SECOND_DUE) AS INT) D2,
       try_cast(any_value(THIRD_DUE) AS INT) D3, try_cast(any_value(FOURTH_DUE) AS INT) D4,
       max(try_cast(AMT_INCOME_MAIN AS DOUBLE)) INCOME_RAW_MAX,
       max(try_cast(AMT_INCOME_MAIN AS DOUBLE)) FILTER (WHERE try_cast(AMT_INCOME_MAIN AS DOUBLE) BETWEEN 1e6 AND 2e8) INCOME_MAIN,
       count(DISTINCT try_cast(AMT_INCOME_MAIN AS DOUBLE)) FILTER (WHERE try_cast(AMT_INCOME_MAIN AS DOUBLE) > 0) > 1 INCOME_GENUINE_CONFLICT,
       count(*) FILTER (WHERE try_cast(AMT_INCOME_MAIN AS DOUBLE) = 0) > 0 AND count(*) > 1 INCOME_PLACEHOLDER_ZERO,
       max(nullif(NAME_INCOME_TYPE, 'XNA')) INCOME_TYPE,
       max(nullif(CODE_PROFESSION, 'XNA')) PROFESSION
FROM CONTRACT GROUP BY SKP_CREDIT_CASE
ORDER BY SKP_CLIENT, SKP_CREDIT_CASE""", 'contract_cases')
contract_audit = con.execute('''SELECT sum(SOURCE_ROWS), count(*),
    sum(INCOME_PLACEHOLDER_ZERO::INT), sum(INCOME_GENUINE_CONFLICT::INT),
    sum((INCOME_RAW_MAX > 2e8)::INT), sum((INCOME_MAIN IS NULL)::INT),
    sum((IS_SIGNED AND SIGN_TIME IS NULL)::INT)
    FROM contract_cases''').fetchone()
audit['contract'] = dict(zip(['source_rows', 'credit_cases', 'cases_with_placeholder_zero_income',
    'cases_with_genuine_income_conflict', 'cases_with_income_above_200M', 'cases_without_valid_income',
    'signed_cases_without_sign_time'], contract_audit))
audit['contract']['row_inflation'] = round(contract_audit[0] / contract_audit[1], 3)
assert contract_audit[0] == source_counts['CONTRACT'] and n_cases == contract_audit[1]
print('Contract:', audit['contract'], flush=True)

# %% [markdown]
# ## D.3 Calls
# `CALL` is the telesales / collection / service call log (2025-01 to 2026-08). 85% of rows have no
# response code. `RESPONSE_RESULT_2` is mapped to one outcome group so that customer behaviour
# (answering, buying, refusing, owing) is separated from company activity (dialling).
# `RPC` = right-party contact: the customer was actually reached.
# %%
OUTCOME = r"""CASE
  WHEN r IS NULL OR r IN ('', 'none') THEN 'no_contact'
  WHEN r IN ('No_Answer','Client busy','No response','CallDisconnected','Drop_call_I','ICre_Drop','DE_DROP',
             'No answer Retention','OWNNotAvailableRPC','OWNNotAvailableNonRPC','OWNNotReachManual',
             'OWNAnswerMachineNonRPC','OWNAnswerMachineRPC','OWNCallDisconnectedNonRPC','OWNCallDisconnectedRPC',
             'OWNClientHungUp') THEN 'no_contact'
  WHEN r IN ('Wrong Phone','Wrong_phone_C','Relative','Relatives','OtherPerson','ClientFamily','OWNDifferentPerson')
       OR r LIKE 'OWNUnidentified%' OR r LIKE 'OWNWrongContact%' OR r LIKE 'OWNCheckWrongContact%'
       OR r LIKE 'OWNDeathSusp%' OR r ILIKE '%nonrpc%' OR r LIKE 'OWNNewSpouse%' THEN 'wrong_party'
  WHEN r LIKE 'OWN%' OR r IN ('ppSimple','Chuyen_Collection_I','Collection_C','ICre_Transfer to Collection',
       'Yeu_cau_mien_giam/gia_han_I','ResultAskRecallRpcRestrict','DoesNotCollaborate')
       OR r LIKE 'ICre_Yeu_cau_mien%' THEN 'collection'
  WHEN r IN ('Sign successful','Made_App_Successful','Transfer for signing','Send_to_SA','Sign Agree','Client agrees',
             'Appointment with SA','Transfer to Sale','Transfer for sales','Signed already','Push Sign') THEN 'sales_conversion'
  WHEN r IN ('More consultation','Client considers','Pending','AskRecall','Ask to call back','CallBack','Dropped_OTP') THEN 'sales_interest'
  WHEN r IN ('Not eligible') THEN 'sales_ineligible'
  WHEN r IN ('Refuse','Sign refuse','Do Not Call','No need of using card','No longer in need','Low credit limit',
             'Wrong product','DE_Cancel application')
       OR r LIKE 'Ngung_nhan_quang_cao%' OR r LIKE 'ICre_Ngung_nhan_quang_cao%' THEN 'sales_decline'
  WHEN r ILIKE '%retention%' OR r LIKE 'Termination%' OR r IN ('Disagree termination','Not eligible for termination',
       'Huy_Hop_dong_I','BNPL_ICre_Huy hop dong','Bao_hiem_Thanh_ly_huy_I')
       OR r LIKE 'ICre_Cham dut%' OR r LIKE 'Thanh_ly%' OR r LIKE 'BNPL_ICre_Thanh ly%' THEN 'retention'
  WHEN CALL_TYPE = 'Inbound' OR r LIKE '%\_I' ESCAPE '\' OR r LIKE '%\_C' ESCAPE '\' OR r LIKE 'ICre%'
       OR r LIKE 'BNPL\_%' ESCAPE '\' OR r LIKE 'CCre%' OR r LIKE 'HAPP%' THEN 'service'
  ELSE 'other' END"""
n_calls = write(f"""
WITH c AS (SELECT *, RESPONSE_RESULT_2 r FROM CALL)
SELECT CAST(SKP_CLIENT AS BIGINT) SKP_CLIENT, CALL_ID,
       try_cast(TIME_OF_CALL_START AS TIMESTAMP) START_AT,
       try_cast(DURATIONS AS INT) DURATION_SECONDS,
       CALL_TYPE, nullif(PRODUCT_TYPE, '') PRODUCT_TYPE, CALL_SOURCE, OFF_BY, OFFER_TYPE,
       r RESPONSE, CODE_DISPOSITION DISPOSITION, nullif(CALL_RESULT_SYSTEM, '') SYSTEM_RESULT,
       {OUTCOME} OUTCOME_GROUP
FROM c ORDER BY SKP_CLIENT, START_AT, CALL_ID, RESPONSE""", 'calls')
con.execute("""CREATE OR REPLACE VIEW calls_rpc AS SELECT *,
    OUTCOME_GROUP NOT IN ('no_contact','wrong_party','other') RPC FROM calls""")
outcome_map = con.execute('''SELECT OUTCOME_GROUP, RESPONSE, count(*) calls,
    round(avg((CALL_TYPE='Inbound')::INT), 3) inbound_share FROM calls GROUP BY 1,2 ORDER BY 1, 3 DESC, 2''').df()
outcome_map.to_csv(MODEL / 'call_outcome_map.csv', index=False)
audit['call_outcome_groups'] = (outcome_map.groupby('OUTCOME_GROUP')['calls'].sum()).to_dict()
assert n_calls == source_counts['CALL']
print('Call outcome groups:', audit['call_outcome_groups'], flush=True)

# %% [markdown]
# ## D.4 Leads and behaviour
# Exact duplicate lead rows are dropped. The year-3000 acceptance date is the documented SOB
# sentinel (no "accept" click), so it becomes null plus a flag. Each record is tagged by who started it:
# leads from GMA (app), landing page and SOB, and scoring through app / web / partner / store
# channels are customer initiated; RA / SA / TLS / RTDM leads and TLS scoring are company initiated.
# %%
n_leads = write("""
SELECT DISTINCT CAST(SKP_CLIENT AS BIGINT) SKP_CLIENT,
       try_cast(DTIME_CREATED AS TIMESTAMP) CREATED_AT,
       CASE WHEN DATE_ACCEPTED LIKE '3000%' THEN NULL ELSE try_cast(DATE_ACCEPTED AS TIMESTAMP) END ACCEPTED_AT,
       DATE_ACCEPTED LIKE '3000%' ACCEPTED_SENTINEL,
       nullif(DESC_FIN_REASON, '') REASON, CODE_PRODUCT_TYPE PRODUCT,
       try_cast(FLAG_ASSIGNED AS INT) ASSIGNED, try_cast(MAX_OFFER AS DOUBLE) MAX_OFFER,
       LEAD_SOURCE, LEAD_SOURCE IN ('GMA','LANDING PAGE','SOB') SELF_INITIATED
FROM LEADS ORDER BY ALL""", 'leads')
n_behavior = write("""
SELECT CAST(SKP_CLIENT AS BIGINT) SKP_CLIENT, try_cast(DTIME_EVENT AS TIMESTAMP) EVENT_AT, ACTION, PRODUCT_CODE,
       CASE WHEN ACTION LIKE 'APP %' THEN 'APP' WHEN ACTION LIKE 'WEB %' THEN 'WEB'
            WHEN ACTION LIKE 'Partner%' THEN 'PARTNER' WHEN ACTION LIKE 'SA %' THEN 'STORE_SA'
            WHEN ACTION LIKE 'TLS %' THEN 'TELESALES' WHEN ACTION LIKE 'SOB%' THEN 'SOB'
            WHEN ACTION = 'WALK_IN' THEN 'WALK_IN' ELSE 'OTHER' END CHANNEL,
       ACTION NOT LIKE 'TLS %' SELF_INITIATED
FROM PAST_BEHAVIOR ORDER BY ALL""", 'behavior')
n_customers = write("""
SELECT CAST(SKP_CLIENT AS BIGINT) SKP_CLIENT, NAME_GENDER GENDER, NAME_EDUCATION_TYPE EDUCATION,
       try_cast(DATE_BIRTH AS DATE) DATE_BIRTH, try_cast(CNT_CHILDREN AS DOUBLE) CHILDREN,
       try_cast(CNT_PERSON_DEPENDENT AS DOUBLE) DEPENDENTS, NAME_SALARY_FREQUENCY SALARY_FREQUENCY,
       try_cast(AVG_SESSION_PER_WEEK_2025 AS DOUBLE) APP_SESSIONS_PER_WEEK_2025
FROM CUSTOMER ORDER BY SKP_CLIENT""", 'customers')   # address is not carried forward (not used, personal data)
assert n_leads == source_counts['LEADS'] - audit['lead_exact_duplicate_extra_rows']
assert n_behavior == source_counts['PAST_BEHAVIOR'] and n_customers == source_counts['CUSTOMER']
audit['activity_ranges'] = {name: [str(v) for v in con.execute(sql).fetchone()] for name, sql in {
    'contract_apply': 'SELECT min(APPLY_TIME), max(APPLY_TIME) FROM contract_cases',
    'leads': 'SELECT min(CREATED_AT), max(CREATED_AT) FROM leads',
    'behavior': 'SELECT min(EVENT_AT), max(EVENT_AT) FROM behavior',
    'calls': 'SELECT min(START_AT), max(START_AT) FROM calls'}.items()}
print('Activity ranges:', audit['activity_ranges'], flush=True)

# %% [markdown]
# ## D.5 Customer 360
# Aggregate each event table first, then left-join to the customer list (no fan-out).
# Contract status is a snapshot at extraction, not a dated history. Call columns are split
# into before / after the cutoff: only "before" columns may be used as features.
# %%
select_sql = f"""
WITH k AS (
  SELECT SKP_CLIENT,
         sum(SOURCE_ROWS)::BIGINT CONTRACT_SOURCE_ROWS, count(*)::BIGINT CONTRACT_CASE_COUNT,
         count(*) FILTER (WHERE IS_SIGNED)::BIGINT CONTRACT_SIGNED_COUNT,
         count(*) FILTER (WHERE STATUS = 'Active')::BIGINT CONTRACT_ACTIVE_COUNT,
         count(*) FILTER (WHERE STATUS = 'Written off')::BIGINT CONTRACT_WRITTEN_OFF_COUNT,
         count(*) FILTER (WHERE STATUS = 'Rejected')::BIGINT CONTRACT_REJECTED_COUNT,
         count(*) FILTER (WHERE STATUS = 'Cancelled')::BIGINT CONTRACT_CANCELLED_COUNT,
         count(DISTINCT PRODUCT) FILTER (WHERE IS_SIGNED)::BIGINT CONTRACT_PRODUCT_COUNT,
         sum(CREDIT_AMOUNT) FILTER (WHERE IS_SIGNED) CONTRACT_SIGNED_CREDIT_TOTAL,
         sum(CREDIT_AMOUNT) FILTER (WHERE STATUS = 'Active') CONTRACT_ACTIVE_CREDIT_TOTAL,
         sum(ANNUITY) FILTER (WHERE STATUS = 'Active' AND PRODUCT IN ('CD','TW','CL','CW')) CONTRACT_ACTIVE_ANNUITY_TOTAL,
         count(*) FILTER (WHERE IS_SIGNED AND D1 + D2 + D3 + D4 > 0)::BIGINT CONTRACT_CASES_WITH_DUE,
         sum(D1 + D2 + D3 + D4) FILTER (WHERE IS_SIGNED)::BIGINT DUE_FLAGS_TOTAL,
         sum(INCOME_PLACEHOLDER_ZERO::INT)::BIGINT CONTRACT_INCOME_PLACEHOLDER_CASES,
         sum(INCOME_GENUINE_CONFLICT::INT)::BIGINT CONTRACT_INCOME_GENUINE_CONFLICT_CASES,
         min(coalesce(SIGN_TIME, APPROVE_TIME, APPLY_TIME)) FILTER (WHERE IS_SIGNED) FIRST_CONTRACT_SIGNED_AT,
         max(coalesce(SIGN_TIME, APPROVE_TIME, APPLY_TIME)) FILTER (WHERE IS_SIGNED) LAST_CONTRACT_SIGNED_AT,
         max(APPLY_TIME) LAST_CONTRACT_APPLICATION_AT,
         arg_max(STATUS, (APPLY_TIME, SKP_CREDIT_CASE)) LATEST_CONTRACT_STATUS,
         arg_max(PRODUCT, (APPLY_TIME, SKP_CREDIT_CASE)) LATEST_CONTRACT_PRODUCT,
         arg_max(INCOME_MAIN, (APPLY_TIME, SKP_CREDIT_CASE)) FILTER (WHERE INCOME_MAIN IS NOT NULL) LATEST_INCOME_MAIN,
         arg_max(INCOME_TYPE, (APPLY_TIME, SKP_CREDIT_CASE)) FILTER (WHERE INCOME_TYPE IS NOT NULL) LATEST_INCOME_TYPE
  FROM contract_cases GROUP BY 1),
a AS (
  SELECT SKP_CLIENT,
         count(*) FILTER (WHERE START_AT < '{CUTOFF}')::BIGINT CALLS_PRE,
         count(*) FILTER (WHERE START_AT < '{CUTOFF}' AND RPC)::BIGINT CALLS_RPC_PRE,
         count(*) FILTER (WHERE START_AT < '{CUTOFF}' AND CALL_TYPE = 'Inbound')::BIGINT CALLS_INBOUND_PRE,
         count(*) FILTER (WHERE START_AT < '{CUTOFF}' AND OUTCOME_GROUP = 'collection')::BIGINT CALLS_COLLECTION_PRE,
         count(*) FILTER (WHERE START_AT < '{CUTOFF}' AND OUTCOME_GROUP IN ('sales_conversion','sales_interest'))::BIGINT CALLS_SALES_POSITIVE_PRE,
         max(START_AT) FILTER (WHERE START_AT < '{CUTOFF}' AND RPC) LAST_RPC_CALL_PRE_AT,
         count(*) FILTER (WHERE START_AT >= '{CUTOFF}')::BIGINT CALLS_POST,
         count(*) FILTER (WHERE START_AT >= '{CUTOFF}' AND RPC)::BIGINT CALLS_RPC_POST,
         count(*) FILTER (WHERE START_AT >= '{CUTOFF}' AND OUTCOME_GROUP = 'collection')::BIGINT CALLS_COLLECTION_POST,
         count(*) FILTER (WHERE START_AT >= '{CUTOFF}' AND OUTCOME_GROUP IN ('sales_conversion','sales_interest'))::BIGINT CALLS_SALES_POSITIVE_POST
  FROM calls_rpc GROUP BY 1),
l AS (
  SELECT SKP_CLIENT, count(*)::BIGINT LEAD_COUNT,
         count(*) FILTER (WHERE SELF_INITIATED)::BIGINT LEAD_SELF_INITIATED_COUNT,
         count(*) FILTER (WHERE REASON = 'Deal')::BIGINT LEAD_DEAL_COUNT,
         max(MAX_OFFER) LEAD_MAX_OFFER, max(CREATED_AT) LAST_LEAD_AT
  FROM leads GROUP BY 1),
b AS (
  SELECT SKP_CLIENT, count(*)::BIGINT BEHAVIOR_EVENT_COUNT,
         count(*) FILTER (WHERE SELF_INITIATED)::BIGINT BEHAVIOR_SELF_INITIATED_COUNT,
         count(DISTINCT PRODUCT_CODE)::BIGINT BEHAVIOR_PRODUCT_COUNT, max(EVENT_AT) LAST_BEHAVIOR_AT
  FROM behavior GROUP BY 1)
SELECT c.SKP_CLIENT, c.GENDER, c.EDUCATION, c.DATE_BIRTH, c.APP_SESSIONS_PER_WEEK_2025,
       k.* EXCLUDE (SKP_CLIENT),
       coalesce(a.CALLS_PRE, 0) CALLS_PRE, coalesce(a.CALLS_RPC_PRE, 0) CALLS_RPC_PRE,
       coalesce(a.CALLS_INBOUND_PRE, 0) CALLS_INBOUND_PRE, coalesce(a.CALLS_COLLECTION_PRE, 0) CALLS_COLLECTION_PRE,
       coalesce(a.CALLS_SALES_POSITIVE_PRE, 0) CALLS_SALES_POSITIVE_PRE, a.LAST_RPC_CALL_PRE_AT,
       coalesce(a.CALLS_POST, 0) CALLS_POST, coalesce(a.CALLS_RPC_POST, 0) CALLS_RPC_POST,
       coalesce(a.CALLS_COLLECTION_POST, 0) CALLS_COLLECTION_POST,
       coalesce(a.CALLS_SALES_POSITIVE_POST, 0) CALLS_SALES_POSITIVE_POST,
       coalesce(l.LEAD_COUNT, 0) LEAD_COUNT, coalesce(l.LEAD_SELF_INITIATED_COUNT, 0) LEAD_SELF_INITIATED_COUNT,
       coalesce(l.LEAD_DEAL_COUNT, 0) LEAD_DEAL_COUNT, l.LEAD_MAX_OFFER, l.LAST_LEAD_AT,
       coalesce(b.BEHAVIOR_EVENT_COUNT, 0) BEHAVIOR_EVENT_COUNT,
       coalesce(b.BEHAVIOR_SELF_INITIATED_COUNT, 0) BEHAVIOR_SELF_INITIATED_COUNT,
       coalesce(b.BEHAVIOR_PRODUCT_COUNT, 0) BEHAVIOR_PRODUCT_COUNT, b.LAST_BEHAVIOR_AT
FROM customers c
LEFT JOIN k USING (SKP_CLIENT) LEFT JOIN a USING (SKP_CLIENT)
LEFT JOIN l USING (SKP_CLIENT) LEFT JOIN b USING (SKP_CLIENT)
ORDER BY SKP_CLIENT"""

# %% [markdown]
# ## D.6 Reconcile and save
# Customer grain, source totals and the cutoff split must all reconcile before the file is replaced.
# %%
TEMP = ROOT / 'data_clean.partial.parquet'
TEMP.unlink(missing_ok=True)
try:
    con.execute(f"COPY ({select_sql}) TO '{TEMP.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"CREATE VIEW final AS SELECT * FROM read_parquet('{TEMP.as_posix()}')")
    row = con.execute('''SELECT count(*), count(DISTINCT SKP_CLIENT), sum(CONTRACT_SOURCE_ROWS), sum(CONTRACT_CASE_COUNT),
        sum(CALLS_PRE + CALLS_POST), sum(LEAD_COUNT), sum(BEHAVIOR_EVENT_COUNT) FROM final''').fetchone()
    assert row[:2] == (source_counts['CUSTOMER'],) * 2, 'customer grain changed'
    assert row[2:] == (source_counts['CONTRACT'], n_cases, source_counts['CALL'], n_leads, n_behavior), row
    assert con.execute(f'''SELECT count(*) FROM final WHERE CONTRACT_ACTIVE_COUNT > CONTRACT_SIGNED_COUNT
        OR CALLS_RPC_PRE > CALLS_PRE OR LAST_RPC_CALL_PRE_AT >= '{CUTOFF}' OR LAST_LEAD_AT >= '{CUTOFF}'
        OR LAST_BEHAVIOR_AT >= '{CUTOFF}' ''').fetchone()[0] == 0
    TEMP.replace(OUTPUT)
except Exception:
    TEMP.unlink(missing_ok=True)
    raise
audit['customer_360'] = {'rows': row[0], 'columns': len(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{OUTPUT.as_posix()}')").fetchall())}
audit['data_model_files'] = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(MODEL.glob('*.parquet'))}
(MODEL / 'audit_summary.json').write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
print(f'Wrote {OUTPUT.name}: {row[0]:,} customers. Contract rows/cases {row[2]:,}/{row[3]:,}; '
      f'calls {row[4]:,}; leads {row[5]:,}; behaviour {row[6]:,}.', flush=True)
print('Data model (MB):', audit['data_model_files'], flush=True)

# %% [markdown]
# ## D.7 Output
# `data_model/` (event level) feeds every score; `data_clean.parquet` is the one-row-per-customer view.
# Historical arrears are flags on instalments 1–4, not current balances.


# %% [markdown]
# # SETUP — shared paths, data views, metrics, WoE scorecard and chart style
# %%
"""Shared paths, data views, metrics, the WoE scorecard and chart style used by every section below.

Everything reads only the Parquet files written by the DATA PROCESSING section. Nothing here reads the raw CSVs.
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap

from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL = REPO / 'data_model'
CUSTOMER_360 = REPO / 'data_clean.parquet'
TABLES, FIGURES, OUTPUTS, REPORTS = (ROOT / name for name in ('tables', 'figures', 'outputs', 'reports'))
for folder in (TABLES, FIGURES, OUTPUTS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CUSTOMERS = 1_000_528
CUTOFF = '2025-06-05'          # scoring date T: day after the last contract / lead / behaviour record
CALLS_START = '2025-01-01'     # first call in the log
CALLS_END = '2026-09-01'       # exclusive; calls after T are the independent future window
SIGNED = ('Active', 'Finished', 'Paid off', 'Signed', 'Written off')
INSTALLMENT = ('CD', 'TW', 'CL', 'CW')
REVOLVING = ('BNPL', 'CC')
PRODUCT_NAMES = {'CD': 'Consumer durables', 'TW': 'Two-wheeler', 'CL': 'Cash loan', 'CW': 'Cash (walk-in)',
                 'BNPL': 'Buy now pay later', 'CC': 'Credit card'}


# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------
def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with one view per Parquet file plus `k`, the contract-case view used by the scores."""
    missing = [n for n in ('contract_cases', 'calls', 'leads', 'behavior', 'customers') if not (MODEL / f'{n}.parquet').exists()]
    if missing:
        raise FileNotFoundError(f'Run the DATA PROCESSING section first; missing data_model/{missing}')
    con = duckdb.connect()
    con.execute("SET threads=4")
    for name in ('contract_cases', 'calls', 'leads', 'behavior', 'customers'):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{(MODEL / f'{name}.parquet').as_posix()}')")
    con.execute(f"CREATE VIEW customer_360 AS SELECT * FROM read_parquet('{CUSTOMER_360.as_posix()}')")
    con.execute("""CREATE VIEW calls_rpc AS SELECT *,
        OUTCOME_GROUP NOT IN ('no_contact','wrong_party','other') AS RPC FROM calls""")
    # t_ref: sign time, falling back to approval / application time (pre-2017 contracts lack sign time).
    # maturity: t_ref + tenor months, instalment products only.
    con.execute(f"""CREATE TABLE k AS
        SELECT SKP_CREDIT_CASE, SKP_CLIENT, INCOME_MAIN AS inc, STATUS AS status, PRODUCT AS product,
               CREDIT_AMOUNT AS credit, TENOR AS tenor, INITIAL_PAYMENT AS init_pay, ANNUITY AS annuity,
               APPLY_TIME AS t_apply, APPROVE_TIME AS t_appr, SIGN_TIME AS t_sign,
               D1 AS d1, D2 AS d2, D3 AS d3, D4 AS d4, IS_SIGNED AS is_signed,
               coalesce(SIGN_TIME, APPROVE_TIME, APPLY_TIME) AS t_ref,
               CASE WHEN PRODUCT IN {INSTALLMENT} AND TENOR > 0
                    THEN coalesce(SIGN_TIME, APPROVE_TIME, APPLY_TIME) + to_months(CAST(TENOR AS INTEGER)) END AS maturity,
               STATUS = 'Written off' AS wo
        FROM contract_cases""")
    return con


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> pd.DataFrame:
    df.to_csv(TABLES / f'{name}.csv', index=index)
    return df


def save_json(obj, name: str) -> None:
    (TABLES / f'{name}.json').write_text(json.dumps(obj, indent=2, default=_json_default), encoding='utf-8')


def load_json(name: str):
    return json.loads((TABLES / f'{name}.json').read_text(encoding='utf-8'))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def ks_stat(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, -np.asarray(s))
    return float(np.max(tpr - fpr))


def metrics(y, s) -> dict:
    """Discrimination of score s for a 0/1 outcome y where LOW score should mean outcome = 1."""
    y = np.asarray(y)
    auc = roc_auc_score(y, -np.asarray(s))
    return {'n': int(len(y)), 'event_rate': round(float(y.mean()), 4), 'auc': round(auc, 4),
            'gini': round(2 * auc - 1, 4), 'ks': round(ks_stat(y, s), 4)}


def stratified_auc(df: pd.DataFrame, y: str, s: str, by: str, higher_is_event: bool = False) -> dict:
    """AUC computed inside each stratum and averaged by stratum size (removes the stratum effect)."""
    rows = {}
    for key, g in df.groupby(by, observed=True):
        if g[y].nunique() == 2 and len(g) >= 500:
            score = g[s] if higher_is_event else -g[s]
            rows[str(key)] = {'n': len(g), 'event_rate': round(g[y].mean(), 4), 'auc': round(roc_auc_score(g[y], score), 4)}
    total = sum(v['n'] for v in rows.values())
    return {'weighted_auc': round(sum(v['auc'] * v['n'] for v in rows.values()) / total, 4), 'strata': rows}


def decile_table(y, s, q: int = 10) -> pd.DataFrame:
    """Decile 1 = lowest score. Ties are split by rank so every decile has the same size."""
    d = pd.DataFrame({'y': np.asarray(y), 's': np.asarray(s)})
    d['decile'] = pd.qcut(d['s'].rank(method='first'), q, labels=range(1, q + 1))
    t = d.groupby('decile', observed=True).agg(n=('y', 'size'), events=('y', 'sum'),
                                               score_min=('s', 'min'), score_max=('s', 'max')).reset_index()
    t['event_rate'] = t['events'] / t['n']
    t['lift_vs_avg'] = t['event_rate'] / d['y'].mean()
    t['cum_event_capture_from_low'] = t['events'].cumsum() / t['events'].sum()
    return t


def psi(expected, actual, bins) -> float:
    e = np.histogram(expected, bins)[0] / len(expected) + 1e-6
    a = np.histogram(actual, bins)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


# --------------------------------------------------------------------------------------
# WoE scorecard (points-to-double-odds scale)
# --------------------------------------------------------------------------------------
PDO, ANCHOR_SCORE, ANCHOR_BAD = 10.0, 60.0, 0.02   # 2% stress probability = 60 points; odds x2 = +10
FACTOR = PDO / np.log(2)
OFFSET = ANCHOR_SCORE - FACTOR * np.log((1 - ANCHOR_BAD) / ANCHOR_BAD)


def woe_table(df: pd.DataFrame, feat: str, target: str = 'bad', smooth: float = 0.5) -> pd.DataFrame:
    g = df.groupby(feat, observed=True)[target].agg(n='count', bad='sum').reset_index()
    g['good'] = g['n'] - g['bad']
    tb, tg = g['bad'].sum(), g['good'].sum()
    g['dist_bad'] = (g['bad'] + smooth) / (tb + smooth * len(g))
    g['dist_good'] = (g['good'] + smooth) / (tg + smooth * len(g))
    g['woe'] = np.log(g['dist_good'] / g['dist_bad'])
    g['iv_part'] = (g['dist_good'] - g['dist_bad']) * g['woe']
    g['bad_rate'] = g['bad'] / g['n']
    g.insert(0, 'feature', feat)
    return g.rename(columns={feat: 'bin'})


def apply_woe(df: pd.DataFrame, woes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for f, w in woes.items():
        X[f] = df[f].map(dict(zip(w['bin'], w['woe']))).fillna(0.0)   # unseen bin -> neutral
    return X


class Scorecard:
    """WoE + logistic regression. A feature whose coefficient contradicts its WoE direction
    (overlap with stronger features) is dropped and the model refitted, so every point is explainable."""

    def __init__(self, name: str, feats: list[str], pillar: dict[str, str], log=print):
        self.name, self.feats, self.pillar, self.log = name, list(feats), pillar, log

    def fit(self, df: pd.DataFrame, target: str = 'bad') -> 'Scorecard':
        self.woes = {f: woe_table(df, f, target) for f in self.feats}
        self.dropped = {}
        active = list(self.feats)
        while True:
            lr = LogisticRegression(C=1e3, max_iter=5000).fit(apply_woe(df, {f: self.woes[f] for f in active}), df[target])
            coef = dict(zip(active, lr.coef_[0]))
            wrong = [f for f, b in coef.items() if b >= 0]     # bad=1 and WoE=ln(good/bad) => beta must be < 0
            if not wrong:
                break
            drop = max(wrong, key=lambda f: coef[f])
            self.dropped[drop] = 'coefficient sign contradicts univariate WoE (collinear with stronger features)'
            self.log(f'[{self.name}] dropping {drop}: sign contradicts WoE')
            active.remove(drop)
        self.active, self.coef, self.intercept = active, coef, float(lr.intercept_[0])
        rows = []
        for f in active:
            w = self.woes[f].copy()
            w['points'] = -FACTOR * coef[f] * w['woe']     # signed points; higher = healthier
            w['beta'] = coef[f]
            rows.append(w)
        self.pts = pd.concat(rows, ignore_index=True)
        self.pts['pillar'] = self.pts['feature'].map(self.pillar)
        self.pts['model'] = self.name
        self.base_points = OFFSET - FACTOR * self.intercept
        return self

    def points(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for f, g in self.pts.groupby('feature'):
            out[f] = df[f].map(dict(zip(g['bin'], g['points']))).fillna(0.0)
        return out

    def score(self, df: pd.DataFrame, clip: bool = True) -> pd.Series:
        s = self.base_points + self.points(df).sum(axis=1)
        return s.clip(0, 100).round(1) if clip else s

    def weights(self) -> pd.Series:
        """Points range per feature as a share of the model's total points range."""
        span = self.pts.groupby('feature')['points'].agg(lambda p: p.max() - p.min())
        return (100 * span / span.sum()).sort_values(ascending=False)


# --------------------------------------------------------------------------------------
# Chart system (validated with the dataviz palette validator, light surface #fcfcfb):
# * categorical identity: blue, orange, aqua in that fixed order (all adjacent CVD checks pass);
# * emphasis: the story in ACCENT, context in MUTED gray;
# * FHS bands: diverging blue (healthy) - gray (neutral) - red (stressed), always direct-labelled;
# * engagement tiers: one-hue orange ordinal ramp; magnitude: one-hue blue ramp.
# Text always uses ink tokens, never a series colour. Every chart has a source/caption line.
# --------------------------------------------------------------------------------------
SURFACE, INK, INK_2, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e34948'
ACCENT, MUTED = BLUE, '#c3c2b7'
BLUE_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
BAND_LABELS_ORDER = ['A Thriving', 'B Healthy', 'C Stable', 'D Fragile', 'E Stressed']
BAND_COLORS = dict(zip(BAND_LABELS_ORDER, ['#1c5cab', '#5598e7', '#b9b7ae', '#e86a5f', '#a8201f']))
TIER_ORDER = ['Passive', 'Warm', 'Engaged', 'Highly engaged']
TIER_COLORS = dict(zip(TIER_ORDER, ['#f29a6f', '#eb6834', '#c4501f', '#8a3510']))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.labelcolor': INK_2, 'text.color': INK, 'axes.labelsize': 11.5,
    'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.spines.bottom': True, 'axes.edgecolor': AXIS, 'axes.linewidth': .8,
    'xtick.major.size': 0, 'ytick.major.size': 0,
    'grid.color': GRID, 'grid.linewidth': .6, 'legend.frameon': False, 'legend.fontsize': 10.5,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.titlelocation': 'left',
    'lines.linewidth': 2, 'patch.linewidth': 0})


def frame(title: str, subtitle: str, footer: str, ncols: int = 1, nrows: int = 1, figsize=(12.8, 7.2), **kw):
    """16:9 slide figure: title (the insight, as a sentence, wrapped to two lines if needed),
    subtitle (what is plotted), footer (source)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout='constrained', **kw)
    width = int(80 * figsize[0] / 12.8)       # characters per title line at 17pt bold
    lines = textwrap.wrap(title, width)
    sub = textwrap.wrap(subtitle, int(width * 1.7))
    h = figsize[1]
    line_t, line_s = 0.36 / h, 0.25 / h
    top = 0.985
    fig.text(0.012, top, '\n'.join(lines), fontsize=17, fontweight='bold', color=INK, va='top', linespacing=1.15)
    y_sub = top - line_t * len(lines) - 0.012
    fig.text(0.012, y_sub, '\n'.join(sub), fontsize=11.5, color=INK_2, va='top')
    plot_top = y_sub - line_s * len(sub) - 0.03
    fig.get_layout_engine().set(rect=(0, 0.045, 1, plot_top - 0.045), w_pad=0.08, h_pad=0.08)
    fig.text(0.012, 0.014, textwrap.shorten(footer, int(width * 2.5), placeholder=' ...'), fontsize=9, color=INK_MUTED)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_axisbelow(True)
    return fig, axes


def finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def bar_labels(ax, bars, texts, horizontal: bool = False, pad: float = 0.01, color: str = INK, size: float = 10.5) -> None:
    """Value label just past the end of each bar (never inside a bar it might not fit)."""
    lim = ax.get_xlim()[1] if horizontal else ax.get_ylim()[1]
    for b, t in zip(bars, texts):
        if horizontal:
            ax.text(b.get_width() + lim * pad, b.get_y() + b.get_height() / 2, t, va='center', ha='left', color=color, fontsize=size)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + lim * pad, t, ha='center', va='bottom', color=color, fontsize=size)


def pct_axis(ax, axis: str = 'y', decimals=None) -> None:
    """Percent ticks; decimals=None picks the fewest decimals that keep tick labels distinct."""
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(PercentFormatter(xmax=1, decimals=decimals))


def count_axis(ax, axis: str = 'y') -> None:
    fmt = FuncFormatter(lambda v, _: f'{v / 1e6:.1f}M' if abs(v) >= 1e6 else f'{v / 1e3:.0f}k' if abs(v) >= 1e3 else f'{v:.0f}')
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(fmt)


# %% [markdown]
# # 0. EDA — Business understanding, data quality and EDA
# Input: `data_model/*.parquet` and `data_clean.parquet`. Writes tables only; section 4 draws them.
#
# Five facts shape every later decision:
# 1. **Who is in the file.** All 1,000,528 customers applied for credit in Dec-2024/Jan-2025:
#    the dataset is the Tet-2025 applicant cohort, not the whole customer book.
# 2. **Data traps.** Contract rows are repeated (x1.46); the zero incomes are placeholders of the
#    copies; 85% of calls carry no outcome code.
# 3. **Right-censoring.** Missed-payment flags on instalments 2–4 cannot exist yet for recent loans,
#    so recent customers look healthier than they are unless flags are "seasoned".
# 4. **Risk differs by product and by borrower type** (first loan vs repeat), most of all in the Tet wave.
# 5. **Most contact is company-initiated.** Calls are 97% outbound; most leads are pushed lists.
# %%
import json

import numpy as np
import pandas as pd


con = connect()
audit = json.loads((MODEL / 'audit_summary.json').read_text(encoding='utf-8'))
eda: dict = {}

# %% [markdown]
# ## 0.1 Coverage: one row per customer, every source joined without loss
# %%
cov = con.sql("""SELECT count(*) AS customers,
        count(*) FILTER (WHERE CONTRACT_SIGNED_COUNT > 0) AS with_signed_contract,
        count(*) FILTER (WHERE CONTRACT_ACTIVE_COUNT > 0) AS with_active_contract,
        count(*) FILTER (WHERE LATEST_INCOME_MAIN IS NOT NULL) AS with_valid_income,
        count(*) FILTER (WHERE CALLS_PRE + CALLS_POST > 0) AS with_any_call,
        count(*) FILTER (WHERE LEAD_COUNT > 0) AS with_lead,
        count(*) FILTER (WHERE BEHAVIOR_EVENT_COUNT > 0) AS with_behaviour,
        count(*) FILTER (WHERE APP_SESSIONS_PER_WEEK_2025 > 0) AS app_users_2025,
        count(*) FILTER (WHERE CONTRACT_WRITTEN_OFF_COUNT > 0) AS with_write_off
    FROM customer_360""").df().T.rename(columns={0: 'customers'})
cov['share'] = cov.customers / N_CUSTOMERS
save_table(cov.reset_index(names='measure'), 'eda_coverage')
eda['coverage'] = cov.customers.to_dict()

# %% [markdown]
# ## 0.2 Who is in the file: the Tet-2025 applicant cohort
# %%
monthly = con.sql("""SELECT date_trunc('month', t_apply) AS month, count(*) AS applications,
        count(DISTINCT SKP_CLIENT) AS applicants, avg(is_signed::INT) AS signed_rate
    FROM k WHERE t_apply >= TIMESTAMP '2023-01-01' GROUP BY 1 ORDER BY 1""").df()
save_table(monthly, 'eda_monthly_applications')
cohort = con.sql("""WITH f AS (SELECT SKP_CLIENT, min(t_apply) AS first_app,
        max((t_apply >= TIMESTAMP '2024-12-01' AND t_apply < TIMESTAMP '2025-02-01')::INT) AS tet_applicant
        FROM k GROUP BY 1)
    SELECT count(*) AS customers, sum(tet_applicant) AS applied_dec24_jan25,
           sum((first_app >= TIMESTAMP '2024-12-01')::INT) AS first_ever_application_in_wave FROM f""").df().iloc[0]
baseline = monthly[(monthly.month >= '2024-06-01') & (monthly.month < '2024-12-01')].applications.mean()
peak = monthly[monthly.month.isin(pd.to_datetime(['2024-12-01', '2025-01-01']))].applications.mean()
eda['cohort'] = {'customers': int(cohort.customers), 'applied_in_tet_wave': int(cohort.applied_dec24_jan25),
                 'first_time_applicants_in_wave': int(cohort.first_ever_application_in_wave),
                 'first_time_share': round(cohort.first_ever_application_in_wave / cohort.customers, 4),
                 'monthly_applications_jun_nov_2024': round(baseline), 'monthly_applications_dec_jan': round(peak),
                 'peak_multiple': round(peak / baseline, 1)}

# %% [markdown]
# ## 0.3 Data traps found and fixed in Phase 1
# Each line: what was wrong, how big, what the fix was, and what would have gone wrong without it.
# %%
c = audit['contract']
calls_total = sum(audit['call_outcome_groups'].values())
traps = pd.DataFrame([
    {'trap': 'Contract rows repeated (2x or 8x per credit case)', 'affected': c['source_rows'] - c['credit_cases'],
     'base': c['source_rows'], 'fix': 'one row per SKP_CREDIT_CASE',
     'impact_if_ignored': f"every count and amount inflated x{c['row_inflation']}"},
    {'trap': 'Income 0 on the repeated copies (placeholder, not zero income)', 'affected': c['cases_with_placeholder_zero_income'],
     'base': c['credit_cases'], 'fix': 'keep the positive income of the case; 1M-200M VND/month valid',
     'impact_if_ignored': '306k customers wrongly treated as "income unknown"'},
    {'trap': 'Calls with no outcome code (dialled, not reached)', 'affected': audit['call_outcome_groups']['no_contact'],
     'base': calls_total, 'fix': 'outcome taxonomy: no contact / wrong party / sales / collection / service',
     'impact_if_ignored': 'dialling counted as customer engagement'},
    {'trap': 'Lead accepted date = 3000-01-01 (documented SOB sentinel)', 'affected': audit['lead_accepted_sentinel_3000'],
     'base': audit['source_rows']['LEADS'], 'fix': 'null + flag', 'impact_if_ignored': 'acceptance delays of ~975 years'},
    {'trap': 'Signed contracts without sign time (pre-2017)', 'affected': c['signed_cases_without_sign_time'],
     'base': c['credit_cases'], 'fix': 'fall back to approval / application time',
     'impact_if_ignored': 'old history dropped from tenure and repayment'},
    {'trap': 'Exact duplicate lead rows', 'affected': audit['lead_exact_duplicate_extra_rows'],
     'base': audit['source_rows']['LEADS'], 'fix': 'dropped', 'impact_if_ignored': 'lead activity overstated'},
    {'trap': 'Income outside 1M-200M VND/month (e.g. 9,000,000,000)', 'affected': c['cases_with_income_above_200M'],
     'base': c['credit_cases'], 'fix': 'set to missing', 'impact_if_ignored': 'instalment-to-income near 0'},
])
traps['share'] = traps.affected / traps.base
save_table(traps, 'eda_data_traps')

# %% [markdown]
# ## 0.4 Right-censoring of missed-payment flags
# Flag k can only be 1 after instalment k has fallen due. For loans signed in 2025 the later flags are
# structurally zero by the extraction date: these loans look clean because they are young.
# %%
censor = con.sql("""SELECT date_trunc('month', t_ref) AS sign_month, count(*) AS contracts,
        avg(d1) AS first_due, avg(d2) AS second_due, avg(d3) AS third_due, avg(d4) AS fourth_due
    FROM k WHERE is_signed AND product IN ('CD','TW','CL','CW') AND t_ref >= TIMESTAMP '2023-07-01'
    GROUP BY 1 ORDER BY 1""").df()
save_table(censor, 'eda_right_censoring')

# %% [markdown]
# ## 0.5 Early stress by product and borrower type, Tet wave vs the months before
# Early stress = missed payment on instalments 1–4 or written off. Instalment loans signed Jul-2024 to
# Jan-2025 have all four instalments observable by Jun-2025; revolving products (BNPL, CC) carry the same flags.
# %%
risk = con.sql("""WITH s AS (
        SELECT *, row_number() OVER (PARTITION BY SKP_CLIENT ORDER BY t_ref, SKP_CREDIT_CASE) AS loan_no
        FROM k WHERE is_signed)
    SELECT CASE WHEN t_ref >= TIMESTAMP '2024-12-01' THEN 'Tet wave (Dec-24 to Jan-25)' ELSE 'Before (Jul-24 to Nov-24)' END AS period,
           product, CASE WHEN loan_no = 1 THEN 'First loan' ELSE 'Repeat loan' END AS borrower,
           count(*) AS contracts, avg(((d1 + d2 + d3 + d4) > 0 OR wo)::INT) AS early_stress_rate
    FROM s WHERE t_ref >= TIMESTAMP '2024-07-01' AND t_ref < TIMESTAMP '2025-02-01' AND product IS NOT NULL
    GROUP BY ALL ORDER BY 1, 2, 3""").df()
save_table(risk, 'eda_early_stress_by_product')
tet = risk[risk.period.str.startswith('Tet')]
eda['tet_wave_risk'] = {
    'first_loan_stress': round(float(np.average(tet[tet.borrower == 'First loan'].early_stress_rate, weights=tet[tet.borrower == 'First loan'].contracts)), 4),
    'repeat_loan_stress': round(float(np.average(tet[tet.borrower == 'Repeat loan'].early_stress_rate, weights=tet[tet.borrower == 'Repeat loan'].contracts)), 4)}
signed_mix = con.sql("""SELECT product, count(*) AS signed_contracts FROM k
    WHERE is_signed AND t_ref >= TIMESTAMP '2024-12-01' AND t_ref < TIMESTAMP '2025-02-01' GROUP BY 1 ORDER BY 2 DESC""").df()
save_table(signed_mix, 'eda_tet_product_mix')
footprint = con.sql("""WITH first_loan AS (SELECT SKP_CLIENT, min(t_ref) AS t FROM k WHERE is_signed GROUP BY 1)
    SELECT count(*) AS first_loans_in_wave,
           count(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM leads l WHERE l.SKP_CLIENT = f.SKP_CLIENT AND l.CREATED_AT < f.t - INTERVAL 1 DAY)
                            AND NOT EXISTS (SELECT 1 FROM behavior b WHERE b.SKP_CLIENT = f.SKP_CLIENT AND b.EVENT_AT < f.t - INTERVAL 1 DAY)
                            AND NOT EXISTS (SELECT 1 FROM k k2 WHERE k2.SKP_CLIENT = f.SKP_CLIENT AND k2.t_apply < f.t - INTERVAL 1 DAY)) AS no_prior_footprint
    FROM first_loan f WHERE t >= TIMESTAMP '2024-12-01' AND t < TIMESTAMP '2025-02-01'""").df().iloc[0]
eda['first_loans_in_wave'] = {'count': int(footprint.first_loans_in_wave), 'no_prior_footprint': int(footprint.no_prior_footprint),
                              'no_prior_footprint_share': round(footprint.no_prior_footprint / footprint.first_loans_in_wave, 4)}

# %% [markdown]
# ## 0.6 Who starts the contact: company vs customer
# %%
initiator = con.sql("""
    SELECT 'Calls' AS source, CASE WHEN CALL_TYPE = 'Inbound' THEN 'Customer' ELSE 'Company' END AS initiated_by, count(*) AS records FROM calls GROUP BY 1, 2
    UNION ALL SELECT 'Leads', CASE WHEN SELF_INITIATED THEN 'Customer' ELSE 'Company' END, count(*) FROM leads GROUP BY 1, 2
    UNION ALL SELECT 'Scoring events', CASE WHEN SELF_INITIATED THEN 'Customer' ELSE 'Company' END, count(*) FROM behavior GROUP BY 1, 2
    ORDER BY 1, 2""").df()
initiator['share'] = initiator.records / initiator.groupby('source').records.transform('sum')
save_table(initiator, 'eda_contact_initiator')
outcomes = pd.Series(audit['call_outcome_groups']).rename('calls').to_frame()
outcomes['share'] = outcomes.calls / outcomes.calls.sum()
save_table(outcomes.sort_values('calls', ascending=False).reset_index(names='outcome_group'), 'eda_call_outcomes')
eda['calls'] = {'outbound_share': round(float(initiator.query("source == 'Calls' and initiated_by == 'Company'").share.iloc[0]), 4),
                'no_contact_share': round(audit['call_outcome_groups']['no_contact'] / calls_total, 4)}

save_json(eda, 'eda_summary')
print(json.dumps(eda, indent=1))
print(traps[['trap', 'affected', 'share']].to_string(index=False))
print(risk.round(4).to_string(index=False))


# %% [markdown]
# # 1. CHALLENGE 1 — Customer Financial Health Score (FHS)
# Input: `data_model/*.parquet`. Runs after section 0 (order does not matter for the numbers).
#
# **What the score means.** FHS 0–100 ranks how likely a customer is to fall into repayment stress
# in the next months. It is a points scorecard: every customer's score is a base plus readable
# points per factor, and every factor's direction is checked against economics and against time.
#
# **How it is built (no future information).**
# 1. Features are computed *as of* a snapshot date D from contracts signed before D. A missed
#    instalment k is only visible once it has fallen due (k+1 months after signing); a write-off
#    only after 9 months; a loan only counts as completed once its maturity date has passed.
# 2. Stress label after D: a new loan misses instalments 1–4 or is written off, or an existing
#    loan misses an instalment that falls due after D, or a young loan is written off.
# 3. Train on D = 2023-07-01 and 2024-01-01 (6-month windows); test out of time on D = 2024-07-01
#    (window to Jan-2025, includes the Tet 2025 lending wave).
# 4. WoE + logistic regression, points-to-double-odds scale; two scorecards on one scale
#    (customers with credit history / thin file). Score all customers at T = 2025-06-05.
# 5. Independent check on information never used in training: calls after T (Jun-2025..Aug-2026).
#
# Demographics are excluded from the score and used only in a fairness check.
# %%
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


con = connect()
TRAIN_SNAPSHOTS = [('2023-07-01', '2024-01-01'), ('2024-01-01', '2024-07-01')]
OOT_SNAPSHOT = ('2024-07-01', '2025-02-01')   # Jan-2025 contracts still have 4 instalments observed by Jun-2025

HIST_FEATURES = ['repay_state', 'max_clean_credit', 'tenure', 'recency', 'down_payment', 'income', 'pti', 'apps_90d']
# income is not used for thin files: "unknown income" there just means "never applied before",
# which unsigned_apps already measures
THIN_FEATURES = ['unsigned_apps', 'apps_90d', 'scoring_90d']
# tested for the history model and excluded on economic / stability grounds (evidence in tables/fhs_information_value.csv)
EXCLUDED = {
    'n_active': 'looks protective only because the label sees instalments 1-4 and the lender tops up trusted '
                'clients; collection calls after T rise with the number of live loans (selection effect)',
    'active_cash': 'cash-loan holders passed stricter underwriting; IV unstable over time',
    'scoring_90d': 'direction flips between train (protective) and out-of-time (risky)',
    'rejected_12m': "the lender's own past decision (circular); weak and unstable",
    'due_recency': 'duplicates repay_state (cured vs not cured) with a weaker, less stable signal',
}
PILLAR = {'repay_state': 'Repayment', 'due_recency': 'Repayment', 'max_clean_credit': 'Proven capacity',
          'tenure': 'Stability', 'recency': 'Stability', 'down_payment': 'Affordability', 'income': 'Affordability',
          'pti': 'Affordability', 'n_active': 'Affordability', 'active_cash': 'Affordability',
          'apps_90d': 'Credit seeking', 'scoring_90d': 'Credit seeking', 'unsigned_apps': 'Credit seeking',
          'rejected_12m': 'Credit seeking'}

# %% [markdown]
# ## 1.1 Point-in-time features and stress label
# %%
def feature_sql(D: str, prod: bool) -> str:
    """Customer features using only what is observable before date D (prod=True: scoring date T)."""
    P = 'TRUE' if prod else 'FALSE'
    Dt = f"TIMESTAMP '{D}'"
    return f"""
    WITH s AS (
        SELECT *,
          -- instalment k falls due ~k months after signing; its flag is treated as known one month later
          (CASE WHEN t_ref + INTERVAL 2 MONTH <= {Dt} THEN d1 ELSE 0 END
         + CASE WHEN t_ref + INTERVAL 3 MONTH <= {Dt} THEN d2 ELSE 0 END
         + CASE WHEN t_ref + INTERVAL 4 MONTH <= {Dt} THEN d3 ELSE 0 END
         + CASE WHEN t_ref + INTERVAL 5 MONTH <= {Dt} THEN d4 ELSE 0 END) AS due_obs,
          wo AND t_ref + INTERVAL 9 MONTH <= {Dt} AS wo_obs,
          status IN ('Finished','Paid off') AND ({P} OR maturity <= {Dt}) AS fin_obs
        FROM k WHERE is_signed AND t_ref < {Dt}),
    s2 AS (
        SELECT *,
          NOT wo_obs AND CASE WHEN {P} THEN status IN ('Active','Signed')
                              ELSE (maturity > {Dt} OR (maturity IS NULL AND product IN {REVOLVING})) END AS active
        FROM s),
    c AS (
        SELECT SKP_CLIENT,
          count(*) AS n_signed, min(t_ref) AS first_ref, max(t_ref) AS last_ref,
          sum((fin_obs AND due_obs = 0)::INT) AS n_fin_clean,
          max(CASE WHEN fin_obs AND due_obs = 0 THEN credit END) AS max_clean_credit_v,
          sum((due_obs > 0)::INT) AS n_due,
          max(CASE WHEN due_obs > 0 THEN t_ref END) AS last_due_ref,
          max(CASE WHEN fin_obs AND due_obs = 0 THEN t_ref END) AS last_clean_fin_ref,
          max((due_obs > 0 AND fin_obs)::INT) AS due_repaid,
          max(wo_obs::INT) AS wo_obs,
          sum(CASE WHEN active AND product IN {INSTALLMENT} THEN coalesce(annuity, 0) ELSE 0 END) AS act_annuity,
          sum(active::INT) AS n_active_v,
          max((active AND product IN ('CL','CW'))::INT) AS act_cash_v,
          round(avg(CASE WHEN product IN ('CD','TW') AND credit > 0
                   THEN coalesce(init_pay, 0) / (credit + coalesce(init_pay, 0)) END), 6) AS dp_ratio
        FROM s2 GROUP BY 1),
    u AS (SELECT SKP_CLIENT, count(*) AS n_unsigned,
                 count(*) FILTER (WHERE status = 'Rejected' AND t_apply >= {Dt} - INTERVAL 12 MONTH) AS n_rej_12m
          FROM k WHERE NOT is_signed AND t_apply < {Dt} GROUP BY 1),
    inc AS (SELECT SKP_CLIENT, arg_max(inc, (t_apply, SKP_CREDIT_CASE)) AS inc_last
            FROM k WHERE t_apply < {Dt} AND inc IS NOT NULL GROUP BY 1),
    apps AS (SELECT SKP_CLIENT, count(*) AS apps_90d_v
             FROM k WHERE t_apply >= {Dt} - INTERVAL 90 DAY AND t_apply < {Dt} GROUP BY 1),
    sc AS (SELECT SKP_CLIENT, count(*) AS scoring_90d_v
           FROM behavior WHERE EVENT_AT >= {Dt} - INTERVAL 90 DAY AND EVENT_AT < {Dt} GROUP BY 1)
    SELECT cu.SKP_CLIENT,
      c.n_signed IS NOT NULL AS has_history,
      coalesce(c.n_signed, 0) AS n_signed, coalesce(c.n_fin_clean, 0) AS n_fin_clean,
      coalesce(c.n_due, 0) AS n_due, coalesce(c.wo_obs, 0) AS wo_obs, coalesce(c.n_active_v, 0) AS n_active_v,
      c.act_annuity, i.inc_last, c.dp_ratio, c.max_clean_credit_v,
      CASE WHEN c.act_annuity > 0 AND i.inc_last > 0 THEN c.act_annuity / i.inc_last END AS pti_val,
      date_diff('month', c.first_ref, {Dt}) AS tenure_m,
      date_diff('month', c.last_ref, {Dt}) AS recency_m,
      coalesce(a.apps_90d_v, 0) AS apps_90d_v, coalesce(sc.scoring_90d_v, 0) AS scoring_90d_v,
      coalesce(u.n_unsigned, 0) AS n_unsigned,
      CASE
        WHEN c.n_signed IS NULL THEN '0 No signed contract'
        WHEN c.wo_obs = 1 THEN '1 Written off'
        WHEN c.n_due > 0 AND NOT (c.due_repaid = 1 OR coalesce(c.last_clean_fin_ref > c.last_due_ref, FALSE))
             THEN '2 Past due, not cured'
        WHEN c.n_due > 0 THEN '3 Past due, cured'
        WHEN c.n_fin_clean = 0 THEN '4 Clean, no loan completed yet'
        WHEN c.n_fin_clean <= 2 THEN '5 Clean, 1-2 loans completed'
        ELSE '6 Clean, 3+ loans completed' END AS repay_state,
      CASE
        WHEN c.last_due_ref IS NULL THEN '0 Never'
        WHEN date_diff('month', c.last_due_ref, {Dt}) < 12 THEN '1 <1y'
        WHEN date_diff('month', c.last_due_ref, {Dt}) < 36 THEN '2 1-3y'
        ELSE '3 3y+' END AS due_recency,
      CASE
        WHEN c.max_clean_credit_v IS NULL THEN '0 None completed cleanly'
        WHEN c.max_clean_credit_v < 5e6 THEN '1 <5M'
        WHEN c.max_clean_credit_v < 15e6 THEN '2 5-15M'
        ELSE '3 >=15M' END AS max_clean_credit,
      CASE
        WHEN coalesce(c.act_annuity, 0) = 0 OR i.inc_last IS NULL THEN '0 No active instalment'
        WHEN c.act_annuity / i.inc_last < 0.20 THEN '1 <20%'
        WHEN c.act_annuity / i.inc_last < 0.35 THEN '2 20-35%'
        WHEN c.act_annuity / i.inc_last < 0.50 THEN '3 35-50%'
        ELSE '4 >=50%' END AS pti,
      CASE WHEN coalesce(c.act_cash_v, 0) = 1 THEN '1 Yes' ELSE '0 No' END AS active_cash,
      CASE WHEN coalesce(c.n_active_v, 0) = 0 THEN '0' WHEN c.n_active_v = 1 THEN '1'
           WHEN c.n_active_v = 2 THEN '2' ELSE '3+' END AS n_active,
      CASE
        WHEN c.dp_ratio IS NULL THEN '0 No goods/bike loan'
        WHEN c.dp_ratio < 0.20 THEN '1 <20%'
        WHEN c.dp_ratio < 0.40 THEN '2 20-40%'
        ELSE '3 >=40%' END AS down_payment,
      CASE
        WHEN i.inc_last IS NULL THEN '0 Unknown'
        WHEN i.inc_last < 7e6 THEN '1 <7M'
        WHEN i.inc_last < 10e6 THEN '2 7-10M'
        WHEN i.inc_last < 15e6 THEN '3 10-15M'
        ELSE '4 >=15M' END AS income,
      CASE
        WHEN c.first_ref IS NULL THEN '0 None'
        WHEN date_diff('month', c.first_ref, {Dt}) < 12 THEN '1 <1y'
        WHEN date_diff('month', c.first_ref, {Dt}) < 36 THEN '2 1-3y'
        WHEN date_diff('month', c.first_ref, {Dt}) < 60 THEN '3 3-5y'
        ELSE '4 5y+' END AS tenure,
      CASE
        WHEN c.last_ref IS NULL THEN '0 None'
        WHEN date_diff('month', c.last_ref, {Dt}) < 3 THEN '1 <3m'
        WHEN date_diff('month', c.last_ref, {Dt}) < 12 THEN '2 3-12m'
        WHEN date_diff('month', c.last_ref, {Dt}) < 36 THEN '3 1-3y'
        ELSE '4 3y+' END AS recency,
      CASE WHEN coalesce(a.apps_90d_v, 0) = 0 THEN '0' WHEN a.apps_90d_v <= 2 THEN '1-2' ELSE '3+' END AS apps_90d,
      CASE WHEN coalesce(sc.scoring_90d_v, 0) = 0 THEN '0' ELSE '1+' END AS scoring_90d,
      CASE WHEN coalesce(u.n_rej_12m, 0) = 0 THEN '0' ELSE '1+' END AS rejected_12m,
      CASE WHEN coalesce(u.n_unsigned, 0) = 0 THEN '0' WHEN u.n_unsigned = 1 THEN '1'
           WHEN u.n_unsigned <= 3 THEN '2-3' ELSE '4+' END AS unsigned_apps
    FROM customers cu
    LEFT JOIN c USING (SKP_CLIENT) LEFT JOIN inc i USING (SKP_CLIENT)
    LEFT JOIN apps a USING (SKP_CLIENT) LEFT JOIN sc USING (SKP_CLIENT) LEFT JOIN u USING (SKP_CLIENT)
    ORDER BY cu.SKP_CLIENT
    """


def label_sql(D: str, D_end: str) -> str:
    """Stress observed AFTER D on new and existing exposures:
       (a) a contract signed in [D, D_end) misses instalments 1-4 or is written off;
       (b) a contract signed before D misses an instalment that falls due after D
           (exactly the flags hidden from the features by the seasoning rule);
       (c) a contract too young at D to be written off (signed < 9 months before D) is written off."""
    Dt, Et = f"TIMESTAMP '{D}'", f"TIMESTAMP '{D_end}'"
    return f"""
    SELECT SKP_CLIENT,
      coalesce(max(CASE WHEN t_ref >= {Dt} THEN ((d1 + d2 + d3 + d4) > 0 OR wo)::INT END), 0) AS new_bad,
      coalesce(max(CASE WHEN t_ref >= {Dt} THEN 1 END), 0) AS has_new,
      coalesce(max(CASE WHEN t_ref < {Dt} AND t_ref + INTERVAL 5 MONTH > {Dt} THEN
          ((t_ref + INTERVAL 2 MONTH > {Dt} AND d1 = 1) OR (t_ref + INTERVAL 3 MONTH > {Dt} AND d2 = 1)
        OR (t_ref + INTERVAL 4 MONTH > {Dt} AND d3 = 1) OR (t_ref + INTERVAL 5 MONTH > {Dt} AND d4 = 1))::INT END), 0) AS exist_due,
      coalesce(max(CASE WHEN t_ref < {Dt} AND t_ref + INTERVAL 9 MONTH > {Dt} AND wo THEN 1 END), 0) AS exist_wo,
      arg_min(product, (t_ref, SKP_CREDIT_CASE)) FILTER (WHERE t_ref >= {Dt}) AS new_product,
      coalesce(max((t_ref >= GREATEST({Dt}, TIMESTAMP '2024-12-01') AND t_ref < TIMESTAMP '2025-02-01')::INT), 0) AS tet_wave
    FROM k
    WHERE is_signed AND t_ref < {Et}
    GROUP BY 1
    """


def snapshot(D: str, D_end: str | None, prod: bool = False) -> pd.DataFrame:
    df = con.sql(feature_sql(D, prod)).df()
    assert df['SKP_CLIENT'].is_unique and len(df) == N_CUSTOMERS, 'feature table must be 1 row per customer'
    if D_end:
        lab = con.sql(label_sql(D, D_end)).df()
        df = df.merge(lab, on='SKP_CLIENT', how='left')
        for c in ['new_bad', 'has_new', 'exist_due', 'exist_wo', 'tet_wave']:
            df[c] = df[c].fillna(0).astype(int)
        # population = customers with a live exposure at D, or who take new credit in the window
        df = df[(df.has_new == 1) | (df.n_active != '0')].copy()
        df['bad'] = ((df.new_bad + df.exist_due + df.exist_wo) > 0).astype(int)
        df['snapshot'] = D
    return df


train = pd.concat([snapshot(d, e) for d, e in TRAIN_SNAPSHOTS], ignore_index=True)
oot = snapshot(*OOT_SNAPSHOT)
label_mix = pd.DataFrame({name: {'customers': len(df), 'stress_rate': df.bad.mean(),
    'thin_file_share': (~df.has_history).mean(), 'stress_on_new_loans': df.new_bad.mean(),
    'stress_on_existing_instalments': df.exist_due.mean(), 'write_off_on_young_loans': df.exist_wo.mean()}
    for name, df in [('train', train), ('out_of_time', oot)]})
save_table(label_mix.reset_index(names='metric'), 'fhs_label_summary')
log('label summary:\n' + label_mix.round(4).to_string())

# %% [markdown]
# ## 1.2 Feature evidence
# Information Value (IV) per candidate, in train and out of time. A feature must keep its direction
# over time and make economic sense. Excluded features stay in the evidence table with the reason.
# %%
cand = HIST_FEATURES + list(EXCLUDED)
uni = []
for samp_name, df in [('train', train), ('oot', oot)]:
    for pop, sub in [('history', df[df.has_history]), ('thin', df[~df.has_history])]:
        for f in (cand if pop == 'history' else THIN_FEATURES):
            uni.append(woe_table(sub, f).assign(sample=samp_name, population=pop))
uni = pd.concat(uni, ignore_index=True)
iv = uni.groupby(['population', 'feature', 'sample'])['iv_part'].sum().unstack('sample')
iv['status'] = [('EXCLUDED: ' + EXCLUDED[f]) if (pop == 'history' and f in EXCLUDED) else 'candidate'
                for pop, f in iv.index]
iv['pillar'] = [PILLAR[f] for _, f in iv.index]
save_table(uni, 'fhs_univariate_woe')
save_table(iv.reset_index(), 'fhs_information_value')
log('Information Value:\n' + iv[['train', 'oot', 'pillar']].round(4).to_string())

# %% [markdown]
# ## 1.3 Scorecards
# History scorecard for customers with a signed contract before D; thin-file scorecard otherwise.
# Points: 2% stress probability = 60 points; every doubling of the good:bad odds = +10 points.
# %%
hist = Scorecard('history', HIST_FEATURES, PILLAR, log).fit(train[train.has_history])
thin = Scorecard('thin_file', THIN_FEATURES, PILLAR, log).fit(train[~train.has_history])
points_table = pd.concat([hist.pts, thin.pts], ignore_index=True)
save_table(points_table, 'fhs_scorecard_points')
weights = pd.concat([hist.weights().rename('history_model_weight_pct'),
                     thin.weights().rename('thin_model_weight_pct')], axis=1).round(1)
weights['pillar'] = weights.index.map(PILLAR)
save_table(weights.reset_index(names='feature'), 'fhs_feature_weights')
log(f'base points: history={hist.base_points:.1f} thin={thin.base_points:.1f}')
log('feature weights (% of points range):\n' + weights.to_string())


def score_all(df: pd.DataFrame, clip: bool = True) -> pd.Series:
    s = pd.Series(np.nan, index=df.index)
    h = df['has_history'].astype(bool)
    s[h] = hist.score(df[h], clip)
    s[~h] = thin.score(df[~h], clip)
    return s

# %% [markdown]
# ## 1.4 Validation
# AUC / Gini / KS in time and out of time; decile table; benchmarks on the same OOT customers
# (repayment history alone, gradient boosting on the same inputs); subgroups; weight perturbation.
# %%
train['score'], oot['score'] = score_all(train), score_all(oot)
m = {'scale': {'pdo': PDO, 'anchor': f'{ANCHOR_BAD:.0%} stress probability = {ANCHOR_SCORE:.0f} points'},
     'train_in_time': metrics(train.bad, train.score), 'oot': metrics(oot.bad, oot.score),
     'oot_history_only': metrics(oot.bad[oot.has_history], oot.score[oot.has_history]),
     'oot_thin_only': metrics(oot.bad[~oot.has_history], oot.score[~oot.has_history]),
     'dropped_features': {'history': hist.dropped, 'thin_file': thin.dropped, 'excluded_by_design': EXCLUDED}}
save_table(decile_table(oot.bad, oot.score), 'fhs_validation_oot_deciles')
save_table(decile_table(train.bad, train.score), 'fhs_validation_train_deciles')

tr_h, oh = train[train.has_history], oot[oot.has_history]
rs_woe = dict(zip(hist.woes['repay_state']['bin'], hist.woes['repay_state']['woe']))
m['benchmark_repay_state_only'] = metrics(oh.bad, oh.repay_state.map(rs_woe).fillna(0))
Xd, Xo = tr_h[hist.active].astype('category'), oh[hist.active].astype('category')
for c in hist.active:
    Xo[c] = Xo[c].cat.set_categories(Xd[c].cat.categories)
gbm = HistGradientBoostingClassifier(categorical_features='from_dtype', max_iter=300, learning_rate=0.05,
                                     random_state=SEED).fit(Xd, tr_h.bad)
m['benchmark_gbm_same_features_history'] = metrics(oh.bad, -gbm.predict_proba(Xo)[:, 1])

sub = {'tet_wave_contracts': metrics(oot.bad[oot.tet_wave == 1], oot.score[oot.tet_wave == 1]),
       'non_tet_contracts': metrics(oot.bad[oot.tet_wave == 0], oot.score[oot.tet_wave == 0])}
for prod_name, g in oot.groupby('new_product'):
    if len(g) > 3000 and g.bad.nunique() == 2:
        sub[f'new_product_{prod_name}'] = metrics(g.bad, g.score)
m['oot_subgroups'] = sub

rng = np.random.default_rng(SEED)
samp = oh.sample(min(150_000, len(oh)), random_state=SEED)
base_pts = hist.points(samp)[hist.active]
aucs, rhos = [], []
for _ in range(200):   # every feature's points x random factor in [0.75, 1.25]
    s2 = (base_pts * pd.Series({f: 1 + rng.uniform(-.25, .25) for f in hist.active})).sum(axis=1)
    aucs.append(roc_auc_score(samp.bad, -s2))
    rhos.append(spearmanr(base_pts.sum(axis=1), s2).statistic)
m['weight_perturbation_25pct_history'] = {'auc_min': round(min(aucs), 4), 'auc_max': round(max(aucs), 4),
    'rank_corr_min': round(float(min(rhos)), 4), 'rank_corr_median': round(float(np.median(rhos)), 4)}
log('validation: ' + json.dumps({k: m[k] for k in ['train_in_time', 'oot', 'oot_history_only', 'oot_thin_only',
    'benchmark_repay_state_only', 'benchmark_gbm_same_features_history']}, indent=1))

# %% [markdown]
# ## 1.5 Structural score at T = 2025-06-05
# The scorecard above is the *structural* part of FHS: what the customer's credit history says.
# %%
prod = snapshot(CUTOFF, None, prod=True)
prod['structural'] = score_all(prod)
prod['affordability'] = (prod.n_active.eq('3+') | (prod.pti_val >= 0.5)).astype(int)

# %% [markdown]
# ## 1.6 Live status layer, validated on calls after T
# The contract table only records missed payments on instalments 1–4. Arrears later in a loan's
# life are invisible there; the collection call log is the only window on them. Two facts measured
# in the call log:
#
# * **Selection paradox.** In the early-delinquency label, customers with more live loans look
#   *safer* (the lender only tops up clients it trusts). After T the same customers receive
#   collection calls far more often. A score trained on early delinquency alone rewards leverage.
# * **Collection contact persists.** A customer reached by collection in the months before a date
#   is ~10x more likely to be in collection in the months after it.
#
# So FHS = structural points + live points. Live points come from a logistic regression on a
# *fit* window (collection contact Jan–Jun 2025 -> collection Jun–Dec 2025), controlling for how often
# the customer was dialled, converted to the same points scale (points = 10/ln2 x log-odds). They
# are then checked on a later, untouched *validation* window (Jul–Dec 2025 -> Jan–Aug 2026).
#
# Traps handled: the call list is chosen by the company (written-off customers are almost absent);
# a customer dialled 40 times has more chances to get a collection code than one dialled once, so
# comparisons are made inside groups with the same number of calls in the outcome window.
# %%
LIVE_FIT = ('2025-01-01', CUTOFF, CUTOFF, '2026-01-01')
LIVE_VALID = ('2025-07-01', '2026-01-01', '2026-01-01', CALLS_END)
EW_BINS, EW_LABELS = [-1, 0, 1, 3, 10_000], ['0', '1', '2-3', '4+']
VOLUME_BINS, VOLUME_LABELS = [0, 1, 3, 7, 15, 30, 10_000], ['1', '2-3', '4-7', '8-15', '16-30', '31+']


def call_window(fs: str, fe: str, os_: str, oe: str) -> pd.DataFrame:
    """Per customer: collection contacts in the feature window, calls and collection contacts in the outcome window."""
    return con.sql(f"""
        SELECT SKP_CLIENT,
               count(*) FILTER (WHERE START_AT >= '{fs}' AND START_AT < '{fe}' AND OUTCOME_GROUP = 'collection') AS collection_before,
               count(*) FILTER (WHERE START_AT >= '{os_}' AND START_AT < '{oe}') AS calls_after,
               count(*) FILTER (WHERE START_AT >= '{os_}' AND START_AT < '{oe}' AND OUTCOME_GROUP = 'collection') AS collection_after
        FROM calls_rpc GROUP BY 1""").df()


def live_frame(window: tuple) -> pd.DataFrame:
    d = prod[['SKP_CLIENT', 'structural', 'affordability', 'n_active', 'has_history']].merge(call_window(*window), on='SKP_CLIENT')
    d = d[d.calls_after > 0].copy()
    d['y'] = (d.collection_after > 0).astype(int)
    d['collection_share'] = d.collection_after / d.calls_after
    d['ew'] = pd.cut(d.collection_before, EW_BINS, labels=EW_LABELS).astype(str)
    d['call_volume'] = pd.cut(d.calls_after, VOLUME_BINS, labels=VOLUME_LABELS).astype(str)
    return d


fit_df, valid_df = live_frame(LIVE_FIT), live_frame(LIVE_VALID)
X = pd.get_dummies(fit_df[['ew', 'call_volume']]).drop(columns=['ew_0', 'call_volume_1']).astype(float)
X['affordability'] = fit_df.affordability
live_lr = LogisticRegression(C=1e4, max_iter=5000).fit(X, fit_df.y)
live_coef = dict(zip(X.columns, live_lr.coef_[0]))
LIVE_POINTS = {'ew': {'0': 0.0, **{b: round(-FACTOR * live_coef[f'ew_{b}'], 1) for b in EW_LABELS[1:]}},
               'affordability': round(-FACTOR * live_coef['affordability'], 1)}
assert all(v <= 0 for v in LIVE_POINTS['ew'].values()) and LIVE_POINTS['affordability'] <= 0, 'live signals must lower FHS'
log(f'live points (from fit window): {LIVE_POINTS}')


def live_points(ew: pd.Series, affordability: pd.Series) -> pd.Series:
    return ew.map(LIVE_POINTS['ew']).astype(float) + LIVE_POINTS['affordability'] * affordability


BAND_CUTS = [-0.1, 40, 55, 65, 75, 100.1]
BAND_LABELS = ['E Stressed', 'D Fragile', 'C Stable', 'B Healthy', 'A Thriving']
to_band = lambda s: pd.cut(s, BAND_CUTS, labels=BAND_LABELS).astype(str)
live_check = {'live_points': LIVE_POINTS, 'live_logit_coefficients': {k: round(v, 4) for k, v in live_coef.items()}}
for name, d in [('fit_window', fit_df), ('validation_window', valid_df)]:
    d['final'] = (d.structural + live_points(d.ew, d.affordability)).clip(0, 100)
    d['band'] = to_band(d.final)
    live_check[name] = {
        'customers': len(d), 'collection_rate': round(d.y.mean(), 4),
        'structural_only': metrics(d.y, d.structural)['auc'],
        'final_fhs': metrics(d.y, d.final)['auc'],
        'structural_only_volume_adjusted': stratified_auc(d, 'y', 'structural', 'call_volume')['weighted_auc'],
        'final_fhs_volume_adjusted': stratified_auc(d, 'y', 'final', 'call_volume')['weighted_auc'],
        'collection_rate_by_ew': d.groupby('ew').y.mean().round(4).to_dict(),
        'collection_rate_by_affordability': d.groupby('affordability').y.mean().round(4).to_dict()}
m['live_layer'] = live_check
by_band = valid_df.groupby('band').agg(customers=('SKP_CLIENT', 'size'), any_collection=('y', 'mean'),
    collection_share_of_calls=('collection_share', 'mean'), calls_per_customer=('calls_after', 'mean'))
save_table(by_band.reset_index(), 'fhs_future_collection_by_band')
ew_tab = pd.concat([d.groupby('ew').agg(customers=('y', 'size'), collection_rate=('y', 'mean')).assign(window=n)
                    for n, d in [('fit: Jan-Jun 2025 -> Jun-Dec 2025', fit_df), ('validation: Jul-Dec 2025 -> Jan-Aug 2026', valid_df)]])
save_table(ew_tab.reset_index(), 'fhs_early_warning_validation')
log('live layer:\n' + json.dumps(live_check, indent=1, default=str))
log('validation window, collection by final band:\n' + by_band.round(4).to_string())

# the lender-selection paradox: early-delinquency label vs later collection, by number of live loans
post = live_frame((CUTOFF, CUTOFF, CUTOFF, CALLS_END))
paradox = pd.concat([
    oot[oot.has_history].groupby('n_active').bad.mean().rename('stress_rate_label_oot'),
    post[post.has_history].groupby('n_active').y.mean().rename('any_collection_after_T'),
    post[post.has_history].groupby('n_active').collection_share.mean().rename('collection_share_after_T'),
    post[post.has_history].groupby('n_active').size().rename('called_after_T'),
    prod[prod.has_history].n_active.value_counts().rename('customers_at_T')], axis=1)
save_table(paradox.reset_index(names='n_active'), 'fhs_selection_paradox')
log('selection paradox:\n' + paradox.round(4).to_string())

# cost of the affordability points on the early-delinquency label (2024), where the paradox hides leverage
oot_aff = (oot.n_active.eq('3+') | (oot.pti_val >= 0.5)).astype(int)
m['affordability_tradeoff_on_oot_label'] = {
    'structural': metrics(oot.bad, oot.score)['auc'],
    'structural_plus_affordability_points': metrics(oot.bad, (oot.score + LIVE_POINTS['affordability'] * oot_aff).clip(0, 100))['auc']}

# %% [markdown]
# ## 1.7 Final FHS, bands and stability
# Band cut points sit where the out-of-time stress rate steps up. The expected stress rate of a band
# comes from the 2024 out-of-time sample (structural part; the call log does not exist before 2025).
# %%
ew_at_T = call_window('2025-01-01', CUTOFF, CUTOFF, CUTOFF).set_index('SKP_CLIENT').collection_before
prod['collection_before_T'] = prod.SKP_CLIENT.map(ew_at_T).fillna(0).astype(int)
prod['ew'] = pd.cut(prod.collection_before_T, EW_BINS, labels=EW_LABELS).astype(str)
prod['live_points'] = live_points(prod.ew, prod.affordability)
prod['score'] = (prod.structural + prod.live_points).clip(0, 100).round(1)
oot['band'], prod['band'] = to_band(oot.score), to_band(prod.score)
bt = oot.groupby('band').agg(n_oot=('bad', 'size'), stress_rate_oot=('bad', 'mean'),
                             score_min=('score', 'min'), score_max=('score', 'max'))
bt['share_oot'] = bt.n_oot / bt.n_oot.sum()
bt['n_production'] = prod.band.value_counts()
bt['share_production'] = bt.n_production / len(prod)
bt['stress_lift_vs_A'] = bt.stress_rate_oot / bt.loc['A Thriving', 'stress_rate_oot']
bt['future_collection_rate_validation_window'] = by_band.any_collection
save_table(bt.reset_index(), 'fhs_band_table')
assert bt.sort_index().stress_rate_oot.is_monotonic_increasing, 'bands must rank stress (A lowest ... E highest)'
log('bands:\n' + bt.round(4).to_string())
edges = np.linspace(0, 100, 11)
live = prod[prod.n_active != '0']
m['psi'] = {'train_vs_oot': round(psi(train.score, oot.score, edges), 4),
            'train_vs_production_live_exposure_structural': round(psi(train.score, live.structural, edges), 4),
            'train_vs_production_all_customers_structural': round(psi(train.score, prod.structural, edges), 4),
            'note': 'all-customer PSI compares different populations (incl. dormant and thin-file customers)'}
m['production'] = {'customers': len(prod), 'with_live_points': int((prod.live_points < 0).sum()),
                   'collection_contact_before_T': int((prod.collection_before_T > 0).sum()),
                   'affordability_flag': int(prod.affordability.sum())}

# %% [markdown]
# ## 1.8 Fairness check
# Gender and age are not inputs. Do scores differ between groups, and does the score rank within each group?
# %%
fair = con.sql("SELECT SKP_CLIENT, GENDER, 2025 - year(DATE_BIRTH) AS age FROM customers").df()
fo = oot[['SKP_CLIENT', 'bad', 'score']].merge(fair, on='SKP_CLIENT')
fo['age_band'] = pd.cut(fo.age, [0, 25, 35, 45, 55, 200], right=False,
                        labels=['<25', '25-34', '35-44', '45-54', '55+']).astype(str)
rows = []
for dim in ['GENDER', 'age_band']:
    for gval, g in fo.groupby(dim):
        if len(g) >= 1000 and g.bad.nunique() == 2:
            rows.append({'dimension': dim, 'group': gval, 'n': len(g), 'avg_score': g.score.mean(),
                         'actual_stress_rate': g.bad.mean(), 'auc_within_group': roc_auc_score(g.bad, -g.score)})
fair_tab = save_table(pd.DataFrame(rows), 'fhs_fairness_check')
log('fairness:\n' + fair_tab.round(4).to_string(index=False))

# %% [markdown]
# ## 1.9 Reason codes, pillars and segments
# Reasons compare a customer with the portfolio-average customer: the factor where the customer
# loses the most points is the main drag, the one where they gain the most is the main strength
# (shown only when it moves the score by at least 1 point).
#
# Segments start from the band; four rules override it, in this order:
# Distressed (write-off or uncured missed payment) > Live arrears (collection contact Jan–Jun 2025)
# > New to credit (thin file) > Over-extended (3+ live loans or instalments >= 50% of income).
# %%
REASON_TEXT = {
    'repay_state': 'Repayment history: {bin}', 'max_clean_credit': 'Largest loan repaid cleanly: {bin}',
    'tenure': 'Credit relationship length: {bin}', 'recency': 'Last new loan signed: {bin} ago',
    'down_payment': 'Average down payment on goods/bike loans: {bin}', 'income': 'Declared monthly income: {bin} VND',
    'pti': 'Instalment-to-income: {bin}', 'apps_90d': '{bin} credit applications in last 90 days',
    'scoring_90d': '{bin} credit scoring checks in last 90 days', 'unsigned_apps': '{bin} past applications rejected or cancelled',
}
SPECIAL_TEXT = {
    ('repay_state', '4 Clean, no loan completed yet'): 'No missed payments, no loan completed yet (track record building)',
    ('repay_state', '5 Clean, 1-2 loans completed'): 'No missed payments, 1-2 loans completed',
    ('repay_state', '6 Clean, 3+ loans completed'): 'No missed payments across 3+ completed loans',
    ('max_clean_credit', '0 None completed cleanly'): 'No loan repaid cleanly yet',
    ('down_payment', '0 No goods/bike loan'): 'No goods/bike loan history (cash/revolving only)',
    ('income', '0 Unknown'): 'Income not declared',
}


def clean_bin(b) -> str:
    b = str(b)
    return b.split(' ', 1)[1] if b[:1].isdigit() and ' ' in b and not b[1:2].isdigit() else b


def describe(feat: str, b: str) -> str:
    return SPECIAL_TEXT.get((feat, b)) or REASON_TEXT[feat].format(bin=clean_bin(b))


def reason_codes(df: pd.DataFrame, sc: Scorecard, k: int = 2) -> pd.DataFrame:
    feats = sc.active
    P = sc.points(df)[feats].to_numpy()
    avgp = sc.pts.groupby('feature').apply(lambda g: np.average(g['points'], weights=g['n']), include_groups=False)[feats].to_numpy()
    delta = P - avgp
    bins = df[feats].astype(str).to_numpy()
    neg, pos = np.argsort(delta, axis=1)[:, :k], np.argsort(-delta, axis=1)[:, :k]
    r = np.arange(len(P))
    out = {}
    for j in range(k):
        out[f'fhs_reason_neg_{j + 1}'] = [describe(feats[c], bins[i, c]) if delta[i, c] <= -1 else '' for i, c in zip(r, neg[:, j])]
        out[f'fhs_reason_pos_{j + 1}'] = [describe(feats[c], bins[i, c]) if delta[i, c] >= 1 else '' for i, c in zip(r, pos[:, j])]
    return pd.DataFrame(out, index=df.index)


allp = pd.concat([hist.points(prod[prod.has_history]), thin.points(prod[~prod.has_history])]).reindex(prod.index)
allp['live_status'] = prod.live_points
model = np.where(prod.has_history, 'history', 'thin_file')
live_rows = pd.DataFrame([{'model': mdl, 'feature': 'live_status', 'pillar': 'Live status', 'points': p}
                          for mdl in ('history', 'thin_file')
                          for p in (0.0, min(LIVE_POINTS['ew'].values()) + LIVE_POINTS['affordability'])])
pillar_cols = {}
for pillar, g in pd.concat([points_table, live_rows], ignore_index=True).groupby('pillar'):
    lo = g.groupby(['model', 'feature'])['points'].min().groupby('model').sum()
    hi = g.groupby(['model', 'feature'])['points'].max().groupby('model').sum()
    raw = allp[g.feature.unique()].sum(axis=1, min_count=1)
    lo_v, hi_v = pd.Series(model).map(lo).to_numpy(), pd.Series(model).map(hi).to_numpy()
    pillar_cols['fhs_pillar_' + pillar.lower().replace(' ', '_')] = np.round(100 * (raw.to_numpy() - lo_v) / (hi_v - lo_v), 1)
rc = pd.concat([reason_codes(prod[prod.has_history], hist), reason_codes(prod[~prod.has_history], thin)]).reindex(prod.index)
# a live collection contact is the strongest drag on the score, so it becomes the first negative reason
live_text = ('Collection contact in Jan-Jun 2025: ' + prod.collection_before_T.astype(str) + ' call(s) ('
             + prod.ew.map(LIVE_POINTS['ew']).map('{:+.0f}'.format) + ' pts)')
has_live = prod.collection_before_T > 0
rc.loc[has_live, 'fhs_reason_neg_2'] = rc.loc[has_live, 'fhs_reason_neg_1']
rc.loc[has_live, 'fhs_reason_neg_1'] = live_text[has_live]
SEG_DISTRESSED, SEG_LIVE, SEG_THIN, SEG_OVER = 'Distressed', 'Live arrears', 'New to credit', 'Over-extended'
out = pd.DataFrame({
    'SKP_CLIENT': prod.SKP_CLIENT,
    'fhs_score': prod.score,
    'fhs_band': prod.band,
    'fhs_structural_score': prod.structural,
    'fhs_live_points': prod.live_points,
    'fhs_thin_file': (~prod.has_history).astype(int),
    **pillar_cols,                     # NaN = pillar not applicable (thin file)
    'fhs_expected_stress_rate': prod.band.map(bt['stress_rate_oot'].to_dict()).round(4),
    'repay_state': prod.repay_state.map(clean_bin),
    'live_loans': prod.n_active_v.astype(int),
    'instalment_to_income': prod.pti_val.round(3),
    'affordability_flag': np.select([(prod.pti_val >= 0.5).to_numpy(), prod.n_active.eq('3+').to_numpy()],
                                    ['Instalment-to-income >= 50%', '3+ loans live at once'], default=''),
    'collection_calls_before_T': prod.collection_before_T,
    **rc,
})
rs = prod.repay_state.str[:1]
out['fhs_segment'] = np.select(
    [rs.isin(['1', '2']).to_numpy(), has_live.to_numpy(), out.fhs_thin_file.eq(1).to_numpy(), (out.affordability_flag != '').to_numpy()],
    [SEG_DISTRESSED, SEG_LIVE, SEG_THIN, SEG_OVER], default=out.fhs_band.str[2:])
out['fhs_explanation'] = (
    'FHS ' + out.fhs_score.map('{:.0f}'.format) + '/100 (' + out.fhs_band.str[2:] + ')'
    + np.where(out.fhs_thin_file == 1, ', thin file - provisional', '') + '. '
    + np.where(out.fhs_reason_pos_1 != '', 'Strength: ' + out.fhs_reason_pos_1 + '. ', '')
    + np.where(out.fhs_reason_neg_1 != '', 'Watch: ' + out.fhs_reason_neg_1 + '. ', '')
    + np.where(out.affordability_flag != '', 'Affordability flag: ' + out.affordability_flag + '.', ''))
out['fhs_explanation'] = out.fhs_explanation.str.strip()
assert len(out) == N_CUSTOMERS and out.SKP_CLIENT.is_unique
assert out.fhs_score.notna().all() and out.fhs_score.between(0, 100).all()
seg = out.groupby('fhs_segment').agg(customers=('SKP_CLIENT', 'size'), avg_score=('fhs_score', 'mean'),
                                     expected_stress=('fhs_expected_stress_rate', 'mean'),
                                     live_loans=('live_loans', 'mean')).reset_index()
seg['share'] = seg.customers / len(out)
seg_future = post.merge(out[['SKP_CLIENT', 'fhs_segment']], on='SKP_CLIENT').groupby('fhs_segment').agg(
    called_after_T=('SKP_CLIENT', 'size'), collection_rate_after_T=('y', 'mean'),
    collection_share_after_T=('collection_share', 'mean'))
seg = seg.merge(seg_future.reset_index(), on='fhs_segment', how='left')
save_table(seg, 'fhs_segment_summary')
log('segments:\n' + seg.round(4).to_string(index=False))
reasons = pd.concat([
    out.fhs_reason_neg_1.replace('', np.nan).value_counts().head(8).rename('customers').to_frame().assign(kind='main drag'),
    out.fhs_reason_pos_1.replace('', np.nan).value_counts().head(8).rename('customers').to_frame().assign(kind='main strength')])
save_table(reasons.reset_index(names='reason'), 'fhs_top_reasons')

# %% [markdown]
# ## 1.10 Save
# %%
out.to_parquet(OUTPUTS / 'fhs_scores.parquet', index=False)
prod.drop(columns=['score', 'band']).to_parquet(OUTPUTS / 'fhs_features_at_T.parquet', index=False)
oot[['SKP_CLIENT', 'bad', 'score', 'band', 'has_history', 'n_active', 'tet_wave', 'new_product']].to_parquet(OUTPUTS / 'fhs_oot_scored.parquet', index=False)
save_json(m, 'fhs_validation_metrics')
log(f'done in {time.time() - T0:.0f}s')
(TABLES / 'fhs_run_log.txt').write_text('\n'.join(LOG), encoding='utf-8')


# %% [markdown]
# # 2. CHALLENGE 2 — Customer Engagement Score (CES)
# Input: `data_model/*.parquet`, `outputs/fhs_features_at_T.parquet` (live loans at T).
#
# **What the score means.** CES 0–100 measures how much the *customer* reaches out to the company:
# how recently, how often, how responsively, through which products. Company activity is not
# engagement: 97% of calls are outbound telesales and most are never answered; 53% of leads come
# from a source (RA) whose leads convert 0.2% of the time. Counting dials or pushed leads would
# make a heavily-called customer look "engaged".
#
# **Customer-initiated touches** (before the scoring date): self-service or in-store scoring
# (app, web, partner site, store, SOB QR/link, walk-in), credit applications, leads the customer
# created (app, landing page, SOB) or accepted (Deal), inbound calls, and telesales calls where the
# customer showed interest or signed.
#
# | Component | Weight | Definition (as of date D) |
# |---|---|---|
# | Recency | 35% | 100 x 0.5^(days since last touch / 90): every 90 days of silence halves it |
# | Frequency | 25% | distinct active days in the last 12 months (log scale, capped at p95) |
# | Interaction | 15% | right-party-contact rate on outbound calls in 2025, shrunk to the population rate (no calls = population rate) |
# | Lead activity | 10% | self-created or accepted leads in the last 12 months (log, capped at p95) |
# | Product interaction | 15% | product families touched in the last 12 months (60%) + holds a live loan (40%) |
#
# **Validation against business outcomes, never used to build the score**
# 1. Back-test: score at D0 = 2025-02-01, outcome = the customer comes back and applies (or signs) by 2025-06-04.
# 2. Future: score at T = 2025-06-05, outcomes in calls Jun-2025..Aug-2026 — answered, showed interest / signed.
# 3. Convergent check: 2025 mobile-app usage (GMA sessions), which is not an input.
# %%
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


T0 = time.time()
con = connect()
D0 = '2025-02-01'                      # back-test scoring date (after the Tet application wave)
WEIGHTS = {'recency': .35, 'frequency': .25, 'interaction': .15, 'lead': .10, 'product': .15}
HALF_LIFE_DAYS, SHRINK_CALLS = 90, 5
POSITIVE = "('sales_conversion','sales_interest')"
FAMILY = """CASE WHEN {c} IN ('CD','EPOS') THEN 'durables' WHEN {c} = 'TW' THEN 'two_wheeler'
                 WHEN {c} IN ('BNPL','BNP') THEN 'bnpl'
                 WHEN {c} IN ('CC','CCX','CC_SC','CC_CC','VCC','CCW') THEN 'card'
                 WHEN {c} IN ('CL','CLX','CW','ACL','ACLX','ACLD','SCW') THEN 'cash' ELSE 'other' END"""
log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    log_lines.append(msg)

# %% [markdown]
# ## 2.1 Touch log and raw features as of a date
# %%
def touch_sql(D: str) -> str:
    Dt = f"TIMESTAMP '{D}'"
    return f"""
    SELECT SKP_CLIENT, EVENT_AT AS t, 'self-service / store scoring' AS kind, {FAMILY.format(c='PRODUCT_CODE')} AS family
      FROM behavior WHERE SELF_INITIATED AND EVENT_AT < {Dt}
    UNION ALL SELECT SKP_CLIENT, t_apply, 'credit application', {FAMILY.format(c='product')} FROM k WHERE t_apply < {Dt}
    UNION ALL SELECT SKP_CLIENT, CREATED_AT, 'lead created / accepted', {FAMILY.format(c='PRODUCT')}
      FROM leads WHERE (SELF_INITIATED OR REASON = 'Deal') AND CREATED_AT < {Dt}
    UNION ALL SELECT SKP_CLIENT, START_AT, CASE WHEN CALL_TYPE = 'Inbound' THEN 'inbound call' ELSE 'interest on a sales call' END,
                     {FAMILY.format(c='PRODUCT_TYPE')}
      FROM calls_rpc WHERE START_AT < {Dt} AND (CALL_TYPE = 'Inbound' OR OUTCOME_GROUP IN {POSITIVE})"""


def raw_features(D: str, prod: bool) -> pd.DataFrame:
    Dt = f"TIMESTAMP '{D}'"
    live = ("SELECT SKP_CLIENT, count(*) AS live_loans FROM k WHERE is_signed AND t_ref < {Dt} AND "
            + ("status IN ('Active','Signed')" if prod else f"(maturity > {Dt} OR (maturity IS NULL AND product IN {REVOLVING})) AND NOT (wo AND t_ref + INTERVAL 9 MONTH <= {Dt})")
            + " GROUP BY 1").format(Dt=Dt)
    df = con.sql(f"""
    WITH t AS ({touch_sql(D)}),
    agg AS (SELECT SKP_CLIENT, max(t) AS last_touch, min(t) AS first_touch,
                   count(DISTINCT CAST(t AS DATE)) FILTER (WHERE t >= {Dt} - INTERVAL 365 DAY) AS active_days_12m,
                   count(*) FILTER (WHERE t >= {Dt} - INTERVAL 90 DAY) AS touches_90d,
                   count(DISTINCT family) FILTER (WHERE t >= {Dt} - INTERVAL 365 DAY AND family <> 'other') AS families_12m,
                   count(*) FILTER (WHERE kind = 'lead created / accepted' AND t >= {Dt} - INTERVAL 365 DAY) AS leads_12m,
                   count(*) FILTER (WHERE kind = 'inbound call') AS inbound_calls,
                   arg_max(kind, (t, kind)) AS last_touch_kind
            FROM t GROUP BY 1),
    c AS (SELECT SKP_CLIENT, count(*) FILTER (WHERE CALL_TYPE = 'Outbound') AS outbound_calls,
                 count(*) FILTER (WHERE CALL_TYPE = 'Outbound' AND RPC) AS outbound_rpc,
                 count(*) FILTER (WHERE OUTCOME_GROUP IN {POSITIVE}) AS positive_calls
          FROM calls_rpc WHERE START_AT < {Dt} GROUP BY 1),
    lv AS ({live})
    SELECT cu.SKP_CLIENT, a.last_touch, a.first_touch, a.last_touch_kind,
           date_diff('day', a.last_touch, {Dt}) AS days_since_touch,
           coalesce(a.active_days_12m, 0) AS active_days_12m, coalesce(a.touches_90d, 0) AS touches_90d,
           coalesce(a.families_12m, 0) AS families_12m, coalesce(a.leads_12m, 0) AS leads_12m,
           coalesce(a.inbound_calls, 0) AS inbound_calls,
           coalesce(c.outbound_calls, 0) AS outbound_calls, coalesce(c.outbound_rpc, 0) AS outbound_rpc,
           coalesce(c.positive_calls, 0) AS positive_calls, coalesce(lv.live_loans, 0) AS live_loans
    FROM customers cu LEFT JOIN agg a USING (SKP_CLIENT) LEFT JOIN c USING (SKP_CLIENT) LEFT JOIN lv USING (SKP_CLIENT)
    ORDER BY cu.SKP_CLIENT""").df()
    assert len(df) == N_CUSTOMERS and df.SKP_CLIENT.is_unique
    return df


def capped_log(x: pd.Series, cap: float) -> pd.Series:
    return (100 * np.log1p(x.clip(lower=0)) / np.log1p(cap)).clip(0, 100) if cap > 0 else pd.Series(0.0, index=x.index)


def components(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    c = pd.DataFrame(index=df.index)
    c['recency'] = (100 * 0.5 ** (df.days_since_touch / HALF_LIFE_DAYS)).fillna(0.0)
    c['frequency'] = capped_log(df.active_days_12m, params['cap_active_days'])
    shrunk = (df.outbound_rpc + SHRINK_CALLS * params['rpc_prior']) / (df.outbound_calls + SHRINK_CALLS)
    c['interaction'] = (100 * shrunk / params['rpc_cap']).clip(0, 100)
    c['lead'] = capped_log(df.leads_12m, params['cap_leads'])
    c['product'] = 0.6 * (100 * df.families_12m.clip(upper=3) / 3) + 0.4 * 100 * (df.live_loans > 0)
    c['score'] = sum(WEIGHTS[k] * c[k] for k in WEIGHTS).round(2).clip(0, 100)
    return c


def fit_params(df: pd.DataFrame) -> dict:
    """Scaling constants, fixed on the production snapshot and reused unchanged for the back-test."""
    called = df[df.outbound_calls > 0]
    prior = called.outbound_rpc.sum() / called.outbound_calls.sum()
    shrunk = (called.outbound_rpc + SHRINK_CALLS * prior) / (called.outbound_calls + SHRINK_CALLS)
    return {'cap_active_days': float(df.active_days_12m[df.active_days_12m > 0].quantile(.95)),
            'cap_leads': float(df.leads_12m[df.leads_12m > 0].quantile(.95)),
            'rpc_prior': float(prior), 'rpc_cap': float(shrunk.quantile(.95))}

# %% [markdown]
# ## 2.2 Score at T and at the back-test date
# %%
at_t = raw_features(CUTOFF, prod=True)
params = fit_params(at_t)
comp_t = components(at_t, params)
log(f'scaling parameters: {params}')
log('score at T: ' + comp_t.score.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_dict().__repr__())
touch_mix = con.sql(f"""SELECT kind, count(*) AS touches, count(DISTINCT SKP_CLIENT) AS customers
    FROM ({touch_sql(CUTOFF)}) WHERE t >= TIMESTAMP '{CUTOFF}' - INTERVAL 365 DAY GROUP BY 1 ORDER BY 2 DESC""").df()
save_table(touch_mix, 'ces_touch_mix_12m')
log('customer-initiated touches, last 12 months:\n' + touch_mix.to_string(index=False))

# %% [markdown]
# ## 2.3 Sampling frame and validation 1 — back-test on repeat activity
# Every customer in the file made their last credit application between Dec-2024 and Jun-2025
# (84% in Dec-2024/Jan-2025, the Tet wave): the dataset is an *applicant cohort*. An outcome such as
# "applies after Dec-2024" is therefore true by construction and cannot validate anything.
#
# Clean back-test: customers who had already applied in Dec-2024/Jan-2025 (so they are in the
# file regardless of what follows), scored at D0 = 2025-02-01; outcome = they come back and apply
# again, or sign, between Feb and Jun-2025.
# %%
frame_tab = con.sql("""SELECT strftime(last_app, '%Y-%m') AS last_application_month, count(*) AS customers
    FROM (SELECT SKP_CLIENT, max(t_apply) AS last_app FROM k GROUP BY 1) GROUP BY 1 ORDER BY 1""").df()
frame_tab['share'] = frame_tab.customers / frame_tab.customers.sum()
save_table(frame_tab, 'sampling_frame_last_application')
log('sampling frame (month of last application):\n' + frame_tab.round(4).to_string(index=False))
assert frame_tab.last_application_month.min() >= '2024-12', 'sampling frame changed'

at_d0 = raw_features(D0, prod=False)
comp_d0 = components(at_d0, params)
repeat = con.sql(f"""SELECT SKP_CLIENT,
        max((t_apply >= TIMESTAMP '{D0}')::INT) AS applied_again,
        max((is_signed AND t_ref >= TIMESTAMP '{D0}')::INT) AS signed_again
    FROM k WHERE t_apply >= TIMESTAMP '2024-12-01' GROUP BY 1
    HAVING min(t_apply) < TIMESTAMP '{D0}'""").df()   # in the frame before D0
bt_df = at_d0[['SKP_CLIENT']].join(comp_d0).merge(repeat, on='SKP_CLIENT')
val = {'sampling_frame': {'customers': int(frame_tab.customers.sum()),
                          'last_application_dec2024_jan2025_share': round(float(frame_tab.share[frame_tab.last_application_month.isin(['2024-12', '2025-01'])].sum()), 4)},
       'backtest': {'scoring_date': D0, 'population': 'applied in Dec-2024/Jan-2025 (in the file before D0)', 'customers': len(bt_df)}}
for y in ['applied_again', 'signed_again']:
    val['backtest'][y] = {**metrics(bt_df[y], -bt_df.score),
                          'component_auc': {k: round(roc_auc_score(bt_df[y], bt_df[k]), 4) for k in WEIGHTS}}
bt_dec = decile_table(bt_df.applied_again, -bt_df.score)   # decile 1 = most engaged (score negated)
bt_dec['decile'] = 11 - bt_dec.decile.astype(int)          # 10 = most engaged
save_table(bt_dec.sort_values('decile'), 'ces_backtest_deciles')
# benchmark: weights learnt by a logistic regression on the same back-test (in-sample upper bound for re-weighting)
lr = LogisticRegression(max_iter=2000).fit(bt_df[list(WEIGHTS)], bt_df.applied_again)
val['backtest']['benchmark_logistic_weights_in_sample'] = round(roc_auc_score(bt_df.applied_again, lr.decision_function(bt_df[list(WEIGHTS)])), 4)
lw = np.clip(lr.coef_[0], 0, None)
val['backtest']['logistic_implied_weights'] = dict(zip(WEIGHTS, np.round(lw / lw.sum(), 3)))
log('back-test: ' + json.dumps(val['backtest'], indent=1, default=str))

# %% [markdown]
# ## 2.4 Validation 2 — calls after T: are engaged customers easier to reach and more interested?
# Only customers the company called after T. The call list is the company's choice, so this measures
# how the score ranks *within* the people telesales actually works.
# %%
future = con.sql(f"""SELECT SKP_CLIENT, count(*) AS calls_after,
        count(*) FILTER (WHERE CALL_TYPE = 'Outbound') AS outbound_after,
        count(*) FILTER (WHERE CALL_TYPE = 'Outbound' AND RPC) AS rpc_after,
        count(*) FILTER (WHERE OUTCOME_GROUP IN {POSITIVE}) AS positive_after,
        count(*) FILTER (WHERE OUTCOME_GROUP = 'sales_conversion') AS conversion_after,
        count(*) FILTER (WHERE OUTCOME_GROUP = 'sales_decline') AS decline_after
    FROM calls_rpc WHERE START_AT >= TIMESTAMP '{CUTOFF}' GROUP BY 1""").df()
fv = at_t[['SKP_CLIENT', 'outbound_calls', 'live_loans']].join(comp_t).merge(future, on='SKP_CLIENT')
fv = fv[fv.outbound_after > 0].copy()
fv['reached'] = (fv.rpc_after > 0).astype(int)
fv['interested'] = (fv.positive_after > 0).astype(int)
fv['converted'] = (fv.conversion_after > 0).astype(int)
fv['rpc_rate'] = fv.rpc_after / fv.outbound_after
fv['call_volume'] = pd.cut(fv.outbound_after, [0, 1, 3, 7, 15, 30, 10_000], labels=['1', '2-3', '4-7', '8-15', '16-30', '31+']).astype(str)
val['future_calls'] = {'window': f'{CUTOFF} .. 2026-08-31', 'called_customers': len(fv)}
for y in ['reached', 'interested', 'converted']:
    val['future_calls'][y] = {'rate': round(fv[y].mean(), 4), 'auc': round(roc_auc_score(fv[y], fv.score), 4),
        'auc_call_volume_adjusted': stratified_auc(fv.assign(neg=-fv.score), y, 'neg', 'call_volume')['weighted_auc'],
        'component_auc': {k: round(roc_auc_score(fv[y], fv[k]), 4) for k in WEIGHTS}}
fv['decile'] = pd.qcut(fv.score.rank(method='first'), 10, labels=range(1, 11)).astype(int)   # 10 = most engaged
fut_dec = fv.groupby('decile').agg(customers=('SKP_CLIENT', 'size'), score_min=('score', 'min'), score_max=('score', 'max'),
    outbound_calls=('outbound_after', 'sum'), rpc_rate=('rpc_rate', 'mean'), reached=('reached', 'mean'),
    interested=('interested', 'mean'), converted=('converted', 'mean'), positive_calls=('positive_after', 'sum')).reset_index()
fut_dec['lift_interested'] = fut_dec.interested / fv.interested.mean()
save_table(fut_dec, 'ces_future_deciles')
log('future calls: ' + json.dumps(val['future_calls'], indent=1, default=str))
log(fut_dec.round(4).to_string(index=False))

# telesales targeting: if the same calls had been ordered by CES, how many interested customers per call?
order = fv.sort_values('score', ascending=False)
gains = pd.DataFrame({'share_of_called_customers': np.arange(1, len(order) + 1) / len(order),
                      'share_of_outbound_calls': order.outbound_after.cumsum().to_numpy() / order.outbound_after.sum(),
                      'share_of_interested_customers': order.interested.cumsum().to_numpy() / order.interested.sum(),
                      'share_of_conversions': order.converted.cumsum().to_numpy() / max(order.converted.sum(), 1)})
gains_points = gains.iloc[np.linspace(0, len(gains) - 1, 101).astype(int)].reset_index(drop=True)
save_table(gains_points, 'ces_telesales_gains')
for q in (.3, .5):
    row = gains.iloc[int(q * len(gains)) - 1]
    val['future_calls'][f'top_{int(q * 100)}pct_by_ces'] = {k: round(float(v), 4) for k, v in row.items()}
log('targeting: ' + json.dumps({k: v for k, v in val['future_calls'].items() if k.startswith('top_')}, indent=1))

# %% [markdown]
# ## 2.5 Validation 3 — convergent check with app usage (GMA sessions per week, 2025; not an input)
# %%
app = con.sql('SELECT SKP_CLIENT, APP_SESSIONS_PER_WEEK_2025 AS app_sessions FROM customers ORDER BY SKP_CLIENT').df()
cv = app.join(comp_t.score)
cv['score_band'] = pd.cut(cv.score, [-0.1, 5, 20, 40, 60, 100.1], labels=['<5', '5-20', '20-40', '40-60', '60+'])
app_tab = cv.groupby('score_band', observed=True).agg(customers=('SKP_CLIENT', 'size'),
    app_users=('app_sessions', lambda s: (s > 0).mean()), sessions_per_week=('app_sessions', 'mean')).reset_index()
save_table(app_tab, 'ces_app_usage_check')
val['app_usage'] = {'auc_app_user': round(roc_auc_score((cv.app_sessions > 0).astype(int), cv.score), 4)}
log('app usage check:\n' + app_tab.round(4).to_string(index=False))

# %% [markdown]
# ## 2.6 Interaction behaviour: when and how often to call
# Outbound calls only. Right-party contact (RPC) = the customer was actually reached.
# %%
by_hour = con.sql("""SELECT hour(START_AT) AS hour, count(*) AS calls, avg(RPC::INT) AS rpc_rate,
        avg((OUTCOME_GROUP IN ('sales_conversion','sales_interest'))::INT) AS interest_rate
    FROM calls_rpc WHERE CALL_TYPE = 'Outbound' GROUP BY 1 HAVING count(*) >= 5000 ORDER BY 1""").df()
save_table(by_hour, 'ces_rpc_by_hour')
fatigue = con.sql("""
    WITH x AS (SELECT SKP_CLIENT, START_AT, RPC::INT AS r FROM calls_rpc WHERE CALL_TYPE = 'Outbound'),
    y AS (SELECT *, coalesce(sum(r) OVER w, 0) AS prior_rpc, count(*) OVER w AS prior_attempts
          FROM x WINDOW w AS (PARTITION BY SKP_CLIENT ORDER BY START_AT ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING))
    SELECT least(prior_attempts, 15) AS prior_attempts, prior_rpc > 0 AS reached_before,
           count(*) AS calls, avg(r) AS rpc_rate
    FROM y GROUP BY 1, 2 ORDER BY 2, 1""").df()
save_table(fatigue, 'ces_attempt_fatigue')
never = fatigue[~fatigue.reached_before]
waste = never[never.prior_attempts >= 10]
ops = {'outbound_calls': int(fatigue.calls.sum()),
       'calls_to_never_reached_after_10_failures': int(waste.calls.sum()),
       'rpc_rate_after_10_failures': round(float(np.average(waste.rpc_rate, weights=waste.calls)), 4),
       'rpc_rate_first_attempt': round(float(never.loc[never.prior_attempts == 0, 'rpc_rate'].iloc[0]), 4),
       'best_hours': by_hour.nlargest(3, 'rpc_rate')[['hour', 'rpc_rate']].round(4).to_dict('records'),
       'worst_hours': by_hour.nsmallest(3, 'rpc_rate')[['hour', 'rpc_rate']].round(4).to_dict('records')}
ops['share_of_calls_wasted'] = round(ops['calls_to_never_reached_after_10_failures'] / ops['outbound_calls'], 4)
ops['rpc_lost_if_capped_at_10'] = int(round((waste.calls * waste.rpc_rate).sum()))
val['call_operations'] = ops
lead_src = con.sql("""SELECT LEAD_SOURCE, SELF_INITIATED, count(*) AS leads, count(DISTINCT SKP_CLIENT) AS customers,
        avg((REASON = 'Deal')::INT) AS deal_rate, avg((REASON = 'Not Interest')::INT) AS not_interested_rate
    FROM leads GROUP BY 1, 2 ORDER BY leads DESC""").df()
save_table(lead_src, 'ces_lead_source_funnel')
log('call operations: ' + json.dumps(ops, indent=1))
log('lead sources:\n' + lead_src.round(4).to_string(index=False))

# %% [markdown]
# ## 2.7 Segments and drivers
# Tiers are set on the score where the future interest rate steps up. Because every customer applied
# during Dec-2024..Jun-2025, the lowest tier is not "never engaged" but *Passive*: one application in
# the wave and silence since. One rule comes first: *Unreachable* (called 5+ times in 2025, never reached).
# %%
TIER_CUTS = [-0.1, 35, 45, 60, 100.1]
TIER_LABELS = ['Passive', 'Warm', 'Engaged', 'Highly engaged']
ces = at_t[['SKP_CLIENT', 'days_since_touch', 'active_days_12m', 'touches_90d', 'families_12m', 'leads_12m',
            'inbound_calls', 'outbound_calls', 'outbound_rpc', 'positive_calls', 'live_loans', 'last_touch_kind']].join(
    comp_t.rename(columns={k: f'ces_{k}' for k in comp_t.columns}))
ces['ces_tier'] = pd.cut(ces.ces_score, TIER_CUTS, labels=TIER_LABELS).astype(str)
never_touched = at_t.last_touch.isna()
assert never_touched.sum() == 0, 'every customer in the applicant cohort has at least one touch'
unreachable = (ces.outbound_calls >= 5) & (ces.outbound_rpc == 0)
ces['ces_segment'] = np.where(unreachable, 'Unreachable', ces.ces_tier)
seg_future = fv[['SKP_CLIENT', 'reached', 'interested', 'converted', 'rpc_rate']].merge(ces[['SKP_CLIENT', 'ces_segment', 'ces_tier']], on='SKP_CLIENT')
tier_tab = seg_future.groupby('ces_tier').agg(called_after_T=('SKP_CLIENT', 'size'), reached=('reached', 'mean'),
                                              interested=('interested', 'mean'), converted=('converted', 'mean')).reindex(TIER_LABELS)
log('future outcomes by tier:\n' + tier_tab.round(4).to_string())
assert tier_tab.interested.is_monotonic_increasing, 'tiers must rank future interest'
seg = ces.groupby('ces_segment').agg(customers=('SKP_CLIENT', 'size'), avg_score=('ces_score', 'mean'),
    median_days_since_touch=('days_since_touch', 'median'), avg_active_days_12m=('active_days_12m', 'mean')).reset_index()
seg['share'] = seg.customers / len(ces)
seg = seg.merge(seg_future.groupby('ces_segment').agg(called_after_T=('SKP_CLIENT', 'size'), reached_after_T=('reached', 'mean'),
    interested_after_T=('interested', 'mean'), converted_after_T=('converted', 'mean')).reset_index(), on='ces_segment', how='left')
save_table(seg, 'ces_segment_summary')
log('segments:\n' + seg.round(4).to_string(index=False))

# drivers: future interest rate by observable factor, relative to the called average
fhs_feat = pd.read_parquet(OUTPUTS / 'fhs_features_at_T.parquet', columns=['SKP_CLIENT', 'repay_state', 'tenure'])
first_src = con.sql(f"""SELECT SKP_CLIENT, arg_min(LEAD_SOURCE, (CREATED_AT, LEAD_SOURCE)) AS first_lead_source
    FROM leads WHERE CREATED_AT < TIMESTAMP '{CUTOFF}' GROUP BY 1""").df()
dv = fv.merge(ces[['SKP_CLIENT', 'last_touch_kind', 'families_12m', 'days_since_touch', 'outbound_rpc']], on='SKP_CLIENT') \
       .merge(fhs_feat, on='SKP_CLIENT').merge(first_src, on='SKP_CLIENT', how='left').merge(app, on='SKP_CLIENT')
dv['Last customer touch'] = pd.cut(dv.days_since_touch, [-1, 30, 90, 365, 100_000], labels=['<=30 days', '31-90 days', '91-365 days', '>1 year']).astype(str).replace('nan', 'never')
dv['Last touch type'] = dv.last_touch_kind.fillna('none')
dv['Product families touched (12m)'] = dv.families_12m.clip(upper=3).astype(str).replace({'3': '3+'})
dv['Reached by phone before T (2025)'] = np.select([dv.outbound_calls == 0, dv.outbound_rpc > 0], ['not called', 'yes'], 'called, never reached')
dv['Uses the mobile app (2025)'] = np.where(dv.app_sessions > 0, 'yes', 'no')
dv['First lead source'] = dv.first_lead_source.fillna('no lead')
dv['Repayment history'] = dv.repay_state.str[2:]
drivers = []
for factor in ['Last customer touch', 'Last touch type', 'Product families touched (12m)', 'Reached by phone before T (2025)',
               'Uses the mobile app (2025)', 'First lead source', 'Repayment history']:
    g = dv.groupby(factor).agg(customers=('SKP_CLIENT', 'size'), interested=('interested', 'mean'), reached=('reached', 'mean')).reset_index()
    g = g[g.customers >= 1000].rename(columns={factor: 'level'})
    g.insert(0, 'factor', factor)
    drivers.append(g)
drivers = pd.concat(drivers, ignore_index=True)
drivers['interest_lift'] = drivers.interested / fv.interested.mean()
drivers['reach_lift'] = drivers.reached / fv.reached.mean()
save_table(drivers, 'ces_drivers')
log('drivers:\n' + drivers.round(3).to_string(index=False))

# %% [markdown]
# ## 2.8 Reason text and save
# %%
COMP_TEXT = {'recency': 'last reached out {d} days ago', 'frequency': 'active on {a} day(s) in the last 12 months',
             'interaction': 'answers {r} of telesales calls', 'lead': '{l} self-made or accepted lead(s) in 12 months',
             'product': '{p} product famil(ies) touched in 12 months'}
avg_comp = comp_t[list(WEIGHTS)].mean()
delta = (comp_t[list(WEIGHTS)] - avg_comp) * pd.Series(WEIGHTS)
top_pos, top_neg = delta.idxmax(axis=1), delta.idxmin(axis=1)
rpc_txt = np.where(ces.outbound_calls > 0, (ces.outbound_rpc.astype(str) + '/' + ces.outbound_calls.astype(str)), 'no')


def comp_text(name: pd.Series) -> pd.Series:
    out = pd.Series('', index=name.index)
    for comp, tmpl in COMP_TEXT.items():
        mk = name == comp
        out[mk] = [tmpl.format(d='' if pd.isna(dd) else int(dd), a=int(a), r=r, l=int(l), p=int(p))
                   for dd, a, r, l, p in zip(ces.days_since_touch[mk], ces.active_days_12m[mk], rpc_txt[mk.to_numpy()],
                                             ces.leads_12m[mk], ces.families_12m[mk])]
    return out


ces['ces_reason_pos'] = np.where(delta.max(axis=1) >= 1, comp_text(top_pos), '')
ces['ces_reason_neg'] = np.where(delta.min(axis=1) <= -1, comp_text(top_neg), '')
ces['ces_explanation'] = ('CES ' + ces.ces_score.map('{:.0f}'.format) + '/100 (' + ces.ces_segment + '). '
    + np.where(ces.ces_reason_pos != '', 'Strength: ' + ces.ces_reason_pos + '. ', '')
    + np.where(ces.ces_reason_neg != '', 'Gap: ' + ces.ces_reason_neg + '.', '')).str.strip()
assert ces.ces_score.between(0, 100).all() and len(ces) == N_CUSTOMERS
ces.to_parquet(OUTPUTS / 'ces_scores.parquet', index=False)
val['weights'] = WEIGHTS
val['scaling'] = params
val['sensitivity'] = {}
for name, w in {'recency_heavier': {'recency': .45, 'frequency': .20, 'interaction': .10, 'lead': .10, 'product': .15},
                'equal_weights': {k: .2 for k in WEIGHTS}}.items():
    alt = sum(w[k] * comp_t[k] for k in w)
    alt_by_client = pd.Series(alt.to_numpy(), index=at_t.SKP_CLIENT)
    val['sensitivity'][name] = {'rank_corr_with_primary': round(float(comp_t.score.rank().corr(alt.rank())), 4),
                                'future_interest_auc': round(roc_auc_score(fv.interested, fv.SKP_CLIENT.map(alt_by_client)), 4)}
val['sensitivity']['primary'] = {'future_interest_auc': val['future_calls']['interested']['auc']}
save_json(val, 'ces_validation_metrics')
log(f'done in {time.time() - T0:.0f}s')
(TABLES / 'ces_run_log.txt').write_text('\n'.join(log_lines), encoding='utf-8')


# %% [markdown]
# # 3. Customer-level output and the health x engagement view (input to Challenge 3)
# Input: `outputs/fhs_scores.parquet`, `outputs/ces_scores.parquet`.
# Output: `outputs/customer_scores.csv` / `.parquet` — one row per customer with both scores, bands,
# segments, components, reasons and a plain-language explanation; plus the joint tables that
# Challenge 3 (segmentation and next best action) starts from.
#
# The joint view also checks how the call centre spends its effort today: after T, which customers
# receive sales offers, and how they respond.
# %%
import numpy as np
import pandas as pd


con = connect()
fhs = pd.read_parquet(OUTPUTS / 'fhs_scores.parquet')
ces = pd.read_parquet(OUTPUTS / 'ces_scores.parquet')
mismatch = fhs[['SKP_CLIENT', 'live_loans']].merge(ces[['SKP_CLIENT', 'live_loans']], on='SKP_CLIENT').query('live_loans_x != live_loans_y')
assert mismatch.empty, f'live loans differ between FHS and CES for {len(mismatch)} customers'
df = fhs.merge(ces.drop(columns='live_loans'), on='SKP_CLIENT', validate='one_to_one').sort_values('SKP_CLIENT', ignore_index=True)
assert len(df) == N_CUSTOMERS and df.SKP_CLIENT.is_unique

# %% [markdown]
# ## 3.1 Customer table
# %%
df['health_engagement_cell'] = df.fhs_band.str[2:] + ' x ' + df.ces_tier
cols = ['SKP_CLIENT',
        'fhs_score', 'fhs_band', 'fhs_segment', 'fhs_structural_score', 'fhs_live_points', 'fhs_expected_stress_rate',
        'fhs_thin_file', *[c for c in df.columns if c.startswith('fhs_pillar_')],
        'repay_state', 'live_loans', 'instalment_to_income', 'affordability_flag', 'collection_calls_before_T',
        'fhs_reason_pos_1', 'fhs_reason_pos_2', 'fhs_reason_neg_1', 'fhs_reason_neg_2', 'fhs_explanation',
        'ces_score', 'ces_tier', 'ces_segment', 'ces_recency', 'ces_frequency', 'ces_interaction', 'ces_lead', 'ces_product',
        'days_since_touch', 'active_days_12m', 'last_touch_kind', 'outbound_calls', 'outbound_rpc',
        'ces_reason_pos', 'ces_reason_neg', 'ces_explanation', 'health_engagement_cell']
out = df[cols].rename(columns={'outbound_calls': 'outbound_calls_2025_before_T', 'outbound_rpc': 'calls_answered_2025_before_T'})
for c in ['ces_recency', 'ces_frequency', 'ces_interaction', 'ces_lead', 'ces_product', 'ces_score']:
    out[c] = out[c].round(1)
out.to_parquet(OUTPUTS / 'customer_scores.parquet', index=False)
out.to_csv(OUTPUTS / 'customer_scores.csv', index=False, float_format='%.4g')
dictionary = pd.DataFrame([
    ('SKP_CLIENT', 'Customer id'),
    ('fhs_score', 'Financial Health Score 0-100 at 2025-06-05 = structural points + live points, clipped'),
    ('fhs_band', 'A Thriving (75+) / B Healthy (65-75) / C Stable (55-65) / D Fragile (40-55) / E Stressed (<40)'),
    ('fhs_segment', 'Distressed > Live arrears > New to credit > Over-extended > band name'),
    ('fhs_structural_score', 'Scorecard points from credit history (validated out of time on 2024 data)'),
    ('fhs_live_points', 'Points from current status: collection contact Jan-Jun 2025, affordability flag (validated on 2026 calls)'),
    ('fhs_expected_stress_rate', 'Out-of-time early-stress rate of the band'),
    ('fhs_thin_file', '1 = no signed contract before T; score is provisional'),
    ('fhs_pillar_*', 'Position 0 (worst) - 100 (best) within each pillar; empty if the pillar does not apply'),
    ('repay_state', 'Repayment history state at T'),
    ('live_loans', 'Loans active at T'),
    ('instalment_to_income', 'Monthly instalments of live instalment loans / latest declared income'),
    ('affordability_flag', 'Instalment-to-income >= 50% or 3+ live loans'),
    ('collection_calls_before_T', 'Calls coded as collection, Jan-Jun 2025'),
    ('fhs_reason_*', 'Factors furthest above (pos) / below (neg) the average customer, >= 1 point'),
    ('ces_score', 'Customer Engagement Score 0-100 at 2025-06-05 (customer-initiated activity only)'),
    ('ces_tier', 'Passive (<35) / Warm (35-45) / Engaged (45-60) / Highly engaged (60+)'),
    ('ces_segment', 'Unreachable (5+ calls in 2025, never answered) or the tier'),
    ('ces_recency..ces_product', 'Component scores 0-100 (weights 35/25/15/10/15)'),
    ('days_since_touch', 'Days since the last customer-initiated touch'),
    ('active_days_12m', 'Distinct days with a customer-initiated touch in the last 12 months'),
    ('ces_reason_pos / ces_reason_neg', 'Component that adds / removes the most versus the average customer'),
    ('health_engagement_cell', 'FHS band x CES tier, the starting grid for segmentation and next best action'),
], columns=['column', 'meaning'])
save_table(dictionary, 'customer_scores_dictionary')

# %% [markdown]
# ## 3.2 Health x engagement grid, with what happened after T
# %%
fut = con.sql(f"""SELECT SKP_CLIENT,
        count(*) FILTER (WHERE CALL_TYPE = 'Outbound') AS outbound_after,
        count(*) FILTER (WHERE OUTCOME_GROUP IN ('sales_conversion','sales_decline','sales_ineligible')) AS sales_calls_after,
        count(*) FILTER (WHERE CALL_TYPE = 'Outbound' AND PRODUCT_TYPE IS NOT NULL AND PRODUCT_TYPE <> 'XNA') AS product_dials_after,
        count(*) FILTER (WHERE OUTCOME_GROUP IN ('sales_conversion','sales_interest')) AS positive_after,
        count(*) FILTER (WHERE OUTCOME_GROUP = 'sales_conversion') AS conversion_after,
        count(*) FILTER (WHERE OUTCOME_GROUP = 'collection') AS collection_after
    FROM calls_rpc WHERE START_AT >= TIMESTAMP '{CUTOFF}' GROUP BY 1""").df()
j = out[['SKP_CLIENT', 'fhs_band', 'fhs_segment', 'ces_tier', 'ces_segment', 'live_loans']].merge(fut, on='SKP_CLIENT', how='left').fillna(0)
j['called'] = (j.outbound_after > 0).astype(int)
j['interested'] = (j.positive_after > 0).astype(int)
j['converted'] = (j.conversion_after > 0).astype(int)
j['in_collection'] = (j.collection_after > 0).astype(int)
called = j[j.called == 1]
grid = j.groupby(['fhs_band', 'ces_tier']).agg(customers=('SKP_CLIENT', 'size'), called_after_T=('called', 'mean')).join(
    called.groupby(['fhs_band', 'ces_tier']).agg(interested_if_called=('interested', 'mean'), converted_if_called=('converted', 'mean'),
                                                 collection_if_called=('in_collection', 'mean'))).reset_index()
grid['share'] = grid.customers / N_CUSTOMERS
save_table(grid, 'joint_band_tier_grid')
print(grid.round(4).to_string(index=False))

# %% [markdown]
# ## 3.3 Where do sales offers go after T?
# A *delivered sales offer* = a call whose outcome shows the customer heard a product offer and answered it
# (signed / applied, declined, or found not eligible). Product-tagged dials are not used for this: for
# customers in collection a large part of the product-tagged dials are collection campaigns.
# Responsible lending: offers to customers already in collection or with unresolved arrears.
# %%
alloc = j.groupby('fhs_segment').agg(customers=('SKP_CLIENT', 'size'), sales_calls=('sales_calls_after', 'sum'),
    customers_offered=('sales_calls_after', lambda s: (s > 0).sum()), conversions=('conversion_after', 'sum'),
    product_dials=('product_dials_after', 'sum'), collection_calls=('collection_after', 'sum')).reset_index()
alloc['share_of_sales_calls'] = alloc.sales_calls / alloc.sales_calls.sum()
alloc['share_of_customers'] = alloc.customers / alloc.customers.sum()
alloc['conversions_per_1000_sales_calls'] = 1000 * alloc.conversions / alloc.sales_calls
alloc['sales_calls_per_customer'] = alloc.sales_calls / alloc.customers
alloc['offers_per_1000_customers'] = 1000 * alloc.sales_calls / alloc.customers
alloc['share_of_customers_offered'] = alloc.customers_offered / alloc.customers
save_table(alloc.sort_values('sales_calls', ascending=False), 'joint_sales_call_allocation')
print(alloc.round(4).to_string(index=False))
alloc_tier = j.groupby('ces_tier').agg(customers=('SKP_CLIENT', 'size'), sales_calls=('sales_calls_after', 'sum'),
    product_dials=('product_dials_after', 'sum'), conversions=('conversion_after', 'sum')).reindex(TIER_ORDER).reset_index()
alloc_tier['share_of_product_dials'] = alloc_tier.product_dials / alloc_tier.product_dials.sum()
alloc_tier['share_of_sales_calls'] = alloc_tier.sales_calls / alloc_tier.sales_calls.sum()
alloc_tier['share_of_conversions'] = alloc_tier.conversions / alloc_tier.conversions.sum()
alloc_tier['conversions_per_1000_sales_calls'] = 1000 * alloc_tier.conversions / alloc_tier.sales_calls
save_table(alloc_tier, 'joint_sales_call_allocation_by_tier')
print(alloc_tier.round(4).to_string(index=False))
risky = j.fhs_segment.isin(['Live arrears', 'Distressed']) | j.fhs_band.eq('E Stressed')
seg_rate = alloc.set_index('fhs_segment')
summary = {
    'customers': N_CUSTOMERS,
    'delivered_sales_offers_after_T': int(j.sales_calls_after.sum()),
    'offers_to_live_arrears_distressed_or_E': int(j.sales_calls_after[risky].sum()),
    'customers_live_arrears_distressed_or_E_offered': int(((j.sales_calls_after > 0) & risky).sum()),
    'live_arrears_offered_share': round(float(seg_rate.loc['Live arrears', 'share_of_customers_offered']), 4),
    'healthy_offered_share': round(float(seg_rate.loc['Healthy', 'share_of_customers_offered']), 4),
    'live_arrears_conversions': int(seg_rate.loc['Live arrears', 'conversions']),
    'healthy_and_engaged_customers': int((j.fhs_band.isin(['A Thriving', 'B Healthy']) & j.ces_tier.isin(['Engaged', 'Highly engaged'])).sum()),
    'healthy_and_engaged_not_called_after_T': int((j.fhs_band.isin(['A Thriving', 'B Healthy']) & j.ces_tier.isin(['Engaged', 'Highly engaged']) & (j.called == 0)).sum()),
    'fhs_ces_rank_correlation': round(float(out.fhs_score.rank().corr(out.ces_score.rank())), 4),
}
summary['share_of_offers_to_risky'] = round(summary['offers_to_live_arrears_distressed_or_E'] / summary['delivered_sales_offers_after_T'], 4)
save_json(summary, 'joint_summary')
print(json.dumps(summary, indent=1))
print('customer_scores.csv:', round((OUTPUTS / 'customer_scores.csv').stat().st_size / 1e6, 1), 'MB;',
      'parquet:', round((OUTPUTS / 'customer_scores.parquet').stat().st_size / 1e6, 1), 'MB')


# %% [markdown]
# # 4. Figures
# Reads only `tables/` (written by sections 0-3), so every chart traces to a saved CSV/JSON.
# Each title states the insight; the subtitle says what is plotted; the footer gives the source.
# 16:9 PNGs in `figures/`, numbered in story order.
# %%
import textwrap

import numpy as np
import pandas as pd


T = lambda name: pd.read_csv(TABLES / f'{name}.csv')
J = load_json
SRC = 'Source: BI10 Round 2 data (contract, call, leads, past behaviour, customer); team analysis'
fhs_m, ces_m, eda, joint = J('fhs_validation_metrics'), J('ces_validation_metrics'), J('eda_summary'), J('joint_summary')


def hgrid(ax):
    ax.grid(axis='x'); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(True)


def vgrid(ax):
    ax.grid(axis='y')


def strip_code(b: str) -> str:
    b = str(b)
    return b.split(' ', 1)[1] if b[:1].isdigit() and ' ' in b and not b[1:2].isdigit() else b

# %% [markdown]
# ## 4.1 Business understanding and data
# %%
m = T('eda_monthly_applications')
m['month'] = pd.to_datetime(m.month)
c = eda['cohort']
fig, ax = frame(f"The data is the Tet-2025 applicant cohort: {c['peak_multiple']:.0f}x normal volume, {c['first_time_share']:.0%} first-time applicants",
                f"Credit applications per month (unique credit cases). All {c['customers']:,} customers applied in Dec-2024 or Jan-2025.",
                SRC + ' | applications = contract cases by APPLY_CONTRACT_TIME')
tet = m.month.isin(pd.to_datetime(['2024-12-01', '2025-01-01']))
ax.bar(m.month, m.applications, width=24, color=np.where(tet, ACCENT, MUTED))
vgrid(ax); count_axis(ax)
for _, r in m[tet].iterrows():
    ax.text(r.month, r.applications * 1.015, f'{r.applications / 1e3:,.0f}k', ha='center', va='bottom', fontsize=11, color=INK)
ax.text(pd.Timestamp('2024-08-15'), 75_000, f"Jun-Nov 2024: {c['monthly_applications_jun_nov_2024'] / 1e3:.0f}k / month",
        ha='center', color=INK_2, fontsize=10.5)
ax.set_ylabel('Applications')
finish(fig, FIGURES / '01_tet_cohort.png')

tr = T('eda_data_traps').sort_values('share')
COUNT_WORD = {5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}
fig, ax = frame(f'{COUNT_WORD.get(len(tr), len(tr))} data traps were fixed before any score was built',
                'Share of the affected table touched by each problem, and the fix applied in Phase 1',
                SRC + ' | data_model/audit_summary.json')
bars = ax.barh(tr.trap, tr.share, color=ACCENT, height=.6)
ax.set_xlim(0, 1.9); hgrid(ax); pct_axis(ax, 'x')
for b, (_, r) in zip(bars, tr.iterrows()):
    share = f'{r.share:.1%}' if r.share >= .001 else '<0.1%'
    ax.text(b.get_width() + .015, b.get_y() + b.get_height() / 2, f"{share}  ·  {textwrap.shorten(r.fix, 50, placeholder='...')}",
            va='center', fontsize=9.5, color=INK_2)
ax.set_xticks([0, .25, .5, .75, 1]); ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Share of records affected')
finish(fig, FIGURES / '02_data_traps.png')

cz = T('eda_right_censoring')
cz['sign_month'] = pd.to_datetime(cz.sign_month)
cz = cz[cz.sign_month < '2025-06-01']
fig, ax = frame('Missed payments jumped for loans signed around Tet - and loans signed in 2025 only look clean because their instalments are not due yet',
                'Share of instalment loans with a missed payment on instalment 1 to 4, by month of signing',
                SRC + ' | FIRST_DUE..FOURTH_DUE on CD/TW/CL/CW contracts; extraction 2025-06-04')
for col, color, name in [('first_due', BLUE, 'Instalment 1'), ('second_due', ORANGE, 'Instalment 2'),
                         ('third_due', AQUA, 'Instalment 3'), ('fourth_due', YELLOW, 'Instalment 4')]:
    ax.plot(cz.sign_month, cz[col], color=color, label=name)
ax.axvspan(pd.Timestamp('2025-01-20'), cz.sign_month.max() + pd.Timedelta(days=20), color=GRID, alpha=.6, lw=0)
ax.text(pd.Timestamp('2025-01-25'), ax.get_ylim()[1] * .97, 'later instalments\nnot yet due', color=INK_2, fontsize=10.5, va='top')
vgrid(ax); pct_axis(ax, 'y', 1); ax.legend(loc='upper left', ncols=4)
ax.set_xlim(cz.sign_month.min() - pd.Timedelta(days=10), cz.sign_month.max() + pd.Timedelta(days=20))
ax.set_ylabel('Missed-payment rate')
finish(fig, FIGURES / '03_right_censoring.png')

rk = T('eda_early_stress_by_product')
rk = rk[rk.contracts >= 300]
prods = ['TW', 'CW', 'CD', 'BNPL', 'CL', 'CC']
w = eda['tet_wave_risk']
fig, axes = frame(f"First loans are {w['first_loan_stress'] / w['repeat_loan_stress']:.1f}x riskier in the Tet wave; cash-loan and card top-ups to existing clients got 1.6-2.4x riskier than before Tet",
                  'Early stress (missed instalment 1-4 or written off) by product, first loan vs repeat loan',
                  SRC + ' | contracts signed Jul-2024..Jan-2025; groups with <300 contracts hidden', ncols=2, sharey=True)
for ax, period in zip(axes, ['Before (Jul-24 to Nov-24)', 'Tet wave (Dec-24 to Jan-25)']):
    d = rk[rk.period == period]
    x = np.arange(len(prods))
    for i, (borrower, color) in enumerate([('Repeat loan', BLUE), ('First loan', ORANGE)]):
        v = d[d.borrower == borrower].set_index('product').early_stress_rate.reindex(prods)
        b = ax.bar(x + (i - .5) * .38, v.fillna(0), width=.36, color=color, label=borrower)
        for bb, val in zip(b, v):
            if pd.notna(val):
                ax.text(bb.get_x() + bb.get_width() / 2, val + .0015, f'{val:.1%}', ha='center', fontsize=9, color=INK_2)
    ax.set_xticks(x, ['Two-\nwheeler', 'Cash\n(walk-in)', 'Consumer\ndurables', 'BNPL', 'Cash\nloan', 'Credit\ncard'], fontsize=9.5)
    ax.set_title(period); vgrid(ax); pct_axis(ax, 'y')
axes[0].legend(loc='upper right'); axes[0].set_ylabel('Early-stress rate')
finish(fig, FIGURES / '04_tet_risk_by_product.png')

# %% [markdown]
# ## 4.2 Task 1 — Financial Health Score
# %%
pts = T('fhs_scorecard_points').query("model == 'history'")
wts = T('fhs_feature_weights').set_index('feature')
NAMES = {'repay_state': 'Repayment history', 'down_payment': 'Down payment', 'apps_90d': 'Applications, 90 days',
         'income': 'Income, VND/month', 'recency': 'Last loan signed', 'max_clean_credit': 'Largest clean loan, VND',
         'tenure': 'Relationship length'}
BIN_TEXT = {'Clean, no loan completed yet': 'Clean, none done', 'Clean, 1-2 loans completed': 'Clean, 1-2 done',
            'Clean, 3+ loans completed': 'Clean, 3+ done', 'Past due, not cured': 'Missed, not cured', 'Past due, cured': 'Missed, cured',
            'No goods/bike loan': 'No goods loan', 'None completed cleanly': 'None'}
order = list(wts.history_model_weight_pct.dropna().sort_values(ascending=False).index)
fig, axes = frame('Repayment history carries 55% of the score; a bigger down payment is the strongest affordability signal',
                  'Points each customer state adds (+) or removes (-) from the base, history scorecard. Panel title = share of the score range.',
                  SRC + ' | WoE logistic scorecard, train snapshots 2023-07 and 2024-01; tables/fhs_scorecard_points.csv',
                  ncols=4, nrows=2, figsize=(12.8, 7.2))
axes = axes.ravel()
for ax, f in zip(axes, order):
    d = pts[pts.feature == f].sort_values('bin')
    labels = [BIN_TEXT.get(strip_code(b), strip_code(b)) for b in d.bin]
    ax.barh(labels, d.points, color=np.where(d.points >= 0, BLUE, RED), height=.62)
    ax.axvline(0, color=AXIS, lw=.8); ax.invert_yaxis(); ax.grid(axis='x')
    ax.set_title(f"{NAMES[f]}  ·  {wts.loc[f, 'history_model_weight_pct']:.0f}%", fontsize=10.5)
    lim = max(abs(d.points).max(), 3) * 1.45
    ax.set_xlim(-lim, lim); ax.tick_params(axis='y', labelsize=8.8); ax.tick_params(axis='x', labelsize=8.5)
    for y, v in enumerate(d.points):
        ax.text(v + (lim * .04 if v >= 0 else -lim * .04), y, f'{round(v) + 0:+d}', va='center', ha='left' if v >= 0 else 'right', fontsize=8.5, color=INK_2)
axes[-1].axis('off')
axes[-1].text(0, .95, 'How to read', fontweight='bold', fontsize=11, va='top')
axes[-1].text(0, .80, 'FHS = base points + one value from\neach panel + live-status points,\nclipped to 0-100.\n\n'
              'Blue adds to health, red removes.\n2% stress probability = 60 points;\nodds of no stress x2 = +10 points.\n\n'
              'Excluded on evidence: live loans,\ninstalment-to-income, cash loan held\n(they look protective only through\nlender selection - see figure 08).',
              va='top', fontsize=9.5, color=INK_2)
finish(fig, FIGURES / '05_fhs_drivers.png')

bt = T('fhs_band_table').set_index('band').reindex(BAND_LABELS_ORDER)
fig, axes = frame(f"Each band steps up in risk: E is {bt.loc['E Stressed', 'stress_lift_vs_A']:.0f}x A out of time, and E is in collection 4 times in 10 a year later",
                  'Left: early-stress rate, 2024 out-of-time sample (structural score). Right: share in collection Jan-Aug 2026, customers called (final score).',
                  SRC + ' | tables/fhs_band_table.csv', ncols=2)
for ax, col, title in [(axes[0], 'stress_rate_oot', 'Early stress, Jul-2024..Jan-2025 (out of time)'),
                       (axes[1], 'future_collection_rate_validation_window', 'Collection contact, Jan..Aug 2026 (untouched window)')]:
    b = ax.bar(bt.index.str[0], bt[col], color=[BAND_COLORS[x] for x in bt.index], width=.62)
    ax.set_title(title); vgrid(ax); pct_axis(ax, 'y'); ax.set_ylim(0, bt[col].max() * 1.18)
    bar_labels(ax, b, [f'{v:.1%}' for v in bt[col]])
    ax.set_xticks(range(5), [f"{x}\n{bt.loc[x, 'share_production']:.0%} of base" for x in bt.index], fontsize=9.5)
finish(fig, FIGURES / '06_fhs_bands.png')

bm = [('Repayment history only', fhs_m['benchmark_repay_state_only']['auc']),
      ('FHS points scorecard (7 factors)', fhs_m['oot_history_only']['auc']),
      ('Gradient boosting, same inputs', fhs_m['benchmark_gbm_same_features_history']['auc'])]
fig, ax = frame('A 7-factor points scorecard matches gradient boosting out of time - complexity buys nothing here',
                'AUC on the 2024 out-of-time sample, customers with credit history (490k). 0.5 = random.',
                SRC + ' | tables/fhs_validation_metrics.json', figsize=(12.8, 5.4))
b = ax.barh([n for n, _ in bm], [v - .5 for _, v in bm], left=.5, color=[MUTED, ACCENT, MUTED], height=.5)
ax.set_xlim(.5, .65); hgrid(ax); ax.invert_yaxis()
for bb, (_, v) in zip(b, bm):
    ax.text(v + .002, bb.get_y() + bb.get_height() / 2, f'{v:.3f}', va='center', fontsize=11)
ax.set_xlabel('AUC (out of time)')
finish(fig, FIGURES / '07_fhs_benchmark.png')

px = T('fhs_selection_paradox')
px['n_active'] = px.n_active.astype(str)
fig, axes = frame('The lender-selection paradox: more live loans look safe on early repayment, then show up far more in collection',
                  'Customers with credit history, by number of loans live at the scoring date',
                  SRC + ' | left: 2024 out-of-time label; right: calls Jun-2025..Aug-2026 | tables/fhs_selection_paradox.csv', ncols=2)
for ax, col, title, color in [(axes[0], 'stress_rate_label_oot', 'Early stress on instalments 1-4 (what a naive model learns)', MUTED),
                              (axes[1], 'any_collection_after_T', 'In collection within 15 months (what happens later)', RED)]:
    b = ax.bar(px.n_active, px[col], color=color, width=.6)
    ax.set_title(title, fontsize=11.5); vgrid(ax); pct_axis(ax, 'y'); ax.set_ylim(0, px[col].max() * 1.2)
    bar_labels(ax, b, [f'{v:.1%}' for v in px[col]]); ax.set_xlabel('Live loans')
finish(fig, FIGURES / '08_selection_paradox.png')

ew = T('fhs_early_warning_validation')
ew['window'] = ew.window.str.split(':').str[0]
lp = fhs_m['live_layer']['live_points']
fig, ax = frame('Collection contact is persistent: one call means 38% odds of collection next window, 4+ calls 95% - the same a year later',
                'Share of called customers in collection in the next window, by collection contacts in the previous window',
                SRC + f" | fit Jan-Jun 2025 -> Jun-Dec 2025; validation Jul-Dec 2025 -> Jan-Aug 2026 | live points: {lp['ew']['1']:.0f} / {lp['ew']['2-3']:.0f} / {lp['ew']['4+']:.0f}")
x = np.arange(4)
for i, (win, color) in enumerate([('fit', BLUE), ('validation', ORANGE)]):
    d = ew[ew.window == win].set_index('ew').reindex(['0', '1', '2-3', '4+'])
    b = ax.bar(x + (i - .5) * .38, d.collection_rate, width=.36, color=color, label={'fit': 'Fit window (points set here)', 'validation': 'Validation window (untouched)'}[win])
    bar_labels(ax, b, [f'{v:.0%}' for v in d.collection_rate], size=10)
ax.set_xticks(x, ['No contact', '1 contact', '2-3 contacts', '4+ contacts']); vgrid(ax); pct_axis(ax, 'y')
ax.set_ylim(0, 1.1); ax.legend(loc='upper left'); ax.set_ylabel('In collection next window')
finish(fig, FIGURES / '09_live_status_validation.png')

sg = T('fhs_segment_summary').set_index('fhs_segment')
seg_order = ['Thriving', 'Healthy', 'Stable', 'Over-extended', 'New to credit', 'Fragile', 'Stressed', 'Live arrears', 'Distressed']
sg = sg.reindex(seg_order)
fig, axes = frame(f"{sg.loc['Live arrears', 'customers'] / 1e3:.1f}k customers are in live arrears: {sg.loc['Live arrears', 'collection_rate_after_T']:.0%} of them are in collection within 15 months",
                  'FHS segments at 2025-06-05: size, and share of called customers with a collection contact Jun-2025..Aug-2026',
                  SRC + ' | override order: Distressed > Live arrears > New to credit > Over-extended > band | tables/fhs_segment_summary.csv', ncols=2)
b = axes[0].barh(sg.index, sg.customers, color=MUTED, height=.6); hgrid(axes[0]); count_axis(axes[0], 'x'); axes[0].invert_yaxis()
axes[0].set_title('Customers'); axes[0].set_xlim(0, sg.customers.max() * 1.25)
bar_labels(axes[0], b, [f'{v / 1e3:,.0f}k  ({s:.1%})' for v, s in zip(sg.customers, sg.share)], horizontal=True, size=9.5)
colors = [RED if s in ('Live arrears', 'Distressed', 'Stressed', 'Fragile') else ACCENT for s in sg.index]
b = axes[1].barh(sg.index, sg.collection_rate_after_T, color=colors, height=.6); hgrid(axes[1]); pct_axis(axes[1], 'x'); axes[1].invert_yaxis()
axes[1].set_title('In collection after T (if called)'); axes[1].set_xlim(0, sg.collection_rate_after_T.max() * 1.2)
bar_labels(axes[1], b, [f'{v:.1%}' for v in sg.collection_rate_after_T], horizontal=True, size=9.5)
axes[1].set_yticklabels([])
finish(fig, FIGURES / '10_fhs_segments.png')

# %% [markdown]
# ## 4.3 Task 2 — Engagement
# %%
ini = T('eda_contact_initiator').pivot(index='source', columns='initiated_by', values='share').reindex(['Calls', 'Leads', 'Scoring events'])
ls = T('ces_lead_source_funnel').sort_values('deal_rate')
fig, axes = frame('Most contact is pushed by the company - and pushed leads almost never convert',
                  'Left: who started the record. Right: share of leads ending in a deal, by source; blue = customer-initiated source, label = volume.',
                  SRC + ' | customer-initiated: inbound calls; GMA app, landing page, SOB leads; non-telesales scoring', ncols=2)
ax = axes[0]
ax.barh(ini.index, ini['Company'], color=MUTED, height=.55, label='Company-initiated')
ax.barh(ini.index, ini['Customer'], left=ini['Company'] + .004, color=ACCENT, height=.55, label='Customer-initiated')
for y, (cmp, cus) in enumerate(zip(ini['Company'], ini['Customer'])):
    ax.text(cmp / 2, y, f'{cmp:.0%}', ha='center', va='center', color=INK, fontsize=10.5)
    ax.text(1.02, y, f'{cus:.0%} customer', va='center', fontsize=10, color=INK_2)
ax.set_xlim(0, 1.35); ax.invert_yaxis(); hgrid(ax); pct_axis(ax, 'x'); ax.set_xticks([0, .5, 1])
ax.legend(loc='upper center', bbox_to_anchor=(.45, -.07), ncols=2, fontsize=9.5)
ax.set_title('Who starts the contact')
ax = axes[1]
b = ax.barh(ls.LEAD_SOURCE, ls.deal_rate, color=np.where(ls.SELF_INITIATED, ACCENT, MUTED), height=.6)
hgrid(ax); pct_axis(ax, 'x'); ax.set_xlim(0, .55)
bar_labels(ax, b, [f'{r:.1%}  ({n / 1e3:,.0f}k leads)' for r, n in zip(ls.deal_rate, ls.leads)], horizontal=True, size=9.5)
ax.set_title('Deal rate by lead source')
finish(fig, FIGURES / '11_who_initiates.png')

fd = T('ces_future_deciles')
lift = fd.converted.iloc[-1] / fd.converted.iloc[0]
fig, axes = frame(f'Engagement predicts sales: the top decile converts {lift:.0f}x the bottom decile on calls a year later',
                  f"Customers called Jun-2025..Aug-2026 ({ces_m['future_calls']['called_customers'] / 1e3:,.0f}k), by CES decile at 2025-06-05 (10 = most engaged)",
                  SRC + f" | AUC interested {ces_m['future_calls']['interested']['auc']:.2f}, converted {ces_m['future_calls']['converted']['auc']:.2f}; outcomes never used to build CES", ncols=2)
for ax, col, title in [(axes[0], 'interested', 'Showed interest or signed on a call'), (axes[1], 'converted', 'Signed / made an application on a call')]:
    b = ax.bar(fd.decile.astype(str), fd[col], color=[ACCENT if d == 10 else MUTED for d in fd.decile], width=.65)
    ax.set_title(title); vgrid(ax); pct_axis(ax, 'y', 1); ax.set_ylim(0, fd[col].max() * 1.18); ax.set_xlabel('CES decile')
    bar_labels(ax, [b[0], b[-1]], [f'{fd[col].iloc[0]:.2%}', f'{fd[col].iloc[-1]:.2%}'])
finish(fig, FIGURES / '12_ces_validation.png')

at = T('joint_sales_call_allocation_by_tier')
fig, ax = frame(f"Passive customers absorb {at.share_of_sales_calls.iloc[0]:.0%} of delivered sales offers but bring {at.share_of_conversions.iloc[0]:.0%} of sales; the most engaged bring {at.share_of_conversions.iloc[-1]:.0%}",
                'Sales offers the customer heard and answered (signed, declined, not eligible), Jun-2025..Aug-2026, and the sales they produced, by CES tier at 2025-06-05',
                SRC + ' | tables/joint_sales_call_allocation_by_tier.csv')
x = np.arange(len(at))
for i, (col, color, name) in enumerate([('share_of_sales_calls', MUTED, 'Share of delivered offers'), ('share_of_conversions', ACCENT, 'Share of sales')]):
    b = ax.bar(x + (i - .5) * .38, at[col], width=.36, color=color, label=name)
    bar_labels(ax, b, [f'{v:.0%}' for v in at[col]])
ax.set_xticks(x, [f"{t}\n{r:.0f} sales per 1,000 offers" for t, r in zip(at.ces_tier, at.conversions_per_1000_sales_calls)])
vgrid(ax); pct_axis(ax, 'y'); ax.set_ylim(0, .7); ax.legend(loc='upper left')
finish(fig, FIGURES / '13_telesales_reallocation.png')

fat = T('ces_attempt_fatigue').query('not reached_before')
hr = T('ces_rpc_by_hour')
ops = ces_m['call_operations']
fig, axes = frame(f"{ops['share_of_calls_wasted']:.0%} of dials go to people who already ignored 10+ calls - they answer {ops['rpc_rate_after_10_failures']:.1%} of the time; mornings and 5pm work best",
                  'Right-party contact rate on outbound calls. Left: by attempt number, customers never reached so far. Right: by hour of day.',
                  SRC + f" | capping at 10 failed attempts loses {ops['rpc_lost_if_capped_at_10'] / 1e3:.1f}k contacts out of ~{ops['outbound_calls'] / 1e6:.1f}M dials | tables/ces_attempt_fatigue.csv", ncols=2)
ax = axes[0]
ax.plot(fat.prior_attempts + 1, fat.rpc_rate, color=ACCENT, marker='o', ms=5)
ax.axvspan(10.5, 16.5, color=GRID, alpha=.7, lw=0); ax.text(11, fat.rpc_rate.max() * .9, 'after 10\nfailures', color=INK_2, fontsize=10)
ax.set_xlabel('Attempt number (16 = 16th or later)'); vgrid(ax); pct_axis(ax, 'y'); ax.set_ylim(0, fat.rpc_rate.max() * 1.15)
ax.text(1.3, fat.rpc_rate.iloc[0], f'{fat.rpc_rate.iloc[0]:.1%}', fontsize=10, va='bottom')
ax.set_title('Never-reached customers, by attempt')
ax = axes[1]
best = hr.rpc_rate.nlargest(3).index
b = ax.bar(hr.hour.astype(str), hr.rpc_rate, color=[ACCENT if i in best else MUTED for i in hr.index], width=.7)
vgrid(ax); pct_axis(ax, 'y'); ax.set_xlabel('Hour of call'); ax.set_title('By hour of day (hours with 5k+ calls)')
bar_labels(ax, [b[i] for i in best], [f'{hr.rpc_rate[i]:.1%}' for i in best], size=9.5)
finish(fig, FIGURES / '14_call_operations.png')

dr = T('ces_drivers')
keep = {('Last touch type', 'interest on a sales call'): 'Last touch: showed interest on a sales call',
        ('Reached by phone before T (2025)', 'yes'): 'Answered a call in 2025',
        ('Last touch type', 'inbound call'): 'Last touch: called in themselves',
        ('First lead source', 'TLS'): 'First lead from telesales',
        ('First lead source', 'LANDING PAGE'): 'First lead from landing page',
        ('Last customer touch', '<=30 days'): 'Touched in the last 30 days',
        ('Product families touched (12m)', '3+'): 'Touched 3+ product families',
        ('Uses the mobile app (2025)', 'yes'): 'Uses the mobile app',
        ('Last customer touch', '91-365 days'): 'Silent 91-365 days',
        ('Repayment history', 'Past due, not cured'): 'Unresolved missed payment',
        ('Reached by phone before T (2025)', 'called, never reached'): 'Called in 2025, never reached',
        ('Reached by phone before T (2025)', 'not called'): 'Not called in 2025'}
dd = pd.DataFrame([{'label': v, 'lift': dr[(dr.factor == f) & (dr.level == l)].interest_lift.iloc[0],
                    'n': dr[(dr.factor == f) & (dr.level == l)].customers.iloc[0]} for (f, l), v in keep.items()]).sort_values('lift')
fig, ax = frame('What drives engagement: a past positive conversation (17x) and being reachable (3.5x); app use alone does not',
                'Interest rate on calls after T relative to the average called customer (1.0 = average), log scale',
                SRC + ' | descriptive association, not causation | tables/ces_drivers.csv')
b = ax.barh(dd.label, np.log2(dd.lift), color=np.where(dd.lift >= 1, BLUE, RED), height=.6)
ax.axvline(0, color=AXIS, lw=.8); hgrid(ax)
ticks = [.25, .5, 1, 2, 4, 8, 16]
ax.set_xticks(np.log2(ticks), [f'{t:g}x' for t in ticks]); ax.set_xlim(np.log2(.2), np.log2(40))
for bb, (_, r) in zip(b, dd.iterrows()):
    v = np.log2(r.lift)
    ax.text(v + (.08 if v >= 0 else -.08), bb.get_y() + bb.get_height() / 2, f'{r.lift:.1f}x  ({r.n / 1e3:,.0f}k)', va='center',
            ha='left' if v >= 0 else 'right', fontsize=9.5, color=INK_2)
finish(fig, FIGURES / '15_ces_drivers.png')

# %% [markdown]
# ## 4.4 Joint view — the canvas for Challenge 3
# %%
g = T('joint_band_tier_grid')
fig, axes = frame(f"Health and engagement are different things (rank correlation {joint['fhs_ces_rank_correlation']:.2f}): the grid shows where to sell and where to protect",
                  'FHS band x CES tier. Cell = customers; colour = outcome for customers called Jun-2025..Aug-2026.',
                  SRC + ' | tables/joint_band_tier_grid.csv', ncols=2)
for ax, col, title, cmap in [(axes[0], 'converted_if_called', 'Signed / applied on a call (sell here)', 'Blues'),
                             (axes[1], 'collection_if_called', 'In collection (protect here)', 'Reds')]:
    mat = g.pivot(index='fhs_band', columns='ces_tier', values=col).reindex(index=BAND_LABELS_ORDER, columns=TIER_ORDER)
    cnt = g.pivot(index='fhs_band', columns='ces_tier', values='customers').reindex(index=BAND_LABELS_ORDER, columns=TIER_ORDER)
    im = ax.imshow(mat.to_numpy(), cmap=cmap, aspect='auto', vmin=0)
    for i in range(5):
        for k in range(4):
            v = mat.iloc[i, k]
            dark = v > np.nanmax(mat.to_numpy()) * .55
            ax.text(k, i - .12, f'{v:.1%}', ha='center', va='center', fontsize=10.5, color='white' if dark else INK, fontweight='bold')
            ax.text(k, i + .22, f'{cnt.iloc[i, k] / 1e3:,.0f}k cust.', ha='center', va='center', fontsize=8.5, color='white' if dark else INK_2)
    ax.set_xticks(range(4), TIER_ORDER, fontsize=9.5); ax.set_yticks(range(5), BAND_LABELS_ORDER, fontsize=9.5)
    ax.set_title(title); ax.spines['bottom'].set_visible(False); ax.grid(False)
    ax.set_xlabel('Engagement tier (CES)')
axes[0].set_ylabel('Financial health band (FHS)')
finish(fig, FIGURES / '16_health_engagement_grid.png')

al = T('joint_sales_call_allocation').set_index('fhs_segment').reindex(seg_order)
fig, ax = frame('Financial stress does not stop sales offers: over-extended customers are offered credit most often, customers in collection as often as healthy ones',
                'Share of customers who heard and answered a product offer on a call, Jun-2025..Aug-2026, by FHS segment at 2025-06-05. Label = sales made.',
                SRC + ' | offer = call outcome signed / declined / not eligible')
flag = {'Over-extended', 'Live arrears'}
b = ax.bar(al.index, al.share_of_customers_offered, color=[RED if s in flag else MUTED for s in al.index], width=.62)
bar_labels(ax, b, [f"{v:.1%}\n{int(n):,} sales" for v, n in zip(al.share_of_customers_offered, al.conversions)], size=9.5)
vgrid(ax); pct_axis(ax, 'y'); ax.set_ylim(0, al.share_of_customers_offered.max() * 1.25)
ax.set_ylabel('Customers offered credit'); ax.tick_params(axis='x', labelsize=10)
finish(fig, FIGURES / '17_offers_to_stressed.png')
print('figures:', sorted(p.name for p in FIGURES.glob('*.png')))


# %% [markdown]
# # 5. Written report
# Builds `reports/ANALYTICAL_REPORT.md` and `reports/EXECUTIVE_SUMMARY.md` from `tables/` only, so every
# number in the text is the number the code produced.
# %%
import pandas as pd


T = lambda name: pd.read_csv(TABLES / f'{name}.csv')
eda, fhs, ces, joint = load_json('eda_summary'), load_json('fhs_validation_metrics'), load_json('ces_validation_metrics'), load_json('joint_summary')
audit = json.loads((MODEL / 'audit_summary.json').read_text(encoding='utf-8'))
pct = lambda v, d=1: f'{v:.{d}%}'


def md_table(df: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    cells = [[formats.get(c, str)(v) if pd.notna(v) else '' for c, v in row.items()] for _, row in df.iterrows()]
    head = '| ' + ' | '.join(df.columns) + ' |\n|' + '|'.join('---' for _ in df.columns) + '|\n'
    return head + '\n'.join('| ' + ' | '.join(r) + ' |' for r in cells)


bt = T('fhs_band_table').set_index('band').reindex(BAND_LABELS_ORDER)
seg = T('fhs_segment_summary')
iv = T('fhs_information_value')
w = T('fhs_feature_weights')
cseg = T('ces_segment_summary')
at = T('joint_sales_call_allocation_by_tier').set_index('ces_tier')
al = T('joint_sales_call_allocation').set_index('fhs_segment')
traps = T('eda_data_traps')
fat = T('ces_attempt_fatigue')
total_rpc = (fat.calls * fat.rpc_rate).sum()
ops = ces['call_operations']
live = fhs['live_layer']
lead = T('ces_lead_source_funnel').set_index('LEAD_SOURCE')
c = eda['cohort']

# %% [markdown]
# ## 5.1 Executive summary
# %%
summary = f"""# BI10 Round 2 — Financial Health and Engagement: executive summary

**Who is in the data.** {c['customers']:,} customers, every one of whom applied for credit in Dec-2024/Jan-2025:
the Tet-2025 applicant cohort ({c['peak_multiple']:.0f}x normal monthly volume, {pct(c['first_time_share'], 0)} first-time applicants).
Scores are taken at **T = 2025-06-05**, the day after the last contract record; calls continue to Aug-2026 and
are kept as an untouched future window for validation.

**Financial Health Score (FHS, 0-100).** Points scorecard on 7 credit-history factors plus a live-status layer.
* Out of time (2024 data, never used for fitting): band E is {bt.loc['E Stressed', 'stress_lift_vs_A']:.0f}x as stressed as band A
  ({pct(bt.loc['E Stressed', 'stress_rate_oot'])} vs {pct(bt.loc['A Thriving', 'stress_rate_oot'])}); AUC {fhs['oot']['auc']:.3f}.
  Gradient boosting on the same inputs: {fhs['benchmark_gbm_same_features_history']['auc']:.3f} vs scorecard {fhs['oot_history_only']['auc']:.3f} - no gain from complexity.
* A year later (Jan-Aug 2026, untouched): {pct(bt.loc['E Stressed', 'future_collection_rate_validation_window'], 0)} of band E and
  {pct(bt.loc['D Fragile', 'future_collection_rate_validation_window'], 0)} of band D are in collection, vs at most
  {pct(bt.loc[['A Thriving', 'B Healthy', 'C Stable'], 'future_collection_rate_validation_window'].max())} in A-C.
* {seg.set_index('fhs_segment').loc['Live arrears', 'customers']:,.0f} customers are in **live arrears** today; {pct(seg.set_index('fhs_segment').loc['Live arrears', 'collection_rate_after_T'], 0)} of them are in collection within 15 months.

**Customer Engagement Score (CES, 0-100).** Customer-initiated activity only (97% of calls and 97% of leads are company-pushed).
* On calls after T the top CES decile converts {T('ces_future_deciles').converted.iloc[-1] / T('ces_future_deciles').converted.iloc[0]:.0f}x the bottom decile
  (AUC {ces['future_calls']['converted']['auc']:.2f} for a sale, {ces['future_calls']['interested']['auc']:.2f} for interest).

**What the business should change now**
1. **Re-target telesales by CES.** Passive customers absorb {pct(at.loc['Passive', 'share_of_sales_calls'], 0)} of delivered offers and bring
   {pct(at.loc['Passive', 'share_of_conversions'], 0)} of sales; highly engaged customers bring {pct(at.loc['Highly engaged', 'share_of_conversions'], 0)}
   ({at.loc['Highly engaged', 'conversions_per_1000_sales_calls']:.0f} vs {at.loc['Passive', 'conversions_per_1000_sales_calls']:.0f} sales per 1,000 offers).
2. **Cap dialling.** {pct(ops['share_of_calls_wasted'], 0)} of dials go to people who ignored 10+ calls ({pct(ops['rpc_rate_after_10_failures'])} answer).
   A 10-attempt cap saves {ops['calls_to_never_reached_after_10_failures'] / 1e3:,.0f}k dials and loses {pct(ops['rpc_lost_if_capped_at_10'] / total_rpc)} of contacts.
   Call 7-10am and 5pm.
3. **Responsible-lending guardrail.** Over-extended customers are the most-offered segment ({pct(al.loc['Over-extended', 'share_of_customers_offered'])} offered);
   customers already in collection are offered credit as often as healthy ones ({pct(al.loc['Live arrears', 'share_of_customers_offered'])} vs
   {pct(al.loc['Healthy', 'share_of_customers_offered'])}) and signed {int(al.loc['Live arrears', 'conversions'])} new loans. Block offers on live arrears; require an
   affordability check for over-extended customers.
4. **Tet underwriting.** First loans in the Tet wave were {eda['tet_wave_risk']['first_loan_stress'] / eda['tet_wave_risk']['repeat_loan_stress']:.1f}x as risky as repeat loans;
   {pct(eda['first_loans_in_wave']['no_prior_footprint_share'], 0)} of first-time borrowers had no prior footprint. Down payment is the strongest affordability signal in the score.
5. **Lead sourcing.** RA leads are {lead.loc['RA', 'leads'] / lead.leads.sum():.0%} of all leads and convert {pct(lead.loc['RA', 'deal_rate'])};
   SOB and landing-page leads convert {pct(lead.loc['SOB', 'deal_rate'], 0)} and {pct(lead.loc['LANDING PAGE', 'deal_rate'], 0)}.
"""
(REPORTS / 'EXECUTIVE_SUMMARY.md').write_text(summary, encoding='utf-8')

# %% [markdown]
# ## 5.2 Full analytical report
# %%
fig = lambda n, cap: f'![{cap}](../figures/{n})\n*Figure: {cap}. Source: BI10 Round 2 data, team analysis.*\n'
weights_tab = w.dropna(subset=['history_model_weight_pct'])[['feature', 'pillar', 'history_model_weight_pct']].sort_values('history_model_weight_pct', ascending=False)
iv_tab = iv[iv.population == 'history'][['feature', 'pillar', 'train', 'oot', 'status']].sort_values('train', ascending=False)
iv_tab['status'] = iv_tab.status.str.replace('EXCLUDED: ', 'excluded - ', regex=False)
band_tab = bt.reset_index()[['band', 'score_min', 'score_max', 'stress_rate_oot', 'stress_lift_vs_A', 'future_collection_rate_validation_window', 'share_production']]
seg_tab = seg[['fhs_segment', 'customers', 'share', 'avg_score', 'expected_stress', 'collection_rate_after_T']].sort_values('avg_score', ascending=False)
cseg_tab = cseg[['ces_segment', 'customers', 'share', 'avg_score', 'median_days_since_touch', 'interested_after_T', 'converted_after_T']].sort_values('avg_score', ascending=False)
fair = T('fhs_fairness_check')
report = f"""# BI10 Round 2 — Customer Financial Health and Engagement

{summary.split(chr(10), 2)[2]}

---

## 1. Business understanding and data

{fig('01_tet_cohort.png', 'Monthly credit applications; the Tet-2025 wave defines the file')}

* The file is an **applicant cohort**: all {c['customers']:,} customers applied in Dec-2024/Jan-2025; {pct(c['first_time_share'], 0)} applied for the
  first time. Consequence: an outcome like "applies after Dec-2024" is true by construction, so it is never used for validation.
* Contracts, leads and behaviour end on 2025-06-04 (scoring date T = 2025-06-05). Calls run Jan-2025 to Aug-2026: the
  15 months after T are the independent future window.

### Data traps fixed in Phase 1

{fig('02_data_traps.png', 'Data problems, their size and the fix')}

{md_table(traps[['trap', 'affected', 'share', 'fix', 'impact_if_ignored']], {'affected': lambda v: f'{int(v):,}', 'share': lambda v: pct(v)})}

The previous version of this pipeline treated the income placeholder as a genuine conflict and lost the income of 306k customers;
it is fixed. Every source file is checked against the SHA-256 of the distributed file before it is read.

### Right-censoring and the Tet risk

{fig('03_right_censoring.png', 'Missed-payment flags by month of signing')}
{fig('04_tet_risk_by_product.png', 'Early stress by product, first vs repeat loan')}

* Missed-payment flags exist only for instalments 1-4, and only once each instalment has fallen due. Every feature is therefore
  computed *as of* a date with a seasoning rule (instalment k counts only if the loan was signed k+1 months earlier).
* Loans signed Oct-2024 to Jan-2025 show 3-6x the usual missed-payment rate. First loans in the wave: {pct(eda['tet_wave_risk']['first_loan_stress'])} early stress vs
  {pct(eda['tet_wave_risk']['repeat_loan_stress'])} for repeat loans; first-time two-wheeler loans {pct(T('eda_early_stress_by_product').query("period.str.startswith('Tet') and product == 'TW' and borrower == 'First loan'").early_stress_rate.iloc[0])}.

---

## 2. Task 1 — Financial Health Score

### Method

| Step | Choice | Why |
|---|---|---|
| Target | Stress after a date: a new loan misses instalment 1-4 or is written off; an existing loan misses an instalment falling due after the date; a young loan is written off | The only repayment outcome in the data; covers new and existing exposure |
| Time design | Train on snapshots 2023-07-01 and 2024-01-01 (6-month windows); test out of time on 2024-07-01 (to Jan-2025) | No future information; the test includes the Tet wave |
| Model | Weight-of-evidence bins + logistic regression, points-to-double-odds scale (2% stress = 60 pts, odds x2 = +10) | Every point is readable; ML is not required by the brief |
| Thin file | Separate scorecard for the {seg.set_index('fhs_segment').loc['New to credit', 'customers']:,.0f} customers with no signed contract; flagged *provisional* | {pct(eda['first_loans_in_wave']['no_prior_footprint_share'], 0)} of first-time borrowers have no prior footprint at all |
| Live status | Points for collection contact Jan-Jun 2025 and an affordability flag, fitted on calls Jun-Dec 2025 and validated on calls Jan-Aug 2026 | The contract table cannot see arrears after instalment 4 |
| Demographics | Not inputs; used only in a fairness check | Brief: no demographic risk factors without justification |

### Drivers (what raises and lowers financial health)

{fig('05_fhs_drivers.png', 'Points per factor state, history scorecard')}

{md_table(weights_tab.rename(columns={'history_model_weight_pct': 'share of score range %'}), {'share of score range %': lambda v: f'{v:.1f}'})}

Information Value per candidate (train vs out of time). Features whose signal flips over time or only reflects the lender's own
selection were excluded and stay listed with the reason:

{md_table(iv_tab, {'train': lambda v: f'{v:.3f}', 'oot': lambda v: f'{v:.3f}'})}

### Validation

{md_table(pd.DataFrame([
    {'check': 'Train, in time', **{k: fhs['train_in_time'][k] for k in ['n', 'event_rate', 'auc', 'gini', 'ks']}},
    {'check': 'Out of time, all', **{k: fhs['oot'][k] for k in ['n', 'event_rate', 'auc', 'gini', 'ks']}},
    {'check': 'Out of time, credit history', **{k: fhs['oot_history_only'][k] for k in ['n', 'event_rate', 'auc', 'gini', 'ks']}},
    {'check': 'Out of time, thin file', **{k: fhs['oot_thin_only'][k] for k in ['n', 'event_rate', 'auc', 'gini', 'ks']}},
    {'check': 'Benchmark: repayment history only', **{k: fhs['benchmark_repay_state_only'][k] for k in ['n', 'event_rate', 'auc', 'gini', 'ks']}},
    {'check': 'Benchmark: gradient boosting, same inputs', **{k: fhs['benchmark_gbm_same_features_history'][k] for k in ['n', 'event_rate', 'auc', 'gini', 'ks']}},
]), {'n': lambda v: f'{int(v):,}'})}

* Weight perturbation (every factor x0.75-1.25, 200 draws): rank correlation >= {fhs['weight_perturbation_25pct_history']['rank_corr_min']}; AUC {fhs['weight_perturbation_25pct_history']['auc_min']}-{fhs['weight_perturbation_25pct_history']['auc_max']}.
* PSI train vs out of time {fhs['psi']['train_vs_oot']}; train vs production (customers with a live loan) {fhs['psi']['train_vs_production_live_exposure_structural']} (stable < 0.1).

{fig('07_fhs_benchmark.png', 'Scorecard vs benchmarks out of time')}

### The lender-selection paradox and the live-status layer

{fig('08_selection_paradox.png', 'Early stress vs later collection by number of live loans')}
{fig('09_live_status_validation.png', 'Collection persistence, fit vs validation window')}

* On the early-delinquency label, customers with 3+ live loans look *safer* ({pct(T('fhs_selection_paradox').set_index('n_active').loc['3+', 'stress_rate_label_oot'])}),
  because the lender only tops up clients it trusts. After T they are in collection {pct(T('fhs_selection_paradox').set_index('n_active').loc['3+', 'any_collection_after_T'])} of the time vs
  {pct(T('fhs_selection_paradox').set_index('n_active').loc['1', 'any_collection_after_T'])} with one loan. Live loans and instalment-to-income are therefore not rewarded by the score.
* Live points (fit window): 1 collection contact {live['live_points']['ew']['1']}, 2-3 contacts {live['live_points']['ew']['2-3']}, 4+ contacts {live['live_points']['ew']['4+']},
  affordability flag {live['live_points']['affordability']}. On the untouched window the final FHS reaches AUC {live['validation_window']['final_fhs']} for later collection
  (structural part alone {live['validation_window']['structural_only']}). The affordability points cost almost nothing on the 2024 label
  (AUC {fhs['affordability_tradeoff_on_oot_label']['structural']} -> {fhs['affordability_tradeoff_on_oot_label']['structural_plus_affordability_points']}).

### Bands and segments

{fig('06_fhs_bands.png', 'Stress by band, out of time and one year later')}

{md_table(band_tab, {'score_min': lambda v: f'{v:.0f}', 'score_max': lambda v: f'{v:.0f}', 'stress_rate_oot': pct, 'stress_lift_vs_A': lambda v: f'{v:.1f}x',
                     'future_collection_rate_validation_window': pct, 'share_production': pct})}

{fig('10_fhs_segments.png', 'FHS segments')}

{md_table(seg_tab, {'customers': lambda v: f'{int(v):,}', 'share': pct, 'avg_score': lambda v: f'{v:.1f}', 'expected_stress': pct, 'collection_rate_after_T': pct})}

### Fairness

{md_table(fair, {'n': lambda v: f'{int(v):,}', 'avg_score': lambda v: f'{v:.1f}', 'actual_stress_rate': pct, 'auc_within_group': lambda v: f'{v:.3f}'})}

Men and women get almost the same average score while men are more often stressed: gender is not used, by design. Under-25s
score lower and are ranked poorly (AUC ~0.52) because most are new to credit, not because of age.

---

## 3. Task 2 — Customer Engagement Score

{fig('11_who_initiates.png', 'Who starts the contact and how leads convert')}

### Method

Only customer-initiated touches count: self-service and in-store scoring, credit applications, leads the customer created or accepted,
inbound calls, and calls where the customer showed interest or signed. Components (weights): recency 35% (half-life 90 days),
frequency 25% (active days, 12 months), interaction 15% (answer rate on 2025 calls, shrunk to the population rate),
lead activity 10%, product interaction 15% (product families touched + live loan).

### Validation against business outcomes

{fig('12_ces_validation.png', 'Future interest and sales by CES decile')}

| Check | Result |
|---|---|
| Calls after T: showed interest or signed | AUC {ces['future_calls']['interested']['auc']} (call-volume adjusted {ces['future_calls']['interested']['auc_call_volume_adjusted']}) |
| Calls after T: signed / applied | AUC {ces['future_calls']['converted']['auc']} (adjusted {ces['future_calls']['converted']['auc_call_volume_adjusted']}) |
| Back-test: score at 2025-02-01, comes back and applies by June | AUC {ces['backtest']['applied_again']['auc']} |
| Back-test: score at 2025-02-01, signs again by June | AUC {ces['backtest']['signed_again']['auc']} |
| Logistic re-weighting on the back-test (in-sample upper bound) | AUC {ces['backtest']['benchmark_logistic_weights_in_sample']} |
| Weight sensitivity (equal weights / recency-heavier) | rank correlation {ces['sensitivity']['equal_weights']['rank_corr_with_primary']} / {ces['sensitivity']['recency_heavier']['rank_corr_with_primary']} |
| App usage 2025 (not an input) | users {pct(T('ces_app_usage_check').app_users.iloc[-1])} in CES 60+ vs {pct(T('ces_app_usage_check').app_users.iloc[-3])} in CES 20-40 |

### Drivers and segments

{fig('15_ces_drivers.png', 'Interest lift by observable factor')}

{md_table(cseg_tab, {'customers': lambda v: f'{int(v):,}', 'share': pct, 'avg_score': lambda v: f'{v:.1f}', 'median_days_since_touch': lambda v: f'{v:.0f}',
                      'interested_after_T': pct, 'converted_after_T': lambda v: pct(v, 2)})}

### Call operations

{fig('14_call_operations.png', 'Answer rate by attempt and by hour')}

---

## 4. Health x engagement: where to sell, where to protect

{fig('16_health_engagement_grid.png', 'FHS band x CES tier with future outcomes')}
{fig('13_telesales_reallocation.png', 'Delivered offers vs sales by CES tier')}
{fig('17_offers_to_stressed.png', 'Customers offered credit by FHS segment')}

* The two scores are almost independent (rank correlation {joint['fhs_ces_rank_correlation']}): engagement does not imply health.
* {joint['healthy_and_engaged_customers']:,} customers are healthy (A/B) and engaged; {joint['healthy_and_engaged_not_called_after_T']:,} of them were never called after T.
* Engaged customers in band E show interest on calls but end in collection most often - interest there signals need, not capacity.

This grid, with both scores, reasons and explanations per customer in `outputs/customer_scores.csv`, is the input for Challenge 3.

---

## 5. Limitations

* Repayment is observed only on instalments 1-4; later arrears are seen only through collection calls, which cover a
  company-selected list (written-off customers are almost absent).
* The file is the Tet-2025 applicant cohort, not the full book; customers who left before Dec-2024 are not represented.
* Thin-file scores are provisional (AUC {fhs['oot_thin_only']['auc']}); segment, not score, should drive decisions for them.
* Within bands A-C the score does not separate later collection ({pct(bt.loc['C Stable', 'future_collection_rate_validation_window'])}-{pct(bt.loc['A Thriving', 'future_collection_rate_validation_window'])}); its job there is early-stress ranking.
* All associations are descriptive. Recommendations should be rolled out as controlled tests (e.g. CES-ranked call list vs current list).
* App sessions are a 2025 average with no exact period; they are used only as a validation signal.
"""
(REPORTS / 'ANALYTICAL_REPORT.md').write_text(report, encoding='utf-8')
print('reports written:', [p.name for p in REPORTS.glob('*.md')])
