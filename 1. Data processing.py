# %% [markdown]
# # Phase 1 — Audit and build
# Check five CSVs; create one customer-level Parquet. Raw files stay unchanged.
# %%
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'BI10_ROUND02_DATASET'
OUTPUT = ROOT / 'data_clean.parquet'
TEMP = ROOT / 'data_clean.partial.parquet'
TABLES = ('CUSTOMER', 'CONTRACT', 'CALL', 'LEADS', 'PAST_BEHAVIOR')

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='2GB'")
for table in TABLES:
    path = (SOURCE / f'{table}.csv').as_posix().replace("'", "''")
    con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_csv('{path}', header=true, all_varchar=true, delim=',', quote='\"', escape='\"')")

# %% [markdown]
# ## Source checks
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
    checks = []
    for column in columns:
        quoted = '"' + column.replace('"', '""') + '"'
        checks.append(f"count(*) FILTER (WHERE {quoted} IS NULL OR trim({quoted}) = '')")
    missing = con.execute('SELECT ' + ', '.join(checks) + f' FROM {table}').fetchone()
    print(f'{table}: {rows:,} rows; {clients:,} customers; null/blank by column: {dict(zip(columns, missing))}', flush=True)

call_span = con.execute('SELECT min(try_cast(TIME_OF_CALL_START AS TIMESTAMP)), max(try_cast(TIME_OF_CALL_START AS TIMESTAMP)) FROM CALL').fetchone()
case_conflicts = con.execute('''SELECT count(*) FILTER (WHERE n>1), count(*) FILTER (WHERE incomes>1),
    count(*) FILTER (WHERE income_types>1), count(*) FILTER (WHERE professions>1)
    FROM (SELECT SKP_CREDIT_CASE, count(*) n, count(DISTINCT AMT_INCOME_MAIN) incomes,
    count(DISTINCT NAME_INCOME_TYPE) income_types, count(DISTINCT CODE_PROFESSION) professions
    FROM CONTRACT GROUP BY 1)''').fetchone()
call_key_duplicates = con.execute('''SELECT count(*) FROM (SELECT SKP_CLIENT,CALL_ID,TIME_OF_CALL_START
    FROM CALL GROUP BY 1,2,3 HAVING count(*)>1)''').fetchone()[0]
assert call_key_duplicates == 0, 'The call counting grain has changed'
print('Call date range:', call_span, '| repeated cases / income / type / profession conflicts:', case_conflicts, flush=True)
quality = {
    'household zero/positive, main income negative': '''SELECT
        count(*) FILTER (WHERE try_cast(AMT_INCOME_HOUSEHOLD AS DOUBLE)=0),
        count(*) FILTER (WHERE try_cast(AMT_INCOME_HOUSEHOLD AS DOUBLE)>0),
        count(*) FILTER (WHERE try_cast(AMT_INCOME_MAIN AS DOUBLE)<0) FROM CONTRACT''',
    'signed statuses without sign time': '''SELECT count(DISTINCT SKP_CREDIT_CASE)
        FROM CONTRACT WHERE NAME_CREDIT_STATUS IN
        ('Active','Finished','Paid off','Signed','Written off') AND SIGN_CONTRACT_TIME IS NULL''',
    'lead acceptance sentinel': "SELECT count(*) FROM LEADS WHERE DATE_ACCEPTED LIKE '3000%'",
    'unparseable activity timestamps': '''SELECT
        (SELECT count(*) FROM CALL WHERE try_cast(TIME_OF_CALL_START AS TIMESTAMP) IS NULL),
        (SELECT count(*) FROM LEADS WHERE try_cast(DTIME_CREATED AS TIMESTAMP) IS NULL),
        (SELECT count(*) FROM PAST_BEHAVIOR WHERE try_cast(DTIME_EVENT AS TIMESTAMP) IS NULL)''',
    'credit cases spanning customers': '''SELECT count(*) FROM
        (SELECT SKP_CREDIT_CASE FROM CONTRACT GROUP BY 1 HAVING count(DISTINCT SKP_CLIENT)>1)''',
}
for label, sql in quality.items():
    result = con.execute(sql).fetchone()
    print(label, result, flush=True)
    if label in ('unparseable activity timestamps', 'credit cases spanning customers'):
        assert not any(result), (label, result)

# %% [markdown]
# ## Contract cases
# Collapse repeated cases. Conflicting values stay null.
# %%
con.execute("""
CREATE TEMP TABLE cases AS
SELECT CAST(SKP_CLIENT AS BIGINT) SKP_CLIENT,
       CAST(SKP_CREDIT_CASE AS BIGINT) SKP_CREDIT_CASE,
       count(*) SOURCE_CONTRACT_ROWS,
       max(NAME_CREDIT_STATUS) STATUS,
       max(PRODUCT) PRODUCT,
       CASE WHEN count(DISTINCT NAME_INCOME_TYPE)>1 THEN NULL ELSE max(NAME_INCOME_TYPE) END INCOME_TYPE,
       CASE WHEN count(DISTINCT CODE_PROFESSION)>1 THEN NULL ELSE max(CODE_PROFESSION) END PROFESSION,
       (count(DISTINCT AMT_INCOME_MAIN)>1)::BIGINT INCOME_MAIN_CONFLICT,
       (count(DISTINCT AMT_INCOME_HOUSEHOLD)>1)::BIGINT INCOME_HOUSEHOLD_CONFLICT,
       (count(DISTINCT NAME_INCOME_TYPE)>1)::BIGINT INCOME_TYPE_CONFLICT,
       (count(DISTINCT CODE_PROFESSION)>1)::BIGINT PROFESSION_CONFLICT,
       max(try_cast(PAYMENT_NUM AS DOUBLE)) PAYMENT_NUM,
       max(try_cast(APPLY_CONTRACT_TIME AS TIMESTAMP)) APPLY_TIME,
       max(try_cast(SIGN_CONTRACT_TIME AS TIMESTAMP)) SIGN_TIME,
       max(try_cast(APPROVE_CONTRACT_TIME AS TIMESTAMP)) APPROVE_TIME,
       CASE WHEN count(DISTINCT TRANSAC)>1 THEN NULL ELSE max(TRANSAC) END TRANSAC,
       max(try_cast(INIT_PAY AS DOUBLE)) INITIAL_PAYMENT,
       max(try_cast(FLAG_INS AS BIGINT)) INSURED,
       CASE WHEN count(DISTINCT AMT_INCOME_MAIN)>1 THEN NULL ELSE max(try_cast(AMT_INCOME_MAIN AS DOUBLE)) END INCOME_MAIN,
       max(try_cast(AMT_INCOME_MAIN AS DOUBLE)) FILTER (WHERE try_cast(AMT_INCOME_MAIN AS DOUBLE)>0) INCOME_MAIN_POSITIVE,
       CASE WHEN count(DISTINCT AMT_INCOME_HOUSEHOLD)>1 THEN NULL ELSE max(try_cast(AMT_INCOME_HOUSEHOLD AS DOUBLE)) END INCOME_HOUSEHOLD,
       max(try_cast(AMT_INCOME_HOUSEHOLD AS DOUBLE)) FILTER (WHERE try_cast(AMT_INCOME_HOUSEHOLD AS DOUBLE)>0) INCOME_HOUSEHOLD_POSITIVE,
       max(try_cast(AMT_CREDIT AS DOUBLE)) CREDIT_AMOUNT,
       max(try_cast(ANNUITY AS DOUBLE)) ANNUITY_AMOUNT,
       max(try_cast(AMT_BILLING AS DOUBLE)) BILLING_AMOUNT,
       max(try_cast(FIRST_DUE AS BIGINT)) FIRST_DUE,
       max(try_cast(SECOND_DUE AS BIGINT)) SECOND_DUE,
       max(try_cast(THIRD_DUE AS BIGINT)) THIRD_DUE,
       max(try_cast(FOURTH_DUE AS BIGINT)) FOURTH_DUE
FROM CONTRACT GROUP BY 1,2
""")
print('Distinct credit cases:', con.execute('SELECT count(*) FROM cases').fetchone()[0], flush=True)

# %% [markdown]
# ## Contract features
# Keep active and historical arrears separate. Status has no event date.
# %%
con.execute("""
CREATE TEMP TABLE contracts_by_client AS
SELECT SKP_CLIENT,
       sum(SOURCE_CONTRACT_ROWS)::BIGINT CONTRACT_SOURCE_ROWS,
       count(*)::BIGINT CONTRACT_CASE_COUNT,
       sum(INCOME_MAIN_CONFLICT)::BIGINT CONTRACT_INCOME_MAIN_CONFLICT_CASE_COUNT,
       sum(INCOME_HOUSEHOLD_CONFLICT)::BIGINT CONTRACT_INCOME_HOUSEHOLD_CONFLICT_CASE_COUNT,
       sum(INCOME_TYPE_CONFLICT)::BIGINT CONTRACT_INCOME_TYPE_CONFLICT_CASE_COUNT,
       sum(PROFESSION_CONFLICT)::BIGINT CONTRACT_PROFESSION_CONFLICT_CASE_COUNT,
       count(*) FILTER (WHERE STATUS IN ('Active','Finished','Paid off','Signed','Written off'))::BIGINT CONTRACT_SIGNED_COUNT,
       count(*) FILTER (WHERE STATUS IN ('Active','Finished','Paid off','Signed','Written off') AND SIGN_TIME IS NULL)::BIGINT CONTRACT_SIGNED_MISSING_SIGN_TIME_COUNT,
       count(*) FILTER (WHERE STATUS = 'Active')::BIGINT CONTRACT_ACTIVE_COUNT,
       count(*) FILTER (WHERE STATUS = 'Written off')::BIGINT CONTRACT_WRITTEN_OFF_COUNT,
       count(*) FILTER (WHERE STATUS = 'Rejected')::BIGINT CONTRACT_REJECTED_COUNT,
       count(DISTINCT PRODUCT)::BIGINT CONTRACT_PRODUCT_COUNT,
       sum(CREDIT_AMOUNT) FILTER (WHERE STATUS IN ('Active','Finished','Paid off','Signed','Written off')) CONTRACT_SIGNED_CREDIT_TOTAL,
       sum(CREDIT_AMOUNT) FILTER (WHERE STATUS = 'Active') CONTRACT_ACTIVE_CREDIT_TOTAL,
       sum(INITIAL_PAYMENT) FILTER (WHERE STATUS IN ('Active','Finished','Paid off','Signed','Written off')) CONTRACT_SIGNED_INITIAL_PAYMENT_TOTAL,
       count(*) FILTER (WHERE INSURED=1)::BIGINT CONTRACT_INSURED_CASE_COUNT,
       sum(ANNUITY_AMOUNT) FILTER (WHERE STATUS = 'Active') CONTRACT_ACTIVE_ANNUITY_TOTAL,
       sum(BILLING_AMOUNT) FILTER (WHERE STATUS = 'Active') CONTRACT_ACTIVE_BILLING_TOTAL,
       sum(coalesce(FIRST_DUE,0))::BIGINT FIRST_DUE_TOTAL,
       sum(coalesce(SECOND_DUE,0))::BIGINT SECOND_DUE_TOTAL,
       sum(coalesce(THIRD_DUE,0))::BIGINT THIRD_DUE_TOTAL,
       sum(coalesce(FOURTH_DUE,0))::BIGINT FOURTH_DUE_TOTAL,
       sum(coalesce(FIRST_DUE,0)) FILTER (WHERE STATUS='Active')::BIGINT ACTIVE_FIRST_DUE_TOTAL,
       sum(coalesce(SECOND_DUE,0)) FILTER (WHERE STATUS='Active')::BIGINT ACTIVE_SECOND_DUE_TOTAL,
       sum(coalesce(THIRD_DUE,0)) FILTER (WHERE STATUS='Active')::BIGINT ACTIVE_THIRD_DUE_TOTAL,
       sum(coalesce(FOURTH_DUE,0)) FILTER (WHERE STATUS='Active')::BIGINT ACTIVE_FOURTH_DUE_TOTAL,
       max(APPLY_TIME) LAST_CONTRACT_APPLICATION_AT,
       max(APPROVE_TIME) LAST_CONTRACT_APPROVED_AT,
       max(SIGN_TIME) LAST_CONTRACT_SIGNED_AT,
       arg_max(TRANSAC, struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE))
          FILTER (WHERE TRANSAC IS NOT NULL) LATEST_RECORDED_CONTRACT_TRANSAC,
       arg_max(STATUS, struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE)) LATEST_CONTRACT_STATUS,
       arg_max(PRODUCT, struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE)) LATEST_CONTRACT_PRODUCT,
       arg_max(struct_pack(v := INCOME_TYPE), struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE)).v LATEST_INCOME_TYPE,
       arg_max(struct_pack(v := PROFESSION), struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE)).v LATEST_PROFESSION,
       arg_max(PAYMENT_NUM, struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE)) LATEST_PAYMENT_NUM,
       arg_max(struct_pack(v := INCOME_MAIN), struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE)).v LATEST_INCOME_MAIN,
       arg_max(INCOME_MAIN_POSITIVE, struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE))
          FILTER (WHERE INCOME_MAIN_POSITIVE IS NOT NULL) LATEST_POSITIVE_INCOME_MAIN,
       arg_max(struct_pack(v := INCOME_HOUSEHOLD), struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE)).v LATEST_INCOME_HOUSEHOLD_RAW,
       arg_max(INCOME_HOUSEHOLD_POSITIVE, struct_pack(t := APPLY_TIME, id := SKP_CREDIT_CASE))
          FILTER (WHERE INCOME_HOUSEHOLD_POSITIVE IS NOT NULL) LATEST_POSITIVE_INCOME_HOUSEHOLD
FROM cases GROUP BY SKP_CLIENT
""")

# %% [markdown]
# ## Call features
# (client, call ID, start) defines a call. Add June-2025 windows.
# %%
con.execute("""
CREATE TEMP TABLE calls_by_client AS
WITH parsed AS (
  SELECT CAST(SKP_CLIENT AS BIGINT) SKP_CLIENT,
         try_cast(TIME_OF_CALL_START AS TIMESTAMP) START_AT,
         try_cast(DURATIONS AS DOUBLE) DURATION_SECONDS,
         CALL_RESULT_SYSTEM, CALL_TYPE, CODE_DISPOSITION, PRODUCT_TYPE, CALL_SOURCE
  FROM CALL
)
SELECT SKP_CLIENT,
       count(*)::BIGINT CALL_SOURCE_ROWS,
       count(*)::BIGINT CALL_COUNT,
       count(*) FILTER (WHERE CALL_RESULT_SYSTEM='ANSWER')::BIGINT CALL_SYSTEM_ANSWER_COUNT,
       count(*) FILTER (WHERE CODE_DISPOSITION='Refuse')::BIGINT CALL_DISPOSITION_REFUSE_COUNT,
       count(*) FILTER (WHERE upper(CALL_TYPE)='INBOUND')::BIGINT CALL_INBOUND_COUNT,
       sum(DURATION_SECONDS) CALL_DURATION_SECONDS_TOTAL,
       max(START_AT) LAST_CALL_AT,
       count(DISTINCT PRODUCT_TYPE)::BIGINT CALL_PRODUCT_TYPE_COUNT,
       arg_max(struct_pack(v := PRODUCT_TYPE), START_AT).v LAST_CALL_PRODUCT_TYPE,
       arg_max(CALL_SOURCE, START_AT) LAST_CALL_SOURCE,
       arg_max(struct_pack(v := CALL_RESULT_SYSTEM), START_AT).v LAST_CALL_SYSTEM_RESULT,
       arg_max(CODE_DISPOSITION, START_AT) LAST_CALL_DISPOSITION,
       count(*) FILTER (WHERE START_AT < TIMESTAMP '2025-06-05')::BIGINT CALL_COUNT_ASOF_2025_06_04,
       count(*) FILTER (WHERE START_AT >= TIMESTAMP '2025-06-05' - INTERVAL '90 days'
                          AND START_AT < TIMESTAMP '2025-06-05')::BIGINT CALL_COUNT_90D_ASOF_2025_06_04,
       count(*) FILTER (WHERE CALL_RESULT_SYSTEM='ANSWER' AND START_AT < TIMESTAMP '2025-06-05')::BIGINT CALL_SYSTEM_ANSWER_ASOF_2025_06_04,
       count(*) FILTER (WHERE CODE_DISPOSITION='Refuse' AND START_AT < TIMESTAMP '2025-06-05')::BIGINT CALL_REFUSE_ASOF_2025_06_04,
       count(*) FILTER (WHERE upper(CALL_TYPE)='INBOUND' AND START_AT < TIMESTAMP '2025-06-05')::BIGINT CALL_INBOUND_ASOF_2025_06_04,
       count(DISTINCT PRODUCT_TYPE) FILTER (WHERE START_AT < TIMESTAMP '2025-06-05')::BIGINT CALL_PRODUCT_TYPES_ASOF_2025_06_04,
       sum(DURATION_SECONDS) FILTER (WHERE START_AT < TIMESTAMP '2025-06-05') CALL_DURATION_SECONDS_ASOF_2025_06_04,
       max(START_AT) FILTER (WHERE START_AT < TIMESTAMP '2025-06-05') LAST_CALL_ASOF_2025_06_04,
       arg_max(struct_pack(v := PRODUCT_TYPE), START_AT)
          FILTER (WHERE START_AT < TIMESTAMP '2025-06-05') .v LAST_CALL_PRODUCT_ASOF_2025_06_04,
       arg_max(struct_pack(v := CALL_RESULT_SYSTEM), START_AT)
          FILTER (WHERE START_AT < TIMESTAMP '2025-06-05') .v LAST_CALL_RESULT_ASOF_2025_06_04
FROM parsed GROUP BY 1
""")

# %% [markdown]
# ## Lead and behavior features
# Preserve source rows; flag the year-3000 lead sentinel.
# %%
con.execute("""
CREATE TEMP TABLE leads_by_client AS
SELECT CAST(SKP_CLIENT AS BIGINT) SKP_CLIENT,
       count(*)::BIGINT LEAD_COUNT,
       count(*) FILTER (WHERE DESC_FIN_REASON = 'Deal')::BIGINT LEAD_DEAL_COUNT,
       count(*) FILTER (WHERE DESC_FIN_REASON = 'Not Interest')::BIGINT LEAD_NOT_INTERESTED_COUNT,
       count(*) FILTER (WHERE FLAG_ASSIGNED = '1')::BIGINT LEAD_ASSIGNED_COUNT,
       count(*) FILTER (WHERE DATE_ACCEPTED LIKE '3000%')::BIGINT LEAD_ACCEPTED_SENTINEL_COUNT,
       max(try_cast(MAX_OFFER AS DOUBLE)) LEAD_MAX_OFFER,
       max(try_cast(DTIME_CREATED AS TIMESTAMP)) LAST_LEAD_AT,
       count(*) FILTER (WHERE try_cast(DTIME_CREATED AS TIMESTAMP) >= TIMESTAMP '2025-06-05' - INTERVAL '90 days'
                          AND try_cast(DTIME_CREATED AS TIMESTAMP) < TIMESTAMP '2025-06-05')::BIGINT LEAD_COUNT_90D_ASOF_2025_06_04,
       count(*) FILTER (WHERE DESC_FIN_REASON='Deal' AND try_cast(DTIME_CREATED AS TIMESTAMP) >= TIMESTAMP '2025-06-05' - INTERVAL '90 days'
                          AND try_cast(DTIME_CREATED AS TIMESTAMP) < TIMESTAMP '2025-06-05')::BIGINT LEAD_DEAL_90D_ASOF_2025_06_04,
       arg_max(CODE_PRODUCT_TYPE, struct_pack(t := try_cast(DTIME_CREATED AS TIMESTAMP), p := CODE_PRODUCT_TYPE)) LAST_LEAD_PRODUCT,
       arg_max(struct_pack(v := DESC_FIN_REASON), struct_pack(t := try_cast(DTIME_CREATED AS TIMESTAMP), p := coalesce(DESC_FIN_REASON,''))).v LAST_LEAD_REASON,
       arg_max(LEAD_SOURCE, struct_pack(t := try_cast(DTIME_CREATED AS TIMESTAMP), p := LEAD_SOURCE)) LAST_LEAD_SOURCE
FROM LEADS GROUP BY 1
""")
# Multiple behavior products can share the customer's latest timestamp.
# Preserve that ambiguity rather than selecting a plan-dependent "last" event.
con.execute("""
CREATE TEMP TABLE behavior_events AS
SELECT CAST(SKP_CLIENT AS BIGINT) SKP_CLIENT,
       try_cast(DTIME_EVENT AS TIMESTAMP) EVENT_AT, ACTION, PRODUCT_CODE
FROM PAST_BEHAVIOR
""")
con.execute("""
CREATE TEMP TABLE behavior_last_dates AS
SELECT SKP_CLIENT, max(EVENT_AT) LAST_BEHAVIOR_AT
FROM behavior_events GROUP BY 1
""")
con.execute("""
CREATE TEMP TABLE behavior_by_client AS
WITH latest AS (
  SELECT e.SKP_CLIENT,
         CASE WHEN count(DISTINCT e.ACTION) > 1 THEN NULL ELSE max(e.ACTION) END LAST_BEHAVIOR_ACTION,
         CASE WHEN count(DISTINCT e.PRODUCT_CODE) > 1 THEN NULL ELSE max(e.PRODUCT_CODE) END LAST_BEHAVIOR_PRODUCT,
         (count(DISTINCT e.ACTION)>1)::BIGINT LAST_BEHAVIOR_ACTION_AMBIGUOUS,
         (count(DISTINCT e.PRODUCT_CODE)>1)::BIGINT LAST_BEHAVIOR_PRODUCT_AMBIGUOUS
  FROM behavior_events e JOIN behavior_last_dates l
    ON e.SKP_CLIENT=l.SKP_CLIENT AND e.EVENT_AT=l.LAST_BEHAVIOR_AT
  GROUP BY 1
), historical AS (
  SELECT SKP_CLIENT,
       count(*)::BIGINT BEHAVIOR_EVENT_COUNT,
       count(*) FILTER (WHERE ACTION LIKE '%SCORING%')::BIGINT BEHAVIOR_SCORING_COUNT,
       count(*) FILTER (WHERE ACTION LIKE 'SOB%')::BIGINT BEHAVIOR_SOB_COUNT,
       count(DISTINCT PRODUCT_CODE)::BIGINT BEHAVIOR_PRODUCT_COUNT,
       count(*) FILTER (WHERE EVENT_AT >= TIMESTAMP '2025-06-05' - INTERVAL '90 days'
                          AND EVENT_AT < TIMESTAMP '2025-06-05')::BIGINT BEHAVIOR_EVENTS_90D_ASOF_2025_06_04,
       count(DISTINCT PRODUCT_CODE) FILTER (WHERE EVENT_AT >= TIMESTAMP '2025-06-05' - INTERVAL '90 days'
                          AND EVENT_AT < TIMESTAMP '2025-06-05')::BIGINT BEHAVIOR_PRODUCTS_90D_ASOF_2025_06_04
  FROM behavior_events GROUP BY 1
)
SELECT h.*, l.LAST_BEHAVIOR_AT, v.LAST_BEHAVIOR_ACTION, v.LAST_BEHAVIOR_PRODUCT,
       v.LAST_BEHAVIOR_ACTION_AMBIGUOUS, v.LAST_BEHAVIOR_PRODUCT_AMBIGUOUS
FROM historical h JOIN behavior_last_dates l USING (SKP_CLIENT)
JOIN latest v USING (SKP_CLIENT)
""")

# %% [markdown]
# ## Customer join
# Aggregate first, then left-join. Keep missing amounts null; omit address.
# %%
select_sql = """
SELECT CAST(c.SKP_CLIENT AS BIGINT) SKP_CLIENT,
       c.NAME_GENDER, c.NAME_EDUCATION_TYPE,
       try_cast(c.DATE_BIRTH AS DATE) DATE_BIRTH,
       try_cast(c.CNT_CHILDREN AS DOUBLE) CNT_CHILDREN,
       c.FLAG_CAR_OWNER, c.NAME_SALARY_FREQUENCY,
       try_cast(c.CNT_PERSON_DEPENDENT AS DOUBLE) CNT_PERSON_DEPENDENT,
       try_cast(c.AVG_SESSION_PER_WEEK_2025 AS DOUBLE) AVG_SESSION_PER_WEEK_2025,
       k.CONTRACT_SOURCE_ROWS, k.CONTRACT_CASE_COUNT,
       k.CONTRACT_INCOME_MAIN_CONFLICT_CASE_COUNT,
       k.CONTRACT_INCOME_HOUSEHOLD_CONFLICT_CASE_COUNT,
       k.CONTRACT_INCOME_TYPE_CONFLICT_CASE_COUNT,
       k.CONTRACT_PROFESSION_CONFLICT_CASE_COUNT,
       k.CONTRACT_SIGNED_COUNT,
       k.CONTRACT_SIGNED_MISSING_SIGN_TIME_COUNT,
       k.CONTRACT_ACTIVE_COUNT, k.CONTRACT_WRITTEN_OFF_COUNT, k.CONTRACT_REJECTED_COUNT,
       k.CONTRACT_PRODUCT_COUNT, k.CONTRACT_SIGNED_CREDIT_TOTAL,
       k.CONTRACT_ACTIVE_CREDIT_TOTAL, k.CONTRACT_ACTIVE_ANNUITY_TOTAL,
       k.CONTRACT_SIGNED_INITIAL_PAYMENT_TOTAL, k.CONTRACT_INSURED_CASE_COUNT,
       k.CONTRACT_ACTIVE_BILLING_TOTAL, k.FIRST_DUE_TOTAL, k.SECOND_DUE_TOTAL,
       k.THIRD_DUE_TOTAL, k.FOURTH_DUE_TOTAL,
       k.ACTIVE_FIRST_DUE_TOTAL, k.ACTIVE_SECOND_DUE_TOTAL,
       k.ACTIVE_THIRD_DUE_TOTAL, k.ACTIVE_FOURTH_DUE_TOTAL,
       k.LAST_CONTRACT_APPLICATION_AT, k.LAST_CONTRACT_APPROVED_AT,
       k.LAST_CONTRACT_SIGNED_AT, k.LATEST_RECORDED_CONTRACT_TRANSAC,
       k.LATEST_CONTRACT_STATUS, k.LATEST_CONTRACT_PRODUCT,
       k.LATEST_INCOME_TYPE, k.LATEST_PROFESSION, k.LATEST_PAYMENT_NUM,
       k.LATEST_INCOME_MAIN, k.LATEST_POSITIVE_INCOME_MAIN,
       k.LATEST_INCOME_HOUSEHOLD_RAW, k.LATEST_POSITIVE_INCOME_HOUSEHOLD,
       coalesce(a.CALL_SOURCE_ROWS,0) CALL_SOURCE_ROWS,
       coalesce(a.CALL_COUNT,0) CALL_COUNT,
       coalesce(a.CALL_SYSTEM_ANSWER_COUNT,0) CALL_SYSTEM_ANSWER_COUNT,
       coalesce(a.CALL_DISPOSITION_REFUSE_COUNT,0) CALL_DISPOSITION_REFUSE_COUNT,
       coalesce(a.CALL_INBOUND_COUNT,0) CALL_INBOUND_COUNT,
       a.CALL_DURATION_SECONDS_TOTAL, a.LAST_CALL_AT,
       coalesce(a.CALL_PRODUCT_TYPE_COUNT,0) CALL_PRODUCT_TYPE_COUNT, a.LAST_CALL_PRODUCT_TYPE,
       a.LAST_CALL_SOURCE, a.LAST_CALL_SYSTEM_RESULT, a.LAST_CALL_DISPOSITION,
       coalesce(a.CALL_COUNT_ASOF_2025_06_04,0) CALL_COUNT_ASOF_2025_06_04,
       coalesce(a.CALL_COUNT_90D_ASOF_2025_06_04,0) CALL_COUNT_90D_ASOF_2025_06_04,
       coalesce(a.CALL_SYSTEM_ANSWER_ASOF_2025_06_04,0) CALL_SYSTEM_ANSWER_ASOF_2025_06_04,
       coalesce(a.CALL_REFUSE_ASOF_2025_06_04,0) CALL_REFUSE_ASOF_2025_06_04,
       coalesce(a.CALL_INBOUND_ASOF_2025_06_04,0) CALL_INBOUND_ASOF_2025_06_04,
       coalesce(a.CALL_PRODUCT_TYPES_ASOF_2025_06_04,0) CALL_PRODUCT_TYPES_ASOF_2025_06_04,
       a.CALL_DURATION_SECONDS_ASOF_2025_06_04,
       a.LAST_CALL_ASOF_2025_06_04,
       a.LAST_CALL_PRODUCT_ASOF_2025_06_04,
       a.LAST_CALL_RESULT_ASOF_2025_06_04,
       coalesce(l.LEAD_COUNT,0) LEAD_COUNT,
       coalesce(l.LEAD_DEAL_COUNT,0) LEAD_DEAL_COUNT,
       coalesce(l.LEAD_NOT_INTERESTED_COUNT,0) LEAD_NOT_INTERESTED_COUNT,
       coalesce(l.LEAD_ASSIGNED_COUNT,0) LEAD_ASSIGNED_COUNT,
       coalesce(l.LEAD_ACCEPTED_SENTINEL_COUNT,0) LEAD_ACCEPTED_SENTINEL_COUNT,
       l.LEAD_MAX_OFFER, l.LAST_LEAD_AT, l.LAST_LEAD_PRODUCT,
       coalesce(l.LEAD_COUNT_90D_ASOF_2025_06_04,0) LEAD_COUNT_90D_ASOF_2025_06_04,
       coalesce(l.LEAD_DEAL_90D_ASOF_2025_06_04,0) LEAD_DEAL_90D_ASOF_2025_06_04,
       l.LAST_LEAD_REASON, l.LAST_LEAD_SOURCE,
       coalesce(b.BEHAVIOR_EVENT_COUNT,0) BEHAVIOR_EVENT_COUNT,
       coalesce(b.BEHAVIOR_SCORING_COUNT,0) BEHAVIOR_SCORING_COUNT,
       coalesce(b.BEHAVIOR_SOB_COUNT,0) BEHAVIOR_SOB_COUNT,
       coalesce(b.BEHAVIOR_PRODUCT_COUNT,0) BEHAVIOR_PRODUCT_COUNT,
       b.LAST_BEHAVIOR_AT, b.LAST_BEHAVIOR_ACTION, b.LAST_BEHAVIOR_PRODUCT,
       coalesce(b.LAST_BEHAVIOR_ACTION_AMBIGUOUS,0) LAST_BEHAVIOR_ACTION_AMBIGUOUS,
       coalesce(b.LAST_BEHAVIOR_PRODUCT_AMBIGUOUS,0) LAST_BEHAVIOR_PRODUCT_AMBIGUOUS,
       coalesce(b.BEHAVIOR_EVENTS_90D_ASOF_2025_06_04,0) BEHAVIOR_EVENTS_90D_ASOF_2025_06_04,
       coalesce(b.BEHAVIOR_PRODUCTS_90D_ASOF_2025_06_04,0) BEHAVIOR_PRODUCTS_90D_ASOF_2025_06_04
FROM CUSTOMER c
LEFT JOIN contracts_by_client k ON cast(c.SKP_CLIENT AS BIGINT) = k.SKP_CLIENT
LEFT JOIN calls_by_client a ON cast(c.SKP_CLIENT AS BIGINT) = a.SKP_CLIENT
LEFT JOIN leads_by_client l ON cast(c.SKP_CLIENT AS BIGINT) = l.SKP_CLIENT
LEFT JOIN behavior_by_client b ON cast(c.SKP_CLIENT AS BIGINT) = b.SKP_CLIENT
"""
# %% [markdown]
# ## Reconcile and save
# Check keys, source totals and dates. Existing Parquet must match exactly.
# %%
assert con.execute('SELECT count(*) = count(DISTINCT SKP_CLIENT) FROM CUSTOMER').fetchone()[0]
if TEMP.exists():
    TEMP.unlink()
try:
    con.execute(f"COPY ({select_sql} ORDER BY SKP_CLIENT) TO '{TEMP.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"CREATE VIEW final AS SELECT * FROM read_parquet('{TEMP.as_posix()}')")
    row = con.execute('''SELECT count(*), count(DISTINCT SKP_CLIENT), sum(CONTRACT_SOURCE_ROWS),
        sum(CONTRACT_CASE_COUNT), sum(CALL_SOURCE_ROWS), sum(CALL_COUNT), sum(LEAD_COUNT),
        sum(BEHAVIOR_EVENT_COUNT), sum(CALL_COUNT_ASOF_2025_06_04)
        FROM final''').fetchone()
    expected_cases = con.execute('SELECT count(*) FROM cases').fetchone()[0]
    expected_calls = con.execute("SELECT count(*) FROM CALL WHERE try_cast(TIME_OF_CALL_START AS TIMESTAMP) < TIMESTAMP '2025-06-05'").fetchone()[0]
    assert row[:2] == (source_counts['CUSTOMER'],) * 2
    assert row[2:8] == (source_counts['CONTRACT'], expected_cases, source_counts['CALL'],
                        source_counts['CALL'], source_counts['LEADS'], source_counts['PAST_BEHAVIOR'])
    assert row[8] == expected_calls
    for table, date_column, output_column in (
        ('CALL', 'TIME_OF_CALL_START', 'CALL_COUNT_90D_ASOF_2025_06_04'),
        ('LEADS', 'DTIME_CREATED', 'LEAD_COUNT_90D_ASOF_2025_06_04'),
        ('PAST_BEHAVIOR', 'DTIME_EVENT', 'BEHAVIOR_EVENTS_90D_ASOF_2025_06_04'),
    ):
        actual = con.execute(f'SELECT sum({output_column}) FROM final').fetchone()[0]
        expected = con.execute(f"SELECT count(*) FROM {table} WHERE try_cast({date_column} AS TIMESTAMP) >= TIMESTAMP '2025-03-07' AND try_cast({date_column} AS TIMESTAMP) < TIMESTAMP '2025-06-05'").fetchone()[0]
        assert actual == expected, (table, actual, expected)
    assert con.execute('''SELECT count(*) FROM final WHERE CONTRACT_CASE_COUNT IS NULL
        OR CONTRACT_ACTIVE_COUNT > CONTRACT_SIGNED_COUNT
        OR CONTRACT_ACTIVE_CREDIT_TOTAL > CONTRACT_SIGNED_CREDIT_TOTAL
        OR CALL_COUNT_90D_ASOF_2025_06_04 > CALL_COUNT_ASOF_2025_06_04
        OR LAST_CALL_ASOF_2025_06_04 >= TIMESTAMP '2025-06-05'
        OR (LAST_BEHAVIOR_PRODUCT_AMBIGUOUS=1 AND LAST_BEHAVIOR_PRODUCT IS NOT NULL)
        OR (LAST_BEHAVIOR_ACTION_AMBIGUOUS=1 AND LAST_BEHAVIOR_ACTION IS NOT NULL)''').fetchone()[0] == 0
    if OUTPUT.exists():
        con.execute(f"CREATE VIEW previous AS SELECT * FROM read_parquet('{OUTPUT.as_posix()}')")
        diff = con.execute('''SELECT count(*) FROM (
            (SELECT * FROM final EXCEPT ALL SELECT * FROM previous)
            UNION ALL
            (SELECT * FROM previous EXCEPT ALL SELECT * FROM final)
        )''').fetchone()[0]
        assert diff == 0, f'Existing data_clean.parquet differs in {diff} rows; left untouched'
        TEMP.unlink()
        print(f'Existing {OUTPUT.name} matches the rebuilt {row[0]:,}-customer file; unchanged.', flush=True)
    else:
        TEMP.replace(OUTPUT)
        print(f'Wrote {OUTPUT.name}: {row[0]:,} unique customers.', flush=True)
    print(f'Reconciled contract rows/cases {row[2]:,}/{row[3]:,}; calls {row[4]:,}; leads {row[6]:,}; behavior {row[7]:,}.', flush=True)
except Exception:
    TEMP.unlink(missing_ok=True)
    raise

# %% [markdown]
# ## Output
# One row per customer. Historical arrears are not current balances.
