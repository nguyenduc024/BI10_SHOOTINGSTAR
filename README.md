# BI10 Round 2 — Customer Financial Health & Engagement (team ShootingStar)

Explainable scores for 1,000,528 consumer-finance customers, built from contracts, calls, leads and
scoring events, and validated against what actually happened afterwards.

* **Financial Health Score (FHS, 0–100)**: a points scorecard on credit history, plus a live-status layer for
  current arrears. Validated out of time on 2024 data and on an untouched 2026 window.
* **Customer Engagement Score (CES, 0–100)**: built from customer-initiated activity only. Validated on sales
  outcomes of calls made in the 15 months after scoring.
* **Customer-level output**: both scores, bands, segments, component scores, reason codes and a plain-language
  explanation for every customer (`BI10_ROUND02_ANALYTICS/outputs/customer_scores.csv`).

Start with [`reports/EXECUTIVE_SUMMARY.md`](BI10_ROUND02_ANALYTICS/reports/EXECUTIVE_SUMMARY.md), then
[`reports/ANALYTICAL_REPORT.md`](BI10_ROUND02_ANALYTICS/reports/ANALYTICAL_REPORT.md) (method, evidence, all figures).

## Headline results

| | |
|---|---|
| Who is in the data | All customers applied in Dec-2024/Jan-2025: the Tet-2025 applicant cohort (12x normal volume, 32% first-time) |
| FHS, out of time (2024) | Band E 16.0% early stress vs band A 1.6% (10x); AUC 0.629; gradient boosting on the same inputs adds nothing (0.613 vs 0.611) |
| FHS, one year later (2026) | 40% of band E and 14% of band D in collection vs ≤3.5% in bands A–C |
| CES, calls after scoring | Top decile converts 15x the bottom decile; AUC 0.76 for a sale |
| Telesales | Passive customers take 32% of delivered offers and bring 9% of sales; highly engaged bring 60% |
| Call operations | 21% of dials go to people who ignored 10+ calls (1.4% answer); a 10-attempt cap loses 4.4% of contacts |
| Responsible lending | Customers in collection are offered credit as often as healthy ones (10.7% vs 12.1%) |

## Repository

```
BI10_ROUND02_DATASET/            the five source CSVs (input; not committed)
data_clean.parquet               customer 360, one row per customer (rebuilt by DATA PROCESSING)
data_model/                      event-level Parquet (contract cases, calls, leads, behaviour, customers) + audit_summary.json
BI10_ROUND02_ANALYTICS/
  BI10_full_analysis.py          the whole pipeline in one file, run top to bottom, sections numbered by task:
                                   DATA PROCESSING  audit the 5 CSVs, build the data model and the customer 360
                                   SETUP  shared paths, metrics, WoE scorecard, chart style
                                   0      EDA: data traps, right-censoring, Tet risk, contact initiator
                                   1      Challenge 1 (FHS): point-in-time features, scorecard, validation,
                                          live-status layer, bands, fairness, segments
                                   2      Challenge 2 (CES): customer-initiated touches, components,
                                          validation, call operations, segments, drivers
                                   3      customer_scores.csv / .parquet, health x engagement grid, offer allocation
                                   4      17 slide-ready charts, read from tables/ only
                                   5      reports/*.md, numbers read from tables/ only
  tables/                        every table and metric behind the text and charts (CSV / JSON)
  figures/                       PNG charts, 16:9, numbered in story order
  outputs/                       customer-level scores
  reports/                       executive summary and analytical report
```

## How to reproduce

```bash
pip install -r requirements.txt
# put CALL.csv, CONTRACT.csv, CUSTOMER.csv, LEADS.csv, PAST_BEHAVIOR.csv in ./BI10_ROUND02_DATASET/
python BI10_ROUND02_ANALYTICS/BI10_full_analysis.py   # about 6 minutes on a laptop; deterministic
```

The script is also a notebook: open it in VS Code / Jupyter and run the `# %%` cells top to bottom.

The five source CSVs in `BI10_ROUND02_DATASET/` are the only inputs; nothing outside this repository and no
database file is read. Every CSV is checked against the SHA-256 of the distributed file and the run stops if a
copy differs: during the project a team copy of `LEADS.csv` turned out to be truncated (file size unchanged),
which only this check revealed; it was replaced by a byte-identical original.

## Analytical workflow

1. **Data model.** Contract rows repeat 2–8 times per credit case (x1.46 inflation): deduplicated to one row per
   case. The zero incomes on the repeated copies are placeholders, not missing income. Calls get an outcome
   taxonomy (no contact, wrong party, sales conversion / interest / decline, collection, retention, service).
2. **No future information.** Scores are taken at T = 2025-06-05, the day after the last contract record. Missed
   payments on instalment k count only once that instalment has fallen due; write-offs only after 9 months.
3. **FHS.** Weight-of-evidence logistic scorecard on 7 factors (repayment history, down payment, applications in
   90 days, income, last loan, largest clean loan, relationship length) trained on 2023–2024 snapshots and tested
   out of time; thin-file customers scored separately and flagged provisional. Live points for collection contact
   (Jan–Jun 2025) and over-extension are fitted on Jun–Dec 2025 calls and validated on Jan–Aug 2026 calls.
4. **CES.** Recency 35%, frequency 25%, interaction 15%, lead activity 10%, product interaction 15%, all from
   actions the customer started. Validated by a back-test (Feb-2025 score → repeat applications) and by sales
   outcomes on calls after T.
5. **Joint view.** FHS band x CES tier grid with future sales and collection rates: the starting point for
   segmentation and next best action.

Demographics (gender, age, education, address) are never score inputs; gender and age are used only in a
fairness check. All associations are descriptive; recommended changes should be rolled out as controlled tests.

## Output columns

See [`tables/customer_scores_dictionary.csv`](BI10_ROUND02_ANALYTICS/tables/customer_scores_dictionary.csv).
The CSV (≈520 MB) is not committed; the identical Parquet is, and `BI10_full_analysis.py` rebuilds the CSV.
