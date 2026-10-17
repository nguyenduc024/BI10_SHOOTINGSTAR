# AI Usage Report

> **Draft — the team must complete and confirm this before submission.** It lists the AI use we know of in
> this repository; add any other tools (e.g. for slides, the dashboard or translation) and who reviewed what.

## Tools

| Tool | Used for |
|---|---|
| Claude (Claude Code, Anthropic) | Reviewing the earlier pipelines, restructuring the code into the current scripts, writing data-integrity checks, drafting the scoring code, charts and the analytical report text |
| _add other tools here_ | |

## Stages supported

| Stage | What the AI did | How the team checked it |
|---|---|---|
| Data cleaning | Proposed the contract de-duplication, income placeholder rule, call outcome taxonomy, SHA-256 source check (which found a truncated local LEADS.csv) | Row counts reconciled to the source in the DATA PROCESSING section of `BI10_full_analysis.py` (asserts); taxonomy listed code by code in `data_model/call_outcome_map.csv` |
| Data recovery | Rebuilt the truncated LEADS.csv once, outside the pipeline, from the team's earlier raw import of the original file | The rebuilt file has the same SHA-256 as the distributed file (byte-identical); the pipeline itself reads only the five CSVs |
| Feature and score code | Ported the team's earlier DuckDB scorecard to the Parquet data model; added the live-status layer and the engagement score | Parity test against the earlier version: identical features except 1,365 customers on an exact 20%/40% down-payment boundary (now binned consistently); rank correlation of scores 0.999 |
| Validation | Suggested out-of-time and future-call checks, the call-volume adjustment and the clean back-test after finding the applicant-cohort sampling frame | Every metric recomputed by the scripts; two full runs produce byte-identical outputs |
| Charts and text | Drafted chart code and report wording; numbers are read from `tables/` by code, never typed by hand | Team review of every chart and claim against the tables |

## Principles followed

* No number in the report or charts is typed by hand; all come from `tables/`, produced by the scripts.
* AI suggestions that changed a result were kept only after a check in the data (several were reversed, e.g. an
  initial "sales calls to customers in collection" claim was replaced after finding that many product-tagged
  calls to these customers were collection campaigns).
* The AI assistant ran on the team's machine with access to the competition files, and read sample rows of the raw
  data (including a few address values) while profiling it. Addresses are not carried into the data model or any output.
