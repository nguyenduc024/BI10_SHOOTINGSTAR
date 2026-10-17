# BI10 Round 2 — Financial Health and Engagement: executive summary

**Who is in the data.** 1,000,528 customers, every one of whom applied for credit in Dec-2024/Jan-2025:
the Tet-2025 applicant cohort (12x normal monthly volume, 32% first-time applicants).
Scores are taken at **T = 2025-06-05**, the day after the last contract record; calls continue to Aug-2026 and
are kept as an untouched future window for validation.

**Financial Health Score (FHS, 0-100).** Points scorecard on 7 credit-history factors plus a live-status layer.
* Out of time (2024 data, never used for fitting): band E is 10x as stressed as band A
  (16.0% vs 1.6%); AUC 0.629.
  Gradient boosting on the same inputs: 0.613 vs scorecard 0.611 - no gain from complexity.
* A year later (Jan-Aug 2026, untouched): 40% of band E and
  14% of band D are in collection, vs at most
  3.5% in A-C.
* 11,751 customers are in **live arrears** today; 63% of them are in collection within 15 months.

**Customer Engagement Score (CES, 0-100).** Customer-initiated activity only (97% of calls and 97% of leads are company-pushed).
* On calls after T the top CES decile converts 15x the bottom decile
  (AUC 0.76 for a sale, 0.68 for interest).

**What the business should change now**
1. **Re-target telesales by CES.** Passive customers absorb 32% of delivered offers and bring
   9% of sales; highly engaged customers bring 60%
   (71 vs 7 sales per 1,000 offers).
2. **Cap dialling.** 21% of dials go to people who ignored 10+ calls (1.4% answer).
   A 10-attempt cap saves 802k dials and loses 4.4% of contacts.
   Call 7-10am and 5pm.
3. **Responsible-lending guardrail.** Over-extended customers are the most-offered segment (17.2% offered);
   customers already in collection are offered credit as often as healthy ones (10.7% vs
   12.1%) and signed 69 new loans. Block offers on live arrears; require an
   affordability check for over-extended customers.
4. **Tet underwriting.** First loans in the Tet wave were 2.1x as risky as repeat loans;
   78% of first-time borrowers had no prior footprint. Down payment is the strongest affordability signal in the score.
5. **Lead sourcing.** RA leads are 53% of all leads and convert 0.2%;
   SOB and landing-page leads convert 40% and 38%.
