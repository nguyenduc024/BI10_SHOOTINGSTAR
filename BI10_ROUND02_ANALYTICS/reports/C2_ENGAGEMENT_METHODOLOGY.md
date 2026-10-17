# Challenge 2 — Customer Engagement Score: methodology

**Population:** 1,000,528 customers | **As of:** 4 Jun 2025 | **Input:** `data_clean.parquet` (SHA-256 `6c48fc813a5927cf…`)

## Definition
Engagement is customer-initiated behaviour towards the company beyond a single transaction
(Pansari & Kumar, 2017; Lim et al., 2022). Only customer-side touchpoints count:
* applications and scoring through any channel;
* calls the customer answered, net of refusals;
* lead offers the customer accepted or initiated.

Calls placed by the company are company effort and are used only as a baseline.

## Dimensions (RFM+)
RFM is the backbone (Mena et al., 2024; Brito et al., 2024). The Interaction, Lead and Product
dimensions extend it, as RFMA does for banking (Mosa et al., 2023).

| Dim | Measure | Coding |
|---|---|---|
| R | days since the last customer-side touchpoint | quintiles among dated customers; most recent = 5; no date = 1 |
| F | application / scoring events, last 90 days | 0 → 1; positives in quartiles 2–5 |
| M | total signed credit amount | same as F |
| I | answered calls minus refusals | same as F |
| L | accepted lead offers | same as F |
| P | distinct products applied for | same as F |

Some customers' last answered call or recent accepted offer is not the latest record, so their date
gets a conservative bound: 49,131 customers (call) and 1,062 customers (offer).

## Weights (outcome-free)
D-CRITIC (Krishnan et al., 2021): `C_j = σ_j Σ_k (1 − dCor_jk)`, `w_j = C_j / ΣC`. Distance
correlation is computed exactly from the 5 × 5 joint frequency tables and checked against brute force.

The composite follows the JRC (2019) 10-step guide: framework → selection → treatment →
normalisation → weighting → aggregation → sensitivity → validation.

| Dim | Meaning | D-CRITIC | CRITIC | Equal |
|---|---|---|---|---|
| R | Recency | 0.214 | 0.217 | 0.167 |
| F | Frequency (90d events) | 0.165 | 0.166 | 0.167 |
| M | Monetary (signed credit) | 0.247 | 0.245 | 0.167 |
| I | Interaction (net answered calls) | 0.143 | 0.143 | 0.167 |
| L | Lead activity (accepted offers) | 0.104 | 0.105 | 0.167 |
| P | Product interaction | 0.127 | 0.125 | 0.167 |

`Engagement = 100 × Σ w_j (s_j − 1) / 4`. Median: 24.9.

## Segments and tiers
The R × FM grid (FM = mean of F and M, rounded half up) is applied in a fixed precedence order.
The segments are then merged into tiers, plus two groups without customer-side touchpoints.

| Tier | Customers | Share | Median score | Answer rate after cutoff (called before) | Inbound after cutoff | Signed (90d, descriptive) | Credit-seeking |
|---|---|---|---|---|---|---|---|
| Highly engaged | 156,210 | 15.6% | 58.5 | 12.8% | 9.6% | 24.6% | 5.6% |
| Growing | 98,841 | 9.9% | 45.1 | 18.2% | 8.1% | 11.5% | 5.4% |
| Cooling | 49,335 | 4.9% | 26.4 | 12.1% | 3.7% | 2.8% | 1.9% |
| Slipping high-value | 109,296 | 10.9% | 36.6 | 11.9% | 7.8% | 10.7% | 2.2% |
| Dormant | 97,556 | 9.8% | 18.7 | 13.0% | 4.6% | 3.7% | 2.8% |
| Called, no response | 66,527 | 6.6% | 18.7 | 9.5% | 11.1% | 11.2% | 1.1% |
| No interaction recorded | 422,763 | 42.3% | 12.5 | n/a (never called) | 0.5% | 4.2% | 1.7% |

## Why same-window conversions are not used for validation
Customers with a lead record in the window applied at 100% versus 4% with neither; 92% of in-window leads were created within one day of the application. Lead records are application by-products, so same-window conversion is not used for validation.

## Hand-off to Challenge 3
`credit_seeking_flag` = the latest application in the 90-day window was rejected **and** the customer
has at least two rejected applications in total. In the top engagement decile 33% signed, but 9% had their latest application rejected and 7.0% are flagged credit-seeking (bottom decile: 3% and 3.2%). Engagement alone must not trigger a credit offer.

## Limitations
* The customer file keeps only the latest call and lead, so some recency dates are bounds, not exact dates.
* Collection calls cannot be separated from sales calls in the customer-level file, so answered calls
  may include collection contacts.
* The only forward outcome is call behaviour. No contract data exists after 4 Jun 2025, so a forward
  conversion test is not possible with this file.
* Who gets called is the company's choice. Stratifying by call exposure controls for intensity, not for selection.
* `AVG_SESSION_PER_WEEK_2025` has no documented window, so it is analysed only as a candidate driver.
* Demographics are not used.

## References
1. Brito, J. B. G., Bucco, G. B., Heldt, R., Becker, J. L., Silveira, C. S., Luce, F. B., & Anzanello, M. J. (2024). A framework to improve churn prediction performance in retail banking. *Financial Innovation*, 10, 17. https://doi.org/10.1186/s40854-023-00558-3
2. European Commission, Joint Research Centre (2019). *Your 10-Step Pocket Guide to Composite Indicators & Scoreboards*. Knowledge4Policy, Competence Centre on Composite Indicators and Scoreboards.
3. Krishnan, A. R., Kasim, M. M., Hamid, R., & Ghazali, M. F. (2021). A modified CRITIC method to estimate the objective weights of decision criteria. *Symmetry*, 13(6), 973. https://doi.org/10.3390/sym13060973
4. Lim, W. M., Rasul, T., Kumar, S., & Ala, M. (2022). Past, present, and future of customer engagement. *Journal of Business Research*, 140, 439–458. https://doi.org/10.1016/j.jbusres.2021.11.014
5. Mena, C. G., Coussement, K., De Bock, K. W., De Caigny, A., & Lessmann, S. (2024). Exploiting time-varying RFM measures for customer churn prediction with deep neural networks. *Annals of Operations Research*, 339(1–2), 765–787.
6. Mosa, M., Agami, N., Elkhayat, G., & Kholief, M. (2023). A novel hybrid segmentation approach for decision support: A case study in banking. *The Computer Journal*, 66(5), 1228–. https://doi.org/10.1093/comjnl/bxac009 [VERIFY author order]
7. Pansari, A., & Kumar, V. (2017). Customer engagement: The construct, antecedents, and consequences. *Journal of the Academy of Marketing Science*, 45(3), 294–311.
8. Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards* (2nd ed.). Wiley.

Origins, cited for attribution only:
* Hughes (1994), *Strategic Database Marketing* — quintile RFM coding.
* Diakoulaki, Mavrotas & Papayannakis (1995), *Computers & Operations Research* 22(7) — CRITIC.
* Székely, Rizzo & Bakirov (2007), *Annals of Statistics* 35(6) — distance correlation.
* Hanley & MacGibbon (2006), *Computer Methods and Programs in Biomedicine* 83(1) — Poisson bootstrap.

The 11-segment R × FM grid is a widely used practitioner convention [VERIFY original source].
