# Challenge 2 — Validation report

The weights never see an outcome. The score is frozen at 4 Jun 2025, and the outcomes come only from
calls made between 5 Jun 2025 and 31 Aug 2026. AUCs are stratified jointly by post-cutoff and pre-cutoff call volume
(ten mid-rank bins each) to remove the "more calls, more chances" and "never called" effects. The 95% CIs come from
200 paired Poisson-bootstrap replicates (seed 20251018).

## AUC by outcome
| Outcome | Score | Stratified AUC | 95% CI | Pooled AUC | Engagement minus this (95% CI) |
|---|---|---|---|---|---|
| Answered a call after cutoff | Engagement Score | 0.545 | 0.543–0.548 | 0.543 | +0.000 (+0.000 to +0.000) |
| Answered a call after cutoff | Engagement, CRITIC weights | 0.545 | 0.543–0.548 | 0.543 | -0.000 (-0.000 to -0.000) |
| Answered a call after cutoff | Engagement, equal weights | 0.550 | 0.548–0.553 | 0.545 | -0.005 (-0.005 to -0.005) |
| Answered a call after cutoff | Recency only | 0.536 | 0.533–0.539 | 0.537 | +0.009 (+0.008 to +0.011) |
| Answered a call after cutoff | Frequency only | 0.488 | 0.486–0.490 | 0.500 | +0.057 (+0.055 to +0.059) |
| Answered a call after cutoff | Interaction only | 0.599 | 0.598–0.601 | 0.576 | -0.054 (-0.057 to -0.052) |
| Answered a call after cutoff | Classic RFM (equal) | 0.520 | 0.518–0.523 | 0.526 | +0.025 (+0.024 to +0.026) |
| Answered a call after cutoff | Previous team score | 0.498 | 0.496–0.501 | 0.445 | +0.047 (+0.044 to +0.049) |
| Answered a call after cutoff | Company effort (calls placed, 90d) | 0.521 | 0.519–0.523 | 0.419 | +0.024 (+0.021 to +0.028) |
| Inbound call after cutoff | Engagement Score | 0.633 | 0.626–0.639 | 0.715 | +0.000 (+0.000 to +0.000) |
| Inbound call after cutoff | Engagement, CRITIC weights | 0.633 | 0.626–0.639 | 0.715 | +0.000 (+0.000 to +0.000) |
| Inbound call after cutoff | Engagement, equal weights | 0.634 | 0.627–0.640 | 0.717 | -0.001 (-0.001 to -0.000) |
| Inbound call after cutoff | Recency only | 0.606 | 0.601–0.612 | 0.680 | +0.027 (+0.024 to +0.031) |
| Inbound call after cutoff | Frequency only | 0.561 | 0.555–0.565 | 0.565 | +0.073 (+0.069 to +0.076) |
| Inbound call after cutoff | Interaction only | 0.522 | 0.520–0.523 | 0.601 | +0.112 (+0.104 to +0.117) |
| Inbound call after cutoff | Classic RFM (equal) | 0.622 | 0.615–0.628 | 0.696 | +0.011 (+0.010 to +0.012) |
| Inbound call after cutoff | Previous team score | 0.611 | 0.605–0.617 | 0.785 | +0.022 (+0.017 to +0.026) |
| Inbound call after cutoff | Company effort (calls placed, 90d) | 0.517 | 0.516–0.519 | 0.772 | +0.116 (+0.109 to +0.122) |

## Key results
* Answering a call after the cutoff: stratified AUC 0.545 (95% CI 0.543–0.548) versus 0.521 for company effort and 0.498 for the previous score. Inbound call: 0.633 (CI 0.626–0.639).
* Answer rate after the cutoff: 16.8% in the top decile versus 12.8% in the bottom decile (pooled 14.5%); decile-rate Spearman ρ = 0.90.
* Among customers already called before the cutoff, Highly engaged customers answered 12.8% of later calls versus 9.5% for 'Called, no response'. Customers never called before answered 41.1% of their first calls versus 12.8% for previously called customers — a call-fatigue signal for Challenge 3.
* Rank agreement between weighting schemes: D-CRITIC (primary) vs CRITIC (Pearson) ρ = 1.000; D-CRITIC (primary) vs Equal weights ρ = 0.996; CRITIC (Pearson) vs Equal weights ρ = 0.996.
* Customers with a lead record in the window applied at 100% versus 4% with neither; 92% of in-window leads were created within one day of the application. Lead records are application by-products, so same-window conversion is not used for validation.

## Drivers (association, not causation)
| Factor | Score input | ρ with score | Forward exposure-adjusted AUC | IV, forward answer (band) |
|---|---|---|---|---|
| Answered calls net of refusals | yes | +0.33 | 0.599 (raises) | 0.195 (medium) |
| Refused calls before cutoff |  | +0.18 | 0.521 (raises) | 0.008 (unpredictive) |
| Company calls placed, 90d |  | +0.40 | 0.521 (raises) | 0.200 (medium) |
| Not-interested lead outcomes |  | +0.34 | 0.518 (raises) | 0.001 (unpredictive) |
| Signed credit amount (lifetime) | yes | +0.82 | 0.512 (raises) | 0.003 (unpredictive) |
| Inbound calls before cutoff |  | +0.12 | 0.506 (raises) | 0.001 (unpredictive) |
| Lead records, 90d |  | +0.15 | 0.504 (raises) | 0.001 (unpredictive) |
| App sessions per week (2025) |  | +0.11 | 0.501 (raises) | 0.001 (unpredictive) |
| Accepted lead offers | yes | +0.27 | 0.500 (raises) | 0.000 (unpredictive) |
| Products applied for | yes | +0.68 | 0.499 (reduces) | 0.000 (unpredictive) |
| Rejected applications (lifetime) |  | -0.06 | 0.495 (reduces) | 0.002 (unpredictive) |
| Products explored in scoring, 90d |  | +0.58 | 0.491 (reduces) | 0.000 (unpredictive) |
| Application / scoring events, 90d | yes | +0.58 | 0.488 (reduces) | 0.001 (unpredictive) |
| Latest action self-service digital |  | +0.29 | 0.477 (reduces) | 0.005 (unpredictive) |
| Days since last customer touchpoint | yes | -0.73 | 0.464 (reduces) | 0.029 (weak) |

## Data-quality checks
| Check | Value | Result |
|---|---|---|
| one row per customer | 1,000,528 rows / 1,000,528 ids | pass |
| calls <= calls_all | 0 | pass |
| answered <= answered_all | 0 | pass |
| inbound <= inbound_all | 0 | pass |
| refused <= refused_all | 0 | pass |
| calls_90d <= calls | 0 | pass |
| leads_90d <= leads | 0 | pass |
| lead_deals_90d <= lead_deals | 0 | pass |
| lead_deals <= leads | 0 | pass |
| behavior_events_90d <= behavior_events | 0 | pass |
| last_call_at before cutoff | 2025-06-04 22:15:51 | pass |
| last_lead_at before cutoff | 2025-06-04 20:56:56 | pass |
| last_behavior_at before cutoff | 2025-06-03 22:30:14 | pass |
| last_application_at before cutoff | 2025-06-04 22:29:02 | pass |
| last_signed_at before cutoff | 2025-06-04 22:29:47 | pass |
| refusals coded without a system ANSWER (customers) | 780 | pass |
| customers with calls after the cutoff | 351792 | pass |
| ambiguous latest behaviour action (customers) | 501 | pass |
| leads with SOB acceptance sentinel 3000-01-01 | 587914 | pass |
| customers with any app session in 2025 | 70477 | pass |
| answered-call dates bounded by call-log start | 49131 | pass |
| recent Deal dates bounded by window start | 1062 | pass |
| customer-side evidence without recoverable date | 2694 | pass |
| exact discrete dCor equals brute force (max abs gap) | 5.55e-16 | pass |
| engagement score within [0, 100] | 0.0–100.0 | pass |
| every customer matched by the segment grid | 0 | pass |
| strata for answered a call after cutoff | 70 | pass |
| strata for inbound call after cutoff | 20 | pass |

## Proposed causal test (for Challenge 3)
Randomly hold out a share of each tier from the Next Best Action campaign. Then compare answer,
application and signing rates between treated and holdout customers. This turns the associations
above into measured uplift.
