# How accurate is it?

These numbers come from the bundled synthetic data, so they show the method works end to end; they are not a benchmark. Run the same backtest on your own history with `renewal-risk evaluate past_renewals.csv --profile saas`.

Each profile is backtested the way it would be used: trained on earlier renewals, then scored on later renewals the model never saw, and compared with what actually happened.

| Profile | ROC AUC: scorecard → scorer | Top-10% lift | Churned ARR in top 20% | Watermelon churn vs other green | Actual GRR vs forecast |
|---|---|---|---|---|---|
| AI products (credits, tokens, hybrid plans) | 0.924 → **0.939** | 5.3× | 83% | too few to judge (11 flagged) | 84.8% (inside range) |
| Consumption and compute services | 0.907 → **0.920** | 5.7× | 63% | too few to judge (3 flagged) | 89.8% (inside range) |
| Hardware and support contracts | 0.925 → **0.943** | 6.0× | 78% | too few to judge (3 flagged) | 88.5% (inside range) |
| Seat-based SaaS | 0.876 → **0.898** | 5.3× | 83% | too few to judge (1 flagged) | 86.4% (inside range) |
| Professional and managed services | 0.885 → **0.905** | 6.0× | 84% | too few to judge (19 flagged) | 91.8% (outside range) |

How to read it: ROC AUC is how often a churned account ranks above a renewed one (0.5 is a coin flip). Lift is how much more often the riskiest tenth churns than average. The watermelon column compares green accounts the scorer flagged with the green accounts it didn't.

## AI products (credits, tokens, hybrid plans)

Trained on 3,000 renewals (2023-12-04 to 2026-02-04). Tested on the next 1,000 (2026-02-04 to 2026-11-01), of which 15.5% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.924 | 0.717 | 0.0963 | 4.90 | 82% |
| Model alone | 0.940 | 0.768 | 0.0657 | 5.16 | 78% |
| Model + scorecard (what the scorer uses) | 0.939 | 0.775 | 0.0656 | 5.29 | 83% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.1310 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.4% | 0.0% | 200 |
| 1.1% | 0.5% | 200 |
| 3.2% | 2.5% | 200 |
| 12.9% | 13.0% | 200 |
| 60.4% | 61.5% | 200 |

**Scorecard bands:** green 0.8% churned (517), yellow 25.5% churned (432), red 80.4% churned (51).

**Watermelons:** 11 of 517 green accounts were flagged. 9.1% of them churned, against 0.6% of the other green accounts. That is too few flags to judge (under 20).

**GRR forecast:** expected 86.0%, P10 84.1% to P90 87.9%. Actual: 84.8%, inside the range. Predicted churned ARR was 8.5% of the book; actual was 8.1%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._

## Consumption and compute services

Trained on 3,000 renewals (2023-12-03 to 2026-01-30). Tested on the next 1,000 (2026-01-31 to 2026-11-01), of which 10.3% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.907 | 0.596 | 0.0618 | 5.53 | 79% |
| Model alone | 0.920 | 0.664 | 0.0568 | 5.63 | 60% |
| Model + scorecard (what the scorer uses) | 0.920 | 0.672 | 0.0562 | 5.73 | 63% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.0925 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.4% | 0.5% | 200 |
| 0.8% | 0.0% | 200 |
| 2.0% | 0.5% | 200 |
| 6.3% | 11.5% | 200 |
| 37.3% | 39.0% | 200 |

**Scorecard bands:** green 0.6% churned (537), yellow 12.4% churned (372), red 59.3% churned (91).

**Watermelons:** 3 of 537 green accounts were flagged. 0.0% of them churned, against 0.6% of the other green accounts. That is too few flags to judge (under 20).

**GRR forecast:** expected 90.8%, P10 89.3% to P90 92.4%. Actual: 89.8%, inside the range. Predicted churned ARR was 5.3% of the book; actual was 6.6%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._

## Hardware and support contracts

Trained on 3,000 renewals (2023-12-03 to 2026-02-10). Tested on the next 1,000 (2026-02-10 to 2026-11-01), of which 11.9% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.925 | 0.666 | 0.0706 | 5.88 | 90% |
| Model alone | 0.939 | 0.719 | 0.0582 | 6.05 | 61% |
| Model + scorecard (what the scorer uses) | 0.943 | 0.729 | 0.0579 | 6.05 | 78% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.1054 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.4% | 0.0% | 200 |
| 0.9% | 1.0% | 200 |
| 2.1% | 1.0% | 200 |
| 6.7% | 7.5% | 200 |
| 40.0% | 50.0% | 200 |

**Scorecard bands:** green 0.6% churned (476), yellow 15.4% churned (468), red 78.6% churned (56).

**Watermelons:** 3 of 476 green accounts were flagged. 33.3% of them churned, against 0.4% of the other green accounts. That is too few flags to judge (under 20).

**GRR forecast:** expected 89.8%, P10 88.3% to P90 91.2%. Actual: 88.5%, inside the range. Predicted churned ARR was 4.4% of the book; actual was 4.7%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._

## Seat-based SaaS

Trained on 3,000 renewals (2023-12-03 to 2026-02-10). Tested on the next 1,000 (2026-02-10 to 2026-11-01), of which 10.0% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.876 | 0.434 | 0.0725 | 4.70 | 79% |
| Model alone | 0.897 | 0.546 | 0.0619 | 5.40 | 82% |
| Model + scorecard (what the scorer uses) | 0.898 | 0.534 | 0.0627 | 5.30 | 83% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.0900 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.6% | 0.0% | 200 |
| 1.3% | 1.5% | 200 |
| 2.7% | 3.5% | 200 |
| 6.9% | 5.5% | 200 |
| 35.6% | 39.5% | 200 |

**Scorecard bands:** green 1.2% churned (514), yellow 16.7% churned (443), red 46.5% churned (43).

**Watermelons:** 1 of 514 green accounts was flagged. 0.0% of them churned, against 1.2% of the other green accounts. That is too few flags to judge (under 20).

**GRR forecast:** expected 88.0%, P10 85.9% to P90 90.0%. Actual: 86.4%, inside the range. Predicted churned ARR was 7.4% of the book; actual was 9.2%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._

## Professional and managed services

Trained on 3,000 renewals (2023-12-03 to 2026-02-08). Tested on the next 1,000 (2026-02-08 to 2026-11-01), of which 8.9% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.885 | 0.516 | 0.0643 | 5.39 | 83% |
| Model alone | 0.907 | 0.596 | 0.0531 | 6.29 | 88% |
| Model + scorecard (what the scorer uses) | 0.905 | 0.576 | 0.0540 | 5.96 | 84% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.0811 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.6% | 0.5% | 200 |
| 1.3% | 0.0% | 200 |
| 2.9% | 2.5% | 200 |
| 7.8% | 8.0% | 200 |
| 38.5% | 33.5% | 200 |

**Scorecard bands:** green 1.6% churned (637), yellow 18.1% churned (342), red 81.0% churned (21).

**Watermelons:** 19 of 637 green accounts were flagged. 15.8% of them churned, against 1.1% of the other green accounts. That is too few flags to judge (under 20).

**GRR forecast:** expected 89.1%, P10 87.2% to P90 91.0%. Actual: 91.8%, outside the range. Predicted churned ARR was 7.5% of the book; actual was 4.0%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._
