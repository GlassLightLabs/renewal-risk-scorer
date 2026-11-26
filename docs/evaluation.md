# How accurate is it?

These numbers come from the bundled synthetic data, so they show the method works end to end; they are not a benchmark. Run the same backtest on your own history with `renewal-risk evaluate past_renewals.csv --profile saas`.

Each profile is backtested the way it would be used: trained on earlier renewals, then scored on later renewals the model never saw, and compared with what actually happened.

| Profile | ROC AUC: scorecard → scorer | Top-10% lift | Churned ARR in top 20% | Watermelon churn vs other green | Actual GRR vs forecast |
|---|---|---|---|---|---|
| AI products (credits, tokens, hybrid plans) | 0.927 → **0.940** | 5.3× | 82% | too few to judge (7 flagged) | 84.8% (inside range) |
| Consumption and compute services | 0.911 → **0.924** | 5.7× | 75% | too few to judge (3 flagged) | 90.5% (inside range) |
| Hardware and support contracts | 0.925 → **0.942** | 6.1× | 80% | too few to judge (4 flagged) | 88.4% (inside range) |
| Seat-based SaaS | 0.874 → **0.886** | 5.7× | 73% | too few to judge (4 flagged) | 87.5% (inside range) |
| Professional and managed services | 0.879 → **0.898** | 5.7× | 88% | 13% vs 1% | 91.8% (outside range) |

How to read it: ROC AUC is how often a churned account ranks above a renewed one (0.5 is a coin flip). Lift is how much more often the riskiest tenth churns than average. The watermelon column compares green accounts the scorer flagged with the green accounts it didn't.

## AI products (credits, tokens, hybrid plans)

Trained on 3,000 renewals (2023-12-04 to 2026-02-04). Tested on the next 1,000 (2026-02-04 to 2026-11-01), of which 15.6% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.927 | 0.725 | 0.0965 | 4.94 | 82% |
| Model alone | 0.941 | 0.783 | 0.0652 | 5.26 | 75% |
| Model + scorecard (what the scorer uses) | 0.940 | 0.782 | 0.0654 | 5.26 | 82% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.1317 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.2% | 0.0% | 200 |
| 0.8% | 1.0% | 200 |
| 2.8% | 2.5% | 200 |
| 12.7% | 12.5% | 200 |
| 63.3% | 62.0% | 200 |

**Scorecard bands:** green 0.8% churned (517), yellow 25.7% churned (432), red 80.4% churned (51).

**Watermelons:** 7 of 517 green accounts were flagged. 0.0% of them churned, against 0.8% of the other green accounts. That is too few flags to judge (under 20).

**GRR forecast:** expected 86.2%, P10 84.4% to P90 88.0%. Actual: 84.8%, inside the range. Predicted churned ARR was 8.2% of the book; actual was 8.2%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._

## Consumption and compute services

Trained on 3,000 renewals (2023-12-03 to 2026-01-30). Tested on the next 1,000 (2026-01-31 to 2026-11-01), of which 10.6% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.911 | 0.612 | 0.0624 | 5.47 | 89% |
| Model alone | 0.926 | 0.684 | 0.0561 | 5.75 | 70% |
| Model + scorecard (what the scorer uses) | 0.924 | 0.681 | 0.0563 | 5.66 | 75% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.0948 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.5% | 0.5% | 200 |
| 1.1% | 0.0% | 200 |
| 2.7% | 1.5% | 200 |
| 8.3% | 9.0% | 200 |
| 39.7% | 42.0% | 200 |

**Scorecard bands:** green 0.6% churned (534), yellow 13.0% churned (377), red 60.7% churned (89).

**Watermelons:** 3 of 534 green accounts were flagged. 0.0% of them churned, against 0.6% of the other green accounts. That is too few flags to judge (under 20).

**GRR forecast:** expected 89.5%, P10 87.7% to P90 91.2%. Actual: 90.5%, inside the range. Predicted churned ARR was 6.5% of the book; actual was 5.6%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._

## Hardware and support contracts

Trained on 3,000 renewals (2023-12-03 to 2026-02-10). Tested on the next 1,000 (2026-02-10 to 2026-11-01), of which 12.0% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.925 | 0.675 | 0.0701 | 5.75 | 84% |
| Model alone | 0.937 | 0.736 | 0.0602 | 6.00 | 67% |
| Model + scorecard (what the scorer uses) | 0.942 | 0.750 | 0.0590 | 6.08 | 80% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.1064 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.6% | 0.0% | 200 |
| 1.2% | 1.0% | 200 |
| 2.4% | 0.5% | 200 |
| 6.6% | 8.5% | 200 |
| 36.6% | 50.0% | 200 |

**Scorecard bands:** green 0.8% churned (477), yellow 14.7% churned (462), red 78.7% churned (61).

**Watermelons:** 4 of 477 green accounts were flagged. 25.0% of them churned, against 0.6% of the other green accounts. That is too few flags to judge (under 20).

**GRR forecast:** expected 89.4%, P10 87.9% to P90 90.9%. Actual: 88.4%, inside the range. Predicted churned ARR was 4.6% of the book; actual was 4.9%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._

## Seat-based SaaS

Trained on 3,000 renewals (2023-12-03 to 2026-02-10). Tested on the next 1,000 (2026-02-10 to 2026-11-01), of which 9.5% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.874 | 0.421 | 0.0692 | 4.74 | 74% |
| Model alone | 0.880 | 0.520 | 0.0618 | 5.58 | 73% |
| Model + scorecard (what the scorer uses) | 0.886 | 0.520 | 0.0614 | 5.68 | 73% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.0860 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.6% | 0.0% | 200 |
| 1.3% | 1.0% | 200 |
| 2.8% | 4.0% | 200 |
| 7.1% | 8.0% | 200 |
| 36.5% | 34.5% | 200 |

**Scorecard bands:** green 1.2% churned (514), yellow 15.9% churned (447), red 46.2% churned (39).

**Watermelons:** 4 of 514 green accounts were flagged. 0.0% of them churned, against 1.2% of the other green accounts. That is too few flags to judge (under 20).

**GRR forecast:** expected 88.5%, P10 86.4% to P90 90.4%. Actual: 87.5%, inside the range. Predicted churned ARR was 6.9% of the book; actual was 7.9%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._

## Professional and managed services

Trained on 3,000 renewals (2023-12-03 to 2026-02-08). Tested on the next 1,000 (2026-02-08 to 2026-11-01), of which 9.2% churned. Source: synthetic data.

| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |
|---|---|---|---|---|---|
| Scorecard alone (rules, as most CS platforms score) | 0.879 | 0.523 | 0.0662 | 5.22 | 83% |
| Model alone | 0.897 | 0.579 | 0.0562 | 5.98 | 87% |
| Model + scorecard (what the scorer uses) | 0.898 | 0.572 | 0.0564 | 5.65 | 88% |
| Everyone at the historical churn rate | 0.500 | n/a | 0.0836 | 1.00 | 20% |

**Calibration** (accounts split into fifths by predicted risk):

| Predicted churn | Actual churn | Accounts |
|---|---|---|
| 0.7% | 0.5% | 200 |
| 1.4% | 1.5% | 200 |
| 3.2% | 1.0% | 200 |
| 8.8% | 8.5% | 200 |
| 39.9% | 34.5% | 200 |

**Scorecard bands:** green 1.6% churned (643), yellow 19.1% churned (335), red 81.8% churned (22).

**Watermelons:** 30 of 643 green accounts were flagged. 13.3% of them churned, against 1.0% of the other green accounts.

**GRR forecast:** expected 88.0%, P10 85.9% to P90 90.0%. Actual: 91.8%, outside the range. Predicted churned ARR was 8.4% of the book; actual was 4.1%.

_Synthetic data: this shows the method works end to end. It is not a benchmark._
