# Methodology

This document explains how the scorer turns a CRM export into a ranked, explained, dollarized list of renewal risk, and why each choice was made. The short version: a transparent scorecard of the kind CS teams already run, a calibrated gradient-boosted model that learns from your own renewal history, a stack that reconciles the two, and a forecast that turns account probabilities into a gross revenue retention (GRR) range.

All results quoted here and in the README come from **synthetic data**. The generator is designed to behave like a real book (correlated signals, sparse surveys, shocks the data can't see), but the numbers prove the pipeline works, not that it will hit the same accuracy on your customers. Train on your own history before trusting it.

## 1. Ingest: meet the data where it is

CS and revenue data lives in Salesforce reports, Gainsight exports and spreadsheets, each with its own headers. Rather than force a template, `ingest` maps each column to a canonical field through an alias table (`schema.py`): `ARR__c`, `Annual Recurring Revenue` and `arr` all land on `arr`; `Licensed Users` and `Licensed_Users__c` land on `licensed_seats`. A YAML mapping file covers anything unusual (see `examples/mapping.yaml`).

Ingest also cleans what exports get wrong: currency strings, `Yes`/`true`/`1.0` booleans, fractions that should be percents, Red/Yellow/Green sentiment, date formats. It drops rows it cannot score (no ID, no ARR, no renewal date) and reports how many and why. It derives signals the raw export does not hold: seat utilization, usage trend, commit pacing, projected commit use at term end, seat change and seat compression.

## 2. The scorecard: what a CS team can read and tune

The scorecard follows the structure CS platforms use, Gainsight's in particular: numeric **measures** scored 0 to 100, rolled into weighted **groups**, rolled into one health score with green, yellow and red bands. Each measure maps its raw value to a score through piecewise-linear breakpoints written in YAML, so a CS leader can read and change them without code.

Two design choices matter:

- **Missing data is reweighted, not scored as zero.** If an account has no NPS response, the NPS weight is spread across the measures that do have data, and the account's **confidence** (share of total weight with data) drops. A zero would punish accounts for a survey they never got.
- **Leading and lagging indicators are labeled.** Usage momentum, pacing and engagement lead; support history, payment and sentiment lag. The SaaS profile covers all four parts of Gainsight's DEAR framework: Deployment, Engagement, Adoption and ROI ([Gainsight, DEAR ebook](https://www.gainsight.com/resource/the-dear-framework-for-customer-health-scoring-to-grow-and-forecast-nrr/)).

**Alerts** sit beside the score, like calls to action in a CS platform: specific, explainable conditions (champion left inside six months of renewal, competitor evaluation, invoice 60+ days past due, a stalled pilot) that fire whatever the overall score says. A high-severity alert floors the account's tier at "watch", so neither a green score nor a confident model can bury a known trigger.

## 3. Industry profiles

Renewal risk looks different depending on what is being renewed. Each profile is one YAML file with its own groups, measures, alerts and model features.

| Profile | What it is for | Weighted toward |
|---|---|---|
| `ai_usage` (default) | AI APIs, copilots, agents and AI-native apps on credits, token commits or hybrid plans | Credit burn in both directions, pilot-to-production progress, embedding depth, seat compression, model-retirement exposure |
| `saas` | Per-seat subscriptions | Seat activation, active-user trend, breadth of adoption, stakeholder engagement |
| `consumption` | Cloud, data platforms, compute and data-center capacity on commit-and-drawdown contracts | Commit pacing and run rate against commit, consumption trend, workload breadth |
| `hardware` | Devices and infrastructure sold with support contracts | Contract coverage of the installed base, fleet age past end of service life, failure and return rates, telemetry |
| `services` | Professional and managed services, retainers | Hours pacing against contract, SLA attainment, milestones on time, delivery-lead stability |

The consumption profile reflects how public consumption businesses describe their own risk: growth in existing customers' consumption is the key measure, and slower-than-expected consumption is a stated retention risk. The hardware profile uses the support-business levers that TSIA's work on service revenue centers on: attach (share of the installed base under contract) and renewal of those contracts.

### Why AI products get their own profile

AI renewals fail in ways a seat-SaaS scorecard misreads.

- **Retention is much weaker, and it depends on commitment.** ChartMogul's analysis with Kyle Poyar of about 3,500 software companies found a median GRR of 40% and median net revenue retention of 48% for AI-native companies with at least $250k ARR in 2025. That headline hides a split by price point: plans over $250 a month retained 70% gross, close to B2B SaaS, while plans under $50 a month retained 23% ([ChartMogul, "The AI churn wave", Dec 2025](https://chartmogul.com/reports/saas-retention-the-ai-churn-wave/)). Commitment and depth of use are the signal, so the profile weights them.
- **Pilots ("tourists") are the biggest single risk.** In a16z's June 2025 survey of 100 enterprise CIOs, innovation budgets fell from about a quarter of LLM spend to 7% as AI moved into permanent budget lines, and 37% of enterprises now use five or more models ([a16z, June 2025](https://a16z.com/ai-enterprise-2025/)). A pilot that has not reached production workflows by its end date, or that is still paid from an experimental budget, is exposed when budgets are consolidated. Multi-model use also means switching providers is cheap unless the customer is deeply integrated.
- **Burn is two-sided.** Under-burning a prepaid commit means a smaller re-commit at true-up. Over-burning is not only upside: overage, rate limits and spend caps cause bill shock. Cursor's June 2025 move from request limits to compute-based limits ended in an apology and refunds of surprise charges ([Simon Willison's summary](https://simonwillison.net/2025/Jul/5/cursor-clarifying-our-pricing/)), and Anthropic's weekly limits on Claude plans, announced in July 2025 for late August, drew a similar response from heavy users ([Engadget](https://www.engadget.com/ai/anthropic-is-rate-limiting-claude-code-blaming-some-users-for-never-turning-it-off-211134730.html)). So **projected commit use at term end** scores best near 100% and worse in both directions, and the `bill_shock` alert needs both over-pacing and repeated throttling.
- **Seat compression is repricing, not churn.** When AI absorbs work, a customer cuts seats while usage grows. A seat scorecard reads that as churn. Here it is a contraction signal that feeds the downsell model and a seat-to-usage repricing play.
- **Model dependency.** Usage concentrated on models the vendor will retire before the renewal forces a migration inside the renewal window. Shallow integration makes that migration an easy moment to leave.

The specific breakpoints (for example, production share under 40% for a stalled pilot) are judgment calls informed by these sources, not published benchmarks. They are meant to be tuned on your own history.

## 4. The model: learn what the scorecard can't

A scorecard is a hypothesis about what matters. Renewal history tests it. The model layer trains two LightGBM gradient-boosted models on past renewals: one for **churn** (the customer leaves) and one for **contraction** (they renew smaller).

- **Monotone constraints** keep the model sensible where direction is known: more P1 tickets never lowers risk, higher seat utilization never raises it. Signals that are risky in both directions (burn pacing, seat change) are left unconstrained.
- **Native missing-value handling.** Gradient boosting learns what a missing survey means instead of imputing it.
- **Exact reason codes.** Each prediction is decomposed with TreeSHAP (LightGBM's `pred_contrib`), so every account gets its top risk drivers and protective factors in plain English, with their contribution in log-odds.
- **Calibration.** Raw boosted scores are not probabilities. A held-out calibration split fits Platt scaling (isotonic only when there are 500+ positives and negatives, where it is stable), so a 30% churn probability means about 3 in 10 accounts like it churned.
- **Honest evaluation.** Metrics come from 5-fold stratified out-of-fold predictions, compared against the scorecard alone on the same rows: ROC AUC, PR AUC, Brier score, top-decile lift and the share of churned ARR captured in the top 20% of accounts.
- **No pickles.** Models are saved as JSON plus LightGBM's text format, so a model file can be inspected and loaded safely.

### Reconciling the scorecard and the model

CS teams trust their scorecard; models catch interactions it misses. Rather than pick one, the final churn probability is a **logistic stack** over the logits of the model's out-of-fold prediction and the scorecard's prior. On every synthetic profile the stack beats the scorecard on ROC AUC and Brier score, and lands within 0.001 ROC AUC of the model alone or better (table in the README).

**Watermelons** are where the two disagree in a dangerous direction: green on the scorecard, but the model alone puts churn risk at least 4 times the churn rate green accounts showed in training (and never below 8%). These are the accounts to pressure-test before the renewal call.

## 5. From probability to priority

- **Expected ARR loss** = ARR × (P(churn) + (1 − P(churn)) × P(contraction) × expected contraction depth).
- **Urgency** follows the common 180/120/90-day renewal cadence (plan, engage, negotiate). An account renewing in 30 days counts fully; one renewing in a year counts about a fifth.
- **Priority** ranks accounts by expected loss × urgency, so the list answers "where should I spend this week" in dollars.
- **Tiers** use churn probability and time left: critical (45%+ inside 180 days), high (25%+ inside 270 days), watch (12%+), low. Downsell risk shows in the dollars and the rank, not the tier.
- **Save plays** come from the strongest alert, then the top driver group: an owner, a timing window and concrete steps (for example "Pilot-to-production push" or "Seat-to-usage repricing").

## 6. GRR forecast

For renewals inside the forecast window, a Monte Carlo simulation draws churn and contraction for each account from its probabilities and reports expected GRR with P10, P50 and P90. Accounts are treated as independent, which understates correlated shocks (a recession, a competitor launch, a pricing change); the report says so.

## 7. Claude account briefs

With an `ANTHROPIC_API_KEY` set, the scorer asks Claude (`claude-opus-5-5`) to turn an account's computed facts into a short brief: headline, why it is at risk, a save plan with owners and timing, and a talk track. The prompt is grounded in the facts the pipeline computed and the response is constrained to a JSON schema. Without a key, or on any API error or refusal, the same brief is written from the playbook rules, so nothing depends on the API being available. Only the selected account's facts are sent.

## 8. Synthetic data

`synth.py` generates a history of past renewals with outcomes and a current book for each profile. Signals are driven by a latent account-health factor plus profile-specific mechanics (pilots, over-burners, seat compression, ageing fleets, under-used retainers), with realistic sparsity (NPS response rates well under half), weak survey signals, multi-year contracts that mute risk, and unobserved shocks. Churn rates are set near 10% for classic profiles and 16% for AI. Five named story accounts are planted in every book so tests can check the pipeline surfaces them for the right reasons.

## Limits

- Synthetic results show the method works; they are not a benchmark. Expect lower accuracy on real data, where labels are noisier.
- Breakpoints and weights are informed defaults. Tune them, and retrain the model, on your own outcomes.
- The forecast assumes independent accounts.
- Causal claims are out of scope: the model ranks risk, it does not prove a play will work. Track save-play outcomes to learn that.
