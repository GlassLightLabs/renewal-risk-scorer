# Renewal Risk Scorer

**Find the renewals you're about to lose, know why, and know what to do this week.** Built first for AI companies, whose renewals break in ways classic SaaS health scores miss, with profiles for seat-based SaaS, consumption and compute, hardware and services too.

Feed it a Salesforce or Gainsight export of open renewals. It returns every account ranked by dollars at risk and how soon it renews, with:

- a **health scorecard** CS teams can read and tune (Gainsight-style weighted measures and groups),
- a **calibrated churn and downsell model** trained on your renewal history, with plain-English reason codes for every account,
- a **save play** with an owner, timing and steps,
- a **GRR forecast** with a P10 to P90 range for the quarter,
- **watermelon** flags (green on the scorecard, red underneath),
- optional **Claude-written account briefs** for call prep, with a rules-based fallback when no API key is set,
- **data add-ons** that pull signals from other tools, starting with **Zendesk** (ticket trend per seat, urgent tickets, CSAT, reopen rate, time to resolution), configured in the app or a YAML file.

It runs as a Streamlit app for reps and leaders, and as a CLI for RevOps and scheduled runs.

> All numbers in this README come from the bundled synthetic data. They show the method works end to end. Train on your own history before trusting the accuracy.

## Why another churn tool

Most open-source churn projects predict whether a subscriber cancels, from event logs, as a single number. Renewal teams need something else: which contracts, in dollars, by renewal date, for what reason, and what to do about it. No public project we found models the things that decide AI renewals:

| AI renewal risk | What the scorer does |
|---|---|
| **Pilots that never reach production.** Innovation budgets fell from about 25% to 7% of enterprise LLM spend in a year ([a16z, Jun 2025](https://a16z.com/ai-enterprise-2025/)); pilots get cut when budgets consolidate. | Production share of usage, pilot and experimental-budget flags, and a `stalled_pilot` alert with a pilot-to-production play. |
| **Burn in both directions.** Under-burn means a smaller re-commit. Over-burn means overage, rate limits and bill shock, which turned into refunds and backlash in 2025 pricing changes. | Projects commit use at term end and scores it best near 100%, worse either side. `under_burn` and `bill_shock` alerts. |
| **Seat compression.** AI absorbs work, so customers cut seats while usage grows. That's a repricing conversation, not churn. | Detects seats down while usage grows, routes it to the downsell model and a seat-to-usage repricing play. |
| **Model retirement.** Usage on models retiring before renewal forces a migration inside the renewal window. | Deprecated-model exposure and active integrations as stickiness. |
| **Weak, commitment-driven retention.** Median GRR for AI-native companies was 40%, but 70% on plans over $250 a month ([ChartMogul and Kyle Poyar, Dec 2025](https://chartmogul.com/reports/saas-retention-the-ai-churn-wave/)). | Weights commitment and depth of embedding, and forecasts GRR as a range, not a point. |

Beyond AI, three things are uncommon in public tools: the scorecard and the model are **reconciled** rather than competing (a logistic stack of both, plus watermelons where they disagree), risk is **dollarized and time-weighted** (expected ARR loss × renewal urgency), and every score comes with its **reasons and a play**.

## Quick start

```bash
git clone https://github.com/GlassLightLabs/renewal-risk-scorer && cd renewal-risk-scorer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"

renewal-risk demo                         # AI profile: generate, train, score, write demo_out/report.md
streamlit run app/streamlit_app.py        # the app, on synthetic data by default
```

The demo takes about 10 seconds. Sample output (AI profile, as of 2026-12-01):

```
                        ROC AUC   PR AUC    Brier  Lift@10%  ARR@20%
Scorecard alone           0.912    0.713   0.1044      4.68      73%
Model alone               0.928    0.774   0.0691      5.03      67%
Model + scorecard         0.928    0.778   0.0685      5.06      71%

AI products (credits, tokens, hybrid plans): 600 accounts, $139,469,100 ARR, expected ARR loss $14,912,102
Tiers: critical 33, high 49, watch 239, low 279 | watermelons 7 | overdue 35
GRR forecast (316 renewals): expected 88.6%, P10 86.0% to P90 90.9%

  #  Account                       ARR  Days  Churn Downsell  Exp. loss  Tier     Why
  1  Oakmont Supply          1,154,800   167    90%      42%  1,056,731  critical Pilot with under 40% of usage in production
  2  Stonebridge Labs          391,000    44    72%      44%    294,083  critical Three or more P1 tickets in the last 90 days
  3  Copperline Cloudworks   1,340,700     8    13%      33%    276,918  watch    Invoice 60+ days past due
```

## How a rep uses it

1. **Load the book.** Export open renewals from Salesforce or Gainsight as a CSV and upload it. Columns are matched automatically. Optionally upload past renewals with outcomes so the model learns your customers, or load a `model.json` RevOps trained.
2. **Read the top line.** ARR, expected ARR loss, GRR forecast range, critical and high counts, watermelons.
3. **Work the priority list.** Ranked by expected loss × renewal urgency. Filter by tier and by CSM.
4. **Open an account.** Top risk drivers in plain English, alerts, group scores, the recommended play, and a one-click brief for call prep.
5. **Leaders and RevOps** use the Portfolio tab (what drives risk across the book, watermelons), Model quality (accuracy against the scorecard, calibration) and Data quality (what mapped, what's missing).

Data stays on the machine running the app. The outbound calls are the ones you turn on: a data add-on reads from its tool (Zendesk, read-only), and the optional account brief sends that one account's computed facts to Claude when `ANTHROPIC_API_KEY` is set.

## Bring your own data

```bash
renewal-risk inspect my_renewals.csv --profile saas            # how columns map, what's missing
renewal-risk train past_renewals.csv --profile saas --out model.json   # train and evaluate
renewal-risk score my_renewals.csv --model model.json --out out/ --narrate 10
```

`score` writes `scored_accounts.csv`, `summary.json` and `report.md`. Without a model it scores from the scorecard alone and says so. `examples/` has Salesforce- and Gainsight-shaped exports and a column mapping file:

```bash
renewal-risk inspect examples/salesforce_report.csv --mapping examples/mapping.yaml
```

The minimum is an account ID, ARR and a renewal date. Every other signal is optional: missing measures are reweighted, and each account gets a confidence level so thin data is visible. For training, history needs an outcome column (renewed, contracted or churned; common CRM values like "Closed Lost" are recognized).

To tune a profile, copy its YAML from `src/renewal_risk/profiles/`, edit breakpoints, weights or alerts, and pass `--config my_profile.yaml`.

## Data add-ons

A CRM export rarely carries good support data. Add-ons fetch it from the source and merge it into the book before scoring. In the app, open **Data add-ons** in the sidebar, tick Zendesk, enter a subdomain, admin email and API token, and press **Test connection**. The synthetic demo has a simulated Zendesk switched on, so you can see it without an account. On the command line:

```bash
renewal-risk sources test --sources examples/sources.yaml
renewal-risk score my_renewals.csv --model model.json --sources examples/sources.yaml
```

The Zendesk add-on doesn't just copy ticket counts. It scores volume per 100 seats, compares each account with its own previous quarter, raises **support strain** when reopens or fix times climb, and raises **support gone quiet** when a formerly active account stops filing tickets while its usage falls. It honors Zendesk's rate limits, explains bad credentials in one line, and never prints a token. Other tools plug in the same way, as separate pip-installable packages. See [docs/add-ons.md](docs/add-ons.md) to set one up or write your own.

## Profiles

| Profile | For | Weighted toward |
|---|---|---|
| `ai_usage` (default) | AI APIs, copilots, agents, AI-native apps on credits, tokens or hybrid plans | Burn pacing both ways, pilot to production, embedding depth, seat compression, model retirement |
| `saas` | Per-seat subscriptions | Seat activation, active-user trend, adoption breadth, engagement (Gainsight DEAR) |
| `consumption` | Cloud, data platforms, compute and data-center capacity | Commit pacing and run rate, consumption trend, workload breadth |
| `hardware` | Devices and infrastructure with support contracts | Installed-base coverage, end-of-service-life exposure, failure rates, telemetry |
| `services` | Professional and managed services | Hours pacing, SLA attainment, milestones, delivery-lead stability |

## Results on synthetic data

Five-fold out-of-fold evaluation on 4,000 synthetic past renewals per profile.

| Profile | ROC AUC: scorecard / model / stacked | Brier: scorecard / stacked |
|---|---|---|
| AI products | 0.912 / 0.928 / **0.928** | 0.104 / **0.069** |
| Seat-based SaaS | 0.872 / 0.882 / **0.884** | 0.064 / **0.057** |
| Consumption and compute | 0.890 / 0.899 / **0.899** | 0.066 / **0.061** |
| Hardware and support | 0.896 / 0.936 / **0.935** | 0.064 / **0.048** |
| Professional services | 0.866 / 0.873 / **0.875** | 0.074 / **0.065** |

The model adds the most where interactions matter (hardware fleets, AI burn) and the least where the scorecard already captures the story. The stack keeps the scorecard's transparency and the model's calibration: Brier score, which measures whether a 30% probability really means 30%, improves on every profile.

Every synthetic book also carries five planted accounts that tests check are surfaced for the right reason. In the AI profile: a stalled pilot (critical, pilot-to-production play), an over-burner with 14 rate-limit hits (bill-shock alert), seat compression from 230 to 155 seats (repricing alert), a green account whose champion left as usage fell 35% on soon-retired models (alerts floor it to watch even though the model rates churn low), and a healthy control.

## How it works

```
CSV export ──> ingest ──> scorecard ──┐
 (any headers)  map, clean,  measures,  │──> stack ──> expected ARR loss ──> priority, tier, play
                derive       groups,    │     (logistic)  × renewal urgency
                             alerts     │
past renewals ──> LightGBM churn + ─────┘                └──> GRR forecast (Monte Carlo)
                  downsell, calibrated,                  └──> Claude brief (or rules)
                  TreeSHAP reasons
```

Full detail, design choices and sources are in [docs/methodology.md](docs/methodology.md).

## Project layout

```
src/renewal_risk/
  schema.py      canonical fields and header aliases
  ingest.py      load, map, clean and derive signals; data-quality report
  config.py      profile loading and validation
  profiles/      one YAML per industry profile
  scorecard.py   measures, groups, reweighting, bands, alerts
  model.py       LightGBM churn and downsell models, calibration, SHAP reasons, stacking
  pipeline.py    expected loss, urgency, tiers, watermelons, GRR forecast
  playbook.py    save plays
  narrate.py     Claude account briefs with rules fallback
  report.py      CSV, JSON and Markdown outputs
  synth.py       synthetic history and books for every profile
  cli.py         the renewal-risk command
  sources/       data add-ons: plug-in contract, HTTP client, registry, Zendesk
app/streamlit_app.py
tests/           ingest, scorecard, model, pipeline, add-ons, narratives, CLI and app smoke tests
```

## Host the demo

The app runs on [Streamlit Community Cloud](https://streamlit.io/cloud) with no secrets: it opens on synthetic data with a simulated Zendesk. Sign in there with GitHub, choose **Create app**, pick this repository, branch `main` and main file `app/streamlit_app.py`, and set Python 3.12 under Advanced settings. `app/requirements.txt` and `packages.txt` cover the dependencies.

## Development

```bash
pip install -e ".[app,claude,dev]"
ruff check . && ruff format --check . && mypy src && pytest
```

CI runs lint, type checks, tests and the end-to-end demo for every profile on Python 3.10, 3.12 and 3.13.

## Limits

- Synthetic accuracy is not a benchmark. Real labels are noisier; expect lower numbers and retrain on your outcomes.
- Breakpoints and weights are informed defaults, not published standards. Tune them.
- The GRR forecast treats accounts as independent, so it understates correlated shocks.
- The model ranks risk. It doesn't prove a play works; track play outcomes for that.

## License

MIT
