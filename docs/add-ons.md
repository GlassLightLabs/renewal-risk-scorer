# Data add-ons

Add-ons pull signals from tools other than your CRM export into the book before it is scored. Five are built in: Zendesk, Jira, HubSpot, ChurnZero and Pendo. Each add-on is read-only, runs only when you turn it on, and can only add information: if it can't match an account, that account keeps what your export said.

| Add-on | How it reads | Signals it fills |
|---|---|---|
| [Zendesk](#zendesk) | API (admin token) | `tickets_90d`, `tickets_90d_prior`, `p1_tickets_90d`, `csat`, `ticket_reopen_rate_pct`, `ticket_resolution_hours` |
| [Jira](#jira) | Issue export (CSV) or API (API token) | `open_bugs`, `open_bug_age_days`, `escalations_90d` |
| [HubSpot](#hubspot) | Companies and deals exports (CSV) or API (private app token) | `days_since_last_meeting`, `price_increase_pct`, `competitor_eval`, `discount_pct` |
| [ChurnZero](#churnzero) | Account export (CSV) | `active_users`, `active_users_prior`, `days_since_last_login`, `nps`, `csm_sentiment`, `exec_sponsor`, `champion_change_90d` |
| [Pendo](#pendo) | Account report (CSV) | `active_users`, `active_users_prior`, `days_since_last_login`, `feature_adoption_pct`, `nps` |

Every add-on also has a **simulated demo mode** (`simulate: true`) that builds fake data from the book and runs it through the add-on's real parsing code, so you can try it with no account. When two add-ons fill the same signal (ChurnZero and Pendo both count active users), the one listed later in the settings file wins.

## Using an add-on

### In the app

Open **Data add-ons** in the sidebar.

1. Tick **Use Zendesk**.
2. On the synthetic demo, **Simulated demo data** is on, so you can see the add-on work without a Zendesk account. Turn it off to connect your own.
3. Fill in the subdomain (the `acme` in `acme.zendesk.com`), the admin email and an API token. The token box hides what you type.
4. Press **Test connection**. You'll see who you're connected as, or a plain message saying what to fix: a wrong token, a subdomain that doesn't exist, or a rate limit.
5. Scores refresh with the Zendesk signals in. The **Data quality** tab shows how many accounts matched, which didn't and any notes.

Zendesk is read again at most every 15 minutes, so clicking around the app doesn't use up your API allowance.

When you host the app, you can keep credentials out of the form by adding them to `.streamlit/secrets.toml` (or the host's secrets settings):

```toml
[zendesk]
subdomain = "acme"
email = "admin@acme.com"
api_token = "..."
```

### On the command line

Write a settings file, keeping secrets in environment variables:

```yaml
# sources.yaml
sources:
  zendesk:
    subdomain: acme
    email: ${ZENDESK_EMAIL}
    api_token: ${ZENDESK_API_TOKEN}
    match_on: auto
```

```bash
export ZENDESK_EMAIL=admin@acme.com ZENDESK_API_TOKEN=...
renewal-risk sources list                                    # what's installed
renewal-risk sources test --sources sources.yaml             # check settings and credentials
renewal-risk score book.csv --model model.json --sources sources.yaml
renewal-risk demo --profile saas --sources examples/sources.demo.yaml   # simulated Zendesk, no account needed
```

The four export-based add-ons need no credentials at all. `examples/sources.exports.yaml` points them at synthetic sample exports that match the accounts in `examples/salesforce_report.csv`:

```bash
renewal-risk inspect examples/salesforce_report.csv --mapping examples/mapping.yaml \
  --as-of 2026-12-01 --sources examples/sources.exports.yaml
renewal-risk demo --profile saas --sources examples/sources.demo.yaml   # all five simulated
```

`score`, `inspect` and `demo` print one line per add-on (accounts matched, records read) and any notes. An add-on that fails is reported and skipped: scoring always finishes.

## Zendesk

**Token.** In Zendesk, go to Admin Center, then Apps and integrations, then APIs, then Zendesk API, and add an API token. Reading tickets in bulk needs an admin's token. The add-on only reads.

**What it reads.** Organizations, and tickets created in the last 180 days from the incremental export API, with each ticket's metrics.

**What it fills in, per account** (the as-of date counts as today):

| Signal | Meaning |
|---|---|
| `tickets_90d` | Tickets created in the last 90 days |
| `tickets_90d_prior` | Tickets in the 90 days before that |
| `p1_tickets_90d` | Of those, tickets at urgent priority (set `urgent_priorities: urgent,high` to count high too) |
| `csat` | Share of satisfaction ratings that were "good", when at least 3 were given |
| `ticket_reopen_rate_pct` | Share of solved tickets reopened at least once |
| `ticket_resolution_hours` | Median calendar hours from creation to full resolution |

Rates need at least three tickets behind them. Below that they are left blank rather than swinging on one ticket, and any CSAT your export already had is kept.

**What the scorer does with them.** Raw counts punish big customers for being big, so volume is scored per 100 seats. Each account is also compared with its own previous quarter:

- **Support strain** (alert): a quarter or more of tickets reopened, or a median fix time over five days, on three or more tickets.
- **Support gone quiet** (alert): an account that filed six or more tickets last quarter has nearly stopped, while product usage fell. A sudden silence usually means people stopped trying, not that the product started working.
- Reopen rate and resolution time are scored as measures in each profile's support group.

**Matching organizations to accounts.** `match_on` decides how:

- `auto` (default): the organization's external ID equals the account ID, otherwise the company name matches.
- `external_id`: external ID only.
- `name`: company name only. "Acme, Inc." matches "ACME"; suffixes like Inc, LLC and GmbH are ignored. Two organizations or accounts with the same name are never guessed; they are listed as unmatched.
- `field:<key>`: a custom organization field holding the account ID, such as `field:salesforce_id`.

**Limits it handles for you.** Zendesk allows 10 incremental export requests a minute. When it says slow down, the add-on waits as long as Zendesk asks (up to 90 seconds) and retries, up to four times, and backs off on server errors. Bad credentials, a wrong subdomain and timeouts come back as one-line messages. Credentials are never printed, logged or put in a URL.

## Working from CSV exports

Jira, HubSpot, ChurnZero and Pendo can each read the CSV their tool exports, which is often quicker than getting an API token approved. What they have in common:

- **Columns are found by name.** Each add-on knows the headers its tool uses ("Last Contacted", "Number of Active Visitors"...) and common variants. Case and punctuation don't matter. Columns it doesn't need are ignored. If a required column is missing, the message names it and lists the columns it did find.
- **Rows are matched to accounts** by an account ID column first, then by company name, the same way Zendesk organizations are. Set `match_on` to `id` or `name` to use only one. If your ID lives in an unusual column, name it with `account_id_column`. Two records or accounts with the same name are never guessed.
- **Dates** can be ISO, US style (`11/05/2026`), Jira style (`26/Nov/26 3:12 PM`) or epoch milliseconds. "Days since" counts calendar days to the as-of date.
- **Several records on one account** (a parent and a subsidiary in Pendo, say) are combined: users add up, the most recent login wins.
- In the app, the export add-ons take a file path, so they suit a local run (`streamlit run app/streamlit_app.py`). On a hosted demo, use their simulated mode.

## Jira

**What it's for.** Support tools show how often a customer asks for help. Jira shows what happens after support hands a problem to engineering: bugs that sit open for months and escalations that pull in developers. Renewal teams usually hear about these late.

**Two ways in.** An issue export (in Jira, run a filter, then Export, then Export CSV (all fields)), or the Jira Cloud API with your site, email and an API token from id.atlassian.com (Security, then API tokens). The API reads issues created in the last 180 days or still open, through `/rest/api/3/search/jql`. Narrow it with `jql`, for example `project in (SUP, ENG)`.

**Naming the customer.** Each issue needs a field that says which customer it belongs to: a Jira Service Management Organizations field, or a custom field such as Customer or Account ID. In an export, common column names are detected; set `customer_field` to the header if yours differs. For the API, set `customer_field` to the field ID (for example `customfield_10050`; the ID is in the field's link under Settings, Issues, Custom fields). Issues with no customer are skipped and counted in a note.

```yaml
sources:
  jira:
    file: exports/jira_issues.csv        # or: site, email, api_token
    customer_field: ""                   # blank detects Customer, Organizations, Account ID...
    bug_types: Bug,Defect
    escalation_labels: escalation,customer-escalation,escalated
    escalation_priorities: Highest,Blocker
```

| Signal | Meaning |
|---|---|
| `open_bugs` | Issues of a bug type (Bug, Defect by default) that are not done |
| `open_bug_age_days` | Median age in days of those open bugs; blank when none are open |
| `escalations_90d` | Issues of any type created in the last 90 days with an escalation label or top priority |

An issue counts as done when its status category is Done, or, in an export without that column, when it has a resolved date or a status such as Done, Closed or Resolved. Accounts with no linked issue are left blank rather than assumed clean, because a missing customer field and a bug-free customer look the same.

**New alerts.** `engineering_escalations` (high): two or more escalations in 90 days. `aging_bugs` (medium): three or more open bugs with a median age of 60 days or more. The three signals are also accepted as columns in your own export.

## HubSpot

**What it's for.** Many CS and account teams log calls and emails and run renewals as deals in HubSpot, even when another system holds the contract.

**Two ways in.** A companies export, optionally with a deals export (include the "Associated Company IDs" column), or the CRM API with a private app access token that has the companies and deals read scopes.

```yaml
sources:
  hubspot:
    companies_file: exports/hubspot_companies.csv   # or: access_token: ${HUBSPOT_TOKEN}
    deals_file: exports/hubspot_deals.csv           # optional
    account_id_column: salesforce_account_id        # the company property holding your account ID
    renewal_keyword: renewal                        # deals whose name, pipeline or type contain this
    competitor_property: ""                         # blank detects a "Competitor" property
    discount_property: ""                           # blank detects a "Discount" property
```

| Signal | Meaning |
|---|---|
| `days_since_last_meeting` | Days since HubSpot's Last Contacted date: the last logged call, meeting or email |
| `price_increase_pct` | The open renewal deal closing soonest, versus the account's current ARR, as a percent change |
| `competitor_eval` | 1 when the competitor property is filled in on the company or its open renewal deal |
| `discount_pct` | The discount property on the most recent won deal (0.25 and 25 both read as 25%) |

A renewal deal is one whose name, pipeline or type contains the renewal keyword, or whose type is Existing Business. A renewal quoted below current ARR comes out negative: that's a downsell already in the forecast. Deals reach accounts through their company (by HubSpot company ID, or by company name when there is no ID).

## ChurnZero

**What it's for.** ChurnZero already tracks logins, active contacts and NPS per account. The add-on reads an account grid export (choose your columns, then Export), so it needs no API access.

```yaml
sources:
  churnzero:
    file: exports/churnzero_accounts.csv
    account_id_column: ""     # blank detects Account External ID, CRM ID, Account ID...
```

| Signal | From columns like |
|---|---|
| `active_users` | Active Contacts (Last 30 Days), Active Users |
| `active_users_prior` | Active Contacts (Prior Period): gives the usage trend |
| `days_since_last_login` | Last Login Date, Last Activity Date, or Days Since Last Login |
| `nps` | Last NPS Score. Account-level NPS on the -100 to 100 scale is converted to 0 to 10 |
| `csm_sentiment` | CSM Sentiment, as 1 to 5 or Red, Yellow, Green |
| `exec_sponsor`, `champion_change_90d` | Executive Sponsor, Champion Left (yes or no) |

ChurnZero's own ChurnScore is ignored on purpose. The scorer builds an explainable score from the underlying signals; adding another vendor's score as one more input would hide the reasons behind the number.

## Pendo

**What it's for.** Who actually uses the product: active visitors, the last visit and how many of your core features each account touches. The add-on reads an account report downloaded as CSV from Data Explorer or an Accounts report.

```yaml
sources:
  pendo:
    file: exports/pendo_accounts.csv
    account_id_column: ""     # blank uses Account ID; most teams pass the CRM ID to Pendo
    core_features: 20         # only if the report has "Core Features Used" without a total
```

| Signal | From columns like |
|---|---|
| `active_users` | Number of Active Visitors, Active Visitors (Last 30 Days) |
| `active_users_prior` | Active Visitors (Prior Period): gives the usage trend |
| `days_since_last_login` | Last Visit, or Days Since Last Visit |
| `feature_adoption_pct` | Core Feature Adoption %, or Core Features Used out of Core Features Total |
| `nps` | NPS Score from Pendo NPS polls, on the 0 to 10 scale |

If Pendo account IDs aren't your CRM IDs, match on name, or point `account_id_column` at the metadata column that holds the CRM ID.

## Writing your own add-on

An add-on is a class that says what settings it needs and which canonical fields it fills, and returns per-account values.

```python
from datetime import date

import pandas as pd

from renewal_risk.sources import ConfigField, DataSource, SourceResult
from renewal_risk.sources.http import JsonClient


class StatuspageSource(DataSource):
    name = "statuspage"
    label = "Statuspage"
    description = "Incidents affecting each customer in the last 90 days."
    fields = (
        ConfigField("page_id", "Page ID"),
        ConfigField("api_key", "API key", secret=True),
    )
    provides = ("incidents_90d",)

    def fetch(self, config: dict, accounts: pd.DataFrame, as_of: date) -> SourceResult:
        client = JsonClient(
            "https://api.statuspage.io",
            headers={"Authorization": f"OAuth {config['api_key']}"},
            service="Statuspage",
        )
        data = client.get(f"v1/pages/{config['page_id']}/incidents")
        counts = ...  # turn `data` into {account_id: count}
        signals = pd.DataFrame({"incidents_90d": counts}).rename_axis("account_id")
        return SourceResult(
            source=self.label,
            signals=signals,
            matched_accounts=len(signals),
            unmatched_accounts=[a for a in accounts["account_id"] if a not in signals.index],
        )
```

What you get for free:

- **A settings form.** The app draws one from `fields`: secrets get a password box, `kind="choice"` gets a dropdown, and **Test connection** calls your `test_connection` (override it with one cheap API call).
- **Validation.** `validate` fills defaults and rejects missing required settings with a message naming the field.
- **A careful HTTP client.** `JsonClient` handles timeouts, `Retry-After` on 429, backoff on 5xx, clear errors for 401, 403 and 404, and refuses to follow links to another host.
- **Errors people can act on.** Raise `SourceError(kind, message)` with `kind` one of `auth`, `rate_limit`, `network`, `config`, `not_found` or `bad_response`. The app and CLI show the message with a hint, and other add-ons keep running.
- **Safe merging.** Matched accounts get your values, blanks never overwrite existing data, and derived signals are recomputed.

A new signal needs a canonical field in `schema.py`, and a measure or alert in a profile if it should affect the score. Until then it is carried in the output but not scored.

### Packaging it as a plug-in

Ship the class in its own package and register it under the `renewal_risk.sources` entry-point group:

```toml
# pyproject.toml of your package
[project.entry-points."renewal_risk.sources"]
statuspage = "renewal_risk_statuspage:StatuspageSource"
```

After `pip install renewal-risk-statuspage`, it appears in `renewal-risk sources list` and in the app's Data add-ons panel. No change to this project is needed. A plug-in that fails to import is reported as a warning and never stops scoring.

### Testing without a network

Every request goes through a swappable `transport(url, headers, timeout) -> (status, headers, body)`. Pass a fake one to test pagination, rate limits and bad credentials offline, as `tests/test_sources.py` does for Zendesk and `tests/test_add_ons.py` does for Jira and HubSpot. An add-on that reads a CSV export can use `renewal_risk.sources.exports` (`read_export`, `match_records`) for column detection and account matching. For a demo mode, set `supports_simulation = True` and serve fake API responses built from the book, as `SimulatedZendesk` does, so the demo runs your real parsing code.
