# Data add-ons

Add-ons pull signals from tools other than your CRM export into the book before it is scored. Zendesk is the first. Each add-on is read-only, runs only when you turn it on, and can only add information: if it can't match an account, that account keeps what your export said.

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

Every request goes through a swappable `transport(url, headers, timeout) -> (status, headers, body)`. Pass a fake one to test pagination, rate limits and bad credentials offline, as `tests/test_sources.py` does for Zendesk. For a demo mode, set `supports_simulation = True` and serve fake API responses built from the book, as `SimulatedZendesk` does, so the demo runs your real parsing code.
