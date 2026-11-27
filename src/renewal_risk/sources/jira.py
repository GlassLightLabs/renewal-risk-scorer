"""Jira add-on: the engineering side of a customer's experience, from issues linked to each account.

Support tools show how often a customer asks for help. Jira shows what happens after support hands a problem to
engineering: bugs that stay open for months and escalations that pull in developers. Renewal teams hear about
these late, usually from an angry champion. This add-on brings them in early.

Two ways in, both read-only:

* **An issue export** (Filters > Export > Export CSV (all fields)), with no token at all.
* **The Jira Cloud REST API** with an email and API token, using the ``/rest/api/3/search/jql`` endpoint.

Either way, each issue needs a field that names the customer: a Jira Service Management "Organizations" field,
a custom "Customer" or "Account ID" field, or similar. Issues with an empty customer field are skipped.

What it returns per matched account (an account is matched when at least one issue names it):

========================  =================================================================================
open_bugs                 customer bugs (issue type Bug or Defect by default) that are not done
open_bug_age_days         median age in days of those open bugs (blank when none are open)
escalations_90d           issues created in the last 90 days with an escalation label or top priority
========================  =================================================================================

Accounts with no linked issue are left blank rather than assumed clean, because a missing customer field and a
bug-free customer look the same from here.
"""

from __future__ import annotations

import base64
import re
import time
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .base import ConfigField, DataSource, SourceError, SourceResult
from .exports import (
    MATCH_FIELD,
    days_since,
    end_of,
    match_records,
    normalize_export,
    read_export,
    timestamps,
)
from .http import JsonClient, Transport

WINDOW_DAYS = 90
LOOKBACK_DAYS = 180
MAX_PAGES = 500
PAGE_SIZE = 100
DONE_STATUSES = {"done", "closed", "resolved", "won't do", "wont do", "won't fix", "cancelled", "canceled", "rejected"}
_SITE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]+)+$")
_FIELD_ID = re.compile(r"^(customfield_\d+|labels|components|[a-z][a-zA-Z0-9_]*)$")

# Column names in Jira's CSV export ("Export CSV (all fields)"), and the API field each one comes from.
COLUMNS: dict[str, tuple[str, ...]] = {
    "key": ("Issue key", "key"),
    "issue_type": ("Issue Type", "issuetype", "type"),
    "status": ("Status",),
    "status_category": ("Status Category", "statuscategory"),
    "priority": ("Priority",),
    "created": ("Created", "created date"),
    "resolved": ("Resolved", "resolutiondate", "resolution date"),
    "labels": ("Labels",),
    "customer": (
        "Custom field (Customer)",
        "Custom field (Organizations)",
        "Custom field (Account ID)",
        "Custom field (Account)",
        "Custom field (Salesforce Account ID)",
        "Custom field (Company)",
        "Organizations",
        "Organization",
        "Customer",
        "Customer name",
        "Account ID",
        "Account",
        "Account name",
        "Company",
    ),
}


def clean_site(raw: str) -> str:
    """Accept 'acme', 'acme.atlassian.net' or 'https://acme.atlassian.net/jira/...'; return the host or raise."""
    s = str(raw or "").strip().lower()
    s = re.sub(r"^https?://", "", s).split("/")[0]
    if s and "." not in s:
        s = f"{s}.atlassian.net"
    if not _SITE.match(s):
        raise SourceError("config", "Jira: the site should look like 'acme.atlassian.net'.")
    return s


def _csv_list(raw: Any, default: str) -> tuple[str, ...]:
    return tuple(p.strip().lower() for p in str(raw if raw not in (None, "") else default).split(",") if p.strip())


class JiraSource(DataSource):
    name = "jira"
    label = "Jira"
    description = "Open customer bugs, how long they have been open, and engineering escalations per account."
    supports_simulation = True
    fields = (
        ConfigField(
            "file",
            "Issue export (CSV)",
            required=False,
            help="Path to a Jira issue export (Filters > Export CSV, all fields). Or fill in the API settings.",
        ),
        ConfigField("site", "Jira site", required=False, help="For the API: e.g. acme.atlassian.net"),
        ConfigField("email", "Email", required=False, help="For the API: the Atlassian account that owns the token"),
        ConfigField(
            "api_token",
            "API token",
            secret=True,
            required=False,
            help="For the API: id.atlassian.com > Security > API tokens. Read access is all it uses.",
        ),
        ConfigField(
            "customer_field",
            "Field naming the customer",
            required=False,
            help="Export: the column header. API: the field ID, e.g. customfield_10050. Blank detects common names.",
        ),
        ConfigField(
            "jql",
            "Limit to (JQL)",
            required=False,
            default="",
            help="Optional, API only, e.g. project in (SUP, ENG). Issues created in the last 180 days or still open.",
        ),
        MATCH_FIELD,
        ConfigField(
            "bug_types", "Issue types counted as bugs", required=False, default="Bug,Defect", help="Comma-separated"
        ),
        ConfigField(
            "escalation_labels",
            "Labels that mark an escalation",
            required=False,
            default="escalation,customer-escalation,escalated",
            help="Comma-separated",
        ),
        ConfigField(
            "escalation_priorities",
            "Priorities that count as an escalation",
            required=False,
            default="Highest,Blocker",
            help="Comma-separated",
        ),
    )
    provides = ("open_bugs", "open_bug_age_days", "escalations_90d")

    def __init__(self, transport: Transport | None = None, sleep: Callable[[float], None] = time.sleep):
        self._transport = transport
        self._sleep = sleep

    # -- settings ---------------------------------------------------------------------------------------------
    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        cfg = super().validate(config)
        cfg["bug_types"] = _csv_list(cfg.get("bug_types"), "Bug,Defect")
        cfg["escalation_labels"] = _csv_list(cfg.get("escalation_labels"), "escalation,customer-escalation,escalated")
        cfg["escalation_priorities"] = _csv_list(cfg.get("escalation_priorities"), "Highest,Blocker")
        cfg["customer_field"] = str(cfg.get("customer_field") or "").strip()
        if cfg["simulate"]:
            cfg["mode"] = "simulate"
            return cfg
        api = [cfg.get(k) for k in ("site", "email", "api_token")]
        if cfg.get("file"):
            cfg["mode"] = "file"
        elif any(api):
            if not all(api):
                raise SourceError("config", "Jira: the API needs the site, the email and an API token.")
            cfg["site"] = clean_site(str(cfg["site"]))
            if "@" not in str(cfg["email"]):
                raise SourceError("config", "Jira: 'Email' should be an email address.")
            if not cfg["customer_field"] or not _FIELD_ID.match(cfg["customer_field"]):
                raise SourceError(
                    "config",
                    "Jira: for the API, set 'Field naming the customer' to a field ID such as customfield_10050 "
                    "(Settings > Issues > Custom fields shows the ID in each field's link).",
                )
            cfg["mode"] = "api"
        else:
            raise SourceError("config", "Jira: add an issue export file, or the site, email and API token.")
        return cfg

    def _client(self, cfg: dict[str, Any]) -> JsonClient:
        raw = f"{cfg['email']}:{cfg['api_token']}".encode()
        return JsonClient(
            f"https://{cfg['site']}",
            headers={"Authorization": "Basic " + base64.b64encode(raw).decode()},
            transport=self._transport,
            sleep=self._sleep,
            service="Jira",
        )

    def test_connection(self, config: dict[str, Any]) -> str:
        cfg = self.validate(config)
        if cfg["mode"] == "simulate":
            return "Jira: simulated demo data, no account needed."
        if cfg["mode"] == "file":
            issues = self._read_file(cfg)
            named = int(issues["customer"].notna().sum()) if "customer" in issues else 0
            return f"Jira export read: {len(issues):,} issues, {named:,} of them name a customer."
        me = self._client(cfg).get("rest/api/3/myself")
        return f"Connected to {cfg['site']} as {me.get('displayName') or me.get('emailAddress') or 'your user'}."

    # -- fetch ------------------------------------------------------------------------------------------------
    def fetch(self, config: dict[str, Any], accounts: pd.DataFrame, as_of: date) -> SourceResult:
        cfg = config if "mode" in config else self.validate(config)
        warnings: list[str] = []
        if cfg["mode"] == "simulate":
            issues = normalize_export(simulated_issues(accounts, as_of), COLUMNS, service="Jira")
        elif cfg["mode"] == "file":
            issues = self._read_file(cfg)
        else:
            issues = self._read_api(cfg, as_of, warnings)

        if "customer" not in issues.columns:
            raise SourceError(
                "config",
                "Jira: no column names the customer. Set 'Field naming the customer' to the column header.",
            )
        named = issues[issues["customer"].notna()].copy()
        skipped = len(issues) - len(named)
        if skipped:
            warnings.append(f"{skipped:,} issues name no customer and were not counted.")
        named["account_ref"] = named["customer"].astype(str).str.strip()
        named["name"] = named["account_ref"]
        acct, unmatched_customers, match_warnings = match_records(
            named, accounts, mode=cfg["match_on"], service="Jira customer", one_per_account=False
        )
        warnings += match_warnings
        named["account_id"] = acct
        signals = compute_signals(named[named["account_id"].notna()], as_of, cfg)

        ids = accounts["account_id"].astype(str)
        matched = set(signals.index)
        unmatched = [a for a in ids if a not in matched]
        if not matched and len(named):
            warnings.append("No Jira customer matched an account. Check 'Field naming the customer' and 'Match by'.")
        return SourceResult(
            source=self.label,
            signals=signals,
            matched_accounts=len(matched & set(ids)),
            unmatched_accounts=unmatched,
            unmatched_records=unmatched_customers,
            records_read=len(issues),
            warnings=warnings,
            simulated=cfg["mode"] == "simulate",
            records_label="issues",
            unmatched_label="customers named in Jira",
        )

    def _read_file(self, cfg: dict[str, Any]) -> pd.DataFrame:
        extra = {cfg["customer_field"]: "customer"} if cfg["customer_field"] else None
        return read_export(
            cfg["file"], COLUMNS, service="Jira", required=("created", "issue_type"), extra_aliases=extra
        )

    def _read_api(self, cfg: dict[str, Any], as_of: date, warnings: list[str]) -> pd.DataFrame:
        client = self._client(cfg)
        scope = f"created >= -{LOOKBACK_DAYS}d OR statusCategory != Done"
        jql = f"({cfg['jql']}) AND ({scope})" if cfg.get("jql") else scope
        field = cfg["customer_field"]
        params: dict[str, Any] = {
            "jql": jql + " ORDER BY created DESC",
            "fields": f"issuetype,status,priority,created,resolutiondate,labels,{field}",
            "maxResults": PAGE_SIZE,
        }
        rows: list[dict[str, Any]] = []
        for _ in range(MAX_PAGES):
            page = client.get("rest/api/3/search/jql", params)
            for issue in page.get("issues") or []:
                if isinstance(issue, dict):
                    rows.append(_flatten(issue, field))
            token = page.get("nextPageToken")
            if page.get("isLast", True) or not token:
                break
            params = {**params, "nextPageToken": token}
        else:
            warnings.append(f"Stopped after {len(rows):,} issues; some were not read.")
        raw = pd.DataFrame(rows, columns=list(COLUMNS))
        return normalize_export(raw, COLUMNS, service="Jira")


def _field_text(value: Any) -> str | None:
    """A customer field as text, whatever its type: text, a select option, an organization list or labels."""
    if value is None:
        return None
    if isinstance(value, list):
        for v in value:
            text = _field_text(v)
            if text:
                return text
        return None
    if isinstance(value, dict):
        for k in ("value", "name", "displayName", "id"):
            if value.get(k) not in (None, ""):
                return str(value[k])
        return None
    text = str(value).strip()
    return text or None


def _flatten(issue: dict[str, Any], customer_field: str) -> dict[str, Any]:
    f = issue.get("fields") or {}
    status = f.get("status") or {}
    return {
        "key": issue.get("key"),
        "issue_type": (f.get("issuetype") or {}).get("name"),
        "status": status.get("name"),
        "status_category": (status.get("statusCategory") or {}).get("key"),
        "priority": (f.get("priority") or {}).get("name"),
        "created": f.get("created"),
        "resolved": f.get("resolutiondate"),
        "labels": ",".join(str(x) for x in f.get("labels") or []),
        "customer": _field_text(f.get(customer_field)),
    }


# -- metrics --------------------------------------------------------------------------------------------------
def _is_done(issues: pd.DataFrame) -> pd.Series:
    done = pd.Series(False, index=issues.index)
    if "status_category" in issues.columns:
        cat = issues["status_category"].astype(str).str.strip().str.lower()
        known = issues["status_category"].notna()
        done |= known & cat.isin({"done", "complete", "completed"})
        unknown = ~known
    else:
        unknown = pd.Series(True, index=issues.index)
    fallback = pd.Series(False, index=issues.index)
    if "resolved" in issues.columns:
        fallback |= timestamps(issues["resolved"]).notna()
    if "status" in issues.columns:
        fallback |= issues["status"].astype(str).str.strip().str.lower().isin(DONE_STATUSES)
    return done | (unknown & fallback)


def compute_signals(issues: pd.DataFrame, as_of: date, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-account engineering signals from issues that already carry an ``account_id``."""
    cols = list(JiraSource.provides)
    accounts = sorted(set(issues["account_id"].astype(str))) if len(issues) else []
    out = pd.DataFrame(index=pd.Index(accounts, name="account_id"), columns=cols, dtype=float)
    if not accounts:
        return out
    df = issues.copy()
    df["account_id"] = df["account_id"].astype(str)
    df["created_ts"] = timestamps(df["created"])
    end = end_of(as_of)
    df = df[df["created_ts"].isna() | (df["created_ts"] < end)]
    df["done"] = _is_done(df)
    df["type"] = df.get("issue_type", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    df["prio"] = df.get("priority", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    labels = df.get("labels", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    df["label_set"] = labels.map(lambda s: {p.strip() for p in re.split(r"[,;\s]+", s) if p.strip()})

    bugs_open = df[df["type"].isin(cfg["bug_types"]) & ~df["done"]].copy()
    out["open_bugs"] = bugs_open.groupby("account_id").size().reindex(accounts).fillna(0).astype(float)
    bugs_open["age"] = days_since(bugs_open["created"], as_of)
    age = bugs_open.dropna(subset=["age"]).groupby("account_id")["age"].median()
    out["open_bug_age_days"] = age.round(0).reindex(accounts)

    recent = df[df["created_ts"] >= end - pd.Timedelta(days=WINDOW_DAYS)]
    esc_labels = set(cfg["escalation_labels"])
    escalated = recent["prio"].isin(cfg["escalation_priorities"]) | recent["label_set"].map(
        lambda s: bool(s & esc_labels)
    )
    out["escalations_90d"] = recent[escalated].groupby("account_id").size().reindex(accounts).fillna(0).astype(float)
    return out


# -- simulated export -----------------------------------------------------------------------------------------
def simulated_issues(accounts: pd.DataFrame | None, as_of: date, *, seed: int = 23) -> pd.DataFrame:
    """A Jira issue export built from the book, in Jira's own column names, for demos and tests.

    Accounts with urgent tickets or poor CSAT get more bugs and escalations, so the demo tells a coherent story.
    About a fifth of accounts have no linked issues, and a few issues name the customer by company name only.
    """
    header = {k: v[0] for k, v in COLUMNS.items()}
    if accounts is None or not len(accounts):
        return pd.DataFrame(columns=list(header.values()))
    rng = np.random.default_rng(seed)
    acc = accounts.reset_index(drop=True)
    end = end_of(as_of)

    def col(name: str, default: float) -> np.ndarray:
        if name in acc.columns:
            return pd.to_numeric(acc[name], errors="coerce").fillna(default).to_numpy(float)
        return np.full(len(acc), default)

    p1, csat = col("p1_tickets_90d", 0), col("csat", 88)
    names = acc["account_name"].astype(str) if "account_name" in acc.columns else acc["account_id"].astype(str)
    rows: list[dict[str, Any]] = []
    n = 0
    for i in range(len(acc)):
        if rng.random() < 0.2:
            continue
        pressure = max(0.0, (88 - csat[i]) / 10) + p1[i] * 0.8
        by_name = rng.random() < 0.1
        ref = names[i] if by_name else str(acc.loc[i, "account_id"])
        for _ in range(int(rng.poisson(1 + pressure * 1.5))):
            n += 1
            age = float(rng.uniform(1, LOOKBACK_DAYS)) * (1.4 if pressure > 2 else 1.0)
            created = end - pd.Timedelta(days=min(age, 400))
            bug = rng.random() < 0.6
            done = rng.random() < (0.75 if pressure < 2 else 0.45)
            escalate = rng.random() < min(0.6, 0.04 + pressure * 0.08)
            resolved = created + pd.Timedelta(days=float(rng.uniform(1, 30))) if done else None
            rows.append(
                {
                    header["key"]: f"ENG-{1000 + n}",
                    header["issue_type"]: "Bug" if bug else str(rng.choice(["Task", "Story", "Improvement"])),
                    header["status"]: "Done" if done else str(rng.choice(["To Do", "In Progress", "In Review"])),
                    header["status_category"]: "Done" if done else ("In Progress" if rng.random() < 0.5 else "To Do"),
                    header["priority"]: "Highest" if escalate and rng.random() < 0.5 else "Medium",
                    header["created"]: created.strftime("%d/%b/%y %I:%M %p"),
                    header["resolved"]: resolved.strftime("%d/%b/%y %I:%M %p") if resolved is not None else "",
                    header["labels"]: "escalation" if escalate else "",
                    header["customer"]: ref,
                }
            )
    internal = {v: "" for v in header.values()}  # an issue for a customer that is not in the book
    internal.update(
        {
            header["key"]: "ENG-1",
            header["issue_type"]: "Bug",
            header["created"]: (end - pd.Timedelta(days=3)).strftime("%d/%b/%y %I:%M %p"),
            header["customer"]: "Internal QA",
        }
    )
    rows.append(internal)
    return pd.DataFrame(rows, columns=list(header.values()))
