"""ChurnZero add-on: usage, NPS and relationship fields from a ChurnZero account export.

ChurnZero already tracks logins, active contacts and NPS responses for each account. Rather than ask for a
second login to ChurnZero's API, this add-on reads the account export a CS admin can pull from any account grid
(Accounts > choose columns > Export), so it works on day one with no token.

What it returns per matched account, from whichever of these columns the export carries:

========================  =================================================================================
active_users              active contacts or users in the last 30 days
active_users_prior        the same count from about 90 days earlier, when the export has it (gives the trend)
days_since_last_login     days since the account's last login or activity date
nps                       latest NPS response on the 0 to 10 scale (account-level -100 to 100 is converted)
csm_sentiment             a sentiment field, 1 to 5 or Red, Yellow, Green
exec_sponsor              an executive sponsor flag
champion_change_90d       a champion change flag
========================  =================================================================================

ChurnZero's own ChurnScore is deliberately not used. This tool builds its own explainable score from the
underlying signals; feeding another vendor's score in as one more input would hide the reasons behind a number.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..ingest import parse_sentiment
from .base import ConfigField, DataSource, SourceResult
from .exports import (
    MATCH_FIELD,
    account_id_field,
    days_since,
    end_of,
    flags,
    match_records,
    normalize_export,
    numbers,
    read_export,
)

COLUMNS: dict[str, tuple[str, ...]] = {
    "account_ref": ("Account External ID", "AccountExternalId", "External ID", "CRM ID", "Account ID", "Salesforce ID"),
    "name": ("Account Name", "Name", "Account"),
    "active_users": (
        "Active Contacts (Last 30 Days)",
        "Active Users (Last 30 Days)",
        "Active Users (30 Days)",
        "Active Contacts",
        "Active Users",
        "Unique Users (Last 30 Days)",
    ),
    "active_users_prior": (
        "Active Contacts (Prior Period)",
        "Active Users (Prior Period)",
        "Active Users 90 Days Ago",
        "Active Contacts 90 Days Ago",
    ),
    "last_login": ("Last Login Date", "Last Login", "Last Activity Date", "Last Active Date", "Last Activity"),
    "days_since_login": ("Days Since Last Login", "Days Since Last Activity", "Days Inactive"),
    "nps": ("Last NPS Score", "Latest NPS Score", "NPS Score", "NPS", "Last NPS Response"),
    "sentiment": ("CSM Sentiment", "Sentiment", "Account Sentiment"),
    "exec_sponsor": ("Executive Sponsor", "Exec Sponsor", "Has Executive Sponsor"),
    "champion_change": ("Champion Left (90 Days)", "Champion Changed", "Champion Left", "Champion Change"),
}


class ChurnZeroSource(DataSource):
    name = "churnzero"
    label = "ChurnZero"
    description = "Active users and trend, last login, NPS, sentiment and stakeholder flags from a ChurnZero export."
    supports_simulation = True
    fields = (
        ConfigField("file", "Account export (CSV)", help="Path to a ChurnZero account grid export saved as CSV."),
        account_id_field("Salesforce Account ID"),
        MATCH_FIELD,
    )
    provides = (
        "active_users",
        "active_users_prior",
        "days_since_last_login",
        "nps",
        "csm_sentiment",
        "exec_sponsor",
        "champion_change_90d",
    )

    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        cfg = super().validate(config)
        cfg["account_id_column"] = str(cfg.get("account_id_column") or "").strip()
        return cfg

    def _read(self, cfg: dict[str, Any], accounts: pd.DataFrame | None, as_of: date) -> pd.DataFrame:
        if cfg["simulate"]:
            return normalize_export(simulated_export(accounts, as_of), COLUMNS, service="ChurnZero")
        extra = {cfg["account_id_column"]: "account_ref"} if cfg["account_id_column"] else None
        return read_export(cfg["file"], COLUMNS, service="ChurnZero", required=("name",), extra_aliases=extra)

    def test_connection(self, config: dict[str, Any]) -> str:
        cfg = self.validate(config)
        if cfg["simulate"]:
            return "ChurnZero: simulated demo data, no account needed."
        rows = self._read(cfg, None, date.today())
        used = [k for k in COLUMNS if k in rows.columns and k not in ("account_ref", "name")]
        return f"ChurnZero export read: {len(rows):,} accounts, with {', '.join(used) or 'no usable signal columns'}."

    def fetch(self, config: dict[str, Any], accounts: pd.DataFrame, as_of: date) -> SourceResult:
        cfg = config if "simulate" in config else self.validate(config)
        rows = self._read(cfg, accounts, as_of)
        acct, unmatched_rows, warnings = match_records(rows, accounts, mode=cfg["match_on"], service="ChurnZero")
        signals = compute_signals(rows.assign(account_id=acct), as_of)
        ids = accounts["account_id"].astype(str)
        matched = set(signals.index)
        if not matched and len(rows):
            warnings.append("No ChurnZero account matched. Set 'Column holding your account ID'.")
        return SourceResult(
            source=self.label,
            signals=signals,
            matched_accounts=len(matched & set(ids)),
            unmatched_accounts=[a for a in ids if a not in matched],
            unmatched_records=unmatched_rows,
            records_read=len(rows),
            warnings=warnings,
            simulated=bool(cfg["simulate"]),
            records_label="ChurnZero accounts",
            unmatched_label="ChurnZero records",
        )


def compute_signals(rows: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Per-account signals. Several ChurnZero accounts on one CRM account are combined (users add up)."""
    cols = list(ChurnZeroSource.provides)
    m = rows[rows["account_id"].notna()].copy()
    ids = sorted(set(m["account_id"].astype(str)))
    out = pd.DataFrame(index=pd.Index(ids, name="account_id"), columns=cols, dtype=float)
    if not ids:
        return out
    g = m["account_id"].astype(str)

    def agg(values: pd.Series, how: str) -> pd.Series:
        return values.groupby(g).agg(how).reindex(ids)

    for key in ("active_users", "active_users_prior"):
        if key in m.columns:
            out[key] = numbers(m[key]).groupby(g).sum(min_count=1).reindex(ids)
    days = pd.Series(np.nan, index=m.index)
    if "days_since_login" in m.columns:
        days = numbers(m["days_since_login"])
    if "last_login" in m.columns:
        days = days.fillna(days_since(m["last_login"], as_of))
    out["days_since_last_login"] = agg(days, "min")
    if "nps" in m.columns:
        nps = numbers(m["nps"])
        if nps.notna().any() and (nps.min() < 0 or nps.max() > 10):
            nps = ((nps.clip(-100, 100) + 100) / 20).round(1)  # account-level NPS to the 0-10 response scale
        out["nps"] = agg(nps, "mean").round(1)
    if "sentiment" in m.columns:
        sent = m["sentiment"].map(parse_sentiment).astype(float)
        sent = sent.where(sent <= 5, (sent.clip(0, 100) / 20).clip(1, 5))
        out["csm_sentiment"] = agg(sent, "mean")
    if "exec_sponsor" in m.columns:
        out["exec_sponsor"] = agg(flags(m["exec_sponsor"]), "max")
    if "champion_change" in m.columns:
        out["champion_change_90d"] = agg(flags(m["champion_change"]), "max")
    return out


def simulated_export(accounts: pd.DataFrame | None, as_of: date, *, seed: int = 41) -> pd.DataFrame:
    """A ChurnZero account export built from the book, in ChurnZero's column names, for demos and tests.

    Values follow the book's own, so turning the simulated add-on on leaves the demo coherent; a few accounts
    are missing and some carry only a name, so the match report has something to say.
    """
    cols = [
        "Account External ID",
        "Account Name",
        "Active Contacts (Last 30 Days)",
        "Active Contacts (Prior Period)",
        "Last Login Date",
        "Last NPS Score",
        "ChurnScore",
    ]
    if accounts is None or not len(accounts):
        return pd.DataFrame(columns=cols)
    rng = np.random.default_rng(seed)
    acc = accounts.reset_index(drop=True)
    end = end_of(as_of)

    def col(name: str) -> np.ndarray:
        if name in acc.columns:
            return pd.to_numeric(acc[name], errors="coerce").to_numpy(float)
        return np.full(len(acc), np.nan)

    users, prior, login, nps = col("active_users"), col("active_users_prior"), col("days_since_last_login"), col("nps")
    util = np.nan_to_num(col("seat_utilization"), nan=0.6)
    names = acc["account_name"].astype(str) if "account_name" in acc.columns else acc["account_id"].astype(str)
    out = []
    for i in range(len(acc)):
        u = rng.random()
        if u < 0.06:
            continue
        gap = login[i] if not np.isnan(login[i]) else float(rng.integers(0, 30))
        score = nps[i]
        if np.isnan(score) and rng.random() < 0.6:  # the book has no NPS here: about 60% of accounts answered
            score = float(np.clip(round(rng.normal(3 + util[i] * 7, 1.5)), 0, 10))
        out.append(
            {
                "Account External ID": "" if u < 0.14 else str(acc.loc[i, "account_id"]),
                "Account Name": names[i],
                "Active Contacts (Last 30 Days)": "" if np.isnan(users[i]) else f"{users[i]:.0f}",
                "Active Contacts (Prior Period)": "" if np.isnan(prior[i]) else f"{prior[i]:.0f}",
                "Last Login Date": (end - pd.Timedelta(days=gap + 0.5)).strftime("%m/%d/%Y %I:%M %p"),
                "Last NPS Score": "" if np.isnan(score) else f"{score:g}",
                "ChurnScore": f"{rng.integers(20, 100)}",
            }
        )
    out.append({"Account External ID": "", "Account Name": "Sandbox - Internal", "Last Login Date": ""})
    return pd.DataFrame(out, columns=cols)
