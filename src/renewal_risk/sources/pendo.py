"""Pendo add-on: product usage per account from a Pendo account report export.

Pendo knows who actually uses the product: active visitors, when anyone last showed up, and how many of the
features you call "core" each account touches. This add-on reads a Pendo account report saved as CSV (Data
Explorer or an Accounts report, then Download CSV), so it needs no integration key.

What it returns per matched account, from whichever of these columns the export carries:

========================  =================================================================================
active_users              active visitors in the report period (30 days is the usual choice)
active_users_prior        active visitors in the same period about 90 days earlier, when exported (the trend)
days_since_last_login     days since the account's last visit
feature_adoption_pct      core feature adoption: a percent column, or core features used out of a total
nps                       average NPS poll response on the 0 to 10 scale, when the report includes it
========================  =================================================================================

Pendo account IDs are usually your CRM account IDs (most teams pass the Salesforce ID when installing the Pendo
snippet). If yours are different, match on account name or point the add-on at a metadata column with the ID.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .base import ConfigField, DataSource, SourceError, SourceResult
from .exports import (
    MATCH_FIELD,
    account_id_field,
    days_since,
    end_of,
    match_records,
    normalize_export,
    numbers,
    read_export,
)

COLUMNS: dict[str, tuple[str, ...]] = {
    "account_ref": ("Account ID", "accountId", "Account Id", "Salesforce Account ID", "CRM Account ID"),
    "name": ("Account Name", "Name", "Account"),
    "active_users": (
        "Active Visitors (Last 30 Days)",
        "Active Visitors",
        "Number of Active Visitors",
        "Number of Visitors",
        "Visitors",
        "Unique Visitors",
    ),
    "active_users_prior": (
        "Active Visitors (Prior Period)",
        "Active Visitors 90 Days Ago",
        "Visitors (Prior Period)",
        "Number of Visitors (Prior Period)",
    ),
    "last_visit": ("Last Visit", "Last Visited", "Last Visit Date", "lastvisit"),
    "days_since_visit": ("Days Since Last Visit", "Days Inactive"),
    "adoption_pct": ("Core Feature Adoption %", "Core Feature Adoption", "Feature Adoption %", "Feature Adoption"),
    "core_used": ("Core Features Used", "Core Events Used", "Features Used"),
    "core_total": ("Core Features Total", "Total Core Features", "Core Events Total"),
    "nps": ("NPS Score", "Average NPS Response", "NPS", "Latest NPS Response"),
}


class PendoSource(DataSource):
    name = "pendo"
    label = "Pendo"
    description = "Active visitors and trend, last visit, core feature adoption and NPS polls from a Pendo report."
    supports_simulation = True
    fields = (
        ConfigField("file", "Account report (CSV)", help="Path to a Pendo account report downloaded as CSV."),
        account_id_field("agent_salesforceId"),
        MATCH_FIELD,
        ConfigField(
            "core_features",
            "Number of core features",
            required=False,
            kind="int",
            help="Optional: if the report has 'Core Features Used' but no total, how many core features you tag.",
        ),
    )
    provides = ("active_users", "active_users_prior", "days_since_last_login", "feature_adoption_pct", "nps")

    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        cfg = super().validate(config)
        cfg["account_id_column"] = str(cfg.get("account_id_column") or "").strip()
        if cfg.get("core_features") is not None and int(cfg["core_features"]) <= 0:
            raise SourceError("config", "Pendo: 'Number of core features' must be above zero.")
        return cfg

    def _read(self, cfg: dict[str, Any], accounts: pd.DataFrame | None, as_of: date) -> pd.DataFrame:
        if cfg["simulate"]:
            return normalize_export(simulated_export(accounts, as_of), COLUMNS, service="Pendo")
        extra = {cfg["account_id_column"]: "account_ref"} if cfg["account_id_column"] else None
        rows = read_export(cfg["file"], COLUMNS, service="Pendo", extra_aliases=extra)
        if "account_ref" not in rows.columns and "name" not in rows.columns:
            raise SourceError("config", "Pendo: the report needs an Account ID or Account Name column.")
        return rows

    def test_connection(self, config: dict[str, Any]) -> str:
        cfg = self.validate(config)
        if cfg["simulate"]:
            return "Pendo: simulated demo data, no account needed."
        rows = self._read(cfg, None, date.today())
        used = [k for k in COLUMNS if k in rows.columns and k not in ("account_ref", "name")]
        return f"Pendo report read: {len(rows):,} accounts, with {', '.join(used) or 'no usable signal columns'}."

    def fetch(self, config: dict[str, Any], accounts: pd.DataFrame, as_of: date) -> SourceResult:
        cfg = config if "simulate" in config else self.validate(config)
        rows = self._read(cfg, accounts, as_of)
        acct, unmatched_rows, warnings = match_records(rows, accounts, mode=cfg["match_on"], service="Pendo")
        signals = compute_signals(rows.assign(account_id=acct), as_of, cfg.get("core_features"))
        no_total = "core_used" in rows.columns and "core_total" not in rows.columns and not cfg.get("core_features")
        if no_total and "adoption_pct" not in rows.columns:
            warnings.append("Core features used was exported without a total; set 'Number of core features'.")
        ids = accounts["account_id"].astype(str)
        matched = set(signals.index)
        if not matched and len(rows):
            warnings.append("No Pendo account matched. Check the account ID column or match by name.")
        return SourceResult(
            source=self.label,
            signals=signals,
            matched_accounts=len(matched & set(ids)),
            unmatched_accounts=[a for a in ids if a not in matched],
            unmatched_records=unmatched_rows,
            records_read=len(rows),
            warnings=warnings,
            simulated=bool(cfg["simulate"]),
            records_label="Pendo accounts",
            unmatched_label="Pendo records",
        )


def compute_signals(rows: pd.DataFrame, as_of: date, core_features: int | None = None) -> pd.DataFrame:
    """Per-account usage. Several Pendo accounts on one CRM account are combined (visitors add up)."""
    cols = list(PendoSource.provides)
    m = rows[rows["account_id"].notna()].copy()
    ids = sorted(set(m["account_id"].astype(str)))
    out = pd.DataFrame(index=pd.Index(ids, name="account_id"), columns=cols, dtype=float)
    if not ids:
        return out
    g = m["account_id"].astype(str)
    for key in ("active_users", "active_users_prior"):
        if key in m.columns:
            out[key] = numbers(m[key]).groupby(g).sum(min_count=1).reindex(ids)
    days = numbers(m["days_since_visit"]) if "days_since_visit" in m.columns else pd.Series(np.nan, index=m.index)
    if "last_visit" in m.columns:
        days = days.fillna(days_since(m["last_visit"], as_of))
    out["days_since_last_login"] = days.groupby(g).min().reindex(ids)

    adoption = numbers(m["adoption_pct"]) if "adoption_pct" in m.columns else pd.Series(np.nan, index=m.index)
    if adoption.notna().any() and adoption.dropna().abs().max() <= 1.0:
        adoption = adoption * 100  # exported as a fraction
    if "core_used" in m.columns:
        used = numbers(m["core_used"])
        total = numbers(m["core_total"]) if "core_total" in m.columns else pd.Series(np.nan, index=m.index)
        if core_features:
            total = total.fillna(float(core_features))
        adoption = adoption.fillna(used / total.where(total > 0) * 100)
    out["feature_adoption_pct"] = adoption.clip(0, 100).groupby(g).max().reindex(ids).round(1)
    if "nps" in m.columns:
        out["nps"] = numbers(m["nps"]).clip(0, 10).groupby(g).mean().reindex(ids).round(1)
    return out


def simulated_export(accounts: pd.DataFrame | None, as_of: date, *, seed: int = 53) -> pd.DataFrame:
    """A Pendo account report built from the book, in Pendo's column names, for demos and tests.

    Values follow the book's own, so turning the simulated add-on on leaves the demo coherent. Some accounts
    never installed Pendo, a few report under a different ID (name only), and there is one internal test account.
    """
    cols = [
        "Account ID",
        "Account Name",
        "Number of Active Visitors",
        "Active Visitors (Prior Period)",
        "Last Visit",
        "Core Features Used",
        "Core Features Total",
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

    users, prior, login = col("active_users"), col("active_users_prior"), col("days_since_last_login")
    adoption = col("feature_adoption_pct")
    util = np.nan_to_num(col("seat_utilization"), nan=0.6)
    names = acc["account_name"].astype(str) if "account_name" in acc.columns else acc["account_id"].astype(str)
    out = []
    for i in range(len(acc)):
        u = rng.random()
        if u < 0.07:
            continue
        gap = login[i] if not np.isnan(login[i]) else float(rng.integers(0, 20))
        pct = adoption[i]
        if np.isnan(pct):  # the book has no adoption figure here: derive one from seat use
            pct = float(np.clip(rng.normal(15 + util[i] * 60, 10), 0, 100))
        used = f"{round(pct / 5):.0f}"  # 20 core features tagged
        out.append(
            {
                "Account ID": f"pendo-{i:05d}" if u < 0.12 else str(acc.loc[i, "account_id"]),
                "Account Name": names[i],
                "Number of Active Visitors": "" if np.isnan(users[i]) else f"{users[i]:.0f}",
                "Active Visitors (Prior Period)": "" if np.isnan(prior[i]) else f"{prior[i]:.0f}",
                "Last Visit": (end - pd.Timedelta(days=gap + 0.5)).isoformat(),
                "Core Features Used": used,
                "Core Features Total": "20",
            }
        )
    out.append({"Account ID": "internal-test", "Account Name": "Pendo Test Account", "Number of Active Visitors": "3"})
    return pd.DataFrame(out, columns=cols)
