"""HubSpot add-on: relationship recency and renewal-deal signals from the CRM teams actually work in.

Many CS and account teams log calls and emails in HubSpot and run renewals as HubSpot deals, even when the
finance system of record is elsewhere. This add-on reads two things, read-only:

* **Companies**, for the last time anyone on your side reached the customer (HubSpot's "Last Contacted").
* **Deals** (optional), for the open renewal deal and the most recent won deal on each company.

Two ways in: a CSV export of companies (and optionally deals), or the CRM API with a private app's access token
(scopes ``crm.objects.companies.read`` and ``crm.objects.deals.read``).

What it returns per matched account:

========================  =================================================================================
days_since_last_meeting   days since HubSpot's Last Contacted date (a logged call, meeting or email)
price_increase_pct        open renewal deal amount versus the account's current ARR, as a percent change
competitor_eval           1 when the competitor property you name is filled in on the company or renewal deal
discount_pct              the discount property you name, from the most recent won deal
========================  =================================================================================

The renewal comparison needs ARR in the book, and only uses the open renewal deal closing soonest. A renewal
quoted below current ARR shows up as a negative change, which is a downsell warning in its own right.
"""

from __future__ import annotations

import time
from collections.abc import Callable
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
    flags,
    match_records,
    normalize_export,
    numbers,
    read_export,
    timestamps,
)
from .http import JsonClient, Transport

MAX_PAGES = 1000
PAGE_SIZE = 100

COMPANY_COLUMNS: dict[str, tuple[str, ...]] = {
    "record_id": ("Record ID", "Company ID", "hs_object_id", "id"),
    "name": ("Company name", "Name", "Company"),
    "account_ref": (
        "Account ID",
        "Salesforce Account ID",
        "External account ID",
        "CRM account ID",
        "account_id",
        "salesforce_account_id",
    ),
    "last_contacted": ("Last Contacted", "notes_last_contacted", "Last contacted date"),
    "competitor": ("Competitor", "Competitor in deal", "competitor"),
}
DEAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "deal_id": ("Record ID", "Deal ID", "hs_object_id", "id"),
    "deal_name": ("Deal Name", "dealname"),
    "amount": ("Amount", "amount", "Deal amount"),
    "close_date": ("Close Date", "closedate"),
    "stage": ("Deal Stage", "dealstage", "Stage"),
    "pipeline": ("Pipeline", "pipeline"),
    "deal_type": ("Deal Type", "dealtype", "Type"),
    "is_closed": ("Is Closed", "Is deal closed?", "hs_is_closed"),
    "is_won": ("Is Closed Won", "Is closed won", "hs_is_closed_won"),
    "company_ref": (
        "Associated Company IDs (Primary)",
        "Associated Company IDs",
        "Associated Company ID",
        "Company ID",
        "company_id",
    ),
    "company_name": ("Associated Company", "Associated Company (Primary)", "Company name", "Company"),
    "competitor": ("Competitor", "competitor"),
    "discount": ("Discount", "Discount %", "Discount percent", "discount"),
}


class HubSpotSource(DataSource):
    name = "hubspot"
    label = "HubSpot"
    description = "Days since last contact, renewal deal versus current ARR, competitors and discounts per account."
    supports_simulation = True
    fields = (
        ConfigField(
            "companies_file",
            "Companies export (CSV)",
            required=False,
            help="Path to a HubSpot companies export. Or use an access token instead.",
        ),
        ConfigField(
            "deals_file",
            "Deals export (CSV)",
            required=False,
            help="Optional. A deals export with the associated company column, for renewal and discount signals.",
        ),
        ConfigField(
            "access_token",
            "Private app access token",
            secret=True,
            required=False,
            help="For the API: Settings > Integrations > Private apps, with companies and deals read scopes.",
        ),
        account_id_field("salesforce_account_id"),
        MATCH_FIELD,
        ConfigField(
            "renewal_keyword",
            "Deals counted as renewals",
            required=False,
            default="renewal",
            help="A word in the deal name, pipeline or type. Deals of type existingbusiness also count.",
        ),
        ConfigField(
            "competitor_property",
            "Competitor property",
            required=False,
            help="Optional: a company or deal property naming a competitor in play. Blank detects 'Competitor'.",
        ),
        ConfigField(
            "discount_property",
            "Discount property",
            required=False,
            help="Optional: a deal property holding the discount percent. Blank detects 'Discount'.",
        ),
    )
    provides = ("days_since_last_meeting", "price_increase_pct", "competitor_eval", "discount_pct")

    def __init__(self, transport: Transport | None = None, sleep: Callable[[float], None] = time.sleep):
        self._transport = transport
        self._sleep = sleep

    # -- settings ---------------------------------------------------------------------------------------------
    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        cfg = super().validate(config)
        for k in ("account_id_column", "competitor_property", "discount_property"):
            cfg[k] = str(cfg.get(k) or "").strip()
        cfg["renewal_keyword"] = str(cfg.get("renewal_keyword") or "renewal").strip().lower()
        if cfg["simulate"]:
            cfg["mode"] = "simulate"
        elif cfg.get("companies_file"):
            cfg["mode"] = "file"
        elif cfg.get("access_token"):
            cfg["mode"] = "api"
        elif cfg.get("deals_file"):
            raise SourceError("config", "HubSpot: add the companies export too; deals are matched through companies.")
        else:
            raise SourceError("config", "HubSpot: add a companies export file, or a private app access token.")
        return cfg

    def _client(self, cfg: dict[str, Any]) -> JsonClient:
        return JsonClient(
            "https://api.hubapi.com",
            headers={"Authorization": f"Bearer {cfg['access_token']}"},
            transport=self._transport,
            sleep=self._sleep,
            service="HubSpot",
        )

    def test_connection(self, config: dict[str, Any]) -> str:
        cfg = self.validate(config)
        if cfg["mode"] == "simulate":
            return "HubSpot: simulated demo data, no account needed."
        if cfg["mode"] == "file":
            companies, deals = self._read_files(cfg)
            extra = f" and {len(deals):,} deals" if deals is not None else ""
            return f"HubSpot export read: {len(companies):,} companies{extra}."
        self._client(cfg).get("crm/v3/objects/companies", {"limit": 1})
        return "Connected to HubSpot. The token can read companies."

    # -- fetch ------------------------------------------------------------------------------------------------
    def fetch(self, config: dict[str, Any], accounts: pd.DataFrame, as_of: date) -> SourceResult:
        cfg = config if "mode" in config else self.validate(config)
        warnings: list[str] = []
        deals: pd.DataFrame | None
        if cfg["mode"] == "simulate":
            raw_companies, raw_deals = simulated_export(accounts, as_of)
            companies = normalize_export(raw_companies, COMPANY_COLUMNS, service="HubSpot", required=("name",))
            deals = normalize_export(raw_deals, DEAL_COLUMNS, service="HubSpot")
        elif cfg["mode"] == "file":
            companies, deals = self._read_files(cfg)
        else:
            companies, deals = self._read_api(cfg, warnings)

        acct, unmatched_companies, match_warnings = match_records(
            companies, accounts, mode=cfg["match_on"], service="HubSpot company"
        )
        warnings += match_warnings
        companies = companies.assign(account_id=acct)
        if deals is not None and len(deals):
            deals = link_deals(deals, companies, accounts, warnings)
        signals = compute_signals(companies, deals, accounts, as_of, cfg["renewal_keyword"])

        ids = accounts["account_id"].astype(str)
        matched = set(signals.index)
        if not matched and len(companies):
            warnings.append("No HubSpot company matched an account. Set 'Column holding your account ID'.")
        return SourceResult(
            source=self.label,
            signals=signals,
            matched_accounts=len(matched & set(ids)),
            unmatched_accounts=[a for a in ids if a not in matched],
            unmatched_records=unmatched_companies,
            records_read=len(companies) + (len(deals) if deals is not None else 0),
            warnings=warnings,
            simulated=cfg["mode"] == "simulate",
            records_label="companies and deals",
            unmatched_label="companies",
        )

    def _aliases(self, cfg: dict[str, Any], *, deal: bool) -> dict[str, str]:
        extra: dict[str, str] = {}
        if cfg["account_id_column"] and not deal:
            extra[cfg["account_id_column"]] = "account_ref"
        if cfg["competitor_property"]:
            extra[cfg["competitor_property"]] = "competitor"
        if cfg["discount_property"] and deal:
            extra[cfg["discount_property"]] = "discount"
        return extra

    def _read_files(self, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        companies = read_export(
            cfg["companies_file"],
            COMPANY_COLUMNS,
            service="HubSpot",
            required=("name",),
            extra_aliases=self._aliases(cfg, deal=False),
        )
        deals = None
        if cfg.get("deals_file"):
            deals = read_export(
                cfg["deals_file"],
                DEAL_COLUMNS,
                service="HubSpot deals",
                required=("amount", "close_date"),
                extra_aliases=self._aliases(cfg, deal=True),
            )
        return companies, deals

    def _read_api(self, cfg: dict[str, Any], warnings: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        client = self._client(cfg)
        id_prop = cfg["account_id_column"]
        comp_prop = cfg["competitor_property"] or "competitor"
        disc_prop = cfg["discount_property"] or "discount"
        props = ["name", "notes_last_contacted", comp_prop] + ([id_prop] if id_prop else [])
        companies = []
        for item in self._pages(client, "crm/v3/objects/companies", {"properties": ",".join(props)}, warnings):
            p = item.get("properties") or {}
            companies.append(
                {
                    "record_id": item.get("id"),
                    "name": p.get("name"),
                    "account_ref": p.get(id_prop) if id_prop else None,
                    "last_contacted": p.get("notes_last_contacted"),
                    "competitor": p.get(comp_prop),
                }
            )
        deal_props = "dealname,amount,closedate,dealstage,pipeline,dealtype,hs_is_closed,hs_is_closed_won"
        deals = []
        params = {"properties": f"{deal_props},{comp_prop},{disc_prop}", "associations": "companies"}
        for item in self._pages(client, "crm/v3/objects/deals", params, warnings):
            p = item.get("properties") or {}
            assoc = ((item.get("associations") or {}).get("companies") or {}).get("results") or []
            deals.append(
                {
                    "deal_id": item.get("id"),
                    "deal_name": p.get("dealname"),
                    "amount": p.get("amount"),
                    "close_date": p.get("closedate"),
                    "stage": p.get("dealstage"),
                    "pipeline": p.get("pipeline"),
                    "deal_type": p.get("dealtype"),
                    "is_closed": p.get("hs_is_closed"),
                    "is_won": p.get("hs_is_closed_won"),
                    "company_ref": next((str(a.get("id")) for a in assoc if isinstance(a, dict)), None),
                    "competitor": p.get(comp_prop),
                    "discount": p.get(disc_prop),
                }
            )
        comp_df = normalize_export(
            pd.DataFrame(companies, columns=list(COMPANY_COLUMNS)), COMPANY_COLUMNS, service="HubSpot"
        )
        deal_df = normalize_export(pd.DataFrame(deals, columns=list(DEAL_COLUMNS)), DEAL_COLUMNS, service="HubSpot")
        return comp_df, deal_df

    def _pages(self, client: JsonClient, path: str, params: dict[str, Any], warnings: list[str]) -> list[dict]:
        out: list[dict[str, Any]] = []
        query = {"limit": PAGE_SIZE, **params}
        for _ in range(MAX_PAGES):
            page = client.get(path, query)
            out += [r for r in page.get("results") or [] if isinstance(r, dict)]
            after = ((page.get("paging") or {}).get("next") or {}).get("after")
            if not after:
                return out
            query = {**query, "after": after}
        warnings.append(f"Stopped after {len(out):,} HubSpot records from {path.rsplit('/', 1)[-1]}.")
        return out


# -- linking and metrics --------------------------------------------------------------------------------------
def link_deals(
    deals: pd.DataFrame, companies: pd.DataFrame, accounts: pd.DataFrame, warnings: list[str]
) -> pd.DataFrame:
    """Give each deal the account of its company: by HubSpot company ID, otherwise by company name."""
    out = deals.copy()
    out["account_id"] = np.nan
    out["account_id"] = out["account_id"].astype(object)
    if "company_ref" in out.columns and "record_id" in companies.columns:
        by_record = dict(zip(companies["record_id"].astype(str), companies["account_id"], strict=True))
        first_ref = out["company_ref"].astype(str).str.split(r"[;,]").str[0].str.strip()
        out["account_id"] = first_ref.map(by_record)
    pending = out["account_id"].isna()
    if pending.any() and "company_name" in out.columns:
        by_name, _, _ = match_records(
            out.loc[pending, ["company_name"]].rename(columns={"company_name": "name"}),
            accounts,
            mode="name",
            service="HubSpot deal company",
            one_per_account=False,
        )
        out.loc[pending, "account_id"] = by_name
    unlinked = int(out["account_id"].isna().sum())
    if unlinked:
        warnings.append(f"{unlinked:,} deals are on companies that matched no account and were not used.")
    return out


def _is_renewal(deals: pd.DataFrame, keyword: str) -> pd.Series:
    hit = pd.Series(False, index=deals.index)
    for col in ("deal_name", "pipeline", "deal_type", "stage"):
        if col in deals.columns:
            hit |= deals[col].astype(str).str.lower().str.contains(keyword, regex=False, na=False)
    if "deal_type" in deals.columns:
        hit |= deals["deal_type"].astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True).eq("existingbusiness")
    return hit


def _closed_won(deals: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    stage = deals.get("stage", pd.Series("", index=deals.index)).astype(str).str.lower().str.replace(" ", "")
    closed = flags(deals["is_closed"]) == 1 if "is_closed" in deals.columns else pd.Series(False, index=deals.index)
    won = flags(deals["is_won"]) == 1 if "is_won" in deals.columns else pd.Series(False, index=deals.index)
    closed = closed | stage.str.contains("closedwon|closedlost", regex=True)
    won = won | stage.str.contains("closedwon", regex=False)
    return closed, won


def compute_signals(
    companies: pd.DataFrame,
    deals: pd.DataFrame | None,
    accounts: pd.DataFrame,
    as_of: date,
    renewal_keyword: str = "renewal",
) -> pd.DataFrame:
    cols = list(HubSpotSource.provides)
    matched = companies[companies["account_id"].notna()].copy()
    ids = sorted(set(matched["account_id"].astype(str)))
    out = pd.DataFrame(index=pd.Index(ids, name="account_id"), columns=cols, dtype=float)
    if not ids:
        return out
    matched["account_id"] = matched["account_id"].astype(str)
    if "last_contacted" in matched.columns:
        matched["days"] = days_since(matched["last_contacted"], as_of)
        out["days_since_last_meeting"] = matched.groupby("account_id")["days"].min().reindex(ids)
    comp_flag = pd.Series(np.nan, index=pd.Index(ids))
    have_competitor = "competitor" in matched.columns
    if have_competitor:
        comp_flag = flags(matched["competitor"]).groupby(matched["account_id"]).max().reindex(ids)

    if deals is not None and len(deals) and "account_id" in deals.columns:
        d = deals[deals["account_id"].astype(str).isin(ids)].copy()
        d["account_id"] = d["account_id"].astype(str)
        d["amount_num"] = numbers(d["amount"]) if "amount" in d.columns else np.nan
        d["close_ts"] = timestamps(d["close_date"]) if "close_date" in d.columns else pd.NaT
        closed, won = _closed_won(d)
        renewal_open = d[_is_renewal(d, renewal_keyword) & ~closed & (d["amount_num"] > 0)]
        nxt = renewal_open.sort_values("close_ts", na_position="last").groupby("account_id").first()
        if len(nxt) and "arr" in accounts.columns:
            arr = pd.to_numeric(accounts.set_index(accounts["account_id"].astype(str))["arr"], errors="coerce")
            arr = arr[~arr.index.duplicated()].reindex(nxt.index)
            change = (nxt["amount_num"] / arr.where(arr > 0) - 1) * 100
            out["price_increase_pct"] = change.clip(-100, 300).round(1).reindex(ids)
        have_competitor |= "competitor" in d.columns
        if "competitor" in d.columns and len(renewal_open):
            deal_flag = flags(renewal_open["competitor"]).groupby(renewal_open["account_id"]).max().reindex(ids)
            comp_flag = pd.concat([comp_flag, deal_flag], axis=1).max(axis=1)
        if "discount" in d.columns:
            won_deals = d[won & numbers(d["discount"]).notna()]
            last = won_deals.sort_values("close_ts", na_position="first").groupby("account_id").last()
            disc = numbers(last["discount"]) if len(last) else pd.Series(dtype=float)
            disc = disc.where(disc.abs() > 1, disc * 100)  # 0.25 and 25 both mean 25%
            out["discount_pct"] = disc.clip(0, 100).reindex(ids)
    # A competitor property that exists but is blank means "none recorded": 0, not unknown.
    if have_competitor:
        out["competitor_eval"] = comp_flag.reindex(ids).fillna(0.0)
    return out


# -- simulated export -----------------------------------------------------------------------------------------
def simulated_export(
    accounts: pd.DataFrame | None, as_of: date, *, seed: int = 31
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HubSpot companies and deals exports built from the book, in HubSpot's column names.

    Last-contacted dates and renewal quotes follow the book's own values where it has them, so the demo stays
    coherent. Some companies carry only a name (no account ID) and a few accounts are missing from HubSpot.
    """
    comp_cols = ["Record ID", "Company name", "Account ID", "Last Contacted", "Competitor"]
    deal_cols = [
        "Record ID",
        "Deal Name",
        "Amount",
        "Close Date",
        "Deal Stage",
        "Pipeline",
        "Deal Type",
        "Associated Company IDs (Primary)",
        "Discount",
    ]
    if accounts is None or not len(accounts):
        return pd.DataFrame(columns=comp_cols), pd.DataFrame(columns=deal_cols)
    rng = np.random.default_rng(seed)
    acc = accounts.reset_index(drop=True)
    end = end_of(as_of)

    def col(name: str, default: float) -> np.ndarray:
        if name in acc.columns:
            return pd.to_numeric(acc[name], errors="coerce").fillna(default).to_numpy(float)
        return np.full(len(acc), default)

    last_meeting = col("days_since_last_meeting", np.nan)
    uplift = col("price_increase_pct", np.nan)
    discount = col("discount_pct", np.nan)
    competitor = col("competitor_eval", 0)
    arr = col("arr", 0)
    days_left = col("days_to_renewal", 200)
    names = acc["account_name"].astype(str) if "account_name" in acc.columns else acc["account_id"].astype(str)
    companies, deals = [], []
    for i in range(len(acc)):
        u = rng.random()
        if u < 0.05:
            continue  # not in HubSpot
        rid = str(5_000_000 + i)
        gap = last_meeting[i] if not np.isnan(last_meeting[i]) else float(rng.integers(3, 120))
        companies.append(
            {
                "Record ID": rid,
                "Company name": names[i],
                "Account ID": "" if u < 0.13 else str(acc.loc[i, "account_id"]),
                "Last Contacted": (end - pd.Timedelta(days=gap + 0.5)).strftime("%Y-%m-%d %H:%M"),
                "Competitor": str(rng.choice(["Vendr", "Rivalry Co"])) if competitor[i] >= 1 else "",
            }
        )
        if arr[i] > 0:
            disc = discount[i] if not np.isnan(discount[i]) else float(rng.choice([0, 0, 5, 10, 15, 20]))
            signed = end - pd.Timedelta(days=float(365 - min(max(days_left[i], 0), 365)) + 1)
            deals.append(
                {
                    "Record ID": f"9{rid}",
                    "Deal Name": f"{names[i]} - New business",
                    "Amount": f"{arr[i]:.2f}",
                    "Close Date": signed.strftime("%Y-%m-%d"),
                    "Deal Stage": "Closed won",
                    "Pipeline": "Sales Pipeline",
                    "Deal Type": "New Business",
                    "Associated Company IDs (Primary)": rid,
                    "Discount": f"{disc:g}",
                }
            )
            if 0 <= days_left[i] <= 240 and rng.random() < 0.85:
                change = uplift[i] if not np.isnan(uplift[i]) else float(rng.choice([0, 3, 5, 7]))
                deals.append(
                    {
                        "Record ID": f"8{rid}",
                        "Deal Name": f"{names[i]} - Renewal",
                        "Amount": f"{arr[i] * (1 + change / 100):.2f}",
                        "Close Date": (end + pd.Timedelta(days=days_left[i])).strftime("%Y-%m-%d"),
                        "Deal Stage": str(rng.choice(["Renewal scoping", "Proposal sent", "Negotiation"])),
                        "Pipeline": "Renewals",
                        "Deal Type": "Existing Business",
                        "Associated Company IDs (Primary)": rid,
                        "Discount": "",
                    }
                )
    companies.append(
        {
            "Record ID": "4999999",
            "Company name": "Prospect Holdings",
            "Account ID": "",
            "Last Contacted": "",
            "Competitor": "",
        }
    )
    return pd.DataFrame(companies, columns=comp_cols), pd.DataFrame(deals, columns=deal_cols)
