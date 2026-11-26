"""Zendesk Support add-on: per-account support health from tickets, not just raw counts.

What it reads (read-only, with an admin's API token):

* Organizations, to match each one to an account in the book: by external ID, by a custom organization field,
  or by company name ("Acme, Inc." matches "ACME").
* Tickets created in the last 180 days, from the incremental export API, with each ticket's metrics.

What it returns per matched account, using the as-of date as "today":

========================  =================================================================================
tickets_90d               tickets created in the last 90 days
tickets_90d_prior         tickets created in the 90 days before that (for the trend, and "gone quiet")
p1_tickets_90d            of those, tickets at an urgent priority (configurable)
csat                      share of satisfaction ratings that were "good", when at least 3 were given
ticket_reopen_rate_pct    share of solved tickets that were reopened at least once
ticket_resolution_hours   median calendar hours from creation to full resolution, solved tickets only
========================  =================================================================================

Rates need at least three tickets behind them; below that they stay blank rather than swing on one ticket.

Why this shape: connectors that copy raw open/solved counts into a health score punish big customers for being
big. The scorer normalizes volume per seat, compares each account with its own previous quarter, and treats a
sudden silence from a formerly active account as a risk, which is what support leaders actually look for.
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.parse
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .base import ConfigField, DataSource, SourceError, SourceResult, normalize_name
from .http import JsonClient, Transport

WINDOW_DAYS = 90
MIN_FOR_RATE = 3
MAX_ORG_PAGES = 500
MAX_TICKET_PAGES = 1000
SOLVED = {"solved", "closed"}
_SUBDOMAIN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def clean_subdomain(raw: str) -> str:
    """Accept 'acme', 'acme.zendesk.com' or 'https://acme.zendesk.com/'; return 'acme' or raise."""
    s = str(raw or "").strip().lower()
    s = re.sub(r"^https?://", "", s).split("/")[0]
    if s.endswith(".zendesk.com"):
        s = s[: -len(".zendesk.com")]
    if not _SUBDOMAIN.match(s):
        raise SourceError("config", "Zendesk: the subdomain should look like 'acme' (the part before .zendesk.com).")
    return s


class ZendeskSource(DataSource):
    name = "zendesk"
    label = "Zendesk"
    description = "Ticket volume and trend, urgent tickets, CSAT, reopen rate and time to resolution per account."
    supports_simulation = True
    fields = (
        ConfigField("subdomain", "Zendesk subdomain", help="The part before .zendesk.com, e.g. acme"),
        ConfigField("email", "Admin email", help="Email of the Zendesk admin who owns the API token"),
        ConfigField(
            "api_token",
            "API token",
            secret=True,
            help="Admin Center > Apps and integrations > APIs > Zendesk API. Read access is all it uses.",
        ),
        ConfigField(
            "match_on",
            "Match organizations by",
            required=False,
            default="auto",
            kind="choice",
            choices=("auto", "external_id", "name", "field:"),
            help="auto tries the organization's external ID, then its name. field:<key> uses a custom field.",
        ),
        ConfigField(
            "urgent_priorities",
            "Priorities counted as P1",
            required=False,
            default="urgent",
            help="Comma-separated Zendesk priorities, e.g. urgent or urgent,high",
        ),
    )
    provides = (
        "tickets_90d",
        "tickets_90d_prior",
        "p1_tickets_90d",
        "csat",
        "ticket_reopen_rate_pct",
        "ticket_resolution_hours",
    )

    def __init__(self, transport: Transport | None = None, sleep: Callable[[float], None] = time.sleep):
        self._transport = transport
        self._sleep = sleep

    # -- settings ---------------------------------------------------------------------------------------------
    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        cfg = super().validate(config)
        if cfg["simulate"]:
            cfg["subdomain"] = "demo"
        else:
            cfg["subdomain"] = clean_subdomain(cfg["subdomain"])
            if "@" not in str(cfg["email"]):
                raise SourceError("config", "Zendesk: 'Admin email' should be an email address.")
        cfg["urgent_priorities"] = tuple(
            p.strip().lower() for p in str(cfg.get("urgent_priorities") or "urgent").split(",") if p.strip()
        )
        return cfg

    def _client(self, cfg: dict[str, Any], accounts: pd.DataFrame | None, as_of: date | None) -> JsonClient:
        if cfg["simulate"]:
            transport: Transport = SimulatedZendesk(accounts, as_of or date.today())
            headers: dict[str, str] = {}
        else:
            transport = self._transport or _default_transport()
            raw = f"{cfg['email']}/token:{cfg['api_token']}".encode()
            headers = {"Authorization": "Basic " + base64.b64encode(raw).decode()}
        return JsonClient(
            f"https://{cfg['subdomain']}.zendesk.com",
            headers=headers,
            transport=transport,
            sleep=self._sleep,
            service="Zendesk",
        )

    def test_connection(self, config: dict[str, Any]) -> str:
        cfg = self.validate(config)
        user = self._client(cfg, None, None).get("api/v2/users/me.json").get("user") or {}
        if user.get("id") is None:
            # Zendesk answers bad credentials on this endpoint with an anonymous user rather than a 401.
            raise SourceError("auth", "Zendesk did not accept the email and API token.")
        role = str(user.get("role") or "unknown")
        msg = f"Connected to {cfg['subdomain']}.zendesk.com as {user.get('name') or 'your user'} ({role})."
        if role != "admin" and not cfg["simulate"]:
            msg += " Reading tickets in bulk needs an admin's token, so scoring may be refused."
        return msg

    # -- fetch ------------------------------------------------------------------------------------------------
    def fetch(self, config: dict[str, Any], accounts: pd.DataFrame, as_of: date) -> SourceResult:
        cfg = config if "simulate" in config else self.validate(config)
        client = self._client(cfg, accounts, as_of)
        warnings: list[str] = []

        orgs = self._organizations(client, warnings)
        org_to_account, unmatched_orgs, match_warnings = match_organizations(orgs, accounts, cfg["match_on"])
        warnings += match_warnings

        tickets, metrics = self._tickets(client, as_of, warnings)
        signals = compute_signals(tickets, metrics, org_to_account, as_of, cfg["urgent_priorities"])

        no_org = sum(1 for t in tickets.values() if t.get("organization_id") is None)
        if no_org:
            warnings.append(f"{no_org:,} tickets have no organization and were not counted.")
        ids = accounts["account_id"].astype(str)
        matched = set(org_to_account.values())
        unmatched = [a for a in ids if a not in matched]
        if not matched:
            warnings.append(
                "No Zendesk organizations matched an account. Try another 'Match organizations by' setting."
            )
        return SourceResult(
            source=self.label,
            signals=signals,
            matched_accounts=len(matched & set(ids)),
            unmatched_accounts=unmatched,
            unmatched_records=unmatched_orgs,
            records_read=len(tickets),
            warnings=warnings,
            simulated=bool(cfg["simulate"]),
        )

    def _organizations(self, client: JsonClient, warnings: list[str]) -> list[dict[str, Any]]:
        orgs: list[dict[str, Any]] = []
        page: dict[str, Any] = client.get("api/v2/organizations.json", {"page[size]": 100})
        for _ in range(MAX_ORG_PAGES):
            orgs += [o for o in page.get("organizations") or [] if isinstance(o, dict)]
            nxt = (page.get("links") or {}).get("next")
            if not (page.get("meta") or {}).get("has_more") or not nxt:
                return orgs
            page = client.get(nxt)
        warnings.append(f"Stopped after {len(orgs):,} organizations; some were not read.")
        return orgs

    def _tickets(
        self, client: JsonClient, as_of: date, warnings: list[str]
    ) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
        start = _utc(as_of) + timedelta(days=1) - timedelta(days=2 * WINDOW_DAYS)
        latest_allowed = datetime.now(timezone.utc) - timedelta(minutes=2)  # API wants start_time >= 1 min ago
        start = min(start, latest_allowed)
        page = client.get(
            "api/v2/incremental/tickets/cursor.json",
            {"start_time": int(start.timestamp()), "include": "metric_sets", "exclude_deleted": "true"},
        )
        tickets: dict[int, dict[str, Any]] = {}
        metrics: dict[int, dict[str, Any]] = {}
        for _ in range(MAX_TICKET_PAGES):
            for t in page.get("tickets") or []:
                if isinstance(t, dict) and t.get("id") is not None:
                    tickets[int(t["id"])] = t  # later versions of a ticket replace earlier ones
            for m in page.get("metric_sets") or page.get("ticket_metrics") or []:
                if isinstance(m, dict) and m.get("ticket_id") is not None:
                    metrics[int(m["ticket_id"])] = m
            if page.get("end_of_stream", True) or not page.get("after_url"):
                return tickets, metrics
            page = client.get(str(page["after_url"]))
        warnings.append(f"Stopped after {len(tickets):,} tickets; the newest ones may be missing.")
        return tickets, metrics


def _default_transport() -> Transport:
    from .http import urllib_transport

    return urllib_transport


def _utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


# -- matching -------------------------------------------------------------------------------------------------
def match_organizations(
    orgs: list[dict[str, Any]], accounts: pd.DataFrame, mode: str
) -> tuple[dict[int, str], int, list[str]]:
    """Map Zendesk organization id -> account_id. Returns the map, the unmatched org count and warnings."""
    warnings: list[str] = []
    ids = set(accounts["account_id"].astype(str))
    mapping: dict[int, str] = {}
    mode = (mode or "auto").strip()
    if mode not in ("auto", "external_id", "name") and not (mode.startswith("field:") and len(mode) > 6):
        raise SourceError("config", f"Zendesk: unknown matching mode '{mode}'.")

    def by_key(get: Callable[[dict[str, Any]], Any]) -> None:
        for o in orgs:
            oid = o.get("id")
            if oid is None or int(oid) in mapping:
                continue
            key = get(o)
            if key not in (None, "") and str(key).strip() in ids:
                mapping[int(oid)] = str(key).strip()

    if mode in ("auto", "external_id"):
        by_key(lambda o: o.get("external_id"))
    if mode.startswith("field:"):
        fname = mode.split(":", 1)[1]
        by_key(lambda o: (o.get("organization_fields") or {}).get(fname))
    if mode in ("auto", "name") and "account_name" in accounts.columns:
        taken = set(mapping.values())
        names = accounts.assign(_n=accounts["account_name"].map(normalize_name))
        names = names[(names["_n"] != "") & ~names["account_id"].astype(str).isin(taken)]
        dup_accounts = set(names.loc[names["_n"].duplicated(keep=False), "_n"])
        by_name = {n: str(a) for n, a in zip(names["_n"], names["account_id"], strict=True) if n not in dup_accounts}
        pending = [o for o in orgs if o.get("id") is not None and int(o["id"]) not in mapping]
        org_names = pd.Series([normalize_name(o.get("name")) for o in pending], dtype=object)
        dup_orgs = set(org_names[org_names.duplicated(keep=False)])
        ambiguous = 0
        for o, n in zip(pending, org_names, strict=True):
            if n in dup_orgs or n in dup_accounts:
                ambiguous += 1
                continue
            if n in by_name and by_name[n] not in taken:
                mapping[int(o["id"])] = by_name[n]
                taken.add(by_name[n])
        if ambiguous:
            warnings.append(f"{ambiguous} organizations share a name with another, so they were not matched by name.")
    unmatched = sum(1 for o in orgs if o.get("id") is not None and int(o["id"]) not in mapping)
    return mapping, unmatched, warnings


# -- metrics --------------------------------------------------------------------------------------------------
def compute_signals(
    tickets: dict[int, dict[str, Any]],
    metrics: dict[int, dict[str, Any]],
    org_to_account: dict[int, str],
    as_of: date,
    urgent: tuple[str, ...] = ("urgent",),
) -> pd.DataFrame:
    """Per-account support signals from raw tickets and ticket metrics."""
    cols = list(ZendeskSource.provides)
    accounts = sorted(set(org_to_account.values()))
    out = pd.DataFrame(index=pd.Index(accounts, name="account_id"), columns=cols, dtype=float)
    if not accounts:
        return out
    out[["tickets_90d", "tickets_90d_prior", "p1_tickets_90d"]] = 0.0

    end = pd.Timestamp(_utc(as_of) + timedelta(days=1))
    cur_start = end - pd.Timedelta(days=WINDOW_DAYS)
    prior_start = cur_start - pd.Timedelta(days=WINDOW_DAYS)

    rows = []
    for tid, t in tickets.items():
        org = t.get("organization_id")
        if org is None or int(org) not in org_to_account:
            continue
        m = metrics.get(tid) or {}
        res = m.get("full_resolution_time_in_minutes") or {}
        rows.append(
            {
                "account_id": org_to_account[int(org)],
                "created_at": t.get("created_at"),
                "priority": str(t.get("priority") or "").lower(),
                "status": str(t.get("status") or "").lower(),
                "rating": str((t.get("satisfaction_rating") or {}).get("score") or "").lower(),
                "reopens": m.get("reopens"),
                "resolution_min": res.get("calendar") if isinstance(res, dict) else None,
                "has_metrics": bool(m),
            }
        )
    if not rows:
        return out
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df[df["created_at"].notna() & (df["created_at"] < end)]
    cur = df[df["created_at"] >= cur_start]
    prior = df[(df["created_at"] >= prior_start) & (df["created_at"] < cur_start)]

    out["tickets_90d"] = cur.groupby("account_id").size().reindex(accounts).fillna(0).astype(float)
    out["tickets_90d_prior"] = prior.groupby("account_id").size().reindex(accounts).fillna(0).astype(float)
    p1 = cur[cur["priority"].isin(urgent)].groupby("account_id").size()
    out["p1_tickets_90d"] = p1.reindex(accounts).fillna(0).astype(float)

    rated = cur[cur["rating"].isin(["good", "bad"])]
    g = rated.groupby("account_id")["rating"].agg(n="size", good=lambda s: (s == "good").sum())
    g = g[g["n"] >= MIN_FOR_RATE]
    out["csat"] = (g["good"] / g["n"] * 100).round(1).reindex(accounts)

    solved = cur[cur["status"].isin(SOLVED) & cur["has_metrics"]]
    reo = solved[solved["reopens"].notna()]
    r = reo.groupby("account_id")["reopens"].agg(n="size", reopened=lambda s: (pd.to_numeric(s) > 0).sum())
    r = r[r["n"] >= MIN_FOR_RATE]
    out["ticket_reopen_rate_pct"] = (r["reopened"] / r["n"] * 100).round(1).reindex(accounts)

    fix = solved[pd.to_numeric(solved["resolution_min"], errors="coerce").notna()].copy()
    fix["hours"] = pd.to_numeric(fix["resolution_min"]) / 60.0
    h = fix.groupby("account_id")["hours"].agg(n="size", median="median")
    h = h[h["n"] >= MIN_FOR_RATE]
    out["ticket_resolution_hours"] = h["median"].round(1).reindex(accounts)
    return out


# -- simulated Zendesk ----------------------------------------------------------------------------------------
class SimulatedZendesk:
    """A stand-in Zendesk API built from the book, for demos and tests.

    It answers the same endpoints, with the same pagination, as the real API, so demo mode exercises every
    line of the real add-on except the network call. Ticket volumes follow the book's own counts where it has
    them, so the demo stays coherent, and some accounts are deliberately hard to match (name only, missing)
    so the match report has something to say.
    """

    def __init__(self, accounts: pd.DataFrame | None, as_of: date, *, seed: int = 11, page_size: int = 1000):
        self.page_size = page_size
        self.requests = 0
        self.orgs: list[dict[str, Any]] = []
        self.tickets: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []
        if accounts is not None and len(accounts):
            self._build(accounts.reset_index(drop=True), as_of, np.random.default_rng(seed))

    def _build(self, acc: pd.DataFrame, as_of: date, rng: np.random.Generator) -> None:
        end = _utc(as_of) + timedelta(days=1)
        next_tid = 1000

        def col(name: str, default: float) -> np.ndarray:
            if name in acc.columns:
                return pd.to_numeric(acc[name], errors="coerce").fillna(default).to_numpy(float)
            return np.full(len(acc), default)

        seats = col("licensed_seats", 50)
        cur_n = col("tickets_90d", np.nan)
        cur_n = np.where(np.isnan(cur_n), rng.poisson(np.maximum(seats * 0.07, 2)), cur_n)
        p1 = col("p1_tickets_90d", 0)
        csat = col("csat", 88)
        trend = col("usage_trend_pct", 0)
        names = acc["account_name"].astype(str) if "account_name" in acc.columns else acc["account_id"].astype(str)

        for i in range(len(acc)):
            row = acc.iloc[i]
            u = rng.random()
            if u < 0.04:
                continue  # never set up in Zendesk
            org_id = 900_000 + i
            ext = None if u < 0.12 else str(row["account_id"])  # some orgs only carry the company name
            name = f"{names[i]} Inc." if ext is None else names[i]
            self.orgs.append({"id": org_id, "name": name, "external_id": ext, "organization_fields": {}})

            n_cur = int(min(cur_n[i], 400))
            n_prior = int(min(max(0, round(n_cur * rng.lognormal(0, 0.3))), 400))
            if trend[i] <= -25 and rng.random() < 0.5:
                n_cur, n_prior = int(n_cur * 0.2), max(8, n_cur * 3)  # usage fell and support went quiet
            sat = float(np.clip(csat[i], 30, 100))
            reopen_p = float(np.clip(0.06 + 0.006 * (90 - sat), 0.02, 0.6))
            median_h = float(np.clip(18 * (1 + (90 - sat) / 15), 4, 300))
            for k, (n, lo, hi) in enumerate(((n_cur, 0, WINDOW_DAYS), (n_prior, WINDOW_DAYS, 2 * WINDOW_DAYS))):
                for j in range(n):
                    created = end - timedelta(days=float(rng.uniform(lo, hi)), seconds=1)
                    solved = rng.random() < 0.85
                    urgent = k == 0 and j < p1[i]
                    rating: dict[str, Any] = {"score": "unoffered"}
                    if solved and rng.random() < 0.4:
                        rating = {"score": "good" if rng.random() < sat / 100 else "bad"}
                    tid = next_tid
                    next_tid += 1
                    self.tickets.append(
                        {
                            "id": tid,
                            "organization_id": org_id,
                            "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "updated_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "priority": "urgent" if urgent else str(rng.choice(["normal", "normal", "low", "high"])),
                            "status": str(rng.choice(["solved", "closed"])) if solved else "open",
                            "satisfaction_rating": rating,
                        }
                    )
                    minutes = float(rng.lognormal(np.log(median_h * 60), 0.7)) if solved else None
                    self.metrics.append(
                        {
                            "ticket_id": tid,
                            "reopens": int(rng.random() < reopen_p) if solved else 0,
                            "replies": int(rng.integers(1, 6)),
                            "full_resolution_time_in_minutes": {"calendar": minutes, "business": minutes},
                        }
                    )
        for k in range(3):  # organizations with no account in the book (trials, partners)
            self.orgs.append({"id": 990_000 + k, "name": f"Trial workspace {k + 1}", "external_id": None})

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, str], bytes]:
        self.requests += 1
        parts = urllib.parse.urlsplit(url)
        q = dict(urllib.parse.parse_qsl(parts.query))
        base = f"{parts.scheme}://{parts.netloc}{parts.path}"
        if parts.path == "/api/v2/users/me.json":
            body: dict[str, Any] = {"user": {"id": 1, "name": "Demo admin", "role": "admin"}}
        elif parts.path == "/api/v2/organizations.json":
            size = int(q.get("page[size]", 100))
            start = int(q.get("page[after]", 0))
            chunk = self.orgs[start : start + size]
            more = start + size < len(self.orgs)
            nxt = f"{base}?{urllib.parse.urlencode({'page[size]': size, 'page[after]': start + size})}"
            body = {
                "organizations": chunk,
                "meta": {"has_more": more, "after_cursor": str(start + size) if more else None},
                "links": {"next": nxt if more else None},
            }
        elif parts.path == "/api/v2/incremental/tickets/cursor.json":
            start = int(q.get("cursor", 0))
            stop = start + self.page_size
            done = stop >= len(self.tickets)
            body = {
                "tickets": self.tickets[start:stop],
                "metric_sets": self.metrics[start:stop],
                "after_cursor": None if done else str(stop),
                "after_url": None if done else f"{base}?cursor={stop}",
                "end_of_stream": done,
            }
        else:
            return 404, {}, b'{"error":"RecordNotFound"}'
        return 200, {"content-type": "application/json"}, json.dumps(body).encode()
