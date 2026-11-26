"""Data-source add-ons, tested against a scripted Zendesk API (no network)."""

from __future__ import annotations

import json
import urllib.parse
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from renewal_risk.ingest import ingest
from renewal_risk.sources import (
    DataSource,
    SourceError,
    SourceResult,
    apply_sources,
    discover,
    interpolate,
    load_sources_config,
    merge_signals,
    normalize_name,
    redact,
)
from renewal_risk.sources import registry as registry_mod
from renewal_risk.sources.http import JsonClient
from renewal_risk.sources.zendesk import (
    SimulatedZendesk,
    ZendeskSource,
    clean_subdomain,
    compute_signals,
    match_organizations,
)

AS_OF = date(2026, 11, 26)
TOKEN = "s3cr3t-token-value"
CFG = {"subdomain": "acme", "email": "admin@acme.test", "api_token": TOKEN}


class Scripted:
    """Answers by URL path from a queue of (status, headers, body) responses, and records every call."""

    def __init__(self, routes: dict[str, list[tuple[int, dict[str, str], Any]]]):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((url, headers))
        path = urllib.parse.urlsplit(url).path
        queue = self.routes.get(path)
        if not queue:
            return 404, {}, b"{}"
        status, hdrs, body = queue.pop(0) if len(queue) > 1 else queue[0]
        return status, hdrs, json.dumps(body).encode() if not isinstance(body, bytes) else body


def ok(body: Any) -> tuple[int, dict[str, str], Any]:
    return 200, {}, body


def ticket(tid: int, org: int | None, days_ago: float, **kw: Any) -> dict[str, Any]:
    created = pd.Timestamp("2026-11-27T00:00:00Z") - pd.Timedelta(days=days_ago)
    return {
        "id": tid,
        "organization_id": org,
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "priority": kw.get("priority", "normal"),
        "status": kw.get("status", "solved"),
        "satisfaction_rating": {"score": kw.get("rating", "unoffered")},
    }


def metric(tid: int, reopens: int = 0, minutes: float | None = 600) -> dict[str, Any]:
    return {"ticket_id": tid, "reopens": reopens, "full_resolution_time_in_minutes": {"calendar": minutes}}


def book() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["A1", "A2", "A3", "A4"],
            "account_name": ["Acme Corp", "Globex, Inc.", "Initech", "Umbrella"],
            "tickets_90d": [1.0, 2.0, 3.0, 4.0],
            "csat": [70.0, 80.0, 90.0, 95.0],
        }
    )


ORGS_PAGE_1 = {
    "organizations": [{"id": 1, "name": "Acme Corp", "external_id": "A1"}],
    "meta": {"has_more": True},
    "links": {"next": "https://acme.zendesk.com/api/v2/organizations.json?page%5Bafter%5D=x"},
}
ORGS_PAGE_2 = {
    "organizations": [
        {"id": 2, "name": "GLOBEX", "external_id": None},  # matched by name
        {"id": 3, "name": "Initech", "external_id": None},  # matched by name, no tickets
        {"id": 9, "name": "Some trial", "external_id": None},  # no account
    ],
    "meta": {"has_more": False},
    "links": {"next": None},
}


def tickets_pages() -> list[tuple[int, dict[str, str], Any]]:
    a1 = [ticket(i, 1, 10 + i, rating="good" if i < 13 else "bad") for i in range(10, 14)]  # 3 good, 1 bad
    a1.append(ticket(20, 1, 5, priority="urgent", status="open"))
    a1.append(ticket(21, 1, 120))  # prior window
    a1.append(ticket(22, 1, 200))  # outside both windows
    a1.append(ticket(23, None, 3))  # no organization
    page1 = {
        "tickets": a1[:3],
        "metric_sets": [metric(10, reopens=1, minutes=60), metric(11, minutes=120), metric(12, minutes=180)],
        "after_url": "https://acme.zendesk.com/api/v2/incremental/tickets/cursor.json?cursor=abc",
        "end_of_stream": False,
    }
    page2 = {
        "tickets": a1[3:] + [ticket(30, 2, 1), dict(ticket(10, 1, 20, rating="good"), status="closed")],
        "metric_sets": [metric(13, minutes=240), metric(30, minutes=None)],
        "after_url": None,
        "end_of_stream": True,
    }
    return [ok(page1), ok(page2)]


def scripted_zendesk() -> Scripted:
    return Scripted(
        {
            "/api/v2/organizations.json": [ok(ORGS_PAGE_1), ok(ORGS_PAGE_2)],
            "/api/v2/incremental/tickets/cursor.json": tickets_pages(),
            "/api/v2/users/me.json": [ok({"user": {"id": 5, "name": "Pat", "role": "admin"}})],
        }
    )


def test_fetch_follows_pagination_and_computes_signals() -> None:
    t = scripted_zendesk()
    res = ZendeskSource(transport=t, sleep=lambda s: None).fetch(ZendeskSource().validate(CFG), book(), AS_OF)
    s = res.signals
    assert res.matched_accounts == 3 and res.unmatched_accounts == ["A4"] and res.unmatched_records == 1
    assert s.loc["A1", "tickets_90d"] == 5  # 4 rated + 1 urgent; the duplicate of ticket 10 counts once
    assert s.loc["A1", "tickets_90d_prior"] == 1
    assert s.loc["A1", "p1_tickets_90d"] == 1
    assert s.loc["A1", "csat"] == 75.0  # 3 good of 4 rated
    assert s.loc["A1", "ticket_reopen_rate_pct"] == 25.0  # 1 of 4 solved reopened
    assert s.loc["A1", "ticket_resolution_hours"] == 2.5  # median of 1, 2, 3, 4 hours
    assert s.loc["A2", "tickets_90d"] == 1 and np.isnan(s.loc["A2", "csat"])  # too few to rate
    assert s.loc["A3", "tickets_90d"] == 0  # matched, no tickets: a real zero
    assert any("no organization" in w for w in res.warnings)
    # The incremental export is asked for metrics, without deleted tickets, from 180 days back.
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(t.calls[2][0]).query))
    assert q["include"] == "metric_sets" and q["exclude_deleted"] == "true"
    assert int(q["start_time"]) == int(pd.Timestamp("2026-05-31T00:00:00Z").timestamp())
    # Basic auth as email/token:api_token, sent only in the header.
    assert t.calls[0][1]["Authorization"].startswith("Basic ")
    assert all(TOKEN not in url for url, _ in t.calls)


def test_rate_limit_honours_retry_after() -> None:
    waits: list[float] = []
    t = Scripted({"/x": [(429, {"retry-after": "7"}, {}), ok({"fine": True})]})
    client = JsonClient("https://acme.zendesk.com", transport=t, sleep=waits.append)
    assert client.get("/x") == {"fine": True}
    assert waits == [7.0]


def test_rate_limit_gives_up_when_the_wait_is_too_long() -> None:
    t = Scripted({"/x": [(429, {"retry-after": "3600"}, {})]})
    with pytest.raises(SourceError) as e:
        JsonClient("https://acme.zendesk.com", transport=t, sleep=lambda s: None, service="Zendesk").get("/x")
    assert e.value.kind == "rate_limit"


def test_server_errors_back_off_then_fail_cleanly() -> None:
    waits: list[float] = []
    t = Scripted({"/x": [(503, {}, {})]})
    with pytest.raises(SourceError) as e:
        JsonClient("https://acme.zendesk.com", transport=t, sleep=waits.append, max_retries=3).get("/x")
    assert e.value.kind == "network" and waits == [1.0, 2.0, 4.0]


def test_bad_credentials_are_reported_without_the_token() -> None:
    t = Scripted({"/api/v2/users/me.json": [(401, {}, {"error": "Couldn't authenticate you"})]})
    with pytest.raises(SourceError) as e:
        ZendeskSource(transport=t).test_connection(CFG)
    assert e.value.kind == "auth"
    assert TOKEN not in str(e.value) and TOKEN not in repr(e.value)


def test_anonymous_user_counts_as_bad_credentials() -> None:
    t = Scripted({"/api/v2/users/me.json": [ok({"user": {"id": None, "name": "Anonymous user"}})]})
    with pytest.raises(SourceError) as e:
        ZendeskSource(transport=t).test_connection(CFG)
    assert e.value.kind == "auth"


def test_connection_message_names_the_user() -> None:
    msg = ZendeskSource(transport=scripted_zendesk()).test_connection(CFG)
    assert "acme.zendesk.com" in msg and "Pat" in msg


def test_client_refuses_to_follow_links_to_other_hosts() -> None:
    client = JsonClient("https://acme.zendesk.com", transport=Scripted({}))
    with pytest.raises(SourceError):
        client.get("https://evil.example.com/api/v2/tickets.json")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("acme", "acme"), ("ACME.zendesk.com", "acme"), ("https://acme-eu.zendesk.com/agent", "acme-eu")],
)
def test_subdomain_accepts_what_people_paste(raw: str, expected: str) -> None:
    assert clean_subdomain(raw) == expected


@pytest.mark.parametrize("raw", ["", "acme.evil.com", "a b", "-acme"])
def test_subdomain_rejects_other_hosts(raw: str) -> None:
    with pytest.raises(SourceError):
        clean_subdomain(raw)


def test_validation_messages_are_specific() -> None:
    with pytest.raises(SourceError, match="API token"):
        ZendeskSource().validate({"subdomain": "acme", "email": "a@b.c"})
    with pytest.raises(SourceError, match="email"):
        ZendeskSource().validate({**CFG, "email": "not-an-email"})
    with pytest.raises(SourceError, match="Match organizations by"):
        ZendeskSource().validate({**CFG, "match_on": "phone"})
    assert ZendeskSource().validate({**CFG, "match_on": "field:sfdc_id"})["match_on"] == "field:sfdc_id"
    assert ZendeskSource().validate({"simulate": True})["subdomain"] == "demo"


def test_matching_modes() -> None:
    accounts = book()
    orgs = [
        {"id": 1, "name": "Something else", "external_id": "A1"},
        {"id": 2, "name": "Globex Incorporated", "external_id": None},
        {"id": 3, "name": "Initech", "organization_fields": {"sfdc_id": "A3"}},
    ]
    m, unmatched, _ = match_organizations(orgs, accounts, "external_id")
    assert m == {1: "A1"} and unmatched == 2
    m, _, _ = match_organizations(orgs, accounts, "auto")
    assert m == {1: "A1", 2: "A2", 3: "A3"}
    m, _, _ = match_organizations(orgs, accounts, "field:sfdc_id")
    assert m == {3: "A3"}


def test_duplicate_names_are_never_guessed() -> None:
    orgs = [{"id": 1, "name": "Acme"}, {"id": 2, "name": "ACME, Inc."}]
    m, unmatched, warnings = match_organizations(orgs, book(), "name")
    assert m == {} and unmatched == 2 and warnings


def test_normalize_name() -> None:
    assert normalize_name("Acme, Inc.") == normalize_name("ACME") == "acme"
    assert normalize_name("Smith & Sons Ltd") == "smith and sons"


def test_merge_only_adds_information() -> None:
    sig = pd.DataFrame(
        {"tickets_90d": [0.0, 9.0], "csat": [np.nan, 60.0], "ticket_reopen_rate_pct": [np.nan, 30.0]},
        index=pd.Index(["A1", "A2"], name="account_id"),
    )
    out = merge_signals(book(), SourceResult(source="Zendesk", signals=sig))
    assert out["tickets_90d"].tolist() == [0.0, 9.0, 3.0, 4.0]  # matched get add-on values, others keep CRM
    assert out["csat"].tolist() == [70.0, 60.0, 90.0, 95.0]  # a rate the add-on can't compute keeps the CRM value
    assert out.loc[1, "ticket_reopen_rate_pct"] == 30.0 and np.isnan(out.loc[0, "ticket_reopen_rate_pct"])


def test_env_interpolation(tmp_path: Any) -> None:
    assert interpolate({"a": "${X}", "b": ["x-${X}"]}, {"X": "1"}) == {"a": "1", "b": ["x-1"]}
    with pytest.raises(SourceError, match="ZENDESK_API_TOKEN"):
        interpolate("${ZENDESK_API_TOKEN}", {})
    f = tmp_path / "sources.yaml"
    f.write_text("sources:\n  zendesk:\n    subdomain: acme\n    api_token: ${TOK}\n  other:\n    enabled: false\n")
    cfg = load_sources_config(f, {"TOK": TOKEN})
    assert cfg == {"zendesk": {"subdomain": "acme", "api_token": TOKEN}}


def test_bad_settings_files_explain_themselves(tmp_path: Any) -> None:
    with pytest.raises(SourceError, match="not found"):
        load_sources_config(tmp_path / "missing.yaml")
    f = tmp_path / "s.yaml"
    f.write_text("sources: [1, 2]\n")
    with pytest.raises(SourceError, match="mapping"):
        load_sources_config(f)


def test_redact_hides_secrets() -> None:
    assert redact(CFG, ZendeskSource.fields)["api_token"] == "••••"


def test_one_failing_add_on_never_stops_scoring() -> None:
    df = book()
    out, run = apply_sources(df, {"zendesk": {"subdomain": "acme"}, "nope": {}}, as_of=AS_OF)
    assert set(run.errors) == {"zendesk", "nope"} and not run.results
    assert out.equals(df)
    assert any("not used" in line for line in run.lines())


class _Fake(DataSource):
    name = "fake"
    label = "Fake"
    provides = ("csat",)

    def fetch(self, config: dict[str, Any], accounts: pd.DataFrame, as_of: date) -> SourceResult:
        return SourceResult(source=self.label, signals=pd.DataFrame(index=pd.Index([], name="account_id")))


class _EP:
    def __init__(self, name: str, obj: Any = None, exc: Exception | None = None):
        self.name, self.value, self._obj, self._exc = name, f"pkg:{name}", obj, exc

    def load(self) -> Any:
        if self._exc:
            raise self._exc
        return self._obj


def test_entry_point_add_ons_are_discovered_and_broken_ones_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    eps = [_EP("fake", _Fake), _EP("broken", exc=ImportError("no module")), _EP("odd", obj=object)]
    monkeypatch.setattr(registry_mod, "entry_points", lambda group: eps)
    sources, problems = discover()
    assert {"zendesk", "fake"} <= set(sources)
    assert len(problems) == 2 and any("broken" in p for p in problems)


def test_simulated_zendesk_runs_the_real_code_path() -> None:
    from renewal_risk.synth import generate_book

    df, _ = ingest(generate_book(150, profile="saas", as_of=AS_OF), as_of=AS_OF)
    out, run = apply_sources(df, {"zendesk": {"simulate": True}}, as_of=AS_OF)
    (res,) = run.results
    assert res.simulated and res.matched_accounts > 120 and res.unmatched_records >= 3
    assert out["ticket_reopen_rate_pct"].notna().sum() > 20
    assert {"support_strain_flag", "ticket_volume_change_pct", "support_silence_flag"} <= set(out.columns)


def test_simulated_pagination_matches_the_api() -> None:
    sim = SimulatedZendesk(book(), AS_OF, page_size=5)
    client = JsonClient("https://demo.zendesk.com", transport=sim)
    first = client.get("api/v2/incremental/tickets/cursor.json", {"start_time": 0})
    assert first["end_of_stream"] is (len(sim.tickets) <= 5)
    assert compute_signals({}, {}, {}, AS_OF).empty


def test_errors_survive_pickling() -> None:
    import pickle

    err = pickle.loads(pickle.dumps(SourceError("auth", "Zendesk refused the credentials.", status=401)))
    assert err.kind == "auth" and err.status == 401 and "Check the email" in str(err)
