"""Jira, HubSpot, ChurnZero and Pendo add-ons, plus the shared export helpers. No network: API calls go
through a scripted transport, and file modes read small CSVs written by the tests."""

from __future__ import annotations

import base64
import json
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from renewal_risk.config import load_config
from renewal_risk.ingest import ingest, load_mapping
from renewal_risk.scorecard import evaluate_alerts
from renewal_risk.sources import SourceError, apply_sources, discover, load_sources_config
from renewal_risk.sources.churnzero import ChurnZeroSource
from renewal_risk.sources.exports import days_since, match_records, normalize_export, read_export, timestamps
from renewal_risk.sources.hubspot import HubSpotSource
from renewal_risk.sources.jira import COLUMNS as JIRA_COLUMNS
from renewal_risk.sources.jira import JiraSource, clean_site
from renewal_risk.sources.pendo import PendoSource

AS_OF = date(2026, 11, 26)
ROOT = Path(__file__).resolve().parents[1]
TOKEN = "s3cr3t-token-value"


def book() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["A1", "A2", "A3", "A4"],
            "account_name": ["Acme Corp", "Globex, Inc.", "Initech", "Umbrella"],
            "arr": [100_000.0, 50_000.0, 80_000.0, 20_000.0],
        }
    )


def write(tmp_path: Path, name: str, rows: list[dict[str, Any]]) -> str:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def ago(days: float, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return (pd.Timestamp("2026-11-26T12:00:00Z") - pd.Timedelta(days=days)).strftime(fmt)


class Scripted:
    """Answers by URL path from a queue of (status, body) responses, and records every call."""

    def __init__(self, routes: dict[str, list[tuple[int, Any]]]):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((url, headers))
        queue = self.routes.get(urllib.parse.urlsplit(url).path)
        if not queue:
            return 404, {}, b"{}"
        status, body = queue.pop(0) if len(queue) > 1 else queue[0]
        return status, {}, json.dumps(body).encode()


# -- shared export helpers ------------------------------------------------------------------------------------
def test_normalize_export_maps_aliases_and_joins_repeated_headers() -> None:
    raw = pd.DataFrame(
        [["ENG-1", "Bug", "escalation", "p1", "", "A1"]],
        columns=["Issue key", "Issue Type", "Labels", "Labels.1", "Labels.2", "Custom field (Organizations)"],
    )
    out = normalize_export(raw, JIRA_COLUMNS, service="Jira")
    assert set(out.columns) == {"key", "issue_type", "labels", "customer"}
    assert out.loc[0, "labels"] == "escalation,p1"
    assert out.loc[0, "customer"] == "A1"


def test_missing_columns_are_named_with_what_was_found(tmp_path: Path) -> None:
    path = write(tmp_path, "x.csv", [{"Summary": "a", "Owner": "b"}])
    with pytest.raises(SourceError) as e:
        read_export(path, JIRA_COLUMNS, service="Jira", required=("created",))
    assert e.value.kind == "config"
    assert "created" in str(e.value) and "Summary" in str(e.value)


def test_missing_and_non_csv_files_explain_themselves(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not found"):
        read_export(tmp_path / "nope.csv", JIRA_COLUMNS, service="Jira")
    xlsx = tmp_path / "export.xlsx"
    xlsx.write_bytes(b"PK")
    with pytest.raises(SourceError, match="save the export as CSV"):
        read_export(xlsx, JIRA_COLUMNS, service="Jira")


def test_timestamps_read_the_shapes_tools_export() -> None:
    ts = timestamps(
        pd.Series(
            [
                "26/Nov/26 3:12 PM",
                "05/Nov/26 9:00 AM",
                "2026-11-05T10:00:00.000+0000",
                "11/05/2026",
                "1762300800000",
                "",
                None,
                "not a date",
            ]
        )
    )
    assert ts[0] == pd.Timestamp("2026-11-26T15:12Z")
    assert ts[1] == pd.Timestamp("2026-11-05T09:00Z")
    assert ts[2] == pd.Timestamp("2026-11-05T10:00Z")
    assert ts[3] == pd.Timestamp("2026-11-05T00:00Z")
    assert ts[4] == pd.Timestamp("2025-11-05T00:00Z")
    assert ts[5:].isna().all()
    assert days_since(pd.Series(["2026-11-26", "2026-11-16", "2026-12-25"]), AS_OF).tolist() == [0.0, 10.0, 0.0]


def test_match_records_by_id_then_name_and_never_guesses() -> None:
    accounts = pd.DataFrame(
        {"account_id": ["A1", "A2", "A3", "A4"], "account_name": ["Acme", "Globex", "Twin", "Twin LLC"]}
    )
    records = pd.DataFrame(
        {
            "account_ref": ["A1", None, None, "X9", None],
            "name": ["whatever", "GLOBEX Inc.", "Twin", "Unknown", None],
        }
    )
    acct, unmatched, warnings = match_records(records, accounts, service="Test")
    assert acct.tolist()[:2] == ["A1", "A2"]
    assert acct[2:].isna().all()  # "Twin" matches two accounts: not guessed
    assert unmatched == 2  # Twin and X9 (the empty row has no key)
    assert warnings and "more than one" in warnings[0]
    by_id, _, _ = match_records(records, accounts, mode="id", service="Test")
    assert by_id.notna().sum() == 1
    with pytest.raises(SourceError):
        match_records(records, accounts, mode="email", service="Test")


# -- Jira -----------------------------------------------------------------------------------------------------
def jira_rows() -> list[dict[str, Any]]:
    fmt = "%d/%b/%y %I:%M %p"
    base = {"Status Category": "To Do", "Priority": "Medium", "Resolved": "", "Labels": ""}

    def issue(key: str, typ: str, age: float, customer: str, **kw: Any) -> dict[str, Any]:
        return {
            **base,
            "Issue key": key,
            "Issue Type": typ,
            "Status": "Open",
            "Created": ago(age, fmt),
            "Custom field (Customer)": customer,
            **kw,
        }

    return [
        issue("E-1", "Bug", 10, "A1"),
        issue("E-2", "Bug", 70, "A1"),
        issue("E-3", "Bug", 130, "A1"),
        issue("E-4", "Bug", 20, "A1", **{"Status": "Done", "Status Category": "Done"}),
        issue("E-5", "Task", 5, "A1", Labels="escalation"),
        issue("E-6", "Story", 30, "A1", Priority="Highest"),
        issue("E-7", "Task", 120, "A1", Labels="escalation"),  # outside the 90-day window
        issue("E-8", "Defect", 15, "Globex Inc", **{"Status Category": "", "Status": "Closed"}),
        issue("E-9", "Bug", 3, ""),  # names no customer
        issue("E-10", "Bug", 3, "Internal QA"),  # not in the book
    ]


def test_jira_file_mode_computes_bugs_age_and_escalations(tmp_path: Path) -> None:
    path = write(tmp_path, "jira.csv", jira_rows())
    src = JiraSource()
    res = src.fetch(src.validate({"file": path}), book(), AS_OF)
    s = res.signals
    assert s.loc["A1", "open_bugs"] == 3
    assert s.loc["A1", "open_bug_age_days"] == 70
    assert s.loc["A1", "escalations_90d"] == 2
    assert s.loc["A2", "open_bugs"] == 0  # its one defect is closed (no status category, status says Closed)
    assert np.isnan(s.loc["A2", "open_bug_age_days"])
    assert res.matched_accounts == 2 and set(res.unmatched_accounts) == {"A3", "A4"}
    assert res.unmatched_records == 1 and res.records_read == 10
    assert any("name no customer" in w for w in res.warnings)
    assert src.test_connection({"file": path}).startswith("Jira export read: 10 issues")


def test_jira_custom_column_and_settings(tmp_path: Path) -> None:
    rows = [{k if k != "Custom field (Customer)" else "Client Ref": v for k, v in r.items()} for r in jira_rows()]
    path = write(tmp_path, "jira.csv", rows)
    src = JiraSource()
    with pytest.raises(SourceError, match="no column names the customer"):
        src.fetch(src.validate({"file": path}), book(), AS_OF)
    cfg = src.validate({"file": path, "customer_field": "Client Ref", "escalation_labels": "nothing"})
    s = src.fetch(cfg, book(), AS_OF).signals
    assert s.loc["A1", "escalations_90d"] == 1  # only the Highest-priority one now


def test_jira_validation_messages() -> None:
    src = JiraSource()
    with pytest.raises(SourceError, match="add an issue export file"):
        src.validate({})
    with pytest.raises(SourceError, match="site, the email and an API token"):
        src.validate({"site": "acme"})
    with pytest.raises(SourceError, match="field ID"):
        src.validate({"site": "acme", "email": "a@b.co", "api_token": "t"})
    cfg = src.validate(
        {
            "site": "https://Acme.atlassian.net/jira/x",
            "email": "a@b.co",
            "api_token": "t",
            "customer_field": "customfield_10050",
        }
    )
    assert cfg["mode"] == "api" and cfg["site"] == "acme.atlassian.net"
    assert clean_site("acme") == "acme.atlassian.net"
    with pytest.raises(SourceError):
        clean_site("not a site")


def api_issue(key: str, typ: str, age: float, org: Any, *, done: bool = False, labels: Any = ()) -> dict[str, Any]:
    created = (pd.Timestamp("2026-11-26T12:00Z") - pd.Timedelta(days=age)).strftime("%Y-%m-%dT%H:%M:%S.000+0000")
    return {
        "key": key,
        "fields": {
            "issuetype": {"name": typ},
            "status": {
                "name": "Done" if done else "In Progress",
                "statusCategory": {"key": "done" if done else "indeterminate"},
            },
            "priority": {"name": "Medium"},
            "created": created,
            "resolutiondate": None,
            "labels": list(labels),
            "customfield_10050": org,
        },
    }


def test_jira_api_paginates_reads_organization_fields_and_hides_the_token() -> None:
    page1 = {
        "issues": [api_issue("E-1", "Bug", 40, [{"id": "7", "name": "Acme Corp"}])],
        "nextPageToken": "abc",
        "isLast": False,
    }
    page2 = {
        "issues": [
            api_issue("E-2", "Bug", 80, {"value": "A3"}),
            api_issue("E-3", "Task", 10, "A3", labels=["Escalation"]),
            api_issue("E-4", "Bug", 5, None),
        ],
        "isLast": True,
    }
    t = Scripted({"/rest/api/3/search/jql": [(200, page1), (200, page2)]})
    src = JiraSource(transport=t, sleep=lambda s: None)
    cfg = src.validate(
        {
            "site": "acme",
            "email": "a@acme.test",
            "api_token": TOKEN,
            "customer_field": "customfield_10050",
            "jql": "project = SUP",
        }
    )
    res = src.fetch(cfg, book(), AS_OF)
    assert res.signals.loc["A1", "open_bugs"] == 1 and res.signals.loc["A1", "open_bug_age_days"] == 40
    assert res.signals.loc["A3", "escalations_90d"] == 1
    assert len(t.calls) == 2 and "nextPageToken=abc" in t.calls[1][0]
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(t.calls[0][0]).query)
    assert q["jql"][0].startswith("(project = SUP) AND (created >= -180d")
    assert "customfield_10050" in q["fields"][0]
    url, headers = t.calls[0]
    assert TOKEN not in url
    assert headers["Authorization"] == "Basic " + base64.b64encode(f"a@acme.test:{TOKEN}".encode()).decode()


def test_jira_bad_credentials_are_reported_without_the_token() -> None:
    t = Scripted({"/rest/api/3/myself": [(401, {})]})
    src = JiraSource(transport=t, sleep=lambda s: None)
    with pytest.raises(SourceError) as e:
        src.test_connection(
            {"site": "acme", "email": "a@acme.test", "api_token": TOKEN, "customer_field": "customfield_1"}
        )
    assert e.value.kind == "auth" and TOKEN not in str(e.value)


# -- HubSpot --------------------------------------------------------------------------------------------------
def hubspot_files(tmp_path: Path) -> tuple[str, str]:
    companies = write(
        tmp_path,
        "companies.csv",
        [
            {
                "Record ID": "1",
                "Company name": "Acme Corp",
                "SFDC ID": "A1",
                "Last Contacted": ago(12),
                "Competitor": "",
            },
            {"Record ID": "2", "Company name": "Globex", "SFDC ID": "", "Last Contacted": ago(40), "Competitor": ""},
            {"Record ID": "3", "Company name": "Initech", "SFDC ID": "A3", "Last Contacted": "", "Competitor": "Rival"},
            {
                "Record ID": "9",
                "Company name": "Prospect Co",
                "SFDC ID": "",
                "Last Contacted": ago(1),
                "Competitor": "",
            },
        ],
    )
    deal = {"Deal Stage": "Proposal", "Pipeline": "Renewals", "Deal Type": "Existing Business", "Discount": ""}
    deals = write(
        tmp_path,
        "deals.csv",
        [
            {
                **deal,
                "Deal Name": "Acme renewal 2027",
                "Amount": "$110,000",
                "Close Date": "2027-02-01",
                "Associated Company IDs (Primary)": "1",
            },
            {
                **deal,
                "Deal Name": "Acme renewal later",
                "Amount": "90000",
                "Close Date": "2028-02-01",
                "Associated Company IDs (Primary)": "1",
            },
            {
                **deal,
                "Deal Name": "Acme old",
                "Amount": "100000",
                "Close Date": "2025-02-01",
                "Deal Stage": "Closed won",
                "Deal Type": "New Business",
                "Pipeline": "Sales",
                "Discount": "0.1",
                "Associated Company IDs (Primary)": "1",
            },
            {
                **deal,
                "Deal Name": "Acme 2024",
                "Amount": "90000",
                "Close Date": "2024-02-01",
                "Deal Stage": "Closed won",
                "Deal Type": "New Business",
                "Pipeline": "Sales",
                "Discount": "35",
                "Associated Company IDs (Primary)": "1",
            },
            {
                **deal,
                "Deal Name": "Globex renewal",
                "Amount": "45000",
                "Close Date": "2027-01-10",
                "Associated Company IDs (Primary)": "2",
            },
            {
                **deal,
                "Deal Name": "Mystery",
                "Amount": "5000",
                "Close Date": "2027-01-10",
                "Associated Company IDs (Primary)": "999",
            },
        ],
    )
    return companies, deals


def test_hubspot_file_mode_signals(tmp_path: Path) -> None:
    companies, deals = hubspot_files(tmp_path)
    src = HubSpotSource()
    cfg = src.validate({"companies_file": companies, "deals_file": deals, "account_id_column": "SFDC ID"})
    res = src.fetch(cfg, book(), AS_OF)
    s = res.signals
    assert s.loc["A1", "days_since_last_meeting"] == 12
    assert s.loc["A1", "price_increase_pct"] == 10.0  # the renewal closing soonest: 110k against 100k ARR
    assert s.loc["A1", "discount_pct"] == 10.0  # the most recent won deal, 0.1 read as 10%
    assert s.loc["A2", "price_increase_pct"] == -10.0  # matched by name; quoted below ARR
    assert s.loc["A3", "competitor_eval"] == 1 and s.loc["A1", "competitor_eval"] == 0
    assert np.isnan(s.loc["A3", "days_since_last_meeting"])
    assert res.matched_accounts == 3 and res.unmatched_records == 1
    assert any("1 deals are on companies" in w for w in res.warnings)
    assert "4 companies and 6 deals" in src.test_connection(cfg)


def test_hubspot_validation() -> None:
    src = HubSpotSource()
    with pytest.raises(SourceError, match="companies export file, or a private app access token"):
        src.validate({})
    with pytest.raises(SourceError, match="companies export too"):
        src.validate({"deals_file": "deals.csv"})
    assert src.validate({"access_token": "x"})["mode"] == "api"


def test_hubspot_api_pages_and_associates_deals() -> None:
    comp1 = {
        "results": [
            {
                "id": "1",
                "properties": {"name": "Acme Corp", "sfdc": "A1", "notes_last_contacted": "2026-11-16T12:00:00.000Z"},
            }
        ],
        "paging": {"next": {"after": "100", "link": "https://api.hubapi.com/x"}},
    }
    comp2 = {"results": [{"id": "2", "properties": {"name": "Initech", "competitor": "Rival"}}]}
    deals = {
        "results": [
            {
                "id": "50",
                "properties": {
                    "dealname": "Acme Renewal",
                    "amount": "105000",
                    "closedate": "2027-01-15T00:00:00Z",
                    "dealstage": "contractsent",
                    "pipeline": "default",
                    "dealtype": "existingbusiness",
                    "hs_is_closed": "false",
                },
                "associations": {"companies": {"results": [{"id": "1", "type": "deal_to_company"}]}},
            }
        ]
    }
    t = Scripted({"/crm/v3/objects/companies": [(200, comp1), (200, comp2)], "/crm/v3/objects/deals": [(200, deals)]})
    src = HubSpotSource(transport=t, sleep=lambda s: None)
    res = src.fetch(src.validate({"access_token": TOKEN, "account_id_column": "sfdc"}), book(), AS_OF)
    assert res.signals.loc["A1", "days_since_last_meeting"] == 10
    assert res.signals.loc["A1", "price_increase_pct"] == 5.0
    assert res.signals.loc["A3", "competitor_eval"] == 1
    assert "after=100" in t.calls[1][0]
    assert all(h["Authorization"] == f"Bearer {TOKEN}" and TOKEN not in u for u, h in t.calls)
    deal_query = urllib.parse.parse_qs(urllib.parse.urlsplit(t.calls[-1][0]).query)
    assert deal_query["associations"] == ["companies"]


# -- ChurnZero ------------------------------------------------------------------------------------------------
def test_churnzero_file_mode(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "cz.csv",
        [
            {
                "Account External ID": "A1",
                "Account Name": "Acme",
                "Active Contacts (Last 30 Days)": "40",
                "Last Login Date": ago(3, "%m/%d/%Y %I:%M %p"),
                "Last NPS Score": "50",
                "CSM Sentiment": "Red",
                "Champion Left": "Yes",
                "ChurnScore": "12",
            },
            {
                "Account External ID": "A1",
                "Account Name": "Acme EU",
                "Active Contacts (Last 30 Days)": "10",
                "Last Login Date": ago(1, "%m/%d/%Y %I:%M %p"),
                "Last NPS Score": "-20",
                "CSM Sentiment": "Green",
                "Champion Left": "No",
            },
            {
                "Account External ID": "",
                "Account Name": "Initech",
                "Active Contacts (Last 30 Days)": "0",
                "Days Since Last Login": "45",
                "Last NPS Score": "",
                "CSM Sentiment": "3",
            },
        ],
    )
    src = ChurnZeroSource()
    res = src.fetch(src.validate({"file": path}), book(), AS_OF)
    s = res.signals
    assert s.loc["A1", "active_users"] == 50  # two ChurnZero accounts on one CRM account add up
    assert s.loc["A1", "days_since_last_login"] == 1
    assert s.loc["A1", "nps"] == 5.8  # account-level -100..100 mapped to 0-10: (7.5 + 4.0) / 2, rounded
    assert s.loc["A1", "champion_change_90d"] == 1
    assert s.loc["A3", "days_since_last_login"] == 45 and s.loc["A3", "active_users"] == 0
    assert s.loc["A3", "csm_sentiment"] == 3
    assert "nps" in src.test_connection({"file": path})
    assert "ChurnScore" not in str(s.columns)
    with pytest.raises(SourceError, match="Account export"):
        src.validate({})


# -- Pendo ----------------------------------------------------------------------------------------------------
def test_pendo_file_mode_adoption_and_matching(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "pendo.csv",
        [
            {
                "Account ID": "A1",
                "Account Name": "Acme",
                "Number of Active Visitors": "30",
                "Active Visitors (Prior Period)": "60",
                "Last Visit": ago(2),
                "Core Features Used": "6",
            },
            {
                "Account ID": "pendo-77",
                "Account Name": "Globex",
                "Number of Active Visitors": "8",
                "Active Visitors (Prior Period)": "",
                "Last Visit": ago(20),
                "Core Features Used": "9",
            },
            {"Account ID": "test", "Account Name": "Pendo test account", "Number of Active Visitors": "2"},
        ],
    )
    src = PendoSource()
    res = src.fetch(src.validate({"file": path}), book(), AS_OF)
    assert np.isnan(res.signals.loc["A1", "feature_adoption_pct"])
    assert any("Number of core features" in w for w in res.warnings)
    res = src.fetch(src.validate({"file": path, "core_features": "12"}), book(), AS_OF)
    s = res.signals
    assert s.loc["A1", "feature_adoption_pct"] == 50.0 and s.loc["A2", "feature_adoption_pct"] == 75.0
    assert s.loc["A1", "active_users"] == 30 and s.loc["A1", "active_users_prior"] == 60
    assert s.loc["A2", "days_since_last_login"] == 20  # matched by name: its Pendo ID is not an account ID
    assert res.unmatched_records == 1
    with pytest.raises(SourceError, match="above zero"):
        src.validate({"file": path, "core_features": "0"})


def test_pendo_fraction_adoption_column(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "p.csv",
        [{"Account ID": "A1", "Core Feature Adoption": "0.42"}, {"Account ID": "A2", "Core Feature Adoption": "0.9"}],
    )
    src = PendoSource()
    s = src.fetch(src.validate({"file": path}), book(), AS_OF).signals
    assert s.loc["A1", "feature_adoption_pct"] == 42.0


# -- together -------------------------------------------------------------------------------------------------
def test_all_add_ons_are_built_in() -> None:
    sources, problems = discover()
    assert {"zendesk", "jira", "hubspot", "churnzero", "pendo"} <= set(sources)
    assert not problems


def test_every_simulated_add_on_runs_and_feeds_the_scorecard(books: dict[str, pd.DataFrame], as_of: date) -> None:
    cfg = {name: {"simulate": True} for name in ("zendesk", "jira", "hubspot", "churnzero", "pendo")}
    out, run = apply_sources(books["saas"], cfg, as_of=as_of)
    assert not run.errors, run.errors
    assert all(r.simulated and r.matched_accounts > 0 for r in run.results)
    for col in ("open_bugs", "escalations_90d", "days_since_last_meeting", "nps", "feature_adoption_pct"):
        assert out[col].notna().any(), col
    # Simulated usage follows the book, so the demo stays coherent.
    before = books["saas"].set_index("account_id")["active_users"]
    after = out.set_index("account_id")["active_users"]
    assert (after - before).abs().max() < 1e-9


def test_jira_alerts_fire() -> None:
    df = pd.DataFrame(
        {"escalations_90d": [2.0, 1.0, np.nan], "open_bugs": [3.0, 5.0, 4.0], "open_bug_age_days": [61.0, 30.0, np.nan]}
    )
    fired = evaluate_alerts(df, load_config("saas").alerts)
    ids = [{a["id"] for a in row} for row in fired]
    assert ids[0] >= {"engineering_escalations", "aging_bugs"}
    assert not ids[1] & {"engineering_escalations", "aging_bugs"}
    assert not ids[2] & {"engineering_escalations", "aging_bugs"}


def test_the_bundled_sample_exports_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    book_df, _ = ingest(
        "examples/salesforce_report.csv", as_of=date(2026, 12, 1), mapping=load_mapping("examples/mapping.yaml")
    )
    out, run = apply_sources(book_df, load_sources_config("examples/sources.exports.yaml"), as_of=date(2026, 12, 1))
    assert not run.errors, run.errors
    assert {r.source for r in run.results} == {"Jira", "HubSpot", "ChurnZero", "Pendo"}
    assert all(r.matched_accounts >= 100 for r in run.results)
    assert out["nps"].notna().sum() > 50 and out["open_bugs"].notna().sum() > 50
