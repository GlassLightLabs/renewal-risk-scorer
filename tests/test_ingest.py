from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renewal_risk.ingest import IngestError, ingest, map_columns, parse_bool, parse_number, parse_sentiment


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$1,200", 1200.0),
        ("1.2M", 1_200_000.0),
        ("45k", 45_000.0),
        ("(500)", -500.0),
        ("12%", 12.0),
        ("", np.nan),
        ("n/a", np.nan),
        (3, 3.0),
    ],
)
def test_parse_number(raw, expected):
    got = parse_number(raw)
    assert (np.isnan(got) and np.isnan(expected)) or got == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Yes", 1.0),
        ("TRUE", 1.0),
        ("x", 1.0),
        ("no", 0.0),
        ("0", 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        ("1.0", 1.0),
        ("", np.nan),
        (None, np.nan),
        ("maybe", np.nan),
        (2.0, np.nan),
        (True, 1.0),
    ],
)
def test_parse_bool_keeps_unknown_distinct_from_false(raw, expected):
    got = parse_bool(raw)
    assert (np.isnan(got) and np.isnan(expected)) or got == expected


def test_parse_sentiment_words_and_numbers():
    assert parse_sentiment("Red") == 1.0
    assert parse_sentiment("green") == 5.0
    assert parse_sentiment("4") == 4.0


def test_salesforce_style_headers_map_without_a_template():
    mapped, unmapped = map_columns(["Account ID", "ARR__c", "Renewal_Date__c", "Licensed Users", "Favorite Color"])
    assert mapped == {
        "Account ID": "account_id",
        "ARR__c": "arr",
        "Renewal_Date__c": "renewal_date",
        "Licensed Users": "licensed_seats",
    }
    assert unmapped == ["Favorite Color"]


def test_crm_deal_terms_map_from_common_headers():
    mapped, _ = map_columns(["Discount__c", "Contract_Signer_Left__c"])
    assert mapped == {"Discount__c": "discount_pct", "Contract_Signer_Left__c": "signer_departed"}
    mapped, _ = map_columns(["Deal Discount", "Economic Buyer Left"])
    assert mapped == {"Deal Discount": "discount_pct", "Economic Buyer Left": "signer_departed"}


def _messy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Account ID": ["A1", "A2", "A3", "", "A5", "A1"],
            "Account Name": ["Acme", "Beta", "Gamma", "Nobody", "Epsilon", "Acme dupe"],
            "ARR": ["$120,000", "85k", "1.1M", "10k", "abc", "$120,000"],
            "Renewal Date": ["2027-01-15", "2027-03-01", "2026-12-20", "2027-02-01", "2027-02-01", "2027-01-15"],
            "Licensed Users": ["100", "50", "400", "10", "20", "100"],
            "Active Users": ["80", "10", "390", "5", "20", "80"],
            "CSAT": ["4.5", "3", "5", "", "4", "4.5"],
            "Feature Adoption": ["0.6", "0.2", "0.9", "", "0.5", "0.6"],
            "Exec Sponsor": ["Yes", "No", "", "No", "Yes", "Yes"],
            "CSM Sentiment": ["Green", "Red", "4", "", "Yellow", "Green"],
        }
    )


def test_ingest_cleans_messy_export(as_of):
    df, report = ingest(_messy(), as_of=as_of)
    assert list(df["account_id"]) == ["A1", "A2", "A3"]  # blank id, bad ARR and duplicate dropped
    assert sum(report.dropped.values()) == 3
    assert df.loc[0, "arr"] == 120_000 and df.loc[2, "arr"] == 1_100_000
    assert df.loc[0, "csat"] == pytest.approx(90.0)  # 1-5 scale converted to percent
    assert df.loc[0, "feature_adoption_pct"] == pytest.approx(60.0)  # fractions converted to percent
    assert np.isnan(df.loc[2, "exec_sponsor"])  # unknown stays unknown
    assert df.loc[1, "csm_sentiment"] == 1.0
    assert df.loc[0, "seat_utilization"] == pytest.approx(0.8)
    assert df.loc[2, "days_to_renewal"] == 19
    assert any("fractions" in w for w in report.warnings)


def test_missing_required_column_is_a_clear_error(as_of):
    with pytest.raises(IngestError, match="renewal_date"):
        ingest(pd.DataFrame({"Account ID": ["A1"], "ARR": ["100"]}), as_of=as_of)


def test_history_requires_outcome(as_of):
    with pytest.raises(IngestError, match="outcome"):
        ingest(_messy(), as_of=as_of, require_outcome=True)


def test_mapping_override(as_of):
    raw = _messy().rename(columns={"ARR": "Contract Value (Annual)"})
    df, _ = ingest(raw, as_of=as_of, mapping={"Contract Value (Annual)": "arr"})
    assert df.loc[0, "arr"] == 120_000


def test_ai_derivations(as_of):
    raw = pd.DataFrame(
        {
            "account_id": ["X"],
            "arr": ["100000"],
            "renewal_date": ["2027-06-01"],
            "term_start_date": ["2026-06-01"],
            "committed_spend": ["100000"],
            "consumed_to_date": ["30000"],
            "consumption_30d": ["5000"],
            "consumption_30d_prior": ["4000"],
            "licensed_seats": ["70"],
            "licensed_seats_prior": ["100"],
        }
    )
    df, _ = ingest(raw, as_of=as_of)
    r = df.iloc[0]
    assert r["seat_change_pct"] == pytest.approx(-30.0)
    assert r["consumption_trend_pct"] == pytest.approx(25.0)
    assert r["seat_compression"] == 1.0
    # 30k used + 182 days left at 5k per 30.44 days ~= 59.9k of a 100k commit
    assert r["projected_commit_use_pct"] == pytest.approx(59.9, abs=0.5)
