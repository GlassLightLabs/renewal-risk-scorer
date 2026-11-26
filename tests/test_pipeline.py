from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renewal_risk.config import load_config
from renewal_risk.pipeline import AI_NATIVE_GRR_BENCHMARK, by_plan_size, forecast_grr, score
from renewal_risk.playbook import recommend, timing
from renewal_risk.report import markdown_report


@pytest.mark.parametrize("profile", ["saas", "consumption", "hardware", "services", "ai_usage"])
def test_end_to_end(trained, books, as_of, profile):
    _, model = trained[profile]
    cfg = load_config(profile)
    r = score(books[profile], cfg, model, as_of=as_of, horizon_days=180, simulations=500)
    a = r.accounts
    assert len(a) == len(books[profile])
    assert list(a["rank"]) == list(range(1, len(a) + 1))
    assert a["priority_arr"].is_monotonic_decreasing
    assert a["churn_probability"].between(0, 1).all()
    expected = a["arr"] * (
        a["churn_probability"] + (1 - a["churn_probability"]) * a["contraction_probability"] * model.contraction_depth
    )
    assert np.allclose(a["expected_arr_loss"], expected.round(0), atol=1)
    assert not (a["watermelon"] & (a["health_band"] != "green")).any()
    assert set(a["risk_tier"]) <= {"critical", "high", "watch", "low"}
    high_alert = a["alerts"].map(lambda xs: any(x["severity"] == "high" for x in xs))
    assert (a.loc[high_alert, "risk_tier"] != "low").all()
    fc = r.forecast
    assert 0 <= fc["grr_p10"] <= fc["grr_p50"] <= fc["grr_p90"] <= 1
    assert r.summary["accounts"] == len(a) and r.summary["profile"] == profile
    assert a["save_play"].notna().all() and a["play_actions"].map(len).min() >= 1
    assert len(r.drivers) > 0 and r.drivers["share_of_risk"].sum() == pytest.approx(1, abs=0.01)


def test_planted_stories_land(trained, books, as_of):
    _, model = trained["ai_usage"]
    a = score(books["ai_usage"], load_config("ai_usage"), model, as_of=as_of).accounts.set_index("account_id")
    assert a.loc["A-90003", "risk_tier"] == "critical"  # stalled pilot, overdue
    assert a.loc["A-90003", "save_play"] == "Pilot-to-production push"
    assert a.loc["A-90004", "risk_tier"] == "low"  # healthy control
    assert "seat_compression" in {x["id"] for x in a.loc["A-90005", "alerts"]}
    assert "bill_shock" in {x["id"] for x in a.loc["A-90002", "alerts"]}
    assert "model_retirement" in {x["id"] for x in a.loc["A-90001", "alerts"]}


def test_scoring_without_a_model_uses_the_scorecard(books, as_of):
    r = score(books["saas"], load_config("saas"), None, as_of=as_of)
    a = r.accounts
    assert (a["churn_probability"] == a["scorecard_churn_probability"]).all()
    assert "scorecard" in r.summary["risk_source"]
    assert a["risk_drivers"].map(len).sum() > 0


def test_profile_mismatch_is_refused(trained, books, as_of):
    _, model = trained["saas"]
    with pytest.raises(ValueError, match="profile"):
        score(books["hardware"], load_config("hardware"), model, as_of=as_of)


def test_forecast_math():
    arr = np.array([100.0, 100.0])
    fc = forecast_grr(arr, np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.25, simulations=200)
    assert fc["expected_grr"] == 0.5 and fc["grr_p10"] == fc["grr_p90"] == 0.5
    assert forecast_grr(np.array([]), np.array([]), np.array([]), 0.25)["accounts"] == 0


def test_playbook_prefers_high_alerts_then_drivers():
    drivers = [{"group": "product_adoption", "label": "x", "value": "y", "effect": 1.0}]
    alerts = [
        {"id": "detractor", "severity": "medium", "message": "m"},
        {"id": "competitor_eval", "severity": "high", "message": "m"},
    ]
    assert recommend("saas", drivers, alerts, 40)["play"] == "Competitive defense"
    assert recommend("saas", drivers, [], 40)["play"] == "Adoption recovery"
    assert recommend("saas", [], [], 40, watermelon=True)["play"] == "Watermelon check"
    assert recommend("saas", [], [], 400)["play"] == "Standard renewal motion"
    assert timing(-5).startswith("Overdue") and timing(60).startswith("Negotiation")


def test_plan_size_bands_split_the_book_by_monthly_spend():
    accounts = pd.DataFrame(
        {
            "arr": [12 * 30, 12 * 40, 12 * 100, 12 * 5_000, 12 * 20_000],
            "expected_arr_loss": [12 * 30 * 0.8, 12 * 40 * 0.5, 12 * 100 * 0.5, 12 * 5_000 * 0.1, 0.0],
        }
    )
    rows = by_plan_size(accounts, "ai_usage")
    assert [r["band"] for r in rows] == ["Under $50/mo", "$50-249/mo", "$1k-9.9k/mo", "$10k+/mo"]
    small = rows[0]
    assert small["accounts"] == 2 and small["arr"] == 12 * 70
    assert small["expected_grr"] == pytest.approx(1 - (12 * 30 * 0.8 + 12 * 40 * 0.5) / (12 * 70), abs=1e-4)
    assert rows[-1]["expected_grr"] == 1.0
    assert all(r["benchmark_grr"] == AI_NATIVE_GRR_BENCHMARK[r["band"]] for r in rows)
    # Other profiles get the split without the AI-native benchmark.
    assert all("benchmark_grr" not in r for r in by_plan_size(accounts, "saas"))


def test_report_shows_retention_by_plan_size(trained, books, as_of):
    _, model = trained["ai_usage"]
    r = score(books["ai_usage"], load_config("ai_usage"), model, as_of=as_of, simulations=200)
    bands = r.summary["by_plan_size"]
    assert sum(b["accounts"] for b in bands) == len(r.accounts)
    md = markdown_report(r)
    assert "## Retention by plan size" in md and "AI-native median GRR" in md


def test_deal_signals_from_the_crm_raise_alerts_and_plays(books, as_of):
    a = score(books["saas"], load_config("saas"), None, as_of=as_of).accounts.set_index("account_id")
    quarry = {x["id"] for x in a.loc["A-90003", "alerts"]}  # first renewal, signed at 40% off, signer gone
    assert {"signer_gone", "discount_cliff"} <= quarry
    ironleaf = {x["id"] for x in a.loc["A-90004", "alerts"]}
    assert not {"signer_gone", "discount_cliff"} & ironleaf
    play = recommend("saas", [], [{"id": "signer_gone", "severity": "medium", "message": ""}], 120)
    assert play["play"] == "Find the new economic buyer"
    play = recommend("saas", [], [{"id": "discount_cliff", "severity": "medium", "message": ""}], 120)
    assert play["play"] == "Plan the discount step-down"
