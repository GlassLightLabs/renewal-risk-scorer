from __future__ import annotations

import numpy as np
import pytest

from renewal_risk.config import load_config
from renewal_risk.pipeline import forecast_grr, score
from renewal_risk.playbook import recommend, timing


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
