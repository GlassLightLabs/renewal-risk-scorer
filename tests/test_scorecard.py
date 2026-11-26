from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renewal_risk.config import Alert, load_config
from renewal_risk.scorecard import compute_health, evaluate_alerts, score_measure, urgency


def test_score_measure_interpolates_and_keeps_missing():
    s = score_measure(pd.Series([0.1, 0.5, np.nan, 2.0]), ((0.2, 0), (0.6, 100)))
    assert s.iloc[0] == 0 and s.iloc[1] == 75 and np.isnan(s.iloc[2]) and s.iloc[3] == 100


@pytest.mark.parametrize("profile", ["saas", "consumption", "hardware", "services", "ai_usage"])
def test_points_lost_add_up_to_the_gap(books, profile):
    sc = compute_health(books[profile], load_config(profile))
    gap = 100 - sc.health["health_score"]
    assert np.allclose(sc.points_lost.sum(axis=1), gap, atol=0.3)


def test_missing_measures_are_reweighted_not_zeroed():
    cfg = load_config("saas")
    full = pd.DataFrame({"seat_utilization": [0.9], "usage_trend_pct": [15.0], "nps": [10.0]})
    sparse = pd.DataFrame({"seat_utilization": [0.9]})
    h_full = compute_health(full, cfg).health.iloc[0]
    h_sparse = compute_health(sparse, cfg).health.iloc[0]
    assert h_sparse["health_score"] == 100.0  # the one known measure is perfect
    assert h_sparse["coverage"] < h_full["coverage"]
    assert h_sparse["confidence"] == "low"


def test_bands():
    cfg = load_config("saas")
    df = pd.DataFrame({"seat_utilization": [0.9, 0.5, 0.25]})
    bands = compute_health(df, cfg).health["health_band"].tolist()
    assert bands == ["green", "yellow", "red"]


def test_alert_conditions():
    alerts = (
        Alert("a", "high", "both", (("x", ">=", 3), ("seg", "==", "SMB"))),
        Alert("b", "low", "in", (("seg", "in", ["SMB", "Mid-Market"]),)),
        Alert("c", "low", "missing column", (("nope", "==", 1),)),
    )
    df = pd.DataFrame({"x": [5, 1, np.nan], "seg": ["SMB", "Mid-Market", "Enterprise"]})
    fired = evaluate_alerts(df, alerts)
    assert [a["id"] for a in fired.iloc[0]] == ["a", "b"]
    assert [a["id"] for a in fired.iloc[1]] == ["b"]
    assert fired.iloc[2] == []


def test_urgency_decreases_with_time_and_caps_overdue():
    cfg = load_config("saas")
    u = urgency(pd.Series([-30.0, 0.0, 90.0, 180.0, 365.0, 900.0]), cfg).tolist()
    assert u[0] == u[1] == 1.0
    assert u == sorted(u, reverse=True)
