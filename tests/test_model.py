from __future__ import annotations

import numpy as np
import pytest

from renewal_risk.model import RiskModel, reason_codes


@pytest.mark.parametrize("profile", ["saas", "consumption", "hardware", "services", "ai_usage"])
def test_model_beats_chance_and_stack_matches_or_beats_scorecard(trained, profile):
    _, model = trained[profile]
    m = model.churn.metrics
    assert m["roc_auc"] > 0.75
    assert model.blend is not None
    assert model.blend.metrics["roc_auc"] >= m["scorecard_baseline"]["roc_auc"] - 0.01
    assert model.blend.metrics["brier"] <= m["scorecard_baseline"]["brier"] + 0.002
    assert "cross-validation" in m["evaluation"]


def test_calibration_is_reasonable(trained):
    hist, model = trained["saas"]
    p = model.churn.predict(model.matrix(hist))
    observed = (hist["outcome"] == "churned").mean()
    assert abs(p.mean() - observed) < 0.03


def test_monotone_constraints_hold(trained, books):
    _, model = trained["saas"]
    X = model.matrix(books["saas"]).copy()
    base = model.churn.predict(X)
    X2 = X.copy()
    X2["seat_utilization"] = X2["seat_utilization"] + 0.3  # more active seats must never raise risk
    assert np.all(model.churn.predict(X2) <= base + 1e-9)
    X3 = X.copy()
    X3["days_past_due"] = X3["days_past_due"].fillna(0) + 60  # a later invoice must never lower risk
    assert np.all(model.churn.predict(X3) >= base - 1e-9)


def test_save_load_roundtrip(trained, books, tmp_path):
    _, model = trained["ai_usage"]
    path = tmp_path / "m.json"
    model.save(path)
    loaded = RiskModel.load(path)
    a = model.predict(books["ai_usage"])
    b = loaded.predict(books["ai_usage"])
    assert np.allclose(a["churn_probability"], b["churn_probability"])
    assert loaded.blend is not None and loaded.profile == "ai_usage"
    assert "booster" in path.read_text()[:2000] or path.stat().st_size > 1000


def test_load_rejects_other_files(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"format": "something-else"}')
    with pytest.raises(ValueError):
        RiskModel.load(p)


def test_reason_codes_are_structured_and_ordered(trained, books):
    _, model = trained["saas"]
    book = books["saas"]
    contrib = model.churn.contributions(model.matrix(book))
    risks, protects = reason_codes(contrib, book, model.features)
    for items in risks:
        effects = [i["effect"] for i in items]
        assert effects == sorted(effects, reverse=True)
        assert all(set(i) == {"feature", "group", "label", "value", "effect"} for i in items)
    assert all(i["effect"] < 0 for items in protects for i in items)


def test_shap_values_add_up_to_the_raw_score(trained, books):
    _, model = trained["hardware"]
    X = model.matrix(books["hardware"])
    full = model.churn.booster.predict(
        X[model.churn.feature_names], pred_contrib=True, num_iteration=model.churn.booster.best_iteration
    )
    raw = model.churn.booster.predict(
        X[model.churn.feature_names], raw_score=True, num_iteration=model.churn.booster.best_iteration
    )
    assert np.allclose(full.sum(axis=1), raw, atol=1e-6)


def test_training_needs_enough_outcomes(trained):
    from renewal_risk.config import load_config
    from renewal_risk.model import train

    hist, _ = trained["saas"]
    tiny = hist.head(60).copy()
    tiny["outcome"] = "renewed"
    with pytest.raises(ValueError, match="churned"):
        train(tiny, load_config("saas"))
