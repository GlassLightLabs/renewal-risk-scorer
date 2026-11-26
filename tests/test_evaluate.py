"""Backtest on synthetic history: time-ordered split, all four checks reported."""

from __future__ import annotations

from datetime import date

import pytest

from renewal_risk.config import load_config
from renewal_risk.evaluate import backtest, to_markdown
from renewal_risk.ingest import ingest
from renewal_risk.synth import generate_history

AS_OF = date(2026, 12, 1)


@pytest.fixture(scope="module")
def result():
    hist, _ = ingest(generate_history(1600, profile="saas", as_of=AS_OF), as_of=AS_OF, require_outcome=True)
    return backtest(hist, load_config("saas"), source="synthetic data")


def test_split_is_time_ordered(result) -> None:
    assert result.n_train == 1200 and result.n_test == 400
    assert result.train_period[1] <= result.test_period[0]


def test_reports_every_method_and_check(result) -> None:
    assert set(result.methods) == {"scorecard", "model", "blend"}
    for m in result.methods.values():
        assert 0.5 < m["roc_auc"] <= 1.0
    assert result.methods["blend"]["brier"] < result.base_rate_brier
    assert len(result.calibration) == 5 and sum(r["n"] for r in result.calibration) == 400
    assert {b["band"] for b in result.bands} <= {"green", "yellow", "red"}
    assert 0 < result.grr["actual"] <= 1 and result.grr["p10"] <= result.grr["p90"]
    assert "flagged" in result.watermelons


def test_markdown_is_honest_about_synthetic_data(result) -> None:
    md = to_markdown([result])
    assert "| Seat-based SaaS |" in md and "not a benchmark" in md
    assert "outside the range" in md or "inside the range" in md


def test_needs_enough_later_renewals() -> None:
    hist, _ = ingest(generate_history(300, profile="saas", as_of=AS_OF), as_of=AS_OF, require_outcome=True)
    with pytest.raises(ValueError, match="Too few later renewals"):
        backtest(hist, load_config("saas"))
