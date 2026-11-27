"""Smoke test for the Streamlit demo. Skipped when Streamlit isn't installed (it's in the `app` extra)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_demo_app_renders_without_errors() -> None:
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) >= 4
    assert at.metric, "expected KPI metrics on the first screen"
    assert any(s.value == "Retention by plan size" for s in at.subheader), "expected the plan-size breakdown"
    assert any(s.value == "Plays to run" for s in at.subheader), "expected the plays panel on the portfolio tab"
    assert any(s.value.startswith("Play: ") for s in at.subheader), "expected the account's play"
    assert any(m.value == "**Talk track**" for m in at.markdown), "expected the play's talk track"


def test_demo_app_pulls_in_simulated_zendesk() -> None:
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert any("Zendesk: matched" in s.value for s in at.success), "expected the add-on match report"
    at.checkbox(key="zendesk_on").uncheck().run()
    assert not at.exception, at.exception
    assert not any("Zendesk: matched" in s.value for s in at.success)


def test_account_detail_shows_support_signals() -> None:
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert any(s.value == "Support" for s in at.subheader), "expected a Support panel on the account tab"
    assert any("Data add-ons: Zendesk (simulated)" in c.value for c in at.caption)


def test_real_zendesk_waits_for_settings() -> None:
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.toggle(key="zendesk_sim").set_value(False).run()
    assert not at.exception, at.exception
    assert any("to use Zendesk" in c.value for c in at.caption), "expected a prompt for missing settings"
    assert not at.sidebar.error, "an incomplete form should not show errors"
