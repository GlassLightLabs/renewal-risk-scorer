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
