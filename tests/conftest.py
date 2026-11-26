from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from renewal_risk.config import load_config
from renewal_risk.ingest import ingest
from renewal_risk.model import RiskModel, train
from renewal_risk.synth import PROFILES, generate_book, generate_history

AS_OF = date(2026, 12, 1)


@pytest.fixture(scope="session")
def as_of() -> date:
    return AS_OF


@pytest.fixture(scope="session")
def trained() -> dict[str, tuple[pd.DataFrame, RiskModel]]:
    """One small trained model per profile, shared across tests (training is the slow part)."""
    out = {}
    for p in PROFILES:
        hist, _ = ingest(generate_history(2000, profile=p, as_of=AS_OF, seed=3), as_of=AS_OF, require_outcome=True)
        out[p] = (hist, train(hist, load_config(p), seed=1))
    return out


@pytest.fixture(scope="session")
def books() -> dict[str, pd.DataFrame]:
    out = {}
    for p in PROFILES:
        book, _ = ingest(generate_book(300, profile=p, as_of=AS_OF, seed=5), as_of=AS_OF)
        out[p] = book
    return out
