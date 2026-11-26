# Temporary diagnostic, removed before merge.
import lightgbm as lgb
import numpy as np


def test_debug_monotone(trained, books):
    _, model = trained["saas"]
    X = model.matrix(books["saas"]).copy()
    base = model.churn.predict(X)
    X2 = X.copy()
    X2["seat_utilization"] = X2["seat_utilization"] + 0.3
    p2 = model.churn.predict(X2)
    bad = np.where(p2 > base + 1e-9)[0]
    b = model.churn.booster
    raw1 = b.predict(X[model.churn.feature_names])
    raw2 = b.predict(X2[model.churn.feature_names])
    rawbad = np.where(raw2 > raw1 + 1e-12)[0]
    msg = [
        f"lgb {lgb.__version__}",
        f"features {model.churn.feature_names}",
        f"calibrator {model.churn.calibrator.method}",
        f"bad rows {len(bad)} max diff {float((p2 - base).max()):.3e}",
        f"raw bad rows {len(rawbad)} max raw diff {float((raw2 - raw1).max()):.3e}",
        f"examples {X.iloc[bad[:3]].to_dict('records')}",
    ]
    params = b.dump_model()
    msg.append(f"monotone {params.get('monotone_constraints')}")
    raise AssertionError("\n".join(msg))
