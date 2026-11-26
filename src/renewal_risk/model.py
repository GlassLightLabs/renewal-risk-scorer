"""The predictive layer: calibrated gradient-boosted churn and contraction models with exact reason codes.

Design choices, and why:

* **LightGBM with monotone constraints.** Each signal's direction is fixed to what CS practice says it
  should be (more active seats can never raise risk, a late invoice can never lower it). The model can
  still learn thresholds and interactions, but it cannot learn a backwards rule from a quirk in the data,
  so every reason code it gives is one a CS leader would accept.
* **Native missing-value handling.** A missing NPS survey is information, not zero. The trees learn
  where to send missing values instead of us imputing them.
* **Probability calibration.** Raw boosting scores are not probabilities. Platt scaling (or isotonic
  regression once there are 500+ positives and negatives to fit it) on held-out rows makes "30% churn
  risk" mean 30%, which matters because expected ARR loss and the GRR forecast multiply by it.
* **Exact TreeSHAP reason codes.** LightGBM computes exact Shapley values for tree ensembles
  (``pred_contrib=True``), so each account's drivers are additive and faithful to the model, with no
  sampling approximation and no extra dependency.
* **Safe persistence.** Models are saved as LightGBM's text format inside a JSON file, never pickled, so
  loading a model file cannot execute code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import __version__
from .config import FeatureSpec, ScorecardConfig
from .metrics import classification_report
from .scorecard import compute_health, scorecard_prior_probability

SEGMENT_CODES = {"SMB": 0, "Mid-Market": 1, "Enterprise": 2}

PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 30,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "monotone_constraints_method": "advanced",
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
}


def build_features(df: pd.DataFrame, features: tuple[FeatureSpec, ...]) -> pd.DataFrame:
    """Canonical frame -> model matrix. Absent signals become NaN columns (LightGBM handles them)."""
    X = pd.DataFrame(index=df.index)
    for spec in features:
        name = spec.name
        if name == "log_arr":
            X[name] = np.log10(df["arr"].clip(lower=1)) if "arr" in df.columns else np.nan
        elif name == "segment_code":
            X[name] = df["segment"].map(SEGMENT_CODES) if "segment" in df.columns else np.nan
        else:
            X[name] = df[name] if name in df.columns else np.nan
    return X.astype(float)


@dataclass
class Calibrator:
    method: str  # "isotonic", "platt" or "identity"
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, raw: np.ndarray, y: np.ndarray, min_positives: int = 500) -> Calibrator:
        if y.sum() >= min_positives and (len(y) - y.sum()) >= min_positives:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.002, y_max=0.998).fit(raw, y)
            return cls("isotonic", x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist())
        if y.sum() >= 5:
            logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            lr = LogisticRegression(C=1.0).fit(logit, y)
            return cls("platt", a=float(lr.coef_[0][0]), b=float(lr.intercept_[0]))
        return cls("identity")

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return np.interp(raw, np.array(self.x), np.array(self.y))
        if self.method == "platt":
            r = np.clip(raw, 1e-6, 1 - 1e-6)
            return 1 / (1 + np.exp(-(self.a * np.log(r / (1 - r)) + self.b)))
        return raw


@dataclass
class BinaryModel:
    booster: lgb.Booster
    calibrator: Calibrator
    metrics: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        return list(self.booster.feature_name())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(X[self.feature_names], num_iteration=self.booster.best_iteration or None)
        return np.clip(self.calibrator(np.asarray(raw)), 0.0, 1.0)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Exact SHAP values in log-odds space; the last column (bias) is dropped."""
        contrib = self.booster.predict(
            X[self.feature_names], pred_contrib=True, num_iteration=self.booster.best_iteration or None
        )
        return pd.DataFrame(np.asarray(contrib)[:, :-1], columns=self.feature_names, index=X.index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "booster": self.booster.model_to_string(),
            "best_iteration": self.booster.best_iteration,
            "calibrator": self.calibrator.__dict__,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BinaryModel:
        booster = lgb.Booster(model_str=d["booster"])
        booster.best_iteration = int(d.get("best_iteration") or 0)
        return cls(booster, Calibrator(**d["calibrator"]), d.get("metrics", {}))


@dataclass
class RiskModel:
    profile: str
    features: tuple[FeatureSpec, ...]
    churn: BinaryModel
    contraction: BinaryModel | None
    contraction_rate: float
    contraction_depth: float
    meta: dict[str, Any]
    blend: Blend | None = None

    def matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df, self.features)

    def predict(self, df: pd.DataFrame, scorecard_probability: pd.Series | None = None) -> pd.DataFrame:
        """Churn and contraction probabilities. Pass the scorecard's probability to get the stacked churn risk."""
        X = self.matrix(df)
        out = pd.DataFrame(index=df.index)
        p_model = self.churn.predict(X)
        out["model_churn_probability"] = p_model.round(4)
        if self.blend is not None and scorecard_probability is not None:
            p = self.blend(p_model, scorecard_probability.reindex(df.index).to_numpy(dtype=float))
        else:
            p = p_model
        out["churn_probability"] = np.round(p, 4)
        if self.contraction is not None:
            out["contraction_probability"] = self.contraction.predict(X).round(4)
        else:
            out["contraction_probability"] = self.contraction_rate
        return out

    def save(self, path: str | Path) -> None:
        payload = {
            "format": "renewal-risk-model/1",
            "profile": self.profile,
            "features": [f.__dict__ for f in self.features],
            "churn": self.churn.to_dict(),
            "contraction": self.contraction.to_dict() if self.contraction else None,
            "contraction_rate": self.contraction_rate,
            "contraction_depth": self.contraction_depth,
            "blend": self.blend.__dict__ if self.blend else None,
            "meta": self.meta,
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> RiskModel:
        d = json.loads(Path(path).read_text())
        if d.get("format") != "renewal-risk-model/1":
            raise ValueError(f"{path} is not a renewal-risk model file")
        return cls(
            profile=str(d.get("profile", "saas")),
            features=tuple(FeatureSpec(**f) for f in d["features"]),
            churn=BinaryModel.from_dict(d["churn"]),
            contraction=BinaryModel.from_dict(d["contraction"]) if d.get("contraction") else None,
            contraction_rate=float(d["contraction_rate"]),
            contraction_depth=float(d["contraction_depth"]),
            meta=d.get("meta", {}),
            blend=Blend(**d["blend"]) if d.get("blend") else None,
        )


def _fit_once(X: pd.DataFrame, y: np.ndarray, seed: int, directions: list[int]) -> BinaryModel:
    """Fit on 75%, early-stop and calibrate on the other 25%."""
    idx = np.arange(len(y))
    tr, ca = train_test_split(idx, test_size=0.25, stratify=y, random_state=seed)
    params = dict(PARAMS, seed=seed, monotone_constraints=directions)
    dtrain = lgb.Dataset(X.iloc[tr], y[tr], free_raw_data=False)
    dcal = lgb.Dataset(X.iloc[ca], y[ca], reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=2000, valid_sets=[dcal], callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    raw = np.asarray(booster.predict(X.iloc[ca], num_iteration=booster.best_iteration))
    return BinaryModel(booster, Calibrator.fit(raw, y[ca]), {})


def _fit_binary(
    X: pd.DataFrame,
    y: np.ndarray,
    arr: np.ndarray,
    seed: int,
    directions: list[int],
    baseline: np.ndarray | None = None,
    folds: int = 5,
) -> tuple[BinaryModel, np.ndarray]:
    """Evaluate by stratified k-fold (every row is scored by a model that never saw it), then fit the final model.

    Out-of-fold metrics use all the history instead of a small test slice, so the comparison with the
    scorecard is far less noisy than a single split. The scorecard baseline is scored on the same rows.
    """
    oof = np.full(len(y), np.nan)
    iterations = []
    for k, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)):
        fold_model = _fit_once(X.iloc[tr].reset_index(drop=True), y[tr], seed + k, directions)
        oof[te] = fold_model.predict(X.iloc[te])
        iterations.append(fold_model.booster.best_iteration)
    model = _fit_once(X, y, seed, directions)
    metrics = classification_report(y, oof, arr)
    metrics.update(
        {
            "evaluation": f"{folds}-fold stratified cross-validation (out-of-fold)",
            "n_rows": len(y),
            "fold_iterations": iterations,
            "best_iteration": model.booster.best_iteration,
            "calibration_method": model.calibrator.method,
        }
    )
    if baseline is not None:
        metrics["scorecard_baseline"] = classification_report(y, baseline, arr)
    model.metrics = metrics
    return model, oof


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.asarray(np.log(q / (1 - q)))


@dataclass
class Blend:
    """Stacked churn probability: a logistic regression over the model's and the scorecard's log-odds.

    The scorecard carries expert judgement that holds up when history is thin; the model carries what the
    history actually shows. Stacking them on out-of-fold predictions lets the data set the weights, and it
    is still explainable: risk log-odds = intercept + w_model x model log-odds + w_scorecard x scorecard log-odds.
    """

    w_model: float
    w_scorecard: float
    intercept: float
    metrics: dict[str, Any]

    def __call__(self, p_model: np.ndarray, p_scorecard: np.ndarray) -> np.ndarray:
        z = self.intercept + self.w_model * logit(p_model) + self.w_scorecard * logit(p_scorecard)
        return np.asarray(1 / (1 + np.exp(-z)))


def _fit_blend(oof: np.ndarray, prior: np.ndarray, y: np.ndarray, arr: np.ndarray, seed: int) -> Blend:
    Z = np.c_[logit(oof), logit(prior)]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    stacked = cross_val_predict(LogisticRegression(C=1.0), Z, y, cv=cv, method="predict_proba")[:, 1]
    lr = LogisticRegression(C=1.0).fit(Z, y)
    metrics = classification_report(y, stacked, arr)
    metrics["evaluation"] = "5-fold cross-validation of the stack over out-of-fold model predictions"
    return Blend(float(lr.coef_[0][0]), float(lr.coef_[0][1]), float(lr.intercept_[0]), metrics)


def train(history: pd.DataFrame, config: ScorecardConfig, *, seed: int = 42, stack_scorecard: bool = True) -> RiskModel:
    """Train churn and contraction models on a canonical history frame (from ``ingest(..., require_outcome=True)``).

    The profile's scorecard is computed on the same history. Its probability is the baseline every metric is
    compared against, and (with ``stack_scorecard``) it is stacked with the model into the final churn risk.
    """
    if "outcome" not in history.columns:
        raise ValueError("History needs an outcome column (renewed / contracted / churned).")
    h = history.reset_index(drop=True)
    y_churn = (h["outcome"] == "churned").to_numpy().astype(int)
    if y_churn.sum() < 20 or (len(y_churn) - y_churn.sum()) < 20:
        raise ValueError(f"Need at least 20 churned and 20 retained renewals to train; got {y_churn.sum()} churned.")
    features = config.model_features
    X = build_features(h, features)
    present = [f for f in features if X[f.name].notna().mean() >= 0.05]
    if len(present) < 3:
        raise ValueError(f"History has too few {config.label} signals to train on; check the profile and columns.")
    features = tuple(present)
    X = X[[f.name for f in features]]
    directions = [f.direction for f in features]
    arr = h["arr"].to_numpy()
    sc = compute_health(h, config)
    baseline = scorecard_prior_probability(sc.health["health_score"]).to_numpy()
    churn, oof = _fit_binary(X, y_churn, arr, seed, directions, baseline)
    blend = _fit_blend(oof, baseline, y_churn, arr, seed) if stack_scorecard else None
    green = (sc.health["health_band"] == "green").to_numpy()

    retained = h["outcome"] != "churned"
    y_con = (h.loc[retained, "outcome"] == "contracted").to_numpy().astype(int)
    contraction_rate = float(y_con.mean()) if len(y_con) else 0.0
    contraction: BinaryModel | None = None
    if y_con.sum() >= 30 and (len(y_con) - y_con.sum()) >= 30:
        contraction, _ = _fit_binary(
            X.loc[retained].reset_index(drop=True), y_con, arr[retained.to_numpy()], seed + 1, directions
        )

    depth = 0.25
    if "renewed_arr" in h.columns:
        c = h[(h["outcome"] == "contracted") & h["renewed_arr"].notna()]
        if len(c) >= 10:
            depth = float((1 - c["renewed_arr"] / c["arr"]).clip(0.01, 0.99).median())

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "n_history": len(h),
        "churn_rate": round(float(y_churn.mean()), 4),
        "contraction_rate": round(contraction_rate, 4),
        "feature_coverage": {f.name: round(float(X[f.name].notna().mean()), 3) for f in features},
        "seed": seed,
        "profile_label": config.label,
        # Churn rate among accounts the scorecard called green: the yardstick for spotting watermelons.
        "green_band_churn_rate": round(float(y_churn[green].mean()), 4) if green.sum() >= 30 else None,
    }
    return RiskModel(config.profile, features, churn, contraction, contraction_rate, depth, meta, blend)


def reason_codes(
    contrib: pd.DataFrame,
    df: pd.DataFrame,
    features: tuple[FeatureSpec, ...],
    *,
    top_n: int = 3,
    min_effect: float = 0.05,
) -> tuple[pd.Series, pd.Series]:
    """Per account: the top risk drivers and the top protective factors, as structured dicts."""
    spec = {f.name: f for f in features}
    risks: list[list[dict[str, Any]]] = []
    protects: list[list[dict[str, Any]]] = []
    for i in range(len(contrib)):
        row = contrib.iloc[i]
        ordered = row.sort_values(ascending=False)
        up = [(k, v) for k, v in ordered.items() if v >= min_effect][:top_n]
        down = [(k, v) for k, v in ordered[::-1].items() if v <= -min_effect][:2]
        rec = df.iloc[i]
        risks.append([_reason(spec[str(k)], v, rec) for k, v in up])
        protects.append([_reason(spec[str(k)], v, rec) for k, v in down])
    return pd.Series(risks, index=contrib.index), pd.Series(protects, index=contrib.index)


def _reason(f: FeatureSpec, effect: float, rec: pd.Series) -> dict[str, Any]:
    return {
        "feature": f.name,
        "group": f.group,
        "label": f.label,
        "value": describe_value(f, rec),
        "effect": round(float(effect), 3),
    }


def describe_value(f: FeatureSpec, rec: pd.Series) -> str:
    """Human-readable value for a reason code, e.g. '31% (124 of 400 seats active)'."""

    def g(name: str) -> Any:
        v = rec.get(name, np.nan)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    if f.unit == "money":
        return f"${rec['arr']:,.0f} ARR"
    if f.unit == "segment":
        return str(g("segment") or "unknown")
    v = g(f.name)
    if v is None:
        return "no data"
    if f.name == "seat_utilization":
        a, s = g("active_users"), g("licensed_seats")
        return f"{v:.0%}" + (f" ({a:,.0f} of {s:,.0f} seats active)" if a is not None and s is not None else "")
    if f.name == "seat_change_pct":
        before, now = g("licensed_seats_prior"), g("licensed_seats")
        known = before is not None and now is not None
        return f"{v:+.0f}%" + (f" ({before:,.0f} to {now:,.0f} seats)" if known else "")
    formats = {
        "ratio": lambda x: f"{x:.0%}",
        "trend": lambda x: f"{x:+.0f}% vs 90 days ago",
        "pct": lambda x: f"{x:.0f}%",
        "pct1": lambda x: f"{x:.1f}%",
        "days": lambda x: f"{x:,.0f} days",
        "bool": lambda x: "yes" if x >= 0.5 else "no",
        "count": lambda x: f"{x:,.0f}",
        "score10": lambda x: f"{x:.0f}/10",
        "score5": lambda x: f"{x:.0f}/5",
        "months": lambda x: f"{x:.0f} months",
        "years": lambda x: f"{x:.1f} years",
    }
    return formats.get(f.unit, lambda x: f"{x:,.1f}")(v)
