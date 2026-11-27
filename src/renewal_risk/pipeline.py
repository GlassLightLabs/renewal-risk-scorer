"""End-to-end scoring: scorecard + model (or scorecard prior) -> ranked, explained, dollarized renewal risk.

The output answers the three questions a renewals leader asks every week:

1. **Who is at risk, and how much money is it?** Churn and contraction probabilities become expected ARR
   loss, so a $2M account at 20% risk outranks a $20k account at 60%.
2. **Who do we act on first?** Expected loss is weighted by how close the renewal is (urgency), so the
   priority list reflects what can still be saved this quarter.
3. **Why, and what do we do?** Every account carries its top risk drivers (exact SHAP values from the
   model), the scorecard measures that cost it the most points, any rule-based alerts, and a save play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import FeatureSpec, ScorecardConfig
from .ingest import DataQualityReport
from .model import RiskModel, reason_codes
from .playbook import Playbook, default_playbook, plays_summary, recommend
from .scorecard import ScorecardResult, compute_health, scorecard_prior_probability, urgency

# Used only when there is no trained model: industry-typical contraction odds and depth for a renewal.
DEFAULT_CONTRACTION_RATE = 0.15
DEFAULT_CONTRACTION_DEPTH = 0.25

# Output column -> key in the play returned by :func:`renewal_risk.playbook.recommend`.
PLAY_COLUMNS = {
    "save_play": "play",
    "play_id": "id",
    "play_owner": "owner",
    "play_timing": "timing",
    "play_actions": "actions",
    "play_motion": "motion",
    "play_cadence": "cadence",
    "play_escalation": "escalation",
    "play_talk_track": "talk_track",
    "play_exit_criteria": "exit_criteria",
    "supporting_plays": "supporting",
}


@dataclass
class ScoreResult:
    accounts: pd.DataFrame  # one row per account, ranked by priority
    summary: dict[str, Any]
    drivers: pd.DataFrame  # portfolio-level risk drivers by group
    forecast: dict[str, Any]
    config: ScorecardConfig
    scorecard: ScorecardResult
    quality: DataQualityReport | None = None
    model_metrics: dict[str, Any] = field(default_factory=dict)

    def top(self, n: int = 20) -> pd.DataFrame:
        return self.accounts.head(n)


def _label(name: str, specs: dict[str, FeatureSpec]) -> str:
    return specs[name].label if name in specs else name.replace("_", " ").capitalize()


def scorecard_gaps(
    points_lost: pd.DataFrame, config: ScorecardConfig, *, top_n: int = 3, min_points: float = 2.0
) -> pd.Series:
    """The measures that cost each account the most health points, e.g. Seat utilization (-11.2 pts)."""
    specs = {f.name: f for f in config.features}
    groups = config.measure_groups
    out: list[list[dict[str, Any]]] = []
    for _, row in points_lost.iterrows():
        top = row[row >= min_points].sort_values(ascending=False).head(top_n)
        out.append(
            [
                {
                    "feature": k,
                    "group": groups.get(str(k), ""),
                    "label": _label(str(k), specs),
                    "points": round(float(v), 1),
                }
                for k, v in top.items()
            ]
        )
    return pd.Series(out, index=points_lost.index, dtype=object)


def _scorecard_reasons(df: pd.DataFrame, gaps: pd.Series, config: ScorecardConfig) -> pd.Series:
    """Reason codes in the model's format, built from scorecard gaps, for when no model is trained."""
    from .model import describe_value

    specs = {f.name: f for f in config.features}
    out = []
    for i, items in enumerate(gaps):
        rec = df.iloc[i]
        reasons = []
        for it in items:
            spec = specs.get(it["feature"])
            value = describe_value(spec, rec) if spec else str(rec.get(it["feature"], ""))
            reasons.append(
                {
                    **{k: it[k] for k in ("feature", "group", "label")},
                    "value": value,
                    "effect": round(it["points"] / 25.0, 3),
                }
            )
        out.append(reasons)
    return pd.Series(out, index=df.index, dtype=object)


def assign_tier(p: pd.Series, days: pd.Series, config: ScorecardConfig, alerts: pd.Series | None = None) -> pd.Series:
    """First tier whose probability and time-window thresholds both hold. A high-severity alert (champion
    left, competitor evaluation, 60+ days past due...) never sits in "low": it is at least on the watch list."""
    tier = pd.Series("low", index=p.index, dtype=object)
    assigned = pd.Series(False, index=p.index)
    for t in config.tiers:
        hit = (p >= t.min_churn_probability) & ~assigned
        if t.max_days_to_renewal is not None:
            hit &= days <= t.max_days_to_renewal
        tier[hit] = t.name
        assigned |= hit
    if alerts is not None and config.tiers:
        flagged = alerts.map(lambda items: any(a["severity"] == "high" for a in items)).astype(bool)
        tier[flagged & (tier == "low")] = config.tiers[-1].name
    return tier


# Monthly spend bands. ChartMogul's 2025 AI retention study found the split between "tourists" and
# committed buyers sits on monthly spend, so the book is broken out the same way.
PLAN_SIZE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Under $50/mo", 0.0, 50.0),
    ("$50-249/mo", 50.0, 250.0),
    ("$250-999/mo", 250.0, 1_000.0),
    ("$1k-9.9k/mo", 1_000.0, 10_000.0),
    ("$10k+/mo", 10_000.0, float("inf")),
)
# Median gross revenue retention for AI-native products by monthly spend (ChartMogul, December 2025).
# Only published for spend under $250/mo as separate bands; $250+ is one band, so the last three share it.
AI_NATIVE_GRR_BENCHMARK: dict[str, float] = {
    "Under $50/mo": 0.23,
    "$50-249/mo": 0.45,
    "$250-999/mo": 0.70,
    "$1k-9.9k/mo": 0.70,
    "$10k+/mo": 0.70,
}


def by_plan_size(accounts: pd.DataFrame, profile: str) -> list[dict[str, Any]]:
    """Accounts, ARR, expected loss and expected GRR for each monthly-spend band that has accounts."""
    monthly = accounts["arr"].astype(float) / 12.0
    rows: list[dict[str, Any]] = []
    for label, lo, hi in PLAN_SIZE_BANDS:
        part = accounts[(monthly >= lo) & (monthly < hi)]
        if part.empty:
            continue
        arr = float(part["arr"].sum())
        loss = float(part["expected_arr_loss"].sum())
        row: dict[str, Any] = {
            "band": label,
            "accounts": int(len(part)),
            "arr": round(arr, 2),
            "expected_arr_loss": round(loss, 2),
            "expected_grr": round(1 - loss / arr, 4) if arr > 0 else None,
        }
        if profile == "ai_usage":
            row["benchmark_grr"] = AI_NATIVE_GRR_BENCHMARK[label]
        rows.append(row)
    return rows


def watermelon_bar(config: ScorecardConfig, model: RiskModel | None) -> float:
    """Churn probability above which a green account counts as a watermelon: the profile's floor, or a multiple
    of how often green accounts churned in the training history, whichever is higher."""
    bar = config.watermelon_min_churn_probability
    green_rate = model.meta.get("green_band_churn_rate") if model is not None else None
    if green_rate:
        bar = max(bar, config.watermelon_lift * float(green_rate))
    return float(bar)


def forecast_grr(
    arr: np.ndarray,
    p_churn: np.ndarray,
    p_contract: np.ndarray,
    depth: float,
    *,
    simulations: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Monte Carlo of gross revenue retention for a set of renewals.

    Each simulated quarter draws churn, then contraction, per account. Outcomes are treated as independent,
    which understates the tail when risks are correlated (a recession, a competitor launch), so read P10 as
    "a bad quarter", not "the worst case".
    """
    total = float(arr.sum())
    if total <= 0 or len(arr) == 0:
        return {"accounts": 0, "arr_up_for_renewal": 0.0}
    rng = np.random.default_rng(seed)
    losses = np.empty(simulations)
    chunk = max(1, min(simulations, 2_000_000 // max(1, len(arr))))
    done = 0
    while done < simulations:
        k = min(chunk, simulations - done)
        u1 = rng.random((k, len(arr)))
        u2 = rng.random((k, len(arr)))
        churned = u1 < p_churn
        contracted = (~churned) & (u2 < p_contract)
        losses[done : done + k] = (churned * arr).sum(axis=1) + (contracted * arr * depth).sum(axis=1)
        done += k
    grr = 1 - losses / total
    expected_loss = float((arr * (p_churn + (1 - p_churn) * p_contract * depth)).sum())
    return {
        "accounts": int(len(arr)),
        "arr_up_for_renewal": round(total, 2),
        "expected_arr_loss": round(expected_loss, 2),
        "expected_grr": round(1 - expected_loss / total, 4),
        "grr_p10": round(float(np.percentile(grr, 10)), 4),
        "grr_p50": round(float(np.percentile(grr, 50)), 4),
        "grr_p90": round(float(np.percentile(grr, 90)), 4),
        "arr_loss_p90": round(float(np.percentile(losses, 90)), 2),
        "expected_churned_logos": round(float(p_churn.sum()), 1),
        "simulations": simulations,
    }


def portfolio_drivers(reasons: pd.Series, arr: pd.Series, config: ScorecardConfig) -> pd.DataFrame:
    """Which signal groups drive risk across the book, weighted by ARR, from each account's reason codes."""
    labels = config.group_labels | {"money": "Account size", "segment": "Segment"}
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for items, a in zip(reasons, arr, strict=True):
        for it in items:
            key = (it["group"], it["label"])
            r = rows.setdefault(key, {"accounts": 0, "arr_weighted_effect": 0.0})
            r["accounts"] += 1
            r["arr_weighted_effect"] += float(it["effect"]) * float(a)
    if not rows:
        return pd.DataFrame(columns=["group", "group_label", "driver", "accounts", "share_of_risk"])
    df = pd.DataFrame(
        [{"group": g, "group_label": labels.get(g, g), "driver": lbl, **v} for (g, lbl), v in rows.items()]
    )
    df["share_of_risk"] = (df["arr_weighted_effect"] / df["arr_weighted_effect"].sum()).round(4)
    return df.sort_values("share_of_risk", ascending=False).drop(columns="arr_weighted_effect").reset_index(drop=True)


def score(
    df: pd.DataFrame,
    config: ScorecardConfig,
    model: RiskModel | None = None,
    *,
    as_of: date | None = None,
    horizon_days: int | None = None,
    quality: DataQualityReport | None = None,
    simulations: int = 5000,
    seed: int = 0,
    playbook: Playbook | None = None,
) -> ScoreResult:
    """Score a canonical book of business (from :func:`renewal_risk.ingest.ingest`).

    ``playbook`` defaults to the built-in one; pass :func:`renewal_risk.playbook.load_playbook` output to use
    a team's own plays.
    """
    if model is not None and model.profile != config.profile:
        raise ValueError(
            f"Model was trained for the '{model.profile}' profile but the book is scored as "
            f"'{config.profile}'. Train a model for this profile or pass --profile {model.profile}."
        )
    as_of = as_of or date.today()
    book = df.copy()
    sc = compute_health(book, config)
    prior = scorecard_prior_probability(sc.health["health_score"])
    gaps = scorecard_gaps(sc.points_lost, config)

    out = pd.DataFrame(index=book.index)
    keep = ["account_id", "account_name", "segment", "csm_owner", "arr", "renewal_date", "days_to_renewal"]
    for c in keep:
        if c in book.columns:
            out[c] = book[c]
    out = out.join(sc.health)
    out["scorecard_churn_probability"] = prior

    if model is not None:
        preds = model.predict(book, prior)
        out["churn_probability"] = preds["churn_probability"]
        out["model_churn_probability"] = preds["model_churn_probability"]
        out["contraction_probability"] = preds["contraction_probability"]
        depth = model.contraction_depth
        contrib = model.churn.contributions(model.matrix(book))
        drivers, protective = reason_codes(contrib, book, model.features)
        out["risk_source"] = "model + scorecard (stacked)" if model.blend is not None else "model"
        model_metrics = {
            "churn": model.churn.metrics,
            "blend": model.blend.metrics if model.blend else None,
            "contraction": model.contraction.metrics if model.contraction else None,
            "meta": model.meta,
        }
    else:
        out["churn_probability"] = prior
        out["contraction_probability"] = DEFAULT_CONTRACTION_RATE
        depth = DEFAULT_CONTRACTION_DEPTH
        drivers = _scorecard_reasons(book, gaps, config)
        protective = pd.Series([[] for _ in range(len(book))], index=book.index, dtype=object)
        out["risk_source"] = "scorecard prior (no model trained)"
        model_metrics = {}

    p = out["churn_probability"].astype(float)
    pc = out["contraction_probability"].astype(float)
    arr = book["arr"].astype(float)
    out["expected_arr_loss"] = (arr * (p + (1 - p) * pc * depth)).round(0)
    out["renewal_urgency"] = urgency(book["days_to_renewal"], config)
    out["priority_arr"] = (out["expected_arr_loss"] * out["renewal_urgency"]).round(0)
    out["risk_tier"] = assign_tier(p, book["days_to_renewal"].astype(float), config, sc.alerts)
    # Watermelon: green on the scorecard, red underneath. Judged on the model's own view (before stacking with
    # the scorecard), because the point is to catch accounts where the two disagree.
    # The bar is the higher of the profile's floor and a multiple of how often green accounts churned in the
    # training history, so "watermelon" means several times the risk the green band normally carries.
    p_view = out["model_churn_probability"].astype(float) if "model_churn_probability" in out.columns else p
    bar = watermelon_bar(config, model)
    out["watermelon"] = (out["health_band"] == "green") & (p_view >= bar)
    out["risk_drivers"] = drivers
    out["protective_factors"] = protective
    out["scorecard_gaps"] = gaps
    out["alerts"] = sc.alerts
    out["alert_count"] = sc.alerts.map(len)
    out["top_reason"] = [_headline(d, a) for d, a in zip(out["risk_drivers"], out["alerts"], strict=True)]

    pb = playbook or default_playbook()
    plays = [
        recommend(config.profile, d, a, days, w, tier=t, playbook=pb)
        for d, a, days, w, t in zip(
            out["risk_drivers"],
            out["alerts"],
            book["days_to_renewal"],
            out["watermelon"],
            out["risk_tier"],
            strict=True,
        )
    ]
    for col, key in PLAY_COLUMNS.items():
        out[col] = [pl[key] for pl in plays]

    out = out.sort_values(["priority_arr", "expected_arr_loss"], ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    window = out if horizon_days is None else out[out["days_to_renewal"] <= horizon_days]
    fc = forecast_grr(
        window["arr"].to_numpy(float),
        window["churn_probability"].to_numpy(float),
        window["contraction_probability"].to_numpy(float),
        depth,
        simulations=simulations,
        seed=seed,
    )
    fc["horizon_days"] = horizon_days

    tiers = out["risk_tier"].value_counts().to_dict()
    summary = {
        "as_of": as_of.isoformat(),
        "profile": config.profile,
        "profile_label": config.label,
        "risk_source": str(out["risk_source"].iloc[0]) if len(out) else "",
        "accounts": int(len(out)),
        "total_arr": round(float(arr.sum()), 2),
        "expected_arr_loss": round(float(out["expected_arr_loss"].sum()), 2),
        "accounts_by_tier": {t: int(tiers.get(t, 0)) for t in [*(x.name for x in config.tiers), "low"]},
        "arr_by_tier": {
            t: round(float(out.loc[out["risk_tier"] == t, "arr"].sum()), 2)
            for t in [*(x.name for x in config.tiers), "low"]
        },
        "watermelons": int(out["watermelon"].sum()),
        "overdue_renewals": int((out["days_to_renewal"] < 0).sum()),
        "low_confidence_scores": int((out["confidence"] == "low").sum()),
        "health_bands": {k: int(v) for k, v in out["health_band"].value_counts().items()},
        "forecast": fc,
        "by_plan_size": by_plan_size(out, config.profile),
        "plays": plays_summary(out, tier_order=[t.name for t in config.tiers]),
        "play_tiers": {t.name: pb.tiers[t.name] for t in config.tiers if t.name in pb.tiers},
    }
    return ScoreResult(
        accounts=out,
        summary=summary,
        drivers=portfolio_drivers(out["risk_drivers"], out["arr"], config),
        forecast=fc,
        config=config,
        scorecard=sc,
        quality=quality,
        model_metrics=model_metrics,
    )


def _headline(drivers: list[dict[str, Any]], alerts: list[dict[str, str]]) -> str:
    high = [a for a in alerts if a["severity"] == "high"]
    if high:
        return high[0]["message"]
    if drivers:
        d = drivers[0]
        return f"{d['label']}: {d['value']}"
    return "No material risk drivers"
