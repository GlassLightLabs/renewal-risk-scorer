"""The transparent layer: a weighted multi-signal health score, risk alerts and renewal urgency.

This is the part a CS leader can audit line by line. Every point an account loses traces to one measure,
and the "detractors" output says exactly where the points went.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Alert, ScorecardConfig


@dataclass
class ScorecardResult:
    health: pd.DataFrame  # health_score, health_band, coverage, confidence, <group>_score columns
    measure_scores: pd.DataFrame  # one column per measure, 0-100 or NaN
    points_lost: pd.DataFrame  # health points lost per measure, after weighting
    alerts: pd.Series  # list of {"id", "severity", "message"} per account


def score_measure(
    values: pd.Series,
    measure_points: tuple[tuple[float, float], ...] = (),
    if_true: float | None = None,
    if_false: float | None = None,
) -> pd.Series:
    """Map raw values to 0-100. NaN stays NaN so the scorecard can reweight around it."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    if if_true is not None and if_false is not None:
        return v.map(lambda x: np.nan if np.isnan(x) else (if_true if x >= 0.5 else if_false))
    xs = np.array([p[0] for p in measure_points])
    ys = np.array([p[1] for p in measure_points])
    out = np.interp(v.to_numpy(), xs, ys)
    return pd.Series(np.where(np.isnan(v.to_numpy()), np.nan, out), index=values.index)


def compute_health(df: pd.DataFrame, config: ScorecardConfig) -> ScorecardResult:
    total_group_w = sum(g.weight for g in config.groups)
    measure_scores = pd.DataFrame(index=df.index)
    global_w: dict[str, float] = {}
    for g in config.groups:
        mw_total = sum(m.weight for m in g.measures)
        for m in g.measures:
            col = df[m.name] if m.name in df.columns else pd.Series(np.nan, index=df.index)
            measure_scores[m.name] = score_measure(col, m.points, m.if_true, m.if_false)
            global_w[m.name] = (g.weight / total_group_w) * (m.weight / mw_total)

    health = pd.DataFrame(index=df.index)
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_avail = pd.Series(0.0, index=df.index)
    within_group = pd.DataFrame(0.0, index=df.index, columns=measure_scores.columns)
    group_present: dict[str, pd.Series] = {}
    for g in config.groups:
        names = [m.name for m in g.measures]
        w = pd.Series({m.name: m.weight for m in g.measures})
        avail = measure_scores[names].notna()
        g_weight = (avail * w).sum(axis=1)
        g_score = (measure_scores[names].fillna(0) * w).sum(axis=1) / g_weight.replace(0, np.nan)
        health[f"{g.name}_score"] = g_score.round(1)
        has = g_score.notna()
        group_present[g.name] = has.astype(float) * g.weight
        within_group[names] = (avail * w).div(g_weight.replace(0, np.nan), axis=0).fillna(0.0)
        weighted_sum += np.where(has, g_score.fillna(0) * g.weight, 0.0)
        weight_avail += np.where(has, g.weight, 0.0)

    health["health_score"] = (weighted_sum / weight_avail.replace(0, np.nan)).round(1)
    gw = pd.Series(global_w)
    health["coverage"] = (measure_scores.notna() * gw).sum(axis=1).round(3)
    health["confidence"] = np.select(
        [health["coverage"] >= config.confidence_high, health["coverage"] >= config.confidence_medium],
        ["high", "medium"],
        default="low",
    )
    health["health_band"] = np.select(
        [health["health_score"] >= config.green, health["health_score"] >= config.yellow],
        ["green", "yellow"],
        default="red",
    )
    health.loc[health["health_score"].isna(), "health_band"] = "unknown"

    # Points lost: how far each measure pulled the score below 100, using the weights that actually applied
    # to that account after reweighting. These sum to (100 - health_score).
    eff_w = pd.DataFrame(0.0, index=df.index, columns=measure_scores.columns)
    for g in config.groups:
        share = group_present[g.name] / weight_avail.replace(0, np.nan)
        for m in g.measures:
            eff_w[m.name] = within_group[m.name] * share.fillna(0.0)
    points_lost = ((100 - measure_scores).fillna(0.0) * eff_w).round(2)

    alerts = evaluate_alerts(df, config.alerts)
    return ScorecardResult(health=health, measure_scores=measure_scores, points_lost=points_lost, alerts=alerts)


def _condition(series: pd.Series, op: str, value: object) -> pd.Series:
    if op == "in":
        return series.isin(list(value))  # type: ignore[call-overload]
    if isinstance(value, str):
        s = series.astype("string")
        return (s == value).fillna(False) if op == "==" else (s != value).fillna(False)
    num = pd.to_numeric(series, errors="coerce")
    ops = {
        "<": num < value,
        "<=": num <= value,
        ">": num > value,
        ">=": num >= value,
        "==": num == value,
        "!=": num != value,
    }
    return ops[op].fillna(False)


def evaluate_alerts(df: pd.DataFrame, alerts: tuple[Alert, ...]) -> pd.Series:
    fired: list[list[dict[str, str]]] = [[] for _ in range(len(df))]
    for alert in alerts:
        mask = pd.Series(True, index=df.index)
        for fname, op, value in alert.when:
            if fname not in df.columns:
                mask &= False
                break
            mask &= _condition(df[fname], op, value)
        for i in np.flatnonzero(mask.to_numpy()):
            fired[i].append({"id": alert.id, "severity": alert.severity, "message": alert.message})
    return pd.Series(fired, index=df.index, dtype=object)


def urgency(days_to_renewal: pd.Series, config: ScorecardConfig) -> pd.Series:
    xs = np.array([p[0] for p in config.urgency_points])
    ys = np.array([p[1] for p in config.urgency_points])
    d = days_to_renewal.astype(float).clip(lower=0).to_numpy()
    return pd.Series(np.interp(d, xs, ys), index=days_to_renewal.index).round(3)


def scorecard_prior_probability(health_score: pd.Series) -> pd.Series:
    """Map a health score to a rough churn probability when no trained model is available.

    This is an uncalibrated prior (score 90 is about 1%, 70 about 7%, 50 about 32%, 30 about 75%). It keeps
    the ranking and ARR math usable on day one; train on your own renewal history to replace it.
    """
    h = health_score.fillna(60.0).astype(float)
    prob = 1.0 / (1.0 + np.exp((h.to_numpy() - 42.0) / 11.0))
    return pd.Series(prob, index=health_score.index).round(4)
