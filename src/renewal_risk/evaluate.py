"""Backtest: would the scorer have called the churn before it happened?

Vendors quote accuracy figures nobody can check. This module produces a report anyone can rerun: train on the
earlier renewals in a history file, score the later renewals the model never saw, and compare with what really
happened. It checks the four claims the scorer makes:

1. **Ranking.** Do the riskiest accounts churn most? (ROC AUC, PR AUC, lift in the top 10%, share of churned
   ARR in the top 20%), against the scorecard alone, which is how most CS platforms score health.
2. **Calibration.** When it says 30%, do about 30% churn? (Brier score and a predicted-vs-actual table.)
3. **Watermelons.** Do green accounts it flags really churn more than other green accounts?
4. **The GRR forecast.** Does actual gross revenue retention land inside the P10 to P90 range?

A time-ordered split is stricter than the cross-validation used inside training, because the test renewals
come after everything the model learned from, as they would in real use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import ScorecardConfig
from .metrics import calibration_table, classification_report
from .model import train
from .pipeline import forecast_grr, watermelon_bar
from .scorecard import compute_health, scorecard_prior_probability

MIN_TO_JUDGE = 20  # fewer flagged watermelons than this is too few to compare churn rates

METHODS = {
    "scorecard": "Scorecard alone (rules, as most CS platforms score)",
    "model": "Model alone",
    "blend": "Model + scorecard (what the scorer uses)",
}


@dataclass
class Backtest:
    profile: str
    profile_label: str
    source: str
    n_train: int
    n_test: int
    train_period: tuple[str, str]
    test_period: tuple[str, str]
    test_churn_rate: float
    methods: dict[str, dict[str, Any]]
    base_rate_brier: float
    calibration: list[dict[str, float]]
    bands: list[dict[str, Any]]
    watermelons: dict[str, Any]
    grr: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def backtest(
    history: pd.DataFrame,
    config: ScorecardConfig,
    *,
    test_share: float = 0.25,
    seed: int = 42,
    source: str = "",
) -> Backtest:
    """Train on the earliest renewals, test on the latest ``test_share`` of them."""
    if "outcome" not in history.columns or "renewal_date" not in history.columns:
        raise ValueError("A backtest needs a history with outcome and renewal date columns.")
    h = history[history["renewal_date"].notna()].sort_values("renewal_date", kind="stable").reset_index(drop=True)
    cut = int(round(len(h) * (1 - test_share)))
    tr, te = h.iloc[:cut].reset_index(drop=True), h.iloc[cut:].reset_index(drop=True)
    y = (te["outcome"] == "churned").to_numpy().astype(int)
    if len(te) < 100 or y.sum() < 10:
        raise ValueError(
            f"Too few later renewals to test on ({len(te)} renewals, {y.sum()} churned). "
            "A backtest needs at least 100, with 10 churned."
        )
    model = train(tr, config, seed=seed)

    sc = compute_health(te, config)
    prior = scorecard_prior_probability(sc.health["health_score"])
    preds = model.predict(te, prior)
    arr = te["arr"].to_numpy(float)
    p = {
        "scorecard": prior.to_numpy(float),
        "model": preds["model_churn_probability"].to_numpy(float),
        "blend": preds["churn_probability"].to_numpy(float),
    }
    methods = {k: classification_report(y, v, arr) for k, v in p.items()}
    base = float((tr["outcome"] == "churned").mean())
    base_brier = float(np.mean((y - base) ** 2))

    bands = []
    for band in ("green", "yellow", "red"):
        m = (sc.health["health_band"] == band).to_numpy()
        if m.sum():
            bands.append({"band": band, "n": int(m.sum()), "churn_rate": round(float(y[m].mean()), 4)})

    green = (sc.health["health_band"] == "green").to_numpy()
    flagged = green & (p["model"] >= watermelon_bar(config, model))
    wm: dict[str, Any] = {"flagged": int(flagged.sum()), "green": int(green.sum())}
    if flagged.sum():
        wm["churn_rate"] = round(float(y[flagged].mean()), 4)
    if (green & ~flagged).sum():
        wm["other_green_churn_rate"] = round(float(y[green & ~flagged].mean()), 4)

    pc = preds["contraction_probability"].to_numpy(float)
    fc = forecast_grr(arr, p["blend"], pc, model.contraction_depth, seed=seed)
    contracted = (te["outcome"] == "contracted").to_numpy()
    if "renewed_arr" in te.columns:
        renewed = te["renewed_arr"].astype(float).fillna(pd.Series(arr * (1 - model.contraction_depth)))
        contraction_loss = np.clip(arr - renewed.to_numpy(float), 0, None)
    else:
        contraction_loss = arr * model.contraction_depth
    loss = (y * arr).sum() + (contracted * contraction_loss).sum()
    actual = 1 - loss / arr.sum()
    grr = {
        "predicted_churned_arr_share": round(float((p["blend"] * arr).sum() / arr.sum()), 4),
        "actual_churned_arr_share": round(float((y * arr).sum() / arr.sum()), 4),
        "expected": fc["expected_grr"],
        "p10": fc["grr_p10"],
        "p90": fc["grr_p90"],
        "actual": round(float(actual), 4),
        "inside_range": bool(fc["grr_p10"] <= actual <= fc["grr_p90"]),
    }

    notes = []
    if "synthetic" in source.lower():
        notes.append("Synthetic data: this shows the method works end to end. It is not a benchmark.")

    def period(df: pd.DataFrame) -> tuple[str, str]:
        return (str(df["renewal_date"].min().date()), str(df["renewal_date"].max().date()))

    return Backtest(
        profile=config.profile,
        profile_label=config.label,
        source=source,
        n_train=len(tr),
        n_test=len(te),
        train_period=period(tr),
        test_period=period(te),
        test_churn_rate=round(float(y.mean()), 4),
        methods=methods,
        base_rate_brier=round(base_brier, 4),
        calibration=calibration_table(y, np.clip(p["blend"], 0, 1), bins=5),
        bands=bands,
        watermelons=wm,
        grr=grr,
        notes=notes,
    )


# -- report ---------------------------------------------------------------------------------------------------
def _pct(x: Any, digits: int = 0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.{digits}%}"


def _num(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None else f"{float(x):.{digits}f}"


def summary_row(b: Backtest) -> str:
    m = b.methods["blend"]
    s = b.methods["scorecard"]
    wm = b.watermelons
    if wm.get("flagged", 0) >= MIN_TO_JUDGE:
        wm_txt = f"{_pct(wm.get('churn_rate'))} vs {_pct(wm.get('other_green_churn_rate'))}"
    else:
        wm_txt = f"too few to judge ({wm.get('flagged', 0)} flagged)"
    grr = b.grr
    grr_txt = f"{_pct(grr['actual'], 1)} ({'inside' if grr['inside_range'] else 'outside'} range)"
    return (
        f"| {b.profile_label} | {_num(s.get('roc_auc'))} → **{_num(m.get('roc_auc'))}** | "
        f"{_num(m.get('lift_top_10pct'), 1)}× | {_pct(m.get('arr_capture_top_20pct'))} | {wm_txt} | {grr_txt} |"
    )


def to_markdown(results: list[Backtest], *, title: str = "How accurate is it?", intro: str = "") -> str:
    lines = [f"# {title}", ""]
    if intro:
        lines += [intro, ""]
    lines += [
        "Each profile is backtested the way it would be used: trained on earlier renewals, then scored on later "
        "renewals the model never saw, and compared with what actually happened.",
        "",
        "| Profile | ROC AUC: scorecard → scorer | Top-10% lift | Churned ARR in top 20% "
        "| Watermelon churn vs other green | Actual GRR vs forecast |",
        "|---|---|---|---|---|---|",
        *[summary_row(b) for b in results],
        "",
        "How to read it: ROC AUC is how often a churned account ranks above a renewed one (0.5 is a coin flip). "
        "Lift is how much more often the riskiest tenth churns than average. The watermelon column compares green "
        "accounts the scorer flagged with the green accounts it didn't.",
        "",
    ]
    for b in results:
        lines += _section(b)
    return "\n".join(lines).rstrip() + "\n"


def _section(b: Backtest) -> list[str]:
    out = [
        f"## {b.profile_label}",
        "",
        f"Trained on {b.n_train:,} renewals ({b.train_period[0]} to {b.train_period[1]}). Tested on the next "
        f"{b.n_test:,} ({b.test_period[0]} to {b.test_period[1]}), of which {_pct(b.test_churn_rate, 1)} churned."
        + (f" Source: {b.source}." if b.source else ""),
        "",
        "| Method | ROC AUC | PR AUC | Brier | Top-10% lift | Churned ARR in top 20% |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in METHODS.items():
        m = b.methods[key]
        out.append(
            f"| {label} | {_num(m.get('roc_auc'))} | {_num(m.get('pr_auc'))} | {_num(m.get('brier'), 4)} | "
            f"{_num(m.get('lift_top_10pct'), 2)} | {_pct(m.get('arr_capture_top_20pct'))} |"
        )
    out += [f"| Everyone at the historical churn rate | 0.500 | n/a | {_num(b.base_rate_brier, 4)} | 1.00 | 20% |", ""]
    out += [
        "**Calibration** (accounts split into fifths by predicted risk):",
        "",
        "| Predicted churn | Actual churn | Accounts |",
        "|---|---|---|",
        *[f"| {_pct(r['predicted'], 1)} | {_pct(r['actual'], 1)} | {r['n']:,} |" for r in b.calibration],
        "",
        "**Scorecard bands:** "
        + ", ".join(f"{r['band']} {_pct(r['churn_rate'], 1)} churned ({r['n']:,})" for r in b.bands)
        + ".",
        "",
    ]
    wm = b.watermelons
    if wm.get("flagged"):
        line = (
            f"**Watermelons:** {wm['flagged']} of {wm['green']:,} green accounts were flagged. "
            f"{_pct(wm.get('churn_rate'), 1)} of them churned, against {_pct(wm.get('other_green_churn_rate'), 1)} "
            "of the other green accounts."
        )
        if wm["flagged"] < MIN_TO_JUDGE:
            line += f" That is too few flags to judge (under {MIN_TO_JUDGE})."
        out.append(line)
    else:
        out.append(f"**Watermelons:** none flagged among {wm['green']:,} green accounts.")
    g = b.grr
    out += [
        "",
        f"**GRR forecast:** expected {_pct(g['expected'], 1)}, P10 {_pct(g['p10'], 1)} to P90 {_pct(g['p90'], 1)}. "
        f"Actual: {_pct(g['actual'], 1)}, {'inside' if g['inside_range'] else 'outside'} the range. "
        f"Predicted churned ARR was {_pct(g['predicted_churned_arr_share'], 1)} of the book; actual was "
        f"{_pct(g['actual_churned_arr_share'], 1)}.",
        "",
    ]
    out += [f"_{n}_" for n in b.notes] + ([""] if b.notes else [])
    return out
