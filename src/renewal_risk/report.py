"""Write scoring results as files a team can use: a ranked CSV, a JSON summary and a Markdown brief."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import ScoreResult

CSV_COLUMNS = [
    "rank",
    "account_id",
    "account_name",
    "segment",
    "csm_owner",
    "arr",
    "renewal_date",
    "days_to_renewal",
    "risk_tier",
    "churn_probability",
    "contraction_probability",
    "expected_arr_loss",
    "priority_arr",
    "health_score",
    "health_band",
    "confidence",
    "watermelon",
    "top_reason",
    "risk_drivers",
    "protective_factors",
    "scorecard_gaps",
    "alerts",
    "save_play",
    "play_owner",
    "play_timing",
    "play_actions",
    "model_churn_probability",
    "scorecard_churn_probability",
    "risk_source",
]


def _fmt_items(items: list[dict[str, Any]], key: str = "label") -> str:
    parts = []
    for it in items:
        if "value" in it:
            parts.append(f"{it[key]}: {it['value']}")
        elif "points" in it:
            parts.append(f"{it[key]} (-{it['points']} pts)")
        else:
            parts.append(str(it.get("message", it)))
    return "; ".join(parts)


def accounts_table(result: ScoreResult) -> pd.DataFrame:
    """Flat, spreadsheet-friendly version of the ranked accounts."""
    df = result.accounts.copy()
    df["risk_drivers"] = df["risk_drivers"].map(_fmt_items)
    df["protective_factors"] = df["protective_factors"].map(_fmt_items)
    df["scorecard_gaps"] = df["scorecard_gaps"].map(_fmt_items)
    df["alerts"] = df["alerts"].map(lambda xs: "; ".join(a["message"] for a in xs))
    df["play_actions"] = df["play_actions"].map(lambda xs: " | ".join(xs))
    group_cols = [c for c in df.columns if c.endswith("_score") and c != "health_score"]
    cols = [c for c in CSV_COLUMNS if c in df.columns] + group_cols
    return df[cols]


def _money(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}k"
    return f"${x:,.0f}"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def _r3(v: object, nd: int = 3) -> str:
    return "" if v is None else f"{float(v):.{nd}f}"  # type: ignore[arg-type]


def markdown_report(result: ScoreResult, *, top: int = 15, briefs_md: str | None = None) -> str:
    s, fc, a = result.summary, result.forecast, result.accounts
    tiers = s["accounts_by_tier"]
    lines = [
        f"# Renewal risk report: {s['profile_label']}",
        "",
        f"As of {s['as_of']}. {s['accounts']} accounts, {_money(s['total_arr'])} ARR. Risk source: {s['risk_source']}.",
        "",
        "## Summary",
        "",
        f"- **Expected ARR loss (whole book):** {_money(s['expected_arr_loss'])}",
        "- **Priority tiers:** " + ", ".join(f"{k} {v} ({_money(s['arr_by_tier'][k])} ARR)" for k, v in tiers.items()),
        f"- **Watermelons (green score, model sees risk):** {s['watermelons']}",
        f"- **Overdue renewals:** {s['overdue_renewals']}",
        f"- **Low-confidence scores (thin data):** {s['low_confidence_scores']}",
    ]
    if fc.get("accounts"):
        horizon = f"next {fc['horizon_days']} days" if fc.get("horizon_days") is not None else "all open renewals"
        lines += [
            "",
            f"## Gross revenue retention forecast ({horizon})",
            "",
            f"{fc['accounts']} renewals worth {_money(fc['arr_up_for_renewal'])}. Expected GRR "
            f"**{fc['expected_grr']:.1%}** (P10 {fc['grr_p10']:.1%}, P50 {fc['grr_p50']:.1%}, "
            f"P90 {fc['grr_p90']:.1%}). Expected loss {_money(fc['expected_arr_loss'])}; "
            f"in a bad quarter (P90 loss) {_money(fc['arr_loss_p90'])}. "
            f"About {fc['expected_churned_logos']:.0f} logos expected to churn.",
            "",
            "_Monte Carlo over account-level churn and contraction probabilities, treated as independent, so the "
            "P10 understates correlated shocks._",
        ]
    view = a.head(top).copy()
    view["ARR"] = view["arr"].map(_money)
    view["Exp. loss"] = view["expected_arr_loss"].map(_money)
    view["Churn"] = view["churn_probability"].map(lambda p: f"{p:.0%}")
    view["Downsell"] = view["contraction_probability"].map(lambda p: f"{p:.0%}")
    view["Days"] = view["days_to_renewal"].map(lambda d: f"{d:.0f}")
    view["Health"] = view.apply(lambda r: f"{r['health_score']:.0f} {r['health_band']}", axis=1)
    view["Why"] = view["top_reason"]
    view["Play"] = view["save_play"]
    view["Account"] = view["account_name"].fillna(view["account_id"]) + view["watermelon"].map(
        lambda w: " (watermelon)" if w else ""
    )
    lines += [
        "",
        f"## Top {len(view)} accounts to act on",
        "",
        _md_table(
            view[
                [
                    "rank",
                    "Account",
                    "ARR",
                    "Days",
                    "Health",
                    "Churn",
                    "Downsell",
                    "Exp. loss",
                    "risk_tier",
                    "Why",
                    "Play",
                ]
            ].rename(columns={"rank": "#", "risk_tier": "Tier"})
        ),
    ]
    wm = a[a["watermelon"]].head(10)
    if len(wm):
        lines += [
            "",
            "## Watermelons",
            "",
            "Green on the scorecard, but the model rates them at several times the churn risk green accounts "
            "usually carry. Pressure-test these before the renewal conversation.",
            "",
        ]
        for _, r in wm.iterrows():
            drivers = _fmt_items(r["risk_drivers"]) or "see account detail"
            lines.append(
                f"- **{r.get('account_name') or r['account_id']}** ({_money(r['arr'])}, "
                f"{r['days_to_renewal']:.0f} days): {drivers}"
            )
    if len(result.drivers):
        d = result.drivers.head(8)
        lines += [
            "",
            "## What is driving risk across the book",
            "",
            _md_table(
                pd.DataFrame(
                    {
                        "Driver": d["driver"],
                        "Group": d["group_label"],
                        "Accounts": d["accounts"],
                        "Share of risk (ARR-weighted)": d["share_of_risk"].map(lambda x: f"{x:.0%}"),
                    }
                )
            ),
        ]
    if result.model_metrics.get("churn"):
        m = result.model_metrics["churn"]
        b = m.get("scorecard_baseline", {})
        bl = result.model_metrics.get("blend") or {}
        rows = [("Scorecard alone", b), ("Model alone", m)] + ([("Model + scorecard (used)", bl)] if bl else [])
        lines += [
            "",
            "## Model quality (cross-validated on training history)",
            "",
            _md_table(
                pd.DataFrame(
                    [
                        {
                            "Approach": n,
                            "ROC AUC": _r3(r.get("roc_auc")),
                            "PR AUC": _r3(r.get("pr_auc")),
                            "Brier": _r3(r.get("brier"), 4),
                            "Top-10% lift": _r3(r.get("lift_top_10pct"), 2),
                            "Churned ARR in top 20%": f"{r.get('arr_capture_top_20pct', 0):.0%}",
                        }
                        for n, r in rows
                    ]
                )
            ),
        ]
    if result.quality is not None:
        q = result.quality
        lines += ["", "## Data quality", "", f"{q.rows_in} rows read, {q.rows_out} scored."]
        lines += [f"- Dropped {n}: {why}" for why, n in q.dropped.items()]
        lines += [f"- {w}" for w in q.warnings]
        if q.unmapped_columns:
            lines.append("- Ignored columns: " + ", ".join(q.unmapped_columns))
    if briefs_md:
        lines += ["", "## Account briefs", "", briefs_md]
    return "\n".join(lines) + "\n"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def write_outputs(
    result: ScoreResult,
    out_dir: str | Path,
    *,
    top: int = 15,
    briefs: list[dict[str, Any]] | None = None,
    briefs_md: str | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"accounts": out / "scored_accounts.csv", "summary": out / "summary.json", "report": out / "report.md"}
    accounts_table(result).to_csv(paths["accounts"], index=False)
    payload = {
        "summary": result.summary,
        "drivers": result.drivers.to_dict(orient="records"),
        "model_metrics": result.model_metrics,
        "data_quality": result.quality.to_dict() if result.quality else None,
    }
    if briefs is not None:
        payload["briefs"] = briefs
    paths["summary"].write_text(json.dumps(_jsonable(payload), indent=2, default=str))
    paths["report"].write_text(markdown_report(result, top=top, briefs_md=briefs_md))
    return paths
