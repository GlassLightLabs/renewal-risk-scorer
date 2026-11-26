"""Streamlit demo: run with `streamlit run app/streamlit_app.py` (after `pip install -e ".[app]"`).

Works on synthetic data out of the box, or on your own exports: upload a book of open renewals, and either
a renewal history to train on or a model.json from `renewal-risk train`.
"""

from __future__ import annotations

import io
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from renewal_risk.config import list_profiles, load_config
from renewal_risk.ingest import ingest
from renewal_risk.model import RiskModel, train
from renewal_risk.narrate import narrate_one
from renewal_risk.pipeline import ScoreResult, score
from renewal_risk.report import accounts_table, markdown_report
from renewal_risk.synth import generate_book, generate_history

st.set_page_config(page_title="Renewal Risk Scorer", layout="wide")


@st.cache_resource(show_spinner="Training on synthetic renewal history...")
def demo_model(profile: str, as_of: date) -> RiskModel:
    hist, _ = ingest(generate_history(4000, profile=profile, as_of=as_of), as_of=as_of, require_outcome=True)
    return train(hist, load_config(profile))


@st.cache_data(show_spinner=False)
def demo_book(profile: str, as_of: date) -> pd.DataFrame:
    return generate_book(600, profile=profile, as_of=as_of)


def money(x: float) -> str:
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}k"


def sidebar() -> tuple[str, date, int, str]:
    st.sidebar.title("Renewal Risk Scorer")
    profiles = list_profiles()
    names = list(profiles)
    profile = st.sidebar.selectbox(
        "Industry profile", names, index=names.index("ai_usage"), format_func=lambda p: load_config(p).label
    )
    st.sidebar.caption(profiles[profile])
    as_of = st.sidebar.date_input("As of", value=date.today())
    horizon = st.sidebar.slider("GRR forecast window (days)", 30, 365, 180, step=30)
    source = st.sidebar.radio("Data", ["Synthetic demo", "Upload my exports"])
    return profile, as_of, horizon, source


def load_inputs(profile: str, as_of: date, source: str) -> tuple[pd.DataFrame, RiskModel | None, object]:
    if source == "Synthetic demo":
        book, report = ingest(demo_book(profile, as_of), as_of=as_of)
        return book, demo_model(profile, as_of), report
    book_file = st.sidebar.file_uploader("Open renewals (CSV)", type="csv")
    hist_file = st.sidebar.file_uploader("Renewal history with outcomes (CSV, optional)", type="csv")
    model_file = st.sidebar.file_uploader("Or a trained model.json (optional)", type="json")
    if book_file is None:
        st.info(
            "Upload a CSV of open renewals. Column names are matched automatically (Salesforce and "
            "Gainsight style headers work); `renewal-risk inspect` shows the mapping."
        )
        st.stop()
    book, report = ingest(io.BytesIO(book_file.getvalue()), as_of=as_of)  # type: ignore[arg-type]
    model = None
    if model_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(model_file.getvalue())
        model = RiskModel.load(Path(tmp.name))
    elif hist_file is not None:
        hist, _ = ingest(io.BytesIO(hist_file.getvalue()), as_of=as_of, require_outcome=True)  # type: ignore[arg-type]
        with st.spinner("Training on your renewal history..."):
            model = train(hist, load_config(profile))
    else:
        st.warning("No history or model uploaded: risk comes from the scorecard alone, uncalibrated.")
    return book, model, report


def kpis(r: ScoreResult) -> None:
    s, fc = r.summary, r.forecast
    c = st.columns(5)
    c[0].metric("ARR in book", money(s["total_arr"]), f"{s['accounts']} accounts", delta_color="off")
    c[1].metric("Expected ARR loss", money(s["expected_arr_loss"]))
    if fc.get("accounts"):
        c[2].metric(
            f"Expected GRR, next {fc['horizon_days']}d",
            f"{fc['expected_grr']:.1%}",
            f"P10 {fc['grr_p10']:.1%} to P90 {fc['grr_p90']:.1%}",
            delta_color="off",
        )
    t = s["accounts_by_tier"]
    c[3].metric("Critical / high", f"{t.get('critical', 0)} / {t.get('high', 0)}")
    c[4].metric("Watermelons", s["watermelons"], help="Green on the scorecard, high risk in the model")


def priority_tab(r: ScoreResult) -> None:
    table = accounts_table(r)
    tiers = st.multiselect("Tier", ["critical", "high", "watch", "low"], default=["critical", "high", "watch"])
    owners = sorted(table["csm_owner"].dropna().unique()) if "csm_owner" in table else []
    owner = st.selectbox("CSM", ["All", *owners])
    view = table[table["risk_tier"].isin(tiers)]
    if owner != "All":
        view = view[view["csm_owner"] == owner]
    cols = [
        "rank",
        "account_name",
        "arr",
        "days_to_renewal",
        "risk_tier",
        "churn_probability",
        "contraction_probability",
        "expected_arr_loss",
        "health_score",
        "health_band",
        "watermelon",
        "top_reason",
        "save_play",
    ]
    st.dataframe(
        view[[c for c in cols if c in view]],
        hide_index=True,
        width="stretch",
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Churn", format="%.0f%%", min_value=0, max_value=1),
            "contraction_probability": st.column_config.NumberColumn("Downsell", format="%.2f"),
            "arr": st.column_config.NumberColumn("ARR", format="$%d"),
            "expected_arr_loss": st.column_config.NumberColumn("Exp. loss", format="$%d"),
        },
    )
    st.download_button("Download scored accounts (CSV)", table.to_csv(index=False), "scored_accounts.csv")
    st.download_button("Download report (Markdown)", markdown_report(r), "report.md")


def account_tab(r: ScoreResult) -> None:
    a = r.accounts
    labels = (a["rank"].astype(str) + ". " + a["account_name"].fillna(a["account_id"])).tolist()
    pick = st.selectbox("Account", range(len(a)), format_func=lambda i: labels[i])
    row = a.iloc[pick]
    c = st.columns(4)
    c[0].metric("ARR", money(row["arr"]))
    c[1].metric(
        "Churn risk",
        f"{row['churn_probability']:.0%}",
        f"downsell {row['contraction_probability']:.0%}",
        delta_color="off",
    )
    c[2].metric(
        "Health",
        f"{row['health_score']:.0f} ({row['health_band']})",
        f"confidence {row['confidence']}",
        delta_color="off",
    )
    c[3].metric("Days to renewal", f"{row['days_to_renewal']:.0f}", row["risk_tier"], delta_color="off")
    if row["watermelon"]:
        st.warning("Watermelon: the scorecard is green, but the model sees risk it is missing.")
    left, right = st.columns(2)
    with left:
        st.subheader("Why")
        for d in row["risk_drivers"]:
            st.markdown(f"- **{d['label']}**: {d['value']}  (+{d['effect']:.2f} log-odds)")
        for al in row["alerts"]:
            st.markdown(f"- :red[{al['severity']}] {al['message']}")
        if row["protective_factors"]:
            st.caption(
                "Working in their favor: " + "; ".join(f"{d['label']}: {d['value']}" for d in row["protective_factors"])
            )
        groups = {g.label: row.get(f"{g.name}_score") for g in r.config.groups}
        st.bar_chart(pd.Series(groups, name="Group score").dropna(), horizontal=True)
    with right:
        st.subheader(f"Play: {row['save_play']}")
        st.caption(f"{row['play_owner']} · {row['play_timing']}")
        for step in row["play_actions"]:
            st.markdown(f"1. {step}")
        if st.button("Write account brief"):
            brief = narrate_one(row)
            st.markdown(f"**{brief['headline']}**")
            st.markdown("\n".join(f"- {w}" for w in brief["why_at_risk"]))
            st.markdown("\n".join(f"1. {s['action']} ({s['owner']}, {s['when']})" for s in brief["save_plan"]))
            st.info(brief["talk_track"])
            st.caption(f"Source: {brief.get('source')}" + (f" ({brief['note']})" if brief.get("note") else ""))


def portfolio_tab(r: ScoreResult) -> None:
    st.subheader("What is driving risk across the book")
    d = r.drivers.head(12)
    st.bar_chart(d.set_index("driver")["share_of_risk"], horizontal=True)
    st.subheader("Watermelons")
    wm = r.accounts[r.accounts["watermelon"]]
    if len(wm):
        st.dataframe(
            accounts_table(r).loc[
                wm.index,
                ["account_name", "arr", "days_to_renewal", "health_score", "model_churn_probability", "risk_drivers"],
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("None in this book.")


def model_tab(r: ScoreResult) -> None:
    m = r.model_metrics.get("churn")
    if not m:
        st.info("No trained model: scores come from the scorecard alone.")
        return
    rows = {"Scorecard alone": m.get("scorecard_baseline", {}), "Model alone": m}
    if r.model_metrics.get("blend"):
        rows["Model + scorecard (used)"] = r.model_metrics["blend"]
    st.dataframe(
        pd.DataFrame(
            {
                k: {x: v.get(x) for x in ("roc_auc", "pr_auc", "brier", "lift_top_10pct", "arr_capture_top_20pct")}
                for k, v in rows.items()
            }
        ).T
    )
    st.caption(m.get("evaluation", ""))
    cal = pd.DataFrame(rows.get("Model + scorecard (used)", m).get("calibration", []))
    if len(cal):
        st.subheader("Calibration: predicted vs actual churn by decile")
        st.line_chart(cal.set_index("predicted")["actual"])


def main() -> None:
    profile, as_of, horizon, source = sidebar()
    book, model, report = load_inputs(profile, as_of, source)
    result = score(book, load_config(profile), model, as_of=as_of, horizon_days=horizon, quality=report)  # type: ignore[arg-type]
    st.title(f"Renewal risk: {result.config.label}")
    st.caption(f"Risk source: {result.summary['risk_source']}")
    kpis(result)
    tabs = st.tabs(["Priority list", "Account detail", "Portfolio", "Model quality", "Data quality"])
    with tabs[0]:
        priority_tab(result)
    with tabs[1]:
        account_tab(result)
    with tabs[2]:
        portfolio_tab(result)
    with tabs[3]:
        model_tab(result)
    with tabs[4]:
        st.text(report.summary())  # type: ignore[attr-defined]


main()
