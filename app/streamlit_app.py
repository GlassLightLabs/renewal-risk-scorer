"""Streamlit demo: run with `streamlit run app/streamlit_app.py` (after `pip install -e ".[app]"`).

Works on synthetic data out of the box, or on your own exports: upload a book of open renewals, and either
a renewal history to train on or a model.json from `renewal-risk train`.
"""

from __future__ import annotations

import io
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import renewal_risk  # noqa: F401
except ImportError:  # hosted demo: run from a clone without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renewal_risk.config import list_profiles, load_config
from renewal_risk.ingest import ingest
from renewal_risk.model import RiskModel, train
from renewal_risk.narrate import narrate_one
from renewal_risk.pipeline import ScoreResult, score
from renewal_risk.report import accounts_table, markdown_report
from renewal_risk.sources import DataSource, SourceError, SourceRun, apply_sources, discover
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


def _secret_defaults(name: str) -> dict[str, str]:
    """Settings from .streamlit/secrets.toml ([zendesk] subdomain=..., email=..., api_token=...), if any."""
    try:
        return {str(k): str(v) for k, v in dict(st.secrets.get(name, {})).items()}
    except Exception:  # no secrets file is the normal case
        return {}


def addon_settings(src: DataSource, demo: bool) -> dict[str, object] | None:
    """Draw one add-on's settings from its declared fields. Returns its config, or None when it is off."""
    name = src.name
    if not st.checkbox(f"Use {src.label}", value=demo, key=f"{name}_on", help=src.description):
        return None
    if src.supports_simulation and st.toggle(
        "Simulated demo data", value=demo, key=f"{name}_sim", help="No account or credentials needed."
    ):
        return {"simulate": True}
    saved = _secret_defaults(name)
    cfg: dict[str, object] = {}
    for f in src.fields:
        key = f"{name}_{f.name}"
        default = saved.get(f.name, "" if f.default is None else str(f.default))
        if f.kind == "choice" and f.choices:
            plain = [c for c in f.choices if not c.endswith(":")]
            options = plain + (["custom field"] if len(plain) < len(f.choices) else [])
            picked = st.selectbox(f.label, options, key=key, help=f.help)
            if picked == "custom field":
                field_key = st.text_input("Organization field key", key=f"{key}_field", placeholder="sfdc_account_id")
                cfg[f.name] = f"field:{field_key.strip()}" if field_key.strip() else "auto"
            else:
                cfg[f.name] = picked
        else:
            cfg[f.name] = st.text_input(
                f.label, value=default, key=key, help=f.help, type="password" if f.secret else "default"
            )
    missing = [f.label for f in src.fields if f.required and not str(cfg.get(f.name) or "").strip()]
    if missing:
        st.caption(f"Add {', '.join(missing)} to use {src.label}.")
        return None
    if st.button("Test connection", key=f"{name}_test"):
        try:
            st.success(src.test_connection(cfg))
        except SourceError as err:
            st.error(str(err))
    return cfg


def addons_panel(demo: bool) -> dict[str, dict[str, object]]:
    sources, problems = discover()
    chosen: dict[str, dict[str, object]] = {}
    with st.sidebar.expander("Data add-ons", expanded=False):
        st.caption("Pull signals from other tools into the book before scoring. Read-only.")
        for name, cls in sorted(sources.items()):
            cfg = addon_settings(cls(), demo)
            if cfg is not None:
                chosen[name] = cfg
        for p in problems:
            st.warning(p)
    return chosen


@st.cache_data(ttl=900, show_spinner="Reading data add-ons...")
def run_addons(
    book: pd.DataFrame, sources: dict[str, dict[str, object]], as_of: date
) -> tuple[pd.DataFrame, SourceRun]:
    # Cached for 15 minutes so moving around the app doesn't re-read Zendesk (and hit its rate limit).
    return apply_sources(book, sources, as_of=as_of)


def addons_report(run: SourceRun, book: pd.DataFrame) -> None:
    if not run.results and not run.errors:
        st.caption("No data add-ons in use. Turn one on under Data add-ons in the sidebar.")
        return
    for res in run.results:
        st.success(res.summary())
        for w in res.warnings:
            st.caption(w)
        if res.unmatched_accounts:
            with st.expander(f"{len(res.unmatched_accounts)} accounts not found in {res.source}"):
                st.caption(
                    "These keep the values from your export. To match them, set the organization's external ID "
                    "to the account ID, or try another 'Match organizations by' setting."
                )
                cols = [c for c in ("account_id", "account_name", "arr", "renewal_date") if c in book.columns]
                missing = book[book["account_id"].astype(str).isin(res.unmatched_accounts)][cols]
                st.dataframe(
                    missing.sort_values("arr", ascending=False) if "arr" in cols else missing,
                    hide_index=True,
                    width="stretch",
                    column_config={"arr": st.column_config.NumberColumn("ARR", format="$%d")},
                )
    for name, err in run.errors.items():
        st.error(f"{name}: not used. {err}")


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


SUPPORT_FIELDS = {
    "tickets_90d": ("Tickets (90d)", "{:.0f}"),
    "tickets_90d_prior": ("Previous 90d", "{:.0f}"),
    "tickets_per_100_seats": ("Per 100 seats", "{:.1f}"),
    "p1_tickets_90d": ("Urgent", "{:.0f}"),
    "csat": ("CSAT", "{:.0f}%"),
    "ticket_reopen_rate_pct": ("Reopened", "{:.0f}%"),
    "ticket_resolution_hours": ("Median fix", "{:.0f} h"),
}


def support_panel(book: pd.DataFrame, account_id: object, run: SourceRun) -> None:
    """The account's support signals, with where they came from."""
    rows = book[book["account_id"].astype(str) == str(account_id)]
    if rows.empty:
        return
    rec = rows.iloc[0]
    shown = [(label, fmt.format(rec[k])) for k, (label, fmt) in SUPPORT_FIELDS.items() if k in rec and pd.notna(rec[k])]
    if not shown:
        return
    origin = "your export"
    for res in run.results:
        if str(account_id) not in res.unmatched_accounts and res.matched_accounts:
            origin = res.source + (" (simulated)" if res.simulated else "")
    st.subheader("Support")
    st.caption(f"From {origin}")
    cols = st.columns(min(4, len(shown)))
    for i, (label, value) in enumerate(shown):
        cols[i % len(cols)].metric(label, value)


def account_tab(r: ScoreResult, book: pd.DataFrame, run: SourceRun) -> None:
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
        support_panel(book, row["account_id"], run)
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
    plan_size_panel(r.summary.get("by_plan_size") or [])


def plan_size_panel(bands: list[dict]) -> None:
    has_bench = any("benchmark_grr" in b for b in bands)
    if len(bands) < 2 and not has_bench:
        return
    st.subheader("Retention by plan size")
    table = pd.DataFrame(
        [
            {
                "Monthly spend": b["band"],
                "Accounts": b["accounts"],
                "ARR": money(b["arr"]),
                "Expected loss": money(b["expected_arr_loss"]),
                "Expected GRR": f"{b['expected_grr']:.0%}" if b["expected_grr"] is not None else "n/a",
                **({"AI-native median GRR": f"{b['benchmark_grr']:.0%}"} if has_bench else {}),
            }
            for b in bands
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    if has_bench:
        st.caption(
            "Benchmark: median GRR for AI-native products by monthly spend (ChartMogul, December 2025). "
            "Small self-serve plans churn far faster than committed spend, so read each band on its own."
        )


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
    chosen = addons_panel(demo=source == "Synthetic demo")
    run = SourceRun()
    if chosen:
        book, run = run_addons(book, chosen, as_of)
        for name, err in run.errors.items():
            st.sidebar.error(f"{name}: {err}")
    result = score(book, load_config(profile), model, as_of=as_of, horizon_days=horizon, quality=report)  # type: ignore[arg-type]
    st.title(f"Renewal risk: {result.config.label}")
    used = [r.source + (" (simulated)" if r.simulated else "") for r in run.results]
    st.caption(
        f"Risk source: {result.summary['risk_source']}" + (f" · Data add-ons: {', '.join(used)}" if used else "")
    )
    kpis(result)
    tabs = st.tabs(["Priority list", "Account detail", "Portfolio", "Model quality", "Data quality"])
    with tabs[0]:
        priority_tab(result)
    with tabs[1]:
        account_tab(result, book, run)
    with tabs[2]:
        portfolio_tab(result)
    with tabs[3]:
        model_tab(result)
    with tabs[4]:
        st.subheader("Data add-ons")
        addons_report(run, book)
        st.subheader("Columns and coverage")
        st.text(report.summary())  # type: ignore[attr-defined]


main()
