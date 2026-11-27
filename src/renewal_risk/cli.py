"""Command-line interface: ``renewal-risk <command>``.

renewal-risk profiles                                    list industry profiles
renewal-risk inspect book.csv                            show how columns map and what data is thin
renewal-risk generate --profile ai_usage --out data/     write synthetic history and book CSVs
renewal-risk train history.csv --profile ai_usage        train and evaluate; writes model.json
renewal-risk score book.csv --model model.json           rank, explain and dollarize renewal risk
renewal-risk demo --profile ai_usage                     all of the above on synthetic data
renewal-risk evaluate history.csv --profile saas         backtest on your history; writes evaluation.md
renewal-risk sources list                                show installed data add-ons (Zendesk...)
renewal-risk sources test --sources sources.yaml         check add-on settings and credentials

score, inspect and demo take --sources sources.yaml to pull add-on signals into the book before scoring.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, ScorecardConfig, list_profiles, load_config
from .ingest import IngestError, ingest, load_mapping
from .model import RiskModel, train
from .narrate import narrate, to_markdown
from .pipeline import ScoreResult, score
from .playbook import load_playbook, missing_alert_plays
from .report import write_outputs
from .sources import SourceError, apply_sources, discover, get_source, load_sources_config


def _with_sources(args: argparse.Namespace, df: Any, as_of: date | None) -> Any:
    """Run the add-ons named in --sources and merge their signals; report what matched and what failed."""
    path = getattr(args, "sources", None)
    if not path:
        return df
    out, run = apply_sources(df, load_sources_config(path), as_of=as_of or date.today())
    for line in run.lines():
        print(line)
    return out


def _config(args: argparse.Namespace) -> ScorecardConfig:
    return load_config(profile=getattr(args, "profile", None), path=getattr(args, "config", None))


def _as_of(args: argparse.Namespace) -> date | None:
    return date.fromisoformat(args.as_of) if getattr(args, "as_of", None) else None


def _metrics_table(m: dict[str, Any], blend: dict[str, Any] | None) -> str:
    b = m.get("scorecard_baseline", {})
    rows = [("Scorecard alone", b), ("Model alone", m)] + ([("Model + scorecard", blend)] if blend else [])
    head = f"{'':22}{'ROC AUC':>9}{'PR AUC':>9}{'Brier':>9}{'Lift@10%':>10}{'ARR@20%':>9}"
    lines = [head]
    for name, r in rows:
        lines.append(
            f"{name:22}{r.get('roc_auc', float('nan')):>9.3f}{r.get('pr_auc', float('nan')):>9.3f}"
            f"{r.get('brier', float('nan')):>9.4f}{r.get('lift_top_10pct', float('nan')):>10.2f}"
            f"{r.get('arr_capture_top_20pct', float('nan')):>9.0%}"
        )
    return "\n".join(lines)


def _print_top(result: ScoreResult, n: int) -> None:
    s = result.summary
    print(
        f"\n{s['profile_label']}: {s['accounts']} accounts, ${s['total_arr']:,.0f} ARR, "
        f"expected ARR loss ${s['expected_arr_loss']:,.0f}"
    )
    print(
        "Tiers: "
        + ", ".join(f"{k} {v}" for k, v in s["accounts_by_tier"].items())
        + f" | watermelons {s['watermelons']} | overdue {s['overdue_renewals']}"
    )
    fc = result.forecast
    if fc.get("accounts"):
        print(
            f"GRR forecast ({fc['accounts']} renewals): expected {fc['expected_grr']:.1%}, "
            f"P10 {fc['grr_p10']:.1%} to P90 {fc['grr_p90']:.1%}"
        )
    print(
        f"\n{'#':>3}  {'Account':28}{'ARR':>12}{'Days':>6}{'Churn':>7}{'Downsell':>9}{'Exp. loss':>12}  {'Tier':9}Why"
    )
    for _, r in result.accounts.head(n).iterrows():
        name = str(r.get("account_name") or r["account_id"])[:27] + ("*" if r["watermelon"] else "")
        print(
            f"{r['rank']:>3}  {name:28}{r['arr']:>12,.0f}{r['days_to_renewal']:>6.0f}{r['churn_probability']:>7.0%}"
            f"{r['contraction_probability']:>9.0%}{r['expected_arr_loss']:>12,.0f}  "
            f"{r['risk_tier']:9}{str(r['top_reason'])[:70]}"
        )
    if result.summary["watermelons"]:
        print("* watermelon: green on the scorecard, high risk in the model")


def cmd_profiles(args: argparse.Namespace) -> int:
    for name, desc in list_profiles().items():
        print(f"{name:13} {desc}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    df, report = ingest(args.file, as_of=_as_of(args), mapping=load_mapping(args.mapping))
    print(report.summary())
    df = _with_sources(args, df, _as_of(args))
    cfg = _config(args)
    present = [f.name for f in cfg.model_features if f.name in df.columns and df[f.name].notna().any()]
    missing = [f.name for f in cfg.features if f.name not in present]
    print(f"\n{cfg.label}: {len(present)} of {len(cfg.model_features)} model signals present.")
    if missing:
        print("Not found (scored with reweighting): " + ", ".join(missing))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .synth import generate_book, generate_history, to_gainsight_export, to_salesforce_report

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    as_of = _as_of(args) or date.today()
    generate_history(args.history, profile=args.profile, as_of=as_of, seed=args.seed).to_csv(
        out / f"{args.profile}_history.csv", index=False
    )
    book = generate_book(args.book, profile=args.profile, as_of=as_of, seed=args.seed + 4)
    book.to_csv(out / f"{args.profile}_book.csv", index=False)
    written = [f"{args.profile}_history.csv", f"{args.profile}_book.csv"]
    if args.profile == "saas":
        to_salesforce_report(book).to_csv(out / "salesforce_report.csv", index=False)
        to_gainsight_export(book).to_csv(out / "gainsight_export.csv", index=False)
        written += ["salesforce_report.csv", "gainsight_export.csv"]
    print(f"Wrote {', '.join(written)} to {out} (as of {as_of}).")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _config(args)
    df, report = ingest(args.history, as_of=_as_of(args), mapping=load_mapping(args.mapping), require_outcome=True)
    print(report.summary())
    model = train(df, cfg, seed=args.seed, stack_scorecard=not args.no_stack)
    m = model.churn.metrics
    print(f"\nChurn model, {m['evaluation']}, {m['n_rows']} renewals ({m['positives']} churned):")
    print(_metrics_table(m, model.blend.metrics if model.blend else None))
    if model.blend:
        print(f"Stack weights: model {model.blend.w_model:.2f}, scorecard {model.blend.w_scorecard:.2f}")
    if model.contraction:
        print(
            f"Contraction model ROC AUC {model.contraction.metrics['roc_auc']:.3f}; "
            f"typical contraction depth {model.contraction_depth:.0%}"
        )
    model.save(args.out)
    print(f"\nSaved {args.out}")
    return 0


def _score(args: argparse.Namespace, cfg: ScorecardConfig, model: RiskModel | None) -> ScoreResult:
    df, report = ingest(args.book, as_of=_as_of(args), mapping=load_mapping(args.mapping))
    df = _with_sources(args, df, _as_of(args))
    return score(
        df,
        cfg,
        model,
        as_of=_as_of(args),
        horizon_days=args.horizon_days,
        quality=report,
        playbook=load_playbook(args.playbook),
    )


def _write(result: ScoreResult, args: argparse.Namespace) -> None:
    briefs = narrate(result.accounts, top=args.narrate) if args.narrate else None
    paths = write_outputs(
        result, args.out, top=args.top, briefs=briefs, briefs_md=to_markdown(briefs) if briefs else None
    )
    _print_top(result, args.top)
    if briefs:
        sources = {b.get("source") for b in briefs}
        print(f"\nAccount briefs: {len(briefs)} ({', '.join(sorted(map(str, sources)))})")
    print("\nWrote " + ", ".join(str(p) for p in paths.values()))


def cmd_score(args: argparse.Namespace) -> int:
    model = RiskModel.load(args.model) if args.model else None
    if model is not None and not getattr(args, "profile", None) and not getattr(args, "config", None):
        args.profile = model.profile
    cfg = _config(args)
    if model is None:
        print(
            "No model given: using the scorecard's prior probability. Train on your renewal history for "
            "calibrated risk (renewal-risk train).",
            file=sys.stderr,
        )
    _write(_score(args, cfg, model), args)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .synth import generate_book, generate_history

    cfg = load_config(args.profile)
    playbook = load_playbook(args.playbook)
    as_of = _as_of(args) or date.today()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic {cfg.label} data as of {as_of}...")
    hist, _ = ingest(
        generate_history(args.history, profile=args.profile, as_of=as_of), as_of=as_of, require_outcome=True
    )
    model = train(hist, cfg)
    model.save(out / "model.json")
    m = model.churn.metrics
    print(_metrics_table(m, model.blend.metrics if model.blend else None))
    book, report = ingest(generate_book(args.book, profile=args.profile, as_of=as_of), as_of=as_of)
    book = _with_sources(args, book, as_of)
    result = score(book, cfg, model, as_of=as_of, horizon_days=args.horizon_days, quality=report, playbook=playbook)
    _write(result, args)
    return 0


def cmd_playbook(args: argparse.Namespace) -> int:
    """Validate a playbook and show which alerts, in which profiles, have no play behind them."""
    pb = load_playbook(args.playbook)
    cfgs = [_config(args)] if args.config else [load_config(p) for p in list_profiles()]
    print(f"Playbook OK: {len(pb.alerts)} alert plays, {len(pb.groups)} group plays, tiers {', '.join(pb.tiers)}")
    gaps = 0
    for cfg in cfgs:
        no_group = sorted(g.name for g in cfg.groups if g.name not in pb.groups)
        missing = missing_alert_plays((a.id for a in cfg.alerts), pb)
        if missing or no_group:
            gaps += 1
            print(
                f"  {cfg.profile}: alerts with no play: {', '.join(missing) or 'none'}; "
                f"groups with no play: {', '.join(no_group) or 'none'}"
            )
        else:
            print(f"  {cfg.profile}: every alert and group has a play")
    return 1 if gaps else 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import backtest
    from .evaluate import to_markdown as backtest_markdown

    results = []
    if args.demo:
        from .synth import generate_history

        as_of = _as_of(args) or date.today()
        for profile in list_profiles():
            print(f"Backtesting {profile} on synthetic history...")
            hist, _ = ingest(
                generate_history(args.history, profile=profile, as_of=as_of, seed=args.seed),
                as_of=as_of,
                require_outcome=True,
            )
            results.append(backtest(hist, load_config(profile), source="synthetic data", seed=args.seed))
        intro = (
            "These numbers come from the bundled synthetic data, so they show the method works end to end; they are "
            "not a benchmark. Run the same backtest on your own history with "
            "`renewal-risk evaluate past_renewals.csv --profile saas`."
        )
    else:
        if not args.history_file:
            print("error: pass a history CSV, or --demo for synthetic data", file=sys.stderr)
            return 2
        cfg = _config(args)
        hist, report = ingest(
            args.history_file, as_of=_as_of(args), mapping=load_mapping(args.mapping), require_outcome=True
        )
        print(report.summary())
        results.append(backtest(hist, cfg, test_share=args.test_share, source=Path(args.history_file).name))
        intro = ""
    md = backtest_markdown(results, intro=intro)
    Path(args.out).write_text(md)
    for r in results:
        m = r.methods["blend"]
        print(
            f"{r.profile_label}: ROC AUC {m.get('roc_auc')} (scorecard alone {r.methods['scorecard'].get('roc_auc')}), "
            f"actual GRR {r.grr['actual']:.1%} vs forecast {r.grr['p10']:.1%} to {r.grr['p90']:.1%}"
        )
    print(f"Wrote {args.out}")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    if args.action == "list":
        sources, problems = discover()
        for name, cls in sorted(sources.items()):
            print(f"{name:12} {cls.label}: {cls.description}")
            print(f"{'':12} settings: {', '.join(f.name for f in cls.fields)}")
        for p in problems:
            print(f"warning: {p}", file=sys.stderr)
        return 0
    if not args.sources:
        print("error: pass --sources sources.yaml (see examples/sources.yaml)", file=sys.stderr)
        return 2
    failed = 0
    for name, cfg in load_sources_config(args.sources).items():
        try:
            print(get_source(name).test_connection(cfg))
        except SourceError as err:
            failed += 1
            print(f"{name}: {err}", file=sys.stderr)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="renewal-risk", description="Explainable renewal risk scoring for CS and renewals teams."
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser, profile_default: str | None = "saas") -> None:
        sp.add_argument(
            "--profile",
            default=profile_default,
            choices=list(list_profiles()),
            help="industry profile (default: %(default)s)",
        )
        sp.add_argument("--config", help="custom scorecard YAML (overrides --profile)")
        sp.add_argument("--mapping", help="YAML file mapping your column names to canonical fields")
        sp.add_argument("--as-of", help="score as of this date (YYYY-MM-DD); default today")
        sp.add_argument("--sources", help="YAML file of data add-ons to pull in (see examples/sources.yaml)")

    sp = sub.add_parser("profiles", help="list industry profiles")
    sp.set_defaults(func=cmd_profiles)

    sp = sub.add_parser("inspect", help="show column mapping and data coverage for a CSV")
    sp.add_argument("file")
    common(sp)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("generate", help="write synthetic history and book CSVs")
    sp.add_argument("--profile", default="saas", choices=list(list_profiles()))
    sp.add_argument("--out", default="data")
    sp.add_argument("--history", type=int, default=4000, help="past renewals to generate")
    sp.add_argument("--book", type=int, default=600, help="open renewals to generate")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--as-of")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("train", help="train and evaluate on renewal history")
    sp.add_argument("history")
    common(sp)
    sp.add_argument("--out", default="model.json")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--no-stack", action="store_true", help="use the model alone, without the scorecard stack")
    sp.set_defaults(func=cmd_train)

    def outputs(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--out", default="out", help="output directory")
        sp.add_argument("--top", type=int, default=15, help="accounts to show")
        sp.add_argument("--horizon-days", type=int, default=180, help="GRR forecast window (default 180)")
        sp.add_argument(
            "--narrate",
            type=int,
            default=0,
            metavar="N",
            help="write briefs for the top N accounts (uses Claude if ANTHROPIC_API_KEY is set)",
        )
        sp.add_argument("--playbook", help="your playbook YAML, layered on the built-in plays (see docs/playbooks.md)")

    sp = sub.add_parser("score", help="score a book of open renewals")
    sp.add_argument("book")
    common(sp, profile_default=None)
    sp.add_argument("--model", help="model.json from `train`; without it the scorecard prior is used")
    outputs(sp)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("demo", help="generate, train and score synthetic data end to end")
    sp.add_argument("--profile", default="ai_usage", choices=list(list_profiles()))
    sp.add_argument("--history", type=int, default=4000)
    sp.add_argument("--book", type=int, default=600)
    sp.add_argument("--as-of")
    sp.add_argument("--sources", help="YAML file of data add-ons (examples/sources.demo.yaml simulates Zendesk)")
    outputs(sp)
    sp.set_defaults(func=cmd_demo, out="demo_out")

    sp = sub.add_parser("evaluate", help="backtest: train on earlier renewals, test on later ones")
    sp.add_argument("history_file", nargs="?", help="renewal history CSV with outcomes")
    common(sp)
    sp.add_argument("--demo", action="store_true", help="backtest every profile on synthetic history")
    sp.add_argument("--history", type=int, default=4000, help="synthetic renewals per profile (with --demo)")
    sp.add_argument("--test-share", type=float, default=0.25, help="latest share of renewals held out (0.25)")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--out", default="evaluation.md")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("playbook", help="check a playbook covers every alert and group")
    sp.add_argument("--playbook", help="your playbook YAML (default: the built-in one)")
    sp.add_argument("--config", help="check against this custom profile instead of every built-in one")
    sp.add_argument("--profile", default=None, help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_playbook)

    sp = sub.add_parser("sources", help="list data add-ons or test their settings")
    sp.add_argument("action", choices=["list", "test"])
    sp.add_argument("--sources", help="YAML file of add-on settings (for test)")
    sp.set_defaults(func=cmd_sources)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (IngestError, ConfigError, SourceError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
