"""Load a CRM or CS-platform export, map it to the canonical schema, clean it, and report on data quality.

The goal is to accept real-world exports without hand-editing: currency strings like "$1.2M", dates in
mixed formats, Yes/No booleans, Gainsight Red/Yellow/Green sentiment, NPS on either the 0-10 response
scale or the -100..100 account scale, and percentages written as 0.42 or 42.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .schema import ALIAS_INDEX, FIELD_BY_NAME, FIELDS, OUTCOME_SYNONYMS, REQUIRED, normalize_header

TRUE_STRINGS = {"true", "t", "yes", "y", "1", "x", "checked", "on"}
FALSE_STRINGS = {"false", "f", "no", "n", "0", "", "unchecked", "off", "none"}
SENTIMENT_WORDS = {
    "red": 1.0,
    "poor": 1.0,
    "at risk": 1.0,
    "yellow": 3.0,
    "amber": 3.0,
    "neutral": 3.0,
    "average": 3.0,
    "green": 5.0,
    "good": 5.0,
    "healthy": 5.0,
}
SEGMENTS = {
    "smb": "SMB",
    "small business": "SMB",
    "small": "SMB",
    "velocity": "SMB",
    "mid market": "Mid-Market",
    "midmarket": "Mid-Market",
    "mm": "Mid-Market",
    "mid": "Mid-Market",
    "commercial": "Mid-Market",
    "enterprise": "Enterprise",
    "ent": "Enterprise",
    "strategic": "Enterprise",
    "large enterprise": "Enterprise",
}


class IngestError(ValueError):
    """Raised when an export cannot be scored at all (for example, a required column is missing)."""


@dataclass
class DataQualityReport:
    rows_in: int = 0
    rows_out: int = 0
    column_map: dict[str, str] = field(default_factory=dict)
    unmapped_columns: list[str] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "column_map": self.column_map,
            "unmapped_columns": self.unmapped_columns,
            "coverage": {k: round(v, 3) for k, v in self.coverage.items()},
            "warnings": self.warnings,
            "dropped": self.dropped,
        }

    def summary(self) -> str:
        lines = [f"Rows: {self.rows_in} in, {self.rows_out} scored"]
        for reason, n in self.dropped.items():
            lines.append(f"  dropped {n}: {reason}")
        lines.append("Columns mapped: " + ", ".join(f"{src} -> {dst}" for src, dst in self.column_map.items()))
        if self.unmapped_columns:
            lines.append("Ignored columns: " + ", ".join(self.unmapped_columns))
        sparse = [f"{k} {v:.0%}" for k, v in self.coverage.items() if 0 < v < 0.6]
        if sparse:
            lines.append("Partly filled columns (missing values are reweighted): " + ", ".join(sparse))
        lines.extend(f"Warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def load_mapping(path: str | Path | None) -> dict[str, str]:
    """Read an optional YAML file of {source column: canonical field} overrides."""
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text()) or {}
    mapping = data.get("columns", data) if isinstance(data, dict) else None
    if not isinstance(mapping, dict):
        raise IngestError("Mapping file must be a YAML mapping of {source column: canonical field}")
    unknown = sorted(v for v in mapping.values() if v not in FIELD_BY_NAME)
    if unknown:
        raise IngestError(f"Mapping file targets unknown fields: {', '.join(unknown)}")
    return {normalize_header(k): v for k, v in mapping.items()}


def map_columns(columns: list[str], overrides: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Return {source column: canonical field} and the list of columns that matched nothing."""
    overrides = {normalize_header(k): v for k, v in (overrides or {}).items()}
    mapped: dict[str, str] = {}
    taken: set[str] = set()
    unmapped: list[str] = []
    for col in columns:
        key = normalize_header(col)
        target = overrides.get(key) or ALIAS_INDEX.get(key)
        if target and target not in taken:
            mapped[col] = target
            taken.add(target)
        else:
            unmapped.append(col)
    return mapped, unmapped


_MULT = {"k": 1e3, "m": 1e6, "b": 1e9}


def parse_number(value: object) -> float:
    """Parse '$1,250.00', '1.2M', '35k', '(500)' and '42%' into floats. Unparseable values become NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip().lower().replace(",", "").replace("$", "").replace("usd", "").replace("%", "").strip()
    if s in {"", "-", "n/a", "na", "null", "none", "nan", "--"}:
        return np.nan
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    m = re.fullmatch(r"(-?\d*\.?\d+)\s*([kmb])?", s)
    if not m:
        return np.nan
    num = float(m.group(1)) * _MULT.get(m.group(2) or "", 1.0)
    return -num if negative else num


def parse_bool(value: object) -> float:
    """Return 1.0, 0.0 or NaN (unknown). Unknown is kept distinct from False on purpose."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    s = str(value).strip().lower()
    if s in TRUE_STRINGS:
        return 1.0
    if s in FALSE_STRINGS and s != "":
        return 0.0
    try:  # 1.0 / 0.0 from spreadsheets and numeric exports
        num = float(s)
    except ValueError:
        return np.nan
    return num if num in (0.0, 1.0) else np.nan


def parse_sentiment(value: object) -> float:
    """CSM sentiment on a 1-5 scale. Accepts numbers or Red/Yellow/Green words."""
    if isinstance(value, str) and value.strip().lower() in SENTIMENT_WORDS:
        return SENTIMENT_WORDS[value.strip().lower()]
    return parse_number(value)


def _as_percent(series: pd.Series, name: str, report: DataQualityReport) -> pd.Series:
    valid = series.dropna()
    if len(valid) and valid.abs().max() <= 1.0 and (valid != 0).any():
        report.warnings.append(f"{name} looked like fractions (max <= 1); converted to percent.")
        return series * 100.0
    return series


def ingest(
    source: str | Path | pd.DataFrame,
    *,
    as_of: date | str | None = None,
    mapping: dict[str, str] | None = None,
    require_outcome: bool = False,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load, map, clean and enrich an export. Returns the canonical frame and a data-quality report."""
    raw = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source, dtype=str, keep_default_na=False)
    report = DataQualityReport(rows_in=len(raw))
    col_map, unmapped = map_columns(list(raw.columns), mapping)
    report.column_map, report.unmapped_columns = col_map, unmapped

    missing = [f for f in REQUIRED if f not in col_map.values()]
    if require_outcome and "outcome" not in col_map.values():
        missing.append("outcome")
    if missing:
        raise IngestError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Rename them, or pass a mapping file (see examples/mapping.yaml)."
        )

    df = raw[list(col_map)].rename(columns=col_map)
    out = pd.DataFrame(index=df.index)
    for name in df.columns:
        f = FIELD_BY_NAME[name]
        col = df[name]
        if f.type == "str" or f.type == "category":
            out[name] = col.astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
        elif f.type == "bool":
            out[name] = col.map(parse_bool).astype(float)
        elif f.type == "date":
            out[name] = pd.to_datetime(col.mask(col.astype(str).str.strip() == ""), errors="coerce", format="mixed")
        elif name == "csm_sentiment":
            out[name] = col.map(parse_sentiment).astype(float)
        else:
            out[name] = col.map(parse_number).astype(float)

    # Drop rows that cannot be scored, and say why.
    def _drop(mask: pd.Series, reason: str) -> None:
        n = int(mask.sum())
        if n:
            report.dropped[reason] = report.dropped.get(reason, 0) + n

    checks = (
        ("missing account_id", lambda d: d["account_id"].isna()),
        ("missing or invalid ARR", lambda d: d["arr"].isna() | (d["arr"] <= 0)),
        ("missing or unparseable renewal date", lambda d: d["renewal_date"].isna()),
    )
    for reason, test in checks:  # each check runs on the rows that survived the previous one
        mask = test(out)
        _drop(mask, reason)
        out = out[~mask]
    dupes = out["account_id"].duplicated(keep="first")
    if dupes.any() and not require_outcome:
        _drop(dupes, "duplicate account_id (kept first row)")
        out = out[~dupes]

    if "outcome" in out.columns:
        normalized = out["outcome"].astype(str).str.strip().str.lower().map(OUTCOME_SYNONYMS)
        bad = normalized.isna()
        if require_outcome:
            _drop(bad, "unrecognized outcome value")
            out, normalized = out[~bad], normalized[~bad]
        out["outcome"] = normalized

    _normalize_scales(out, report)
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(date.today())
    _derive(out, as_of_ts)

    if "account_name" not in out.columns:
        out["account_name"] = out["account_id"]
    out["account_name"] = out["account_name"].fillna(out["account_id"])

    report.rows_out = len(out)
    report.coverage = {
        f.name: float(out[f.name].notna().mean()) if f.name in out.columns and len(out) else 0.0
        for f in FIELDS
        if not f.history_only and not f.required
    }
    if report.rows_out == 0:
        raise IngestError("No scorable rows after cleaning. " + report.summary())
    return out.reset_index(drop=True), report


def _normalize_scales(df: pd.DataFrame, report: DataQualityReport) -> None:
    for pct in (
        "feature_adoption_pct",
        "price_increase_pct",
        "reserved_capacity_utilization_pct",
        "connected_units_pct",
        "supported_firmware_pct",
        "sla_attainment_pct",
        "milestones_on_time_pct",
        "production_usage_pct",
        "deprecated_model_usage_pct",
        "ticket_reopen_rate_pct",
    ):
        if pct in df.columns:
            df[pct] = _as_percent(df[pct], pct, report)
    if "csat" in df.columns:
        valid = df["csat"].dropna()
        if len(valid) and valid.max() <= 5.0:
            df["csat"] = df["csat"] * 20.0  # 1-5 scale to percent
    if "nps" in df.columns:
        valid = df["nps"].dropna()
        if len(valid) and (valid.min() < 0 or valid.max() > 10):
            report.warnings.append("NPS looked like account-level scores (-100..100); mapped to the 0-10 scale.")
            df["nps"] = ((df["nps"].clip(-100, 100) + 100.0) / 20.0).round(1)
    if "csm_sentiment" in df.columns:
        valid = df["csm_sentiment"].dropna()
        if len(valid) and valid.max() > 5:
            df["csm_sentiment"] = (df["csm_sentiment"].clip(0, 100) / 20.0).clip(1, 5)
    if "segment" in df.columns:
        df["segment"] = df["segment"].map(lambda s: SEGMENTS.get(normalize_header(s), s) if isinstance(s, str) else s)
    for count in (
        "licensed_seats",
        "active_users",
        "active_users_prior",
        "tickets_90d",
        "p1_tickets_90d",
        "days_since_last_login",
        "days_since_last_meeting",
        "days_past_due",
        "committed_spend",
        "consumed_to_date",
        "consumption_30d",
        "consumption_30d_prior",
        "installed_units",
        "units_under_contract",
        "units_past_eosl",
        "contracted_hours",
        "hours_used",
    ):
        if count in df.columns:
            neg = df[count] < 0
            if neg.any():
                report.warnings.append(f"{int(neg.sum())} negative values in {count} treated as missing.")
                df.loc[neg, count] = np.nan


def rederive(df: pd.DataFrame, as_of: date | str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Recompute derived signals after columns change (for example after a data add-on fills ticket counts)."""
    out = df.copy()
    _derive(out, pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(date.today()))
    return out


def _derive(df: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Add the derived signals the scorecard and model use."""
    if {"active_users", "licensed_seats"} <= set(df.columns):
        seats = df["licensed_seats"].where(df["licensed_seats"] > 0)
        df["seat_utilization"] = (df["active_users"] / seats).clip(0, 1.5)
    if {"active_users", "active_users_prior"} <= set(df.columns):
        prior = df["active_users_prior"].where(df["active_users_prior"] > 0)
        df["usage_trend_pct"] = ((df["active_users"] - prior) / prior * 100.0).clip(-100, 200)
    if {"tickets_90d", "licensed_seats"} <= set(df.columns):
        seats = df["licensed_seats"].where(df["licensed_seats"] > 0)
        df["tickets_per_100_seats"] = (df["tickets_90d"] / seats * 100.0).clip(0, 500)
    # Each row is measured from its own snapshot date when one is given (history exports usually carry one,
    # because signals are captured some weeks before each renewal), otherwise from the as-of date.
    ref = df["snapshot_date"].fillna(as_of) if "snapshot_date" in df.columns else pd.Series(as_of, index=df.index)
    days_left = (df["renewal_date"] - ref).dt.days
    # Share of the current term elapsed, used to pace commits and hour blocks. Without a term start we assume
    # a 12-month term ending on the renewal date.
    if "term_start_date" in df.columns:
        term_days = (df["renewal_date"] - df["term_start_date"]).dt.days.where(lambda d: d > 0)
        elapsed = ((ref - df["term_start_date"]).dt.days / term_days).fillna(1 - days_left / 365.0)
    else:
        elapsed = 1 - days_left / 365.0
    elapsed = elapsed.clip(0.08, 1.0)
    if {"committed_spend", "consumed_to_date"} <= set(df.columns):
        commit = df["committed_spend"].where(df["committed_spend"] > 0)
        df["commit_pacing_pct"] = (df["consumed_to_date"] / (commit * elapsed) * 100).clip(0, 300)
    if {"consumption_30d", "consumption_30d_prior"} <= set(df.columns):
        prior = df["consumption_30d_prior"].where(df["consumption_30d_prior"] > 0)
        df["consumption_trend_pct"] = ((df["consumption_30d"] - prior) / prior * 100).clip(-100, 300)
    if {"committed_spend", "consumption_30d"} <= set(df.columns):
        commit = df["committed_spend"].where(df["committed_spend"] > 0)
        df["run_rate_vs_commit_pct"] = (df["consumption_30d"] * 12.17 / commit * 100).clip(0, 400)
    if {"committed_spend", "consumed_to_date", "consumption_30d"} <= set(df.columns):
        # Where the commit lands at term end if the last 30 days' run rate holds. Two-sided: well under 100%
        # means a smaller re-commit; well over it means overage, throttling or bill shock before renewal.
        commit = df["committed_spend"].where(df["committed_spend"] > 0)
        remaining = days_left.clip(lower=0) / 30.44 * df["consumption_30d"]
        df["projected_commit_use_pct"] = ((df["consumed_to_date"] + remaining) / commit * 100).clip(0, 400)
    if {"licensed_seats", "licensed_seats_prior"} <= set(df.columns):
        prior = df["licensed_seats_prior"].where(df["licensed_seats_prior"] > 0)
        df["seat_change_pct"] = ((df["licensed_seats"] - prior) / prior * 100).clip(-100, 300)
    if {"seat_change_pct", "consumption_trend_pct"} <= set(df.columns):
        # Seat compression: fewer seats while AI usage grows. The customer is substituting work, not leaving,
        # which is a repricing risk (contraction), not a churn signal.
        known = df["seat_change_pct"].notna() & df["consumption_trend_pct"].notna()
        flag = (df["seat_change_pct"] <= -10) & (df["consumption_trend_pct"] >= 10)
        df["seat_compression"] = flag.astype(float).where(known)
    if {"installed_units", "units_under_contract"} <= set(df.columns):
        units = df["installed_units"].where(df["installed_units"] > 0)
        df["contract_coverage_pct"] = (df["units_under_contract"] / units * 100).clip(0, 100)
    if {"installed_units", "units_past_eosl"} <= set(df.columns):
        units = df["installed_units"].where(df["installed_units"] > 0)
        df["eosl_exposure_pct"] = (df["units_past_eosl"] / units * 100).clip(0, 100)
    if {"contracted_hours", "hours_used"} <= set(df.columns):
        hours = df["contracted_hours"].where(df["contracted_hours"] > 0)
        df["hours_pacing_pct"] = (df["hours_used"] / (hours * elapsed) * 100).clip(0, 300)
    if {"ticket_reopen_rate_pct", "ticket_resolution_hours"} & set(df.columns):
        # Support strain: a quarter or more of tickets reopened, or a median fix time over five days.
        reopen = df.get("ticket_reopen_rate_pct", pd.Series(np.nan, index=df.index))
        hours = df.get("ticket_resolution_hours", pd.Series(np.nan, index=df.index))
        strain = (reopen >= 25) | (hours >= 120)
        df["support_strain_flag"] = strain.astype(float).where(reopen.notna() | hours.notna())
    if {"tickets_90d", "tickets_90d_prior"} <= set(df.columns):
        prior = df["tickets_90d_prior"]
        change = (df["tickets_90d"] - prior) / prior.where(prior > 0) * 100.0
        df["ticket_volume_change_pct"] = change.clip(-100, 300)
        # Support gone quiet: an account that used to file tickets has nearly stopped. Practitioners read a
        # sudden silence as disengagement more often than as a product that suddenly works.
        known = df["tickets_90d"].notna() & prior.notna()
        silent = (prior >= 6) & (df["tickets_90d"] <= prior * 0.3)
        df["support_silence_flag"] = silent.astype(float).where(known)
    if "contract_start_date" in df.columns:
        # Tenure at the renewal being scored, so training history and the current book measure the same thing.
        df["tenure_months"] = ((df["renewal_date"] - df["contract_start_date"]).dt.days / 30.44).clip(lower=0)
    # Drives urgency and some alerts, never the model.
    df["days_to_renewal"] = days_left.astype(float)
