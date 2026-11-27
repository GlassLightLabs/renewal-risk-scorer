"""Shared helpers for add-ons that read a tool's own CSV export.

Most CS tools let an admin export a report as CSV in a few clicks, with no API token and no IT ticket. These
helpers turn such an export into a tidy table, whatever the tool or the person exporting it called each column,
and match its rows to accounts in the book the same careful way the Zendesk add-on does: by an ID column first,
then by company name, and never by guessing between two records that share a name.

An add-on built on them declares the columns it understands as ``{key: (aliases...)}``. ``read_export`` loads a
file, ``normalize_export`` does the same for a DataFrame (the simulated demo modes build one from the book, so the
demo runs the real parsing code), and ``match_records`` maps each row to an account.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..ingest import parse_bool, parse_number
from ..schema import normalize_header
from .base import ConfigField, SourceError, normalize_name

MATCH_CHOICES = ("auto", "id", "name")

# The settings every export-reading add-on shares. Add-ons copy these into their own ``fields``.
MATCH_FIELD = ConfigField(
    "match_on",
    "Match rows to accounts by",
    required=False,
    default="auto",
    kind="choice",
    choices=MATCH_CHOICES,
    help="auto uses the account ID column when it is filled in, otherwise the company name.",
)


def account_id_field(example: str) -> ConfigField:
    return ConfigField(
        "account_id_column",
        "Column holding your account ID",
        required=False,
        help=f"Only if the export names it something unusual, e.g. {example}. Leave blank to detect it.",
    )


def read_export(
    path: str | Path,
    columns: dict[str, tuple[str, ...]],
    *,
    service: str,
    required: Iterable[str] = (),
    extra_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a CSV export and rename its columns to ``columns``' keys. Raises SourceError a person can act on."""
    p = Path(str(path)).expanduser()
    if not str(path).strip():
        raise SourceError("config", f"{service}: add the path to the export file.")
    if not p.exists():
        raise SourceError("not_found", f"{service}: export file not found: {p}.")
    if p.suffix.lower() not in (".csv", ".txt", ".tsv"):
        raise SourceError("config", f"{service}: save the export as CSV (got {p.suffix or 'no extension'}).")
    try:
        raw = pd.read_csv(p, sep=None, engine="python", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
        what = type(exc).__name__
        raise SourceError("bad_response", f"{service}: could not read {p.name} as CSV ({what}).") from None
    return normalize_export(raw, columns, service=service, required=required, extra_aliases=extra_aliases)


def normalize_export(
    raw: pd.DataFrame,
    columns: dict[str, tuple[str, ...]],
    *,
    service: str,
    required: Iterable[str] = (),
    extra_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rename recognized columns to their keys and drop the rest. Blank cells become missing values.

    Some tools repeat a header for multi-value fields (Jira writes one "Labels" column per label, which pandas
    reads as "Labels", "Labels.1"...). Those repeats are joined into the first column, comma-separated.
    """
    raw = _join_repeats(raw)
    index: dict[str, str] = {}
    for header, key in (extra_aliases or {}).items():
        if str(header).strip():
            index[normalize_header(header)] = key
    for key, aliases in columns.items():
        for alias in (key, *aliases):
            index.setdefault(normalize_header(alias), key)
    rename: dict[str, str] = {}
    for col in raw.columns:
        hit = index.get(normalize_header(col))
        if hit is not None and hit not in rename.values():
            rename[col] = hit
    out = raw[list(rename)].rename(columns=rename).copy()
    for key in out.columns:
        out[key] = out[key].map(lambda v: np.nan if v is None or str(v).strip() == "" else v)
    missing = [k for k in required if k not in out.columns]
    if missing:
        wanted = "; ".join(f"{k} (e.g. {columns[k][0]!r})" if columns.get(k) else k for k in missing)
        found = ", ".join(map(str, list(raw.columns)[:12])) or "none"
        raise SourceError("config", f"{service}: the export has no column for {wanted}. Columns found: {found}.")
    return out.reset_index(drop=True)


def _join_repeats(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c) for c in raw.columns]
    repeats: dict[str, list[str]] = {}
    for c in cols:
        m = re.fullmatch(r"(.+)\.(\d+)", c)
        if m and m.group(1) in cols:
            repeats.setdefault(m.group(1), []).append(c)
    if not repeats:
        return raw
    out = raw.copy()
    for base, extra in repeats.items():
        parts = out[[base, *extra]].astype(str).replace({"nan": "", "None": ""})
        out[base] = parts.apply(lambda r: ",".join(v.strip() for v in r if v.strip()), axis=1)
        out = out.drop(columns=extra)
    return out


def numbers(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype(float)


def flags(series: pd.Series) -> pd.Series:
    """Yes/no columns. Any non-blank text that is not a clear 'no' also counts as yes (e.g. a competitor's name)."""

    def one(v: Any) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        b = parse_bool(v)
        if not np.isnan(b):
            return b
        return 1.0 if str(v).strip() else np.nan

    return series.map(one).astype(float)


def timestamps(series: pd.Series) -> pd.Series:
    """Dates in the shapes tools export them: ISO, '26/Nov/26 3:12 PM' (Jira), '11/05/2026', epoch milliseconds."""

    def one(v: Any) -> Any:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NaT
        s = str(v).strip()
        if not s:
            return pd.NaT
        if s.isdigit() and len(s) >= 11:  # epoch milliseconds, as HubSpot and Pendo APIs send them
            return pd.Timestamp(int(s), unit="ms", tz="UTC")
        try:
            ts = pd.to_datetime(s, utc=True, format="mixed")
        except (ValueError, OverflowError):
            return pd.NaT
        return ts

    return pd.to_datetime(series.map(one), utc=True)


def end_of(as_of: date) -> pd.Timestamp:
    """The instant the as-of day ends, so everything that happened on the as-of date counts."""
    return pd.Timestamp(datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc) + timedelta(days=1))


def days_since(series: pd.Series, as_of: date) -> pd.Series:
    """Calendar days from each date to the as-of date (the same day is 0). Future dates count as 0."""
    day = timestamps(series).dt.tz_convert(None).dt.normalize()
    return (pd.Timestamp(as_of) - day).dt.days.clip(lower=0).astype(float)


def match_records(
    records: pd.DataFrame,
    accounts: pd.DataFrame,
    *,
    mode: str = "auto",
    id_col: str = "account_ref",
    name_col: str = "name",
    service: str = "The add-on",
    one_per_account: bool = True,
) -> tuple[pd.Series, int, list[str]]:
    """Map each row of ``records`` to an account_id.

    Returns a Series (same index as ``records``, missing where unmatched), the number of distinct unmatched
    records and any warnings. Several rows may map to one account (e.g. many issues per customer); in name mode
    two records, or two accounts, that share a normalized name are never matched, because guessing would move
    one customer's signals onto another. With ``one_per_account`` (a list of companies), an account already
    matched by ID is not matched again by name; without it (a list of issues), it can be.
    """
    mode = (mode or "auto").strip().lower()
    if mode not in MATCH_CHOICES:
        choices = ", ".join(MATCH_CHOICES)
        raise SourceError("config", f"{service}: 'Match rows to accounts by' must be one of {choices}.")
    warnings: list[str] = []
    ids = set(accounts["account_id"].astype(str))
    result = pd.Series(np.nan, index=records.index, dtype=object)

    if mode in ("auto", "id") and id_col in records.columns:
        ref = records[id_col].map(lambda v: str(v).strip() if isinstance(v, str) or pd.notna(v) else "")
        ref = ref.map(lambda s: s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s)
        hit = ref.isin(ids)
        result[hit] = ref[hit]

    if mode in ("auto", "name") and name_col in records.columns and "account_name" in accounts.columns:
        taken_by_id = set(result.dropna()) if one_per_account else set()
        acc = accounts.assign(_n=accounts["account_name"].map(normalize_name))
        acc = acc[acc["_n"] != ""]
        dup_accounts = set(acc.loc[acc["_n"].duplicated(keep=False), "_n"])
        by_name = {
            n: str(a)
            for n, a in zip(acc["_n"], acc["account_id"], strict=True)
            if n not in dup_accounts and str(a) not in taken_by_id
        }
        pending = result.isna()
        names = records.loc[pending, name_col].map(normalize_name)
        # Distinct records sharing a name are ambiguous. With an ID column, distinct means distinct IDs; without
        # one (a list of issues), rows with the same customer name are the same customer.
        if id_col in records.columns:
            keyed = pd.DataFrame({"n": names, "ref": records.loc[pending, id_col].astype(str)})
            per_name = keyed[keyed["n"] != ""].groupby("n")["ref"].nunique()
            dup_records = set(per_name[per_name > 1].index)
        else:
            dup_records = set()
        ambiguous = names.isin(dup_records | dup_accounts) & (names != "")
        if ambiguous.any():
            n_amb = int(names[ambiguous].nunique())
            warnings.append(f"{service}: {n_amb} name(s) match more than one record or account, so were not matched.")
        mapped = names[~ambiguous].map(by_name)
        result[mapped.dropna().index] = mapped.dropna()

    unmatched_mask = result.isna()
    if id_col in records.columns:
        fallback = records[name_col] if name_col in records.columns else pd.Series(np.nan, index=records.index)
        key = records[id_col].where(records[id_col].notna(), fallback)
    elif name_col in records.columns:
        key = records[name_col]
    else:
        key = pd.Series(records.index, index=records.index)
    unmatched = int(key[unmatched_mask].dropna().astype(str).nunique())
    return result, unmatched, warnings
