"""Loading add-on settings and merging what add-ons return into the book.

Settings live in a small YAML file so they can sit next to a mapping file and be reviewed like code. Secrets
never need to be written into it: ``${ZENDESK_API_TOKEN}`` reads an environment variable at run time.

Merge rules, chosen so an add-on can only ever add information:

* An account the add-on matched gets the add-on's values. A matched account with no activity gets real zeros
  (no tickets is a fact). Where the add-on cannot compute a rate (too few ratings, say), any value the CRM
  export already had is kept.
* An account the add-on could not match keeps whatever the CRM export said.
* Derived signals (tickets per 100 seats, support strain...) are recomputed afterwards.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..ingest import rederive
from .base import SourceError, SourceResult
from .registry import get_source

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class SourceRun:
    """The outcome of running every enabled add-on: results that worked, and errors that did not."""

    results: list[SourceResult] = field(default_factory=list)
    errors: dict[str, SourceError] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [r.summary() for r in self.results]
        for r in self.results:
            out += [f"  note: {w}" for w in r.warnings]
        out += [f"{name}: not used. {err}" for name, err in self.errors.items()]
        return out


def interpolate(value: Any, env: dict[str, str] | None = None) -> Any:
    """Replace ${VAR} with the environment variable's value. A missing variable is a config error."""
    env = dict(os.environ) if env is None else env
    if isinstance(value, dict):
        return {k: interpolate(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, env) for v in value]
    if not isinstance(value, str):
        return value

    def sub(m: re.Match[str]) -> str:
        var = m.group(1)
        if var not in env:
            raise SourceError(
                "config", f"Environment variable {var} is not set. Set it before running (export {var}=...)."
            )
        return env[var]

    return _ENV.sub(sub, value)


def load_sources_config(path: str | Path | None, env: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Read a sources YAML file: ``{"sources": {"zendesk": {...}}}``. Returns enabled add-ons only."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SourceError("config", f"Add-on settings file not found: {p}.")
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise SourceError("config", f"{p} is not valid YAML: {exc}.") from None
    sources = raw.get("sources", raw) if isinstance(raw, dict) else None
    if not isinstance(sources, dict):
        raise SourceError("config", f"{p} should contain a 'sources:' mapping of add-on name to settings.")
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in sources.items():
        cfg = cfg or {}
        if not isinstance(cfg, dict):
            raise SourceError("config", f"Settings for '{name}' should be a mapping.")
        if cfg.get("enabled", True) in (False, "false", "no", 0):
            continue
        out[str(name)] = interpolate({k: v for k, v in cfg.items() if k != "enabled"}, env)
    return out


def merge_signals(df: pd.DataFrame, result: SourceResult) -> pd.DataFrame:
    """Write an add-on's per-account signals into the book (matched accounts only)."""
    if result.signals.empty:
        return df
    out = df.copy()
    sig = result.signals.copy()
    sig.index = sig.index.astype(str)
    ids = out["account_id"].astype(str)
    hit = ids.isin(sig.index)
    for col in sig.columns:
        values = ids.map(sig[col]).astype(float)
        if col not in out.columns:
            out[col] = float("nan")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        write = hit & values.notna()
        out.loc[write, col] = values[write]
    return out


def apply_sources(
    df: pd.DataFrame,
    sources: dict[str, dict[str, Any]],
    *,
    as_of: date | None = None,
) -> tuple[pd.DataFrame, SourceRun]:
    """Run each configured add-on and merge its signals. One failing add-on never stops the others."""
    run = SourceRun()
    if not sources:
        return df, run
    as_of = as_of or date.today()
    out = df
    for name, cfg in sources.items():
        try:
            src = get_source(name)
            result = src.fetch(src.validate(cfg), out, as_of)
        except SourceError as err:
            run.errors[name] = err
            continue
        out = merge_signals(out, result)
        run.results.append(result)
    if run.results:
        out = rederive(out, as_of)
    return out, run
