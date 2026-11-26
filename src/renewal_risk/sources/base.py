"""The add-on contract: what a data source declares, what it returns, and how it fails.

A data source (Zendesk, Jira, a product-analytics tool, a billing system...) pulls signals from somewhere other
than the CRM export and hands them back per account, in canonical field names. It never touches the scoring
itself: the pipeline merges what it returns into the book, recomputes derived signals, then scores as usual.

To write one, subclass :class:`DataSource`, declare its settings in ``fields`` and the canonical columns it fills
in ``provides``, and implement :meth:`DataSource.fetch`. See docs/add-ons.md.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar, Literal

import pandas as pd

ErrorKind = Literal["auth", "rate_limit", "network", "config", "not_found", "bad_response"]

# What to tell a person for each kind of failure. Messages never include credentials.
_HINTS: dict[str, str] = {
    "auth": "Check the email and API token, and that the token belongs to an admin.",
    "rate_limit": "The service is limiting requests. Wait a minute and try again.",
    "network": "Check the address and your internet connection.",
    "config": "",
    "not_found": "Check the subdomain or address.",
    "bad_response": "The service sent something unexpected. Try again, and report it if it keeps happening.",
}


class SourceError(Exception):
    """A failure a person can act on. ``kind`` says what went wrong; the message says what to do."""

    def __init__(self, kind: ErrorKind, message: str, *, status: int | None = None):
        self.kind: ErrorKind = kind
        self.status = status
        self.hint = _HINTS.get(kind, "")
        super().__init__(message)

    def __reduce__(self) -> tuple[Any, ...]:
        # Keeps the kind when the error is pickled (Streamlit caches results that can hold errors).
        return (SourceError, (self.kind, Exception.__str__(self)), {"status": self.status})

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} {self.hint}".strip() if self.hint and self.hint not in base else base


@dataclass(frozen=True)
class ConfigField:
    """One setting an add-on needs. The app draws a form from these; secrets get a password box."""

    name: str
    label: str
    help: str = ""
    secret: bool = False
    required: bool = True
    default: Any = None
    kind: Literal["text", "int", "bool", "choice"] = "text"
    choices: tuple[str, ...] = ()


@dataclass
class SourceResult:
    """What an add-on returns: canonical signals per account, plus an honest account of what matched."""

    source: str
    signals: pd.DataFrame  # index: account_id; columns: canonical field names
    matched_accounts: int = 0
    unmatched_accounts: list[str] = field(default_factory=list)
    unmatched_records: int = 0  # records in the source (e.g. Zendesk organizations) that matched no account
    records_read: int = 0  # e.g. tickets read
    warnings: list[str] = field(default_factory=list)
    simulated: bool = False

    def summary(self) -> str:
        total = self.matched_accounts + len(self.unmatched_accounts)
        parts = [f"{self.source}: matched {self.matched_accounts} of {total} accounts"]
        if self.records_read:
            parts.append(f"{self.records_read:,} records read")
        if self.unmatched_records:
            parts.append(f"{self.unmatched_records} source records matched no account")
        line = ", ".join(parts)
        if self.simulated:
            line += " (simulated demo data)"
        return line


class DataSource(ABC):
    """Base class for data-source add-ons."""

    name: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str] = ""
    fields: ClassVar[tuple[ConfigField, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()
    supports_simulation: ClassVar[bool] = False

    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Fill defaults, coerce types and check required settings. Raises SourceError(kind="config")."""
        out: dict[str, Any] = {}
        simulate = _as_bool(config.get("simulate", False))
        if simulate and not self.supports_simulation:
            raise SourceError("config", f"{self.label} has no simulated demo mode.")
        for f in self.fields:
            raw = config.get(f.name, f.default)
            if isinstance(raw, str):
                raw = raw.strip()
            if raw in (None, ""):
                if f.required and not simulate:
                    raise SourceError("config", f"{self.label}: '{f.label}' is required.")
                out[f.name] = f.default
                continue
            if f.kind == "int":
                try:
                    out[f.name] = int(raw)
                except (TypeError, ValueError):
                    raise SourceError("config", f"{self.label}: '{f.label}' must be a whole number.") from None
            elif f.kind == "bool":
                out[f.name] = _as_bool(raw)
            elif f.kind == "choice" and f.choices and str(raw) not in f.choices and not _choice_ok(f, str(raw)):
                raise SourceError("config", f"{self.label}: '{f.label}' must be one of {', '.join(f.choices)}.")
            else:
                out[f.name] = raw
        out["simulate"] = simulate
        return out

    def test_connection(self, config: dict[str, Any]) -> str:
        """A cheap call that proves the settings work. Returns a one-line confirmation."""
        self.validate(config)
        return f"{self.label}: settings look complete."

    @abstractmethod
    def fetch(self, config: dict[str, Any], accounts: pd.DataFrame, as_of: date) -> SourceResult:
        """Return signals for the accounts in ``accounts`` (canonical columns, at least ``account_id``)."""

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "label": cls.label,
            "description": cls.description,
            "provides": list(cls.provides),
            "settings": [f.name for f in cls.fields],
        }


def redact(config: dict[str, Any], fields: tuple[ConfigField, ...]) -> dict[str, Any]:
    """A copy of a config that is safe to print or log."""
    secret = {f.name for f in fields if f.secret}
    return {k: ("••••" if k in secret and v else v) for k, v in config.items()}


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y"}


def _choice_ok(f: ConfigField, raw: str) -> bool:
    # A choice ending in ":" accepts a suffix, e.g. "field:" accepts "field:sfdc_id".
    return any(c.endswith(":") and raw.startswith(c) and len(raw) > len(c) for c in f.choices)


_SUFFIXES = re.compile(r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|gmbh|plc|sa|ag|bv|pty|the)\b")


def normalize_name(name: Any) -> str:
    """Company names as people actually type them: 'Acme, Inc.' and 'ACME' match; suffixes are ignored."""
    s = str(name or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = _SUFFIXES.sub(" ", s)
    return " ".join(s.split())
