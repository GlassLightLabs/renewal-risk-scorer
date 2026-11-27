"""Data-source add-ons: pull signals from tools beyond the CRM export (Zendesk, Jira, HubSpot, ChurnZero, Pendo).

See docs/add-ons.md for how to configure one and how to write your own.
"""

from .apply import SourceRun, apply_sources, interpolate, load_sources_config, merge_signals
from .base import ConfigField, DataSource, SourceError, SourceResult, normalize_name, redact
from .registry import ENTRY_POINT_GROUP, discover, get_source

__all__ = [
    "ENTRY_POINT_GROUP",
    "ConfigField",
    "DataSource",
    "SourceError",
    "SourceResult",
    "SourceRun",
    "apply_sources",
    "discover",
    "get_source",
    "interpolate",
    "load_sources_config",
    "merge_signals",
    "normalize_name",
    "redact",
]
