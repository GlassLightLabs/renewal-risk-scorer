"""Finding add-ons: the built-in ones, plus any installed package that registers an entry point.

A third-party add-on is an ordinary Python package that declares, in its ``pyproject.toml``::

    [project.entry-points."renewal_risk.sources"]
    jira = "renewal_risk_jira:JiraSource"

Once it is installed (``pip install renewal-risk-jira``), it shows up in ``renewal-risk sources list`` and in
the app's Data add-ons panel with no change to this project. An add-on that fails to load is reported, not
fatal: one broken plug-in must never stop scoring.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from .base import DataSource, SourceError

ENTRY_POINT_GROUP = "renewal_risk.sources"


def _builtins() -> dict[str, type[DataSource]]:
    from .zendesk import ZendeskSource

    return {ZendeskSource.name: ZendeskSource}


def discover() -> tuple[dict[str, type[DataSource]], list[str]]:
    """All available add-ons by name, and a list of problems loading third-party ones."""
    found = dict(_builtins())
    problems: list[str] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            cls = ep.load()
        except Exception as exc:  # a broken plug-in is reported, never fatal
            problems.append(f"Add-on '{ep.name}' could not be loaded: {type(exc).__name__}: {exc}")
            continue
        if not (isinstance(cls, type) and issubclass(cls, DataSource)):
            problems.append(f"Add-on '{ep.name}' is not a DataSource subclass; skipped.")
            continue
        name = getattr(cls, "name", ep.name)
        if name in found and found[name] is not cls:
            problems.append(f"Add-on '{name}' from {ep.value} clashes with an existing add-on; skipped.")
            continue
        found[name] = cls
    return found, problems


def get_source(name: str) -> DataSource:
    sources, _ = discover()
    if name not in sources:
        known = ", ".join(sorted(sources)) or "none"
        raise SourceError("config", f"Unknown add-on '{name}'. Available: {known}.")
    return sources[name]()
