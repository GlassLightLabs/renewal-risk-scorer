"""Typed loading and validation of the scorecard configuration (see config/scorecard.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

OPS = {"<", "<=", ">", ">=", "==", "!=", "in"}
SEVERITIES = ("high", "medium", "low")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Measure:
    name: str
    weight: float
    points: tuple[tuple[float, float], ...] = ()
    if_true: float | None = None
    if_false: float | None = None

    @property
    def is_bool(self) -> bool:
        return self.if_true is not None


@dataclass(frozen=True)
class Group:
    name: str
    label: str
    weight: float
    indicator: str
    measures: tuple[Measure, ...]


@dataclass(frozen=True)
class Alert:
    id: str
    severity: str
    message: str
    when: tuple[tuple[str, str, Any], ...]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    direction: int  # -1: higher value lowers churn risk, 1: raises it, 0: unconstrained
    group: str
    label: str
    unit: str = "number"


# Added to every profile's model: contract size and segment carry risk in every industry.
GENERIC_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("log_arr", 0, "commercial", "Contract size", "money"),
    FeatureSpec("segment_code", 0, "commercial", "Segment", "segment"),
)
UNITS = {
    "ratio",
    "trend",
    "pct",
    "pct1",
    "days",
    "bool",
    "count",
    "number",
    "score10",
    "score5",
    "months",
    "years",
    "money",
    "segment",
}


@dataclass(frozen=True)
class Tier:
    name: str
    min_churn_probability: float
    max_days_to_renewal: float | None = None


@dataclass(frozen=True)
class ScorecardConfig:
    profile: str
    label: str
    description: str
    groups: tuple[Group, ...]
    features: tuple[FeatureSpec, ...]
    green: float
    yellow: float
    confidence_high: float
    confidence_medium: float
    urgency_points: tuple[tuple[float, float], ...]
    tiers: tuple[Tier, ...]
    watermelon_min_churn_probability: float
    watermelon_lift: float = 4.0
    alerts: tuple[Alert, ...] = field(default=())

    @property
    def model_features(self) -> tuple[FeatureSpec, ...]:
        return self.features + GENERIC_FEATURES

    @property
    def measure_groups(self) -> dict[str, str]:
        return {m.name: g.name for g in self.groups for m in g.measures}

    @property
    def group_labels(self) -> dict[str, str]:
        return {g.name: g.label for g in self.groups}


def _points(raw: Any, where: str) -> tuple[tuple[float, float], ...]:
    try:
        pts = tuple((float(x), float(y)) for x, y in raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: points must be a list of [value, score] pairs") from exc
    if len(pts) < 2:
        raise ConfigError(f"{where}: needs at least two breakpoints")
    xs = [p[0] for p in pts]
    if xs != sorted(xs) or len(set(xs)) != len(xs):
        raise ConfigError(f"{where}: breakpoint values must be strictly increasing")
    return pts


def parse_config(data: dict[str, Any]) -> ScorecardConfig:
    groups: list[Group] = []
    for gname, g in (data.get("groups") or {}).items():
        measures: list[Measure] = []
        for mname, m in (g.get("measures") or {}).items():
            where = f"groups.{gname}.measures.{mname}"
            w = float(m.get("weight", 0))
            if w <= 0:
                raise ConfigError(f"{where}: weight must be positive")
            if "points" in m:
                measures.append(Measure(mname, w, points=_points(m["points"], where)))
            elif "if_true" in m and "if_false" in m:
                measures.append(Measure(mname, w, if_true=float(m["if_true"]), if_false=float(m["if_false"])))
            else:
                raise ConfigError(f"{where}: give either points or if_true/if_false")
        if not measures:
            raise ConfigError(f"groups.{gname}: has no measures")
        gw = float(g.get("weight", 0))
        if gw <= 0:
            raise ConfigError(f"groups.{gname}: weight must be positive")
        groups.append(Group(gname, g.get("label", gname), gw, g.get("indicator", "concurrent"), tuple(measures)))
    if not groups:
        raise ConfigError("scorecard has no groups")

    alerts: list[Alert] = []
    for a in data.get("alerts") or []:
        if a.get("severity") not in SEVERITIES:
            raise ConfigError(f"alert {a.get('id')}: severity must be one of {SEVERITIES}")
        conds = []
        for cond in a.get("when") or []:
            if len(cond) != 3 or cond[1] not in OPS:
                raise ConfigError(f"alert {a.get('id')}: each condition is [field, op, value] with op in {OPS}")
            conds.append((str(cond[0]), str(cond[1]), cond[2]))
        alerts.append(Alert(str(a["id"]), a["severity"], str(a.get("message", a["id"])), tuple(conds)))

    features: list[FeatureSpec] = []
    for item in data.get("model_features") or []:
        if len(item) not in (4, 5) or int(item[1]) not in (-1, 0, 1):
            raise ConfigError(f"model_features entry {item}: use [field, direction(-1|0|1), group, label, unit]")
        unit = str(item[4]) if len(item) == 5 else "number"
        if unit not in UNITS:
            raise ConfigError(f"model_features {item[0]}: unknown unit {unit}")
        features.append(FeatureSpec(str(item[0]), int(item[1]), str(item[2]), str(item[3]), unit))
    if not features:
        raise ConfigError("profile defines no model_features")
    names = [f.name for f in features]
    if len(set(names)) != len(names) or set(names) & {g.name for g in GENERIC_FEATURES}:
        raise ConfigError("model_features must be unique and must not repeat log_arr or segment_code")

    tiers_raw = data.get("tiers") or {}
    tiers = tuple(
        Tier(name, float(t["min_churn_probability"]), t.get("max_days_to_renewal")) for name, t in tiers_raw.items()
    )
    bands = data.get("bands") or {}
    conf = data.get("confidence") or {}
    return ScorecardConfig(
        profile=str(data.get("profile", "custom")),
        label=str(data.get("label", data.get("profile", "Custom profile"))),
        description=str(data.get("description", "")),
        groups=tuple(groups),
        features=tuple(features),
        green=float(bands.get("green", 75)),
        yellow=float(bands.get("yellow", 50)),
        confidence_high=float(conf.get("high", 0.8)),
        confidence_medium=float(conf.get("medium", 0.55)),
        urgency_points=_points((data.get("urgency") or {}).get("points", [[0, 1], [365, 0.2]]), "urgency"),
        tiers=tiers,
        watermelon_min_churn_probability=float(data.get("watermelon_min_churn_probability", 0.08)),
        watermelon_lift=float(data.get("watermelon_lift", 4.0)),
        alerts=tuple(alerts),
    )


PROFILE_DIR = "profiles"


def list_profiles() -> dict[str, str]:
    """Built-in industry profiles: {name: label}."""
    out: dict[str, str] = {}
    for entry in sorted(resources.files("renewal_risk").joinpath(PROFILE_DIR).iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".yaml") and not entry.name.startswith("_"):
            data = yaml.safe_load(entry.read_text()) or {}
            out[entry.name[:-5]] = str(data.get("label", entry.name[:-5]))
    return out


def _read_builtin(name: str) -> dict[str, Any]:
    entry = resources.files("renewal_risk").joinpath(PROFILE_DIR, name)
    if not entry.is_file():
        raise ConfigError(f"Unknown profile file {name}. Built-in profiles: {', '.join(list_profiles())}")
    return yaml.safe_load(entry.read_text()) or {}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {**base, **{k: v for k, v in override.items() if k not in {"alerts", "base"}}}
    merged["alerts"] = list(base.get("alerts") or []) + list(override.get("alerts") or [])
    return merged


def load_config(profile: str | None = None, path: str | Path | None = None) -> ScorecardConfig:
    """Load a built-in industry profile by name (default: saas), or a custom profile YAML by path.

    A profile may name a ``base`` file (built-in name such as ``_common.yaml``, or a path relative to the
    profile) whose settings it inherits; alerts from the base and the profile are combined.
    """
    if path is not None:
        p = Path(path)
        data = yaml.safe_load(p.read_text()) or {}
        base_name = data.get("base")
        if base_name:
            local = p.parent / str(base_name)
            base = yaml.safe_load(local.read_text()) if local.is_file() else _read_builtin(str(base_name))
            data = _merge(base or {}, data)
        return parse_config(data)
    name = profile or "saas"
    data = _read_builtin(f"{name}.yaml")
    if data.get("base"):
        data = _merge(_read_builtin(str(data["base"])), data)
    return parse_config(data)
