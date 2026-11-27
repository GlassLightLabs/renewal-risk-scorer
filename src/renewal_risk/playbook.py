"""Renewal playbooks: turn each account's risk into the play a CS team would run, sized to its risk tier.

The plays follow common customer success practice (the "success plan" and "save play" motions found in CS
platforms and TSIA's LAER model): fix the specific cause, put the right owner on it, say the right thing on
the next call, know when it worked, and pace it to the renewal window. Plays live in a YAML file
(``playbooks/default.yaml``) that a team can copy and edit, so changing a play never means changing code.

This is the deterministic baseline the optional Claude narrative builds on, so the tool still recommends
something useful with no API key.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError

Play = dict[str, Any]

PLAY_FIELDS = ("name", "owner", "actions", "talk_track", "exit_criteria")
SPECIAL_PLAYS = ("watermelon", "healthy", "default")
TIER_FIELDS = ("motion", "cadence", "escalation")
# Model drivers that describe the account rather than a cause anyone can fix. They never pick a play.
NOT_ACTIONABLE = frozenset({"log_arr", "segment_code"})
MAX_SUPPORTING = 2


@dataclass(frozen=True)
class Playbook:
    alerts: dict[str, Play]
    groups: dict[str, Play]
    special: dict[str, Play]
    tiers: dict[str, dict[str, str]] = field(default_factory=dict)


def _parse_play(raw: Any, where: str) -> Play:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: a play is a mapping with {', '.join(PLAY_FIELDS)}")
    missing = [f for f in PLAY_FIELDS if not raw.get(f)]
    if missing:
        raise ConfigError(f"{where}: missing {', '.join(missing)}")
    unknown = set(raw) - set(PLAY_FIELDS)
    if unknown:
        raise ConfigError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")
    actions = raw["actions"]
    if not isinstance(actions, list) or not all(isinstance(a, str) and a.strip() for a in actions):
        raise ConfigError(f"{where}: actions must be a list of steps")
    play: Play = {f: str(raw[f]).strip() for f in PLAY_FIELDS if f != "actions"}
    play["actions"] = [a.strip() for a in actions]
    return play


def parse_playbook(data: dict[str, Any], base: Playbook | None = None) -> Playbook:
    """Validate a playbook mapping. With ``base``, entries in ``data`` add to or replace the base's."""
    unknown = set(data) - {"alerts", "groups", "special", "tiers"}
    if unknown:
        raise ConfigError(f"playbook: unknown section(s) {', '.join(sorted(unknown))}")
    sections: dict[str, dict[str, Play]] = {}
    for section in ("alerts", "groups", "special"):
        raw = data.get(section) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"playbook.{section}: must map ids to plays")
        merged: dict[str, Play] = dict(getattr(base, section)) if base else {}
        merged |= {str(k): _parse_play(v, f"playbook.{section}.{k}") for k, v in raw.items()}
        sections[section] = merged
    bad_special = set(sections["special"]) - set(SPECIAL_PLAYS)
    if bad_special:
        raise ConfigError(f"playbook.special: unknown play(s) {', '.join(sorted(bad_special))}")
    absent = [s for s in SPECIAL_PLAYS if s not in sections["special"]]
    if absent:
        raise ConfigError(f"playbook.special: missing {', '.join(absent)}")

    tiers: dict[str, dict[str, str]] = {k: dict(v) for k, v in base.tiers.items()} if base else {}
    raw_tiers = data.get("tiers") or {}
    if not isinstance(raw_tiers, dict):
        raise ConfigError("playbook.tiers: must map tier names to settings")
    for name, t in raw_tiers.items():
        if not isinstance(t, dict) or set(t) - set(TIER_FIELDS):
            raise ConfigError(f"playbook.tiers.{name}: use only {', '.join(TIER_FIELDS)}")
        tiers[str(name)] = tiers.get(str(name), {}) | {k: str(v) for k, v in t.items()}
    return Playbook(sections["alerts"], sections["groups"], sections["special"], tiers)


@lru_cache(maxsize=1)
def default_playbook() -> Playbook:
    text = resources.files("renewal_risk").joinpath("playbooks").joinpath("default.yaml").read_text()
    return parse_playbook(yaml.safe_load(text) or {})


def load_playbook(path: str | Path | None = None) -> Playbook:
    """The built-in playbook, or a custom YAML file layered on top of it."""
    if path is None:
        return default_playbook()
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Playbook file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: a playbook is a YAML mapping with alerts, groups, special and tiers")
    return parse_playbook(data, base=default_playbook())


def missing_alert_plays(alert_ids: Iterable[str], playbook: Playbook | None = None) -> list[str]:
    """Alert ids that would fire with no play behind them."""
    pb = playbook or default_playbook()
    return sorted({a for a in alert_ids if a not in pb.alerts})


def timing(days_to_renewal: float | None) -> str:
    if days_to_renewal is None or (isinstance(days_to_renewal, float) and math.isnan(days_to_renewal)):
        return "Renewal date unknown"
    d = float(days_to_renewal)
    if d < 0:
        return "Overdue: escalate this week"
    if d <= 90:
        return "Negotiation window: act this week"
    if d <= 180:
        return "Engagement window: start within two weeks"
    if d <= 270:
        return "Planning window: build the plan this month"
    return "Early: monitor and fix root causes"


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def recommend(
    profile: str,
    drivers: list[dict[str, Any]],
    alerts: list[dict[str, str]],
    days_to_renewal: float | None,
    watermelon: bool = False,
    *,
    tier: str | None = None,
    playbook: Playbook | None = None,
) -> Play:
    """Pick the primary play and up to two supporting plays for one account.

    Order: a high or medium alert with a play; the watermelon check; the healthy renewal play for a
    low-tier account; the play for the top actionable driver's group; the default motion. Leaving out
    ``tier`` skips the healthy-renewal step. ``profile`` is accepted for callers that pass it and does not
    change the choice: plays are keyed by alert and group, which are already profile-specific.
    """
    del profile
    pb = playbook or default_playbook()
    ranked = sorted(
        (a for a in alerts if a.get("id") in pb.alerts),
        key=lambda a: _SEVERITY_ORDER.get(a.get("severity", "low"), 3),
    )
    alert_plays = [(f"alert:{a['id']}", pb.alerts[a["id"]], a.get("severity")) for a in ranked]

    chosen: tuple[str, Play] | None = None
    for pid, play, sev in alert_plays:
        if sev in ("high", "medium"):
            chosen = (pid, play)
            break
    if chosen is None and watermelon:
        chosen = ("special:watermelon", pb.special["watermelon"])
    if chosen is None and tier == "low":
        chosen = ("special:healthy", pb.special["healthy"])
    if chosen is None:
        for d in drivers:
            if d.get("feature") in NOT_ACTIONABLE:
                continue
            if d.get("group") in pb.groups:
                chosen = (f"group:{d['group']}", pb.groups[d["group"]])
                break
    if chosen is None:
        chosen = ("special:default", pb.special["default"])

    pid, play = chosen
    supporting: list[str] = []
    # A watermelon flag is the reason the account needs a second look, so its check leads the supporting plays.
    candidates = [("special:watermelon", pb.special["watermelon"])] if watermelon else []
    candidates += [(c_id, c_play) for c_id, c_play, _ in alert_plays]
    for c_id, c_play in candidates:
        if c_id == pid or c_play["name"] == play["name"] or c_play["name"] in supporting:
            continue
        supporting.append(c_play["name"])
        if len(supporting) == MAX_SUPPORTING:
            break

    tier_info = pb.tiers.get(tier or "", {})
    return {
        "id": pid,
        "play": play["name"],
        "owner": play["owner"],
        "actions": list(play["actions"]),
        "talk_track": play["talk_track"],
        "exit_criteria": play["exit_criteria"],
        "timing": timing(days_to_renewal),
        "motion": tier_info.get("motion", ""),
        "cadence": tier_info.get("cadence", ""),
        "escalation": tier_info.get("escalation", ""),
        "supporting": supporting,
    }


def plays_summary(
    accounts: Any, *, tier_order: Iterable[str] = (), exclude_tiers: tuple[str, ...] = ("low",)
) -> list[dict[str, Any]]:
    """Plays to run across the book: accounts, ARR and expected loss per primary play, largest loss first.

    Low-tier accounts are left out by default: their healthy-renewal motion is business as usual, not work
    to staff.
    """
    df = accounts[~accounts["risk_tier"].isin(exclude_tiers)]
    if df.empty:
        return []
    order = {t: i for i, t in enumerate(tier_order)}
    rows = []
    for (name, owner), part in df.groupby(["save_play", "play_owner"], sort=False):
        counts = part["risk_tier"].value_counts()
        tiers = counts.loc[sorted(counts.index, key=lambda t: order.get(t, len(order)))]
        rows.append(
            {
                "play": name,
                "owner": owner,
                "accounts": int(len(part)),
                "arr": round(float(part["arr"].sum()), 2),
                "expected_arr_loss": round(float(part["expected_arr_loss"].sum()), 2),
                "by_tier": {str(k): int(v) for k, v in tiers.items()},
            }
        )
    return sorted(rows, key=lambda r: r["expected_arr_loss"], reverse=True)
