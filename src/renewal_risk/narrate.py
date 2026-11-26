"""Optional account narratives written by Claude, grounded strictly in the scored facts.

Each at-risk account gets a short brief a CSM or renewals rep can act on: a headline, why it is at risk, a
save plan with owners and dates, and a talk track for the next customer call. Claude only sees the facts the
scorer computed (scores, drivers, alerts, the rule-based play), is told to use nothing else, and returns
JSON that is validated against a schema.

Without an API key, or if a call fails, every account still gets a deterministic narrative built from the
same facts and the rule-based play, so the tool never depends on the API to be useful.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

import pandas as pd

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a senior customer success and renewals strategist at a B2B technology company.
You write short, specific account briefs for CSMs and renewals reps from scored account data.

Rules:
- Use only the facts in the account JSON. Never invent numbers, names, dates, products or events.
- Lead with what matters most for the renewal and what to do about it.
- Tie every risk to a fact from the data (a driver, an alert or a score) and every action to a risk.
- Owners are roles (CSM, Account Executive, Solutions Engineer, Support manager, CS leader), not people.
- Dates are relative to the renewal ("this week", "within 30 days", "before the renewal").
- The talk track is what the rep would actually say on the next call: plain, warm and specific, two or
  three sentences, no jargon, no mention of scores or models.
- If the data shows little risk, say so plainly and keep the plan light.
- Write in plain English without em dashes."""

NARRATIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "why_at_risk": {"type": "array", "items": {"type": "string"}},
        "save_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "owner": {"type": "string"},
                    "when": {"type": "string"},
                },
                "required": ["action", "owner", "when"],
                "additionalProperties": False,
            },
        },
        "talk_track": {"type": "string"},
    },
    "required": ["headline", "why_at_risk", "save_plan", "talk_track"],
    "additionalProperties": False,
}


def account_facts(row: pd.Series) -> dict[str, Any]:
    """The only data the narrative may use."""

    def num(key: str, nd: int = 3) -> Any:
        v = row.get(key)
        return None if v is None or pd.isna(v) else round(float(v), nd)

    return {
        "account": row.get("account_name") or row.get("account_id"),
        "segment": row.get("segment"),
        "arr": num("arr", 0),
        "days_to_renewal": num("days_to_renewal", 0),
        "health_score": num("health_score", 1),
        "health_band": row.get("health_band"),
        "score_confidence": row.get("confidence"),
        "churn_probability": num("churn_probability"),
        "contraction_probability": num("contraction_probability"),
        "expected_arr_loss": num("expected_arr_loss", 0),
        "risk_tier": row.get("risk_tier"),
        "watermelon": bool(row.get("watermelon", False)),
        "risk_drivers": [{"signal": d["label"], "value": d["value"]} for d in row.get("risk_drivers", [])],
        "protective_factors": [{"signal": d["label"], "value": d["value"]} for d in row.get("protective_factors", [])],
        "scorecard_gaps": [{"measure": g["label"], "points_lost": g["points"]} for g in row.get("scorecard_gaps", [])],
        "alerts": [a["message"] for a in row.get("alerts", [])],
        "recommended_play": {
            "name": row.get("save_play"),
            "owner": row.get("play_owner"),
            "timing": row.get("play_timing"),
            "actions": list(row.get("play_actions", [])),
        },
    }


def fallback_narrative(row: pd.Series) -> dict[str, Any]:
    """A deterministic brief from the same facts, used without an API key or when a call fails."""
    f = account_facts(row)
    p = f["churn_probability"] or 0.0
    loss = f["expected_arr_loss"] or 0.0
    days = f["days_to_renewal"]
    when_txt = "overdue" if days is not None and days < 0 else f"renews in {days:.0f} days" if days is not None else ""
    headline = f"{f['account']}: {p:.0%} churn risk, ${loss:,.0f} expected ARR at risk, {when_txt}".rstrip(", ")
    why = [a for a in f["alerts"]][:2] + [f"{d['signal']}: {d['value']}" for d in f["risk_drivers"]]
    if f["watermelon"]:
        why.insert(0, "Scorecard is green, but the model sees risk the scorecard is missing")
    if not why:
        why = ["No material risk drivers in the data"]
    play = f["recommended_play"]
    plan = [{"action": a, "owner": play["owner"] or "CSM", "when": play["timing"] or ""} for a in play["actions"]]
    talk = (
        "I'd like to make sure the next term is set up around what matters most to your team. "
        "Can we find 30 minutes this week to look at how things are going and what you need from us?"
    )
    return {"headline": headline, "why_at_risk": why[:4], "save_plan": plan, "talk_track": talk, "source": "rules"}


def _client() -> Any | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic()


def narrate_one(row: pd.Series, client: Any | None = None, *, model: str = MODEL) -> dict[str, Any]:
    """Claude-written brief for one account, or the rules-based fallback."""
    client = client if client is not None else _client()
    if client is None:
        return fallback_narrative(row)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[
                {
                    "role": "user",
                    "content": "Write the account brief for this account.\n\n"
                    + json.dumps(account_facts(row), default=str),
                }
            ],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": NARRATIVE_SCHEMA}},
        )
        if getattr(response, "stop_reason", None) == "refusal":
            return fallback_narrative(row) | {"note": "model declined; rules-based brief used"}
        text = next(b.text for b in response.content if getattr(b, "type", None) == "text")
        data = json.loads(text)
        return {**data, "source": model}
    except Exception as exc:  # network, auth, rate limit, schema: never break scoring over a narrative
        return fallback_narrative(row) | {"note": f"Claude call failed ({type(exc).__name__}); rules-based brief used"}


def narrate(
    accounts: pd.DataFrame, *, top: int = 10, client: Any | None = None, model: str = MODEL
) -> list[dict[str, Any]]:
    """Briefs for the top ``top`` accounts by priority."""
    client = client if client is not None else _client()
    rows: Iterable[tuple[Any, pd.Series]] = accounts.head(top).iterrows()
    return [{"account_id": r.get("account_id"), **narrate_one(r, client, model=model)} for _, r in rows]


def to_markdown(briefs: list[dict[str, Any]]) -> str:
    out = []
    for b in briefs:
        out.append(f"### {b['headline']}")
        out.append("")
        out.append("**Why it is at risk**")
        out.extend(f"- {w}" for w in b["why_at_risk"])
        out.append("")
        out.append("**Save plan**")
        out.extend(f"1. {s['action']} ({s['owner']}, {s['when']})" for s in b["save_plan"])
        out.append("")
        out.append(f"**Talk track:** {b['talk_track']}")
        out.append(f"\n_Source: {b.get('source', 'rules')}_" + (f" _({b['note']})_" if b.get("note") else ""))
        out.append("")
    return "\n".join(out)
