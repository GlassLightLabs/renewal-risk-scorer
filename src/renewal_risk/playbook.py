"""Rule-based save plays: map an account's strongest risk signal to the play a CS team would run.

These follow common customer success practice (the "success plan" and "save play" motions found in CS
platforms and TSIA's LAER model): fix the specific cause, put the right owner on it, and pace it to the
renewal window. They are the deterministic baseline that the optional Claude narrative builds on, so the
tool still recommends something useful with no API key.
"""

from __future__ import annotations

import math
from typing import Any

Play = dict[str, Any]

# Specific, high-signal alerts take precedence over model drivers: they name the cause outright.
ALERT_PLAYS: dict[str, Play] = {
    "champion_left": {
        "play": "Re-establish sponsorship",
        "owner": "CSM + Account Executive",
        "actions": [
            "Identify the new decision maker and book an intro within two weeks",
            "Re-sell the business case with outcomes delivered to date",
            "Multi-thread: add an executive sponsor and one more day-to-day champion",
        ],
    },
    "competitor_eval": {
        "play": "Competitive defense",
        "owner": "Account Executive + CS leader",
        "actions": [
            "Find out what triggered the evaluation and who is running it",
            "Bring a quantified value review and a roadmap briefing to the economic buyer",
            "Prepare a retention offer with a clear walk-away point before negotiation",
        ],
    },
    "payment_risk": {
        "play": "Collections and commercial review",
        "owner": "Finance + Account Executive",
        "actions": [
            "Confirm the invoice reached the right AP contact and is not disputed",
            "Agree a payment plan if cash is the issue; flag the account to leadership",
            "Hold renewal pricing changes until the balance is current",
        ],
    },
    "critical_support": {
        "play": "Technical escalation",
        "owner": "Support manager + CSM",
        "actions": [
            "Open an executive escalation with a named engineering owner",
            "Send the customer a dated remediation plan and run a weekly status call",
            "Hold a post-incident review before the renewal conversation",
        ],
    },
    "support_strain": {
        "play": "Support quality recovery",
        "owner": "Support manager + CSM",
        "actions": [
            "Review the account's reopened and slow tickets with the support lead and find the common cause",
            "Assign a named support contact until reopen rate and fix time are back to normal",
            "Share the fix plan with the customer's admin before the renewal conversation",
        ],
    },
    "support_gone_quiet": {
        "play": "Re-engagement check",
        "owner": "CSM",
        "actions": [
            "Ask the customer's admin directly whether they are still rolling out, or working around problems",
            "Compare which teams stopped using the product with which stopped filing tickets",
            "Book a working session on their next milestone before the renewal conversation starts",
        ],
    },
    "eosl_cliff": {
        "play": "Lifecycle refresh",
        "owner": "Account Executive + Solutions Engineer",
        "actions": [
            "Share the end-of-support timeline and the risk of running unsupported",
            "Propose a refresh or migration path with trade-in credit",
            "Bundle support renewal with the refresh to avoid a coverage gap",
        ],
    },
    "coverage_erosion": {
        "play": "Contract coverage recovery",
        "owner": "Service sales + CSM",
        "actions": [
            "Reconcile the install base against contracts and list uncovered units",
            "Quote coverage for the gap with a co-terming offer",
            "Show the cost of break-fix and third-party maintenance versus coverage",
        ],
    },
    "under_consuming": {
        "play": "Commit recovery",
        "owner": "CSM + Solutions Architect",
        "actions": [
            "Find the workloads planned at signing that never launched and why",
            "Run a workshop to activate one new workload in the next 30 days",
            "Right-size the commit early rather than lose the renewal outright",
        ],
    },
    "consumption_decline": {
        "play": "Consumption recovery",
        "owner": "Solutions Architect",
        "actions": [
            "Diagnose the drop: migration away, optimization, or a project ending",
            "Offer an architecture review and cost optimization that keeps workloads on platform",
            "Identify the next workload to land before the commit renews",
        ],
    },
    "sla_breach": {
        "play": "Delivery recovery",
        "owner": "Delivery lead + Engagement manager",
        "actions": [
            "Publish a root-cause analysis and a corrective plan with dates",
            "Offer service credits where the contract calls for them",
            "Review staffing on the account and add senior oversight",
        ],
    },
    "delivery_lead_change": {
        "play": "Delivery continuity",
        "owner": "Engagement manager",
        "actions": [
            "Introduce the new delivery lead in person with a written handover",
            "Re-confirm milestones and success criteria with the client sponsor",
            "Schedule a 30-day check-in to confirm confidence is back",
        ],
    },
    # AI products
    "stalled_pilot": {
        "play": "Pilot-to-production push",
        "owner": "CSM + Solutions Engineer",
        "actions": [
            "Agree written success criteria and a production go-live date with the pilot sponsor",
            "Ship one production workflow with an integration before the pilot end date",
            "Line up the economic buyer and a core budget line for the conversion",
        ],
    },
    "bill_shock": {
        "play": "Spend governance and re-commit",
        "owner": "Account Executive + CSM",
        "actions": [
            "Walk the customer through usage, overage and throttling before the next invoice lands",
            "Offer a right-sized larger commit or burst capacity at committed rates",
            "Set spend alerts and model-routing guidance so costs stay predictable",
        ],
    },
    "under_burn": {
        "play": "Burn recovery before true-up",
        "owner": "CSM + Solutions Engineer",
        "actions": [
            "Find the planned use cases that never launched and the blocker for each",
            "Launch one new production use case in the next 30 days",
            "Propose a commit rollover or right-size early instead of losing the renewal",
        ],
    },
    "seat_compression": {
        "play": "Seat-to-usage repricing",
        "owner": "Account Executive + Deal desk",
        "actions": [
            "Show the customer the shift: fewer seats, more AI-driven output",
            "Offer a credits or outcome-based plan that prices the work, not the headcount",
            "Protect ARR by trading seats for a usage commit at renewal",
        ],
    },
    "model_retirement": {
        "play": "Model migration plan",
        "owner": "Solutions Engineer",
        "actions": [
            "List the customer's workflows on retiring models and their shutdown dates",
            "Run evals on the replacement model and share the results",
            "Schedule the cutover well before both the shutdown and the renewal",
        ],
    },
    "experimental_budget": {
        "play": "Budget conversion",
        "owner": "Account Executive",
        "actions": [
            "Quantify outcomes delivered so far in the buyer's own metrics",
            "Find the core budget owner and present a business case",
            "Time the renewal proposal to the customer's planning cycle",
        ],
    },
    "usage_decline": {
        "play": "Usage recovery",
        "owner": "CSM + Solutions Engineer",
        "actions": [
            "Find out what changed: a reorg, a cheaper model, or a project ending",
            "Re-engage the team that owned the workload and agree a recovery plan",
            "Land a second use case so one decline cannot sink the renewal",
        ],
    },
}

GROUP_PLAYS: dict[str, Play] = {
    "product_adoption": {
        "play": "Adoption recovery",
        "owner": "CSM",
        "actions": [
            "Find the teams with inactive seats and the reason they stalled",
            "Run enablement for the two highest-value unused features",
            "Agree adoption targets with the customer and review them monthly",
        ],
    },
    "relationship": {
        "play": "Executive re-engagement",
        "owner": "CSM + Account Executive",
        "actions": [
            "Book an executive business review with the economic buyer",
            "Map stakeholders and close gaps in sponsor and champion coverage",
            "Set a meeting cadence and keep it until renewal",
        ],
    },
    "support": {
        "play": "Support experience recovery",
        "owner": "Support manager",
        "actions": [
            "Review open and recent tickets for repeat issues",
            "Assign a named support contact until the backlog clears",
            "Close the loop with the customer on fixes delivered",
        ],
    },
    "sentiment": {
        "play": "Voice-of-customer follow-up",
        "owner": "CSM",
        "actions": [
            "Call the detractors and find the root cause",
            "Agree one visible fix and deliver it before renewal",
            "Re-survey after the fix to confirm sentiment moved",
        ],
    },
    "commercial": {
        "play": "Commercial structuring",
        "owner": "Account Executive + Deal desk",
        "actions": [
            "Model renewal options: term length, price protection, right-sizing",
            "Tie any price increase to delivered value and a roadmap commitment",
            "Get verbal agreement before the formal quote goes out",
        ],
    },
    "consumption": {
        "play": "Consumption recovery",
        "owner": "Solutions Architect",
        "actions": [
            "Review commit burn-down and the gap to plan",
            "Land a new workload or use case before the commit period ends",
            "Discuss commit right-sizing early if the gap cannot close",
        ],
    },
    "capacity": {
        "play": "Capacity optimization",
        "owner": "Solutions Architect",
        "actions": [
            "Find idle reserved capacity and the workloads it was meant for",
            "Move or launch workloads onto the reserved capacity",
            "Adjust the reservation mix at renewal to match real use",
        ],
    },
    "installed_base": {
        "play": "Install base review",
        "owner": "Service sales",
        "actions": [
            "Reconcile installed units against contracts and support status",
            "Quote coverage for uncovered units and a refresh for aging ones",
            "Co-term contracts to simplify the renewal",
        ],
    },
    "product_health": {
        "play": "Fleet health review",
        "owner": "Support engineering",
        "actions": [
            "Review RMA history and firmware levels across the fleet",
            "Schedule firmware upgrades and proactive replacements",
            "Turn on telemetry for unmonitored units",
        ],
    },
    "service": {
        "play": "Service experience recovery",
        "owner": "Support manager",
        "actions": [
            "Review recent cases and time to resolution",
            "Assign a technical account manager until stable",
            "Share a service report with the customer",
        ],
    },
    "delivery": {
        "play": "Delivery recovery",
        "owner": "Engagement manager",
        "actions": [
            "Re-baseline milestones with the client sponsor",
            "Fix staffing gaps and add senior oversight",
            "Report progress weekly until back on plan",
        ],
    },
    "burn": {
        "play": "Burn recovery before true-up",
        "owner": "CSM + Solutions Engineer",
        "actions": [
            "Review projected commit use and the usage trend with the customer",
            "Launch or expand a production use case to close the gap",
            "Discuss rollover or right-sizing early if the gap cannot close",
        ],
    },
    "production": {
        "play": "Embed in production",
        "owner": "CSM + Solutions Engineer",
        "actions": [
            "Move sandbox and pilot usage into production workflows",
            "Add integrations to the customer's core systems",
            "Expand to a second team or use case",
        ],
    },
    "experience": {
        "play": "Client experience recovery",
        "owner": "Engagement manager",
        "actions": [
            "Hold a candid feedback session with the client sponsor",
            "Agree on what success looks like for the next term",
            "Deliver one visible quick win before renewal",
        ],
    },
}

DEFAULT_PLAY: Play = {
    "play": "Standard renewal motion",
    "owner": "CSM",
    "actions": [
        "Confirm renewal intent and decision process",
        "Share a value summary before the quote",
        "Send the renewal quote on the normal schedule",
    ],
}

WATERMELON_PLAY: Play = {
    "play": "Watermelon check",
    "owner": "CSM + CS leader",
    "actions": [
        "Pressure-test the green score: ask the customer directly about renewal plans",
        "Verify usage and sponsor data are current, not stale",
        "Escalate if the call surfaces a risk the scorecard is not seeing",
    ],
}


def timing(days_to_renewal: float) -> str:
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


def recommend(
    profile: str,
    drivers: list[dict[str, Any]],
    alerts: list[dict[str, str]],
    days_to_renewal: float,
    watermelon: bool = False,
) -> Play:
    """Pick one play: the first high-severity alert with a play, else the top driver's group."""
    chosen: Play | None = None
    for sev in ("high", "medium"):
        for a in alerts:
            if a["severity"] == sev and a["id"] in ALERT_PLAYS:
                chosen = ALERT_PLAYS[a["id"]]
                break
        if chosen:
            break
    if chosen is None and drivers:
        for d in drivers:
            if d["group"] in GROUP_PLAYS:
                chosen = GROUP_PLAYS[d["group"]]
                break
    if chosen is None and watermelon:
        chosen = WATERMELON_PLAY
    chosen = chosen or DEFAULT_PLAY
    return {**chosen, "actions": list(chosen["actions"]), "timing": timing(days_to_renewal)}
