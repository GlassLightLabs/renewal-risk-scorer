"""Realistic synthetic data for each industry profile: a current book and a history of renewals with outcomes.

All companies and people are fictional. Each profile has a hidden "true" churn process with the kinds of
effects renewals teams see in practice, including interactions and cliffs a linear scorecard misses:

* a champion leaving hurts far more when there is no executive sponsor,
* a price uplift hurts more when the customer is getting little value,
* low value is a cliff, not a slope (seat utilization under ~35%, commit pacing under ~75%, installed-base
  coverage under ~60%, units past end of service life over ~30%, SLA attainment under ~93%),
* first renewals, SMB accounts and competitor evaluations raise risk; multi-year and auto-renew lower it,
* part of the risk is unobservable noise, as in real life, so no model can be perfect.

Signals are correlated through a latent account-health factor, survey fields are sparse (NPS response rates
are low), and overall rates land near published SaaS norms: roughly 10% logo churn and gross revenue
retention around 90% (SaaS Capital reports a median near 91%).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

PROFILES = ("saas", "consumption", "hardware", "services", "ai_usage")

ADJ = [
    "Amber",
    "Atlas",
    "Beacon",
    "Birch",
    "Blue Heron",
    "Bluefin",
    "Boulder",
    "Brightwater",
    "Cedar Ridge",
    "Clearpath",
    "Cobalt",
    "Copperline",
    "Crescent",
    "Driftwood",
    "Ember",
    "Evergreen",
    "Fairhaven",
    "Falcon",
    "Foxglove",
    "Granite",
    "Harbor",
    "Hawthorn",
    "Highland",
    "Ironleaf",
    "Juniper",
    "Keystone",
    "Lakeshore",
    "Lantern",
    "Larkspur",
    "Maple Bay",
    "Meridian",
    "Millbrook",
    "Northgate",
    "Oakmont",
    "Orchard",
    "Pinecrest",
    "Prairie",
    "Quarry Lane",
    "Redwood",
    "Riverbend",
    "Saltmarsh",
    "Sandstone",
    "Silverline",
    "Stonebridge",
    "Summit",
    "Tidewater",
    "Timberline",
    "Union Grove",
    "Westbrook",
    "Willow",
    "Windward",
    "Yellowpine",
]
NOUN = [
    "Analytics",
    "Biotech",
    "Builders",
    "Clinics",
    "Cloudworks",
    "Consulting",
    "Credit Union",
    "Dental Group",
    "Distribution",
    "Energy",
    "Engineering",
    "Foods",
    "Freight",
    "Health",
    "Hospitality",
    "Insurance Partners",
    "Labs",
    "Legal",
    "Logistics",
    "Manufacturing",
    "Media",
    "Medical",
    "Outfitters",
    "Pharma",
    "Properties",
    "Retail",
    "Robotics",
    "Schools",
    "Software",
    "Staffing",
    "Supply",
    "Telecom",
    "Therapeutics",
    "Transit",
    "Travel",
    "Utilities",
    "Ventures",
    "Wealth",
]
CSMS = ["A. Rivera", "J. Okafor", "M. Lindqvist", "S. Patel", "D. Moreau", "K. Nakamura", "R. Haddad", "T. Brennan"]
INDUSTRY_BY_NOUN = {
    "Clinics": "Healthcare",
    "Dental Group": "Healthcare",
    "Health": "Healthcare",
    "Medical": "Healthcare",
    "Biotech": "Life Sciences",
    "Pharma": "Life Sciences",
    "Therapeutics": "Life Sciences",
    "Labs": "Life Sciences",
    "Credit Union": "Financial Services",
    "Insurance Partners": "Financial Services",
    "Wealth": "Financial Services",
    "Freight": "Logistics",
    "Logistics": "Logistics",
    "Distribution": "Logistics",
    "Transit": "Logistics",
    "Supply": "Logistics",
    "Manufacturing": "Manufacturing",
    "Engineering": "Manufacturing",
    "Robotics": "Manufacturing",
    "Builders": "Manufacturing",
    "Retail": "Retail",
    "Outfitters": "Retail",
    "Foods": "Retail",
    "Software": "Technology",
    "Cloudworks": "Technology",
    "Analytics": "Technology",
    "Telecom": "Technology",
    "Media": "Media",
    "Legal": "Professional Services",
    "Consulting": "Professional Services",
    "Staffing": "Professional Services",
    "Schools": "Education",
}
SEG_MIX = {"SMB": 0.45, "Mid-Market": 0.35, "Enterprise": 0.20}
SEG_SIZE = {"SMB": 0.0, "Mid-Market": 1.9, "Enterprise": 3.9}  # log-scale size offset
SEG_TERMS = {
    "SMB": dict(multi=0.10, auto=0.60, sponsor=0.30, qbr=0.20, meet=55),
    "Mid-Market": dict(multi=0.25, auto=0.35, sponsor=0.55, qbr=0.55, meet=35),
    "Enterprise": dict(multi=0.45, auto=0.15, sponsor=0.75, qbr=0.80, meet=22),
}
REGIONS = ["NA", "NA", "NA", "EMEA", "EMEA", "APAC", "LATAM"]
# Annual logo churn targets. AI products churn more: ChartMogul's 2025 AI cohort data shows much weaker gross
# retention than classic SaaS, concentrated in pilots and low-commitment plans.
TARGET_CHURN = {"saas": 0.10, "consumption": 0.10, "hardware": 0.10, "services": 0.10, "ai_usage": 0.16}
SPARSE = {
    "nps": 0.45,
    "csat": 0.15,
    "csm_sentiment": 0.10,
    "qbr_last_180d": 0.05,
    "competitor_eval": 0.30,
    "feature_adoption_pct": 0.08,
    "active_users_prior": 0.04,
    "reserved_capacity_utilization_pct": 0.40,
    "connected_units_pct": 0.20,
    "milestones_on_time_pct": 0.10,
}


def _sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def _names(rng: np.random.Generator, n: int) -> tuple[list[str], list[str]]:
    names: list[str] = []
    industries: list[str] = []
    seen: set[str] = set()
    while len(names) < n:
        a, b = str(rng.choice(ADJ)), str(rng.choice(NOUN))
        base = f"{a} {b}"
        name = base if base not in seen else f"{base} {rng.choice(['Group', 'Co', 'Partners', 'Holdings'])}"
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        industries.append(INDUSTRY_BY_NOUN.get(b, "Other"))
    return names, industries


def _base(rng: np.random.Generator, n: int, profile: str) -> pd.DataFrame:
    """Fields every profile shares: identity, relationship, support, sentiment and commercial terms."""
    segs = rng.choice(list(SEG_MIX), size=n, p=list(SEG_MIX.values()))
    t = {k: np.array([SEG_TERMS[s][k] for s in segs]) for k in ("multi", "auto", "sponsor", "qbr", "meet")}
    h = rng.normal(0, 1, n)
    # A sudden decline event (reorg, consolidation, project ended, budget cut). It moves usage and often takes
    # the champion with it, but surveys and the CSM's read lag behind: this is what makes watermelons.
    shock = rng.random(n) < 0.08
    names, industries = _names(rng, n)
    meet_scale = 2.2 if profile == "hardware" else 1.0  # installed-base accounts meet less often
    d = pd.DataFrame(
        {
            "account_name": names,
            "segment": segs,
            "industry": industries,
            "region": rng.choice(REGIONS, size=n),
            "csm_owner": rng.choice(CSMS, size=n),
            "exec_sponsor": (rng.random(n) < np.clip(t["sponsor"] + 0.1 * h, 0.05, 0.95)).astype(float),
            "champion_change_90d": (rng.random(n) < 0.06 + 0.04 * (h < -0.5) + 0.40 * shock).astype(float),
            "days_since_last_meeting": np.clip(
                np.round(t["meet"] * meet_scale * np.exp(-0.35 * h + rng.normal(0, 0.5, n))), 1, 365
            ),
            "qbr_last_180d": (rng.random(n) < np.clip(t["qbr"] + 0.12 * h, 0.02, 0.98)).astype(float),
            "competitor_eval": (rng.random(n) < 0.03 + 0.07 * (h < -0.7)).astype(float),
            "p1_tickets_90d": rng.poisson(0.25 + 0.7 * (h < -1.0) + 0.4 * (segs == "Enterprise")).astype(float),
            "csat": np.clip(np.round(87 + 5 * h + rng.normal(0, 6, n)), 35, 100),
            "nps": np.clip(np.round(7.3 + 1.4 * h + rng.normal(0, 1.5, n)), 0, 10),
            "csm_sentiment": np.clip(np.round(3.3 + 0.9 * h + rng.normal(0, 0.7, n)), 1, 5),
            "days_past_due": np.where(
                rng.random(n) < 0.06 + 0.06 * (h < -0.8) + 0.04 * (segs == "SMB"),
                np.round(rng.uniform(10, 120, n)),
                0.0,
            ),
            "price_increase_pct": rng.choice([0, 3, 5, 7, 10, 15], size=n, p=[0.25, 0.2, 0.25, 0.15, 0.1, 0.05]).astype(
                float
            ),
            "auto_renew": (rng.random(n) < t["auto"]).astype(float),
            "multi_year": (rng.random(n) < t["multi"]).astype(float),
            "tenure_months": np.round(np.clip(rng.gamma(2.0, 14.0, n), 11, 120)),
        }
    )
    d["_h"] = h
    d["_shock"] = shock
    d["_size"] = np.array([SEG_SIZE[s] for s in segs]) + rng.normal(0, 0.6, n)
    return d


def _saas(rng: np.random.Generator, d: pd.DataFrame) -> None:
    n, h, shock = len(d), d["_h"].to_numpy(), d["_shock"].to_numpy()
    seats = np.maximum(3, np.round(np.exp(3.0 + d["_size"].to_numpy())))
    util = np.clip(_sigmoid(0.55 + 0.9 * h + rng.normal(0, 0.45, n)) - 0.25 * shock, 0.03, 1.15)
    active = np.round(seats * util)
    trend = 7 * h + rng.normal(0, 11, n) - shock * rng.uniform(30, 55, n)
    d["licensed_seats"] = seats
    d["active_users"] = active
    d["active_users_prior"] = np.maximum(1, np.round(active / np.clip(1 + trend / 100, 0.2, 3)))
    d["feature_adoption_pct"] = np.clip(100 * _sigmoid(0.2 + 0.8 * h + rng.normal(0, 0.6, n)), 2, 98).round(0)
    d["days_since_last_login"] = np.clip(
        np.round(np.exp(1.4 - 0.7 * h + rng.normal(0, 0.6, n) + 0.4 * (d["segment"] == "SMB"))), 0, 120
    )
    d["tickets_90d"] = np.round(seats * np.exp(rng.normal(np.log(0.07), 0.6, n)))
    price = d["segment"].map({"SMB": 420, "Mid-Market": 600, "Enterprise": 520}).to_numpy()
    d["arr"] = np.round(seats * price * np.exp(rng.normal(0, 0.18, n)), -2)


def _consumption(rng: np.random.Generator, d: pd.DataFrame) -> None:
    n, h, shock = len(d), d["_h"].to_numpy(), d["_shock"].to_numpy()
    commit = np.round(np.exp(9.6 + d["_size"].to_numpy() + rng.normal(0, 0.3, n)), -3)
    pacing = np.clip(100 * np.exp(rng.normal(-0.02 + 0.17 * h, 0.17, n)) - 28 * shock, 15, 220)
    trend = 9 * h + rng.normal(0, 10, n) - shock * rng.uniform(25, 50, n)
    d["arr"] = commit
    d["committed_spend"] = commit
    d["_pacing"] = pacing
    d["consumption_30d"] = np.round(commit / 12.17 * pacing / 100 * (1 + trend / 250), 0)
    d["consumption_30d_prior"] = np.round(d["consumption_30d"] / np.clip(1 + trend / 100, 0.2, 3), 0)
    lam = np.exp(0.6 + 0.35 * h + 0.25 * d["_size"].to_numpy())
    d["active_workloads"] = (1 + rng.poisson(lam)).astype(float)
    d["reserved_capacity_utilization_pct"] = np.clip(
        100 * _sigmoid(1.0 + 0.8 * h + rng.normal(0, 0.5, n)), 10, 100
    ).round(0)


def _hardware(rng: np.random.Generator, d: pd.DataFrame) -> None:
    n, h, shock = len(d), d["_h"].to_numpy(), d["_shock"].to_numpy()
    units = np.maximum(5, np.round(np.exp(3.4 + d["_size"].to_numpy())))
    coverage = np.clip(100 * _sigmoid(1.3 + 0.7 * h + rng.normal(0, 0.5, n)) - 25 * shock, 5, 100)
    eosl = np.clip(14 - 7 * h + rng.normal(0, 10, n), 0, 90)
    d["installed_units"] = units
    d["units_under_contract"] = np.round(units * coverage / 100)
    d["units_past_eosl"] = np.round(units * eosl / 100)
    d["avg_asset_age_years"] = np.clip(4.0 - 0.5 * h + eosl / 25 + rng.normal(0, 1.0, n), 0.5, 12).round(1)
    d["rma_rate_pct"] = np.clip(np.exp(rng.normal(np.log(1.5) - 0.3 * h, 0.5, n)), 0.1, 15).round(2)
    d["supported_firmware_pct"] = np.clip(100 * _sigmoid(1.2 + 0.8 * h + rng.normal(0, 0.6, n)), 5, 100).round(0)
    d["connected_units_pct"] = np.clip(100 * _sigmoid(0.3 + 0.6 * h + rng.normal(0, 0.8, n)), 0, 100).round(0)
    fee = d["segment"].map({"SMB": 260, "Mid-Market": 220, "Enterprise": 180}).to_numpy()
    d["arr"] = np.maximum(1000, np.round(d["units_under_contract"] * fee * np.exp(rng.normal(0, 0.2, n)), -2))


def _services(rng: np.random.Generator, d: pd.DataFrame) -> None:
    n, h, shock = len(d), d["_h"].to_numpy(), d["_shock"].to_numpy()
    hours = np.round(np.exp(5.7 + d["_size"].to_numpy() * 0.85 + rng.normal(0, 0.3, n)), -1)
    d["contracted_hours"] = hours
    d["_hpace"] = np.clip(100 * np.exp(rng.normal(-0.06 + 0.2 * h, 0.22, n)) - 30 * shock, 10, 200)
    d["sla_attainment_pct"] = np.clip(97.4 + 1.3 * h + rng.normal(0, 1.2, n) - 4 * shock, 80, 100).round(1)
    d["milestones_on_time_pct"] = np.clip(82 + 9 * h + rng.normal(0, 10, n), 20, 100).round(0)
    d["delivery_lead_change_90d"] = (rng.random(n) < 0.10 + 0.06 * (h < -0.5)).astype(float)
    rate = d["segment"].map({"SMB": 175, "Mid-Market": 210, "Enterprise": 245}).to_numpy()
    d["arr"] = np.round(hours * rate * np.exp(rng.normal(0, 0.1, n)), -2)


def _ai_usage(rng: np.random.Generator, d: pd.DataFrame) -> None:
    """AI product on a hybrid plan: a seat platform fee plus a prepaid credit (token) commit."""
    n, h, shock = len(d), d["_h"].to_numpy(), d["_shock"].to_numpy()
    size = d["_size"].to_numpy()
    smb = (d["segment"] == "SMB").to_numpy()
    pilot = rng.random(n) < np.clip(0.16 + 0.10 * smb - 0.06 * (d["segment"] == "Enterprise").to_numpy(), 0, 1)
    d["is_pilot"] = pilot.astype(float)
    d["innovation_budget"] = (rng.random(n) < np.where(pilot, 0.65, 0.12)).astype(float)
    d["production_usage_pct"] = np.clip(
        100 * _sigmoid(np.where(pilot, -1.2, 1.4) + 0.6 * h + rng.normal(0, 0.7, n)), 0, 100
    ).round(0)
    seats = np.maximum(3, np.round(np.exp(2.6 + size)))
    # Seat compression: in some accounts AI is replacing seat-based work, so seats fall while usage grows.
    compress = (rng.random(n) < 0.14) & ~pilot
    seat_change = np.where(compress, rng.uniform(-40, -12, n), rng.normal(4 + 4 * h, 8, n))
    d["licensed_seats"] = seats
    d["licensed_seats_prior"] = np.maximum(3, np.round(seats / np.clip(1 + seat_change / 100, 0.3, 3)))
    util = np.clip(_sigmoid(0.4 + 0.8 * h + rng.normal(0, 0.5, n)) - 0.25 * shock, 0.03, 1.1)
    d["active_users"] = np.round(seats * util)
    commit = np.round(np.exp(9.2 + size + rng.normal(0, 0.35, n)), -3)
    # Burn pacing is two-sided: under-burn means a smaller re-commit; heavy over-burn means overage and
    # throttling, which in AI products is a churn trigger of its own (bill shock, rate limits).
    over = rng.random(n) < 0.12
    pacing = np.where(
        over,
        rng.uniform(130, 210, n),
        np.clip(100 * np.exp(rng.normal(-0.05 + 0.2 * h, 0.2, n)) - 30 * shock - 25 * pilot, 10, 160),
    )
    trend = 10 * h + rng.normal(0, 12, n) - shock * rng.uniform(25, 55, n) + 25 * compress
    d["committed_spend"] = commit
    d["_pacing"] = pacing
    d["consumption_30d"] = np.round(commit / 12.17 * pacing / 100 * (1 + trend / 250), 0)
    d["consumption_30d_prior"] = np.round(d["consumption_30d"] / np.clip(1 + trend / 100, 0.2, 3), 0)
    d["throttle_events_90d"] = rng.poisson(np.where(pacing > 125, 6.0, 0.3)).astype(float)
    d["active_workloads"] = (1 + rng.poisson(np.exp(0.4 + 0.35 * h + 0.2 * size - 0.6 * pilot))).astype(float)
    d["active_integrations"] = rng.poisson(np.exp(0.6 + 0.4 * h + 0.2 * size - 0.7 * pilot)).astype(float)
    exposed = rng.random(n) < 0.28
    d["deprecated_model_usage_pct"] = np.where(exposed, rng.uniform(5, 85, n), 0.0).round(0)
    d["arr"] = np.round(commit + seats * 360 * np.exp(rng.normal(0, 0.15, n)), -2)


SIGNALS = {
    "saas": _saas,
    "consumption": _consumption,
    "hardware": _hardware,
    "services": _services,
    "ai_usage": _ai_usage,
}


def _low_value(d: pd.DataFrame, profile: str) -> np.ndarray:
    if profile == "saas":
        return np.asarray(d["active_users"] / d["licensed_seats"] < 0.5)
    if profile == "consumption":
        return np.asarray(d["_pacing"] < 80)
    if profile == "hardware":
        return np.asarray(d["units_under_contract"] / d["installed_units"] < 0.7)
    if profile == "ai_usage":
        return np.asarray((d["_pacing"] < 75) | (d["production_usage_pct"] < 40))
    return np.asarray(d["_hpace"] < 70)


def true_churn_logit(d: pd.DataFrame, profile: str = "saas") -> np.ndarray:
    """The hidden data-generating process. Not used by the scorer; it exists so the demo has ground truth."""
    a = {
        c: d[c].to_numpy(dtype=float)
        for c in d.columns
        if c not in {"segment", "account_name", "industry", "region", "csm_owner", "_shock"}
    }
    no_sponsor = 1 - a["exec_sponsor"]
    z = (
        a["champion_change_90d"] * (0.5 + 1.5 * no_sponsor)
        - 0.45 * a["exec_sponsor"]
        + 0.010 * np.maximum(0, a["days_since_last_meeting"] - (60 if profile == "hardware" else 30))
        - 0.30 * a["qbr_last_180d"]
        + 1.2 * a["competitor_eval"]
        + 0.30 * a["p1_tickets_90d"]
        - 0.008 * (a["csat"] - 85)  # survey scores are weak predictors next to behaviour
        - 0.06 * (a["nps"] - 7)
        - 0.20 * (a["csm_sentiment"] - 3)
        + 0.018 * a["days_past_due"]
        + 0.05 * a["price_increase_pct"] * (1 + 1.5 * _low_value(d, profile))
        - 0.35 * a["auto_renew"]
        + 0.70 * (a["tenure_months"] < 14)
        + 0.45 * (d["segment"].to_numpy() == "SMB")
        - 0.35 * (d["segment"].to_numpy() == "Enterprise")
    )
    if profile == "saas":
        util = np.clip(a["active_users"] / a["licensed_seats"], 0, 1.2)
        trend = np.clip((a["active_users"] - a["active_users_prior"]) / a["active_users_prior"] * 100, -100, 200)
        z += (
            -1.8 * (util - 0.65)
            + 4.0 * np.maximum(0, 0.35 - util)
            - 0.018 * trend
            + 0.035 * np.maximum(0, -20 - trend)
            - 0.012 * (a["feature_adoption_pct"] - 50)
            + 0.012 * a["days_since_last_login"]
        )
    elif profile == "consumption":
        trend = np.clip(
            (a["consumption_30d"] - a["consumption_30d_prior"]) / a["consumption_30d_prior"] * 100, -100, 300
        )
        z += (
            -0.03 * (a["_pacing"] - 90)
            + 0.05 * np.maximum(0, 75 - a["_pacing"])
            - 0.015 * trend
            + 0.03 * np.maximum(0, -20 - trend)
            - 0.3 * np.log1p(a["active_workloads"])
            + 0.7 * (a["active_workloads"] <= 1)
            - 0.012 * (a["reserved_capacity_utilization_pct"] - 70)
        )
    elif profile == "hardware":
        cov = a["units_under_contract"] / a["installed_units"] * 100
        eosl = a["units_past_eosl"] / a["installed_units"] * 100
        z += (
            -0.03 * (cov - 75)
            + 0.05 * np.maximum(0, 60 - cov)
            + 0.03 * eosl
            + 0.05 * np.maximum(0, eosl - 30)
            + 0.12 * (a["avg_asset_age_years"] - 4)
            + 0.18 * a["rma_rate_pct"]
            - 0.012 * (a["supported_firmware_pct"] - 75)
            - 0.008 * (a["connected_units_pct"] - 60)
            + 0.6 * a["competitor_eval"]
        )  # third-party maintenance is a real threat in hardware
    elif profile == "services":
        z += (
            -0.25 * (a["sla_attainment_pct"] - 96)
            + 0.35 * np.maximum(0, 93 - a["sla_attainment_pct"])
            - 0.025 * (a["milestones_on_time_pct"] - 80)
            - 0.02 * (a["_hpace"] - 85)
            + 0.035 * np.maximum(0, 60 - a["_hpace"])
            + 0.6 * a["delivery_lead_change_90d"]
        )
    elif profile == "ai_usage":
        trend = np.clip(
            (a["consumption_30d"] - a["consumption_30d_prior"]) / a["consumption_30d_prior"] * 100, -100, 300
        )
        util = np.clip(a["active_users"] / a["licensed_seats"], 0, 1.2)
        pacing = a["_pacing"]
        stalled_pilot = a["is_pilot"] * (a["production_usage_pct"] < 40)
        z += (
            -0.025 * (np.minimum(pacing, 110) - 90)
            + 0.05 * np.maximum(0, 70 - pacing)
            + 0.02 * np.maximum(0, pacing - 130)
            + 0.25 * np.minimum(a["throttle_events_90d"], 10)  # bill shock
            - 0.012 * trend
            + 0.03 * np.maximum(0, -20 - trend)
            + 1.6 * stalled_pilot
            + 0.4 * a["is_pilot"]
            + 0.6 * a["innovation_budget"]
            - 0.012 * (a["production_usage_pct"] - 60)
            - 0.35 * np.log1p(a["active_integrations"])
            - 0.25 * np.log1p(a["active_workloads"])
            + 0.012 * a["deprecated_model_usage_pct"] * (1 + (a["active_integrations"] <= 1))
            - 1.0 * (util - 0.6)
            + 0.6 * a["competitor_eval"]
        )  # switching costs between model providers are low
    else:
        raise ValueError(f"Unknown profile {profile}")
    # Interactions a weighted scorecard cannot express: a multi-year contract mutes every other signal at this
    # renewal, and a sudden decline event (reorg, budget cut, project ended) adds risk beyond what trends show.
    z = z * (1 - 0.45 * a["multi_year"]) - 0.5 * a["multi_year"]
    z = z + 0.9 * d["_shock"].to_numpy(dtype=float)
    return np.asarray(z, dtype=float)


def _contraction_logit(d: pd.DataFrame, profile: str, rng: np.random.Generator) -> np.ndarray:
    n = len(d)
    if profile == "saas":
        value = (d["active_users"] / d["licensed_seats"]).to_numpy() - 0.6
        z = -2.1 - 3.0 * value
    elif profile == "consumption":
        z = -1.9 - 0.045 * (d["_pacing"].to_numpy() - 90)
    elif profile == "hardware":
        z = (
            -2.0
            - 0.04 * ((d["units_under_contract"] / d["installed_units"]).to_numpy() * 100 - 75)
            + 0.03 * (d["units_past_eosl"] / d["installed_units"]).to_numpy() * 100
        )
    elif profile == "ai_usage":
        seat_change = (d["licensed_seats"] / d["licensed_seats_prior"] - 1).to_numpy() * 100
        trend = ((d["consumption_30d"] - d["consumption_30d_prior"]) / d["consumption_30d_prior"]).to_numpy() * 100
        compression = ((seat_change <= -10) & (trend >= 10)).astype(float)
        z = -2.0 - 0.04 * (np.minimum(d["_pacing"].to_numpy(), 110) - 90) + 1.5 * compression
    else:
        z = -2.0 - 0.04 * (d["_hpace"].to_numpy() - 85)
    z = z + 0.06 * d["price_increase_pct"].to_numpy() + 0.6 * d["champion_change_90d"].to_numpy()
    return np.asarray(z + rng.normal(0, 0.5, n), dtype=float)


def _intercept(z: np.ndarray, target: float) -> float:
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _finish(rng: np.random.Generator, d: pd.DataFrame, renewal: pd.Series, snapshot: pd.Series) -> pd.DataFrame:
    """Set dates, turn hidden pacing into observable consumption and hours, mask sparse fields, drop helpers."""
    d = d.copy()
    elapsed = np.clip(1 - (renewal - snapshot).dt.days.to_numpy() / 365.0, 0.08, 1.0)
    if "_pacing" in d.columns:
        d["consumed_to_date"] = np.round(d["committed_spend"] * elapsed * d["_pacing"] / 100, 0)
    if "_hpace" in d.columns:
        d["hours_used"] = np.round(d["contracted_hours"] * elapsed * d["_hpace"] / 100, 0)
    d["renewal_date"] = renewal.dt.date
    d["contract_start_date"] = (renewal - pd.to_timedelta(d["tenure_months"] * 30.44, unit="D")).dt.date
    for col, rate in SPARSE.items():
        if col in d.columns:
            d.loc[rng.random(len(d)) < rate, col] = np.nan
    return d.drop(columns=[c for c in d.columns if c.startswith("_")] + ["tenure_months"])


def generate_history(n: int = 4000, *, profile: str = "saas", as_of: date | None = None, seed: int = 7) -> pd.DataFrame:
    """Past renewals: a signal snapshot taken 60-120 days before each renewal, plus the outcome."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile}; choose from {PROFILES}")
    rng = np.random.default_rng(seed)
    as_of = as_of or date.today()
    d = _base(rng, n, profile)
    SIGNALS[profile](rng, d)
    z = true_churn_logit(d, profile) + rng.normal(0, 1.1, n)  # unobserved causes: M&A, budget, politics
    z += _intercept(z, TARGET_CHURN[profile])
    churned = rng.random(n) < _sigmoid(z)
    contracted = (~churned) & (rng.random(n) < _sigmoid(_contraction_logit(d, profile, rng)))
    depth = rng.uniform(0.10, 0.45, n)
    grow = (rng.random(n) < 0.40) * rng.uniform(0.05, 0.40, n)  # expansion keeps NRR near the ~100% median
    arr = d["arr"].to_numpy()
    renewed = np.where(
        churned,
        0.0,
        np.where(contracted, arr * (1 - depth), arr * (1 + d["price_increase_pct"].to_numpy() / 100 + grow)),
    )
    d["outcome"] = np.where(churned, "churned", np.where(contracted, "contracted", "renewed"))
    d["renewed_arr"] = np.round(renewed, -2)
    renewal = pd.Series(pd.to_datetime(as_of) - pd.to_timedelta(rng.integers(30, 3 * 365, n), unit="D"))
    snapshot = renewal - pd.to_timedelta(rng.integers(60, 121, n), unit="D")
    d = _finish(rng, d, renewal, snapshot)
    d.insert(0, "account_id", [f"H-{i:05d}" for i in range(1, n + 1)])
    d["snapshot_date"] = snapshot.dt.date
    return d


def generate_book(
    n: int = 600, *, profile: str = "saas", as_of: date | None = None, seed: int = 11, plant_stories: bool = True
) -> pd.DataFrame:
    """The current book: open renewals over the next 12 months (a few already overdue), no outcomes."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile}; choose from {PROFILES}")
    rng = np.random.default_rng(seed)
    as_of = as_of or date.today()
    d = _base(rng, n, profile)
    SIGNALS[profile](rng, d)
    renewal = pd.Series(pd.to_datetime(as_of) + pd.to_timedelta(rng.integers(-20, 366, n), unit="D"))
    d = _finish(rng, d, renewal, pd.Series(pd.to_datetime(as_of), index=d.index))
    d.insert(0, "account_id", [f"A-{i:05d}" for i in range(1, n + 1)])
    if plant_stories:
        stories = _stories(profile, as_of)
        d = pd.concat([stories, d.iloc[: n - len(stories)]], ignore_index=True)
    return d


# Five planted accounts give each demo concrete stories to find. Shared fields first, then each profile's
# own signals. Values are fixed so the README can point at them.
#   A-90001 Halcyon Freight      the watermelon: green scorecard, but the champion left with no exec sponsor
#                                and the leading usage signal is sliding into a 12% uplift
#   A-90002 Brightwater Clinics  plainly red: low value, P1s, detractor, late invoice, renews in 40 days
#   A-90003 Quarry Lane Software overdue and dark: no meeting in four months, first renewal
#   A-90004 Ironleaf Manufacturing healthy control: strong on every signal, multi-year
#   A-90005 Copperline Media     healthy usage, but an invoice 78 days past due
# In the AI profile the same accounts tell AI stories: Halcyon's usage drops after a reorg took its champion
# and 45% of its traffic sits on a retiring model; Brightwater is over-burning with constant rate-limit hits
# (bill shock); Quarry Lane is a stalled pilot on an experimental budget; Copperline cut seats by a third
# while usage grew (seat compression: a repricing talk, not churn).
STORY_BASE: list[dict[str, Any]] = [
    dict(
        account_id="A-90001",
        account_name="Halcyon Freight",
        segment="Enterprise",
        industry="Logistics",
        region="NA",
        csm_owner="J. Okafor",
        p1_tickets_90d=0,
        csat=94,
        nps=9,
        csm_sentiment=5,
        exec_sponsor=1,
        champion_change_90d=1,
        days_since_last_meeting=12,
        qbr_last_180d=1,
        competitor_eval=None,
        days_past_due=0,
        price_increase_pct=12,
        auto_renew=0,
        multi_year=0,
        renew_in=75,
        tenure_months=38,
    ),
    dict(
        account_id="A-90002",
        account_name="Brightwater Clinics",
        segment="Mid-Market",
        industry="Healthcare",
        region="NA",
        csm_owner="S. Patel",
        p1_tickets_90d=4,
        csat=68,
        nps=3,
        csm_sentiment=2,
        exec_sponsor=1,
        champion_change_90d=0,
        days_since_last_meeting=40,
        qbr_last_180d=0,
        competitor_eval=1,
        days_past_due=45,
        price_increase_pct=7,
        auto_renew=0,
        multi_year=0,
        renew_in=40,
        tenure_months=26,
    ),
    dict(
        account_id="A-90003",
        account_name="Quarry Lane Software",
        segment="SMB",
        industry="Technology",
        region="EMEA",
        csm_owner="M. Lindqvist",
        p1_tickets_90d=0,
        csat=None,
        nps=None,
        csm_sentiment=2,
        exec_sponsor=0,
        champion_change_90d=0,
        days_since_last_meeting=124,
        qbr_last_180d=0,
        competitor_eval=None,
        days_past_due=0,
        price_increase_pct=5,
        auto_renew=0,
        multi_year=0,
        renew_in=-10,
        tenure_months=12,
    ),
    dict(
        account_id="A-90004",
        account_name="Ironleaf Manufacturing",
        segment="Enterprise",
        industry="Manufacturing",
        region="NA",
        csm_owner="A. Rivera",
        p1_tickets_90d=0,
        csat=95,
        nps=10,
        csm_sentiment=5,
        exec_sponsor=1,
        champion_change_90d=0,
        days_since_last_meeting=9,
        qbr_last_180d=1,
        competitor_eval=0,
        days_past_due=0,
        price_increase_pct=5,
        auto_renew=0,
        multi_year=1,
        renew_in=300,
        tenure_months=64,
    ),
    dict(
        account_id="A-90005",
        account_name="Copperline Media",
        segment="Mid-Market",
        industry="Media",
        region="APAC",
        csm_owner="K. Nakamura",
        p1_tickets_90d=0,
        csat=90,
        nps=8,
        csm_sentiment=3,
        exec_sponsor=1,
        champion_change_90d=0,
        days_since_last_meeting=28,
        qbr_last_180d=1,
        competitor_eval=0,
        days_past_due=78,
        price_increase_pct=5,
        auto_renew=0,
        multi_year=0,
        renew_in=150,
        tenure_months=30,
    ),
]
STORY_SIGNALS: dict[str, list[dict[str, Any]]] = {
    "saas": [
        dict(
            arr=1_240_000,
            licensed_seats=2200,
            active_users=1540,
            active_users_prior=2370,
            feature_adoption_pct=72,
            days_since_last_login=1,
            tickets_90d=90,
        ),
        dict(
            arr=243_000,
            licensed_seats=400,
            active_users=124,
            active_users_prior=150,
            feature_adoption_pct=28,
            days_since_last_login=6,
            tickets_90d=71,
        ),
        dict(
            arr=38_400,
            licensed_seats=80,
            active_users=31,
            active_users_prior=44,
            feature_adoption_pct=35,
            days_since_last_login=45,
            tickets_90d=2,
        ),
        dict(
            arr=910_000,
            licensed_seats=1600,
            active_users=1440,
            active_users_prior=1380,
            feature_adoption_pct=81,
            days_since_last_login=0,
            tickets_90d=60,
        ),
        dict(
            arr=186_000,
            licensed_seats=310,
            active_users=248,
            active_users_prior=240,
            feature_adoption_pct=70,
            days_since_last_login=1,
            tickets_90d=14,
        ),
    ],
    "consumption": [
        dict(
            arr=1_240_000,
            committed_spend=1_240_000,
            pacing=92,
            consumption_30d=66_000,
            consumption_30d_prior=104_000,
            active_workloads=3,
            reserved_capacity_utilization_pct=78,
        ),
        dict(
            arr=243_000,
            committed_spend=243_000,
            pacing=58,
            consumption_30d=11_000,
            consumption_30d_prior=13_500,
            active_workloads=1,
            reserved_capacity_utilization_pct=41,
        ),
        dict(
            arr=38_400,
            committed_spend=38_400,
            pacing=66,
            consumption_30d=1_900,
            consumption_30d_prior=2_600,
            active_workloads=1,
            reserved_capacity_utilization_pct=None,
        ),
        dict(
            arr=910_000,
            committed_spend=910_000,
            pacing=112,
            consumption_30d=88_000,
            consumption_30d_prior=80_000,
            active_workloads=11,
            reserved_capacity_utilization_pct=91,
        ),
        dict(
            arr=186_000,
            committed_spend=186_000,
            pacing=101,
            consumption_30d=15_800,
            consumption_30d_prior=15_200,
            active_workloads=5,
            reserved_capacity_utilization_pct=83,
        ),
    ],
    "hardware": [
        dict(
            arr=1_240_000,
            installed_units=7400,
            units_under_contract=6600,
            units_past_eosl=2300,
            avg_asset_age_years=5.9,
            rma_rate_pct=1.6,
            supported_firmware_pct=71,
            connected_units_pct=66,
        ),
        dict(
            arr=243_000,
            installed_units=1900,
            units_under_contract=1050,
            units_past_eosl=520,
            avg_asset_age_years=6.4,
            rma_rate_pct=4.8,
            supported_firmware_pct=48,
            connected_units_pct=30,
        ),
        dict(
            arr=38_400,
            installed_units=210,
            units_under_contract=140,
            units_past_eosl=70,
            avg_asset_age_years=6.8,
            rma_rate_pct=2.2,
            supported_firmware_pct=55,
            connected_units_pct=None,
        ),
        dict(
            arr=910_000,
            installed_units=4300,
            units_under_contract=4200,
            units_past_eosl=120,
            avg_asset_age_years=2.6,
            rma_rate_pct=0.6,
            supported_firmware_pct=97,
            connected_units_pct=94,
        ),
        dict(
            arr=186_000,
            installed_units=950,
            units_under_contract=880,
            units_past_eosl=60,
            avg_asset_age_years=3.4,
            rma_rate_pct=1.1,
            supported_firmware_pct=88,
            connected_units_pct=81,
        ),
    ],
    "services": [
        dict(
            arr=1_240_000,
            contracted_hours=5200,
            hpace=78,
            sla_attainment_pct=97.8,
            milestones_on_time_pct=84,
            delivery_lead_change_90d=1,
        ),
        dict(
            arr=243_000,
            contracted_hours=1150,
            hpace=52,
            sla_attainment_pct=90.5,
            milestones_on_time_pct=58,
            delivery_lead_change_90d=0,
        ),
        dict(
            arr=38_400,
            contracted_hours=210,
            hpace=48,
            sla_attainment_pct=95.0,
            milestones_on_time_pct=None,
            delivery_lead_change_90d=0,
        ),
        dict(
            arr=910_000,
            contracted_hours=3900,
            hpace=97,
            sla_attainment_pct=99.6,
            milestones_on_time_pct=96,
            delivery_lead_change_90d=0,
        ),
        dict(
            arr=186_000,
            contracted_hours=850,
            hpace=92,
            sla_attainment_pct=98.4,
            milestones_on_time_pct=88,
            delivery_lead_change_90d=0,
        ),
    ],
    "ai_usage": [
        dict(
            arr=1_240_000,
            committed_spend=1_000_000,
            pacing=95,
            consumption_30d=61_000,
            consumption_30d_prior=94_000,
            licensed_seats=650,
            licensed_seats_prior=640,
            active_users=520,
            is_pilot=0,
            innovation_budget=0,
            production_usage_pct=82,
            active_integrations=4,
            active_workloads=5,
            throttle_events_90d=0,
            deprecated_model_usage_pct=45,
        ),
        dict(
            arr=243_000,
            committed_spend=180_000,
            pacing=176,
            consumption_30d=27_000,
            consumption_30d_prior=21_000,
            licensed_seats=170,
            licensed_seats_prior=165,
            active_users=120,
            is_pilot=0,
            innovation_budget=0,
            production_usage_pct=74,
            active_integrations=2,
            active_workloads=3,
            throttle_events_90d=14,
            deprecated_model_usage_pct=0,
        ),
        dict(
            arr=38_400,
            committed_spend=30_000,
            pacing=38,
            consumption_30d=900,
            consumption_30d_prior=1_300,
            licensed_seats=25,
            licensed_seats_prior=25,
            active_users=9,
            is_pilot=1,
            innovation_budget=1,
            production_usage_pct=12,
            active_integrations=0,
            active_workloads=1,
            throttle_events_90d=0,
            deprecated_model_usage_pct=0,
        ),
        dict(
            arr=910_000,
            committed_spend=720_000,
            pacing=104,
            consumption_30d=66_000,
            consumption_30d_prior=58_000,
            licensed_seats=520,
            licensed_seats_prior=480,
            active_users=460,
            is_pilot=0,
            innovation_budget=0,
            production_usage_pct=93,
            active_integrations=9,
            active_workloads=8,
            throttle_events_90d=0,
            deprecated_model_usage_pct=0,
        ),
        dict(
            arr=186_000,
            committed_spend=120_000,
            pacing=112,
            consumption_30d=12_500,
            consumption_30d_prior=9_300,
            licensed_seats=155,
            licensed_seats_prior=230,
            active_users=140,
            is_pilot=0,
            innovation_budget=0,
            production_usage_pct=88,
            active_integrations=5,
            active_workloads=4,
            throttle_events_90d=1,
            deprecated_model_usage_pct=10,
        ),
    ],
}


def _stories(profile: str, as_of: date) -> pd.DataFrame:
    rows = []
    for base, sig in zip(STORY_BASE, STORY_SIGNALS[profile], strict=True):
        r: dict[str, Any] = {**base, **sig}
        renew = as_of + timedelta(days=int(r.pop("renew_in")))
        tenure = float(r.pop("tenure_months"))
        days_left = (renew - as_of).days
        elapsed = min(1.0, max(0.08, 1 - days_left / 365.0))
        if "pacing" in r:
            r["consumed_to_date"] = round(r["committed_spend"] * elapsed * r.pop("pacing") / 100)
        if "hpace" in r:
            r["hours_used"] = round(r["contracted_hours"] * elapsed * r.pop("hpace") / 100)
        r["renewal_date"] = renew
        r["contract_start_date"] = renew - timedelta(days=round(tenure * 30.44))
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------
# Vendor-shaped exports, used to show that ingest maps real-world headers without a template.


def to_salesforce_report(book: pd.DataFrame) -> pd.DataFrame:
    yn = {1.0: "true", 0.0: "false"}
    return pd.DataFrame(
        {
            "Account ID": book["account_id"],
            "Account Name": book["account_name"],
            "Account Owner": book["csm_owner"],
            "Customer_Segment__c": book["segment"],
            "Industry": book["industry"],
            "ARR__c": book["arr"].map(lambda v: f"${v:,.2f}"),
            "Renewal_Date__c": pd.to_datetime(book["renewal_date"]).dt.strftime("%m/%d/%Y"),
            "Customer_Since__c": pd.to_datetime(book["contract_start_date"]).dt.strftime("%m/%d/%Y"),
            "Licensed_Users__c": book["licensed_seats"],
            "Active_Users_30d__c": book["active_users"],
            "Active_Users_Prior_Quarter__c": book["active_users_prior"],
            "Open Cases": book["tickets_90d"],
            "Priority_1_Cases__c": book["p1_tickets_90d"],
            "Executive_Sponsor__c": book["exec_sponsor"].map(yn),
            "Champion_Departed__c": book["champion_change_90d"].map(yn),
            "Days_Past_Due__c": book["days_past_due"],
            "Renewal_Uplift__c": (book["price_increase_pct"] / 100).round(3),
            "Auto_Renewal__c": book["auto_renew"].map(yn),
            "Multi_Year__c": book["multi_year"].map(yn),
        }
    )


def to_gainsight_export(book: pd.DataFrame) -> pd.DataFrame:
    ryg = {1.0: "Red", 2.0: "Red", 3.0: "Yellow", 4.0: "Green", 5.0: "Green"}
    yn = {1.0: "Yes", 0.0: "No"}
    return pd.DataFrame(
        {
            "GS Company ID": book["account_id"],
            "Company Name": book["account_name"],
            "CSM": book["csm_owner"],
            "Tier": book["segment"],
            "ARR": book["arr"],
            "Next Renewal Date": pd.to_datetime(book["renewal_date"]).dt.strftime("%Y-%m-%d"),
            "Licensed Users": book["licensed_seats"],
            "Active Users (30d)": book["active_users"],
            "Active Users 90d Ago": book["active_users_prior"],
            "Key Feature Adoption": book["feature_adoption_pct"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}%"),
            "Days Since Last Login": book["days_since_last_login"],
            "Support CSAT": (book["csat"] / 20).round(1),
            "Latest NPS": book["nps"],
            "CSM Sentiment": book["csm_sentiment"].map(ryg),
            "Executive Sponsor": book["exec_sponsor"].map(yn),
            "Days Since Last Meeting": book["days_since_last_meeting"],
            "EBR Completed": book["qbr_last_180d"].map(yn),
        }
    )
