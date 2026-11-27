"""Canonical account schema and the column aliases used to map CRM and CS-platform exports onto it.

Every export looks a little different. Salesforce reports use API names like ``ARR__c``, Gainsight uses
labels like "Licensed Users", and spreadsheets use whatever someone typed. Rather than force a template,
the ingest step maps each incoming column to one canonical field through the alias table below. A YAML
mapping file can override or extend the aliases for an unusual export.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

FieldType = Literal["str", "float", "int", "bool", "date", "category"]


@dataclass(frozen=True)
class Field:
    name: str
    type: FieldType
    label: str
    aliases: tuple[str, ...] = ()
    required: bool = False
    history_only: bool = False


FIELDS: tuple[Field, ...] = (
    # Identity
    Field(
        "account_id",
        "str",
        "Account ID",
        (
            "account id",
            "accountid",
            "id",
            "account_id__c",
            "sfdc account id",
            "salesforce account id",
            "gs company id",
            "company id",
        ),
        required=True,
    ),
    Field(
        "account_name",
        "str",
        "Account",
        ("account name", "account", "company", "company name", "customer", "customer name", "name"),
    ),
    Field(
        "csm_owner",
        "str",
        "CSM",
        ("csm", "csm owner", "csm name", "customer success manager", "account owner", "owner", "owner name"),
    ),
    Field("segment", "category", "Segment", ("segment", "customer segment", "tier", "account tier", "market segment")),
    Field("industry", "category", "Industry", ("industry", "vertical")),
    Field("region", "category", "Region", ("region", "geo", "territory", "billing region")),
    # Commercial
    Field(
        "arr",
        "float",
        "ARR",
        (
            "arr",
            "arr__c",
            "annual recurring revenue",
            "current arr",
            "arr (usd)",
            "contract value",
            "annual contract value",
            "acv",
            "mrr x 12",
        ),
        required=True,
    ),
    Field(
        "renewal_date",
        "date",
        "Renewal date",
        (
            "renewal date",
            "renewal_date__c",
            "next renewal date",
            "contract end date",
            "contract end",
            "end date",
            "subscription end date",
            "renewal",
        ),
        required=True,
    ),
    Field(
        "contract_start_date",
        "date",
        "Customer since",
        (
            "customer since",
            "contract start date",
            "original contract start",
            "start date",
            "first contract date",
            "customer_since__c",
        ),
    ),
    Field(
        "licensed_seats",
        "float",
        "Licensed seats",
        ("licensed seats", "licensed users", "seats", "licenses", "purchased seats", "entitled users", "seat count"),
    ),
    Field(
        "active_users",
        "float",
        "Active users",
        (
            "active users",
            "monthly active users",
            "mau",
            "active users (30d)",
            "active users 30d",
            "active licenses",
            "users active",
        ),
    ),
    Field(
        "active_users_prior",
        "float",
        "Active users 90 days ago",
        (
            "active users prior",
            "active users 90d ago",
            "mau 90 days ago",
            "prior mau",
            "active users (prior quarter)",
            "active users last quarter",
        ),
    ),
    Field(
        "feature_adoption_pct",
        "float",
        "Feature adoption",
        (
            "feature adoption",
            "feature adoption %",
            "feature adoption pct",
            "breadth of use",
            "key feature adoption",
            "adoption score",
        ),
    ),
    Field(
        "days_since_last_login",
        "float",
        "Days since last login",
        ("days since last login", "days since login", "last login (days)", "days since last activity"),
    ),
    Field(
        "tickets_90d",
        "float",
        "Support tickets (90d)",
        ("tickets 90d", "support tickets", "tickets (90d)", "case count", "cases last 90 days", "open cases"),
    ),
    Field(
        "p1_tickets_90d",
        "float",
        "P1 tickets (90d)",
        ("p1 tickets", "p1 tickets 90d", "critical tickets", "sev1 tickets", "priority 1 cases", "escalated cases"),
    ),
    Field(
        "tickets_90d_prior",
        "float",
        "Support tickets (prior 90d)",
        ("tickets prior 90d", "prior tickets", "tickets previous 90 days", "tickets 90d prior", "prior case count"),
    ),
    Field("csat", "float", "CSAT", ("csat", "csat score", "support csat", "customer satisfaction")),
    Field(
        "ticket_reopen_rate_pct",
        "float",
        "Ticket reopen rate",
        ("reopen rate", "ticket reopen rate", "reopened tickets pct", "reopen pct", "reopened rate"),
    ),
    Field(
        "ticket_resolution_hours",
        "float",
        "Median time to resolution (hours)",
        ("resolution hours", "median resolution hours", "time to resolution", "full resolution time hours"),
    ),
    # Engineering issues linked to the customer (filled by the Jira add-on, or by a column in the export)
    Field(
        "open_bugs",
        "float",
        "Open customer bugs",
        ("open bugs", "open customer bugs", "open defects", "open jira bugs", "open engineering bugs"),
    ),
    Field(
        "open_bug_age_days",
        "float",
        "Median age of open bugs (days)",
        ("open bug age days", "median open bug age", "open bug age", "bug age (days)", "median bug age"),
    ),
    Field(
        "escalations_90d",
        "float",
        "Engineering escalations (90d)",
        ("escalations 90d", "engineering escalations", "escalations (90d)", "jira escalations", "dev escalations"),
    ),
    Field("nps", "float", "NPS", ("nps", "nps score", "latest nps", "nps response", "likelihood to recommend")),
    Field(
        "csm_sentiment",
        "float",
        "CSM sentiment",
        ("csm sentiment", "sentiment", "csm health", "csm pulse", "csm score"),
    ),
    Field(
        "exec_sponsor",
        "bool",
        "Executive sponsor",
        (
            "exec sponsor",
            "executive sponsor",
            "has exec sponsor",
            "exec sponsor engaged",
            "executive sponsor identified",
        ),
    ),
    Field(
        "champion_change_90d",
        "bool",
        "Champion left (90d)",
        (
            "champion change",
            "champion change 90d",
            "champion left",
            "key contact change",
            "champion departed",
            "primary contact changed",
        ),
    ),
    Field(
        "days_since_last_meeting",
        "float",
        "Days since last meeting",
        ("days since last meeting", "days since last touch", "days since last engagement", "last meeting (days)"),
    ),
    Field(
        "qbr_last_180d",
        "bool",
        "QBR in last 180d",
        ("qbr last 180d", "qbr completed", "ebr completed", "business review last 180d", "qbr in last 6 months"),
    ),
    Field(
        "days_past_due",
        "float",
        "Days past due",
        ("days past due", "ar days past due", "invoice days past due", "dpd", "past due days"),
    ),
    Field(
        "price_increase_pct",
        "float",
        "Renewal uplift",
        ("price increase", "price increase %", "uplift", "renewal uplift", "renewal uplift %", "price uplift"),
    ),
    Field("auto_renew", "bool", "Auto-renew", ("auto renew", "auto-renew", "evergreen", "auto renewal")),
    Field(
        "multi_year", "bool", "Multi-year contract", ("multi year", "multi-year", "multiyear", "multi year contract")
    ),
    Field(
        "competitor_eval",
        "bool",
        "Competitor evaluation",
        (
            "competitor eval",
            "competitive evaluation",
            "competitor evaluation",
            "evaluating competitor",
            "competitive threat",
            "rfp in progress",
        ),
    ),
    Field(
        "discount_pct",
        "float",
        "Discount at signing",
        (
            "discount",
            "discount %",
            "discount pct",
            "discount percent",
            "contract discount",
            "deal discount",
            "discount off list",
            "last discount",
            "signing discount",
        ),
    ),
    Field(
        "signer_departed",
        "bool",
        "Contract signer left",
        (
            "signer departed",
            "signer left",
            "contract signer left",
            "signatory left",
            "signatory departed",
            "economic buyer left",
            "buyer left",
        ),
    ),
    Field(
        "snapshot_date",
        "date",
        "Snapshot date",
        ("snapshot date", "as of date", "as of", "data as of", "signal date", "captured on"),
    ),
    Field(
        "term_start_date",
        "date",
        "Current term start",
        ("term start date", "current term start", "subscription start date", "contract term start"),
    ),
    # Consumption / usage-based (cloud compute, data platforms, APIs, GPU and data-center capacity)
    Field(
        "committed_spend",
        "float",
        "Annual commit",
        (
            "committed spend",
            "commit",
            "annual commit",
            "commitment",
            "committed amount",
            "capacity commitment",
            "commit amount",
        ),
    ),
    Field(
        "consumed_to_date",
        "float",
        "Consumed this term",
        (
            "consumed to date",
            "consumption to date",
            "spend to date",
            "usage to date",
            "drawdown to date",
            "credits consumed",
            "burned to date",
        ),
    ),
    Field(
        "consumption_30d",
        "float",
        "Consumption (last 30d)",
        (
            "consumption 30d",
            "spend last 30 days",
            "last 30d consumption",
            "monthly consumption",
            "run rate 30d",
            "usage 30d",
        ),
    ),
    Field(
        "consumption_30d_prior",
        "float",
        "Consumption (30d, 90 days ago)",
        ("consumption 30d prior", "prior 30d consumption", "consumption 90d ago", "run rate 90d ago"),
    ),
    Field(
        "active_workloads",
        "float",
        "Active workloads",
        ("active workloads", "workloads", "active projects", "active use cases", "production workloads"),
    ),
    Field(
        "reserved_capacity_utilization_pct",
        "float",
        "Reserved capacity used",
        (
            "reserved capacity utilization",
            "reserved capacity used",
            "capacity utilization",
            "reserved instance utilization",
            "committed capacity utilization",
            "power utilization",
        ),
    ),
    # AI products: prepaid credits or token commits, hybrid seat + usage plans, pilots, model dependency
    Field(
        "licensed_seats_prior",
        "float",
        "Licensed seats a year ago",
        ("licensed seats prior", "seats 12 months ago", "seats last year", "prior seats", "seats at last renewal"),
    ),
    Field(
        "throttle_events_90d",
        "float",
        "Rate-limit or spend-cap hits (90d)",
        (
            "throttle events 90d",
            "rate limit hits",
            "rate limit events",
            "spend cap hits",
            "overage events",
            "usage cap hits",
            "429 errors 90d",
        ),
    ),
    Field(
        "production_usage_pct",
        "float",
        "Production share of usage",
        (
            "production usage pct",
            "production usage share",
            "prod usage %",
            "production share",
            "share of usage in production",
        ),
    ),
    Field(
        "is_pilot",
        "bool",
        "Pilot or proof of concept",
        ("is pilot", "pilot", "poc", "proof of concept", "trial", "pilot flag"),
    ),
    Field(
        "innovation_budget",
        "bool",
        "Funded from an innovation or experimental budget",
        ("innovation budget", "experimental budget", "budget source innovation", "lab budget"),
    ),
    Field(
        "deprecated_model_usage_pct",
        "float",
        "Usage on models retiring before renewal",
        ("deprecated model usage pct", "deprecated model share", "usage on deprecated models", "retiring model usage"),
    ),
    Field(
        "active_integrations",
        "float",
        "Active integrations",
        ("active integrations", "integrations", "connected systems", "api integrations"),
    ),
    # Hardware / devices / infrastructure with support and maintenance contracts
    Field(
        "installed_units",
        "float",
        "Installed units",
        ("installed units", "install base", "installed base", "units installed", "asset count", "devices"),
    ),
    Field(
        "units_under_contract",
        "float",
        "Units under contract",
        ("units under contract", "covered units", "units under support", "contracted units", "assets under contract"),
    ),
    Field(
        "units_past_eosl",
        "float",
        "Units past end of service life",
        ("units past eosl", "units past eol", "end of service life units", "eosl units", "units end of life"),
    ),
    Field(
        "avg_asset_age_years",
        "float",
        "Average asset age (years)",
        ("avg asset age", "average asset age", "asset age years", "fleet age"),
    ),
    Field(
        "rma_rate_pct",
        "float",
        "RMA / failure rate",
        ("rma rate", "failure rate", "return rate", "rma %", "annualized failure rate"),
    ),
    Field(
        "connected_units_pct",
        "float",
        "Telemetry-connected units",
        ("connected units", "connected units pct", "telemetry connected", "phone home %", "call home enabled"),
    ),
    Field(
        "supported_firmware_pct",
        "float",
        "On supported firmware",
        ("supported firmware", "firmware currency", "on supported firmware", "software currency"),
    ),
    # Professional and managed services
    Field(
        "contracted_hours",
        "float",
        "Contracted hours",
        ("contracted hours", "hours purchased", "retainer hours", "sow hours", "hours in contract"),
    ),
    Field(
        "hours_used",
        "float",
        "Hours used this term",
        ("hours used", "hours consumed", "hours delivered", "billable hours used", "hours burned"),
    ),
    Field(
        "sla_attainment_pct",
        "float",
        "SLA attainment",
        ("sla attainment", "sla met", "sla %", "sla compliance", "service level attainment"),
    ),
    Field(
        "milestones_on_time_pct",
        "float",
        "Milestones on time",
        ("milestones on time", "on time delivery", "milestone on time %", "delivery on time"),
    ),
    Field(
        "delivery_lead_change_90d",
        "bool",
        "Delivery lead changed (90d)",
        ("delivery lead change", "engagement manager change", "consultant turnover", "delivery manager changed"),
    ),
    # Outcomes (training history only)
    Field(
        "outcome",
        "category",
        "Outcome",
        ("outcome", "renewal outcome", "renewal status", "stage", "result", "closed status"),
        history_only=True,
    ),
    Field(
        "renewed_arr",
        "float",
        "Renewed ARR",
        ("renewed arr", "renewal arr", "closed arr", "new arr", "renewed amount"),
        history_only=True,
    ),
)

FIELD_BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}
REQUIRED: tuple[str, ...] = tuple(f.name for f in FIELDS if f.required)

OUTCOME_VALUES = ("renewed", "contracted", "churned")
OUTCOME_SYNONYMS: dict[str, str] = {
    "renewed": "renewed",
    "renew": "renewed",
    "closed won": "renewed",
    "won": "renewed",
    "expanded": "renewed",
    "renewed - flat": "renewed",
    "renewed - expanded": "renewed",
    "upsell": "renewed",
    "contracted": "contracted",
    "downsell": "contracted",
    "down-sell": "contracted",
    "renewed - downsell": "contracted",
    "partial churn": "contracted",
    "contraction": "contracted",
    "churned": "churned",
    "churn": "churned",
    "closed lost": "churned",
    "lost": "churned",
    "cancelled": "churned",
    "canceled": "churned",
    "non-renewal": "churned",
    "not renewed": "churned",
}


def normalize_header(text: str) -> str:
    """Lower-case, strip Salesforce suffixes and punctuation so 'Licensed_Users__c' matches 'licensed users'."""
    t = str(text).strip().lower()
    t = re.sub(r"__c$", "", t)
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"[^a-z0-9%() ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for f in FIELDS:
        for candidate in (f.name, f.label, *f.aliases):
            index.setdefault(normalize_header(candidate), f.name)
    return index


ALIAS_INDEX: dict[str, str] = _alias_index()
