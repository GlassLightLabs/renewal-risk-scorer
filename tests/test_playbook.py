"""The playbook: every risk gets a play, plays fit the tier, and teams can bring their own."""

from __future__ import annotations

from importlib import resources

import pytest

from renewal_risk.cli import main
from renewal_risk.config import ConfigError, list_profiles, load_config
from renewal_risk.narrate import fallback_narrative
from renewal_risk.pipeline import score
from renewal_risk.playbook import default_playbook, load_playbook, missing_alert_plays, plays_summary, recommend

ADOPTION = [{"feature": "seat_utilization", "group": "product_adoption", "label": "x", "value": "y", "effect": 1.0}]
SIZE = [{"feature": "log_arr", "group": "commercial", "label": "Contract size", "value": "$1M ARR", "effect": 1.0}]


def alert(id_: str, severity: str) -> dict[str, str]:
    return {"id": id_, "severity": severity, "message": id_}


@pytest.mark.parametrize("profile", list(list_profiles()))
def test_every_alert_and_group_in_every_profile_has_a_play(profile):
    cfg = load_config(profile)
    pb = default_playbook()
    assert missing_alert_plays(a.id for a in cfg.alerts) == []
    assert {g.name for g in cfg.groups} <= set(pb.groups)
    assert {t.name for t in cfg.tiers} | {"low"} <= set(pb.tiers)


def test_built_in_plays_are_complete_and_free_of_em_dashes():
    text = resources.files("renewal_risk").joinpath("playbooks/default.yaml").read_text()
    assert chr(0x2014) not in text and chr(0x2013) not in text  # no em or en dashes
    pb = default_playbook()
    for play in [*pb.alerts.values(), *pb.groups.values(), *pb.special.values()]:
        assert play["name"] and play["owner"] and play["talk_track"] and play["exit_criteria"]
        assert len(play["actions"]) >= 2


def test_low_tier_gets_a_healthy_renewal_not_a_save_play():
    assert recommend("saas", ADOPTION, [], 200, tier="low")["play"] == "Healthy renewal"
    assert recommend("saas", ADOPTION, [], 200, tier="high")["play"] == "Adoption recovery"
    # A specific alert still names the cause, even on a healthy account.
    assert recommend("saas", ADOPTION, [alert("discount_cliff", "medium")], 120, tier="low")["play"] == (
        "Plan the discount step-down"
    )


def test_contract_size_never_picks_a_play():
    assert recommend("saas", SIZE, [], 200, tier="watch")["play"] == "Standard renewal motion"
    assert recommend("saas", SIZE + ADOPTION, [], 200, tier="watch")["play"] == "Adoption recovery"


def test_watermelons_always_get_their_check():
    wm = recommend("saas", ADOPTION, [alert("first_renewal", "low")], 200, True, tier="watch")
    assert wm["play"] == "Watermelon check" and wm["supporting"] == ["First renewal readiness"]
    alerts = [alert("champion_left", "high"), alert("detractor", "medium"), alert("dark_account", "medium")]
    with_alert = recommend("saas", ADOPTION, alerts, 100, True, tier="watch")
    assert with_alert["play"] == "Re-establish sponsorship"
    assert with_alert["supporting"] == ["Watermelon check", "Detractor recovery"]


def test_alerts_rank_by_severity_and_low_severity_only_supports():
    alerts = [alert("first_renewal", "low"), alert("detractor", "medium"), alert("payment_risk", "high")]
    play = recommend("saas", ADOPTION, alerts, 60, tier="critical")
    assert play["play"] == "Collections and commercial review"
    assert play["supporting"] == ["Detractor recovery", "First renewal readiness"]
    assert recommend("saas", ADOPTION, [alert("first_renewal", "low")], 60, tier="high")["play"] == "Adoption recovery"


def test_tier_sets_motion_cadence_and_escalation():
    crit = recommend("saas", ADOPTION, [], 60, tier="critical")
    assert crit["motion"] == "Save" and "weekly" in crit["cadence"] and crit["escalation"]
    assert recommend("saas", ADOPTION, [], 60, tier="watch")["motion"] == "Monitor"
    untiered = recommend("saas", ADOPTION, [], 60)
    assert untiered["motion"] == "" and untiered["talk_track"] and untiered["exit_criteria"]


def test_custom_playbook_layers_on_the_built_in(tmp_path):
    f = tmp_path / "ours.yaml"
    f.write_text(
        "alerts:\n"
        "  champion_left:\n"
        "    name: Champion rescue\n"
        "    owner: CS leader\n"
        "    actions: [Call the new VP]\n"
        "    talk_track: Hi there.\n"
        "    exit_criteria: VP met\n"
        "tiers:\n"
        "  critical: {cadence: Daily standup}\n"
    )
    pb = load_playbook(f)
    assert pb.alerts["champion_left"]["name"] == "Champion rescue"
    assert pb.alerts["detractor"]["name"] == "Detractor recovery"  # untouched plays come from the built-in
    assert pb.tiers["critical"] == {**default_playbook().tiers["critical"], "cadence": "Daily standup"}
    play = recommend("saas", [], [alert("champion_left", "high")], 30, tier="critical", playbook=pb)
    assert play["play"] == "Champion rescue" and play["cadence"] == "Daily standup"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("alerts:\n  x: {name: a, owner: b, actions: [c], talk_track: d}\n", "missing exit_criteria"),
        ("alerts:\n  x: {name: a, owner: b, actions: c, talk_track: d, exit_criteria: e}\n", "list of steps"),
        ("alerts:\n  x: {name: a, owner: b, actions: [c], talk_track: d, exit_criteria: e, why: f}\n", "unknown"),
        ("plays: {}\n", "unknown section"),
        ("special:\n  rescue: {name: a, owner: b, actions: [c], talk_track: d, exit_criteria: e}\n", "rescue"),
        ("tiers:\n  critical: {colour: red}\n", "use only"),
    ],
)
def test_bad_playbooks_are_explained(tmp_path, body, message):
    f = tmp_path / "bad.yaml"
    f.write_text(body)
    with pytest.raises(ConfigError, match=message):
        load_playbook(f)


@pytest.mark.parametrize("profile", ["ai_usage", "saas"])
def test_scored_book_plays_fit_the_tier(trained, books, as_of, profile):
    _, model = trained[profile]
    cfg = load_config(profile)
    r = score(books[profile], cfg, model, as_of=as_of, simulations=200)
    a = r.accounts
    low = a[a["risk_tier"] == "low"]
    assert not low["play_id"].str.startswith("group:").any(), "a low-tier account got a driver-based save play"
    wm = a[a["watermelon"]]
    has_check = (wm["save_play"] == "Watermelon check") | wm["supporting_plays"].map(
        lambda xs: "Watermelon check" in xs
    )
    assert has_check.all()
    for col in ("play_talk_track", "play_exit_criteria", "play_motion", "play_cadence"):
        assert a[col].map(bool).all(), col
    plays = r.summary["plays"]
    assert sum(p["accounts"] for p in plays) == int((a["risk_tier"] != "low").sum())
    assert [p["expected_arr_loss"] for p in plays] == sorted((p["expected_arr_loss"] for p in plays), reverse=True)
    assert set(r.summary["play_tiers"]) == {t.name for t in cfg.tiers}
    assert plays == plays_summary(a, tier_order=[t.name for t in cfg.tiers])


def test_fallback_brief_uses_the_plays_talk_track(trained, books, as_of):
    _, model = trained["ai_usage"]
    row = score(books["ai_usage"], load_config("ai_usage"), model, as_of=as_of, simulations=200).accounts.iloc[0]
    brief = fallback_narrative(row)
    assert brief["talk_track"] == row["play_talk_track"]
    assert brief["done_when"] == row["play_exit_criteria"]


def test_cli_playbook_check_and_custom_playbook(tmp_path, capsys):
    assert main(["playbook"]) == 0
    assert "every alert and group has a play" in capsys.readouterr().out
    f = tmp_path / "ours.yaml"
    f.write_text(
        "special:\n  healthy:\n    name: Renewal on autopilot\n    owner: Renewals rep\n"
        "    actions: [Send the quote]\n    talk_track: Thanks for another year.\n    exit_criteria: Signed\n"
    )
    out = tmp_path / "out"
    argv = ["demo", "--history", "1500", "--book", "150", "--as-of", "2026-12-01", "--out", str(out)]
    assert main([*argv, "--playbook", str(f)]) == 0
    assert "Renewal on autopilot" in (out / "scored_accounts.csv").read_text()
    assert "## Plays to run" in (out / "report.md").read_text()
    assert main([*argv, "--playbook", str(tmp_path / "missing.yaml")]) == 2
