from __future__ import annotations

import pytest
import yaml

from renewal_risk.config import ConfigError, list_profiles, load_config, parse_config


def test_all_profiles_load_and_cover_their_model_features():
    profiles = list_profiles()
    assert set(profiles) == {"saas", "consumption", "hardware", "services", "ai_usage"}
    for name in profiles:
        cfg = load_config(name)
        assert cfg.profile == name
        assert sum(g.weight for g in cfg.groups) == pytest.approx(100)
        assert any(a.id == "champion_left" for a in cfg.alerts)  # common alerts merged in
        assert {f.direction for f in cfg.model_features} <= {-1, 0, 1}


def test_custom_config_file_inherits_from_a_profile(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump({"base": "saas.yaml", "profile": "saas", "bands": {"green": 80, "yellow": 60}}))
    cfg = load_config(path=path)
    assert cfg.green == 80 and cfg.yellow == 60
    assert len(cfg.groups) == len(load_config("saas").groups)


def test_bad_breakpoints_are_rejected():
    bad = {
        "profile": "x",
        "groups": {
            "g": {
                "label": "G",
                "weight": 100,
                "indicator": "leading",
                "measures": {"m": {"weight": 1, "points": [[1, 2]]}},
            }
        },
    }
    with pytest.raises(ConfigError):
        parse_config(bad)


def test_unknown_profile():
    with pytest.raises((ConfigError, FileNotFoundError, ValueError)):
        load_config("crypto")
