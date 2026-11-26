from __future__ import annotations

import json

from renewal_risk.cli import main


def test_profiles(capsys):
    assert main(["profiles"]) == 0
    assert "ai_usage" in capsys.readouterr().out


def test_generate_train_score_inspect(tmp_path, capsys):
    d = tmp_path / "data"
    assert (
        main(
            [
                "generate",
                "--profile",
                "saas",
                "--out",
                str(d),
                "--history",
                "1500",
                "--book",
                "150",
                "--as-of",
                "2026-12-01",
            ]
        )
        == 0
    )
    model = tmp_path / "model.json"
    assert (
        main(["train", str(d / "saas_history.csv"), "--profile", "saas", "--out", str(model), "--as-of", "2026-12-01"])
        == 0
    )
    out = tmp_path / "out"
    assert (
        main(
            [
                "score",
                str(d / "salesforce_report.csv"),
                "--model",
                str(model),
                "--out",
                str(out),
                "--as-of",
                "2026-12-01",
                "--narrate",
                "2",
            ]
        )
        == 0
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["summary"]["accounts"] == 150 and len(summary["briefs"]) == 2
    assert (out / "report.md").read_text().startswith("# Renewal risk report")
    assert main(["inspect", str(d / "gainsight_export.csv"), "--as-of", "2026-12-01"]) == 0
    assert "model signals present" in capsys.readouterr().out


def test_bad_input_returns_error_code(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    assert main(["score", str(bad)]) == 2
    assert "Missing required columns" in capsys.readouterr().err


def test_sources_list_and_test(tmp_path, capsys, monkeypatch):
    assert main(["sources", "list"]) == 0
    assert "zendesk" in capsys.readouterr().out
    demo = tmp_path / "demo.yaml"
    demo.write_text("sources:\n  zendesk:\n    simulate: true\n")
    assert main(["sources", "test", "--sources", str(demo)]) == 0
    assert "Connected to demo.zendesk.com" in capsys.readouterr().out
    real = tmp_path / "real.yaml"
    real.write_text("sources:\n  zendesk:\n    subdomain: acme\n    email: ${ZD_EMAIL}\n    api_token: x\n")
    monkeypatch.delenv("ZD_EMAIL", raising=False)
    assert main(["sources", "test", "--sources", str(real)]) == 2
    assert "ZD_EMAIL is not set" in capsys.readouterr().err


def test_demo_with_simulated_zendesk(tmp_path, capsys):
    demo = tmp_path / "demo.yaml"
    demo.write_text("sources:\n  zendesk:\n    simulate: true\n")
    args = ["demo", "--profile", "saas", "--history", "1200", "--book", "120", "--as-of", "2026-12-01"]
    assert main([*args, "--sources", str(demo), "--out", str(tmp_path / "o")]) == 0
    assert "Zendesk: matched" in capsys.readouterr().out
