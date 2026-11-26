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
