from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from renewal_risk.config import load_config
from renewal_risk.narrate import MODEL, account_facts, narrate, narrate_one, to_markdown
from renewal_risk.pipeline import score


@pytest.fixture(scope="module")
def scored(trained, books, as_of):
    _, model = trained["ai_usage"]
    return score(books["ai_usage"], load_config("ai_usage"), model, as_of=as_of).accounts


class FakeClient:
    def __init__(self, payload=None, stop_reason="end_turn", error=None):
        self.payload, self.stop_reason, self.error, self.calls = payload, stop_reason, error, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        text = json.dumps(self.payload)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=text)])


GOOD = {
    "headline": "h",
    "why_at_risk": ["a"],
    "save_plan": [{"action": "x", "owner": "CSM", "when": "now"}],
    "talk_track": "t",
}


def test_fallback_without_api_key(scored, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    briefs = narrate(scored, top=3)
    assert len(briefs) == 3 and all(b["source"] == "rules" for b in briefs)
    assert all(b["save_plan"] for b in briefs)
    assert "Save plan" in to_markdown(briefs)


def test_claude_path_uses_schema_cache_and_latest_model(scored):
    client = FakeClient(GOOD)
    out = narrate_one(scored.iloc[0], client)
    assert out["headline"] == "h" and out["source"] == MODEL == "claude-opus-5-5"
    call = client.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    sent = json.loads(call["messages"][0]["content"].split("\n\n", 1)[1])
    assert sent == json.loads(json.dumps(account_facts(scored.iloc[0]), default=str))


def test_failures_and_refusals_fall_back(scored):
    assert narrate_one(scored.iloc[0], FakeClient(error=RuntimeError("down")))["source"] == "rules"
    refused = narrate_one(scored.iloc[0], FakeClient(GOOD, stop_reason="refusal"))
    assert refused["source"] == "rules" and "declined" in refused["note"]
