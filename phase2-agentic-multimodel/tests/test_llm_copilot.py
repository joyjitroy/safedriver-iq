"""M7 tests — the LLM co-pilot is additive, guarded, and never on the safety path.

No network: the "off" path and an injected fake completion exercise everything. The key
guarantees under test are the FALLBACKS (timeout/error/off -> deterministic text) and that
the co-pilot only phrases already-decided facts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from sdiq.config import LLMConfig
from sdiq.llm_copilot import LLMCopilot
from sdiq.scenario_summary import ScenarioSummary


# lightweight stand-ins so we don't load the heavy models
def _summary(**over) -> ScenarioSummary:
    base = dict(
        scene_id="scene-x", source="nuscenes", is_night=True, is_rain=True,
        n_agents=20, n_vru=8, env_safety_score=15.0, env_risk=0.85, env_multiplier=1.35,
        env_risk_level="Critical", trajectory_risk=0.6, trajectory_mean_risk=0.2,
        vru_risk=0.7, vru_realistic_risk=0.4, vru_near_misses=1, min_distance=2.1,
        min_ttc=1.0, fused_risk=0.8, combined_safety_score=20.0,
    )
    base.update(over)
    return ScenarioSummary(**base)


@dataclass
class _Decision:
    tier: int
    tier_name: str
    shap: dict
    explanation: str


_DECISION = _Decision(
    tier=3, tier_name="emergency",
    shap={"vru_risk": 0.5, "imminence": 0.3, "is_night": 0.1, "env_multiplier": 0.05,
          "trajectory_risk": 0.0, "proximity": 0.2, "env_risk": 0.0, "is_rain": 0.0},
    explanation="Emergency intervention, driven mainly by pedestrian/cyclist conflict.",
)


# ---------------------------------------------------------------------------
# off / fallback behavior (no provider)
# ---------------------------------------------------------------------------
def test_off_provider_uses_fallbacks():
    co = LLMCopilot(cfg=LLMConfig(provider="off"))
    assert not co.enabled
    # scenario summary falls back to the deterministic ScenarioSummary.summary()-style text
    s = co.summarize_scenario(_summary())
    assert isinstance(s, str) and len(s) > 0
    # intervention explanation falls back to the SHAP explanation verbatim
    assert co.explain_intervention(_summary(), _DECISION) == _DECISION.explanation


def test_completion_failure_falls_back():
    def boom(system, prompt, model, timeout):
        raise TimeoutError("simulated timeout")
    co = LLMCopilot(cfg=LLMConfig(provider="off"), completion_fn=boom)
    # even though a completion_fn is set, a raised error -> deterministic fallback
    assert co.explain_intervention(_summary(), _DECISION) == _DECISION.explanation
    assert co.summarize_scenario(_summary())  # non-empty fallback


def test_silent_tier_uses_shap_explanation_directly():
    # tier 0 (silent) should never call the LLM — just return the SHAP text
    calls = []
    def spy(system, prompt, model, timeout):
        calls.append(model); return "LLM TEXT"
    co = LLMCopilot(cfg=LLMConfig(provider="off"), completion_fn=spy)
    silent = _Decision(0, "silent", _DECISION.shap, "No intervention — conditions nominal.")
    assert co.explain_intervention(_summary(), silent) == "No intervention — conditions nominal."
    assert calls == []  # LLM not invoked for silent


# ---------------------------------------------------------------------------
# fake-completion behavior (exercises prompt building + model routing)
# ---------------------------------------------------------------------------
def test_fake_completion_used_when_present():
    seen = {}
    def fake(system, prompt, model, timeout):
        seen["system"] = system; seen["model"] = model; seen["prompt"] = prompt
        return "A pedestrian is crossing ahead at night."
    co = LLMCopilot(cfg=LLMConfig(provider="off"), completion_fn=fake)
    out = co.summarize_scenario(_summary())
    assert out == "A pedestrian is crossing ahead at night."
    # cheap model used for summaries
    assert seen["model"] == LLMConfig().claude_cheap_model
    # the system prompt forbids changing decided values (guardrail)
    assert "NEVER" in seen["system"].upper() or "do not" in seen["system"].lower()


def test_payload_contains_only_decided_facts():
    captured = {}
    def fake(system, prompt, model, timeout):
        captured["payload"] = json.loads(prompt.split("\n", 1)[1]); return "ok"
    co = LLMCopilot(cfg=LLMConfig(provider="off"), completion_fn=fake)
    co.summarize_scenario(_summary())
    p = captured["payload"]
    # carries the decided assessment...
    assert p["combined_safety_score"] == 20.0 and p["vru_near_misses"] == 1
    assert p["is_night"] and p["is_rain"]
    # ...and does NOT smuggle in raw model internals the LLM might "recompute"
    assert "to_vector" not in p and "shap" not in p


def test_explain_routes_top_factors():
    captured = {}
    def fake(system, prompt, model, timeout):
        captured["payload"] = json.loads(prompt.split("\n", 1)[1]); return "because VRU."
    co = LLMCopilot(cfg=LLMConfig(provider="off"), completion_fn=fake)
    out = co.explain_intervention(_summary(), _DECISION)
    assert out == "because VRU."
    # the top SHAP factor (vru_risk) is passed; the chosen tier is passed verbatim
    assert captured["payload"]["intervention"] == "emergency"
    assert "vru_risk" in captured["payload"]["top_factors"]


def test_narrate_memory_fallback_without_llm():
    from sdiq.agentic_layer import AgentMemory, MemoryEntry, TIER_NAMES
    mem = AgentMemory()
    for i in range(4):
        mem.add(MemoryEntry([0.0] * 8, i % 4, TIER_NAMES[i % 4], 0.1 * i))
    co = LLMCopilot(cfg=LLMConfig(provider="off"))
    out = co.narrate_memory(mem)
    assert "4 situations" in out
    # empty memory has its own message
    assert "No driving history" in co.narrate_memory(AgentMemory())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
