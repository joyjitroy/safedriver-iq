"""M6 tests — RL fusion policy, reward, memory, and SHAP-explained decisions.

The reward/severity/memory tests are exact and fast. The policy is trained small (synthetic
states) so it runs in seconds without the heavy perception models.
"""
from __future__ import annotations

import numpy as np
import pytest

from sdiq.agentic_layer import (
    AgentMemory, AgenticReasoner, Decision, MemoryEntry, N_TIERS, TIER_NAMES,
    QNet, reward, tier_from_severity, train_policy, true_severity,
)
from sdiq.scenario_summary import VECTOR_FIELDS


# ---------------------------------------------------------------------------
# reward / severity (exact)
# ---------------------------------------------------------------------------
def test_tier_from_severity_monotonic():
    tiers = [tier_from_severity(s) for s in (0.1, 0.3, 0.6, 0.9)]
    assert tiers == [0, 1, 2, 3]


def test_reward_peaks_at_ideal_and_asymmetric():
    sev = 0.9                                # ideal tier = 3 (emergency)
    assert reward(3, sev) == 1.0
    # under-reaction (too soft) is penalized more than equivalent over-reaction
    sev2 = 0.6                               # ideal = 2
    assert reward(1, sev2) < reward(3, sev2)   # missing < false alarm (both 1 off)


def test_true_severity_interaction():
    # imminent VRU conflict at night is worse than the same in daylight
    base = np.zeros(8, dtype=np.float32)
    idx = {f: i for i, f in enumerate(VECTOR_FIELDS)}
    day = base.copy(); day[idx["vru_risk"]] = 0.8; day[idx["imminence"]] = 0.8
    day[idx["env_multiplier"]] = 1.0
    night = day.copy(); night[idx["is_night"]] = 1.0; night[idx["env_multiplier"]] = 1.4
    assert true_severity(night) > true_severity(day)


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
def test_qnet_shape():
    import torch
    out = QNet()(torch.randn(5, len(VECTOR_FIELDS)))
    assert out.shape == (5, N_TIERS)


@pytest.fixture(scope="module")
def trained():
    net, report = train_policy(n=15000, epochs=50, save_path="/tmp/agentic_test.pt", seed=0)
    return net, report


def test_policy_learns_graduated_mapping(trained):
    _, report = trained
    assert report.tier_accuracy > 0.8, f"tier accuracy too low: {report.tier_accuracy}"
    assert report.underreaction_rate < 0.05, "too many safety-critical under-reactions"


def test_policy_escalates_with_risk(trained):
    net, _ = trained
    reasoner = AgenticReasoner(net=net)
    idx = {f: i for i, f in enumerate(VECTOR_FIELDS)}
    safe = np.zeros(8, dtype=np.float32); safe[idx["env_multiplier"]] = 0.8
    danger = np.zeros(8, dtype=np.float32)
    danger[idx["vru_risk"]] = 0.95; danger[idx["imminence"]] = 0.9
    danger[idx["proximity"]] = 0.9; danger[idx["env_multiplier"]] = 1.4
    danger[idx["is_night"]] = 1.0
    d_safe = reasoner.decide(safe, explain=False)
    d_danger = reasoner.decide(danger, explain=False)
    assert d_safe.tier < d_danger.tier
    assert d_danger.tier == 3 and d_safe.tier == 0


def test_decision_has_shap_and_explanation(trained):
    net, _ = trained
    reasoner = AgenticReasoner(net=net)
    idx = {f: i for i, f in enumerate(VECTOR_FIELDS)}
    state = np.zeros(8, dtype=np.float32)
    state[idx["vru_risk"]] = 0.9; state[idx["imminence"]] = 0.9
    state[idx["env_multiplier"]] = 1.3
    d = reasoner.decide(state, explain=True)
    assert isinstance(d, Decision)
    assert set(d.shap) == set(VECTOR_FIELDS)
    # SHAP attribution must be non-degenerate (not all ~0)
    assert max(abs(v) for v in d.shap.values()) > 1e-3
    assert d.tier_name in TIER_NAMES and len(d.explanation) > 10


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
def test_memory_consolidate_and_recall(tmp_path):
    mem = AgentMemory(short_capacity=10, path=tmp_path / "mem.json")
    for i in range(3):
        mem.add(MemoryEntry(state=[float(i)] * 8, action=i % N_TIERS,
                            tier=TIER_NAMES[i % N_TIERS], severity=0.1 * i))
    assert len(mem.short) == 3 and len(mem.long) == 0
    moved = mem.consolidate()
    assert moved == 3 and len(mem.long) == 3 and len(mem.short) == 0
    # recall nearest
    near = mem.recall_similar(np.array([2.0] * 8), k=1)
    assert near and near[0].state == [2.0] * 8
    # persistence round-trip
    mem.save()
    mem2 = AgentMemory(path=tmp_path / "mem.json")
    assert len(mem2.long) == 3


def test_reasoner_records_to_memory(trained):
    net, _ = trained
    reasoner = AgenticReasoner(net=net, memory=AgentMemory())
    reasoner.decide(np.zeros(8, dtype=np.float32), explain=False)
    assert len(reasoner.memory.short) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
