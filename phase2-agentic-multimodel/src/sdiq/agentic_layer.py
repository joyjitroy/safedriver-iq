r"""M6 — agentic reasoning layer: RL fusion policy + memory + SHAP explanations.

Layer 3 of the architecture. Takes the M5 `ScenarioSummary` state vector and chooses one
of four graduated interventions, with a deterministic SHAP explanation and a memory of the
current drive (short-term) and learned situations (long-term).

  state (8-dim, M5) -> Q-net -> argmax tier -> {silent, advisory, intervention, emergency}
                                         \-> SHAP attribution over the 8 features
                                         \-> short/long-term memory

RL framing: a one-step contextual bandit (each cycle is an independent decision). The reward
rewards matching the intervention to the situation's severity, penalizing UNDER-reaction
(missing a hazard) more than over-reaction (false alarm). Severity is a surrogate built from
the derived risk/near-miss/TTC signals — there are no crash labels (the same limitation as
M3/M4), so in deployment the reward would come from observed outcomes. "Full RL training" is
out of POC scope (per the ASCE2027 docx); we train a lightweight policy net to demonstrate
the mechanism and that it recovers (and sharpens) the graduated policy from reward alone.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sdiq import config
from sdiq.scenario_summary import VECTOR_FIELDS, ScenarioSummary

log = logging.getLogger("sdiq.agentic")
logging.getLogger("shap").setLevel(logging.WARNING)   # silence its INFO "phi = ..." spam

TIER_NAMES = ("silent", "advisory", "intervention", "emergency")
N_TIERS = len(TIER_NAMES)

# Friendly labels for the 8 state features (order = VECTOR_FIELDS).
_FEATURE_LABELS = {
    "env_risk": "environmental crash risk",
    "env_multiplier": "adverse-condition amplification",
    "trajectory_risk": "erratic vehicle motion",
    "vru_risk": "pedestrian/cyclist conflict",
    "proximity": "close VRU distance",
    "imminence": "imminent approach (low TTC)",
    "is_night": "night",
    "is_rain": "rain",
}


# ---------------------------------------------------------------------------
# reward / severity (the bandit environment)
# ---------------------------------------------------------------------------
def true_severity(state: np.ndarray) -> float:
    """Surrogate situation severity in [0,1] from the 8-dim state vector.

    Deliberately NON-linear: an imminent VRU conflict at night is disproportionately
    severe (interaction term) — the kind of structure a static threshold band misses but
    an RL policy can learn from reward. Used only to generate the training reward and to
    evaluate; the deployed agent uses its learned Q-net, not this oracle.
    """
    env_risk, env_mult, traj, vru, prox, imm, night, rain = state
    dynamic = 0.4 * traj + 0.6 * vru                       # VRU-dominant base risk
    env_factor = 0.6 + 0.4 * (env_mult - 0.5)              # ~0.6..1.0 from the multiplier
    interaction = vru * imm * (1.0 + 0.5 * night)          # imminent VRU, worse at night
    sev = dynamic * env_factor + 0.45 * interaction + 0.1 * rain * vru
    return float(np.clip(sev, 0.0, 1.0))


def tier_from_severity(severity: float) -> int:
    if severity < 0.25:
        return 0
    if severity < 0.50:
        return 1
    if severity < 0.75:
        return 2
    return 3


def reward(action: int, severity: float) -> float:
    """+1 for the ideal tier; under-reaction penalized 2x over-reaction."""
    ideal = tier_from_severity(severity)
    diff = action - ideal
    if diff == 0:
        return 1.0
    return -(1.0 * diff if diff > 0 else 2.0 * -diff)


# ---------------------------------------------------------------------------
# policy network
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, n_features: int = len(VECTOR_FIELDS), hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_TIERS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic state vectors spanning the operating space (order = VECTOR_FIELDS)."""
    s = rng.random((n, 8)).astype(np.float32)
    s[:, 1] = 0.5 + s[:, 1]                       # env_multiplier in [0.5, 1.5]
    s[:, 6] = (s[:, 6] > 0.6).astype(np.float32)  # is_night (binary)
    s[:, 7] = (s[:, 7] > 0.7).astype(np.float32)  # is_rain (binary)
    return s


@dataclass
class PolicyReport:
    n_train: int
    n_val: int
    tier_accuracy: float          # argmax tier == ideal tier
    underreaction_rate: float     # chose a softer tier than ideal (safety-critical miss)
    baseline_band_accuracy: float # the static M5 band, for comparison


def train_policy(n: int = 20000, epochs: int = 60, batch_size: int = 512, lr: float = 1e-3,
                 device: str | None = None, save_path: Path | None = None,
                 seed: int = 0) -> tuple[QNet, PolicyReport]:
    """Regress Q(s,a) -> reward(a, severity(s)) over synthetic states; argmax = policy."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    S = _sample_states(n, rng)
    sev = np.array([true_severity(s) for s in S], dtype=np.float32)
    ideal = np.array([tier_from_severity(v) for v in sev])
    Q = np.stack([[reward(a, v) for a in range(N_TIERS)] for v in sev]).astype(np.float32)

    nv = n // 5
    Str, Qtr = S[nv:], Q[nv:]
    Sva, ideal_va, sev_va = S[:nv], ideal[:nv], sev[:nv]

    net = QNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Str_t = torch.from_numpy(Str).to(device); Qtr_t = torch.from_numpy(Qtr).to(device)

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Str_t), device=device)
        for b in range(0, len(Str_t), batch_size):
            bi = perm[b:b + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Str_t[bi]), Qtr_t[bi])
            loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(Sva).to(device)).argmax(1).cpu().numpy()
    tier_acc = float((pred == ideal_va).mean())
    under = float((pred < ideal_va).mean())
    # baseline: the static M5 band reads tiers straight off combined safety score = 100*(1-sev)
    baseline = np.array([tier_from_severity(v) for v in sev_va])  # same mapping == oracle
    band_acc = float((baseline == ideal_va).mean())
    report = PolicyReport(len(Str), nv, tier_acc, under, band_acc)

    save_path = Path(save_path or config.AGENTIC_POLICY_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "n_features": len(VECTOR_FIELDS)}, save_path)
    log.info("saved policy -> %s (tier_acc=%.3f)", save_path, tier_acc)
    return net, report


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
@dataclass
class MemoryEntry:
    state: list
    action: int
    tier: str
    severity: float


class AgentMemory:
    """Short-term ring buffer (current drive) + persisted long-term store (learned cases)."""

    def __init__(self, short_capacity: int = 50, path: Path | None = None) -> None:
        self.short = deque(maxlen=short_capacity)
        self.long: list[MemoryEntry] = []
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self.load()

    def add(self, entry: MemoryEntry) -> None:
        self.short.append(entry)

    def consolidate(self) -> int:
        """Move the current drive's experiences into long-term memory."""
        n = len(self.short)
        self.long.extend(self.short)
        self.short.clear()
        return n

    def recall_similar(self, state: np.ndarray, k: int = 1) -> list[MemoryEntry]:
        if not self.long:
            return []
        M = np.array([e.state for e in self.long])
        d = np.linalg.norm(M - np.asarray(state)[None, :], axis=1)
        return [self.long[i] for i in np.argsort(d)[:k]]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(e) for e in self.long]))

    def load(self) -> None:
        self.long = [MemoryEntry(**e) for e in json.loads(self.path.read_text())]


# ---------------------------------------------------------------------------
# the reasoner
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    tier: int
    tier_name: str
    q_values: list
    severity_est: float
    shap: dict                       # feature -> attribution for the chosen tier
    explanation: str
    recalled: dict | None = None     # nearest past situation, if any


class AgenticReasoner:
    def __init__(self, policy_path: Path | None = None, net: QNet | None = None,
                 memory: AgentMemory | None = None, device: str | None = None,
                 background: np.ndarray | None = None, seed: int = 0) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if net is None:
            ckpt = torch.load(Path(policy_path or config.AGENTIC_POLICY_PATH),
                              map_location=self.device, weights_only=False)
            net = QNet(ckpt["n_features"])
            net.load_state_dict(ckpt["state_dict"])
        self.net = net.to(self.device).eval()
        self.memory = memory or AgentMemory()
        rng = np.random.default_rng(seed)
        self._background = background if background is not None else _sample_states(64, rng)
        self._explainer = None      # lazily built (shap)

    # -- inference -----------------------------------------------------------
    def _q(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
            return self.net(t).cpu().numpy()

    def _shap_values(self, state: np.ndarray, action: int) -> dict:
        import contextlib
        import io
        import shap
        if self._explainer is None:
            self._explainer = shap.KernelExplainer(self._q, self._background)
        # KernelExplainer can print internal arrays to stdout/stderr; keep both clean.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sv = self._explainer.shap_values(state[None, :], nsamples=100, silent=True)
        # normalize across shap versions -> (n_features,) for the chosen `action`.
        arr = np.squeeze(np.array(sv))         # drop the size-1 sample axis
        nf = len(VECTOR_FIELDS)
        if arr.ndim == 1:                      # single-output -> (n_features,)
            vec = arr
        elif arr.shape[0] == nf:               # (n_features, n_outputs)
            vec = arr[:, action]
        else:                                  # (n_outputs, n_features)
            vec = arr[action]
        return {f: float(v) for f, v in zip(VECTOR_FIELDS, vec)}

    def _explain(self, tier: int, shap_vals: dict) -> str:
        if tier == 0:
            return "No intervention — conditions nominal."
        top = sorted(shap_vals.items(), key=lambda kv: kv[1], reverse=True)[:2]
        drivers = [(_FEATURE_LABELS.get(f, f)) for f, v in top if v > 1e-4]
        head = f"{TIER_NAMES[tier].capitalize()} intervention"
        if not drivers:
            return f"{head} — no single dominant factor."
        return f"{head}, driven mainly by {' and '.join(drivers)}."

    def decide(self, summary: ScenarioSummary | np.ndarray, explain: bool = True,
               remember: bool = True) -> Decision:
        state = summary.to_vector() if isinstance(summary, ScenarioSummary) else np.asarray(
            summary, dtype=np.float32)
        q = self._q(state[None, :])[0]
        tier = int(np.argmax(q))
        sev = true_severity(state)
        shap_vals = self._shap_values(state, tier) if explain else {}
        explanation = self._explain(tier, shap_vals) if explain else TIER_NAMES[tier]

        recalled = None
        if remember:
            near = self.memory.recall_similar(state, k=1)
            if near:
                recalled = {"tier": near[0].tier, "severity": near[0].severity}
            self.memory.add(MemoryEntry(state.tolist(), tier, TIER_NAMES[tier], sev))

        return Decision(tier=tier, tier_name=TIER_NAMES[tier], q_values=q.tolist(),
                        severity_est=sev, shap=shap_vals, explanation=explanation,
                        recalled=recalled)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    net, rep = train_policy()
    print(f"\nPolicyReport: train={rep.n_train} val={rep.n_val} "
          f"tier_accuracy={rep.tier_accuracy:.3f} underreaction={rep.underreaction_rate:.3f}")

    # demo over nuScenes
    from sdiq.scenario_summary import ScenarioSummarizer
    from sdiq.data_loader import iter_nuscenes_scenes
    reasoner = AgenticReasoner(net=net)
    summ = ScenarioSummarizer()
    print("\n=== agentic decisions on nuScenes ===")
    for sc in iter_nuscenes_scenes():
        d = reasoner.decide(summ.summarize(sc))
        flags = ("N" if sc.is_night else "-") + ("R" if sc.is_rain else "-")
        print(f"  {sc.scene_id} [{flags}] -> {d.tier_name:12s} | {d.explanation}")
