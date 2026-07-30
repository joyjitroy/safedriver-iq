"""M5 — per-cycle scenario state: the integration contract for the agentic layer.

Runs the three Layer-2 models on a DrivingScene and fuses their outputs into ONE small,
stable `ScenarioSummary`:

    env (RF bridge)        -> safety_score, env_risk, env_multiplier
    trajectory (LSTM)      -> trajectory_risk (max / mean over agents)
    VRU (SFM + LSTM)       -> vru_risk, near_misses, min_distance, min_ttc

The same object feeds BOTH downstream consumers:
  * `to_vector()` -> fixed-length float array for the RL agent (M6).
  * `to_dict()`   -> JSON-friendly dict for the LLM co-pilot (M7).

Fusion follows the ASCE2027 design: the environmental model is a *context multiplier*,
not an equal vote — it amplifies the trajectory/VRU risk in hostile conditions (night,
rain) and damps it in benign ones. The combined number here is a transparent heuristic
baseline; the RL agent (M6) learns the actual policy on top of the same state vector.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

from sdiq import config
from sdiq.data_loader import DrivingScene
from sdiq.config import VRU_RISK


# ---------------------------------------------------------------------------
# the structured state
# ---------------------------------------------------------------------------
# Stable order of the RL state vector — DO NOT reorder (M6 depends on it).
VECTOR_FIELDS = (
    "env_risk", "env_multiplier", "trajectory_risk", "vru_risk",
    "proximity", "imminence", "is_night", "is_rain",
)


@dataclass
class ScenarioSummary:
    scene_id: str
    source: str
    # scene meta
    is_night: bool
    is_rain: bool
    n_agents: int
    n_vru: int
    # environmental model (RF bridge)
    env_safety_score: float       # 0-100 (higher = safer)
    env_risk: float               # 0-1
    env_multiplier: float         # [ENV_MULT_MIN, ENV_MULT_MAX]
    env_risk_level: str
    # trajectory kinematics (LSTM)
    trajectory_risk: float        # 0-1 (worst agent)
    trajectory_mean_risk: float   # 0-1
    # VRU interaction (SFM + LSTM)
    vru_risk: float               # 0-1 (conservative collision-course)
    vru_realistic_risk: float     # 0-1 (SFM reaction-aware, for context)
    vru_near_misses: int
    min_distance: float           # m (closest ego-VRU approach; inf if no VRU)
    min_ttc: float                # s (time to that approach; inf if no VRU)
    # fused (heuristic baseline; RL refines)
    fused_risk: float             # 0-1
    combined_safety_score: float  # 0-100

    @property
    def band(self) -> str:
        """Graduated intervention band (ASCE2027 thresholds on the 0-100 safety score).
        Baseline only — the RL agent (M6) makes the real call."""
        s = self.combined_safety_score
        if s >= 70:
            return "silent"
        if s >= 40:
            return "advisory"
        if s >= 20:
            return "intervention"
        return "emergency"

    # -- consumers ---------------------------------------------------------
    def to_vector(self):
        """Fixed-length feature vector for the RL agent (order = VECTOR_FIELDS)."""
        import numpy as np
        # proximity/imminence are normalized, bounded surrogates of the raw m/s values.
        proximity = _clip01(1.0 - self.min_distance / VRU_RISK.d_warn) if math.isfinite(self.min_distance) else 0.0
        imminence = _clip01(1.0 - self.min_ttc / VRU_RISK.ttc_max) if math.isfinite(self.min_ttc) else 0.0
        vals = {
            "env_risk": self.env_risk,
            "env_multiplier": self.env_multiplier,
            "trajectory_risk": self.trajectory_risk,
            "vru_risk": self.vru_risk,
            "proximity": proximity,
            "imminence": imminence,
            "is_night": float(self.is_night),
            "is_rain": float(self.is_rain),
        }
        return np.array([vals[k] for k in VECTOR_FIELDS], dtype="float32")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band"] = self.band
        # JSON can't hold inf; surface as None.
        for k in ("min_distance", "min_ttc"):
            if not math.isfinite(d[k]):
                d[k] = None
        return d

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ---------------------------------------------------------------------------
# the summarizer (loads the three models once, reuses across scenes)
# ---------------------------------------------------------------------------
class ScenarioSummarizer:
    def __init__(self, env_bridge=None, kinematic_model=None, vru_model=None,
                 traj_weight: float = 0.5, vru_weight: float = 1.0) -> None:
        """Models are lazily loaded if not injected. `vru_weight > traj_weight` reflects
        that VRU conflicts are the safety-critical case (the Phase 1 blind spot)."""
        self._env = env_bridge
        self._kin = kinematic_model
        self._vru = vru_model
        self.traj_weight = traj_weight
        self.vru_weight = vru_weight

    @property
    def env(self):
        if self._env is None:
            from sdiq.safedriver_iq_bridge import EnvRiskBridge
            self._env = EnvRiskBridge()
        return self._env

    @property
    def kin(self):
        if self._kin is None:
            from sdiq.kinematic_features import KinematicRiskModel
            self._kin = KinematicRiskModel()
        return self._kin

    @property
    def vru(self):
        if self._vru is None:
            from sdiq.vru_features import VRURiskModel
            self._vru = VRURiskModel()
        return self._vru

    def summarize(self, scene: DrivingScene, parallel: bool = False) -> ScenarioSummary:
        # The three Layer-2 models are independent — run them concurrently when asked
        # ("3 models in parallel every cycle", per the ASCE2027 design). torch and numpy
        # release the GIL during the heavy work, so threads give a real speedup.
        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as ex:
                f_env = ex.submit(self.env.assess, scene)
                f_kin = ex.submit(self.kin.score_scene, scene)
                f_vru = ex.submit(self.vru.assess_scene, scene)
                env, kin, vru = f_env.result(), f_kin.result(), f_vru.result()
        else:
            env = self.env.assess(scene)
            kin = self.kin.score_scene(scene)
            vru = self.vru.assess_scene(scene)

        # base dynamic risk = weighted blend of trajectory and VRU risk, VRU-dominant.
        w = self.traj_weight + self.vru_weight
        base = (self.traj_weight * kin["max_risk"] + self.vru_weight * vru["max_risk"]) / w
        # environmental context multiplier amplifies / damps the dynamic risk.
        fused = _clip01(env.multiplier * base)
        combined_score = 100.0 * (1.0 - fused)

        return ScenarioSummary(
            scene_id=scene.scene_id, source=scene.source,
            is_night=scene.is_night, is_rain=scene.is_rain,
            n_agents=len(scene.agents), n_vru=len(scene.vru_agents),
            env_safety_score=env.safety_score, env_risk=env.env_risk,
            env_multiplier=env.multiplier, env_risk_level=env.risk_level,
            trajectory_risk=kin["max_risk"], trajectory_mean_risk=kin["mean_risk"],
            vru_risk=vru["max_risk"], vru_realistic_risk=vru["realistic_max_risk"],
            vru_near_misses=vru["near_misses"],
            min_distance=vru["min_distance"], min_ttc=vru["ttc"],
            fused_risk=fused, combined_safety_score=combined_score,
        )


if __name__ == "__main__":
    from sdiq.data_loader import iter_nuscenes_scenes

    summ = ScenarioSummarizer()
    print(f"{'scene':12s} {'flags':5s} {'env_mult':>8s} {'traj':>5s} {'vru':>5s} "
          f"{'fused':>5s} {'score':>6s} {'band':>12s}")
    for sc in iter_nuscenes_scenes():
        s = summ.summarize(sc)
        flags = ("N" if s.is_night else "-") + ("R" if s.is_rain else "-")
        print(f"{s.scene_id:12s} {flags:5s} {s.env_multiplier:8.3f} {s.trajectory_risk:5.2f} "
              f"{s.vru_risk:5.2f} {s.fused_risk:5.2f} {s.combined_safety_score:6.1f} {s.band:>12s}")
