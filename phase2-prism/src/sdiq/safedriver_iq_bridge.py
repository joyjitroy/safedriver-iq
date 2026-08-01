"""M2 — bridge to the frozen Phase 1 environmental-risk model.

Layer-2, model #1 of the agentic architecture. We do NOT retrain anything: we reuse
the Phase 1 64-feature RandomForest (via its own `RealtimeSafetyCalculator`, so the
feature engineering exactly matches how the model was trained) and turn its 0-100
"inverse crash" safety score into a **context multiplier** for the trajectory and VRU
models. Per the ASCE2027 design the environmental model is a multiplier, not an equal
vote: a hostile environment (night, rain, dense VRUs) should amplify the kinematic/VRU
risk, a benign one should damp it.

Mapping (linear, configurable via ENV_MULT_MIN/MAX in config):
    safety_score 100 (safest)  -> multiplier ENV_MULT_MIN (0.5, damp)
    safety_score  50           -> multiplier ~1.0         (neutral)
    safety_score   0 (riskiest)-> multiplier ENV_MULT_MAX (1.5, amplify)

Reuse note: Phase 1 imports itself as `src.*` and `models.py` does a bare
`import gpu_config`, so both the repo root and its `src/` dir must be on sys.path.
`_ensure_phase1_importable()` handles that once and quiets Phase 1's chatty INFO logs.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sdiq import config
from sdiq.data_loader import DrivingScene

_PHASE1_READY = False


def _ensure_phase1_importable(repo: Path | None = None) -> None:
    """Put the Phase 1 repo (and its src/) on sys.path; quiet its loggers. Idempotent."""
    global _PHASE1_READY
    if _PHASE1_READY:
        return
    repo = Path(repo or config.PHASE1_REPO)
    for p in (repo, repo / "src"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    # Phase 1 logs "Created temporal features" etc. on every scoring call.
    for name in ("src", "src.realtime_calculator", "src.feature_engineering", "gpu_config"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _PHASE1_READY = True


@dataclass
class EnvRisk:
    """Output of the bridge for one scene/context."""
    safety_score: float            # Phase 1 inverse-crash score, 0-100 (higher = safer)
    risk_level: str                # Phase 1 label: Low/Medium/High/Critical
    env_risk: float                # normalized risk in [0, 1] (1 = riskiest)
    multiplier: float              # context multiplier in [ENV_MULT_MIN, ENV_MULT_MAX]
    scenario: dict[str, Any] = field(default_factory=dict)  # features fed to the RF


# CRSS code conventions used when synthesizing the scenario dict.
_WEATHER_CLEAR, _WEATHER_RAIN = 1, 2
_LGT_DAYLIGHT, _LGT_DARK_LIGHTED, _LGT_DARK_UNLIT = 1, 2, 3


class EnvRiskBridge:
    """Wraps the frozen Phase 1 RF and exposes a scene -> env-risk multiplier API."""

    def __init__(self,
                 model_path: Path | None = None,
                 feature_names_path: Path | None = None,
                 mult_min: float | None = None,
                 mult_max: float | None = None,
                 phase1_repo: Path | None = None) -> None:
        _ensure_phase1_importable(phase1_repo)
        from src.realtime_calculator import RealtimeSafetyCalculator  # noqa: E402

        self.model_path = Path(model_path or config.RF_MODEL_PATH)
        self.feature_names_path = Path(feature_names_path or config.RF_FEATURE_NAMES_PATH)
        self.mult_min = mult_min if mult_min is not None else config.ENV_MULT_MIN
        self.mult_max = mult_max if mult_max is not None else config.ENV_MULT_MAX
        self._calc = RealtimeSafetyCalculator(
            str(self.model_path), str(self.feature_names_path))
        self.feature_names: list[str] = list(self._calc.feature_names)

    # -- scene -> CRSS-style scenario dict ---------------------------------
    @staticmethod
    def scene_to_scenario(scene: DrivingScene) -> dict[str, Any]:
        """Derive the environmentally-meaningful RF inputs from a DrivingScene.

        Only the features that have a real scene equivalent are set; the Phase 1
        calculator fills the rest (survey/crash-outcome codes) with neutral defaults.
        VRU counts use CRSS semantics (VRU = pedestrians + cyclists); motorcyclists are
        powered two-wheelers (vehicle-class in CRSS) and are handled by the M4 VRU model,
        not counted here.
        """
        ped = sum(1 for a in scene.agents if a.object_type == "pedestrian")
        cyc = sum(1 for a in scene.agents if a.object_type == "cyclist")
        night = bool(scene.is_night)
        rain = bool(scene.is_rain)
        return {
            # time-of-day: scenes carry no wall-clock, so use a night/day proxy hour.
            "HOUR": 23 if night else 14,
            "DAY_WEEK": 1,                       # unknown -> neutral
            "IS_NIGHT": int(night),
            "NIGHT_AND_DARK": int(night),
            "POOR_LIGHTING": int(night),
            "LGT_COND": _LGT_DARK_UNLIT if night else _LGT_DAYLIGHT,
            "WEATHER": _WEATHER_RAIN if rain else _WEATHER_CLEAR,
            "WEATHER1": _WEATHER_RAIN if rain else _WEATHER_CLEAR,
            "ADVERSE_WEATHER": int(rain),
            "ADVERSE_CONDITIONS": int(night or rain),
            "pedestrian_count": ped,
            "cyclist_count": cyc,
            "total_vru": ped + cyc,
        }

    # -- core scoring ------------------------------------------------------
    def _score_to_risk(self, safety_score: float) -> tuple[float, float]:
        env_risk = max(0.0, min(1.0, (100.0 - safety_score) / 100.0))
        multiplier = self.mult_min + (self.mult_max - self.mult_min) * env_risk
        return env_risk, multiplier

    def assess_scenario(self, scenario: dict[str, Any]) -> EnvRisk:
        out = self._calc.calculate_safety_score(scenario)
        score = float(out["safety_score"])
        env_risk, multiplier = self._score_to_risk(score)
        return EnvRisk(
            safety_score=score,
            risk_level=str(out.get("risk_level", "")),
            env_risk=env_risk,
            multiplier=multiplier,
            scenario=scenario,
        )

    def assess(self, scene: DrivingScene) -> EnvRisk:
        return self.assess_scenario(self.scene_to_scenario(scene))

    def multiplier(self, scene: DrivingScene) -> float:
        return self.assess(scene).multiplier


if __name__ == "__main__":
    from sdiq.data_loader import iter_nuscenes_scenes

    bridge = EnvRiskBridge()
    print(f"bridge model: {bridge.model_path.name} | {len(bridge.feature_names)} features "
          f"| multiplier range [{bridge.mult_min}, {bridge.mult_max}]\n")
    for scene in iter_nuscenes_scenes():
        r = bridge.assess(scene)
        flags = "".join(["N" if scene.is_night else "-", "R" if scene.is_rain else "-"])
        print(f"  {scene.scene_id} [{flags}] vru={len(scene.vru_agents):3d} "
              f"-> score={r.safety_score:5.1f} ({r.risk_level:8s}) "
              f"env_risk={r.env_risk:.2f} mult={r.multiplier:.3f}")
