"""Central configuration for the Phase 2 POC.

Single source of truth for dataset roots, the frozen Phase 1 model artifacts, and
the LLM co-pilot provider. Everything resolves relative to this file's location so
the POC is portable regardless of where the repo is checked out. Any value can be
overridden via an environment variable (see `os.environ` lookups below) so the same
code runs on a workstation, in CI, or in a headless cron job without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo geometry
# ---------------------------------------------------------------------------
# .../safedriver-iq/phase2-agentic-multimodel/src/sdiq/config.py
PKG_DIR = Path(__file__).resolve().parent          # src/sdiq/
PROJECT_ROOT = PKG_DIR.parent.parent               # phase2-agentic-multimodel/
PHASE1_REPO = PROJECT_ROOT.parent                  # safedriver-iq/  (Phase 1 root)
DATASETS_DIR = PROJECT_ROOT / "datasets"


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


# ---------------------------------------------------------------------------
# Dataset roots (consolidated under phase2-agentic-multimodel/datasets/)
# ---------------------------------------------------------------------------
# nuScenes mini: the *dataroot* is the directory that CONTAINS the version folder
# plus samples/sweeps/maps — i.e. datasets/nuscenes-mini/.
NUSCENES_DATAROOT = _env_path("SDIQ_NUSCENES_DATAROOT", DATASETS_DIR / "nuscenes-mini")
NUSCENES_VERSION = os.environ.get("SDIQ_NUSCENES_VERSION", "v1.0-mini")
NUSCENES_TABLES = NUSCENES_DATAROOT / NUSCENES_VERSION

# Argoverse 2 motion-forecasting validation split: directory of per-scenario folders.
AV2_VAL_ROOT = _env_path("SDIQ_AV2_VAL_ROOT", DATASETS_DIR / "argoverse2-val")

# Phase 1 tabular datasets (referenced, not duplicated here).
CRSS_DATA_DIR = _env_path("SDIQ_CRSS_DIR", PHASE1_REPO / "CRSS_Data")
WAYMO_DATA_DIR = _env_path("SDIQ_WAYMO_DIR", PHASE1_REPO / "waymo")

# ---------------------------------------------------------------------------
# Frozen Phase 1 model artifacts (the env-risk RF bridge)
# ---------------------------------------------------------------------------
# NOTE (M0 finding): the ASCE2027 docx names `results/crash_investigation_rf_model.pkl`
# as the "64-feature" bridge, but that artifact is actually a 5-feature investigation
# model. The real 64-feature production RandomForest — the one matching the 64 names in
# feature_names.txt and the docx's "64-feature vector" — is `best_safety_model.pkl`.
# The bridge (M2) therefore targets best_safety_model.pkl. Both load via joblib.load().
RF_MODEL_PATH = _env_path(
    "SDIQ_RF_MODEL", PHASE1_REPO / "results" / "models" / "best_safety_model.pkl"
)
RF_FEATURE_NAMES_PATH = _env_path(
    "SDIQ_RF_FEATURES", PHASE1_REPO / "results" / "models" / "feature_names.txt"
)
# The smaller 5-feature investigation model, kept for reference / ablation.
RF_INVESTIGATION_MODEL_PATH = _env_path(
    "SDIQ_RF_INVESTIGATION_MODEL",
    PHASE1_REPO / "results" / "crash_investigation_rf_model.pkl",
)

# Where the POC writes its own trained models / artifacts.
ARTIFACTS_DIR = _env_path("SDIQ_ARTIFACTS", PROJECT_ROOT / "artifacts")

# ---------------------------------------------------------------------------
# Env-risk bridge (M2): map the Phase 1 safety score (0-100) to a context
# multiplier that AMPLIFIES the trajectory/VRU models in risky environments.
#   safety_score 100 (safest) -> ENV_MULT_MIN   (dampen)
#   safety_score  50          -> ~1.0           (neutral)
#   safety_score   0 (riskiest)-> ENV_MULT_MAX   (amplify)
# ---------------------------------------------------------------------------
ENV_MULT_MIN = float(os.environ.get("SDIQ_ENV_MULT_MIN", "0.5"))
ENV_MULT_MAX = float(os.environ.get("SDIQ_ENV_MULT_MAX", "1.5"))


# ---------------------------------------------------------------------------
# M3 — Trajectory kinematics & LSTM risk model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KinematicThresholds:
    """Comfort/aggression thresholds (vehicle-dynamics literature, SI units).

    Each is the point where a behaviour starts being "unsafe"; the per-step risk
    heuristic measures normalized exceedance above these. ~0.3 g hard braking,
    ~0.4 g aggressive lateral, etc. Calibrated against the Argoverse 2 distribution
    so normal urban driving scores ~0 and only the aggressive tail is flagged
    (normal-driving |lon_accel| p90≈2.3, p99≈5.3 m/s^2 after smoothing).
    """
    hard_brake: float = 3.0     # m/s^2 deceleration (lon_accel < -3)
    hard_accel: float = 3.0     # m/s^2 acceleration
    lateral: float = 4.0        # m/s^2 |lateral accel| (swerve / sharp turn)
    speed: float = 16.0         # m/s (~58 km/h) — elevated for an urban scene
    scale: float = 4.0          # exceedance normalizer (saturates the heuristic)


KINEMATICS = KinematicThresholds()

# Smoothing window (seconds) for the Savitzky-Golay derivatives. Converted to an odd
# sample count per-track from its sample rate, so 10 Hz (AV2) and 2 Hz (nuScenes) both
# use a ~2 s physical window. Denoises accel without erasing real braking events.
KIN_SMOOTH_SECONDS = float(os.environ.get("SDIQ_KIN_SMOOTH_SECONDS", "2.0"))

# LSTM trajectory-risk model hyperparameters.
KIN_FEATURES = ("speed", "lon_accel", "lat_accel", "yaw_rate")
KIN_HISTORY = int(os.environ.get("SDIQ_KIN_HISTORY", "20"))     # input steps (2s @10Hz)
KIN_HORIZON = int(os.environ.get("SDIQ_KIN_HORIZON", "20"))     # anticipation steps ahead
KIN_HIDDEN = int(os.environ.get("SDIQ_KIN_HIDDEN", "64"))
KIN_LAYERS = int(os.environ.get("SDIQ_KIN_LAYERS", "2"))
KIN_MODEL_PATH = _env_path("SDIQ_KIN_MODEL", ARTIFACTS_DIR / "kinematic_lstm.pt")


# ---------------------------------------------------------------------------
# M4 — VRU interaction: Social Force Model + LSTM correction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SFMParams:
    """Helbing & Molnár (1995) social-force parameters (SI units).

    A pedestrian accelerates toward its desired velocity (relaxation time `tau`) while
    being repelled exponentially by other agents — most strongly by the ego vehicle,
    which is bigger and faster (`a_ego`/`b_ego` >> ped-ped). Physics only, no training.
    """
    tau: float = 0.5            # relaxation time (s)
    a_ped: float = 2.0          # ped<-ped repulsion strength (m/s^2)
    b_ped: float = 0.8          # ped<-ped repulsion range (m)
    a_ego: float = 10.0         # ped<-ego(vehicle) repulsion strength (m/s^2)
    b_ego: float = 2.5          # ped<-ego repulsion range (m)
    radius_ped: float = 0.3     # VRU radius (m)
    radius_ego: float = 1.6     # ego footprint radius (m)
    v_max: float = 3.5          # VRU max speed cap (m/s)


SFM = SFMParams()


@dataclass(frozen=True)
class VRURiskThresholds:
    """Ego-VRU interaction-risk thresholds (closest approach over the forecast)."""
    d_critical: float = 1.0     # m — distance at/under which risk saturates to 1
    d_warn: float = 6.0         # m — distance at/above which proximity risk is 0
    d_nearmiss: float = 2.5     # m — near-miss flag ("uncomfortably close pass")
    ttc_max: float = 4.0        # s — beyond this time-to-closest-approach, not imminent


VRU_RISK = VRURiskThresholds()

# SFM rollout/forecast horizons are in SECONDS (rate-agnostic; converted per scene).
VRU_HISTORY_S = float(os.environ.get("SDIQ_VRU_HISTORY_S", "1.0"))
VRU_HORIZON_S = float(os.environ.get("SDIQ_VRU_HORIZON_S", "2.0"))

# LSTM forecast-correction (trained on Argoverse 2 @10 Hz; nuScenes uses SFM-only).
VRU_HISTORY = int(os.environ.get("SDIQ_VRU_HISTORY", "10"))     # steps (1s @10Hz)
VRU_HORIZON = int(os.environ.get("SDIQ_VRU_HORIZON", "20"))     # steps (2s @10Hz)
VRU_HIDDEN = int(os.environ.get("SDIQ_VRU_HIDDEN", "64"))
VRU_LAYERS = int(os.environ.get("SDIQ_VRU_LAYERS", "2"))
VRU_MODEL_PATH = _env_path("SDIQ_VRU_MODEL", ARTIFACTS_DIR / "vru_lstm.pt")


# ---------------------------------------------------------------------------
# M6 — Agentic reasoning layer (RL policy + memory)
# ---------------------------------------------------------------------------
AGENTIC_POLICY_PATH = _env_path("SDIQ_AGENTIC_POLICY", ARTIFACTS_DIR / "agentic_policy.pt")
AGENT_MEMORY_PATH = _env_path("SDIQ_AGENT_MEMORY", ARTIFACTS_DIR / "agent_memory.json")


# ---------------------------------------------------------------------------
# LLM co-pilot (Hybrid scope) — provider-agnostic, OFF the safety-critical path
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    # "claude" -> Anthropic API; "local" -> llama.cpp OpenAI-compatible server;
    # "off" -> disable the co-pilot entirely (classical SHAP explanations only).
    provider: str = os.environ.get("SDIQ_LLM_PROVIDER", "off")

    # Claude path. Synthesis model (Opus 4.8) + a cheap model (Haiku 4.5) for the
    # short per-scenario summaries/explanations. Bare aliases, per the Anthropic SDK.
    claude_model: str = os.environ.get("SDIQ_CLAUDE_MODEL", "claude-opus-4-8")
    claude_cheap_model: str = os.environ.get(
        "SDIQ_CLAUDE_CHEAP_MODEL", "claude-haiku-4-5"
    )

    # Local llama.cpp server (OpenAI-compatible /v1/chat/completions).
    local_base_url: str = os.environ.get(
        "SDIQ_LLM_BASE_URL", "http://127.0.0.1:8080/v1"
    )
    local_model: str = os.environ.get("SDIQ_LLM_LOCAL_MODEL", "local-model")

    # Hard guardrail: the co-pilot must never stall the loop.
    timeout_s: float = float(os.environ.get("SDIQ_LLM_TIMEOUT_S", "8"))


LLM = LLMConfig()


@dataclass
class Paths:
    """Convenience bundle passed around the POC."""
    nuscenes_dataroot: Path = NUSCENES_DATAROOT
    nuscenes_version: str = NUSCENES_VERSION
    av2_val_root: Path = AV2_VAL_ROOT
    crss_dir: Path = CRSS_DATA_DIR
    waymo_dir: Path = WAYMO_DATA_DIR
    rf_model: Path = RF_MODEL_PATH
    rf_features: Path = RF_FEATURE_NAMES_PATH
    artifacts: Path = ARTIFACTS_DIR


PATHS = Paths()


def validate(require_phase1_model: bool = True) -> list[str]:
    """Return a list of human-readable problems with the current config.

    Empty list == everything the POC needs is present. Used by the sanity tests
    and `main.py` startup so misconfigured paths fail loudly and early.
    """
    problems: list[str] = []
    if not NUSCENES_TABLES.is_dir():
        problems.append(f"nuScenes tables not found: {NUSCENES_TABLES}")
    if not AV2_VAL_ROOT.is_dir():
        problems.append(f"Argoverse 2 val root not found: {AV2_VAL_ROOT}")
    if require_phase1_model and not RF_MODEL_PATH.is_file():
        problems.append(f"Phase 1 RF model not found: {RF_MODEL_PATH}")
    return problems


if __name__ == "__main__":
    print("SafeDriver-IQ Phase 2 POC configuration")
    print(f"  nuScenes dataroot : {NUSCENES_DATAROOT}  (version={NUSCENES_VERSION})")
    print(f"  AV2 val root      : {AV2_VAL_ROOT}")
    print(f"  RF model          : {RF_MODEL_PATH}")
    print(f"  LLM provider      : {LLM.provider}")
    issues = validate()
    print("  config OK" if not issues else "  PROBLEMS:\n    " + "\n    ".join(issues))
