"""Evaluation metrics for PRISM-AR vs. static AR baseline.

Metrics are defined per the PRISM-AR proposal:
- Under-warning rate
- Over-warning rate
- Warning lead time
- Minimum distance at crossing
- Cue flicker (tier changes per second)
- Visual clutter
- SHAP alignment
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy import stats

from prism_ar.data_ingestion.driving_scene import DrivingScene
from prism_ar.prism.risk_engine import PRISMOutput
from prism_ar.ar_overlay.cue_mapper import ARCue


@dataclass
class AREvaluation:
    """Evaluation result for a single scenario."""
    scene_id: str
    under_warning_rate: float          # deprecated: adaptive < static in high risk
    over_warning_rate: float           # deprecated: adaptive > static in low risk
    adaptive_under_warning_rate: float # adaptive fails to warn during high risk
    adaptive_over_warning_rate: float  # adaptive warns during low risk
    static_under_warning_rate: float   # static fails to warn during high risk
    static_over_warning_rate: float    # static warns during low risk
    gt_tier_distribution: Dict[str, int]  # ground-truth tier distribution
    adaptive_tier_accuracy: float       # fraction of frames matching ground-truth tier
    static_tier_accuracy: float
    adaptive_intervention_recall: float # fraction of gt intervention frames correctly flagged
    static_intervention_recall: float
    adaptive_emergency_recall: float
    static_emergency_recall: float
    cue_risk_monotonicity: float  # Spearman rho between cue opacity and PRISM score
    shap_alignment: float         # fraction of high-risk frames where dominant risk matches top_factor
    warning_lead_time_s: float
    min_distance_m: float
    cue_flicker_hz: float
    visual_clutter: float
    mean_score: float
    tier_distribution: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "scene_id": self.scene_id,
            "under_warning_rate": self.under_warning_rate,
            "over_warning_rate": self.over_warning_rate,
            "adaptive_under_warning_rate": self.adaptive_under_warning_rate,
            "adaptive_over_warning_rate": self.adaptive_over_warning_rate,
            "static_under_warning_rate": self.static_under_warning_rate,
            "static_over_warning_rate": self.static_over_warning_rate,
            "gt_tier_distribution": self.gt_tier_distribution,
            "adaptive_tier_accuracy": self.adaptive_tier_accuracy,
            "static_tier_accuracy": self.static_tier_accuracy,
            "adaptive_intervention_recall": self.adaptive_intervention_recall,
            "static_intervention_recall": self.static_intervention_recall,
            "adaptive_emergency_recall": self.adaptive_emergency_recall,
            "static_emergency_recall": self.static_emergency_recall,
            "cue_risk_monotonicity": self.cue_risk_monotonicity,
            "shap_alignment": self.shap_alignment,
            "warning_lead_time_s": self.warning_lead_time_s,
            "min_distance_m": self.min_distance_m,
            "cue_flicker_hz": self.cue_flicker_hz,
            "visual_clutter": self.visual_clutter,
            "mean_score": self.mean_score,
            "tier_distribution": self.tier_distribution,
        }


def _tier_level(tier: str) -> int:
    levels = {"silent": 0, "advisory": 1, "intervention": 2, "emergency": 3}
    return levels.get(tier, 0)


def compute_under_warning_rate(
    adaptive_cues: List[ARCue],
    static_cues: List[ARCue],
    prism_output: PRISMOutput,
) -> float:
    """Fraction of frames where adaptive cue is weaker than static cue despite high PRISM risk."""
    n = len(adaptive_cues)
    if n == 0:
        return 0.0
    high_risk_frames = prism_output.scores < 40.0
    count = 0
    denom = 0
    for i in range(n):
        if high_risk_frames[i]:
            denom += 1
            if _tier_level(adaptive_cues[i].tier) < _tier_level(static_cues[i].tier):
                count += 1
    return count / denom if denom > 0 else 0.0


def compute_over_warning_rate(
    adaptive_cues: List[ARCue],
    static_cues: List[ARCue],
    prism_output: PRISMOutput,
) -> float:
    """Fraction of frames where adaptive warns but static does not."""
    n = len(adaptive_cues)
    if n == 0:
        return 0.0
    low_risk_frames = prism_output.scores >= 70.0
    count = 0
    denom = 0
    for i in range(n):
        if low_risk_frames[i]:
            denom += 1
            if _tier_level(adaptive_cues[i].tier) > _tier_level(static_cues[i].tier):
                count += 1
    return count / denom if denom > 0 else 0.0


def compute_adaptive_under_warning_rate(
    adaptive_cues: List[ARCue],
    prism_output: PRISMOutput,
    high_risk_threshold: float = 40.0,
) -> float:
    """Fraction of high-risk frames where adaptive cue is advisory or lower."""
    n = len(adaptive_cues)
    if n == 0:
        return 0.0
    high_risk = prism_output.scores < high_risk_threshold
    denom = high_risk.sum()
    if denom == 0:
        return 0.0
    count = sum(1 for i in range(n) if high_risk[i] and _tier_level(adaptive_cues[i].tier) < 2)
    return count / denom


def compute_adaptive_over_warning_rate(
    adaptive_cues: List[ARCue],
    prism_output: PRISMOutput,
    low_risk_threshold: float = 70.0,
) -> float:
    """Fraction of low-risk frames where adaptive cue is intervention or higher."""
    n = len(adaptive_cues)
    if n == 0:
        return 0.0
    low_risk = prism_output.scores >= low_risk_threshold
    denom = low_risk.sum()
    if denom == 0:
        return 0.0
    count = sum(1 for i in range(n) if low_risk[i] and _tier_level(adaptive_cues[i].tier) > 1)
    return count / denom


def compute_static_under_warning_rate(
    static_cues: List[ARCue],
    prism_output: PRISMOutput,
    high_risk_threshold: float = 40.0,
) -> float:
    """Fraction of high-risk frames where static cue is advisory or lower."""
    n = len(static_cues)
    if n == 0:
        return 0.0
    high_risk = prism_output.scores < high_risk_threshold
    denom = high_risk.sum()
    if denom == 0:
        return 0.0
    count = sum(1 for i in range(n) if high_risk[i] and _tier_level(static_cues[i].tier) < 2)
    return count / denom


def compute_static_over_warning_rate(
    static_cues: List[ARCue],
    prism_output: PRISMOutput,
    low_risk_threshold: float = 70.0,
) -> float:
    """Fraction of low-risk frames where static cue is intervention or higher."""
    n = len(static_cues)
    if n == 0:
        return 0.0
    low_risk = prism_output.scores >= low_risk_threshold
    denom = low_risk.sum()
    if denom == 0:
        return 0.0
    count = sum(1 for i in range(n) if low_risk[i] and _tier_level(static_cues[i].tier) > 1)
    return count / denom


def compute_min_ttc_per_frame(scene: DrivingScene) -> np.ndarray:
    """Return per-frame minimum TTC from ego to any VRU."""
    from prism_ar.prism.risk_engine import _compute_ttc
    ego = scene.get_ego()
    if ego is None:
        return np.full(scene.frames, np.inf)
    ttc = np.full(scene.frames, np.inf)
    for vru in scene.get_vrus():
        ttc = np.minimum(ttc, _compute_ttc(ego, vru))
    return ttc


def compute_ground_truth_tiers(
    min_distances: np.ndarray,
    ttc: np.ndarray,
) -> List[str]:
    """Compute per-frame ground-truth tier labels from TTC and distance.

    Tiers are assigned by the most dangerous condition present, using
    thresholds that correspond to real near-miss conditions:
    - emergency: distance < 1m OR TTC < 0.5s
    - intervention: distance < 2m OR TTC < 1.0s
    - advisory: distance < 5m OR TTC < 2.5s
    - silent: otherwise
    """
    n = len(min_distances)
    tiers = []
    for i in range(n):
        d = min_distances[i]
        t = ttc[i] if ttc[i] is not None else np.inf
        if d < 1.0 or t < 0.5:
            tiers.append("emergency")
        elif d < 2.0 or t < 1.0:
            tiers.append("intervention")
        elif d < 5.0 or t < 2.5:
            tiers.append("advisory")
        else:
            tiers.append("silent")
    return tiers


def _tier_accuracy(cues: List[ARCue], gt_tiers: List[str]) -> float:
    """Fraction of frames where cue tier matches the ground-truth tier."""
    if not cues:
        return 0.0
    correct = sum(1 for c, gt in zip(cues, gt_tiers) if c.tier == gt)
    return correct / len(cues)


def _tier_recall(cues: List[ARCue], gt_tiers: List[str], target: str) -> float:
    """Fraction of ground-truth target frames where cue tier is at least target."""
    denom = sum(1 for gt in gt_tiers if gt == target)
    if denom == 0:
        return 0.0
    count = sum(1 for c, gt in zip(cues, gt_tiers) if gt == target and _tier_level(c.tier) >= _tier_level(target))
    return count / denom


def compute_cue_risk_monotonicity(
    cues: List[ARCue],
    scores: np.ndarray,
) -> float:
    """Spearman correlation between cue opacity and PRISM safety score.

    Higher opacity should correspond to lower safety score (higher risk).
    Returns negative rho for perfect monotonic increase of intensity with risk.
    """
    if len(cues) < 3 or len(scores) < 3:
        return 0.0
    opacities = np.array([c.opacity for c in cues])
    # Risk increases as score decreases; perfect monotonicity => rho = -1
    rho, _ = stats.spearmanr(opacities, scores)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_shap_alignment(prism_output: PRISMOutput) -> float:
    """Fraction of high-risk frames where the dominant risk component matches top_factor.

    The dominant risk component per frame is the max of env_risk, traj_risk[i],
    and vru_risk[i]. This measures whether the SHAP-style explanation
    (top_factor) is consistent with the actual drivers of high-risk frames.
    """
    n = len(prism_output.scores)
    if n == 0:
        return 0.0
    env = prism_output.env_risk
    traj = np.asarray(prism_output.traj_risk)
    vru = np.asarray(prism_output.vru_risk)
    high_risk = np.asarray(prism_output.scores) < 40.0
    denom = int(high_risk.sum())
    if denom == 0:
        return 0.0

    factor_map = {"environmental": 0, "trajectory": 1, "vru_proximity": 2}
    top_idx = factor_map.get(prism_output.top_factor, 0)
    per_frame = np.argmax(np.stack([np.full(n, env), traj, vru], axis=1), axis=1)
    aligned = int((per_frame[high_risk] == top_idx).sum())
    return aligned / denom


def compute_warning_lead_time(
    adaptive_cues: List[ARCue],
    timestamps: np.ndarray,
) -> float:
    """Seconds between first red-tier cue and end of scene."""
    n = len(adaptive_cues)
    if n == 0:
        return 0.0
    for i in range(n):
        if adaptive_cues[i].tier in ("intervention", "emergency"):
            return timestamps[-1] - timestamps[i]
    return 0.0


def compute_cue_flicker(cues: List[ARCue], duration_s: float) -> float:
    """Number of tier changes per second."""
    if duration_s <= 0:
        return 0.0
    changes = sum(1 for i in range(1, len(cues)) if cues[i].tier != cues[i - 1].tier)
    return changes / duration_s


def compute_visual_clutter(cues: List[ARCue]) -> float:
    """Average opacity of overlay elements."""
    if not cues:
        return 0.0
    return float(np.mean([c.opacity for c in cues]))


def evaluate_scenario(
    scene_id: str,
    adaptive_cues: List[ARCue],
    static_cues: List[ARCue],
    prism_output: PRISMOutput,
    timestamps: np.ndarray,
    min_distances: np.ndarray,
    min_ttc: np.ndarray,
) -> AREvaluation:
    """Compute full evaluation metrics for one scenario."""
    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    gt_tiers = compute_ground_truth_tiers(min_distances, min_ttc)
    return AREvaluation(
        scene_id=scene_id,
        under_warning_rate=compute_under_warning_rate(adaptive_cues, static_cues, prism_output),
        over_warning_rate=compute_over_warning_rate(adaptive_cues, static_cues, prism_output),
        adaptive_under_warning_rate=compute_adaptive_under_warning_rate(adaptive_cues, prism_output),
        adaptive_over_warning_rate=compute_adaptive_over_warning_rate(adaptive_cues, prism_output),
        static_under_warning_rate=compute_static_under_warning_rate(static_cues, prism_output),
        static_over_warning_rate=compute_static_over_warning_rate(static_cues, prism_output),
        gt_tier_distribution={tier: gt_tiers.count(tier) for tier in set(gt_tiers)},
        adaptive_tier_accuracy=_tier_accuracy(adaptive_cues, gt_tiers),
        static_tier_accuracy=_tier_accuracy(static_cues, gt_tiers),
        adaptive_intervention_recall=_tier_recall(adaptive_cues, gt_tiers, "intervention"),
        static_intervention_recall=_tier_recall(static_cues, gt_tiers, "intervention"),
        adaptive_emergency_recall=_tier_recall(adaptive_cues, gt_tiers, "emergency"),
        static_emergency_recall=_tier_recall(static_cues, gt_tiers, "emergency"),
        cue_risk_monotonicity=compute_cue_risk_monotonicity(adaptive_cues, prism_output.scores),
        shap_alignment=compute_shap_alignment(prism_output),
        warning_lead_time_s=compute_warning_lead_time(adaptive_cues, timestamps),
        min_distance_m=float(np.min(min_distances)) if min_distances.size else 0.0,
        cue_flicker_hz=compute_cue_flicker(adaptive_cues, duration_s),
        visual_clutter=compute_visual_clutter(adaptive_cues),
        mean_score=float(np.mean(prism_output.scores)),
        tier_distribution={tier: prism_output.tiers.count(tier) for tier in set(prism_output.tiers)},
    )
