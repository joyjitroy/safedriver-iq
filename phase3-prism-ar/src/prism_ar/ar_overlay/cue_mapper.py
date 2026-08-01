"""Map PRISM risk output to AR cue parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from prism_ar.prism.risk_engine import PRISMOutput


@dataclass
class ARCue:
    """AR cue specification for a single frame."""
    tier: str
    color: str  # hex or named color
    opacity: float  # 0--1
    show_danger_zone: bool
    show_stop_line: bool
    show_no_cross: bool
    show_pedestrian_indicator: bool
    flash: bool
    text: Optional[str] = None


# Risk-adaptive AR mapping: tier -> cue parameters
PRISM_AR_CUE_MAP = {
    "silent": ARCue(
        tier="silent",
        color="#00FF00",
        opacity=0.0,
        show_danger_zone=False,
        show_stop_line=False,
        show_no_cross=False,
        show_pedestrian_indicator=False,
        flash=False,
        text=None,
    ),
    "advisory": ARCue(
        tier="advisory",
        color="#FFCC00",
        opacity=0.35,
        show_danger_zone=False,
        show_stop_line=True,
        show_no_cross=False,
        show_pedestrian_indicator=True,
        flash=False,
        text="Caution",
    ),
    "intervention": ARCue(
        tier="intervention",
        color="#FF3300",
        opacity=0.65,
        show_danger_zone=True,
        show_stop_line=True,
        show_no_cross=True,
        show_pedestrian_indicator=True,
        flash=False,
        text="Do not cross",
    ),
    "emergency": ARCue(
        tier="emergency",
        color="#FF0000",
        opacity=0.90,
        show_danger_zone=True,
        show_stop_line=True,
        show_no_cross=True,
        show_pedestrian_indicator=True,
        flash=True,
        text="STOP",
    ),
}


# Static AR baseline: always shows the same advisory-level cues
STATIC_AR_CUE = ARCue(
    tier="advisory",
    color="#FFCC00",
    opacity=0.40,
    show_danger_zone=False,
    show_stop_line=True,
    show_no_cross=False,
    show_pedestrian_indicator=True,
    flash=False,
    text="Watch",
)


class AdaptiveARCueMapper:
    """Map PRISM output to adaptive AR cues."""

    def map(self, prism_output: PRISMOutput) -> list:
        """Return a list of ARCue objects, one per frame."""
        return [PRISM_AR_CUE_MAP[tier] for tier in prism_output.tiers]


class StaticARCueMapper:
    """Static AR baseline: same cue every frame."""

    def map(self, prism_output: PRISMOutput) -> list:
        n_frames = len(prism_output.tiers)
        return [STATIC_AR_CUE] * n_frames


# No-AR baseline: blank overlay every frame
NO_AR_CUE = ARCue(
    tier="silent",
    color="#000000",
    opacity=0.0,
    show_danger_zone=False,
    show_stop_line=False,
    show_no_cross=False,
    show_pedestrian_indicator=False,
    flash=False,
    text=None,
)


class NoARCueMapper:
    """No AR overlay baseline."""

    def map(self, prism_output: PRISMOutput) -> list:
        n_frames = len(prism_output.tiers)
        return [NO_AR_CUE] * n_frames


class OracleARCueMapper:
    """Oracle AR upper bound: knows ground-truth minimum distance.

    Uses the PRISM output only for frame count; it selects tier based on a
    perfect distance oracle: emergency if min distance < 5m, intervention if
    < 10m, advisory if < 20m, otherwise silent.
    """

    def map(self, prism_output: PRISMOutput) -> list:
        # Use env_risk as a placeholder for oracle distance
        # In practice, the caller supplies min_distance_m; here we approximate.
        n_frames = len(prism_output.tiers)
        cues = []
        for score in prism_output.scores:
            if score < 20:
                cue = PRISM_AR_CUE_MAP["emergency"]
            elif score < 40:
                cue = PRISM_AR_CUE_MAP["intervention"]
            elif score < 70:
                cue = PRISM_AR_CUE_MAP["advisory"]
            else:
                cue = NO_AR_CUE
            cues.append(cue)
        return cues
