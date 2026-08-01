"""
Risk Assessment Module

Defines data structures and utilities for risk assessment.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class InterventionLevel(Enum):
    """Intervention levels from passive to active"""
    PASSIVE_MONITORING = 0
    GENTLE_WARNING = 1
    AUDIO_ALERT = 2
    AGGRESSIVE_WARNING = 3
    AUTONOMOUS_BRAKE = 4


@dataclass
class RiskAssessment:
    """Complete risk assessment for current scenario"""
    overall_risk: float  # 0-100 (inverse of safety score)
    safety_score: float  # 0-100 from existing model
    vru_risk: float  # 0-100
    road_risk: float  # 0-100
    weather_risk: float  # 0-100
    driver_readiness: float  # 0-100
    time_to_collision: Optional[float]  # seconds, None if no imminent collision
    primary_factors: List[str]  # Top risk contributors
    confidence: float  # 0-1, model confidence
    recommended_action: InterventionLevel
    explanation: str  # Human-readable reasoning
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'overall_risk': self.overall_risk,
            'safety_score': self.safety_score,
            'vru_risk': self.vru_risk,
            'road_risk': self.road_risk,
            'weather_risk': self.weather_risk,
            'driver_readiness': self.driver_readiness,
            'time_to_collision': self.time_to_collision,
            'primary_factors': self.primary_factors,
            'confidence': self.confidence,
            'recommended_action': self.recommended_action.name,
            'explanation': self.explanation
        }
