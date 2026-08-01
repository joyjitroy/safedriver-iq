"""
Decision Engine Core Module

Implements the central decision-making logic for the Agentic AI system.
Combines multiple risk factors to determine optimal interventions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from .risk_assessment import RiskAssessment, InterventionLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DrivingContext:
    """Complete context of current driving scenario"""
    # Vehicle state
    speed_mph: float
    acceleration: float
    
    # Road conditions
    road_surface: str  # 'dry', 'wet', 'icy', 'snow'
    road_quality: float  # 0-100
    construction_zone: bool
    
    # Environmental
    weather: str  # 'clear', 'rain', 'fog', 'snow'
    visibility_meters: float
    lighting: str  # 'daylight', 'dawn', 'dusk', 'dark'
    
    # VRU presence
    pedestrians_detected: int
    cyclists_detected: int
    vru_distances: List[float]  # meters
    vru_trajectories: List[Tuple[float, float]]  # (velocity, angle)
    
    # Location context
    location_type: str  # 'urban', 'suburban', 'rural'
    special_zone: Optional[str]  # 'school', 'construction', 'hospital', None
    historical_crashes: int  # crashes at this location in database
    
    # Driver state
    attention_level: float  # 0-100
    recent_overrides: int  # number of recent system overrides
    
    # Timestamp
    timestamp: datetime


class AgentDecisionEngine:
    """
    Core decision engine for the Agentic AI system.
    
    Combines multiple risk factors using learned weights to determine
    optimal intervention strategies in real-time.
    """
    
    def __init__(
        self,
        safety_model,
        intervention_thresholds: Optional[Dict[str, float]] = None,
        learning_mode: bool = True
    ):
        """
        Initialize decision engine.
        
        Args:
            safety_model: Trained SafetyScoreModel from existing system
            intervention_thresholds: Custom thresholds for each intervention level
            learning_mode: Enable continuous learning from experiences
        """
        self.safety_model = safety_model
        self.learning_mode = learning_mode
        
        # Default intervention thresholds (risk scores)
        self.thresholds = intervention_thresholds or {
            'passive': 15,  # risk < 15: just monitor
            'gentle': 30,   # risk 15-30: gentle warning
            'audio': 50,    # risk 30-50: audio alert
            'aggressive': 70,  # risk 50-70: aggressive warning
            'brake': 70     # risk > 70: autonomous brake
        }
        
        # Risk factor weights (will be learned over time)
        self.risk_weights = {
            'safety_score': 0.30,
            'vru_proximity': 0.25,
            'road_conditions': 0.15,
            'weather_visibility': 0.10,
            'driver_readiness': 0.10,
            'historical_risk': 0.10
        }
        
        # Decision history for learning
        self.decision_history = []
        
        logger.info("AgentDecisionEngine initialized")
    
    def assess_situation(self, context: DrivingContext) -> RiskAssessment:
        """
        Perform comprehensive risk assessment of current situation.
        
        Args:
            context: Complete driving context
            
        Returns:
            RiskAssessment with overall risk and recommended action
        """
        # 1. Get safety score from existing model
        safety_score = self._get_safety_score(context)
        
        # 2. Assess individual risk components
        vru_risk = self._assess_vru_risk(context)
        road_risk = self._assess_road_risk(context)
        weather_risk = self._assess_weather_risk(context)
        driver_readiness = self._assess_driver_readiness(context)
        historical_risk = self._assess_historical_risk(context)
        
        # 3. Compute weighted overall risk
        overall_risk = self._compute_weighted_risk(
            safety_score, vru_risk, road_risk, weather_risk,
            driver_readiness, historical_risk
        )
        
        # 4. Determine time to collision if VRU present
        ttc = self._calculate_time_to_collision(context)
        
        # 5. Identify primary risk factors
        primary_factors = self._identify_primary_factors(
            safety_score, vru_risk, road_risk, weather_risk,
            driver_readiness, historical_risk
        )
        
        # 6. Determine recommended intervention
        intervention, confidence = self._determine_intervention(
            overall_risk, ttc, context
        )
        
        # 7. Generate explanation
        explanation = self._generate_explanation(
            overall_risk, primary_factors, intervention, context
        )
        
        # Create assessment
        assessment = RiskAssessment(
            overall_risk=overall_risk,
            safety_score=safety_score,
            vru_risk=vru_risk,
            road_risk=road_risk,
            weather_risk=weather_risk,
            driver_readiness=driver_readiness,
            time_to_collision=ttc,
            primary_factors=primary_factors,
            confidence=confidence,
            recommended_action=intervention,
            explanation=explanation
        )
        
        # Store decision for learning
        if self.learning_mode:
            self._record_decision(context, assessment)
        
        return assessment
    
    def _get_safety_score(self, context: DrivingContext) -> float:
        """Get safety score from existing SafeDriver-IQ model"""
        # Convert context to feature vector
        features = self._context_to_features(context)
        
        # Get prediction from model
        score = self.safety_model.predict_safety_score(features)
        
        return float(score[0]) if hasattr(score, '__iter__') else float(score)
    
    def _assess_vru_risk(self, context: DrivingContext) -> float:
        """
        Assess VRU collision risk.
        
        Risk factors:
        - Number of VRUs present
        - Proximity to vehicle
        - Predicted trajectories
        - VRU type (pedestrian vs cyclist)
        - VRU attention state
        """
        if not context.vru_distances:
            return 0.0
        
        risk = 0.0
        
        # Count factor
        total_vrus = context.pedestrians_detected + context.cyclists_detected
        count_risk = min(total_vrus * 10, 30)  # Max 30 points from count
        
        # Proximity factor
        min_distance = min(context.vru_distances)
        if min_distance < 5:
            proximity_risk = 50
        elif min_distance < 10:
            proximity_risk = 30
        elif min_distance < 20:
            proximity_risk = 15
        else:
            proximity_risk = 5
        
        # Trajectory risk (simplified - would need actual trajectory analysis)
        trajectory_risk = 0.0
        if context.vru_trajectories:
            # Check for crossing trajectories
            for vel, angle in context.vru_trajectories:
                if 45 < abs(angle) < 135:  # Crossing path
                    trajectory_risk = max(trajectory_risk, 20)
        
        risk = count_risk + proximity_risk + trajectory_risk
        return min(risk, 100.0)
    
    def _assess_road_risk(self, context: DrivingContext) -> float:
        """
        Assess risk from road conditions.
        
        Risk factors:
        - Surface type (wet, icy, snow)
        - Road quality
        - Construction zones
        - Speed relative to conditions
        """
        risk = 0.0
        
        # Surface condition
        surface_risk = {
            'dry': 0,
            'wet': 20,
            'icy': 50,
            'snow': 40,
            'gravel': 30
        }.get(context.road_surface.lower(), 0)
        
        # Road quality
        quality_risk = (100 - context.road_quality) * 0.3
        
        # Construction zone
        construction_risk = 20 if context.construction_zone else 0
        
        # Speed factor
        if context.speed_mph > 60 and context.road_surface != 'dry':
            speed_risk = 20
        elif context.speed_mph > 40 and context.road_surface in ['icy', 'snow']:
            speed_risk = 30
        else:
            speed_risk = 0
        
        risk = surface_risk + quality_risk + construction_risk + speed_risk
        return min(risk, 100.0)
    
    def _assess_weather_risk(self, context: DrivingContext) -> float:
        """
        Assess risk from weather and visibility.
        
        Risk factors:
        - Weather type
        - Visibility range
        - Lighting conditions
        - Combined effects (e.g., night + rain)
        """
        risk = 0.0
        
        # Weather condition
        weather_risk = {
            'clear': 0,
            'rain': 20,
            'fog': 40,
            'snow': 30,
            'storm': 50
        }.get(context.weather.lower(), 0)
        
        # Visibility
        if context.visibility_meters < 50:
            visibility_risk = 50
        elif context.visibility_meters < 100:
            visibility_risk = 30
        elif context.visibility_meters < 200:
            visibility_risk = 15
        else:
            visibility_risk = 0
        
        # Lighting
        lighting_risk = {
            'daylight': 0,
            'dawn': 10,
            'dusk': 15,
            'dark': 25
        }.get(context.lighting.lower(), 0)
        
        # Combined effect (multiplicative for worst conditions)
        if context.lighting == 'dark' and context.weather in ['rain', 'fog']:
            combined_bonus = 15
        else:
            combined_bonus = 0
        
        risk = weather_risk + visibility_risk + lighting_risk + combined_bonus
        return min(risk, 100.0)
    
    def _assess_driver_readiness(self, context: DrivingContext) -> float:
        """
        Assess driver's readiness to respond.
        
        Lower readiness = higher intervention threshold
        
        Factors:
        - Attention level
        - Recent override pattern (frequent overrides = less trustworthy)
        """
        # Invert attention (low attention = high risk)
        attention_risk = (100 - context.attention_level) * 0.6
        
        # Override pattern (too many overrides suggests disagreement with system)
        if context.recent_overrides > 5:
            override_risk = 20
        elif context.recent_overrides > 2:
            override_risk = 10
        else:
            override_risk = 0
        
        risk = attention_risk + override_risk
        return min(risk, 100.0)
    
    def _assess_historical_risk(self, context: DrivingContext) -> float:
        """
        Assess risk based on historical crash patterns at location.
        
        More crashes at this location = higher baseline risk
        """
        # Scale historical crashes to risk score
        # Assume 10+ crashes = maximum historical risk
        historical_risk = min(context.historical_crashes * 5, 50)
        
        # Special zones
        if context.special_zone == 'school':
            zone_risk = 20
        elif context.special_zone == 'construction':
            zone_risk = 15
        elif context.special_zone == 'hospital':
            zone_risk = 10
        else:
            zone_risk = 0
        
        return min(historical_risk + zone_risk, 100.0)
    
    def _compute_weighted_risk(
        self,
        safety_score: float,
        vru_risk: float,
        road_risk: float,
        weather_risk: float,
        driver_readiness: float,
        historical_risk: float
    ) -> float:
        """Combine all risk factors with learned weights"""
        
        # Safety score is inverse of risk
        safety_risk = 100 - safety_score
        
        overall_risk = (
            safety_risk * self.risk_weights['safety_score'] +
            vru_risk * self.risk_weights['vru_proximity'] +
            road_risk * self.risk_weights['road_conditions'] +
            weather_risk * self.risk_weights['weather_visibility'] +
            driver_readiness * self.risk_weights['driver_readiness'] +
            historical_risk * self.risk_weights['historical_risk']
        )
        
        return min(overall_risk, 100.0)
    
    def _calculate_time_to_collision(
        self, context: DrivingContext
    ) -> Optional[float]:
        """
        Calculate time to collision with nearest VRU.
        
        Returns:
            Time in seconds, or None if no imminent collision
        """
        if not context.vru_distances:
            return None
        
        min_distance = min(context.vru_distances)
        
        # Simple TTC calculation (would be more sophisticated with actual trajectories)
        # Assume VRU might cross path
        if context.speed_mph == 0:
            return None
        
        # Convert speed to m/s
        speed_ms = context.speed_mph * 0.44704
        
        # Time to reach VRU position
        ttc = min_distance / speed_ms if speed_ms > 0 else None
        
        # Only return if collision seems possible (< 10 seconds)
        if ttc and ttc < 10:
            return ttc
        
        return None
    
    def _identify_primary_factors(
        self,
        safety_score: float,
        vru_risk: float,
        road_risk: float,
        weather_risk: float,
        driver_readiness: float,
        historical_risk: float
    ) -> List[str]:
        """Identify top 3 risk contributors"""
        factors = {
            'Low safety score': 100 - safety_score,
            'VRU proximity': vru_risk,
            'Road conditions': road_risk,
            'Weather/visibility': weather_risk,
            'Driver attention': driver_readiness,
            'Historical crash risk': historical_risk
        }
        
        # Sort by risk level and take top 3
        sorted_factors = sorted(
            factors.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Only include factors with significant risk (> 20)
        primary = [name for name, risk in sorted_factors if risk > 20][:3]
        
        return primary if primary else ['Normal driving conditions']
    
    def _determine_intervention(
        self,
        overall_risk: float,
        ttc: Optional[float],
        context: DrivingContext
    ) -> Tuple[InterventionLevel, float]:
        """
        Determine appropriate intervention level.
        
        Returns:
            (intervention_level, confidence)
        """
        # Emergency override: imminent collision
        if ttc and ttc < 2.0:
            return InterventionLevel.AUTONOMOUS_BRAKE, 0.95
        
        # Risk-based thresholds
        if overall_risk < self.thresholds['passive']:
            level = InterventionLevel.PASSIVE_MONITORING
            confidence = 0.9
        elif overall_risk < self.thresholds['gentle']:
            level = InterventionLevel.GENTLE_WARNING
            confidence = 0.8
        elif overall_risk < self.thresholds['audio']:
            level = InterventionLevel.AUDIO_ALERT
            confidence = 0.85
        elif overall_risk < self.thresholds['aggressive']:
            level = InterventionLevel.AGGRESSIVE_WARNING
            confidence = 0.9
        else:
            level = InterventionLevel.AUTONOMOUS_BRAKE
            confidence = 0.95
        
        # Adjust confidence based on model uncertainty
        # (would use actual model confidence in production)
        
        return level, confidence
    
    def _generate_explanation(
        self,
        overall_risk: float,
        primary_factors: List[str],
        intervention: InterventionLevel,
        context: DrivingContext
    ) -> str:
        """Generate human-readable explanation of decision"""
        
        safety_score = 100 - overall_risk
        
        explanation = f"Safety Score: {safety_score:.0f}/100\n"
        explanation += f"Risk Level: {self._risk_level_name(overall_risk)}\n\n"
        
        if primary_factors and primary_factors[0] != 'Normal driving conditions':
            explanation += "Primary Risk Factors:\n"
            for factor in primary_factors:
                explanation += f"  • {factor}\n"
            explanation += "\n"
        
        explanation += f"Recommended Action: {intervention.name.replace('_', ' ').title()}\n\n"
        
        # Add specific guidance
        if intervention == InterventionLevel.AUTONOMOUS_BRAKE:
            explanation += "⚠️ Emergency braking applied - imminent collision risk"
        elif intervention == InterventionLevel.AGGRESSIVE_WARNING:
            explanation += "⚠️ Reduce speed immediately - high collision risk"
        elif intervention == InterventionLevel.AUDIO_ALERT:
            explanation += "⚠️ Exercise caution - elevated risk conditions"
        elif intervention == InterventionLevel.GENTLE_WARNING:
            explanation += "ℹ️ Stay alert - potentially hazardous conditions"
        else:
            explanation += "✓ Continue safe driving"
        
        # Add context-specific advice
        if context.vru_distances and min(context.vru_distances) < 20:
            explanation += f"\n  VRU detected {min(context.vru_distances):.0f}m ahead"
        
        return explanation
    
    def _risk_level_name(self, risk: float) -> str:
        """Convert risk score to level name"""
        if risk < 15:
            return "MINIMAL"
        elif risk < 30:
            return "LOW"
        elif risk < 50:
            return "MODERATE"
        elif risk < 70:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _context_to_features(self, context: DrivingContext) -> pd.DataFrame:
        """
        Convert DrivingContext to feature vector for safety model.
        
        This bridges the new agent system with existing SafeDriver-IQ model.
        """
        # Map context to feature dictionary
        features = {
            'SPEED_REL': self._map_speed_category(context.speed_mph),
            'WEATHER': self._map_weather(context.weather),
            'LGTCON_IM': self._map_lighting(context.lighting),
            'ROAD_COND': self._map_road_surface(context.road_surface),
            'total_vru': context.pedestrians_detected + context.cyclists_detected,
            'PEDS': context.pedestrians_detected,
            # Add more feature mappings as needed
        }
        
        return pd.DataFrame([features])
    
    def _map_speed_category(self, speed_mph: float) -> int:
        """Map speed to CRSS speed category"""
        if speed_mph < 10:
            return 1
        elif speed_mph < 25:
            return 2
        elif speed_mph < 50:
            return 3
        elif speed_mph < 65:
            return 4
        else:
            return 5
    
    def _map_weather(self, weather: str) -> int:
        """Map weather to CRSS code"""
        mapping = {
            'clear': 1,
            'rain': 2,
            'fog': 5,
            'snow': 3
        }
        return mapping.get(weather.lower(), 1)
    
    def _map_lighting(self, lighting: str) -> int:
        """Map lighting to CRSS code"""
        mapping = {
            'daylight': 1,
            'dawn': 2,
            'dusk': 3,
            'dark': 4
        }
        return mapping.get(lighting.lower(), 1)
    
    def _map_road_surface(self, surface: str) -> int:
        """Map road surface to CRSS code"""
        mapping = {
            'dry': 1,
            'wet': 2,
            'snow': 3,
            'icy': 4
        }
        return mapping.get(surface.lower(), 1)
    
    def _record_decision(self, context: DrivingContext, assessment: RiskAssessment):
        """Record decision for later learning"""
        self.decision_history.append({
            'timestamp': context.timestamp,
            'context': context,
            'assessment': assessment,
            'outcome': None  # Will be filled in later
        })
        
        # Keep last 10000 decisions
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-10000:]
    
    def update_from_experience(
        self,
        context: DrivingContext,
        assessment: RiskAssessment,
        actual_outcome: str,
        driver_feedback: Optional[str] = None
    ):
        """
        Update model based on actual outcome.
        
        Args:
            context: The driving context
            assessment: The risk assessment made
            actual_outcome: 'avoided_crash', 'false_alarm', 'crash_occurred', etc.
            driver_feedback: Optional feedback from driver
        """
        # This would implement online learning
        # For now, just log the experience
        logger.info(f"Experience recorded: {actual_outcome}")
        
        # In production, this would:
        # 1. Update risk weights
        # 2. Adjust intervention thresholds
        # 3. Retrain model components
        # 4. Update scenario library
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        if not self.decision_history:
            return {}
        
        total_decisions = len(self.decision_history)
        
        intervention_counts = {
            level: sum(1 for d in self.decision_history 
                      if d['assessment'].recommended_action == level)
            for level in InterventionLevel
        }
        
        return {
            'total_decisions': total_decisions,
            'intervention_counts': intervention_counts,
            'learning_mode': self.learning_mode,
            'current_weights': self.risk_weights
        }
