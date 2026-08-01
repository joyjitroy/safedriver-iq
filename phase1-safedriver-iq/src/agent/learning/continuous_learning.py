"""
Continuous Learning Module

Implements online learning and adaptation for the Agentic AI system.
Learns from real-world experiences to improve decision-making over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single driving experience for learning"""
    context: Dict  # Driving context at decision time
    assessment: Dict  # Risk assessment made
    intervention: str  # Action taken
    outcome: str  # 'avoided_crash', 'false_alarm', 'crash_occurred', 'no_event'
    driver_feedback: Optional[str]  # Driver override or feedback
    reward: float  # Computed reward for this experience
    timestamp: datetime


class ExperienceBuffer:
    """
    Stores and manages driving experiences for learning.
    
    Implements prioritized experience replay for efficient learning.
    """
    
    def __init__(
        self,
        max_size: int = 100000,
        priority_alpha: float = 0.6
    ):
        """
        Initialize experience buffer.
        
        Args:
            max_size: Maximum number of experiences to store
            priority_alpha: Exponent for priority weighting (0 = uniform, 1 = full priority)
        """
        self.max_size = max_size
        self.priority_alpha = priority_alpha
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
        logger.info(f"ExperienceBuffer initialized with capacity {max_size}")
    
    def store(self, experience: Experience, priority: Optional[float] = None):
        """
        Store a new experience.
        
        Args:
            experience: Experience object
            priority: Optional priority (higher = more important). 
                     If None, uses default based on outcome.
        """
        if priority is None:
            priority = self._compute_default_priority(experience)
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        
        logger.debug(f"Stored experience with priority {priority:.2f}")
    
    def _compute_default_priority(self, experience: Experience) -> float:
        """
        Compute default priority based on experience characteristics.
        
        High priority for:
        - Crashes (learn from mistakes)
        - Near-misses (learn from close calls)
        - High-risk scenarios
        - Novel situations
        """
        priority = 1.0
        
        # Outcome-based priority
        if experience.outcome == 'crash_occurred':
            priority = 10.0  # Highest priority
        elif experience.outcome == 'avoided_crash':
            priority = 5.0  # Learn from successful interventions
        elif experience.outcome == 'false_alarm':
            priority = 3.0  # Learn to reduce false positives
        
        # Risk-based boost
        risk = experience.assessment.get('overall_risk', 0)
        if risk > 70:
            priority *= 1.5
        
        # Intervention-based boost
        if experience.intervention == 'AUTONOMOUS_BRAKE':
            priority *= 1.3
        
        return priority
    
    def sample(
        self,
        batch_size: int,
        prioritized: bool = True
    ) -> List[Experience]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            prioritized: Whether to use prioritized sampling
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) == 0:
            return []
        
        batch_size = min(batch_size, len(self.buffer))
        
        if prioritized and len(self.priorities) > 0:
            # Prioritized sampling
            priorities = np.array(self.priorities) ** self.priority_alpha
            probs = priorities / priorities.sum()
            indices = np.random.choice(
                len(self.buffer),
                size=batch_size,
                replace=False,
                p=probs
            )
        else:
            # Uniform sampling
            indices = np.random.choice(
                len(self.buffer),
                size=batch_size,
                replace=False
            )
        
        return [self.buffer[i] for i in indices]
    
    def get_recent(self, n: int = 100) -> List[Experience]:
        """Get n most recent experiences"""
        return list(self.buffer)[-n:]
    
    def get_by_outcome(self, outcome: str) -> List[Experience]:
        """Get all experiences with specific outcome"""
        return [exp for exp in self.buffer if exp.outcome == outcome]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {'size': 0}
        
        outcomes = {}
        interventions = {}
        
        for exp in self.buffer:
            outcomes[exp.outcome] = outcomes.get(exp.outcome, 0) + 1
            interventions[exp.intervention] = interventions.get(exp.intervention, 0) + 1
        
        return {
            'size': len(self.buffer),
            'outcomes': outcomes,
            'interventions': interventions,
            'avg_reward': np.mean([exp.reward for exp in self.buffer]),
            'avg_priority': np.mean(list(self.priorities)) if self.priorities else 0
        }


class OnlineLearner:
    """
    Implements online learning algorithms for continuous improvement.
    
    Updates model parameters based on real-world experiences.
    """
    
    def __init__(
        self,
        decision_engine,
        learning_rate: float = 0.01,
        update_frequency: int = 100
    ):
        """
        Initialize online learner.
        
        Args:
            decision_engine: AgentDecisionEngine to update
            learning_rate: How quickly to adapt weights
            update_frequency: Update model every N experiences
        """
        self.engine = decision_engine
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        
        self.experience_buffer = ExperienceBuffer()
        self.update_counter = 0
        self.learning_history = []
        
        logger.info("OnlineLearner initialized")
    
    def record_experience(
        self,
        context: Dict,
        assessment: Dict,
        intervention: str,
        outcome: str,
        driver_feedback: Optional[str] = None
    ):
        """
        Record a new experience and potentially trigger learning.
        
        Args:
            context: DrivingContext dict
            assessment: RiskAssessment dict
            intervention: Intervention level taken
            outcome: Actual outcome
            driver_feedback: Optional driver feedback
        """
        # Compute reward
        reward = self._compute_reward(intervention, outcome, driver_feedback)
        
        # Create experience
        experience = Experience(
            context=context,
            assessment=assessment,
            intervention=intervention,
            outcome=outcome,
            driver_feedback=driver_feedback,
            reward=reward,
            timestamp=datetime.now()
        )
        
        # Store in buffer
        self.experience_buffer.store(experience)
        
        # Trigger learning if enough experiences accumulated
        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self.learn_from_experiences()
            self.update_counter = 0
    
    def _compute_reward(
        self,
        intervention: str,
        outcome: str,
        driver_feedback: Optional[str]
    ) -> float:
        """
        Compute reward for reinforcement learning.
        
        Reward structure:
        - High reward for preventing crashes
        - Penalty for false alarms
        - Bonus for appropriate warnings
        - Penalty for missed interventions
        """
        reward = 0.0
        
        # Outcome-based rewards
        if outcome == 'avoided_crash':
            reward = 10.0  # Successfully prevented crash
        elif outcome == 'crash_occurred':
            reward = -8.0  # Failed to prevent crash
        elif outcome == 'false_alarm':
            reward = -2.0  # Unnecessary intervention
        elif outcome == 'no_event':
            reward = 0.5  # Normal operation
        
        # Intervention appropriateness
        if intervention == 'AUTONOMOUS_BRAKE' and outcome == 'avoided_crash':
            reward += 5.0  # Correct emergency intervention
        elif intervention == 'PASSIVE_MONITORING' and outcome == 'crash_occurred':
            reward -= 5.0  # Should have intervened
        
        # Driver feedback adjustment
        if driver_feedback == 'agreed':
            reward += 1.0
        elif driver_feedback == 'too_aggressive':
            reward -= 1.0
        elif driver_feedback == 'too_passive':
            reward -= 1.0
        
        return reward
    
    def learn_from_experiences(self, batch_size: int = 32):
        """
        Update model parameters based on recent experiences.
        
        Implements gradient-based weight updates for risk factors.
        """
        # Sample batch of experiences
        batch = self.experience_buffer.sample(batch_size)
        
        if len(batch) < 10:
            logger.debug("Not enough experiences for learning")
            return
        
        # Extract experiences by outcome
        successful = [e for e in batch if e.reward > 0]
        unsuccessful = [e for e in batch if e.reward < 0]
        
        # Update risk weights based on successful/unsuccessful outcomes
        if successful:
            self._update_risk_weights(successful, increase=True)
        
        if unsuccessful:
            self._update_risk_weights(unsuccessful, increase=False)
        
        # Update intervention thresholds
        self._update_thresholds(batch)
        
        # Log learning progress
        avg_reward = np.mean([e.reward for e in batch])
        self.learning_history.append({
            'timestamp': datetime.now(),
            'batch_size': len(batch),
            'avg_reward': avg_reward,
            'successful': len(successful),
            'unsuccessful': len(unsuccessful)
        })
        
        logger.info(f"Learning update: avg_reward={avg_reward:.2f}, "
                   f"successful={len(successful)}, unsuccessful={len(unsuccessful)}")
    
    def _update_risk_weights(
        self,
        experiences: List[Experience],
        increase: bool
    ):
        """
        Update risk factor weights based on experiences.
        
        Args:
            experiences: List of experiences to learn from
            increase: Whether to increase or decrease weights
        """
        # Analyze which risk factors were present in these experiences
        factor_importance = {
            'safety_score': 0,
            'vru_proximity': 0,
            'road_conditions': 0,
            'weather_visibility': 0,
            'driver_readiness': 0,
            'historical_risk': 0
        }
        
        for exp in experiences:
            # Identify dominant risk factors
            if exp.assessment.get('vru_risk', 0) > 50:
                factor_importance['vru_proximity'] += 1
            if exp.assessment.get('road_risk', 0) > 50:
                factor_importance['road_conditions'] += 1
            if exp.assessment.get('weather_risk', 0) > 50:
                factor_importance['weather_visibility'] += 1
            if exp.assessment.get('driver_readiness', 0) > 50:
                factor_importance['driver_readiness'] += 1
        
        # Update weights
        total_importance = sum(factor_importance.values())
        if total_importance > 0:
            for factor, count in factor_importance.items():
                if count > 0:
                    adjustment = (count / total_importance) * self.learning_rate
                    if increase:
                        self.engine.risk_weights[factor] += adjustment
                    else:
                        self.engine.risk_weights[factor] = max(
                            0.05,
                            self.engine.risk_weights[factor] - adjustment
                        )
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.engine.risk_weights.values())
        for factor in self.engine.risk_weights:
            self.engine.risk_weights[factor] /= total_weight
    
    def _update_thresholds(self, experiences: List[Experience]):
        """
        Update intervention thresholds based on outcomes.
        
        Adjusts thresholds to reduce false alarms and missed interventions.
        """
        false_alarms = [e for e in experiences if e.outcome == 'false_alarm']
        missed = [e for e in experiences if e.outcome == 'crash_occurred']
        
        # If too many false alarms, increase thresholds (be less aggressive)
        if len(false_alarms) > len(experiences) * 0.2:  # > 20% false alarms
            for key in self.engine.thresholds:
                self.engine.thresholds[key] += 2
            logger.info("Increased thresholds to reduce false alarms")
        
        # If missing interventions, decrease thresholds (be more aggressive)
        if len(missed) > len(experiences) * 0.05:  # > 5% missed
            for key in self.engine.thresholds:
                self.engine.thresholds[key] = max(5, self.engine.thresholds[key] - 3)
            logger.info("Decreased thresholds to catch more events")
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning performance metrics"""
        if not self.learning_history:
            return {}
        
        recent = self.learning_history[-10:]  # Last 10 updates
        
        return {
            'total_updates': len(self.learning_history),
            'recent_avg_reward': np.mean([h['avg_reward'] for h in recent]),
            'improvement_trend': self._compute_trend(),
            'buffer_stats': self.experience_buffer.get_statistics(),
            'current_weights': self.engine.risk_weights,
            'current_thresholds': self.engine.thresholds
        }
    
    def _compute_trend(self) -> str:
        """Compute whether learning is improving, stable, or declining"""
        if len(self.learning_history) < 5:
            return 'insufficient_data'
        
        recent_rewards = [h['avg_reward'] for h in self.learning_history[-10:]]
        older_rewards = [h['avg_reward'] for h in self.learning_history[-20:-10]]
        
        if not older_rewards:
            return 'insufficient_data'
        
        recent_avg = np.mean(recent_rewards)
        older_avg = np.mean(older_rewards)
        
        if recent_avg > older_avg + 0.5:
            return 'improving'
        elif recent_avg < older_avg - 0.5:
            return 'declining'
        else:
            return 'stable'


class ScenarioLibrary:
    """
    Maintains a library of driving scenarios for learning and testing.
    
    Categorizes scenarios by type, risk level, and outcome.
    """
    
    def __init__(self):
        """Initialize scenario library"""
        self.scenarios = {
            'school_zone': [],
            'construction': [],
            'rush_hour': [],
            'night_driving': [],
            'weather_events': [],
            'vru_encounters': [],
            'general': []
        }
        
        self.scenario_count = 0
        
        logger.info("ScenarioLibrary initialized")
    
    def add_scenario(
        self,
        category: str,
        context: Dict,
        assessment: Dict,
        outcome: str
    ):
        """Add a scenario to the library"""
        if category not in self.scenarios:
            category = 'general'
        
        scenario = {
            'id': self.scenario_count,
            'context': context,
            'assessment': assessment,
            'outcome': outcome,
            'timestamp': datetime.now()
        }
        
        self.scenarios[category].append(scenario)
        self.scenario_count += 1
    
    def get_scenarios(
        self,
        category: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Retrieve scenarios matching criteria.
        
        Args:
            category: Scenario category filter
            outcome: Outcome filter
            limit: Maximum number to return
            
        Returns:
            List of matching scenarios
        """
        if category:
            scenarios = self.scenarios.get(category, [])
        else:
            scenarios = []
            for category_scenarios in self.scenarios.values():
                scenarios.extend(category_scenarios)
        
        if outcome:
            scenarios = [s for s in scenarios if s['outcome'] == outcome]
        
        return scenarios[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics"""
        stats = {
            'total_scenarios': self.scenario_count,
            'by_category': {}
        }
        
        for category, scenarios in self.scenarios.items():
            stats['by_category'][category] = len(scenarios)
        
        return stats
    
    def save_to_file(self, filepath: str):
        """Save library to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scenarios, f)
        logger.info(f"Saved scenario library to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load library from disk"""
        with open(filepath, 'rb') as f:
            self.scenarios = pickle.load(f)
        
        # Recount scenarios
        self.scenario_count = sum(len(s) for s in self.scenarios.values())
        logger.info(f"Loaded {self.scenario_count} scenarios from {filepath}")
