"""
Agentic AI Package for SafeDriver-IQ

This package contains the autonomous decision-making system that extends
SafeDriver-IQ from passive safety scoring to active intervention.

Modules:
- core: Decision engine and agent logic
- perception: Environmental monitoring and context understanding
- learning: Continuous learning and adaptation
- control: Intervention execution and driver interface
- explainability: Transparency and reasoning communication
"""

__version__ = "0.1.0"
__author__ = "SafeDriver-IQ Team"

from .core.decision_engine import AgentDecisionEngine
from .core.risk_assessment import RiskAssessment
from .perception.context_engine import PerceptionEngine
from .control.intervention_controller import InterventionController

__all__ = [
    'AgentDecisionEngine',
    'RiskAssessment',
    'PerceptionEngine',
    'InterventionController',
]
