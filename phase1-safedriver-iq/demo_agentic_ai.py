"""
Agentic AI System Demo

Demonstrates the autonomous decision-making capabilities of SafeDriver-IQ's
Agentic AI system.
"""

import sys
import os
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.core.decision_engine import AgentDecisionEngine, DrivingContext
from agent.perception.context_engine import PerceptionEngine
from agent.control.intervention_controller import InterventionController, InterventionCommand
from agent.learning.continuous_learning import OnlineLearner
from simulation.scenario_simulator import ScenarioSimulator

import numpy as np
import pandas as pd
from datetime import datetime


def load_safety_model():
    """
    Load trained SafeDriver-IQ model from results/models/best_safety_model.pkl
    
    Falls back to mock model if file not found.
    """
    model_path = 'results/models/best_safety_model.pkl'
    feature_path = 'results/models/feature_names.txt'
    
    if os.path.exists(model_path):
        try:
            print(f"   Loading trained model from {model_path}...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load feature names
            feature_names = []
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    feature_names = [line.strip() for line in f]
            
            # Wrap the model to provide predict_safety_score interface
            class SafetyModelWrapper:
                def __init__(self, model, feature_names):
                    self.model = model
                    self.feature_names = feature_names
                
                def predict_safety_score(self, X):
                    """Predict safety score from features"""
                    # Ensure X is a DataFrame with correct features
                    if not isinstance(X, pd.DataFrame):
                        X = pd.DataFrame([X])
                    
                    # Add missing features with defaults
                    for feature in self.feature_names:
                        if feature not in X.columns:
                            X[feature] = 0
                    
                    # Ensure correct order
                    X = X[self.feature_names]
                    
                    # Get probability of NO crash (class 0)
                    proba_no_crash = self.model.predict_proba(X)[:, 0]
                    
                    # Convert to 0-100 scale
                    safety_scores = proba_no_crash * 100
                    
                    return safety_scores
            
            wrapped_model = SafetyModelWrapper(model, feature_names)
            print(f"   ✓ Model loaded: {model.__class__.__name__}")
            print(f"   ✓ Features: {len(feature_names)} features")
            return wrapped_model
            
        except Exception as e:
            print(f"   ⚠️  Error loading model: {e}")
            print(f"   Using mock model as fallback...")
    else:
        print(f"   ⚠️  Model not found at {model_path}")
        print(f"   Using mock model for demonstration...")
    
    # Fallback mock model
    class MockSafetyModel:
        def predict_safety_score(self, features):
            # Mock prediction based on features
            # Higher speed = lower safety
            if isinstance(features, pd.DataFrame):
                speed_rel = features.get('SPEED_REL', pd.Series([3])).iloc[0]
                vru_count = features.get('total_vru', pd.Series([0])).iloc[0]
            else:
                speed_rel = 3
                vru_count = 0
            
            base_score = 85
            base_score -= (speed_rel - 1) * 10  # Penalty for speed
            base_score -= vru_count * 5  # Penalty for VRU presence
            
            return np.array([max(30, min(100, base_score))])
    
    return MockSafetyModel()


def demo_single_scenario():
    """Demonstrate agent response to a single scenario"""
    print("\n" + "="*80)
    print("AGENTIC AI DEMO: Single Scenario Test")
    print("="*80)
    
    # Initialize components
    print("\n1. Initializing Agentic AI system...")
    safety_model = load_safety_model()
    agent = AgentDecisionEngine(safety_model, learning_mode=True)
    perception = PerceptionEngine()
    controller = InterventionController(simulation_mode=True)
    learner = OnlineLearner(agent)
    
    print("   ✓ Decision Engine initialized")
    print("   ✓ Perception Engine initialized")
    print("   ✓ Intervention Controller initialized")
    print("   ✓ Learning System initialized")
    
    # Generate scenario
    print("\n2. Generating test scenario...")
    simulator = ScenarioSimulator(random_seed=42)
    scenario = simulator.generate_scenario('vru_crossing')
    
    print("   Scenario: VRU Crossing")
    print(f"   Speed: {scenario['sensor_data']['speed_mph']:.1f} mph")
    print(f"   VRUs detected: {len(scenario['sensor_data']['vru_detections'])}")
    print(f"   Lighting: {scenario['sensor_data']['ambient_light_lux']:.0f} lux")
    print(f"   Weather: {scenario['sensor_data']['weather']}")
    
    # Build context
    print("\n3. Building driving context...")
    context = perception.build_context(
        scenario['sensor_data'],
        scenario['location_data'],
        scenario['driver_data']
    )
    
    print(f"   ✓ Context built at {context.timestamp}")
    print(f"   - Speed: {context.speed_mph:.1f} mph")
    print(f"   - VRUs: {context.pedestrians_detected} pedestrians, "
          f"{context.cyclists_detected} cyclists")
    print(f"   - Road: {context.road_surface}, {context.lighting}")
    print(f"   - Weather: {context.weather}, visibility {context.visibility_meters:.0f}m")
    
    # Assess situation
    print("\n4. Performing risk assessment...")
    assessment = agent.assess_situation(context)
    
    print(f"\n   {'='*70}")
    print(f"   RISK ASSESSMENT RESULTS")
    print(f"   {'='*70}")
    print(f"   Overall Risk: {assessment.overall_risk:.1f}/100")
    print(f"   Safety Score: {assessment.safety_score:.1f}/100")
    print(f"   VRU Risk: {assessment.vru_risk:.1f}/100")
    print(f"   Road Risk: {assessment.road_risk:.1f}/100")
    print(f"   Weather Risk: {assessment.weather_risk:.1f}/100")
    print(f"   Driver Readiness: {assessment.driver_readiness:.1f}/100")
    
    if assessment.time_to_collision:
        print(f"   ⚠️ Time to Collision: {assessment.time_to_collision:.1f}s")
    
    print(f"\n   Primary Risk Factors:")
    for factor in assessment.primary_factors:
        print(f"   • {factor}")
    
    print(f"\n   Recommended Action: {assessment.recommended_action.name}")
    print(f"   Confidence: {assessment.confidence*100:.1f}%")
    
    print(f"\n   {'='*70}")
    print(f"   EXPLANATION")
    print(f"   {'='*70}")
    for line in assessment.explanation.split('\n'):
        print(f"   {line}")
    print(f"   {'='*70}")
    
    # Execute intervention
    print("\n5. Executing intervention...")
    
    # Determine brake level
    brake_level = None
    if assessment.recommended_action.value >= 3:  # Aggressive warning or brake
        brake_level = 0.3 + (assessment.overall_risk / 100) * 0.5
    
    command = InterventionCommand(
        level=assessment.recommended_action,
        urgency=assessment.overall_risk / 100,
        reasoning=assessment.explanation,
        brake_level=brake_level,
        notification_message=assessment.explanation.split('\n\n')[-1]  # Last part
    )
    
    result = controller.execute_intervention(command, context.__dict__)
    
    print(f"   ✓ Intervention executed")
    print(f"   Actions taken: {len(result['actions_taken'])}")
    
    # Record experience
    print("\n6. Recording experience for learning...")
    learner.record_experience(
        context=context.__dict__,
        assessment=assessment.to_dict(),
        intervention=assessment.recommended_action.name,
        outcome='avoided_crash',  # Simulated outcome
        driver_feedback='agreed'
    )
    print("   ✓ Experience recorded")
    
    # Show system status
    print("\n7. System Status:")
    status = controller.get_system_status()
    print(f"   Total interventions: {status['total_interventions']}")
    print(f"   Brake active: {status['brake_status']['active']}")
    print(f"   Learning history: {len(learner.learning_history)} updates")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)


def demo_scenario_sequence():
    """Demonstrate agent response to multiple scenarios"""
    print("\n" + "="*80)
    print("AGENTIC AI DEMO: Multi-Scenario Test")
    print("="*80)
    
    # Initialize system
    safety_model = load_safety_model()
    agent = AgentDecisionEngine(safety_model, learning_mode=True)
    perception = PerceptionEngine()
    controller = InterventionController(simulation_mode=True)
    learner = OnlineLearner(agent, update_frequency=3)
    
    # Define scenario sequence
    scenarios = [
        'school_zone',
        'vru_crossing',
        'night_rain',
        'construction',
        'ice_road'
    ]
    
    print(f"\nRunning {len(scenarios)} scenarios...")
    print("="*80)
    
    simulator = ScenarioSimulator(random_seed=42)
    
    for i, scenario_type in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"SCENARIO {i}/{len(scenarios)}: {scenario_type.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        # Generate and process scenario
        scenario = simulator.generate_scenario(scenario_type)
        context = perception.build_context(
            scenario['sensor_data'],
            scenario['location_data'],
            scenario['driver_data']
        )
        
        # Assess and intervene
        assessment = agent.assess_situation(context)
        
        print(f"\nSpeed: {context.speed_mph:.1f} mph | "
              f"Safety Score: {assessment.safety_score:.0f}/100 | "
              f"Risk: {assessment.overall_risk:.0f}/100")
        print(f"VRUs: {context.pedestrians_detected} peds, {context.cyclists_detected} cyclists")
        print(f"Conditions: {context.weather}, {context.lighting}, {context.road_surface} road")
        
        print(f"\n→ Recommended: {assessment.recommended_action.name}")
        print(f"→ Primary factors: {', '.join(assessment.primary_factors[:2])}")
        
        # Execute
        brake_level = None
        if assessment.recommended_action.value >= 3:
            brake_level = 0.2 + (assessment.overall_risk / 100) * 0.4
        
        command = InterventionCommand(
            level=assessment.recommended_action,
            urgency=assessment.overall_risk / 100,
            reasoning=assessment.explanation,
            brake_level=brake_level,
            notification_message=f"Safety score: {assessment.safety_score:.0f}/100"
        )
        
        result = controller.execute_intervention(command, context.__dict__)
        
        # Record for learning
        outcome = 'avoided_crash' if assessment.overall_risk > 60 else 'no_event'
        learner.record_experience(
            context=context.__dict__,
            assessment=assessment.to_dict(),
            intervention=assessment.recommended_action.name,
            outcome=outcome,
            driver_feedback='agreed' if assessment.overall_risk > 40 else None
        )
    
    # Show learning progress
    print(f"\n{'='*80}")
    print("LEARNING SUMMARY")
    print(f"{'='*80}")
    
    metrics = learner.get_learning_metrics()
    print(f"\nTotal learning updates: {metrics.get('total_updates', 0)}")
    print(f"Recent average reward: {metrics.get('recent_avg_reward', 0):.2f}")
    print(f"Trend: {metrics.get('improvement_trend', 'N/A')}")
    
    print(f"\nUpdated Risk Weights:")
    for factor, weight in metrics.get('current_weights', {}).items():
        print(f"  {factor:.<30} {weight:.3f}")
    
    print(f"\nBuffer Statistics:")
    buffer_stats = metrics.get('buffer_stats', {})
    print(f"  Experiences stored: {buffer_stats.get('size', 0)}")
    print(f"  Outcomes: {buffer_stats.get('outcomes', {})}")
    
    print(f"\nSystem Status:")
    status = controller.get_system_status()
    print(f"  Total interventions: {status['total_interventions']}")
    print(f"  Notifications sent: {status['notification_stats'].get('total', 0)}")
    
    print("\n" + "="*80)
    print("MULTI-SCENARIO DEMO COMPLETE")
    print("="*80)


def main():
    """Run demo"""
    print("\n" + "="*80)
    print("SafeDriver-IQ: AGENTIC AI SYSTEM DEMONSTRATION")
    print("="*80)
    print("\nThis demo showcases the autonomous decision-making capabilities")
    print("of the SafeDriver-IQ Agentic AI system.")
    print("\nFeatures demonstrated:")
    print("  • Real-time risk assessment")
    print("  • Multi-factor decision making")
    print("  • Autonomous intervention control")
    print("  • Continuous learning from experiences")
    print("  • Transparent reasoning and explanation")
    
    # Run demos
    demo_single_scenario()
    
    print("\n\nPress Enter to continue with multi-scenario demo...")
    input()
    
    demo_scenario_sequence()
    
    print("\n\nThank you for exploring the SafeDriver-IQ Agentic AI system!")
    print("This represents Phase 1 of the implementation roadmap.")
    print("\nNext steps:")
    print("  • Integrate with real vehicle sensors")
    print("  • Train on actual driving data")
    print("  • Expand scenario library")
    print("  • Implement advanced RL algorithms")
    print("  • Conduct real-world pilot testing")


if __name__ == "__main__":
    main()
