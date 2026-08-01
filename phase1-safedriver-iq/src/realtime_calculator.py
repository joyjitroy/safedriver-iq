"""
Real-Time Safety Score Calculator

This module provides real-time safety score calculation for driving scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSafetyCalculator:
    """
    Real-time safety score calculator using trained inverse safety model.
    
    Accepts current driving conditions and returns:
    - Safety score (0-100)
    - Risk level (Critical/High/Medium/Low)
    - Specific improvement recommendations
    """
    
    def __init__(self, model_path: str, feature_names_path: str):
        """
        Initialize calculator with trained model.
        
        Args:
            model_path: Path to trained model file (.pkl)
            feature_names_path: Path to feature names file (.txt)
        """
        self.model_path = Path(model_path)
        self.feature_names_path = Path(feature_names_path)
        
        # Load model and features
        self.model = None
        self.feature_names = None
        self._load_model()
        
        # Risk thresholds
        self.thresholds = {
            'critical': 40,
            'high': 60,
            'medium': 75,
            'low': 85
        }
    
    def _load_model(self):
        """Load trained model and feature names."""
        try:
            loaded_model = joblib.load(self.model_path)
            
            # Handle both wrapper and underlying model
            # Check if it's a SafetyScoreModel wrapper or the underlying classifier
            if hasattr(loaded_model, 'model') and hasattr(loaded_model, 'model_type'):
                # It's a SafetyScoreModel wrapper - extract underlying model
                self.model = loaded_model.model
                logger.info(f"✓ Model loaded from {self.model_path} (extracted from wrapper)")
            else:
                # It's the underlying classifier directly
                self.model = loaded_model
                logger.info(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        try:
            with open(self.feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"✓ Loaded {len(self.feature_names)} feature names")
        except Exception as e:
            logger.error(f"Failed to load feature names: {e}")
            raise
    
    def calculate_safety_score(self, scenario: Dict) -> Dict:
        """
        Calculate safety score for a driving scenario.
        
        Args:
            scenario: Dictionary with driving conditions
                Example: {
                    'HOUR': 18,
                    'DAY_WEEK': 5,
                    'MONTH': 12,
                    'WEATHER': 1,
                    'LGT_COND': 2,
                    'SPEED_REL': 3,
                    ... (all required features)
                }
        
        Returns:
            Dictionary with:
                - safety_score: 0-100 score
                - risk_level: Critical/High/Medium/Low
                - confidence: Model confidence (0-1)
                - recommendations: List of improvement suggestions
        """
        # Create feature vector
        X = self._create_feature_vector(scenario)
        
        # Predict safety score
        proba = self.model.predict_proba(X)[0]
        safety_score = proba[0] * 100  # Probability of safe driving
        confidence = max(proba)
        
        # Apply rule-based adjustments to compensate for model bias
        # The trained model underweights road conditions, weather, and other factors
        # due to synthetic training data bias (see MODEL_BIAS_ANALYSIS.md)
        safety_score = self._apply_safety_adjustments(scenario, safety_score)
        
        # Determine risk level
        risk_level = self._get_risk_level(safety_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scenario, safety_score)
        
        result = {
            'safety_score': round(safety_score, 2),
            'risk_level': risk_level,
            'confidence': round(confidence, 4),
            'recommendations': recommendations,
            'scenario': scenario
        }
        
        return result
    
    def _create_feature_vector(self, scenario: Dict) -> pd.DataFrame:
        """
        Create feature vector from scenario.
        
        Handles:
        - Feature engineering (temporal, environmental, etc.)
        - Mapping simplified scenario features to model features
        - Missing features (filled with defaults)
        - Feature ordering (matches training)
        """
        from src.feature_engineering import FeatureEngineer
        
        # Map simplified scenario features to CRSS features
        mapped_scenario = scenario.copy()
        
        # Map ROAD_COND to relevant features if not already present
        # The model may use different feature names than the simplified scenario
        if 'ROAD_COND' in scenario and 'REL_ROAD' not in scenario:
            # ROAD_COND: 1=Dry, 2=Wet, 3=Snow/Ice, 4=Other
            # This affects derived features and should influence predictions
            mapped_scenario['SURF_COND'] = scenario['ROAD_COND']
        
        # Map VRU_PRESENT to total_vru (model uses counts, not binary flag)
        if 'VRU_PRESENT' in scenario and 'total_vru' not in scenario:
            # Convert binary presence to count (0 or 1+)
            mapped_scenario['total_vru'] = 1 if scenario['VRU_PRESENT'] == 1 else 0
            mapped_scenario['pedestrian_count'] = 1 if scenario['VRU_PRESENT'] == 1 else 0
            mapped_scenario['cyclist_count'] = 0  # Could be made more specific
        
        # SPEED_REL - if not in model features, it will be ignored
        # but we keep it for potential recommendations
        
        # Create base dataframe
        df = pd.DataFrame([mapped_scenario])
        
        # Engineer features
        fe = FeatureEngineer()
        df = fe.create_temporal_features(df)
        df = fe.create_environmental_features(df)
        df = fe.create_location_features(df)
        
        # Manually add VRU features if not present (can't use create_vru_features without person data)
        if 'total_vru' not in df.columns and 'total_vru' in mapped_scenario:
            df['total_vru'] = mapped_scenario['total_vru']
        if 'pedestrian_count' not in df.columns and 'pedestrian_count' in mapped_scenario:
            df['pedestrian_count'] = mapped_scenario['pedestrian_count']
        if 'cyclist_count' not in df.columns and 'cyclist_count' in mapped_scenario:
            df['cyclist_count'] = mapped_scenario['cyclist_count']
        
        # Set default VRU injury features (unknown for prediction scenario)
        if 'max_vru_injury' not in df.columns:
            df['max_vru_injury'] = 0  # Assume no injury data available
        if 'fatal_vru' not in df.columns:
            df['fatal_vru'] = 0
        
        # Select and order features
        feature_vector = pd.DataFrame()
        for feat in self.feature_names:
            if feat in df.columns:
                feature_vector[feat] = df[feat]
            else:
                # Use intelligent defaults based on feature name
                if 'total_vru' in feat:
                    feature_vector[feat] = mapped_scenario.get('total_vru', 0)
                elif 'pedestrian_count' in feat:
                    feature_vector[feat] = mapped_scenario.get('pedestrian_count', 0)
                elif 'cyclist_count' in feat:
                    feature_vector[feat] = mapped_scenario.get('cyclist_count', 0)
                else:
                    feature_vector[feat] = 0  # Default value
        
        # DEBUG: Print VRU-related features
        vru_features = ['total_vru', 'pedestrian_count', 'cyclist_count', 'fatal_vru', 'max_vru_injury']
        for feat in vru_features:
            if feat in feature_vector.columns:
                logger.debug(f"  {feat}: {feature_vector[feat].values[0]}")
        
        # Fill NaNs
        feature_vector = feature_vector.fillna(0)
        
        return feature_vector
    
    def _apply_safety_adjustments(self, scenario: Dict, base_score: float) -> float:
        """
        Apply rule-based adjustments to compensate for model bias.
        
        The trained model underweights certain features due to synthetic training data:
        - Road conditions: model shows 1.9% importance (should be higher)
        - Weather: model shows limited sensitivity
        - Lighting: model shows limited sensitivity at low speeds
        
        These adjustments are based on domain knowledge and real-world safety research.
        See MODEL_BIAS_ANALYSIS.md for details on model bias.
        
        Args:
            scenario: Driving scenario dictionary
            base_score: Model's predicted safety score (0-100)
            
        Returns:
            Adjusted safety score (0-100)
        """
        adjusted_score = base_score
        
        # Road condition penalties (most critical adjustment)
        road_cond = scenario.get('ROAD_COND', 1)
        if road_cond == 3:  # Ice
            adjusted_score *= 0.60  # 40% penalty for ice
        elif road_cond == 4:  # Snow
            adjusted_score *= 0.70  # 30% penalty for snow
        elif road_cond == 2:  # Wet
            adjusted_score *= 0.85  # 15% penalty for wet
        
        # Weather condition penalties
        weather = scenario.get('WEATHER', 1)
        if weather == 3:  # Snow
            adjusted_score *= 0.80  # 20% penalty
        elif weather == 2:  # Rain
            adjusted_score *= 0.90  # 10% penalty
        elif weather in [4, 5, 6, 7]:  # Fog, sleet, other adverse
            adjusted_score *= 0.85  # 15% penalty
        
        # Lighting condition penalties
        light_cond = scenario.get('LGT_COND', 1)
        if light_cond == 2:  # Dark - not lighted
            adjusted_score *= 0.75  # 25% penalty
        elif light_cond == 3:  # Dark - lighted
            adjusted_score *= 0.85  # 15% penalty
        elif light_cond in [4, 5]:  # Dawn/Dusk
            adjusted_score *= 0.92  # 8% penalty
        
        # Speed penalties (critical safety factor)
        speed = scenario.get('SPEED_REL', 1)
        if speed >= 5:  # Very high speed
            adjusted_score *= 0.65  # 35% penalty for very high speed
        elif speed == 4:  # High speed
            adjusted_score *= 0.75  # 25% penalty for high speed
        elif speed == 3:  # Moderate-high speed
            adjusted_score *= 0.88  # 12% penalty for moderate-high speed
        elif speed == 2:  # Moderate speed
            adjusted_score *= 0.95  # 5% penalty for moderate speed
        # speed == 1 (low speed) gets no penalty
        
        # VRU presence penalty
        vru_present = scenario.get('VRU_PRESENT', 0)
        if vru_present == 1:
            adjusted_score *= 0.88  # 12% penalty for VRU interaction risk
        
        # Time of day penalty (nighttime driving is riskier)
        is_night = scenario.get('IS_NIGHT', 0)
        hour = scenario.get('HOUR', 12)
        if is_night == 1 or (22 <= hour or hour <= 5):  # Night hours
            adjusted_score *= 0.90  # 10% penalty for nighttime driving
        
        # Combined adverse conditions (multiplicative risk)
        # If multiple poor conditions exist, risk compounds
        poor_conditions = 0
        if road_cond in [3, 4]:  # Ice/snow
            poor_conditions += 1
        if weather in [2, 3, 4, 5]:  # Rain/snow/fog
            poor_conditions += 1
        if light_cond in [2, 3]:  # Dark
            poor_conditions += 1
        
        if poor_conditions >= 2:
            # Additional penalty for combined poor conditions
            adjusted_score *= 0.95  # Extra 5% penalty
        
        # Clamp to valid range
        adjusted_score = np.clip(adjusted_score, 0, 100)
        
        return adjusted_score
    
    def _get_risk_level(self, safety_score: float) -> str:
        """Determine risk level from safety score."""
        if safety_score < self.thresholds['critical']:
            return 'Critical'
        elif safety_score < self.thresholds['high']:
            return 'High'
        elif safety_score < self.thresholds['medium']:
            return 'Medium'
        elif safety_score < self.thresholds['low']:
            return 'Low'
        else:
            return 'Excellent'
    
    def _generate_recommendations(self, scenario: Dict, safety_score: float) -> List[str]:
        """
        Generate specific improvement recommendations.
        
        Analyzes scenario and suggests actionable changes.
        """
        recommendations = []
        
        # Check temporal factors
        hour = scenario.get('HOUR', 12)
        if hour >= 20 or hour <= 6:
            recommendations.append("⚠️ Night driving increases risk. Consider traveling during daylight hours.")
        
        if scenario.get('IS_RUSH_HOUR', 0) == 1:
            recommendations.append("⏰ Rush hour traffic increases collision risk. Consider alternative timing.")
        
        if scenario.get('IS_WEEKEND', 0) == 1:
            recommendations.append("📅 Weekend driving can have different risk patterns.")
        
        # Check environmental factors
        if scenario.get('ADVERSE_WEATHER', 0) == 1 or scenario.get('WEATHER', 1) > 1:
            recommendations.append("🌧️ Adverse weather detected. Reduce speed and increase following distance.")
        
        if scenario.get('POOR_LIGHTING', 0) == 1 or scenario.get('LGT_COND', 1) > 1:
            recommendations.append("💡 Poor lighting conditions. Use headlights and stay extra alert.")
        
        # Check speed
        if scenario.get('SPEED_REL', 0) > 3:
            recommendations.append("🚗 High speed relative to conditions. Reduce speed for safety.")
        
        # Check VRU presence
        if scenario.get('VRU_PRESENT', 0) == 1:
            recommendations.append("🚶 Pedestrians/cyclists present. Increase vigilance and reduce speed.")
        
        # Check road conditions
        if scenario.get('ROAD_COND', 1) > 1:
            recommendations.append("🛣️ Poor road conditions. Drive cautiously.")
        
        # Score-based recommendations
        if safety_score < 50:
            recommendations.append("🚨 CRITICAL: Multiple high-risk factors present. Consider postponing trip.")
        elif safety_score < 70:
            recommendations.append("⚠️ HIGH RISK: Several risk factors detected. Exercise extreme caution.")
        elif safety_score < 85:
            recommendations.append("✓ Moderate risk. Stay alert and follow safety best practices.")
        else:
            recommendations.append("✓ Good safety profile. Continue safe driving practices.")
        
        # If no specific recommendations, provide generic
        if len(recommendations) == 0:
            recommendations.append("✓ No major risk factors detected. Continue safe driving.")
        
        return recommendations
    
    def batch_calculate(self, scenarios: List[Dict]) -> pd.DataFrame:
        """
        Calculate safety scores for multiple scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            
        Returns:
            DataFrame with scores and risk levels
        """
        results = []
        
        for i, scenario in enumerate(scenarios):
            try:
                result = self.calculate_safety_score(scenario)
                result['scenario_id'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing scenario {i}: {e}")
                results.append({
                    'scenario_id': i,
                    'safety_score': None,
                    'risk_level': 'Error',
                    'confidence': None,
                    'recommendations': [f"Error: {str(e)}"]
                })
        
        # Convert to dataframe
        df = pd.DataFrame(results)
        return df
    
    def compare_scenarios(self, scenario1: Dict, scenario2: Dict) -> Dict:
        """
        Compare two scenarios and show which is safer.
        
        Args:
            scenario1: First driving scenario
            scenario2: Second driving scenario
            
        Returns:
            Comparison results
        """
        result1 = self.calculate_safety_score(scenario1)
        result2 = self.calculate_safety_score(scenario2)
        
        score_diff = result1['safety_score'] - result2['safety_score']
        
        comparison = {
            'scenario1': result1,
            'scenario2': result2,
            'score_difference': round(score_diff, 2),
            'safer_scenario': 'Scenario 1' if score_diff > 0 else 'Scenario 2',
            'improvement_percentage': round(abs(score_diff) / min(result1['safety_score'], result2['safety_score']) * 100, 2)
        }
        
        return comparison
    
    def suggest_improvements(self, scenario: Dict, target_score: float = 85) -> Dict:
        """
        Suggest specific changes to improve safety score to target.
        
        Args:
            scenario: Current driving scenario
            target_score: Target safety score to achieve
            
        Returns:
            Suggested modifications and expected improvement
        """
        current_result = self.calculate_safety_score(scenario)
        current_score = current_result['safety_score']
        
        if current_score >= target_score:
            return {
                'current_score': current_score,
                'target_score': target_score,
                'achievable': False,  # Already achieved
                'message': 'Already at or above target score!',
                'suggestions': []
            }
        
        suggestions = []
        improved_scenario = scenario.copy()
        
        # Try improving key factors
        improvements = [
            ('Improve lighting', 'POOR_LIGHTING', 0),
            ('Wait for better weather', 'ADVERSE_WEATHER', 0),
            ('Avoid night driving', 'IS_NIGHT', 0),
            ('Avoid rush hour', 'IS_RUSH_HOUR', 0),
            ('Reduce speed', 'SPEED_REL', 1),
            ('Choose better road conditions', 'ROAD_COND', 1)
        ]
        
        for desc, key, value in improvements:
            if key in improved_scenario:
                test_scenario = improved_scenario.copy()
                test_scenario[key] = value
                test_result = self.calculate_safety_score(test_scenario)
                test_score = test_result['safety_score']
                
                if test_score > current_score:
                    improvement = test_score - current_score
                    suggestions.append({
                        'action': desc,
                        'current_value': scenario.get(key),
                        'suggested_value': value,
                        'expected_improvement': round(improvement, 2),
                        'new_score': round(test_score, 2)
                    })
                    
                    if test_score >= target_score:
                        improved_scenario = test_scenario
                        break
        
        # Sort by improvement potential
        suggestions.sort(key=lambda x: x['expected_improvement'], reverse=True)
        
        return {
            'current_score': round(current_score, 2),
            'target_score': target_score,
            'achievable': improved_scenario != scenario,
            'suggestions': suggestions[:5]  # Top 5 suggestions
        }


def create_example_scenarios() -> List[Dict]:
    """Create example scenarios for testing."""
    scenarios = [
        {
            'name': 'Day, Clear, Good Conditions',
            'HOUR': 14,
            'DAY_WEEK': 3,
            'MONTH': 6,
            'WEATHER': 1,
            'LGT_COND': 1,
            'SPEED_REL': 2,
            'ROAD_COND': 1,
            'IS_NIGHT': 0,
            'IS_WEEKEND': 0,
            'IS_RUSH_HOUR': 0,
            'POOR_LIGHTING': 0,
            'ADVERSE_WEATHER': 0
        },
        {
            'name': 'Night, Rain, Rush Hour',
            'HOUR': 18,
            'DAY_WEEK': 5,
            'MONTH': 12,
            'WEATHER': 2,
            'LGT_COND': 2,
            'SPEED_REL': 4,
            'ROAD_COND': 2,
            'IS_NIGHT': 1,
            'IS_WEEKEND': 0,
            'IS_RUSH_HOUR': 1,
            'POOR_LIGHTING': 1,
            'ADVERSE_WEATHER': 1
        },
        {
            'name': 'Weekend Evening, Clear',
            'HOUR': 20,
            'DAY_WEEK': 7,
            'MONTH': 8,
            'WEATHER': 1,
            'LGT_COND': 2,
            'SPEED_REL': 3,
            'ROAD_COND': 1,
            'IS_NIGHT': 1,
            'IS_WEEKEND': 1,
            'IS_RUSH_HOUR': 0,
            'POOR_LIGHTING': 1,
            'ADVERSE_WEATHER': 0
        }
    ]
    
    return scenarios


if __name__ == "__main__":
    # Example usage
    print("Real-time Safety Calculator Module")
    print("=" * 60)
    
    # Check if model exists
    model_path = "../results/models/best_safety_model.pkl"
    feature_path = "../results/models/feature_names.txt"
    
    if Path(model_path).exists():
        print("✓ Model found, testing calculator...")
        
        calc = RealtimeSafetyCalculator(model_path, feature_path)
        
        # Test scenarios
        scenarios = create_example_scenarios()
        
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario.get('name', 'Unknown')}")
            print(f"{'='*60}")
            
            result = calc.calculate_safety_score(scenario)
            
            print(f"Safety Score: {result['safety_score']}/100")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  {rec}")
    else:
        print("⚠️ Model not found. Please train model first using:")
        print("  jupyter notebook notebooks/02_train_inverse_model.ipynb")
