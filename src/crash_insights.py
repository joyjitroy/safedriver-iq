"""
Crash Factor Investigation Insights Module

Provides crash analysis insights for the SafeDriver-IQ dashboard.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CrashInsightsAnalyzer:
    """Analyzer for crash factor investigation insights."""
    
    def __init__(self, results_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Path to results directory containing crash investigation outputs
        """
        self.results_dir = Path(results_dir)
        
        # Load crash prediction model
        model_path = self.results_dir / 'crash_investigation_rf_model.pkl'
        if model_path.exists():
            self.crash_model = joblib.load(model_path)
            # Store the feature names the model was trained with
            if hasattr(self.crash_model, 'feature_names_in_'):
                self.model_features = list(self.crash_model.feature_names_in_)
                print(f"✓ Crash model loaded with {len(self.model_features)} features: {self.model_features}")
            else:
                self.model_features = None
                print(f"⚠ Crash model loaded but feature names not available")
        else:
            self.crash_model = None
            self.model_features = None
            print(f"⚠ Crash model not found at {model_path}")
        
        # Load feature importance
        fi_path = self.results_dir / 'crash_investigation_feature_importance.csv'
        if fi_path.exists():
            self.feature_importance = pd.read_csv(fi_path)
        else:
            self.feature_importance = None
        
        # Load behavior clusters
        bc_path = self.results_dir / 'crash_investigation_behavior_clusters.csv'
        if bc_path.exists():
            self.behavior_clusters = pd.read_csv(bc_path)
        else:
            self.behavior_clusters = None
        
        # Define high-risk patterns from investigation
        self.high_risk_patterns = {
            'Night + Bad Weather': {
                'conditions': ['IS_NIGHT', 'ADVERSE_WEATHER'],
                'historical_crashes': '8,234',
                'risk_multiplier': 2.8
            },
            'Urban + Rush Hour': {
                'conditions': ['IS_URBAN', 'IS_RUSH_HOUR'],
                'historical_crashes': '12,456',
                'risk_multiplier': 2.1
            },
            'VRU + Poor Lighting': {
                'conditions': ['total_vru', 'POOR_LIGHTING'],
                'historical_crashes': '3,892',
                'risk_multiplier': 3.5
            },
            'High Speed + Poor Conditions': {
                'conditions': ['HIGH_SPEED_ROAD', 'ADVERSE_CONDITIONS'],
                'historical_crashes': '6,721',
                'risk_multiplier': 2.9
            }
        }
        
        # Crash factor descriptions
        self.crash_factors = {
            'IS_NIGHT': {
                'name': 'Night Driving',
                'description': 'Reduced visibility increases crash risk',
                'prevention': 'Avoid unnecessary night trips, use high beams when safe'
            },
            'ADVERSE_WEATHER': {
                'name': 'Adverse Weather',
                'description': 'Rain, snow, or fog reduces traction and visibility',
                'prevention': 'Reduce speed, increase following distance, delay trip if possible'
            },
            'POOR_LIGHTING': {
                'name': 'Poor Lighting',
                'description': 'Inadequate road lighting at night',
                'prevention': 'Use headlights, reduce speed, stay alert for pedestrians'
            },
            'IS_URBAN': {
                'name': 'Urban Environment',
                'description': 'Complex traffic with pedestrians and cyclists',
                'prevention': 'Lower speed, scan for VRUs, yield at crosswalks'
            },
            'IS_RUSH_HOUR': {
                'name': 'Rush Hour Traffic',
                'description': 'Congestion increases collision risk',
                'prevention': 'Maintain safe following distance, avoid aggressive lane changes'
            },
            'HIGH_SPEED_ROAD': {
                'name': 'High-Speed Road',
                'description': 'Highway driving at high speeds',
                'prevention': 'Maintain attention, check blind spots, avoid distractions'
            }
        }
    
    def predict_crash_probability(self, scenario: Dict) -> Dict:
        """
        Predict crash probability for a scenario.
        
        Args:
            scenario: Dictionary with scenario features
        
        Returns:
            Dictionary with crash prediction results
        """
        if self.crash_model is None:
            return {'error': 'Crash model not available'}
        
        # Use the features the model was actually trained with
        if self.model_features is not None:
            expected_features = self.model_features
        else:
            # Fallback: try common features
            expected_features = ['IS_RUSH_HOUR', 'IS_NIGHT', 'ADVERSE_WEATHER', 
                               'POOR_LIGHTING', 'IS_URBAN']
        
        # Build feature vector as DataFrame with proper column names
        features = {}
        for feat in expected_features:
            features[feat] = [scenario.get(feat, 0)]
        
        # Predict - pass DataFrame with column names to avoid sklearn warning
        features_df = pd.DataFrame(features)
        crash_prob = self.crash_model.predict_proba(features_df)[0][1]  # Probability of crash
        
        # Classify risk level based on crash probability
        if crash_prob >= 0.7:
            risk_level = 'Critical'
        elif crash_prob >= 0.5:
            risk_level = 'High'
        elif crash_prob >= 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'crash_probability': crash_prob,
            'risk_level': risk_level,
            'confidence': abs(crash_prob - 0.5) * 2  # Higher confidence when away from 0.5
        }
    
    def identify_active_risk_factors(self, scenario: Dict) -> List[Dict]:
        """
        Identify which crash factors are active in the scenario.
        
        Args:
            scenario: Dictionary with scenario features
        
        Returns:
            List of active risk factors with details
        """
        active_factors = []
        
        for factor_key, factor_info in self.crash_factors.items():
            is_active = False
            
            # Check if factor is active
            if factor_key in scenario:
                if scenario[factor_key] == 1:
                    is_active = True
                elif factor_key == 'total_vru' and scenario.get(factor_key, 0) > 0:
                    is_active = True
            
            if is_active:
                # Get importance if available
                importance = 0.0
                if self.feature_importance is not None:
                    feat_row = self.feature_importance[
                        self.feature_importance['Feature'] == factor_key
                    ]
                    if not feat_row.empty:
                        importance = feat_row['Average_Importance'].values[0]
                
                active_factors.append({
                    'key': factor_key,
                    'name': factor_info['name'],
                    'description': factor_info['description'],
                    'prevention': factor_info['prevention'],
                    'importance': importance
                })
        
        # Sort by importance
        active_factors.sort(key=lambda x: x['importance'], reverse=True)
        
        return active_factors
    
    def identify_high_risk_patterns(self, scenario: Dict) -> List[Dict]:
        """
        Identify which high-risk patterns are present in the scenario.
        
        Args:
            scenario: Dictionary with scenario features
        
        Returns:
            List of matching high-risk patterns
        """
        matching_patterns = []
        
        for pattern_name, pattern_info in self.high_risk_patterns.items():
            # Check if all conditions are met
            all_conditions_met = True
            for condition in pattern_info['conditions']:
                if condition in scenario:
                    if condition == 'total_vru':
                        if scenario.get(condition, 0) <= 0:
                            all_conditions_met = False
                            break
                    elif scenario.get(condition, 0) != 1:
                        all_conditions_met = False
                        break
                else:
                    all_conditions_met = False
                    break
            
            if all_conditions_met:
                matching_patterns.append({
                    'name': pattern_name,
                    'historical_crashes': pattern_info['historical_crashes'],
                    'risk_multiplier': pattern_info['risk_multiplier'],
                    'warning': f"⚠️ {pattern_name} increases crash risk by {pattern_info['risk_multiplier']}x"
                })
        
        return matching_patterns
    
    def classify_driver_behavior(self, scenario: Dict) -> Optional[Dict]:
        """
        Classify driver behavior based on scenario.
        
        Args:
            scenario: Dictionary with scenario features
        
        Returns:
            Behavior classification or None if not available
        """
        # Calculate behavior scores
        aggression_score = 0
        risk_taking_score = 0
        environmental_risk_score = 0
        
        # Aggression
        if scenario.get('HIGH_SPEED_ROAD', 0) == 1:
            aggression_score += 1
        if scenario.get('IS_RUSH_HOUR', 0) == 1:
            aggression_score += 1
        
        # Risk-taking
        if scenario.get('IS_NIGHT', 0) == 1:
            risk_taking_score += 1
        if scenario.get('ADVERSE_WEATHER', 0) == 1:
            risk_taking_score += 1
        if scenario.get('POOR_LIGHTING', 0) == 1:
            risk_taking_score += 1
        
        # Environmental risk
        if scenario.get('IS_URBAN', 0) == 1:
            environmental_risk_score += 1
        if scenario.get('LOW_SPEED_ROAD', 0) == 1:
            environmental_risk_score += 1
        
        # Classify based on thresholds
        if aggression_score > 0 and risk_taking_score > 1:
            behavior_type = "Aggressive Risk-Taker"
            description = "High-risk driving in challenging conditions"
            advice = "Reduce aggression and avoid driving in poor conditions"
        elif aggression_score > 0:
            behavior_type = "Aggressive Driver"
            description = "Fast-paced driving in high-demand situations"
            advice = "Reduce speed and maintain safer following distances"
        elif risk_taking_score > 1:
            behavior_type = "Environmental Risk-Taker"
            description = "Driving in challenging environmental conditions"
            advice = "Delay trip or take extra precautions in poor conditions"
        else:
            behavior_type = "Cautious Driver"
            description = "Relatively safe driving conditions selected"
            advice = "Continue safe practices and stay alert"
        
        return {
            'type': behavior_type,
            'description': description,
            'advice': advice,
            'aggression_score': aggression_score,
            'risk_taking_score': risk_taking_score,
            'environmental_risk_score': environmental_risk_score
        }
    
    def get_top_features(self, n: int = 5) -> Optional[pd.DataFrame]:
        """
        Get top N most important features.
        
        Args:
            n: Number of features to return
        
        Returns:
            DataFrame with top features or None
        """
        if self.feature_importance is None:
            return None
        
        return self.feature_importance.head(n)
    
    def get_crash_statistics(self) -> Dict:
        """
        Get overall crash statistics from the investigation.
        
        Returns:
            Dictionary with crash statistics
        """
        return {
            'total_crashes_analyzed': '417,335',
            'vru_crashes': '38,462',
            'years_analyzed': '2016-2023 (8 years)',
            'features_engineered': '120+',
            'models_compared': '4 (RF, XGBoost, Permutation, SHAP)',
            'top_risk_factor': 'Night driving + Poor lighting',
            'highest_risk_multiplier': '3.5x (VRU + Poor Lighting)',
            'data_source': 'NHTSA CRSS (Crash Report Sampling System)'
        }
    
    def compare_predictions(self, scenario: Dict, safety_score: float, 
                          safety_risk_level: str) -> Dict:
        """
        Compare inverse model (safety score) vs direct model (crash probability).
        
        Args:
            scenario: Scenario features
            safety_score: Safety score from inverse model
            safety_risk_level: Risk level from inverse model
        
        Returns:
            Comparison analysis
        """
        crash_pred = self.predict_crash_probability(scenario)
        
        # Calculate agreement
        # Safety score: 85+ = Low, 75-85 = Medium, 60-75 = High, <60 = Critical
        # Crash prob: <0.3 = Low, 0.3-0.5 = Medium, 0.5-0.7 = High, 0.7+ = Critical
        
        agreement = "Agreement"
        if (safety_risk_level in ['Low', 'Excellent'] and crash_pred['risk_level'] == 'Low') or \
           (safety_risk_level == 'Medium' and crash_pred['risk_level'] in ['Low', 'Medium']) or \
           (safety_risk_level == 'High' and crash_pred['risk_level'] in ['Medium', 'High']) or \
           (safety_risk_level == 'Critical' and crash_pred['risk_level'] in ['High', 'Critical']):
            agreement = "Strong Agreement"
        elif abs(self._risk_to_num(safety_risk_level) - self._risk_to_num(crash_pred['risk_level'])) <= 1:
            agreement = "Partial Agreement"
        else:
            agreement = "Disagreement"
        
        return {
            'crash_probability': crash_pred['crash_probability'],
            'crash_risk_level': crash_pred['risk_level'],
            'safety_score': safety_score,
            'safety_risk_level': safety_risk_level,
            'agreement': agreement,
            'interpretation': self._interpret_comparison(
                safety_score, crash_pred['crash_probability'], agreement
            )
        }
    
    def _risk_to_num(self, risk_level: str) -> int:
        """Convert risk level to number for comparison."""
        mapping = {'Low': 1, 'Excellent': 0, 'Medium': 2, 'High': 3, 'Critical': 4}
        return mapping.get(risk_level, 2)
    
    def _interpret_comparison(self, safety_score: float, crash_prob: float, 
                            agreement: str) -> str:
        """Interpret the comparison between models."""
        if agreement == "Strong Agreement":
            if safety_score >= 85 and crash_prob < 0.3:
                return "✅ Both models indicate SAFE conditions. Maintain current practices."
            elif safety_score < 60 and crash_prob > 0.7:
                return "⚠️ Both models indicate HIGH RISK. Take immediate action to improve conditions."
            else:
                return "⚠️ Both models indicate MODERATE RISK. Follow recommendations to improve."
        elif agreement == "Partial Agreement":
            return "⚠️ Models show similar but slightly different risk assessments. Exercise caution."
        else:
            return "⚠️ Models disagree. Consider the more conservative risk assessment and exercise extra caution."
