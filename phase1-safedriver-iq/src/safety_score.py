"""
Safety Score Calculator Module

Computes safety scores and provides driver feedback.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyScoreCalculator:
    """
    Calculates safety scores and generates driver feedback.
    
    Transforms model predictions into actionable safety metrics.
    """
    
    def __init__(self, model):
        """
        Initialize calculator with trained model.
        
        Args:
            model: Trained SafetyScoreModel instance
        """
        self.model = model
        
        # Define safety thresholds
        self.thresholds = {
            'critical': 40,
            'high_risk': 60,
            'medium_risk': 75,
            'low_risk': 85,
        }
    
    def calculate_score(self, features: pd.DataFrame) -> np.ndarray:
        """
        Calculate safety score for given features.
        
        Args:
            features: DataFrame with driving scenario features
            
        Returns:
            Array of safety scores (0-100)
        """
        return self.model.predict_safety_score(features)
    
    def get_risk_level(self, score: float) -> str:
        """
        Determine risk level from safety score.
        
        Args:
            score: Safety score (0-100)
            
        Returns:
            Risk level string
        """
        if score < self.thresholds['critical']:
            return 'CRITICAL'
        elif score < self.thresholds['high_risk']:
            return 'HIGH'
        elif score < self.thresholds['medium_risk']:
            return 'MEDIUM'
        elif score < self.thresholds['low_risk']:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def generate_alert(self, score: float, features: Dict) -> str:
        """
        Generate safety alert message.
        
        Args:
            score: Safety score
            features: Dictionary of current scenario features
            
        Returns:
            Alert message string
        """
        risk = self.get_risk_level(score)
        
        if risk == 'CRITICAL':
            return f"⚠️ CRITICAL: Safety score {score:.0f}/100. Immediate action required!"
        elif risk == 'HIGH':
            return f"⚠️ HIGH RISK: Safety score {score:.0f}/100. Exercise extreme caution."
        elif risk == 'MEDIUM':
            return f"⚠️ MEDIUM RISK: Safety score {score:.0f}/100. Increase vigilance."
        elif risk == 'MODERATE':
            return f"ℹ️ MODERATE: Safety score {score:.0f}/100. Stay alert."
        else:
            return f"✓ LOW RISK: Safety score {score:.0f}/100. Continue safe driving."
    
    def compare_to_good_driver(
        self, 
        current_score: float, 
        good_driver_score: float = 90.0
    ) -> Dict[str, float]:
        """
        Compare current score to good driver benchmark.
        
        Args:
            current_score: Current safety score
            good_driver_score: Benchmark good driver score (default 90)
            
        Returns:
            Dictionary with comparison metrics
        """
        gap = good_driver_score - current_score
        percentile = (current_score / good_driver_score) * 100
        
        return {
            'current_score': current_score,
            'good_driver_score': good_driver_score,
            'gap': gap,
            'percentile': percentile,
        }
    
    def suggest_improvements(self, features: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """
        Suggest improvements based on feature importance.
        
        Args:
            features: Current scenario features
            top_n: Number of suggestions to generate
            
        Returns:
            List of improvement suggestions
        """
        feature_importance = self.model.get_feature_importance(top_n)
        
        suggestions = []
        
        for _, row in feature_importance.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            suggestion = {
                'feature': feature,
                'importance': importance,
                'recommendation': self._get_recommendation(feature, features)
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _get_recommendation(self, feature: str, features: pd.DataFrame) -> str:
        """
        Generate specific recommendation for a feature.
        
        Args:
            feature: Feature name
            features: Current feature values
            
        Returns:
            Recommendation string
        """
        # Simple rule-based recommendations
        if 'SPEED' in feature or 'SPD' in feature:
            return "Reduce speed to increase safety margin"
        elif 'NIGHT' in feature or 'DARK' in feature:
            return "Increase vigilance in low visibility conditions"
        elif 'URBAN' in feature:
            return "Watch for pedestrians and cyclists in urban areas"
        elif 'WEATHER' in feature or 'ADVERSE' in feature:
            return "Adjust driving for weather conditions"
        elif 'WEEKEND' in feature:
            return "Be aware of increased traffic and impaired drivers"
        else:
            return f"Monitor {feature} to improve safety"
    
    def generate_safety_report(
        self, 
        features: pd.DataFrame,
        scores: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Generate comprehensive safety report.
        
        Args:
            features: Scenario features
            scores: Pre-calculated scores (optional)
            
        Returns:
            Dictionary with complete safety analysis
        """
        if scores is None:
            scores = self.calculate_score(features)
        
        avg_score = scores.mean()
        min_score = scores.min()
        max_score = scores.max()
        
        risk_distribution = pd.Series([self.get_risk_level(s) for s in scores]).value_counts()
        
        report = {
            'summary': {
                'average_score': avg_score,
                'min_score': min_score,
                'max_score': max_score,
                'total_scenarios': len(scores),
            },
            'risk_distribution': risk_distribution.to_dict(),
            'comparison': self.compare_to_good_driver(avg_score),
            'top_improvements': self.suggest_improvements(features),
        }
        
        return report


if __name__ == "__main__":
    print("SafetyScoreCalculator module loaded")
