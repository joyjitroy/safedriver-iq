"""
Driver Behavior Classification Module

Classifies driver behaviors from crash and good driving data using
multiple clustering and classification techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriverProfile:
    """Represents a driver behavior profile."""
    profile_id: int
    profile_name: str
    aggression_score: float
    risk_taking_score: float
    environmental_awareness_score: float
    vru_safety_score: float
    speed_behavior_score: float
    crash_likelihood: float
    characteristics: Dict[str, float]
    recommendations: List[str]


class DriverBehaviorClassifier:
    """
    Classify driver behaviors into distinct profiles.
    """
    
    def __init__(self, n_profiles: int = 5):
        """
        Initialize classifier.
        
        Args:
            n_profiles: Number of driver profiles to identify
        """
        self.n_profiles = n_profiles
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.pca_model = None
        self.profiles = {}
    
    def compute_behavior_scores(
        self,
        crash_data: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Compute behavior scores from crash data features.
        
        Args:
            crash_data: DataFrame with crash records
            feature_columns: List of feature column names
            
        Returns:
            DataFrame with computed behavior scores
        """
        logger.info("Computing driver behavior scores")
        
        scores = crash_data.copy()
        
        # Aggression score (speed, rush hour, aggressive maneuvers)
        aggression_features = [
            'HIGH_SPEED_ROAD', 'IS_RUSH_HOUR', 'aggressive_acceleration',
            'aggressive_lane_change'
        ]
        aggression_cols = [col for col in aggression_features if col in scores.columns]
        if aggression_cols:
            scores['aggression_score'] = scores[aggression_cols].sum(axis=1) / len(aggression_cols)
        else:
            scores['aggression_score'] = 0
        
        # Risk-taking score (night driving, adverse weather, distractions)
        risk_features = [
            'IS_NIGHT', 'ADVERSE_WEATHER', 'POOR_LIGHTING',
            'speed_limit_violation', 'red_light_running'
        ]
        risk_cols = [col for col in risk_features if col in scores.columns]
        if risk_cols:
            scores['risk_taking_score'] = scores[risk_cols].sum(axis=1) / len(risk_cols)
        else:
            scores['risk_taking_score'] = 0
        
        # Environmental awareness score (urban, VRU presence)
        env_features = [
            'IS_URBAN', 'LOW_SPEED_ROAD', 'total_vru',
            'has_vru_interaction'
        ]
        env_cols = [col for col in env_features if col in scores.columns]
        if env_cols:
            # Normalize VRU count if present
            if 'total_vru' in env_cols:
                scores['total_vru_normalized'] = (scores['total_vru'] > 0).astype(int)
                env_cols = [col if col != 'total_vru' else 'total_vru_normalized' for col in env_cols]
            scores['environmental_risk_score'] = scores[env_cols].sum(axis=1) / len(env_cols)
        else:
            scores['environmental_risk_score'] = 0
        
        # VRU safety score (inverse - higher is worse)
        vru_features = [
            'pedestrian_count', 'cyclist_count', 'fatal_vru'
        ]
        vru_cols = [col for col in vru_features if col in scores.columns]
        if vru_cols:
            scores['vru_safety_score'] = scores[vru_cols].sum(axis=1) / len(vru_cols)
        else:
            scores['vru_safety_score'] = 0
        
        # Speed behavior score
        if 'SPD_LIM' in scores.columns:
            # Normalize speed limit
            scores['speed_behavior_score'] = scores['SPD_LIM'] / 100.0
        else:
            scores['speed_behavior_score'] = 0
        
        logger.info("Behavior scores computed successfully")
        return scores
    
    def cluster_drivers(
        self,
        behavior_data: pd.DataFrame,
        score_columns: Optional[List[str]] = None,
        method: str = 'kmeans'
    ) -> pd.DataFrame:
        """
        Cluster drivers into behavior profiles.
        
        Args:
            behavior_data: DataFrame with behavior scores
            score_columns: Columns to use for clustering
            method: Clustering method ('kmeans', 'gmm', 'hierarchical', 'dbscan')
            
        Returns:
            DataFrame with cluster assignments
        """
        logger.info(f"Clustering drivers using {method}")
        
        if score_columns is None:
            score_columns = [
                'aggression_score', 'risk_taking_score', 
                'environmental_risk_score', 'vru_safety_score',
                'speed_behavior_score'
            ]
        
        # Select available score columns
        available_cols = [col for col in score_columns if col in behavior_data.columns]
        if not available_cols:
            raise ValueError("No valid score columns found in data")
        
        # Extract features for clustering
        X = behavior_data[available_cols].fillna(0)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply clustering method
        if method == 'kmeans':
            self.cluster_model = KMeans(
                n_clusters=self.n_profiles,
                random_state=42,
                n_init=10
            )
        elif method == 'gmm':
            self.cluster_model = GaussianMixture(
                n_components=self.n_profiles,
                random_state=42
            )
        elif method == 'hierarchical':
            self.cluster_model = AgglomerativeClustering(
                n_clusters=self.n_profiles
            )
        elif method == 'dbscan':
            self.cluster_model = DBSCAN(eps=0.5, min_samples=10)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Fit and predict
        behavior_data['cluster'] = self.cluster_model.fit_predict(X_scaled)
        
        # Store cluster characteristics
        self._analyze_clusters(behavior_data, available_cols)
        
        logger.info(f"Identified {behavior_data['cluster'].nunique()} driver profiles")
        return behavior_data
    
    def _analyze_clusters(
        self,
        behavior_data: pd.DataFrame,
        score_columns: List[str]
    ):
        """
        Analyze and characterize each cluster.
        
        Args:
            behavior_data: DataFrame with cluster assignments
            score_columns: Score columns used for clustering
        """
        logger.info("Analyzing cluster characteristics")
        
        for cluster_id in behavior_data['cluster'].unique():
            cluster_data = behavior_data[behavior_data['cluster'] == cluster_id]
            
            # Compute average scores
            characteristics = {}
            for col in score_columns:
                characteristics[col] = cluster_data[col].mean()
            
            # Determine profile name based on characteristics
            profile_name = self._determine_profile_name(characteristics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(characteristics)
            
            # Compute crash likelihood (higher scores = higher risk)
            crash_likelihood = np.mean(list(characteristics.values()))
            
            # Create profile
            profile = DriverProfile(
                profile_id=cluster_id,
                profile_name=profile_name,
                aggression_score=characteristics.get('aggression_score', 0),
                risk_taking_score=characteristics.get('risk_taking_score', 0),
                environmental_awareness_score=characteristics.get('environmental_risk_score', 0),
                vru_safety_score=characteristics.get('vru_safety_score', 0),
                speed_behavior_score=characteristics.get('speed_behavior_score', 0),
                crash_likelihood=crash_likelihood,
                characteristics=characteristics,
                recommendations=recommendations
            )
            
            self.profiles[cluster_id] = profile
    
    def _determine_profile_name(self, characteristics: Dict[str, float]) -> str:
        """
        Determine profile name based on characteristic scores.
        
        Args:
            characteristics: Dictionary of behavior scores
            
        Returns:
            Profile name string
        """
        aggression = characteristics.get('aggression_score', 0)
        risk = characteristics.get('risk_taking_score', 0)
        env_risk = characteristics.get('environmental_risk_score', 0)
        vru_risk = characteristics.get('vru_safety_score', 0)
        
        # High on multiple dimensions
        if aggression > 0.6 and risk > 0.6:
            return "High-Risk Aggressive Driver"
        elif risk > 0.7:
            return "Extreme Risk-Taker"
        elif aggression > 0.7:
            return "Aggressive Driver"
        elif vru_risk > 0.5:
            return "VRU-Unsafe Driver"
        elif env_risk > 0.6 and risk > 0.5:
            return "Urban Risk-Taker"
        elif aggression > 0.4 and risk > 0.4:
            return "Moderate Risk Driver"
        else:
            return "Cautious Crash-Involved Driver"
    
    def _generate_recommendations(
        self,
        characteristics: Dict[str, float]
    ) -> List[str]:
        """
        Generate safety recommendations based on characteristics.
        
        Args:
            characteristics: Dictionary of behavior scores
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        aggression = characteristics.get('aggression_score', 0)
        risk = characteristics.get('risk_taking_score', 0)
        env_risk = characteristics.get('environmental_risk_score', 0)
        vru_risk = characteristics.get('vru_safety_score', 0)
        
        if aggression > 0.5:
            recommendations.append("Reduce aggressive driving: maintain safe following distances")
            recommendations.append("Avoid rushing: plan departure times to reduce time pressure")
        
        if risk > 0.5:
            recommendations.append("Increase caution in adverse weather conditions")
            recommendations.append("Avoid driving during night hours when possible")
            recommendations.append("Always use proper lighting and visibility aids")
        
        if env_risk > 0.5:
            recommendations.append("Increase awareness in urban environments")
            recommendations.append("Reduce speed in pedestrian-dense areas")
        
        if vru_risk > 0.3:
            recommendations.append("CRITICAL: Enhanced pedestrian/cyclist awareness required")
            recommendations.append("Always yield to VRUs at crosswalks")
            recommendations.append("Extra caution near schools, parks, and residential areas")
        
        if not recommendations:
            recommendations.append("Maintain defensive driving practices")
            recommendations.append("Continue safe driving habits")
        
        return recommendations
    
    def classify_new_driver(
        self,
        driver_features: Dict[str, float]
    ) -> DriverProfile:
        """
        Classify a new driver based on their features.
        
        Args:
            driver_features: Dictionary of driver behavior features
            
        Returns:
            DriverProfile object
        """
        if self.cluster_model is None:
            raise ValueError("Model not trained. Run cluster_drivers first.")
        
        # Convert to DataFrame
        driver_df = pd.DataFrame([driver_features])
        
        # Compute behavior scores
        scored_df = self.compute_behavior_scores(driver_df, list(driver_features.keys()))
        
        # Extract score columns
        score_columns = [
            'aggression_score', 'risk_taking_score',
            'environmental_risk_score', 'vru_safety_score',
            'speed_behavior_score'
        ]
        X = scored_df[[col for col in score_columns if col in scored_df.columns]]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        cluster_id = self.cluster_model.predict(X_scaled)[0]
        
        # Return profile
        return self.profiles.get(cluster_id)
    
    def get_profile_summary(self) -> pd.DataFrame:
        """
        Get summary of all driver profiles.
        
        Returns:
            DataFrame with profile summaries
        """
        summaries = []
        
        for profile_id, profile in self.profiles.items():
            summaries.append({
                'Profile ID': profile.profile_id,
                'Profile Name': profile.profile_name,
                'Aggression': f"{profile.aggression_score:.2f}",
                'Risk-Taking': f"{profile.risk_taking_score:.2f}",
                'Environmental Risk': f"{profile.environmental_awareness_score:.2f}",
                'VRU Risk': f"{profile.vru_safety_score:.2f}",
                'Crash Likelihood': f"{profile.crash_likelihood:.2f}",
                'Top Recommendation': profile.recommendations[0] if profile.recommendations else "N/A"
            })
        
        return pd.DataFrame(summaries).sort_values('Crash Likelihood', ascending=False)
    
    def visualize_profiles(self):
        """
        Visualize driver profiles in 2D using PCA.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if not self.profiles:
            logger.warning("No profiles to visualize")
            return
        
        # Extract profile characteristics
        profile_data = []
        profile_names = []
        
        for profile in self.profiles.values():
            profile_data.append([
                profile.aggression_score,
                profile.risk_taking_score,
                profile.environmental_awareness_score,
                profile.vru_safety_score,
                profile.speed_behavior_score
            ])
            profile_names.append(profile.profile_name)
        
        profile_data = np.array(profile_data)
        
        # PCA for visualization
        if profile_data.shape[1] > 2:
            pca = PCA(n_components=3)
            profile_data_pca = pca.fit_transform(profile_data)
        else:
            profile_data_pca = profile_data
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 6))
        
        # 3D scatter
        ax1 = fig.add_subplot(121, projection='3d')
        scatter = ax1.scatter(
            profile_data_pca[:, 0],
            profile_data_pca[:, 1],
            profile_data_pca[:, 2] if profile_data_pca.shape[1] > 2 else np.zeros(len(profile_data_pca)),
            c=range(len(profile_names)),
            cmap='viridis',
            s=200,
            alpha=0.7
        )
        
        # Add labels
        for i, name in enumerate(profile_names):
            ax1.text(
                profile_data_pca[i, 0],
                profile_data_pca[i, 1],
                profile_data_pca[i, 2] if profile_data_pca.shape[1] > 2 else 0,
                name,
                fontsize=8
            )
        
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        ax1.set_title('Driver Profiles (PCA)', fontweight='bold')
        
        # Radar chart for profile characteristics
        ax2 = fig.add_subplot(122, projection='polar')
        
        categories = ['Aggression', 'Risk-Taking', 'Env. Risk', 'VRU Risk', 'Speed']
        N = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        
        for i, profile in enumerate(self.profiles.values()):
            values = [
                profile.aggression_score,
                profile.risk_taking_score,
                profile.environmental_awareness_score,
                profile.vru_safety_score,
                profile.speed_behavior_score
            ]
            values += values[:1]
            
            ax2.plot(angles, values, 'o-', linewidth=2, label=profile.profile_name)
            ax2.fill(angles, values, alpha=0.15)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 1)
        ax2.set_title('Profile Characteristics', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


class GoodDriverProfileExtractor:
    """
    Extract characteristics of good drivers from non-crash data.
    """
    
    def __init__(self):
        """Initialize extractor."""
        self.good_driver_profile = None
    
    def extract_from_waymo(
        self,
        waymo_scenarios: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract good driver characteristics from Waymo data.
        
        Args:
            waymo_scenarios: DataFrame with Waymo scenarios
            
        Returns:
            Dictionary with good driver characteristics
        """
        logger.info("Extracting good driver profile from Waymo data")
        
        # Filter for safe scenarios (no collisions, no near-misses)
        safe_scenarios = waymo_scenarios[
            (waymo_scenarios['has_collision'] == False) &
            (waymo_scenarios['has_near_miss'] == False)
        ]
        
        if len(safe_scenarios) == 0:
            logger.warning("No safe scenarios found in Waymo data")
            return {}
        
        profile = {
            'average_speed': safe_scenarios['ego_mean_speed'].mean(),
            'min_safe_distance': safe_scenarios['min_distance_to_vehicle'].mean(),
            'vru_interaction_rate': (safe_scenarios['num_pedestrians'] > 0).mean(),
            'safe_scenario_count': len(safe_scenarios),
            'aggressive_behavior_rate': 0.0,  # Assumed low for safe scenarios
            'night_driving_rate': 0.0,  # Would need time-of-day data
        }
        
        self.good_driver_profile = profile
        
        logger.info("Good driver profile extracted successfully")
        return profile
    
    def compare_to_crash_drivers(
        self,
        crash_driver_profiles: Dict[int, DriverProfile]
    ) -> pd.DataFrame:
        """
        Compare good driver profile to crash driver profiles.
        
        Args:
            crash_driver_profiles: Dictionary of crash driver profiles
            
        Returns:
            DataFrame with comparison results
        """
        if self.good_driver_profile is None:
            raise ValueError("Good driver profile not extracted yet")
        
        comparisons = []
        
        for profile_id, crash_profile in crash_driver_profiles.items():
            comparison = {
                'Profile': crash_profile.profile_name,
                'Aggression Gap': crash_profile.aggression_score - 0.1,  # Good drivers ~0.1
                'Risk-Taking Gap': crash_profile.risk_taking_score - 0.1,
                'VRU Safety Gap': crash_profile.vru_safety_score - 0.0,  # Good drivers ~0.0
                'Overall Safety Gap': crash_profile.crash_likelihood - 0.1
            }
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons).sort_values('Overall Safety Gap', ascending=False)
