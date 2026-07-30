"""
Waymo Open Motion Dataset Loader

This module handles loading and processing Waymo Open Motion Dataset TFRecord files.
Extracts relevant features for crash analysis including:
- Vehicle trajectories and behaviors
- Pedestrian/cyclist interactions
- Environmental conditions
- Near-miss scenarios
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WaymoScenario:
    """Represents a single Waymo scenario with extracted features."""
    scenario_id: str
    timestamps: np.ndarray
    
    # Ego vehicle (self-driving car) data
    ego_trajectory: np.ndarray  # (timesteps, 2) - x, y positions
    ego_velocity: np.ndarray    # (timesteps,)
    ego_heading: np.ndarray     # (timesteps,)
    
    # Surrounding vehicles
    vehicle_trajectories: Dict[int, np.ndarray]  # vehicle_id -> (timesteps, 2)
    vehicle_velocities: Dict[int, np.ndarray]
    vehicle_types: Dict[int, int]
    
    # VRUs (Vulnerable Road Users)
    pedestrian_trajectories: Dict[int, np.ndarray]
    cyclist_trajectories: Dict[int, np.ndarray]
    
    # Road features
    road_lines: List[np.ndarray]
    road_edges: List[np.ndarray]
    crosswalks: List[np.ndarray]
    speed_limits: List[float]
    
    # Traffic signals/signs
    traffic_signals: Dict[int, str]  # signal_id -> state (green/yellow/red)
    stop_signs: List[np.ndarray]
    
    # Metadata
    scenario_type: str  # interactive, testing, etc.
    has_collision: bool
    has_near_miss: bool


class WaymoDataLoader:
    """
    Loads and processes Waymo Open Motion Dataset.
    
    Attributes:
        data_dir: Path to waymo/motion_dataset directory
        dataset_type: Type of dataset (training, validation, testing, etc.)
    """
    
    def __init__(self, data_dir: str = "waymo/motion_dataset", dataset_type: str = "training"):
        """
        Initialize Waymo data loader.
        
        Args:
            data_dir: Path to motion_dataset directory
            dataset_type: Dataset type (training, validation, testing, etc.)
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Define feature description for TFRecord parsing
        self._setup_feature_description()
    
    def _setup_feature_description(self):
        """Setup feature description for parsing TFRecords."""
        # This will be populated based on Waymo Open Motion Dataset format
        # Features include: state/past, state/current, state/future, roadgraph, traffic lights, etc.
        pass
    
    def get_tfrecord_files(self, limit: Optional[int] = None) -> List[Path]:
        """
        Get list of TFRecord files for the specified dataset type.
        
        Args:
            limit: Maximum number of files to return (None for all)
            
        Returns:
            List of TFRecord file paths
        """
        # Only use tf_example datasets (scenario format has different schema)
        tf_example_dir = self.data_dir / "tf_example_datasets" / self.dataset_type
        
        files = []
        
        # Collect tf_example files
        if tf_example_dir.exists():
            tf_files = sorted(tf_example_dir.glob("*.tfrecord*"))
            files.extend(tf_files)
            logger.info(f"Found {len(tf_files)} tf_example files in {tf_example_dir}")
        else:
            logger.warning(f"TF example directory not found: {tf_example_dir}")
        
        if limit:
            files = files[:limit]
        
        logger.info(f"Total files to process: {len(files)}")
        return files
    
    def parse_scenario_proto(self, serialized_example: bytes) -> Dict:
        """
        Parse a single scenario from TFRecord.
        
        Args:
            serialized_example: Serialized protocol buffer
            
        Returns:
            Dictionary with parsed scenario data
        """
        # Define the feature structure based on Waymo Open Motion Dataset v1.2
        feature_description = {
            'scenario/id': tf.io.FixedLenFeature([], tf.string),
            'state/id': tf.io.VarLenFeature(tf.float32),
            'state/type': tf.io.VarLenFeature(tf.float32),
            'state/x': tf.io.VarLenFeature(tf.float32),
            'state/y': tf.io.VarLenFeature(tf.float32),
            'state/length': tf.io.VarLenFeature(tf.float32),
            'state/width': tf.io.VarLenFeature(tf.float32),
            'state/height': tf.io.VarLenFeature(tf.float32),
            'state/heading': tf.io.VarLenFeature(tf.float32),
            'state/velocity_x': tf.io.VarLenFeature(tf.float32),
            'state/velocity_y': tf.io.VarLenFeature(tf.float32),
            'state/valid': tf.io.VarLenFeature(tf.float32),
            'state/is_sdc': tf.io.VarLenFeature(tf.int64),
            'state/tracks_to_predict': tf.io.VarLenFeature(tf.int64),
            
            # Roadgraph features
            'roadgraph_samples/id': tf.io.VarLenFeature(tf.int64),
            'roadgraph_samples/type': tf.io.VarLenFeature(tf.int64),
            'roadgraph_samples/xyz': tf.io.VarLenFeature(tf.float32),
            
            # Traffic light features
            'traffic_light_state/current/state': tf.io.VarLenFeature(tf.int64),
            'traffic_light_state/current/id': tf.io.VarLenFeature(tf.int64),
            'traffic_light_state/current/x': tf.io.VarLenFeature(tf.float32),
            'traffic_light_state/current/y': tf.io.VarLenFeature(tf.float32),
        }
        
        try:
            example = tf.io.parse_single_example(serialized_example, feature_description)
            
            # Convert sparse tensors to dense
            parsed = {}
            for key, value in example.items():
                if isinstance(value, tf.sparse.SparseTensor):
                    parsed[key] = tf.sparse.to_dense(value).numpy()
                else:
                    parsed[key] = value.numpy()
            
            return parsed
        except Exception as e:
            logger.error(f"Error parsing scenario: {e}")
            return {}
    
    def extract_features_from_scenario(self, parsed_scenario: Dict) -> WaymoScenario:
        """
        Extract relevant features from parsed scenario for crash analysis.
        
        Args:
            parsed_scenario: Dictionary from parse_scenario_proto
            
        Returns:
            WaymoScenario object with extracted features
        """
        # Extract scenario ID
        scenario_id = parsed_scenario.get('scenario/id', b'unknown').decode('utf-8')
        
        # Extract agent states
        agent_ids = parsed_scenario.get('state/id', np.array([]))
        agent_types = parsed_scenario.get('state/type', np.array([]))
        x_positions = parsed_scenario.get('state/x', np.array([]))
        y_positions = parsed_scenario.get('state/y', np.array([]))
        velocities_x = parsed_scenario.get('state/velocity_x', np.array([]))
        velocities_y = parsed_scenario.get('state/velocity_y', np.array([]))
        headings = parsed_scenario.get('state/heading', np.array([]))
        valid_mask = parsed_scenario.get('state/valid', np.array([]))
        is_sdc = parsed_scenario.get('state/is_sdc', np.array([]))
        
        # Reshape data (typical format: [num_agents * num_timesteps])
        # Waymo dataset typically has 91 timesteps (1 current + 10 past + 80 future)
        num_timesteps = 91  # This might vary, need to check actual data
        
        # Find ego vehicle (self-driving car)
        ego_idx = np.where(is_sdc == 1)[0]
        
        # Initialize scenario object
        # Note: This is a simplified version, actual implementation would need
        # proper reshaping based on the actual data structure
        
        scenario = WaymoScenario(
            scenario_id=scenario_id,
            timestamps=np.arange(num_timesteps) * 0.1,  # 10Hz sampling
            ego_trajectory=np.zeros((num_timesteps, 2)),
            ego_velocity=np.zeros(num_timesteps),
            ego_heading=np.zeros(num_timesteps),
            vehicle_trajectories={},
            vehicle_velocities={},
            vehicle_types={},
            pedestrian_trajectories={},
            cyclist_trajectories={},
            road_lines=[],
            road_edges=[],
            crosswalks=[],
            speed_limits=[],
            traffic_signals={},
            stop_signs=[],
            scenario_type=self.dataset_type,
            has_collision=False,  # Need to compute from trajectories
            has_near_miss=False   # Need to compute from trajectories
        )
        
        return scenario
    
    def load_scenarios(self, num_files: int = 1, max_scenarios: Optional[int] = None) -> List[WaymoScenario]:
        """
        Load scenarios from TFRecord files.
        
        Args:
            num_files: Number of TFRecord files to load
            max_scenarios: Maximum number of scenarios to load (None for all)
            
        Returns:
            List of WaymoScenario objects
        """
        tfrecord_files = self.get_tfrecord_files(limit=num_files)
        
        if not tfrecord_files:
            logger.warning("No TFRecord files found")
            return []
        
        scenarios = []
        
        for file_path in tfrecord_files:
            logger.info(f"Loading scenarios from {file_path.name}")
            
            try:
                # Create TFRecord dataset
                dataset = tf.data.TFRecordDataset(str(file_path), compression_type='')
                
                for raw_record in dataset:
                    if max_scenarios and len(scenarios) >= max_scenarios:
                        break
                    
                    parsed = self.parse_scenario_proto(raw_record.numpy())
                    if parsed:
                        scenario = self.extract_features_from_scenario(parsed)
                        scenarios.append(scenario)
                
                logger.info(f"Loaded {len(scenarios)} scenarios so far")
                
                if max_scenarios and len(scenarios) >= max_scenarios:
                    break
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        logger.info(f"Total scenarios loaded: {len(scenarios)}")
        return scenarios
    
    def compute_crash_indicators(self, scenario: WaymoScenario) -> Dict[str, any]:
        """
        Compute crash-related indicators from a scenario.
        
        Args:
            scenario: WaymoScenario object
            
        Returns:
            Dictionary with crash indicators
        """
        indicators = {
            'has_collision': False,
            'has_near_miss': False,
            'min_distance_to_vehicle': float('inf'),
            'min_distance_to_pedestrian': float('inf'),
            'min_distance_to_cyclist': float('inf'),
            'time_to_collision': float('inf'),
            'aggressive_acceleration': False,
            'aggressive_braking': False,
            'aggressive_lane_change': False,
            'speed_limit_violation': False,
            'red_light_running': False,
        }
        
        # Compute minimum distances
        for vehicle_traj in scenario.vehicle_trajectories.values():
            if len(vehicle_traj) > 0:
                distances = np.linalg.norm(scenario.ego_trajectory - vehicle_traj, axis=1)
                min_dist = np.min(distances[~np.isnan(distances)])
                indicators['min_distance_to_vehicle'] = min(
                    indicators['min_distance_to_vehicle'], min_dist
                )
        
        # Detect collisions (distance < threshold)
        if indicators['min_distance_to_vehicle'] < 2.0:  # 2 meters threshold
            indicators['has_collision'] = True
        elif indicators['min_distance_to_vehicle'] < 5.0:  # 5 meters threshold
            indicators['has_near_miss'] = True
        
        # Compute aggressive behaviors
        if len(scenario.ego_velocity) > 1:
            acceleration = np.diff(scenario.ego_velocity) / 0.1  # dt = 0.1s
            indicators['aggressive_acceleration'] = np.any(acceleration > 4.0)  # 4 m/s²
            indicators['aggressive_braking'] = np.any(acceleration < -4.0)
        
        return indicators
    
    def scenarios_to_dataframe(self, scenarios: List[WaymoScenario]) -> pd.DataFrame:
        """
        Convert list of scenarios to pandas DataFrame for analysis.
        
        Args:
            scenarios: List of WaymoScenario objects
            
        Returns:
            DataFrame with scenario features
        """
        data = []
        
        for scenario in scenarios:
            indicators = self.compute_crash_indicators(scenario)
            
            row = {
                'scenario_id': scenario.scenario_id,
                'scenario_type': scenario.scenario_type,
                'num_vehicles': len(scenario.vehicle_trajectories),
                'num_pedestrians': len(scenario.pedestrian_trajectories),
                'num_cyclists': len(scenario.cyclist_trajectories),
                'ego_max_speed': np.max(scenario.ego_velocity) if len(scenario.ego_velocity) > 0 else 0,
                'ego_mean_speed': np.mean(scenario.ego_velocity) if len(scenario.ego_velocity) > 0 else 0,
                **indicators
            }
            
            data.append(row)
        
        return pd.DataFrame(data)


class WaymoFeatureExtractor:
    """
    Extract crash-relevant features from Waymo scenarios for ML models.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract_temporal_features(self, scenario: WaymoScenario) -> Dict[str, float]:
        """Extract time-based features."""
        features = {}
        
        if len(scenario.ego_velocity) > 0:
            # Speed statistics
            features['speed_mean'] = np.mean(scenario.ego_velocity)
            features['speed_std'] = np.std(scenario.ego_velocity)
            features['speed_max'] = np.max(scenario.ego_velocity)
            features['speed_min'] = np.min(scenario.ego_velocity)
            
            # Acceleration statistics
            if len(scenario.ego_velocity) > 1:
                accel = np.diff(scenario.ego_velocity) / 0.1
                features['accel_mean'] = np.mean(accel)
                features['accel_std'] = np.std(accel)
                features['accel_max'] = np.max(accel)
                features['accel_min'] = np.min(accel)
        
        return features
    
    def extract_proximity_features(self, scenario: WaymoScenario) -> Dict[str, float]:
        """Extract proximity-based features."""
        features = {
            'min_dist_vehicle': float('inf'),
            'min_dist_pedestrian': float('inf'),
            'min_dist_cyclist': float('inf'),
            'avg_dist_vehicle': float('inf'),
            'num_close_vehicles': 0,  # within 20m
            'num_very_close_vehicles': 0,  # within 5m
        }
        
        # Compute distances to vehicles
        vehicle_distances = []
        for vehicle_traj in scenario.vehicle_trajectories.values():
            if len(vehicle_traj) > 0:
                distances = np.linalg.norm(scenario.ego_trajectory - vehicle_traj, axis=1)
                valid_distances = distances[~np.isnan(distances)]
                if len(valid_distances) > 0:
                    vehicle_distances.append(np.min(valid_distances))
        
        if vehicle_distances:
            features['min_dist_vehicle'] = min(vehicle_distances)
            features['avg_dist_vehicle'] = np.mean(vehicle_distances)
            features['num_close_vehicles'] = sum(1 for d in vehicle_distances if d < 20)
            features['num_very_close_vehicles'] = sum(1 for d in vehicle_distances if d < 5)
        
        return features
    
    def extract_interaction_features(self, scenario: WaymoScenario) -> Dict[str, float]:
        """Extract interaction-based features."""
        features = {
            'has_vru_interaction': 0,
            'has_vehicle_interaction': 0,
            'interaction_complexity': 0,  # number of agents within 30m
        }
        
        # Count nearby agents
        nearby_count = 0
        
        # Check vehicle interactions
        for vehicle_traj in scenario.vehicle_trajectories.values():
            if len(vehicle_traj) > 0:
                distances = np.linalg.norm(scenario.ego_trajectory - vehicle_traj, axis=1)
                if np.any(distances < 30):
                    nearby_count += 1
                if np.any(distances < 10):
                    features['has_vehicle_interaction'] = 1
        
        # Check VRU interactions
        for ped_traj in scenario.pedestrian_trajectories.values():
            if len(ped_traj) > 0:
                distances = np.linalg.norm(scenario.ego_trajectory - ped_traj, axis=1)
                if np.any(distances < 30):
                    nearby_count += 1
                if np.any(distances < 10):
                    features['has_vru_interaction'] = 1
        
        for cyc_traj in scenario.cyclist_trajectories.values():
            if len(cyc_traj) > 0:
                distances = np.linalg.norm(scenario.ego_trajectory - cyc_traj, axis=1)
                if np.any(distances < 30):
                    nearby_count += 1
                if np.any(distances < 10):
                    features['has_vru_interaction'] = 1
        
        features['interaction_complexity'] = nearby_count
        
        return features
    
    def extract_all_features(self, scenario: WaymoScenario) -> Dict[str, float]:
        """Extract all features from a scenario."""
        features = {}
        
        features.update(self.extract_temporal_features(scenario))
        features.update(self.extract_proximity_features(scenario))
        features.update(self.extract_interaction_features(scenario))
        
        return features
