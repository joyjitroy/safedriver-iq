"""
Driving Scenario Simulator

Simulates driving scenarios for testing the Agentic AI system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import random


class ScenarioSimulator:
    """
    Generates and simulates driving scenarios for testing.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize simulator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.current_scenario = None
    
    def generate_scenario(self, scenario_type: str = 'random') -> Dict:
        """
        Generate a driving scenario.
        
        Args:
            scenario_type: Type of scenario to generate
                'random', 'school_zone', 'construction', 'vru_crossing',
                'night_rain', 'high_speed', 'ice_road'
                
        Returns:
            Scenario dictionary with sensor data, location data, driver data
        """
        if scenario_type == 'school_zone':
            return self._generate_school_zone()
        elif scenario_type == 'construction':
            return self._generate_construction_zone()
        elif scenario_type == 'vru_crossing':
            return self._generate_vru_crossing()
        elif scenario_type == 'night_rain':
            return self._generate_night_rain()
        elif scenario_type == 'high_speed':
            return self._generate_high_speed()
        elif scenario_type == 'ice_road':
            return self._generate_ice_road()
        else:
            return self._generate_random()
    
    def _generate_school_zone(self) -> Dict:
        """Generate school zone scenario"""
        return {
            'sensor_data': {
                'speed_mph': random.uniform(15, 35),
                'acceleration': random.uniform(-0.5, 0.5),
                'road_surface': 'dry',
                'road_quality': random.uniform(80, 100),
                'weather': 'clear',
                'visibility_meters': random.uniform(500, 1000),
                'ambient_light_lux': random.uniform(10000, 50000),
                'vru_detections': self._generate_vru_detections(
                    num_pedestrians=(2, 5),
                    num_cyclists=(0, 2),
                    distances=(10, 50)
                )
            },
            'location_data': {
                'type': 'urban',
                'construction_zone': False,
                'tags': ['school', 'school_zone'],
                'latitude': 37.7749,
                'longitude': -122.4194
            },
            'driver_data': {
                'attention_level': random.uniform(70, 100),
                'recent_overrides': random.randint(0, 2)
            }
        }
    
    def _generate_construction_zone(self) -> Dict:
        """Generate construction zone scenario"""
        return {
            'sensor_data': {
                'speed_mph': random.uniform(20, 45),
                'acceleration': random.uniform(-1.0, 0.5),
                'road_surface': random.choice(['dry', 'gravel']),
                'road_quality': random.uniform(40, 70),
                'weather': 'clear',
                'visibility_meters': random.uniform(300, 800),
                'ambient_light_lux': random.uniform(5000, 40000),
                'vru_detections': self._generate_vru_detections(
                    num_pedestrians=(1, 3),  # Construction workers
                    num_cyclists=(0, 0),
                    distances=(20, 60)
                )
            },
            'location_data': {
                'type': 'urban',
                'construction_zone': True,
                'tags': ['construction', 'work_zone'],
                'latitude': 37.7749,
                'longitude': -122.4194
            },
            'driver_data': {
                'attention_level': random.uniform(60, 90),
                'recent_overrides': random.randint(0, 3)
            }
        }
    
    def _generate_vru_crossing(self) -> Dict:
        """Generate VRU crossing scenario"""
        return {
            'sensor_data': {
                'speed_mph': random.uniform(25, 40),
                'acceleration': random.uniform(-0.5, 0.0),
                'road_surface': 'dry',
                'road_quality': random.uniform(80, 100),
                'weather': 'clear',
                'visibility_meters': random.uniform(500, 1000),
                'ambient_light_lux': random.uniform(10000, 50000),
                'vru_detections': self._generate_vru_detections(
                    num_pedestrians=(1, 2),
                    num_cyclists=(0, 1),
                    distances=(5, 25),  # Close!
                    crossing=True
                )
            },
            'location_data': {
                'type': 'urban',
                'construction_zone': False,
                'tags': ['crosswalk'],
                'latitude': 37.7749,
                'longitude': -122.4194
            },
            'driver_data': {
                'attention_level': random.uniform(50, 80),
                'recent_overrides': random.randint(0, 1)
            }
        }
    
    def _generate_night_rain(self) -> Dict:
        """Generate night + rain scenario"""
        return {
            'sensor_data': {
                'speed_mph': random.uniform(30, 50),
                'acceleration': random.uniform(-0.3, 0.3),
                'road_surface': 'wet',
                'road_quality': random.uniform(70, 90),
                'weather': 'rain',
                'visibility_meters': random.uniform(100, 300),
                'ambient_light_lux': random.uniform(10, 100),  # Dark
                'vru_detections': self._generate_vru_detections(
                    num_pedestrians=(0, 2),
                    num_cyclists=(0, 1),
                    distances=(15, 50)
                )
            },
            'location_data': {
                'type': 'urban',
                'construction_zone': False,
                'tags': [],
                'latitude': 37.7749,
                'longitude': -122.4194
            },
            'driver_data': {
                'attention_level': random.uniform(70, 95),
                'recent_overrides': random.randint(0, 2)
            }
        }
    
    def _generate_high_speed(self) -> Dict:
        """Generate high-speed highway scenario"""
        return {
            'sensor_data': {
                'speed_mph': random.uniform(60, 75),
                'acceleration': random.uniform(-0.2, 0.5),
                'road_surface': 'dry',
                'road_quality': random.uniform(85, 100),
                'weather': 'clear',
                'visibility_meters': random.uniform(800, 2000),
                'ambient_light_lux': random.uniform(20000, 50000),
                'vru_detections': []  # Rare VRUs on highway
            },
            'location_data': {
                'type': 'rural',
                'construction_zone': False,
                'tags': ['highway'],
                'latitude': 37.7749,
                'longitude': -122.4194
            },
            'driver_data': {
                'attention_level': random.uniform(80, 100),
                'recent_overrides': random.randint(0, 1)
            }
        }
    
    def _generate_ice_road(self) -> Dict:
        """Generate icy road scenario"""
        return {
            'sensor_data': {
                'speed_mph': random.uniform(20, 35),
                'acceleration': random.uniform(-0.5, 0.2),
                'road_surface': 'icy',
                'road_quality': random.uniform(30, 60),
                'weather': 'snow',
                'visibility_meters': random.uniform(200, 500),
                'ambient_light_lux': random.uniform(5000, 20000),
                'road_temp_c': random.uniform(-5, 1),
                'moisture_level': 0.8,
                'vru_detections': self._generate_vru_detections(
                    num_pedestrians=(0, 1),
                    num_cyclists=(0, 0),
                    distances=(20, 60)
                )
            },
            'location_data': {
                'type': 'suburban',
                'construction_zone': False,
                'tags': [],
                'latitude': 37.7749,
                'longitude': -122.4194
            },
            'driver_data': {
                'attention_level': random.uniform(80, 100),
                'recent_overrides': random.randint(0, 1)
            }
        }
    
    def _generate_random(self) -> Dict:
        """Generate random scenario"""
        scenarios = [
            'school_zone', 'construction', 'vru_crossing',
            'night_rain', 'high_speed', 'ice_road'
        ]
        scenario_type = random.choice(scenarios)
        return self.generate_scenario(scenario_type)
    
    def _generate_vru_detections(
        self,
        num_pedestrians: Tuple[int, int],
        num_cyclists: Tuple[int, int],
        distances: Tuple[float, float],
        crossing: bool = False
    ) -> List[Dict]:
        """
        Generate VRU detections.
        
        Args:
            num_pedestrians: (min, max) pedestrians
            num_cyclists: (min, max) cyclists
            distances: (min, max) distance in meters
            crossing: Whether VRUs are crossing path
        """
        detections = []
        
        # Pedestrians
        n_peds = random.randint(*num_pedestrians)
        for _ in range(n_peds):
            dist = random.uniform(*distances)
            angle = random.uniform(60, 120) if crossing else random.uniform(-30, 30)
            
            detections.append({
                'type': 'pedestrian',
                'distance_m': dist,
                'velocity_ms': random.uniform(0.5, 2.0),
                'angle_deg': angle,
                'x': dist * np.cos(np.radians(angle)),
                'y': dist * np.sin(np.radians(angle))
            })
        
        # Cyclists
        n_cyclists = random.randint(*num_cyclists)
        for _ in range(n_cyclists):
            dist = random.uniform(*distances)
            angle = random.uniform(60, 120) if crossing else random.uniform(-30, 30)
            
            detections.append({
                'type': 'cyclist',
                'distance_m': dist,
                'velocity_ms': random.uniform(3.0, 8.0),
                'angle_deg': angle,
                'x': dist * np.cos(np.radians(angle)),
                'y': dist * np.sin(np.radians(angle))
            })
        
        return detections
    
    def run_scenario_sequence(
        self,
        scenario_types: List[str],
        duration_per_scenario: int = 5
    ) -> List[Dict]:
        """
        Run a sequence of scenarios.
        
        Args:
            scenario_types: List of scenario types to run
            duration_per_scenario: Simulated seconds per scenario
            
        Returns:
            List of scenario results
        """
        results = []
        
        for scenario_type in scenario_types:
            print(f"\n{'='*60}")
            print(f"Running scenario: {scenario_type}")
            print(f"{'='*60}")
            
            scenario = self.generate_scenario(scenario_type)
            
            result = {
                'type': scenario_type,
                'scenario': scenario,
                'duration': duration_per_scenario,
                'timestamp': datetime.now()
            }
            
            results.append(result)
        
        return results
