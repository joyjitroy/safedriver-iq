"""
Perception Engine Module

Monitors environment and builds context for decision-making.
Integrates sensor data and historical patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

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


class PerceptionEngine:
    """
    Monitors environment and builds driving context.
    
    Integrates multiple data sources:
    - Vehicle sensors (speed, acceleration)
    - Environmental sensors (weather, visibility)
    - VRU detection systems (cameras, radar)
    - Map data (location, zones)
    - Historical crash database
    - Driver monitoring (attention, state)
    """
    
    def __init__(
        self,
        crash_database=None,
        update_frequency_hz: float = 10.0
    ):
        """
        Initialize perception engine.
        
        Args:
            crash_database: Historical crash data for location risk
            update_frequency_hz: How often to update context (Hz)
        """
        self.crash_db = crash_database
        self.update_frequency = update_frequency_hz
        self.last_update = None
        
        # Cache for location-based patterns
        self.location_cache = {}
        
        logger.info(f"PerceptionEngine initialized at {update_frequency_hz} Hz")
    
    def build_context(
        self,
        sensor_data: Dict,
        location_data: Dict,
        driver_data: Dict
    ) -> DrivingContext:
        """
        Build complete driving context from sensor inputs.
        
        Args:
            sensor_data: Vehicle and environmental sensors
            location_data: GPS, map, zone information
            driver_data: Driver monitoring system data
            
        Returns:
            DrivingContext object with all relevant information
        """
        # Extract vehicle state
        speed = sensor_data.get('speed_mph', 0.0)
        acceleration = sensor_data.get('acceleration', 0.0)
        
        # Extract road conditions
        road_surface = sensor_data.get('road_surface', 'dry')
        road_quality = sensor_data.get('road_quality', 100.0)
        construction = location_data.get('construction_zone', False)
        
        # Extract environmental conditions
        weather = sensor_data.get('weather', 'clear')
        visibility = sensor_data.get('visibility_meters', 1000.0)
        lighting = self._determine_lighting(sensor_data)
        
        # Extract VRU data
        vru_data = self._process_vru_detections(sensor_data.get('vru_detections', []))
        
        # Get location context
        location_type = location_data.get('type', 'urban')
        special_zone = self._identify_special_zone(location_data)
        historical_crashes = self._get_historical_crashes(location_data)
        
        # Extract driver state
        attention = driver_data.get('attention_level', 100.0)
        overrides = driver_data.get('recent_overrides', 0)
        
        context = DrivingContext(
            speed_mph=speed,
            acceleration=acceleration,
            road_surface=road_surface,
            road_quality=road_quality,
            construction_zone=construction,
            weather=weather,
            visibility_meters=visibility,
            lighting=lighting,
            pedestrians_detected=vru_data['pedestrians'],
            cyclists_detected=vru_data['cyclists'],
            vru_distances=vru_data['distances'],
            vru_trajectories=vru_data['trajectories'],
            location_type=location_type,
            special_zone=special_zone,
            historical_crashes=historical_crashes,
            attention_level=attention,
            recent_overrides=overrides,
            timestamp=datetime.now()
        )
        
        self.last_update = datetime.now()
        return context
    
    def _determine_lighting(self, sensor_data: Dict) -> str:
        """
        Determine lighting condition from sensor data.
        
        Uses ambient light sensor, time of day, and sun position.
        """
        # Check if sensor provides lighting directly
        if 'lighting' in sensor_data:
            return sensor_data['lighting']
        
        # Otherwise infer from ambient light level
        light_level = sensor_data.get('ambient_light_lux', 50000)
        
        if light_level > 10000:
            return 'daylight'
        elif light_level > 1000:
            return 'dusk'
        elif light_level > 100:
            return 'dawn'
        else:
            return 'dark'
    
    def _process_vru_detections(self, detections: List[Dict]) -> Dict:
        """
        Process VRU detections from object detection system.
        
        Args:
            detections: List of detected VRUs with positions, types, velocities
            
        Returns:
            Processed VRU data
        """
        pedestrians = 0
        cyclists = 0
        distances = []
        trajectories = []
        
        for detection in detections:
            vru_type = detection.get('type', 'unknown')
            distance = detection.get('distance_m', float('inf'))
            velocity = detection.get('velocity_ms', 0.0)
            angle = detection.get('angle_deg', 0.0)
            
            if vru_type == 'pedestrian':
                pedestrians += 1
            elif vru_type == 'cyclist':
                cyclists += 1
            
            distances.append(distance)
            trajectories.append((velocity, angle))
        
        return {
            'pedestrians': pedestrians,
            'cyclists': cyclists,
            'distances': distances,
            'trajectories': trajectories
        }
    
    def _identify_special_zone(self, location_data: Dict) -> Optional[str]:
        """
        Identify if current location is a special zone.
        
        Uses map data to detect school zones, construction areas, etc.
        """
        # Check map tags
        tags = location_data.get('tags', [])
        
        if 'school' in tags or 'school_zone' in tags:
            return 'school'
        elif 'construction' in tags or 'work_zone' in tags:
            return 'construction'
        elif 'hospital' in tags or 'emergency' in tags:
            return 'hospital'
        
        return None
    
    def _get_historical_crashes(self, location_data: Dict) -> int:
        """
        Get number of historical crashes at current location.
        
        Queries crash database within radius of current position.
        """
        if not self.crash_db:
            return 0
        
        lat = location_data.get('latitude')
        lon = location_data.get('longitude')
        
        if lat is None or lon is None:
            return 0
        
        # Check cache first
        location_key = f"{lat:.4f},{lon:.4f}"
        if location_key in self.location_cache:
            return self.location_cache[location_key]
        
        # Query database (simplified - would use spatial query in production)
        # For now, return 0 as placeholder
        crash_count = 0
        
        # Cache result
        self.location_cache[location_key] = crash_count
        
        return crash_count
    
    def monitor_road_conditions(self, sensor_data: Dict) -> Dict:
        """
        Continuously monitor road surface conditions.
        
        Returns:
            Road condition assessment
        """
        surface_type = sensor_data.get('road_surface', 'dry')
        temperature = sensor_data.get('road_temp_c', 20)
        moisture = sensor_data.get('moisture_level', 0)
        
        # Assess road quality
        quality = 100.0
        
        # Temperature-based adjustments
        if temperature < 0 and moisture > 0:
            surface_type = 'icy'
            quality -= 40
        elif temperature < 5 and surface_type == 'wet':
            quality -= 20  # Risk of black ice
        
        # Moisture adjustments
        if moisture > 0.5:
            surface_type = 'wet'
            quality -= 15
        
        return {
            'surface_type': surface_type,
            'quality': max(quality, 0),
            'temperature': temperature,
            'moisture': moisture
        }
    
    def track_vru_trajectories(
        self,
        current_detections: List[Dict],
        previous_detections: List[Dict],
        dt: float
    ) -> List[Dict]:
        """
        Track VRU movement over time to predict trajectories.
        
        Args:
            current_detections: Current frame detections
            previous_detections: Previous frame detections
            dt: Time difference between frames (seconds)
            
        Returns:
            List of VRUs with predicted trajectories
        """
        tracked_vrus = []
        
        # Simple nearest-neighbor tracking (would use Kalman filter in production)
        for curr in current_detections:
            curr_pos = np.array([curr['x'], curr['y']])
            
            # Find nearest previous detection
            min_dist = float('inf')
            matched_prev = None
            
            for prev in previous_detections:
                prev_pos = np.array([prev['x'], prev['y']])
                dist = np.linalg.norm(curr_pos - prev_pos)
                
                if dist < min_dist:
                    min_dist = dist
                    matched_prev = prev
            
            # Compute velocity
            if matched_prev and min_dist < 5:  # Only match if < 5m movement
                prev_pos = np.array([matched_prev['x'], matched_prev['y']])
                velocity = (curr_pos - prev_pos) / dt
                
                tracked_vrus.append({
                    'position': curr_pos,
                    'velocity': velocity,
                    'type': curr['type'],
                    'distance': curr['distance_m']
                })
        
        return tracked_vrus
    
    def assess_driver_state(self, driver_data: Dict) -> Dict:
        """
        Assess driver attention and readiness.
        
        Args:
            driver_data: Data from driver monitoring system
            
        Returns:
            Driver state assessment
        """
        # Head pose
        head_forward = driver_data.get('head_forward', True)
        
        # Eye tracking
        eyes_on_road = driver_data.get('eyes_on_road', True)
        blink_rate = driver_data.get('blink_rate', 15)  # blinks per minute
        
        # Steering behavior
        steering_variance = driver_data.get('steering_variance', 0.1)
        
        # Compute attention level
        attention = 100.0
        
        if not head_forward:
            attention -= 30
        if not eyes_on_road:
            attention -= 40
        if blink_rate < 5:  # Too few blinks = fatigue
            attention -= 20
        if blink_rate > 30:  # Too many = distraction
            attention -= 15
        if steering_variance > 0.3:  # Erratic steering
            attention -= 20
        
        return {
            'attention_level': max(attention, 0),
            'eyes_on_road': eyes_on_road,
            'head_forward': head_forward,
            'fatigue_indicator': blink_rate < 5
        }
