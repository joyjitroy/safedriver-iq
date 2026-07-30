"""
Scenario Simulator Module

Simulates various driving scenarios and computes safety scores.
Useful for:
- Testing model behavior across conditions
- Identifying high-risk combinations
- Planning intervention strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import itertools
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioSimulator:
    """
    Simulates driving scenarios and analyzes safety scores.
    
    Can generate:
    - Factorial combinations of risk factors
    - Monte Carlo random scenarios
    - Time-series scenarios (trips)
    - Specific risk pattern scenarios
    """
    
    def __init__(self, calculator=None):
        """
        Initialize simulator.
        
        Args:
            calculator: RealtimeSafetyCalculator instance (optional)
        """
        self.calculator = calculator
        self.scenarios = []
        self.results = None
    
    def generate_factorial_scenarios(
        self,
        factors: Dict[str, List],
        sample_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate all combinations of factors (or random sample).
        
        Args:
            factors: Dictionary mapping feature names to possible values
                Example: {
                    'HOUR': [6, 12, 18, 22],
                    'WEATHER': [1, 2, 3],
                    'SPEED_REL': [1, 3, 5]
                }
            sample_size: If provided, randomly sample this many combinations
            
        Returns:
            List of scenario dictionaries
        """
        logger.info("Generating factorial scenarios...")
        
        # Get all combinations
        keys = list(factors.keys())
        values = [factors[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        # Sample if requested
        if sample_size and len(combinations) > sample_size:
            indices = np.random.choice(len(combinations), sample_size, replace=False)
            combinations = [combinations[i] for i in indices]
            logger.info(f"Sampled {sample_size} from {len(combinations)} total combinations")
        
        # Convert to dictionaries
        scenarios = []
        for combo in combinations:
            scenario = {k: v for k, v in zip(keys, combo)}
            scenarios.append(scenario)
        
        self.scenarios = scenarios
        logger.info(f"Generated {len(scenarios)} scenarios")
        
        return scenarios
    
    def generate_monte_carlo_scenarios(
        self,
        n_scenarios: int,
        feature_distributions: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Generate random scenarios using specified distributions.
        
        Args:
            n_scenarios: Number of scenarios to generate
            feature_distributions: Dictionary mapping features to distribution specs
                Example: {
                    'HOUR': {'type': 'uniform', 'low': 0, 'high': 23},
                    'WEATHER': {'type': 'choice', 'values': [1, 2, 3], 'p': [0.7, 0.2, 0.1]},
                    'SPEED_REL': {'type': 'normal', 'mean': 3, 'std': 1}
                }
            
        Returns:
            List of scenario dictionaries
        """
        logger.info(f"Generating {n_scenarios} Monte Carlo scenarios...")
        
        scenarios = []
        
        for i in range(n_scenarios):
            scenario = {}
            
            for feature, dist_spec in feature_distributions.items():
                dist_type = dist_spec['type']
                
                if dist_type == 'uniform':
                    value = np.random.uniform(dist_spec['low'], dist_spec['high'])
                    if 'int' in dist_spec and dist_spec['int']:
                        value = int(value)
                
                elif dist_type == 'choice':
                    value = np.random.choice(
                        dist_spec['values'],
                        p=dist_spec.get('p', None)
                    )
                
                elif dist_type == 'normal':
                    value = np.random.normal(dist_spec['mean'], dist_spec['std'])
                    if 'clip' in dist_spec:
                        value = np.clip(value, *dist_spec['clip'])
                    if 'int' in dist_spec and dist_spec['int']:
                        value = int(value)
                
                elif dist_type == 'constant':
                    value = dist_spec['value']
                
                else:
                    raise ValueError(f"Unknown distribution type: {dist_type}")
                
                scenario[feature] = value
            
            scenarios.append(scenario)
        
        self.scenarios = scenarios
        logger.info(f"Generated {len(scenarios)} scenarios")
        
        return scenarios
    
    def generate_trip_scenario(
        self,
        duration_minutes: int,
        time_step_minutes: int = 5,
        start_conditions: Optional[Dict] = None,
        dynamic_factors: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate time-series scenario (simulated trip).
        
        Args:
            duration_minutes: Total trip duration
            time_step_minutes: Time between scenario samples
            start_conditions: Initial driving conditions
            dynamic_factors: Factors that change over time with change rules
                Example: {
                    'HOUR': 'increment',  # Hour increases
                    'WEATHER': {'change_prob': 0.1, 'values': [1, 2, 3]}
                }
            
        Returns:
            List of scenarios (one per time step)
        """
        logger.info(f"Generating trip scenario: {duration_minutes} min, {time_step_minutes} min steps")
        
        n_steps = duration_minutes // time_step_minutes
        scenarios = []
        
        # Initialize conditions
        current = start_conditions.copy() if start_conditions else {
            'HOUR': 12,
            'DAY_WEEK': 3,
            'MONTH': 6,
            'WEATHER': 1,
            'LGT_COND': 1,
            'SPEED_REL': 2,
            'ROAD_COND': 1
        }
        
        for step in range(n_steps):
            # Record current state
            scenarios.append(current.copy())
            
            # Update dynamic factors
            if dynamic_factors:
                for factor, rule in dynamic_factors.items():
                    if rule == 'increment':
                        current[factor] = (current.get(factor, 0) + 1) % 24  # Wrap at 24 for hour
                    
                    elif isinstance(rule, dict):
                        if 'change_prob' in rule:
                            if np.random.random() < rule['change_prob']:
                                current[factor] = np.random.choice(rule['values'])
                    
                    elif callable(rule):
                        current[factor] = rule(current[factor], step)
            
            # Update derived features
            self._update_derived_features(current)
        
        self.scenarios = scenarios
        logger.info(f"Generated trip with {len(scenarios)} time steps")
        
        return scenarios
    
    def _update_derived_features(self, scenario: Dict):
        """Update derived features based on base features."""
        if 'HOUR' in scenario:
            scenario['IS_NIGHT'] = 1 if scenario['HOUR'] >= 20 or scenario['HOUR'] <= 6 else 0
            scenario['IS_RUSH_HOUR'] = 1 if scenario['HOUR'] in [7, 8, 9, 16, 17, 18] else 0
        
        if 'DAY_WEEK' in scenario:
            scenario['IS_WEEKEND'] = 1 if scenario['DAY_WEEK'] in [1, 7] else 0
        
        if 'LGT_COND' in scenario:
            scenario['POOR_LIGHTING'] = 1 if scenario['LGT_COND'] > 1 else 0
        
        if 'WEATHER' in scenario:
            scenario['ADVERSE_WEATHER'] = 1 if scenario['WEATHER'] > 1 else 0
    
    def generate_risk_pattern_scenarios(self, pattern: str, n_scenarios: int = 100) -> List[Dict]:
        """
        Generate scenarios matching specific risk patterns.
        
        Args:
            pattern: Risk pattern type
                - 'high_risk': Multiple high-risk factors
                - 'low_risk': Multiple protective factors
                - 'night_crash': Night + other risk factors
                - 'weather_crash': Bad weather patterns
                - 'speed_crash': Speed-related patterns
                - 'vru_encounter': VRU present scenarios
            n_scenarios: Number of scenarios to generate
            
        Returns:
            List of scenarios matching pattern
        """
        logger.info(f"Generating {n_scenarios} '{pattern}' scenarios...")
        
        scenarios = []
        
        for i in range(n_scenarios):
            if pattern == 'high_risk':
                scenario = {
                    'HOUR': np.random.choice([20, 21, 22, 23, 0, 1, 2]),
                    'DAY_WEEK': np.random.choice([1, 7]),  # Weekend
                    'WEATHER': np.random.choice([2, 3]),  # Rain/snow
                    'LGT_COND': np.random.choice([2, 3]),  # Dark
                    'SPEED_REL': np.random.choice([4, 5]),  # High speed
                    'ROAD_COND': np.random.choice([2, 3]),  # Wet/ice
                    'VRU_PRESENT': 1
                }
            
            elif pattern == 'low_risk':
                scenario = {
                    'HOUR': np.random.choice([10, 11, 12, 13, 14, 15]),
                    'DAY_WEEK': np.random.choice([2, 3, 4, 5]),  # Weekday
                    'WEATHER': 1,  # Clear
                    'LGT_COND': 1,  # Daylight
                    'SPEED_REL': np.random.choice([1, 2]),  # Low speed
                    'ROAD_COND': 1,  # Dry
                    'VRU_PRESENT': 0
                }
            
            elif pattern == 'night_crash':
                scenario = {
                    'HOUR': np.random.choice([20, 21, 22, 23, 0, 1, 2, 3]),
                    'DAY_WEEK': np.random.randint(1, 8),
                    'WEATHER': np.random.choice([1, 1, 1, 2]),  # Mostly clear
                    'LGT_COND': np.random.choice([2, 3]),  # Dark
                    'SPEED_REL': np.random.choice([3, 4, 5]),
                    'ROAD_COND': np.random.choice([1, 2]),
                    'VRU_PRESENT': np.random.choice([0, 1])
                }
            
            elif pattern == 'weather_crash':
                scenario = {
                    'HOUR': np.random.randint(6, 20),
                    'DAY_WEEK': np.random.randint(1, 8),
                    'WEATHER': np.random.choice([2, 3, 4]),  # Rain/snow/fog
                    'LGT_COND': np.random.choice([1, 2, 4]),
                    'SPEED_REL': np.random.choice([3, 4]),
                    'ROAD_COND': np.random.choice([2, 3]),  # Wet/ice
                    'VRU_PRESENT': np.random.choice([0, 1])
                }
            
            elif pattern == 'speed_crash':
                scenario = {
                    'HOUR': np.random.randint(0, 24),
                    'DAY_WEEK': np.random.randint(1, 8),
                    'WEATHER': np.random.choice([1, 2]),
                    'LGT_COND': np.random.choice([1, 2, 3]),
                    'SPEED_REL': 5,  # Very high speed
                    'ROAD_COND': np.random.choice([1, 2]),
                    'VRU_PRESENT': np.random.choice([0, 1])
                }
            
            elif pattern == 'vru_encounter':
                scenario = {
                    'HOUR': np.random.choice([7, 8, 9, 16, 17, 18]),  # Commute times
                    'DAY_WEEK': np.random.choice([2, 3, 4, 5, 6]),
                    'WEATHER': 1,
                    'LGT_COND': np.random.choice([1, 2, 4]),
                    'SPEED_REL': np.random.choice([2, 3, 4]),
                    'ROAD_COND': 1,
                    'VRU_PRESENT': 1  # Always present
                }

            # ----------------------------------------------------------
            # New patterns — contextual risk factors
            # ----------------------------------------------------------

            elif pattern == 'rush_hour_aggressive':
                # Peak commute + high nearby lane-change + tailgating frequency
                scenario = {
                    'HOUR': np.random.choice([7, 8, 9, 16, 17, 18]),
                    'DAY_WEEK': np.random.choice([2, 3, 4, 5, 6]),
                    'WEATHER': np.random.choice([1, 1, 2]),
                    'LGT_COND': np.random.choice([1, 2]),
                    'SPEED_REL': np.random.choice([3, 4, 5]),
                    'ROAD_COND': np.random.choice([1, 2]),
                    'VRU_PRESENT': np.random.choice([0, 1], p=[0.80, 0.20]),
                    # Contextual overrides
                    'TRAFFIC_DENSITY_INDEX': np.random.uniform(3.5, 5.0),
                    'LANE_CHANGE_FREQ_PER_MILE': np.random.uniform(4, 10),
                    'TAILGATING_DETECTED_NEARBY': 1,
                    'NEARBY_AGGRESSIVE_DRIVER_COUNT': np.random.randint(2, 7),
                    'SPEED_VARIANCE_NEARBY_MPH': np.random.uniform(8, 20),
                }

            elif pattern == 'work_zone':
                # Active construction — lane reduction, workers present
                scenario = {
                    'HOUR': np.random.choice(list(range(6, 19))),
                    'DAY_WEEK': np.random.choice([2, 3, 4, 5, 6]),
                    'WEATHER': np.random.choice([1, 1, 2]),
                    'LGT_COND': np.random.choice([1, 2]),
                    'SPEED_REL': np.random.choice([3, 4]),
                    'ROAD_COND': np.random.choice([1, 2]),
                    'VRU_PRESENT': 0,
                    # Contextual overrides
                    'WORK_ZONE_PRESENT': 1,
                    'WORK_ZONE_WORKERS_PRESENT': np.random.choice([0, 1], p=[0.35, 0.65]),
                    'WORK_ZONE_LANE_REDUCTION': np.random.choice([1, 2]),
                    'WORK_ZONE_LENGTH_MILES': np.random.uniform(0.3, 3.0),
                    'LANE_WIDTH_FT': np.random.choice([9, 10, 11]),
                    'SIGNAGE_VISIBILITY_SCORE': np.random.uniform(2, 4),
                }

            elif pattern == 'school_zone_active':
                # School dismissal / arrival during school session hours
                scenario = {
                    'HOUR': np.random.choice([7, 8, 14, 15, 16]),
                    'DAY_WEEK': np.random.choice([2, 3, 4, 5, 6]),
                    'WEATHER': np.random.choice([1, 1, 2]),
                    'LGT_COND': 1,
                    'SPEED_REL': np.random.choice([1, 2, 3]),
                    'ROAD_COND': 1,
                    'VRU_PRESENT': np.random.choice([0, 1], p=[0.40, 0.60]),
                    # Contextual overrides
                    'SCHOOL_ZONE': 1,
                    'SCHOOL_HOURS_ACTIVE': 1,
                    'RESIDENTIAL_DENSITY_INDEX': np.random.uniform(5, 9),
                    'PEDESTRIAN_COUNT': np.random.randint(1, 5),
                }

            elif pattern == 'dui_risk_night':
                # Late-night weekend near bar-dense area — high DUI probability
                scenario = {
                    'HOUR': np.random.choice([21, 22, 23, 0, 1, 2]),
                    'DAY_WEEK': np.random.choice([1, 7, 6]),  # Fri/Sat/Sun
                    'WEATHER': np.random.choice([1, 1, 2]),
                    'LGT_COND': np.random.choice([2, 3]),  # Dark
                    'SPEED_REL': np.random.choice([3, 4, 5]),
                    'ROAD_COND': np.random.choice([1, 2]),
                    'VRU_PRESENT': np.random.choice([0, 1], p=[0.65, 0.35]),
                    # Contextual overrides
                    'DUI_RISK_INDEX': np.random.uniform(0.55, 0.95),
                    'BAR_DENSITY_CATEGORY': np.random.choice(['medium', 'high'], p=[0.40, 0.60]),
                    'DRIVER_IMPAIRED': 1,
                    'COMMERCIAL_DENSITY_INDEX': np.random.uniform(5, 9),
                }

            elif pattern == 'black_ice_winter':
                # Winter black-ice conditions: near-freezing temp + moisture + curve
                scenario = {
                    'HOUR': np.random.choice([5, 6, 7, 20, 21, 22, 23, 0, 1]),
                    'DAY_WEEK': np.random.randint(1, 8),
                    'MONTH': np.random.choice([1, 2, 12, 11]),
                    'WEATHER': np.random.choice([3, 4, 7]),  # Snow / freezing rain
                    'LGT_COND': np.random.choice([1, 2, 3]),
                    'SPEED_REL': np.random.choice([2, 3, 4]),
                    'ROAD_COND': np.random.choice([3, 4]),   # Ice / snow
                    'VRU_PRESENT': 0,
                    # Contextual overrides
                    'TEMPERATURE_F': np.random.uniform(20, 35),
                    'BLACK_ICE_RISK': np.random.uniform(0.55, 0.95),
                    'ROAD_SURFACE_TEMP_F': np.random.uniform(18, 33),
                    'PRECIPITATION_RATE_IN_HR': np.random.uniform(0.05, 0.3),
                    'HAS_HORIZONTAL_CURVE': np.random.choice([0, 1], p=[0.40, 0.60]),
                }

            elif pattern == 'narrow_road_curve':
                # Rural narrow lane on curve — high run-off-road risk  
                scenario = {
                    'HOUR': np.random.choice(list(range(0, 24))),
                    'DAY_WEEK': np.random.randint(1, 8),
                    'WEATHER': np.random.choice([1, 1, 2, 3]),
                    'LGT_COND': np.random.choice([1, 2, 3]),
                    'SPEED_REL': np.random.choice([3, 4, 5]),
                    'ROAD_COND': np.random.choice([1, 2, 3]),
                    'VRU_PRESENT': 0,
                    # Contextual overrides
                    'LANE_WIDTH_FT': np.random.choice([9, 10]),
                    'HAS_HORIZONTAL_CURVE': 1,
                    'CURVE_RADIUS_FT': np.random.randint(150, 600),
                    'ROAD_GRADE_PERCENT': np.random.uniform(4, 12),
                    'SIGHT_DISTANCE_FT': np.random.randint(80, 250),
                    'HAS_GUARDRAIL': 0,
                    'LANE_MARKINGS_VISIBLE': np.random.choice([0, 1], p=[0.45, 0.55]),
                    'IS_RURAL': 1,
                }

            elif pattern == 'construction_zone_night':
                # Night-time construction — poor lighting, workers risk
                scenario = {
                    'HOUR': np.random.choice([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]),
                    'DAY_WEEK': np.random.randint(1, 8),
                    'WEATHER': np.random.choice([1, 1, 2]),
                    'LGT_COND': np.random.choice([2, 3, 4]),  # Dark / unknown
                    'SPEED_REL': np.random.choice([3, 4]),
                    'ROAD_COND': np.random.choice([1, 2]),
                    'VRU_PRESENT': 0,
                    # Contextual overrides
                    'WORK_ZONE_PRESENT': 1,
                    'WORK_ZONE_WORKERS_PRESENT': np.random.choice([0, 1], p=[0.25, 0.75]),
                    'WORK_ZONE_LANE_REDUCTION': np.random.choice([1, 2, 3]),
                    'LANE_MARKINGS_VISIBLE': np.random.choice([0, 1], p=[0.60, 0.40]),
                    'SIGNAGE_VISIBILITY_SCORE': np.random.uniform(1.5, 3.0),
                    'FATIGUE_RISK_INDEX': np.random.uniform(0.40, 0.80),  # Night workers
                }

            elif pattern == 'distracted_rush':
                # Rush-hour distracted driving — phone use, high density
                scenario = {
                    'HOUR': np.random.choice([7, 8, 9, 16, 17, 18]),
                    'DAY_WEEK': np.random.choice([2, 3, 4, 5, 6]),
                    'WEATHER': 1,
                    'LGT_COND': 1,
                    'SPEED_REL': np.random.choice([2, 3]),
                    'ROAD_COND': 1,
                    'VRU_PRESENT': np.random.choice([0, 1], p=[0.70, 0.30]),
                    # Contextual overrides
                    'DISTRACTED_DRIVING_RISK': np.random.uniform(0.55, 0.90),
                    'DRIVER_DISTRACTED': 1,
                    'TRAFFIC_DENSITY_INDEX': np.random.uniform(3.0, 5.0),
                    'TAILGATING_DETECTED_NEARBY': 1,
                }

            elif pattern == 'poor_infrastructure':
                # Degraded road quality — faded markings, no guardrails, poor signs
                scenario = {
                    'HOUR': np.random.choice(list(range(0, 24))),
                    'DAY_WEEK': np.random.randint(1, 8),
                    'WEATHER': np.random.choice([1, 2, 3]),
                    'LGT_COND': np.random.choice([1, 2, 3]),
                    'SPEED_REL': np.random.choice([3, 4]),
                    'ROAD_COND': np.random.choice([2, 3]),
                    'VRU_PRESENT': np.random.choice([0, 1]),
                    # Contextual overrides
                    'ROAD_QUALITY_INDEX': np.random.uniform(1, 2),
                    'LANE_MARKINGS_VISIBLE': 0,
                    'SIGNAGE_VISIBILITY_SCORE': np.random.uniform(1, 2.5),
                    'HAS_GUARDRAIL': 0,
                    'HAS_RUMBLE_STRIPS': 0,
                }
            
            else:
                raise ValueError(
                    f"Unknown pattern: {pattern!r}. "
                    "Valid patterns: high_risk, low_risk, night_crash, weather_crash, "
                    "speed_crash, vru_encounter, rush_hour_aggressive, work_zone, "
                    "school_zone_active, dui_risk_night, black_ice_winter, "
                    "narrow_road_curve, construction_zone_night, distracted_rush, "
                    "poor_infrastructure"
                )
            
            # Add common fields
            scenario['MONTH'] = np.random.randint(1, 13)
            
            # Update derived features
            self._update_derived_features(scenario)
            
            scenarios.append(scenario)
        
        self.scenarios = scenarios
        logger.info(f"Generated {len(scenarios)} scenarios")
        
        return scenarios
    
    def simulate(self, scenarios: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Run simulation on scenarios using calculator.
        
        Args:
            scenarios: List of scenarios (uses self.scenarios if None)
            
        Returns:
            DataFrame with scenarios and safety scores
        """
        if self.calculator is None:
            raise ValueError("Calculator not provided. Initialize simulator with calculator.")
        
        if scenarios is None:
            scenarios = self.scenarios
        
        if not scenarios:
            raise ValueError("No scenarios to simulate")
        
        logger.info(f"Simulating {len(scenarios)} scenarios...")
        
        results = self.calculator.batch_calculate(scenarios)
        self.results = results
        
        logger.info("Simulation complete")
        
        return results
    
    def analyze_results(self) -> Dict:
        """
        Analyze simulation results.
        
        Returns:
            Dictionary with analysis metrics
        """
        if self.results is None:
            raise ValueError("No results to analyze. Run simulate() first.")
        
        analysis = {
            'total_scenarios': len(self.results),
            'safety_score_stats': self.results['safety_score'].describe().to_dict(),
            'risk_level_distribution': self.results['risk_level'].value_counts().to_dict(),
            'low_safety_scenarios': (self.results['safety_score'] < 50).sum(),
            'high_safety_scenarios': (self.results['safety_score'] >= 85).sum(),
        }
        
        return analysis
    
    def export_results(self, filepath: str):
        """Export simulation results to CSV."""
        if self.results is None:
            raise ValueError("No results to export")
        
        self.results.to_csv(filepath, index=False)
        logger.info(f"Results exported to {filepath}")


def create_comprehensive_test_suite() -> Dict[str, List[Dict]]:
    """
    Create comprehensive test suite for model validation.
    
    Returns:
        Dictionary mapping test categories to scenario lists
    """
    simulator = ScenarioSimulator()
    
    test_suite = {}
    
    # 1. Temporal variations
    test_suite['temporal'] = simulator.generate_factorial_scenarios({
        'HOUR': list(range(0, 24, 3)),  # Every 3 hours
        'DAY_WEEK': [1, 3, 5, 7],  # Sun, Tue, Thu, Sat
        'WEATHER': [1],
        'LGT_COND': [1],
        'SPEED_REL': [2]
    })
    
    # 2. Environmental variations
    test_suite['environmental'] = simulator.generate_factorial_scenarios({
        'HOUR': [12],
        'WEATHER': [1, 2, 3, 4],  # All weather types
        'LGT_COND': [1, 2, 3],  # All lighting
        'ROAD_COND': [1, 2, 3],  # All road conditions
        'SPEED_REL': [2]
    })
    
    # 3. Risk patterns — original
    test_suite['high_risk'] = simulator.generate_risk_pattern_scenarios('high_risk', 50)
    test_suite['low_risk'] = simulator.generate_risk_pattern_scenarios('low_risk', 50)
    test_suite['night_crash'] = simulator.generate_risk_pattern_scenarios('night_crash', 50)
    test_suite['weather_crash'] = simulator.generate_risk_pattern_scenarios('weather_crash', 50)
    
    # 4. VRU scenarios
    test_suite['vru_encounter'] = simulator.generate_risk_pattern_scenarios('vru_encounter', 100)

    # 5. Extended contextual risk patterns
    test_suite['rush_hour_aggressive']   = simulator.generate_risk_pattern_scenarios('rush_hour_aggressive', 50)
    test_suite['work_zone']              = simulator.generate_risk_pattern_scenarios('work_zone', 50)
    test_suite['school_zone_active']     = simulator.generate_risk_pattern_scenarios('school_zone_active', 50)
    test_suite['dui_risk_night']         = simulator.generate_risk_pattern_scenarios('dui_risk_night', 50)
    test_suite['black_ice_winter']       = simulator.generate_risk_pattern_scenarios('black_ice_winter', 50)
    test_suite['narrow_road_curve']      = simulator.generate_risk_pattern_scenarios('narrow_road_curve', 50)
    test_suite['construction_zone_night']= simulator.generate_risk_pattern_scenarios('construction_zone_night', 50)
    test_suite['distracted_rush']        = simulator.generate_risk_pattern_scenarios('distracted_rush', 50)
    test_suite['poor_infrastructure']    = simulator.generate_risk_pattern_scenarios('poor_infrastructure', 50)
    
    return test_suite


if __name__ == "__main__":
    print("Scenario Simulator Module")
    print("=" * 60)
    
    # Example: Generate factorial scenarios
    simulator = ScenarioSimulator()
    
    scenarios = simulator.generate_factorial_scenarios({
        'HOUR': [8, 12, 18, 22],
        'WEATHER': [1, 2],
        'SPEED_REL': [2, 4]
    })
    
    print(f"\nGenerated {len(scenarios)} factorial scenarios")
    print("\nFirst 3 scenarios:")
    for i, s in enumerate(scenarios[:3], 1):
        print(f"{i}. {s}")
    
    # Example: Generate Monte Carlo scenarios
    mc_scenarios = simulator.generate_monte_carlo_scenarios(
        n_scenarios=100,
        feature_distributions={
            'HOUR': {'type': 'uniform', 'low': 0, 'high': 23, 'int': True},
            'WEATHER': {'type': 'choice', 'values': [1, 2, 3], 'p': [0.7, 0.2, 0.1]},
            'SPEED_REL': {'type': 'normal', 'mean': 3, 'std': 1, 'clip': (1, 5), 'int': True}
        }
    )
    
    print(f"\nGenerated {len(mc_scenarios)} Monte Carlo scenarios")
    
    # Example: Risk patterns
    high_risk = simulator.generate_risk_pattern_scenarios('high_risk', 10)
    print(f"\nGenerated {len(high_risk)} high-risk scenarios")
    print("\nFirst high-risk scenario:")
    print(high_risk[0])
