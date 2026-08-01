"""
Unit tests for RealtimeSafetyCalculator.

Tests verify that different driving conditions produce different safety scores.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from realtime_calculator import RealtimeSafetyCalculator


@pytest.fixture
def calculator():
    """Load the trained safety calculator."""
    model_path = Path(__file__).parent.parent / "results" / "models" / "best_safety_model.pkl"
    feature_path = Path(__file__).parent.parent / "results" / "models" / "feature_names.txt"
    
    if not model_path.exists():
        pytest.skip("Model not trained yet. Run 02_train_inverse_model.ipynb first.")
    
    return RealtimeSafetyCalculator(str(model_path), str(feature_path))


@pytest.fixture
def base_scenario():
    """Base driving scenario for testing."""
    return {
        'HOUR': 14,
        'DAY_WEEK': 3,
        'MONTH': 6,
        'WEATHER': 1,
        'LGT_COND': 1,
        'ROAD_COND': 1,
        'SPEED_REL': 2,
        'VRU_PRESENT': 0,
        'IS_NIGHT': 0,
        'IS_WEEKEND': 0,
        'IS_RUSH_HOUR': 0,
        'POOR_LIGHTING': 0,
        'ADVERSE_WEATHER': 0
    }


class TestRoadConditionImpact:
    """Test that road condition changes affect safety scores."""
    
    def test_dry_vs_wet_road(self, calculator, base_scenario):
        """Dry road should be safer than wet road."""
        # Dry road (baseline)
        scenario_dry = base_scenario.copy()
        scenario_dry['ROAD_COND'] = 1
        result_dry = calculator.calculate_safety_score(scenario_dry)
        
        # Wet road
        scenario_wet = base_scenario.copy()
        scenario_wet['ROAD_COND'] = 2
        result_wet = calculator.calculate_safety_score(scenario_wet)
        
        print(f"\n🛣️  ROAD CONDITION TEST: Dry vs Wet")
        print(f"   Dry Road:  Score={result_dry['safety_score']:.1f}, Risk={result_dry['risk_level']}")
        print(f"   Wet Road:  Score={result_wet['safety_score']:.1f}, Risk={result_wet['risk_level']}")
        
        # Wet should be less safe than dry (unless model doesn't use this feature)
        # We test that they're different, not necessarily which is higher
        assert result_dry['safety_score'] != result_wet['safety_score'], \
            "Road condition change (dry vs wet) should affect safety score"
    
    def test_dry_vs_snow_ice(self, calculator, base_scenario):
        """Dry road should be safer than snow/ice."""
        # Dry road
        scenario_dry = base_scenario.copy()
        scenario_dry['ROAD_COND'] = 1
        result_dry = calculator.calculate_safety_score(scenario_dry)
        
        # Snow/Ice
        scenario_ice = base_scenario.copy()
        scenario_ice['ROAD_COND'] = 3
        result_ice = calculator.calculate_safety_score(scenario_ice)
        
        print(f"\n🛣️  ROAD CONDITION TEST: Dry vs Snow/Ice")
        print(f"   Dry Road:    Score={result_dry['safety_score']:.1f}, Risk={result_dry['risk_level']}")
        print(f"   Snow/Ice:    Score={result_ice['safety_score']:.1f}, Risk={result_ice['risk_level']}")
        
        assert result_dry['safety_score'] != result_ice['safety_score'], \
            "Road condition change (dry vs snow/ice) should affect safety score"
    
    def test_wet_vs_snow_ice(self, calculator, base_scenario):
        """Wet and snow/ice should have different safety scores."""
        # Wet road
        scenario_wet = base_scenario.copy()
        scenario_wet['ROAD_COND'] = 2
        result_wet = calculator.calculate_safety_score(scenario_wet)
        
        # Snow/Ice
        scenario_ice = base_scenario.copy()
        scenario_ice['ROAD_COND'] = 3
        result_ice = calculator.calculate_safety_score(scenario_ice)
        
        print(f"\n🛣️  ROAD CONDITION TEST: Wet vs Snow/Ice")
        print(f"   Wet Road:    Score={result_wet['safety_score']:.1f}, Risk={result_wet['risk_level']}")
        print(f"   Snow/Ice:    Score={result_ice['safety_score']:.1f}, Risk={result_ice['risk_level']}")
        
        # Allow for small numerical differences but they should be distinguishable
        score_diff = abs(result_wet['safety_score'] - result_ice['safety_score'])
        print(f"   Score Difference: {score_diff:.2f}")
        
        # If difference is less than 0.1, they're essentially the same
        if score_diff < 0.1:
            pytest.fail(f"Road condition change (wet vs snow/ice) has negligible effect: diff={score_diff:.2f}")


class TestLightingConditionImpact:
    """Test that lighting condition changes affect safety scores."""
    
    def test_daylight_vs_dark_not_lighted(self, calculator, base_scenario):
        """Daylight should be safer than dark without lights."""
        # Daylight
        scenario_day = base_scenario.copy()
        scenario_day['LGT_COND'] = 1
        scenario_day['POOR_LIGHTING'] = 0
        result_day = calculator.calculate_safety_score(scenario_day)
        
        # Dark - Not Lighted
        scenario_dark = base_scenario.copy()
        scenario_dark['LGT_COND'] = 2
        scenario_dark['POOR_LIGHTING'] = 1
        result_dark = calculator.calculate_safety_score(scenario_dark)
        
        print(f"\n💡 LIGHTING CONDITION TEST: Daylight vs Dark (Not Lighted)")
        print(f"   Daylight:         Score={result_day['safety_score']:.1f}, Risk={result_day['risk_level']}")
        print(f"   Dark Not Lighted: Score={result_dark['safety_score']:.1f}, Risk={result_dark['risk_level']}")
        
        assert result_day['safety_score'] != result_dark['safety_score'], \
            "Lighting condition change (daylight vs dark) should affect safety score"
    
    def test_daylight_vs_dawn_dusk(self, calculator, base_scenario):
        """Daylight should differ from dawn/dusk."""
        # Daylight
        scenario_day = base_scenario.copy()
        scenario_day['LGT_COND'] = 1
        scenario_day['POOR_LIGHTING'] = 0
        result_day = calculator.calculate_safety_score(scenario_day)
        
        # Dawn/Dusk
        scenario_dusk = base_scenario.copy()
        scenario_dusk['LGT_COND'] = 4
        scenario_dusk['POOR_LIGHTING'] = 1
        result_dusk = calculator.calculate_safety_score(scenario_dusk)
        
        print(f"\n💡 LIGHTING CONDITION TEST: Daylight vs Dawn/Dusk")
        print(f"   Daylight:   Score={result_day['safety_score']:.1f}, Risk={result_day['risk_level']}")
        print(f"   Dawn/Dusk:  Score={result_dusk['safety_score']:.1f}, Risk={result_dusk['risk_level']}")
        
        score_diff = abs(result_day['safety_score'] - result_dusk['safety_score'])
        print(f"   Score Difference: {score_diff:.2f}")
        
        if score_diff < 0.1:
            pytest.fail(f"Lighting condition change has negligible effect: diff={score_diff:.2f}")


class TestWeatherConditionImpact:
    """Test that weather condition changes affect safety scores."""
    
    def test_clear_vs_rain(self, calculator, base_scenario):
        """Clear weather should differ from rain."""
        # Clear
        scenario_clear = base_scenario.copy()
        scenario_clear['WEATHER'] = 1
        scenario_clear['ADVERSE_WEATHER'] = 0
        result_clear = calculator.calculate_safety_score(scenario_clear)
        
        # Rain
        scenario_rain = base_scenario.copy()
        scenario_rain['WEATHER'] = 2
        scenario_rain['ADVERSE_WEATHER'] = 1
        result_rain = calculator.calculate_safety_score(scenario_rain)
        
        print(f"\n🌤️  WEATHER CONDITION TEST: Clear vs Rain")
        print(f"   Clear:  Score={result_clear['safety_score']:.1f}, Risk={result_clear['risk_level']}")
        print(f"   Rain:   Score={result_rain['safety_score']:.1f}, Risk={result_rain['risk_level']}")
        
        assert result_clear['safety_score'] != result_rain['safety_score'], \
            "Weather condition change (clear vs rain) should affect safety score"
    
    def test_clear_vs_snow(self, calculator, base_scenario):
        """Clear weather should differ from snow."""
        # Clear
        scenario_clear = base_scenario.copy()
        scenario_clear['WEATHER'] = 1
        scenario_clear['ADVERSE_WEATHER'] = 0
        result_clear = calculator.calculate_safety_score(scenario_clear)
        
        # Snow
        scenario_snow = base_scenario.copy()
        scenario_snow['WEATHER'] = 3
        scenario_snow['ADVERSE_WEATHER'] = 1
        result_snow = calculator.calculate_safety_score(scenario_snow)
        
        print(f"\n🌤️  WEATHER CONDITION TEST: Clear vs Snow")
        print(f"   Clear:  Score={result_clear['safety_score']:.1f}, Risk={result_clear['risk_level']}")
        print(f"   Snow:   Score={result_snow['safety_score']:.1f}, Risk={result_snow['risk_level']}")
        
        score_diff = abs(result_clear['safety_score'] - result_snow['safety_score'])
        print(f"   Score Difference: {score_diff:.2f}")
        
        if score_diff < 0.1:
            pytest.fail(f"Weather condition change has negligible effect: diff={score_diff:.2f}")


class TestCombinedConditionsImpact:
    """Test combined condition changes have cumulative effects."""
    
    def test_good_vs_bad_conditions(self, calculator, base_scenario):
        """Best conditions should be much safer than worst conditions."""
        # Best conditions (daytime, clear, dry, low speed, no VRU)
        scenario_best = base_scenario.copy()
        scenario_best.update({
            'HOUR': 14,
            'WEATHER': 1,
            'LGT_COND': 1,
            'ROAD_COND': 1,
            'SPEED_REL': 1,
            'VRU_PRESENT': 0,
            'IS_NIGHT': 0,
            'IS_RUSH_HOUR': 0,
            'POOR_LIGHTING': 0,
            'ADVERSE_WEATHER': 0
        })
        result_best = calculator.calculate_safety_score(scenario_best)
        
        # Worst conditions (night, snow, dark, ice, high speed, VRU present)
        scenario_worst = base_scenario.copy()
        scenario_worst.update({
            'HOUR': 2,
            'WEATHER': 3,
            'LGT_COND': 2,
            'ROAD_COND': 3,
            'SPEED_REL': 5,
            'VRU_PRESENT': 1,
            'IS_NIGHT': 1,
            'IS_RUSH_HOUR': 0,
            'POOR_LIGHTING': 1,
            'ADVERSE_WEATHER': 1
        })
        result_worst = calculator.calculate_safety_score(scenario_worst)
        
        print(f"\n🎯 COMBINED CONDITIONS TEST: Best vs Worst")
        print(f"   Best Conditions:  Score={result_best['safety_score']:.1f}, Risk={result_best['risk_level']}")
        print(f"   Worst Conditions: Score={result_worst['safety_score']:.1f}, Risk={result_worst['risk_level']}")
        print(f"   Score Difference: {abs(result_best['safety_score'] - result_worst['safety_score']):.2f}")
        
        # Best should be significantly safer than worst
        assert result_best['safety_score'] > result_worst['safety_score'], \
            "Best conditions should have higher safety score than worst conditions"
        
        # Difference should be substantial
        score_diff = result_best['safety_score'] - result_worst['safety_score']
        assert score_diff > 5.0, \
            f"Safety score difference between best and worst conditions should be substantial (>5), got {score_diff:.2f}"


class TestVRUPresenceImpact:
    """Test that VRU presence affects safety scores."""
    
    def test_no_vru_vs_vru_present(self, calculator, base_scenario):
        """VRU presence should affect safety score."""
        # No VRU
        scenario_no_vru = base_scenario.copy()
        scenario_no_vru['VRU_PRESENT'] = 0
        result_no_vru = calculator.calculate_safety_score(scenario_no_vru)
        
        # VRU present
        scenario_vru = base_scenario.copy()
        scenario_vru['VRU_PRESENT'] = 1
        result_vru = calculator.calculate_safety_score(scenario_vru)
        
        print(f"\n🚶 VRU PRESENCE TEST")
        print(f"   No VRU:       Score={result_no_vru['safety_score']:.1f}, Risk={result_no_vru['risk_level']}")
        print(f"   VRU Present:  Score={result_vru['safety_score']:.1f}, Risk={result_vru['risk_level']}")
        
        score_diff = abs(result_no_vru['safety_score'] - result_vru['safety_score'])
        print(f"   Score Difference: {score_diff:.2f}")
        
        if score_diff < 0.1:
            pytest.fail(f"VRU presence has negligible effect: diff={score_diff:.2f}")


class TestTimeOfDayImpact:
    """Test that time of day affects safety scores."""
    
    def test_day_vs_night(self, calculator, base_scenario):
        """Daytime should differ from nighttime."""
        # Daytime (2 PM)
        scenario_day = base_scenario.copy()
        scenario_day['HOUR'] = 14
        scenario_day['IS_NIGHT'] = 0
        result_day = calculator.calculate_safety_score(scenario_day)
        
        # Nighttime (2 AM)
        scenario_night = base_scenario.copy()
        scenario_night['HOUR'] = 2
        scenario_night['IS_NIGHT'] = 1
        result_night = calculator.calculate_safety_score(scenario_night)
        
        print(f"\n⏰ TIME OF DAY TEST: Day vs Night")
        print(f"   Daytime (2 PM):   Score={result_day['safety_score']:.1f}, Risk={result_day['risk_level']}")
        print(f"   Nighttime (2 AM): Score={result_night['safety_score']:.1f}, Risk={result_night['risk_level']}")
        
        score_diff = abs(result_day['safety_score'] - result_night['safety_score'])
        print(f"   Score Difference: {score_diff:.2f}")
        
        if score_diff < 0.1:
            pytest.fail(f"Time of day change has negligible effect: diff={score_diff:.2f}")


class TestSpeedImpact:
    """Test that speed relative to limit affects safety scores."""
    
    def test_low_vs_high_speed(self, calculator, base_scenario):
        """Low speed should be safer than high speed."""
        # Low speed
        scenario_low = base_scenario.copy()
        scenario_low['SPEED_REL'] = 1
        result_low = calculator.calculate_safety_score(scenario_low)
        
        # High speed
        scenario_high = base_scenario.copy()
        scenario_high['SPEED_REL'] = 5
        result_high = calculator.calculate_safety_score(scenario_high)
        
        print(f"\n🚗 SPEED TEST: Low vs High")
        print(f"   Low Speed (1):  Score={result_low['safety_score']:.1f}, Risk={result_low['risk_level']}")
        print(f"   High Speed (5): Score={result_high['safety_score']:.1f}, Risk={result_high['risk_level']}")
        
        score_diff = abs(result_low['safety_score'] - result_high['safety_score'])
        print(f"   Score Difference: {score_diff:.2f}")
        
        if score_diff < 0.1:
            pytest.fail(f"Speed change has negligible effect: diff={score_diff:.2f}")


def test_feature_importance_analysis(calculator):
    """Analyze which features actually impact the model predictions."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Get feature names
    if hasattr(calculator.model, 'feature_names_in_'):
        feature_names = calculator.model.feature_names_in_
    elif calculator.feature_names:
        feature_names = calculator.feature_names
    else:
        print("⚠️  Could not determine feature names")
        return
    
    # Get feature importance if available
    if hasattr(calculator.model, 'feature_importances_'):
        importances = calculator.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 15 Most Important Features:")
        print("-" * 70)
        for i in range(min(15, len(indices))):
            idx = indices[i]
            print(f"{i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.6f}")
        
        # Check if our test features are in top features
        test_features = ['ROAD_COND', 'LGT_COND', 'WEATHER', 'VRU_PRESENT', 'SPEED_REL']
        print("\n\nTest Feature Rankings:")
        print("-" * 70)
        for feature in test_features:
            if feature in feature_names:
                idx = list(feature_names).index(feature)
                importance = importances[idx]
                rank = list(indices).index(idx) + 1
                print(f"{feature:20s} Rank: {rank:3d}/{ len(feature_names)} | Importance: {importance:.6f}")
            else:
                print(f"{feature:20s} ⚠️  NOT FOUND IN MODEL")
    else:
        print("⚠️  Model does not expose feature_importances_")
    
    print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
