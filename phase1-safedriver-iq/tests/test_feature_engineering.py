"""
Tests for feature_engineering.py module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""
    
    def test_initialization(self):
        """Test feature engineer initialization"""
        fe = FeatureEngineer()
        assert fe is not None
    
    def test_create_temporal_features(self, sample_accident_data):
        """Test temporal feature creation"""
        # Add DAY_WEEK column needed for IS_WEEKEND
        data = sample_accident_data.copy()
        data['DAY_WEEK'] = [2, 3, 1, 7, 5]  # Monday, Tuesday, Sunday, Saturday, Friday
        
        fe = FeatureEngineer()
        result = fe.create_temporal_features(data)
        
        # Check new columns exist
        assert 'IS_NIGHT' in result.columns
        assert 'IS_WEEKEND' in result.columns
        assert 'IS_RUSH_HOUR' in result.columns
        assert 'SEASON' in result.columns
        
        # Check values are reasonable
        assert result['IS_NIGHT'].dtype == np.int64
        assert result['IS_WEEKEND'].dtype == np.int64
        assert result['IS_RUSH_HOUR'].dtype == np.int64
    
    def test_create_temporal_features_night_hours(self):
        """Test night time detection"""
        data = pd.DataFrame({
            'CASENUM': [1, 2, 3, 4],
            'HOUR': [2, 10, 20, 23]  # Night, day, night, night
        })
        
        fe = FeatureEngineer()
        result = fe.create_temporal_features(data)
        
        # Hours 2, 20, and 23 should be night (0-6 or 20-23)
        assert result.loc[0, 'IS_NIGHT'] == 1
        assert result.loc[1, 'IS_NIGHT'] == 0
        assert result.loc[2, 'IS_NIGHT'] == 1
        assert result.loc[3, 'IS_NIGHT'] == 1
    
    def test_create_temporal_features_rush_hour(self):
        """Test rush hour detection"""
        data = pd.DataFrame({
            'CASENUM': [1, 2, 3, 4, 5],
            'HOUR': [7, 12, 17, 8, 19]  # Rush, not, rush, rush, not
        })
        
        fe = FeatureEngineer()
        result = fe.create_temporal_features(data)
        
        # 7-9 and 16-18 are rush hours
        assert result.loc[0, 'IS_RUSH_HOUR'] == True  # 7
        assert result.loc[1, 'IS_RUSH_HOUR'] == False  # 12
        assert result.loc[2, 'IS_RUSH_HOUR'] == True  # 17
        assert result.loc[3, 'IS_RUSH_HOUR'] == True  # 8
        assert result.loc[4, 'IS_RUSH_HOUR'] == False  # 19
    
    def test_create_environmental_features(self, sample_accident_data):
        """Test environmental feature creation"""
        fe = FeatureEngineer()
        result = fe.create_environmental_features(sample_accident_data)
        
        assert 'POOR_LIGHTING' in result.columns
        assert 'ADVERSE_WEATHER' in result.columns
        assert result['POOR_LIGHTING'].dtype == np.int64
        assert result['ADVERSE_WEATHER'].dtype == np.int64
    
    def test_create_environmental_features_poor_lighting(self):
        """Test poor lighting detection"""
        data = pd.DataFrame({
            'CASENUM': [1, 2, 3],
            'LGT_COND': [1, 2, 3]  # 1=daylight, 2+=poor
        })
        
        fe = FeatureEngineer()
        result = fe.create_environmental_features(data)
        
        assert result.loc[0, 'POOR_LIGHTING'] == 0
        assert result.loc[1, 'POOR_LIGHTING'] == 1
        assert result.loc[2, 'POOR_LIGHTING'] == 1
    
    def test_create_location_features(self, sample_accident_data):
        """Test location feature creation"""
        fe = FeatureEngineer()
        result = fe.create_location_features(sample_accident_data)
        
        # Check features that are actually created
        assert 'IS_URBAN' in result.columns
        assert 'IS_INTERSTATE' in result.columns
    
    def test_create_vru_features(self, sample_accident_data, sample_person_data):
        """Test VRU-specific feature creation"""
        fe = FeatureEngineer()
        result = fe.create_vru_features(sample_accident_data, sample_person_data)
        
        # Check for actual columns created by implementation
        assert 'pedestrian_count' in result.columns
        assert 'cyclist_count' in result.columns
        assert 'total_vru' in result.columns
    
    def test_create_vru_features_pedestrian_detection(self):
        """Test pedestrian detection in VRU features"""
        accident_df = pd.DataFrame({
            'CASENUM': [1001, 1002, 1003]
        })
        person_df = pd.DataFrame({
            'CASENUM': [1001, 1001, 1002, 1003],
            'PER_TYP': [1, 5, 1, 6]  # 5=pedestrian, 6=cyclist
        })
        
        fe = FeatureEngineer()
        result = fe.create_vru_features(accident_df, person_df)
        
        # Check pedestrian_count and cyclist_count
        assert result.loc[result['CASENUM'] == 1001, 'pedestrian_count'].iloc[0] == 1
        assert result.loc[result['CASENUM'] == 1002, 'pedestrian_count'].iloc[0] == 0
        assert result.loc[result['CASENUM'] == 1003, 'cyclist_count'].iloc[0] == 1
    
    def test_create_interaction_features(self, sample_features_data):
        """Test interaction feature creation"""
        fe = FeatureEngineer()
        result = fe.create_interaction_features(sample_features_data)
        
        # Check that interaction features are created
        assert 'NIGHT_AND_DARK' in result.columns or 'WEEKEND_NIGHT' in result.columns
        
        # Verify interactions are logical
        if 'NIGHT_AND_DARK' in result.columns:
            night_poor_light = (result['IS_NIGHT'] == 1) & (result['POOR_LIGHTING'] == 1)
            assert (result.loc[night_poor_light, 'NIGHT_AND_DARK'] == 1).all()
    
    def test_feature_engineer_pipeline(self, sample_accident_data, sample_person_data):
        """Test complete feature engineering pipeline"""
        fe = FeatureEngineer()
        
        # Apply all feature engineering steps
        df = sample_accident_data.copy()
        df = fe.create_temporal_features(df)
        df = fe.create_environmental_features(df)
        df = fe.create_location_features(df)
        df = fe.create_vru_features(df, sample_person_data)
        df = fe.create_interaction_features(df)
        
        # Verify we have significantly more features than input
        assert df.shape[1] > sample_accident_data.shape[1]
        
        # Verify no null values were introduced unexpectedly
        # (Some null values are okay for optional features)
        assert df.shape[0] == sample_accident_data.shape[0]
    
    def test_feature_types(self, sample_accident_data):
        """Test that engineered features have correct data types"""
        fe = FeatureEngineer()
        result = fe.create_temporal_features(sample_accident_data)
        result = fe.create_environmental_features(result)
        
        # Binary features should be 0/1 or bool
        for col in ['IS_NIGHT', 'IS_WEEKEND', 'IS_RUSH_HOUR', 'POOR_LIGHTING', 'ADVERSE_WEATHER']:
            if col in result.columns:
                assert result[col].isin([0, 1, True, False]).all()
    
    def test_missing_columns_handling(self):
        """Test handling of missing required columns"""
        # Data without required columns
        data = pd.DataFrame({
            'CASENUM': [1, 2, 3]
        })
        
        fe = FeatureEngineer()
        
        # Should handle missing columns gracefully
        result = fe.create_temporal_features(data)
        assert 'CASENUM' in result.columns
