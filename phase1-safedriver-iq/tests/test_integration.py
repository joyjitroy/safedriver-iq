"""
Integration tests for complete workflow
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_loader import CRSSDataLoader
from feature_engineering import FeatureEngineer
from preprocessing import CrashPreprocessor
from models import SafetyScoreModel


class TestIntegration:
    """Integration tests for complete workflow"""
    
    @patch.object(CRSSDataLoader, 'load_complete_dataset')
    def test_data_to_features_pipeline(self, mock_load_dataset, temp_data_dir,
                                      sample_accident_data, sample_person_data):
        """Test data loading to feature engineering pipeline"""
        # Mock data loading
        mock_load_dataset.return_value = {
            'accident': sample_accident_data,
            'person': sample_person_data
        }
        
        # Load data
        loader = CRSSDataLoader(data_dir=str(temp_data_dir), years=[2020])
        data = loader.load_complete_dataset()
        
        # Engineer features
        fe = FeatureEngineer()
        accident_df = data['accident'].copy()
        accident_df = fe.create_temporal_features(accident_df)
        accident_df = fe.create_environmental_features(accident_df)
        accident_df = fe.create_location_features(accident_df)
        accident_df = fe.create_vru_features(accident_df, data['person'])
        accident_df = fe.create_interaction_features(accident_df)
        
        # Verify pipeline worked
        assert accident_df.shape[1] > sample_accident_data.shape[1]
        assert 'IS_NIGHT' in accident_df.columns
        assert 'POOR_LIGHTING' in accident_df.columns
    
    def test_features_to_model_pipeline(self, sample_features_data):
        """Test feature engineering to model training pipeline"""
        # Prepare features with larger dataset
        df = sample_features_data.copy()
        
        # Create larger dataset (repeat rows to have enough for splitting)
        df = pd.concat([df] * 20, ignore_index=True)
        
        # Create target
        np.random.seed(42)
        df['TARGET'] = np.random.randint(0, 2, len(df))
        
        # Select features for modeling
        feature_cols = [col for col in df.columns 
                       if col not in ['CASENUM', 'TARGET', 'PSU', 'YEAR']]
        X = df[feature_cols].fillna(0)
        y = df['TARGET']
        
        # Train model
        model = SafetyScoreModel(model_type='random_forest', random_state=42)
        model.feature_names = feature_cols
        
        metrics = model.train(X, y, test_size=0.2, validate=False)
        
        # Verify training worked
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        assert metrics['train_accuracy'] > 0
    
    def test_model_prediction_pipeline(self, sample_model_data):
        """Test complete model training and prediction pipeline"""
        X, y = sample_model_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = SafetyScoreModel(model_type='random_forest', random_state=42)
        model.feature_names = X.columns.tolist()
        model.train(X_train, y_train, test_size=0.2, validate=False)
        
        # Make predictions
        safety_scores = model.predict_safety_score(X_test)
        
        # Verify predictions
        assert len(safety_scores) == len(X_test)
        assert all(0 <= score <= 100 for score in safety_scores)
        assert len(set(safety_scores)) > 1  # Should have variation
    
    def test_preprocessing_to_model_pipeline(self, sample_accident_data):
        """Test preprocessing to model training pipeline"""
        # Create larger sample with unique case numbers
        df = sample_accident_data.copy()
        
        # Create 50 unique records
        all_dfs = []
        for i in range(50):
            temp_df = df.copy()
            temp_df['CASENUM'] = temp_df['CASENUM'] + (i * 10000)  # Make unique
            all_dfs.append(temp_df)
        
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Add some issues
        df.loc[0, 'WEATHER'] = np.nan
        
        # Preprocess
        preprocessor = CrashPreprocessor()
        df = preprocessor.handle_missing_values(df, strategy='drop')
        df = preprocessor.remove_duplicates(df, subset=['CASENUM'])
        
        # Add features
        fe = FeatureEngineer()
        df = fe.create_temporal_features(df)
        df = fe.create_environmental_features(df)
        
        # Create balanced target with enough samples
        np.random.seed(42)
        # Ensure at least 10 samples of each class
        n_class_1 = len(df) // 2
        n_class_0 = len(df) - n_class_1
        target = [0] * n_class_0 + [1] * n_class_1
        np.random.shuffle(target)
        df['TARGET'] = target
        
        # Train model
        feature_cols = [col for col in df.columns 
                       if col not in ['CASENUM', 'TARGET', 'PSU', 'YEAR', 'SEASON']]  # Exclude string columns
        X = df[feature_cols].fillna(0)
        # Ensure all numeric
        X = X.select_dtypes(include=[np.number])
        y = df['TARGET']
        
        model = SafetyScoreModel(model_type='random_forest', random_state=42)
        model.feature_names = feature_cols
        metrics = model.train(X, y, test_size=0.2, validate=False)
        
        # Pipeline should complete successfully
        assert metrics['train_accuracy'] > 0
    
    @patch.object(CRSSDataLoader, 'load_complete_dataset')
    @patch.object(CRSSDataLoader, 'get_vru_crashes')
    def test_end_to_end_vru_crash_prediction(self, mock_get_vru, mock_load_dataset,
                                            temp_data_dir, sample_accident_data, sample_person_data):
        """Test end-to-end VRU crash prediction workflow"""
        # Mock data
        mock_load_dataset.return_value = {
            'accident': sample_accident_data,
            'person': sample_person_data
        }
        mock_get_vru.return_value = {1001, 1003}
        
        # Load VRU crashes
        loader = CRSSDataLoader(data_dir=str(temp_data_dir), years=[2020])
        vru_cases = loader.get_vru_crashes()
        data = loader.load_complete_dataset()
        
        # Filter to VRU crashes
        vru_crashes = data['accident'][data['accident']['CASENUM'].isin(vru_cases)].copy()
        
        # Engineer features
        fe = FeatureEngineer()
        vru_crashes = fe.create_temporal_features(vru_crashes)
        vru_crashes = fe.create_environmental_features(vru_crashes)
        vru_crashes = fe.create_vru_features(vru_crashes, data['person'])
        
        # Verify VRU-specific features exist
        assert 'pedestrian_count' in vru_crashes.columns or 'total_vru' in vru_crashes.columns
        assert len(vru_crashes) > 0
