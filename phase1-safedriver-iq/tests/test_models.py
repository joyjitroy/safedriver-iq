"""
Tests for models.py module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models import SafetyScoreModel


class TestSafetyScoreModel:
    """Test suite for SafetyScoreModel class"""
    
    def test_initialization_random_forest(self):
        """Test initialization with random forest"""
        model = SafetyScoreModel(model_type='random_forest')
        assert model.model_type == 'random_forest'
        assert model.model is not None
        assert model.feature_names is None
    
    def test_initialization_xgboost(self):
        """Test initialization with XGBoost"""
        model = SafetyScoreModel(model_type='xgboost')
        assert model.model_type == 'xgboost'
        assert model.model is not None
    
    def test_initialization_gradient_boost(self):
        """Test initialization with gradient boosting"""
        model = SafetyScoreModel(model_type='gradient_boost')
        assert model.model_type == 'gradient_boost'
        assert model.model is not None
    
    def test_initialization_invalid_model_type(self):
        """Test initialization with invalid model type"""
        with pytest.raises(ValueError):
            SafetyScoreModel(model_type='invalid_model')
    
    def test_initialization_with_random_state(self):
        """Test initialization with custom random state"""
        model1 = SafetyScoreModel(model_type='random_forest', random_state=42)
        model2 = SafetyScoreModel(model_type='random_forest', random_state=42)
        
        assert model1.random_state == 42
        assert model2.random_state == 42
    
    def test_prepare_data(self, sample_model_data):
        """Test data preparation"""
        X, y = sample_model_data
        df = X.copy()
        df['target'] = y
        
        model = SafetyScoreModel(model_type='random_forest')
        X_prep, y_prep = model.prepare_data(df, 'target')
        
        assert isinstance(X_prep, pd.DataFrame)
        assert isinstance(y_prep, pd.Series)
        assert len(X_prep) == len(y_prep)
        assert 'target' not in X_prep.columns
    
    def test_train(self, sample_model_data):
        """Test model training"""
        X, y = sample_model_data
        
        model = SafetyScoreModel(model_type='random_forest', random_state=42)
        model.feature_names = X.columns.tolist()
        
        metrics = model.train(X, y, test_size=0.2, validate=True)
        
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 'cv_scores' in metrics
        assert 'cv_mean' in metrics
        
        # Accuracies should be between 0 and 1
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['test_accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_train_without_validation(self, sample_model_data):
        """Test training without cross-validation"""
        X, y = sample_model_data
        
        model = SafetyScoreModel(model_type='random_forest', random_state=42)
        model.feature_names = X.columns.tolist()
        
        metrics = model.train(X, y, test_size=0.2, validate=False)
        
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        assert 'cv_scores' not in metrics
    
    def test_feature_importance_extraction(self, sample_model_data):
        """Test feature importance extraction"""
        X, y = sample_model_data
        
        model = SafetyScoreModel(model_type='random_forest', random_state=42)
        model.feature_names = X.columns.tolist()
        model.train(X, y, test_size=0.2, validate=False)
        
        assert model.feature_importance is not None
        assert isinstance(model.feature_importance, pd.DataFrame)
        assert 'feature' in model.feature_importance.columns
        assert 'importance' in model.feature_importance.columns
        assert len(model.feature_importance) == len(X.columns)
    
    def test_get_feature_importance(self, sample_model_data):
        """Test getting top N feature importances"""
        X, y = sample_model_data
        
        model = SafetyScoreModel(model_type='random_forest', random_state=42)
        model.feature_names = X.columns.tolist()
        model.train(X, y, test_size=0.2, validate=False)
        
        top_5 = model.get_feature_importance(top_n=5)
        
        assert len(top_5) == 5
        assert top_5['importance'].is_monotonic_decreasing
    
    def test_predict_safety_score(self, sample_model_data):
        """Test safety score prediction"""
        X, y = sample_model_data
        
        model = SafetyScoreModel(model_type='random_forest', random_state=42)
        model.feature_names = X.columns.tolist()
        model.train(X, y, test_size=0.2, validate=False)
        
        # Predict on training data
        scores = model.predict_safety_score(X)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(X)
        assert all(0 <= score <= 100 for score in scores)
    
    def test_predict_safety_score_untrained_model(self, sample_model_data):
        """Test prediction before training raises error"""
        X, _ = sample_model_data
        
        model = SafetyScoreModel(model_type='random_forest')
        model.model = None  # Simulate untrained state
        
        with pytest.raises(ValueError):
            model.predict_safety_score(X)
    
    def test_model_reproducibility(self, sample_model_data):
        """Test that same random state produces same results"""
        X, y = sample_model_data
        
        model1 = SafetyScoreModel(model_type='random_forest', random_state=42)
        model1.feature_names = X.columns.tolist()
        metrics1 = model1.train(X, y, test_size=0.2, validate=False)
        
        model2 = SafetyScoreModel(model_type='random_forest', random_state=42)
        model2.feature_names = X.columns.tolist()
        metrics2 = model2.train(X, y, test_size=0.2, validate=False)
        
        # Results should be identical with same random state
        assert metrics1['train_accuracy'] == metrics2['train_accuracy']
    
    def test_save_and_load_model(self, sample_model_data, tmp_path):
        """Test model saving and loading"""
        X, y = sample_model_data
        
        model = SafetyScoreModel(model_type='random_forest', random_state=42)
        model.feature_names = X.columns.tolist()
        model.train(X, y, test_size=0.2, validate=False)
        
        # Save model
        save_path = tmp_path / "test_model.pkl"
        model.save_model(str(save_path))
        
        assert save_path.exists()
        
        # Load model
        new_model = SafetyScoreModel(model_type='random_forest')
        new_model.load_model(str(save_path))
        
        # Predictions should match
        scores1 = model.predict_safety_score(X.head(10))
        scores2 = new_model.predict_safety_score(X.head(10))
        
        np.testing.assert_array_almost_equal(scores1, scores2)
    
    def test_different_model_types_train(self, sample_model_data):
        """Test that all model types can train"""
        X, y = sample_model_data
        
        for model_type in ['random_forest', 'xgboost', 'gradient_boost']:
            model = SafetyScoreModel(model_type=model_type, random_state=42)
            model.feature_names = X.columns.tolist()
            metrics = model.train(X, y, test_size=0.2, validate=False)
            
            assert 'train_accuracy' in metrics
            assert 'test_accuracy' in metrics
            assert metrics['train_accuracy'] > 0
