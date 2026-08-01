"""
Tests for preprocessing.py module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from preprocessing import CrashPreprocessor


class TestCrashPreprocessor:
    """Test suite for CrashPreprocessor class"""
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = CrashPreprocessor()
        assert preprocessor is not None
    
    def test_handle_missing_values_drop(self, sample_accident_data):
        """Test dropping rows with missing values"""
        # Add some missing values
        data = sample_accident_data.copy()
        data.loc[0, 'WEATHER'] = np.nan
        data.loc[1, 'LGT_COND'] = np.nan
        
        preprocessor = CrashPreprocessor()
        result = preprocessor.handle_missing_values(data, strategy='drop')
        
        # Should have fewer rows
        assert len(result) < len(data)
        # Should have no missing values
        assert result.isnull().sum().sum() == 0
    
    def test_handle_missing_values_fill(self, sample_accident_data):
        """Test filling missing values"""
        # Add some missing values
        data = sample_accident_data.copy()
        data.loc[0, 'WEATHER'] = np.nan
        data.loc[1, 'HOUR'] = np.nan
        
        preprocessor = CrashPreprocessor()
        result = preprocessor.handle_missing_values(data, strategy='fill', fill_value=0)
        
        # Should have same number of rows
        assert len(result) == len(data)
        # Missing values should be filled with 0
        assert result.loc[0, 'WEATHER'] == 0
        assert result.loc[1, 'HOUR'] == 0
    
    def test_handle_missing_values_median(self, sample_accident_data):
        """Test filling missing values with median"""
        # Add some missing values
        data = sample_accident_data.copy()
        data.loc[0, 'HOUR'] = np.nan
        
        original_median = data['HOUR'].median()
        
        preprocessor = CrashPreprocessor()
        result = preprocessor.handle_missing_values(data, strategy='median')
        
        # Missing value should be filled with median
        assert result.loc[0, 'HOUR'] == original_median
    
    def test_remove_duplicates(self, sample_accident_data):
        """Test duplicate removal"""
        # Add duplicates
        data = pd.concat([sample_accident_data, sample_accident_data.head(2)], ignore_index=True)
        
        preprocessor = CrashPreprocessor()
        result = preprocessor.remove_duplicates(data, subset=['CASENUM'])
        
        # Should have original length
        assert len(result) == len(sample_accident_data)
        # No duplicate case numbers
        assert result['CASENUM'].nunique() == len(result)
    
    def test_filter_outliers_iqr(self):
        """Test outlier filtering using IQR method"""
        # Create data with outliers
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        })
        
        preprocessor = CrashPreprocessor()
        result = preprocessor.filter_outliers(data, column='value', method='iqr')
        
        # Outlier should be removed
        assert len(result) < len(data)
        assert 100 not in result['value'].values
    
    def test_filter_outliers_zscore(self):
        """Test outlier filtering using Z-score method"""
        # Create data with outliers
        np.random.seed(42)
        data = pd.DataFrame({
            'value': np.concatenate([np.random.normal(0, 1, 100), [10, -10]])  # Extreme values
        })
        
        preprocessor = CrashPreprocessor()
        result = preprocessor.filter_outliers(data, column='value', method='zscore', threshold=3)
        
        # Extreme outliers should be removed
        assert len(result) < len(data)
        assert result['value'].max() < 10
        assert result['value'].min() > -10
    
    def test_encode_categorical(self):
        """Test categorical encoding"""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B']
        })
        
        preprocessor = CrashPreprocessor()
        result = preprocessor.encode_categorical(data, columns=['category'])
        
        # Should have encoded columns
        assert 'category_B' in result.columns or 'category' not in result.columns
        # Original column might be removed or transformed
        assert result.shape[1] >= data.shape[1]
    
    def test_normalize_features(self):
        """Test feature normalization"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        preprocessor = CrashPreprocessor()
        result = preprocessor.normalize_features(data, columns=['feature1', 'feature2'])
        
        # Normalized features should be between 0 and 1
        assert result['feature1'].min() >= 0
        assert result['feature1'].max() <= 1
        assert result['feature2'].min() >= 0
        assert result['feature2'].max() <= 1
    
    def test_create_balanced_sample(self):
        """Test creating balanced samples"""
        # Create imbalanced data
        data = pd.DataFrame({
            'feature': range(100),
            'target': [0]*90 + [1]*10  # 90% class 0, 10% class 1
        })
        
        preprocessor = CrashPreprocessor()
        result = preprocessor.create_balanced_sample(data, target_column='target')
        
        # Should have balanced classes
        class_counts = result['target'].value_counts()
        assert len(class_counts) == 2
        # Classes should be roughly equal (allowing for sampling variance)
        assert abs(class_counts[0] - class_counts[1]) / class_counts[0] < 0.5
    
    def test_preprocessing_pipeline(self, sample_accident_data):
        """Test complete preprocessing pipeline"""
        # Add some issues
        data = sample_accident_data.copy()
        data.loc[0, 'WEATHER'] = np.nan
        data = pd.concat([data, data.head(1)], ignore_index=True)  # Add duplicate
        
        preprocessor = CrashPreprocessor()
        
        # Apply preprocessing steps
        result = data.copy()
        result = preprocessor.handle_missing_values(result, strategy='drop')
        result = preprocessor.remove_duplicates(result, subset=['CASENUM'])
        
        # Should have cleaned data
        assert len(result) <= len(sample_accident_data)
        assert result.isnull().sum().sum() == 0
        assert result['CASENUM'].nunique() == len(result)
