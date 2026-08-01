"""
Data Preprocessing Module

Handles data cleaning, missing value imputation, and basic transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrashPreprocessor:
    """
    Preprocesses CRSS crash data for analysis.
    
    Handles:
    - Missing value treatment
    - Data type conversions
    - Outlier detection
    - Data quality checks
    """
    
    def __init__(self):
        """Initialize preprocessor with default settings."""
        self.missing_threshold = 0.5  # Drop columns with >50% missing
        
    def check_data_quality(self, df: pd.DataFrame, name: str = "Dataset") -> Dict:
        """
        Perform basic data quality checks.
        
        Args:
            df: DataFrame to check
            name: Name of dataset for logging
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info(f"\n=== Data Quality Report: {name} ===")
        
        quality_metrics = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'duplicate_records': df.duplicated().sum(),
            'missing_summary': {},
        }
        
        # Missing values summary
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        high_missing = missing_pct[missing_pct > 30].sort_values(ascending=False)
        
        if len(high_missing) > 0:
            logger.warning(f"Columns with >30% missing values: {len(high_missing)}")
            for col, pct in high_missing.head(10).items():
                logger.warning(f"  {col}: {pct}%")
        
        quality_metrics['missing_summary'] = {
            'high_missing_cols': len(high_missing),
            'total_missing_values': missing.sum(),
        }
        
        logger.info(f"Total records: {quality_metrics['total_records']:,}")
        logger.info(f"Duplicate records: {quality_metrics['duplicate_records']:,}")
        logger.info(f"Total missing values: {quality_metrics['missing_summary']['total_missing_values']:,}")
        
        return quality_metrics
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "auto", fill_value: Optional[int] = None) -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Args:
            df: DataFrame to process
            strategy: Strategy for missing values ('auto', 'drop', 'fill', 'median')
            fill_value: Value to fill missing data with (when strategy='fill')
            
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        if strategy == "auto":
            # Drop columns with >50% missing
            missing_pct = df.isnull().sum() / len(df)
            cols_to_drop = missing_pct[missing_pct > self.missing_threshold].index.tolist()
            
            if cols_to_drop:
                logger.info(f"Dropping {len(cols_to_drop)} columns with >{self.missing_threshold*100}% missing")
                df = df.drop(columns=cols_to_drop)
            
            # For remaining columns, fill categorical with 'Unknown', numeric with median
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        elif strategy == "drop":
            initial_len = len(df)
            df = df.dropna()
            logger.info(f"Dropped {initial_len - len(df):,} records with missing values")
        
        elif strategy == "fill":
            if fill_value is not None:
                df = df.fillna(fill_value)
            else:
                logger.warning("fill_value not provided, using 0")
                df = df.fillna(0)
        
        elif strategy == "median":
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate records.
        
        Args:
            df: DataFrame to process
            subset: Columns to consider for duplicates (None = all columns)
            
        Returns:
            DataFrame without duplicates
        """
        df = df.copy()
        initial_len = len(df)
        df = df.drop_duplicates(subset=subset)
        logger.info(f"Removed {initial_len - len(df):,} duplicate records")
        return df
    
    def filter_outliers(self, df: pd.DataFrame, column: str, method: str = "iqr", threshold: float = 3) -> pd.DataFrame:
        """
        Filter outliers from a specific column.
        
        Args:
            df: DataFrame to process
            column: Column to check for outliers
            method: Method to use ('iqr' or 'zscore')
            threshold: Threshold for z-score method (default=3)
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        initial_len = len(df)
        
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        elif method == "zscore":
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column], nan_policy='omit'))
            df = df[z_scores < threshold]
        
        logger.info(f"Removed {initial_len - len(df):,} outliers from {column}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical columns using one-hot encoding.
        
        Args:
            df: DataFrame to process
            columns: List of columns to encode
            
        Returns:
            DataFrame with encoded columns
        """
        df = df.copy()
        df = pd.get_dummies(df, columns=columns, drop_first=True)
        logger.info(f"Encoded {len(columns)} categorical columns")
        return df
    
    def normalize_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Normalize features to 0-1 range.
        
        Args:
            df: DataFrame to process
            columns: List of columns to normalize
            
        Returns:
            DataFrame with normalized columns
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        logger.info(f"Normalized {len(columns)} columns")
        return df
    
    def create_balanced_sample(self, df: pd.DataFrame, target_column: str, method: str = "undersample") -> pd.DataFrame:
        """
        Create balanced sample for imbalanced classes.
        
        Args:
            df: DataFrame to process
            target_column: Target column name
            method: Balancing method ('undersample' or 'oversample')
            
        Returns:
            Balanced DataFrame
        """
        df = df.copy()
        
        # Get class counts
        class_counts = df[target_column].value_counts()
        min_class_size = class_counts.min()
        
        if method == "undersample":
            # Undersample to minority class size
            balanced_dfs = []
            for class_val in class_counts.index:
                class_df = df[df[target_column] == class_val]
                sampled = class_df.sample(n=min_class_size, random_state=42)
                balanced_dfs.append(sampled)
            
            df = pd.concat(balanced_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        logger.info(f"Created balanced sample with {len(df)} records")
        return df
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to uppercase.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with standardized column names
        """
        df.columns = df.columns.str.upper()
        return df
    
    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for better memory efficiency.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with optimized data types
        """
        df = df.copy()
        
        # Convert object columns with few unique values to category
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Downcast numeric types
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def filter_vru_crashes(self, person_df: pd.DataFrame, accident_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        """
        Filter for VRU (pedestrian, bicyclist) crashes.
        
        Args:
            person_df: PERSON.csv data
            accident_df: ACCIDENT.csv data
            
        Returns:
            Tuple of (filtered accidents, list of VRU case IDs)
        """
        # PER_TYP codes:
        # 5 = Pedestrian
        # 6 = Bicyclist
        # Check if column exists
        if 'PER_TYP' not in person_df.columns:
            logger.error("PER_TYP column not found in person data")
            return accident_df, []
        
        vru_persons = person_df[person_df['PER_TYP'].isin([5, 6])]
        vru_case_ids = vru_persons['CASENUM'].unique().tolist()
        
        logger.info(f"Identified {len(vru_case_ids):,} VRU crashes")
        
        # Filter accidents
        vru_accidents = accident_df[accident_df['CASENUM'].isin(vru_case_ids)]
        
        logger.info(f"VRU accidents: {len(vru_accidents):,}")
        
        return vru_accidents, vru_case_ids
    
    def preprocess_pipeline(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Complete preprocessing pipeline for all datasets.
        
        Args:
            datasets: Dictionary with 'accident', 'vehicle', 'person', 'pbtype' DataFrames
            
        Returns:
            Dictionary with preprocessed DataFrames
        """
        logger.info("\n=== Starting Preprocessing Pipeline ===")
        
        processed = {}
        
        for name, df in datasets.items():
            logger.info(f"\nProcessing {name}...")
            
            # Quality check
            self.check_data_quality(df, name)
            
            # Standardize columns
            df = self.standardize_column_names(df)
            
            # Convert data types
            df = self.convert_data_types(df)
            
            # Handle missing values
            df = self.handle_missing_values(df, strategy='auto')
            
            processed[name] = df
            
            logger.info(f"Processed {name}: {len(df):,} records, {len(df.columns)} columns")
        
        logger.info("\n=== Preprocessing Complete ===")
        
        return processed


if __name__ == "__main__":
    # Example usage
    from data_loader import CRSSDataLoader
    
    loader = CRSSDataLoader()
    datasets = loader.load_complete_dataset()
    
    preprocessor = CrashPreprocessor()
    processed_datasets = preprocessor.preprocess_pipeline(datasets)
    
    print("\nPreprocessing Complete!")
