"""
Machine Learning Models Module

Contains models for crash prediction and inverse safety modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import logging

# Import centralized GPU configuration
from gpu_config import get_gpu_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyScoreModel:
    """
    Machine learning model for inverse safety score calculation.
    
    This model learns crash patterns and computes the "distance"
    from crash-producing conditions as a safety metric.
    """
    
    def __init__(self, model_type: str = "xgboost", random_state: int = 42):
        """
        Initialize safety score model.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm', 'gradient_boost')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.gpu_config = get_gpu_config()  # Get GPU configuration
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying ML model."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "xgboost":
            # Check if GPU is available
            try:
                import subprocess
                gpu_check = subprocess.run(['nvidia-smi'], capture_output=True)
                gpu_available = gpu_check.returncode == 0
            except:
                gpu_available = False
            
            if gpu_available:
                logger.info("🚀 GPU detected - enabling GPU acceleration for XGBoost")
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    tree_method='hist',  # Use hist for XGBoost 3.x
                    device='cuda'  # Enable GPU (XGBoost 3.x syntax)
                )
            else:
                logger.info("CPU mode for XGBoost")
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1
                )
        elif self.model_type == "gradient_boost":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature columns (if None, use all numeric)
            
        Returns:
            Tuple of (X, y) for training
        """
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle any remaining NaNs
        X = X.fillna(0)
        
        self.feature_names = feature_cols
        
        logger.info(f"Prepared data: {len(X)} samples, {len(feature_cols)} features")
        
        return X, y
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = 0.2,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Train the safety model.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data for testing
            validate: Whether to perform validation
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("\n=== Training Safety Model ===")
        
        # Split data using model's random_state for variation across iterations
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Class distribution (train): {y_train.value_counts().to_dict()}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Evaluate
        metrics = {}
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        metrics['train_accuracy'] = train_score
        metrics['test_accuracy'] = test_score
        
        logger.info(f"Train accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Classification report
        logger.info("\nClassification Report:")
        report = classification_report(y_test, y_pred)
        print(report)
        metrics['classification_report'] = report
        
        # ROC AUC
        if len(np.unique(y)) == 2:  # Binary classification
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            metrics['roc_auc'] = roc_auc
        
        # Cross-validation
        if validate:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            metrics['cv_scores'] = cv_scores
            metrics['cv_mean'] = cv_scores.mean()
        
        logger.info("\n=== Training Complete ===")
        
        return metrics
    
    def predict_safety_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict safety score (0-100).
        
        Higher score = safer (further from crash conditions).
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of safety scores (0-100)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get probability of NO crash (assumed class 0)
        # Higher probability of no crash = higher safety
        proba_no_crash = self.model.predict_proba(X)[:, 0]
        
        # Convert to 0-100 scale
        safety_scores = proba_no_crash * 100
        
        return safety_scores
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet or does not support feature importance")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: str):
        """Save model to file."""
        import joblib
        joblib.dump(self, filepath)  # Save the SafetyScoreModel wrapper
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Note: This loads the entire SafetyScoreModel wrapper and copies
        its attributes to this instance.
        """
        import joblib
        loaded_model = joblib.load(filepath)
        
        # Copy all attributes from loaded model to self
        self.model = loaded_model.model
        self.model_type = loaded_model.model_type
        self.random_state = loaded_model.random_state
        self.feature_names = loaded_model.feature_names
        self.feature_importance = loaded_model.feature_importance
        
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    print("SafetyScoreModel module loaded")
