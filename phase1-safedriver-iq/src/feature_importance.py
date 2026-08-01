"""
Advanced Feature Importance and Selection Module

Provides multiple methods for feature importance analysis and selection
for crash prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import (
    mutual_info_classif,
    SelectKBest,
    chi2,
    RFE
)
import xgboost as xgb
import lightgbm as lgb
import shap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis using multiple methods.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.results = {}
        self.models = {}
    
    def compute_tree_importance(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        model_type: str = 'random_forest'
    ) -> pd.DataFrame:
        """
        Compute feature importance using tree-based models.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: 'random_forest', 'gradient_boosting', 'xgboost', or 'lightgbm'
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Computing tree-based importance using {model_type}")
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                random_state=42, 
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X, y)
        self.models[model_type] = model
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_,
            'method': model_type
        }).sort_values('importance', ascending=False)
        
        self.results[model_type] = importance_df
        return importance_df
    
    def compute_permutation_importance(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model=None,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Compute model-agnostic permutation importance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model: Pre-trained model (if None, uses random forest)
            n_repeats: Number of random shuffles
            
        Returns:
            DataFrame with permutation importance scores
        """
        logger.info("Computing permutation importance")
        
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        perm_importance = permutation_importance(
            model, 
            X_test, 
            y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std,
            'method': 'permutation'
        }).sort_values('importance', ascending=False)
        
        self.results['permutation'] = importance_df
        return importance_df
    
    def compute_shap_importance(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        model=None,
        sample_size: int = 500
    ) -> pd.DataFrame:
        """
        Compute SHAP-based feature importance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            model: Pre-trained model (if None, uses random forest)
            sample_size: Number of samples for SHAP computation
            
        Returns:
            DataFrame with SHAP importance scores
        """
        logger.info("Computing SHAP importance")
        
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Sample for efficiency
        X_shap = X_test.sample(min(sample_size, len(X_test)), random_state=42)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        # For binary classification, use class 1 (crash)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(shap_values).mean(axis=0),
            'method': 'shap'
        }).sort_values('importance', ascending=False)
        
        self.results['shap'] = importance_df
        self.shap_values = shap_values
        self.X_shap = X_shap
        
        return importance_df
    
    def compute_mutual_information(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute mutual information scores.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            DataFrame with mutual information scores
        """
        logger.info("Computing mutual information")
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores,
            'method': 'mutual_info'
        }).sort_values('importance', ascending=False)
        
        self.results['mutual_info'] = importance_df
        return importance_df
    
    def compute_statistical_tests(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute statistical significance of features.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            DataFrame with statistical test results
        """
        logger.info("Computing statistical tests")
        
        from scipy.stats import chi2_contingency, ttest_ind
        
        results = []
        
        for col in X.columns:
            # For binary features, use chi-square test
            if X[col].nunique() <= 10:
                contingency_table = pd.crosstab(X[col], y)
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                test_stat = chi2
                test_type = 'chi2'
            else:
                # For continuous features, use t-test
                group0 = X[y == 0][col]
                group1 = X[y == 1][col]
                test_stat, p_value = ttest_ind(group0, group1, nan_policy='omit')
                test_type = 't-test'
            
            results.append({
                'feature': col,
                'test_statistic': abs(test_stat),
                'p_value': p_value,
                'test_type': test_type,
                'significant': p_value < 0.05
            })
        
        importance_df = pd.DataFrame(results).sort_values('test_statistic', ascending=False)
        self.results['statistical'] = importance_df
        
        return importance_df
    
    def compute_consensus_ranking(
        self,
        methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute consensus ranking across all methods.
        
        Args:
            methods: List of methods to include (None for all)
            
        Returns:
            DataFrame with consensus importance scores
        """
        logger.info("Computing consensus ranking")
        
        if methods is None:
            methods = list(self.results.keys())
        
        # Normalize all importance scores to 0-1
        normalized_results = {}
        for method in methods:
            if method not in self.results:
                logger.warning(f"Method {method} not found in results")
                continue
            
            df = self.results[method].copy()
            max_importance = df['importance'].max()
            if max_importance > 0:
                df['importance_normalized'] = df['importance'] / max_importance
            else:
                df['importance_normalized'] = df['importance']
            
            normalized_results[method] = df[['feature', 'importance_normalized']]
        
        # Merge all method results
        consensus = None
        for method, df in normalized_results.items():
            df = df.rename(columns={'importance_normalized': f'{method}_importance'})
            if consensus is None:
                consensus = df
            else:
                consensus = consensus.merge(df, on='feature', how='outer')
        
        # Fill missing values with 0
        consensus = consensus.fillna(0)
        
        # Compute average importance
        importance_cols = [col for col in consensus.columns if col.endswith('_importance')]
        consensus['average_importance'] = consensus[importance_cols].mean(axis=1)
        consensus = consensus.sort_values('average_importance', ascending=False)
        
        self.results['consensus'] = consensus
        return consensus
    
    def select_top_features(
        self,
        n_features: int = 10,
        method: str = 'consensus'
    ) -> List[str]:
        """
        Select top N most important features.
        
        Args:
            n_features: Number of features to select
            method: Method to use for selection
            
        Returns:
            List of top feature names
        """
        if method not in self.results:
            raise ValueError(f"Method {method} not found. Run compute methods first.")
        
        top_features = self.results[method].head(n_features)['feature'].tolist()
        logger.info(f"Selected top {n_features} features using {method}")
        
        return top_features
    
    def plot_importance_comparison(self, top_n: int = 15):
        """
        Plot feature importance comparison across methods.
        
        Args:
            top_n: Number of top features to display
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if 'consensus' not in self.results:
            logger.warning("Consensus ranking not computed. Run compute_consensus_ranking first.")
            return
        
        consensus = self.results['consensus'].head(top_n)
        
        # Get importance columns
        importance_cols = [col for col in consensus.columns if col.endswith('_importance')]
        
        if not importance_cols:
            logger.warning("No importance columns found")
            return
        
        # Prepare data for plotting
        plot_data = consensus[['feature'] + importance_cols].set_index('feature')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_data.plot(kind='barh', ax=ax)
        ax.set_xlabel('Normalized Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Feature Importance Comparison (Top {top_n})', fontsize=14, fontweight='bold')
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()


class FeatureSelector:
    """
    Advanced feature selection using multiple techniques.
    """
    
    def __init__(self):
        """Initialize the selector."""
        self.selected_features = {}
    
    def select_by_variance(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Select features with variance above threshold.
        
        Args:
            X: Feature matrix
            threshold: Minimum variance threshold
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected = X.columns[selector.get_support()].tolist()
        self.selected_features['variance'] = selected
        
        logger.info(f"Selected {len(selected)} features with variance > {threshold}")
        return selected
    
    def select_by_correlation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        threshold: float = 0.1
    ) -> List[str]:
        """
        Select features with correlation to target above threshold.
        
        Args:
            X: Feature matrix
            y: Target variable
            threshold: Minimum correlation threshold
            
        Returns:
            List of selected feature names
        """
        # Compute correlation with target
        correlations = {}
        for col in X.columns:
            corr = np.corrcoef(X[col], y)[0, 1]
            correlations[col] = abs(corr)
        
        # Select features above threshold
        selected = [col for col, corr in correlations.items() if corr >= threshold]
        self.selected_features['correlation'] = selected
        
        logger.info(f"Selected {len(selected)} features with |correlation| > {threshold}")
        return selected
    
    def select_by_rfe(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_features: int = 10
    ) -> List[str]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        selected = X.columns[rfe.support_].tolist()
        self.selected_features['rfe'] = selected
        
        logger.info(f"Selected {len(selected)} features using RFE")
        return selected
    
    def select_by_sequential(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_features: int = 10,
        direction: str = 'forward'
    ) -> List[str]:
        """
        Select features using Sequential Feature Selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            direction: 'forward' or 'backward'
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import SequentialFeatureSelector
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        sfs = SequentialFeatureSelector(
            estimator=model,
            n_features_to_select=n_features,
            direction=direction,
            cv=3,
            n_jobs=-1
        )
        
        sfs.fit(X, y)
        
        selected = X.columns[sfs.get_support()].tolist()
        self.selected_features[f'sequential_{direction}'] = selected
        
        logger.info(f"Selected {len(selected)} features using sequential {direction} selection")
        return selected


def analyze_feature_interactions(
    X: pd.DataFrame,
    y: np.ndarray,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Analyze interactions between top features.
    
    Args:
        X: Feature matrix
        y: Target variable
        top_n: Number of top feature pairs to return
        
    Returns:
        DataFrame with interaction strengths
    """
    from sklearn.ensemble import RandomForestClassifier
    from itertools import combinations
    
    logger.info("Analyzing feature interactions")
    
    # First, get base model performance
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    base_score = model.score(X, y)
    
    # Test all pairwise interactions
    interactions = []
    for feat1, feat2 in combinations(X.columns, 2):
        # Create interaction feature
        X_with_interaction = X.copy()
        X_with_interaction[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
        
        # Train model with interaction
        model_inter = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_inter.fit(X_with_interaction, y)
        inter_score = model_inter.score(X_with_interaction, y)
        
        # Record improvement
        improvement = inter_score - base_score
        interactions.append({
            'feature_1': feat1,
            'feature_2': feat2,
            'interaction': f'{feat1}_x_{feat2}',
            'improvement': improvement,
            'base_score': base_score,
            'interaction_score': inter_score
        })
    
    interactions_df = pd.DataFrame(interactions).sort_values('improvement', ascending=False)
    return interactions_df.head(top_n)
