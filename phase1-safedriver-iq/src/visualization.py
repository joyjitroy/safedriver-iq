"""
Visualization Module

Creates plots and charts for crash analysis and safety modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class CrashVisualizer:
    """
    Visualization tools for crash data analysis.
    """
    
    def __init__(self, save_dir: str = "results/figures"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = save_dir
    
    def plot_crash_trends(self, df: pd.DataFrame, save: bool = True):
        """
        Plot crash trends over time.
        
        Args:
            df: DataFrame with YEAR column
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        crash_by_year = df['YEAR'].value_counts().sort_index()
        
        ax.plot(crash_by_year.index, crash_by_year.values, marker='o', linewidth=2)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Crashes', fontsize=12)
        ax.set_title('Crash Trends Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(f"{self.save_dir}/crash_trends.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure: crash_trends.png")
        
        plt.show()
    
    def plot_vru_distribution(self, person_df: pd.DataFrame, save: bool = True):
        """
        Plot distribution of VRU types.
        
        Args:
            person_df: Person-level data
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        vru_df = person_df[person_df['PER_TYP'].isin([5, 6])]
        vru_counts = vru_df['PER_TYP'].value_counts()
        
        labels = ['Pedestrian', 'Bicyclist']
        colors = ['#ff7f0e', '#2ca02c']
        
        ax.bar(labels, vru_counts.values, color=colors)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('VRU Crash Distribution', fontsize=14, fontweight='bold')
        
        for i, v in enumerate(vru_counts.values):
            ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=11)
        
        if save:
            plt.savefig(f"{self.save_dir}/vru_distribution.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure: vru_distribution.png")
        
        plt.show()
    
    def plot_feature_importance(
        self, 
        importance_df: pd.DataFrame, 
        top_n: int = 20,
        save: bool = True
    ):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to plot
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        if save:
            plt.savefig(f"{self.save_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure: feature_importance.png")
        
        plt.show()
    
    def plot_safety_score_distribution(
        self, 
        scores: np.ndarray,
        save: bool = True
    ):
        """
        Plot distribution of safety scores.
        
        Args:
            scores: Array of safety scores
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.1f}')
        ax.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.1f}')
        
        ax.set_xlabel('Safety Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Safety Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(f"{self.save_dir}/safety_score_distribution.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure: safety_score_distribution.png")
        
        plt.show()
    
    def plot_risk_matrix(
        self, 
        df: pd.DataFrame,
        feature1: str,
        feature2: str,
        score_col: str = 'safety_score',
        save: bool = True
    ):
        """
        Plot 2D risk matrix.
        
        Args:
            df: DataFrame with features and scores
            feature1: First feature name
            feature2: Second feature name
            score_col: Safety score column name
            save: Whether to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        pivot = df.pivot_table(
            values=score_col,
            index=feature1,
            columns=feature2,
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax)
        ax.set_title(f'Risk Matrix: {feature1} vs {feature2}', fontsize=14, fontweight='bold')
        
        if save:
            plt.savefig(f"{self.save_dir}/risk_matrix_{feature1}_{feature2}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure: risk_matrix_{feature1}_{feature2}.png")
        
        plt.show()


if __name__ == "__main__":
    print("CrashVisualizer module loaded")
