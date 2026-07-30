"""
Feature Engineering Module

Creates features for inverse safety modeling from crash data.

Feature categories
------------------
  Tier 1 — derived from CRSS columns (always available):
    - Temporal          : rush hour, night, weekend, season, school window
    - Environmental     : weather, lighting, glare proxy
    - Location          : road type, urban/rural, speed limit buckets

  Tier 2 — synthesized via ContextualFeatureGenerator (when real data absent):
    - Traffic           : congestion index, aggressive driver count, speed variance
    - Road geometry     : lane width, curve, grade, sight distance
    - Work zone         : construction present, workers on road, lane reduction
    - Area context      : school zone, bar density, commercial density
    - Driver state      : DUI risk, fatigue, distraction probability
    - Enhanced env      : temperature, wind, precipitation, black-ice risk
    - Infrastructure    : markings, guardrails, road quality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Engineers features for crash analysis and safety modeling.
    
    Creates features in categories:
    - Temporal features (time, day, season)
    - Environmental features (weather, lighting)
    - Location features (road type, area, speed limit)
    - VRU features (pedestrian/cyclist specific)
    - Driver/Vehicle features
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with MONTH, DAY_WEEK, HOUR columns
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        
        # Hour of day features
        if 'HOUR' in df.columns:
            df['HOUR'] = pd.to_numeric(df['HOUR'], errors='coerce')
            df['IS_RUSH_HOUR'] = df['HOUR'].isin([7, 8, 9, 16, 17, 18]).astype(int)
            df['IS_NIGHT'] = df['HOUR'].isin(range(20, 24)) | df['HOUR'].isin(range(0, 6))
            df['IS_NIGHT'] = df['IS_NIGHT'].astype(int)
        
        # Day of week features
        if 'DAY_WEEK' in df.columns:
            df['DAY_WEEK'] = pd.to_numeric(df['DAY_WEEK'], errors='coerce')
            df['IS_WEEKEND'] = df['DAY_WEEK'].isin([1, 7]).astype(int)  # 1=Sunday, 7=Saturday
        
        # Season features
        if 'MONTH' in df.columns:
            df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')
            df['SEASON'] = df['MONTH'].apply(self._get_season)
        
        logger.info("Created temporal features")
        return df
    
    @staticmethod
    def _get_season(month: int) -> str:
        """Map month to season."""
        if pd.isna(month):
            return 'Unknown'
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def create_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather and lighting features.
        
        Args:
            df: DataFrame with weather and lighting columns
            
        Returns:
            DataFrame with environmental features
        """
        df = df.copy()
        
        # Weather features
        if 'WEATHER' in df.columns:
            # Adverse weather indicator
            adverse_weather_codes = [2, 3, 4, 5, 10, 11, 12]  # Rain, snow, fog, etc.
            df['ADVERSE_WEATHER'] = df['WEATHER'].isin(adverse_weather_codes).astype(int)
        
        # Lighting features
        if 'LGT_COND' in df.columns:
            # Poor lighting indicator
            poor_lighting_codes = [2, 3, 4]  # Dark (not lighted, lighted, unknown)
            df['POOR_LIGHTING'] = df['LGT_COND'].isin(poor_lighting_codes).astype(int)
        
        # Combined adverse conditions
        if 'ADVERSE_WEATHER' in df.columns and 'POOR_LIGHTING' in df.columns:
            df['ADVERSE_CONDITIONS'] = (df['ADVERSE_WEATHER'] & df['POOR_LIGHTING']).astype(int)
        
        logger.info("Created environmental features")
        return df
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location-based features.
        
        Args:
            df: DataFrame with road and location columns
            
        Returns:
            DataFrame with location features
        """
        df = df.copy()
        
        # Road type features
        if 'ROUTE' in df.columns:
            # Interstate/highway indicator
            df['IS_INTERSTATE'] = df['ROUTE'].isin([1]).astype(int)
        
        # Urban/rural
        if 'RUR_URB' in df.columns:
            df['IS_URBAN'] = (df['RUR_URB'] == 1).astype(int)
        
        # Speed limit features
        if 'SPD_LIM' in df.columns:
            df['SPD_LIM'] = pd.to_numeric(df['SPD_LIM'], errors='coerce')
            df['HIGH_SPEED_ROAD'] = (df['SPD_LIM'] >= 55).astype(int)
            df['LOW_SPEED_ROAD'] = (df['SPD_LIM'] <= 35).astype(int)
        
        # Traffic control device
        if 'TYP_INT' in df.columns:
            # Intersection type
            df['HAS_TRAFFIC_SIGNAL'] = df['TYP_INT'].isin([1]).astype(int)
        
        logger.info("Created location features")
        return df
    
    def create_vru_features(self, accident_df: pd.DataFrame, person_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create VRU-specific features.
        
        Args:
            accident_df: Accident-level data
            person_df: Person-level data
            
        Returns:
            DataFrame with VRU features added
        """
        df = accident_df.copy()
        
        # Count pedestrians and cyclists per crash
        vru_counts = person_df[person_df['PER_TYP'].isin([5, 6])].groupby('CASENUM').agg({
            'PER_TYP': lambda x: {
                'pedestrian_count': (x == 5).sum(),
                'cyclist_count': (x == 6).sum(),
                'total_vru': len(x)
            }
        })
        
        # Flatten the nested dict
        vru_counts_df = pd.DataFrame(vru_counts['PER_TYP'].tolist(), index=vru_counts.index)
        
        # Merge with accident data
        df = df.merge(vru_counts_df, left_on='CASENUM', right_index=True, how='left')
        df[['pedestrian_count', 'cyclist_count', 'total_vru']] = df[
            ['pedestrian_count', 'cyclist_count', 'total_vru']
        ].fillna(0)
        
        # VRU injury severity (from person data)
        if 'INJ_SEV' in person_df.columns:
            vru_severity = person_df[person_df['PER_TYP'].isin([5, 6])].groupby('CASENUM')['INJ_SEV'].agg([
                ('max_vru_injury', 'max'),
                ('fatal_vru', lambda x: (x == 4).any().astype(int))
            ])
            df = df.merge(vru_severity, left_on='CASENUM', right_index=True, how='left')
        
        logger.info("Created VRU features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Night + Poor lighting
        if 'IS_NIGHT' in df.columns and 'POOR_LIGHTING' in df.columns:
            df['NIGHT_AND_DARK'] = (df['IS_NIGHT'] & df['POOR_LIGHTING']).astype(int)
        
        # Urban + High speed
        if 'IS_URBAN' in df.columns and 'HIGH_SPEED_ROAD' in df.columns:
            df['URBAN_HIGH_SPEED'] = (df['IS_URBAN'] & df['HIGH_SPEED_ROAD']).astype(int)
        
        # Weekend + Night
        if 'IS_WEEKEND' in df.columns and 'IS_NIGHT' in df.columns:
            df['WEEKEND_NIGHT'] = (df['IS_WEEKEND'] & df['IS_NIGHT']).astype(int)

        # Rush hour + high traffic density
        if 'IS_RUSH_HOUR' in df.columns and 'TRAFFIC_DENSITY_INDEX' in df.columns:
            df['RUSH_HOUR_HIGH_TRAFFIC'] = (
                df['IS_RUSH_HOUR'] & (df['TRAFFIC_DENSITY_INDEX'] >= 3)
            ).astype(int)

        # DUI risk + night
        if 'DUI_RISK_INDEX' in df.columns and 'IS_NIGHT' in df.columns:
            df['NIGHT_DUI_RISK'] = (
                df['IS_NIGHT'] * df['DUI_RISK_INDEX']
            ).round(3)

        # Work zone + adverse weather
        if 'WORK_ZONE_PRESENT' in df.columns and 'ADVERSE_WEATHER' in df.columns:
            df['WORK_ZONE_BAD_WEATHER'] = (
                df['WORK_ZONE_PRESENT'] & df['ADVERSE_WEATHER']
            ).astype(int)

        # Curve + poor visibility
        curve_col = 'HAS_HORIZONTAL_CURVE' if 'HAS_HORIZONTAL_CURVE' in df.columns else None
        poor_vis = any(c in df.columns for c in ['POOR_LIGHTING', 'ADVERSE_WEATHER', 'BLACK_ICE_RISK'])
        if curve_col and poor_vis:
            fv = df.get('POOR_LIGHTING', 0).fillna(0) | df.get('ADVERSE_WEATHER', 0).fillna(0)
            df['CURVE_POOR_VISIBILITY'] = (df[curve_col] & fv).astype(int)

        # Narrow lane + high speed
        if 'LANE_WIDTH_FT' in df.columns and 'HIGH_SPEED_ROAD' in df.columns:
            df['NARROW_HIGH_SPEED'] = (
                (df['LANE_WIDTH_FT'] < 11) & df['HIGH_SPEED_ROAD']
            ).astype(int)

        # Fatigue + early morning
        if 'FATIGUE_RISK_INDEX' in df.columns and 'HOUR' in df.columns:
            hour = pd.to_numeric(df['HOUR'], errors='coerce').fillna(12).astype(int)
            df['FATIGUE_EARLY_MORNING'] = (
                (hour.isin([3, 4, 5, 6])) & (df['FATIGUE_RISK_INDEX'] > 0.5)
            ).astype(int)

        # Black ice + curve
        if 'BLACK_ICE_RISK' in df.columns and 'HAS_HORIZONTAL_CURVE' in df.columns:
            df['BLACK_ICE_ON_CURVE'] = (
                (df['BLACK_ICE_RISK'] > 0.3) & (df['HAS_HORIZONTAL_CURVE'] == 1)
            ).astype(int)
        
        logger.info("Created interaction features")
        return df

    # ------------------------------------------------------------------
    # Tier-2 synthetic feature creation (uses ContextualFeatureGenerator)
    # ------------------------------------------------------------------

    def create_contextual_features(
        self,
        df: pd.DataFrame,
        random_seed: Optional[int] = 42
    ) -> pd.DataFrame:
        """
        Synthesise ~40 contextual features that are NOT present in CRSS records
        but are known to influence crash probability. Uses the
        ``ContextualFeatureGenerator`` with condition-aware sampling so the
        generated values are statistically consistent with available CRSS columns
        (HOUR, WEATHER, SPD_LIM, RUR_URB, etc.).

        Args:
            df:           Accident-level DataFrame (CRSS or synthetic)
            random_seed:  Seed for reproducibility

        Returns:
            DataFrame with all contextual feature columns appended
        """
        from src.contextual_feature_generator import ContextualFeatureGenerator
        gen = ContextualFeatureGenerator(random_seed=random_seed)
        df_aug = gen.augment_crss_dataframe(df)
        logger.info(
            f"Created contextual features: "
            f"{len(df_aug.columns) - len(df.columns)} new columns added"
        )
        return df_aug

    def create_driver_behavior_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive driver behaviour risk proxies from CRSS-available columns.

        CRSS has several tables (distract.csv, drimpair.csv, violatn.csv,
        maneuver.csv) that, when joined at vehicle level, reveal driver state.
        This method creates aggregate flags at the crash level from those
        pre-joined columns if present, otherwise zero-fills.

        Args:
            df: Accident-level DataFrame (optionally with distract/impair columns)

        Returns:
            DataFrame with driver behaviour proxy features
        """
        df = df.copy()

        # DRIMPAIR (from drimpair.csv join): physical impairment code
        # Code 5 = Under Influence of Alcohol, 6 = medications, 7 = illicit drugs
        if 'DRIMPAIR' in df.columns:
            df['DRIVER_IMPAIRED'] = df['DRIMPAIR'].isin([5, 6, 7]).astype(int)
        else:
            df['DRIVER_IMPAIRED'] = 0

        # DISTRACT (from distract.csv join): distraction codes
        # Code 5 = phone hand-held, 6 = phone hands-free, 15 = looked not seen
        if 'DISTRACT' in df.columns:
            df['DRIVER_DISTRACTED'] = df['DISTRACT'].isin([5, 6, 12, 15]).astype(int)
        else:
            df['DRIVER_DISTRACTED'] = 0

        # MDRMANAV (from maneuver.csv): improper manoeuvre
        # Code 15 = improper lane change, 4 = ran red light, 5 = ran stop sign
        if 'MDRMANAV' in df.columns:
            df['IMPROPER_MANEUVER'] = df['MDRMANAV'].isin([4, 5, 15, 17]).astype(int)
        else:
            df['IMPROPER_MANEUVER'] = 0

        # SPEEDREL (from vehicle.csv): speed relative to speed limit
        # Code 3 = exceeding speed limit, 4 = racing, 5 = too fast for conditions
        if 'SPEEDREL' in df.columns:
            df['SPEED_RELATED'] = df['SPEEDREL'].isin([3, 4, 5]).astype(int)
        elif 'SPEED_REL' in df.columns:
            df['SPEED_RELATED'] = (df['SPEED_REL'] >= 3).astype(int)
        else:
            df['SPEED_RELATED'] = 0

        # Composite driver risk score
        risk_cols = [c for c in ['DRIVER_IMPAIRED', 'DRIVER_DISTRACTED',
                                  'IMPROPER_MANEUVER', 'SPEED_RELATED'] if c in df.columns]
        df['DRIVER_RISK_SCORE'] = df[risk_cols].sum(axis=1)

        logger.info("Created driver behaviour proxy features")
        return df

    def create_school_hours_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag records in school-zone arrival/dismissal windows.

        Uses HOUR and (if available) SCHOOL_ZONE column.  If SCHOOL_ZONE
        is absent, applies base probability proportional to urbanisation.

        Args:
            df: DataFrame with HOUR column

        Returns:
            DataFrame with SCHOOL_HOURS_ACTIVE column
        """
        df = df.copy()
        school_hours = list(range(7, 9)) + list(range(14, 17))

        if 'HOUR' in df.columns:
            hour = pd.to_numeric(df['HOUR'], errors='coerce').fillna(12).astype(int)
            in_school_window = hour.isin(school_hours).astype(int)
            if 'SCHOOL_ZONE' in df.columns:
                df['SCHOOL_HOURS_ACTIVE'] = (
                    in_school_window & df['SCHOOL_ZONE'].fillna(0).astype(int)
                ).astype(int)
            else:
                df['SCHOOL_HOURS_ACTIVE'] = in_school_window
        else:
            df['SCHOOL_HOURS_ACTIVE'] = 0

        logger.info("Created school hours feature")
        return df

    def engineer_features_pipeline(
        self, 
        accident_df: pd.DataFrame, 
        person_df: Optional[pd.DataFrame] = None,
        include_contextual: bool = True,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            accident_df:          Accident-level CRSS data
            person_df:            Person-level data (optional, for VRU features)
            include_contextual:   When True, synthesises ~40 contextual features
                                  for factors not captured in CRSS (traffic
                                  aggressiveness, road geometry, work zones,
                                  driver state proxies, black-ice risk, etc.)
            random_seed:          Seed for contextual feature synthesis
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("\n=== Starting Feature Engineering Pipeline ===")
        
        df = accident_df.copy()
        
        # Tier-1: derived from CRSS columns
        df = self.create_temporal_features(df)
        df = self.create_environmental_features(df)
        df = self.create_location_features(df)
        df = self.create_school_hours_feature(df)
        df = self.create_driver_behavior_proxy_features(df)
        
        if person_df is not None:
            df = self.create_vru_features(df, person_df)
        
        # Tier-2: synthesised contextual features
        if include_contextual:
            logger.info("Synthesising contextual features (traffic, geometry, work zone, …)")
            df = self.create_contextual_features(df, random_seed=random_seed)

        # Interaction features (runs last so it can use contextual columns)
        df = self.create_interaction_features(df)
        
        logger.info(f"\nFeature engineering complete")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Records: {len(df):,}")
        
        return df


if __name__ == "__main__":
    # Example usage
    from data_loader import CRSSDataLoader
    from preprocessing import CrashPreprocessor
    
    loader = CRSSDataLoader()
    datasets = loader.load_complete_dataset()
    
    preprocessor = CrashPreprocessor()
    processed = preprocessor.preprocess_pipeline(datasets)
    
    engineer = FeatureEngineer()
    featured_df = engineer.engineer_features_pipeline(
        processed['accident'], 
        processed['person']
    )
    
    print("\nFeature Engineering Complete!")
    print(f"Features created: {len(featured_df.columns)}")
