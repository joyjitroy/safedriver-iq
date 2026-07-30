"""
Contextual Feature Generator for SafeDriver-IQ

Synthesizes realistic contextual features that are NOT directly available in CRSS
crash reports but are known to significantly influence crash probability.

Data sources informing the distributions:
  - NHTSA FARS/CRSS crash factor studies
  - FHWA Highway Safety Manual (HSM)
  - IIHS insurance loss statistics
  - AAA Foundation for Traffic Safety driver behavior studies
  - BTS National Transportation Statistics

Feature categories synthesized:
  1.  Traffic congestion & nearby driver aggressiveness
  2.  Road construction / work-zone conditions
  3.  Road geometry (lane width, curve, grade, sight distance)
  4.  Area characteristics (school zones, bar density, commercial density)
  5.  Driver state proxies (fatigue window, DUI risk by time/day)
  6.  Enhanced lighting / visibility (temperature, wind, black ice)
  7.  Infrastructure quality (guardrails, markings, signage)
  8.  Speed differential dynamics (surrounding vehicle speed variance)

Usage
-----
    from src.contextual_feature_generator import ContextualFeatureGenerator

    gen = ContextualFeatureGenerator(random_seed=42)

    # Augment an existing CRSS accident DataFrame
    df_augmented = gen.augment_crss_dataframe(accident_df)

    # Or generate a standalone synthetic scenario
    scenarios = gen.generate_synthetic_scenarios(n=1000)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Research-backed prior probabilities / distributions
# ---------------------------------------------------------------------------

# NHTSA: ~26 % of crashes occur in work zones (when workers are present ~60 %)
WORK_ZONE_BASE_PROB = 0.06          # ~6 % of all CRSS crashes involve a work zone
WORKERS_PRESENT_GIVEN_WZ = 0.60

# FHWA: crash rates on narrow lanes (<10 ft) are up to 3× higher than 12-ft lanes
LANE_WIDTH_DIST = {                 # feet, weighted toward standard widths
    'values': [9, 10, 11, 12, 13, 14],
    'probs': [0.03, 0.07, 0.15, 0.50, 0.18, 0.07]
}

# HSM: ~12 % of road miles have significant horizontal curves
CURVE_PROB = 0.12
# Grade distribution across US roads (%)
GRADE_DIST = {'mean': 2.0, 'std': 3.5, 'clip': (0, 15)}

# IIHS / AAA: ~30 % of fatal crashes involve alcohol, peak 12-4 AM weekend
DUI_RISK_BY_HOUR = {               # baseline multipliers (relative to 1.0)
    **{h: 1.0 for h in range(6, 20)},
    20: 1.5, 21: 2.0, 22: 3.0, 23: 4.0,
    0: 5.5, 1: 6.0, 2: 5.5, 3: 4.0, 4: 2.0, 5: 1.2
}
DUI_WEEKEND_MULTIPLIER = 1.8

# AAA: aggressive driving (tailgating, rapid lane changes) present in ~50 % crashes
AGGRESSIVE_PROB_BASE = 0.15        # base probability per scenario
AGGRESSIVE_RUSH_HOUR_MULT = 2.5   # during rush hour risk doubles+

# NOAA: temperature and precipitation correlate strongly with black ice
BLACK_ICE_TEMP_THRESHOLD_F = 35   # below 35 °F + moisture = high risk

# School zone hours (when active)
SCHOOL_HOURS = list(range(7, 9)) + list(range(14, 17))

# BAR density (establishments / sq mile) affects late-night DUI risk
# Low: 0-2, Medium: 3-8, High: 9-20
BAR_DENSITY_DIST = {'values': ['low', 'medium', 'high'], 'probs': [0.50, 0.35, 0.15]}


class ContextualFeatureGenerator:
    """
    Generates synthetic contextual driving features with realistic statistical
    distributions to augment CRSS crash records or to build standalone scenarios.

    When augmenting real CRSS data the generator is **condition-aware**: it uses
    existing CRSS columns (HOUR, WEATHER, RUR_URB, SPD_LIM, etc.) to set
    conditionally-correct distributions matching known research findings.
    """

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.default_rng(random_seed)
        logger.info("ContextualFeatureGenerator initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment_crss_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment a CRSS accident DataFrame with synthesized contextual features.

        Uses existing columns (HOUR, DAY_WEEK, MONTH, WEATHER, LGT_COND,
        RUR_URB, SPD_LIM, etc.) to sample contextually-consistent values for
        each crash record.

        Args:
            df: CRSS accident-level DataFrame (one row = one crash)

        Returns:
            DataFrame with ~40 additional contextual feature columns
        """
        logger.info(f"Augmenting {len(df):,} CRSS records with contextual features …")
        out = df.copy()

        # Pull columns if present, fall back to neutral defaults
        hour = pd.to_numeric(out.get('HOUR', 12), errors='coerce').fillna(12).astype(int)
        day  = pd.to_numeric(out.get('DAY_WEEK', 3), errors='coerce').fillna(3).astype(int)
        month = pd.to_numeric(out.get('MONTH', 6), errors='coerce').fillna(6).astype(int)
        weather = pd.to_numeric(out.get('WEATHER', 1), errors='coerce').fillna(1).astype(int)
        spd_lim = pd.to_numeric(out.get('SPD_LIM', 35), errors='coerce').fillna(35)
        rural = (pd.to_numeric(out.get('RUR_URB', 2), errors='coerce').fillna(2) == 2).astype(int)

        n = len(out)

        # --- 1. Traffic & Congestion features ----------------------------
        out = self._add_traffic_features(out, n, hour, day, rural)

        # --- 2. Nearby driver behavior -----------------------------------
        out = self._add_nearby_driver_features(out, n, hour, day)

        # --- 3. Road construction / work zone ----------------------------
        out = self._add_work_zone_features(out, n, hour, day)

        # --- 4. Road geometry -------------------------------------------
        out = self._add_road_geometry_features(out, n, rural, spd_lim)

        # --- 5. Area characteristics ------------------------------------
        out = self._add_area_features(out, n, hour, rural)

        # --- 6. Driver state proxies ------------------------------------
        out = self._add_driver_state_features(out, n, hour, day)

        # --- 7. Enhanced environmental / visibility ---------------------
        out = self._add_enhanced_env_features(out, n, month, weather)

        # --- 8. Infrastructure quality ----------------------------------
        out = self._add_infrastructure_features(out, n, rural, spd_lim)

        logger.info(f"Augmentation complete — added {len(out.columns) - len(df.columns)} features")
        return out

    def generate_synthetic_scenarios(
        self,
        n: int = 1000,
        scenario_type: str = 'mixed'
    ) -> pd.DataFrame:
        """
        Generate a fully synthetic scenario DataFrame with all contextual features.

        Args:
            n: Number of scenarios to generate
            scenario_type: 'mixed' | 'rush_hour' | 'night_dui' | 'work_zone' |
                           'bad_weather' | 'aggressive_traffic' | 'school_zone' |
                           'construction_zone' | 'narrow_road' | 'black_ice'

        Returns:
            DataFrame with all base + contextual features
        """
        logger.info(f"Generating {n} synthetic '{scenario_type}' scenarios …")

        base = self._generate_base_conditions(n, scenario_type)
        df = pd.DataFrame(base)

        hour  = df['HOUR'].astype(int)
        day   = df['DAY_WEEK'].astype(int)
        month = df['MONTH'].astype(int)
        weather = df['WEATHER'].astype(int)
        spd_lim = df['SPD_LIM']
        rural = df['IS_RURAL'].astype(int)

        df = self._add_traffic_features(df, n, hour, day, rural)
        df = self._add_nearby_driver_features(df, n, hour, day)
        df = self._add_work_zone_features(df, n, hour, day)
        df = self._add_road_geometry_features(df, n, rural, spd_lim)
        df = self._add_area_features(df, n, hour, rural)
        df = self._add_driver_state_features(df, n, hour, day)
        df = self._add_enhanced_env_features(df, n, month, weather)
        df = self._add_infrastructure_features(df, n, rural, spd_lim)
        df = self._add_crash_probability_label(df)

        logger.info(f"Generated {len(df)} synthetic scenarios with {len(df.columns)} features")
        return df

    def scenario_feature_catalog(self) -> pd.DataFrame:
        """Return a DataFrame listing every synthesized feature with description."""
        catalog = [
            # Traffic & Congestion
            ('TRAFFIC_DENSITY_INDEX', 'traffic',
             'Vehicles-per-mile index  [1=free-flow … 5=gridlock]'),
            ('TRAFFIC_VOLUME_CATEGORY', 'traffic',
             'Categorical: low / medium / high / peak'),
            ('CONGESTION_DELAY_MINUTES', 'traffic',
             'Estimated delay due to congestion (min)'),
            # Nearby driver behaviour
            ('NEARBY_AGGRESSIVE_DRIVER_COUNT', 'nearby_behavior',
             'Estimated no. of aggressive drivers within 0.5 mi'),
            ('LANE_CHANGE_FREQ_PER_MILE', 'nearby_behavior',
             'Observed lane changes per mile by surrounding vehicles'),
            ('TAILGATING_DETECTED_NEARBY', 'nearby_behavior',
             'Binary: at least one following-too-close vehicle nearby'),
            ('AVG_SURROUNDING_SPEED_OVER_LIMIT', 'nearby_behavior',
             'Average speed of surrounding vehicles above posted limit (mph)'),
            ('SPEED_VARIANCE_NEARBY_MPH', 'nearby_behavior',
             'Standard deviation of speeds in vicinity (mph) — higher = chaotic'),
            # Work zone
            ('WORK_ZONE_PRESENT', 'work_zone',
             'Binary: active road construction in crash segment'),
            ('WORK_ZONE_WORKERS_PRESENT', 'work_zone',
             'Binary: workers physically on roadway (higher penalty)'),
            ('WORK_ZONE_LANE_REDUCTION', 'work_zone',
             'No. of lanes closed/reduced by construction (0–3)'),
            ('WORK_ZONE_LENGTH_MILES', 'work_zone',
             'Length of active work zone (miles)'),
            # Road geometry
            ('LANE_WIDTH_FT', 'geometry',
             'Width of travel lane (feet); narrow < 10 ft = higher risk'),
            ('NO_OF_LANES', 'geometry',
             'Total number of through-travel lanes'),
            ('HAS_HORIZONTAL_CURVE', 'geometry',
             'Binary: crash on or near a horizontal curve'),
            ('CURVE_RADIUS_FT', 'geometry',
             'Curve radius in feet (lower = sharper); 99999 = straight'),
            ('ROAD_GRADE_PERCENT', 'geometry',
             'Road slope  (%) — affects stopping distance & wet-road risk'),
            ('SIGHT_DISTANCE_FT', 'geometry',
             'Available stopping sight distance (feet)'),
            ('MEDIAN_TYPE', 'geometry',
             'Categorical: none / painted / raised_curb / barrier / divided'),
            # Area characteristics
            ('SCHOOL_ZONE', 'area',
             'Binary: active school zone (school session + school hours)'),
            ('SCHOOL_HOURS_ACTIVE', 'area',
             'Binary: within morning/afternoon school arrival-departure window'),
            ('BAR_DENSITY_CATEGORY', 'area',
             'Density of bars/liquor establishments: low / medium / high'),
            ('COMMERCIAL_DENSITY_INDEX', 'area',
             'Index of commercial activity (driveway density, pedestrian generators)'),
            ('RESIDENTIAL_DENSITY_INDEX', 'area',
             'Population density index (people per sq mi, normalised 1–10)'),
            ('NEAR_HOSPITAL', 'area',
             'Binary: emergency vehicle activity (within 0.5 mi of hospital)'),
            ('EVENT_NEARBY', 'area',
             'Binary: large crowd event (game, concert) within 2 mi'),
            # Driver state proxies
            ('DUI_RISK_INDEX', 'driver_state',
             'Probabilistic DUI risk [0–1] from hour × day × bar density'),
            ('FATIGUE_RISK_INDEX', 'driver_state',
             'Fatigue risk [0–1]: early morning + long-haul corridor indicator'),
            ('DISTRACTED_DRIVING_RISK', 'driver_state',
             'Estimated probability of cell-phone / in-car distraction'),
            ('ESTIMATED_TRIP_DURATION_MIN', 'driver_state',
             'Synthetic estimated trip length (proxy for fatigue exposure)'),
            # Enhanced environmental
            ('TEMPERATURE_F', 'enhanced_env',
             'Ambient temperature (°F)'),
            ('WIND_SPEED_MPH', 'enhanced_env',
             'Wind speed (mph) — affects high-profile vehicles & cyclists'),
            ('PRECIPITATION_RATE_IN_HR', 'enhanced_env',
             'Precipitation intensity (in/hr): 0=none, 0.1=light, 0.5=heavy'),
            ('BLACK_ICE_RISK', 'enhanced_env',
             'Probability of black ice [0–1] from temp × moisture × month'),
            ('ROAD_SURFACE_TEMP_F', 'enhanced_env',
             'Estimated road surface temperature (lags ambient by ~2 °F)'),
            ('GLARE_CONDITIONS', 'enhanced_env',
             'Binary: sun glare (sunrise/sunset hours with clear sky)'),
            ('FOG_VISIBILITY_METERS', 'enhanced_env',
             'Estimated visibility in fog (meters); 10000 = clear'),
            # Infrastructure quality
            ('LANE_MARKINGS_VISIBLE', 'infrastructure',
             'Binary: lane markings clearly visible (0=faded/missing)'),
            ('SIGNAGE_VISIBILITY_SCORE', 'infrastructure',
             'Quality of road sign visibility [1–5]'),
            ('HAS_GUARDRAIL', 'infrastructure',
             'Binary: guardrail present on at least one side'),
            ('HAS_RUMBLE_STRIPS', 'infrastructure',
             'Binary: shoulder or centreline rumble strips present'),
            ('SPEED_CAMERA_WITHIN_1MI', 'infrastructure',
             'Binary: speed enforcement camera within 1 mile'),
            ('ROAD_QUALITY_INDEX', 'infrastructure',
             'Pavement quality score [1=poor … 5=excellent]'),
        ]
        return pd.DataFrame(catalog, columns=['feature', 'category', 'description'])

    # ------------------------------------------------------------------
    # Private helpers — feature groups
    # ------------------------------------------------------------------

    def _add_traffic_features(self, df, n, hour, day, rural):
        """Traffic density and congestion features."""
        is_rush = hour.isin([7, 8, 9, 16, 17, 18])
        is_weekend = day.isin([1, 7])

        # Traffic density index [1–5]: peaks during rush, lower rural/night
        density_base = self.rng.uniform(1, 3, n)
        density_base = np.where(is_rush, density_base * self.rng.uniform(1.5, 2.5, n), density_base)
        density_base = np.where(is_weekend, density_base * 0.75, density_base)
        density_base = np.where(rural == 1, density_base * 0.60, density_base)
        density_base = np.clip(density_base, 1, 5)
        df['TRAFFIC_DENSITY_INDEX'] = density_base.round(2)

        def _vol_cat(v):
            if v < 1.8:
                return 'low'
            elif v < 2.8:
                return 'medium'
            elif v < 4.0:
                return 'high'
            return 'peak'

        df['TRAFFIC_VOLUME_CATEGORY'] = pd.Series(density_base).apply(_vol_cat).values

        # Congestion delay proportional to density
        delay = np.maximum(0, (density_base - 2) * self.rng.exponential(4, n))
        delay = np.where(is_rush, delay * 1.8, delay)
        df['CONGESTION_DELAY_MINUTES'] = np.round(delay, 1)

        return df

    def _add_nearby_driver_features(self, df, n, hour, day):
        """Aggressive driving and speed variance features."""
        is_rush  = hour.isin([7, 8, 9, 16, 17, 18]).to_numpy()
        is_night = (hour.isin(list(range(20, 24)) + list(range(0, 5)))).to_numpy()

        # Aggressive driver count
        agg_rate = np.where(is_rush,
                            self.rng.poisson(2.5, n),
                            np.where(is_night,
                                     self.rng.poisson(1.8, n),
                                     self.rng.poisson(0.8, n)))
        df['NEARBY_AGGRESSIVE_DRIVER_COUNT'] = agg_rate.astype(int)

        # Lane change frequency (changes / mile)
        lc_base = self.rng.gamma(2, 0.8, n)
        lc = np.where(is_rush, lc_base * 2.2, lc_base)
        df['LANE_CHANGE_FREQ_PER_MILE'] = np.round(lc, 2)

        # Tailgating detected
        tailgate_p = np.where(is_rush, 0.45, np.where(is_night, 0.20, 0.12))
        df['TAILGATING_DETECTED_NEARBY'] = (self.rng.random(n) < tailgate_p).astype(int)

        # Speed over limit
        speed_excess = self.rng.gamma(2, 3, n)
        speed_excess = np.where(is_night, speed_excess * 1.5, speed_excess)
        df['AVG_SURROUNDING_SPEED_OVER_LIMIT'] = np.round(speed_excess, 1)

        # Speed variance nearby
        spd_var = self.rng.gamma(3, 2, n)
        spd_var = np.where(is_rush, spd_var * 1.6, spd_var)
        df['SPEED_VARIANCE_NEARBY_MPH'] = np.round(spd_var, 2)

        return df

    def _add_work_zone_features(self, df, n, hour, day):
        """Road construction / work-zone features."""
        # Work zones more common on weekdays daytime
        is_weekday_day = (~day.isin([1, 7])) & hour.isin(range(6, 20))
        wz_prob = np.where(is_weekday_day, WORK_ZONE_BASE_PROB * 1.8, WORK_ZONE_BASE_PROB * 0.4)
        wz = (self.rng.random(n) < wz_prob).astype(int)
        df['WORK_ZONE_PRESENT'] = wz

        workers = np.where(wz == 1,
                           (self.rng.random(n) < WORKERS_PRESENT_GIVEN_WZ).astype(int),
                           0)
        df['WORK_ZONE_WORKERS_PRESENT'] = workers

        lane_red = np.where(wz == 1,
                            self.rng.choice([0, 1, 2, 3], n, p=[0.1, 0.5, 0.3, 0.1]),
                            0)
        df['WORK_ZONE_LANE_REDUCTION'] = lane_red.astype(int)

        wz_len = np.where(wz == 1,
                          self.rng.exponential(1.5, n),
                          0.0)
        df['WORK_ZONE_LENGTH_MILES'] = np.round(wz_len, 2)

        return df

    def _add_road_geometry_features(self, df, n, rural, spd_lim):
        """Lane width, curve, grade and sight-distance features."""
        # Lane width — narrower on rural/local roads
        lane_w_vals = LANE_WIDTH_DIST['values']
        lane_w_probs = LANE_WIDTH_DIST['probs']
        lane_base = self.rng.choice(lane_w_vals, n, p=lane_w_probs)
        # Rural roads tend narrower
        shift = np.where(rural == 1, -1, 0)
        lane_w = np.clip(lane_base + shift, 9, 14)
        df['LANE_WIDTH_FT'] = lane_w.astype(int)

        # Number of lanes
        lanes = np.where(spd_lim >= 55,
                         self.rng.choice([4, 6, 8], n, p=[0.5, 0.35, 0.15]),
                         np.where(rural == 1,
                                  self.rng.choice([2, 4], n, p=[0.75, 0.25]),
                                  self.rng.choice([2, 4, 6], n, p=[0.4, 0.45, 0.15])))
        df['NO_OF_LANES'] = lanes.astype(int)

        # Horizontal curve
        curve_p = np.where(rural == 1, CURVE_PROB * 1.8, CURVE_PROB * 0.8)
        has_curve = (self.rng.random(n) < curve_p).astype(int)
        df['HAS_HORIZONTAL_CURVE'] = has_curve

        radius = np.where(has_curve == 1,
                          np.clip(self.rng.gamma(5, 200, n), 100, 3000),
                          99999)
        df['CURVE_RADIUS_FT'] = radius.astype(int)

        # Grade
        grade = np.abs(self.rng.normal(GRADE_DIST['mean'], GRADE_DIST['std'], n))
        grade = np.clip(grade, *GRADE_DIST['clip'])
        grade = np.where(rural == 1, grade * 1.3, grade)
        df['ROAD_GRADE_PERCENT'] = np.round(grade, 1)

        # Sight distance: inversely related to curve and grade
        base_sd = np.where(spd_lim >= 55, 800, np.where(spd_lim >= 35, 400, 200))
        sd = base_sd - (has_curve * self.rng.uniform(100, 400, n)) - (grade * 10)
        df['SIGHT_DISTANCE_FT'] = np.clip(sd, 50, 2000).astype(int)

        # Median type
        median_opts = ['none', 'painted', 'raised_curb', 'barrier', 'divided']
        median_p_urban  = [0.25, 0.30, 0.20, 0.10, 0.15]
        median_p_rural  = [0.40, 0.25, 0.10, 0.08, 0.17]
        median_urban = self.rng.choice(median_opts, n, p=median_p_urban)
        median_rural = self.rng.choice(median_opts, n, p=median_p_rural)
        df['MEDIAN_TYPE'] = np.where(rural == 1, median_rural, median_urban)

        return df

    def _add_area_features(self, df, n, hour, rural):
        """School zone, bar density, commercial density."""
        # School zone — suburban/urban areas more likely
        school_active = hour.isin(SCHOOL_HOURS)
        school_zone_p = np.where(rural == 1, 0.02, 0.12)
        school_zone = (self.rng.random(n) < school_zone_p).astype(int)
        df['SCHOOL_ZONE'] = school_zone

        school_hrs_active = (school_active & (school_zone == 1)).astype(int)
        df['SCHOOL_HOURS_ACTIVE'] = school_hrs_active.to_numpy().astype(int) if hasattr(school_hrs_active, 'to_numpy') else school_hrs_active.astype(int)

        # Bar density
        bar_vals = BAR_DENSITY_DIST['values']
        bar_p = BAR_DENSITY_DIST['probs']
        bar_dense_urban = self.rng.choice(bar_vals, n, p=bar_p)
        bar_dense_rural = self.rng.choice(bar_vals, n, p=[0.70, 0.25, 0.05])
        df['BAR_DENSITY_CATEGORY'] = np.where(rural == 1, bar_dense_rural, bar_dense_urban)

        # Commercial density index [1–10]
        comm = np.where(rural == 1,
                        self.rng.gamma(2, 1, n),
                        self.rng.gamma(3, 2, n))
        df['COMMERCIAL_DENSITY_INDEX'] = np.round(np.clip(comm, 1, 10), 1)

        # Residential density index [1–10]
        res = np.where(rural == 1,
                       self.rng.gamma(1.5, 1, n),
                       self.rng.gamma(4, 1.5, n))
        df['RESIDENTIAL_DENSITY_INDEX'] = np.round(np.clip(res, 1, 10), 1)

        # Near hospital
        df['NEAR_HOSPITAL'] = (self.rng.random(n) < np.where(rural == 1, 0.03, 0.12)).astype(int)

        # Large event nearby (weekend + evening)
        is_weekend_eve = hour.isin([17, 18, 19, 20, 21, 22]) & hour.isin([1, 7]).pipe(lambda x: ~x)
        # Simplified: weekend evening ~8 %
        event_p = np.where(rural == 1, 0.01, 0.05)
        df['EVENT_NEARBY'] = (self.rng.random(n) < event_p).astype(int)

        return df

    def _add_driver_state_features(self, df, n, hour, day):
        """DUI risk, fatigue, distraction proxies."""
        # DUI risk index
        dui_mult = np.array([DUI_RISK_BY_HOUR.get(int(h), 1.0) for h in hour])
        weekend_mult = np.where(day.isin([1, 7]), DUI_WEEKEND_MULTIPLIER, 1.0)
        bar_cat = df.get('BAR_DENSITY_CATEGORY', pd.Series(['low'] * n))
        bar_mult = bar_cat.map({'low': 1.0, 'medium': 1.5, 'high': 2.2}).fillna(1.0).to_numpy()
        raw_dui = dui_mult * weekend_mult * bar_mult * 0.05  # scale to ~0-1
        df['DUI_RISK_INDEX'] = np.round(np.clip(raw_dui, 0, 1), 3)

        # Fatigue risk: peaks early morning (4-7 AM) and very late night
        fatigue_base = np.where(hour.isin([4, 5, 6, 3, 2]),
                                 self.rng.beta(4, 2, n),      # high fatigue
                                 np.where(hour.isin([13, 14]),
                                          self.rng.beta(2, 3, n),  # post-lunch dip
                                          self.rng.beta(1, 5, n)))  # otherwise low
        df['FATIGUE_RISK_INDEX'] = np.round(np.clip(fatigue_base, 0, 1), 3)

        # Distracted driving (cell phone / in-cabin distraction)
        # AAA: 1 in 4 crashes involve distracted driving
        distract_base = self.rng.beta(2, 6, n)   # mean ~0.25
        # Rush hour slightly higher (phone use in traffic)
        distract = np.where(hour.isin([7, 8, 9, 16, 17, 18]),
                            distract_base * 1.2, distract_base)
        df['DISTRACTED_DRIVING_RISK'] = np.round(np.clip(distract, 0, 1), 3)

        # Estimated trip duration
        trip_dur = self.rng.exponential(25, n)    # minutes; most trips < 60 min
        trip_dur = np.where(hour.isin([7, 8, 16, 17, 18]),
                            trip_dur * 1.5, trip_dur)   # longer during commute
        df['ESTIMATED_TRIP_DURATION_MIN'] = np.round(np.clip(trip_dur, 1, 300), 1)

        return df

    def _add_enhanced_env_features(self, df, n, month, weather):
        """Temperature, wind, precipitation, black-ice, glare."""
        # Temperature by month (°F) — simplified US average
        temp_by_month = {1: 30, 2: 33, 3: 43, 4: 54, 5: 63, 6: 72,
                         7: 78, 8: 76, 9: 67, 10: 55, 11: 43, 12: 33}
        temp_mean = month.map(temp_by_month).fillna(55)
        temp = self.rng.normal(temp_mean, 10, n)
        df['TEMPERATURE_F'] = np.round(temp, 1)

        # Wind speed (mph) — higher in spring/winter
        wind_seasonal = month.map({1: 12, 2: 11, 3: 13, 4: 12, 5: 10, 6: 9,
                                   7: 8, 8: 8, 9: 9, 10: 10, 11: 12, 12: 13}).fillna(10)
        wind = self.rng.gamma(2, wind_seasonal / 2, n)
        df['WIND_SPEED_MPH'] = np.round(np.clip(wind, 0, 60), 1)

        # Precipitation rate (in/hr)
        rain_weather = [2, 3, 12]   # rain codes in CRSS
        snow_weather = [4, 5, 7]
        is_rain = weather.isin(rain_weather).to_numpy()
        is_snow = weather.isin(snow_weather).to_numpy()
        precip = np.where(is_rain,
                          self.rng.gamma(2, 0.15, n),
                          np.where(is_snow,
                                   self.rng.gamma(1.5, 0.05, n),
                                   0.0))
        df['PRECIPITATION_RATE_IN_HR'] = np.round(precip, 3)

        # Black ice risk
        cold = (temp < BLACK_ICE_TEMP_THRESHOLD_F).astype(float)
        moisture = np.clip(
            (precip > 0).astype(float) + is_rain.astype(float) + is_snow.astype(float),
            0, 1
        )
        winter = month.isin([11, 12, 1, 2, 3]).astype(float)
        black_ice = cold * moisture * winter * self.rng.beta(3, 2, n)
        df['BLACK_ICE_RISK'] = np.round(np.clip(black_ice, 0, 1), 3)

        # Road surface temperature lags ambient by ~2–5 °F at night
        df['ROAD_SURFACE_TEMP_F'] = np.round(temp - self.rng.uniform(0, 5, n), 1)

        # Glare: sunrise (6-8 AM) or sunset (4-7 PM) + clear weather
        hour_col = None
        for col in ['HOUR']:
            if col in df.columns:
                hour_col = pd.to_numeric(df[col], errors='coerce').fillna(12).astype(int)
        is_glare_hour = np.zeros(n, dtype=int)
        if hour_col is not None:
            is_glare_hour = hour_col.isin([6, 7, 8, 16, 17, 18]).to_numpy().astype(int)
        clear_sky = (weather == 1).to_numpy().astype(int)
        df['GLARE_CONDITIONS'] = (is_glare_hour & clear_sky).astype(int)

        # Fog visibility
        is_fog = weather.isin([5, 6]).to_numpy()
        fog_vis = np.where(is_fog,
                           self.rng.gamma(2, 100, n),   # poor visibility
                           10000)                        # clear
        df['FOG_VISIBILITY_METERS'] = np.round(np.clip(fog_vis, 10, 10000), 0).astype(int)

        return df

    def _add_infrastructure_features(self, df, n, rural, spd_lim):
        """Guardrails, markings, signage, road quality."""
        # Lane markings visibility — older rural roads worse
        marking_p = np.where(rural == 1, 0.75, 0.90)
        df['LANE_MARKINGS_VISIBLE'] = (self.rng.random(n) < marking_p).astype(int)

        # Signage visibility score [1–5]
        sign_base = self.rng.normal(3.5, 0.8, n)
        sign = np.where(rural == 1, sign_base - 0.5, sign_base)
        df['SIGNAGE_VISIBILITY_SCORE'] = np.round(np.clip(sign, 1, 5), 1)

        # Guardrail presence: highways/high speed roads more likely
        gr_p = np.where(spd_lim >= 55, 0.75,
                        np.where(spd_lim >= 35, 0.40, 0.20))
        df['HAS_GUARDRAIL'] = (self.rng.random(n) < gr_p).astype(int)

        # Rumble strips
        rs_p = np.where(spd_lim >= 55, 0.65,
                        np.where(rural == 1, 0.30, 0.15))
        df['HAS_RUMBLE_STRIPS'] = (self.rng.random(n) < rs_p).astype(int)

        # Speed camera within 1 mile
        sc_p = np.where(spd_lim >= 55, 0.10, 0.20)
        df['SPEED_CAMERA_WITHIN_1MI'] = (self.rng.random(n) < sc_p).astype(int)

        # Road quality index [1–5]: urban roads slightly better maintained
        quality = self.rng.normal(3.2, 0.8, n)
        quality = np.where(rural == 1, quality - 0.3, quality)
        df['ROAD_QUALITY_INDEX'] = np.round(np.clip(quality, 1, 5), 1)

        return df

    def _add_crash_probability_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a synthetic crash probability label using a weighted logistic
        combination of the key risk factors.  This is used for model training /
        validation when no real crash outcome is available.

        Higher probability = higher crash risk.
        """
        score = np.zeros(len(df))

        # Temporal
        if 'IS_NIGHT' in df.columns:
            score += df['IS_NIGHT'].fillna(0) * 0.15
        if 'IS_RUSH_HOUR' in df.columns:
            score += df['IS_RUSH_HOUR'].fillna(0) * 0.10

        # Traffic
        if 'TRAFFIC_DENSITY_INDEX' in df.columns:
            score += (df['TRAFFIC_DENSITY_INDEX'] - 1) / 4 * 0.12
        if 'TAILGATING_DETECTED_NEARBY' in df.columns:
            score += df['TAILGATING_DETECTED_NEARBY'] * 0.08
        if 'SPEED_VARIANCE_NEARBY_MPH' in df.columns:
            score += np.clip(df['SPEED_VARIANCE_NEARBY_MPH'] / 20, 0, 1) * 0.07

        # Work zone
        if 'WORK_ZONE_PRESENT' in df.columns:
            score += df['WORK_ZONE_PRESENT'] * 0.10
        if 'WORK_ZONE_WORKERS_PRESENT' in df.columns:
            score += df['WORK_ZONE_WORKERS_PRESENT'] * 0.08

        # Geometry
        if 'HAS_HORIZONTAL_CURVE' in df.columns:
            score += df['HAS_HORIZONTAL_CURVE'] * 0.07
        if 'ROAD_GRADE_PERCENT' in df.columns:
            score += np.clip(df['ROAD_GRADE_PERCENT'] / 15, 0, 1) * 0.05
        if 'LANE_WIDTH_FT' in df.columns:
            score += np.clip((12 - df['LANE_WIDTH_FT']) / 3, 0, 1) * 0.06

        # Environment
        if 'ADVERSE_WEATHER' in df.columns:
            score += df['ADVERSE_WEATHER'].fillna(0) * 0.08
        if 'BLACK_ICE_RISK' in df.columns:
            score += df['BLACK_ICE_RISK'] * 0.12
        if 'GLARE_CONDITIONS' in df.columns:
            score += df['GLARE_CONDITIONS'] * 0.05

        # Driver state
        if 'DUI_RISK_INDEX' in df.columns:
            score += df['DUI_RISK_INDEX'] * 0.15
        if 'FATIGUE_RISK_INDEX' in df.columns:
            score += df['FATIGUE_RISK_INDEX'] * 0.10
        if 'DISTRACTED_DRIVING_RISK' in df.columns:
            score += df['DISTRACTED_DRIVING_RISK'] * 0.08

        # Area
        if 'SCHOOL_HOURS_ACTIVE' in df.columns:
            score += df['SCHOOL_HOURS_ACTIVE'] * 0.05

        # Infrastructure deficiencies
        if 'LANE_MARKINGS_VISIBLE' in df.columns:
            score += (1 - df['LANE_MARKINGS_VISIBLE']) * 0.04
        if 'ROAD_QUALITY_INDEX' in df.columns:
            score += np.clip((5 - df['ROAD_QUALITY_INDEX']) / 4, 0, 1) * 0.04

        # Normalise to [0, 1] via logistic
        def logistic(x):
            return 1 / (1 + np.exp(-5 * (x - 0.5)))

        df['CRASH_PROBABILITY'] = np.round(logistic(score), 4)
        df['SAFETY_SCORE'] = np.round(100 * (1 - df['CRASH_PROBABILITY']), 2)

        return df

    # ------------------------------------------------------------------
    # Base condition generator for standalone synthesis
    # ------------------------------------------------------------------

    def _generate_base_conditions(self, n: int, scenario_type: str) -> Dict:
        """Generate base CRSS-like conditions for standalone scenario generation."""

        scenarios_map = {
            'rush_hour': dict(
                HOUR=self.rng.choice([7, 8, 9, 16, 17, 18], n),
                DAY_WEEK=self.rng.choice([2, 3, 4, 5, 6], n),  # weekday
                WEATHER=self.rng.choice([1, 1, 1, 2], n, p=[0.75, 0.10, 0.08, 0.07]),
                LGT_COND=self.rng.choice([1, 2], n, p=[0.85, 0.15]),
            ),
            'night_dui': dict(
                HOUR=self.rng.choice([21, 22, 23, 0, 1, 2], n),
                DAY_WEEK=self.rng.choice([1, 7, 6], n),  # weekend nights
                WEATHER=self.rng.choice([1, 2], n, p=[0.80, 0.20]),
                LGT_COND=self.rng.choice([2, 3], n, p=[0.60, 0.40]),  # dark
            ),
            'work_zone': dict(
                HOUR=self.rng.choice(list(range(6, 20)), n),
                DAY_WEEK=self.rng.choice([2, 3, 4, 5, 6], n),
                WEATHER=self.rng.choice([1, 2], n, p=[0.85, 0.15]),
                LGT_COND=self.rng.choice([1, 2], n, p=[0.90, 0.10]),
            ),
            'bad_weather': dict(
                HOUR=self.rng.choice(list(range(0, 24)), n),
                DAY_WEEK=self.rng.integers(1, 8, n),
                WEATHER=self.rng.choice([2, 3, 4, 5], n),  # adverse
                LGT_COND=self.rng.choice([1, 2, 4], n, p=[0.5, 0.35, 0.15]),
            ),
            'aggressive_traffic': dict(
                HOUR=self.rng.choice([7, 8, 9, 16, 17, 18, 19], n),
                DAY_WEEK=self.rng.choice([2, 3, 4, 5, 6], n),
                WEATHER=self.rng.choice([1, 2], n, p=[0.90, 0.10]),
                LGT_COND=self.rng.choice([1, 2], n, p=[0.85, 0.15]),
            ),
            'school_zone': dict(
                HOUR=self.rng.choice([7, 8, 14, 15, 16], n),
                DAY_WEEK=self.rng.choice([2, 3, 4, 5, 6], n),
                WEATHER=self.rng.choice([1, 2], n, p=[0.85, 0.15]),
                LGT_COND=self.rng.choice([1], n),
            ),
            'construction_zone': dict(
                HOUR=self.rng.choice(list(range(6, 20)), n),
                DAY_WEEK=self.rng.choice([2, 3, 4, 5, 6], n),
                WEATHER=self.rng.choice([1, 2, 3], n, p=[0.70, 0.20, 0.10]),
                LGT_COND=self.rng.choice([1, 2], n, p=[0.80, 0.20]),
            ),
            'narrow_road': dict(
                HOUR=self.rng.choice(list(range(0, 24)), n),
                DAY_WEEK=self.rng.integers(1, 8, n),
                WEATHER=self.rng.choice([1, 2, 3], n, p=[0.70, 0.20, 0.10]),
                LGT_COND=self.rng.choice([1, 2, 3], n, p=[0.55, 0.30, 0.15]),
            ),
            'black_ice': dict(
                HOUR=self.rng.choice([5, 6, 7, 20, 21, 22, 23, 0, 1], n),
                DAY_WEEK=self.rng.integers(1, 8, n),
                WEATHER=self.rng.choice([3, 4, 7], n),   # snow/ice
                LGT_COND=self.rng.choice([1, 2, 3], n, p=[0.40, 0.40, 0.20]),
            ),
        }

        if scenario_type in scenarios_map:
            base = scenarios_map[scenario_type]
        else:  # 'mixed'
            base = dict(
                HOUR=self.rng.integers(0, 24, n),
                DAY_WEEK=self.rng.integers(1, 8, n),
                WEATHER=self.rng.choice([1, 1, 1, 2, 3, 4, 5], n,
                                        p=[0.60, 0.10, 0.05, 0.10, 0.08, 0.04, 0.03]),
                LGT_COND=self.rng.choice([1, 2, 3, 4], n, p=[0.55, 0.25, 0.12, 0.08]),
            )

        # Common fields
        base['MONTH'] = self.rng.integers(1, 13, n)
        base['SPD_LIM'] = self.rng.choice([25, 30, 35, 45, 55, 65, 70], n,
                                           p=[0.10, 0.15, 0.20, 0.20, 0.20, 0.10, 0.05])
        base['IS_RURAL'] = self.rng.choice([0, 1], n, p=[0.65, 0.35])
        base['IS_NIGHT'] = ((base['HOUR'] >= 20) | (base['HOUR'] <= 5)).astype(int)
        base['IS_RUSH_HOUR'] = np.isin(base['HOUR'], [7, 8, 9, 16, 17, 18]).astype(int)
        base['IS_WEEKEND'] = np.isin(base['DAY_WEEK'], [1, 7]).astype(int)
        base['ADVERSE_WEATHER'] = (base['WEATHER'] > 1).astype(int)
        base['POOR_LIGHTING'] = (base['LGT_COND'] > 1).astype(int)

        return base


# ---------------------------------------------------------------------------
# Crash factor risk weight table (for explainability / reporting)
# ---------------------------------------------------------------------------

CRASH_RISK_FACTOR_WEIGHTS = {
    # Factor : (relative weight, description)
    'DUI_risk_late_night_weekend':        (0.28, 'Impaired driving — late night + weekend + bar density'),
    'black_ice_temp_moisture':            (0.24, 'Black ice — temperature < 35°F + precipitation (winter)'),
    'work_zone_workers_present':          (0.20, 'Active construction zone with workers on roadway'),
    'rush_hour_traffic_congestion':       (0.18, 'Peak commute — dense traffic + tailgating + lane changes'),
    'aggressive_nearby_driver':           (0.16, 'Aggressive surrounding traffic — speeding + lane changes'),
    'narrow_lane_horizontal_curve':       (0.15, 'Narrow lane (<11 ft) on curve — reduced margin for error'),
    'driver_fatigue_early_morning':       (0.14, 'Driver fatigue — 2–6 AM (circadian low)'),
    'distracted_driving':                 (0.13, 'Distracted driver (phone / in-cabin)'),
    'school_zone_active_hours':           (0.12, 'School zone during arrival/dismissal — vulnerable pedestrians'),
    'adverse_weather_poor_road':          (0.12, 'Rain/snow on low-quality road surface'),
    'poor_lighting_no_guardrail':         (0.10, 'Dark road without guardrail — run-off risk'),
    'sun_glare_sunrise_sunset':           (0.09, 'Sun glare at sunrise/sunset — reduced forward visibility'),
    'steep_grade_wet_road':               (0.09, 'Steep downhill grade on wet or icy surface'),
    'high_speed_variance_nearby':         (0.08, 'High speed variance in surrounding traffic — erratic flow'),
    'event_nearby_crowd':                 (0.07, 'Large event nearby — increased volume + unfamiliar drivers'),
    'faded_lane_markings_night':          (0.06, 'Faded/missing lane markings — reduced lane-keeping cue'),
}


if __name__ == '__main__':
    gen = ContextualFeatureGenerator(random_seed=42)

    # Show feature catalog
    catalog = gen.scenario_feature_catalog()
    print(f"\n{'='*70}")
    print(f"  SafeDriver-IQ  |  Contextual Feature Catalog  ({len(catalog)} features)")
    print(f"{'='*70}")
    for cat, grp in catalog.groupby('category'):
        print(f"\n  [{cat.upper()}]")
        for _, row in grp.iterrows():
            print(f"    {row['feature']:<45}  {row['description'][:55]}")

    # Generate mixed scenarios
    df = gen.generate_synthetic_scenarios(n=5000, scenario_type='mixed')
    print(f"\n\nGenerated {len(df)} synthetic scenarios  ({len(df.columns)} features)")
    print(f"\nCrash probability distribution:")
    print(df['CRASH_PROBABILITY'].describe().round(3))
    print(f"\nSafety score distribution:")
    print(df['SAFETY_SCORE'].describe().round(2))

    # Top risk drivers
    print(f"\n\nTop 5 crash risk factors (by weight):")
    for k, (w, desc) in sorted(CRASH_RISK_FACTOR_WEIGHTS.items(), key=lambda x: -x[1][0])[:5]:
        print(f"  [{w:.2f}]  {desc}")
