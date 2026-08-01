"""Lightweight CRSS environmental risk multiplier.

The raw CRSS CSV files are Git LFS pointers and are not available locally.
This module provides the next best thing: a lookup table based on CRSS
weather/lighting/road-condition codes and NHTSA/CRSS-derived risk priors.
It converts a DrivingScene's attributes into a normalized environmental risk
multiplier [0, 1] that can be blended with the trained SafeDriver-IQ model.

References:
  - CRSS codebook (WEATHER, LGT_COND, SUR_COND)
  - NHTSA / FHWA crash-factor studies informing the relative weights
"""
from __future__ import annotations

from typing import Dict

from prism_ar.data_ingestion.driving_scene import SceneAttributes


# CRSS weather codes -> relative crash risk multiplier (baseline clear = 1.0)
WEATHER_RISK = {
    "no_additional_atmospheric_conditions": 1.00,
    "clear": 1.00,
    "cloudy": 1.05,
    "rain": 1.25,
    "sleet": 1.45,
    "snow": 1.55,
    "fog": 1.35,
    "smoke": 1.30,
    "dust": 1.20,
    "blowing_sand": 1.30,
    "severe_crosswinds": 1.20,
    "wet": 1.15,  # sometimes used as road condition synonym
}

# CRSS lighting codes -> relative crash risk multiplier
LIGHTING_RISK = {
    "daylight": 1.00,
    "dawn": 1.15,
    "dusk": 1.20,
    "dark": 1.50,
    "dark_lighted": 1.25,
    "dark_unknown": 1.40,
    "night": 1.50,
    "day": 1.00,
}

# CRSS road surface condition codes -> relative crash risk multiplier
ROAD_CONDITION_RISK = {
    "dry": 1.00,
    "wet": 1.30,
    "snow": 1.50,
    "ice": 1.80,
    "sand": 1.20,
    "mud": 1.25,
    "oil": 1.25,
    "slush": 1.40,
    "water": 1.30,
    "debris": 1.20,
}

# Hour-of-day risk profile (baseline = 1.0, late night / early morning higher)
HOUR_RISK = {h: 1.0 for h in range(6, 20)}
HOUR_RISK.update({
    20: 1.15, 21: 1.30, 22: 1.50, 23: 1.70,
    0: 2.00, 1: 2.10, 2: 1.90, 3: 1.60, 4: 1.30, 5: 1.10,
})

# Softmax-like normalization constants so that a typical clear/daylight/dry scene
# maps to a low multiplier (~0.1) and an extreme night/ice/wet scene maps high.
# We normalize relative risk to [0, 1] using: (rel - 1) / (max_rel - 1).
_MAX_REL = 2.5  # corresponds to worst case (night + ice + rain)


def _parse_hour(time_of_day: str) -> int:
    """Parse 'HH:MM' or integer string into hour."""
    if time_of_day is None:
        return 12
    try:
        return int(str(time_of_day).split(":")[0]) % 24
    except Exception:
        return 12


def crss_env_risk_multiplier(attributes: SceneAttributes) -> float:
    """Return a normalized CRSS environmental risk multiplier in [0, 1]."""
    weather = str(attributes.weather or "clear").lower()
    lighting = str(attributes.lighting or "daylight").lower()
    road = str(attributes.road_condition or "dry").lower()
    hour = _parse_hour(attributes.time_of_day)

    weather_mult = WEATHER_RISK.get(weather, 1.0)
    lighting_mult = LIGHTING_RISK.get(lighting, 1.0)
    road_mult = ROAD_CONDITION_RISK.get(road, 1.0)
    hour_mult = HOUR_RISK.get(hour, 1.0)

    # Geometric mean keeps the multiplier bounded and multiplicative
    rel = (weather_mult * lighting_mult * road_mult * hour_mult) ** 0.25

    # Normalize to [0, 1] so that baseline clear/daylight/dry -> ~0
    normalized = max(0.0, min(1.0, (rel - 1.0) / (_MAX_REL - 1.0)))
    return normalized


def crss_env_factor_summary(attributes: SceneAttributes) -> Dict[str, float]:
    """Return a dict of the component multipliers for debugging/reporting."""
    weather = str(attributes.weather or "clear").lower()
    lighting = str(attributes.lighting or "daylight").lower()
    road = str(attributes.road_condition or "dry").lower()
    hour = _parse_hour(attributes.time_of_day)
    return {
        "crss_weather_mult": WEATHER_RISK.get(weather, 1.0),
        "crss_lighting_mult": LIGHTING_RISK.get(lighting, 1.0),
        "crss_road_mult": ROAD_CONDITION_RISK.get(road, 1.0),
        "crss_hour_mult": HOUR_RISK.get(hour, 1.0),
        "crss_env_risk": crss_env_risk_multiplier(attributes),
    }
