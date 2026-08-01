"""Check actual feature distributions in crash data."""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from src.data_loader import CRSSDataLoader
from src.feature_engineering import create_safety_features

# Load VRU crashes
print("Loading data...")
loader = CRSSDataLoader()
data = loader.load_complete_dataset()
accidents = data['accident']
pbtype = data['pbtype']
vru_crashes = accidents[accidents['CASENUM'].isin(pbtype['CASENUM'])].copy()
vru_crashes = create_safety_features(vru_crashes)

print(f"\nTotal VRU crashes: {len(vru_crashes)}")

# Check SPEED_REL
if 'SPEED_REL' in vru_crashes.columns:
    print('\n=== SPEED_REL in actual VRU crashes ===')
    speed_dist = vru_crashes['SPEED_REL'].value_counts().sort_index()
    for speed, count in speed_dist.head(10).items():
        pct = count / len(vru_crashes) * 100
        print(f'  {speed}: {count:6,} ({pct:5.1f}%)')
    print(f'Mean: {vru_crashes["SPEED_REL"].mean():.2f}')
else:
    print('\nSPEED_REL not found - creating it')
    # Create SPEED_REL if missing
    vru_crashes['SPEED_REL'] = np.random.randint(1, 6, len(vru_crashes))

# Check ROAD_COND  
if 'ROAD_COND' in vru_crashes.columns:
    print('\n=== ROAD_COND in actual VRU crashes ===')
    road_dist = vru_crashes['ROAD_COND'].value_counts().sort_index()
    for road, count in road_dist.items():
        pct = count / len(vru_crashes) * 100
        road_name = {1: 'Dry', 2: 'Wet', 3: 'Snow', 4: 'Ice'}.get(road, f'Unknown({road})')
        print(f'  {road} ({road_name:7s}): {count:6,} ({pct:5.1f}%)')
    print(f'Mean: {vru_crashes["ROAD_COND"].mean():.2f}')
else:
    print('\nROAD_COND not found')

print('\n=== KEY INSIGHT ===')
print('If crash data shows mostly low speeds (1-2) AND good roads (1-2),')
print('then creating "safe" samples with the same distribution creates NO contrast!')
print('The model cannot learn what makes crashes different from safe driving.')
