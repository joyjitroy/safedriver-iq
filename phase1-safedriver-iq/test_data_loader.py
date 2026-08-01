"""
Test script to verify CRSS data loading

Run this to verify that the data loader works correctly.
"""

import sys
sys.path.append('src')

from data_loader import CRSSDataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("\n" + "="*70)
    print("SafeDriver-IQ: Data Loader Test")
    print("="*70 + "\n")
    
    # Initialize loader
    print("Initializing CRSS Data Loader...")
    loader = CRSSDataLoader(data_dir='CRSS_Data', years=[2023])  # Test with one year first
    
    print("\nTest 1: Loading 2023 Accident Data")
    print("-" * 50)
    accident_2023 = loader.load_accident_data(2023)
    
    if not accident_2023.empty:
        print(f"✓ Successfully loaded {len(accident_2023):,} accident records")
        print(f"  Columns: {len(accident_2023.columns)}")
        print(f"  Sample columns: {accident_2023.columns[:10].tolist()}")
    else:
        print("✗ Failed to load accident data")
        return
    
    print("\nTest 2: Loading 2023 Person Data")
    print("-" * 50)
    person_2023 = loader.load_person_data(2023)
    
    if not person_2023.empty:
        print(f"✓ Successfully loaded {len(person_2023):,} person records")
        
        # Check for VRUs
        if 'PER_TYP' in person_2023.columns:
            vru_count = person_2023[person_2023['PER_TYP'].isin([5, 6])].shape[0]
            print(f"  VRU persons found: {vru_count:,}")
            print(f"    Pedestrians: {(person_2023['PER_TYP'] == 5).sum():,}")
            print(f"    Bicyclists: {(person_2023['PER_TYP'] == 6).sum():,}")
    else:
        print("✗ Failed to load person data")
    
    print("\nTest 3: Loading All Years")
    print("-" * 50)
    print("Loading data for years 2016-2023...")
    loader_full = CRSSDataLoader(data_dir='CRSS_Data', years=list(range(2016, 2024)))
    
    datasets = loader_full.load_complete_dataset()
    
    print("\nDataset Summary:")
    for name, df in datasets.items():
        if not df.empty:
            print(f"  {name.upper()}: {len(df):,} records, {len(df.columns)} columns")
    
    print("\n" + "="*70)
    print("Data Loader Test Complete!")
    print("="*70 + "\n")
    
    print("Next Steps:")
    print("1. Open Jupyter Notebook: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("2. Run the notebook to explore the data")
    print("3. Continue with feature engineering and modeling")
    print()

if __name__ == "__main__":
    main()
