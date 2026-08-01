"""
Pytest configuration and fixtures for test suite
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_accident_data():
    """Create sample accident data for testing"""
    return pd.DataFrame({
        'CASENUM': [1001, 1002, 1003, 1004, 1005],
        'PSU': [101, 102, 103, 104, 105],
        'YEAR': [2020, 2020, 2021, 2021, 2022],
        'MONTH': [6, 7, 8, 9, 10],
        'DAY': [15, 20, 25, 10, 5],
        'HOUR': [14, 18, 22, 8, 12],
        'MINUTE': [30, 45, 15, 0, 30],
        'LATITUDE': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
        'LONGITUD': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740],
        'WEATHER': [1, 2, 1, 3, 1],
        'LGT_COND': [1, 2, 3, 1, 2],
        'ROUTE': [1, 2, 1, 3, 2],
        'TRAF_WAY': [1, 2, 1, 2, 1],
        'RUR_URB': [1, 1, 2, 1, 2],
        'FUNC_SYS': [1, 2, 3, 1, 2],
        'HARM_EV': [12, 12, 12, 12, 12],
        'MAN_COLL': [1, 2, 1, 3, 2],
        'RELJCT1': [1, 2, 1, 3, 2],
        'RELJCT2': [1, 2, 1, 3, 2],
        'TYP_INT': [1, 2, 1, 3, 2],
        'REL_ROAD': [1, 2, 1, 3, 2],
        'FATALS': [0, 0, 1, 0, 0],
        'DRUNK_DR': [0, 1, 0, 0, 0]
    })


@pytest.fixture
def sample_person_data():
    """Create sample person data for testing"""
    return pd.DataFrame({
        'CASENUM': [1001, 1001, 1002, 1003, 1003],
        'VEH_NO': [1, 2, 1, 1, 2],
        'PER_NO': [1, 1, 1, 1, 1],
        'PER_TYP': [1, 5, 1, 5, 1],  # 1=driver, 5=pedestrian
        'AGE': [35, 42, 28, 65, 30],
        'SEX': [1, 2, 1, 2, 1],
        'INJ_SEV': [0, 4, 0, 3, 0],
        'SEAT_POS': [11, 99, 11, 99, 11],
        'REST_USE': [3, 99, 3, 99, 3],
        'AIR_BAG': [1, 99, 1, 99, 1],
        'DRINKING': [0, 99, 1, 99, 0]
    })


@pytest.fixture
def sample_vehicle_data():
    """Create sample vehicle data for testing"""
    return pd.DataFrame({
        'CASENUM': [1001, 1001, 1002, 1003, 1003],
        'VEH_NO': [1, 2, 1, 1, 2],
        'VEH_YEAR': [2018, 2015, 2020, 2010, 2019],
        'MAKE': [20, 12, 37, 49, 20],
        'BODY_TYP': [1, 2, 1, 3, 1],
        'SPEED_REL': [0, 1, 0, 2, 0],
        'DR_DRINK': [0, 1, 0, 0, 0],
        'PREV_ACC': [1, 2, 1, 3, 1],
        'PREV_SUS': [0, 1, 0, 0, 0],
        'PREV_DWI': [0, 1, 0, 0, 0],
        'PREV_SPD': [0, 0, 1, 0, 0]
    })


@pytest.fixture
def sample_pbtype_data():
    """Create sample pedestrian/bicyclist data for testing"""
    return pd.DataFrame({
        'CASENUM': [1001, 1003],
        'VEH_NO': [2, 1],
        'PER_NO': [1, 1],
        'PBPTYPE': [1, 2],  # 1=pedestrian, 2=bicyclist
        'PBAGE': [42, 65],
        'PBSEX': [2, 2],
        'PB_INJSEV': [4, 3],
        'LOCATION': [1, 2]
    })


@pytest.fixture
def sample_features_data(sample_accident_data):
    """Create sample data with engineered features"""
    df = sample_accident_data.copy()
    # Add some engineered features
    df['IS_NIGHT'] = (df['HOUR'] >= 18) | (df['HOUR'] <= 6)
    df['IS_WEEKEND'] = 0
    df['IS_RUSH_HOUR'] = 0
    df['POOR_LIGHTING'] = (df['LGT_COND'] > 1).astype(int)
    df['ADVERSE_WEATHER'] = (df['WEATHER'] > 1).astype(int)
    return df


@pytest.fixture
def sample_model_data():
    """Create sample X, y data for model training"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create year directories
    for year in [2020, 2021]:
        year_dir = data_dir / str(year)
        year_dir.mkdir()
    
    return data_dir
