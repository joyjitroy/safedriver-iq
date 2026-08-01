"""
Tests for data_loader.py module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_loader import CRSSDataLoader


class TestCRSSDataLoader:
    """Test suite for CRSSDataLoader class"""
    
    def test_initialization(self, temp_data_dir):
        """Test loader initialization"""
        loader = CRSSDataLoader(data_dir=str(temp_data_dir), years=[2020, 2021])
        assert loader.data_dir == Path(temp_data_dir)
        assert loader.years == [2020, 2021]
    
    def test_initialization_invalid_directory(self):
        """Test initialization with non-existent directory"""
        with pytest.raises(FileNotFoundError):
            CRSSDataLoader(data_dir='/nonexistent/path')
    
    @patch('data_loader.pd.read_csv')
    def test_load_year_data(self, mock_read_csv, temp_data_dir, sample_accident_data):
        """Test loading data for a single year"""
        # Create mock file
        accident_file = temp_data_dir / "2020" / "accident.csv"
        accident_file.parent.mkdir(exist_ok=True)
        accident_file.touch()
        
        # Mock the read_csv function
        mock_read_csv.return_value = sample_accident_data
        
        loader = CRSSDataLoader(data_dir=str(temp_data_dir), years=[2020])
        data = loader.load_accident_data(2020)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'YEAR' in data.columns
        mock_read_csv.assert_called_once()
    
    def test_load_year_data_missing_file(self, temp_data_dir):
        """Test loading data with missing file"""
        loader = CRSSDataLoader(data_dir=str(temp_data_dir), years=[2020])
        
        # Should return empty DataFrame for missing file
        data = loader.load_accident_data(2020)
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 0
    
    @patch.object(CRSSDataLoader, 'load_accident_data')
    @patch.object(CRSSDataLoader, 'load_person_data')
    @patch.object(CRSSDataLoader, 'load_vehicle_data')
    @patch.object(CRSSDataLoader, 'load_pbtype_data')
    def test_load_complete_dataset(self, mock_pbtype, mock_vehicle, mock_person, mock_accident,
                                   temp_data_dir, sample_accident_data, sample_person_data,
                                   sample_vehicle_data, sample_pbtype_data):
        """Test loading complete dataset"""
        # Mock data loading for each table
        mock_accident.return_value = sample_accident_data
        mock_person.return_value = sample_person_data
        mock_vehicle.return_value = sample_vehicle_data
        mock_pbtype.return_value = sample_pbtype_data
        
        loader = CRSSDataLoader(data_dir=str(temp_data_dir), years=[2020])
        data = loader.load_complete_dataset()
        
        assert isinstance(data, dict)
        assert 'accident' in data
        assert 'person' in data
        assert 'vehicle' in data
        assert 'pbtype' in data
        assert all(isinstance(df, pd.DataFrame) for df in data.values())
    
    @patch.object(CRSSDataLoader, 'load_complete_dataset')
    def test_get_vru_crashes(self, mock_load_dataset, temp_data_dir, sample_person_data):
        """Test extracting VRU crash case numbers"""
        # Mock the dataset
        mock_load_dataset.return_value = {
            'person': sample_person_data
        }
        
        loader = CRSSDataLoader(data_dir=str(temp_data_dir), years=[2020])
        vru_cases = loader.get_vru_crashes()
        
        # Should return numpy array of case numbers where PER_TYP is 5 (pedestrian) or 6 (cyclist)
        assert isinstance(vru_cases, np.ndarray)
        assert len(vru_cases) > 0
        # Cases 1001 and 1003 have pedestrians
        assert 1001 in vru_cases
        assert 1003 in vru_cases
    
    @patch.object(CRSSDataLoader, 'load_complete_dataset')
    def test_get_vru_crashes_empty(self, mock_load_dataset, temp_data_dir):
        """Test VRU crash extraction with no VRU crashes"""
        # Mock dataset with no pedestrians/cyclists
        mock_load_dataset.return_value = {
            'person': pd.DataFrame({
                'CASENUM': [1001, 1002],
                'PER_TYP': [1, 1]  # All drivers
            })
        }
        
        loader = CRSSDataLoader(data_dir=str(temp_data_dir), years=[2020])
        vru_cases = loader.get_vru_crashes()
        
        assert isinstance(vru_cases, np.ndarray)
        assert len(vru_cases) == 0
    
    def test_year_range_detection(self, temp_data_dir):
        """Test automatic year range detection"""
        # Create additional year directories
        for year in [2022, 2023]:
            (temp_data_dir / str(year)).mkdir()
        
        loader = CRSSDataLoader(data_dir=str(temp_data_dir))
        assert 2020 in loader.years
        assert 2021 in loader.years
        assert 2022 in loader.years
        assert 2023 in loader.years
    
    @patch('data_loader.pd.read_csv')
    def test_data_concatenation(self, mock_read_csv, temp_data_dir):
        """Test that data from multiple years is concatenated correctly"""
        # Create sample data for different years
        data_2020 = pd.DataFrame({
            'CASENUM': [1001, 1002],
            'MONTH': [6, 7]
        })
        data_2021 = pd.DataFrame({
            'CASENUM': [2001, 2002],
            'MONTH': [8, 9]
        })
        
        mock_read_csv.side_effect = [data_2020, data_2021]
        
        # Create files
        for year in [2020, 2021]:
            file = temp_data_dir / str(year) / "accident.csv"
            file.parent.mkdir(exist_ok=True)
            file.touch()
        
        loader = CRSSDataLoader(data_dir=str(temp_data_dir), years=[2020, 2021])
        
        # Manually concatenate (simulating what happens in load_complete_dataset)
        all_data = []
        for year in [2020, 2021]:
            data = loader.load_accident_data(year)
            if len(data) > 0:
                all_data.append(data)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            assert len(combined) == 4  # 2 from 2020 + 2 from 2021
            assert 2020 in combined['YEAR'].values
            assert 2021 in combined['YEAR'].values
