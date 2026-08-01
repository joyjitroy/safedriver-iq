"""
Data Loading Module for CRSS Dataset

This module handles loading and basic parsing of CRSS crash data files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CRSSDataLoader:
    """
    Loads and combines CRSS data files from multiple years.
    
    Attributes:
        data_dir: Path to CRSS_Data directory
        years: List of years to load (2016-2023)
    """
    
    def __init__(self, data_dir: str = "CRSS_Data", years: Optional[List[int]] = None):
        """
        Initialize CRSS data loader.
        
        Args:
            data_dir: Path to directory containing CRSS data
            years: List of years to load (default: 2016-2023)
        """
        self.data_dir = Path(data_dir)
        self.years = years or list(range(2016, 2024))
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_accident_data(self, year: int) -> pd.DataFrame:
        """
        Load ACCIDENT.csv for a specific year.
        
        Args:
            year: Year to load (e.g., 2020)
            
        Returns:
            DataFrame with accident-level data
        """
        # Try different file path patterns (NHTSA changed structure over years)
        possible_paths = [
            self.data_dir / str(year) / f"CRSS{year}CSV" / "accident.csv",
            self.data_dir / str(year) / "accident.csv",
            self.data_dir / str(year) / "ACCIDENT.csv",
            self.data_dir / str(year) / "ACCIDENT.CSV",
            self.data_dir / str(year) / "accident.CSV",
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            logger.warning(f"Accident file not found for year {year}")
            return pd.DataFrame()
        
        logger.info(f"Loading accident data for {year} from {file_path.name}")
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                df['YEAR'] = year
                return df
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Failed to load {file_path} with any encoding")
        return pd.DataFrame()
    
    def load_vehicle_data(self, year: int) -> pd.DataFrame:
        """
        Load VEHICLE.csv for a specific year.
        
        Args:
            year: Year to load
            
        Returns:
            DataFrame with vehicle-level data
        """
        possible_paths = [
            self.data_dir / str(year) / f"CRSS{year}CSV" / "vehicle.csv",
            self.data_dir / str(year) / "vehicle.csv",
            self.data_dir / str(year) / "VEHICLE.csv",
            self.data_dir / str(year) / "VEHICLE.CSV",
            self.data_dir / str(year) / "vehicle.CSV",
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            logger.warning(f"Vehicle file not found for year {year}")
            return pd.DataFrame()
        
        logger.info(f"Loading vehicle data for {year} from {file_path.name}")
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                df['YEAR'] = year
                return df
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Failed to load {file_path} with any encoding")
        return pd.DataFrame()
    
    def load_person_data(self, year: int) -> pd.DataFrame:
        """
        Load PERSON.csv for a specific year.
        
        Args:
            year: Year to load
            
        Returns:
            DataFrame with person-level data
        """
        possible_paths = [
            self.data_dir / str(year) / f"CRSS{year}CSV" / "person.csv",
            self.data_dir / str(year) / "person.csv",
            self.data_dir / str(year) / "PERSON.csv",
            self.data_dir / str(year) / "PERSON.CSV",
            self.data_dir / str(year) / "person.CSV",
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            logger.warning(f"Person file not found for year {year}")
            return pd.DataFrame()
        
        logger.info(f"Loading person data for {year} from {file_path.name}")
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                df['YEAR'] = year
                return df
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Failed to load {file_path} with any encoding")
        return pd.DataFrame()
    
    def load_pbtype_data(self, year: int) -> pd.DataFrame:
        """
        Load PBTYPE.csv (Pedestrian/Bicyclist Type) for a specific year.
        
        Args:
            year: Year to load
            
        Returns:
            DataFrame with pedestrian/bicyclist crash typing data
        """
        possible_paths = [
            self.data_dir / str(year) / f"CRSS{year}CSV" / "pbtype.csv",
            self.data_dir / str(year) / "pbtype.csv",
            self.data_dir / str(year) / "PBTYPE.csv",
            self.data_dir / str(year) / "PBTYPE.CSV",
            self.data_dir / str(year) / "pbtype.CSV",
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            logger.warning(f"PBTYPE file not found for year {year}")
            return pd.DataFrame()
        
        logger.info(f"Loading PBTYPE data for {year} from {file_path.name}")
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                df['YEAR'] = year
                return df
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Failed to load {file_path} with any encoding")
        return pd.DataFrame()
    
    def load_all_years(self, file_type: str = "accident") -> pd.DataFrame:
        """
        Load and combine data from all specified years.
        
        Args:
            file_type: Type of file to load ('accident', 'vehicle', 'person', 'pbtype')
            
        Returns:
            Combined DataFrame from all years
        """
        load_functions = {
            "accident": self.load_accident_data,
            "vehicle": self.load_vehicle_data,
            "person": self.load_person_data,
            "pbtype": self.load_pbtype_data,
        }
        
        if file_type not in load_functions:
            raise ValueError(f"Invalid file_type: {file_type}. Must be one of {list(load_functions.keys())}")
        
        load_func = load_functions[file_type]
        
        dfs = []
        for year in self.years:
            df = load_func(year)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            logger.warning(f"No data loaded for file_type: {file_type}")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {file_type} data: {len(combined_df):,} records from {len(dfs)} years")
        
        return combined_df
    
    def load_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CRSS data files (accident, vehicle, person, pbtype).
        
        Returns:
            Dictionary with keys: 'accident', 'vehicle', 'person', 'pbtype'
        """
        logger.info("Loading complete CRSS dataset...")
        
        datasets = {
            'accident': self.load_all_years('accident'),
            'vehicle': self.load_all_years('vehicle'),
            'person': self.load_all_years('person'),
            'pbtype': self.load_all_years('pbtype'),
        }
        
        logger.info("Dataset loading complete")
        for name, df in datasets.items():
            logger.info(f"  {name}: {len(df):,} records")
        
        return datasets
    
    def get_vru_crashes(self, datasets: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Extract VRU (Vulnerable Road User) crashes.
        
        VRUs include: pedestrians, bicyclists, and work zone crashes.
        
        Args:
            datasets: Pre-loaded datasets (if None, will load)
            
        Returns:
            DataFrame with VRU crashes
        """
        if datasets is None:
            datasets = self.load_complete_dataset()
        
        person_df = datasets['person']
        
        # Filter for pedestrians (PER_TYP == 5) and bicyclists (PER_TYP == 6)
        # Note: Actual column names may vary - check CRSS documentation
        vru_persons = person_df[person_df['PER_TYP'].isin([5, 6])]
        
        # Get unique case numbers
        vru_case_ids = vru_persons['CASENUM'].unique()
        
        logger.info(f"Found {len(vru_case_ids):,} VRU crashes")
        
        return vru_case_ids


if __name__ == "__main__":
    # Example usage
    loader = CRSSDataLoader()
    datasets = loader.load_complete_dataset()
    
    print("\nDataset Summary:")
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Records: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Years: {sorted(df['YEAR'].unique())}")
