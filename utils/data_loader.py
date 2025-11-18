"""
Data Loader Component
Handles MetaTrader CSV loading and preprocessing
"""

import pandas as pd
import numpy as np


class MetaTraderDataLoader:
    """Handles loading and basic preprocessing of MetaTrader CSV files"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
    
    def load(self) -> pd.DataFrame:
        """Load MetaTrader CSV and return clean DataFrame"""
        print("\n" + "="*60)
        print("LOADING METATRADER CSV")
        print("="*60)
        
        df = pd.read_csv(self.csv_path, sep='\t')
        df.columns = df.columns.str.replace('<', '').str.replace('>', '').str.strip()
        
        # Combine DATE and TIME into Datetime
        if 'DATE' in df.columns and 'TIME' in df.columns:
            df['Datetime'] = pd.to_datetime(
                df['DATE'].astype(str) + ' ' + df['TIME'].astype(str)
            )
        
        # Rename columns
        column_mapping = {
            'OPEN': 'Open', 
            'HIGH': 'High', 
            'LOW': 'Low', 
            'CLOSE': 'Close', 
            'TICKVOL': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Set index and select OHLCV
        df = df.set_index('Datetime')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Range: {df.index.min()} to {df.index.max()}")
        
        self.df = df
        return df
    
    def get_data(self) -> pd.DataFrame:
        """Return loaded data"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.df
