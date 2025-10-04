"""
Data Handler - CSV Loading and Preprocessing
Handles loading and preprocessing of events.csv and covariates.csv files.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')


class DataHandler:
    """
    Handles loading and preprocessing of CSV data files for LOB intensity modeling.
    """
    
    def __init__(self):
        self.events_df = None
        self.covariates_df = None
        self.is_loaded = False
        
    def load_csv_files(self, events_path: str, covariates_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load events.csv and covariates.csv files.
        
        Args:
            events_path: Path to events.csv file
            covariates_path: Path to covariates.csv file
            
        Returns:
            Tuple of (events_df, covariates_df)
        """
        print(f"Loading events from: {events_path}")
        print(f"Loading covariates from: {covariates_path}")
        
        # Load events.csv
        try:
            self.events_df = pd.read_csv(events_path)
            print(f"Loaded {len(self.events_df)} events")
        except Exception as e:
            raise ValueError(f"Error loading events.csv: {e}")
        
        # Load covariates.csv
        try:
            self.covariates_df = pd.read_csv(covariates_path)
            print(f"Loaded {len(self.covariates_df)} covariate points")
        except Exception as e:
            raise ValueError(f"Error loading covariates.csv: {e}")
        
        # Validate data
        self._validate_data()
        
        # Preprocess data
        self._preprocess_data()
        
        self.is_loaded = True
        
        return self.events_df, self.covariates_df
    
    def _validate_data(self):
        """Validate loaded data has required columns and formats."""
        
        # Validate events.csv
        required_event_cols = ['time', 'event_type']
        missing_cols = set(required_event_cols) - set(self.events_df.columns)
        if missing_cols:
            raise ValueError(f"events.csv missing required columns: {missing_cols}")
        
        # Validate event types
        valid_event_types = ['market', 'limit', 'cancel']
        invalid_types = set(self.events_df['event_type'].unique()) - set(valid_event_types)
        if invalid_types:
            print(f"Warning: Found invalid event types: {invalid_types}")
        
        # Validate covariates.csv
        required_cov_cols = ['time', 'S', 'q1', 'Q10']
        missing_cols = set(required_cov_cols) - set(self.covariates_df.columns)
        if missing_cols:
            raise ValueError(f"covariates.csv missing required columns: {missing_cols}")
        
        # Check for negative values
        if (self.covariates_df[['S', 'q1', 'Q10']] < 0).any().any():
            print("Warning: Found negative values in covariates")
        
        print("Data validation completed successfully")
    
    def _preprocess_data(self):
        """Preprocess loaded data."""
        
        # Sort by time
        self.events_df = self.events_df.sort_values('time').reset_index(drop=True)
        self.covariates_df = self.covariates_df.sort_values('time').reset_index(drop=True)
        
        # Ensure time columns are numeric
        self.events_df['time'] = pd.to_numeric(self.events_df['time'], errors='coerce')
        self.covariates_df['time'] = pd.to_numeric(self.covariates_df['time'], errors='coerce')
        
        # Remove any rows with NaN times
        self.events_df = self.events_df.dropna(subset=['time'])
        self.covariates_df = self.covariates_df.dropna(subset=['time'])
        
        # Remove events with negative times (they're outside our covariate range)
        self.events_df = self.events_df[self.events_df['time'] >= 0]
        
        # Add price column to events if missing (for placement model)
        if 'price' not in self.events_df.columns:
            print("Adding synthetic price column to events...")
            self.events_df['price'] = self._generate_synthetic_prices()
        
        # Add side column to events if missing
        if 'side' not in self.events_df.columns:
            print("Adding synthetic side column to events...")
            self.events_df['side'] = np.random.choice(['bid', 'ask'], len(self.events_df))
        
        print("Data preprocessing completed")
    
    def _generate_synthetic_prices(self) -> np.ndarray:
        """Generate synthetic prices for events (when not provided)."""
        prices = []
        
        for _, event in self.events_df.iterrows():
            time = event['time']
            event_type = event['event_type']
            
            # Find closest covariate data
            cov_idx = self.covariates_df['time'].searchsorted(time, side='right') - 1
            cov_idx = max(0, min(cov_idx, len(self.covariates_df) - 1))
            
            spread = self.covariates_df.iloc[cov_idx]['S']
            mid_price = 100.0  # Assume mid price
            
            if event_type == 'market':
                # Market orders at best bid/ask
                side = np.random.choice(['bid', 'ask'])
                if side == 'bid':
                    price = mid_price - spread / 2
                else:
                    price = mid_price + spread / 2
            else:  # limit or cancel
                # Limit orders/cancels at various prices
                side = np.random.choice(['bid', 'ask'])
                if side == 'bid':
                    price = mid_price - spread / 2 - np.random.exponential(scale=0.01)
                else:
                    price = mid_price + spread / 2 + np.random.exponential(scale=0.01)
            
            prices.append(price)
        
        return np.array(prices)
    
    def get_data_summary(self) -> dict:
        """Get summary statistics of loaded data."""
        if not self.is_loaded:
            raise ValueError("No data loaded")
        
        summary = {
            'n_events': len(self.events_df),
            'n_covariates': len(self.covariates_df),
            'time_range': (self.events_df['time'].min(), self.events_df['time'].max()),
            'event_types': self.events_df['event_type'].value_counts().to_dict(),
            'covariate_stats': {
                'S': {'mean': self.covariates_df['S'].mean(), 'std': self.covariates_df['S'].std()},
                'q1': {'mean': self.covariates_df['q1'].mean(), 'std': self.covariates_df['q1'].std()},
                'Q10': {'mean': self.covariates_df['Q10'].mean(), 'std': self.covariates_df['Q10'].std()}
            }
        }
        
        return summary
    
    def create_sample_data(self, T: float = 100.0, n_events: int = 50, 
                          n_covariates: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create sample data for testing.
        
        Args:
            T: Total time period
            n_events: Number of events to generate
            n_covariates: Number of covariate change points
            
        Returns:
            Tuple of (events_df, covariates_df)
        """
        print("Creating sample data...")
        
        # Generate covariate data
        cov_times = np.sort(np.random.uniform(0, T, n_covariates))
        cov_times = np.concatenate(([0], cov_times, [T]))
        
        covariates_data = []
        for i in range(len(cov_times) - 1):
            covariates_data.append({
                'time': cov_times[i],
                'S': max(0.001, np.random.lognormal(mean=-4.5, sigma=0.5)),
                'q1': max(1, int(np.random.lognormal(mean=4.5, sigma=0.8))),
                'Q10': max(1, int(np.random.lognormal(mean=6.5, sigma=0.8)))
            })
        
        covariates_df = pd.DataFrame(covariates_data)
        
        # Generate event data
        event_times = np.sort(np.random.uniform(0, T, n_events))
        event_types = np.random.choice(['market', 'limit', 'cancel'], n_events, 
                                     p=[0.3, 0.5, 0.2])
        
        events_data = []
        for i, (time, event_type) in enumerate(zip(event_times, event_types)):
            events_data.append({
                'time': time,
                'event_type': event_type,
                'price': 100.0 + np.random.normal(0, 0.1),  # Synthetic price
                'side': np.random.choice(['bid', 'ask'])
            })
        
        events_df = pd.DataFrame(events_data)
        
        print(f"Created sample data: {len(events_df)} events, {len(covariates_df)} covariate points")
        
        return events_df, covariates_df


# Example usage
if __name__ == "__main__":
    handler = DataHandler()
    
    # Create sample data
    events_df, covariates_df = handler.create_sample_data()
    
    print("Sample Events:")
    print(events_df.head())
    print("\nSample Covariates:")
    print(covariates_df.head())
    
    # Get summary
    summary = handler.get_data_summary()
    print(f"\nData Summary: {summary}")
