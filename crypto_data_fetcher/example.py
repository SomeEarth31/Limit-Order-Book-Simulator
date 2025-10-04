#!/usr/bin/env python3
"""
Simple example script for fetching crypto data.
This script demonstrates basic usage of the CryptoDataFetcher.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_lob_csvs import CryptoDataFetcher

def main():
    """Simple example of fetching crypto data."""
    print("=== Crypto Data Fetcher Example ===\n")
    
    # Initialize fetcher for BTC/USDT on Binance
    print("Initializing fetcher for BTC/USDT on Binance...")
    fetcher = CryptoDataFetcher(symbol="BTC/USDT", exchange_name="binance")
    
    # Fetch a small amount of data for testing
    print("\nFetching 10 snapshots with 2-second intervals...")
    cov_df, evt_df = fetcher.fetch_data(n_snapshots=10, snapshot_interval=2.0)
    
    # Save the data
    print("\nSaving data...")
    cov_path, evt_path, metadata_path = fetcher.save_data(cov_df, evt_df, "sample_data")
    
    # Display summary
    print(f"\n=== Data Summary ===")
    print(f"Covariates: {len(cov_df)} rows")
    print(f"Events: {len(evt_df)} rows")
    print(f"Time range: {cov_df['time'].min():.2f} - {cov_df['time'].max():.2f} seconds")
    print(f"Spread range: {cov_df['S'].min():.6f} - {cov_df['S'].max():.6f}")
    
    print(f"\nEvent types:")
    print(evt_df['event_type'].value_counts())
    
    print(f"\n✅ Sample data ready!")
    print(f"Files saved in 'sample_data/' directory")
    print(f"You can now upload these to the LOB Intensity Simulator at http://localhost:8080")

if __name__ == "__main__":
    main()
