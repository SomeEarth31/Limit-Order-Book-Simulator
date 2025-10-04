#!/usr/bin/env python3
"""
Generate clean crypto data for LOB Intensity Simulator web interface.
This script creates properly formatted CSV files that work with the web app.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_lob_csvs import CryptoDataFetcher

def main():
    """Generate clean crypto data for web interface."""
    print("=== Generating Clean Crypto Data for Web Interface ===\n")
    
    # Initialize fetcher
    fetcher = CryptoDataFetcher(symbol="BTC/USDT", exchange_name="binance")
    
    # Fetch a reasonable amount of data
    print("Fetching crypto data...")
    cov_df, evt_df = fetcher.fetch_data(n_snapshots=30, snapshot_interval=3.0)
    
    if len(cov_df) == 0:
        print("❌ No data fetched. Check your internet connection.")
        return 1
    
    print(f"✅ Fetched {len(cov_df)} covariate points and {len(evt_df)} events")
    
    # Save to examples folder for web interface
    output_dir = "../examples"
    cov_path, evt_path, metadata_path = fetcher.save_data(cov_df, evt_df, output_dir)
    
    # Rename files to be more descriptive
    import shutil
    shutil.move(os.path.join(output_dir, "covariates.csv"), 
                os.path.join(output_dir, "crypto_covariates.csv"))
    shutil.move(os.path.join(output_dir, "events.csv"), 
                os.path.join(output_dir, "crypto_events.csv"))
    
    print(f"\n✅ Clean crypto data saved to examples folder:")
    print(f"  📊 {os.path.join(output_dir, 'crypto_covariates.csv')}")
    print(f"  📈 {os.path.join(output_dir, 'crypto_events.csv')}")
    
    print(f"\nYou can now upload these files to the web interface at http://localhost:8080")
    
    return 0

if __name__ == "__main__":
    exit(main())
