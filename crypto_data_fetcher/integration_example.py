#!/usr/bin/env python3
"""
Integration script: Crypto Data → LOB Intensity Simulator
This script demonstrates the complete workflow from fetching crypto data
to running simulations with the LOB Intensity Simulator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_data_fetcher.generate_lob_csvs import CryptoDataFetcher
from core.simulator import OrderFlowSimulator
from core.data_handler import DataHandler

def main():
    """Complete workflow: Fetch crypto data → Fit models → Simulate."""
    print("=== Crypto Data → LOB Intensity Simulator Integration ===\n")
    
    # Step 1: Fetch real crypto data
    print("Step 1: Fetching real crypto data from Binance...")
    fetcher = CryptoDataFetcher(symbol="BTC/USDT", exchange_name="binance")
    
    # Fetch a reasonable amount of data for model fitting
    cov_df, evt_df = fetcher.fetch_data(n_snapshots=50, snapshot_interval=3.0)
    
    if len(cov_df) == 0:
        print("❌ No data fetched. Check your internet connection.")
        return 1
    
    print(f"✅ Fetched {len(cov_df)} covariate points and {len(evt_df)} events")
    
    # Step 2: Save the data
    print("\nStep 2: Saving crypto data...")
    cov_path, evt_path, metadata_path = fetcher.save_data(cov_df, evt_df, "real_crypto_data")
    
    # Step 3: Load data into LOB simulator format
    print("\nStep 3: Loading data into LOB simulator...")
    handler = DataHandler()
    
    # Create a temporary events file with the required format
    evt_df_formatted = evt_df.copy()
    if 'price' not in evt_df_formatted.columns:
        # Generate synthetic prices if not available
        evt_df_formatted['price'] = 50000.0  # Default BTC price
    
    # Save formatted data
    evt_df_formatted.to_csv("real_crypto_data/events_formatted.csv", index=False)
    cov_df.to_csv("real_crypto_data/covariates_formatted.csv", index=False)
    
    # Step 4: Initialize and fit LOB simulator
    print("\nStep 4: Fitting LOB intensity models...")
    simulator = OrderFlowSimulator(random_state=42)
    
    try:
        simulator.fit_models(evt_df_formatted, cov_df)
        print("✅ Models fitted successfully!")
        
        # Display fitted parameters
        print("\nFitted Parameters:")
        print("Market Order Intensity (β):", simulator.market_model.beta)
        print("Limit Order Intensity (β):", simulator.limit_model.beta)
        
    except Exception as e:
        print(f"❌ Error fitting models: {e}")
        return 1
    
    # Step 5: Run simulation with fitted models
    print("\nStep 5: Running simulation with fitted models...")
    try:
        simulated_events = simulator.simulate_order_flow(
            cov_df, T=100.0, initial_mid_price=50000.0
        )
        
        print(f"✅ Simulated {len(simulated_events)} events")
        print("Event distribution:")
        print(simulated_events['event_type'].value_counts())
        
        # Save simulation results
        simulated_events.to_csv("real_crypto_data/simulated_events.csv", index=False)
        print("Simulation results saved to: real_crypto_data/simulated_events.csv")
        
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return 1
    
    # Step 6: Summary
    print("\n=== Integration Complete ===")
    print("Files generated:")
    print(f"  📊 Real crypto data: {cov_path}, {evt_path}")
    print(f"  📈 Simulation results: real_crypto_data/simulated_events.csv")
    print(f"  📋 Metadata: {metadata_path}")
    
    print("\nNext steps:")
    print("1. Upload the CSV files to the web interface at http://localhost:8080")
    print("2. Or use the files programmatically with the LOB simulator")
    print("3. Compare real vs simulated order flow patterns")
    
    return 0

if __name__ == "__main__":
    exit(main())
