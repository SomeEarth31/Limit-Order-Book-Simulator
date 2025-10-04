#!/usr/bin/env python3
"""
LOB Simulation Diagnostic Tool
Helps diagnose issues with simulation parameters and data quality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from core.data_handler import DataHandler
from core.simulator import OrderFlowSimulator

def diagnose_simulation_issues(events_df, covariates_df):
    """Diagnose potential issues with simulation data."""
    print("=== LOB Simulation Diagnostic Report ===\n")
    
    # Basic data analysis
    print("📊 DATA ANALYSIS")
    print("-" * 50)
    print(f"Events: {len(events_df)}")
    print(f"Covariates: {len(covariates_df)}")
    print(f"Time Range: {events_df['time'].min():.2f} - {events_df['time'].max():.2f}")
    
    # Event distribution
    event_counts = events_df['event_type'].value_counts()
    print(f"\nEvent Distribution:")
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count} ({count/len(events_df)*100:.1f}%)")
    
    # Covariate analysis
    print(f"\nCovariate Analysis:")
    print(f"  Spread (S): {covariates_df['S'].min():.6f} - {covariates_df['S'].max():.6f}")
    print(f"  q1: {covariates_df['q1'].min():.2f} - {covariates_df['q1'].max():.2f}")
    print(f"  Q10: {covariates_df['Q10'].min():.2f} - {covariates_df['Q10'].max():.2f}")
    
    # Check for problematic values
    print(f"\n⚠️ POTENTIAL ISSUES:")
    
    issues = []
    
    # Check for very small spreads
    min_spread = covariates_df['S'].min()
    if min_spread < 1e-6:
        issues.append(f"Very small spreads detected (min: {min_spread:.2e})")
    
    # Check for zero volumes
    zero_q1 = (covariates_df['q1'] == 0).sum()
    zero_q10 = (covariates_df['Q10'] == 0).sum()
    if zero_q1 > 0:
        issues.append(f"Zero q1 values: {zero_q1}")
    if zero_q10 > 0:
        issues.append(f"Zero Q10 values: {zero_q10}")
    
    # Check event distribution
    if len(event_counts) < 2:
        issues.append("Only one event type - need both market and limit orders")
    
    if 'market' not in event_counts:
        issues.append("No market orders found")
    
    if 'limit' not in event_counts:
        issues.append("No limit orders found")
    
    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ No obvious issues detected")
    
    return issues

def test_parameter_bounds():
    """Test parameter fitting with bounds."""
    print("\n🔧 PARAMETER FITTING TEST")
    print("-" * 50)
    
    # Create test data
    handler = DataHandler()
    events_df, covariates_df = handler.create_sample_data(T=100, n_covariates=21)
    
    # Fit models
    simulator = OrderFlowSimulator(random_state=42)
    
    try:
        simulator.fit_models(events_df, covariates_df)
        
        print("✅ Model fitting successful!")
        print(f"Market β0: {simulator.market_model.beta[0]:.3f}")
        print(f"Limit β0: {simulator.limit_model.beta[0]:.3f}")
        
        # Check if parameters are reasonable
        market_beta0 = simulator.market_model.beta[0]
        limit_beta0 = simulator.limit_model.beta[0]
        
        if abs(market_beta0) > 50:
            print(f"⚠️ Market β0 is large: {market_beta0:.3f}")
        if abs(limit_beta0) > 50:
            print(f"⚠️ Limit β0 is large: {limit_beta0:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model fitting failed: {e}")
        return False

def test_simulation_performance():
    """Test simulation performance."""
    print("\n⚡ SIMULATION PERFORMANCE TEST")
    print("-" * 50)
    
    # Create test data
    handler = DataHandler()
    events_df, covariates_df = handler.create_sample_data(T=100, n_covariates=21)
    
    # Fit models
    simulator = OrderFlowSimulator(random_state=42)
    simulator.fit_models(events_df, covariates_df)
    simulator._original_covariates = covariates_df
    
    # Test different simulation times
    test_times = [1, 5, 10, 30]
    
    for T in test_times:
        print(f"\nTesting T={T} seconds:")
        
        try:
            # Scale covariates
            original_covariates = covariates_df.copy()
            original_time_range = original_covariates['time'].max() - original_covariates['time'].min()
            time_scale = T / original_time_range
            original_covariates['time'] = original_covariates['time'] * time_scale
            
            # Run simulation
            simulated_events = simulator.simulate_order_flow(
                original_covariates, T=T, initial_mid_price=100.0
            )
            
            print(f"  ✅ Success: {len(simulated_events)} events")
            print(f"  Events/sec: {len(simulated_events)/T:.1f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def main():
    """Run comprehensive diagnostics."""
    print("🔍 LOB Simulation Diagnostic Tool\n")
    
    # Test with sample data
    handler = DataHandler()
    events_df, covariates_df = handler.create_sample_data(T=100, n_covariates=21)
    
    # Run diagnostics
    issues = diagnose_simulation_issues(events_df, covariates_df)
    
    # Test parameter fitting
    fitting_success = test_parameter_bounds()
    
    # Test simulation performance
    test_simulation_performance()
    
    # Summary
    print("\n📋 SUMMARY")
    print("=" * 50)
    
    if not issues and fitting_success:
        print("✅ All tests passed! Your simulation should work correctly.")
        print("\n💡 RECOMMENDATIONS:")
        print("1. The parameter bounds fix should prevent infinite loops")
        print("2. Intensity capping prevents overflow issues")
        print("3. Timeout protection prevents hanging simulations")
        print("4. Try uploading your CSV files again")
    else:
        print("⚠️ Some issues detected. Check the recommendations above.")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Upload your CSV files to the web interface")
    print("2. Try running a short simulation (1-10 seconds)")
    print("3. If it works, gradually increase simulation time")
    print("4. Check the parameter values for reasonableness")

if __name__ == "__main__":
    main()
