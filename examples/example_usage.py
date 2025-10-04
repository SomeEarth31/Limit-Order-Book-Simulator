"""
Complete Example Usage - LOB Intensity Simulator
Demonstrates full workflow from data loading to simulation.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intensity_models import MarketOrderIntensityModel, LimitOrderIntensityModel
from core.placement_model import LimitOrderPlacementModel
from core.simulator import OrderFlowSimulator
from core.data_handler import DataHandler


def main():
    """Complete example demonstrating the LOB Intensity Simulator."""
    
    print("=== LOB Intensity Simulator - Complete Example ===\n")
    
    # Step 1: Generate or load sample data
    print("Step 1: Creating sample data...")
    handler = DataHandler()
    events_df, covariates_df = handler.create_sample_data(T=200.0, n_events=100, n_covariates=30)
    
    print(f"Created {len(events_df)} events and {len(covariates_df)} covariate points")
    print("Sample Events:")
    print(events_df.head())
    print("\nSample Covariates:")
    print(covariates_df.head())
    
    # Step 2: Initialize simulator
    print("\nStep 2: Initializing simulator...")
    simulator = OrderFlowSimulator(random_state=42)
    
    # Step 3: Fit models
    print("\nStep 3: Fitting models...")
    simulator.fit_models(events_df, covariates_df)
    
    # Step 4: Display fitted parameters
    print("\nStep 4: Fitted Model Parameters")
    print("=" * 50)
    
    market_params = simulator.market_model.beta
    limit_params = simulator.limit_model.beta
    
    print("Market Order Intensity Parameters (β):")
    param_names = ['β0', 'β1', 'β11', 'β2', 'β22', 'β12']
    for name, param in zip(param_names, market_params):
        print(f"  {name}: {param:.6f}")
    
    print("\nLimit Order Intensity Parameters (β):")
    for name, param in zip(param_names, limit_params):
        print(f"  {name}: {param:.6f}")
    
    # Step 5: Placement model parameters
    print("\nPlacement Model Parameters:")
    ask_params = simulator.ask_placement_model.get_parameters()
    bid_params = simulator.bid_placement_model.get_parameters()
    
    print("Ask Side Placement:")
    print(f"  Weights: {ask_params['weights']}")
    print(f"  Means: {ask_params['means']}")
    print(f"  Std Devs: {ask_params['std_devs']}")
    
    print("Bid Side Placement:")
    print(f"  Weights: {bid_params['weights']}")
    print(f"  Means: {bid_params['means']}")
    print(f"  Std Devs: {bid_params['std_devs']}")
    
    # Step 6: Simulate new order flow
    print("\nStep 5: Simulating new order flow...")
    simulated_events = simulator.simulate_order_flow(covariates_df, T=100.0, initial_mid_price=100.0)
    
    print(f"Simulated {len(simulated_events)} events")
    print("Event type distribution:")
    print(simulated_events['event_type'].value_counts())
    
    # Step 7: Save results
    print("\nStep 6: Saving results...")
    
    # Save simulated events
    simulated_events.to_csv('examples/simulated_events.csv', index=False)
    covariates_df.to_csv('examples/simulated_covariates.csv', index=False)
    
    # Save parameters
    results = {
        'market_order_intensity_params': market_params.tolist(),
        'limit_order_intensity_params': limit_params.tolist(),
        'ask_placement_params': ask_params,
        'bid_placement_params': bid_params,
        'simulation_summary': {
            'n_simulated_events': len(simulated_events),
            'event_distribution': simulated_events['event_type'].value_counts().to_dict(),
            'simulation_time': 100.0,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    with open('examples/simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to:")
    print("  - examples/simulated_events.csv")
    print("  - examples/simulated_covariates.csv")
    print("  - examples/simulation_results.json")
    
    # Step 8: Demonstrate individual model usage
    print("\nStep 7: Individual Model Usage Examples")
    print("=" * 50)
    
    # Market order intensity
    print("Market Order Intensity Example:")
    market_model = MarketOrderIntensityModel()
    market_model.beta = market_params  # Use fitted parameters
    
    # Compute intensity at specific spread and volume
    S = np.array([0.01, 0.015, 0.02])
    q1 = np.array([100, 150, 200])
    intensities = market_model.intensity(S, q1)
    
    for i, (s, q, intensity) in enumerate(zip(S, q1, intensities)):
        print(f"  S={s:.3f}, q1={q}, λ_M={intensity:.4f}")
    
    # Limit order placement
    print("\nLimit Order Placement Example:")
    ask_model = LimitOrderPlacementModel(side='ask')
    ask_model.weights_ = ask_params['weights']
    ask_model.means_ = ask_params['means']
    ask_model.covariances_ = ask_params['std_devs'] ** 2
    ask_model.is_fitted = True
    
    # Sample new placements
    samples = ask_model.sample(10)
    print(f"  Generated 10 ask placement samples: {samples}")
    
    print("\n=== Example Complete ===")
    print("The LOB Intensity Simulator is ready for use!")
    print("Run 'python app.py' to start the web interface.")


if __name__ == "__main__":
    main()
