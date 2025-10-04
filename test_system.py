#!/usr/bin/env python3
"""
Quick Test Script for LOB Intensity Simulator
Tests all core functionality to ensure everything works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.intensity_models import MarketOrderIntensityModel, LimitOrderIntensityModel
from core.placement_model import LimitOrderPlacementModel
from core.simulator import OrderFlowSimulator
from core.data_handler import DataHandler

def test_data_handler():
    """Test data handler functionality."""
    print("Testing DataHandler...")
    handler = DataHandler()
    events_df, covariates_df = handler.create_sample_data(T=50.0, n_events=20, n_covariates=10)
    
    assert len(events_df) > 0, "No events created"
    assert len(covariates_df) > 0, "No covariates created"
    assert 'time' in events_df.columns, "Missing time column in events"
    assert 'event_type' in events_df.columns, "Missing event_type column in events"
    assert 'time' in covariates_df.columns, "Missing time column in covariates"
    assert 'S' in covariates_df.columns, "Missing S column in covariates"
    
    print("✓ DataHandler test passed")
    return events_df, covariates_df

def test_intensity_models(events_df, covariates_df):
    """Test intensity model functionality."""
    print("Testing Intensity Models...")
    
    # Test market order model
    market_model = MarketOrderIntensityModel()
    market_params = market_model.fit(events_df, covariates_df)
    
    assert len(market_params) == 6, "Market model should have 6 parameters"
    assert market_model.is_fitted, "Market model should be marked as fitted"
    
    # Test limit order model
    limit_model = LimitOrderIntensityModel()
    limit_params = limit_model.fit(events_df, covariates_df)
    
    assert len(limit_params) == 6, "Limit model should have 6 parameters"
    assert limit_model.is_fitted, "Limit model should be marked as fitted"
    
    # Test intensity computation
    S = [0.01, 0.02]
    Q = [100, 200]
    intensities = market_model.intensity(S, Q)
    
    assert len(intensities) == 2, "Should return 2 intensity values"
    assert all(i > 0 for i in intensities), "All intensities should be positive"
    
    print("✓ Intensity Models test passed")
    return market_model, limit_model

def test_placement_model(events_df, covariates_df):
    """Test placement model functionality."""
    print("Testing Placement Model...")
    
    placement_model = LimitOrderPlacementModel(side='ask')
    weights, means, stds = placement_model.fit(events_df, covariates_df)
    
    assert len(weights) == 3, "Should have 3 Gaussian components"
    assert len(means) == 3, "Should have 3 means"
    assert len(stds) == 3, "Should have 3 standard deviations"
    assert placement_model.is_fitted, "Placement model should be marked as fitted"
    
    # Test sampling
    samples = placement_model.sample(10)
    assert len(samples) == 10, "Should generate 10 samples"
    
    print("✓ Placement Model test passed")
    return placement_model

def test_simulator(events_df, covariates_df):
    """Test simulator functionality."""
    print("Testing Simulator...")
    
    simulator = OrderFlowSimulator(random_state=42)
    simulator.fit_models(events_df, covariates_df)
    
    assert simulator.models_fitted, "Simulator should be marked as fitted"
    
    # Test simulation
    simulated_events = simulator.simulate_order_flow(covariates_df, T=30.0)
    
    assert len(simulated_events) > 0, "Should simulate some events"
    assert 'time' in simulated_events.columns, "Simulated events should have time column"
    assert 'event_type' in simulated_events.columns, "Simulated events should have event_type column"
    assert 'price' in simulated_events.columns, "Simulated events should have price column"
    
    print("✓ Simulator test passed")
    return simulator

def test_web_app():
    """Test web app import."""
    print("Testing Web App...")
    
    try:
        from app import app
        assert app is not None, "Flask app should be importable"
        
        # Check routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/upload', '/simulate', '/results', '/api/results', '/download/<file_type>', '/health']
        
        for route in expected_routes:
            assert route in routes, f"Missing route: {route}"
        
        print("✓ Web App test passed")
        
    except ImportError as e:
        print(f"✗ Web App test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=== LOB Intensity Simulator - Test Suite ===\n")
    
    try:
        # Test 1: Data Handler
        events_df, covariates_df = test_data_handler()
        
        # Test 2: Intensity Models
        market_model, limit_model = test_intensity_models(events_df, covariates_df)
        
        # Test 3: Placement Model
        placement_model = test_placement_model(events_df, covariates_df)
        
        # Test 4: Simulator
        simulator = test_simulator(events_df, covariates_df)
        
        # Test 5: Web App
        web_app_ok = test_web_app()
        
        print("\n=== Test Results ===")
        print("✓ DataHandler: PASSED")
        print("✓ Intensity Models: PASSED")
        print("✓ Placement Model: PASSED")
        print("✓ Simulator: PASSED")
        if web_app_ok:
            print("✓ Web App: PASSED")
        else:
            print("✗ Web App: FAILED")
        
        print("\n🎉 All core tests passed! The LOB Intensity Simulator is ready to use.")
        print("\nTo start the web interface:")
        print("  python3 app.py")
        print("Then open: http://localhost:8080")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
