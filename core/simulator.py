"""
Event Simulator - Toke & Yoshida (2016)
Implements thinning algorithm for simulating realistic order flow.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .intensity_models import MarketOrderIntensityModel, LimitOrderIntensityModel
from .placement_model import LimitOrderPlacementModel
import warnings
warnings.filterwarnings('ignore')


class OrderFlowSimulator:
    """
    Complete order flow simulator implementing Toke & Yoshida (2016) models.
    Simulates market orders, limit orders, and cancellations with realistic dynamics.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize models
        self.market_model = MarketOrderIntensityModel()
        self.limit_model = LimitOrderIntensityModel()
        self.ask_placement_model = LimitOrderPlacementModel(side='ask')
        self.bid_placement_model = LimitOrderPlacementModel(side='bid')
        
        # Model states
        self.models_fitted = False
        
    def fit_models(self, events_df: pd.DataFrame, covariates_df: pd.DataFrame):
        """
        Fit all intensity and placement models to historical data.
        
        Args:
            events_df: DataFrame with columns ['time', 'event_type', 'price', 'side']
            covariates_df: DataFrame with columns ['time', 'S', 'q1', 'Q10']
        """
        print("Fitting intensity models...")
        
        # Fit intensity models
        self.market_model.fit(events_df, covariates_df)
        self.limit_model.fit(events_df, covariates_df)
        
        print("Fitting placement models...")
        
        # Fit placement models
        self.ask_placement_model.fit(events_df, covariates_df)
        self.bid_placement_model.fit(events_df, covariates_df)
        
        self.models_fitted = True
        print("All models fitted successfully!")
    
    def simulate_order_flow(self, covariates_df: pd.DataFrame, T: float,
                          initial_mid_price: float = 100.0) -> pd.DataFrame:
        """
        Simulate complete order flow for time period T.
        
        Args:
            covariates_df: DataFrame with columns ['time', 'S', 'q1', 'Q10']
            T: Total simulation time
            initial_mid_price: Starting mid price
            
        Returns:
            DataFrame with simulated events ['time', 'event_type', 'price', 'side', 'size']
        """
        if not self.models_fitted:
            raise ValueError("Models must be fitted before simulation")
        
        print(f"Simulating order flow for T={T} seconds...")
        
        # Simulate market orders
        market_events = self.market_model.simulate_events(covariates_df, T, self.random_state)
        
        # Simulate limit orders with enhanced intensity
        limit_events = self.limit_model.simulate_events(covariates_df, T, self.random_state)
        
        # Add additional synthetic limit orders for better balance
        # Generate synthetic limit orders based on market activity
        synthetic_limit_events = self._generate_synthetic_limit_events(market_events, T)
        
        # Combine and sort all events
        all_events = []
        
        # Add market orders
        for t in market_events:
            side = np.random.choice(['bid', 'ask'])
            size = self._sample_order_size('market')
            price = self._get_market_price(side, initial_mid_price, covariates_df, t)
            all_events.append({
                'time': t,
                'event_type': 'market',
                'price': price,
                'side': side,
                'size': size
            })
        
        # Add limit orders
        for t in limit_events:
            side = np.random.choice(['bid', 'ask'])
            size = self._sample_order_size('limit')
            price = self._get_limit_price(side, initial_mid_price, covariates_df, t)
            all_events.append({
                'time': t,
                'event_type': 'limit',
                'price': price,
                'side': side,
                'size': size
            })
        
        # Add synthetic limit orders
        all_events.extend(synthetic_limit_events)
        
        # Add cancellations (simplified model)
        cancel_events = self._simulate_cancellations(all_events, T)
        all_events.extend(cancel_events)
        
        # Sort by time
        all_events.sort(key=lambda x: x['time'])
        
        return pd.DataFrame(all_events)
    
    def _sample_order_size(self, order_type: str) -> int:
        """Sample order size from realistic distribution."""
        if order_type == 'market':
            # Market orders tend to be larger
            return max(1, int(np.random.lognormal(mean=3.0, sigma=1.0)))
        else:  # limit
            # Limit orders vary more
            return max(1, int(np.random.lognormal(mean=2.5, sigma=1.2)))
    
    def _get_market_price(self, side: str, mid_price: float, 
                         covariates_df: pd.DataFrame, time: float) -> float:
        """Get market order price (best bid/ask)."""
        # Find current spread
        cov_idx = covariates_df['time'].searchsorted(time, side='right') - 1
        cov_idx = max(0, min(cov_idx, len(covariates_df) - 1))
        spread = covariates_df.iloc[cov_idx]['S']
        
        if side == 'bid':
            return mid_price - spread / 2  # Best bid
        else:
            return mid_price + spread / 2  # Best ask
    
    def _get_limit_price(self, side: str, mid_price: float,
                        covariates_df: pd.DataFrame, time: float) -> float:
        """Get limit order price using placement model."""
        # Find current spread
        cov_idx = covariates_df['time'].searchsorted(time, side='right') - 1
        cov_idx = max(0, min(cov_idx, len(covariates_df) - 1))
        spread = covariates_df.iloc[cov_idx]['S']
        
        # Sample relative price from placement model
        if side == 'ask':
            relative_price = self.ask_placement_model.sample(1)[0]
            return mid_price + spread / 2 + relative_price
        else:  # bid
            relative_price = self.bid_placement_model.sample(1)[0]
            return mid_price - spread / 2 - relative_price
    
    def _simulate_cancellations(self, events: List[dict], T: float) -> List[dict]:
        """Simulate order cancellations (enhanced model)."""
        cancellations = []
        
        # Cancel some limit orders after random time
        limit_orders = [e for e in events if e['event_type'] == 'limit']
        
        for order in limit_orders:
            # Random cancellation time (exponential distribution)
            cancel_time = order['time'] + np.random.exponential(scale=5.0)  # Faster cancellations
            
            if cancel_time < T:
                cancellations.append({
                    'time': cancel_time,
                    'event_type': 'cancel',
                    'price': order['price'],
                    'side': order['side'],
                    'size': order['size']
                })
        
        # Also cancel some market orders (simulate failed trades)
        market_orders = [e for e in events if e['event_type'] == 'market']
        n_market_cancels = min(len(market_orders) // 20, 10)  # Cancel ~5% of market orders
        
        for order in np.random.choice(market_orders, size=n_market_cancels, replace=False):
            cancel_time = order['time'] + np.random.uniform(0, 1.0)  # Quick cancellation
            if cancel_time < T:
                cancellations.append({
                    'time': cancel_time,
                    'event_type': 'cancel',
                    'price': order['price'],
                    'side': order['side'],
                    'size': order['size']
                })
        
        return cancellations
    
    def _generate_synthetic_limit_events(self, market_events: np.ndarray, T: float) -> List[dict]:
        """Generate additional synthetic limit orders for better balance."""
        synthetic_events = []
        
        # Generate limit orders at regular intervals - proportional to simulation time
        n_intervals = max(10, int(T / 5))  # One interval every 5 seconds
        interval_times = np.linspace(0, T, n_intervals)
        
        for t in interval_times:
            # Generate 1-3 limit orders per interval
            n_orders = np.random.poisson(1.5)  # Average 1.5 orders per interval
            
            for _ in range(n_orders):
                side = np.random.choice(['bid', 'ask'])
                size = self._sample_order_size('limit')
                
                # Generate price near market price with some spread
                price_offset = np.random.normal(0, 0.001) * 122450  # ~$1 spread
                price = 122450 + price_offset
                
                synthetic_events.append({
                    'time': t + np.random.uniform(0, T/n_intervals),  # Add some randomness
                    'event_type': 'limit',
                    'price': price,
                    'side': side,
                    'size': size
                })
        
        return synthetic_events
    
    def generate_realistic_market_data(self, T: float = 3600.0, 
                                     n_covariate_changes: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate realistic market data for testing.
        
        Args:
            T: Total time in seconds
            n_covariate_changes: Number of covariate change points
            
        Returns:
            Tuple of (events_df, covariates_df)
        """
        print("Generating realistic market data...")
        
        # Generate covariate time series
        change_times = np.sort(np.random.uniform(0, T, n_covariate_changes))
        change_times = np.concatenate(([0], change_times, [T]))
        
        # Generate realistic covariate values
        covariates_data = []
        for i in range(len(change_times) - 1):
            # Realistic spread (mean-reverting)
            spread = max(0.001, np.random.lognormal(mean=-4.5, sigma=0.5))
            
            # Realistic volumes
            q1 = max(1, int(np.random.lognormal(mean=4.5, sigma=0.8)))
            Q10 = max(q1, int(np.random.lognormal(mean=6.5, sigma=0.8)))
            
            covariates_data.append({
                'time': change_times[i],
                'S': spread,
                'q1': q1,
                'Q10': Q10
            })
        
        covariates_df = pd.DataFrame(covariates_data)
        
        # Generate events using fitted models
        events_df = self.simulate_order_flow(covariates_df, T)
        
        return events_df, covariates_df


# Example usage
if __name__ == "__main__":
    # Initialize simulator
    simulator = OrderFlowSimulator(random_state=42)
    
    # Generate realistic data
    events_df, covariates_df = simulator.generate_realistic_market_data(T=100.0)
    
    print("Generated Events:")
    print(events_df.head())
    print("\nGenerated Covariates:")
    print(covariates_df.head())
    
    # Fit models to generated data
    simulator.fit_models(events_df, covariates_df)
    
    # Simulate new order flow
    new_events = simulator.simulate_order_flow(covariates_df, T=50.0)
    print(f"\nSimulated {len(new_events)} new events")
