"""
Limit Order Placement Model - Toke & Yoshida (2016)
Implements 3-component Gaussian mixture model for limit order price placement.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LimitOrderPlacementModel:
    """
    Model for limit order placement using Gaussian mixture distribution.
    Based on equation 12 from the paper:
    πL(p; G, μ, σ, π) = Σ(i=1 to G) πi * φ(p; μi, σi)
    where φ(μ, σ) is the density of the Gaussian distribution.
    """
    
    def __init__(self, n_components: int = 3, side: str = 'ask'):
        """
        Args:
            n_components: Number of Gaussian components (G in paper)
            side: 'ask' or 'bid' - which side of the book
        """
        self.n_components = n_components
        self.side = side
        self.model = None
        self.is_fitted = False
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        
    def calculate_relative_prices(self, events_df: pd.DataFrame, 
                                 covariates_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate relative prices (distance from same-side best quote).
        
        Args:
            events_df: DataFrame with columns ['time', 'event_type', 'price']
            covariates_df: DataFrame with columns ['time', 'S', 'q1', 'Q10']
            
        Returns:
            Array of relative prices
        """
        # Filter limit orders only
        limit_orders = events_df[events_df['event_type'] == 'limit'].copy()
        
        if len(limit_orders) == 0:
            raise ValueError("No limit orders found in events data")
        
        relative_prices = []
        
        for _, order in limit_orders.iterrows():
            order_time = order['time']
            order_price = order['price']
            
            # Find closest covariate data point
            cov_idx = covariates_df['time'].searchsorted(order_time, side='right') - 1
            cov_idx = max(0, min(cov_idx, len(covariates_df) - 1))
            
            # Get spread and compute best quote
            spread = covariates_df.iloc[cov_idx]['S']
            
            if self.side == 'ask':
                # For ask orders, relative price = order_price - best_ask
                # Approximate best_ask as mid_price + spread/2
                mid_price = 100.0  # Assume mid price (can be made more sophisticated)
                best_ask = mid_price + spread / 2
                relative_price = order_price - best_ask
            else:  # bid
                # For bid orders, relative price = best_bid - order_price
                mid_price = 100.0
                best_bid = mid_price - spread / 2
                relative_price = best_bid - order_price
            
            relative_prices.append(relative_price)
        
        return np.array(relative_prices)
    
    def fit(self, events_df: pd.DataFrame, covariates_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit Gaussian mixture model to limit order placement data.
        
        Args:
            events_df: DataFrame with columns ['time', 'event_type', 'price']
            covariates_df: DataFrame with columns ['time', 'S', 'q1', 'Q10']
            
        Returns:
            Tuple of (weights, means, standard_deviations)
        """
        # Calculate relative prices
        relative_prices = self.calculate_relative_prices(events_df, covariates_df)
        
        if len(relative_prices) < self.n_components:
            raise ValueError(f"Not enough limit orders ({len(relative_prices)}) for {self.n_components} components")
        
        # Reshape for sklearn
        X = relative_prices.reshape(-1, 1)
        
        # Initialize and fit Gaussian Mixture
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',  # Each component has its own variance
            random_state=42,
            n_init=10,
            max_iter=1000
        )
        
        self.model.fit(X)
        
        # Extract parameters
        self.weights_ = self.model.weights_
        self.means_ = self.model.means_.flatten()
        self.covariances_ = self.model.covariances_.flatten()
        
        # Sort components by mean for consistency
        idx = np.argsort(self.means_)
        self.weights_ = self.weights_[idx]
        self.means_ = self.means_[idx]
        self.covariances_ = self.covariances_[idx]
        
        self.is_fitted = True
        
        return self.weights_, self.means_, np.sqrt(self.covariances_)
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the fitted distribution.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed
            
        Returns:
            Generated relative price samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        samples, _ = self.model.sample(n_samples)
        return samples.flatten()
    
    def density(self, x: np.ndarray) -> np.ndarray:
        """
        Compute probability density at given points.
        
        Args:
            x: Points at which to evaluate density
            
        Returns:
            Probability density values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing density")
        
        return np.exp(self.model.score_samples(x.reshape(-1, 1)))
    
    def get_parameters(self) -> dict:
        """Get fitted model parameters."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return {
            'weights': self.weights_,
            'means': self.means_,
            'std_devs': np.sqrt(self.covariances_),
            'n_components': self.n_components,
            'side': self.side
        }


# Example usage
if __name__ == "__main__":
    # Sample data with prices
    events = pd.DataFrame({
        'time': [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0],
        'event_type': ['market', 'limit', 'market', 'limit', 'market', 'limit', 'cancel'],
        'price': [100.0, 100.05, 100.0, 100.03, 100.0, 100.02, 100.0]
    })
    
    covariates = pd.DataFrame({
        'time': [0.0, 2.0, 5.0, 8.0],
        'S': [0.01, 0.015, 0.012, 0.013],
        'q1': [100, 150, 120, 140],
        'Q10': [1000, 1500, 1200, 1400]
    })
    
    # Fit ask side placement model
    ask_model = LimitOrderPlacementModel(n_components=3, side='ask')
    weights, means, stds = ask_model.fit(events, covariates)
    
    print("Ask Side Placement Model Parameters:")
    print(f"Weights (π): {weights}")
    print(f"Means (μ): {means}")
    print(f"Standard Deviations (σ): {stds}")
    
    # Sample new placements
    samples = ask_model.sample(100, random_state=42)
    print(f"\nGenerated {len(samples)} sample placements")
