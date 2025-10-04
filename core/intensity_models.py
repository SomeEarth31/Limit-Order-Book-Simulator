"""
LOB Intensity Models - Toke & Yoshida (2016)
Implements parametric intensity functions for market and limit orders with MLE estimation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class IntensityModel:
    """
    Base class for intensity models implementing the general form:
    λ(t; S, Q) = exp(β0 + β1·ln(S) + β11·(ln S)² + β2·ln(1+Q) + β22·(ln(1+Q))² + β12·ln S·ln(1+Q))
    
    where S is spread and Q is either q1 (best quote volume) or Q10 (10-level total volume).
    """
    
    def __init__(self, model_type: str = 'market'):
        """
        Args:
            model_type: 'market' or 'limit' order intensity model
        """
        self.model_type = model_type
        self.beta = None  # Parameters [β0, β1, β11, β2, β22, β12]
        self.is_fitted = False
        
    def intensity(self, S: np.ndarray, Q: np.ndarray, beta: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute intensity λ(t; S, Q) for given spread and volume.
        
        Args:
            S: Spread values (array)
            Q: Volume values (q1 or Q10, array)
            beta: Parameter vector [β0, β1, β11, β2, β22, β12]. Uses fitted params if None.
            
        Returns:
            Intensity values (array)
        """
        if beta is None:
            if not self.is_fitted:
                raise ValueError("Model must be fitted first or beta must be provided")
            beta = self.beta
            
        # Ensure positive values for log
        S = np.maximum(S, 1e-10)
        Q = np.maximum(Q, 0)
        
        # Compute log terms
        ln_S = np.log(S)
        ln_1_plus_Q = np.log(1 + Q)
        
        # Compute intensity
        log_intensity = (
            beta[0] +                      # β0
            beta[1] * ln_S +               # β1·ln(S)
            beta[2] * ln_S**2 +            # β11·(ln S)²
            beta[3] * ln_1_plus_Q +        # β2·ln(1+Q)
            beta[4] * ln_1_plus_Q**2 +     # β22·(ln(1+Q))²
            beta[5] * ln_S * ln_1_plus_Q   # β12·ln S·ln(1+Q)
        )
        
        # Cap log intensity to prevent overflow
        log_intensity = np.minimum(log_intensity, 10.0)  # Cap at exp(10) ≈ 22,000 events/second
        
        return np.exp(log_intensity)
    
    def log_likelihood(self, beta: np.ndarray, 
                      event_times: np.ndarray,
                      S_series: np.ndarray,
                      Q_series: np.ndarray,
                      change_times: np.ndarray) -> float:
        """
        Compute log-likelihood for piecewise-constant covariates (Equation 7 in paper).
        
        Args:
            beta: Parameter vector
            event_times: Times when events (market/limit orders) occur
            S_series: Spread values (piecewise constant)
            Q_series: Volume values (piecewise constant)
            change_times: Times when S or Q changes
            
        Returns:
            Negative log-likelihood (for minimization)
        """
        # First term: sum of log intensities at event times
        if len(event_times) == 0:
            return 1e10  # Large value if no events
        
        # Filter events to only include those within the covariate time range
        valid_events = event_times[(event_times >= 0) & (event_times < len(S_series))]
        if len(valid_events) == 0:
            return 1e10  # No valid events
            
        intensities_at_events = self.intensity(
            S_series[valid_events.astype(int)],
            Q_series[valid_events.astype(int)],
            beta
        )
        
        log_intensities_sum = np.sum(np.log(np.maximum(intensities_at_events, 1e-300)))
        
        # Second term: integral of intensity (sum over piecewise-constant intervals)
        if len(change_times) == 0:
            return -log_intensities_sum  # Only first term
            
        # Compute intensity at each change point
        intensities_at_changes = self.intensity(
            S_series[change_times.astype(int)],
            Q_series[change_times.astype(int)],
            beta
        )
        
        # Compute time differences between change points
        time_diffs = np.diff(np.concatenate(([0], change_times)))
        
        # Integral approximation (piecewise constant)
        integral = np.sum(time_diffs * intensities_at_changes)
        
        # Log-likelihood = sum log λ(t_i) - ∫λ(t)dt
        log_likelihood = log_intensities_sum - integral
        
        return -log_likelihood  # Return negative for minimization
    
    def fit(self, events_df: pd.DataFrame, covariates_df: pd.DataFrame) -> np.ndarray:
        """
        Fit model parameters via Maximum Likelihood Estimation.
        
        Args:
            events_df: DataFrame with columns ['time', 'event_type']
                      event_type: 'market', 'limit', or 'cancel'
            covariates_df: DataFrame with columns ['time', 'S', 'q1', 'Q10']
                          Values are constant until next time point
                          
        Returns:
            Fitted parameter vector [β0, β1, β11, β2, β22, β12]
        """
        # Filter events based on model type
        if self.model_type == 'market':
            event_mask = events_df['event_type'] == 'market'
            Q_col = 'q1'  # Use best quote volume for market orders
        else:  # limit
            event_mask = events_df['event_type'] == 'limit'
            Q_col = 'Q10'  # Use 10-level volume for limit orders
            
        event_times = events_df[event_mask]['time'].values
        
        if len(event_times) == 0:
            raise ValueError(f"No {self.model_type} orders found in events data")
        
        # Create piecewise-constant series for S and Q
        max_time = int(np.ceil(max(events_df['time'].max(), covariates_df['time'].max())))
        S_series = np.zeros(max_time + 1)
        Q_series = np.zeros(max_time + 1)
        
        # Fill in piecewise-constant values
        for i in range(len(covariates_df)):
            start_idx = int(covariates_df.iloc[i]['time'])
            end_idx = int(covariates_df.iloc[i+1]['time']) if i < len(covariates_df)-1 else max_time + 1
            S_series[start_idx:end_idx] = covariates_df.iloc[i]['S']
            Q_series[start_idx:end_idx] = covariates_df.iloc[i][Q_col]
        
        # Find change times (union of S and Q changes)
        S_changes = np.where(np.diff(S_series) != 0)[0]
        Q_changes = np.where(np.diff(Q_series) != 0)[0]
        change_times = np.unique(np.concatenate(([0], S_changes, Q_changes)))
        
        # Initial parameter guess
        beta_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Optimize using scipy.optimize.minimize with bounds
        bounds = [
            (-50, 50),   # β0: intercept
            (-20, 20),   # β1: ln(S) coefficient
            (-10, 10),   # β11: (ln S)² coefficient
            (-20, 20),   # β2: ln(1+Q) coefficient
            (-10, 10),   # β22: (ln(1+Q))² coefficient
            (-10, 10)    # β12: ln S·ln(1+Q) coefficient
        ]
        
        result = minimize(
            fun=self.log_likelihood,
            x0=beta_init,
            args=(event_times, S_series, Q_series, change_times),
            method='L-BFGS-B',  # Use bounded optimization
            bounds=bounds,
            options={
                'maxiter': 10000,
                'ftol': 1e-8,
                'gtol': 1e-8,
                'disp': False
            }
        )
        
        if not result.success:
            print(f"Warning: Optimization did not fully converge: {result.message}")
        
        self.beta = result.x
        self.is_fitted = True
        
        return self.beta
    
    def simulate_events(self, covariates_df: pd.DataFrame, T: float, 
                       random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate event arrivals using thinning algorithm (Ogata's thinning).
        
        Args:
            covariates_df: DataFrame with columns ['time', 'S', 'q1', 'Q10']
            T: Total simulation time
            random_state: Random seed for reproducibility
            
        Returns:
            Array of simulated event times
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation")
            
        if random_state is not None:
            np.random.seed(random_state)
        
        Q_col = 'q1' if self.model_type == 'market' else 'Q10'
        
        # Create piecewise-constant series with higher resolution
        max_time = int(np.ceil(T))
        time_resolution = max(1, max_time // 1000)  # Higher resolution for better accuracy
        n_points = max_time * time_resolution + 1
        
        S_series = np.zeros(n_points)
        Q_series = np.zeros(n_points)
        
        # Fill in piecewise-constant values with interpolation
        for i in range(len(covariates_df)):
            start_time = covariates_df.iloc[i]['time']
            end_time = covariates_df.iloc[i+1]['time'] if i < len(covariates_df)-1 else T
            
            start_idx = int(start_time * time_resolution)
            end_idx = min(int(end_time * time_resolution), n_points)
            
            if start_idx < n_points:
                S_series[start_idx:end_idx] = covariates_df.iloc[i]['S']
                Q_series[start_idx:end_idx] = covariates_df.iloc[i][Q_col]
        
        # Thinning algorithm with timeout protection
        t = 0
        events = []
        
        # Compute maximum intensity for thinning
        lambda_max = self.intensity(S_series, Q_series).max() * 1.5  # Add buffer
        
        max_iterations = int(T * lambda_max * 10)  # Safety limit
        iteration_count = 0
        
        while t < T and iteration_count < max_iterations:
            iteration_count += 1
            
            # Generate proposal time from homogeneous Poisson with rate lambda_max
            t += np.random.exponential(1.0 / lambda_max)
            
            if t >= T:
                break
            
            # Get current intensity using higher resolution
            t_idx = min(int(t * time_resolution), len(S_series) - 1)
            lambda_t = self.intensity(
                np.array([S_series[t_idx]]),
                np.array([Q_series[t_idx]])
            )[0]
            
            # Accept/reject
            if np.random.uniform() <= lambda_t / lambda_max:
                events.append(t)
        
        if iteration_count >= max_iterations:
            print(f"Warning: Simulation stopped due to iteration limit ({max_iterations})")
        
        return np.array(events)


class MarketOrderIntensityModel(IntensityModel):
    """Market order intensity model λ_M(t; S, q1)"""
    def __init__(self):
        super().__init__(model_type='market')


class LimitOrderIntensityModel(IntensityModel):
    """Limit order intensity model λ_L(t; S, Q10)"""
    def __init__(self):
        super().__init__(model_type='limit')


# Example usage
if __name__ == "__main__":
    # Sample data
    events = pd.DataFrame({
        'time': [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0],
        'event_type': ['market', 'limit', 'market', 'limit', 'market', 'limit', 'cancel']
    })
    
    covariates = pd.DataFrame({
        'time': [0.0, 2.0, 5.0, 8.0],
        'S': [0.01, 0.015, 0.012, 0.013],
        'q1': [100, 150, 120, 140],
        'Q10': [1000, 1500, 1200, 1400]
    })
    
    # Fit market order model
    market_model = MarketOrderIntensityModel()
    beta_market = market_model.fit(events, covariates)
    print("Market Order Model Parameters (β):", beta_market)
    
    # Fit limit order model
    limit_model = LimitOrderIntensityModel()
    beta_limit = limit_model.fit(events, covariates)
    print("Limit Order Model Parameters (β):", beta_limit)
    
    # Simulate events
    simulated_events = market_model.simulate_events(covariates, T=10.0, random_state=42)
    print(f"\nSimulated {len(simulated_events)} market order events")
