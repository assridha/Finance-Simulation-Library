import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
from .base import OptionSimulator, OptionStrategy, OptionContract
from ..stock_simulator.models.gbm import GBMModel
from datetime import datetime

class MonteCarloOptionSimulator(OptionSimulator):
    """Monte Carlo simulator for option strategies using various price models."""
    
    def __init__(self, 
                 strategy: OptionStrategy,
                 price_model: str = 'gbm',
                 volatility: Optional[float] = None,
                 risk_free_rate: Optional[float] = None):
        super().__init__(strategy)
        self.price_model = price_model
        
        # Get initial parameters from the first option contract
        if strategy.positions:
            first_contract = strategy.positions[0].contract
            self.symbol = first_contract.symbol
            self.initial_price = first_contract.underlying_price
            self.volatility = volatility or first_contract.implied_volatility
            self.risk_free_rate = risk_free_rate or first_contract.risk_free_rate
            
            # Initialize the stock price model
            self.stock_model = GBMModel(
                ticker=self.symbol,
                volatility=self.volatility,
                risk_free_rate=self.risk_free_rate
            )
    
    def run_simulation(self, num_paths: int = 1000, num_steps: int = 252) -> Dict[str, np.ndarray]:
        """Run the Monte Carlo simulation for the option strategy.
        
        Args:
            num_paths: Number of price paths to simulate
            num_steps: Number of time steps per path
            
        Returns:
            Dictionary containing simulation results:
                - price_paths: Array of simulated price paths
                - strategy_values: Array of strategy values along each path
                - time_steps: Array of time points
        """
        # Get time to expiry from the first option contract
        if not self.strategy.positions:
            raise ValueError("No positions in strategy")
            
        first_contract = self.strategy.positions[0].contract
        time_to_expiry = (first_contract.expiration_date - datetime.now()).days / 365
        
        # Simulate price paths
        price_paths = self.simulate_price_paths(num_paths, num_steps, time_to_expiry)
        
        # Create time steps array
        time_steps = np.linspace(0, time_to_expiry, num_steps)
        
        # Calculate option prices
        option_prices = self.calculate_option_prices(price_paths, time_steps)
        
        # Calculate strategy values
        strategy_values = self.calculate_strategy_values(price_paths, option_prices)
        
        return {
            'price_paths': price_paths,
            'strategy_values': strategy_values,
            'time_steps': time_steps
        }
    
    def simulate_price_paths(self, 
                           num_paths: int, 
                           num_steps: int, 
                           time_to_expiry: float) -> np.ndarray:
        """Simulate price paths using the stock simulator's GBM model."""
        # Convert time_to_expiry to days
        days_to_simulate = int(time_to_expiry * 365)
        
        # Run simulation using the stock model
        results = self.stock_model.simulate(
            days_to_simulate=days_to_simulate,
            num_simulations=num_paths
        )
        
        # Get price paths and time points
        price_paths = results['price_paths']
        time_points = results['time_points']
        
        # Resample to match requested number of steps
        if len(time_points) != num_steps:
            # Create evenly spaced time points
            new_time_points = np.linspace(0, time_to_expiry, num_steps)
            
            # Interpolate price paths to match new time points
            new_price_paths = np.zeros((num_paths, num_steps))
            for path in range(num_paths):
                new_price_paths[path] = np.interp(
                    new_time_points,
                    time_points,
                    price_paths[path]
                )
            return new_price_paths
        
        return price_paths
    
    def calculate_option_prices(self, 
                              price_paths: np.ndarray, 
                              time_steps: np.ndarray) -> np.ndarray:
        """Calculate option prices using Black-Scholes formula."""
        num_paths, num_steps = price_paths.shape
        option_prices = np.zeros((num_paths, num_steps))
        
        for t in range(num_steps):
            time_to_expiry = max(0.0001, time_steps[-1] - time_steps[t])  # Avoid zero time to expiry
            for path in range(num_paths):
                S = price_paths[path, t]
                for pos in self.strategy.positions:
                    K = pos.contract.strike_price
                    r = self.risk_free_rate
                    sigma = self.volatility
                    q = pos.contract.dividend_yield
                    
                    if pos.contract.option_type == 'call':
                        option_prices[path, t] += 100*pos.quantity * self._black_scholes_call(
                            S, K, time_to_expiry, r, sigma, q
                        )
                    else:
                        option_prices[path, t] += 100*pos.quantity * self._black_scholes_put(
                            S, K, time_to_expiry, r, sigma, q
                        )
        
        return option_prices
    
    def calculate_strategy_values(self, 
                                price_paths: np.ndarray, 
                                option_prices: np.ndarray) -> np.ndarray:
        """Calculate the total strategy value along each price path."""
        num_paths, num_steps = price_paths.shape
        strategy_values = np.zeros((num_paths, num_steps))
        
        for t in range(num_steps):
            for path in range(num_paths):
                # Calculate stock position values
                for stock_pos in self.strategy.stock_positions:
                    strategy_values[path, t] += stock_pos.quantity * (price_paths[path, t])
                
                # Add option position values
                strategy_values[path, t] += option_prices[path, t]
        
        return strategy_values
    
    def _black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
        """Calculate call option price using Black-Scholes formula."""
        if T <= 0:
            return max(0, S - K)
            
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def _black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
        """Calculate put option price using Black-Scholes formula."""
        if T <= 0:
            return max(0, K - S)
            
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return put_price 