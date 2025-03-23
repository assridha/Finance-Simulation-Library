import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
from .strategies.base import OptionSimulator, OptionStrategy, OptionContract
from ..stock_simulator.models.gbm import GBMModel
from datetime import datetime
from scipy.optimize import minimize_scalar

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
            # Only use provided volatility or calculate from first contract
            self.stock_volatility = volatility or self._calculate_implied_volatility(
                first_contract.premium, 
                first_contract.underlying_price,
                first_contract.strike_price,
                first_contract.time_to_expiry,
                first_contract.risk_free_rate,
                first_contract.option_type,
                first_contract.dividend_yield
            )
            self.risk_free_rate = risk_free_rate or first_contract.risk_free_rate
            
            # Initialize the stock price model
            self.stock_model = GBMModel(
                ticker=self.symbol,
                volatility=self.stock_volatility,
                risk_free_rate=self.risk_free_rate
            )
    
    def _calculate_implied_volatility(self, market_price, S, K, T, r, option_type='call', q=0):
        """Calculate implied volatility using Black-Scholes and market price."""
        # Define the objective function to minimize
        def objective(sigma):
            bs_price = self._black_scholes(S, K, T, r, sigma, option_type, q)
            return (bs_price - market_price) ** 2
        
        # Use a bounded optimization to find implied volatility
        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
        
        if result.success:
            return result.x
        else:
            # Default to a reasonable IV if calculation fails
            return 0.3
    
    def _black_scholes(self, S, K, T, r, sigma, option_type='call', q=0):
        """Calculate Black-Scholes price for a single option."""
        if T <= 0:
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
                
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return price
    
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
        """Simulate price paths using vectorized operations."""
        # Convert time_to_expiry to days
        days_to_simulate = int(time_to_expiry * 365)
        
        # Set a minimum number of simulations to run for efficiency
        days_to_simulate = max(days_to_simulate, 30)
        
        # Run simulation using the stock model with cached results
        model_key = f"{self.symbol}_{days_to_simulate}_{num_paths}"
        if hasattr(self, 'cached_simulations') and model_key in self.cached_simulations:
            results = self.cached_simulations[model_key]
        else:
            results = self.stock_model.simulate(
                days_to_simulate=days_to_simulate,
                num_simulations=num_paths
            )
            # Create cache if it doesn't exist
            if not hasattr(self, 'cached_simulations'):
                self.cached_simulations = {}
            self.cached_simulations[model_key] = results
        
        # Get price paths and time points
        price_paths = results['price_paths']
        time_points = results['time_points']
        
        # Resample to match requested number of steps 
        if len(time_points) != num_steps:
            # Create evenly spaced time points
            new_time_points = np.linspace(0, time_to_expiry, num_steps)
            
            # Vectorized interpolation for all paths at once
            # Use linear interpolation to improve performance
            x_new = np.linspace(0, len(time_points)-1, num_steps)
            x_old = np.arange(len(time_points))
            
            # Using vectorized operations instead of loop
            new_price_paths = np.zeros((num_paths, num_steps))
            for i in range(num_paths):
                new_price_paths[i] = np.interp(x_new, x_old, price_paths[i])
            
            return new_price_paths
        
        return price_paths
    
    def calculate_option_prices(self, 
                              price_paths: np.ndarray, 
                              time_steps: np.ndarray) -> np.ndarray:
        """Calculate option prices using vectorized Black-Scholes formula."""
        num_paths, num_steps = price_paths.shape
        option_prices = np.zeros((num_paths, num_steps))
        
        # Extract contract parameters
        option_params = []
        for pos in self.strategy.positions:
            # Calculate IV from entry price instead of using the field
            calculated_iv = self._calculate_implied_volatility(
                pos.entry_price,
                pos.contract.underlying_price,
                pos.contract.strike_price,
                pos.contract.time_to_expiry,
                pos.contract.risk_free_rate,
                pos.contract.option_type,
                pos.contract.dividend_yield
            )
            
            option_params.append({
                'quantity': pos.quantity,
                'strike': pos.contract.strike_price,
                'option_type': pos.contract.option_type,
                'dividend_yield': pos.contract.dividend_yield,
                'entry_price': pos.entry_price,
                'implied_volatility': calculated_iv  # Use calculated IV instead
            })
        
        # Process time steps in batches for better performance
        for t in range(num_steps):
            time_to_expiry = max(0.0001, time_steps[-1] - time_steps[t])  # Avoid zero time to expiry
            r = self.risk_free_rate
            
            # Process all paths at once (vectorized)
            S = price_paths[:, t].reshape(-1, 1)  # Reshape for broadcasting
            
            for param in option_params:
                K = param['strike']
                q = param['dividend_yield']
                quantity = param['quantity']
                sigma = param['implied_volatility']  # Use calculated IV
                
                if param['option_type'] == 'call':
                    # Vectorized Black-Scholes for calls
                    option_values = self._vectorized_black_scholes(
                        S.flatten(), K, time_to_expiry, r, sigma, q, is_call=True
                    )
                else:
                    # Vectorized Black-Scholes for puts
                    option_values = self._vectorized_black_scholes(
                        S.flatten(), K, time_to_expiry, r, sigma, q, is_call=False
                    )
                
                # For long positions: loss = current price - entry price (positive means we paid more)
                # For short positions: loss = entry price - current price (positive means we received less)
                if quantity > 0:  # Long position
                    option_prices[:, t] += 100 * quantity * (option_values - param['entry_price'])
                else:  # Short position
                    option_prices[:, t] += 100 * abs(quantity) * (param['entry_price'] - option_values)
        
        return option_prices
    
    def _vectorized_black_scholes(self, S, K, T, r, sigma, q, is_call=True):
        """Vectorized Black-Scholes formula calculation for multiple prices at once."""
        if T <= 0:
            if is_call:
                return np.maximum(0, S - K)
            else:
                return np.maximum(0, K - S)
        
        # Calculate d1 and d2 (vectorized)
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # SciPy's norm.cdf is already vectorized
        if is_call:
            option_prices = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            option_prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return option_prices
    
    def calculate_strategy_values(self, 
                                price_paths: np.ndarray, 
                                option_prices: np.ndarray) -> np.ndarray:
        """Calculate the total strategy value along each price path."""
        num_paths, num_steps = price_paths.shape
        strategy_values = np.zeros((num_paths, num_steps))
        
        # Calculate initial investment/credit
        initial_value = 0.0
        
        # Add stock position initial values (positive for debit/cost)
        for stock_pos in self.strategy.stock_positions:
            initial_value += stock_pos.quantity * stock_pos.entry_price  # Positive because buying stock is a debit
        
        # Add option position initial values
        for pos in self.strategy.positions:
            # For long positions (positive quantity), premium is a debit (positive)
            # For short positions (negative quantity), premium is a credit (negative)
            initial_value += pos.quantity * pos.entry_price * 100  # Times 100 for contract size
        
        # Set initial value for all paths
        strategy_values[:, 0] = initial_value
        
        for t in range(1, num_steps):  # Start from t=1 since t=0 is initial value
            for path in range(num_paths):
                # Start with initial value
                strategy_values[path, t] = strategy_values[path, 0]
                
                # Add stock position P&L
                for stock_pos in self.strategy.stock_positions:
                    # P&L = Current Value - Initial Value
                    stock_value = stock_pos.quantity * (price_paths[path, t] - stock_pos.entry_price)
                    strategy_values[path, t] += stock_value
                
                # Add option position P&L (already calculated as P&L in calculate_option_prices)
                strategy_values[path, t] += option_prices[path, t]
        
        return strategy_values 