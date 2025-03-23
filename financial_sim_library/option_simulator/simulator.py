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
            # Only use provided volatility or first contract IV for stock price simulation
            self.stock_volatility = volatility or first_contract.implied_volatility
            self.risk_free_rate = risk_free_rate or first_contract.risk_free_rate
            
            # Initialize the stock price model
            self.stock_model = GBMModel(
                ticker=self.symbol,
                volatility=self.stock_volatility,
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
                              time_steps: np.ndarray) -> dict:
        """Calculate option prices using vectorized Black-Scholes formula."""
        num_paths, num_steps = price_paths.shape
        option_prices = {}  # Dictionary to store prices for each option position
        
        # Debug container to store premiums
        debug_info = {}
        
        # Process time steps in batches for better performance
        for t in range(num_steps):
            time_to_expiry = max(0.0001, time_steps[-1] - time_steps[t])  # Avoid zero time to expiry
            r = self.risk_free_rate
            
            # Process all paths at once (vectorized)
            S = price_paths[:, t].reshape(-1, 1)  # Reshape for broadcasting
            
            # Calculate prices for each option position separately
            for i, pos in enumerate(self.strategy.positions):
                contract = pos.contract
                # Create a unique key for each position
                position_key = f"pos_{i}"
                
                if position_key not in option_prices:
                    option_prices[position_key] = np.zeros((num_paths, num_steps))
                
                K = contract.strike_price
                q = contract.dividend_yield
                sigma = contract.implied_volatility  # Use option-specific IV
                
                # Debug info - store important values
                if t == 0 and position_key not in debug_info:
                    intrinsic = 0
                    if contract.option_type == 'call':
                        intrinsic = max(0, contract.underlying_price - K)
                    else:
                        intrinsic = max(0, K - contract.underlying_price)
                    debug_info[position_key] = {
                        'strike': K,
                        'premium': contract.premium,
                        'intrinsic': intrinsic,
                        'time_value': contract.premium - intrinsic,
                        'type': contract.option_type,
                        'iv': sigma
                    }
                    
                    # Print debug info
                    print(f"\nOption {position_key} - {contract.option_type.upper()} {K}:")
                    print(f"  Premium: ${contract.premium:.2f}")
                    print(f"  Intrinsic value: ${intrinsic:.2f}")
                    print(f"  Time value: ${contract.premium - intrinsic:.2f}")
                    print(f"  IV: {sigma*100:.2f}%")
                
                if contract.option_type == 'call':
                    # Vectorized Black-Scholes for calls
                    values = self._vectorized_black_scholes(
                        S.flatten(), K, time_to_expiry, r, sigma, q, is_call=True
                    )
                else:
                    # Vectorized Black-Scholes for puts
                    values = self._vectorized_black_scholes(
                        S.flatten(), K, time_to_expiry, r, sigma, q, is_call=False
                    )
                
                # For t=0, make sure we're using the actual premium instead of calculated value
                # This ensures the initial value is correct, especially for ITM options
                if t == 0:
                    option_prices[position_key][:, t] = pos.entry_price
                    
                    # Debug information - print expected vs. actual values
                    print(f"  Model price at t=0: ${values[0]:.2f}, Using market price: ${pos.entry_price:.2f}")
                else:
                    option_prices[position_key][:, t] = values
        
        return option_prices
    
    def _vectorized_black_scholes(self, S, K, T, r, sigma, q, is_call=True):
        """Vectorized Black-Scholes formula calculation for multiple prices at once."""
        # Calculate intrinsic value
        if is_call:
            intrinsic_value = np.maximum(0, S - K)
        else:
            intrinsic_value = np.maximum(0, K - S)
        
        # At expiration, option is worth only intrinsic value
        if T <= 0:
            return intrinsic_value
        
        # For very small T values, numerical issues can arise
        # Add extra precision for very small T
        T = np.maximum(T, 1e-8)
        
        # For deep ITM or OTM options, adjust calculation to avoid numerical issues
        # For very large or small d1/d2 values, numerical issues can arise
        try:
            # Calculate d1 and d2 (vectorized)
            d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Handle extreme values to prevent numerical issues
            d1 = np.clip(d1, -10, 10)  # Limit to reasonable range
            d2 = np.clip(d2, -10, 10)  # Limit to reasonable range
            
            # SciPy's norm.cdf is already vectorized
            if is_call:
                time_value = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                # Ensure option price is at least intrinsic value (prevent tiny precision errors)
                option_prices = np.maximum(time_value, intrinsic_value)
            else:
                time_value = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
                # Ensure option price is at least intrinsic value (prevent tiny precision errors)
                option_prices = np.maximum(time_value, intrinsic_value)
            
            return option_prices
        except Exception as e:
            # If any numerical issues occur, return the intrinsic value as a fallback
            print(f"Warning: Black-Scholes calculation error: {e}. Using intrinsic value.")
            return intrinsic_value
    
    def calculate_strategy_values(self, 
                                price_paths: np.ndarray, 
                                option_prices: dict) -> np.ndarray:
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
        
        # Debug information
        print(f"\nStrategy Initial Value: ${initial_value:.2f}")
        for i, pos in enumerate(self.strategy.positions):
            print(f"  Position {i}: {pos.quantity} contracts @ ${pos.entry_price:.2f} = ${pos.quantity * pos.entry_price * 100:.2f}")
        
        # Set initial value for all paths
        strategy_values[:, 0] = initial_value
        
        # Create array to hold the current market value of each position at each time step
        # This will help with debugging
        current_market_values = np.zeros((len(self.strategy.positions), num_paths, num_steps))
        
        # Fill the initial values (t=0)
        for i, pos in enumerate(self.strategy.positions):
            position_key = f"pos_{i}"
            current_market_values[i, :, 0] = pos.entry_price * pos.quantity * 100
        
        for t in range(1, num_steps):  # Start from t=1 since t=0 is initial value
            for path in range(num_paths):
                # Calculate the current market value of the entire strategy
                current_value = 0.0
                
                # Add stock position values
                for stock_pos in self.strategy.stock_positions:
                    current_stock_value = stock_pos.quantity * price_paths[path, t]
                    current_value += current_stock_value
                
                # Add option position values
                for i, pos in enumerate(self.strategy.positions):
                    position_key = f"pos_{i}" 
                    current_option_value = option_prices[position_key][path, t]
                    
                    # Value of the position is (premium * quantity * 100)
                    position_value = current_option_value * pos.quantity * 100
                    current_value += position_value
                    
                    # Store for debugging
                    current_market_values[i, path, t] = position_value
                
                # Calculate P&L as (current_value - initial_value)
                strategy_values[path, t] = current_value - initial_value
                
                # For path 0, if t is near the end, print some debug values
                if path == 0 and t == num_steps - 1:
                    print(f"\nAt end of simulation (path 0):")
                    print(f"  Final stock price: ${price_paths[0, t]:.2f}")
                    for i, pos in enumerate(self.strategy.positions):
                        position_key = f"pos_{i}"
                        print(f"  Position {i}: {pos.quantity} contracts, value = ${current_market_values[i, 0, t]:.2f}")
                    print(f"  Strategy P&L: ${strategy_values[0, t]:.2f}")
        
        return strategy_values 