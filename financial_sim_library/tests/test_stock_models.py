"""
Stock Model Tests

This module contains tests for the stock price simulation models.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_sim_library.stock_simulator.models.gbm import GBMModel

class TestGBMModel(unittest.TestCase):
    """Test cases for the GBM model."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a seed for reproducibility
        self.seed = 42
        
        # Set fixed parameters for testing
        self.ticker = "TEST"
        self.current_price = 100.0
        self.volatility = 0.2
        self.risk_free_rate = 0.03
        
        # Create model instance with fixed parameters
        self.model = GBMModel(
            ticker=self.ticker,
            current_price=self.current_price,
            volatility=self.volatility,
            risk_free_rate=self.risk_free_rate,
            seed=self.seed
        )
    
    def test_init(self):
        """Test model initialization."""
        self.assertEqual(self.model.ticker, self.ticker)
        self.assertEqual(self.model.current_price, self.current_price)
        self.assertEqual(self.model.volatility, self.volatility)
        self.assertEqual(self.model.risk_free_rate, self.risk_free_rate)
        self.assertEqual(self.model.seed, self.seed)
    
    def test_validate_inputs(self):
        """Test input validation."""
        # Test invalid ticker
        with self.assertRaises(TypeError):
            GBMModel(ticker=123)
        
        # Test invalid current_price
        with self.assertRaises(TypeError):
            GBMModel(ticker="TEST", current_price="100")
        
        # Test negative current_price
        with self.assertRaises(ValueError):
            GBMModel(ticker="TEST", current_price=-100)
        
        # Test invalid volatility
        with self.assertRaises(ValueError):
            GBMModel(ticker="TEST", current_price=100, volatility=-0.2)
        
        # Test volatility > 1
        with self.assertRaises(ValueError):
            GBMModel(ticker="TEST", current_price=100, volatility=1.5)
        
        # Test negative risk_free_rate
        with self.assertRaises(ValueError):
            GBMModel(ticker="TEST", current_price=100, volatility=0.2, risk_free_rate=-0.03)
    
    def test_calibrate(self):
        """Test model calibration."""
        # Create mock historical data
        dates = pd.date_range(start='2023-01-01', periods=30)
        prices = np.array([100] * 30) * np.exp(np.cumsum(np.random.normal(0, 0.015, 30)))
        hist_data = pd.DataFrame({'Close': prices}, index=dates)
        
        # Create new model without volatility and risk_free_rate
        model = GBMModel(ticker=self.ticker, current_price=self.current_price)
        
        # Set these to None to force calibration
        model.volatility = None
        model.risk_free_rate = None
        
        # Calibrate with the mock data
        result = model.calibrate(hist_data)
        
        # Check that calibration sets the parameters
        self.assertIsNotNone(model.volatility)
        self.assertIsNotNone(model.risk_free_rate)
        
        # Check that result contains the parameters
        self.assertIn('volatility', result)
        self.assertIn('risk_free_rate', result)
        self.assertIn('ticker', result)
        self.assertIn('current_price', result)
    
    def test_simulate(self):
        """Test simulation generation."""
        days_to_simulate = 30
        num_simulations = 1000
        
        # Run simulation
        results = self.model.simulate(
            days_to_simulate=days_to_simulate,
            num_simulations=num_simulations
        )
        
        # Check results structure
        self.assertIn('ticker', results)
        self.assertIn('current_price', results)
        self.assertIn('price_paths', results)
        self.assertIn('dates', results)
        self.assertIn('parameters', results)
        self.assertIn('statistics', results)
        
        # Check dimensions of price paths
        price_paths_shape = results['price_paths'].shape
        self.assertEqual(price_paths_shape[0], num_simulations, "Number of simulations doesn't match")
        # Either the simulation creates days_to_simulate + 1 points (including initial price)
        # or days_to_simulate points (trading days, not including initial)
        self.assertIn(price_paths_shape[1], [days_to_simulate, days_to_simulate + 1], 
                     f"Expected {days_to_simulate} or {days_to_simulate + 1} days, got {price_paths_shape[1]}")
        
        # Check that all paths start with the current price
        np.testing.assert_array_equal(
            results['price_paths'][:, 0],
            np.array([self.current_price] * num_simulations)
        )
        
        # Check statistics
        self.assertIn('mean', results['statistics'])
        self.assertIn('median', results['statistics'])
        self.assertIn('std', results['statistics'])
        self.assertIn('min', results['statistics'])
        self.assertIn('max', results['statistics'])
        self.assertIn('percentiles', results['statistics'])
    
    def test_analyze_results(self):
        """Test the analyze_results method."""
        # Generate a simple simulation result
        days_to_simulate = 10
        num_simulations = 100
        
        # Create mock price paths
        np.random.seed(self.seed)
        price_paths = np.zeros((num_simulations, days_to_simulate + 1))
        price_paths[:, 0] = self.current_price
        for i in range(1, days_to_simulate + 1):
            price_paths[:, i] = price_paths[:, i-1] * np.exp(
                np.random.normal(0.001, 0.02, num_simulations))
        
        # Create mock dates
        start_date = datetime.now()
        dates = [start_date + timedelta(days=i) for i in range(days_to_simulate + 1)]
        
        # Create mock simulation results
        simulation_results = {
            'ticker': self.ticker,
            'current_price': self.current_price,
            'price_paths': price_paths,
            'dates': dates,
            'parameters': {
                'volatility': self.volatility,
                'risk_free_rate': self.risk_free_rate
            }
        }
        
        # Analyze results
        analyzed_results = self.model.analyze_results(simulation_results)
        
        # Check that statistics were added
        self.assertIn('statistics', analyzed_results)
        stats = analyzed_results['statistics']
        
        # Check that expected keys exist
        expected_keys = ['mean', 'median', 'std', 'min', 'max', 'percentiles', 
                        'expected_return', 'prob_above_current', 'prob_10pct_up', 
                        'prob_10pct_down']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check that percentiles include expected quantiles
        for percentile in ['10', '25', '50', '75', '90']:
            self.assertIn(percentile, stats['percentiles'])

if __name__ == '__main__':
    unittest.main() 