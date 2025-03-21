import unittest
import numpy as np
from ..stock_simulator.models.growth_models import (
    FixedGrowthModel,
    PowerLawGrowthModel,
    ExogenousGrowthModel,
    CompositeGrowthModel
)

class TestGrowthModels(unittest.TestCase):
    """Test cases for growth models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dt = 1/252  # Daily time step
        self.t = np.arange(0, 1, self.dt)  # One year of daily steps
    
    def test_fixed_growth_model(self):
        """Test the fixed growth model."""
        model = FixedGrowthModel(growth_rate=0.1)  # 10% annual growth
        
        # Test growth rate calculation
        growth_rate = model.calculate_growth_rate(0, self.dt)
        expected_growth = 0.1 * self.dt
        self.assertAlmostEqual(growth_rate, expected_growth)
        
        # Test parameter get/set
        params = model.get_parameters()
        self.assertEqual(params['growth_rate'], 0.1)
        
        model.set_parameters(growth_rate=0.15)
        self.assertEqual(model.growth_rate, 0.15)
    
    def test_power_law_growth_model(self):
        """Test the power law growth model."""
        model = PowerLawGrowthModel(k=0.1)
        
        # Test growth rate calculation
        growth_rate = model.calculate_growth_rate(0.5, self.dt)
        expected_growth = 0.1 * np.log(1 + self.dt/0.5)
        self.assertAlmostEqual(growth_rate, expected_growth)
        
        # Test behavior at t=0
        growth_rate_zero = model.calculate_growth_rate(0, self.dt)
        self.assertEqual(growth_rate_zero, 0.0)
        
        # Test parameter get/set
        params = model.get_parameters()
        self.assertEqual(params['k'], 0.1)
        
        model.set_parameters(k=0.15)
        self.assertEqual(model.k, 0.15)
    
    def test_exogenous_growth_model(self):
        """Test the exogenous growth model."""
        def custom_growth(t, dt, amplitude=0.1, **kwargs):
            return amplitude * np.sin(t)  # Example: oscillating growth rate
        
        model = ExogenousGrowthModel(growth_function=custom_growth)
        
        # Test growth rate calculation
        growth_rate = model.calculate_growth_rate(0.5, self.dt)
        expected_growth = 0.1 * np.sin(0.5)
        self.assertAlmostEqual(growth_rate, expected_growth)
        
        # Test with additional parameters
        growth_rate_with_params = model.calculate_growth_rate(0.5, self.dt, amplitude=0.2)
        expected_growth_with_params = 0.2 * np.sin(0.5)
        self.assertAlmostEqual(growth_rate_with_params, expected_growth_with_params)
        
        # Test parameter get/set
        params = model.get_parameters()
        self.assertEqual(params['growth_function'], custom_growth)
        
        def new_growth(t, dt, **kwargs):
            return 0.2 * np.cos(t)
        
        model.set_parameters(growth_function=new_growth)
        self.assertEqual(model.growth_function, new_growth)
    
    def test_composite_growth_model(self):
        """Test the composite growth model."""
        # Create individual models
        fixed_model = FixedGrowthModel(growth_rate=0.1)
        power_model = PowerLawGrowthModel(k=0.1)
        
        # Create composite model
        models = {
            'fixed': fixed_model,
            'power': power_model
        }
        weights = {'fixed': 0.6, 'power': 0.4}
        model = CompositeGrowthModel(models, weights)
        
        # Test growth rate calculation
        growth_rate = model.calculate_growth_rate(0.5, self.dt)
        expected_growth = (
            0.6 * fixed_model.calculate_growth_rate(0.5, self.dt) +
            0.4 * power_model.calculate_growth_rate(0.5, self.dt)
        )
        self.assertAlmostEqual(growth_rate, expected_growth)
        
        # Test parameter get/set
        params = model.get_parameters()
        self.assertEqual(params['models'], models)
        self.assertEqual(params['weights'], weights)
        
        # Test weight normalization
        new_weights = {'fixed': 1.0, 'power': 1.0}
        model.set_parameters(weights=new_weights)
        self.assertAlmostEqual(model.weights['fixed'], 0.5)
        self.assertAlmostEqual(model.weights['power'], 0.5)
    
    def test_growth_model_combinations(self):
        """Test combining different growth models."""
        # Create a complex composite model
        fixed_model = FixedGrowthModel(growth_rate=0.1)
        power_model = PowerLawGrowthModel(k=0.1)
        
        def custom_growth(t, dt, **kwargs):
            return 0.1 * np.sin(t)
        
        exogenous_model = ExogenousGrowthModel(growth_function=custom_growth)
        
        # Create nested composite models
        inner_composite = CompositeGrowthModel(
            {'fixed': fixed_model, 'power': power_model},
            {'fixed': 0.6, 'power': 0.4}
        )
        
        outer_composite = CompositeGrowthModel(
            {'inner': inner_composite, 'exogenous': exogenous_model},
            {'inner': 0.7, 'exogenous': 0.3}
        )
        
        # Test the nested composite model
        growth_rate = outer_composite.calculate_growth_rate(0.5, self.dt)
        self.assertIsInstance(growth_rate, float)
        self.assertGreater(growth_rate, -float('inf'))
        self.assertLess(growth_rate, float('inf'))

if __name__ == '__main__':
    unittest.main() 