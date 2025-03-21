import numpy as np
from financial_sim_library.stock_simulator.models.gbm import GBMModel
from financial_sim_library.stock_simulator.models.growth_models import (
    FixedGrowthModel,
    PowerLawGrowthModel,
    ExogenousGrowthModel,
    CompositeGrowthModel
)
from financial_sim_library.visualization.price_plots import (
    plot_price_simulations,
    plot_price_distribution
)

def run_growth_model_examples():
    """Run examples demonstrating different growth models with GBM simulation."""
    print("\n" + "="*80)
    print("GROWTH MODEL EXAMPLES")
    print("="*80)
    
    # Example 1: Fixed Growth Model
    print("\nExample 1: Fixed Growth Model (10% annual growth)")
    fixed_growth = FixedGrowthModel(growth_rate=0.1)
    model = GBMModel(ticker="AAPL", growth_model=fixed_growth)
    results = model.simulate()
    print(f"Expected return: {results['statistics']['expected_return']:.2f}%")
    plot_price_simulations(results, title="GBM with Fixed Growth (10% annual)")
    plot_price_distribution(results, title="Price Distribution - Fixed Growth")
    
    # Example 2: Power Law Growth Model
    print("\nExample 2: Power Law Growth Model")
    power_growth = PowerLawGrowthModel(k=0.1)
    model = GBMModel(ticker="AAPL", growth_model=power_growth)
    results = model.simulate()
    print(f"Expected return: {results['statistics']['expected_return']:.2f}%")
    plot_price_simulations(results, title="GBM with Power Law Growth")
    plot_price_distribution(results, title="Price Distribution - Power Law Growth")
    
    # Example 3: Exogenous Growth Model (Oscillating Growth)
    print("\nExample 3: Exogenous Growth Model (Oscillating Growth)")
    def oscillating_growth(t, dt, amplitude=0.1, frequency=2*np.pi, **kwargs):
        return amplitude * np.sin(frequency * t)
    
    exogenous_growth = ExogenousGrowthModel(growth_function=oscillating_growth)
    model = GBMModel(ticker="AAPL", growth_model=exogenous_growth)
    results = model.simulate(amplitude=0.1, frequency=2*np.pi)
    print(f"Expected return: {results['statistics']['expected_return']:.2f}%")
    plot_price_simulations(results, title="GBM with Oscillating Growth")
    plot_price_distribution(results, title="Price Distribution - Oscillating Growth")
    
    # Example 4: Composite Growth Model
    print("\nExample 4: Composite Growth Model (Fixed + Power Law)")
    composite_growth = CompositeGrowthModel(
        models={
            'fixed': FixedGrowthModel(growth_rate=0.1),
            'power': PowerLawGrowthModel(k=0.1)
        },
        weights={'fixed': 0.6, 'power': 0.4}
    )
    model = GBMModel(ticker="AAPL", growth_model=composite_growth)
    results = model.simulate()
    print(f"Expected return: {results['statistics']['expected_return']:.2f}%")
    plot_price_simulations(results, title="GBM with Composite Growth")
    plot_price_distribution(results, title="Price Distribution - Composite Growth")
    
    # Example 5: Complex Composite Model with Exogenous Variables
    print("\nExample 5: Complex Composite Model with Exogenous Variables")
    def market_cycle_growth(t, dt, phase=0, **kwargs):
        """Growth rate that follows market cycles."""
        return 0.1 * np.sin(2*np.pi*t + phase)
    
    complex_growth = CompositeGrowthModel(
        models={
            'fixed': FixedGrowthModel(growth_rate=0.1),
            'power': PowerLawGrowthModel(k=0.1),
            'cycle': ExogenousGrowthModel(growth_function=market_cycle_growth)
        },
        weights={'fixed': 0.4, 'power': 0.3, 'cycle': 0.3}
    )
    model = GBMModel(ticker="AAPL", growth_model=complex_growth)
    results = model.simulate(phase=np.pi/4)  # Start at 45 degrees in the cycle
    print(f"Expected return: {results['statistics']['expected_return']:.2f}%")
    plot_price_simulations(results, title="GBM with Complex Composite Growth")
    plot_price_distribution(results, title="Price Distribution - Complex Growth")
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    run_growth_model_examples() 