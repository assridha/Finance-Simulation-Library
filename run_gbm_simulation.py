from financial_sim_library.stock_simulator.models.gbm import GBMModel
from financial_sim_library.visualization.price_plots import plot_price_simulations
from financial_sim_library.stock_simulator.models.growth_models import FixedGrowthModel
import numpy as np
# Create a fixed growth model with 130% annual growth rate
fixed_growth = FixedGrowthModel(growth_rate=1.3)

# Create a GBM model for Apple stock with the fixed growth model
model = GBMModel(ticker="IBIT", growth_model=fixed_growth)

days_to_simulate = 250

# Run a simulation
results = model.simulate(
    days_to_simulate=days_to_simulate,
    num_simulations=1000
)
print(np.shape(results['price_paths']))
print(np.shape(results['time_points']))
# Print key statistics
stats = results['statistics']
print(f"Current price: ${results['current_price']:.2f}")
print(f"Expected price after {days_to_simulate} days: ${stats['mean']:.2f}")
print(f"Expected return: {stats['expected_return']:.2f}%")
print(f"Probability of price increase: {stats['prob_above_current']:.2f}%")

# Visualize the results
plot_price_simulations(results)