from financial_sim_library.stock_simulator.models.gbm import GBMModel
from financial_sim_library.visualization.price_plots import plot_price_simulations

# Create a GBM model for Apple stock
model = GBMModel(ticker="MSTR")

# Run a simulation
results = model.simulate(
    days_to_simulate=30,
    num_simulations=1000
)

# Print key statistics
stats = results['statistics']
print(f"Current price: ${results['current_price']:.2f}")
print(f"Expected price after 30 days: ${stats['mean']:.2f}")
print(f"Expected return: {stats['expected_return']:.2f}%")
print(f"Probability of price increase: {stats['prob_above_current']:.2f}%")

# Visualize the results
plot_price_simulations(results)