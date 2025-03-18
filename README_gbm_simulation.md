# Geometric Brownian Motion Stock Price Simulator

This tool provides functionality to simulate stock price paths using the Geometric Brownian Motion (GBM) model, commonly used in financial mathematics to model stock price movements.

## Features

- **Monte Carlo Simulation**: Generates multiple price paths to estimate future price distributions
- **Historical Volatility**: Uses recent price data to calculate volatility
- **Statistical Analysis**: Provides detailed statistics about simulated prices
- **Benchmark Comparison**: Compare a stock's simulated performance against an index (e.g., SPY)
- **Interactive Visualization**: View price paths, confidence intervals, and relative performance

## Mathematical Background

The Geometric Brownian Motion model is described by the stochastic differential equation:

$$dS = \mu S dt + \sigma S dW$$

Where:
- $S$ is the stock price
- $\mu$ is the drift (expected return)
- $\sigma$ is the volatility
- $dW$ is a Wiener process (random term)

For simulation, we use the discretized version:

$$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z \right)$$

Where $Z$ is a standard normal random variable.

## Usage

### Basic Usage

```bash
python run_gbm_simulation.py AAPL
```

### Advanced Options

```bash
python run_gbm_simulation.py AAPL --days 365 --simulations 1000 --benchmark SPY
```

Parameters:
- `ticker`: Stock symbol to simulate (required)
- `--days`: Number of days to simulate (default: 180)
- `--simulations`: Number of simulation paths to generate (default: 1000)
- `--benchmark`: Compare performance against this ticker (optional)
- `--save`: Path to save the plot (optional)

## Example Output

The simulation provides:

1. **Price Target Analysis**: Percentile-based price targets (10th, 25th, 50th, 75th, 90th)
2. **Benchmark Comparison**: Expected returns and probability of outperformance
3. **Probability Analysis**: Likelihood of different price scenarios
4. **Visualization**: Price paths and confidence intervals

## Integration with Option Simulator

The GBM simulation functionality is also integrated into the Option Simulator. When running an option simulation, the GBM model is used to generate realistic price paths for the underlying stock, which can be used to better understand the probability distribution of option PnL.

### Example in Option Simulation

```python
# Import and use the simulation function 
from option_simulator import simulate_geometric_brownian_motion, plot_price_simulations

# Run a simulation
gbm_results = simulate_geometric_brownian_motion(
    ticker="AAPL",
    current_price=150.0,
    days_to_simulate=180,
    num_simulations=100
)

# Plot the simulation results
plot_price_simulations(gbm_results, num_paths_to_plot=10)
```

## Benefits of the GBM Model

- **Realistic Price Distributions**: Models the non-negative and log-normal distribution of stock prices
- **Consistent with Option Pricing Theory**: Same model underlying Black-Scholes
- **Business Days Only**: Simulates price changes only on trading days
- **Volatility Clustering**: Captures the tendency of volatility to vary over time

## Limitations

- Assumes constant volatility (not accounting for volatility clustering)
- Assumes returns follow a normal distribution (may not capture fat tails)
- Does not model jumps in price (can be significant for some stocks)
- Past volatility may not represent future volatility

## Future Enhancements

Potential improvements to consider:
- Implement GARCH models for time-varying volatility
- Add jump-diffusion processes for modeling price jumps
- Include mean-reversion for certain assets
- Implement correlated simulations for multiple assets 