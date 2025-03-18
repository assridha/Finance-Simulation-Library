# Option and Stock Simulator

This tool provides three main functionalities:
1. Fetching option data for a given ticker, option type, expiry date, and delta value
2. Simulating the PnL of options trades using the Black-Scholes model
3. Simulating stock price paths using Geometric Brownian Motion (GBM)

## Features

### Option Data Fetcher
- Fetch option data for any publicly traded ticker with options
- Support for both call and put options
- Automatically finds the closest expiry date if the target date is not available
- Automatically finds the option with the closest delta value
- Calculates approximate delta values if not provided by the data source
- Compatible with the latest yfinance library (v0.2.54)
- Includes fallback methods for different yfinance API versions
- Provides underlying price information
- Detailed logging for troubleshooting

### Option Simulator
- Simulate the PnL of buying or selling options using the Black-Scholes model
- Calculate a 2D matrix of PnLs for a range of stock prices and dates
- Generate PnL slices and probability plots to visualize potential outcomes
- Calculate option greeks (delta, gamma, theta, vega)
- Generate detailed reports with trade information and potential outcomes
- Support for both call and put options, and both buying and selling positions
- Simulations always start from today's date and current market prices
- Price range is calculated using implied volatility and 2-standard deviations (95% confidence interval)

### Geometric Brownian Motion (GBM) Stock Simulator
- **Monte Carlo Simulation**: Generates multiple price paths to estimate future price distributions
- **Historical Volatility**: Uses recent price data to calculate volatility
- **Statistical Analysis**: Provides detailed statistics about simulated prices
- **Benchmark Comparison**: Compare a stock's simulated performance against an index (e.g., SPY)
- **Interactive Visualization**: View price paths, confidence intervals, and relative performance

## Mathematical Background

### Black-Scholes Model
The option simulator uses the Black-Scholes model to calculate option prices and greeks.

### Geometric Brownian Motion Model
The GBM model is described by the stochastic differential equation:

$$dS = \mu S dt + \sigma S dW$$

Where:
- $S$ is the stock price
- $\mu$ is the drift (expected return)
- $\sigma$ is the volatility
- $dW$ is a Wiener process (random term)

For simulation, we use the discretized version:

$$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z \right)$$

Where $Z$ is a standard normal random variable.

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option Data Fetcher

#### Basic Usage

```python
from option_fetcher import fetch_option_data

# Fetch a call option for AAPL with delta around 0.5 and expiry close to 2023-12-15
result = fetch_option_data(
    ticker="AAPL",
    option_type="call",
    target_expiry_date="2023-12-15",
    target_delta=0.5
)

# Access the results
print(f"Expiry date: {result['expiry_date']}")
print(f"Delta: {result['delta']}")
print(f"Strike: {result['strike']}")
print(f"Underlying price: {result['underlying_price']}")
print(result['option_data'])  # Full option data
```

#### Test Scripts

##### Standard Test Script

To see the option fetcher in action with predefined test parameters:

```bash
python test_option_fetcher.py
```

The standard test script will:
- Test with multiple tickers (AAPL, SPY)
- Test both call and put options
- Test multiple expiry dates (30, 60, and 90 days from now)
- Show how the function finds the closest available expiry date
- Calculate whether options are in-the-money (ITM) or out-of-the-money (OTM)
- Display the days difference between target and actual expiry dates

##### Custom Test Script

For more flexibility, use the custom test script with your own parameters:

```bash
python custom_option_test.py AAPL --option-type call --expiry 2023-12-15 --delta 0.5
```

Command line arguments:
- `ticker`: Required. The ticker symbol (e.g., AAPL, SPY)
- `--option-type`, `-t`: Option type (call or put). Default: call
- `--expiry`, `-e`: Target expiry date in YYYY-MM-DD format. Default: 30 days from now
- `--delta`, `-d`: Target delta value. Default: 0.5
- `--verbose`, `-v`: Enable verbose logging

Example:
```bash
# Test AAPL call option with delta 0.7 expiring in January 2024
python custom_option_test.py AAPL -t call -e 2024-01-19 -d 0.7

# Test SPY put option with default parameters (delta 0.5, expiry 30 days from now)
python custom_option_test.py SPY -t put

# Enable verbose logging for troubleshooting
python custom_option_test.py AAPL -v
```

### Option Simulator

#### Command Line Interface

The option simulator can be run from the command line:

```bash
python run_option_simulator.py AAPL --option-type call --position buy --expiry 2023-12-15 --delta 0.5
```

Command line arguments:
- `ticker`: Required. The ticker symbol (e.g., AAPL, SPY)
- `--option-type`, `-t`: Option type (call or put). Default: call
- `--position`, `-p`: Position type (buy or sell). Default: buy
- `--expiry`, `-e`: Target expiry date in YYYY-MM-DD format. Default: 30 days from now
- `--delta`, `-d`: Target delta value. Default: 0.5
- `--contracts`, `-c`: Number of contracts. Default: 1
- `--output-dir`, `-o`: Directory to save output files. Default: option_simulation_results
- `--save-plots`: Save plots instead of displaying them
- `--plot-type`: Type of plot to generate (all, slices, probability). Default: all
- `--probability-plot`: Type of probability plot to generate (none, line). Default: line
- `--verbose`, `-v`: Enable verbose logging

Examples:
```bash
# Simulate buying an AAPL call option with delta 0.7 expiring in January 2024
python run_option_simulator.py AAPL -t call -p buy -e 2024-01-19 -d 0.7

# Simulate selling a SPY put option with default parameters
python run_option_simulator.py SPY -t put -p sell

# Save plots instead of displaying them
python run_option_simulator.py AAPL --save-plots

# Generate only the slices plot
python run_option_simulator.py AAPL --plot-type slices
```

#### Programmatic Usage

You can also use the option simulator programmatically:

```python
from option_simulator import simulate_option_pnl, plot_pnl_slices, generate_simulation_report

# Simulate buying an AAPL call option
results = simulate_option_pnl(
    ticker="AAPL",
    option_type="call",
    expiry_date="2023-12-15",
    target_delta=0.5,
    position_type="buy",      # 'buy' or 'sell'
    num_contracts=1           # Number of contracts
)

# Generate and print a report
report = generate_simulation_report(results)
print(report)

# Plot the results
plot_pnl_slices(results)
```

### GBM Stock Price Simulator

#### Command Line Interface

```bash
python run_gbm_simulation.py AAPL
```

#### Advanced Options

```bash
python run_gbm_simulation.py AAPL --days 365 --simulations 1000 --benchmark SPY
```

Parameters:
- `ticker`: Stock symbol to simulate (required)
- `--days`: Number of days to simulate (default: 180)
- `--simulations`: Number of simulation paths to generate (default: 1000)
- `--benchmark`: Compare performance against this ticker (optional)
- `--save`: Path to save the plot (optional)

#### Example Output

The simulation provides:

1. **Price Target Analysis**: Percentile-based price targets (10th, 25th, 50th, 75th, 90th)
2. **Benchmark Comparison**: Expected returns and probability of outperformance
3. **Probability Analysis**: Likelihood of different price scenarios
4. **Visualization**: Price paths and confidence intervals

#### Programmatic Usage

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

## API Reference

### Option Fetcher Function Parameters

- `ticker` (str): The ticker symbol (e.g., 'AAPL', 'SPY')
- `option_type` (str): 'call' or 'put'
- `target_expiry_date` (str): Target expiry date in 'YYYY-MM-DD' format
- `target_delta` (float): Target delta value (absolute value, e.g., 0.5)

### Option Fetcher Return Value

The function returns a dictionary containing:
- `option_data`: DataFrame row with the selected option data
- `expiry_date`: The actual expiry date used
- `delta`: The actual delta value of the selected option
- `strike`: The strike price of the selected option
- `ticker`: The ticker symbol
- `option_type`: The option type ('call' or 'put')
- `underlying_price`: Current price of the underlying asset

### Option Simulator Return Value

The simulator returns a dictionary containing:
- Trade details (ticker, option type, position type, strike price, etc.)
- Option pricing information (option price, implied volatility, etc.)
- Date information (start date, expiry date, days to expiry)
- PnL matrix (2D array of PnLs for different dates and prices)
- Option value matrix (2D array of option values)
- Initial greeks (delta, gamma, theta, vega)
- Maximum profit and loss potential

### GBM Simulator Return Value

The GBM simulator returns a dictionary containing:
- Stock information (ticker, current price, etc.)
- Simulation parameters (volatility, risk-free rate)
- Date range for the simulation
- Price paths (2D array of simulated prices)
- Statistical analyses (mean path, standard deviation, percentiles)

## Benefits and Limitations

### Benefits of the GBM Model

- **Realistic Price Distributions**: Models the non-negative and log-normal distribution of stock prices
- **Consistent with Option Pricing Theory**: Same model underlying Black-Scholes
- **Business Days Only**: Simulates price changes only on trading days
- **Volatility Clustering**: Captures the tendency of volatility to vary over time

### Limitations

- The option simulator uses the Black-Scholes model, which has known limitations (e.g., assumes constant volatility, no dividends, etc.)
- GBM model assumes constant volatility (not accounting for volatility clustering)
- Assumes returns follow a normal distribution (may not capture fat tails)
- Does not model jumps in price (can be significant for some stocks)
- Past volatility may not represent future volatility

## Notes

- The option fetcher uses the `yfinance` library to fetch option data, which may have limitations on data availability and accuracy.
- If the exact delta value is not available in the data, the function calculates an approximate delta based on the option's strike price and the current stock price.
- For put options, the function works with the absolute value of delta for finding the closest match.
- The implementation includes fallback methods to handle different versions of the yfinance API.
- Uses `fast_info['lastPrice']` for more efficient price retrieval in the latest yfinance versions.
- Starting with yfinance 0.2.28, the `option_chain()` method also returns underlying data, which this tool can utilize.
- Comprehensive logging is included to help troubleshoot any issues with the API.
- Options expiry dates are typically on the third Friday of each month, so the closest available expiry date may differ from your target date.
- The risk-free rate is fetched from the 3-month Treasury yield (^IRX) or defaults to 3% if unavailable.
- All simulations start from today's date using current market prices, as historical option data is not available.
- The price range for simulations is calculated using the implied volatility of the option and covers a 2-standard deviation move (approximately 95% confidence interval) by the expiry date.

## Future Enhancements

Potential improvements to consider:
- Implement GARCH models for time-varying volatility
- Add jump-diffusion processes for modeling price jumps
- Include mean-reversion for certain assets
- Implement correlated simulations for multiple assets 