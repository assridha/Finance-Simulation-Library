# Finance Simulation Library

A comprehensive Python library for simulating and analyzing financial instruments, including stock prices, options, and portfolios.

## Latest Updates (v0.1.3)

🚀 **Major Improvements**
- **Modular Portfolio System**: New strategy composers and analyzers for flexible strategy creation
- **Advanced Option Strategies**: Enhanced butterfly spread implementation using both calls and puts
- **Customizable Simulations**: Command-line parameters for growth rate and volatility
- **Scenario Analysis**: Volatility multiplier for stress-testing strategies
- **Improved Position Display**: Clear BOUGHT/SOLD indicators for option positions
- **Better Architecture**: Separated data fetching from strategy logic
- **Strategy-Specific Selection**: Intelligent contract selection based on strategy requirements

See the [CHANGELOG.md](CHANGELOG.md) for full details.

## Features

- **Stock Price Simulation**: Generate realistic stock price paths using Geometric Brownian Motion (GBM)
- **Option Strategy Simulation**: Analyze various option strategies with Monte Carlo simulations
  - **Multiple Strategy Types**: Simulate simple calls/puts, covered calls, poor man's covered calls, vertical spreads, and more
  - **Real-time Option Data**: Fetch current option chains with accurate pricing
  - **Comprehensive Analysis**: View payoff diagrams, profit/loss metrics, and probability distributions
- **Growth Models**: Flexible framework for incorporating various growth models into price simulations
- **Volatility Analysis**: Calculate historical volatility and other key market metrics
- **Financial Data Fetching**: Easily obtain historical price data, risk-free rates, and more
- **Visualization Tools**: Create beautiful, informative plots of simulation results
- **Command-Line Interface**: Run simulations directly from the terminal
- **Extensible Architecture**: Add your own models and strategies with the base classes provided

## Installation

### Requirements

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

### Using Virtual Environment (Recommended)

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Install from source

```bash
# Clone the repository
git clone https://github.com/your-username/financial_sim_library.git
cd financial_sim_library

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Stock Price Simulation with Growth Models

```python
from financial_sim_library.stock_simulator.models.gbm import GBMModel
from financial_sim_library.stock_simulator.models.growth_models import (
    FixedGrowthModel,
    PowerLawGrowthModel,
    ExogenousGrowthModel,
    CompositeGrowthModel
)
from financial_sim_library.visualization.price_plots import plot_price_simulations

# Create a GBM model with fixed growth
model = GBMModel(
    ticker="AAPL",
    growth_model=FixedGrowthModel(growth_rate=0.1)  # 10% annual growth
)

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
```

### Available Growth Models

1. **Fixed Growth Model**
   - Constant annual growth rate
   - Suitable for stable, predictable growth scenarios

2. **Power Law Growth Model**
   - Growth rate scales with time
   - Useful for modeling diminishing returns or accelerating growth

3. **Exogenous Growth Model**
   - Custom growth function based on external factors
   - Supports market cycles, seasonal patterns, and other complex growth patterns

4. **Composite Growth Model**
   - Combines multiple growth models with weights
   - Allows for complex growth scenarios combining different factors

### Growth Model Examples

Run the growth model examples to see different growth scenarios in action:

```bash
python3 -m financial_sim_library.examples.growth_model_examples
```

This will demonstrate:
- Fixed growth with 10% annual rate
- Power law growth with diminishing returns
- Oscillating growth based on market cycles
- Composite growth combining multiple factors
- Complex growth with market cycles and exogenous variables

### Option Strategy Simulator

The library includes a powerful option strategy simulator with full support for:

```python
from financial_sim_library.option_simulator.data_fetcher import MarketDataFetcher
from financial_sim_library.option_simulator.strategies import SimpleStrategy
from financial_sim_library.option_simulator.simulator import MonteCarloOptionSimulator
from datetime import datetime, timedelta

# Fetch market data and option contracts
fetcher = MarketDataFetcher()
current_price = fetcher.get_stock_price("AAPL")
# Set expiry date to 30 days from now
expiry_date = datetime.now() + timedelta(days=30)


# Get option contracts for a strategy
strategy_contracts = fetcher.get_option_strategy_contracts(
    "AAPL", 'covered_call', expiry_date
)

# Define strategy positions
positions = [
    # Long 100 shares of stock
    {'type': 'stock', 'symbol': "AAPL", 'quantity': 100, 'entry_price': current_price},
    # Short 1 call option
    {'contract': strategy_contracts['call'], 'quantity': -1}
]

# Create and run the simulation
strategy = SimpleStrategy("Covered Call", positions)
simulator = MonteCarloOptionSimulator(
    strategy=strategy,
    price_model='gbm',
    volatility=fetcher.get_historical_volatility("AAPL"),
    risk_free_rate=fetcher.get_risk_free_rate()
)

# Run 1000 price path simulations
results = simulator.run_simulation(num_paths=1000, num_steps=100)
```

#### Supported Option Strategies

The option simulator supports various strategies, including:

1. **Simple Option Strategies**
   - Long/Short Calls and Puts

2. **Covered Call**
   - Long stock position + Short call option

3. **Poor Man's Covered Call**
   - Long deep ITM call + Short OTM call

4. **Vertical Spreads**
   - Bull Call Spread
   - Bear Put Spread
   - Custom strike selection

5. **Custom Strategy Builder**
   - Build any combination of option and stock positions

#### Running Option Examples

Try the option strategy simulator with pre-built examples:

```bash
python3 -m financial_sim_library.examples.option_simulation_example
```

This will run simulations for multiple option strategies and display:
- Stock price path projections
- Strategy payoff diagrams
- Profit/loss analysis
- Probability distributions of outcomes

#### Option Simulator Command-Line Interface

The option simulation example now supports command-line arguments that allow you to run selected strategies. This makes it easier to focus on specific option strategies you're interested in analyzing.

##### Usage

```bash
python -m financial_sim_library.examples.option_simulation_example [options]
```

##### Options

- `-s, --symbol SYMBOL`: Stock symbol to simulate (default: AAPL)
- `-n, --num-paths NUM_PATHS`: Number of simulation paths (default: 1000)
- `-st, --strategies STRATEGY [STRATEGY ...]`: Strategies to simulate
- `-g, --growth-rate RATE`: Custom annual growth rate (decimal, e.g., 0.05 for 5%)
- `-v, --volatility-override VOL`: Custom annual volatility (decimal, e.g., 0.25 for 25%)
- `-vm, --volatility-multiplier MULT`: Multiplier to apply to historical volatility (e.g., 1.5)

##### Available Strategies

- `simple_call`: Simple Buy Call Option
- `covered_call`: Covered Call Strategy
- `pmcc`: Poor Man's Covered Call Strategy
- `vertical_spread`: Vertical Spread Strategy
- `butterfly`: Custom Butterfly Spread Strategy
- `all`: Run all available strategies (default)

##### Examples

Run all strategies for the default symbol:
```bash
python -m financial_sim_library.examples.option_simulation_example
```

Run only butterfly and vertical spread strategies for TSLA with 2000 paths:
```bash
python -m financial_sim_library.examples.option_simulation_example -s TSLA -n 2000 -st butterfly vertical_spread
```

Run just the simple call option strategy:
```bash
python -m financial_sim_library.examples.option_simulation_example -st simple_call
```

Run butterfly strategy with custom volatility (40%) and growth rate (5%):
```bash
python -m financial_sim_library.examples.option_simulation_example -st butterfly -v 0.4 -g 0.05
```

Run covered call strategy with 1.5x historical volatility:
```bash
python -m financial_sim_library.examples.option_simulation_example -st covered_call -vm 1.5
```

### Command Line Usage

The library includes a command-line interface for quick simulations:

```bash
# Run a basic simulation
python3 -m financial_sim_library.run_stock_simulator AAPL

# Customize the simulation
python3 -m financial_sim_library.run_stock_simulator AAPL --days 90 --simulations 500 --plot-type paths

# Save the results
python3 -m financial_sim_library.run_stock_simulator AAPL --save-plots --output-dir results
```

## Main Components

### Stock Simulator

- `StockPriceModel`: Abstract base class for all stock price models
- `GBMModel`: Implementation of Geometric Brownian Motion for stock price simulation

### Option Simulator

- `OptionContract`: Class representing an option contract with all relevant parameters
- `OptionStrategy`: Abstract base class for implementing option strategies
- `SimpleStrategy`: Implementation for constructing custom option strategies
- `MonteCarloOptionSimulator`: Simulator for option strategies using Monte Carlo methods
- `MarketDataFetcher`: Utility for fetching real-time option chain and market data

### Utilities

- `financial_calcs`: Utilities for financial calculations (volatility, risk-free rate, etc.)
- `data_fetcher`: Functions for fetching historical data and other market information

### Visualization

- `price_plots`: Functions for creating various types of plots for simulation results
  - Price path visualizations
  - Price distribution histograms
  - Price probability heatmaps
- Option strategy visualizations
  - Payoff diagrams
  - Profit/loss charts
  - Risk profile analysis

## Examples

More detailed examples are available in the `examples` directory:

- Basic simulation with default parameters
- Custom simulation with specific parameters
- Comparing simulations with different volatilities
- Simulating multiple tickers and comparing results
- Option strategy simulations:
  - Simple option strategies (long/short calls and puts)
  - Covered call strategies
  - Poor man's covered calls
  - Vertical spreads
  - Custom multi-leg strategies

To run the examples:

```bash
# Stock simulation examples
python3 -m financial_sim_library.examples.stock_simulation_examples

# Growth model examples
python3 -m financial_sim_library.examples.growth_model_examples

# Option strategy examples
python3 -m financial_sim_library.examples.option_simulation_example
```

## Architecture

The library is designed with extensibility in mind:

1. **Base Models**: Abstract base classes define the interfaces for different types of models
2. **Implementations**: Concrete implementations provide specific functionality
3. **Utilities**: Common functions for data handling and calculations
4. **Visualization**: Tools to analyze and display results

## Development

### Running Tests

Tests are available in the `tests` directory and can be run with:

```bash
python3 -m unittest discover -s financial_sim_library/tests
# or
pytest financial_sim_library/tests
```

### Adding New Models

To add a new stock price model:

1. Create a new class that inherits from `StockPriceModel`
2. Implement the required methods (`calibrate` and `simulate`)
3. Add any additional methods specific to your model

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This library uses [yfinance](https://github.com/ranaroussi/yfinance) for fetching market data
- Visualization is powered by [matplotlib](https://matplotlib.org/)
- Code was generated using Cursor with Sonnet 3.7

# Option Simulator v2

## Recent Updates

### Frontend Improvements
- Added caching for historical data to improve performance
- Implemented debounced API calls to prevent server overload
- Enhanced error handling and user feedback
- Improved plot styling and visibility
- Added loading states for better user experience

### Backend Changes
- Fixed CORS issues in Flask backend
- Improved error handling for invalid stock symbols
- Using port 5001 by default to avoid conflicts with AirPlay

## Setup Instructions

### Backend Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```bash
   python -m flask --app financial_sim_library.web_interface.app run --debug --port 5001
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   The app will run on http://localhost:3000

## Known Issues and Solutions
- If port 5000 is in use (common on macOS due to AirPlay), the backend will use port 5001
- If you see TypeScript errors related to lodash, run:
  ```bash
  cd frontend && npm install --save-dev @types/lodash
  ```
- If the app seems slow, try reducing the number of simulations or increasing the debounce timeout

## API Documentation

The application provides several REST API endpoints for market data and simulations.

### Market Data Endpoints

#### Get Historical Data
```http
GET /api/market-data/historical
```
Fetches historical price data for a given stock symbol.

**Query Parameters:**
- `symbol` (required): Stock symbol (e.g., AAPL)
- `period` (optional): Time period (default: '6mo', options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
- `interval` (optional): Data interval (default: '1d', options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

**Response:**
```json
{
    "timestamps": ["2024-03-24", "2024-03-25", ...],
    "open": [100.0, 101.0, ...],
    "high": [102.0, 103.0, ...],
    "low": [99.0, 98.0, ...],
    "close": [101.5, 102.5, ...],
    "volume": [1000000, 1100000, ...],
    "symbol": "AAPL",
    "period": "6mo",
    "interval": "1d",
    "timestamp": "2024-03-25T12:00:00"
}
```

#### Get Current Market Data
```http
GET /api/market-data?symbol=AAPL
```
Fetches current market data including price, change, and volatility.

**Query Parameters:**
- `symbol` (required): Stock symbol (e.g., AAPL)

**Response:**
```json
{
    "symbol": "AAPL",
    "price": 175.50,
    "change": 2.50,
    "change_percent": 1.45,
    "volume": 55000000,
    "volatility": 25.5,
    "timestamp": "2024-03-25T12:00:00"
}
```

### Simulation Endpoints

#### Run Market Simulation
```http
POST /api/market/simulate
```
Runs a Monte Carlo simulation for future stock prices.

**Request Body:**
```json
{
    "symbol": "AAPL",
    "days_to_simulate": 30,
    "num_simulations": 5
}
```

**Parameters:**
- `symbol` (required): Stock symbol to simulate
- `days_to_simulate` (optional): Number of days to simulate (default: 30, max: 365)
- `num_simulations` (optional): Number of simulation paths (default: 5, max: 1000)

**Response:**
```json
{
    "symbol": "AAPL",
    "current_price": 175.50,
    "simulated_paths": [
        [175.50, 176.20, 177.10, ...],
        [175.50, 174.80, 173.90, ...],
        ...
    ],
    "statistics": {
        "mean": 180.25,
        "std": 5.32,
        "min": 170.15,
        "max": 190.35,
        "expected_return": 0.0271,
        "prob_above_current": 0.56,
        "volatility": 25.5
    },
    "timestamps": ["2024-03-25", "2024-03-26", ...],
    "timestamp": "2024-03-25T12:00:00"
}
```

### Error Responses

All endpoints return standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Symbol not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

Error Response Format:
```json
{
    "error": "Error message describing what went wrong",
    "status_code": 400
}
```

### Rate Limiting

- Market data endpoints: 30 requests per minute
- Simulation endpoints: 10 requests per minute

### CORS Support

The API supports CORS for web applications running on:
- `http://localhost:3000` (Development)
- Other authorized domains (Production)
