# Release Notes - v0.1.2

## Option Strategy Simulation Improvements

This patch release focuses on improving the accuracy and usability of option strategy simulations.

### New Features
1. **Command-Line Interface for Option Simulations**
   - Added support for running selected option strategies via command-line arguments
   - Flexible symbol selection and simulation path configuration
   - Easy-to-use interface for testing different strategies

### Major Improvements
1. **Enhanced Option Pricing Accuracy**
   - Each option contract now uses its own specific implied volatility
   - Properly accounts for volatility smile/skew effects
   - More accurate pricing for multi-leg strategies

2. **Fixed Strategy Value Calculations**
   - Updated sign convention to match standard trading notation
   - Debits (paying money) are now positive values
   - Credits (receiving money) are now negative values
   - Accurate P&L tracking throughout the simulation

### Usage Example
```bash
# Run specific strategies for a custom symbol
python -m financial_sim_library.examples.option_simulation_example -s TSLA -st butterfly vertical_spread

# Run all strategies with increased simulation paths
python -m financial_sim_library.examples.option_simulation_example -n 2000
```

### Breaking Changes
None. This is a backward-compatible improvement to existing functionality.

### Installation
```bash
pip install financial_sim_library==0.1.2
``` 