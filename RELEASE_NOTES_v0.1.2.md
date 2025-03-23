# Finance Simulation Library v0.1.2 Release Notes

## New Features
- **Command-line Interface**: Introduced command-line interface for running option simulations.
  - Users can now run selected strategies via command-line arguments.
  - Added support for specifying the number of simulation paths from the command line.

## Major Improvements
- **Enhanced Option Pricing Accuracy**:
  - Fixed option pricing for in-the-money (ITM) options in multi-leg strategies.
  - Corrected the handling of deep ITM options showing negative time value in Butterfly and PMCC strategies.
  - Added position-specific tracking for options with same strike/type in complex strategies.
  - Now using the actual market entry price at t=0 for consistent valuation and P&L calculations.

- **Strategy Value Calculation**:
  - Improved strategy value calculation to properly handle debits and credits.
  - Updated P&L calculation to follow standard trading notation (debit positive, credit negative).
  - Fixed P&L calculation throughout simulations for both long and short positions.

## Usage Example
```bash
# Run specific strategy simulations
python -m financial_sim_library.option_simulator.main --strategy butterfly

# Increase simulation paths for higher accuracy
python -m financial_sim_library.option_simulator.main --paths 10000
```

## Breaking Changes
None - This release maintains backward compatibility with existing code.

## Installation
```bash
pip install finance-simulation-library==0.1.2
``` 