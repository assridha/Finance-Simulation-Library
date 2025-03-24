# Release Notes - v0.1.3

## Overview

This release introduces significant enhancements to the Option Simulator, focusing on modular design, advanced strategy implementation, and customizable simulation parameters. These improvements provide greater flexibility in analyzing option strategies under various market conditions.

## Key Enhancements

1. **Modular Portfolio System**
   - Introduced `StrategyComposer` abstract class for creating option strategies
   - Added `StrategyAnalyzer` class for evaluating and presenting strategy results
   - Implemented a clear separation between strategy composition and analysis logic

2. **Butterfly Spread Strategy Implementation**
   - Complete implementation using both calls and puts
   - Intelligent strike selection based on ATM pricing
   - Proper position sizing and contract selection

3. **Customizable Simulation Parameters**
   - Added command-line arguments for defining custom growth rates
   - Implemented volatility override parameter for precise scenario testing
   - Created volatility multiplier for stress-testing existing strategies

4. **Improved Output Clarity**
   - Enhanced position display with LONG/SHORT indicators for stocks
   - Added BOUGHT/SOLD indicators for option contracts
   - Clearer presentation of premium values and position details

5. **Architectural Improvements**
   - Separated data fetching logic from strategy implementation
   - Added strategy-specific contract selection logic
   - Improved error handling for unavailable strike prices

## Usage Examples

### Running with Custom Volatility and Growth Rate

```bash
python -m financial_sim_library.examples.option_simulation_example -st butterfly -v 0.4 -g 0.05
```

### Using Volatility Multiplier for Stress Testing

```bash
python -m financial_sim_library.examples.option_simulation_example -st covered_call -vm 1.5
```

## Breaking Changes

None. All changes are backward compatible.

## Bug Fixes

- Fixed issue with abstract `StrategyComposer` class instantiation
- Improved butterfly spread strategy to properly select strike prices
- Enhanced error handling for cases where exact strike prices aren't available

## Known Issues

None.

## Future Plans

- Expand the strategy library with more complex option strategies
- Implement additional portfolio metrics for strategy evaluation
- Add optimization capabilities for strategy parameters 