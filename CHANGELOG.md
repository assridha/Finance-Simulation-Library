# Changelog

## [v0.2.0] - 2024-03-25

### Added
- Web interface with React frontend for stock price visualization
- Flask backend with RESTful API endpoints
- Real-time Monte Carlo simulation with interactive plots
- Historical data visualization with candlestick charts
- Frontend caching for historical data
- Debounced simulation requests
- Loading states and error handling in the frontend
- CORS support for web applications
- Comprehensive API documentation

### Changed
- Optimized `StockSimulator` component for better performance
- Enhanced error handling and user feedback
- Updated plot styling with better visibility and contrast
- Improved API response formats
- Changed default backend port to 5001 to avoid AirPlay conflicts
- Updated setup instructions for frontend and backend
- Enhanced documentation with API details and troubleshooting guide

### Fixed
- Multiple simultaneous API calls issue
- Undefined property access errors in statistics display
- Port conflicts with AirPlay on macOS
- Error handling for invalid stock symbols
- TypeScript type definitions for dependencies

### Technical Debt
- Added proper useCallback dependencies
- Improved code organization in StockSimulator component
- Better state management for loading and error states
- Memoized API base URL and callback functions
- Added proper TypeScript types

## [v0.1.3] - 2024-03-24

### Added
- Modular portfolio system with strategy composers and analyzers
- New butterfly spread strategy implementation (using both calls and puts)
- Command-line arguments for customizing growth rate and volatility
- Volatility multiplier parameter for scenario analysis
- Enhanced output clarity for option positions (BOUGHT/SOLD indicators)
- Separated data fetching from strategy logic for improved maintainability
- Strategy-specific logic for selecting optimal contracts

### Changed
- Refactored option simulation example to use new modular components
- Improved display of position information with clearer terminology

### Fixed
- Resolved issue with abstract `StrategyComposer` class instantiation
- Fixed butterfly spread strategy to properly select strike prices
- Improved error handling for cases where exact strike prices aren't available

## [v0.1.2] - 2024-03-24

### Added
- Command-line interface for option simulation examples
- Support for selecting specific strategies to simulate

### Fixed
- Improved option strategy value calculation to properly handle debits and credits
- Updated sign convention to follow standard trading notation (debit positive, credit negative)
- Fixed P&L calculation throughout simulation for long/short positions
- Enhanced option pricing to use individual implied volatilities for each contract

## [v0.1.1] - 2024-03-23

### Added
- New caching mechanisms for option chain and market data
- Added vectorized Black-Scholes calculations

### Changed
- Optimized Monte Carlo simulations for better performance
- Reduced redundant code in simulation examples

### Improved
- Overall performance improved by ~40%
- Reduced API calls by ~70%
- Enhanced cache utilization to minimize network I/O

### Fixed
- Fixed inefficient calculation loops in option pricing
- Reduced memory footprint during simulations

## [v0.1.0] - 2024-03-20

### Added
- Basic GBM model with growth models
- Black-Scholes option pricing
- Essential utilities for data fetching and calculations
- Basic visualization capabilities
- Initial documentation
- Basic testing framework

### Changed
- None (initial release)

### Deprecated
- None (initial release)

### Removed
- None (initial release)

### Fixed
- None (initial release)

### Security
- None (initial release)

## [0.3.0] - YYYY-MM-DD (Planned)

### Added
- TimGAN implementation
- Advanced portfolio optimization
- Interactive visualizations
- Performance optimizations
- Parallel processing support
- Caching mechanisms

### Changed
- None (planned)

### Deprecated
- None (planned)

### Removed
- None (planned)

### Fixed
- None (planned)

### Security
- None (planned)

## [1.0.0] - YYYY-MM-DD (Planned)

### Added
- Complete documentation
- Comprehensive test coverage
- Performance benchmarks
- Release package
- Future roadmap

### Changed
- None (planned)

### Deprecated
- None (planned)

### Removed
- None (planned)

### Fixed
- None (planned)

### Security
- None (planned) 