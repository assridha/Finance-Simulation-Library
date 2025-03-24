# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Basic documentation
- Design specification
- Version release plan
- Bid-ask spread tracking for option contracts and simulation results
- Total bid-ask cost impact calculation for option strategies
- Object-oriented cost basis calculation system with strategy-specific implementations
- Maximum potential loss and breakeven point calculations for all option strategies

### Changed
- Refactored cost basis calculation to use polymorphic design for better extensibility
- Improved strategy composer interface to include cost basis calculation methods
- Refactored option simulation example to use new modular components
- Improved display of position information with clearer terminology

### Fixed
- Improved option pricing accuracy by calculating implied volatility from entry prices instead of using reported IVs
- Fixed discrepancies in deep ITM option pricing by eliminating dependency on externally reported implied volatility values

## [0.1.3] - 2025-03-24
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

## [0.1.2] - 2025-03-24
### Added
- Command-line interface for option simulation examples
- Support for selecting specific strategies to simulate

### Fixed
- Improved option strategy value calculation to properly handle debits and credits
- Updated sign convention to follow standard trading notation (debit positive, credit negative)
- Fixed P&L calculation throughout simulation for long/short positions
- Enhanced option pricing to use individual implied volatilities for each contract

## [0.1.1] - 2025-03-23
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

## [0.1.0] - 2025-03-20
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

## [0.2.0] - YYYY-MM-DD
### Added
- ARIMA and GARCH models
- Monte Carlo option pricing
- Basic portfolio management
- Expanded utility functions
- Improved visualizations
- Increased test coverage

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

## [0.3.0] - YYYY-MM-DD
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

## [1.0.0] - YYYY-MM-DD
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