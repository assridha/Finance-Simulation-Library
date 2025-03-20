#!/usr/bin/env python3
"""
Stock Price Simulator CLI

This script provides a command-line interface for running stock price simulations.
"""

import argparse
import os
import sys
import numpy as np
import logging
from datetime import datetime

# Add the package directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from financial_sim_library.stock_simulator.models.gbm import GBMModel
from financial_sim_library.visualization.price_plots import (
    plot_price_simulations,
    plot_price_distribution,
    plot_price_heatmap
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run stock price simulations.')
    
    # Required arguments
    parser.add_argument('ticker', type=str, help='Ticker symbol (e.g., AAPL, SPY)')
    
    # Simulation parameters
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to simulate (default: 30)')
    parser.add_argument('--simulations', type=int, default=1000,
                       help='Number of simulation paths to generate (default: 1000)')
    parser.add_argument('--volatility', type=float,
                       help='Volatility to use (default: calculated from historical data)')
    parser.add_argument('--risk-free-rate', type=float,
                       help='Risk-free rate to use (default: fetched from market data)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    
    # Plot options
    parser.add_argument('--plot-type', choices=['all', 'paths', 'distribution', 'heatmap'],
                       default='all', help='Type of plots to generate (default: all)')
    parser.add_argument('--path-count', type=int, default=10,
                       help='Number of individual paths to plot (default: 10)')
    parser.add_argument('--confidence-interval', type=float, default=0.95,
                       help='Confidence interval for path plots (default: 0.95)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory to save output files (default: output)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots instead of displaying them')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def setup_logging(verbose=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logger.debug("Logging configured with level: %s", "DEBUG" if verbose else "INFO")

def ensure_output_dir(output_dir):
    """Ensure the output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    return output_dir

def generate_simulation_report(simulation_results):
    """
    Generate a text report of the simulation results.
    
    Args:
        simulation_results (dict): Results from GBMModel.simulate()
    
    Returns:
        str: Text report
    """
    ticker = simulation_results['ticker']
    current_price = simulation_results['current_price']
    stats = simulation_results['statistics']
    params = simulation_results['parameters']
    
    days_simulated = params['days_simulated']
    num_simulations = params['num_simulations']
    volatility = params['volatility']
    risk_free_rate = params['risk_free_rate']
    
    # Create the report
    report = [
        "=" * 80,
        f"{'STOCK PRICE SIMULATION REPORT':^80}",
        "=" * 80,
        "",
        f"Simulation Parameters:",
        f"  - Ticker: {ticker}",
        f"  - Current Price: ${current_price:.2f}",
        f"  - Volatility: {volatility:.2%}",
        f"  - Risk-Free Rate: {risk_free_rate:.2%}",
        f"  - Days Simulated: {days_simulated}",
        f"  - Number of Simulations: {num_simulations}",
        "",
        f"Price Distribution at End of Simulation:",
        f"  - Mean Price: ${stats['mean']:.2f} ({(stats['mean']/current_price - 1)*100:.2f}%)",
        f"  - Median Price: ${stats['median']:.2f} ({(stats['median']/current_price - 1)*100:.2f}%)",
        f"  - Standard Deviation: ${stats['std']:.2f} ({stats['std']/current_price*100:.2f}%)",
        f"  - Minimum Price: ${stats['min']:.2f} ({(stats['min']/current_price - 1)*100:.2f}%)",
        f"  - Maximum Price: ${stats['max']:.2f} ({(stats['max']/current_price - 1)*100:.2f}%)",
        "",
        f"Percentiles:",
        f"  - 10th Percentile: ${stats['percentiles']['10']:.2f} ({(stats['percentiles']['10']/current_price - 1)*100:.2f}%)",
        f"  - 25th Percentile: ${stats['percentiles']['25']:.2f} ({(stats['percentiles']['25']/current_price - 1)*100:.2f}%)",
        f"  - 50th Percentile: ${stats['percentiles']['50']:.2f} ({(stats['percentiles']['50']/current_price - 1)*100:.2f}%)",
        f"  - 75th Percentile: ${stats['percentiles']['75']:.2f} ({(stats['percentiles']['75']/current_price - 1)*100:.2f}%)",
        f"  - 90th Percentile: ${stats['percentiles']['90']:.2f} ({(stats['percentiles']['90']/current_price - 1)*100:.2f}%)",
        "",
        f"Probability Analysis:",
        f"  - Probability of Price Increase: {stats['prob_above_current']:.2f}%",
        f"  - Probability of >10% Increase: {stats['prob_10pct_up']:.2f}%",
        f"  - Probability of >10% Decrease: {stats['prob_10pct_down']:.2f}%",
        "",
        f"Expected Return: {stats['expected_return']:.2f}%",
        f"  - Annualized: {stats['expected_return'] * (365 / days_simulated):.2f}%",
        "",
        "=" * 80
    ]
    
    return "\n".join(report)

def main():
    """Main function to run the stock price simulation."""
    args = parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(args.output_dir)
    
    # Log the args
    logger.debug(f"Command-line arguments: {args}")
    
    try:
        # Create the model
        model = GBMModel(
            ticker=args.ticker,
            volatility=args.volatility,
            risk_free_rate=args.risk_free_rate,
            seed=args.seed
        )
        
        # Run the simulation
        logger.info(f"Running simulation for {args.ticker} over {args.days} days with {args.simulations} paths...")
        results = model.simulate(
            days_to_simulate=args.days,
            num_simulations=args.simulations
        )
        
        # Generate and print the report
        report = generate_simulation_report(results)
        print(report)
        
        # Save the report
        report_path = os.path.join(
            output_dir, 
            f"{args.ticker}_simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to {report_path}")
        
        # Generate plots
        if args.plot_type in ['all', 'paths']:
            # Generate paths plot
            logger.info("Generating price paths plot...")
            
            # Determine save path if needed
            save_path = None
            if args.save_plots:
                save_path = os.path.join(
                    output_dir,
                    f"{args.ticker}_price_paths_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            
            # Generate the plot
            plot_price_simulations(
                results,
                num_paths_to_plot=args.path_count,
                confidence_interval=args.confidence_interval,
                save_path=save_path
            )
        
        if args.plot_type in ['all', 'distribution']:
            # Generate distribution plot
            logger.info("Generating price distribution plot...")
            
            # Determine save path if needed
            save_path = None
            if args.save_plots:
                save_path = os.path.join(
                    output_dir,
                    f"{args.ticker}_price_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            
            # Generate the plot
            plot_price_distribution(
                results,
                save_path=save_path
            )
        
        if args.plot_type in ['all', 'heatmap']:
            # Generate heatmap plot
            logger.info("Generating price heatmap plot...")
            
            # Determine save path if needed
            save_path = None
            if args.save_plots:
                save_path = os.path.join(
                    output_dir,
                    f"{args.ticker}_price_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            
            # Generate the plot
            plot_price_heatmap(
                results,
                save_path=save_path
            )
        
        logger.info("Simulation completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}", exc_info=args.verbose)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 