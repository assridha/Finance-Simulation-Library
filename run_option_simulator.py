#!/usr/bin/env python3
"""
Option Simulator CLI

This script provides a command-line interface for the option simulator.
"""

import argparse
import os
from datetime import datetime, timedelta
import logging
from option_simulator import (
    simulate_option_pnl,
    plot_pnl_heatmap,
    plot_pnl_slices,
    plot_price_probability,
    plot_probability_heatmap,
    generate_simulation_report
)

def setup_logging(verbose=False):
    """Configure logging based on verbosity level"""
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Option Trade PnL Simulator')
    
    # Required arguments
    parser.add_argument('ticker', type=str, help='Ticker symbol (e.g., AAPL, SPY)')
    
    # Option parameters
    parser.add_argument('--option-type', '-t', type=str, choices=['call', 'put'], default='call',
                        help='Option type: call or put (default: call)')
    
    parser.add_argument('--position', '-p', type=str, choices=['buy', 'sell'], default='buy',
                        help='Position type: buy or sell (default: buy)')
    
    parser.add_argument('--expiry', '-e', type=str, 
                        help='Target expiry date in YYYY-MM-DD format (default: 30 days from now)')
    
    parser.add_argument('--delta', '-d', type=float, default=0.5,
                        help='Target delta value (default: 0.5)')
    
    parser.add_argument('--contracts', '-c', type=int, default=1,
                        help='Number of contracts (default: 1)')
    
    # Output options
    parser.add_argument('--output-dir', '-o', type=str, default='option_simulation_results',
                        help='Directory to save output files (default: option_simulation_results)')
    
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots instead of displaying them')
    
    parser.add_argument('--plot-type', choices=['all', 'heatmap', 'slices', 'probability'], default='all',
                        help='Type of plot to generate (default: all)')
    
    parser.add_argument('--probability-plot', choices=['none', 'heatmap', 'line', 'both'], default='both',
                        help='Type of probability plot to generate (default: both)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def validate_date(date_str):
    """Validate date string format"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def main():
    """Main function"""
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Validate dates if provided
    if args.expiry and not validate_date(args.expiry):
        logger.error(f"Invalid expiry date format: {args.expiry}. Please use YYYY-MM-DD format.")
        return
    
    # Create output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    
    # Run simulation
    try:
        logger.info(f"Running simulation for {args.ticker} {args.option_type} option")
        results = simulate_option_pnl(
            ticker=args.ticker,
            option_type=args.option_type,
            expiry_date=args.expiry if args.expiry else (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            target_delta=args.delta,
            position_type=args.position,
            num_contracts=args.contracts
        )
        
        # Generate report
        report = generate_simulation_report(results)
        print(report)
        
        # Save report to file if saving plots
        if args.save_plots:
            report_file = os.path.join(args.output_dir, f"{args.ticker}_{args.option_type}_{args.position}_report.txt")
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {report_file}")
        
        # Generate PnL plots
        if args.plot_type in ['all', 'heatmap']:
            if args.save_plots:
                heatmap_file = os.path.join(args.output_dir, f"{args.ticker}_{args.option_type}_{args.position}_heatmap.png")
                plot_pnl_heatmap(results, save_path=heatmap_file)
                logger.info(f"PnL heatmap saved to {heatmap_file}")
            else:
                plot_pnl_heatmap(results)
        
        if args.plot_type in ['all', 'slices']:
            if args.save_plots:
                slices_file = os.path.join(args.output_dir, f"{args.ticker}_{args.option_type}_{args.position}_slices.png")
                plot_pnl_slices(results, save_path=slices_file)
                logger.info(f"PnL slices plot saved to {slices_file}")
            else:
                plot_pnl_slices(results)
        
        # Generate probability plots
        if args.plot_type in ['all', 'probability'] and args.probability_plot != 'none':
            # Generate probability heatmap
            if args.probability_plot in ['heatmap', 'both']:
                if args.save_plots:
                    prob_heatmap_file = os.path.join(args.output_dir, f"{args.ticker}_probability_heatmap.png")
                    plot_probability_heatmap(results, save_path=prob_heatmap_file)
                    logger.info(f"Price probability heatmap saved to {prob_heatmap_file}")
                else:
                    plot_probability_heatmap(results)
            
            # Generate probability line graph
            if args.probability_plot in ['line', 'both']:
                if args.save_plots:
                    prob_line_file = os.path.join(args.output_dir, f"{args.ticker}_probability_line.png")
                    plot_price_probability(
                        results['price_probabilities'], 
                        results['current_price'],
                        [results['strike_price']], 
                        save_path=prob_line_file
                    )
                    logger.info(f"Price probability line plot saved to {prob_line_file}")
                else:
                    plot_price_probability(
                        results['price_probabilities'], 
                        results['current_price'],
                        [results['strike_price']]
                    )
        
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 