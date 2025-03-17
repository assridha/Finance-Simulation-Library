#!/usr/bin/env python3
"""
Custom Option Test Script

This script allows testing the option_fetcher with custom parameters from the command line.
"""

import argparse
import datetime
import pandas as pd
from option_fetcher import fetch_option_data
import logging

def setup_logging(verbose=False):
    """Configure logging based on verbosity level"""
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test option fetcher with custom parameters')
    
    parser.add_argument('ticker', type=str, help='Ticker symbol (e.g., AAPL, SPY)')
    
    parser.add_argument('--option-type', '-t', type=str, choices=['call', 'put'], default='call',
                        help='Option type: call or put (default: call)')
    
    parser.add_argument('--expiry', '-e', type=str, 
                        help='Target expiry date in YYYY-MM-DD format (default: 30 days from now)')
    
    parser.add_argument('--delta', '-d', type=float, default=0.5,
                        help='Target delta value (default: 0.5)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def format_option_data(option_data):
    """Format option data for display"""
    # Define important fields to display first
    important_fields = [
        'contractSymbol', 'lastTradeDate', 'bid', 'ask', 'lastPrice',
        'impliedVolatility', 'inTheMoney', 'volume', 'openInterest'
    ]
    
    result = []
    
    # Display important fields first
    for key in important_fields:
        if key in option_data:
            value = option_data[key]
            if isinstance(value, float):
                result.append(f"  - {key}: {value:.4f}")
            else:
                result.append(f"  - {key}: {value}")
    
    # Display any other available fields
    for key, value in option_data.items():
        if key not in important_fields and key not in ['strike', 'delta']:
            if isinstance(value, float):
                result.append(f"  - {key}: {value:.4f}")
            else:
                result.append(f"  - {key}: {value}")
    
    return '\n'.join(result)

def main():
    """Main function"""
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Set default expiry date if not provided
    if not args.expiry:
        today = datetime.datetime.now()
        expiry_date = (today + datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        logger.info(f"No expiry date provided, using default: {expiry_date}")
    else:
        expiry_date = args.expiry
        try:
            # Validate date format
            datetime.datetime.strptime(expiry_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {expiry_date}. Please use YYYY-MM-DD format.")
            return
    
    print("\n" + "="*80)
    print(f"{'CUSTOM OPTION TEST':^80}")
    print("="*80)
    
    print(f"\nParameters:")
    print(f"  - Ticker: {args.ticker}")
    print(f"  - Option Type: {args.option_type}")
    print(f"  - Target Expiry Date: {expiry_date}")
    print(f"  - Target Delta: {args.delta}")
    
    try:
        # Fetch option data
        print("\nFetching option data...")
        result = fetch_option_data(
            ticker=args.ticker,
            option_type=args.option_type,
            target_expiry_date=expiry_date,
            target_delta=args.delta
        )
        
        # Display results
        print("\n" + "="*80)
        print(f"{'RESULTS':^80}")
        print("="*80)
        
        print(f"\nUnderlying Asset:")
        print(f"  - Ticker: {result['ticker']}")
        print(f"  - Current Price: ${result['underlying_price']:.2f}")
        
        print(f"\nOption Details:")
        print(f"  - Option Type: {result['option_type'].upper()}")
        print(f"  - Strike Price: ${result['strike']:.2f}")
        print(f"  - Delta: {result['delta']:.4f}")
        
        # Calculate % OTM/ITM
        strike = result['strike']
        current_price = result['underlying_price']
        pct_diff = (strike - current_price) / current_price * 100
        
        if args.option_type.lower() == 'call':
            if pct_diff > 0:
                status = f"{abs(pct_diff):.2f}% OTM"
            else:
                status = f"{abs(pct_diff):.2f}% ITM"
        else:  # put
            if pct_diff < 0:
                status = f"{abs(pct_diff):.2f}% OTM"
            else:
                status = f"{abs(pct_diff):.2f}% ITM"
        
        print(f"  - Status: {status}")
        
        print(f"\nExpiry Information:")
        print(f"  - Target Expiry Date: {expiry_date}")
        print(f"  - Actual Expiry Date: {result['expiry_date']}")
        
        # Calculate days difference
        target_date = datetime.datetime.strptime(expiry_date, '%Y-%m-%d').date()
        actual_date = datetime.datetime.strptime(result['expiry_date'], '%Y-%m-%d').date()
        days_diff = abs((actual_date - target_date).days)
        
        if days_diff == 0:
            print(f"  - Exact match found!")
        else:
            print(f"  - Days Difference: {days_diff} days")
        
        # Calculate days to expiry
        today = datetime.datetime.now().date()
        days_to_expiry = (actual_date - today).days
        print(f"  - Days to Expiry: {days_to_expiry} days")
        
        print(f"\nAdditional Option Data:")
        print(format_option_data(result['option_data']))
        
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 