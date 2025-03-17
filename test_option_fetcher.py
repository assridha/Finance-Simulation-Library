import pandas as pd
from option_fetcher import fetch_option_data
import datetime
import logging

# Configure logging - set to WARNING to reduce output from the test script
# The option_fetcher module will still log at INFO level
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define test parameters
    tickers = ['AAPL', 'SPY']
    option_types = ['call', 'put']
    
    # Get dates for testing
    today = datetime.datetime.now()
    
    # Test with different expiry dates (30, 60, 90 days from now)
    expiry_dates = [
        (today + datetime.timedelta(days=30)).strftime('%Y-%m-%d'),  # 1 month
        (today + datetime.timedelta(days=60)).strftime('%Y-%m-%d'),  # 2 months
        (today + datetime.timedelta(days=90)).strftime('%Y-%m-%d'),  # 3 months
    ]
    
    # Test with different delta values
    delta_values = [0.5]  # Using just one delta value to keep output manageable
    
    print("\n" + "="*80)
    print(f"{'OPTION FETCHER TEST RESULTS':^80}")
    print("="*80 + "\n")
    
    for ticker in tickers:
        print(f"\n{'#'*80}")
        print(f"{'TESTING WITH TICKER: ' + ticker:^80}")
        print(f"{'#'*80}\n")
        
        for option_type in option_types:
            print(f"\n{'-'*80}")
            print(f"{option_type.upper() + ' OPTIONS':^80}")
            print(f"{'-'*80}")
            
            for target_date in expiry_dates:
                print(f"\n{'='*70}")
                print(f"{'TARGET EXPIRY DATE: ' + target_date:^70}")
                print(f"{'='*70}")
                
                for delta in delta_values:
                    try:
                        print(f"\n{'*'*60}")
                        print(f"Target Parameters:")
                        print(f"  - Delta: {delta}")
                        print(f"  - Expiry Date: {target_date}")
                        print(f"{'*'*60}")
                        
                        # Fetch option data
                        result = fetch_option_data(
                            ticker=ticker,
                            option_type=option_type,
                            target_expiry_date=target_date,
                            target_delta=delta
                        )
                        
                        # Display results
                        print("\nResults:")
                        print(f"  - Underlying Price: ${result['underlying_price']:.2f}")
                        print(f"  - Actual Expiry Date: {result['expiry_date']}")
                        print(f"  - Days Difference: {abs((datetime.datetime.strptime(result['expiry_date'], '%Y-%m-%d').date() - datetime.datetime.strptime(target_date, '%Y-%m-%d').date()).days)} days")
                        print(f"  - Actual Delta: {result['delta']:.4f}")
                        print(f"  - Strike Price: ${result['strike']:.2f}")
                        
                        # Calculate % OTM/ITM
                        strike = result['strike']
                        current_price = result['underlying_price']
                        pct_diff = (strike - current_price) / current_price * 100
                        
                        if option_type.lower() == 'call':
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
                        
                        # Display more option details
                        option_data = result['option_data']
                        print("\nOption Details:")
                        
                        # Define important fields to display
                        important_fields = [
                            'contractSymbol', 'lastTradeDate', 'bid', 'ask', 'lastPrice',
                            'impliedVolatility', 'inTheMoney', 'volume', 'openInterest'
                        ]
                        
                        # Display important fields first
                        for key in important_fields:
                            if key in option_data:
                                value = option_data[key]
                                if isinstance(value, float):
                                    print(f"  - {key}: {value:.4f}")
                                else:
                                    print(f"  - {key}: {value}")
                        
                        # Display any other available fields
                        for key, value in option_data.items():
                            if key not in important_fields and key not in ['strike', 'delta']:
                                if isinstance(value, float):
                                    print(f"  - {key}: {value:.4f}")
                                else:
                                    print(f"  - {key}: {value}")
                        
                    except Exception as e:
                        print(f"\nError: {str(e)}")
                    
                    print("\n" + "-"*80)
        
        print("\n")

if __name__ == "__main__":
    main() 