"""
Stock Price Visualization

This module provides functions for visualizing stock price simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
import pandas as pd
import mplfinance as mpf
from typing import Optional, Dict, List, Tuple

# Set up logger
logger = logging.getLogger(__name__)

def plot_price_simulations(simulation_results, title=None, num_paths_to_plot=5, confidence_interval=0.95, 
                          show_mean=True, show_ci=True, figsize=(12, 8), save_path=None,
                          historical_data: Optional[pd.DataFrame] = None, days_to_show: int = 30):
    """
    Plot the simulated price paths with confidence intervals and optional historical candlestick data.
    
    Args:
        simulation_results (dict): Results from StockPriceModel.simulate()
        title (str, optional): Custom title for the plot. If None, a default title is used.
        num_paths_to_plot (int, optional): Number of individual paths to plot. Defaults to 5.
        confidence_interval (float, optional): Confidence interval to display (0-1). Defaults to 0.95.
        show_mean (bool, optional): Whether to show the mean path. Defaults to True.
        show_ci (bool, optional): Whether to show confidence intervals. Defaults to True.
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
        historical_data (pd.DataFrame, optional): Historical OHLCV data for candlestick chart.
            DataFrame should have columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        days_to_show (int, optional): Number of days of historical data to show. Defaults to 30.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract data from simulation results
    price_paths = simulation_results['price_paths']
    current_price = simulation_results['current_price']
    time_points = simulation_results['time_points']
    
    # Convert time points to dates
    start_date = datetime.now()
    # Convert time points to business days (excluding weekends)
    dates = []
    for t in time_points:
        # Calculate target date by adding business days
        current_date = start_date
        business_days_to_add = int(t * 252)  # 252 trading days per year
        days_added = 0
        
        while days_added < business_days_to_add:
            current_date += timedelta(days=1)
            # Skip weekends (5=Saturday, 6=Sunday)
            if current_date.weekday() < 5:  # Monday-Friday
                days_added += 1
                
        dates.append(current_date)
    
    # Calculate statistics
    mean_path = np.mean(price_paths, axis=0)
    lower_percentile = (1 - confidence_interval) / 2 * 100
    upper_percentile = (1 + confidence_interval) / 2 * 100
    lower_ci = np.percentile(price_paths, lower_percentile, axis=0)
    upper_ci = np.percentile(price_paths, upper_percentile, axis=0)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Format dates for x-axis with dynamic tick placement
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Calculate optimal number of ticks based on figure width
    fig_width = figsize[0]
    optimal_ticks = max(5, min(15, int(fig_width / 2)))  # Between 5 and 15 ticks
    
    # Calculate date range
    if historical_data is not None:
        date_range = historical_data.index[-1] - historical_data.index[0]
    else:
        date_range = dates[-1] - dates[0]
    
    # Choose appropriate locator based on date range
    if date_range.days <= 30:
        # For short ranges, show daily ticks
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//optimal_ticks)))
    elif date_range.days <= 90:
        # For medium ranges, show weekly ticks
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=max(1, len(dates)//(optimal_ticks*5))))
    else:
        # For long ranges, show monthly ticks
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates)//(optimal_ticks*20))))
    
    # Calculate y-axis limits based on all data
    y_min = float('inf')
    y_max = float('-inf')
    
    # Initialize x-axis limits
    x_min = None
    x_max = None
    
    # Plot historical candlestick data if provided
    if historical_data is not None:
        # Filter historical data to show only the last N days
        historical_data = historical_data.tail(days_to_show)
        
        # Update y-axis limits with historical data
        y_min = min(y_min, historical_data['Low'].min())
        y_max = max(y_max, historical_data['High'].max())
        
        # Update x-axis limits with historical data
        x_min = historical_data.index[0]
        x_max = historical_data.index[-1]
        
        # Plot candlesticks
        mpf.plot(historical_data, type='candle', style='charles',
                ax=ax, volume=False, show_nontrading=True,
                datetime_format='%Y-%m-%d', xrotation=45,
                returnfig=True)  # Add returnfig=True to prevent automatic display
        
        # Adjust the simulation plot dates to start from the last historical date
        last_hist_date = historical_data.index[-1]
        simulation_start = last_hist_date + timedelta(days=1)
        
        # Convert time points to business days (excluding weekends)
        dates = []
        for t in time_points:
            # Calculate target date by adding business days
            current_date = simulation_start
            business_days_to_add = int(t * 252)  # 252 trading days per year
            days_added = 0
            
            while days_added < business_days_to_add:
                current_date += timedelta(days=1)
                # Skip weekends (5=Saturday, 6=Sunday)
                if current_date.weekday() < 5:  # Monday-Friday
                    days_added += 1
                    
            dates.append(current_date)
    
    # Update y-axis limits with simulation data
    if show_ci:
        y_min = min(y_min, lower_ci.min())
        y_max = max(y_max, upper_ci.max())
    else:
        y_min = min(y_min, price_paths.min())
        y_max = max(y_max, price_paths.max())
    
    # Add padding to y-axis limits (5% on each side)
    y_range = y_max - y_min
    y_min = y_min - 0.05 * y_range
    y_max = y_max + 0.05 * y_range
    
    # Set y-axis limits
    ax.set_ylim(y_min, y_max)
    
    # Update x-axis limits with simulation dates
    if x_min is None:
        x_min = dates[0]
    x_max = max(x_max, dates[-1])
    
    # Add padding to x-axis limits (1 day on each side)
    x_min = x_min - timedelta(days=1)
    x_max = x_max + timedelta(days=1)
    
    # Set x-axis limits
    ax.set_xlim(x_min, x_max)
    
    # Plot confidence intervals
    if show_ci:
        ax.fill_between(dates, lower_ci, upper_ci, color='lightblue', alpha=0.5, 
                       label=f'{int(confidence_interval*100)}% Confidence Interval')
    
    # Plot mean path
    if show_mean:
        ax.plot(dates, mean_path, 'b-', linewidth=2, label='Mean Path')
    
    # Plot individual paths (sample)
    if num_paths_to_plot > 0:
        # Randomly select paths to plot
        if price_paths.shape[0] > num_paths_to_plot:
            indices = np.random.choice(price_paths.shape[0], num_paths_to_plot, replace=False)
        else:
            indices = range(price_paths.shape[0])
        
        for i in indices:
            ax.plot(dates, price_paths[i], 'k-', alpha=0.2)
    
    # Mark starting price
    ax.plot(dates[0], current_price, 'go', markersize=8, label='Starting Price')
    
    # Set title and labels
    if title is None:
        title = "Price Simulation\n"
        if 'parameters' in simulation_results:
            params = simulation_results['parameters']
            if 'volatility' in params:
                title += f"Volatility: {params['volatility']:.2%}, "
            if 'risk_free_rate' in params:
                title += f"Risk-Free Rate: {params['risk_free_rate']:.2%}"
            
            # Add growth model information
            if 'growth_model' in params:
                growth_model = params['growth_model']
                model_type = growth_model['metadata']['model_type']
                
                if model_type == 'fixed':
                    title += f"\nGrowth Model: {growth_model['metadata']['description']} ({growth_model['growth_rate']:.1%} annual)"
                elif model_type == 'power_law':
                    title += f"\nGrowth Model: {growth_model['metadata']['description']} (k={growth_model['k']:.2f})"
                elif model_type == 'exogenous':
                    title += f"\nGrowth Model: {growth_model['metadata']['description']}"
                elif model_type == 'composite':
                    title += f"\nGrowth Model: {growth_model['metadata']['description']}"
                    for name, component in growth_model['models'].items():
                        component_type = component['metadata']['model_type']
                        if component_type == 'fixed':
                            title += f"\n  - {name}: Fixed ({component['growth_rate']:.1%} annual)"
                        elif component_type == 'power_law':
                            title += f"\n  - {name}: Power Law (k={component['k']:.2f})"
                        elif component_type == 'exogenous':
                            title += f"\n  - {name}: Exogenous"
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')

    # Apply semilogy scale to y-axis (logarithmic scale) if price gain is significant
    if np.max(mean_path) / current_price > 5.0:
        ax.set_yscale('log')
        logger.info("Applied logarithmic scale to y-axis due to significant price gain (>5x)")
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    fig.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    
    return fig

def plot_price_distribution(simulation_results, title=None, date_index=-1, bins=50, 
                           figsize=(12, 8), save_path=None):
    """
    Plot the distribution of simulated prices at a specific date.
    
    Args:
        simulation_results (dict): Results from StockPriceModel.simulate()
        title (str, optional): Custom title for the plot. If None, a default title is used.
        date_index (int, optional): Index of the date to plot. Defaults to -1 (final date).
        bins (int, optional): Number of histogram bins. Defaults to 50.
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract data from simulation results
    price_paths = simulation_results['price_paths']
    current_price = simulation_results['current_price']
    time_points = simulation_results['time_points']
    
    # Convert time points to dates
    start_date = datetime.now()
    dates = [start_date + timedelta(days=int(t*252)) for t in time_points]
    
    # Get prices at the specified date
    prices_at_date = price_paths[:, date_index]
    selected_date = dates[date_index]
    
    # Calculate statistics
    mean_price = np.mean(prices_at_date)
    median_price = np.median(prices_at_date)
    std_dev = np.std(prices_at_date)
    
    # Calculate percentiles
    percentiles = {
        '5%': np.percentile(prices_at_date, 5),
        '25%': np.percentile(prices_at_date, 25),
        '50%': np.percentile(prices_at_date, 50),
        '75%': np.percentile(prices_at_date, 75),
        '95%': np.percentile(prices_at_date, 95)
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins, patches = ax.hist(prices_at_date, bins=bins, alpha=0.7, color='skyblue')
    
    # Add vertical lines for key statistics
    ax.axvline(x=current_price, color='green', linestyle='-', linewidth=2, label=f'Current: ${current_price:.2f}')
    ax.axvline(x=mean_price, color='red', linestyle='-', linewidth=2, label=f'Mean: ${mean_price:.2f}')
    ax.axvline(x=median_price, color='blue', linestyle='--', linewidth=2, label=f'Median: ${median_price:.2f}')
    
    # Add vertical lines for percentiles
    for label, value in percentiles.items():
        ax.axvline(x=value, color='purple', linestyle=':', alpha=0.5, linewidth=1)
    
    # Set title and labels
    if title is None:
        title = f"Price Distribution on {selected_date.strftime('%Y-%m-%d')}\n"
        if 'parameters' in simulation_results:
            params = simulation_results['parameters']
            if 'volatility' in params:
                title += f"Volatility: {params['volatility']:.2%}, "
            if 'risk_free_rate' in params:
                title += f"Risk-Free Rate: {params['risk_free_rate']:.2%}"
    
    ax.set_title(title)
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    
    # Add text box with statistics
    stats_text = (
        f"Statistics:\n"
        f"Mean: ${mean_price:.2f}\n"
        f"Median: ${median_price:.2f}\n"
        f"Std Dev: ${std_dev:.2f}\n"
        f"\nPercentiles:\n"
        f"5%: ${percentiles['5%']:.2f}\n"
        f"25%: ${percentiles['25%']:.2f}\n"
        f"50%: ${percentiles['50%']:.2f}\n"
        f"75%: ${percentiles['75%']:.2f}\n"
        f"95%: ${percentiles['95%']:.2f}\n"
    )
    
    # Add text box
    plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add expected return information
    expected_return = (mean_price / current_price - 1) * 100
    prob_above_current = np.mean(prices_at_date > current_price) * 100
    days_from_now = int(time_points[date_index] * 252)
    
    return_text = (
        f"Expected Return: {expected_return:.2f}%\n"
        f"Probability Above Current: {prob_above_current:.1f}%\n"
        f"Days from Start: {days_from_now}"
    )
    
    plt.text(0.95, 0.95, return_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        # Clear any existing plots to prevent double display
        plt.close('all')
        plt.show()
    
    return fig

def plot_price_heatmap(simulation_results, figsize=(14, 8), save_path=None):
    """
    Plot a heatmap of price probabilities over time.
    
    Args:
        simulation_results (dict): Results from StockPriceModel.simulate()
        figsize (tuple, optional): Figure size. Defaults to (14, 8).
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract data from simulation results
    ticker = simulation_results['ticker']
    current_price = simulation_results['current_price']
    price_paths = simulation_results['price_paths']
    dates = simulation_results['dates']
    
    # Calculate price range for histogram bins
    min_price = np.min(price_paths)
    max_price = np.max(price_paths)
    
    # Add a buffer to the price range
    price_range = max_price - min_price
    min_price -= price_range * 0.05
    max_price += price_range * 0.05
    
    # Create price bins
    num_price_bins = 50
    price_bins = np.linspace(min_price, max_price, num_price_bins + 1)
    price_centers = (price_bins[:-1] + price_bins[1:]) / 2
    
    # Initialize heatmap data
    heatmap_data = np.zeros((num_price_bins, len(dates)))
    
    # Calculate probability density for each date
    for t in range(len(dates)):
        hist, _ = np.histogram(price_paths[:, t], bins=price_bins, density=True)
        heatmap_data[:, t] = hist
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis',
                  extent=[0, len(dates) - 1, 0, num_price_bins - 1])
    
    # Create colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability Density')
    
    # Set x-axis ticks and labels (dates)
    num_date_ticks = min(10, len(dates))
    date_indices = np.linspace(0, len(dates) - 1, num_date_ticks, dtype=int)
    date_labels = [dates[i].strftime('%Y-%m-%d') for i in date_indices]
    ax.set_xticks(date_indices)
    ax.set_xticklabels(date_labels, rotation=45)
    
    # Set y-axis ticks and labels (prices)
    num_price_ticks = 10
    price_indices = np.linspace(0, num_price_bins - 1, num_price_ticks, dtype=int)
    price_labels = [f'${price_centers[i]:.0f}' for i in price_indices]
    ax.set_yticks(price_indices)
    ax.set_yticklabels(price_labels)
    
    # Draw the current price line
    current_price_idx = np.argmin(np.abs(price_centers - current_price))
    ax.axhline(y=current_price_idx, color='r', linestyle='-', linewidth=1, label=f'Current: ${current_price:.2f}')
    
    # Set title and labels
    title = f"{ticker} Price Probability Heatmap\n"
    if 'parameters' in simulation_results:
        params = simulation_results['parameters']
        if 'volatility' in params:
            title += f"Volatility: {params['volatility']:.2%}, "
        if 'risk_free_rate' in params:
            title += f"Risk-Free Rate: {params['risk_free_rate']:.2%}"
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    
    # Add legend
    ax.legend(loc='upper left')
    
    fig.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        # Clear any existing plots to prevent double display
        plt.close('all')
        plt.show()
    
    return fig 