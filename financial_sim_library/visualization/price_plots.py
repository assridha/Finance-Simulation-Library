"""
Stock Price Visualization

This module provides functions for visualizing stock price simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import logging

# Set up logger
logger = logging.getLogger(__name__)

def plot_price_simulations(simulation_results, num_paths_to_plot=5, confidence_interval=0.95, 
                          show_mean=True, show_ci=True, figsize=(12, 8), save_path=None):
    """
    Plot the simulated price paths with confidence intervals.
    
    Args:
        simulation_results (dict): Results from StockPriceModel.simulate()
        num_paths_to_plot (int, optional): Number of individual paths to plot. Defaults to 5.
        confidence_interval (float, optional): Confidence interval to display (0-1). Defaults to 0.95.
        show_mean (bool, optional): Whether to show the mean path. Defaults to True.
        show_ci (bool, optional): Whether to show confidence intervals. Defaults to True.
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract data from simulation results
    ticker = simulation_results['ticker']
    current_price = simulation_results['current_price']
    price_paths = simulation_results['price_paths']
    dates = simulation_results['dates']
    
    # Calculate statistics if not already in results
    if 'statistics' not in simulation_results:
        # Calculate mean and percentiles for confidence intervals
        mean_path = np.mean(price_paths, axis=0)
        lower_percentile = (1 - confidence_interval) / 2 * 100
        upper_percentile = (1 + confidence_interval) / 2 * 100
        lower_ci = np.percentile(price_paths, lower_percentile, axis=0)
        upper_ci = np.percentile(price_paths, upper_percentile, axis=0)
    else:
        # If statistics are already calculated, extract mean path
        mean_path = simulation_results['mean_path'] if 'mean_path' in simulation_results else np.mean(price_paths, axis=0)
        
        # Calculate confidence intervals if not provided
        lower_percentile = (1 - confidence_interval) / 2 * 100
        upper_percentile = (1 + confidence_interval) / 2 * 100
        lower_ci = np.percentile(price_paths, lower_percentile, axis=0)
        upper_ci = np.percentile(price_paths, upper_percentile, axis=0)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Format dates for x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))  # Dynamic interval
    
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
    title = f"{ticker} Price Simulation\n"
    if 'parameters' in simulation_results:
        params = simulation_results['parameters']
        if 'volatility' in params:
            title += f"Volatility: {params['volatility']:.2%}, "
        if 'risk_free_rate' in params:
            title += f"Risk-Free Rate: {params['risk_free_rate']:.2%}, "
        if 'days_simulated' in params:
            title += f"Days: {params['days_simulated']}"
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    
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

def plot_price_distribution(simulation_results, date_index=-1, bins=50, 
                           figsize=(12, 8), save_path=None):
    """
    Plot the distribution of simulated prices at a specific date.
    
    Args:
        simulation_results (dict): Results from StockPriceModel.simulate()
        date_index (int, optional): Index of the date to plot. Defaults to -1 (final date).
        bins (int, optional): Number of histogram bins. Defaults to 50.
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract data from simulation results
    ticker = simulation_results['ticker']
    current_price = simulation_results['current_price']
    price_paths = simulation_results['price_paths']
    dates = simulation_results['dates']
    
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
    title = f"{ticker} Price Distribution on {selected_date.strftime('%Y-%m-%d')}\n"
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
    days_from_now = (selected_date - dates[0]).days
    
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
        plt.show()
    
    return fig 