from setuptools import setup, find_packages

setup(
    name="financial_sim_library",
    version="0.1.2",
    description="A comprehensive financial simulation library for stock prices, options, and portfolios",
    author="Ashwin Sridhar",
    packages=find_packages(),
    install_requires=[
        "yfinance>=0.2.54",
        "pandas>=1.3.5",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 