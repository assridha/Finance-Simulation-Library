from setuptools import setup, find_packages

setup(
    name="financial_sim_library",
    version="0.2.0",
    description="A library for financial simulations and option pricing",
    author="Ashwin Sridhar",
    author_email="assridha@gmail.com",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'yfinance',
        'scipy',
        'flask',
        'flask-cors'
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/assridha/Finance-Simulation-Library",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 