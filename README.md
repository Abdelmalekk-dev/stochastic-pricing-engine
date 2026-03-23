# Quantitative Options Pricing Engine & Volatility Analyzer

A full-stack financial engineering and data science pipeline that prices European options using a vectorized Monte Carlo engine, analyzes real-world market volatility, and deploys the results via an interactive web dashboard.

## Project Overview

This project bridges theoretical quantitative finance and applied machine learning. It was built to demonstrate proficiency in stochastic calculus, object-oriented software architecture, and data pipeline engineering.

### Core Features

1. **Vectorized Pricing Engine (`models/engine.py`)**
   - Solves the Stochastic Differential Equation (SDE) for Geometric Brownian Motion.
   - Utilizes `NumPy` vectorization to simulate thousands of asset paths simultaneously without relying on slow `for` loops.
   - Calculates the discounted expected payoff based on the Law of Total Expectation.

2. **Market Data Pipeline (`data/market_data_fetcher.py`)**
   - Automated script using `yfinance` to ingest historical daily closing prices for major DAX-listed equities.
   - Cleans missing values, calculates logarithmic returns, and computes rolling 30-day annualized historical volatility.

3. **Machine Learning & Volatility Analysis (`notebooks/`)**
   - **Convergence Proofs:** Visually demonstrates the Central Limit Theorem and the convergence of the Monte Carlo estimate to the theoretical true price as simulations approach infinity.
   - **Volatility Smile Regression:** Uses `scikit-learn` to build a polynomial regression model that maps the non-linear "Volatility Smile" observed in real-world options markets, contrasting it against the Black-Scholes assumption of constant volatility.

4. **Interactive Dashboard (`app.py`)**
   - A lightweight web application built with `Dash` and `Plotly`.
   - Allows users to dynamically adjust strike prices, time to maturity, and volatility to visualize the shifting simulated asset paths in real-time.

## Repository Structure

├── data/ 
│   ├── .gitkeep
│   └── market_data_fetcher.py 
├── models/ 
│   ├── __init__.py 
│   └── engine.py 
├── notebooks/ 
│   ├── 01_distribution_and_convergence.ipynb 
│   └── 02_implied_volatility_regression.ipynb 
├── app.py 
├── requirements.txt 
├── .gitignore
└── README.md


## Installation & Usage

1. **Clone the repository:**
   git clone https://github.com/YourUsername/stochastic-pricing-engine.git
   cd stochastic-pricing-engine

2. **Install dependencies:**
   pip install -r requirements.txt

3. **Fetch real-world market data:**
   python data/market_data_fetcher.py

4. **Run the interactive dashboard:**
   python app.py

## Tech Stack
* **Mathematics & Core Logic:** Python, NumPy, SciPy
* **Data Engineering:** Pandas, yfinance
* **Machine Learning:** Scikit-learn
* **Visualization & Deployment:** Matplotlib, Seaborn, Dash, Plotly
* **Development Environment:** Jupyter Notebook

## License
This project is licensed under the MIT License.