# Advanced Quantitative Pricing Engine & Risk Analytics Dashboard

An interactive, production-grade options pricing engine and risk analytics dashboard. This project bridges theoretical stochastic calculus with modern data science practices, providing a comprehensive toolkit for simulating, pricing, and analyzing European options under varying probability measures.

## 📌 Project Overview

This engine goes beyond standard Black-Scholes implementations by integrating advanced numerical methods and stochastic integration techniques. It serves as a practical, computational proof of core financial mathematics theorems, featuring a dual-engine architecture that compares probabilistic simulations with deterministic partial differential equation (PDE) solvers.

### 🚀 Core Advanced Features

#### 1. The Feynman-Kac Representation (PDE vs. Monte Carlo)
The engine demonstrates the Feynman-Kac theorem by pricing options using two distinct methods simultaneously:
* **Monte Carlo Simulation:** A probabilistic approach simulating Geometric Brownian Motion (GBM) paths.
* **Implicit Finite Difference Method (FDM):** A deterministic solver for the Black-Scholes Cauchy problem. 
* **Result:** Validates that the expected value of the stochastic payoff exactly matches the deterministic PDE solution at $t=0$, ensuring robust, cross-verified pricing.

#### 2. Girsanov's Theorem & Measure Change ($\mathbb{P}$ vs. $\mathbb{Q}$)
The data pipeline explicitly models the transition between the physical and risk-neutral worlds:
* **Real-World Measure ($\mathbb{P}$):** Extracts historical realized volatility ($\sigma$) and true historical drift ($\mu$) from market data.
* **Risk-Neutral Measure ($\mathbb{Q}$):** Prices derivatives utilizing the risk-free rate ($r$).
* **Market Price of Risk ($\theta$):** Calculates the compensation demanded per unit of volatility: $\theta = \frac{\mu - r}{\sigma}$.
* **Result:** The dashboard visually plots expected asset paths under both measures, demonstrating the Radon-Nikodym derivative's effect on the Wiener process: $dW_t^{\mathbb{Q}} = dW_t^{\mathbb{P}} + \theta dt$.

#### 3. Martingale Variance Reduction
To achieve production-level computational efficiency, the Monte Carlo engine utilizes advanced variance reduction techniques:
* **Martingale Control Variates:** Leverages the property that the discounted stock price is a martingale under $\mathbb{Q}$ (i.e., $\mathbb{E}^{\mathbb{Q}}[e^{-rT} S_T] = S_0$). By using the simulated asset path as a control variate, the engine drastically reduces simulation variance and sampling error.
* **Antithetic Variates:** Mirrors random shocks to ensure symmetric distributions and faster convergence.

---

## 🛠️ Technical Stack
* **Backend Mathematical Engine:** `NumPy` (Vectorized Operations), `SciPy` (Linear Algebra for Implicit FDM).
* **Data Pipeline:** `pandas`, `yfinance` (Live Market Data & Historical Parameter Extraction).
* **Frontend Dashboard:** `Dash`, `Plotly` (Interactive UI, Real-time Visualizations).

## 📂 Repository Structure

```text
├── data/ 

│   ├── .gitkeep

│   └── market_data_fetcher.py 

├── models/ 

│   ├── __init__.py 

│   └── engine.py  

├── app.py 

├── requirements.txt 

├── .gitignore

└── README.md