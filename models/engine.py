import numpy as np
import scipy.linalg as linalg

class MonteCarloPricingEngine:
    def __init__(self, S0, K, T, r, sigma, num_simulations=10000, num_steps=252):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.dt = T / num_steps

    def generate_paths(self, seed=None):
        """Generates paths using Antithetic Variates for variance reduction."""
        if seed:
            np.random.seed(seed)
            
        # Generate half the shocks and mirror them
        half_sims = self.num_simulations // 2
        Z_half = np.random.standard_normal((self.num_steps, half_sims))
        Z = np.concatenate([Z_half, -Z_half], axis=1)

        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        
        path_returns = np.exp(drift + diffusion)
        paths = np.vstack([np.ones(self.num_simulations) * self.S0, path_returns])
        paths = np.cumprod(paths, axis=0)
        return paths

    def price_european_option(self, option_type="call", use_control_variate=True):
        """
        Prices a European option, optionally using the Martingale property 
        of the discounted stock price as a Control Variate for variance reduction.
        """
        paths = self.generate_paths()
        S_T = paths[-1] # Terminal prices at maturity
        
        # 1. Calculate raw payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(S_T - self.K, 0)
        else: # Put Option
            payoffs = np.maximum(self.K - S_T, 0)
            
        # Discounted payoffs (Y)
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        if use_control_variate:
            # 2. Control Variate (X): The discounted stock price
            # Under Q, e^{-rT} S_T is a martingale, so its expected value is S0
            X = np.exp(-self.r * self.T) * S_T
            EX = self.S0
            
            # 3. Calculate optimal covariance multiplier (c*)
            covariance = np.cov(X, discounted_payoffs)[0, 1]
            variance_X = np.var(X)
            c_star = covariance / variance_X if variance_X > 0 else 0
            
            # 4. Apply the control variate adjustment
            # Y_adjusted = Y - c*(X - E[X])
            Y_cv = discounted_payoffs - c_star * (X - EX)
            
            price = np.mean(Y_cv)
            
            # Optional: Calculate Variance Reduction Factor (for logging/dashboard)
            # var_orig = np.var(discounted_payoffs)
            # var_cv = np.var(Y_cv)
            # variance_reduction = 1 - (var_cv / var_orig)
        else:
            price = np.mean(discounted_payoffs)
            
        return price, paths

    def calculate_greeks(self, option_type="call", h_s=1.0, h_v=0.01):
        """Calculates Delta and Vega using Central Difference Method."""
        orig_s0, orig_sigma = self.S0, self.sigma
        
        # Delta
        self.S0 = orig_s0 + h_s
        p_up, _ = self.price_european_option(option_type, use_control_variate=True)
        self.S0 = orig_s0 - h_s
        p_down, _ = self.price_european_option(option_type, use_control_variate=True)
        delta = (p_up - p_down) / (2 * h_s)
        self.S0 = orig_s0 # Reset

        # Vega
        self.sigma = orig_sigma + h_v
        p_v_up, _ = self.price_european_option(option_type, use_control_variate=True)
        vega = (p_v_up - ((p_up + p_down)/2)) / (h_v * 100)
        self.sigma = orig_sigma # Reset

        return {"Delta": delta, "Vega": vega}


class FiniteDifferencePricingEngine:
    """
    Prices European options solving the Black-Scholes PDE via Implicit FDM.
    Demonstrates the Feynman-Kac representation.
    """
    def __init__(self, S0, K, T, r, sigma, S_max_multiplier=3, M=100, N=100):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M
        self.N = N
        self.S_max = S0 * S_max_multiplier
        self.dS = self.S_max / M
        self.dt = T / N

    def price_european_option(self, option_type="call"):
        S = np.linspace(0, self.S_max, self.M + 1)
        
        if option_type.lower() == "call":
            V = np.maximum(S - self.K, 0)
        else:
            V = np.maximum(self.K - S, 0)
            
        # --- THE FIX: Correct signs for the IMPLICIT scheme ---
        j = np.arange(1, self.M)
        alpha = -0.5 * self.dt * (self.sigma**2 * j**2 - self.r * j)
        beta  = 1.0 + self.dt * (self.sigma**2 * j**2 + self.r)
        gamma = -0.5 * self.dt * (self.sigma**2 * j**2 + self.r * j)
        # ------------------------------------------------------
        
        A = np.diag(beta) + np.diag(alpha[1:], -1) + np.diag(gamma[:-1], 1)
        
        for i in range(self.N - 1, -1, -1):
            offset = np.zeros(self.M - 1)
            time_to_maturity = self.T - i * self.dt
            
            if option_type.lower() == "call":
                offset[-1] = gamma[-1] * (self.S_max - self.K * np.exp(-self.r * time_to_maturity))
            else: 
                offset[0] = alpha[0] * (self.K * np.exp(-self.r * time_to_maturity))
                
            V[1:self.M] = linalg.solve(A, V[1:self.M] - offset)
            
            if option_type.lower() == "call":
                V[0] = 0
                V[-1] = self.S_max - self.K * np.exp(-self.r * time_to_maturity)
            else:
                V[0] = self.K * np.exp(-self.r * time_to_maturity)
                V[-1] = 0
                
        price = np.interp(self.S0, S, V)
        return price
