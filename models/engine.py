import numpy as np

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
        """Generates paths using Antithetic Variates for better convergence."""
        if seed:
            np.random.seed(seed)
            
        # Generate half the shocks and mirror them to reduce variance
        half_sims = self.num_simulations // 2
        Z_half = np.random.standard_normal((self.num_steps, half_sims))
        Z = np.concatenate([Z_half, -Z_half], axis=1)

        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        
        path_returns = np.exp(drift + diffusion)
        paths = np.vstack([np.ones(self.num_simulations) * self.S0, path_returns])
        paths = np.cumprod(paths, axis=0)
        return paths

    def price_european_option(self, option_type="call"):
        """Prices a Call or Put option and returns the price and full paths."""
        paths = self.generate_paths()
        S_T = paths[-1] # Terminal prices at maturity
        
        # Determine payoff based on option type
        if option_type.lower() == "call":
            payoffs = np.maximum(S_T - self.K, 0)
        else: # Put Option
            payoffs = np.maximum(self.K - S_T, 0)
            
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return price, paths

    def calculate_greeks(self, option_type="call", h_s=1.0, h_v=0.01):
        """
        Calculates Delta and Vega using the Central Difference Method.
        Delta (Δ): Sensitivity to Stock Price.
        Vega (ν): Sensitivity to 1% change in Volatility.
        """
        # Save original values
        orig_s0, orig_sigma = self.S0, self.sigma
        
        # --- Delta (Central Difference) ---
        self.S0 = orig_s0 + h_s
        p_up, _ = self.price_european_option(option_type)
        self.S0 = orig_s0 - h_s
        p_down, _ = self.price_european_option(option_type)
        delta = (p_up - p_down) / (2 * h_s)
        self.S0 = orig_s0 # Reset

        # --- Vega (Bumping Volatility) ---
        self.sigma = orig_sigma + h_v
        p_v_up, _ = self.price_european_option(option_type)
        vega = (p_v_up - ((p_up + p_down)/2)) / (h_v * 100) # Per 1% vol change
        self.sigma = orig_sigma # Reset

        return {"Delta": delta, "Vega": vega}
    
