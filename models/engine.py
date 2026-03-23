import numpy as np

class MonteCarloPricingEngine:
    """
    A vectorized Monte Carlo engine for pricing European options using 
    Geometric Brownian Motion (GBM).
    """
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, 
                 num_simulations: int = 10000, num_steps: int = 252):
        """
        Initializes the pricing engine.
        
        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Annualized volatility
        num_simulations (int): Number of simulated price paths
        num_steps (int): Number of time steps per path (default 252 trading days)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        
        # Time increment
        self.dt = self.T / self.num_steps

    def simulate_paths(self) -> np.ndarray:
        """
        Simulates asset price paths using Geometric Brownian Motion.
        
        Returns:
        np.ndarray: A 2D array of simulated paths where rows are time steps 
                    and columns are individual simulations.
        """
        # Generate random standard normal variables for the Brownian motion
        Z = np.random.standard_normal((self.num_steps, self.num_simulations))
        
        # Calculate the drift and diffusion components derived from Ito's Lemma
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        
        # Calculate daily returns
        daily_returns = np.exp(drift + diffusion)
        
        # Create an empty array for the paths and set the initial price
        paths = np.zeros((self.num_steps + 1, self.num_simulations))
        paths[0] = self.S0
        
        # Generate the full paths using cumulative products of the daily returns
        paths[1:] = self.S0 * np.cumprod(daily_returns, axis=0)
        
        return paths

    def price_european_call(self) -> tuple[float, np.ndarray]:
        """
        Calculates the price of a European Call option.
        
        Returns:
        tuple: (Call option price, Array of simulated paths)
        """
        paths = self.simulate_paths()
        
        # Extract the terminal prices (the last row of the paths array)
        terminal_prices = paths[-1]
        
        # Calculate the payoff of the call option: max(S_T - K, 0)
        payoffs = np.maximum(terminal_prices - self.K, 0)
        
        # Discount the expected payoff back to present value
        call_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        return call_price, paths
