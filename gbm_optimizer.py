import numpy as np
import yfinance as yf
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from multiprocess import Pool


def gbm(s0, mu, sigma, deltaT, dt):
    """
    Models stock price using the Geometric Brownian Motion (GBM) model.

    Arguments:
        s0: Initial stock price
        mu: Drift of the stock
        sigma: Volatility of the stock
        deltaT: Total time for the simulation
        dt: Time step

    Returns:
        s: Simulated stock prices over time
    """
    n_steps = int(deltaT / dt)
    time_vector = np.linspace(0, deltaT, n_steps)
    
    # Wiener process: random increments
    random_increments = np.random.normal(0, 1, size=n_steps) * np.sqrt(dt)
    weiner_process = np.cumsum(random_increments)
    
    # Stock price evolution according to GBM formula
    stock_var = (mu - (sigma ** 2 / 2)) * time_vector
    s = s0 * np.exp(stock_var + sigma * weiner_process)
    
    return s

def simulate_gbm(s0, mu, sigma, days_to_expiration, n_simulations):
    dt = 1
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    random_shocks = np.random.normal(0, 1, (days_to_expiration, n_simulations))
    prices = s0 * np.exp(np.cumsum(drift + diffusion * random_shocks, axis=0))
    return prices


def objective(params, real_prices, s0):
    """Objective function for optimization to minimize MSE."""
    mu, sigma = params
    gbm_prices = gbm(s0, mu, sigma, deltaT=len(real_prices), dt=1)

    if len(gbm_prices) != len(real_prices):
        raise ValueError("Mismatch in GBM output size and real_prices length.")

    return mean_squared_error(real_prices, gbm_prices)

def process_bin(i, real_prices, bin_length, bounds):
    """Process a single bin to optimize μ and σ."""
    bin_prices = real_prices[i * bin_length : (i + 1) * bin_length]
    s0 = bin_prices[0]
    result = differential_evolution(objective, bounds, args=(bin_prices, s0), disp=False)
    return result.x[0], result.x[1], result.fun

def optimize_gbm(symbol: str, training_period: str, bin_length: int):
    """
    Optimize μ and σ over multiple time bins using differential evolution.
    """
    # Download stock data from Yahoo Finance
    stock_data = yf.download(symbol, period=training_period, interval="1d", progress=False)
    real_prices = stock_data["Close"].dropna().values

    num_bins = len(real_prices) // bin_length
    bounds = [(-0.3, 0.3), (0.001, 0.35)]  # Bounds for mu and sigma

    # Parallelize the optimization process across bins
    from multiprocessing import Pool
    tasks = [(i, real_prices, bin_length, bounds) for i in range(num_bins)]

    with Pool() as pool:
        results = pool.starmap(process_bin, tasks)

    mu_values, sigma_values, mses = zip(*results) if results else ([], [], [])

    # Use a weighted average of mu and sigma based on bin index or another method
    avg_mu = np.mean(mu_values)
    avg_sigma = np.mean(sigma_values)

    return avg_mu, avg_sigma

