import numpy as np
import yfinance as yf
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from multiprocess import Pool



def gbm(s0, mu, sigma, deltaT, dt, simulations):
    """
    Models multiple stock price simulations using the Wiener process.
    
    Arguments:
        s0: Initial stock price
        mu: Drift of the stock
        sigma: Volatility of the stock
        deltaT: Time period for which future prices are computed
        dt: Granularity of the time-period
        simulations: Number of simulations to run
    
    Returns:
        s: A 2D array with simulated stock prices for each simulation (rows) and each time step (columns)
    """
    n_step = int(deltaT / dt)
    time_vector = np.linspace(0, deltaT, num=n_step)
    
    # Wiener process: cumulative sum of random normal increments
    random_increments = np.random.normal(0, 1, size=(simulations, n_step)) * np.sqrt(dt)
    weiner_process = np.cumsum(random_increments, axis=1)
    
    # Stock price simulation
    stock_var = (mu - (sigma**2 / 2)) * time_vector
    s = s0 * np.exp(stock_var + sigma * weiner_process)
    
    return s


def objective(params, real_prices, s0):
    """Objective function for optimization."""
    mu, sigma = params
    gbm_prices = gbm(s0, mu, sigma, deltaT=len(real_prices), dt=1, simulations=500)

    if len(gbm_prices) != len(real_prices):
        raise ValueError("Mismatch in GBM output size and real_prices length.")

    return mean_squared_error(real_prices, gbm_prices) 


def process_bin(i, real_prices, bin_length, bounds, simulation_attempts):
    """Process a single bin to optimize μ and σ."""
    bin_prices = real_prices[i * bin_length : (i + 1) * bin_length]
    s0 = bin_prices[0]
    result = differential_evolution(objective, bounds, args=(bin_prices, s0, simulation_attempts))
    return (result.x[0], result.x[1], result.fun)

def optimize_gbm(symbol: str, training_period: str, bin_length: int, simulation_attempts: int):
    """
    Optimize μ and σ over multiple time bins using multiprocessing.
    """
    stock_data = yf.download(symbol, period=training_period, interval="1d")
    real_prices = stock_data["Close"].dropna().values

    num_bins = len(real_prices) // bin_length
    weights = np.linspace(1, 2, num_bins)  
    bounds = [(-0.3, 0.3), (0.001, 0.35)]

    tasks = [(i, real_prices, bin_length, bounds, simulation_attempts) for i in range(num_bins)]

    with Pool() as pool:
        results = pool.starmap(process_bin, tasks)

    mu_values, sigma_values, mses = zip(*results) if results else ([], [], [])

    weight_sum = np.sum(weights)
    avg_mu = np.sum(np.array(mu_values) * weights) / weight_sum
    avg_sigma = np.sum(np.array(sigma_values) * weights) / weight_sum

    return avg_mu, avg_sigma
