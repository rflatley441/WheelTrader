import numpy as np
import yfinance as yf
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from multiprocess import Pool



def gbm(S0, mu, sigma, T, num_simulations=500):
    steps = T  # Steps should match the number of days to expiration
    dt = 1  # Daily time step
    Z = np.random.standard_normal((steps, num_simulations))
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    price_paths = np.exp(drift + diffusion).cumprod(axis=0)  
    return S0 * price_paths  # Remove extra row


def objective(params, real_prices, s0):
    """Objective function for optimization."""
    mu, sigma = params
    gbm_prices = gbm(s0, mu, sigma, T=len(real_prices))

    if len(gbm_prices) != len(real_prices):
        raise ValueError("Mismatch in GBM output size and real_prices length.")

    return mean_squared_error(real_prices, gbm_prices) 


def process_bin(i, real_prices, bin_length, bounds):
    """Process a single bin to optimize μ and σ."""
    bin_prices = real_prices[i * bin_length : (i + 1) * bin_length]
    s0 = bin_prices[0]
    result = differential_evolution(objective, bounds, args=(bin_prices, s0))
    return (result.x[0], result.x[1], result.fun)


def optimize_gbm(symbol: str, training_period: str, bin_length: int):
    """
    Optimize μ and σ over multiple time bins using multiprocessing.
    """
    stock_data = yf.download(symbol, period=training_period, interval="1d")
    real_prices = stock_data["Close"].dropna().values

    num_bins = len(real_prices) // bin_length
    weights = np.linspace(1, 2, num_bins)  
    bounds = [(-0.3, 0.3), (0.001, 0.35)]

    tasks = [(i, real_prices, bin_length, bounds) for i in range(num_bins)]

    with Pool() as pool:
        results = pool.starmap(process_bin, tasks)

    mu_values, sigma_values, mses = zip(*results) if results else ([], [], [])

    weight_sum = np.sum(weights)
    avg_mu = np.sum(np.array(mu_values) * weights) / weight_sum
    avg_sigma = np.sum(np.array(sigma_values) * weights) / weight_sum

    return avg_mu, avg_sigma
