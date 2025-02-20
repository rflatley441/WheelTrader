import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.optimize import differential_evolution

# Define parameters
S0 = 156.91  # initial stock price
r = 0.0023  # risk-free rate
T = 0.528  # time to maturity
kappa = 1.5  # mean reversion speed
theta = 0.04  # long-run variance
sigma = 0.2994  # implied volatility
rho = -0.3  # correlation between the stock price and volatility
V0 = sigma**2  # initial variance
K = 165.0  # strike price
N = 1000  # number of simulation paths
M = 500  # number of time steps
dt = T / M  # time step size

# Define the function for the Heston model
def heston(s0, v0, kappa, theta, sigma, rho, r, T, M, N):
    # Initialize arrays
    S = np.zeros((M + 1, N))
    V = np.zeros((M + 1, N))
    S[0, :] = s0
    V[0, :] = v0

    # Generate correlated Brownian motions
    z1 = np.random.normal(size=(M, N))
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=(M, N))

    # Generate stock price and volatility paths
    for t in range(1, M + 1):
        # Calculate the drift and volatility
        drift = r * S[t - 1, :]
        vol = np.sqrt(V[t - 1, :])

        # Update the stock price and volatility
        S[t, :] = S[t - 1, :] * np.exp((drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z1[t - 1, :])
        V[t, :] = np.maximum(
            0.0, V[t - 1, :] + kappa * (theta - V[t - 1, :]) * dt + sigma * np.sqrt(V[t - 1, :]) * np.sqrt(dt) * z2[t - 1, :]
        )

    # Calculate the option payoff at maturity
    payoff = np.maximum(S[-1, :] - K, 0)

    # Calculate the option value using the LSM method
    for t in range(M - 1, 0, -1):
        # Calculate the cash flows and exercise values
        cash_flows = np.zeros(N)
        exercise_values = np.zeros(N)
        for i in range(N):
            if payoff[i] > 0:
                cash_flows[i] = payoff[i] * np.exp(-r * (T - t) / (M - t))
                exercise_values[i] = payoff[i]

        # Calculate the continuation values
        x = np.log(S[t, :] / K)
        y = np.sqrt(V[t, :])
        X = np.column_stack((np.ones(N), x, x ** 2, x ** 3, x ** 4))
        X_inv = np.linalg.inv(np.dot(X.T, X))
        beta = np.dot(X_inv, np.dot(X.T, cash_flows))
        continuation_values = np.dot(X, beta)

        # Determine the optimal exercise policy
        early_exercise = exercise_values > continuation_values
        payoff[early_exercise] = exercise_values[early_exercise]

    # Calculate the option value using Monte Carlo simulation
    discount_factor = np.exp(-r * T)
    option_value = discount_factor * np.mean(payoff)

    return option_value, S

# Define the objective function for differential evolution
def objective_function(params, real_prices):
    kappa, theta, sigma, rho = params
    option_value, simulated_prices = heston(S0, V0, kappa, theta, sigma, rho, r, T, M, N)
    # Calculate the mean squared error between simulated and real prices
    mse = np.mean((simulated_prices[-1, :] - real_prices) ** 2)
    return mse

# Example real prices (replace with actual real prices)
real_prices = np.random.normal(loc=S0, scale=10, size=N)

# Define bounds for the parameters
bounds = [(0.1, 2.0), (0.01, 0.1), (0.1, 0.5), (-0.9, 0.0)]

# Use differential evolution to optimize the parameters
result = differential_evolution(objective_function, bounds, args=(real_prices,))

# Get the optimized parameters
optimized_params = result.x
optimized_kappa, optimized_theta, optimized_sigma, optimized_rho = optimized_params

# Calculate the option value with optimized parameters
optimized_option_value, optimized_simulated_prices = heston(S0, V0, optimized_kappa, optimized_theta, optimized_sigma, optimized_rho, r, T, M, N)

print("Optimized Parameters:")
print("kappa = ", optimized_kappa)
print("theta = ", optimized_theta)
print("sigma = ", optimized_sigma)
print("rho = ", optimized_rho)
print("Optimized Option Value: ", optimized_option_value)

# Plot the optimized simulated paths
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Optimized Heston Model Paths")
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.set_zlabel("Volatility")
for i in range(10):
    ax.plot(np.arange(M + 1), optimized_simulated_prices[:, i], V[:, i])
plt.show()