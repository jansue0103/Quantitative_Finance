import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count  

# TODO: Multiprocessing of Monte Carlo Simulation

def portfolio_simulation(num_simulations: int, r_f: float, assets: list, start: str, end: str) -> list:
    start_t = time.perf_counter()
    data = yf.download(assets, start=start, end=end, interval="1d")["Adj Close"]
    log_return = np.log(1 + data.pct_change())[1:]

    all_weights = np.zeros((num_simulations, len(assets)))
    all_returns = np.zeros(num_simulations)
    all_vol = np.zeros(num_simulations)
    all_sharpe = np.zeros(num_simulations)

    for run in range(num_simulations):
        weights = np.random.random(len(assets))
        weights = weights / np.sum(weights)

        all_weights[run, :] = weights

        all_returns[run] = np.sum((log_return.mean() * weights) * 252)

        all_vol[run] = np.sqrt(np.dot(weights.T, np.dot(np.cov(log_return, rowvar=False) * 252, weights)))
                
        all_sharpe[run] = (all_returns[run] - r_f) / all_vol[run]
        all_weights = all_weights.round(decimals=4)
        sim_data = [all_returns, all_vol, all_sharpe, all_weights]
        simulation_data = pd.DataFrame(data=sim_data)
        simulation_data = simulation_data.T
        simulation_data.columns= ["Return", "Volatility", "Sharpe Ratio", "Weights"]
        simulation_data = simulation_data.infer_objects()     
    min_vol = simulation_data.loc[simulation_data["Volatility"].idxmin()]
    max_sharpe = simulation_data.loc[simulation_data["Sharpe Ratio"].idxmax()]
    
    plt.scatter(x=simulation_data["Volatility"], y=simulation_data["Return"], c=simulation_data["Sharpe Ratio"])
    plt.xlabel("Standard deviation")
    plt.ylabel("Return")
    plt.title("Risks vs. Returns")
    plt.colorbar(label="Sharpe Ratio")

    print("="*80)
    print("MAX SHARPE RATIO PORTFOLIO MONTE CARLO:")
    print("-"*80)
    print(f"Return: {np.round(max_sharpe["Return"] * 100, decimals=2)}%")
    print(f"Volatility:, {np.round(max_sharpe["Volatility"] * 100, decimals=2)}%")
    print("Sharpe Ratio:", np.round(max_sharpe["Sharpe Ratio"], decimals=2))
    print("Weights:" , tuple(zip(assets, np.round(max_sharpe["Weights"], decimals=3))))
    print("-"*80)
    print(" ")
    print("="*80)
    print("MIN VOLATILITY PORTFOLIO MONTE CARLO:")
    print(f"Return: {np.round(min_vol["Return"] * 100, decimals=2)}%")
    print(f"Volatility:, {np.round(min_vol["Volatility"] * 100, decimals=2)}%")
    print("Sharpe Ratio:", np.round(min_vol["Sharpe Ratio"], decimals=2))
    print("Weights:" , tuple(zip(assets, np.round(min_vol["Weights"], decimals=3))))
    print("-"*80)

    end_t = time.perf_counter()
    total_t = end_t-start_t
    print(f"Process took {total_t: .2f}s") 

    plt.show()
    return [max_sharpe, min_vol]