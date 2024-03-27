import yfinance as yf
import numpy as np
import scipy.optimize as optimize


def portfolio_optimization(assets: list, start=str, end=str, risk_free_rate=float) -> list:
    data = yf.download(assets, start=start, end=end, keepna=False)["Adj Close"]
    data.dropna(inplace=True)
    log_return = np.log(1 + data.pct_change())[1:]

    def minimize_sharpe(weights: list) -> np.array:
        return -portfolio_stats(weights, risk_free_rate)[2]

    def minimize_volatility(weights: list) -> np.array:
        return portfolio_stats(weights, risk_free_rate)[1]

    def portfolio_stats(weights, r_f):
        weights = np.array(weights)
        port_return = np.sum(log_return.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(np.cov(log_return, rowvar=False) * 252, weights)))
        sharpe = (port_return - r_f) / vol

        return [port_return, vol, sharpe]

    # Equal weight initialization
    initializer_equal = len(assets) * [1.0 / len(assets)]

    # Each weight is between 0% and 100%
    bounds = tuple((0, 1) for _ in range(len(assets)))

    # Equality constraint, all weights sum up to 100%
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})

    opt_sharpe = optimize.minimize(minimize_sharpe, x0=np.array(initializer_equal), method="SLSQP", bounds=bounds,
                                   constraints=constraints)
    opt_vol = optimize.minimize(minimize_volatility, x0=np.array(initializer_equal), method="SLSQP", bounds=bounds,
                                constraints=constraints)

    result_sharpe = portfolio_stats(opt_sharpe.x, risk_free_rate)
    result_vol = portfolio_stats(opt_vol.x, risk_free_rate)

    print("=" * 80)
    print("MAX SHARPE RATIO PORTFOLIO MARKOWITZ:")
    print("-" * 80)
    print(f"Return: {np.round(result_sharpe[0] * 100, decimals=2)}%")
    print(f"Volatility:, {np.round(result_sharpe[1] * 100, decimals=2)}%")
    print("Sharpe Ratio:", np.round(result_sharpe[2], decimals=2))
    print("Weights:", list(zip(assets, np.round(opt_sharpe.x, decimals=3))))
    print("-" * 80)
    print(" ")
    print("=" * 80)
    print("MIN VOLATILITY PORTFOLIO MARKOWITZ:")
    print(f"Return: {np.round(result_vol[0] * 100, decimals=2)}%")
    print(f"Volatility:, {np.round(result_vol[1] * 100, decimals=2)}%")
    print("Sharpe Ratio:", np.round(result_vol[2], decimals=2))
    print("Weights:", list(zip(assets, np.round(opt_vol.x, decimals=3))))
    print("-" * 80)

    return [result_sharpe, result_vol, assets, np.round(opt_sharpe.x, decimals=3), np.round(opt_vol.x, decimals=3)]
