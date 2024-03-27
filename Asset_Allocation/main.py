from Monte_Carlo_Simulation import portfolio_simulation
from Portfolio_Optimization_Markowitz import portfolio_optimization
from Portfolio_Optimization_HRP import hrp
from misc_functions import get_stock_tickers
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':

    assets = ["AAPL", "SPY", "HD", "QQQ", "^FTSE", "GC=F", "CL=F", "^N225", "BTC-USD", "TSLA", "PEP", "AMD", "SHY", "TLT", "KO", "NVDA", "ZS=F"]
    assets2 = ["EEM", "EWG", "TIP", "EWJ", "EFA", "IDF", "EWQ", "EWU", "XLU", "XLE", "XLF", "IEF", "XLK", "AAPL", "EPP", "FXI", "VGK", "VPL", "SPY", "TLT", "BND", "QQQ", "DJIA"]
    assets3 = ["AAPL", "AMD", "TSLA", "QQQ", "^TNX", "^FVX", "^TYX", "TLT", "GC=F", "ZS=F", "^RUT", "^FTSE", "^N225", "BTC-USD"]
    assets_equities = ["AAPL", "AMD", "TSLA", "QQQ", "SPY", "HD", "KO", "OXY", "PEP", "NVDA", "ADBE", "UL"]
    start = "2015-03-16"
    end = "2024-03-16"
    
    data_hrp = hrp(assets_equities, start, end, "complete")
    data_opt = portfolio_optimization(assets=assets_equities, start=start, end=end, risk_free_rate=0.0)

    assets_hrp, weights_hrp = list(zip(*data_hrp))
    plt.bar(assets_hrp, weights_hrp)
    plt.title("Weights HRP")
    plt.figure()
    plt.bar(data_opt[2], data_opt[3])
    plt.title("Weights Mean-variance")
    plt.show()

# Read about quadratic and other optimization methods