from Monte_Carlo_Simulation import portfolio_simulation
from Portfolio_Optimization_Markowitz import portfolio_optimization
from Portfolio_Optimization_HRP import hrp
from misc_functions import get_stock_tickers
import yfinance as yf
import random
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
if __name__ == '__main__':

    assets = ["AAPL", "SPY", "HD", "QQQ", "^FTSE", "GC=F", "CL=F", "^N225", "BTC-USD", "TSLA", "PEP", "AMD", "SHY", "TLT", "KO", "NVDA", "ABNB", "ZS=F"]
    assets2 = ["EEM", "EWG", "TIP", "EWJ", "EFA", "IDF", "EWQ", "EWU", "XLU", "XLE", "XLF", "IEF", "XLK", "AAPL", "EPP", "FXI", "VGK", "VPL", "SPY", "TLT", "BND", "QQQ", "DJIA"]
    assets3 = ["AAPL", "AMD", "TSLA", "QQQ", "^TNX", "^FVX", "^TYX", "TLT", "GC=F", "ZS=F", "^RUT", "^FTSE", "^N225", "BTC-USD"]
    start = "2015-03-16"
    end = "2024-03-16"
    
    data = hrp(assets2, start, end, "complete")
    #result_mc = portfolio_simulation(300, 0.0, assets, start=start, end=end)
    #result_opt = portfolio_optimization(assets, start, end, risk_free_rate=0.0)
    
 
    
# Read about quadratic and other optimization methods
# Implement other portfolio construction methods such as risk parity and hierarchical risk parity by de Lopez