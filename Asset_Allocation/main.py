from Monte_Carlo_Simulation import portfolio_simulation
from Portfolio_Optimization_Markowitz import portfolio_optimization
from Portfolio_Optimization_HRP import hrp
import yfinance as yf

if __name__ == '__main__':

    assets = ["AAPL", "SPY", "HD", "QQQ", "^FTSE", "GC=F", "CL=F", "^N225", "BTC-USD", "TSLA", "^TNX", "^TYX", "^FVX", "TLT"]
    start="2021-03-16"
    end="2024-03-16"
    
    hrp(assets, start, end, "complete")
    #result_mc = portfolio_simulation(300, 0.0, assets, start=start, end=end)
    #result_opt = portfolio_optimization(assets, start, end, risk_free_rate=0.0)

# Read about quadratic and other optimization methods
# Implement other portfolio construction methods such as risk parity and hierarchical risk parity by de Lopez