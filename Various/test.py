import yfinance as yf
import pandas as pd
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from statsmodels.stats.stattools import jarque_bera

data_set = yf.download("SPY", start="2024-03-06", end="2024-03-07",interval="1m")
data_set.to_csv("RawData.csv", index=True) ####

def to_dollar_bars(data, dollar_metric):
    data = data.drop(["Adj Close"], axis=1)
    data_dollar_bars = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "DollarValue"], index=pd.to_datetime([]))
    data["DollarValue"] = 0.0
    prev_index = None
    for index, row in data.iterrows():
        dollar_value = ((row.Open + row.Close) / 2) * row.Volume
        data.loc[index, "DollarValue"] = dollar_value
    data = data.round(decimals=4)
    data.to_csv("DollarValue.csv", index=True) ###

    for index, row in data.iterrows():
        no_rows = row["DollarValue"]/ dollar_metric
        iterations = int(no_rows) + 1
        for _ in range(iterations):
            data_dollar_bars.loc[len(data_dollar_bars)] = {"Date": index, "Open": row.Open, "High": row.High, "Low": row.Low, "Close": row.Close, "Volume": row.Volume, "DollarValue": dollar_metric}
            data.loc[index,"DollarValue"] = data.loc[index,"DollarValue"] - dollar_metric
        #print(_, row.DollarValue)
        if (prev_index is not None and data.loc[prev_index, "DollarValue"] > 0):
            weighted_prev= data.loc[prev_index, "DollarValue"] / (data.loc[prev_index, "DollarValue"] + data.loc[index, "DollarValue"])
            weighted_current = 1 - weighted_prev 
            weighted_open = row.Open * weighted_current + data.loc[prev_index, "Open"] * weighted_prev
            weighted_high = row.High * weighted_current + data.loc[prev_index, "High"] * weighted_prev
            weighted_low = row.Low * weighted_current + data.loc[prev_index, "Low"] * weighted_prev
            weighted_close = row.Close * weighted_current + data.loc[prev_index, "Close"] * weighted_prev
            weighted_volume = row.Volume * weighted_current + data.loc[prev_index, "Volume"] * weighted_prev
            weighted_index = index if weighted_current > weighted_prev else prev_index

            data_dollar_bars.loc[len(data_dollar_bars)] = {"Date": weighted_index, "Open": weighted_open, "High": weighted_high, "Low": weighted_low, "Close": weighted_close, "Volume": weighted_volume, "DollarValue": dollar_metric}
            data.loc[index, "DollarValue"] = data.loc[index, "DollarValue"] - data.loc[prev_index, "DollarValue"]
            data.loc[prev_index, "DollarValue"] = 0
        prev_index = index
    return data_dollar_bars



dollar_bar_data = to_dollar_bars(data_set, 20000000)
dollar_bar_data.to_csv("Output.csv", index=True) ###
chart = go.Figure(data=[go.Candlestick(x=dollar_bar_data.index, open=dollar_bar_data.Open, high=dollar_bar_data.High, low=dollar_bar_data.Low, close=dollar_bar_data.Close)])
#chart2 = go.Figure(data=[go.Candlestick(x=data.index, open=data.Open, high=data.High, low=data.Low, close=data.Close)])
chart.show()
#print(dollar_bar_data)
print(data_set)

