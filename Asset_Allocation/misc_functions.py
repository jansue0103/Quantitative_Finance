import pandas as pd


def impute_nan_values(series: pd.DataFrame) -> pd.DataFrame:
    data = series.ffill()
    data = series.bfill()
    data = series.fillna(series.mean())
    return data


def get_stock_tickers() -> list:
    url_spy = 'https://en.m.wikipedia.org/wiki/Nasdaq-100'
    url_qqq = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
    spy = pd.read_html(url_spy, attrs={'id': "constituents"}, index_col='Ticker')[0]
    qqq = pd.read_html(url_qqq, attrs={'id': 'constituents'}, index_col='Symbol')[0]
    spy = spy.index.to_list()
    qqq = qqq.index.to_list()
    stocks = spy + qqq
    stocks = list(set(stocks))
    stocks.remove("BRK.B")
    stocks.remove("BF.B")
    stocks.sort()
    return stocks
