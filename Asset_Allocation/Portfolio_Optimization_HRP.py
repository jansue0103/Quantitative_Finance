import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

def remove_nan_values(series):
    for i in range(len(series)):
        if pd.isna(series.iloc[i]):
            prev_val = series.iloc[i-1] if i > 0 else np.nan
            next_val = series.iloc[i+1] if i < len(series)-1 else np.nan
            avg_val = np.nanmean([prev_val, next_val])
            
            noise = np.random.normal(scale=0.2)
            series.iloc[i] = avg_val + noise
    return series


def hrp(assets: pd.DataFrame, start: str, end: str, linkage_method: str):
    data_raw = yf.download(assets, start=start, end=end)["Adj Close"]

    data_imputed = data_raw.apply(remove_nan_values)

    # Transform percentage returns into log returns
    data_log_returns = np.log(1 + data_imputed.pct_change())[1:]
    #data_pct_change = data_raw.pct_change()[1:]

    # Calculate distance matrix from log return correlation matrix
    distances = pdist(data_log_returns.corr(), metric="euclidean")
    clusters = linkage(y=distances, method=linkage_method, metric="euclidean")
    dend = dendrogram(Z=clusters, labels=data_imputed.columns)
    plt.figure()

    cov_matrix = data_log_returns.cov()

    pca = PCA()
    pca.fit(cov_matrix)
    loadings = pca.components_
    ordered_indices = np.argsort(np.abs(loadings[0]))[::-1]

    quasi_diag_columns = cov_matrix.iloc[:, ordered_indices]
    quasi_diag = quasi_diag_columns.iloc[ordered_indices]

    sns.heatmap(quasi_diag)
    plt.figure()
    sns.heatmap(cov_matrix)
    # Figure out if the clustering from HCA is used in the quasi diagonalization
    plt.show()
    pass 





# distance_matrix = squareform(distances)
    # distance_df = pd.DataFrame(distance_matrix, columns=data_raw.columns, index=data_raw.columns)