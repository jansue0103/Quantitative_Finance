import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from misc_functions import impute_nan_values


def hrp(assets: pd.DataFrame, start: str, end: str, linkage_method: str):
    data_raw = yf.download(assets, start=start, end=end)["Adj Close"]

    data_imputed = data_raw.apply(impute_nan_values)

    # Transform percentage returns into log returns
    data_log_returns = np.log(1 + data_imputed.pct_change())[1:]
    #data_pct_change = data_raw.pct_change()[1:]

    # Calculate distance matrix from log return correlation matrix
    distances = pdist(data_log_returns.corr(), metric="euclidean")
    clusters = linkage(y=distances, method=linkage_method, metric="euclidean")
    #dend = dendrogram(Z=clusters, labels=data_imputed.columns)
    #plt.figure()

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