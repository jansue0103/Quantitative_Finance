import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from misc_functions import impute_nan_values
from itertools import groupby

def hrp(assets: pd.DataFrame, start: str, end: str, linkage_method: str): #TODO: define output type
    data_raw = yf.download(assets, start=start, end=end)["Adj Close"]

    data_imputed = data_raw.apply(impute_nan_values)
    no_assets = len(assets)
    # Transform percentage returns into log returns
    data_log_returns = pd.DataFrame(np.log(1 + data_imputed.pct_change())[1:])

    corr_matrix = data_log_returns.corr()
    cov_matrix = data_log_returns.cov()
    # Transform correlation matrix into correlation distance matrix
    # Distance formula used in "BUILDING DIVERSIFIED PORTFOLIOS THAT OUTPERFORM OUT-OF-SAMPLE" by Marcos Lopez de Prado (2016)
    distance_matrix = corr_matrix.apply(lambda x: np.sqrt(0.5 * (1 - x)))

    # Calculate condensed pairwise distance matrix
    pairwise_distance = pdist(distance_matrix, metric="euclidean")

    clusters = linkage(y=pairwise_distance, method=linkage_method, metric="euclidean")
    dend = dendrogram(Z=clusters, labels=data_log_returns.columns)
    plt.figure()

    # Creates a tuple with asset ticker, order of assets in dendogram, and cluster family
    dendogram_tuple = list(zip(dend["ivl"], dend["leaves"], dend["leaves_color_list"]))
    quasi_idx_height = []

    # Adds the merged hight to each asset tuple
    for row in clusters:
        if(row[0] <= no_assets or row[1] <= no_assets):
            for item in dendogram_tuple:
                if(item[1] == row[0]):
                    quasi_idx_height.append(tuple(list(item) + [row[2]]))
                if(item[1] == row[1]):
                    quasi_idx_height.append(tuple(list(item) + [row[2]]))
  
    quasi_diag = []
    temp = sorted(quasi_idx_height, key=lambda x: x[2], reverse=True)###
    prev_cluster = None
    temp2 = []
    for item in temp:
        if (prev_cluster is not None and prev_cluster != item[2]):
            temp2 = sorted(temp2, key=lambda x: x[3], reverse=False)
            quasi_diag.append(temp2)
            temp2 = []
        temp2.append(item)
        prev_cluster = item[2]

    temp2 = sorted(temp2, key=lambda x: x[3], reverse=False)
    quasi_diag.append(temp2)

    result = [t[0] for sub in quasi_diag for t in sub]
    #result = result[::-1]

    quasi_diag_columns = corr_matrix.loc[:, result]
    quasi_diag_cov = quasi_diag_columns.loc[result]

    print(quasi_diag_cov)
    sns.heatmap(quasi_diag_cov)
    #plt.figure()
    #sns.heatmap(cov_matrix)
    
    "TODO: Some, but not all variances are placed correctly along the diagonal, reversed=True vs. False. Algorithm needs to be adapted"
    plt.show()
    pass