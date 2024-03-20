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


def hrp(assets: list, start: str, end: str, linkage_method: str) -> list:  # TODO: define output type
    data_raw = yf.download(assets, start=start, end=end)["Adj Close"]

    data_imputed = data_raw.apply(impute_nan_values)

    # Transform percentage returns into log returns
    data_log_returns = pd.DataFrame(np.log(1 + data_imputed.pct_change())[1:])

    corr_matrix = data_log_returns.corr()
    cov_matrix = data_log_returns.cov()
    # Transform correlation matrix into correlation distance matrix
    # Distance formula used in "BUILDING DIVERSIFIED PORTFOLIOS THAT OUTPERFORM OUT-OF-SAMPLE" by Marcos Lopez de Prado (2016)
    distance_matrix = corr_matrix.apply(lambda x: np.sqrt(0.5 * (1 - x)))

    # Calculate condensed pairwise distance matrix
    pairwise_distance = pdist(distance_matrix, metric="euclidean")
    full_distance = squareform(pairwise_distance)

    # TODO: Check with pairwise and full distance differences
    # TODO: Test with new asset set with multiple goups very similar assets with a group but dissimilar groups, eg. 5 bonds, 5 stocks, 5 commidites, 5 foreign etfs

    # Creating clusters from distance matrix
    clusters = linkage(y=full_distance, method=linkage_method, metric="euclidean")

    # Creating dendrogram from clusters
    dendro = dendrogram(Z=clusters, labels=data_log_returns.columns)
    plt.figure()

    # Get reordered asset indexes
    idx = dendro["ivl"]

    # Create new matrix from return corr matrix with ordered asset indexes
    quasi_diag_matrix = corr_matrix.loc[:, idx]
    quasi_diag_matrix = quasi_diag_matrix.loc[idx, :]

    sns.heatmap(quasi_diag_matrix)
    plt.figure()
    plt.show()
    # TODO: ffs check if corr or cov should be used
    return idx

    # #Creates a tuple with asset ticker, order of assets in dendogram, and cluster family
    # dendogram_tuple = list(zip(dend["ivl"], dend["leaves"], dend["leaves_color_list"]))
    # quasi_idx_height = []
    #
    # # Adds the merged height to each asset tuple
    # for row in clusters:
    #     if row[0] <= no_assets or row[1] <= no_assets:
    #         for item in dendogram_tuple:
    #             if item[1] == row[0]:
    #                 quasi_idx_height.append(tuple(list(item) + [row[2]]))
    #             if item[1] == row[1]:
    #                 quasi_idx_height.append(tuple(list(item) + [row[2]]))
    #
    # quasi_diag = []
    # temp = sorted(quasi_idx_height, key=lambda x: x[2], reverse=True)  ###
    # prev_cluster = None
    # temp2 = []
    # for item in temp:
    #     if prev_cluster is not None and prev_cluster != item[2]:
    #         temp2 = sorted(temp2, key=lambda x: x[3], reverse=False)
    #         quasi_diag.append(temp2)
    #         temp2 = []
    #     temp2.append(item)
    #     prev_cluster = item[2]
    #
    # temp2 = sorted(temp2, key=lambda x: x[3], reverse=False)
    # quasi_diag.append(temp2)
    #
    # result = [t[0] for sub in quasi_diag for t in sub]
    #
    # quasi_diag_columns = corr_matrix.loc[:, result]
    # quasi_diag_cov = quasi_diag_columns.loc[result]
