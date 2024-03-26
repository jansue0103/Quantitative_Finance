import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from misc_functions import impute_nan_values


def hrp(assets: list, start: str, end: str, linkage_method: str) -> list:
    data_raw = yf.download(assets, start=start, end=end)["Adj Close"]

    data_imputed = data_raw.apply(impute_nan_values)

    # Transform percentage returns into log returns
    data_log_returns = pd.DataFrame(np.log(1 + data_imputed.pct_change())[1:])

    corr_matrix = data_log_returns.corr()
    # TODO: Determine method to estmate covariance matrix
    cov_matrix = data_log_returns.cov()  # TODO: Why are the variances so low?
    # Transform correlation matrix into correlation distance matrix
    # Distance formula used in "BUILDING DIVERSIFIED PORTFOLIOS THAT OUTPERFORM OUT-OF-SAMPLE" by Marcos Lopez de Prado (2016)
    distance_matrix = corr_matrix.apply(lambda x: np.sqrt(0.5 * (1 - x)))

    # Calculating condensed pairwise distance matrix
    pairwise_distance = pdist(distance_matrix, metric="euclidean")

    # Creating clusters from distance matrix
    clusters = linkage(y=pairwise_distance, method=linkage_method, metric="euclidean")

    # Plotting dendrogram
    # plt.figure(figsize=(20, 15))
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)  # TODO: Fix xlabel font size
    dendro = dendrogram(Z=clusters, labels=data_log_returns.columns)

    # Get reordered asset indexes
    idx = dendro["ivl"]

    # Create new matrix from return corr matrix with ordered asset indexes
    quasi_diag_matrix = cov_matrix.loc[:, idx]  # TODO: Lopez says both reordered cov matrix and corr matrix????
    quasi_diag_matrix = quasi_diag_matrix.loc[idx, :]

    # Plotting heatmap
    plt.figure(figsize=(20, 15))
    plt.title("Quasi-diagonalized correlation matrix", fontsize=35)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax = sns.heatmap(quasi_diag_matrix)

    plt.xlabel("Assets", fontsize=20)
    plt.ylabel("Assets", fontsize=20)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)

    weights = []

    def getClusterVar(cov, asset):
        return cov.loc[asset, asset]

    def rec_bisection(items: list):
        print(items, "\n")
        mid = (len(items)) // 2
        while len(weights) < len(idx):
            if len(items) > 2:
                rec_bisection(items[:mid])
                rec_bisection(items[mid:])
            else:
                var1 = quasi_diag_matrix.loc[items[0], items[0]]
                var2 = quasi_diag_matrix.loc[items[1], items[1]]
                weights.append(var1)
                weights.append(var2)
                break
        pass

    asset_clusters = []
    def split_list(lst):
        #print(lst)
        if len(lst) <= 2:
            asset_clusters.append(lst)
            return lst
        mid = len(lst) // 2
        left_half = split_list(lst[:mid])
        right_half = split_list(lst[mid:])
        #TODO: Review if the assets are just split into pairs or into pairs based on the HCA clusters
    split_list(idx)
    print(asset_clusters)
    #rec_bisection(idx)
    #print(weights)
    plt.show()
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
