import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


def hrp(tickers: list, start: str, end: str, linkage_method: str) -> list:
    data = yf.download(tickers, start=start, end=end, interval="1d")["Adj Close"]
    data.dropna(inplace=True)

    # Transform percentage returns into log returns
    data_log_returns = pd.DataFrame(np.log(1 + data.pct_change())[1:])

    corr_matrix = data_log_returns.corr()

    # Transform correlation matrix into correlation distance matrix,Distance formula used Lopez de Prado (2016)
    distance_matrix = corr_matrix.apply(lambda x: np.sqrt(0.5 * (1 - x)))

    # Calculating condensed pairwise distance matrix
    pairwise_distance = pdist(distance_matrix, metric="euclidean")

    # Create the covariance matrix based on the new distance matrix
    cov_matrix = distance_matrix.cov()

    # Creating clusters from distance matrix
    clusters = linkage(y=pairwise_distance, method=linkage_method, metric="euclidean")

    # Plotting dendrogram
    plt.figure(figsize=(20, 15))
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    dendro = dendrogram(Z=clusters, labels=data_log_returns.columns)

    # Get reordered asset indexes
    idx = dendro["ivl"]

    # Create new matrix from return corr matrix with ordered asset indexes
    quasi_diag_matrix = cov_matrix.loc[:, idx]
    quasi_diag_matrix = quasi_diag_matrix.loc[idx, :]

    # Plotting heatmap
    plt.figure(figsize=(20, 15))
    plt.title("Quasi-diagonalized covariance matrix", fontsize=35)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax = sns.heatmap(quasi_diag_matrix)

    plt.xlabel("Assets", fontsize=20)
    plt.ylabel("Assets", fontsize=20)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)

    def get_cluster_var(assets: list):
        var_sum = 0.0
        for asset in assets:
            var_sum += quasi_diag_matrix.loc[asset[0], asset[0]]
        return var_sum

    weights = [1.0 for _ in range(len(idx))]
    asset_weights = [[x, y] for x, y in zip(idx, weights)]

    # TODO: Implement 2. approach that bisects clusters based on the dendrogram instead of splitting in half
    def rec_bisection(assets: list):
        mid = len(assets) // 2

        cluster1 = assets[:mid]
        cluster2 = assets[mid:]

        cluster1var = get_cluster_var(cluster1)
        cluster2var = get_cluster_var(cluster2)
        alpha1 = (1 - cluster1var / (cluster1var + cluster2var))
        alpha2 = 1 - alpha1
        if alpha1 > 0:
            for i in range(mid):
                assets[i][1] = assets[i][1] * alpha1
        if alpha2 > 0:
            for j in range(mid, len(assets)):
                assets[j][1] = assets[j][1] * alpha2

        if len(assets) <= 1:
            return
        rec_bisection(cluster1)
        rec_bisection(cluster2)
        return

    rec_bisection(asset_weights) #TODO: Evaluate and confirm correct asset weights
    summation = 0.0
    for weight in asset_weights:
        summation += weight[1]
    print(summation)
    plt.show()

    return asset_weights
