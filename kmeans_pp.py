"""
calculating the kmeans++ algorithm
"""
import numpy as np
from mykmeanssp import kmeans


def k_means_pp(observations, K, N, d, max_iter):
    """
    the kmeans++ algorithm as was presented in assignment 2
    :param observations: the points to classify
    :param K: number of centers
    :param N: number of points
    :param d: dimension of points
    :param max_iter: maximum iterations for the kmeans algorithm
    :return:
    1. the indexes for each observation as obtained from the kmeans algorithm, aka, indexes_empirical[i] is the cluster of obs[i] in the algorithm
    2. list of clusters as obtained from the kmeans algorithm, aka, clusters_empirical[i] is all j such that obs[j] is in cluster i
    """
    np.random.seed(0)

    centroids = np.zeros((K, d), np.float_)
    r = np.random.choice(N, 1)
    centroids[0] = observations[r]
    D = np.full(N, float('inf'))
    p = np.zeros(N)
    indices = np.zeros(K, np.int_)  # list and not np.array
    indices[0] = r  # r[0]
    for j in range(1, K):
        for i in range(j):
            dist = np.linalg.norm(observations - centroids[i], axis=1) ** 2
            D = np.minimum(D, dist)
        s = sum(D)
        p = D / s
        w = np.random.choice(N, 1, p=p)
        centroids[j] = observations[w]
        indices[j] = w
    obs = observations.tolist()
    cen = centroids.tolist()
    indexes = kmeans(obs, cen, K, N, d, max_iter)
    if indexes is None:
        return None, None
    clusters = [[] for _ in range(K)]
    for i in range(N):
        clusters[indexes[i]].append(i)
    return indexes, clusters
