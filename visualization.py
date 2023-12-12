"""
function for plotting the points by each algorithm, and exporting the plot to a pdf file
"""
import matplotlib.pyplot as plt
from eigen import jaccard_measure


def vis(points, indexes_real, indexes_nsc, indexes_kmeans, clusters_nsc, clusters_kmeans, n, real_k, found_k):
    """
    visualizing the result from the algorithms
    :param points: points from make_blobs
    :param indexes_real: the real cluster indexes for each observation, aka, indexes_real[i] is the cluster of obs[i]
    :param indexes_nsc: the indexes for each observation as obtained from the nsc algorithm, aka, indexes_empirical[i] is the cluster of obs[i] from nsc
    :param indexes_kmeans: the indexes for each observation as obtained from the kmeans algorithm, aka, indexes_empirical[i] is the cluster of obs[i] from kmeans
    :param clusters_nsc: list of clusters as obtained from the nsc algorithm, aka, clusters_empirical[i] is all j such that obs[j] is in cluster i
    :param clusters_kmeans: list of clusters as obtained from the kmeans algorithm, aka, clusters_empirical[i] is all j such that obs[j] is in cluster i
    :param n: number of points
    :param real_k: the k from which the points were generated
    :param found_k: the k that we found from the algorithm, if random was True
    :return: None
    """
    jaccard_nsc = jaccard_measure(indexes_real, indexes_nsc, clusters_nsc, max(real_k, found_k))
    jaccard_kmeans = jaccard_measure(indexes_real, indexes_kmeans, clusters_kmeans, max(real_k, found_k))
    dim = points.shape[1]
    X = points[:, 0]
    Y = points[:, 1]
    fig = plt.figure()

    msg = "Data was generated from the values:\nn = {} , k = {}\nThe k that was used for both algorithms was {}\nThe " \
          "Jaccard measure for Spectral Clustering: {}\nThe Jaccard measure for K-means: {}".format(
            n, real_k, found_k, jaccard_nsc, jaccard_kmeans)
    if dim == 3:
        Z = points[:, 2]
        ps = (X, Y, Z)
        ax_nsc = fig.add_subplot(2, 2, 1, projection='3d')
        ax_kmeans = fig.add_subplot(2, 2, 2, projection='3d')
    else:
        ps = (X, Y)
        ax_nsc = fig.add_subplot(2, 2, 1)
        ax_kmeans = fig.add_subplot(2, 2, 2)
    ax_text = fig.add_subplot(2, 2, (3, 4))
    ax_text.axis("off")
    ax_text.text(0.5, 0.3, msg, horizontalalignment='center', size='x-large')
    ax_nsc.set_title("nsc")
    ax_nsc.scatter(*ps, c=indexes_nsc, cmap='gist_rainbow', alpha=1)
    ax_kmeans.set_title("kmeans")
    ax_kmeans.scatter(*ps, c=indexes_kmeans, cmap='gist_rainbow', alpha=1)
    fig.savefig("clusters.pdf", bbox_inches='tight')
