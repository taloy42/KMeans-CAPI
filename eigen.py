"""
all the algorithms regarding the QR iteration, the eigen gap heuristic and the jaccard measure
"""

import numpy as np


# def gram_schmidt(a):
#     n = a.shape[0]
#     # u = a.copy()
#     u = a
#     q = np.zeros((n, n), dtype=np.float64)
#     r = np.zeros((n, n), dtype=np.float64)
#
#     for i in range(n):
#         r[i, i] = np.linalg.norm(u[:, i])
#         # q[:, i] = u[:, i] / r[i, i]
#         if r[i, i] == 0:
#             q[:, i] = 0
#         else:
#             q[:, i] = u[:, i] / r[i, i]
#         for j in range(i + 1, n):
#             r[i, j] = q[:, i].T @ u[:, j]
#             u[:, j] = u[:, j] - r[i, j] * q[:, i]
#     return q, r

def gram_schmidt(M):
    """
    The Modified Graham Schmidt algorithm as presented in the assignment
    :param M - a matrix:
    :return q, r - the qr decomposition of M, such as M=q@r:
    """
    n = M.shape[0]
    # u = a.copy()
    u = M
    q = np.zeros((n, n), dtype=np.float64)
    r = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        r[i, i] = np.linalg.norm(u[:, i])
        if r[i, i] == 0:
            q[:, i] = 0
        else:
            q[:, i] = u[:, i] / r[i, i]
        r[i, i + 1:] = q[:, i] @ u[:, i + 1:]
        u[:, i + 1:] = u[:, i + 1:] - (r[i, i + 1:][np.newaxis, :] * q[:, i][:, np.newaxis])
    return q, r


def very_small(a, b, ep):
    """
    :param a: matrix
    :param b: matrix
    :param ep: small epsilon
    :return: True iff max_{i,j in [n]*[n]} |a_i,j - b_i,j| <= ep
    """
    c = np.abs(a) - np.abs(b)
    return np.all(np.abs(c) <= ep)


def qr_iteration(a, ep=0.0001):
    """
    QR iteration as presented in the assignment
    :param a: matrix
    :param ep: small epsilon, representing zero
    :return: a_ upper triangular matrix whose diagonal is the eigen values of a, and q_ the eigen vectors
    """
    n = a.shape[0]
    # a_ = a.copy()
    a_ = a
    q_ = np.eye(n)
    for i in range(n):
        q, r = gram_schmidt(a_)
        a_ = r @ q
        if very_small(q_, q_ @ q, ep):
            return a_, q_
        q_ = q_ @ q
    return a_, q_


def eigen_gap(a):
    """
    the eigen gap heuristic
    :param a: upper triangular matrix whose diagonal is the eigen values
    :return: argmax_{1<=i<=n/2} (|a_(i+1,i+1) - a_(i,i)|)
    """
    ev = np.diag(a)
    n = len(ev)
    ev = np.msort(ev)
    k = -1
    mx = -1
    for i in range(1, n // 2 + 1):
        if ev[i+1] - ev[i] > mx:
            mx = ev[i+1] - ev[i]
            k = i
    return k


def jaccard_measure(indexes_real, indexes_empirical, clusters_empirical, k):
    """
    calculate the jaccard measure as presented in the assignment
    :param indexes_real: the real cluster indexes for each observation, aka, indexes_real[i] is the cluster of obs[i]
    :param indexes_empirical: the indexes for each observation as obtained from a certain algorithm, aka, indexes_empirical[i] is the cluster of obs[i] in the algorithm
    :param clusters_empirical: list of clusters as obtained from a certain algorithm, aka, clusters_empirical[i] is all j such that obs[j] is in cluster i
    :param k: number of clusters
    :return: the jaccard measure
    """
    n = len(indexes_real)
    denominator = 0
    numerator = 0

    clusters_real = [[] for i in range(k)]
    for i in range(n):
        clusters_real[indexes_real[i]].append(i)

    for cluster in clusters_real:
        count = np.zeros(k)
        for p in cluster:
            count[indexes_empirical[p]] += 1
        for r in count:
            numerator += r * (r - 1) // 2
        denominator += len(cluster) * (len(cluster) - 1) // 2
    for cluster in clusters_empirical:
        denominator += len(cluster) * (len(cluster) - 1) // 2

    denominator -= numerator
    return round(numerator / denominator, 3)
