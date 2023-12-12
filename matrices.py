"""
creating the matrices needed for the project
"""
import numpy as np


def weighted_matrix(obs):
    """
    create the weighted matrix W as presented in the assignment
    """
    w = obs[:, :, np.newaxis] - np.swapaxes(obs[np.newaxis, :, :], 1, 2)
    w = np.linalg.norm(w, axis=1)
    w = np.exp(-w / 2) - np.eye(w.shape[0])
    return w


def diagonal_degree_matrix(w):
    """
    calculate the diagonal degree matrix for W
    :param w: weight matrix
    :return: D, diagonal degree matrix
    """
    return np.diag(np.sum(w, axis=1))


def diagonal_minus_sqrt(w):
    """
    calculate w^-0.5
    :param w: matrix
    :return: w^-0.5
    """
    return np.diag(np.power(np.sum(w, axis=1), -0.5))


def normalized_laplacian(w):
    """
    calculate the normalized laplacian of the weight matrix
    :param w: weight matrix
    :return: normalized laplacian as presented in the assignment
    """
    n = w.shape[0]
    d = diagonal_minus_sqrt(w)
    return np.eye(n)-d@w@d