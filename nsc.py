"""
wrapper for the nsc algorithm
"""
import numpy as np
import matrices
import eigen

ep = 0.0001
max_iter = 300


def normalized_spectral_clustering(obs, random, real_k):
    """
    calculating the Normalized Spectral Clustering algorithm
    :param obs: list of points
    :param random: boolean flag representing if we use the eigengap heuristic or not
    :param real_k: k from which obs were generated
    :return:
    1. matrix to use for the kmeans++ algorithm and find the clusters
    2. k that we used to create the T matrix (eigengap if random else real_k)
    """
    # 1
    W = matrices.weighted_matrix(obs)
    n = W.shape[0]
    L = matrices.normalized_laplacian(W)
    a_, q_ = eigen.qr_iteration(L, ep)
    ev = np.diag(a_).argsort()
    k = eigen.eigen_gap(a_)
    if not random:
        k = real_k
    U = q_[:, ev[:k]]
    U_n = np.linalg.norm(U, axis=1)
    T = U / U_n[:, np.newaxis]
    return T, k
    # clusters = kmeans(T, k, n, k, max_iter)

    # return clusters, k


# if __name__ == '__main__':
#     points = np.array([[1.05937462, -5.43792517, 6.42558562],
#                        [-10.05112361, 2.04009153, 6.98458187],
#                        [-3.7811002, 5.28896226, 4.51733151],
#                        [-0.11347536, 1.22593216, 4.45980985],
#                        [3.54849158, -3.66450451, 2.24934561],
#                        [-1.14201363, 3.66193321, 4.12001458],
#                        [1.91095653, -4.04072839, 3.4051221],
#                        [6.00177547, 5.74220283, 2.146897],
#                        [-2.83185619, -1.80837938, 5.86441057],
#                        [0.57857932, -2.50789818, 7.31182911],
#                        [-3.36888538, 0.13314093, 5.00670704],
#                        [6.37009464, -9.81780791, 1.09262253],
#                        [0.19179555, -7.36831047, -6.94050445],
#                        [-4.42479947, 0.80445078, 0.45079205],
#                        [6.73895593, -10.30996741, 9.50158835],
#                        [2.5173836, -7.53776347, 4.76533074],
#                        [-5.7507966, 3.73453708, 8.16525921],
#                        [-0.97747179, 1.17857791, -0.55504477],
#                        [-7.42713474, 5.92786872, 8.62067743],
#                        [2.5764994, -7.74440258, -2.38574537],
#                        [-9.64207474, -0.44839748, 7.55372394],
#                        [9.51644585, -1.19143382, 6.16212157],
#                        [2.25344571, 1.4590045, 4.11803692],
#                        [-2.49226674, 3.22468553, 10.44366305],
#                        [3.34528247, -8.37151378, -2.90079278],
#                        [-4.24519798, -6.38864899, 2.88454107],
#                        [9.59970165, -1.47438674, -1.19564938],
#                        [6.9918479, -8.65431437, 9.7450287],
#                        [-3.55857795, -5.25419416, 2.68787911],
#                        [8.05237927, -1.80291359, -1.17594091]])
#     with open('data.txt', 'r') as f:
#         l = []
#         for line in f:
#             v = list(map(float, line.split(',')))[:-1]
#             l.append(v)
#     points = np.array(l)
#     print('points\n', repr(points), '\n-------------\n')
#     W = matrices.weighted_matrix(points)
#     print('\nWeighted\n', repr(W), '\n-------------\n')
#     n = W.shape[0]
#     L = matrices.normalized_laplacian(W)
#     print('\nLaplacian\n', repr(L), '\n-------------\n')
#     a_, q_ = eigen.qr_iteration(L, ep)
#     print('\nfrom qr_iter\na_bar :\n', repr(a_), '\n\nq_bar :\n', repr(q_), '\n-------------\n')
#     ev = np.diag(a_).argsort()
#     print('\nsorted eigen values :\n', ev, '\n-------------\n')
#     k = eigen.eigen_gap(a_)
#     print('\nheuristic K\n', k, '\n-------------\n')
#     U = q_[:, ev[:k]]
#     print('\nU\n', repr(U), '\n-------------\n')
#     U_n = np.linalg.norm(U, axis=1)
#     T = U / U_n[:, np.newaxis]
#     print('\nT\n', repr(T), '\n-------------\n')
