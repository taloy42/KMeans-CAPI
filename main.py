"""
wrapper function for the whole project
"""
import numpy as np
from sklearn.datasets import make_blobs
import argparse
import kmeans_pp
import nsc
import visualization

K_max2 = 20
n_max2 = 500

K_max3 = 20
n_max3 = 500

ep = 0.0001
max_iter = 300


def main():
    """
    create points, calculate the nsc algorithm and the kmeans algorithm on them, and visualize the results
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="Number of observations - integer >=n")
    parser.add_argument("k", type=int, help="Number of centroids - integer >=1")
    parser.add_argument('--rand', default=False, action='store_true')
    args = parser.parse_args()

    n = args.n
    k = args.k
    random = args.rand
    flag = False
    if k < 1 and not random:
        print("K has to be >=1")
        flag = True
    if n <= k and not random:
        print("N has to be >K")
        flag = True

    if flag:
        exit(1)
    d = np.random.randint(2, 4)
    if random:
        if d == 2:
            n_obs = np.random.randint(n_max2 // 2, n_max2 + 1)
            k_obs = np.random.randint(K_max2 // 2, K_max2 + 1)

        else:
            n_obs = np.random.randint(n_max3 // 2, n_max3 + 1)
            k_obs = np.random.randint(K_max3 // 2, K_max3 + 1)

    else:
        n_obs = n
        k_obs = k
    observations, indexes_real = make_blobs(n_samples=n_obs, centers=k_obs, n_features=d)
    real_k = k_obs
    T, k = nsc.normalized_spectral_clustering(observations, random, real_k)
    k_obs = k if random else k_obs

    indexes_nsc, clusters_nsc = kmeans_pp.k_means_pp(T, k_obs, n_obs, k, max_iter)

    indexes_kmeans, clusters_kmeans = kmeans_pp.k_means_pp(observations, k_obs, n_obs, d, max_iter)

    if indexes_kmeans is None or indexes_nsc is None:
        print("allocation error")
        return None

    np.savetxt(fname='data.txt', X=np.append(observations, indexes_real[:, np.newaxis], axis=1), delimiter=',',
               fmt=['%lf'] * d + ['%d'])

    msg = str(k_obs) + "\n"
    for lst in clusters_nsc:
        msg += ','.join(map(str, lst)) + '\n'
    for lst in clusters_kmeans:
        msg += ','.join(map(str, lst)) + '\n'

    with open('clusters.txt', 'w') as f:
        f.write(msg)

    visualization.vis(observations,
                      indexes_real, indexes_nsc, indexes_kmeans,
                      clusters_nsc, clusters_nsc,
                      n_obs, real_k, k_obs)


if __name__ == '__main__':
    main()
