'''

    Determine whether we should make the next split or not.

'''

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import BallTree


def distance(centroids: np.ndarray) -> float:
    '''Evaluate the distance between two centroids.'''
    return np.linalg.norm(centroids[0] - centroids[1])


def densest_radius(X: np.ndarray,
        support_idx: np.ndarray,
        tree: BallTree,
        d_centroids: float) -> int:
    '''Identify the support vector with the densest radius.'''

    return np.argmax([
        len(tree.query_radius(np.atleast_2d(X[i]), r=(0.1*d_centroids))[0]) for i in support_idx
    ])


def ratio(X: np.ndarray,
        vector_idx: int,
        centroid: np.ndarray,
        d_centroids: float,
        tree: BallTree) -> float:
    '''Compute the ratio between the density at the support vector and a centroid.'''

    density_vector = len(tree.query_radius(np.atleast_2d(X[vector_idx]), r=(0.1 * d_centroids))[0])
    density_centroid = len(tree.query_radius(np.atleast_2d(centroid), r=(0.1 * d_centroids))[0])

    if density_centroid == 0:
        density_centroid = 1

    print(f'Density of vector location: {density_vector}')
    print(f'Density of centroid: {density_centroid}')

    return density_vector / density_centroid


def cluster_evaluation(X: np.ndarray, y: np.ndarray, centroids: np.ndarray) -> bool:
    '''Evaluates a clustering and returns true if it should be split.'''

    # Find the geometric relationships with a ball tree.
    print('> Building ball tree.')
    tree = BallTree(X)

    # Find the support vectors for the data.
    print('> Computing SVM.')
    svm = SVC(kernel='linear', gamma='auto').fit(X, y)
    support_idx = svm.support_

    # Find the support vector with the densest radius.
    print('> Determining vector with densest radius.')
    d_centroids = distance(centroids)
    celeb_idx = support_idx[densest_radius(X, support_idx, tree, d_centroids)]

    # Find the ratio of densities.
    av_ratio = np.mean([ratio(X, celeb_idx, centroids[i], d_centroids, tree) for i in range(2)])

    # Bound the acceptance by a threshold.
    from matplotlib import pyplot as plt
    plt.close('all')
    plt.ion()
    first_cluster = np.where(y == 0)[0]
    second = np.where(y == 1)[0]
    plt.plot(X[first_cluster, 0], X[first_cluster, 3], 'b.')
    plt.plot(X[second, 0], X[second, 3], 'r.')
    plt.plot([centroids[0][3]], [centroids[0][3]], 'g+')
    plt.plot([centroids[1][0]], [centroids[1][3]], 'k+')
    plt.plot([X[celeb_idx][0]], [X[celeb_idx][3]], 'y+')

    # breakpoint()
    # w = svm.coef_
    # a = -w[0] / w[1]
    # x = np.linspace(min(X[:, 0]) - 10, max(X[:, 0]) + 10, 1000)
    # plt.plot(x, a * x - (svm.intercept_[0] / w[1]), 'k-')
    return av_ratio < 1




















