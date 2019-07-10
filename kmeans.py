'''

    Pure NumPy K-Means implementation.

'''

import numpy as np
from pudb import set_trace as debug


def d(x: np.ndarray, y: np.ndarray) -> float:
    '''Compute euclidean distance between x and y.'''
    debug()
    return np.linalg.norm(x - y)


def D(x: np.ndarray, centroids: list) -> float:
    '''Compute the distance from x to the closest centroid.'''
    return min([d(x, c) for c in centroids])


def centroid_initialization(X: np.ndarray, k: int) -> np.ndarray:
    '''Use k-means++ centroid initialization.'''

    centroids, indices = [ ], [ ]

    # 1. Uniformly random choice of initial centroid.
    indices.append(np.random.randint(X.shape[0]))
    centroids.append(np.copy(X[indices[-1]]))

    # 2. Choose the rest of the centroids with probabilities to ensure separation.
    for centroid in range(1, k):

        # Find probability of choosing each next point.
        probs = np.array([
            D(x, centroids) ** 2 if i not in indices else 0
            for i, x in enumerate(X)
        ])
        probs /= np.sum(probs)

        # Choose the next centroid.
        indices.append(np.random.choice(X.shape[0], p=probs))
        centroids.append(np.copy(X[indices[-1]]))

    return np.array(centroids)


def k_means(X: np.ndarray, k: int = 2) -> np.array:
    '''Takes the data as a 2d matrix where the rows are data points and returns a vector of assignments.'''

    # 1. Initialize the centroids.
    centroids = centroid_initialization(X, k)

    # 2. Cluster!
    previous = np.zeros(X.shape[0], dtype=int)
    assignments = -np.ones(X.shape[0], dtype=int)

    while not all(previous == assignments):

        previous = assignments

        for idx in range(len(assignments)):
            assignments[idx] = np.argmin([d(X[idx], c) for c in centroids])

        # Reassign the centroids to be the center of mass for the cluster.
        centroids = np.array([np.mean(X[assignments == c], axis=0) for c in range(k)])

    return assignments


def time_means(X: np.ndarray, k: np.ndarray) -> None:
    '''Time my code vs sklearn. Results - sklearn uses threads so not much competition.'''

    from time import time
    from sklearn.cluster import KMeans

    start = time()
    for _ in range(1000):
        assignments = k_means(X, k=k)
    end = time()

    print(f'Native K-Means took {(end - start) / 1000} seconds')

    start = time()
    for _ in range(1000):
        assign = KMeans(n_clusters=2).fit(data).labels_
    end = time()

    print(f'SKLearn K-Means took {(end - start) / 1000} seconds')



if __name__ == '__main__':

    # Generate some test data.
    data = list(np.random.multivariate_normal([1, 2], [[1, 0], [5, 10]], 500))
    data += list(np.random.multivariate_normal([14, 8], 4 * np.eye(2), 750))

    data = np.array(data)
    np.random.shuffle(data)

    time_means(data, 2)

    assignments = k_means(data)

    from sklearn.cluster import KMeans
    sk_assign = KMeans(n_clusters=2).fit(data).labels_

    from matplotlib import pyplot as plt
    colors = ['b' if a == 0 else 'g' for a in assignments]
    colors_sk = ['r' if a == 0 else 'k' for a in sk_assign]

    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], color=colors)

    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], color=colors_sk)

    plt.show()




















