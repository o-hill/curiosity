import numpy as np
from sklearn.cluster import Birch as Cluster
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.svm import SVC
from sklearn.neighbors import BallTree


def compute_bic(X, labels):
    """Compute the Bayesian Information Criterion for multivariate Gaussian likelihoods."""
    unique_labels = np.unique(labels)
    N, d = X.shape
    K = len(unique_labels)
    bic = 0
    for k in range(K):
        X_ = X[labels == k, :]
        n = X_.shape[0]
        det_cov = np.linalg.det(np.cov(X_.T))
        bic -= 0.5 * n * np.log(det_cov)
    bic -= N * K * (d + 0.5 * d * (d + 1))
    return bic


class LARC:
    """LAtent Representation Clustering — the curiosity engine!"""

    def __init__(self, img_rows=28, img_cols=28):
        """Build the CNN, etc."""
        # self.cluster_quality_threshold = cluster_quality_threshold
        self.img_rows, self.img_cols = img_rows, img_cols

        # self.X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        # Build a CNN something like VGGNet.
        inputs = Input(shape=input_shape)
        x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        self.x = Dense(128, activation="relu", name="latent")(x)
        y = Dense(64, activation="relu")(self.x)
        self.y = Dense(2, activation="softmax")(y)
        self.model = Model(inputs=inputs, outputs=self.y)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy")

    def latent(self, X):
        """Return the latent representations, given raw input."""
        # X = X.reshape(X.shape[0], self.img_rows, self.img_cols, 1)
        ins, outs = self.model.layers[0].input, self.model.layers[5].output
        return K.function([ins], [outs])([X])[0]

    def fit(self, X):
        """Explore data in X."""
        # Find the network's latest latent representation.
        X_ = self.latent(X)

        # Project data to a lower dimension.
        p = PCA(n_components=5)
        p.fit(X_)
        X_proj = p.transform(X_)

        # Cluster the data.
        c = Cluster(n_clusters=2)
        self.clustering = c.fit(X_proj)
        self.labels = self.clustering.labels_

        # Now train the neural network given these labels...
        self.model.fit(X, self.labels, epochs=5)



def distance(centroids: np.ndarray) -> float:
    '''Evaluate the distance between two centroids.'''
    return np.linalg.norm(centroids[0] - centroids[1])


def densest_radius(X: np.ndarray,
        support_idx: np.ndarray,
        tree: BallTree,
        d_centroids: float) -> int:
    '''Identify the support vector with the densest radius.'''

    return np.argmax([
        len(tree.query_radius(np.atleast_2d(X[i]), r=(0.3*d_centroids))[0]) for i in support_idx
    ])


def ratio(X: np.ndarray,
        vector_idx: int,
        centroid: np.ndarray,
        d_centroids: float,
        tree: BallTree) -> float:
    '''Compute the ratio between the density at the support vector and a centroid.'''

    density_vector = len(tree.query_radius(np.atleast_2d(X[vector_idx]), r=(0.3 * d_centroids))[0])
    density_centroid = len(tree.query_radius(np.atleast_2d(centroid), r=(0.3 * d_centroids))[0])

    if density_centroid == 0:
        density_centroid = 1

    print(f'Density of vector location: {density_vector}')
    print(f'Density of centroid: {density_centroid}')

    return density_vector / density_centroid


def cluster_evaluation(X: np.ndarray, y: np.ndarray, centroids: np.ndarray) -> bool:
    '''Evaluates a clustering and returns true if it should be split.'''

    # Find the geometric relationships with a ball tree.
    print('> Building ball tree.')
    tree = BallTree(np.vstack((X, centroids)))

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
    plt.plot(X[first_cluster, 0], X[first_cluster, 1], 'b.')
    plt.plot(X[second, 0], X[second, 1], 'r.')
    plt.plot([centroids[0][0]], [centroids[0][1]], 'g+')
    plt.plot([centroids[1][0]], [centroids[1][1]], 'k+')
    plt.plot([X[celeb_idx][0]], [X[celeb_idx][1]], 'y+')

    breakpoint()
    w = svm.coef_
    a = -w[0] / w[1]
    x = np.linspace(min(X[:, 0]) - 10, max(X[:, 0]) + 10, 1000)
    plt.plot(x, a * x - (svm.intercept_[0] / w[1]), 'k-')
    return av_ratio < 1


def label_to_one_hot(labels):
    nb_labels = np.unique_labels(labels)


def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)


def power_iteration(A):
    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new



















