import numpy as np
from sklearn.cluster import Birch as Cluster
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


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
        bic += 0.5 * n * np.log(det_cov)
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
