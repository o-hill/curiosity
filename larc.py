import numpy as np
from keras.datasets import mnist
from sklearn.cluster import AgglomerativeClustering as Cluster
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras
import pylab as plt

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


def label_to_one_hot(labels):
    """Convert labels to one-hot encodings."""
    nb_labels = len(np.unique(labels))
    one_hot = np.zeros((len(labels), nb_labels))
    for itr, l in enumerate(labels):
        one_hot[itr, l] = 1
    return one_hot


class LARC:
    """LAtent Representation Clustering — the curiosity engine!"""

    def __init__(self, img_rows=28, img_cols=28):
        """Build the CNN, etc."""
        # self.cluster_quality_threshold = cluster_quality_threshold
        self.img_rows, self.img_cols = img_rows, img_cols
        self.labels = None
        num_classes = 2

        # self.X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
        )
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu", name="latent"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=["accuracy"],
        )
        self.model = model

        # Build a CNN something like VGGNet.
        # inputs = Input(shape=input_shape)
        # x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
        # x = Conv2D(64, (3, 3), activation="relu")(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # # x = Dropout(0.5)(x)
        # x = Conv2D(64, (3, 3), activation="relu")(x)
        # x = Flatten()(x)
        # self.x = Dense(128, activation="relu", name="latent")(x)
        # out = Dropout(0.5)(self.x)
        # y = Dense(64, activation="relu")(out)
        # self.y = Dense(2, activation="softmax")(y)
        # self.model = Model(inputs=inputs, outputs=self.y)
        # self.model.compile(optimizer="adam", loss="categorical_crossentropy")

    def latent(self, X):
        """Return the latent representations, given raw input."""
        # X = X.reshape(X.shape[0], self.img_rows, self.img_cols, 1)
        ins, outs = self.model.layers[0].input, self.model.layers[5].output
        return K.function([ins], [outs])([X])[0]

    def project_latent(self, X, nb_dims=10):
        """Project latent image representations into target dimensions."""
        # Find the network's latest latent representation.
        print("> Finding latent representations.")
        X_ = self.latent(X)

        # Project data to a lower dimension.
        print("> Projecting to lower dimension.")
        p = PCA(n_components=nb_dims)
        p.fit(X_)
        X_proj = p.transform(X_)
        return X_, X_proj

    def validate_labels(self, new_labels):
        """See if labels need to be swapped."""
        if (new_labels == self.labels).sum() / len(new_labels) < 0.5:
            idx0 = new_labels == 0
            idx1 = new_labels == 1
            new_labels[idx0] = 1
            new_labels[idx1] = 0
        return new_labels

    def fit(self, X, nb_epochs=1):
        """Explore data in X."""
        # Find the network's latest latent representation.
        # self.X_ = self.latent(X)
        X_latent, X_proj = self.project_latent(X)
        self.X_ = X_latent

        # Project data to a lower dimension.
        # p = PCA(n_components=10)
        # p.fit(self.X_)
        # self.X_, X_proj = p.transform(self.X_)

        # Cluster the data.
        print("> Clustering projected data.")
        c = Cluster(n_clusters=2)
        c.fit(X_proj)
        new_labels = c.labels_
        if self.labels is not None:
            self.labels = self.validate_labels(new_labels)
        else:
            self.labels = new_labels
        self.y = label_to_one_hot(self.labels)

        # Now train the neural network given these labels...
<<<<<<< HEAD
        print("> Training the network.")
        self.model.fit(X, self.y, epochs=nb_epochs, batch_size=32, shuffle=True)


if __name__ == "__main__":

    # Load some MNIST data.
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    ones_idx = np.where(y_train == 1)[0]
    twos_idx = np.where(y_train == 2)[0]
    fours_idx = np.where(y_train == 4)[0]

    ones = x_train[ones_idx] / 255
    twos = x_train[twos_idx] / 255
    fours = x_train[fours_idx] / 255

    ones_twos = np.vstack((twos, fours))
    # idx = np.arange(len(ones_twos))
    # np.random.shuffle(idx)
    # ones_twos = ones_twos[idx, :]

    # TODO: Add Mahalanobis refinement!

    l = LARC()
    l.fit(ones_twos, nb_epochs=1)
    _, X_ = l.project_latent(ones_twos, nb_dims=4)
    labels = l.labels

    plt.close("all")
    first_cluster = np.where(labels == 0)[0]
    second_cluster = np.where(labels == 1)[0]
    third_cluster = np.where(labels == 2)[0]
    plt.plot(X_[first_cluster, 0], X_[first_cluster, 1], "b.")
    plt.plot(X_[second_cluster, 0], X_[second_cluster, 1], "r.")

    acc_1 = (
        (labels[: len(ones)] == 1).sum()
        + (labels[len(ones) : len(ones) + len(twos)] == 0).sum()
    ) / (len(ones) + len(twos))

    acc_2 = (
        (labels[: len(ones)] == 0).sum()
        + (labels[len(ones) : len(ones) + len(twos)] == 1).sum()
    ) / (len(ones) + len(twos))
    acc = np.max((acc_1, acc_2))
    print(f"Clustering accuracy is {acc*100:0.2f}%")
=======
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
    support_idx = SVC(gamma='auto').fit(X, y).support_

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



















>>>>>>> 35fff97218b55e6a4b0ee0fb22111f3a890306a0
