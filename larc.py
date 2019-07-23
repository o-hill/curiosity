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
from ipdb import set_trace as debug
import keras
import pylab as plt
from mahal import find_valid_indices


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
        in_shape = (img_rows, img_cols, 1)

        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=in_shape)
        )
        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(100, activation="relu", name="latent"))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        self.model = model

    def latent(self, X):
        """Return the latent representations, given raw input."""
        # X = X.reshape(X.shape[0], self.img_rows, self.img_cols, 1)
        ins, outs = self.model.layers[0].input, self.model.get_layer("latent").output
        return K.function([ins], [outs])([X])[0]

    def project_latent(self, X, nb_dims=32):
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

    def predict_labels(self, X):
        """Predict labels given using the CNN."""
        print("> Predicting labels with network.")
        probs = self.model.predict(X)
        labels = np.argmax(probs, 1)
        return labels

    def fit(self, X, nb_epochs=1, use_network=False):
        """Explore data in X."""
        # Find the network's latest latent representation.
        # self.X_ = self.latent(X)
        X_latent, X_proj = self.project_latent(X)
        self.X_ = X_latent
        self.X_proj = X_proj

        if use_network:
            # Predict labels using network now.
            new_labels = self.predict_labels(X)
        else:
            # Cluster the data.
            print("> Clustering projected data.")
            c = Cluster(n_clusters=3)
            c.fit(X_proj)
            new_labels = c.labels_
        if self.labels is not None:
            self.labels = self.validate_labels(new_labels)
        else:
            self.labels = new_labels
        self.find_valid_training_data()
        self.X_train = X[self.valid_idx, :]
        self.y_train = label_to_one_hot(self.labels[self.valid_idx])

        # Now train the neural network given these labels...
        print("> Training the network.")
        self.model.fit(
            self.X_train, self.y_train, epochs=nb_epochs, batch_size=32, shuffle=True
        )

    def find_valid_training_data(self):
        """Look for core vectors to train the neural network."""
        self.X_train = []
        self.y_train = []
        self.valid_idx = []

        print("> Finding valid training data.")
        for label in np.unique(self.labels):
            if label == 2:
                continue
            class_idx = np.where(self.labels == label)[0]
            Z = self.X_proj[class_idx, :]
            valid_vector_idx = find_valid_indices(Z)
            self.valid_idx.extend(
                class_idx[valid_vector_idx]
            )  # map back to original indices
        self.valid_idx = np.array(self.valid_idx)


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
    fives_idx = np.where(y_train == 5)[0]
    eights_idx = np.where(y_train == 8)[0]
    threes_idx = np.where(y_train == 3)[0]

    x_train = x_train / 255
    ones = x_train[ones_idx]
    twos = x_train[twos_idx]
    threes = x_train[threes_idx]
    fours = x_train[fours_idx]
    fives = x_train[fives_idx]
    eights = x_train[eights_idx]
    first = threes
    second = fives
    # second = np.vstack((eights, twos))

    ones_twos = np.vstack((first, second))
    ones_twos -= ones_twos.mean(0)

    l = LARC()
    l.fit(ones_twos, nb_epochs=1, use_network=False)
    # l.fit(ones_twos, nb_epochs=1, use_network=False)
    # l.fit(ones_twos, nb_epochs=1, use_network=False)
    # l.fit(ones_twos, nb_epochs=1, use_network=False)
    # l.fit(ones_twos, nb_epochs=1, use_network=False)
    l.fit(ones_twos, nb_epochs=1, use_network=True)
    _, X_ = l.project_latent(ones_twos, nb_dims=32)
    labels = l.predict_labels(ones_twos)

    plt.ion()
    plt.close("all")
    first_cluster = np.where(labels == 0)[0]
    second_cluster = np.where(labels == 1)[0]
    third_cluster = np.where(labels == 2)[0]
    plt.plot(X_[first_cluster, 0], X_[first_cluster, 1], "b.")
    plt.plot(X_[second_cluster, 0], X_[second_cluster, 1], "r.")

    acc_1 = (
        (labels[: len(first)] == 1).sum()
        + (labels[len(first) : len(first) + len(second)] == 0).sum()
    ) / (len(first) + len(second))

    acc_2 = (
        (labels[: len(first)] == 0).sum()
        + (labels[len(first) : len(first) + len(second)] == 1).sum()
    ) / (len(first) + len(second))
    acc = np.max((acc_1, acc_2))
    print(f"Clustering accuracy is {acc*100:0.2f}%")
