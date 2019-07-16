import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import pylab as plt

# from sklearn.cluster import AgglomerativeClustering as KMeans
from sklearn.cluster import Birch as KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# from autoencoder import *


if __name__ == "__main__":

    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 2

    # The data, split between train and test sets, like I care.
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

    ones = x_train[ones_idx]
    twos = x_train[twos_idx]
    fours = x_train[fours_idx]

    if True:
        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
        )
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        # model.add(Dropout(0.5))
        # model.add(Dense(num_classes, activation="softmax"))
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=["accuracy"],
        )
    else:
        ae = Autoencoder("autoencoder_weights.h5")
        model = ae.encoder

    ones_latent = model.predict(ones)
    twos_latent = model.predict(twos)
    fours_latent = model.predict(fours)

    all_latent = np.vstack((ones_latent, twos_latent, fours_latent))
    all_latent = np.vstack((ones_latent))
    p = PCA(n_components=5)
    p.fit(all_latent)
    low_d = p.transform(all_latent)
    k = KMeans(n_clusters=3)
    k.fit(low_d)
    labels = k.labels_
    lda = LDA(n_components=2)
    lda.fit(all_latent, labels)
    low_d = lda.transform(all_latent)
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

    plt.close("all")
    plt.ion()
    first_cluster = np.where(labels == 0)[0]
    second_cluster = np.where(labels == 1)[0]
    third_cluster = np.where(labels == 2)[0]
    plt.plot(low_d[first_cluster, 0], low_d[first_cluster, 1], "b.")
    plt.plot(low_d[second_cluster, 0], low_d[second_cluster, 1], "r.")
    plt.plot(low_d[third_cluster, 0], low_d[third_cluster, 1], "g.")

    c_one = np.mean(low_d[first_cluster], axis=0)
    c_two = np.mean(low_d[second_cluster], axis=0)

    plt.plot([c_one[0]], [c_one[1]], 'y+')
    plt.plot([c_two[0]], [c_two[1]], 'k+')

    # X = np.vstack((ones, twos))
    # y = np.atleast_1d(labels[: len(ones) + len(twos)])
    # model.fit(X, y, epochs=1)
    X_ = low_d[labels == 0, :]
    labels_ = np.zeros(X_.shape[0])
