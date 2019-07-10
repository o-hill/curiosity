import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import pylab as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

    ones = x_train[ones_idx]
    twos = x_train[twos_idx]

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

    ones_latent = model.predict(ones)
    twos_latent = model.predict(ones)
    all_latent = np.vstack((ones_latent, twos_latent))
    p = PCA(n_components=2)
    p.fit(all_latent)
    low_d = p.transform(all_latent)
    k = KDE(n_clusters=2)
    k.fit(low_d)
    labels = k.labels_

    plt.close("all")
    first_cluster = np.where(labels == 0)[0]
    second_cluster = np.where(labels == 1)[0]
    for idx in first_cluster:
        plt.plot(low_d[idx, 0], low_d[idx, 1], "bo")
    for idx in second_cluster:
        plt.plot(low_d[idx, 0], low_d[idx, 1], "ro")
