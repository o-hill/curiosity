import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Subtract, Dot
from keras.layers import Conv2D, MaxPooling2D
import keras.layers
from sklearn.decomposition import PCA
from pylab import ion, close, plot, imshow, imread, figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


def larc_loss(_, vector):
    """Custom loss function for LARC."""
    # return K.square(vector)
    return -K.sum(K.square(vector), axis=1)


if __name__ == "__main__":

    # Load some MNIST data.
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Grab indices for specific numerals.
    ones_idx = np.where(y_train == 1)[0]
    twos_idx = np.where(y_train == 2)[0]
    fours_idx = np.where(y_train == 4)[0]
    fives_idx = np.where(y_train == 5)[0]
    eights_idx = np.where(y_train == 8)[0]
    threes_idx = np.where(y_train == 3)[0]

    # Now grab the actual images.
    ones = x_train[ones_idx]
    twos = x_train[twos_idx]
    threes = x_train[threes_idx]
    fours = x_train[fours_idx]
    fives = x_train[fives_idx]
    eights = x_train[eights_idx]

    # Build a model.
    img_shape = (img_rows, img_cols, 1)
    x1 = Input(shape=img_shape)
    x2 = Input(shape=img_shape)

    # Create a convolutional neural network.
    layer_1 = Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=img_shape)
    layer_2 = Conv2D(32, (3, 3), activation="relu")
    layer_3 = MaxPooling2D(pool_size=(2, 2))
    layer_4 = Conv2D(64, (3, 3), activation="relu")
    layer_5 = MaxPooling2D(pool_size=(2, 2))
    layer_6 = Flatten()
    layer_7 = Dense(200, activation="relu")
    layer_8 = Dropout(0.5)
    layer_9 = Dense(500, activation="relu")
    vector = Dense(100, activation="tanh")

    # First pass through the network.
    x = layer_1(x1)
    x = layer_2(x)
    x = layer_3(x)
    x = layer_4(x)
    x = layer_5(x)
    x = layer_6(x)
    x = layer_7(x)
    x = layer_9(x)
    y1 = vector(x)

    # Second pass through the network.
    x = layer_1(x2)
    x = layer_2(x)
    x = layer_3(x)
    x = layer_4(x)
    x = layer_5(x)
    x = layer_6(x)
    x = layer_7(x)
    x = layer_9(x)
    y2 = vector(x)

    # Our differencing scheme.
    y = Subtract()([y1, y2])
    # y = Dot(axes=1, normalize=True)([y1, y2])
    # y = keras.layers.dot([y1, y2], axes=1, normalize=True)

    model = Model(inputs=[x1, x2], outputs=y)
    model.compile(optimizer="adam", loss=larc_loss)

    forward_model = Model(inputs=x1, outputs=y1)
    forward_model.compile(optimizer="adam", loss="mse")

    data_1 = ones[:500, :]
    data_2 = twos[:500, :]
    data_3 = threes[:500, :]
    data_4 = fours[:500, :]
    data = np.vstack((data_1, data_2, data_3, data_4))
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data_1 = data[idx[: int(len(idx) / 2)], :]
    data_2 = data[idx[int(len(idx) / 2) :], :]
    unnecessary_y = np.zeros(1000)

    model.fit([data_1, data_2], unnecessary_y, epochs=50, shuffle=True)

    # data_1 = ones[:1000, :]
    # data_2 = twos[:1000, :]
    # forward_model = load_model("forward_model.h5")
    x1 = forward_model.predict(ones[:500, :])
    x2 = forward_model.predict(twos[:500, :])
    x3 = forward_model.predict(threes[:500, :])
    x4 = forward_model.predict(fours[:500, :])
    x = np.vstack((x1, x2, x3, x4))
    p = PCA(n_components=3)
    p.fit(x)
    x_ = p.transform(x)
    x1_ = p.transform(x1)
    x2_ = p.transform(x2)
    x3_ = p.transform(x3)
    x4_ = p.transform(x4)
    # for x in x1:
    #     x /= np.linalg.norm(x)
    # for x in x2:
    #     x /= np.linalg.norm(x)
    # x = np.vstack((x1_, x2_, x3_, x4_))
    # for z in x_:
    #     if z[0] < 0 and z[1] < 0:
    #         z[0] += 0.05 * np.random.randn()
    #         z[1] += 0.05 * np.random.randn()

    close("all")
    plt.ion()
    # plot(x1[:, 0], x1[:, 1], "b.")
    # plot(x2[:, 0], x2[:, 1], "r.")
    c1 = 0
    c2 = 1
    plot(x1_[:, c1], x1_[:, c2], "b.")
    plot(x2_[:, c1], x2_[:, c2], "r.")
    plot(x3_[:, c1], x3_[:, c2], "g.")
    plot(x4_[:, c1], x4_[:, c2], "k.")

    # plot(x_[:, 0], x_[:, 1], "y.")

    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection="3d")
    # # ax.scatter(x1_[:, 0], x1_[:, 1], x1_[:, 2], marker="o")
    # # ax.scatter(x2_[:, 0], x2_[:, 1], x2_[:, 2], marker="o")
    # # ax.scatter(x3_[:, 0], x3_[:, 1], x3_[:, 2], marker="o")
    # # ax.scatter(x4_[:, 0], x4_[:, 1], x4_[:, 2], marker="o")
