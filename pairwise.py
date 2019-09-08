import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


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

    # Build a model.
    img_shape = (img_rows, img_cols, 1)
    x1 = Input(shape=img_shape)
    x2 = Input(shape=img_shape)

    # Create a convolutional neural network.
    layer_1 = Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=img_shape)
    layer_2 = Conv2D(32, (3, 3), activation="relu")
    layer_3 = MaxPooling(pool_size=(2, 2))
    layer_4 = Conv2D(64, (3, 3), activation="relu")
    layer_5 = MaxPooling(pool_size=(2, 2))
    vector = Flatten()

    x = layer_1(x1)
    x = layer_2(x)
    x = layer_3(x)
    x = layer_4(x)
    x = layer_5(x)
    y1 = vector(x)
