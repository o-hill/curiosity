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


if __name__ == "__main__":

    num_classes = 2
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

    plt.ion()
    plt.close("all")
    plt.imshow(threes[355, :, :, 0], cmap="gray")
