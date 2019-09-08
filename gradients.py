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


def surprise(n, kde):
    p = 100 * np.exp(kde.score(n)) * 0.5
    surprise = p * np.log2(2 * p)
    return surprise


def label_to_one_hot(labels):
    """Convert labels to one-hot encodings."""
    nb_labels = len(np.unique(labels))
    one_hot = np.zeros((len(labels), nb_labels))
    for itr, l in enumerate(labels):
        one_hot[itr, l] = 1
    return one_hot


def both_ways(x):
    return np.mean([grad_norm(x), grad_norm(x, y=[1, 0])])


def both_ways_array(x):
    return np.array([grad_norm(x), grad_norm(x, y=[1, 0])])


def scanner(X):
    return np.array([both_ways(x).mean() for x in X])


def grad_norm(x, y=[0, 1]):
    """Return concatenated list of gradients."""
    grads = get_gradients([np.reshape(x, (1, 28, 28, 1)), [1], [y], 0])
    summed_squares = np.array([(g ** 2).sum() for g in grads])
    norms = np.sqrt(summed_squares.sum())
    return norms


def normies(x):
    """Return concatenated list of gradients."""
    grads = get_gradients([np.reshape(x, (1, 28, 28, 1)), [1], [[1]], 0])
    summed_squares = np.array([(g ** 2).sum() for g in grads])
    norms = np.sqrt(summed_squares)
    return summed_squares


def stats(X):
    grads = np.array([concat_gradients(x).std() for x in X[:500, :]])
    grads = grads[grads > 0]
    return grads.mean()


def get_grads(X):
    return np.array([both_ways_array(x) for x in X])


if __name__ == "__main__":

    # Image sizes.
    num_classes = 2
    img_rows, img_cols = 28, 28
    in_shape = (img_rows, img_cols, 1)

    # BUILD A NETWORK
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=in_shape))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(100, activation="relu", name="latent"))
    model.add(Dense(num_classes, activation="softmax", name="final_output"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    # GET SOME DATA FOR TRAINING
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

    X = np.vstack((ones, twos))
    Z = threes
    y_ = np.hstack((np.zeros(len(ones)), np.ones(len(twos)))).astype(int)
    y = label_to_one_hot(y_)

    model.fit(X, y, epochs=5, batch_size=128, shuffle=True)
    weights = model.trainable_weights  # weight tensors
    gradients = model.optimizer.get_gradients(
        model.total_loss, weights
    )  # gradient tensors
    input_tensors = [
        model.inputs[0],
        model.sample_weights[0],
        model.targets[0],
        K.learning_phase(),
    ]
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    grad_out = get_gradients([X, [1], [[0, 1]], 0])
