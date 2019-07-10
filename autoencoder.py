from glob import glob
from keras.layers import (
    Input,
    Reshape,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    AveragePooling2D,
)

from keras.models import Model
from keras import backend as K
import numpy as np


class Layer:

    def __init__(self, name, output):
        self.name = name
        self.output = output


class Autoencoder:
    """Image autoencoder for CT scan images."""

    def __init__(self,
            weights_path: str = '',
            input_tensor = None,
            image_size: tuple = (28, 28, 1)):

        # Create an input of the appropriate size.
        self.input_image = Input(image_size)

        latent_dim = (image_size[0] // 2) ** 2

        self.build_encoder(latent_dim)
        self.encoder = Model(self.input_image, self.encoded)

        self.build_decoder(image_size[0] // 2)
        self.autoencoder = Model(self.input_image, self.decoded)
        self.autoencoder.compile(optimizer="adam", loss="mean_squared_error")

        if weights_path:
            self.autoencoder.load_weights(weights_path)


    def build_encoder(self, latent_dim: int):
        # Build the first half of autoencoder... the encoder.

        one = Conv2D(32, 3, activation="relu", padding="same")(self.input_image)
        two = Conv2D(32, 3, activation="relu", padding="same")(one)
        # three = MaxPooling2D((2, 2))(two)
        # four = Conv2D(64, 3, activation="relu", padding="same")(two)
        # five = Conv2D(64, 3, activation="relu", padding="same")(four)
        # six = MaxPooling2D((2, 2))(five)
        # seven = Conv2D(128, 3, activation="relu", padding="same")(five)
        eight = Conv2D(64, 3, activation="relu", padding="same")(two)
        nine = MaxPooling2D((2, 2), name="encode_output")(eight)
        ten = Flatten()(nine)
        self.encoded = Dense(latent_dim, activation='relu')(ten)

    def build_decoder(self, dim: int):
        # Decode the encoded CT scan images.

        reshape = Reshape((dim, dim, 1))(self.encoded)
        one = Conv2D(64, 3, activation="relu", padding="same")(reshape)
        two = Conv2D(16, 3, activation="relu", padding="same")(one)
        # three = UpSampling2D((2, 2))(two)
        # four = Conv2D(64, 3, activation="relu", padding="same")(two)
        # five = Conv2D(64, 3, activation="relu", padding="same")(four)
        # six = UpSampling2D((2, 2))(five)
        # seven = Conv2D(32, 3, activation="relu", padding="same")(five)
        # eight = Conv2D(32, 3, activation="relu", padding="same")(seven)
        nine = UpSampling2D((2, 2))(two)
        self.decoded = Conv2D(1, 3, activation="sigmoid", padding="same")(nine)

    def train(self, X: np.ndarray):
        # Train the autoencoder.
        for itr in range(1000):
            print(f"> Iteration {itr} of 1000")
            self.autoencoder.fit(X, X, epochs=1, batch_size=64, shuffle=True)
            from ipdb import set_trace as debug; debug()
            self.autoencoder.save('autoencoder_weights.h5')

if __name__ == '__main__':

    from sklearn.decomposition import PCA
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Need (n_rows, n_cols, n_channels) for keras.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    ones = X_train[np.where(y_train == 1)[0]]
    twos = X_train[np.where(y_train == 2)[0]]

    ae = Autoencoder()
    train = np.vstack((ones, twos))
    ae.train(train)

    latent_representations = ae.encoder.predict(train)[0]
    decomp = PCA(n_components=2)
    plottable = decomp.fit(latent_representations).transform(latent_representations)

    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib as mpl
    mpl.style.use('seaborn')

    plt.plot(plottable)





























