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


class Autoencoder:
    """Image autoencoder."""

    def __init__(
        self,
        weights_path: str = "",
        image_size: tuple = (28, 28, 1),
        latent_dimension=128,
    ):
        # Create an input of the appropriate size.
        self.input_image = Input(image_size)

        self.latent_dim = latent_dim

        self.build_encoder(latent_dim)
        self.encoder = Model(self.input_image, self.encoded)

        self.build_decoder(image_size[0] // 2)
        self.autoencoder = Model(self.input_image, self.decoded)
        self.autoencoder.compile(optimizer="adadelta", loss="mean_squared_error")

    def build_encoder(self):
        """Build the encoder half of the autoencoder."""
        one = Conv2D(32, 3, activation="relu", padding="same")(self.input_image)
        two = Conv2D(64, 3, activation="relu", padding="same")(one)
        three = MaxPooling2D((2, 2))(two)
        four = Conv2D(64, 3, activation="relu", padding="same")(two)
        five = Conv2D(64, 3, activation="relu", padding="same")(four)
        six = MaxPooling2D((2, 2))(five)
        seven = Conv2D(128, 3, activation="relu", padding="same")(five)
        eight = Conv2D(64, 3, activation="relu", padding="same")(two)
        nine = MaxPooling2D((2, 2), name="encode_output")(two)
        ten = Flatten()(nine)
        self.encoded = Dense(latent_dim, activation='relu')(ten)
