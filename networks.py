'''

    Neural network components of the curiosity algorithm.

    Thoughts:
        Thinking about training the network at each iteration of the
        algorithm, as this will let the network learn the data in each
        of the clusters as it comes up, instead of us trying to make it
        very generalizable. So it will minimize the loss function for every
        pair of clusters, and then make the predictions which will be the
        new embedding for the data points.

'''

import numpy as np
from keras.models import Sequential
import keras.backend as K
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    AveragePooling2D
    Lambda
)


class Encoder:

    def __init__(self, input_size: tuple,
            model_path: str = '',
            X: np.ndarray,
            labels: np.ndarray) -> None:
        '''Initialize the encoding network.'''
        self.network = load_model(model_path) if model_path else initialize_graph()
        self._X = X
        self.labels = labels


    def initialize_graph(self):
        '''Creates the computational graph for the network.'''

        # For now, just think about the data as a vector in some
        # arbitrary space, and we are maximizing distances between
        # the vectors.
        network = Sequential([
            Dense(32, activation='relu', input_shape=(input_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu')
        ])

        network.compile(optimizer='adam', loss=[self.maximize_distance])
        return network


    def maximize_distance(self, y, y_hat):
        '''Maximize the distance between the clusters.

            @y:     [index, 0, 0, ...] -> maybe? Is this how loss functions work?
            @y:     Cluster labels for each data point.
            @y_hat: The current representations.

            Currently using the L2 norm.
        '''
        index = y[0]
        l2 = K.sqrt(K.dot(y_hat, y_hat))
        cluster_one = 1 / K.sum(1 - y) * K.sum((1 - y) * l2)
        cluster_two = 1 / K.sum(y) * K.sum(y * l2)

        # Maximize the positive by minimizing the negative.
        return -K.abs(cluster_one - cluster_two)


    def embed(self) -> np.ndarray:
        '''Embed the given examples into a new vector space.'''

        # Train the network to maximize the distance between the two clusters.
        self.network.fit(self._X, self._y)

        # Use the learned embeddings for the next iteration.
        return self.network.predict(self._X)















