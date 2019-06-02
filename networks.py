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
from keras.models import Model
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

    def __init__(self, input_size: tuple, model_path: str = '') -> None:
        '''Initialize the encoding network.'''

        self.initialize_graph()



    def initialize_graph(self) -> None:
        '''Creates the computational graph for the network. self.network will be defined after this.'''

        # For now, just think about the data as a vector in some
        # arbitrary space, and we are maximizing distances between
        # the vectors.
        self.network = Sequential([
            Dense(32, activation='relu', input_shape=(input_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu')
        ])

        self.network.compile(optimizer='adam', loss=[self.maximize_distance])


    def maximize_distance(self, y, y_hat):
        '''Maximize the distance between the clusters.'''
        cluster_one = 1 / K.sum(1 - y) * K.sum()















