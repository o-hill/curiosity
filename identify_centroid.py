'''

    Identify the centroid of a distribution, as
    defined by the densest portion of the distribution.

'''

import numpy as np
from sklearn.neighbors import BallTree

from numpy import atleast_2d as two_d


def determine_radius(X: np.ndarray, tree: np.ndarray) -> float:
    '''Determine an appropriate radius for querying.'''

    d_max = [ ]

    for node in np.random.choice(X.shape[0], min(100, X.shape[0])):
        dist, idx = tree.query(two_d(X[node]), k=15)
        d_max.append(max(dist))

    return np.mean(d_max)


def centroid(X: np.ndarray, tree: BallTree) -> np.ndarray:
    '''Find the centroid of the distribution given by X.'''

    # Find an appropriate radius.
    radius = determine_radius(X, tree)

    # Choose a random initilization.
    start = np.random.choice(X.shape[0])
    density = tree.query_radius(two_d(X[start]), r=

    # Start MCMC-esque exploration procedure.
    for i in range(100):


