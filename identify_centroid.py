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
        dist, idx = tree.query(two_d(X[node]), k=20)
        d_max.append(max(dist))

    return np.mean(d_max)


def densest(run: list, tree: BallTree, radius: float) -> np.ndarray:
    '''Find the densest point in the run.'''
    return run[np.argmax([
        len(tree.query_radius(np.atleast_2d(p), r=radius)[0]) for p in run
    ])]


def centroid(X: np.ndarray, tree: BallTree) -> np.ndarray:
    '''Find the centroid of the distribution given by X.'''

    # Find an appropriate radius.
    radius = determine_radius(X, tree)
    rho_max = 0
    runs = [ ]

    # Make sure to sample the whole space.
    for init in range(20):

        # Choose a random initilization.
        points = [X[np.random.choice(X.shape[0])]]
        density = len(tree.query_radius(two_d(points[-1]), r=radius)[0])

        # Start MCMC-esque exploration procedure.
        for i in range(100):

            potential = tree.query_radius(two_d(points[-1]), r=radius)
            new_point = X[np.random.choice(potential[0])]
            new_density = len(tree.query_radius(two_d(new_point), r=radius)[0])

            if np.random.random() < (new_density / density):
                points.append(new_point)
                density = new_density

            if rho_max < density:
                rho_max = density
                best_run = init

        runs.append(points)

    return np.array(runs[best_run]), radius, densest(runs[best_run], tree, radius)



