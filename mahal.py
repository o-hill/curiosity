"""mahal.py
--
Clean your clusters using Mahalanobis distance metrics!
"""
import numpy as np
from sklearn.neighbors import DistanceMetric


def pseudoinverse(M, return_condition_number=False):
    """Find the pseudoinverse of the matrix M using singular value
       decomposition.
    INPUT
        M - array_like
            An (m x n) matrix whose pseudoinverse we want to find.
        return_condition_number - bool [default: False]
            If True, the function will also return the condition number
            associated with the inversion.
    OUTPUT
        pinv - array_like
            The Moore-Penrose pseudoinverse of the matrix M.
        cond - float
            The condition number of the inversion (i.e., the ratio of the
            largest to smallest singular values).
    """
    # Compute the singular value decomposition.
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T

    # Construct the pseudoinverse and compute the condition number.
    M_pinv = V.dot(np.diag(1 / s)).dot(U.T)
    condition_number = s.max() / s.min()

    # If requested, return condition number; otherwise, don't.
    if return_condition_number:
        return M_pinv, condition_number
    else:
        return M_pinv


def mahalanobis(x, mu=None, S=None, return_stats=False):
    """Find the Mahalanobis distance between row vectors in x and mean mu, with
       covariance S. If mu and S are not provided, the mean and covariance of
       x are used instead.
    INPUTS
        x - array_like
            A matrix of row vectors.
        mu - array_like
            Mean vector. mu.shape[1] must equal x.shape[1].
        S - array_like
            The covariance matrix. S.shape[0] == S.shape[1] == x.shape[1]
        return_stats - boolean
            If True, returns the mean and covariance used in the calcuations.
    OUTPUTS
        mahal_dist - array_like
            An array of the Mahalanobis distances associated with the vectors
            in x.
        mu - array_like
            The mean vector used in Mahalanobis calculations.
        S - array_like
            The covariance matrix used.
        """
    x = np.array(x)
    if mu is None:
        mu = x.mean(0)
        S = np.cov(x.T)

    mahal_dist = np.zeros(x.shape[0])
    inv_S = np.linalg.pinv(S)
    for i, x_i in enumerate(x):
        mahal_dist[i] = (x_i - mu).T.dot(inv_S).dot((x_i - mu))

    if return_stats:
        return (mahal_dist, mu, S)
    else:
        return mahal_dist


def find_valid_indices(X):
    """Use Mahalanobis distance to identify core vectors in a cluster."""
    d = mahalanobis(X)
    mu = d.mean()
    std = d.std()
    return np.where(d < mu + 3 * std)[0]
