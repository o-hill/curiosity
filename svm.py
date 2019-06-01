'''

    Lagrangian Support Vector Machine implementation based on
    Mangesarian & Musicant 2001.

'''

import numpy as np



def svm_lk(A: np.ndarray, D: np.ndarray, nu: int, itmax: int, tol: float) -> tuple:
    '''
        Computes a solution to the given SVM problem using the
        Lagrangian Support Vector Machine algorithm, and constrained
        to a linear kernel.

        Inputs:
            @A: (m x n) matrix containing the data points.
            @D: (m x m) matrix with +/- 1 on the diagonal representing classification.
            @nu: parameter for the Sherman-Morrison-Woodbury inversion scheme.
            @itmax: maximum number of iterations to perform.
            @tol: tolerance of error between successive iterations in the solution.

        Returns:
            @it: the number of iterations performed.
            @opt: measure of error in the solution.
            @w: variables describing the solution hyperplane.
            @gamma: hyperplane bias.
    '''

    m, n = A.shape
    alpha = 1.9 / nu
    e = np.ones((m, 1))

    # Construct the H matrix.
    H = D @ np.hstack((A, -e))

    # Compute the single SMW inverse.
    S = H @ np.linalg.inv(np.eye(n + 1) / nu + H.T @ H)

    u = nu * (1 - S @ (H.T @ e))
    u_old = u + 1

    nonneg = lambda x: (np.abs(x) + x) / 2

    it = 0
    while it < itmax and np.linalg.norm(u - u_old, 2) > tol:

        # e + ((Q * u_i - e) - alpha * u)_+ -> equation 15.
        z = (1 + nonneg(((u / nu + H @ (H.T @ u)) - alpha * u) - 1))

        # Update u_old.
        u_old = u

        # Finish update of u in equation 15 -> inv(Q) * z.
        u = nu * (z - S @ (H.T @ z))
        it += 1

    # Extract the hyperplane variables.
    w = A.T @ D @ u

    # Extract the bias term.
    gamma = -e.T @ D @ u

    return it, np.linalg.norm(u - u_old, 2), w, gamma[0][0]



if __name__ == '__main__':

    # Generate some test data.
    data = list(np.random.multivariate_normal([1, 2], [[1, 0], [0, 10]], 500))
    data += list(np.random.multivariate_normal([14, 8], 4 * np.eye(2), 750))

    # data -> (m x n) -> (1250 x 2)
    A = np.array(data)
    indices = np.arange(0, A.shape[0], 1)
    np.random.shuffle(indices)

    it, opt, w, gamma = svm_lk(A[indices],
        np.diag([1 if i > 499 else -1 for i in indices]),
        0.1,
        4000,
        0.001)

    print(it, opt, w, gamma)
    m = -w[0] / w[1]
    b = gamma / w[1]

    from matplotlib import pyplot as plt
    colors = ['b' if a > 499 else 'g' for a in range(indices.shape[0])]
    line = np.linspace(0, 15, 10000)

    plt.scatter(A[:, 0], A[:, 1], color=colors)
    plt.plot(line, m * line + b)
    plt.ylim(-10, 18)
    plt.show()










