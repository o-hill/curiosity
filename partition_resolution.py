'''

    Sandbox for testing cluster partition resolution methods.

'''


from data import main
from larc import *

from sklearn.neighbors import BallTree
from identify_centroid import centroid, determine_radius
from tqdm import tqdm


all_latent, low_d, labels = main()
centroids = [np.mean(low_d[labels == l], axis=0) for l in range(2)]

# cluster_evaluation(low_d, labels, centroids)

ones = low_d[labels == 0]
tree = BallTree(ones)
one_points, radius, proposal = centroid(ones, tree)


def approx_equal(one: np.ndarray, two: np.ndarray) -> bool:
    '''Are the two arrays approximately equal?'''
    return (one - two < 1).all()


# from matplotlib import pyplot as plt

# plt.close('all')
# plt.ion()
# plt.plot(ones[:, 0], ones[:, 1], 'r.')
# plt.plot(one_points[:, 0], one_points[:, 1], 'k+')

# # proposal = np.mean(one_points[30:], axis=0)
# plt.plot([proposal[0]], [proposal[1]], 'g.')
# print(f'Density at proposal: {len(tree.query_radius(np.atleast_2d(proposal), r=radius)[0])}')

max_point = np.argmax([len(tree.query_radius(np.atleast_2d(p), r=radius)[0]) for p in ones])
# plt.plot(ones[max_point][0], ones[max_point][1], 'b.')
# print(f'Density at truth: {len(tree.query_radius(np.atleast_2d(ones[max_point]), r=radius)[0])}')

# print(f'Arrays approximately equal? {approx_equal(proposal, max_point)}')


print('Times right:')
print(sum([approx_equal(centroid(ones, tree)[2], max_point) for _ in tqdm(range(30))]))










