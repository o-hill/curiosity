'''

    Sandbox for testing cluster partition resolution methods.

'''


from data import main
from larc import *


all_latent, low_d, labels = main()
centroids = [np.mean(low_d[labels == l], axis=0) for l in range(2)]

cluster_evaluation(low_d, labels, centroids)
