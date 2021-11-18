from sklearn.mixture import GaussianMixture
from scipy import stats
import numpy as np


def gmm_cluster(params):

    M_r = params["M_k"]
    min_ = params["min_cluster_search"]
    max_ = params["max_cluster_search"]
    random_state = params["random_state"]

    nopt = GaussianMixture_find_n(M_r, min_, max_, random_state)
    return GaussianMixture_get_labels(M_r, nopt, random_state), nopt


def GaussianMixture_find_n(M_r, min_, max_, random_state):
    """
    Finds the optimal number of clusters using BIC score for component r of KRUSKAL tensor M_i.
    Parameters:
        M_r: dict, KRUSKAL tensor's first latent factor.
    """

    if len(np.unique(M_r)) == 1:
        return 1

    gm_bic = []
    gm_score = []
    for n in range(min_, max_):

        # the number of clusters attempting to search is more than the number of unique values
        if n > len(np.unique(M_r)):
            break

        gm = GaussianMixture(
            n_components=n, n_init=10, tol=1e-3, max_iter=250, random_state=random_state
        ).fit(M_r.reshape(-1, 1))
        gm_bic.append(-gm.bic(M_r.reshape(-1, 1)))
        gm_score.append(gm.score(M_r.reshape(-1, 1)))

    if len(gm_bic) == 0:
        n_opt = 1
    else:
        n_opt = np.argmax(np.log(gm_bic)) + min_

    return n_opt


def GaussianMixture_get_labels(M_r, n, random_state):
    """
    Extracts the cluster labels for KRUSKAL tensor M_r for n clusters.
    M_r is the rth component of the KRUSKAL tensor M_i.
    Parameters:
        M_r: dict, KRUSKAL tensor's first latent factor.
        n: int, number of clusters.
    """
    if len(M_r) == 1:
        return [0]

    gm = GaussianMixture(
        n_components=n, random_state=random_state, n_init=10, tol=1e-3, max_iter=1000
    ).fit(M_r.reshape(-1, 1))
    labels = gm.predict(M_r.reshape(-1, 1))

    return labels
