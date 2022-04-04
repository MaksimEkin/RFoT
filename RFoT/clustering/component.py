"""
Treats a single component as a cluster.
"""

import numpy as np
# -*- coding: utf-8 -*-
def component_cluster(params):
    """
    Uses a single component as a cluster.

    Parameters
    ----------
    params : dict
        Dict containing the latent component.

    Returns
    -------
    np.ndarray
        Cluster labels.
    int
        Number of clusters.

    """
    M_r = params["M_k"]
    return np.array([0] * len(M_r)), 1
