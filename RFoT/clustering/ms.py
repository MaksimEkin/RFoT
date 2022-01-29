# -*- coding: utf-8 -*-
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np


def ms_cluster(params):
    """"""

    M_r = params["M_k"]
    random_state = params["random_state"]

    # single sample in the factor
    if len(M_r) == 1:
        return np.array([0]), 1

    # single unique element(s) in the factor
    if len(np.unique(M_r)) == 1:
        return np.array([0] * len(M_r)), 1

    X = np.array(list(zip(M_r, np.zeros(len(M_r)))))
    bandwidth = estimate_bandwidth(X, quantile=0.1, random_state=random_state)

    if bandwidth <= 0:
        bandwidth = None

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    try:
        ms.fit(X)
    except Exception:
        raise Exception("Error when clustering!")

    labels = ms.labels_
    nopt = len(np.unique(labels))
    return labels, nopt