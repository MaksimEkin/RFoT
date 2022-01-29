import numpy as np
# -*- coding: utf-8 -*-
def component_cluster(params):
    M_r = params["M_k"]
    return np.array([0] * len(M_r)), 1