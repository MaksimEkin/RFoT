"""
Builds a tensor in COO format.
"""
import pandas as pd
import numpy as np

def setup_sptensor(df:pd.DataFrame, tensor_option:dict):
    """
    Sets up sparse tensor specified by non-zero coordinates and list of values for each
    coordinate entry (index) in the tensor.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset X.
    tensor_option : dict
        Dictionary of tensor configuration defining the column indices to be dimensions, tensor rank, and entry.

    Returns
    -------
    dict
        Dicitonary with non-zero coordinates and non-zero values.

    """

    tensor_dict = {}
    nnz_coords = [] # coordinates of the non-zero values
    nnz_values = [] # non-zero values for each coordinate

    # find the links
    unique_links = pd.DataFrame(df.groupby(tensor_option["dimensions"] +[tensor_option["entry"]]).size().rename('Freq'))

    # build tensor
    for key,value in unique_links['Freq'].items():
        subscript = []
        for ii in range(len(tensor_option["dimensions"])):
            subscript.append(str(int(key[ii])))

        subs_key = ",".join(subscript)

        if subs_key in tensor_dict:
            tensor_dict[subs_key] = (tensor_dict[subs_key] + key[-1]) / 2
        else:
            tensor_dict[subs_key] = key[-1] # tensor_option["entry"]

    for key, value in tensor_dict.items():
        subscript_str = key.split(",")
        subscript = []
        for sub in subscript_str:
            subscript.append(int(sub))

        nnz_coords.append(subscript)
        nnz_values.append(value)

    return {"nnz_coords":np.array(nnz_coords, dtype="int"), "nnz_values":np.array(nnz_values)}
