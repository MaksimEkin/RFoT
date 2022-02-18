# -*- coding: utf-8 -*-
import pandas as pd

def bin_columns(df:pd.DataFrame, bin_max_map, dont_bin:[], bin_scale=1.0):
    """
    Maps the tensor dimensions to bins for indexing the tensor.
    The number of bins is specified by the number of unique entries
    for the given dimension in the dataset, and scaled if specified.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset X.
    bin_max_map : dict
        Determines the maximum dimension size.
    dont_bin : []
        List of column indixes from X to not to bin.
    bin_scale : float, optional
        Size of the dimension. The default is 1.0.

    Returns
    -------
    df : pd.DataFrame
        Dataset X with binned features.

    """

    dimension_ranges = dict()
    names = list(df.columns)
    for name in names:
        if name not in dont_bin:
            dimension_ranges[name] = int(df[name].nunique() * bin_scale)

    for dim, range_ in dimension_ranges.items():

        if range_ > bin_max_map["max"]:
            range_ = bin_max_map["bin"]

        if range_ == 0:
            range_ = 1

        df[dim] = pd.cut(df[dim], bins=range_, right=True, labels=False)

    return df
