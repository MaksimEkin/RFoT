# -*- coding: utf-8 -*-
import pandas as pd

def bin_columns(df:pd.DataFrame, bin_max_map, dont_bin:[], bin_scale=1.0):
    """
    Maps the tensor dimensions to bins for indexing the tensor.
    The number of bins is specified by the number of unique entries
    for the given dimension in the dataset, and scaled if specified.
    Parameters:
        df: pd.DataFrame,
        dont_bin: str, name of the feature to not create bins.
                  Used to avoid mapping feature values to bins for the
                  feature used as the tensor entry.
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