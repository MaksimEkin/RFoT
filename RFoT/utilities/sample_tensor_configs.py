import pandas as pd
import numpy as np
import random

def setup_tensors(
    min_dimensions:int, 
    max_dimensions:int,
    X:pd.DataFrame,
    random_state:int,
    n_estimators:int,
    rank,
    min_rank:int,
    max_rank:int):
    """
    Samples random set of tensor configurations specified by the dimension
    names, entry feature name, and the rank.
    """
    if min_rank < 1:
        min_rank = 1
    
    # reduce max dimensions by one since we will add the first dimension to be samples
    max_dimensions -= 1
    
    # random seed for sampling
    random.seed(random_state)
    np.random.seed(random_state)
    
    # get the name for the possible dimensions
    dimensions = list(X.columns)[1:]

    # Possible tensor dimension sizes
    possible_dimension_sizes = list(range(min_dimensions-1, max_dimensions+1))
    
    # Sample n from the possible dimension sizes
    target_dim_sizes = random.choices(possible_dimension_sizes, k=n_estimators)
    random_dimensions = list()
    
    # for each possible number of dimensions setup, choose random set of dimensions
    for dim_size in target_dim_sizes:
        random_dimensions.append(random.sample(dimensions, dim_size))
    
    # sort the dimensions, randomly choose entry, randomly choose rank, add the sample dimension
    tensor_setups=dict()
    for ii, dims in enumerate(random_dimensions):

        # sort the dimensions by the unique number of elements
        n_unique_per_dimension = list()
        for d in dimensions:
            n_unique_per_dimension.append(X[d].nunique())
        dims = [x for _,x in sorted(zip(n_unique_per_dimension,dims), reverse=True)]

        # find the possible tensor entries that are not in the dimensions already
        not_in_dimensions = list(list(set(dims)-set(dimensions)) + list(set(dimensions)-set(dims)))

        tensor_setups[str(ii)] = dict()
        tensor_setups[str(ii)]["dimensions"] = [0]+dims # [0] for the Sample dimension (for mode 0)
        tensor_setups[str(ii)]["entry"] = random.choice(not_in_dimensions)
        if rank == "random":
            tensor_setups[str(ii)]["rank"] = np.random.randint(min_rank, max_rank, 1)[0]
        else:
            tensor_setups[str(ii)]["rank"] = rank
            
    return tensor_setups