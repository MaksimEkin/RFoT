.. RFoT documentation master file, created by
   sphinx-quickstart on Fri Feb 18 04:59:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RFoT's documentation!
================================
Tensor decomposition is a powerful unsupervised ML method that enables the modeling of multi-dimensional data, including malware data. This paper introduces a novel ensemble semi-supervised classification algorithm, named Random Forest of Tensors (RFoT), that utilizes tensor decomposition to extract the complex and multi-faceted latent patterns from data. Our hybrid model leverages the strength of multi-dimensional analysis combined with clustering to capture the sample groupings in the latent components whose combinations distinguish malware and benign-ware. The patterns extracted from a given data with tensor decomposition depend on the configuration of the tensor such as dimension, entry, and rank selection. To capture the unique perspectives of different tensor configurations we employ the "wisdom of crowds" philosophy, and make use of decisions made by the majority of a randomly generated ensemble of tensors with varying dimensions, entries, and ranks. We show the capabilities of RFoT when classifying malware and benign-ware from the EMBER-2018 dataset.

Resources
========================================
* `Example Notebooks <https://github.com/MaksimEkin/RFoT/tree/main/examples>`_
* `Poster <https://www.maksimeren.com/poster/Random_Forest_of_Tensors_RFoT_MTEM.pdf>`_
* `Example Usage <https://github.com/MaksimEkin/RFoT/tree/main/examples>`_
* `Code <https://github.com/MaksimEkin/RFoT>`_

Installation
========================================
**Option 1: Install using pip**

.. code-block:: shell

    pip install git+https://github.com/MaksimEkin/RFoT.git

**Option 2: Install from source**

.. code-block:: shell

    git clone https://github.com/MaksimEkin/RFoT.git
    cd RFoT
    conda create --name RFoT python=3.8.5
    source activate RFoT
    pip install git+https://github.com/lanl/pyCP_APR.git
    python setup.py install
    
Example Usage
========================================

.. code-block:: python

    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.metrics import f1_score
    from RFoT import RFoT

    # Load/pre-process the dataset
    # Use a small version of EMBER-2018 dataset
    df = pd.read_pickle("data/mini_ember_df.p")
    df.dropna(inplace=True)
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    print(df.info())

    # organize the dataset
    X = df.drop("y", axis=1)
    y_true = np.array(df["y"].tolist())
    y_experiment = y_true.copy()

    # randomly label some as unknown (-1)
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(y_experiment.shape[0]) < 0.3
    y_experiment[random_unlabeled_points] = -1

    # Predict the unknown sample labels
    model = RFoT(
            bin_scale=1,
            min_dimensions=3,
            max_dimensions=8,
            component_purity_tol=1.0,
            rank=2,
            n_estimators=200,
            bin_entry=False,
            clustering="ms",
            n_jobs=50,
    )
    y_pred = model.predict(X, y_experiment)

    # Results
    unknown_indices = np.argwhere(y_experiment == -1).flatten()
    did_predict_indices = np.argwhere(y_pred[unknown_indices] != -1).flatten()
    abstaining_count = len(np.argwhere(y_pred == -1))
    f1 = f1_score(
        y_true[unknown_indices][did_predict_indices],
        y_pred[unknown_indices][did_predict_indices],
        average="weighted",
    )
    print(f1)

`See the examples for more. <https://github.com/MaksimEkin/RFoT/tree/main/examples>`_

References
========================================

.. bibliography:: refs.bib



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   RFoT
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
