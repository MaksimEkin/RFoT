.. RFoT documentation master file, created by
   sphinx-quickstart on Fri Feb 18 04:59:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RFoT's documentation!
================================
Tensor decomposition is a powerful unsupervised Machine Learning method that enables the modeling of multi-dimensional data, including malware data. We introduce a novel ensemble semi-supervised classification algorithm, named Random Forest of Tensors (RFoT), that utilizes tensor decomposition to extract the complex and multi-faceted latent patterns from data. Our hybrid model leverages the strength of multi-dimensional analysis combined with clustering to capture the sample groupings in the latent components, whose combinations distinguish malware and benign-ware. The patterns extracted from a given data with tensor decomposition depend upon the configuration of the tensor such as dimension, entry, and rank selection. To capture the unique perspectives of different tensor configurations, we employ the “wisdom of crowds” philosophy and make use of decisions made by the majority of a randomly generated ensemble of tensors with varying dimensions, entries, and ranks.

As the tensor decomposition backend, RFoT offers two CPD algorithms. First, RFoT package includes the Python implementation of **CP-ALS** algorithm that was originally introduced in the `MATLAB Tensor Toolbox <https://www.tensortoolbox.org/cp.html>`_  :cite:p:`TTB_Software,Bader2006,Bader2008,Eren2022pyCP_ALS`. CP-ALS backend can also be used to **decompose each random tensor in a parallel manner**. RFoT can also be used with the Python implentation of the **CP-APR** algorithm with the **GPU capability** :cite:p:`10.1145/3519602`. Use of CP-APR backend allows decomposing each random tensor configuration both in an **embarrassingly parallel fashion in a single GPU**, and in a **multi-GPU parallel execution**.


Resources
========================================
* `RFoT API Documentation <https://maksimekin.github.io/RFoT/RFoT_API.html>`_
* `Example Notebooks <https://github.com/MaksimEkin/RFoT/tree/main/examples>`_
* `Poster <https://www.maksimeren.com/poster/Random_Forest_of_Tensors_RFoT_MTEM.pdf>`_
* `Example Usage <https://github.com/MaksimEkin/RFoT/tree/main/examples>`_
* `Code <https://github.com/MaksimEkin/RFoT>`_

Installation
========================================

.. code-block:: shell

    conda create --name RFoT python=3.8.5
    conda activate RFoT
    pip install git+https://github.com/MaksimEkin/RFoT.git


Example Usage
========================================
In below example, we use a small sample from `EMBER-2018 <https://github.com/elastic/ember>`_ dataset to classify malware and benign-ware:

* Random tensors in the ensemble are decomposed in a multi-GPU parallel fashion using 2 GPUs. (``n_jobs=2``, ``n_gpus=2``).
* Use CP-APR tensor decomposition backend with GPU (``decomp="cp_apr_gpu"``).
* 200 tensor configurations are randomly sampled (``n_estimators=200``).
* A tensor's dimension in the ensemble could be between 3 and 8 (``min_dimensions=3``, ``max_dimensions=8``). 
* Rank is between 2 and 10. (``rank="random"``, ``min_rank=2``, ``max_rank=10``).
* Cluster uniformity threshold of 1.0 is used (``cluster_purity_tol=1.0``).
* Patterns are captured with Mean Shift (MS) clustering (``clustering="ms"``).
* Feature values representing the tensor entry are not binned (``bin_entry=False``).
* Maximum tensor dimension size representing any feature is equals to the total number of unique values for that feature, where the values are mapped to an index in the tensor dimension (``bin_scale=1``).


.. code-block:: python

    import pickle
    import numpy as np
    from sklearn.metrics import f1_score
    from RFoT import RFoT

    # load the exmple data
    data = pickle.load(open("data/example.p", "rb"))
    X = data["X"]
    y_experiment = data["y_experiment"]
    y_true = data["y_true"]

    # Predict the unknown sample labels
    model = RFoT(
        bin_scale=1,
        min_dimensions=3,
        max_dimensions=8,
        cluster_purity_tol=1.0,
        rank="random",
        min_rank=2,
        max_rank=10,
        n_estimators=200,
        bin_entry=False,
        decomp="cp_apr_gpu",
        clustering="ms",
        n_jobs=2,
        n_gpus=2
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

    print("Num. of Abstaining", abstaining_count)
    print("Percent Abstaining", (abstaining_count / len(unknown_indices)) * 100, "%")
    print("F1=", f1)

`See the examples for more. <https://github.com/MaksimEkin/RFoT/tree/main/examples>`_

How to Cite RFoT?
========================================
.. code-block:: console
    
    @MISC{Eren2022RFoT,
      author = {M. E. {Eren} and C. {Nicholas} and E. {Raff} and R. {Yus} and J. S. {Moore} and B. S. {Alexandrov}},
      title = {RFoT},
      year = {2022},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/MaksimEkin/RFoT}}
    }

    @MISC{eren2021RFoT,
          title={Random Forest of Tensors (RFoT)}, 
          author={M. E. {Eren} and C. {Nicholas} and R. {McDonald} and C. {Hamer}},
          year={2021},
          note={Presented at the 12th Annual Malware Technical Exchange Meeting, Online, 2021}
    }



Acknowledgments
========================================
This work was done as part of Maksim E. Eren's Master's Thesis at the University of Maryland, Baltimore County with the thesis committee members and collaborators Charles Nicholas, Edward Raff, Roberto Yus, Boian S. Alexandrov, and Juston S. Moore.

References
========================================

.. bibliography:: refs.bib



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   RFoT_API
   RFoT
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
