# -*- coding: utf-8 -*-
"""
Tensor decomposition is a powerful unsupervised Machine Learning method that enables the modeling of multi-dimensional data, including malware data. We introduce a novel ensemble semi-supervised classification algorithm, named Random Forest of Tensors (RFoT), that utilizes tensor decomposition to extract the complex and multi-faceted latent patterns from data. Our hybrid model leverages the strength of multi-dimensional analysis combined with clustering to capture the sample groupings in the latent components, whose combinations distinguish malware and benign-ware. The patterns extracted from a given data with tensor decomposition depend upon the configuration of the tensor such as dimension, entry, and rank selection. To capture the unique perspectives of different tensor configurations, we employ the “wisdom of crowds” philosophy and make use of decisions made by the majority of a randomly generated ensemble of tensors with varying dimensions, entries, and ranks.

As the tensor decomposition backend, RFoT offers two CPD algorithms. First, RFoT package includes the Python implementation of **CP-ALS** algorithm that was originally introduced in the `MATLAB Tensor Toolbox <https://www.tensortoolbox.org/cp.html>`_  :cite:p:`TTB_Software,Bader2006,Bader2008`. CP-ALS backend can also be used to **decompose each random tensor in a parallel manner**. RFoT can also be used with the Python implentation of the **CP-APR** algorithm with the **GPU capability** :cite:p:`10.1145/3519602`. Use of CP-APR backend allows decomposing each random tensor configuration both in an **embarrassingly parallel fashion in a single GPU**, and in a **multi-GPU parallel execution**.

"""

from pyCP_ALS import CP_ALS
from pyCP_APR import CP_APR

from .utilities.bin_columns import bin_columns
from .utilities.sample_tensor_configs import setup_tensors
from .utilities.build_tensor import setup_sptensor
from .utilities.istarmap import istarmap

from .clustering.gmm import gmm_cluster
from .clustering.ms import ms_cluster
from .clustering.component import component_cluster

from multiprocessing import Pool
from collections import Counter
import tqdm
import numpy as np
import pandas as pd
import operator


class RFoT:
    def __init__(self,
                 max_depth=1,
                 min_rank=2,
                 max_rank=20,
                 min_dimensions=3,
                 max_dimensions=3,
                 min_cluster_search=2,
                 max_cluster_search=12,
                 component_purity_tol=-1,
                 cluster_purity_tol=0.9,
                 n_estimators=80,
                 rank="random",
                 clustering="ms",
                 decomp="cp_als",
                 zero_tol=1e-08,
                 dont_bin=list(),
                 bin_scale=1.0,
                 bin_entry=False,
                 bin_max_map={"max": 10 ** 6,
                 "bin": 10 ** 3},
                 tol=1e-4,
                 n_iters=50,
                 verbose=True,
                 decomp_verbose=False,
                 fixsigns=True,
                 random_state=42,
                 n_jobs=1,
                 n_gpus=1,
                 gpu_id=0):
        """
        Initilize the **RFoT.RFoT** class.

        Parameters
        ----------
        
        
        max_depth : int, optional
            Maximum number of times to run RFoT. The default is 1.
            
            .. note::
                * If ``max_depth=1``, data is fit with RFoT once.
                * Otherwise, when ``max_depth`` is more than 1, each corresponding fit of the data with RFoT will work on the abstaining predictions from the prior fit.
            
        min_rank : int, optional
            Minimum tensor rank R to be randomly sampled. The default is 2.
            
            .. note::
                * Should be more than 1. ``min_rank`` should be less than ``max_rank``.
                * Only used when ``rank="random"``.


        max_rank : int, optional
            Maximum tensor rank R to be randomly sampled. The default is 20.
            
            .. note::
                * ``max_rank`` should be more than ``min_rank``.
                * Only used when ``rank="random"``.


        min_dimensions : int, optional
            When randomly sampling tensor configurations, minimum number of dimensions a tensor should have within the ensemble of random tensor configurations. The default is 3.

        max_dimensions : int, optional
            When randomly sampling tensor configurations, maximum number of dimensions a tensor should have within the ensemble of random tensor configurations. The default is 3.

        min_cluster_search : int, optional
            When searching for the number of clusters via likelihood in GMM, minimum number of clusters to try. The default is 2.
        max_cluster_search : int, optional
            When searching for the number of clusters via likelihood in GMM, maximum number of clusters to try. The default is 12.
        component_purity_tol : float or int, optional
            The purity score threshold for the latent factors. The default is -1.\n
            This threshold is calculated based on the known instances in the component.\n
            If the purity score of the latent factor is lower then the threshold ``component_purity_tol``, component is discarded and would not be used in obtaining clusters.
            
            .. note::
                * By default ``component_purity_tol=-1``.
                * When ``component_purity_tol=-1``, component uniformity is not used in deciding whether to discard the components, and only ``cluster_purity_tol`` is used.
                * Either ``component_purity_tol`` or ``cluster_purity_tol`` must be more than 0.
            
        cluster_purity_tol : float, optional
            The purity score threshold for the clusters. The default is 0.9.
            This threshold is calculated based on the known instances in the cluster.\n
            If the purity score of the cluster is lower then the threshold ``cluster_purity_tol``, cluster is discarded and would not be used in the semi-supervised class voting of the unknown samples in the same cluster.
            
            .. note::
                * When ``cluster_purity_tol=-1``, cluster uniformity is not used in deciding whether to discard the clusters, and only ``component_purity_tol`` is used. 
                * Either ``component_purity_tol`` or ``cluster_purity_tol`` must be more than 0.
            
        n_estimators : int, optional
            Number of random tensor configurations in the ensemble. The default is 80.
            
            .. caution::
                * Based on the hyper-parameter configurations, and the number of features in the dataset, it is possible to have less number of random tensor configurations than the one specified in ``n_estimators``.
            
        rank : int or string, optional
            Method for assigning rank for each random tensor to be decomposed. The default is "random".\n
            When ``rank="random"``, the rank for decomposition is sampled randomly from the range (``min_rank``, ``max_rank``).\n
            All the tensors in the ensemble can also be decomposed with same rank (example: ``rank=2``).
            
        clustering : string, optional
            Clustering method to be used for capturing the patterns from the latent factors. The default is "ms".\n
            
            .. admonition:: Options
            
                * ``clustering="ms"`` (Mean Shift)
                * ``clustering="component"``
                * ``clustering="gmm"`` (Gaussian Mixture Model)
            
            
        decomp : string, optional
            Tensor decomposition backend/algorithm to be used. The default is "cp_als".\n
            
            .. admonition:: Options
            
                * ``decomp="cp_als"`` (Alternating least squares for CANDECOMP/PARAFAC Decomposition)
                * ``decomp="cp_apr"`` (CANDECOMP/PARAFAC Alternating Poisson Regression) 
                * ``decomp="cp_apr_gpu"`` (CP-APR with GPU)
                * ``decomp="debug"``
            
            .. note::
                * GPU is used when ``decomp="cp_apr_gpu"``.
                * ``decomp="debug"`` allows serial computation where any error or warning would be raised to the user level.
            
        zero_tol : float, optional
            Samples who are close to the zero, where closeness defined by ``zero_tol``, are removed from the latent factor. The default is 1e-08.
            
        dont_bin : list, optional
            List of column (feature) indices whose values should not be binned. The default is list().
            
        bin_scale : float, optional
            When using a given column (feature) as a tensor dimension, the feature values are binned to create feature value to tensor dimension mapping. This allows a feature value to be represented by an index in the tensor dimension for that feature. The default is 1.0.\n
            When ``bin_scale=1.0``, the size of the dimension that represents the given feature will be equal to the number of unique values in that column (feature).
            
            .. seealso::
                * See `Pandas Cut <https://pandas.pydata.org/docs/reference/api/pandas.cut.html>`_ for value binning.
            
        bin_entry : bool, optional
            If ``bin_entry=True``, the features that are used as tensor entry are also binned. The default is False.
            
        bin_max_map : dict, optional
            ``bin_max_map`` prevents any dimension of any of the tensors in the ensemble to be too large. The default is ``bin_max_map={"max": 10 ** 6, "bin": 10 ** 3}``. \n
            Specifically, ``bin_max_map["bin"]`` is used to determine the size of the dimension when:
            
            :math:`bin\_scale \cdot |f_i| > bin\_max\_map["max"]`

            
        tol : float, optional
            CP-ALS hyper-parameter. The default is 1e-4.
        n_iters : int, optional
             Maximum number of iterations (epoch) to run the tensor decomposition algorithm. The default is 50.
        verbose : bool, optional
            If ``verbose=True``, progress of the method is displayed. The default is True.
        decomp_verbose : bool, optional
            If ``decomp_verbose=True``, progress of the tensor decomposition backend is displayed for each random tensor. The default is False.
        fixsigns : bool, optional
            CP-ALS hyper-parameter. The default is True.
        random_state : int, optional
            Random seed. The default is 42.
        n_jobs : int, optional
            Number of prallel tensor decompositions to perform when decomposing the random tensors from the ensemble. The default is 1.
        n_gpus : int, optional
            Number of GPUs. The default is 1.
            
            .. note::
                * Only used when ``decomp="cp_apr_gpu"``.
                * When ``n_gpus`` is more than 1, and when ``n_jobs`` is more than one, multi-GPU parallel execution is performed. For example, ``n_gpus=2`` and ``n_jobs=2`` will use 2 GPUs, and 1 job will be run on each GPU in parallel.
            
        gpu_id : int, optional
            GPU device ID when using GPU. The default is 0.
            
            .. note::
                * Only used when ``decomp="cp_apr_gpu"``.
                * Not considered when ``n_gpus`` is more than 1.

        Raises
        ------
        Exception
            Invalid parameter selection.

        Returns
        -------
        None.

        """

        self.max_depth = max_depth
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.min_dimensions = min_dimensions
        self.max_dimensions = max_dimensions
        self.min_cluster_search = min_cluster_search
        self.max_cluster_search = max_cluster_search
        self.component_purity_tol = component_purity_tol
        self.cluster_purity_tol = cluster_purity_tol
        self.n_estimators = n_estimators
        self.rank = rank
        self.clustering = clustering
        self.decomp = decomp
        self.zero_tol = zero_tol
        self.dont_bin = dont_bin
        self.bin_scale = bin_scale
        self.bin_entry = bin_entry
        self.bin_max_map = bin_max_map
        self.tol = tol
        self.n_iters = n_iters
        self.verbose = verbose
        self.decomp_verbose = decomp_verbose
        self.fixsigns = fixsigns
        self.random_state = random_state
        self.classes = None
        self.n_jobs = n_jobs
        self.n_gpus = n_gpus
        self.gpu_id = gpu_id

        self.allowed_decompositions = ["cp_als", "cp_apr", "cp_apr_gpu", "debug"]


        assert (
            self.cluster_purity_tol > 0 or self.component_purity_tol > 0
        ), "Cluster purity and/or component purity must be >0"

        if self.clustering == "gmm":
            self.cluster = gmm_cluster
        elif self.clustering == "ms":
            self.cluster = ms_cluster
        elif self.clustering == "component":
            self.cluster = component_cluster
        else:
            raise Exception("Unknown clustering method is chosen.")

    def get_params(self):
        """
        Returns the parameters of the RFoT object.

        Returns
        -------
        dict
            Parameters and data stored in the RFoT object.

        """

        return vars(self)

    def set_params(self, **parameters):
        """
        Used to set the parameters of RFoT object.

        Parameters
        ----------
        **parameters : dict
            Dictionary of parameters where keys are the variable names.

        Returns
        -------
        object
            RFoT object.

        """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X: np.array, y: np.ndarray):
        """
        Semi-supervised prediction of the unknown samples (with labels -1) based on the known samples.

        .. important::
            * Use -1 for the unknown samples.
            * In returned ``y_pred``, samples with -1 predictions are said to be **abstaining predictions** (i.e. model says **"we do not know that the label for that sample is"**).
            * Returned ``y_pred`` includes both known and unknown samples, where the labels of unknown samples may have changed from the original ``y``.

        .. admonition:: Example Usage
        
            .. code-block:: python

                from RFoT import RFoT
                from sklearn import datasets
                from sklearn.metrics import f1_score
                import numpy as np

                # load the dataset
                iris = datasets.load_iris()
                X = iris["data"]
                y = (iris["target"] == 2).astype(np.int) 

                y_true = y.copy()
                y_experiment = y_true.copy()

                # label 30% some as unknown
                rng = np.random.RandomState(42)
                random_unlabeled_points = rng.rand(y_experiment.shape[0]) < 0.3
                y_experiment[random_unlabeled_points] = -1

                # predict with RFoT
                model = RFoT(
                        bin_scale=1,
                        max_dimensions=3,
                        component_purity_tol=1.0,
                        min_rank=2,
                        max_rank=3,
                        n_estimators=50,
                        bin_entry=True,
                        clustering="ms",
                        max_depth=2,
                        n_jobs=50,
                )
                y_pred = model.predict(X, y_experiment)

                # look at results
                unknown_indices = np.argwhere(y_experiment == -1).flatten()
                did_predict_indices = np.argwhere(y_pred[unknown_indices] != -1).flatten()
                abstaining_count = len(np.argwhere(y_pred == -1))
                f1 = f1_score(
                    y_true[unknown_indices][did_predict_indices],
                    y_pred[unknown_indices][did_predict_indices],
                    average="weighted",
                )

                print("------------------------")
                print("Num. of Abstaining", abstaining_count)
                print("Percent Abstaining", (abstaining_count / len(unknown_indices)) * 100, "%")
                print("F1=", f1)

        .. admonition:: Example Usage

            .. code-block:: python

                # y is the vector of known and unknown labels passed to RFoT
                # y_pred is the prediction returned by RFoT
                # y_true is the ground truth

                import numpy as np
                from sklearn.metrics import f1_score

                unknown_indices = np.argwhere(y == -1).flatten()
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


        Parameters
        ----------
        X : np.array
            Features matrix X where columns are the m features and rows are the n samples.
        y : np.ndarray
            Vector of size n with the label for each sample. Unknown samples have the labels -1.

        Returns
        -------
        y_pred : np.ndarray
            Predictions made over the original y. Known samples are kept as is. Unknown samples
            that are no longer labeled as -1 did have prediction. Samples that are still -1 are
            the abstaining predictions.

        """

        #
        # Input verification
        #
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(y, list):
            y = np.array(y)

        if np.count_nonzero(y == -1) == 0:
            raise Exception("No unknown samples found. Label unknown with -1 in y.")

        assert (
            np.count_nonzero(y == -2) == 0
        ), "Do not use -2 as the label as it is internally used!"
        assert len(X) == len(
            y
        ), "Number of samples does not match the number of labels!"
        assert (
            np.count_nonzero(y == -1) > 0
        ), "No unknown samples found! Use -1 in labels to mark the unknown samples."

        #
        # Setup
        #

        # add column to X to serve as sample IDs
        X = np.insert(X, 0, np.arange(0, len(X)), axis=1)

        # convert the features matrix into dataframe
        X = pd.DataFrame(X)
        y_pred = y.copy()
        y_pred_flag = y.copy()

        X_curr = X.copy()
        y_curr = y.copy()
        curr_indices = np.arange(0, len(X))

        #
        # Main loop, depth
        #
        for depth in range(self.max_depth):
            n_abstaining = np.count_nonzero(y_pred == -1)

            y_pred_curr, predicted_indices = self._tree(X_curr, y_curr, depth)
            y_pred[curr_indices] = y_pred_curr
            y_pred_flag[curr_indices[predicted_indices]] = -2
            n_abstaining_new = np.count_nonzero(y_pred == -1)

            # no change in abstaining predictions, no need to continue
            if n_abstaining_new == n_abstaining:
                break

            # form the data for the next depth
            curr_indices = np.argwhere(y_pred_flag >= -1).flatten()

            X_curr = X.iloc[curr_indices].copy()
            X_curr[0] = np.arange(0, len(X_curr))
            y_curr = y[curr_indices].copy()

        return y_pred

    def _tree(self, X: np.array, y: np.array, depth: int):
        """
        Creates random tensor configurations, then builds the tensors in COO format.
        These tensors are then decomposed and the sample patters are captured with clustering
        over the latent factors for the first dimension. Then semi-supervised voting
        is performed over the clusters. These votes are returned.

        Parameters
        ----------
        X : np.array
            Features matrix X where columns are the m features and rows are the n samples.
        y : np.ndarray
            Vector of size n with the label for each sample. Unknown samples have the labels -1.
        depth : int
            Number of times to run RFoT.

        Returns
        -------
        y_pred : np.ndarray
            Predictions made over the original y. Known samples are kept as is. Unknown samples
            that are no longer labeled as -1 did have prediction. Samples that are still -1 are
            the abstaining predictions.
        predicted_indices : np.ndarray
            The indices in y_pred where unknown labels did have prediction values.

        """

        # unique classes
        classes = list(set(y))
        classes.pop(classes.index(-1))
        self.classes = classes

        #
        # Sample the first set of tensor options
        #
        tensor_configs = setup_tensors(
            self.min_dimensions,
            self.max_dimensions,
            X,
            self.random_state + depth,
            self.n_estimators,
            self.rank,
            self.min_rank,
            self.max_rank,
        )

        #
        # Work on each random tensor configuration
        #
        tensor_votes = list()
        tasks = []

        for idx, (key, config) in enumerate(tensor_configs.items()):
            if self.decomp in ["cp_apr_gpu"]:
                if self.n_gpus == 1:
                    tasks.append((config, X, y, self.gpu_id))
                else:
                    tasks.append((config, X, y, idx%self.n_gpus))
            else:
                tasks.append((config, X, y))

        if self.decomp in ["cp_als", "cp_apr"]:
            pool = Pool(processes=self.n_jobs)
            for tv in tqdm.tqdm(
                pool.istarmap(self._get_tensor_votes, tasks, chunksize=1),
                total=len(tasks),
                disable=not (self.verbose),
            ):
                tensor_votes.append(tv)

        elif self.decomp in ["cp_apr_gpu"]:
            pool = Pool(processes=self.n_jobs)
            for tv in tqdm.tqdm(
                pool.istarmap(self._get_tensor_votes, tasks, chunksize=1),
                total=len(tasks),
                disable=not (self.verbose),
            ):
                tensor_votes.append(tv)

            #for task in tqdm.tqdm(tasks, total=len(tasks), disable=not (self.verbose)):
            #    tv = self._get_tensor_votes(config=task[0], X=task[1], y=task[2], gpu_id=task[3])
            #    tensor_votes.append(tv)

        elif self.decomp in ["debug"]:
            for task in tqdm.tqdm(tasks, total=len(tasks), disable=not (self.verbose)):
                tv = self._get_tensor_votes(config=task[0], X=task[1], y=task[2])
                tensor_votes.append(tv)

        #
        # Combine votes from each tensor
        #
        votes = dict()
        for tv in tensor_votes:
            for sample_idx, sample_votes in tv.items():
                if sample_idx in votes:
                    for idx, v in enumerate(sample_votes):
                        votes[sample_idx][idx] += v
                else:
                    votes[sample_idx] = sample_votes

        #
        # Max vote on the results of current depth
        #
        self.votes = votes
        predicted_indices = []
        y_pred = y.copy()
        for sample_idx, sample_votes in votes.items():

            # no decision was made (50-50)
            if len(set(sample_votes)) == 1:
                y_pred[sample_idx] = -1
                continue

            max_value = max(sample_votes)
            max_index = sample_votes.index(max_value)
            y_pred[sample_idx] = max_index
            predicted_indices.append(sample_idx)

        return y_pred, predicted_indices

    def _get_tensor_votes(self, config, X, y, gpu_id=0):
        """
        Sets up the tensor decomposition backend. Then bins the features to build the given
        current tensor. Then the tensor is decomposed and the sample patters are captured with
        clustering over the latent factors for the first dimension. Then semi-supervised voting
        is performed over the clusters. These votes are returned.

        Parameters
        ----------
        config : dict
            Dictionary of tensor configuration.
        X : np.array
            Features matrix X where columns are the m features and rows are the n samples.
        y : np.ndarray
            Vector of size n with the label for each sample. Unknown samples have the labels -1.
        gpu_id : int, optional
            If running CP-APR, which GPU to use. The default is 0.

        Returns
        -------
        votes : dict
            Dictionary of votes for the samples.

        """

        #
        # setup backend
        #
        if self.decomp in ["cp_als", "debug"]:
            backend = CP_ALS(
                tol=self.tol,
                n_iters=self.n_iters,
                verbose=self.decomp_verbose,
                fixsigns=self.fixsigns,
                random_state=self.random_state,
            )
        elif self.decomp in ["cp_apr"]:
            backend = CP_APR(
                n_iters=self.n_iters,
                verbose=self.decomp_verbose,
                random_state=self.random_state,
            )
        elif self.decomp in ["cp_apr_gpu"]:
            backend = CP_APR(
                n_iters=self.n_iters,
                verbose=self.decomp_verbose,
                random_state=self.random_state,
                method='torch',
                device='gpu',
                device_num=gpu_id,
                return_type='numpy'
            )
        else:
            raise Exception(
                "Unknown tensor decomposition method. Choose from: "
                + ", ".join(self.allowed_decompositions)
            )


        # original indices
        all_indices = np.arange(0, len(X))
        votes = {}

        #
        # bin the dimensons
        #
        if self.bin_entry:
            curr_entry = config["entry"]
            curr_dims = config["dimensions"]
            curr_df = bin_columns(
                X[curr_dims + [curr_entry]].copy(),
                self.bin_max_map,
                self.dont_bin + [0],
                self.bin_scale,
            )
        else:
            curr_entry = config["entry"]
            curr_dims = config["dimensions"]
            curr_df = bin_columns(
                X[curr_dims + [curr_entry]].copy(),
                self.bin_max_map,
                self.dont_bin + [curr_entry] + [0],
                self.bin_scale,
            )

        #
        # Factorize the current tensor
        #
        curr_tensor = setup_sptensor(curr_df, config)

        decomp = backend.fit(
            coords = curr_tensor["nnz_coords"],
            values = curr_tensor["nnz_values"],
            rank = int(config["rank"]),
        )
        del backend
        # use the latent factor representing the samples (mode 0)
        latent_factor_0 = decomp["Factors"]["0"]

        #
        # Work on each component
        #
        for k in range(latent_factor_0.shape[1]):

            Mk = latent_factor_0[:, k]

            #
            # mask out elements close to 0
            #
            mask = ~np.isclose(Mk, 0, atol=self.zero_tol)
            M_m = Mk[mask]
            curr_y = y[mask]
            known_sample_indices = np.argwhere(y[mask] != -1).flatten()
            unknown_sample_indices = np.argwhere(y[mask] == -1).flatten()

            if len(curr_y) == 0:
                continue

            #
            # Capture clusters from current component
            #
            params = {
                "M_k": M_m,
                "min_cluster_search": self.min_cluster_search,
                "max_cluster_search": self.max_cluster_search,
                "random_state": self.random_state,
            }
            try:
                cluster_labels, n_opt = self.cluster(params)
            except Exception:
                # error when clustering this component, skip
                continue

            #
            # Calculate Component Quality
            #
            if self.component_purity_tol > 0:
                purity_score = self._component_quality(
                    n_opt, cluster_labels, known_sample_indices, curr_y
                )

                # poor component quality, poor purity among clusters, skip component
                if purity_score < self.component_purity_tol:
                    continue

            #
            # Semi-supervised voting
            #
            votes = self._get_cluster_votes(
                n_opt,
                cluster_labels,
                known_sample_indices,
                unknown_sample_indices,
                curr_y,
                all_indices,
                mask,
                votes,
            )

        return votes

    def _get_cluster_votes(
        self,
        n_opt,
        cluster_labels,
        known_sample_indices,
        unknown_sample_indices,
        curr_y,
        all_indices,
        mask,
        votes,):
        """
        Performs semi-supervised voting.

        Parameters
        ----------
        n_opt : int
            Number of clusters.
        cluster_labels : np.ndarray
            List if cluster labels for each sample.
        known_sample_indices : np.ndarray
            Indices of the known samples.
        unknown_sample_indices : TYPE
            Indices of the unknown samples.
        curr_y : np.ndarray
            Labels.
        all_indices : np.ndarray
            Original indices.
        mask : np.ndarray
            Zero tol mask.
        votes : dict
            Votes so far.

        Returns
        -------
        votes : dict
            Updated votes.

        """

        for c in range(n_opt):

            # current cluster sample informations
            cluster_c_indices = np.argwhere(cluster_labels == c).flatten()

            # empty cluster
            if len(cluster_c_indices) == 0:
                continue

            cluster_c_known_indices = np.intersect1d(
                known_sample_indices, cluster_c_indices
            )
            cluster_c_unknown_indices = np.intersect1d(
                unknown_sample_indices, cluster_c_indices
            )
            cluster_c_known_labels = curr_y[cluster_c_known_indices]

            # everyone is known in the cluster
            if len(cluster_c_unknown_indices) == 0:
                continue

            # no known samples in the cluster - abstaining prediction
            if len(cluster_c_known_labels) == 0:
                continue

            # count the known labels in the cluster
            cluster_c_known_label_counts = dict(Counter(cluster_c_known_labels))

            # cluster quality
            if self.cluster_purity_tol > 0:
                cluster_quality_score = max(
                    cluster_c_known_label_counts.values()
                ) / sum(cluster_c_known_label_counts.values())

                # cluster quality is poor, skip this cluster
                if cluster_quality_score < self.cluster_purity_tol:
                    continue

            # vote
            vote_label = max(
                cluster_c_known_label_counts.items(), key=operator.itemgetter(1)
            )[0]

            org_unknown_indices = all_indices[mask][cluster_c_unknown_indices]
            for idx in org_unknown_indices:
                if idx in votes:
                    votes[idx][vote_label] += 1

                else:
                    votes[idx] = [0] * len(self.classes)
                    votes[idx][vote_label] += 1

        return votes

    def _component_quality(self, n_opt, cluster_labels, known_sample_indices, curr_y):
        """
        Calculates component quality based on cluster purity score.

        Parameters
        ----------
        n_opt : int
            Number of clusters.
        cluster_labels : np.ndarray
            Labels for the samples in the cluster.
        known_sample_indices : np.ndarray
            Array of indices for known samples.
        curr_y : np.ndarray
            Labels for known and unknown samples.

        Returns
        -------
        float
            Purity score.

        """

        maximums = []
        total = 0

        for c in range(n_opt):
            cluster_c_indices = np.argwhere(cluster_labels == c).flatten()

            # empty cluster
            if len(cluster_c_indices) == 0:
                continue

            cluster_c_known_indices = np.intersect1d(
                known_sample_indices, cluster_c_indices
            )

            # no known samples in the cluster - abstaining prediction
            if len(cluster_c_known_indices) == 0:
                continue

            cluster_c_known_labels = curr_y[cluster_c_known_indices]
            cluster_c_known_label_counts = dict(Counter(cluster_c_known_labels))
            maximums.append(max(cluster_c_known_label_counts.values()))
            total += len(cluster_c_known_indices)

        # if none of the clusters had known instances
        if total == 0:
            return -np.inf

        purity_score = sum(maximums) / total

        return purity_score
