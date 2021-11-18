# -*- coding: utf-8 -*-

from .cp_als_numpy.cp_als import CP_ALS

from .utilities.bin_columns import bin_columns
from .utilities.sample_tensor_configs import setup_tensors
from .utilities.build_tensor import setup_sptensor
from .utilities.istarmap import istarmap

from .clustering.gmm import gmm_cluster
from .clustering.ms import ms_cluster

from multiprocessing import Pool
from collections import Counter
import tqdm
import numpy as np
import pandas as pd
import operator


class RFoT:
    def __init__(
        self,
        max_depth=1,
        min_rank=2,
        max_rank=20,
        min_dimensions=3,
        max_dimensions=3,
        min_cluster_search=2,
        max_cluster_search=12,
        component_purity_tol=0.9,
        cluster_quality_tol=-1,
        n_estimators=80,
        rank="random",
        clustering="gmm",
        decomp="cp_als",
        zero_tol=1e-08,
        dont_bin=list(),
        bin_scale=1.0,
        bin_entry=False,
        bin_max_map={"max": 10 ** 6, "bin": 10 ** 3},
        tol=1e-4,
        n_iters=50,
        verbose=True,
        decomp_verbose=False,
        fixsigns=True,
        random_state=42,
        n_jobs=1,
    ):

        self.max_depth = max_depth
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.min_dimensions = min_dimensions
        self.max_dimensions = max_dimensions
        self.min_cluster_search = min_cluster_search
        self.max_cluster_search = max_cluster_search
        self.component_purity_tol = component_purity_tol
        self.cluster_quality_tol = cluster_quality_tol
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

        allowed_decompositions = ["cp_als", "numpy_cp_als"]

        if self.decomp in ["cp_als", "numpy_cp_als"]:
            self.backend = CP_ALS(
                tol=self.tol,
                n_iters=self.n_iters,
                verbose=self.decomp_verbose,
                fixsigns=self.fixsigns,
                random_state=self.random_state,
            )
        else:
            raise Exception(
                "Unknown tensor decomposition method. Choose from: "
                + ", ".join(allowed_decompositions)
            )

        assert (
            self.cluster_quality_tol > 0 or self.component_purity_tol > 0
        ), "Cluster purity and/or component purity must be >0"

        if self.clustering == "gmm":
            self.cluster = gmm_cluster
        elif self.clustering == "ms":
            self.cluster = ms_cluster
        else:
            raise Exception("Unknown clustering method is chosen.")

    def get_params(self):
        """ """

        return vars(self)

    def get_params_backend(self):
        """ """

        return vars(self.backend)

    def set_params_backend(self, **parameters):
        """ """

        for parameter, value in parameters.items():
            setattr(self.backend, parameter, value)
        return self.backend

    def set_params(self, **parameters):
        """ """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X: np.array, y: np.ndarray):
        """ """

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
        for key, config in tensor_configs.items():
            tasks.append((config, X, y))

        pool = Pool(processes=self.n_jobs)
        for tv in tqdm.tqdm(
            pool.istarmap(self._get_tensor_votes, tasks, chunksize=1),
            total=len(tasks),
            disable=not (self.verbose),
        ):
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
        predicted_indices = []
        y_pred = y.copy()
        for sample_idx, sample_votes in votes.items():
            max_value = max(sample_votes)
            max_index = sample_votes.index(max_value)
            y_pred[sample_idx] = max_index
            predicted_indices.append(sample_idx)

        return y_pred, predicted_indices

    def _get_tensor_votes(self, config, X, y):

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

        decomp = self.backend.train(
            curr_tensor["nnz_coords"],
            curr_tensor["nnz_values"],
            int(config["rank"]),
        )
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
            cluster_labels, n_opt = self.cluster(params)

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
        votes,
    ):

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
            if self.cluster_quality_tol > 0:
                cluster_quality_score = max(
                    cluster_c_known_label_counts.values()
                ) / sum(cluster_c_known_label_counts.values())

                # cluster quality is poor, skip this cluster
                if cluster_quality_score < self.cluster_quality_tol:
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
        """"""

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
