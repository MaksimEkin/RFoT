# -*- coding: utf-8 -*-
from .ktensor import K_TENSOR
from .sptensor import SP_TENSOR
from .khatrirao_sptensor import khatrirao as mttkrp
from .norm_ktensor import norm
from .arrange_ktensor import arrange
from .fixsigns_ktensor import fixsigns_oneargin
from .innerprod_ktensor import innerprod

from tqdm import tqdm
import numpy as np


class CP_ALS:
    def __init__(
        self, tol=1e-4, n_iters=50, verbose=True, fixsigns=True, random_state=42
    ):

        self.fitchangetol = tol
        self.n_iters = n_iters
        self.verbose = verbose
        self.fixsigns = fixsigns
        self.random_state = random_state
        self.dimorder = []
        np.random.seed(self.random_state)

    def fit(self, coords=[], values=[], rank=2, Minit="random"):

        if rank <= 0:
            raise Exception("Number of components requested must be positive")

        #
        #  Set up for iterations - initializing M and the fit.
        #
        X, M = self.__setup(coords, values, Minit, rank)

        # Extract number of dimensions and norm of X.
        N = X.Dimensions
        normX = np.linalg.norm(X.data, ord=2)

        self.dimorder = np.arange(0, N)

        fit = 0
        R = rank
        M_mttkrp = np.zeros((X.Size[-1], R))

        if self.verbose:
            print("CP_ALS:")

        #
        # Main Loop: Iterate until convergence
        #
        UtU = np.zeros((R, R, N))
        for n in range(N):
            if len(M.Factors[str(n)]) != 0:
                UtU[:, :, n] = np.dot(M.Factors[str(n)].T, M.Factors[str(n)])

        for itr in tqdm(range(self.n_iters), disable=not (self.verbose)):

            fitold = fit

            #
            # Iterate over all N modes of the tensor
            #
            for n in self.dimorder:

                # Calculate Unew = X_(n) * khatrirao(all M except n, 'r').
                Unew = mttkrp(X, M, n)

                # Save the last MTTKRP result for fitness check.
                if n == self.dimorder[-1]:
                    U_mttkrp = Unew

                # Compute the matrix of coefficients for linear system
                target_dimensions = list(np.arange(0, N))
                target_dimensions.pop(target_dimensions.index(n))
                Y = np.prod(UtU[:, :, target_dimensions], 2)
                Unew = np.linalg.lstsq(Y.T, Unew.T, rcond=None)[0].T

                # Normalize each vector to prevent singularities in coefmatrix
                if itr == 0:
                    lambda_ = np.sqrt(np.sum(Unew ** 2, axis=0)).T  # 2-norm
                else:
                    lambda_ = np.max(np.abs(Unew), axis=0).T  # max-norm

                Unew = np.divide(Unew, lambda_)

                M.Factors[str(n)] = Unew
                UtU[:, :, n] = np.dot(M.Factors[str(n)].T, M.Factors[str(n)])

            Utmp = {"Factors": [], "Weights": []}
            Utmp["Factors"] = M.deep_copy_factors()
            Utmp["Weights"] = lambda_
            P = K_TENSOR(Rank=R, Size=M.Size, Minit=Utmp)

            # This is equivalent to innerprod(X,P).
            iprod = np.sum(
                np.multiply(
                    np.sum(
                        np.multiply(M.Factors[str(self.dimorder[-1])], U_mttkrp), axis=0
                    ),
                    lambda_.T,
                )
            )

            if normX == 0:
                fit = norm(P) ** 2 - 2 * iprod
            else:
                normresidual = np.sqrt(normX ** 2 + norm(P) ** 2 - 2 * iprod)
                fit = 1 - (normresidual / normX)

            fitchange = np.abs(fitold - fit)

            # Check for convergence
            if (itr > 0) and (fitchange < self.fitchangetol):
                converged = True
            else:
                converged = False

            if converged:
                break

        #
        # Clean up final result
        #
        P = arrange(P)
        if self.fixsigns:
            P = fixsigns_oneargin(P)

        if self.verbose:

            if normX == 0:
                fit = norm(P) ** 2 - 2 * innerprod(P, X)
            else:
                normresidual = np.sqrt(normX ** 2 + norm(P) ** 2 - 2 * innerprod(P, X))
                fit = 1 - (normresidual / normX)

            print("\nFinal fit=", fit)

        results = {"Factors": [], "Weights": []}
        results["Factors"] = P.Factors
        results["Weights"] = P.Weights

        return results

    def __setup(self, coords, values, Minit, rank):

        if len(coords) == 0:
            raise Exception(
                "Coordinates of the non-zero elements is not passed for sptensor.\
                                Use the Coords parameter."
            )
        if len(values) == 0:
            raise Exception(
                "Non-zero values are not passed for sptensor.\
                            Use the Values parameter"
            )
        if (values < 0).all():
            raise Exception(
                "Data tensor must be nonnegative for Poisson-based factorization"
            )

        X = SP_TENSOR(coords, values)

        M = K_TENSOR(rank, X.Size, Minit, self.random_state)

        return X, M
