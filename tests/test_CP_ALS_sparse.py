"""
    Tests CP-APR Numpy implementation using Sparse Tensor
    
    Run with: python -m unittest test_sparse_Numpy.py
"""
from RFoT.cp_als_numpy.cp_als import CP_ALS
import unittest
import scipy.io as spio

import numpy as np

class TestNumpy(unittest.TestCase):
    
    def setUp(self):
        """Setup the test."""
        
        # Sparse tensor coordinates and non-zero values
        coords = spio.loadmat('../data/test_data/subs.mat', squeeze_me=True)['values'] - 1
        nnz_count = spio.loadmat('../data/test_data/vals_count.mat', squeeze_me=True)['values']
        Uinit = spio.loadmat("../data/test_data/Uinit.mat", squeeze_me=True)["U"]
        
        M_expected_count = dict()
        dimension = 0
        for key, values in spio.loadmat('../data/test_data/m_expected.mat', squeeze_me=True).items():
            if 'm_' in key:
                M_expected_count[str(dimension)] = values
                dimension += 1
            if 'lambd' in key:
                M_expected_count['lambda'] = values

                
        self.sparse = dict()
        self.sparse['coords'] = coords
        self.sparse['nnz_count'] = nnz_count
        self.sparse['M_init'] = Uinit
        self.sparse['M_expected_count'] = M_expected_count
        
        # Initilize pyGCP
        self.model = CP_ALS(n_iters=100, verbose=True)
        
        
    def take_norm_diff_factor(self, decomposition, d, M_type):
        """Helper function to take norm difference between two factors."""
        
        pred_di = decomposition['Factors'][str(d)]
        expected_di = self.sparse[M_type][str(d)]
        norm_diff_di = np.linalg.norm(pred_di - expected_di)

        return norm_diff_di
    
    
    def test_latent_factors_count(self):
        """Make sure the resulting latent factors are as expected for count tensor."""
        
        decomposition = self.cp_apr.train(coords=self.sparse['coords'], 
                                          values=self.sparse['nnz_count'], 
                                          rank=2, Minit=self.sparse['M_init'])
        
        # Check each latent factor
        for d in range(len(self.sparse['coords'][0])):
            
            norm_diff_di = self.take_norm_diff_factor(decomposition, d, 'M_expected_count')
      
            # check if norm of difference is very small
            self.assertEqual(True ,(np.abs(norm_diff_di) < 0.0000001))
            
            
        # Compare the weights
        norm_diff_lambd = self.take_norm_diff_weights(decomposition, 'M_expected_count')
        
        # check if norm of difference is very small
        self.assertEqual(True ,(np.abs(norm_diff_lambd) < 0.0000001))


            
    def test_latent_factors_count(self):
        """Make sure the resulting latent factors are as expected for count tensor."""
        
        decomposition = self.model.train(coords=self.sparse['coords'], 
                                       values=self.sparse['nnz_count'], 
                                       rank=2)
        # TODO: actualtest
        self.assertIn("Factors", decomposition)
        self.assertIn("Weights", decomposition)
