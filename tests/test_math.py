import unittest
from bayes_tda.math import _vectorized_mahalanobis_distance2D
import numpy as np

class TestFunctions(unittest.TestCase):
    
    def test_vectorized_mahalanobis_distance2D(self):
        X = np.array([[0, 0], [1,1]])
        U = np.array([[0, 0],[1, 1],[2, 2]])
        P = np.array([0.5*np.eye(2), 0.1*np.eye(2), 0.2*np.eye(2)])
        
        D = _vectorized_mahalanobis_distance2D(X, U, P)
        ans = np.array([[0, 0.2, 1.6],
                        [1, 0, 0.4]])
        
        self.assertEqual(D.tolist(), ans.tolist())
        

if __name__ == '__main__':
    unittest.main()

