import unittest
from bayes_tda.intensities import RestrictedGaussian, RGaussianMixture
import numpy as np
import matplotlib.pyplot as plt


# unit tests

class TestRestrictedGaussian(unittest.TestCase):
    
    def test_compute_normalizing_constants(self):
        mu = np.array([0, 0])
        sigma = 1
        min_birth_inf = -np.inf
        
        rg1 = RestrictedGaussian(mu, sigma)
        rg2 = RestrictedGaussian(mu, sigma, min_birth = min_birth_inf)
        rg3 = RestrictedGaussian(mu, sigma, fastQ = True)
        
        self.assertEqual(rg1.Q, 0.25)
        self.assertEqual(rg2.Q, 0.5)
        self.assertEqual(rg3.Q, 1)
        
    def test_evaluate(self):
        mu = np.array([1, 5])
        sigma = 1
        rg = RestrictedGaussian(mu, sigma)
        
        pt1 = np.array([1,-1])
        pt2 = np.array([-1, 1])
        pt3 = mu
        pt4 = np.array([0,0])
        
        eval1 = rg.evaluate(pt1)
        eval2 = rg.evaluate(pt2)
        eval3 = rg.evaluate(pt3)
        eval4 = rg.evaluate(pt4)
        
        pt3_target = (1 / rg.Q)*(1 / (2*np.pi))
        pt4_target = (1 / rg.Q)*(1 / (2*np.pi))*np.exp(-0.5*(26))
        pt3_err = (pt3_target - eval3)**2
        pt4_err = (pt4_target - eval4)**2
        
        self.assertEqual(eval1, 0)
        self.assertEqual(eval2, 0)
        self.assertLess(pt3_err, 1e-7)
        self.assertLess(pt4_err, 1e-7)
        
class TestRGaussianMixture(unittest.TestCase):
    
    def test_compute_normalizing_constants(self):
        mus = np.zeros((2,2))
        sigmas = np.ones(2)
        weights = np.ones(2)
        
        rgm = RGaussianMixture(mus, sigmas, weights)
        
        self.assertEqual(rgm.Qs[0], 0.25)
        self.assertEqual(rgm.Qs[1], 0.25)
        
    def test_compute_mask(self):
        x = np.array([[-1, 5], [0, 5], [1, 5], [2, 5], [2, -3]])
        
        mus = np.zeros((2,2))
        sigmas = np.ones(2)
        weights = np.ones(2)
        
        rgm = RGaussianMixture(mus, sigmas, weights)
        mask = rgm._compute_mask(x)
        ans = np.array([0, 1, 1, 1, 0])
        
        self.assertEqual(mask.tolist(), ans.tolist())
        
    def test_evaluate(self):
        mus = np.zeros((2,2))
        sigmas = np.ones(2)
        weights = np.ones(2)
        
        rgm = RGaussianMixture(mus, sigmas, weights)
        
        x1 = np.array([1, 1])
        x2 = np.array([[1, 1], [-1, 1], [2, -3]])
        
        d1 = rgm.evaluate(x1)
        d2 = rgm.evaluate(x2)
        
        ans1 = [(1 / (2 * np.pi)) * np.exp(-1)]
        ans2 = [(1 / (2 * np.pi)) * np.exp(-1), 0, 0]
        
        self.assertEqual(d1.tolist(), ans1)
        self.assertEqual(d2.tolist(), ans2)

        
if __name__ == '__main__':
    unittest.main()