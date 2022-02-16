import numpy as np
from bayes_tda.intensities import RGaussianMixture

mus = np.array([[1, 2], [4, 4]])
sigmas = np.array([1, 0.5])
weights = np.array([5, 1])

rgm = RGaussianMixture(mus, sigmas, weights)

linear_grid = np.linspace(0, 6, 20)
rgm.show_density(linear_grid)
