import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from bayes_tda.math import _vectorized_mahalanobis_distance2D

class RestrictedGaussian:
    '''
    Two-dimensional isotropic Gaussian density restricted to a subset of R^2.
    '''
    
    def __init__(self, mu, sigma, tilted = True, min_birth = 0, fastQ = False):
        '''
        Parameters
        ----------
        mu : array-like, shape = (2,). 
             Mean.
        sigma : float.
                Diagonal value of covariance matrix.
        tilted : TYPE, optional
                 Boolean to specify whether to use tilted coordinates. The default is True.
        min_birth : TYPE, optional
                    float that specifies minimum allowable birth time, e.g, this value
                    should be set to 0 when used with diagrams created from Rips filtrations. 
                    The default is 0.
        fastQ: bool, optional
               Boolean to specify whether to approximate normalizing constant Q
               with 1. The default is False.
               
        Returns
        -------
        None.
        '''
        
        self.mu = mu
        self.sigma = sigma
        self.tilted = tilted
        self.min_birth = min_birth
        self.fastQ = fastQ
        
        mean = np.array(mu)
        covariance = (sigma**2)*np.eye(2)
        self.dist = mvn(mean = mean, cov = covariance)
        
        self._compute_normalizing_constant()
        
    def _compute_normalizing_constant(self):
        '''
        Raises
        ------
        NotImplementedError
        Only implemented for titled coordinates.

        Assigns
        -------
        self.Q : float.
        Normalizing constant for Gaussian density. 
        '''
            
        if self.fastQ:
            Q = 1
                
        elif self.tilted and self.min_birth > -np.inf:
            left_half = self.dist.cdf(np.array([self.min_birth, np.inf]))
            lower_half = self.dist.cdf(np.array([np.inf, 0]))
            q3 = self.dist.cdf(np.array([self.min_birth, 0]))
                
            q4 = lower_half - q3
            right_half = 1 - left_half
                
            Q = right_half - q4
            
        elif self.tilted and self.min_birth == -np.inf:
            lower_half = self.dist.cdf(np.array([np.inf, 0]))
            Q = 1 - lower_half
                
        else:
            raise NotImplementedError('Q computation only implemented for tilted coordinates!')
            
        self.Q = Q
        
    def evaluate(self, pt):
        '''
        Evaluate density at a point.

        Parameters
        ----------
        pt : array-like, shape = (2,).
             Birth-persistence coordinates of point.
        
        Returns
        -------
        density: float.
        '''
        
        b, p = pt
        
        if p < 0 or b < self.min_birth:
            density = 0
            
        else:
            density = (1 / self.Q) * self.dist.pdf(pt)
            
        return density
    
class RGaussianMixture:
    '''
    Mixture of two-dimensional isotropic Gaussians restricted to a subset of R^2.
    '''
    
    def __init__(self, mus, sigmas, weights, tilted = True, min_birth = 0, fastQ = False):
        '''
        Parameters
        ----------
        mus : array-like, shape = (num_means, 2).
              Means of each Gaussian component in mixture.
        sigmas : array_like, shape = (num_means,).
                 Diagonal entries in covariance matrices of each Gaussian component in mixture. 
        weights: array-like, shape = (num_means,).
                Non-negative weights of each Gaussian component in mixture. Can be unnormalized. 
       tilted : TYPE, optional
                Boolean to specify whether to use tilted coordinates. The default is True.
        min_birth : TYPE, optional
                    float that specifies minimum allowable birth time, e.g, this value
                    should be set to 0 when used with diagrams created from Rips filtrations. 
                    The default is 0.
        fastQ: TYPE, optional
               Boolean to specify whether to approximate normalizing constants Q
               with 1. The default is False.

        Returns
        -------
        None.

        '''
        
        self.mus = mus
        self.sigmas = sigmas
        self.weights = weights / weights.sum()
        self.tilted = tilted
        self.min_birth = min_birth
        self.fastQ = fastQ
        self._compute_normalizing_constants()
        
    def _compute_normalizing_constants(self):
        '''
        Assigns
        -------
        self.Qs: np.array, shape = (num_means,).
                 Normalizing constants for each Gaussian component.
        '''
        
        Qs = np.zeros(self.mus.shape[0])
        
        for i, params in enumerate(zip(self.mus, self.sigmas)):
            mu, sigma = params
            component = RestrictedGaussian(mu, sigma, self.tilted, self.min_birth, self.fastQ)
            Qs[i] = component.Q
        
        self.Qs = Qs
        
    def _compute_mask(self, x) :
        '''
        Compute mask to zero out points in restricted Gaussian density evaluation.

        Parameters
        ----------
        x: np.array of birth-pers coordinates, shape = (num_points, 2)
        
        Returns
        -------
        mask: np.array, shape = (num_points, ).
              0 where birth coordinates are less than min birth or persistence is negative.

        '''
        
        num_points = x.shape[0]
        min_births = np.repeat(self.min_birth, num_points)
        
        b, p = x[:, 0], x[:, 1]
        mask1 = b >= min_births
        mask2 = p > np.zeros(num_points)
        mask = mask1*mask2
        
        return mask
        
    def evaluate(self, x):
        '''
        Evaluate density at points in array x.

        Parameters
        ----------
        x : np.array of birth-pers coordinates, shape = (num_points, 2).

        Returns
        -------
        densities : np.array shape = (num_points, ).

        '''
        
        x = np.atleast_2d(x)
        mask = self._compute_mask(x)
        
        P = np.array([(1 / (sigma ** 2)) * np.eye(2) for sigma in self.sigmas])
        dists = _vectorized_mahalanobis_distance2D(X = x, U = self.mus, P = P)
        
        gaussian_normalizing_consts = 1 / (2 * np.pi * (self.sigmas ** 2))
        exps = np.exp(-0.5*dists)
        gaussian_densities = exps*gaussian_normalizing_consts
        densities = (gaussian_densities*self.weights).sum(axis = 1)
        
        
        return densities*mask
    
    def show_density(self, linear_grid, title = 'Mixed Gaussian Density', show_means = True, plot_additional_pts = False, additional_pts = None):
        '''
        Plot pdf of Gaussian mixture. 

        Parameters
        ----------
        linear_grid : np.array, shape = (sqrt(grid_size), )
                      Cartesian product is taken to form grid.
        title : str, optional
                 Title of plot. The default is 'Mixed Gaussian Density'.
        show_means : bool, optional
                     Specifies whether to show means. The default is True.
        plot_additional_pts : bool, optional
                              Specifies whether to add additional points to plot. The default is False.
        additional_pts : None or np.array with shape = (num_additional_pts, 2), optional
                         Coordinates of additional points to add. The default is None.

        '''
        
        X, Y = np.meshgrid(linear_grid, linear_grid)
        XY = np.vstack([X.flatten(), Y.flatten()]).T
        Z = self.evaluate(XY)
        
        plt.contourf(X, Y, Z.reshape(X.shape))
        plt.xlabel('Birth')
        plt.ylabel('Persistence')
        
        cb = plt.colorbar()
        cb.set_label('Density')
        
        plt.title(title)
        
        if show_means:
            mu_b, mu_p = self.mus[:, 0], self.mus[:, 1]
            plt.scatter(mu_b, mu_p, label = 'Means')
            plt.legend()
            
        if plot_additional_pts and additional_pts:
            plt.scatter(additional_pts[:, 0], additional_pts[:, 1])
        
        
        plt.gca().set_aspect('equal')
        plt.show()
        plt.close()
        
    
    
    
                
            
            