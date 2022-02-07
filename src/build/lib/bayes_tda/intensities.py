import numpy as np
from scipy.stats import multivariate_normal as mvn

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
        fastQ: TYPE, optional
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
                
            
            