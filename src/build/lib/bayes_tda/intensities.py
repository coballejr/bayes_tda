import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from bayes_tda.math import _vectorized_mahalanobis_distance2D, _cartesian_product

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
    
    def __init__(self, mus, sigmas, weights, normalize_weights = True, tilted = True, min_birth = 0, fastQ = False):
        '''
        Parameters
        ----------
        mus : array-like, shape = (num_means, 2).
              Means of each Gaussian component in mixture.
        sigmas : array_like, shape = (num_means,).
                 Diagonal entries in covariance matrices of each Gaussian component in mixture. 
        weights: array-like, shape = (num_means,).
                Non-negative weights of each Gaussian component in mixture. Can be unnormalized. 
        normalize_weights: bool, optional.
                           Boolean to specify whether to normalize component weights. The default is True.
        tilted : bool, optional
                Boolean to specify whether to use tilted coordinates. The default is True.
        min_birth : TYPE, optional
                    float that specifies minimum allowable birth time, e.g, this value
                    should be set to 0 when used with diagrams created from Rips filtrations. 
                    The default is 0.
        fastQ: bool, optional
               Boolean to specify whether to approximate normalizing constants Q
               with 1. The default is False.

        Returns
        -------
        None.

        '''
        
        self.mus = mus
        self.sigmas = sigmas
        self.weights = weights / weights.sum() if normalize_weights else weights
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
        
        P = np.array([(1 / sigma) * np.eye(2) for sigma in self.sigmas])
        dists = _vectorized_mahalanobis_distance2D(X = x, U = self.mus, P = P)
        
        gaussian_normalizing_consts = 1 / (2 * np.pi * (self.sigmas))
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
        cb.set_label('Intensity')
        
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
        

class Posterior:
    '''
    Posterior intensity class for persistence diagrams.
    '''
        
    def __init__(self, DYO, prior, clutter, sigma_DYO, alpha = 1, min_birth = 0):
        '''
        

        Parameters
        ----------
        DYO : list or tuple of np.arrays
             Observed persistence diagrams.
        prior : RGaussianMixture object.
                Prior intensity.
        clutter : RGaussianMixture object.
                  Intensity for clutter/ spurious features.
        sigma_DYO : float.
                    \sigma^{D_{Y_O}} parameter in posterior intensity.
        alpha : float, optional
                Alpha parameter in posterior intensity. The default is 1.
        min_birth: float, optional
                   Minimum birth times for pd filtration. Should be 0 for Rips
                   and -inf for cubical. The default is 0. 

        Returns
        -------
        None.

        '''
        
        self.DYO = DYO
        self.num_obs_dgms = len(DYO)
        self.sigma_DYO = sigma_DYO
        self.alpha = alpha
        self.prior = prior
        self.clutter = clutter
        self.min_birth = min_birth
        
        posterior_means, posterior_sigmas = self._compute_posterior_mus_and_sigmas()
        
        self.posterior_means = posterior_means
        self.posterior_sigmas = posterior_sigmas
        self.ws = self._compute_ws()
        self.Qs = self._compute_Qs(posterior_means, posterior_sigmas)
        self.Cs = self._compute_Cs(self.ws, self.Qs)
        self.lambda_DYO = RGaussianMixture(posterior_means, 
                                           posterior_sigmas,
                                           weights = self.Cs,
                                           normalize_weights= False,
                                           min_birth = min_birth)
        
        
    def _compute_posterior_mus_and_sigmas(self):
        '''
        Computes posterior means and standard deviations.
        
        Returns
        -------
        posterior_means: np.array, shape = (n_prior_components*n_obs_pd_features, 2).
        posterior_sigmas: np.array, shape = (n_prior_components*n_obs_pd_features,)
        '''
        
        Y = np.vstack(self.DYO)
        prior_means = self.prior.mus
        
        prior_means, Y_expanded = _cartesian_product(prior_means, Y)
        prior_sigmas = self.prior.sigmas.repeat(Y.shape[0]).reshape(-1, 1)
        
        posterior_means = (self.sigma_DYO*prior_means + prior_sigmas*Y_expanded) / (self.sigma_DYO + prior_sigmas)
        posterior_sigmas = (self.sigma_DYO*prior_sigmas) / (self.sigma_DYO + prior_sigmas)
        
        return posterior_means, posterior_sigmas.flatten()
    
    def _compute_ws(self):
        '''
        Computes w_{j}^{y} values in posterior intensity

        Returns
        -------
        w: np.array, shape = (n_prior_components*n_obs_pd_features,).
        '''
        
        Y = np.vstack(self.DYO)
        
        prior_means = self.prior.mus
        prior_means, Y_expanded = _cartesian_product(prior_means, Y)
        sigmas = self.prior.sigmas.repeat(Y.shape[0]) + self.sigma_DYO
        prior_weights = self.prior.weights.repeat(Y.shape[0])
        
        d_sq = ((Y_expanded - prior_means)**2).sum(axis = 1)
        dens = (1 / (2*np.pi*sigmas))*np.exp(-0.5*(d_sq / sigmas))
        w = prior_weights*dens
        
        return w
    
    def _compute_Qs(self, posterior_means, posterior_sigmas):
        '''
        Computes Q_{j}^{y} in posterior intensity.

        Parameters
        ----------
        posterior_means: np.array, shape = (n_prior_components*n_obs_pd_features, 2).
        posterior_sigmas: np.array, shape = (n_prior_components*n_obs_pd_features,)

        Returns
        -------
        np.array, shape = (n_prior_components*n_obs_pd_features,).
        '''
        
        tmp_rgm = RGaussianMixture(mus = posterior_means, sigmas = posterior_sigmas, 
                                   weights = np.ones(posterior_means.shape[0]),
                                   min_birth = self.min_birth)
        
        return tmp_rgm.Qs
    
    def _compute_Cs(self, w, Q):
        '''
        Computes C_{j}^{y} in posterior intensity.

        Parameters
        ----------
        w : np.array, shape = (n_prior_components*n_obs_pd_features,).
        Q : np.array, shape = (n_prior_components*n_obs_pd_features,).

        Returns
        -------
        C: np.array, shape = (n_prior_components*n_obs_pd_features,).
        '''
        
        Y = np.vstack(self.DYO)
        _, Y_expanded = _cartesian_product(self.prior.mus, Y)
        
        n_prior_components = self.prior.mus.shape[0]
        n_obs_pd_features = Y.shape[0]
        
        wQ = (w*Q).reshape((n_prior_components, n_obs_pd_features))
        swQ = wQ.sum(axis = 1)
        swQ = swQ.repeat(n_obs_pd_features)
        clutter = self.clutter.evaluate(Y_expanded)
        
        C = w / (clutter + self.alpha*swQ)
        
        return C
    
    def evaluate(self, x):
        '''
        Evaluate posterior intensity at points in array x.

        Parameters
        ----------
        x : np.array of birth-pers coordinates, shape = (num_points, 2).

        Returns
        -------
        intensities: np.array shape = (num_points, ).
        '''
        
        alpha = self.alpha
        prior = self.prior
        lambda_DYO = self.lambda_DYO
        m = self.num_obs_dgms
        
        intensities = (1 - alpha)*prior.evaluate(x) + (alpha/ m)*lambda_DYO.evaluate(x)
        
        return intensities
    
    def evaluate_dgm(self, dgm, log = True):
        '''
        Evaluate persistence diagram in posterior intensity.
        

        Parameters
        ----------
        dgm : np.array, shape = (n_features, 2)
              persistence diagram.
        log : bool, optional
              Specifies whether to return log intensity. The default is True.

        Returns
        -------
        intensity: float.
        '''
        
        intensity_pts = self.evaluate(dgm)
        

        intensity = np.log(intensity_pts).sum()
        
        return intensity if log else np.exp(intensity)
    
    def evaluate_dgms(self, dgms, log = True):
        '''
        Evaluate multiple persistence diagrams in posterior intensity.
        

        Parameters
        ----------
        dgms : list of np.arrays.
                list of persistence diagrams.
        log : bool, optional
              Specifies whether to return log intensity. The default is True.

        Returns
        -------
        intensities: np.array of intensites.

        '''
        
        intensities = np.zeros(len(dgms))
        
        for i, dgm in enumerate(dgms):
            intensities[i] = self.evaluate_dgm(dgm, log = log)
            
        return intensities
    
    def show_lambda_DYO(self, linear_grid, **kwargs):
        self.lambda_DYO.show_density(linear_grid, title = r'$\lambda_{D_{Y_O}}$', **kwargs)
        
    def show_prior(self, linear_grid, **kwargs):
        self.prior.show_density(linear_grid, title = 'Prior', **kwargs)
        
    def show_clutter(self, linear_grid, **kwargs):
        self.clutter.show_density(linear_grid, title = r'$\lambda_{D_{Y_S}}$', **kwargs)
        
        

# TO DO: unit tests for Posterior and RGaussian mixture visualization
    
if __name__ == '__main__':

    mus = np.array([[1, 2], [4, 4]])
    sigmas = np.array([1, 0.5])
    weights = np.array([5, 1])

    rgm = RGaussianMixture(mus, sigmas, weights)

    linear_grid = np.linspace(0, 6, 20)
    rgm.show_density(linear_grid)
    
    dgms = [d for d in 6*np.random.rand(100, 30, 2)]
    post = Posterior(DYO = dgms,
                     prior = rgm,
                     clutter = rgm,
                     sigma_DYO= 0.1)
    
    
    l_intensities = post.evaluate_dgms(dgms)
    intensities = post.evaluate_dgms(dgms, log = False)
    post.show_lambda_DYO(linear_grid, show_means = False)
    post.show_prior(linear_grid, show_means = False)
    post.show_clutter(linear_grid, show_means = False) 
    

        
        
    
    
    
                
            
            