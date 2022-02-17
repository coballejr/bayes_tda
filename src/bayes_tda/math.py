import numpy as np

def _vectorized_mahalanobis_distance2D(X, U, P):
    '''
    Vectorized computation of two-dimensional Mahalanobis distance. Used to 
    evaluate pdf of Gaussian mixtures. 

    Parameters
    ----------
    X : np.array, shape = (num_points, 2).
        Coordinates of points at which to evaluate density.
    U : np.array, shape = (num_means, 2).
        Coordinates of means in Gaussian mixture.
    P : np.array, shape = (num_means, 2, 2).
        Precision matrices of components in Gaussian mixture.

    Returns
    -------
    D: np.array, shape = (num_points, num_means).
    D_{ij} = squared Mahalanobis distance between X[i] and U[j] using precision
             matrix P[j]
    '''
    
    diff = X[np.newaxis, :, :] - U[:, np.newaxis, :]
    D = np.einsum('jik,jkl,jil->ij', diff, P, diff)
    
    return D

def _cartesian_product(U, Y):
    '''
    Expand two arrays for vectorized computations with their Cartesian product.

    Parameters
    ----------
    U : np.array, shape = (m, 2).
    Y : np.array, shape = (n, 2)

    Returns
    -------
    U_expanded: np.array, shape = (m*n, 2).
    Y_expanded: np.array, shape = (m*n, 2)

    '''
    
    m, n = U.shape[0], Y.shape[0]
    U_expanded = np.repeat(U, n, axis = 0)
    Y_expanded = np.tile(Y, reps = (m, 1))
    
    return U_expanded, Y_expanded
    