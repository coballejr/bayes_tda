import numpy as np
from ripser import ripser
from gudhi import CubicalComplex

# TODO: refactor functions into methods of a class

# Post processing pds

def tilt_dgm(dgm):
    
    dgm[:, 1] = dgm[:, 1] - dgm[:, 0]
        
    return dgm

def drop_inf_features(dgm):
    
    pers = dgm[:, 1]
    finite_inds = np.isfinite(pers)

    return dgm[finite_inds, :]    


# PD computations with different filtrations

def compute_rips(data, hdim, finite_persistence = True):
    
    dgm = ripser(data, maxdim = hdim)['dgms'][hdim]
    
    if finite_persistence:
        dgm = drop_inf_features(dgm)
        
    return dgm

def compute_cubical(data, hdim, finite_persistence = True):
    
    # compute filtration
    filtration_values = data.flatten()
    
    st = CubicalComplex(dimensions = data.shape, 
                        top_dimensional_cells = filtration_values)
    
    st.compute_persistence()
    
    # compute pd from filtration
    dgm = st.persistence_intervals_in_dimension(hdim)
    
    if finite_persistence:
        dgm = drop_inf_features(dgm)
        
    return dgm


# General PD computation

def compute_dgms(data, filtration = 'rips', hdim = 1, finite_persistence = True):
    
    if filtration == 'rips':
        pd_func = compute_rips
    elif filtration == 'cubical':
        pd_func = compute_cubical
    else:
        raise NotImplementedError('Only rips and cubical filtrations are supported.')
    
    dgms = []    
    for X in data:
        dgm = pd_func(X, hdim = hdim, finite_persistence = finite_persistence)
        dgm = tilt_dgm(dgm)
        dgms.append(dgm)
        
    return dgms

# TODO: units tests for comp_topo

if __name__ == '__main__':
    
    pcs = np.random.randn(10, 100, 3)
    imgs = np.random.rand(10, 28, 28) 
    
    rips_dgms0 = compute_dgms(pcs, hdim = 0)
    rips_dgms1 = compute_dgms(pcs, hdim = 1)
    rips_dgms2 = compute_dgms(pcs, hdim = 2)
    
    cubical_dgms0 = compute_dgms(imgs, filtration = 'cubical', hdim = 0)
    cubical_dgms1 = compute_dgms(imgs, filtration = 'cubical', hdim = 1)

    
    
    
    