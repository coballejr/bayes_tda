import numpy as np
import matplotlib.pyplot as plt
from bayes_tda.data import LabeledPointClouds

DATA_PATH = '/home/chris/projects/bayes_tda/data/'
DATA = 'ala2.npy'
LABELS = 'ala2_labels.npy'

DATA_SCALE = 10
HEAVY_ATOM_INDS = [1, 4, 6, 8, 14, 16, 18]
HDIM = 1

if __name__ == '__main__':
    
    # load data
    data = DATA_SCALE*np.load(DATA_PATH + DATA)
    #data = data[:, HEAVY_ATOM_INDS, :]
    labels = np.load(DATA_PATH + LABELS)
    
    # create dgms
    labeled_data = LabeledPointClouds(data, labels, hdim = HDIM)
    grouped_dgms = labeled_data.grouped_dgms
    
    
    # plot all diagrams in group
    fig, ax = plt.subplots(1, 2)
    for k in grouped_dgms.keys():
        dgms = grouped_dgms[k]
        features = np.vstack(dgms)
        ax[k].scatter(features[:, 0], features[:, 1], s = 1)
        ax[k].set_title('Class ' + str(k))
    
    plt.show()
    plt.close()
    
    # plot histgram of birth / persistence
    fig, ax = plt.subplots(1, 2)
    for k in grouped_dgms.keys():
        dgms = grouped_dgms[k]
        features = np.vstack(dgms)
        ax[k].hist(features[:, 1])
        ax[k].set_title('Class ' + str(k))
    
    plt.show()
    plt.close()
    
    # plot individual PDs
    fig, ax = plt.subplots(1, 2)
    for k in grouped_dgms.keys():
        dgms = grouped_dgms[k]
        dgm = dgms[1]
        ax[k].scatter(dgm[:, 0], dgm[:, 1], s = 1)
        ax[k].set_title('Class ' + str(k))
    
    plt.tight_layout()
    plt.show()
    plt.close()
        
    
    