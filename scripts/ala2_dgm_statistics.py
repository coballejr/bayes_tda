import numpy as np
import matplotlib.pyplot as plt
from bayes_tda.data import LabeledPointClouds

DATA_PATH = '/home/chris/projects/bayes_tda/data/'
DATA = 'ala2.npy'
LABELS = 'ala2_labels.npy'

HDIM = 1

if __name__ == '__main__':
    
    # load data
    data = np.load(DATA_PATH + DATA)
    labels = np.load(DATA_PATH + LABELS)
    
    # create dgms
    labeled_data = LabeledPointClouds(data, labels, hdim = HDIM)
    grouped_dgms = labeled_data.grouped_dgms
    
    fig, ax = plt.subplots(1, 2)
    for k in grouped_dgms.keys():
        dgms = grouped_dgms[k]
        features = np.vstack(dgms)
        ax[k].scatter(features[:, 0], features[:, 1], s = 1)
        ax[k].set_title('Class ' + str(k))
    
    plt.show()
    plt.close()
    
    fig, ax = plt.subplots(1, 2)
    for k in grouped_dgms.keys():
        dgms = grouped_dgms[k]
        features = np.vstack(dgms)
        ax[k].hist(features[:, 1])
        ax[k].set_title('Class ' + str(k))
    
    plt.show()
    plt.close()
    
    