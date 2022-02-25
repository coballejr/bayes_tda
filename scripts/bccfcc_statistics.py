import numpy as np
import matplotlib.pyplot as plt
from bayes_tda.data import LabeledPDs

DATA_PATH = '/home/chris/projects/bayes_tda/data/'
DATA = 'bccfcc.npy'
LABELS = 'bccfcc_labels.npy'

if __name__ == '__main__':
    
    # load data
    data = np.load(DATA_PATH + DATA)
    labels = np.load(DATA_PATH + LABELS)
    
    # create dgms
    labeled_data = LabeledPDs(data, labels)
    grouped_data = labeled_data.grouped_data
    
    
    # plot all diagrams in group
    fig, ax = plt.subplots(1, 2)
    for k in grouped_data.keys():
        dgms = grouped_data[k]
        features = np.vstack(dgms)
        ax[k].scatter(features[:, 0], features[:, 1], s = 1)
        ax[k].set_title('Class ' + str(k))
    
    plt.show()
    plt.close()
    
    # plot histgram of birth / persistence
    fig, ax = plt.subplots(1, 2)
    for k in grouped_data.keys():
        dgms = grouped_data[k]
        features = np.vstack(dgms)
        ax[k].hist(features[:, 1])
        ax[k].set_title('Class ' + str(k))
    
    plt.show()
    plt.close()
    
    # plot individual PDs
    fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
    for k in grouped_data.keys():
        dgms = grouped_data[k]
        dgm = dgms[2]
        ax[k].scatter(dgm[:, 0], dgm[:, 1], s = 1)
        ax[k].set_title('Class ' + str(k))
    
    plt.tight_layout()
    plt.show()
    plt.close()
        