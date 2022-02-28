import numpy as np
from bayes_tda.intensities import RGaussianMixture
from bayes_tda.classifiers import EmpBayesFactorClassifier as EBFC
import matplotlib.pyplot as plt
from bayes_tda.data import _remove_padding

DATA_PATH = '/home/chris/projects/bayes_tda/data/'
DATA = 'ala2.npy'
LABELS = 'ala2_labels.npy'

PRIOR_MUS = np.array([[0.22, 0.05]])

PRIOR_SIGMAS = np.array([10])

PRIOR_WEIGHTS = np.array([1])

CLUTTER_MUS = np.array([[0, 0],
                      [0.1, 0.02]])

CLUTTER_SIGMAS = np.array([0.1,
                         0.5])

CLUTTER_WEIGHTS = np.array([10,
                          1])

SIGMA_DYO = 0.1

PRIOR_PROP = 0.2
HDIM = 1



if __name__ == '__main__':
    
    # load data
    data = np.load(DATA_PATH + DATA)
    labels = np.load(DATA_PATH + LABELS)
    data, labels = _remove_padding(data, labels)
    data, labels = data[0:200], labels[0:200]
    
    # set hyperparameters
    prior = RGaussianMixture( mus = PRIOR_MUS, 
                             sigmas = PRIOR_SIGMAS, 
                             weights = PRIOR_WEIGHTS, 
                             normalize_weights= False)
    
    clutter = RGaussianMixture(mus = PRIOR_MUS, 
                             sigmas = PRIOR_SIGMAS, 
                             weights = PRIOR_WEIGHTS, 
                             normalize_weights= False)
    
    # build classifier
    classifier = EBFC(data = data,
                      labels = labels,
                      hdim = HDIM)
    
    scores = classifier.compute_scores(clutter, 
                                       prior,
                                       prior_prop = PRIOR_PROP,
                                       sigma_DYO = SIGMA_DYO)
    
    # examine score distributions
    scores_0 = np.exp(scores[0])
    scores_1 = np.exp(scores[1])
    
    plt.hist(scores_0[:, 0] / scores_0[:, 1], label = 'beta', color = 'blue', alpha = 0.5)
    plt.hist(scores_1[:, 0] / scores_1[:, 1], label = 'alpha', color = 'red', alpha = 0.5)
    plt.show()
    plt.close()
    