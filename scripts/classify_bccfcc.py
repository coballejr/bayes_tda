import numpy as np
from bayes_tda.intensities import RGaussianMixture
from bayes_tda.classifiers import EmpBayesFactorClassifier as EBFC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

DATA_PATH = '/home/chris/projects/bayes_tda/data/'
DATA = 'bccfcc.npy'
LABELS = 'bccfcc_labels.npy'

PRIOR_MUS = np.array([[4,5],
                      [3, 3]])

PRIOR_SIGMAS = np.array([1,
                         10])

PRIOR_WEIGHTS = np.array([3,
                          3])

CLUTTER_MUS = np.array([[0, 0],
                      [0.1, 0.02]])

CLUTTER_SIGMAS = np.array([1,
                         1])

CLUTTER_WEIGHTS = np.array([3,
                          3])

SIGMA_DYO = 0.1

PRIOR_PROP = 0.5
HDIM = 1



if __name__ == '__main__':
    
    # load data
    data = np.load(DATA_PATH + DATA)
    labels = np.load(DATA_PATH + LABELS)
    
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
    
    scores_0 = scores_0[:, 1]/ scores_0[:, 0]
    scores_1 = scores_1[:, 1]/ scores_1[:, 0]
    
    plt.hist(scores_0, label = 'beta', color = 'blue', alpha = 0.5)
    plt.hist(scores_1, label = 'alpha', color = 'red', alpha = 0.5)
    plt.show()
    plt.close()
    
    # compute aucs
    y0 = np.zeros(len(scores_0))
    y1 = np.ones(len(scores_1))
    
    y_true = np.concatenate([y0, y1])
    y_score = np.concatenate([scores_0, scores_1])
    tpr, fpr, _ = roc_curve(y_true, y_score, pos_label = 1)
    AUC = auc(fpr,tpr)
    print('AUC: ' + str(AUC))
    
    