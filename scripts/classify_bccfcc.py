import numpy as np
from bayes_tda.intensities import RGaussianMixture
from bayes_tda.classifiers import EmpBayesFactorClassifier as EBFC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

DATA_PATH = '/home/chris/projects/bayes_tda/data/'
DATA = 'bccfcc_small.npy'
LABELS = 'bccfcc_small_labels.npy'

PRIOR_MUS = np.array([[5,5]])

PRIOR_SIGMAS = np.array([20])

PRIOR_WEIGHTS = np.array([1])

CLUTTER_MUS = np.array([[0, 0],
                      [0.1, 0.02]])

CLUTTER_SIGMAS = np.array([1,
                         1])

CLUTTER_WEIGHTS = np.array([0,
                          0])

SIGMA_DYO = 0.1

PRIOR_PROP = 0.2
HDIM = 1

#BCC_INDS = 

if __name__ == '__main__':
    
    # load data
    data = np.load(DATA_PATH + DATA, allow_pickle = True)
    labels = np.load(DATA_PATH + LABELS, allow_pickle = True)
    
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
                      data_type= 'diagrams')
    
    scores = classifier.compute_scores(clutter, 
                                       prior,
                                       prior_prop = PRIOR_PROP,
                                       sigma_DYO = SIGMA_DYO)
    
    # examine score distributions
    scores_0 = scores[0]
    scores_1 = scores[1]
    
    scores_0 = scores_0[:, 0] - scores_0[:, 1]
    scores_1 = scores_1[:, 0] - scores_1[:, 1]
    
    plt.hist(scores_0, label = 'BCC', color = 'blue', alpha = 0.5)
    plt.hist(scores_1, label = 'FCC', color = 'red', alpha = 0.5)
    plt.xlabel('$log p(BCC) / p(FCC) $')
    plt.ylabel('Count')
    plt.title('Bayes Factor Distributions')
    plt.legend()
    plt.show()
    plt.close()
    
    # compute aucs
    y0 = np.zeros(len(scores_0))
    y1 = np.ones(len(scores_1))
    
    y_true = np.concatenate([y0, y1])
    y_score = np.concatenate([scores_0, scores_1])
 
    tpr, fpr, _ = roc_curve(y_true, y_score)
    AUC = auc(fpr,tpr)
    print('AUC: ' + str(AUC))
    
    