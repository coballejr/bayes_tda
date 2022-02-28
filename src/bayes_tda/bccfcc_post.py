import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def _show_bayes_factor_distribution(scores):
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
    
    return AUC