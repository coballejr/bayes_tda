import numpy as np
from bayes_tda.data import LabeledPointClouds, LabeledImages, LabeledPDs
from bayes_tda.intensities import Posterior

class EmpBayesFactorClassifier:
    
    def __init__(self, data, labels, data_type = 'point_cloud', **kwargs):
        
        if data_type == 'point_cloud':
            data_handler = LabeledPointClouds
        elif data_type == 'images':
            data_handler = LabeledImages
        elif data_type == 'diagrams':
            data_handler = LabeledPDs
            
        self.labeled_data = data_handler(data, labels, **kwargs)
        
        has_dgms = hasattr(self.labeled_data,'grouped_dgms')
        self.grouped_dgms = self.labeled_data.grouped_dgms if has_dgms else self.labeled_data.grouped_data
        
    def compute_scores(self, 
                       clutter,
                       prior,
                       sigma_DYO,
                       prior_prop = 0.5,
                       alpha = 1, 
                       min_birth = 0,
                       log = True):
        
        grouped_dgms = self.grouped_dgms
        
        # partition training and test sets
        train = {}
        test = {}
        for k in grouped_dgms.keys():
            
            dgms = grouped_dgms[k]
            
            idx = np.arange(len(dgms))
            np.random.shuffle(idx)
            train_cut = int(prior_prop*len(dgms))
            train_idx, test_idx = [i for i in idx[:train_cut]], [i for i in idx[train_cut:]]
            
            train[k] = [dgms[i] for i in train_idx]
            test[k] = [dgms[i] for i in test_idx]
            
        # compute posteriors
        posteriors = {}
        for k in train.keys():
            posteriors[k] = Posterior(DYO = train[k], 
                                      prior = prior, 
                                      clutter = clutter, 
                                      sigma_DYO = sigma_DYO, 
                                      alpha = alpha)
            
        # compute scores
        scores_dict = {}
        for k in test.keys():
            test_dgms = test[k]
            scores = np.zeros((len(test_dgms), len(posteriors)))
            
            for label, kk in enumerate(posteriors.keys()):
                posterior = posteriors[kk]
                scores_label = posterior.evaluate_dgms(test_dgms, log = log)
                scores[:, label] = scores_label
            
            scores_dict[k] = scores 
                
        return scores_dict
    
# TODO: unit tests for classifieres
            
            
            
        
        
        
        
