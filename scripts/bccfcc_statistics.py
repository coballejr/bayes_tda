import numpy as np
import matplotlib.pyplot as plt
from bayes_tda.data import _remove_padding
from bayes_tda.data import LabeledPDs
from bayes_tda.intensities import RGaussianMixture, Posterior

DATA_PATH = '/home/chris/projects/bayes_tda/data/'
DATA = 'bccfcc.npy'
LABELS = 'bccfcc_labels.npy'

if __name__ == '__main__':
    
    # load data
    data = np.load(DATA_PATH + DATA)
    labels = np.load(DATA_PATH + LABELS)
    data, labels = _remove_padding(data, labels)
    
    # make small dataset for demos
    data_small = data[0:1000]
    labels_small = labels[0:1000]
    
    np.save(DATA_PATH + 'bccfcc_small.npy', data_small, allow_pickle = True)
    np.save(DATA_PATH + 'bccfcc_small_labels.npy', labels_small, allow_pickle = True)
    
    
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
    
    # view compute some posteriors
    prior_mus = np.array([[3, 3]])
    prior_sigmas = np.array([10])
    prior_weights = np.array([5])
    prior = RGaussianMixture(prior_mus, prior_sigmas, prior_weights)
    
    clutter_mus = np.array([[0.0, 0.0]])
    clutter_sigmas = np.array([0.1])
    clutter_weights = np.array([100])
    clutter = RGaussianMixture(clutter_mus, clutter_sigmas, clutter_weights)
    
    sigma_DYO = 0.5
    
    bcc, fcc = grouped_data[0][20], grouped_data[1][20]
    post_bcc = Posterior(DYO = [bcc], prior = prior, clutter = clutter, sigma_DYO = sigma_DYO)
    post_fcc = Posterior(DYO = [fcc], prior = prior, clutter = clutter, sigma_DYO = sigma_DYO)
    
    linear_grid = np.linspace(0, 5.5, 20)
    post_bcc.show_lambda_DYO(linear_grid, show_means = False)
    post_fcc.show_lambda_DYO(linear_grid, show_means = False)
    
    
    
        