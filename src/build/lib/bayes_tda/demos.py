import numpy as np
import matplotlib.pyplot as plt
from bayes_tda.data import LabeledPointClouds

class Circle:
    
    def __init__(self, n_pts = 100, noise_level = 0.1, seed = 0):
        
        np.random.seed(seed)
        
        t = np.linspace(0, 1, n_pts)
        x,y = np.cos(2*np.pi*t), np.sin(2*np.pi*t)
        self.circ = np.array([[h,v] for h,v in zip(x,y)])
        
        noise = np.random.normal(scale = noise_level, size = [n_pts,2])
        self.n_circ = self.circ+noise
        
        label = 1 # dummy label
        data = LabeledPointClouds([self.n_circ], np.array(label))
        self.dgm = data.grouped_dgms[label][0]
    
    def show(self):
        
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(self.circ[:,0],self.circ[:,1], color = 'red') # circle
        ax[0].scatter(self.n_circ[:,0], self.n_circ[:,1], color = 'black') # noisy sample
        ax[0].set_title('Noisy Circle')
        ax[0].set_xlim([-1.5,1.5])
        ax[0].set_ylim([-1.5,1.5])
        ax[0].set_aspect('equal')
        
        
        birth, persistence = self.dgm[:, 0], self.dgm[:, 1]
        ax[1].scatter(birth, persistence, s = 8, color = 'black')
        ax[1].set_title('Noisy Circle PD')
        ax[1].set_xlabel('Birth')
        ax[1].set_ylabel('Persistence')
        ax[1].set_xlim([0, 1.5])
        ax[1].set_ylim([0, 1.5])
        ax[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        plt.close()