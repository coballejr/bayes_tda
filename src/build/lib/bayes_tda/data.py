import numpy as np
from comp_topo import compute_dgms

class LabeledData:
    '''
    Abstract class for labeled data.
    '''
    
    def __init__(self, data, labels):
        '''
        

        Parameters
        ----------
        data : array-like, shape = (n_examples, data_dim).
        labels : array-like, shape = (n_examples, ) or (n_examples, 1)
        '''
        
        self.data = data
        self.labels = np.array(labels).reshape(-1, 1)
        self._group_data()
        
        
    def _group_data(self):
        
        labels = self.labels.flatten()
        classes = np.unique(labels)
        grouped_data = {k: [] for k in classes}
        
        for k, x in zip(labels, self.data):
            grouped_data[k].append(x)
            
        self.grouped_data = grouped_data
    

class LabeledPointClouds(LabeledData):
    
    def __init__(self, point_clouds, labels, filtration = 'rips', hdim = 1):
        super(LabeledPointClouds, self).__init__(data = point_clouds, labels = labels)
        
        self.filtration = filtration
        self.hdim = hdim
        self._make_dgms()
        self._group_dgms()
        
    def _make_dgms(self):
        dgms = compute_dgms(self.data, filtration = self.filtration, hdim = self.hdim)
        self.dgms = dgms
    
    def _group_dgms(self):
        labels = self.labels.flatten()
        classes = np.unique(labels)
        grouped_dgms = {k: [] for k in classes}
        
        for k, x in zip(labels, self.dgms):
            grouped_dgms[k].append(x)
            
        self.grouped_dgms = grouped_dgms
        
class LabeledImages(LabeledData):
    
    def __init__(self, point_clouds, labels, filtration = 'cubical', hdim = 1):
        super(LabeledImages, self).__init__(data = point_clouds, labels = labels)
        
        self.filtration = filtration
        self.hdim = hdim
        self._make_dgms()
        self._group_dgms()
        
    def _make_dgms(self):
        dgms = compute_dgms(self.data, filtration = self.filtration, hdim = self.hdim)
        self.dgms = dgms
    
    def _group_dgms(self):
        labels = self.labels.flatten()
        classes = np.unique(labels)
        grouped_dgms = {k: [] for k in classes}
        
        for k, x in zip(labels, self.dgms):
            grouped_dgms[k].append(x)
            
        self.grouped_dgms = grouped_dgms
        
class LabeledPDs(LabeledData):
    
    def __init__(self, dgms, labels):
        super(LabeledPDs, self).__init__(data = dgms, labels = labels)
        
# TODO: unit tests for data

if __name__ == '__main__':
    
    pcs = np.random.randn(10, 100, 3)
    imgs = np.random.rand(10, 28, 28)
    labels = np.random.choice([0, 1], 10, replace = True)
    
    pc_data = LabeledPointClouds(pcs, labels, hdim = 1)
    img_data = LabeledImages(imgs, labels, hdim = 1)
        