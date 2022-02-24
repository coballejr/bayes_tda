import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

def cluster_by_phi(phi, lower_lim = -2.5, upper_lim = 0.5):
    return np.where((lower_lim < phi) * (phi < upper_lim), 1, 0)
    

if __name__ == '__main__':
    
    # load data
    xyz = np.load('/home/chris/projects/bayes_tda/data/ala2.npy')
    pdb = md.load_pdb('/home/chris/projects/bayes_tda/scripts/md_simulations/ala2_adopted.pdb')
    traj = md.Trajectory(xyz = xyz, topology = pdb.topology)
    
    # compute dihedral angles
    psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
    angles = md.compute_dihedrals(traj, [phi_indices, psi_indices])
    psi, phi = angles[:, 0], angles[:, 1]
    
    # plot dihedrals
    plt.scatter(psi, phi, s = 1)
    plt.show()
    plt.close()
    
    # plot phi histogram
    plt.hist(angles[:,1])
    plt.show()
    plt.close()
    
    # compute labels
    labels = cluster_by_phi(phi)
    
    # plot dihedrals with labels
    plt.scatter(psi, phi, s = 1, c = labels)
    plt.show()
    plt.close()
    
    
    np.save('/home/chris/projects/bayes_tda/data/ala2_labels.npy', labels)
    
    
    
