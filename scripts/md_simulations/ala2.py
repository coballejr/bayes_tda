from openmm.app import *
from openmm import *
from openmm.unit import *
import mdtraj
from sys import stdout
import os
import numpy as np

# set paths
data_dir   =  '/home/chris/projects/bayes_tda/data/' # directory for output np.array
og_pdb_dir = '/home/chris/projects/bayes_tda/scripts/md_simulations/' # set to directory containing ala2_adopted.pdb

og_pdb_file = 'ala2_adopted.pdb'
working_dir = og_pdb_dir
os.chdir(working_dir)

print('Creating temporary file for full trajectory...')
tmp_file = 'ala2.pdb'

# define system 
temp = 330
fric_coeff = 0.5
dt = 0.001

# md simulation
print('Simulating full trajectory...')
pdb = PDBFile(og_pdb_file)
forcefield = ForceField('amber96.xml','amber96_obc.xml')
system = forcefield.createSystem(pdb.topology, constraints = HBonds ,energy = True)
system.addForce(AndersenThermostat(temp*kelvin, fric_coeff/picosecond)) 
integrator = LangevinIntegrator(temp*kelvin , fric_coeff/picosecond , dt*picosecond)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter(tmp_file, 10000))
simulation.reporters.append(StateDataReporter(stdout, 10000, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(10000000)

# pdb -> np.arrays
print('Converting pdb format to np array...')
traj = mdtraj.load(tmp_file)
xyz = traj.xyz

print('Saving trajectory as np array...')
out_file = 'ala2.npy'
np.save(data_dir + out_file, xyz) 

print('Removing temporary pdb file ...')
os.remove(tmp_file)

print('Done.')        
