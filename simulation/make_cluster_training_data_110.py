# This is based on Jacob Madsen's cluster_simulation.ipynb file.

from ase.visualize import view
import numpy as np
from pyqstem.util import atoms_plot
from labels import project_positions,create_label
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from pyqstem import PyQSTEM
from pyqstem.imaging import CTF
#from IPython.display import clear_output
from random_cluster import RandomCluster
from util import discrete_cmap
from ase.io import write
import os
import time

# Parameters
N=424
#sampling=24.48/2048*10 # 0.1195
sampling = 0.1
L=sampling*N
num_examples=1000   # Training
#num_examples=146   # Catch up
#num_examples=500     # Test
num_classes=1

dir_name='../data/cluster-110-single-class-2deg5'

# Generate test set?
if False:
    dir_name += '-test'

first_number=0

# Make sure working folders exist
if not os.path.exists(dir_name):
    print("Creating folder", dir_name)
    for subf in ['label', 'wave', 'model', 'points']:
        os.makedirs(os.path.join(dir_name, subf))

# Create the main object for making clusters
rc=RandomCluster(4.08,14)   # Parameters are lattice constant and grid size
qstem=PyQSTEM('TEM')

# Create the models
before = time.time()
for i in range(num_examples):
    radius=5+np.random.rand()*9
    
    lengths100=np.random.uniform(radius,radius+.2*radius,6)
    lengths111=np.random.uniform(radius-.2*radius,radius,8)

    rc.create_seed(lengths100,lengths111)
    rc.build(int(np.sum(rc.active)/4.),10,2)
    rc.build(int(np.sum(rc.active)/4.),2,2)

    atoms=rc.get_cluster('Au')
    
    atoms.rotate(v='y',a=45.0)

    # Find the atomic columns
    positions=atoms.get_positions()
    clusters = fcluster(linkage(positions[:,:2]), 1, criterion='distance')
    unique,indices=np.unique(clusters, return_index=True)
    
    atoms.rotate(v='z',a=np.random.random()*360)

    omega=np.random.random()*360
    alpha=np.random.random()*2.5

    atoms.rotate(v='z',a=omega,center='COP')
    atoms.rotate(v='y',a=alpha,center='COP')
    atoms.rotate(v='z',a=-omega,center='COP')

    atoms.center(vacuum=0)
    size=np.diag(atoms.get_cell())

    atoms.set_cell((L,)*3)
    atoms.center()

    tx=(L-size[0]-5)*(np.random.rand()-.5)
    ty=(L-size[1]-5)*(np.random.rand()-.5)

    atoms.translate((tx,ty,0))

    # Find the positions of the columns and the number of atoms per column.
    positions=atoms.get_positions()[:,:2]
    sites = np.array([np.mean(positions[clusters==u],axis=0) for u in unique])
    heights = np.array([np.sum(clusters==u) for u in unique])
    
    #positions,counts=project_positions(atoms,distance=.8,return_counts=True)
    
    model = atoms

    wave_size=(int(model.get_cell()[0,0]*10),int(model.get_cell()[1,1]*10))
    qstem.set_atoms(model)
    qstem.build_wave('plane',300,wave_size)
    qstem.build_potential(int(model.get_cell()[2,2]*2))
    qstem.run()
    wave=qstem.get_wave()

    wave.array=wave.array.astype(np.complex64)
    
    positions,counts=project_positions(model,distance=.8,return_counts=True)
    label=create_label(positions/sampling,(N,)*2,width=12,num_classes=num_classes,null_class=False)

    np.savez('{0}/points/points_{1:04d}.npz'.format(dir_name,first_number+i), sites=sites, heights=heights)
    np.save('{0}/label/label_{1:04d}.npy'.format(dir_name,first_number+i),label)
    wave.save('{0}/wave/wave_{1:04d}.npz'.format(dir_name,first_number+i))
    write('{0}/model/model_{1:04d}.cfg'.format(dir_name,first_number+i),model)

    print('iteration', i, flush=True)

print("Time to simulate models: {:.2f} s.".format(time.time() - before))
