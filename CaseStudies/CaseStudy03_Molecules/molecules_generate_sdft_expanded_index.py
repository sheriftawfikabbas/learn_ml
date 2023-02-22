import numpy as np
import importlib
import json
from ase.build import bulk
from ase.io import read
import pandas as pd
from sympy import symbols
import descriptors_rosa
from glob import glob
import sys

xyzs = glob('../Data/Molecules/*.xyz')

dataset = []
band_gaps = []
material_ids = []
    
for i in xyzs[0:5000]:
    xyzf = open(i, 'r')
    l = xyzf.readlines()
    n = int(l[0])
    positions = l[2:2+n]
    positions = np.array([x.split() for x in positions])
    pos = positions[:, 1:4]
    sym = positions[:, 0]
    l = l[1]
    l = l.split()
    gap = l[9]
    xyzf.close()
    try:
        xyz = Atoms(positions=pos, symbols=sym, pbc=False)
        xrange = xyz.get_positions()[:, 0].max()-xyz.get_positions()[:, 0].min()
        yrange = xyz.get_positions()[:, 0].max()-xyz.get_positions()[:, 0].min()
        zrange = xyz.get_positions()[:, 0].max()-xyz.get_positions()[:, 0].min()
        xyz.cell = [[10+xrange, 0, 0], [0, 10+yrange, 0], [0, 0, 10+zrange]]
        

        system_name = i.replace('Molecules/dsgdb9nsd.xyz.tar/', '')
        system_name = i.replace('.xyz', '')
        xyz.center()
    
        d = descriptors_rosa.descriptors(xyz)
        if len(d) > 0:
            dataset += [d]
            band_gaps += [gap]
            material_ids += [system_name]
           
        else:
            continue
    except:
        print('Problem in material: '+system_name+'\n')
        continue

dataset_df = pd.DataFrame(dataset)
full_dataset_df = pd.DataFrame(dataset)
full_dataset_df['band_gaps'] = band_gaps
full_dataset_df['material_id'] = material_ids
full_dataset_df.to_csv(
    'dataset.csv')
