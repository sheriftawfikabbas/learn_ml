
import os
import warnings
import joblib
import dataset_split
import descriptors_fragments
import descriptors_bacd
import importlib
import json
from pymatgen.io.cif import CifParser
from urllib.request import urlopen
import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.ext.matproj import MPRestError
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read cif json in results
fout = open('../cif_results.json', 'r')
results = json.load(fout)

importlib.reload(descriptors_bacd)
importlib.reload(descriptors_fragments)
importlib.reload(dataset_split)

dataset = []
band_gaps = []
material_ids = []
for i in range(len(results)):
    r = results[i]

    cif = r['cif']
    # parser = CifParser.from_string(cif)

    # structure = parser.get_structures()
    # structure = structure[0]
    # if len(structure)<10:
    bg = r['band_gap']
    # if bg < 3:
    material_ids += [r['material_id']]
    dataset += [descriptors_bacd.descriptors(cif)]
    band_gaps += [bg]
    print(r['material_id'], bg)

df = pd.DataFrame(dataset)
df['material_id'] = material_ids
df.to_csv('BandGap/Dataset/dataset_bacd.csv')


