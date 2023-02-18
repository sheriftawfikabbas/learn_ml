import json
import pandas as pd
import descriptors_rosa
import descriptors_bacd
from pymatgen.ext.matproj import MPRester
from pymatgen.ext.matproj import MPRestError

m = MPRester("Ua7LfrKkn9yTWA3t")

results = m.query({
    "formula_anonymous": "ABC3",
    "elements": {"$all": ["O"]},
    "nsites": {"$lte": 20},
    "band_gap": {"$gt": 1}}, properties=[
    "material_id",
    "pretty_formula",
    "cif",
    "band_gap",
    "energy_per_atom",
    "formation_energy_per_atom",
    "e_above_hull",
    "created_at"])

with open('tutorial.json', 'w') as fout:
    json.dump(results, fout)

# fout = open('tutorial.json', 'r')
# results = json.load(fout)

dataset = []
band_gaps = []
material_ids = []

for r in results:

    cif = r['cif']
    bg = r['band_gap']
    formula = r['pretty_formula']
    material_id = r['material_id']
    d = descriptors_rosa.descriptors(cif)
    if len(d) > 0:
        dataset += [descriptors_rosa.descriptors(cif)]
        band_gaps += [bg]
        material_ids += [material_id]
        print(formula, material_id, str(bg))
    else:
        print('Problem in material:', r['material_id'])
        continue

dataset_df = pd.DataFrame(dataset)
full_dataset_df = pd.DataFrame(dataset)
full_dataset_df['band_gaps'] = band_gaps
full_dataset_df['material_id'] = material_ids
full_dataset_df.to_csv(
    'dataset_rosa.csv')


dataset = []
material_ids = []
for r in results:

    cif = r['cif']
    bg = r['band_gap']
    formula = r['pretty_formula']
    material_id = r['material_id']
    d = descriptors_bacd.descriptors(cif)
    dataset += [d]
    material_ids += [material_id]
    print(formula, material_id, str(bg))

dataset_df = pd.DataFrame(dataset)
full_dataset_df = pd.DataFrame(dataset)
full_dataset_df['material_id'] = material_ids
full_dataset_df.to_csv(
    'dataset_bacd.csv')