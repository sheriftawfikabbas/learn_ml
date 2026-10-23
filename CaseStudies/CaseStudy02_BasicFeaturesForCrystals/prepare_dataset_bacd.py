"""Build the bacd descriptor table for crystal structures from cif_results.json.

Input
-----
cif_results.json : list of dicts with keys ``material_id``, ``cif``, ``band_gap``
    (a Materials Project query export — see README.md on how to regenerate it).
    Default location: ../cif_results.json (override with argv[1]).

Output
------
BandGap/Dataset/dataset_bacd.csv : descriptor vector per material, indexed by
    material_id. This is the input of ``ml_formation_energy.py``.

The descriptor vector is computed by ``descriptors_bacd.descriptors`` — see its
docstring for what each block of features represents.
"""

import json
import os
import sys

import pandas as pd

import descriptors_bacd

JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else '../cif_results.json'
OUT_DIR = 'BandGap/Dataset'
OUT_CSV = os.path.join(OUT_DIR, 'dataset_bacd.csv')


def main():
    if not os.path.exists(JSON_PATH):
        raise SystemExit(
            f"Input file not found: {JSON_PATH}\n"
            "Generate it by querying the Materials Project (see README.md), "
            "or pass its path:  python prepare_dataset_bacd.py /path/to/cif_results.json")

    with open(JSON_PATH, 'r') as fout:
        results = json.load(fout)

    os.makedirs(OUT_DIR, exist_ok=True)

    dataset, material_ids = [], []
    for r in results:
        try:
            desc = descriptors_bacd.descriptors(r['cif'])
        except Exception as e:
            print(f"skipping {r.get('material_id', '?')}: {type(e).__name__}: {e}")
            continue
        material_ids.append(r['material_id'])
        dataset.append(desc)

    df = pd.DataFrame(dataset)
    df['material_id'] = material_ids
    df.to_csv(OUT_CSV)
    print(f"wrote {OUT_CSV}  ({df.shape[0]} materials x {df.shape[1] - 1} descriptors)")


if __name__ == '__main__':
    main()
