"""Regenerate tutorial.json and the two descriptor CSVs from the Materials Project.

Requirements
------------
- ``pip install mp-api pymatgen`` plus a Materials Project API key exported as
  ``MP_API_KEY`` (create one at https://next-gen.materialsproject.org/api —
  and NEVER commit it to version control; gitignore covers local key files).
- GPAW + PAW setups for the rosa descriptors (see descriptors_rosa.py).

What it does
------------
1. Query MP for ABC3 oxides with nsites<=20 and band gap > 1 eV (the teaching
   cohort: small-ish, gapped ternary oxides).
2. Save the raw query to tutorial.json (cached: if the file exists, reuse it).
3. Featurise every structure twice: rosa (sDFT spectral fingerprint, needs
   GPAW) and bacd (pymatgen-only statistics).
4. Write dataset_rosa.csv / dataset_bacd.csv (index: material_id).

Rows whose descriptor computation fails are skipped with a warning (the
rosa/featuriser returns [] on GPAW failure — see descriptors_rosa.py).
"""

import json
import os

import pandas as pd

import descriptors_bacd
import descriptors_rosa

PROPERTIES = ['material_id', 'pretty_formula', 'cif', 'band_gap',
              'energy_per_atom', 'formation_energy_per_atom',
              'e_above_hull', 'created_at']


def query_materials_project():
    """Fetch the teaching cohort (cached to tutorial.json for reruns)."""
    cache = 'tutorial.json'
    if os.path.exists(cache):
        print(f'reusing cached {cache}')
        with open(cache) as f:
            return json.load(f)

    from mp_api.client import MPRester   # modern MP API client
    api_key = os.environ.get('MP_API_KEY')
    if not api_key:
        raise SystemExit('Set your Materials Project key in MP_API_KEY. '
                         'See README.md / https://next-gen.materialsproject.org/api')

    with MPRester(api_key) as m:
        # note: mp-api evolves; consult the MPRester docs if a field moves
        docs = m.materials.summary.search(
            formula_anonymous='ABC3', elements=['O'],
            num_sites=(1, 20), band_gap=(1.0, None),
            fields=['material_id', 'structure', 'band_gap', 'energy_per_atom',
                    'formation_energy_per_atom', 'e_above_hull',
                    'formula_pretty', 'created_at'])
    results = [{'material_id': d.material_id,
                'pretty_formula': d.formula_pretty,
                'cif': d.structure.to(fmt='cif'),
                'band_gap': d.band_gap,
                'energy_per_atom': d.energy_per_atom,
                'formation_energy_per_atom': d.formation_energy_per_atom,
                'e_above_hull': d.e_above_hull,
                'created_at': str(d.created_at)}
               for d in docs]
    with open(cache, 'w') as f:
        json.dump(results, f)
    return results


def featurise(results, descriptor_fn, kind):
    """One CSV row per material; failing rows are skipped, not fatal."""
    rows, gaps, ids = [], [], []
    for r in results:
        d = descriptor_fn(r['cif'])
        if len(d) == 0:
            print(f"skipping {r['material_id']} ({kind} featurisation failed)")
            continue
        rows.append(d)
        gaps.append(r['band_gap'])
        ids.append(r['material_id'])
        print(r['pretty_formula'], r['material_id'], r['band_gap'])

    df = pd.DataFrame(rows)
    df['band_gaps'] = gaps
    df['material_id'] = ids
    return df.set_index('material_id')


def main():
    results = query_materials_project()

    rosa = featurise(results, descriptors_rosa.descriptors, 'rosa')
    rosa.to_csv('dataset_rosa.csv')
    print(f'dataset_rosa.csv: {rosa.shape}')

    bacd = featurise(results, descriptors_bacd.descriptors, 'bacd')
    bacd = bacd.drop(columns=['band_gaps'])   # bacd CSV keeps no target column
    bacd.to_csv('dataset_bacd.csv')
    print(f'dataset_bacd.csv: {bacd.shape}')


if __name__ == '__main__':
    main()
