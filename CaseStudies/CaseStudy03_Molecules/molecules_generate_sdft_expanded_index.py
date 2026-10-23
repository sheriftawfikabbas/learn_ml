"""Featurise GDB-9 molecules with rosa sDFT descriptors -> dataset.csv.

Input
-----
../../Data/Molecules/*.xyz   (GDB-9 quantum-chemistry dataset)

Each xyz file has the GDB-9 layout::

    <n_atoms>
    <comment line with scalar properties>
    <element x y z>  x n_atoms

Field index 9 of the comment line is the HOMO-LUMO gap in HARTREE. The
target is converted to eV only at ML time (molecule_ml_bandgap.py).

Output
------
- dataset.csv                     one row per molecule (109 rosa features +
                                  band_gaps target + material_id)
- molecules_column_names_sdft.csv feature names (Pristine_SDFT_*), consumed
                                  by molecules_ml_featureimportance.py

Usage:  python molecules_generate_sdft_expanded_index.py [limit]
With no argument, all molecules in the folder are processed; pass an integer
to featurise only the first N (alphabetical order) for a quick smoke run.
Requires GPAW + PAW setups (see descriptors_rosa.py).
"""

import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from ase import Atoms

import descriptors_rosa

MOLECULES_DIR = '../../Data/Molecules/*.xyz'
ENERGY_TERMS = ['ef', 'e_band', 'e_coulomb', 'e_entropy', 'e_external',
                'e_kinetic', 'e_kinetic0', 'e_xc', 'e_total_free']

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None


def column_names():
    """Feature names matching the 109-feature rosa layout."""
    return (['Pristine_SDFT_BelowFermi_{}'.format(i) for i in range(50)] +
            ['Pristine_SDFT_AboveFermi_{}'.format(i) for i in range(50)] +
            ['Pristine_SDFT_' + e for e in ENERGY_TERMS])


def read_gdb9_xyz(path):
    """Parse one GDB-9 xyz file -> (ASE Atoms in a padded cell, gap float)."""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    fields = lines[1].split()
    gap_hartree = float(fields[9])          # field index 9 = HOMO-LUMO gap
    atoms = [ln.split() for ln in lines[2:2 + n]]
    symbols = [a[0] for a in atoms]
    positions = np.array([[float(v) for v in a[1:4]] for a in atoms])
    mol = Atoms(positions=positions, symbols=symbols, pbc=False)
    # isolated molecule in a GPAW cell: pad by 10 Angstrom of vacuum per axis
    spans = mol.positions.max(axis=0) - mol.positions.min(axis=0)
    mol.cell = np.diag(spans + 10.0)
    mol.center()
    return mol, gap_hartree


def main():
    xyz_files = sorted(glob(MOLECULES_DIR))
    if LIMIT is not None:
        xyz_files = xyz_files[:LIMIT]
    if not xyz_files:
        raise SystemExit(f'no xyz files found at {MOLECULES_DIR}')
    print(f'{len(xyz_files)} molecules to featurise')

    rows, gaps, ids = [], [], []
    for i, path in enumerate(xyz_files, 1):
        system_name = os.path.splitext(os.path.basename(path))[0]
        try:
            mol, gap = read_gdb9_xyz(path)
            d = descriptors_rosa.descriptors(mol)
            if len(d) == 0:
                raise RuntimeError('featuriser returned no descriptors')
        except Exception as e:
            print(f'skipping {system_name}: {type(e).__name__}: {e}')
            continue
        rows.append(d)
        gaps.append(gap)
        ids.append(system_name)
        if i % 500 == 0:
            print(f'  {i}/{len(xyz_files)} done')

    df = pd.DataFrame(rows, columns=column_names())
    df['band_gaps'] = gaps
    df['material_id'] = ids
    df.to_csv('dataset.csv', index=False)
    print(f'wrote dataset.csv: {df.shape}')

    # feature-name reference is just the header line (read with nrows=0)
    pd.DataFrame(columns=column_names()).to_csv(
        'molecules_column_names_sdft.csv', index=False)
    print('wrote molecules_column_names_sdft.csv')


if __name__ == '__main__':
    main()
