"""rosa descriptors: structure -> sDFT spectral fingerprint (109 components).

"rosa" featurises a structure with a **single-shot DFT** calculation: a GPAW
LCAO job whose SCF cycle is capped at ``maxiter=1`` (deliberately NOT
converged). The trick — the "sDFT" idea — is that a one-iteration calculation
already produces a sensible one-electron spectrum at a tiny fraction of a
converged calculation's cost. Treat the resulting numbers as fingerprints,
not as physical observables.

Feature layout (``descriptor_size=100`` -> 109 features)
---------------------------------------------------------
[0:50]    eigenvalues BELOW the Fermi level, ascending, VALENCE-side:
          the 50 closest lie at the END of the block (feature 49 = HOMO).
          Fewer than 50 states -> zero-padded at the START of the block.
[50:100]  eigenvalues ABOVE the Fermi level, ascending (feature 50 = LUMO);
          fewer than 50 -> zero-padded at the END of the block.
[100]     the Fermi level itself
[101:109] Hamiltonian energy terms, per atom: band, Coulomb, entropy,
          external, kinetic, kinetic(0), XC, total-free energies.

Interface
---------
``descriptors(x)`` accepts:
- an ASE ``Atoms`` object (molecules, CaseStudy03), or
- a CIF *string* (crystals, CaseStudy02/07; parsed with pymatgen and
  converted to ASE Atoms with periodic boundary conditions).

It returns a list of 109 floats, or ``[]`` if the calculator failed (the
generator scripts skip such rows with a warning).

Requirements: GPAW plus PAW setup files. Point the environment variable
``GPAW_SETUP_PATH`` at your setups directory (e.g. .../gpaw-setups-0.9.20000);
paw-file availability is checked from there. Elements without a setup raise
a clear error instead of the historical silent Yttrium substitution.

NOTE on committed artefacts: CaseStudy07's committed dataset_rosa.csv has
218 columns = 109 pristine + 109 legacy "perturbed-copy" features. The
perturbation body in the legacy module was an always-False comparison and the
second copy was an expensive duplicate; it has been removed. Regenerating the
dataset now produces 109 features; ml.py works with either width.
"""

import os

import numpy as np

_SETUP_DIR = os.environ.get('GPAW_SETUP_PATH', '')


def _available_elements():
    if not _SETUP_DIR:
        return None   # unknown; validation disabled
    import glob
    files = glob.glob(os.path.join(_SETUP_DIR, '*.PBE.gz'))
    return {os.path.basename(f).split('.')[0] for f in files}


def fix_psuedo(a):
    """Check every element has a PAW setup; otherwise refuse, loudly.

    (Kept name for compatibility; this historical helper SILENTLY replaced
    missing elements with Yttrium, corrupting the chemistry without a record.
    It now raises instead.)"""
    available = _available_elements()
    if available is None:
        return
    missing = sorted({a[i].symbol for i in range(len(a)) if a[i].symbol not in available})
    if missing:
        raise ValueError(f"no GPAW PAW setup for {missing} in {_SETUP_DIR}")


def get_descriptors_for_structure(a, descriptor_size=100):
    """Run the one-shot GPAW calculation and pack the spectral fingerprint.

    Parameters
    ----------
    a : ase.Atoms
    descriptor_size : int, split evenly below/above the Fermi level
    """
    from gpaw import GPAW   # lazy: importing this module must not need GPAW

    half = descriptor_size // 2
    calc = GPAW(mode='lcao', xc='PBE', maxiter=1,
                convergence={'density': 1}, kpts=[1, 1, 1],
                txt='gpaw_lcao.txt')
    calc.calculate(a)

    num_atoms = a.get_global_number_of_atoms()
    eigenvalues = np.sort(calc.get_eigenvalues())
    ef = calc.get_fermi_level()

    below = eigenvalues[eigenvalues <= ef][-half:]   # ascending, HOMO last
    above = eigenvalues[eigenvalues > ef][:half]     # ascending, LUMO first
    below = np.pad(below.tolist(), (half - len(below), 0))
    above = np.pad(above.tolist(), (0, half - len(above)))

    H = calc.hamiltonian
    return (below.tolist() + above.tolist() + [
        ef,
        H.e_band / num_atoms,
        H.e_coulomb / num_atoms,
        H.e_entropy / num_atoms,
        H.e_external / num_atoms,
        H.e_kinetic / num_atoms,
        H.e_kinetic0 / num_atoms,
        H.e_xc / num_atoms,
        H.e_total_free / num_atoms,
    ])


def descriptors(x, descriptor_size=100, calculation_type='bulk'):
    """Structure (ASE Atoms or CIF string) -> 109 rosa features, [] on failure."""
    from ase import Atoms
    from ase.build import niggli_reduce

    if isinstance(x, Atoms):
        a = x.copy()
    elif isinstance(x, str):
        from pymatgen.io.cif import CifParser
        structure = CifParser.from_str(x).parse_structures(primitive=True)[0]
        a = Atoms(pbc=True, cell=structure.lattice.matrix,
                  positions=structure.cart_coords,
                  numbers=structure.atomic_numbers)
    else:
        raise TypeError(f"expected ASE Atoms or a CIF string, got {type(x)}")

    try:
        if calculation_type == 'bulk':
            niggli_reduce(a)
        fix_psuedo(a)
        return get_descriptors_for_structure(a, descriptor_size)
    except Exception as e:
        print(f"descriptors failed ({type(e).__name__}): {e}")
        return []
