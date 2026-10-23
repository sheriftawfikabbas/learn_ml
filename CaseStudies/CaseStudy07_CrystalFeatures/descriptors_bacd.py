"""bacd crystal descriptors: structure (CIF) -> 325-component feature vector.

The function ``descriptors`` accepts either:
- a CIF *string* (parsed with pymatgen), or
- any object with ``.cell``, ``.symbols`` and ``.positions`` (e.g. ASE ``Atoms``,
  pymatgen ``Structure``).

Feature layout (calculation_type='bulk', 325 features)
------------------------------------------------------
[0:4]     atomic-number statistics (mean, max, min, std)
[4]       cell volume per atom
[5]       "empty" volume: V_cell - sum(4/3 * pi * r_ionic^3) over species.
          A crude packing measure — it can be NEGATIVE when ionic radii
          overlap; treat it as a heuristic, not a physical volume.
[6]       metal fraction (see is_metal below)
[7:11]    interatomic distance statistics (mean, max, min, std).
          The minimum is computed over *off-diagonal* entries only; with the
          diagonal included it is trivially 0 for every structure.
[11:95]   21 elemental-property blocks x 4 statistics (mean, max, min, std
          over the atoms in the compound). Missing (None) property values are
          ZERO-IMPUTED: a simple, defensible choice that nonetheless conflates
          "unknown" with "zero". Alternatives (mean-impute per property,
          dropping the property block) are discussed in the case-study README.
[95:325]  230-component space-group one-hot.

With calculation_type='formula', the space-group/volume features vanish and
the vector is shorter — the intention is comparing bulk crystals against
formula-level aggregates, but note the vector layout CHANGES with this flag,
so never mix rows featurised with different calculation_type in one table.
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# The 21 per-atom properties, in the exact order of the feature blocks.
# (Order is part of the featurisation contract - do not reorder.)
_ELEMENTAL_PROPERTIES = [
    'van_der_waals_radius', 'electrical_resistivity', 'velocity_of_sound',
    'reflectivity', 'poissons_ratio', 'molar_volume', 'thermal_conductivity',
    'melting_point', 'critical_temperature', 'superconduction_temperature',
    'liquid_range', 'bulk_modulus', 'youngs_modulus', 'brinell_hardness',
    'rigidity_modulus', 'vickers_hardness', 'density_of_solid',
    'coefficient_of_linear_thermal_expansion', 'average_ionic_radius',
    'average_cationic_radius', 'average_anionic_radius',
]

# Elemental metals: alkali + alkaline-earth + d-block (all series incl.
# lanthanides/actinides) + post-transition metals Al, Ga, In, Sn, Tl, Pb, Bi,
# Nh, Fl, Mc, Lv. Metalloids (B, Si, Ge, As, Sb, Te) are deliberately left out.
_METAL_ZS = (
    {3, 11, 19, 37, 55, 87}                       # alkali
    | {4, 12, 20, 38, 56, 88}                    # alkaline earth
    | set(range(21, 31)) | set(range(39, 49))    # 3d, 4d transition metals
    | set(range(57, 72)) | set(range(72, 81))    # La-Lu, 5d
    | set(range(89, 104)) | set(range(104, 113))  # Ac-Lr, 6d
    | {13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116}
)


def _stats(values):
    """[mean, max, min, std], with None property values zero-imputed."""
    vals = [0.0 if v is None else v for v in values]
    return [float(np.mean(vals)), float(np.max(vals)),
            float(np.min(vals)), float(np.std(vals))]


def descriptors(cif, calculation_type='bulk'):
    """Compute the bacd feature vector for one structure; see module docstring."""
    if isinstance(cif, str):
        structure = CifParser.from_str(cif).parse_structures(primitive=True)[0]
    else:
        structure = Structure(lattice=cif.cell, species=cif.symbols,
                              coords=cif.positions)

    num_elements = len(structure.atomic_numbers)
    num_metals = sum(1 for e in structure.species if e.Z in _METAL_ZS)
    metals_fraction = num_metals / num_elements

    feats = [float(np.mean(structure.atomic_numbers)),
             float(np.max(structure.atomic_numbers)),
             float(np.min(structure.atomic_numbers)),
             float(np.std(structure.atomic_numbers))]

    if calculation_type == 'bulk':
        volume = structure.volume
        feats.append(float(volume / num_elements))
        atom_volume = sum(4 / 3 * np.pi * e.average_ionic_radius ** 3
                          for e in structure.species)
        feats.append(float(volume - atom_volume))   # "empty" volume (heuristic)

    feats.append(float(metals_fraction))

    if calculation_type != 'formula':
        dm = structure.distance_matrix
        off_diag = dm[~np.eye(dm.shape[0], dtype=bool)]
        feats += [float(np.mean(dm)), float(np.max(dm)),
                  float(np.min(off_diag)), float(np.std(dm))]

    for prop in _ELEMENTAL_PROPERTIES:
        feats += _stats([getattr(e, prop) for e in structure.species])

    if calculation_type == 'bulk':
        spg_number = SpacegroupAnalyzer(structure).get_space_group_number()
        feats += [1.0 if i == spg_number else 0.0 for i in range(1, 231)]

    return feats
