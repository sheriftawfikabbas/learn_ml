# Case Study 02 — Basic features for crystals

Predict **formation energy per atom** of crystals from descriptor features
computed directly from their CIF structures. This case study uses
**regressors on tabular features** — no images, no time series.

## Pipeline

```
cif_results.json          (Materials Project query export: material_id, cif, band_gap)
        │
        ▼
prepare_dataset_bacd.py   →  BandGap/Dataset/dataset_bacd.csv   (descriptor table)
        │
        ▼   (joined with ../dataset_energies.csv — DFT formation energies)
ml_formation_energy.py    →  parity plots (formation_energy_per_atom/Figures/)
                             models + scaler (formation_energy_per_atom/SavedModels/)
```

## Required inputs (not committed)

This case study is **not runnable as-is**; it documents a research pipeline
and requires two upstream files:

- `../cif_results.json` — a Materials Project export: a JSON list of
  `{"material_id": ..., "cif": ..., "band_gap": ...}` records. Regenerate it
  with the `mp-api` package (see CaseStudy07's `generate_data.py` for a query
  template; the MP API now requires an `MP_API_KEY` environment variable).
- `../dataset_energies.csv` — a CSV indexed by `material_id` with a
  `formation_energy_per_atom` column (DFT-computed energies for the same
  material set).

Both paths are configurable as arguments/constants at the top of each script.

## The descriptors (`descriptors_bacd.py`)

Each crystal is turned into a fixed-length feature vector:

- **elemental-property statistics** — for each of ~25 per-atom properties
  (electronegativity, atomic radius, valence electrons, …), the mean / max /
  min / standard deviation over the atoms in the compound;
- **structure statistics** — cell volume, packing, density aggregates;
- **space-group one-hot** — 230 components flagging the crystal's space group.

See the docstring in `descriptors_bacd.py` for the exact definition and the
choices involved (zero-imputation of missing properties, etc.).

## Honest modelling notes

- The scaler is fit on the training split only (`random_state=0` for
  reproducibility).
- Default hyperparameters are sane starting points, not tuned optima — expect
  XGBoost ≳ RandomForest > SVR on this task.
- `descriptors_bacd.py` also exists (in adapted forms) in CaseStudy03 and
  CaseStudy07; CaseStudy07's `ml.py` applies the same ideas to band-gap
  prediction — where feature/label leakage becomes the central lesson.
