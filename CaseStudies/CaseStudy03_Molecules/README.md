# Case Study 03 — Molecules

Predict the **HOMO–LUMO gap** of small organic molecules from a fast spectral
fingerprint of the electronic structure.

## Pipeline

```
molecules_generate_sdft_expanded_index.py
     GDB-9 xyz files (Data/Molecules/, committed)  +  GPAW one-shot DFT
          →  dataset.csv                       (109 rosa features + target)
          →  molecules_column_names_sdft.csv   (feature-name reference)
molecule_ml_bandgap.py
          →  Molecules/SavedModels/  (gap_{XGBOOST,RF,SVM}_Regression.pkl)
          →  Molecules/Dataset/      (scaled train/test splits)
          →  parity plots + Molecules/metrics.txt
molecules_ml_featureimportance.py
          →  Molecules/Figures/molecules_featureimportances_heatmap_total_gain.png
             (importances grouped into physical blocks: VBM / CBM / energies)
```

## The descriptors ("rosa", single-shot DFT)

Each molecule is featurised by a GPAW LCAO calculation capped at
`maxiter=1`: deliberately **unconverged**. One iteration already yields a
meaningful one-electron spectrum at a tiny fraction of the cost — the
resulting eigenvalues (50 below / 50 above the Fermi level), the Fermi level,
and 8 per-atom Hamiltonian energy terms form a 109-component fingerprint.
Treat them as fingerprints, not observables. See `descriptors_rosa.py`.

Requires **GPAW** and **PAW setups** (`GPAW_SETUP_PATH` environment variable);
GPAW is a heavy HPC-grade install — this is why `dataset.csv` is the natural
checkpoint and the featuriser is separate from the ML scripts. Skip the
generation step and read `molecule_ml_bandgap.py` to study the ML if you
don't have GPAW.

## The GDB-9 xyz format

Each `dsgdb9nsd_*.xyz` file is:

```
<n_atoms>
<comment line: 15 scalar quantum-chemical properties>
<element x y z>   (× n_atoms)
```

Field **index 9** of the comment line is the HOMO–LUMO gap in **Hartree**
(converted to eV — ×27.2114 — in `molecule_ml_bandgap.py`).

## Honest-modelling notes (things the legacy version got wrong)

- The old training script re-sampled train/test splits in a
  `while R2 < 0.97` loop — that **optimises against the test set**, and the
  reported R² measured repeated luck, not skill. The rewrite uses one fixed
  split (`random_state=0`) plus 5-fold cross-validation.
- The rosa features include eigenvalues bracketing the Fermi level; on
  molecules the HOMO/LUMO gap sits inside that window. The same
  leakage-auditing mindset as Case Study 07 applies: know which features
  can read your label.
- Scaler fit on the training split only.
