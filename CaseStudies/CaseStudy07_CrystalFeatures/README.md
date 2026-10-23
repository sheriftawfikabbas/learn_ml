# Case Study 07 — Crystal features

Predict the **band gap** of ABC3 oxides from two descriptor families, and —
just as importantly — learn to audit your features for **label leakage**.

## Pipeline

```
generate_data.py        Materials Project query (MP_API_KEY required)
     │                  → tutorial.json (raw export, cached)
     │                  → dataset_rosa.csv   (109 sDFT-spectral features; GPAW)
     │                  → dataset_bacd.csv   (325 pymatgen statistics)
     ▼
ml.py                   merge -> split -> scale (train fit only) -> XGBoost
     │                  trains TWICE: full features vs leakage-removed
     ▼
Figures/, Dataset/, SavedModels/
```

## Running it

```bash
pip install xgboost scikit-learn pandas matplotlib
python ml.py
```

The committed CSVs are ready, so `ml.py` works offline. Regenerating data is
optional and needs: `pip install mp-api pymatgen`, an `MP_API_KEY`
environment variable (get a key at https://next-gen.materialsproject.org/api;
do **not** commit it), and GPAW + PAW setups for the rosa descriptors
(`GPAW_SETUP_PATH` pointing at your `gpaw-setups-*` directory).

## The descriptor families

- **bacd** (pymatgen-only): elemental-property statistics (mean/max/min/std
  of 21 per-atom properties), structural statistics, off-diagonal distance
  statistics, and a 230-component space-group one-hot. See
  `descriptors_bacd.py`.
- **rosa** (needs GPAW): a **one-shot DFT fingerprint** — eigenvalues
  bracketing the Fermi level (50 below / 50 above), the Fermi level itself,
  and per-atom Hamiltonian energy terms from a single LCAO iteration
  (`maxiter=1`). Cheap to compute, noisy by construction; see
  `descriptors_rosa.py` for the exact layout.

## The leakage lesson (the heart of this case study)

The rosa features **contain a direct proxy of the label**: features
49/50/99 of each 109-feature block are the HOMO, LUMO and Fermi level —
so the band gap is, in principle, readable from the input. `ml.py` trains
two runs so you can see what that costs; on the committed data:

```
full features (INCLUDES leakage) MAE 0.3689 | R2 0.8157
honest features (leakage removed) MAE 0.3671 | R2 0.8066
```

Two honest conclusions rather than one lazy one:

1. **Leakage exists even when it doesn't inflate the score much.** Here it
   barely moves R² — the one-shot spectrum is too noisy a proxy and the
   320+ bacd features carry most of the signal. The correct action is still
   to drop the leaky features: quit re-reading the label before declaring a
   model "predictive".
2. **Beware the other extreme: overfitting.** Train R² is 1.0000 on both
   runs (printed next to test R² for exactly this reason) — with 780 samples
   and ~540 features, `max_depth=10` memorises. A deeper investigation would
   tune capacity and use cross-validation.

## Notes

- `tutorial.json` / both CSVs are committed, so the pipeline reproduces
  without API access. The live query path in `generate_data.py` uses the
  modern `mp-api` client (`MP_API_KEY` env var, never hard-coded).
- An earlier version of this repo accidentally committed an MP API key in
  `generate_data.py`; that key must be considered compromised and rotated.
- Regenerated `dataset_rosa.csv` will have 109 columns (the legacy CSV has
  218: the second half was a duplicated "perturbed" copy — see the note in
  `descriptors_rosa.py`). `ml.py` handles either width.
