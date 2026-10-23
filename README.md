# learn_ml

Hands-on machine learning with Python: a sequence of classes followed by
worked case studies, with a focus on materials-science and real (messy) data.

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Some case studies need extra, heavyweight dependencies that are **commented
out** in `requirements.txt` (GPAW, Materials Project API, Google API client).
Install them only if you intend to run those specific scripts.

## Classes

| Class | Contents |
|---|---|
| `Classes/Class01_PythonReview` | Python fundamentals for the rest of the course: variables, containers, control flow, functions, imports. |
| `Classes/Class02_DataEngineering` | NumPy and pandas from zero: arrays, vectorisation, DataFrames, missing data, one-hot encoding, outlier removal, text manipulation. Includes `Lab_DataEngineering.ipynb` exercises (with a solutions notebook). |
| `Classes/Class03_ExploratoryDataAnalysis` | EDA workflow on a deliberately dirty satellite database: dtype inspection, distributions, categorical counts, correlations, and what each plot tells you. `eda.py` is an advanced script version of a full EDA pipeline. |
| `Classes/Class04_EnsembleLearning` | Why ensembles work (bagging, boosting, voting); a CNN ensemble for image classification plus a tabular `VotingClassifier`/`RandomForest` example. |

## Case studies

| Case study | Problem | Data | Methods |
|---|---|---|---|
| `CaseStudy01_ClimateChange` | Forecasting global temperature anomalies | Berkeley Earth (download or `Data/GlobalTemperatures.csv`) | SARIMAX, `auto_arima`, XGBoost, NN, naive baseline |
| `CaseStudy02_BasicFeaturesForCrystals` | Formation-energy prediction from elemental/structural descriptors | Materials Project (regeneration script provided) | XGBoost / RF / SVR regression |
| `CaseStudy03_Molecules` | HOMO–LUMO gap prediction from one-shot-DFT ("sDFT") spectral descriptors | `Data/Molecules` (GDB-9) + GPAW | XGBoost, feature-importance analysis |
| `CaseStudy04_COVIDDeaths` | Forecasting cumulative COVID-19 deaths | `Data/time_series_covid19_deaths_global.csv` | SARIMAX, `auto_arima`, XGBoost, NN |
| `CaseStudy05_MalariaClassification` | Parasitised vs healthy blood-cell images | `Data/cell_images` | CNN, hard/soft voting ensembles, patient-grouped evaluation |
| `CaseStudy06_YoutubeComments` | Downloading YouTube comments + intro NLP | YouTube Data API v3 | TF-IDF + logistic regression |
| `CaseStudy07_CrystalFeatures` | Band-gap prediction from two descriptor families | committed CSVs (Materials Project + GPAW) | XGBoost, descriptor/label-leakage discussion |

The script-based case studies (02, 03, 06, 07) each document their own
pipeline — **generate/prepare data → train model → evaluate** — in their
local README; the notebook case studies (01, 04, 05) are self-contained.

## Data

Shared datasets live in `Data/` and are documented in `Data/README.md`.
