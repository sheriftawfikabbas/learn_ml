"""Train regressors to predict formation energy per atom from bacd descriptors.

Inputs
------
BandGap/Dataset/dataset_bacd.csv   descriptor table (prepare_dataset_bacd.py)
../dataset_energies.csv            DFT energies, with column
                                   ``formation_energy_per_atom`` (see README.md)

Methodology notes
-----------------
- The data is split BEFORE the scaler is fit: fitting the scaler on the whole
  dataset would leak test-set information into training.
- ``random_state`` is fixed so runs are reproducible — always do this in
  experiments you intend to compare.
- Metrics and parity plots are produced for every model in MODELS, on both
  the held-out test split and the training split. A large train/test gap
  signals overfitting; this is why a sane ``max_depth`` matters.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

DATASET_CSV = 'BandGap/Dataset/dataset_bacd.csv'
ENERGIES_CSV = '../dataset_energies.csv'
TARGET = 'formation_energy_per_atom'
OUT_DIR = TARGET  # figures and models go under formation_energy_per_atom/

# Keyed by display name so parity-plot labels can never drift from the model.
MODELS = {
    'XGBOOST': XGBRegressor(n_estimators=100, max_depth=10, random_state=0),
    'RF': RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1),
    'SVM': SVR(kernel='rbf', epsilon=0.1),
}


def main():
    for path in (DATASET_CSV, ENERGIES_CSV):
        if not os.path.exists(path):
            raise SystemExit(f"missing input: {path} — see README.md for the pipeline")

    dataset = pd.read_csv(DATASET_CSV, index_col='material_id')
    energies = pd.read_csv(ENERGIES_CSV, index_col='material_id')[TARGET]
    df = dataset.join(energies, how='inner')

    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print('Dataset size:', df.shape)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # fit the scaler on TRAIN ONLY — test statistics must stay unseen
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fig_dir = os.path.join(OUT_DIR, 'Figures')
    model_dir = os.path.join(OUT_DIR, 'SavedModels')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'regression_scaler.pkl'))

    def parity_plot(y_true, y_pred, name, split):
        plt.figure(figsize=(10, 10))
        plt.plot(y_true, y_pred, 'ro')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, 'k-')
        plt.ylabel(f'{name} (predicted)')
        plt.xlabel('DFT (actual)')
        plt.title(f'{name} — {split}')
        plt.savefig(os.path.join(fig_dir, f'{name}_Regression_Correlation_{split}.png'),
                    bbox_inches='tight')
        plt.close()

    for name, regr in MODELS.items():
        regr.fit(X_train_scaled, y_train)

        for split, Xs, ys in [('Test', X_test_scaled, y_test),
                              ('Train', X_train_scaled, y_train)]:
            y_pred = regr.predict(Xs)
            print(f'{name} {split:<6s} MSE {mean_squared_error(ys, y_pred):.4f} | '
                  f'R2 {r2_score(ys, y_pred):.4f}')
            parity_plot(np.asarray(ys), y_pred, name, split)

        joblib.dump(regr, os.path.join(model_dir, f'{name}_Regression.pkl'))


if __name__ == '__main__':
    main()
