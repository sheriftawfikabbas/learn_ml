"""Train regressors to predict the HOMO-LUMO gap of GDB-9 molecules.

Input:  dataset.csv  (from molecules_generate_sdft_expanded_index.py;
        109 rosa sDFT features + band_gaps target in HARTREE)
Output: Molecules/SavedModels/{gap_XGBOOST,RF,SVR}_Regression.pkl,
        Molecules/Dataset/*.csv, parity plots, metrics printed below.

Methodology — what changed from the legacy script, and why it matters:

- *** No `while R2 < threshold` loop. *** Resampling train/test splits until
  the TEST score exceeds 0.97 is cherry-picking the test set: the reported
  R2 then measures repeated luck, not generalisation. A single fixed split
  (`random_state=0`) plus 5-fold cross-validation gives honest numbers.
- The target is converted to eV once, OUTSIDE any loop (27.2114 eV/Hartree).
- The scaler is fit on the training split only.
- Cross-validation estimate (5-fold R2 on the full data) is reported
  alongside the hold-out estimate — they should tell the same story.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

HARTREE_TO_EV = 27.2114    # 1 Hartree = 27.2114 eV (physics.nist.gov)
TARGET = 'band_gaps'
MODELS = {
    'XGBOOST': XGBRegressor(n_estimators=100, max_depth=10, random_state=0),
    'RF': RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
    'SVM': SVR(kernel='rbf', epsilon=0.1),
}

plt.rcParams.update({'font.size': 14})


def main():
    if not os.path.exists('dataset.csv'):
        raise SystemExit('dataset.csv not found — run '
                         'molecules_generate_sdft_expanded_index.py first')

    df = pd.read_csv('dataset.csv', index_col='material_id')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print('Dataset size:', df.shape)

    y = df[TARGET] * HARTREE_TO_EV   # HOMO-LUMO gap in eV
    X = df.drop(columns=[TARGET])

    model_dir = 'Molecules/SavedModels'
    data_dir = 'Molecules/Dataset'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler().fit(X_train)   # fit on TRAIN only
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_dir, 'regression_scalerX.pkl'))

    pd.DataFrame(X_train_s).to_csv(os.path.join(data_dir, 'X_train_scaled.csv'), index=False)
    pd.DataFrame(X_test_s).to_csv(os.path.join(data_dir, 'X_test_scaled.csv'), index=False)
    y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_dir, 'y_test.csv'), index=False)

    log = []
    for name, regr in MODELS.items():
        regr.fit(X_train_s, y_train)

        # 5-fold CV generalisation estimate (no test-set contact at all)
        cv_r2 = cross_val_score(regr, X_train_s, y_train, cv=5,
                                scoring='r2', n_jobs=-1).mean()

        for split, Xs, ys in [('Test', X_test_s, y_test),
                              ('Train', X_train_s, y_train)]:
            y_pred = regr.predict(Xs)
            line = (f'{name} {split:<6s} MAE {mean_absolute_error(ys, y_pred):.4f} | '
                    f'R2 {r2_score(ys, y_pred):.4f}')
            print(line)
            log.append(line)
            if split == 'Test':
                line = f'{name} 5-fold CV R2 {cv_r2:.4f}'
                print(line)
                log.append(line)
                plt.figure(figsize=(8, 8))
                plt.plot(ys, y_pred, 'ro', alpha=0.3)
                lims = [min(ys.min(), y_pred.min()), max(ys.max(), y_pred.max())]
                plt.plot(lims, lims, 'k-')
                plt.ylabel(f'{name} (predicted, eV)')
                plt.xlabel('DFT (actual, eV)')
                plt.title(f'{name} — gap prediction')
                plt.savefig(f'{name}_Regression_Correlation_Test.png',
                            bbox_inches='tight', dpi=120)
                plt.close()

        # model-name convention used by molecules_ml_featureimportance.py
        joblib.dump(regr, os.path.join(model_dir, f'gap_{name}_Regression.pkl'))

    with open('Molecules/metrics.txt', 'w') as f:
        f.write('\n'.join(log) + '\n')
    print('wrote Molecules/metrics.txt')


if __name__ == '__main__':
    main()
