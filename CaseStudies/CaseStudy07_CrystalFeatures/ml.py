"""Band-gap prediction from bacd + rosa descriptors — with a leakage lesson.

Pipeline: load the two committed descriptor CSVs, merge on material_id,
train/test split, standardise (train-only fit), train XGBoost, report
MAE/R2 and parity plots. Model, scaler and splits are saved under
SavedModels/ and Dataset/.

*** FEATURE/LABEL LEAKAGE ***
The rosa features include the eigenvalues immediately bracketing the Fermi
level (features 49 and 50 of each 109-feature rosa copy) and the Fermi level
itself (feature 99). The HOMO-LUMO band gap is *directly readable* from
those features, so "predicting" it from the full feature set is close to
re-reading the answer off the label. The second run below drops those
leakage features so you can compare -- the honest number is the second one.
Sensitive to future maintenance: the rosa layout is 50 below + 50 above + ef
+ 8 energy terms, see descriptors_rosa.py.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TARGET = 'band_gaps'
# leakage features per 109-feature rosa copy: HOMO(49), LUMO(50), Fermi level(99)
ROSA_BLOCK = 109
LEAKAGE_OFFSETS = (49, 50, 99)

plt.rcParams.update({'font.size': 14})


def leakage_columns(n_rosa_features):
    """Merged column names of the Fermi-adjacent rosa features."""
    n_copies = n_rosa_features // ROSA_BLOCK
    return [f'{block * ROSA_BLOCK + off}_rosa'
            for block in range(n_copies) for off in LEAKAGE_OFFSETS
            if block * ROSA_BLOCK + off < n_rosa_features]


def train_and_report(X, y, label, regr, fig_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # fit the scaler on TRAIN only (leakage through shared statistics is leakage too)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    regr.fit(X_train_s, y_train)
    y_pred = regr.predict(X_test_s)
    print(f'{label:<28s} MAE {mean_absolute_error(y_test, y_pred):.4f} | '
          f'R2 {r2_score(y_test, y_pred):.4f}   '
          f'(train R2 {r2_score(y_train, regr.predict(X_train_s)):.4f})')

    plt.figure(figsize=(8, 8))
    plt.plot(y_test, y_pred, 'ro', alpha=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'k-')
    plt.ylabel('XGBoost (predicted)')
    plt.xlabel('DFT (actual)')
    plt.title(label)
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'XGBOOST_Regression_Correlation_'
                             + ('Test' if 'full' in label else 'Test_noleak') + '.png'),
                bbox_inches='tight', dpi=120)
    plt.close()

    if 'full' in label:
        os.makedirs('Dataset', exist_ok=True)
        os.makedirs('SavedModels', exist_ok=True)
        pd.DataFrame(X_train_s).to_csv('Dataset/X_train_scaled.csv', index=False)
        pd.DataFrame(X_test_s).to_csv('Dataset/X_test_scaled.csv', index=False)
        y_train.to_csv('Dataset/y_train.csv', index=False)
        y_test.to_csv('Dataset/y_test.csv', index=False)
        joblib.dump(scaler, 'SavedModels/regression_scalerX.pkl')
        joblib.dump(regr, 'SavedModels/XGBOOST_Regression.pkl')


def main():
    rosa = pd.read_csv('dataset_rosa.csv', index_col='material_id')
    bacd = pd.read_csv('dataset_bacd.csv', index_col='material_id')
    n_rosa = sum(1 for c in rosa.columns if c != TARGET)

    # explicit suffixes keep the duplicate feature indices ('0'..'324' vs
    # '0'..'217') distinguishable and self-documenting
    df = pd.merge(bacd, rosa, left_index=True, right_index=True,
                  suffixes=('_bacd', '_rosa'))
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print('Dataset size:', df.shape)

    y = df[TARGET]
    X_full = df.drop(columns=[TARGET])
    regr = XGBRegressor(n_estimators=100, max_depth=10, random_state=0)

    # --- the mirror-image runs -------------------------------------------
    train_and_report(X_full, y, 'full features (INCLUDES leakage)', regr, 'Figures')

    leak_cols = leakage_columns(n_rosa)
    X_honest = X_full.drop(columns=leak_cols)
    print(f'dropped leakage features: {leak_cols}')
    train_and_report(X_honest, y, 'honest features (leakage removed)',
                     XGBRegressor(n_estimators=100, max_depth=10, random_state=0),
                     'Figures')


if __name__ == '__main__':
    main()
