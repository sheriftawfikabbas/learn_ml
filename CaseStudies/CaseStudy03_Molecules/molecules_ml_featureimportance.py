"""Group XGBoost feature importances into the physical descriptor blocks.

For every trained model found in Molecules/SavedModels (named
{property}_XGBOOST_Regression.pkl, trained by molecule_ml_bandgap.py), the
total_gain importance of each feature is summed into three PHYSICAL groups:

- VBM      : the 50 eigenvalues below the Fermi level (valence-band-edge
             character)
- CBM      : the 50 eigenvalues above the Fermi level (conduction-band-edge
             character)
- energies : the Fermi level and the 8 per-atom Hamiltonian energy terms

and rendered as a heatmap (rows = predicted properties), so you can read
which physics the model actually consults. Importances are reported as
percentages of the total gain; since all 109 features are assigned to a
group here, the three values per property sum to 100% by construction.

Feature-name reference: molecules_column_names_sdft.csv (written by
molecules_generate_sdft_expanded_index.py).
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL_DIR = 'Molecules/SavedModels'
NAMES_CSV = 'molecules_column_names_sdft.csv'
FIG_DIR = 'Molecules/Figures'

PROPERTY_NAMES = {'gap': r'$E_{\mathrm{HOMO-LUMO}}$',
                  'alpha': r'$\alpha$',
                  'G': '$G$'}
# group label -> feature-name test
GROUPS = [
    ('VBM', lambda n: n.startswith('Pristine_SDFT_BelowFermi')),
    ('CBM', lambda n: n.startswith('Pristine_SDFT_AboveFermi')),
    ('energies', lambda n: n.startswith('Pristine_SDFT_e')),
]


def main():
    column_names = pd.read_csv(NAMES_CSV, nrows=0).columns.tolist()

    rows = []
    for prop, pretty in PROPERTY_NAMES.items():
        model_file = os.path.join(MODEL_DIR, f'{prop}_XGBOOST_Regression.pkl')
        if not os.path.exists(model_file):
            print(f'skipping {prop}: {model_file} not found '
                  '(run molecule_ml_bandgap.py)')
            continue
        regr = joblib.load(model_file)
        importance = regr.get_booster().get_score(importance_type='total_gain')

        # booster names features f0..f108 positionally unless fit with names
        scores = pd.Series(
            {name: importance.get(f'f{i}', 0.0)
             for i, name in enumerate(column_names)})

        total = scores.sum()
        row = {}
        for group_label, test in GROUPS:
            row[group_label] = 100 * scores[[test(n) for n in column_names]].sum() / total
        rows.append(pd.Series(row, name=pretty))
        print(f'{prop}: VBM {row["VBM"]:.1f}% | CBM {row["CBM"]:.1f}% | '
              f'energies {row["energies"]:.1f}%  (sum {sum(row.values()):.1f}%)')

    if not rows:
        raise SystemExit('no trained models found; nothing to plot')

    feature_data = pd.DataFrame(rows)
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(feature_data, annot=True, fmt='.1f', cmap='viridis')
    plt.ylabel('')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'molecules_featureimportances_heatmap_total_gain.png')
    plt.savefig(out, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
