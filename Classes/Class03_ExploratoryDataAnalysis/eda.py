"""Scriptable exploratory data analysis for a tabular dataset with a target column.

This is the advanced, command-line counterpart of EDA.ipynb: point it at any
CSV and it produces a standard battery of diagnostic plots in an output
folder. It was originally written for materials-science datasets
(target: band gap in eV, rows indexed by material_id), but both the index
column and the target column are configurable.

Usage
-----
    python eda.py DATA.csv [--index material_id] [--target band_gaps]
                          [--outdir EDA] [--features 40] [--umap]

Sections of the output (in <outdir>/):
    0  duplicates + info() (console)
    1  target distribution histogram
    2  per-feature distribution grid
    3  strongest correlations with the target (console)
    4  pair plots of features vs target (5 features per figure)
    5  correlations after removing zero values (console)
    6  feature-to-feature correlation heatmap
    7  k-means cluster sizes vs target statistics
    8  UMAP embedding (optional, --umap; requires umap-learn)
"""

import argparse
import operator
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csv', help='input CSV dataset')
    p.add_argument('--index', default='material_id', help='index column name')
    p.add_argument('--target', default='band_gaps', help='target column name')
    p.add_argument('--outdir', default='EDA', help='output folder for figures')
    p.add_argument('--features', type=int, default=40,
                   help='max number of leading feature columns for k-means/UMAP')
    p.add_argument('--umap', action='store_true',
                   help='also produce UMAP embedding plots (needs umap-learn)')
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"Dataset not found: {args.csv}\n"
                         "Pass the path as the first argument "
                         "(e.g. python eda.py Data/dataset_expanded.csv).")
    os.makedirs(args.outdir, exist_ok=True)
    sns.set_theme()

    df = pd.read_csv(args.csv, index_col=args.index)
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found. "
                         f"Available: {list(df.columns)}")

    def save(name):
        plt.savefig(os.path.join(args.outdir, name + '.png'), bbox_inches='tight')
        plt.close('all')
        print(f"  wrote {os.path.join(args.outdir, name + '.png')}")

    # 0 duplicates and structure
    print("Duplicated rows:", df.duplicated().sum())
    df.info()

    # 1 descriptive statistics and target distribution
    print(df.describe())
    sns.histplot(df[args.target], kde=True, color='g', bins=100, alpha=0.4)
    save(args.target + '_histogram')

    # 2 distribution grid of numeric features
    df_num = df.select_dtypes(include=['float64', 'int64'])
    df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    save('data_distribution')

    # 3 correlation with the target
    df_num_corr = df_num.corr()[args.target].iloc[:-1]   # drop target-vs-target row
    golden_features = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
    print("There are {} strongly correlated values:\n{}".format(
        len(golden_features), golden_features))

    # 4 pair plots: features vs target, five features per figure
    feature_names = [c for c in df_num.columns if c != args.target]
    for i in range(0, len(feature_names), 5):
        sns.pairplot(data=df_num, x_vars=feature_names[i:i + 5],
                     y_vars=[args.target])
        save(f'pairplot_{i // 5}')

    # 5 correlations after removing zero measurements from each feature
    zero_free_corr = {}
    for col in feature_names:
        view = df_num[[col, args.target]]
        view = view[view[col] != 0].dropna()
        if len(view) > 1:
            zero_free_corr[col] = view.corr()[args.target].iloc[0]
    for key, value in sorted(zero_free_corr.items(), key=operator.itemgetter(1)):
        print("{:>25}: {: .3f}".format(key, value))

    golden_zero_free = [k for k, v in zero_free_corr.items() if abs(v) >= 0.5]
    print("There are {} strongly correlated values after zero removal: {}"
          .format(len(golden_zero_free), golden_zero_free))

    # 6 feature-to-feature correlation heatmap
    corr = df_num.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, cmap='vlag', vmax=1.0, vmin=-1.0, center=0,
                linewidths=0.1, square=True)
    save('f2f')

    # 7 k-means clustering: do cluster sizes relate to the target?
    # KMeans is distance-based, so features must be on a common scale;
    # and neither the imputer nor KMeans tolerate NaNs, so impute first.
    feats = df_num[feature_names].iloc[:, :args.features]
    y = df_num[args.target]
    valid = y.notna()
    X = feats.loc[valid].to_numpy()
    y = y.loc[valid]
    X = SimpleImputer(strategy='mean').fit_transform(X)
    X = StandardScaler().fit_transform(X)

    n_clusters = min(20, len(X))
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(X)

    sizes, y_mean, y_std = [], [], []
    for k in range(n_clusters):
        members = y.index[labels == k]
        sizes.append((labels == k).sum())
        y_mean.append(100 * y.loc[members].mean())
        y_std.append(100 * y.loc[members].std())
    order = np.argsort(sizes)                # reorder ALL series consistently
    sizes = np.array(sizes)[order]
    y_mean = np.array(y_mean)[order]
    y_std = np.array(y_std)[order]

    plt.figure(figsize=(12, 10))
    plt.scatter(range(n_clusters), sizes, s=np.nan_to_num(y_mean, nan=10.0))
    plt.errorbar(range(n_clusters), sizes, yerr=np.nan_to_num(y_std), fmt='o')
    plt.xlabel('cluster (sorted by size)')
    plt.ylabel('cluster size')
    save('kclasses')

    # 8 optional UMAP embedding
    if args.umap:
        try:
            import umap
            import umap.plot
        except ImportError:
            print("umap-learn is not installed; skipping the UMAP section "
                  "(pip install umap-learn).")
            return
        rng = np.random.RandomState(0)
        take = rng.rand(len(X)) < 0.2
        mapper = umap.UMAP(random_state=0).fit(X[take])
        umap.plot.points(mapper)
        save('umap')
        umap.plot.diagnostic(mapper, diagnostic_type='pca')
        save('umap_diagnostics')
        umap.plot.connectivity(mapper, show_points=True)
        save('umap_connectivity')


if __name__ == '__main__':
    main()
