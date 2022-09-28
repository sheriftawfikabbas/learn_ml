# !pip3 install pymatgen xgboost sklearn pandas

from sklearn.cluster import KMeans
import matplotlib as mpl
import umap.plot
import umap
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from torch.utils.data import DataLoader
from torch import nn
import torch
from pickle import dump
from xgboost import XGBClassifier, plot_importance
from sklearn import linear_model, decomposition, datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, roc_auc_score
import joblib
import importlib
import json
from pymatgen.io.cif import CifParser
from urllib.request import urlopen
import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.ext.matproj import MPRestError
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import average_precision_score, precision_score
import seaborn as sns
plt.clf()
plt.style.use('bmh')

folder = 'band_gaps'
# folder = 'formation_energy_per_atom'
# full_dataset_df = pd.read_csv('/mnt/c/MyResearch/MachineLearning/ML/ML_DScribe/ewald_dataset_e.csv', index_col='material_id')
df = pd.read_csv('band_gaps/Dataset/dataset_expanded.csv', index_col='material_id')

for c in df.columns:
    if 'Unnamed' in c:
        del df[c]

# df = df.loc[df.band_gaps > 0]

# 0 nulls and duplicates
df_X = df#.iloc[:,0:40]
duplicated = df_X[df_X.duplicated()]

#np.argwhere(np.isnan(X))

df.info()

# 1 Descriptive statistics
df.describe()
sns.distplot(df['band_gaps'], color='g', bins=100, hist_kws={'alpha': 0.4})
plt.savefig('EDA/band_gaps_histogram')
plt.clf()

# 2 Numerical data distribution
df_num = df.select_dtypes(include=['float64', 'int64'])
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.savefig('EDA/data_distribution')
plt.clf()

# 3 Correlation
df_num_corr = df_num.corr()['band_gaps'][:-1]
golden_features_list = df_num_corr[abs(
    df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values:\n{}".format(
    len(golden_features_list), golden_features_list))

# 4 Plot correlating features
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                 x_vars=df_num.columns[i:i+5],
                 y_vars=['band_gaps'])
plt.savefig('EDA/pairplot')
plt.clf()

# 5 Remove 0 values for features

individual_features_df = []
for i in range(0, len(df_num.columns) - 1):
    tmpDf = df_num[[df_num.columns[i], 'band_gaps']]
    tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]
    individual_features_df.append(tmpDf)

all_correlations = {feature.columns[0]: feature.corr(
)['band_gaps'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))

golden_features_list = [key for key,
                        value in all_correlations if abs(value) >= 0.5]
print("There is {} strongly correlated values:\n{}".format(
    len(golden_features_list), golden_features_list))

# 6 Feature-to-feature

corr = df_num.corr()
plt.figure(figsize=(20, 20))

sns.heatmap(corr,
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)
plt.savefig('EDA/f2f')
plt.clf()

# 8 K-means
X = df.iloc[:, 0:40]
y = df['band_gaps']
X = X.to_numpy()


kmeans = KMeans(n_clusters=20, random_state=0).fit(X)
kmeans.labels_

kmeans_classes = {}
for i in range(20):
    kmeans_classes[i] = []
    for j in range(len(kmeans.labels_)):
        if kmeans.labels_[j] == i:
            kmeans_classes[i] += [df.index.values[j]]

kclasses = []
y_mean = []
y_std = []
for k in kmeans_classes.keys():
    print(k, len(kmeans_classes[k]))
    kclasses += [len(kmeans_classes[k])]
    y_mean +=[100*y[kmeans_classes[k]].mean()]
    y_std +=[100*y[kmeans_classes[k]].std()]
kclasses = np.sort(np.array(kclasses))

plt.figure(figsize=(12, 10))
plt.scatter(range(20),kclasses,s=y_mean)
plt.errorbar(range(20),kclasses, yerr=y_std, fmt="o")
plt.savefig('EDA/kclasses', bbox_inches='tight')
plt.clf()

# 7 UMap

# df = df.drop(
# index=df.index[np.isinf(df).any(1)])
# df = df.drop(
#     index=df.index[np.isnan(df).any(1)])
df_sample = df.sample(frac=0.2)
X = df_sample.iloc[:, 0:40]
y = df_sample['band_gaps']
mapper = umap.UMAP().fit(X)
# mpl.rcParams['figure.dpi'] = 3000
plt.clf()
plt.figure(figsize=(10, 10))
umap.plot.points(mapper)
plt.savefig('EDA/umap_K', bbox_inches='tight')
umap.plot.diagnostic(mapper, diagnostic_type='pca')
plt.savefig('EDA/umap_diagnostics_K', bbox_inches='tight')
umap.plot.connectivity(mapper, show_points=True)
plt.savefig('EDA/umap_connectivity_K', bbox_inches='tight')
