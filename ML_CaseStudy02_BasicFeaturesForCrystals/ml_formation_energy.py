# !pip3 install pymatgen xgboost sklearn pandas

from xgboost import XGBClassifier, plot_importance
from sklearn import linear_model, decomposition, datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
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

full_dataset_df = pd.read_csv('BandGap/Dataset/dataset_bacd.csv', index_col='material_id')
# full_dataset_df = pd.read_csv('Figures_v6/dataset.csv', index_col=0)
# Data visualization: let's have a look at our data.
folder = 'formation_energy_per_atom'
figfolder=folder+'/Figures/'

energies_df = pd.read_csv(
    '../dataset_energies.csv', index_col='material_id')
energies_df = energies_df['formation_energy_per_atom']
full_dataset_df = pd.merge(
    full_dataset_df, energies_df, left_index=True, right_index=True)

full_dataset_df = full_dataset_df.drop(
index=full_dataset_df.index[np.isinf(full_dataset_df).any(1)])

full_dataset_df = full_dataset_df.drop(
    index=full_dataset_df.index[np.isnan(full_dataset_df).any(1)])

for c in full_dataset_df.columns:
    if 'Unnamed' in c:
        del full_dataset_df[c]

print('Dataset size:',full_dataset_df.shape)
numXColumns = full_dataset_df.shape[1]-1
X = full_dataset_df.iloc[:, :numXColumns]
y = full_dataset_df['formation_energy_per_atom']
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=None)

scaler = StandardScaler().fit(X_train_scaled)
X_train_scaled = scaler.transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)
from pickle import dump
dump(scaler, open(folder+
                      '/SavedModels/regression_scaler.pkl', 'wb'))

regr_names = ['XGBOOST', 'RF', 'SVM']
regr_objects = [XGBRegressor(n_estimators=100, max_depth=100, random_state=0),
                RandomForestRegressor(
                    n_estimators=400, max_depth=400, random_state=0),
                svm.SVR(kernel='rbf', epsilon=0.1, verbose=True)
                # RVR(kernel='rbf', n_iter=10000, tol=0.0001, verbose=True),
                # linear_model.HuberRegressor(
                #     epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05),

                ]

for regr_choice in range(0, 1):

    regr = regr_objects[regr_choice]
    regr_name = regr_names[regr_choice]

    regr.fit(X_train_scaled, y_train)

    y_predicted = regr.predict(X_test_scaled)

    print(regr_name+' MSE\t'+str(mean_squared_error(y_test, y_predicted))+'\n')
    print(regr_name+' R2\t'+str(r2_score(y_test, y_predicted))+'\n')

    xPlot = y_test
    yPlot = y_predicted
    plt.figure(figsize=(10, 10))
    plt.plot(xPlot, yPlot, 'ro')
    plt.plot(xPlot, xPlot)
    plt.ylabel('RF')
    plt.xlabel('DFT')
    plt.savefig(figfolder+regr_name +
                '_Regression_Correlation_Test', bbox_inches='tight')

    y_predicted = regr.predict(X_train_scaled)

    print(regr_name+' MSE\t'+str(mean_squared_error(y_train, y_predicted))+'\n')
    print(regr_name+' R2\t'+str(r2_score(y_train, y_predicted))+'\n')


    xPlot = y_train
    yPlot = y_predicted
    plt.figure(figsize=(10, 10))
    plt.plot(xPlot, yPlot, 'ro')
    plt.plot(xPlot, xPlot)
    plt.ylabel('RF')
    plt.xlabel('DFT')
    plt.savefig(figfolder+regr_name +
                '_Regression_Correlation_Train', bbox_inches='tight')

    joblib.dump(regr, folder+'/SavedModels/'+regr_name+'_Regression.pkl')

