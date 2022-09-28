# pip3 install cmake --user
# pip3 install xgboost --user
# pip3 install sklearn pandas keras --user

from operator import index
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
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import average_precision_score, precision_score


import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 40})
importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
# for importance_type in importance_types:
importance_type = importance_types[3]

column_names = pd.read_csv('molecules_column_names_sdft.csv', index_col='material_id')
column_names = column_names.columns
properties = ['gap', 'alpha','G']
property_names = ['$E_{HOMO-LUMO}$', r'$\alpha$', '$G$']
prefixes = ['f', 'f', 'f']


row_list = []
feature_data_df = pd.DataFrame()
for iproperty in range(3):

    print('Working on', properties[iproperty])
    feature_group = []
    group_score = []
    property = []
    figfolder = 'Molecules/Figures/'

    regr = joblib.load(
       'Molecules/SavedModels/'+properties[iproperty]+'_' 
            + 'XGBOOST_Regression.pkl')

    importance = regr.get_booster().get_score(importance_type=importance_type)
    keys = list(importance.keys())
    values = list(importance.values())
    imp = []
    prefix = prefixes[iproperty]
    for k in range(len(column_names)):
        if prefix+str(k) in keys:
            imp += [importance[prefix+str(k)]]
        else:
            imp += [0]

    feature_importances = pd.DataFrame(data=imp, index=column_names, columns=[
        "score"]).sort_values(by="score", ascending=False)

    total_scores = feature_importances.sum().values[0]

    Pristine_SDFT_BelowFermi = feature_importances.loc[feature_importances.index.str.startswith(
        'Pristine_SDFT_BelowFermi')].count()[0]
    Pristine_SDFT_AboveFermi = feature_importances.loc[feature_importances.index.str.startswith(
        'Pristine_SDFT_AboveFermi')].count()[0]
    Pristine_SDFT_e = feature_importances.loc[feature_importances.index.str.startswith(
        'Pristine_SDFT_e')].count()[0]
    print('Number of descriptors:', Pristine_SDFT_BelowFermi +
          Pristine_SDFT_AboveFermi+Pristine_SDFT_e)

    feature_group += ['VBM']
    group_score += [
        feature_importances.loc[feature_importances.index.str.startswith('Pristine_SDFT_BelowFermi')].sum().values[0]/total_scores*100]
    feature_group += ['CBM']
    group_score += [
        feature_importances.loc[feature_importances.index.str.startswith('Pristine_SDFT_AboveFermi')].sum().values[0]/total_scores*100]
    feature_group += ['$e$']
    group_score += [
        feature_importances.loc[feature_importances.index.str.startswith('Pristine_SDFT_e')].sum().values[0]/total_scores*100]
    print('Verify totals:',sum(group_score))
    row_df = pd.DataFrame(
        [[property_names[iproperty]]+group_score], columns=['property']+feature_group)
    feature_data_df = feature_data_df.append(row_df)
    """
    import shap
    X_test_scaled = pd.read_csv(
        properties[iproperty]+'/Dataset/X_test_scaled.csv')
    del X_test_scaled['Unnamed: 0']
    print('Done loading X_test_scaled')
    explainer = shap.TreeExplainer(regr,feature_names=column_names)
    print('Done creating explainer')
    shap_values = explainer(X_test_scaled.sample(frac=1000/X_test_scaled.shape[0]),check_additivity=False)
    print('Done calculating shap values')
    plt.figure(figsize=(10, 10))
    shap.plots.beeswarm(shap_values,max_display=6)
    plt.savefig(figfolder+'SHAP_beeswarm_XGBOOST_Regression_Correlation_Test', bbox_inches='tight')
    plt.clf()
    """
feature_data_df=feature_data_df.set_index('property')
plt.figure(figsize=(20, 20))
plt.rcParams.update({'font.size': 60})
sns.heatmap(feature_data_df, annot=True, fmt='.2f')
# sns.set(font_scale=0.2)
plt.ylabel('')
plt.savefig('molecules_featureimportances_heatmap_'+importance_type, bbox_inches='tight')
