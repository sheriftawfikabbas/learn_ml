import joblib
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
 
plt.rcParams.update({'font.size': 30})

#Target property: band_gaps
energy = 'band_gaps'

figfolder = 'Figures/'

regr_name = 'XGBOOST'
regr = XGBRegressor(n_estimators=100, max_depth=100, random_state=0)

rosa_dataset_df = pd.read_csv(
    'dataset_rosa.csv', index_col='material_id')

bacd_dataset_df = pd.read_csv(
    'dataset_bacd.csv', index_col='material_id')

full_dataset_df = pd.merge(bacd_dataset_df,rosa_dataset_df,left_index=True,right_index=True)

full_dataset_df = full_dataset_df.drop(
index=full_dataset_df.index[np.isinf(full_dataset_df).any(1)])

full_dataset_df = full_dataset_df.drop(
    index=full_dataset_df.index[np.isnan(full_dataset_df).any(1)])

for c in full_dataset_df.columns:
    if 'Unnamed' in c:
        del full_dataset_df[c]

print('Dataset size:',full_dataset_df.shape)

# Create the dataset X and y
y = full_dataset_df[energy]
X = full_dataset_df
del X[energy]
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=None)

# Normalize the X
scalerX = StandardScaler().fit(X_train_scaled)
X_train_scaled = scalerX.transform(X_train_scaled)
X_test_scaled = scalerX.transform(X_test_scaled)

# Save the normalization model
from pickle import dump
dump(scalerX, open('SavedModels/' +
                    'regression_scalerX.pkl', 'wb'))

# Save the datasets
X_train_scaled_df = pd.DataFrame(X_train_scaled)
X_test_scaled_df = pd.DataFrame(X_test_scaled)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)
X_train_scaled_df.to_csv('Dataset/' +
                            'X_train_scaled.csv')
X_test_scaled_df.to_csv('Dataset/X_test_scaled.csv')
y_train_df.to_csv('Dataset/y_train.csv')
y_test_df.to_csv('Dataset/y_test.csv')

# ML training starts here!
regr.fit(X_train_scaled, y_train)
y_predicted = regr.predict(X_test_scaled)

# Check the results for the test set
print(regr_name+' MAE\t'+str(mean_absolute_error(y_test, y_predicted))+'\n')
print(regr_name+' R2\t'+str(r2_score(y_test, y_predicted))+'\n')

# Correlation plot
xPlot = y_test
yPlot = y_predicted
plt.figure(figsize=(10, 10))
plt.plot(xPlot, yPlot, 'ro', alpha=0.1)
plt.plot(xPlot, xPlot)
plt.ylabel(regr_name)
plt.xlabel('DFT')
plt.savefig(figfolder+regr_name +
            '_Regression_Correlation_Test', bbox_inches='tight')

y_predicted = regr.predict(X_train_scaled)

print(regr_name+' MAE\t'+str(mean_absolute_error(y_train, y_predicted))+'\n')
print(regr_name+' R2\t'+str(r2_score(y_train, y_predicted))+'\n')
 
xPlot = y_train
yPlot = y_predicted
plt.figure(figsize=(10, 10))
plt.plot(xPlot, yPlot, 'ro', alpha=0.1)
plt.plot(xPlot, xPlot)
plt.ylabel(regr_name)
plt.xlabel('DFT')
plt.savefig(figfolder+regr_name +
            '_Regression_Correlation_Train', bbox_inches='tight')



joblib.dump(regr, 'SavedModels/' +
            regr_name+'_Regression.pkl')
