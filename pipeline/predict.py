import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# custom files
sys.path.append('src')
import model_best_hyperparameters
import columns

# read train data
ds = pd.read_csv("data/new_input.csv")


# feature engineering
param_dict = pickle.load(open('models/param_dict.pickle', 'rb'))

def impute_na(df, variable, value):
    return df[variable].fillna(value)

# Outlier Engineering
for column in columns.outlier_columns:
    ds[column] = ds[column].astype(float)
    ds = ds[~ np.where(ds[column] > param_dict['upper_lower_limits'][column+'_upper_limit'], True,
                       np.where(ds[column] < param_dict['upper_lower_limits'][column+'_lower_limit'], True, False))]


# Categorical encoding
for column in columns.cat_columns:
    ds[column] = ds[column].map(param_dict['map_dicts'][column])
    # to encoding missing (new) categories
    ds[column] = impute_na(ds, column, max(param_dict['map_dicts'][column].values())+1)
 
# Define target and features columns
X = ds[columns.X_columns]

# load the model and predict
rf = pickle.load(open('models/finalized_model.sav', 'rb'))

y_pred = rf.predict(X)

ds['ObesityLevel_pred'] = rf.predict(X)
ds.to_csv('models/prediction_results.csv', index=False)