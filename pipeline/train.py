import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
# custom files
sys.path.append('src')
import model_best_hyperparameters
import columns

# Шлях до файлу "train.csv"
file_path = os.path.join("data", "train.csv")


# read train data
ds = pd.read_csv(file_path)

# Outlier Engineering
def find_skewed_boundaries(df, variable, distance):
    df[variable] = pd.to_numeric(df[variable],errors='coerce')
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

upper_lower_limits = dict()
for column in columns.outlier_columns:
    upper_lower_limits[column+'_upper_limit'], upper_lower_limits[column+'_lower_limit'] = find_skewed_boundaries(ds, column, 5)
for column in columns.outlier_columns:
    ds = ds[~ np.where(ds[column] > upper_lower_limits[column+'_upper_limit'], True,
                       np.where(ds[column] < upper_lower_limits[column+'_lower_limit'], True, False))]

# Categorical encoding
map_dicts = dict()
for column in columns.cat_columns:
    ds[column] = ds[column].astype('category')
    map_dicts[column] = dict(zip(ds[column], ds[column].cat.codes))
    ds[column] = ds[column].cat.codes


# save parameters 
param_dict = {
              'upper_lower_limits':upper_lower_limits,
              'map_dicts':map_dicts
             }

with open('models/param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# Define target and features columns
X = ds[columns.X_columns]
y = ds[columns.y_column]

# Let's say we want to split the data in 90:10 for train:test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9)

# Building and train Random Forest Model
rf = RandomForestClassifier(**model_best_hyperparameters.params)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('test set metrics: ', metrics.classification_report(y_test, y_pred))

filename = 'models/finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))