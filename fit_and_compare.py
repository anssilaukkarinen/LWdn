# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 22:14:33 2021

@author: Anssi Laukkarinen

This file contains code that:
    - First uses 'read_in_data.py' to read in RASMI and ERA5 data
    - Fits a collection of machine learning models to outdoor and LWdn
      data and exports some figures and statistics
    - Saves the fitted LightGBM models to file

When setting up virtual environment, there were DLL errors when
importing pvlib. It was necessary to run:
    pip uninstall h5py
    pip install h5py

This file is run first.
The file is run multiple times while changing the "location" parameter manually


"""


import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import uniform as ss_uniform # [loc, loc+scale]
from scipy.stats import loguniform as ss_loguniform # [a, b)
from scipy.stats import randint as ss_randint # [low, high)]

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV


import read_in_data
import helper

################################



# Van, Jok, Jyv, Sod
location = 'Sod'


clim_current = location + '_1989-2018'




root_folder = r'C:\Users\laukkara\github\LWdn'

folder_rasmi = os.path.join(root_folder,
                            'input',
                            'RASMI')

folder_era5 = os.path.join(root_folder,
                           'input',
                           'era5_strd_rami_20211005')

# Here 'data' has index in UTC time
data = read_in_data.read_RASMI_and_ERA5(folder_rasmi,
                                        folder_era5,
                                        location)




output_folder = os.path.join(root_folder,
                             'output')

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)



t_now_str = pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M-%S')
log_file = os.path.join(output_folder,
                        'log_' + clim_current + '_' + t_now_str + '.txt')
f_log = open(log_file, 'w')

print('Start!', file=f_log)



#################################


LAT_deg, LON_deg = helper.get_LAT_and_LON(location)


X_all, y_all = helper.create_X_y(data[clim_current], LAT_deg, LON_deg)


idx_split = int(X_all.shape[0] * (27.0/30.0))
X_train = X_all[:idx_split, :]
y_train = y_all[:idx_split]

X_test = X_all[idx_split:, :]
y_test = y_all[idx_split:]


scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(X_train)
scaler_y.fit(y_train)


X_train_scaled = scaler_X.transform(X_train)
y_train_scaled = scaler_y.transform(y_train)

X_test_scaled = scaler_X.transform(X_test)
# y_test_scaled = scaler_y.transform(y_test) # not needed


y_pred = {}


#################################


## Fit

# Dummy Regressor
ml_model = 'DummyRegressor'
print(ml_model)
print(ml_model, file=f_log)
model = DummyRegressor()
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
y_pred_scaled = y_pred_scaled.reshape(-1,1)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

helper.print_and_plot(ml_model,
                      clim_current,
                      y_test,
                      y_pred[ml_model],
                      f_log,
                      output_folder)







# Wallenten
ml_model = 'K-model'
print(ml_model)
print(ml_model, file=f_log)
y_pred[ml_model] = helper.K_model(X_test[:, 0], X_test[:, 1], X_test[:,-1])

helper.print_and_plot(ml_model,
                      clim_current,
                      y_test,
                      y_pred[ml_model],
                      f_log,
                      output_folder)







# LinearRegression
ml_model = 'LinearRegression'
print(ml_model)
print(ml_model, file=f_log)
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

helper.print_and_plot(ml_model,
                      clim_current,
                      y_test,
                      y_pred[ml_model],
                      f_log,
                      output_folder)






# Elastic Net
ml_model = 'ElasticNet'
print(ml_model)
print(ml_model, file=f_log)
reg_model = ElasticNet()
param_distributions = {'alpha': ss_loguniform(0.01, 1e2),
                       'l1_ratio': ss_uniform(0.1, 0.8)}
tss = TimeSeriesSplit(n_splits=5)
model = RandomizedSearchCV(estimator=reg_model,
                            param_distributions=param_distributions,
                            n_iter=500,
                            cv=tss,
                            scoring='neg_mean_absolute_error',
                            refit=True)
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
y_pred_scaled = y_pred_scaled.reshape(-1,1)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_, file=f_log)

helper.print_and_plot(ml_model,
                      clim_current,
                      y_test,
                      y_pred[ml_model],
                      f_log,
                      output_folder)








# Random Forest
t_start = pd.Timestamp.now()
ml_model = 'RandomForestRegressor'
print(ml_model)
print(ml_model, file=f_log)
reg_model = RandomForestRegressor()
param_distributions = {'n_estimators': ss_randint(10, 50),
                       'max_depth': ss_randint(3, 5),
                       'criterion': ['absolute_error'],
                       'n_jobs': [1],
                       'max_samples': [1000]}
tss = TimeSeriesSplit(n_splits=5)
model = RandomizedSearchCV(estimator=reg_model,
                        param_distributions=param_distributions,
                        n_iter=500,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        refit=True)
model.fit(X_train_scaled, y_train_scaled.ravel())
y_pred_scaled = model.predict(X_test_scaled)
y_pred_scaled = y_pred_scaled.reshape(-1,1)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_, file=f_log)

helper.print_and_plot(ml_model,
                      clim_current,
                      y_test,
                      y_pred[ml_model],
                      f_log,
                      output_folder)

t_end = pd.Timestamp.now()
t_duration = t_end - t_start
print('Random Forest time:', t_duration.total_seconds()/60, 'min',
      file=f_log)






# xgboost
t_start = pd.Timestamp.now()
ml_model = 'XGBRegressor'
print(ml_model)
print(ml_model, file=f_log)
reg_model = xgb.XGBRegressor()
param_distributions = {'n_estimators': ss_randint(50, 200),
                       'max_depth': ss_randint(5, 10),
                       'learning_rate': ss_uniform(0.5, 0.4),
                       'objective': ['reg:squarederror'],
                       'booster': ['gbtree'],
                       'n_jobs': [1],
                       'subsample': ss_uniform(0.5, 0.2)}
tss = TimeSeriesSplit(n_splits=5)
model = RandomizedSearchCV(estimator=reg_model,
                        param_distributions=param_distributions,
                        n_iter=500,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        refit=True)
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
print(ml_model, y_pred_scaled.shape)
y_pred_scaled = y_pred_scaled.reshape(-1,1)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_, file=f_log)

helper.print_and_plot(ml_model,
                      clim_current,
                      y_test,
                      y_pred[ml_model],
                      f_log,
                      output_folder)

t_end = pd.Timestamp.now()
t_duration = t_end - t_start
print('xgboost time:', t_duration.total_seconds()/60, 'min',
      file=f_log)







# lightgbm
# LightGBM requires additional coding for feature names,
# so that it would not give warnings.
# There is already functionality in the .fit() method to handle feature names,
# but pandas DataFrame is used separately, because the same DataFrames
# would probably be usable in other methods also.

X_column_names_for_LGBM = ['Te', 'Tdew', 'Rglob', 'R_hor_max', 'K_0']
y_column_names_for_LGBM = ['LW_dn']

t_start = pd.Timestamp.now()
ml_model = 'LGBMRegressor'
print(ml_model)
print(ml_model, file=f_log)
reg_model = lgb.LGBMRegressor()
param_distributions = {'boosting_type': ['dart'],
                       'max_depth': ss_randint(4, 10),
                       'learning_rate': ss_uniform(0.5, 0.4),
                       'n_estimators': ss_randint(50, 200),
                       'subsample': ss_uniform(0.5, 0.2),
                       'n_jobs': [1]}
tss = TimeSeriesSplit(n_splits=5)
model = RandomizedSearchCV(estimator=reg_model,
                        param_distributions=param_distributions,
                        n_iter=500,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        refit=True)

X_train_scaled_df = pd.DataFrame(data=X_train_scaled,
                                 columns=X_column_names_for_LGBM)
y_train_scaled_df = pd.DataFrame(data=y_train_scaled,
                                 columns=y_column_names_for_LGBM)
X_test_scaled_df = pd.DataFrame(data=X_test_scaled,
                                columns=X_column_names_for_LGBM)
model.fit(X_train_scaled_df, y_train_scaled_df)
y_pred_scaled = model.predict(X_test_scaled_df)
y_pred_scaled = y_pred_scaled.reshape(-1,1)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_, file=f_log)

helper.print_and_plot(ml_model,
                      clim_current,
                      y_test,
                      y_pred[ml_model],
                      f_log,
                      output_folder)

t_end = pd.Timestamp.now()
t_duration = t_end - t_start
print('lgbm time:', t_duration.total_seconds()/60, 'min',
      file=f_log)


## 

model_best = model.best_estimator_
model_scalerX_scalery = [model_best, scaler_X, scaler_y]

fname = os.path.join(output_folder,
                     f'LGBMRegressor_bestModel_scalerX_scalery_LWdn_{location}.pickle')
with open(fname, 'wb') as f:
    pickle.dump(model_scalerX_scalery, f)






## Close log file
print('End', file=f_log)
f_log.close()







