# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 22:14:33 2021

@author: laukkara
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import radpy

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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


#########################################################
# Input data
with open('./input/data_RASMI.pickle', 'rb') as f:
    data = pickle.load(f)

output_folder = r'./output_plots'

clim = 'Sod_1989-2018'

T_e = data[clim].loc[:, 'Te']

RHe_water = data[clim].loc[:,'RHe_water']
pv = (RHe_water/100.0) * radpy.calc_psat(T_e)
T_dew = radpy.calc_Tdew(pv)

R_glob = data[clim].loc[:, 'Rglob']

t_local_hour = np.arange(30*8760) - 0.5
kwargs_betaphi = {'LAT_deg': 65.0,
          'LON_deg': 24.5,
          'tz': 2.0,
          'plot_folder_path': None}
beta_deg, phi_deg = radpy.calc_solar_position(t_local_hour,
                                              kwargs_betaphi)
beta_rad = beta_deg * (np.pi/180.0)

kwargs_Eo = {'plot_folder_path': None}
E_o = radpy.calc_extraterrestrial_radiant_flux(t_local_hour,
                                               kwargs_Eo)

R_hor_max = E_o * np.sin( np.maximum(0, beta_rad) )

LW_dn = data[clim].loc[:, 'LWdn_era5']




# Some feature engineering could be practiced here, i.e. by creating a 
# two-times-per-day reciprocal of sum(R_glob)/sum(R_hor_max)

radpy.calc_Rglob_vs_Rmax()


################################################

X_all = np.column_stack( (T_e,
                          T_dew,
                          R_glob,
                          R_hor_max) )

y_all = LW_dn.values.reshape(-1,1)

idx_split = 28*8760
X_train = X_all[:idx_split, :]
y_train = y_all[:idx_split]

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)


X_test = X_all[idx_split:, :]
y_test = y_all[idx_split:]

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test) # not needed


y_pred = {}



## Fit

# Dummy Regressor
ml_model = 'DummyRegressor'
print(ml_model)
model = DummyRegressor()
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

mae = mean_absolute_error(y_test, y_pred[ml_model])
print('  MAE: {:.3f}'.format(mae))
r2 = r2_score(y_test, y_pred[ml_model])
print('  R2: {:.3f}'.format(r2))

fig, ax = plt.subplots()
ax.plot(y_test, y_pred[ml_model], '.', ms=0.4)
ax.plot([150,400],[150,400])
plt.title(ml_model)
fname = os.path.join(output_folder, 'LWdn_' + clim + '_' + ml_model + '.png')
fig.savefig(fname, dpi=200, bbox_inches='tight')




# LinearRegression
ml_model = 'LinearRegression'
print(ml_model)
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

mae = mean_absolute_error(y_test, y_pred[ml_model])
print('  MAE: {:.3f}'.format(mae))
r2 = r2_score(y_test, y_pred[ml_model])
print('  R2: {:.3f}'.format(r2))

fig, ax = plt.subplots()
ax.plot(y_test, y_pred[ml_model], '.', ms=0.4)
ax.plot([150,400],[150,400])
plt.title(ml_model)
fname = os.path.join(output_folder, 'LWdn_' + clim + '_' + ml_model + '.png')
fig.savefig(fname, dpi=200, bbox_inches='tight')




# Elastic Net
ml_model = 'ElasticNet'
print(ml_model)
reg_model = ElasticNet()
param_distributions = {'alpha': ss_loguniform(1e-2, 1e2),
                       'l1_ratio': ss_uniform(0.1, 0.8)}
tss = TimeSeriesSplit(n_splits=5)
model = RandomizedSearchCV(estimator=reg_model,
                        param_distributions=param_distributions,
                        n_iter=1000,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        refit=True)
model.fit(X_train_scaled, y_train_scaled.ravel())
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_)
mae = mean_absolute_error(y_test, y_pred[ml_model])
print('  MAE test: {:.3f}'.format(mae))
r2 = r2_score(y_test, y_pred[ml_model])
print('  R2 test: {:.3f}'.format(r2))

fig, ax = plt.subplots()
ax.plot(y_test, y_pred[ml_model], '.', ms=0.4)
ax.plot([150,400],[150,400])
plt.title(ml_model)
fname = os.path.join(output_folder, 'LWdn_' + clim + '_' + ml_model + '.png')
fig.savefig(fname, dpi=200, bbox_inches='tight')





# Random Forest
ml_model = 'RandomForestRegressor'
print(ml_model)
reg_model = RandomForestRegressor()
param_distributions = {'n_estimators': ss_randint(10, 30),
                       'max_depth': ss_randint(3, 5),
                       'criterion': ['mae'],
                       'n_jobs': [1],
                       'max_samples': [1000]}
tss = TimeSeriesSplit(n_splits=3)
model = RandomizedSearchCV(estimator=reg_model,
                        param_distributions=param_distributions,
                        n_iter=3,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        refit=True)
model.fit(X_train_scaled, y_train_scaled.ravel())
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_)
mae = mean_absolute_error(y_test, y_pred[ml_model])
print('  MAE: {:.3f}'.format(mae))
r2 = r2_score(y_test, y_pred[ml_model])
print('  R2: {:.3f}'.format(r2))

fig, ax = plt.subplots()
ax.plot(y_test, y_pred[ml_model], '.', ms=0.4)
ax.plot([150,400],[150,400])
plt.title(ml_model)
fname = os.path.join(output_folder, 'LWdn_' + clim + '_' + ml_model + '.png')
fig.savefig(fname, dpi=200, bbox_inches='tight')




# xgboost
ml_model = 'XGBRegressor'
print(ml_model)
reg_model = xgb.XGBRegressor()
param_distributions = {'n_estimators': ss_randint(50, 200),
                       'max_depth': ss_randint(5, 10),
                       'learning_rate': ss_uniform(0.5, 0.4),
                       'objective': ['reg:squarederror'],
                       'booster': ['gbtree'],
                       'n_jobs': [1],
                       'subsample': ss_uniform(0.5, 0.2)}
tss = TimeSeriesSplit(n_splits=3)
model = RandomizedSearchCV(estimator=reg_model,
                        param_distributions=param_distributions,
                        n_iter=3,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        refit=True)
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_)
mae = mean_absolute_error(y_test, y_pred[ml_model])
print('  MAE: {:.3f}'.format(mae))
r2 = r2_score(y_test, y_pred[ml_model])
print('  R2: {:.3f}'.format(r2))

fig, ax = plt.subplots()
ax.plot(y_test, y_pred[ml_model], '.', ms=0.4)
ax.plot([150,400],[150,400])
plt.title(ml_model)
fname = os.path.join(output_folder, 'LWdn_' + clim + '_' + ml_model + '.png')
fig.savefig(fname, dpi=200, bbox_inches='tight')




# lightgbm
ml_model = 'LGBMRegressor'
print(ml_model)
reg_model = lgb.LGBMRegressor()
param_distributions = {'boosting_type': ['dart'],
                       'max_depth': ss_randint(4, 10),
                       'learning_rate': ss_uniform(0.5, 0.4),
                       'n_estimators': ss_randint(50, 200),
                       'subsample': ss_uniform(0.5, 0.2),
                       'n_jobs': [1]}
tss = TimeSeriesSplit(n_splits=3)
model = RandomizedSearchCV(estimator=reg_model,
                        param_distributions=param_distributions,
                        n_iter=3,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        refit=True)
model.fit(X_train_scaled, y_train_scaled.ravel())
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_)
mae = mean_absolute_error(y_test, y_pred[ml_model])
print('  MAE: {:.3f}'.format(mae))
r2 = r2_score(y_test, y_pred[ml_model])
print('  R2: {:.3f}'.format(r2))

fig, ax = plt.subplots()
ax.plot(y_test, y_pred[ml_model], '.', ms=0.4)
ax.plot([150,400],[150,400])
plt.title(ml_model)
fname = os.path.join(output_folder, 'LWdn_' + clim + '_' + ml_model + '.png')
fig.savefig(fname, dpi=200, bbox_inches='tight')



