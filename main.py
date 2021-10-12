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
import datetime

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


#################################

def MPW(T_e, T_dew, K_0):
    # [T_e] = degC, [T_dew] = degC, K_0 is clearness index [0...1]
    print('Initiating MPW...')
    
    sigma_SB = 5.67e-8
    
    fig, ax = plt.subplots()
    ax.plot(T_e)
    
    fig, ax = plt.subplots()
    ax.plot(T_dew)
    
    fig, ax = plt.subplots()
    ax.plot(K_0)
    
    
    epsilon_sky = 1.5357 \
                  + 0.5981 * (T_dew/100.0) \
                  - 0.5687 * ((T_e + 273.15)/273.15) \
                  - 0.2799 * K_0

    fig, ax = plt.subplots()
    ax.plot(epsilon_sky)

    LW_dn_MPW = epsilon_sky * sigma_SB * (T_e + 273.15)**4
    
    print('Finished MPW!')
    
    return(LW_dn_MPW)



def create_X_y(data, clim, LAT_deg, LON_deg):
    
    
    n_hours = data[clim].loc[:,'Te'].shape[0]
    
    t_local_hour = np.arange(n_hours) - 0.5
    
    # Interpolate T_e, RHe_water to previous half hours
    x = t_local_hour
    xp = np.arange(n_hours)
    fp = data[clim].loc[:, 'Te'].values
    T_e = np.interp(x, xp, fp)
    
    x = t_local_hour
    xp = np.arange(n_hours)
    fp = data[clim].loc[:,'RHe_water'].values
    RHe_water = np.interp(x, xp, fp)
    
    
    pv = (RHe_water/100.0) * radpy.calc_psat(T_e)
    T_dew = radpy.calc_Tdew(pv)
    
    R_glob = data[clim].loc[:, 'Rglob']
    
    kwargs_betaphi = {'LAT_deg': LAT_deg,
                      'LON_deg': LON_deg,
                      'tz': 2.0,
                      'plot_folder_path': 'output',
                      'plot_file_name': clim}
    beta_deg, phi_deg = radpy.calc_solar_position(t_local_hour,
                                                  kwargs_betaphi)
    beta_rad = beta_deg * (np.pi/180.0)
    
    kwargs_Eo = {'plot_folder_path': 'output',
                 'plot_file_name': clim}
    E_o = radpy.calc_extraterrestrial_radiant_flux(t_local_hour,
                                                   kwargs_Eo)
    
    R_hor_max = E_o * np.sin( np.maximum(0, beta_rad) )
    
    K_0 = radpy.calc_clearness_index(R_glob, R_hor_max)
    
    if 'LWdn_era5' in data[clim].columns:
        LW_dn = data[clim].loc[:, 'LWdn_era5']
    else:
        n_shape = data[clim].loc[:, 'Te'].shape
        LW_dn = pd.DataFrame(np.zeros(n_shape))
    
    
    X_all = np.column_stack( (T_e,
                              T_dew,
                              R_glob,
                              R_hor_max,
                              K_0) )
    
    y_all = LW_dn.values.reshape(-1,1)
    
    return(X_all, y_all)


def print_to_screen_and_file(y_measured, y_predicted, log_file_handle):
    
    mae = mean_absolute_error(y_measured, y_predicted)
    print('  MAE: {:.3f}'.format(mae), file=log_file_handle)
    print('  MAE: {:.3f}'.format(mae))
    
    r2 = r2_score(y_measured, y_predicted)
    print('  R2: {:.3f}'.format(r2), file=log_file_handle)
    print('  R2: {:.3f}'.format(r2))
    
    print('\n', file=log_file_handle)
    

def plot_scatter(y_measured, y_predicted, output_folder, clim, ml_model):
    fig, ax = plt.subplots()
    ax.plot(y_measured, y_predicted, '.', ms=0.4)
    ax.plot([150,400],[150,400])
    ax.set_xlabel('measured, W/m$^2$')
    ax.set_ylabel('predicted, W/m$^2$')
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.title(ml_model)
    fname = os.path.join(output_folder, 'LWdn_' + clim + '_scatter_' + ml_model + '.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_timeseries(y_measured, y_predicted, output_folder, clim, ml_model):
    fig, ax = plt.subplots()
    ax.plot(np.arange(8760), y_measured[-8760:],
            np.arange(8760), y_predicted[-8760:], linewidth=0.6)
    ax.set_xlabel('t, h')
    ax.set_ylabel('LWdn, W/m$^2$')
    ax.legend(['measured','predicted'])
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.title(ml_model)
    fname = os.path.join(output_folder, 'LWdn_' + clim + '_timeseries_' + ml_model + '.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_ecdf(y_measured, y_predicted, output_folder, clim, ml_model):
    fig, ax = plt.subplots()
    sns.ecdfplot(x=y_measured.ravel(), ax=ax)
    sns.ecdfplot(x=y_predicted.ravel(), ax=ax)
    ax.set_xlabel('LWdn, W/m$^2$')
    ax.legend(['measured','predicted'])
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.title(ml_model)
    fname = os.path.join(output_folder, 'LWdn_' + clim + '_ecdf_' + ml_model + '.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_30a_timeseries(y, output_folder, climate_name):
    fig, ax = plt.subplots()
    x = np.arange(y.shape[0]) / 8760.0
    ax.plot(x, y)
    ax.set_xlabel('t, a')
    ax.set_ylabel('LWdn, W/m$^2$')
    ax.grid(True)
    ax.set_axisbelow(True)
    fname = os.path.join(output_folder,
                         'LWdn_30a_' + climate_name + '.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)


def calculate_yearly_means(y, climate_name, log_file_handle):
    
    # yearly means
    yearly_means = np.reshape(y, (8760, -1), order='F').mean(axis=0)
    
    print('Yearly means:')
    print(climate_name)
    print(yearly_means.round(2))
    
    print('Yearly means:', file=log_file_handle)
    print(climate_name, file=log_file_handle)
    np.savetxt(log_file_handle, yearly_means[None, :], fmt='%.2f')
    
    print('\n')
    print('\n', file=log_file_handle)
    

def calculate_monthly_means(y, climate_name, log_file_handle):
    
    n_years = int(len(y) / 8760)
    
    X = np.zeros((12, n_years))
    
    for idx_year in range(n_years):
        
        for idx_month in range(12):
            
            idx_start = idx_year*8760 + idx_month*730
            idx_end = idx_year*8760 +(idx_month+1)*730
            
            print(idx_year, idx_month, idx_start, idx_end)
            X[idx_month, idx_year] = np.mean( y[idx_start:idx_end] )
            
    print('Monthly means:')
    # print(climate_name)
    # print(X.round(2))
    
    print('Monthly means:', file=log_file_handle)
    print(climate_name, file=log_file_handle)
    np.savetxt(log_file_handle, X, fmt='%.2f')
    
    print('\n', file=log_file_handle)
    
    print(X.mean(axis=1))
    print(X.mean(axis=1), file=log_file_handle)
    
    print('\n')
    print('\n', file=log_file_handle)
    

def export_LWdn_to_csv(y, output_folder, climate_name):
    
    fname = os.path.join(output_folder, 'LWdn_' + climate_name + '.csv')
    np.savetxt(fname, y, fmt='%.2f')


                        



#################################


## Input data
with open('./input/data_RASMI.pickle', 'rb') as f:
    data = pickle.load(f)


output_folder = r'./output'

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)


location = 'Sod'

clim = location + '_1989-2018'

t_now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
log_file = os.path.join(output_folder,
                        'log_' + clim + t_now_str + '.txt')
f_log = open(log_file, 'w')

print('Start!', file=f_log)


# Van 	60.33	24.96	51
# Jok 	60.81	23.5	104
# Jyv 	62.4	25.67	139
# Sod 	67.37	26.63	179

if 'Van' in clim:
    LAT_deg = 60.33
    LON_deg = 24.96

elif 'Jok' in clim:
    LAT_deg = 60.81
    LON_deg = 23.50

elif 'Jyv' in clim:
    LAT_deg = 62.40
    LON_deg = 25.67

elif 'Sod' in clim:
    LAT_deg = 67.37
    LON_deg = 26.63
    
else:
    print('Error in location!')
    LAT_deg = np.nan
    LON_deg = np.nan



#################################


X_all, y_all = create_X_y(data, clim, LAT_deg, LON_deg)


idx_split = int(X_all.shape[0] * (27.0/30.0))
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
print(ml_model, file=f_log)
model = DummyRegressor()
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print_to_screen_and_file(y_test, y_pred[ml_model], f_log)
plot_scatter(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_timeseries(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_ecdf(y_test, y_pred[ml_model], output_folder, clim, ml_model)









# Mundt-Petersen & Wallenten
ml_model = 'MPW'
print(ml_model)
print(ml_model, file=f_log)
y_pred[ml_model] = MPW(X_test[:, 0], X_test[:, 1], X_test[:,-1])

print_to_screen_and_file(y_test, y_pred[ml_model], f_log)
plot_scatter(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_timeseries(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_ecdf(y_test, y_pred[ml_model], output_folder, clim, ml_model)







# LinearRegression
ml_model = 'LinearRegression'
print(ml_model)
print(ml_model, file=f_log)
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print_to_screen_and_file(y_test, y_pred[ml_model], f_log)
plot_scatter(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_timeseries(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_ecdf(y_test, y_pred[ml_model], output_folder, clim, ml_model)







# Elastic Net
ml_model = 'ElasticNet'
print(ml_model)
print(ml_model, file=f_log)
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

print('Best params:\n', model.best_params_, file=f_log)

print_to_screen_and_file(y_test, y_pred[ml_model], f_log)
plot_scatter(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_timeseries(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_ecdf(y_test, y_pred[ml_model], output_folder, clim, ml_model)








# Random Forest
ml_model = 'RandomForestRegressor'
print(ml_model)
print(ml_model, file=f_log)
reg_model = RandomForestRegressor()
param_distributions = {'n_estimators': ss_randint(10, 50),
                       'max_depth': ss_randint(3, 5),
                       'criterion': ['mae'],
                       'n_jobs': [1],
                       'max_samples': [1000]}
tss = TimeSeriesSplit(n_splits=5)
model = RandomizedSearchCV(estimator=reg_model,
                        param_distributions=param_distributions,
                        n_iter=20,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        refit=True)
model.fit(X_train_scaled, y_train_scaled.ravel())
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_, file=f_log)

print_to_screen_and_file(y_test, y_pred[ml_model], f_log)
plot_scatter(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_timeseries(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_ecdf(y_test, y_pred[ml_model], output_folder, clim, ml_model)







# xgboost
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
                        n_iter=20,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        refit=True)
model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_, file=f_log)

print_to_screen_and_file(y_test, y_pred[ml_model], f_log)
plot_scatter(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_timeseries(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_ecdf(y_test, y_pred[ml_model], output_folder, clim, ml_model)








# lightgbm
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
                        n_iter=20,
                        cv=tss,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        refit=True)
model.fit(X_train_scaled, y_train_scaled.ravel())
y_pred_scaled = model.predict(X_test_scaled)
y_pred[ml_model] = scaler_y.inverse_transform(y_pred_scaled)

print('Best params:\n', model.best_params_, file=f_log)

print_to_screen_and_file(y_test, y_pred[ml_model], f_log)
plot_scatter(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_timeseries(y_test, y_pred[ml_model], output_folder, clim, ml_model)
plot_ecdf(y_test, y_pred[ml_model], output_folder, clim, ml_model)









###################################

## Predict LWdn for future climates

# Make predictions
for key in data:
    if location in key and 'RCP' in key:
        # We have a future climate
        # the ML model uesd here is the last one of the above list (LightGBM)
        
        X_all, y_all = create_X_y(data, key, LAT_deg, LON_deg)
        
        X_test_scaled = scaler_X.transform(X_all)
        
        y_pred_scaled = model.predict(X_test_scaled)
        
        y_pred[key] = scaler_y.inverse_transform(y_pred_scaled)
        
        
# Make plots
plot_30a_timeseries(data[clim].loc[:,'LWdn_era5'].ravel(), output_folder, clim)
for key in y_pred:
    if location in key and 'RCP' in key:
        plot_30a_timeseries(y_pred[key], output_folder, key)


# calculate yearly means
calculate_yearly_means(data[clim].loc[:, 'LWdn_era5'].values, clim, f_log)
for key in y_pred:
    if location in key and 'RCP' in key:
        calculate_yearly_means(y_pred[key], key, f_log)


# calculate monthly means
calculate_monthly_means(data[clim].loc[:,'LWdn_era5'], clim, f_log)
for key in y_pred:
    if location in key and 'RCP' in key:
        calculate_monthly_means(y_pred[key], key, f_log)



## Close log file
print('End', file=f_log)
f_log.close()


###################################

## Export LWdn data to csv files


export_LWdn_to_csv(data[clim].loc[:,'LWdn_era5'], output_folder, clim)
for key in y_pred:
    if location in key and 'RCP' in key:
        print('To csv', key)
        export_LWdn_to_csv(y_pred[key], output_folder, key)








