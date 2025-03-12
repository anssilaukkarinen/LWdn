# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 22:23:05 2025

@author: Anssi Laukkarinen

This file contains functions that are used in other modules.

In RASMI data radiation and precipitation are cumulative values from the
previous time stamp.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score





def get_LAT_and_LON(location_):
    
    if 'Van' in location_:
        LAT_deg = 60.33
        LON_deg = 24.96
    
    elif 'Jok' in location_:
        LAT_deg = 60.81
        LON_deg = 23.50
    
    elif 'Jyv' in location_:
        LAT_deg = 62.40
        LON_deg = 25.67
    
    elif 'Sod' in location_:
        LAT_deg = 67.37
        LON_deg = 26.63
        
    else:
        print('Error in location!')
        LAT_deg = np.nan
        LON_deg = np.nan
    
    return(LAT_deg, LON_deg)


def get_name_converter_dict(TEMP_to_Te=True):
    
    if TEMP_to_Te:
        
        name_converter_dict_ = {'YEAR': 'year',
                                'MON': 'month',
                                'DAY': 'day',
                                'HOUR': 'hour',
                                'TEMP': 'Te',
                                'RH': 'RHe_water',
                                'WS': 'ws',
                                'WDIR': 'wd',
                                'GHI': 'Rglob',
                                'DHI': 'Rdif',
                                'DNI': 'Rbeam',
                                'PRECIP': 'precip'}
        
    else:
        
        name_converter_dict_ = {'year': 'YEAR',
                                'month': 'MON',
                                'day': 'DAY',
                                'hour': 'HOUR',
                                'Te': 'TEMP',
                                'RHe_water': 'RH',
                                'ws': 'WS',
                                'wd': 'WDIR',
                                'Rglob': 'GHI',
                                'Rdif': 'DHI',
                                'Rbeam': 'DNI',
                                'precip': 'PRECIP'}
    
    return(name_converter_dict_)





def calc_pvsat(T):
    # calculate water vapour saturation pressure over liquid water
    # The relative humidity values are given with respect to liquid water
    # also in subzero temperatures
    p_sat = 610.5 * np.exp((17.269*T)/(237.3+T))
    return(p_sat)    


def calc_Tdew(p):
    T_dew = (237.3*np.log(p/610.5)) / (17.269 - np.log(p/610.5))
    return(T_dew)




def create_X_y(df, LAT_deg, LON_deg):
    
    n_hours = df.loc[:,'Te'].shape[0]
    
    t_hour = np.arange(n_hours) - 0.5
    
    # Interpolate T_e, RHe_water to previous half hours
    # to match radiation values
    x = t_hour
    xp = np.arange(n_hours)
    fp = df.loc[:, 'Te'].values
    T_e = np.interp(x, xp, fp)
    
    x = t_hour
    xp = np.arange(n_hours)
    fp = df.loc[:,'RHe_water'].values
    RHe_water = np.interp(x, xp, fp)
    pv = (RHe_water/100.0) * calc_pvsat(T_e)
    T_dew = calc_Tdew(pv)
    
    
    solar_position = pvlib.solarposition.get_solarposition(
                            time=df.index - pd.Timedelta('30min'),
                            latitude=LAT_deg,
                            longitude=LON_deg,
                            temperature=df.loc[:,'Te'].mean())
    
    R_glob = df.loc[:,'Rglob']
    
    R_extraterrestrial = pvlib.irradiance.get_extra_radiation(
                                datetime_or_doy=df.index-pd.Timedelta('30min'))
    
    R_hor_max = R_extraterrestrial \
                * np.maximum(0.0, 
                             np.cos( solar_position['zenith']*(np.pi/180.0) ))
                            
    solar_position.index = df.index
    R_extraterrestrial.index = df.index
    R_hor_max.index = df.index
    
    method_selector = 'method2'
    
    if method_selector == 'method1':
        # Using pvlib function
        # There is zero values during night and this did not seem as
        # accurate approach, as method2.
        
        # extra_radiation is beam radiation at the top-of-atmosphere,
        # and zenith angle is given separately.
        # MAE (Van): 17...19-27 W/m2
        # MAE (Jok): 17...19-25 W/m2
        
        solar_zenith_deg = solar_position['zenith']
        K_0 = pvlib.irradiance.clearness_index(ghi=R_glob,
                                               solar_zenith=solar_zenith_deg,
                                               extra_radiation=R_extraterrestrial,
                                               max_clearness_index=1.0)
    
    elif method_selector == 'method2':
        # This produced slightly better prediction accuracy
        K_0 = calc_K_0(R_glob, R_hor_max)
    
    

    if 'LWdn' in df.columns:
        LW_dn = df.loc[:, 'LWdn']
        
    else:
        n_shape = df.loc[:, 'Te'].shape
        LW_dn = pd.DataFrame(np.zeros(n_shape))
    
    
    X_all = np.column_stack( (T_e,
                              T_dew,
                              R_glob,
                              R_hor_max,
                              K_0) )
    
    y_all = LW_dn.values.reshape(-1,1)
    
    return(X_all, y_all)




def calc_K_0(R_glob_, R_extra_):
    # Extraterrestrial radiation is given on the horizontal surface
    # at ground level (assuming no atmosphere).
    # Time step is always one hour, so it isn't written out.
    
    n_days = int(len(R_glob_) / 24)
    
    K_0_days = np.zeros(shape=(n_days*2, 2))
    
    # These are initial values, which are then updated as calculation
    # progresses.
    t_midpoint_morning = 9.0
    t_midpoint_evening = 15.0
    
    for idx_day in range(n_days):
        
        # Morning
        idxs_morning = np.arange(idx_day*24,idx_day*24+13)
        
        R_glob_morning = np.maximum(0.0, R_glob_.iloc[idxs_morning])
        R_extra_morning = np.maximum(0.0, R_extra_.iloc[idxs_morning])
        
        R_glob_morning_sum = np.sum(R_glob_morning)
        R_extra_morning_sum = np.sum(R_extra_morning)
        
        if R_extra_morning_sum > 0.0:
            
            t_midpoint_morning = 12 - np.sum(R_extra_morning > 0.0)/2.0
            K_0_days[idx_day*2, 1] = R_glob_morning_sum / R_extra_morning_sum
        
        else:
            K_0_days[idx_day*2, 1] = 0.5
        
        K_0_days[idx_day*2, 0] = idx_day*24 + t_midpoint_morning
        
        
        # Evening
        idxs_evening = np.arange(idx_day*24+13, (idx_day+1)*24)
        
        R_glob_evening = np.maximum(0.0, R_glob_.iloc[idxs_evening])
        R_extra_evening = np.maximum(0.0, R_extra_.iloc[idxs_evening])
        
        R_glob_evening_sum = np.sum(R_glob_evening)
        R_extra_evening_sum = np.sum(R_extra_evening)
        
        if R_extra_evening_sum > 0.0:
            
            t_midpoint_evening = 13 + np.sum(R_extra_evening > 0.0)/2
            K_0_days[idx_day*2+1, 1] = R_glob_evening_sum / R_extra_evening_sum
        
        else:
            K_0_days[idx_day*2+1, 1] = 0.5
        
        K_0_days[idx_day*2+1, 0] = idx_day*24 + t_midpoint_evening
        
    t_hourly = np.arange(0, n_days*24)
    K_0_hourly = np.interp(t_hourly,
                           K_0_days[:,0], 
                           K_0_days[:,1])
    
    K_0_hourly = np.maximum(0.0, K_0_hourly)
    K_0_hourly = np.minimum(1.0, K_0_hourly)
        
    return(K_0_hourly)
        





def K_model(T_e, T_dew, K_0):
    # [T_e] = degC, [T_dew] = degC, K_0 is clearness index [0...1]
    print('Initiating K-model...')
    
    sigma_SB = 5.67e-8
    
    epsilon_sky = 1.5357 \
                  + 0.5981 * (T_dew/100.0) \
                  - 0.5687 * ((T_e + 273.15)/273.15) \
                  - 0.2799 * K_0
    

    LW_dn_K_model = epsilon_sky * sigma_SB * (T_e + 273.15)**4
    
    print('Finished K-model!')
    
    return(LW_dn_K_model)




def print_and_plot(ml_model,
                    clim,
                    y_test,
                    y_pred_ml_model,
                    f_log,
                    output_folder):
    
    print_to_screen_and_file(y_test, y_pred_ml_model, f_log)
    
    plot_scatter(y_test, y_pred_ml_model, output_folder, clim, ml_model)
    
    plot_timeseries(y_test, y_pred_ml_model, output_folder, clim, ml_model)
    
    plot_ecdf(y_test, y_pred_ml_model, output_folder, clim, ml_model)



def print_to_screen_and_file(y_measured, y_predicted, log_file_handle):
    
    mae = mean_absolute_error(y_measured, y_predicted)
    print('  MAE: {:.3f}'.format(mae), file=log_file_handle)
    
    r2 = r2_score(y_measured, y_predicted)
    print('  R2: {:.3f}'.format(r2), file=log_file_handle)
    
    

def plot_scatter(y_measured, y_predicted, output_folder, clim, ml_model):
    fig, ax = plt.subplots()
    ax.plot(y_measured, y_predicted, '.', label='data', ms=0.4)
    ax.plot([150,400],[150,400], label=r'$y = x$')
    ax.set_xlabel('ERA5, W/m$^2$')
    ax.set_ylabel('Predicted, W/m$^2$')
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(ml_model)
    fname = os.path.join(output_folder,
                         'LWdn_' + clim + '_scatter_' + ml_model + '.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_timeseries(y_measured, y_predicted, output_folder, clim, ml_model):
    fig, ax = plt.subplots()
    ax.plot(np.arange(8760), y_measured[-8760:], label='ERA5', linewidth=0.6)
    ax.plot(np.arange(8760), y_predicted[-8760:], label='Predicted', linewidth=0.6)
    ax.set_xlabel('t, h')
    ax.set_ylabel('LWdn, W/m$^2$')
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(ml_model)
    fname = os.path.join(output_folder, 
                         'LWdn_' + clim + '_timeseries_' + ml_model + '.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_ecdf(y_measured, y_predicted, output_folder, clim, ml_model):
    fig, ax = plt.subplots()
    ax.ecdf(x=y_measured.ravel(), label='ERA5')
    ax.ecdf(x=y_predicted.ravel(), label='predicted')
    ax.set_xlabel('LWdn, W/m$^2$')
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(ml_model)
    fname = os.path.join(output_folder,
                         'LWdn_' + clim + '_ecdf_' + ml_model + '.png')
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


def calculate_yearly_means(df_, climate_name, log_file_handle):
    
    yearly_means = df_.groupby(by=df_.index.year) \
                    .mean()
    
    print('Yearly means:', file=log_file_handle)
    print(climate_name, file=log_file_handle)
    yearly_means.to_csv(log_file_handle, 
                        mode='a',
                        float_format='%.2f',
                        lineterminator='\n')
    
    print('\n', file=log_file_handle)
    


def calculate_monthly_means(df_, climate_name, log_file_handle):
    
    
    monthly_means = df_.groupby(by=[df_.index.year,
                                    df_.index.month]) \
                        .mean()
                        
    X = np.reshape(monthly_means,
                   shape=(12,-1),
                   order='F')
    
    print('Monthly means:', file=log_file_handle)
    print(climate_name, file=log_file_handle)
    np.savetxt(log_file_handle, X, fmt='%.2f')
    print('\n', file=log_file_handle)

    





