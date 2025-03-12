# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 21:27:51 2025

@author: Anssi Laukkarinen

This file contains code that reads in the fitted machine learning models and
calculates the predicted hourly LWdn values for future climatic conditions.

This file is run second.
This file is run once, after the first file is run four times with 
different locations.

"""

import os
import pickle
import pandas as pd

import read_in_data
import helper



root_folder = r'C:\Users\laukkara\github\LWdn'

folder_rasmi = os.path.join(root_folder,
                            'input',
                            'RASMI')

folder_era5 = os.path.join(root_folder,
                           'input',
                           'era5_strd_rami_20211005')


output_folder = os.path.join(root_folder,
                             'output')

output_folder_predicted = os.path.join(output_folder,
                                 'RASMI_ERA5_ML')
if not os.path.exists(output_folder_predicted):
    os.makedirs(output_folder_predicted)


t_now_str = pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M-%S')
log_file = os.path.join(output_folder_predicted,
                        'log_prediction_' + '_' + t_now_str + '.txt')
f_log = open(log_file, 'w')

print('Start!', file=f_log)




###################



# location = 'Jok'
# for location in ['Jok']:
for location in ['Van', 'Jok', 'Jyv', 'Sod']:
    
    print(location, file=f_log)
    
    # Here 'data' has index in UTC time (31.12. 22:00, 23:00, ...)
    data = read_in_data.read_RASMI_and_ERA5(folder_rasmi,
                                            folder_era5,
                                            location)

    
    fname = os.path.join(root_folder,
                         'output',
                         f'LGBMRegressor_bestModel_scalerX_scalery_LWdn_{location}.pickle')
    with open(fname, 'rb') as f:
        model, scaler_X, scaler_y = pickle.load(f)
    
    
    LAT_deg, LON_deg = helper.get_LAT_and_LON(location)
    
    
    
    ###############################
    
    
    ## Predict LWdn for future climates
    
    print('Make predictions...', file=f_log)
    
    for key in data:
    
        if location in key and 'RCP' in key:
            # We have a future climate
            
            X_all, y_all = helper.create_X_y(data[key], LAT_deg, LON_deg)
            
            X_all_scaled = scaler_X.transform(X_all)
            
            y_all_scaled = model.predict(X_all_scaled)
            y_all_scaled = y_all_scaled.reshape(-1,1)
            
            y_all_pred = scaler_y.inverse_transform(y_all_scaled)
            
            data[key].loc[:,'LWdn'] = y_all_pred
        
        # Change to Finnish normal time, i.e. winter time
        data[key].index = data[key].index + pd.Timedelta('2h')
        data[key].index.name = 't_fin'
    
    
    
    ## Plots and exports
    
    print('Make 30 a time series plots...', file=f_log)
    
    for key in data:
        if location in key:
            helper.plot_30a_timeseries(data[key].loc[:,'LWdn'], 
                                       output_folder_predicted, 
                                       key)
    
    
    print('Calculate yearly means...', file=f_log)
    
    for key in data:
        print(key, flush=True)
        if location in key:
            helper.calculate_yearly_means(data[key].loc[:,'LWdn'],
                                          key, 
                                          f_log)
    
    
    print('Calculate monthly means...', file=f_log)
    
    for key in data:
        if location in key:
            helper.calculate_monthly_means(data[key].loc[:,'LWdn'], 
                                           key, 
                                           f_log)
    
    
    
    print('Export LWdn data to csv files...', file=f_log)
    
    for key in data:
        
        if location in key:
        
            fname = os.path.join(output_folder_predicted,
                                 f'{key}.csv')
            data[key].to_csv(fname,
                             float_format='%.2f')
    
    

## Close log file
print('End', file=f_log)
f_log.close()
