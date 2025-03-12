# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:38:48 2021, updated 2025

@author: Anssi Laukkarinen

The function reads in RASMI data and the LWdn radiation data for the
current climate, combines them and returns everything as a 
python dictionary.

This is a helper module that is currently not run directly,
but it is called from another file.

"""

import os
import numpy as np
import pandas as pd

import helper



def read_RASMI_and_ERA5(folder_rasmi,
                        folder_era5,
                        location):
    
    name_converter_dict = helper.get_name_converter_dict(TEMP_to_Te=True)
    
    file_names_era5 = {'Van':'Helsinki-Vantaa_era5_strd.csv',
                  'Jok':'Jokioinen_era5_strd.csv',
                  'Jyv':'Jyvaskyla_era5_strd.csv',
                  'Sod':'Sodankyla_era5_strd.csv'}
    
    
    ## Read in RASMI
    
    print('Reading RASMI files...')
    
    
    files = [file for file in os.listdir(folder_rasmi) \
             if file.endswith('.prn') \
             and location in file]
    
    data_rasmi = {}
    
    for file in files:
        
        print(file)
        
        key = file.replace('_tuntidata30v', '').replace('.prn', '')
        key = key.replace('Jokioinen', 'Jok')
        key = key.replace('Jyvaskyla', 'Jyv')
        key = key.replace('Sodankyla', 'Sod')
        key = key.replace('Vantaa', 'Van')
        
        key = key.replace('RCP26_2', 'RCP26-2')
        key = key.replace('RCP45_2', 'RCP45-2')
        key = key.replace('RCP85_2', 'RCP85-2')
        
        fname = os.path.join(folder_rasmi, file)
        data_rasmi[key] = pd.read_csv(fname,
                                         sep='\s+',
                                         header=1,
                                         dtype=np.float64)
        
        # Rename columns so that working with pandas is easier
        data_rasmi[key].rename(columns=name_converter_dict,
                               inplace=True)
    
        
        
        # Create UTC index
        # The RASMI data is in the Finnish normal time (winter time, UTC+2)
        # pvlib assumes UTC, if not otherwise specified
        data_rasmi[key].index \
            = pd.to_datetime(data_rasmi[key][['year','month','day','hour']])
            
        data_rasmi[key].drop(columns=['STEP','year','month','day','hour'],
                             inplace=True)
        
        data_rasmi[key].index \
            = data_rasmi[key].index - pd.Timedelta('2h')
        
        data_rasmi[key].index.name == 't_utc'
    
    
    print('RASMI files read')
    
    
    
    ## Read in ERA5
    print('Reading ERA5 files...')
    
    data_era5 = {}
    
    for key_era5 in file_names_era5:
        print(key_era5, file_names_era5[key_era5])
        
        fname = os.path.join(folder_era5,
                             file_names_era5[key_era5])
        
        data_era5[key_era5] = pd.read_csv(fname,
                                        sep=';',
                                        index_col=0,
                                        parse_dates=True,
                                        comment='#')
    
        data_era5[key_era5].index.name == 't_utc'
    
    print('ERA5 files read')
    
    
    
        
    ## Combine data
    print('Combining RASMI and ERA5 data...')
    
    data = data_rasmi.copy()
    
    clim = location + '_1989-2018'
    
    idxs = data_rasmi[clim].index
    
    data[clim].loc[:, 'LWdn'] \
        = data_era5[location].loc[idxs, 'strd']
    
    
    return(data)



