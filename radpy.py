# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 22:46:41 2021

@author: laukkara



# 'plot_folder_path': './output_plots'

import radpy

t_local_hour = np.arange(8760)
kwargs = {'LAT_deg': 65.0,
          'LON_deg': 24.5,
          'tz': 2.0,
          'plot_folder_path': None}

radpy.calc_slar_position(t_local_hour, kwargs)

"""




import os
import numpy as np
import matplotlib.pyplot as plt



def calc_psat(T):
    # calculate water vapour saturation pressure over liquid water
    p_sat = 610.5 * np.exp((17.269*T)/(237.3+T))
    return(p_sat)    


def calc_Tdew(p):
    T_dew = (237.3*np.log(p/610.5)) / (17.269 - np.log(p/610.5))
    return(T_dew)




def calc_solar_position(t_local_hour, kwargs):

    # ASHRAE Handbook Fundamentals 2017, Ch. 14 Climatic Design Information
    
    n_day = t_local_hour / 24.0 + 1.0
    
    # Equation of time, the results are in minutes
    Gamma_rad = 2*np.pi * (n_day-1)/365.0
    dummy = 0.0075 \
            + 0.1868*np.cos(Gamma_rad) \
            - 3.2077*np.sin(Gamma_rad) \
            - 1.4615*np.cos(2*Gamma_rad) \
            - 4.089*np.sin(2*Gamma_rad)
    ET = 2.2918 * dummy
    
    # Apparent solar time
    LST = t_local_hour % 24 # Always local standard time i.e. winter time
    LON_deg = kwargs['LON_deg']
    LSM_deg = 15.0 * kwargs['tz']
    AST = LST + ET/60.0 + (LON_deg - LSM_deg)/15.0
    
    # Declination
    delta_deg = 23.45 * np.sin(2.0*np.pi * (n_day+284.0)/365.0)
    delta_rad = (np.pi/180.0) * delta_deg
    
    # Hour angle, zero at solar noon, postive in the afternoon
    H_deg = 15.0 * (AST - 12.0)
    H_rad = (np.pi/180.0) * H_deg
    
    # Local latitude
    LAT_rad = (np.pi/180.0) * kwargs['LAT_deg']
    
    # Altitude angle, 0 deg for sun at horizon, 90 deg for directly overhead
    sin_beta = np.cos(LAT_rad) * np.cos(delta_rad) * np.cos(H_rad) \
                + np.sin(LAT_rad) * np.sin(delta_rad)
    beta_rad = np.arcsin(sin_beta)
    beta_deg = (180.0/np.pi) * beta_rad
    
    # Azimuth angle, counted positive for afternoon hours
    # and negative for morning hours
    # dividing with zero is not handled separately
    sin_phi = np.sin(H_rad) * np.cos(delta_rad) / np.cos(beta_rad)
    
    cos_phi = (np.cos(H_rad)*np.cos(delta_rad)*np.sin(LAT_rad) \
                - np.sin(delta_rad)*np.cos(LAT_rad)) / np.cos(beta_rad)
    
    phi_rad = np.zeros(len(sin_phi) )
        
    for idx, (val_sin_phi, val_cos_phi) in enumerate(zip(sin_phi, cos_phi)):
        # numpy arcsin() returns values in the closed range [-pi/2, pi/2]
        # numpy arccos() returns values in the closed range [0, pi]
        
        if val_sin_phi >= 0.0:
            # afternoon
            phi_rad[idx] = np.arccos(val_cos_phi)

        else:
            # morning
            phi_rad[idx] = -np.arccos(val_cos_phi)
        
    phi_deg = (180.0/np.pi) * phi_rad
    
    
    ## make plots
    
    if kwargs['plot_folder_path'] is not None:
        # If path is given, make the plots
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])
        
        # ET
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, ET)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$ET$, min')
        fname = os.path.join(kwargs['plot_folder_path'], 'ET.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # declimation
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, delta_deg)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$\delta, \degree$')
        fname = os.path.join(kwargs['plot_folder_path'], 'delta_deg.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # local solar altitude angle beta, positive when sun is above horizon
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, beta_deg)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$\\beta, \degree$')
        fname = os.path.join(kwargs['plot_folder_path'], 'beta_deg.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # local azimuth angle phi, positive in the afternoon
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, phi_deg)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$\phi, \degree$')
        fname = os.path.join(kwargs['plot_folder_path'], 'phi_deg.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    
    # Return
    return(beta_deg, phi_deg)




def calc_extraterrestrial_radiant_flux(t_local_hour, kwargs):
    
    # ASHRAE Handbook Fundamentals 2017, Ch. 14 Climatic Design Information
    
    # Solar constant
    E_sc = 1367.0
    
    # Extraterrestrial radiant flux
    n_day = t_local_hour / 24.0 + 1.0
    E_o = E_sc * (1.0 + 0.033 * np.cos(2*np.pi*((n_day-3)/365.0)))


    if kwargs['plot_folder_path'] is not None:
        # If path is given, make the plots
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])

        # E_o
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, E_o)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$E_o$, W/m$^2$')
        fname = os.path.join(kwargs['plot_folder_path'], 'E_o.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    return(E_o)





def calc_Rglob_vs_Rmax():
    pass




