# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 22:46:41 2021

Source: ASHRAE Handbook Fundamentals 2017

@author: laukkara

# Van 	60.33	24.96	51
# Jok 	60.81	23.5	104
# Jyv 	62.4	25.67	139
# Sod 	67.37	26.63	179


# 'plot_folder_path': './output_plots'

import radpy

t_local_hour = np.arange(8760)
kwargs = {'LAT_deg': 65.0,
          'LON_deg': 24.5,
          'tz': 2.0,
          'plot_folder_path': None,
          'plot_file_name': 'Toni'}

radpy.calc_solar_position(t_local_hour, kwargs)

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
    
    if kwargs['plot_figures_bool']:
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])
        
        # ET
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, ET)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$ET$, min')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_ET.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        
        # declimation
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, delta_deg)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$\delta, \degree$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_delta_deg.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        
        # local solar altitude angle beta, positive when sun is above horizon
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, beta_deg)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$\\beta, \degree$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_beta_deg.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        
        # local azimuth angle phi, positive in the afternoon
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, phi_deg)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$\phi, \degree$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_phi_deg.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        
        
    if kwargs['make_csv_files_bool']:
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])
        
        # export sun position data to csv file
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_sun_position_angles.csv')
        np.savetxt(fname,
                   np.column_stack((t_local_hour,
                                    ET,
                                    delta_deg,
                                    beta_deg,
                                    phi_deg)),
                   fmt='%-10.3f',
                   delimiter=',',
                   comments='',
                   header='t_local_hour, ET_minutes, delta_deg, beta_deg, phi_deg')
        
    # Return
    return(beta_deg, phi_deg)




def calc_extraterrestrial_radiant_flux(t_local_hour, kwargs):
    
    # ASHRAE Handbook Fundamentals 2017, Ch. 14 Climatic Design Information
    
    # Solar constant
    E_sc = 1367.0
    
    # Extraterrestrial radiant flux
    n_day = t_local_hour / 24.0 + 1.0
    E_o = E_sc * (1.0 + 0.033 * np.cos(2*np.pi*((n_day-3)/365.0)))


    if kwargs['plot_figures_bool']:
        # If path is given, make the plots
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])

        # E_o
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, E_o)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$E_o$, W/m$^2$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_E_o.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        
        
    if kwargs['make_csv_files_bool']:
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])
        
        np.savetxt(fname.replace('png', 'csv'),
                   np.column_stack((t_local_hour, E_o)),
                   fmt='%-10.3f',
                   delimiter=',',
                   header='t_local_hour, E_0_W_per_m2')
        
    
    return(E_o)





def calc_clearness_index(R_glob, R_hor_max):
    # Two values per day, then interpolate
    
    n_days = len(R_glob) // 24
    
    S_day = []
    t_day = []
    
    for idx_day in range(n_days):
        # loop through data in 24 hour steps
        
        # start and end indexis for morning and afternoon
        idx_morning_start = idx_day*24
        idx_morning_end = idx_day*24 + 13
        idx_afternoon_start = idx_day*24 + 13
        idx_afternoon_end = idx_day*24 + 24
        
        # Take global radiation and no-atmosphere horizontal radiation to 
        # separate variables
        R_glob_morning = R_glob[idx_morning_start:idx_morning_end]
        R_glob_afternoon = R_glob[idx_afternoon_start:idx_afternoon_end]
        R_hor_max_morning = R_hor_max[idx_morning_start:idx_morning_end]
        R_hor_max_afternoon = R_hor_max[idx_afternoon_start:idx_afternoon_end]
        
        # Calculate the number of hours (integer value) that the sun would 
        # ideally be above the horizon 
        n_hours_above_horizon_morning = len( R_hor_max_morning[R_hor_max_morning > 5.0] )
        n_hours_above_horizon_afternoon = len( R_hor_max_afternoon[R_hor_max_afternoon > 5.0] )
        
        if n_hours_above_horizon_morning == 0:
            # If the sun doesn't rise above the horizon, use NaN
            #S_day.append(np.nan)
            #S_day.append(np.nan)
            S_day.append(0.5)
            S_day.append(0.5)
            
            
            t_day.append(idx_day*24+12)
            t_day.append(idx_day*24+13)
        
        else:
            # If the sun does rise above the horizon in a particular day,
            # calculate the mean radiation for daylight time and its quotient
            dummy_morning_nominator = np.sum(R_glob_morning[R_glob_morning>0.0]) / n_hours_above_horizon_morning
            dummy_morning_denominator = np.sum(R_hor_max_morning[R_hor_max_morning>0.0]) / n_hours_above_horizon_morning
            dummy_morning = dummy_morning_nominator / dummy_morning_denominator
            
            dummy_afternoon_nominator = np.sum(R_glob_afternoon[R_glob_afternoon>0.0]) / n_hours_above_horizon_afternoon
            dummy_afternoon_denominator = np.sum(R_hor_max_afternoon[R_hor_max_afternoon>0.0]) / n_hours_above_horizon_afternoon
            dummy_afternoon = dummy_afternoon_nominator / dummy_afternoon_denominator
            
            S_day.append(dummy_morning)
            S_day.append(dummy_afternoon)
            t_day.append(idx_day*24 + 12 - n_hours_above_horizon_morning/2.0)
            t_day.append(idx_day*24 + 13 + n_hours_above_horizon_afternoon/2.0)
            
            
    # After we know the two-times-per-day vaues, interpolate the data to
    # hourly values
    
    x = np.arange(len(R_glob))
    xp = np.array(t_day)
    fp = np.array(S_day)
    S_hourly = np.interp(x, xp, fp)
    
    return(S_hourly)


        
def calc_angle_of_incidence(t_local_hour, kwargs):
    # kwargs for this function contains the same
    # dict keywords than what are needed for the 
    # calc_solar_position() function
    
    # Additional keyword arguments are:
    # phi_deg = solar azimuth
    # psi_deg = surface azimuth angle, positive to west from south
    # gamma_deg = surface-solar azimuth
    # Sigma_deg = tilt angle, 0 deg for horizontal, 90 deg for vertical surface
    # theta_deg = angle of incidence, angle between
    # surface normal and earth-sun line
    
    ## Angle of incidence
    beta_deg, phi_deg = calc_solar_position(t_local_hour, kwargs)
    
    gamma_deg = phi_deg - kwargs['psi_deg']
    
    beta_rad = beta_deg * (np.pi/180.0)
    gamma_rad = gamma_deg * (np.pi/180.0)
    Sigma_rad = kwargs['Sigma_deg'] * (np.pi/180.0)
    
    cos_theta = np.cos(beta_rad) \
                * np.cos(gamma_rad) \
                * np.sin(Sigma_rad) \
                + np.sin(beta_rad) \
                * np.cos(Sigma_rad)
    
    theta_rad = np.arccos(cos_theta)
    theta_deg = theta_rad * (180.0/np.pi)
    
    
    # theta_limited to -90...90 deg
    theta_deg_limited = np.zeros(theta_deg.shape)
    cos_theta = np.zeros(theta_deg.shape)
    
    bool_conditions = (beta_deg > 0.0) \
                        & (gamma_deg >-90.0) \
                        & (gamma_deg < 90.0)
    
    for idx in range(len(theta_deg)):
        
        if bool_conditions[idx]:
            # sun is shining towards the surface
            theta_deg_limited[idx] = theta_deg[idx]
            cos_theta[idx] = np.cos(theta_deg[idx] * (np.pi/180.0))
        else:
            # sun is not shining towards the surface
            theta_deg_limited[idx] = 90.0
            cos_theta[idx] = 0.0
    
    
    if kwargs['plot_figures_bool']:
        # If path is given, make the plots
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])
        
        
        # gamma_deg
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, gamma_deg)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$\gamma$, $\degree$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_gamma_deg.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        

        # theta_deg
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, theta_deg)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$\\theta$, $\degree$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_theta_deg.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
    
    
        # theta_deg_limited
        fig, ax = plt.subplots()
        ax.plot(t_local_hour, theta_deg_limited)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$\\theta$, -90...90 $\degree$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_theta_deg_limited.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)
        

    if kwargs['make_csv_files_bool']:
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])
        
        # to csv
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_incidence_angles.csv')
        np.savetxt(fname,
                   np.column_stack((gamma_deg,
                                    theta_deg,
                                    theta_deg_limited,
                                    cos_theta)),
                   fmt='%-10.3f',
                   delimiter=',',
                   header='gamma_deg, theta_deg, theta_deg_limited, cos_theta')


    return(beta_deg, phi_deg, gamma_deg, theta_deg, theta_deg_limited, cos_theta)

    

def calc_solar_components_to_surface(t_local_hour, Rbeam, Rdif, Rglob,
                                     theta_deg_limited,
                                     kwargs):
    
    Sigma_deg = kwargs['Sigma_deg']
    
    
    # Calculate the sum of incoming radiation to surface
    # According to WUFI equations
    
    # beam
    k_cos = np.cos(theta_deg_limited * (np.pi/180.0))
    E_beam_surf = Rbeam * k_cos
    
    # dif
    g_atm = ( np.cos(Sigma_deg * (np.pi/180.0) / 2.0) )**2
    E_dif_surf = g_atm * Rdif
    
    # refl
    g_terr = 1.0 - g_atm
    E_refl_surf = g_terr * 0.2 * Rglob
    
    # total
    E_surf = E_beam_surf + E_dif_surf + E_refl_surf
    
    
    
    if kwargs['plot_figures_bool']:
        # If path is given, make the plots
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])
            
        
        fig, ax = plt.subplots()
        ax.plot(E_beam_surf)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$E_{beam,surf}$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_E_beam_surf.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)

        
        fig, ax = plt.subplots()
        ax.plot(E_dif_surf)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$E_{dif,surf}$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_E_dif_surf.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)


        fig, ax = plt.subplots()
        ax.plot(E_refl_surf)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$E_{refl,surf}$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_E_refl_surf.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)


        fig, ax = plt.subplots()
        ax.plot(E_surf)
        ax.set_xlabel('$t_{hour}$')
        ax.set_ylabel('$E_{surf}$')
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_E_surf.png')
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)


    if kwargs['make_csv_files_bool']:
        
        if not os.path.exists(kwargs['plot_folder_path']):
            os.makedirs(kwargs['plot_folder_path'])
        
        # to csv
        fname = os.path.join(kwargs['plot_folder_path'],
                             kwargs['plot_file_name'] + '_W_per_m2_values.csv')
        np.savetxt(fname,
                   np.column_stack((t_local_hour,
                                    E_beam_surf,
                                    E_dif_surf,
                                    E_refl_surf,
                                    E_surf)),
                   fmt='%-12.2f',
                   delimiter=',',
                   comments='',
                   header='t_local_hour, E_beam_surf, E_dif_surf, E_refl_surf, E_surf')

    
    return(E_beam_surf, E_dif_surf, E_refl_surf, E_surf)