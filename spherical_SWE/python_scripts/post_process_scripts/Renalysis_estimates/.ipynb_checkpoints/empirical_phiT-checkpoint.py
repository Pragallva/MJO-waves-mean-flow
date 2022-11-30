import netCDF4 as nc
import glob as glob
import pylab as py
import numpy as np
import hickle as hkl
import cartopy.crs as ccrs
import cartopy.util as cutil
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
import hickle as hkl
import sys
import logging
import time as ti
import scipy.signal as signal
from netCDF4 import Dataset,num2date,date2num
import datetime
from tqdm import tqdm
import scipy.integrate as integrate

sys.path.append('/data/pbarpanda/python_scripts/modules/')

import logruns as logruns
import save_and_load_hdf5_files as h5saveload
import eulerian_fluxes as eflux
sys.path.append('/data/pbarpanda/python_scripts/rerun_ERAI_data_extraction/Aug11_on_Georges_data_calculate_mean_fields_for_different_MJO_phases/')
import Aug11_netcdf_utilities_Georges_data as ncutil

import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'  ### This is because NOAA PSL lab computers are somehow not able to use this feature


month_strings = ncutil.month_strings
month_string  = ncutil.month_string

def calculate_mean_field(infile):
                
    for keys,values in infile.items():
        globals()['%s'%keys]= values 
                
    i = 0
    pres_lev = [200]
    
    start_time_in = ti.time()
    
    DJF_dates = ncutil.return_dates(YEARS             = np.arange(1979, 2021, 1), \
                                    MONTHS            = [12,1,2], \
                                    all_days_in_month = True)

    
    if not os.path.exists(path+'DJF_winds.hdf5'):
                               
        uwnd, lat, lon, netcdf_dates = ncutil.return_all_field(var = 'uwnd', filename     ='/Projects/era5_regrid/2p5/uwnd.200.2p5.nc', \
                                                                             default_file = '/Projects/era5_regrid/2p5/uwnd.200.2p5.nc')
        logging_object.write("FINISHED READING %s ...."%('uwnd'))  
        netcdf_data = {'netcdf_field':  uwnd, 'netcdf_dates': netcdf_dates, 'lat':lat, 'lon': lon}
        uwnd, lat, lon = ncutil.return_field_in_certain_dates(netcdf_data, mjo_dates = DJF_dates)
        logging_object.write("DJF data loaded for %s ...."%('uwnd'))


        vwnd, lat, lon, netcdf_dates = ncutil.return_all_field(var = 'vwnd', filename     ='/Projects/era5_regrid/2p5/vwnd.200.2p5.nc', \
                                                                             default_file = '/Projects/era5_regrid/2p5/vwnd.200.2p5.nc')
        netcdf_data = {'netcdf_field':  vwnd, 'netcdf_dates': netcdf_dates, 'lat':lat, 'lon': lon}
        vwnd, lat, lon = ncutil.return_field_in_certain_dates(netcdf_data, mjo_dates = DJF_dates)
        logging_object.write("DJF data loaded for %s ...."%('vwnd'))
        logging_object.write("FINISHED READING %s ...."%('vwnd'))  

        vort, lat, lon, netcdf_dates = ncutil.return_all_field(var = 'vort', filename     ='/Projects/era5_regrid/2p5/vort.200.2p5.nc', \
                                                                             default_file = '/Projects/era5_regrid/2p5/vort.200.2p5.nc')
        netcdf_data = {'netcdf_field':  vort, 'netcdf_dates': netcdf_dates, 'lat':lat, 'lon': lon}
        vort, lat, lon = ncutil.return_field_in_certain_dates(netcdf_data, mjo_dates = DJF_dates)
        logging_object.write("DJF data loaded for %s ...."%('vort'))
        logging_object.write("FINISHED READING %s ...."%('vort'))   

        geopot_Z, lat, lon, netcdf_dates = ncutil.return_all_field(var = 'hgt', filename     ='/Projects/era5_regrid/2p5/hgt.200.2p5.nc', \
                                                                   default_file = '/Projects/era5_regrid/2p5/hgt.200.2p5.nc')
        netcdf_data = {'netcdf_field':  geopot_Z, 'netcdf_dates': netcdf_dates, 'lat':lat, 'lon': lon}
        geopot_Z, lat, lon = ncutil.return_field_in_certain_dates(netcdf_data, mjo_dates = DJF_dates)
        logging_object.write("FINISHED READING %s ...."%('geopotential')) 
    
        netcdf_data = {'uwnd':uwnd, 'vwnd':vwnd, 'vort':vort, 'geopot_Z':geopot_Z, 'lat':lat, 'lon':lon, 'netcdf_dates': DJF_dates,}
        h5saveload.make_sure_path_exists(path)  
        h5saveload.save_dict_to_hdf5(netcdf_data, path+'DJF_winds.hdf5')
                        
    else:
        logging_object.write('Good news is, the netcdf data was already saved')
        netcdf_data = h5saveload.load_dict_from_hdf5(path+'DJF_winds.hdf5')
        logging_object.write("Loaded the data now.. Go on... ")  
     
    lat, lon = netcdf_data['lat'], netcdf_data['lon']
    omega    = 7.2921159*1e-5
    R        = 6371e3
    
    cos_phi  = np.cos(np.deg2rad(netcdf_data['lat']))[None,:,None]
    sin_phi  = np.sin(np.deg2rad(netcdf_data['lat']))[None,:,None]
    vort_abs = netcdf_data['vort'] + 2*omega*sin_phi
    E        = ((netcdf_data['uwnd'])**2 + (netcdf_data['vwnd'])**2)/2

    dE_by_dy = (1/R)*np.gradient(E, np.deg2rad(netcdf_data['lat']), axis=2)
    logging_object.write("dE by dy") 
    dE_by_dx = (1/(R*cos_phi))*np.gradient(E, np.deg2rad(netcdf_data['lon']), axis=-1)
    logging_object.write("dE by dx") 
    
    dphiT_by_dy = np.nanmean(np.nanmean(-(vort_abs*netcdf_data['uwnd'] + dE_by_dy),axis=0), axis=0)
    dphiT_by_dx = np.nanmean(np.nanmean(+(vort_abs*netcdf_data['vwnd'] + dE_by_dx),axis=0), axis=0)

    logging_object.write("phiT gradient is calculated ....")  
    
    uwnd_mean     = np.nanmean(np.nanmean(netcdf_data['uwnd'], axis=0), axis=0)
    vwnd_mean     = np.nanmean(np.nanmean(netcdf_data['vwnd'], axis=0), axis=0)
    vort_mean     = np.nanmean(np.nanmean(netcdf_data['vort'], axis=0), axis=0)
    geopot_Z_mean = np.nanmean(np.nanmean(netcdf_data['geopot_Z'], axis=0), axis=0)
    
    
    phi_T_data = {'dphiT_by_dy':dphiT_by_dy, 'dphiT_by_dx'  :dphiT_by_dx, \
                  'lat':lat,      'lon':lon, 'netcdf_dates' :DJF_dates,\
                  'uwnd_mean'  :uwnd_mean,   'vwnd_mean'    :vwnd_mean,\
                  'vort_mean'  :vort_mean,   'geopot_Z_mean':geopot_Z_mean }
        
    def integrate_NH():
        
        logging_object.write("Integrating now to calculate phi_T")    
        la=np.deg2rad(lat)
        lo=np.deg2rad(lon)
        
        x = dphiT_by_dy
        phi_T1 = integrate.cumtrapz(x, la, axis =0, initial=0)*R
        x = dphiT_by_dx*cos_phi
        phi_T2 = integrate.cumtrapz(x, lo, axis=-1, initial=0)*R
        return phi_T1, phi_T2
    
    def integrate_SH():
        la=np.deg2rad(lat)
        lo=np.deg2rad(lon)
        
        x = dphiT_by_dy
        phi_T1 = integrate.cumtrapz(x[::-1, :],la[::-1],  axis =0, initial=0)*R
        x = dphiT_by_dx*cos_phi
        phi_T2 = integrate.cumtrapz(x[:,::-1], lo[::-1],  axis=-1, initial=0)*R
        return phi_T1[::-1,:], phi_T2[:,::-1]
    
    
    phi_T1, phi_T2 = integrate_NH()
    phi_T_data['phi_T1_N'] = phi_T1
    phi_T_data['phi_T2_N'] = phi_T2
    phi_T_data['phi_T_N']  = phi_T2+phi_T1
    
    
    phi_T1, phi_T2 = integrate_SH()
    phi_T_data['phi_T1_S'] = phi_T1
    phi_T_data['phi_T2_S'] = phi_T2
    phi_T_data['phi_T_S']  = phi_T2+phi_T1 
    
    h5saveload.make_sure_path_exists(path)  
    h5saveload.save_dict_to_hdf5(phi_T_data, path+'phi_T_data.hdf5')

    logging_object.write('Awesome! U, V, phi_T successfully saved for DJF winds')
    
    end_time_in = ti.time()

    logging_object.write(' --------------------------------------')
    logging_object.write(' -----> Time taken = %1.3f  <----'%(end_time_in-start_time_in))
    logging_object.write(' --------------------------------------')
    logging_object.write('                                                             ')
    logging_object.write('                                                             ')
    
    
if __name__ == "__main__":  
    
    i        = 0
    pres_lev = [200]
                               
    land_mask_correction =  False # 
    mask_value           =  np.nan

    infile = {'lev'         : pres_lev[i],\
              'path'        : '/scratch/pbarpanda/spherical_SWE/empirical_data_reanalysis',}

    logging_object          = logruns.default_log(logfilename   = 'empirical_phi_T', \
                                                  log_directory = './logs/')
    logging_object.write('STARTING NOW')

    start_time=ti.time()
    calculate_mean_field(infile)
    end_time=ti.time()

    logging_object.write(' ========== CODE RAN SUCCESSFULLY, congrats! ================')
    logging_object.write(' -----> Total Time taken = %1.3f  <----'%(end_time-start_time))
    logging_object.write(' ============================================================')
    