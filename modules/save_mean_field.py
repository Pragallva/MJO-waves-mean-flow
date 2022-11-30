import netCDF4 as nc
import os
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
import h5py

sys.path.append('/Users/pbarpanda/Work/EP_fluxes/python_scripts/modules/')
import save_and_load_hdf5_files as h5saveload
import netcdf_utilities as ncutil


import warnings
warnings.filterwarnings('ignore')
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


def return_field(var='uwnd', months=[12, 1, 2]):
    
    field1 = []
    for year in range(1979,1981):
        print (year, end=', ')
        for mo in months:
            filename     = '/Projects/era5_regrid/2p5/%s.2p5.%s%s.nc'%(var, year, ncutil.month_string[mo-1])
            file         = glob.glob(filename)
            for ncfile in file:
                print (ncfile.split('.2p5')[1])
                v_var        = nc.Dataset(ncfile,'r')
                lat          = v_var['lat'][:]
                lon          = v_var['lon'][:]
                time         = v_var['time'][:]
                pres         = v_var['level'][::-1]
                ucomp        = v_var.variables[var][:,::-1,:,:].reshape(len(time)//8, 8, len(pres), len(lat), len(lon))
                field1.append(np.nanmean(ucomp, axis=1))
    field = np.squeeze(np.array(field1))
    print (field.shape)
    if var == 'air':
        P0     = pres[0]
        field  = field*(P0/pres[None,:,None,None])**0.286            
    return field, lat, lon, pres


def return_field(var='uwnd', months=[12, 1, 2]):
    
    filename     = '/Projects/era5_regrid/2p5/%s.2p5.%s%s.nc'%('uwnd', '1979', ncutil.month_string[0])
    v_var        = nc.Dataset(filename,'r')
    lat          = v_var['lat'][:]
    lon          = v_var['lon'][:]
    pres         = v_var['level'][::-1]
    time         = v_var['time'][:]
        
    field1 = np.zeros((1, len(pres), len(lat), len(lon)))
    for year in range(1979,2020):
        print (year, end=', ')
        for mo in months:
            filename     = '/Projects/era5_regrid/2p5/%s.2p5.%s%s.nc'%(var, year, ncutil.month_string[mo-1])
            file         = glob.glob(filename)
            for ncfile in file:
                print (ncfile.split('.2p5')[1])
                v_var        = nc.Dataset(ncfile,'r')
                lat          = v_var['lat'][:]
                lon          = v_var['lon'][:]
                time         = v_var['time'][:]
                pres         = v_var['level'][::-1]
                ucomp        = v_var.variables[var][:,::-1,:,:].reshape(len(time)//8, 8, len(pres), len(lat), len(lon))
                field1 = np.append(field1, np.nanmean(ucomp, axis=1), axis=0)
    field = np.squeeze(np.array(field1))
    if var == 'air':
        P0     = pres[0]
        field  = field*(P0/pres[None,:,None,None])**0.286            
    return field, lat, lon, pres


field, lat, lon, pres = return_field(var='uwnd', months=[12, 1, 2])
DJF_u  = {'zonal_wind':field, 'lat':lat, 'lon':lon, 'pres':pres}
h5saveload.make_sure_path_exists('/data/pbarpanda/all_years/')
h5saveload.save_dict_to_hdf5(DJF_u, '/data/pbarpanda/all_years/DJF_u.hdf5')

field, lat, lon, pres =  return_field(var='vwnd', months=[12,1,2])
DJF_v  = {'meridional_wind':field, 'lat':lat, 'lon':lon, 'pres':pres}
h5saveload.save_dict_to_hdf5(DJF_v, '/data/pbarpanda/all_years/DJF_v.hdf5')

field, lat, lon, pres = return_field(var='air', months=[12,1,2])
DJF_theta = {'theta':field, 'lat':lat, 'lon':lon, 'pres':pres}
h5saveload.save_dict_to_hdf5(DJF_theta, '/data/pbarpanda/all_years/DJF_theta.hdf5')
