import netCDF4 as nc
import os
import glob as glob
import pylab as py
import numpy as np
import cartopy.crs as ccrs
import cartopy.util as cutil
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
# import hickle as hkl
import sys
import logging
import time as ti
import h5py

sys.path.append('/data/pbarpanda/python_modules/')
import save_and_load_hdf5_files as h5saveload
import netcdf_utilities as ncutil
import eulerian_fluxes as eflux
import map_create  as mapc
import logruns as logruns

import warnings
warnings.filterwarnings('ignore')
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
      [  "darkred", "darkorange", "pink", "white", "white","skyblue", "dodgerblue", "navy"][::-1])
cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list("", 
      [  "darkred", "darkorange", "pink", "white", "white","skyblue", "dodgerblue", "navy"])


def taper_vector_to_zero(data, window_length):
    """
    Taper the data in the given vector to zero at both the beginning and the ending.

    :param data: The data to taper.
    :param window_length: The length of the window (measured in vector indices),
        in which the tapering is applied for the beginning and the ending independently

    :return: The tapered data.
    """
    startinds = np.arange(0, window_length, 1)
    endinds = np.arange(-window_length - 1, -1, 1) + 2

    result = np.copy(data)
    result[0:window_length,...] = result[0:window_length,...] * (0.5 * (1 - np.cos(startinds * np.pi / window_length)))[:,None,None,None]
    result[data.shape[0] - window_length:data.shape[0],...] = \
        result[data.shape[0] - window_length:data.shape[0],...] * 0.5 * ((1 - np.cos(endinds * np.pi / window_length)))[:,None,None,None]
    return result

def reshape(Y):
    shapes = Y.shape
    return Y.reshape(shapes[0]*shapes[1], shapes[2], shapes[3], shapes[4])

def reshape_back(Y, data_per_day):
    shapes = Y.shape
    return Y.reshape(shapes[0]//int(data_per_day), int(data_per_day), shapes[1], shapes[2], shapes[3])


def padding(Y, nt=2**17):
    orig_nt = Y.shape[0]
    if orig_nt > nt:
        raise ValueError('Time series is longer than hard-coded value for zero-padding!')
        nt = nt*((orig_nt//nt)+1)
    data = np.zeros([nt, *Y.shape[1:]])
    data[0:orig_nt, ...] = Y  
    return data


default_infile = {'period_max'      : 90,   \
                  'period_min'      : 20,   \
                  'nt'              : 2**17,\
                  'tapering_window' : 10*8,\
                  'logname'         : os.path.basename(sys.argv[0]).split('.py')[0],\
                  'data_per_day'    : 8.0}    


def band_pass_filter_days(final_field, infile=None, logging_object=None) :  ### 10 days)

    if infile is None :
        infile = default_infile
    
    if logging_object is None :
        logging_object = logruns.default_log(logfilename   = infile['logname'], \
                                         log_directory = './')
    
    for keys,values in infile.items():
        globals()['%s'%keys]= values 
        
    final_field_reshape = reshape(final_field)
    logging_object.write("Reshaped data") 
    print("Reshaped data") 
    
    orig_nt = final_field_reshape.shape[0]
    
    tapered_data        = taper_vector_to_zero(final_field_reshape, tapering_window)
    logging_object.write("Tapered data") 
    print("Tapered data") 
    
    padded_data         = padding(tapered_data, nt)
    logging_object.write("Padded data with zeros") 
    print("Padded data with zeros")
    
    ############# Forward Fourier transform ############
    fourier_fft         = np.fft.fft(padded_data, axis=0)
#    nt                  = fourier_fft.shape[0]
    freq_axis           = np.fft.fftfreq(nt, d=1/data_per_day)  #### d = dt = 1/8th of a day or 3*24*3600
    
    freq_min   =  (1/period_max)
    freq_max   =  (1/period_min)
    
    slice0 = ((np.abs(freq_axis)  > freq_max) | (np.abs(freq_axis) <= freq_min))
    
    fourier_fft_filtered             = np.copy(fourier_fft)
    fourier_fft_filtered[slice0,...] = 0
    freq_axis[slice0]                = 0
    
    logging_object.write("Filtered data temporally, now doing back transform")
    print("Filtered data temporally, now doing back transform")
    
    ############ Backward Fourier transform ############
    filtered_data        = np.fft.ifft(fourier_fft_filtered, axis=0)
    logging_object.write("Awesome! Returning filtered data")
    print("Awesome! Returning filtered data")
    
    filtered_reshape = reshape_back(filtered_data[:orig_nt, ...].real, int(data_per_day))
    
    return filtered_reshape, freq_axis
    
        
# filtered_data  = band_pass_filter_days(final_field, infile)    
    
    

    