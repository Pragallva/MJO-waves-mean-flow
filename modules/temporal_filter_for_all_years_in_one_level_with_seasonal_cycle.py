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
from scipy import signal as ss 
import calendar
from scipy.interpolate import interp1d

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
    result[0:window_length,...] = result[0:window_length,...] * (0.5 * (1 - np.cos(startinds * np.pi / window_length)))[:,None,None]
    result[data.shape[0] - window_length:data.shape[0],...] = \
        result[data.shape[0] - window_length:data.shape[0],...] * 0.5 * ((1 - np.cos(endinds * np.pi / window_length)))[:,None,None]
    return result



def reshape(Y):
#     shapes = Y.shape
    return Y #.reshape(shapes[0], shapes[1], shapes[2], shapes[3])

def reshape_back(Y, data_per_day):
    shapes = Y.shape
    return Y.reshape(shapes[0]//int(data_per_day), int(data_per_day), shapes[1], shapes[2])


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
                  'data_per_day'    : 8.0,\
                  'no_of_harmonics' : 3,\
                  'remove_seasonal_cycle': False}    


def compute_seasonal_cyle(final_field, infile = None, logging_object=None):
        
            start_time1 = ti.time()
            if infile is None :
                infile = default_infile
                
            if logging_object is None :
                logging_object = logruns.default_log(logfilename   = infile['logname'], \
                                                 log_directory = './')
                
            for keys,values in infile.items():
                globals()['%s'%keys] = values
                print (keys, values)
               
            time_len             =  final_field.shape[0]
            no_of_years          =  int(time_len/(365*data_per_day))
            time_lon_to_consider =  int(no_of_years*365*data_per_day  )
            
            
            climatology_field    =  final_field[:time_lon_to_consider, ...].reshape( (int(no_of_years), \
                                                                                      365, int(data_per_day), \
                                                                                      final_field.shape[-2], final_field.shape[-1]) )
            climatology_mean     =  np.nanmean(climatology_field, axis=0)

            climatology_mean     =  np.reshape(climatology_mean, (int(365*data_per_day), final_field.shape[-2], final_field.shape[-1]))
            logging_object.write("Reshaped and calculated climatolgy mean") 
            
            NN                   =  climatology_mean.shape[0]
            seasonal_fft         =  np.fft.fft((climatology_mean ), axis=0)
            logging_object.write("FFT of climatology done") 
            
            seasonal_fft[no_of_harmonics     : NN//2, ...] = 0
            seasonal_fft[NN//2 : NN-no_of_harmonics,  ...] = 0
            seasonal_cycle       =  np.fft.ifft(seasonal_fft, axis=0)
            logging_object.write("IFFT of climatology done") 
            
            X          = np.linspace(1,int(365*data_per_day),int(365*data_per_day))
            X_leap     = np.linspace(1,int(365*data_per_day),int(366*data_per_day))
            
#             f2         = interp1d(X, seasonal_cycle, kind='cubic', axis=0, fill_value="extrapolate")            
#             leap_cycle = f2(X_leap)
            
            leap_cycle = np.zeros((int(366*data_per_day), final_field.shape[-2], final_field.shape[-1]))
            for la in range(final_field.shape[-2]):
                for lo in range(final_field.shape[-1]):
                    f2              = interp1d(X, seasonal_cycle[:, la, lo], kind='cubic', fill_value="extrapolate")
                    leap_cycle[:, la, lo] = f2(X_leap)
            
            logging_object.write("Interpolated to leap years") 
                       
            full_seasonal_cycle  = np.zeros(final_field.shape)
            index  = 0
            logging_object.write("Beginning to loop through all the years") 
            logging_object.write("---------------------------------------") 
            for year in range(1979, 1979+no_of_years):
                logging_object.write("year - %d"%(year)) 
                if calendar.isleap(year):
                    new_seas    = leap_cycle
                    no_of_days  = int(366*data_per_day)
                else:
                    new_seas    = seasonal_cycle
                    no_of_days  = int(365*data_per_day)
                index += no_of_days
                full_seasonal_cycle[index-no_of_days : index , ...] = new_seas
                        
            logging_object.write("---------------------------------------") 
            logging_object.write("Returning the full seasonal cycle now") 
            
            end_time1 = ti.time()

            HRS         = (end_time1-start_time1)/3600
            MIN         = ((HRS - int(HRS))*60)
            SEC         = ((MIN - int(MIN))*60)
            logging_object.write(' ')
            logging_object.write('Climatology calculation time = %1d hrs, %1d min, %1.2f sec'%(int(HRS), int(MIN), (SEC)))
            logging_object.write(' ')

            return full_seasonal_cycle
    
            
def band_pass_filter_days(final_field, seasonal_field = None, infile=None, logging_object=None) :  ### 10 days)

            start_time = ti.time()
            if infile is None :
                infile = default_infile

            if logging_object is None :
                logging_object = logruns.default_log(logfilename   = infile['logname'], \
                                                 log_directory = './')

            for keys,values in infile.items():
                globals()['%s'%keys]= values

            
            if isinstance(final_field, np.ma.MaskedArray): ##### Because fft can't seem to deal with masked array properly.
                final_field = final_field.filled(np.nan)    ##### Just fill them with nans and it will work out.
            
            ###### Remove seasonal cycle if it needs to be removed
            if remove_seasonal_cycle:
                if seasonal_field is None:
                    seasonal_field = compute_seasonal_cyle(final_field, infile = infile, logging_object=logging_object)
                    logging_object.write("Computed the seasonal cycle")
#                 final_field2 = final_field - seasonal_field
                logging_object.write("Removed the seasonal cycle") 
            else:
                 seasonal_field = 0
                
             
            final_field2 = final_field - seasonal_field
            # Remove long term linear trend
            long_mean     = np.nanmean(final_field2,axis=0, keepdims=True)
            final_field2  = final_field2-long_mean
                        
           
            ###### print (final_field.shape) #################
            
            signal       = np.copy(final_field2)
            mask         = ~np.isnan(signal)
            signal[mask] = ss.detrend(signal[mask],axis=0,type='linear')
            detrend      = signal
            #remove just trend conserving the mean
            #array_dt = detrend+long_mean
            final_field  = np.copy(detrend)       

            logging_object.write("Detrended the data") 
            #print("Detrended data")         
           
        
            final_field_reshape = reshape(final_field)
            logging_object.write("Reshaped data") 
            #print("Reshaped data") 

            orig_nt = final_field_reshape.shape[0]
            
#             print (final_field_reshape.shape) #################

            tapered_data        = taper_vector_to_zero(final_field_reshape, tapering_window)
            logging_object.write("Tapered data") 
            #print("Tapered data") 

            padded_data         = padding(tapered_data, nt)
            logging_object.write("Padded data with zeros") 
            #print("Padded data with zeros")

            ############# Forward Fourier transform ############
            fourier_fft         = np.fft.fft(padded_data, axis=0)
            freq_axis           = np.fft.fftfreq(nt, d=1/data_per_day)  #### d = dt = 1/8th of a day or 3*24*3600

            if period_max is None:
                freq_max   =  (1/period_min)
                slice0     =  (np.abs(freq_axis) > freq_max)

            elif period_min is None:
                freq_min   =  (1/period_max)
                slice0     =  (np.abs(freq_axis) <= freq_min)

            else:
                freq_min   =  (1/period_max)
                freq_max   =  (1/period_min)        
                slice0     =  ((np.abs(freq_axis)  > freq_max) | (np.abs(freq_axis) <= freq_min))

            fourier_fft_filtered             = np.copy(fourier_fft)
            fourier_fft_filtered[slice0,...] = 0
            freq_axis[slice0]                = 0

            logging_object.write("Filtered data temporally, now doing back transform")
            #print("Filtered data temporally, now doing back transform")

            ############ Backward Fourier transform ############
            filtered_data        = np.fft.ifft(fourier_fft_filtered, axis=0)
            logging_object.write("Awesome! Returning filtered data")
            #print("Awesome! Returning filtered data")

        #    filtered_reshape = reshape_back(filtered_data[:orig_nt, ...].real, int(data_per_day))

            filtered_data = filtered_data[:orig_nt, ...].real #, int(data_per_day))

            end_time = ti.time()

            HRS         = (end_time-start_time)/3600
            MIN         = ((HRS - int(HRS))*60)
            SEC         = ((MIN - int(MIN))*60)
            logging_object.write(' ')
            logging_object.write('Filtering time = %1d hrs, %1d min, %1.2f sec'%(int(HRS), int(MIN), (SEC)))
            logging_object.write(' ')
            
            return filtered_data, freq_axis
    
        
# filtered_data  = band_pass_filter_days(final_field, infile)    
    
    

    