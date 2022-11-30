""""
Nicholas Lutsko -- EAPS department, MIT

Functions for calculating eddy flux co-spectra. Script follows technique of Hayashi (1971) (see also Randel and Held (1991)) and calculates spectra at a specific height.
Includes functions to calculate space-time cross-spectra and phase-speed cross-spectrum.
Updated May 30th 2018 -- fixed bug identified by Ben Toms
Updated March 14th 2019 -- fixed several bugs identified by Neil Lewis
Tested using Python 2.7.12
"""
import numpy as np
import scipy.signal as ss
import scipy.interpolate as si
import matplotlib.mlab as mm
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
# import logging
import time as ti

sys.path.append('/Users/pbarpanda/Work/EP_fluxes/python_scripts/modules/')
import save_and_load_hdf5_files as h5saveload

import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

def save(filename,dictionary):
    hkl.dump(dictionary, filename, mode='w')

def load(filename):
    dictionary = hkl.load(filename)
    return dictionary

def make_sure_path_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def calc_time_mean_space_cross_spec( a, b, dx, ts = 1., smooth = 1, width = 15., NFFT = 256 ):
    """
    Calculate space-time co-spectra, following method of Hayashi (1971)

    Input:
      a - variable 1, dimensions = (time, space)
      b - variable 2, dimensions = (time, space)
      a and b should be zonal eddies ###
      dx - x-grid spacing (unit = space)
      ts - sampling interval (unit = time)
    """
#    print (a.shape)
    
    t, l = np.shape( a )
    lf = int(l/2)
    
    #### a and b should be time mean ###
    a  = np.nanmean( np.copy(a), axis=0)
    b  = np.nanmean( np.copy(b), axis=0)
        
    NFFT = l
    lf   = int(NFFT / 2 + 1)
        
    cospec_lon, lon_freq = mm.csd( a, b, Fs = 1. / dx, NFFT = NFFT, scale_by_freq = True, window=mm.window_hanning)
    
    ## Try calculating cospec on your own
    #Calculate spatial ffts. 
    window = np.hanning( NFFT )
    Fa = np.fft.fft( a*window ) / float( l ) #normalize as in Randel and Held 1991
    Fb = np.fft.fft( b*window ) / float( l ) #normalize as in Randel and Held 1991    
    cospec_lon_raw = (Fa*np.conj(Fb))[:lf]; lon_freq_raw = np.fft.fftfreq(l, d = dx)[:lf] #=n / 2pi a cos\phi in Randel Held (if dx = dlon x acos\phi)
    
    cospec_dict = {'cospec_lon': cospec_lon, 'lon_freq':lon_freq,  'cospec_lon_raw': cospec_lon, 'lon_freq_raw':lon_freq_raw }

    return cospec_dict

 
def calc_co_spectra( x, y, dx, lat ):
    
    """
    Calculate eddy phase speed co-spectra, following method of Hayashi (1974)

    Input:
      x - variable 1, dimensions = (time, lat, lon)
      y - variable 2, dimensions = (time, lat, lon)
      dx - spacing of spatial points (unit = m)
      lat - latitudes -> note that if working in spherical co-ordinates dx must be scaled by 
            a * cos(lat)
      dt - sampling interval (unit = s)

    Output:
      p_spec - the spectra
      ncps - phase speeds
    """
    if x.ndim != 3:
        print ("WARNING: Dimensions of x != 3")
    if y.ndim != 3:
        print ("WARNING: Dimensions of y != 3")

    t, l, j = np.shape( x )
    x -= np.mean( x, axis = -1 )[:, :, np.newaxis]   ## only removes the longitudinal mean and calculates the zonal eddies
    y -= np.mean( y, axis = -1 )[:, :, np.newaxis]
    
    wave_number_with_latitude = []    
    cospectra_with_latitude   = []
    
    #Cycle through latitudes
    for i in range( l ):

        #print ("Doing: ", i, end=',')
        
        #Calculate space cross-spectra
        cospec_dict   =   calc_time_mean_space_cross_spec(x[:, i, :], y[:, i, :], dx = dx*6371e3*np.cos(np.deg2rad(lat[i])))        
        cospec_lon    = cospec_dict['cospec_lon']
        lon_freq      = cospec_dict['lon_freq']
        
        cospectra_with_latitude.append(cospec_lon)
        wave_number_with_latitude.append(lon_freq)


    return np.array(cospectra_with_latitude), np.array(wave_number_with_latitude)

## Create a sub data in certain longitude and latitude sectors
def remove_dominant_signals(array_in, nDayWin = 365, no_of_hours=8):
    """
    This function removes the dominant signals by removing the long term
    linear trend (conserving the mean) and by eliminating the annual cycle
    by removing all time periods less than a corresponding critical
    frequency.
    """
    
    nDayTot = array_in.shape[0]//no_of_hours
    dt_unit = no_of_hours*3600
    ntim    = array_in.shape[0]
    
    # Critical frequency
    fCrit   = 1./(nDayWin*dt_unit)

    # Remove long term linear trend
    long_mean = np.mean(array_in,axis=0)
    array_dt  = array_in-long_mean
    
    signal = np.copy(array_dt)
    mask   = ~np.isnan(signal)
    signal[mask] = ss.detrend(signal[mask],axis=0,type='linear')
    detrend      = signal
    #remove just trend conserving the mean
    #array_dt = detrend+long_mean
    array_dt  = np.copy(detrend)

    if (nDayTot>=365):
        fourier      = np.fft.rfft(array_dt,axis=0)
        fourier_mean = np.copy(fourier[0,:,:])
        freq         = np.fft.rfftfreq(ntim,1./dt_unit)
        ind          = np.where(freq<=fCrit)[0]
        fourier[ind,:,:] = 0.0
        fourier[0,:,:] = fourier_mean
        array_dt = np.fft.irfft(fourier,axis=0)

    return array_dt


def spectra_at_all_levels_lat(X,Y, lat, lon, pres,\
                              file='spectra', 
                              path = '/data/pbarpanda/co_spectra_data1/UV/',\
                              logging_object=None):
    
    make_sure_path_exists(path)
    
    dx = np.deg2rad(lon[1]-lon[0])
    #############  UV  #############

    ## calculate at each pressure level

    spectra_pres_lat_WaveNumber=[];

    for pj in range(len(pres)):


        logging_object.write(line='pres no - %d'%pj)
    
        a=np.copy(X)[:,:,pj,:,:].reshape(((X.shape[0]*X.shape[1]), len(lat), len(lon)))
        b=np.copy(Y)[:,:,pj,:,:].reshape(((X.shape[0]*X.shape[1]), len(lat), len(lon))); DT = (24//X.shape[1])*3600
    
        ### remove seasonal mean and detrend data and remove annual cycle if there is one year of signal       
        a = remove_dominant_signals(np.copy(a), nDayWin = 365, no_of_hours=(24//X.shape[1]))
        b = remove_dominant_signals(np.copy(b), nDayWin = 365, no_of_hours=(24//X.shape[1]))
        
        cospectra_with_latitude, wave_number_with_latitude = calc_co_spectra( a, b, np.copy(dx), np.copy(lat) )            
        spectra_pres_lat_WaveNumber.append(cospectra_with_latitude)
        
    spectra_pres_lat_WaveNumber                 = np.array(spectra_pres_lat_WaveNumber)
    name = file
    wave_number = (np.copy(wave_number_with_latitude)*2*np.pi*6371e3*np.cos( np.deg2rad(lat[:,None])))[0,:]
    
    logging_object.write( line = 'wave_number = %s'%str(wave_number.shape))
    logging_object.write( line = 'wave_number_with_latitude = %s'%str(wave_number_with_latitude.shape))
    logging_object.write( line = 'wave_number_with_latitude[0,:] = %s'%str(wave_number_with_latitude[0,:].shape))
    
    
    dictionary= {'spectra_lat_pres_WaveNumber':spectra_pres_lat_WaveNumber,\
                 'wave_number_with_latitude':wave_number_with_latitude, 'wave_number':wave_number,\
                 'pres':pres, 'sub_lat':lat, 'sub_lon':lon, 'name': name, \
                 'var1': np.nanmean(np.nanmean(np.nanmean(X, axis=-1), axis=0), axis=0), \
                 'var2': np.nanmean(np.nanmean(np.nanmean(Y, axis=-1), axis=0), axis=0)}

    #save(path+'%s.hkl'%(file), dictionary)
    h5saveload.save_dict_to_hdf5(dictionary, path+'%s.hdf5'%(file)) 
       
    return dictionary
##################################################################################################################
