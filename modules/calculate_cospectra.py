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
            

def calc_spacetime_cross_spec( a, b, dx, ts = 1., smooth = 1, width = 15., NFFT = 256 ):
    """
    Calculate space-time co-spectra, following method of Hayashi (1971)

    Input:
      a - variable 1, dimensions = (time, space)
      b - variable 2, dimensions = (time, space)
      dx - x-grid spacing (unit = space)
      ts - sampling interval (unit = time)
      smooth - 1 = apply Gaussian smoothing
              width - width of Gaussian smoothing
              NFFT - length of FFT in cross-spectra calculations, sets the size of the 

    Output:
      K_p - spectra for positive frequencies
      K_n - spectra for negative frequencies
      lon_freqs - wavenumbers
      om - frequencies 
    """
    
#     print (a.shape)
    
    t, l = np.shape( a )
    lf = int(l/2)
    
    #Calculate spatial ffts. 
    Fa = np.fft.fft( a, axis = 1 ) / float( l  ) #normalize as in Randel and Held 1991
    Fb = np.fft.fft( b, axis = 1 ) / float( l  ) #normalize as in Randel and Held 1991

#     print (lf)
    
    #Only keep positive wavenumbers
    lon_freq = np.fft.fftfreq( l, d = dx )[:lf] #=n / 2pi a cos\phi in Randel Held (if dx = dlon x acos\phi)
    
    ## Note that Randel and Held (1991) , wave number = n/ a cos\phi. So the actual wave_number = lon_freq x 2 pi

    CFa = Fa[:, :lf].real
    SFa = Fa[:, :lf].imag
    CFb = Fb[:, :lf].real
    SFb = Fb[:, :lf].imag

    
#     NFFT = 256
    tf = int(NFFT / 2 + 1)

    #K_n,w arrays
    K_p = np.zeros( ( tf, lf ) )
    K_n = np.zeros( ( tf, lf ) )
    
    #Cross-spectra
    for i in range( lf ):
        window = np.hamming( NFFT )
        csd_CaCb, om = mm.csd( CFa[:, i], CFb[:, i], Fs = 1. / ts, NFFT = NFFT, scale_by_freq = True, window=mm.window_hanning)
        csd_SaSb, om = mm.csd( SFa[:, i], SFb[:, i], Fs = 1. / ts, NFFT = NFFT, scale_by_freq = True, window=mm.window_hanning) 
        csd_CaSb, om = mm.csd( CFa[:, i], SFb[:, i], Fs = 1. / ts, NFFT = NFFT, scale_by_freq = True, window=mm.window_hanning) 
        csd_SaCb, om = mm.csd( SFa[:, i], CFb[:, i], Fs = 1. / ts, NFFT = NFFT, scale_by_freq = True, window=mm.window_hanning)
        K_p[:, i] = csd_CaCb.real + csd_SaSb.real + csd_CaSb.imag - csd_SaCb.imag
        K_n[:, i] = csd_CaCb.real + csd_SaSb.real - csd_CaSb.imag + csd_SaCb.imag
        #Don't need factor 4 from Hayashi eq4.11, since Fourier co-efficients are 1/2 as large due to only retaining positive wavenumbers

    #Combine
    K_combine = np.zeros( ( tf * 2, lf ) ) 
    K_combine[:tf, :] = K_n[::-1, :]   #for the convolution
    K_combine[tf:, :] = K_p[:, :]  


    if smooth == 1.:
        x = np.linspace( -tf / 2, tf / 2., tf )
        gauss_filter = np.exp( -x ** 2 / (2. * width ** 2 ) )
        gauss_filter /= sum( gauss_filter )
        for i in range( lf ):
            K_combine[:, i] = np.convolve( K_combine[:, i], gauss_filter, 'same' )

    #Take out positive and negative parts
    K_n = K_combine[:tf, :]
    K_p = K_combine[tf:, :]
    K_n = K_n[::-1, :]

    return K_p , K_n, K_combine, lon_freq, om


 
def calPhaseSpeedSpectrum( P_p, P_n, f_lon,  om, cmax, nps, i1 = 1, i2 = 50 ):
    """
    Calculate space-time co-spectra, following method of Hayashi (1971)

    Input:
      P_p - spectra for positive phase speeds
      P_n - spectra for negative phase speeds
      f_lon - wavenumbers
      om - frequencies 
      cmax - maximum phase speed
      nps - size of phase speed grid
      i1 - lowest wave number to sum over
      i2 - highest wave number to sum over
    
    Output:
      P_cp - spectra for positive phase speeds
      P_cn - spectra for negative phase speeds
      C * lon_unit / time_unit - phase speeds
    """
    
    if i2 < i1:
        print ("WARNING: highest wavenumber smaller than lowest wavenumber")

    j = len( f_lon )
    t = len( om )

    #Make phase speed grid	
    C = np.linspace(0., cmax, nps)

    #K_n,c arrays
    P_cp = np.zeros( ( nps, j ) )  ### function of (w,k)
    P_cn = np.zeros( ( nps, j ) )  ### function of (w,k)

#     print ('om = %s'%str(om.shape), ' P_p = %s'%str(P_p.shape), \
#            'flon = %s'%str(f_lon.shape), 'i1 = %s'%str(i1), 'i2 = %s'%(i2))
    
    #Interpolate
    for i in range( int(i1), int(i2) ):
    #Make interpolation functions c = omega / k
        f1 = si.interp1d(om / f_lon[i], P_p[:, i], 'linear' )
        f2 = si.interp1d(om / f_lon[i], P_n[:, i], 'linear' )

        #interp1d doesn't handle requested points outside data range well, so just zero out these points
        k = -1
        for j in range( len(C) ):
            if C[j] > max(om) / f_lon[i]:
                k = j
                break
        if k == -1:
            k = len( C )
        ad1 = np.zeros( nps )
        ad1[:k] =  f1( C[:k]  )
        ad2 = np.zeros( nps )
        ad2[:k] =  f2( C[:k] )

        #Interpolate
        P_cp[:, i] = ad1 * f_lon[i] 
        P_cn[:, i] = ad2 * f_lon[i] 

    #Sum over all wavenumbers
    return np.sum(P_cp, axis = 1), np.sum(P_cn, axis = 1), C, P_cp, P_cn ### function of (w,k)

def calc_co_spectra( x, y, dx, lat, dt, cmax = 50, nps = 50, NFFT=256):
    
    """
    Calculate eddy phase speed co-spectra, following method of Hayashi (1974)

    Input:
      x - variable 1, dimensions = (time, lat, lon)
      y - variable 2, dimensions = (time, lat, lon)
      dx - spacing of spatial points (unit = m)
      lat - latitudes -> note that if working in spherical co-ordinates dx must be scaled by 
            a * cos(lat)
      dt - sampling interval (unit = s)
      cmax - maximum phase speed 
      nps - grid of phase speeds

    Output:
      p_spec - the spectra
      ncps - phase speeds
    """
    if x.ndim != 3:
        print ("WARNING: Dimensions of x != 3")
    if y.ndim != 3:
        print ("WARNING: Dimensions of y != 3")

    t, l, j = np.shape( x )
    x -= np.nanmean( x, axis = 2 )[:, :, np.newaxis]   ## only removes the longitudinal mean
    y -= np.nanmean( y, axis = 2 )[:, :, np.newaxis]
    
    p_spec      = np.zeros( ( l, 2 * nps ) ) #array to hold spectra
    p_spec_indi = np.zeros( ( l, 2 * nps, int(x.shape[-1]/2) ) )
    K_combine1   = []
    wave_number_with_latitude=[]
    
    #Cycle through latitudes
    for i in range( l ):

        #print ("Doing: ", i, end=',')
        
        #Calculate space - time cross-spectra
        K_p, K_n, K_combine, lon_freq, om      = calc_spacetime_cross_spec(x[:, i, :], y[:, i, :], dx = dx*6371e3*np.cos(np.deg2rad(lat[i])), ts = dt, NFFT=NFFT)
        
        #Convert to phase speed spectra  
        #P_p, P_n, f_lon, om, cmax, nps, i1 = 1, i2 = 50 
        P_Cp, P_Cn, cp, P_cp_indiv, P_cn_indiv = calPhaseSpeedSpectrum(K_p, K_n, lon_freq, om, cmax, nps, 1, nps / 2 );
        

        p_spec[i, :nps]          = P_Cn[::-1] #negative phase speeds
        p_spec[i, nps:]          = P_Cp[:]    #positive phase speeds
        p_spec_indi[i, :nps,  :] = P_cn_indiv[::-1, :]
        p_spec_indi[i,  nps:, :] = P_cp_indiv
        K_combine1.append(K_combine)
        wave_number_with_latitude.append(lon_freq)

        ncps       = np.zeros( 2 * nps )    #full array of phase speeds 
        ncps[:nps] = -1. * cp[::-1] 
        ncps[nps:] = cp[:] 

    return p_spec, ncps, p_spec_indi, np.array(K_combine1), om, np.array(wave_number_with_latitude)


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
    long_mean = np.nanmean(array_in,axis=0)
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
                              path = '/data/pbarpanda/co_spectra_data1/UV/', nps=None, NFFT=None, \
                              logging_object=None):
    
    make_sure_path_exists(path)
    
    dx = np.deg2rad(lon[1]-lon[0])
    #############  UV  #############
    
    if nps is None:
        nps = len(lon)
    if NFFT is None:
        NFFT = (X.shape[0]*X.shape[1])

    ## calculate at each pressure level

    spectra_lat_PhaseSpeed=[];
    spectra_lat_PhaseSpeed_WaveNumber=[];
    spectra_lat_Frequency_WaveNumber=[];  
  

    for pj in range(len(pres)):


        logging_object.write(line='pres no - %d'%pj)
    
        a=np.copy(X)[:,:,pj,:,:].reshape(((X.shape[0]*X.shape[1]), len(lat), len(lon)))
        b=np.copy(Y)[:,:,pj,:,:].reshape(((X.shape[0]*X.shape[1]), len(lat), len(lon))); DT = (24//X.shape[1])*3600
    
        ### remove seasonal mean and detrend data and remove annual cycle if there is one year of signal       
        a = remove_dominant_signals(np.copy(a), nDayWin = 365, no_of_hours=(24//X.shape[1]))
        b = remove_dominant_signals(np.copy(b), nDayWin = 365, no_of_hours=(24//X.shape[1]))
        
        p_spec, ncps, p_spec_indi, K_combine1, om, wave_number_with_latitude = \
                                   calc_co_spectra(a, b, np.copy(dx), np.copy(lat), DT, cmax = 60, nps = nps, NFFT=NFFT)

        spectra_lat_PhaseSpeed.append(p_spec)
        spectra_lat_PhaseSpeed_WaveNumber.append(p_spec_indi)
        spectra_lat_Frequency_WaveNumber.append(K_combine1)

    spectra_lat_pres_PhaseSpeed            = np.array(spectra_lat_PhaseSpeed)
    spectra_lat_pres_PhaseSpeed_WaveNumber = np.array(spectra_lat_PhaseSpeed_WaveNumber)
    spectra_lat_pres_Frequency_WaveNumber  = np.array(spectra_lat_Frequency_WaveNumber)
    frequency                              = np.append(-om[::-1],om)*24*3600/(2*np.pi)
    phase_speed                            = ncps
    name = file
    wave_number = wave_number_with_latitude*2*np.pi*6371e3*np.cos( np.deg2rad(lat[:,None])) [0,:]

    dictionary= {'spectra_lat_pres_PhaseSpeed':spectra_lat_pres_PhaseSpeed,\
                 'spectra_lat_pres_PhaseSpeed_WaveNumber':spectra_lat_pres_PhaseSpeed_WaveNumber,\
                 'spectra_lat_pres_Frequency_WaveNumber':spectra_lat_pres_Frequency_WaveNumber,\
                 'frequency':frequency, 'phase_speed':phase_speed,\
                 'wave_number_with_latitude':wave_number_with_latitude, 'wave_number':wave_number,\
                 'pres':pres, 'sub_lat':lat, 'sub_lon':lon, 'name': name, \
                 'var1': np.nanmean(np.nanmean(np.nanmean(X, axis=-1), axis=0), axis=0), \
                 'var2': np.nanmean(np.nanmean(np.nanmean(Y, axis=-1), axis=0), axis=0)}

    #save(path+'%s.hkl'%(file), dictionary)
    h5saveload.save_dict_to_hdf5(dictionary, path+'%s.hdf5'%(file)) 
       
    return dictionary
##################################################################################################################
