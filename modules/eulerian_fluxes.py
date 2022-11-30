import numpy as np


def transient_eddy_climatology_mean(x, y, vertical_levels=None): #(A'B')_bar
    X_teddy       = x - np.nanmean(x,axis=0, keepdims=True)
    Y_teddy       = y - np.nanmean(y,axis=0, keepdims=True)
    X_tzeddy      = X_teddy
    Y_tzeddy      = Y_teddy   
    eulerian_flux = (X_tzeddy*Y_tzeddy)
    mean_eulerian_flux = np.nanmean(np.nanmean(eulerian_flux, axis=0), axis=0)  
    ### Average over days and years
    name          = 'transient_eddy_climatology_mean'
    return mean_eulerian_flux, name

def steady_state_climatology_mean(x, y, vertical_levels=None): #(A_bar B_bar)
    X_teddy       = np.nanmean(x,axis=0, keepdims=True)
    Y_teddy       = np.nanmean(y,axis=0, keepdims=True)
    X_tzeddy      = X_teddy
    Y_tzeddy      = Y_teddy   
    eulerian_flux = (X_tzeddy*Y_tzeddy)
    mean_eulerian_flux = np.nanmean(np.nanmean(eulerian_flux, axis=0), axis=0)  
    ### Average over days and years
    name          = 'steady_state_climatology_mean'
    return mean_eulerian_flux, name





def transient_zonal_eddy(x, y, vertical_levels=None): #[A'* B'*]   
    X_teddy       = x - np.nanmean(x,axis=0, keepdims=True)
    Y_teddy       = y - np.nanmean(y,axis=0, keepdims=True)
    X_tzeddy      = X_teddy - np.nanmean(X_teddy,axis=-1, keepdims=True)
    Y_tzeddy      = Y_teddy - np.nanmean(Y_teddy,axis=-1, keepdims=True)    
    eulerian_flux = X_tzeddy*Y_tzeddy 
    name          = 'time_eddy_zonal_eddy_transient'
    return eulerian_flux, name, X_tzeddy, Y_tzeddy


def transient_zonal_mean(x, y, vertical_levels=None): #[A]'[B']    
    X_zmean       = np.nanmean(x,axis=-1, keepdims=True)
    Y_zmean       = np.nanmean(y,axis=-1, keepdims=True)
    X_tzeddy      = X_zmean - np.nanmean(X_zmean, axis=0, keepdims=True)
    Y_tzeddy      = Y_zmean - np.nanmean(Y_zmean, axis=0, keepdims=True)    
    eulerian_flux =  X_tzeddy*Y_tzeddy
    name          = 'time_eddy_zonal_mean_transient'
    return eulerian_flux, name, X_tzeddy, Y_tzeddy


def stationary_zonal_eddy(x, y, vertical_levels=None): #[A_bar_* B_bar_*]    
    X_tmean       = np.nanmean(x,axis=0, keepdims=True)
    Y_tmean       = np.nanmean(y,axis=0, keepdims=True)
    X_tzeddy      = X_tmean - np.nanmean(X_tmean, axis=-1, keepdims=True)
    Y_tzeddy      = Y_tmean - np.nanmean(Y_tmean, axis=-1, keepdims=True)    
    eulerian_flux =  X_tzeddy*Y_tzeddy 
    name          = 'time_mean_zonal_eddy_stationary'
    return eulerian_flux, name, X_tzeddy, Y_tzeddy

def mean_meridional_uncorrected(x, y, vertical_levels=None): #[A_bar_* B_bar_*]
    
    X_tmean  = np.nanmean(x,axis=0, keepdims=True)
    Y_tmean  = np.nanmean(y,axis=0, keepdims=True)
    X_tzmean = np.nanmean(X_tmean, axis=-1, keepdims=True)
    Y_tzmean = np.nanmean(Y_tmean, axis=-1, keepdims=True)    
    eulerian_flux =  X_tzmean*Y_tzmean 
    name          = 'mean_meridional_uncorrected'
    return eulerian_flux, name, X_tzmean, Y_tzmean

def mean_meridional_corrected(x, y, vertical_levels): #[A_bar_* B_bar_*]        
    g = 10 #m/s
    weights  = np.abs(np.gradient(vertical_levels)/g)
    X_tmean  = np.nanmean(x,axis=0, keepdims=True)
    Y_tmean  = np.nanmean(y,axis=0, keepdims=True)    
    Y_tmean  = np.copy(Y_tmean) - np.nansum(np.copy(Y_tmean)*weights[None,:,None,None], axis=1, keepdims=True)  
    ### This is to correct for mass flux at eah latitude
    X_tzmean = np.nanmean(X_tmean, axis=-1, keepdims=True)
    Y_tzmean = np.nanmean(Y_tmean, axis=-1, keepdims=True)    
    eulerian_flux =  X_tzmean*Y_tzmean 
    name          = 'mean_meridional_corrected'
    return eulerian_flux, name, X_tzmean, Y_tzmean
