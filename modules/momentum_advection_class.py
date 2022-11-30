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
from scipy import signal as ss 

sys.path.append('/data/pbarpanda/python_scripts/modules/')

import logruns as logruns
import save_and_load_hdf5_files as h5saveload
import eulerian_fluxes as eflux
import netcdf_utilities_solve_error as ncutil
import temporal_filter_for_all_years_in_one_level as tf


import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'  ### This is because NOAA PSL lab computers are somehow not able to use this feature


class momentum_advection:
    
    def __init__(self, uuu, vvv, lat1, lon1, logging_object):
        self.uuu   = uuu
        self.vvv   = vvv
        self.lat1  = lat1
        self.lon1  = lon1
        self.omega = 7.2921*1e-5
        self.f     = 2*self.omega*np.sin(np.deg2rad(self.lat1))
        self.R     = 6371e3
        self.cos_phi = np.cos(np.deg2rad(self.lat1))
        self.logging_object = logging_object
        
    def spher_div_y(self, Y, latt, lonn, ax=-2):
        self.logging_object.write('d/dy ....')
        cos_phi = np.cos(np.deg2rad(latt))
        grad    = np.gradient(Y*cos_phi[None, :, None], np.squeeze(np.deg2rad(latt)), axis=ax)/cos_phi[None, :, None]
        return grad/self.R

    def spher_div_x(self, Y, latt, lonn, ax=-1):
        self.logging_object.write('d/dx ....')
        cos_phi = np.cos(np.deg2rad(latt))
        grad    = np.gradient(Y, np.squeeze(np.deg2rad(lonn)), axis=ax)/cos_phi[None, :, None]
        return grad/self.R

    def mom_flux_div(self, uu, vv, latt, lonn):
        self.logging_object.write('mom_flux_div ...')
        duu_by_dx  = self.spher_div_x(uu, latt=latt, lonn=lonn, ax=-1)
        duv_by_dy  = self.spher_div_y(vv, latt=latt, lonn=lonn, ax=-2)
        divergence = duu_by_dx + duv_by_dy
        return duu_by_dx, duv_by_dy, divergence

    def vorticity(self, uu, vv, latt, lonn):
        self.logging_object.write('vorticty ...')
        dv_by_dx, minus_du_by_dy, curl = self.mom_flux_div(vv, -uu, latt=latt, lonn=lonn)
        return np.squeeze(dv_by_dx), np.squeeze(minus_du_by_dy), np.squeeze(curl)

    def zmean(self, X):
        self.logging_object.write('zmean ...')
        return np.nanmean(X, axis=-1, keepdims=True)

    def zanom(self, X):
        self.logging_object.write('zanom ...')
        return X - self.zmean(X)
    
    def tmean(self, X):
        self.logging_object.write('tmean ...')
        return np.nanmean(X, axis=0, keepdims=True)

    def tanom(self, X):
        self.logging_object.write('tanom ...')
        return X - self.tmean(X)
    
    def zonal_advection_terms(self):
        self.logging_object.write('zonal_advection_terms ...')
        
        du_by_dx, du_by_dy, divergence = self.mom_flux_div( self.zanom(self.uuu),  self.zanom(self.uuu), latt=self.lat1, lonn=self.lon1)

        L11 = np.squeeze( self.zmean(self.uuu)*du_by_dx)
        l11 = 'u0_du*_by_dx'; self.logging_object.write('%s calculated'%(l11))
        L12 = np.squeeze( self.zmean(self.vvv)*du_by_dy)
        l12 = 'v0_du*_by_dx'; self.logging_object.write('%s calculated'%(l12))   
        ###### I guess this term is small too. 

        du_by_dx, du_by_dy, divergence = self.mom_flux_div(np.repeat( self.zmean(self.uuu), len(self.lon1), axis=-1), \
                                                      np.repeat( self.zmean(self.uuu), len(self.lon1), axis=-1),\
                                                      latt=self.lat1, lonn=self.lon1)
        L21 = np.squeeze(self.zanom(self.uuu)*du_by_dx)  
        l21 = 'u*_du0_by_dx'; self.logging_object.write('%s calculated'%(l21))  
        ###### This should be zero by definition. So I am not calculating this for saving time
        L22 = np.squeeze( self.zanom(self.vvv)*du_by_dy)
        l22 = 'v*_du0_by_dy'; self.logging_object.write('%s calculated'%(l22)) 

        L01 = np.squeeze(self.zmean(self.uuu)*du_by_dx)
        l01 = 'u0_du0_by_dx' ; self.logging_object.write('%s calculated'%(l01))  
        ###### This should be zero by definition. So I am not calculating this for saving time
        L02 = np.squeeze( self.zmean(self.vvv)*du_by_dy)
        l02 = 'v0_du0_by_dy' ; self.logging_object.write('%s calculated'%(l02))    
        ###### I guess this term is small too. Because v0 should be zero. Still calculating it to save time 

        du_by_dx, du_by_dy, divergence = self.mom_flux_div( self.zanom(self.uuu),  self.zanom(self.uuu), latt=self.lat1, lonn=self.lon1)
        L31 = np.squeeze( self.zanom(self.uuu)*du_by_dx)
        l31 = 'u*_du*_by_dx'; self.logging_object.write('%s calculated'%(l31))    
        L32 = np.squeeze( self.zanom(self.vvv)*du_by_dy)
        l32 = 'v*_du*_by_dy'; self.logging_object.write('%s calculated'%(l32))
        
        
        du_by_dx, du_by_dy, divergence = self.mom_flux_div((self.uuu), (self.uuu), latt=self.lat1, lonn=self.lon1)
        L01_T = ((self.uuu)*du_by_dx)
        l01_T = 'u_du_by_dx'; self.logging_object.write('%s calculated'%(l01_T))
        L02_T = ((self.vvv)*du_by_dy)
        l02_T = 'v_du_by_dy'; self.logging_object.write('%s calculated'%(l02_T))
        

        flux_variables = { 'L01'  : l01,     'L02'  :l02, \
                           'L11'  : l11,     'L12'  :l12, \
                           'L21'  : l21,     'L22'  :l22, \
                           'L31'  : l31,     'L32'  :l32, \
                           'L01_T': l01_T ,  'L02_T':l02_T}

        
        fluxes         = { 'L01':   L01,    'L02': L02, \
                           'L11':   L11,    'L12': L12, \
                           'L21':   L21 ,   'L22': L22, \
                           'L31':   L31,    'L32': L32, \
                           'L01_T': L01_T , 'L02_T': L02_T }
        
        units           = 'm/s^2'
        
        self.logging_object.write('Returning all flux and flux variables ...')                
        return fluxes, flux_variables, units   
        #### In future, add a function for meridional advection_terms
        
        
    def zonal_momentum_fluxes(self):
        self.logging_object.write('zonal_momentum_terms ...')
        
        L11 = np.squeeze( np.repeat( self.zmean(self.uuu), len(self.lon1), axis=-1)*self.zanom(self.uuu))
        l11 = 'u0_u*'; self.logging_object.write('%s calculated'%(l11))
        L12 = np.squeeze( np.repeat( self.zmean(self.vvv), len(self.lon1), axis=-1)*self.zanom(self.uuu))
        l12 = 'v0_u*' ; self.logging_object.write('%s calculated'%(l12))  
        ###### I guess this term is small too. Because v0 should be zero. Still calculating it.

        L21 = np.squeeze( self.zanom(self.uuu)*np.repeat( self.zmean(self.uuu), len(self.lon1), axis=-1))
        l21 = 'u*_u0'; self.logging_object.write('%s calculated'%(l21))  
        ###### This should be zero by definition. So I am not calculating this for saving time
        L22 = np.squeeze( self.zanom(self.vvv)*np.repeat( self.zmean(self.uuu), len(self.lon1), axis=-1))
        l22 = 'v*_u0'; self.logging_object.write('%s calculated'%(l22))  

        L01 = np.squeeze( np.repeat( self.zmean(self.uuu), len(self.lon1), axis=-1)*np.repeat( self.zmean(self.uuu), len(self.lon1), axis=-1))
        l01 = 'u0_u0' ; self.logging_object.write('%s calculated'%(l01))  
        ###### This should be zero by definition. So I am not calculating this for saving time
        L02 = np.squeeze( np.repeat( self.zmean(self.vvv), len(self.lon1), axis=-1)*np.repeat( self.zmean(self.uuu), len(self.lon1), axis=-1))
        l02 = 'v0_u0' ; self.logging_object.write('%s calculated'%(l02))  
        ###### I guess this term is small too. Because v0 should be zero. Still calculating it to save time 

        L31 = np.squeeze( self.zanom(self.uuu)*self.zanom(self.uuu))
        l31 = 'u*_u*'; self.logging_object.write('%s calculated'%(l31))
        L32 = np.squeeze( self.zanom(self.vvv)*self.zanom(self.uuu))
        l32 = 'v*_u*'; self.logging_object.write('%s calculated'%(l32))
                
        L01_T = ((self.uuu)*self.uuu)
        l01_T = 'uu'; self.logging_object.write('%s calculated'%(l01_T))
        L02_T = ((self.vvv)*self.uuu)
        l02_T = 'vu'; self.logging_object.write('%s calculated'%(l02_T))
        
        flux_variables = { 'L01'  : l01,     'L02'  :l02, \
                           'L11'  : l11,     'L12'  :l12, \
                           'L21'  : l21,     'L22'  :l22, \
                           'L31'  : l31,     'L32'  :l32, \
                           'L01_T': l01_T ,  'L02_T':l02_T}
        
        fluxes         = { 'L01':     L01,    'L02':   L02, \
                           'L11':     L11,    'L12':   L12, \
                           'L21':     L21 ,   'L22':   L22, \
                           'L31':     L31,    'L32':   L32, \
                           'L01_T': L01_T , 'L02_T': L02_T }
         
        units           = 'm^2/s^2'
        
        self.logging_object.write('Returning all flux and flux variables ...')   
        return fluxes, flux_variables, units
  
        #### In future, add a function for meridional advection_terms