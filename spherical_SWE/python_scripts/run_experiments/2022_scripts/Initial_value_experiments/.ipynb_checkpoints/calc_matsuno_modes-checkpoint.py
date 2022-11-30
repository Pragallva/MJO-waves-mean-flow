
import scipy.integrate as sci
import numpy as np
import netCDF4 as nc
import glob as glob
import pylab as py
import numpy as np
import hickle as hkl
import cartopy.crs as ccrs
import cartopy.util as cutil
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
import sys
import logging
import time as ti
import scipy.signal as signal
from scipy import stats
from scipy.linalg import lstsq
import matplotlib as mpl
import numbers
from scipy.integrate import solve_bvp
from scipy import interpolate
from scipy.integrate import odeint
from scipy.special import hermite as He
import math
import numpy.ma as ma
from IPython.display import Image, display
from scipy.optimize import fsolve
import os
sys.path.append('/data/pbarpanda/python_scripts/modules/')

import momentum_advection_class as momentum_advect
import logruns as logruns
import save_and_load_hdf5_files as h5saveload
import eulerian_fluxes as eflux
import Aug11_netcdf_utilities_TRACK as ncutil

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
      [  "darkred", "darkorange", "pink", "white", "white","skyblue", "dodgerblue", "navy"][::-1])

def colorbar(fontsize=20):
    cbar = py.colorbar()
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(fontsize)
            
def lrange(F,n=10):
    max_abs = np.max([np.abs(np.min(F)), np.abs(np.max(F))])
    range1  = np.linspace(-max_abs, max_abs, n)
    return range1
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
      ["maroon", "red", "pink",  "white", "white", "skyblue", "dodgerblue", "navy"][::-1])


def solve_dispersion(wd, *args):
    kd, n = args
    return (wd**3 - wd*kd**2 - kd - wd*(2*n+1))

def Dn(n, zeta):
    from scipy.special import hermite as He
    from scipy.special import hermitenorm as He_norm 
    Hn   = He(n)(zeta)
    gauss = np.exp(-(zeta**2)/2)
    pi   = np.pi
    norm = 2**(-n/2) #/((np.sqrt(pi)*math.factorial(n))**(1/2.0))
    PCF  = Hn*gauss*norm
    return PCF #/PCF.max()

def vector(n, kn=2, w=0, t = 0, typi = None, params=None, grids=None):
          
    lon, yy = np.meshgrid(grids['x']/params['R'], grids['y'])
    
    zeta = yy/(params['yT'])
       
    two_beta_c = (2*params['beta']*params['c'])

    if ((n>=1) & (typi == 'Rossby')):
        
        ########## Calculate Rossby wave dispersion relation  ####### 
        ## For Rossby wave ##
        args    = (kn, n)
        guess   = kn/(kn**2 + 2*n + 1) ## theoretical estimate
        params['wd']  = fsolve(solve_dispersion, guess, args)

        params['w_scale'] = np.sqrt(params['beta']*params['c'])
        params['w']       = params['wd']*params['w_scale']
        params['k_scale'] = 1/params['yT']    
        params['k']       = kn*params['k_scale']   
        ###################################################
            
        k, x = params['k'], grids['x']
        
        w_p_ck     =  (params['w'] + params['k']*params['c'])
        w_m_ck     =  (params['w'] - params['k']*params['c'])
                
        qn_plus_1  =   Dn(n+1, zeta)*(1/w_m_ck)*np.sqrt(two_beta_c)                
        rn_minus_1 = n*Dn(n-1, zeta)*(1/w_p_ck)*np.sqrt(two_beta_c)
        
        ux         = ((+1)*np.exp(1j*(kn*lon - params['w']*t)))
        uy         = (qn_plus_1 + rn_minus_1)/2
        un         = (ux*uy)
        
        phix       = ((1)*np.exp(1j*(kn*lon - params['w']*t)))
        phiy       = params['c']*(qn_plus_1 - rn_minus_1)/2
        phin       = phix*phiy
        
        vx         = ((-1j)*np.exp(1j*(kn*lon - params['w']*t)))
        vy         = Dn(n, zeta)
        vn         = vx*vy
        
        div_x = (1j)*params['k']*un
        div_y = (1.0/(np.sqrt(2)*params['yT']))*(n*Dn(n-1, zeta) - Dn(n+1, zeta))*vx
        #### y gradient using recursion relation
        div   = div_x + div_y
        
    if ((n>=1) & (typi == 'WIG')):
        
        ########## Calculate WIG wave dispersion relation  ####### 
        ## For Rossby wave ##
        args    = (kn, n)
        guess   = +np.sqrt((2*n+1) + kn**2) ## theoretical estimate
        params['wd']  = fsolve(solve_dispersion, guess, args)

        params['w_scale'] = np.sqrt(params['beta']*params['c'])
        params['w']       = params['wd']*params['w_scale']
        params['k_scale'] = 1/params['yT']    
        params['k']       = kn*params['k_scale']   
        ###################################################
            
        k, x = params['k'], grids['x']
        
        w_p_ck     =  (params['w'] + params['k']*params['c'])
        w_m_ck     =  (params['w'] - params['k']*params['c'])
                
        qn_plus_1  =   Dn(n+1, zeta)*(1/w_m_ck)*np.sqrt(two_beta_c)                
        rn_minus_1 = n*Dn(n-1, zeta)*(1/w_p_ck)*np.sqrt(two_beta_c)
        
        ux         = ((+1)*np.exp(1j*(kn*lon - params['w']*t)))
        uy         = (qn_plus_1 + rn_minus_1)/2
        un         = (ux*uy)
        
        phix       = ((1)*np.exp(1j*(kn*lon - params['w']*t)))
        phiy       = params['c']*(qn_plus_1 - rn_minus_1)/2
        phin       = phix*phiy
        
        vx         = ((-1j)*np.exp(1j*(kn*lon - params['w']*t)))
        vy         = Dn(n, zeta)
        vn         = vx*vy
        
        div_x = (1j)*params['k']*un
        div_y = (1.0/(np.sqrt(2)*params['yT']))*(n*Dn(n-1, zeta) - Dn(n+1, zeta))*vx
        #### y gradient using recursion relation
        div   = div_x + div_y
        
        
    if ((n>=1) & (typi == 'EIG')):
        
        ########## Calculate EIG wave dispersion relation  ####### 
        ## For Rossby wave ##
        args    = (kn, n)
        guess   = +np.sqrt((2*n+1) + kn**2) ## theoretical estimate
        params['wd']  = fsolve(solve_dispersion, guess, args)

        params['w_scale'] = np.sqrt(params['beta']*params['c'])
        params['w']       = params['wd']*params['w_scale']
        params['k_scale'] = 1/params['yT']    
        params['k']       = kn*params['k_scale']   
        ###################################################
            
        k, x = params['k'], grids['x']
        
        w_p_ck     =  (params['w'] + params['k']*params['c'])
        w_m_ck     =  (params['w'] - params['k']*params['c'])
                
        qn_plus_1  =   Dn(n+1, zeta)*(1/w_m_ck)*np.sqrt(two_beta_c)                
        rn_minus_1 = n*Dn(n-1, zeta)*(1/w_p_ck)*np.sqrt(two_beta_c)
        
        ux         = ((+1)*np.exp(1j*(kn*lon - params['w']*t)))
        uy         = (qn_plus_1 + rn_minus_1)/2
        un         = (ux*uy)
        
        phix       = ((1)*np.exp(1j*(kn*lon - params['w']*t)))
        phiy       = params['c']*(qn_plus_1 - rn_minus_1)/2
        phin       = phix*phiy
        
        vx         = ((-1j)*np.exp(1j*(kn*lon - params['w']*t)))
        vy         = Dn(n, zeta)
        vn         = vx*vy
        
        div_x = (1j)*params['k']*un
        div_y = (1.0/(np.sqrt(2)*params['yT']))*(n*Dn(n-1, zeta) - Dn(n+1, zeta))*vx
        #### y gradient using recursion relation
        div   = div_x + div_y
    
    if n==-1 :
                
        ########## Calculate dispersion relation  ####### 
        ## For Kelvin wave ##
        params['wd']  = kn

        params['w_scale'] = np.sqrt(params['beta']*params['c'])
        params['w']       = params['wd']*params['w_scale']
        params['k_scale'] = 1/params['yT']    
        params['k']       = kn*params['k_scale']   
        ###################################################
        k, x = params['k'], grids['x']
        
        qn_plus_1  =     Dn(n+1, zeta)           
        
        ux  = ((1)*np.exp(1j*(kn*lon - params['w']*t)))
        uy  = (qn_plus_1)/2
        un  = ux*uy
        
        phix  = ((1)*np.exp(1j*(kn*lon - params['w']*t)))
        phiy  = params['c']*(qn_plus_1)/2
        phin  = phix*phiy
        
        vx    = ((-1j)*np.exp(1j*(kn*lon - params['w']*t)))
        vy    = 0*Dn(n+1, zeta)
        vn    = vx*vy
        
        div_x = (1j)*params['k']*un
        div_y = 0
        
        div   = div_x + div_y
          
            
    if (n==0) :
        
        ########## Calculate dispersion relation  ####### 
        ## For Rossby wave ##
        args    = (kn, n)
        
        if kn > 0:
            guess   =  (kn/2)*( 1+ (1+4/kn**2)**0.5 ) ## theoretical estimate for MRG eastward
        if kn < 0:
            guess   =  (kn/2)*( 1- (1+4/kn**2)**0.5 ) ## theoretical estimate for MRG westward

        params['wd']  = fsolve(solve_dispersion, guess, args)

        params['w_scale'] = np.sqrt(params['beta']*params['c'])
        params['w']       = params['wd']*params['w_scale']
        params['k_scale'] = 1/params['yT']    
        params['k']       = kn*params['k_scale']   
        ###################################################
          
        k, x = params['k'], grids['x']
        
        w_m_ck     =  (params['w'] - params['k']*params['c'])
        
        qn_plus_1  =   Dn(n+1, zeta)*(1/w_m_ck)*np.sqrt(two_beta_c)                
        
        ux    = ((1)*np.exp(1j*(kn*lon - params['w']*t)))
        uy    = (qn_plus_1)/2
        un    = ux*uy
        
        phix  = ((1)*np.exp(1j*(kn*lon - params['w']*t)))
        phiy  = params['c']*(qn_plus_1)/2
        phin  = phix*phiy
        
        vx    = ((-1j)*np.exp(1j*(kn*lon - params['w']*t)))
        vy    = Dn(n, zeta)
        vn    = vx*vy
        
        div_x = (1j)*params['k']*un
        div_y = (1.0/(np.sqrt(2)*params['yT']))*(- Dn(n+1, zeta))*vx
        
        div   = div_x + div_y

    
    vec = dict(n=n, un=un.real, vn=vn.real, phin=phin.real,                \
               ux=ux.real, uy=uy.real, vx=vx.real, vy=vy.real, phix=phix.real, phiy=phiy.real,               \
               zeta=zeta, div=div, div_x = div_x.real, div_y=div_y.real,                \
               lon=np.rad2deg(grids['x']/params['R']),                \
               lat=np.rad2deg(grids['y']/params['R']),                \
               trap_scale = np.rad2deg(params['yT']/params['R']), params=params,                \
               wd=params['wd'], w=params['w'], kn=kn, k=params['k'], grids=grids )
    
    return vec



def MODE_N(t = 0, params=None, grids=None):
    
    py.figure(figsize=(4,5))
    def return_wd_kn(dicti, c='k', ms = 8):
        wd, kn = dicti['wd'], dicti['kn']
        py.plot(kn, wd, 'o', color=c, mfc='none', ms=ms)        
        return wd, kn
    
    matsuno_modes = {}
    matsuno_modes['Kelvin'] = {}
    matsuno_modes['MRG']    = {}
    matsuno_modes['Rossby'] = {}
    matsuno_modes['WIG']    = {}
    matsuno_modes['EIG']    = {}
    
    for kn in np.append(np.arange(-10, 0, 1), np.arange(1, 11, 1)):
        matsuno_modes['Kelvin']['(kn=%d)'%(kn)]    = vector(n=-1, kn=kn,   t=t, params=params,  grids= grids)
        return_wd_kn(matsuno_modes['Kelvin']['(kn=%d)'%(kn)], 'r')
        matsuno_modes['MRG']   ['(kn=%d)'%(kn)]    = vector(n=0, kn=kn,    t=t, params=params,  grids= grids)
        return_wd_kn(matsuno_modes['MRG']   ['(kn=%d)'%(kn)], 'b')
    
    for n in np.arange(1,6,1):
        matsuno_modes['Rossby']['(n=%d)'%(n)]={}
        for kn in np.arange(-10, 0, 1):
            matsuno_modes['Rossby']['(n=%d)'%(n)]['(kn=%d)'%(kn)] = vector(n=n, kn=kn,  t=t, typi = 'Rossby', params=params,  grids= grids)
            return_wd_kn(matsuno_modes['Rossby']['(n=%d)'%(n)]['(kn=%d)'%(kn)],  'c')
    
    for n in np.arange(1,6,1):
        matsuno_modes['WIG']['(n=%d)'%(n)]={}
        for kn in  np.arange(-10, 0, 1):
            matsuno_modes['WIG']['(n=%d)'%(n)]['(kn=%d)'%(kn)] = vector(n=n, kn=kn,  t=t, typi = 'WIG', params=params,  grids= grids)
            return_wd_kn(matsuno_modes['WIG']['(n=%d)'%(n)]['(kn=%d)'%(kn)],  'm')
            
    for n in np.arange(1,6,1):
        matsuno_modes['EIG']['(n=%d)'%(n)]={}
        for kn in  np.arange(0, 11, 1):
            matsuno_modes['EIG']['(n=%d)'%(n)]['(kn=%d)'%(kn)] = vector(n=n, kn=kn,  t=t, typi = 'EIG', params=params,  grids= grids)
            return_wd_kn(matsuno_modes['EIG']['(n=%d)'%(n)]['(kn=%d)'%(kn)],  'orange' )
            
    return matsuno_modes



Hmean = 500   
params = dict(alpha  = 1/(10*24*3600),      \
              Hmean = Hmean,                \
              beta   = 2*7.292e-5/6371e3,       \
              L      = np.pi*6371e3/4, R=6371e3, \
              xwidth = 20, ywidth=10, xpos=30)

params['c'] = np.sqrt(10*Hmean)

############ This extra parameters are for the case when there will be shear in the future. Ignore for now. ############
ep=0
params['yT0']       = np.sqrt(params['c']/params['beta'])
params['epsilon']   = 1.0
params['yT']        = np.sqrt(params['epsilon'])*params['yT0']
params['dU2dy2']    = params['beta']*(1-1/params['epsilon'])
params['ywidth']    = np.sqrt(2)*params['yT0'] ### the ywidth is by default chosen to be sqrt(2 c/beta). So that Q = Qo for \epsilon = 1
########################################################################################################################


#### 
source='/data/pbarpanda/spherical_SWE/evaluate_final_budget/transient_U_propagate_forcing_diff_Heq/dt_150_Q_forcing_10_forcing_y_0_Hmean_200_forcing_phase_speed_0_ms/U_up_days_217/H0_2500/post_process/'
dicti = h5saveload.load_dict_from_hdf5(source+'raw_field_dict.hdf5')

lat = (dicti['0']['lat'])
lats = np.deg2rad(lat)
lon = (dicti['0']['lon'])
lons = np.deg2rad(lon)

grids = dict(x = lons*params['R'], \
             y = lats*params['R'] )

       

if __name__ == "__main__": 
        

    matsuno_modes= MODE_N(t = 0, params=params, grids=grids)
    h5saveload.save_dict_to_hdf5(matsuno_modes, '/data/pbarpanda/spherical_SWE/matsuno_modes_%d.hdf5'%(Hmean))
    


# # In[11]:


# py.figure(figsize=(25,6))
# lo, la = 8, 2

# py.subplot(1,3,1)

# ve = matsuno_modes['WIG']['(n=1)']['(kn=-1)']
# py.contourf(ve['lon'], ve['lat'], ve['div'], cmap=cmap)
# colorbar()
# py.quiver(ve['lon'][::lo], ve['lat'][::la], ve['un'][::la, ::lo], ve['vn'][::la, ::lo], cmap=py.cm.seismic)
# py.title(ve['trap_scale'])
# py.ylim(-40,40)

# py.subplot(1,3,2)

# ve = matsuno_modes['MRG']['(kn=1)']
# py.contourf(ve['lon'], ve['lat'], ve['div'], cmap=cmap)
# colorbar()
# py.quiver(ve['lon'][::lo], ve['lat'][::la], ve['un'][::la, ::lo], ve['vn'][::la, ::lo], cmap=py.cm.seismic)
# py.title(ve['trap_scale'])
# py.ylim(-40,40)

# py.subplot(1,3,3)

# ve = matsuno_modes['Rossby']['(n=1)']['(kn=-1)']
# py.contourf(ve['lon'], ve['lat'], ve['div'], cmap=cmap)
# colorbar()
# py.quiver(ve['lon'][::lo], ve['lat'][::la], ve['un'][::la, ::lo], ve['vn'][::la, ::lo], cmap=py.cm.seismic)
# py.title(ve['trap_scale'])
# py.ylim(-40,40)



# # In[14]:


# py.figure(figsize=(25,6))
# lo, la = 8, 2

# py.subplot(1,3,1)

# ve = matsuno_modes['MRG']['(kn=-1)']
# py.contourf(ve['lon'], ve['lat'], ve['phin'], cmap=cmap)
# colorbar()
# py.quiver(ve['lon'][::lo], ve['lat'][::la], ve['un'][::la, ::lo], ve['vn'][::la, ::lo], cmap=py.cm.seismic)
# py.title(ve['trap_scale'])
# py.ylim(-40,40)

# py.subplot(1,3,2)

# ve = matsuno_modes['MRG']['(kn=1)']
# py.contourf(ve['lon'], ve['lat'], ve['phin'], cmap=cmap)
# colorbar()
# py.quiver(ve['lon'][::lo], ve['lat'][::la], ve['un'][::la, ::lo], ve['vn'][::la, ::lo], cmap=py.cm.seismic)
# py.title(ve['trap_scale'])
# py.ylim(-40,40)

# py.subplot(1,3,3)

# ve = matsuno_modes['Kelvin']['(kn=1)']
# py.contourf(ve['lon'], ve['lat'], ve['phin'], cmap=cmap)
# colorbar()
# py.quiver(ve['lon'][::lo], ve['lat'][::la], ve['un'][::la, ::lo], ve['vn'][::la, ::lo], cmap=py.cm.seismic)
# py.title(ve['trap_scale'])
# py.ylim(-40,40)


# # In[17]:


# py.contourf(ve['lon'], ve['lat'], ve['un'], cmap=cmap); colorbar(20)


# # In[59]:


# Rossby = matsuno_modes['Rossby']['(n=1)']['(kn=-1)']
# Kelvin = matsuno_modes['Kelvin']['(kn=1)']

# py.figure(figsize=(30,8))

# py.subplot(1,2,1)
# phin = Rossby['phin']*0 + Kelvin['phin']
# v    = Rossby['vn']*0   + Kelvin['vn']
# u    = Rossby['un']*0  + Kelvin['un']
# py.contourf(ve['lon'], ve['lat'], phin, cmap=cmap); colorbar(20)
# py.ylim(-40,40)

# phin = Rossby['phin'] + Kelvin['phin']*0
# v    = Rossby['vn']   + Kelvin['vn']*0
# u    = Rossby['un']   + Kelvin['un']*0

# py.contour(ve['lon'], ve['lat'], phin, colors='k'); #colorbar(20)
# lo, la = 10, 3
# py.quiver(ve['lon'][::lo], ve['lat'][::la], u[::la, ::lo], v[::la, ::lo], cmap=py.cm.seismic)

# py.tick_params(labelsize=20)
# py.ylim(-40,40)


# fac=1
# py.subplot(1,2,2)
# phin = Rossby['phin'] + Kelvin['phin']*fac
# v    = Rossby['vn']   + Kelvin['vn']*fac
# u    = Rossby['un']   + Kelvin['un']*fac

# py.contourf(ve['lon'], ve['lat'], phin, cmap=cmap); colorbar(20)
# lo, la = 10, 3
# py.quiver(ve['lon'][::lo], ve['lat'][::la], u[::la, ::lo], v[::la, ::lo], cmap=py.cm.seismic)
# py.ylim(-40,40)

# py.tick_params(labelsize=20)


# # In[82]:


# Rossby = matsuno_modes['Rossby']['(n=1)']['(kn=-1)']
# Kelvin = matsuno_modes['Kelvin']['(kn=1)']

# roll=0

# py.figure(figsize=(30,8))

# py.subplot(1,2,1)
# phin = Rossby['phin']*0 +  np.roll(Kelvin['phin'], roll, axis=-1)
# v    = Rossby['vn']*0   +  np.roll(Kelvin['vn'], roll, axis=-1)
# u    = Rossby['un']*0   +  np.roll(Kelvin['un'], roll, axis=-1)
# py.contourf(ve['lon'], ve['lat'], phin, cmap=cmap); colorbar(20)
# py.ylim(-40,40)

# phin = Rossby['phin'] + Kelvin['phin']*0
# v    = Rossby['vn']   + Kelvin['vn']*0
# u    = Rossby['un']   + Kelvin['un']*0

# py.contour(ve['lon'], ve['lat'], phin, colors='k'); #colorbar(20)
# lo, la = 10, 2
# py.quiver(ve['lon'][::lo], ve['lat'][::la], u[::la, ::lo], v[::la, ::lo], cmap=py.cm.seismic)

# py.tick_params(labelsize=20)
# py.ylim(-40,40)


# fac=1
# py.subplot(1,2,2)
# phin = Rossby['phin'] + np.roll(Kelvin['phin'], roll, axis=-1)*fac
# v    = Rossby['vn']   + np.roll(Kelvin['vn'], roll, axis=-1)*fac
# u    = Rossby['un']   + np.roll(Kelvin['un'], roll, axis=-1)*fac

# py.contourf(ve['lon'], ve['lat'], phin, cmap=cmap); colorbar(20)
# lo, la = 10, 2
# py.quiver(ve['lon'][::lo], ve['lat'][::la], u[::la, ::lo], v[::la, ::lo], cmap=py.cm.seismic)
# py.ylim(-40,40)

# py.tick_params(labelsize=20)






# # Rossby = matsuno_modes['Rossby']['(n=1)']['(kn=-1)']
# # Kelvin = matsuno_modes['Kelvin']['(kn=1)']

# roll=+80

# py.figure(figsize=(30,8))

# py.subplot(1,2,1)
# phin = Rossby['phin']*0 +  np.roll(Kelvin['phin'], roll, axis=-1)
# v    = Rossby['vn']*0   +  np.roll(Kelvin['vn'], roll, axis=-1)
# u    = Rossby['un']*0   +  np.roll(Kelvin['un'], roll, axis=-1)
# py.contourf(ve['lon'], ve['lat'], phin, cmap=cmap); colorbar(20)
# py.ylim(-40,40)

# phin = Rossby['phin'] + Kelvin['phin']*0
# v    = Rossby['vn']   + Kelvin['vn']*0
# u    = Rossby['un']   + Kelvin['un']*0

# py.contour(ve['lon'], ve['lat'], phin, colors='k'); #colorbar(20)
# lo, la = 10, 2
# py.quiver(ve['lon'][::lo], ve['lat'][::la], u[::la, ::lo], v[::la, ::lo], cmap=py.cm.seismic)

# py.tick_params(labelsize=20)
# py.ylim(-40,40)


# fac=1
# py.subplot(1,2,2)
# phin = Rossby['phin'] + np.roll(Kelvin['phin'], roll, axis=-1)*fac
# v    = Rossby['vn']   + np.roll(Kelvin['vn'], roll, axis=-1)*fac
# u    = Rossby['un']   + np.roll(Kelvin['un'], roll, axis=-1)*fac

# py.contourf(ve['lon'], ve['lat'], phin, cmap=cmap); colorbar(20)
# lo, la = 10, 2
# py.quiver(ve['lon'][::lo], ve['lat'][::la], u[::la, ::lo], v[::la, ::lo], cmap=py.cm.seismic)
# py.ylim(-40,40)

# py.tick_params(labelsize=20)




