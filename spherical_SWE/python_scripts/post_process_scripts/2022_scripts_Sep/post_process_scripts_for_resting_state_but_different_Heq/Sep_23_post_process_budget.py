#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
# import shtns
import pylab as py
import matplotlib
from tqdm import tqdm
import sys

sys.path.append('/data/pbarpanda/python_scripts/modules/')
import logruns as logruns
import save_and_load_hdf5_files as h5saveload
import eulerian_fluxes as eflux
import netcdf_utilities as ncutil
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
from tqdm import tqdm
import glob
from PIL import Image

import time as ti
import numpy.ma as ma
import math

sys.path.append('/data/pbarpanda/python_scripts/modules/')
import logruns as logruns
import save_and_load_hdf5_files as h5saveload
import eulerian_fluxes as eflux
import netcdf_utilities as ncutil
from obspy.geodetics import kilometers2degrees
import momentum_advection_class as momentum_advect
from PIL import Image
import imageio
from IPython.display import Video
import scipy.special as special
from scipy import integrate
from IPython.display import Image
import matplotlib.ticker as ticker

# import cmasher as cmr
# cmap = cmr.rainforest                   # CMasher
# cmap = plt.get_cmap('cmr.rainforest')   # MPL

import matplotlib as mpl
from cycler import cycler

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.edgecolor'] = 'black'
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['lines.color'] = 'black'

import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'  ### This is because NOAA PSL lab computers are somehow not able to use 

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
      [  "darkred", "darkorange", "pink", "white", "white","skyblue", "dodgerblue", "navy"][::-1])


def colorbar(fontsize=20):
    cbar = py.colorbar()
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(fontsize)
    cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
    
def colorbar2(fontsize=20, im=None, AX=None, fig=py, ):
    if im is None:
        cbar = fig.colorbar()
    else:
        cbar = fig.colorbar(im, ax=AX)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(fontsize)
    cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)

            
import warnings
warnings.filterwarnings('ignore')

import os
import glob

def remove_files(direc):
    files = glob.glob(direc, recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
            
def fmt(x, pos):
    a, b = ('%1.2e'%(x)).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)
fmt1 = ticker.FuncFormatter(fmt)


# In[105]:


def eddy(X):
    return X-X.mean(axis=-1)[...,None]

def zmean(X):
    return X.mean(axis=-1)[...,None]


def t_eddy(X):
    return X-X.mean(axis=0)[None,...]

def locate(Y, x, gap=0.1):
    index = np.where(np.abs(Y-x) < gap)[0][0]
    return index

def locate(Y, x, gap=None):
    if gap is None:
        gap = 0.5*np.abs(np.diff(Y))[0] 
    index = np.where(np.abs(Y-x) <= gap)[0][0]
    return index


# In[106]:


def find_index_for_umax(UMAX, dicti):
    if dicti['U'].shape[0] == 4752:
        start=400; end=2256
    if dicti['U'].shape[0] == 9552:
        start=500; end=6800
    U_mean = transient_U['U'][start:end].mean(axis=-1)
    U_max  = np.max(U_mean, axis=1)
    umax_index = np.squeeze(np.where(np.isclose(U_max, UMAX, atol = 0.1)==True))
    umax_index = np.array([(int(i+start)) for i in umax_index])
    return umax_index, U_max


# In[117]:


def return_fields_FOR_U(UMAX, dicti, ps):
    
    dicti2 = dicti

    umax_index, U_max = find_index_for_umax(UMAX, dicti=dicti)
    if umax_index.any():
        n = umax_index[0]
        lrange = 10
        lat_bnd, lat_bnd1 = 0, 50

        KEY = 'div'

        lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )

        c1 = np.sqrt(dicti2['PHI'][n, ...].mean(axis=-1))[:,None]
        c2 = np.sqrt(dicti2['phi_B'])

        c=c1

        u   = (dicti2['U'])[n, ...]
        v   = (dicti2['V'])[n, ...]
        phi = (dicti2['PHI'])[n, ...]
        
        dvrt_dt = np.gradient(dicti['vrt'], dicti['T_in_days'], axis=0)/(24*3600)
               
        dicti = {'U'  : u, 'V': v, 'PHI': phi,        \
                 'lat': np.rad2deg((dicti['lats'])),        \
                 'lon': np.rad2deg((dicti['lons'])),         \
                 'phi_B'      : dicti2['phi_B'], 'U_max': U_max,       \
#                  'phi_T'      : dicti['phi_T'][n, ...],         \
                 'phi_forcing': dicti['phi_forcing'][n,...],        \
                 'div'  : dicti['div'][n,...],                \
                 'vrt'  : dicti['vrt'][n,...],        \
                 'dvrt_dt':dvrt_dt[n,...], 'cf':ps, \
                 'vrt_forcing': dicti['vrt_forcing'][n,...],}
        
        
        dims   = ['lat', 'lon']
        coords = {"lon": np.float32(dicti['lon']), "lat":   np.float32(dicti['lat'])},
        
        return dicti
    
    
    
#### This is yet to be calculated ####
def return_fields_FOR_UV_decomp(UMAX, dicti2, dicti):

    umax_index, U_max = find_index_for_umax(UMAX, dicti2)
    
    if umax_index.any():
        n = umax_index[0]
        lrange = 10
        lat_bnd, lat_bnd1 = 0, 50

        KEY = 'div'

        lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )

        uD = (dicti['uD'])[n, ...]
        vD = (dicti['vD'])[n, ...]
        uR = (dicti['uR'])[n, ...]
        vR = (dicti['vR'])[n, ...]
        sf = (dicti['sf'])[n, ...]
        vp = (dicti['vp'])[n, ...]
        div = (dicti['div'])[n, ...]
        vrt = (dicti['vrt'])[n, ...]
        
                       
        dicti = {'uD'  : uD, 'vD'  : vD,          \
                 'uR'  : uR, 'vR'  : vR,          \
                 'sf'  : sf, 'vp'  : vp,          \
                 'div' : div, 'vrt': vrt,         \
                 'lat': np.rad2deg((dicti['lats'])),     \
                 'lon': np.rad2deg((dicti['lons'])), }
                
        return dicti
    
    ##tropics
bnd2 = 40
# lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )

def tropical(field, bnd1=0, bnd2=20, sign=1):
    field = ma.masked_where(   ((( np.abs(latt) < bnd1)  | ( np.abs(latt) > bnd2))   |   (sign*field < 0) ), field)
    return field.filled(np.nan)

##subtropics
def subtropical(field,  bnd1=20, bnd2=40, sign=1):
    field = ma.masked_where(    ((( np.abs(latt) < bnd1)  | ( np.abs(latt) > bnd2))  |   (sign*field < 0) ) , field)
    return field.filled(np.nan)

def extratropical(field,  bnd1=40, bnd2=80, sign = 1):
    field = ma.masked_where(    ((( np.abs(latt) < bnd1)  | ( np.abs(latt) > bnd2)) |   (sign*field < 0) ), field)
    return field.filled(np.nan)

def forcing_region(field, forcing, sign = 1, scale=4):
    
    max_phi = np.max(forcing)/scale
    field   = ma.masked_where( sign*forcing < max_phi, field)
    return field.filled(np.nan)

def return_max_div_location(field, bnd1=0, bnd2=20, sign = 1):
    
    trop_field        = tropical(field, bnd1=bnd1, bnd2=bnd2, sign=sign)
    max_trop_field    = np.nanmax(trop_field*sign)
    index             = np.where(trop_field*sign == max_trop_field)
    la_pos, lo_pos = latt[index], lonn[index]
    return la_pos, lo_pos

def d_by_dx(lat, lon, field, logging_object=None):
    if logging_object is None:
        logging_object = logruns.default_log(logfilename   = 'momentum',  log_directory = './logs/')
    obj_momentum       = momentum_advect.momentum_advection(field, field, lat, lon, logging_object)            
    dudx  = obj_momentum.spher_div_x(field, lat,  lon, ax=-1)
    return np.squeeze(dudx)

def d_by_dy(lat, lon, field, logging_object=None):
    if logging_object is None:
        logging_object = logruns.default_log(logfilename   = 'momentum',  log_directory = './logs/')
    obj_momentum       = momentum_advect.momentum_advection(field, field, lat, lon, logging_object)            
    dvdy  = obj_momentum.spher_div_y(field, lat,  lon, ax=-2)
    return np.squeeze(dvdy)

def tropical_mask(field, bnd1=0, bnd2=bnd2):
    field = ma.masked_where(   ((( np.abs(latt) < bnd1)  | ( np.abs(latt) > bnd2))   ), field)
    return field.filled(np.nan)

def tropical_mask_outside(field, bnd1=0, bnd2=bnd2):
    field = ma.masked_where(   ((( np.abs(latt) > bnd1)  & ( np.abs(latt) < bnd2))   ), field)
    return field.filled(np.nan)


def rms_near_equator(field, lat_bnd=2):
    forcing_region_field = tropical_mask( R*field,  0, lat_bnd)
    rms_div              = np.sqrt(np.nanmean(np.nanmean((forcing_region_field**2), axis=-1), axis=-1))
    return rms_div


# In[113]:


def vorticity_budget(ds):

    f = 2*7.2921e-5*np.sin(np.deg2rad(ds['lat']))
    grav = 10

    #### LHS terms
    vrt_x            =   d_by_dx(ds['lat'], ds['lon'] ,ds['V'][None,...])
    vrt_y            = - d_by_dy(ds['lat'], ds['lon'], ds['U'][None,...])
    vrt              =   vrt_x + vrt_y

    cos_phi   = np.cos(np.deg2rad(ds['lat']))[:,None]
    abs_vrt   = (f[:, None] + vrt)
    u_abs_vrt = ds['U']*(abs_vrt)
    v_abs_vrt = ds['V']*(abs_vrt)


    div_x = d_by_dx(ds['lat'], ds['lon'] ,(ds['U'])[None,...])
    div_y = d_by_dy(ds['lat'], ds['lon'], (ds['V'])[None,...]) 
    div              = div_x + div_y
    div_abs_vrt_flux = d_by_dx(ds['lat'], ds['lon'], (abs_vrt*ds['U'])[None,...]) + d_by_dy(ds['lat'], ds['lon'], (abs_vrt*ds['V'])[None,...])    
    vrt_advection    = (ds['U'])*d_by_dx(ds['lat'], ds['lon'], abs_vrt[None,...]) + (ds['V'])*d_by_dy(ds['lat'], ds['lon'], (abs_vrt/cos_phi)[None,...])*cos_phi    
    
    
    ########### Look at the linear terms only #########
    
    stretching_term_div_prime      =  eddy(div)*(zmean(abs_vrt))
    stretching_term_div_mean       = zmean(div)*(eddy(abs_vrt))
    
    linear_adv_Vprime     =  eddy(ds['V'])*d_by_dy(ds['lat'], ds['lon'], (abs_vrt*0+zmean(abs_vrt)/cos_phi)[None,...])*cos_phi
    linear_adv_Umean      =  zmean(ds['U'])*d_by_dx(ds['lat'], ds['lon'], (eddy(abs_vrt))[None,...])
    
    linear_adv_Vmean      =  zmean(ds['V'])*d_by_dy(ds['lat'], ds['lon'], (eddy(abs_vrt))[None,...])
    linear_adv_Uprime     =  eddy(ds['U'])*d_by_dx(ds['lat'], ds['lon'],  (abs_vrt*0+zmean(abs_vrt)/cos_phi)[None,...])*cos_phi  

    K_M = 20*24*3600
    damping_term         = -ds['vrt']/K_M
    
    VRT_dicti = {    'div'  : div,   'vrt': vrt,  'abs_vrt':abs_vrt,      \
                     'div_x': div_x, 'div_y': div_y,                      \
                     'vrt_x': vrt_x, 'vrt_y': vrt_y,                      \
                     'div_abs_vrt_flux': div_abs_vrt_flux,                \
                     'vrt_advection': vrt_advection,                      \
                     'stretching_term': div*abs_vrt,                      \
                     'stretching_term_div_prime': stretching_term_div_prime,             \
                     'stretching_term_div_mean': stretching_term_div_mean,               \
                     'linear_adv_Vprime': linear_adv_Vprime,                \
                     'linear_adv_Umean': linear_adv_Umean,                   \
                     'linear_adv_Vmean': linear_adv_Vmean,                   \
                     'linear_adv_Uprime': linear_adv_Uprime ,                \
                     'lat':ds['lat'], 'lon':ds['lon'], 'damping_term':damping_term, 'dvrt_dt':ds['dvrt_dt'] }
              
    return VRT_dicti 


# In[114]:


def vorticity_budget_new(ds, ps):

    f = 2*7.2921e-5*np.sin(np.deg2rad(ds['lat']))
    grav = 10

    #### LHS terms
    vrt_x            =   d_by_dx(ds['lat'], ds['lon'] ,ds['V'][None,...])
    vrt_y            = - d_by_dy(ds['lat'], ds['lon'], ds['U'][None,...])
    vrt              =   vrt_x + vrt_y

    cos_phi   = np.cos(np.deg2rad(ds['lat']))[:,None]
    abs_vrt   = (f[:, None] + vrt)
    u_abs_vrt = ds['U']*(abs_vrt)
    v_abs_vrt = ds['V']*(abs_vrt)


    div_x = d_by_dx(ds['lat'], ds['lon'] ,(ds['U'])[None,...])
    div_y = d_by_dy(ds['lat'], ds['lon'], (ds['V'])[None,...]) 
    div   = div_x + div_y
        
    ZETA_a                     = zmean(abs_vrt) + div*0
    ########### Look at the linear terms only #########
    
    ZETA_a_D                     =  eddy(div)*(zmean(abs_vrt))
    
    beta_term                  =  -eddy(ds['V'])*d_by_dy(ds['lat'], ds['lon'], ( ZETA_a /cos_phi)[None,...])*cos_phi   
    hadley_term                =  -d_by_dy(ds['lat'], ds['lon'], ( zmean(ds['V'])*eddy(ds['vrt']) )[None,...]) 
    zonal_wind_term            =  -d_by_dx(ds['lat'], ds['lon'], ( zmean(ds['U'])*eddy(ds['vrt']) )[None,...]) 
    cf_term                    =   d_by_dx(ds['lat'], ds['lon'], ( ps*eddy(ds['vrt']) )[None,...]) 
    vrt_forcing_term           =  eddy(ds['vrt_forcing'])
    
    K_M = 20*24*3600
    damping_term               =  -eddy(ds['vrt'])/K_M
    
       
    VRT_dicti = {'div'  : div,   'vrt': vrt,  'abs_vrt':abs_vrt,          \
                 'div_x': div_x, 'div_y': div_y,                   \
                 'vrt_x': vrt_x, 'vrt_y': vrt_y,                    \
                 'ZETA_a_D': ZETA_a_D , 'beta_term': beta_term, \
                 'hadley_term':hadley_term, 'zonal_wind_term':zonal_wind_term,         \
                 'div_mean_winds_vrt': hadley_term + zonal_wind_term,             \
                 'D' : eddy(div), 'ZETA_a': ZETA_a,                 \
                 'lat':ds['lat'], 'lon':ds['lon'],                \
                 'damping_term':damping_term, 'tendency_term':-eddy(ds['dvrt_dt']),           \
                 'dy_ZETA_a': d_by_dy(ds['lat'], ds['lon'], ( ZETA_a/cos_phi)[None,...])*cos_phi,          \
                 'V': eddy(ds['V']), 'cf':ps, 'cf_term':cf_term, 'vrt_forcing_term':vrt_forcing_term}

    
    var_name = {}
    var_name['ZETA_a_D']         = r'$\overline{\zeta_a} \left(D^{*}\right)$'
    var_name['beta_term']        = r'$-v^{*}\left(\partial_y{\overline{\zeta_a}}\right)$'
    var_name['hadley_term']      = r'$-\partial_y\left({\overline{V}\zeta^{*}}\right)$'
    var_name['zonal_wind_term']  = r'$-\partial_x\left({\overline{U}\zeta^{*}}\right)$'
    var_name['div']              = r'$R \overrightarrow{\nabla}.{\overrightarrow{u}^{\ *}}$ [m/s]'
    var_name['div_x']            = r'$R \dfrac{du^{*}}{dx}$ [m/s]'
    var_name['div_y']            = r'$R \dfrac{dv^{*}}{dy}$ [m/s]'
    var_name['V']                = r'$ v^{*}$'
    var_name['dy_ZETA_a']        = r'$\left(\partial_y{\overline{\zeta_a}}\right)$'
    var_name['ZETA_a']           = r'$\overline{\zeta_a}$'
    var_name['damping_term']     = r'$-\left(\zeta^{*} / \kappa_M \right)$'
    var_name['tendency_term']    = r'$-\partial_t\left(\zeta^{*}\right)  / \overline{\zeta_a}$'
    var_name['vrt_forcing_term'] = r'$\left(F_{\zeta}\right)$'

    
    div_name = {}
    div_name['ZETA_a_D']         = r'$\left(D^{*}\right)$'
    div_name['beta_term']        = r'$-v^{*}\left(\partial_y{\overline{\zeta_a}}\right)/ \overline{\zeta_a}$'
    div_name['hadley_term']      = r'$-\partial_y\left({\overline{V}\zeta^{*}}\right)  / \overline{\zeta_a}$'
    div_name['zonal_wind_term']  = r'$-\partial_x\left({\overline{U}\zeta^{*}}\right)  / \overline{\zeta_a}$'    
    div_name['damping_term']     = r'$-\left(\zeta^{*} / \kappa_M\right)/ \overline{\zeta_a}$'
    div_name['tendency_term']    = r'$-\partial_t\left(\zeta^{*}\right)  / \overline{\zeta_a}$'
    div_name['vrt_forcing_term'] = r'$\left(F_{\zeta}\right)/ \overline{\zeta_a}$'

    return VRT_dicti, var_name, div_name


# In[115]:


def vorticity_budget_new_rot_div(ds, dUV):

    f    = 2*7.2921e-5*np.sin(np.deg2rad(ds['lat']))
    grav = 10

    #### LHS terms
    vrt_x            =   d_by_dx(ds['lat'], ds['lon'], ds['V'][None,...])
    vrt_y            = - d_by_dy(ds['lat'], ds['lon'], ds['U'][None,...])
    vrt              =   vrt_x + vrt_y

    cos_phi   = np.cos(np.deg2rad(ds['lat']))[:,None]
    abs_vrt   = (f[:, None] + vrt)

    div_x = d_by_dx(ds['lat'], ds['lon'] ,(ds['U'])[None,...])
    div_y = d_by_dy(ds['lat'], ds['lon'], (ds['V'])[None,...]) 
    div   = div_x + div_y
        
    ZETA_a                         = zmean(abs_vrt) + div*0
    ########### Look at the linear terms only #########
    
    ZETA_a_D                       =  eddy(div)*(zmean(abs_vrt))
    
    beta_term_rot                  =  -eddy(dUV['vR'])*d_by_dy(ds['lat'], ds['lon'], ( ZETA_a /cos_phi)[None,...])*cos_phi   
    beta_term_div                  =  -eddy(dUV['vD'])*d_by_dy(ds['lat'], ds['lon'], ( ZETA_a /cos_phi)[None,...])*cos_phi   
    
    hadley_term_rot                =  -d_by_dy(ds['lat'], ds['lon'], ( zmean(dUV['vR'])*eddy(ds['vrt']) )[None,...]) 
    hadley_term_div                =  -d_by_dy(ds['lat'], ds['lon'], ( zmean(dUV['vD'])*eddy(ds['vrt']) )[None,...]) 
    
    zonal_wind_term_rot            =  -d_by_dx(ds['lat'], ds['lon'], ( zmean(dUV['uR'])*eddy(ds['vrt']) )[None,...]) 
    zonal_wind_term_div            =  -d_by_dx(ds['lat'], ds['lon'], ( zmean(dUV['uD'])*eddy(ds['vrt']) )[None,...]) 
        
    
    VRT_dicti = {'ZETA_a_D': ZETA_a_D ,                              'beta_term_rot': beta_term_rot,            \
                 'beta_term_div': beta_term_div,                     'hadley_term_rot':hadley_term_rot,         \
                 'hadley_term_div':hadley_term_div,                  'zonal_wind_term_rot':zonal_wind_term_rot, \
                 'zonal_wind_term_div':zonal_wind_term_div,          'D' : eddy(div), 'ZETA_a': ZETA_a,                      \
                 'lat':ds['lat'], 'lon':ds['lon'], }
    
    var_name = {}
    var_name['ZETA_a_D']             = r'$\left(D^{*}\right)$'
    var_name['beta_term_rot']        = r'$-{v_{r}}^{*}\left(\partial_y{\overline{\zeta_a}}\right)/ \overline{\zeta_a}$'
    var_name['beta_term_div']        = r'$-{v_{d}}^{*}\left(\partial_y{\overline{\zeta_a}}\right)/ \overline{\zeta_a}$'

    var_name['hadley_term_rot']      = r'$-\partial_y\left({\overline{{V_{r}}}\zeta^{*}}\right)  / \overline{\zeta_a}$'
    var_name['hadley_term_div']      = r'$-\partial_y\left({\overline{{V_{d}}}\zeta^{*}}\right)  / \overline{\zeta_a}$'

    var_name['zonal_wind_term_rot']  = r'$-\partial_x\left({\overline{{U_{r}}}\zeta^{*}}\right)  / \overline{\zeta_a}$'
    var_name['zonal_wind_term_div']  = r'$-\partial_x\left({\overline{{U_{d}}}\zeta^{*}}\right)  / \overline{\zeta_a}$'
              
    return VRT_dicti, var_name


# In[128]:


if __name__ == "__main__":

    source0  = '/data/pbarpanda/spherical_SWE/evaluate_final_budget/momentum_forcing_Aug_2022/*'
    sources  = glob.glob(source0)

    for source in sources:
        ps      = 0 #int(source.split('_')[-2]) ### Here the phase speed of the vorticity forcing is set to 0
        source = glob.glob(source+'/*')[0]

        if os.path.exists(source+'/velocity_decomp.hdf5'):

            if not os.path.exists(source+'/post_process/'):

                vars()['transient_U'] = h5saveload.load_dict_from_hdf5(source+'/spatial_data.hdf5', track=True)    
                vars()['transient_UV_decomposed'] = h5saveload.load_dict_from_hdf5(source+'/velocity_decomp.hdf5', track=True)  

                UMAX                    = np.arange(1e-3, 80, 2)
                raw_field_dict          = {}
                vrt_budget_dict         = {}
                veloc_decomp_dict       = {}
                vrt_budget_rot_div_dict = {} 

                for umax_key in tqdm(UMAX):

                    ds = return_fields_FOR_U(UMAX = int(umax_key), dicti = transient_U, ps=ps)
                    if ds is not None:

                        de, vrt_var_name, div_var_name  = vorticity_budget_new(ds, ps)
                        dUV = return_fields_FOR_UV_decomp(UMAX = int(umax_key), dicti2 = transient_U, dicti = transient_UV_decomposed)        
                        der, vrt_vel_decomp_name   = vorticity_budget_new_rot_div(ds, dUV)

                        raw_field_dict           [str(int(umax_key))] = ds
                        vrt_budget_dict          [str(int(umax_key))] = de
                        veloc_decomp_dict        [str(int(umax_key))] = dUV
                        vrt_budget_rot_div_dict  [str(int(umax_key))] = der


                h5saveload.make_sure_path_exists(source+'/post_process/')

                h5saveload.save_dict_to_hdf5(raw_field_dict,          source+'/post_process/raw_field_dict.hdf5')
                h5saveload.save_dict_to_hdf5(vrt_budget_dict,         source+'/post_process/vrt_budget_dict.hdf5')
                h5saveload.save_dict_to_hdf5(veloc_decomp_dict ,      source+'/post_process/veloc_decomp_dict.hdf5')
                h5saveload.save_dict_to_hdf5(vrt_budget_rot_div_dict, source+'/post_process/vrt_budget_rot_div_dict.hdf5')

                h5saveload.save_dict_to_hdf5(vrt_var_name, source+'/post_process/vrt_var_name.hdf5')
                h5saveload.save_dict_to_hdf5(div_var_name, source+'/post_process/div_var_name.hdf5')
                h5saveload.save_dict_to_hdf5(vrt_vel_decomp_name, source+'/post_process/vrt_vel_decomp_name.hdf5')


# In[ ]:




