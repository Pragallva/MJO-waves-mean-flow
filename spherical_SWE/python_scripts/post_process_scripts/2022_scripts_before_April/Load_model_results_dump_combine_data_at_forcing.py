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

import numpy as np
# import shtns
import pylab as py
import matplotlib
from tqdm import tqdm
import sys
import time as ti
import numpy.ma as ma

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
            
            

def eddy(X):
    return X-np.nanmean(X, axis=-1)[...,None]

def zmean(X):
    return np.nanmean(X, axis=-1)[...,None]

def t_eddy(X):
    return X-np.nanmean(X, axis=0)[None,...]

def locate(Y, x, gap=0.1):
    index = np.where(np.abs(Y-x) < gap)[0][0]
    return index

def locate(Y, x, gap=None):
    if gap is None:
        gap = 0.5*np.abs(np.diff(Y))[0] 
    index = np.where(np.abs(Y-x) <= gap)[0][0]
    return index

                    
def return_data_at_forcing_region(Hmean = 200, Q0 = 10, H0 = 0, PLOT=False, lat_lim = 40):
    
    dicti  = globals()['Q0_%d_loc_0_H0_%d_Hmean_%d_equation'%(Q0, H0, Hmean)]
    dicti2 = globals()['Q0_%d_loc_0_H0_%d_Hmean_%d'%(Q0, H0, Hmean)]
     
    PHI = dicti2['PHI']  
    
    R= 6371e3
    
    n = -350

    f = 2*7.2921e-5*np.sin((dicti2['lats']))[None, :, None]
    
    lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )

    logging_object = logruns.default_log(logfilename   = 'momentum',  log_directory = './logs/')
    obj_momentum   = momentum_advect.momentum_advection(dicti2['U'], dicti2['V'], np.rad2deg(dicti['lats']), np.rad2deg(dicti['lons']), logging_object)            
    dU_dy          = obj_momentum.spher_div_y(dicti2['U']*0+zmean(dicti2['U']), np.rad2deg(dicti['lats']), np.rad2deg(dicti['lons']), ax=-2)
    dV_dy          = obj_momentum.spher_div_y(dicti2['V']*0+zmean(dicti2['V']), np.rad2deg(dicti['lats']), np.rad2deg(dicti['lons']), ax=-2)

    div_x          = obj_momentum.spher_div_x(dicti2['U'], np.rad2deg(dicti['lats']), np.rad2deg(dicti['lons']), ax=-1)
    div_y          = obj_momentum.spher_div_y(dicti2['V'], np.rad2deg(dicti['lats']), np.rad2deg(dicti['lons']), ax=-2)
    
    div =  eddy(dicti['div'])
    total_div_vrt            = eddy(dicti['VRT_term1a'])  ## -D(f+vrt)  ## -eddy(dicti['div'] *(f + dicti['vrt']))
    linear_div_vrt           = eddy(dicti['div'])*(f - dU_dy)
    total_vortcity_advection = eddy(dicti['VRT_term1b'])  #### -v.del(vrt+f)
    total_vortcity_flux      = eddy(dicti['VRT_term1'])   #### -del.(v vrt + f)
    gradient_vrt             = eddy(np.gradient(dicti['vrt'], dicti['T_in_days']*24*3600, axis=0 ))
    
    div_x                    = eddy(div_x)
    div_y                    = eddy(div_y)
    
    def return_at_forcing(field=div, n=-350, PLOT=False):
                     
        max_phi = np.max(np.abs(eddy(dicti2['phi_forcing'][n, ...])))

        field1  = ma.masked_where((   (   eddy(dicti2['phi_forcing'][n, ...]) < max_phi/10) | (latt<0)), field[n,...]) 
        field2  = ma.masked_where((   (  -eddy(dicti2['phi_forcing'][n, ...]) < max_phi/10) | (latt<0)), field[n,...])

        divergence  = field1.filled(np.nan)
        convergence = field2.filled(np.nan)
        
        if PLOT:
            py.figure()
            py.contourf(lonn, latt, field1, cmap=cmap); py.colorbar()
            py.contourf(lonn, latt, field2, cmap=cmap); py.colorbar()
    
        sum_conv    = np.nansum(convergence)
        sum_div     = np.nansum( divergence)
                          
        return sum_conv, sum_div
    
    def return_spread_out_data(field=div, n=-350, lat_lim = 30):
                     
        max_phi = np.max(np.abs(eddy(dicti2['phi_forcing'][n, ...])))

        field1  = ma.masked_where((   (   eddy(dicti2['phi_forcing'][n, ...]) < max_phi/10)), field[n,...]) 
        field2  = ma.masked_where((   (  -eddy(dicti2['phi_forcing'][n, ...]) < max_phi/10)), field[n,...])

        divergence  = field1.filled(np.nan)
        convergence = field2.filled(np.nan)
        
        row  = np.where( np.abs(np.rad2deg((dicti['lats'])))<lat_lim)
        lats = np.squeeze( np.rad2deg((dicti['lats']))[row] )
        lons = np.squeeze( np.rad2deg((dicti['lons'])) )
        
        divergence  = np.squeeze(np.copy(divergence)[row, :])
        convergence = np.squeeze(np.copy(convergence)[row, :])
                          
        return convergence, divergence, lons, lats

    sum_forcing_minus,        sum_forcing_plus        = return_at_forcing(field  = eddy(dicti2['phi_forcing']), n=-350, PLOT = PLOT)
    sum_phi_minus,            sum_phi_plus            = return_at_forcing(field  = eddy(dicti2['PHI']), n=-350, PLOT = PLOT)    
    sum_dudx_minus,           sum_dudx_plus           = return_at_forcing(field = (div_x),          n=-350, PLOT = PLOT)
    sum_dvdy_minus,           sum_dvdy_plus           = return_at_forcing(field = (div_y),          n=-350, PLOT = PLOT)    
    sum_div_minus,            sum_div_plus            = return_at_forcing(field = (div),            n=-350, PLOT = PLOT)
    sum_meanVRT_minus,        sum_meanVRT_plus        = return_at_forcing(field = (f-dU_dy),        n=-350, PLOT = PLOT)
    sum_div_vrt_minus,        sum_div_vrt_plus        = return_at_forcing(field = (total_div_vrt),  n=-350, PLOT = PLOT)
    sum_div_vrt_linear_minus, sum_div_vrt_linear_plus = return_at_forcing(field = (linear_div_vrt), n=-350, PLOT = PLOT)  
    sum_vortcity_adv_minus,   sum_vortcity_adv_plus   = return_at_forcing(field = (total_vortcity_advection), n=-350, PLOT = PLOT)
    sum_gradient_vrt_minus,   sum_gradient_vrt_plus   = return_at_forcing(field = (gradient_vrt),   n=-350, PLOT = PLOT)
    sum_vorticity_flux_minus, sum_vorticity_flux_plus = return_at_forcing(field = (total_vortcity_flux),   n=-350, PLOT = PLOT)
    
    Umax  = np.nanmax(zmean(dicti2['U']))
    
    forcing_minus,        forcing_plus, lons, lats          = return_spread_out_data(field = eddy(dicti2['phi_forcing']), n=-350, lat_lim = lat_lim)
    phi_minus,            phi_plus,     lons, lats          = return_spread_out_data(field = eddy(dicti2['PHI']), n=-350, lat_lim = lat_lim)    
    dudx_minus,           dudx_plus,    lons, lats          = return_spread_out_data(field = (div_x),          n=-350, lat_lim = lat_lim)
    dvdy_minus,           dvdy_plus,    lons, lats          = return_spread_out_data(field = (div_y),          n=-350, lat_lim = lat_lim)    
    div_minus,            div_plus,     lons, lats          = return_spread_out_data(field = (div),            n=-350, lat_lim = lat_lim)
    meanVRT_minus,        meanVRT_plus, lons, lats          = return_spread_out_data(field = (f-dU_dy),        n=-350, lat_lim = lat_lim)
    div_vrt_minus,        div_vrt_plus, lons, lats          = return_spread_out_data(field = (total_div_vrt),  n=-350, lat_lim = lat_lim)
    div_vrt_linear_minus, div_vrt_linear_plus, lons, lats   = return_spread_out_data(field = (linear_div_vrt), n=-350, lat_lim = lat_lim)  
    vortcity_adv_minus,   vortcity_adv_plus,   lons, lats   = return_spread_out_data(field = (total_vortcity_advection), n=-350, lat_lim = lat_lim)
    gradient_vrt_minus,   gradient_vrt_plus,   lons, lats   = return_spread_out_data(field = (gradient_vrt),   n=-350, lat_lim = lat_lim)
    vorticity_flux_minus, vorticity_flux_plus, lons, lats   = return_spread_out_data(field = (total_vortcity_flux),   n=-350, lat_lim = lat_lim)
    
    
    at_forcing_dicti = \
    {'sum_phi_minus'    : sum_phi_minus,                   'sum_phi_plus'    : sum_phi_plus, \
     'sum_dudx_minus'   : sum_dudx_minus,                  'sum_dudx_plus'   : sum_dudx_plus, \
     'sum_dvdy_minus'   : sum_dvdy_minus,                  'sum_dvdy_plus'   : sum_dvdy_plus, \
     'sum_div_minus'    : sum_div_minus,                   'sum_div_plus'    : sum_div_plus, \
     'sum_meanVRT_minus': sum_meanVRT_minus,               'sum_meanVRT_plus': sum_meanVRT_plus, \
     'sum_div_vrt_minus': sum_div_vrt_minus,               'sum_div_vrt_plus': sum_div_vrt_plus, \
     'sum_div_vrt_linear_minus': sum_div_vrt_linear_minus, 'sum_div_vrt_linear_plus': sum_div_vrt_linear_plus, \
     'sum_vortcity_adv_minus'  : sum_vortcity_adv_minus,   'sum_vortcity_adv_plus'  : sum_vortcity_adv_plus, \
     'sum_gradient_vrt_minus':sum_gradient_vrt_minus,      'sum_gradient_vrt_plus': sum_gradient_vrt_plus,\
     'sum_vorticity_flux_minus': sum_vorticity_flux_minus, 'sum_vorticity_flux_plus': sum_vorticity_flux_plus,\
     'sum_forcing_minus': forcing_minus,                   'sum_forcing_plus':sum_forcing_plus,\
     'Umax': Umax, 'H0_value': H0}
    
    
    spread_out_data_at_forcing_dicti = \
    {'phi_minus'    : phi_minus,                   'phi_plus'    : phi_plus, \
     'dudx_minus'   : dudx_minus,                  'dudx_plus'   : dudx_plus, \
     'dvdy_minus'   : dvdy_minus,                  'dvdy_plus'   : dvdy_plus, \
     'div_minus'    : div_minus,                   'div_plus'    : div_plus, \
     'meanVRT_minus': meanVRT_minus,               'meanVRT_plus': meanVRT_plus, \
     'div_vrt_minus': div_vrt_minus,               'div_vrt_plus': div_vrt_plus, \
     'div_vrt_linear_minus': div_vrt_linear_minus, 'div_vrt_linear_plus': div_vrt_linear_plus, \
     'vortcity_adv_minus'  : vortcity_adv_minus,   'vortcity_adv_plus'  : vortcity_adv_plus, \
     'gradient_vrt_minus': gradient_vrt_minus,     'gradient_vrt_plus': gradient_vrt_plus,\
     'vorticity_flux_minus': vorticity_flux_minus, 'vorticity_flux_plus': vorticity_flux_plus,\
     'forcing_minus': forcing_minus,               'forcing_plus':forcing_plus,\
     'Umax': Umax, 'H0_value': H0, 'lat':lats, 'lons':lons}
        
    return at_forcing_dicti, spread_out_data_at_forcing_dicti
    

if __name__ == "__main__":
    
    constant_fluid_depth = '_constant_fluid_depth'
    
    Q0   = 10 #sys.argv[1] ## 10
    
    H0_exists = []

    for Hmean in [200]:  #, 200, 500, 800
        for H0 in tqdm([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000], desc=str(Hmean)): #0, 500, 1000, 1500, 2000, 2500, 3000, 3500
            for loc in [0]:
                source = '/data/pbarpanda/spherical_SWE/figure_out_budget/linear_Q_forcing/dt_150_Q_forcing_%d_forcing_y_%d_Hmean_%d_forcing_phase_speed_5_ms%s/dipole_heat_switch_on/H0_%d/'%(Q0, loc, Hmean, constant_fluid_depth, H0)
                if os.path.exists(source):
                    vars()['Q0_%d_loc_%d_H0_%d_Hmean_%d'%(Q0, loc, H0, Hmean)]          = h5saveload.load_dict_from_hdf5(source+'spatial_data.hdf5')
                    vars()['Q0_%d_loc_%d_H0_%d_Hmean_%d_equation'%(Q0, loc, H0, Hmean)] = h5saveload.load_dict_from_hdf5(source+'equation_data.hdf5')
                    H0_exists.append(H0)
                else:
                    print ('Q0_%d_loc_%d_H0_%d_Hmean_%d  does not exist yet'%(Q0, loc, H0, Hmean))
                        
                        
        vars()['Hmean_%d_Q0_%d_sum'%(Hmean, Q0)] = {}
        vars()['Hmean_%d_Q0_%d_spread_out'%(Hmean, Q0)] = {}


        for H0 in tqdm(H0_exists): #  should be in the range 0 to 5000 at increments of 500
            at_forcing_dicti, spread_out_data_at_forcing_dicti = return_data_at_forcing_region(Hmean = Hmean, Q0 = Q0, H0 = H0, PLOT=False, lat_lim=30)
            vars()['Hmean_%d_Q0_%d_sum'%(Hmean, Q0)][str(H0).zfill(4)]        = at_forcing_dicti
            vars()['Hmean_%d_Q0_%d_spread_out'%(Hmean, Q0)][str(H0).zfill(4)] = spread_out_data_at_forcing_dicti

        path = '/data/pbarpanda/spherical_SWE/evaluate_final_budget/linear_Q_forcing/combined_data_at_forcing_region%s/'%(constant_fluid_depth)
        h5saveload.make_sure_path_exists(path)
        h5saveload.save_dict_to_hdf5( vars()['Hmean_%d_Q0_%d_sum'%(Hmean, Q0)],        path + 'Hmean_%d_Q0_%d_sum.hdf5'%(Hmean, Q0) )
        h5saveload.save_dict_to_hdf5( vars()['Hmean_%d_Q0_%d_spread_out'%(Hmean, Q0)], path + 'Hmean_%d_Q0_%d_spread_out.hdf5'%(Hmean, Q0) )
        
