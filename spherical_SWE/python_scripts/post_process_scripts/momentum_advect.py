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
import velocity_decomposition_rotational_divergent as velocity_decomp

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


def locate(Y, x, gap=0.1):
    index = np.where(np.abs(Y-x) < gap)[0][0]
    return index

def locate(Y, x, gap=None):
    if gap is None:
        gap = 0.5*np.abs(np.diff(Y))[0] 
    index = np.where(np.abs(Y-x) <= gap)[0][0]
    return index
                     
            
logging_object = logruns.default_log(logfilename = 'vd', log_directory = './log/')

for H0 in tqdm([0, 20, 200, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]):
    
    logging_object.write('**********************************')
    logging_object.write('H0 = '+str(H0))
    
    for Hmean in [500]:
        
        logging_object.write('Hmean = '+str(Hmean))
        
        loc = 0
        source = '/data/pbarpanda/spherical_SWE/Hoskins_type_forcing/linear_Q_forcing/Q_forcing_500_forcing_y_%d_Hmean_%d_forcing_phase_speed_5_ms/dipole_heat_switch_on/H0_%d/'%(loc, Hmean, H0)

        if os.path.exists(source):
            dicti = h5saveload.load_dict_from_hdf5(source+'spatial_data.hdf5')

            uwnd = dicti['U']
            vwnd = dicti['V']
            R    = 6371e3
            D    = 46
            days = dicti['T_in_days']
            
            latt = np.rad2deg(dicti['lats'])
            lonn = np.rad2deg(dicti['lons'])
            
            linear_u = np.mean(uwnd[locate(days, D),:], axis=0, keepdims=True)
            linear_v = np.mean(vwnd[locate(days, D),:], axis=0, keepdims=True)
            
            eddy_u = uwnd - linear_u
            eddy_v = vwnd - linear_v          

            
            #### LINEAR 
                        
            obj_momentum                  = momentum_advect.momentum_advection(linear_u, linear_v, latt, lonn, logging_object)            
            fluxes, flux_variables, units = obj_momentum.zonal_advection_terms() 
            ubar_ubar_dx                   = fluxes['L01_T']
            vbar_ubar_dy                   = fluxes['L02_T']
            
                    
            #### LINEAR-EDDY
            obj_momentum                  = momentum_advect.momentum_advection(linear_u, eddy_v, latt, lonn, logging_object)            
            fluxes, flux_variables, units = obj_momentum.zonal_advection_terms() 
            uprime_ubar_dx                = eddy_u*obj_momentum.spher_div_x(linear_u, latt, lonn, ax=-1)      
            vprime_ubar_dy                = fluxes['L02_T']
            
            obj_momentum                  = momentum_advect.momentum_advection(eddy_u, linear_v, latt, lonn, logging_object)            
            fluxes, flux_variables, units = obj_momentum.zonal_advection_terms() 
            ubar_uprime_dx                = linear_u*obj_momentum.spher_div_x(eddy_u, latt, lonn, ax=-1)    
            vbar_uprime_dy                = fluxes['L02_T']
            
            
            #### EDDY-EDDY
            obj_momentum                  = momentum_advect.momentum_advection(eddy_u, eddy_v, latt, lonn, logging_object)            
            fluxes, flux_variables, units = obj_momentum.zonal_advection_terms() 
            uprime_uprime_dx              = fluxes['L01_T']
            vprime_uprime_dy              = fluxes['L02_T']
            
            linear_linear = {'ubar_ubar_dx': ubar_ubar_dx, 'vbar_ubar_dy': vbar_ubar_dy}
            
            linear_eddy   = {'uprime_ubar_dx': uprime_ubar_dx, 'vprime_ubar_dy': vprime_ubar_dy, \
                             'ubar_uprime_dx': ubar_uprime_dx, 'vbar_uprime_dy': vbar_uprime_dy}
            
            eddy_eddy     = {'uprime_uprime_dx': uprime_uprime_dx, 'vprime_uprime_dy': vprime_uprime_dy}
            
            zonal_momentum_adv = {'linear_linear': linear_linear, 'linear_eddy': linear_eddy, \
                                  'eddy_eddy': eddy_eddy, 'lats':dicti['lats'], 'lons':dicti['lons'], \
                                  'T_in_days':dicti['T_in_days'] }

            h5saveload.save_dict_to_hdf5(zonal_momentum_adv, source+'zonal_momentum_adv.hdf5')
