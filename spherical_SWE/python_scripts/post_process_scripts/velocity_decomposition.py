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

            sf, vp, uD, vD, uR, vR, vrt, div = velocity_decomp.return_sf_vp_ur_ud(uwnd, vwnd, np.rad2deg(dicti['lats']), dim_order = 'tyx')
            fields = {'div':div*R, 'vrt':vrt*R, 'uD':uD, \
                      'vD':vD, 'uR': uR, 'vR': vR,\
                      'sf':sf*1e-6, 'vp':vp*1e-6, \
                      'lats':dicti['lats'], 'lons':dicti['lons'],\
                      'T_in_days': dicti['T_in_days']}

            h5saveload.save_dict_to_hdf5(fields, source+'velocity_decomp.hdf5')
