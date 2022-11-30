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

logging_object.write('**********************************')



master_source = '/data/pbarpanda/spherical_SWE/evaluate_final_budget/Nudge_remove_U/'
sources       = os.listdir(master_source)

for source in [master_source+s+'/' for s in sources]:
    source  = source + os.listdir(source)[0]+'/'
    source  = source + os.listdir(source)[0]+'/'
    
    if os.path.exists(source):
        
        if not os.path.exists(source+'velocity_decomp.hdf5'):
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
            
    else:
        print (source+'velocity_decomp.hdf5 \n file already exists')
