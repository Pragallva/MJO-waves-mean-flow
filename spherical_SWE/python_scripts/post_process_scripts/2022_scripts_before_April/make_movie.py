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

def colorbar(fontsize=20, ax=None, plot_plot=None):
    if ax is None:
        cbar = py.colorbar()
    else:
        cbar = py.colorbar(plot_plot,ax=ax)
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
            
            
 

def make_movie(gif_name, save_gif=True, loop=0, movie=True, fps=10, duration = 300, interval=2 ):
        if save_gif:

            image_dest = './Figures_different_fields_vrt_forcing_subplot/%s/'%(gif_name)
            gif_dest   = './Figures_different_fields_vrt_forcing_subplot/animate/'
            fp_in  = image_dest+"/*.png"        
            fp_out = gif_dest+gif_name+".gif"
            movie_out = gif_dest+gif_name+".mp4"

            h5saveload.make_sure_path_exists(os.path.dirname(fp_out))
            img, *imgs = [Image.open(f) for f in list(np.sort(glob.glob(fp_in)))]
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                     save_all=True, duration=duration, loop=loop)

            if movie :
                images    = []
                filenames = list(np.sort(glob.glob(fp_in)))
                for filename in tqdm(filenames):
                    images.append(imageio.imread(filename))
                imageio.mimsave(movie_out, images, fps=fps)                

            remove_files(fp_in)
            
            
            
for extra in ['vortcity_forcing_midlat_loc_40_transient_Umax'] :
    for Hmean in ([500]):
        for H0 in ([5500]) : #3500 #3000, 1000, 0]): #0, 500, 1000, 1500, 2000, 2500, 3000, 3500 #0, 1000, 'alpha_20_switch_on_80_day'

                ps = 5;
                dicti_str = 'H0_%d_Hmean_%d_%s'%( H0, Hmean, extra)
                dir_name  = './Figures_different_fields_vrt_forcing_subplot/%s/eddy_phi'%(dicti_str)

                dicti_str = dicti_str

                gif_name  = '/%s/eddy_phi'%(dicti_str)


                eddy_lrange     = 10
                absolute_lrange = np.arange(-50, 55, 5)*1.5 #np.arange(-0.036, 0.038, 0.002)*200 #np.arange(-0.40,0.42, 0.02)
                forcing_range   = 10 #np.linspace(-51,51,20)*200 #np.linspace(-601,601,12)

                SAVE         = True

                # ############# Animated the time evolution of winds 30 day ############## 


                make_movie(gif_name, save_gif=SAVE, loop=0, movie=True, fps=10,)
