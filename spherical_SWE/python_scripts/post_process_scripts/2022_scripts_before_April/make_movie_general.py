import numpy as np
# import shtns
import pylab as py
import matplotlib
from tqdm import tqdm
import sys

sys.path.append('/data/pbarpanda/python_scripts/modules/')
import logruns as logruns
import save_and_load_hdf5_files as h5saveload
import movie_maker as movie_maker
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
from datetime import date
import sys

# Importing Image class from PIL module
from PIL import Image


# dicti_str = 'H0_%d_Hmean_%d_%s'%( H0, Hmean, extra)
dir_name  = '../../Figures/H0_5000_Hmean_500_damp_div_by_1_days_and_vrt_by_50_days_midlat_VRT_forcing/'

today = date.today()
movie_name =  today.strftime("%b-%d-%Y")+'eddy_phi_divergence'
# ############# Animated the time evolution of winds 30 day ############## 


movie_maker.make_movie(movie_name = movie_name, dir_name = dir_name, \
                       loop=0, movie=True, gif = False, fps=10, duration = 300, \
                       interval=2, delete_images=True, extension='png', debug=False )
