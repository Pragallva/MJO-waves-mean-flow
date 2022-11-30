import numpy as np
import pylab as py
import matplotlib
from tqdm import tqdm
import sys

import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
from tqdm import tqdm
import glob
from PIL import Image
import time as ti
import numpy.ma as ma

from PIL import Image
import imageio
from IPython.display import Video

sys.path.append('/data/pbarpanda/python_scripts/modules/')
import save_and_load_hdf5_files as h5saveload

import matplotlib as mpl
from cycler import cycler

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
            
                       
def make_movie(movie_name, dir_name, loop=0, movie=True, gif= True, fps=10, \
               duration = 300, interval=2, delete_images=True, extension='png', debug=False ):
    
        ####### directory name of the images #####################
        ####### movie name is Name of final file #####################
        ####### Final data will be saved in dirname/animate/ ##############
        ####### Ultimately delete all images in directory name ##############
    
        image_dest = dir_name ##'./Figures_different_fields_vrt_forcing_subplot/%s/'%(gif_name)
        gif_dest   = dir_name + '/animate/'
        fp_in      = image_dest+"/*.%s"%(extension)        
        fp_out     = gif_dest+movie_name+".gif"
        movie_out  = gif_dest+movie_name+".mp4"

        h5saveload.make_sure_path_exists(os.path.dirname(fp_out))
        
        if gif:
            img, *imgs = [Image.open(f) for f in list(np.sort(glob.glob(fp_in)))]
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                     save_all=True, duration=duration, loop=loop)
        
        if movie :
            images    = []
            filenames = list(np.sort(glob.glob(fp_in)))
            for filename in tqdm(filenames):
                if debug:
                    print (filename, end=', ')
                images.append(imageio.imread(filename))
            imageio.mimsave(movie_out, images, fps=fps)                

        if delete_images:
            remove_files(fp_in)
        
    
# make_movie(gif_name, dir_name, save_gif=SAVE, loop=0, movie=True, fps=10,)


