import numpy as np
import shtns
import pylab as py
import matplotlib
from tqdm import tqdm
import sys
from scipy.stats import norm

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
import shtns
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

import matplotlib as mpl
from cycler import cycler

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
    return X-X.mean(axis=-1)[...,None]

def eddy(X):
    return X-X.mean(axis=-1)[...,None]

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

def gauss_lat_curve(lat, mu=40, sigma=10):
    
    x1 = lat[0:len(lat)//2]
    x2 = lat[len(lat)//2 :]
    
    x = np.zeros(len(lat))
    y = np.zeros(len(lat))
    x[0:len(lat)//2] = x1
    x[len(lat)//2 :] = x2
       
    y1 = norm.pdf((x1 - mu)/sigma,0,1)
    y2 = norm.pdf((x2 + mu)/sigma,0,1)
    y[0:len(lat)//2] = y1
    y[len(lat)//2 :] = y2
    
    return y

def VRT_calc(uu, vv, lat, lon):
    R = 6371e3
    logging_object = logruns.default_log(logfilename   = 'momentum',  log_directory = './logs/')
    obj_momentum                  = momentum_advect.momentum_advection(uu, vv, lat, lon, logging_object)            
    dudy = obj_momentum.spher_div_y(uu, lat, lon, ax=-2)
    dvdx = obj_momentum.spher_div_x(vv, lat, lon, ax=-1)
    vrt =  (dvdx - dudy)
    f = 2*7.2921e-5*np.sin(np.deg2rad(lat))[:, None]
    return np.squeeze(vrt), f


def DIV_calc(uu, vv, lat, lon):
    R = 6371e3
    logging_object = logruns.default_log(logfilename   = 'momentum',  log_directory = './logs/')
    obj_momentum                  = momentum_advect.momentum_advection(uu, vv, lat, lon, logging_object) 

    dvdy = obj_momentum.spher_div_y(vv, lat, lon, ax=-2)
    dudx = obj_momentum.spher_div_x(uu, lat, lon, ax=-1)
    div  =  (dvdy + dudx)

    return np.squeeze(div)
         
            
if __name__ == "__main__": 
    
    mu    = int(sys.argv[1])
    sigma = int(sys.argv[2])
    
    T = -350
    
    ##### This is to analyze experiments for H0 = 500
    # for H0 in tqdm([500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000]):
    for Hmean in ([500]):
        for H0 in ([3000]):
            for loc in [0]:
                  for Q0 in ([10]):       #50, 500          
                    source = \
                    '/data/pbarpanda/spherical_SWE/evaluate_final_budget/linear_Q_forcing/dt_150_Q_forcing_%d_forcing_y_%d_Hmean_%d_forcing_phase_speed_5_ms/dipole_heat_switch_on/H0_%d/'%(\
                    Q0, loc, Hmean, H0)
                    
                    if os.path.exists(source):
                        vars()['Q0_%d_loc_%d_H0_%d_Hmean_%d'%(Q0, loc, H0, Hmean)] = h5saveload.load_dict_from_hdf5(source+'spatial_data.hdf5')
                        dicti = vars()['Q0_%d_loc_%d_H0_%d_Hmean_%d'%(Q0, loc, H0, Hmean)]
#                         vars()['Q0_%d_loc_%d_H0_%d_Hmean_%d_equation'%(Q0, loc, H0, Hmean)] = h5saveload.load_dict_from_hdf5(source+'equation_data.hdf5')
                        print ('reading over')
                    else:
                        print ('Q0_%d_loc_%d_H0_%d_Hmean_%d  does not exist yet'%(Q0, loc, H0, Hmean))

    
    
    uu = dicti['U'][T, ...]
    vv = dicti['V'][T, ...]
    
    R=6371e3

    lon, lat = np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats'])
    gauss = gauss_lat_curve(lat, mu=mu, sigma=sigma)[:,None]
    gauss = gauss/np.max(gauss)

    uu_gauss, vv_gauss   = eddy(uu*gauss),eddy(vv*gauss)
    vrt, f               = VRT_calc(uu_gauss, vv_gauss, lat, lon)
    div                  = DIV_calc(uu_gauss, vv_gauss, lat, lon)
        
    forcing_data = {'U':(uu_gauss), 'V':vv_gauss, 'vrt': vrt, 'div': div, 'lat': lat, 'lon':lon}
    
    py.figure()
    py.contourf(lon, lat, vrt, cmap=cmap); colorbar(20)
    py.quiver(lon[::10], lat[::2], uu_gauss[::2,::10], vv_gauss[::2,::10], color='gray')
    py.savefig('./forcing_data_mu_%d_sigma_%d.png'%(mu, sigma))
    py.close()
    
    h5saveload.save_dict_to_hdf5(forcing_data, './momentum_forcing_data_from_H0_3000_Hmean_500_mu_%d_sigma_%d.hdf5'%(mu, sigma))
    print ('done')
    
    
    
    
    
    
    


    
    
    
    

    
    
    
    
    
    
