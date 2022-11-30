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

def perturb(lons, lats, Q0=100, yp=0, xp=90, Lx=30, Ly=10):
    xp, yp, Lx, Ly      = map(lambda z: np.deg2rad(z), [xp, yp, Lx, Ly ])
    phi_perturb         = Q0*np.exp(-   (((lats-yp)**2/Ly**2) +  ((lons-xp)**2/Lx**2))   )
    phi_mean            = np.mean(phi_perturb, axis=-1, keepdims=True)    
    phi_peturb_anom     = phi_perturb - phi_mean
    return phi_perturb,  phi_peturb_anom

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
      ["maroon", "red", "pink",  "white", "white", "skyblue", "dodgerblue", "navy"][::-1])


def return_at_forcing(dicti, force_dict, key='div', n = 0, lonn=None, latt=None):
    
        
    field = dicti[key]
    max_phi = np.max(np.abs(eddy(force_dict['phi_forcing'][n, ...])))

    field1  = ma.masked_where((   (   eddy(force_dict['phi_forcing'][n, ...]) < max_phi/10) | (latt<0)), field[n,...]) 
    field2  = ma.masked_where((   (  -eddy(force_dict['phi_forcing'][n, ...]) < max_phi/10) | (latt<0)), field[n,...])

    divergence  = field1.filled(np.nan)
    convergence = field2.filled(np.nan)

    sum_conv    = np.nansum(convergence)
    sum_div     = np.nansum( divergence)

    return sum_conv, sum_div


def plot_vectors(gif_name, dicti, dicti2, force_dict, \
                 start_day=46, end_day=47, save_gif=False, arrow_vector_size=800,\
                 eddy_lrange=10, eddy_lrange_style='Fixed', absolute_lrange=10, variable='Absolute_phi', KEY='PHI',\
                 forcing_key = 'vrt_forcing' , forcing_range = [-1.8, -1.5, -1.2,  -1, -0.5, 0.5, 1, 1.2, 1.5, 1.8], \
                 SUPTITLE='', y=0, duration=300, loop=1, interval=2, forcing_scale=10**25, movie=True, fps=2, forcing_contour=20, mag=None,\
                 lat_bnd = 0, lat_bnd1 = 85, l1=8, l2=2, FIGSIZE=(30,6)):
    
    
    py.rcParams.update( {   "text.usetex": False,
                            "font.family": "serif",
                            "font.serif": ["Palatino"] } )

        
    R = 6371e3 
    days = dicti['T_in_days']
    lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )
    
    all_time = [locate(days, (D)) for D in np.arange(start_day,end_day+0.125*interval)]
    
    for t in tqdm(all_time, desc = SUPTITLE): #tqdm
        
        FILENAME = './Figures_different_fields_no_fixed_range/%s/%s.png'%(gif_name, str(t).zfill(7))
        if not os.path.exists(FILENAME):
                    
                
            fig, axs = py.subplots(1, 2, figsize=FIGSIZE, sharey=True, sharex=True, constrained_layout=True)
            
            ax1      = axs[0]
            
            lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )
            slice1 = ((np.abs(latt) < lat_bnd)  | (np.abs(latt) > lat_bnd1))
            
            
            
            field = ma.masked_where( slice1, (dicti[KEY])[t, ...]) 
    
            cc = ax1.contourf(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                             (field), absolute_lrange, cmap=cmap, alpha=0.5, extend='both'); 
            
            cbar = fig.colorbar(cc, ax=ax1, location='right', aspect=30, shrink=1)
            for t1 in cbar.ax.get_yticklabels():
                t1.set_fontsize(20)
    
            xx, yy = np.meshgrid(dicti['lons'], dicti['lats'])  
            ax1.tick_params(axis='both', labelsize=20)
            ax1.set_xlabel('longitude', fontsize=20)
            ax1.set_ylabel('latitude', fontsize=20)
            ax1.set_ylim(-lat_bnd1, lat_bnd1)
            ax1.set_title('Zonal mean zonal U [m/s]',fontsize=20)
            

            ax2 = axs[1]
            
            if eddy_lrange_style != 'Fixed':
                maxi   =  np.nanmax ([ np.abs(np.nanmin(eddy(dicti['PHI'])[t])), np.abs(np.nanmax(eddy(dicti['PHI'])[t])) ])
                mini   = -maxi
                NN     = 20
                eddy_lrange =  np.linspace(mini, maxi, NN)
            else:
                maxi   =  int(np.nanmax ([ np.abs(np.nanmin(eddy(dicti['PHI']))), np.abs(np.nanmax(eddy(dicti['PHI']))) ]))
                mini   = -maxi
                NN     = 20
                eddy_lrange =  np.linspace(mini, maxi, NN)


            lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )
            slice1 = ((np.abs(latt) < lat_bnd)  | (np.abs(latt) > lat_bnd1))
            field = ma.masked_where( slice1, (dicti['PHI'])[t, ...]) 
            
            cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", 
                   ["darkred", "darkorange", "pink", "white", "white","skyblue", "dodgerblue", "navy"][::-1])
            
            cc = ax2.contourf(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                              eddy(field), eddy_lrange, cmap=cmap2, alpha=1, extend='both'); 
    
            cbar = fig.colorbar(cc, ax=ax2, location='right', aspect=30, shrink=1)
            for t1 in cbar.ax.get_yticklabels():
                t1.set_fontsize(20)
            
            latt = np.rad2deg(dicti['lats'])
            Umean = (dicti['U'])[t].mean(axis=-1)


            uu  = eddy(dicti['U'])[t][::l2,::l1]
            vv  = eddy(dicti['V'])[t][::l2,::l1]
            hyp = (np.sqrt(uu**2 + vv**2))

            if mag is None:
                mag = (np.max( np.abs(eddy(dicti['U']))))/2

            Q = py.quiver(np.rad2deg(dicti['lons'][::l1]), np.rad2deg(dicti['lats'][::l2]), uu, vv, \
                          angles='uv',  color='gray', scale=arrow_vector_size, minlength=0)
            ax2.quiverkey(Q, 0.6, 0.78, mag, r'%1.1e $m/s$'%(mag), labelpos='E', coordinates='figure', fontproperties = {'size':18})

            xx, yy = np.meshgrid(dicti['lons'], dicti['lats'])  
            ax2.tick_params(axis='both', labelsize=20)
            ax2.set_xlabel('longitude', fontsize=20)
#             ax2.set_ylabel('latitude', fontsize=20)
            ax2.set_ylim(-lat_bnd1, lat_bnd1)
            
            ax2.set_title(r'$\phi^*$ [colors, $m^2/s^2$], $u^*,v^*$ [vectors, m/s]',fontsize=22)

            fig.suptitle(SUPTITLE+'\n Day %d'%(dicti['T_in_days'][t]), fontsize=22)

            FILENAME = './Figures_different_fields_no_fixed_range/%s/%s.png'%(gif_name, str(t).zfill(7))
            h5saveload.make_sure_path_exists(os.path.dirname(FILENAME))

            if save_gif:
                py.savefig(FILENAME, bbox_inches='tight', dpi=300)    
                py.close()
                
        else:
            pass

    if save_gif:

        image_dest = './Figures_different_fields_no_fixed_range/%s/'%(gif_name)
        gif_dest   = './Figures_different_fields_no_fixed_range/animate/'
        fp_in  = image_dest+"/*.png"        
        fp_out = gif_dest+gif_name+".gif"
        movie_out = gif_dest+gif_name+".mp4"

        h5saveload.make_sure_path_exists(os.path.dirname(fp_out))
#       img, *imgs = [Image.open(f) for f in list(np.sort(glob.glob(fp_in)))]
#       img.save(fp=fp_out, format='GIF', append_images=imgs,
#                  save_all=True, duration=duration, loop=loop)
        
        if movie :
            images    = []
            filenames = list(np.sort(glob.glob(fp_in)))
            for filename in filenames:
                images.append(imageio.imread(filename))
            imageio.mimsave(movie_out, images, fps=fps)                

        remove_files(fp_in)
        

if __name__ == "__main__":
    
    
    Hmean = 500
    for typi in ['Rossby']:
        for H0 in [100, 1000, 2000, 3000, 4000] :
            
                    source  = '/data/pbarpanda/spherical_SWE/initial_value_exps/dt_150_Q_forcing_0_forcing_y_0_Hmean_%d_forcing_phase_speed_0_ms/U_up_days_150_%s/H0_%d/'%(Hmean, typi, H0)

                    dicti_str = '%s_H0_%d_Hmean_%d'%(typi, H0, Hmean)
                    dir_name  = './Figures_different_fields_no_fixed_range/animate/'                              
                    gif_name  = dicti_str+'/eddy_u_v_phi.mp4'

                    if not os.path.exists(dir_name+gif_name):                        

                        if os.path.exists(source):
                            globals()['%s_H0_%d_Hmean_%d'%(typi, H0, Hmean)] = h5saveload.load_dict_from_hdf5(source+'spatial_data.hdf5')

                            dicti_str = '%s_H0_%d_Hmean_%d'%(typi, H0, Hmean)
                            dicti     = globals()[dicti_str]                                
                            gif_name     = dicti_str+'/eddy_u_v_phi'

                            globals()['force_dict'] =  dicti 

                            eddy_lrange     = 10
                            absolute_lrange = np.arange(-50, 52, 2) #*1.5 #np.arange(-0.036, 0.038, 0.002)*200 #np.arange(-0.40,0.42, 0.02)
                            forcing_range   = 10 #np.linspace(-51,51,20)*200 #np.linspace(-601,601,12)

                            SUPTITLE     = 'Rossby wave initialise - (%s)'%(dicti_str)
                            SAVE         = True

                            # ############# Animated the time evolution of winds in day ############## 


                            plot_vectors(gif_name, dicti = dicti, dicti2 = None, force_dict=force_dict, \
                                         start_day=80, end_day=dicti['T_in_days'][-5], save_gif=SAVE, arrow_vector_size = 3.5,\
                                         eddy_lrange=eddy_lrange, eddy_lrange_style='Fixed', absolute_lrange=absolute_lrange,  \
                                         KEY='U', variable=r'Eddy $\phi$ [m] and wind [vectors]',\
                                         forcing_key = 'phi_forcing' , forcing_range = forcing_range, forcing_scale=1, \
                                         SUPTITLE=SUPTITLE, y=1.15, loop=0, movie=True, fps=10, forcing_contour=500, mag=None, \
                                         lat_bnd = 0, lat_bnd1 = 50, l1=14, l2=2, FIGSIZE=(20,7))

                        else:
                            print ('%s_H0_%d_Hmean_%d does not exist'%(typi, H0, Hmean))

                    else:
                        print (gif_name+' exists')


