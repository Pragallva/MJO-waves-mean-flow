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


def plot_vectors(gif_name, dicti, dicti2, \
                 start_day=46, end_day=47, save_gif=False, arrow_vector_size=800,\
                 eddy_lrange=10, eddy_lrange_style='Fixed', absolute_lrange=10, variable='Absolute_phi', KEY='PHI',\
                 forcing_key = 'vrt_forcing' , forcing_range = [-1.8, -1.5, -1.2,  -1, -0.5, 0.5, 1, 1.2, 1.5, 1.8], \
                 SUPTITLE='', y=0, duration=300, loop=1, interval=2, forcing_scale=10**25, movie=True, fps=2, forcing_contour=20, mag=None,\
                 lat_bnd = 0, lat_bnd1 = 85, l1=8, l2=2, FIGSIZE=(30,6)):
        
        
    R = 6371e3 
    days = dicti['T_in_days']
    lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )
    
    all_time = [locate(days, (D)) for D in np.arange(start_day,end_day+0.125*interval)]
#     if not save_gif:
#         all_time = [all_time[-1]]
    
    for t in tqdm(all_time, desc = SUPTITLE): #tqdm
        
        FILENAME = './Figures_different_fields_no_fixed_range/%s/%s.png'%(gif_name, str(t).zfill(7))
        if not os.path.exists(FILENAME):
                    
            fig = py.figure(figsize = FIGSIZE)

            ax1 = fig.add_subplot(121)
            lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )  
            field = ma.masked_where( ((  np.abs(latt) < lat_bnd)  | (np.abs(latt) > lat_bnd1)), (dicti['U'])[t, ...])

            cc=ax1.contourf(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        (field), absolute_lrange, cmap=py.cm.RdBu_r, extend='both', alpha=0.7); # colorbar(20, ax1, cc)
            cc=ax1.contour(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        (field), absolute_lrange, colors='gray', extend='both', alpha=0.7); # colorbar(20, ax1, cc)
            py.clabel(cc, fmt='%1d', fontsize=15)
            ax1.set_title('Absolute zonal wind [m/s, colors] and Forcing [contours]'+'\n \n Day %d'%(dicti['T_in_days'][t]), fontsize=20)


            max_phi = np.max(np.abs(eddy(dicti['phi_forcing'][...])))        
            PHI_CONTOUR = np.linspace( -max_phi, max_phi, 16 )
            c = ax1.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   \
                           eddy(dicti['phi_forcing'][t, ...]), \
                           PHI_CONTOUR[PHI_CONTOUR>0],    cmap = py.cm.hot_r, alpha=0.8, linewidths=3, linestyles='solid'); 
            c = ax1.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   \
                           eddy(dicti['phi_forcing'][t, ...]), \
                           PHI_CONTOUR[PHI_CONTOUR<0],    cmap = py.cm.hot, alpha=0.8,  linewidths=3, linestyles='dashdot'); 

            py.tick_params(axis='both', labelsize=18)
            ax1.set_xlabel('longitude', fontsize=20)
            ax1.set_ylabel('latitude', fontsize=20)
            ax1.set_ylim(-lat_bnd1, lat_bnd1)

            if eddy_lrange_style != 'Fixed':
                maxi   =  np.nanmax ([ np.abs(np.nanmin(eddy(dicti[KEY])[t])), np.abs(np.nanmax(eddy(dicti[KEY])[t])) ])
                mini   = -maxi
                NN     = 20
                eddy_lrange =  np.linspace(mini, maxi, NN)

            ax2 = fig.add_subplot(122)


            lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )        

            field = ma.masked_where(((  np.abs(latt) < lat_bnd)  | (np.abs(latt) > lat_bnd1)), (dicti[KEY])[t, ...])
#             cc = ax2.contourf(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
#                         eddy(field), eddy_lrange, cmap=cmap, extend='both'); colorbar(20, ax2, cc)

            cc = ax2.contourf(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        eddy(field), eddy_lrange, cmap=cmap, alpha=0.3); 
    
            cc = ax2.contour(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        eddy(field), eddy_lrange[::4][eddy_lrange[::4]>0], colors='maroon', linewidths=2); 
            ax2.clabel(cc, fmt =  '%1.0e', fontsize=23)
            
            cc = ax2.contour(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        eddy(field), eddy_lrange[::4][eddy_lrange[::4]<0], colors='navy', linewidths=2, linestyles='solid'); 
            ax2.clabel(cc, fmt =  '%1.0e', fontsize=23)



            max_phi = np.max(np.abs(eddy(dicti['phi_forcing'][t, ...])))
            ax2.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   eddy(dicti['phi_forcing'][t, ...]), \
                       [-(max_phi/1.1), (max_phi/1.1),],    colors='orange', linewidths=10, linestyles='solid'); 
            ax2.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   eddy(dicti['phi_forcing'][t, ...]), \
                       [-(max_phi/1.1), (max_phi/1.1),],    colors='white', linewidths=3); 



            left, bottom, width, height = [0.75, 0.7, 0.2, 0.2]
            ax3 = ax2.inset_axes([left, bottom, width, height])
            F = np.copy(dicti['phi_forcing'])   
            time_so_far = days[all_time[0]:t]
            MAX = F.max()
            ff2 = F[all_time[0]:t, ...]
            
            try:
                ff  = [((ff2[i,...])[ff2[i,...] > 0]).max()/MAX for i in range(len(time_so_far))] 
                ax3.plot(time_so_far, ff, '-', color='lime', lw=4)
                ax3.set_title('forcing magnitude', color='lime') 
                
            except ValueError:
                pass

            ax3.set_xlim(days[all_time[0]], days[all_time[-1]])
            ax3.set_ylim(0, 1)   
            ax3.set_xticks( [days[all_time[0]], days[all_time[-1]]] )
            ax3.set_yticks( [0,1] )
            ax3.set_axis_off()

            latt = np.rad2deg(dicti['lats'])
            Umean = (dicti['U'])[t].mean(axis=-1)

            ax2.set_title(variable+'\n \n Day %d'%(dicti['T_in_days'][t]), fontsize=20)

            uu  = eddy(dicti['U'])[t][::l2,::l1]
            vv  = eddy(dicti['V'])[t][::l2,::l1]
            hyp = (np.sqrt(uu**2 + vv**2))

            if mag is None:
                mag = (np.max( np.abs(uu) ))/2

            Q = py.quiver(np.rad2deg(dicti['lons'][::l1]), np.rad2deg(dicti['lats'][::l2]), uu, vv, \
                      angles='uv',  color='gray') #scale=arrow_vector_size,
            ax2.quiverkey(Q, 0.6, 0.9, mag, r'%1.1e $m/s$'%(mag), labelpos='E', coordinates='figure', fontproperties = {'size':18})

            xx, yy = np.meshgrid(dicti['lons'], dicti['lats'])  
            py.tick_params(axis='both', labelsize=18)
            ax2.set_xlabel('longitude', fontsize=20)
            ax2.set_ylabel('latitude', fontsize=20)
            ax2.set_ylim(-lat_bnd1, lat_bnd1)


            py.suptitle(SUPTITLE, fontsize=20, y=y)

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
        img, *imgs = [Image.open(f) for f in list(np.sort(glob.glob(fp_in)))]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=duration, loop=loop)
        
        if movie :
            images    = []
            filenames = list(np.sort(glob.glob(fp_in)))
            for filename in filenames:
                images.append(imageio.imread(filename))
            imageio.mimsave(movie_out, images, fps=fps)                

        remove_files(fp_in)
        

if __name__ == "__main__":
    
    for extra in ['U_up_days_170'] :#'alpha_5_switch_on_45_day', 'alpha_20_switch_on_80_day']:
        for Hmean in ([500]):
            for H0 in ([3500]) : #3000, 1000, 0]): #0, 500, 1000, 1500, 2000, 2500, 3000, 3500 #0, 1000, 'alpha_20_switch_on_80_day'
                for loc in [0]:
                    for Q0 in ([10]):       #50, 500         #10, 100 
                         
                        source = '/data/pbarpanda/spherical_SWE/evaluate_final_budget/transient_U_fixed_forcing/dt_150_Q_forcing_%d_forcing_y_%d_Hmean_%d_forcing_phase_speed_0_ms/%s/H0_%d/'%(Q0, loc, Hmean, extra, H0)
                        #source = '/data/pbarpanda/spherical_SWE/evaluate_final_budget/transient_Q_forcing/dt_150_Q_forcing_%d_forcing_y_%d_Hmean_%d_forcing_phase_speed_0_ms/%s/H0_%d/'%(Q0, loc, Hmean, extra, H0)

                        dicti_str = 'Q0_%d_loc_%d_H0_%d_Hmean_%d_%s'%(Q0, loc, H0, Hmean, extra)
                        dir_name     = './Figures_different_fields_no_fixed_range/%s/eddy_phi'%(dicti_str)
                        
                        if not os.path.exists(dir_name):                        
                        
                            if os.path.exists(source):
                                globals()['Q0_%d_loc_%d_H0_%d_Hmean_%d_%s'%(Q0, loc, H0, Hmean, extra)] = h5saveload.load_dict_from_hdf5(source+'spatial_data.hdf5')
                                globals()['Q0_%d_loc_%d_H0_%d_Hmean_%d_%s_equation'%(Q0, loc, H0, Hmean, extra)] = h5saveload.load_dict_from_hdf5(source+'equation_data.hdf5')
                            

                                dicti_str = 'Q0_%d_loc_%d_H0_%d_Hmean_%d_%s'%(Q0, loc, H0, Hmean, extra)
                                dicti = globals()[dicti_str]                                
                                gif_name     = '/%s/eddy_phi'%(dicti_str)



                                dicti_str2 = 'Q0_%d_loc_%d_H0_%d_Hmean_%d_%s_equation'%(Q0, loc, H0, Hmean, extra)
                                dicti2 = globals()[dicti_str2]


                                eddy_lrange     = 10
                                absolute_lrange = np.arange(-50, 55, 5)*1.5 #np.arange(-0.036, 0.038, 0.002)*200 #np.arange(-0.40,0.42, 0.02)
                                forcing_range   = 10 #np.linspace(-51,51,20)*200 #np.linspace(-601,601,12)

                                SUPTITLE     = '%s'%(dicti_str)
                                SAVE         = True

                                # ############# Animated the time evolution of winds 30 day ############## 


                                plot_vectors(gif_name, dicti = dicti, dicti2 = dicti2,  \
                                             start_day=6, end_day=dicti['T_in_days'][-5], save_gif=SAVE, arrow_vector_size = 3.5,\
                                             eddy_lrange=eddy_lrange, eddy_lrange_style='Not Fixed', absolute_lrange=absolute_lrange,  \
                                             KEY='PHI', variable=r'Eddy $\phi$ [m] and wind [arrows]',\
                                             forcing_key = 'phi_forcing' , forcing_range = forcing_range, forcing_scale=1, \
                                             SUPTITLE=SUPTITLE, y=1.25, loop=0, movie=True, fps=10, forcing_contour=500, mag=None, \
                                             lat_bnd = 0, lat_bnd1 = 50, l1=14, l2=2, FIGSIZE=(30,8))
                            
                            else:
                                print ('Q0_%d_loc_%d_H0_%d_Hmean_%d  does not exist yet'%(Q0, loc, H0, Hmean))
                            
                        else:
                            print (dir_name+' exists')


