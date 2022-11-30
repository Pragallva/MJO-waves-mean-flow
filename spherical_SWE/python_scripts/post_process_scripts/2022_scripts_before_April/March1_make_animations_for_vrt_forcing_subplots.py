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


def return_response_at_tropics(dicti, key='div', n = 0, lonn=None, latt=None, tropical_lim=10):    
        
    tropical_field = eddy(dicti[key]) 
    
    div_field = eddy(dicti['div'])
    tropical_div = ma.masked_where( (latt < -tropical_lim) | (latt > tropical_lim), div_field[n,...]   ) 
                                                                      
    tropical_field_plus  = ma.masked_where( tropical_div < 0 , tropical_field[n,...])
    tropical_field_minus = ma.masked_where( tropical_div > 0 , tropical_field[n,...])
                                                                          
    divergence  = tropical_field_plus .filled(np.nan)
    convergence = tropical_field_minus.filled(np.nan)

    sum_conv    = np.nansum(convergence)
    sum_div     = np.nansum( divergence)
                                   
    return sum_conv, sum_div, convergence, divergence


def plot_vectors(gif_name, dicti, dicti2, force_dict, tropical_lim=10, \
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
    
    SUM_DIV = []; SUM_CONV = []; time_so_far_ax5=[]
    for t in tqdm(all_time, desc = SUPTITLE): #tqdm
        
        FILENAME = './Figures_different_fields_vrt_forcing_subplot/%s/%s.png'%(gif_name, str(t).zfill(7))
        if not os.path.exists(FILENAME):
                    
            fig = py.figure(figsize = FIGSIZE)


            ax2 = fig.add_subplot(121)
            
            if eddy_lrange_style != 'Fixed':
                maxi   =  np.nanmax ([ np.abs(np.nanmin(eddy(dicti[KEY])[t])), np.abs(np.nanmax(eddy(dicti[KEY])[t])) ])
                mini   = -maxi
                NN     = 20
                eddy_lrange =  np.linspace(mini, maxi, NN)


            lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )        

            field = ma.masked_where(((  np.abs(latt) < lat_bnd)  | (np.abs(latt) > lat_bnd1)), (dicti[KEY])[t, ...])

            cc = ax2.contourf(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        eddy(field), eddy_lrange, cmap=cmap, alpha=0.3); 
    
            cc = ax2.contour(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        eddy(field), eddy_lrange[::4][eddy_lrange[::4]>0], colors='maroon', linewidths=2); 
            ax2.clabel(cc, fmt =  '%1.0e', fontsize=23)
            
            cc = ax2.contour(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        eddy(field), eddy_lrange[::4][eddy_lrange[::4]<0], colors='navy', linewidths=2, linestyles='solid'); 
            ax2.clabel(cc, fmt =  '%1.0e', fontsize=23)



            max_phi = np.max(np.abs(eddy(force_dict['vrt_forcing'][t, ...])))
            ax2.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   eddy(force_dict['vrt_forcing'][t, ...]), \
                       [-(max_phi/2), (max_phi/2),],    colors='orange', linewidths=10, linestyles='solid'); 
            ax2.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   eddy(force_dict['vrt_forcing'][t, ...]), \
                       [-(max_phi/2), (max_phi/2),],    colors='white', linewidths=3); 


            ################### Inset figure #############################
            left, bottom, width, height = [0.8, 1.1, 0.2, 0.2]
            ax3 = ax2.inset_axes([left, bottom, width, height])
            F   = np.abs(np.copy(force_dict['vrt_forcing']))   
            time_so_far = days[all_time[0]:t]
            MAX = F.max()
            ff2 = F[all_time[0]:t, ...]
            
            try:
                ff  = [((ff2[i,...])[ff2[i,...] > 0]).max()/MAX for i in range(len(time_so_far))] 
                ax3.plot(time_so_far, ff, '-', color='teal', lw=2)
                ax3.set_title('vorticity forcing scale', color='teal') 
                
            except ValueError:
                pass

            ax3.set_xlim(days[all_time[0]]-10, days[all_time[-1]]+10)
            ax3.set_ylim(0, 1+0.3)   
            ax3.set_xticks( [days[all_time[0]], days[all_time[-1]]] )
            ax3.set_yticks( [0,1] )
            ax3.spines['bottom'].set_color('k')
            ax3.spines['top'].set_color('white')
            ax3.spines['left'].set_color('k')
            ax3.spines['right'].set_color('white')
            ax3.tick_params(length=0)
            ax3.set_xlabel('days', labelpad=1)
            ###############################################################
            
                    
                        
            
            ################### Inset figure #############################
            left, bottom, width, height = [0.0, 1.1, 0.2, 0.2]
            ax4 = ax2.inset_axes([left, bottom, width, height])
            
            u_mean  = dicti['U'][t, ...].mean(axis=-1)
            lat     = np.rad2deg(dicti['lats'])
            cos_phi = np.cos(dicti['lats'])
            umax    = dicti['U'].mean(axis=-1)
            try:
                ax4.plot(lat,  u_mean*cos_phi, '-', color='magenta', lw=2)
                ax4.set_title('Zonal mean zonal U', color='magenta') 
                ax4.set_ylabel('U [m/s]', labelpad=1)
                ax4.set_xlabel('lat', labelpad=1)
                ax4.spines['bottom'].set_color('k')
                ax4.spines['top'].set_color('white')
                ax4.spines['left'].set_color('k')
                ax4.spines['right'].set_color('white')
                ax4.tick_params(length=0)
            except ValueError:
                pass

            ax4.set_xlim(-90, 90)
            ax4.set_ylim(0, np.max(umax))   
            ax4.set_xticks( [-90, 90] )
            ax4.set_yticks( [0, int(np.max(umax))] )

            ###############################################################

            
            latt = np.rad2deg(dicti['lats'])
            Umean = (dicti['U'])[t].mean(axis=-1)

            ax2.set_title(variable+'\n Day %d'%(dicti['T_in_days'][t]), fontsize=20, y=1.05)

            uu  = eddy(dicti['U'])[t][::l2,::l1]
            vv  = eddy(dicti['V'])[t][::l2,::l1]
            hyp = (np.sqrt(uu**2 + vv**2))

#             if mag is None:
            mag = (np.max( np.abs(uu) ))

            Q = py.quiver(np.rad2deg(dicti['lons'][::l1]), np.rad2deg(dicti['lats'][::l2]), uu, vv, \
                      angles='uv',  color='gray') #scale=arrow_vector_size,
            ax2.quiverkey(Q, 0.3, 0.9, mag, r'%1.3e $m/s$'%(mag), labelpos='E', coordinates='figure', fontproperties = {'size':18})

            xx, yy = np.meshgrid(dicti['lons'], dicti['lats'])  
            py.tick_params(axis='both', labelsize=18)
            ax2.set_xlabel('longitude', fontsize=20)
            ax2.set_ylabel('latitude', fontsize=20)
            ax2.set_ylim(-lat_bnd1, lat_bnd1)
            ax2.axhline(-tropical_lim, color='teal', linewidth=1)
            ax2.axhline( tropical_lim, color='teal', linewidth=1)
            
            
            
            
            
            ax8 = fig.add_subplot(122)
            
            R= 6371e3
            
            if eddy_lrange_style != 'Fixed':
                maxi   =  np.nanmax ([ np.abs(np.nanmin(eddy(dicti['div'])[t])), np.abs(np.nanmax(eddy(dicti['div'])[t])) ])
                mini   = -maxi
                NN     = 20
                eddy_lrange =  np.linspace(mini, maxi, NN)*R


            lonn, latt = np.meshgrid( np.rad2deg((dicti['lons'])), np.rad2deg((dicti['lats'])) )        

            field = ma.masked_where(((  np.abs(latt) < lat_bnd)  | (np.abs(latt) > lat_bnd1)), R*(dicti['div'])[t, ...])

            cc = ax8.contourf(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        eddy(field), eddy_lrange, cmap=cmap, alpha=0.3); 
    
            cc = ax8.contour(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        eddy(field), eddy_lrange[::4][eddy_lrange[::4]>0], colors='maroon', linewidths=2); 
            ax8.clabel(cc, fmt =  '%1.1e', fontsize=23)
            
            cc = ax8.contour(np.rad2deg(dicti['lons']), np.rad2deg(dicti['lats']), \
                        eddy(field), eddy_lrange[::4][eddy_lrange[::4]<0], colors='navy', linewidths=2, linestyles='solid'); 
            ax8.clabel(cc, fmt =  '%1.1e', fontsize=23)
            xx, yy = np.meshgrid(dicti['lons'], dicti['lats'])  
            py.tick_params(axis='both', labelsize=18)
            ax8.set_xlabel('longitude', fontsize=20)
            ax8.set_ylabel('latitude', fontsize=20)
            ax8.set_ylim(-lat_bnd1, lat_bnd1)
            ax8.axhline(-tropical_lim, color='teal', linewidth=1)
            ax8.axhline( tropical_lim, color='teal', linewidth=1)
            ax8.set_title('R x Divergence [m/s, colors]'+'\n Day %d'%(dicti['T_in_days'][t]), fontsize=20, y=1.05)
            
            
            
            
            ################### Inset figure #############################
            left, bottom, width, height = [0.8, 1.1, 0.2, 0.2]
            
            R   = 6371e3
            
            sum_div_minus,  sum_div_plus, convergence_field, divergence_field = return_response_at_tropics( dicti, 'div',  n = t, lonn=lonn, latt=latt, tropical_lim=tropical_lim)
                        
            SUM_CONV.append(sum_div_minus); 
            SUM_DIV.append(sum_div_plus);
            time_so_far_ax5.append(days[t])
            max_div = np.max(np.array(SUM_DIV)*R)
                                   
#             max_divergence_field = np.nanmax((divergence_field))
#             ax2.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   divergence_field, \
#                        [ (max_divergence_field/2), ],    colors='red', linewidths=6, linestyles='solid'); 
#             ax2.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   divergence_field, \
#                        [(max_divergence_field/2), ],   colors='white', linewidths=2, linestyles='solid'); 
                                   
                                   
#             max_convergence_field = np.nanmin((convergence_field))
#             ax2.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   convergence_field, \
#                        [(max_convergence_field/2)],    colors='red', linewidths=6, linestyles='solid'); 
#             ax2.contour(np.rad2deg((dicti['lons'])),  np.rad2deg((dicti['lats'])),   convergence_field, \
#                        [(max_convergence_field/2)],    colors='white', linewidths=2, linestyles='dashed'); 
             
            try:
                ax5 = ax8.inset_axes([left, bottom, width, height])
                ax5.plot( np.array(time_so_far_ax5),   np.array(SUM_DIV)*R, '.',  color='dodgerblue')
    #             ax5.plot( np.array(time_so_far_ax5), -np.array(SUM_CONV)*R, '--', color='dodgerblue', lw=2)

                ax5.set_xlim(days[all_time[0]]-10, days[all_time[-1]]+10)
                ax5.set_title(r'$ R \mathbf{\nabla} . \mathbf{u^{*}} $ in the tropics', color='blue')
                ax5.set_ylim(0, max_div+5)   
                ax5.set_xticks( [days[all_time[0]], days[all_time[-1]]] )
                ax5.set_yticks( [0, int(max_div)] )
                ax5.spines['bottom'].set_color('k')
                ax5.spines['top'].set_color('white')
                ax5.spines['left'].set_color('k')
                ax5.spines['right'].set_color('white')
                ax5.tick_params(length=0)
                ax5.set_xlabel('days', labelpad=1)
                ax5.set_ylabel('[m/s]', labelpad=1)
                
                                
            except ValueError:
                pass

            ###############################################################
            
            
            ################### Inset figure #############################
            left, bottom, width, height = [0.0, 1.1, 0.2, 0.2]
            ax9 = ax8.inset_axes([left, bottom, width, height])
            
            u_mean  = dicti['U'][t, ...].mean(axis=-1)
            lat     = np.rad2deg(dicti['lats'])
            cos_phi = np.cos(dicti['lats'])
            umax    = dicti['U'].mean(axis=-1)
            try:
                ax9.plot(lat,  u_mean*cos_phi, '-', color='magenta', lw=2)
                ax9.set_title('Zonal mean zonal U', color='magenta') 
                ax9.set_ylabel('U [m/s]', labelpad=1)
                ax9.set_xlabel('lat', labelpad=1)
                ax9.spines['bottom'].set_color('k')
                ax9.spines['top'].set_color('white')
                ax9.spines['left'].set_color('k')
                ax9.spines['right'].set_color('white')
                ax9.tick_params(length=0)
            except ValueError:
                pass

            ax9.set_xlim(-90, 90)
            ax9.set_ylim(0, np.max(umax))   
            ax9.set_xticks( [-90, 90] )
            ax9.set_yticks( [0, int(np.max(umax))] )

            ###############################################################




#             py.suptitle(SUPTITLE, fontsize=20, y=y)

            FILENAME = './Figures_different_fields_vrt_forcing_subplot/%s/%s.png'%(gif_name, str(t).zfill(7))
            h5saveload.make_sure_path_exists(os.path.dirname(FILENAME))

            if save_gif:
                py.savefig(FILENAME, bbox_inches='tight', dpi=300)    
                py.close()
                
        else:
            pass

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
            for filename in filenames:
                images.append(imageio.imread(filename))
            imageio.mimsave(movie_out, images, fps=fps)                

        remove_files(fp_in)
        

if __name__ == "__main__":
    
    for extra in ['vortcity_forcing_midlat_loc_40_transient_Umax'] :#'alpha_5_switch_on_45_day', 'alpha_20_switch_on_80_day']:U_up_days_170 'U_up_days_237_mu_32_sigma_10', 'U_up_days_237_mu_55_sigma_10'
        for Hmean in ([500]):
            for H0 in ([5500]) : #3500 #3000, 1000, 0]): #0, 500, 1000, 1500, 2000, 2500, 3000, 3500 #0, 1000, 'alpha_20_switch_on_80_day'
                        
                    ps = 5;

                    source  = '/data/pbarpanda/spherical_SWE/evaluate_final_budget/momentum_forcing/Hmean_%d_forcing_phase_speed_%d_ms/%s/H0_%d/'%(Hmean, ps, extra, H0)

                    dicti_str = 'H0_%d_Hmean_%d_%s'%( H0, Hmean, extra)
                    dir_name  = './Figures_different_fields_vrt_forcing_subplot/%s/eddy_phi'%(dicti_str)

                    if not os.path.exists(dir_name):                        

                        if os.path.exists(source):
                            globals()[dicti_str] = h5saveload.load_dict_from_hdf5(source+'spatial_data.hdf5')


                            dicti_str = dicti_str
                            dicti     = globals()[dicti_str]                                
                            gif_name  = '/%s/eddy_phi'%(dicti_str)


                            globals()['force_dict'] =  globals()[dicti_str]


                            eddy_lrange     = 10
                            absolute_lrange = np.arange(-50, 55, 5)*1.5 #np.arange(-0.036, 0.038, 0.002)*200 #np.arange(-0.40,0.42, 0.02)
                            forcing_range   = 10 #np.linspace(-51,51,20)*200 #np.linspace(-601,601,12)

                            SUPTITLE     = '%s'%(dicti_str)
                            SAVE         = True

                            # ############# Animated the time evolution of winds 30 day ############## 


                            plot_vectors(gif_name, dicti = dicti, dicti2 = None, force_dict=force_dict, tropical_lim=15, \
                                         start_day=6, end_day=dicti['T_in_days'][-5], save_gif=SAVE, arrow_vector_size = 3.5,\
                                         eddy_lrange=eddy_lrange, eddy_lrange_style='Not Fixed', absolute_lrange=absolute_lrange,  \
                                         KEY='PHI', variable=r'Eddy $\phi$ [m] and wind [arrows]',\
                                         forcing_key = 'phi_forcing' , forcing_range = forcing_range, forcing_scale=1, \
                                         SUPTITLE=SUPTITLE, y=1.15, loop=0, movie=True, fps=10, forcing_contour=500, mag=None, \
                                         lat_bnd = 0, lat_bnd1 = 50, l1=14, l2=2, FIGSIZE=(36,10))

                        else:
                            print ('%s  does not exist yet'%(source))

                    else:
                        print (dir_name+' exists')


