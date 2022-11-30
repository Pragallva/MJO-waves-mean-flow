import netCDF4 as nc
import glob as glob
import pylab as py
import numpy as np
import hickle as hkl
import cartopy.crs as ccrs
import cartopy.util as cutil
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
import hickle as hkl
import sys
import logging
import time as ti
import scipy.signal as signal
from netCDF4 import Dataset,num2date,date2num
import datetime
from tqdm import tqdm
import numpy.ma as ma
from datetime import datetime, date

sys.path.append('/data/pbarpanda/python_scripts/modules/')

import logruns as logruns
import save_and_load_hdf5_files as h5saveload
import eulerian_fluxes as eflux
import netcdf_utilities as ncutil
# import imp
# imp.reload(ncutil)

import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

from windspharm.standard import VectorWind
from windspharm.tools import prep_data, get_recovery, recover_data, order_latdim


def return_sf_vp_ur_ud(uwnd, vwnd, lat, dim_order = 'tzyx'):
    
    masked = np.ma.getmask(np.ma.masked_where(np.isnan(uwnd), uwnd))  
    
    uwnd = np.nan_to_num(uwnd, nan=0.0)
    vwnd = np.nan_to_num(vwnd, nan=0.0)
    
    uwnd_new, uwnd_info = prep_data(uwnd, dim_order)
    vwnd_new, uwnd_info = prep_data(vwnd, dim_order)
    lats,   uwnd1,  vwnd1 = order_latdim(lat, uwnd_new, vwnd_new)
    
    w = VectorWind(uwnd1, vwnd1)
    
    sf, vp             =  w.sfvp()    
    u_D, v_D, u_R, v_R =  w.helmholtz()
    vort, div          =  w.vrtdiv()
    
    sf = recover_data(sf, uwnd_info)
    vp = recover_data(vp, uwnd_info)
    
    u_D = recover_data(u_D, uwnd_info)
    v_D = recover_data(v_D, uwnd_info)
    u_R = recover_data(u_R, uwnd_info)
    v_R = recover_data(v_R, uwnd_info)
    
    vort = recover_data(vort, uwnd_info)
    div  = recover_data(div,  uwnd_info)
    
    def mask(X):
        return np.ma.masked_where(masked, X) 
    
    if (lats[::-1] == lat).all():
        return mask(sf [...,::-1,:]),  mask(vp [...,::-1,:]),  mask(u_D [...,::-1,:]), mask(v_D[...,::-1,:]), \
               mask(u_R[...,::-1,:]),  mask(v_R[...,::-1,:]),  mask(vort[...,::-1,:]), mask(div[...,::-1,:])
    else:
        return mask(sf), mask(vp), mask(u_D), mask(v_D), mask(u_R), mask(v_R), mask(vort), mask(div)

 

def return_daily_field_sorted(dicti):
    
    new_dicti = {'phase1':0, 'phase2':0, 'phase3':0, 'phase4':0,\
                 'phase5':0, 'phase6':0, 'phase7':0, 'phase8':0,}
            
    for key in ['lat', 'lon', 'pres']:
        new_dicti[key]  = dicti[key]

    for p in range(1,9):

        key             = 'phase%d'%(p); key_date = 'phase%d_date'%(p)
        variable        = dicti[key]
        dates           = dicti[key_date]
        dates           = np.array([datetime.strptime(n.decode(),'%m.%d.%Y') for n in dates])
        sorted_indices  = list(dates.argsort())
        variable        = variable[sorted_indices, ...] 

        dates           = dates[sorted_indices, ...]             
        dates           = ['%s.%s.%s'%(str(d.month).zfill(2), str(d.day).zfill(2), str(d.year).zfill(4)) for d in dates]
        dates           = [d.encode('utf-8') for d in dates]

        dicti[key]      = variable
        dicti[key_date] = dates

        new_dicti[key]       = new_dicti[key]+dicti[key]
        new_dicti[key_date]  = dicti[key_date]
        
    return new_dicti    
           
    
def return_wind_fields_daily_field(source = '/data/pbarpanda/separate_mjo_phases_corrected_July30/all_phase/',\
                                   window = 'window_15', dest='daily', seas='DJF', filter_data='',  dim_order='etzyx', logging_object=None):
    
    if logging_object is None:
        logruns.default_log(logfilename='vel_decomp', log_directory='./logs/') 
        
    logging_object.write("Saving [ %s - %s - %s - %s]"%(filter_data, dest, seas, window))
    
    source = source+'/%s/%s/%s/%s/'%(window, dest, filter_data, seas)
#     if (not os.path.exists(source+'stream_function.hdf5')):
    
    vwnd_dict   = return_daily_field_sorted(h5saveload.load_dict_from_hdf5(source+'meridional_wind.hdf5'))  #### to make sure the dates are arranged in ascending order
    uwnd_dict   = return_daily_field_sorted(h5saveload.load_dict_from_hdf5(source+'zonal_wind.hdf5'))       #### to make sure the dates are arranged in ascending order

    print ('U and V dates sorted')
    
    stream_function    = {'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    velocity_potential = {'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    u_divergent        = {'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    v_divergent        = {'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    u_rotational       = {'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    v_rotational       = {'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    vorticity          = {'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    divergence         = {'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}

    for phase in tqdm(range(1,9), desc=filter_data):

        logging_object.write("Decomposing phase %d"%(phase))

        stream_function['phase%d_date'%(phase)]    = vwnd_dict['phase%d_date'%(phase)]
        velocity_potential['phase%d_date'%(phase)] = vwnd_dict['phase%d_date'%(phase)]
        u_divergent['phase%d_date'%(phase)]        = vwnd_dict['phase%d_date'%(phase)]
        v_divergent['phase%d_date'%(phase)]        = vwnd_dict['phase%d_date'%(phase)]
        u_rotational['phase%d_date'%(phase)]       = vwnd_dict['phase%d_date'%(phase)]
        v_rotational['phase%d_date'%(phase)]       = vwnd_dict['phase%d_date'%(phase)]
        vorticity['phase%d_date'%(phase)]          = vwnd_dict['phase%d_date'%(phase)]
        divergence['phase%d_date'%(phase)]         = vwnd_dict['phase%d_date'%(phase)]

        vwnd = np.array(vwnd_dict['phase%d'%(phase)])
        uwnd = np.array(uwnd_dict['phase%d'%(phase)])

        sf, vp, u_D, v_D, u_R, v_R, vort, div = return_sf_vp_ur_ud(uwnd, vwnd, vwnd_dict['lat'], dim_order = dim_order)

        stream_function['phase%d'%(phase)]    = np.array(sf)
        velocity_potential['phase%d'%(phase)] = np.array(vp)
        u_divergent['phase%d'%(phase)]        = np.array(u_D)
        v_divergent['phase%d'%(phase)]        = np.array(v_D)
        u_rotational['phase%d'%(phase)]       = np.array(u_R)
        v_rotational['phase%d'%(phase)]       = np.array(v_R)
        vorticity['phase%d'%(phase)]          = np.array(vort)
        divergence['phase%d'%(phase)]         = np.array(div)
        

    h5saveload.save_dict_to_hdf5(uwnd_dict,          source+'zonal_wind.hdf5')  
    h5saveload.save_dict_to_hdf5(vwnd_dict,          source+'meridional_wind.hdf5')  

    h5saveload.save_dict_to_hdf5(stream_function,    source+'stream_function.hdf5')  
    h5saveload.save_dict_to_hdf5(velocity_potential, source+'velocity_potential.hdf5')
    h5saveload.save_dict_to_hdf5(u_divergent,        source+'u_divergent.hdf5')
    h5saveload.save_dict_to_hdf5(u_rotational,       source+'u_rotational.hdf5')
    h5saveload.save_dict_to_hdf5(v_divergent,        source+'v_divergent.hdf5')
    h5saveload.save_dict_to_hdf5(v_rotational,       source+'v_rotational.hdf5')
    h5saveload.save_dict_to_hdf5(vorticity,          source+'vorticity.hdf5')
    h5saveload.save_dict_to_hdf5(divergence,         source+'divergence.hdf5')

    logging_object.write("Finished saving [ %s - %s - %s - %s]"%(filter_data, dest, seas, window))
    logging_object.write("**********************************************************************")

    # else:   

    #     logging_object.write("Don't worry! Already calculated [ %s - %s - %s - %s]"%(filter_data, dest, seas, window))
    #     logging_object.write("**********************************************************************")
            
    
    
    
def return_wind_fields_average_field(source = '/data/pbarpanda/separate_mjo_phases_corrected_July30/all_phase/',\
                       window = 'window_15', dest='phase_averaged_composite', seas='DJF', filter_data='', dim_order = 'ptzyx'):
    
    if logging_object is None:
        logruns.default_log(logfilename='vel_decomp', log_directory='./logs/')
    
    logging_object.write("Saving [ %s - %s - %s - %s]"%(filter_data, dest, seas, window))
    
    source = source+'/%s/%s/%s/%s/'%(window, dest, filter_data, seas)
#     if (not os.path.exists(source+'stream_function.hdf5')):
    
    vwnd_dict   = h5saveload.load_dict_from_hdf5(source+'meridional_wind.hdf5')
    uwnd_dict   = h5saveload.load_dict_from_hdf5(source+'zonal_wind.hdf5')      

    vwnd = vwnd_dict['meridional_wind']
    uwnd = uwnd_dict['zonal_wind']

    stream_function    = {'dates':vwnd_dict['dates'], 'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    velocity_potential = {'dates':vwnd_dict['dates'], 'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    u_divergent        = {'dates':vwnd_dict['dates'], 'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    v_divergent        = {'dates':vwnd_dict['dates'], 'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    u_rotational       = {'dates':vwnd_dict['dates'], 'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    v_rotational       = {'dates':vwnd_dict['dates'], 'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    vorticity          = {'dates':vwnd_dict['dates'], 'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}
    divergence         = {'dates':vwnd_dict['dates'], 'lat':vwnd_dict['lat'], 'lon':vwnd_dict['lon'], 'pres':vwnd_dict['pres']}

    logging_object.write(".........  decomposing now .............")
    sf, vp, u_D, v_D, u_R, v_R, vort, div = return_sf_vp_ur_ud(uwnd, vwnd, vwnd_dict['lat'], dim_order = dim_order)

    stream_function['stream_function']      = sf
    velocity_potential['velocity_potential']= vp
    u_divergent['u_divergent']              = u_D
    u_rotational['u_rotational']            = u_R
    v_divergent['v_divergent']              = v_D
    v_rotational['v_rotational']            = v_R
    vorticity['vorticity']                  = vort
    divergence['divergence']                = div

    h5saveload.save_dict_to_hdf5(stream_function,    source+'stream_function.hdf5')
    h5saveload.save_dict_to_hdf5(velocity_potential, source+'velocity_potential.hdf5')
    h5saveload.save_dict_to_hdf5(u_divergent,        source+'u_divergent.hdf5')
    h5saveload.save_dict_to_hdf5(u_rotational,       source+'u_rotational.hdf5')
    h5saveload.save_dict_to_hdf5(v_divergent,        source+'v_divergent.hdf5')
    h5saveload.save_dict_to_hdf5(v_rotational,       source+'v_rotational.hdf5')
    h5saveload.save_dict_to_hdf5(vorticity,          source+'vorticity.hdf5')
    h5saveload.save_dict_to_hdf5(divergence,         source+'divergence.hdf5')

    logging_object.write("Finished saving [ %s - %s - %s - %s]"%(filter_data, dest, seas, window))
    logging_object.write("**********************************************************************")     
            
#     else:        
        
#             logging_object.write("Don't worry! Already calculated [ %s - %s - %s - %s]"%(filter_data, dest, seas, window))
#             logging_object.write("**********************************************************************")
    
    
   
    
# if __name__ == "__main__":  
    
    
#     start_time  = ti.time()   
    
#     window_no   = int(sys.argv[1])
#     window      = 'window_%d'%(window_no)
#     seasons     = {'DJF':[12,1,2] } #, 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}    
    
    
#     if window == 'window_15' :
#         filtered_data_directories = ['']    
#     if window == 'window_101' :    
#         filtered_data_directories = ['',\
#                                      'filtered_data_0_to_10_day',\
#                                      'filtered_data_10_to_20_day',\
#                                      'filtered_data_20_to_96_day',\
#                                      'filtered_data_96_to_360_day',\
#                                      'filtered_data_360_to_higher_day']  
    
#     logging_object = logruns.default_log(logfilename=window+'_decomp', log_directory='./')  
        
#     logging_object.write("BEGINNING ....")   
    
#     for filter_data in filtered_data_directories:    
#         for seas in seasons.keys():
            
# #             return_wind_fields_average_field(source = '/data/pbarpanda/separate_mjo_phases_corrected_July30/all_phase/',\
# #                                              window = window, dest='phase_averaged_composite', \
# #                                              seas=seas, filter_data=filter_data, dim_order = 'ptzyx')

# #             return_wind_fields_average_field(source = '/data/pbarpanda/separate_mjo_phases_corrected_July30/all_phase/',\
# #                                              window = window, dest='phase_averaged_window_mean_composite', \
# #                                              seas=seas, filter_data=filter_data, dim_order = 'tzyx')

#             return_wind_fields_daily_field(source = '/data/pbarpanda/separate_mjo_phases_corrected_July30/all_phase/',\
#                                            window = window, dest='daily', seas=seas, filter_data=filter_data,  dim_order='etzyx', logging_object=logging_object)
        
#     end_time  = ti.time()
    
#     logging_object.write(' ========== CODE RAN SUCCESSFULLY, congrats! ================')
#     logging_object.write(' -----> Total Time taken = %1.3f  <----'%(end_time-start_time))
#     logging_object.write(' ============================================================')
    
    
    
    
    
    




