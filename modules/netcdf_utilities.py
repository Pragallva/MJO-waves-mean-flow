from netCDF4 import Dataset,num2date,date2num
import netCDF4 as nc
import glob as glob
import numpy as np
import datetime
from calendar import monthrange

month_strings = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_string  = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

def return_dates(YEARS             = np.arange(2019, 2022, 1), \
                 MONTHS            = np.arange(1, 13), \
                 all_days_in_month = True, DAYS=[1, 2] )  :
    dates = []
    for year in YEARS:
        for month in MONTHS:
            number_of_days = monthrange(year, month)[-1]
            if all_days_in_month:  
                DAYS = np.arange(1, number_of_days+1)                     
            for day in DAYS:
                try:
                    date = [int(year), int(month), int(day)]
                    dates.append(date)
                except ValueError:
                    pass
    return dates


def nc_dates_in_tuple(nctime):
    convert_time = (num2date(nctime[:], nctime.units, nctime.calendar)) 
    just_dates   = convert_time 
    dates_tuple=[]
    for date in just_dates:
        y,m,d = (date.year, date.month, date.day)
        dates_tuple.append([y,m,d])
    return dates_tuple


def common_date_indices(ncdates, smaller_subset_dates):
    day_index = []
    for i in range(len(ncdates)):
        if ncdates[i] in smaller_subset_dates:
            day_index.append(i)
    return day_index 


def correct_for_land_mask(field, p_sfc, pres, mask_value = 0):
    mask         = np.ones(field.shape)
    for i in range(len(pres)):
        time_lat_lon_index = np.where(p_sfc<pres[i]*100)
        mask[:,:,i,...][time_lat_lon_index] = mask_value
    return mask


def return_field(var = 'uwnd', mjo_dates = [[2008, 1, 7]], potential_temp=True):
    
    if var !='pres.sfc':
        field1 = []
        for mjo_date in mjo_dates:
            year, mo, day = mjo_date
            filename = '/Projects/era5_regrid/2p5/%s.2p5.%s%s*'%(var, year, month_string[mo-1])
            file         = glob.glob(filename)
            
            if  (len(file)) == 0:
                print (var, filename, file)
            ncfile       = file[0]
            v_var        = nc.Dataset(ncfile,'r')
            lat          = v_var['lat'][:]
            lon          = v_var['lon'][:]
            time         = v_var['time'][:]
            pres         = v_var['level'][::-1]
            netcdf_dates = nc_dates_in_tuple(nctime = v_var['time'])
            day_index    = common_date_indices(netcdf_dates, [mjo_date])
            if day_index:
                ucomp    = v_var.variables[var][day_index,::-1,:,:]
                field1.append(ucomp)
        field = np.squeeze(np.array(field1))
        if ((var == 'air') and (potential_temp)):
            P0     = pres[0]
            field  = field*(P0/pres[None,None,:,None,None])**0.286    
        return field, lat, lon, pres
        
    if var == 'pres.sfc':
        field1  = []
        for mjo_date in mjo_dates:
            year, mo, day = mjo_date
            filename = '/Projects/era5_regrid/2p5/%s.2p5.%s*'%(var,year)
            file          = glob.glob(filename)
            if  (len(file)) == 0:
                print (filename)
            ncfile        = file[0]
            v_var         = nc.Dataset(ncfile,'r')
            lat           = v_var['lat'][:]
            lon           = v_var['lon'][:]
            time          = v_var['time'][:]
            netcdf_dates  = nc_dates_in_tuple(nctime = v_var['time'])
            day_index     = common_date_indices(netcdf_dates, [mjo_date])
            if day_index:
                pres      = v_var.variables['pres'][day_index,:,:]
                field1.append(pres)
                
        field = np.squeeze(np.array(field1))
        return field, lat, lon
    
def sub_data(DATA1, lat, lon, EXTENTs = (120, -120), Hemi = 'N',):
    
        if Hemi == 'N':            
            lat_ind = np.squeeze(np.where( (lat>0) & (lat<91)  ))
        elif Hemi == 'S':            
            lat_ind = np.squeeze(np.where( (lat>-91) & (lat<0)  ))
        else:            
            lat_ind = np.squeeze(np.where( (lat>-91) & (lat<91)  ))
        DATA2 = DATA1[...,lat_ind,:]
        s=0;
        subdata_lon=[];
        sub_lon    =[];
        for EXTENT in EXTENTs:
            if (EXTENT[0] < 0) and (EXTENT[1] < 0):
                lon1 = EXTENT[0]+360; lon2 = EXTENT[1]+360
            else:
                lon1 = EXTENT[0]; lon2 = EXTENT[1]
            lon_ind = np.squeeze(np.where( (lon>lon1) & (lon<lon2)) )
            subdata_lon.append(DATA2[...,lon_ind])
            sub_lon.append(lon[lon_ind])
        return lat[lat_ind], np.concatenate(sub_lon,axis=-1), np.concatenate(subdata_lon, axis=-1)
    
    
    

    
    
