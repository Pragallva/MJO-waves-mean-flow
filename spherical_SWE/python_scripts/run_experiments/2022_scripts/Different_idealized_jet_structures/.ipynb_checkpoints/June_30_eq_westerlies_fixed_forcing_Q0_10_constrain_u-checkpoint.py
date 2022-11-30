import numpy as np
import shtns
import pylab as py
import matplotlib
from tqdm import tqdm
import sys
import time as ti
from scipy.stats import norm
import glob
import os

sys.path.append('/data/pbarpanda/python_scripts/modules/')
import logruns as logruns
import save_and_load_hdf5_files as h5saveload
import eulerian_fluxes as eflux
import netcdf_utilities as ncutil
from obspy.geodetics import kilometers2degrees
import momentum_advection_class as momentum_advect

def remove_files(direc):
    files = glob.glob(direc, recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
  

remove_files('./temp_June25/*/*/')
            
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'  ### This is because NOAA PSL lab computers are somehow not able to use 


class Spharmt(object):
    """
    wrapper class for commonly used spectral transform operations in
    atmospheric models.  Provides an interface to shtns compatible
    with pyspharm (pyspharm.googlecode.com).
    """
    def __init__(self, nlons, nlats, ntrunc, rsphere, gridtype="gaussian"):
        """initialize
        nlons:  number of longitudes
        nlats:  number of latitudes"""
        self._shtns = shtns.sht(ntrunc, ntrunc, 1,
                                shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)

        if gridtype == "gaussian":
            # self._shtns.set_grid(nlats, nlons,
            #         shtns.sht_gauss_fly | shtns.SHT_PHI_CONTIGUOUS, 1.e-10)
            self._shtns.set_grid(nlats, nlons,
                    shtns.sht_quick_init | shtns.SHT_PHI_CONTIGUOUS, 1.e-10)
        elif gridtype == "regular":
            self._shtns.set_grid(nlats, nlons,
                    shtns.sht_reg_dct | shtns.SHT_PHI_CONTIGUOUS, 1.e-10)

        self.lats = np.arcsin(self._shtns.cos_theta)
        self.lons = (2.*np.pi/nlons)*np.arange(nlons)
        self.nlons = nlons
        self.nlats = nlats
        self.ntrunc = ntrunc
        self.nlm = self._shtns.nlm
        self.degree = self._shtns.l
        self.m      = self._shtns.m
        self.lap    = -self.degree*(self.degree+1.0).astype(np.complex128)
        self.invlap = np.zeros(self.lap.shape, self.lap.dtype)
        self.invlap[1:] = 1./self.lap[1:]
        self.rsphere = rsphere
        self.lap     = self.lap/rsphere**2
        self.invlap  = self.invlap*rsphere**2

    def grdtospec(self, data):
        """compute spectral coefficients from gridded data"""
        return self._shtns.analys(data)

    def spectogrd(self, dataspec):
        """compute gridded data from spectral coefficients"""
        return self._shtns.synth(dataspec)

    def getuv(self, vrtspec, divspec):
        """compute wind vector from spectral coeffs of vorticity and divergence"""
        return self._shtns.synth((self.invlap/self.rsphere)*vrtspec, (self.invlap/self.rsphere)*divspec)

    def getvrtdivspec(self, u, v):
        """compute spectral coeffs of vorticity and divergence from wind vector"""
        vrtspec, divspec = self._shtns.analys(u, v)
        return self.lap*self.rsphere*vrtspec, self.lap*rsphere*divspec

    def getgrad(self, divspec):
        """compute gradient vector from spectral coeffs"""
        vrtspec = np.zeros(divspec.shape, dtype=np.complex128)
        u, v = self._shtns.synth(vrtspec, divspec)
        return u/rsphere, v/rsphere
    
      
import matplotlib.pyplot as plt
import time
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
      [  "darkred", "darkorange", "pink", "white", "white","skyblue", "dodgerblue", "navy"][::-1])

def phi_T(y, y0=0, H0=100,  N=2):
    ## y takes latitude in radians
    y0=np.deg2rad(y0)
    phi = grav*H0*(1-(np.sin(y)-np.sin(y0))**N)
    return phi

def phi_B(Hmean):
    phi = grav*Hmean
    return phi

def gauss_lat_curve(lat, mu=40, sigma=10):
    
    x1 = lat[0:len(lat)//2,:]
    x2 = lat[len(lat)//2 :,:]
    
    x = np.copy(lat)*0
    y = np.copy(lat)*0
    x[0:len(lat)//2,:] = x1
    x[len(lat)//2 :,:] = x2
       
    y1 = norm.pdf((x1 - mu)/sigma,0,1)
    y2 = norm.pdf((x2 + mu)/sigma,0,1)
    y[0:len(lat)//2,:] = y1
    y[len(lat)//2 :,:] = y2
    
    return y

# def extract_phiT_from_given_U(UU = None, lat = None, lon = None):
#     ########################################################################################################
#     def f_dU_dy(UU, lat, lon):
#         R = 6371e3
#         logging_object = logruns.default_log(logfilename   = 'momentum',  log_directory = './logs/')
#         obj_momentum                  = momentum_advect.momentum_advection(UU[None,...], (UU[None,...])*0, lat, lon, logging_object)            
#         dudy = obj_momentum.spher_div_y(UU[None,...], lat[:,0], lon[:,0], ax=-2)
#         vrt =  np.squeeze(- dudy)
#         f = 2*7.2921e-5*np.sin(np.deg2rad(lat))
#         return vrt + f 

#     cos_phi  = np.cos(np.deg2rad(lat))
#     sin_phi  = np.sin(np.deg2rad(lat))

#     if UU is None:
#         UU   = gauss_lat_curve(lat, mu=mu,  sigma=sigma)*100
#     vort_abs = f_dU_dy(UU, lat, lon)

#     dphiT_by_dy1 = (-(vort_abs*UU))
#     E           = (((UU)**2)/2)

# #     dE_by_dy    = (1/R)*np.gradient(E, np.deg2rad(lat))*0 ##### I have zeroed out the E vector for now
# #     dphiT_by_dy2 = (-(vort_abs*UU + dE_by_dy))
#     import scipy.integrate as integrate

#     def integrate_NH(lat, lon, dphiT_by_dy):

#         R = 6371e3
#         la=np.deg2rad(lat)
#         lo=np.deg2rad(lon)

#         x = dphiT_by_dy
#         phi_T1 = integrate.cumtrapz(x, la, axis=0, initial=0)*R
#         phi_T  = (phi_T1/np.max(phi_T1) - np.min(phi_T1/np.max(phi_T1))) 
#         return phi_T1
    
#     phiT  = integrate_NH(lat, lon, dphiT_by_dy1) + (Hmean*grav)
        
#     return phiT #*grav
# #################################################################################################################


def extract_phiT_from_given_U(UU = None, lat = None, lon = None): 
    R = 6371e3
    ########################################################################################################
    def f_dU_dy(UU, lat, lon):
        R = 6371e3
        logging_object = logruns.default_log(logfilename   = 'momentum',  log_directory = './logs/')
        obj_momentum                  = momentum_advect.momentum_advection(UU[None,...], (UU[None,...])*0, lat, lon, logging_object)            
        dudy = obj_momentum.spher_div_y(UU[None,...], lat[:,0], lon[0,:], ax=-2)
        vrt =  np.squeeze( -dudy )
        f = 2*7.2921e-5*np.sin(np.deg2rad(lat))
        return f 

    cos_phi  = np.cos(np.deg2rad(lat))
    sin_phi  = np.sin(np.deg2rad(lat))

    vort_abs = f_dU_dy(UU, lat, lon)
    dphiT_by_dy1 = (-(vort_abs*UU))
        
    import scipy.integrate as integrate   
    def integrate_NH(lat, lon, dphiT_by_dy):
        
        R = 6371e3
        la=np.deg2rad(lat)
        lo=np.deg2rad(lon)

        x = dphiT_by_dy
        phi_T1 = integrate.cumtrapz(x, la, axis=0, initial=0)*R
        return phi_T1

    phiT  = integrate_NH(lat, lon, dphiT_by_dy1)
    grav = 10  
    return phiT #*grav
##########################################################################################################



########################################################################################################
def extract_phiT_from_given_U_try_new2(UU = None, lat = None, lon = None):
    
    R = 6371e3

    cos_phi  = np.cos(np.deg2rad(lat))
    sin_phi  = np.sin(np.deg2rad(lat))
    tan_phi  = sin_phi/cos_phi

    f = 2*7.2921e-5*sin_phi
    
    dphiT_by_dy1 = -(f*UU + (UU**2)*tan_phi/R)
        
    import scipy.integrate as integrate   
    def integrate_NH(lat, lon, dphiT_by_dy):
        
        R = 6371e3
        la=np.deg2rad(lat)
        lo=np.deg2rad(lon)

        x = dphiT_by_dy
        phi_T1 = integrate.cumtrapz(x, la, axis=0, initial=0)*R
        return phi_T1

    phiT  = integrate_NH(lat, lon, dphiT_by_dy1) 
    return phiT
########################################################################################################


def PHI_perturb(lons, lats, Q0=100, yp=0, xp=90, Lx=30, Ly=10, time = 0, switch_on_day=None):
    
    if switch_on_day:
        switch_on_time = switch_on_day*24*3600
        if time >= switch_on_time:
            flag = 1
        else:
            flag = 0
    else:
        flag = 1
        
    xp, yp, Lx, Ly      = map(lambda z: np.deg2rad(z), [xp, yp, Lx, Ly ])
    phi_perturb         = flag * grav * Q0*np.exp(-   (((lats-yp)**2/Ly**2) +  ((lons-xp)**2/Lx**2))   )
    phi_mean            = np.mean(phi_perturb, axis=-1, keepdims=True)    
    phi_peturb_anom     = phi_perturb - phi_mean
    return phi_perturb, phi_peturb_anom


def PHI_perturb_propagating(lons, lats, Q0=100, yp=0, xp=90, Lx=30, Ly=10, time = 0, switch_on_day=None, c=5, wave_number=2, DIPOLE = True, switch_off_day=None):
    
    switch_on_time  = switch_on_day*24*3600
    switch_off_time = switch_off_day*24*3600
    
    flag = 1 if ((time >= switch_on_time) and (time <= switch_off_time))  else 0
    
    def heating_dipole_lon(shift, wave_number=2,):
        
        shift = shift%(2*np.pi)
        K     = wave_number
        dipole        = np.sin(K*(lons - shift))
        if (shift+2*np.pi/K) > (2*np.pi):
            end_point = shift
            beg_point = shift + (2*np.pi/K) - 2*np.pi
            zero_lon  = ((lons<end_point)) & (lons>beg_point)
        else:
            beg_point = shift
            end_point = shift + 2*np.pi/K
            zero_lon  = ((lons<beg_point)) | (lons>end_point)
            
        dipole[zero_lon] = 0
        
        return dipole
    
    
    def heating_monopole_lon(shift, wave_number=2,):
        
        shift = shift%(2*np.pi)
        K     = wave_number
        dipole        = np.sin(K*(lons - shift))
        if (shift+1*np.pi/K) > (2*np.pi):
            end_point = shift
            beg_point = shift + (1*np.pi/K) - 2*np.pi
            zero_lon  = ((lons<end_point)) & (lons>beg_point)
        else:
            beg_point = shift
            end_point = shift + 1*np.pi/K
            zero_lon  = ((lons<beg_point)) | (lons>end_point)
            
        dipole[zero_lon] = 0
        
        return dipole
        
             
    if c != 0:
        shift  = kilometers2degrees((c*(time-switch_on_time))/1000)
    else:
        shift  = 100  #### This will create a stationary forcing at 100 degree longitude
    
    xp, yp, Lx, Ly, shift      = map(lambda z: np.deg2rad(z), [xp, yp, Lx, Ly, shift ])
    if DIPOLE:
        heating_lon =   heating_dipole_lon  (shift, wave_number=wave_number)
    else:
        heating_lon =   heating_monopole_lon(shift, wave_number=wave_number)
            
    ################ TRANSIENT FORCING ###########################
    
    switch_on_time  = switch_on_day*24*3600
    switch_off_time = switch_off_day*24*3600  

    tt1 = (time - switch_on_time)/(24*3600)
    tt2 = (time - switch_off_time)/(24*3600)

    transient_effect1 = np.tanh(tt1/alpha)
    transient_effect2 = np.tanh(tt2/alpha)
    transient_effect  = (transient_effect1) ### - transient_effect2)/2

    ##############################################################
       
        
    phi_perturb         = grav * Q0*np.exp(  -(((lats-yp)**2/Ly**2)) ) * heating_lon * transient_effect
    phi_mean            = np.mean(phi_perturb, axis=-1, keepdims=True)    
    phi_peturb_anom     = phi_perturb - phi_mean
    return phi_perturb, phi_peturb_anom


def PHI_perturb_from_file(external_data, time = 0, switch_on_day=None,):
    if switch_on_day:
        switch_on_time  = switch_on_day*24*3600
        if time        >= switch_on_time:
            flag        = 1
        else:
            flag        = 0
    else:
        flag            = 1
            
    time_in_hours       = int(time//3600)
    time_in_hours       = time_in_hours%(48*24)    
    phi_perturb         = flag*external_data[time_in_hours]
    return phi_perturb[time], 
    
    
def F_uv_perturb(lons, lats, Fo=100, yp=0, xp=90, Lx=30, Ly=10, time = 0, switch_on_day=None):
    
    if switch_on_day:
        switch_on_time = switch_on_day*24*3600
        if time >= switch_on_time:
            flag = 1
        else:
            flag = 0
    else:
        flag = 1
        
    xp, yp, Lx, Ly  = map(lambda z: np.deg2rad(z), [xp, yp, Lx, Ly ])
    perturb         = flag * Fo * np.exp(-   (((lats-yp)**2/Ly**2) +  ((lons-xp)**2/Lx**2))   )
    perturb_mean    = np.nanmean(perturb, axis=-1, keepdims=True)    
    peturb_anom     = perturb - perturb_mean
    return perturb, peturb_anom

def A(Z):
    return np.array(Z)


def initial_conditions(sp_harmonic):
        
    vg   =               np.zeros((nlats, nlons), np.float64)
    ug   =               np.zeros((nlats, nlons), np.float64)
    phig =               np.zeros((nlats, nlons), np.float64)

    # initial vorticity, divergence in spectral space
    vrtspec, divspec = sp_harmonic.getvrtdivspec(ug, vg)
    vrtg = sp_harmonic.spectogrd(vrtspec)
    divg = sp_harmonic.spectogrd(divspec)

    # Nonlinear terms on the RHS
    u_f_plus_vort = ug*(vrtg+f)
    v_f_plus_vort = vg*(vrtg+f)

    curl_NL_spec, div_NL_spec = sp_harmonic.getvrtdivspec(u_f_plus_vort, v_f_plus_vort)
    KE_spec = sp_harmonic.grdtospec(0.5*(ug**2+vg**2))
    phispec = sp_harmonic.grdtospec(phig)


    ### Try this part later. Where you start from a balanced condition with a perturbation
    # balanced_phispec = sp_harmonic.invlap*curl_NL_spec - KE_spec
    # phig    = phi_B(Hmean)+ \
    #           perturb(lons, lats, Q0=100, yp=0, xp=90, Lx=30, Ly=10)[1] + \
    #           sp_harmonic.spectogrd(balanced_phispec)
    # phispec = x.grdtospec(phig)
    
    return ug, vg, phig, vrtg, divg, vrtspec, divspec, KE_spec, phispec



def integrate_model(input_file2):
    
    abort_status = 'False'
    
    U   = []; 
    V   = [];
    PHI = [];
    VRT = [];
    DIV = [];    
    PHI_forcing   = []
    F_vrt_forcing = []
    F_div_forcing = []
    PHI_T = []
    
    U_spec   = []; 
    V_spec   = [];
    PHI_spec = [];
    VRT_spec = [];
    DIV_spec = [];
    PHI_forcing_spec   = []
    F_vrt_forcing_spec = []
    F_div_forcing_spec = []
    
    T = [];
    
    VRT_term1  = [] ## vrt advection
    VRT_term1a = []
    VRT_term1b = []
    VRT_term2  = [] ## damping vrt
    
    DIV_term1  = [] ## div advection (curl of u(abs vrt))
    DIV_term1a = []
    DIV_term1b = []
    DIV_term2 = [] ## laplacian phi
    DIV_term3 = [] ## laplacian phiT
    DIV_term4 = [] ## laplacian KE
    DIV_term5 = [] ## damping div
    
    PHI_term1 = [] ## phi advection
    PHI_term1a = []
    PHI_term1b = []
    PHI_term2 = [] ## damping phi
    PHI_term3 = [] ## forcing phi
    
        
    for key in input_file.keys():
        globals()[key] = input_file2[key] 
    
    # setup up spherical harmonic instance, set lats/lons of grid
    sp_harmonic = Spharmt(nlons, nlats, ntrunc, rsphere, gridtype="gaussian")
    lons, lats  = np.meshgrid(sp_harmonic.lons, sp_harmonic.lats)
    global f
    f           = 2.*omega*np.sin(lats)   # coriolis
    
        
    ug, vg, phig, vrtg, divg, vrtspec, divspec, KE_spec, phispec = initial_conditions(sp_harmonic)

    ddivdtspec = np.zeros(vrtspec.shape+(3,), np.complex128)
    dvrtdtspec = np.zeros(vrtspec.shape+(3,), np.complex128)
    dphidtspec = np.zeros(vrtspec.shape+(3,), np.complex128)
    nnew = 0
    nnow = 1
    nold = 2
    
    
    H0_values     = np.zeros(itmax)
        
    Q_spinup_time = int(alpha*3)*int(86400/dt)    
    tmax_25       = (U_up_days)*int(86400/dt)
    tmin_25       = (U_up_days*2 + alpha*3)*int(86400/dt) - tmax_25 - Q_spinup_time
    rest_time     = itmax - tmin_25 - tmax_25 - Q_spinup_time
    
    H0_values[ : Q_spinup_time]                                                       = 0
    H0_values[Q_spinup_time : tmax_25+Q_spinup_time]                                  = np.linspace(0, Hmax, tmax_25)
    H0_values[tmax_25 + Q_spinup_time : tmin_25 + tmax_25 + Q_spinup_time   ]         = np.linspace(Hmax, 0, tmin_25)
    H0_values[tmin_25 + tmax_25 + Q_spinup_time : ]                                   = 0
    
    #################################### create a phiT based on the structure of U mean ########################################
    
    den = 1
            
    for ncycle in tqdm(range(itmax)):
        
        t = ncycle*dt
        
        if int(t/(24*3600)) > 5 :
            
            if t % (3*3600) == 0: ### Save every 3 hours
                U.  append(ug)
                V.  append(vg)
                PHI.append(phig)
                VRT.append(vrtg)
                DIV.append(divg)
                
                U_spec.  append(sp_harmonic.grdtospec(ug))
                V_spec.  append(sp_harmonic.grdtospec(vg))
                PHI_spec.append(phispec)
                VRT_spec.append(vrtspec)
                DIV_spec.append(divspec)
                
                T.append(t/(24*3600)) 
        
        # get vort, u, v, phi on grid
        divg   = sp_harmonic.spectogrd(divspec)
        vrtg   = sp_harmonic.spectogrd(vrtspec)
        ug, vg = sp_harmonic.getuv(vrtspec, divspec)
        phig   = sp_harmonic.spectogrd(phispec)
        H0     = H0_values[ncycle]
        
        #################################### create a phiT based on the structure of U mean ########################################
#       UU     = gauss_lat_curve( np.rad2deg(lats), mu,  sigma)*100   ##### Note zonal mean zonal U can also enter here as an array
        phiT           = phi_T(lats, y0, H0,  N, )
        phig_plus_phiT1 = phig + phiT 
        
        mu = 0 
        
        phimax            = np.max(phig_plus_phiT1)
        extra_gauss_term  = 0*(np.cos(np.deg2rad(np.rad2deg(lats)-mu))**10)*phimax/2
        extra_gauss_term1 = 0*(np.cos(np.deg2rad(np.rad2deg(lats)-mu))**100)*phimax/5
        phig_plus_phiT   = phig_plus_phiT1 - extra_gauss_term - extra_gauss_term1
        
        if int(t/(24*3600)) > 5 :
            if t % (4*24*3600) == 0: ### Save every 2 day   
                temp_dict = {'lat':np.rad2deg(lats), 'lon':np.rad2deg(lons), 'phiT':phiT, \
                              'U': ug, 'lat':np.rad2deg(lats), 'phig_plus_phiT':phig_plus_phiT, \
                              'phig_plus_phiT1':phig_plus_phiT1, 'phig':phig, \
                              'extra_gauss_term':extra_gauss_term}
                
                h5saveload.make_sure_path_exists('./temp_June30_plus/')  
                h5saveload.save_dict_to_hdf5(temp_dict, './temp_June30_plus/%d.hdf5'%(t/(24*3600)))
                
                py.figure()
                py.plot(np.rad2deg(lats)[:,0], ug.mean(axis=-1), 'r-')
                py.axhline(0, color='k')
                py.twinx()
                py.plot(np.rad2deg(lats)[:,0], vg.mean(axis=-1), 'b-')
                
                h5saveload.make_sure_path_exists('./temp_June30_plus/figures/') 
                py.savefig('./temp_June30_plus/figures/U_%d_eq_easterly.png'%( int(t/(24*3600) )))
                
                py.figure()
                py.plot(np.rad2deg(lats)[:,0], phig_plus_phiT.mean(axis=-1), 'b-', label='phig_plus_phiT')
                py.plot(np.rad2deg(lats)[:,0], phiT.mean(axis=-1), 'm', label='phiT')
                py.plot(np.rad2deg(lats)[:,0], (phi_T(lats, y0, H0,  N,)).mean(axis=-1), 'm--', label='phiT_calc')
                py.plot(np.rad2deg(lats)[:,0], phig.mean(axis=-1), 'c', label='phig')
                py.plot(np.rad2deg(lats)[:,0], (extra_gauss_term + extra_gauss_term1).mean(axis=-1), 'y-', label='extra_gauss_term', lw=3)
                py.legend(loc='best')

                
                h5saveload.make_sure_path_exists('./temp_June30_plus/figures/') 
                py.savefig('./temp_June30_plus/figures/PHI_%d_eq_easterly.png'%( int(t/(24*3600) )))
                
                ## phig_plus_phiT = phig + phi_T(lats, y0, H0,  N, )
        ############################################################################################################################
        
        ug = ug + 10
        
        
        # compute tendencies.
        u_f_plus_vort = ug*(vrtg+f)
        v_f_plus_vort = vg*(vrtg+f)
        curl_NL_spec, div_NL_spec =   sp_harmonic.getvrtdivspec(u_f_plus_vort, v_f_plus_vort)
        ddivdtspec[:, nnew]       =   curl_NL_spec
        dvrtdtspec[:, nnew]       = - div_NL_spec
                
        
        ############### This one runs with a variable fluid depth ###########################      
        u_phi = ug*phig
        v_phi = vg*phig
        curl_uvphi_NL_spec, div_uvphi_NL_spec = sp_harmonic.getvrtdivspec(u_phi, v_phi)
        ######################################################################################


#         ################ This one runs with a constant fluid depth semi Non linear ###########################
#         u_phi = ug*phig
#         v_phi = vg*phig
#         curl_uvphi_spec,  div_uvphi_spec  = sp_harmonic.getvrtdivspec(ug*(phig + phi_B(Hmean)), vg*(phig + phi_B(Hmean)) )       
#         div_uvphi_L_spec     = sp_harmonic.grdtospec(divg*phig)
#         div_uvphi_NL_spec    = div_uvphi_spec - div_uvphi_L_spec
#         #####################################################################################################
 
        
        

        dphidtspec[:, nnew]   = -div_uvphi_NL_spec
        KE_plus_phi_spec      =  sp_harmonic.grdtospec(phig_plus_phiT+0.5*(ug**2+vg**2))
        ddivdtspec[:, nnew]  += -sp_harmonic.lap*KE_plus_phi_spec

        #### Diffusion term ####
        vrtg_diffuse_spec, div_diffuse_spec   =  sp_harmonic.getvrtdivspec(ug, vg)
        phi_diffuse_spec                      =  sp_harmonic.grdtospec(phig - phi_B(Hmean))
        dvrtdtspec[:, nnew]   += -vrtg_diffuse_spec/K_M
        ddivdtspec[:, nnew]   += -div_diffuse_spec/K_M
        dphidtspec[:, nnew]   += -phi_diffuse_spec/K_T
        
        
        ##### EXTERNAL FORCING ######
        time_component        = 0 #np.sin(2*np.pi*t/(time_period*24*3600))

        phi_forcing           = PHI_perturb_propagating(lons, lats, Q0, yp, xp, Lx, Ly, t, switch_on_day, \
                                                        c=forcing_phase_speed, wave_number=forcing_wave_number, DIPOLE = DIPOLE, switch_off_day=switch_off_day)[0]

#       phi_forcing           = (PHI_perturb_from_file(external_data, time = 0, switch_on_day=None)[0])
        fu_forcing            = (F_uv_perturb(lons, lats, FUo, yp, xp, Lx, Ly, t, switch_on_day)[0])*time_component 
        fv_forcing            = (F_uv_perturb(lons, lats, FVo, yp, xp, Lx, Ly, t, switch_on_day)[0])*time_component 
        
        f_vrt_forcing_spec, f_div_forcing_spec    =   sp_harmonic.getvrtdivspec(fu_forcing, fv_forcing)
        f_vrt_forcing    =   sp_harmonic.spectogrd(f_vrt_forcing_spec)
        f_div_forcing    =   sp_harmonic.spectogrd(f_div_forcing_spec)

        phi_forcing_spec      = sp_harmonic.grdtospec(phi_forcing)/K_T
        dphidtspec[:, nnew]   += phi_forcing_spec
        dvrtdtspec[:, nnew]   += f_vrt_forcing_spec
        ddivdtspec[:, nnew]   += f_div_forcing_spec   
        
        
#         ################################## SAVE THOSE EXTRA ADVECTION TERMS ################################
#         div_NL_grid                   = -sp_harmonic.spectogrd(div_NL_spec)  #### RHS of vorticity equation
#         damping_vrt                   = -vrtg/K_M
        
#         curl_NL_grid                  =  sp_harmonic.spectogrd(curl_NL_spec) #### RHS of divergence equation
#         laplacian_phi_grid            = -sp_harmonic.spectogrd(sp_harmonic.lap*sp_harmonic.grdtospec(phig)) ###  RHS of divergence equation
#         laplacian_phiT_grid           = -sp_harmonic.spectogrd(sp_harmonic.lap*sp_harmonic.grdtospec(phiT)) ###  RHS of divergence equation
#         laplacian_KE_grid             = -sp_harmonic.spectogrd(sp_harmonic.lap*sp_harmonic.grdtospec(0.5*(ug**2+vg**2)))  ###  RHS of divergence equation
#         damping_div                   = -divg/K_M
        
#         div_uvphi_NL_grid             = -sp_harmonic.spectogrd(div_uvphi_NL_spec)  ###  RHS of phi equation
#         damping_phi                   = -(phig - phi_B(Hmean))/K_T
#         forcing_phi                   = +(phi_forcing)/K_T
#         ####################################################################################################

        
        if int(t/(24*3600)) > 5 :           
            if t % (3*3600) == 0: ### Save every 6 hours
                
                PHI_forcing       .append(phi_forcing)
                PHI_T.append(phiT)
     
        # update vort, div, phiv with third-order adams-bashforth.
        # forward euler, then 2nd-order adams-bashforth time steps to start.
        if ncycle == 0:
            dvrtdtspec[:, nnow] = dvrtdtspec[:, nnew]
            dvrtdtspec[:, nold] = dvrtdtspec[:, nnew]
            ddivdtspec[:, nnow] = ddivdtspec[:, nnew]
            ddivdtspec[:, nold] = ddivdtspec[:, nnew]
            dphidtspec[:, nnow] = dphidtspec[:, nnew]
            dphidtspec[:, nold] = dphidtspec[:, nnew]
        elif ncycle == 1:
            dvrtdtspec[:, nold] = dvrtdtspec[:, nnew]
            ddivdtspec[:, nold] = ddivdtspec[:, nnew]
            dphidtspec[:, nold] = dphidtspec[:, nnew]

        vrtspec += dt*(
            (23./12.)*dvrtdtspec[:, nnew] - (16./12.)*dvrtdtspec[:, nnow]
            + (5./12.)*dvrtdtspec[:, nold])
        divspec += dt*(
            (23./12.)*ddivdtspec[:, nnew] - (16./12.)*ddivdtspec[:, nnow]
            + (5./12.)*ddivdtspec[:, nold])
        phispec += dt*(
            (23./12.)*dphidtspec[:, nnew] - (16./12.)*dphidtspec[:, nnow]
            + (5./12.)*dphidtspec[:, nold])
        # implicit hyperdiffusion for vort and div.
        # vrtspec *= hyperdiff_fact
        # divspec *= hyperdiff_fact

        # switch indices, do next time step.
        nsav1 = nnew
        nsav2 = nnow
        nnew = nold
        nnow = nsav1
        nold = nsav2
        
        if t/(24*3600) % 1 == 0 :
            logging_object.write("Calculated day %d"%(t/(24*3600)))
            
            
        if np.isnan(ug).any():  #### this is to make sure that the model which doesn't run stably is aborted immediately
            logging_object.write("ABORTING for Q0 = %d because of a runaway scenario at %d"%(Q0, t/(24*3600)))
            print ("ABORTING for Q0 = %d because of a runaway scenario at %d"%(Q0, t/(24*3600)))
            abort_status='True'
            break

     
    spatial_data  =  {'U': A(U),      'V': A(V),      'PHI': A(PHI),  \
                      'lats': sp_harmonic.lats, 'lons': sp_harmonic.lons, 'T_in_days':A(T),  \
                      'phi_T': A(PHI_T), 'phi_B': phi_B(Hmean), 'phi_forcing': A(PHI_forcing), \
                      'vrt_forcing': A(F_vrt_forcing), 'div_forcing': A(F_div_forcing), 'abort_status':abort_status,\
                      'div': A(DIV),  'vrt': A(VRT),}
    
        
    path2 = path +'/H0_%s/'%(Hmax)
    
    if abort_status == 'False':
        h5saveload.make_sure_path_exists(path2)  
        h5saveload.save_dict_to_hdf5(spatial_data,     path2+'spatial_data.hdf5')
#         h5saveload.save_dict_to_hdf5(equation_terms,   path2+'equation_data.hdf5')
        h5saveload.save_dict_to_hdf5(input_file2,      path2+'input_file.hdf5')

        logging_object.write("Saved data in %s"%(path2))
    
    
if __name__ == "__main__": 
    
    for H0 in [5000]:   #, 3000, 3500, 4000, 4500, 5000]:  
        for Q0 in [10]:  ### 0.1, 10, 50, 100, 125, 250, 500
            for HMEAN in [500]:  ##50   
                for forcing_y_loc in [0]:  
                    for forcing_phase_speed in [0]: ##0                                            
                        for DIPOLE_or_MONOPOLE in [True]:
                            
                            for alphas, switch_on_days, keep_forcing_const_for_days  in zip([1], [170], [350]):
                                                            
                                start_time=ti.time() 

                                input_file = {  'nlons'        : 256        , \
                                                'ntrunc'       : int(256/3) , \
                                                'nlats'        : int(256/2) , \
                                                'dt'           : int(150/1)        , \
                                                'itmax'        : 1*40*int(86400/150) , \
                                                'U_up_days'    : 150         , \
                                                'rsphere'      : 6.37122e6  , \
                                                'omega'        : 7.292e-5   , \
                                                'grav'         : 9.80616    , \
                                                'y0'           : 0,  'N':2, 'Hmax': 2000, \
                                                'Hmean'        : HMEAN , \
                                                'Q0'           : Q0 , 'yp'  : 0 , 'xp': 90 , 'Lx': 30 , 'Ly': 10, \
                                                'FUo'          : 0 ,  'FVo' : 0 ,  \
                                                'K_M'          : 20*24*3600, 'K_T': 10*24*3600, \
                                                'forcing_phase_speed': forcing_phase_speed, \
                                                'forcing_wave_number': 2 , \
                                                'DIPOLE'         : DIPOLE_or_MONOPOLE, \
                                                'switch_on_day'  :  switch_on_days, \
                                                'alpha'          : alphas, \
                                                'keep_forcing_const_for_day' : keep_forcing_const_for_days,\
                                                'path'           : './equator_easterlies_June30_constrain_Heq/',} ; 

                                input_file['ntrunc']         = int(input_file['nlons']/3)
                                input_file['nlats']          = int(input_file['nlons']/2)
                                input_file['itmax']          = 180*int(86400/int(input_file['dt']))   #### Here 150 is in the units of days
                                input_file['Hmax']           = H0
                                input_file['switch_off_day'] = input_file['switch_on_day'] + input_file['keep_forcing_const_for_day']
                                input_file['yp']             = forcing_y_loc
                                input_file['U_up_days']      = (130 - input_file['alpha']*3)//2

                                forcing_struct = 'dipole_heat' if DIPOLE_or_MONOPOLE else 'mono_heat'

                                input_file['forcing_struct']   = forcing_struct
                                input_file['path']             = \
                                input_file['path']+'/dt_%d_Q_forcing_%d_forcing_y_%d_Hmean_%d_forcing_phase_speed_%d_ms/U_up_days_%d/'%(\
                                                                                input_file['dt'],\
                                                                                input_file['Q0'],                  input_file['yp'], \
                                                                                input_file['Hmean'], input_file['forcing_phase_speed'], \
                                                                                input_file['U_up_days'])

                                logging_object = logruns.default_log(logfilename   = 'H0_%d_Q0_%d_Hmean_%d_eq_easterlies'%(\
                                                                                      H0, Q0, HMEAN),  \
                                                                     log_directory = './log/')



                                logging_object.write('**********************************')
                                logging_object.write(input_file['path'])
                                logging_object.write('**********************************')
                                path2 = input_file['path'] +'/H0_%s/'%(input_file['Hmax'])

                                if not os.path.exists(path2):
                                    print (' %s integrating'%(path2))
                                    integrate_model(input_file)
                                else:
                                    logging_object.write( ' %s exists'%(path2) )
                                    print (' %s exists'%(path2))

                                end_time=ti.time() 

                                logging_object.write(' ========== CODE RAN SUCCESSFULLY, congrats! ================')
                                logging_object.write(' -----> Total Time taken = %1.3f  <----'%(end_time-start_time))






