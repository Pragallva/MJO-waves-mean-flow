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

def perturb(lons, lats, Q0=100, yp=0, xp=90, Lx=30, Ly=10, time = 0, switch_on_day=None):
    
    if switch_on_day:
        switch_on_time = switch_on_day*24*3600
        if time >= switch_on_time:
            flag = 1
        else:
            flag = 0
    else:
        flag = 1
        
    xp, yp, Lx, Ly      = map(lambda z: np.deg2rad(z), [xp, yp, Lx, Ly ])
    phi_perturb         = flag* Q0*np.exp(-   (((lats-yp)**2/Ly**2) +  ((lons-xp)**2/Lx**2))   )
    phi_mean            = np.mean(phi_perturb, axis=-1, keepdims=True)    
    phi_peturb_anom     = phi_perturb - phi_mean
    return phi_perturb,  phi_peturb_anom

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
    
    U   = []; 
    V   = [];
    PHI = [];
    VRT = [];
    DIV = [];
    
    U_spec   = []; 
    V_spec   = [];
    PHI_spec = [];
    VRT_spec = [];
    DIV_spec = [];
    
    T = [];
    
    
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
    
    H0_values = np.ones(itmax)*Hmax
    tmax_25   = spin_up_time*int(86400/dt)
    H0_spinup = np.linspace(0, Hmax, tmax_25)
    H0_values[:tmax_25] = H0_spinup
    
    for ncycle in tqdm(range(itmax)):
        
        t = ncycle*dt
        
        if int(t/(24*3600)) > 5 :
            
            if t % (3*3600) == 0: ### Save every 6 hours
                U.append(ug)
                V.append(vg)
                PHI.append(phig)
                VRT.append(vrtg)
                DIV.append(divg)
                
                U_spec.append(sp_harmonic.grdtospec(ug))
                V_spec.append(sp_harmonic.grdtospec(vg))
                PHI_spec.append(phispec)
                VRT_spec.append(vrtspec)
                DIV_spec.append(divspec)
                
                T.append(t/(24*3600))
        
        
        # get vort, u, v, phi on grid
        vrtg   = sp_harmonic.spectogrd(vrtspec)
        ug, vg = sp_harmonic.getuv(vrtspec, divspec)
        phig   = sp_harmonic.spectogrd(phispec)
        H0 = H0_values[ncycle]
        phig_plus_phiT = phig + phi_T(lats, y0, H0,  N, )

        # compute tendencies.
        u_f_plus_vort = ug*(vrtg+f)
        v_f_plus_vort = vg*(vrtg+f)
        curl_NL_spec, div_NL_spec = sp_harmonic.getvrtdivspec(u_f_plus_vort, v_f_plus_vort)
        ddivdtspec[:, nnew]       =   curl_NL_spec
        dvrtdtspec[:, nnew]       = - div_NL_spec

        u_phi = ug*phig
        v_phi = vg*phig
        curl_uvphi_NL_spec, div_uvphi_NL_spec = sp_harmonic.getvrtdivspec(u_phi, v_phi)

        dphidtspec[:, nnew]   = -div_uvphi_NL_spec
        KE_plus_phi_spec      =  sp_harmonic.grdtospec(phig_plus_phiT+0.5*(ug**2+vg**2))
        ddivdtspec[:, nnew]  += -sp_harmonic.lap*KE_plus_phi_spec

        #### Diffusion term ####
        vrtg_diffuse_spec, div_diffuse_spec   =  sp_harmonic.getvrtdivspec(ug, vg)
        phi_diffuse_spec                      =  sp_harmonic.grdtospec(phig - phi_B(Hmean))
        dvrtdtspec[:, nnew]   += -vrtg_diffuse_spec/K_M
        ddivdtspec[:, nnew]   += -div_diffuse_spec/K_M
        dphidtspec[:, nnew]   += -phi_diffuse_spec/K_T

        ### Forcing term ###perturb(lons, lats, Q0=100, yp=0, xp=90, Lx=30, Ly=10, time = 0, switch_on_day=None)
        phi_forcing            = perturb(lons, lats, Q0, yp, xp, Lx, Ly, t, switch_on_day)[1]
        phi_forcing_spec       = sp_harmonic.grdtospec(phi_forcing)/K_T
        dphidtspec[:, nnew]   += phi_forcing_spec


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
            logging_object.write("Calculated day %d"%(t/24*3600))

    
    spatial_data  =  {'U': A(U),      'V': A(V),      'PHI': A(PHI),     'VRT': A(VRT),      'DIV': A(DIV),      \
                      'lats': sp_harmonic.lats, 'lons': sp_harmonic.lons, 'T_in_days':A(T),  \
                      'phi_T': phi_T(lats, y0, H0,  N)[0], 'phi_B': phi_B(Hmean)}
    spectral_data =  {'U': A(U_spec), 'V': A(V_spec), 'PHI': A(PHI_spec),'VRT': A(VRT_spec), 'DIV': A(DIV_spec), \
                      'l': sp_harmonic.degree, 'm': sp_harmonic.m, 'T_in_days':A(T), \
                      'phi_T': phi_T(lats, y0, H0,  N)[0], 'phi_B': phi_B(Hmean)}
    
    path2 = path +'/H0_%s/'%(Hmax)
    h5saveload.make_sure_path_exists(path2)  
    h5saveload.save_dict_to_hdf5(spatial_data,  path2+'spatial_data.hdf5')
    h5saveload.save_dict_to_hdf5(spectral_data, path2+'spectral_data.hdf5')
    
    logging_object.write("Saved data in %s"%(path2))
    

    
     
if __name__ == "__main__": 
    
    start_time=ti.time() 
    
    H0            = int(sys.argv[1])
    switch_on_day = int(sys.argv[2])
    
    logging_object = logruns.default_log(logfilename   = 'H0=%d'%(H0), \
                                         log_directory = '~/Work/spherical_SWE/shell_scripts/log/')
    
    
    input_file ={   'nlons'   : 256,          \
                    'ntrunc'  : int(256/3), \
                    'nlats'   : int(256/2), \
                    'dt'      : 150        , \
                    'itmax'   : 100*int(86400/150) , \
                    'spin_up_time' : 25,     \
                    'rsphere' : 6.37122e6  , \
                    'omega'   : 7.292e-5   , \
                    'grav'    : 9.80616    , \
                    'umax'    : 30.        , \
                    'hamp '   : 120.       , \
                    'efold'   : 3.*3600.   , \
                    'ndiss'   : 8          , \
                    'y0'      : 0,  'N':2, 'Hmax': 2000, \
                    'Hmean'   : 800 , \
                    'Q0'      : 100 , 'yp': 0 , 'xp': 90 , 'Lx': 30 , 'Ly': 10, \
                    'K_M'     : 20*24*3600, 'K_T': 10*24*3600, \
                    'switch_on_day': 25,\
                    'path'    : '/data/pbarpanda/spherical_SWE/reproduce_monteiro_2014/'} ; 
    
    
    input_file['ntrunc']        = int(input_file['nlons']/3)
    input_file['nlats']         = int(input_file['nlons']/2)
    input_file['itmax']         = 150*int(86400/int(input_file['dt']))
    input_file['Hmax']          = H0
    input_file['switch_on_day'] = switch_on_day
    input_file['path']          = input_file['path']+'/transient/switch_day_%d/'%(switch_on_day)
    
    
    integrate_model(input_file)
    
    end_time=ti.time() 
    
    logging_object.write(' ========== CODE RAN SUCCESSFULLY, congrats! ================')
    logging_object.write(' -----> Total Time taken = %1.3f  <----'%(end_time-start_time))
    
    
    
    



