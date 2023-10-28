from typing import Optional
from config import dtype, backend

from gt4py.cartesian import gtscript, IJ, K
from phyex_gt4py.constants import Constants

from phyex_gt4py.dimphyex import DIMPhyex
from phyex_gt4py.functions import compute_ice_frac
from phyex_gt4py.functions.ice_adjust import latent_heat, _cph
from phyex_gt4py.functions.icecloud import icecloud
from phyex_gt4py.nebn import Neb
from phyex_gt4py.rain_ice_param import ParamIce, RainIceDescr, RainIceParam


@gtscript.stencil(backend=backend)
def condensation(
    Cst: Constants,
    nebn: Neb, 
    icep: RainIceParam,
    zz: gtscript.Field[dtype],              # height of model levels
    pabs: gtscript.Field[dtype],            # pressure (Pa)
    rhodref: gtscript.Field[dtype],   
    t: gtscript.Field[dtype],               # T (K)      
    rv_in: gtscript.Field[dtype],
    rc_in: gtscript.Field[dtype],
    ri_in: gtscript.Field[dtype],
    rv_out: gtscript.Field[dtype],
    rc_out: gtscript.Field[dtype],
    ri_out: gtscript.Field[dtype],  
    rs: gtscript.Field[dtype],              # grid scale mixing ratio of snow (kg/kg)
    rr: gtscript.Field[dtype],              # grid scale mixing ratio of rain (kg/kg)
    rg: gtscript.Field[dtype],              # grid scale mixing ratio of graupel (kg/kg)
    sigs: gtscript.Field[dtype],            # Sigma_s from turbulence scheme
    prifact: gtscript.Field[dtype],
    picldfr: gtscript.Field[dtype],         # pure ice cloud fraction
    pwcldfr: gtscript.Field[dtype],         # pure water cloud fraction
    ls: Optional[gtscript.Field[dtype]],
    lv: Optional[gtscript.Field[dtype]],
    cph: Optional[gtscript.Field[dtype]],
    ifr: gtscript.Field[dtype],             # ratio cloud ice moist part
    ssio: gtscript.Field[dtype],            # super-saturation with respect to ice in the super saturated fraction 
    ssiu: gtscript.Field[dtype],            # super-saturatiin with respect to in in the sub saturated fraction
   
    # Temporary fields
    itpl: gtscript.Field[IJ, dtype],        # 2D field to store tropopause height
    tmin: gtscript.Field[IJ, dtype],        # Temperature at tropopause
    cpd: gtscript.Field[dtype],
    
    tlk: gtscript.Field[dtype],             # working array for T_l
    rt: gtscript.Field[dtype],              # work array for total water mixing ratio
    pv: gtscript.Field[dtype],              # thermodynamics
    piv: gtscript.Field[dtype],             # thermodynamics
    qsl: gtscript.Field[dtype],             # thermodynamics
    qsi: gtscript.Field[dtype],
    
    t_tropo: gtscript.Field[IJ, dtype],     # temperature at tropopause
    z_tropo: gtscript.Field[IJ, dtype],     # height at tropopause
    z_ground: gtscript.Field[IJ, dtype],    # height at ground level (orography) 
    l: gtscript.Field[dtype],               # length scale       
    
    
    dz: gtscript.Field[dtype],              # Layer thickness   
    cldini: gtscript.Field[dtype],          # To be initialized for icecloud
    
    frac: gtscript.Field[dtype],            # ice fraction
    
    ouseri: bool,                           # switch to compute both liquid and solid condensate (True) or only solid condensate (False)
    osigmas: bool,                          # use present global sigma_s values (sigs) or that from turbulence scheme
    ocnd2: bool = False,                    # logical switch to separate liquid and ice
                             
    
    csigma: dtype = 0.2,                    # constant in sigma_s parametrization
    csig_conv: dtype = 3e-3                 # scaling factor for ZSIG_CONV as function of mass flux 
    
    
):
    
    # TODO : toutes les initialisations sont à écrire
    prifact = 0 if ocnd2 else 1
    
    with computation(PARALLEL), interval(...):
        
        rt = rv_in + rc_in + ri_in * prifact
        
    if ls is None and lv is None:
        # TODO : init latent heat of vaporisation / sublimation
        with computation(PARALLEL), interval(...):
            lv, ls = latent_heat(Cst, t)
            
    # line 264        
    if cph is None:
        with computation(PARALLEL), interval(...):
            cpd = _cph(Cst, rv_in, rc_in, ri_in, rr, rs, rg)
    
    # Preliminary calculations for computing the turbulent part of Sigma_s   
    if not osigmas:
        
        with computation(PARALLEL), interval(...):
            # Temperature at saturation
            tlk = t[0, 0, 0] - lv * rc_in / cpd - ls * ri_in / cpd * prifact
                      
        # Set the mixing length scale 
        # @stencil
        mixing_length_scale(
            t 
            zz,
            t_tropo,
            z_tropo,
            z_ground,
            l
        )
        
    # line 313 
    if ocnd2:
        
        with computation(FORWARD):
            
            with interval(0, 1):
                dz = zz[0, 0, 0] - zz[0, 0, 1]
                
            with interval(1, None):
                dz = zz[0, 0, 1] - zz[0, 0, 0]      
                
        with computation(FORWARD), interval(...):
            icecloud(
                Cst=Cst,
                p=pabs,
                z=zz,
                dz=dz,
                t=t,
                r=rv_in,
                tstep=1,
                pblh=-1,
                wcld=cldini,
                sifrc=ifr,
                ssio=ssio,
                ssiu=ssiu,
                w2_out=ardum2,
                rsi=ardum 
            )
            
        with computation(PARALLEL), interval(...):
            esatw_t[0, 0] = esatw(t[0, 0, 0])
            pv[0, 0] = min(esatw_t[0, 0], 0.99 * pabs[0, 0, 0])
            piv[0, 0] = min(esati(t[0, 0, 0]),0.99 * pabs[0, 0, 0]) 
            
    else:    
        with computation(PARALLEL), interval(...):
            pv[0, 0] = min(exp(Cst.alpw - Cst.betaw / t[0, 0, 0] - Cst.gamw * log(t[0, 0, 0])), 0.99 * pabs[0, 0, 0])
            piv[0, 0] = min(exp(Cst.alpi - Cst.betai / t[0, 0, 0]) - Cst.gami * log(t[0, 0, 0]), 0.99 * pabs[0, 0, 0])
            
            
    if ouseri and not ocnd2:
        
        
        with computation(PARALLEL), interval(...):
            if rc_in[0, 0, 0] > ri_in[0, 0, 0] > 1e-20:
                frac[0, 0] = ri_in[0, 0, 0] / (rc_in[0, 0, 0] + ri_in[0, 0, 0])
                
            compute_ice_frac(hfrac_ice, nebn, frac, t)
            
            qsl[0, 0] = Cst.Rd / Cst.Rv * pv[0, 0] / (pabs[0, 0, 0] - pv[0, 0])
            qsi[0, 0] = Cst.Rd / Cst.Rv * piv[0, 0] / (pabs[0, 0, 0] - piv[0, 0])
            
            # interpolate bewteen liquid and solid as a function of temperature
            qsl = (1 - frac) * qsl + frac * qsi
            lvs = (1 - frac) * lv + frac * ls
            
            # coefficients a et b
            ah = lvs * qsl / (Cst.Rv * t[0, 0, 0] ** 2) * (1 + Cst.Rv * qsl / Cst.Rd)
            a = 1 / (1 + lvs / cpd[0, 0, 0] * ah)
            b = ah * a
            sbar = a * (rt[0, 0, 0] - qsl[0, 0] + ah * lvs * (rc_in + ri_in * prifact) / cpd)            
    
    # Meso-NH turbulence scheme
    if osigmas:
        with computation(PARALLEL):

            if sigqsat != 0:
                
                # intialization
                with interval(...):
                    dzfact = 1
                if  nebn.hgt_qs:
                    with interval(0, -1):                
                        dzfact = max(icep.frmin[23], min(icep.frmin[24], (zz[0, 0, 0] - zz[0, 0, 1]) / dzref))
                    with interval(-1, -2):
                        dzfact = max(icep.frmin[23], min(icep.frmin[24], (zz[0, 0, 0] - zz[0, 0, 1]) * 0.8 / dzref))
                        
                if nebn.statnw:
                    with interval(...):
                        sigma = sqrt( sigs ** 2 + (sigqsat * dzfact * qsl * a) **2)    
                else: 
                    with interval(...):
                        sigma = sqrt( (2 * sigs) ** 2 + (sigqsat * qsl * a) **2)
                        
            else:
                if nebn.statnw:
                    with interval(...):
                        sigma = sigs[0, 0, 0]
                else: 
                    with interval(...):
                        sigma = 2 * sigs[0, 0, 0]
                        
    else: 
        
        # Parametrize Sigam_s with first order closure
        with computation(FORWARD):
            
            with interval(0, 1):
                NotImplemented
             
            with interval(1, -1):
                dzz = zz[0, 0, 1] - zz[0, 0, -1]
                drw = rt[0, 0, 1] - rt[0, 0, -1]
                dtl = tlk[0, 0, 1] - tlk[0, 0, -1] + Cst.gravity0 / cpd[0, 0, 0] * dzz
                ll = l[0, 0, 0]
                
            with interval(-1, -2):
                NotImplemented
                
            # Standard deviation due to convection
            sig_conv = 0
            if lmfconv:
                sig_conv = csig_conv * mfconv[0, 0, 0] / a[0, 0]
                
            sigma[0, 0] = sqrt(max(1e-25, (csigma * ll / zz * a* drw) ** 2 - 2 * a * b * drw * dtl + (b * dtl) ** 2 + sig_conv ** 2))    
     

# TODO: enable computation on an AROME like grid        
@gtscript.stencil(backend=backend)           
def mixing_length_scale(
    t: gtscript.Field[dtype],               # temperature
    zz: gtscript.Field[dtype],              # equivalent to zcr
    t_tropo: gtscript.Field[IJ, dtype],     # champ 2D temperature tropopause   
    z_tropo: gtscript.Field[IJ, dtype],     # champ 2D hauteur tropopause   
    z_ground: gtscript.Field[IJ, dtype],    # champ 2D hauteur de référence
    l: gtscript.Field[dtype],               # mixing length scale 
    l0: dtype = 600                         # tropospheric length scale
):
    
    # tropopause height computation
    with computation(PARALLEL):
        
        t_tropo[0, 0] = 400
        while t_tropo[0, 0] > t[0, 0, 0]:
            t_tropo[0, 0] = t[0, 0, 0]
            z_tropo[0, 0] = zz[0, 0, 0]
            
    # length scale computation
    # ground to top
    # TODO: enable computation on an AROME like grid
    with computation(PARALLEL):
        
        with interval(0, 1):
            l[0, 0, 0] = 20
            
        with interval(1, None):
        
            zz_to_ground = zz[0, 0, 0] - z_ground[0, 0]

            # approximate length for boundary layer
            if zz_to_ground > l0:
                l[0, 0, 0] = zz_to_ground
            # gradual decrease of length-scale near and above tropopause
            if zz_to_ground > 0.9 * (z_tropo[0, 0] - z_ground[0, 0]):
                l[0, 0, 0] = 0.6 * l[0, 0, -1]
            # free troposphere 
            else:
                l[0, 0, 0] = l0
                
