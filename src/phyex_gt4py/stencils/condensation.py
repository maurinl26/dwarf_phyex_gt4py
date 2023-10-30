from typing import Optional
from config import dtype, backend

from gt4py.cartesian import gtscript, IJ, K
from phyex_gt4py.constants import Constants

from phyex_gt4py.dimphyex import DIMPhyex
from phyex_gt4py.functions import compute_ice_frac
from phyex_gt4py.functions.cloud_fractions import cloud_fraction
from phyex_gt4py.functions.ice_adjust import latent_heat, _cph
from phyex_gt4py.functions.icecloud import icecloud
from phyex_gt4py.functions.erf import erf
from phyex_gt4py.functions.temperature import update_temperature
from phyex_gt4py.functions.tiwmx import esati, esatw
from phyex_gt4py.nebn import Neb
from phyex_gt4py.rain_ice_param import ParamIce, RainIceDescr, RainIceParam


@gtscript.stencil(backend=backend)
def condensation(
    Cst: Constants,
    nebn: Neb, 
    icep: RainIceParam,
    hfrac_ice: str,
    hcondens: str,
    hlambda3: str,                          # formulation for lambda3 coeff
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
    
    lmfconv: bool,                          # True if mfconv.size != 0
    mfconv: gtscript.Field[dtype],          # convective mass flux (kg.s⁻1.m⁻2)
    
    prifact: gtscript.Field[dtype],
    cldfr: gtscript.Field[dtype],
    sigrc: gtscript.Field[dtype],           # s r_c / sig_s ** 2
    icldfr: gtscript.Field[dtype],         # ice cloud fraction
    wcldfr: gtscript.Field[dtype],         # water por mixed-phase cloud fraction
    ls: Optional[gtscript.Field[dtype]],
    lv: Optional[gtscript.Field[dtype]],
    cph: Optional[gtscript.Field[dtype]],
    ifr: gtscript.Field[dtype],             # ratio cloud ice moist part
    sigqsat: gtscript.Field[dtype],         # use an extra qsat variance contribution (if osigma is True)
    
    ssio: gtscript.Field[dtype],            # super-saturation with respect to ice in the super saturated fraction 
    ssiu: gtscript.Field[dtype],            # super-saturation with respect to in in the sub saturated fraction
    
    hlc_hrc: Optional[gtscript.Field[dtype]],         #
    hlc_hcf: Optional[gtscript.Field[dtype]],         # cloud fraction
    hli_hri: Optional[gtscript.Field[dtype]],         # 
    hli_hcf: Optional[gtscript.Field[dtype]],  
    ice_cld_wgt: gtscript.Field[dtype],     # in        
   
    # Temporary fields
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
    
    frac_tmp: gtscript.Field[IJ, dtype],    # ice fraction
    cond_tmp: gtscript.Field[IJ, dtype],   # condensate
    
    a: gtscript.Field[IJ, dtype],           # related to computation of Sig_s
    b: gtscript.Field[IJ, dtype],
    sbar: gtscript.Field[IJ, dtype],
    sigma: gtscript.Field[IJ, dtype],
    q1: gtscript.Field[IJ, dtype],
        
    # related to ocnd2 ice cloud calculation
    esatw_t: gtscript.Field[IJ, dtype],     
    ardum: gtscript.Field[IJ, dtype],
    ardum2: gtscript.Field[IJ, dtype],
    dz: gtscript.Field[IJ, dtype],          # Layer thickness 
    cldini: gtscript.Field[IJ, dtype],      # To be initialized for icecloud
    
    dum4: gtscript.Field[dtype],
    prifact_tmp: gtscript.Field[IJ, dtype],
    lwinc: gtscript.Field[IJ, dtype],
    
    # related to ocnd2 noise check
    rsp: dtype,
    rsw: dtype,
    rfrac: dtype,
    rsdif: dtype,
    rcold: dtype,
    
    dzfact: dtype,                          # lhgt_qs
    dzref: dtype,                            
    
    inq1: int,
    inc: dtype,
    
    ouseri: bool,                           # switch to compute both liquid and solid condensate (True) or only solid condensate (False)
    osigmas: bool,                          # use present global sigma_s values (sigs) or that from turbulence scheme
    ocnd2: bool = False,                    # logical switch to separate liquid and ice              
    
    csigma: dtype = 0.2,                    # constant in sigma_s parametrization
    csig_conv: dtype = 3e-3                 # scaling factor for ZSIG_CONV as function of mass flux 
    
    
):
    
    dzref = icep.frmin[25]
    prifact = 0 if ocnd2 else 1

    # Initialize values
    with computation(PARALLEL), interval(...):
        cldfr[0, 0, 0] = 0
        sigrc[0, 0, 0] = 0
        rv_out[0, 0, 0] = 0
        rc_out[0, 0, 0] = 0
        ri_out[0, 0, 0] = 0
        
        # local fields 
        ardum2[0, 0] = 0
        cldini[0, 0, 0] = 0
        ifr[0, 0, 0] = 10
        frac_tmp[0, 0] = 0
    
    with computation(PARALLEL), interval(...):
        
        rt[0, 0, 0] = rv_in + rc_in + ri_in * prifact
        
        if ls is None and lv is None:
            lv, ls = latent_heat(Cst, t)

        if cph is None:
            cpd = _cph(Cst, rv_in, rc_in, ri_in, rr, rs, rg)
    
    # Preliminary calculations for computing the turbulent part of Sigma_s   
    if not osigmas:
        
        with computation(PARALLEL), interval(...):
            # Temperature at saturation
            tlk = t[0, 0, 0] - lv * rc_in / cpd - ls * ri_in / cpd * prifact
                      
        # Set the mixing length scale 
        # @stencil
        mixing_length_scale(
            t,
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
            piv[0, 0] = min(esati(t[0, 0, 0]), 0.99 * pabs[0, 0, 0]) 
            
    else:    
        with computation(PARALLEL), interval(...):
            pv[0, 0] = min(exp(Cst.alpw - Cst.betaw / t[0, 0, 0] - Cst.gamw * log(t[0, 0, 0])), 0.99 * pabs[0, 0, 0])
            piv[0, 0] = min(exp(Cst.alpi - Cst.betai / t[0, 0, 0]) - Cst.gami * log(t[0, 0, 0]), 0.99 * pabs[0, 0, 0])
                    

    if ouseri and not ocnd2:
         
        with computation(PARALLEL), interval(...):
            if rc_in[0, 0, 0] > ri_in[0, 0, 0] > 1e-20:
                frac_tmp[0, 0] = ri_in[0, 0, 0] / (rc_in[0, 0, 0] + ri_in[0, 0, 0])
                
            compute_ice_frac(hfrac_ice, nebn, frac_tmp, t)
            
            qsl[0, 0] = Cst.Rd / Cst.Rv * pv[0, 0] / (pabs[0, 0, 0] - pv[0, 0])
            qsi[0, 0] = Cst.Rd / Cst.Rv * piv[0, 0] / (pabs[0, 0, 0] - piv[0, 0])
            
            # interpolate bewteen liquid and solid as a function of temperature
            qsl = (1 - frac_tmp) * qsl + frac_tmp * qsi
            lvs = (1 - frac_tmp) * lv + frac_tmp * ls
            
            # coefficients a et b
            ah = lvs * qsl / (Cst.Rv * t[0, 0, 0] ** 2) * (1 + Cst.Rv * qsl / Cst.Rd)
            a = 1 / (1 + lvs / cpd[0, 0, 0] * ah)
            b = ah * a
            sbar = a * (rt[0, 0, 0] - qsl[0, 0] + ah * lvs * (rc_in + ri_in * prifact) / cpd)            
    
    # Meso-NH turbulence scheme
    if osigmas:
        with computation(PARALLEL):

            if sigqsat[0, 0, 0] != 0:
                
                # intialization
                if nebn.statnw :     
                    if nebn.hgt_qs:
                        with interval(0, -1):                
                            dzfact = max(icep.frmin[23], min(icep.frmin[24], (zz[0, 0, 0] - zz[0, 0, 1]) / dzref))
                        with interval(-1, -2):
                            dzfact = max(icep.frmin[23], min(icep.frmin[24], (zz[0, 0, 0] - zz[0, 0, 1]) * 0.8 / dzref))
                    else:
                        with interval(...):
                            dzfact = 1
                        
                    with interval(...):
                        sigma = sqrt(sigs ** 2 + (sigqsat * dzfact * qsl * a) **2)    
                else: 
                    sigma = sqrt((2 * sigs) ** 2 + (sigqsat * qsl * a) **2)
                        
            else:
                with interval(...):
                    sigma = sigs[0, 0, 0] if nebn.statnw else 2 * sigs[0, 0, 0]
            
                        
    else: 
        
        # Parametrize Sigma_s with first order closure
        with computation(FORWARD):
            
            with interval(0, 1):
                dzz = zz[0, 0, 1] - zz[0, 0, 0]
                drw = rt[0, 0, 1] - rt[0, 0, 0]
                dtl = tlk[0, 0, 1] - tlk[0, 0, 0] + Cst.gravity0 / cpd[0, 0, 0] * dzz
             
            with interval(1, -1):
                dzz = zz[0, 0, 1] - zz[0, 0, -1]
                drw = rt[0, 0, 1] - rt[0, 0, -1]
                dtl = tlk[0, 0, 1] - tlk[0, 0, -1] + Cst.gravity0 / cpd[0, 0, 0] * dzz
                
            with interval(-1, -2):
                dzz = zz[0, 0, 0] - zz[0, 0, -1]
                drw = rt[0, 0, 0] - rt[0, 0, -1]
                dtl = tlk[0, 0, 0] - tlk[0, 0, -1] + Cst.gravity0 / cpd[0, 0, 0] * dzz
                
            with interval(...):
                # Standard deviation due to convection
                sig_conv = csig_conv * mfconv[0, 0, 0] / a[0, 0] if lmfconv else 0
                sigma[0, 0] = sqrt(max(1e-25, (csigma * l / zz * a* drw) ** 2 - 2 * a * b * drw * dtl + (b * dtl) ** 2 + sig_conv ** 2))   
            
    with computation(PARALLEL), interval(...):
        sigma[0, 0] = max(1e-10, sigma[0, 0])
        
        # normalized saturation deficit
        q1[0, 0] = sbar[0, 0] / sigma[0, 0]

    # TODO : line 422 
    if hcondens == "GAUS":
        
        with computation(PARALLEL), interval(...):
        
            # Gaussian probability density function around q1
            # computation of g and gam(=erf(g))
            gcond, gauv = erf(Cst, q1)
        
            # computation of cloud fraction (output)
            cldfr[0, 0, 0] = max(0, min(1, 0.5 * gauv))
        
            # computation of condensate
            cond_tmp[0, 0] = (exp(-gcond ** 2) - gcond * sqrt(Cst.pi) * gauv) * sigma[0, 0] / sqrt(2 * Cst.pi)
            cond_tmp[0, 0] = max(cond_tmp[0, 0], 0)
        
            sigrc[0, 0, 0] = cldfr[0, 0, 0]
        
            # Computation warm/cold cloud fraction and content in high water
            if hlc_hcf is not None and hlc_hrc is not None:
            
                if 1- frac_tmp > 1e-20:
                    autc = (sbar[0, 0] - icep.criautc / (rhodref[0, 0, 0] * (1 - frac_tmp[0, 0]))) / sigma[0, 0]
                
                    gautc, gauc = erf(Cst, autc) # approximation of erf function for Gaussian distribution

                    hlc_hcf[0, 0, 0] = max(0, min(1, 0.5 * gauc))
                    hlc_hrc[0, 0, 0] = (1 - frac_tmp[0, 0]) * (exp(-gautc ** 2) - gautc * sqrt(Cst.pi) * gauc) * sigma[0, 0] / sqrt(2 * Cst.pi)
                    hlc_hrc[0, 0, 0] += icep.criautc / rhodref[0, 0, 0] * hlc_hcf[0, 0, 0]
                    hlc_hrc[0, 0, 0] = max(0, hlc_hrc[0, 0, 0])
                
                # TODO : initialize hlc_hcf, hlc_hrc to 0 and remove unused conditions
                else:
                    hlc_hcf[0, 0, 0] = 0
                    hlc_hrc[0, 0, 0] = 0
                
            if hli_hcf is not None and hli_hri is not None:
                # TODO : same as above with hli_hcf and hli_hri (to be factorized)
                if frac_tmp[0, 0, 0] > 1e-20:
                    criauti = min(icep.criauti, 10**(icep.acriauti * (t[0, 0, 0] - Cst.tt) + icep.bcriauti))
                    auti = (sbar[0, 0] - criauti / frac_tmp[0, 0]) / sigma[0, 0]
                    
                    gauti, gaui = erf(Cst, auti)
                    hli_hcf = max(0, min(1, 0.5 * gaui))
                    
                    hli_hri[0, 0, 0] = frac_tmp[0, 0] * (exp(-gauti**2)- gauti * sqrt(Cst.pi) * gaui)* sigma[0, 0] / sqrt(2 * Cst.pi)
                    hli_hri[0, 0, 0] += criauti * hli_hcf[0, 0, 0]
                    hli_hri[0, 0, 0] = max(0, hli_hri[0, 0, 0])
                    
                else:
                    hli_hcf[0, 0, 0] = 0
                    hli_hri[0, 0, 0] = 0
                             
    elif hcondens == "CB02":
        
        with computation(PARALLEL), interval(...):
            
            if q1 > 0 and q1 <= 2:
                cond_tmp[0, 0] = min(exp(-1) + 0.66 * q1[0, 0] + 0.086 * q1[0, 0] ** 2, 2)  # we use the MIN function for continuity
            elif q1 > 2:
                cond_tmp[0, 0] = q1
            else:
                cond_tmp[0, 0] = exp(1.2 * q1[0, 0] - 1)
            
            cond_tmp[0, 0] *= sigma[0, 0]
            
            # cloud fraction
            if cond_tmp[0, 0] < 1e-12:
                cldfr[0, 0, 0] = 0
            else:
                cldfr[0, 0, 0] = max(0, min(1, 0.5 + 0.36 * atan(1.55 * q1[0, 0])))
                
            if cldfr[0, 0, 0] == 0:
                cond_tmp[0, 0] = 0
                
            inq1 = min(10, max(-22, floor(min(-100, 2*q1[0, 0])))) # inner min/max prevents sigfpe when 2*zq1 does not fit into an int
            inc = 2*q1 - inq1
            
            sigrc[0, 0, 0] = min(1, (1 - inc) * src_1d[inq1] + inc * src_1d[inq1 + 1])

            if hlc_hcf is not None and hlc_hrc is not None:
                hlc_hcf[0, 0, 0] = 0
                hlc_hrc[0, 0, 0] = 0
                
            if hli_hcf is not None and hli_hri is not None:
                hli_hcf[0, 0, 0] = 0
                hli_hri[0, 0, 0] = 0             
                
    # ref -> line 515    
    if not ocnd2:
        
        with computation(PARALLEL), interval(...):
        
            rc_out[0, 0, 0] = (1 - frac_tmp[0, 0]) * cond_tmp[0, 0] # liquid condensate
            ri_out[0, 0, 0] = frac_tmp[0, 0] * cond_tmp[0, 0]       # solid condensate
            t[0, 0, 0] = update_temperature(t, rc_in, rc_out, ri_in, ri_out, lv, ls, cpd)
            rv_out[0, 0, 0] = rt[0, 0, 0] - rc_out[0, 0, 0] - ri_out[0, 0, 0] * prifact
            
    else:
                
        with computation(FORWARD):
        
            with interval(0, 1):
                dum4[0, 0, 0] = ri_in[0, 0, 0]
            with interval(1, None):
                dum4[0, 0, 0] = ri_in[0, 0, 0] + 0.5 * rs[0, 0, 0] + 0.25 * rg[0, 0, 0]
                
        with computation(PARALLEL), interval(...):
            rc_out[0, 0, 0] = (1 - frac_tmp[0, 0]) * cond_tmp[0, 0]
            
            lwinc = rc_out[0, 0, 0] - rc_in[0, 0, 0]  
            if abs(lwinc) > 1e-12 and esatw(t[0, 0, 0]) < 0.5 * pabs[0, 0, 0]:
                rcold = rc_out[0, 0, 0]
                rfrac = rv_in[0, 0, 0] - lwinc
                
                rsdif = min(0, rsp - rfrac) if rc_in[0, 0, 0] < rsw else 0
                # sub saturation over water (True) or supersaturation over water (False) 

                rc_out[0, 0, 0] = cond_tmp[0, 0] - rsdif
            
            else:
                rcold = rc_in[0, 0, 0]
        
            # Compute separate ice cloud 
            wcldfr[0, 0, 0] = cldfr[0, 0, 0]
            
            dum1 = min(1, 20 * rc_out[0, 0, 0] * sqrt(dz[0, 0]) / qsi[0, 0])    # cloud liquid factor
            dum3 = max(0, icldfr[0, 0, 0] - wcldfr[0, 0, 0])   
            dum4[0, 0, 0] = max(0, min(1, ice_cld_wgt[0, 0] * dum4[0, 0, 0] * sqrt(dz[0, 0]) / qsi[0, 0]))
            dum2 = (0.8 * cldfr[0, 0, 0] + 0.2) * min(1, dum1 + dum4 * cldfr[0, 0, 0])
            cldfr[0, 0, 0] = min(1, dum2 + (0.5 + 0.5 * dum3) * dum4)
                        
            ri_out[0, 0, 0] = ri_in[0, 0, 0]
            
            # TODO : same as 341 with rcold instead of ri_in
            t[0, 0, 0] = update_temperature(t, rcold, rc_out, ri_in, ri_out, lv, ls, cpd)
                  
            rv_out[0, 0, 0] = rt[0, 0, 0] - rc_out[0, 0, 0] - ri_out[0, 0, 0] * prifact
                    
    if hlambda3 == "CB":
        with computation(PARALLEL), interval(...):          
            sigrc[0, 0, 0] *= min(3, max(1, 1 - q1[0, 0]))
            

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
                
    
                
