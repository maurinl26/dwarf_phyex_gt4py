from typing import Tuple

from gt4py.cartesian import gtscript, IJ
from config import backend, dtype_float, dtype_int

from phyex_gt4py.constants import Constants
from phyex_gt4py.dimphyex import Phyex
from phyex_gt4py.functions.ice_adjust import latent_heat
from phyex_gt4py.nebn import Neb
from phyex_gt4py.rain_ice_param import ParamIce, RainIceParam
from phyex_gt4py.stencils.condensation import condensation


@gtscript.stencil(backend=backend)
def ice_adjust(
    cst: Constants,
    parami: ParamIce,
    icep: RainIceParam,
    neb: Neb,
    compute_srcs: bool,
    itermax: dtype_int, 
    tstep: dtype_float,                   # Double timestep
    nrr: dtype_int,                 # Number of moist variables
    lmfconv: bool, # size (mfconv) != 0
    sigqsat: gtscript.Field[dtype_float], # coeff applied to qsat variance
    exnref: gtscript.Field[dtype_float],  # ref exner pression
    rhodref: gtscript.Field[dtype_float], #        
    sigs: gtscript.Field[dtype_float],    # Sigma_s at time t
    pabs_t: gtscript.Field[dtype_float],  # absolute pressure at t
    exn: gtscript.Field[dtype_float],     # exner function
    
    cf_mf: gtscript.Field[dtype_float],   # convective mass flux fraction
    rc_mf: gtscript.Field[dtype_float],   # convective mass flux liquid mixing ratio
    ri_mf: gtscript.Field[dtype_float],   # convective mass flux ice mixing ratio
    
       
      
    rv: gtscript.Field[dtype_float],   # water vapour m.r. to adjust 
    rc: gtscript.Field[dtype_float],   # cloud water m.r. to adjust
    ri: gtscript.Field[dtype_float],   # cloud ice m.r. to adjust
    rvs: gtscript.Field[dtype_float],  # water vapour m.r. source
    rcs: gtscript.Field[dtype_float],  # cloud water m.r. source
    ris: gtscript.Field[dtype_float],  # cloud ice m.r. at t+1
    rv_out: gtscript.Field[dtype_float],  # water vapour m.r. source
    rc_out: gtscript.Field[dtype_float],  # cloud water m.r. source
    ri_out: gtscript.Field[dtype_float],  # cloud ice m.r. source
    th: gtscript.Field[dtype_float],   # theta to adjust
    ths: gtscript.Field[dtype_float],  # theta source
    th_out: gtscript.Field[dtype_float], # theta out
    cldfr: gtscript.Field[dtype_float],
    srcs: gtscript.Field[dtype_float],    # second order flux s at time t+1

    # Out
    icldfr: gtscript.Field[dtype_float],          # ice cloud fraction
    wcldfr: gtscript.Field[dtype_float],          # water or mixed-phase cloud fraction
    
    ifr: gtscript.Field[dtype_float],             # ratio cloud ice moist part to dry part
    ssio: gtscript.Field[dtype_float],            # super-saturation with respect to ice in the super saturated fraction
    ssiu: gtscript.Field[dtype_float],            # sub-saturation with respect to ice in the subsaturated fraction
    
    rr: gtscript.Field[dtype_float],              # rain water m.r. to adjust
    rs: gtscript.Field[dtype_float],              # aggregate m.r. to adjust
    rg: gtscript.Field[dtype_float],              # graupel m.r. to adjust
    rh: gtscript.Field[dtype_float],              # hail m.r. to adjust
    

    hlc_hrc: gtscript.Field[dtype_float],
    hlc_hcf: gtscript.Field[dtype_float],
    hli_hri: gtscript.Field[dtype_float],
    hli_hcf: gtscript.Field[dtype_float],
    
    # Temporary fields 
    rv_tmp: gtscript.Field[dtype_float],
    ri_tmp: gtscript.Field[dtype_float],
    rc_tmp: gtscript.Field[dtype_float],
    t_tmp: gtscript.Field[dtype_float],
    cph: gtscript.Field[dtype_float],             # guess of the CPh for the mixing
    lv: gtscript.Field[dtype_float],              # guess of the Lv at t+1
    ls: gtscript.Field[dtype_float],              # guess of the Ls at t+1
    
    criaut: gtscript.Field[dtype_float],          # autoconversion thresholds
    
    # Temporary fields # Condensation
    cpd: gtscript.Field[dtype_float],
    rt: gtscript.Field[dtype_float],  # work array for total water mixing ratio
    pv: gtscript.Field[dtype_float],  # thermodynamics
    piv: gtscript.Field[dtype_float],  # thermodynamics
    qsl: gtscript.Field[dtype_float],  # thermodynamics
    qsi: gtscript.Field[dtype_float],
    frac_tmp: gtscript.Field[IJ, dtype_float],  # ice fraction
    cond_tmp: gtscript.Field[IJ, dtype_float],  # condensate
    a: gtscript.Field[IJ, dtype_float],  # related to computation of Sig_s
    sbar: gtscript.Field[IJ, dtype_float],
    sigma: gtscript.Field[IJ, dtype_float],
    q1: gtscript.Field[IJ, dtype_float],          
):   
    """_summary_

    Args:
        cst (Constants): physical constants
        parami (ParamIce): mixed phase cloud parameters
        icep (RainIceParam): microphysical factors used in the warm and cold schemes
        neb (Neb): constants for nebulosity calculations
        compute_srcs (bool): boolean to compute second order flux
        itermax (dtype_int): _description_
        tstep (dtype_float): double time step
        krr (int)
        icldfr (gtscript.Field[dtype_float]): _description_
        hlc_hcf (gtscript.Field[dtype_float]): _description_
        hli_hri (gtscript.Field[dtype_float]): _description_
        hli_hcf (gtscript.Field[dtype_float]): _description_
        rv_tmp (gtscript.Field[dtype_float]): _description_
        ri_tmp (gtscript.Field[dtype_float]): _description_
        rc_tmp (gtscript.Field[dtype_float]): _description_
        sigqsat_tmp (gtscript.Field[dtype_float]): _description_
        cph (gtscript.Field[dtype_float]): _description_
    """
     

    # 2.3 Compute the variation of mixing ratio
    with computation(PARALLEL), interval(...):
        t_tmp = th[0, 0, 0] * exn[0, 0, 0]
        lv, ls = latent_heat(cst, t_tmp)
        
    # jiter  = 0
    rv_tmp, rc_tmp, ri_tmp = iteration(
        rv_in=rv, 
        rc_in=rc, 
        ri_in=ri, 
        rv_out=rv_tmp, 
        rc_out=rc_tmp, 
        ri_out=ri_tmp,
        
        # 
        cst=cst,
        neb=neb,
        icep=icep,
        parami=parami,
        krr=nrr,
        lmfconv=lmfconv,
        pabs=pabs_t,
        t_tmp=t_tmp,
        lv=lv,
        ls=ls,
        rr=rr,
        rs=rs,
        rg=rg,
        rh=rh,
        sigs=sigs,
        cldfr=cldfr,
        srcs=srcs,
        sigqsat=sigqsat,
        cph=cph,
        ifr=ifr,
        
        # super-saturation with respect to in in the sub saturated fraction
        hlc_hrc=hlc_hrc,
        hlc_hcf=hlc_hcf, # cloud fraction
        hli_hri=hli_hri, 
        hli_hcf=hli_hcf, 
        
        # Temporary fields - Condensation
        cpd=cpd,       
        rt=rt,        # work array for total water mixing ratio
        pv=pv,        # thermodynamics
        piv=piv,       # thermodynamics
        qsl=qsl,       # thermodynamics
        qsi=qsi,       
        frac_tmp=frac_tmp,  # ice fraction
        cond_tmp=cond_tmp,  # condensate
        a=a,         # related to computation of Sig_s
        sbar=sbar,        
        sigma=sigma,        
        q1=q1,        
        )
              
    # jiter > 0 
    for jiter in range(1, itermax):
        # backup(rv_tmp, rc_tmp, ri_tmp)
        iteration(
            rv_in=rv_tmp, 
            rc_in=rc_tmp, 
            ri_in=ri_tmp, 
            rv_out=rv_tmp, 
            rc_out=rc_tmp, 
            ri_out=ri_tmp,
            
            # 
            cst=cst,
            neb=neb,
            icep=icep,
            parami=parami,
            krr=nrr,
            lmfconv=lmfconv,
            pabs=pabs_t,
            t_tmp=t_tmp,
            lv=lv,
            ls=ls,
            rr=rr,
            rs=rs,
            rg=rg,
            rh=rh,
            sigs=sigs,
            cldfr=cldfr,
            srcs=srcs,
            sigqsat=sigqsat,
            cph=cph,
            ifr=ifr,
            
            # super-saturation with respect to in in the sub saturated fraction
            hlc_hrc=hlc_hrc,
            hlc_hcf=hlc_hcf, # cloud fraction
            hli_hri=hli_hri, 
            hli_hcf=hli_hcf, 
            
            # Temporary fields - Condensation
            cpd=cpd,       
            rt=rt,        # work array for total water mixing ratio
            pv=pv,        # thermodynamics
            piv=piv,       # thermodynamics
            qsl=qsl,       # thermodynamics
            qsi=qsi,       
            frac_tmp=frac_tmp,  # ice fraction
            cond_tmp=cond_tmp,  # condensate
            a=a,         # related to computation of Sig_s
            sbar=sbar,        
            sigma=sigma,        
            q1=q1,   
        ) 
        
    ##### 5.     COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION #####
    with computation(PARALLEL), interval(...):
        
        # 5.0 compute the variation of mixing ratio
        w1 = (rc_tmp[0, 0, 0] - rc[0, 0, 0]) / tstep
        w2 = (ri_tmp[0, 0, 0] - ri[0, 0, 0]) / tstep

        # 5.1 compute the sources
        w1 = max(w1, - rcs[0, 0, 0]) if w1 > 0 else min(w1, rvs[0, 0, 0])
        rvs[0, 0, 0] -= w1
        rc_tmp[0, 0, 0] += w1
        ths[0, 0, 0] += w1 * lv[0, 0, 0] / (cph[0, 0, 0] * exnref[0, 0, 0])
        
        w2 = max(w2, - ris[0, 0, 0]) if w1 > 0 else min(w2, rvs[0, 0, 0])
        
        if not neb.subg_cond:
            cldfr[0, 0, 0] = 1 if rcs[0, 0, 0] + ris[0, 0, 0] > 1e-12 / tstep else 0
            srcs[0, 0, 0] = cldfr[0, 0, 0] if compute_srcs else None
                         
        else: 
        
            w1 = rc_mf[0, 0, 0] / tstep
            w2 = ri_mf[0, 0, 0] / tstep
            
            if w1 + w2 > rvs[0, 0, 0]:
                w1 *= rvs[0, 0, 0] / (w1 + w2)
                w2 = rvs[0, 0, 0] - w1
            
            
            cldfr[0, 0, 0] = min(1, cldfr[0, 0, 0] + cf_mf[0, 0, 0])
            rcs[0, 0, 0] += w1
            ris[0, 0, 0] += w2
            rvs[0, 0, 0] -= (w1 + w2)
            ths[0, 0, 0] += (w1 * lv[0, 0, 0] + w2 * ls[0, 0, 0]) / cph[0, 0, 0] / exnref[0, 0, 0]
            
            if hlc_hrc is not None and hlc_hcf is not None:
                criaut = icep.criautc / rhodref[0, 0, 0]
                hlc_hrc, hlc_hcf, w1 = subgrid_mf(
                    criaut, 
                    parami.subg_mf_pdf,
                    hlc_hrc,
                    hlc_hcf,
                    cf_mf,
                    w1, 
                    tstep
                )
                    
            if hli_hri is not None and hli_hcf is not None:
                criaut = min(icep.criauti, 10**(icep.acriauti * (t_tmp[0, 0, 0] - cst.tt) + icep.bcriauti))
                hli_hri, hli_hcf, w2 = subgrid_mf(
                    criaut, 
                    parami.subg_mf_pdf,
                    hli_hri,
                    hli_hcf,
                    cf_mf,
                    w2, 
                    tstep
                )
                       
        if rv_out is not None or rc_out is not None or ri_out is not None or th is not None:
        
            w1 = rc_mf
            w2 = ri_mf
            
            if w1 + w2 > rv_out[0, 0, 0]:
                w1 *= rv_tmp / (w1 + w2)
                w2 = rv_tmp - w1
            
            rc_tmp[0, 0, 0] += w1
            ri_tmp[0, 0, 0] += w2
            rv_tmp[0, 0, 0] -= (w1 + w2)
            t_tmp += (w1 * lv + w2 * ls) /cph
            
            # TODO :  remove unused out variables 
            rv_out[0, 0, 0] = rv_tmp[0, 0, 0]
            ri_out[0, 0, 0] = ri_tmp[0, 0, 0]
            rc_out[0, 0, 0] = rc_tmp[0, 0, 0]
            th_out[0, 0, 0] = t_tmp[0, 0, 0] / exn[0, 0, 0]
                                           
            
@gtscript.function
def subgrid_mf(
    criaut: gtscript.Field[dtype_float],
    subg_mf_pdf: gtscript.Field[dtype_float],
    hl_hr: gtscript.Field[dtype_float],
    hl_hc: gtscript.Field[dtype_float],
    cf_mf: gtscript.Field[dtype_float],
    w: gtscript.Field[dtype_float],
    tstep: dtype_float,     
) -> Tuple[gtscript.Field]:   
    """Compute subgrid mass fluxes

    Args:
        criaut (gtscript.Field[dtype_float]): _description_
        subg_mf_pdf (gtscript.Field[dtype_float]): _description_
        hl_hr (gtscript.Field[dtype_float]): _description_
        hl_hc (gtscript.Field[dtype_float]): _description_
        cf_mf (gtscript.Field[dtype_float]): _description_
        w (gtscript.Field[dtype_float]): _description_
        tstep (dtype_float): time step

    Returns:
        _type_: _description_
    """
                
    if subg_mf_pdf == "NONE":
        if w*tstep > cf_mf[0, 0, 0] * criaut:
            hl_hr += w * tstep
            hl_hc = min(1, hl_hc[0, 0, 0] + cf_mf[0, 0, 0])
                
    elif subg_mf_pdf == "TRIANGLE":
        if w *tstep > cf_mf[0, 0, 0] * criaut:
            hcf = 1 - 0.5 * (criaut * cf_mf[0, 0, 0]) / max(1e-20, w * tstep)
            hr = w * tstep - (criaut*cf_mf[0, 0, 0])**3 / (3*max(1e-20, w * tstep))
                     
        elif 2 * w * tstep <= cf_mf[0, 0, 0] * criaut:
                        hcf = 0
                        hr = 0
                        
        else:
            hcf = (2 * w * tstep -criaut * cf_mf[0, 0, 0])**2 / (2.*max(1.e-20, w * tstep)**2)
            hr = (
                (4. *(w * tstep)**3 - 3.* w * tstep * (criaut * cf_mf[0, 0, 0])**2 
                + (criaut * cf_mf[0, 0, 0])**3)
                / (3 * max(1.e-20, w* tstep)**2)
                )
                        
        hcf *= cf_mf[0, 0, 0]
        hl_hc = min(1, hl_hc + hcf)
        hl_hr += hr
        
    return hl_hr, hl_hc, w     


@gtscript.stencil(backend=backend)            
def iteration(
    cst: Constants,
    neb: Neb,
    icep: RainIceParam,
    parami: ParamIce,
    krr: dtype_int,
    lmfconv: bool,
    pabs: gtscript.Field[dtype_float],
    zz: gtscript.Field[dtype_float],
    rhodref: gtscript.Field[dtype_float],
    t_tmp: gtscript.Field[dtype_float],
    lv: gtscript.Field[dtype_float],
    ls: gtscript.Field[dtype_float],
    rv_in: gtscript.Field[dtype_float],
    ri_in: gtscript.Field[dtype_float],
    rc_in: gtscript.Field[dtype_float],
    rv_out: gtscript.Field[dtype_float],
    rc_out: gtscript.Field[dtype_float],
    ri_out: gtscript.Field[dtype_float],
    rr: gtscript.Field[dtype_float],
    rs: gtscript.Field[dtype_float],
    rg: gtscript.Field[dtype_float],
    rh: gtscript.Field[dtype_float],
    sigs: gtscript.Field[dtype_float],
    mfconv: gtscript.Field[dtype_float],
    cldfr: gtscript.Field[dtype_float],
    
    sigqsat: gtscript.Field[dtype_float],
    srcs: gtscript.Field[dtype_float],
    icldfr: gtscript.Field[dtype_float],
    wcldfr: gtscript.Field[dtype_float],
    ssio: gtscript.Field[dtype_float],
    ssiu: gtscript.Field[dtype_float],
    ifr: gtscript.Field[dtype_float],

    hlc_hrc: gtscript.Field[dtype_float],
    hlc_hcf: gtscript.Field[dtype_float],
    hli_hri: gtscript.Field[dtype_float],
    hli_hcf: gtscript.Field[dtype_float],
    ice_cld_wgt: gtscript.Field[dtype_float],
    
    # For condensation 
    cpd: gtscript.Field[dtype_float],
    rt: gtscript.Field[dtype_float],  # work array for total water mixing ratio
    pv: gtscript.Field[dtype_float],  # thermodynamics
    piv: gtscript.Field[dtype_float],  # thermodynamics
    qsl: gtscript.Field[dtype_float],  # thermodynamics
    qsi: gtscript.Field[dtype_float],
    frac_tmp: gtscript.Field[IJ, dtype_float],  # ice fraction
    cond_tmp: gtscript.Field[IJ, dtype_float],  # condensate
    a: gtscript.Field[IJ, dtype_float],  # related to computation of Sig_s
    sbar: gtscript.Field[IJ, dtype_float],
    sigma: gtscript.Field[IJ, dtype_float],
    q1: gtscript.Field[IJ, dtype_float],
    
        
):
    
    # 2.4 specific heat for moist air at t+1
    with computation(PARALLEL), interval(...):
        if krr == 7:
            cph = (
                cst.cpd + cst.cpv * rv_in
                + cst.Cl * (rc_in + rr)
                + cst.Ci * (ri_in + rs + rg + rh)
            )
            
        if krr == 6:
            cph = (
                cst.cpd + cst.cpv * rv_in
                + cst.Cl * (rc_in + rr)
                + cst.Ci * (ri_in + rs + rg)
            )
        if krr == 5:
            cph = (
                cst.cpd + cst.cpv * rv_in
                + cst.Cl * (rc_in + rr)
                + cst.Ci * (ri_in + rs)
            )
        if krr == 4:
            cph = (
                cst.cpd + cst.cpv * rv_in
                + cst.Cl * (rc_in + rr)
            )
        if krr == 2:
            cph = (
                cst.cpd + cst.cpv * rv_in
                + cst.Cl * rc_in
                + cst.Ci * ri_in 
            )
    
    # 3. subgrid condensation scheme
    if neb.subg_cond:
        condensation(
            cst=cst,
            nebn=neb,
            icep=icep,
            parami=parami,
            lmfconv=lmfconv,
            ouseri=True,
            pabs=pabs,
            zz=zz,
            rhodref=rhodref,
            t=t_tmp,
            rv_in=rv_in,
            rv_out=rv_out,
            rc_in=rc_in,
            rc_out=rc_out,
            ri_in=ri_in,
            ri_out=ri_out,
            rr=rr,
            rs=rs, 
            rg=rg, 
            sigs=sigs,
            mfconv=mfconv,
            cldfr=cldfr,
            sigrc=srcs,
            icldfr=icldfr,
            wcldfr=wcldfr,
            ls=ls,
            lv=lv,
            cph=cph,
            ifr=ifr,
            sigqsat=sigqsat,   
            ssio=ssio,
            ssiu=ssiu,
            hlc_hrc=hlc_hrc,
            hlc_hcf=hlc_hcf,
            hli_hri=hli_hri,
            hli_hcf=hli_hcf,
            ice_cld_wgt=ice_cld_wgt,
                
            # Temp fields (to initiate)
            cpd=cpd,
            rt=rt,  # work array for total water mixing ratio
            pv=pv,  # thermodynamics
            piv=piv,  # thermodynamics
            qsl=qsl,  # thermodynamics
            qsl=qsi,
            frac_tmp=frac_tmp,  # ice fraction
            cond_tmp=cond_tmp,  # condensate
            a=a,  # related to computation of Sig_s
            sbar=sbar,
            sigma=sigma,
            q1=q1,
        )
    
    # 3. subgrid condensation scheme
    else:
        # initialization
        with computation(PARALLEL), interval(...):
            sigs[0, 0] = 0
            sigqsat[0, 0, 0] = 0
            
        with computation(PARALLEL), interval(...):
            condensation(
                cst=cst,
                nebn=neb,
                icep=icep,
                parami=parami,
                lmfconv=lmfconv,
                ouseri=True,
                pabs=pabs,
                zz=zz,
                rhodref=rhodref,
                t=t_tmp,
                rv_in=rv_in,
                rv_out=rv_out,
                rc_in=rc_in,
                rc_out=rc_out,
                ri_in=ri_in,
                ri_out=ri_out,
                rr=rr,
                rs=rs, 
                rg=rg, 
                sigs=sigs,
                mfconv=mfconv,
                cldfr=cldfr,
                sigrc=srcs,
                icldfr=icldfr,
                wcldfr=wcldfr,
                ls=ls,
                lv=lv,
                cph=cph,    # zcph
                ifr=ifr,
                sigqsat=sigqsat, # zsigqsat   
                ssio=ssio,
                ssiu=ssiu,
                hlc_hrc=hlc_hrc,
                hlc_hcf=hlc_hcf,
                ice_cld_wgt=ice_cld_wgt, 
                
                # Tmp fields used in routine
                cpd=cpd,
                rt=rt,  # work array for total water mixing ratio
                pv=pv,  # thermodynamics
                piv=piv,  # thermodynamics
                qsl=qsl,  # thermodynamics
                qsl=qsi,
                frac_tmp=frac_tmp,  # ice fraction
                cond_tmp=cond_tmp,  # condensate
                a=a,  # related to computation of Sig_s
                sbar=sbar,
                sigma=sigma,
                q1=q1
            )
