from typing import Tuple
from config import dtype, dtype_int, backend

import gt4py.cartesian.gtscript as gtscript
from phyex_gt4py.constants import Constants
from phyex_gt4py.functions.ice_adjust import latent_heat
from phyex_gt4py.nebn import Neb

from phyex_gt4py.dimphyex import DIMPhyex
from phyex_gt4py.rain_ice_param import ParamIce, RainIceDescr, RainIceParam


@gtscript.stencil(backend=backend)
def ice_adjust(
    D: DIMPhyex,
    cst: Constants,
    parami: ParamIce,
    icep: RainIceParam,
    iced: RainIceDescr,
    neb: Neb,
    compute_srcs: bool,
    itermax: dtype_int, 
    tstep: dtype,
    rhodref: gtscript.Field[dtype],
    
    th: gtscript.Field[dtype],
    th_out: gtscript.Field[dtype],
    exn: gtscript.Field[dtype],
    rv_in: gtscript.Field[dtype],
    rc_in: gtscript.Field[dtype],
    ri_in: gtscript.Field[dtype],
    rv_out: gtscript.Field[dtype],
    rc_out: gtscript.Field[dtype],
    ri_out: gtscript.Field[dtype],
    cph: gtscript.Field[dtype],
    exnref: gtscript.Field[dtype],
    ris: gtscript.Field[dtype],
    cldfr: gtscript.Field[dtype],
    srcs: gtscript.Field[dtype],
    rc_mf: gtscript.Field[dtype],
    cf_mf: gtscript.Field[dtype],
    ri_mf: gtscript.Field[dtype],
    
    hlc_hrc: gtscript.Field[dtype],
    hlc_hcf: gtscript.Field[dtype],
    hli_hri: gtscript.Field[dtype],
    hli_hcf: gtscript.Field[dtype],
    
    # Temporary fields 
    rv_tmp: gtscript.Field[dtype],
    ri_tmp: gtscript.Field[dtype],
    rc_tmp: gtscript.Field[dtype],

):    

    # 2.3 Compute the variation of mixing ratio
    with computation(PARALLEL), interval(...):
        t = th[0, 0, 0] * exn[0, 0, 0]
        lv, ls = latent_heat(cst, t)
        
    # jiter  = 0
    rv_tmp, rc_tmp, ri_tmp = iteration(rv_in, rc_in, ri_in, rv_tmp, rc_tmp, ri_tmp)
              
    # jiter > 0 
    for jiter in range(1, itermax):
        backup(rv_tmp, rc_tmp, ri_tmp)
        iteration(rv_tmp, rc_tmp, ri_tmp, rv_out, rc_out, ri_out) 
        
    ##### 5.     COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION #####
    with computation(PARALLEL), interval(...):
        
        # 5.0 compute the variation of mixing ratio
        w1 = (rc_out[0, 0, 0] - rc_in[0, 0, 0]) / tstep
        w2 = (ri_out[0, 0, 0] - ri_in[0, 0, 0]) / tstep

        # 5.1 compute the sources
        w1 = max(w1, - rcs[0, 0, 0]) if w1 > 0 else min(w1, rvs[0, 0, 0])
        rvs -= w1
        rcs += w1
        ths += w1 * lv[0, 0, 0] / (cph[0, 0, 0] * exnref[0, 0, 0])
        
        w2 = max(w2, - ris[0, 0, 0]) if w1 > 0 else min(w2, ris[0, 0, 0])
        
        if not neb.subg_cond:
        
            
            cldfr[0, 0, 0] = 1 if rcs[0, 0, 0] + ris[0, 0, 0] > 1e-12 / tstep else 0
            if compute_srcs:
                srcs[0, 0, 0] = cldfr[0, 0, 0]
                         
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
                criaut = min(icep.criauti, 10**(icep.acriauti * (t[0, 0, 0] - cst.tt) + icep.bcriauti))
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
                w1 *= rv_out / (w1 + w2)
                w2 = rv_out - w1
            
            rc_out[0, 0, 0] += w1
            ri_out[0, 0, 0] += w2
            rv_out[0, 0, 0] -= (w1 + w2)
            t += (w1 * lv + w2 * ls) /cph
            
@gtscript.function()
def backup(
    rv_tmp: gtscript.Field,
    ri_tmp: gtscript.Field,
    rc_tmp: gtscript.Field,
    rv_out: gtscript.Field,
    ri_out: gtscript.Field,
    rc_out: gtscript.Field
) -> Tuple[gtscript.Field]:
    """
    Dump out fields into temporary fields to 
    perform loop iterations

    Args:
        rv_tmp (gtscript.Field): vapour mixing ratio (temp field)
        ri_tmp (gtscript.Field): ice mixing ratio (temp field)
        rc_tmp (gtscript.Field): cloud mixing ratio (temp field)
        rv_out (gtscript.Field): vapour mixing ratio (out field)
        ri_out (gtscript.Field): ice mixing ratio (out field)
        rc_out (gtscript.Field): cloud mixing ratio (out field)

    Returns:
        Tuple[gtscript.Field]: temporary fields
    """
    
    rv_tmp = rv_out[0, 0, 0]
    ri_tmp = ri_out[0, 0, 0]
    rc_tmp = rc_out[0, 0, 0]
    
    return rv_tmp, ri_tmp, rc_tmp
                            
            
@gtscript.function
def subgrid_mf(
    criaut: gtscript.Field[dtype],
    subg_mf_pdf: gtscript.Field[dtype],
    hl_hr: gtscript.Field[dtype],
    hl_hc: gtscript.Field[dtype],
    cf_mf: gtscript.Field[dtype],
    w: gtscript.Field[dtype],
    tstep: dtype,     
) -> Tuple[gtscript.Field]:   
    """_summary_

    Args:
        criaut (gtscript.Field[dtype]): _description_
        subg_mf_pdf (gtscript.Field[dtype]): _description_
        hl_hr (gtscript.Field[dtype]): _description_
        hl_hc (gtscript.Field[dtype]): _description_
        cf_mf (gtscript.Field[dtype]): _description_
        w (gtscript.Field[dtype]): _description_
        tstep (dtype): time step

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
    pabs: gtscript.Field[dtype],
    zz: gtscript.Field[dtype],
    rhodref: gtscript.Field[dtype],
    t: gtscript.Field[dtype],
    rv_in: gtscript.Field[dtype],
    ri_in: gtscript.Field[dtype],
    rc_in: gtscript.Field[dtype],
    rv_out: gtscript.Field[dtype],
    rc_out: gtscript.Field[dtype],
    ri_out: gtscript.Field[dtype],
    rr: gtscript.Field[dtype],
    rs: gtscript.Field[dtype],
    rg: gtscript.Field[dtype],
    rh: gtscript.Field[dtype],
    sigs: gtscript.Field[dtype],
    lmfconv: bool,
    mfconv: gtscript.Field[dtype],
    cldfr: gtscript.Field[dtype],
    
    sigqsat: gtscript.Field[dtype],
    krr: dtype_int
    
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
            cst,
            icep,
            neb,
            pabst,
            zz,
            rhodref,
            t,
            rv_in,
            rv_out,
            rc_in,
            rc_out,
            ri_in,
            ri_out,
            rr,
            rs, 
            rg, 
            sigs,
            lmfconv,
            mfconv,
            cldfr,
            srcs,
            True,   # ouseri
            icldfr,
            wcldfr,
            ssio,
            ssiu,
            ifr,
            sigqsat, 
            lv,
            ls,
            cph,
            hlc_hrc,
            hlc_hcf
            hli_hrc,
            hli_hcf,
            ice_cld_wgt  
        )
    
    # 3. subgrid condensation scheme
    else:
        # initialization
        with computation(PARALLEL), interval(...):
            sigs[0, 0] = 0
            sigqsat[0, 0, 0] = 0
            
        with computation(PARALLEL), interval(...):
            condensation(
            cst,
            icep,
            neb,
            pabst,
            zz,
            rhodref,
            t,
            rv_in,
            rv_out,
            rc_in,
            rc_out,
            ri_in,
            ri_out,
            rr,
            rs, 
            rg, 
            sigs,
            lmfconv,
            mfconv,
            cldfr,
            srcs,
            True,   # ouseri
            icldfr,
            wcldfr,
            ssio,
            ssiu,
            ifr,
            sigqsat, 
            lv,
            ls,
            cph,
            hlc_hrc,
            hlc_hcf
            hli_hrc,
            hli_hcf,
            ice_cld_wgt  
        )
            

