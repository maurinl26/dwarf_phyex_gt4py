from __future__ import annotations
from typing import Dict, Optional, Tuple

from gt4py.cartesian.gtscript import IJ, K, Field
from gt4py.cartesian.gtscript import stencil

from phyex_gt4py.config import backend, dtype_float, dtype_int
from phyex_gt4py.constants import Constants
from phyex_gt4py.functions.ice_adjust import latent_heat
from phyex_gt4py.nebn import Neb
from phyex_gt4py.rain_ice_param import ParamIce, RainIceParam
from phyex_gt4py.stencils.condensation import condensation


@stencil(backend=backend)
def ice_adjust(
    cst: Constants,
    parami: ParamIce,
    icep: RainIceParam,
    neb: Neb,
    compute_srcs: bool,
    itermax: dtype_int,
    tstep: dtype_float,  # Double timestep
    nrr: dtype_int,  # Number of moist variables
    lmfconv: bool,  # size (mfconv) != 0
    buname: str,  # TODO : implement budget storage methods
    # IN - Inputs
    sigqsat: Field[dtype_float],  # coeff applied to qsat variance
    rhodj: Field[dtype_float],  # density x jacobian
    exnref: Field[dtype_float],  # ref exner pression
    rhodref: Field[dtype_float],  #
    pabs: Field[dtype_float],  # absolute pressure at t
    sigs: Field[dtype_float],  # Sigma_s at time t
    mfconv: Field[dtype_float],
    cf_mf: Field[dtype_float],  # convective mass flux fraction
    rc_mf: Field[dtype_float],  # convective mass flux liquid mixing ratio
    ri_mf: Field[dtype_float],  # convective mass flux ice mixing ratio
    # INOUT - Tendencies
    th: Field[dtype_float],
    rv: Field[dtype_float],  # water vapour m.r. to adjust
    rc: Field[dtype_float],  # cloud water m.r. to adjust
    ri: Field[dtype_float],  # cloud ice m.r. to adjust
    rr: Field[dtype_float],  # rain water m.r. to adjust
    rs: Field[dtype_float],  # aggregate m.r. to adjust
    rg: Field[dtype_float],  # graupel m.r. to adjust
    rh: Optional[Field[dtype_float]],  # hail m.r. to adjust (if krr = 7)
    ths: Field[dtype_float],  # theta source
    rvs: Field[dtype_float],  # water vapour m.r. source
    rcs: Field[dtype_float],  # cloud water m.r. source
    ris: Field[dtype_float],  # cloud ice m.r. at t+1
    cldfr: Field[dtype_float],
    srcs: Field[dtype_float],  # second order flux s at time t+1
    # OUT - Diagnostics
    icldfr: Field[dtype_float],  # ice cloud fraction
    wcldfr: Field[dtype_float],  # water or mixed-phase cloud fraction
    ifr: Field[dtype_float],  # ratio cloud ice moist part to dry part
    ssio: Field[
        dtype_float
    ],  # super-saturation with respect to ice in the super saturated fraction
    ssiu: Field[
        dtype_float
    ],  # sub-saturation with respect to ice in the subsaturated fraction
    hlc_hrc: Field[dtype_float],
    hlc_hcf: Field[dtype_float],
    hli_hri: Field[dtype_float],
    hli_hcf: Field[dtype_float],
    # TODO : rework, remove unused fields
    th_out: Field[dtype_float],  # theta out
    rv_out: Field[dtype_float],  # water vapour m.r. source
    rc_out: Field[dtype_float],  # cloud water m.r. source
    ri_out: Field[dtype_float],  # cloud ice m.r. source
    # Temporary fields
    rv_tmp: Field[dtype_float],
    ri_tmp: Field[dtype_float],
    rc_tmp: Field[dtype_float],
    t_tmp: Field[dtype_float],
    cph: Field[dtype_float],  # guess of the CPh for the mixing
    lv: Field[dtype_float],  # guess of the Lv at t+1
    ls: Field[dtype_float],  # guess of the Ls at t+1
    criaut: Field[dtype_float],  # autoconversion thresholds
    # Temporary fields # Condensation
    cpd: Field[dtype_float],
    rt: Field[dtype_float],  # work array for total water mixing ratio
    pv: Field[dtype_float],  # thermodynamics
    piv: Field[dtype_float],  # thermodynamics
    qsl: Field[dtype_float],  # thermodynamics
    qsi: Field[dtype_float],
    frac_tmp: Field[IJ, dtype_float],  # ice fraction
    cond_tmp: Field[IJ, dtype_float],  # condensate
    a: Field[IJ, dtype_float],  # related to computation of Sig_s
    sbar: Field[IJ, dtype_float],
    sigma: Field[IJ, dtype_float],
    q1: Field[IJ, dtype_float],
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
        icldfr (Field[dtype_float]): _description_
        hlc_hcf (Field[dtype_float]): _description_
        hli_hri (Field[dtype_float]): _description_
        hli_hcf (Field[dtype_float]): _description_
        rv_tmp (Field[dtype_float]): _description_
        ri_tmp (Field[dtype_float]): _description_
        rc_tmp (Field[dtype_float]): _description_
        sigqsat_tmp (Field[dtype_float]): _description_
        cph (Field[dtype_float]): _description_
    """
    
    # GT4Py compatibility
    lvtt = cst.lvtt
    lstt = cst.lstt
    cpv = cst.cpv
    tt = cst.tt
    Cl = cst.Cl
    Ci = cst.Ci
    condensation_constants = {
        "lvtt":lvtt,
        "lstt":lstt,
        "cpv":cpv,
        "tt":tt,
        "Cl":Cl,
        "Ci":Ci,
        "Rd":cst.Rd,
        "Rv":cst.Rv,
        "alpw": cst.alpw,
        "betaw":cst.betaw,
        "gamw":cst.gamw,
        "alpi":cst.alpi,
        "betai":cst.betai,
        "gami":cst.gami,
    }
    
    neb_parameters = {
        "frac_ice_adjust": neb.frac_ice_adjust,
        "tmaxmix": neb.tmaxmix,
        "tminmix": neb.tminmix
    }
    
    # 2.3 Compute the variation of mixing ratio
    with computation(PARALLEL), interval(...):
        t_tmp = th[0, 0, 0] * exn[0, 0, 0]
        lv, ls = latent_heat(
                lvtt,
                lstt,
                cpv,
                tt, t_tmp)

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
        krr=nrr,
        lmfconv=lmfconv,
        pabs=pabs,
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
        hlc_hcf=hlc_hcf,  # cloud fraction
        hli_hri=hli_hri,
        hli_hcf=hli_hcf,
        # Temporary fields - Condensation
        cpd=cpd,
        rt=rt,  # work array for total water mixing ratio
        pv=pv,  # thermodynamics
        piv=piv,  # thermodynamics
        qsl=qsl,  # thermodynamics
        qsi=qsi,
        frac_tmp=frac_tmp,  # ice fraction
        cond_tmp=cond_tmp,  # condensate
        a=a,  # related to computation of Sig_s
        sbar=sbar,
        sigma=sigma,
        q1=q1,
        condensation_constants=condensation_constants,
        neb_parameters=neb_parameters
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
            krr=nrr,
            lmfconv=lmfconv,
            pabs=pabs,
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
            hlc_hcf=hlc_hcf,  # cloud fraction
            hli_hri=hli_hri,
            hli_hcf=hli_hcf,
            # Temporary fields - Condensation
            cpd=cpd,
            rt=rt,  # work array for total water mixing ratio
            pv=pv,  # thermodynamics
            piv=piv,  # thermodynamics
            qsl=qsl,  # thermodynamics
            qsi=qsi,
            frac_tmp=frac_tmp,  # ice fraction
            cond_tmp=cond_tmp,  # condensate
            a=a,  # related to computation of Sig_s
            sbar=sbar,
            sigma=sigma,
            q1=q1,
            condensation_constants=condensation_constants,
            neb_parameters=neb_parameters
        )

    ##### 5.     COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION #####
    with computation(PARALLEL), interval(...):
        # 5.0 compute the variation of mixing ratio
        w1 = (rc_tmp[0, 0, 0] - rc[0, 0, 0]) / tstep
        w2 = (ri_tmp[0, 0, 0] - ri[0, 0, 0]) / tstep

        # 5.1 compute the sources
        w1 = max(w1, -rcs[0, 0, 0]) if w1 > 0 else min(w1, rvs[0, 0, 0])
        rvs[0, 0, 0] -= w1
        rc_tmp[0, 0, 0] += w1
        ths[0, 0, 0] += w1 * lv[0, 0, 0] / (cph[0, 0, 0] * exnref[0, 0, 0])

        w2 = max(w2, -ris[0, 0, 0]) if w1 > 0 else min(w2, rvs[0, 0, 0])

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
            rvs[0, 0, 0] -= w1 + w2
            ths[0, 0, 0] += (
                (w1 * lv[0, 0, 0] + w2 * ls[0, 0, 0]) / cph[0, 0, 0] / exnref[0, 0, 0]
            )

            if hlc_hrc is not None and hlc_hcf is not None:
                criaut = icep.criautc / rhodref[0, 0, 0]
                
                if parami.subg_mf_pdf == "NONE":
                    if w1 * tstep > cf_mf[0, 0, 0] * criaut:
                        hlc_hrc += w1 * tstep
                        hlc_hcf = min(1, hlc_hcf[0, 0, 0] + cf_mf[0, 0, 0])

                elif parami.subg_mf_pdf == "TRIANGLE":
                    if w1 * tstep > cf_mf[0, 0, 0] * criaut:
                        hcf = 1 - 0.5 * (criaut * cf_mf[0, 0, 0]) / max(1e-20, w1 * tstep)
                        hr = w1 * tstep - (criaut * cf_mf[0, 0, 0]) ** 3 / (
                            3 * max(1e-20, w1 * tstep)
                        )

                    elif 2 * w1 * tstep <= cf_mf[0, 0, 0] * criaut:
                        hcf = 0
                        hr = 0

                    else:
                        hcf = (2 * w1 * tstep - criaut * cf_mf[0, 0, 0]) ** 2 / (
                            2.0 * max(1.0e-20, w1 * tstep) ** 2
                        )
                        hr = (
                            4.0 * (w1 * tstep) ** 3
                            - 3.0 * w1 * tstep * (criaut * cf_mf[0, 0, 0]) ** 2
                            + (criaut * cf_mf[0, 0, 0]) ** 3
                        ) / (3 * max(1.0e-20, w1 * tstep) ** 2)

                    hcf *= cf_mf[0, 0, 0]
                    hlc_hcf = min(1, hlc_hcf + hcf)
                    hlc_hrc += hr

            if hli_hri is not None and hli_hcf is not None:
                criaut = min(
                    icep.criauti,
                    10 ** (icep.acriauti * (t_tmp[0, 0, 0] - cst.tt) + icep.bcriauti),
                )
                
                if parami.subg_mf_pdf == "NONE":
                    if w2 * tstep > cf_mf[0, 0, 0] * criaut:
                        hli_hri += w2 * tstep
                        hli_hcf = min(1, hli_hcf[0, 0, 0] + cf_mf[0, 0, 0])

                elif parami.subg_mf_pdf == "TRIANGLE":
                    if w2 * tstep > cf_mf[0, 0, 0] * criaut:
                        hcf = 1 - 0.5 * (criaut * cf_mf[0, 0, 0]) / max(1e-20, w2 * tstep)
                        hr = w2 * tstep - (criaut * cf_mf[0, 0, 0]) ** 3 / (
                            3 * max(1e-20, w2 * tstep)
                        )

                    elif 2 * w2 * tstep <= cf_mf[0, 0, 0] * criaut:
                        hcf = 0
                        hr = 0

                    else:
                        hcf = (2 * w2 * tstep - criaut * cf_mf[0, 0, 0]) ** 2 / (
                            2.0 * max(1.0e-20, w2 * tstep) ** 2
                        )
                        hr = (
                            4.0 * (w2 * tstep) ** 3
                            - 3.0 * w2 * tstep * (criaut * cf_mf[0, 0, 0]) ** 2
                            + (criaut * cf_mf[0, 0, 0]) ** 3
                        ) / (3 * max(1.0e-20, w2 * tstep) ** 2)

                    hcf *= cf_mf[0, 0, 0]
                    hli_hcf = min(1, hli_hcf + hcf)
                    hli_hri += hr

        if (
            rv_out is not None
            or rc_out is not None
            or ri_out is not None
            or th is not None
        ):
            w1 = rc_mf
            w2 = ri_mf

            if w1 + w2 > rv_out[0, 0, 0]:
                w1 *= rv_tmp / (w1 + w2)
                w2 = rv_tmp - w1

            rc_tmp[0, 0, 0] += w1
            ri_tmp[0, 0, 0] += w2
            rv_tmp[0, 0, 0] -= w1 + w2
            t_tmp += (w1 * lv + w2 * ls) / cph

            # TODO :  remove unused out variables
            rv_out[0, 0, 0] = rv_tmp[0, 0, 0]
            ri_out[0, 0, 0] = ri_tmp[0, 0, 0]
            rc_out[0, 0, 0] = rc_tmp[0, 0, 0]
            th_out[0, 0, 0] = t_tmp[0, 0, 0] / exn[0, 0, 0]



@stencil(backend=backend)
def iteration(
    cst: Constants,
    neb: Neb,
    krr: dtype_int,
    lmfconv: bool,
    pabs: Field[dtype_float],
    zz: Field[dtype_float],
    rhodref: Field[dtype_float],
    t_tmp: Field[dtype_float],
    lv: Field[dtype_float],
    ls: Field[dtype_float],
    rv_in: Field[dtype_float],
    ri_in: Field[dtype_float],
    rc_in: Field[dtype_float],
    rv_out: Field[dtype_float],
    rc_out: Field[dtype_float],
    ri_out: Field[dtype_float],
    rr: Field[dtype_float],
    rs: Field[dtype_float],
    rg: Field[dtype_float],
    rh: Field[dtype_float],
    sigs: Field[dtype_float],
    mfconv: Field[dtype_float],
    cldfr: Field[dtype_float],
    sigqsat: Field[dtype_float],
    srcs: Field[dtype_float],
    icldfr: Field[dtype_float],
    wcldfr: Field[dtype_float],
    ssio: Field[dtype_float],
    ssiu: Field[dtype_float],
    ifr: Field[dtype_float],
    hlc_hrc: Field[dtype_float],
    hlc_hcf: Field[dtype_float],
    hli_hri: Field[dtype_float],
    hli_hcf: Field[dtype_float],
    ice_cld_wgt: Field[dtype_float],
    # For condensation
    cpd: Field[dtype_float],
    rt: Field[dtype_float],  # work array for total water mixing ratio
    pv: Field[dtype_float],  # thermodynamics
    piv: Field[dtype_float],  # thermodynamics
    qsl: Field[dtype_float],  # thermodynamics
    qsi: Field[dtype_float],
    frac_tmp: Field[IJ, dtype_float],  # ice fraction
    cond_tmp: Field[IJ, dtype_float],  # condensate
    a: Field[IJ, dtype_float],  # related to computation of Sig_s
    sbar: Field[IJ, dtype_float],
    sigma: Field[IJ, dtype_float],
    q1: Field[IJ, dtype_float],
    
    #parameters
    condensation_constants: Dict,
    neb_parameters: Dict

):
    # 2.4 specific heat for moist air at t+1
    with computation(PARALLEL), interval(...):
        if krr == 7:
            cph = (
                cst.cpd
                + cst.cpv * rv_in
                + cst.Cl * (rc_in + rr)
                + cst.Ci * (ri_in + rs + rg + rh)
            )

        if krr == 6:
            cph = (
                cst.cpd
                + cst.cpv * rv_in
                + cst.Cl * (rc_in + rr)
                + cst.Ci * (ri_in + rs + rg)
            )
        if krr == 5:
            cph = (
                cst.cpd
                + cst.cpv * rv_in
                + cst.Cl * (rc_in + rr)
                + cst.Ci * (ri_in + rs)
            )
        if krr == 4:
            cph = cst.cpd + cst.cpv * rv_in + cst.Cl * (rc_in + rr)
        if krr == 2:
            cph = cst.cpd + cst.cpv * rv_in + cst.Cl * rc_in + cst.Ci * ri_in

    # 3. subgrid condensation scheme
    if neb.subg_cond:
        condensation(
            pabs=pabs,
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
            cldfr=cldfr,
            sigrc=srcs,
            ls=ls,
            lv=lv,
            ifr=ifr,
            sigqsat=sigqsat,
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
            qsi=qsi,
            frac_tmp=frac_tmp,  # ice fraction
            cond_tmp=cond_tmp,  # condensate
            a=a,  # related to computation of Sig_s
            sbar=sbar,
            sigma=sigma,
            q1=q1,
            **condensation_constants,
            **neb_parameters
        )

    # 3. subgrid condensation scheme
    else:
        # initialization
        with computation(PARALLEL), interval(...):
            sigs[0, 0] = 0
            sigqsat[0, 0, 0] = 0

        with computation(PARALLEL), interval(...):
            condensation(
                pabs=pabs,
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
                cldfr=cldfr,
                sigrc=srcs,
                ls=ls,
                lv=lv,
                ifr=ifr,
                sigqsat=sigqsat,
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
                qsi=qsi,
                frac_tmp=frac_tmp,  # ice fraction
                cond_tmp=cond_tmp,  # condensate
                a=a,  # related to computation of Sig_s
                sbar=sbar,
                sigma=sigma,
                q1=q1,
                **condensation_constants,
                **neb_parameters
            )

