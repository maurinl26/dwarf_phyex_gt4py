# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional

from gt4py.cartesian.gtscript import IJ, Field

from phyex_gt4py.phyex_common.constants import Constants
from phyex_gt4py.functions.ice_adjust import latent_heat
from phyex_gt4py.phyex_common.nebn import Neb
from phyex_gt4py.phyex_common.rain_ice_param import RainIceParam
from phyex_gt4py.phyex_common.param_ice import ParamIce
from phyex_gt4py.stencils.condensation import condensation
from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("ice_adjust")
def ice_adjust(
    cst: Constants,
    parami: ParamIce,
    icep: RainIceParam,
    neb: Neb,
    compute_srcs: bool,
    itermax: "int",
    tstep: "float",  # Double timestep
    nrr: "int",  # Number of moist variables
    lmfconv: bool,  # size (mfconv) != 0
    buname: str,  # TODO : implement budget storage methods
    # IN - Inputs
    sigqsat: Field["float"],  # coeff applied to qsat variance
    rhodj: Field["float"],  # density x jacobian
    exnref: Field["float"],  # ref exner pression
    rhodref: Field["float"],  #
    pabs: Field["float"],  # absolute pressure at t
    sigs: Field["float"],  # Sigma_s at time t
    mfconv: Field["float"],
    cf_mf: Field["float"],  # convective mass flux fraction
    rc_mf: Field["float"],  # convective mass flux liquid mixing ratio
    ri_mf: Field["float"],  # convective mass flux ice mixing ratio
    # INOUT - Tendencies
    th: Field["float"],
    rv: Field["float"],  # water vapour m.r. to adjust
    rc: Field["float"],  # cloud water m.r. to adjust
    ri: Field["float"],  # cloud ice m.r. to adjust
    rr: Field["float"],  # rain water m.r. to adjust
    rs: Field["float"],  # aggregate m.r. to adjust
    rg: Field["float"],  # graupel m.r. to adjust
    rh: Optional[Field["float"]],  # hail m.r. to adjust (if krr = 7)
    ths: Field["float"],  # theta source
    rvs: Field["float"],  # water vapour m.r. source
    rcs: Field["float"],  # cloud water m.r. source
    ris: Field["float"],  # cloud ice m.r. at t+1
    cldfr: Field["float"],
    srcs: Field["float"],  # second order flux s at time t+1
    # OUT - Diagnostics
    icldfr: Field["float"],  # ice cloud fraction
    wcldfr: Field["float"],  # water or mixed-phase cloud fraction
    ifr: Field["float"],  # ratio cloud ice moist part to dry part
    ssio: Field[
        "float"
    ],  # super-saturation with respect to ice in the super saturated fraction
    ssiu: Field[
        "float"
    ],  # sub-saturation with respect to ice in the subsaturated fraction
    hlc_hrc: Field["float"],
    hlc_hcf: Field["float"],
    hli_hri: Field["float"],
    hli_hcf: Field["float"],
    # TODO : rework, remove unused fields
    th_out: Field["float"],  # theta out
    rv_out: Field["float"],  # water vapour m.r. source
    rc_out: Field["float"],  # cloud water m.r. source
    ri_out: Field["float"],  # cloud ice m.r. source
    # Temporary fields
    rv_tmp: Field["float"],
    ri_tmp: Field["float"],
    rc_tmp: Field["float"],
    t_tmp: Field["float"],
    cph: Field["float"],  # guess of the CPh for the mixing
    lv: Field["float"],  # guess of the Lv at t+1
    ls: Field["float"],  # guess of the Ls at t+1
    criaut: Field["float"],  # autoconversion thresholds
    # TODO : set constants as externals
    # Temporary fields # Condensation
    cpd: Field["float"],
    rt: Field["float"],  # work array for total water mixing ratio
    pv: Field["float"],  # thermodynamics
    piv: Field["float"],  # thermodynamics
    qsl: Field["float"],  # thermodynamics
    qsi: Field["float"],
    frac_tmp: Field[IJ, "float"],  # ice fraction
    cond_tmp: Field[IJ, "float"],  # condensate
    a: Field[IJ, "float"],  # related to computation of Sig_s
    sbar: Field[IJ, "float"],
    sigma: Field[IJ, "float"],
    q1: Field[IJ, "float"],
):
    """_summary_

    Args:
        cst (Constants): physical constants
        parami (ParamIce): mixed phase cloud parameters
        icep (RainIceParam): microphysical factors used in the warm and cold schemes
        neb (Neb): constants for nebulosity calculations
        compute_srcs (bool): boolean to compute second order flux
        itermax ("int"): _description_
        tstep ("float"): double time step
        krr (int)
        icldfr (Field["float"]): _description_
        hlc_hcf (Field["float"]): _description_
        hli_hri (Field["float"]): _description_
        hli_hcf (Field["float"]): _description_
        rv_tmp (Field["float"]): _description_
        ri_tmp (Field["float"]): _description_
        rc_tmp (Field["float"]): _description_
        sigqsat_tmp (Field["float"]): _description_
        cph (Field["float"]): _description_
    """

    from __externals__ import (
        lvtt,
        lstt,
        cpv,
        tt,
        subg_mf_pdf,
        subg_cond,
    )

    # 2.3 Compute the variation of mixing ratio
    with computation(PARALLEL), interval(...):
        t_tmp = th[0, 0, 0] * exn[0, 0, 0]
        lv, ls = latent_heat(lvtt, lstt, cpv, tt, t_tmp)

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

        if not subg_cond:
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

                if subg_mf_pdf == "NONE":
                    if w1 * tstep > cf_mf[0, 0, 0] * criaut:
                        hlc_hrc += w1 * tstep
                        hlc_hcf = min(1, hlc_hcf[0, 0, 0] + cf_mf[0, 0, 0])

                elif subg_mf_pdf == "TRIANGLE":
                    if w1 * tstep > cf_mf[0, 0, 0] * criaut:
                        hcf = 1 - 0.5 * (criaut * cf_mf[0, 0, 0]) / max(
                            1e-20, w1 * tstep
                        )
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
                    10 ** (icep.acriauti * (t_tmp[0, 0, 0] - TT) + icep.bcriauti),
                )

                if subg_mf_pdf == "NONE":
                    if w2 * tstep > cf_mf[0, 0, 0] * criaut:
                        hli_hri += w2 * tstep
                        hli_hcf = min(1, hli_hcf[0, 0, 0] + cf_mf[0, 0, 0])

                elif subg_mf_pdf == "TRIANGLE":
                    if w2 * tstep > cf_mf[0, 0, 0] * criaut:
                        hcf = 1 - 0.5 * (criaut * cf_mf[0, 0, 0]) / max(
                            1e-20, w2 * tstep
                        )
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


@stencil_collection("ice_adjust")
def iteration(
    krr: "int",
    lmfconv: bool,
    pabs: Field["float"],
    zz: Field["float"],
    rhodref: Field["float"],
    t_tmp: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
    rv_in: Field["float"],
    ri_in: Field["float"],
    rc_in: Field["float"],
    rv_out: Field["float"],
    rc_out: Field["float"],
    ri_out: Field["float"],
    rr: Field["float"],
    rs: Field["float"],
    rg: Field["float"],
    rh: Field["float"],
    sigs: Field["float"],
    mfconv: Field["float"],
    cldfr: Field["float"],
    sigqsat: Field["float"],
    srcs: Field["float"],
    icldfr: Field["float"],
    wcldfr: Field["float"],
    ssio: Field["float"],
    ssiu: Field["float"],
    ifr: Field["float"],
    hlc_hrc: Field["float"],
    hlc_hcf: Field["float"],
    hli_hri: Field["float"],
    hli_hcf: Field["float"],
    ice_cld_wgt: Field["float"],
    # For condensation
    cpd: Field["float"],
    rt: Field["float"],  # work array for total water mixing ratio
    pv: Field["float"],  # thermodynamics
    piv: Field["float"],  # thermodynamics
    qsl: Field["float"],  # thermodynamics
    qsi: Field["float"],
    frac_tmp: Field[IJ, "float"],  # ice fraction
    cond_tmp: Field[IJ, "float"],  # condensate
    a: Field[IJ, "float"],  # related to computation of Sig_s
    sbar: Field[IJ, "float"],
    sigma: Field[IJ, "float"],
    q1: Field[IJ, "float"],
    # parameters
    condensation_constants: Dict,
    neb_parameters: Dict,
):

    # Constants
    from __externals__ import cpd, cpv, Cl, Ci, subg_cond

    # 2.4 specific heat for moist air at t+1
    with computation(PARALLEL), interval(...):
        if krr == 7:
            cph = cpd + cpv * rv_in + Cl * (rc_in + rr) + Ci * (ri_in + rs + rg + rh)

        if krr == 6:
            cph = cpd + cpv * rv_in + Cl * (rc_in + rr) + Ci * (ri_in + rs + rg)
        if krr == 5:
            cph = cpd + cpv * rv_in + Cl * (rc_in + rr) + Ci * (ri_in + rs)
        if krr == 4:
            cph = cpd + cpv * rv_in + Cl * (rc_in + rr)
        if krr == 2:
            cph = cpd + cpv * rv_in + Cl * rc_in + Ci * ri_in

    # 3. subgrid condensation scheme
    if subg_cond:
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
        )

    # 3. not subgrid condensation scheme
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
            )
