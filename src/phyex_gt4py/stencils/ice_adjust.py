# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field
from gt4py.cartesian.gtscript import exp, log, sqrt, floor, atan
from phyex_gt4py.functions.compute_ice_frac import compute_frac_ice
from phyex_gt4py.functions.src_1d import src_1d
from phyex_gt4py.functions.temperature import update_temperature

from phyex_gt4py.functions.ice_adjust import vaporisation_latent_heat, sublimation_latent_heat
from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("ice_adjust")
def ice_adjust(
    # IN - Inputs
    sigqsat: Field["float"],  # coeff applied to qsat variance
    exnref: Field["float"],  # ref exner pression
    exn: Field["float"],
    rhodref: Field["float"],  #
    pabs: Field["float"],  # absolute pressure at t
    sigs: Field["float"],  # Sigma_s at time t
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
    rh: Field["float"],  # hail m.r. to adjust (if nrr = 7)
    ths: Field["float"],  # theta source
    rvs: Field["float"],  # water vapour m.r. source
    rcs: Field["float"],  # cloud water m.r. source
    ris: Field["float"],  # cloud ice m.r. at t+1
    cldfr: Field["float"],
    # srcs: Field["float"],  # second order flux s at time t+1
    # OUT - Diagnostics
    ifr: Field["float"],  # ratio cloud ice moist part to dry part
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
    sigrc: Field["float"],
    rv_tmp: Field["float"],
    ri_tmp: Field["float"],
    rc_tmp: Field["float"],
    t_tmp: Field["float"],
    cph: Field["float"],  # guess of the CPh for the mixing
    lv: Field["float"],  # guess of the Lv at t+1
    ls: Field["float"],  # guess of the Ls at t+1
    criaut: Field["float"],  # autoconversion thresholds
    # Temporary fields # Condensation
    rt: Field["float"],  # work array for total water mixing ratio
    pv: Field["float"],  # thermodynamics
    piv: Field["float"],  # thermodynamics
    qsl: Field["float"],  # thermodynamics
    qsi: Field["float"],
    frac_tmp: Field["float"],  # ice fraction
    cond_tmp: Field["float"],  # condensate
    a: Field["float"],  # related to computation of Sig_s
    sbar: Field["float"],
    sigma: Field["float"],
    q1: Field["float"],
):
    """_summary_

    Args:

    """

    from __externals__ import (
        tt,
        subg_mf_pdf,
        subg_cond,
        cpd,
        cpv,
        Cl,
        Ci,
        tt,
        alpw,
        betaw,
        gamw,
        alpi,
        betai,
        gami,
        Rd,
        Rv,
        criautc,
        tstep,
        criauti,
        acriauti,
        bcriauti,
        nrr,  # number of moist variables
    )

    # 2.3 Compute the variation of mixing ratio
    with computation(PARALLEL), interval(...):
        t_tmp = th[0, 0, 0] * exn[0, 0, 0]
        lv = vaporisation_latent_heat(t_tmp)
        ls = sublimation_latent_heat(t_tmp)

        # Rem
        rv_tmp[0, 0, 0] = rv[0, 0, 0]
        ri_tmp[0, 0, 0] = ri[0, 0, 0]
        rc_tmp[0, 0, 0] = rc[0, 0, 0]

    # Translation note : in Fortran, ITERMAX = 1, DO JITER =1,ITERMAX
    # Translation note : version without iteration is kept (1 iteration)
    #                   IF jiter = 1; CALL ITERATION()
    # jiter > 0

    # TODO : fix number of moist variables to 6 (without hail)
    # 2.4 specific heat for moist air at t+1
    with computation(PARALLEL), interval(...):
        if nrr == 7:
            cph = cpd + cpv * rv_tmp + Cl * (rc_tmp + rr) + Ci * (ri_tmp + rs + rg + rh)

        if nrr == 6:
            cph = cpd + cpv * rv_tmp + Cl * (rc_tmp + rr) + Ci * (ri_tmp + rs + rg)
        if nrr == 5:
            cph = cpd + cpv * rv_tmp + Cl * (rc_tmp + rr) + Ci * (ri_tmp + rs)
        if nrr == 4:
            cph = cpd + cpv * rv_tmp + Cl * (rc_tmp + rr)
        if nrr == 2:
            cph = cpd + cpv * rv_tmp + Cl * rc_tmp + Ci * ri_tmp

    # 3. subgrid condensation scheme
    # Translation : only the case with subg_cond = True retained
    with computation(PARALLEL), interval(...):
        cldfr[0, 0, 0] = 0
        sigrc[0, 0, 0] = 0
        rv_out[0, 0, 0] = 0
        rc_out[0, 0, 0] = 0
        ri_out[0, 0, 0] = 0

        # local fields
        # Translation note : 506 -> 514 kept (ocnd2 == False) # Arome default setting
        # Translation note : 515 -> 575 skipped (ocnd2 == True)
        prifact = 1  # ocnd2 == False for AROME
        ifr[0, 0, 0] = 10
        frac_tmp[0, 0, 0] = 0  # l340 in Condensation .f90

        # Translation note : 493 -> 503 : hlx_hcf and hlx_hrx are assumed to be present
        hlc_hcf[0, 0, 0] = 0
        hlc_hrc[0, 0, 0] = 0
        hli_hcf[0, 0, 0] = 0
        hli_hri[0, 0, 0] = 0

        # Translation note : 252 -> 263 if(present(PLV)) skipped (ls/lv are assumed to be present present)
        # Translation note : 264 -> 274 if(present(PCPH)) skipped (files are assumed to be present)

        # store total water mixing ratio (244 -> 248)
        rt[0, 0, 0] = rv_tmp + rc_tmp + ri_tmp * prifact

        # Translation note : 276 -> 310 (not osigmas) skipped (osigmas = True) for Arome default version
        # Translation note : 316 -> 331 (ocnd2 == True) skipped

        #
        pv[0, 0, 0] = min(
            exp(alpw - betaw / t_tmp[0, 0, 0] - gamw * log(t_tmp[0, 0, 0])),
            0.99 * pabs[0, 0, 0],
        )
        piv[0, 0, 0] = min(
            exp(alpi - betai / t_tmp[0, 0, 0]) - gami * log(t_tmp[0, 0, 0]),
            0.99 * pabs[0, 0, 0],
        )

        if rc_tmp > ri_tmp:
            if ri_tmp > 1e-20:
                frac_tmp[0, 0, 0] = ri_tmp[0, 0, 0] / (
                    rc_tmp[0, 0, 0] + ri_tmp[0, 0, 0]
                )

        frac_tmp = compute_frac_ice(t_tmp)

        qsl[0, 0, 0] = Rd / Rv * pv[0, 0, 0] / (pabs[0, 0, 0] - pv[0, 0, 0])
        qsi[0, 0, 0] = Rd / Rv * piv[0, 0, 0] / (pabs[0, 0, 0] - piv[0, 0, 0])

        # # dtype_interpolate bewteen liquid and solid as a function of temperature
        qsl = (1 - frac_tmp) * qsl + frac_tmp * qsi
        lvs = (1 - frac_tmp) * lv + frac_tmp * ls

        # # coefficients a et b
        ah = lvs * qsl / (Rv * t_tmp[0, 0, 0] ** 2) * (1 + Rv * qsl / Rd)
        a[0, 0, 0] = 1 / (1 + lvs / cph[0, 0, 0] * ah)
        # # b[0, 0, 0] = ah * a
        sbar = a * (
            rt[0, 0, 0] - qsl[0, 0, 0] + ah * lvs * (rc_tmp + ri_tmp * prifact) / cpd
        )

        sigma[0, 0, 0] = sqrt((2 * sigs) ** 2 + (sigqsat * qsl * a) ** 2)
        sigma[0, 0, 0] = max(1e-10, sigma[0, 0, 0])

        # Translation notes : 469 -> 504 (hcondens = "CB02")
        # normalized saturation deficit
        q1[0, 0, 0] = sbar[0, 0, 0] / sigma[0, 0, 0]
        if q1 > 0:
            if q1 <= 2:
                cond_tmp[0, 0, 0] = min(
                    exp(-1) + 0.66 * q1[0, 0, 0] + 0.086 * q1[0, 0, 0] ** 2, 2
                )  # we use the MIN function for continuity
        elif q1 > 2:
            cond_tmp[0, 0, 0] = q1[0, 0, 0]
        else:
            cond_tmp[0, 0, 0] = exp(1.2 * q1[0, 0, 0] - 1)

        cond_tmp[0, 0, 0] *= sigma[0, 0, 0]

        # cloud fraction
        if cond_tmp < 1e-12:
            cldfr[0, 0, 0] = 0
        else:
            cldfr[0, 0, 0] = max(0, min(1, 0.5 + 0.36 * atan(1.55 * q1[0, 0, 0])))

        if cldfr[0, 0, 0] == 0:
            cond_tmp[0, 0, 0] = 0

        inq1 = min(
            10, max(-22, floor(min(-100, 2 * q1[0, 0, 0])))
        )  # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an "int"
        inc = 2 * q1 - inq1

        sigrc[0, 0, 0] = min(
            1, (1 - inc) * src_1d(inq1 + 22) + inc * src_1d(inq1 + 1 + 22)
        )

        # # Translation notes : 506 -> 514 (not ocnd2)
        rc_out[0, 0, 0] = (1 - frac_tmp[0, 0, 0]) * cond_tmp[
            0, 0, 0
        ]  # liquid condensate
        ri_out[0, 0, 0] = frac_tmp[0, 0, 0] * cond_tmp[0, 0, 0]  # solid condensate
        t_tmp[0, 0, 0] = update_temperature(
            t_tmp, rc_tmp, rc_out, ri_tmp, ri_out, lv, ls, cpd
        )
        rv_out[0, 0, 0] = rt[0, 0, 0] - rc_out[0, 0, 0] - ri_out[0, 0, 0] * prifact

        sigrc[0, 0, 0] = sigrc[0, 0, 0] * min(3, max(1, 1 - q1[0, 0, 0]))

    # Translation note : end jiter

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

        if subg_cond == 0:
            if rcs[0, 0, 0] + ris[0, 0, 0] > 1e-12 / tstep:
                cldfr[0, 0, 0] = 1
            else:
                cldfr[0, 0, 0] = 0

        # Translation note : LSUBG_COND = TRUE for Arome
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

    # Droplets subgrid autoconversion
    with computation(PARALLEL), interval(...):
        criaut = criautc / rhodref[0, 0, 0]

        if subg_mf_pdf == 0:
            if w1 * tstep > cf_mf[0, 0, 0] * criaut:
                hlc_hrc += w1 * tstep
                hlc_hcf = min(1, hlc_hcf[0, 0, 0] + cf_mf[0, 0, 0])

        elif subg_mf_pdf == 1:
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

    # Ice subgrid autoconversion
    with computation(PARALLEL), interval(...):
        criaut = min(
            criauti,
            10 ** (acriauti * (t_tmp[0, 0, 0] - tt) + bcriauti),
        )

        if subg_mf_pdf == 0:
            if w2 * tstep > cf_mf[0, 0, 0] * criaut:
                hli_hri += w2 * tstep
                hli_hcf = min(1, hli_hcf[0, 0, 0] + cf_mf[0, 0, 0])

        elif subg_mf_pdf == 1:
            if w2 * tstep > cf_mf[0, 0, 0] * criaut:
                hli_hcf = 1 - 0.5 * (criaut * cf_mf[0, 0, 0]) / max(1e-20, w2 * tstep)
                hli_hri = w2 * tstep - (criaut * cf_mf[0, 0, 0]) ** 3 / (
                    3 * max(1e-20, w2 * tstep)
                )

        elif 2 * w2 * tstep <= cf_mf[0, 0, 0] * criaut:
            hli_hcf = 0
            hli_hri = 0

        else:
            hli_hcf = (2 * w2 * tstep - criaut * cf_mf[0, 0, 0]) ** 2 / (
                2.0 * max(1.0e-20, w2 * tstep) ** 2
            )
            hli_hri = (
                4.0 * (w2 * tstep) ** 3
                - 3.0 * w2 * tstep * (criaut * cf_mf[0, 0, 0]) ** 2
                + (criaut * cf_mf[0, 0, 0]) ** 3
            ) / (3 * max(1.0e-20, w2 * tstep) ** 2)

        hli_hcf *= cf_mf[0, 0, 0]
        hli_hcf = min(1, hli_hcf + hli_hcf)
        hli_hri += hli_hri

    # Subgrid mass fluxes
    with computation(PARALLEL), interval(...):
        w1 = rc_mf
        w2 = ri_mf

        if w1 + w2 > rv_out[0, 0, 0]:
            w1 *= rv_tmp / (w1 + w2)
            w2 = rv_tmp - w1

        rc_tmp[0, 0, 0] += w1
        ri_tmp[0, 0, 0] += w2
        rv_tmp[0, 0, 0] -= w1 + w2
        t_tmp += (w1 * lv + w2 * ls) / cph

        # TODO : remove unused out variables
        rv_out[0, 0, 0] = rv_tmp[0, 0, 0]
        ri_out[0, 0, 0] = ri_tmp[0, 0, 0]
        rc_out[0, 0, 0] = rc_tmp[0, 0, 0]
        th_out[0, 0, 0] = t_tmp[0, 0, 0] / exn[0, 0, 0]
