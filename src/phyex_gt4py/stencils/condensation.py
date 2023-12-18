# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import IJ, Field
from gt4py.cartesian.gtscript import sqrt, exp, log, atan, floor

from phyex_gt4py.functions.compute_ice_frac import compute_frac_ice
from phyex_gt4py.functions.temperature import update_temperature
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_function


@ported_function(from_file="PHYEX/src/common/micro/condensation.F90")
@stencil_collection("condensation")
def condensation(
    pabs: Field["float"],  # pressure (Pa)
    t: Field["float"],  # T (K)
    rv_in: Field["float"],
    rc_in: Field["float"],
    ri_in: Field["float"],
    rv_out: Field["float"],
    rc_out: Field["float"],
    ri_out: Field["float"],
    # rs: Field["float"],  # grid scale mixing ratio of snow (kg/kg)
    # rr: Field["float"],  # grid scale mixing ratio of rain (kg/kg)
    # rg: Field["float"],  # grid scale mixing ratio of graupel (kg/kg)
    sigs: Field["float"],  # Sigma_s from turbulence scheme
    cldfr: Field["float"],
    sigrc: Field["float"],  # s r_c / sig_s ** 2
    ls: Field["float"],
    lv: Field["float"],
    cph: Field["float"],
    ifr: Field["float"],  # ratio cloud ice moist part
    sigqsat: Field[
        "float"
    ],  # use an extra qsat variance contribution (if osigma is True)
    # super-saturation with respect to in in the sub saturated fraction
    hlc_hrc: Field["float"],  #
    hlc_hcf: Field["float"],  # cloud fraction
    hli_hri: Field["float"],  #
    hli_hcf: Field["float"],
    # Temporary fields
    cpd: Field["float"],
    rt: Field["float"],  # work array for total water mixing ratio
    pv: Field["float"],  # thermodynamics
    piv: Field["float"],  # thermodynamics
    qsl: Field["float"],  # thermodynamics
    qsi: Field["float"],
    frac_tmp: Field[
        "float"
    ],  # ice fraction # Translation note - 2D field in condensation.f90
    # cond_tmp: Field[ "float"],  # condensate  # Translation note - 2D field in condensation.f90
    a: Field["float"],  # related to computation of Sig_s
    sbar: Field["float"],
    sigma: Field["float"],  # 2D field in condensation.f90
    # q1: Field[IJ, "float"],
):

    from __externals__ import (
        #     # lvtt,
        #     # lstt,
        tt,
        #     # cpv,
        #     # Cl,
        #     # Ci,
        alpw,
        betaw,
        gamw,
        alpi,
        betai,
        gami,
        Rd,
        Rv,
        frac_ice_adjust,
        tmaxmix,
        tminmix,
    )

    # TODO : move src_1d into externals
    # src_1d = [
    #     0.0,
    #     0.0,
    #     2.0094444e-04,
    #     0.316670e-03,
    #     4.9965648e-04,
    #     0.785956e-03,
    #     1.2341294e-03,
    #     0.193327e-02,
    #     3.0190963e-03,
    #     0.470144e-02,
    #     7.2950651e-03,
    #     0.112759e-01,
    #     1.7350994e-02,
    #     0.265640e-01,
    #     4.0427860e-02,
    #     0.610997e-01,
    #     9.1578111e-02,
    #     0.135888e00,
    #     0.1991484,
    #     0.230756e00,
    #     0.2850565,
    #     0.375050e00,
    #     0.5000000,
    #     0.691489e00,
    #     0.8413813,
    #     0.933222e00,
    #     0.9772662,
    #     0.993797e00,
    #     0.9986521,
    #     0.999768e00,
    #     0.9999684,
    #     0.999997e00,
    #     1.0000000,
    #     1.000000,
    # ]

    # Initialize values
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
        rt[0, 0, 0] = rv_in + rc_in + ri_in * prifact

        # Translation note : 276 -> 310 (not osigmas) skipped (osigmas = True) for Arome default version
        # Translation note : 316 -> 331 (ocnd2 == True) skipped

        #
        pv[0, 0, 0] = min(
            exp(alpw - betaw / t[0, 0, 0] - gamw * log(t[0, 0, 0])),
            0.99 * pabs[0, 0, 0],
        )
        piv[0, 0, 0] = min(
            exp(alpi - betai / t[0, 0, 0]) - gami * log(t[0, 0, 0]),
            0.99 * pabs[0, 0, 0],
        )

        if rc_in > ri_in:
            if ri_in > 1e-20:
                frac_tmp[0, 0, 0] = ri_in[0, 0, 0] / (rc_in[0, 0, 0] + ri_in[0, 0, 0])

        frac_tmp = compute_frac_ice(frac_ice_adjust, tmaxmix, tminmix, t, frac_tmp, tt)

        qsl[0, 0, 0] = Rd / Rv * pv[0, 0, 0] / (pabs[0, 0, 0] - pv[0, 0, 0])
        qsi[0, 0, 0] = Rd / Rv * piv[0, 0, 0] / (pabs[0, 0, 0] - piv[0, 0, 0])

        # # dtype_interpolate bewteen liquid and solid as a function of temperature
        qsl = (1 - frac_tmp) * qsl + frac_tmp * qsi
        lvs = (1 - frac_tmp) * lv + frac_tmp * ls

        # # coefficients a et b
        ah = lvs * qsl / (Rv * t[0, 0, 0] ** 2) * (1 + Rv * qsl / Rd)
        a[0, 0, 0] = 1 / (1 + lvs / cph[0, 0, 0] * ah)
        # # b[0, 0, 0] = ah * a
        sbar = a * (
            rt[0, 0, 0] - qsl[0, 0, 0] + ah * lvs * (rc_in + ri_in * prifact) / cpd
        )

        sigma[0, 0, 0] = sqrt((2 * sigs) ** 2 + (sigqsat * qsl * a) ** 2)
        sigma[0, 0, 0] = max(1e-10, sigma[0, 0, 0])

        # # normalized saturation deficit
        # q1[0, 0] = sbar[0, 0] / sigma[0, 0, 0]
        # if q1 > 0:
        #     if q1 <= 2:
        #         cond_tmp[0, 0, 0] = min(
        #             exp(-1) + 0.66 * q1[0, 0] + 0.086 * q1[0, 0] ** 2, 2
        #     )  # we use the MIN function for continuity
        # elif q1 > 2:
        #     cond_tmp[0, 0, 0] = q1[0, 0]
        # else:
        #     cond_tmp[0, 0, 0] = exp(1.2 * q1[0, 0] - 1)

        # cond_tmp[0, 0, 0] *= sigma[0, 0, 0]

        # # cloud fraction
        # if cond_tmp < 1e-12:
        #     cldfr[0, 0, 0] = 0
        # else:
        #     cldfr[0, 0, 0] = max(0, min(1, 0.5 + 0.36 * atan(1.55 * q1[0, 0])))

        # if cldfr[0, 0, 0] == 0:
        #     cond_tmp[0, 0, 0] = 0

        # inq1 = min(
        #     10, max(-22, floor(min(-100, 2 * q1[0, 0])))
        # )  # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an "int"
        # inc = 2 * q1 - inq1

        # sigrc[0, 0, 0] = min(
        #     1, (1 - inc) * src_1d[inq1 + 22] + inc * src_1d[inq1 + 1 + 22]
        # )

        # rc_out[0, 0, 0] = (1 - frac_tmp[0, 0, 0]) * cond_tmp[0, 0, 0]  # liquid condensate
        # ri_out[0, 0, 0] = frac_tmp[0, 0, 0] * cond_tmp[0, 0, 0]  # solid condensate
        # t[0, 0, 0] = update_temperature(t, rc_in, rc_out, ri_in, ri_out, lv, ls, cpd)
        # rv_out[0, 0, 0] = rt[0, 0, 0] - rc_out[0, 0, 0] - ri_out[0, 0, 0] * prifact

        # sigrc[0, 0, 0] *= min(3, max(1, 1 - q1[0, 0]))
