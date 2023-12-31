from __future__ import annotations

from typing import Optional

import gt4py.cartesian.gtscript as gtscript
from config import backend, dtype_float, dtype_int
from functions.gamma import gamma
from functions.ice4_sedimentation_stat import pristine_ice
from gt4py.cartesian.gtscript import IJ, Field, K
from ifs_physics_common.framework.stencil import stencil_collection

from phyex_gt4py.constants import Constants
from phyex_gt4py.dimphyex import DIMPhyex
from phyex_gt4py.rain_ice_param import ParamIce, RainIceDescr, RainIceParam


@gtscript.stencil(backend)
def ice4_sedimentation_stat(
    D: DIMPhyex,
    Cst: Constants,
    icep: RainIceParam,
    iced: RainIceDescr,
    parami: ParamIce,
    ptstep: dtype_float,
    krr: int,  # number of moist variables
    # 3D Fields
    pdzz: Field[dtype_float],  # Layer thickness
    prhodef: Field[dtype_float],  # Reference density
    ppbabst: Field[dtype_float],  # Absolute pressure at t
    ptht: Field[dtype_float],  # Theta at t
    pt: Field[dtype_float],  # Temperature
    prhodj: Field[dtype_float],  # Density * jacobian
    prcs: Field[dtype_float],  # Cloud water mixing ratio source
    prct: Field[dtype_float],  # Cloud water m. r. at t
    prrs: Field[dtype_float],  # Rain water m. r. source
    prrt: Field[dtype_float],  # Rain water m. r. at t
    pris: Field[dtype_float],  # Pristine ice m. r. source
    prit: Field[dtype_float],  # Pristine ice m. r. at t
    prss: Field[dtype_float],  # Snow m. r. source
    prst: Field[dtype_float],  # Snow m. r. at t
    prgs: Field[dtype_float],  # Graupel m. r. source
    prgt: Field[dtype_float],  # Graupel m. r. at t
    # 2D Fields
    pinprc: Field[dtype_float],  # Cloud instant precip
    pinprr: Field[dtype_float],  # Rain instant precip
    pinpri: Field[dtype_float],  # Pristine ice instant precip
    pinprs: Field[dtype_float],  # Snow instant precip
    pinprg: Optional[Field[dtype_float]],
    psea: Optional[Field[dtype_float]],
    ptown: Optional[Field[dtype_float]],
    pinprh: Optional[Field[dtype_float]],
    prht: Optional[Field[dtype_float]],
    prhs: Optional[Field[dtype_float]],
    pfpr: Optional[Field[dtype_float]],
):
    # TODO :
    invtstep = 1 / ptstep
    gac = gamma(iced.nuc + 1 / iced.alphac)
    gc = gamma(iced.nuc)
    gac2 = gamma(iced.nuc2 + 1 / iced.alphac2)
    gc2 = gamma(iced.nuc2)
    raydefo = max(1, 0.5 * (gac / gc))

    # Init
    with computation(BACKWARD), interval(0, 1):
        sed_c = 0
        sed_r = 0
        sed_i = 0
        sed_s = 0
        sed_g = 0
        sed_h = 0 if krr == 7 else None

    # 2.1 clouds
    with computation(BACKWARD), interval(...):
        tsorhodz = ptstep / (prhodef[0, 0, 0] * pdzz[0, 0, 0])

        sed_c = cloud(prct) if parami.lsedic else 0
        sed_r = other_species(icep.fsedr, icep.exsedr, prrt)
        sed_i = pristine_ice(prit)
        sed_s = other_species(icep.fseds, icep.exseds, prst)
        sed_g = other_species(icep.fsedg, icep.exsedg, prgt)
        sed_h = other_species(icep.fsedh, icep.exsedh, prht) if krr == 7 else None

        # Wrap up
        prcs[0, 0, 0] = (
            prcs[0, 0, 0]
            + tsorhodz[0, 0, 0] * (sed_c[0, 0, 1] - sed_c[0, 0, 0]) * invtstep
        )
        prrs[0, 0, 0] = (
            prrs[0, 0, 0]
            + tsorhodz[0, 0, 0] * (sed_r[0, 0, 1] - sed_r[0, 0, 0]) * invtstep
        )
        pris[0, 0, 0] = (
            pris[0, 0, 0]
            + tsorhodz[0, 0, 0] * (sed_i[0, 0, 1] - sed_i[0, 0, 0]) * invtstep
        )
        prss[0, 0, 0] = (
            prss[0, 0, 0]
            + tsorhodz[0, 0, 0] * (sed_s[0, 0, 1] - sed_s[0, 0, 0]) * invtstep
        )
        prgs[0, 0, 0] = (
            prgs[0, 0, 0]
            + tsorhodz[0, 0, 0] * (sed_g[0, 0, 1] - sed_g[0, 0, 0]) * invtstep
        )

        if krr == 7:
            prhs[0, 0, 0] = (
                prhs[0, 0, 0]
                + tsorhodz[0, 0, 0] * (sed_h[0, 0, 1] - sed_h[0, 0, 0]) * invtstep
            )

    with computation(BACKWARD), interval(0, 1):
        pinprc = sed_c / Cst.rholw
        pinprr = sed_r / Cst.rholw
        pinpri = sed_i / Cst.rholw
        pinprs = sed_s / Cst.rholw
        pinprg = sed_g / Cst.rholw
        pinprh = sed_h / Cst.rholw if krr == 7 else None


@gtscript.function
def cloud(
    iced: RainIceDescr,
    icep: RainIceParam,
    jrr: dtype_int,
    sed: Field[dtype_float],
    pxrt: Field[dtype_float],
    tsorhodz: Field[dtype_float],
    ppabst: Field[dtype_float],
    prhodref: Field[dtype_float],
    ptht: Field[dtype_float],
    psea: Field[dtype_float],
    ptown: Field[dtype_float],
    gac,
    gac2,
    gc,
    gc2,
    raydefo: dtype_float,
    ptstep: dtype_float,
    pdzz: Field[dtype_float],
    invstep: dtype_float,
):
    qp = sed[0, 0, 1] * tsorhodz[0, 0, 0]

    if pxrt[0, 0, 0] > iced.rtmin[jrr] or qp[0, 0, 0] > iced.rtmin[jrr]:
        if psea[0, 0] * ptown[0, 0] == 1:
            ray = max(
                1, 0.5 * ((1.0 - psea[0, 0]) * gac / gc + psea[0, 0] * gac2 / gc2)
            )
            lbc = max(
                min(iced.lbc[0], iced.lbc[1]),
                (psea[0, 0] * iced.lbc[1] + (1 - psea[0, 0])) * iced.lbc[0],
            )
            fsedc = max(
                min(icep.fsedc[0], icep.fsedc[1]),
                (psea[0, 0] * icep.fsedc[1] + (1.0 - psea[0, 0]) * icep.fsedc[0]),
            )
            conc3d = (1.0 - ptown[0, 0]) * (
                psea[0, 0] * iced.conc_sea + (1.0 - psea[0, 0]) * iced.conc_land
            ) + ptown[0, 0] * iced.conc_urban

        else:
            ray = raydefo
            lbc = iced.lbc[0]
            fsedc = icep.fsedc[0]
            conc3d = iced.conc_land

        # l269, mode_ice4_sedimentation_stat.F90
        if pxrt[0, 0, 0] > iced.rtmin[jrr]:
            wlbda = 6.6e-8 * (101325 / ppabst[0, 0, 0]) * (ptht[0, 0, 0] / 293.15)
            wlbdc = (lbc * conc3d / (prhodref[0, 0, 0] * pxrt[0, 0, 0])) ** iced.lbexc
            cc = iced.cc * (1 + 1.26 * wlbda * wlbdc / ray)  # Fall speed
            wsedw1 = (
                prhodref[0, 0, 0] ** (-iced.cexvt) * wlbdc ** (-iced.dc) * cc * fsedc
            )
        else:
            wsedw1 = 0

        # l277, mode_ice4_sedimentation_stat.F90
        if qp[0, 0, 0] > iced.rtmin[jrr]:
            wlbda = 6.6e-8 * (101325 / ppabst[0, 0, 0]) * (ptht[0, 0, 0] / 293.15)
            wlbdc = (lbc * conc3d / (prhodref[0, 0, 0] * qp[0, 0, 0])) ** iced.lbexc
            cc = iced.cc * (1 + 1.26 * wlbda * wlbdc / ray)  # Fall speed
            wsedw2 = (
                prhodref[0, 0, 0] ** (-iced.cexvt) * wlbdc ** (-iced.dc) * cc * fsedc
            )
        else:
            wsedw2 = 0

    else:
        wsedw1 = 0
        wsedw2 = 0

    if wsedw2 != 0:
        sed[0, 0, 0] = fwsed1(wsedw1, ptstep, pdzz, prhodref, pxrt, invstep) + fwsed2(
            wsedw2, ptstep, pdzz, prhodref, pxrt, invstep
        )

    else:
        sed[0, 0, 0] = fwsed1(wsedw1, ptstep, pdzz, prhodref, pxrt, invstep)

    return sed[0, 0, 0]


@gtscript.function
def fwsed1(
    wsedw: Field[dtype_float],
    pdzz1: dtype_float,
    phrodref1: Field[dtype_float],
    pxrt: Field[dtype_float],
    pinvstep: dtype_float,
):
    return min(
        phrodref1 * pdzz1 * pxrt[0, 0, 0] * pinvstep, wsedw * phrodref1 * pxrt[0, 0, 0]
    )


@gtscript.function
def fwsed2(wsedw: Field[dtype_float], pdzz1: dtype_float, ptstep1: dtype_float, pwsedwsup: dtype_float):
    return max(0, 1 - pdzz1 / (ptstep1 * wsedw)) * pwsedwsup


@gtscript.function
def other_species(
    iced: RainIceDescr,
    sed: Field[dtype_float],
    pxrt: Field[dtype_float],
    tsorhodz: Field[dtype_float],
    prhodref: Field[dtype_float],
    pdzz: Field[dtype_float],
    pinvstep: dtype_float,
    ptstep: dtype_float,
    jrr: dtype_int,
    fsed: dtype_float,
    exsed: dtype_float,
):
    qp = sed[0, 0, 1] * tsorhodz[0, 0, 0]
    if pxrt > iced.rtmin[jrr] or qp > iced.rtmin[jrr]:
        if pxrt > iced.rtmin[jrr]:
            wsedw1 = (
                fsed
                * pxrt[0, 0, 0] ** (exsed - 1)
                * prhodref[0, 0, 0] ** (exsed - iced.cexvt)
            )
        else:
            wsedw1 = 0

        if qp > iced.rtmin[jrr]:
            wsedw2 = (
                fsed * qp ** (exsed - 1) * prhodref[0, 0, 0] ** (exsed - iced.cexvt)
            )

    else:
        wsedw1 = 0
        wsedw2 = 0

    if wsedw2 != 0:
        sed[0, 0, 0] = fwsed1(wsedw1, pdzz, prhodref, pxrt, pinvstep) + fwsed2(
            wsedw2, pdzz, ptstep, sed[0, 0, 1]
        )
    else:
        sed[0, 0, 0] = fwsed1(wsedw1, pdzz, prhodref, pxrt, pinvstep)

    return sed


@gtscript.function
def pristine_ice(
    iced: RainIceDescr,
    icep: RainIceParam,
    parami: ParamIce,
    cst: Constants,
    jrr: dtype_int,
    pxrt: Field[dtype_float],
    t: Field[dtype_float],
    sed: Field[dtype_float],
    tsorhodz: Field[dtype_float],
    prhodref: Field[dtype_float],
    pdzz: Field[dtype_float],
    ptstep: Field[dtype_float],
    pinvstep: Field[dtype_float],
):
    qp = sed[0, 0, 1] * tsorhodz[0, 0, 0]
    if pxrt[0, 0, 0] > iced.rtmin[jrr]:
        if parami.lsnow_t:
            if t[0, 0, 0] > cst.tt - 10:
                lbdas = (
                    max(
                        min(iced.lbdas_max, 10 ** (14.554 - 0.0423 * t[0, 0, 0])),
                        iced.lbdas_min,
                    )
                    * iced.trans_mp_gammas
                )
            else:
                lbdas = (
                    max(
                        min(iced.lbdas_max, 10 ** (6.226 - 0.0106 * t[0, 0, 0])),
                        iced.lbdas_min,
                    )
                    * iced.trans_mp_gammas
                )
        else:
            lbdas = max(
                min(
                    iced.lbdas_max,
                    iced.lbs * (prhodref[0, 0, 0] * pxrt[0, 0, 0]) ** iced.lbexs,
                ),
                iced.lbdas_min,
            )

        if pxrt[0, 0, 0] > iced.rtmin[jrr]:
            wsedw1 = (
                icep.fseds
                * prhodref[0, 0, 0] ** (-iced.cexvt)
                * (1 + (iced.fvelos / lbdas) ** iced.alphas)
                ** (-iced.nuc + icep.exseds / iced.alphas)
                * lbdas ** (iced.bs + icep.exseds)
            )
        else:
            wsedw1 = 0

        if qp[0, 0, 0] > iced.rtmin[jrr]:
            wsedw2 = (
                icep.fseds
                * prhodref[0, 0, 0] ** (-iced.cexvt)
                * (1 + (iced.fvelos / lbdas) ** iced.alphas)
                ** (-iced.nuc + icep.exseds / iced.alphas)
                * lbdas ** (iced.bs + icep.exseds)
            )
        else:
            wsedw2 = 0
    else:
        wsedw1 = 0
        wsedw2 = 0

    if wsedw2 != 0:
        sed[0, 0, 0] = fwsed1(wsedw1, pdzz, prhodref, pxrt, pinvstep) + fwsed2(
            wsedw2, pdzz, ptstep, sed[0, 0, 1]
        )
    else:
        sed[0, 0, 0] = fwsed1(wsedw1, pdzz, prhodref, pxrt, pinvstep)

    return sed
