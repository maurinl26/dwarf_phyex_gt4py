from __future__ import annotations
from typing import Optional
import numpy as np
from functions.gamma import gamma
from functions.ice4_sedimentation_stat import cloud, other_species, pristine_ice

from config import dtype, dtype_int, backend

import gt4py.cartesian.gtscript as gtscript

from gt4py.cartesian.gtscript import Field, IJ, K
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
    ptstep: dtype,
    krr: int,  # number of moist variables
    # 3D Fields
    pdzz: Field[dtype],     # Layer thickness
    prhodef: Field[dtype],  # Reference density
    ppbabst: Field[dtype],  # Absolute pressure at t
    ptht: Field[dtype],     # Theta at t
    pt: Field[dtype],       # Temperature
    prhodj: Field[dtype],   # Density * jacobian
    prcs: Field[dtype],     # Cloud water mixing ratio source
    prct: Field[dtype],     # Cloud water m. r. at t
    prrs: Field[dtype],     # Rain water m. r. source
    prrt: Field[dtype],     # Rain water m. r. at t
    pris: Field[dtype],     # Pristine ice m. r. source
    prit: Field[dtype],     # Pristine ice m. r. at t
    prss: Field[dtype],     # Snow m. r. source
    prst: Field[dtype],     # Snow m. r. at t
    prgs: Field[dtype],     # Graupel m. r. source
    prgt: Field[dtype],     # Graupel m. r. at t
    # 2D Fields
    pinprc: Field[dtype],   # Cloud instant precip
    pinprr: Field[dtype],   # Rain instant precip
    pinpri: Field[dtype],   # Pristine ice instant precip
    pinprs: Field[dtype],   # Snow instant precip
    pinprg: Optional[Field[dtype]],
    psea: Optional[Field[dtype]],
    ptown: Optional[Field[dtype]],
    pinprh: Optional[Field[dtype]],
    prht: Optional[Field[dtype]],
    prhs: Optional[Field[dtype]],
    pfpr: Optional[Field[dtype]],
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
        
        sed_2 = 0
        sed_3 = 0
        sed_4 = 0
        sed_5 = 0
        sed_6 = 0
        sed_7 = 0

    # 2.1 clouds
    with computation(BACKWARD), interval(...):
        
        tsorhodz = ptstep / (prhodef[0, 0, 0] * pdzz[0, 0, 0])
        
        # jrr == 2
        sed_c = cloud(prct) if parami.lsedic else 0 
  
        # jrr == 3   
        sed_r = other_species(icep.fsedr, icep.exsedr, prrt)
        
        # jrr == 4
        sed_i = pristine_ice(prit)
        
        # jrr == 5
        # TODO : special case snow
        sed_s = other_species(icep.fseds, icep.exseds, prst)
        
        # jrr == 6
        sed_g = other_species(icep.fsedg, icep.exsedg, prgt)
        
        # jrr = 7
        sed_h = other_species(icep.fsedh, icep.exsedh, prht) if krr == 7 else None
        
        # Wrap up
        prcs[0, 0, 0] = prcs[0, 0, 0] + tsorhodz[0, 0, 0] * (sed_c[0, 0, 1] - sed_c[0, 0, 0]) * invtstep
        # TODO : extension to other species
        
        # Bottom
        
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
    sed: Field[dtype],
    pxrt: Field[dtype],
    tsorhodz: Field[dtype],
    ppabst: Field[dtype],
    prhodref: Field[dtype],
    ptht: Field[dtype],
    psea: Field[dtype],
    ptown: Field[dtype],
    gac,
    gac2, 
    gc, 
    gc2,
    raydefo: dtype,
    ptstep: dtype,
    pdzz: Field[dtype],
    invstep: dtype
):
    
    qp = sed[0, 0, 1] * tsorhodz[0, 0, 0]
    
    if pxrt[0, 0, 0] > iced.rtmin[jrr] or qp[0, 0, 0] > iced.rtmin[jrr]:
        
        if psea[0, 0] * ptown[0, 0] == 1:
            ray = max(1, 0.5 * ((1.- psea[0, 0]) * gac / gc + psea[0, 0] * gac2 / gc2))
            lbc = max(min(iced.lbc[0], iced.lbc[1]),(psea[0, 0] * iced.lbc[1] + (1 - psea[0, 0])) * iced.lbc[0])
            fsedc = max(min(icep.fsedc[0], icep.fsedc[1]), ( psea[0, 0] * icep.fsedc[1] + (1. - psea[0, 0]) * icep.fsedc[0]))
            conc3d = (1. - ptown[0, 0]) * ( psea[0, 0] * iced.conc_sea + (1.-psea[0, 0]) * iced.conc_land) + ptown[0, 0] * iced.conc_urban
            
        else:
            ray = raydefo
            lbc = iced.lbc[0]
            fsedc = icep.fsedc[0]
            conc3d = iced.conc_land       
        
        # l269, mode_ice4_sedimentation_stat.F90
        if pxrt[0, 0, 0] > iced.rtmin[jrr]:
            wlbda = 6.6e-8 * (101325 / ppabst[0, 0, 0]) * (ptht[0, 0, 0] / 293.15)
            wlbdc = (lbc * conc3d / (prhodref[0, 0, 0] * pxrt[0, 0, 0])) ** iced.lbexc
            cc = iced.cc * (1 + 1.26 * wlbda * wlbdc / ray) # Fall speed
            wsedw1 = prhodref[0, 0, 0] ** (-iced.cexvt) * wlbdc ** (-iced.dc) * cc * fsedc
        else:
            wsedw1 = 0
            
        # l277, mode_ice4_sedimentation_stat.F90
        if qp[0, 0, 0] > iced.rtmin[jrr]:
            wlbda = 6.6e-8 * (101325 / ppabst[0, 0, 0]) * (ptht[0, 0, 0] / 293.15)
            wlbdc = (lbc * conc3d / (prhodref[0, 0, 0] * qp[0, 0, 0])) ** iced.lbexc
            cc = iced.cc * (1 + 1.26 * wlbda * wlbdc / ray) # Fall speed
            wsedw2 = prhodref[0, 0, 0] ** (-iced.cexvt) * wlbdc ** (-iced.dc) * cc * fsedc
        else:
            wsedw2 = 0
            
    else:
        wsedw1 = 0
        wsedw2 = 0
        
    if wsedw2 != 0:
        sed[0, 0, 0] = (
            fwsed1(wsedw1, ptstep, pdzz, prhodref, pxrt, invstep) 
            + fwsed2(wsedw2, ptstep, pdzz, prhodref, pxrt, invstep) 
        )
            
    else:
        sed[0, 0, 0] = fwsed1(wsedw1, ptstep, pdzz, prhodref, pxrt, invstep) 
            
    return sed[0, 0, 0]

@gtscript.function
def fwsed1(
    wsedw: Field[dtype],
    pdzz1: dtype,
    phrodref1: Field[dtype],
    pxrt: Field[dtype],
    pinvstep: dtype
    ):
    return min(phrodref1*pdzz1*pxrt[0, 0, 0]*pinvstep, wsedw*phrodref1*pxrt[0, 0, 0])

@gtscript.function
def fwsed2(
    wsedw: Field[dtype],
    pdzz1: dtype,
    ptstep1: dtype,
    pwsedwsup: dtype
):
    return max(0, 1 - pdzz1 / (ptstep1 * wsedw)) * pwsedwsup