from __future__ import annotations
from typing import Optional

from config import dtype, backend

import gt4py.cartesian.gtscript as gtscript
from phyex_gt4py.constants import Constants

from phyex_gt4py.dimphyex import DIMPhyex
from phyex_gt4py.rain_ice_param import ParamIce, RainIceDescr, RainIceParam


# Init 
@gtscript.stencil(backend)
def rain_ice_init(
    D: DIMPhyex,
    Cst: Constants,
    parami: ParamIce,
    icep: RainIceParam,
    iced: RainIceDescr,
    
    # Fields
    
    
):
    
    return None

# 2.     COMPUTE THE SEDIMENTATION (RS) SOURCE


# 3.     INITIAL VALUES SAVING
@gtscript.stencil(backend)
def save_initial_values():
    
    return None

#
