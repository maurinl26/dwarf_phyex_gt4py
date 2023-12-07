# -*- coding: utf-8 -*-

from __future__ import annotations
from functools import cached_property
import sys

from phyex_gt4py.drivers.ice_adjust import IceAdjust
from phyex_gt4py.drivers.main_ice_adjust import initialize_fields

sys.path = [
    *sys.path,
    *[
        "./src/cloudsc_python/src",
        "../ifs-physics-common/src",
        "./src/cloudsc_gt4py/src",
        "./src",
    ],
]
print(sys.path)

from typing import Dict, Optional

from gt4py.storage import zeros
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.utils.typingx import PropertyDict
from ifs_physics_common.framework.grid import I, J, K

from phyex_gt4py.config import dtype_float
from phyex_gt4py.phyex import Phyex
from phyex_gt4py.stencils.ice_adjust import ice_adjust


if __name__ == "__main__":

    cprogram = "AROME"
    iulout = 20
    dzmin = 20
    cmicro = "ICE3"
    csconv = "NONE"
    cturb = "TKEL"
    pstep = 50.0
    hbuname = "NONE"

    # Especes microphysiques (starting from 1)
    nrr = 6

    nx = 100
    ny = 100
    nz = 90

    # Phyex parameters
    phyex = Phyex(cprogram)

    ##### Define computational grid #####
    grid = ComputationalGrid(nx, ny, nz)

    aro_adjust = IceAdjust()

    initialize_fields(grid, aro_adjust.tendency_properties)
    initialize_fields(grid, aro_adjust.diagnostic_properties)
    initialize_fields(grid, aro_adjust.input_properties)
    initialize_fields(grid, aro_adjust.temporaries)

    #### Launch ice adjust #####
    ice_adjust(
        cst=phyex.cst,
        parami=phyex.param_icen,
        icep=phyex.rain_ice_paramn,
        neb=phyex.nebn,
        compute_srcs=phyex.compute_src,
        itermax=phyex.itermax,
        tstep=phyex.tstep,
        krr=nrr,
        lmfconv=phyex.lmfconv,
        buname=hbuname,
        **aro_adjust.input_properties,  # IN
        **aro_adjust.tendency_properties  # INOUT
        ** aro_adjust.diagnostic_properties  # OUT
        ** aro_adjust.temporaries,  # Temporary  fields
    )
