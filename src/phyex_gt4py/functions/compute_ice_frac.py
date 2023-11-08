from __future__ import annotations
from typing import Optional, Tuple

import gt4py.cartesian.gtscript as gtscript
from phyex_gt4py.config import dtype_float
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants
from phyex_gt4py.nebn import Neb


@gtscript.function
def compute_frac_ice(
    hfrac_ice: str,
    tmaxmix: dtype_float,
    tminmix: dtype_float,
    cst: Constants,
    t: dtype_float,
    frac_ice: dtype_float,
    kerr: Optional[int],
) -> Tuple[int, gtscript.Field[dtype_float]]:
    kerr = 0 if kerr is not None else None

    # using temperature
    # FracIceAdujst.T.value
    if hfrac_ice == 0:
        frac_ice = max(
            0, min(1, ((tmaxmix - t[0, 0, 0]) / (tmaxmix - tminmix)))
        )

    # using temperature with old formula
    # FracIceAdujst.O.value
    elif hfrac_ice == 1:
        frac_ice = max(0, min(1, ((cst.tt - t[0, 0, 0]) / 40)))

    # no ice
    # FracIceAdujst.N.value
    elif hfrac_ice == 2:
        frac_ice = 0

    # same as previous
    # FracIceAdujst.S.value
    elif hfrac_ice == 3:
        frac_ice = max(0, min(1, frac_ice))

    else:
        kerr = 1 if kerr is not None else None

    return kerr, frac_ice
