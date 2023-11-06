from typing import Optional, Tuple

import gt4py.cartesian.gtscript as gtscript
from config import dtype_float
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants
from phyex_gt4py.nebn import Neb


@gtscript.function
def compute_frac_ice(
    hfrac_ice: str,
    nebn: Neb,
    cst: Constants,
    t: dtype_float,
    frac_ice: dtype_float,
    kerr: Optional[int],
) -> Tuple[int, gtscript.Field[dtype_float]]:
    kerr = 0 if kerr is not None else None

    # using temperature
    if hfrac_ice == "T":
        frac_ice = max(
            0, min(1, ((nebn.tmaxmix - t[0, 0, 0]) / (nebn.tmaxmix - nebn.tminmix)))
        )

    # using temperature with old formula
    elif hfrac_ice == "O":
        frac_ice = max(0, min(1, ((cst.tt - t[0, 0, 0]) / 40)))

    # no ice
    elif hfrac_ice == "N":
        frac_ice = 0

    # same as previous
    elif hfrac_ice == "S":
        frac_ice = max(0, min(1, frac_ice))

    else:
        kerr = 1 if kerr is not None else None

    return kerr, frac_ice
