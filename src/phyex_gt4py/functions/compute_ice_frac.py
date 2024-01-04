# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

from gt4py.cartesian.gtscript import Field, function


@function
def compute_frac_ice(
    t: Field["float"],
):

    from __externals__ import frac_ice_adjust, tmaxmix, tminmix, tt
    
    frac_ice = 0

    # using temperature
    # FracIceAdujst.T.value
    if frac_ice_adjust == 0:
        frac_ice = max(0, min(1, ((tmaxmix - t[0, 0, 0]) / (tmaxmix - tminmix))))

    # using temperature with old formula
    # FracIceAdujst.O.value
    elif frac_ice_adjust == 1:
        frac_ice = max(0, min(1, ((tt - t[0, 0, 0]) / 40)))

    # no ice
    # FracIceAdujst.N.value
    elif frac_ice_adjust == 2:
        frac_ice = 0

    # same as previous
    # FracIceAdujst.S.value
    elif frac_ice_adjust == 3:
        frac_ice = max(0, min(1, frac_ice))

    return frac_ice
