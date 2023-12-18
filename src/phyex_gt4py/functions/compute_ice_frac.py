# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

from gt4py.cartesian.gtscript import Field, function


@function
def compute_frac_ice(
    hfrac_ice: int,
    tmaxmix: float,
    tminmix: float,
    t: Field["float"],
    frac_ice: float,
    tt: float,
) -> Tuple[int, Field["float"]]:

    # using temperature
    # FracIceAdujst.T.value
    if hfrac_ice == 0:
        frac_ice = max(0, min(1, ((tmaxmix - t[0, 0, 0]) / (tmaxmix - tminmix))))

    # using temperature with old formula
    # FracIceAdujst.O.value
    elif hfrac_ice == 1:
        frac_ice = max(0, min(1, ((tt - t[0, 0, 0]) / 40)))

    # no ice
    # FracIceAdujst.N.value
    elif hfrac_ice == 2:
        frac_ice = 0

    # same as previous
    # FracIceAdujst.S.value
    elif hfrac_ice == 3:
        frac_ice = max(0, min(1, frac_ice))

    return frac_ice
