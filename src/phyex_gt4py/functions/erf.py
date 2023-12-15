# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, function, sqrt

from phyex_gt4py.functions.sign import sign


@function
def erf(
    pi: float,
    z: Field["float"],
):
    gc = -z / sqrt(2)
    gv = 1 - sign(1, gc) * sqrt(1 - exp(-4 * gc**2 / pi))

    return gc, gv
