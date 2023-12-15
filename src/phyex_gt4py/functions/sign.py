# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import function


@function
def sign(
    x: float,
):
    if x > 0:
        sign = 1
    elif x == 0:
        sign = 0
    elif x < 0:
        sign = -1

    return sign
