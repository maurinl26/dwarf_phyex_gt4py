# -*- coding: utf-8 -*-
from gt4py.cartesian import gtscript
from phyex_gt4py.config import dtype_float


@gtscript.function
def sign(
    x: dtype_float,
):
    if x > 0:
        sign = 1
    elif x == 0:
        sign = 0
    elif x < 0:
        sign = -1

    return sign
