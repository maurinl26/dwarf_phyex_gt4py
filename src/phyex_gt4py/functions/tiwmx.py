from __future__ import annotations
from gt4py.cartesian.gtscript import Field
from phyex_gt4py.config  import dtype_float
from gt4py.cartesian import gtscript

@gtscript.function
def esatw(alpw: dtype_float, betaw: dtype_float, tt: Field[dtype_float]):
    esatw = exp(alpw - betaw / tt[0, 0, 0] - log(tt[0, 0, 0]))
    return esatw


@gtscript.function
def esati(cst_tt: dtype_float, alpw: dtype_float, betaw: dtype_float, tt: Field[dtype_float]):
    esati = (0.5 + sign(0.5, tt - cst_tt)) * esatw(alpw, betaw, tt) - (
        sign(0.5, tt - cst_tt) - 0.5
    ) * esatw(alpw, betaw, tt)

    return esati
