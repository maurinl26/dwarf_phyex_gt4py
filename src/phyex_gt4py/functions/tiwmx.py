# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, function, log

from phyex_gt4py.functions.sign import sign


@function
def esatw(alpw: float, betaw: float, tt: Field["float"]):
    esatw = exp(alpw - betaw / tt[0, 0, 0] - log(tt[0, 0, 0]))
    return esatw


@function
def esati(cst_tt: float, alpw: float, betaw: float, tt: Field["float"]):
    esati = (0.5 + sign(0.5, tt - cst_tt)) * esatw(alpw, betaw, tt) - (
        sign(0.5, tt - cst_tt) - 0.5
    ) * esatw(alpw, betaw, tt)

    return esati
