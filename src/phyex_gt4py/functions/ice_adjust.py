# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function


@function
def latent_heat(
    lvtt: float,
    lstt: float,
    cpv: float,
    tt: float,
    Ci: float,
    Cl: float,
    t: Field["float"],
):
    lv = lvtt + (cpv - Cl) * (t[0, 0, 0] - tt)
    ls = lstt + (cpv - Ci) * (t[0, 0, 0] - tt)

    return lv, ls


@function
def _cph(
    rv: Field["float"],
    rc: Field["float"],
    ri: Field["float"],
    rr: Field["float"],
    rs: Field["float"],
    rg: Field["float"],
    cpd: float,
    cpv: float,
    Cl: float,
    Ci: float,
):
    cph = cpd + cpv * rv + Cl * (rc + rr) + Ci * (ri + rs + rg)

    return cph
