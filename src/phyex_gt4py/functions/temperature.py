# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function


@function
def update_temperature(
    t: Field["float"],
    rc_in: Field["float"],
    rc_out: Field["float"],
    ri_in: Field["float"],
    ri_out: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
    cpd: float,
):
    t = (
        t[0, 0, 0]
        + (
            (rc_out[0, 0, 0] - rc_in[0, 0, 0]) * lv[0, 0, 0]
            + (ri_out[0, 0, 0] - ri_in[0, 0, 0]) * ls[0, 0, 0]
        )
        / cpd
    )

    return t
