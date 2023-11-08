from __future__ import annotations
import gt4py.cartesian.gtscript as gtscript
from phyex_gt4py.config  import dtype_float
from gt4py.cartesian.gtscript import Field

@gtscript.function
def update_temperature(
    t: Field[dtype_float],
    rc_in: Field[dtype_float],
    rc_out: Field[dtype_float],
    ri_in: Field[dtype_float],
    ri_out: Field[dtype_float],
    lv: Field[dtype_float],
    ls: Field[dtype_float],
    cpd: Field[dtype_float],
):
    t[0, 0, 0] = (
        t[0, 0, 0]
        + (
            (rc_out[0, 0, 0] - rc_in[0, 0, 0]) * lv[0, 0, 0]
            + (ri_out[0, 0, 0] - ri_in[0, 0, 0]) * ls[0, 0, 0]
        )
        / cpd[0, 0, 0]
    )

    return t
