import gt4py.cartesian.gtscript as gtscript
from phyex_gt4py.config  import dtype_float
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants


@gtscript.function
def update_temperature(
    t: gtscript.Field[dtype_float],
    rc_in: gtscript.Field[dtype_float],
    rc_out: gtscript.Field[dtype_float],
    ri_in: gtscript.Field[dtype_float],
    ri_out: gtscript.Field[dtype_float],
    lv: gtscript.Field[dtype_float],
    ls: gtscript.Field[dtype_float],
    cpd: gtscript.Field[dtype_float],
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
