import gt4py.cartesian.gtscript as gtscript
from phyex_gt4py.config  import dtype_float
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants


@gtscript.function
def erf(
    Cst: Constants,
    z: gtscript.Field[dtype_float],
):
    gc = -z / sqrt(2)
    gv = 1 - sign(1, gc) * sqrt(1 - exp(-4 * gc**2 / Cst.pi))

    return gc, gv
