import gt4py.cartesian.gtscript as gtscript
from config import dtype
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants

@gtscript.function
def esatw(
    cst: Constants,
    tt: gtscript.Field[dtype]
):
    esatw = exp(cst.alpw - cst.betaw / tt[0, 0, 0] - cst.log(tt[0, 0, 0]))
    return esatw

@gtscript.function
def esati(
    cst: Constants,
    tt: gtscript.Field[dtype]
):
    esati = (
        (0.5 + sign(0.5, tt - cst.tt)) * esatw(cst, tt)
        - (sign(0.5, tt - cst.tt) - 0.5) * esatw(cst, tt)
        )
    
    return esati