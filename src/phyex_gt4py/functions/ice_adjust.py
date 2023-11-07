import gt4py.cartesian.gtscript as gtscript
from phyex_gt4py.config  import dtype_float
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants


@gtscript.function
def latent_heat(cst: Constants, t: gtscript.Field[dtype_float]):
    lv = cst.lvtt + (cst.cpv - cst.Cl) * (t[0, 0, 0] - cst.tt)
    ls = cst.lstt + (cst.cpv - cst.Ci) * (t[0, 0, 0] - cst.tt)

    return lv, ls


@gtscript.function
def _cph(
    cst: Constants,
    rv: gtscript.Field[dtype_float],
    rc: gtscript.Field[dtype_float],
    ri: gtscript.Field[dtype_float],
    rr: gtscript.Field[dtype_float],
    rs: gtscript.Field[dtype_float],
    rg: gtscript.Field[dtype_float],
):
    cph = cst.cpd + cst.cpv * rv + cst.Cl * (rc + rr) + cst.Ci * (ri + rs + rg)

    return cph
