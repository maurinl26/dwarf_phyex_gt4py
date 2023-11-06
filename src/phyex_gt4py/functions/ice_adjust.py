import gt4py.cartesian.gtscript as gtscript
from config import dtype
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants


@gtscript.function
def latent_heat(cst: Constants, t: gtscript.Field[dtype]):
    lv = cst.lvtt + (cst.cpv - cst.Cl) * (t[0, 0, 0] - cst.tt)
    ls = cst.lstt + (cst.cpv - cst.Ci) * (t[0, 0, 0] - cst.tt)

    return lv, ls


@gtscript.function
def _cph(
    cst: Constants,
    rv: gtscript.Field[dtype],
    rc: gtscript.Field[dtype],
    ri: gtscript.Field[dtype],
    rr: gtscript.Field[dtype],
    rs: gtscript.Field[dtype],
    rg: gtscript.Field[dtype],
):
    cph = cst.cpd + cst.cpv * rv + cst.Cl * (rc + rr) + cst.Ci * (ri + rs + rg)

    return cph
