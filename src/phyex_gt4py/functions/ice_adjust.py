import gt4py.cartesian.gtscript as gtscript
from phyex_gt4py.config  import dtype_float
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants


@gtscript.function
def latent_heat(lvtt: dtype_float,
                lstt: dtype_float,
                cpv: dtype_float,
                tt: dtype_float,
                Ci: dtype_float,
                Cl: dtype_float,
                t: Field[dtype_float]):
    lv = lvtt + (cpv - Cl) * (t[0, 0, 0] - tt)
    ls = lstt + (cpv - Ci) * (t[0, 0, 0] - tt)

    return lv, ls


@gtscript.function
def _cph(
    rv: gtscript.Field[dtype_float],
    rc: gtscript.Field[dtype_float],
    ri: gtscript.Field[dtype_float],
    rr: gtscript.Field[dtype_float],
    rs: gtscript.Field[dtype_float],
    rg: gtscript.Field[dtype_float],
    cpd: dtype_float,
    cpv: dtype_float,
    Cl: dtype_float,
    Ci: dtype_float
):
    cph = cpd + cpv * rv + Cl * (rc + rr) + Ci * (ri + rs + rg)

    return cph
