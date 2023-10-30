import gt4py.cartesian.gtscript as gtscript
from config import dtype
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants

@gtscript.function
def update_temperature(
    t: gtscript.Field[dtype],
    rc_in: gtscript.Field[dtype],
    rc_out: gtscript.Field[dtype],
    ri_in: gtscript.Field[dtype],
    ri_out: gtscript.Field[dtype],
    lv: gtscript.Field[dtype],
    ls: gtscript.Field[dtype],
    cpd: gtscript.Field[dtype]
):
    
    t[0, 0, 0] = t[0, 0, 0] + (
            (rc_out[0, 0, 0] - rc_in[0, 0, 0]) * lv[0, 0, 0] 
            + (ri_out[0, 0, 0] - ri_in[0, 0, 0]) * ls[0, 0, 0]  
            ) / cpd[0, 0, 0]
    
    return t