import gt4py.cartesian.gtscript as gtscript
from config import dtype_float
from gt4py.cartesian.gtscript import Field

from phyex_gt4py.constants import Constants
from typing import Tuple

@gtscript.function()
def backup(
    rv_tmp: gtscript.Field,
    ri_tmp: gtscript.Field,
    rc_tmp: gtscript.Field,
    rv_out: gtscript.Field,
    ri_out: gtscript.Field,
    rc_out: gtscript.Field
) -> Tuple[gtscript.Field]:
    """
    Dump out fields into temporary fields to 
    perform loop iterations

    Args:
        rv_tmp (gtscript.Field): vapour mixing ratio (temp field)
        ri_tmp (gtscript.Field): ice mixing ratio (temp field)
        rc_tmp (gtscript.Field): cloud mixing ratio (temp field)
        rv_out (gtscript.Field): vapour mixing ratio (out field)
        ri_out (gtscript.Field): ice mixing ratio (out field)
        rc_out (gtscript.Field): cloud mixing ratio (out field)

    Returns:
        Tuple[gtscript.Field]: temporary fields
    """
    
    rv_tmp = rv_out[0, 0, 0]
    ri_tmp = ri_out[0, 0, 0]
    rc_tmp = rc_out[0, 0, 0]
    
    return rv_tmp, ri_tmp, rc_tmp