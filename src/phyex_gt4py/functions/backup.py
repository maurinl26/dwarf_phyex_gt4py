# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

from gt4py.cartesian.gtscript import Field, function


@function
def backup(
    rv_tmp: Field["float"],
    ri_tmp: Field["float"],
    rc_tmp: Field["float"],
    rv_out: Field["float"],
    ri_out: Field["float"],
    rc_out: Field["float"],
) -> Tuple[Field["float"]]:
    """
    Dump out fields into temporary fields to
    perform loop iterations

    Args:
        rv_tmp (Field["float"]): vapour mixing ratio (temp field)
        ri_tmp (Field["float"]): ice mixing ratio (temp field)
        rc_tmp (Field["float"]): cloud mixing ratio (temp field)
        rv_out (Field["float"]): vapour mixing ratio (out field)
        ri_out (Field["float"]): ice mixing ratio (out field)
        rc_out (Field["float"]): cloud mixing ratio (out field)

    Returns:
        Tuple[Field["float"]]: temporary fields
    """

    rv_tmp = rv_out[0, 0, 0]
    ri_tmp = ri_out[0, 0, 0]
    rc_tmp = rc_out[0, 0, 0]

    return rv_tmp, ri_tmp, rc_tmp
