# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import Field
from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("mass_flux")
def mass_flux(
    rc_mf: Field["float"],
    ri_mf: Field["float"],
    rv_tmp: Field["float"],
    rc_tmp: Field["float"],
    ri_tmp: Field["float"],
    rv_out: Field["float"],
    rc_out: Field["float"],
    ri_out: Field["float"],
    t_tmp: Field["float"],
    th_out: Field["float"],
    exn: Field["float"],
    ls: Field["float"],
    lv: Field["float"],
    cph: Field["float"],
):

    with computation(PARALLEL), interval(...):
        w1 = rc_mf
        w2 = ri_mf

        if w1 + w2 > rv_out[0, 0, 0]:
            w1 *= rv_tmp / (w1 + w2)
            w2 = rv_tmp - w1

        rc_tmp[0, 0, 0] += w1
        ri_tmp[0, 0, 0] += w2
        rv_tmp[0, 0, 0] -= w1 + w2
        t_tmp += (w1 * lv + w2 * ls) / cph

        # TODO :  remove unused out variables
        rv_out[0, 0, 0] = rv_tmp[0, 0, 0]
        ri_out[0, 0, 0] = ri_tmp[0, 0, 0]
        rc_out[0, 0, 0] = rc_tmp[0, 0, 0]
        th_out[0, 0, 0] = t_tmp[0, 0, 0] / exn[0, 0, 0]
