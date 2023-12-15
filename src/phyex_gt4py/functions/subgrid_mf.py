# -*- coding: utf-8 -*-
from typing import Tuple

from gt4py.cartesian.gtscript import Field, function


@function
def subgrid_mf(
    criaut: Field["float"],
    subg_mf_pdf: Field["float"],
    hl_hr: Field["float"],
    hl_hc: Field["float"],
    cf_mf: Field["float"],
    w: Field["float"],
    tstep: float,
) -> Tuple[Field["float"]]:
    """Compute subgrid mass fluxes

    Args:
        criaut (Field[float]): _description_
        subg_mf_pdf (Field[float]): _description_
        hl_hr (Field[float]): _description_
        hl_hc (Field[float]): _description_
        cf_mf (Field[float]): _description_
        w (Field[float]): _description_
        tstep (float): time step

    Returns:
        _type_: _description_
    """

    if subg_mf_pdf == "NONE":
        if w * tstep > cf_mf[0, 0, 0] * criaut:
            hl_hr += w * tstep
            hl_hc = min(1, hl_hc[0, 0, 0] + cf_mf[0, 0, 0])

    elif subg_mf_pdf == "TRIANGLE":
        if w * tstep > cf_mf[0, 0, 0] * criaut:
            hcf = 1 - 0.5 * (criaut * cf_mf[0, 0, 0]) / max(1e-20, w * tstep)
            hr = w * tstep - (criaut * cf_mf[0, 0, 0]) ** 3 / (
                3 * max(1e-20, w * tstep)
            )

        elif 2 * w * tstep <= cf_mf[0, 0, 0] * criaut:
            hcf = 0
            hr = 0

        else:
            hcf = (2 * w * tstep - criaut * cf_mf[0, 0, 0]) ** 2 / (
                2.0 * max(1.0e-20, w * tstep) ** 2
            )
            hr = (
                4.0 * (w * tstep) ** 3
                - 3.0 * w * tstep * (criaut * cf_mf[0, 0, 0]) ** 2
                + (criaut * cf_mf[0, 0, 0]) ** 3
            ) / (3 * max(1.0e-20, w * tstep) ** 2)

        hcf *= cf_mf[0, 0, 0]
        hl_hc = min(1, hl_hc + hcf)
        hl_hr += hr

    return hl_hr, hl_hc, w
