from gt4py.cartesian.gtscript import Field
from ifs_physics_common.framework.stencil import stencil_collection

@stencil_collection("droplet_subgrid_autoconversion")
def droplet_subgrid_autoconversions(
    cf_mf: Field["float"],
    hlc_hrc: Field["float"],
    hlc_hcf: Field["float"],
    w1: Field["float"]
):
    from __externals__ import criautc, subg_mf_pdf, tstep

    with computation(PARALLEL), interval(...):
        criaut = criautc / rhodref[0, 0, 0]

        if subg_mf_pdf == 0:
            if w1 * tstep > cf_mf[0, 0, 0] * criaut:
                hlc_hrc += w1 * tstep
                hlc_hcf = min(1, hlc_hcf[0, 0, 0] + cf_mf[0, 0, 0])

        elif subg_mf_pdf == 1:
            if w1 * tstep > cf_mf[0, 0, 0] * criaut:
                hcf = 1 - 0.5 * (criaut * cf_mf[0, 0, 0]) / max(1e-20, w1 * tstep)
                hr = w1 * tstep - (criaut * cf_mf[0, 0, 0]) ** 3 / (
                    3 * max(1e-20, w1 * tstep)
                )

            elif 2 * w1 * tstep <= cf_mf[0, 0, 0] * criaut:
                hcf = 0
                hr = 0

            else:
                hcf = (2 * w1 * tstep - criaut * cf_mf[0, 0, 0]) ** 2 / (
                    2.0 * max(1.0e-20, w1 * tstep) ** 2
                )
                hr = (
                    4.0 * (w1 * tstep) ** 3
                    - 3.0 * w1 * tstep * (criaut * cf_mf[0, 0, 0]) ** 2
                    + (criaut * cf_mf[0, 0, 0]) ** 3
                ) / (3 * max(1.0e-20, w1 * tstep) ** 2)

            hcf *= cf_mf[0, 0, 0]
            hlc_hcf = min(1, hlc_hcf + hcf)
            hlc_hrc += hr


@stencil_collection("ice_subgrid_autoconversion")
def ice_subgrid_autoconversions(
    cf_mf: Field["float"],
    hli_hri: Field["float"],
    hli_hcf: Field["float"],
    w2: Field["float"],
    t_tmp: Field["float"]
):
    
    
    from __externals__ import criauti, subg_mf_pdf, tt, acriauti, bcriauti, tstep

    with computation(PARALLEL), interval(...):
        criaut = min(
            criauti,
            10 ** (acriauti * (t_tmp[0, 0, 0] - tt) + bcriauti),
        )

        if subg_mf_pdf == 0:
            if w2 * tstep > cf_mf[0, 0, 0] * criaut:
                hli_hri += w2 * tstep
                hli_hcf = min(1, hli_hcf[0, 0, 0] + cf_mf[0, 0, 0])

        elif subg_mf_pdf == 1:
            if w2 * tstep > cf_mf[0, 0, 0] * criaut:
                hli_hcf = 1 - 0.5 * (criaut * cf_mf[0, 0, 0]) / max(1e-20, w2 * tstep)
                hli_hri = w2 * tstep - (criaut * cf_mf[0, 0, 0]) ** 3 / (
                    3 * max(1e-20, w2 * tstep)
                )

        elif 2 * w2 * tstep <= cf_mf[0, 0, 0] * criaut:
            hli_hcf = 0
            hli_hri = 0

        else:
            hli_hcf = (2 * w2 * tstep - criaut * cf_mf[0, 0, 0]) ** 2 / (
                2.0 * max(1.0e-20, w2 * tstep) ** 2
            )
            hli_hri = (
                4.0 * (w2 * tstep) ** 3
                - 3.0 * w2 * tstep * (criaut * cf_mf[0, 0, 0]) ** 2
                + (criaut * cf_mf[0, 0, 0]) ** 3
            ) / (3 * max(1.0e-20, w2 * tstep) ** 2)

        hli_hcf *= cf_mf[0, 0, 0]
        hli_hcf = min(1, hli_hcf + hli_hcf)
        hli_hri += hli_hri
