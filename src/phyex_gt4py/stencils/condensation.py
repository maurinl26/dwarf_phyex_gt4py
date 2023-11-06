from typing import Optional

from config import backend, dtype_float, dtype_int
from gt4py.cartesian import IJ, K, gtscript

from phyex_gt4py.constants import Constants
from phyex_gt4py.functions import compute_ice_frac
from phyex_gt4py.functions.erf import erf
from phyex_gt4py.functions.ice_adjust import _cph, latent_heat
from phyex_gt4py.functions.icecloud import icecloud
from phyex_gt4py.functions.temperature import update_temperature
from phyex_gt4py.functions.tiwmx import esati, esatw
from phyex_gt4py.nebn import Neb
from phyex_gt4py.rain_ice_param import ParamIce, RainIceDescr, RainIceParam


@gtscript.stencil(backend=backend)
def condensation(
    cst: Constants,
    nebn: Neb,
    icep: RainIceParam,  # formulation for lambda3 coeff
    parami: ParamIce,
    lmfconv: bool,  # True if mfconv.size != 0
    pabs: gtscript.Field[dtype_float],  # pressure (Pa)
    zz: gtscript.Field[dtype_float],  # height of model levels
    rhodref: gtscript.Field[dtype_float],
    t: gtscript.Field[dtype_float],  # T (K)
    rv_in: gtscript.Field[dtype_float],
    rc_in: gtscript.Field[dtype_float],
    ri_in: gtscript.Field[dtype_float],
    rv_out: gtscript.Field[dtype_float],
    rc_out: gtscript.Field[dtype_float],
    ri_out: gtscript.Field[dtype_float],
    rs: gtscript.Field[dtype_float],  # grid scale mixing ratio of snow (kg/kg)
    rr: gtscript.Field[dtype_float],  # grid scale mixing ratio of rain (kg/kg)
    rg: gtscript.Field[dtype_float],  # grid scale mixing ratio of graupel (kg/kg)
    sigs: gtscript.Field[dtype_float],  # Sigma_s from turbulence scheme
    mfconv: gtscript.Field[dtype_float],  # convective mass flux (kg.s⁻1.m⁻2)
    prifact: gtscript.Field[dtype_float],
    cldfr: gtscript.Field[dtype_float],
    sigrc: gtscript.Field[dtype_float],  # s r_c / sig_s ** 2
    icldfr: gtscript.Field[dtype_float],  # ice cloud fraction
    wcldfr: gtscript.Field[dtype_float],  # water por mixed-phase cloud fraction
    ls: Optional[gtscript.Field[dtype_float]],
    lv: Optional[gtscript.Field[dtype_float]],
    cph: Optional[gtscript.Field[dtype_float]],
    ifr: gtscript.Field[dtype_float],  # ratio cloud ice moist part
    sigqsat: gtscript.Field[
        dtype_float
    ],  # use an extra qsat variance contribution (if osigma is True)
    ssio: gtscript.Field[
        dtype_float
    ],  # super-saturation with respect to ice in the super saturated fraction
    ssiu: gtscript.Field[
        dtype_float
    ],  # super-saturation with respect to in in the sub saturated fraction
    hlc_hrc: Optional[gtscript.Field[dtype_float]],  #
    hlc_hcf: Optional[gtscript.Field[dtype_float]],  # cloud fraction
    hli_hri: Optional[gtscript.Field[dtype_float]],  #
    hli_hcf: Optional[gtscript.Field[dtype_float]],
    ice_cld_wgt: gtscript.Field[dtype_float],  # in
    # Temporary fields
    cpd: gtscript.Field[dtype_float],
    tlk: gtscript.Field[dtype_float],  # working array for T_l
    rt: gtscript.Field[dtype_float],  # work array for total water mixing ratio
    pv: gtscript.Field[dtype_float],  # thermodynamics
    piv: gtscript.Field[dtype_float],  # thermodynamics
    qsl: gtscript.Field[dtype_float],  # thermodynamics
    qsi: gtscript.Field[dtype_float],
    t_tropo: gtscript.Field[IJ, dtype_float],  # temperature at tropopause
    z_tropo: gtscript.Field[IJ, dtype_float],  # height at tropopause
    z_ground: gtscript.Field[IJ, dtype_float],  # height at ground level (orography)
    l: gtscript.Field[dtype_float],  # length scale
    frac_tmp: gtscript.Field[IJ, dtype_float],  # ice fraction
    cond_tmp: gtscript.Field[IJ, dtype_float],  # condensate
    a: gtscript.Field[IJ, dtype_float],  # related to computation of Sig_s
    b: gtscript.Field[IJ, dtype_float],
    sbar: gtscript.Field[IJ, dtype_float],
    sigma: gtscript.Field[IJ, dtype_float],
    q1: gtscript.Field[IJ, dtype_float],
    # related to ocnd2 ice cloud calculation
    esatw_t: gtscript.Field[IJ, dtype_float],
    ardum: gtscript.Field[IJ, dtype_float],
    ardum2: gtscript.Field[IJ, dtype_float],
    dz: gtscript.Field[IJ, dtype_float],  # Layer thickness
    cldini: gtscript.Field[IJ, dtype_float],  # To be initialized for icecloud
    dum4: gtscript.Field[dtype_float],
    lwinc: gtscript.Field[IJ, dtype_float],
    # related to ocnd2 noise check
    rsp: dtype_float,
    rsw: dtype_float,
    rfrac: dtype_float,
    rsdif: dtype_float,
    rcold: dtype_float,
    dzfact: dtype_float,  # lhgt_qs
    dzref: dtype_float,
    inq1: dtype_int,
    inc: dtype_float,
    ouseri: bool,  # switch to compute both liquid and solid condensate (True) or only solid condensate (False)
    csigma: dtype_float = 0.2,  # constant in sigma_s parametrization
    csig_conv: dtype_float = 3e-3,  # scaling factor for ZSIG_CONV as function of mass flux
    l0: dtype_float = 600,
):
    """_summary_

    Args:
        cst (Constants): physical constants
        nebn (Neb): parameters related to
        icep (RainIceParam): _description_
        pabs (gtscript.Field[dtype_float]): _description_
        t (gtscript.Field[dtype_float]): _description_
        rc_in (gtscript.Field[dtype_float]): _description_
        ri_in (gtscript.Field[dtype_float]): _description_
        rv_out (gtscript.Field[dtype_float]): _description_
        rc_out (gtscript.Field[dtype_float]): _description_
        ri_out (gtscript.Field[dtype_float]): _description_
        rs (gtscript.Field[dtype_float]): _description_
        cldfr (gtscript.Field[dtype_float]): _description_
        sigrc (gtscript.Field[dtype_float]): _description_
        lv (Optional[gtscript.Field[dtype_float]]): _description_
        cph (Optional[gtscript.Field[dtype_float]]): _description_
        ifr (gtscript.Field[dtype_float]): _description_
        ice_cld_wgt (gtscript.Field[dtype_float]): _description_
        tlk (gtscript.Field[dtype_float]): _description_
        t_tropo (gtscript.Field[IJ, dtype_float]): _description_
        sbar (gtscript.Field[IJ, dtype_float]): _description_
        sigma (gtscript.Field[IJ, dtype_float]): _description_
        q1 (gtscript.Field[IJ, dtype_float]): _description_
        esatw_t (gtscript.Field[IJ, dtype_float]): _description_
        ardum (gtscript.Field[IJ, dtype_float]): _description_
        ardum2 (gtscript.Field[IJ, dtype_float]): _description_
        dz (gtscript.Field[IJ, dtype_float]): layer thickness
        lwinc (gtscript.Field[IJ, dtype_float]): _description_
        rsp (dtype_float): _description_
        rsw (dtype_float): _description_
        rfrac (dtype_float): _description_
        rsdif (dtype_float): _description_
        rcold (dtype_float): _description_
        dzfact (dtype_float): _description_
        inq1 (dtype_int): _description_
        inc (dtype_float): _description_
        ouseri (bool): _description_
    """

    src_1d = [
        0.0,
        0.0,
        2.0094444e-04,
        0.316670e-03,
        4.9965648e-04,
        0.785956e-03,
        1.2341294e-03,
        0.193327e-02,
        3.0190963e-03,
        0.470144e-02,
        7.2950651e-03,
        0.112759e-01,
        1.7350994e-02,
        0.265640e-01,
        4.0427860e-02,
        0.610997e-01,
        9.1578111e-02,
        0.135888e00,
        0.1991484,
        0.230756e00,
        0.2850565,
        0.375050e00,
        0.5000000,
        0.691489e00,
        0.8413813,
        0.933222e00,
        0.9772662,
        0.993797e00,
        0.9986521,
        0.999768e00,
        0.9999684,
        0.999997e00,
        1.0000000,
        1.000000,
    ]
    dzref = icep.frmin[25]
    prifact = 0 if parami.cnd2 else 1

    # Initialize values
    with computation(PARALLEL), interval(...):
        cldfr[0, 0, 0] = 0
        sigrc[0, 0, 0] = 0
        rv_out[0, 0, 0] = 0
        rc_out[0, 0, 0] = 0
        ri_out[0, 0, 0] = 0

        # local fields
        ardum2[0, 0] = 0
        cldini[0, 0, 0] = 0
        ifr[0, 0, 0] = 10
        frac_tmp[0, 0] = 0

        rt[0, 0, 0] = rv_in + rc_in + ri_in * prifact

        if ls is None and lv is None:
            lv, ls = latent_heat(cst, t)

        if cph is None:
            cpd = _cph(cst, rv_in, rc_in, ri_in, rr, rs, rg)

    # Preliminary calculations for computing the turbulent part of Sigma_s
    if not nebn.sigmas:
        with computation(PARALLEL), interval(...):
            # Temperature at saturation
            tlk = t[0, 0, 0] - lv * rc_in / cpd - ls * ri_in / cpd * prifact

        # tropopause height computation
        with computation(BACKWARD):
            t_tropo[0, 0] = 400
            while t_tropo[0, 0] > t[0, 0, 0]:
                t_tropo[0, 0] = t[0, 0, 0]
                z_tropo[0, 0] = zz[0, 0, 0]

        # length scale computation
        # ground to top
        with computation(BACKWARD):
            with interval(0, 1):
                l[0, 0, 0] = 20

            with interval(1, None):
                zz_to_ground = zz[0, 0, 0] - z_ground[0, 0]

                # approximate length for boundary layer
                if zz_to_ground > l0:
                    l[0, 0, 0] = zz_to_ground
                # gradual decrease of length-scale near and above tropopause
                if zz_to_ground > 0.9 * (z_tropo[0, 0] - z_ground[0, 0]):
                    l[0, 0, 0] = 0.6 * l[0, 0, -1]
                # free troposphere
                else:
                    l[0, 0, 0] = l0

    # line 313
    if parami.lcond2:
        with computation(FORWARD):
            with interval(0, 1):
                dz = zz[0, 0, 0] - zz[0, 0, 1]

            with interval(1, None):
                dz = zz[0, 0, 1] - zz[0, 0, 0]

        with computation(FORWARD), interval(...):
            icecloud(
                cst=cst,
                p=pabs,
                z=zz,
                dz=dz,
                t=t,
                r=rv_in,
                tstep=1,
                pblh=-1,
                wcld=cldini,
                sifrc=ifr,
                ssio=ssio,
                ssiu=ssiu,
                w2_out=ardum2,
                rsi=ardum,
            )

        with computation(PARALLEL), interval(...):
            esatw_t[0, 0] = esatw(t[0, 0, 0])
            pv[0, 0] = min(esatw_t[0, 0], 0.99 * pabs[0, 0, 0])
            piv[0, 0] = min(esati(t[0, 0, 0]), 0.99 * pabs[0, 0, 0])

    else:
        with computation(PARALLEL), interval(...):
            pv[0, 0] = min(
                exp(cst.alpw - cst.betaw / t[0, 0, 0] - cst.gamw * log(t[0, 0, 0])),
                0.99 * pabs[0, 0, 0],
            )
            piv[0, 0] = min(
                exp(cst.alpi - cst.betai / t[0, 0, 0]) - cst.gami * log(t[0, 0, 0]),
                0.99 * pabs[0, 0, 0],
            )

    if ouseri and not parami.lcond2:
        with computation(PARALLEL), interval(...):
            if rc_in[0, 0, 0] > ri_in[0, 0, 0] > 1e-20:
                frac_tmp[0, 0] = ri_in[0, 0, 0] / (rc_in[0, 0, 0] + ri_in[0, 0, 0])

            compute_ice_frac(nebn.frac_ice_adjust, nebn, frac_tmp, t)

            qsl[0, 0] = cst.Rd / cst.Rv * pv[0, 0] / (pabs[0, 0, 0] - pv[0, 0])
            qsi[0, 0] = cst.Rd / cst.Rv * piv[0, 0] / (pabs[0, 0, 0] - piv[0, 0])

            # dtype_interpolate bewteen liquid and solid as a function of temperature
            qsl = (1 - frac_tmp) * qsl + frac_tmp * qsi
            lvs = (1 - frac_tmp) * lv + frac_tmp * ls

            # coefficients a et b
            ah = lvs * qsl / (cst.Rv * t[0, 0, 0] ** 2) * (1 + cst.Rv * qsl / cst.Rd)
            a = 1 / (1 + lvs / cpd[0, 0, 0] * ah)
            b = ah * a
            sbar = a * (
                rt[0, 0, 0] - qsl[0, 0] + ah * lvs * (rc_in + ri_in * prifact) / cpd
            )

    # Meso-NH turbulence scheme
    if nebn.sigmas:
        with computation(PARALLEL):
            if sigqsat[0, 0, 0] != 0:
                # dtype_intialization
                if nebn.statnw:
                    if nebn.hgt_qs:
                        with interval(0, -1):
                            dzfact = max(
                                icep.frmin[23],
                                min(
                                    icep.frmin[24], (zz[0, 0, 0] - zz[0, 0, 1]) / dzref
                                ),
                            )
                        with interval(-1, -2):
                            dzfact = max(
                                icep.frmin[23],
                                min(
                                    icep.frmin[24],
                                    (zz[0, 0, 0] - zz[0, 0, 1]) * 0.8 / dzref,
                                ),
                            )
                    else:
                        with interval(...):
                            dzfact = 1

                    with interval(...):
                        sigma = sqrt(sigs**2 + (sigqsat * dzfact * qsl * a) ** 2)
                else:
                    sigma = sqrt((2 * sigs) ** 2 + (sigqsat * qsl * a) ** 2)

            else:
                with interval(...):
                    sigma = sigs[0, 0, 0] if nebn.statnw else 2 * sigs[0, 0, 0]

    else:
        # not nebn osigmas
        # Parametrize Sigma_s with first order closure
        with computation(FORWARD):
            with interval(0, 1):
                dzz = zz[0, 0, 1] - zz[0, 0, 0]
                drw = rt[0, 0, 1] - rt[0, 0, 0]
                dtl = tlk[0, 0, 1] - tlk[0, 0, 0] + cst.gravity0 / cpd[0, 0, 0] * dzz

            with interval(1, -1):
                dzz = zz[0, 0, 1] - zz[0, 0, -1]
                drw = rt[0, 0, 1] - rt[0, 0, -1]
                dtl = tlk[0, 0, 1] - tlk[0, 0, -1] + cst.gravity0 / cpd[0, 0, 0] * dzz

            with interval(-1, -2):
                dzz = zz[0, 0, 0] - zz[0, 0, -1]
                drw = rt[0, 0, 0] - rt[0, 0, -1]
                dtl = tlk[0, 0, 0] - tlk[0, 0, -1] + cst.gravity0 / cpd[0, 0, 0] * dzz

            with interval(...):
                # Standard deviation due to convection
                sig_conv = csig_conv * mfconv[0, 0, 0] / a[0, 0] if lmfconv else 0
                sigma[0, 0] = sqrt(
                    max(
                        1e-25,
                        (csigma * l / zz * a * drw) ** 2
                        - 2 * a * b * drw * dtl
                        + (b * dtl) ** 2
                        + sig_conv**2,
                    )
                )

    with computation(PARALLEL), interval(...):
        sigma[0, 0] = max(1e-10, sigma[0, 0])

        # normalized saturation deficit
        q1[0, 0] = sbar[0, 0] / sigma[0, 0]

    if nebn.condens == "GAUS":
        with computation(PARALLEL), interval(...):
            # Gaussian probability density function around q1
            # computation of g and gam(=erf(g))
            gcond, gauv = erf(cst, q1)

            # computation of cloud fraction (output)
            cldfr[0, 0, 0] = max(0, min(1, 0.5 * gauv))

            # computation of condensate
            cond_tmp[0, 0] = (
                (exp(-(gcond**2)) - gcond * sqrt(cst.pi) * gauv)
                * sigma[0, 0]
                / sqrt(2 * cst.pi)
            )
            cond_tmp[0, 0] = max(cond_tmp[0, 0], 0)

            sigrc[0, 0, 0] = cldfr[0, 0, 0]

            # Computation warm/cold cloud fraction and content in high water
            if hlc_hcf is not None and hlc_hrc is not None:
                if 1 - frac_tmp > 1e-20:
                    autc = (
                        sbar[0, 0]
                        - icep.criautc / (rhodref[0, 0, 0] * (1 - frac_tmp[0, 0]))
                    ) / sigma[0, 0]

                    gautc, gauc = erf(
                        cst, autc
                    )  # approximation of erf function for Gaussian distribution

                    hlc_hcf[0, 0, 0] = max(0, min(1, 0.5 * gauc))
                    hlc_hrc[0, 0, 0] = (
                        (1 - frac_tmp[0, 0])
                        * (exp(-(gautc**2)) - gautc * sqrt(cst.pi) * gauc)
                        * sigma[0, 0]
                        / sqrt(2 * cst.pi)
                    )
                    hlc_hrc[0, 0, 0] += (
                        icep.criautc / rhodref[0, 0, 0] * hlc_hcf[0, 0, 0]
                    )
                    hlc_hrc[0, 0, 0] = max(0, hlc_hrc[0, 0, 0])

                else:
                    hlc_hcf[0, 0, 0] = 0
                    hlc_hrc[0, 0, 0] = 0

            if hli_hcf is not None and hli_hri is not None:
                if frac_tmp[0, 0, 0] > 1e-20:
                    criauti = min(
                        icep.criauti,
                        10 ** (icep.acriauti * (t[0, 0, 0] - cst.tt) + icep.bcriauti),
                    )
                    auti = (sbar[0, 0] - criauti / frac_tmp[0, 0]) / sigma[0, 0]

                    gauti, gaui = erf(cst, auti)
                    hli_hcf = max(0, min(1, 0.5 * gaui))

                    hli_hri[0, 0, 0] = (
                        frac_tmp[0, 0]
                        * (exp(-(gauti**2)) - gauti * sqrt(cst.pi) * gaui)
                        * sigma[0, 0]
                        / sqrt(2 * cst.pi)
                    )
                    hli_hri[0, 0, 0] += criauti * hli_hcf[0, 0, 0]
                    hli_hri[0, 0, 0] = max(0, hli_hri[0, 0, 0])

                else:
                    hli_hcf[0, 0, 0] = 0
                    hli_hri[0, 0, 0] = 0

    elif nebn.condens == "CB02":
        with computation(PARALLEL), interval(...):
            if q1 > 0 and q1 <= 2:
                cond_tmp[0, 0] = min(
                    exp(-1) + 0.66 * q1[0, 0] + 0.086 * q1[0, 0] ** 2, 2
                )  # we use the MIN function for continuity
            elif q1 > 2:
                cond_tmp[0, 0] = q1
            else:
                cond_tmp[0, 0] = exp(1.2 * q1[0, 0] - 1)

            cond_tmp[0, 0] *= sigma[0, 0]

            # cloud fraction
            if cond_tmp[0, 0] < 1e-12:
                cldfr[0, 0, 0] = 0
            else:
                cldfr[0, 0, 0] = max(0, min(1, 0.5 + 0.36 * atan(1.55 * q1[0, 0])))

            if cldfr[0, 0, 0] == 0:
                cond_tmp[0, 0] = 0

            inq1 = min(
                10, max(-22, floor(min(-100, 2 * q1[0, 0])))
            )  # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an dtype_int
            inc = 2 * q1 - inq1

            sigrc[0, 0, 0] = min(
                1, (1 - inc) * src_1d[inq1 + 22] + inc * src_1d[inq1 + 1 + 22]
            )

            if hlc_hcf is not None and hlc_hrc is not None:
                hlc_hcf[0, 0, 0] = 0
                hlc_hrc[0, 0, 0] = 0

            if hli_hcf is not None and hli_hri is not None:
                hli_hcf[0, 0, 0] = 0
                hli_hri[0, 0, 0] = 0

    # ref -> line 515
    if not parami.lcond2:
        with computation(PARALLEL), interval(...):
            rc_out[0, 0, 0] = (1 - frac_tmp[0, 0]) * cond_tmp[0, 0]  # liquid condensate
            ri_out[0, 0, 0] = frac_tmp[0, 0] * cond_tmp[0, 0]  # solid condensate
            t[0, 0, 0] = update_temperature(
                t, rc_in, rc_out, ri_in, ri_out, lv, ls, cpd
            )
            rv_out[0, 0, 0] = rt[0, 0, 0] - rc_out[0, 0, 0] - ri_out[0, 0, 0] * prifact

    else:
        with computation(FORWARD):
            with interval(0, 1):
                dum4[0, 0, 0] = ri_in[0, 0, 0]
            with interval(1, None):
                dum4[0, 0, 0] = ri_in[0, 0, 0] + 0.5 * rs[0, 0, 0] + 0.25 * rg[0, 0, 0]

        with computation(PARALLEL), interval(...):
            rc_out[0, 0, 0] = (1 - frac_tmp[0, 0]) * cond_tmp[0, 0]

            lwinc = rc_out[0, 0, 0] - rc_in[0, 0, 0]
            if abs(lwinc) > 1e-12 and esatw(t[0, 0, 0]) < 0.5 * pabs[0, 0, 0]:
                rcold = rc_out[0, 0, 0]
                rfrac = rv_in[0, 0, 0] - lwinc

                rsdif = min(0, rsp - rfrac) if rc_in[0, 0, 0] < rsw else 0
                # sub saturation over water (True) or supersaturation over water (False)

                rc_out[0, 0, 0] = cond_tmp[0, 0] - rsdif

            else:
                rcold = rc_in[0, 0, 0]

            # Compute separate ice cloud
            wcldfr[0, 0, 0] = cldfr[0, 0, 0]

            dum1 = min(
                1, 20 * rc_out[0, 0, 0] * sqrt(dz[0, 0]) / qsi[0, 0]
            )  # cloud liquid factor
            dum3 = max(0, icldfr[0, 0, 0] - wcldfr[0, 0, 0])
            dum4[0, 0, 0] = max(
                0,
                min(1, ice_cld_wgt[0, 0] * dum4[0, 0, 0] * sqrt(dz[0, 0]) / qsi[0, 0]),
            )
            dum2 = (0.8 * cldfr[0, 0, 0] + 0.2) * min(1, dum1 + dum4 * cldfr[0, 0, 0])
            cldfr[0, 0, 0] = min(1, dum2 + (0.5 + 0.5 * dum3) * dum4)

            ri_out[0, 0, 0] = ri_in[0, 0, 0]

            # TODO : same as 341 with rcold instead of ri_in
            t[0, 0, 0] = update_temperature(
                t, rcold, rc_out, ri_in, ri_out, lv, ls, cpd
            )

            rv_out[0, 0, 0] = rt[0, 0, 0] - rc_out[0, 0, 0] - ri_out[0, 0, 0] * prifact

    if nebn.clambda3 == "CB":
        with computation(PARALLEL), interval(...):
            sigrc[0, 0, 0] *= min(3, max(1, 1 - q1[0, 0]))
