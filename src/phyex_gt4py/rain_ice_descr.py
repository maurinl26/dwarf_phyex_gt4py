# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from math import gamma
from typing import List, Tuple

import numpy as np
from phyex_gt4py.config import dtype_float, dtype_int

from phyex_gt4py.constants import Constants
from phyex_gt4py.param_ice import ParamIce


@dataclass
class RainIceDescr:

    cst: Constants
    parami: ParamIce

    cexvt: dtype_float = 0.4  # Air density fall speed correction

    rtmin: np.ndarray = field(init=False)  # Min values allowed for mixing ratios

    # Cloud droplet charact.
    ac: dtype_float = field(init=False)
    bc: dtype_float = 3.0
    cc: dtype_float = field(init=False)
    dc: dtype_float = 2.0

    # Rain drop charact
    ar: dtype_float = field(init=False)
    br: dtype_float = field(default=3.0)
    cr: dtype_float = field(default=842)
    dr: dtype_float = field(default=0.8)
    ccr: dtype_float = field(default=8e-6)
    f0r: dtype_float = field(default=1.0)
    f1r: dtype_float = field(default=0.26)
    c1r: dtype_float = field(default=0.5)

    # Cloud ice charact.
    ai: dtype_float = field(init=False)
    bi: dtype_float = field(init=False)
    c_i: dtype_float = field(init=False)
    di: dtype_float = field(init=False)
    f0i: dtype_float = field(default=1.00)
    f2i: dtype_float = field(default=0.14)
    c1i: dtype_float = field(init=False)

    # Snow/agg charact.
    a_s: dtype_float = field(default=0.02)
    bs: dtype_float = field(default=1.9)
    cs: dtype_float = field(default=5.1)
    ds: dtype_float = field(default=0.27)
    ccs: dtype_float = field(default=5.0)  # not lsnow
    cxs: dtype_float = field(default=1.0)
    f0s: dtype_float = field(default=0.86)
    f1s: dtype_float = field(default=0.28)
    c1s: dtype_float = field(init=False)

    # Graupel charact.
    ag: dtype_float = 19.6
    bg: dtype_float = 2.8
    cg: dtype_float = 124
    dg: dtype_float = 0.66
    ccg: dtype_float = 5e5
    cxg: dtype_float = -0.5
    f0g: dtype_float = 0.86
    f1g: dtype_float = 0.28
    c1g: dtype_float = 1 / 2

    # Hail charact.
    ah: dtype_float = 470
    bh: dtype_float = 3.0
    ch: dtype_float = 207
    dh: dtype_float = 0.64
    cch: dtype_float = 4e4
    cxh: dtype_float = -1.0
    f0h: dtype_float = 0.86
    f1h: dtype_float = 0.28
    c1h: dtype_float = 1 / 2

    # Cloud droplet distribution parameters

    # Over land
    alphac: dtype_float = (
        1.0  # Gamma law of the Cloud droplet (here volume-like distribution)
    )
    nuc: dtype_float = 3.0  # Gamma law with little dispersion

    # Over sea
    alphac2: dtype_float = 1.0
    nuc2: dtype_float = 1.0

    lbexc: dtype_float = field(init=False)
    lbc: Tuple[dtype_float] = field(init=False)

    # Rain drop distribution parameters
    alphar: dtype_float = (
        3.0  # Gamma law of the Cloud droplet (here volume-like distribution)
    )
    nur: dtype_float = 1.0  # Gamma law with little dispersion
    lbexr: dtype_float = field(init=False)
    lbr: dtype_float = field(init=False)

    # Cloud ice distribution parameters
    alphai: dtype_float = 1.0  # Exponential law
    nui: dtype_float = 1.0  # Exponential law
    lbexi: dtype_float = field(init=False)
    lbi: dtype_float = field(init=False)

    # Snow/agg. distribution parameters
    alphas: dtype_float = field(default=1.0)
    nus: dtype_float = field(default=1.0)
    lbexs: dtype_float = field(init=False)
    lbs: dtype_float = field(init=False)
    ns: dtype_float = field(init=False)

    # Graupel distribution parameters
    alphag: dtype_float = 1.0
    nug: dtype_float = 1.0
    lbexg: dtype_float = field(init=False)
    lbg: dtype_float = field(init=False)

    # Hail distribution parameters
    alphah: dtype_float = 1.0
    nuh: dtype_float = 8.0
    lbexh: dtype_float = field(init=False)
    lbh: dtype_float = field(init=False)

    fvelos: dtype_float = field(
        default=0.097
    )  # factor for snow fall speed after Thompson
    trans_mp_gammas: dtype_float = field(
        init=False
    )  # coefficient to convert lambda for gamma functions
    lbdar_max: dtype_float = (
        1e5  # Max values allowed for the shape parameters (rain,snow,graupeln)
    )
    lbdas_max: dtype_float = 1e5
    lbdag_max: dtype_float = 1e5
    lbdas_min: dtype_float = field(init=False)

    rtmin: List[dtype_float]  # min value allowed for mixing ratios
    conc_sea: dtype_float = 1e8  # Diagnostic concentration of droplets over sea
    conc_land: dtype_float = 3e8  # Diagnostic concentration of droplets over land
    conc_urban: dtype_float = (
        5e8  # Diagnostic concentration of droplets over urban area
    )

    def __post_init__(self):
        # 2.2    Ice crystal characteristics
        if self.parami.pristine_ice == "PLAT":
            self.ai = 0.82
            self.bi = 2.5
            self.c_i = 800
            self.di = 1.0
            self.c1i = 1 / self.cst.pi

        elif self.parami.pristine_ice == "COLU":
            self.ai = 2.14e-3
            self.bi = 1.7
            self.c_i = 2.1e5
            self.di = 1.585
            self.c1i = 0.8

        elif self.parami.pristine_ice == "BURO":
            self.ai = 44.0
            self.bi = 3.0
            self.c_i = 4.3e5
            self.di = 1.663
            self.c1i = 0.5

        if self.parami.lsnow_t:
            self.cs = 5.1
            self.ds = 0.27
            self.fvelos = 25.14

            self.alphas = 0.214
            self.nus = 43.7

        self.c1s = 1 / self.cst.pi

        self.lbexc = 1 / self.bc
        self.lbexr = 1 / (-1 - self.br)
        self.lbexi = 1 / -self.bi
        self.lbexs = 1 / (self.cxs - self.bs)
        self.lbexg = 1 / (self.cxg - self.bg)
        self.lbexh = 1 / (self.cxh - self.bh)

        # 3.4 Constant for shape parameter
        momg = lambda alpha, nu, p: gamma(nu + p / alpha) / gamma(nu)

        gamc = momg(self.alphac, self.nuc, 3)
        gamc2 = momg(self.alphac2, self.nuc2, 3)
        self.lbc[0] = self.ar * gamc
        self.lbc[1] = self.ar * gamc2

        self.lbr = (self.ar * self.ccr * momg(self.alphar, self.nur, self.br)) ** (
            -self.lbexr
        )
        self.lbi = (self.ai * self.cci * momg(self.alphai, self.nui, self.bi)) ** (
            -self.lbexi
        )
        self.lbs = (self.a_s * self.ccs * momg(self.alphas, self.nus, self.bs)) ** (
            -self.lbexs
        )
        self.lbg = (self.ag * self.ccg * momg(self.alphag, self.nug, self.bg)) ** (
            -self.lbexg
        )
        self.lbh = (self.ah * self.cch * momg(self.alphah, self.nuh, self.bh)) ** (
            -self.lbexh
        )


@dataclass
class CloudPar:
    """Declaration of the model-n dependant Microphysic constants

    Args:
        nsplitr (int): Number of required small time step integration
            for rain sedimentation computation
        nsplitg (int): Number of required small time step integration
            for ice hydrometeor sedimentation computation

    """

    nsplitr: dtype_int
    nsplitg: dtype_int
