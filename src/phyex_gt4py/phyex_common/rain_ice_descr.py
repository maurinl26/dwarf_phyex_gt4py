# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from math import gamma
from typing import List, Tuple

import numpy as np
from phyex_gt4py.phyex_common.constants import Constants
from phyex_gt4py.phyex_common.param_ice import ParamIce


@dataclass
class RainIceDescr:

    cst: Constants
    parami: ParamIce

    cexvt: "float" = 0.4  # Air density fall speed correction

    rtmin: np.ndarray = field(init=False)  # Min values allowed for mixing ratios

    # Cloud droplet charact.
    ac: "float" = field(init=False)
    bc: "float" = 3.0
    cc: "float" = field(init=False)
    dc: "float" = 2.0

    # Rain drop charact
    ar: "float" = field(init=False)
    br: "float" = field(default=3.0)
    cr: "float" = field(default=842)
    dr: "float" = field(default=0.8)
    ccr: "float" = field(default=8e-6)
    f0r: "float" = field(default=1.0)
    f1r: "float" = field(default=0.26)
    c1r: "float" = field(default=0.5)

    # Cloud ice charact.
    ai: "float" = field(init=False)
    bi: "float" = field(init=False)
    c_i: "float" = field(init=False)
    di: "float" = field(init=False)
    f0i: "float" = field(default=1.00)
    f2i: "float" = field(default=0.14)
    c1i: "float" = field(init=False)

    # Snow/agg charact.
    a_s: "float" = field(default=0.02)
    bs: "float" = field(default=1.9)
    cs: "float" = field(default=5.1)
    ds: "float" = field(default=0.27)
    ccs: "float" = field(default=5.0)  # not lsnow
    cxs: "float" = field(default=1.0)
    f0s: "float" = field(default=0.86)
    f1s: "float" = field(default=0.28)
    c1s: "float" = field(init=False)

    # Graupel charact.
    ag: "float" = 19.6
    bg: "float" = 2.8
    cg: "float" = 124
    dg: "float" = 0.66
    ccg: "float" = 5e5
    cxg: "float" = -0.5
    f0g: "float" = 0.86
    f1g: "float" = 0.28
    c1g: "float" = 1 / 2

    # Hail charact.
    ah: "float" = 470
    bh: "float" = 3.0
    ch: "float" = 207
    dh: "float" = 0.64
    cch: "float" = 4e4
    cxh: "float" = -1.0
    f0h: "float" = 0.86
    f1h: "float" = 0.28
    c1h: "float" = 1 / 2

    # Cloud droplet distribution parameters

    # Over land
    alphac: "float" = (
        1.0  # Gamma law of the Cloud droplet (here volume-like distribution)
    )
    nuc: "float" = 3.0  # Gamma law with little dispersion

    # Over sea
    alphac2: "float" = 1.0
    nuc2: "float" = 1.0

    lbexc: "float" = field(init=False)
    lbc: Tuple["float"] = field(init=False)

    # Rain drop distribution parameters
    alphar: "float" = (
        3.0  # Gamma law of the Cloud droplet (here volume-like distribution)
    )
    nur: "float" = 1.0  # Gamma law with little dispersion
    lbexr: "float" = field(init=False)
    lbr: "float" = field(init=False)

    # Cloud ice distribution parameters
    alphai: "float" = 1.0  # Exponential law
    nui: "float" = 1.0  # Exponential law
    lbexi: "float" = field(init=False)
    lbi: "float" = field(init=False)

    # Snow/agg. distribution parameters
    alphas: "float" = field(default=1.0)
    nus: "float" = field(default=1.0)
    lbexs: "float" = field(init=False)
    lbs: "float" = field(init=False)
    ns: "float" = field(init=False)

    # Graupel distribution parameters
    alphag: "float" = 1.0
    nug: "float" = 1.0
    lbexg: "float" = field(init=False)
    lbg: "float" = field(init=False)

    # Hail distribution parameters
    alphah: "float" = 1.0
    nuh: "float" = 8.0
    lbexh: "float" = field(init=False)
    lbh: "float" = field(init=False)

    fvelos: "float" = field(default=0.097)  # factor for snow fall speed after Thompson
    trans_mp_gammas: "float" = field(
        init=False
    )  # coefficient to convert lambda for gamma functions
    lbdar_max: "float" = (
        1e5  # Max values allowed for the shape parameters (rain,snow,graupeln)
    )
    lbdas_max: "float" = 1e5
    lbdag_max: "float" = 1e5
    lbdas_min: "float" = field(init=False)

    rtmin: List["float"]  # min value allowed for mixing ratios
    conc_sea: "float" = 1e8  # Diagnostic concentration of droplets over sea
    conc_land: "float" = 3e8  # Diagnostic concentration of droplets over land
    conc_urban: "float" = 5e8  # Diagnostic concentration of droplets over urban area

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

    nsplitr: "int"
    nsplitg: "int"
