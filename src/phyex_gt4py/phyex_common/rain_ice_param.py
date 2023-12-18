# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from math import gamma, log
from typing import Tuple

import numpy as np

from phyex_gt4py.phyex_common.constants import Constants
from phyex_gt4py.phyex_common.param_ice import ParamIce
from phyex_gt4py.phyex_common.rain_ice_descr import RainIceDescr
from ifs_physics_common.utils.f2py import ported_class


@ported_class(from_file="PHYEX/src/common/aux/modd_rain_ice_paramn.F90")
@dataclass
class RainIceParam:

    # Constants dependencies
    cst: Constants
    rid: RainIceDescr
    parami: ParamIce

    """Parameters for microphysical sources and transformations"""
    fsedc: Tuple[float] = field(init=False)  # Constants for sedimentation fluxes of C
    fsedr: float = field(init=False)  # Constants for sedimentation
    exsedr: float = field(init=False)
    fsedi: float = field(init=False)
    excsedi: float = field(init=False)
    exrsedi: float = field(init=False)
    fseds: float = field(init=False)
    exseds: float = field(init=False)
    fsedg: float = field(init=False)
    exsedg: float = field(init=False)

    # Constants for heterogeneous ice nucleation HEN
    nu10: float = field(init=False)
    alpha1: float = 4.5
    beta1: float = 0.6
    nu20: float = field(init=False)
    alpha2: float = 12.96
    beta2: float = 0.639
    mnu0: float = 6.88e-13  # Mass of nucleated ice crystal

    # Constants for homogeneous ice nucleation HON
    alpha3: float = -3.075
    beta3: float = 81.00356
    hon: float = field(init=False)

    # Constants for raindrop and evaporation EVA
    scfac: float = field(init=False)
    o0evar: float = field(init=False)
    o1evar: float = field(init=False)
    ex0evar: float = field(init=False)
    ex1evar: float = field(init=False)
    o0depi: float = field(init=False)  # deposition DEP on I
    o2depi: float = field(init=False)
    o0deps: float = field(init=False)  # on S
    o1deps: float = field(init=False)
    ex0deps: float = field(init=False)
    ex1deps: float = field(init=False)
    rdepsred: float = field(init=False)
    o0depg: float = field(init=False)  # on G
    o1depg: float = field(init=False)
    ex0depg: float = field(init=False)
    ex1depg: float = field(init=False)
    rdepgred: float = field(init=False)

    # Constants for pristine ice autoconversion : AUT
    timauti: float = 1e-3  # Time constant at T=T_t
    texauti: float = 0.015
    criauti: float = field(init=False)
    t0criauti: float = field(init=False)
    acriauti: float = field(init=False)
    bcriauti: float = field(init=False)

    # Constants for snow aggregation : AGG
    colis: float = 0.25  # Collection efficiency of I + S
    colexis: float = 0.05  # Temperature factor of the I+S collection efficiency
    fiaggs: float = field(init=False)
    exiaggs: float = field(init=False)

    # Constants for cloud droplet autoconversion AUT
    timautc: float = 1e-3
    criautc: float = field(init=False)

    # Constants for cloud droplets accretion on raindrops : ACC
    fcaccr: float = field(init=False)
    excaccr: float = field(init=False)

    # Constants for the riming of the aggregates : RIM
    dcslim: float = 0.007
    colcs: float = 1.0
    excrimss: float = field(init=False)
    crimss: float = field(init=False)
    excrimsg: float = field(init=False)
    crimsg: float = field(init=False)

    excrimsg: float = field(init=False)
    crimsg: float = field(init=False)
    exsrimcg: float = field(init=False)
    crimcg: float = field(init=False)
    exsrimcg2: float = field(init=False)
    rimcg2: float = field(init=False)
    srimcg3: float = field(init=False)

    gaminc_bound_min: float = field(init=False)
    gaminc_bound_max: float = field(init=False)
    rimintp1: float = field(init=False)
    rimintp2: float = field(init=False)

    ngaminc: int = field(init=False)  # Number of tab. Lbda_s

    def __post_init__(self):
        # 4. CONSTANTS FOR THE SEDIMENTATION
        # 4.1 Exponent of the fall-speed air density correction

        e = 0.5 * np.exp(
            self.cst.alpw - self.cst.betaw / 293.15 - self.cst.gamw * log(293.15)
        )
        rv = (self.cst.Rd_Rv) * e / (101325 - e)
        rho00 = 101325 * (1 + rv) / (self.cst.Rd + rv * self.cst.Rv) / 293.15

        # 4.2    Constants for sedimentation
        self.fsedc[0] = (
            gamma(self.rid.nuc + (self.rid.dc + 3) / self.rid.alphac)
            / gamma(self.rid.nuc + 3 / self.rid.alphac)
            * rho00**self.rid.cexvt
        )
        self.fsedc[1] = (
            gamma(self.rid.nuc2 + (self.rid.dc + 3) / self.rid.alphac2)
            / gamma(self.rid.nuc2 + 3 / self.rid.alphac2)
            * rho00**self.rid.cexvt
        )

        momg = lambda alpha, nu, p: gamma(nu + p / alpha) / gamma(nu)

        self.exrsedr = (self.rid.br + self.rid.rd + 1.0) / (self.rid.br + 1.0)
        self.fsedr = (
            self.rid.cr
            + self.rid.ar
            + self.rid.ccr
            * momg(self.rid.alphar, self.rid.nur, self.rid.br)
            * (
                self.rid.ar
                * self.rid.ccr
                * momg(self.rid.alphar, self.rid.nur, self.rid.br) ** (-self.exsedr)
                * rho00**self.rid.cexvt
            )
        )

        self.exrsedi = (self.rid.bi + self.rid.di) / self.rid.bi
        self.excsedi = 1 - self.exrsedi
        self.fsedi = (
            (4 * 900 * self.cst.pi) ** (-self.excsedi)
            * self.rid.c_i
            * self.rid.ai
            * self.rid.cci
            * momg(self.rid.alphai, self.rid.nui, self.rid.bi + self.rid.di)
            * (
                (self.rid.ai * momg(self.rid.alphai, self.rid.nui, self.rid.bi))
                ** (-self.exrsedi)
                * rho00**self.rid.cexvt
            )
        )

        self.exseds = (self.rid.bs + self.rid.ds - self.rid.cxs) / (
            self.rid.bs - self.rid.cxs
        )
        self.seds = (
            self.rid.cs
            * self.rid.a_s
            * self.rid.ccs
            * momg(self.rid.alphas, self.rid.nus, self.rid.bs + self.rid.ds)
            * (
                self.rid.a_s
                * self.rid.ccs
                * momg(self.rid.alphas, self.rid.nus, self.rid.bs)
            )
            ** (-self.exseds)
            * rho00**self.rid.cexvt
        )

        if self.parami.lred:
            self.exseds = self.rid.ds - self.rid.bs
            self.fseds = (
                self.cs
                * momg(self.rid.alphas, self.rid.nus, self.rid.bs + self.rid.ds)
                / momg(self.rid.alphas, self.rid.nus, self.rid.bs)
                * rho00**self.rid.cexvt
            )

        self.exsedg = (self.rid.bg + self.rid.dg - self.rid.cxg) / (
            self.rid.bg - self.rid.cxg
        )
        self.fsedg = (
            self.rid.cg
            * self.rid.ag
            * self.rid.ccg
            * momg(self.rid.alphag, self.rid.nug, self.rid.bg + self.rid.dg)
            * (
                self.rid.ag
                * self.rid.ccg
                * momg(self.rid.alphag, self.rid.nug, self.rid.bg)
            )
            ** (-self.exsedg)
            * rho00
            * self.rid.cexvt
        )

        self.exsedh = (self.rid.bh + self.rid.dh - self.rid.cxh) / (
            self.rid.bh - self.rid.cxh
        )
        self.fsedh = (
            self.rid.ch
            * self.rid.ah
            * self.rid.cch
            * momg(self.rid.alphah, self.rid.nuh, self.rid.bh + self.rid.dh)
            * (
                self.rid.ah
                * self.rid.cch
                * momg(self.rid.alphah, self.rid.nuh, self.rid.bh)
            )
            ** (-self.exsedh)
            * rho00
            * self.rid.cexvt
        )

        # 5. Constants for the skow cold processes
        fact_nucl = 0
        if self.parami.pristine_ice == "PLAT":
            fact_nucl = 1.0  # Plates
        elif self.parami.pristine_ice == "COLU":
            fact_nucl = 25.0  # Columns
        elif self.parami.pristine_ice == "BURO":
            fact_nucl = 17.0  # Bullet rosettes

        self.nu10 = 50 * fact_nucl
        self.nu20 = 1000 * fact_nucl

        self.hon = (self.cst.pi / 6) * ((2 * 3 * 4 * 5 * 6) / (2 * 3)) * (1.1e5) ** (-3)

        # 5.2 Constants for vapor deposition on ice
        self.scfac = (0.63 ** (1 / 3)) * np.sqrt((rho00) ** self.rid.cexvt)
        self.o0depi = (
            (4 * self.cst.pi)
            * self.rid.c1i
            * self.rid.f0i
            * momg(self.rid.alphai, self.rid.nui, 1)
        )
        self.o2depi = (
            (4 * self.cst.pi)
            * self.rid.c1i
            * self.rid.f2i
            * self.rid.c_i
            * momg(self.rid.alpjai, self.rid.nui, self.rid.di + 2.0)
        )

        # TODO : add ifdef case
        self.o0deps = (
            self.rid.ns
            * (4 * self.cst.pi)
            * self.rid.ccs
            * self.rid.c1s
            * self.rid.f0s
            * momg(self.rid.alphas, self.rid.nus, 1)
        )
        self.o1deps = (
            self.rid.ns
            * (4 * self.cst.pi)
            * self.rid.ccs
            * self.rid.c1s
            * self.rid.f1s
            * np.sqrt(self.rid.cs)
            * momg(self.rid.alphas, self.rid.nus, 0.5 * self.rid.ds + 1.5)
        )
        self.ex0deps = self.rid.cxs - 1.0
        self.ex1deps = self.rid.cxs - 0.5 * (self.rid.ds + 3.0)
        self.rdepsred = self.parami.rdepsred_nam

        self.o0depg = (
            (4 * self.cst.pi)
            * self.rid.ccg
            * self.rid.c1g
            * self.rid.f0g
            * momg(self.rid.alphag, self.rid.nug, 1)
        )
        self.o1depg = (
            (4 * self.cst.pi)
            * self.rid.ccg
            * self.rid.c1g
            * self.rid.f1g
            * np.sqrt(self.rid.cg)
            * momg(self.rid.alphag, self.rid.nug, 0.5 * self.rid.dg + 1.5)
        )
        self.ex0depg = self.rid.cxg - 1.0
        self.ex1depg = self.rid.cxg - 0.5 * (self.rid.dg + 3.0)
        self.rdepgred = self.parami.rdepgred_nam

        self.o0deph = (
            (4 * self.cst.pi)
            * self.rid.cch
            * self.rid.c1h
            * self.rid.f0h
            * momg(self.rid.alphah, self.rid.nuh, 1)
        )
        self.o1deph = (
            (4 * self.cst.pi)
            * self.rid.cch
            * self.rid.c1h
            * self.rid.f1h
            * np.sqrt(self.rid.ch)
            * momg(self.rid.alphah, self.rid.nuh, 0.5 * self.rid.dh + 1.5)
        )
        self.ex0deph = self.rid.cxh - 1.0
        self.ex1deph = self.rid.cxh - 0.5 * (self.rid.dh + 3.0)

        # 5.3 Constants for pristine ice autoconversion
        self.criauti = self.parami.criauti_nam
        if self.parami.lcriauti:
            self.t0criauti = self.parami.t0criauti_nam
            tcrio = -40
            crio = 1.25e-6
            self.bcriauti = (
                -(np.log10(self.criauti) - np.log10(crio) * self.t0criauti / tcrio)
                * tcrio
                / (self.t0criauti - tcrio)
            )
            self.acriauti = (np.log10(crio) - self.bcriauti) / tcrio

        else:
            self.acriauti = self.parami.acriauti_nam
            self.bcriauti = self.parami.brcriauti_nam
            self.t0criauti = (np.log10(self.criauti) - self.bcriauti) / 0.06

        # 5.4 Constants for snow aggregation
        self.fiaggs = (
            (self.cst.pi / 4)
            * self.colis
            * self.rid.ccs
            * self.rid.cs
            * (rho00**self.rid.cexvt)
            * momg(self.rid.alphas, self.rid.nus, self.rid.ds + 2.0)
        )
        self.exiaggs = self.rid.cxs - self.rid.ds - 2.0

        # TODO: ifdef case

        # 6. Constants for the slow warm processes
        # 6.1 Constants for the accretion of cloud droplets autoconversion
        self.criautc = self.parami.criautc_nam

        # 6.2 Constants for the accretion of cloud droplets by raindrops
        self.fcaccr = (
            (self.cst.pi / 4)
            * self.rid.ccr
            * self.rid.cr
            * (rho00**self.rid.cexvt)
            * momg(self.rid.alphar, self.rid.nur, self.rid.dr + 2.0)
        )
        self.excaccr = -self.rid.dr - 3.0

        # 6.3 Constants for the evaporation of rain drops
        self.o0evar = (
            (4.0 * self.cst.pi)
            * self.rid.ccr
            * self.rid.cr
            * self.rid.f0r
            * momg(self.rid.alphar, self.rid.nur, 1)
        )
        self.o1evar = (
            (4.0 * self.cst.pi)
            * self.rid.ccr
            * self.rid.c1r
            * self.rid.f1r
            * momg(self.rid.alphar, self.rid.nur, 0.5 * self.rid.dr + 1.5)
        )
        self.ex0evar = -2.0
        self.ex1evar = -1.0 - 0.5 * (self.rid.dr + 3.0)

        # 7. Constants for the fast cold processes for the aggregateds
        # 7.1 Constants for the riming of the aggregates

        # TODO :  2 ifdef
        self.excrimss = -self.rid.ds - 2.0
        self.crimss = (
            self.rid.ns
            * (self.cst.pi / 4)
            * self.colcs
            * self.rid.cs
            * (rho00**self.rid.cexvt)
            * momg(self.rid.alphas, self.rid.nus, self.rid.ds + 2)
        )

        self.excrimsg = self.excrimss
        self.crimsg = self.crimsg


@dataclass
class CloudPar:
    """Declaration of the model-n dependant Microphysic constants

    Args:
        nsplitr (int): Number of required small time step integration
            for rain sedimentation computation
        nsplitg (int): Number of required small time step integration
            for ice hydrometeor sedimentation computation

    """

    nsplitr: int
    nsplitg: int
