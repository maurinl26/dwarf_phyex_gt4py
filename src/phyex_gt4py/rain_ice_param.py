from dataclasses import dataclass, field
from math import gamma, log
from typing import List, Tuple

import numpy as np
from config import dtype_float, dtype_int

from phyex_gt4py.constants import Constants
from phyex_gt4py.param_ice import ParamIce
from phyex_gt4py.rain_ice_descr import RainIceDescr


@dataclass
class RainIceParam:
    fsedc: Tuple[dtype_float] = field(
        init=False
    )  # Constants for sedimentation fluxes of C
    fsedr: dtype_float = field(init=False)  # Constants for sedimentation
    exsedr: dtype_float = field(init=False)
    fsedi: dtype_float = field(init=False)
    excsedi: dtype_float = field(init=False)
    exrsedi: dtype_float = field(init=False)
    fseds: dtype_float = field(init=False)
    exseds: dtype_float = field(init=False)
    fsedg: dtype_float = field(init=False)
    exsedg: dtype_float = field(init=False)

    # Constants for heterogeneous ice nucleation HEN
    nu10: dtype_float = field(init=False)
    alpha1: dtype_float = 4.5
    beta1: dtype_float = 0.6
    nu20: dtype_float = field(init=False)
    alpha2: dtype_float = 12.96
    beta2: dtype_float = 0.639
    mnu0: dtype_float = 6.88e-13  # Mass of nucleated ice crystal

    # Constants for homogeneous ice nucleation HON
    alpha3: dtype_float = -3.075
    beta3: dtype_float = 81.00356
    hon: dtype_float = field(init=False)

    # Constants for raindrop and evaporation EVA
    scfac: dtype_float = field(init=False)
    o0evar: dtype_float = field(init=False)
    o1evar: dtype_float = field(init=False)
    ex0evar: dtype_float = field(init=False)
    ex1evar: dtype_float = field(init=False)
    o0depi: dtype_float = field(init=False)  # deposition DEP on I
    o2depi: dtype_float = field(init=False)
    o0deps: dtype_float = field(init=False)  # on S
    o1deps: dtype_float = field(init=False)
    ex0deps: dtype_float = field(init=False)
    ex1deps: dtype_float = field(init=False)
    rdepsred: dtype_float = field(init=False)
    o0depg: dtype_float = field(init=False)  # on G
    o1depg: dtype_float = field(init=False)
    ex0depg: dtype_float = field(init=False)
    ex1depg: dtype_float = field(init=False)
    rdepgred: dtype_float = field(init=False)

    # Constants for pristine ice autoconversion : AUT
    timauti: dtype_float = 1e-3  # Time constant at T=T_t
    texauti: dtype_float = 0.015
    criauti: dtype_float = field(init=False)
    t0criauti: dtype_float = field(init=False)
    acriauti: dtype_float = field(init=False)
    bcriauti: dtype_float = field(init=False)

    # Constants for snow aggregation : AGG
    colis: dtype_float = 0.25  # Collection efficiency of I + S
    colexis: dtype_float = 0.05  # Temperature factor of the I+S collection efficiency
    fiaggs: dtype_float = field(init=False)
    exiaggs: dtype_float = field(init=False)

    # Constants for cloud droplet autoconversion AUT
    timautc: dtype_float = 1e-3
    criautc: dtype_float

    # Constants for cloud droplets accretion on raindrops : ACC
    fcaccr: dtype_float = field(init=False)
    excaccr: dtype_float = field(init=False)

    # Constants for the riming of the aggregates : RIM
    dcslim: dtype_float = 0.007
    colcs: dtype_float = 1.0
    excrimss: dtype_float = field(init=False)
    crimss: dtype_float = field(init=False)
    excrimsg: dtype_float = field(init=False)
    crimsg: dtype_float = field(init=False)

    excrimsg: dtype_float = field(init=False)
    crimsg: dtype_float = field(init=False)
    exsrimcg: dtype_float = field(init=False)
    crimcg: dtype_float = field(init=False)
    exsrimcg2: dtype_float = field(init=False)
    rimcg2: dtype_float = field(init=False)
    srimcg3: dtype_float = field(init=False)

    gaminc_bound_min: dtype_float = field(init=False)
    gaminc_bound_max: dtype_float = field(init=False)
    rimintp1: dtype_float = field(init=False)
    rimintp2: dtype_float = field(init=False)

    ngaminc: dtype_int = field(init=False)  # Number of tab. Lbda_s

    def __post_init__(self, rid: RainIceDescr, cst: Constants, parami: ParamIce):
        # 4. CONSTANTS FOR THE SEDIMENTATION
        # 4.1 Exponent of the fall-speed air density correction

        e = 0.5 * np.exp(cst.alpw - cst.betaw / 293.15 - cst.gamw * log(293.15))
        rv = (cst.Rd_Rv) * e / (101325 - e)
        rho00 = 101325 * (1 + rv) / (cst.Rd + rv * cst.Rv) / 293.15

        # 4.2    Constants for sedimentation
        self.fsedc[0] = (
            gamma(rid.nuc + (rid.dc + 3) / rid.alphac)
            / gamma(rid.nuc + 3 / rid.alphac)
            * rho00**rid.cexvt
        )
        self.fsedc[1] = (
            gamma(rid.nuc2 + (rid.dc + 3) / rid.alphac2)
            / gamma(rid.nuc2 + 3 / rid.alphac2)
            * rho00**rid.cexvt
        )

        momg = lambda alpha, nu, p: gamma(nu + p / alpha) / gamma(nu)

        self.exrsedr = (rid.br + rid.rd + 1.0) / (rid.br + 1.0)
        self.fsedr = (
            rid.cr
            + rid.ar
            + rid.ccr
            * momg(rid.alphar, rid.nur, rid.br)
            * (
                rid.ar
                * rid.ccr
                * momg(rid.alphar, rid.nur, rid.br) ** (-self.exsedr)
                * rho00**rid.cexvt
            )
        )

        self.exrsedi = (rid.bi + rid.di) / rid.bi
        self.excsedi = 1 - self.exrsedi
        self.fsedi = (
            (4 * 900 * cst.pi) ** (-self.excsedi)
            * rid.c_i
            * rid.ai
            * rid.cci
            * momg(rid.alphai, rid.nui, rid.bi + rid.di)
            * (
                (rid.ai * momg(rid.alphai, rid.nui, rid.bi)) ** (-self.exrsedi)
                * rho00**rid.cexvt
            )
        )

        self.exseds = (rid.bs + rid.ds - rid.cxs) / (rid.bs - rid.cxs)
        self.seds = (
            rid.cs
            * rid.a_s
            * rid.ccs
            * momg(rid.alphas, rid.nus, rid.bs + rid.ds)
            * (rid.a_s * rid.ccs * momg(rid.alphas, rid.nus, rid.bs)) ** (-self.exseds)
            * rho00**rid.cexvt
        )

        if parami.lred:
            self.exseds = rid.ds - rid.bs
            self.fseds = (
                self.cs
                * momg(rid.alphas, rid.nus, rid.bs + rid.ds)
                / momg(rid.alphas, rid.nus, rid.bs)
                * rho00**rid.cexvt
            )

        self.exsedg = (rid.bg + rid.dg - rid.cxg) / (rid.bg - rid.cxg)
        self.fsedg = (
            rid.cg
            * rid.ag
            * rid.ccg
            * momg(rid.alphag, rid.nug, rid.bg + rid.dg)
            * (rid.ag * rid.ccg * momg(rid.alphag, rid.nug, rid.bg)) ** (-self.exsedg)
            * rho00
            * rid.cexvt
        )

        self.exsedh = (rid.bh + rid.dh - rid.cxh) / (rid.bh - rid.cxh)
        self.fsedh = (
            rid.ch
            * rid.ah
            * rid.cch
            * momg(rid.alphah, rid.nuh, rid.bh + rid.dh)
            * (rid.ah * rid.cch * momg(rid.alphah, rid.nuh, rid.bh)) ** (-self.exsedh)
            * rho00
            * rid.cexvt
        )

        # 5. Constants for the skow cold processes
        fact_nucl = 0
        if parami.pristine_ice == "PLAT":
            fact_nucl = 1.0  # Plates
        elif parami.pristine_ice == "COLU":
            fact_nucl = 25.0  # Columns
        elif parami.pristine_ice == "BURO":
            fact_nucl = 17.0  # Bullet rosettes

        self.nu10 = 50 * fact_nucl
        self.nu20 = 1000 * fact_nucl

        self.hon = (cst.pi / 6) * ((2 * 3 * 4 * 5 * 6) / (2 * 3)) * (1.1e5) ** (-3)

        # 5.2 Constants for vapor deposition on ice
        self.scfac = (0.63 ** (1 / 3)) * np.sqrt((rho00) ** rid.cexvt)
        self.o0depi = (4 * cst.pi) * rid.c1i * rid.f0i * momg(rid.alphai, rid.nui, 1)
        self.o2depi = (
            (4 * cst.pi)
            * rid.c1i
            * rid.f2i
            * rid.c_i
            * momg(rid.alpjai, rid.nui, rid.di + 2.0)
        )

        # TODO : add ifdef case
        self.o0deps = (
            rid.ns
            * (4 * cst.pi)
            * rid.ccs
            * rid.c1s
            * rid.f0s
            * momg(rid.alphas, rid.nus, 1)
        )
        self.o1deps = (
            rid.ns
            * (4 * cst.pi)
            * rid.ccs
            * rid.c1s
            * rid.f1s
            * np.sqrt(rid.cs)
            * momg(rid.alphas, rid.nus, 0.5 * rid.ds + 1.5)
        )
        self.ex0deps = rid.cxs - 1.0
        self.ex1deps = rid.cxs - 0.5 * (rid.ds + 3.0)
        self.rdepsred = parami.rdepsred_nam

        self.o0depg = (
            (4 * cst.pi) * rid.ccg * rid.c1g * rid.f0g * momg(rid.alphag, rid.nug, 1)
        )
        self.o1depg = (
            (4 * cst.pi)
            * rid.ccg
            * rid.c1g
            * rid.f1g
            * np.sqrt(rid.cg)
            * momg(rid.alphag, rid.nug, 0.5 * rid.dg + 1.5)
        )
        self.ex0depg = rid.cxg - 1.0
        self.ex1depg = rid.cxg - 0.5 * (rid.dg + 3.0)
        self.rdepgred = parami.rdepgred_nam

        self.o0deph = (
            (4 * cst.pi) * rid.cch * rid.c1h * rid.f0h * momg(rid.alphah, rid.nuh, 1)
        )
        self.o1deph = (
            (4 * cst.pi)
            * rid.cch
            * rid.c1h
            * rid.f1h
            * np.sqrt(rid.ch)
            * momg(rid.alphah, rid.nuh, 0.5 * rid.dh + 1.5)
        )
        self.ex0deph = rid.cxh - 1.0
        self.ex1deph = rid.cxh - 0.5 * (rid.dh + 3.0)

        # 5.3 Constants for pristine ice autoconversion
        self.criauti = parami.criauti_nam
        if parami.lcriauti:
            self.t0criauti = parami.t0criauti_nam
            tcrio = -40
            crio = 1.25e-6
            self.bcriauti = (
                -(np.log10(self.criauti) - np.log10(crio) * self.t0criauti / tcrio)
                * tcrio
                / (self.t0criauti - tcrio)
            )
            self.acriauti = (np.log10(crio) - self.bcriauti) / tcrio

        else:
            self.acriauti = parami.acriauti_nam
            self.bcriauti = parami.brcriauti_nam
            self.t0criauti = (np.log10(self.criauti) - self.bcriauti) / 0.06

        # 5.4 Constants for snow aggregation
        self.fiaggs = (
            (cst.pi / 4)
            * self.colis
            * rid.ccs
            * rid.cs
            * (rho00**rid.cexvt)
            * momg(rid.alphas, rid.nus, rid.ds + 2.0)
        )
        self.exiaggs = rid.cxs - rid.ds - 2.0

        # TODO: ifdef case

        # 6. Constants for the slow warm processes
        # 6.1 Constants for the accretion of cloud droplets autoconversion
        self.criautc = parami.criautc_nam

        # 6.2 Constants for the accretion of cloud droplets by raindrops
        self.fcaccr = (
            (cst.pi / 4)
            * rid.ccr
            * rid.cr
            * (rho00**rid.cexvt)
            * momg(rid.alphar, rid.nur, rid.dr + 2.0)
        )
        self.excaccr = -rid.dr - 3.0

        # 6.3 Constants for the evaporation of rain drops
        self.o0evar = (
            (4.0 * cst.pi) * rid.ccr * rid.cr * rid.f0r * momg(rid.alphar, rid.nur, 1)
        )
        self.o1evar = (
            (4.0 * cst.pi)
            * rid.ccr
            * rid.c1r
            * rid.f1r
            * momg(rid.alphar, rid.nur, 0.5 * rid.dr + 1.5)
        )
        self.ex0evar = -2.0
        self.ex1evar = -1.0 - 0.5 * (rid.dr + 3.0)

        # 7. Constants for the fast cold processes for the aggregateds
        # 7.1 Constants for the riming of the aggregates

        # TODO :  2 ifdef
        self.excrimss = -rid.ds - 2.0
        self.crimss = (
            rid.ns
            * (cst.pi / 4)
            * self.colcs
            * rid.cs
            * (rho00**rid.cexvt)
            * momg(rid.alphas, rid.nus, rid.ds + 2)
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

    nsplitr: dtype_int
    nsplitg: dtype_int
