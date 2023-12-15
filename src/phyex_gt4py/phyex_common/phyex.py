# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Literal, Tuple

from phyex_gt4py.phyex_common.constants import Constants
from phyex_gt4py.phyex_common.nebn import Neb
from phyex_gt4py.phyex_common.rain_ice_param import ParamIce, RainIceDescr, RainIceParam
from ifs_physics_common.utils.f2py import ported_class


@ported_class(from_file="PHYEX/src/common/aux/modd_phyex.F90")
@dataclass
class Phyex:
    """Data class for physics parametrizations

    Args:
        cst (Constants): Physical constants description
        param_icen (ParamIce): Control parameters for microphysics
        rain_ice_descrn (RainIceDescr): Microphysical descriptive constants
        rain_ice_paramn (RainIceParam): Microphysical factors

        tstep (float): time step employed for physics
        itermax (int): number of iterations for ice adjust

        lmfconv (bool): use convective mass flux in the condensation scheme
        compute_src (bool): compute s'r'
        khalo (int): size of the halo for parallel distribution (in turb)
        program (str): Name of the model
        nomixlg (bool): turbulence for lagrangian variables
        ocean (bool): ocean version of the turbulent scheme
        couples (bool): ocean atmo LES interactive coupling
        blowsnow (bool): blowsnow
        rsnow (float): blowing factor
        lbcx (Tuple[str]): boundary conditions
        lbcy (Tuple[str]): boundary conditions
        ibm (bool): run with ibm$
        flyer (bool): MesoNH flyer diagnostic
        diag_in_run (bool): LES diagnostics
        o2d (bool): 2D version of turbulence
    """

    program: Literal["AROME", "MESO-NH"]
    timestep: float = field(default=1)

    cst: Constants = field(init=False)
    param_icen: ParamIce = field(init=False)
    rain_ice_descrn: RainIceDescr = field(init=False)
    rain_ice_paramn: RainIceParam = field(init=False)
    nebn: Neb = field(init=False)

    itermax: int = field(default=1)

    # Miscellaneous terms
    lmfconv: bool = field(default=True)
    compute_src: bool = field(default=True)
    khalo: int = field(default=1)
    program: str = field(default="AROME")
    nomixlg: bool = field(default=False)
    ocean: bool = field(default=False)
    deepoc: bool = field(default=False)
    couples: bool = field(default=False)
    blowsnow: bool = field(default=False)
    rsnow: float = field(default=1.0)
    lbcx: Tuple[str] = field(default=("CYCL", "CYCL"))
    lbcy: Tuple[str] = field(default=("CYCL", "CYCL"))
    ibm: bool = field(default=False)
    flyer: bool = field(default=False)
    diag_in_run: bool = field(default=False)
    o2d: bool = field(default=False)

    # flat: bool
    # tbuconf: TBudgetConf

    def __post_init__(self):
        self.cst = Constants()
        self.param_icen = ParamIce(hprogram=self.program)
        self.nebn = Neb(hprogram=self.program)
        self.rain_ice_descrn = RainIceDescr(self.cst, self.param_icen)
        self.rain_ice_paramn = RainIceParam(self.cst, self.param_icen)
