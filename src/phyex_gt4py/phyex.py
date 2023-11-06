from dataclasses import dataclass, field

from config import dtype_float, dtype_int

from phyex_gt4py.budget import TBudgetConf
from phyex_gt4py.constants import Constants
from phyex_gt4py.nebn import Neb
from phyex_gt4py.rain_ice_param import ParamIce, RainIceDescr, RainIceParam


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

    cst: Constants
    param_icen: ParamIce
    rain_ice_descrn: RainIceDescr
    rain_ice_paramn: RainIceParam
    nebn: Neb

    tstep: dtype_float
    itermax: dtype_int = field(default=1)

    # Miscellaneous terms
    lmfconv: bool = field(default=True)
    compute_src: bool = field(default=True)
    khalo: dtype_int = field(default=1)
    program: str = field(default="AROME")
    nomixlg: bool = field(default=False)
    ocean: bool = field(default=False)
    deepoc: bool = field(default=False)
    couples: bool = field(default=False)
    blowsnow: bool = field(default=False)
    rsnow: dtype_float = field(default=1.0)
    lbcx: str = field(default="CYCL")
    lbcy: str = field(default="CYCL")
    ibm: bool = field(default=False)
    flyer: bool = field(default=False)
    diag_in_run: bool = field(default=False)
    o2d: bool = field(default=False)

    flat: bool
    tbuconf: TBudgetConf
