from dataclasses import dataclass, field
from enum import Enum

from config import dtype, dtype_int

from phyex_gt4py.budget import TBudgetConf
from phyex_gt4py.constants import Constants
from phyex_gt4py.nebn import Neb
from phyex_gt4py.rain_ice_param import CloudPar, ParamIce, RainIceDescr, RainIceParam


class VerticalLevelOrder(Enum):
    """Specify order of index on vertical levels

    SPACE_TO_GROUND for AROME-like indexing
    GROUND_TO_PACE for Meso-NH like indexing
    """

    SPACE_TO_GROUND = -1
    GROUND_TO_SPACE = 1


@dataclass
class DIMPhyex:
    """Specify dimensions and index for Phyex"""

    # x dimension
    nit: int  # Array dim

    nib: int = field(init=False)  # First index
    nie: int = field(init=False)  # Last index

    # y dimension
    njt: int
    njb: int = field(init=False)
    nje: int = field(init=False)

    # z dimension
    vertical_level_order: VerticalLevelOrder

    # TODO: remove nkl (FORTRAN implementation) to use VerticalLevelOrder
    nkl: int  # Order of the vertical levels
    # 1 : Meso NH order (bottom to top)
    # -1 : AROME order (top to bottom)

    nkt: int  # Array total dimension on z (nz)
    nkles: int  # Total physical k dimension

    nka: int  # Near ground array index
    nku: int  # Uppest atmosphere array index

    nkb: int  # Near ground physical array index
    nke: int  # Uppest atmosphere physical array index

    nktb: int  # smaller index for the physical domain
    nkte: int  # greater index for the physical domain

    nibc: int
    njbc: int
    niec: int
    nijt: int = field(init=False)  # horizontal packing
    nijb: int = field(init=False)  # first index for horizontal packing
    nije: int = field(init=False)  # last index for horizontal packing

    def __post_init__(self):
        self.nib, self.nie = 0, self.nit - 1  # python like indexing
        self.njb, self.nje = 0, self.njt - 1

        self.nijt = self.nit * self.njt
        self.nijb, self.nije = 0, self.nijt - 1
