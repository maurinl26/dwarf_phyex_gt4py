# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from ifs_physics_common.utils.f2py import ported_class


class FracIceAdjust(Enum):
    T = 0
    O = 1
    N = 2
    S = 3


@ported_class(from_file="PHYEX/src/common/aux/modd_nebn.F90")
@dataclass
class Neb:
    """Declaration of

    Args:
        tminmix (float): minimum temperature for mixed phase
        tmaxmix (float): maximum temperature for mixed phase
        hgt_qs (float): switch for height dependant VQSIGSAT
        frac_ice_adjust (str): ice fraction for adjustments
        frac_ice_shallow (str): ice fraction for shallow_mf
        vsigqsat (float): coeff applied to qsat variance contribution
        condens (str): subgrid condensation PDF
        lambda3 (str): lambda3 choice for subgrid cloud scheme
        statnw (bool): updated full statistical cloud scheme
        sigmas (bool): switch for using sigma_s from turbulence scheme
        subg_cond (bool): switch for subgrid condensation

    """

    hprogram: Literal["AROME", "MESO-NH", "LMDZ"]

    tminmix: float = field(default=273.16)  # minimum temperature for mixed phase
    tmaxmix: float = field(default=253.16)  # maximum temperature for mixed phase
    hgt_qs: float = field(default=False)  # switch for height dependant VQSIGSAT
    frac_ice_adjust: FracIceAdjust = field(default="S")  # ice fraction for adjustments
    frac_ice_shallow: str = field(default="S")  # ice fraction for shallow_mf
    vsigqsat: float = field(default=0.02)  # coeff applied to qsat variance contribution
    condens: str = field(default="CB02")  # subgrid condensation PDF
    lambda3: str = field(default="CB")  # lambda3 choice for subgrid cloud scheme
    statnw: bool = field(default=False)  # updated full statistical cloud scheme
    sigmas: bool = field(
        default=True
    )  # switch for using sigma_s from turbulence scheme
    subg_cond: bool = field(default=False)  # switch for subgrid condensation

    def __post_init__(self):
        if self.hprogram == "AROME":
            self.frac_ice_adjust = "S"
            self.frac_ice_shallow = "T"
            self.vsigqsat = 0
            self.sigmas = False

        elif self.hprogram == "LMDZ":
            self.subg_cond = True
