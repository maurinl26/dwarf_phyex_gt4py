from dataclasses import dataclass, field

from config import dtype


@dataclass
class Neb:
    """Declaration of

    Args:
        tminmix (dtype): minimum temperature for mixed phase
        tmaxmix (dtype): maximum temperature for mixed phase
        hgt_qs (dtype): switch for height dependant VQSIGSAT
        frac_ice_adjust (str): ice fraction for adjustments
        frac_ice_shallow (str): ice fraction for shallow_mf
        vsigqsat (dtype): coeff applied to qsat variance contribution
        condens (str): subgrid condensation PDF
        lambda3 (str): lambda3 choice for subgrid cloud scheme
        statnw (bool): updated full statistical cloud scheme
        sigmas (bool): switch for using sigma_s from turbulence scheme
        subg_cond (bool): switch for subgrid condensation

    """

    tminmix: dtype = field(default=273.16)  # minimum temperature for mixed phase
    tmaxmix: dtype = field(default=253.16)  # maximum temperature for mixed phase
    hgt_qs: dtype = field(default=False)  # switch for height dependant VQSIGSAT
    frac_ice_adjust: str = field(default="S")  # ice fraction for adjustments
    frac_ice_shallow: str = field(default="S")  # ice fraction for shallow_mf
    vsigqsat: dtype = field(default=0.02)  # coeff applied to qsat variance contribution
    condens: str = field(default="CB02")  # subgrid condensation PDF
    lambda3: str = field(default="CB")  # lambda3 choice for subgrid cloud scheme
    statnw: bool = field(default=False)  # updated full statistical cloud scheme
    sigmas: bool = field(
        default=True
    )  # switch for using sigma_s from turbulence scheme
    subg_cond: bool = field(default=False)  # switch for subgrid condensation

    def __post_init__(self, hprogram: str):
        if hprogram == "AROME":
            self.frac_ice_adjust = "T"
            self.frac_ice_shallow = "T"
            self.vsigqsat = 0
            self.sigmas = False

        elif hprogram == "LMDZ":
            self.subg_cond = True
