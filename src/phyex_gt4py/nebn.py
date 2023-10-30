from dataclasses import dataclass
from config import dtype


@dataclass
class Neb:
    
    tminmix: dtype          # minimum temperature for mixed phase
    tmaxmix: dtype          # maximum temperature for mixed phase
    hgt_qs: dtype           # switch for height dependant VQSIGSAT
    frac_ice_adjust: str    # ice fraction for adjustments
    frac_ice_shallow: str   # ice fraction for shallow_mf
    vsigqsat: dtype         # coeff applied to qsat variance contribution
    condens: str            # subgrid condensation PDF
    lambda3: str            # lambda3 choice for subgrid cloud scheme
    statnw: bool            # updated full statistical cloud scheme
    sigmas: bool            # switch for using sigma_s from turbulence scheme
    subg_cond: bool         # switch for subgrid condensation
    
    