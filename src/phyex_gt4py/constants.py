from dataclasses import dataclass
import numpy as np
from config import dtype

@dataclass
class Constants:
    
    # 1. Fondamental constants
    pi: dtype
    karman: dtype
    lightspeed: dtype
    planck: dtype
    boltz: dtype
    avogadro: dtype
    
    # 2. Astronomical constants
    day: dtype         # day duration
    siyea: dtype       # sideral year duration
    siday: dtype       # sideral day duration
    nsday: np.int64         # number of seconds in a day
    omega: dtype       # earth rotation
    
    # 3. Terrestrial geoide constants
    radius: dtype      # earth radius
    gravity0: dtype    # gravity constant 
    
    # 4. Reference pressure
    p00: dtype         # Reference pressure
    p00ocean: dtype    # Ref pressure for ocean model
    rho0ocean: dtype   # Ref density for ocean model
    th00: dtype        # Ref value for potential temperature
    th00ocean: dtype   # Ref value for pot temp in ocean model
    sa00ocean: dtype   # Ref value for salinity in ocean model
    
    # 5. Radiation constants
    stefan: dtype      # Stefan-Boltzman constant
    io: dtype          # Solar constant
    
    # 6. Thermodynamic constants
    Md: dtype          # Molar mass of dry air
    Mv: dtype          # Molar mass of water vapour
    Rd: dtype          # Gas constant for dry air
    Rv: dtype          # Gas constant for vapour
    epsilo: dtype      # Mv / Md
    cpd: dtype         # Cpd (dry air)
    cpv: dtype         # Cpv (vapour)
    rholw: dtype       # Volumic mass of liquid water
    Cl: dtype          # Cl (liquid)
    Ci: dtype          # Ci (ice)
    tt: dtype          # triple point temperature
    lvtt: dtype        # vaporisation heat constant
    lstt: dtype        # sublimation heat constant
    lmtt: dtype        # melting heat constant
    estt: dtype        # Saturation vapor pressure at triple point temperature
    
    alpw: dtype        # Constants for saturation vapor pressure function
    betaw: dtype
    gamw: dtype
    
    alpi: dtype        # Constants for saturation vapor pressure function over solid ice
    betai: dtype
    gami: dtype
    
    condi: dtype       # Thermal conductivity of ice (W m-1 K-1)
    alphaoc: dtype     # Thermal expansion coefficient for ocean (K-1)
    betaoc: dtype      # Haline contraction coeff for ocean (S-1)
    roc: dtype = 0.69  # coeff for SW penetration in ocean (Hoecker et al)
    d1: dtype = 1.1    # coeff for SW penetration in ocean (Hoecker et al)
    d2: dtype = 23.0   # coeff for SW penetration in ocean (Hoecker et al)
    
    rholi: dtype       # Volumic mass of ice
    
    # 7. Precomputed constants
    Rd_Rv: dtype       # Rd / Rv
    Rd_cpd: dtype      # Rd / cpd
    invxp00: dtype     # 1 / p00
    
    # 8. Machine precision
    mnh_tiny: dtype    # minimum real on this machine
    mnh_tiny_12: dtype # sqrt(minimum real on this machine)
    mnh_epsilon: dtype # minimum space with 1.0
    mnh_huge: dtype    # minimum real on this machine
    mnh_huge_12_log: dtype # maximum log(sqrt(real)) on this machine
    eps_dt: dtype      # default value for dt
    res_flat_cart: dtype   # default     flat&cart residual tolerance
    res_other: dtype   # default not flat&cart residual tolerance
    res_prep: dtype    # default     prep      residual tolerance           
    

    def __post_init__(self):
        self.Rd_Rv = self.Rd / self.Rv
        self.Rd_cpd = self.Rd / self.cpd
        self.invxp00 = 1 / self.p00
    