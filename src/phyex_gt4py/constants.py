from dataclasses import dataclass
import numpy as np

@dataclass
class Constants:
    
    # 1. Fondamental constants
    pi: np.float64
    karman: np.float64
    lightspeed: np.float64
    planck: np.float64
    boltz: np.float64
    avogadro: np.float64
    
    # 2. Astronomical constants
    day: np.float64         # day duration
    siyea: np.float64       # sideral year duration
    siday: np.float64       # sideral day duration
    nsday: np.int64         # number of seconds in a day
    omega: np.float64       # earth rotation
    
    # 3. Terrestrial geoide constants
    radius: np.float64      # earth radius
    gravity0: np.float64    # gravity constant 
    
    # 4. Reference pressure
    p00: np.float64         # Reference pressure
    p00ocean: np.float64    # Ref pressure for ocean model
    rho0ocean: np.float64   # Ref density for ocean model
    th00: np.float64        # Ref value for potential temperature
    th00ocean: np.float64   # Ref value for pot temp in ocean model
    sa00ocean: np.float64   # Ref value for salinity in ocean model
    
    # 5. Radiation constants
    stefan: np.float64      # Stefan-Boltzman constant
    io: np.float64          # Solar constant
    
    # 6. Thermodynamic constants
    Md: np.float64          # Molar mass of dry air
    Mv: np.float64          # Molar mass of water vapour
    Rd: np.float64          # Gas constant for dry air
    Rv: np.float64          # Gas constant for vapour
    epsilo: np.float64      # Mv / Md
    cpd: np.float64         # Cpd (dry air)
    cpv: np.float64         # Cpv (vapour)
    rholw: np.float64       # Volumic mass of liquid water
    Cl: np.float64          # Cl (liquid)
    Ci: np.float64          # Ci (ice)
    tt: np.float64          # triple point temperature
    lvtt: np.float64        # vaporisation heat constant
    lstt: np.float64        # sublimation heat constant
    lmtt: np.float64        # melting heat constant
    estt: np.float64        # Saturation vapor pressure at triple point temperature
    
    alpw: np.float64        # Constants for saturation vapor pressure function
    betaw: np.float64
    gamw: np.float64
    
    alpi: np.float64        # Constants for saturation vapor pressure function over solid ice
    betai: np.float64
    gami: np.float64
    
    condi: np.float64       # Thermal conductivity of ice (W m-1 K-1)
    alphaoc: np.float64     # Thermal expansion coefficient for ocean (K-1)
    betaoc: np.float64      # Haline contraction coeff for ocean (S-1)
    roc: np.float64 = 0.69  # coeff for SW penetration in ocean (Hoecker et al)
    d1: np.float64 = 1.1    # coeff for SW penetration in ocean (Hoecker et al)
    d2: np.float64 = 23.0   # coeff for SW penetration in ocean (Hoecker et al)
    
    rholi: np.float64       # Volumic mass of ice
    
    # 7. Precomputed constants
    Rd_Rv: np.float64       # Rd / Rv
    Rd_cpd: np.float64      # Rd / cpd
    invxp00: np.float64     # 1 / p00
    
    # 8. Machine precision
    mnh_tiny: np.float64    # minimum real on this machine
    mnh_tiny_12: np.float64 # sqrt(minimum real on this machine)
    mnh_epsilon: np.float64 # minimum space with 1.0
    mnh_huge: np.float64    # minimum real on this machine
    mnh_huge_12_log: np.float64 # maximum log(sqrt(real)) on this machine
    eps_dt: np.float64      # default value for dt
    res_flat_cart: np.float64   # default     flat&cart residual tolerance
    res_other: np.float64   # default not flat&cart residual tolerance
    res_prep: np.float64    # default     prep      residual tolerance           
    


    