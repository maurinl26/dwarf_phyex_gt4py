from dataclasses import dataclass, field

import numpy as np
from config import dtype, dtype_int


@dataclass
class Constants:
    """Data class for physical constants

    Args:

        # 1. Fondamental constants
        pi (float):
        karman (float):
        lightspeed (float):
        planck (float):
        boltz (float):
        avogadro (float):

        # 2. Astronomical constants
        day (float): day duration
        siyea (float): sideral year duration
        siday (float): sidearl day duration
        nsday (int): number of seconds in a day
        omega (flaot): earth rotation

        # 3. Terrestrial geoide constants
        radius: (float): earth radius
        gravity0: (float): gravity constant

        # 4. Reference pressure
        # Ocean model cst same as 1D/CMO SURFEX
        p00ocean: (float)  Ref pressure for ocean model
        rho0ocean: (float) Ref density for ocean model
        th00ocean: (float) Ref value for pot temp in ocean model
        sa00ocean: (float) Ref value for salinity in ocean model

        # Atmospheric model
        p00: (float) Reference pressure
        th00: (float) Ref value for potential temperature

        # 5. Radiation constants
        stefan: (float) Stefan-Boltzman constant
        io: (float) Solar constant

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

    """

    # 1. Fondamental constants
    pi: dtype = field(default=2 * np.arcsin(1.0))
    karman: dtype = field(default=0.4)
    lightspeed: dtype = field(default=299792458.0)
    planck: dtype = field(default=6.6260775e-34)
    boltz: dtype = field(default=1.380658e-23)
    avogadro: dtype = field(default=6.0221367e23)

    # 2. Astronomical constants
    day: dtype = field(default=86400)  # day duration
    siyea: dtype = field(init=False)  # sideral year duration
    siday: dtype = field(init=False)  # sideral day duration
    nsday: dtype_int = field(default=24 * 3600)  # number of seconds in a day
    omega: dtype = field(init=False)  # earth rotation

    # 3. Terrestrial geoide constants
    radius: dtype = field(default=6371229)  # earth radius
    gravity0: dtype = field(default=9.80665)  # gravity constant

    # 4. Reference pressure
    p00ocean: dtype = field(default=201e5)  # Ref pressure for ocean model
    rho0ocean: dtype = field(default=1024)  # Ref density for ocean model
    th00ocean: dtype = field(default=286.65)  # Ref value for pot temp in ocean model
    sa00ocean: dtype = field(default=32.6)  # Ref value for salinity in ocean model

    p00: dtype = field(default=1e5)  # Reference pressure
    th00: dtype = field(default=300)  # Ref value for potential temperature

    # 5. Radiation constants
    stefan: dtype = field(init=False)  # Stefan-Boltzman constant
    io: dtype = field(default=1370)  # Solar constant

    # 6. Thermodynamic constants
    Md: dtype = field(default=28.9644e-3)  # Molar mass of dry air
    Mv: dtype = field(default=18.0153e-3)  # Molar mass of water vapour
    Rd: dtype = field(init=False)  # Gas constant for dry air
    Rv: dtype = field(init=False)  # Gas constant for vapour
    epsilo: dtype = field(init=False)  # Mv / Md
    cpd: dtype = field(init=False)  # Cpd (dry air)
    cpv: dtype = field(init=False)  # Cpv (vapour)
    rholw: dtype = field(default=1000)  # Volumic mass of liquid water
    rholi: dtype = field(default=900)  # Volumic mass of ice
    Cl: dtype = field(default=4.218e3)  # Cl (liquid)
    Ci: dtype = field(default=2.106e3)  # Ci (ice)
    tt: dtype = field(default=273.16)  # triple point temperature
    lvtt: dtype = field(default=2.5008e6)  # vaporisation heat constant
    lstt: dtype = field(default=2.8345e6)  # sublimation heat constant
    lmtt: dtype = field(init=False)  # melting heat constant
    estt: dtype = field(
        default=611.24
    )  # Saturation vapor pressure at triple point temperature

    alpw: dtype = field(init=False)  # Constants for saturation vapor pressure function
    betaw: dtype = field(init=False)
    gamw: dtype = field(init=False)

    alpi: dtype = field(
        init=False
    )  # Constants for saturation vapor pressure function over solid ice
    betai: dtype = field(init=False)
    gami: dtype = field(init=False)

    condi: dtype = field(default=2.2)  # Thermal conductivity of ice (W m-1 K-1)
    alphaoc: dtype = field(
        default=1.9e-4
    )  # Thermal expansion coefficient for ocean (K-1)
    betaoc: dtype = field(default=7.7475)  # Haline contraction coeff for ocean (S-1)
    roc: dtype = 0.69  # coeff for SW penetration in ocean (Hoecker et al)
    d1: dtype = 1.1  # coeff for SW penetration in ocean (Hoecker et al)
    d2: dtype = 23.0  # coeff for SW penetration in ocean (Hoecker et al)

    # 7. Precomputed constants
    Rd_Rv: dtype = field(init=False)  # Rd / Rv
    Rd_cpd: dtype = field(init=False)  # Rd / cpd
    invxp00: dtype = field(init=False)  # 1 / p00

    # 8. Machine precision
    mnh_tiny: dtype  # minimum real on this machine
    mnh_tiny_12: dtype  # sqrt(minimum real on this machine)
    mnh_epsilon: dtype  # minimum space with 1.0
    mnh_huge: dtype  # minimum real on this machine
    mnh_huge_12_log: dtype  # maximum log(sqrt(real)) on this machine
    eps_dt: dtype  # default value for dt
    res_flat_cart: dtype  # default     flat&cart residual tolerance
    res_other: dtype  # default not flat&cart residual tolerance
    res_prep: dtype  # default     prep      residual tolerance

    def __post_init__(self):
        # 2. Astronomical constants
        self.siyea = 365.25 * self.day / 6.283076
        self.siday = self.day / (1 + self.day / self.siyea)
        self.omega = 2 * self.pi / self.siday

        # 5. Radiation constants
        self.stefan = (
            2
            * self.pi**5
            * self.boltz**4
            / (15 * self.lightspeed**2 * self.planck**3)
        )

        # 6. Thermodynamic constants
        self.Rd = self.avogadro * self.boltz / self.Md
        self.Rv = self.avogadro * self.boltz / self.Mv
        self.epsilo = self.Mv / self.Md
        self.cpd = (7 / 2) * self.Rd
        self.cpv = 4 * self.Rv

        self.lmtt = self.lstt - self.lvtt
        self.gamw = (self.Cl - self.cpv) / self.Rv
        self.betaw = (self.lvtt / self.Rv) + (self.gamw * self.tt)
        self.alpw = (
            np.log(self.estt) + (self.betaw / self.tt) + (self.gamw * np.log(self.tt))
        )
        self.gami = (self.Ci - self.cpv) / self.Rv
        self.betai = (self.lstt / self.Rv) + self.gami * self.tt
        self.alpi = (
            np.log(self.estt) + (self.betai / self.tt) + self.gami * np.log(self.tt)
        )

        self.Rd_Rv = self.Rd / self.Rv
        self.Rd_cpd = self.Rd / self.cpd
        self.invxp00 = 1 / self.p00
