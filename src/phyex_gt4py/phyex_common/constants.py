# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import numpy as np


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
        Md: "float"          # Molar mass of dry air
        Mv: "float"          # Molar mass of water vapour
        Rd: "float"          # Gas constant for dry air
        Rv: "float"          # Gas constant for vapour
        epsilo: "float"      # Mv / Md
        cpd: "float"         # Cpd (dry air)
        cpv: "float"         # Cpv (vapour)
        rholw: "float"       # Volumic mass of liquid water
        Cl: "float"          # Cl (liquid)
        Ci: "float"          # Ci (ice)
        tt: "float"          # triple point temperature
        lvtt: "float"        # vaporisation heat constant
        lstt: "float"        # sublimation heat constant
        lmtt: "float"        # melting heat constant
        estt: "float"        # Saturation vapor pressure at triple point temperature

        alpw: "float"        # Constants for saturation vapor pressure function
        betaw: "float"
        gamw: "float"

        alpi: "float"        # Constants for saturation vapor pressure function over solid ice
        betai: "float"
        gami: "float"

        condi: "float"       # Thermal conductivity of ice (W m-1 K-1)
        alphaoc: "float"     # Thermal expansion coefficient for ocean (K-1)
        betaoc: "float"      # Haline contraction coeff for ocean (S-1)
        roc: "float" = 0.69  # coeff for SW penetration in ocean (Hoecker et al)
        d1: "float" = 1.1    # coeff for SW penetration in ocean (Hoecker et al)
        d2: "float" = 23.0   # coeff for SW penetration in ocean (Hoecker et al)

        rholi: "float"       # Volumic mass of ice

        # 7. Precomputed constants
        Rd_Rv: "float"       # Rd / Rv
        Rd_cpd: "float"      # Rd / cpd
        invxp00: "float"     # 1 / p00

        # 8. Machine precision
        mnh_tiny: "float"    # minimum real on this machine
        mnh_tiny_12: "float" # sqrt(minimum real on this machine)
        mnh_epsilon: "float" # minimum space with 1.0
        mnh_huge: "float"    # minimum real on this machine
        mnh_huge_12_log: "float" # maximum log(sqrt(real)) on this machine
        eps_dt: "float"      # default value for dt
        res_flat_cart: "float"   # default     flat&cart residual tolerance
        res_other: "float"   # default not flat&cart residual tolerance
        res_prep: "float"    # default     prep      residual tolerance

    """

    # 1. Fondamental constants
    pi: "float" = field(default=2 * np.arcsin(1.0))
    karman: "float" = field(default=0.4)
    lightspeed: "float" = field(default=299792458.0)
    planck: "float" = field(default=6.6260775e-34)
    boltz: "float" = field(default=1.380658e-23)
    avogadro: "float" = field(default=6.0221367e23)

    # 2. Astronomical constants
    day: "float" = field(default=86400)  # day duration
    siyea: "float" = field(init=False)  # sideral year duration
    siday: "float" = field(init=False)  # sideral day duration
    nsday: "int" = field(default=24 * 3600)  # number of seconds in a day
    omega: "float" = field(init=False)  # earth rotation

    # 3. Terrestrial geoide constants
    radius: "float" = field(default=6371229)  # earth radius
    gravity0: "float" = field(default=9.80665)  # gravity constant

    # 4. Reference pressure
    p00ocean: "float" = field(default=201e5)  # Ref pressure for ocean model
    rho0ocean: "float" = field(default=1024)  # Ref density for ocean model
    th00ocean: "float" = field(default=286.65)  # Ref value for pot temp in ocean model
    sa00ocean: "float" = field(default=32.6)  # Ref value for salinity in ocean model

    p00: "float" = field(default=1e5)  # Reference pressure
    th00: "float" = field(default=300)  # Ref value for potential temperature

    # 5. Radiation constants
    stefan: "float" = field(init=False)  # Stefan-Boltzman constant
    io: "float" = field(default=1370)  # Solar constant

    # 6. Thermodynamic constants
    Md: "float" = field(default=28.9644e-3)  # Molar mass of dry air
    Mv: "float" = field(default=18.0153e-3)  # Molar mass of water vapour
    Rd: "float" = field(init=False)  # Gas constant for dry air
    Rv: "float" = field(init=False)  # Gas constant for vapour
    epsilo: "float" = field(init=False)  # Mv / Md
    cpd: "float" = field(init=False)  # Cpd (dry air)
    cpv: "float" = field(init=False)  # Cpv (vapour)
    rholw: "float" = field(default=1000)  # Volumic mass of liquid water
    rholi: "float" = field(default=900)  # Volumic mass of ice
    Cl: "float" = field(default=4.218e3)  # Cl (liquid)
    Ci: "float" = field(default=2.106e3)  # Ci (ice)
    tt: "float" = field(default=273.16)  # triple point temperature
    lvtt: "float" = field(default=2.5008e6)  # vaporisation heat constant
    lstt: "float" = field(default=2.8345e6)  # sublimation heat constant
    lmtt: "float" = field(init=False)  # melting heat constant
    estt: "float" = field(
        default=611.24
    )  # Saturation vapor pressure at triple point temperature

    alpw: "float" = field(
        init=False
    )  # Constants for saturation vapor pressure function
    betaw: "float" = field(init=False)
    gamw: "float" = field(init=False)

    alpi: "float" = field(
        init=False
    )  # Constants for saturation vapor pressure function over solid ice
    betai: "float" = field(init=False)
    gami: "float" = field(init=False)

    condi: "float" = field(default=2.2)  # Thermal conductivity of ice (W m-1 K-1)
    alphaoc: "float" = field(
        default=1.9e-4
    )  # Thermal expansion coefficient for ocean (K-1)
    betaoc: "float" = field(default=7.7475)  # Haline contraction coeff for ocean (S-1)
    roc: "float" = 0.69  # coeff for SW penetration in ocean (Hoecker et al)
    d1: "float" = 1.1  # coeff for SW penetration in ocean (Hoecker et al)
    d2: "float" = 23.0  # coeff for SW penetration in ocean (Hoecker et al)

    # 7. Precomputed constants
    Rd_Rv: "float" = field(init=False)  # Rd / Rv
    Rd_cpd: "float" = field(init=False)  # Rd / cpd
    invxp00: "float" = field(init=False)  # 1 / p00

    # 8. Machine precision
    # mnh_tiny: "float"    # minimum real on this machine
    # mnh_tiny_12: "float" # sqrt(minimum real on this machine)
    # mnh_epsilon: "float" # minimum space with 1.0
    # mnh_huge: "float"    # minimum real on this machine
    # mnh_huge_12_log: "float" # maximum log(sqrt(real)) on this machine
    # eps_dt: "float"      # default value for dt
    # res_flat_cart: "float"   # default     flat&cart residual tolerance
    # res_other: "float"   # default not flat&cart residual tolerance
    # res_prep: "float"    # default     prep      residual tolerance

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
