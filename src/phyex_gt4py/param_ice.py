from dataclasses import dataclass, field
from phyex_gt4py.config  import dtype_float, dtype_int
import numpy as np

@dataclass
class ParamIce:
    lwarm: bool = field(default=True)  # Formation of rain by warm processes
    lsedic: bool = field(default=True)  # Enable the droplets sedimentation
    ldeposc: bool = field(default=False)  # Enable cloud droplets deposition

    vdeposc: dtype_float = field(default=0.02)  # Droplet deposition velocity

    pristine_ice: str = field(default="PLAT")  # Pristine ice type PLAT, COLU, or BURO
    sedim: str = field(default="SPLI")  # Sedimentation calculation mode

    lred: bool = field(
        default=True
    )  # To use modified ice3/ice4 - to reduce time step dependency
    lfeedbackt: bool = field(default=True)
    levlimit: bool = field(default=True)
    lnullwetg: bool = field(default=True)
    lwetgpost: bool = field(default=True)
    lnullweth: bool = field(default=True)
    lwethpost: bool = field(default=True)
    snowriming: str = field(default="M90")  # OLD or M90 for Murakami 1990 formulation
    fracm90: dtype_float = field(default=0.1)
    nmaxiter_micro: dtype_int = field(
        default=5
    )  # max number of iterations for mixing ratio
    mrstep: dtype_float = field(default=5.0e-5)  # max mixing ratio for mixing ratio splitting
    lconvhg: bool = field(default=False)  # Allow the conversion from hail to graupel
    lcrflimit: bool = field(
        default=True
    )  # Limit rain contact freezing to possible heat exchange

    step_ts: dtype_float = field(default=0)  # Approximative time step for time-splitting

    subg_rc_rr_accr: str = field(default="NONE")  # subgrid rc-rr accretion
    subg_rr_evap: str = field(default="NONE")  # subgrid rr evaporation
    subg_rr_pdf: str = field(default="SIGM")  # pdf for subgrid precipitation
    subg_aucv_rc: str = field(default="NONE")  # type of subgrid rc->rr autoconv. method
    subg_aucv_ri: str = field(default="NONE")  # type of subgrid ri->rs autoconv. method
    subg_mf_pdf: str = field(
        default="TRIANGLE"
    )  # PDF to use for MF cloud autoconversions

    ladj_before: bool = field(
        default=True
    )  # must we perform an adjustment before rain_ice call
    ladj_after: bool = field(
        default=True
    )  # must we perform an adjustment after rain_ice call
    lsedim_after: bool = field(
        default=False
    )  # sedimentation done before (.FALSE.) or after (.TRUE.) microphysics

    split_maxcfl: dtype_float = field(
        default=0.8
    )  # Maximum CFL number allowed for SPLIT scheme
    lsnow_t: bool = field(default=False)  # Snow parameterization from Wurtz (2021)

    lpack_interp: bool = field(default=True)
    lpack_micro: bool = field(default=True)
    lcriauti: bool = field(default=True)

    npromicro: dtype_int = field(default=0)

    criauti_nam: dtype_float = field(default=0.2e-4)
    acriauti_nam: dtype_float = field(default=0.06)
    brcriauti_nam: dtype_float = field(default=-3.5)
    t0criauti_nam: dtype_float = field(init=False)
    criautc_nam: dtype_float = field(default=0.5e-3)
    rdepsred_nam: dtype_float = field(default=1)
    rdepgred_nam: dtype_float = field(default=1)
    lcond2: bool = field(default=False)
    frmin_nam: np.ndarray[40] = field(init=False)

    def __post_init__(self, hprogram: str):
        self.t0criauti_nam = (np.log10(self.criauti_nam) - self.brcriauti_nam) / 0.06

        self.frmin_nam[1:6] = 0
        self.frmin_nam[7:9] = 1.0
        self.frmin_nam[10] = 10.0
        self.frmin_nam[11] = 1.0
        self.frmin_nam[12] = 0.0
        self.frmin_nam[13] = 1.0e-15
        self.frmin_nam[14] = 120.0
        self.frmin_nam[15] = 1.0e-4
        self.frmin_nam[16:20] = 0.0
        self.frmin_nam[21:22] = 1.0
        self.frmin_nam[23] = 0.5
        self.frmin_nam[24] = 1.5
        self.frmin_nam[25] = 30.0
        self.frmin_nam[26:38] = 0.0
        self.frmin_nam[39] = 0.25
        self.frmin_nam[40] = 0.15

        if hprogram == "AROME":
            self.lconvhg = True
            self.ladj_before = True
            self.ladj_after = False
            self.lred = False
            self.sedim = "STAT"
            self.mrstep = 0
            self.subg_aucv_rc = "PDF"

        elif hprogram == "LMDZ":
            self.subg_aucv_rc = "PDF"
            self.sedim = "STAT"
            self.nmaxiter_micro = 1
            self.criautc_nam = 0.001
            self.criauti_nam = 0.0002
            self.t0criauti_nam = -5
            self.lred = True
            self.lconvhg = True
            self.ladj_before = True
            self.ladj_after = True
