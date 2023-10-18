from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from config import dtype

@dataclass
class ParamIce:
    
    lwarm: bool                 # Formation of rain by warm processes
    lsedic: bool                # Enable the droplets sedimentation
    ldeposc: bool               # Enable cloud droplets deposition
    
    vdeposc: dtype         # Droplet deposition velocity
    
    pristine_ice: str           # Pristine ice type PLAT, COLU, or BURO
    sedim: str                  # Sedimentation calculation mode
    
    lred: bool                  # To use modified ice3/ice4 - to reduce time step dependency
    lfeedbackt: bool
    levlimit: bool
    lnullwetg: bool
    lwetgpost: bool
    lnullweth: bool
    lwethpost: bool
    snowriming: str             # OLD or M90 for Murakami 1990 formulation
    fracm90: dtype
    nmaxiter_micro: np.int64    # max number of iterations for mixing ratio
    mrstep: dtype          # max mixing ratio for mixing ratio splitting
    lconvhg: bool               # Allow the conversion from hail to graupel
    lcrflimit: bool             # Limit rain contact freezing to possible heat exchange
    
    step_ts: dtype         # Approximative time step for time-splitting
    
    subg_rc_rr_accr: str        # subgrid rc-rr accretion
    subg_rr_evap: str           # subgrid rr evaporation
    subg_rr_pdf: str            # pdf for subgrid precipitation
    subg_aucv_rc: str           # type of subgrid rc->rr autoconv. method
    subg_aucv_ri: str           # type of subgrid ri->rs autoconv. method
    strsubg_mf_pdf: str         # PDF to use for MF cloud autoconversions
    
    ladj_before: bool           # must we perform an adjustment before rain_ice call
    ladj_after: bool            # must we perform an adjustment after rain_ice call
    lsedim_after: bool          # sedimentation done before (.FALSE.) or after (.TRUE.) microphysics
    
    split_maxcfl: dtype    # Maximum CFL number allowed for SPLIT scheme
    lsnow_t: bool               # Snow parameterization from Wurtz (2021)
    
    # TODO : rm machine computation values 
    lpack_interp: bool
    lpack_micro: bool
    
    npromicro: np.int64
    
    criauti_nam: dtype
    t0criauti_nam: dtype
    brcriauti_nam: dtype
    acriauti_nam: dtype
    criautc_nam: dtype
    rdepsred_nam: dtype
    rdepgred_nam: dtype
    lcond2: bool 
    xfrmin_nam: np.ndarray[40]
    
    
    
    
@dataclass
class RainIceParam:
    
    fsedc : Tuple[dtype]   # Constants for sedimentation fluxes of C
    fsedr: dtype           # Constants for sedimentation
    exsedr: dtype
    fsedi: dtype
    excsedi: dtype
    exrsedi: dtype
    fseds: dtype
    exseds: dtype
    fsedg: dtype
    exsedg: dtype
    
    nu10: dtype            # Constants for heterogeneous ice nucleation HEN
    alpha1: dtype
    beta1: dtype
    nu20: dtype            
    alpha2: dtype
    beta2: dtype
    mnu0: dtype            # Mass of nucleated ice crystal
    
    alpha3: dtype          # Constants for homogeneous ice nucleation HON
    beta3: dtype 
    
    scfac: dtype           # Constants for raindrop and
    o0evar: dtype          # evaporation EVA and for 
    o1evar: dtype
    ex0evar: dtype
    ex1evar: dtype
    o0depi: dtype          # deposition DEP on I
    o2depi: dtype
    o0deps: dtype          # on S
    o1deps: dtype
    ex0deps: dtype
    ex1deps: dtype
    rdepsred: dtype
    o0depg: dtype          # on G
    o1depg: dtype
    ex0depg: dtype
    ex1depg: dtype
    rdepgred: dtype
    
    timauti: dtype         # Constants for pristine ice autoconversion : AUT
    texauti: dtype         
    criauti: dtype 
    t0criauti: dtype 
    acriauti: dtype 
    bcriauti: dtype 
    
    colis: dtype           # Constants for snow aggregation : AGG
    colexis: dtype
    fiaggs: dtype
    exiaggs: dtype
    
    timautc: dtype         # Constants for cloud droplet autoconversion AUT
    criautc: dtype
    
    fcaccr: dtype          # Constants for cloud droplets accretion on raindrops : ACC
    excaccr: dtype
    
    dcsslim: dtype
    colcs: dtype
    excrimss: dtype
    crimss: dtype
    excrimsg: dtype
    crimsg: dtype
    
@dataclass
class RainIceDescr:
    
    cexvt: dtype # Air density fall speed correction
    
    rtmin: np.ndarray # Min values allowed for mixing ratios
    
    # Cloud droplet charact.
    ac: dtype
    bc: dtype
    cc: dtype
    dc: dtype
    
    # Rain drop charact
    ar: dtype
    br: dtype
    cr: dtype
    dr: dtype
    ccr: dtype
    f0r: dtype
    f1r: dtype
    c1r: dtype
    
    # Cloud ice charact.
    ai: dtype
    bi: dtype
    c_i: dtype
    di: dtype
    f0i: dtype
    f2i: dtype
    c1i: dtype
    
    # Snow/agg charact.
    a_s: dtype
    bs: dtype
    cs: dtype
    ds: dtype
    ccs: dtype
    cxs: dtype
    f0s: dtype
    f1s: dtype
    c1s: dtype
    
    # Graupel charact.
    ag: dtype
    bg: dtype
    cg: dtype
    dg: dtype
    ccg: dtype
    cxg: dtype
    f0g: dtype
    f1g: dtype
    c1g: dtype
    
    # Hail charact.
    ah: dtype
    bh: dtype
    ch: dtype
    dh: dtype
    cch: dtype
    cxh: dtype
    f0h: dtype
    f1h: dtype
    c1h: dtype
    
    # Cloud droplet distribution parameters
    alphac: dtype
    nuc: dtype
    alphac2: dtype
    nuc2: dtype
    lbexc: dtype
    lbc: Tuple[dtype]
    
    # Rain drop distribution parameters
    alphar: dtype
    nur: dtype
    lbexr: dtype
    lbr: dtype
    
    # Cloud ice distribution parameters
    alphai: dtype
    nui: dtype
    lbexi: dtype
    lbi: dtype
    
    # Snow/agg. distribution parameters
    alphas: dtype
    nus: dtype
    lbexs: dtype
    lbs: dtype
    ns: dtype
    
    # Graupel distribution parameters 
    alphag: dtype
    nug: dtype
    lbexg: dtype
    lbg: dtype
    
    # Hail distribution parameters 
    alphag: dtype
    nuh: dtype
    lbexh: dtype
    lbh: dtype
    
    fvelos: dtype           # factor for snow fall speed after Thompson
    trans_mp_gammas: dtype  # coefficient to convert lmabdas for gamma functions
    lbdar_max: dtype        # Max values allowed for the shape parameters (rain,snow,graupeln)
    lbdas_min: dtype
    lbdas_max: dtype
    lbdag_max: dtype
    
    rtmin: List[dtype]      # min value allowed for mixing ratios 
    conc_sea: dtype         # Diagnostic concentration of droplets over sea
    conc_land: dtype        # Diagnostic concentration of droplets over land
    conc_urban: dtype       # Diagnostic concentration of droplets over urban area
     
    
    