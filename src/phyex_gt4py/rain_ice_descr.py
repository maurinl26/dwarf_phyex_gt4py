from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from config import dtype, dtype_int
from math import gamma

@dataclass
class RainIceDescr:
    cexvt: dtype = 0.4 # Air density fall speed correction

    rtmin: np.ndarray  # Min values allowed for mixing ratios

    # Cloud droplet charact.
    ac: dtype = field(init=False)
    bc: dtype = 3.0
    cc: dtype = field(init=False)
    dc: dtype = 2.0

    # Rain drop charact
    ar: dtype = field(init=False)
    br: dtype = field(default=3.0)
    cr: dtype = field(default=842)
    dr: dtype = field(default=0.8)
    ccr: dtype = field(default=8e-6)
    f0r: dtype = field(default=1.0)
    f1r: dtype = field(default=0.26)
    c1r: dtype = field(default=0.5)

    # Cloud ice charact.
    ai: dtype = field(init=False)
    bi: dtype = field(init=False)
    c_i: dtype = field(init=False)
    di: dtype = field(init=False)
    f0i: dtype = field(default=1.00)
    f2i: dtype = field(default=0.14)
    c1i: dtype = field(init=False)

    # Snow/agg charact.
    a_s: dtype = field(default=0.02)
    bs: dtype = field(default=1.9)
    cs: dtype = field(default=5.1)
    ds: dtype = field(default=0.27)
    ccs: dtype = field(default=5.0) # not lsnow
    cxs: dtype = field(default=1.0)
    f0s: dtype = field(default=0.86)
    f1s: dtype = field(default=0.28)
    c1s: dtype = field(init=False)

    # Graupel charact.
    ag: dtype = 19.6
    bg: dtype = 2.8
    cg: dtype = 124
    dg: dtype = 0.66
    ccg: dtype = 5e5
    cxg: dtype = -0.5
    f0g: dtype = 0.86
    f1g: dtype = 0.28
    c1g: dtype = 1 / 2

    # Hail charact.
    ah: dtype = 470
    bh: dtype = 3.0
    ch: dtype = 207
    dh: dtype = 0.64
    cch: dtype = 4e4
    cxh: dtype = -1.0
    f0h: dtype = 0.86
    f1h: dtype = 0.28
    c1h: dtype = 1 / 2

    # Cloud droplet distribution parameters
    
    # Over land
    alphac: dtype = 1.0 # Gamma law of the Cloud droplet (here volume-like distribution)
    nuc: dtype = 3.0 # Gamma law with little dispersion
    
    # Over sea
    alphac2: dtype = 1.0
    nuc2: dtype = 1.0
    
    lbexc: dtype = field(init=False)
    lbc: Tuple[dtype]

    # Rain drop distribution parameters
    alphar: dtype = 3.0 # Gamma law of the Cloud droplet (here volume-like distribution)
    nur: dtype = 1.0 # Gamma law with little dispersion
    lbexr: dtype = field(init=False)
    lbr: dtype = field(init=False)

    # Cloud ice distribution parameters
    alphai: dtype = 1.0 # Exponential law
    nui: dtype = 1.0 # Exponential law
    lbexi: dtype = field(init=False)
    lbi: dtype = field(init=False)

    # Snow/agg. distribution parameters
    alphas: dtype = field(default=1.0)
    nus: dtype = field(default=1.0)
    lbexs: dtype = field(init=False)
    lbs: dtype = field(init=False)
    ns: dtype = field(init=False)

    # Graupel distribution parameters
    alphag: dtype = 1.0
    nug: dtype = 1.0
    lbexg: dtype = field(init=False)
    lbg: dtype = field(init=False)

    # Hail distribution parameters
    alphah: dtype = 1.0
    nuh: dtype = 8.0
    lbexh: dtype = field(init=False)
    lbh: dtype = field(init=False)

    fvelos: dtype = field(default=0.097) # factor for snow fall speed after Thompson
    trans_mp_gammas: dtype = field(init=False) # coefficient to convert lambda for gamma functions
    lbdar_max: dtype = 1e5 # Max values allowed for the shape parameters (rain,snow,graupeln)
    lbdas_max: dtype = 1e5
    lbdag_max: dtype = 1e5
    lbdas_min: dtype = field(init=False)


    rtmin: List[dtype]  # min value allowed for mixing ratios
    conc_sea: dtype = 1e8  # Diagnostic concentration of droplets over sea
    conc_land: dtype = 3e8  # Diagnostic concentration of droplets over land
    conc_urban: dtype = 5e8 # Diagnostic concentration of droplets over urban area
    
    def __post_init__(self, pristine_ice: str, pi: dtype, lsnow: bool):
        
        
        # 2.2    Ice crystal characteristics
        if pristine_ice == "PLAT":
            self.ai = 0.82
            self.bi = 2.5
            self.c_i = 800
            self.di = 1.0
            self.c1i = 1 / pi
            
        elif pristine_ice == "COLU":
            self.ai = 2.14e-3
            self.bi = 1.7
            self.c_i = 2.1e5
            self.di = 1.585
            self.c1i = 0.8
            
        elif pristine_ice == "BURO":
            self.ai = 44.0
            self.bi = 3.0
            self.c_i = 4.3e5
            self.di = 1.663
            self.c1i = 0.5
            
        if lsnow:
            self.cs = 5.1
            self.ds = 0.27
            self.fvelos = 25.14
            
        self.c1s = 1 / pi
        
        if lsnow:
            self.alphas = 0.214
            self.nus = 43.7
            
        self.lbexc = 1 / self.bc
        self.lbexr = 1 / (-1 - self.br)
        self.lbexi = 1 / -self.bi
        self.lbexs = 1 / (self.cxs - self.bs)
        self.lbexg = 1 / (self.cxg - self.bg)
        self.lbexh = 1 / (self.cxh - self.bh)
        
        # 3.4 Constant for shape parameter
        momg = lambda alpha, nu, p: gamma(nu + p / alpha) / gamma(nu)

        gamc = momg(self.alphac, self.nuc, 3)
        gamc2 = momg(self.alphac2, self.nuc2, 3)
        self.lbc[0] = self.ar * gamc
        self.lbc[1] = self.ar * gamc2
        
        
        self.lbr = (self.ar * self.ccr * momg(self.alphar, self.nur, self.br)) ** (-self.lbexr)
        self.lbi = (self.ai * self.cci * momg(self.alphai, self.nui, self.bi)) ** (-self.lbexi)
        self.lbs = (self.a_s * self.ccs * momg(self.alphas, self.nus, self.bs)) ** (-self.lbexs)
        self.lbg = (self.ag * self.ccg * momg(self.alphag, self.nug, self.bg)) ** (-self.lbexg)
        self.lbh = (self.ah * self.cch * momg(self.alphah, self.nuh, self.bh)) ** (-self.lbexh)
            

@dataclass
class CloudPar:
    """Declaration of the model-n dependant Microphysic constants

    Args:
        nsplitr (int): Number of required small time step integration
            for rain sedimentation computation
        nsplitg (int): Number of required small time step integration
            for ice hydrometeor sedimentation computation

    """

    nsplitr: dtype_int
    nsplitg: dtype_int
