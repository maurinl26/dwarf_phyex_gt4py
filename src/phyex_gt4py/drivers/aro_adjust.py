
import sys
sys.path = [*sys.path, *[
        "./src/cloudsc_python/src",
        "../ifs-physics-common/src",
        "./src/cloudsc_gt4py/src",
        "./src"
    ]]
print(sys.path)

from typing import Dict, Optional
from sympl import ImplicitTendencyComponent

from gt4py.storage import zeros
from ifs_physics_common.framework.grid import ComputationalGrid


from phyex_gt4py.config import dtype_float
from phyex_gt4py.phyex import Phyex
from phyex_gt4py.stencils.ice_adjust import ice_adjust


class IceAdjust(ImplicitTendencyComponent):
    
    # from aro_adjust.h
    input_properties = {
        "zz",
        "rhodj",
        "exnref",
        "rhodref",
        "pabsm",
        "tht",
        "mfconv",
        "sigs",
        
    }
    
    tendency_properties = {
        "ths",      
        "rvs",      # PRS(1)
        "rcs",      # PRS(2)
        "ris",      # PRS(4)        
        
        "th",       # ZRS(0)
        "rv",       # ZRS(1)
        "rc",       # ZRS(2)
        "rr",       # ZRS(3)
        "ri",       # ZRS(4)
        "rs",       # ZRS(5)
        "rg",       # ZRS(6)
        
        "cldfr",
        "sigqsat",
        "ice_cld_wgt"
        
    }
    
    diagnostic_properties = {
        "icldfr",
        "wcldfr",
        "ssio",
        "ssiu",
        "srcs",
        "ifr",
        "hlc_hrc",
        "hlc_hcf",
        "hli_hri",
        "hli_hcf"
    }
    
    temporaries = {
       "cpd", 
        "rt",  # work array for total water mixing ratio
        "pv",  # thermodynamics
        "piv", # thermodynamics
        "qsl", # thermodynamics
        "qsi",
        "frac_tmp", # ice fraction
        "cond_tmp", # condensate
        "a", # related to computation of Sig_s
        "sbar",
        "sigma",
        "q1",   
    }
    
    
    
    def __init__(self):
        NotImplemented
        
    def array_call(self, state):
        NotImplemented

def initialize_fields(grid: ComputationalGrid, fields: Dict):
    
    return {
        key: zeros(grid.shape, dtype_float)
        for key in fields.keys()
    }
    

    
    
    
    

if __name__ == "__main__":
    
    cprogram = "AROME"
    iulout = 20
    dzmin = 20
    cmicro = "ICE3"
    csconv = "NONE"
    cturb = "TKEL"
    pstep = 50.0
    hbuname = "NONE"
    
    # Especes microphysiques (starting from 1)
    nrr = 6
    
    nx = 100
    ny = 100
    nz = 90
    
    # Phyex parameters
    phyex = Phyex(cprogram)

    ##### Define computational grid #####
    grid = ComputationalGrid(nx, ny, nz)
    
    aro_adjust = IceAdjust()
    
    initialize_fields(grid, aro_adjust.tendency_properties)
    initialize_fields(grid, aro_adjust.diagnostic_properties)
    initialize_fields(grid, aro_adjust.input_properties)
    initialize_fields(grid, aro_adjust.temporaries)
    
    #### Launch ice adjust #####
    ice_adjust(
        cst=phyex.cst,
        parami=phyex.param_icen,
        icep=phyex.rain_ice_paramn,
        neb=phyex.nebn,
        compute_srcs=phyex.compute_src,
        itermax=phyex.itermax,
        tstep=phyex.tstep,
        krr=nrr,
        lmfconv=phyex.lmfconv,    
        buname=hbuname,       
        **aro_adjust.input_properties,  # IN
        **aro_adjust.tendency_properties # INOUT
        **aro_adjust.diagnostic_properties # OUT 
        **aro_adjust.temporaries, # Temporary  fields
    )
    
    
    
    