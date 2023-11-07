

from typing import Dict, Optional
from sympl import ImplicitTendencyComponent

from config import dtype_float
from gt4py.storage import zeros
from ifs_physics_common.framework.grid import ComputationalGrid

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
        "ifr",
        "hlc_hrc",
        "hlc_hcf",
        "hli_hri",
        "hli_hcf"
    }
    
    temporaries = {
        
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
    
    # Especes microphysiques (starting from 1)
    nrr = 7
    
    nx = 100
    ny = 100
    nz = 90
    
    # Phyex parameters
    phyex = Phyex(cprogram)

    ##### Define computational grid #####
    grid = ComputationalGrid(nx, ny)
    
    
    aro_adjust = IceAdjust()
    
    initialize_fields(grid, aro_adjust.tendency_properties)
    initialize_fields(grid, aro_adjust.diagnostic_properties)
    initialize_fields(grid, aro_adjust.input_properties)
    initialize_fields(grid, aro_adjust.temporaries)
    
   
    
    #### Launch ice adjust #####
    # (stencil call)
    ice_adjust(
        cst=phyex.cst,
        parami=phyex.param_icen,
        icep=phyex.rain_ice_paramn,
        neb=phyex.nebn,
        compute_srcs=phyex.compute_srcs,
        itermax=phyex.itermax,
        tstep=phyex.tstep,
        krr=nrr,
        lmfconv=phyex.lmfconv,           
        **aro_adjust.input_properties,  # IN
        **aro_adjust.tendency_properties # INOUT
        **aro_adjust.diagnostic_properties # OUT 
        **aro_adjust.temporaries, # Temporary  fields
    )
    
    
    
    