

from typing import Optional

from config import dtype_float
from gt4py.storage import empty, zeros
from ifs_physics_common.framework.grid import ComputationalGrid

from phyex_gt4py.constants import Constants
from phyex_gt4py.dimphyex import Phyex
from phyex_gt4py.nebn import Neb
from phyex_gt4py.rain_ice_param import ParamIce, RainIceDescr, RainIceParam
from phyex_gt4py.stencils.ice_adjust import ice_adjust


def ini_phyex(
    hprogram: str,
    kunitnml: int,
    ldneednam: bool,
    kluout: int,
    kfrom: int,
    kto:  int,
    ptstep: float,
    dzmin: float,
    cmicro: str,
    cturb: str,
    csconv: str,
    ldhangemodel: Optional[bool],
    lddefaultval: Optional[bool],
    ldreadnam: Optional[bool],
    ldcheck: Optional[bool],
    krpint: Optional[int],
    ldinit: Optional[bool],
    phyex_in: Phyex,
    phyex_out: Phyex
) -> Phyex:
    
    NotImplemented
    
    
    

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
    
    klon = 100
    klev = 100
    krr = 7
    
    nx = 100
    ny = 100
    nz = 90
    
    
    #### Init Phyex ####
    cst = Constants()
    param_icen = ParamIce(hprogram=cprogram)
    nebn = Neb(hprogram=cprogram) 
    rain_ice_descrn = RainIceDescr(cst, param_icen)
    rain_ice_paramn = RainIceParam(cst, param_icen)
    
    phyex = Phyex(
        cst=cst,
        param_icen=param_icen,
        rain_ice_descrn=rain_ice_descrn,
        rain_ice_paramn=rain_ice_paramn,
        nebn=nebn
    )
    
    ##### Define computational grid #####
    grid = ComputationalGrid(nx, ny)
    
    exnref = zeros((nx, ny, nz), dtype_float=dtype_float)
    rhodref = zeros((nx, ny, nz), dtype_float=dtype_float)
    rhodj = zeros((nx, ny, nz), dtype_float=dtype_float)
    sigqsat = zeros((nx, ny, nz), dtype_float=dtype_float)
    sigs = zeros((nx, ny, nz), dtype_float=dtype_float)
    mfconv = zeros((nx, ny, nz), dtype_float=dtype_float)
    pabs = zeros((nx, ny, nz), dtype_float=dtype_float)
    
    # In -> 
    
    # Tmp 
    
    # Outs -> zeros()
    
    hlc_hrc = zeros()
    hlc_hcf = zeros()
    hli_hri = zeros()
    hli_hcf = zeros()
    
    
    ##### Get data #####
    
    
    #### Launch ice adjust #####
    # (stencil call)
    # TODO: shift fields to a sympl state
    ice_adjust(
        cst=phyex.cst,
        parami=phyex.param_icen,
        icep=phyex.rain_ice_paramn,
        neb=phyex.nebn,
        compute_srcs=phyex.compute_srcs,
        itermax=phyex.itermax,
        tstep=phyex.tstep,
        krr=nrr,
        lmfconv=phyex.lmfconv,    # TODO : PHYEX MISC lmfconv
        
        # IN
        sigqsat=sigqsat,
        rhodj=rhodj,    # TODO check if used 
        exnref=exnref,
        rhodref=rhodref,
        sigs,
        mfconv,
        pabs,
        zz,
        exn,
        cf_mf,
        rc_mf,
        ri_mf,
        ifr,
        icldfr,
        wcldfr,
        ssio, 
        ssiu,
        rc_tmp,         # tmp
        rs_tmp,         # tmp
        rs,
        rs,
        th,
        ths,
        scrs,
        cldfr,
        
    )
    
    
    
    