from ifs_physics_common.framework.stencil import compile_stencil 
from phyex_gt4py.drivers.config import default_python_config
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


if __name__ == "__main__":
    
    logging.info("Compile condensation stencil")
    
    externals = {
        "lvtt":0,
        "lstt":0,
        "tt":0,
        "subg_mf_pdf":0,
        "subg_cond":0,
        "cpd":0,
        "cpv":0,
        "Cl":0,
        "Ci":0,
        "tt":0,
        "alpw":0,
        "betaw":0,
        "gamw":0,
        "alpi":0,
        "betai":0,
        "gami":0,
        "Rd":0,
        "Rv":0,
        "frac_ice_adjust":0,
        "tmaxmix":0,
        "tminmix":0,
        "criautc":0,
        "tstep":0,
        "criauti":0,
        "acriauti":0,
        "bcriauti":0,
        "nrr":6, 
    }

    ice_adjust = compile_stencil(
        "ice_adjust",
        default_python_config.gt4py_config,
        externals=externals
        )
    
    logging.info("Compilation succeeded")

