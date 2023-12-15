# -*- coding: utf-8 -*-
from dataclasses import asdict
from phyex_gt4py.phyex_common.phyex import Phyex
from phyex_gt4py.phyex_common.constants import Constants
from phyex_gt4py.phyex_common.nebn import Neb
from phyex_gt4py.phyex_common.param_ice import ParamIce
from phyex_gt4py.phyex_common.rain_ice_descr import RainIceDescr

if __name__ == "__main__":

    cprogram = "AROME"

    cst = Constants()
    print(asdict(cst))

    nebn = Neb(cprogram)
    print(asdict(nebn))

    parami = ParamIce(cprogram)
    print(asdict(parami))

    rain_ice_descr = RainIceDescr(cst, parami)
    print(asdict(rain_ice_descr))

    phyex = Phyex("AROME")
    print(asdict(phyex))

    phyex = Phyex("MESO-NH")
    print(asdict(phyex))
