from __future__ import annotations
import gt4py.cartesian.gtscript as gtscript
from config import dtype_float
from gt4py.cartesian.gtscript import Field


@gtscript.function
def other_species(fsed: dtype_float, exsed: dtype_float, pxrt: Field[dtype_float]):
    return None


@gtscript.function
def pristine_ice(prit: Field[dtype_float]):
    return None
