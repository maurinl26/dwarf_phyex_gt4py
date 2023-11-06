import gt4py.cartesian.gtscript as gtscript
from config import dtype
from gt4py.cartesian.gtscript import Field


@gtscript.function
def other_species(fsed: dtype, exsed: dtype, pxrt: Field[dtype]):
    return None


@gtscript.function
def pristine_ice(prit: Field[dtype]):
    return None
