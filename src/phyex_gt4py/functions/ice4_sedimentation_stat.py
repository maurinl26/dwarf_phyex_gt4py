# -*- coding: utf-8 -*-
from __future__ import annotations

import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import Field


@gtscript.function
def other_species(fsed: float, exsed: float, pxrt: Field["float"]):
    return None


@gtscript.function
def pristine_ice(prit: Field["float"]):
    return None
