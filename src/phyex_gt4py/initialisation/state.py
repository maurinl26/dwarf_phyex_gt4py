# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING

from gt4py.storage import ones

from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from phyex_gt4py.initialisation.utils import initialize_field

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import (
        DataArray,
        DataArrayDict,
        NDArrayLikeDict,
    )


def allocate_state(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
) -> NDArrayLikeDict:
    def _allocate(
        grid_id: Tuple[DimSymbol, ...],
        units: str,
        dtype: Literal["bool", "float", "int"],
    ) -> DataArray:
        return allocate_data_array(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
        )

    allocate_b_ij = partial(_allocate, grid_id=(I, J), units="", dtype="bool")
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_f_h = partial(
        _allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float"
    )
    allocate_f_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

    return {
        "zzf": allocate_f(),
        "rhodj": allocate_f(),
        "exnref": allocate_f(),
        "rhodref": allocate_f(),
        "pabsm": allocate_f(),
        "tht": allocate_f(),
        "sigs": allocate_f(),
        "mfconv": allocate_f(),
        "rc_mf": allocate_f(),
        "ri_mf": allocate_f(),
        "cf_mf": allocate_f(),
    }

def initialize_state_with_constant(state: DataArrayDict, C: float, gt4py_config: GT4PyConfig) -> None:

    for name in state.keys():
        state[name] = C * ones(state[name].shape, backend=gt4py_config.backend)


def get_state(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    state = allocate_state(computational_grid, gt4py_config=gt4py_config)
    initialize_state_with_constant(state, 0.5)
    return state
