# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from gt4py.storage import ones
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K

from phyex_gt4py.drivers.aro_adjust import AroAdjust
from phyex_gt4py.initialisation.state import allocate_data_array, allocate_state
from phyex_gt4py.initialisation.utils import initialize_field
from phyex_gt4py.phyex_common.phyex import Phyex

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import DataArray, DataArrayDict


def allocate_state(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
) -> DataArrayDict:
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


def initialize_state_with_constant(state: DataArrayDict, C: float) -> None:

    for name in state.keys():
        buffer = C * ones(state[name].shape)
        initialize_field(state[name], buffer)


def get_state_with_constant(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig, c: float
) -> DataArrayDict:
    """All arrays are filled with a constant between 0 and 1.

    Args:
        computational_grid (ComputationalGrid): _description_
        gt4py_config (GT4PyConfig): _description_

    Returns:
        DataArrayDict: _description_
    """
    state = allocate_state(computational_grid, gt4py_config=gt4py_config)
    initialize_state_with_constant(state, c)
    return state


if __name__ == "__main__":

    nx = 100
    ny = 1
    nz = 90

    cprogram = "AROME"
    phyex_config = Phyex(cprogram)
    gt4py_config = GT4PyConfig()
    grid = ComputationalGrid(nx, ny, nz)
    dt = 1

    aro_adjust = AroAdjust(
        grid,
        gt4py_config,
    )

    # Test 1
    state = get_state_with_constant(grid, gt4py_config, 0)
    tends, diags = aro_adjust(state, dt)

    # Test 2
    state = get_state_with_constant(grid, gt4py_config, 1)
    tends, diags = aro_adjust(state, dt)

    # Test 3
    state = get_state_with_constant(grid, gt4py_config, 0.5)
    tends, diags = aro_adjust(state, dt)
