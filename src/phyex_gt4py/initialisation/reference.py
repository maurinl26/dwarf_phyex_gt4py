# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING

from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import DataArray, DataArrayDict


def allocate_tendencies(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    def allocate(units: str = "") -> DataArray:
        return allocate_data_array(
            computational_grid,
            (I, J, K),
            units,
            gt4py_config=gt4py_config,
            dtype="float",
        )

    return {
        "time": datetime(year=2022, month=1, day=1),
        "ths": allocate(),
        "r_i": allocate(),
        "r_s": allocate(),
        "r_r": allocate(),
        "r_g": allocate(),
        "r_v": allocate(),
        "cldfr": allocate(),
    }


def allocate_diagnostics(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    def _allocate(
        grid_id: Tuple[DimSymbol, ...],
        units: str,
        dtype: Literal["bool", "float", "int"],
    ) -> DataArray:
        return allocate_data_array(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
        )

    allocate = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_h = partial(_allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float")
    allocate_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")

    return {
        "time": datetime(year=2022, month=1, day=1),
        "icldfr": allocate(),
        "wcldfr": allocate_h(),
        "ssio": allocate_h(),
        "ssiu": allocate_h(),
        "ifr": allocate_h(),
        "hlc_hrc": allocate_h(),
        "hlc_hcf": allocate_h(),
        "hli_hri": allocate_h(),
        "hli_hcf": allocate_h(),
    }


def get_reference_tendencies(
    computational_grid: ComputationalGrid,
    *,
    gt4py_config: GT4PyConfig,
) -> DataArrayDict:
    tendencies = allocate_tendencies(computational_grid, gt4py_config=gt4py_config)
    # initialize_tendencies(tendencies, hdf5_reader)
    return tendencies


def get_reference_diagnostics(
    computational_grid: ComputationalGrid,
    *,
    gt4py_config: GT4PyConfig,
) -> DataArrayDict:
    diagnostics = allocate_diagnostics(computational_grid, gt4py_config=gt4py_config)
    # initialize_diagnostics(diagnostics, hdf5_reader)
    return diagnostics
