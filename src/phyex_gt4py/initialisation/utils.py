# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from ifs_physics_common.utils.numpyx import assign

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ifs_physics_common.utils.typingx import DataArray, NDArrayLike


def initialize_storage_2d(storage: NDArrayLike, buffer: NDArray) -> None:
    ni = storage.shape[0]
    mi = buffer.size
    nb = ni // mi
    for b in range(nb):
        assign(storage[b * mi : (b + 1) * mi, 0:1], buffer[:, np.newaxis])
    assign(storage[nb * mi :, 0:1], buffer[: ni - nb * mi, np.newaxis])


def initialize_storage_3d(storage: NDArrayLike, buffer: NDArray) -> None:
    ni, _, nk = storage.shape
    mi, mk = buffer.shape
    lk = min(nk, mk)
    nb = ni // mi
    for b in range(nb):
        assign(storage[b * mi : (b + 1) * mi, 0:1, :lk], buffer[:, np.newaxis, :lk])
    assign(storage[nb * mi :, 0:1, :lk], buffer[: ni - nb * mi, np.newaxis, :lk])


def initialize_field(field: DataArray, buffer: NDArray) -> None:
    if field.ndim == 2:
        initialize_storage_2d(field.data, buffer)
    elif field.ndim == 3:
        initialize_storage_3d(field.data, buffer)
    else:
        raise ValueError("The field to initialize must be either 2-d or 3-d.")
