# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from os.path import dirname, join, normpath

from ifs_physics_common.framework.config import (
    DataTypes,
    FortranConfig,
    GT4PyConfig,
    IOConfig,
    PythonConfig,
)


config_files_dir = normpath(join(dirname(__file__), "../../../config-files"))
default_python_config = PythonConfig(
    num_cols=1,
    enable_validation=True,
    input_file=join(config_files_dir, "input.h5"),
    reference_file=join(config_files_dir, "reference.h5"),
    num_runs=15,
    precision="double",
    data_types=DataTypes(bool=bool, float=np.float64, int=np.int64),
    gt4py_config=GT4PyConfig(
        backend="numpy", rebuild=False, validate_args=True, verbose=True
    ),
    sympl_enable_checks=True,
)

default_fortran_config = FortranConfig(
    build_dir=".",
    precision="double",
    variant="fortran",
    nproma=32,
    num_cols=1,
    num_runs=1,
    num_threads=1,
)

default_io_config = IOConfig(output_csv_file=None, host_name="")
