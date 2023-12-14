# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import timedelta
from functools import cached_property
from itertools import repeat
from typing import Dict
import numpy as np

from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.utils.typingx import PropertyDict, NDArrayLikeDict
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage
from ifs_physics_common.utils.numpyx import assign


class AroAdjust(ImplicitTendencyComponent):
    def __init__(
        self, computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
    ):

        self.gt4py_config = gt4py_config
        self.ice_adjust = self.compile_stencil("ice_adjust")

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "zz": {"grid": (I, J, K), "units": ""},
            "rhodj": {"grid": (I, J, K), "units": ""},
            "exnref": {"grid": (I, J, K), "units": ""},
            "rhodref": {"grid": (I, J, K), "units": ""},
            "pabsm": {"grid": (I, J, K), "units": ""},
            "tht": {"grid": (I, J, K), "units": ""},
            "mfconv": {"grid": (I, J, K), "units": ""},
            "sigs": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {
            "ths": {"grid": (I, J, K), "units": ""},
            "rvs": {"grid": (I, J, K), "units": ""},  # PRS(1)
            "rcs": {"grid": (I, J, K), "units": ""},  # PRS(2)
            "ris": {"grid": (I, J, K), "units": ""},  # PRS(4)
            "th": {"grid": (I, J, K), "units": ""},  # ZRS(0)
            "rv": {"grid": (I, J, K), "units": ""},  # ZRS(1)
            "rc": {"grid": (I, J, K), "units": ""},  # ZRS(2)
            "rr": {"grid": (I, J, K), "units": ""},  # ZRS(3)
            "ri": {"grid": (I, J, K), "units": ""},  # ZRS(4)
            "rs": {"grid": (I, J, K), "units": ""},  # ZRS(5)
            "rg": {"grid": (I, J, K), "units": ""},  # ZRS(6)
            "cldfr": {"grid": (I, J, K), "units": ""},
            "sigqsat": {"grid": (I, J, K), "units": ""},
            "ice_cld_wgt": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _diagnostic_property(self) -> PropertyDict:

        return {
            "icldfr": {"grid": (I, J, K), "units": ""},
            "wcldfr": {"grid": (I, J, K), "units": ""},
            "ssio": {"grid": (I, J, K), "units": ""},
            "ssiu": {"grid": (I, J, K), "units": ""},
            "srcs": {"grid": (I, J, K), "units": ""},
            "ifr": {"grid": (I, J, K), "units": ""},
            "hlc_hrc": {"grid": (I, J, K), "units": ""},
            "hlc_hcf": {"grid": (I, J, K), "units": ""},
            "hli_hri": {"grid": (I, J, K), "units": ""},
            "hli_hcf": {"grid": (I, J, K), "units": ""},
        }

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {
            "cpd": {"grid": (I, J, K), "units": ""},
            "rt": {
                "grid": (I, J, K),
                "units": "",
            },  # work array for total water mixing ratio
            "pv": {"grid": (I, J, K), "units": ""},  # thermodynamics
            "piv": {"grid": (I, J, K), "units": ""},  # thermodynamics
            "qsl": {"grid": (I, J, K), "units": ""},  # thermodynamics
            "qsi": {"grid": (I, J, K), "units": ""},
            "frac_tmp": {"grid": (I, J, K), "units": ""},  # ice fraction
            "cond_tmp": {"grid": (I, J, K), "units": ""},  # condensate
            "a": {"grid": (I, J, K), "units": ""},  # related to computation of Sig_s
            "sbar": {"grid": (I, J, K), "units": ""},
            "sigma": {"grid": (I, J, K), "units": ""},
            "q1": {"grid": (I, J, K), "units": ""},
        }

    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: Dict[str, bool],
    ):

        with managed_temporary_storage(
            self.computational_grid,
            *repeat(((I, J), "float"), 6),
            ((I, J), "bool"),
            ((K,), "int"),
            gt4py_config=self.gt4py_config,
        ) as ():
            inputs = {
                "in_" + name.split("_", maxsplit=1)[1]: state[name]
                for name in self.input_properties
            }
            tendencies = {
                "out_tnd_loc_" + name.split("_", maxsplit=1)[1]: out_tendencies[name]
                for name in self.tendency_properties
            }
            diagnostics = {
                "out_" + name.split("_", maxsplit=1)[1]: out_diagnostics[name]
                for name in self.diagnostic_properties
            }
            temporaries = {}

            self.ice_adjust(
                **inputs,
                **tendencies,
                **diagnostics,
                **temporaries,
                dt=timestep.total_seconds(),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
