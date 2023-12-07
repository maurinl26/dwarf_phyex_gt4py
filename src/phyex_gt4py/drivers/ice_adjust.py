# -*- coding: utf-8 -*-

from __future__ import annotations
from functools import cached_property

from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.utils.typingx import PropertyDict
from ifs_physics_common.framework.grid import I, J, K


class IceAdjust(ImplicitTendencyComponent):
    def __init__(
        self, computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
    ):

        self.ice_adjust = self.compile_stencil("ice_adjust")

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "zz",
            "rhodj",
            "exnref",
            "rhodref",
            "pabsm",
            "tht",
            "mfconv",
            "sigs",
        }

    @cached_property
    def _tendency_properties(self) -> PropertyDict:
        return {
            "ths",
            "rvs",  # PRS(1)
            "rcs",  # PRS(2)
            "ris",  # PRS(4)
            "th",  # ZRS(0)
            "rv",  # ZRS(1)
            "rc",  # ZRS(2)
            "rr",  # ZRS(3)
            "ri",  # ZRS(4)
            "rs",  # ZRS(5)
            "rg",  # ZRS(6)
            "cldfr",
            "sigqsat",
            "ice_cld_wgt",
        }

    @cached_property
    def _diagnostic_property(self) -> PropertyDict:

        return {
            "icldfr",
            "wcldfr",
            "ssio",
            "ssiu",
            "srcs",
            "ifr",
            "hlc_hrc",
            "hlc_hcf",
            "hli_hri",
            "hli_hcf",
        }

    @cached_property
    def _temporaries(self) -> PropertyDict:
        return {
            "cpd",
            "rt",  # work array for total water mixing ratio
            "pv",  # thermodynamics
            "piv",  # thermodynamics
            "qsl",  # thermodynamics
            "qsi",
            "frac_tmp",  # ice fraction
            "cond_tmp",  # condensate
            "a",  # related to computation of Sig_s
            "sbar",
            "sigma",
            "q1",
        }

    def array_call(self, state):
        NotImplemented
