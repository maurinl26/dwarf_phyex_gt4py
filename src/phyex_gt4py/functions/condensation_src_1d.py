# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import function


@function
def src_1d(inq: int) -> float:
    """Replace src_1d table in Fortran version

    Args:
        inq (int): increment

    Returns:
        float: src_1d value on increment index
    """

    src = 0

    if inq == 0:
        src = 0.0
    if inq == 1:
        src = 0.0
    if inq == 2:
        src = 2.0094444e-04
    if inq == 3:
        src = 0.316670e-03
    if inq == 4:
        src = 4.9965648e-04
    if inq == 5:
        src = 0.785956e-03
    if inq == 6:
        src = 1.2341294e-03
    if inq == 7:
        src = 0.193327e-02
    if inq == 8:
        src = 3.0190963e-03
    if inq == 9:
        src = 0.470144e-02
    if inq == 10:
        src = 7.2950651e-03
    if inq == 11:
        src = 0.112759e-01
    if inq == 12:
        src = 1.7350994e-02
    if inq == 13:
        src = 0.265640e-01
    if inq == 14:
        4.0427860e-02
    if inq == 15:
        src = 0.610997e-01
    if inq == 16:
        src = 9.1578111e-02
    if inq == 17:
        src = 0.135888
    if inq == 18:
        src = 0.1991484
    if inq == 19:
        src = 0.230756
    if inq == 20:
        src = 0.2850565
    if inq == 21:
        src = 0.375050
    if inq == 22:
        src = 0.5000000
    if inq == 23:
        src = 0.691489
    if inq == 24:
        src = 0.8413813
    if inq == 25:
        src = 0.933222
    if inq == 26:
        src = 0.9772662
    if inq == 27:
        src = 0.993797
    if inq == 28:
        src = 0.9986521
    if inq == 29:
        src = 0.999768
    if inq == 30:
        src = 0.9999684
    if inq == 31:
        src = 0.999997
    if inq == 32:
        src = 1.0000000
    if inq == 33:
        src = 1.000000

    return src
