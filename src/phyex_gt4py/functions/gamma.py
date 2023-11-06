import numpy as np
from config import dtype

zcoef = tuple()
zcoef[0] = 76.18009172947146
zcoef[1] = -86.50532032941677
zcoef[2] = 24.01409824083091
zcoef[3] = -1.231739572450155
zcoef[4] = 0.1208650973866179e-2
zcoef[5] = -0.5395239384953e-5
zstp = 2.5066282746310005
ZPI = 3.141592654


def gamma(x: dtype) -> dtype:
    if x < 0:
        zx = 1 - x
    else:
        zx = x

    tmp = (zx + 6) * np.log(zx + 5.5) - (zx + 5.5)

    ser = sum([1.000000000190015 + zcoef[i] / (zx + 1 + i) for i in range(0, 6)])

    return None
