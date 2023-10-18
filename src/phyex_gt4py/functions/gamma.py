from config import dtype

zcoef = tuple()
zcoef[0] = 76.18009172947146
zcoef[1] =-86.50532032941677
zcoef[2] = 24.01409824083091
zcoef[3] = -1.231739572450155
zcoef[4] =  0.1208650973866179E-2
zcoef[5] = -0.5395239384953E-5
zstp     =  2.5066282746310005
!
ZPI = 3.141592654

def gamma_0D(x: dtype) -> dtype:
    
    if x < 0:
        zx = 1 - x
    else:
        zx = x
        
    tmp = (zx + 6) * alog(zx + 5.5) - (zx + 5.5)
    
    ser = 1.000000000190015
    ser = sum([ser + zcoef[i] / (zx + 1 + i) for i in range(0, 6)])
    
    
    
    return None
