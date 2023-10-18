from dataclasses import dataclass

@dataclass
class DIMPhyex:
    
    # x dimension
    nit: int # Array dim
    nib: int # First index
    nie: int # Last index
    
    # y dimension
    njt: int
    njb: int
    nje: int
    
    # z dimension
    nkl: int
    # 1 : Meso NH order (bottom to top)
    # -1 : AROME order (top to bottom)
    
    nkt: int
    nkles: int
    nka: int
    nkb: int
    nke: int
    nktb: int
    nkte: int
    
    nibc: int
    njbc: int
    niec: int
    nijt: int # horizontal packing
    nijb: int # first index for horizontal packing
    nije: int # last index for horizontal packing
    
    
    