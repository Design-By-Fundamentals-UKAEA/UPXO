import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import generate_binary_structure

@dataclass(frozen=False)
class part_cntr_CLS:
    """A class to store partitioning controls"""
    connectivity: int
    bstruct: any = field(init=False)
    def __post_init__(self):
        self.bstruct = generate_binary_structure(3, self.connectivity)
@dataclass(frozen=False)
class ids_counts_CLS:
    """Guidance to set-up the feature names.       
    <<< Please restrict featuire names to only from below options: 
    grain, subgrain,ptwin i.e. primary twin, stwin i.e. 
    secondary twin, pag for prior austenitic grain, pck for packets in 
    Martensitic steels (FMS), blk for blocks in FMS, sblk for sub 
    blocks in FMS, l for laths in FMS >>>
    """
    f: str  # feature name
    """Cells (C)"""
    c: np.array = field(init=False)  # C IDs. np.array of np.int32
    n_c: int = field(init=False)  # C count
    v_c: dict = field(init=False) # C vox IDs. @ c
    nv_c: dict = field(init=False)  # C voxel count. @ c
    """Cell boundaries: CB"""
    cb: np.array = field(init=False)  # CB IDs. np.array of np.int32
    n_cb: int = field(init=False)  # CB count.
    v_cb: dict = field(init=False) # CB vox IDs. @c: np.array of np.int32
    nv_cb: dict = field(init=False)  # CB vox count. @c: int
    """CB segments (CBS)"""
    cbs: dict = field(init=False)  # CBS IDs. @c: list of int
    n_cbs: dict = field(init=False)  # CBS count. @c: list of int
    v_cbs: dict = field(init=False) # CBS vox IDs. @c: np.array of np.int32
    nv_cbs: dict = field(init=False)  # CBS vox count. @c: list of int
    """CBS edges (CBSE)"""
    cbse: dict = field(init=False)  # CBSE IDs. @c: list of int
    n_cbse: dict = field(init=False)  # CBSE count. @c: list of int
    v_cbse: dict = field(init=False) # CBSE vox IDs. @c: np.array of np.int32
    nv_cbse: dict = field(init=False)  # CBSE vox count. @c: list of int
    """CBSE points (CBSEP)"""
    cbsep: dict = field(init=False)  # CBSEP IDs. @c: list of int
    n_cbsep: dict = field(init=False)  # CBSEP count. @c
    v_cbsep: dict = field(init=False) # CBSEP vox IDs. @c: np.array of np.int32
    nv_cbsep: dict = field(init=False)  # CBSEP vox count. @c: list of int
    """CBSE inflection vert (CBSEV)"""
    cbsev: dict = field(init=False)  # CBSEV IDs. @c: list of int
    n_cbsev: dict = field(init=False)  # CBSEV count. @c - int
    v_cbsev: dict = field(init=False) # CBSEV vox IDs. @c: np.array of np.int32
    nv_cbsev: dict = field(init=False)  # CBSEV vox count. @c: list if int
    """CBSE junc. pnts (CBSEJ)"""
    cbsej: dict = field(init=False)  # CBSEJ IDs. @c - list of int
    n_cbsej: dict = field(init=False)  # CBSEJ count. @c - list of int
    v_cbsej: dict = field(init=False) # CBSEJ vox IDs. @c: np.array of np.int32
    nv_cbsej: dict = field(init=False)  # CBSEJ vox count. @c: list of int
    
@dataclass(frozen=False)
class ids_CLS:
    """A class store IDs of all features."""
    v: np.array = field(init=False)  # Voxel IDs
      # Cell IDs
      # Cell boundary (CB) IDs
      # CB segment IDs, local @ cid
      # CBS edge IDs, local @ cid
      # CBSE Junc pnt IDs, lcl @ cid
    cbsG: np.array = field(init=False)  # CB segment IDs, global numbering
    cbseG: np.array = field(init=False)  # CBS edge IDs, global numbering
    cbsejG: np.array = field(init=False)  # CBSE Junc pnt IDs, glb nmbrng
    def __repr__(self):
        return f"UPXO.IMG.3D - IDs object. {id(self)}"
@dataclass(frozen=False)
class vox_CLS:
    """ A class to represent basic voxel details in the RVE.
    Attributes ====>
    l: float
        Voxel edge length. Physical nuits: micro-meters
    """
    l: float
    @property
    def vol(self) -> float:
        return self.l**3
@dataclass(frozen=False)
class rve_CLS:
    """ A class to represent base domain details.
    Attributes ====>
    nx, ny, nz: int
        Number of voxzels along x, y, z.
    """
    nx: int
    ny: int
    nz: int
    vox: any
    n: int = field(init=False)
    vol: float = field(init=False)
    def __post_init__(self):
        self.n = self.get_n()
        self.vol = self.set_vol()
    def set_n(self) -> float:
        return self.nx*self.ny*self.nz
    def set_vol(self) -> float:
        return self.n*(self.vox.l**3)
