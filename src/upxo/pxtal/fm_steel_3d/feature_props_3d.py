"""
feature_props_3d.py

Slotted property class and query API for closed-cell features in the FM steel
3D hierarchy (grain → PAG → packet → block → sub-block) and any future
closed-cell types (precipitates, voids, etc.).

Public API
----------
FMID                    — frozen hierarchical feature identifier
FeatureProps            — slotted class; one instance per feature
FeaturePropsCollection  — iterable container with array accessors + statistics
query_feature(fm, fmid)         → FeatureProps
query_features(fm, [fmid, ...]) → FeaturePropsCollection
query_level(fm, level)          → FeaturePropsCollection  (all features at level)

Example
-------
from upxo.pxtal.fm_steel_3d.feature_props_3d import FMID, query_level, query_features

# All PAGs:
pag_props = query_level(fm_ori, 'pag')
print(pag_props.summary())

# Specific blocks:
ids = [FMID.block('B_1_3_0'), FMID.block('B_1_3_1')]
blk_props = query_features(fm_ori, ids)
for fp in blk_props:
    print(fp)

Level strings
-------------
'grain'    — grains in the base LFI (FMSteel3DBase)
'pag'      — parent austenite grains
'packet'   — slicing packets (each grain = one packet)
'block'    — martensitic blocks  (native_id is str 'B_...')
'subblock' — lath sub-blocks     (native_id is str 'SB_...')
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from upxo.pxtal.fm_steel_3d.geom_metrics_3d import (
    all_metrics,
    feature_volumes,
    feature_surface_faces,
    feature_junction_line_edges,
    feature_junction_points,
    feature_neighbours,
)

# ---------------------------------------------------------------------------
# Hierarchical feature identifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FMID:
    """
    Immutable identifier for one feature in the FM steel hierarchy.

    Attributes
    ----------
    level     : str   — 'grain' | 'pag' | 'packet' | 'block' | 'subblock'
    native_id : int or str
        int for grain / pag / packet levels (matches the integer label in
        the corresponding LGI).
        str for block / subblock levels (e.g. 'B_1_3_0', 'SB_B_1_3_0_1').

    Convenience constructors
    ------------------------
    FMID.grain(gid)      FMID.pag(pag_id)     FMID.packet(grain_id)
    FMID.block(bid_str)  FMID.subblock(sb_str)
    """
    level:     str
    native_id: Any

    # -- convenience constructors --
    @classmethod
    def grain(cls, gid: int) -> 'FMID':
        return cls('grain', int(gid))

    @classmethod
    def pag(cls, pag_id: int) -> 'FMID':
        return cls('pag', int(pag_id))

    @classmethod
    def packet(cls, grain_id: int) -> 'FMID':
        return cls('packet', int(grain_id))

    @classmethod
    def block(cls, block_id: str) -> 'FMID':
        return cls('block', str(block_id))

    @classmethod
    def subblock(cls, sb_id: str) -> 'FMID':
        return cls('subblock', str(sb_id))

    def __repr__(self) -> str:
        return f"FMID.{self.level}({self.native_id!r})"


# ---------------------------------------------------------------------------
# Single-feature property container
# ---------------------------------------------------------------------------

_LEVEL_INT = {'grain': 0, 'pag': 1, 'packet': 2, 'block': 3, 'subblock': 4}


class FeatureProps:
    """
    All geometric properties of one closed-cell feature.

    Slot variables (stored directly)
    ---------------------------------
    name          str   human-readable label / ID
    feature_type  str   'grain' | 'pag' | 'packet' | 'block' | 'subblock'
    feature_id    int|str  native feature ID (int for grain/pag/packet, str for block/sb)
    level         int   hierarchy depth (grain=0 … subblock=4)

    vol_nvox      int   voxel count
    surf_nfaces   int   boundary voxel-face count  (surface area proxy)
    jl_nedges     int   triple-line voxel-edge count (junction line length proxy)
    jp_nverts     int   junction-point vertex count  (≥3 labels meet at vertex)
    n_neighbours  int   number of distinct face-adjacent real features (void excluded)
    neighbour_ids tuple native IDs of those neighbours

    units         str   'microns' | 'mm' | 'm'
    voxel_size    float physical length of one voxel edge in `units`

    Property methods (computed, never stored)
    -----------------------------------------
    vol_physical          voxel_size^3  × vol_nvox         [units^3]
    surf_physical         voxel_size^2  × surf_nfaces       [units^2]
    jl_physical           voxel_size    × jl_nedges         [units]
    vol_equiv_diameter    diameter of equal-volume sphere   [units]
    vol_equiv_radius      radius  of equal-volume sphere    [units]
    sa_vol_ratio          surf_physical / vol_physical      [units^-1]
    sphericity            Wadell sphericity ∈ (0, 1]
    isoperimetric_quotient  36π V² / S³  ∈ (0, 1]  (= 1 for sphere)
    """

    __slots__ = (
        # identity
        'name', 'feature_type', 'feature_id', 'level',
        # voxel-space topology counts
        'vol_nvox', 'surf_nfaces', 'jl_nedges', 'jp_nverts',
        # neighbourhood
        'n_neighbours', 'neighbour_ids',
        # scale
        'units', 'voxel_size',
    )

    def __init__(
        self,
        name:          str,
        feature_type:  str,
        feature_id:    Any,
        level:         int,
        vol_nvox:      int,
        surf_nfaces:   int,
        jl_nedges:     int,
        jp_nverts:     int,
        n_neighbours:  int,
        neighbour_ids: tuple,
        units:         str,
        voxel_size:    float,
    ) -> None:
        self.name          = name
        self.feature_type  = feature_type
        self.feature_id    = feature_id
        self.level         = level
        self.vol_nvox      = vol_nvox
        self.surf_nfaces   = surf_nfaces
        self.jl_nedges     = jl_nedges
        self.jp_nverts     = jp_nverts
        self.n_neighbours  = n_neighbours
        self.neighbour_ids = neighbour_ids
        self.units         = units
        self.voxel_size    = voxel_size

    # -- physical scalings ---------------------------------------------------

    @property
    def vol_physical(self) -> float:
        return self.vol_nvox * self.voxel_size ** 3

    @property
    def surf_physical(self) -> float:
        return self.surf_nfaces * self.voxel_size ** 2

    @property
    def jl_physical(self) -> float:
        return self.jl_nedges * self.voxel_size

    # -- derived shape descriptors -------------------------------------------

    @property
    def vol_equiv_diameter(self) -> float:
        return (6.0 * self.vol_physical / math.pi) ** (1.0 / 3.0)

    @property
    def vol_equiv_radius(self) -> float:
        return self.vol_equiv_diameter / 2.0

    @property
    def sa_vol_ratio(self) -> float:
        v = self.vol_physical
        return self.surf_physical / v if v > 0.0 else math.inf

    @property
    def sphericity(self) -> float:
        """Wadell sphericity: ratio of surface area of equal-volume sphere to actual surface area."""
        v = self.vol_physical
        s = self.surf_physical
        if v <= 0.0 or s <= 0.0:
            return math.nan
        return (math.pi ** (1.0 / 3.0)) * (6.0 * v) ** (2.0 / 3.0) / s

    @property
    def isoperimetric_quotient(self) -> float:
        """36π V² / S³ — equals 1 for a perfect sphere, <1 for everything else."""
        v = self.vol_physical
        s = self.surf_physical
        if s <= 0.0:
            return math.nan
        return 36.0 * math.pi * v ** 2 / s ** 3

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FeatureProps({self.feature_type!r} id={self.feature_id!r} "
            f"vol={self.vol_nvox}vox "
            f"surf={self.surf_nfaces}f jl={self.jl_nedges}e jp={self.jp_nverts}v "
            f"nn={self.n_neighbours} "
            f"sph={self.sphericity:.3f})"
        )


# ---------------------------------------------------------------------------
# Collection of features
# ---------------------------------------------------------------------------

class FeaturePropsCollection:
    """
    Ordered, iterable container of FeatureProps instances with vectorised
    array accessors and statistical summary.

    Access
    ------
    len(c)          number of features
    c[i]            FeatureProps at index i
    for fp in c     iterate

    Voxel-space arrays (dtype int64)
    ---------------------------------
    .vol_nvox_array
    .surf_nfaces_array
    .jl_nedges_array
    .jp_nverts_array
    .n_neighbours_array

    Physical arrays (dtype float64)
    --------------------------------
    .vol_physical_array
    .surf_physical_array
    .jl_physical_array
    .vol_equiv_diameter_array

    Shape descriptor arrays (dtype float64)
    ----------------------------------------
    .sphericity_array
    .sa_vol_ratio_array
    .isoperimetric_quotient_array

    Methods
    -------
    .summary()              → dict of {prop: {mean, std, min, q25, q50, q75, max, n}}
    .distribution(prop, bins=None) → (hist_counts, bin_edges)
    .feature_ids()          → list of native IDs in collection order
    """

    __slots__ = ('_items',)

    def __init__(self, items: Sequence[FeatureProps]) -> None:
        self._items = tuple(items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def __repr__(self) -> str:
        if not self._items:
            return "FeaturePropsCollection(empty)"
        t = self._items[0].feature_type
        return f"FeaturePropsCollection({len(self._items)} {t}s)"

    # -- array helpers -------------------------------------------------------

    def _arr_slot(self, attr: str) -> np.ndarray:
        return np.array([getattr(fp, attr) for fp in self._items], dtype=np.int64)

    def _arr_prop(self, attr: str) -> np.ndarray:
        return np.array([getattr(fp, attr) for fp in self._items], dtype=np.float64)

    # -- voxel-space integer arrays ------------------------------------------

    @property
    def vol_nvox_array(self) -> np.ndarray:
        return self._arr_slot('vol_nvox')

    @property
    def surf_nfaces_array(self) -> np.ndarray:
        return self._arr_slot('surf_nfaces')

    @property
    def jl_nedges_array(self) -> np.ndarray:
        return self._arr_slot('jl_nedges')

    @property
    def jp_nverts_array(self) -> np.ndarray:
        return self._arr_slot('jp_nverts')

    @property
    def n_neighbours_array(self) -> np.ndarray:
        return self._arr_slot('n_neighbours')

    # -- physical float arrays -----------------------------------------------

    @property
    def vol_physical_array(self) -> np.ndarray:
        return self._arr_prop('vol_physical')

    @property
    def surf_physical_array(self) -> np.ndarray:
        return self._arr_prop('surf_physical')

    @property
    def jl_physical_array(self) -> np.ndarray:
        return self._arr_prop('jl_physical')

    @property
    def vol_equiv_diameter_array(self) -> np.ndarray:
        return self._arr_prop('vol_equiv_diameter')

    @property
    def vol_equiv_radius_array(self) -> np.ndarray:
        return self._arr_prop('vol_equiv_radius')

    # -- shape descriptor arrays ---------------------------------------------

    @property
    def sa_vol_ratio_array(self) -> np.ndarray:
        return self._arr_prop('sa_vol_ratio')

    @property
    def sphericity_array(self) -> np.ndarray:
        return self._arr_prop('sphericity')

    @property
    def isoperimetric_quotient_array(self) -> np.ndarray:
        return self._arr_prop('isoperimetric_quotient')

    # -- ID list -------------------------------------------------------------

    def feature_ids(self) -> list:
        return [fp.feature_id for fp in self._items]

    # -- summary statistics --------------------------------------------------

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return per-property descriptive statistics.

        Returns
        -------
        dict  {prop_name: {'mean', 'std', 'min', 'q25', 'q50', 'q75', 'max', 'n'}}
            Non-finite values (inf, nan) are excluded before computing statistics.
            Properties with no finite values are omitted.
        """
        _named_arrays = {
            'vol_nvox':               self.vol_nvox_array.astype(float),
            'surf_nfaces':            self.surf_nfaces_array.astype(float),
            'jl_nedges':              self.jl_nedges_array.astype(float),
            'jp_nverts':              self.jp_nverts_array.astype(float),
            'n_neighbours':           self.n_neighbours_array.astype(float),
            'vol_physical':           self.vol_physical_array,
            'surf_physical':          self.surf_physical_array,
            'jl_physical':            self.jl_physical_array,
            'vol_equiv_diameter':     self.vol_equiv_diameter_array,
            'sa_vol_ratio':           self.sa_vol_ratio_array,
            'sphericity':             self.sphericity_array,
            'isoperimetric_quotient': self.isoperimetric_quotient_array,
        }
        result = {}
        for name, arr in _named_arrays.items():
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                continue
            result[name] = {
                'mean': float(np.mean(finite)),
                'std':  float(np.std(finite)),
                'min':  float(np.min(finite)),
                'q25':  float(np.percentile(finite, 25)),
                'q50':  float(np.median(finite)),
                'q75':  float(np.percentile(finite, 75)),
                'max':  float(np.max(finite)),
                'n':    int(len(finite)),
            }
        return result

    def distribution(
        self,
        prop: str,
        bins: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (counts, bin_edges) histogram for property `prop`.

        Parameters
        ----------
        prop : str
            Any attribute or property name of FeatureProps
            (e.g. 'vol_physical', 'sphericity', 'n_neighbours').
        bins : int, optional
            Number of histogram bins.  Default: min(50, max(5, n//10)).
        """
        arr = np.array([getattr(fp, prop) for fp in self._items], dtype=float)
        arr = arr[np.isfinite(arr)]
        if bins is None:
            bins = max(5, min(50, len(arr) // 10))
        return np.histogram(arr, bins=bins)


# ---------------------------------------------------------------------------
# LGI builders (one per hierarchy level)
# ---------------------------------------------------------------------------

def _build_grain_lgi(fm) -> Tuple[np.ndarray, dict, dict]:
    """Grain level: LGI is the raw fm.lgi.  IDs are integers, no mapping needed."""
    return fm.lgi.copy(), {}, {}


def _build_pag_lgi(fm) -> Tuple[np.ndarray, dict, dict]:
    """PAG level: remap grain IDs → PAG IDs via a lookup table."""
    lgi = fm.lgi
    max_gid = int(lgi.max())
    lut = np.zeros(max_gid + 1, dtype=np.int32)
    for pag_id, grain_ids in fm.clusters_dict.items():
        for gid in grain_ids:
            if 0 < gid <= max_gid:
                lut[gid] = int(pag_id)
    return lut[lgi], {}, {}


def _build_phase_map(fm) -> np.ndarray:
    """Per-voxel phase ID (see phases_3d.py), via a grain-id -> phase-id
    lookup table applied to fm.lgi -- mirrors _build_pag_lgi's remap
    technique above, one level of grouping over instead of PAG ID.

    Retained-austenite PAGs never host blocks (with_pags_3d.py explicitly
    skips block generation for them), so this is how a caller asks "which
    voxels are actually martensite/block-bearing" by phase identity, rather
    than relying on those PAGs happening to already be label 0 in whatever
    block/sub-block label array is being filtered.
    """
    from .phases_3d import PHASE_MARTENSITE, PHASE_RETAINED_AUSTENITE
    lgi = fm.lgi
    max_gid = int(lgi.max())
    lut = np.full(max_gid + 1, PHASE_MARTENSITE, dtype=np.int32)
    retained_ids = getattr(fm, 'retained_austenite_pag_ids', set()) or set()
    for pag_id, grain_ids in fm.clusters_dict.items():
        if pag_id in retained_ids:
            for gid in grain_ids:
                if 0 < gid <= max_gid:
                    lut[gid] = PHASE_RETAINED_AUSTENITE
    lut[0] = 0  # background/void voxels carry no real phase
    return lut[lgi]


def _build_packet_lgi(fm) -> Tuple[np.ndarray, dict, dict]:
    """Packet level: for visualization, packets are the base grain structure.

    Each grain represents individual packets in the martensitic microstructure.
    LGI is the raw grain LGI (finest grain-level detail between PAG and Block).
    """
    return fm.lgi.copy(), {}, {}


def _build_block_lgi(fm) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """Block level: stamp block voxels with a sequential integer label."""
    NX, NY, NZ = fm.lgi.shape
    blk_lgi = np.zeros((NX, NY, NZ), dtype=np.int32)
    s2i: Dict[str, int] = {}
    i2s: Dict[int, str] = {}
    for bid, (block_id, voxels) in enumerate(fm.all_blocks.items(), start=1):
        s2i[block_id] = bid
        i2s[bid]      = block_id
        blk_lgi[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = bid
    return blk_lgi, s2i, i2s


def _build_subblock_lgi(fm) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """Sub-block level: stamp sub-block voxels with a sequential integer label."""
    NX, NY, NZ = fm.lgi.shape
    sb_lgi = np.zeros((NX, NY, NZ), dtype=np.int32)
    s2i: Dict[str, int] = {}
    i2s: Dict[int, str] = {}
    for bid, (sb_id, voxels) in enumerate(fm.all_subblocks.items(), start=1):
        s2i[sb_id] = bid
        i2s[bid]   = sb_id
        sb_lgi[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = bid
    return sb_lgi, s2i, i2s


_LGI_BUILDERS = {
    'grain':    _build_grain_lgi,
    'pag':      _build_pag_lgi,
    'packet':   _build_packet_lgi,
    'block':    _build_block_lgi,
    'subblock': _build_subblock_lgi,
}


def compute_boundary_mask_2d(
    label_slice: np.ndarray,
    background_val: int = 0,
    mode: str = "sharp",
) -> np.ndarray:
    """Return a boundary mask for a 2D integer label image.

    A pixel is True (or >0 for smooth/subpixel modes) when it neighbours at
    least one pixel with a different label value. Pixels whose label equals
    *background_val* are always False/0, suppressing RVE-edge artefacts.

    Parameters
    ----------
    label_slice : ndarray, shape (H, W), dtype int
        2D label image — one slice of a 3D LGI, already transposed to image
        orientation (rows = Y axis, cols = X axis).
    background_val : int, optional
        Label value treated as background; boundary pixels carrying this
        label are zeroed out.  Default 0.
    mode : str, optional
        Boundary detection mode:
        - "sharp": Pixel-aligned boundaries via find_boundaries(mode='inner').
          Returns boolean mask, same shape as input. Fastest.
        - "smooth" (under dev.): Upsamples label grid 2x, finds boundaries,
          then downscales. Returns float [0,1] smoothed boundaries.
        - "subpixel": Pure sub-pixel via find_boundaries(mode='subpixel').
          Uses scikit-image's mathematical sub-pixel algorithm.
          Returns float [0,1] downscaled to original shape. Smoothest.
        Default "sharp".

    Returns
    -------
    ndarray, shape (H, W), dtype bool or float
        Boolean mask for "sharp" mode; float array [0, 1] for smooth/subpixel.
    """
    if mode == "sharp":
        try:
            from skimage.segmentation import find_boundaries as _fb
            bnd = _fb(label_slice, connectivity=1, mode='inner')
        except Exception:
            bnd = np.zeros(label_slice.shape, dtype=bool)
            h = label_slice[:, :-1] != label_slice[:, 1:]
            bnd[:, :-1] |= h
            bnd[:, 1:]  |= h
            v = label_slice[:-1, :] != label_slice[1:, :]
            bnd[:-1, :] |= v
            bnd[1:, :]  |= v
        bnd[label_slice == background_val] = False
        return bnd

    elif mode == "smooth":
        from scipy.ndimage import zoom
        zoom_factor = 2
        zoomed_slice = zoom(label_slice.astype(float), zoom_factor, order=1).astype(int)
        try:
            from skimage.segmentation import find_boundaries as _fb
            bnd_zoomed = _fb(zoomed_slice, connectivity=1, mode='inner')
        except Exception:
            bnd_zoomed = np.zeros(zoomed_slice.shape, dtype=bool)
            h = zoomed_slice[:, :-1] != zoomed_slice[:, 1:]
            bnd_zoomed[:, :-1] |= h
            bnd_zoomed[:, 1:]  |= h
            v = zoomed_slice[:-1, :] != zoomed_slice[1:, :]
            bnd_zoomed[:-1, :] |= v
            bnd_zoomed[1:, :]  |= v
        bnd_zoomed[zoomed_slice == background_val] = False
        bnd = zoom(bnd_zoomed.astype(float), 1/zoom_factor, order=1)
        return bnd

    elif mode == "subpixel":
        from scipy.ndimage import zoom
        zoom_factor = 2
        zoomed_slice = zoom(label_slice.astype(float), zoom_factor, order=1).astype(int)
        try:
            from skimage.segmentation import find_boundaries as _fb
            bnd_subpixel = _fb(zoomed_slice, connectivity=1, mode='subpixel')
        except Exception:
            bnd_subpixel = np.zeros(zoomed_slice.shape, dtype=bool)
        if bnd_subpixel.size == 0 or not bnd_subpixel.any():
            return np.zeros(label_slice.shape, dtype=float)
        bnd_float = bnd_subpixel.astype(float)
        scale_back = label_slice.shape[0] / bnd_subpixel.shape[0]
        bnd = zoom(bnd_float, scale_back, order=1)
        return bnd

    else:
        raise ValueError(f"Unknown boundary mode: {mode!r}. Must be 'sharp', 'smooth', or 'subpixel'.")


def _resolve_int_label(fmid: FMID, s2i: Dict[str, int]) -> int:
    if isinstance(fmid.native_id, int):
        return fmid.native_id
    if isinstance(fmid.native_id, str):
        if fmid.native_id not in s2i:
            raise KeyError(f"Feature ID {fmid.native_id!r} not found.")
        return s2i[fmid.native_id]
    raise TypeError(
        f"native_id must be int or str, got {type(fmid.native_id).__name__!r}"
    )


def _native_id(int_label: int, i2s: Dict[int, str]) -> Any:
    """Convert integer LGI label back to native ID (identity for int-keyed levels)."""
    return i2s.get(int_label, int_label)


def _build_props(
    fmid: FMID,
    int_label: int,
    metrics: Dict,
    i2s: Dict,
    units: str,
    voxel_size: float,
    level_int: int,
) -> FeatureProps:
    nb_labels = metrics['nb_ids'].get(int_label, frozenset())
    real_nbs  = nb_labels - {0}
    return FeatureProps(
        name          = str(fmid.native_id),
        feature_type  = fmid.level,
        feature_id    = fmid.native_id,
        level         = level_int,
        vol_nvox      = int(metrics['vol'][int_label]) if int_label < len(metrics['vol'])   else 0,
        surf_nfaces   = int(metrics['surf'][int_label]) if int_label < len(metrics['surf']) else 0,
        jl_nedges     = int(metrics['jl'][int_label]) if int_label < len(metrics['jl'])     else 0,
        jp_nverts     = int(metrics['jp'][int_label]) if int_label < len(metrics['jp'])     else 0,
        n_neighbours  = len(real_nbs),
        neighbour_ids = tuple(_native_id(nb, i2s) for nb in real_nbs),
        units         = units,
        voxel_size    = voxel_size,
    )


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def query_feature(fm, fmid: FMID) -> FeatureProps:
    """
    Query all geometric properties of a single feature.

    Parameters
    ----------
    fm   : any FMSteel3D* state object
    fmid : FMID

    Returns
    -------
    FeatureProps
    """
    return query_features(fm, [fmid])[0]


def query_features(fm, fmids: Sequence[FMID]) -> FeaturePropsCollection:
    """
    Query properties for a list of features at the same hierarchy level.

    All FMIDs must have the same .level.

    Parameters
    ----------
    fm    : any FMSteel3D* state object
    fmids : sequence of FMID  (all at same level)

    Returns
    -------
    FeaturePropsCollection  (same order as fmids)
    """
    if not fmids:
        return FeaturePropsCollection([])

    levels = {fmid.level for fmid in fmids}
    if len(levels) != 1:
        raise ValueError(
            f"All FMIDs must be at the same hierarchy level. Got: {sorted(levels)}"
        )
    lvl = levels.pop()

    if lvl not in _LGI_BUILDERS:
        raise ValueError(
            f"Unknown level {lvl!r}. Choose from: {list(_LGI_BUILDERS)}"
        )

    lgi, s2i, i2s = _LGI_BUILDERS[lvl](fm)
    metrics       = all_metrics(lgi)
    units         = getattr(fm, 'units', 'microns')
    voxel_size    = float(fm.voxel_size)
    level_int     = _LEVEL_INT.get(lvl, -1)

    items = [
        _build_props(fmid, _resolve_int_label(fmid, s2i),
                     metrics, i2s, units, voxel_size, level_int)
        for fmid in fmids
    ]
    return FeaturePropsCollection(items)


def query_level(fm, level: str) -> FeaturePropsCollection:
    """
    Query properties for ALL features at a given hierarchy level.

    This is the most efficient bulk query — it builds the LGI and computes
    all metrics exactly once, then constructs FeatureProps for every
    non-void label present in the LGI.

    Parameters
    ----------
    fm    : any FMSteel3D* state object
    level : str  — 'grain' | 'pag' | 'packet' | 'block' | 'subblock'

    Returns
    -------
    FeaturePropsCollection
    """
    if level not in _LGI_BUILDERS:
        raise ValueError(
            f"Unknown level {level!r}. Choose from: {list(_LGI_BUILDERS)}"
        )

    lgi, s2i, i2s = _LGI_BUILDERS[level](fm)
    metrics       = all_metrics(lgi)
    units         = getattr(fm, 'units', 'microns')
    voxel_size    = float(fm.voxel_size)
    level_int     = _LEVEL_INT.get(level, -1)

    active_labels = np.unique(lgi)
    active_labels = active_labels[active_labels > 0]

    items = []
    for int_label in active_labels.tolist():
        native = _native_id(int_label, i2s)
        fmid   = FMID(level, native)
        items.append(
            _build_props(fmid, int_label, metrics, i2s, units, voxel_size, level_int)
        )

    return FeaturePropsCollection(items)
