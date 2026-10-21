"""Kinds of grain boundary: where two neighbouring features sit in the hierarchy, and how far apart they are in orientation.

FM steel, from the finest level: a sub-block boundary separates two sub-blocks of one block, a block boundary two
blocks of one packet, a packet boundary two packets of one PAG, and a PAG boundary two PAGs. Which of these occur
depends on the level the structure was meshed at (meshed at block level there are no sub-block boundaries).

Twinned FCC: a twin boundary separates two grains of one host family (a host and its twins, or two twins of the
same host); every other boundary between grains is a grain boundary.

Each boundary also has a misorientation angle (cubic symmetry) and is low-angle when the angle is below a limit.
"""
from dataclasses import dataclass, field
import numpy as np

from .interfaces import Interfaces

FM_CLASSES = ('subblock', 'block', 'packet', 'pag')
TWINNED_CLASSES = ('twin', 'grain')
LEVEL_CLASSES = {'pag': ('pag',), 'pck': ('packet', 'pag'), 'blk': ('block', 'packet', 'pag'), 'sblk': FM_CLASSES}


def class_order(hierarchy):
    """The kinds of boundary the hierarchy can have, finest first."""
    return TWINNED_CLASSES if hierarchy.extra else LEVEL_CLASSES.get(hierarchy.level, FM_CLASSES)


def pair_class(hierarchy, a, b):
    """The kind of boundary between features ``a`` and ``b`` (their labels)."""
    fa, fb = hierarchy.features[a], hierarchy.features[b]
    if hierarchy.extra:
        family = hierarchy.extra[a]['family']
        return 'twin' if family >= 0 and family == hierarchy.extra[b]['family'] else 'grain'
    if fa.pag_id != fb.pag_id:
        return 'pag'
    if fa.pck_key != fb.pck_key:
        return 'packet'
    if fa.blk_key != fb.blk_key:
        return 'block'
    return 'subblock'


@dataclass
class BoundaryClasses:
    """The boundaries of a mesh by kind. ``face_*`` arrays have one entry per interface triangle."""
    order: tuple
    pairs: np.ndarray                 # (P, 2) feature labels
    kinds: list                       # (P,) kind of each pair
    area: np.ndarray                  # (P,) output units squared
    faces: np.ndarray                 # (P,) triangles
    misorientation: np.ndarray        # (P,) degrees, NaN where not available
    low_angle_degrees: float
    face_pair: np.ndarray             # (K,) pair of each triangle
    face_kind: np.ndarray             # (K,) index into ``order``
    symmetry: str
    summary: dict = field(default_factory=dict)

    @property
    def has_misorientation(self):
        return bool(len(self.misorientation)) and bool(np.isfinite(self.misorientation).any())

    def low_angle(self):
        """Boolean per pair, ``None`` when no misorientation is available."""
        if not self.has_misorientation:
            return None
        return self.misorientation < self.low_angle_degrees


def _angles(hierarchy, pairs, symmetry):
    out = np.full(len(pairs), np.nan)
    if symmetry != 'cubic':
        return out
    from upxo.xtalphy.crystal_orientation import cubic_misorientation

    for i, (a, b) in enumerate(pairs):
        out[i] = cubic_misorientation(hierarchy.features[int(a)].euler, hierarchy.features[int(b)].euler)[0]
    return out


def classify(hierarchy, interfaces: Interfaces, *, scale=1.0, symmetry='cubic', low_angle_degrees=15.0):
    """Sort the interfaces by kind; areas are multiplied by ``scale`` squared (mesh units to output units)."""
    order = class_order(hierarchy)
    pairs, inverse = interfaces.pairs()
    area = np.bincount(inverse, weights=interfaces.area, minlength=len(pairs)) * float(scale) ** 2
    faces = np.bincount(inverse, minlength=len(pairs))
    kinds = [pair_class(hierarchy, int(a), int(b)) for a, b in pairs]
    kind_index = np.array([order.index(k) for k in kinds], dtype=np.int64)
    angle = _angles(hierarchy, pairs, symmetry)
    result = BoundaryClasses(
        order=tuple(order), pairs=pairs, kinds=kinds, area=area, faces=faces, misorientation=angle,
        low_angle_degrees=float(low_angle_degrees), face_pair=inverse,
        face_kind=kind_index[inverse] if len(pairs) else np.zeros(0, dtype=np.int64), symmetry=symmetry)
    result.summary = summarize(result)
    return result


def summarize(classes: BoundaryClasses):
    """Per kind: pairs, triangles, area, share of the total area, area-weighted mean misorientation, low-angle share."""
    total = float(classes.area.sum())
    low = classes.low_angle()
    out = {'total_area': total, 'pairs': int(len(classes.pairs)), 'triangles': int(classes.faces.sum()),
           'low_angle_degrees': classes.low_angle_degrees, 'symmetry': classes.symmetry,
           'misorientation_available': classes.has_misorientation, 'by_kind': {}}
    if low is not None:
        out['low_angle_area_fraction'] = float(classes.area[low].sum() / total) if total else None
        out['mean_misorientation_deg'] = (float(np.average(classes.misorientation, weights=classes.area))
                                          if total else None)
    for index, name in enumerate(classes.order):
        rows = np.array([k == name for k in classes.kinds], dtype=bool) if len(classes.kinds) else np.zeros(0, bool)
        area = float(classes.area[rows].sum()) if rows.any() else 0.0
        entry = {'pairs': int(rows.sum()), 'triangles': int(classes.faces[rows].sum()) if rows.any() else 0,
                 'area': area, 'area_fraction': area / total if total else 0.0,
                 'mean_misorientation_deg': None, 'low_angle_area_fraction': None}
        if low is not None and rows.any() and area > 0:
            entry['mean_misorientation_deg'] = float(np.average(classes.misorientation[rows], weights=classes.area[rows]))
            entry['low_angle_area_fraction'] = float(classes.area[rows & low].sum() / area)
        out['by_kind'][name] = entry
    return out


def class_nodes(interfaces: Interfaces, classes: BoundaryClasses):
    """``{set name: node indices}`` of the nodes on the triangles of each kind, and of low- and high-angle ones.

    Names are ``ns_gb_<kind>`` and, when a misorientation is available, ``ns_gb_lowangle`` and ``ns_gb_highangle``.
    A node on the edge between two kinds is in both sets. Empty sets are left out.
    """
    out = {}
    for index, name in enumerate(classes.order):
        selected = interfaces.faces[classes.face_kind == index]
        if len(selected):
            out[f'ns_gb_{name}'] = np.unique(selected)
    low = classes.low_angle()
    if low is not None and len(classes.face_pair):
        per_face = low[classes.face_pair]
        for name, mask in (('ns_gb_lowangle', per_face), ('ns_gb_highangle', ~per_face)):
            if mask.any():
                out[name] = np.unique(interfaces.faces[mask])
    return out
