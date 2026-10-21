"""Twinned FCC structures as input to the deck writer.

A twinned structure folder holds ``lgi_twinned_clean.npy`` (one label per grain, axes z, y, x),
``orientations_all.pkl`` (unit quaternion [w, x, y, z] per grain), ``twin_role.pkl`` (one of
``non_host``, ``host``, ``primary_twin``, ``secondary_twin``) and ``twin_parent_of.pkl`` (twin to the
grain it grew from). The folder is read into a :class:`~.hierarchy.Hierarchy` whose features are the
grains, named as the voxel deck of the twinned structure names them (``es_cell_host_12``), with
element sets that join the grains by role and by host family.
"""
import math
from pathlib import Path
import pickle
import numpy as np

from .hierarchy import Feature, Hierarchy

LABELS_NAME = 'lgi_twinned_clean.npy'
FILES = (LABELS_NAME, 'orientations_all.pkl', 'twin_role.pkl', 'twin_parent_of.pkl')
ROLES = ('non_host', 'host', 'primary_twin', 'secondary_twin')
# role key -> (element-set stem, material stem, role-group element set)
ROLE_NAMES = {
    'non_host': ('es_cell_nonpart', 'MAT_CELL_NONPART', 'es_role_nonpart'),
    'host': ('es_cell_host', 'MAT_CELL_HOST', 'es_role_host'),
    'primary_twin': ('es_cell_ptwin', 'MAT_CELL_PTWIN', 'es_role_ptwin'),
    'stwin_a': ('es_cell_stwin_a', 'MAT_CELL_STWIN_A', 'es_role_stwin_a'),
    'stwin_b': ('es_cell_stwin_b', 'MAT_CELL_STWIN_B', 'es_role_stwin_b'),
}
FAMILY_PREFIX = 'es_family_'
ROLE_CODES = {'non_host': 0, 'host': 1, 'primary_twin': 2, 'stwin_a': 3, 'stwin_b': 4}


def quaternion_to_bunge(q):
    """Bunge-Euler angles (phi1, Phi, phi2) in degrees from a unit quaternion [w, x, y, z] (ZXZ, active)."""
    w, x, y, z = (float(v) for v in q)
    r11, r12 = 1 - 2 * (y * y + z * z), 2 * (x * y - z * w)
    r13 = 2 * (x * z + y * w)
    r23 = 2 * (y * z - x * w)
    r31, r32 = 2 * (x * z - y * w), 2 * (y * z + x * w)
    r33 = 1 - 2 * (x * x + y * y)
    if math.sqrt(max(0.0, 1.0 - r33 * r33)) > 1e-6:
        big_phi = math.acos(max(-1.0, min(1.0, r33)))
        phi1, phi2 = math.atan2(r31, -r32), math.atan2(r13, r23)
    else:
        big_phi = 0.0 if r33 > 0 else math.pi
        phi1, phi2 = math.atan2(-r12, r11), 0.0
    return math.degrees(phi1), math.degrees(big_phi), math.degrees(phi2)


def is_twinned_folder(folder):
    folder = Path(folder)
    return folder.is_dir() and all((folder / name).is_file() for name in FILES)


def _read(folder, name):
    with (Path(folder) / name).open('rb') as handle:
        return pickle.load(handle)


def role_keys(twin_role, twin_parent_of):
    """The role of each grain with secondary twins told apart: ``stwin_b`` grew from a primary twin, ``stwin_a`` from a host."""
    primary = {g for g, r in twin_role.items() if r == 'primary_twin'}
    out = {}
    for gid, role in twin_role.items():
        if role == 'secondary_twin':
            out[gid] = 'stwin_b' if twin_parent_of.get(gid) in primary else 'stwin_a'
        else:
            out[gid] = role
    return out


def host_family(label, twin_parent_of, hosts):
    """The host a grain descends from (itself when it is a host); ``None`` when no host is found."""
    seen, current = set(), label
    while current in twin_parent_of and current not in seen:
        seen.add(current)
        current = twin_parent_of[current]
    return current if current in hosts else None


def load_twinned(folder, voxel_size=1.0, units='microns'):
    """Read the twinned structure in ``folder`` into a :class:`Hierarchy` (one feature per grain)."""
    folder = Path(folder)
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f'{folder} lacks {", ".join(missing)}')
    if not voxel_size > 0:
        raise ValueError('voxel_size must be > 0.')
    raw = np.load(folder / LABELS_NAME, allow_pickle=False)
    if raw.ndim != 3 or raw.dtype.kind not in 'iu':
        raise ValueError('The label array must be a 3D integer array.')
    lgi = np.ascontiguousarray(np.transpose(raw, (2, 1, 0)))        # stored z, y, x; meshed as x, y, z
    labels, counts = np.unique(lgi, return_counts=True)
    if labels[0] <= 0:
        raise NotImplementedError('Label 0 (void) is present in the label array.')
    quats = _read(folder, 'orientations_all.pkl')
    twin_role = _read(folder, 'twin_role.pkl')
    parent_of = _read(folder, 'twin_parent_of.pkl')
    absent = [int(g) for g in labels if int(g) not in quats]
    if absent:
        raise ValueError(f'{len(absent)} grain(s) have no orientation (first few: {absent[:5]}).')
    twin_role = {int(k): v for k, v in twin_role.items()}
    unknown = sorted({str(twin_role.get(int(g))) for g in labels} - set(ROLES))
    if unknown:
        raise ValueError(f'Grains without a known twin role (found {unknown}; expected {list(ROLES)}).')
    role_of = role_keys(twin_role, parent_of)

    features, names, material_names = {}, {}, {}
    for number, label in enumerate(labels.tolist(), start=1):
        stem, material, _group = ROLE_NAMES[role_of[label]]
        names[label] = f'{stem}_{label}'
        material_names[label] = f'{material}_{label}'
        features[label] = Feature(label, 'pag', label, label, None, None, None,
                                  tuple(float(a) % 360.0 for a in quaternion_to_bunge(quats[label])), number)
    present = set(features)
    hosts = {g for g in present if role_of[g] == 'host'}
    groups = {}
    for role, (_stem, _material, group) in ROLE_NAMES.items():
        members = tuple(sorted(g for g in present if role_of[g] == role))
        if members:
            groups[group] = members
    twins = tuple(sorted(g for g in present if role_of[g] in ('primary_twin', 'stwin_a', 'stwin_b')))
    if twins:
        groups['es_role_twin'] = twins
    stwins = tuple(sorted(g for g in present if role_of[g] in ('stwin_a', 'stwin_b')))
    if stwins:
        groups['es_role_stwin'] = stwins
    families = {}
    for label in sorted(present):
        host = host_family(label, parent_of, hosts)
        if host is not None:
            families.setdefault(host, []).append(label)
    for host, members in families.items():
        groups[f'{FAMILY_PREFIX}{host}'] = tuple(members)

    extra = {label: {'role': role_of[label], 'role_code': ROLE_CODES[role_of[label]], 'family': -1}
             for label in present}
    for host, members in families.items():
        for member in members:
            extra[member]['family'] = int(host)
    return Hierarchy(
        level='pag', lgi=lgi, features=features, voxel_counts=dict(zip(labels.tolist(), counts.tolist())),
        units=str(units), voxel_size=float(voxel_size), pag_order=sorted(features),
        global_ids={('pag', g): f.global_id for g, f in features.items()},
        names=names, material_names=material_names, groups=groups, extra=extra, system='twinned FCC')
