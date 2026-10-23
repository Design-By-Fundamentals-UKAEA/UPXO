"""
End-to-end check: on a real generated structure (PAG clustering -> HAGB-
constrained PAG orientations -> block generation -> KS block orientations),
the resulting orientations must obey the two physical constraints the whole
KS/HAGB machinery exists to guarantee:

  * every pair of blocks inside one PAG shares a common parent (FCC) orientation
    and differs only by one of the 24 KS variants, so their disorientation must
    land on 0 deg (same global variant) or one of the 10 tabulated
    KS_CHARACTERISTIC_ANGLES (different variant) -- never anything else.
  * every pair of neighbouring PAGs must be at least hagb_threshold apart,
    provided assign_pag_orientations_with_hagb did not have to fall back.

Unlike tests/fm_steel_3d/hierarchy_state.py's synthetic fixtures (random,
non-KS-derived orientations), this exercises the real generation pipeline so
the KS variant composition (OrientationAssigner3D.assign_orientations_to_all_blocks)
is covered end-to-end, not just in isolation on the raw 24-variant pool.
"""
import warnings

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase
from upxo.pxtal.fm_steel_3d import pag_technique_selector_3d
from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import OrientationAssigner3D
from upxo.pxtal.fm_steel_3d.disorientation_stats import KS_CHARACTERISTIC_ANGLES

_TOL_DEG = 0.1
_HAGB_THRESHOLD = 15.0


def _voronoi_lfi(n=24, n_seeds=60, seed=1):
    rng = np.random.default_rng(seed)
    cx = rng.integers(0, n, n_seeds).astype(np.float32)
    cy = rng.integers(0, n, n_seeds).astype(np.float32)
    cz = rng.integers(0, n, n_seeds).astype(np.float32)
    xi = np.arange(n, dtype=np.float32)[:, None, None, None]
    yi = np.arange(n, dtype=np.float32)[None, :, None, None]
    zi = np.arange(n, dtype=np.float32)[None, None, :, None]
    sq = (xi - cx) ** 2 + (yi - cy) ** 2 + (zi - cz) ** 2
    return (np.argmin(sq, axis=3) + 1).astype(np.int32)


@pytest.fixture(scope="module")
def pipeline():
    """Small real pipeline run: base -> PAGs -> HAGB orientations -> blocks -> KS orientations."""
    fm_base = FMSteel3DBase.from_lfi(
        _voronoi_lfi(), physical_dimensions=(24.0, 24.0, 24.0), voxel_size=1.0,
        connectivity=6, min_grain_nvoxels=8, random_seed=1, verbosity=0)

    fm_pag = pag_technique_selector_3d.generate_pags(
        fm_base, technique='A',
        pag_size_distribution={'sizes': [2, 3, 4], 'probs': [0.4, 0.4, 0.2]},
        pag_grain_fraction=1.0, random_seed=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fm_pag.assign_pag_orientations(
            pag_ori_mode='hagb_constrained',
            pag_ori_params={'hagb_threshold': _HAGB_THRESHOLD, 'max_attempts': 500},
            random_seed=1)
        n_fallback_warnings = sum(
            1 for w in caught if issubclass(w.category, RuntimeWarning))

    fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 3.0), random_seed=1)
    fm_ori = fm_blk.assign_orientations(ks_variant_selection='random_per_block', random_seed=1)

    return fm_pag, fm_ori, n_fallback_warnings


def test_fixture_hits_no_hagb_fallback(pipeline):
    """A precondition of the HAGB assertion below: this seed/params draw must
    not need the best-effort fallback, or the >= threshold guarantee does not
    hold and the test would be asserting something the pipeline never promised."""
    _, _, n_fallback_warnings = pipeline
    assert n_fallback_warnings == 0


def test_within_pag_block_disorientations_land_on_ks_angles(pipeline):
    fm_pag, fm_ori, _ = pipeline
    assigner = OrientationAssigner3D()

    checked = 0
    for pag_id in fm_pag.clusters_dict:
        block_ids = [bid for bid in fm_ori.block_orientations
                     if int(bid.split('_')[1]) == pag_id]
        for i in range(len(block_ids)):
            for j in range(i + 1, len(block_ids)):
                bi, bj = block_ids[i], block_ids[j]
                angle = assigner.disorientation_deg(
                    fm_ori.block_orientations[bi], fm_ori.block_orientations[bj])
                checked += 1
                if angle < _TOL_DEG:
                    continue
                nearest = min(KS_CHARACTERISTIC_ANGLES, key=lambda a: abs(a - angle))
                assert abs(angle - nearest) < _TOL_DEG, (
                    f"{bi} vs {bj}: disorientation {angle:.3f} deg is not within "
                    f"{_TOL_DEG} deg of 0 or any tabulated KS angle "
                    f"(nearest={nearest})")
    assert checked > 0


def test_across_pag_disorientations_meet_hagb_threshold(pipeline):
    fm_pag, _, n_fallback_warnings = pipeline
    assert n_fallback_warnings == 0  # see test_fixture_hits_no_hagb_fallback
    assigner = OrientationAssigner3D()

    seen, checked = set(), 0
    for pag_id, neigh in fm_pag.neigh_clid.items():
        for npag_id in neigh:
            key = (min(pag_id, npag_id), max(pag_id, npag_id))
            if key in seen:
                continue
            seen.add(key)
            angle = assigner.disorientation_deg(
                fm_pag.pag_orientations[pag_id], fm_pag.pag_orientations[npag_id])
            checked += 1
            assert angle >= _HAGB_THRESHOLD - 1e-6, (
                f"PAG {pag_id} vs {npag_id}: disorientation {angle:.3f} deg is "
                f"below the {_HAGB_THRESHOLD} deg HAGB threshold with no fallback")
    assert checked > 0
