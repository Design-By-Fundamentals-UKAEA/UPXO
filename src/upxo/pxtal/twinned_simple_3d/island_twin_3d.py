"""
island_twin_3d.py
==================
Island Twin Introduction -- isolated, Sigma3-oriented twin inclusions that
touch no boundary of whatever grain/region contains them, carved as bounded
oblate ellipsoids (thin along the {111} habit-plane normal, like the real
slab twins, but bounded in-plane so they never reach the container's own
edge -- the one thing TwinGenerator3D's own plane-slab carving can never
produce, see its introduce_twin_lamella_3d: an infinite-plane cut through a
finite grain always touches that grain's boundary by construction).

Runs strictly AFTER Structure Cleaning (Topology Cleaning), on the final
cleaned post-twin structure -- never before, and never interleaved with it:
an island's own surface voxels touch nothing of the same label on any face
(isolated by construction), which is exactly the shape
StructureCleaner3D._fast_clean_spikes is built to erase. Placing islands
only after cleaning is done means that erosion risk never arises.

Size: the short (thickness) axis is bootstrap-sampled from the SAME EBSD
thickness array (`twin_thickness['thick_um']`) the lamella twins use, for
consistency; the long (in-plane) axis is this short axis times
`island_aspect_ratio` -- an explicit, user-adjustable multiplier, not a
separately-calibrated distribution. Calibrating island size against EBSD's
own 2D island-twin statistics (not just thickness) needs a dedicated EBSD
characterization function that does not exist yet; this aspect-ratio
approach is a deliberate, honest placeholder until that exists, not a
disguised "real" calibration.
"""
import time
from typing import Dict, List, Optional

import numpy as np

from upxo._sup import progress

__all__ = ["introduce_island_twin_3d", "IslandTwinIntroducer",
           "ISLAND_CATEGORIES"]

# The three host-region categories a user can place islands into, and the
# container-grain-id set each one draws from (see IslandTwinIntroducer).
ISLAND_CATEGORIES = ('remaining_parent', 'non_host', 'primary_twin')

# The ellipsoid-membership test (the per-attempt hot loop, run for every
# placement attempt including every failed one) has an optional numba
# kernel: a single scalar pass per voxel with no intermediate numpy
# allocations (no outer/norm temporaries), vs. the always-available numpy
# path below. Serial numpy is the fallback whenever numba is missing or
# fails to compile/run for any reason -- this speed-up is never a hard
# dependency.
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _ellipsoid_select_numba(coords, cx, cy, cz, nx, ny, nz, ht2, hw2, r):
        # Bounding-box reject AND the exact ellipsoid-membership test in
        # one scalar pass over every (int) container voxel -- no
        # intermediate numpy allocation (no separate bbox mask, no
        # astype-to-float copy, no outer/norm temporaries) the way the
        # numpy fallback below needs to stay vectorized-efficient.
        n = coords.shape[0]
        sel = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            rx = coords[i, 0] - cx
            ry = coords[i, 1] - cy
            rz = coords[i, 2] - cz
            if abs(rx) > r or abs(ry) > r or abs(rz) > r:
                continue
            d_n = rx * nx + ry * ny + rz * nz
            qx = rx - d_n * nx
            qy = ry - d_n * ny
            qz = rz - d_n * nz
            d_r2 = qx * qx + qy * qy + qz * qz
            sel[i] = (d_n * d_n) / ht2 + d_r2 / hw2 <= 1.0
        return sel

    # Warm up the JIT once at import time (tiny dummy call) so the first
    # real placement attempt doesn't pay the compile cost.
    _ellipsoid_select_numba(
        np.zeros((1, 3), dtype=np.int64), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 1.0)
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def _boundary_distance(binary_mask: np.ndarray) -> np.ndarray:
    """Distance from every voxel to the nearest False voxel -- including
    the domain's own outer edge, padded in as a boundary too, so a grain
    touching the RVE's physical edge doesn't get treated as having
    unlimited margin there (an island reaching the RVE boundary isn't an
    isolated interior inclusion either). This is the expensive part of
    finding a valid island centre; IslandTwinIntroducer caches it per
    container and only recomputes when that container's mask actually
    changes, instead of once per placement attempt."""
    from scipy.ndimage import distance_transform_edt
    if not binary_mask.any():
        return np.zeros(binary_mask.shape, dtype=float)
    padded = np.pad(binary_mask, 1, mode='constant', constant_values=False)
    return distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]


def introduce_island_twin_3d(
        lgi: np.ndarray,
        container_gid: int,
        center,
        normal,
        half_thickness: float,
        half_width: float,
        next_gid: int,
        container_coords: Optional[np.ndarray] = None,
):
    """
    Carve a single bounded, oblate-ellipsoid island twin out of
    `container_gid`'s own voxels in a 3D lgi array.

    Membership test: for each container voxel, its signed distance along
    `normal` (`d_n`) and its in-plane radial distance from `center`
    (`d_r`) must satisfy `(d_n/half_thickness)^2 + (d_r/half_width)^2 <= 1`
    -- the short axis (half_thickness) runs along the {111} habit-plane
    normal, matching the slab twins' own orientation convention; the two
    long in-plane axes are both half_width (an oblate spheroid, thin like
    a real twin plate).

    Parameters mirror pxtalops.twin3d.introduce_twin_lamella_3d's shape
    (lgi, container id, geometry, next_gid) but add the second,
    in-plane-bounding radius that makes this an island instead of a slab.

    Returns
    -------
    island_voxels : ndarray (N, 3) or None
        Global voxel coordinates of the carved region, or None if no
        voxels were selected.
    """
    coords = (container_coords if container_coords is not None
             else np.argwhere(lgi == container_gid))
    if coords.shape[0] == 0:
        return None

    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # R: no voxel farther than this (the ellipsoid's own largest semi-axis,
    # +1 slack) from `center` in any single coordinate can possibly
    # satisfy the membership test below.
    R = max(half_thickness, half_width) + 1.0
    ht2 = half_thickness * half_thickness
    hw2 = half_width * half_width

    if _HAS_NUMBA:
        # The bbox reject and the exact ellipsoid test run together in a
        # single scalar pass over the raw (int) container coords -- no
        # intermediate numpy allocation at all, so there is no benefit to
        # pre-filtering in numpy first the way the fallback below needs.
        sel = _ellipsoid_select_numba(
            coords, center[0], center[1], center[2],
            normal[0], normal[1], normal[2], ht2, hw2, R)
        if not np.any(sel):
            return None
    else:
        # Cheap axis-aligned bounding-box pre-filter first: for a
        # container that spans a large fraction of the domain (an
        # ordinary host grain, say), this cuts the expensive vector math
        # (rel/outer/norm, O(container size) every single attempt,
        # including every failed one) down to just the local
        # neighbourhood the island could actually reach.
        bbox_sel = (
            (np.abs(coords[:, 0] - center[0]) <= R)
            & (np.abs(coords[:, 1] - center[1]) <= R)
            & (np.abs(coords[:, 2] - center[2]) <= R)
        )
        coords = coords[bbox_sel]
        if coords.shape[0] == 0:
            return None

        # Squared distances throughout (no sqrt-then-square round trip
        # through np.linalg.norm) -- matches the numba kernel's own
        # arithmetic exactly, so the two paths agree even for inputs that
        # land a voxel precisely on the ellipsoid's surface, not just in
        # its interior/exterior.
        rel = coords.astype(np.float64) - center
        d_n = rel @ normal
        d_r2 = np.sum((rel - np.outer(d_n, normal)) ** 2, axis=1)
        sel = (d_n * d_n) / ht2 + d_r2 / hw2 <= 1.0
        if not np.any(sel):
            return None

    island_voxels = coords[sel]
    lgi[island_voxels[:, 0], island_voxels[:, 1], island_voxels[:, 2]] = next_gid
    return island_voxels


class IslandTwinIntroducer:
    """Introduces island twins into a final, cleaned post-twin structure.

    Parameters
    ----------
    lgi_clean, all_quats_clean, twin_role_clean, twin_parent_of_clean :
        StructureCleaner3D's own output attributes -- the structure this
        operates on and extends in place.
    voxel_size : float
        Physical voxel edge length (same units as twin_thickness/
        island_vf_target apply in).
    category_probs : dict
        {'remaining_parent', 'non_host', 'primary_twin'} -> probability,
        for whichever subset the user selected (missing/zero-probability
        categories are simply never drawn from). Must sum to 1 over the
        selected categories.
    island_vf_target : float
        Target ADDITIONAL volume fraction from islands alone, relative to
        the structure's total voxel count -- additive on top of whatever
        the existing twins already achieved, not drawn from the same
        budget. 0 (the default) introduces no islands at all.
    twin_thickness : dict
        Same EBSD twin-thickness dict (`{'thick_um': array, ...}`)
        TwinGenerator3D consumes -- islands' short axis bootstrap-samples
        from this SAME array, for consistency with the lamella twins.
    island_aspect_ratio : float
        long (in-plane) axis / short (thickness) axis -- see module
        docstring for why this, not a separately-calibrated distribution.
    thick_scale_factor : float
        Same role as TwinGenerator3D's twin_thick_scale_factor, applied
        to the sampled thickness before computing the long axis from it.
    twin_orient_scatter_deg : float
        Same meaning as TwinGenerator3D's -- Gaussian scatter added to
        the exact Sigma3 orientation relationship.
    max_attempts_per_island : int
        Retries (a fresh category + container grain + size + placement
        each time) before giving up on one island and moving to the next.
    max_total_attempts : int
        Hard safety cap on attempts across ALL islands combined, in case
        island_vf_target is unreachable (e.g. too few/too small eligible
        regions) -- stops instead of looping forever.
    rng_seed : int or None
    """

    def __init__(
            self,
            lgi_clean: np.ndarray,
            all_quats_clean: Dict[int, np.ndarray],
            twin_role_clean: Dict[int, str],
            twin_parent_of_clean: Dict[int, int],
            voxel_size: float,
            category_probs: Dict[str, float],
            island_vf_target: float = 0.0,
            twin_thickness: Optional[Dict] = None,
            island_aspect_ratio: float = 4.0,
            thick_scale_factor: float = 1.0,
            twin_orient_scatter_deg: float = 1.5,
            max_attempts_per_island: int = 20,
            max_total_attempts: int = 2000,
            rng_seed: Optional[int] = None,
    ):
        if island_vf_target < 0:
            raise ValueError('island_vf_target must be >= 0.')
        if island_aspect_ratio <= 0:
            raise ValueError('island_aspect_ratio must be > 0.')
        probs = {k: float(v) for k, v in category_probs.items() if v > 0}
        if not probs:
            if island_vf_target > 0:
                raise ValueError(
                    'At least one category must have a positive probability '
                    'when island_vf_target > 0.')
        elif abs(sum(probs.values()) - 1.0) > 1e-6:
            raise ValueError(
                f'category_probs must sum to 1.0 over the selected '
                f'categories, got {sum(probs.values()):.4f}.')

        self.lgi_islanded = lgi_clean.copy()
        self.all_quats = dict(all_quats_clean)
        self.twin_role = dict(twin_role_clean)
        self.twin_parent_of = dict(twin_parent_of_clean)
        self.voxel_size = float(voxel_size)
        self.category_probs = probs
        self.island_vf_target = float(island_vf_target)
        self.twin_thickness = twin_thickness
        self.island_aspect_ratio = float(island_aspect_ratio)
        self.thick_scale_factor = float(thick_scale_factor)
        self.twin_orient_scatter_deg = float(twin_orient_scatter_deg)
        self.max_attempts_per_island = int(max_attempts_per_island)
        self.max_total_attempts = int(max_total_attempts)
        self._rng = np.random.default_rng(rng_seed)

        self.island_quats: Dict[int, np.ndarray] = {}
        self.island_halfwidths_vox: Dict[int, float] = {}
        self.island_category_of: Dict[int, str] = {}
        self.cumulative_island_vox: int = 0
        self.n_islands_placed: int = 0
        self.n_attempts: int = 0
        self.n_failed_placements: int = 0
        self.vf_stopped_early: bool = False
        # Attempts per category, successful or not -- lets a caller tell
        # "this category was never drawn" apart from "this category was
        # drawn repeatedly but never fit" (e.g. a thin primary-twin slab
        # too narrow for the sampled island thickness range -- a genuine
        # geometric fact, not a bug, but silent otherwise).
        self.n_attempts_by_category: Dict[str, int] = {c: 0 for c in ISLAND_CATEGORIES}
        # {container_gid: (global_coords, local_dist, offset)} -- global_coords
        # are this container's own voxel coordinates (for the ellipsoid
        # carve), local_dist is its boundary-distance array cropped to its
        # own (padded) bounding box rather than the full domain, and offset
        # is the (x,y,z) origin of that crop (subtract from a global coord
        # to index into local_dist). Reused across every attempt that
        # targets the same container without having actually changed it
        # (most attempts, including every failed one), instead of
        # recomputing a full distance transform -- or even rescanning the
        # whole domain for this container's coordinates -- per attempt.
        # Popped (invalidated) only when a successful carve actually
        # changes that specific container's mask.
        self._container_cache: Dict[int, tuple] = {}
        # Each grain's bounding box as it stood BEFORE any islands were
        # introduced, computed once (one O(domain) pass) up front. Island
        # carving only ever REMOVES voxels from a container, never adds
        # any outside its original footprint, so this box can never
        # shrink the true extent -- using it (slightly padded) to crop
        # every later coords/distance recomputation for that container is
        # always safe, just possibly a little more generous than the
        # tightest-possible box would be.
        from scipy.ndimage import find_objects
        self._container_bbox: Dict[int, tuple] = {}
        if lgi_clean.size:
            for gid, obj in enumerate(find_objects(lgi_clean), start=1):
                if obj is not None:
                    self._container_bbox[gid] = obj

    # ── Category -> container grain id set ──────────────────────────────

    def _category_grain_ids(self, category: str) -> set:
        if category == 'remaining_parent':
            return {gid for gid, role in self.twin_role.items() if role == 'host'}
        if category == 'non_host':
            return {gid for gid, role in self.twin_role.items() if role == 'non_host'}
        if category == 'primary_twin':
            return {gid for gid, role in self.twin_role.items() if role == 'primary_twin'}
        raise ValueError(f'Unknown island category {category!r}.')

    def _sample_category(self) -> str:
        cats = list(self.category_probs)
        weights = np.array([self.category_probs[c] for c in cats])
        return cats[self._rng.choice(len(cats), p=weights)]

    def _container_data(self, container_gid: int):
        """Returns (global_coords, local_dist, offset) for this container,
        computed within its own (padded) bounding box rather than the full
        domain -- see `_container_cache`'s docstring above."""
        cached = self._container_cache.get(container_gid)
        if cached is not None:
            return cached

        shape = self.lgi_islanded.shape
        box = self._container_bbox.get(container_gid)
        if box is None:
            # No precomputed box (shouldn't happen for a genuine category
            # grain id, which always existed in lgi_clean at construction
            # time) -- fall back to a full-domain scan rather than fail.
            coords = np.argwhere(self.lgi_islanded == container_gid)
            local_dist = _boundary_distance(self.lgi_islanded == container_gid)
            offset = (0, 0, 0)
            result = (coords, local_dist, offset)
            self._container_cache[container_gid] = result
            return result

        # Pad the ORIGINAL (pre-island, never-shrinking) bbox by 1 voxel
        # on each side, clipped to the domain -- the same single-voxel
        # margin `_boundary_distance`'s own np.pad gives the full domain,
        # just applied to a crop instead so the expensive EDT call only
        # ever touches this container's own neighbourhood, not every
        # other voxel in the RVE.
        lo = [max(s.start - 1, 0) for s in box]
        hi = [min(s.stop + 1, shape[i]) for i, s in enumerate(box)]
        sl = tuple(slice(lo[i], hi[i]) for i in range(3))
        local_mask = self.lgi_islanded[sl] == container_gid
        local_coords = np.argwhere(local_mask)
        offset = tuple(lo)
        coords = local_coords + np.array(offset)
        local_dist = _boundary_distance(local_mask)

        result = (coords, local_dist, offset)
        self._container_cache[container_gid] = result
        return result

    def _sample_half_axes(self):
        thick_um = (self.twin_thickness['thick_um'] * self.thick_scale_factor
                   if self.twin_thickness is not None else None)
        if thick_um is None or len(thick_um) == 0:
            raise ValueError('twin_thickness with a non-empty thick_um array is required.')
        thickness_um = float(self._rng.choice(thick_um))
        half_thickness = max(1.0, thickness_um / self.voxel_size / 2.0)
        half_width = max(1.0, half_thickness * self.island_aspect_ratio)
        return half_thickness, half_width

    def introduce_islands(self, verbose: bool = True):
        """Accumulate islands (sample category -> container grain -> size
        -> placement attempt) until island_vf_target is reached or the
        attempt budget runs out."""
        total_vox = int(np.sum(self.lgi_islanded > 0))
        vf_stop_vox = self.island_vf_target * total_vox
        next_gid = int(self.lgi_islanded.max()) + 1
        t0 = time.perf_counter()

        if not self.category_probs or vf_stop_vox <= 0:
            if verbose:
                print('IslandTwinIntroducer: island_vf_target is 0 (or no '
                      'category selected) -- no islands introduced.')
            return

        empty_categories = [c for c in self.category_probs
                           if not self._category_grain_ids(c)]
        if empty_categories and verbose:
            print(f'IslandTwinIntroducer WARNING: categor{"y" if len(empty_categories)==1 else "ies"} '
                 f'{empty_categories} ha{"s" if len(empty_categories)==1 else "ve"} no eligible grains -- '
                 f'every attempt drawing {"it" if len(empty_categories)==1 else "them"} will fail '
                 f'immediately, wasting part of the attempt budget on a known-unreachable share '
                 f'of island_vf_target.')

        from upxo.xtalphy.crystal_orientation import compute_s3_habit_plane_3d

        for _ in progress.track(range(self.max_total_attempts), message="Introducing island twins"):
            if self.cumulative_island_vox >= vf_stop_vox:
                break
            self.n_attempts += 1

            # Category is fixed for this whole island attempt -- resampling
            # it on every inner retry would let whichever category is
            # geometrically easiest to place in silently dominate the
            # achieved mix (the first attempt that succeeds wins,
            # regardless of which category it came from), defeating the
            # user's chosen category_probs. A category that's genuinely
            # hard to place in should fail honestly (counted below) and
            # free the next island attempt to draw a fresh category, not
            # get silently substituted within the same attempt.
            category = self._sample_category()
            candidates = self._category_grain_ids(category)
            self.n_attempts_by_category[category] += 1

            placed = False
            for _attempt in range(self.max_attempts_per_island):
                if not candidates:
                    continue
                container_gid = int(self._rng.choice(list(candidates)))
                q_par = self.all_quats.get(container_gid)
                if q_par is None:
                    continue

                half_thickness, half_width = self._sample_half_axes()
                coords, local_dist, offset = self._container_data(container_gid)
                off = np.array(offset)

                # Cheap, LENIENT pre-filter: only the thin axis (the
                # ellipsoid's short semi-axis, along the habit-plane
                # normal) needs isotropic clearance to find a reasonable
                # starting region -- using half_width (the long, in-plane
                # semi-axis) here instead, as an earlier version of this
                # code did, massively over-constrains placement: it
                # demands width-sized clearance in EVERY direction,
                # including the one the ellipsoid is actually thin along.
                # At a realistic (fine) voxel size this routinely exceeds
                # the size of ordinary grains -- and primary twins
                # specifically, being thin slabs by design, are often
                # smaller than that requirement outright, making the
                # 'primary_twin' category nearly always fail. The wide
                # in-plane axes are instead checked EXACTLY, post-carve,
                # below -- introduce_island_twin_3d already clips the
                # carve to the container's own voxels, so all that
                # actually needs verifying is that none of the carved
                # voxels sit right at the container's true boundary.
                # dist_vals is indexed against `coords` (global, same
                # order as local_dist's own lookup) -- O(container size),
                # not O(domain size), since both now come from the
                # bounding-box-cropped cache instead of a full-domain
                # array/argwhere.
                prefilter_margin = half_thickness + 1.0
                local_coords = coords - off
                dist_vals = local_dist[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
                centers_sel = dist_vals > prefilter_margin
                if not np.any(centers_sel):
                    continue
                centers = coords[centers_sel]
                center = centers[self._rng.integers(len(centers))].astype(float)

                var_idx = int(self._rng.integers(0, 4))
                normal = compute_s3_habit_plane_3d(q_par, self._rng, variant_idx=var_idx)

                island_voxels = introduce_island_twin_3d(
                    self.lgi_islanded, container_gid, center, normal,
                    half_thickness, half_width, next_gid, container_coords=coords)
                if island_voxels is None or len(island_voxels) == 0:
                    continue

                # Exact anisotropic verification: every carved voxel must
                # be strictly MORE than 1 voxel (not >=, which would
                # permit the island's surface sitting exactly at a
                # face-adjacent obstacle -- touching, not separated) from
                # the container's TRUE boundary, per the distance array
                # computed before this carve. Catches the wide in-plane
                # axes poking too close to a DIFFERENT boundary than the
                # one the cheap thin-axis-only prefilter above checked.
                # Reject (revert the carve) and retry rather than accept
                # a touching "island".
                iv = island_voxels
                iv_local = iv - off
                if not np.all(local_dist[iv_local[:, 0], iv_local[:, 1], iv_local[:, 2]] > 1.0):
                    self.lgi_islanded[iv[:, 0], iv[:, 1], iv[:, 2]] = container_gid
                    continue

                from upxo.xtalphy.crystal_orientation import (
                    _SIGMA3_Q_ALL_VARIANTS, _quat_mul, _positive_w)
                q_island = _quat_mul(_SIGMA3_Q_ALL_VARIANTS[var_idx], q_par)
                if self.twin_orient_scatter_deg > 0.0:
                    sc = self._rng.normal(0.0, np.radians(self.twin_orient_scatter_deg))
                    ax = self._rng.standard_normal(3)
                    ax /= np.linalg.norm(ax)
                    pt = np.array([np.cos(sc / 2.0), *np.sin(sc / 2.0) * ax])
                    q_island = _quat_mul(pt, q_island)
                q_island = _positive_w(q_island)

                self.all_quats[next_gid] = q_island
                self.island_quats[next_gid] = q_island
                self.twin_role[next_gid] = 'island_twin'
                self.twin_parent_of[next_gid] = container_gid
                self.island_halfwidths_vox[next_gid] = half_thickness
                self.island_category_of[next_gid] = category
                self.cumulative_island_vox += len(island_voxels)
                self.n_islands_placed += 1
                next_gid += 1
                placed = True
                # This container's own voxels just changed (some became
                # the new island) -- its cached coords/distance are stale
                # now; fresh ones are computed lazily next time anything
                # targets this same container again.
                self._container_cache.pop(container_gid, None)
                break

            if not placed:
                self.n_failed_placements += 1

        if self.cumulative_island_vox < vf_stop_vox:
            self.vf_stopped_early = True

        if verbose:
            achieved_vf = self.cumulative_island_vox / total_vox if total_vox else 0.0
            print(f'IslandTwinIntroducer: placed {self.n_islands_placed} island(s), '
                 f'achieved VF {achieved_vf:.4f} (target {self.island_vf_target:.4f}), '
                 f'{self.n_failed_placements} failed attempt(s)  '
                 f'({time.perf_counter() - t0:.1f}s)')
            if self.vf_stopped_early:
                print('IslandTwinIntroducer WARNING: stopped before reaching '
                     'island_vf_target -- ran out of eligible placements '
                     '(too few/too small regions in the selected categories).')

    def summary(self) -> Dict:
        total_vox = int(np.sum(self.lgi_islanded > 0))
        achieved_vf = self.cumulative_island_vox / total_vox if total_vox else 0.0
        by_category = {c: sum(1 for v in self.island_category_of.values() if v == c)
                       for c in ISLAND_CATEGORIES}
        return {
            'n_islands_placed': self.n_islands_placed,
            'island_vf_target': self.island_vf_target,
            'island_vf_achieved': achieved_vf,
            'vf_stopped_early': self.vf_stopped_early,
            'n_attempts': self.n_attempts,
            'n_failed_placements': self.n_failed_placements,
            'n_by_category': by_category,
            'n_attempts_by_category': dict(self.n_attempts_by_category),
        }
