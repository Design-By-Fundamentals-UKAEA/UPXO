"""
twin_generator_3d.py
====================
Primary and secondary Sigma3 twin lamella introduction for the
twinned simple 3D pipeline.

Path A note: this module is hardcoded for Sigma3 (FCC annealing twin)
regardless of the CSL label selected in Part A.  A warning is emitted
when the selected CSL label is not 'S3  (twin)'.  Generalisation to
other CSL types (Path B / csl-registry) is deferred -- see project
memory 'csl-registry-path-b'.
"""

import time
import numpy as np
from typing import Optional, Dict, List

_SUPPORTED_CSL = 'S3  (twin)'
_VALID_NUCLEATION_SITES = ('centroid', 'gb_centroid', 'random_gb')


class TwinGenerator3D:
    """
    Introduces Sigma3 twin lamellae into a 3D grain structure.

    Primary lamellae
    ----------------
    Each host grain receives up to ``n_lamellae_per_host`` lamella
    carvings.  The first lamella (k=0) always uses ``twin_nucleation_site``
    to determine its plane origin.  Subsequent lamellae (k >= 1) are
    placed according to ``prob_lamella_separated``:

    * Separated (prob_lamella_separated): independent placement using
      ``twin_nucleation_site`` on the remaining host voxels.
    * Contacting (prob_lamella_contacting): geometric offset from
      lamella-0's outer face, guaranteeing face-adjacency and hence
      Sigma9 misorientation between the two primary lamellae.
      ``twin_nucleation_site`` is overridden for this path.

    Different {111} variants are pre-selected *without replacement* per
    grain so that contacting lamellae always use different variants,
    producing Sigma9 (not Sigma1/Sigma3) at the twin-twin boundary.

    Secondary lamellae
    ------------------
    Secondary twins nucleate from the primary twin's surface according to
    ``prob_secondary_outward_twinNucleation``:

    * Outward 2a: carved from the HOST grain at the primary-host
      interface centroid.  Creates a direct planar Sigma9 boundary
      between secondary and host.  ``twin_nucleation_site`` is overridden.
    * Inward 2b: carved from INSIDE the primary twin (nested).  Creates
      a Sigma3 boundary with the primary and a Sigma9 boundary with the
      host if the lamella spans the full primary width.  Uses
      ``twin_nucleation_site`` on the primary twin voxel set.

    Volume-fraction control
    -----------------------
    ``cumulative_twin_vox`` is tracked across BOTH primary and secondary
    introduction.  Introduction stops when the running twin VF exceeds
    ``tvf['overall_twin_frac'] * (1 + tvf_tolerance)``.
    """

    # Human-readable identity of the twin type this class actually
    # generates -- always Sigma3 (Path A), regardless of which CSL label
    # was used upstream to select the host pool or measure the EBSD target
    # (see the CSL-label-mismatch check in :meth:`diagnose`).  Read by the
    # caller's results block instead of re-deriving a label from caller state, so
    # a future multi-CSL Path B implementation only needs to change this
    # (or make it per-instance) rather than touching that caller.
    TWIN_TYPE_LABEL = 'Sigma3 -- FCC Annealing Twin (60 deg / <111>)'

    # Diagnostic thresholds (see `diagnose`) -- heuristic cutoffs for
    # flagging results worth a closer look, not correctness criteria.
    # Deliberately not exposed as user-configurable settings: they gate
    # advisory notes, not pass/fail, so there's no reason for a user to
    # need to tune them to get past a screen.
    DIAG_SHORTFALL_PCT_THRESHOLD = 70.0
    DIAG_ABRUPT_FRAC_THRESHOLD = 0.5
    DIAG_CLAMPED_FRAC_THRESHOLD = 0.3
    DIAG_HOST_SATURATION_THRESHOLD = 0.95

    __slots__ = (
        'base',
        'n_lamellae_per_host',
        'twin_nucleation_site',
        'tvf_tolerance',
        'twin_orient_scatter_deg',
        'meshing_route',
        'min_lamella_thickness_conformal',
        'min_lamella_thickness_non_conformal',
        'min_host_vox_for_lamella',
        'prob_lamella_separated',
        'prob_lamella_contacting',
        'prob_secondary_outward_twinNucleation',
        'prob_secondary_inward_twinNucleation',
        'lgi_twinned',
        'primary_twin_quats',
        'secondary_twin_quats',
        'all_quats',
        'twin_role',
        'twin_parent_of',
        'max_lamella_thickness_um',
        'max_lamella_vf_per_host',
        'n_clamped',
        'n_abrupt_primary',
        'n_hosts_with_primary_twin',
        'vf_stopped_early',
        'cumulative_twin_vox',
        'twin_halfwidths_vox',
        'twin_thick_scale_factor',
        'tvf_2d_to_3d_scale_factor',
        'tvf_2d_ebsd',
        'tvf_target_3d',
        'schmid_loading_direction',
        'host_schmid_weight',
        'use_schmid_for_variant_selection',
        '_rng',
        '_grain_coords',   # {gid: ndarray(M,3)} voxel index — built once, updated per carve
    )

    def __init__(
            self,
            base,
            n_lamellae_per_host: int = 2,
            twin_nucleation_site: str = 'random_gb',
            tvf_tolerance: float = 0.05,
            twin_orient_scatter_deg: float = 1.5,
            meshing_route: str = 'conformal',
            min_lamella_thickness_conformal: int = 2,
            min_lamella_thickness_non_conformal: int = 1,
            min_host_vox_for_lamella: int = 8,
            prob_lamella_separated: float = 0.5,
            prob_lamella_contacting: float = 0.5,
            prob_secondary_outward_twinNucleation: float = 0.3,
            prob_secondary_inward_twinNucleation: float = 0.7,
            twin_thick_scale_factor: float = 1.0,
            tvf_2d_to_3d_scale_factor: float = 1.0,
            max_lamella_thickness_um: Optional[float] = None,
            max_lamella_vf_per_host: float = 0.4,
            schmid_loading_direction: tuple = (0., 0., 1.),
            host_schmid_weight: float = 0.0,
            use_schmid_for_variant_selection: bool = True,
            rng_seed: Optional[int] = None,
    ):
        self.base = base
        self.n_lamellae_per_host = n_lamellae_per_host
        self.twin_nucleation_site = twin_nucleation_site
        self.tvf_tolerance = tvf_tolerance
        self.twin_orient_scatter_deg = twin_orient_scatter_deg
        self.meshing_route = meshing_route
        self.min_lamella_thickness_conformal = min_lamella_thickness_conformal
        self.min_lamella_thickness_non_conformal = min_lamella_thickness_non_conformal
        self.min_host_vox_for_lamella = min_host_vox_for_lamella
        self.prob_lamella_separated = prob_lamella_separated
        self.prob_lamella_contacting = prob_lamella_contacting
        self.prob_secondary_outward_twinNucleation = prob_secondary_outward_twinNucleation
        self.prob_secondary_inward_twinNucleation = prob_secondary_inward_twinNucleation
        self.lgi_twinned: Optional[np.ndarray] = None
        self.primary_twin_quats: Dict = {}
        self.secondary_twin_quats: Dict = {}
        self.all_quats: Dict = {}
        self.twin_role: Dict = {}
        self.twin_parent_of: Dict = {}
        self.n_clamped: int = 0
        self.n_abrupt_primary: int = 0
        self.n_hosts_with_primary_twin: int = 0
        self.vf_stopped_early: bool = False
        self.cumulative_twin_vox: int = 0
        self.twin_halfwidths_vox: Dict = {}
        self.twin_thick_scale_factor: float     = float(twin_thick_scale_factor)
        self.tvf_2d_to_3d_scale_factor: float   = float(tvf_2d_to_3d_scale_factor)
        self.max_lamella_thickness_um: Optional[float] = (
            float(max_lamella_thickness_um) if max_lamella_thickness_um is not None else None)
        self.max_lamella_vf_per_host: float = float(np.clip(max_lamella_vf_per_host, 0.01, 1.0))
        self.tvf_2d_ebsd: float = 0.0
        self.tvf_target_3d: float = 0.0
        self.schmid_loading_direction: tuple = tuple(float(v) for v in schmid_loading_direction)
        self.host_schmid_weight: float = float(np.clip(host_schmid_weight, 0.0, 1.0))
        self.use_schmid_for_variant_selection: bool = bool(use_schmid_for_variant_selection)
        self._rng = np.random.default_rng(rng_seed)
        self._grain_coords: Dict = {}  # populated in introduce_primary_twins

        if twin_thick_scale_factor <= 0:
            raise ValueError('twin_thick_scale_factor must be > 0.')
        if tvf_2d_to_3d_scale_factor <= 0:
            raise ValueError('tvf_2d_to_3d_scale_factor must be > 0.')

        # ── Validation warnings ────────────────────────────────────────────
        if twin_nucleation_site not in _VALID_NUCLEATION_SITES:
            raise ValueError(
                f'twin_nucleation_site must be one of '
                f'{_VALID_NUCLEATION_SITES}, got "{twin_nucleation_site}"')

        if abs(prob_lamella_separated + prob_lamella_contacting - 1.0) > 1e-6:
            print(
                f'TwinGenerator3D WARNING: '
                f'prob_lamella_separated ({prob_lamella_separated:.4f}) + '
                f'prob_lamella_contacting ({prob_lamella_contacting:.4f}) '
                f'!= 1.0')

        if abs(prob_secondary_outward_twinNucleation
               + prob_secondary_inward_twinNucleation - 1.0) > 1e-6:
            print(
                f'TwinGenerator3D WARNING: '
                f'prob_secondary_outward_twinNucleation '
                f'({prob_secondary_outward_twinNucleation:.4f}) + '
                f'prob_secondary_inward_twinNucleation '
                f'({prob_secondary_inward_twinNucleation:.4f}) != 1.0')

        if n_lamellae_per_host == 1 and (
                abs(prob_lamella_separated - 0.5) > 1e-9 or
                abs(prob_lamella_contacting - 0.5) > 1e-9):
            print(
                'TwinGenerator3D NOTE: n_lamellae_per_host=1 so '
                'prob_lamella_separated / prob_lamella_contacting '
                'have no effect (no k>=1 lamellae are attempted).')

    # ── Static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _find_gb_voxels(gid: int,
                        coords: np.ndarray,
                        lgi: np.ndarray) -> np.ndarray:
        """
        Return the subset of *coords* (voxels of grain *gid*) that are
        face-adjacent to any voxel with a different label (including the
        domain boundary, which counts as a grain boundary).

        Vectorised over 6 face directions — O(len(coords)).
        """
        sx, sy, sz = lgi.shape
        c = coords.astype(int)
        is_gb = np.zeros(len(c), dtype=bool)
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),
                            (0,-1,0),(0,0,1),(0,0,-1)]:
            nx_ = c[:, 0] + dx
            ny_ = c[:, 1] + dy
            nz_ = c[:, 2] + dz
            oob = ((nx_ < 0) | (nx_ >= sx) |
                   (ny_ < 0) | (ny_ >= sy) |
                   (nz_ < 0) | (nz_ >= sz))
            # Domain boundary voxels count as GB
            in_bounds = ~oob
            nbr_labels = np.where(
                in_bounds,
                lgi[np.clip(nx_, 0, sx-1),
                    np.clip(ny_, 0, sy-1),
                    np.clip(nz_, 0, sz-1)],
                -1)   # -1 ≠ any valid gid
            is_gb |= oob | (nbr_labels != gid)
        return coords[is_gb].astype(float)

    @staticmethod
    def _find_gb_voxels_facing_neighbor(gid: int,
                                        neighbor_gid: int,
                                        coords: np.ndarray,
                                        lgi: np.ndarray) -> np.ndarray:
        """
        Return the subset of *coords* (voxels of grain *gid*) that are
        face-adjacent specifically to grain *neighbor_gid*.

        Used to build ``gb_pool_primary_twin`` for inward secondary (2b)
        with ``twin_nucleation_site='random_gb'``.
        """
        sx, sy, sz = lgi.shape
        c = coords.astype(int)
        is_facing = np.zeros(len(c), dtype=bool)
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),
                            (0,-1,0),(0,0,1),(0,0,-1)]:
            nx_ = c[:, 0] + dx
            ny_ = c[:, 1] + dy
            nz_ = c[:, 2] + dz
            in_bounds = ((nx_ >= 0) & (nx_ < sx) &
                         (ny_ >= 0) & (ny_ < sy) &
                         (nz_ >= 0) & (nz_ < sz))
            nbr_labels = np.where(
                in_bounds,
                lgi[np.clip(nx_, 0, sx-1),
                    np.clip(ny_, 0, sy-1),
                    np.clip(nz_, 0, sz-1)],
                -1)
            is_facing |= (nbr_labels == neighbor_gid)
        return coords[is_facing].astype(float)

    @staticmethod
    def _snap_to_grain(origin: np.ndarray,
                       gid: int,
                       coords: np.ndarray,
                       lgi: np.ndarray) -> np.ndarray:
        """
        If *origin* does not fall inside grain *gid* in *lgi*, snap it
        to the nearest voxel in *coords*.  Handles non-convex grains and
        recombined grain chains where the centroid may lie outside the
        grain volume.
        """
        origin_int = np.clip(
            np.round(origin).astype(int), 0,
            np.array(lgi.shape) - 1)
        if int(lgi[tuple(origin_int)]) != gid:
            dists = np.linalg.norm(coords - origin, axis=1)
            return coords[np.argmin(dists)].astype(float)
        return origin.copy()

    def _get_grain_origin(self,
                          gid: int,
                          coords: np.ndarray,
                          lgi: np.ndarray) -> np.ndarray:
        """
        Compute the habit-plane origin for a new lamella in grain *gid*
        according to ``self.twin_nucleation_site``.

        Returns a coordinate guaranteed to be inside (or on the boundary
        of) grain *gid*.
        """
        if self.twin_nucleation_site == 'random_gb':
            gb_pool = self._find_gb_voxels(gid, coords, lgi)
            if len(gb_pool) > 0:
                return gb_pool[int(self._rng.integers(len(gb_pool)))].copy()
            # Fallback: centroid (grain interior with no detected boundary)
            return coords.mean(axis=0)

        elif self.twin_nucleation_site == 'gb_centroid':
            gb_pool = self._find_gb_voxels(gid, coords, lgi)
            origin = gb_pool.mean(axis=0) if len(gb_pool) > 0 \
                else coords.mean(axis=0)
            return self._snap_to_grain(origin, gid, coords, lgi)

        else:  # 'centroid'
            origin = coords.mean(axis=0)
            return self._snap_to_grain(origin, gid, coords, lgi)

    def _min_hw(self) -> float:
        """Effective minimum lamella half-width for current meshing route."""
        if self.meshing_route == 'conformal':
            return self.min_lamella_thickness_conformal / 2.0
        return self.min_lamella_thickness_non_conformal / 2.0

    def _apply_sigma3_variant(self,
                               q_parent: np.ndarray,
                               variant_idx: int) -> np.ndarray:
        """Apply Sigma3 rotation for *variant_idx* + Gaussian scatter."""
        from upxo.xtalphy.crystal_orientation import (
            _SIGMA3_Q_ALL_VARIANTS, _quat_mul, _positive_w)
        q_twin = _quat_mul(_SIGMA3_Q_ALL_VARIANTS[variant_idx], q_parent)
        if self.twin_orient_scatter_deg > 0.0:
            sc = self._rng.normal(0.0, np.radians(self.twin_orient_scatter_deg))
            ax = self._rng.standard_normal(3)
            ax /= np.linalg.norm(ax)
            pt = np.array([np.cos(sc / 2.0), *np.sin(sc / 2.0) * ax])
            q_twin = _quat_mul(pt, q_twin)
        return _positive_w(q_twin)

    # ── Public API ────────────────────────────────────────────────────────

    def introduce_primary_twins(
            self,
            host_orientations: Dict[int, np.ndarray],
            twin_thickness: Dict,
            tvf: Optional[Dict] = None,
        tvf_stage1: Optional[float] = None,
    ):
        """
        Introduce primary Sigma3 twin lamellae into all designated host
        grains.

        Parameters
        ----------
        host_orientations : dict {gid: ndarray(4,)}
        twin_thickness : dict
            Output of ``rg.compute_mc_twin_thickness(parent_info)``.
        tvf : dict or None
            Output of ``rg.compute_ebsd_tvf(parent_info)``.  Provides
            ``overall_twin_frac`` (VF stopping target) and ``csl_label``
            (for Path A CSL warning).
        """
        from upxo.xtalphy.crystal_orientation import compute_s3_habit_plane_3d
        from upxo.pxtalops.twin3d import introduce_twin_lamella_3d

        if self.base.host_grain_ids is None:
            raise RuntimeError('Call base.allocate_twin_hosts() first.')

        # Path A CSL check
        if tvf is not None:
            csl = tvf.get('csl_label', '')
            if csl and csl != _SUPPORTED_CSL:
                print(
                    f'TwinGenerator3D WARNING: selected CSL "{csl}" but '
                    f'only "{_SUPPORTED_CSL}" is supported (Path A). '
                    f'Twin orientations will still use Sigma3.')

        # VF stopping parameters — scale EBSD 2D area fraction to 3D target
        self.tvf_2d_ebsd   = tvf.get('overall_twin_frac', 1.0) if tvf else 1.0
        self.tvf_target_3d = self.tvf_2d_ebsd * self.tvf_2d_to_3d_scale_factor
        tvf_stop           = self.tvf_target_3d * (1.0 + self.tvf_tolerance)
        # tvf_stage1 overrides: use EBSD-partitioned primary target (vf_int + vf_2b)
        # instead of overall_twin_frac, so Stage-2 secondaries can add vf_2a on top.
        if tvf_stage1 is not None:
            self.tvf_target_3d = float(tvf_stage1)
            tvf_stop           = self.tvf_target_3d * (1.0 + self.tvf_tolerance)
        total_vox          = int(np.sum(self.base.lgi > 0))

        # Initialise output structures
        self.lgi_twinned  = self.base.lgi.copy()
        # Initialise ALL base grain IDs as 'non_host' so no grain falls through
        # without a role — safer than restricting to non_host_grain_ids alone.
        self.twin_role    = {gid: 'non_host' for gid in self.base.grain_ids}
        self.all_quats    = dict(host_orientations)
        self.twin_parent_of           = {}
        self.primary_twin_quats       = {}
        self.twin_halfwidths_vox      = {}
        self.n_clamped                = 0
        self.n_abrupt_primary         = 0
        self.n_hosts_with_primary_twin = 0
        self.cumulative_twin_vox      = 0
        self.vf_stopped_early         = False

        thick_um      = twin_thickness['thick_um'] * self.twin_thick_scale_factor
        abrupt_frac   = twin_thickness.get('abrupt_frac_ebsd', 0.0)
        vs            = self.base.voxel_size
        twin_minHalfWidth = self._min_hw()

        # Optionally reorder hosts by max Schmid factor (descending)
        # so that crystallographically favourable grains are twinned first
        # and the VF ceiling cuts off unfavourable ones naturally.
        _host_list = list(self.base.host_grain_ids)
        if self.host_schmid_weight > 0.0:
            from upxo.xtalphy.crystal_orientation import schmid_factors_fcc_twinning
            _schmid_max = {}
            for _gid in _host_list:
                _q = host_orientations.get(_gid)
                if _q is not None:
                    _m, _ = schmid_factors_fcc_twinning(_q, self.schmid_loading_direction)
                    _schmid_max[_gid] = float(_m.max())
                else:
                    _schmid_max[_gid] = 0.0
            _host_list = sorted(_host_list,
                                key=lambda g: _schmid_max[g],
                                reverse=True)
        else:
            _host_list = _host_list  # preserve existing order

        next_gid   = int(self.lgi_twinned.max()) + 1
        _noted_contacting_override = False

        # ── Build grain voxel index once (replaces O(N_vox) np.argwhere per call) ──
        print('  [S29] Building grain voxel index...', end='', flush=True)
        _t_idx = time.perf_counter()
        _lgi_flat = self.lgi_twinned.ravel()
        _sort_idx = np.argsort(_lgi_flat, kind='stable')
        _sorted   = _lgi_flat[_sort_idx]
        _bounds   = np.where(np.diff(_sorted, prepend=_sorted[0] - 1))[0]
        _bounds   = np.append(_bounds, len(_sorted))
        _gc: Dict = {}
        for _i in range(len(_bounds) - 1):
            _gid = int(_sorted[_bounds[_i]])
            if _gid > 0:
                _flat = _sort_idx[_bounds[_i]:_bounds[_i + 1]]
                _gc[_gid] = np.stack(
                    np.unravel_index(_flat, self.lgi_twinned.shape), axis=1)
        self._grain_coords = _gc
        _shp = self.lgi_twinned.shape
        print(f' done  {len(_gc)} grains  ({time.perf_counter()-_t_idx:.1f}s)')

        for gid in _host_list:
            self.twin_role[gid] = 'host'
            q_par = host_orientations.get(gid)
            if q_par is None:
                continue

            # Select {111} variants: Schmid-ranked (physical) or random
            n_attempt = min(self.n_lamellae_per_host, 4)
            if self.use_schmid_for_variant_selection and q_par is not None:
                from upxo.xtalphy.crystal_orientation import (
                    schmid_factors_fcc_twinning)
                _m, _ = schmid_factors_fcc_twinning(
                    q_par, self.schmid_loading_direction)
                # Primary variant = highest Schmid factor (most favourable
                # {111} plane); subsequent variants in decreasing Schmid order
                twin_111_variants = np.argsort(-_m)[:n_attempt].tolist()
            else:
                twin_111_variants = self._rng.choice(
                    4, size=n_attempt, replace=False).tolist()

            # Track twins count before this host (to detect success)
            n_before_host = len(self.primary_twin_quats)

            # First-lamella geometry (for contacting placement of k>=1)
            lamella_centroid_0  = None
            lamella_normal_0    = None
            lamella_halfwidth_0 = None

            for k, var_idx in enumerate(twin_111_variants):
                # VF guard
                if (total_vox > 0
                        and self.cumulative_twin_vox / total_vox >= tvf_stop):
                    self.vf_stopped_early = True
                    break

                # O(1) dict lookup instead of O(N_vox) np.argwhere scan
                coords = self._grain_coords.get(gid, np.empty((0, 3), dtype=np.intp))
                if coords.shape[0] < self.min_host_vox_for_lamella:
                    break

                # ── Determine plane origin ────────────────────────────────
                if k == 0:
                    origin = self._get_grain_origin(
                        gid, coords, self.lgi_twinned)

                else:
                    # k>=1: separated or contacting
                    first_carved = lamella_centroid_0 is not None
                    use_contacting = (
                        first_carved and
                        self._rng.random() >= self.prob_lamella_separated)

                    if use_contacting:
                        if (not _noted_contacting_override
                                and self.twin_nucleation_site != 'random_gb'):
                            print(
                                f'TwinGenerator3D NOTE: '
                                f'twin_nucleation_site="'
                                f'{self.twin_nucleation_site}" overridden '
                                f'for contacting lamellae (geometric offset).')
                            _noted_contacting_override = True

                        # Offset to outer face of lamella 0
                        candidate = (lamella_centroid_0
                                     + lamella_halfwidth_0 * lamella_normal_0)
                        # Snap guard: offset may overshoot for small grains
                        origin_int = np.clip(
                            np.round(candidate).astype(int),
                            0, np.array(self.lgi_twinned.shape) - 1)
                        if int(self.lgi_twinned[tuple(origin_int)]) == gid:
                            origin = candidate
                        else:
                            # Fallback to separated placement
                            origin = self._get_grain_origin(
                                gid, coords, self.lgi_twinned)
                    else:
                        origin = self._get_grain_origin(
                            gid, coords, self.lgi_twinned)

                # ── Carve lamella ─────────────────────────────────────────
                normal = compute_s3_habit_plane_3d(
                    q_par, self._rng, variant_idx=var_idx)

                hw = max(1.0,
                         float(self._rng.choice(thick_um)) / vs / 2.0)

                # ── Upper bounds on lamella size ──────────────────────────
                # 1. Absolute thickness cap
                if self.max_lamella_thickness_um is not None:
                    hw_abs_max = self.max_lamella_thickness_um / vs / 2.0
                    hw = min(hw, hw_abs_max)

                # 2. Volume-fraction cap relative to host grain size
                #    Approximation: twin slab width / grain eq-diameter
                #    hw_max = max_vf * grain_eqdia_vox / 2
                n_host_vox = coords.shape[0]
                if n_host_vox > 0:
                    grain_eqdia_vox = (6.0 / np.pi * n_host_vox) ** (1.0 / 3.0)
                    hw_vf_max = self.max_lamella_vf_per_host * grain_eqdia_vox / 2.0
                    hw = min(hw, hw_vf_max)

                # Enforce lower bound after upper-bound capping
                if hw < twin_minHalfWidth:
                    hw = twin_minHalfWidth
                    self.n_clamped += 1

                is_abrupt = self._rng.random() < abrupt_frac

                twin_voxels = introduce_twin_lamella_3d(
                    self.lgi_twinned, gid, origin, normal, hw,
                    next_gid, abrupt=is_abrupt, rng=self._rng,
                    host_coords=coords)   # skip internal argwhere scan

                if twin_voxels is not None and len(twin_voxels) > 0:
                    q_twin = self._apply_sigma3_variant(q_par, var_idx)
                    self.primary_twin_quats[next_gid]   = q_twin
                    self.all_quats[next_gid]             = q_twin
                    self.twin_role[next_gid]              = 'primary_twin'
                    self.twin_parent_of[next_gid]         = gid
                    self.twin_halfwidths_vox[next_gid]    = hw
                    self.cumulative_twin_vox             += len(twin_voxels)
                    # Update voxel index: remove carved voxels from host, add twin entry
                    _tv_flat  = np.ravel_multi_index(twin_voxels.T, _shp)
                    _hv_flat  = np.ravel_multi_index(
                        self._grain_coords[gid].T, _shp)
                    _hv_new   = np.setdiff1d(_hv_flat, _tv_flat, assume_unique=True)
                    self._grain_coords[gid] = (
                        np.stack(np.unravel_index(_hv_new, _shp), axis=1)
                        if _hv_new.size > 0
                        else np.empty((0, 3), dtype=np.intp))
                    self._grain_coords[next_gid] = twin_voxels
                    if is_abrupt:
                        self.n_abrupt_primary += 1
                    # Record first-lamella geometry for contact path
                    if k == 0:
                        lamella_centroid_0  = origin.copy()
                        lamella_normal_0    = normal.copy()
                        lamella_halfwidth_0 = hw
                    next_gid += 1

            # Count hosts that received at least one primary twin
            n_before = len(self.primary_twin_quats)
            if len(self.primary_twin_quats) > n_before_host:
                self.n_hosts_with_primary_twin += 1

            if self.vf_stopped_early:
                break

        n_p = len(self.primary_twin_quats)
        achieved = self.cumulative_twin_vox / total_vox if total_vox > 0 else 0.0
        print(f'TwinGenerator3D (primary): {n_p} twins across '
              f'{self.n_hosts_with_primary_twin}/{len(self.base.host_grain_ids)} hosts')
        if self.vf_stopped_early:
            print('  NOTE: VF target reached — introduction stopped early.')

    def introduce_secondary_twins(
            self,
            tvf: Dict,
            twin_thickness: Dict,
            tvf_2a: Optional[float] = None,
            tvf_2b: Optional[float] = None,
    ):
        """
        Introduce secondary Sigma3 twins nucleated from the primary twin
        surface.

        Nucleation mode (2a vs 2b) is chosen per event using
        ``prob_secondary_outward_twinNucleation``.

        2a (outward): carved from HOST grain at primary-host interface;
            ``twin_nucleation_site`` overridden.  Creates planar Sigma9
            boundary between secondary and host.

        2b (inward): carved inside the primary twin (nested);
            ``twin_nucleation_site`` applied to primary twin voxels.
            Creates Sigma3 with primary and Sigma9 with host if it spans
            the full primary width.
        """
        from upxo.xtalphy.crystal_orientation import compute_s3_habit_plane_3d
        from upxo.pxtalops.twin3d import introduce_twin_lamella_3d
        from scipy.ndimage import binary_dilation, generate_binary_structure

        sec_frac  = tvf.get('secondary_twin_frac', 0.0)
        prim_frac = tvf.get('primary_twin_frac', 1.0)

        if sec_frac <= 0 or not self.primary_twin_quats:
            print('TwinGenerator3D: no secondary twins '
                  '(sec_frac=0 or no primary twins).')
            return

        sec_ratio = sec_frac / max(prim_frac, 1e-9)
        n_attempts_secondary_events = max(
            1, round(len(self.primary_twin_quats) * sec_ratio))

        twin_vox_counts = {
            tgid: int(np.sum(self.lgi_twinned == tgid))
            for tgid in self.primary_twin_quats}
        sec_hosts = sorted(
            twin_vox_counts, key=twin_vox_counts.get,
            reverse=True)[:n_attempts_secondary_events]

        thick_um      = twin_thickness['thick_um'] * self.twin_thick_scale_factor
        vs            = self.base.voxel_size
        twin_minHalfWidth = self._min_hw()
        next_gid      = int(self.lgi_twinned.max()) + 1
        face_struct   = generate_binary_structure(3, 1)

        total_vox  = int(np.sum(self.base.lgi > 0))
        # Use explicit 2a/2b VF targets when provided (ebsd_partitioned mode)
        # otherwise fall back to 3D-scaled target (simple mode)
        # Cumulative VF target for secondary stage:
        #   Stage-1 already placed tvf_target_3d (= vf_int + vf_2b).
        #   2b secondaries relabel within primary — no new parent voxels, so
        #   cumulative_twin_vox is NOT incremented for them (see below).
        #   2a secondaries carve new host voxels → cumulative target rises by vf_2a.
        #   Overall cumulative ceiling = tvf_stage1 + vf_2a ≈ overall_twin_frac.
        if tvf_2a is not None:
            tvf_target = self.tvf_target_3d + float(tvf_2a)
        else:
            tvf_target = (self.tvf_target_3d if self.tvf_target_3d > 0
                          else (tvf.get('overall_twin_frac', 1.0) if tvf else 1.0))
        tvf_stop   = tvf_target * (1.0 + self.tvf_tolerance)

        _noted_2a_override = False
        n_outward = 0
        n_inward  = 0

        for tgid in sec_hosts:
            # Shared VF check (same counter as primary)
            if (total_vox > 0
                    and self.cumulative_twin_vox / total_vox >= tvf_stop):
                print('TwinGenerator3D: secondary stopped — VF target reached.')
                break

            q_primary_twin  = self.primary_twin_quats[tgid]
            parent_host_gid = self.twin_parent_of[tgid]

            use_outward = (self._rng.random()
                           < self.prob_secondary_outward_twinNucleation)

            var_idx = int(self._rng.integers(0, 4))
            normal  = compute_s3_habit_plane_3d(
                q_primary_twin, self._rng, variant_idx=var_idx)
            hw = max(1.0, float(self._rng.choice(thick_um)) / vs / 2.0)
            if self.max_lamella_thickness_um is not None:
                hw = min(hw, self.max_lamella_thickness_um / vs / 2.0)
            if hw < twin_minHalfWidth:
                hw = twin_minHalfWidth

            if use_outward:
                # ── 2a: outward into host ─────────────────────────────────
                if (not _noted_2a_override
                        and self.twin_nucleation_site != 'random_gb'):
                    print(
                        'TwinGenerator3D NOTE: twin_nucleation_site '
                        'overridden for outward secondary (interface '
                        'centroid used).')
                    _noted_2a_override = True

                primary_twin_mask = (self.lgi_twinned == tgid)
                host_mask         = (self.lgi_twinned == parent_host_gid)
                dilated           = binary_dilation(
                    primary_twin_mask, structure=face_struct)
                host_primTwin_interface_mask = dilated & host_mask

                if not np.any(host_primTwin_interface_mask):
                    continue

                iface_coords = np.argwhere(host_primTwin_interface_mask)
                centroid_sec_outward = iface_coords.mean(axis=0)
                # Snap guard for outward centroid
                host_coords = np.argwhere(host_mask)
                centroid_sec_outward = self._snap_to_grain(
                    centroid_sec_outward, parent_host_gid,
                    host_coords, self.lgi_twinned)

                twin_voxels = introduce_twin_lamella_3d(
                    self.lgi_twinned, parent_host_gid,
                    centroid_sec_outward, normal, hw,
                    next_gid, abrupt=False, rng=self._rng)

            else:
                # ── 2b: inward into primary (nested) ─────────────────────
                primary_coords = np.argwhere(self.lgi_twinned == tgid)
                if primary_coords.shape[0] < self.min_host_vox_for_lamella:
                    continue

                if self.twin_nucleation_site == 'random_gb':
                    # random_gb for 2b: pick primary voxel facing host grain
                    gb_pool_primary_twin = self._find_gb_voxels_facing_neighbor(
                        tgid, parent_host_gid,
                        primary_coords, self.lgi_twinned)
                    if len(gb_pool_primary_twin) > 0:
                        centroid_sec_inward = gb_pool_primary_twin[
                            int(self._rng.integers(
                                len(gb_pool_primary_twin)))].copy()
                    else:
                        # Fallback: primary twin centroid with snap guard
                        centroid_sec_inward = self._snap_to_grain(
                            primary_coords.mean(axis=0),
                            tgid, primary_coords, self.lgi_twinned)
                else:
                    centroid_sec_inward = self._get_grain_origin(
                        tgid, primary_coords, self.lgi_twinned)

                twin_voxels = introduce_twin_lamella_3d(
                    self.lgi_twinned, tgid,
                    centroid_sec_inward, normal, hw,
                    next_gid, abrupt=False, rng=self._rng,
                    host_coords=self._grain_coords.get(tgid))

            if twin_voxels is not None and len(twin_voxels) > 0:
                q_sec = self._apply_sigma3_variant(q_primary_twin, var_idx)
                self.secondary_twin_quats[next_gid] = q_sec
                self.all_quats[next_gid]             = q_sec
                self.twin_role[next_gid]              = 'secondary_twin'
                self.twin_parent_of[next_gid]         = tgid
                self.twin_halfwidths_vox[next_gid]    = hw   # actual hw used
                # Only 2a (outward) twins consume new parent voxels.
                # 2b (inward) twins relabel voxels already counted in Stage 1.
                if use_outward:
                    self.cumulative_twin_vox += len(twin_voxels)
                if use_outward:
                    n_outward += 1
                else:
                    n_inward += 1
                next_gid += 1

        n_sec = len(self.secondary_twin_quats)
        print(f'TwinGenerator3D: secondary twins: {n_sec} '
              f'(outward 2a: {n_outward}, inward 2b: {n_inward})')

    def compute_achieved_2d_tvf(
            self,
            n_slices_per_axis: int = 10,
            axes=('x', 'y', 'z'),
            return_raw: bool = False,
    ) -> Dict:
        """
        Cross-sectional (2D) measurement of the achieved twin area fraction
        in ``lgi_twinned``, for direct comparability with the EBSD 2D twin
        area fraction -- EBSD data is inherently 2D, but
        ``tvf_achieved_3d`` is measured over the whole 3D structure, so
        neither is directly comparable to what a real 2D EBSD
        cross-section of the synthetic structure would show.

        For each of ``n_slices_per_axis`` evenly-spaced 2D cross-sections
        along each axis in *axes*, computes twin pixel count / indexed
        pixel count (role in ``{'primary_twin', 'secondary_twin'}`` vs.
        any assigned role), then averages (mean and std) across every
        sampled slice -- matching
        ``TwinnedSimple3DBase.assess_hosting_representativeness_2d``'s
        convention of not filtering down to whichever slices happen to
        already be close to target. No cc3d re-labelling is needed here
        (unlike that method) since classification is by pixel value
        against ``twin_role``, not by reassociating disconnected 2D
        regions back to an original 3D grain ID.

        Must be called after :meth:`introduce_primary_twins` (reads
        ``lgi_twinned`` / ``twin_role``).

        Parameters
        ----------
        n_slices_per_axis : int
            Evenly-spaced 2D cross-sections sampled per axis.
        axes : iterable of str
            Subset of ``('x', 'y', 'z')`` (array axes 0/1/2 respectively --
            same convention as ``TwinnedSimple3DBase``).

        Returns
        -------
        dict with keys ``mean``, ``std``, ``n_slices_used``, and (only when
        ``return_raw=True``) ``ratios`` -- the raw per-slice twin area
        fraction values, for callers needing the full distribution rather
        than its mean/std (e.g. Summary Report's EBSD-vs-MC comparison).
        """
        from upxo.gsdataops.grid_ops import section_from_3d

        if self.lgi_twinned is None:
            raise RuntimeError('Call introduce_primary_twins() first.')

        # lgi_twinned inherits base.lgi's native (nz, ny, nx) axis order
        # (axis0=Z, axis2=X); array shape is left as-is, only the label
        # mapping is corrected so 'x'/'z' resolve to the physically
        # correct index -- see TwinnedSimple3DBase.plot_temporal_slice_3d.
        axis_map = {'x': 2, 'y': 1, 'z': 0}
        axes_int = [axis_map[a.lower()] for a in axes if a.lower() in axis_map]

        twin_gids = np.array(sorted(
            gid for gid, role in self.twin_role.items()
            if role in ('primary_twin', 'secondary_twin')), dtype=np.int64)

        ratios = []
        for ax in axes_int:
            domain_size = self.lgi_twinned.shape[ax]
            slice_positions = np.linspace(
                0, domain_size - 1, n_slices_per_axis, dtype=int)
            for slice_pos in slice_positions:
                lgi_2d = section_from_3d(
                    self.lgi_twinned, axis=ax, location=int(slice_pos))
                total_px = int(np.sum(lgi_2d > 0))
                if total_px == 0:
                    continue
                twin_px = int(np.sum(np.isin(lgi_2d, twin_gids))) if twin_gids.size else 0
                ratios.append(twin_px / total_px)

        result = {
            'mean': float(np.mean(ratios)) if ratios else 0.0,
            'std': float(np.std(ratios)) if len(ratios) > 1 else 0.0,
            'n_slices_used': len(ratios),
        }
        if return_raw:
            result['ratios'] = ratios
        return result

    def _achieved_tvf(self) -> float:
        """Current achieved twin VF in lgi_twinned."""
        if self.lgi_twinned is None:
            return 0.0
        all_twin = (list(self.primary_twin_quats)
                    + list(self.secondary_twin_quats))
        total = int(np.sum(self.lgi_twinned > 0))
        if total == 0 or not all_twin:
            return 0.0
        return sum(int(np.sum(self.lgi_twinned == g))
                   for g in all_twin) / total

    def twin_volume_fraction(self) -> float:
        """Return current twin VF in ``lgi_twinned``."""
        if self.lgi_twinned is None:
            return 0.0
        all_twin = (list(self.primary_twin_quats)
                    + list(self.secondary_twin_quats))
        total = int(np.sum(self.lgi_twinned > 0))
        if total == 0 or not all_twin:
            return 0.0
        return sum(int(np.sum(self.lgi_twinned == g))
                   for g in all_twin) / total

    def summary(self) -> Dict:
        """Return a rich summary dict of twin introduction results."""
        n_h   = len(self.base.host_grain_ids) if self.base.host_grain_ids else 0
        n_p   = len(self.primary_twin_quats)
        n_sec = len(self.secondary_twin_quats)
        achieved = self._achieved_tvf()
        return {
            # Identity
            'twin_csl_type':                         self.TWIN_TYPE_LABEL,
            # Configuration
            'n_lamellae_per_host':                   self.n_lamellae_per_host,
            'twin_nucleation_site':                  self.twin_nucleation_site,
            'meshing_route':                         self.meshing_route,
            'twin_orient_scatter_deg':               self.twin_orient_scatter_deg,
            'twin_thick_scale_factor':               self.twin_thick_scale_factor,
            'tvf_2d_to_3d_scale_factor':             self.tvf_2d_to_3d_scale_factor,
            'tvf_tolerance':                         self.tvf_tolerance,
            'prob_lamella_separated':                self.prob_lamella_separated,
            'prob_lamella_contacting':               self.prob_lamella_contacting,
            'prob_secondary_outward_twinNucleation': self.prob_secondary_outward_twinNucleation,
            'prob_secondary_inward_twinNucleation':  self.prob_secondary_inward_twinNucleation,
            # Volume fraction
            'tvf_2d_ebsd':                           self.tvf_2d_ebsd,
            'tvf_target_3d':                         self.tvf_target_3d,
            'tvf_achieved_3d':                       achieved,
            'tvf_achieved_pct_of_target':            (100 * achieved / self.tvf_target_3d
                                                      if self.tvf_target_3d > 0 else 0.0),
            'vf_stopped_early':                      self.vf_stopped_early,
            'vf_stop_status':                        (
                'Target reached -- introduction capped early' if self.vf_stopped_early
                else 'Target NOT reached -- ran through all hosts/lamellae'),
            # Primary twins
            'n_host_grains':                         n_h,
            'n_hosts_with_primary_twin':             self.n_hosts_with_primary_twin,
            'n_primary_twins':                       n_p,
            'n_abrupt_primary':                      self.n_abrupt_primary,
            'abrupt_frac_mc':                        (self.n_abrupt_primary / n_p
                                                      if n_p > 0 else 0.0),
            'n_clamped':                             self.n_clamped,
            'n_twin_halfwidths_recorded':            len(self.twin_halfwidths_vox),
            # Secondary twins
            'n_secondary_twins':                     n_sec,
            # Totals
            'n_all_twin_grains':                     n_p + n_sec,
            'twin_volume_fraction':                  achieved,
            'schmid_loading_direction':              self.schmid_loading_direction,
            'host_schmid_weight':                    self.host_schmid_weight,
            'use_schmid_for_variant_selection':      self.use_schmid_for_variant_selection,
        }

    def diagnose(self, tvf: Optional[Dict] = None) -> List[Dict]:
        """
        Return advisory notes explaining surprising values in :meth:`summary`,
        in a suggestive (not asserted) tone -- these are heuristic hypotheses
        about *why* a number looks the way it does, not a pass/fail verdict.
        They never gate progression to the next page.

        Parameters
        ----------
        tvf : dict or None
            The ``tvf`` dict passed to :meth:`introduce_primary_twins` (output
            of ``rg.compute_ebsd_tvf``).  When provided, enables the CSL-label
            mismatch check.  Not stored on the instance, so callers must pass
            it again here (kept out of ``__slots__`` deliberately -- it's
            EBSD-side provenance, not twin-generation state).

        Returns
        -------
        list of dict
            Each entry: ``{'anchor_fields': [...], 'message': str}``.
            ``anchor_fields`` names the :meth:`summary` keys the note
            explains (a caller can attach "(Ref N)" to those fields, N = this
            note's 1-based position in the returned list).
        """
        s = self.summary()
        notes: List[Dict] = []

        host_frac = getattr(self.base, 'actual_hosting_fraction', None)
        if host_frac is not None and host_frac > 0 and s['tvf_target_3d'] > 0:
            # A single lamella's half-width is capped at
            # max_lamella_vf_per_host * grain_eqdia_vox / 2 (see
            # introduce_primary_twins) -- a HALF-WIDTH bound, not a volume
            # bound. For a slab through the centre of a roughly spherical
            # grain, slab volume / grain volume scales as ~3*hw/diameter,
            # so the true per-host volume fraction such a lamella can reach
            # is closer to ~1.5x max_lamella_vf_per_host, not the nominal
            # value directly. This factor was calibrated against two real
            # runs (host_frac=0.2061 -> achieved 0.1625; host_frac=0.3192
            # -> achieved 0.2466), both landing within ~5% of a 1.5x
            # correction -- still an approximation (real hosts aren't
            # spherical, and multiple lamellae per host compound further),
            # not a hard bound, so this is phrased as a rough estimate
            # rather than an impossibility claim.
            _GEOM_CORRECTION = 1.5
            ceiling = host_frac * self.max_lamella_vf_per_host * _GEOM_CORRECTION
            if s['tvf_target_3d'] > ceiling:
                notes.append({
                    'anchor_fields': ['tvf_target_3d', 'tvf_achieved_3d', 'vf_stop_status'],
                    'message': (
                        f"The target volume fraction ({s['tvf_target_3d']:.4f}) "
                        f"exceeds a rough structural ceiling estimate of "
                        f"{ceiling:.4f} (host volume fraction {host_frac:.4f} "
                        f"x Max Lamella VF per Host "
                        f"{self.max_lamella_vf_per_host:.2f} x ~"
                        f"{_GEOM_CORRECTION:.1f} geometric factor) -- this is "
                        "an approximation, not a hard limit, but suggests "
                        "reaching target may require more/larger host "
                        "grains. Consider raising Target Hosting Fraction / "
                        "2D->3D Scale Factor on Twin-Host Allocation, and/or "
                        "Max Lamella VF per Host here."
                    ),
                })

        if (not self.vf_stopped_early and s['tvf_target_3d'] > 0
                and s['tvf_achieved_pct_of_target'] < self.DIAG_SHORTFALL_PCT_THRESHOLD):
            notes.append({
                'anchor_fields': ['tvf_achieved_3d', 'vf_stop_status'],
                'message': (
                    f"Achieved volume fraction reached only "
                    f"{s['tvf_achieved_pct_of_target']:.1f}% of target without "
                    "the volume-fraction cap ever triggering -- this suggests "
                    "the twin-hosting capacity of the currently selected host "
                    "grains, and/or the Max Lamella VF per Host cap, may be "
                    "the limiting factor rather than the VF Tolerance setting "
                    "itself. Consider revisiting the Target Hosting Fraction "
                    "on Twin-Host Allocation, or raising Max Lamella VF per "
                    "Host / Max Lamellae per Host here."
                ),
            })

        if tvf is not None:
            csl = tvf.get('csl_label', '')
            if csl and csl != _SUPPORTED_CSL:
                notes.append({
                    'anchor_fields': ['twin_csl_type', 'tvf_2d_ebsd'],
                    'message': (
                        f'The CSL label selected upstream ("{csl}") does not '
                        f'match the twin type actually generated here '
                        f'({self.TWIN_TYPE_LABEL}) -- Path A of this pipeline '
                        'only supports Sigma3 twins, so the EBSD 2D twin area '
                        'fraction above may not correspond one-to-one to '
                        'genuine Sigma3 twins. Treat it as approximate until '
                        'CSL Registry Path B is available.'
                    ),
                })

        if s['n_primary_twins'] > 0 and s['abrupt_frac_mc'] >= self.DIAG_ABRUPT_FRAC_THRESHOLD:
            notes.append({
                'anchor_fields': ['n_abrupt_primary'],
                'message': (
                    f"{s['abrupt_frac_mc']:.0%} of primary twins were "
                    "truncated abruptly inside their host grain rather than "
                    "fully spanning it -- this reduces effective twinned "
                    "volume relative to full-span lamellae of the same "
                    "thickness, and may be contributing to any shortfall "
                    "between target and achieved volume fraction."
                ),
            })

        if s['n_primary_twins'] > 0:
            clamped_frac = s['n_clamped'] / s['n_primary_twins']
            if clamped_frac >= self.DIAG_CLAMPED_FRAC_THRESHOLD:
                notes.append({
                    'anchor_fields': ['n_clamped'],
                    'message': (
                        f"{clamped_frac:.0%} of lamellae had their thickness "
                        "raised to the configured minimum rather than using "
                        "the EBSD-sampled value -- the EBSD thickness "
                        "distribution may be finer than the current voxel "
                        "size supports at this domain's resolution."
                    ),
                })

        if s['n_host_grains'] > 0:
            coverage = s['n_hosts_with_primary_twin'] / s['n_host_grains']
            if (coverage >= self.DIAG_HOST_SATURATION_THRESHOLD
                    and s['tvf_achieved_pct_of_target'] < 100.0):
                notes.append({
                    'anchor_fields': ['n_hosts_with_primary_twin'],
                    'message': (
                        f"{coverage:.0%} of available host grains already "
                        "carry at least one twin, yet the target volume "
                        "fraction was not reached -- reaching target may "
                        "require more or larger host grains (see Twin-Host "
                        "Allocation) rather than a twin-generation parameter "
                        "change."
                    ),
                })

        return notes

    def print_summary(self) -> None:
        """Print a formatted, detailed summary of twin introduction results."""
        s = self.summary()
        sep = '=' * 60
        print(sep)
        print('TwinGenerator3D  —  Introduction Summary')
        print(sep)
        print('\nCONFIGURATION')
        print(f'  n_lamellae_per_host              : {s["n_lamellae_per_host"]}')
        print(f'  twin_nucleation_site             : {s["twin_nucleation_site"]}')
        print(f'  meshing_route                    : {s["meshing_route"]}')
        print(f'  twin_orient_scatter_deg          : {s["twin_orient_scatter_deg"]}')
        print(f'  twin_thick_scale_factor          : {s["twin_thick_scale_factor"]}')
        print(f'  tvf_2d_to_3d_scale_factor        : {s["tvf_2d_to_3d_scale_factor"]}')
        print(f'  schmid_loading_direction         : {s["schmid_loading_direction"]}')
        print(f'  host_schmid_weight               : {s["host_schmid_weight"]}'
              f'  ({"Schmid-ranked hosts" if s["host_schmid_weight"]>0 else "volume+adj only"})')
        print(f'  use_schmid_for_variant_selection : {s["use_schmid_for_variant_selection"]}')
        print(f'  prob_lamella_separated           : {s["prob_lamella_separated"]}')
        print(f'  prob_lamella_contacting          : {s["prob_lamella_contacting"]}')
        print(f'  prob_secondary_outward           : {s["prob_secondary_outward_twinNucleation"]}')
        print(f'  prob_secondary_inward            : {s["prob_secondary_inward_twinNucleation"]}')
        print('\nCONSTRAINTS')
        tum = self.max_lamella_thickness_um
        print(f'  max_lamella_thickness_um         : '
              f'{tum:.2f} um' if tum is not None else
              f'  max_lamella_thickness_um         : None  (no absolute cap)')
        print(f'  max_lamella_vf_per_host          : {self.max_lamella_vf_per_host:.2f}')
        mc = self.meshing_route
        mt = (self.min_lamella_thickness_conformal if mc == 'conformal'
              else self.min_lamella_thickness_non_conformal)
        print(f'  min_lamella_thickness            : {mt} vox  ({mc})')
        print('\nHOST ALLOCATION')
        _sf = self.base.host_fraction_2d_to_3d_scale_factor
        _eff = min(getattr(self.base, 'target_hosting_fraction', 0) * _sf, 1.0)
        print(f'  host_fraction_2d_to_3d_scale     : {_sf:.2f}x  '
              f'(effective target {_eff:.4f})')
        _vw = self.base.host_ranking_volume_weight
        print(f'  host_ranking_volume_weight       : {_vw:.2f}  '
              f'(adj={1-_vw:.2f})')
        print(f'  hosts selected / total eligible  : '
              f'{getattr(self.base, "n_grains", "?")} total grains')
        print('\nVOLUME FRACTION')
        print(f'  EBSD 2D area fraction            : {s["tvf_2d_ebsd"]:.4f}')
        sf = s["tvf_2d_to_3d_scale_factor"]
        print(f'  3D target  (2D x {sf:.2f})          : {s["tvf_target_3d"]:.4f}')
        print(f'  Achieved 3D twin VF              : {s["tvf_achieved_3d"]:.4f}'
              f'   ({s["tvf_achieved_pct_of_target"]:.1f}% of target)')
        print(f'  VF tolerance                     : +/-{100*s["tvf_tolerance"]:.0f}%')
        print(f'  Early stop triggered             : {s["vf_stopped_early"]}')
        print('\nPRIMARY TWINS')
        n_h = s["n_host_grains"]
        n_hw = s["n_hosts_with_primary_twin"]
        n_p  = s["n_primary_twins"]
        print(f'  Host grains                      : {n_h}')
        print(f'  Hosts with >= 1 twin             : {n_hw}'
              f'   ({100*n_hw/n_h:.1f}%)'  if n_h > 0 else '')
        print(f'  Primary twin lamellae            : {n_p}')
        n_ab = s["n_abrupt_primary"]
        print(f'  Abrupt twins                     : {n_ab}'
              f'   ({100*s["abrupt_frac_mc"]:.1f}%)'  if n_p > 0 else
              f'  Abrupt twins                     : 0')
        print(f'  Half-widths clamped              : {s["n_clamped"]}'
              f'   ({s["meshing_route"]} min thickness)')
        print(f'  Half-widths recorded             : {s["n_twin_halfwidths_recorded"]}')
        print('\nSECONDARY TWINS')
        print(f'  Secondary twin grains            : {s["n_secondary_twins"]}')
        print('\nTOTAL')
        print(f'  All twin grains                  : {s["n_all_twin_grains"]}'
              f'   (primary {n_p}  +  secondary {s["n_secondary_twins"]})')
        print(f'  Achieved twin VF                 : {s["tvf_achieved_3d"]:.4f}')
        print(sep)
