"""
base_3d.py
==========
Foundation class for the twinned simple 3D grain structure pipeline.
"""

import numpy as np
from typing import Optional, Dict, Union, Tuple


class TwinnedSimple3DBase:
    """
    Foundation 3D grain structure for the twinned simple FCC pipeline.

    Wraps a 3D labelled grain image (lgi) from an SGC simulation,
    attaches physical dimensions, characterises grain morphology, and
    allocates twin host grains driven by the EBSD-measured twin hosting
    fraction.  Entry point to the full pipeline.
    """

    __slots__ = (
        'lgi', 'voxel_size', 'units',
        'n_grains', 'grain_ids',
        'mprop',
        'host_grain_ids', 'non_host_grain_ids', 'eligible_grain_ids',
        'target_hosting_fraction', 'actual_hosting_fraction',
        'host_fraction_2d_to_3d_scale_factor',
        'host_ranking_volume_weight',
        'n_pool_a', 'n_pool_b', 'allocation_summary',
        '_tslice_key', '_rng',
    )

    def __init__(
            self,
            lgi: np.ndarray,
            voxel_size: float,
            units: str = 'microns',
            rng_seed: Optional[int] = None,
    ):
        self.lgi = lgi.copy()
        self.voxel_size = float(voxel_size)
        self.units = units
        self.n_grains = int(np.unique(lgi[lgi > 0]).size)
        self.grain_ids = sorted(int(g) for g in np.unique(lgi) if g > 0)
        self.mprop: Dict = {}
        self.host_grain_ids: Optional[set] = None
        self.non_host_grain_ids: Optional[set] = None
        self.eligible_grain_ids: Optional[set] = None
        self.target_hosting_fraction: Optional[float] = None
        self.actual_hosting_fraction: Optional[float] = None
        self.host_fraction_2d_to_3d_scale_factor: float = 1.0
        self.host_ranking_volume_weight: float = 0.5
        self.n_pool_a: Optional[int] = None
        self.n_pool_b: Optional[int] = None
        self.allocation_summary: Optional[Dict] = None
        self._tslice_key: Optional[int] = None
        self._rng = np.random.default_rng(rng_seed)

    @classmethod
    def from_mcgs(
            cls,
            pxt,
            tslice_key: int,
            voxel_size: Optional[float] = None,
            units: str = 'microns',
            rng_seed: Optional[int] = None,
    ):
        """
        Construct from an ``mcgs`` simulation object at a chosen time slice.

        Parameters
        ----------
        pxt : mcgs
            UPXO Synthetic Grain Structure (SGC) generator object (after
            ``pxt.simulate()``).
        tslice_key : int
            Key into ``pxt.gs`` (i.e. one of the values in ``pxt.m``).
        voxel_size : float or None
            Physical voxel edge length.  Reads ``pxt.vox_size`` if None.
        units : str
            Physical units, default 'microns'.
        rng_seed : int or None
        """
        gstslice = pxt.gs[tslice_key]
        # lgi is only populated after char_morphology_of_grains() runs
        if not hasattr(gstslice, 'lgi') or gstslice.lgi is None:
            gstslice.char_morphology_of_grains(
                label_str_order=1,
                find_grain_voxel_locs=True,
                find_spatial_bounds_of_grains=True,
                force_compute=True,
            )
        lgi = gstslice.lgi
        if voxel_size is not None:
            vs = float(voxel_size)
        else:
            # pxt.vox_size is always a (x, y, z) tuple in mcgsV1_1 (and
            # after MC Qualification's scale-factor calibration overwrites
            # it) -- never a bare scalar -- so float(pxt.vox_size) would
            # raise. This pipeline only ever produces isotropic voxels, so
            # the first component is the physical edge length.
            raw = pxt.vox_size
            vs = float(raw[0]) if isinstance(raw, (tuple, list, np.ndarray)) else float(raw)
        obj = cls(lgi, vs, units, rng_seed)
        obj._tslice_key = tslice_key
        return obj

    def char_morphology(
            self,
            volnv: bool = True,
            eqdia: bool = True,
            sanv: bool = False,
            force_compute: bool = True,
    ):
        """
        Compute per-grain morphological properties and populate ``self.mprop``.

        Properties are stored as ``{property_key: {grain_id: value}}``.

        Parameters
        ----------
        volnv : bool
            Compute voxel count (volume in voxels).
        eqdia : bool
            Compute equivalent spherical diameter (in ``self.units``).
        sanv : bool
            Compute surface area in voxels (expensive; deferred by default).
        force_compute : bool
            Recompute even if already populated.
        """
        if not force_compute and self.mprop:
            return

        gids = self.grain_ids

        if volnv:
            self.mprop['volnv'] = {gid: int(np.sum(self.lgi == gid)) for gid in gids}

        if eqdia:
            vols = self.mprop.get('volnv') or {gid: int(np.sum(self.lgi == gid)) for gid in gids}
            vs3 = self.voxel_size ** 3
            pi = np.pi
            self.mprop['eqdia'] = {
                gid: float((6.0 / pi * vox * vs3) ** (1.0 / 3.0))
                for gid, vox in vols.items()
            }

        if sanv:
            # Surface area: count exposed faces between different grain IDs.
            # Deferred — expensive for large structures.
            from scipy.ndimage import convolve as _conv
            face_kernel = np.array([
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ], dtype=np.int32)
            sanv_d = {}
            for gid in gids:
                mask = (self.lgi == gid)
                face_count = _conv(
                    mask.astype(np.int32), face_kernel, mode='constant', cval=0)
                # Each face exposed to a different grain contributes 1
                exposed = int(np.sum(mask) * 6) - int(np.sum(face_count[mask]))
                sanv_d[gid] = exposed
            self.mprop['sanv'] = sanv_d

    def compute_aspect_ratio_bbox(self, force_compute: bool = True):
        """
        Per-grain 3D aspect ratio from each grain's axis-aligned bounding
        box: ``max(extent) / min(extent)`` across the three voxel-count
        axes.  Populates ``self.mprop['aspect_ratio']``
        (``{grain_id: float}``).

        Bounding-box extents scale identically on every axis for an
        isotropic (cubic-voxel) structure, so the ratio itself is
        dimensionless and doesn't need ``voxel_size``.

        This class has no other 3D shape-descriptor beyond volnv/eqdia/
        sanv -- the only other working 3D aspect-ratio logic anywhere in
        the codebase lives on the unrelated ``mcgs3_grain_structure``
        class (``set_mprop_arbbox``); this reimplements the same
        bounding-box idea directly against ``self.lgi``, using
        ``scipy.ndimage.find_objects`` for a single vectorised pass over
        every grain ID at once rather than one ``np.where`` per grain.

        Parameters
        ----------
        force_compute : bool
            Recompute even if already populated.
        """
        if not force_compute and 'aspect_ratio' in self.mprop:
            return

        from scipy import ndimage
        objects = ndimage.find_objects(self.lgi)
        ar = {}
        for gid in self.grain_ids:
            sl = objects[gid - 1] if 0 <= gid - 1 < len(objects) else None
            if sl is None:
                continue
            extents = [s.stop - s.start for s in sl]
            ar[gid] = float(max(extents) / min(extents))
        self.mprop['aspect_ratio'] = ar

    def compute_2d_slice_properties_by_axis(
            self,
            selected_props,
            comparison_axes: Optional[list] = None,
            n_slices_per_axis: Union[int, dict, None] = None,
            connectivity_2d: int = 4,
    ) -> dict:
        """
        Same evenly-spaced-cross-section sampling as
        :meth:`compute_2d_slice_properties`, but returns per-axis,
        per-individual-slice breakdowns instead of one pooled array per
        property -- needed by callers that need the individual slice
        populations, not just their union (e.g. a min/max KDE envelope
        band showing slice-to-slice spread, alongside the combined/
        pooled distribution).

        :meth:`compute_2d_slice_properties` is a thin wrapper around this
        method that concatenates everything back into flat pooled
        arrays, so both share exactly the same underlying sampling/
        measurement code -- see that method's docstring for the shared
        parameter semantics.

        Returns
        -------
        dict {axis: {prop_name: [ndarray, ndarray, ...]}}
            One ndarray per sampled slice position along that axis (in
            slice-position order), each holding that slice's per-region
            values for that property (may be an empty array if no
            regions were found in that slice). Axes not present in
            comparison_axes are absent from the returned dict.
        """
        import cc3d
        from skimage.measure import regionprops
        from upxo.gsdataops.grid_ops import section_from_3d

        selected_props = list(selected_props)
        if comparison_axes is None:
            comparison_axes = ['x', 'y', 'z']
        if n_slices_per_axis is None:
            n_slices_per_axis = 5

        axis_map = {'x': 2, 'y': 1, 'z': 0}
        vs = self.voxel_size
        by_axis: dict = {}

        for axis in comparison_axes:
            axis = axis.lower()
            if axis not in axis_map:
                continue
            ax = axis_map[axis]
            n_this_axis = (n_slices_per_axis[axis]
                           if isinstance(n_slices_per_axis, dict) else n_slices_per_axis)
            domain_size = self.lgi.shape[ax]
            slice_positions = np.linspace(0, domain_size - 1, n_this_axis, dtype=int)

            per_prop_slices = {p: [] for p in selected_props}
            for slice_pos in slice_positions:
                lgi_2d = section_from_3d(self.lgi, axis=ax, location=int(slice_pos))
                relabelled = cc3d.connected_components(lgi_2d, connectivity=connectivity_2d)
                slice_vals = {p: [] for p in selected_props}
                for region in regionprops(relabelled):
                    if 'area' in selected_props:
                        slice_vals['area'].append(region.area * vs ** 2)
                    if 'eqdia' in selected_props:
                        slice_vals['eqdia'].append(region.equivalent_diameter_area * vs)
                    if 'perimeter' in selected_props:
                        slice_vals['perimeter'].append(region.perimeter * vs)
                    if 'solidity' in selected_props:
                        slice_vals['solidity'].append(region.solidity)
                    if 'aspect_ratio' in selected_props:
                        maj, mn = region.major_axis_length, region.minor_axis_length
                        if mn > 0:
                            slice_vals['aspect_ratio'].append(maj / mn)
                for p in selected_props:
                    per_prop_slices[p].append(np.asarray(slice_vals[p], dtype=float))
            by_axis[axis] = per_prop_slices

        return by_axis

    def compute_2d_slice_properties(
            self,
            selected_props,
            comparison_axes: Optional[list] = None,
            n_slices_per_axis: Union[int, dict, None] = None,
            connectivity_2d: int = 4,
    ) -> dict:
        """
        Pool 2D grain-morphology properties from evenly-spaced
        cross-sections of ``self.lgi``, independent of any twin-host
        allocation (contrast with
        :meth:`assess_hosting_representativeness_2d`, which reads
        ``self.host_grain_ids``/``self.eligible_grain_ids`` and requires
        ``allocate_twin_hosts``/``allocate_twin_hosts_spatial`` to have
        already run -- this method needs neither).

        Reuses the same evenly-spaced-cross-section / ``cc3d``
        relabelling skeleton as
        :meth:`assess_hosting_representativeness_2d`, but pools
        ``skimage.measure.regionprops``-derived scalar properties per 2D
        region directly instead of tallying host/non-host ratios. A thin
        wrapper around :meth:`compute_2d_slice_properties_by_axis` that
        flattens its per-axis/per-slice breakdown back into one pooled
        array per property.

        Parameters
        ----------
        selected_props : iterable of str
            Subset of ``{'area', 'eqdia', 'perimeter', 'aspect_ratio',
            'solidity'}``.
        comparison_axes : list of str or None
            Subset of ``['x', 'y', 'z']`` (array axes 2/1/0
            respectively -- same convention as
            ``assess_hosting_representativeness_2d``). Defaults to all
            three.
        n_slices_per_axis : int or dict or None
            Evenly-spaced 2D cross-sections sampled per axis -- same
            int-or-``{'x': int, 'y': int, 'z': int}`` convention as
            ``assess_hosting_representativeness_2d``'s
            ``n_comparison_slices``. Defaults to 5 per axis.
        connectivity_2d : int
            ``cc3d`` connectivity for the 2D re-labelling (4 or 8).

        Returns
        -------
        dict {prop_name: 1-D ndarray}
            Every sampled region's value for that property, pooled
            across every sampled slice on every requested axis.
        """
        selected_props = list(selected_props)
        by_axis = self.compute_2d_slice_properties_by_axis(
            selected_props, comparison_axes, n_slices_per_axis, connectivity_2d)

        pooled = {p: [] for p in selected_props}
        for per_prop in by_axis.values():
            for p in selected_props:
                pooled[p].extend(per_prop[p])

        return {p: (np.concatenate(v) if v else np.asarray([], dtype=float))
                for p, v in pooled.items()}

    def allocate_twin_hosts(
            self,
            target_hosting_fraction: float,
            min_host_voxels: int = 4,
            host_fraction_2d_to_3d_scale_factor: float = 1.0,
            host_ranking_volume_weight: float = 0.5,
    ):
        """
        Designate twin host grains and populate ``self.host_grain_ids``.

        Parameters
        ----------
        target_hosting_fraction : float
            EBSD 2D hosting fraction from
            ``rg.merge_info['twin_hosting_fraction']``.
        min_host_voxels : int
            Grains smaller than this are ineligible to host (but still
            receive the 'non_host' role for visualisation).
        host_fraction_2d_to_3d_scale_factor : float
            Stereological correction: the EBSD hosting fraction is a 2D
            apparent value; set > 1.0 to designate more hosts than the
            2D fraction suggests.  Effective target is capped at 1.0.
        host_ranking_volume_weight : float (0.0 -- 1.0)
            Weight given to grain volume in the composite host-selection
            rank.  1.0 = pure volume sort (largest first).  0.5 = equal
            weight between volume and number of face-adjacent neighbours
            (higher coordination = more grain-boundary area = more twin
            nucleation sites).
        """
        if 'volnv' not in self.mprop:
            self.char_morphology(volnv=True, eqdia=False)

        vols     = self.mprop['volnv']
        eligible = [(gid, v) for gid, v in vols.items() if v >= min_host_voxels]
        n        = len(eligible)

        if n == 0:
            self.host_grain_ids    = set()
            self.non_host_grain_ids = set(vols.keys())
            self.target_hosting_fraction  = target_hosting_fraction
            self.actual_hosting_fraction  = 0.0
            return self.host_grain_ids, self.non_host_grain_ids

        # ── Scale 2D EBSD fraction to 3D target ──────────────────────────
        effective_target = min(
            target_hosting_fraction * host_fraction_2d_to_3d_scale_factor, 1.0)
        n_target     = int(np.round(effective_target * n))
        total_volume = sum(v for _, v in eligible)

        # ── Rank eligible grains ──────────────────────────────────────────
        alpha = float(np.clip(host_ranking_volume_weight, 0.0, 1.0))

        if abs(alpha - 1.0) < 1e-9:
            # Pure volume sort — fast path, no neighbour computation needed
            eligible_sorted = sorted(eligible, key=lambda x: x[1], reverse=True)
            rank_desc = 'volume only'
        else:
            # Composite volume + face-neighbour-count rank
            import cc3d
            from collections import defaultdict
            edges  = cc3d.region_graph(self.lgi.astype(np.int32), connectivity=6)
            n_neigh: dict = defaultdict(int)
            for edge in edges:
                a, b = int(edge[0]), int(edge[1])
                if a > 0: n_neigh[a] += 1
                if b > 0: n_neigh[b] += 1

            gids_e  = [gid for gid, _ in eligible]
            vols_e  = [v   for _, v   in eligible]
            neigh_e = [n_neigh.get(gid, 0) for gid in gids_e]

            sv = sorted(range(n), key=lambda i: vols_e[i],  reverse=True)
            sn = sorted(range(n), key=lambda i: neigh_e[i], reverse=True)
            rank_v = [0] * n
            rank_n = [0] * n
            for pos, i in enumerate(sv): rank_v[i] = pos
            for pos, i in enumerate(sn): rank_n[i] = pos

            composite = [alpha * rank_v[i] + (1 - alpha) * rank_n[i]
                         for i in range(n)]
            eligible_sorted = [eligible[i]
                                for i in sorted(range(n), key=lambda i: composite[i])]
            rank_desc = (f'composite  vol={alpha:.2f}  '
                         f'adj={1-alpha:.2f}')

        # ── Greedy selection ──────────────────────────────────────────────
        host_ids: set = set()
        host_volume   = 0
        for gid, vol in eligible_sorted:
            host_ids.add(gid)
            host_volume += vol
            if (len(host_ids) >= n_target
                    or host_volume / total_volume >= effective_target):
                break

        ineligible_ids = set(vols.keys()) - {gid for gid, _ in eligible}
        # Store host allocation parameters for downstream access in print_summary
        self.host_fraction_2d_to_3d_scale_factor = host_fraction_2d_to_3d_scale_factor
        self.host_ranking_volume_weight           = host_ranking_volume_weight
        self.host_grain_ids    = host_ids
        self.non_host_grain_ids = (
            set(gid for gid, _ in eligible_sorted if gid not in host_ids)
            | ineligible_ids
        )
        self.target_hosting_fraction = target_hosting_fraction
        self.actual_hosting_fraction = (
            host_volume / total_volume if total_volume > 0 else 0.0)

        print('TwinnedSimple3DBase.allocate_twin_hosts:')
        print(f'  EBSD 2D hosting fraction   : {target_hosting_fraction:.4f}')
        if abs(host_fraction_2d_to_3d_scale_factor - 1.0) > 1e-9:
            print(f'  3D effective target        : {effective_target:.4f}'
                  f'  (x{host_fraction_2d_to_3d_scale_factor:.2f})')
        print(f'  Achieved hosting fraction  : {self.actual_hosting_fraction:.4f}')
        print(f'  Host grains               : {len(self.host_grain_ids)}')
        print(f'  Non-host grains           : {len(self.non_host_grain_ids)}')
        print(f'  Ranking                   : {rank_desc}')

        return self.host_grain_ids, self.non_host_grain_ids

    def allocate_twin_hosts_spatial(
            self,
            target_hosting_fraction: float,
            min_host_voxels: int = 4,
            host_fraction_2d_to_3d_scale_factor: float = 1.0,
            host_ranking_volume_weight: float = 0.5,
            mis_fraction: float = 0.80,
            pool_b_size_measure: str = 'vol_vox',
            pool_b_band_method: str = 'std',
            pool_b_band_shape: str = 'between',
            pool_b_n_std: float = 1.0,
            pool_b_percentile_lo: float = 25.0,
            pool_b_percentile_hi: float = 75.0,
            neighbour_frac: float = 100.0,
            mis_runs: int = 20,
            seed: int = 0,
    ):
        """
        Spatial-dispersal-constrained twin host selection.

        Selects ``n_target`` host grains in two pools:

        **Pool A — 80% (mis_fraction) from a maximal independent set (MIS):**
        Grains in a MIS are non-adjacent, guaranteeing that no two selected
        hosts share a face.  This prevents twins from different hosts from
        clustering and touching each other, which would create spurious
        twin-twin (Sigma9) boundaries and dilute the Sigma3 MDF peak.
        ``maximal_independent_set`` (networkx) is called ``mis_runs`` times
        with different seeds; the largest result is used.

        **Pool B — 20% (1 - mis_fraction) from a size band:**
        The remaining fraction is drawn from non-MIS grains whose size
        falls in a selected band of the eligible-grain size distribution
        AND that have at least one neighbour also in that band.  This
        allows physically realistic, limited host-host adjacency between
        similarly-positioned-in-size grains, confined to whichever part of
        the size distribution ``pool_b_band_shape`` selects (by default,
        the middle -- avoiding abnormally large or small grain pairs --
        but ``'below'``/``'above'`` deliberately target the small/large
        tails instead, e.g. to stop large grains being systematically
        under-hosted).

        Parameters
        ----------
        target_hosting_fraction : float
            EBSD 2D hosting fraction.
        min_host_voxels : int
            Minimum voxel count for eligibility.
        host_fraction_2d_to_3d_scale_factor : float
            Multiplier on the 2D fraction for 3D target.
        host_ranking_volume_weight : float
            0 = rank by adjacency count, 1 = rank by volume, 0.5 = equal.
        mis_fraction : float
            Fraction of n_target drawn from the MIS (default 0.80).
        pool_b_size_measure : str
            Size metric used to define the Pool B band.
            Options:
              ``'vol_vox'``  – grain volume in voxels (integer count).
              ``'vol_um3'``  – grain volume in μm³ (vol_vox × voxel_size³).
              ``'eqdia_um'`` – sphere-equivalent diameter in μm
                               = (6 × vol_um3 / π)^(1/3).
        pool_b_band_method : str
            How the band's cutoff(s) are computed from the chosen size
            measure's distribution over eligible grains:
              ``'std'``        – cutoffs are ``mean ± pool_b_n_std × std``.
              ``'percentile'`` – cutoffs are ``pool_b_percentile_lo``/
                                 ``pool_b_percentile_hi`` percentiles of the
                                 distribution.
            Default ``'std'`` (unchanged prior behaviour).
        pool_b_band_shape : str
            Which side(s) of the distribution the band covers:
              ``'below'``   – ``(-inf, hi]`` (near-*smaller* grains only).
              ``'between'`` – ``[lo, hi]`` (near-mean; default, unchanged
                              prior behaviour).
              ``'above'``   – ``[lo, +inf)`` (near-*larger* grains only,
                              uncapped).
            ``lo``/``hi`` are whichever of the method's two cutoffs apply
            to the selected shape -- e.g. ``'below'`` only ever uses the
            upper cutoff, ``'above'`` only the lower one.
        pool_b_n_std : float
            Used when ``pool_b_band_method == 'std'``. Half-band width (for
            ``'between'``) or single-sided offset (for ``'below'``/
            ``'above'``) in units of the standard deviation of the chosen
            size measure. Default 1.0.
        pool_b_percentile_lo, pool_b_percentile_hi : float
            Used when ``pool_b_band_method == 'percentile'`` (0-100).
            ``pool_b_percentile_lo`` is the band's lower cutoff (used by
            ``'between'`` and ``'above'``); ``pool_b_percentile_hi`` is the
            upper cutoff (used by ``'between'`` and ``'below'``). Defaults
            25.0 / 75.0.
        neighbour_frac : float
            Percentage (0-100) of Pool B drawn from in-band grains that DO
            have an in-band neighbour, vs. in-band grains that do NOT. 100
            (default) reproduces the original, neighbour-only behaviour
            exactly. 0 draws entirely from non-neighbouring in-band grains.
            Any other value mixes the two proportionally. Whichever
            sub-pool is short of its share falls back to drawing the
            remainder from the other sub-pool (so the total Pool B count is
            unaffected by this split, only its do/do-not composition is).
        mis_runs : int
            Number of random MIS calls; the largest result is kept.
        seed : int
            Base seed for the MIS random runs.
        """
        if pool_b_band_method not in ('std', 'percentile'):
            raise ValueError(
                f'pool_b_band_method must be "std" or "percentile"; '
                f'got "{pool_b_band_method}"')
        if pool_b_band_shape not in ('below', 'between', 'above'):
            raise ValueError(
                f'pool_b_band_shape must be "below", "between", or "above"; '
                f'got "{pool_b_band_shape}"')
        if (pool_b_band_method == 'percentile' and pool_b_band_shape == 'between'
                and pool_b_percentile_lo >= pool_b_percentile_hi):
            raise ValueError(
                f'pool_b_percentile_lo ({pool_b_percentile_lo}) must be < '
                f'pool_b_percentile_hi ({pool_b_percentile_hi}) for '
                f'pool_b_band_shape="between"')

        import cc3d
        from collections import defaultdict
        from networkx.algorithms.mis import maximal_independent_set
        from upxo.netops.kmake import make_gid_net_from_neighlist

        if 'volnv' not in self.mprop:
            self.char_morphology(volnv=True, eqdia=False)

        vols     = self.mprop['volnv']
        eligible = [(gid, v) for gid, v in vols.items() if v >= min_host_voxels]
        n        = len(eligible)

        if n == 0:
            self.host_grain_ids     = set()
            self.non_host_grain_ids = set(vols.keys())
            self.eligible_grain_ids = set()
            self.target_hosting_fraction = target_hosting_fraction
            self.actual_hosting_fraction = 0.0
            self.n_pool_a = 0
            self.n_pool_b = 0
            self.allocation_summary = {
                'target_hosting_fraction': target_hosting_fraction,
                'effective_target_3d': None,
                'n_target': 0,
                'n_eligible': 0,
                'mis_best_size': 0,
                'mis_runs': mis_runs,
                'mis_seed_base': seed,
                'pool_b_size_measure': pool_b_size_measure,
                'pool_b_size_measure_label': None,
                'pool_b_band_method': pool_b_band_method,
                'pool_b_band_shape': pool_b_band_shape,
                'pool_b_tolerance_std': pool_b_n_std,
                'pool_b_percentile_lo': pool_b_percentile_lo,
                'pool_b_percentile_hi': pool_b_percentile_hi,
                'pool_b_mean_size': None,
                'pool_b_std_size': None,
                'pool_b_band_lo': None,
                'pool_b_band_hi': None,
                'n_pool_b_candidates': 0,
                'pool_b_neighbour_frac': neighbour_frac,
                'n_pool_b_do_candidates': 0,
                'n_pool_b_donot_candidates': 0,
                'n_pool_b_from_do': 0,
                'n_pool_b_from_donot': 0,
                'n_pool_a': 0,
                'n_pool_b': 0,
                'n_host': 0,
                'n_non_host': len(self.non_host_grain_ids),
                'actual_hosting_fraction': 0.0,
            }
            return self.host_grain_ids, self.non_host_grain_ids

        effective_target = min(
            target_hosting_fraction * host_fraction_2d_to_3d_scale_factor, 1.0)
        n_target     = int(np.round(effective_target * n))
        total_volume = sum(v for _, v in eligible)
        eligible_set = {gid for gid, _ in eligible}
        vol_map      = {gid: v for gid, v in eligible}

        # ── Build adjacency structures ────────────────────────────────────
        edges = cc3d.region_graph(self.lgi.astype(np.int32), connectivity=6)
        neigh: dict = defaultdict(set)
        n_neigh_count: dict = defaultdict(int)
        for edge in edges:
            a, b = int(edge[0]), int(edge[1])
            if a > 0 and b > 0:
                neigh[a].add(b)
                neigh[b].add(a)
                if a in eligible_set: n_neigh_count[a] += 1
                if b in eligible_set: n_neigh_count[b] += 1

        # Build networkx graph over eligible grains only
        eligible_neigh = {g: list(neigh[g] & eligible_set) for g in eligible_set}
        G = make_gid_net_from_neighlist(eligible_neigh)

        # ── Ranking score (same composite as allocate_twin_hosts) ─────────
        alpha   = float(np.clip(host_ranking_volume_weight, 0.0, 1.0))
        gids_e  = [gid for gid, _ in eligible]
        vols_e  = [v   for _, v   in eligible]
        neigh_e = [n_neigh_count.get(gid, 0) for gid in gids_e]
        sv = sorted(range(n), key=lambda i: vols_e[i],  reverse=True)
        sn = sorted(range(n), key=lambda i: neigh_e[i], reverse=True)
        rank_v = [0] * n;  rank_n = [0] * n
        for pos, i in enumerate(sv): rank_v[i] = pos
        for pos, i in enumerate(sn): rank_n[i] = pos
        score = {gids_e[i]: alpha * rank_v[i] + (1 - alpha) * rank_n[i]
                 for i in range(n)}   # lower score = better rank

        # ── Pool A: maximal independent set, best of mis_runs trials ─────
        best_mis: set = set()
        for k in range(mis_runs):
            candidate = set(maximal_independent_set(G, seed=seed + k))
            candidate &= eligible_set
            if len(candidate) > len(best_mis):
                best_mis = candidate

        n_pool_a = min(int(np.round(mis_fraction * n_target)), len(best_mis))
        pool_a = set(sorted(best_mis, key=lambda g: score[g])[:n_pool_a])

        # ── Pool B: near-mean-size grains that neighbour near-mean-size ──
        # Compute the requested size measure for all eligible grains
        _vs = getattr(self, 'voxel_size', 1.0)  # μm per voxel edge
        if pool_b_size_measure == 'vol_vox':
            size_vals = np.array([vol_map[g] for g in gids_e], dtype=float)
            size_map  = {g: float(vol_map[g]) for g in eligible_set}
            _measure_label = 'vol_vox  (grain volume in voxels)'
        elif pool_b_size_measure == 'vol_um3':
            _vox_vol = _vs ** 3
            size_vals = np.array([vol_map[g] * _vox_vol for g in gids_e])
            size_map  = {g: vol_map[g] * _vox_vol for g in eligible_set}
            _measure_label = f'vol_um3  (grain volume in μm³; voxel={_vs:.4f} μm)'
        elif pool_b_size_measure == 'eqdia_um':
            import math
            _vox_vol = _vs ** 3
            size_vals = np.array(
                [(6.0 * vol_map[g] * _vox_vol / math.pi) ** (1.0/3.0)
                 for g in gids_e])
            size_map  = {g: (6.0 * vol_map[g] * _vox_vol / math.pi) ** (1.0/3.0)
                         for g in eligible_set}
            _measure_label = f'eqdia_um  (sphere-equiv. diameter in μm; voxel={_vs:.4f} μm)'
        else:
            raise ValueError(
                f'pool_b_size_measure must be "vol_vox", "vol_um3", or '
                f'"eqdia_um"; got "{pool_b_size_measure}"')

        mean_size = float(np.mean(size_vals))
        std_size  = float(np.std(size_vals, ddof=1)) if len(size_vals) > 1 else 0.0

        # lo/hi bound whichever side(s) of the size distribution
        # pool_b_band_shape selects; 'below'/'above' only ever use one of
        # the two cutoffs the chosen method produces (the other is +-inf).
        if pool_b_band_method == 'percentile':
            p_lo = float(np.percentile(size_vals, pool_b_percentile_lo))
            p_hi = float(np.percentile(size_vals, pool_b_percentile_hi))
        else:  # 'std'
            band = pool_b_n_std * std_size
            p_lo = mean_size - band
            p_hi = mean_size + band

        if pool_b_band_shape == 'below':
            lo, hi = -np.inf, p_hi
        elif pool_b_band_shape == 'above':
            lo, hi = p_lo, np.inf
        else:  # 'between'
            lo, hi = p_lo, p_hi

        band_members = {g for g in eligible_set
                        if lo <= size_map[g] <= hi}
        # Split in-band, not-already-in-pool-A grains into those that DO
        # have an in-band neighbour and those that do NOT; neighbour_frac
        # controls how much of Pool B is drawn from each, with each side
        # falling back to the other if its own share can't be satisfied.
        eligible_b   = band_members - pool_a
        pool_b_do    = {g for g in eligible_b if any(nb in band_members for nb in neigh[g])}
        pool_b_donot = eligible_b - pool_b_do
        pool_b_candidates = pool_b_do  # kept for the existing summary/print field below

        n_pool_b    = max(0, n_target - len(pool_a))
        n_from_do   = int(round(n_pool_b * neighbour_frac / 100.0))
        n_from_do   = min(max(n_from_do, 0), n_pool_b)
        n_from_donot = n_pool_b - n_from_do

        do_sorted    = sorted(pool_b_do, key=lambda g: score[g])
        donot_sorted = sorted(pool_b_donot, key=lambda g: score[g])

        picked_do = do_sorted[:n_from_do]
        shortfall_do = n_from_do - len(picked_do)
        # donot's own share absorbs do's shortfall
        picked_donot = donot_sorted[:n_from_donot + shortfall_do]
        shortfall_donot = (n_from_donot + shortfall_do) - len(picked_donot)
        if shortfall_donot > 0:
            # top up from do's remainder (grains not already picked)
            picked_do = picked_do + do_sorted[len(picked_do):len(picked_do) + shortfall_donot]

        pool_b = set(picked_do) | set(picked_donot)

        host_ids    = pool_a | pool_b
        host_volume = sum(vol_map[g] for g in host_ids)

        ineligible_ids = set(vols.keys()) - eligible_set
        self.host_grain_ids     = host_ids
        self.non_host_grain_ids = (eligible_set - host_ids) | ineligible_ids
        self.eligible_grain_ids = eligible_set
        self.n_pool_a = len(pool_a)
        self.n_pool_b = len(pool_b)
        self.host_fraction_2d_to_3d_scale_factor = host_fraction_2d_to_3d_scale_factor
        self.host_ranking_volume_weight           = host_ranking_volume_weight
        self.target_hosting_fraction = target_hosting_fraction
        self.actual_hosting_fraction = (
            host_volume / total_volume if total_volume > 0 else 0.0)

        self.allocation_summary = {
            'target_hosting_fraction': target_hosting_fraction,
            'effective_target_3d': effective_target,
            'n_target': n_target,
            'n_eligible': n,
            'mis_best_size': len(best_mis),
            'mis_runs': mis_runs,
            'mis_seed_base': seed,
            'pool_b_size_measure': pool_b_size_measure,
            'pool_b_size_measure_label': _measure_label,
            'pool_b_band_method': pool_b_band_method,
            'pool_b_band_shape': pool_b_band_shape,
            'pool_b_tolerance_std': pool_b_n_std,
            'pool_b_percentile_lo': pool_b_percentile_lo,
            'pool_b_percentile_hi': pool_b_percentile_hi,
            'pool_b_mean_size': mean_size,
            'pool_b_std_size': std_size,
            'pool_b_band_lo': lo,
            'pool_b_band_hi': hi,
            'n_pool_b_candidates': len(pool_b_candidates),
            'pool_b_neighbour_frac': neighbour_frac,
            'n_pool_b_do_candidates': len(pool_b_do),
            'n_pool_b_donot_candidates': len(pool_b_donot),
            'n_pool_b_from_do': len(picked_do),
            'n_pool_b_from_donot': len(picked_donot),
            'n_pool_a': len(pool_a),
            'n_pool_b': len(pool_b),
            'n_host': len(host_ids),
            'n_non_host': len(self.non_host_grain_ids),
            'actual_hosting_fraction': self.actual_hosting_fraction,
        }

        print('TwinnedSimple3DBase.allocate_twin_hosts_spatial:')
        print(f'  EBSD 2D hosting fraction   : {target_hosting_fraction:.4f}')
        print(f'  3D effective target        : {effective_target:.4f}')
        print(f'  n_target                   : {n_target}')
        print(f'  MIS best size              : {len(best_mis)}  '
              f'(from {mis_runs} runs, seed base {seed})')
        print(f'  Pool A (MIS, {mis_fraction:.0%})         : {len(pool_a)} grains')
        print(f'  Pool B size measure        : {_measure_label}')
        _band_method_desc = (f'{pool_b_percentile_lo:g}/{pool_b_percentile_hi:g} percentile'
                              if pool_b_band_method == 'percentile' else f'+/-{pool_b_n_std:g} std')
        print(f'  Pool B band                : {_band_method_desc}, shape={pool_b_band_shape}  '
              f'(mean={mean_size:.3g}  std={std_size:.3g}  '
              f'band [{lo:.3g}, {hi:.3g}])')
        print(f'  Pool B candidates          : {len(pool_b_do)} do-neighbour, '
              f'{len(pool_b_donot)} do-not-neighbour')
        print(f'  Pool B neighbour fraction  : {neighbour_frac:.0f}%  '
              f'({len(picked_do)} from do-neighbour, {len(picked_donot)} from do-not)')
        print(f'  Pool B (in-band)           : {len(pool_b)} grains selected')
        print(f'  Total hosts                : {len(host_ids)}')
        print(f'  Non-host grains            : {len(self.non_host_grain_ids)}')
        print(f'  Achieved hosting fraction  : {self.actual_hosting_fraction:.4f}')

        return self.host_grain_ids, self.non_host_grain_ids

    def assess_hosting_representativeness_2d(
            self,
            n_comparison_slices: Union[int, dict] = 5,
            comparison_axes: Optional[list] = None,
            connectivity_2d: int = 4,
    ) -> dict:
        """
        Cross-sectional (2D) reassessment of the host/non-host allocation
        already computed by ``allocate_twin_hosts_spatial``, for direct
        comparability with a 2D EBSD-measured hosting fraction -- EBSD
        data is inherently 2D, but host selection above operates on the
        whole 3D structure, so neither ``actual_hosting_fraction`` (a 3D
        volume fraction) nor a raw 3D grain-count fraction is directly
        comparable to it.

        Must be called AFTER ``allocate_twin_hosts_spatial`` (reads
        ``self.host_grain_ids`` / ``self.eligible_grain_ids``).

        For each of ``n_comparison_slices`` evenly-spaced 2D
        cross-sections of ``self.lgi`` along each axis in
        ``comparison_axes``:

        1. Re-run ``cc3d.connected_components`` on the 2D slice
           (``connectivity_2d``) -- a single 3D-connected grain can
           appear as multiple spatially disconnected pieces within a
           given cross-section (Monte-Carlo grains can be highly
           convoluted), so tallying by the ORIGINAL 3D grain ID directly
           would misrepresent what a real 2D cross-section actually
           shows. This reassociation runs unconditionally on every
           slice -- the disconnection is invisible from raw ID counts
           alone, and its extent depends on ``connectivity_2d`` (a
           mismatch between whatever connectivity produced ``self.lgi``
           and this 2D re-slicing connectivity can throw the count off).
        2. Reassociate each freshly 2D-labelled region back to its
           underlying original 3D grain ID (a single lookup is safe -- a
           2D-connected region can never straddle two different original
           IDs, since ``cc3d`` only ever splits same-ID regions further,
           never merges differing ones) and thereby to its
           host / eligible-non-host / ineligible classification.
        3. Tally, per slice: the COUNT-based ratio (number of discrete
           2D host regions / number of discrete 2D eligible
           [host + non-host] regions) and the AREA-based ratio (host
           pixel count / eligible pixel count).

        Both ratios are then averaged (mean and std) across every
        sampled slice -- deliberately NOT filtered down to whichever
        slices happen to already be close to target, which would make
        the reported "achieved" value circular. The std is itself a
        useful representativeness diagnostic: a large spread means the
        host distribution is spatially inhomogeneous, so any single real
        EBSD cross-section would be a noisy comparison point regardless
        of how well this mean matches.

        Parameters
        ----------
        n_comparison_slices : int or dict
            Evenly-spaced 2D cross-sections sampled per axis. Either one
            int applied uniformly to every axis in ``comparison_axes``
            (unchanged from before), or a ``{'x': int, 'y': int, 'z': int}``
            dict for a different count per axis -- axes missing from the
            dict are not sampled, same as omitting them from
            ``comparison_axes``.
        comparison_axes : list of str or None
            Subset of ``['x', 'y', 'z']`` (array axes 0/1/2
            respectively -- same convention as
            ``rank_temporal_slices_by_n``). Defaults to all three.
        connectivity_2d : int
            ``cc3d`` connectivity for the 2D re-labelling (4 or 8).

        Returns
        -------
        dict with keys ``count_ratio_mean``, ``count_ratio_std``,
        ``area_ratio_mean``, ``area_ratio_std`` (``None`` if no slice
        yielded a usable ratio), ``n_slices_used``, and ``per_slice``
        (list of ``{'axis', 'location', 'count_ratio', 'area_ratio',
        'n_host_2d', 'n_eligible_2d'}`` dicts, one per sampled
        cross-section, for diagnostic inspection).
        """
        import cc3d
        from upxo.gsdataops.grid_ops import section_from_3d

        if self.host_grain_ids is None or self.eligible_grain_ids is None:
            raise ValueError(
                'allocate_twin_hosts_spatial must be called before '
                'assess_hosting_representativeness_2d.')

        if comparison_axes is None:
            comparison_axes = ['x', 'y', 'z']
        # self.lgi carries the raw MC array's native (nz, ny, nx) axis
        # order (axis0=Z, axis2=X -- see plot_temporal_slice_3d's P/R/C
        # convention comment); array shape is left as-is (a shared/native
        # representation other UPXO code may depend on) -- only the
        # LABEL mapping is corrected here so 'x'/'z' resolve to the
        # physically-correct index.
        axis_map = {'x': 2, 'y': 1, 'z': 0}
        axes_int = [axis_map[a.lower()] for a in comparison_axes
                    if a.lower() in axis_map]

        host_ids = self.host_grain_ids
        eligible_ids = self.eligible_grain_ids

        per_slice = []
        for ax in axes_int:
            axis_name = next(k for k, v in axis_map.items() if v == ax)
            n_this_axis = (n_comparison_slices[axis_name]
                           if isinstance(n_comparison_slices, dict) else n_comparison_slices)
            domain_size = self.lgi.shape[ax]
            slice_positions = np.linspace(
                0, domain_size - 1, n_this_axis, dtype=int)
            for slice_pos in slice_positions:
                lgi_2d = section_from_3d(self.lgi, axis=ax, location=int(slice_pos))
                relabelled = cc3d.connected_components(
                    lgi_2d, connectivity=connectivity_2d)

                n_host_2d = 0
                n_eligible_2d = 0
                area_host_2d = 0
                area_eligible_2d = 0
                for new_label in np.unique(relabelled):
                    if new_label == 0:
                        continue
                    mask = relabelled == new_label
                    # Safe: a 2D-connected region can never straddle two
                    # different original grain IDs -- cc3d only ever
                    # splits same-ID regions further.
                    orig_gid = int(lgi_2d[mask][0])
                    area = int(np.sum(mask))
                    if orig_gid in host_ids:
                        n_host_2d += 1
                        n_eligible_2d += 1
                        area_host_2d += area
                        area_eligible_2d += area
                    elif orig_gid in eligible_ids:
                        n_eligible_2d += 1
                        area_eligible_2d += area
                    # else: ineligible (below min_host_voxels) -- excluded
                    # from both numerator and denominator, matching how
                    # actual_hosting_fraction is normalised against
                    # eligible volume only, not the whole structure.

                count_ratio = n_host_2d / n_eligible_2d if n_eligible_2d > 0 else None
                area_ratio = area_host_2d / area_eligible_2d if area_eligible_2d > 0 else None
                per_slice.append({
                    'axis': axis_name,
                    'location': int(slice_pos),
                    'count_ratio': count_ratio,
                    'area_ratio': area_ratio,
                    'n_host_2d': n_host_2d,
                    'n_eligible_2d': n_eligible_2d,
                })

        count_ratios = [r['count_ratio'] for r in per_slice if r['count_ratio'] is not None]
        area_ratios = [r['area_ratio'] for r in per_slice if r['area_ratio'] is not None]

        return {
            'count_ratio_mean': float(np.mean(count_ratios)) if count_ratios else None,
            'count_ratio_std': float(np.std(count_ratios)) if len(count_ratios) > 1 else 0.0,
            'area_ratio_mean': float(np.mean(area_ratios)) if area_ratios else None,
            'area_ratio_std': float(np.std(area_ratios)) if len(area_ratios) > 1 else 0.0,
            'n_slices_used': len(per_slice),
            'per_slice': per_slice,
        }

    def domain_shape(self):
        """Return (nx, ny, nz) voxel dimensions of the domain.

        ``self.lgi`` is stored in the raw MC array's native (nz, ny, nx)
        axis order (see :meth:`plot_temporal_slice_3d`'s P/R/C convention
        comment) -- reversed here so the RETURNED tuple is genuinely
        (nx, ny, nz) as documented."""
        return tuple(reversed(self.lgi.shape))

    def physical_size(self):
        """Return physical domain size (Lx, Ly, Lz) in ``self.units``.

        See :meth:`domain_shape` -- ``self.lgi.shape`` is reversed first
        so this is genuinely (Lx, Ly, Lz), not (Lz, Ly, Lx)."""
        return tuple(s * self.voxel_size for s in reversed(self.lgi.shape))

    @classmethod
    def rank_temporal_slices_by_n(
            cls,
            pxt,
            start: int = 0,
            step: int = 5,
            ebsd_n_parents: Optional[int] = None,
            n_comparison_slices: int = 5,
            comparison_axes: Optional[list] = None,
            selected_props: Optional[list] = None,
            voxel_size: Optional[float] = None,
            verbose: bool = True,
    ) -> list:
        """
        Rank MC temporal slices by their 2D-equivalent grain count.

        The 3D total grain count is not directly comparable to a 2D EBSD
        cross-section count.  Instead, this method extracts
        ``n_comparison_slices`` evenly-spaced 2D slices along each axis
        in ``comparison_axes``, counts grains per slice, and reports the
        mean across all sampled slices as ``n_grains_2d_avg``.  This is
        the correct quantity to compare against the EBSD parent grain count.

        Reads the persisted LFI (``pxt.gs[t].lgi``, from
        :meth:`calculate_lfi`) directly rather than re-deriving a fresh
        connected-component labelling from the raw spin-state array
        (``pxt.gs[t].s``) -- a fresh relabelling of ``.s`` is not
        guaranteed to reproduce the same grain identity as the persisted,
        possibly-cleaned LFI once true grain count exceeds the number of
        Monte-Carlo states (see :meth:`clean_temporal_slices`'s
        docstring). Requires :meth:`calculate_lfi` to have already been
        run for every ranked slice.

        Parameters
        ----------
        pxt : mcgs
            Simulated mcgs object (after ``pxt.simulate()``).
        start : int
            Starting positional index into ``pxt.m``.
        step : int
            Positional increment -- every ``step``-th slice is sampled.
        ebsd_n_parents : int or None
            EBSD pure-parent grain count (2D); if provided, a ratio
            ``n_grains_2d_avg / ebsd_n_parents`` is computed and the
            default widget selection is set to the closest match.
        n_comparison_slices : int
            Number of evenly-spaced 2D slices to extract per axis.
            Default 5.
        comparison_axes : list of str or None
            Axes to extract slices along.  Any subset of
            ``['x', 'y', 'z']``.  Defaults to all three if None.
        selected_props : list of str or None
            Optional -- subset of ``upxo.viz.ebsdviz.GRAIN_ROLE_ALL_PROPS``
            (``'area'``, ``'aspect_ratio'``, ``'perimeter'``,
            ``'solidity'``, ``'n_neighbours'``). When given, pools these
            per-grain morphological properties from the SAME already-2D
            -relabelled slices used for the grain-count ranking above
            (via ``regionprops``/``adjacency_from_labels``), fusing in
            what :meth:`compute_slice_grain_properties` would otherwise
            compute from a SECOND, independent ``cc3d.connected_components``
            pass over the identical slices -- for a caller (Synthetic GS
            Assessment's "Rank Temporal Slices") that always wants both,
            this halves the per-slice relabelling cost. Each returned row
            then also carries ``'prop_stats_raw'``. Leave ``None`` (the
            default) for a plain ranking with no property pooling.
        voxel_size : float or None
            Only used when ``selected_props`` is given -- forwarded to the
            same area/perimeter scaling :meth:`compute_slice_grain_properties`
            applies. Defaults to ``mean(pxt.vox_size)`` if None.

        Returns
        -------
        list of dict, each with keys
            ``tslice_key``, ``n_grains_2d_avg``, ``n_slices_used``,
            ``ratio`` (None if no reference), plus the full-3D topology
            counts derived from the SAME ``lgi_3d`` labelling already
            computed here (no extra labelling pass):
            ``n_grains_3d`` (total 3D grain count), ``n_face_grains``
            (grains with at least one voxel on any of the 6 RVE
            boundary faces), ``n_interior_grains`` (``n_grains_3d -
            n_face_grains``), and ``pct_ng_in_3d`` (percentage of
            ``n_grains_3d`` that are completely interior -- no voxel on
            any of the 6 RVE faces -- ``None`` if ``n_grains_3d`` is 0).
            Also, from the SAME per-slice ``lgi_2d`` relabelling already
            done for ``n_grains_2d_avg`` (no extra labelling pass):
            ``pct_ng_in_2d_mean`` / ``pct_ng_in_2d_std`` -- mean/std,
            across every sampled 2D cross-section, of the percentage of
            that slice's grains with no pixel on any of the slice's 4
            edges (the 2D analogue of ``pct_ng_in_3d``, comparable to a
            real EBSD map's own edge-grain fraction --
            see ``ebsdviz.compute_pct_interior_grains``). Plus, only when
            ``selected_props`` is given: ``prop_stats_raw`` --
            ``{prop_name: ndarray}``, identical in definition to
            :meth:`compute_slice_grain_properties`'s return value.
        """
        import cc3d
        from upxo.gsdataops.grid_ops import section_from_3d

        if comparison_axes is None:
            comparison_axes = ['x', 'y', 'z']

        need_props = bool(selected_props)
        if need_props:
            from skimage.measure import regionprops
            from upxo.netops.neighops import adjacency_from_labels
            need_regionprops = any(p != 'n_neighbours' for p in selected_props)
            need_neighbours = 'n_neighbours' in selected_props
            vs = float(voxel_size) if voxel_size is not None else float(np.mean(pxt.vox_size))

        # self.lgi carries the raw MC array's native (nz, ny, nx) axis
        # order (axis0=Z, axis2=X -- see plot_temporal_slice_3d's P/R/C
        # convention comment); array shape is left as-is (a shared/native
        # representation other UPXO code may depend on) -- only the
        # LABEL mapping is corrected here so 'x'/'z' resolve to the
        # physically-correct index.
        axis_map = {'x': 2, 'y': 1, 'z': 0}
        axes_int = [axis_map[a.lower()] for a in comparison_axes
                    if a.lower() in axis_map]

        # positional index into pxt.m, NOT a key into pxt.gs directly --
        # pxt.gs is keyed by the actual MC step number a slice was saved
        # at (e.g. 0, 2, 4, ... when save_interval=2), which only equals
        # its position in pxt.m when every single step was saved.
        temporal_positions = np.arange(start, len(pxt.m), step)
        rows = []
        n_total = len(temporal_positions)

        for i, pos in enumerate(temporal_positions):
            tslice_key = pxt.m[int(pos)]
            if verbose:
                print(f"  [{i + 1}/{n_total}] tslice={tslice_key} ...", end='', flush=True)
            gstslice = pxt.gs[tslice_key]
            if not hasattr(gstslice, 'lgi') or gstslice.lgi is None:
                raise RuntimeError(
                    f'calculate_lfi() must be run before '
                    f'rank_temporal_slices_by_n() -- missing LFI for '
                    f'slice {tslice_key}.')
            lgi_3d = gstslice.lgi

            slice_counts = []
            slice_pct_interior = []
            pooled = {p: [] for p in selected_props} if need_props else None
            for ax in axes_int:
                domain_size = lgi_3d.shape[ax]
                slice_positions = np.linspace(
                    0, domain_size - 1, n_comparison_slices, dtype=int)
                for slice_pos in slice_positions:
                    lgi_2d = section_from_3d(lgi_3d, axis=ax, location=int(slice_pos))
                    # Re-label in 2D: a single 3D-connected MC grain can be
                    # complex enough (fingering, branching) to intersect this
                    # slice plane as two or more spatially disjoint pieces
                    # that still carry the same 3D label ID. Counting unique
                    # label VALUES would undercount them as one grain; 2D
                    # connectivity is what actually defines a grain in a
                    # cross-section.
                    lgi_2d = cc3d.connected_components(lgi_2d, connectivity=4)
                    n_slice = int(np.unique(lgi_2d[lgi_2d > 0]).size)
                    slice_counts.append(n_slice)

                    # Percentage of this slice's grains with no pixel on
                    # any of the slice's 4 edges -- the 2D analogue of the
                    # 3D face/interior split below, directly comparable to
                    # a real EBSD map's own edge-grain fraction (a single
                    # 2D measurement, see ebsdviz.compute_pct_interior_grains).
                    if n_slice > 0:
                        edge_labels = set()
                        edge_labels.update(np.unique(lgi_2d[0, :]).tolist())
                        edge_labels.update(np.unique(lgi_2d[-1, :]).tolist())
                        edge_labels.update(np.unique(lgi_2d[:, 0]).tolist())
                        edge_labels.update(np.unique(lgi_2d[:, -1]).tolist())
                        edge_labels.discard(0)
                        n_interior_2d = n_slice - len(edge_labels)
                        slice_pct_interior.append(100.0 * n_interior_2d / n_slice)

                    # Pool morphological properties from THIS SAME relabelled
                    # slice (see selected_props docstring above) instead of
                    # compute_slice_grain_properties() re-slicing and
                    # re-labelling it from scratch.
                    if need_props:
                        try:
                            if need_regionprops:
                                for region in regionprops(lgi_2d):
                                    if 'area' in pooled:
                                        pooled['area'].append(region.area * vs ** 2)
                                    if 'perimeter' in pooled:
                                        pooled['perimeter'].append(region.perimeter * vs)
                                    if 'solidity' in pooled:
                                        pooled['solidity'].append(region.solidity)
                                    if 'aspect_ratio' in pooled:
                                        maj = region.major_axis_length
                                        mn = region.minor_axis_length
                                        if mn > 0:
                                            pooled['aspect_ratio'].append(maj / mn)
                            if need_neighbours:
                                neigh_map = adjacency_from_labels(lgi_2d, connectivity=4)
                                for neighbours in neigh_map.values():
                                    pooled['n_neighbours'].append(len(neighbours))
                        except Exception as e:
                            print(f"    Warning: property pooling failed for "
                                  f"tslice={tslice_key} axis={ax} "
                                  f"slice={slice_pos}: {e}")

            n_avg = float(np.mean(slice_counts)) if slice_counts else 0.0
            ratio = n_avg / ebsd_n_parents if ebsd_n_parents else None
            pct_ng_in_2d_mean = float(np.mean(slice_pct_interior)) if slice_pct_interior else None
            pct_ng_in_2d_std = float(np.std(slice_pct_interior)) if len(slice_pct_interior) > 1 else 0.0

            n_grains_3d = int(np.unique(lgi_3d[lgi_3d > 0]).size)
            face_labels = set()
            for ax in range(3):
                idx_lo = [slice(None)] * 3
                idx_lo[ax] = 0
                idx_hi = [slice(None)] * 3
                idx_hi[ax] = -1
                face_labels.update(np.unique(lgi_3d[tuple(idx_lo)]).tolist())
                face_labels.update(np.unique(lgi_3d[tuple(idx_hi)]).tolist())
            face_labels.discard(0)
            n_face_grains = len(face_labels)
            n_interior_grains = n_grains_3d - n_face_grains
            pct_ng_in_3d = (
                100.0 * n_interior_grains / n_grains_3d if n_grains_3d > 0 else None)

            row = {
                'tslice_key': int(tslice_key),
                'n_grains_2d_avg': n_avg,
                'n_slices_used': len(slice_counts),
                'ratio': ratio,
                'n_grains_3d': n_grains_3d,
                'n_face_grains': n_face_grains,
                'n_interior_grains': n_interior_grains,
                'pct_ng_in_3d': pct_ng_in_3d,
                'pct_ng_in_2d_mean': pct_ng_in_2d_mean,
                'pct_ng_in_2d_std': pct_ng_in_2d_std,
            }
            if need_props:
                row['prop_stats_raw'] = {p: np.array(v, dtype=float) for p, v in pooled.items()}
            rows.append(row)

            if verbose:
                ratio_txt = f", ratio={ratio:.2f}" if ratio is not None else ""
                props_txt = ""
                if need_props:
                    props_txt = "  |  pooled: " + ", ".join(
                        f"{p}(n={len(pooled[p])})" for p in selected_props)
                print(f"  2D-avg grains={n_avg:.1f} (n_slices={len(slice_counts)}{ratio_txt})"
                      f"  3D grains={n_grains_3d} (face={n_face_grains}, "
                      f"interior={n_interior_grains}){props_txt}")

        return rows

    @classmethod
    def compute_slice_grain_properties(
            cls,
            pxt,
            tslice_key: int,
            selected_props: list,
            comparison_axes: Optional[list] = None,
            n_comparison_slices: int = 5,
            voxel_size: Optional[float] = None,
    ) -> dict:
        """
        Pool per-grain morphological property values across 2D
        cross-sections of one MC temporal slice, sampled the same way
        :meth:`rank_temporal_slices_by_n` samples slices for grain
        counting.

        At this pipeline stage the synthetic structure has no twin/parent
        role labels yet (those only exist on the EBSD side, post
        ``identify_parent_grains``) -- every grain found in the sampled
        cross-sections is pooled, matching the "untwinned whole grain"
        analogue of an EBSD parent grain.

        Property definitions mirror
        ``upxo.viz.ebsdviz.compute_grain_role_property_distributions``:
        ``area`` and ``perimeter`` are scaled once by ``voxel_size`` (in
        the corresponding power), ``aspect_ratio`` is
        ``major_axis_length / minor_axis_length``, ``solidity`` is
        skimage's own ``area / convex_area``, and ``n_neighbours`` comes
        from face-adjacency on the labelled 2D slice.

        Reads the persisted LFI (``pxt.gs[t].lgi``, from
        :meth:`calculate_lfi`) directly rather than re-deriving a fresh
        connected-component labelling from ``pxt.gs[t].s`` -- see
        :meth:`rank_temporal_slices_by_n`'s docstring for why. Requires
        :meth:`calculate_lfi` to have already been run for this slice.

        Parameters
        ----------
        pxt : mcgsV1_1 (or mcgs)
            Simulated grain-growth object (after ``pxt.simulate()``).
        tslice_key : int
            Key into ``pxt.gs``.
        selected_props : list of str
            Subset of ``upxo.viz.ebsdviz.GRAIN_ROLE_ALL_PROPS``.
        comparison_axes : list of str or None
            Axes to sample, subset of ``['x', 'y', 'z']``. Defaults to all three.
        n_comparison_slices : int
            Evenly-spaced 2D slices to extract per axis. Default 5.
        voxel_size : float or None
            Physical voxel edge length (assumed isotropic in-plane, same
            assumption the EBSD side makes with a single ``step_size``).
            Reads the mean of ``pxt.vox_size`` if None.

        Returns
        -------
        dict
            ``{prop_name: ndarray of pooled values}``, one entry per
            requested property.
        """
        import cc3d
        from skimage.measure import regionprops
        from upxo.gsdataops.grid_ops import section_from_3d
        from upxo.netops.neighops import adjacency_from_labels

        if comparison_axes is None:
            comparison_axes = ['x', 'y', 'z']
        # self.lgi carries the raw MC array's native (nz, ny, nx) axis
        # order (axis0=Z, axis2=X -- see plot_temporal_slice_3d's P/R/C
        # convention comment); array shape is left as-is (a shared/native
        # representation other UPXO code may depend on) -- only the
        # LABEL mapping is corrected here so 'x'/'z' resolve to the
        # physically-correct index.
        axis_map = {'x': 2, 'y': 1, 'z': 0}
        axes_int = [axis_map[a.lower()] for a in comparison_axes
                    if a.lower() in axis_map]

        vs = float(voxel_size) if voxel_size is not None else float(np.mean(pxt.vox_size))
        need_regionprops = any(p != 'n_neighbours' for p in selected_props)
        need_neighbours = 'n_neighbours' in selected_props

        gstslice = pxt.gs[tslice_key]
        if not hasattr(gstslice, 'lgi') or gstslice.lgi is None:
            raise RuntimeError(
                f'calculate_lfi() must be run before '
                f'compute_slice_grain_properties() -- missing LFI for '
                f'slice {tslice_key}.')
        lgi_3d = gstslice.lgi

        pooled = {p: [] for p in selected_props}
        for ax in axes_int:
            domain_size = lgi_3d.shape[ax]
            slice_positions = np.linspace(
                0, domain_size - 1, n_comparison_slices, dtype=int)
            for pos in slice_positions:
                lgi_2d = section_from_3d(lgi_3d, axis=ax, location=int(pos))
                # Re-label in 2D -- see the matching comment in
                # rank_temporal_slices_by_n. Without this, two spatially
                # disjoint pieces of the same 3D grain sharing one label ID
                # get treated by regionprops as a single region spanning
                # both, producing nonsensical merged geometry (this is what
                # was causing the multi-million aspect-ratio outliers).
                lgi_2d = cc3d.connected_components(lgi_2d, connectivity=4)

                if need_regionprops:
                    for region in regionprops(lgi_2d):
                        if 'area' in pooled:
                            pooled['area'].append(region.area * vs ** 2)
                        if 'perimeter' in pooled:
                            pooled['perimeter'].append(region.perimeter * vs)
                        if 'solidity' in pooled:
                            pooled['solidity'].append(region.solidity)
                        if 'aspect_ratio' in pooled:
                            maj, mn = region.major_axis_length, region.minor_axis_length
                            if mn > 0:
                                pooled['aspect_ratio'].append(maj / mn)

                if need_neighbours:
                    neigh_map = adjacency_from_labels(lgi_2d, connectivity=4)
                    for neighbours in neigh_map.values():
                        pooled['n_neighbours'].append(len(neighbours))

        return {p: np.array(v, dtype=float) for p, v in pooled.items()}

    @staticmethod
    def _iqr_trim(arr: np.ndarray, trim_left: bool = True, trim_right: bool = True) -> np.ndarray:
        """Drop points outside the Tukey (1.5*IQR) fence.

        Parameters
        ----------
        trim_left : bool
            Drop points below Q1 - 1.5*IQR (the low/small-value tail).
        trim_right : bool
            Drop points above Q3 + 1.5*IQR (the high/large-value tail).
        Either side can be disabled independently -- e.g. for a grain-area
        distribution, a caller may want to exclude only the large-outlier
        tail while keeping every small grain, or vice versa.

        Returns arr unchanged if too few points (< 4) to define quartiles
        meaningfully, or if both sides are disabled."""
        arr = np.asarray(arr, dtype=float)
        if arr.size < 4 or not (trim_left or trim_right):
            return arr
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = np.ones(arr.shape, dtype=bool)
        if trim_left:
            mask &= arr >= lo
        if trim_right:
            mask &= arr <= hi
        trimmed = arr[mask]
        return trimmed if trimmed.size > 0 else arr

    @staticmethod
    def _squash_distance(x: float, score_function: str = 'exp') -> float:
        """
        Maps a non-negative, already-dimensionless distance ``x`` (see
        ``compare_property_distributions``'s ``*_score`` fields) to a
        [0, 1] similarity score: 1.0 at ``x = 0`` (identical
        distributions), decaying smoothly toward 0.0 as ``x`` grows.

        ``score_function``:
          - ``'exp'`` (default): ``exp(-x)`` -- the standard
            distance-to-similarity transform (same idea as an RBF
            kernel); decays faster near 0.
          - ``'reciprocal'``: ``1 / (1 + x)`` -- decays more gradually.
        """
        if score_function == 'reciprocal':
            return float(1.0 / (1.0 + x))
        return float(np.exp(-x))

    @classmethod
    def compare_property_distributions(cls, ebsd_vals, synth_vals,
                                        trim_left: bool = True, trim_right: bool = True,
                                        score_function: str = 'exp') -> Optional[dict]:
        """
        Compare an EBSD property distribution against a synthetic
        candidate's, after removing outliers from each side independently
        (IQR / Tukey's rule).

        Uses Wasserstein distance and energy distance (both scipy,
        operating directly on the two 1D sample arrays, in the property's
        own units) plus a KS-test-derived similarity score. These three
        specifically for property-distribution comparison, not (yet) for
        network-level grain-structure representativeness assessment,
        which is a separate, later piece of work.

        Wasserstein/energy distance are unbounded and expressed in the
        property's own units, so they're neither comparable across
        properties (Area's µm² vs Perimeter's µm) nor usable directly as
        a [0, 1] "representativeness" score. Both are first made
        dimensionless by dividing by the EBSD reference's own trimmed std
        (its natural scale of variability), then mapped to [0, 1] via
        ``_squash_distance``. ``ks_similarity`` is already bounded in
        [0, 1] (``1 - KS statistic``) and used as-is. The mean of the
        three is ``representativeness_score`` -- an absolute (not
        candidate-set-relative) [0, 1] measure, comparable across
        properties and stable regardless of what other candidates exist.

        Parameters
        ----------
        ebsd_vals, synth_vals : array-like
            Raw per-grain property values (not pre-aggregated).
        trim_left, trim_right : bool
            Which Tukey-fence side(s) to trim before comparing -- applied
            identically to both the EBSD and synthetic sides. See
            ``_iqr_trim``.
        score_function : str
            ``'exp'`` (default) or ``'reciprocal'`` -- see
            ``_squash_distance``.

        Returns
        -------
        dict or None
            ``{'wasserstein': float, 'energy': float, 'ks_similarity': float,
            'ratio': float, 'wasserstein_score': float, 'energy_score':
            float, 'representativeness_score': float}`` -- ``ratio`` is
            (trimmed synthetic mean) / (trimmed EBSD mean), the same
            quantity the property-column star tolerance check is applied
            to. ``None`` if either side has no values left after
            trimming.
        """
        from scipy.stats import wasserstein_distance, energy_distance, ks_2samp

        e = cls._iqr_trim(np.asarray(ebsd_vals, dtype=float), trim_left=trim_left, trim_right=trim_right)
        s = cls._iqr_trim(np.asarray(synth_vals, dtype=float), trim_left=trim_left, trim_right=trim_right)
        if e.size == 0 or s.size == 0:
            return None

        wd = float(wasserstein_distance(e, s))
        ed = float(energy_distance(e, s))
        ks_stat, _ks_p = ks_2samp(e, s)
        ks_similarity = 1.0 - float(ks_stat)

        e_mean = float(np.mean(e))
        ratio = float(np.mean(s) / e_mean) if e_mean != 0 else float('nan')

        e_std = float(np.std(e))
        wd_norm = (wd / e_std) if e_std > 0 else (0.0 if wd == 0 else float('inf'))
        ed_norm = (ed / e_std) if e_std > 0 else (0.0 if ed == 0 else float('inf'))
        wasserstein_score = cls._squash_distance(wd_norm, score_function)
        energy_score = cls._squash_distance(ed_norm, score_function)
        representativeness_score = float(np.mean([wasserstein_score, energy_score, ks_similarity]))

        return {
            'wasserstein': wd,
            'energy': ed,
            'ks_similarity': ks_similarity,
            'ratio': ratio,
            'wasserstein_score': wasserstein_score,
            'energy_score': energy_score,
            'representativeness_score': representativeness_score,
        }

    @staticmethod
    def rescale_property_values(prop_name: str, values, scale_factor: float):
        """
        Convert one candidate's raw (voxel-unit) property values into
        physical units using an already-derived linear scale factor
        (micrometres per voxel edge length -- see
        ``calibrate_scale_factor``).

        Only length-based properties change: area scales as
        (scale_factor)**2, perimeter as (scale_factor)**1. Aspect ratio,
        solidity, and n_neighbours are dimensionless/topological and are
        returned unchanged -- rescaling voxel size cannot affect them.
        """
        values = np.asarray(values, dtype=float)
        if prop_name == 'area':
            return values * scale_factor ** 2
        if prop_name == 'perimeter':
            return values * scale_factor
        return values

    @classmethod
    def calibrate_scale_factor(
            cls,
            pxt,
            tslice_key: int,
            ebsd_area_vals,
            synth_area_vals,
            ebsd_perimeter_vals=None,
            synth_perimeter_vals=None,
            trim_left: bool = True,
            trim_right: bool = True,
            cross_check_tolerance_pct: float = 5.0,
    ) -> Optional[dict]:
        """
        Derive the physical voxel edge length (micrometres/voxel) that
        makes this candidate temporal slice's mean 2D cross-sectional
        grain area match the EBSD reference's mean grain area, then
        sanity-check it against an independently-derived perimeter-based
        scale factor.

        Method (see the twinned_simple_3d GUI design discussion this
        implements):

        1. Outlier-trim (IQR/Tukey) both the EBSD and synthetic Area
           distributions independently, take their means.
        2. Area scales as (length)**2, so
           ``scale_factor = sqrt(ebsd_mean_area / synth_mean_area_voxels)``
           is the implied voxel edge length in the EBSD's physical units.
           This is computed independently per candidate -- deliberately no
           relationship is assumed across temporal slices, since MC step
           number here is used purely as a stochastic generator of
           candidate grain topologies, not a physical growth timeline.
        3. If Perimeter values are also available, repeat with Perimeter
           (which scales as length**1, no sqrt) as an independent
           cross-check: if the two scale factors disagree by more than
           ``cross_check_tolerance_pct``, the candidate's grains have a
           different area-to-perimeter relationship (shape family) than
           the real EBSD grains, so forcing an area match there would not
           be physically meaningful -- ``cross_check_ok`` is False.

        Caveats callers/UI should surface to the user (see the GUI design
        notes this implements):

        * The perimeter cross-check is measured on a voxel/pixel grid, so
          it is biased by discretisation ("staircase") error -- coarser
          voxel resolution per grain inflates measured perimeter more,
          which can fail the cross-check for a purely digitisation reason
          rather than a genuine shape mismatch. ``n_synth_grains`` /
          mean voxels-per-grain should be considered before trusting a
          failed cross-check as a real shape difference.
        * The scale factor is itself a mean-based estimate -- candidates
          with few grains (e.g. heavily coarsened, late-stage slices) give
          a statistically noisier estimate. ``n_ebsd_grains`` and
          ``n_synth_grains`` (the outlier-trimmed sample sizes actually
          used) are returned so this can be judged.
        * This calibrates *size* only. It says nothing about whether grain
          *shape*/anisotropy is representative -- that is judged
          separately by the property distribution comparisons
          (``compare_property_distributions``).

        Parameters
        ----------
        pxt : mcgsV1_1 (or mcgs)
            Simulated grain-growth object (after ``pxt.simulate()``).
        tslice_key : int
            Key into ``pxt.gs`` -- used to read the candidate's voxel grid
            shape for the implied RVE size.
        ebsd_area_vals, synth_area_vals : array-like
            Raw per-grain Area values (EBSD in physical units; synthetic
            in voxel units).
        ebsd_perimeter_vals, synth_perimeter_vals : array-like or None
            Raw per-grain Perimeter values, for the cross-check. If either
            is None, the cross-check is skipped (``cross_check_ok`` is
            None, not False).
        trim_left, trim_right : bool
            Outlier-trim sides, applied identically to area and perimeter
            (matching whatever the Area/Perimeter columns are configured
            with elsewhere).
        cross_check_tolerance_pct : float
            Max allowed relative disagreement between the area- and
            perimeter-derived scale factors for the cross-check to pass.

        Returns
        -------
        dict or None
            ``{'scale_factor': float, 'n_ebsd_grains': int,
            'n_synth_grains': int, 'scale_factor_perimeter': float or None,
            'cross_check_ok': bool or None, 'cross_check_diff_pct': float or None,
            'implied_rve_size_um': (x, y, z)}`` -- ``None`` if area data is
            insufficient (empty after trimming) to derive a scale factor
            at all.
        """
        e_area = cls._iqr_trim(np.asarray(ebsd_area_vals, dtype=float), trim_left=trim_left, trim_right=trim_right)
        s_area = cls._iqr_trim(np.asarray(synth_area_vals, dtype=float), trim_left=trim_left, trim_right=trim_right)
        if e_area.size == 0 or s_area.size == 0:
            return None

        e_area_mean = float(np.mean(e_area))
        s_area_mean = float(np.mean(s_area))
        if s_area_mean <= 0:
            return None
        scale_factor = float(np.sqrt(e_area_mean / s_area_mean))

        scale_factor_perimeter = None
        cross_check_ok = None
        cross_check_diff_pct = None
        if ebsd_perimeter_vals is not None and synth_perimeter_vals is not None:
            e_per = cls._iqr_trim(np.asarray(ebsd_perimeter_vals, dtype=float),
                                   trim_left=trim_left, trim_right=trim_right)
            s_per = cls._iqr_trim(np.asarray(synth_perimeter_vals, dtype=float),
                                   trim_left=trim_left, trim_right=trim_right)
            if e_per.size > 0 and s_per.size > 0:
                s_per_mean = float(np.mean(s_per))
                if s_per_mean > 0:
                    scale_factor_perimeter = float(np.mean(e_per) / s_per_mean)
                    cross_check_diff_pct = 100.0 * abs(scale_factor_perimeter - scale_factor) / scale_factor
                    cross_check_ok = cross_check_diff_pct <= cross_check_tolerance_pct

        s = pxt.gs[tslice_key].s
        nz, ny, nx = s.shape  # axis0=Z, axis1=Y, axis2=X -- see mcgsV1_1.__init__
        implied_rve_size_um = (nx * scale_factor, ny * scale_factor, nz * scale_factor)

        return {
            'scale_factor': scale_factor,
            'n_ebsd_grains': int(e_area.size),
            'n_synth_grains': int(s_area.size),
            'scale_factor_perimeter': scale_factor_perimeter,
            'cross_check_ok': cross_check_ok,
            'cross_check_diff_pct': cross_check_diff_pct,
            'implied_rve_size_um': implied_rve_size_um,
        }

    @classmethod
    def plot_temporal_slice_3d(cls, pxt, tslice_key: int, title: Optional[str] = None):
        """
        Render one MC temporal slice's persisted LFI (``pxt.gs[t].lgi``,
        from :meth:`calculate_lfi`) in a separate PyVista window, each
        grain shown in a distinct colour via a qualitative colormap.

        Reads the persisted LFI directly rather than re-deriving a fresh
        connected-component labelling from ``pxt.gs[t].s`` -- see
        :meth:`rank_temporal_slices_by_n`'s docstring for why. Requires
        :meth:`calculate_lfi` to have already been run for this slice.

        Parameters
        ----------
        pxt : mcgsV1_1 (or mcgs)
            Simulated grain-growth object (after ``pxt.simulate()`` and
            ``calculate_lfi()``).
        tslice_key : int
            Key into ``pxt.gs``.
        title : str or None
            Window title text; defaults to a slice/grain-count summary.
        """
        from upxo.pxtal.twinned_simple_3d.viz_3d import render_lgi_3d

        gstslice = pxt.gs[tslice_key]
        if not hasattr(gstslice, 'lgi') or gstslice.lgi is None:
            raise RuntimeError(
                f'calculate_lfi() must be run before plot_temporal_slice_3d() '
                f'-- missing LFI for slice {tslice_key}.')
        lgi_3d = gstslice.lgi
        n_grains = int(np.unique(lgi_3d[lgi_3d > 0]).size)

        vs = pxt.vox_size if hasattr(pxt.vox_size, '__len__') else (pxt.vox_size,) * 3

        # lgi_3d (like the raw .s it's derived from) has shape
        # (nz, ny, nx) -- axis0=Z, axis1=Y, axis2=X, matching
        # alg300a/alg300b's P/R/C convention (mcgsV1_1.__init__);
        # find_grains() populates .lgi straight from .s with no
        # transpose, so this axis order carries over unchanged. PyVista's
        # ImageData.dimensions is always positionally (nx, ny, nz)
        # regardless of the source array's axis order -- assigning
        # lgi_3d.shape directly silently mislabels X<->Z, which (since
        # the box's extents differ per axis) visibly distorts which
        # grains render as spatially contiguous. Transpose to
        # (nx, ny, nz) before handing off to render_lgi_3d, which assumes
        # its input is already in that axis order.
        lgi_xyz = np.transpose(lgi_3d, (2, 1, 0))

        render_lgi_3d(
            lgi_xyz, voxel_size=vs,
            title=title or f"Temporal slice t={tslice_key}  ({n_grains} grains)")

    @classmethod
    def calculate_lfi(
            cls,
            pxt,
            tslice_keys: Optional[list] = None,
            verbose: bool = True,
    ) -> dict:
        """
        Compute and persist the Labelled Feature Index (LFI -- ``.lgi``, a
        slight generalisation of "local"/"labelled grain index") for every
        saved MC temporal slice at once, via
        ``gstslice.char_morphology_of_grains`` -- the same call
        :meth:`from_mcgs` uses lazily, run here explicitly and in bulk
        instead.

        Run this once, right after simulation and before
        :meth:`clean_temporal_slices` -- cleaning then operates directly
        on this persisted LFI rather than re-deriving a fresh label field
        from ``.s`` on demand, giving every downstream consumer (cleaning,
        ``from_mcgs``, and anything else that reads ``gstslice.lgi``) a
        single, explicitly-computed source of truth instead of a
        lazily-cached value whose freshness relative to ``.s`` was
        previously only guaranteed by GUI click ordering.

        Parameters
        ----------
        pxt : mcgsV1_1 (or mcgs)
            Simulated grain-growth object (after ``pxt.simulate()``).
        tslice_keys : list of int or None
            Which ``pxt.m`` entries to compute LFI for. Defaults to all
            of them.
        verbose : bool
            Print a per-slice grain-count summary.

        Returns
        -------
        dict
            ``{tslice_key: {'n_grains': int}}``
        """
        if tslice_keys is None:
            tslice_keys = list(pxt.m)

        summary = {}
        for tslice_key in tslice_keys:
            gstslice = pxt.gs[tslice_key]
            gstslice.char_morphology_of_grains(
                label_str_order=1,
                find_grain_voxel_locs=True,
                find_spatial_bounds_of_grains=True,
                force_compute=True,
            )
            n_grains = int(np.unique(gstslice.lgi[gstslice.lgi > 0]).size)
            summary[tslice_key] = {'n_grains': n_grains}
            if verbose:
                print(f"  t={tslice_key}: {n_grains} grain(s)")

        return summary

    @classmethod
    def clean_temporal_slices(
            cls,
            pxt,
            min_grain_size: int = 4,
            tslice_keys: Optional[list] = None,
            do_merge_small: bool = True,
            do_spike_removal: bool = True,
            verbose: bool = True,
    ) -> dict:
        """
        Clean every saved MC temporal slice's persisted LFI
        (``pxt.gs[t].lgi``) in place: merge small grains into their
        largest neighbour, then remove spike voxels -- the same two
        defect classes
        ``upxo.pxtal.twinned_simple_3d.cleaning_3d.StructureCleaner3D``
        already handles for post-twin structures, generalised to the
        base (pre-twin) MC structure here.

        Requires :meth:`calculate_lfi` to have already been run for every
        targeted slice (raises if ``pxt.gs[t].lgi`` is missing) --
        cleaning operates directly on the persisted LFI rather than
        re-deriving a fresh connected-component labelling from
        ``pxt.gs[t].s`` on every call, so there is a single, explicit,
        trusted source of truth for "the current grain labelling" that
        every downstream consumer (ranking, property stats, GS Plot,
        ``from_mcgs``) can rely on without guessing whether it reflects
        the latest cleaning pass.

        The cleaned result is written back to BOTH ``pxt.gs[t].lgi`` (the
        persisted LFI, now updated) and ``pxt.gs[t].s`` (as STATE values,
        not label IDs, so ``.s`` keeps meaning "MC state" for any code
        that still reads it directly) -- kept in sync so nothing
        downstream needs to be aware cleaning happened elsewhere.

        Calling this again on an already-cleaned slice cleans the
        ALREADY-CLEANED LFI a second time (not fresh raw simulation
        data) -- ``.lgi`` and ``.s`` are already in sync after the first
        pass, so a second pass with the same ``min_grain_size`` is
        normally a no-op; it only does something new if
        ``min_grain_size`` changed since the first pass. The GUI
        surfaces this explicitly before re-cleaning.

        Small-grain removal is a 3D generalisation of
        ``upxo.pxtalops.gssmooth2d._merge_small_grains`` (2D-only, Shapely
        + ``find_neighs2d``-based), built instead from the shape-agnostic
        primitives ``gid_ops.find_small_fids`` and
        ``netops.neighops.adjacency_from_labels`` (3D via
        ``cc3d.contacts``), with every pass's merge decisions applied via
        a single vectorised lookup-table relabelling rather than a
        per-grain ``gid_ops.merge_label_into`` call. Spike removal reuses
        ``StructureCleaner3D._fast_clean_spikes`` directly.

        Parameters
        ----------
        pxt : mcgsV1_1 (or mcgs)
            Simulated grain-growth object (after ``pxt.simulate()`` and
            ``calculate_lfi()``). ``pxt.gs[t].lgi`` and ``pxt.gs[t].s``
            are both overwritten in place for every cleaned slice.
        min_grain_size : int
            Grains with fewer voxels than this are merged into their
            largest (by voxel count) neighbour, repeated until no grain
            remains below the threshold (or it has no neighbours left to
            merge into).
        tslice_keys : list of int or None
            Which ``pxt.m`` entries to clean. Defaults to all of them.
        do_merge_small : bool
            Run Stage 1 (small-grain merging). Disabling it leaves grains
            below ``min_grain_size`` untouched.
        do_spike_removal : bool
            Run Stage 2 (spike-voxel removal). Disabling it leaves spike
            voxels untouched.
        verbose : bool
            Print a per-slice summary.

        Returns
        -------
        dict
            ``{tslice_key: {'n_small_merged': int, 'n_spikes_removed': int}}``
        """
        from upxo.gsdataops.gid_ops import find_small_fids
        from upxo.netops.neighops import adjacency_from_labels
        from upxo.pxtal.twinned_simple_3d.cleaning_3d import StructureCleaner3D

        if tslice_keys is None:
            tslice_keys = list(pxt.m)

        missing = [t for t in tslice_keys
                   if not hasattr(pxt.gs[t], 'lgi') or pxt.gs[t].lgi is None]
        if missing:
            raise RuntimeError(
                f'calculate_lfi() must be run before clean_temporal_slices() '
                f'-- missing LFI for slice(s): {missing}.')

        spike_cleaner = StructureCleaner3D()
        summary = {}

        for tslice_key in tslice_keys:
            gstslice = pxt.gs[tslice_key]
            s = gstslice.s
            lgi = gstslice.lgi.copy()

            # Each label's representative state, captured once before any
            # merging in THIS pass. merge_label_into always keeps the
            # *surviving* (dominant) label's ID unchanged -- only the
            # absorbed label disappears -- so this stays valid throughout
            # Stage 1; it is never looked up for a label that no longer
            # exists. Read from .s (kept in sync with .lgi by the
            # previous pass, if any) rather than assuming that sync a
            # priori.
            flat_lgi0 = lgi.ravel()
            flat_s0 = s.ravel()
            _, first_idx0 = np.unique(flat_lgi0, return_index=True)
            s_by_label = {int(flat_lgi0[i]): int(flat_s0[i]) for i in first_idx0}

            # Stage 1 -- small-grain removal. Deliberately does NOT re-run
            # cc3d after each merge (that would reassign every label's ID
            # from scratch and break the "dominant keeps its ID" identity
            # this loop and s_by_label both rely on) -- areas/adjacency
            # are just recomputed from the current (already-merged) lgi.
            #
            # All of a pass's merge decisions are applied in ONE vectorised
            # relabelling (a lookup-table pass, `lgi = lut[lgi]`) instead of
            # one merge_label_into() call per small grain: that call is
            # `out[out == other_id] = parent_id`, an O(N) full-array scan,
            # so doing it once per small grain turned a single pass into
            # O(n_small_grains x N) -- on a large domain with many small
            # grains (typical right after MC initialisation) this dwarfed
            # every other cost in this method. A single LUT application is
            # O(N) regardless of how many grains merge in that pass.
            n_small_merged = 0
            if do_merge_small:
                for _ in range(50):  # merging only grows grains -> guaranteed to converge
                    small_gids = find_small_fids(lgi, min_grain_size)
                    if small_gids.size == 0:
                        break
                    areas = np.bincount(lgi.ravel())
                    neigh = adjacency_from_labels(lgi, connectivity=6)
                    merges: dict = {}  # small gid -> its chosen dominant neighbour
                    for gid in small_gids.tolist():
                        gid = int(gid)
                        nbrs = neigh.get(gid)
                        if not nbrs:
                            continue  # isolated small grain with no neighbour to absorb into
                        dominant = max(nbrs, key=lambda g: areas[g] if g < len(areas) else 0)
                        if dominant == gid:
                            continue
                        merges[gid] = dominant
                    if not merges:
                        break

                    # Resolve merge chains decided within this SAME pass (small
                    # grain A's dominant is B, which was ALSO decided to merge
                    # into C this pass) so every absorbed grain maps straight
                    # to its final surviving label -- a `seen` guard makes this
                    # robust even against a mutual A<->B pick (both grains'
                    # only neighbour is each other), which terminates instead
                    # of looping forever and just resolves to one arbitrary
                    # (but well-defined) survivor.
                    max_label = int(lgi.max())
                    lut = np.arange(max_label + 1, dtype=lgi.dtype)
                    for gid, dominant in merges.items():
                        root, seen = dominant, {gid}
                        while root in merges and root not in seen:
                            seen.add(root)
                            root = merges[root]
                        lut[gid] = root
                    lgi = lut[lgi]
                    n_small_merged += len(merges)

            # Stage 2 -- spike removal (label-level, matching
            # StructureCleaner3D's own design intent: "same value" must
            # mean "same grain", not "same Potts state" -- doing this on
            # raw state values instead would under-detect spikes whenever
            # a spike's neighbour happens to share its state by chance
            # without being part of the same grain).
            if do_spike_removal:
                lgi, spike_count = spike_cleaner._fast_clean_spikes(lgi)
            else:
                spike_count = 0

            # Persist the cleaned LFI -- the new source of truth.
            gstslice.lgi = lgi

            # Convert the final labelling back to state values, keeping
            # .s in sync for any code that still reads it directly.
            final_labels = np.unique(lgi[lgi > 0])
            lut = np.zeros(int(lgi.max()) + 1, dtype=s.dtype)
            for lbl in final_labels.tolist():
                lut[int(lbl)] = s_by_label.get(int(lbl), 0)
            gstslice.s = lut[lgi]

            summary[tslice_key] = {
                'n_small_merged': n_small_merged,
                'n_spikes_removed': int(spike_count),
            }
            if verbose:
                print(f"  t={tslice_key}: merged {n_small_merged} small grain(s), "
                      f"removed {spike_count} spike voxel(s)")

        return summary

    @classmethod
    def clean_temporal_slices_recursive(
            cls,
            pxt,
            min_grain_size: int = 4,
            tslice_keys: Optional[list] = None,
            do_merge_small: bool = True,
            do_spike_removal: bool = True,
            n_passes: int = 1,
            verbose: bool = True,
    ) -> Tuple[dict, int]:
        """
        Repeats :meth:`clean_temporal_slices` with the same settings up
        to ``n_passes`` times, stopping early the moment a pass finds
        nothing left to merge or reassign -- a single pass can leave a
        grain that only drops below ``min_grain_size`` after that pass's
        own spike removal, which a subsequent pass then catches.

        Parameters
        ----------
        n_passes : int
            Maximum number of cleaning passes. 1 reproduces a single
            :meth:`clean_temporal_slices` call exactly.
        Other parameters : see :meth:`clean_temporal_slices`.

        Returns
        -------
        (cumulative, n_passes_run) : (dict, int)
            cumulative : ``{tslice_key: {'n_small_merged': int,
            'n_spikes_removed': int}}`` -- summed across every pass run.
            n_passes_run : how many passes actually ran (<= n_passes;
            less if convergence was reached early).
        """
        if tslice_keys is None:
            tslice_keys = list(pxt.m)

        cumulative = {t: {'n_small_merged': 0, 'n_spikes_removed': 0} for t in tslice_keys}
        n_passes_run = 0
        for it in range(1, n_passes + 1):
            if n_passes > 1 and verbose:
                print(f"-- Clean pass {it}/{n_passes} --")
            pass_summary = cls.clean_temporal_slices(
                pxt, min_grain_size=min_grain_size, tslice_keys=tslice_keys,
                do_merge_small=do_merge_small, do_spike_removal=do_spike_removal, verbose=verbose)
            n_passes_run = it
            changed = False
            for t, v in pass_summary.items():
                cumulative[t]['n_small_merged'] += v['n_small_merged']
                cumulative[t]['n_spikes_removed'] += v['n_spikes_removed']
                if v['n_small_merged'] or v['n_spikes_removed']:
                    changed = True
            if n_passes > 1 and not changed:
                if verbose:
                    print(f"Converged after {it} pass(es) -- no further "
                          "merges or spikes found.")
                break
        else:
            if n_passes > 1 and verbose:
                print(f"Reached the {n_passes}-pass limit without "
                      "full convergence -- some defects may remain.")

        return cumulative, n_passes_run

    @classmethod
    def select_temporal_slice(
            cls,
            rank_info: list,
            ebsd_n_parents: Optional[int] = None,
    ):
        """
        Interactive single-select widget for choosing a temporal slice.

        Displays RadioButtons with one option per entry in *rank_info*.
        The default selection is the slice whose grain count is closest
        to *ebsd_n_parents* (if provided), otherwise the first entry.

        Parameters
        ----------
        rank_info : list of dict
            Output of :meth:`rank_temporal_slices_by_n`.
        ebsd_n_parents : int or None
            EBSD pure-parent grain count shown in the header.

        Returns
        -------
        ipywidgets.RadioButtons
            Read ``.value`` after running the cell to get the chosen
            ``tslice_key``.
        """
        import ipywidgets as widgets
        from IPython.display import display

        if ebsd_n_parents and rank_info:
            default_idx = min(
                range(len(rank_info)),
                key=lambda i: abs(rank_info[i]['n_grains_2d_avg'] - ebsd_n_parents),
            )
        else:
            default_idx = 0

        options = []
        for r in rank_info:
            label = (f"t={r['tslice_key']:>4d}  |  "
                     f"n_grains_2d_avg={r['n_grains_2d_avg']:>6.1f}")
            if r['ratio'] is not None:
                label += f"  |  ratio={r['ratio']:.2f}x"
            options.append((label, r['tslice_key']))

        default_val = rank_info[default_idx]['tslice_key'] if rank_info else None

        n_slices_used = rank_info[0]['n_slices_used'] if rank_info else '?'
        header_html = '<b style="font-size:13px">Temporal slice selector</b>'
        if ebsd_n_parents:
            header_html += (
                f'<br><span style="color:#555">Comparison: mean 2D grain count '
                f'across {n_slices_used} slices vs EBSD reference '
                f'<b>{ebsd_n_parents}</b> pure parent grains '
                f'(closest match pre-selected)</span>'
            )
        header = widgets.HTML(value=header_html)
        radio = widgets.RadioButtons(
            options=options,
            value=default_val,
            description='',
            layout=widgets.Layout(width='520px'),
            style={'description_width': '0px'},
        )
        display(widgets.VBox([header, radio]))
        return radio

    def __repr__(self):
        s = self.lgi.shape
        hosted = len(self.host_grain_ids) if self.host_grain_ids is not None else '?'
        return (
            f"TwinnedSimple3DBase("
            f"domain={s[0]}x{s[1]}x{s[2]}, "
            f"n_grains={self.n_grains}, "
            f"hosts={hosted}, "
            f"voxel_size={self.voxel_size} {self.units})"
        )


def compute_dropped_features(original: 'TwinnedSimple3DBase', current: 'TwinnedSimple3DBase') -> Dict:
    """
    Grain-loss sanity check between an untouched ``original`` structure
    and a ``current`` one derived from it (e.g. by resampling/rescaling)
    -- which grains present in ``original`` are absent from ``current``,
    and what fraction of the original structure's volume they accounted
    for.

    Parameters
    ----------
    original, current : TwinnedSimple3DBase
        ``current.grain_ids`` is compared against ``original.grain_ids``
        by set difference; ``original.mprop['volnv']`` is computed (via
        ``char_morphology``) if not already present, to weight dropped
        grains by voxel count rather than by plain grain count.

    Returns
    -------
    dict
        ``{'dropped_ids': list of int, 'dropped_vol': int,
        'total_vol': int, 'pct': float}`` -- ``dropped_ids`` sorted
        ascending; ``pct`` is the dropped volume as a percentage of
        ``original``'s total grain volume (0.0 if nothing was dropped).
    """
    dropped_ids = sorted(set(original.grain_ids) - set(current.grain_ids))
    if not dropped_ids:
        return {'dropped_ids': [], 'dropped_vol': 0, 'total_vol': 0, 'pct': 0.0}

    original.char_morphology(volnv=True, eqdia=False, sanv=False, force_compute=True)
    vols = original.mprop.get('volnv', {})
    dropped_vol = sum(vols.get(gid, 0) for gid in dropped_ids)
    total_vol = sum(vols.values()) or 1
    pct = 100.0 * dropped_vol / total_vol
    return {'dropped_ids': dropped_ids, 'dropped_vol': dropped_vol, 'total_vol': total_vol, 'pct': pct}
