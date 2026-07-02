"""
base_3d.py
==========
Foundation class for the twinned simple 3D grain structure pipeline.
"""

import numpy as np
from typing import Optional, Dict


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
        'host_grain_ids', 'non_host_grain_ids',
        'target_hosting_fraction', 'actual_hosting_fraction',
        'host_fraction_2d_to_3d_scale_factor',
        'host_ranking_volume_weight',
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
        self.target_hosting_fraction: Optional[float] = None
        self.actual_hosting_fraction: Optional[float] = None
        self.host_fraction_2d_to_3d_scale_factor: float = 1.0
        self.host_ranking_volume_weight: float = 0.5
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
        vs = float(voxel_size) if voxel_size is not None else float(pxt.vox_size)
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

    def domain_shape(self):
        """Return (nx, ny, nz) voxel dimensions of the domain."""
        return self.lgi.shape

    def physical_size(self):
        """Return physical domain size (Lx, Ly, Lz) in ``self.units``."""
        return tuple(s * self.voxel_size for s in self.lgi.shape)

    @classmethod
    def rank_temporal_slices_by_n(
            cls,
            pxt,
            start: int = 0,
            step: int = 5,
            ebsd_n_parents: Optional[int] = None,
            n_comparison_slices: int = 5,
            comparison_axes: Optional[list] = None,
    ) -> list:
        """
        Rank MC temporal slices by their 2D-equivalent grain count.

        The 3D total grain count is not directly comparable to a 2D EBSD
        cross-section count.  Instead, this method extracts
        ``n_comparison_slices`` evenly-spaced 2D slices along each axis
        in ``comparison_axes``, counts grains per slice, and reports the
        mean across all sampled slices as ``n_grains_2d_avg``.  This is
        the correct quantity to compare against the EBSD parent grain count.

        Uses ``cc3d.connected_components`` on the raw spin-state array
        (6-connectivity) so grains are correctly identified even when
        voxels of the same spin are not face-connected.

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

        Returns
        -------
        list of dict, each with keys
            ``tslice_key``, ``n_grains_2d_avg``, ``n_slices_used``,
            ``ratio`` (None if no reference).
        """
        import cc3d
        from upxo.gsdataops.grid_ops import section_from_3d

        if comparison_axes is None:
            comparison_axes = ['x', 'y', 'z']

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axes_int = [axis_map[a.lower()] for a in comparison_axes
                    if a.lower() in axis_map]

        temporal_indices = np.arange(start, len(pxt.m), step)
        rows = []

        for i in temporal_indices:
            lgi_3d = cc3d.connected_components(
                pxt.gs[i].s, connectivity=6)

            slice_counts = []
            for ax in axes_int:
                domain_size = lgi_3d.shape[ax]
                slice_positions = np.linspace(
                    0, domain_size - 1, n_comparison_slices, dtype=int)
                for pos in slice_positions:
                    lgi_2d = section_from_3d(lgi_3d, axis=ax, location=int(pos))
                    n_slice = int(np.unique(lgi_2d[lgi_2d > 0]).size)
                    slice_counts.append(n_slice)

            n_avg = float(np.mean(slice_counts)) if slice_counts else 0.0
            ratio = n_avg / ebsd_n_parents if ebsd_n_parents else None
            rows.append({
                'tslice_key': int(i),
                'n_grains_2d_avg': n_avg,
                'n_slices_used': len(slice_counts),
                'ratio': ratio,
            })

        return rows

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
