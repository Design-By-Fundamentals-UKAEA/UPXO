"""
cleaning_3d.py
==============
Post-twin voxel topology cleaning for the twinned simple 3D pipeline.

Both cleaning stages use fully vectorised local implementations that are
O(1) in grain count (single array pass), replacing the per-grain loop
approach in the generic ``gsdataops.grid_ops`` versions which scale
poorly on large domains.
"""

import numpy as np
from typing import Optional, Dict


class StructureCleaner3D:
    """
    Post-twin voxel topology cleaner for the twinned simple 3D pipeline.

    Stage 1 -- Vertex spike removal (one global atomic pass, vectorised).
    Stage 2 -- Lobe splitting via a single cc3d pass on the full array.

    If defects persist and ``upscale_fallback`` is True the structure
    is doubled in each spatial dimension (~8x voxel count) and both
    stages are re-applied.
    """

    __slots__ = (
        'upscale_fallback', 'split_jitter_deg', 'min_clean_voxels',
        'do_spike_removal', 'do_lobe_split',
        'lgi_clean', 'all_quats_clean', 'twin_role_clean', 'twin_parent_of_clean',
        'split_events', 'spike_count', 'n_splits',
        'remaining_spikes', 'remaining_multi', 'upscale_applied', '_rng',
    )

    def __init__(
            self,
            upscale_fallback: bool = False,
            split_jitter_deg: float = 0.0,
            rng_seed: Optional[int] = None,
            min_clean_voxels: int = 0,
            do_spike_removal: bool = True,
            do_lobe_split: bool = True,
    ):
        self.upscale_fallback  = upscale_fallback
        self.split_jitter_deg  = split_jitter_deg
        # Independently disable either cleaning stage -- forward-looking
        # for when the stages themselves gain alternative implementations
        # (see clean()'s per-stage gating below). Both default on,
        # matching the previous unconditional behaviour.
        self.do_spike_removal: bool = bool(do_spike_removal)
        self.do_lobe_split: bool = bool(do_lobe_split)
        # Grains smaller than this are skipped for lobe-splitting topology
        # check.  Larger values speed up S30 at the expense of leaving
        # very small grains unchecked for topological defects.  A value of
        # 0 (default) checks all grains regardless of size.
        self.min_clean_voxels: int = int(min_clean_voxels)
        self.lgi_clean: Optional[np.ndarray] = None
        self.all_quats_clean: Optional[Dict] = None
        self.twin_role_clean: Optional[Dict] = None
        self.twin_parent_of_clean: Optional[Dict] = None
        self.split_events: Dict = {}
        self.spike_count: int = 0
        self.n_splits: int = 0
        self.remaining_spikes: int = 0
        self.remaining_multi: int = 0
        self.upscale_applied: bool = False
        self._rng = np.random.default_rng(rng_seed)

    # ── fast private helpers ──────────────────────────────────────────────────

    def _count_face_same_grain(self, lgi: np.ndarray) -> np.ndarray:
        """
        For every voxel, count how many of its 6 face-connected neighbours
        carry the same grain label.  Single vectorised pass — O(1) in grain
        count.

        Returns
        -------
        ndarray, dtype int32, same shape as *lgi*.
        """
        c = np.zeros_like(lgi, dtype=np.int32)
        for ax in range(3):
            sl1 = [slice(None)] * 3
            sl2 = [slice(None)] * 3
            sl1[ax] = slice(1, None)
            sl2[ax] = slice(None, -1)
            eq = (lgi[tuple(sl1)] == lgi[tuple(sl2)]).astype(np.int32)
            c[tuple(sl1)] += eq
            c[tuple(sl2)] += eq
        return c

    # A voxel with same_count <= this many face-connected same-grain
    # neighbours is treated as a spike. Was same_count == 0 (i.e. <= 0)
    # until an empirical test against real MC-grown structures showed that
    # criterion can only ever match fully-isolated singleton voxels, which
    # are always already consumed by small-grain merging before spike
    # removal runs -- so it never fired. Loosening to <= 1 catches genuine
    # thin-necked protrusions (same_count == 1, the neck voxel) without
    # over-flagging: a same_count == 2 comparison showed those extra
    # voxels have roughly double the local same-grain mass and typically
    # touch 2-4 distinct grains -- normal rough/faceted boundary or
    # triple-junction geometry, not defects -- so <= 2 was rejected.
    _SPIKE_SAME_COUNT_THRESHOLD = 1

    def _fast_clean_spikes(self, lgi: np.ndarray):
        """
        Remove spike voxels (<= _SPIKE_SAME_COUNT_THRESHOLD face-connected
        same-grain neighbours) in one global atomic pass.  Detection is
        fully vectorised; the repair loop only iterates over detected
        spikes, which are typically very few.

        A grain's very last remaining voxel is never reassigned away, even
        if it's spike-classified -- without this, a thin sliver or
        single-voxel grain composed entirely of spike voxels could be
        erased completely in one pass, leaving its grain ID referenced by
        nothing in `lgi` while every other per-grain array (orientations,
        twin role, twin parent) still carries an entry for it.
        `grain_counts` is seeded from `lgi` as it stood before this pass
        and decremented as this same pass's own reassignments happen, so
        two spike voxels belonging to a 2-voxel grain still correctly
        protect only the second one, not both.

        Parameters
        ----------
        lgi : ndarray (modified in place)

        Returns
        -------
        lgi, spike_count
        """
        face_offsets = [
            (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        sx, sy, sz = lgi.shape

        same_count = self._count_face_same_grain(lgi)
        spike_coords = np.argwhere(
            (lgi > 0) & (same_count <= self._SPIKE_SAME_COUNT_THRESHOLD))
        spike_count = 0
        grain_counts = np.bincount(lgi.ravel())

        for ix, iy, iz in spike_coords.tolist():
            gid = int(lgi[ix, iy, iz])
            if grain_counts[gid] <= 1:
                continue  # last voxel of this grain -- never reassign it away
            nb_counts: Dict[int, int] = {}
            for dx, dy, dz in face_offsets:
                nx_, ny_, nz_ = ix+dx, iy+dy, iz+dz
                if 0 <= nx_ < sx and 0 <= ny_ < sy and 0 <= nz_ < sz:
                    ng = int(lgi[nx_, ny_, nz_])
                    if ng > 0 and ng != gid:
                        nb_counts[ng] = nb_counts.get(ng, 0) + 1
            if nb_counts:
                lgi[ix, iy, iz] = max(nb_counts, key=nb_counts.get)
                spike_count += 1
                grain_counts[gid] -= 1

        return lgi, spike_count

    def _fast_split_lobes(
            self,
            lgi: np.ndarray,
            all_quats: Dict,
            twin_role: Dict,
            twin_parent_of: Dict,
            next_gid: int,
            skip_gids: set = None,
    ):
        """
        Split edge/corner-only connected lobes via a **single** cc3d pass on
        the full array, rather than one ``ndlabel`` call per grain.

        Returns
        -------
        lgi, all_quats, twin_role, twin_parent_of, split_events, next_gid
        """
        import cc3d
        from collections import defaultdict

        split_events: Dict = {}

        # Single connected-components pass on the full label array
        cc = cc3d.connected_components(lgi.astype(np.uint32), connectivity=6)

        # Vectorised: map each cc component to its original grain ID
        cc_flat  = cc.ravel()
        lgi_flat = lgi.ravel()
        valid    = cc_flat > 0

        cc_valid  = cc_flat[valid]
        lgi_valid = lgi_flat[valid]

        sort_idx  = np.argsort(cc_valid, kind='stable')
        cc_sorted = cc_valid[sort_idx]
        lgi_sorted= lgi_valid[sort_idx]
        first_occ = np.concatenate(
            [[0], np.where(np.diff(cc_sorted) != 0)[0] + 1])

        cc_ids    = cc_sorted[first_occ]
        grain_ids = lgi_sorted[first_occ]

        # Build grain → [cc component ids] mapping
        grain_to_ccs: Dict = defaultdict(list)
        for cc_id, gid in zip(cc_ids.tolist(), grain_ids.tolist()):
            grain_to_ccs[int(gid)].append(int(cc_id))

        _skip = skip_gids or set()
        for gid, ccs in grain_to_ccs.items():
            if len(ccs) <= 1:
                continue
            if gid in _skip:
                continue   # small grain — skip lobe check

            # Sort components by voxel count; largest keeps original ID
            comp_sizes = sorted(
                [(cc_id, int(np.sum(cc == cc_id))) for cc_id in ccs],
                key=lambda x: x[1], reverse=True,
            )

            parent_q    = all_quats.get(gid)
            parent_role = twin_role.get(gid, 'non_host')

            for rank, (comp_id, comp_size) in enumerate(comp_sizes):
                if rank == 0:
                    continue

                lgi[cc == comp_id] = next_gid

                q_new = parent_q.copy() if parent_q is not None \
                    else np.array([1., 0., 0., 0.])
                jitter_applied = 0.0

                if self.split_jitter_deg > 0.0 and parent_q is not None:
                    try:
                        from upxo.xtalphy.crystal_orientation import \
                            apply_orientation_jitter
                        q_new = apply_orientation_jitter(
                            parent_q, self.split_jitter_deg, self._rng)
                        cos_h = float(np.clip(
                            np.abs(np.dot(parent_q, q_new)), 0., 1.))
                        jitter_applied = float(
                            np.degrees(2. * np.arccos(cos_h)))
                    except Exception:
                        pass

                all_quats[next_gid]      = q_new
                twin_role[next_gid]      = parent_role
                twin_parent_of[next_gid] = gid
                split_events[next_gid]   = {
                    'parent_gid':         gid,
                    'component_size_vox': comp_size,
                    'jitter_applied_deg': jitter_applied,
                }
                next_gid += 1

        return lgi, all_quats, twin_role, twin_parent_of, split_events, next_gid

    def _count_defects(self, lgi: np.ndarray) -> Dict:
        """Count remaining spikes and multi-component grains."""
        import cc3d
        same_count = self._count_face_same_grain(lgi)
        spikes = int(np.sum((lgi > 0) & (same_count <= self._SPIKE_SAME_COUNT_THRESHOLD)))

        cc = cc3d.connected_components(lgi.astype(np.uint32), connectivity=6)
        cc_flat  = cc.ravel()
        lgi_flat = lgi.ravel()
        valid    = cc_flat > 0
        cc_valid = cc_flat[valid]
        lgi_valid= lgi_flat[valid]
        sort_idx = np.argsort(cc_valid, kind='stable')
        lgi_sorted = lgi_valid[sort_idx]
        cc_sorted  = cc_valid[sort_idx]
        first_occ  = np.concatenate(
            [[0], np.where(np.diff(cc_sorted) != 0)[0] + 1])
        grain_ids  = lgi_sorted[first_occ]
        from collections import Counter
        multi = sum(1 for cnt in Counter(grain_ids.tolist()).values() if cnt > 1)

        return {'spikes': spikes, 'multi': multi}

    # ── public API ────────────────────────────────────────────────────────────

    def clean(
            self,
            lgi: np.ndarray,
            all_quats: Dict,
            twin_role: Dict,
            twin_parent_of: Dict,
    ):
        """
        Run Stage 1 (spike removal) + Stage 2 (lobe splitting) on the
        post-twin grain structure using fast vectorised local methods.

        Grains with fewer than ``self.min_clean_voxels`` voxels are skipped
        in Stage 2.  Set via the ``min_clean_voxels`` constructor argument.
        """
        import time as _t
        _t0 = _t.perf_counter()
        nx, ny, nz = lgi.shape
        n_grains   = int((lgi > 0).sum())
        print(f'[S30] StructureCleaner3D  domain={nx}x{ny}x{nz}  '
              f'active_vox={n_grains:,}  min_clean_vox={self.min_clean_voxels}')

        lgi_c  = lgi.copy()
        q_c    = dict(all_quats)
        role_c = dict(twin_role)
        par_c  = dict(twin_parent_of)

        # Stage 1 — spikes
        if self.do_spike_removal:
            print('  [S30] Step 1/2  Spike removal...', end='', flush=True)
            _t1 = _t.perf_counter()
            lgi_c, self.spike_count = self._fast_clean_spikes(lgi_c)
            print(f'  done  {self.spike_count} spikes  ({_t.perf_counter()-_t1:.1f}s)')
        else:
            print('  [S30] Step 1/2  Spike removal...  SKIPPED (disabled)')
            self.spike_count = 0

        # Stage 2 — lobes (skip grains below min_clean_voxels)
        if self.do_lobe_split:
            print('  [S30] Step 2/2  Lobe splitting...', end='', flush=True)
            _t2 = _t.perf_counter()
            all_gids  = sorted(int(g) for g in np.unique(lgi_c) if g > 0)
            next_gid  = max(all_gids) + 1
            if self.min_clean_voxels > 0:
                # Build size map once with bincount, skip small grains in the lobe pass
                _counts = np.bincount(lgi_c.ravel(), minlength=next_gid)
                _skip   = {g for g in all_gids if _counts[g] < self.min_clean_voxels}
                n_skip  = len(_skip)
            else:
                _skip  = set()
                n_skip = 0
            lgi_c, q_c, role_c, par_c, self.split_events, next_gid = \
                self._fast_split_lobes(lgi_c, q_c, role_c, par_c, next_gid,
                                       skip_gids=_skip)
            self.n_splits = len(self.split_events)
            print(f'  done  {self.n_splits} splits  skipped {n_skip} small grains  '
                  f'({_t.perf_counter()-_t2:.1f}s)')
        else:
            print('  [S30] Step 2/2  Lobe splitting...  SKIPPED (disabled)')
            self.split_events = {}
            self.n_splits = 0

        rem = self._count_defects(lgi_c)
        self.remaining_spikes = rem['spikes']
        self.remaining_multi  = rem['multi']

        # Upscale fallback -- only triggered by, and only re-applies, the
        # stage(s) actually enabled above; a disabled stage's leftover
        # defects are expected (the user asked to skip it), not a failure
        # to retry with a bigger domain.
        trigger_spike = self.do_spike_removal and self.remaining_spikes > 0
        trigger_multi = self.do_lobe_split and self.remaining_multi > 0
        if (trigger_spike or trigger_multi) and self.upscale_fallback:
            print('StructureCleaner3D: upscale fallback (2x per axis, ~8x voxels).')
            print('  WARNING: FEM element count will increase ~8x.')
            lgi_c = self.apply_upscale(lgi_c, factor=2)
            self.upscale_applied = True
            if self.do_spike_removal:
                lgi_c, sp2 = self._fast_clean_spikes(lgi_c)
                self.spike_count += sp2
            if self.do_lobe_split:
                all_gids2 = sorted(int(g) for g in np.unique(lgi_c) if g > 0)
                next_gid2 = max(all_gids2) + 1
                lgi_c, q_c, role_c, par_c, ev2, _ = self._fast_split_lobes(
                    lgi_c, q_c, role_c, par_c, next_gid2)
                self.split_events.update(ev2)
                self.n_splits = len(self.split_events)
            rem2 = self._count_defects(lgi_c)
            self.remaining_spikes = rem2['spikes']
            self.remaining_multi  = rem2['multi']

        self.lgi_clean          = lgi_c
        self.all_quats_clean    = q_c
        self.twin_role_clean    = role_c
        self.twin_parent_of_clean = par_c

        if self.remaining_spikes == 0 and self.remaining_multi == 0:
            print(f'StructureCleaner3D: PASS '
                  f'(spikes removed: {self.spike_count}, '
                  f'lobes split: {self.n_splits})')
        else:
            print(f'StructureCleaner3D: WARNING — '
                  f'{self.remaining_spikes} spikes, '
                  f'{self.remaining_multi} multi-component grains remain.')

    def check_remaining_defects(self) -> Dict:
        return {
            'remaining_spikes': self.remaining_spikes,
            'remaining_multi':  self.remaining_multi,
            'upscale_applied':  self.upscale_applied,
        }

    def apply_upscale(self, lgi: np.ndarray, factor: int = 2) -> np.ndarray:
        result = lgi
        for ax in range(3):
            result = np.repeat(result, factor, axis=ax)
        return result

    def split_events_report(self) -> str:
        if not self.split_events:
            return 'No split events.'
        lines = [f'Split events: {len(self.split_events)}']
        for new_gid, info in self.split_events.items():
            j = (f', jitter {info["jitter_applied_deg"]:.2f} deg'
                 if info['jitter_applied_deg'] > 0 else ' (same orientation)')
            lines.append(
                f'  {new_gid} <- parent {info["parent_gid"]} '
                f'| {info["component_size_vox"]} vox{j}')
        return '\n'.join(lines)

    def jitter_report(self) -> str:
        """Report focused specifically on the orientation jitter applied
        to split-off lobes -- split_events_report() covers every split
        event (parent, voxel size, jitter all together in one line); this
        isolates just the jitter part with summary statistics (min/max/
        mean applied jitter), since that's the piece someone tracking
        orientation-noise/meshing concerns wants on its own, not mixed in
        with lobe size/parent bookkeeping."""
        if not self.split_events:
            return 'No split events -- no orientation jitter to report.'

        if self.split_jitter_deg <= 0.0:
            return (f'Split Jitter was 0 deg for this cleaning pass -- all '
                     f'{len(self.split_events)} split-off lobe(s) retained '
                     f'their parent grain\'s orientation exactly (no jitter '
                     f'applied).')

        jitters = [info['jitter_applied_deg'] for info in self.split_events.values()]
        nonzero = [j for j in jitters if j > 0.0]
        n_zero = len(jitters) - len(nonzero)

        lines = [
            f'Orientation jitter report (Split Jitter setting: '
            f'{self.split_jitter_deg:.2f} deg)',
            f'  Split-off lobes: {len(jitters)}',
            f'  Jitter applied (> 0 deg): {len(nonzero)}',
            f'  No jitter applied (parent had no orientation, or jitter '
            f'failed): {n_zero}',
        ]
        if nonzero:
            lines.append(
                f'  Applied jitter (deg): min={min(nonzero):.2f}  '
                f'max={max(nonzero):.2f}  mean={sum(nonzero) / len(nonzero):.2f}')
        lines.append('')
        lines.append('Per-lobe detail:')
        for new_gid, info in self.split_events.items():
            j = info['jitter_applied_deg']
            note = f'{j:.2f} deg' if j > 0 else 'none (same orientation as parent)'
            lines.append(f'  {new_gid} <- parent {info["parent_gid"]}: {note}')
        return '\n'.join(lines)
