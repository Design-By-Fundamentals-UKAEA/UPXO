"""
orientation_3d.py
=================
Conflict-free crystallographic orientation assignment for the twinned
simple 3D pipeline.

Three progressive orientation assignment modes are available, selectable
via ``orientation_assignment_mode``:

Level 0  ``'conflict_free'``
    Orientations sampled independently from the EBSD pool with a
    conflict-free constraint (no two adjacent grains receive the same
    quaternion).  Texture correlations between neighbours are NOT
    reproduced.  Fastest; used as the baseline.

Level 1  ``'paired_pool'``
    Orientations sampled from the set of ACTUALLY-OBSERVED adjacent
    pure-parent orientation pairs in the EBSD.  When grain G is being
    assigned and a neighbour N already has orientation q_N, the algorithm
    searches the EBSD pair pool for pairs where one member is close to
    q_N and proposes the other member as q_G.  This directly reproduces
    the neighbour-to-neighbour orientation correlations (texture) from
    the EBSD, fixing the systematic MDF right-shift of Level 0.

Level 2  ``'mdf_targeted'``   [NOT YET IMPLEMENTED -- stub only]
    Misorientation-targeted assignment: explicitly samples a target
    misorientation angle from the EBSD parent-state MDF, then finds an
    orientation that achieves approximately that angle with the assigned
    neighbours.

Level 3  ``'mrf'``            [NOT YET IMPLEMENTED -- stub only]
    Full Markov Random Field treatment: grain orientation depends jointly
    on ALL assigned neighbours, maximising consistency with the EBSD
    joint orientation distribution.
"""

import math
import numpy as np
from typing import Optional, Dict, Tuple

# Valid mode identifiers (ordered by physical fidelity)
_VALID_ORIENT_MODES = ('conflict_free', 'paired_pool', 'mdf_targeted', 'mrf')


class OrientationAssigner3D:
    """
    Assigns crystallographic orientations to all SGC grains.

    The assignment mode is controlled by ``orientation_assignment_mode``
    (see module docstring for a description of each level).
    """

    __slots__ = (
        'base', 'neigh_graph',
        'parent_pool', 'full_pool', 'fallback_quats',
        'all_grain_orientations', 'n_fallback_host', 'n_fallback_nonhost',
        'n_conflicts',
        # Adjacency model (Level 1+)
        'orientation_assignment_mode',
        'pair_similarity_deg',
        '_paired_pool_host',     # [(q_A, q_B)] from EBSD adjacent pure-parent pairs
        '_paired_pool_full',     # [(q_A, q_B)] from ALL EBSD adjacent grain pairs
        '_n_paired_used_host',
        '_n_paired_used_nonhost',
        '_rng',
    )

    def __init__(
            self,
            base,
            orientation_assignment_mode: str = 'conflict_free',
            pair_similarity_deg: float = 10.0,
            rng_seed: Optional[int] = None,
    ):
        if orientation_assignment_mode not in _VALID_ORIENT_MODES:
            raise ValueError(
                f'orientation_assignment_mode must be one of '
                f'{_VALID_ORIENT_MODES}, got "{orientation_assignment_mode}"')
        self.base = base
        self.orientation_assignment_mode = orientation_assignment_mode
        self.pair_similarity_deg = float(pair_similarity_deg)
        self.neigh_graph: Optional[Dict] = None
        self.parent_pool: Optional[np.ndarray] = None
        self.full_pool: Optional[np.ndarray] = None
        self.fallback_quats: Optional[np.ndarray] = None
        self.all_grain_orientations: Dict = {}
        self.n_fallback_host: int = 0
        self.n_fallback_nonhost: int = 0
        self.n_conflicts: int = 0
        self._paired_pool_host: Optional[list] = None
        self._paired_pool_full: Optional[list] = None
        self._n_paired_used_host: int = 0
        self._n_paired_used_nonhost: int = 0
        self._rng = np.random.default_rng(rng_seed)

        print(f'OrientationAssigner3D: mode = "{orientation_assignment_mode}"')
        if orientation_assignment_mode in ('mdf_targeted', 'mrf'):
            print(f'  WARNING: "{orientation_assignment_mode}" is not yet '
                  f'implemented -- falling back to "paired_pool".')

    # ── neighbour graph ───────────────────────────────────────────────────

    def build_neighbour_graph(self, connectivity: int = 6):
        """Build the face-connected neighbour graph for the SGC grain structure."""
        from upxo.gsdataops.gid_ops import find_neighs3d
        raw = find_neighs3d(self.base.lgi.astype(np.int32), conn=connectivity)
        self.neigh_graph = {
            int(gid): set(int(n) for n in neighs if n > 0 and n != gid)
            for gid, neighs in raw.items() if gid > 0
        }
        n_edges = sum(len(v) for v in self.neigh_graph.values()) // 2
        print(f'OrientationAssigner3D: neighbour graph '
              f'({connectivity}-connected): '
              f'{len(self.neigh_graph)} grains, {n_edges} edges')

    # ── EBSD orientation pools ─────────────────────────────────────────────

    def build_ebsd_pools(
            self,
            ebsd_lfi: np.ndarray,
            ebsd_quat: np.ndarray,
            parent_info: Dict,
            csl_label: str,
            n_fallback: int = 500,
    ):
        """Build parent-only and full-EBSD quaternion pools."""
        from upxo.xtalphy.crystal_orientation import grain_avg_quats
        from upxo.xtalphy.texops import tops

        ebsd_gids, ebsd_q = grain_avg_quats(ebsd_lfi, ebsd_quat)
        gid2q = {int(g): ebsd_q[i] for i, g in enumerate(ebsd_gids)}

        ebsd_parent_ids = parent_info[csl_label]['pure_parents']
        self.parent_pool = np.array(
            [gid2q[g] for g in sorted(ebsd_parent_ids) if g in gid2q])
        self.full_pool = ebsd_q

        self.fallback_quats = tops.synth_fcc_quats(
            N=max(n_fallback, self.base.n_grains))

        print(f'OrientationAssigner3D: EBSD parent pool = {len(self.parent_pool)}, '
              f'full pool = {len(self.full_pool)}, '
              f'fallback = {len(self.fallback_quats)}')

    # ── Level 1: adjacency model ───────────────────────────────────────────

    def build_adjacency_model(
            self,
            rg,
            parent_info: Dict,
            csl_label: str,
    ):
        """
        Level 1 -- build the EBSD orientation adjacency model.

        Extracts every adjacent pure-parent grain pair from the EBSD neighbour
        graph and records their orientation pair ``(q_A, q_B)``.  Also builds
        a full-grain pair pool (all adjacent EBSD grain pairs) for non-host
        orientation assignment.

        Must be called after :meth:`build_ebsd_pools`.

        Parameters
        ----------
        rg : repgen2d
            EBSD analysis object (provides ``lfi_ebsd``, ``quat_ebsd``,
            ``neigh_gid_ebsd``).
        parent_info : dict
            Output of ``rg.identify_parent_grains()``.
        csl_label : str
            CSL label key into *parent_info*.
        """
        from upxo.xtalphy.crystal_orientation import grain_avg_quats

        gids_all, q_all = grain_avg_quats(rg.lfi_ebsd, rg.quat_ebsd)
        gid2q = {int(g): q_all[i] for i, g in enumerate(gids_all)}

        pure_parents = set(int(g) for g in parent_info[csl_label]['pure_parents'])

        # ── Host pair pool: adjacent pure-parent pairs only ────────────────
        host_pairs = []
        for gid_a, neighbours in rg.neigh_gid_ebsd.items():
            a = int(gid_a)
            if a not in pure_parents or a not in gid2q:
                continue
            for gid_b in np.asarray(neighbours).ravel():
                b = int(gid_b)
                if b > a and b in pure_parents and b in gid2q:
                    host_pairs.append((gid2q[a].copy(), gid2q[b].copy()))
        self._paired_pool_host = host_pairs

        # ── Full pair pool: ALL adjacent EBSD grain pairs ─────────────────
        full_pairs = []
        for gid_a, neighbours in rg.neigh_gid_ebsd.items():
            a = int(gid_a)
            if a not in gid2q:
                continue
            for gid_b in np.asarray(neighbours).ravel():
                b = int(gid_b)
                if b > a and b in gid2q:
                    full_pairs.append((gid2q[a].copy(), gid2q[b].copy()))
        self._paired_pool_full = full_pairs

        print(f'OrientationAssigner3D (Level 1 adjacency model):')
        print(f'  Host pair pool (pure-parent pairs) : {len(host_pairs)}')
        print(f'  Full pair pool (all EBSD pairs)    : {len(full_pairs)}')

    # ── private: paired-pool assignment (Level 1) ─────────────────────────

    def _assign_paired_pool(
            self,
            grain_ids,
            neigh_graph: Dict,
            paired_pool: list,
            fallback_pool: np.ndarray,
            all_assigned: Dict,
            max_retries: int = 50,
    ) -> tuple:
        """
        Level-1 paired-pool orientation assignment.

        For each grain G:
        1. Identify already-assigned neighbours.
        2. Take the first assigned neighbour's orientation q_N as anchor.
        3. Search *paired_pool* for pairs where one member has quaternion dot
           product ``|q·q_N| >= cos(pair_similarity_deg)`` with q_N.
        4. Propose the other member of that pair as q_G.
        5. Verify conflict-free constraint (no identical orientation with any
           already-assigned neighbour).
        6. Fall back to uniform pool sampling if no suitable pair is found.

        The quaternion dot product ``|q_A · q_N|`` is used as a fast
        approximate similarity measure (exact would require 24-element cubic
        symmetry exhaustive search, which is Level 2).

        Returns
        -------
        all_assigned : dict (modified in place)
        n_paired_used : int
        n_fallback : int
        """
        from upxo.xtalphy.crystal_orientation import _positive_w

        cos_thresh = math.cos(math.radians(self.pair_similarity_deg))
        n_paired_used = 0
        n_fallback    = 0

        # Pre-build numpy arrays for vectorised similarity search
        if paired_pool:
            _pool_A = np.array([p[0] for p in paired_pool], dtype=np.float64)
            _pool_B = np.array([p[1] for p in paired_pool], dtype=np.float64)
        else:
            _pool_A = _pool_B = np.zeros((0, 4))

        grain_order = self._rng.permutation(list(grain_ids)).tolist()

        for gid in grain_order:
            gid = int(gid)
            used_by_neigh = {
                all_assigned[nb].tobytes()
                for nb in neigh_graph.get(gid, set())
                if nb in all_assigned
            }
            assigned_neighbours = [
                all_assigned[nb]
                for nb in neigh_graph.get(gid, set())
                if nb in all_assigned
            ]

            chosen = None

            # ── Attempt paired-pool search if neighbour is already assigned ─
            if assigned_neighbours and len(_pool_A) > 0:
                q_N = assigned_neighbours[0]   # anchor: first assigned neighbour

                # Similarity of pool members A to q_N
                sim_A = np.abs(_pool_A @ q_N)  # (N_pairs,)
                # Also check B members (pairs are symmetric)
                sim_B = np.abs(_pool_B @ q_N)

                # Sort by decreasing similarity; interleave A→propose B and B→propose A
                order_A = np.argsort(-sim_A)
                order_B = np.argsort(-sim_B)
                tried = 0

                for ia, ib in zip(order_A, order_B):
                    if tried >= max_retries:
                        break
                    # Pair where A ≅ q_N → propose B
                    if sim_A[ia] >= cos_thresh:
                        candidate = _positive_w(_pool_B[ia].copy())
                        if candidate.tobytes() not in used_by_neigh:
                            chosen = candidate
                            n_paired_used += 1
                            break
                        tried += 1
                    # Pair where B ≅ q_N → propose A
                    if sim_B[ib] >= cos_thresh and chosen is None:
                        candidate = _positive_w(_pool_A[ib].copy())
                        if candidate.tobytes() not in used_by_neigh:
                            chosen = candidate
                            n_paired_used += 1
                            break
                        tried += 1

            # ── Fallback: uniform sampling from full pool ──────────────────
            if chosen is None:
                n_fp = len(fallback_pool)
                for _ in range(max_retries):
                    candidate = _positive_w(
                        fallback_pool[int(self._rng.integers(0, n_fp))].copy())
                    if candidate.tobytes() not in used_by_neigh:
                        chosen = candidate
                        break
                if chosen is None:
                    chosen = _positive_w(
                        fallback_pool[int(self._rng.integers(0, n_fp))].copy())
                n_fallback += 1

            all_assigned[gid] = chosen

        return all_assigned, n_paired_used, n_fallback

    # ── public assignment methods ──────────────────────────────────────────

    def assign_host_orientations(self):
        """
        Assign orientations to host grains.

        Dispatches to the active ``orientation_assignment_mode``.
        """
        from upxo.xtalphy.crystal_orientation import assign_orientations_conflict_free

        if self.neigh_graph is None:
            raise RuntimeError('Call build_neighbour_graph() first.')
        if self.parent_pool is None:
            raise RuntimeError('Call build_ebsd_pools() first.')

        mode = self.orientation_assignment_mode

        if mode == 'conflict_free':
            self.all_grain_orientations, self.n_fallback_host = \
                assign_orientations_conflict_free(
                    self.base.host_grain_ids, self.neigh_graph,
                    self.parent_pool, self.all_grain_orientations,
                    max_retries=50, rng=self._rng,
                    fallback_pool=self.fallback_quats)
            print(f'OrientationAssigner3D [L0 conflict_free]: '
                  f'{len(self.base.host_grain_ids)} host orientations '
                  f'(fallback: {self.n_fallback_host})')

        elif mode in ('paired_pool', 'mdf_targeted', 'mrf'):
            # Levels 2 and 3 fall back to Level 1 until implemented
            if mode != 'paired_pool':
                print(f'  INFO: mode "{mode}" not yet implemented; '
                      f'using "paired_pool" (Level 1).')
            if self._paired_pool_host is None:
                raise RuntimeError(
                    'Call build_adjacency_model() before assign_host_orientations() '
                    'when using mode "paired_pool".')
            (self.all_grain_orientations,
             self._n_paired_used_host,
             self.n_fallback_host) = self._assign_paired_pool(
                grain_ids    = self.base.host_grain_ids,
                neigh_graph  = self.neigh_graph,
                paired_pool  = self._paired_pool_host,
                fallback_pool= self.parent_pool,
                all_assigned = self.all_grain_orientations,
                max_retries  = 50,
            )
            print(f'OrientationAssigner3D [L1 paired_pool]: '
                  f'{len(self.base.host_grain_ids)} host orientations  '
                  f'pair-matched: {self._n_paired_used_host}  '
                  f'fallback: {self.n_fallback_host}')

    def assign_nonhost_orientations(self):
        """
        Assign orientations to non-host grains.

        Dispatches to the active ``orientation_assignment_mode``.
        """
        from upxo.xtalphy.crystal_orientation import assign_orientations_conflict_free

        if self.neigh_graph is None:
            raise RuntimeError('Call build_neighbour_graph() first.')
        if self.full_pool is None:
            raise RuntimeError('Call build_ebsd_pools() first.')

        mode = self.orientation_assignment_mode

        if mode == 'conflict_free':
            self.all_grain_orientations, self.n_fallback_nonhost = \
                assign_orientations_conflict_free(
                    self.base.non_host_grain_ids, self.neigh_graph,
                    self.full_pool, self.all_grain_orientations,
                    max_retries=50, rng=self._rng,
                    fallback_pool=self.fallback_quats)
            print(f'OrientationAssigner3D [L0 conflict_free]: '
                  f'{len(self.base.non_host_grain_ids)} non-host orientations '
                  f'(fallback: {self.n_fallback_nonhost})')

        elif mode in ('paired_pool', 'mdf_targeted', 'mrf'):
            if mode != 'paired_pool':
                print(f'  INFO: mode "{mode}" not yet implemented; '
                      f'using "paired_pool" (Level 1).')
            if self._paired_pool_full is None:
                raise RuntimeError(
                    'Call build_adjacency_model() before assign_nonhost_orientations() '
                    'when using mode "paired_pool".')
            (self.all_grain_orientations,
             self._n_paired_used_nonhost,
             self.n_fallback_nonhost) = self._assign_paired_pool(
                grain_ids    = self.base.non_host_grain_ids,
                neigh_graph  = self.neigh_graph,
                paired_pool  = self._paired_pool_full,
                fallback_pool= self.full_pool,
                all_assigned = self.all_grain_orientations,
                max_retries  = 50,
            )
            print(f'OrientationAssigner3D [L1 paired_pool]: '
                  f'{len(self.base.non_host_grain_ids)} non-host orientations  '
                  f'pair-matched: {self._n_paired_used_nonhost}  '
                  f'fallback: {self.n_fallback_nonhost}')

    # ── conflict check and MDF ────────────────────────────────────────────

    def check_conflicts(self) -> int:
        """Count adjacent grain pairs sharing identical orientations."""
        if self.neigh_graph is None:
            return -1
        conflicts = 0
        for gid, neighbours in self.neigh_graph.items():
            if gid not in self.all_grain_orientations:
                continue
            q_self = self.all_grain_orientations[gid].tobytes()
            for nb in neighbours:
                if nb > gid and nb in self.all_grain_orientations:
                    if self.all_grain_orientations[nb].tobytes() == q_self:
                        conflicts += 1
        self.n_conflicts = conflicts
        status = 'PASS' if conflicts == 0 else 'WARNING'
        print(f'OrientationAssigner3D: conflict check [{status}] '
              f'-- {conflicts} adjacent identical-orientation pairs')
        return conflicts

    def compute_mdf(
            self,
            lgi: Optional[np.ndarray] = None,
            n_bins: int = 65,
            angle_range: Tuple[float, float] = (0.0, 65.0),
    ) -> Dict:
        """Compute MDF for the current grain structure."""
        from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats
        if lgi is None:
            lgi = self.base.lgi
        if self.neigh_graph is None:
            raise RuntimeError('Call build_neighbour_graph() first.')
        quat_3d = np.zeros(lgi.shape + (4,), dtype=np.float64)
        for gid, q in self.all_grain_orientations.items():
            quat_3d[lgi == gid] = q
        neigh_list = {gid: list(ns) for gid, ns in self.neigh_graph.items()}
        return compute_mdf_from_quats(lgi, quat_3d, neigh_list,
                                      n_bins=n_bins, angle_range=angle_range)
