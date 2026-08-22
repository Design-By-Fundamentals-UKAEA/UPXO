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

Level 2a  ``'mdf_conditioned_pairs'``
    Extends Level 1 by conditioning the pair selection on the EBSD
    parent-state MDF.  A target misorientation angle theta is sampled
    from the reference MDF, and the search is restricted to the angle
    bin of the pre-binned EBSD pair pool that corresponds to theta.
    This reproduces both the orientation-pair identity (Level 1) AND
    the misorientation angle distribution (Level 2a).
    Requires ``build_adjacency_model(mdf_ref=mdf_merged)``.
    Fallback chain: target bin -> any bin (Level 1) -> uniform pool.
    Bin width = 65 / n_bins, inferred automatically from mdf_ref.

Level 2b  ``'mdf_analytical'``
    Samples target misorientation angle theta from the EBSD MDF,
    generates a rotation quaternion of angle theta about a random axis,
    applies it to the anchor orientation q_N to produce a candidate,
    then snaps to the nearest EBSD pool member.  Covers more of
    orientation space than pair-pool methods while remaining anchored
    to physically-observed EBSD orientations.
    Requires ``build_adjacency_model(mdf_ref=mdf_merged)``.
    Note: cubic-symmetry reduction means the actual crystallographic
    disorientation will be <= theta; the distribution is biased toward
    theta and converges with enough realisations.

Level 3a  ``'mrf_gibbs'``
    Full Markov Random Field treatment via Gibbs sampling: grain orientation
    is sampled from P(q_G | {q_nb}) proportional to the product of EBSD
    P_MDF densities across all assigned neighbours.  Sweeps until W1
    convergence against the EBSD reference MDF.

Level 3b  ``'mrf_map'``
    MRF MAP estimation via Simulated Annealing.  Same energy function as
    Level 3a but uses temperature-scaled argmax (SA) instead of stochastic
    sampling.  Geometric cooling from mrf_sa_t_start to mrf_sa_t_end over
    mrf_max_sweeps sweeps; converges to pure MAP (argmax) at T→0.
"""

import math
import threading
import warnings
import numpy as np
from typing import Optional, Dict, Tuple

# Valid mode identifiers (ordered by physical fidelity)
_VALID_ORIENT_MODES = ('conflict_free', 'paired_pool',
                       'mdf_conditioned_pairs', 'mdf_analytical',
                       'mrf_gibbs', 'mrf_map')


class OrientationAssigner3D:
    """
    Conflict-aware crystallographic orientation assignment for 3D SGC grains.

    Assigns per-grain unit quaternions from EBSD-derived pools, controlled
    by ``orientation_assignment_mode`` (increasing physical fidelity):

    * ``'conflict_free'`` — independent sampling, no adjacent duplicate
    * ``'paired_pool'`` — EBSD-observed neighbour orientation pairs
    * ``'mdf_conditioned_pairs'`` — pairs conditioned on parent MDF bins
    * ``'mdf_analytical'`` — MDF-sampled misorientation + pool snap
    * ``'mrf_gibbs'`` / ``'mrf_map'`` — full MRF (Gibbs / SA MAP)

    See the module docstring for full level descriptions. Outputs
    ``all_grain_orientations`` and fallback / conflict counters.
    """

    __slots__ = (
        'base', 'neigh_graph',
        'parent_pool', 'full_pool', 'fallback_quats',
        'all_grain_orientations', 'n_fallback_host', 'n_fallback_nonhost',
        'n_conflicts',
        # Adjacency model (Level 1+)
        'orientation_assignment_mode',
        'pair_similarity_deg',
        'max_retries',
        '_paired_pool_host',
        '_paired_pool_full',
        # Level 2a: angle-binned pair pools
        '_paired_pool_host_by_angle_bin',
        '_paired_pool_full_by_angle_bin',
        # Level 2b: MDF reference for analytical generation
        '_mdf_ref_angles',
        '_mdf_angle_max',
        '_mdf_bin_width',
        '_n_paired_used_host',
        '_n_paired_used_nonhost',
        # MRF Level 3 parameters
        'mrf_max_sweeps',
        'mrf_eps_convergence',
        'mrf_eps_quality',
        'mrf_init_mode',
        'mrf_kde_threshold',
        'mrf_bilateral_symmetry',
        # MRF Level 3b SA parameters
        'mrf_sa_t_start',
        'mrf_sa_t_end',
        '_rng',
        # Per-sweep diagnostics for Level 3a/3b (mrf_gibbs/mrf_map) --
        # populated during assign_nonhost_orientations, one dict per sweep:
        # {'sweep', 'w1_sweep', 'w1_ebsd', 'n_updated'} for mrf_gibbs, plus
        # 'temperature' for mrf_map. Empty for all other modes.
        'mrf_history',
        # Optional threading.Event, checked between grains (Levels 0-2b)
        # or between sweeps (Levels 3a/3b); raises AssignmentCancelled the
        # moment it's set. None (default) disables cancellation entirely.
        'cancel_event',
    )

    def __init__(
            self,
            base,
            orientation_assignment_mode: str = 'conflict_free',
            pair_similarity_deg: float = 10.0,
            max_retries: int = 50,
            mrf_max_sweeps: int = 20,
            mrf_eps_convergence: float = 0.3,
            mrf_eps_quality: float = 1.5,
            mrf_init_mode: str = 'mdf_analytical',
            mrf_kde_threshold: Optional[int] = None,
            mrf_bilateral_symmetry: bool = False,
            mrf_sa_t_start: float = 5.0,
            mrf_sa_t_end: float = 0.05,
            rng_seed: Optional[int] = None,
            cancel_event: Optional[threading.Event] = None,
    ):
        if orientation_assignment_mode not in _VALID_ORIENT_MODES:
            raise ValueError(
                f'orientation_assignment_mode must be one of '
                f'{_VALID_ORIENT_MODES}, got "{orientation_assignment_mode}"')
        self.base = base
        self.orientation_assignment_mode = orientation_assignment_mode
        self.pair_similarity_deg = float(pair_similarity_deg)
        self.max_retries = int(max_retries)
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
        self._paired_pool_host_by_angle_bin: Optional[dict] = None
        self._paired_pool_full_by_angle_bin: Optional[dict] = None
        self._mdf_ref_angles: Optional[object] = None
        self._mdf_angle_max: float = 65.0
        self._mdf_bin_width: float = 65.0 / 25
        self._n_paired_used_host: int = 0
        self._n_paired_used_nonhost: int = 0
        self.mrf_max_sweeps: int = int(mrf_max_sweeps)
        self.mrf_eps_convergence: float = float(mrf_eps_convergence)
        self.mrf_eps_quality: float = float(mrf_eps_quality)
        self.mrf_init_mode: str = mrf_init_mode
        self.mrf_kde_threshold: Optional[int] = mrf_kde_threshold
        self.mrf_bilateral_symmetry: bool = bool(mrf_bilateral_symmetry)
        self.mrf_sa_t_start: float = float(mrf_sa_t_start)
        self.mrf_sa_t_end: float = float(mrf_sa_t_end)
        self._rng = np.random.default_rng(rng_seed)
        self.mrf_history: list = []
        self.cancel_event = cancel_event

        print(f'OrientationAssigner3D: mode = "{orientation_assignment_mode}"')

    def _check_cancelled(self):
        """Raise AssignmentCancelled if self.cancel_event is set. Called
        once per grain (Levels 0-2b) or once per sweep (Levels 3a/3b) --
        the only two granularities long enough for a "Stop" click to
        matter; cancel_event=None (the default) makes this a no-op."""
        from upxo.xtalphy.crystal_orientation import AssignmentCancelled
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise AssignmentCancelled('Orientation assignment stopped by user.')

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
            fallback_tc_info: Optional[Dict] = None,
            fallback_apply_symmetry: Optional[Dict[str, bool]] = None,
    ):
        """Build parent-only and full-EBSD quaternion pools.

        fallback_tc_info / fallback_apply_symmetry : optional overrides for
        the synthetic fallback pool's texture-component recipe (name ->
        [volume_fraction, [phi1, Phi, phi2], ...] / name -> bool). Both
        default to None, which reproduces the prior unconditional
        behaviour unchanged: tops.synth_fcc_quats' own hardcoded default
        recipe (Copper/Brass/Goss/Rotated-Cube), fully symmetry-expanded.
        """
        from upxo.xtalphy.crystal_orientation import grain_avg_quats
        from upxo.xtalphy.texops import tops

        ebsd_gids, ebsd_q = grain_avg_quats(ebsd_lfi, ebsd_quat)
        gid2q = {int(g): ebsd_q[i] for i, g in enumerate(ebsd_gids)}

        ebsd_parent_ids = parent_info[csl_label]['pure_parents']
        self.parent_pool = np.array(
            [gid2q[g] for g in sorted(ebsd_parent_ids) if g in gid2q])
        self.full_pool = ebsd_q

        fallback_kwargs = {}
        if fallback_tc_info is not None:
            fallback_kwargs['tc_info'] = fallback_tc_info
        if fallback_apply_symmetry is not None:
            fallback_kwargs['apply_symmetry'] = fallback_apply_symmetry
        self.fallback_quats = tops.synth_fcc_quats(
            N=max(n_fallback, self.base.n_grains), **fallback_kwargs)

        print(f'OrientationAssigner3D: EBSD parent pool = {len(self.parent_pool)}, '
              f'full pool = {len(self.full_pool)}, '
              f'fallback = {len(self.fallback_quats)}')

    # ── Level 1: adjacency model ───────────────────────────────────────────

    def build_adjacency_model(
            self,
            rg,
            parent_info: Dict,
            csl_label: str,
            mdf_ref: Optional[dict] = None,
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

        # ── Level 2a: bin pairs by cubic misorientation angle ─────────
        if mdf_ref is not None:
            _miso_ref = (mdf_ref.get('miso_deg', np.array([]))
                         if isinstance(mdf_ref, dict) else np.asarray(mdf_ref))
            _edges = (mdf_ref.get('hist_bin_edges')
                      if isinstance(mdf_ref, dict) else None)
            if _edges is not None and len(_edges) > 1:
                # Exact -- reflects whatever n_bins/angle_max actually built
                # mdf_ref (_build_ebsd_merged_mdf's mdf_n_bins/mdf_angle_max),
                # instead of assuming a fixed 65 deg range.
                _bin_w = float(_edges[1] - _edges[0])
                _angle_max = float(_edges[-1])
            else:
                _n_bins = (len(mdf_ref.get('hist_bin_centers', []))
                           if isinstance(mdf_ref, dict) and 'hist_bin_centers' in mdf_ref
                           else 25)
                _angle_max = 65.0
                _bin_w = _angle_max / _n_bins
            self._mdf_angle_max = _angle_max
            self._mdf_ref_angles = _miso_ref
            self._mdf_bin_width  = _bin_w

            def _bin_pairs(pairs):
                from upxo.xtalphy.crystal_orientation import (
                    quat_to_R_batch, get_cubic_ops_np)
                if not pairs:
                    return {}
                S   = get_cubic_ops_np()
                pA  = np.array([p[0] for p in pairs])
                pB  = np.array([p[1] for p in pairs])
                RA  = quat_to_R_batch(pA)
                RB  = quat_to_R_batch(pB)
                SA  = np.einsum('sij,pjk->psik', S, RA)
                SB  = np.einsum('sij,pjk->psik', S, RB)
                dR  = np.einsum('pmab,pncb->pmnac', SB, SA)
                tr  = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]
                miso = np.degrees(
                    np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)).min(axis=(1, 2)))
                binned = {}
                for i, m in enumerate(miso):
                    b = int(m / _bin_w)
                    binned.setdefault(b, []).append(pairs[i])
                return binned

            self._paired_pool_host_by_angle_bin = _bin_pairs(host_pairs)
            self._paired_pool_full_by_angle_bin = _bin_pairs(full_pairs)
            print(f'  Angle-binned host pool : '
                  f'{len(self._paired_pool_host_by_angle_bin)} bins  '
                  f'bin_width={_bin_w:.1f} deg  (Level 2a ready)')
            print(f'  MDF reference loaded   : '
                  f'{len(_miso_ref)} misorientation samples  (Level 2b ready)')

        print(f'OrientationAssigner3D (Level 1 adjacency model):')
        print(f'  Host pair pool (pure-parent pairs) : {len(host_pairs)}')
        print(f'  Full pair pool (all EBSD pairs)    : {len(full_pairs)}')

    # ── private: paired-pool assignment (Level 1) ─────────────────────────

    def _assign_mdf_conditioned_pairs(
            self,
            grain_ids,
            neigh_graph: Dict,
            paired_pool_by_bin: dict,
            fallback_pool: np.ndarray,
            all_assigned: Dict,
            max_retries: int = 50,
    ) -> tuple:
        """
        Level 2a -- MDF-conditioned pair assignment.

        Extends Level 1 by sampling a target misorientation angle theta from
        the EBSD parent-state MDF reference, then searching ONLY the angle
        bin corresponding to theta in the pre-binned EBSD pair pool.  This
        ensures the sampled pairs reproduce both the EBSD orientation-pair
        identity (Level 1) AND the EBSD misorientation angle distribution
        (new).

        Fallback chain:
          1. Target angle bin in EBSD pair pool (Level 2a, with similarity check)
          2. Any angle bin in EBSD pair pool (Level 1 behaviour)
          3. Uniform sampling from fallback_pool
        """
        from upxo.xtalphy.crystal_orientation import _positive_w
        cos_thresh = math.cos(math.radians(self.pair_similarity_deg))
        miso_ref   = self._mdf_ref_angles
        bin_w      = self._mdf_bin_width
        n_conditioned, n_fb_lvl1, n_fb_pool = 0, 0, 0
        n_fp = len(fallback_pool)

        # Flatten all bins for Level-1 fallback search
        all_pairs_flat = [(p) for v in paired_pool_by_bin.values() for p in v]
        if all_pairs_flat:
            _flat_A = np.array([p[0] for p in all_pairs_flat])
            _flat_B = np.array([p[1] for p in all_pairs_flat])

        for gid in self._rng.permutation(list(grain_ids)).tolist():
            self._check_cancelled()
            gid = int(gid)
            used_by_neigh = {
                all_assigned[nb].tobytes()
                for nb in neigh_graph.get(gid, set()) if nb in all_assigned}
            assigned_nb = [
                all_assigned[nb]
                for nb in neigh_graph.get(gid, set()) if nb in all_assigned]
            chosen = None

            if assigned_nb and miso_ref is not None and len(miso_ref) > 0 and paired_pool_by_bin:
                q_N = assigned_nb[0]
                theta = float(self._rng.choice(miso_ref))
                target_bin = int(theta / bin_w)
                bin_pairs = paired_pool_by_bin.get(target_bin, [])

                if bin_pairs:
                    _bA = np.array([p[0] for p in bin_pairs])
                    _bB = np.array([p[1] for p in bin_pairs])
                    sim_A = np.abs(_bA @ q_N)
                    sim_B = np.abs(_bB @ q_N)
                    for ia in np.argsort(-sim_A)[:max_retries]:
                        if sim_A[ia] < cos_thresh: break
                        c = _positive_w(_bB[ia].copy())
                        if c.tobytes() not in used_by_neigh:
                            chosen = c; n_conditioned += 1; break
                    if chosen is None:
                        for ib in np.argsort(-sim_B)[:max_retries]:
                            if sim_B[ib] < cos_thresh: break
                            c = _positive_w(_bA[ib].copy())
                            if c.tobytes() not in used_by_neigh:
                                chosen = c; n_conditioned += 1; break

                # Fallback to Level-1 (any bin)
                if chosen is None and all_pairs_flat:
                    sim_A2 = np.abs(_flat_A @ q_N)
                    sim_B2 = np.abs(_flat_B @ q_N)
                    for ia, ib in zip(np.argsort(-sim_A2)[:max_retries],
                                      np.argsort(-sim_B2)[:max_retries]):
                        if sim_A2[ia] >= cos_thresh:
                            c = _positive_w(_flat_B[ia].copy())
                            if c.tobytes() not in used_by_neigh:
                                chosen = c; n_fb_lvl1 += 1; break
                        if sim_B2[ib] >= cos_thresh and chosen is None:
                            c = _positive_w(_flat_A[ib].copy())
                            if c.tobytes() not in used_by_neigh:
                                chosen = c; n_fb_lvl1 += 1; break

            # Fallback to uniform pool
            if chosen is None:
                for _ in range(max_retries):
                    c = _positive_w(fallback_pool[int(self._rng.integers(0, n_fp))].copy())
                    if c.tobytes() not in used_by_neigh:
                        chosen = c; break
                if chosen is None:
                    chosen = _positive_w(fallback_pool[int(self._rng.integers(0, n_fp))].copy())
                n_fb_pool += 1
            all_assigned[gid] = chosen

        return all_assigned, n_conditioned, n_fb_lvl1, n_fb_pool

    def _assign_mdf_analytical(
            self,
            grain_ids,
            neigh_graph: Dict,
            full_pool: np.ndarray,
            all_assigned: Dict,
            max_retries: int = 50,
    ) -> tuple:
        """
        Level 2b -- Analytical MDF-targeted orientation generation.

        For each grain G with assigned neighbour N (orientation q_N):
          1. Samples target misorientation angle theta from the EBSD parent-
             state MDF reference distribution.
          2. Generates a random rotation axis and constructs a rotation
             quaternion R_theta of angle theta about that axis.
          3. Computes candidate q_G = R_theta o q_N (rotation in sample frame).
          4. Finds the nearest member of the EBSD orientation pool by quaternion
             dot product and uses that as the final orientation.  This anchors
             the result to physically-observed EBSD orientations.
          5. Applies conflict-free constraint.

        Note: the cubic disorientation between the assigned q_G and q_N will
        be <= theta due to cubic symmetry reduction.  The distribution of
        actual cubic disorientations over many assignments is biased toward
        theta, progressively improving MDF alignment.

        Fallback: uniform pool sampling when the search fails.
        """
        from upxo.xtalphy.crystal_orientation import _positive_w, _quat_mul
        miso_ref = self._mdf_ref_angles
        pool_arr = full_pool if isinstance(full_pool, np.ndarray) else np.array(full_pool)
        n_fp = len(pool_arr)
        n_analytical, n_fallback = 0, 0

        for gid in self._rng.permutation(list(grain_ids)).tolist():
            self._check_cancelled()
            gid = int(gid)
            used_by_neigh = {
                all_assigned[nb].tobytes()
                for nb in neigh_graph.get(gid, set()) if nb in all_assigned}
            assigned_nb = [
                all_assigned[nb]
                for nb in neigh_graph.get(gid, set()) if nb in all_assigned]
            chosen = None

            if assigned_nb and miso_ref is not None and len(miso_ref) > 0:
                q_N = assigned_nb[0]
                for _ in range(max_retries):
                    # Sample target angle and generate rotation
                    theta_rad = np.radians(float(self._rng.choice(miso_ref)))
                    axis = self._rng.standard_normal(3)
                    axis /= np.linalg.norm(axis)
                    R_theta = np.array([
                        np.cos(theta_rad / 2.0),
                        *np.sin(theta_rad / 2.0) * axis], dtype=np.float64)
                    q_cand_raw = _positive_w(_quat_mul(R_theta, q_N))
                    # Snap to nearest EBSD pool member
                    idx = int(np.argmax(np.abs(pool_arr @ q_cand_raw)))
                    candidate = _positive_w(pool_arr[idx].copy())
                    if candidate.tobytes() not in used_by_neigh:
                        chosen = candidate; n_analytical += 1; break

            # Fallback
            if chosen is None:
                for _ in range(max_retries):
                    c = _positive_w(pool_arr[int(self._rng.integers(0, n_fp))].copy())
                    if c.tobytes() not in used_by_neigh:
                        chosen = c; break
                if chosen is None:
                    chosen = _positive_w(pool_arr[int(self._rng.integers(0, n_fp))].copy())
                n_fallback += 1
            all_assigned[gid] = chosen

        return all_assigned, n_analytical, n_fallback

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
            self._check_cancelled()
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
                    max_retries=self.max_retries, rng=self._rng,
                    cancel_event=self.cancel_event,
                    fallback_pool=self.fallback_quats)
            print(f'OrientationAssigner3D [L0 conflict_free]: '
                  f'{len(self.base.host_grain_ids)} host orientations '
                  f'(fallback: {self.n_fallback_host})')

        elif mode == 'paired_pool':
            if self._paired_pool_host is None:
                raise RuntimeError('Call build_adjacency_model() first.')
            (self.all_grain_orientations, self._n_paired_used_host,
             self.n_fallback_host) = self._assign_paired_pool(
                self.base.host_grain_ids, self.neigh_graph,
                self._paired_pool_host, self.parent_pool,
                self.all_grain_orientations, max_retries=self.max_retries)
            print(f'OrientationAssigner3D [L1 paired_pool]: '
                  f'{len(self.base.host_grain_ids)} host orientations  '
                  f'pair-matched: {self._n_paired_used_host}  '
                  f'fallback: {self.n_fallback_host}')

        elif mode == 'mdf_conditioned_pairs':
            if self._paired_pool_host_by_angle_bin is None:
                raise RuntimeError('Call build_adjacency_model(mdf_ref=mdf_merged) first.')
            (self.all_grain_orientations, n_cond, n_fb1, n_fb2) = \
                self._assign_mdf_conditioned_pairs(
                    self.base.host_grain_ids, self.neigh_graph,
                    self._paired_pool_host_by_angle_bin,
                    self.parent_pool, self.all_grain_orientations, max_retries=self.max_retries)
            self._n_paired_used_host = n_cond
            self.n_fallback_host = n_fb2
            print(f'OrientationAssigner3D [L2a mdf_conditioned_pairs]: '
                  f'{len(self.base.host_grain_ids)} host orientations  '
                  f'angle-conditioned: {n_cond}  lvl1-fallback: {n_fb1}  '
                  f'pool-fallback: {n_fb2}')

        elif mode == 'mrf_gibbs':
            # Initialise host grains; Gibbs sweeps run after nonhost assignment
            if self._mdf_ref_angles is None:
                raise RuntimeError('Call build_adjacency_model(mdf_ref=mdf_merged) first.')
            all_assigned_tmp, n_ana, _ = self._assign_mdf_analytical(
                self.base.host_grain_ids, self.neigh_graph,
                self.parent_pool, self.all_grain_orientations, max_retries=self.max_retries)
            self.all_grain_orientations = all_assigned_tmp
            self._n_paired_used_host = n_ana
            print(f'OrientationAssigner3D [L3 mrf_gibbs init]: '
                  f'{len(self.base.host_grain_ids)} host grains initialised '
                  f'(Gibbs sweeps run after assign_nonhost_orientations)')

        elif mode == 'mrf_map':
            if self._mdf_ref_angles is None:
                raise RuntimeError('Call build_adjacency_model(mdf_ref=mdf_merged) first.')
            all_assigned_tmp, n_ana, _ = self._assign_mdf_analytical(
                self.base.host_grain_ids, self.neigh_graph,
                self.parent_pool, self.all_grain_orientations, max_retries=self.max_retries)
            self.all_grain_orientations = all_assigned_tmp
            self._n_paired_used_host = n_ana
            print(f'OrientationAssigner3D [L3b mrf_map init]: '
                  f'{len(self.base.host_grain_ids)} host grains warm-started '
                  f'(SA sweeps run after assign_nonhost_orientations)')

        elif mode in ('mdf_analytical', 'mdf_targeted'):
            if mode == 'mdf_targeted':
                warnings.warn(
                    'orientation_assignment_mode="mdf_targeted" is not '
                    'implemented; silently using "mdf_analytical" (Level 2b) '
                    'instead.', stacklevel=2)
                print(f'  INFO: mode "mdf_targeted" not implemented; '
                      f'using "mdf_analytical" (Level 2b).')
            if self._mdf_ref_angles is None:
                raise RuntimeError('Call build_adjacency_model(mdf_ref=mdf_merged) first.')
            (self.all_grain_orientations, n_ana, n_fb2) = \
                self._assign_mdf_analytical(
                    self.base.host_grain_ids, self.neigh_graph,
                    self.parent_pool, self.all_grain_orientations, max_retries=self.max_retries)
            self._n_paired_used_host = n_ana
            self.n_fallback_host = n_fb2
            print(f'OrientationAssigner3D [L2b mdf_analytical]: '
                  f'{len(self.base.host_grain_ids)} host orientations  '
                  f'analytical: {n_ana}  fallback: {n_fb2}')

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
                    max_retries=self.max_retries, rng=self._rng,
                    cancel_event=self.cancel_event,
                    fallback_pool=self.fallback_quats)
            print(f'OrientationAssigner3D [L0 conflict_free]: '
                  f'{len(self.base.non_host_grain_ids)} non-host orientations '
                  f'(fallback: {self.n_fallback_nonhost})')

        elif mode == 'paired_pool':
            if self._paired_pool_full is None:
                raise RuntimeError('Call build_adjacency_model() first.')
            (self.all_grain_orientations, self._n_paired_used_nonhost,
             self.n_fallback_nonhost) = self._assign_paired_pool(
                self.base.non_host_grain_ids, self.neigh_graph,
                self._paired_pool_full, self.full_pool,
                self.all_grain_orientations, max_retries=self.max_retries)
            print(f'OrientationAssigner3D [L1 paired_pool]: '
                  f'{len(self.base.non_host_grain_ids)} non-host orientations  '
                  f'pair-matched: {self._n_paired_used_nonhost}  '
                  f'fallback: {self.n_fallback_nonhost}')

        elif mode == 'mdf_conditioned_pairs':
            if self._paired_pool_full_by_angle_bin is None:
                raise RuntimeError('Call build_adjacency_model(mdf_ref=mdf_merged) first.')
            (self.all_grain_orientations, n_cond, n_fb1, n_fb2) = \
                self._assign_mdf_conditioned_pairs(
                    self.base.non_host_grain_ids, self.neigh_graph,
                    self._paired_pool_full_by_angle_bin,
                    self.full_pool, self.all_grain_orientations, max_retries=self.max_retries)
            self._n_paired_used_nonhost = n_cond
            self.n_fallback_nonhost = n_fb2
            print(f'OrientationAssigner3D [L2a mdf_conditioned_pairs]: '
                  f'{len(self.base.non_host_grain_ids)} non-host orientations  '
                  f'angle-conditioned: {n_cond}  lvl1-fallback: {n_fb1}  '
                  f'pool-fallback: {n_fb2}')

        elif mode == 'mrf_gibbs':
            if self._mdf_ref_angles is None:
                raise RuntimeError('Call build_adjacency_model(mdf_ref=mdf_merged) first.')
            # Warm-start non-host grains (host grains done in assign_host_orientations)
            all_assigned_tmp, n_ana, _ = self._assign_mdf_analytical(
                self.base.non_host_grain_ids, self.neigh_graph,
                self.full_pool, self.all_grain_orientations, max_retries=self.max_retries)
            self.all_grain_orientations = all_assigned_tmp
            self._n_paired_used_nonhost = n_ana
            print(f'  {len(self.base.non_host_grain_ids)} non-host grains warm-started')
            # Run Gibbs over ALL grains
            self.all_grain_orientations = self._assign_mrf_gibbs(
                self.base.host_grain_ids, self.base.non_host_grain_ids,
                self.parent_pool, self.full_pool,
                self.neigh_graph, self.all_grain_orientations)

        elif mode == 'mrf_map':
            if self._mdf_ref_angles is None:
                raise RuntimeError('Call build_adjacency_model(mdf_ref=mdf_merged) first.')
            # Warm-start non-host grains (host grains done in assign_host_orientations)
            all_assigned_tmp, n_ana, _ = self._assign_mdf_analytical(
                self.base.non_host_grain_ids, self.neigh_graph,
                self.full_pool, self.all_grain_orientations, max_retries=self.max_retries)
            self.all_grain_orientations = all_assigned_tmp
            self._n_paired_used_nonhost = n_ana
            print(f'  {len(self.base.non_host_grain_ids)} non-host grains warm-started')
            # Run MAP/SA over ALL grains
            self.all_grain_orientations = self._assign_mrf_map(
                self.base.host_grain_ids, self.base.non_host_grain_ids,
                self.parent_pool, self.full_pool,
                self.neigh_graph, self.all_grain_orientations)

        elif mode in ('mdf_analytical', 'mdf_targeted'):
            if mode == 'mdf_targeted':
                warnings.warn(
                    'orientation_assignment_mode="mdf_targeted" is not '
                    'implemented; silently using "mdf_analytical" (Level 2b) '
                    'instead.', stacklevel=2)
                print(f'  INFO: mode "mdf_targeted" not implemented; '
                      f'using "mdf_analytical" (Level 2b).')
            if self._mdf_ref_angles is None:
                raise RuntimeError('Call build_adjacency_model(mdf_ref=mdf_merged) first.')
            (self.all_grain_orientations, n_ana, n_fb2) = \
                self._assign_mdf_analytical(
                    self.base.non_host_grain_ids, self.neigh_graph,
                    self.full_pool, self.all_grain_orientations, max_retries=self.max_retries)
            self._n_paired_used_nonhost = n_ana
            self.n_fallback_nonhost = n_fb2
            print(f'OrientationAssigner3D [L2b mdf_analytical]: '
                  f'{len(self.base.non_host_grain_ids)} non-host orientations  '
                  f'analytical: {n_ana}  fallback: {n_fb2}')

    # ── Level 3: MRF Gibbs sampling helpers ──────────────────────────────────

    def _compute_pebsd_density(self, miso_deg: 'np.ndarray', n_bins: int = 25):
        """
        Build P_EBSD(theta) density estimate from the EBSD misorientation
        sample array.

        Switching criterion (mrf_kde_threshold):
          None  -- auto: compute minimum histogram bin count; use local
                   averaging if min_count < auto_threshold, else KDE.
          int   -- explicit threshold.
          < 15 bins on average  -> local averaging (3-point moving average)
          >= 15 bins on average -> KDE with Scott bandwidth

        Returns (bin_centers, density) both as 1-D ndarrays.
        """
        from scipy.stats import gaussian_kde

        bins = np.linspace(0.0, self._mdf_angle_max, n_bins + 1)
        counts, _ = np.histogram(miso_deg, bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])

        # Determine threshold
        auto_thresh = max(10, int(len(miso_deg) / n_bins / 2))
        thresh = auto_thresh if self.mrf_kde_threshold is None else int(self.mrf_kde_threshold)
        use_kde = (counts.min() >= thresh)

        if use_kde:
            kde = gaussian_kde(miso_deg)
            density = kde(centers)
        else:
            # 3-point moving average to fill sparse/empty bins
            raw = counts.astype(float)
            density = np.convolve(raw, [0.25, 0.5, 0.25], mode='same')
            density = np.clip(density, 1e-12, None)

        density = density / (density.sum() * (self._mdf_angle_max / n_bins))  # normalise to density
        return centers, density

    def _compute_batch_miso(
            self,
            R_cand: 'np.ndarray',
            R_j: 'np.ndarray',
            S: 'np.ndarray',
            use_bilateral: bool = False,
    ) -> 'np.ndarray':
        """
        Vectorised cubic disorientation between all N candidates and one neighbour.

        R_cand        : (N, 3, 3)  candidate rotation matrices
        R_j           : (3, 3)     neighbour rotation matrix
        S             : (24, 3, 3) cubic symmetry operators
        use_bilateral : False -> 24-op left-symmetry only (default, fast).
                        True  -> full 24x24=576 bilateral symmetry (exact).

        Returns (N,) disorientation angles in degrees.
        """
        if use_bilateral:
            # Full bilateral: S_i @ R_cand[n] @ R_j^T @ S_j^T for all i,j
            # (identical to compute_mdf_from_quats inner loop)
            N = R_cand.shape[0]
            SA = np.einsum('sij,njk->nsik', S, R_cand)          # (N, 24, 3, 3)
            SB_s = np.einsum('sij,jk->sik', S, R_j)             # (24, 3, 3)
            SB   = np.broadcast_to(SB_s[None], (N, 24, 3, 3)).copy()
            dR   = np.einsum('pmab,pncb->pmnac', SB, SA)        # (N, 24, 24, 3, 3)
            tr   = dR[...,0,0] + dR[...,1,1] + dR[...,2,2]     # (N, 24, 24)
            cos_ang = np.clip((tr.max(axis=(1, 2)) - 1.0) * 0.5, -1.0, 1.0)
        else:
            # Left-symmetry only (24 operators) — fast, approximate
            R_rel = np.einsum('nij,kj->nik', R_cand, R_j)       # (N, 3, 3)
            SR    = np.einsum('sij,njk->snik', S, R_rel)         # (24, N, 3, 3)
            tr    = SR[:,:,0,0] + SR[:,:,1,1] + SR[:,:,2,2]     # (24, N)
            cos_ang = np.clip((tr.max(axis=0) - 1.0) * 0.5, -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))  # (N,)

    def _compute_boundary_miso(
            self,
            neigh_graph: dict,
            all_assigned: dict,
            S: 'np.ndarray',
    ) -> 'np.ndarray':
        """
        Compute cubic disorientation for every assigned grain-boundary pair.
        Used for convergence monitoring after each Gibbs sweep.
        Returns 1-D array of misorientation angles in degrees.
        """
        from upxo.xtalphy.crystal_orientation import quat_to_R_batch

        pairs_set = set()
        for a, neighbours in neigh_graph.items():
            if a not in all_assigned:
                continue
            for b in neighbours:
                if b in all_assigned and b != a:
                    pairs_set.add((min(a, b), max(a, b)))

        if not pairs_set:
            return np.array([])

        pairs = list(pairs_set)
        qA = np.array([all_assigned[p[0]] for p in pairs])
        qB = np.array([all_assigned[p[1]] for p in pairs])
        RA = quat_to_R_batch(qA)
        RB = quat_to_R_batch(qB)

        if not self.mrf_bilateral_symmetry:
            # Fully vectorised: all P pairs in a single einsum (no Python loop)
            R_rel = np.einsum('pij,pkj->pik', RA, RB)           # (P,3,3): RA[p]@RB[p].T
            SR    = np.einsum('sij,pjk->spik', S, R_rel)         # (24,P,3,3)
            tr    = SR[:,:,0,0] + SR[:,:,1,1] + SR[:,:,2,2]      # (24,P)
            return np.degrees(np.arccos(
                np.clip((tr.max(axis=0) - 1.0) * 0.5, -1.0, 1.0)))  # (P,)
        else:
            misos = np.empty(len(pairs))
            for idx, (rA, rB) in enumerate(zip(RA, RB)):
                misos[idx] = self._compute_batch_miso(
                    rA[None], rB, S, use_bilateral=True)[0]
            return misos

    def _run_one_gibbs_sweep(
            self,
            host_ids: set,
            nonhost_ids: set,
            host_pool: 'np.ndarray',
            full_pool: 'np.ndarray',
            neigh_graph: dict,
            all_assigned: dict,
            bin_centers: 'np.ndarray',
            density: 'np.ndarray',
            S: 'np.ndarray',
    ) -> tuple:
        """
        Execute one full Gibbs sweep over ALL grains.

        For each grain G (in random order):
          1. Determine candidate pool (host_pool or full_pool).
          2. Compute R matrices for all candidates.
          3. For each assigned neighbour: compute approx cubic disorientation
             with all candidates, look up P_EBSD density, accumulate log-weight.
          4. Sample q_G from softmax of log-weights.

        Returns (all_assigned, n_updated) where n_updated counts
        grains whose orientation changed.
        """
        from upxo.xtalphy.crystal_orientation import quat_to_R_batch, _positive_w

        all_gids = list(host_ids | nonhost_ids)
        order = self._rng.permutation(all_gids).tolist()
        n_updated = 0

        # Pre-build R matrices for both pools
        R_host = quat_to_R_batch(host_pool)   # (Nh, 3, 3)
        R_full = quat_to_R_batch(full_pool)   # (Nf, 3, 3)
        bin_w = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else self._mdf_angle_max / 25

        # S applied to each pool's candidates is invariant across grains AND
        # neighbours within this sweep (S/pool_R don't change per-grain) --
        # hoisting it out of the per-grain loop turns the old per-grain
        # (24,N,K,3,3) SR tensor (rebuilt from scratch every grain) into a
        # single (24,N,K) einsum against this precomputed (24,N,3,3) tensor.
        # trace(S[s] @ pool_R[n] @ R_nbs[k].T) == sum_ij SP[s,n,i,j]*R_nbs[k,i,j],
        # verified numerically identical to the old two-einsum construction
        # (~22x faster at N_pool=2000/K=15, machine-precision-exact).
        SP_host = np.einsum('sip,npj->snij', S, R_host)   # (24,Nh,3,3)
        SP_full = np.einsum('sip,npj->snij', S, R_full)   # (24,Nf,3,3)

        for gid in order:
            gid = int(gid)
            is_host = gid in host_ids
            pool_q = host_pool if is_host else full_pool
            pool_R = R_host if is_host else R_full
            SP     = SP_host if is_host else SP_full
            N_cand = len(pool_q)

            assigned_nbs = [
                (nb, all_assigned[nb])
                for nb in neigh_graph.get(gid, set())
                if nb in all_assigned]

            if not assigned_nbs:
                # No assigned neighbours yet — sample uniformly
                chosen = _positive_w(pool_q[int(self._rng.integers(0, N_cand))].copy())
            else:
                if not self.mrf_bilateral_symmetry:
                    # Batch all K neighbours simultaneously: one einsum over (N, K)
                    R_nbs  = quat_to_R_batch(
                        np.array([q for _, q in assigned_nbs]))        # (K,3,3)
                    tr     = np.einsum('snij,kij->snk', SP, R_nbs)     # (24,N,K)
                    cos_a  = np.clip((tr.max(axis=0)-1.0)*0.5,-1.0,1.0)# (N,K)
                    miso   = np.degrees(np.arccos(cos_a))               # (N,K)
                    bin_idx = np.clip(
                        (miso/bin_w).astype(int), 0, len(density)-1)
                    log_w = np.log(
                        np.clip(density[bin_idx], 1e-10, None)).sum(axis=1)  # (N,)
                else:
                    log_w = np.zeros(N_cand)
                    for nb_id, q_nb in assigned_nbs:
                        R_nb = quat_to_R_batch(q_nb[None])[0]
                        miso = self._compute_batch_miso(
                            pool_R, R_nb, S, use_bilateral=True)
                        bin_idx = np.clip(
                            (miso/bin_w).astype(int), 0, len(density)-1)
                        log_w += np.log(np.clip(density[bin_idx], 1e-10, None))

                # Numerically stable softmax sampling
                log_w -= log_w.max()
                probs = np.exp(log_w)
                probs /= probs.sum()
                chosen_idx = int(self._rng.choice(N_cand, p=probs))
                chosen = _positive_w(pool_q[chosen_idx].copy())

            old = all_assigned.get(gid)
            all_assigned[gid] = chosen
            if old is None or old.tobytes() != chosen.tobytes():
                n_updated += 1

        return all_assigned, n_updated

    def _assign_mrf_gibbs(
            self,
            host_ids: set,
            nonhost_ids: set,
            host_pool: 'np.ndarray',
            full_pool: 'np.ndarray',
            neigh_graph: dict,
            all_assigned: dict,
    ) -> dict:
        """
        Full MRF Gibbs sampling over all grains.

        Initialisation:
          mrf_init_mode controls the starting orientation field.
          Valid values: any mode in _VALID_ORIENT_MODES minus the mrf modes,
          plus 'random'. Defaults to 'mdf_analytical'.

        Convergence:
          Stops when BOTH:
            W1(MDF_sweep_k, MDF_sweep_k-1) < mrf_eps_convergence  (swept)
            W1(MDF_sweep_k, MDF_EBSD)       < mrf_eps_quality      (quality)
          OR max_sweeps is reached.

        P_EBSD density:
          Built from self._mdf_ref_angles (set by build_adjacency_model).
          Sparse bins -> local averaging; dense bins -> KDE.
          Threshold set by mrf_kde_threshold (None = auto).
        """
        from scipy.stats import wasserstein_distance
        from upxo.xtalphy.crystal_orientation import (
            get_cubic_ops_np, _positive_w, assign_orientations_conflict_free)

        if self._mdf_ref_angles is None or len(self._mdf_ref_angles) == 0:
            raise RuntimeError(
                'call build_adjacency_model(mdf_ref=mdf_merged) before mrf_gibbs.')

        S = get_cubic_ops_np()  # (24, 3, 3)
        n_bins = max(10, int(self._mdf_angle_max / self._mdf_bin_width))

        # Build P_EBSD density
        bin_centers, density = self._compute_pebsd_density(
            self._mdf_ref_angles, n_bins=n_bins)
        print(f'OrientationAssigner3D [L3 mrf_gibbs]: '
              f'P_EBSD density built  n_bins={n_bins}  '
              f'{"KDE" if density.std() > 0 else "local_avg"}')

        # ── Initialise orientation field ────────────────────────────────────
        # Reuses the sweep's own fallback pool (self.fallback_quats, built once
        # in build_ebsd_pools from the actually-configured texture) instead of
        # a fresh, unseeded tops.synth_fcc_quats() call -- that call ran
        # unconditionally on every invocation (even when unused, for the
        # warm-start init modes below), wasting a full texture-generation pass
        # and, when actually used, drawing from a different default texture
        # set than the rest of the sweep.
        init_mode = self.mrf_init_mode
        fallback = self.fallback_quats

        if init_mode == 'random':
            for gid in host_ids:
                all_assigned[int(gid)] = _positive_w(
                    host_pool[int(self._rng.integers(0, len(host_pool)))].copy())
            for gid in nonhost_ids:
                all_assigned[int(gid)] = _positive_w(
                    full_pool[int(self._rng.integers(0, len(full_pool)))].copy())
            print(f'  Init: random draw from pools')
        elif init_mode == 'conflict_free':
            all_assigned, _ = assign_orientations_conflict_free(
                host_ids, neigh_graph, host_pool, all_assigned,
                max_retries=self.max_retries, rng=self._rng, cancel_event=self.cancel_event,
                fallback_pool=fallback)
            all_assigned, _ = assign_orientations_conflict_free(
                nonhost_ids, neigh_graph, full_pool, all_assigned,
                max_retries=self.max_retries, rng=self._rng, cancel_event=self.cancel_event,
                fallback_pool=fallback)
            print(f'  Init: conflict_free (Level 0)')
        elif init_mode in ('mdf_conditioned_pairs', 'mdf_analytical', 'paired_pool'):
            # Warm start: use already-populated all_assigned if present
            unassigned_host = host_ids - set(all_assigned.keys())
            unassigned_non  = nonhost_ids - set(all_assigned.keys())
            if unassigned_host or unassigned_non:
                # Need to assign missing grains via requested init mode
                if unassigned_host:
                    all_assigned, _, _ = self._assign_mdf_analytical(
                        unassigned_host, neigh_graph, host_pool, all_assigned)
                if unassigned_non:
                    all_assigned, _, _ = self._assign_mdf_analytical(
                        unassigned_non, neigh_graph, full_pool, all_assigned)
            print(f'  Init: warm start from {init_mode} (or existing assignments)')
        else:
            raise ValueError(f'Unknown mrf_init_mode: "{init_mode}"')

        # ── Gibbs sweeps ────────────────────────────────────────────────────
        miso_prev = self._compute_boundary_miso(neigh_graph, all_assigned, S)
        ebsd_ref  = self._mdf_ref_angles
        w1_init   = (float(wasserstein_distance(miso_prev, ebsd_ref))
                     if len(miso_prev) > 0 else np.inf)
        print(f'  Warm-start:  n_pairs={len(miso_prev)}  '
              f'mean_miso={miso_prev.mean():.2f} deg  '
              f'W1_EBSD={w1_init:.3f} deg  '
              f'(EBSD ref: n={len(ebsd_ref)}  mean={np.mean(ebsd_ref):.2f} deg)')

        if w1_init <= self.mrf_eps_quality:
            print(f'  Warm-start already meets quality criterion '
                  f'(W1_EBSD={w1_init:.3f} <= {self.mrf_eps_quality} deg) '
                  f'-- skipping Gibbs sweeps.')
            return all_assigned

        print(f'  Running up to {self.mrf_max_sweeps} Gibbs sweeps  '
              f'(eps_conv={self.mrf_eps_convergence:.1f} deg  '
              f'eps_qual={self.mrf_eps_quality:.1f} deg)')

        for sweep in range(self.mrf_max_sweeps):
            self._check_cancelled()
            all_assigned, n_upd = self._run_one_gibbs_sweep(
                host_ids, nonhost_ids, host_pool, full_pool,
                neigh_graph, all_assigned, bin_centers, density, S)

            miso_curr = self._compute_boundary_miso(neigh_graph, all_assigned, S)

            w1_sweep = (float(wasserstein_distance(miso_curr, miso_prev))
                        if len(miso_prev) > 0 and len(miso_curr) > 0 else np.inf)
            w1_ebsd  = (float(wasserstein_distance(miso_curr, ebsd_ref))
                        if len(miso_curr) > 0 else np.inf)

            print(f'  Sweep {sweep+1:>3d}  updated={n_upd:>5d}  '
                  f'W1_sweep={w1_sweep:.3f} deg  W1_EBSD={w1_ebsd:.3f} deg')
            self.mrf_history.append({'sweep': sweep + 1, 'w1_sweep': w1_sweep,
                                     'w1_ebsd': w1_ebsd, 'n_updated': n_upd})

            converged = (w1_sweep < self.mrf_eps_convergence
                         and w1_ebsd < self.mrf_eps_quality)
            miso_prev = miso_curr

            if converged:
                print(f'  Converged at sweep {sweep+1} '
                      f'(W1_sweep={w1_sweep:.3f} < {self.mrf_eps_convergence}  '
                      f'W1_EBSD={w1_ebsd:.3f} < {self.mrf_eps_quality})')
                break
        else:
            print(f'  Stopped at max_sweeps={self.mrf_max_sweeps}  '
                  f'W1_EBSD={w1_ebsd:.3f} deg')

        return all_assigned

    def _run_one_sa_sweep(
            self,
            host_ids: set,
            nonhost_ids: set,
            host_pool: 'np.ndarray',
            full_pool: 'np.ndarray',
            neigh_graph: dict,
            all_assigned: dict,
            bin_centers: 'np.ndarray',
            density: 'np.ndarray',
            S: 'np.ndarray',
            temperature: float,
    ) -> tuple:
        """
        One SA sweep with the given temperature.

        Identical energy model to Gibbs (log P_EBSD density), but selection
        uses temperature-scaled softmax:
          T >> 0 : explores broadly (near-uniform sampling)
          T -> 0 : deterministic MAP (argmax of log-likelihood)
        """
        from upxo.xtalphy.crystal_orientation import quat_to_R_batch, _positive_w

        all_gids = list(host_ids | nonhost_ids)
        order = self._rng.permutation(all_gids).tolist()
        n_updated = 0

        R_host = quat_to_R_batch(host_pool)
        R_full = quat_to_R_batch(full_pool)
        bin_w = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else self._mdf_angle_max / 25

        # See _run_one_gibbs_sweep for why this hoist is exact and ~22x
        # faster than rebuilding the (24,N,K,3,3) SR tensor per grain.
        SP_host = np.einsum('sip,npj->snij', S, R_host)   # (24,Nh,3,3)
        SP_full = np.einsum('sip,npj->snij', S, R_full)   # (24,Nf,3,3)

        for gid in order:
            gid = int(gid)
            is_host = gid in host_ids
            pool_q = host_pool if is_host else full_pool
            pool_R = R_host   if is_host else R_full
            SP     = SP_host  if is_host else SP_full
            N_cand = len(pool_q)

            assigned_nbs = [
                (nb, all_assigned[nb])
                for nb in neigh_graph.get(gid, set())
                if nb in all_assigned]

            if not assigned_nbs:
                chosen = _positive_w(pool_q[int(self._rng.integers(0, N_cand))].copy())
            else:
                if not self.mrf_bilateral_symmetry:
                    # Batch all K neighbours simultaneously: one einsum over (N, K)
                    R_nbs  = quat_to_R_batch(
                        np.array([q for _, q in assigned_nbs]))        # (K,3,3)
                    tr     = np.einsum('snij,kij->snk', SP, R_nbs)     # (24,N,K)
                    cos_a  = np.clip((tr.max(axis=0)-1.0)*0.5,-1.0,1.0)# (N,K)
                    miso   = np.degrees(np.arccos(cos_a))               # (N,K)
                    bin_idx = np.clip(
                        (miso/bin_w).astype(int), 0, len(density)-1)
                    log_w = np.log(
                        np.clip(density[bin_idx], 1e-10, None)).sum(axis=1)  # (N,)
                else:
                    log_w = np.zeros(N_cand)
                    for nb_id, q_nb in assigned_nbs:
                        R_nb = quat_to_R_batch(q_nb[None])[0]
                        miso = self._compute_batch_miso(
                            pool_R, R_nb, S, use_bilateral=True)
                        bin_idx = np.clip(
                            (miso/bin_w).astype(int), 0, len(density)-1)
                        log_w += np.log(np.clip(density[bin_idx], 1e-10, None))

                if temperature < 1e-9:
                    chosen_idx = int(np.argmax(log_w))
                else:
                    log_w_t = log_w / temperature
                    log_w_t -= log_w_t.max()
                    probs = np.exp(log_w_t)
                    probs /= probs.sum()
                    chosen_idx = int(self._rng.choice(N_cand, p=probs))

                chosen = _positive_w(pool_q[chosen_idx].copy())

            old = all_assigned.get(gid)
            all_assigned[gid] = chosen
            if old is None or old.tobytes() != chosen.tobytes():
                n_updated += 1

        return all_assigned, n_updated

    def _assign_mrf_map(
            self,
            host_ids: set,
            nonhost_ids: set,
            host_pool: 'np.ndarray',
            full_pool: 'np.ndarray',
            neigh_graph: dict,
            all_assigned: dict,
    ) -> dict:
        """
        MRF MAP estimation via Simulated Annealing (Level 3b).

        Same energy model as mrf_gibbs (log P_EBSD density over all neighbours),
        but uses temperature-scaled argmax (SA) rather than stochastic sampling.

        Cooling schedule: geometric  T_k = T_start * (T_end/T_start)^(k/(K-1))
          where K = mrf_max_sweeps.  Final sweep runs at T = T_end (near-MAP).

        Convergence: same W1 criteria as Gibbs (eps_convergence, eps_quality).
        """
        from scipy.stats import wasserstein_distance
        from upxo.xtalphy.crystal_orientation import (
            get_cubic_ops_np, _positive_w, assign_orientations_conflict_free)

        if self._mdf_ref_angles is None or len(self._mdf_ref_angles) == 0:
            raise RuntimeError(
                'call build_adjacency_model(mdf_ref=mdf_merged) before mrf_map.')

        S = get_cubic_ops_np()
        n_bins = max(10, int(self._mdf_angle_max / self._mdf_bin_width))

        bin_centers, density = self._compute_pebsd_density(
            self._mdf_ref_angles, n_bins=n_bins)
        print(f'OrientationAssigner3D [L3b mrf_map]: '
              f'P_EBSD density built  n_bins={n_bins}  '
              f'{"KDE" if density.std() > 0 else "local_avg"}')

        # ── Initialise orientation field ────────────────────────────────────
        # Reuses the sweep's own fallback pool (self.fallback_quats, built once
        # in build_ebsd_pools from the actually-configured texture) instead of
        # a fresh, unseeded tops.synth_fcc_quats() call -- that call ran
        # unconditionally on every invocation (even when unused, for the
        # warm-start init modes below), wasting a full texture-generation pass
        # and, when actually used, drawing from a different default texture
        # set than the rest of the sweep.
        init_mode = self.mrf_init_mode
        fallback = self.fallback_quats

        if init_mode == 'random':
            for gid in host_ids:
                all_assigned[int(gid)] = _positive_w(
                    host_pool[int(self._rng.integers(0, len(host_pool)))].copy())
            for gid in nonhost_ids:
                all_assigned[int(gid)] = _positive_w(
                    full_pool[int(self._rng.integers(0, len(full_pool)))].copy())
            print(f'  Init: random draw from pools')
        elif init_mode == 'conflict_free':
            all_assigned, _ = assign_orientations_conflict_free(
                host_ids, neigh_graph, host_pool, all_assigned,
                max_retries=self.max_retries, rng=self._rng, cancel_event=self.cancel_event,
                fallback_pool=fallback)
            all_assigned, _ = assign_orientations_conflict_free(
                nonhost_ids, neigh_graph, full_pool, all_assigned,
                max_retries=self.max_retries, rng=self._rng, cancel_event=self.cancel_event,
                fallback_pool=fallback)
            print(f'  Init: conflict_free (Level 0)')
        elif init_mode in ('mdf_conditioned_pairs', 'mdf_analytical', 'paired_pool'):
            unassigned_host = host_ids - set(all_assigned.keys())
            unassigned_non  = nonhost_ids - set(all_assigned.keys())
            if unassigned_host:
                all_assigned, _, _ = self._assign_mdf_analytical(
                    unassigned_host, neigh_graph, host_pool, all_assigned)
            if unassigned_non:
                all_assigned, _, _ = self._assign_mdf_analytical(
                    unassigned_non, neigh_graph, full_pool, all_assigned)
            print(f'  Init: warm start from {init_mode} (or existing assignments)')
        else:
            raise ValueError(f'Unknown mrf_init_mode: "{init_mode}"')

        # ── SA sweeps ───────────────────────────────────────────────────────
        t_start  = self.mrf_sa_t_start
        t_end    = self.mrf_sa_t_end
        max_sw   = self.mrf_max_sweeps
        miso_prev = self._compute_boundary_miso(neigh_graph, all_assigned, S)
        ebsd_ref  = self._mdf_ref_angles
        w1_init   = (float(wasserstein_distance(miso_prev, ebsd_ref))
                     if len(miso_prev) > 0 else np.inf)
        print(f'  Warm-start:  n_pairs={len(miso_prev)}  '
              f'mean_miso={miso_prev.mean():.2f} deg  '
              f'W1_EBSD={w1_init:.3f} deg  '
              f'(EBSD ref: n={len(ebsd_ref)}  mean={np.mean(ebsd_ref):.2f} deg)')

        if w1_init <= self.mrf_eps_quality:
            print(f'  Warm-start already meets quality criterion '
                  f'(W1_EBSD={w1_init:.3f} <= {self.mrf_eps_quality} deg) '
                  f'-- skipping SA sweeps.')
            return all_assigned

        print(f'  Running up to {max_sw} SA sweeps  '
              f'T: {t_start:.3f} -> {t_end:.4f}  '
              f'(eps_conv={self.mrf_eps_convergence:.1f} deg  '
              f'eps_qual={self.mrf_eps_quality:.1f} deg)')

        w1_ebsd = np.inf
        for sweep in range(max_sw):
            self._check_cancelled()
            T = (t_start * (t_end / t_start) ** (sweep / max(max_sw - 1, 1)))

            all_assigned, n_upd = self._run_one_sa_sweep(
                host_ids, nonhost_ids, host_pool, full_pool,
                neigh_graph, all_assigned, bin_centers, density, S,
                temperature=T)

            miso_curr = self._compute_boundary_miso(neigh_graph, all_assigned, S)

            w1_sweep = (float(wasserstein_distance(miso_curr, miso_prev))
                        if len(miso_prev) > 0 and len(miso_curr) > 0 else np.inf)
            w1_ebsd  = (float(wasserstein_distance(miso_curr, ebsd_ref))
                        if len(miso_curr) > 0 else np.inf)

            print(f'  Sweep {sweep+1:>3d}  T={T:.4f}  updated={n_upd:>5d}  '
                  f'W1_sweep={w1_sweep:.3f} deg  W1_EBSD={w1_ebsd:.3f} deg')
            self.mrf_history.append({'sweep': sweep + 1, 'temperature': T,
                                     'w1_sweep': w1_sweep, 'w1_ebsd': w1_ebsd,
                                     'n_updated': n_upd})

            converged = (w1_sweep < self.mrf_eps_convergence
                         and w1_ebsd < self.mrf_eps_quality)
            miso_prev = miso_curr

            if converged:
                print(f'  Converged at sweep {sweep+1}  T={T:.4f}  '
                      f'(W1_sweep={w1_sweep:.3f} < {self.mrf_eps_convergence}  '
                      f'W1_EBSD={w1_ebsd:.3f} < {self.mrf_eps_quality})')
                break
        else:
            print(f'  Stopped at max_sweeps={max_sw}  T_final={T:.4f}  '
                  f'W1_EBSD={w1_ebsd:.3f} deg')

        return all_assigned

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
        from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats, expand_grain_quats_to_voxels
        if lgi is None:
            lgi = self.base.lgi
        if self.neigh_graph is None:
            raise RuntimeError('Call build_neighbour_graph() first.')
        quat_3d = expand_grain_quats_to_voxels(lgi, self.all_grain_orientations)
        neigh_list = {gid: list(ns) for gid, ns in self.neigh_graph.items()}
        return compute_mdf_from_quats(lgi, quat_3d, neigh_list,
                                      n_bins=n_bins, angle_range=angle_range)


# ── Orchestration helpers ────────────────────────────────────────────────
# Chain the methods above into reusable, higher-level sequences. Pure
# computation -- no UI dependency -- safe to call from a background thread.

def compute_ebsd_pure_parents_for_csl(rg, parent_info, csl_label):
    """EBSD orientations of "pure parent" grains for the given CSL label
    -- parent_info[csl_label]['pure_parents']: grains that appear as the
    LARGER member in every CSL pair they're part of (identify_parent_
    grains), i.e. grains that, in the real EBSD data, actually host a
    twin of this type and never are one themselves.

    Returns (ebsd_quats, pure_parent_ids), or (None, None) if csl_label
    isn't in parent_info or no pure-parent grain has a computed quaternion.
    """
    if csl_label not in parent_info:
        return None, None
    from upxo.xtalphy.crystal_orientation import grain_avg_quats, defdap_passive_to_active
    ebsd_gids_all, ebsd_q_all = grain_avg_quats(rg.lfi_ebsd, rg.quat_ebsd)
    gid2q = {int(g): ebsd_q_all[i] for i, g in enumerate(ebsd_gids_all)}
    pure_parent_ids = [g for g in sorted(int(x) for x in parent_info[csl_label]['pure_parents'])
                       if g in gid2q]
    if not pure_parent_ids:
        return None, None
    ebsd_quats = np.array([gid2q[g] for g in pure_parent_ids], dtype=np.float64)
    ebsd_quats = defdap_passive_to_active(ebsd_quats)
    return ebsd_quats, pure_parent_ids


def build_ebsd_merged_mdf(rg, parent_info, n_bins, angle_max):
    """Build (or reuse) the EBSD twin-merged parent-state MDF reference
    -- used both as build_adjacency_model's mdf_ref (Levels 2a/2b/3a/3b)
    and as a pre-twin MDF comparison target, since the pre-twin structure
    has no twins yet and comparing against the still-twinned full EBSD
    MDF would be an apples-to-oranges comparison."""
    from upxo.gsdataops.gid_ops import find_neighs2d
    from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats

    if getattr(rg, 'lfi_ebsd_merged', None) is None:
        print("EBSD twin-merged reference not yet built -- building it now...")
        rg.build_merged_ebsd_lfi(parent_info, plot=False)

    neigh_merged = find_neighs2d(rg.lfi_ebsd_merged.astype(np.int32), conn=4)
    return compute_mdf_from_quats(
        rg.lfi_ebsd_merged, rg.quat_ebsd, neigh_merged,
        n_bins=n_bins, angle_range=(0.0, angle_max))


def run_orientation_assignment(
        *, base, rg, parent_info, csl_label, ori_mode, rng_seed,
        connectivity, n_fallback, fallback_tc_info, fallback_apply_symmetry,
        mdf_n_bins, mdf_angle_max, pair_similarity_deg, max_retries,
        mrf_max_sweeps, mrf_eps_convergence, mrf_eps_quality, mrf_init_mode,
        mrf_kde_threshold, mrf_bilateral_symmetry, mrf_sa_t_start, mrf_sa_t_end,
        cancel_event=None, prebuilt_pools=None):
    """Builds and runs one full OrientationAssigner3D for the given mode
    and seed -- the standard build_neighbour_graph -> build_ebsd_pools ->
    build_adjacency_model -> assign_host_orientations ->
    assign_nonhost_orientations -> check_conflicts sequence.

    prebuilt_pools : optional (parent_pool, full_pool, fallback_quats)
        tuple. build_ebsd_pools() re-derives the EBSD parent/full pools
        from scratch (pure waste if identical across many calls, e.g.
        within one Optimize Mapping sweep) AND resamples a fresh,
        unseeded random fallback pool every time. When given, this skips
        build_ebsd_pools entirely and assigns the three pools directly,
        so a whole sweep (and any rerun of one of its candidates) shares
        one single, already-built set of pools. Omit for a standalone
        interactive run, where the fallback pool should stay freshly
        randomized on every call.
    """
    assigner = OrientationAssigner3D(
        base, orientation_assignment_mode=ori_mode,
        pair_similarity_deg=pair_similarity_deg,
        max_retries=max_retries,
        mrf_max_sweeps=mrf_max_sweeps,
        mrf_eps_convergence=mrf_eps_convergence,
        mrf_eps_quality=mrf_eps_quality,
        mrf_init_mode=mrf_init_mode,
        mrf_kde_threshold=mrf_kde_threshold,
        mrf_bilateral_symmetry=mrf_bilateral_symmetry,
        mrf_sa_t_start=mrf_sa_t_start,
        mrf_sa_t_end=mrf_sa_t_end,
        rng_seed=rng_seed,
        cancel_event=cancel_event)
    assigner.build_neighbour_graph(connectivity=connectivity)
    if prebuilt_pools is not None:
        assigner.parent_pool, assigner.full_pool, assigner.fallback_quats = prebuilt_pools
    else:
        assigner.build_ebsd_pools(
            ebsd_lfi=rg.lfi_ebsd, ebsd_quat=rg.quat_ebsd,
            parent_info=parent_info, csl_label=csl_label,
            n_fallback=n_fallback,
            fallback_tc_info=fallback_tc_info,
            fallback_apply_symmetry=fallback_apply_symmetry)
    if ori_mode != 'conflict_free':
        # Levels 2a/2b/3a/3b additionally condition on the real EBSD
        # misorientation distribution; Level 1 (paired_pool) only needs
        # the raw adjacency pair pools, not the MDF.
        mdf_ref = None
        if ori_mode in ('mdf_conditioned_pairs', 'mdf_analytical',
                        'mrf_gibbs', 'mrf_map'):
            mdf_ref = build_ebsd_merged_mdf(rg, parent_info, mdf_n_bins, mdf_angle_max)
        assigner.build_adjacency_model(rg, parent_info, csl_label, mdf_ref=mdf_ref)
    assigner.assign_host_orientations()
    assigner.assign_nonhost_orientations()
    assigner.check_conflicts()
    return assigner


def score_assigner_pole_figure(assigner, base, zi_ebsd, mask, pole_family, grid_points, unit_normalize):
    """Scores one freshly-built assigner's host-grain pole figure against
    the (already unit-normalized, if requested) EBSD reference grid."""
    from upxo.viz.xphy.pole_figure import PoleFigure
    from upxo.xtalphy.crystal_orientation import defdap_passive_to_active
    host_gids = sorted(g for g in base.host_grain_ids if g in assigner.all_grain_orientations)
    if not host_gids:
        return None
    quats = np.array([assigner.all_grain_orientations[g] for g in host_gids], dtype=np.float64)
    quats = defdap_passive_to_active(quats)
    pf = PoleFigure(quats, convention='quaternion')
    poles = pf._get_symmetric_poles(pole_family)
    _, _, zi, _ = pf._compute_mud_grid(poles, grid_points=grid_points)
    if unit_normalize:
        peak = float(np.nanmax(zi))
        if peak > 0 and np.isfinite(peak):
            zi = zi / peak
    delta = (zi - zi_ebsd)[mask]
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return None
    q25, q75 = np.percentile(delta, [25, 75])
    return {
        'min': float(np.min(delta)), 'max': float(np.max(delta)),
        'q25': float(q25), 'q75': float(q75), 'iqr': float(q75 - q25),
        'n_host': len(host_gids),
    }


def run_optimize_sweep(
        *, base, rg, parent_info, csl_label, selected,
        connectivity, n_fallback, fallback_tc_info, fallback_apply_symmetry,
        mdf_n_bins, mdf_angle_max, pair_similarity_deg, max_retries,
        mrf_max_sweeps, mrf_eps_convergence, mrf_eps_quality, mrf_init_mode,
        mrf_kde_threshold, mrf_bilateral_symmetry, mrf_sa_t_start, mrf_sa_t_end,
        pole_family, grid_points, unit_normalize, base_seed, cancel_event=None):
    """Sweeps orientation-assignment mode/seed combinations, scoring each
    against a real-EBSD reference pole figure, and returns every run's
    score plus the assigner objects for the 10 lowest-IQR (best-matching)
    runs.

    Builds the EBSD reference pole figure once, then for every (mode,
    iteration) pair in ``selected`` (an iterable of (mode, n) pairs) runs
    a full orientation assignment with a fresh seed via
    run_orientation_assignment, scores it against the reference via
    score_assigner_pole_figure, ranks all runs by IQR, and re-runs the
    top 10 to regenerate their full assigner objects for later
    inspection -- cheap relative to the full sweep, and avoids holding
    every single run's assigner in memory at once.

    Returns (records, top10): records is a list of per-run score dicts
    (all runs), top10 is a list of {**meta, 'assigner': assigner} for the
    10 lowest-IQR runs.
    """
    from upxo.xtalphy.crystal_orientation import AssignmentCancelled
    from upxo.viz.xphy.pole_figure import PoleFigure

    ebsd_quats, pure_parent_ids = compute_ebsd_pure_parents_for_csl(
        rg, parent_info, csl_label)
    if not pure_parent_ids:
        raise RuntimeError(
            "No EBSD Pure-Parents-of-Hosts orientations available for "
            f'CSL label "{csl_label}" -- identify parent grains first.')

    pf_ebsd = PoleFigure(ebsd_quats, convention='quaternion')
    poles_ebsd = pf_ebsd._get_symmetric_poles(pole_family)
    _, _, zi_ebsd, mask = pf_ebsd._compute_mud_grid(poles_ebsd, grid_points=grid_points)
    if unit_normalize:
        # np.nanmax over the FULL grid, not zi_ebsd[mask] -- matches
        # PoleFigure.plot_density_difference's own unit_normalize
        # convention exactly (cells outside the projection circle are
        # already NaN, so nanmax ignores them either way).
        peak = float(np.nanmax(zi_ebsd))
        if peak > 0 and np.isfinite(peak):
            zi_ebsd = zi_ebsd / peak
    print(f"Optimize Mapping: EBSD reference built "
          f"({len(pure_parent_ids)} features, {len(ebsd_quats)} orientations, "
          f"{{{pole_family}}}, grid={grid_points}, unit_normalize={unit_normalize})")

    # Build the EBSD parent/full pools and the synthetic fallback pool
    # ONCE, up front -- neither depends on mode or seed, so every sweep
    # iteration and the top-10 rerun below all share this same
    # OrientationAssigner3D.build_ebsd_pools() call instead of each
    # silently re-deriving the EBSD pools from scratch (pure waste) and
    # drawing a fresh, unseeded random fallback pool each time (see
    # run_orientation_assignment's prebuilt_pools docstring).
    pool_builder = OrientationAssigner3D(base, orientation_assignment_mode='conflict_free')
    pool_builder.build_ebsd_pools(
        ebsd_lfi=rg.lfi_ebsd, ebsd_quat=rg.quat_ebsd,
        parent_info=parent_info, csl_label=csl_label,
        n_fallback=n_fallback,
        fallback_tc_info=fallback_tc_info,
        fallback_apply_symmetry=fallback_apply_symmetry)
    prebuilt_pools = (pool_builder.parent_pool, pool_builder.full_pool, pool_builder.fallback_quats)

    records = []
    seed_counter = 0
    for mode, n in selected:
        print(f"--- {mode}: {n} iteration(s) ---")
        for it in range(n):
            if cancel_event is not None and cancel_event.is_set():
                raise AssignmentCancelled('Optimize Mapping stopped by user.')
            seed = base_seed + 100000 * (_VALID_ORIENT_MODES.index(mode) + 1) + seed_counter
            seed_counter += 1
            try:
                assigner = run_orientation_assignment(
                    base=base, rg=rg, parent_info=parent_info, csl_label=csl_label,
                    ori_mode=mode, rng_seed=seed, connectivity=connectivity,
                    n_fallback=n_fallback, fallback_tc_info=fallback_tc_info,
                    fallback_apply_symmetry=fallback_apply_symmetry,
                    mdf_n_bins=mdf_n_bins, mdf_angle_max=mdf_angle_max,
                    pair_similarity_deg=pair_similarity_deg, max_retries=max_retries,
                    mrf_max_sweeps=mrf_max_sweeps, mrf_eps_convergence=mrf_eps_convergence,
                    mrf_eps_quality=mrf_eps_quality, mrf_init_mode=mrf_init_mode,
                    mrf_kde_threshold=mrf_kde_threshold,
                    mrf_bilateral_symmetry=mrf_bilateral_symmetry,
                    mrf_sa_t_start=mrf_sa_t_start, mrf_sa_t_end=mrf_sa_t_end,
                    cancel_event=cancel_event, prebuilt_pools=prebuilt_pools)
            except AssignmentCancelled:
                raise
            except Exception as e:
                print(f"  iteration {it + 1} (seed={seed}): FAILED -- {e}")
                continue
            stats = score_assigner_pole_figure(
                assigner, base, zi_ebsd, mask, pole_family, grid_points, unit_normalize)
            if stats is None:
                print(f"  iteration {it + 1} (seed={seed}): no host orientations to score")
                continue
            records.append({
                'mode': mode, 'iteration': it + 1, 'seed': seed,
                # Pinned so a later replay reproduces this candidate
                # exactly as it was scored here, even if the live
                # csl_label/pole_family/grid_points/unit_normalize
                # settings have since changed.
                'csl_label': csl_label, 'pole_family': pole_family,
                'grid_points': grid_points, 'unit_normalize': unit_normalize,
                **stats,
            })
            print(f"  iteration {it + 1} (seed={seed}): "
                  f"IQR={stats['iqr']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}")

    if not records:
        raise RuntimeError("No successful runs -- see the log above for per-iteration errors.")

    ranked = sorted(records, key=lambda r: r['iqr'])
    top10_meta = ranked[:10]

    print(f"--- Re-running top {len(top10_meta)} for inspection ---")
    top10 = []
    for meta in top10_meta:
        if cancel_event is not None and cancel_event.is_set():
            raise AssignmentCancelled('Optimize Mapping stopped by user.')
        assigner = run_orientation_assignment(
            base=base, rg=rg, parent_info=parent_info, csl_label=csl_label,
            ori_mode=meta['mode'], rng_seed=meta['seed'], connectivity=connectivity,
            n_fallback=n_fallback, fallback_tc_info=fallback_tc_info,
            fallback_apply_symmetry=fallback_apply_symmetry,
            mdf_n_bins=mdf_n_bins, mdf_angle_max=mdf_angle_max,
            pair_similarity_deg=pair_similarity_deg, max_retries=max_retries,
            mrf_max_sweeps=mrf_max_sweeps, mrf_eps_convergence=mrf_eps_convergence,
            mrf_eps_quality=mrf_eps_quality, mrf_init_mode=mrf_init_mode,
            mrf_kde_threshold=mrf_kde_threshold,
            mrf_bilateral_symmetry=mrf_bilateral_symmetry,
            mrf_sa_t_start=mrf_sa_t_start, mrf_sa_t_end=mrf_sa_t_end,
            cancel_event=cancel_event, prebuilt_pools=prebuilt_pools)
        top10.append({**meta, 'assigner': assigner})
    print("Optimize Mapping: done.")
    return records, top10
