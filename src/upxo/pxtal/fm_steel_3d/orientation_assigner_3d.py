"""
OrientationAssigner3D: Worker class for assigning Kurdjumov-Sachs orientations.

This module contains the stateless orientation assignment logic using the
Kurdjumov-Sachs (KS) relationship between FCC (austenite) and BCC (ferrite).

Classes:
    OrientationAssigner3D: Orientation assignment worker.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import permutations, product
from scipy.spatial.transform import Rotation


def _build_cubic_sym_ops() -> Tuple[np.ndarray, np.ndarray]:
    """Return the 24 proper cubic symmetry operators and their transposes."""
    ops = []
    for p in permutations(range(3)):
        P = np.eye(3)[list(p)]
        for signs in product([-1, 1], repeat=3):
            S = P * np.array(signs)
            if abs(np.linalg.det(S) - 1) < 1e-6:
                ops.append(S)
    sym_ops = np.array(ops)           # (24, 3, 3)
    return sym_ops, sym_ops.transpose(0, 2, 1)


_CUBIC_SYM_OPS, _CUBIC_SYM_OPS_T = _build_cubic_sym_ops()


class OrientationAssigner3D:
    """
    Worker class for assigning crystal orientations to blocks via KS relationship.
    
    This is a stateless utility class. It receives PAG orientations and block data,
    and returns BCC orientations for each block following the Kurdjumov-Sachs
    relationship (24 variants grouped into 4 packets).
    
    All methods are instance methods to allow storing session config.
    """
    
    __slots__ = ('_verbosity', '_log_sink')

    def __init__(self, verbosity: int = 0, log_sink=None):
        """
        Initialize OrientationAssigner3D.

        Parameters
        ----------
        verbosity : int, optional
            Verbosity level. Default 0.
        log_sink : callable, optional
            callable(str) -> None. If given, progress/diagnostic messages are
            routed through it instead of bare print() -- lets a GUI page
            capture them into its own console widget the same way every
            other stage of this pipeline does (see _emit below). Without
            this, messages went straight to the real process stdout, which
            is invisible from inside a GUI console (they'd only show up in
            whatever terminal launched the GUI, not the app itself).
        """
        self._verbosity = int(verbosity)
        self._log_sink = log_sink

    def _emit(self, level: int, msg: str, component: str = 'ORI') -> None:
        if self._verbosity < int(level):
            return
        text = f"[{component}][L{level}] {msg}"
        if self._log_sink is not None:
            self._log_sink(text)
        else:
            print(text)

    @staticmethod
    def get_ks_rotations() -> np.ndarray:
        """
        Return the 24 Kurdjumov-Sachs rotation matrices.

        One canonical KS variant is constructed from the physical relationship
        {111}FCC || {110}BCC, <110>FCC || <111>BCC, then all 24 cubic symmetry
        operators are applied to generate the complete set of variants.

        Returns
        -------
        np.ndarray
            Shape (24, 3, 3) array of rotation matrices.
        """
        # Canonical KS variant: (111)FCC || (011)BCC,  [1-10]FCC || [1-11]BCC
        # Built by matching the two orthonormal frames that encode both conditions:
        #   FCC frame: { [111]/√3,  [1-10]/√2,  [11-2]/√6 }
        #   BCC frame: { [011]/√2,  [1-11]/√3,  [21-1]/√6 }
        # g_ks @ v_FCC = v_BCC  for each column pair  →  g_ks = B @ F.T
        v1f = np.array([1.,  1.,  1.]) / np.sqrt(3)
        v2f = np.array([1., -1.,  0.]) / np.sqrt(2)
        v3f = np.cross(v1f, v2f)

        v1b = np.array([0.,  1.,  1.]) / np.sqrt(2)
        v2b = np.array([1., -1.,  1.]) / np.sqrt(3)
        v3b = np.cross(v1b, v2b)

        F   = np.column_stack([v1f, v2f, v3f])
        B   = np.column_stack([v1b, v2b, v3b])
        g_ks = B @ F.T

        # All 24 proper cubic symmetry operators (det = +1 permutation-sign matrices)
        ops = []
        for p in permutations(range(3)):
            P = np.eye(3)[list(p)]
            for signs in product([-1, 1], repeat=3):
                S = P * np.array(signs)
                if abs(np.linalg.det(S) - 1) < 1e-6:
                    ops.append(S)

        sym_ops = np.array(ops)          # (24, 3, 3)
        return sym_ops @ g_ks            # (24, 3, 3)
    
    def assign_orientations_to_all_blocks(self,
                                           all_blocks: Dict[str, np.ndarray],
                                           clusters_dict: Dict[int, List[int]],
                                           pag_orientations: Dict[int, Tuple[float, float, float]],
                                           grain_to_plane_idx: Dict[int, int],
                                           ks_variant_selection: str = 'random_per_block',
                                           random_seed: Optional[int] = None
                                           ) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, int]]:
        """
        Assign BCC orientations to all blocks using the KS orientation relationship.

        Block name format: 'B_{pag_id}_{grain_id}_{local_counter}'.
        Each grain within a PAG is one packet, identified by grain_id (parts[2]).
        grain_to_plane_idx records which of the four {111}FCC habit planes (0-3)
        was assigned to each grain during block generation. That index maps directly
        to the packet index returned by group_ks_variants_into_packets, so each block
        draws its KS variant from the physically correct set of 6 variants.

        Parameters
        ----------
        all_blocks : dict
            {block_id: voxel_coords_array}.
        clusters_dict : dict
            {pag_id: [grain_id, ...]}.
        pag_orientations : dict
            {pag_id: (phi1_deg, Phi_deg, phi2_deg)}.
        grain_to_plane_idx : dict
            {grain_id: plane_index (0-3)} — produced by generate_blocks_for_all_pags.
        ks_variant_selection : str, optional
            How to pick a KS variant for each block within its packet.

            * ``'random_per_block'`` (default) — adjacency-aware greedy
              graph-colouring: adjacent blocks within the same packet are
              assigned different KS variants wherever possible, with a random
              choice among the free variants.
            * ``'deterministic'`` — always assign KS variant index 0 from the
              packet, ignoring adjacency.  Produces a fully reproducible,
              spatially uniform variant map useful for debugging and analytical
              checks of the KS geometry.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple of (block_orientations, block_to_variant_idx)
            block_orientations : dict
                {block_id: (phi1_deg, Phi_deg, phi2_deg)}.
            block_to_variant_idx : dict
                {block_id: variant_index (0-5)} — which of the 6 KS variants
                within its packet this block received.  Preserved (rather than
                discarded once used to compute the Euler angles above) so
                callers can build per-packet variant-balance statistics, or
                group blocks by shared variant, without re-deriving it from
                the orientation matrices.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        block_orientations: Dict[str, Tuple[float, float, float]] = {}
        block_to_variant_idx: Dict[str, int] = {}
        ks_rotations = self.get_ks_rotations()

        # Group block IDs by PAG (compute pag_R once per PAG)
        blocks_by_pag: Dict[int, List[str]] = {}
        for block_id in all_blocks.keys():
            pag_id = int(block_id.split('_')[1])
            blocks_by_pag.setdefault(pag_id, []).append(block_id)

        _n_pags_total = len(blocks_by_pag)
        self._emit(1, f"Assigning KS variants across {_n_pags_total} PAG(s), "
                      f"{len(all_blocks)} block(s), selection={ks_variant_selection!r}")

        _uneven_pag_ids: List[int] = []
        _first_uneven_sizes: Optional[Dict[int, int]] = None
        _total_forced_repeats = 0
        _pags_with_forced_repeats = 0
        # Detailed per-PAG lines for small runs stay readable; large runs (the
        # ones that actually take long enough to need progress) get periodic
        # percentage updates instead of one line per PAG, so the console
        # doesn't drown in thousands of near-identical lines at, say, 100^3
        # scale (several thousand PAGs).
        _per_pag_detail = _n_pags_total <= 50
        _progress_step = max(1, _n_pags_total // 20)

        for _pag_i, (pag_id, block_ids) in enumerate(blocks_by_pag.items(), start=1):
            pag_euler = pag_orientations.get(pag_id, (0.0, 0.0, 0.0))
            pag_R = self.cubic_euler_bunge_to_matrix_v1(
                np.array([pag_euler[0]]), np.array([pag_euler[1]]),
                np.array([pag_euler[2]]), degrees=True)
            if pag_R.ndim == 3:
                pag_R = pag_R[0]

            # Group the raw KS variants in the crystal frame into 4 packets.
            # Packet identity is a crystal-frame concept (which {111}FCC plane each
            # variant's {110}BCC normal maps to), so pag_R must NOT be applied here.
            raw_packets = self.group_ks_variants_into_packets(ks_rotations)

            sizes = {k: len(v) for k, v in raw_packets.items()}
            if set(sizes.values()) != {6}:
                _uneven_pag_ids.append(pag_id)
                if _first_uneven_sizes is None:
                    _first_uneven_sizes = sizes

            # Group blocks by packet index across all grains in this PAG.
            packet_block_map: Dict[int, List[str]] = {}
            for block_id in block_ids:
                gid  = int(block_id.split('_')[2])
                # `.get(gid, np.random.randint(4))` would evaluate the random
                # draw eagerly on every call (Python evaluates both .get()
                # arguments up front) even though gid is in the dict for
                # essentially every block -- wasting a draw and silently
                # perturbing the reproducible sequence almost every time.
                pidx = grain_to_plane_idx[gid] if gid in grain_to_plane_idx \
                    else np.random.randint(4)
                packet_block_map.setdefault(pidx, []).append(block_id)

            if _per_pag_detail:
                self._emit(1, f"  [{_pag_i}/{_n_pags_total}] PAG {pag_id}: "
                              f"{len(block_ids)} block(s) across {len(packet_block_map)} packet(s)")
            elif _pag_i % _progress_step == 0 or _pag_i == _n_pags_total:
                self._emit(1, f"  ...{_pag_i}/{_n_pags_total} PAGs processed "
                              f"({100 * _pag_i / _n_pags_total:.0f}%)")

            _pag_forced_repeats = 0
            for plane_idx, pp_block_ids in packet_block_map.items():
                packet_variants = raw_packets.get(plane_idx, [])
                if not packet_variants:
                    packet_variants = [v for vs in raw_packets.values() for v in vs]
                n_v = len(packet_variants)

                var_idx_map: Dict[str, int] = {}

                if ks_variant_selection == 'deterministic':
                    # Every block in this packet receives KS variant index 0.
                    # Fully reproducible regardless of random_seed; useful for
                    # debugging KS geometry with a known, spatially uniform map.
                    for bid in pp_block_ids:
                        var_idx_map[bid] = 0
                else:
                    # 'random_per_block' — adjacency-aware greedy graph-colouring.
                    # Build intra-packet block adjacency from voxel face-neighbours.
                    vox2bid: Dict[Tuple[int, int, int], str] = {}
                    for bid in pp_block_ids:
                        vox = all_blocks.get(bid)
                        if vox is not None and len(vox):
                            for row in vox:
                                vox2bid[(int(row[0]), int(row[1]), int(row[2]))] = bid
                    _face_off = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                    adjacency: Dict[str, set] = {bid: set() for bid in pp_block_ids}
                    for bid in pp_block_ids:
                        vox = all_blocks.get(bid)
                        if vox is None or not len(vox):
                            continue
                        for row in vox:
                            z, y, x = int(row[0]), int(row[1]), int(row[2])
                            for dz, dy, dx in _face_off:
                                nbid = vox2bid.get((z+dz, y+dy, x+dx))
                                if nbid is not None and nbid != bid:
                                    adjacency[bid].add(nbid)

                    # Process highest-degree blocks first to minimise forced repeats.
                    ordered = sorted(pp_block_ids,
                                     key=lambda b: len(adjacency[b]), reverse=True)
                    _forced_repeats = 0
                    for bid in ordered:
                        nbr_used = {var_idx_map[nb]
                                    for nb in adjacency[bid] if nb in var_idx_map}
                        free = [v for v in range(n_v) if v not in nbr_used]
                        if free:
                            var_idx_map[bid] = free[np.random.randint(len(free))]
                        else:
                            freq = [sum(1 for vi in var_idx_map.values() if vi == v)
                                    for v in range(n_v)]
                            var_idx_map[bid] = int(np.argmin(freq))
                            _forced_repeats += 1

                    if _forced_repeats:
                        _total_forced_repeats += _forced_repeats
                        _pag_forced_repeats += _forced_repeats
                        self._emit(
                            1, f"    PAG {pag_id}, packet {plane_idx}: {_forced_repeats} "
                               f"block(s) could not get a unique KS variant without an "
                               f"adjacency conflict (packet has {len(pp_block_ids)} blocks "
                               f"but only {n_v} variants) -- least-used variant used as "
                               f"fallback")
                        import warnings as _w
                        _w.warn(
                            f"PAG {pag_id}, packet {plane_idx}: {_forced_repeats} block(s) "
                            f"could not be assigned a unique KS variant without adjacency "
                            f"conflict (packet has {len(pp_block_ids)} blocks but only "
                            f"{n_v} variants). Least-used variant chosen as fallback.",
                            RuntimeWarning, stacklevel=4)

                for block_id in pp_block_ids:
                    v_idx       = var_idx_map[block_id]
                    raw_variant = packet_variants[v_idx]
                    variant_R   = pag_R @ raw_variant
                    block_orientations[block_id] = self.matrix_to_euler_bunge(
                        variant_R, degrees=True)
                    block_to_variant_idx[block_id] = v_idx

            if _pag_forced_repeats:
                _pags_with_forced_repeats += 1

        if self._verbosity >= 2:
            if not _uneven_pag_ids:
                self._emit(2, f"KS grouping: all {_n_pags_total} PAGs have balanced 4x6 packet structure")
            else:
                self._emit(2, f"KS grouping uneven in {len(_uneven_pag_ids)}/{_n_pags_total} PAGs "
                              f"(expected 6 per packet; first occurrence: {_first_uneven_sizes}); "
                              "assignment continued using available variants")

        self._emit(
            1, f"Done: {len(block_orientations)} block orientation(s) assigned across "
               f"{_n_pags_total} PAG(s) -- {_total_forced_repeats} forced-repeat fallback(s) "
               f"in {_pags_with_forced_repeats} PAG(s)"
        )
        return block_orientations, block_to_variant_idx
    
    @staticmethod
    def group_ks_variants_into_packets(
            variant_Rs: np.ndarray) -> Dict[int, List[np.ndarray]]:
        """
        Group 24 KS variants into 4 packets based on {110}BCC || {111}FCC planes.

        Each packet corresponds to one of the four {111}FCC habit planes.
        For each variant, the {110}BCC plane normal is transformed back into the
        FCC frame and matched to the closest {111}FCC plane via dot product.

        Parameters
        ----------
        variant_Rs : np.ndarray
            Shape (24, 3, 3) array of KS variant rotation matrices.

        Returns
        -------
        dict
            {packet_index (0-3): [list of variant rotation matrices]}.
        """
        fcc_111_planes = np.array([
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1,  1,  1],
            [-1, -1,  1],
        ], dtype=float) / np.sqrt(3)

        # For variant V_k = sym_ops[k] @ g_ks, the {111}FCC habit-plane normal
        # in the FCC crystal frame is sym_ops[k] @ v1f, which equals
        # V_k @ (g_ks.T @ v1f).  The orbit of v1f under the 24 cubic operators
        # has 8 elements {[±1,±1,±1]/√3}, each appearing 3 times; the abs dot
        # product merges antipodal pairs → exactly 6 variants per packet.
        v1f = np.array([1., 1., 1.]) / np.sqrt(3)
        v2f = np.array([1., -1., 0.]) / np.sqrt(2)
        v3f = np.cross(v1f, v2f)
        v1b = np.array([0., 1., 1.]) / np.sqrt(2)
        v2b = np.array([1., -1., 1.]) / np.sqrt(3)
        v3b = np.cross(v1b, v2b)
        _g_ks  = np.column_stack([v1b, v2b, v3b]) @ np.column_stack([v1f, v2f, v3f]).T
        bcc_ref = _g_ks.T @ v1f   # fixed BCC-frame reference; R @ bcc_ref gives habit normal

        packets: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}
        for R in variant_Rs:
            fcc_habit_normal = R @ bcc_ref
            packet_idx = int(np.argmax(np.abs(fcc_111_planes @ fcc_habit_normal)))
            packets[packet_idx].append(R)

        return packets
    
    def verify_block_orientations(self,
                                   block_orientations: Dict[str, Tuple[float, float, float]],
                                   n_samples: int = 2) -> Dict[str, any]:
        """
        Verify block orientations are physically plausible.
        
        Randomly samples pairs of blocks and computes misorientations.
        Expected values depend on their packet assignment and variant selection.
        
        Parameters
        ----------
        block_orientations : dict
            {block_id: (phi1, Phi, phi2)}.
        n_samples : int, optional
            Number of random block pairs to check. Default 2.
        
        Returns
        -------
        dict
            Verification results (e.g., misorientation statistics).
        """
        if len(block_orientations) < 2:
            return {'n_samples': 0, 'mean_misorientation_deg': 0}
        block_ids = list(block_orientations.keys())
        misoris = []
        for _ in range(min(n_samples, len(block_ids))):
            bid1, bid2 = np.random.choice(block_ids, 2, replace=False)
            angle, _, _ = self.cubic_misorientation_old1(
                block_orientations[bid1], block_orientations[bid2], degrees=True)
            misoris.append(angle)
        return {'n_samples': len(misoris),
                'mean_misorientation_deg': float(np.mean(misoris)) if misoris else 0,
                'misorientations': misoris}
    
    # ========== Euler angle and rotation utilities ==========
    
    @staticmethod
    def cubic_euler_bunge_to_matrix_v1(phi1: np.ndarray,
                                        Phi: np.ndarray,
                                        phi2: np.ndarray,
                                        degrees: bool = True) -> np.ndarray:
        """
        Convert Bunge Euler angles to rotation matrix (vectorized).
        
        Parameters
        ----------
        phi1, Phi, phi2 : np.ndarray
            Euler angles (1D arrays of same length, or scalars).
        degrees : bool, optional
            If True, input angles are in degrees. Default True.
        
        Returns
        -------
        np.ndarray
            Rotation matrices, shape (..., 3, 3).
        """
        if degrees:
            phi1 = np.radians(phi1)
            Phi = np.radians(Phi)
            phi2 = np.radians(phi2)
        phi1, Phi, phi2 = np.atleast_1d(phi1), np.atleast_1d(Phi), np.atleast_1d(phi2)
        n = len(phi1)
        c1, s1 = np.cos(phi1), np.sin(phi1)
        cP, sP = np.cos(Phi), np.sin(Phi)
        c2, s2 = np.cos(phi2), np.sin(phi2)
        R = np.zeros((n, 3, 3))
        R[:, 0, 0] = c1*c2 - s1*s2*cP
        R[:, 0, 1] = s1*c2 + c1*s2*cP
        R[:, 0, 2] = s2*sP
        R[:, 1, 0] = -c1*s2 - s1*c2*cP
        R[:, 1, 1] = -s1*s2 + c1*c2*cP
        R[:, 1, 2] = c2*sP
        R[:, 2, 0] = s1*sP
        R[:, 2, 1] = -c1*sP
        R[:, 2, 2] = cP
        return R[0] if n == 1 else R
    
    @staticmethod
    def matrix_to_euler_bunge(R: np.ndarray, degrees: bool = True) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Bunge Euler angles.
        
        Parameters
        ----------
        R : np.ndarray
            Rotation matrix, shape (3, 3).
        degrees : bool, optional
            If True, return angles in degrees. Default True.
        
        Returns
        -------
        tuple
            (phi1, Phi, phi2) in degrees (or radians).
        """
        # arccos already returns Phi canonically in [0, pi] -- do NOT reduce
        # it mod pi afterwards: that used to map the exact Phi=pi case down
        # to 0, silently turning a 180 degree tilt into "no tilt".
        Phi = np.arccos(np.clip(R[2, 2], -1, 1))
        if np.sin(Phi) > 1e-6:
            phi1 = np.arctan2(R[2, 0], -R[2, 1])
            phi2 = np.arctan2(R[0, 2], R[1, 2])
        else:
            # Gimbal lock (Phi = 0 or pi): only phi1+/-phi2 is determined, so
            # phi1 is fixed at 0 and phi2 absorbs the free combination. At
            # Phi=0, R[0,0]=R[1,1]=cos(phi1+phi2) and R[0,1]=-R[1,0]=
            # sin(phi1+phi2); at Phi=pi, R[0,0]=cos(phi1-phi2) and
            # R[0,1]=R[1,0]=sin(phi1-phi2). In both cases the correct
            # recovered phi2 is atan2(-R[1,0], R[0,0]) -- verified by direct
            # round-trip reconstruction. The previous atan2(R[1,0], R[0,0])
            # returned the negated angle, so matrix_to_euler_bunge(R) did not
            # reconstruct the input R for any orientation with Phi at (or
            # numerically touching) 0 or 180 degrees.
            phi1 = 0.0
            phi2 = np.arctan2(-R[1, 0], R[0, 0])
        phi1 = phi1 % (2 * np.pi)
        phi2 = phi2 % (2 * np.pi)
        if degrees:
            return float(np.degrees(phi1)), float(np.degrees(Phi)), float(np.degrees(phi2))
        return float(phi1), float(Phi), float(phi2)
    
    def disorientation_deg(
        self,
        ea1: Tuple[float, float, float],
        ea2: Tuple[float, float, float],
        degrees: bool = True,
    ) -> float:
        """
        Compute the cubic disorientation angle between two orientations.

        Applies full bicrystal cubic symmetry: H_i @ ΔR @ H_j^T over all
        24×24 = 576 combinations of the proper cubic operators (det = +1).
        Returns the minimum rotation angle (disorientation), which lies in
        [0°, 62.8°] for the cubic point group.

        Use this instead of cubic_misorientation_old1 whenever a physically
        correct disorientation is required (e.g. PAG-PAG or block-block plots).

        Parameters
        ----------
        ea1, ea2 : tuple of (phi1, Phi, phi2)
            Bunge ZXZ Euler angles.
        degrees : bool, optional
            If True, angles are in degrees. Default True.

        Returns
        -------
        float
            Disorientation angle in degrees.
        """
        R1 = self.cubic_euler_bunge_to_matrix_v1(
            np.array([ea1[0]]), np.array([ea1[1]]), np.array([ea1[2]]),
            degrees=degrees)
        R2 = self.cubic_euler_bunge_to_matrix_v1(
            np.array([ea2[0]]), np.array([ea2[1]]), np.array([ea2[2]]),
            degrees=degrees)
        R1 = R1[0] if R1.ndim == 3 else R1
        R2 = R2[0] if R2.ndim == 3 else R2

        sym_ops   = _CUBIC_SYM_OPS
        sym_ops_T = _CUBIC_SYM_OPS_T

        dR    = R1.T @ R2
        M1    = sym_ops @ dR                         # (24, 3, 3)
        M_all = M1[:, None] @ sym_ops_T[None, :]    # (24, 24, 3, 3)
        traces = M_all[:, :, 0, 0] + M_all[:, :, 1, 1] + M_all[:, :, 2, 2]
        return float(np.degrees(
            np.arccos(np.clip((traces.max() - 1.0) / 2.0, -1.0, 1.0))))

    def cubic_misorientation_old1(self,
                                   EA1: Tuple[float, float, float],
                                   EA2: Tuple[float, float, float],
                                   unique_tol_deg: float = 1e-4,
                                   degrees: bool = True) -> Tuple[float, float, float]:
        """
        Compute misorientation between two cubic orientations.
        
        Vectorized, fast implementation for cubic (m-3m) symmetry.
        
        Parameters
        ----------
        EA1, EA2 : tuple or ndarray
            Bunge Euler angles.
        unique_tol_deg : float, optional
            Tolerance for symmetry equivalence (degrees).
        degrees : bool, optional
            If True, angles and result in degrees.
        
        Returns
        -------
        tuple
            (misorientation_angle_deg, axis, angle_info).
        """
        R1 = self.cubic_euler_bunge_to_matrix_v1(
            np.array([EA1[0]]), np.array([EA1[1]]), np.array([EA1[2]]), degrees=degrees)
        R2 = self.cubic_euler_bunge_to_matrix_v1(
            np.array([EA2[0]]), np.array([EA2[1]]), np.array([EA2[2]]), degrees=degrees)
        if R1.ndim == 3: R1 = R1[0]
        if R2.ndim == 3: R2 = R2[0]
        
        dR = R1.T @ R2
        trace = np.trace(dR)
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = float(np.degrees(angle_rad))
        
        return angle_deg, dR, angle_deg


    def assign_pag_orientations_with_hagb(
        self,
        pag_ids: List[int],
        pag_neigh_map: Dict[int, List[int]],
        orientation_pool: Optional[List[Tuple[float, float, float]]] = None,
        hagb_threshold: float = 15.0,
        random_seed: Optional[int] = None,
        max_attempts: int = 1000,
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Assign FCC parent orientations to PAGs with a HAGB misorientation constraint.

        For each PAG, a candidate orientation is drawn from orientation_pool (or
        from the uniform SO(3) distribution if pool is None) and accepted only if
        its cubic misorientation with every already-assigned neighbouring PAG is at
        least hagb_threshold degrees.  This ensures all PAG-PAG boundaries are
        high-angle, which is physically required for martensite.

        PAGs are processed in a random order to avoid systematic bias.  If no
        valid candidate is found within max_attempts draws, the best candidate seen
        so far (highest minimum misorientation with neighbours) is used and a
        RuntimeWarning is emitted.

        Parameters
        ----------
        pag_ids : list of int
            PAG IDs to assign orientations to.
        pag_neigh_map : dict
            {pag_id: [neighbor_pag_id, ...]} adjacency map (from neigh_clid).
        orientation_pool : list of (phi1, Phi, phi2), optional
            Pre-computed candidate Euler angle triples in degrees (Bunge ZXZ).
            If None, candidates are drawn from the uniform SO(3) distribution.
        hagb_threshold : float, optional
            Minimum cubic misorientation in degrees required at every PAG-PAG
            interface.  Default 15.0°.
        random_seed : int, optional
            Seed for reproducibility.
        max_attempts : int, optional
            Maximum draws per PAG before falling back to the best candidate seen.
            Default 1000.

        Returns
        -------
        dict
            {pag_id: (phi1_deg, Phi_deg, phi2_deg)}.
        """
        import warnings as _warnings
        rng = np.random.default_rng(seed=random_seed)

        pag_order = list(pag_ids)
        rng.shuffle(pag_order)

        assigned: Dict[int, Tuple[float, float, float]] = {}

        def _rand_ori() -> Tuple[float, float, float]:
            return (
                float(rng.uniform(0.0, 360.0)),
                float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * rng.uniform(), -1.0, 1.0)))),
                float(rng.uniform(0.0, 360.0)),
            )

        pool = list(orientation_pool) if orientation_pool is not None else None
        n_pool = len(pool) if pool is not None else 0

        def _draw() -> Tuple[float, float, float]:
            if pool is not None:
                return pool[int(rng.integers(n_pool))]
            return _rand_ori()

        def _R_from_ea(ea: Tuple[float, float, float]) -> np.ndarray:
            R = self.cubic_euler_bunge_to_matrix_v1(
                np.array([ea[0]]), np.array([ea[1]]), np.array([ea[2]]), degrees=True)
            return R[0] if R.ndim == 3 else R

        def _misori_deg(R1: np.ndarray, R2: np.ndarray) -> float:
            dR    = R1.T @ R2
            M1    = _CUBIC_SYM_OPS @ dR
            M_all = M1[:, None] @ _CUBIC_SYM_OPS_T[None, :]
            traces = M_all[:, :, 0, 0] + M_all[:, :, 1, 1] + M_all[:, :, 2, 2]
            return float(np.degrees(np.arccos(np.clip((traces.max() - 1.0) / 2.0, -1.0, 1.0))))

        n_fallbacks = 0

        for pag_id in pag_order:
            nb_Rs = [
                _R_from_ea(assigned[nb])
                for nb in pag_neigh_map.get(pag_id, [])
                if nb in assigned
            ]

            if not nb_Rs:
                assigned[pag_id] = _draw()
                continue

            best_ea: Optional[Tuple[float, float, float]] = None
            best_min_mis = -1.0
            found = False

            for _ in range(max_attempts):
                candidate = _draw()
                R_cand = _R_from_ea(candidate)
                min_mis = min(_misori_deg(R_cand, R_nb) for R_nb in nb_Rs)
                if min_mis > best_min_mis:
                    best_min_mis = min_mis
                    best_ea = candidate
                if min_mis >= hagb_threshold:
                    found = True
                    break

            if not found:
                n_fallbacks += 1
                self._emit(
                    1, f"PAG {pag_id}: exhausted {max_attempts} attempts; "
                       f"best min-misorientation={best_min_mis:.2f}° "
                       f"(threshold={hagb_threshold:.1f}°), using best candidate",
                    component='ORI-HAGB',
                )

            assigned[pag_id] = best_ea

        if n_fallbacks:
            _warnings.warn(
                f"assign_pag_orientations_with_hagb: {n_fallbacks}/{len(pag_order)} PAG(s) "
                f"could not satisfy the {hagb_threshold:.1f}° HAGB threshold within "
                f"{max_attempts} attempts; best-available orientations were used.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._emit(
            1, f"Assigned {len(assigned)}/{len(pag_order)} PAG orientations "
               f"(threshold={hagb_threshold:.1f}°, fallbacks={n_fallbacks})",
            component='ORI-HAGB',
        )

        return assigned

    def assign_subblock_orientations(
        self,
        all_subblocks: Dict[str, np.ndarray],
        block_orientations: Dict[str, Tuple[float, float, float]],
        intrablock_ori_spread_deg: float = 2.0,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Assign orientations to sub-blocks as small perturbations of their parent block.

        Each sub-block receives the parent block orientation rotated by a random
        axis-angle perturbation sampled from Uniform(-spread, +spread) degrees.
        The rotation axis is drawn uniformly from the unit sphere.

        Sub-block names are expected in the form 'SB_{block_id}_{local}' so that
        the parent block_id can be recovered by stripping 'SB_' prefix and the
        trailing '_{local}' suffix.

        Parameters
        ----------
        all_subblocks : dict
            {subblock_id: voxel_coords_array} — only keys are used here.
        block_orientations : dict
            {block_id: (phi1, Phi, phi2)} in degrees (Bunge ZXZ).
        intrablock_ori_spread_deg : float, optional
            Half-width of the orientation spread in degrees. Each sub-block draws
            a perturbation angle from Uniform(-spread, +spread). Default 2.0.
        random_seed : int, optional
            Seed for reproducibility.

        Returns
        -------
        dict
            {subblock_id: (phi1, Phi, phi2)} in degrees.
        """
        rng = np.random.default_rng(seed=random_seed)
        subblock_orientations: Dict[str, Tuple[float, float, float]] = {}

        for sb_id in all_subblocks:
            # Recover parent block_id: strip 'SB_' prefix, drop final '_N' counter
            inner = sb_id[3:]                       # 'SB_B_pag_pkt_blk_N' → 'B_pag_pkt_blk_N'
            block_id = '_'.join(inner.split('_')[:-1])  # drop the trailing sub-block counter

            parent_ea = block_orientations.get(block_id)
            if parent_ea is None:
                # Block orientation unknown — assign zero perturbation (identity)
                subblock_orientations[sb_id] = (0.0, 0.0, 0.0)
                continue

            # Build parent rotation matrix
            R_block = Rotation.from_euler(
                'ZXZ', [parent_ea[0], parent_ea[1], parent_ea[2]], degrees=True)

            # Random unit axis on the sphere
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis) + 1e-15

            # Random angle within [-spread, +spread]
            angle_deg = rng.uniform(-intrablock_ori_spread_deg, intrablock_ori_spread_deg)

            # Perturbation applied in the sample frame (left-multiply)
            R_perturb = Rotation.from_rotvec(np.radians(angle_deg) * axis)
            R_sub = R_perturb * R_block

            ea = R_sub.as_euler('ZXZ', degrees=True)
            subblock_orientations[sb_id] = (float(ea[0]), float(ea[1]), float(ea[2]))

        return subblock_orientations


__all__ = ['OrientationAssigner3D']
