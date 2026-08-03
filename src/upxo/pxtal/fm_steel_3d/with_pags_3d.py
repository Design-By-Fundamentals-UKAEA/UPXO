"""
FMSteel3DWithPAGs: State class representing grain structure partitioned into PAGs.

This module contains the class for FM steel structures after PAG clustering.
It adds hierarchical structure (PAGs and their contained grains) to the base
grain structure.

Classes:
    FMSteel3DWithPAGs: Base + PAGs state class.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

from .phases_3d import PHASE_MARTENSITE, PHASE_RETAINED_AUSTENITE

# The 8 corner offsets of a unit voxel centred on an integer coordinate --
# used by _compute_pag_morphology to build a convex hull from actual voxel
# geometry rather than bare centre points (see that method's comment).
_CORNER_OFFSETS = np.array([[dx, dy, dz] for dx in (-0.5, 0.5)
                            for dy in (-0.5, 0.5) for dz in (-0.5, 0.5)])


class FMSteel3DWithPAGs:
    """
    FM steel grain structure with PAG (Prior Austenite Grain) hierarchy.
    
    Holds the base grain structure plus the computed PAG clustering.
    PAGs are groups of grains that will be further subdivided into packets
    and then blocks.
    
    This is an intermediate state in the pipeline. It is created by calling
    generate_pag_clusters() on a FMSteel3DBase instance, and transitions to
    FMSteel3DWithBlocks by calling generate_blocks().
    
    Attributes
    ----------
    _parent : FMSteel3DBase
        Reference to parent grain structure (read-only base state).
    
    clusters_dict : dict[int, list[int]]
        Maps PAG ID → list of grain IDs contained in that PAG.
    
    neigh_clid : dict[int, list[int]]
        Maps PAG ID → list of neighboring PAG IDs.
    
    pag_orientations : dict[int, tuple[float, float, float]]
        Will store parent FCC (austenite) orientations for each PAG.
        Filled by assign_pag_orientations() or by downstream methods.
        Format: {pag_id: (phi1_deg, Phi_deg, phi2_deg)}.
    
    isolated_grains : set[int]
        Grain IDs that do NOT participate in PAG clustering
        (controlled by pag_grain_fraction parameter). Kept for backward
        compatibility -- for PAGs produced via pag_technique_selector_3d.
        generate_pags(), this is now just the flattened grain membership of
        retained_austenite_pag_ids; the PAGs themselves are NOT removed from
        clusters_dict (see retained_austenite_pag_ids).

    retained_austenite_pag_ids : set[int]
        PAG IDs (present in clusters_dict) that were selected to remain
        untransformed retained austenite rather than proceed to block
        generation. Populated by pag_technique_selector_3d.generate_pags();
        empty for PAGs produced via the legacy FMSteel3DBase.
        generate_pag_clusters() pre-filter path (which excludes grains
        before clustering ever assigns them a PAG identity to preserve).
        A retained-austenite PAG still gets an orientation from
        assign_pag_orientations() like any other PAG; generate_blocks()
        skips these PAG IDs when slicing blocks.

    isolated_grain_orientations : dict[int, tuple[float, float, float]]
        FCC Bunge Euler orientations (degrees) for each isolated grain.
        Empty until assign_isolated_grain_orientations() is called.
        Format: {grain_id: (phi1_deg, Phi_deg, phi2_deg)}.
        Only meaningful for the legacy flat-isolated-grain case (grains in
        isolated_grains but not covered by retained_austenite_pag_ids);
        grains covered by a tracked retained-austenite PAG already have a
        correct, group-consistent orientation via pag_orientations and
        should not be re-assigned here.

    grain_to_pag_id : dict[int, int]
        Reverse lookup: grain_id -> pag_id.
        Derived from clusters_dict at construction time.

    grain_to_local_pkt_idx : dict[int, int]
        Maps grain_id -> 1-based ordinal of that grain within its PAG's
        grain list (i.e. its local packet index).  Ordinal follows the
        insertion order of clusters_dict[pag_id], which is determined by
        the clustering algorithm and is stable across the pipeline.
        Used to construct canonical elset names es.pck.{pag}.{pkt}.

    _random_seed : int
        Random seed used for this PAG generation.
    """
    
    __slots__ = (
        '_parent',
        'clusters_dict',
        'neigh_clid',
        'pag_orientations',
        'isolated_grains',
        'retained_austenite_pag_ids',
        'isolated_grain_orientations',
        '_target_pag_grain_fraction',
        '_retained_austenite_selection_info',
        '_random_seed',
        'grain_to_pag_id',
        'grain_to_local_pkt_idx',
        '_verbosity',
        '_log_sink',
    )

    def __init__(self,
                 parent,
                 clusters_dict: Dict[int, List[int]],
                 neigh_clid: Dict[int, List[int]],
                 pag_orientations: Optional[Dict[int, Tuple[float, float, float]]] = None,
                 isolated_grains: Optional[set] = None,
                 retained_austenite_pag_ids: Optional[set] = None,
                 target_pag_grain_fraction: Optional[float] = None,
                 retained_austenite_selection_info: Optional[Dict] = None,
                 random_seed: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 log_sink=None):
        """
        Initialize FMSteel3DWithPAGs.

        Typically called internally by FMSteel3DBase.generate_pag_clusters().
        Direct instantiation allowed but not recommended.

        Parameters
        ----------
        parent : FMSteel3DBase
            Parent grain structure instance.
        clusters_dict : dict
            PAG clustering: {pag_id: [grain_id, grain_id, ...]}.
        neigh_clid : dict
            PAG adjacency: {pag_id: [neighbor_pag_id, ...]}.
        isolated_grains : set, optional
            Grain IDs not included in PAGs. Default empty set.
        retained_austenite_pag_ids : set, optional
            PAG IDs (still present in clusters_dict) selected as retained
            austenite -- see class docstring. Default empty set.
        target_pag_grain_fraction : float, optional
            The pag_grain_fraction value the caller requested (fraction of
            material targeted to transform/cluster into PAGs; 1.0 minus this
            is the targeted retained-austenite fraction). Stored so the
            achieved fraction (computed from actual voxel counts) can be
            reported alongside what was actually asked for. None if not
            supplied (e.g. direct construction without going through
            generate_pags()).
        retained_austenite_selection_info : dict, optional
            The dict returned by pag_clustering_3d.select_isolated_units_
            with_tolerance (minus its 'selected' key), i.e.
            {'target_fraction', 'achieved_fraction', 'tolerance',
            'converged', 'n_attempts'} -- surfaced verbatim through
            get_pag_statistics() so a caller can see not just what fraction
            was achieved but whether it actually converged within tolerance
            and how many attempts that took. None if not supplied.
        pag_orientations : dict, optional
            Pre-computed PAG orientations. Default empty dict.
        random_seed : int, optional
            Random seed used for this instantiation.
        """
        self._parent = parent
        self.clusters_dict = clusters_dict
        self.neigh_clid = neigh_clid
        self.isolated_grains = isolated_grains or set()
        self.retained_austenite_pag_ids = retained_austenite_pag_ids or set()
        self._retained_austenite_selection_info = retained_austenite_selection_info
        self.pag_orientations = pag_orientations or {}
        self.isolated_grain_orientations: Dict[int, Tuple[float, float, float]] = {}
        self._target_pag_grain_fraction = target_pag_grain_fraction
        self._random_seed = random_seed
        self._verbosity = int(getattr(parent, '_verbosity', 0) if verbosity is None else verbosity)
        self._log_sink = getattr(parent, '_log_sink', None) if log_sink is None else log_sink

        self.grain_to_pag_id: Dict[int, int] = {
            gid: pag_id
            for pag_id, gids in clusters_dict.items()
            for gid in gids
        }
        self.grain_to_local_pkt_idx: Dict[int, int] = {
            gid: idx
            for pag_id, gids in clusters_dict.items()
            for idx, gid in enumerate(gids, start=1)
        }

    def _emit(self, level: int, msg: str, component: str = 'PAG') -> None:
        if self._verbosity < int(level):
            return
        text = f"[{component}][L{int(level)}] {msg}"
        if self._log_sink is not None:
            self._log_sink(text)
        else:
            print(text)
    
    # ========== Delegation properties (read-only access to parent) ==========
    
    @property
    def lgi(self) -> np.ndarray:
        """Labeled grain image from parent."""
        return self._parent.lgi
    
    @property
    def grain_locs(self) -> Dict[int, np.ndarray]:
        """Grain voxel coordinates from parent."""
        return self._parent.grain_locs
    
    @property
    def neigh_gid(self) -> Dict[int, List[int]]:
        """Grain neighbor relationships from parent."""
        return self._parent.neigh_gid
    
    @property
    def n_grains(self) -> int:
        """Total grains in parent structure."""
        return self._parent.n_grains
    
    @property
    def physical_dimensions(self):
        """Physical domain size from parent."""
        return self._parent.physical_dimensions
    
    @property
    def voxel_size(self) -> float:
        """Voxel size from parent."""
        return self._parent.voxel_size

    @property
    def units(self) -> str:
        """Physical unit string ('microns', 'mm', 'm') from base grain structure."""
        return self._parent.units

    def get_pag_phase(self, pag_id: int) -> int:
        """Phase id (see phases_3d.py) for a given PAG.

        Returns PHASE_RETAINED_AUSTENITE if pag_id is in
        retained_austenite_pag_ids, else PHASE_MARTENSITE. Does not
        validate that pag_id actually exists in clusters_dict.
        """
        return (PHASE_RETAINED_AUSTENITE if pag_id in self.retained_austenite_pag_ids
                else PHASE_MARTENSITE)

    @property
    def n_retained_austenite_pags(self) -> int:
        """Number of PAGs tagged as retained austenite (still counted in
        n_pags/clusters_dict -- these just never proceed to block
        generation)."""
        return len(self.retained_austenite_pag_ids)

    @property
    def n_transformed_pags(self) -> int:
        """Number of PAGs that will proceed to block generation (i.e. not
        retained austenite)."""
        return len(self.clusters_dict) - len(self.retained_austenite_pag_ids)
    
    # ========== Pipeline continuation ==========
    
    def assign_pag_orientations(
        self,
        pag_ori_mode: str = 'random',
        pag_ori_params: Optional[Dict] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Assign FCC Bunge Euler orientations to PAGs in-place.

        Parameters
        ----------
        pag_ori_mode : str
            ``'random'`` — uniform SO(3) (Phi sampled via arccos so the area
            element is correct).
            ``'hagb_constrained'`` — random SO(3) with minimum HAGB constraint
            between neighbouring PAGs; pass ``hagb_threshold`` (float, deg),
            ``max_attempts`` (int), and optionally ``orientation_pool``
            (list of (phi1,Phi,phi2) tuples) in ``pag_ori_params``.
            ``'textured'`` — sample PAG orientations from a synthetic FCC
            texture defined by named texture components; pass ``tc_info``
            (dict of component → [volume_fraction, [phi1,Phi,phi2]]) and
            optionally ``N`` (pool size, default 5000) and
            ``hagb_constraint`` (bool, default False; if True, the texture
            pool is passed to the HAGB-constrained assigner).
            ``'fixed'`` — all PAGs receive the same orientation; pass
            ``euler_angles=(phi1,Phi,phi2)`` in ``pag_ori_params``.
        pag_ori_params : dict, optional
            Extra parameters for the chosen mode (see above).
        random_seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        None
            Results stored in ``self.pag_orientations``.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        pag_ori_params = pag_ori_params or {}

        if pag_ori_mode == 'random':
            self.pag_orientations = {
                pag_id: (np.random.uniform(0, 360),
                         float(np.degrees(np.arccos(1.0 - 2.0 * np.random.uniform()))),
                         np.random.uniform(0, 360))
                for pag_id in self.clusters_dict.keys()
            }
        elif pag_ori_mode == 'hagb_constrained':
            from .orientation_assigner_3d import OrientationAssigner3D
            _ori = OrientationAssigner3D(verbosity=self._verbosity, log_sink=self._log_sink)
            self.pag_orientations = _ori.assign_pag_orientations_with_hagb(
                pag_ids=list(self.clusters_dict.keys()),
                pag_neigh_map=self.neigh_clid,
                orientation_pool=pag_ori_params.get('orientation_pool', None),
                hagb_threshold=float(pag_ori_params.get('hagb_threshold', 15.0)),
                random_seed=random_seed,
                max_attempts=int(pag_ori_params.get('max_attempts', 1000)),
            )
        elif pag_ori_mode == 'textured':
            tc_info = pag_ori_params.get('tc_info')
            if tc_info is None:
                raise ValueError(
                    "'textured' mode requires 'tc_info' in pag_ori_params. "
                    "Example: {'copper': [0.45, [90.0, 35.0, 45.0]], "
                    "'brass': [0.55, [35.0, 45.0, 0.0]]}"
                )
            N = int(pag_ori_params.get('N', 5000))
            from upxo.xtalphy.texops import tops as _tops
            tex = _tops.synth_fcc(N=N, tc_info=tc_info,
                                  n_tex_instances=1, n_sampling_instances=1)
            tc_stacks = (tex.tex['tex_instance.1']
                            ['sampling_instances']['ossi.1']['tc_ori_stacks'])
            pool_euler = np.vstack(list(tc_stacks.values()))  # (M, 3) ZXZ degrees
            pool_list = [tuple(row) for row in pool_euler]
            if pag_ori_params.get('hagb_constraint', False):
                from .orientation_assigner_3d import OrientationAssigner3D
                _ori = OrientationAssigner3D(verbosity=self._verbosity, log_sink=self._log_sink)
                self.pag_orientations = _ori.assign_pag_orientations_with_hagb(
                    pag_ids=list(self.clusters_dict.keys()),
                    pag_neigh_map=self.neigh_clid,
                    orientation_pool=pool_list,
                    hagb_threshold=float(pag_ori_params.get('hagb_threshold', 15.0)),
                    random_seed=random_seed,
                    max_attempts=int(pag_ori_params.get('max_attempts', 1000)),
                )
            else:
                rng = np.random.default_rng(random_seed)
                indices = rng.integers(0, len(pool_list), size=len(self.clusters_dict))
                self.pag_orientations = {
                    pag_id: pool_list[int(i)]
                    for pag_id, i in zip(self.clusters_dict.keys(), indices)
                }
        elif pag_ori_mode == 'fixed':
            default_ori = pag_ori_params.get('euler_angles', (0.0, 0.0, 0.0))
            self.pag_orientations = {
                pid: default_ori for pid in self.clusters_dict.keys()
            }
        else:
            raise ValueError(
                f"Unknown pag_ori_mode: {pag_ori_mode!r}. "
                "Choose 'random', 'hagb_constrained', 'textured', or 'fixed'."
            )
        self._emit(
            1,
            f"PAG orientations assigned: mode={pag_ori_mode!r}, "
            f"n_pags={len(self.pag_orientations)}",
            component='ORI',
        )

    def generate_blocks(self,
                        block_thickness_range: Tuple[float, float] = (2.0, 5.0),
                        random_seed: Optional[int] = None,
                        block_slab_connectivity: int = 26) -> 'FMSteel3DWithBlocks':
        """
        Slice each PAG into martensitic blocks.

        PAG orientations must be assigned before calling this method (via
        ``assign_pag_orientations``).  If ``self.pag_orientations`` is empty
        a RuntimeWarning is issued and random SO(3) orientations are used as
        a fallback so the pipeline can still complete.

        Parameters
        ----------
        block_thickness_range : tuple of (float, float), optional
            (lower, upper) bounds for block thickness in physical units.
            Each packet independently draws its thickness uniformly from this
            range.  Default (2.0, 5.0).
        random_seed : int, optional
            Random seed for block slicing.
        block_slab_connectivity : int, optional
            CC3D connectivity for within-slab connected-component labelling
            (6, 18, or 26).  Default 26.

        Returns
        -------
        FMSteel3DWithBlocks
            New instance with computed block hierarchy.

        Notes
        -----
        Does not modify self; returns a new instance.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self._emit(
            1,
            f"Generating blocks (thickness={block_thickness_range[0]:.3f}-{block_thickness_range[1]:.3f} vox)",
            component='BLOCK',
        )

        if not self.pag_orientations:
            import warnings as _w
            _w.warn(
                "generate_blocks() called before assign_pag_orientations(); "
                "falling back to random uniform SO(3) PAG orientations.",
                RuntimeWarning, stacklevel=2,
            )
            self.assign_pag_orientations('random', random_seed=random_seed)
        
        from .block_generator_3d import BlockGenerator3D
        from .with_blocks_3d import FMSteel3DWithBlocks

        # Retained-austenite PAGs never transform -- they keep the FCC
        # orientation assign_pag_orientations() already gave them and never
        # host blocks. Only feed the transforming PAGs to BlockGenerator3D.
        if self.retained_austenite_pag_ids:
            transforming_clusters = {
                pid: gids for pid, gids in self.clusters_dict.items()
                if pid not in self.retained_austenite_pag_ids
            }
            self._emit(
                1,
                f"Skipping block generation for {len(self.retained_austenite_pag_ids)} "
                f"retained-austenite PAG(s); {len(transforming_clusters)} PAG(s) transform.",
                component='BLOCK',
            )
        else:
            transforming_clusters = self.clusters_dict

        block_gen = BlockGenerator3D()
        all_blocks, grain_to_blocks_map, grain_to_plane_idx, block_slicing_normals = \
            block_gen.generate_blocks_for_all_pags(
                clusters_dict=transforming_clusters, grain_locs=self.grain_locs,
                pag_orientations=self.pag_orientations,
                block_thickness_range=block_thickness_range,
                random_seed=random_seed,
                slab_connectivity=block_slab_connectivity,
            )

        if all_blocks:
            block_sizes = [len(v) for v in all_blocks.values()]
            self._emit(
                1,
                f"Generated {len(all_blocks)} blocks across {len(grain_to_blocks_map)} packets",
                component='BLOCK',
            )
            self._emit(
                2,
                f"voxels/block min={min(block_sizes)}, max={max(block_sizes)}, mean={np.mean(block_sizes):.1f}",
                component='BLOCK',
            )
        else:
            self._emit(1, "Generated 0 blocks", component='BLOCK')

        return FMSteel3DWithBlocks(parent=self, all_blocks=all_blocks,
                                   grain_to_blocks_map=grain_to_blocks_map,
                                   grain_to_plane_idx=grain_to_plane_idx,
                                   block_slicing_normals=block_slicing_normals,
                                   slicing_planes={}, random_seed=random_seed,
                                   verbosity=self._verbosity, log_sink=self._log_sink)
    
    def assign_isolated_grain_orientations(
        self,
        mode: str = 'random',
        hagb_threshold: float = 15.0,
        max_attempts: int = 1000,
        orientation_pool: Optional[List[Tuple[float, float, float]]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Assign FCC Bunge Euler orientations to isolated (unclustered) grains.

        Isolated grains are those excluded from PAG clustering (see
        ``isolated_grains`` attribute).  Because they were never part of a PAG
        they carry no crystallographic orientation after ``generate_blocks``;
        this method assigns one, ensuring HAGB compatibility with all already-
        assigned neighbours (both PAGs and other isolated grains).

        Only processes grains in ``isolated_grains`` that are NOT covered by
        a tracked ``retained_austenite_pag_ids`` entry -- those already have
        a correct, group-consistent orientation from ``assign_pag_orientations``
        (every grain in a retained-austenite PAG shares that PAG's single
        orientation, the same as a transformed PAG's grains do) and must not
        be overwritten with a separately-drawn one here. This method exists
        for the legacy case: grains excluded before clustering ever assigned
        them a PAG identity (FMSteel3DBase.generate_pag_clusters's pre-filter
        path), which have no orientation to inherit at all.

        Parameters
        ----------
        mode : str
            ``'random'`` — draw candidate orientations from uniform SO(3).
            ``'textured'`` — draw candidates from ``orientation_pool`` (which
            must be supplied; typically built from FCCTexture.generate_euler).
        hagb_threshold : float
            Minimum cubic disorientation (degrees) required between this grain
            and each already-assigned neighbour.  Default 15.0°.
        max_attempts : int
            Maximum candidate draws per grain.  If the threshold is never met,
            the best-seen orientation (largest minimum misorientation with
            neighbours) is used and a RuntimeWarning is issued.  Default 1000.
        orientation_pool : list of (phi1, Phi, phi2), optional
            Pre-generated pool of Bunge Euler angles (degrees) for textured
            mode.  Ignored when mode='random'.  Must be non-empty.
        random_seed : int or None
            RNG seed for reproducibility.

        Returns
        -------
        None
            Results stored in ``self.isolated_grain_orientations``.

        Raises
        ------
        ValueError
            If mode='textured' and orientation_pool is None or empty.
        """
        covered_by_retained_pag: set = set()
        for pid in self.retained_austenite_pag_ids:
            covered_by_retained_pag.update(self.clusters_dict.get(pid, []))
        leftover_isolated = self.isolated_grains - covered_by_retained_pag

        if not leftover_isolated:
            self._emit(
                1,
                "No isolated grains need assignment — either none exist, or all are "
                "covered by a tracked retained-austenite PAG (already oriented via "
                "assign_pag_orientations).",
                component='ISO',
            )
            return

        if mode == 'textured' and not orientation_pool:
            raise ValueError(
                "assign_isolated_grain_orientations: mode='textured' requires a non-empty "
                "orientation_pool.  Generate one with FCCTexture.generate_euler() and pass it here."
            )

        import warnings as _warnings
        from .orientation_assigner_3d import (
            OrientationAssigner3D,
            _CUBIC_SYM_OPS,
            _CUBIC_SYM_OPS_T,
        )

        rng = np.random.default_rng(seed=random_seed)
        _oa = OrientationAssigner3D()

        # ---- Build grain-level orientation lookup ----
        # Clustered grains carry their PAG's orientation.
        grain_ori_lookup: Dict[int, Tuple[float, float, float]] = {}
        for pag_id, gids in self.clusters_dict.items():
            pag_ea = self.pag_orientations.get(pag_id)
            if pag_ea is not None:
                for gid in gids:
                    grain_ori_lookup[gid] = pag_ea
        # Isolated grains are added incrementally below as they are assigned.

        # ---- Pool / draw helpers ----
        pool_arr = list(orientation_pool) if orientation_pool is not None else []
        n_pool = len(pool_arr)

        def _rand_ea() -> Tuple[float, float, float]:
            return (
                float(rng.uniform(0.0, 360.0)),
                float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * float(rng.uniform()), -1.0, 1.0)))),
                float(rng.uniform(0.0, 360.0)),
            )

        def _draw() -> Tuple[float, float, float]:
            if pool_arr:
                return tuple(pool_arr[int(rng.integers(n_pool))])
            return _rand_ea()

        def _rot_matrix(ea: Tuple[float, float, float]) -> np.ndarray:
            return _oa.cubic_euler_bunge_to_matrix_v1(
                np.array([ea[0]]), np.array([ea[1]]), np.array([ea[2]]), degrees=True
            )

        def _misori_deg(R1: np.ndarray, R2: np.ndarray) -> float:
            dR = R1.T @ R2
            M_all = (_CUBIC_SYM_OPS @ dR)[:, None] @ _CUBIC_SYM_OPS_T[None, :]
            tr = M_all[:, :, 0, 0] + M_all[:, :, 1, 1] + M_all[:, :, 2, 2]
            return float(np.degrees(np.arccos(np.clip((float(tr.max()) - 1.0) / 2.0, -1.0, 1.0))))

        # ---- Process isolated grains in random order ----
        iso_list = list(leftover_isolated)
        rng.shuffle(iso_list)

        n_fallbacks = 0
        result: Dict[int, Tuple[float, float, float]] = {}

        for gid in iso_list:
            # Collect rotation matrices of all already-resolved neighbours.
            nb_Rs = []
            for nb in self.neigh_gid.get(gid, []):
                ea = grain_ori_lookup.get(nb)
                if ea is not None:
                    nb_Rs.append(_rot_matrix(ea))

            if not nb_Rs:
                # No oriented neighbours yet — draw freely, no constraint to check.
                ea_chosen = _draw()
                result[gid] = ea_chosen
                grain_ori_lookup[gid] = ea_chosen
                continue

            best_ea: Optional[Tuple[float, float, float]] = None
            best_min_mis = -1.0
            satisfied = False

            for _ in range(max_attempts):
                candidate = _draw()
                R_cand = _rot_matrix(candidate)
                min_mis = min(_misori_deg(R_cand, R_nb) for R_nb in nb_Rs)
                if min_mis > best_min_mis:
                    best_min_mis = min_mis
                    best_ea = candidate
                if min_mis >= hagb_threshold:
                    satisfied = True
                    break

            if not satisfied:
                n_fallbacks += 1

            result[gid] = best_ea
            grain_ori_lookup[gid] = best_ea

        self.isolated_grain_orientations = result

        if n_fallbacks:
            _warnings.warn(
                f"assign_isolated_grain_orientations: {n_fallbacks}/{len(iso_list)} isolated "
                f"grain(s) could not satisfy the {hagb_threshold:.1f}° HAGB threshold within "
                f"{max_attempts} attempt(s); best-available orientation used for each.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._emit(
            1,
            f"Assigned orientations to {len(result)}/{len(iso_list)} isolated grains "
            f"(mode={mode!r}, fallbacks={n_fallbacks})",
            component='ISO',
        )

    def _uncovered_isolated_grains(self) -> set:
        """Isolated grains with NO orientation resolvable at all right now --
        neither via a retained-austenite PAG's own orientation (requires
        assign_pag_orientations() to have run) nor via an existing
        isolated_grain_orientations entry (requires
        assign_isolated_grain_orientations() to have run for the legacy
        leftover case). Used by ensure_isolated_grain_orientations and by
        the phase-aware IPF query to know which grains still need fixing up.
        """
        if not self.isolated_grains:
            return set()
        covered_by_pag: set = set()
        for pid in self.retained_austenite_pag_ids:
            if self.pag_orientations.get(pid) is not None:
                covered_by_pag.update(self.clusters_dict.get(pid, []))
        already_assigned = set(self.isolated_grain_orientations.keys())
        return self.isolated_grains - covered_by_pag - already_assigned

    def ensure_isolated_grain_orientations(self, random_seed: Optional[int] = None) -> None:
        """Guarantee every isolated grain has a resolvable crystallographic
        orientation -- non-negotiable: no feature other than voids may go
        unassigned.

        Most isolated grains already satisfy this automatically: a
        retained-austenite PAG produced by generate_pags() gets its
        orientation the same way any PAG does, via assign_pag_orientations()
        (see FMSteel3DWithPAGs class docstring). The only grains that can
        ever be missing one are legacy leftover isolated grains (from
        FMSteel3DBase.generate_pag_clusters()'s pre-filter path, which never
        gave them a PAG identity to inherit from) that nobody has yet run
        assign_isolated_grain_orientations() for.

        This method closes that gap automatically: if any isolated grain is
        found with no orientation available from either source, it
        auto-assigns via assign_isolated_grain_orientations('random', ...)
        for the uncovered ones, with a warning, rather than silently leaving
        a real microstructural feature unoriented. Idempotent and cheap to
        call repeatedly -- a no-op once every isolated grain is covered, and
        never re-draws an orientation that already exists.

        Called automatically by MeshExporter3D and by the phase-aware IPF
        query before either one reads isolated_grain_orientations, so this
        rarely needs to be called directly.
        """
        uncovered = self._uncovered_isolated_grains()
        if not uncovered:
            return
        import warnings as _warnings
        _warnings.warn(
            f"{len(uncovered)} isolated grain(s) had no crystallographic orientation "
            "available (neither a retained-austenite PAG orientation nor an "
            "isolated_grain_orientations entry) -- auto-assigning via "
            "assign_isolated_grain_orientations('random') so no non-void feature "
            "goes unassigned. Call assign_isolated_grain_orientations() yourself "
            "beforehand for more control (e.g. a textured pool or HAGB threshold).",
            RuntimeWarning, stacklevel=2,
        )
        self.assign_isolated_grain_orientations(mode='random', random_seed=random_seed)

    def get_isolated_grain_orientation(self, gid: int) -> Optional[Tuple[float, float, float]]:
        """Resolve one isolated grain's FCC orientation from whichever
        source actually has it: its retained-austenite PAG's orientation
        (grain_to_pag_id -> pag_orientations) if tracked, else
        isolated_grain_orientations for the legacy leftover case. Returns
        None if neither source has an entry (call
        ensure_isolated_grain_orientations() first to guarantee that never
        happens)."""
        pag_id = self.grain_to_pag_id.get(gid)
        if pag_id is not None and pag_id in self.retained_austenite_pag_ids:
            ea = self.pag_orientations.get(pag_id)
            if ea is not None:
                return ea
        return self.isolated_grain_orientations.get(gid)

    def get_pag_statistics(self, histogram_cap: int = 10, compute_morphology: bool = False) -> Dict:
        """Compute statistics on PAG structure.

        Parameters
        ----------
        histogram_cap : int, optional
            Grain-count histogram is broken out per-count from 1 up to and
            including this value; any PAG with more grains than this is
            bucketed into a single "{cap+1}+" key. Default 10.
        compute_morphology : bool, optional
            Also compute per-PAG aspect ratio and solidity (a convex hull
            per PAG -- not free, so opt-in). Default False.

        Notes
        -----
        Histogram keys are strings (e.g. "1", "2", ..., "11+"), not ints —
        this dict is persisted verbatim into shared_state/session JSON, and
        JSON object keys are always strings; using string keys natively
        avoids an int-vs-str mismatch after a save/reload round trip.
        """
        grains_per_pag = [len(g) for g in self.clusters_dict.values()]
        histogram = {str(k): 0 for k in range(1, histogram_cap + 1)}
        overflow_key = f'{histogram_cap + 1}+'
        histogram[overflow_key] = 0
        for c in grains_per_pag:
            if c <= histogram_cap:
                histogram[str(c)] += 1
            else:
                histogram[overflow_key] += 1
        n_pags = len(grains_per_pag)
        frac_single_grain_pags = (histogram['1'] / n_pags) if n_pags else 0.0
        if grains_per_pag:
            stats = {
                'n_pags': n_pags,
                'n_transformed_pags': self.n_transformed_pags,
                'n_retained_austenite_pags': self.n_retained_austenite_pags,
                'min_grains_per_pag': min(grains_per_pag),
                'max_grains_per_pag': max(grains_per_pag),
                'mean_grains_per_pag': float(np.mean(grains_per_pag)),
                'n_isolated_grains': len(self.isolated_grains),
                'total_clustered_grains': sum(grains_per_pag),
                'grains_per_pag_histogram': histogram,
                'frac_single_grain_pags': frac_single_grain_pags,
            }
        else:
            stats = {'n_pags': 0, 'n_transformed_pags': 0, 'n_retained_austenite_pags': 0,
                     'min_grains_per_pag': 0, 'max_grains_per_pag': 0,
                     'mean_grains_per_pag': 0.0, 'n_isolated_grains': len(self.isolated_grains),
                     'total_clustered_grains': 0,
                     'grains_per_pag_histogram': histogram,
                     'frac_single_grain_pags': 0.0}

        stats.update(self._get_size_and_degree_statistics())
        stats['packet_size_balance'] = self._get_packet_balance_statistics()
        if compute_morphology:
            stats['pag_morphology'] = self._compute_pag_morphology()
        return stats

    def _get_packet_balance_statistics(self) -> Dict:
        """How close each PAG's constituent packets are to an equal-size
        split, aggregated across every PAG with 2+ packets (a single-packet
        PAG has no "balance" to speak of, so it's excluded rather than
        padding the average with a trivial zero).

        Meaningful for both techniques, not just Technique B's deliberate
        within-grain splitting: Technique A groups separately-grown grains
        of possibly very different sizes into a PAG with no attempt to
        balance them at all, so this can reveal a very unequal split there
        too -- it isn't only a "did the split algorithm do a good job"
        check.
        """
        from upxo.pxtalops.grain_splitting_3d import size_balance_metrics

        grain_locs = self.grain_locs
        cvs, ratios, ginis = [], [], []
        for gids in self.clusters_dict.values():
            if len(gids) < 2:
                continue
            sizes = [len(grain_locs[g]) for g in gids]
            m = size_balance_metrics(sizes)
            cvs.append(m['cv'])
            ratios.append(m['min_max_ratio'])
            ginis.append(m['gini'])

        def _agg(vals):
            if not vals:
                return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0}
            return {'min': float(np.min(vals)), 'max': float(np.max(vals)),
                   'mean': float(np.mean(vals)), 'median': float(np.median(vals))}

        return {
            'n_multi_packet_pags': len(cvs),
            'cv': _agg(cvs),
            'min_max_ratio': _agg(ratios),
            'gini': _agg(ginis),
        }

    def _get_size_and_degree_statistics(self) -> Dict:
        """PAG/packet size (voxel + physical volume), connectivity/degree,
        and retained-austenite statistics -- the cheap subset of
        get_pag_statistics's extended stats (no convex-hull morphology).
        """
        grain_locs = self.grain_locs
        voxel_vol = float(self.voxel_size) ** 3

        pag_voxel_counts = np.array([sum(len(grain_locs[g]) for g in gids)
                                     for gids in self.clusters_dict.values()])
        if pag_voxel_counts.size:
            pag_size_voxels = {
                'min': int(pag_voxel_counts.min()), 'max': int(pag_voxel_counts.max()),
                'mean': float(pag_voxel_counts.mean()), 'median': float(np.median(pag_voxel_counts)),
                'std': float(pag_voxel_counts.std()),
            }
        else:
            pag_size_voxels = {'min': 0, 'max': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0}
        pag_size_physical = {k: v * voxel_vol for k, v in pag_size_voxels.items()}

        pag_degrees = np.array([len(v) for v in self.neigh_clid.values()]) if self.neigh_clid else np.array([])
        if pag_degrees.size:
            pag_degree = {'min': int(pag_degrees.min()), 'max': int(pag_degrees.max()),
                         'mean': float(pag_degrees.mean()), 'median': float(np.median(pag_degrees))}
        else:
            pag_degree = {'min': 0, 'max': 0, 'mean': 0.0, 'median': 0.0}

        clustered_grains = [g for gids in self.clusters_dict.values() for g in gids]
        neigh_gid = self.neigh_gid
        packet_degrees = np.array([len(neigh_gid.get(g, [])) for g in clustered_grains]) \
            if clustered_grains else np.array([])
        if packet_degrees.size:
            packet_degree = {'min': int(packet_degrees.min()), 'max': int(packet_degrees.max()),
                             'mean': float(packet_degrees.mean()), 'median': float(np.median(packet_degrees))}
        else:
            packet_degree = {'min': 0, 'max': 0, 'mean': 0.0, 'median': 0.0}

        # isolated_grains may include grains already counted inside
        # pag_voxel_counts (any retained-austenite PAG produced by
        # generate_pags() stays in clusters_dict -- see class docstring) as
        # well as grains that never entered clusters_dict at all (the
        # legacy FMSteel3DBase.generate_pag_clusters() pre-filter path, no
        # PAG identity to track). total_voxels must count each domain voxel
        # exactly once, so only the *uncovered* isolated grains get added
        # on top of pag_voxel_counts.sum() -- adding all of iso_voxels
        # unconditionally would double-count the tracked ones.
        covered_by_retained_pag: set = set()
        for pid in self.retained_austenite_pag_ids:
            covered_by_retained_pag.update(self.clusters_dict.get(pid, []))
        leftover_isolated = self.isolated_grains - covered_by_retained_pag
        leftover_iso_voxels = sum(len(grain_locs[g]) for g in leftover_isolated if g in grain_locs)

        iso_voxels = sum(len(grain_locs[g]) for g in self.isolated_grains if g in grain_locs)
        total_voxels = int(pag_voxel_counts.sum()) + leftover_iso_voxels
        volume_fraction_target = (1.0 - self._target_pag_grain_fraction) \
            if self._target_pag_grain_fraction is not None else None
        _sel_info = self._retained_austenite_selection_info or {}
        retained_austenite = {
            'n_grains': len(self.isolated_grains),
            'voxel_volume': iso_voxels,
            'physical_volume': iso_voxels * voxel_vol,
            'volume_fraction_target': volume_fraction_target,
            'volume_fraction_achieved': (iso_voxels / total_voxels) if total_voxels else 0.0,
            # Populated only when this instance came from generate_pags()'s
            # select_isolated_units_with_tolerance -- tells you not just the
            # achieved fraction (above) but whether the iterative
            # trim/retry procedure actually landed within `tolerance` of
            # the target, and how many fresh-ordering attempts that took.
            'tolerance': _sel_info.get('tolerance'),
            'converged': _sel_info.get('converged'),
            'n_attempts': _sel_info.get('n_attempts'),
        }

        return {
            'pag_size_voxels': pag_size_voxels,
            'pag_size_physical': pag_size_physical,
            'pag_size_physical_units': f'{self.units}^3',
            'pag_degree': pag_degree,
            'packet_degree': packet_degree,
            'retained_austenite': retained_austenite,
        }

    def _compute_pag_morphology(self) -> Dict:
        """Per-PAG aspect ratio (sqrt of largest/smallest PCA eigenvalue of
        the voxel coordinates) and solidity (voxel volume / convex hull
        volume), aggregated to min/max/mean/median across all PAGs.

        Opt-in (get_pag_statistics(compute_morphology=True)) -- a convex
        hull per PAG is not free, so this isn't computed by default just to
        show the cheap headline stats.
        """
        from scipy.spatial import ConvexHull

        lgi = self.lgi
        grain_to_pag = {g: pid for pid, gids in self.clusters_dict.items() for g in gids}
        remap = np.zeros(int(lgi.max()) + 1, dtype=np.int64)
        for g, pid in grain_to_pag.items():
            remap[g] = pid
        pag_lgi = remap[lgi]

        flat = pag_lgi.ravel()
        order = np.argsort(flat, kind='stable')
        sorted_labels = flat[order]
        boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(sorted_labels)]))
        unique_labels = sorted_labels[starts]

        shape = lgi.shape
        aspect_ratios, solidities = [], []
        for lbl, s, e in zip(unique_labels.tolist(), starts.tolist(), ends.tolist()):
            if lbl == 0:
                continue
            flat_idx = order[s:e]
            if flat_idx.size < 4:
                continue
            coords = np.column_stack(np.unravel_index(flat_idx, shape)).astype(np.float64)
            centered = coords - coords.mean(axis=0)
            cov = np.cov(centered.T)
            eigvals = np.clip(np.linalg.eigvalsh(cov), 1e-12, None)
            aspect_ratios.append(float(np.sqrt(eigvals[-1] / eigvals[0])))
            try:
                # ConvexHull needs each voxel's 8 physical corners, not just its
                # centre -- a hull built from centres alone omits the outer
                # half-voxel shell on every face, undercounting true volume and
                # pushing "solidity" (voxel volume / hull volume) above 1.0,
                # which is otherwise impossible for a bounded ratio.
                corners = (coords[:, None, :] + _CORNER_OFFSETS[None, :, :]).reshape(-1, 3)
                hull = ConvexHull(corners)
                solidities.append(float(coords.shape[0] / hull.volume))
            except Exception:
                pass

        def _agg(vals):
            if not vals:
                return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0}
            return {'min': float(np.min(vals)), 'max': float(np.max(vals)),
                   'mean': float(np.mean(vals)), 'median': float(np.median(vals))}

        return {'aspect_ratio': _agg(aspect_ratios), 'solidity': _agg(solidities)}

    @property
    def n_pags(self) -> int:
        """Number of PAGs."""
        return len(self.clusters_dict)


__all__ = ['FMSteel3DWithPAGs']
