import numpy as np
import cc3d
from collections import defaultdict
from shapely.geometry import box

"""
CLASSES
=======
patcher_pixvox: A multi-functional orchastrator class using orchastrator functions
calling functions for 2D/3D pixellated/voxellated grain structure modification 
to enable the generation of following features / class of grain structures, 
with statistical representativeness
"""

class patcher_pixvox():
    """
    A multi-functional orchastrator class using orchastrator functions
    calling functions for 2D/3D pixellated/voxellated grain structure modification 
    to enable the generation of following features / class of grain structures, 
    with statistical representativeness:
    (1) Grain strucuture class
        (1.a) Welded grain structures (gs)
        (1.b) Banded gs
        (1.c) Multi-modla gs
        (1.d) Additively manufactured gs

    Import
    ------
    from upxo.pxtalops.temporalPatcher import patcher_pixvox

    Class methods
    -------------
     * by_UPXO_mcgs: Use the UPXO MCGS objects for base and patches(s).
     * by_lfi: Use the Local Feature Indeix Images for base and patches(s).
     * by_mcgsgen: Generate and choose the base and patch gs.

    Slot varaibles
    --------------
    gsbase: upxo.pxtal.mcgs2_temporal_slice.(mcgs2_grain_structure, mcgs3_grain_structure)
        UPXO base grain structure (gs) object. The gs object which 
        is to be modified.
    gspatchers: dict
        UPXO patch gs object. The gs object which is used to modify. Keys
        will be patchID values, whcih coud be same as instance numbers, valued 
        with `upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure` object for 2D
        and `upxo.pxtal.mcgs2_temporal_slice.mcgs3_grain_structure` for 3D.
    lfib: np.ndarray
        Local Feature Index (LFI) of base gs. LFI to be modified. Note: LFI is the 
        gs data image to be modified. Shape: (R, C) for 2D and (P, R, C) for 3D.
    lfip: dict
        Dictionary of Local Feature Index of patch gs. LFI to used to modify lgib. Shape: (R, C) 
        for 2D and (P, R, C) for 3D. Keys will be instance / patchID number, and values
        will the lfi artay.
    eab: np.ndarray
        Pixel/Voxel mapped Euler angles of base gs. Shape: (Nf, 3), where Nf is the 
        number of features.
    eap: dict
        Dictionary of Pixel/Voxel mapped Euler angles of patcher gs. Shape: (Nf, 3), where Nf is the 
        number of features.
    targetDistr: np.ndarray
        Target grain size probability density distribution.
    constraints: dict
        Dictionary of permitrted contraint specifications.
    ninstances: int
        Number of patched instances.
    dim: int
        Dimensionality
    base_distr: np.ndarray
        Cell (i.e. grain) size distriburtion, just a list of number of pixels/voxels.
    patch_distr: np.ndarray, dict
        If creation by gsgen, dict, else np.ndarray. If dict, keys will be 
        patch IDs.
    base_neigh_fid: dict
        Dictionary of central fid valued with neighbouring fids, for base gs.
    patch_neigh_fid: dict
        Dictionary of central fid valued with neighbouring fids, for patch gs.
    patchIds: np.ndarray
        Patched gs IDs starting from 1, will be np.arange(1, ninstance+1)
    patchIDTsliceMap: dict
        Dictionary of patchIDs and their corresponding temporal slice in case of 
        creation by gsgen.
    patched_gs: dict
        Dictionary of patched gs details. Keys will be 'lfi', 'distr', 'neigh_fid', 
        'mesh', etc, valued appropriately.
    lfi_patched: dict
        Dictionary of patched lfi. Keys will be patchIDs, valued with patched lfi.
    fids_patched: dict
        Dictionary of patched fids. Keys will be patchIDs, valued with patched fids.
    neigh_fid_patched: dict
        Dictionary of patched neigh_fid. Keys will be patchIDs, valued with patched neigh_fids.
    conn: int
        Connectivity for connected component analysis. 4 for 2D and 6 for 3D.
    dashboards: dict
        Dictionary of input dashboard paths. Keys will be 'base' and 'patch', valued with the 
        input dashboard paths.
    patchDomainShape: str
        Shape of the patch domain. 'polygonal' for 2D and 'polyhedral' for 3D.
    vizCtrls: dict
        Dictionary of visualization controls.
    a: float
        unit_length in microns (actually, unit pixel / voxel size in microns).
    """
    __slots__ = ('gsbase', 'gspatchers', 'lfib', 'lfip', 'eab', 'eap', 'targetDistr',
                 'constraints', 'ninstances', 'dim', 'base_distr', 'patch_distr',
                 'base_neigh_fid', 'patch_neigh_fid', 'patchIds', 'patchIDTsliceMap',
                 'patched_gs', 'lfi_patched', 'fids_patched', 'neigh_fid_patched', 
                 'conn', 'dashboards', 'patchDomainShape', 'vizCtrls', 'a')

    def __init__(self, **kwargs):
        self.gsbase = kwargs.get("base_grain_structure", None)
        self.gspatchers = kwargs.get("patch_grain_structures", None)
        self.lfib = kwargs.get("base_lfi", None)
        self.lfip = kwargs.get("patch_lfi", None)
        self.base_neigh_fid = kwargs.get("base_neigh_fid", None)
        self.patch_neigh_fid = kwargs.get("patch_neigh_fid", None)
        self.targetDistr = kwargs.get("target_distribution", np.array([]))
        self.ninstances = kwargs.get("ninstances", 1)
        self.constraints = kwargs.get("constraints", {})
        self.dim = kwargs.get("dim", 2)
        self.dashboards = kwargs.get("dashboards", {'base': None, 'patch': None})
        self.patchIds = kwargs.get("patchIds", [1])
        self.patchIDTsliceMap = kwargs.get("patchIDTsliceMap", {1: None})
        self.patchDomainShape = kwargs.get("patchDomainShape", 'polygonal' if self.dim==2 else 'polyhedral')
        self.lfi_patched = kwargs.get("lfi_patched", None)
        self.fids_patched = kwargs.get("fids_patched", None)
        self.neigh_fid_patched = kwargs.get("neigh_fid_patched", None)

    def __repr__(self):
        return f"UPXO-pxmod-mmgs-{self.dim}d {id(self)}"

    @classmethod
    def by_UPXO_mcgs(cls, base_grain_structure=None, patch_grain_structure=None,
            target_distribution=None, ninstances=None, constraints=None, dim=None):
        if base_grain_structure.lgi.shape != patch_grain_structure.lgi.shape:
            raise ValueError("Both grain structures must be the same shape")
        return cls(base_grain_structure=base_grain_structure,
            patch_grain_structures=patch_grain_structure, base_lfi=None,
            patch_lfi=None, target_distribution=target_distribution,
            ninstances=ninstances, constraints=constraints,
            dim=base_grain_structure.dim)

    @classmethod
    def by_lgi(cls, base_lfi=None, patch_lfi=None, target_distribution=None,
            ninstances=None, constraints=None, dim=None):
        if base_lfi.shape != patch_lfi.shape:
            raise ValueError("Both grain structures must be the same shape")
        return cls(base_grain_structure=None, patch_grain_structures=None,
            base_lfi=base_lfi, patch_lfi=patch_lfi,
            target_distribution=target_distribution, ninstances=ninstances,
            constraints=constraints, dim=base_lfi.ndim)

    @classmethod
    def by_mcgsgen(cls, temporalPath='self', base_input_dashboard=None,
            patch_input_dashboard=None, basetslice=10, patchtslices=5,
            target_distribution=None):
        """
        Parameters
        ----------
        temporalPath: str
            Specify whether patch comes from the same temporal set. If 'self',
            the basetslice^th temporal slice of base gs temporal set will be used
            as the base gs. patchtslice^th temporal slie of patch gs temporal set 
            will be used as patcher gs if 'cross'. Options include 'self', 'cross'.
            Defaults to 'self'.
        base_input_dashboard: str
            Defaults to None. Example input: 'C:\\..\\customGSGen1.xls'
        patch_input_dashboard: str
            Defaults to None. Example input: 'C:\\..\\customGSGen2.xls'
        basetslice: int
            Temporal slice for base gs selection.
        patchtslices: int, np.ndarray, list, tuple
            Temporal slice(s) for patch gs selection. In caxse of multyiplre
            temporal slice specifiction, multiple instances will be created.
        target_distribution: np.ndarray
            Target grain size probability density distribution.
        randomize_lfi: bool
            Specify where the lfi IDs must be randomly shuffled. Improves
            visualization.
        """
        # Validations
        if type(patchtslices) not in (int, np.ndarray, list, tuple):
            raise ValueError("patchtslices must be an int or iterable.")
        if type(patchtslices) == int:
            patchtslices = [patchtslices]
        from upxo.ggrowth.mcgs import mcgs
        # --------------------------------
        # Base and patch grain structure initialization
        base = mcgs(input_dashboard = base_input_dashboard)
        patch = mcgs(input_dashboard = patch_input_dashboard)
        # --------------------------------
        baseDim = base.uigrid.dim
        if baseDim != patch.uigrid.dim:
            raise ValueError("Base and path dimensionality must be same.")
        # --------------------------------
        # Base and patch grain structure generation
        base.simulate()
        # Next line to be optimized to detect grain only in specificy temporal slice.
        base.detect_grains(library = 'cc3d')
        # We will re-index grains with stable settigns and shuffle it.
        lfib = cls.shuffleLFIIDs(cls.reindex_lfi(base.gs[basetslice].lfi))
        # make the neighbour dictionary
        base_neigh_fid = cls.find_neighs(lfib)
        # exyrtact grain size distribution: npixels/nvoxels
        base_distr = cls.get_feature_sizes(lfib)
        # --------------------------------
        # Patch grain structure generation
        """If temporalPath is 'self', consider only first in patchtslices, and if
        'cross', consider all the patchtslices."""
        if temporalPath == 'self':
            patchtslices = patchtslices[0]
        elif temporalPath == 'cross':
            pass
        # --------------------------------
        # Patch grain structure generation
        patch.simulate()
        # Next line to be optimized to detect grain only in specificy temporal slice.
        patch.detect_grains(library = 'cc3d')
        # We will re-index grains with stable settigns and shuffle it.
        lfip = {pt: cls.shuffleLFIIDs(cls.reindex_lfi(patch.gs[pt].lfi)) for pt in patchtslices}
        # Make the neighbour dictionary
        patch_neigh_fid = {pt: cls.find_neighs(patch.gs[pt].lfi) for pt in patchtslices}
        # exyrtact grain size distribution: npixels/nvoxels
        patch_distr = {pt: cls.get_feature_sizes(patch.gs[pt].lfi) for pt in patchtslices}
        # --------------------------------
        dashboards = {'base': base_input_dashboard, 'patch': patch_input_dashboard}
        # --------------------------------
        return cls(base_lfi=lfib, patch_lfi=lfip,
            base_neigh_fid=base_neigh_fid, patch_neigh_fid=patch_neigh_fid,
            base_distribution=base_distr, patch_distributions=patch_distr,
            ninstances=len(patchtslices), dim=baseDim, dashboards=dashboards,
            target_distribution=target_distribution)

    def patch(self, lfib=None, lfip=None, patchDomain='rectangular', remove_small_features=False, 
              niterations=4, recharEveryMerge=False, recursionLimit=100, threshold=10, constraints={}):
        if patchDomain == 'rectangular':
            self.patch_rectangular(lfib=lfib, lfip=lfip, constraints=constraints,
                    remove_small_features=remove_small_features, threshold=threshold,
                    niterations=niterations, recharEveryMerge=recharEveryMerge,
                    recursionLimit=recursionLimit)
        elif patchDomain == 'circular':
            self.patch_circular(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'hexahedral':
            self.patch_hexahedral(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'polygonal':
            self.patch_polygonal(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'multi-polygonal':
            self.patch_multiPolygonal(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'mis':
            self.patch_mis(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'gids.N(O)':
            self.patch_gidsNO(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'simplified-weld-1':
            self.patch_simplified_weld_1(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'weld-3D':
            self.patch_weld_3D(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'multi-bead-weld-1':
            self.patch_multi_bead_weld_1(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'mulline-1.2d':
            self.patch_multiline1_2d(lfib=lfib, lfip=lfip, constraints=constraints)
        elif patchDomain == 'mulline-1.3d':
            self.patch_multiline1_3d(lfib=lfib, lfip=lfip, constraints=constraints)

    def find_small_fids(self, lfi, threshold):
        small_fids = np.where(self.get_feature_sizes(lfi) <= threshold)[0]+1 
        return small_fids

    def _patch_rect_(self, lfib, lfip, xbound, ybound):
        xmin, xmax = xbound
        ymin, ymax = ybound
        lfiPatched = lfib.copy()
        lfiPatched[ymin:ymax, xmin: xmax] = lfip[ymin:ymax, xmin: xmax]
        return lfiPatched

    def shuffleLFIIDs(self, lfi):
        unique_ids = np.unique(lfi)
        shuffled_ids = unique_ids.copy()
        np.random.shuffle(shuffled_ids)
        id_map = {old_id: new_id for old_id, new_id in zip(unique_ids, shuffled_ids)}
        lfi = np.vectorize(id_map.get)(lfi)
        return lfi

    def set_connectivity(self):
        self.conn = 4 if self.dim == 2 else 6

    def reindex_lfi(self, lfi):
        self.set_connectivity()
        lfi = cc3d.connected_components(lfi, 
            connectivity=self.conn, out_dtype=np.uint32)
        return lfi.astype(np.int32)

    def find_neighs(self, lfi):
        self.set_connectivity()
        edges = cc3d.region_graph(lfi, connectivity=self.conn)
        neigh_fids = defaultdict(set)
        for edge in edges:
            gid1, gid2 = edge
            neigh_fids[gid1].add(gid2)
            neigh_fids[gid2].add(gid1)
        neigh_fids = {int(gid): np.array(sorted(neighbors), dtype=np.int32) for gid, neighbors in neigh_fids.items()}
        return neigh_fids

    def get_feature_sizes(self, lfi):
        return np.bincount(lfi.ravel())

    def detect_islands(self, neigh_fids):
        islands = []
        for gid, neighs in neigh_fids.items():
            if neighs.size == 1:
                islands.append(gid)
        return islands

    def merge_islands(self, lfi, islands, neigh_fids):
        for island in islands:
            lfi[lfi==island] = neigh_fids[island]
        return lfi

    def find_grains_in_patch(self, lfi, patch_polygon):
        xmin, ymin, xmax, ymax = patch_polygon.bounds
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        patch_region = lfi[ymin:ymax, xmin:xmax]
        grains_in_patch = np.unique(patch_region)
        return grains_in_patch

    def find_external_neighs(self, grains_in_patch, neigh_fids):
        # Find neighbors of grains in patch that do NOT belong to the patch
        grains_in_patch_set = set(grains_in_patch)
        external_neighs = {}

        for gid in grains_in_patch:
            neighbors = neigh_fids[gid]
            external = neighbors[~np.isin(neighbors, grains_in_patch)]
            if external.size > 0:
                external_neighs[int(gid)] = [external]

        return external_neighs

    def find_largest_external_neighs(self, external_neighs, patchedLGI):
        # Find the largest external neighbor for each grain in the patch
        largest_external_neighs = {}

        for gid, external_list in external_neighs.items():
            external = external_list[0]  # Extract the array from the list
            external_sizes = self.get_feature_sizes(patchedLGI)[external]
            largest_external_neigh = external[np.argmax(external_sizes)]
            largest_external_neighs[gid] = int(largest_external_neigh)

        return largest_external_neighs

    def patch_rectangular(self, lfib=None, lfip=None, remove_small_features=False, 
                          niterations=4, recharEveryMerge=False,
                          threshold=10, constraints={}, recursionLimit=100):
        '''constraints={'xbound': None, 'ybound': None, 'cellPropagation': OPTIONS-1A,
                        'propagationProb': 0.5}
        OPTIONS-1A:
            (1) b2p: cells in base at polygon boundary will enter patch domain. Here,
                cells in patch domain at polygon boundary will be merged with largest adjacent 
                cells of base domain. You may want to prefer this if patch domain has smaller 
                grains than base domain; as this would be closer to general physical observation.
            (2) p2b: cells in patch at polygon boundary will enter base domain. Here,
                cells in base domain at polygon boundary will be merged with largest adjacent 
                cells of patch domain. You may want to prefer this if base domain has smaller 
                grains than patch domain; as this would be closer to general physical observation.
            (3) bi-directional: propagationProb*100 percent b2p behaviour and 
                (1-propagationProb)*100 percent p2b behaviour.
            (4) none: polygon boundary will persist in patched grain structue.

        Example call: self.patch(lfib, lfip, patchDomainShape='rectangular',
                                constraints={'xbound': [xmin, xmax], 'ybound': [ymin, ymax],
                                            'cellPropagation': 'b2p',
                                            'propagationProb': 0.5})
        '''
        xbound, ybound = constraints['xbound'], constraints['ybound']
        # Initiate patched lfi
        lfiPatched = self._patch_rect_(lfib, lfip, xbound=xbound, ybound=ybound)
        # Re-index and shuffle the lfi IDs
        lfiPatched = self.shuffleLFIIDs(self.reindex_lfi(lfiPatched))
        # Find neighbourhoods
        neigh_fids = self.find_neighs(lfiPatched)
        # Detect and remoive feature removal, reindex lfi and shuffle lfi IDs
        lfiPatched = self.shuffleLFIIDs(self.reindex_lfi(self.merge_islands(lfiPatched, 
                            self.detect_islands(neigh_fids), neigh_fids)))
        # Find neighbours
        neigh_fids = self.find_neighs(lfiPatched)
        # feature id list
        fids = np.unique(lfiPatched)
        # Build a shapelypolygon patch
        from shapely.geometry import box
        patch_polygon = box(xbound[0], ybound[0], xbound[1], ybound[1])
        # Find grains inside this rectangular patch
        grains_in_patch = self.find_grains_in_patch(lfiPatched, patch_polygon)
        # -----------------------------------------------------------
        if constraints['cellPropagation'] == 'b2p':
            # Find base gs neighbour features of patch polygon
            external_neighs = self.find_external_neighs(grains_in_patch, neigh_fids)
            # Find the largest of the above external_neighs
            largest_external_neighs = self.find_largest_external_neighs(external_neighs,
                                            lfiPatched)
            # Merge internal boundary grains with their largest external neighbors
            for internal_fid, external_fid in largest_external_neighs.items():
                lfiPatched[lfiPatched == internal_fid] = external_fid

        if constraints['cellPropagation'] == 'p2b':
            pass

        if constraints['cellPropagation'] == 'bi-directional':
            pass

        if constraints['cellPropagation'] == 'none':
            pass  # Nothing to do, as boundary will persist.

        # Re-index, shuffle the lfi IDs and find neighs after propagation
        lfiPatched = self.shuffleLFIIDs(self.reindex_lfi(lfiPatched))
        fids = np.unique(lfiPatched)
        neigh_fids = self.find_neighs(lfiPatched)
        # -----------------------------------------------------------
        if remove_small_features:
            fx = self.merge_small_features_iterative
            lfiPatched, fids, neigh_fids = fx(lfiPatched, threshold,
                                mode='largest', max_iterations=recursionLimit)
            '''fx = self.merge_small_features_with_largest_neigh
            lfiPatched, fids, neigh_fids = fx(lfiPatched, threshold,
                        neigh_fids, niterations=niterations,
                        recharEveryMerge=recharEveryMerge, 
                        recursionLimit=recursionLimit)'''
            # Small feature removal might create new islands, so find and merge them again.
            # This is because of multiple neighbours merging into the same small feature, which
            # can create islands of that small feature. So, we need to detect and merge those 
            # islands again. Detect and remoive feature removal, reindex lfi and shuffle lfi IDs.
        # -----------------------------------------------------------
        neigh_fids = self.find_neighs(lfiPatched)
        lfiPatched = self.shuffleLFIIDs(self.reindex_lfi(self.merge_islands(lfiPatched, 
                        self.detect_islands(neigh_fids), neigh_fids)))
        fids = np.unique(lfiPatched)
        neigh_fids = self.find_neighs(lfiPatched)
        # -----------------------------------------------------------
        # Compile patch details
        self.lfi_patched = {1: lfiPatched}
        self.fids_patched = {1: fids}
        self.neigh_fid_patched = {1: neigh_fids}

    def patch_circular(self, lfib=None, lfip=None, constraints=None):
        '''constraints={'xcenter': None, 'ycenter': None, 'radius': None, 'cellPropagation': OPTIONS-1A,
                        'propagationProb': 0.5}
        OPTIONS-1A: All details same as rectangular patch, except that the patch domain
        will be circular instead of rectangular. Circular patch domain will be defined 
        by center (xcenter, ycenter) and radius r.
        '''
        xcenter, ycenter, radius = constraints['xcenter'], constraints['ycenter'], constraints['radius']
        # Initiate patched lfi
        lfiPatched = self._patch_circ_(lfib, lfip, xcenter=xcenter, ycenter=ycenter, radius=radius)
        # Re-index and shuffle the lfi IDs
        lfiPatched = self.shuffleLFIIDs(self.reindex_lfi(lfiPatched))
        # Find neighbourhoods
        neigh_fids = self.find_neighs(lfiPatched)
        # Detect and remoive feature removal, reindex lfi and shuffle lfi IDs
        lfiPatched = self.shuffleLFIIDs(self.reindex_lfi(self.merge_islands(lfiPatched, 
                            self.detect_islands(neigh_fids), neigh_fids)))
        # Find neighbours
        neigh_fids = self.find_neighs(lfiPatched)
        # feature id list
        fids = np.unique(lfiPatched)
        # Build a shapelypolygon patch
        from shapely.geometry import Point
        patch_polygon = Point(xcenter, ycenter).buffer(radius)
        # Find grains inside this circular patch
        grains_in_patch = self.find_grains_in_patch(lfiPatched, patch_polygon)
        # -----------------------------------------------------------
        if constraints['cellPropagation'] == 'b2p':
            # Find base gs neighbour features of patch polygon
            external_neighs = self.find_external_neighs(grains_in_patch, neigh_fids)
            # Find the largest of the above external_neighs
            largest_external_neighs = self.find_largest_external_neighs(external_neighs,
                                            lfiPatched)
            # Merge internal boundary grains with their largest external neighbors

            for internal_fid, external_fid in largest_external_neighs.items():
                lfiPatched[lfiPatched == internal_fid] = external_fid

        if constraints['cellPropagation'] == 'p2b':
            pass

        if constraints['cellPropagation'] == 'bi-directional':
            pass

        if constraints['cellPropagation'] == 'none':
            pass  # Nothing to do, as boundary will persist.
        # Re-index, shuffle the lfi IDs and find neighs after propagation
        lfiPatched = self.shuffleLFIIDs(self.reindex_lfi(lfiPatched))
        fids = np.unique(lfiPatched)
        neigh_fids = self.find_neighs(lfiPatched)
        # -----------------------------------------------------------
        # Compile patch details
        self.lfi_patched = {1: lfiPatched}
        self.fids_patched = {1: fids}
        self.neigh_fid_patched = {1: neigh_fids}

    def _patch_circ_(self, lfib, lfip, xcenter, ycenter, radius):
        y, x = np.ogrid[:lfib.shape[0], :lfib.shape[1]]
        mask = (x - xcenter)**2 + (y - ycenter)**2 <= radius**2
        lfiPatched = lfib.copy()
        lfiPatched[mask] = lfip[mask]
        return lfiPatched

    def merge_small_features_iterative(self, lfi, threshold, mode='smallest', max_iterations=100):
        """
        Iteratively merges features smaller than threshold into neighbors.
        Uses while-loop to avoid recursion limits and improve stability.
        """
        iteration = 0
        
        while iteration < max_iterations:
            # 1. Pre-processing: Clean up isolated "Islands"
            neigh_fids = self.find_neighs(lfi)
            island_fids = self.detect_islands(neigh_fids)
            
            if island_fids:
                lfi = self.merge_islands(lfi, island_fids, neigh_fids)
                # Refresh neighbors after island removal
                neigh_fids = self.find_neighs(lfi)

            # 2. Identify small grains
            small_fids = self.find_small_fids(lfi, threshold)
            
            # Termination condition: No more small grains to merge
            if len(small_fids) == 0:
                break
                
            # 3. Get optimized size array (size_array[fid] = volume)
            size_array = self.get_feature_sizes(lfi)
            
            # 4. Batch merge small grains into neighbors
            for fid in small_fids:
                neighbors = neigh_fids.get(fid, [])
                # Typically we exclude background (ID 0) to keep grain integrity
                valid_neighbors = [n for n in neighbors if n > 0]
                
                if not valid_neighbors:
                    continue

                # Constant time lookup: O(1)
                neighbor_sizes = {n: size_array[n] for n in valid_neighbors}
                    
                if mode == 'largest':
                    target_fid = max(neighbor_sizes, key=neighbor_sizes.get)
                else: # 'smallest'
                    target_fid = min(neighbor_sizes, key=neighbor_sizes.get)
                    
                # Update the array in-place
                lfi[lfi == fid] = target_fid

            # 5. Clean up IDs for the next iteration
            lfi = self.shuffleLFIIDs(self.reindex_lfi(lfi))
            
            iteration += 1
            print(f"Iteration {iteration}: {len(small_fids)} small features processed.")

        return lfi, np.unique(lfi), self.find_neighs(lfi)

    def merge_small_features_with_largest_neigh(self, lfi,
                threshold, neigh_fids, niterations=4, recharEveryMerge=False, 
                recursionLimit=10):
        small_fids = self.find_small_fids(lfi, threshold)
        if len(small_fids) == 0:
            return lfi, np.unique(lfi), neigh_fids
        # -----------------------------------------------------------
        if recharEveryMerge:
            # Re-characterize after every merge oprtation. Could get costly.
            def _vaasu_(lfi, threshold, neigh_fids):
                small_fids = self.find_small_fids(lfi, threshold)
                for small_fid in small_fids:
                    if small_fid in neigh_fids:
                        neighbors = neigh_fids[small_fid]
                        if neighbors.size > 0:
                            neighbor_sizes = self.get_feature_sizes(lfi)[neighbors]
                            largest_neighbor = neighbors[np.argmax(neighbor_sizes)]
                            lfi[lfi == small_fid] = largest_neighbor
                            # Re-index, shuffle the lfi IDs and find neighs after removing small features
                            lfi = self.shuffleLFIIDs(self.reindex_lfi(lfi))
                            fids = np.unique(lfi)
                            neigh_fids = self.find_neighs(lfi)
                return lfi, fids, neigh_fids
            # Run the above function for niterations or until no small features are left, whichever is earlier. Also, break the loop if recursion limit is reached.
            for i in range(recursionLimit+1):
                lfi, fids, neigh_fids = _vaasu_(lfi, threshold, neigh_fids)
                if i >= recursionLimit:
                    print(f"Recursion limit of {recursionLimit} reached. Stopping further merges.")
                    break
        # -----------------------------------------------------------
        if not recharEveryMerge:
            for _ in range(niterations):
                small_fids = self.find_small_fids(lfi, threshold)
                for small_fid in small_fids:
                    if small_fid in neigh_fids:
                        neighbors = neigh_fids[small_fid]
                        if neighbors.size > 0:
                            neighbor_sizes = self.get_feature_sizes(lfi)[neighbors]
                            largest_neighbor = neighbors[np.argmax(neighbor_sizes)]
                            lfi[lfi == small_fid] = largest_neighbor
                # Re-index, shuffle the lfi IDs and find neighs after removing small features
                lfi = self.shuffleLFIIDs(self.reindex_lfi(lfi))
                fids = np.unique(lfi)
                neigh_fids = self.find_neighs(lfi)
        return lfi, fids, neigh_fids

    def patch_hexahedral(self, lfib=None, lfip=None, constraints={}):
        '''constraints={'xbound': None, 'ybound': None, 'zbound': None,
                        'cellPropagation': OPTIONS-1A, 'propagationProb': 0.5}
            OPTIONS-1A: OPTIONS-1A for 'rectangular'.
        '''
        pass

    def patch_polygonal(self, lfib=None, lfip=None, constraints={}):
        '''constraints={'pol': None, 'polType': 'shapely', 'cellPropagation': OPTIONS-1A, 'propagationProb': 0.5}
            OPTIONS-1A: OPTIONS-1A for 'rectangular'.
        '''
        pass

    def patch_multiPolygonal(self, lfib=None, lfip=None, constraints={}):
        '''constraints={'mulpol': None, 'polType': 'shapely', 'cellPropagation': OPTIONS-1A, 'propagationProb': 0.5}
        Where,
            mulpol: A list of polygons, not a multi-polygon.
            OPTIONS-1: list of OPTIONS-1A for each polygon. Each OPTIONS-1A as previous.
        '''
        pass

    def patch_mis(self, lfib=None, lfip=None, constraints={}):
        # Minimal Independent Set base
        pass

    def patch_gidsNO(self, lfib=None, lfip=None, constraints={}):
        # A bunch of GIDs and Neighbour orders for each.
        pass

    def patch_simplified_weld_1(self, lfib=None, lfip=None, constraints={}):
        pass

    def patch_weld_3D(self, lfib=None, lfip=None, constraints={}):
        """H.L. Wei, J.W. Elmer, T. DebRoy, Crystal growth during keyhole mode laser 
        welding, Acta Materialia, Volume 133, 2017, Pages 10-20, ISSN 1359-6454,
        https://doi.org/10.1016/j.actamat.2017.04.074.
        (https://www.sciencedirect.com/science/article/pii/S1359645417303634)
        """
        pass

    def patch_multi_bead_weld_1(self, lfib=None, lfip=None, constraints={}):
        pass

    def patch_multiline1_2d(self, lfib=None, lfip=None, constraints={}):
        """constraints={'mullines': None,
                        'method': 'cut-off-radius',
                        'capradius': 2,
                        'maxWidth': 4,
                        'cellPropagation': OPTIONS-1A,
                        'propagationProb': 0.5}
        Where,
            mulline: list of UPXO 2D multi-line objects. Below are its requirements:
                - Each multi-line object must have atleast 1 UPXO line.
                - The UPXO line must not have UPXO leanest / lean point definition.
                - Starting point of starting line and ending point of ending line must 
                    specify cap radius.
                - Each line must have atleast 1 intermediate UPXO point.
                - Each intermediate point must have a radius specified, which will be 
                    used to identify the pixels within this cut-off radius: this will only
                    be considered if method is 'cut-off-radius'.
            method: Either 'cut-off-radius' or 'polygonal'. If polygonal, more function
                    calls take place, to generate a buffer polygon using maxWidth and capradius.
            cellPropagation: OPTIONS-1: As for 'ractangular.
        """
        pass

    def patch_multiline1_3d(self, lfib=None, lfip=None, constraints={}):
        """Similar to 'mulline-1.2d', but in 3D."""
        pass

    def plot_lfi_2D(self, lfi, **kwargs):
        from upxo.viz.gsviz import see_map
        ax = see_map(lfi, **kwargs)

    def plotImage3D(self, im3d):
        pass