import os
import math
import numpy as np
import cv2
import random
import seaborn as sns
# from pathlib import Path
from copy import deepcopy
from typing import Iterable
import matplotlib.pyplot as plt
from collections import defaultdict
from numba import njit, prange
# from skimage.measure import label as skimg_label
import pandas as pd
# from skimage.measure import label as skim_label
# from upxo.geoEntities.point2d import point2d
# from upxo.meshing.mesher_2d import mesh_mcgs2d
from upxo.dclasses.features import twingen
import upxo.gsdataops.gid_ops as GidOps
import upxo.gsdataops.grid_ops as gridOps
import upxo.connops.neighbour_ops as neighOps
from upxo._sup.gops import att
from upxo._sup.data_ops import find_intersection, find_union_with_counts
from upxo._sup.data_ops import increase_grid_resolution, decrease_grid_resolution
from upxo._sup import dataTypeHandlers as dth
from upxo._sup.validation_values import _validation
from upxo._sup.data_templates import dict_templates
from upxo._sup.console_formats import print_incrementally
from upxo.geoEntities.sline2d import Sline2d as sl2d
from upxo.geoEntities.mulpoint2d import MPoint2d as mulpoint2d
from upxo.pxtalops import manipulator_mergers as manm
import upxo._sup.decorators as decorators

@njit(parallel=True)
def get_neighbor_mask(arr, gid):
    """
    Scans the array. If a pixel equals gid, it checks 4-neighbors.
    If a neighbor is NOT gid, it is marked in the output mask.
    """
    rows, cols = arr.shape
    # Create a boolean mask instead of copying the integer array
    # This saves memory and is faster to initialize
    # mask = np.zeros((rows, cols), dtype=np.bool_)

    # prange allows parallel execution of the outer loop
    for row in prange(rows):
        for col in range(cols):
            # We only care if the current pixel is the Grain ID we are tracking
            if arr[row, col] == gid:
                # Check Up
                if row - 1 >= 0:
                    if arr[row-1, col] != gid:
                        arr[row-1, col] = -1
                # Check Down
                if row + 1 < rows:
                    if arr[row+1, col] != gid:
                        arr[row+1, col] = -1
                # Check Left
                if col - 1 >= 0:
                    if arr[row, col-1] != gid:
                        arr[row, col-1] = -1
                # Check Right
                if col + 1 < cols:
                    if arr[row, col+1] != gid:
                        arr[row, col+1] = -1
            if arr[row, col] != -1:
                arr[row, col] = 0
    return arr

class labeled_feature_image_2d():
    __slots__ = ('lfi', 'ftypes', 'nfeatures', 'props_df',)
    def __init__(self, lfi=None, ftypes=None, nfeatures=None, props_df=None):
        self.lfi = lfi
        self.ftypes = ftypes
        self.nfeatures = nfeatures
        self.props_df = props_df

class mcgs2_grain_structure():
    __slots__ = ('dim', 'uigrid', 'uimesh', 'xgr', 'ygr', 'zgr', 'm', 's', 'S',
                 'fcores', 'fbz', 'pixConn',
                 'binaryStructure2D', 'binaryStructure3D', 'n', 'lgi', 'species',
                 'spart_flag', 'gid', 's_gid', 'gid_s', 's_n', 'g', 'gb',
                 'positions', 'mp', 'vtgs', 'mesh', 'px_size', 'dim',
                 'prop_flag', 'prop', 'are_properties_available', 'prop_stat',
                 '__gi__', 'uinputs', 'display_messages', 'info',
                 'print_interval_bool', 'EAPGLB', 'EASGLB',
                 '__ori_assign_status_stack__', '__ori_assign_status_slice__',
                 'scaled', 'scaled_gs', '__resolution_state__', 'gbjp',
                 'xomap', 'val', 'neigh_gid', 'valid_mprops', 'features',
                 'twingen', 'pxtal', '_gid_bf_merger_', '_char_fx_version_',
                 '_global_iteration_counter_'
                 )
    """
    Slot variables description
    --------------------------
    dim: Dimension of the grain structure. int
    uigrid: User input grid. np.ndarray
    uimesh: User input mesh. dataclass
    xgr, ygr: Grid vectors in x and y directions. np.ndarray
    m: Material properties. dict
    s: State map. np.ndarray
    S: Total number of states. int
    binaryStructure2D: 2D binary structure for ndimage labelling. np.ndarray
    binaryStructure3D: 3D binary structure for ndimage labelling. np.ndarray
    n: Total number of grains. int
    lgi: Labelled grain image. np.ndarray
    species: Species data structure. dict
    spart_flag: State partitioning flag. dict
    gid: List of grain IDs. list
    s_gid: Dict of state to grain IDs. dict
    gid_s: List of state for each grain ID. list
    s_n: List of number of grains in each state. list
    g: Grain data structure. dict
    gb: Grain boundary data structure. dict
    positions: Positions of grains. dict
    mp: Multipoint data structure. dict
    vtgs: Voronoi tessellation grain structure. dataclass
    mesh: Mesh data structure. dataclass
    px_size: Size of pixel. float
    dim: Dimension of the grain structure. int
    prop_flag: Property calculation flag. dict
    prop: Property data structure. dict
    are_properties_available: Flag indicating if properties are available. bool
    prop_stat: Property statistics data structure. dict
    __gi__: Grain iterator index. int
    uinputs: User inputs data structure. dataclass
    display_messages: Flag for displaying messages. bool
    info: Information data structure. dict
    print_interval_bool: Flag for print interval. bool
    EAPGLB: Global equivalent axis-angle orientation data structure. dict
    EASGLB: Global equivalent axis-angle orientation data structure. dict
    __ori_assign_status_stack__: Orientation assignment status for stack. dict
    __ori_assign_status_slice__: Orientation assignment status for slice. dict
    scaled: Scaled data structure. dict
    scaled_gs: Scaled grain structure. dataclass
    __resolution_state__: Resolution state data structure. dict
    gbjp: Grain boundary junction points data structure. dict
    xomap: Orientation map. np.ndarray
    val: Validation data structure. dataclass
    neigh_gid: Neighbour grain IDs data structure. dict
    valid_mprops: Valid material properties flags. dict
    features: Features data structure. dataclass
    twingen: Twin generation data structure. dataclass
    pxtal: Parent crystal data structure. dataclass
    _gid_bf_merger_: Grain ID before merger data structure. dict=
    """
    EPS = 1e-12
    grain_coords_dtype = np.float16
    __maxGridSizeToIgnoreStoringGrids = 1000**2
    valid_skprop_names = ['area', 'area_bbox', 'area_convex', 'area_filled',
                          'axis_major_length', 'axis_minor_length', 'bbox',
                          'centroid', 'centroid_local', 'centroid_weighted', 'centroid_weighted_local',
                          'coords', 'coords_scaled', 'eccentricity', 'equivalent_diameter_area',
                          'euler_number', 'extent', 'feret_diameter_max', 'image', 'image_convex',
                          'image_filled', 'image_intensity', 'inertia_tensor', 'inertia_tensor_eigvals',
                          'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std',
                          'label', 'moments', 'moments_central', 'moments_hu', 'moments_normalized',
                          'moments_weighted', 'moments_weighted_central', 'moments_weighted_hu',
                          'moments_weighted_normalized', 'num_pixels', 'orientation', 'perimeter',
                          'perimeter_crofton', 'slice', 'solidity']
    valid_morpho_props = ['area', 'area_bbox', 'area_convex', 'area_filled',
                          'axis_major_length', 'axis_minor_length', 'bbox',
                          'centroid', 'centroid_local', 'centroid_weighted', 'centroid_weighted_local',
                          'coords', 'coords_scaled', 'eccentricity', 'equivalent_diameter_area',
                          'euler_number', 'extent', 'feret_diameter_max', 'image', 'image_convex',
                          'image_filled', 'image_intensity', 'inertia_tensor', 'inertia_tensor_eigvals',
                          'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std',
                          'label', 'moments', 'moments_central', 'moments_hu', 'moments_normalized',
                          'moments_weighted', 'moments_weighted_central', 'moments_weighted_hu',
                          'moments_weighted_normalized', 'num_pixels', 'orientation', 'perimeter',
                          'perimeter_crofton', 'slice', 'solidity']
    '''
    Global Topological Invariants
        euler_characteristic: Chi = V - E + G (usually 1)
        avg_nneigh: Average number of neighbors (\bar{n})
    Grain-Specific Properties (Per Grain)
        nneigh: Number of neighbors (or edges, n) for a grain
    Junction/Vertex Counts (Global)
        n_vertices: Total number of vertices (V)
        n_boundaries: Total number of grain boundaries (E)
        ntjp: Number of triple junctions (V3)
        nqp: Number of quad junctions (V4, relevant during T1 events)
    Statistical & Correlation Properties
        nneigh_dist_P_n: The probability distribution P(n)
        aboav_weaire_params: Parameters describing the Aboav-Weaire correlation. 
        They quantify the topological correlation between a grain's size and the 
        average size of its neighbors.
    '''
    valid_topo_props = ['euler_characteristic', 'avg_nneigh', 'nneigh',
                        'n_vertices', 'n_boundaries', 'ntjp', 'nqp',
                        'nneigh_dist_P_n', 'aboav_weaire_params']

    def __init__(self, dim=2, m=None, uidata=None, S_total=None, px_size=None,
                 xgr=None, ygr=None, zgr=None, uigrid=None, uimesh=None,
                 EAPGLB=None, assign_ori_stack=False, assign_ori_slice=True,
                 oripert_tc=True, oripert_gr=True):
        self.uinputs = uidata
        self.val = _validation()
        self.dim, self.m, self.S, self.px_size = 2, m, S_total, px_size
        self.uigrid, self.uimesh = uigrid, uimesh
        self.xgr, self.ygr = xgr, ygr
        self.set__spart_flag(S_total)
        self.set__s_gid(S_total)
        self.set__gid_s()
        self.set__s_n(S_total)
        self.g, self.gb, self.info = {}, {}, {}
        self.EAPGLB = {}
        self.EAPGLB['statewise'] = EAPGLB
        self.EASGLB = self.EAPGLB
        # Above EASGLB needs to be updated in the orinetation mapping stage
        self.mp = dict_templates.mulpnt_gs2d
        self.scaled = {'xmin': None, 'xmax': None, 'xinc': None, 'xgr': None,
                       'ymin': None, 'ymax': None, 'yinc': None, 'ygr': None,
                       's': None, 'grains': None, 'prop': None}
        self.scaled_gs = None
        self.are_properties_available, self.display_messages = False, False
        self.__setup__positions__()
        self.xomap = None
        self.neigh_gid = None
        self.species = {}

        if assign_ori_stack:
            self.__ori_assign_status_stack__ = {'status': False,
                                                'info': 'to be developed'}
        if assign_ori_slice:
            __info = "..-t:u-s:u-..-ru-..-d:c-..-ea:s-.."
            self.__ori_assign_status_slice__ = {'status': True,
                                                'info': __info}
        self.valid_mprops = {'npixels': False, 'npixels_gb': False, 'area': True,
                             'eq_diameter': False, 'perimeter': False,
                             'perimeter_crofton': False, 'compactness': False,
                             'gb_length_px': False, 'aspect_ratio': False,
                             'solidity': False, 'morph_ori': False, 'circularity': False,
                             'eccentricity': False, 'feret_diameter': False,
                             'major_axis_length': False, 'minor_axis_length': False,
                             'euler_number': False, 'char_grain_positions': False}
        self.pxtal = {}

    def __iter__(self):
        # Initialize grain iterator
        self.__gi__ = 1
        return self

    def __next1__(self):
        # Iterator to get pixel indices of grains one after the other
        if self.n:
            if self.__gi__ <= self.n:
                grain_pixel_indices = np.argwhere(self.lgi == self.__gi__)
                self.__gi__ += 1
                return grain_pixel_indices
            else:
                raise StopIteration

    def __next__(self):
        # Iterator to get grain objects one after the other
        if self.n:
            if self.__gi__ <= self.n:
                if self._char_fx_version_ == 1:
                    thisgrain = self.g[self.__gi__]['grain']
                elif self._char_fx_version_ == 2:
                    thisgrain = self.g[self.__gi__]
                self.__gi__ += 1
                return thisgrain
            else:
                raise StopIteration

    def __str__(self):
        # String representation of the object
        return 'grains :: att : n, lgi, id, ind, spart'

    def __att__(self):
        return att(self)
    
    def set_grain_coords_dtype(self, dtype):
        """Set the data-type for storing grain coordinates."""
        self.grain_coords_dtype = dtype

    @classmethod
    def from_image(cls, fdb=np.random.random((50, 50)),
                   xmin=0, ymin=0, xinc=1, yinc=1, xmax=50, ymax=50,):
        """Create mcgs2_grain_structure instance from image."""
        uigrid = {'fdb': fdb,
                  'xmin': xmin, 'ymin': ymin,
                  'xinc': xinc, 'yinc': yinc,
                  'xmax': xmax, 'ymax': ymax,
                  'xgr': np.arange(xmin, xmax, xinc),
                  'ygr': np.arange(ymin, ymax, yinc)}
        return cls(uigrid)

    @property
    def get_px_size(self):
        '''Get size of the pixel.'''
        return self.px_size

    def set__s_n(self, S_total,):
        """
        nth value represents the number of grains in the nth state

        Parameters
        ----------
        S_total : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.s_n = [0 for s in range(1, S_total+1)]

    def set__s_gid(self, S_total):
        """
        Set up dict of s as keys with None values.

        Parameters
        ----------
        S_total : int
            Specify the total number of states.

        Returns
        -------
        None
        """
        self.s_gid = {s: None for s in range(1, S_total+1)}

    def set__gid_s(self):
        '''Set up empty list. This would contain list of s values for every gid.'''
        self.gid_s = []

    def set__spart_flag(self, S_total):
        """
        Set up spart flag dictionary with False values for each state.

        Parameters
        ----------
        S_total : int
            Specify the total number of states.

        Returns
        -------
        None
        """
        self.spart_flag = {_s_: False for _s_ in range(1, S_total+1)}

    def _check_lgi_dtype_uint8(self, lgi):
        """Validates and modifies (if needed) lgi user input data-type."""
        if type(lgi) == np.ndarray and np.size(lgi) > 0 and np.ndim(lgi) == 2:
            if self.lgi.dtype.name != 'uint8':
                self.lgi = lgi.astype(np.uint8)
            else:
                self.lgi = lgi
        else:
            self.lgi = 'invalid mcgs 4685'

    @property
    def lfi(self):
        return self.lgi

    def calc_num_grains(self, throw=False):
        """Calculate the total number of grains in this grain structure."""
        if self.lgi:
            self.n = self.lgi.max()
            if throw:
                return self.n
            
    def get_property_bounded_grains(self, pnames=None, mprops=None, pvalue_thresholds=None):
        """
        Get grain IDs of grains with property values beyond specified thresholds.

        Parameters:
        - mprops: dict
            Dictionary containing material properties with property names as keys and numpy arrays of values.
        - pnames: list
            List of property names to evaluate.
        - pvalue_thresholds: dict
            Dictionary with property names as keys and [lower_threshold, upper_threshold] as values.

        Returns:
        - props_gids: dict
            Dictionary with property names as keys and lists of grain IDs that meet the threshold criteria.
        
        Example
        --------
        pnames=['area', 'aspect_ratio', 'perimeter', 'solidity']
        mprops = gsan.gsstack[gsid].get_mprops(pnames, set_missing_mprop=True)
        pvalue_thresholds = {'area': [10, None], 'aspect_ratio': [2.0, None],
                            'perimeter': [15, None], 'solidity': [0.8, None]}
        props_gids = self.get_propery_bounded_grains(mprops, pnames, pvalue_thresholds)
        """
        props_gids = {pname: None for pname in pnames}
        for pname in pnames:
            LTh, UTh = pvalue_thresholds[pname]
            gids_subset_lth = np.where(mprops[pname] <= LTh)[0]+1  # Get grain IDs of low value grains
            gids_subset_uth = np.where(mprops[pname] >= UTh)[0]+1 if UTh else np.array([])  # Get grain IDs of high value grains
            gids_subset = [int(i) for i in np.concatenate((gids_subset_lth, gids_subset_uth))]
            props_gids[pname] = gids_subset
        return props_gids
    
    def extract_neigh_props(self, gids, mprop):
        """
        Extract properties of neighboring grains for given grain IDs.

        Parameters
        ----------
        gids : list or array-like
            List of grain IDs for which to extract neighbor properties.
        mprop : dict
            Dictionary of morphological property values corresponding to grain IDs.

        Returns
        -------
        mprops_neigh : dict
            Dictionary containing neighbor grain IDs and their property values.
        """
        mprops_neigh = {'gids': {}, 'vals': {}}
        for gid in gids:
            neighs = self.find_neigh_gid(gid, include_central_grain=False, throw=True)
            vals = mprop[neighs-1]
            checkinf = np.isinf(vals)
            if np.any(checkinf):
                neighs = neighs[~checkinf]
                vals = [float(a) for a in vals[~checkinf]]
            else:
                vals = [float(a) for a in vals]
            mprops_neigh['gids'][gid] = [int(neigh) for neigh in neighs]
            mprops_neigh['vals'][gid] = vals
        return mprops_neigh

    def find_neigh(self, include_central_grain=False, print_msg=True,
                   user_defined_bbox_ex_bounds=False, bbox_ex_bounds=None,
                   update_grain_object=True, use_numba=False):
        """
        Find the gids of neighbours for every gid.

        Parameters
        ----------
        include_central_grain : bool, optional
            Whether to include the central grain in the neighbor list. Default is False.

        Examples
        --------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent', input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 10
        pxtal.gs[tslice].char_morph_2d(char_gb=True)

        pxtal.gs[tslice].find_neigh(include_central_grain=True)
        pxtal.gs[tslice].neigh_gid[10]

        pxtal.gs[tslice].find_neigh(include_central_grain=False)
        pxtal.gs[tslice].neigh_gid[10]
        """
        self.neigh_gid = {}
        if self.gid.size == 1:
            self.neigh_gid[self.gid[0]] = [int(self.gid[0])]
            return
        # ---------------------------------------------------------------------
        if print_msg:
            print('\nExtracting neigh list for all grains\n')
        for gid in self.gid:
            bbox_ex_bounds_fid = bbox_ex_bounds[gid] if user_defined_bbox_ex_bounds else None
            self.neigh_gid[gid] = [int(ngid) for ngid in self.find_neigh_gid(gid,
                                include_central_grain=include_central_grain, 
                                update_grain_object=update_grain_object,
                                throw=True,
                                user_defined_bbox_ex_bounds=user_defined_bbox_ex_bounds,
                                bbox_ex_bounds_fid=bbox_ex_bounds_fid,
                                use_numba=use_numba)]
            
    def find_neigh_v2(self, p=1.0, include_central_grain=False,
                      throw_numba_dict=False, verbosity_nfids=1000):
        _lgi_ = deepcopy(self.lgi)
        _lgi_ = _lgi_.astype(np.int32)
        self.neigh_gid = GidOps.find_O1_neigh_2d(_lgi_, p=p, 
                                            include_central_grain=include_central_grain,
                                            throw_numba_dict=throw_numba_dict,
                                            validate_input=False, 
                                            verbosity_nfids=verbosity_nfids)

    @decorators.port_doc('upxo.connops.neighbour_ops', 'find_neigh_fid')
    def find_neigh_gid(self, fid, include_central_grain=False, 
                       update_grain_object=True, 
                       user_defined_bbox_ex_bounds=False, 
                       bbox_ex_bounds_fid=None, use_numba=False,
                       get_gbsegs=False, save_gbsegs=False, throw_gbsegs=False,
                       throw=False):
        """'find_neigh_gid' is deprecated function name retained for backward 
        compatibility."""
        if len(self.g) == 0:
            print('\n', 25*'-')
            print('NOTE: ', '\t', "This gs tslice has'nt been characterised. Charecterising to proceed.")
            self.char_morph_2d(char_gb=False)
            print('\n', 25*'-')
        neighbour_ids = neighOps.find_neigh_fid(self.g, self.lfi, fid, self.n,
                                    include_central_grain=include_central_grain,
                                    update_grain_object=update_grain_object, 
                                    user_defined_bbox_ex_bounds=user_defined_bbox_ex_bounds,
                                    bbox_ex_bounds_fid=bbox_ex_bounds_fid,
                                    use_numba=use_numba, _char_fx_version_=self._char_fx_version_,
                                    get_gbsegs=get_gbsegs, save_gbsegs=save_gbsegs,
                                    throw=throw, throw_gbsegs=throw_gbsegs)
        if throw:
            return neighbour_ids

    @decorators.port_doc('upxo.connops.neighbour_ops', 'find_neigh_fid')
    def find_neigh_fid(self, fid, include_central_grain=False, 
                       update_grain_object=True, 
                       user_defined_bbox_ex_bounds=False, 
                       bbox_ex_bounds_fid=None, use_numba=False,
                       get_gbsegs=False, save_gbsegs=False, throw_gbsegs=False,
                       throw=False):
        neighbour_ids = neighOps.find_neigh_fid(self.g, self.lfi, fid, self.n,
                                    include_central_grain=include_central_grain,
                                    update_grain_object=update_grain_object, 
                                    user_defined_bbox_ex_bounds=user_defined_bbox_ex_bounds,
                                    bbox_ex_bounds_fid=bbox_ex_bounds_fid,
                                    use_numba=use_numba, _char_fx_version_=self._char_fx_version_,
                                    get_gbsegs=get_gbsegs, save_gbsegs=save_gbsegs,
                                    throw=throw, throw_gbsegs=throw_gbsegs)
        if throw:
            return neighbour_ids
        
    @property
    def neigh_fid(self):
        """Get the neighbour grain IDs dictionary. neigh_gid is deprecated. neigh_fid is 
        preferred and is to be used in all further development. This property decorator
        is for backward compatibility.
        """
        return self.neigh_gid

    @decorators.port_doc('upxo.pxtalops.grid_ops', 'find_feature_extended_bbox_pix')
    def find_extended_bounding_box(self, fid, make_binary=False):
        grain_lfi_ExtBBox = gridOps.find_feature_extended_bbox_pix(fid=fid, lfi=self.lgi,
                                                                   make_binary=make_binary)
        return grain_lfi_ExtBBox

    @decorators.port_doc('upxo.pxtalops.grid_ops', 'find_extended_bbox_pix_fids')
    def find_extended_bounding_box_all_grains(self, make_binary=False):
        print('Finding extended bounding boxes for all grains...')
        grain_lgi_ex_all = gridOps.find_extended_bbox_pix_fids(fids=self.gid, lfi=self.lgi,
                                                               make_binary=make_binary)
        return grain_lgi_ex_all

    @decorators.port_doc('upxo.pxtalops.grid_ops', 'find_extended_bbox_pix_fids')
    def find_extended_bounding_box_fids(self, fids=None, make_binary=False):
        print('Finding extended bounding boxes for specified grains...')
        grain_lgi_ex_fids = gridOps.find_extended_bbox_pix_fids(fids=self.gid if fids is None else fids, lfi=self.lgi,
                                                               make_binary=make_binary)
        return grain_lgi_ex_fids

    def assign_species(self, method='mc state partitioned global combined',
                       ignore_vf=True, vf={}, spid=1, combineids=[], ninstances=10,
                       detect_features=True, bso=1, characterise_features=True, 
                       make_feature_skprops=True, extract_feature_coords=True,
                       throw_feature_bounding_box=True
                       ):
        """
        Assign species based on state partitioning methods.
        Parameters
        ----------
        method : str, optional
            Method for state partitioning. Default is 'mc state partitioned global combined'.
        ignore_vf : bool, optional
            Whether to ignore volume fractions. Default is True.
        vf : dict, optional
            Volume fractions for each state. Default is an empty dict.
        spid : int, optional
            Species ID. Default is 1.
        combineids : list of list of int, optional
            List of state combinations for partitioning. Default is an empty list.
        ninstances : int, optional
            Number of instances to create. Default is 10.
        detect_features : bool, optional
            Whether to detect features in the image. Default is True.
        bso : int, optional
            Binary structure order for feature detection. Default is 1.
        characterise_features : bool, optional
            Whether to characterise detected features. Default is True.
        make_feature_skprops : bool, optional
            Whether to create scikit-image regionprops for features. Default is True.
        extract_feature_coords : bool, optional
            Whether to extract feature coordinates. Default is True.
        throw_feature_bounding_box : bool, optional
            Whether to throw bounding box for features. Default is True.

        Returns
        -------
        None

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 10
        pxtal.gs[tslice].assign_species(method='mc state partitioned global combined',
                                        ignore_vf=True, vf={}, spid=1,
                                        combineids=[[1,2],[3,4]], ninstances=5,
                                        detect_features=True, bso=1, characterise_features=True,
                                        make_feature_skprops=True, extract_feature_coords=True,
                                        throw_feature_bounding_box=True)
        """
        if type(vf) != dict:
                raise ValueError('Invalid vf type.')
        # ---------------------------------------------
        if method in ('mc state partitioned global',
                      'mc state partitioned global vf',):
            S, S_ = self.S, 1/self.S
            if len(vf) == 0:
                vf = {s_: S_ for s_ in range(self.S)}
                N = {} # To be developed
            elif sum(vf.keys()) < 1.0:
                raise ValueError('sum(Vf) is not unity.')
        # ---------------------------------------------
        if method in ('mc state partitioned local',
                      'mc state partitioned local vf',):
            s = np.unique(self.s, dtype=np.int16)
            if len(vf) == 0:
                vf = {}
                for s_ in self.S:
                    if s_ in s:
                        vf[s_] = 1/len(s)
                    else:
                        vf[s_] = 0.0
            if len(set([k in s for k in vf.keys()])) != 1:
                raise ValueError('Not all keys of vf are in S. Try local instead.')
            N = {s: None} # To be developed
        # ---------------------------------------------
        if method == 'mc state partitioned global' and len(vf) == 0:
            self.species[spid] = deepcopy(self.s)
        # ---------------------------------------------
        fx1 = self.charecterise_features_in_image
        Xgrid, Ygrid = self.xgr, self.ygr
        # ---------------------------------------------
        if ignore_vf:
            if method == 'mc state partitioned global combined':
                if type(combineids) not in dth.dt.ITERABLES:
                    raise ValueError('combine should be an iterable of state values.')
                if not all(isinstance(sublist, (list, tuple)) for sublist in combineids):
                    raise ValueError('combine must be an iterable of iterables (list of lists).')
                if ninstances >= 1:
                    self.species[spid] = {}
                    for i in range(ninstances):
                        species = self.combine_partitions(deepcopy(self.s), combineids)
                        self.species[spid]['inst_'+str(i+1)] = deepcopy(species)
                        if detect_features:
                            features = self.detect_features_in_image(deepcopy(species),
                                                              binary_structure_order=bso)
                            # instance number: i+1: image
                            name = 'inst_'+str(i+1)+'_img'
                            self.species[spid][name] = features[0]
                            # instance number: i+1: image mappings - original labels to feature ID labels
                            name = 'inst_'+str(i+1)+'_orig_to_labels'
                            self.species[spid][name] = {int(k): tuple(v) for k, v in features[1].items()}
                            # instance number: i+1: image mappings - feature ID labels to original labels
                            name = 'inst_'+str(i+1)+'_labels_to_orig'
                            self.species[spid][name] = {k: int(v) for k, v in features[2].items()}
                        if characterise_features:
                            skprops, bbox_limits_ex, bboxes_ex, coords_dict = fx1(features[0], Xgrid, Ygrid,
                                                                                  make_skprops=make_feature_skprops,
                                                                                  extract_coords=extract_feature_coords,
                                                                                  throw_bounding_box=throw_feature_bounding_box)
                            # instance number: i+1: Scikit-image regionprops of features like grains
                            name = 'inst_'+str(i+1)+'_feature_skprops'
                            self.species[spid][name] = {int(k): v for k, v in skprops.items()}
                            # instance number: i+1: Feature bounding box limits (extended) of gfeatures
                            name = 'inst_'+str(i+1)+'_feature_bbox_limits'
                            self.species[spid][name] = bbox_limits_ex
                            # instance number: i+1: Feature bounding box limits (extended) of gfeatures
                            name = 'inst_'+str(i+1)+'_feature_bboxes_ex'
                            self.species[spid][name] = bboxes_ex
                            # instance number: i+1: Feature co-ordinates
                            name = 'inst_'+str(i+1)+'_feature_coords'
                            self.species[spid][name] = coords_dict
        else:
            # To be developed
            pass
        # ---------------------------------------------
        if method == 'mc state partitioned global vf':
            phases = deepcopy(self.s)
            N = {s: None} # To be developed
        # ---------------------------------------------
        if method == 'mc state partitioned local':
            pass
        # ---------------------------------------------
        if method == 'mc state partitioned local vf':
            pass
        # ---------------------------------------------
        if method == 'ng vf':
            pass

    def combine_partitions(self, image_data, combinations):
        """
        Combine multiple partition IDs in an image array according to specified groupings.
        This method merges partition regions by replacing multiple partition IDs with a single
        target ID for each group. After combining, the partition IDs are renumbered sequentially
        starting from 1.
        Parameters
        ----------
        image_data : numpy.ndarray
            Input image array containing partition IDs as integer values.
        combinations : list of list of int
            List of groups where each group is a list of partition IDs to be combined.
            The first valid ID in each group becomes the target ID for that group.
        Returns
        -------
        numpy.ndarray
            Modified image array with combined partitions and renumbered IDs starting from 1.
            Has the same shape as the input image_data.
        Notes
        -----
        - Only partition IDs that are present in the image_data are considered valid.
        - Groups with fewer than 2 valid IDs are skipped.
        - The final renumbering ensures sequential partition IDs without gaps.

        Example
        -------
        import numpy as np
        image_data = np.array([[1, 1, 2, 2],
                               [1, 3, 3, 2],
                               [4, 4, 3, 2]])
        combinations = [[1, 2], [3, 4]]
        combined_image = combine_partitions(image_data, combinations)
        print(combined_image)
        # Output:
        # [[1 1 1 1]
        #  [1 2 2 1]
        #  [2 2 2 1]]
        """
        original_shape = image_data.shape 
        present_values = set(np.unique(image_data))
        for group in combinations:
            valid_ids = [val for val in group if val in present_values]
            if len(valid_ids) < 2:
                continue  
            target_id = valid_ids[0]
            ids_to_replace = valid_ids[1:]
            mask = np.isin(image_data, ids_to_replace)
            image_data[mask] = target_id
            for old_id in ids_to_replace:
                present_values.discard(old_id)
        _, inverse_indices = np.unique(image_data, return_inverse=True)
        img_data_mod = inverse_indices.reshape(original_shape)+1
        return img_data_mod
    
    def detect_features_in_image(self, image_data, binary_structure_order=2,):
        """
        Detect features in an image and label them uniquely.

        Parameters
        ----------
        image_data : numpy.ndarray
            Input image array containing different states as integer values.
        binary_structure_order : int, optional
            Order of the binary structure for connectivity. Default is 2 (8-connectivity).
        Returns
        -------
        labeled_image : numpy.ndarray
            Labeled image where each detected feature has a unique integer label.
        original_to_labels : dict
            Mapping from original state values to lists of new feature labels.
        labels_to_original : dict
            Mapping from new feature labels to their corresponding original state values.

        Example
        -------
        import numpy as np
        image_data = np.array([[1, 1, 0, 2, 2],
                               [1, 0, 0, 2, 2],
                               [0, 0, 3, 3, 0],
                               [4, 4, 0, 0, 0]])
        labeled_image, orig_to_labels, labels_to_orig = detect_features_in_image(image_data, binary_structure_order=2)
        print("Labeled Image:\n", labeled_image)
        print("Original to Labels Mapping:\n", orig_to_labels)
        print("Labels to Original Mapping:\n", labels_to_orig)
        """
        from scipy.ndimage import label as nd_label
        from scipy.ndimage import generate_binary_structure
        binstr = generate_binary_structure(2, binary_structure_order)
        labeled_image = np.zeros_like(image_data, dtype=int)
        original_to_labels = {}  # {original_state_id : [new_label_1, new_label_2]}
        labels_to_original = {}  # {new_label_id: original_state_id}
        current_label_offset = 1
        unique_states = np.unique(image_data)
        for state_val in unique_states:
            mask = (image_data == state_val)
            temp_labels, num_features = nd_label(mask, structure=binstr)
            if num_features == 0:
                continue
            feature_mask = temp_labels > 0
            temp_labels[feature_mask] += (current_label_offset-1)
            labeled_image += temp_labels
            new_ids = list(range(current_label_offset,
                                 current_label_offset+num_features))
            original_to_labels[state_val] = new_ids
            for new_id in new_ids:
                labels_to_original[new_id] = state_val
            current_label_offset += num_features

        return labeled_image, original_to_labels, labels_to_original
    
    def charecterise_features_in_image(self, labelled_image, Xgrid, Ygrid,
                                       make_skprops=True, extract_coords=True,
                                       throw_bounding_box=True
                                       ):
        """
        Charecterise features in an image.
        Parameters
        ----------
        labelled_image : numpy.ndarray
            Labeled image where each feature has a unique integer label.
        Xgrid : numpy.ndarray
            X-coordinates grid corresponding to the image.
        Ygrid : numpy.ndarray
            Y-coordinates grid corresponding to the image.
        make_skprops : bool, optional
            Whether to create scikit-image regionprops for features. Default is True.
        extract_coords : bool, optional
            Whether to extract feature coordinates. Default is True.
        throw_bounding_box : bool, optional
            Whether to throw bounding box for features. Default is True.
        
        Returns
        -------
        skprops : dict
            Dictionary with feature IDs as keys and their scikit-image regionprops as values.
        bbox_limits_ex : dict
            Dictionary with feature IDs as keys and their extended bounding box limits as values.
        bboxes_ex : dict
            Dictionary with feature IDs as keys and their extended bounding box images as values.
        coords_dict : dict
            Dictionary with feature IDs as keys and their coordinates as values.

        Example
        -------
        skprops, bbox_limits, bboxes_ex, coords_dict = self.charecterise_features_in_image(labelled_image, Xgrid, Ygrid,
                                       make_skprops=True, extract_coords=True,
                                       throw_bounding_box=True
                                       )
        """
        fids = [int(fid) for fid in np.unique(labelled_image)]
        skprops = {fid: None for fid in fids}
        bbox_limits_ex = {fid: None for fid in fids}
        bboxes_ex = {fid: None for fid in fids}
        coords_dict = {fid: None for fid in fids}
        if make_skprops:
            from skimage.measure import regionprops
        for fid in fids:
            _, L = cv2.connectedComponents(np.array(labelled_image == fid,
                                                    dtype=np.uint8))
            loc_ = np.where(L == 1)
            rmin, rmax = loc_[0].min(), loc_[0].max()+1
            cmin, cmax = loc_[1].min(), loc_[1].max()+1
            Rlab, Clab = L.shape
            rmin_ex, rmax_ex = rmin-int(rmin != 0), rmax+int(rmin != Rlab)
            cmin_ex, cmax_ex = cmin-int(cmin != 0), cmax+int(cmax != Clab)
            bbox_ex = np.array(L[rmin_ex:rmax_ex, cmin_ex:cmax_ex], dtype=np.uint8)
            if throw_bounding_box:
                bbox_limits_ex[fid] = [rmin_ex, rmax_ex, cmin_ex, cmax_ex]
                bboxes_ex[fid] = bbox_ex
            if extract_coords:
                coords_dict[fid] = np.array([[Xgrid[ij[0], ij[1]], Ygrid[ij[0], ij[1]]]
                                for ij in np.argwhere(L == 1)])
            if make_skprops:
                skprops[fid] = regionprops(bbox_ex, cache=False)[0]
        return skprops, bbox_limits_ex, bboxes_ex, coords_dict
    
    def charecterise_features_in_image_v2(self, labelled_image, Xgrid=None, Ygrid=None, 
                                        make_skprops=True, extract_coords=True, 
                                        throw_bounding_box=True):
        if Xgrid is None or Ygrid is None:
            h, w = labelled_image.shape
            indices = np.indices((h, w))
            Xgrid, Ygrid = indices[1], indices[0]
        # -------------------------------------
        from skimage.measure import regionprops
        props = regionprops(labelled_image)

        skprops = {}
        bbox_limits = {}
        bbox_limits_ex = {}
        bboxes = {}
        bboxes_ex = {}
        coords_dict = {}
        Rlab, Clab = labelled_image.shape
        for prop in props:
            fid = prop.label
            # prop.bbox gives (min_row, min_col, max_row, max_col)
            rmin, cmin, rmax, cmax = prop.bbox

            rmin = max(0, rmin)
            rmax = min(Rlab, rmax)
            cmin = max(0, cmin)
            cmax = min(Clab, cmax)
            
            rmin_ex = max(0, rmin - 1)
            rmax_ex = min(Rlab, rmax + 1)
            cmin_ex = max(0, cmin - 1)
            cmax_ex = min(Clab, cmax + 1)
            if throw_bounding_box:
                bbox_limits[fid] = [rmin, rmax, cmin, cmax]
                bbox_limits_ex[fid] = [rmin_ex, rmax_ex, cmin_ex, cmax_ex]
                # slice for the extended bounding box
                bboxes[fid] = (labelled_image[rmin:rmax, cmin:cmax] == fid).astype(np.int32)
                bboxes_ex[fid] = (labelled_image[rmin_ex:rmax_ex, cmin_ex:cmax_ex] == fid).astype(np.int32)
            if extract_coords:
                # prop.coords gives indices (row, col) of all pixels in the grain
                coords = prop.coords 
                rows = coords[:, 0]
                cols = coords[:, 1]
                coords_dict[fid] = np.column_stack((Xgrid[rows, cols], Ygrid[rows, cols]))
            if make_skprops:
                skprops[fid] = prop

        return skprops, bbox_limits, bbox_limits_ex, bboxes, bboxes_ex, coords_dict

    def extract_feature_properties(self, skprops={}, area=True, eq_diameter=False,
                                   feret_diameter=False, perimeter=False,  perimeter_crofton=False,
                                   npixels_gb=False, gb_length_px=False, major_axis_length=True,
                                   minor_axis_length=True, aspect_ratio=False, compactness=False,
                                   solidity=False,  morph_ori=False, circularity=False,
                                   eccentricity=False, euler_number=True, moments_hu=True,):
        """
        Extract feature properties from skprops.

        Parameters
        ----------
        skprops : dict
            Dictionary with feature IDs as keys and their scikit-image regionprops as values.
        area : bool, optional
            Whether to extract area property. Default is True.
        eq_diameter : bool, optional
            Whether to extract equivalent diameter property. Default is False.
        feret_diameter : bool, optional
            Whether to extract feret diameter property. Default is False.
        perimeter : bool, optional
            Whether to extract perimeter property. Default is False.
        perimeter_crofton : bool, optional
            Whether to extract perimeter crofton property. Default is False.
        npixels_gb : bool, optional
            Whether to extract number of pixels in grain boundary property. Default is False.
        gb_length_px : bool, optional
            Whether to extract grain boundary length in pixels property. Default is False.
        major_axis_length : bool, optional
            Whether to extract major axis length property. Default is True.
        minor_axis_length : bool, optional
            Whether to extract minor axis length property. Default is True.
        aspect_ratio : bool, optional
            Whether to extract aspect ratio property. Default is False.
        compactness : bool, optional
            Whether to extract compactness property. Default is False.
        solidity : bool, optional
            Whether to extract solidity property. Default is False.
        morph_ori : bool, optional
            Whether to extract morphological orientation property. Default is False.
        circularity : bool, optional
            Whether to extract circularity property. Default is False.
        eccentricity : bool, optional
            Whether to extract eccentricity property. Default is False.
        euler_number : bool, optional
            Whether to extract euler number property. Default is True.
        moments_hu : bool, optional
            Whether to extract Hu moments property. Default is True.

        Returns
        -------
        mprops : dict
            Dictionary with property names as keys and their corresponding values as numpy arrays.

        Example
        -------
        skprops, bbox_limits_ex, bboxes_ex, coords_dict = self.charecterise_features_in_image(labelled_image, Xgrid, Ygrid,
                                       make_skprops=True, extract_coords=True,
                                       throw_bounding_box=True
                                       )
        mprops = self.extract_feature_properties(skprops=skprops, area=True, eq_diameter=False,
                                   feret_diameter=False, perimeter=False,  perimeter_crofton=False,
                                   npixels_gb=False, gb_length_px=False, major_axis_length=True,
                                   minor_axis_length=True, aspect_ratio=False, compactness=False,
                                   solidity=False,  morph_ori=False, circularity=False,
                                   eccentricity=False, euler_number=True, moments_hu=True,)
        """
        if len(skprops) == 0 or not isinstance(skprops, dict):
            raise ValueError('Invalid skprops specification')
        mprops = {'area': area, 'npixels_gb': npixels_gb, 'gb_length_px': gb_length_px,
                  'equivalent_diameter_area': eq_diameter, 'feret_diameter_max': feret_diameter, 
                  'perimeter': perimeter, 'perimeter_crofton': perimeter_crofton,
                  'axis_major_length': major_axis_length,
                  'axis_minor_length': minor_axis_length, 'aspect_ratio': aspect_ratio,
                  'compactness': compactness, 'solidity': solidity,
                  'orientation': morph_ori, 'circularity': circularity, 'eccentricity': eccentricity,
                  'euler_number': euler_number, 'moments_hu': moments_hu}
        if moments_hu:
            # As of skimage 0.19.0, moments_hu is a 7-element array, so we create flags for each
            # moment to extract them individually.
            mprops = {**mprops, **{'mhu_1': True, 'mhu_2': True, 'mhu_3': True, 'mhu_4': True,
                                 'mhu_5': True, 'mhu_6': True, 'mhu_7': True}}
        if aspect_ratio:
            # aspect ratio needs major and minor axis lengths
            mprops['axis_major_length'] = True
            mprops['axis_minor_length'] = True
        if compactness or circularity:
            # compactness and circularity need perimeter and area
            mprops['perimeter'] = True
            mprops['area'] = True
        # -------------------------------------
        mprops = {k: [] for k, v in mprops.items() if v}
        mprops_sk = {k: v for k, v in mprops.items() if k in self.valid_skprop_names}
        mprops_nonsk = {k: v for k, v in mprops.items() if k not in self.valid_skprop_names}
        # -------------------------------------
        for fid, skprop in enumerate(skprops.values(), start=0):
            for pname in mprops_sk.keys():
                if pname in self.valid_skprop_names:
                    mprops_sk[pname].append(getattr(skprop, pname))
        # -------------------------------------
        if 'aspect_ratio' in mprops_nonsk.keys():
            mprops_nonsk['aspect_ratio'] = (mprops_sk['axis_minor_length']/mprops_sk['axis_major_length'])
        if 'compactness' in mprops_nonsk.keys():
            mprops_nonsk['compactness'] = mprops_sk['perimeter']**2/(4.0*np.pi*mprops_sk['area'])
        if 'circularity' in mprops_nonsk.keys():
            mprops_nonsk['circularity'] = 4.0*np.pi*mprops_sk['area']/(mprops_sk['perimeter']**2)
        # -------------------------------------
        mprops_sk= {k: np.array(v) for k, v in mprops_sk.items()}
        mprops_nonsk= {k: np.array(v) for k, v in mprops_nonsk.items()}
        if moments_hu:
            for mhu_i, mhu_name in enumerate([f'mhu_{i+1}' for i in range(mprops_sk['moments_hu'].shape[1])], start=0):
                mprops_nonsk[mhu_name] = mprops_sk['moments_hu'][:, mhu_i]
            del mprops_sk['moments_hu']
        mprops = {**mprops_sk, **mprops_nonsk}
        return mprops

    def find_neigh_gid_fast(self, gid, include_parent=False, return_type='tuple'):
        """
        Find neighbouring grains of a given gid.

        Parameters
        ----------
        gid : int
            Grain ID for which to find neighbours.
        include_parent : bool, optional
            Whether to include the parent gid in the neighbours list. Default is False.
        return_type : str, optional
            Type of return value: 'tuple' or 'list'. Default is 'tuple'.

        Returns
        -------
        tuple or list
            Neighbouring grain IDs.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        np.unique(pxtal.gs[16].find_extended_bounding_box(10))
        pxtal.gs[10].find_neigh_gid_fast(10)
        """
        neighbours = list(np.unique(self.find_extended_bounding_box(gid)))
        if not include_parent:
            neighbours.remove(gid)
    
        return tuple(neighbours)

    def find_neigh_gid_fast_all_grains(self, include_parent=False,
                                       saa=True, throw=False):
        """
        Find neighbouring grains for all gids.

        Parameters
        ----------
        include_parent : bool, optional
            Whether to include the parent gid in the neighbours list. Default is False.
        saa : bool, optional
            Whether to store the result as an attribute. Default is True.
        throw : bool, optional
            Whether to return the result. Default is False.
        
        Returns
        -------
        dict
            Dictionary with grain IDs as keys and their neighbouring grain IDs as values.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        np.unique(pxtal.gs[16].find_extended_bounding_box(10))
        pxtal.gs[10].find_neigh_gid_fast_all_grains(include_parent=False)
        pxtal.gs[10].neigh_gid
        """
        neigh_gid = {gid: self.find_neigh_gid_fast(gid, include_parent=include_parent)
                     for gid in self.gid}
        for gid, neighs in neigh_gid.items():
            neigh_gid[gid] = [int(gid) for gid in neighs]
        if saa:
            self.neigh_gid = neigh_gid
        if throw:
            return neigh_gid

    def get_upto_nth_order_neighbors(self, grain_id, neigh_order,
                                     fast_estimate=False,
                                     recalculate=False, include_parent=True,
                                     output_type='list'):
        """
        Calculates the nth order neighbours for a given gid.

        Parameters
        ----------
        grain_id : int
            The ID of the grain for which to find neighbors.
        neigh_order : int
            The order of neighbors to calculate (1st order, 2nd order, etc.).
        fast_estimate : bool, optional
            Whether to use a fast estimation method. Default is False.
        recalculate : bool, optional
            Whether to recalculate neighbors even if they are already computed. Default is False.
        include_parent : bool, optional
            Whether to include the parent grain ID in the neighbors list. Default is True.
        output_type : str, optional
            The type of output: 'list', 'nparray', or 'set'. Default is 'list'.

        Returns
        -------
        list, np.ndarray, or set
            Neighbors of the specified order.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        # pxtal = mcgs(study='independent', input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 18
        pxtal.gs[tslice].char_morph_2d(char_gb=True)
        neigh_order = 3
        gid = 6
        neighbours = pxtal.gs[tslice].get_upto_nth_order_neighbors(gid, neigh_order,
                                                      fast_estimate=False,
                                                      recalculate=True,
                                                      include_parent=True,
                                                      output_type='list')
        pxtal.gs[tslice].neigh_gid[gid]

        pxtal.gs[tslice].plot_grains_gids(pxtal.gs[tslice].gid, gclr='color', title='')
        pxtal.gs[tslice].plot_grains_gids([gid], gclr='color', title='parent gid: '+str(gid))
        pxtal.gs[tslice].plot_grains_gids(neighbours, gclr='color', cmap_name='nipy_spectral')
        """
        # ---------------------------------------------------------------------
        if neigh_order == 0:
            return grain_id
        # ---------------------------------------------------------------------
        if recalculate or self.neigh_gid is None or len(self.neigh_gid) == 0:
            if fast_estimate:
                self.find_neigh_gid_fast_all_grains(include_parent=include_parent)
            else:
                self.find_neigh(include_central_grain=include_parent)
        # ---------------------------------------------------------------------
        # print(self.neigh_gid[grain_id])
        # ---------------------------------------------------------------------
        # Start with 1st-order neighbors
        neighbors = set(self.neigh_gid.get(grain_id, []))
        for _ in range(neigh_order - 1):
            new_neighbors = set()
            for neighbor in neighbors:
                new_neighbors.update(self.neigh_gid.get(neighbor, []))
            neighbors.update(new_neighbors)
        # ---------------------------------------------------------------------
        if not include_parent:
            neighbors.discard(grain_id)
        if output_type == 'list':
            return list(neighbors)
        if output_type == 'nparray':
            return np.array(list(neighbors))
        elif output_type == 'set':
            return neighbors

    def get_nth_order_neighbors(self, grain_id, neigh_order, fast_estimate=False,
                                recalculate=False, include_parent=True):
        """
        Calculates the 1st till nth order neighbours for a given gid.

        Parameters
        ----------
        grain_id : int
            The ID of the grain for which to find neighbors.
        neigh_order : int
            The order of neighbors to calculate (1st order, 2nd order, etc.).
        fast_estimate : bool, optional
            Whether to use a fast estimation method. Default is False.
        recalculate : bool, optional
            Whether to recalculate neighbors even if they are already computed. Default is False.
        include_parent : bool, optional
            Whether to include the parent grain ID in the neighbors list. Default is True.

        Returns
        -------
        list
            A list containing the nth order neighbors.
        
        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        gid = 10
        tslice = 16
        neigh_order = 6
        neighbours = pxtal.gs[tslice].get_nth_order_neighbors(gid, neigh_order,
                                             fast_estimate=False,
                                             recalculate=True,
                                             include_parent=True)
        pxtal.gs[tslice].plot_grains_gids(neighbours, gclr='color')
        """
        neigh_upto_n_minus_1 = self.get_upto_nth_order_neighbors(grain_id, neigh_order-1,
                                                                 fast_estimate=fast_estimate,
                                                                 include_parent=include_parent,
                                                                 output_type='set')
        if type(neigh_upto_n_minus_1) in dth.dt.NUMBERS:
            neigh_upto_n_minus_1 = set([neigh_upto_n_minus_1])

        neigh_upto_n = self.get_upto_nth_order_neighbors(grain_id, neigh_order,
                                                         fast_estimate=fast_estimate,
                                                         include_parent=include_parent,
                                                         output_type='set')
        if type(neigh_upto_n) in dth.dt.NUMBERS:
            neigh_upto_n = set([neigh_upto_n])
        return list(neigh_upto_n.difference(neigh_upto_n_minus_1))

    def get_upto_nth_order_neighbors_all_grains(self, neigh_order,
                                                recalculate=False, fast_estimate=False,
                                                include_parent=True, output_type='list'):
        """
        Calculates 1st to nth order neighbors of all gids.

        Parameters
        ----------
        neigh_order : int
            The order of neighbors to calculate (1st order, 2nd order, etc.).
        recalculate : bool, optional
            Whether to recalculate neighbors even if they are already computed. Default is False.
        fast_estimate : bool, optional
            Whether to use a fast estimation method. Default is False.
        include_parent : bool, optional
            Whether to include the parent grain ID in the neighbors list. Default is True.
        output_type : str, optional
            The type of output: 'list', 'nparray', or 'set'. Default is 'list'.

        Returns
        -------
        dict
            Dictionary with grain IDs as keys and their neighbors of specified order as values.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        neigh_order = 1
        pxtal.gs[16].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,
                                                             output_type='list')
        """
        neighs_upto_nth_order = {gid: self.get_upto_nth_order_neighbors(gid, neigh_order,
                                                                        fast_estimate=fast_estimate,
                                                                        recalculate=recalculate,
                                                                        include_parent=include_parent,
                                                                        output_type='list')
                                 for gid in self.gid}
        return neighs_upto_nth_order

    def get_nth_order_neighbors_all_grains(self, neigh_order, fast_estimate=False,
                                           recalculate=False, include_parent=True):
        """
        Calculates the nth order neighbours of all gids.

        Parameters
        ----------
        neigh_order : int
            The order of neighbors to calculate (1st order, 2nd order, etc.).
        fast_estimate : bool, optional
            Whether to use a fast estimation method. Default is False.
        recalculate : bool, optional
            Whether to recalculate neighbors even if they are already computed. Default is False.
        include_parent : bool, optional
            Whether to include the parent grain ID in the neighbors list. Default is True.

        Returns
        -------
        dict
            Dictionary with grain IDs as keys and their neighbors of specified order as values.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        tslice = 99
        pxtal.gs[tslice].char_morph_2d(char_gb=True)

        no_clr = ['k', 'b', 'r', 'g']
        no_mrk = ['o', 's', 'x', '+']
        no_msz = [8, 8, 8, 8]
        gb_clr = ['k', 'b', 'r', 'g']

        cg = 1
        NO = [2]
        ANO = [None, None, None]
        plt.figure(figsize=(5, 5), dpi=100)
        for no in NO:
            A = pxtal.gs[tslice].get_nth_order_neighbors_all_grains(no,
                                                                    fast_estimate=False,
                                                                    recalculate=False,
                                                                    include_parent=True,)
            for gid in A[cg]:
                plt.plot(*np.roll(pxtal.gs[tslice].g[gid]['grain'].gbloc, 1, axis=1).T,
                         gb_clr[no]+'.', markersize=2)
                gidcentroid = pxtal.gs[tslice].g[gid]['grain'].centroid
                plt.plot(*gidcentroid, no_clr[no]+no_mrk[no],
                         markersize=no_msz[no])
        plt.plot(*np.roll(pxtal.gs[tslice].g[cg]['grain'].gbloc, 1, axis=1).T,
                 'c.', markersize=4)
        cgcentroid = pxtal.gs[tslice].g[cg]['grain'].centroid
        plt.plot(*cgcentroid, no_clr[0]+no_mrk[0],
                 markersize=no_msz[0])
        plt.gca().set_aspect('equal')



        cg = 10  # central_grain
        neigh_order = 1
        A = pxtal.gs[tslice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,
                                                             output_type='list')
        A = pxtal.gs[tslice].get_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,)
        pxtal.gs[tslice].plot_grains_gids(A[cg], gclr='color', title="user grains",
                         cmap_name='CMRmap_r', )

        neigh_order = 2
        A = pxtal.gs[tslice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,
                                                             output_type='list')
        A = pxtal.gs[tslice].get_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,)
        pxtal.gs[tslice].plot_grains_gids(A[cg], gclr='color', title="user grains",
                         cmap_name='CMRmap_r', )

        neigh_order = 3
        A = pxtal.gs[tslice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,
                                                             output_type='list')
        A = pxtal.gs[tslice].get_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,)
        pxtal.gs[tslice].plot_grains_gids(A[cg], gclr='color', title="user grains",
                         cmap_name='CMRmap_r', )



        # -----------------------------------------
        no_clr = ['k', 'b', 'k', 'k']
        no_mrk = ['o', 's', 'x', '+']
        no_msz = [8, 8, 8, 8]

        plt.figure(figsize=(5, 5), dpi=100)
        for gid in A[cg]:
            plt.plot(*np.roll(pxtal.gs[tslice].g[gid]['grain'].gbloc, 1, axis=1).T,
                     'k.', markersize=3)
            gidcentroid = pxtal.gs[tslice].g[gid]['grain'].centroid
            plt.plot(*gidcentroid, no_clr[neigh_order]+no_mrk[neigh_order],
                     markersize=no_msz[neigh_order])

        plt.plot(*np.roll(pxtal.gs[tslice].g[cg]['grain'].gbloc, 1, axis=1).T,
                 'r.', markersize=4)
        cgcentroid = pxtal.gs[tslice].g[cg]['grain'].centroid
        plt.plot(*cgcentroid, no_clr[0]+no_mrk[0],
                 markersize=no_msz[0])
        plt.gca().set_aspect('equal')
        # =================================================================
        all_neighs = pxtal.gs[tslice].neigh_gid
        neigh_order = 1
        cg = 17
        plt.figure(figsize=(5, 5), dpi=100)
        for gid in all_neighs[cg]:
            plt.plot(*np.roll(pxtal.gs[tslice].g[gid]['grain'].gbloc, 1, axis=1).T,
                     'k.', markersize=3)
            gidcentroid = pxtal.gs[tslice].g[gid]['grain'].centroid
            plt.plot(*gidcentroid, no_clr[neigh_order]+no_mrk[neigh_order],
                     markersize=no_msz[neigh_order])

        plt.plot(*np.roll(pxtal.gs[tslice].g[cg]['grain'].gbloc, 1, axis=1).T,
                 'r.', markersize=4)
        cgcentroid = pxtal.gs[tslice].g[cg]['grain'].centroid
        plt.plot(*cgcentroid, no_clr[0]+no_mrk[0],
                 markersize=no_msz[0])
        plt.gca().set_aspect('equal')
        """
        neighs_nth_order = {gid: self.get_nth_order_neighbors(gid, neigh_order,
                                                              fast_estimate=fast_estimate,
                                                              recalculate=recalculate,
                                                              include_parent=include_parent)
                            for gid in self.gid}
        return neighs_nth_order

    def get_upto_nth_order_neighbors_all_grains_prob(self, neigh_order,
                                                     recalculate=False,
                                                     include_parent=False,
                                                     print_msg=False,
                                                     _int_approx_=0.05):
        """
        Calculates 1st to nth order neighbors of all gids.
        Allows float values for neigh_order for probabilistic selection.

        Parameters
        ----------
        neigh_order : int or float
            The order of neighbors to calculate (1st order, 2nd order, etc.).
            If float, probabilistic selection is done between floor and ceil values.
        recalculate : bool, optional
            Whether to recalculate neighbors even if they are already computed. Default is False.
        include_parent : bool, optional
            Whether to include the parent grain ID in the neighbors list. Default is False.
        print_msg : bool, optional
            Whether to print messages during execution. Default is False.
        _int_approx_ : float, optional
            Threshold to consider a float as an integer. Default is 0.05.

        Returns
        -------
        dict
            Dictionary with grain IDs as keys and their neighbors of specified order as values.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 10
        def_neigh = pxt.gs[tslice].get_upto_nth_order_neighbors_all_grains_prob

        neigh0 = def_neigh(1, recalculate=False, include_parent=True)
        neigh1 = def_neigh(1.06, recalculate=False, include_parent=True)
        neigh2 = def_neigh(1.5, recalculate=False, include_parent=True)
        neigh0[22]
        neigh1[2][22]
        neigh2[2][22]
        """
        # @dev:
            # no: neighbour order in these definitions.
        no = neigh_order
        on_neigh_all_grains_upto = self.get_upto_nth_order_neighbors_all_grains
        on_neigh_all_grains_at = self.get_nth_order_neighbors_all_grains
        if isinstance(no, (int, np.int32)):
            if print_msg:
                print('neigh_order is of type int. Adopting the usual method.')
            neigh_on = on_neigh_all_grains_upto(no, recalculate=recalculate,
                                           include_parent=include_parent)
            return neigh_on
        elif isinstance(no, (float, np.float64)):
            if abs(no-round(no)) < _int_approx_:
                if print_msg:
                    print('neigh_order is close to being int. Adopting usual method.')
                neigh_on = on_neigh_all_grains_upto(math.floor(no),
                                                    recalculate=recalculate,
                                                    include_parent=include_parent)
                return neigh_on
            else:
                if print_msg:
                    # Nothing to print
                    pass
                no_low, no_high = math.floor(no), math.ceil(no)
                neigh_upto_low = on_neigh_all_grains_upto(no_low,
                                                          recalculate=recalculate,
                                                          include_parent=include_parent)
                neigh_at_high = on_neigh_all_grains_at(no_low+1,
                                                       recalculate=recalculate,
                                                       include_parent=False)
                delno = np.round(abs(neigh_order-math.floor(neigh_order)), 4)
                neighbours = {}
                for gid in self.gid:
                    nselect = math.ceil(delno * len(neigh_at_high[gid]))
                    if len(neigh_at_high[gid]) > 1:
                        neighbours[gid] = neigh_upto_low[gid] + random.sample(neigh_at_high[gid],
                                                                              nselect)
                return neighbours
        else:
            raise ValueError('Invalid neigh_order')

    def char_morph_2d(self, use_characterization_settings=False, use_version=1,
                      bso=1, def_feat_name='grain',
                      bbox=True, bbox_ex=True, npixels=False,
                      npixels_gb=False, identify_pixel_locations=True,
                      area=False, eq_diameter=False,
                      perimeter=False, perimeter_crofton=False,
                      compactness=False, gb_length_px=False, aspect_ratio=False,
                      solidity=False, morph_ori=False, circularity=False,
                      eccentricity=False, feret_diameter=False,
                      major_axis_length=False, minor_axis_length=False,
                      euler_number=False, moments_hu=True, 
                      append=False, saa=True, throw=False,
                      char_grain_positions=False, find_neigh=False,
                      char_gb=False, make_skim_prop=False,
                      get_grain_coords=True):
        if use_version in [1, 2]:
            # Make data holder for properties
            from upxo._sup.data_templates import pd_templates
            __ = pd_templates()
            __a, __b, __c = __.make_prop2d_df(bbox=bbox, bbox_ex=bbox_ex, npixels=npixels,
                      npixels_gb=npixels_gb, area=area, eq_diameter=eq_diameter,
                      perimeter=perimeter, perimeter_crofton=perimeter_crofton,
                      compactness=compactness, gb_length_px=gb_length_px,
                      aspect_ratio=aspect_ratio, solidity=solidity,
                      morph_ori=morph_ori, circularity=circularity,
                      eccentricity=eccentricity, feret_diameter=feret_diameter,
                      major_axis_length=major_axis_length,
                      minor_axis_length=minor_axis_length,
                      euler_number=euler_number, moments_hu=moments_hu, 
                      append=append)
            self.prop_flag, self.prop, self.prop_stat = __a, __b, __c
            # -------------------------------------
            property_flag_kwargs = {'bbox': bbox, 
                      'bbox_ex': bbox_ex,
                      'npixels': npixels,
                      'npixels_gb': npixels_gb,
                      'identify_pixel_locations': identify_pixel_locations,
                      'area': area,
                      'eq_diameter': eq_diameter,
                      'perimeter': perimeter,
                      'perimeter_crofton': perimeter_crofton,
                      'compactness': compactness,
                      'gb_length_px': gb_length_px, 
                      'aspect_ratio': aspect_ratio,
                      'solidity': solidity,
                      'morph_ori': morph_ori,
                      'circularity': circularity,
                      'eccentricity': eccentricity,
                      'feret_diameter': feret_diameter,
                      'major_axis_length': major_axis_length,
                      'minor_axis_length': minor_axis_length,
                      'euler_number': euler_number,
                      'moments_hu': moments_hu,
                      'char_grain_positions': char_grain_positions,
                      'find_neigh': find_neigh,
                      'char_gb': char_gb,
                      'make_skim_prop': make_skim_prop,
                      'get_grain_coords': get_grain_coords,
                      }
        if use_version == 1:
            print(f"Characterising MC simulation time-slice {self.m}")
            self.prop_flag, self.prop, self.prop_stat = __a, __b, __c
            self.char_morph_2d_v1(use_characterization_settings=use_characterization_settings, 
                      append=append, saa=saa, throw=throw,
                      make_pd=False, **property_flag_kwargs)
        elif use_version == 2:
            print(f"Characterising MC simulation time-slice {self.m}")
            self.char_morph_2d_v2(bso=bso, def_feat_name=def_feat_name, 
                      saa=saa, throw=throw,
                      append=False, make_pd=False, **property_flag_kwargs)
        else:
            raise ValueError('Invalid use_version specified')

    def char_morph_2d_v1(self, use_characterization_settings=False,
                      bbox=True, bbox_ex=True, npixels=False,
                      npixels_gb=False, identify_pixel_locations=True,
                      area=False, eq_diameter=False,
                      perimeter=False, perimeter_crofton=False,
                      compactness=False, gb_length_px=False, aspect_ratio=False,
                      solidity=False, morph_ori=False, circularity=False,
                      eccentricity=False, feret_diameter=False,
                      major_axis_length=False, minor_axis_length=False,
                      euler_number=False, moments_hu=True, 
                      append=False, saa=True, throw=False,
                      char_grain_positions=False, find_neigh=False,
                      char_gb=False, make_skim_prop=False,
                      get_grain_coords=True, make_pd=True):
        """
        This method allows user to calculate morphological parameters
        of a given grain structure slice.

        Parameters
        ----------
        use_characterization_settings : bool, optional
            Whether to use pre-defined characterization settings. Default is False.
        bbox : bool, optional
            Whether to extract bounding box property. Default is True.
        bbox_ex : bool, optional
            Whether to extract extended bounding box property. Default is True.
        npixels : bool, optional
            Whether to extract number of pixels property. Default is False.
        npixels_gb : bool, optional
            Whether to extract number of grain boundary pixels property. Default is False.
        area : bool, optional
            Whether to extract area property. Default is False.
        eq_diameter : bool, optional
            Whether to extract equivalent diameter property. Default is False.
        perimeter : bool, optional
            Whether to extract perimeter property. Default is False.
        perimeter_crofton : bool, optional
            Whether to extract perimeter (Crofton) property. Default is False.
        compactness : bool, optional
            Whether to extract compactness property. Default is False.
        gb_length_px : bool, optional
            Whether to extract grain boundary length in pixels property. Default is False.
        aspect_ratio : bool, optional
            Whether to extract aspect ratio property. Default is False.
        solidity : bool, optional
            Whether to extract solidity property. Default is False.
        morph_ori : bool, optional
            Whether to extract morphological orientation property. Default is False.
        circularity : bool, optional
            Whether to extract circularity property. Default is False.
        eccentricity : bool, optional
            Whether to extract eccentricity property. Default is False.
        feret_diameter : bool, optional
            Whether to extract feret diameter property. Default is False.
        major_axis_length : bool, optional
            Whether to extract major axis length property. Default is False.
        minor_axis_length : bool, optional
            Whether to extract minor axis length property. Default is False.
        euler_number : bool, optional
            Whether to extract euler number property. Default is False.
        moments_hu : bool, optional
            Whether to extract Hu moments property. Default is True.
        append : bool, optional
            Whether to append to existing properties. Default is False.
        saa : bool, optional
            Whether to store as attribute. Default is True.
        throw : bool, optional
            Whether to return the properties. Default is False.
        char_grain_positions : bool, optional
            Whether to characterize grain positions. Default is False.
        find_neigh : bool, optional
            Whether to find neighboring grains. Default is False.
        char_gb : bool, optional
            Whether to characterize grain boundaries. Default is False.
        make_skim_prop : bool, optional
            Whether to make skim properties. Default is False.
        get_grain_coords : bool, optional
            Whether to get grain coordinates. Default is True.
        """
        from upxo.xtal.mcgrain2d_definitions import grain2d
        self._char_fx_version_ = 1
        if make_pd:
            # Make data holder for properties
            from upxo._sup.data_templates import pd_templates
            __ = pd_templates()
            __a, __b, __c = __.make_prop2d_df(bbox=bbox, bbox_ex=bbox_ex, npixels=npixels,
                      npixels_gb=npixels_gb, area=area, eq_diameter=eq_diameter,
                      perimeter=perimeter, perimeter_crofton=perimeter_crofton,
                      compactness=compactness, gb_length_px=gb_length_px,
                      aspect_ratio=aspect_ratio, solidity=solidity,
                      morph_ori=morph_ori, circularity=circularity,
                      eccentricity=eccentricity, feret_diameter=feret_diameter,
                      major_axis_length=major_axis_length,
                      minor_axis_length=minor_axis_length,
                      euler_number=euler_number, moments_hu=moments_hu, 
                      append=append, make_pd=True)
            self.prop_flag, self.prop, self.prop_stat = __a, __b, __c
        # ---------------------------------------------
        Rlab, Clab = self.lgi.shape[0], self.lgi.shape[1]
        # ---------------------------------------------
        if make_skim_prop:
            from skimage.measure import regionprops
        # ---------------------------------------------
        for s in self.s_gid.keys():
            if s % 5 == 0:
                print(f"--------State value: {s} of {self.S}")
            s_gid_keys_npy = [skey for skey in self.s_gid.keys()
                              if self.s_gid[skey]]
            # ---------------------------------------------
            sn = 1
            for state in s_gid_keys_npy:
                grains = self.s_gid[state]
                # Iterate through each grain of this state value
                _ngrains_ = len(grains)
                for i, gn in enumerate(grains, start=1):
                    gn = int(gn)
                    #if _ngrains_%100 == 0:
                    #    print(f'....grain no. {i}/{_ngrains_}')
                    _, L = cv2.connectedComponents(np.array(self.lgi == gn,
                                                            dtype=np.uint8))
                    self.g[gn] = {'s': state, 'grain': grain2d()}
                    self.g[gn]['grain'].gid = gn
                    
                    if identify_pixel_locations:
                        locations = np.argwhere(L == 1)
                        self.g[gn]['grain'].loc = locations
                        _ = locations.T
                        self.g[gn]['grain'].xmin = _[0].min()
                        self.g[gn]['grain'].xmax = _[0].max()
                        self.g[gn]['grain'].ymin = _[1].min()
                        self.g[gn]['grain'].ymax = _[1].max()
                    self.g[gn]['grain'].npixels = locations.shape[0]
                    self.g[gn]['grain'].s = state
                    self.g[gn]['grain'].sn = sn
                    self.g[gn]['grain']._px_area = self.px_size
                    sn += 1
                    # ---------------------------------------------
                    # Extract grain boundary indices
                    if char_gb:
                        mask = np.zeros_like(self.lgi)
                        mask[self.lgi == gn] = 255
                        mask = mask.astype(np.uint8)
                        contours, _ = cv2.findContours(mask,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)
                        gb = np.squeeze(contours[0], axis=1)
                        # Interchange the row and column to get into right
                        # indexing order
                        gb[:, [1, 0]] = gb[:, [0, 1]]
                        self.g[gn]['grain'].gbloc = deepcopy(gb)
                    # ---------------------------------------------
                    rmin = np.where(L == 1)[0].min()
                    rmax = np.where(L == 1)[0].max()+1
                    cmin = np.where(L == 1)[1].min()
                    cmax = np.where(L == 1)[1].max()+1
                    # ---------------------------------------------
                    if bbox_ex:
                        # Extract bounding rectangle
                        Rlab = L.shape[0]
                        Clab = L.shape[1]

                        rmin_ex = rmin - int(rmin != 0)
                        rmax_ex = rmax + int(rmin != Rlab)
                        cmin_ex = cmin - int(cmin != 0)
                        cmax_ex = cmax + int(cmax != Clab)
                    # ---------------------------------------------
                    if bbox:
                        # Store the bounds of the bounding box
                        self.g[gn]['grain'].bbox_bounds = [rmin, rmax, cmin, cmax]
                    if bbox:
                        # Store bounding box
                        self.g[gn]['grain'].bbox = np.array(L[rmin:rmax, cmin:cmax],
                                                            dtype=np.uint8)
                    if bbox_ex:
                        # Store the bounds of the extended bounding box
                        self.g[gn]['grain'].bbox_ex_bounds = [rmin_ex, rmax_ex,
                                                              cmin_ex, cmax_ex]
                    if bbox_ex:
                        # Store the extended bounding box
                        self.g[gn]['grain'].bbox_ex = np.array(L[rmin_ex:rmax_ex,
                                                                 cmin_ex:cmax_ex],
                                                               dtype=np.uint8)
                    if make_skim_prop:
                        # Store the scikit-image regionproperties generator
                        self.g[gn]['grain'].make_prop(regionprops, skprop=True)
                    if get_grain_coords:
                        # Make coordinates
                        _coords_ = np.array([[self.xgr[ij[0], ij[1]],
                                              self.ygr[ij[0], ij[1]]]
                                             for ij in self.g[gn]['grain'].loc],
                                             dtype=self.grain_coords_dtype)
                        self.g[gn]['grain'].coords = deepcopy(_coords_)
        print(40*'-')
        self.build_prop(correct_aspect_ratio=True)
        self.are_properties_available = True
        if char_grain_positions:
            #self.char_grain_positions_2d()
            self.char_grain_positions_2d_v1()
        if find_neigh:
            print('Identifying grain neighbours.')
            self.find_neigh()

    def char_morph_2d_v2(self, bso=1, def_feat_name='grain', 
                    bbox=True, bbox_ex=True, npixels=False,
                    npixels_gb=False, identify_pixel_locations=True,
                    area=False, eq_diameter=False,
                    perimeter=False, perimeter_crofton=False,
                    compactness=False, gb_length_px=False, aspect_ratio=False,
                    solidity=False, morph_ori=False, circularity=False,
                    eccentricity=False, feret_diameter=False,
                    major_axis_length=False, minor_axis_length=False,
                    euler_number=False, moments_hu=True, 
                    char_gb=False, char_grain_positions=False, find_neigh=True,
                    make_skim_prop=False, get_grain_coords=True,
                    append=False, saa=True, throw=False, make_pd=True,
                    _redo_lgi_=False, make_grain_object=True
                    ):
        self._char_fx_version_ = 2
        _mgo_ = make_grain_object
        # ---------------------------------------------
        if make_pd:
            # Make data holder for properties
            from upxo._sup.data_templates import pd_templates
            __ = pd_templates()
            __a, __b, __c = __.make_prop2d_df(bbox=bbox, bbox_ex=bbox_ex, 
                    npixels=npixels, npixels_gb=npixels_gb,
                    area=area, eq_diameter=eq_diameter,
                    perimeter=perimeter, perimeter_crofton=perimeter_crofton,
                    compactness=compactness, gb_length_px=gb_length_px,
                    aspect_ratio=aspect_ratio, solidity=solidity,
                    morph_ori=morph_ori, circularity=circularity,
                    eccentricity=eccentricity, feret_diameter=feret_diameter,
                    major_axis_length=major_axis_length,
                    minor_axis_length=minor_axis_length,
                    euler_number=euler_number, moments_hu=moments_hu, 
                    append=append, )
            self.prop_flag, self.prop, self.prop_stat = __a, __b, __c
        # ---------------------------------------------
        from upxo.xtal.mcgrain2d_definitions import grain2d
        # ---------------------------------------------
        # Rlab, Clab = self.lgi.shape[0], self.lgi.shape[1]
        if _redo_lgi_:
            features = self.detect_features_in_image(deepcopy(self.lgi), binary_structure_order=bso)
            self.lgi = deepcopy(features[0])
        use_charecterise_features_in_image_version = 2
        # ---------------------------------------------
        if use_charecterise_features_in_image_version == 1:
            fx1 = self.charecterise_features_in_image
            skprops, bbox_limits_ex, bboxes_ex, coords_dict = fx1(self.lgi, 
                    self.xgr, self.ygr, make_skprops=True, extract_coords=True,
                    throw_bounding_box=True)
        elif use_charecterise_features_in_image_version == 2:
            fx1 = self.charecterise_features_in_image_v2
            _fx1Output_ = fx1(self.lgi, Xgrid=self.xgr, Ygrid=self.ygr,
                              make_skprops=True, extract_coords=True, 
                              throw_bounding_box=True)
            skprops, bbox_limits, bbox_limits_ex, bboxes, bboxes_ex, coords_dict = _fx1Output_
        # ---------------------------------------------
        self.gid = np.array(list(skprops.keys()), dtype=np.int32)
        self.n = len(self.gid)
        features = {}
        _n_features_ = len(self.gid)
        frequency_progress_print = 9
        if _n_features_ >= frequency_progress_print:
            interval = _n_features_ // frequency_progress_print 
        else:
            interval = 1
        # ---------------------------------------------
        def set_val(target, key, value):
            if isinstance(target, dict):
                target[key] = value
            else:
                setattr(target, key, value)
        # ---------------------------------------------
        for fid in self.gid:
            if fid % interval == 0 or fid == _n_features_:
                print(f"{np.round((list(self.gid).index(fid)+1)/_n_features_*100, 1)}%", 
                      end=', ' if fid != _n_features_ else '')
            features[fid] = grain2d() if _mgo_ else dict()
            # ----------------------
            set_val(features[fid], 'm', self.m)
            set_val(features[fid], 'gid', fid)
            set_val(features[fid], '_px_area', self.px_size)
            # ----------------------
            if make_skim_prop:
                features[fid].skprop = skprops[fid] if _mgo_ else None
            # ----------------------
            if identify_pixel_locations:
                set_val(features[fid], 'loc', np.argwhere(self.lgi == fid))
                _ = features[fid].loc.T if _mgo_ else features[fid]['loc'].T
                set_val(features[fid], 'xmin', _[0].min())
                set_val(features[fid], 'xmax', _[0].max())
                set_val(features[fid], 'ymin', _[1].min())
                set_val(features[fid], 'ymax', _[1].max())
                # ----------------------
                '''if _mgo_:
                    features[fid].loc = np.argwhere(self.lgi == fid)
                    _ = features[fid].loc.T
                    features[fid].xmin = _[0].min()
                    features[fid].xmax = _[0].max()
                    features[fid].ymin = _[1].min()
                    features[fid].ymax = _[1].max()
                else:
                    features[fid]['loc'] = np.argwhere(self.lgi == fid)
                    _ = features[fid]['loc'].T
                    features[fid]['xmin'] = _[0].min()
                    features[fid]['xmax'] = _[0].max()
                    features[fid]['ymin'] = _[1].min()
                    features[fid]['ymax'] = _[1].max()'''
            # ----------------------
            if npixels:
                set_val(features[fid], 'npixels', features[fid].loc.shape[0])
                '''features[fid].npixels = features[fid].loc.shape[0]'''
            # ----------------------
            if get_grain_coords:
                set_val(features[fid], 'coords', np.asarray(coords_dict[fid],
                                                dtype=self.grain_coords_dtype))
                '''features[fid].coords = np.asarray(coords_dict[fid],
                                                dtype=self.grain_coords_dtype)'''
            # ----------------------
            if bbox_ex or bbox:
                set_val(features[fid], 'bbox_ex_bounds', bbox_limits_ex[fid])
                set_val(features[fid], 'bbox_ex', bboxes_ex[fid])
                set_val(features[fid], 'bbox_bounds', bbox_limits[fid])
                set_val(features[fid], 'bbox', bboxes[fid])
            # ----------------------
            if char_gb:
                mask = np.zeros_like(self.lgi)
                mask[self.lgi == fid] = 255
                mask = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                gb = np.squeeze(contours[0], axis=1)
                # Interchange the row and column to get into right
                # indexing order
                gb[:, [1, 0]] = gb[:, [0, 1]]
                set_val(features[fid], 'gbloc', gb)
                '''features[fid].gbloc = gb'''

        self.g = features
        if _mgo_:
            self.build_prop(correct_aspect_ratio=True)
        else:
            print('Skipping building properties as grain objects were not created.')

        if char_grain_positions:
            # self.char_grain_positions_2d()
            self.char_grain_positions_2d_v1()

        if find_neigh:
            self.find_neigh(include_central_grain=False, print_msg=False,
                            user_defined_bbox_ex_bounds=True, 
                            bbox_ex_bounds=bbox_limits_ex,
                            update_grain_object=False)
        if throw:
            return features

    def build_grain_pairs(self, neigh_gid):
        """
        Build unique grain pairs from the neighbor list.

        Parameters
        ----------
        neigh_gid : dict
            Dictionary where key is grain ID and value is list of neighbor grain IDs.
        """
        grain_pairs = []
        for grain_id, neighbors in neigh_gid.items():
            for neighbor_id in neighbors:
                if neighbor_id > grain_id:  # Only add pairs where neighbor_id > grain_id to avoid duplicates
                    grain_pairs.append((grain_id, neighbor_id))
        return grain_pairs
    
    def make_graph(self, neigh_gid):
        """
        Create a graph representation from the neighbor list.

        Parameters
        ----------
        neigh_gid : dict
            Dictionary where key is grain ID and value is list of neighbor grain IDs.

        Returns
        -------
        graph
            Graph representation of the grain neighbors.
        """
        import upxo.netops.kmake as kmake
        return kmake.make_gid_net_from_neighlist(neigh_gid)
        
    def identify_grain_boundary_pixels(self, grain_pairs):
        """
        Identify grain boundary pixels for specified grain pairs.
        
        Parameters
        ----------
        grain_pairs : list of tuples
            List of tuples where each tuple contains two grain IDs representing a grain pair.

        Returns
        -------
        dict
            Dictionary where:
            - Key: tuple (grain_id1, grain_id2) as standard Python ints (sorted)
            - Value: Nx2 NumPy array of coordinates (float64)
        """
        # 1. Prepare target list for fast lookup
        # Convert input to sorted tuples of standard ints
        target_pairs = set(tuple(sorted((int(p[0]), int(p[1])))) for p in grain_pairs)
        
        # Use a temp dictionary with lists to aggregate data first
        # (Appending to lists is faster than stacking numpy arrays in a loop)
        temp_store = defaultdict(list)
        
        # --- Horizontal Pass (detects vertical boundaries) ---
        left = self.lgi[:, :-1]
        right = self.lgi[:, 1:]
        
        mask_h = left != right
        rows_h, cols_h = np.nonzero(mask_h)
        
        id_left = left[mask_h]
        id_right = right[mask_h]
        
        for r, c, g1, g2 in zip(rows_h, cols_h, id_left, id_right):
            # Ensure key consistency
            p1, p2 = int(g1), int(g2)
            pair = tuple(sorted((p1, p2)))
            
            if pair in target_pairs:
                # Append [row, col] to the list for this pair
                # We treat r, c as floats
                temp_store[pair].append([float(r), float(c)])

        # --- Vertical Pass (detects horizontal boundaries) ---
        top = self.lgi[:-1, :]
        bottom = self.lgi[1:, :]
        
        mask_v = top != bottom
        rows_v, cols_v = np.nonzero(mask_v)
        
        id_top = top[mask_v]
        id_bottom = bottom[mask_v]
        
        for r, c, g1, g2 in zip(rows_v, cols_v, id_top, id_bottom):
            p1, p2 = int(g1), int(g2)
            pair = tuple(sorted((p1, p2)))
            
            if pair in target_pairs:
                temp_store[pair].append([float(r), float(c)])

        # --- Final Conversion ---
        # Convert lists of lists into Nx2 NumPy arrays
        gb_dict = {}
        for pair, coords_list in temp_store.items():
            # Convert to numpy array of shape (N, 2)
            gb_dict[pair] = np.array(coords_list, dtype=np.float64)

        return gb_dict
    
    def plot_boundaries_standalone(self, gb_dict):
        """
        Plot grain boundaries from the provided dictionary of boundary coordinates.

        Parameters
        ----------
        gb_dict : dict
            Dictionary where:
            - Key: tuple (grain_id1, grain_id2) as standard Python ints (sorted)
            - Value: Nx2 NumPy array of coordinates (float64)
        """
        plt.figure(figsize=(8, 8))
        
        for pair, coords in gb_dict.items():
            # rows (y), cols (x)
            rows = coords[:, 0]
            cols = coords[:, 1]
            
            plt.scatter(cols, rows, s=1)
        
        # CRITICAL: Invert Y axis to match image coordinates (0,0 at top-left)
        plt.gca().invert_yaxis()
        plt.axis('equal') # Ensures pixels are square, not stretched
        plt.title(f"Extracted Boundaries ({len(gb_dict)} segments)")
        plt.show()

    def find_grain_boundary_junction_points(self, xorimap=False, IN=None):
        """
        Identify grain boundary junction points in the microstructure.
        Parameters
        ----------
        xorimap : bool, optional
            If True, the junction points will be calculated for the
            instance number IN in the pxtal dictionary.
            If False, junction points will be calculated for the current
            instance. Default is False.
        IN : int, optional
            Instance number in the pxtal dictionary for which junction
            points are to be calculated if xorimap is True.
            Default is None.
        """
        from scipy.ndimage import generic_filter
        def __find_junctions(pixel_values):
            """
            Function to be applied on each pixel. It checks if the central
            pixel is a junction point. pixel_values: A flattened array of the
            central pixel and its neighbors. Returns 1 if the central pixel
            is a junction, else 0.
            """
            unique_grains = np.unique(pixel_values)
            '''Count the unique grain IDs excluding the background or border
            if needed'''
            count = np.sum(unique_grains > 0)
            return 1 if count >= 3 else 0
        __footprint__ = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        # Apply the filter to identify junctions
        if not xorimap:
            self.gbjp = generic_filter(self.lgi, __find_junctions,
                                       footprint=__footprint__, mode='nearest',
                                       cval=0)
        else:
            if IN in self.pxtal.keys():
                self.pxtal[IN].gbjp = generic_filter(self.pxtal[IN].lgi,
                                                     __find_junctions,
                                                     footprint=__footprint__,
                                                     mode='nearest',
                                                     cval=0)
            else:
                print(f'Invalid Instance number, IN: {IN}')

    def do_single_pixel_grains_exist(self):
        # Check if any single-pixel grains exist in the grain structure
        single_pixel_gids = self.single_pixel_grains
        if len(single_pixel_gids) > 0:
            return True
        else:
            return False
        pass

    def do_straightline_grains_exist(self):
        """
        Check if any straight-line grains exist in the grain structure.
        
        Straight-line grains are grains that are only one pixel wide in at least one 
        dimension, excluding single-pixel grains. These are identified by having a 
        minor axis length of zero when trying to fit an ellipse.
        
        Returns
        -------
        bool
            True if straight-line grains exist, False otherwise.
        
        Notes
        -----
        This method uses the `straight_line_grains` property which identifies grains 
        where skimage cannot fit an ellipse due to unit pixel width. Single pixel 
        grains are excluded from this check.
        
        Examples
        --------
        >>> from upxo.ggrowth.mcgs import mcgs
        >>> pxt = mcgs(study='independent', 
        ...            input_dashboard='input_dashboard.xls')
        >>> pxt.simulate()
        >>> pxt.detect_grains()
        >>> tslice = 10
        >>> pxt.gs[tslice].char_morph_2d(make_skim_prop=True)
        >>> has_straight = pxt.gs[tslice].do_straightline_grains_exist()
        >>> print(f"Straight-line grains present: {has_straight}")
        
        See Also
        --------
        straight_line_grains : Property that returns the IDs of straight-line grains
        do_single_pixel_grains_exist : Check for single-pixel grains
        single_pixel_grains : Property that returns the IDs of single-pixel grains
        """
        straightline_grain_gids, _ = self.straight_line_grains
        
        if len(straightline_grain_gids) > 0:
            return True
        else:
            return False

    def check_for_neigh(self, parent_gid, other_gid):
        """
        Check if other_gid is indeed a O(1) neighbour of parent_gid.

        Parameters
        ----------
        parent_gid:
            Grain ID of the parent.
        other_gid:
            Grain ID of the other grain being checked for O(1) neighbourhood
            with parent_gid.

        Returns
        -------
        True if other_gid is a valid O(1) neighbour of parent_gid, else False.
        """
        return True if other_gid in self.neigh_gid[parent_gid] else False

    def get_two_rand_o1_neighs(self):
        """
        Calculate at random, two neighbouring O(1) grains.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[35].char_morph_2d()
        mcgs.gs[35].find_neigh()
        mcgs.gs[35].neigh_gid
        mcgs.gs[35].get_two_rand_o1_neighs()
        mcgs.gs[35].plot_two_rand_neighs(return_gids=True)
        """
        if self.neigh_gid:
            rand_gid = random.sample(self.gid, 1)[0]
            rand_neigh_rand_grain = random.sample(self.neigh_gid[rand_gid],
                                                  1)[0]
            return [rand_gid, rand_neigh_rand_grain]
        else:
            print('Please build neigh_gid data before using this function.')
            return [None, None]

    def plot_two_rand_neighs(self, return_gids=True):
        """
        Plot two random neighbouring grains.

        Parameters
        ----------
        return_gids: bool
            Flag to return the random neigh gid numbers. Defaults to True.

        Return
        ------
        rand_neigh_gids: list
            random neigh gid numbers. Will be gids if return_gids is True.
            Else, will be [None, None].

        Example
        -------
        Please refer to use in the example provided for the definition,
        get_two_rand_o1_neighs()
        """
        rand_neigh_gids = self.get_two_rand_o1_neighs()
        self.plot_grains_gids(rand_neigh_gids, cmap_name='viridis')
        if return_gids:
            return rand_neigh_gids
        else:
            return [None, None]

    def find_gids_by_mprop(self, mprop='area', method='at', distr_loc='mean',
                           bounds=[(0, 2), (10, 15)], ineq_spec_lb='>=',
                           ineq_spec_ub='<=', validate_ui=True,
                           recalculate_area=False,):
        """
        Find grain IDs based on morphological property criteria.

        Parameters
        ----------
        mprop : str
            Morphological property to filter grains by.
        method : str
            Method to use for filtering. Options are 'at' or 'bounded'.
        distr_loc : str
            Location statistic to use when method is 'at'. Options are
            'mean', 'median', 'min', 'max', or quantiles like 'q1', 'q2', 'q3'.
        ineq_spec_lb : str
            Inequality specification for lower bound when method is 'bounded'.
            Options are '>' or '>='.
        ineq_spec_ub : str
            Inequality specification for upper bound when method is 'bounded'.
            Options are '<' or '<='.
        bounds : list of tuples
            List of (lower_bound, upper_bound) tuples to use when method is
            'bounded'.

        Returns
        -------
        dict
            Dictionary containing:
            - 'mprop': The morphological property used for filtering.
            - 'method': The method used for filtering.
            - 'gids': Numpy array of grain IDs that meet the specified criteria.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        pxt.simulate()
        pxt.detect_grains()
        tslice = 10
        pxt.gs[tslice].char_morph_2d()
        # Find grains with area close to mean area
        mean_area_gids = pxt.gs[tslice].find_gids_by_mprop(mprop='area',
                                                           method='at',
                                                           distr_loc='mean')
        print("Grain IDs with area close to mean area:", mean_area_gids)
        # Find grains with area within specified bounds
        bounded_area_gids = pxt.gs[tslice].find_gids_by_mprop(mprop='area',
                                                              method='bounded',
                                                              bounds=[(50, 100), (200, 300)])
        print("Grain IDs with area within specified bounds:", bounded_area_gids)
        """
        # -----------------------------------------
        # Validations
        if validate_ui:
            if hasattr(self, 'prop') is False and mprop == 'area' and not recalculate_area:
                raise ValueError("Morphological properties not calculated. "
                                "Please run char_morph_2d() first.")
            if mprop not in self.prop.columns:
                raise ValueError(f"Invalid morphological property: {mprop}")
            if method not in ('at', 'bounded'):
                raise ValueError(f"Invalid method: {method}. Choose 'at' or 'bounded'.")
            if distr_loc not in ('mean', 'median', 'min', 'max', 'q1', 'q2', 'q3') and not (distr_loc[0] == 'q' and distr_loc[1:].isnumeric()):
                raise ValueError(f"Invalid distr_loc: {distr_loc}.")
            if ineq_spec_lb not in ('>', '>='):
                raise ValueError(f"Invalid ineq_spec_lb: {ineq_spec_lb}. Choose '>' or '>='.")
            if ineq_spec_ub not in ('<', '<='):
                raise ValueError(f"Invalid ineq_spec_ub: {ineq_spec_ub}. Choose '<' or '<='.")
        # -----------------------------------------
        if mprop == 'area' and recalculate_area:
            prop_values = self.find_grain_size_fast(metric='pxarea', recalculate_gid=True)
        else: 
            prop_values = self.prop[mprop].to_numpy()
        # -----------------------------------------
        # Finding gids
        if method == 'at':
            if distr_loc == 'mean':
                target_value = np.mean(prop_values)
            elif distr_loc == 'median':
                target_value = np.median(prop_values)
            elif distr_loc == 'min':
                target_value = np.min(prop_values)
            elif distr_loc == 'max':
                target_value = np.max(prop_values)
            elif distr_loc == 'q1':
                target_value = np.percentile(prop_values, 25)
            elif distr_loc == 'q2':
                target_value = np.percentile(prop_values, 50)
            elif distr_loc == 'q3':
                target_value = np.percentile(prop_values, 75)
            elif distr_loc[0] == 'q' and distr_loc[1:].isnumeric():
                qnum = int(distr_loc[1:])
                if 0 < qnum < 100:
                    target_value = np.percentile(prop_values, qnum)
                else:
                    raise ValueError(f"Invalid quantile number in distr_loc: {distr_loc}. Must be between 1 and 99.")
            else:
                raise ValueError(f"Invalid distr_loc: {distr_loc}.")
            gids = self.prop.index[np.isclose(prop_values, target_value)].tolist()
        elif method == 'bounded':
            gids = []
            for bound in bounds:
                lb, ub = min(bound), max(bound)
                b_flags = prop_values > lb if ineq_spec_lb == '>' else prop_values >= lb
                b_flags &= prop_values < ub if ineq_spec_ub == '<' else prop_values <= ub
                bound_gids = self.prop.index[b_flags].tolist()
                gids.extend(bound_gids)
            gids = list(set(gids))
        # -----------------------------------------
        # Convert indexing from 0-based to 1-based as these will be featuire ids
        gids = np.array(gids)+1
        # -----------------------------------------
        # Assimilate return data
        _ = {'mprop': mprop, 'method': method, 'gids': gids}
        return _

    def thresholding(self, prop_type='mprop', threshold_type='lower',
                     method='merge',  bso=2,  recalculate_neigh='all',
                     update_grain_object=False, validate_ui=True,
                     recursive_search_and_merge=True, niter=10,
                     kwargs_lower_mprop_threshold={'threshold': 1.0,
                            'pname': 'area', 'sink_metric': 'mean',
                            'recalculate_mprop': True, 'ineq_spec_ub': '<=',
                            'sink_select_uncertainty': [-5, 5]},):
        for iter_number in range(1, niter+1):
            print(f'Iteration {iter_number} of {niter} for lath application.')
            if prop_type == 'mprop' and threshold_type == 'lower':
                self.lower_mprop_thresholding(method=method, bso=bso,
                                              recalculate_neigh=recalculate_neigh,
                                              update_grain_object=update_grain_object,
                                              validate_ui=validate_ui,
                                              recursive_search_and_merge=recursive_search_and_merge,
                                              **kwargs_lower_mprop_threshold)

    def lower_mprop_thresholding(self, threshold=1.0, sink_metric='mean',
                                 method='merge', pname='area', 
                                 bso=2, recalculate_mprop=True, ineq_spec_ub='<=', 
                                 recalculate_neigh='all', update_grain_object=False, 
                                 validate_ui=True, sink_select_uncertainty=[-5, 5],
                                 post_merge_ops_frequency=1, recursive_search_and_merge=True,):
        # Validations
        qnum = None
        if validate_ui:
            if hasattr(self, 'prop') is False:
                raise ValueError("Morphological properties not calculated. "
                                "Please run char_morph_2d() first.")
            if pname not in self.prop.columns:
                raise ValueError(f"{pname} property not calculated. "
                                f"Please run char_morph_2d() with {pname}=True first.") 
            if sink_metric not in ('mean', 'median', 'min', 'max', 'q0', 'q1', 'q2', 'q3', 'q4') and not (sink_metric[0] == 'q' and sink_metric[1:].isnumeric()):
                raise ValueError(f"Invalid sink_metric: {sink_metric}.")
            if method not in ('erode', 'merge'):
                raise ValueError(f"Invalid method: {method}. Choose 'erode' or 'merge'.")
            if ineq_spec_ub not in ('<', '<='):
                raise ValueError(f"Invalid ineq_spec_ub: {ineq_spec_ub}. Choose '<' or '<='.")
            if recalculate_neigh not in ('all', 'specific'):
                raise ValueError(f"Invalid recalculate_neigh: {recalculate_neigh}. Choose 'all' or 'specific'.")
            if post_merge_ops_frequency == 0:
                raise Warning("post_merge_ops_frequency is set to 0. No post-merge operations will be performed.")
            if post_merge_ops_frequency < 0:
                raise ValueError("post_merge_ops_frequency cannot be negative.")
            if type(post_merge_ops_frequency) not in dth.dt.INTEGERS:
                raise ValueError("post_merge_ops_frequency must be an integer.")
            if post_merge_ops_frequency > 1:
                raise Warning("post_merge_ops_frequency > 1 could make the lath application slower.")
        # -----------------------------------------
        # Find grain areas and gids satisfying threshold specification
        if recalculate_mprop and pname == 'area':
            grain_areas = self.find_grain_size_fast(metric='pxarea', recalculate_gid=True)
            gids_mprop = np.where(grain_areas < threshold)[0]+1  # +1 for gid indexing
        if not recalculate_mprop and pname == 'area':
            if 'area' not in self.prop.columns:
                raise ValueError(f"Invalid morphological property: area. Please run char_morph_2d() with area=True first.")
            grain_areas = self.prop['area'].to_numpy()
            gids_mprop = self.find_gids_by_mprop(mprop='area', method='bounded',
                                                 bounds=[(0, threshold)],
                                                 ineq_spec_ub=ineq_spec_ub,
                                                 recalculate_area=False)['gids']
        # -----------------------------------------
        if len(gids_mprop) == 0:
            print('No grains found below specified threshold value. Exiting lath application.')
            return 'thresholding complete - no grains found below threshold'
        # -----------------------------------------
        # Calculate neigh_gid subset
        if recalculate_neigh == 'all':
            self.find_neigh()
            NG = GidOps.extract_neigh_gid_subset(neigh_gids=self.neigh_gid,
                                                 subset_gids=gids_mprop, 
                                                 type_correction=True,
                                                 validate_input=False)
        elif recalculate_neigh == 'specific':
            NG = {gid: self.find_neigh_gid(gid, include_central_grain=False, throw=True,
                                           update_grain_object=update_grain_object)
                    for gid in gids_mprop}
        # -----------------------------------------
        # Build area dictionary for neighbour grains
        NG_areas = {int(cgid): np.array([grain_areas[gid-1] for gid in neighs], dtype=np.float32)
                    for cgid, neighs in NG.items()}
        # -----------------------------------------
        ssu = sink_select_uncertainty
        NG_sinks = {}
        if method == 'merge' and sink_metric == 'min':
            NG_sinks = manm.select_sinks_min_area(NG, NG_areas, ssu, NG_sinks,
                                                  force_assign=False)
        elif method == 'merge' and sink_metric == 'mean':
            NG_sinks = manm.select_sinks_mean_area(NG, NG_areas, ssu, NG_sinks,
                                                   force_assign=False)
        elif method == 'merge' and sink_metric == 'max':
            NG_sinks = manm.select_sinks_max_area(NG, NG_areas, ssu, NG_sinks,
                                                   force_assign=False)
        elif method == 'merge' and sink_metric == 'median':
            NG_sinks = manm.select_sinks_median_area(NG, NG_areas, ssu, NG_sinks, 
                                                     force_assign=False)
        elif method == 'merge' and sink_metric[0] == 'q' and sink_metric[1:].isnumeric():
            NG_sinks = manm.select_sinks_quantile_area(NG, NG_areas, ssu, NG_sinks, 
                                                       sink_metric=sink_metric, force_assign=False)
        # -----------------------------------------
        print(NG_sinks)
        # recursive_search_and_merge
        # post_merge_ops_frequency
        if method == 'merge':
            '''
            Perform grain merges based on identified sinks.

            Example
            -------
            self.apply_lath(lath=1.0, sink_metric='mean', method='merge',
                   recalculate_area=True, ineq_spec_ub='<=', recalculate_neigh='all',
                   update_grain_object=False, validate_ui=True, sink_select_uncertainty=[-5, 5],
                   post_merge_ops_frequency=1, recursive_search_and_merge=True)
            '''
            for other_gid, parent_gid in NG_sinks.items():
                self._merge_two_grains_(parent_gid, other_gid, print_msg=False)

                # Re-number the lgi
                old_gids = np.unique(self.lgi)
                new_gids = np.arange(start=1, stop=np.unique(self.lgi).size+1, step=1)
                for og, ng in zip(old_gids, new_gids):
                    self.lgi[self.lgi == og] = ng

                if self._char_fx_version_ == 1:
                    self.char_morph_2d(bbox=True, bbox_ex=True, npixels=False,
                        npixels_gb=False, area=True, eq_diameter=False,
                        perimeter=False, perimeter_crofton=False,
                        compactness=False, gb_length_px=False, aspect_ratio=False,
                        solidity=False, morph_ori=False, circularity=False,
                        eccentricity=False, feret_diameter=False,
                        major_axis_length=False, minor_axis_length=False,
                        euler_number=False, moments_hu=False, 
                        append=False, saa=True, throw=False,
                        char_grain_positions=False, find_neigh=False,
                        char_gb=False, make_skim_prop=True,
                        get_grain_coords=True)
                elif self._char_fx_version_ == 2:
                    self.char_morph_2d_v2(bso=bso, def_feat_name='grain', char_gb=False,
                            char_grain_positions=False, find_neigh=True, saa=True,
                            throw=False)
                # Reset gid
                self.gid = np.array(range(1, np.unique(self.lgi).size+1), dtype=np.int32)
                # reset number of grains
                self.n = len(self.gid)
                # Calculate new neighbourhood database
                self.find_neigh(include_central_grain=False, print_msg=False)
        # -----------------------------------------
        if method == 'erode':
            # Identify subset of gb_pixels for only the specific grain neighbours
            grain_pairs = self.build_grain_pairs(NG)
            # Identify grain boundary pixels for all grain pairs
            gb_pixels = self.identify_grain_boundary_pixels(grain_pairs)
        # -----------------------------------------

    def _merge_two_grains_(self, parent_gid, other_gid, print_msg=False):
        """Low level merge operartion. No checks done. Just merging.

        Parameters
        ----------
        parent_gid: int
            Parent grain ID number.
        other_gid: int
            Otrher grain ID number.
        print_msg: bool
            Defgaults to False.

        Returns
        -------
        None

        Usage
        -----
        Internal use only.
        """
        self.lgi[self.lgi == other_gid] = parent_gid
        if print_msg:
            print(f"Grain {other_gid} merged with grain {parent_gid}.")

    def merge_two_neigh_grains(self, parent_gid, other_gid,
                               check_for_neigh=True, simple_merge=True):
        """
        Merge other_gid grain to the parent_gid grain.

        Paramters
        ---------
        parent_gid:
            Grain ID of the parent.
        other_gid:
            Grain ID of the other grain being merged into the parent.
        check_for_neigh: bool.
            If True, other_gid will be checked if it can be merged to the
            parent grain. Defaults to True.

        Returns
        -------
        merge_success: bool
            True, if successfully merged, else False.
        """
        def MergeGrains():
            if simple_merge:
                self._merge_two_grains_(parent_gid, other_gid, print_msg=False)
                merge_success = True
            else:
                print("Special merge process. To be developed.")
                merge_success = False  # As of now, this willd efault to False.
            return merge_success
        # ---------------------------------------
        if not check_for_neigh:
            merge_success = MergeGrains()
        else:
            if check_for_neigh and not self.check_for_neigh(parent_gid, other_gid):
                # print('Check for neigh failed. Nothing merged.')
                merge_success = False
            # ---------------------------------------
            if any((check_for_neigh, self.check_for_neigh(parent_gid, other_gid))):
                merge_success = MergeGrains()
                # print(f"Grain {other_gid} merged with grain {parent_gid}.")
        return merge_success

    def perform_post_grain_merge_ops(self, merge_success, merged_gid):
        """
        Perform post grain merge operations.

        Parameters
        ----------
        merge_success: bool
            If True, post grain merge operations will be performed.
        merged_gid: int
            Grain ID of the grain that was merged into another grain.
        """
        self.renumber_gid_post_grain_merge(merged_gid)
        self.recalculate_ngrains_post_grain_merge()
        # Update lgi
        # Update neigh_gid
        pass

    def renumber_gid_post_grain_merge(self, merged_gid):
        """
        Renumber gid post grain merge.

        Parameters
        ----------
        merged_gid: int
            Grain ID of the grain that was merged into another grain.
        """
        # self._gid_bf_merger_ = deepcopy(self.gid) # May nor be needed
        GID_left = self.gid[0:merged_gid-1]
        GID_right = [gid-1 for gid in self.gid[merged_gid:]]
        self.gid = np.array(GID_left + GID_right, dtype=np.int32)

    def recalculate_ngrains_post_grain_merge(self):
        """
        Recalculate the number of grains post grain merge.
        """
        # gid must have been recalculated for tjhis as a pre-requisite.
        self.n = len(self.gid)

    def renumber_lgi_post_grain_merge(self, merged_gid):
        """
        Renumber lgi post grain merge.

        Parameters
        ----------
        merged_gid: int
            Grain ID of the grain that was merged into another grain.
        """
        # LGI_left = self.lgi[self.lgi < merged_gid]
        self.lgi[self.lgi > merged_gid] -= 1

    def validate_propnames(self, mpnames, return_type='dict'):
        """
        Validate an iterable containing propnames. Mostly for internal use.

        Parameters
        ----------
        mpnames: dth.dt.ITERABLES
            Property names to be validated.
        return_type: str
            Type of function return. Valid choices: dict (default), list,
            tuple.

        Returns
        -------
        validation: dict (default) / tuple
            If return_type is other than dictionary and either list or
            tuple, or numpy array, only tuple will be returned. If return_type
            is dict, then dict with mpnames keys and their individual
            validations will be the values. The values will all be bool.
            If a property is a valid property, then True, else False.

        Example
        -------
        self.validate_propnames(['area', 'perimeter', 'solidity'])
        """
        _ = {pn: pn in self.valid_mprops.keys() for pn in mpnames}
        if return_type == 'dict':
            return _
        elif return_type in ('list', 'tuple'):
            return tuple(_.values())
        else:
            raise ValueError('Invalid return_type specification.')

    def check_mpnamevals_exists(self, mpnames, return_type='dict'):
        """
        Check if the values for the given mpnames exist in self.prop.

        Parameters
        ----------
        mpnames: dth.dt.ITERABLES
            List of user specified names of morphological properties.
        return_type: str
            Type of function return. Valid choices: dict (default), list,
            tuple.

        Returns
        -------
        exists: dict (default) / list / tuple
            If return_type is dict, then dict with mpnames keys and their
            individual existences will be the values. The values will all be
            bool. If a property value exists in self.prop, then True, else
            False.
            If return_type is other than dictionary and either list or
            tuple, or numpy array, only tuple will be returned.
        """
        if return_type == 'dict':
            return {mpn: mpn in self.prop.columns for mpn in mpnames}
        elif return_type in ('list', 'tuple'):
            return [mpn in self.prop.columns for mpn in mpnames]

    def set_mprops(self, mpnames, char_grain_positions=True,
                   char_gb=False, set_grain_coords=True,
                   saa=True, throw=False):
        """
        Targetted use of char_morph_2d.

        Parameters
        ----------
        mpnames: dth.dt.ITERABLES
            List of user specified names of morphological properties
        char_grain_positions: bool
            If True, grain positions will also be characterized. Defaults to
            True.
        char_gb:
            If True, grain boundary will be characterized/re-characterized.
            Degfaults to False.
        set_grain_coords:
            If True, self.g[gn]['grain'].coords will be updated, else not, for
            all gn in self.gid.

        Example
        -------
        self.set_mprops(mpnames, recharacterize=True)
        """
        VALMPROPS = deepcopy(self.valid_mprops)
        # ----------------------------
        if not all(self.validate_propnames(mpnames, return_type='tuple')):
            raise ValueError('Invalid propnames.')
        # ----------------------------
        for mpn in mpnames:
            # Check if each user input morph0ological propetrty name and
            # corresponding values exist in self.prop pd dataFrame.
            VALMPROPS[mpn] = True
        if char_grain_positions:
            VALMPROPS['char_grain_positions'] = True
        # ----------------------------
        self.char_morph_2d(bbox=True, bbox_ex=True, append=False, saa=saa,
                           throw=False, char_gb=char_gb, make_skim_prop=True,
                           get_grain_coords=set_grain_coords, **VALMPROPS)
        # ----------------------------
        if throw:
            mprop_values = {mpn: self.prop[mpn].to_numpy() for mpn in mpnames}
        else:
            mprop_values = {mpn: None for mpn in mpnames}

        return mprop_values

    def get_mprops(self, mpnames, set_missing_mprop=False):
        """
        Get values of mpnames.

        Parameters
        ----------
        mpnames: dth.dt.ITERABLES
            List of user specified names of morphological properties.
        set_missing_mprop: bool
            If True, missing morphological property values will be
            characterized and set before returning the values. Defaults to
            False.

        Returns
        -------
        mprop_values: dict
            Dictionary with mpnames as keys and their corresponding values
            as numpy arrays.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[mcgs.m[-1]].char_morph_2d(bbox=True, bbox_ex=True,
                                     area=True,aspect_ratio=True,
                                     make_skim_prop=True,)

        mpnames=['area', 'aspect_ratio', 'perimeter', 'solidity']
        mcgs.gs[mcgs.m[-1]].prop
        mprop_values = mcgs.gs[mcgs.m[-1]].get_mprops(mpnames,
                                                      set_missing_mprop=True)
        mprop_values
        """
        if not all(self.validate_propnames(mpnames, return_type='list')):
            raise ValueError('Invalid mpname values.')
        val_exists = self.check_mpnamevals_exists(mpnames, return_type='dict')
        # ----------------------------
        if not set_missing_mprop:
            mprop_values = {}
            for mpn in mpnames:
                if val_exists[mpn]:
                    mprop_values[mpn] = self.prop[mpn].to_numpy()
                else:
                    mprop_values[mpn] = None
        # ----------------------------
        if set_missing_mprop:
            set_propnames = [mpn for mpn in mpnames if not val_exists[mpn]]
            self.set_mprops(mpnames, char_grain_positions=False,
                            char_gb=False, set_grain_coords=False)
            mprop_values = self.get_mprops(mpnames, set_missing_mprop=False)

        return mprop_values

    def validata_gids(self, gids):
        """
        Validate the gid values.

        Parameters
        ----------
        gids: Iterable of ints.

        Returns
        -------
        True if all gids are in self.gid else False
        """
        return all([gid in self.gid for gid in gids])

    def get_gids_in_params_bounds(self,
                                  search_gid_source='all',
                                  search_gids=None,
                                  mpnames=['area', 'aspect_ratio',
                                           'perimeter', 'solidity'],
                                  fx_stats=[np.mean, np.mean, np.mean, np.mean],
                                  pdslh=[[50, 50], [50, 50], [50, 50], [50, 50]],
                                  param_priority=[1, 2, 3, 2],
                                  plot_mprop=True
                                  ):
        """
        Get gids of grains whose morphological property values lie within
        user specified bounds.

        Parameters
        ----------
        search_gid_source: str
            Source of gids to be searched. Valid choices:
            'all' : All gids in self.gid will be searched.
            'user': Only user provided gids in search_gids will be searched.
        search_gids: Iterable of ints
            User provided gids to be searched. Only valid if
            search_gid_source is 'user'.
        mpnames: dth.dt.ITERABLES
            List of user specified names of morphological properties.
        fx_stats: dth.dt.ITERABLES
            List of functions to compute the statistic of the morphological
            property. The length of fx_stats must be same as length of
            mpnames. Valid functions include numpy functions like np.mean,
            np.median, np.std, etc.
        pdslh: dth.dt.ITERABLES
            List of lists containing percentages of distance from stat to
            minimum and stat to maximum. The length of pdslh must be same as
            length of mpnames. Each element of pdslh must be a list of two
            numbers. The first number is percentage of distance from stat to
            minimum and the second number is percentage of distance from stat
            to maximum.
        param_priority: dth.dt.ITERABLES
            List of integers specifying the priority of each morphological
            property. The length of param_priority must be same as length of
            mpnames. Higher the number, higher the priority.
        plot_mprop: bool
            If True, plots the morphological property maps with bounds
            indicated in the title.

        Returns
        -------
        GIDs: dict
            Dictionary containing the following keys:
            'intersection': List of gids that lie within the bounds of all
                            morphological properties.
            'union': List of gids that lie within the bounds of at least one
                     morphological property.
            'presence': Dictionary with gids as keys and number of
                         morphological properties that the gid lies within
                         bounds as values.
            'mpmapped': Dictionary with morphological property names as keys
                         and list of gids that lie within the bounds of the
                         morphological property as values.
        VALIND: dict
            Dictionary containing the following keys:
            'stat': Dictionary with morphological property names as keys and
                     their corresponding statistic values as values.
            'statmap': List of functions used to compute the statistic of
                        each morphological property.
            'bounds': Dictionary with morphological property names as keys
                       and their corresponding bounds as values.
            'indices': Dictionary with morphological property names as keys
                        and list of indices of grains that lie within the
                        bounds of the morphological property as values.

        Example
        -------
        """
        # Validations
        # ---------------------------
        pname_val = self.validate_propnames(mpnames, return_type='dict')
        # pname_val = mcgs.gs[35].validate_propnames(mpnames, return_type='dict')
        mprop_values = self.get_mprops(mpnames, set_missing_mprop=True)
        # mcgs.gs[35].prop
        # mprop_values = mcgs.gs[35].get_mprops(mpnames, set_missing_mprop=True)
        # ---------------------------
        '''Sub-select gids as per user request.'''
        if search_gid_source == 'user' and dth.IS_ITER(search_gids):
            if self.validata_gids(search_gids):
                search_gids = np.sort(search_gids)
                for mpn in mpnames:
                    mprop_values[mpn] = mprop_values[mpn][search_gids]
        # ---------------------------
        '''Data processing and extract indices of parameters for parameter
        values valid to the user provided bound.'''
        mprop_KEYS = list(mprop_values.keys())
        mprop_VALS = list(mprop_values.values())
        mpinds = {mpn: None for mpn in mprop_KEYS}
        mp_stats = {mpn: None for mpn in mprop_KEYS}
        mp_bounds = {mpn: None for mpn in mprop_KEYS}
        for i, (KEY, VAL) in enumerate(zip(mprop_KEYS, mprop_VALS)):
            masked_VAL = np.ma.masked_invalid(VAL)
            # Compute the stat value of the morpho prop
            mp_stat = fx_stats[i](masked_VAL)
            mp_stats[KEY] = mp_stat
            # COmpute min and max of the mprop array
            mp_gmin, mp_gmax = np.min(masked_VAL), np.max(masked_VAL)
            # Compute distance from stat to low and stat to high
            mp_dlow, mp_dhigh = abs(mp_stat-mp_gmin), abs(mp_stat-mp_gmax)
            # Compute bounds of arrays using varper
            dfsmin = pdslh[i][0]/100  # Distance factor from stat to prop min.
            dfsmax = pdslh[i][1]/100  # Distance factor from stat to prop max.
            # Compute lower bound and upper boubnd
            boundlow = mp_stat - dfsmin*mp_dlow
            boundhigh = mp_stat + dfsmax*mp_dhigh
            mp_bounds[KEY] = [boundlow, boundhigh]
            # Mask the mprop array and get indices
            mpinds[KEY] = np.where((VAL >= boundlow) & (VAL <= boundhigh))[0]
            # ---------------------------

        # Find the intersection
        intersection = find_intersection(mpinds.values())
        # Find the union with counts
        union, counts = find_union_with_counts(mpinds.values())
        # Copnvert array indices to gid notation.
        intersection = [i+1 for i in intersection]
        union = [u+1 for u in union]
        counts = {c+1: v for c, v in counts.items()}
        mpinds_gids = {}
        for mpn in mpinds:
            mpinds_gids[mpn] = [i+1 for i in mpinds[mpn]]
        # Collate the GID related results
        GIDs = {'intersection': intersection,
                'union': union,
                'presence': counts,
                'mpmapped': mpinds_gids}
        # Collate the Values and Indices related results
        VALIND = {'stat': mp_stats,
                  'statmap': fx_stats,
                  'bounds': mp_bounds,
                  'indices': mpinds,
                  }

        if plot_mprop:
            fig, ax = plt.subplots(nrows=1, ncols=len(GIDs['mpmapped'].keys()),
                                   figsize=(5, 5), dpi=120, sharey=True)
            for i, mpn in enumerate(GIDs['mpmapped'].keys(), start=0):
                LGI = deepcopy(self.lgi)
                if len(GIDs['mpmapped'][mpn]) > 0:
                    for gid in self.gid:
                        if gid in GIDs['mpmapped'][mpn]:
                            pass
                        else:
                            LGI[LGI == gid] = -10
                ax[i].imshow(LGI, cmap='nipy_spectral')
                bounds = ", ".join(f"{b:.2f}" for b in VALIND['bounds'][mpn])
                ax[i].set_title(f"{mpn}: bounds: [{bounds}]", fontsize=10)

        return GIDs, VALIND

    def get_gid_mprop_map(self, mpropname, querry_gids):
        """
        Provide gid mapped values of a valid mprop for valid querry_gids.

        Parameters
        ----------
        mpropname: str
            Name of the morphological property whose values are to be
            mapped to the querry_gids.
        querry_gids: dth.dt.ITERABLES
            Iterable of valid gids for which the mprop values are to be
            mapped.

        Returns
        -------
        gid_mprop_map: dict
            Dictionary with querry_gids as keys and their corresponding
            mprop values as values.
        """
        # Validations
        self.validate_propnames([mpropname])
        self.validata_gids(querry_gids)
        # ------------------------------------
        if mpropname in self.prop.columns:
            mpvalues = self.prop.loc[[gid-1 for gid in list(querry_gids)],
                                     'aspect_ratio'].to_numpy()
            gid_mprop_map = {gid: mpv
                             for gid, mpv in zip(querry_gids, mpvalues)}
            return gid_mprop_map
        else:
            raise ValueError(f'mpropname must be in {list(self.prop.columns())}.')

    def map_scalar_to_lgi(self, scalars_dict, default_scalar=-1,
                          plot=True, throw_axis=True, plot_centroid=True,
                          plot_gid_number=True,
                          title='title',
                          centroid_kwargs={'marker': 'o',
                                           'mfc': 'yellow',
                                           'mec': 'black',
                                           'ms': 2.5},
                          gid_text_kwargs={'fontsize': 10},
                          title_kwargs={'fontsize': 10},
                          label_kwargs={'fontsize': 10}):
        """
        Map to LGI, the gid keyed values in scalars_dict.

        Parameters
        ----------
        scalars_dict: dict
            Dictionary with gids as keys and scalar values as values.
        default_scalar: float
            Default scalar value to be assigned to gids not in scalars_dict.
            Defaults to -1.
        plot: bool
            If True, plot the mapped LGI. Defaults to True.
        throw_axis: bool
            If True, returns the axis object along with the mapped LGI.
            Defaults to True.
        plot_centroid: bool
            If True, plots the centroids of the grains. Defaults to True.
        plot_gid_number: bool
            If True, plots the gid number at the centroid of the grains.
            Defaults to True.
        title: str
            Title of the plot. Defaults to 'title'.
        centroid_kwargs: dict
            Keyword arguments for plotting centroids. Defaults to
            {'marker': 'o', 'mfc': 'yellow', 'mec': 'black', 'ms': 2.5}.
        gid_text_kwargs: dict
            Keyword arguments for plotting gid numbers. Defaults to
            {'fontsize': 10}.
        title_kwargs: dict
            Keyword arguments for the plot title. Defaults to
            {'fontsize': 10}.
        label_kwargs: dict
            Keyword arguments for the plot labels. Defaults to
            {'fontsize': 10}.

        Returns
        -------
        result: dict
            Dictionary with the following keys:
            'lgi': Mapped LGI as a numpy array.
            'ax': Axis object if throw_axis is True, else None.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 10
        def_neigh = pxt.gs[tslice].get_upto_nth_order_neighbors_all_grains_prob

        neigh1 = def_neigh(1.38, recalculate=False, include_parent=True)

        sf_no = pxt.gs[tslice]



        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[35].char_morph_2d(bbox=True, bbox_ex=True, area=True,
                                  aspect_ratio=True,
                                  make_skim_prop=True,)
        GIDs, VALIND = mcgs.gs[35].get_gids_in_params_bounds(mpnames=['aspect_ratio', 'area'],
                                              fx_stats=[np.mean, np.mean],
                                              pdslh=[[50, 30], [50, 30]], plot_mprop=False
                                              )
        mcgs.gs[35].map_scalar_to_lgi(GIDs['presence'], default_scalar=-1,
                              plot=True, throw_axis=True)

        gid_mprop_map = mcgs.gs[35].get_gid_mprop_map('aspect_ratio',
                                                      GIDs['mpmapped']['aspect_ratio'])
        MPLGIAX = mcgs.gs[35].map_scalar_to_lgi(gid_mprop_map, default_scalar=-1,
                              plot=True, throw_axis=True)
        """
        # Validations
        self.validata_gids(scalars_dict.keys())
        # -------------------
        LGI = deepcopy(self.lgi).astype(float)
        for gid in self.gid:
            if gid in scalars_dict.keys():
                LGI[LGI == gid] = scalars_dict[gid]
            else:
                LGI[LGI == gid] = default_scalar
        # -------------------
        if plot:
            # VMIN, VMAX = min(scalars_dict.values()), max(scalars_dict.values())
            plt.figure(figsize=(5, 5), dpi=120)
            plt.imshow(LGI, cmap='viridis')
            if plot_centroid or plot_gid_number:
                centroid_x, centroid_y = [], []
                for gid in scalars_dict.keys():
                    centroid_x.append(self.xgr[self.lgi == gid].mean())
                    centroid_y.append(self.ygr[self.lgi == gid].mean())
            if plot_centroid:
                plt.plot(centroid_x, centroid_y, linestyle='None',
                         marker=centroid_kwargs['marker'],
                         mfc=centroid_kwargs['mfc'],
                         mec=centroid_kwargs['mec'],
                         ms=centroid_kwargs['ms'])
            if plot_gid_number:
                for i, (cenx, ceny) in enumerate(zip(centroid_x,
                                                     centroid_y), start=1):
                    plt.text(cenx, ceny, str(i),
                             fontsize=gid_text_kwargs['fontsize'])
            ax = plt.gca()
            ax.set_title('Title', fontsize=10)
            ax.set_xlabel(r"X-axis, $\mu m$", fontsize=10)
            ax.set_ylabel(r"Y-axis, $\mu m$", fontsize=10)
            plt.colorbar()
        # -------------------
        if plot and throw_axis:
            return {'lgi': LGI, 'ax': ax}
        else:
            return {'lgi': LGI, 'ax': None}

    def merge_two_neigh_grains_simple(self,
                                      method_id='1',
                                      method_params_parent_sel=['area'],
                                      method_params_other_sel=['area'],
                                      method_params_merging=['area'],
                                      parent_gid=[],
                                      return_gids=True,
                                      plot_gs_bf=True,
                                      plot_gs_af=True,
                                      plot_area_kde_diff=True,
                                      bandwidth=1.0):
        """
        Find two random neighbouring grains and merge them.

        Parameters
        ----------
        method_id: int
            0: parenmt_gid will be selected at random and other_gid will also
                be selected at random.
            1: parent_gid should be provided by user and othet_gid should also
                be provided by user.
            2: parent_gid sahould be provide by user and other_gid will be
                selected at random.
            NOTE: other_grain will allways be O(1) neighbour of parent_grain.
        method_params_parent_sel: str
            Morphological parameter of choice for parent grain selection.
        method_params_other_sel: str
            Morphological parameter of choice for other grain selection.
        method_params_merging: str
            Morphological parameter of choice for merging decision makjing.
        plot_bf: bool
            Plot grain structure before merging. Defaults to True.
        plot_af: bool
            Plot grain structure after merging. Defaults to True.

        Returns
        -------
        gids: list
            [parent_gid, other_gid]. Other_gid merged into parent_gid.
        """
        def plot_kde_difference(area1, area2, bandwidth=1):
            """
            Calculates KDEs for two arrays of data and plots their difference.

            Parameters
            ----------
            area1: np.ndarray
                The first array of data.
            area2: np.ndarray
                The second array of data.
            bandwidth: float, optional
                The bandwidth (smoothing parameter) for KDEs (default: 0.2).
            """
            plt.figure(figsize=(5,5), dpi=120)
            kde1 = sns.kdeplot(area1, bw_adjust=bandwidth, fill=True,
                               label='Area 1', color='blue')
            kde2 = sns.kdeplot(area2, bw_adjust=bandwidth, fill=True,
                               label='Area 2', color='orange')
            # Get the KDE curve data
            x1, y1 = kde1.get_lines()[0].get_data()
            x2, y2 = kde2.get_lines()[0].get_data()
            # Interpolate if x values don't exactly match
            # (to ensure we can subtract)
            y2_interp = np.interp(x1, x2, y2)
            # Calculate and plot the difference
            y_diff = y1 - y2_interp
            plt.fill_between(x1, y_diff, 0, color='green',
                             alpha=0.5, label='Difference (Area 1 - Area 2)')
            # Label axes and add a title
            plt.xlabel('Area')
            plt.ylabel('Density')
            plt.title('KDEs of area distributions and their difference.')
            plt.legend()
            plt.show()
        # ============================================================
        if method_id == '1':
            '''parenmt_gid will be selected at random and other_gid will also
            be selected at random. NOTE: other_grain will allways be O(1)
            neighbour of parent_grain.'''
            parent_gid, other_gid = self.get_two_rand_o1_neighs()
        if method_id == '2':
            '''parent_gid should be provided by user and othet_gid should also
            be provided by user.'''
            parent_gid = parent_gid
            other_gid = self.get_o1_neigh(parent_gid)
            parent_gid, other_gid = self.get_two_rand_o1_neighs()
        if method_id == '3':
            '''parent_gid sahould be provide by user and other_gid will be
            selected at random. NOTE: other_grain will allways be O(1)
            neighbour of parent_grain.'''
            pass
        if method_id == '1-stat($varper$)':
            '''method1 + more. Below provides deytails.
            stat: statistic. Valids: mean, median.
            varper: percentage variation in the stat defining target area
            bound for parent_gid.
            '''
            pass
        # -------------------------------------
        if plot_gs_bf:
            self.plotgs(plot_centroid=True, plot_gid_number=True,
                        plot_cbar=False,
                        title=f'Before merging {other_gid} into {parent_gid}.')
        # -------------------------------------
        if plot_area_kde_diff:
            # Get area before merging
            area_bf = self.prop['area'].to_numpy()
        # -------------------------------------
        self.merge_two_neigh_grains(parent_gid, other_gid,
                                    check_for_neigh=False,
                                    simple_merge=True,)
        # -------------------------------------
        if plot_gs_af:
            self.plotgs(plot_centroid=True,
                        plot_gid_number=True,
                        plot_cbar=False,
                        title=f'After merging {other_gid} into {parent_gid}')
        # -------------------------------------
        if plot_area_kde_diff:
            area_af = deepcopy(area_bf)
            area_af[parent_gid-1] += area_af[other_gid-1]
            area_af = np.delete(area_af, other_gid-1)
            # Get area after merging
            plot_kde_difference(area_bf, area_af, bandwidth=bandwidth)
        # -------------------------------------
        if return_gids:
            return parent_gid, other_gid

    def merge_neigh_grains(self, gid_pairs,
                           check_for_neigh=True, simple_merge=True):
        """
        Merge multiple pairs of neighbouring grains.

        Parameters
        ----------
        gid_pairs: dth.dt.ITERABLES
            Iterable of tuples containing pairs of grain IDs to be merged.
            Each tuple should be in the form (parent_gid, other_gid).
        check_for_neigh: bool
            If True, each pair will be checked for neighbourhood before
            merging. Defaults to True.
        simple_merge: bool
            If True, simple merging will be performed. Defaults to True.
        
        Returns
        -------
        None
        """
        hit = 0
        for parent_gid, other_gid in gid_pairs:
            if self.check_for_neigh(parent_gid, other_gid):
                self.merge_two_neigh_grains(parent_gid, other_gid,
                                            check_for_neigh=check_for_neigh,
                                            simple_merge=simple_merge)

    def set_twingen(self, vf=0.2, tspec='absolute', trel='minil',
                    tdis='user', t=[0.2, 0.5, 0.6, 0.7], tw=[1, 1, 1, 1],
                    tmin=0.2, tmean=0.5, tmax=1.0,
                    nmax_pg=1, placement='centroid', factor_min=0.0,
                    factor_max=1.0,
                    ):
        """
        Set twin generation parameters.

        Parameters
        ----------
        vf: float
            Twin volume fraction.
        tspec: str
            Twin thickness specification. Valid choices:
            'absolute': Absolute thickness values will be used.
            'relative': Relative thickness values will be used.
            'minil': Thickness values will be specified as multiples of
                     minimum inter-lattice distance.
        trel: str
            Twin thickness relation. Valid choices:
            'minil': Thickness values will be specified as multiples of
                     minimum inter-lattice distance.
            'grain_size': Thickness values will be specified as multiples of
                           grain size.
        tdis: str
            Twin thickness distribution. Valid choices:
            'user': User provided thickness values will be used.
            'normal': Normal distribution will be used.
            'lognormal': Log-normal distribution will be used.
            'uniform': Uniform distribution will be used.
        t: dth.dt.ITERABLES
            List of twin thickness values. Only valid if tdis is 'user'.
        tw: dth.dt.ITERABLES
            List of twin weights corresponding to the twin thickness values.
            Only valid if tdis is 'user'.
        tmin: float
            Minimum twin thickness. Only valid if tdis is not 'user'.
        tmean: float
            Mean twin thickness. Only valid if tdis is not 'user'.
        tmax: float
            Maximum twin thickness. Only valid if tdis is not 'user'.
        nmax_pg: int
            Maximum number of twins per grain.
        placement: str
            Twin placement method. Valid choices:
            'centroid': Twins will be placed at the centroid of the grain.
            'random': Twins will be placed at random locations within the
                      grain.
        factor_min: float
            Minimum factor for twin placement. Only valid if placement is
            'random'.
        factor_max: float
            Maximum factor for twin placement. Only valid if placement is
            'random'. 

        Returns
        -------
        None
        """
        self.twingen = twingen(vf=0.2, tmin=0.2, tmean=0.5, tmax=1.0,
                     tdis='user', tvalues=[0.2, 0.5, 1.0, 0.75],
                     allow_partial=True, partial_prob=0.2)

    def introduce_single_twins(self, GIDs=[1], full_twin=True,
                               throw_lgi=True,
                               LFAL_kwargs={'factor': 0.5,
                                            'angle_min': 0,
                                            'angle_max': 360,
                                            'length': 50},
                               twdis_kwargs={'max_count_per_grain': 1,
                                             'min_thickness': 4.5,
                                             'mean_thickness': 4.5,
                                             'max_thickness': 4.5,
                                             'distribution': 'normal',
                                             'variance': 1.0,
                                             },
                               plotgs_original=True,
                               plotgs_twinned=True,
                               save_to_features=True,
                               ):
        """
        Introduce twinned grain features into self.lgi.

        This method creates twin lamellae within specified grains by defining slip lines
        supports visualization of grain structures before and after twin introduction,
        and can optionally store the twin features for later analysis.

        Parameters
        ----------        
        GIDs : list of int, optional
            List of grain IDs to be twinned. Default is [1].
        full_twin : bool, optional
            will be introduced. Default is True.
        throw_lgi : bool, optional
            Default is True.
        LFAL_kwargs : dict, optional
            Keyword arguments for the slip line 2D class method by_LFAL.
            Default is {'factor': 0.5, 'angle_min': 0, 'angle_max': 360, 'length': 50}.
            - factor : float
                Scale factor for line generation
            - angle_min : float
                Minimum angle in degrees
            - angle_max : float
                Maximum angle in degrees
            - length : int
                Length of the generated line
        twdis_kwargs : dict, optional
            Keyword arguments for twin thickness distribution. Default parameters:
            - max_count_per_grain : int
                Maximum number of twin lamellae per grain
            - min_thickness : float
                Minimum thickness of twin lamellae in micrometers
            - mean_thickness : float
                Mean thickness of twin lamellae in micrometers
            - max_thickness : float
                Maximum thickness of twin lamellae in micrometers
            - distribution : str
                Type of distribution ('normal', etc.)
            - variance : float
                Variance of the distribution
        plotgs_original : bool, optional
            If True, plots the original grain structure before twin introduction.
            Default is True.
        plotgs_twinned : bool, optional
            Default is True.
        save_to_features : bool, optional
            Default is True.
        None or numpy.ndarray
            Returns None by default. If throw_lgi is True, returns the modified
            Local Grain Index (LGI) array as a numpy array with twin regions
            marked as -1.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[35].char_morph_2d(bbox=True, bbox_ex=True, area=True,
                                  aspect_ratio=True, perimeter=True, solidity=True,
                                  make_skim_prop=True,)
        mcgs.gs[35].prop.columns
        mcgs.gs[35].find_neigh()
        mcgs.gs[35].g[12]['grain'].coords
        mcgs.gs[35].g[12]['grain'].centroid
        GIDs, VALIND = mcgs.gs[35].get_gids_in_params_bounds(mpnames=['area'],
                                              fx_stats=[np.mean],
                                              pdslh=[[50, 50]],
                                              plot_mprop=False
                                              )
        gids = GIDs['mpmapped']['area']
        mcgs.gs[35].introduce_single_twins(GIDs=gids, full_twin=True,
                                   throw_lgi=True, plotgs_original=False,
                                   plotgs_twinned=True)

        Notes
        -----
        - Twin regions are identified perpendicular to generated slip lines
        - Twin indices are marked with value -1 in the output LGI array
        - The method modifies self.lgi_1 to track twinned regions across multiple grains
        - Visualization uses the plotgs method with custom colormap and grain ID labels
        """
        # Validations
        # -----------------------------------------
        if plotgs_original:
            self.plotgs(figsize=(6, 6), dpi=120,
                        cmap='coolwarm', plot_centroid=True,
                        centroid_kwargs={'marker': 'o', 'mfc': 'yellow',
                                         'mec': 'black', 'ms': 2.5},
                        plot_gid_number=True)
        # -----------------------------------------
        gscoords = (self.xgr.ravel(), self.ygr.ravel())
        LGI_1 = deepcopy(self.lgi)
        # -----------------------------------------
        for gid in GIDs:
            LGI = deepcopy(self.lgi).ravel()
            xc = self.xgr[self.lgi == gid].mean()
            yc = self.ygr[self.lgi == gid].mean()
            remaining_indices = list(range(gscoords[0].size))
            # -----------------------------------------
            lines = [sl2d.by_LFAL(location=[xc, yc], **LFAL_kwargs)]
            # -----------------------------------------
            twin_indices = []
            for line in lines:
                _fx_ = line.find_neigh_point_by_perp_distance
                _twin_indices_ = _fx_(gscoords, 4.5, use_bounding_rec=True)
                if _twin_indices_:
                    twin_indices.append(_twin_indices_)
                    remaining_indices = list(set(remaining_indices) - set(_twin_indices_))
            # -----------------------------------------
            LGI[twin_indices[0]] = -1
            LGI = np.reshape(LGI, self.lgi.shape)
            LGI_1[(LGI == -1) & (LGI_1 == gid)] = -1
        # -----------------------------------------
        self.plotgs(figsize=(6, 6), dpi=120,
                    cmap='coolwarm', custom_lgi=LGI_1,
                    plot_centroid=True,
                    centroid_kwargs={'marker': 'o', 'mfc': 'yellow',
                                     'mec': 'black', 'ms': 2.5},
                    plot_gid_number=True)

    def add_pxtal(self):
        """
        Add a new polycrystal orientation map instance to the grain structure.

        Returns
        -------
        None
        """
        from upxo.pxtal.pxtal_ori_map_2d import polyxtal2d as PXTAL
        if len(self.pxtal.keys()) == 0:
            self.pxtal[1] = PXTAL()
        else:
            self.pxtal[max(list(self.pxtal.keys()))+1] = PXTAL()

    def set_pxtal(self, instance_no=1,
                  path_filename_noext=None,
                  map_type='ebsd',
                  apply_kuwahara=False, kuwahara_misori=5, gb_misori=10,
                  min_grain_size=1,
                  print_closs=True,
                  ):
        """
        Crystal Orientation Map. EBSD dataswt is one which can be loadsed.

        Parameters
        ----------
        instance_no: int
            Instance number of the polycrystal orientation map to be set up.
            Defaults to 1.
        path_filename_noext: str
            Path and filename without extension of the orientation map file.
            For example, if the file is 'D:/data/map.ctf', then the
            path_filename_noext should be 'D:/data/map'.
        map_type: str
            Type of orientation map file. Valid choices:
            'ebsd': EBSD orientation map file.
        apply_kuwahara: bool
            If True, applies Kuwahara filter to the orientation map. Defaults to
            False.
        kuwahara_misori: float
            Misorientation threshold for Kuwahara filter in degrees. Defaults to
            5 degrees.
        gb_misori: float
            Grain boundary misorientation threshold in degrees. Defaults to
            10 degrees.
        min_grain_size: int
            Minimum grain size in number of pixels. Defaults to 1.
        print_closs: bool
            If True, prints the conversion loss after setting up the
            polycrystal orientation map. Defaults to True.

        Returns
        -------
        None

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 20  # Temporal slice number
        pxt.char_morph_2d(tslice)
        pxt.gs[tslice].export_ctf(r'D:/export_folder', 'sunil')
        path_filename_noext = r'D:/export_folder/sunil'
        pxt.gs[tslice].set_pxtal(path_filename_noext=path_filename_noext)
        pxt.gs[tslice].pxtal.map
        """
        IN, _fn_ = instance_no, path_filename_noext
        _khfflag_, _khfmo_ = apply_kuwahara, kuwahara_misori
        # -------------------------------------------
        self.add_pxtal()
        print(_fn_)
        self.pxtal[IN].setup(map_type='ebsd',
                             path_filename_noext=_fn_,
                             apply_kuwahara=_khfflag_,
                             kuwahara_misori=_khfmo_,)
        self.pxtal[IN].find_grains_gb(gb_misori=gb_misori,
                                      min_grain_size=min_grain_size,
                                      print_msg=True,)
        self.pxtal[IN].port_essentials(print_msg=True)
        # self.pxtal[IN].char_grain_positions_2d()
        self.pxtal[IN].set_conversion_loss(refn=np.unique(self.lgi).size)
        self.find_grain_boundary_junction_points(xorimap=True, IN=IN)
        self.pxtal[IN].set_bjp()
        self.pxtal[IN].find_neigh(update_gid=True, reset_lgi=False)
        self.pxtal[IN].find_gbseg1()

    def make_linear_grid(self, sf=1):
        """
        Make linear grid for interpolation.

        Parameters
        ----------
        sf: float
            Scale factor for grid spacing.
            
        Returns
        -------
        x: np.ndarray
            1D array of x-coordinates.
        y: np.ndarray
            1D array of y-coordinates.
        xinc: float
            Increment in x-coordinates.
        yinc: float
            Increment in y-coordinates.
        """
        # Validate for maximum sf
        # -------------------------------------------------
        # Make the base space
        xinc, yinc = self.uigrid.xinc*sf, self.uigrid.yinc*sf
        xmin, xmax, xinc = self.uigrid.xmin, self.uigrid.xmax, self.uigrid.xinc
        ymin, ymax, yinc = self.uigrid.ymin, self.uigrid.ymax, self.uigrid.yinc
        x = np.arange(xmin, xmax+xinc, xinc)
        y = np.arange(ymin, ymax+yinc, yinc)
        return x, y, xinc, yinc

    def scale(self, sf):
        """
        Apply a scale factor to the current grain structure temporal slice.

        Parameters
        ----------
        sf: float
            Scale factor to apply.

        Returns
        -------
        None
        """
        from scipy.interpolate import RegularGridInterpolator
        # -------------------------------------------------
        # VALIDATE input, f
        # Make the base linear space
        x, y, xinc, yinc = self.make_linear_grid(sf=1)
        # Construct the inerpolator
        intmeth = 'nearest'
        interpolator = RegularGridInterpolator((x, y), self.s, method=intmeth)
        # Make the updated linear space
        _x_, _y_, _xinc_, _yinc_ = self.make_linear_grid(sf=sf)
        # Make the new grid
        _xgr_, _ygr_ = np.meshgrid(_x_, _y_)
        # Interpolate state values
        s = interpolator(np.array([_xgr_.flatten(), _ygr_.flatten()]).T)
        s = s.reshape(len(_x_), len(_y_)).T
        # ---------------------------------------------
        # TODO: VALIDATE IF CREATED S DIMENTSIONS ARE CONSISTENT
        # ---------------------------------------------
        # Create a new grain structure database
        self.scaled['sf'] = sf
        self.scaled['xmin'], self.scaled['xmax'] = _x_.min(), _x_.max()
        self.scaled['ymin'], self.scaled['ymax'] = _y_.min(), _y_.max()
        self.scaled['xinc'], self.scaled['yinc'] = _xinc_, _yinc_
        self.scaled.xgr, self.scaled.ygr, self.s = _xgr_, _ygr_, s
        self.scaled.char_morph_2d()
        if sf != 1:
            self.__resolution_state__ = f'finer_sf={sf}'


    def coarser(self, Grid_Data, ParentStateMatrix, Factor, InterpMethod):
        """
        Create a coarser grid from parent grid and parent state matrix.

        Parameters
        ----------
        Grid_Data: dict
            Dictionary containing grid parameters of the parent grid.
        ParentStateMatrix: np.ndarray
            2D array representing the state matrix of the parent grid.
        Factor: int
            Factor by which to decrease the resolution.
        InterpMethod: str
            Interpolation method to use. Valid choices: 'nearest', 'linear',
            'cubic'.

        Returns
        -------
        cogrid_NG: tuple of np.ndarray
            Tuple containing the new coordinate grid arrays (X, Y).
        OSM_NG: np.ndarray
            2D array representing the new orientation state matrix.
        """

        # Use to decrease resolution
        # Unpack parent grid parameters
        xmin, xmax, xinc = Grid_Data['xmin'], Grid_Data['xmax'], Grid_Data['xinc']
        ymin, ymax, yinc = Grid_Data['ymin'], Grid_Data['ymax'], Grid_Data['yinc']

        # Reconstruct the parent co-ordinate grid
        xvec_OG = np.arange(xmin, xmax+1, float(xinc))  # Parent grid axes
        yvec_OG = np.arange(ymin, ymax+1, float(yinc))  # Parent grid axes
        cogrid_OG = np.meshgrid(xvec_OG, yvec_OG, copy=True, sparse=False, indexing='xy')  # grid

        # Construct the new co-ordinate grid
        xvec_NG = np.arange(xmin, xmax+1, float(xinc*Factor))  # NG: 'of' New grid
        yvec_NG = np.arange(ymin, ymax+1, float(yinc*Factor))
        cogrid_NG = np.meshgrid(xvec_NG, yvec_NG, copy=True, sparse=False, indexing='xy')

        # Construct the new orientation state matrix
        from scipy.interpolate import griddata
        OSM_NG = np.round(griddata((np.concatenate(cogrid_OG[0]),
                                    np.concatenate(cogrid_OG[1])),
                                   np.concatenate(ParentStateMatrix),
                                   (np.concatenate(cogrid_NG[0]),
                                    np.concatenate(cogrid_NG[1])),
                                   method=InterpMethod)
                          .reshape((xvec_NG.shape[0], yvec_NG.shape[0])))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # print(abc.shape)
        return cogrid_NG, OSM_NG

    def __setup__positions__(self):
        """
        Setup position categories for grains in 2D grain structure.

        Returns
        -------
        None
        """
        self.positions = {'top_left': [], 'bottom_left': [],
                          'bottom_right': [], 'top_right': [],
                          'pure_right': [], 'pure_bottom': [],
                          'pure_left': [], 'pure_top': [],
                          'left': [], 'bottom': [], 'right': [], 'top': [],
                          'boundary': [], 'corner': [], 'internal': [], }

    def char_grain_positions_2d(self):
        """
        Characterize and categorize the spatial positions of grains in a 2D grain structure.
        
        This method analyzes each grain's pixel locations to determine whether the grain
        is positioned at the boundary, corner, or interior of the microstructure domain.
        Grains are classified based on which edges of the image domain they touch.
        
        Position Categories
        -------------------
        Corner Positions:
            - 'top_left': Grain touches both top and left boundaries
            - 'top_right': Grain touches both top and right boundaries
            - 'bottom_left': Grain touches both bottom and left boundaries
            - 'bottom_right': Grain touches both bottom and right boundaries
        
        Edge Positions (not at corners):
            - 'pure_top': Grain touches only the top boundary
            - 'pure_bottom': Grain touches only the bottom boundary
            - 'pure_left': Grain touches only the left boundary
            - 'pure_right': Grain touches only the right boundary
        
        Aggregate Positions:
            - 'top': All grains touching top boundary (corner + pure_top)
            - 'bottom': All grains touching bottom boundary (corner + pure_bottom)
            - 'left': All grains touching left boundary (corner + pure_left)
            - 'right': All grains touching right boundary (corner + pure_right)
            - 'boundary': All grains touching any boundary
            - 'corner': All grains at corners only
            - 'internal': Grains not touching any boundary
        
        Algorithm
        ---------
        For each grain:
            1. Extract all pixel locations belonging to the grain
            2. Check if any pixels lie on domain boundaries (row=0, row=max, col=0, col=max)
            3. Classify based on which boundaries are touched
            4. Store classification in grain.position attribute (list format: [x_centroid, y_centroid, position_string])
        
        Attributes Modified
        -------------------
        For each grain in self.g:
            grain.position : list
                [x_centroid, y_centroid, position_category_string]
        
        self.positions : dict
            Dictionary with position categories as keys and lists of grain IDs as values:
            - Keys: 'top_left', 'bottom_left', 'bottom_right', 'top_right',
                    'pure_right', 'pure_bottom', 'pure_left', 'pure_top',
                    'left', 'bottom', 'right', 'top', 'boundary', 'corner', 'internal'
            - Values: List of grain IDs (gids) belonging to each category
        
        Returns
        -------
        None
            Results are stored in grain.position attributes and self.positions dictionary
        
        Notes
        -----
        - A grain can belong to multiple aggregate categories simultaneously
        - Internal grains are those with no pixels on any boundary
        - Position determination is based on pixel-level analysis, not centroids
        - Domain boundaries are defined by image array indices (0, row_max, col_max)
        
        Examples
        --------
        >>> from upxo.ggrowth.mcgs import mcgs
        >>> pxt = mcgs(study='independent', 
        ...            input_dashboard='input_dashboard.xls')
        >>> pxt.simulate()
        >>> pxt.detect_grains()
        >>> pxt.gs[10].char_grain_positions_2d()
        >>> 
        >>> # Access corner grains
        >>> corner_grains = pxt.gs[10].positions['corner']
        >>> 
        >>> # Access internal grains
        >>> internal_grains = pxt.gs[10].positions['internal']
        >>> 
        >>> # Get position of a specific grain
        >>> grain_5_position = pxt.gs[10].g[5]['grain'].position
        >>> print(f"Grain 5 centroid: ({grain_5_position[0]:.2f}, {grain_5_position[1]:.2f})")
        >>> print(f"Grain 5 category: {grain_5_position[2]}")
        
        See Also
        --------
        plot_grains_at_position : Visualize grains at specific positions
        find_border_internal_grains_fast : Fast alternative for border/internal classification
        """
        row_max, col_max = self.lgi.shape[0]-1, self.lgi.shape[1]-1
        for grain in self:
            # Calculate normalized centroids serving as numerical position
            # values
            grain.position = list(grain.centroid)
            # Determine the location strings for all grains
            all_pixel_locations = grain.loc.tolist()
            apl = np.array(all_pixel_locations).T  # all_pixel_locations
            if 0 in apl[0]:  # TOP
                '''
                grain touches either:
                    top and/or left boundary, OR, top and/or right boundary
                '''
                if 0 in apl[1]:  # TOP AND LEFT
                    '''
                    BRANCH.1.A. Grain touches top and left boundary: top_left
                    grain. This means the grain is TOP_LEFT CORNER GRAIN
                    '''
                    grain.position.append('top_left')
                elif col_max in apl[1]:  # TOP AND RIGHT
                    '''
                    BRANCH.1.B. Grain touches top and right boundary: top_right
                    grain This means the grain is a TOP_RIGHT CORNER GRAIN
                    '''
                    grain.position.append('top_right')
                else:  # TOP, NOT LEFT, NOT RIGHT: //PURE TOP//
                    '''
                    BRANCH.1.C. Grain touches top boundary only and not the
                    corners of the top boundary. This means the grain is a
                    TOP GRAIN
                    '''
                    grain.position.append('pure_top')
            if row_max in apl[0]:  # BOTTOM
                '''
                grain touches either:
                    * bottom and/or left boundary, OR,
                    * bottom and/or right boundary
                '''
                if 0 in apl[1]:  # BOTTOM AND LEFT
                    '''
                    BRANCH.2.A. Grain touches bottom and left boundary:
                    bot_left grain. This means the grain is BOTTOM_LEFT CORNER
                    GRAIN
                    '''
                    grain.position.append('bottom_left')
                elif col_max in apl[1]:  # BOTTOM AND RIGHT
                    '''
                    BRANCH.2.B. Grain touches bottom and right boundary:
                    bot_right grain. This means the grain is BOTTOM_RIGHT
                    CORNER GRAIN
                    '''
                    grain.position.append('bottom_right')
                else:  # BOTTOM, NOT LEFT, NOT RIGHT: //PURE BOTTOM//
                    '''
                    BRANCH.2.C. Grain touches only bottom boundary and not the
                    corners of the bottom boundary. This means the grain is a
                    BOTTOM GRAIN
                    '''
                    grain.position.append('pure_bottom')
            if 0 in apl[1]:  # LEFT
                '''
                grain touches either:
                    * left and/or top boundary, OR,
                    * left and/or bottom boundary
                '''
                if 0 in apl[0]:  # LEFT AND TOP
                    '''
                    BRANCH.3.A. Grain touches left and top boundary: top_left
                    grain. This means the grain is LEFT_TOP CORNER GRAIN
                    '''
                    # THIS BRANCH HAS ALREADY BEEN VISITED IN BRANCH.1.A
                    # NOTHING MORE TO DO HERE. SKIP.
                    pass
                elif row_max in apl[0]:  # LEFT AND BOTTOM
                    '''
                    BRANCH.3.B. Grain touches left and bottom boundary:
                    bot_left grain. This means the grain is a LEFT_BOTTOM
                    CORNER GRAIN
                    '''
                    # THIS BRANCH HAS ALREADY BEEN VISISITED IN BRANCH.2.A
                    # NOTHING MORE TO DO HERE. SKIP.
                    pass
                else:  # LEFT, NOT TOP, NOT BOTTOM: //PURE LEFT//
                    '''
                    BRANCH.3.C. Grain touches left boundary only and not the
                    corners of the left boundary. This means the grain is a #
                    LEFT GRAIN
                    '''
                    grain.position.append('pure_left')
            if col_max in apl[1]:  # RIGHT
                '''
                grain touches either:
                    * right and/or top boundary, OR,
                    * right and/or bottom boundary
                '''
                if 0 in apl[0]:  # RIGHT AND TOP
                    '''
                    BRANCH.4.A. Grain touches right and top boundary: top_right
                    grain. This means the grain is RIGHT_TOP CORNER GRAIN
                    '''
                    # THIS BRANCH HAS ALREADY BEEN VISITED IN BRANCH.1.B
                    # NOTHING MORE TO DO HERE. SKIP.
                    pass
                elif row_max in apl[0]:  # RIGHT AND BOTTOM
                    '''
                    BRANCH.4.B. Grain touches left and bottom boundary:
                    bot_left grain. This means the grain is a RIGHT_BOTTOM
                    CORNER GRAIN
                    '''
                    # THIS BRANCH HAS ALREADY VBEEN VISISITED IN BRANCH.2.B
                    # NOTHING MORE TO DO HERE. SKIP.
                    pass
                else:  # RIGHT, NOT TOP, NOT BOTTOM: //PURE RIGHT//
                    '''
                    BRANCH.4.C. Grain touches left boundary only and not the
                    corners of the left boundary. This means the grain is a
                    RIGHT GRAIN
                    '''
                    grain.position.append('pure_right')
            if 0 not in apl[0] and row_max not in apl[0]:
                # NOT TOP, NOT BOTTOM
                if 0 not in apl[1] and col_max not in apl[1]:
                    # NOT LEFT, NOT RIGHT
                    grain.position.append('internal')

        for grain in self:
            position = grain.position[2]
            gid = grain.gid
            _ = [position == 'top_left', position == 'bottom_left',
                 position == 'bottom_right', position == 'top_right',
                 position == 'pure_right', position == 'pure_bottom',
                 position == 'pure_left', position == 'pure_top',
                 position == 'left', position == 'bottom',
                 position == 'right', position == 'top',
                 position == 'boundary', position == 'corner',
                 position == 'internal']
            self.positions[[_*position for _ in _ if _*position][0]].append(gid)

        for pos in ['top_left', 'bottom_left', 'pure_left']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['left'].append(value)
        for pos in ['bottom_left', 'pure_bottom', 'bottom_right']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['bottom'].append(value)
        for pos in ['bottom_right', 'pure_right', 'top_right']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['right'].append(value)
        for pos in ['top_right', 'pure_top', 'top_left']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['top'].append(value)
        for pos in ['top_left', 'bottom_left', 'bottom_right', 'top_right']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['corner'].append(value)
        for pos in ['top_left', 'bottom_left', 'bottom_right', 'top_right',
                    'pure_left', 'pure_bottom', 'pure_right', 'pure_top'
                    ]:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['boundary'].append(value)

    def char_grain_positions_2d_v1(self):
        """
        Characterize and categorize the spatial positions of grains in a 2D grain structure.
        
        This method analyzes each grain's pixel locations to determine whether the grain
        is positioned at the boundary, corner, or interior of the microstructure domain.
        Grains are classified based on which edges of the image domain they touch.
        
        Position Categories
        -------------------
        Corner Positions:
            - 'top_left': Grain touches both top and left boundaries
            - 'top_right': Grain touches both top and right boundaries
            - 'bottom_left': Grain touches both bottom and left boundaries
            - 'bottom_right': Grain touches both bottom and right boundaries
        
        Edge Positions (not at corners):
            - 'pure_top': Grain touches only the top boundary
            - 'pure_bottom': Grain touches only the bottom boundary
            - 'pure_left': Grain touches only the left boundary
            - 'pure_right': Grain touches only the right boundary
        
        Aggregate Positions:
            - 'top': All grains touching top boundary (corner + pure_top)
            - 'bottom': All grains touching bottom boundary (corner + pure_bottom)
            - 'left': All grains touching left boundary (corner + pure_left)
            - 'right': All grains touching right boundary (corner + pure_right)
            - 'boundary': All grains touching any boundary
            - 'corner': All grains at corners only
            - 'internal': Grains not touching any boundary
        
        Algorithm
        ---------
        For each grain:
            1. Extract all pixel locations belonging to the grain
            2. Check if any pixels lie on domain boundaries (row=0, row=max, col=0, col=max)
            3. Classify based on which boundaries are touched
            4. Store classification in grain.position attribute (list format: [x_centroid, y_centroid, position_string])
        
        Attributes Modified
        -------------------
        For each grain in self.g:
            grain.position : list
                [x_centroid, y_centroid, position_category_string]
        
        self.positions : dict
            Dictionary with position categories as keys and lists of grain IDs as values:
            - Keys: 'top_left', 'bottom_left', 'bottom_right', 'top_right',
                    'pure_right', 'pure_bottom', 'pure_left', 'pure_top',
                    'left', 'bottom', 'right', 'top', 'boundary', 'corner', 'internal'
            - Values: List of grain IDs (gids) belonging to each category
        
        Returns
        -------
        None
            Results are stored in grain.position attributes and self.positions dictionary
        
        Notes
        -----
        - A grain can belong to multiple aggregate categories simultaneously
        - Internal grains are those with no pixels on any boundary
        - Position determination is based on pixel-level analysis, not centroids
        - Domain boundaries are defined by image array indices (0, row_max, col_max)
        
        Examples
        --------
        >>> from upxo.ggrowth.mcgs import mcgs
        >>> pxt = mcgs(study='independent', 
        ...            input_dashboard='input_dashboard.xls')
        >>> pxt.simulate()
        >>> pxt.detect_grains()
        >>> pxt.gs[10].char_grain_positions_2d()
        >>> 
        >>> # Access corner grains
        >>> corner_grains = pxt.gs[10].positions['corner']
        >>> 
        >>> # Access internal grains
        >>> internal_grains = pxt.gs[10].positions['internal']
        >>> 
        >>> # Get position of a specific grain
        >>> grain_5_position = pxt.gs[10].g[5]['grain'].position
        >>> print(f"Grain 5 category: {grain_5_position}[2]}")
        >>> print(f"Grain 5 category: {grain_5_position[2]}")
        
        See Also
        --------
        plot_grains_at_position : Visualize grains at specific positions
        find_border_internal_grains_fast : Fast alternative for border/internal classification
        """
        top_left = np.array([self.lgi[-1, 0]], dtype=int)
        top_right = np.array([self.lgi[-1, -1]], dtype=int)
        bottom_left = np.array([self.lgi[0, 0]], dtype=int)
        bottom_right = np.array([self.lgi[0, -1]], dtype=int)
        left = np.unique(self.lgi[:, 0])
        right = np.unique(self.lgi[:, -1])
        top = np.unique(self.lgi[-1, :])
        bottom = np.unique(self.lgi[0, :])
        pure_top = np.setdiff1d(top, np.union1d(top_left, top_right))
        pure_bottom = np.setdiff1d(bottom, np.union1d(bottom_left, bottom_right))
        pure_left = np.setdiff1d(left, np.union1d(top_left, bottom_left))
        pure_right = np.setdiff1d(right, np.union1d(top_right, bottom_right))
        internal = np.setdiff1d(self.gid, np.union1d(np.union1d(top, bottom), np.union1d(left, right)))
        self.positions['left'] = left
        self.positions['right'] = right
        self.positions['top'] = top
        self.positions['bottom'] = bottom
        self.positions['top_left'] = top_left
        self.positions['top_right'] = top_right
        self.positions['bottom_left'] = bottom_left
        self.positions['bottom_right'] = bottom_right
        self.positions['pure_top'] = pure_top
        self.positions['pure_bottom'] = pure_bottom
        self.positions['pure_left'] = pure_left
        self.positions['pure_right'] = pure_right
        self.positions['internal'] = internal
        self.positions['boundary'] = np.union1d(np.union1d(top, bottom), np.union1d(left, right))
        self.positions['corner'] = np.union1d(np.union1d(top_left, top_right), np.union1d(bottom_left, bottom_right))
        for grain in self:
            gid = grain.gid
            if gid in top_left:
                grain.position = 'top_left'
            elif gid in top_right:
                grain.position = 'top_right'
            elif gid in bottom_left:
                grain.position = 'bottom_left'
            elif gid in bottom_right:
                grain.position = 'bottom_right'
            elif gid in pure_top:
                grain.position = 'pure_top'
            elif gid in pure_bottom:
                grain.position = 'pure_bottom'
            elif gid in pure_left:
                grain.position = 'pure_left'
            elif gid in pure_right:
                grain.position = 'pure_right'
            elif gid in internal:
                grain.position = 'internal'

    def find_border_internal_grains_fast(self):
        """
        Identify border and internal grains.
        Quickly find border and internal grains without doing anything else.
        Returns
        -------
        border_gids: Numpy array of border grain IDs.
        internal_gids: Numpy array of internal grain IDs.
        lgi_border: Numpy array of Local Grain Index (LGI) with only border
            grains.
        lgi_internal: Numpy array of Local Grain Index (LGI) with only internal
            grains.

        Example
        -------
        border_gids, internal_gids, lgi_border, lgi_internal = find_border_internal_grains_fast()

        plt.figure()
        plt.imshow(lgi_border)

        plt.figure()
        plt.imshow(lgi_internal)

        """
        lgi = self.lgi
        lgi_border = deepcopy(lgi)
        lgi_border[1:-1, 1:-1] = 0
        border_gids = np.unique(lgi_border[lgi_border != 0])
        internal_gids = np.array(list(set(self.gid) - set(border_gids)))

        lgi_border = deepcopy(lgi)

        lgi_internal = deepcopy(lgi)

        for bgid in border_gids:
            lgi_internal[lgi_internal == bgid] = 0

        for nbgid in internal_gids:
            lgi_border[lgi_border == nbgid] = 0

        return border_gids, internal_gids, lgi_border, lgi_internal

    def find_grain_size_fast(self, metric='npixels', recalculate_gid=False):
        """
        Quickly find the grain sizes without doing anything else.

        Explanations
        ------------
        Order of grain_sizes is that of pxtal.gs[m].gid

        Parameters
        ----------
        metric: Specify which ares metric is needed. Optoins include:
            * 'npixels': Number of pixels
            * 'pxarea': Pixel wise calculated area
            * 'eq_dia': Equivalent diameter

        Return
        ------
        grain_sizes: Numpy array of grain areas.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        grain_areas_all_grains = pxtal.gs[2].find_grain_size_fast(metric='npixels')
        """
        if recalculate_gid:
            self.gid = np.unique(self.lgi)
            self.n = len(self.gid)
        if len(self.gid) == 0:
            return np.array([])
        max_gid = np.max(self.gid)
        counts = np.bincount(self.lgi.ravel(), minlength=max_gid + 1)
        pixel_counts = counts[self.gid]
        if metric == 'npixels':
            grain_size = pixel_counts
        elif metric == 'pxarea':
            grain_size = pixel_counts*(self.uigrid.px_size**2)
        elif metric == 'eq_dia':
            area = pixel_counts*(self.uigrid.px_size**2)
            grain_size = 2*np.sqrt(area/np.pi)

        return np.array(grain_size)

    def find_npixels_border_grains_fast(self, metric='npixels'):
        """
        Find the number of pixels in each of the border grains.

        Parameters
        ----------
        metric: Specify which ares metric is needed. Optoins include:
            * 'npixels': Number of pixels
            * 'pxarea': Pixel wise calculated area
            * 'eq_dia': Equivalent diameter

        Return
        ------
        border_grain_npixels: Numpy array of number of pixels in each border
            grain.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        grain_areas_border_grains = pxtal.gs[2].find_npixels_border_grains_fast(metric='npixels')
        """
        border_gids, _, __, ___ = self.find_border_internal_grains_fast()
        counts = np.bincount(self.lgi.ravel(), minlength=self.gid.max() + 1)
        pixel_counts = counts[self.gid]
        if metric == 'npixels':
            grain_size = pixel_counts
        elif metric == 'pxarea':
            grain_size = pixel_counts*(self.uigrid.px_size**2)
        elif metric == 'eq_dia':
            area = pixel_counts*(self.uigrid.px_size**2)
            grain_size = 2*np.sqrt(area/np.pi)

        border_grain_size = []
        for bg in border_gids:
            border_grain_size.append(grain_size[bg-1])

        return np.array(border_grain_size)

    def find_npixels_internal_grains_fast(self, metric='npixels'):
        """
        Find the number of pixels in each of the internal grains.

        Parameters
        ----------
        metric: Specify which ares metric is needed. Optoins include:
            * 'npixels': Number of pixels
            * 'pxarea': Pixel wise calculated area
            * 'eq_dia': Equivalent diameter

        Reyturn
        -------
        internal_grain_npixels: Numpy array of number of pixels in each
            internal grain.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        grain_areas_internal_grains = pxtal.gs[2].find_npixels_internal_grains_fast(metric='npixels')
        """
        _, internal_grains, __, ___ = self.find_border_internal_grains_fast()
        counts = np.bincount(self.lgi.ravel(), minlength=self.gid.max() + 1)
        pixel_counts = counts[self.gid]
        if metric == 'npixels':
            grain_size = pixel_counts
        elif metric == 'pxarea':
            grain_size = pixel_counts*(self.uigrid.px_size**2)
        elif metric == 'eq_dia':
            area = pixel_counts*(self.uigrid.px_size**2)
            grain_size = 2*np.sqrt(area/np.pi)

        border_grain_size = []
        for ig in internal_grains:
            border_grain_size.append(grain_size[ig-1])

        return np.array(border_grain_size)
    

        internal_grain_npixels = []

        if metric in ('npixels'):
            for ig in internal_grains:
                internal_grain_npixels.append(np.where(self.lgi == ig)[0].size)

        return np.array(internal_grain_npixels)

    def find_ags(self, grains_to_include='all', gids=None, method='npixels'):
        """
        Find average grain size of the grain structure.

        Parameters
        ----------
        grains_to_include: str
            Specify which grains to include in the average grain size
            calculation. Options include:
            * 'all': Include all grains.
            * 'border': Include only border grains.
            * 'internal': Include only internal grains.
            * 'gid' or 'gids': Include only grains with specified GIDs.
        gids: list or np.ndarray, optional
            List or array of grain IDs to include if grains_to_include is
            'gid' or 'gids'. Defaults to None.
        method: str
            Specify which area metric to use for average grain size calculation.
            Options include:
            * 'npixels': Number of pixels.
            * 'pxarea': Pixel wise calculated area.
            * 'eq_dia': Equivalent diameter.

        Returns
        -------
        ags: float
            Average grain size based on the specified criteria.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        ags_all = pxtal.gs[2].find_ags(grains_to_include='all', method='npixels')
        ags_border = pxtal.gs[2].find_ags(grains_to_include='border', method='npixels')
        ags_internal = pxtal.gs[2].find_ags(grains_to_include='internal', method='npixels')
        ags_specific = pxtal.gs[2].find_ags(grains_to_include='gids', gids=[1,2,3], method='npixels')
        """
        if grains_to_include == 'all':
            pass
        elif grains_to_include == 'border':
            pass
        elif grains_to_include == 'internal':
            pass
        elif grains_to_include in ('gid', 'gids'):
            pass
        return ags

    def find_prop_npixels(self):
        """
        Get grain NUMBER OF PIXELS into pandas dataframe
        Returns
        -------
        None
        """
        npixels = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                npixels.append(len(g['grain'].loc))
            elif self._char_fx_version_ == 2:
                npixels.append(len(g.loc))
        self.prop['npixels'] = npixels
        if self.display_messages:
            print('    Number of Pixels making the grains: DONE')

    def find_prop_npixels_gb(self):
        """
        Get grain GRAIN BOUNDARY LENGTH (NO. PIXELS) into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['npixels_gb']:
        npixels_gb = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                npixels_gb.append(len(g['grain'].gbloc))
            elif self._char_fx_version_ == 2:
                npixels_gb.append(len(g.gbloc))
        self.prop['npixels_gb'] = npixels_gb

    def find_prop_gb_length_px(self):
        """
        Get grain GRAIN BOUNDARY LENGTH (NO. PIXELS) into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['gb_length_px']:
        gb_length_px = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                gb_length_px.append(len(g['grain'].gbloc))
            elif self._char_fx_version_ == 2:
                gb_length_px.append(len(g.gbloc))
        self.prop['gb_length_px'] = gb_length_px

    def find_prop_area(self):
        """
        Get grain AREA into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['area']:
        area = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                area.append(g['grain'].skprop.area)
            elif self._char_fx_version_ == 2:
                area.append(g.skprop.area)
        self.prop['area'] = area

    def find_prop_eq_diameter(self):
        """
        Get grain EQUIVALENT DIAMETER into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['eq_diameter']:
        eq_diameter = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                eq_diameter.append(g['grain'].skprop.equivalent_diameter_area)
            elif self._char_fx_version_ == 2:
                eq_diameter.append(g.skprop.equivalent_diameter_area)
        self.prop['eq_diameter'] = eq_diameter

    def find_prop_perimeter(self):
        """
        Get grain PERIMETER into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['perimeter']:
        perimeter = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                perimeter.append(g['grain'].skprop.perimeter)
            elif self._char_fx_version_ == 2:
                perimeter.append(g.skprop.perimeter)
        self.prop['perimeter'] = perimeter

    def find_prop_perimeter_crofton(self):
        """
        Get grain CROFTON PERIMETER into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['perimeter_crofton']:
        perimeter_crofton = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                perimeter_crofton.append(g['grain'].skprop.perimeter_crofton)
            elif self._char_fx_version_ == 2:
                perimeter_crofton.append(g.skprop.perimeter_crofton)
        self.prop['perimeter_crofton'] = perimeter_crofton

    def find_prop_compactness(self):
        """
        Get grain COMPACTNESS into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['compactness']:
        compactness = []
        if self.prop_flag['area']:
            if self.prop_flag['perimeter']:
                for i, g in enumerate(self.g.values()):
                    area = self.prop['area'][i]
                    # Calculate area of circle with the same perimeter
                    # P = pi*D --> D = P/pi
                    # A = pi*D**2/4 = pi*(P/pi)**2/4 = P/(4*pi)
                    circle_area = self.prop['perimeter'][i]**2/(4*np.pi)
                    if circle_area >= self.EPS:
                        compactness.append(area/circle_area)
                    else:
                        compactness.append(1)
            else:
                for i, g in self.g.values():
                    area = self.prop['area'][i]
                    if self._char_fx_version_ == 1:
                        circle_area = g['grain'].skprop.perimeter**2/(4*np.pi)
                    elif self._char_fx_version_ == 2:
                        circle_area = g.skprop.perimeter**2/(4*np.pi)
                    if circle_area >= self.EPS:
                        compactness.append(area/circle_area)
                    else:
                        compactness.append(1)
        else:
            if self.prop_flag['perimeter']:
                for i, g in self.g.values():
                    if self._char_fx_version_ == 1:
                        area = g['grain'].skprop.area
                    elif self._char_fx_version_ == 2:
                        area = g.skprop.area
                    circle_area = self.prop['perimeter'][i]**2/(4*np.pi)
                    if circle_area >= self.EPS:
                        compactness.append(area/circle_area)
                    else:
                        compactness.append(1)
            else:
                for i, g in self.g.values():
                    if self._char_fx_version_ == 1:
                        area = g['grain'].skprop.area
                        circle_area = g['grain'].skprop.perimeter**2/(4*np.pi)
                    elif self._char_fx_version_ == 2:
                        area = g.skprop.area
                        circle_area = g.skprop.perimeter**2/(4*np.pi)
                    if circle_area >= self.EPS:
                        compactness.append(area/circle_area)
                    else:
                        compactness.append(1)

        self.prop['compactness'] = compactness

    def find_prop_aspect_ratio(self):
        """
        Get grain ASPECT RATIO into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['aspect_ratio']:
        aspect_ratio = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                maj_axis = g['grain'].skprop.major_axis_length
                min_axis = g['grain'].skprop.minor_axis_length
            elif self._char_fx_version_ == 2:
                maj_axis = g.skprop.major_axis_length
                min_axis = g.skprop.minor_axis_length
            if min_axis <= self.EPS:
                aspect_ratio.append(np.inf)
            else:
                aspect_ratio.append(maj_axis/min_axis)
        self.prop['aspect_ratio'] = aspect_ratio

    def find_prop_solidity(self):
        """
        Get grain SOLIDITY into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['solidity']:
        solidity = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                solidity.append(g['grain'].skprop.solidity)
            elif self._char_fx_version_ == 2:
                solidity.append(g.skprop.solidity)
        self.prop['solidity'] = solidity

    def find_prop_circularity(self):
        """
        Get grain CIRCULARITY into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['circularity']:
        circularity = []

    def find_prop_major_axis_length(self):
        """
        Get grain MAJOR AXIS LENGTH into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['major_axis_length']:
        major_axis_length = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                major_axis_length.append(g['grain'].skprop.major_axis_length)
            elif self._char_fx_version_ == 2:
                major_axis_length.append(g.skprop.major_axis_length)
        self.prop['major_axis_length'] = major_axis_length

    def find_prop_minor_axis_length(self):
        """
        Get grain MINOR AXIS LENGTH into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['minor_axis_length']:
        minor_axis_length = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                minor_axis_length.append(g['grain'].skprop.minor_axis_length)
            elif self._char_fx_version_ == 2:
                minor_axis_length.append(g.skprop.minor_axis_length)
        self.prop['minor_axis_length'] = minor_axis_length

    def find_prop_morph_ori(self):
        """
        Get grain MORPHOLOGICAL ORIENTATION into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['morph_ori']:
        morph_ori = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                morph_ori.append(g['grain'].skprop.orientation)
            elif self._char_fx_version_ == 2:
                morph_ori.append(g.skprop.orientation)
        self.prop['morph_ori'] = [mo*180/np.pi for mo in morph_ori]

    def find_prop_feret_diameter(self):
        """
        Get grain FERET DIAMETER into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['feret_diameter']:
        feret_diameter = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                feret_diameter.append(g['grain'].skprop.feret_diameter_max)
            elif self._char_fx_version_ == 2:
                feret_diameter.append(g.skprop.feret_diameter_max)
        self.prop['feret_diameter'] = feret_diameter

    def find_prop_euler_number(self):
        """
        Get grain EULER NUMBER into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['euler_number']:
        euler_number = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                euler_number.append(g['grain'].skprop.euler_number)
            elif self._char_fx_version_ == 2:
                euler_number.append(g.skprop.euler_number)
        self.prop['euler_number'] = euler_number

    def find_prop_eccentricity(self):
        """
        Get grain ECCENTRICITY into pandas dataframe
        Returns
        -------
        None
        """
        # if self.prop_flag['eccentricity']:
        eccentricity = []
        for g in self.g.values():
            if self._char_fx_version_ == 1:
                eccentricity.append(g['grain'].skprop.eccentricity)
            elif self._char_fx_version_ == 2:
                eccentricity.append(g.skprop.eccentricity)
        self.prop['eccentricity'] = eccentricity

    def build_prop(self, correct_aspect_ratio=False):
        """
        Build the grain structure properties as requested by user in the
        'prop_flag' attribute.

        Returns
        -------
        None
        """
        if self.prop_flag['npixels']:
            self.find_prop_npixels()
        if self.prop_flag['npixels_gb']:
            self.find_prop_npixels_gb()
        if self.prop_flag['gb_length_px']:
            self.find_prop_gb_length_px()
        if self.prop_flag['area']:
            self.find_prop_area()
        if self.prop_flag['eq_diameter']:
            self.find_prop_eq_diameter()
        if self.prop_flag['perimeter']:
            self.find_prop_perimeter()
        if self.prop_flag['perimeter_crofton']:
            self.find_prop_perimeter_crofton()
        if self.prop_flag['compactness']:
            self.find_prop_compactness()
        if self.prop_flag['aspect_ratio']:
            self.find_prop_aspect_ratio()
        if self.prop_flag['solidity']:
            self.find_prop_solidity()
        if self.prop_flag['circularity']:
            self.find_prop_circularity()
        if self.prop_flag['major_axis_length']:
            self.find_prop_major_axis_length()
        if self.prop_flag['minor_axis_length']:
            self.find_prop_minor_axis_length()
        if self.prop_flag['morph_ori']:
            self.find_prop_morph_ori()
        if self.prop_flag['feret_diameter']:
            self.find_prop_feret_diameter()
        if self.prop_flag['euler_number']:
            self.find_prop_euler_number()
        if self.prop_flag['eccentricity']:
            self.find_prop_eccentricity()
        # ------------------------------------------
        if correct_aspect_ratio:
            if any((not 'area' in self.prop.columns,
                   not 'aspect_ratio' in self.prop.columns,
                   not 'major_axis_length' in self.prop.columns,
                   not 'minor_axis_length' in self.prop.columns)):
                print('Need area, aspect_ratio, major_axis_length, minor_axis_length', 
                      'to correct aspect ratio. Skipping aspect ratio correction.')
            else:                    
                df = deepcopy(self.prop)
                df.loc[df['area'] == 1, ['major_axis_length', 'minor_axis_length', 'aspect_ratio']] = 1
                df.loc[df['minor_axis_length'] == 0, 'minor_axis_length'] = 1
                df.loc[df['minor_axis_length'] == 1, 'aspect_ratio'] = df['major_axis_length']
                self.prop = df

    def get_stat(self, PROP_NAME, saa=True, throw=False, ):
        """
        Calculates ths statistics of a property in the 'prop' attribute.

        NOTE
        ----
        Input data is not sanitised before calculating the statistics.
        Will results in an error if invalid entries are found.

        Parameters
        ----------
        PROP_NAME : str
            Name of the property, whos statistics is to be calculated. They
            could be from the following list:
                1. npixels
                2. npixels_gb
                3. area
                4. eq_diameter
                5. perimeter
                6. perimeter_crofton
                7. compactness
                8. gb_length_px
                9. aspect_ratio
                10. solidity
                11. morph_ori
                12. circularity
                13. eccentricity
                14. feret_diameter
                15. major_axis_length
                16. minor_axis_length
                17. euler_number
        saa : bool, optional
            Flag to save the statistics as attribute.
            The default is True.
        throw : bool, optional
            Flag to return the computed statistics.
            The default is False.

        Returns
        -------
        metrics : TYPE
            DESCRIPTION.

        Metrics calculated
        ---------------------
        Following stastical metrics will be calculated:
            count: Data count value
            mean: Mean of the data
            std: Standard deviation of the data
            min: Minimum value of the data
            25%: First quartile of the data
            50%: Second quartile of the data
            75%: Third quartile of the data
            max: Maximum value of the data
            median: Median value of the data
            mode: List of modes of the data
            var: Variance of the data
            skew: Skewness of the data
            kurt: Kurtosis of the data
            nunique: Number of unique values in the data
            sem: Standard error of the mean of the data

        Example call
        ------------
            PXGS.gs[4].extract_statistics_prop('area')
        """
        # Extract the values of the PROP_NAME
        values = np.array(self.prop[PROP_NAME])
        # Extract non-inf subset
        values = values[np.where(values != np.inf)[0]]
        # Make the values dataframe
        import pandas as pd
        values_df = pd.DataFrame(columns=['temp'])
        values_df['temp'] = values
        # Extract basic statistics
        values_stats = values_df.describe()
        metrics = {'PROP_NAME': PROP_NAME,
                   'count': values_stats['temp']['count'],
                   'mean': values_stats['temp']['mean'],
                   'std': values_stats['temp']['std'],
                   'min': values_stats['temp']['min'],
                   '25%': values_stats['temp']['25%'],
                   '50%': values_stats['temp']['50%'],
                   '75%': values_stats['temp']['75%'],
                   'max': values_stats['temp']['max'],
                   'median': values_df['temp'].median(),
                   'mode': [i for i in values_df['temp'].mode()],
                   'var': values_df['temp'].var(),
                   'skew': values_df['temp'].skew(),
                   'kurt': values_df['temp'].kurt(),
                   'nunique': values_df['temp'].nunique(),
                   'sem': values_df['temp'].sem(),
                   }
        if saa:
            self.prop_stat = metrics
        if throw:
            return metrics

    def make_valid_prop(self, PROP_NAME='aspect_ratio',
                        rem_nan=True, rem_inf=True, PROP_df_column=None, ):
        """
        Remove invalid entries from a column in a Pandas dataframe and
        returns sanitized pandas column with the PROP_NAME as column name

        Parameters
        ----------
        PROP_NAME : str, optional
            Property to be cleansed. The default is 'aspect_ratio'.
        rem_nan : TYPE, optional
            Boolean flag to remove np.nan. The default is True.
        rem_inf : TYPE, optional
            Boolean flag to remove np.inf. Both negative and positive inf
            will be removed. The default is True.

        Returns
        -------
        subset : pd.data_frame
            A single column pandas dataframe with cleansed values.#
        ratio : float
            Ratio of total number of values removed to the size of the property
            column in the self.prop dataframe

        """
        if not PROP_df_column:
            # TYhis means internal data in prop atrtribute is to be cleaned
            if hasattr(self, 'prop'):
                if PROP_NAME in self.prop.columns:
                    _prop_size_ = self.prop[PROP_NAME].size
                    subset = self.prop[PROP_NAME]
                    subset = subset.replace([-np.inf,
                                             np.inf],
                                            np.nan).dropna()
                    ratio = (_prop_size_-subset.size)/_prop_size_
                else:
                    subset, ratio = None, None
                    print(f"Property {PROP_NAME} has not been calculated in"
                          " temporal slice {self.m}")
            else:
                subset, ratio = None, None
                print(f"Temporal slice {self.m} has no prop. Skipped")
        else:
            # This means the user provided single-colulmn pandas dataframe,
            # named "PROP_df_column" is to be cleaned
            # It will be assumed user has input valid dataframe column
            _prop_size_ = PROP_df_column.size
            PROP_df_column = PROP_df_column.replace([-np.inf,
                                                     np.inf],
                                                    np.nan).dropna()
            ratio = (_prop_size_-PROP_df_column.size)/_prop_size_

        return subset, ratio

    def s_prop(self,
               s=1,
               PROP_NAME='area'
               ):
        """
        Extract state wise partitioned property. Property name has to be
        specified by the user.

        Parameters
        ----------
        s : int, optional
            Value of the
            The default is 1.
        PROP_NAME : TYPE, optional
            DESCRIPTION. The default is 'area'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        # TODO
            1: add validity checking layers for s and PROP_NAME
            2: if s = 0, then any of the available be selected at random and
                returned
            3: if s = -1, then the state with the minimum number of grains
                will be returned
            4: if s = -2, then the state with the maximum number of grains
                will be returned
        """
        if hasattr(self, 'prop'):
            if PROP_NAME in self.prop.columns:
                if s in self.s_gid.keys():
                    # __ = self.make_valid_prop(rem_nan=True,
                    #                           rem_inf=True,
                    #                           PROP_df_column=self.prop[PROP_NAME], )
                    # PROP_VALUES_VALID = __
                    subset = self.prop[PROP_NAME].iloc[[i-1 for i in self.s_gid[s]]]
                else:
                    subset = None
                    print(f"Temporal slice {self.m} has no grains in s:"
                          " {s}. Skipped")
            else:
                subset, ratio = None, None
                print(f"Property {PROP_NAME} has not been calculated in "
                      "temporal slice {self.m}")
        else:
            print(f"Temporal slice {self.m} has no prop. Skipped")
        return subset

    def get_gid_prop_range(self,
                           PROP_NAME='area',
                           reminf=True,
                           remnan=True,
                           range_type='percentage',
                           value_range=[1, 2],
                           percentage_range=[0, 20],
                           rank_range=[60, 90],
                           pivot=None):
        '''
        Get GIDs of grains whose property values fall within a specified
        range.

        Parameters
        ----------
        PROP_NAME : str, optional
            Name of the property to filter grains by. The default is 'area'.
        reminf : bool, optional
            Flag to remove infinite values from the property data. The default is
            True.
        remnan : bool, optional
            Flag to remove NaN values from the property data. The default is
            True.
        range_type : str, optional
            Type of range to use for filtering. Options are 'percentage',
            'value', or 'rank'. The default is 'percentage'.
        value_range : list, optional
            List of two values specifying the min and max property values for
            filtering when range_type is 'value'. The default is [1, 2].
        percentage_range : list, optional
            List of two values specifying the min and max percentage for
            filtering when range_type is 'percentage'. The default is [0, 20].
        rank_range : list, optional
            List of two values specifying the min and max rank percentiles for
            filtering when range_type is 'rank'. The default is [60, 90].
        pivot : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        gids : np.ndarray
            Numpy array of grain IDs (GIDs) that fall within the specified
            property range.
        A_B_values : np.ndarray
            Numpy array of property values corresponding to the selected GIDs.
        A_B_indices : pd.Index
            Pandas Index of the selected GIDs.
        
        Explanation of the data and sub-selection procedure
        --------------------------------------------------------
        To understand how the sub-selection is done, consider the following
        illustration of the property distribution:

        PROP_min--inf--------A-----nan------B---nan----inf------PROP_max
            1. clean data for inf and nans
            2. Then subselect from A to PROP_max
            3. Then subselect from A to B, which is what we need

        Example-1
        ---------
        gid, value, df_loc = PXGS.gs[8].get_gid_prop_range(PROP_NAME='aspect_ratio',
                                                   range_type='rank',
                                                   value_range=[80, 100]
                                                   )
        Example-2
        ---------
        gid, value, df_loc = PXGS.gs[8].get_gid_prop_range(PROP_NAME='area',
                                                   range_type='percentage',
                                                   value_range=[80, 100]
                                                   )
        Example-3
        ---------
        gid, value, df_loc = PXGS.gs[8].get_gid_prop_range(PROP_NAME='aspect_ratio',
                                                   range_type='value',
                                                   value_range=[2, 2.5]
                                                   )
        '''
        gids, A_B_values, A_B_indices = [], [], []
        if PROP_NAME in self.prop.columns:
            PROPERTY = self.prop[PROP_NAME].replace([-np.inf, np.inf],
                                                    np.nan).dropna()
            if range_type in ('percentage', '%',
                              'perc', 'by_percentage',
                              'by_perc', 'by%'
                              ):
                # If the user chooses to use percentage to describe the range
                # Get the minimum and maximum of the property
                PROP_min = PROPERTY.min()
                PROP_max = PROPERTY.max()
                # Calculate the fuill range if the proiperty
                PROP_range_full = PROP_max - PROP_min
                # Calculate the Lower cut-off
                lco = min(percentage_range)*PROP_range_full/100
                # Caluclate the upper cut-off
                uco = max(percentage_range)*PROP_max/100
                # w.r.t the the illustration in the DocString, subselect between A
                # and PROP_max
                A_MAX = self.prop[PROP_NAME][self.prop[PROP_NAME].index[self.prop[PROP_NAME] >= lco]]
                A_B_indices = A_MAX.index[A_MAX <= uco]
                A_B_values = A_MAX[A_B_indices].to_numpy()
                gids = A_B_indices+1
            elif range_type in ('value', 'by_value'):
                # If the user chooses to use values to describe the range of
                # objects
                lco = min(value_range)
                uco = max(value_range)
                # w.r.t the the illustration in the DocString, subselect between A
                # and PROP_max
                A_MAX = self.prop[PROP_NAME][self.prop[PROP_NAME].index[self.prop[PROP_NAME] >= lco]]
                A_B_indices = A_MAX.index[A_MAX <= uco]
                A_B_values = A_MAX[A_B_indices].to_numpy()
                gids = A_B_indices+1
            elif range_type in ('rank', 'by_rank'):
                '''
                # TODO: debug for the case where two entered values are same
                # TODO: Handle invalud user data
                '''
                values = self.prop[PROP_NAME]
                _ = values.replace([-np.inf,
                                    np.inf],
                                   np.nan).dropna().sort_values(ascending=False)
                indices = _.index
                ptile_i, ptile_j = [100-max(rank_range), 100-min(rank_range)]
                A_B_values = _[indices[int(ptile_i*_.size/100):int(ptile_j*_.size/100)]]
                A_B_indices = A_B_values.index
                gids = A_B_values.index.to_numpy()+1

        return gids, A_B_values, A_B_indices

    def plot_largest_grain(self):
        """
        Plot the largest grain in a temporal slice of a grain structure

        Returns
        -------
        None.

        # TODO: WRAP THIS INSIDE A FIND_LARGEST_GRAIN AND HAVE IT TRHOW
        THE GID TO THE USER

        """
        if 'area' in self.prop.columns:
            gid = self.prop['area'].idxmax()+1
        else:
            areas = self.find_grain_size_fast(metric='npixels')
            gid = 1
        self.g[gid]['grain'].plot()

    def plot_longest_grain(self):
        """
        A humble method to just plot the longest grain in a temporal slice
        of a grain structure

        Returns
        -------
        None.

        # TODO: WRAP THIS INSIDE A FIND_LONGEST_GRAIN AND HAVE IT TRHOW
        THE GID TO THE USER
        """
        gids, _, _ = self.get_gid_prop_range(PROP_NAME='aspect_ratio',
                                             range_type='percentage',
                                             percentage_range=[100, 100],
                                             )
        # plt.imshow(self.g[gid[0]]['grain'].bbox_ex)
        self.plot_grains_gids(list(gids))
        #for _gid_ in gid:
        #    self.g[_gid_]['grain'].plot()

    def mask_lgi_with_gids(self, gids, masker=-10):
        """
        Mask the lgi (PXGS.gs[n] specific lgi array: lattice of grain IDs)
        against user input grain indices, with a default UPXO-reserved
        place-holder value of -10.

        Parameters
        ----------
        gids : int/list
            Either a single grain index number or list of them
        kwargs:
            masker:
                An int value, preferably -10, but compulsorily less than -5.
        Returns
        -------
        s_masked : np.ndarray(dtype=int)
            lgi masked against gid values

        Internal calls (@dev)
        ---------------------
        None
        """

        # -----------------------------------------
        lgi_masked = deepcopy(self.lgi).astype(int)
        print('========================================')
        print(gids)
        print('========================================')
        for gid in gids:
            if gid in self.gid:
                lgi_masked[lgi_masked == gid] = masker
            else:
                print(f"Invalid gid: {gid}. Skipped")
        # -----------------------------------------
        return lgi_masked, masker

    def mask_s_with_gids(self, gids, masker=-10, force_masker=False):
        """
        Mask the s (PXGS.gs[n] specific s array) against user input grain
        indices

        Parameters
        ----------
        gids : int/list
            Either a single grain index number or list of them
        kwargs:
            masker:
                An int value, preferably -10.
            force_masker:
                This is here to satisfy the tussle of future development needs
                and user-readiness!! Please go with it for now.

                If True, user value for masker will be forced to
                masker variable, else the defaultr value of -10 will be used.

        Returns
        -------
        lgi_masked : np.ndarray(dtype=int)
            lgi masked against gid values

        Internal calls (@dev)
        ---------------------
        self.mask_lgi_with_gids()

        """
        # Validate suer supplied masker
        masker = (-10*(not force_masker) + int(masker*(force_masker and type(masker)==int)))
        # -----------------------------------------
        lgi_masked, masker = self.mask_lgi_with_gids(gids, masker)
        # -----------------------------------------
        if masker != -10:
            '''
            Redundant branching !!

            ~~RETAIN~~ as an entry space for further development for needs
            of having different masker values, example using differnet#
            masker values for different phases like particles, voids, etc.
            '''
            __new_mask__ = -10
            lgi_masked[lgi_masked == masker] = __new_mask__
            s_masked = deepcopy(self.s)
            s_masked[lgi_masked != __new_mask__] = masker
        else:
            __new_mask__ = -10
            lgi_masked[lgi_masked == -10] = __new_mask__
            s_masked = deepcopy(self.s)
            s_masked[lgi_masked != -10] = masker
        # -----------------------------------------
        return s_masked, masker

    def plotgs(self, figsize=(6, 6), dpi=120,
               custom_lgi=None,
               cmap='coolwarm', plot_cbar=True,
               title='Title',
               plot_centroid=False, plot_gid_number=False,
               centroid_kwargs={'marker': 'o',
                                'mfc': 'yellow',
                                'mec': 'black',
                                'ms': 2.5},
               gid_text_kwargs={'fontsize': 10},
               title_kwargs={'fontsize': 10},
               label_kwargs={'fontsize': 10}
               ):
        """
        Method to plot the grain structure of a temporal slice

        Parameters
        ----------
        figsize : tuple, optional
            Figure size in inches. The default is (6, 6).
        dpi : int, optional
            Dots per inch. The default is 120.
        custom_lgi : np.ndarray, optional
            Custom lgi to be plotted instead of the internal one. The default is None.
        cmap : str, optional
            Colormap name. The default is 'coolwarm'.
        plot_cbar : bool, optional
            Flag to plot colorbar. The default is True.
        title : str, optional
            Plot title. The default is 'Title'.
        plot_centroid : bool, optional
            Flag to plot centroids of grains. The default is False.
        plot_gid_number : bool, optional
            Flag to plot gid numbers at centroids. The default is False.
        centroid_kwargs : dict, optional
            Keyword arguments for centroid plotting. The default is
            {'marker': 'o', 'mfc': 'yellow', 'mec': 'black', 'ms': 2.5}.
        gid_text_kwargs : dict, optional
            Keyword arguments for gid text plotting. The default is
            {'fontsize': 10}.
        title_kwargs : dict, optional
            Keyword arguments for title. The default is {'fontsize': 10}.
        label_kwargs : dict, optional
            Keyword arguments for axis labels. The default is {'fontsize': 10}.

        Returns
        -------
        None.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[35].plotgs(figsize=(6, 6), dpi=120, cmap='coolwarm',
                           plot_centroid=True,
                           centroid_kwargs={'marker':'o','mfc':'yellow',
                                            'mec':'black','ms':2.5},
                           plot_gid_number=True)
        """
        plt.figure(figsize=figsize, dpi=dpi)
        if custom_lgi is None:
            LGI = self.lgi
        else:
            LGI = custom_lgi
        plt.imshow(LGI, cmap=cmap)
        if plot_centroid or plot_gid_number:
            centroid_x, centroid_y = [], []
            for gid in self.gid:
                centroid_x.append(self.xgr[self.lgi == gid].mean())
                centroid_y.append(self.ygr[self.lgi == gid].mean())
        if plot_centroid:
            plt.plot(centroid_x, centroid_y, linestyle='None',
                     marker=centroid_kwargs['marker'],
                     mfc=centroid_kwargs['mfc'], mec=centroid_kwargs['mec'],
                     ms=centroid_kwargs['ms'])
        if plot_gid_number:
            for i, (cenx, ceny) in enumerate(zip(centroid_x, centroid_y), start=1):
                plt.text(cenx, ceny, str(i),
                         fontsize=gid_text_kwargs['fontsize'])
        plt.xlabel(r"X-axis, $\mu m$", fontsize=label_kwargs['fontsize'])
        plt.ylabel(r"Y-axis, $\mu m$", fontsize=label_kwargs['fontsize'])
        plt.title(f'tslice={self.m}. {title}')
        if plot_cbar:
            plt.colorbar()

    def plot_grains_gids(self, gids, gclr='color', title="user grains",
                         cmap_name='CMRmap_r', ):
        """
        Method to plot grains specified by user input grain indices

        Parameters
        ----------
        gids : int/list
            Either a single grain index number or list of them
        title : TYPE, optional
            DESCRIPTION. The default is "user grains".
        gclr : str, optional
            Color scheme for plotting. The default is 'color'.
        cmap_name : str, optional
            Colormap name. The default is 'CMRmap_r'.

        Returns
        -------
        None.

        Example-1
        ---------
        After acquiring gids for aspect_ratio between ranks 80 and 100,
        we will visualize those grains.
        . . . . . . . . . . . . . . . . . . . . . . . . . .
        As we are only interested in gid, we will not use the other
        two values returned by PXGS.gs[n].get_gid_prop_range() method:

        gid, _, __ = PXGS.gs[8].get_gid_prop_range(PROP_NAME='aspect_ratio',
                                                    range_type='rank',
                                                    rank_range=[80, 100]
                                                    )
        . . . . . . . . . . . . . . . . . . . . . . . . . .
        Now, pass gid as input for the PXGS.gs[n].plot_grains_gids(),
        which will then plot the grain strucure with only these values:

        PXGS.gs[8].plot_grains_gids(gid, cmap_name='CMRmap_r')
        """
        if not dth.IS_ITER(gids):
            gids = [gids]
        if gclr not in ('binary', 'grayscale'):
            s, _ = self.mask_s_with_gids(gids)
            plt.imshow(s, cmap=cmap_name, vmin=1)
            plt.colorbar()
        elif gclr in ('binary', 'grayscale'):
            s, _ = self.mask_s_with_gids(gids,
                                         masker=0,
                                         force_masker=True)
            s[s != 0] = 1
            plt.imshow(s, cmap='gray_r', vmin=0, vmax=1)
        plt.title(title)
        plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
        plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
        plt.show()

    def plot_neigh_grains_of_gid(self, neigh_gid_subset):
        """
        Method to plot neighbouring grains of user specified grain indices

        Parameters
        ----------
        neigh_gid_subset : dict
            A dictionary with key as gid and value as list of neighbouring
            gids.

        Returns
        -------
        None.
        """
        if type(neigh_gid_subset) != dict:
            raise ValueError("Input neigh_gid_subset must be a dictionary")
        all_gids = []
        for gid, neighs in neigh_gid_subset.items():
            all_gids.append(gid)
            all_gids.extend(neighs)
            
        self.plot_grains_gids(neighs,
                                gclr='color',
                                title=f"Neigh grains of gid: {gid}",
                                cmap_name='CMRmap_r'
                                )

    def plot_grains_prop_range(self,
                               PROP_NAME='area',
                               range_type='percentage',
                               value_range=[1, 2],
                               percentage_range=[0, 20],
                               rank_range=[60, 90],
                               pivot=None,
                               gclr='color',
                               title=None,
                               cmap_name='CMRmap_r'
                               ):
        """
        Method to plot grains having properties within the domain defined by
        the range description specified by the user.

        Parameters
        ----------
        PROP_NAME : str, optional
            Name of the grain structure property. The default is 'area'.
        range_type : str, optional
            Range descript9ion type. The default is 'percentage'.
        value_range : iterable, optional
            Range of the actual PROP_NAME values. The default is [1, 2].
        percentage_range : iterable, optional
            Percentage range defining the PROP_NAME values. The default is
            [0, 20].
        rank_range : iterable, optional
            Ranks defining the range of PROP_NAME values.
            If rank_range=[6, 10] and there are 20 grains, then
            those grains having 12th to 20th largest PROP_NAME values will
            be selected. The default is [60, 90].
        pivot : str, optional
            Describes the range location.
            Options: ('ends', 'mean', 'primary_mode'):
                - If 'ends' and percentage_range=[5, 8], then this means that
                PROP_NAME vaklues between 5% and 8% of vaklues will be used to
                select the grains.
                - If 'mean' and percentage_range=[5, 8], then this means that
                PROP_NAME values between 0.95*mean and 1.08*mean will be used
                to select the grains.
                - If 'primary_mode' and percentage_range=[5, 8], then this
                means that PROP_NAME values between 0.95*primary_mode and
                1.08*primary_mode will be used to select the grains.
            The default is None.
        gclr : str, optional
            Specify whether grains are to have colours or grayscale.
            Choose 'binary' or 'grayscale' for grayscale
            The default is 'color'.
        title : str, optional
            DESCRIPTION.
            The default is None.
        cmap_name : str, optional
            DESCRIPTION.
            The default is 'CMRmap_r'.

        Returns
        -------
        None.

        """
        if range_type in ('percentage', 'value', 'rank'):
            gid, value, _ = self.get_gid_prop_range(PROP_NAME=PROP_NAME,
                                                    range_type=range_type,
                                                    rank_range=rank_range
                                                    )
            _rdesc_ = {'percentage': percentage_range,
                       'value': value_range,
                       'rank': rank_range
                       }
            title = f"Grains by area. \n {range_type} bounds: {_rdesc_[range_type]}"
            self.plot_grains_gids(gid,
                                  gclr='color',
                                  title=title,
                                  cmap_name=cmap_name
                                  )
        else:
            print(f"Invalid range_type: {range_type}")
            print("range_type must be either of the follwonig:")
            print(".......(percentage, value, rank)")

    def plot_large_grains(self, extent=5):
        """
        Method to plot large grains based on area percentage extent.

        Parameters
        ----------
        extent : int, optional
            Percentage extent to consider for large grains. The default is 5.

        Returns
        -------
        None.
        """
        gids, _, _ = self.get_gid_prop_range(PROP_NAME='area',
                                             range_type='percentage',
                                             percentage_range=[100-extent,
                                                               100],
                                             )
        for gid in gids:
            plt.imshow(self.g[gid]['grain'].bbox_ex)
        plt.imshow

    def plot_neigh_grains(self, gids=[None], throw=True,
                          gclr="color", title="Neigh grains", cmap_name="CMRmap_r"):
        """
        Method to plot neighbouring grains of user specified grain indices

        Parameters
        ----------
        gids : int/list, optional
            Either a single grain index number or list of them. The default is [None].
        throw : bool, optional
            Flag to return the list of neighbouring gids. The default is True.
        gclr : str, optional
            Color scheme for plotting. The default is 'color'.
        title : str, optional
            Plot title. The default is "Neigh grains".
        cmap_name : str, optional
            Colormap name. The default is 'CMRmap_r'.

        Returns
        -------
        neighbours : list
            List of neighbouring grain IDs.
        """
        neighbours = [self.g[gid]["grain"].neigh for gid in gids]
        _neighbours_ = []
        for neighs in neighbours:
            for gid in neighs:
                _neighbours_.append(gid)
        self.plot_grains_gids(gids=_neighbours_,
                              gclr=gclr,
                              title=title+f" of \n grains: {gids}",
                              cmap_name=cmap_name
                              )
        if throw:
            return neighbours

    def plot_grains_with_holes(self):
        # Use Euler number here
        pass

    def plot_skeletons(self):
        # Use sciki-image skeletenoise command here
        pass

    def plot(self, PROP_NAME=None, title='auto', cmap='CMRmap_r',
             vmin = 1, vmax = 5, ):
        '''
        Plot the grain structure based on user input property name

        Parameters
        ----------
        PROP_NAME : str, optional
            Name of the property to plot the grain structure by.
            The default is None.
        title : str, optional
            Plot title. The default is 'auto'.
        cmap : str, optional
            Colormap name. The default is 'CMRmap_r'.

        Returns
        -------
        None.

        Notes
        -----
        1. if no kwargs: plot the entire greain structure: just use plotgs()
        '''
        if not PROP_NAME:
            plt.imshow(self.s, cmap=cmap)
        elif PROP_NAME in ('npixels', 'area', 'aspect_ratio',
                           'perimeter', 'eq_diameter', 'solidity',
                           'eccentricity', 'compactness', 'circularity',
                           'major_axis_length', 'minor_axis_length'
                           ):
            PROP_LGI = deepcopy(self.lgi)
            for gid in self.gid:
                PROP_LGI[PROP_LGI==gid]=self.prop[PROP_NAME][gid-1]
            plt.imshow(PROP_LGI, cmap=cmap)
        elif PROP_NAME in ('phi1', 'psi', 'phi2'):
            pass
        elif PROP_NAME in ('gnd_avg'):
            pass
        if title == 'auto':
            title = f"Grain structure by {PROP_NAME}"
        plt.title(f"{title}")
        plt.xlabel("x-axis, um")
        plt.ylabel("y-axis, um")
        if PROP_NAME and PROP_NAME in ('aspect_ratio'):
            plt.colorbar(extend='both')
        else:
            plt.colorbar()
        plt.show()

    def plot_grain(self, gid, neigh=False, neigh_hops=1, save_png=False,
                   filename='auto', field_variable=None, throw=False):
        """
        Plots the nth grain.

        Parameters
        ----------
        Ng : int
            The grain number to plot. Grain number is global and not state
            specific.
        neigh : bool
            Flag to decide plotting of grains neighbouring to Ng
        neigh_hops : 1
            Non-locality of neighbours.
            If 1, only neighbours of Ng will be plotted along with Ng grain
            If 2, neighbours of neighbours of Ng will be plotted along with
            Ng grain
            NOTE: maximum number of hops permitted = 2
                  If a number greater than 2 is provided, then hops will be
                  restricted to 2.
        save_png : bool
            Flag to consider saving .png image to disk
        filename : str
            Use this filename for the .png imaage.
            If 'auto', then filename will be generated containing:
                * Grain structure temporal slice number
                * Global grain number
            If None or an invalid, image will not be saved to disk.
        field_variable : str
            Global field variable
            This is @ future development when SDVs can be re-mapped from
            CPFE simulation to UPXO.mcgs2d

        Returns
        -------
        grain_plot : bool
            matplotlib.plt.imshow object

        Example call
        ------------
            PXGS.gs[4].plot_grain(3, filename='t4_ng3.png'

        # TODO
            1. Add validity checking layer for gid
            2. Add validity check for save_png and filename
            2. Generate automatic filename
            3. Save image to file
            4. Add branching for dimensionality
            5. Add validity check for existence of data
        """
        operation_validity = False
        if self.g[gid]['grain']:
            if hasattr(self.g[gid]['grain'], 'bbox_ex'):
                if not neigh:
                    if not field_variable:
                        grain_plot = plt.imshow(self.g[gid]['grain'].bbox_ex)
                        operation_validity = True
                    else:
                        # 1. check field variable validity
                        # 2. check if the field variable data is available
                        # 3. Extract field data map relavant to current grain
                        #    only. No need to extract from remaining portions
                        #    of bbox_ex, whcih would be containing neighbouring
                        #    grains
                        # 4. PLot the data
                        pass
                else:
                    if hasattr(self.g[gid]['grain'], 'neigh'):
                        if len(self.g[gid]['grain'].neigh) > 0:
                            grain_plot = plt.imshow(self.g[gid]['grain'].bbox_ex)
                if save_png and type(filename) == str:
                    if filename == 'auto':
                        # Generate automatic filename
                        pass
                    else:
                        # Use the user input name for storing the filename.
                        pass
                    #  Save the image file
                elif save_png and type(filename) != str:
                    print("Invalid filename to store image")
                    pass

        if operation_validity and throw:
            return grain_plot

    def plot_grains(self, gids, hide_non_actors=True,
                    default_cmap='jet', title="user grains",
                    throw_plt_object=False, figsize=(6, 6), dpi=120):
        """
        Method to plot grains specified by user input grain indices

        Parameters
        ----------
        gids : iterable
            An iterable containing grain index numbers

        Example
        -------
        self.plot_grains([1, 2, 3, 4])
        """
        if not isinstance(gids, Iterable):
            raise TypeError('gids should be an Iterable')
        lgi = {gid: None for gid in gids}
        for gid in gids:
            lgi[gid] = gid*(self.lgi == gid)
        lgi = np.sum(list(lgi.values()), axis=0)
        if hide_non_actors:
            lgi[lgi == 0] = -10
            import matplotlib.cm as mpltcm
            cmap = mpltcm.get_cmap(default_cmap, 50)
            cmap.set_under('white')
        # ---------------------------------
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(lgi, vmin=1, cmap=cmap)
        plt.title(title)
        plt.xlabel(r"X-axis, $\mu m$")
        plt.ylabel(r"Y-axis, $\mu m$")
        plt.show()
        if throw_plt_object:
            return plt
        else:
            return None

    def plot_grains_prop_bounds_s(self,
                                  s,
                                  PROP_NAME=None,
                                  prop_min=0,
                                  prop_max='',
                                  ):
        pass

    def plot_grains_at_position(self, position='corner', overlay_centroids=True,
                                markersize=6, ):
        """
        Method to plot grains at specified positions in the grain structure

        Parameters
        ----------
        position : str, optional
            Position in the grain structure to plot grains at.
            Options: 'corner', 'boundary', 'triple_point'
            The default is 'corner'.
        overlay_centroids : bool, optional
            Flag to overlay centroids of the grains on the plot. The default is True.
        markersize : int, optional
            Size of the markers for centroids. The default is 6.

        Returns
        -------
        None.

        Example
        -------
        PXGS.gs[tslice].plot_grains_at_position(position='boundary')
        """
        LGI = deepcopy(self.lgi)
        boundary_array = self.positions[position]
        pseudos = np.arange(-len(boundary_array), 0)
        for pseudo, ba in zip(pseudos, boundary_array):
            LGI[LGI == ba] = pseudo
        LGI[LGI > 0] = 0
        for i, pseudo in enumerate(pseudos):
            LGI[LGI == pseudo] = boundary_array[i]
        plt.figure()
        plt.imshow(LGI)
        if overlay_centroids:
            for grain in self:
                if grain.gid in boundary_array:
                    x, y = grain.position[0:2]
                    plt.plot(x, y,
                             'ko',
                             markersize=markersize,
                             )
        plt.title(f"Corner grains. Ng: {len(self.positions[position])}")
        plt.xlabel("x-axis, um")
        plt.ylabel("y-axis, um")
        plt.show()

    def detect_grain_boundaries(self):
        for label in np.unique(self.lgi):
            pass

    def hist(self, PROP_NAME=None, bins=20, kde=True, bw_adjust=None,
             stat='density', color='blue', edgecolor='black', alpha=1.0,
             line_kws={'color': 'k', 'lw': 2, 'ls': '-'},
             auto_xbounds=True, auto_ybounds=True,
             xbounds=[0, 50], ybounds=[0, 0.2], peaks=False, height=0,
             prominance=0.2, __stack_call__=False, __tslice__=None, ):
        '''
        Plot histogram of grain property distribution

        Parameters
        ----------
        PROP_NAME : str, optional
            Name of the grain property. The default is None.
        bins : int, optional
            Number of bins for the histogram. The default is 20.
        kde : bool, optional
            Flag to plot kernel density estimate (KDE). The default is True.
        bw_adjust : float, optional
            Bandwidth adjustment for KDE. The default is None.
        stat : str, optional
            Statistic to plot. Options: 'density', 'frequency', 'count'.
            The default is 'density'.
        color : str, optional
            Color of the histogram bars. The default is 'blue'.
        edgecolor : str, optional
            Edge color of the histogram bars. The default is 'black'.
        alpha : float, optional
            Transparency of the histogram bars. The default is 1.0.
        line_kws : dict, optional
            Keyword arguments for the KDE line. The default is
            {'color': 'k', 'lw': 2, 'ls': '-'}.
        auto_xbounds : bool, optional
            Flag to automatically set x-axis bounds. The default is True.
        auto_ybounds : bool, optional
            Flag to automatically set y-axis bounds. The default is True.
        xbounds : list, optional
            User-defined x-axis bounds if auto_xbounds is False. The default is [0, 50].
        ybounds : list, optional
            User-defined y-axis bounds if auto_ybounds is False. The default is [0, 0.2].
        peaks : bool, optional
            Flag to identify and plot peaks in the KDE. The default is False.
        height : float, optional
            Minimum height of peaks to identify. The default is 0.
        prominance : float, optional
            Minimum prominence of peaks to identify. The default is 0.2.
        __stack_call__ : bool, optional
            Internal flag for stack calls. The default is False.
        __tslice__ : int, optional
            Temporal slice number for stack calls. The default is None.

        Returns
        -------
        None.
        '''
        if self.are_properties_available:
            if PROP_NAME in self.prop.columns:
                self.prop[PROP_NAME].replace([-np.inf, np.inf],
                                             np.nan,
                                             inplace=True
                                             )
                sns.histplot(self.prop[PROP_NAME].dropna(),
                             bins=bins,
                             kde=False,
                             stat=stat,
                             color=color,
                             edgecolor=edgecolor,
                             line_kws=line_kws
                             )
                if kde and bw_adjust:
                    if peaks:
                        x, y = (sns.kdeplot(data=self.prop[PROP_NAME].dropna(),
                                            bw_adjust=bw_adjust,
                                            color=line_kws['color'],
                                            linewidth=line_kws['lw'],
                                            fill=False,
                                            alpha=0.5,
                                            ).lines[0].get_data()
                                )
                        peaks, peaks_properties = find_peaks(y,
                                                             height=0,
                                                             prominence=0.02
                                                             )
                        plt.plot(x, y)
                        plt.plot(x[peaks],
                                 peaks_properties["peak_heights"],
                                 "o",
                                 markerfacecolor='black',
                                 markersize=8,
                                 markeredgewidth=1.5,
                                 markeredgecolor='black')
                        plt.vlines(x=x[peaks],
                                   ymin=y[peaks] - peaks_properties["prominences"],
                                   ymax=y[peaks],
                                   color="gray",
                                   linewidth=1,
                                   )
                        # Find the minima and plot it
                        minima_indices = argrelextrema(y, np.less)[0]
                        plt.plot(x[minima_indices],
                                 y[minima_indices],
                                 "s",
                                 markerfacecolor='white',
                                 markersize=8,
                                 markeredgewidth=1.5,
                                 markeredgecolor='black')
                    else:
                        sns.kdeplot(self.prop[PROP_NAME].dropna(),
                                    bw_adjust=bw_adjust,
                                    label='KDE',
                                    color=line_kws['color'],
                                    linewidth=line_kws['lw'],
                                    fill=False,
                                    alpha=0.5,
                                    )
                if kde and not bw_adjust:
                    if peaks:
                        x, y = (sns.kdeplot(data=self.prop[PROP_NAME].dropna(),
                                            color=line_kws['color'],
                                            linewidth=line_kws['lw'],
                                            fill=False,
                                            alpha=0.5,
                                            ).lines[0].get_data()
                                )
                        peaks, peaks_properties = find_peaks(y,
                                                             height=0,
                                                             prominence=0.02
                                                             )
                        plt.plot(x, y)
                        plt.plot(x[peaks],
                                 peaks_properties["peak_heights"],
                                 "o",
                                 markerfacecolor='black',
                                 markersize=8,
                                 markeredgewidth=1.5,
                                 markeredgecolor='black')
                        plt.vlines(x=x[peaks],
                                   ymin=y[peaks] - peaks_properties["prominences"],
                                   ymax=y[peaks],
                                   color="gray",
                                   linewidth=1,
                                   )
                        # Find the minima and plot it
                        minima_indices = argrelextrema(y, np.less)[0]
                        plt.plot(x[minima_indices],
                                 y[minima_indices],
                                 "s",
                                 markerfacecolor='white',
                                 markersize=8,
                                 markeredgewidth=1.5,
                                 markeredgecolor='black')
                if __stack_call__:
                    plt.title(f"Distribution of {PROP_NAME} @ tslice: {__tslice__}")
                else:
                    plt.title(f"Distribution of {PROP_NAME}")
                plt.xlabel(f'{PROP_NAME}')
                plt.ylabel(f'{stat}')
                if auto_xbounds == 'user':
                    plt.xlim(xbounds)
                if auto_ybounds == 'user':
                    plt.ylim(ybounds)
                plt.show()
            else:
                if not __stack_call__:
                    print(f"PROP_NAME: {PROP_NAME} has not yet been caluclated. Skipped")
        else:
            print(f"PROP_NAME: {PROP_NAME} has not yet been caluclated. Skipped")

    def kde(self, PROP_NAMES, bw_adjust, ):
        '''
        Plot kernel density estimate (KDE) of grain property distribution

        Parameters
        ----------
        PROP_NAMES : list
            List of grain property names.
        bw_adjust : float
            Bandwidth adjustment for KDE.

        Returns
        -------
        None.
        '''
        print(PROP_NAMES)
        for PROP_NAME in PROP_NAMES:
            if PROP_NAME in self.prop.columns:
                self.prop[PROP_NAME].replace([-np.inf, np.inf],
                                             np.nan,
                                             inplace=True
                                             )
                sns.kdeplot(self.prop[PROP_NAME].dropna(),
                            bw_adjust=bw_adjust,
                            label='KDE',
                            color='red', attrs=['bold'])
                plt.title(f"{PROP_NAME} distribution")
                plt.xlabel(f"{PROP_NAME}")
                plt.ylabel("Density")
                plt.legend()
            if PROP_NAME == PROP_NAMES[-1]:
                plt.show()

    @decorators.port_doc('upxo.viz.dataviz', 'see_distr')
    def see_distr(self, viz='hist', prop_names=['area', 'perimeter', 'orientation', 'solidity', ],
            props={'area': [], 'perimeter': [], 'orientation': [], 'solidity': []},
            prop_units={'area': 'μm²', 'perimeter': 'μm', 'orientation': 'degrees', 'solidity': ''},
            probability_density=False, 
            nbins_values={'area': 30, 'perimeter': 30, 'orientation': 30, 'solidity': 30},
            bw_adjust_values={'area': None, 'perimeter': None, 'orientation': None, 'solidity': None},
            alpha_values={'area': 0.7, 'perimeter': 0.7, 'orientation': 0.7, 'solidity': 0.7},
            color_values={'area': 'blue', 'perimeter': 'blue', 'orientation': 'blue', 'solidity': 'blue'},
            edgecolor_values={'area': 'black', 'perimeter': 'black', 'orientation': 'black', 'solidity': 'black'},
            binsize=30, alpha=0.7, color='blue', edgecolor='black',
            ncolumns=3, ylabel='count', gsdim=2):
        from upxo.viz.dataviz import see_distr
        see_distr(gsdim=gsdim, viz=viz,
                prop_data_format='dataframe', prop_df=self.prop,
                prop_names=prop_names, props=props,
                prop_units=prop_units,
                probability_density=probability_density, 
                nbins_values=nbins_values, bw_adjust_values=bw_adjust_values,
                alpha_values=alpha_values, color_values=color_values,
                edgecolor_values=edgecolor_values,
                binsize=binsize, alpha=alpha, color=color, edgecolor=edgecolor,
                ncolumns=ncolumns, ylabel=ylabel)

    def femesh(self, saa=True, throw=False, ):
        '''
        Set up finite element mesh of the poly-xtal

        Parameters
        ----------
        saa : bool, optional
            Flag to set the mesh attribute of the grain structure object.
            The default is True.
        throw : bool, optional
            Flag to return the mesh object. The default is False.

        Returns
        -------
        mesh : pxtal.mesh.mesh_mcgs2d object
            Finite element mesh of the grain structure.

        Notes
        -----
        Use saa=True to update grain structure mesh atttribute
        Use saa=True and throw=True to update and return mesh
        Use saa=False and throw=True to only return mesh
        '''
        # from mcgs import _uidata_mcgs_gridding_definitions_
        # uigrid = _uidata_mcgs_gridding_definitions_(self.uinputs)
        # from mcgs import _uidata_mcgs_mesh_
        # uimesh = _uidata_mcgs_mesh_(self.uinputs)

        from upxo.meshing.mesher_2d import mesh_mcgs2d

        if saa:
            self.mesh = mesh_mcgs2d(self.uinputs['uimesh'],
                                    self.uigrid, self.dim, self.m, self.lgi)
            if throw:
                return self.mesh
        if not saa:
            if throw:
                return mesh_mcgs2d(self.uinputs['uimesh'],
                                   self.uigrid, self.dim, self.m, self.lgi)
            else:
                return 'Please enter valid saa and throw arguments'

    @property
    def pxtal_length(self):
        # Calculate length of the pxtal in microns
        return self.uigrid.xmax-self.uigrid.xmin+self.uigrid.xinc

    @property
    def pxtal_height(self):
        # Calculate height of the pxtal in microns
        return self.uigrid.ymax-self.uigrid.ymin+self.uigrid.yinc

    @property
    def pxtal_area(self):
        # Calculate area of the pxtal in square microns
        return self.pxtal_length*self.pxtal_height

    @property
    def centroids(self):
        # Calculate centroids of the grains
        centroids = []
        for gid in self.gid:
            locs = self.lgi == gid
            centroids.append([self.xgr[locs].mean(), self.ygr[locs].mean()])
        return np.array(centroids)

    @property
    def bboxes(self):
        # Calculate bounding boxes of the grains
        return [grain.bbox for grain in self]

    @property
    def bboxes_bounds(self):
        # Calculate bounding box bounds of the grains   
        return [grain.bbox_bounds for grain in self]

    @property
    def bboxes_ex(self):
        # Calculate extended bounding boxes of the grains
        return [grain.bbox_ex for grain in self]

    @property
    def bboxes_ex_bounds(self):
        # Calculate extended bounding box bounds of the grains
        return [grain.bbox_ex_bounds for grain in self]

    @property
    def areas(self):
        # Calculate areas of the grains
        return np.array([self.px_size*grain.loc.shape[0] for grain in self])

    @property
    def areas_min(self):
        # Calculate minimum area of the grains
        return self.areas.min()

    @property
    def areas_mean(self):
        # Calculate mean area of the grains
        return self.areas.mean()

    @property
    def areas_std(self):
        # Calculate standard deviation of the areas of the grains
        return self.areas.std()

    @property
    def areas_var(self):
        # Calculate variance of the areas of the grains
        return self.areas.var()

    @property
    def areas_max(self):
        # Calculate maximum area of the grains
        return self.areas.max()

    @property
    def areas_stat(self):
        # Calculate statistics of the areas of the grains
        areas = self.areas
        return {'min': areas.min(),
                'mean': areas.mean(),
                'max': areas.max(),
                'std': areas.std(),
                'var': areas.var()
                }

    @property
    def aspect_ratios(self):
        # Calculate aspect ratios of the grains
        gid_stright_grains = self.straight_line_grains
        mj_axis = [grain.skprop.axis_major_length for grain in self]
        mn_axis = [grain.skprop.axis_minor_length for grain in self]
        npixels = [len(grain.loc) for grain in self]
        ar = []
        for i, (npx, mja, mna) in enumerate(zip(npixels, mj_axis, mn_axis)):
            if i+1 not in gid_stright_grains:
                ar.append(mja/mna)
            else:
                if npx == 1:
                    ar.append(1)
                else:
                    ar.append(len(self.g[i+1]['grain'].loc))
        return ar

    @property
    def aspect_ratios_min(self):
        # Calculate minimum aspect ratio of the grains
        return self.aspect_ratios.min()

    @property
    def aspect_ratios_mean(self):
        # Calculate mean aspect ratio of the grains
        return self.aspect_ratios.mean()

    @property
    def aspect_ratios_std(self):
        # Calculate standard deviation of the aspect ratios of the grains
        return self.aspect_ratios.std()

    @property
    def aspect_ratios_var(self):
        # Calculate variance of the aspect ratios of the grains
        return self.aspect_ratios.var()

    @property
    def aspect_ratios_max(self):
        # Calculate maximum aspect ratio of the grains
        return self.aspect_ratios.max()

    @property
    def aspect_ratios_stat(self):
        # Calculate statistics of the aspect ratios of the grains
        aspect_ratios = self.aspect_ratios
        return {'min': aspect_ratios.min(),
                'mean': aspect_ratios.mean(),
                'max': aspect_ratios.max(),
                'std': aspect_ratios.std(),
                'var': aspect_ratios.var()
                }

    @property
    def npixels(self):
        # Calculate number of pixels of the grains
        npx = np.array([len(grain.loc) for grain in self])
        return npx

    @property
    def single_pixel_grains(self):
        # Retrieve the grain IDs of single pixel grains
        return np.where(self.npixels == 1)[0]+1

    @property
    def plot_single_pixel_grains(self):
        # Plot single pixel grains
        self.plot_grains_gids(self.single_pixel_grains)

    @property
    def straight_line_grains(self):
        # get the axis lengths of all availabel grains
        mja = [grain.skprop.axis_major_length for grain in self]
        mna = np.array([grain.skprop.axis_minor_length for grain in self])
        # retrieve the grains where minor axis is zero. These are the grains
        # where skimage is unable to fit ellipse, as they are unit pixel wide.
        # some of them could be for single pixel grains too.
        gid_mna0 = list(np.where(mna == 0)[0]+1)
        # Now, retrieve the single pixel grains.
        gid_npx1 = self.single_pixel_grains
        # Remove the single pixel grains
        if len(gid_npx1) > 0:
            # This means single pixel grains exist
            for _gid_npx1_ in gid_npx1:
                gid_mna0.remove(_gid_npx1_)
            gid_ar = np.array([len(self.g[_gid_mna0_]['grain'].loc)
                              for _gid_mna0_ in gid_mna0])
        return np.array(gid_mna0, dtype=int), gid_ar

    @property
    def locations(self):
        # Calculate locations of the grains
        return [grain.position for grain in self]

    @property
    def perimeters(self):
        # Calculate perimeters of the grains
        characteristic_length = math.sqrt(self.px_size)
        return np.array([characteristic_length*grain.gbloc.shape[0]
                         for grain in self])

    @property
    def perimeters_min(self):
        # Calculate minimum perimeter of the grains
        return self.perimeters.min()

    @property
    def perimeters_mean(self):
        # Calculate mean perimeter of the grains
        return self.perimeters.mean()

    @property
    def perimeters_std(self):
        # Calculate standard deviation of the perimeters of the grains
        return self.perimeters.std()

    @property
    def perimeters_var(self):
        # Calculate variance of the perimeters of the grains
        return self.perimeters.var()

    @property
    def perimeters_stat(self):
        # Calculate statistics of the perimeters of the grains
        perimeters = self.perimeters
        return {'min': perimeters.min(),
                'mean': perimeters.mean(),
                'max': perimeters.max(),
                'std': perimeters.std(),
                'var': perimeters.var()
                }

    @property
    def ratio_p_a(self):
        # Calculate perimeter to area ratio of the grains
        return np.array([p/a for p, a in zip(self.perimeters, self.areas)])

    @property
    def AF_bgrains_igrains(self):
        # Calculate area fractions of boundary grains and internal grains
        areas = self.areas
        A_bgr = [areas[gid-1]
                 for gid in np.unique(self.positions['boundary'])]
        A_igr = [areas[gid-1]
                 for gid in np.unique(self.positions['internal'])]
        pxtal_area = self.pxtal_area
        AF = (np.array(A_bgr).sum()/pxtal_area,
              np.array(A_igr).sum()/pxtal_area)
        return AF

    @property
    def grains(self):
        # Generator to iterate over grains
        return (_ for _ in self)

    def make_mulpoint2d_grain_centroids(self):
        # Create mulpoint2d object for grain centroids
        from upxo.geoEntities.mulpoint2d import MPoint2d
        self.mp['gc'] = MPoint2d.from_coords(self.centroids)
        # self.mp['gc'] = mulpoint2d(self.centroids)

    def plot_mcgs_mpcentroids(self):
        """
        Method to plot the mulpoint2d object of grain centroids
        """
        plt.figure()
        # Plot the grain structure
        plt.imshow(self.s)
        # Plot the grain mulpoints of the grain centroids
        plt.plot(self.mp['gc'].locx,
                 self.mp['gc'].locy,
                 'ko',
                 markersize=6)
        plt.xlabel('x-axis um', fontdict={'fontsize': 12})
        plt.ylabel('y-axis um', fontdict={'fontsize': 12})
        plt.title(f"MCGS tslice:{self.m}.\nUPXO.mulpoint2d of grain centroids",
                  fontdict={'fontsize': 12})
        plt.show()

    def vtgs2d(self, visualize=True):
        """
        Method to generate voronoi tesselation grain structure from
        mulpoint2d object of grain centroids
        """
        # from polyxtal import polyxtal2d as polyxtal
        from upxo.pxtal.polyxtal import vtpolyxtal2d as vtpxtal
        self.make_mulpoint2d_grain_centroids()
        self.vtgs = vtpxtal(gsgen_method='vt',
                            vt_base_tool='shapely',
                            point_method='mulpoints',
                            mulpoint_object=self.mp['gc'],
                            xbound=[self.uigrid.xmin,
                                    self.uigrid.xmax+self.uigrid.xinc],
                            ybound=[self.uigrid.ymin,
                                    self.uigrid.ymax+self.uigrid.yinc],
                            vis_vtgs=visualize
                            )
        if visualize:
            self.vtgs.plot(dpi=100,
                           default_par_faces={'clr': 'teal', 'alpha': 1.0, },
                           default_par_lines={'width': 1.5, 'clr': 'black', },
                           xtal_marker_vertex=True, xtal_marker_centroid=True)

    def ebsd_write_ctf(self, folder='upxo_ctf', file='ctf.ctf'):
        """
        Method to write a sample ctf file for testing purposes

        Parameters
        ----------
        folder : str
            Folder to write the ctf file to.
        file : str
            Name of the ctf file.

        Returns
        -------
        None.
        """
        x = np.arange(0, 100.1, 2.5)
        y = np.arange(0, 100.1, 2.5)
        X, Y = np.meshgrid(x, y)

        PHI1 = np.random.uniform(low=0, high=360, size=X.shape)
        PSI = np.random.uniform(low=0, high=360, size=X.shape)
        PHI2 = np.random.uniform(low=0, high=180, size=X.shape)

        os.makedirs(folder, exist_ok=True)
        file = file
        file_path = os.path.join(folder, file)

        with open(file_path, 'w') as f:
            f.write("Channel Text File\n")
            f.write("Prj	C:/CHANNEL5_olddata/Joe's Creeping Crud/Joes creeping crud on Cu/Cugrid_after 2nd_15kv_2kx_2.cpr\n")
            f.write("Author	[Unknown]\n")
            f.write("JobMode	Grid\n")
            f.write("XCells	550\n")
            f.write("YCells	400\n")
            f.write("XStep	0.1\n")
            f.write("YStep	0.1\n")
            f.write("AcqE1	0\n")
            f.write("AcqE2	0\n")
            f.write("AcqE3	0\n")
            f.write("Euler angles refer to Sample Coordinate system (CS0)!	Mag	2000	Coverage	100	Device	0	KV	15	TiltAngle	70	TiltAxis	0\n")
            f.write("Phases	1\n")
            f.write("3.6144;3.6144;3.6144	90;90;90	Copper	11	225	3803863129_5.0.6.3	-906185425	Ann. Acad. Sci. Fenn., Ser. A6 [AAFPA4], vol. 223A, pages 1-10\n")
            f.write('Phase X Y Euler1 Euler2 Euler3\n')
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    x = X[i, j]
                    y = Y[i, j]
                    phi1 = PHI1[i, j]
                    psi = PSI[i, j]
                    phi2 = PHI2[i, j]
                    f.write(f"1 {x} {y} {phi1} {psi} {phi2}\n")
        f.close()

    def export_vtk2d(self):
        """
        Method to export the grain structure data to a VTK file for 2D visualization.
        """
        pass

    def export_ctf(self, folder, fileName, pathFinding='direct',
            headerFileLocation='C:\\Development\\UPXO\\upxo_library\\src\\upxo\\_writer_data\\_ctf_header_CuCrZr_1.txt',
            factor=1, method='nearest'):
        """
        Exports the grain structure data to a CTF file for use in MTEX or
        Dream3D's h5ebsd reconstruction pipeline.

        Parameters
        ----------
        folder : str
            Directory path string where the CTF file will be exported.
        fileName : str
            Name of the CTF file to be exported.
        factor : float
            Resolution modification factor.
            If factor < 1.0, the grid resolution will be decreased by the
            specified factor using decimation.
            If factor = 1.0, the grid resolution will remain unchanged.
            If factor > 1.0, the grid resolution will be increased by the
            specified factor using nearest neighbor interpolation.
        method : str
            Method to modify grid resolution.
            Options include 'nearest' and 'decimate'.

        Returns
        -------
        None.

        Example
        -------
        ctf.export_ctf('D:/export_folder', 'sunil')

        CODES before the modication:

            from upxo._sup.export_data import ctf
            ctf = ctf()
            ctf.load_header_file()
            ctf.make_header_from_lines()
            ctf.set_phase_name(phase_name='PHNAME')
            # ------------------------------------
            ctf.set_grid(self.xgr, self.ygr)
            ctf.set_state(self.S, self.s)
            # ------------------------------------
            '''UPDATE TO BE MADE ASAP.'''
            # ctf.set_ori(self.euler1, self.euler2, self.euler3)
            ctf.set_grid_data()
            # ndata = ctf.assemble_grid_data()
            # ndata = ctf.assemble_grid_data_orix()
            ctf.write_ctf_file_ORIX(folder, fileName)
        """
        if method not in ('nearest', 'decimate'):
            raise ValueError('Invalid method provided. Valid: nearest or decimate')
        from upxo._sup.export_data import ctf
        ctf = ctf()
        ctf.load_header_file(pathFinding=pathFinding, filePath=folder,
                             headerFileLocation=headerFileLocation)
        ctf.make_header_from_lines()
        ctf.set_phase_name(phase_name='PHNAME')
        # ------------------------------------
        if factor > 0.0 and factor < 1.0:
            XGRID, YGRID, SMATRIX = decrease_grid_resolution(self.xgr, self.ygr, self.s, factor)
        elif factor == 1.0:
            XGRID, YGRID, SMATRIX = self.xgr, self.ygr, self.s
        elif factor > 1.0:
            XGRID, YGRID, SMATRIX = increase_grid_resolution(self.xgr, self.ygr, self.s, factor)
        # ------------------------------------
        ctf.set_grid(XGRID, YGRID)
        ctf.set_state(self.S, SMATRIX)
        """UPDATE TO BE MADE ASAP."""
        # ctf.set_ori(self.euler1, self.euler2, self.euler3)
        ctf.set_grid_data()
        # ndata = ctf.assemble_grid_data()
        # ndata = ctf.assemble_grid_data_orix()
        ctf.write_ctf_file_ORIX(folder, fileName)

    def export_slices(self, xboundPer=None, yboundPer=None, zboundPer=None,
                      mlist=None, sliceStepSize=None, sliceNormal=None,
                      xoriConsideration=None, resolution_factor=None,
                      exportDir=None, fileFormats=None, overwrite=None, ):
        """
        Exports datafiles of slices through the grain structures.

        Parameters
        ----------
        xboundPer : list/tuple
            (min%, max%), min% < max%. min% shows the percentage length from
            grid's xmin, where the bound starts and the max% shows the
            percentage xlength from grid's xmin, where the bounds ends
        yboundPer : list/tuple
            (min%, max%), min% < max%. min% shows the percentage length from
            grid's ymin, where the bound starts and the max% shows the
            percentage ylength from grid's ymin, where the bounds ends
        zboundPer : list/tuple
            (min%, max%), min% < max%. min% shows the percentage length from
            grid's zmin, where the bound starts and the max% shows the
            percentage ylength from grid's zmin, where the bounds ends
        mlist : list/tuple of int values
            List of monte-carlo temporal time values, where slices are needed.
            For each entry, a seperate folder will be created.
        sliceStepSize : int
            Pixel-distance (number of pixels) between each individual slice.
            Minimum should be 1, in which case, the every adjqacent possible
            slice will be sliced and exported. If 2, slices 0, 2, 4, ... will
            be considered. If 5, slices, 0, 5, 10, ... will be considered.
        sliceNormal : str
            Options include x, y, z
        xoriConsideration : dict
            Xtal orientation consideration
            Mandatory key: 'method'. Options include:
                * 'ignore'. Only when crystallographical orientations have
                already been mapped to grains.
                * 'random'. Value could be a dummy value.
                * 'userValues'. Value to be a numpy array of 3 Bunge's Euler
                angles, shaped (nori, 3).
                * 'import'.
        resolution_factor : float
        exportDir : str
            Directory path string which would be parent directory for all
            exports made from this PXGS.export_slices(.). If directory does
            not exit, it will be created.
        fileFormats : dict
            Keys include txt, h5d, ctf, vtk.
            * Include txt or h5d to export for for further work in UPXO
            * Include ctf for export to MTEX or Dream3D's h5ebsd reconstruction
            pipeline
            * Include vtk2d for export to VTK format of each slice
            * Include vtk3d for export to VTK of entire grain structure
        overwrite : bool
            If True, any existing contents in all child directories inside
            exportDir will be overwritten
            If False, existing contents will not be altered.

        Returns
        -------
        None.

        Example-1
        ---------
        xboundPer = (0, 100)
        yboundPer = (0, 100)
        zboundPer = (0, 100)
        mlist = [0, 10, 20]
        sliceStepSize = 1
        sliceNormal = 'z'
        xoriConsideration = {'method': 'random'}
        exportDir = 'FULL PATH'
        fileFormats = {'.ctf': {},
                       '.vtk3d': {},
                       }
        overwrite = True
        PXGS.export_slices(xboundPer,
                           yboundPer,
                           zboundPer,
                           mlist,
                           sliceStepSize,
                           sliceNormal,
                           exportDir,
                           fileFormats,
                           overwrite)
        """
        xsz = math.floor((self.uigrid.xmax-self.uigrid.xmin)/self.uigrid.xinc)
        ysz = math.floor((self.uigrid.ymax-self.uigrid.ymin)/self.uigrid.yinc)
        zsz = math.floor((self.uigrid.zmax-self.uigrid.zmin)/self.uigrid.zinc)
        Smax = self.uisim.S
        slices = list(range(0, 9, sliceStepSize))
        phase_name = 1
        phi1 = np.random.rand(Smax)*180
        psi = np.random.rand(Smax)*90
        phi2 = np.random.rand(Smax)*180
        textureInstanceNumber = 1

    def import_ctf(self,
                   filePath,
                   fileName,
                   convertUPXOgs=True):
        # Use DefDAP to get the job done here
        pass

    def import_crc(self,
                   filePath,
                   fileName,
                   convertUPXOgs=True):
        # Use DefDAP to get the job done here
        pass

    def clean_exp_gs(self,
                     minGrainSize=10
                     ):
        # Use DefDAP to get the job done here
        pass

    def import_dream3d(self,
                       filePath,
                       fileName,
                       convertUPXOgs=True):
        # Use DefDAP to get the job done here
        pass
