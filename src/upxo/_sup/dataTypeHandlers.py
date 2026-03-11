"""Handler functions to deal with different types of datatypes and formats.

This module contains a collection of handlers which help conversion between
different dataypes in and between UPXO, shapely, scipy, numpy, VTK and GMSH.

This module imports the following on a need by need basis:
    . numpy
    . cKDTree from scipy
    . UPXO > point2d
    . UPXO > point2d_lean_highest
    . UPXO > point2d_lean_highest_mc0
    . UPXO > point2d_lean_highest_mc1

This module contains the following classes:
    . dt
    . valg
    . options

This module contains the following functions:
    . _compare_
    . make_upxo_point2d_RANDU
    . unique_of_datatypes
    . point_list_to_coordxy
    . UpxoPointList_to_coords
    . ShapelyPointList_to_coords
    . UpxoPointList_to_ckdtree
    . UpxoPointListOfList_to_coords
    . UpxoMultiPoint_to_coords
    . UpxoMultiPointList_to_coords
    . UpxoMultiPointList_to_ckdtrees
    . coords_to_UpxoPointList

NOTE: NOT TO BE SHARED WITH ANYONE OTHER THAN:
    *@UKAEA: Sunil Anandatheertha, Chris Hardie, Vikram Phalke
    *@UKAEA:  Ben Poole, Allan Harte, Cori Hamelin
    *@OX,UKAEA:  Eralp Demir, Ed Tarleton
"""

from dataclasses import dataclass
from collections import deque
import numpy as np
# from numpy.random import uniform as npru
# from scipy.spatial import cKDTree as ckdt
from upxo._sup import dataTypeHandlers as dth  # Let this be
# import pandas as pd
# from point2d_04 import point2d


@dataclass
class dt():
    """
    Abbreviations:
        UPXO: UKAEA Poly-XTAL Operations
        SH: Shapely
        LSTRING: LineString
        2D: Two-dimensional
        3D: Three-dimensional
    """
    import types
    NUMBERS = (int, float, np.float16, np.float32, np.float64,
               np.int32, np.int64)
    ITERABLES = (list, tuple, np.ndarray, deque)
    INTEGERS = (int, np.int8, np.int16, np.int32, np.int64)
    FLOATS = (float, np.float16, np.float32, np.float64)
    GENERATOR = (types.GeneratorType)
    UPXO_POINTS = ('point2d')
    UPXO_EDGES = ('edge2d')
    UPXO_2D = ('point2d', 'edge2d')
    SH_LSTRING_2D = ('LineString')
    ALL_EDGES_2D = ('edge2d', 'LineString')
    ALL_COMPARATORS = ('<', 'lt', '<=', 'le', '==', '!=',
                       'eq', '>=', 'ge', '>', 'gt', 'ne')
    # USES OF NUM_UPXOE2D_SHLS
    NUM_UPXOE2D_SHLS = (int, float, np.float16, np.float32, np.float64,
                        'edge2d', 'LineString')

class constants():
    """
    Class containing mathematical constants

    NOTE
    ----
        Please import as K
    """
    from numpy import inf
    # Mathematical constants
    PI = 3.141592653589793238
    # Conversion constnta
    d2r = 0.017453292519943295  # Deg to rad multiplier
    r2d = 57.29577951308232  # Rad to deg multipler
    # Numpy infinity
    INF = inf
    # Number of decimal places to consider while rounding
    ROUND_ZERO_DEC_PLACE = 10

    EPS_p2d_above = 10**-8
    EPS_p2d_below = 10**-8
    EPS_p2d_left = 10**-8
    EPS_p2d_right = 10**-8
    EPS_p2d_divisor = 10**-8

    # Edge2d: PRIMARY GLOBAL FOR EPS low AND high VALUES
    # if S be the slope, and bound extremes are BE: S-EPS_e2dl_low and
    # S+EPS_e2dl_high, then the BOUND is [min(BE), max(BE)]
    EPS_e2dl_low, EPS_e2dl_high = 10**-2, 10**-2
    # Edge2d, Edge3d: EPS differences between coordinates.
    # Helps establish xcoord bounds and ycoord bounds
    EPS_e2d_delx, EPS_e2d_dely, EPS_e2d_delz = 10**-6, 10**-6, 10**-6
    # Edge2d: PRIMARY GLOBAL FOR maximum and minimum slopes.
    # if slope <= -EPS_e2ds_lowest, then slope == INFINITY
    # if slope >= EPS_e2ds_highest, then slope == INFINITY
    EPS_e2ds_lowest, EPS_e2ds_highest = -10**-3, 10**3
    # Edge3d: PRIMARY GLOBAL FOR maximum and minimum slopes.~['//]
    EPS_e3ds_xy_lowest = -10**-5
    EPS_e3ds_yz_lowest = -10**-5
    EPS_e3ds_zx_lowest = -10**-5
    EPS_e3ds_xy_highest = 10**-5
    EPS_e3ds_yz_highest = 10**-5
    EPS_e3ds_zx_highest = 10**-5


class functions():
    """
    Usage:
        from datatype_handlers import functions as f

    f.cosine
    """
    cosd = np.cos
    sind = np.sin
    tand = np.tan

@dataclass
class valg():
    """
    Provides the valid set of algorithms needed for validating inputs.
    """
    mc2d = ('200', 200,
            '201', 201,
            '202', 202,
            '203', 203,
            '204', 204
            )
    # --------------------------------
@dataclass
class opt():
    """
    Abbreviations:
        TM: Test method
        CP: Comparison Parameter
    """
    upxo_upxo = ('upxo', 'ukaea', 'upolxop', )
    angle_unit_deg = ('degrees', 'degs', 'deg', 'd', )
    angle_unit_rad = ('radians', 'rads', 'rad', 'r', )
    upxo_point1d = ('upxo_point1d', 'point1d', 'upoint1d', 'up1d', 'up1', 'p1d', 'p1', )
    upxo_point2d = ('upxo_point2d', 'point2d', 'upoint2d', 'up2d', 'up2', 'p2d', 'p2', )
    upxo_point3d = ('upxo_point3d', 'point3d', 'upoint3d', 'up3d', 'up3', 'p3d', 'p3', )
    upxo_point1d_list = ('upxo_point1d_list', 'point1d_list', 'p1d_list', 'p1dlist', 'p1list', )
    upxo_point2d_list = ('upxo_point2d_list', 'point2d_list', 'p2d_list',
                         'up2dlist', 'up2d_list', 'up2list', 'up2_list',
                         'upoints2d', 'p2dlist', 'p2list', 'points2d', )
    upxo_point2d_list_list = ('upxo_point2d_list_list', 'point2d_list_list',
                              'p2d_list_list', 'p2dlist_list', 'p2list_list',
                              'points2d_list', 'upxo_point2d_lists',
                              'point2d_lists', 'p2d_lists', 'p2dlists',
                              'p2lists', 'points2ds',
                              'upoint2d_lists', 'up2d_lists', 'up2dlists',
                              'up2lists', 'upoints2ds', )
    upxo_point3d_list = ('upxo_point3d_list', 'point3d_list', 'p3d_list', 'p3dlist', 'p3list', )
    shapely_point2d = ('shapely_point', 'shapely_point2d', 'sh_p2d', 'sh_p2', 'shp2', )
    shapely_point3d = ('shapely_point', 'shapely_point3d', 'sh_p3d', 'sh_p3', 'shp3', )
    coord_point2d = ('point_coord_2d', 'point_coord_2d', 'coord2d', 'xy', ) # Preferred: 'xy'
    coord_point3d = ('point_coord_3d', 'point_coord_3d', 'coord3d', 'xyz', )
    coord_point2d_list = ('point_coord_2d_list', 'point_coord_2d_list',
                          'coord2d_list', 'xy_coord_list', 'xy_list', 'xylist',
                          'coord_list', 'clist', ) # Preferred: 'xy_list'
    coord_list = ('xy_list', 'xylist',) # PREFERRED: 'xy_list'
    coord_point2d_list_list = ('point_coord_2d_list_list',
                               'coord2d_list_list', 'xy_coord_list_list',
                               'xy_list_list', 'clist_list', 'coord_list_list',
                               'point_coord_2d_lists', 'coord2d_lists',
                               'xy_coord_lists', 'xy_lists', 'clist_lists',
                               'coord_lists', 'clists', ) # Preferred: 'xy_lists'
    coord_point3d_list = ('point_coord_3d_list', 'point_coord_3d_list',
                          'coord3d', 'xyz_coord_list', 'xyz_list', )
    coord_pair_point2d = ('xy_coord_pair', 'coord_pair', ) # Preferred: 'xy_coord_pair'
    coord_pairs = ('cpairs',)  # USe for making edges # Preferred: 'cpairs'
    coord_pairs_list = ('cpairs_list',)  # USe for making list of edges # Preferred: 'cpairs_list'
    coord_pairs_point2d_list = ('xy_coord_pairs_list', 'xy_coord_pair_list', 'xy_pair_list', ) # Preferred: 'xy_pair_list'
    coord_xlist_ylist = ('xcoord_ycoord', 'xlist_ylist', )
    point2d_tdist = ('tdist', 'toldist', 'tolerance', )
    point2d_jn = ('jn', 'bjn', 'bj_n', 'j_n', 'jporder', 'xvo', 'xtal_vertex_order', )
    point2d_ptype = ('ptype', 'point_type', )
    point2d_loc = ('loc', 'pxtal_loc', )
    point2d_rid = ('rid', 'randid', 'randomid', 'random_id', )
    point2d_mid = ('mid', 'omid', 'memory_id', 'mem_id', 'memid', 'object_memory_id', )
    point2d_dim = ('dim', 'dimensionality', )
    point2d_pol_area = ('sfv_pol_area', 'area', )
    point2d_pol_ar = ('sfv_pol_ar', 'pol_ar', 'pol_aspect_ratio', )
    point2d_pol_perimeter = ('sfv_pol_perimeter', 'pol_perimeter', 'pol_perimeter', )
    point2d_pol_phaseid = ('phase_id', 'phase_id', 'phase_id_number', )
    point2d_pol_phasename = ('phase_name', 'phasename', 'phase_name', 'phase',)
    point2d_pol_tcname = ('tcname', 'texcomp', 'texture_component', 'ori_name', 'oriname', )
    point2d_pol_earepr = ('earepr', 'euler_angle_repr', 'earepresentation',
                          'euler_angle_representation', 'sfv_earepr',
                          'sfv_euler_angle_repr', 'sfv_earepresentation',
                          'sfv_euler_angle_representation', )
    point2d_pol_eaunit = ('eaunit', 'euler_angle_unit', 'eaunits', 'euler_angle_units', )
    point2d_pol_eangle = ('eangle', 'euler_angle', 'euler_angles', 'eangles',
                          'sfv_eangle', 'sfv_euler_angle', 'sfv_euler_angles', 'sfv_eangles', )
    name_point2d_leans = ('p2lean', 'p2dlean', 'leanest', 'p2dleanest', 'point2dlean', 'point2dleanest')
    name_point2d = ('p2d', 'point2d')
    name_mulpoint2d = ('mp2d', 'mulp2d', 'mpoint2d', 'mulpoint2d')
    name_point2d_specs = ('Point2d', '[Point2d]', 'p2d_leanest',
                          '[p2d_leanest]', 'type-[1,2]', 'type-[[1,2]]',
                          'type-[[1,2],[3,4],[5,6]]', 'type-[[1,2,3,4],[5,6,7,8]]')
    name_point3d_specs = ('Point3d', '[Point3d]', 'p3d_leanest',
                          '[p3d_leanest]', 'type-[1,2,3]', 'type-[[1,2,3]]',
                          'type-[[1,2,3],[4,5,6],[7,8,9]]',
                          'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]')
    name_point_specs = name_point2d_specs + name_point3d_specs
    # --------------------------------
    # Preferred: 'mp2d'
    upxo_mp2d = ('mulpoint2d', 'mp2d', )
    # Preferred: 'ump2d_list'
    upxo_mp2d_list = ('mulpoint2d_list', 'list_of_mp2d',
                      'mp2_list', 'mp2d_list', 'u_mulpoint2d_list',
                      'u_list_of_mp2d', 'ump2_list', 'ump2d_list', )
    # --------------------------------
    upxo_xtal2d = ('upxo_xtal2d', )
    upxo_xtal2d_list = ('upxo_xtal2d_list', )
    shapely_xtal2d = ('shapely_xtal2d',)
    shapely_xtal2d_list = ('shapely_xtal2d',)
    scipy_xtal2d = ('scipy_cell2d',)
    # --------------------------------
    upxo_pxtal2d = ('upxo_pxtal2d', )
    # --------------------------------
    ckdtree = ('ckdtree', 'ckdt', )
    # Preferred: 'ckdt_list'
    ckdtree_list = ('ckdtree_list',  'ckdt_list', )
    # Preferred: 'ckdts'
    ckdtree_lists = ('ckdtree_lists', 'ckdt_lists', )
    srtree = ('shapely_srtree', 'srtree', 'shapely_tree', )
    srtree_list = ('shapely_srtree_list', 'srtree_list', 'shapely_tree_list',
                   'shapely_srtrees', 'srtrees', 'shapely_trees', )
    # --------------------------------
    upxo_mp = ('mulpoints', 'mulpoint', 'mp', 'mpoints', )
    upxo_mp_grid_type = ('trigrid1', 'trigrid2', 'recgrid', 'hexgrid',
                         'random', )
    # --------------------------------
    translate_xyincr = ('xyby', 'xydist', 'xydel', 'delxy',
                        'incrxy', 'xyincr', )
    translate_xincr = ('xby', 'xdist', 'xdel', 'delx', 'incrx', 'xincr', )
    translate_yincr = ('yby', 'ydist', 'ydel', 'dely', 'incry', 'yincr', )
    translate_xynew = ('xyalign', 'align', 'alignxy', 'xynew', 'newxy', )
    translate_xnew = ('xalign', 'align', 'alignx', 'xnew', 'newx', )
    translate_ynew = ('yalign', 'align', 'aligny', 'ynew', 'newy', )
    # --------------------------------
    TM_slope_slope = ('slope', 'slopes', 'by_slopes', 'by_slope', )
    TM_slope_angle = ('angle', 'angles', 'inclination',
                      'inclinations', 'by_angles', )
    CP_length_slope = ('length_slope', 'length_and_slope',
                       'slope_and_length', 'slope_length', )
    upxo_edge_length = ('l', 'len', 'e_len', 'e_length',
                        'el', 'edge_length', 'edge2d_length',
                        'lengths', 'edge_lengths',
                        'upxo_edge_length', 'upxo_edge_lengths',
                        'line_length', 'line_lengths', 'upxo_line_lengths',
                        'linestring_lengths', 'linestring_length', )
    upxo_edge_slope = ('s', 'slope', 'e_s', 'e_slope',
                       'edge_slope', 'edge2d_slope',
                       'slopes', 'edge_slopes',
                       'upxo_edge_slopes', 'upxo_edge_slopes',
                       'line_slope', 'line_slopes', 'upxo_line_slopes',
                       'linestring_slopes', 'linestring_slope', )
    CP_edge_length = ('l', 'len', 'e_len', 'e_length',
                      'el', 'edge_length', 'edge2d_length',
                      'lengths', 'edge_lengths',
                      'upxo_edge_length', 'upxo_edge_lengths',
                      'shapely_edge_lengths', 'line_length',
                      'line_lengths', 'upxo_line_lengths',
                      'shapely_line_lengths', 'linestring_lengths',
                      'linestring_length', )
    CP_edge_slope = ('s', 'slope', 'es', 'edge_slope',
                     'slopes', 'edge_slopes', 'upxo_edge_slopes', )
    CP_muledge_lengths = ('m_e_l', 'm_edge_len', 'm_edge_length',
                          'm_edge_lengths', 'mul_edge_lengths',
                          'multi_edge_length', 'multi_edge_lengths',
                          'mul_el', 'mul_edge_length', 'mul_edge_lengths',
                          'multi_edge_length', 'multi_line_lengths',
                          'mul_line_lengths', )
    CP_pol_areas = ('a', 'area', 'pol_area', 'p_a', 'poly_area'
                    'polygonal_area', 'grain_area', 'g_area',
                    'x_area', 'xtal_area', 'c_area',
                    'crystal_area', 'polygonal_area',
                    'xtal_pol_area', 'grain_pol_area',
                    'crystal_pol_area', 'xtal_polygonal_area',
                    'grain_polygonal_area', 'crystal_polygonal_area', )
    CP_chull_lengths = ('cv_h_length', 'cv_h_l', 'l_c_h', 'c_h_l',
                        'c_hull_length', 'convex_hull_length',
                        'length_convex_hull', )

    CP_positives = np.array([CP_edge_length, CP_muledge_lengths,
                             CP_pol_areas, CP_chull_lengths,
                             ],
                            dtype='object').sum()
    e2d_addto_both_pnts = ('bothpoints', 'ab', '12', 'pntapntb', 'pntsab',
                           'bothends', 'pointapointb', )
    e2d_addto_pnta = ('firstpoint', 'pointa', 'point1', 'start', 'pnta',
                      'pnt1', 'startingpoint', 'a', '1', )
    e2d_addto_pntb = ('secondpoint', 'pointb', 'point2', 'end', 'pntb',
                      'pnt2', 'endingpoint', 'b', '2', )
    addtocoord = ('x', 'y', 'xy', 'both', )
    ps_studies = ('gs_analysis',
                  'gs_mesh',
                  'gs_growth',
                  'cpfe_data_analysis'
                  '_development_',
                  )
    mcgs2d_reload_data_xlsx = ('pl', 'ebsd', 'grid', 'simpar', 'sim', 'ori',
                               'gsc', 'mps', 'grep', 'rqual', 'ps', 'mesh',
                               'gsvis', 'export', 'log', 'report')
    ocv_options = ('opencv', 'ocv', 'cv', 'cv2')
    ski_options = ('scikit-image', 'skimg', 'ski', 'si')
    cc3d_options = ('cc3d', 'connected_components_3d', 'cc_3d', 'cc3d_lib')
    # BO: Branch options
    mc_BO = ('mc', 'montecarlo')
    mc2d_BO = ('mc2', 'mc2d', 'montecarlo2', 'montecarlo2d')
    mc3d_BO = ('mc3', 'mc3d', 'montecarlo3', 'montecarlo3d')
    mc_BO_all = mc_BO + mc2d_BO + mc3d_BO

    vt_BO = ('vt', 'voronoi', 'voronoitessellation', 'geo', 'geotess')
    vt2d_BO = ('vt2', 'vt2d', 'voronoi2d', 'voronoi2', 'geo2', 'geotess2')
    vt3d_BO = ('vt3', 'vt3d', 'voronoi3d', 'voronoi3', 'geo3', 'geotess3')
    vt_BO_all = vt_BO + vt2d_BO + vt3d_BO


@dataclass
class valid_region_properties():
    scikitimage_region_properties2d = ['num_pixels', 'area', 'area_bbox',
                                       'area_convex', 'area_filled',
                                       'axis_major_length',
                                       'axis_minor_length', 'bbox', 'centroid',
                                       'centroid_local', 'centroid_weighted',
                                       'centroid_weighted_local',
                                       'coords_scaled', 'coords',
                                       'eccentricity',
                                       'equivalent_diameter_area',
                                       'euler_number', 'extent',
                                       'feret_diameter_max', 'image',
                                       'image_convex', 'image_filled',
                                       'image_intensity', 'inertia_tensor',
                                       'inertia_tensor_eigvals',
                                       'intensity_max', 'intensity_mean',
                                       'intensity_min', 'label', 'moments',
                                       'moments_central', 'moments_hu',
                                       'moments_normalized',
                                       'moments_weighted',
                                       'moments_weighted_central',
                                       'moments_weighted_hu', 'solidity',
                                       'moments_weighted_normalized',
                                       'orientation', 'perimeter',
                                       'perimeter_crofton', 'slice']

def strip_str(options, char=['_', ',', '.', '&', '|', '-', '*']):
    """
    Remove `char` (@ character) from individual words in options.

    Parameters
    ----------
    options : list, tuple, deque
        Iterable containing options. Each option must be a string
    sym : TYPE, optional
        DESCRIPTION. The default is '_'.

    Returns
    -------
    tuple
        Containing original options stirped off char from it.
        Each new option will be a string without the 'char'.
    Example
    -------
    options = ('pol', 'pol_area', 'polarea')
    dth.strip_str(options)
    """
    import re
    if not isinstance(options, str):
        raise TypeError('Invalid type(options). Expected string. ',
                        f'Receieved: {type(options)}')
    extract = False
    if type(options) not in dt.ITERABLES:
        options = [options]
        extract = True
    if type(char) not in dt.ITERABLES:
        char = [char]
    for ch in char:
        options = tuple(set(re.sub("["+ch+"]", '', o) for o in options))
    return options[0] if extract else options


def make_list(data, force_list=False):
    """
    Data could be a single edge object

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    force_list : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    if type(data) not in dt.ITERABLES:
        if data.__class__.__name__ in dt.UPXO_EDGES:
            data = [data]
        elif data.__class__.__name__ in dt.NUMBERS:
            data = [data]
    if type(data) in dt.ITERABLES:
        data = [d for d in data]
    if force_list:
        data = [data]
    return data

###############################################################################
###############################################################################
###############################################################################


def find_neighdata_ckdt_list(xself, yself, ckdtrees, cor, nworkers):
    """
    INPUT VARIABLES:
        xself: x-coordinate of the central point
        yself: y-coordinate of the central point
        ckdtrees: list of ckdtree data-structures
        cor: cut-off radius
        nworkers: number of workers
    OUTPUT VARIABLES:
        indices: indices of points in ckdtree
        distances: actual distances in the
        neigh: actual neighbouring points
               list of two lists. 1st: x-coordinates, 2nd: y-coordinates
        nneigh: list of numbers. Number in the nth place in this list, denotes
                the number of points in the neighbourhood of [xself, yself]
                in each of the tree in ckdtrees.
    """
    indices, distances, neigh, nneigh = [], [], [], []
    for _tree_ in ckdtrees:
        # Find indices of shortlisted points from the original dataset
        ind = _tree_.query_ball_point([xself, yself], cor, workers=nworkers)
        # coordinates of shortlisted points
        _locxy_ = _tree_.data[ind].T
        # actual distances of shortlisted points
        dist = np.sqrt((xself-_locxy_[0])**2+(yself-_locxy_[1])**2)
        # Indices to sort distancesa in ascending order
        ascend_ind = np.argsort(dist)
        # Distances in ascending order
        dist_ascend = dist[ascend_ind]
        # coordinate list of points in ascending distances
        locxy_ascend = _locxy_.T[ascend_ind].T
        # COLLECT THE DATA
        indices.append(ind)
        distances.append(dist_ascend)
        neigh.append(locxy_ascend)
        nneigh.append(len(locxy_ascend[0]))

    return indices, distances, neigh, nneigh


def make_ckdt_from_coordpair_lists(np_coord_pair_lists, copy_data=False, balanced_tree=True):
    # EXAMPLE INPUT FOR "np_coord_pair_lists"
    # np_coord_pair_lists = [np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2],
    #                                  [0.3, 0.3], [0.3, 0.3], [0.1, 0.2],
    #                                  [0.3, 0.5], [1.0, 0.8], [0.0, 0.1],
    #                                  [0.1, 0.0]]).T,
    #                        np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2],
    #                                  [0.3, 0.3], [0.3, 0.3], [0.1, 0.2],
    #                                  [0.3, 0.5], [1.0, 0.8], [0.0, 0.1],
    #                                  [0.1, 0.0]]).T
    #                        ]
    from scipy.spatial import cKDTree
    tree = [cKDTree(coord_pair_list, copy_data=copy_data, balanced_tree=balanced_tree) 
            for coord_pair_list in np_coord_pair_lists]
    return tree


def make_ckdt_from_coord_lists(np_coord_lists,
                               copy_data=False,
                               balanced_tree=True):
    # EXAMPLE INPUT FOR "np_coord_lists"
    # np_coord_lists = [np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2],
    #                             [0.3, 0.3], [0.3, 0.3], [0.1, 0.2],
    #                             [0.3, 0.5], [1.0, 0.8], [0.0, 0.1],
    #                             [0.1, 0.0]]).T,
    #                   np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2],
    #                             [0.3, 0.3], [0.3, 0.3], [0.1, 0.2],
    #                             [0.3, 0.5], [1.0, 0.8], [0.0, 0.1],
    #                             [0.1, 0.0]]).T
    #                  ]
    from scipy.spatial import cKDTree
    tree = [cKDTree(coord_list, copy_data=copy_data, balanced_tree=balanced_tree) for coord_list in np_coord_lists]
    return tree

def make_ckdt_from_upxo_points2d():
    pass


def make_ckdt_from_mpoints2d():
    pass
###############################################################################
###############################################################################
###############################################################################
def make_point2d_RANDU_xy_coord_list(n=5, xlim=[0, 1], ylim=[0, 1]):
    '''
    from datatype_handlers import make_point2d_RANDU_xy_coord_list
    '''
    R = np.random.rand
    return np.vstack((R(1, n)*(xlim[1]-xlim[0])+xlim[0], R(1, n)*(ylim[1]-ylim[0])+ylim[0]))


def make_point2d_RANDU_xy_coordpair_list(n=5, xlim=[0, 1], ylim=[0, 1]):
    '''
    from datatype_handlers import make_point2d_RANDU_xy_coordpair_list
    '''
    R = np.random.rand
    return np.vstack((R(1, n)*(xlim[1]-xlim[0])+xlim[0], R(1, n)*(ylim[1]-ylim[0])+ylim[0])).T


def make_upxo_point2d_RANDU(n_points):
    '''
    from datatype_handlers import make_upxo_point2d_RANDU
    '''
    from point2d import point2d
    from numpy.random import uniform as npru
    return [point2d(x=npru(), y=npru()) for _ in range(n_points)]


def unique_of_datatypes(data_list):
    """
    Parameters
    ----------
    data_list : TYPE
        DESCRIPTION.

    Returns
    -------
    set_data_types : TYPE
        DESCRIPTION.

    """
    return list(set(str(type(data)) for data in data_list))


def are_all_numbers(lst):
    if type(lst) not in dt.ITERABLES:
        lst = [lst]

    return all(isinstance(x, (int, float, np.float16, np.float32, np.float64)) for x in lst)

def are_all_strings(lst):
    if type(lst) not in dt.ITERABLES:
        lst = [lst]

    return all(isinstance(x, str) for x in lst)

def IS_ITER(obj):
    return isinstance(obj, dt.ITERABLES)

def IS_CPAIR(obj):
    is_cpair_truth_value = False
    if IS_ITER(obj):
        if ALL_NUM(obj):
            is_cpair_truth_value = True
    return is_cpair_truth_value


def ALL_ITER(obj):
    return all([isinstance(_, dt.ITERABLES) for _ in obj])

def ALL_NUM(obj):
    return all([isinstance(_, dt.NUMBERS) for _ in obj])

def ALL_UP2D(obj):
    """
    obj = [point2d(0,0), point2d(1,0), point2d(2,0)]
    dth.ALL_UP2D(obj)
    """
    upxo_point2d_list = False
    if IS_ITER(obj):
        type_string = list(set([str(_.__class__.__name__) for _ in obj]))
        if len(type_string) == 1 and type_string[0] == 'point2d':
            upxo_point2d_list = True
    return upxo_point2d_list

def DEEPCHECK_is_coord2d_list(obj):
    """
    obj = [[1, 2], [3, 4]]
    obj = [[1, 2]]
    dth.DEEPCHECK_is_coord2d_list(obj)
    """
    coord_list_truth_value = False
    if type(obj) in (list, tuple, np.ndarray, deque):
        if ALL_ITER(obj):
            if all(len(_) == 2 for _ in obj):
                if all(ALL_NUM(_) for _ in obj):
                    coord_list_truth_value = True
    return coord_list_truth_value

def DEEPCHECK_is_xy2d_list(obj):
    """
    obj = [[1, 2, 3, 4, 5], [1, 2, 32, 43, 54]]
    obj = [[1, 2]]
    obj = [1, 2]
    dth.DEEPCHECK_is_xy2d_list(obj)
    """
    xy_list_truth_value = False
    if IS_ITER(obj) and len(obj) == 2 and ALL_ITER(obj):
        if len(obj[0])==len(obj[1]):
            if ALL_NUM(obj[0]) and ALL_NUM(obj[1]):
                xy_list_truth_value = True
    return xy_list_truth_value


def point_list_to_coordxy(point_objects):
    """


    Parameters
    ----------
    point_objects : TYPE
        DESCRIPTION.

    Returns
    -------
    coord : TYPE
        DESCRIPTION.

    """
    set_types_points = unique_of_datatypes(point_objects)
    if len(set_types_points) == 1:
        if set_types_points[0] == "<class 'UPXO-point.point2d'>":
            coord = np.array(UpxoPointList_to_coords(
                point_objects, target_type='list_locxy'))
        elif set_types_points[0] == "<class 'shapely.geometry.point.Point'>":
            # Same as for point2d, but reproduiced to make space
            # for future compatibiulity
            coord = np.array(ShapelyPointList_to_coords(
                point_objects, target_type='list_locxy'))
        elif set_types_points[0] in ("<class 'list'>",
                                     "<class 'numpy.ndarray'>"):
            # INPUT FORMAT: [[x1, y1], [x2, y2], ... [xn, yn]]
            # Operation redundant when np.array. But, its ok.
            coord = np.array(point_objects)
        elif set_types_points[0] in ("<class 'tuple'>"):
            # TODO: branching to be developed
            pass
    else:
        # When the point_objects contains a mixture of points of
        # different datatypes.
        # TODO: branching to be developed
        pass
    return coord


def UpxoPointList_to_coords(list_of_points, target_type='np@ckdtree'):
    """
    Takes a list of UPXO point objects and returns tehe coordinate list
    in the format specified by the "target_type" argument.

    OPTIONS:
        target_type:
            1. np@ckdtree: coordinate list needed to construct scipy's ckdtree
            2. list_locxy: list of [x, y] for each point
            3. list_locx_locy: list of [x] and [y] of all points
            4. shapely@srtree: targetted at shapely's stree data-structure
            5. shapely@mulpoints: targetted at shapely's mulpoints object

    TODO: @DEVELOPER
        1. codes for 'shapely@srtree':
        2. codes for 'shapely@mulpoint'

    TODO: @DOCUMENTATION
        1. Example for 'shapely@srtree':
        2. Examnple for for 'shapely@mulpoint'

    PRE-EXAMPLE: Let us first create list of 20 upxo points
        from point2d_04 import point2d
        from numpy.random import uniform as randu
        points = [point2d(x=randu(), y=randu()) for _ in range(20)]
        from datatype_handlers import UpxoPointList_to_coords

    EXAMPLE:
        np@ckdtree:
            UpxoPointList_to_coords(points, target_type = 'np@ckdtree')
        list_locxy:
            UpxoPointList_to_coords(points, target_type = 'list_locxy')
        list_locx_locy:
            UpxoPointList_to_coords(points, target_type = 'list_locx_locy')
        shapely@srtree:
            example to come here
        shapely@mulpoint
            example to come here
    """
    locxy = [[point.x, point.y] for point in list_of_points]
    if target_type == 'list_locxy':
        coordinates = locxy
    if target_type == 'list_locx_locy':
        coordinates = np.array(locxy).T
    if target_type == 'np@ckdtree':
        coordinates = np.array(locxy)
    if target_type == 'shapely@srtree':
        pass
    if target_type == 'shapely@mulpoint':
        pass

    return coordinates


def ShapelyPointList_to_coords(list_of_points, target_type='np@ckdtree'):
    """
    Takes a list of Shapely point objects and returns tehe coordinate list
    in the format specified by the "target_type" argument.

    OPTIONS:
        target_type:
            1. np@ckdtree: coordinate list needed to construct scipy's ckdtree
            2. list_locxy: list of [x, y] for each point
            3. list_locx_locy: list of [x] and [y] of all points
            4. shapely@srtree: targetted at shapely's stree data-structure
            5. shapely@mulpoints: targetted at shapely's mulpoints object

    TODO: @DEVELOPER
        1. codes for 'shapely@srtree':
        2. codes for 'shapely@mulpoint'

    TODO: @DOCUMENTATION
        1. Example for 'shapely@srtree':
        2. Examnple for for 'shapely@mulpoint'

    PRE-EXAMPLE: Let us first create list of 20 upxo points
        from point2d_04 import point2d
        from numpy.random import uniform as randu
        points = [point2d(x=randu(), y=randu()) for _ in range(20)]
        from datatype_handlers import UpxoPointList_to_coords

    EXAMPLE:
        np@ckdtree:
            UpxoPointList_to_coords(points, target_type = 'np@ckdtree')
        list_locxy:
            UpxoPointList_to_coords(points, target_type = 'list_locxy')
        list_locx_locy:
            UpxoPointList_to_coords(points, target_type = 'list_locx_locy')
        shapely@srtree:
            example to come here
        shapely@mulpoint
            example to come here
    """
    locxy = [[point.x, point.y] for point in list_of_points]
    if target_type == 'list_locxy':
        coordinates = locxy
    if target_type == 'list_locx_locy':
        coordinates = np.array(locxy).T
    if target_type == 'np@ckdtree':
        coordinates = np.array(locxy)
    if target_type == 'shapely@srtree':
        pass
    if target_type == 'shapely@mulpoint':
        pass

    return coordinates


def inlist(data):
    if type(data) not in dt.ITERABLES:
        data = [data]
    return data


def UpxoPointList_to_ckdtree(list_of_points):
    """
    Takes in a list of upxo point objects and convert to ckdtree

    PRE-EXAMPLE:
        import numpy as np
        from numpy.random import uniform as randu
        n_points = 10
        locxy = [[randu(), randu()] for _ in range(n_points)]
        from datatype_handlers import coords_to_UpxoPointList
        points = coords_to_UpxoPointList(locxy, lean = 'no')

    EXAMPLE:
        from datatype_handlers import UpxoPointList_to_ckdtree
        UpxoPointList_to_ckdtree(points)
    """
    from scipy.spatial import cKDTree
    return cKDTree(dt.UpxoPointList_to_coords(list_of_points, target_type='np@ckdtree'), copy_data=False, balanced_tree=True)


def UpxoPointListOfList_to_coords(list_of_list_of_points):
    """


    Parameters
    ----------
    list_of_list_of_points : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    return []


def UpxoMultiPoint_to_coords(mulpoint,
                             target_type='locxy'
                             ):
    """Take a UPXO mulpoint and make the coord list of its points.

    Parameters
    ----------
    mulpoint : mulpoint2d
        UPXO multi-point object

    Returns
    -------
    coordinate list

    Pre-example
    -----------
    from point2d import point2d
    from numpy.random import uniform as randu
    from mulpoint2d import mulpoint2d

    points = [point2d(x=randu(), y=randu()) for _ in range(20)]
    mp = mulpoint2d(method = 'points', point_objects = points)

    Example
    -------
    from datatype_handlers import UpxoMultiPoint_to_coords
    UpxoMultiPoint_to_coords(mp, target_type = 'locxy')
    UpxoMultiPoint_to_coords(mp, target_type = 'locx_locy')
    """
    if hasattr(mulpoint, 'locx') and hasattr(mulpoint, 'locy'):
        if target_type in ('locxy', 'xycoord'):
            coordinates = np.vstack((mulpoint.locx, mulpoint.locy)).T
        elif target_type in ('locx_locy', 'coordx_coordy',
                             'locxlocy', 'coordxcoordy'):
            coordinates = np.vstack((mulpoint.locx, mulpoint.locy))
    return coordinates


def UpxoMultiPointList_to_coords(list_of_mulpoints):
    """Take a mulpoint list and return list of coord lists of each's points.

    Parameters
    ----------
    list_of_mulpoints : list

    Returns
    -------

    Pre-example
    -----------
    from point2d import point2d
    import datatype_handlers as dth
    from numpy.random import uniform as randu
    points1 = [point2d(x=randu(), y=randu()) for _ in range(20)]
    mp1 = mulpoint2d(method = 'points', point_objects = points1)
    points2 = [point2d(x=randu(), y=randu()) for _ in range(20)]
    mp2 = mulpoint2d(method = 'points', point_objects = points2)
    mp_list = [mp1, mp2]

    dth.UpxoMultiPointList_to_coords(mp_list)
    Example
    -------

    """
    for mp in list_of_mulpoints:
        if hasattr(mp, 'locx') and hasattr(mp, 'locy'):
            locxy = [np.vstack((mp.locx, mp.locy)).T
                     if (hasattr(mp, 'locx') and hasattr(mp, 'locy'))
                     else None for mp in list_of_mulpoints]
        else:
            print('Please ensure that the multi-point object has both')
            print('    locx and locy attributes')
    return locxy


def UpxoMultiPointList_to_ckdtrees(list_of_mulpoints):
    """Take list of upxo mulpoints and return list of ckdtrees

    Parameters
    ----------
    list_of_mulpoints : list

    Returns
    -------

    Pre-examples
    ------------
    from point2d import point2d
    import datatype_handlers as dth
    from numpy.random import uniform as randu
    from scipy.spatial import cKDTree as ckdt

    points1 = [point2d(x=randu(), y=randu()) for _ in range(20)]
    points2 = [point2d(x=randu(), y=randu()) for _ in range(20)]

    mp_list = [mulpoint2d(method = 'points', point_objects = points1),
               mulpoint2d(method = 'points', point_objects = points2)]

    Examples
    --------

    tree = dth.UpxoMultiPointList_to_ckdtrees(mp_list)
    """
    return [mp.maketree(saa=False, throw=True) for mp in list_of_mulpoints]


def coords_to_UpxoPointList(coords=None,
                            dim=2,
                            coords_format='locxy',
                            lean='no'
                            ):
    """
    Takes coords and makes UPXO point object based on lean option and
    dimensionality. May require the coord data format

    #TODO: @DEVELOPER
        1. Codes for point2d_lean_highest
        2. Codes for point2d_lean_highest_mc0
        3. Codes for point2d_lean_highest_mc1

    PRE-EXAMPLE:
        import numpy as np
        from numpy.random import uniform as randu
        import datatype_handlers as dth
        n_points = 10
        coords = [[randu(), randu()] for _ in range(n_points)]
        coords1 = np.array(coords).T
        coords2 = [coords1[0], coords1[1]]
        from datatype_handlers import coords_to_UpxoPointList

    EXAMPLE - 1:
        dth.coords_to_UpxoPointList(coords = coords,  dim = 2,
                                    coords_format = 'locxy',
                                    lean = 'no')
    EXAMPLE - 2:
        dth.coords_to_UpxoPointList(coords = coords1,  dim = 2,
                                    coords_format = 'locx_locy',
                                    lean = 'no')
    EXAMPLE - 3:
        dth.coords_to_UpxoPointList(coords = coords2,  dim = 2,
                                    coords_format = 'locx_and_locy',
                                    lean = 'no')
    """
    from point2d import point2d
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    if lean in ('no', 'lowest', 'low', 'medium',
                'high', 'veryhigh', 'highest'):
        if coords_format in ('locxy', 'coordxy'):
            _point_list = [point2d(x=coord[0], y=coord[1], lean='no'
                                   ) for coord in coords]
        elif coords_format in ('locxlocy', 'locx_locy',
                               'coordxcoordy', 'coordx_coordy',
                               'locx_and_locy', 'coordx_and_coordy'):
            _point_list = [point2d(x=coordx, y=coordy, lean='no'
                                   ) for coordx, coordy in zip(coords[0],
                                                               coords[1])]
    elif lean == 'point2d_lean_highest':
        pass
    elif lean == 'point2d_lean_highest_mc0':
        pass
    elif lean == 'point2d_lean_highest_mc1':
        pass
    return _point_list
