"""
Module: mcgs3_temporal_slice

This module contains the implementation of the mcgs3_grain_structure class, 
which is used for analyzing and processing grain structures in 3D space. It 
includes various methods for handling grain boundary points, grain positions, 
and other related properties.

Classes:
- mcgs3_grain_structure: Class for handling and analyzing grain structures in 
3D space.

Class: mcgs3_grain_structure
Prominant attributes:
- dim (int): Dimensionality of the grain structure.
- uigrid (UPXO object): User input grid requirements.
- uimesh (UPXO object): User input mesh requirements.
- vox_size: Voxel size.
- m (int): Monte-Carlo step or temporal slice.
- s (np.ndarray): State matrix, output of Monte-Carlo (MC) simulation.
- S: Total number of states considered in MC simulation.
- n (int): Number of grains in the grain structure.
- lgi (np.ndarray): Local Grain ID of every voxel in the grain structure.
- gid (np.ndarray): Grain ID.
- g (UPXO object): Individual grain objects.
- gb (UPXO object): Individual grain boundary objects.
- s_gid (dict): State value partitioning of gid.
- gid_s (np.ndarray): gid value based partitioning of state values.
- s_n (list): State value based partitioning of number of grains.
- neigh_gid (dict): Immediate neighbour information of every grain.
- positions (dict): Position name partitioned gids.
- grain_locs (dict): gid partitioned global coordinates of all voxels.
- gpos (dict): Position name partitioned gids.
- spbound (dict): Spatial bounds of all grains.
- spboundex (dict): Extended spatial bounds of all grains.
- sssr: Surface-sub-surface relationships.
- mprop (dict): Morphological properties.
- pvgrid: Py-Vista grid object
- lgi_slice: lgi slice
- prop_flag: Flag for morphological properties
- prop: Morphological properties
- domain_volume: Volume of the domain
- xax: x-axis
- yax: y-axis
- zax: z-axis
- axlim: Axis limits

Methods:
- __init__: Initializes the mcgs3_grain_structure object. (Special Method)
- __iter__: Returns an iterator object. (Special Method)
- __repr__: Returns a string representation of the object. (Special Method)
- __next__: Returns the next grain in the iteration. (Special Method)
- __att__: Returns the attribute handler. (Special Method)

- by_data: Class method to instantiate a temporal slice using a 3D Monte-Carlo state value array.

- calc_num_grains: Calculates the total number of grains in the grain structure.
- char_morphology_of_grains: Characterizes the 3D morphology of the grain structure.
- char_lgi_slice_morpho: Characterize morphology of a 2D slice of self.lgi.
- clean_gs_GMD_by_source_erosion_v1: Clean the gs using grain merger by dissolution by source grain erosion.
- clean_gs_GMD_by_source_erosion_v2: Clean the gs using grain merger by dissolution by source grain erosion.
- create_neigh_gid_pair_ids: Create neighbor grain ID pair IDs
- copy_lgi_1: Copy local grain ID 1

- export_vtk3d: Export data to .vtk format.
- extract_subdomains_random: Extract subdomains random

- find_grains: Detects grains in the local grain ID array.
- find_neigh_gid: Sets the neighbouring grain IDs for all grains.
- find_spatial_bounds_of_grains: Finds the spatial bounds of each grain in the grain structure.
- find_bounding_cube_gid: Finds the subset of the local grain ID array that tightly binds a grain.
- find_exbounding_cube_gid: Finds the subset of the local grain ID array that loosely binds a grain.
- find_grain_voxel_locs: Finds the voxel locations of grains in the local grain ID array.
- find_scalar_array_in_plane: Get the scalar values array in a plane.
- fit_ellipsoids: Fit ellipsoids to all grains in the grain structure.
- find_gid_pair_gbp_IDs: Find the gbp coords at the interface of gidl and gidr.
- find_twin_hosts: Find twin hosts

- get_vox_size: Returns the size of the voxel.
- get_binaryStructure3D: Returns the binary structure type for grain identification.
- get_upto_nth_order_neighbors: Calculates up to nth order neighbours for a given grain ID.
- get_nth_order_neighbors: Calculates only the nth order neighbours for a given grain ID.
- get_upto_nth_order_neighbors_all_grains: Calculates up to nth order neighbours for all grains.
- get_nth_order_neighbors_all_grains: Calculates only the nth order neighbours for all grains.
- get_upto_nth_order_neighbors_all_grains_prob: Calculates up to nth order neighbours for all grains with a probability.
- get_bounding_cube_all: Finds the subsets of the local grain ID array that tightly bind all grains.
- get_exbounding_cube_all: Finds the subsets of the local grain ID array that loosely bind all grains.
- get_scalar_field: Returns the requested scalar field.
- get_scalar_field_slice: Gets scalar field values along the specified slice.
- get_scalar_array_in_plane_unique: Find unique gids in a plane defined by origin and normal.
- get_bbox_diagonal_vectors: Find the vector representing doiagonal of the bounding box.
- get_voxel_volume: Return voxel volume from pvgrid data.
- get_voxel_surfareas: Return voxel surface area from pvgrid data.
- generate_bresenham_line_3d: Generate Bresenham line in 3d between two coordinate locations.
- get_values_along_line: Get values in 3D array along line between loci and locj points.
- get_igs_properties_along_line: Measure intercept properties along line b/w two specified locations.
- get_igs_along_line: Measure intercept properties along line b/w two specified locations.
- get_opposing_points_on_gs_bound_planes: Get points on the opposing boundaries of the grain structure.
- get_igs_along_lines: Measure intercept properties along lines defined by location sets i, j.
- get_igs_along_lines_multiple_samples: Measure intercept properties along lines defined by location sets i, j for multiple samples.
- get_bbox_aspect_ratio: Get aspect ratio of bounding box
- get_bbox_volume: Get volume of bounding box
- get_volnv_gids: Get volume by number of voxels for gids
- get_lgi_subset_around_location: Get lgi subset around location
- get_neigh_grains_next_to_location: Get neighbor grains next to location
- get_local_global_coord_offset: Get local global coordinate offset
- get_cutoff_twvol: Get cutoff twin volume
- get_k_nearest_coords_from_tree: Get k nearest coordinates from tree
- get_points_in_feature_coord: Get points in feature coordinate
- get_gs_instance_pvgrid: Get grain structure instance PyVista grid

- import_ctf: Import a ctf file
- import_crc: Import a crc file
- import_dream3d: Import a dream3d file
- import_vtk: Import a vtk file
- instantiate_twins: Instantiate twins
- initiate_gbp: Initiate grain boundary points
- identify_twins_gid: Identify twins for grain ID
- identify_twins: Identify twins
- igs_sed_ratio: Calculate the ratio of intercept grain size to sphere eq. diameter.

- make_pvgrid: Creates a PyVista grid of the local grain ID array.
- make_zero_non_gids_in_lgi: Returns a grain ID masked copy of the local grain ID array.
- make_zero_non_gids_in_lgisubset: Returns a grain ID masked copy of a subset of the local grain ID array.
- mesh: Mesh the grain structure
- merge_two_neigh_grains: Merges one grain into another if they are neighbours.
- mask_lgi_with_gids: Masks the local grain ID array against user input grain indices.
- mask_s_with_gids: Masks the state matrix against user input grain indices.
- mask_fid: Mask feature ID
- mask_fid_and_make_pvgrid: Mask feature ID and make PyVista grid
- mask_fid_and_plot: Mask feature ID and plot
- smoothen_sds: Smoothen subdomains
- _merge_two_grains_: Low-level merge operation for two grains.

- plot_mprop_correlations: Plots the correlations between morphological properties.
- plot_gs_pvvox: Plots the grain structure as PyVista voxels.
- plot_gs_pvvox_subset: Plots a subset of the voxelated grain structure in PyVista.
- plot_gbpoint_cloud_global: Plots all the grain boundary point clouds.
- plot_scalar_field_slice_orthogonal: Plots the scalar field along three fundamental orthogonal planes.
- plot_scalar_field_slice: Plots the scalar field along the specified slice plane.
- plot_scalar_field_slices: Plots the scalar field along multiple parallel slice planes.
- plot_largest_grain: Plots the largest grain in a temporal slice.
- plot_longest_grain: Plots the longest grain in a temporal slice.
- plot_grains: Plots grains given some grain IDs.
- plot_grain_sets: Plot multiple prominant and non-prominant grains.
- plot_gids_along_plane: Plot grains which fall alomng a plane.
- plot_single_voxel_grains: Plot single voxel grains
- plot_gs_instance: Plot grain structure instance

- recalculate_ngrains_post_grain_merge: Recalculates the number of grains after a grain merger.
- renumber_gid_post_grain_merge: Renumbers the grain IDs after a grain merger.
- renumber_lgi_post_grain_merge: Renumbers the local grain ID array after a grain merger.
- remove_overlaps_in_twins: Remove overlaps in twins
- reset_slice_lgi: Identify and labels grains in a 3D grain structure's 2D slice.
- reset_fdb: Reset feature data base

- sep_gbzcore_from_bbgidmask: Seperate grain boundary zone core from bounding box grain ID mask
- set__s_n: Sets the number of grains in each state.
- set__s_gid: Sets the grain IDs for each state.
- set__gid_s: Sets the state values for each grain ID.
- set__spart_flag: Sets the state partitioning flags.
- set_binaryStructure3D: Sets the binary structure type for grain identification.
- set_skimrp: Sets the region properties of the scikit image.
- set_mprops: Sets the morphological properties of the grain structure.
- set_binaryStructure3D: Sets the binary structure type for grain identification.
- set_gid: Sets the grain IDs.
- set_gbpoints_global_point_cloud: Sets a PyVista PolyData object with global grain boundary points.
- set_mprop_volnv: Calculate the volume by number of voxels.
- set_mprop_volnv_old: Calculate the volume by number of voxels. TO BE NUMBAfied
- set_mprop_pernv: Calculate the total perimeter of the grain by number of voxels.
- set_mprop_eqdia: Calculate equivalent sphere diameter.
- set_mprop_solidity: Set solidity morphological property of all 3D grains.
- set_mprop_arbbox: Calculate aspect ratio of bounding box.
- set_mprop_arellfit: Calculate aspect ratio of grain using ellipsoidal fit.
- set_mprop_sol: Calculate solidity of grains.
- set_mprop_ecc: Calculate eccentricity of grains.
- set_mprop_com: Calculate compactness of grains.
- set_mprop_sph: Calculate sphericity of grains.
- set_mprop_fn: Calculate flatness of grains.
- set_mprop_rnd: Calculate roundness of grains.
- set_mprop_fdim: Calculate fractal dimension of grains.
- set_Lgbp_gid: Set local grain boundary points for a grain ID
- set_Lgbp_all: Set local grain boundary points for all grains
- set_gid_pair_gbp_IDs: Set grain ID pair grain boundary point IDs
- set_neigh_gid_interaction_pairs: Set neighbor grain ID interaction pairs
- setup_gid_pair_gbp_IDs_DS: Setup grain ID pair grain boundary point IDs data structure
- setup_gid_set__gbsegs: Setup grain ID set grain boundary segments
- setup_gid_twin: Setup grain ID twin
- setup_for_twins: Setup for twins
- set_mprop_sanv: Set morphological property surface area by number of voxels
- set_mprop_rat_sanv_volnv: Set morphological property ratio of surface area by number of voxels to volume by number of voxels
- set_grain_positions: Set positions of grains relative to grain structure boundaries.
- set_gid_imap_keys: Assign inverse mapping keys to grains based on relative positions.
- assign_gid_imap_keys: Assign inverse mapping keys to grains based on relative positions.

- sss_rel_morpho: Carry out surface -- sub-surface relationship study.
- sss_rel_morpho_multiple: Carry out surface -- sub-surface relationship study on multiple planes.

- update_dream3d_ABQ_file: Update Dream3D Abaqus file
- validate_scalar_field_name: Validates if a scalar field name is valid.
- viz_browse_grains: Browse grains in the grain structrure using a slider.
- viz_clip_plane: Visualize grain structure along a clip plane.
- viz_mesh_slice: Visualize grain structure along a slice plane.
- viz_mesh_slice_ortho: Viz. grain str. along three fundamental mutually orthogonal planes.

- check_for_neigh: Checks if a grain is a neighbour of another grain.
- create_neigh_gid_pair_ids: Create neighbor grain ID pair IDs
- deform_ortho: Deform orthogonal
- globalise_gbp: Globalise grain boundary points
- is_gid_pair_in_lr_or_rl: Check if grain ID pair is in left-right or right-left configuration
- offset_local_to_global: Offset local to global
- perform_post_grain_merge_ops: Performs necessary operations after a grain merger.
- build_gbp_stack: Build grain boundary point stack
- build_gbpids: Build grain boundary point IDs
- build_gbp: Build grain boundary points
- build_gbp_id_mappings: Build grain boundary point ID mappings
- find_gbsp: Find grain boundary surface points
- build_gid__gid_pair_IDs: Build grain ID to grain pair IDs mapping
- _check_lgi_dtype_uint8: Validates and modifies the local grain ID array data type.
- _compute_volumes_with_bincount: Calculate the volume by number of voxels using Numba and bincount.

Property definitions:
- get_vox_size: Returns the size of voxel.
- nvoxels: Volume by number of voxels
- nvoxels_values: Volume by number of voxels values
- single_voxel_grains: Single voxel grains
- smallest_volume: Smallest volume
- largest_volume: Largest volume
"""

import os
import random
import time
import matplotlib as mpl
from copy import deepcopy
from typing import Iterable
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
# import cv2
import vtk
import warnings
from random import Random
from math import floor
# import vedo as vd
import pyvista as pv



from numba import njit




from scipy.spatial import cKDTree
from itertools import permutations, product, combinations
from scipy.ndimage import zoom
# from skimage.measure import label as skim_label
import seaborn as sns
import collections
# import tqdm
from functools import partial
from collections import defaultdict
from collections.abc import Mapping, Iterable
from matplotlib.figure import Figure
from skimage.segmentation import find_boundaries
from upxo.geoEntities.plane import Plane
from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
# from upxo._sup.console_formats import print_incrementally
from upxo._sup import dataTypeHandlers as dth
from upxo._sup.gops import att
from upxo._sup.data_templates import dict_templates
import networkx
from upxo.netops.kmake import make_gid_net_from_neighlist
# from scipy.ndimage import label
from upxo.misc import make_belief
from scipy.ndimage import generate_binary_structure
from dataclasses import dataclass
from scipy.ndimage import label as spndimg_label
import upxo._sup.data_ops as DO
from upxo.viz.plot_artefacts import cuboid_data
from upxo.viz.helpers import arrange_subplots
from numba.typed import Dict, List
from numba.types import int32, ListType
from scipy.spatial.transform import Rotation

warnings.simplefilter('ignore', DeprecationWarning)

NPA = np.array
_npla = np.logical_and
_npaw = np.argwhere




class mcgs3_grain_structure():
    """
    Nomenclature
    ------------
    id: ID number
    gid: Grain ID
    gb: Grain boundary
    gbp: Grain boundary points
    gpos: Grain position
    imap: Inverse map
    Lgbp_all: All the local grain boundary points
    Ggbp_all: All the globalised grain boundary points

    Parameters
    ----------
    dim: Dimensionality. Type: int
    uigrid: user input grid requirements. Type: UPXO obj
    uimesh: user input mesh requirements. Type: UPXO obj
    vox_size: voxel size.
    m: Monte-Carlo step. Also called tslice, temporal slice. Type: int
    s: State matrix, output of Monte-Carlo (MC) simulation. Type. np.ndarray
    S: Total number of states considered in MC simulation.
    n: Number of grains in the grain structure. Type: int
    lgi: Local Grain ID of every voxel in the grain strycture. Type: np.ndarray
    gid: grain ID. Type: np.ndarray
    g: individual grain objects. Type: UPXO obj
    gb: individual grain boundary objects. Type: UPXO obj
    s_gid: State value partitioning of gid. Type: dict
    gid_s: gid value based partitioning of state values. Type: np.ndarray
    s_n: state value based partitioning of number of grains. Type: list

    neigh_gid: Immediate neighbour information of every grain. Type: dict
    positions: position name partitined gids. Type: dict. to be deprecated
    grain_locs: gid partitined global coordinates of all voxels. Type: dict
    gpos: position name partitined gids. Type: dict
    spbound: spatial bounds of all grains. Type: dict
    spboundex: extended spatial bounds of all grains. Type: dict
    Ggbp_all: gid paritiotned global grain boundary point coords. Type: dict
    gbpstack: Global stack of all grain boundary points. Type: np.ndarray
    gbpids: Global stack of all grain boundary point IDs. Type: np.ndarray
    gbp_id_maps: Map from gbpstack into gbpids. Type: dict
    gbp_ids: gid partitioned gbpids. Type: dict
    gid_pair_ids: Every immediate neighbour pair ID and gids. Type: dict
    gid_pair_ids_rev: Reverse mapping of gid_pair_ids. Type: dict
    gid_pair_ids_unique_lr: Unique left-right gid neigh pairs. Type: np.ndarray
    gid_pair_ids_unique_rl: Unique right-left gid neigh pairs. Type: np.ndarray
    gbsurf_pids_vox: grain boundary surface voxel IDs. Type: dict.

    gid_imap_keys: @dev only.
    gid_imap: @dev only.
    Lgbp_all: @ dev only.

    mp: UPXO multi-point object template. Type: UPXO obj
    binaryStructure3D: structure used in grain identification. Type. np.ndarray
    spart_flag: State value partitioning flags for grains. Type: np.ndarray

    sssr: surface-sub-surface relationships

    mprop: morphhological properties. Type: dict. mprops could have the
    fo9llowing keys:
        volnv: Volume by number of voxels
        volsr: Volume after grain boundary surface reconstruction
        volch: Volume of convex hull

        sanv: surface area by number of voxels
        savi: surface area by voxel interfaces
        sasr: surface area after grain boundary surface reconstruction
        psa: projected surface area

        pernv: perimeter by number of voxels
        pervl: perimeter by voxel edge lines
        pergl: perimeter by geometric grain boundary line segments

        eqdia: eqvivalent diameter
        feqdia: Feret eqvivalent diameter

        kx: grain boundary voxel local curvature in yz plane
        ky: grain boundary voxel local curvature in xz plane
        kz: grain boundary voxel local curvature in xy plane
        kxyz: mean(kx, ky, kz)
        ksr: k computed from surface reconstruction.

        arbbox: aspect ratio by bounding box
        arellfit: aspect ratio by ellipsoidal fit

        sol: solidity
        ecc: eccentricity - how much the shape of the grain differs from
            a sphere.
        com: compactness
        sph: sphericity
        fn: flatness
        rnd: roundness
        mi: moment of inertia tensor

        fdim: fractal dimension

        rat_sanv_volnv: Ratio of sanv to volnv
    """

    __slots__ = ('dim', 'uigrid', 'uimesh', 'm', 's', 'S', 'ndimg_label_pck',
                 'binaryStructure3D', 'n', 'lgi', 'fdb',
                 'xax', 'yax', 'zax', 'vox', 'axlim',
                 '_ckdtree_', '_upxo_mp3d_', 'domain_volume',
                 'spart_flag', 'gid', 's_gid', 'gid_s', 's_n', 'g', 'gb',
                 'positions', 'mp', 'vox_size', 'gid_twin',
                 'prop_flag', 'prop', 'are_properties_available', 'prop_stat',
                 '__gi__', '__ui', 'info',
                 'pvgrid', 'ellfits', 'skimrp', 'sssr',
                 'valid_scalar_fields', 'pointclouds_pv', 'mprop', 'lgi_slice',
                 'grain_locs', 'feat_locs',
                 'gpos', 'spbound', 'spboundex', 'gid_imap_keys',
                 'gid_imap', 'neigh_gid', 'no_gid', 'noth_gid',
                 'Lgbp_all', 'Ggbp_all',
                 'gbpstack', 'gbpids', 'gbp_id_maps', 'gbp_ids',
                 'gid_pair_ids', 'gid_pair_ids_rev',
                 'gid_pair_ids_unique_lr', 'gid_pair_ids_unique_rl',
                 'gbsurf_pids_vox', 'gid_pair_gbp_IDs', 'gid_pair_gbp_coords',
                 'gid_gpid', 'triples', 'ctrls', 'tc_info', 'cluster_sets')
    EPS, __maxGridSizeToIgnoreStoringGrids = 1e-1, 200**3
    CUBIC_SYMM_OPS = None
    _vtk_ievnt_ = vtk.vtkCommand.InteractionEvent
    _mprop3d2d_ = {'eqdia': ('eqdia'),
                   'feqdia': ('feqdia'),
                   'arbbox': ('arbbox', 'arellfit'),
                   'arellfit': ('arbbox', 'arellfit'),
                   'psa': ('area'),
                   'solidity': ('solidity', 'sol'),
                   'sol': ('solidity', 'sol'),
                   'sphericity': ('circularity', 'circ'),
                   'sph': ('circularity', 'circ'),
                   'igs': ('igs'),
                   'fdim': ('fdim', 'fd')
                   }
    fcc_tc = {"copper": (90.0, 35.0, 45.0),  # {112}<111> # Rolling texture
              "brass": (35.0, 45.0, 0.0),  # {110}<112> # Rolling texture
              "s": (59.0, 37.0, 63.0),  # {123}<634> # Rolling texture
              "goss": (90.0, 90.0, 45.0),  # {110}<001> # Rolling texture
              "cube": (0.0, 0.0, 0.0),  # {001}<100> # Annealing / RX
              "rotated_cube": (45.0, 0.0, 0.0),  # {001}<110> # Annealing / RX
              "P": (90.0, 45.0, 0.0),   # {011}<122> # Annealing / RX
              "A1": (35.0, 45.0, 90.0),  # {111}<110> # Shear texture
              "A2": (55.0, 90.0, 45.0),  # {111}<112> # Shear texture
              "B": (45.0, 90.0, 45.0),  # {112}<110> # Shear texture
              "C": (0.0, 90.0, 45.0),  # {001}<110> # Shear texture
              "Q": (35.0, 55.0, 45.0),  # {013}<231> # Minor / transitional
              "D": (59.0, 37.0, 26.0),  # {4411}<1118> approx. # Minor / transitional
              }

    def __init__(self, dim=3, m=None, uidata=None, vox_size=None, S_total=None,
                 uigrid=None, uimesh=None, ndimg_label_pck=None,
                 iroute='regular',
                 udata=None, udata_name='s'):
        # Dictionary to contain controls.
        self.ctrls = {}
        self.ctrls['iroute'] = iroute
        self.ctrls['udata_name'] = udata_name
        self.ctrls['numba_activation_nvox_threshold'] = 75*75*75
        # Package to label the 3D image.
        self.ndimg_label_pck = ndimg_label_pck
        # Dimensionality
        self.dim = dim
        # MOnte-Carlo temporal slixce number
        self.m = m
        # User input grid data
        self.uigrid = uigrid
        # User input mesh control data
        self.uimesh = uimesh
        # Voxel size
        self.vox_size = vox_size
        # grains dictionary -> grain id: grain object
        self.g = {}
        # gb dictionary -> gb id: gb object or gb coordinates
        self.gb = {}
        self.info = {}
        # surface-sub-surface relationships
        self.sssr = {}
        # Feature database -> feature name: feature data
        self.fdb = {}
        # Grain positions ->
        self.gpos = {}
        # Grain locatiopns ->
        self.grain_locs = {}
        # Py-Vista point clouds: USE TO BE DEPRECATED
        self.pointclouds_pv = {}
        # Coordinate tree generator
        self._ckdtree_ = cKDTree
        # UPXO 3D Multi-Point object: USE TO BE DEPRECATED
        self._upxo_mp3d_ = mp3d
        # Check and provide a descripotiopn
        self.mp = dict_templates.mulpnt_gs3d
        # Flag to indicate if morpho prop. are available. TO BE DEPRECATED
        self.are_properties_available = False
        # Py-Vista grid object
        self.pvgrid = None
        # Ellipical fits. Used by fit_ellipsoids function.
        self.ellfits = None
        # Grain ID: Twin data dictionary
        self.gid_twin = None
        # Specify valid scalar data fields
        self.valid_scalar_fields = ["lgi", "s", "nneigh", "fid1", "fid2",
                                    "fid3", "fid4", "fid5"
                                    ]
        # Sci-kit image region propertioes object
        self.skimrp = None
        # Dictionary of morphological properties
        self.mprop = {'volnv': None,  # Volume by number of voxels
                      'volsr': None,  # Volume after gb surf reconstruction
                      'volch': None,  # Volume of convex hull
                      'sanv': None,  # surf area by number of voxels
                      'savi': None,  # surf area by voxel interfaces
                      'sasr': None,  # surf area after gb surf reconstruction
                      'psa': None,  # projected surface area
                      'pernv': None,  # perimeter by number of voxels
                      'pervl': None,  # perimeter by voxel edge lines
                      'pergl': None,  # perimeter by geom. gb line segments
                      'eqdia': None,  # eqvivalent diameter
                      'feqdia': None,  # Feret eqvivalent diameter
                      'kx': None,  # gb voxel local curvature in yz plane
                      'ky': None,  # gb voxel local curvature in xz plane
                      'kz': None,  # gb voxel local curvature in xy plane
                      'ksr': None,  # k computed from surf reconstruction.
                      'arbbox': None,  # aspect ratio by bounding box
                      'arellfit': None,  # aspect ratio by ellipsoidal fit
                      'sol': None,  # solidity of the grains
                      'ecc': None,  # eccentricity of the grains
                      'com': None,  # compactness of the grains
                      'sph': None,  # sphericity of the grains
                      'fn': None,  # flatness of the grains
                      'rnd': None,  # roundness of the grains
                      'mi': None,  # moment of inertia tensor
                      'fdim': None,  # fractal dimension
                      'rat_sanv_volnv': None,  # Ratio of sanv to volnv
                      }
        # Set up the physical domain properties like bounds and volume
        _uig_ = self.uigrid
        self.axlim = {'x': (_uig_.xmin, _uig_.xmax, _uig_.xinc),
                      'y': (_uig_.ymin, _uig_.ymax, _uig_.yinc),
                      'z': (_uig_.zmin, _uig_.zmax, _uig_.zinc)}
        self.xax = np.arange(*self.axlim['x'])
        self.yax = np.arange(*self.axlim['y'])
        self.zax = np.arange(*self.axlim['z'])
        self.domain_volume = self.xax.size * self.yax.size * self.zax.size

        self.neigh_gid = {}

        if iroute == 'direct' and udata_name in ('s', 'state'):
            self.s = udata
            self.S = S_total
            self.set__spart_flag(S_total)
            self.set__s_gid(S_total)
            self.set__gid_s()
            self.set__s_n(S_total)
            self.__setup__positions__()
            self.set_gid_imap_keys()

        if iroute == 'direct' and udata_name in ('lgi', 'fid'):
            # print(udata.max())
            
            self.lgi = deepcopy(udata)
            # print(self.lgi.max())
            self.gid = np.unique(self.lgi)
            self.n = self.gid.size

        if iroute == 'regular':
            self.S = S_total
            self.set__spart_flag(S_total)
            self.set__s_gid(S_total)
            self.set__gid_s()
            self.set__s_n(S_total)
            self.__setup__positions__()
            self.set_gid_imap_keys()

        self.tc_info = {}

    def __iter__(self):
        self.__gi__ = 1
        return self

    def __repr__(self):
        return f'UPXO. gs-tslice.3d. {id(self)}'

    def __next1__(self):
        if self.n:
            if self.__gi__ <= self.n:
                grain_pixel_indices = np.argwhere(self.lgi == self.__gi__)
                self.__gi__ += 1
                return grain_pixel_indices
            else:
                raise StopIteration

    def __next__(self):
        if self.n:
            if self.__gi__ <= self.n:
                thisgrain = self.g[self.__gi__]['grain']
                self.__gi__ += 1
                return thisgrain
            else:
                raise StopIteration

    def __att__(self):
        return att(self)

    @property
    def get_vox_size(self):
        """Return the size of voxel."""
        return self.vox_size

    @classmethod
    def by_parameters(cls, xmin=0.0, xinc=1.0, xmax=100.0,
                      ymin=0.0, yinc=1.0, ymax=100.0, zmin=0.0, zinc=1.0,
                      zmax=100.0, S_total=-1, MCsteps=10, alg='300b'):
        pass

    @classmethod
    def by_data(cls, sf_data, sf_name='s', dim=3, m=0,
                xmin=0.0, xinc=1.0, xmax=100.0,
                ymin=0.0, yinc=1.0, ymax=100.0, zmin=0.0, zinc=1.0,
                zmax=100.0, S_total=-1, nvoxels_max=1.01E9,
                reindex_labels=True):
        r"""
        Instantiate temporal slice using just 3D Monte-Carlo state value array.

        This allows you to exploit the entire module for the input s-array. You
        can start off another branch of simulations from this, enabling the
        following things:
            1. Switching across algorithms.
            2. Differently evolve a local subdomain and plug it back into the
               parent domain.

        Parameters
        ----------
        sf_data: numpy.ndarray
            3D grain structure image data.

        dim: int, optional
            Dimensionality of the problem. Defaults to 3.

        m: int, optional
            A user desired value of temporal slice number. This is to ensure
            we have a starting point for a new Monte-Carlo simulation to
            start off from this grain structure (as specified by data) as the
            starting point. Defaults to 0.

        xmin: float, optional
            Starting point of x-axis. Defaults to 0.0

        xinc: float, optional
            Increment step of x-axis. Defaults to 1.0

        xmax: float, optional
            Ending point of x-axis. Defaults to 100.0

        ymin: float, optional
            Starting point of y-axis. Defaults to 0.0

        yinc: float, optional
            Increment step of y-axis. Defaults to 1.0

        ymax: float, optional
            Ending point of y-axis. Defaults to 100.0

        zmin: float, optional
            Starting point of z-axis. Defaults to 0.0

        zinc: float, optional
            Increment step of z-axis. Defaults to 1.0

        zmax: float, optional
            Ending point of z-axis. Defaults to 100.0

        S_total: int, optional
            Total number of discrete MC state values. Defaults to -1.

        nvoxels_max: int, optional
            Maximum number of voxels. Defaults to 1.01E9

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs

        pxt = mcgs(input_dashboard='input_dashboard.xls')
        pxt.simulate(verbose=False)
        tslice = 10
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                           find_grain_voxel_locs=True,
                           find_spatial_bounds_of_grains=True,
                           force_compute=True)

        gstslice.set_mprops(volnv=True, eqdia=False,
                    eqdia_base_size_spec='volnv',
                    arbbox=False, arbbox_fmt='gid_dict',
                    arellfit=False, arellfit_metric='max',
                    arellfit_calculate_efits=False,
                    arellfit_efit_routine=1,
                    arellfit_efit_regularize_data=False,
                    solidity=False, sol_nan_treatment='replace',
                    sol_inf_treatment='replace',
                    sol_nan_replacement=-1, sol_inf_replacement=-1)

        # ------------------------------------------------------
        # Example - 1. Using MC state values as udata

        p, q, r = 5, 5, 10

        A = gstslice.extract_subdomains_random(p=p, q=q, r=r, n=2,
                               feature_name='s',
                               make_pvgrids=False
                               )

        from upxo.pxtal.mcgs3_temporal_slice import mcgs3_grain_structure

        sf_data = A['sd'][0]
        sd = mcgs3_grain_structure.by_data(sf_data, sf_name='s',
                           dim=3, m=tslice,
                           xmin=0.0, xinc=1.0, xmax=r,
                           ymin=0.0, yinc=1.0, ymax=q,
                           zmin=0.0, zinc=1.0, zmax=p,
                           S_total=gstslice.S,
                           nvoxels_max=1.01E9,)
        sd.char_morphology_of_grains(label_str_order=1,
                        find_grain_voxel_locs=True,
                        find_spatial_bounds_of_grains=True,
                        find_grain_locations=True,
                        find_neigh=[True, [1]],
                        force_compute=True)

        sd.lgi

        # ------------------------------------------------------
        # Example - 2. Using fid as udata

        p, q, r = 5, 5, 10

        sdraw = gstslice.extract_subdomains_random(p=p, q=q, r=r, n=5,
                               feature_name='base',
                               make_pvgrids=False)

        subdomains = []
        for udata in sdraw['sd']:
            sd = mcgs3_grain_structure.by_data(sf_data, sf_name='lgi',
                               dim=3, m=tslice,
                               xmin=0.0, xinc=1.0, xmax=r,
                               ymin=0.0, yinc=1.0, ymax=q,
                               zmin=0.0, zinc=1.0, zmax=p,
                               S_total=gstslice.S,
                               nvoxels_max=1.01E9,)
            subdomains.append(sd)
        # ------------------------------------------------------
        # Example - 3. Including twins

        p, q, r, n = 40, 40, 40, 1

        A = gstslice.extract_subdomains_random(p=p, q=q, r=r, n=n,
                               feature_name='s',
                               make_pvgrids=False
                               )

        from upxo.pxtal.mcgs3_temporal_slice import mcgs3_grain_structure

        udata = A['sd'][0]
        sd = mcgs3_grain_structure.by_data(sf_data, sf_name='s',
                           dim=3, m=tslice,
                           xmin=0.0, xinc=1.0, xmax=r,
                           ymin=0.0, yinc=1.0, ymax=q,
                           zmin=0.0, zinc=1.0, zmax=p,
                           S_total=gstslice.S,
                           nvoxels_max=1.01E9,)
        sd.char_morphology_of_grains(label_str_order=1,
                        find_grain_voxel_locs=True,
                        find_spatial_bounds_of_grains=True,
                        find_grain_locations=True,
                        find_neigh=[True, [1]],
                        force_compute=True)

        sd.set_mprops(volnv=True, eqdia=False,
                            eqdia_base_size_spec='volnv',
                            arbbox=False, arbbox_fmt='gid_dict',
                            arellfit=False, arellfit_metric='max',
                            arellfit_calculate_efits=False,
                            arellfit_efit_routine=1,
                            arellfit_efit_regularize_data=False,
                            solidity=False, sol_nan_treatment='replace',
                            sol_inf_treatment='replace',
                            sol_nan_replacement=-1, sol_inf_replacement=-1)

        mprops = {'volnv': {'use': True, 'reset': False,
                            'k': [.02, 1.0], 'min_vol': 4,},
                  'rat_sanv_volnv': {'use': True, 'reset': False,
                                     'k': [0.0, .8], 'sanv_N': 26},}
        twspec = {'n': [5, 10, 3],
                'tv': np.array([5, -3.5, 5]),
                'dlk': np.array([1.0, -1.0, 1.0]),
                'dnw': np.array([0.5, 0.5, 0.5]),
                'dno': np.array([0.5, 0.5, 0.5]),
                'tdis': 'normal',
                'tpar': {'loc': 1.12, 'scale': 0.1, 'val': 1},
                'vf': [0.05, 1.00], 'sep_bzcz': False}
        twgenspec = {'seedsel': 'random_gb', 'K': 20,
                   'bidir_tp': False, 'checks': [True, True],}

        sd.instantiate_twins(ninstances=4,
                                   base_gs_name_prefix='twin.',
                                   twin_setup={'nprops': 2, 'mprops': mprops},
                                   twspec=twspec,
                                   twgenspec=twgenspec,
                                   reset_fdb=True, )

        sd.mask_fid_and_plot(feature='twins',
                                   instance_names=sd.fdb.keys(),
                                   fid_mask_value=-32,
                                   non_fid_mask=True,
                                   non_fid_mask_value=-31,
                                   write_to_disk=False,
                                   write_sparse=True,
                                   throw=True,
                                   cmap_specs=(['white', 'yellow', 'grey', 'red'], 2),
                                   show_edges=False,
                                   opacity=1.0, rmax_sp=2, cmax_sp=2,
                                   thresholding=False,
                                   threshold_value=-32)
        """
        uigrid = make_belief.uigrid(dim=dim,
                                    npixels_max=nvoxels_max,
                                    xmin=xmin, xinc=xinc, xmax=xmax,
                                    ymin=ymin, yinc=yinc, ymax=ymax,
                                    zmin=zmin, zinc=zinc, zmax=zmax)

        if sf_name in ('base', 'lgi'):
            if reindex_labels:
                sf_data = cls.reindex_labels('', sf_data)

        return cls(dim=dim,
                   m=m,
                   uidata=None,
                   vox_size=(xinc, yinc, zinc),
                   S_total=S_total,
                   uigrid=uigrid,
                   uimesh=None,
                   ndimg_label_pck=spndimg_label,
                   iroute='direct',
                   udata=sf_data,
                   udata_name=sf_name)

    def reindex_labels(self, udata):
        """
        Reindex the labels in the input 3D numpy array such that they are
        consecutive integers starting from 1.

        Parameters
        ----------
        udata : numpy.ndarray
            A 3D numpy array of integers.

        Returns
        -------
        numpy.ndarray
            A 3D numpy array with reindexed labels.
        """
        unique_labels = np.unique(udata)
        # Remove 0 from unique labels if it exists
        unique_labels = unique_labels[unique_labels != 0]
        # Create a mapping from old labels to new labels
        label_map = {old_label: new_label
                     for new_label, old_label in enumerate(unique_labels,
                                                           start=1)}
        # Vectorize the mapping function
        vectorized_map = np.vectorize(lambda x: label_map.get(x, 0))
        # Apply the mapping to the input array
        remapped_data = vectorized_map(udata)
        return remapped_data

    @property
    def lfi(self):
        return self.lgi

    def set__s_n(self,
                 S_total,
                 ):
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

    def set__s_gid(self, S_total,):
        """
        Sets the `s_gid` attribute with a dictionary where keys are integers from 1 to `S_total` and values are None.

        Parameters:
        -----------
        S_total : int
            The total number of elements to include in the `s_gid` dictionary. Must be an integer.

        Raises:
        -------
        ValueError
            If `S_total` is not an integer.
        """

        if not isinstance(S_total, int):
            raise ValueError("S_total must be an integer.")
        self.s_gid = {s: None for s in range(1, S_total+1)}

    def set__gid_s(self):
        self.gid_s = []

    def set__spart_flag(self, S_total,):
        if not isinstance(S_total, int):
            raise ValueError("S_total must be an integer.")
        self.spart_flag = {_s_: False for _s_ in range(1, S_total+1)}

    def get_binaryStructure3D(self):
        """Return the 3D binary structure."""
        return self.binaryStructure3D

    def set_binaryStructure3D(self, n=3):
        """Set value of the binary structure type for grain identification."""
        if n in (1, 2, 3):
            self.binaryStructure3D = generate_binary_structure(3, n)
        else:
             print('Invalid binary structure-3D. n must be in (1, 2, 3). Value not set')

    def set_pxtal(self, bea=None):
        """
        bea: Bunge's Euler angle
        """
        if len(self.neigh_gid) == 0:
            raise ValueError('O(1) neigh data structure does not exist.')
        if type(bea) != np.ndarray:
            raise ValueError('bea is not a numpy array.')
        if bea.shape[1] == 3:
            raise ValueError('bea has improper shape.')
        # -----------------------------------------
        norithresh = np.round(0.60*len(self.n)).astype(np.int64)
        if bea.shape[0] < norithresh:
            raise ValueError('Insuffiient input orientations in bea.',
                             f'Minimum needed: {norithresh} orientations.',
                             f'You provided: {bea.shape[0]}')
        # -----------------------------------------
        from orix.quaternion import Orientation
        from orix.symmetry import Cubic

    @staticmethod
    def reindex_array_by_cmp_old_new_images(old_image, new_image, old_arrays):
        """
        Re-indexes lists of old IDs based on the mapping between an old and new image.
    
        Args:
            old_image (np.ndarray): The 3D array with original, non-contiguous integer IDs.
            new_image (np.ndarray): The 3D array with new, re-indexed, contiguous integer IDs.
            old_arrays (list[np.ndarray]): A list of 1D arrays containing subsets of the old IDs.
    
        Returns:
            list[np.ndarray]: A list of new 1D arrays with the IDs re-indexed according to the new_image.
        """
        # 1. Create the mapping from old IDs to new IDs efficiently.
        # We find the unique pairs of (old_id, new_id) across the entire dataset.
        unique_pairs = np.unique(np.vstack((old_image.ravel(), new_image.ravel())).T, axis=0)
    
        # 2. Convert the unique pairs into a fast lookup dictionary.
        old_to_new_map = {old_id: new_id for old_id, new_id in unique_pairs}
    
        # 3. Apply the mapping to each of the old ID arrays.
        new_arrays = []
        for old_array in old_arrays:
            # Use the map to translate each ID. .get() is used for safety in case
            # an ID in the list somehow doesn't appear in the image.
            new_array = np.array([old_to_new_map.get(old_id) for old_id in old_array],
                                 dtype=np.int32)
            new_arrays.append(new_array)
    
        return new_arrays

    def char_morphology_of_grains(self,
                                  sf_name='s',
                                  label_str_order=1,
                                  ngrains_max=1E3,
                                  make_pvgrid=True,
                                  find_neigh=[False, [1], False, '1-no'],
                                  find_grain_voxel_locs=False,
                                  find_spatial_bounds_of_grains=False,
                                  find_grain_locations=False,
                                  force_compute=False,
                                  extra_sf={'sfname1': None,
                                            'sfname2': None,},
                                  fdb_details={'name': None},
                                  set_mprops=True,
                                  mprops_kwargs={'set_skimrp': False,
                                                 'volnv': True,
                                                 'eqdia': False,
                                                 'eqdia_base_size_spec': 'volnv',
                                                 'arbbox': False,
                                                 'arbbox_fmt': 'gid_dict',
                                                 'arellfit': False,
                                                 'arellfit_metric': 'max',
                                                 'arellfit_calculate_efits': True,
                                                 'arellfit_efit_routine': 1,
                                                 'arellfit_efit_regularize_data': True,
                                                 'solidity': False,
                                                 'sol_nan_treatment': 'replace',
                                                 'sol_inf_treatment': 'replace',
                                                 'sol_nan_replacement': -1,
                                                 'sol_inf_replacement': -1,
                                                 'sanv': False,
                                                 'sanv_N': 26,
                                                 'rat_sanv_volnv': False,
                                                 'sanv_verbosity': 1E2,
                                                 'disp_msg': ''}
                                  ):
        """
        Characterize the 3D morphology of the grain structure.

        Parameters
        ----------
        sf_name: str, optional
            Provide which UPXO data is to be characterised. Optiona include s,
            lgi, fdb.
            Defaults to a value of 's'.

        label_str_order: int, optional
            Provide the voxel connectivity order used for grain identification.
            Defaults to a value of 1.

        ngrains_max: int, optional
            Maximum number of grains stop calculating 1E3,

        make_pvgrid : bool, optional
            Flag to create PyVista grid. Defaults to True.

        find_neigh : list, optional
            List containing flag and a list of niehghbour orders needed. Its
            details arebelow:
                find_neigh[0]: Flag to create neigh grain data-structures.
                find_neigh[1]: List of numbers representing the neigh
                    order values needed. The default self.neigh variable shall
                    contain neighbouring grain informatoipm only for order 1.
                    Other order neighbouir grain datas will be contained in a
                    different dictionary.
                find_neigh[2]: Flag to include gid as self-neighbour inside
                    neighbour list.
                find_neigh[3]: str. This points to whether we need to calculate
                    upto no neighbout order or only the no^th neighbour ourder.
            Defaults to [False, [1]].

        find_grain_voxel_locs : bool, optional
            Flag to find the grain voxel masks in lgi. Find if True, ignore if
            False. Defaults to False.

        find_spatial_bounds_of_grains : bool, optional
            Flag to calculate the spatial bounds of each grain in lgi. Find if
            True and ignore if False. Defaults to False.

        force_compute : bool, optional
            Flag to ignore ngrains_max. Morohological proprties requexted to be
            caluclated will all be calculated even when ngrains_max is not
            satisfied. Defaults to False.

        Notes
        -----
        It is recommended that the label_str_order value be 3 to reduce the
        number of grains which are morphologically diffcult to mesh and would
        require complex grain boundary surface cleaning operations.

        A label_str_order value of 3 results in an increased count of single
        voxel grains in the grain structure.

        The label_str_order of 3 does not completely eliminate the presence of
        difficult types of grain boundarty surface edge connection and surface
        morphologies but certainly leads to a lesser count of such geometries.

        Function order
        --------------
        Secondary. Calls a number of other primary functions.

        Functionality order
        -------------------
        Secondary. Provcided the availableity of the grain structure labels,
        the user can use their own pipelines for grain structure
        characterization, cleaning, meshing and exports. Nevertheless this is a
        useful function to have in the core UPXO.
        """
        # ----------------------------- # Validations
        if sf_name.lower() not in ('s', 'lgi', 'fdb'):
            raise ValueError("Invalid sf_name. Must be 's', 'lgi' or 'fdb'")
        if label_str_order not in (1, 2, 3):
            raise ValueError('Invalid label_str_order. Must be 1, 2, or 3.')
        if not isinstance(make_pvgrid, bool):
            raise TypeError('make_pvgrid must be a boolean.')
        if not isinstance(find_neigh, list) or len(find_neigh) != 4:
            raise ValueError('find_neigh must be a list of length 4.')
        if not isinstance(find_neigh[0], bool):
            raise TypeError('find_neigh[0] must be a boolean.')
        if not isinstance(find_neigh[1], list):
            raise TypeError('find_neigh[1] must be a list.')
        if not isinstance(find_neigh[2], bool):
            raise TypeError('find_neigh[2] must be a boolean.')
        if not isinstance(find_grain_voxel_locs, bool):
            raise TypeError('find_grain_voxel_locs must be a boolean.')
        if not isinstance(find_spatial_bounds_of_grains, bool):
            raise TypeError('find_spatial_bounds_of_grains must be a boolean.')
        if not isinstance(find_grain_locations, bool):
            raise TypeError('find_grain_locations must be a boolean.')
        if not isinstance(force_compute, bool):
            raise TypeError('force_compute must be a boolean.')
        if not isinstance(extra_sf, dict):
            raise TypeError('extra_sf must be a dictionary.')
        # -----------------------------
        if sf_name.lower() in ('s'):
            self.find_grains(label_str_order=label_str_order,
                             pck=self.ndimg_label_pck)
        elif sf_name.lower() in ('lgi'):
            self.find_grains_lgi()
        elif sf_name.lower() in ('fdb'):
            fdb_name = fdb_details['name']
            # REMAINING CODES
            pass
        # -----------------------------
        if any((self.n < ngrains_max, force_compute)):
            if make_pvgrid:
                self.make_pvgrid()
                self.add_scalar_field_to_pvgrid(sf_name="lgi",
                                                sf_value=self.lgi)
                extra_sf_names = list(extra_sf.keys())
                extra_sf_vals = list(extra_sf.values())
                extras = np.argwhere(extra_sf_vals).squeeze()
                if len(extra_sf_names) < len(set(extra_sf_names)):
                    raise ValueError('Duplicate sf names. Invalid input.')
                for i in extras:
                     # VALIDATIONS
                     self.add_scalar_field_to_pvgrid(sf_name=extra_sf_names[i],
                                                     sf_value=extra_sf_vals[i])
            # -----------------
            if find_neigh[0]:
                self.find_neigh_gid()
                '''Call other fnction to calculate other neighbouring grain
                data.'''
                no = find_neigh[1]
                print(40*'-')
                # print(no)
                if 1 in no:
                    no.remove(1)
                # print(no)
                if len(no) > 0:
                    fx = self.get_upto_nth_order_neighbors_all_grains_prob
                    self.no_gid = {}
                    if find_neigh[3] == '1-no':
                        for _no_ in no:
                            print(10*'-', "\nEstimating upto nth order neighbours",
                                  f"of every cell: no={_no_}")
                            self.no_gid[_no_] = fx(_no_, recalculate=False,
                                                   include_parent=find_neigh[2],
                                                   print_msg=True)
                    elif find_neigh[3] == 'no^th':
                        # CODES
                        pass
                print(40*'-')
            # -----------------
            if find_grain_voxel_locs:
                verbosity=int(self.n/20)
                if self.domvol <= self.ctrls['numba_activation_nvox_threshold']:
                    self.find_grain_voxel_locs(verbosity=verbosity)
                else:
                    self.find_grain_voxel_locs_v1(disp_msg=True,
                                                  verbosity=verbosity,
                                                  saa=True,
                                                  throw=False,
                                                  use_uint16=True)
            # -----------------
            if find_spatial_bounds_of_grains:
                self.find_spatial_bounds_of_grains()
                # gstslice.spbound, gstslice.spboundex
            # -----------------
            if find_grain_locations:
                print("Finding grain locatikons in domain")
                if not find_grain_voxel_locs:
                    print("Finding grain voxel locations first")
                    verbosity=int(self.n/20)
                    if self.domvol <= self.ctrls['numba_activation_nvox_threshold']:
                        self.find_grain_voxel_locs(verbosity=verbosity)
                    else:
                        self.find_grain_voxel_locs_v1(disp_msg=True,
                                                      verbosity=verbosity,
                                                      saa=True,
                                                      throw=False,
                                                      use_uint16=True)
                self.set_grain_positions()
        if set_mprops:
            print(40*"-")
            self.set_mprops(**mprops_kwargs)

    def set_skimrp(self):
        """Set the region properties of the scikit image."""
        from skimage.measure import regionprops
        self.skimrp = {}
        for gid in self.gid:
            self.skimrp[gid] = regionprops(1*(self.lgi == gid))[0]

    def set_mprops(self, set_skimrp=False, volnv=True, eqdia=False,
                   eqdia_base_size_spec='volnv',
                   arbbox=False, arbbox_fmt='gid_dict',
                   arellfit=False, arellfit_metric='max',
                   arellfit_calculate_efits=True,
                   arellfit_efit_routine=1,
                   arellfit_efit_regularize_data=True,
                   solidity=True, sol_nan_treatment='replace',
                   sol_inf_treatment='replace',
                   sol_nan_replacement=-1, sol_inf_replacement=-1,
                   sanv=False, sanv_N=26, rat_sanv_volnv=False,
                   sanv_verbosity=1E2, disp_msg='',
                   ):
        """
        Set morphological properties of the grain structure.

        Parameters
        ----------
        volnv : bool, optional
            Default value is True. Flag value for computing grain volumes
            by number of voxels. Compute if True.

        eqdia : bool, optional
            Default value is False. Flag value to compute sphere equivalent
            volume diameter.

        eqdia_base_size_spec : str, optional
            Default value is 'volnv'. Specify which sort of volume or surface
            area is to be used to calculate the equivalent diameter. Options
            include the follwowing:
                * 'volnv': Volume by number of voxels
                * 'volsr': Volume by surface reconstruction
                * 'volch': Volume of convex hull
                * 'sanv': surface ares by number of voxels
                * 'savi': surface area by voxel interfaces
                * 'sasr': surface area by surface reconstruction

        arbbox : bool, optional
            Default value is False. Flag value for computing grain aspect
            ratios by using bonding box dimensions. Compute if True.

        arbbox_fmt : str, optional
            Default value is 'gid_dict'. Specify format of storing the
            calculated arbbox values. Options are:
                * list
                * np / np_array / np.array / numpy

        arellfit : bool, optional
            Default value is False. Flag value to compute aspect ratio by
            using axes of the ellipsoidal fits to grains.

        arellfit_metric : str, optional
            Metric to use in aspect ratrio clauclation. Refer to doicumentation
            of function set_mprop_arellfit for further details. Default value
            is 'max'. Options include the following:
                * max / maximum / maximal
                * min / minimum / minimal
                * xy / yx / z
                * yz / zy / x
                * xz / yz / y

        arellfit_calculate_efits : bool, optional
            Default value is True. Set to True if ellispoids are to be fit
            (or refit, if that be the case) first.

        arellfit_efit_routine : int, optional
            Default value is 1. Refer to doicumentation of function
            set_mprop_arellfit for further details.

        arellfit_efit_regularize_data : bool, optional
            Default value is True. Refer to doicumentation of function
            set_mprop_arellfit for further details.
        """
        '''Set the scikit image region morphological property generators
        for all gids.'''

        # ----------------------------------------------------------
        if set_skimrp:
            self.set_skimrp()
        # ----------------------------------------------------------
        if volnv:
            self.set_mprop_volnv(msg=disp_msg)
        # ----------------------------------------------------------
        if eqdia:
            self.set_mprop_eqdia(base_size_spec='volnv')
        # ----------------------------------------------------------
        if solidity:
            self.set_mprop_solidity(reset_generators=False,
                                    nan_treatment=sol_nan_treatment,
                                    inf_treatment=sol_inf_treatment,
                                    nan_replacement=sol_nan_replacement,
                                    inf_replacement=sol_inf_replacement)
        # ----------------------------------------------------------
        if arbbox:
            self.set_mprop_arbbox(fmt=arbbox_fmt)
        # ----------------------------------------------------------
        if arellfit:
            self.set_mprop_arellfit(metric=arellfit_metric,
                                    calculate_efits=arellfit_calculate_efits,
                                    efit_routine=arellfit_efit_routine,
                                    efit_regularize_data=arellfit_efit_regularize_data)
        # ----------------------------------------------------------
        if sanv:
            self.set_mprop_sanv(N=sanv_N, verbosity=sanv_verbosity)
        # ----------------------------------------------------------
        if rat_sanv_volnv:
            reset_volnv = False
            reset_sanv = False
            if not volnv:
                reset_volnv = True
            if not sanv:
                reset_sanv = True
            self.set_mprop_rat_sanv_volnv(reset_volnv=reset_volnv,
                                          reset_sanv=reset_sanv,
                                          N=sanv_N,
                                          verbosity=sanv_verbosity)

    def plot_mprop_correlations(self):
        # VALIDATIONS
        # -----------------------------
        g = sns.jointplot(x=tgt_npixels, y=tgt_nneigh_field, kind='hex', gridsize=25, cmap='viridis',
                          marginal_kws=dict(bins=50, fill=True))
        g.plot_marginals(sns.histplot, bins=50, kde=True, color='gray', fill=True)
        g.fig.suptitle('Customized Jointplot of A and B', y=1.02)
        g.set_axis_labels('A values', 'B values')
        g.ax_joint.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        plt.scatter(tgt_npixels, tgt_nneigh_field, s=3, color='black', alpha=0.25)

        fit_polynomial_order = 2
        factor = 2
        sort_indices = np.argsort(tgt_npixels)
        tgt_npixels_limited = tgt_npixels[sort_indices][tgt_npixels[sort_indices] <= factor*tgt_npixels.mean()]
        tgt_nneigh_field_limited = tgt_nneigh_field[sort_indices][tgt_npixels[sort_indices] <= factor*tgt_npixels.mean()]
        coefficients = np.polyfit(tgt_npixels_limited, tgt_nneigh_field_limited, fit_polynomial_order)
        polynomial = np.poly1d(coefficients)
        tgt_nneigh_field_fit = polynomial(tgt_npixels_limited)
        plt.plot(tgt_npixels_limited, tgt_nneigh_field_fit, 'k')

    def find_grains(self, label_str_order=1, pck=None):
        """
        Detect grains in lgi.

        Parameters
        ----------
        label_str_order : inr, optional
            Provide the voxel connectivity order used for grain identification.
            Defaults to a value of 1.

        Returns
        -------
        None.

        Explanations
        ------------
        Using the library 'scikit-image'.
        """
        # VALIDATIONS
        # -----------------------------
        print(40*'-', '\nFinding grains.')
        self.set_binaryStructure3D(n=label_str_order)
        _STR_ = self.get_binaryStructure3D()
        for i, _s_ in enumerate(np.unique(self.s)):
            # Mark the presence of this state
            self.spart_flag[_s_] = True
            # Recognize the grains belonging to this state
            bin_img = (self.s == _s_).astype(np.uint8)
            labels, num_labels = pck(bin_img, structure=_STR_)
            if i == 0:
                self.lgi = labels
            else:
                labels[labels > 0] += self.lgi.max()
                self.lgi = self.lgi + labels
            self.s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))

            # print(20*'-', '\n', _s_)

            self.s_n[_s_-1] = len(self.s_gid[_s_])
        self.lgi = np.int32(self.lgi)
        # Get the total number of grains
        self.calc_num_grains()
        # self.n = np.unique(self.lgi).size  # self.n = num_labels
        # Generate and store the gid-s mapping
        self.gid = list(range(1, self.n+1))
        _gid_s_ = []
        for _gs_, _gid_ in zip(self.s_gid.keys(), self.s_gid.values()):
            if _gid_:
                for __gid__ in _gid_:
                    _gid_s_.append(_gs_)
            else:
                pass
                # _gid_s_.append(0)  # Splcing this temporarily. Retain if fully successfull.
        self.gid_s = _gid_s_
        print(f'No. of grains detected = {self.n}')

    def find_grains_lgi(self):
        # Call find_grains_img
        pass

    def find_grains_img(self, img=None):
        for i in np.unique(img):
            pass

    def _check_lgi_dtype_uint8(self,
                               lgi,
                               ):
        """
        Validates and modifies (if needed) lgi user input data-type

        Parameters
        ----------
        lgi : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if type(lgi) == np.ndarray and np.size(lgi) > 0 and np.ndim(lgi) == 2:
            if self.lgi.dtype.name != 'uint8':
                self.lgi = lgi.astype(np.uint8)
            else:
                self.lgi = lgi
        else:
            self.lgi = None

    def set_gid(self):
        self.gid = list(range(1, np.unique(self.lgi).size+1))

    def calc_num_grains(self, throw=False):
        """
        Calculate the total number of grains in this grain structure

        Parameters
        ----------
        throw : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if self.lgi is not None:
            self.n = np.unique(self.lgi).size
            if throw:
                return self.n

    @staticmethod
    @njit
    def find_neigh_gid_numba(lgi):
        """
        Optimized function to compute neighboring grain IDs using numba.
        """
        dxdydz = np.array([
            (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1,  0, -1),
            (-1,  0, 0), (-1,  0, 1), (-1,  1, -1), (-1,  1, 0),
            (-1,  1, 1), (0, -1, -1), (0, -1, 0), (0, -1, 1),
            (0,  0, -1), (0,  0, 1), (0,  1, -1), (0,  1, 0), (0,  1, 1),
            (1, -1, -1), (1, -1, 0), (1, -1, 1), (1,  0, -1), (1,  0, 0),
            (1,  0, 1), (1,  1, -1), (1,  1, 0), (1,  1, 1)
        ], dtype=int32)

        shape_x, shape_y, shape_z = lgi.shape
        max_gid = np.max(lgi)

        # Use a Numba-compatible dictionary with fixed-size NumPy arrays
        neigh_gid = Dict.empty(
            key_type=int32,
            value_type=int32[:]
        )

        # Max possible neighbors (26 for 3D Moore neighborhood)
        max_neighbors = 26

        # Preallocate storage for each grain ID
        for gid in range(max_gid + 1):
            neigh_gid[gid] = np.full(max_neighbors, -1, dtype=int32)  # -1 for unused slots

        # Track the actual count of neighbors for each grain ID
        neighbor_counts = np.zeros(max_gid + 1, dtype=int32)

        for x in range(shape_x):
            for y in range(shape_y):
                for z in range(shape_z):
                    grain_id = lgi[x, y, z]

                    for dx, dy, dz in dxdydz:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < shape_x and 0 <= ny < shape_y and 0 <= nz < shape_z:
                            neighbor_id = lgi[nx, ny, nz]

                            if neighbor_id != grain_id:
                                count = neighbor_counts[grain_id]

                                # Avoid duplicates
                                found = False
                                for i in range(count):
                                    if neigh_gid[grain_id][i] == neighbor_id:
                                        found = True
                                        break

                                if not found:
                                    neigh_gid[grain_id][count] = neighbor_id
                                    neighbor_counts[grain_id] += 1

        # Convert to final format (truncate unused values)
        final_neigh_gid = Dict.empty(
            key_type=int32,
            value_type=int32[:]
        )

        for gid in range(max_gid + 1):
            final_neigh_gid[gid] = neigh_gid[gid][:neighbor_counts[gid]]  # Remove `-1` slots

        return neigh_gid

    def find_neigh_gid_v1(self, verbose=False, verbosity=4):
        """
        gstslice.find_neigh_gid_v1()
        """
        print("Calculating neigh_gid database.")
        self.neigh_gid = self.find_neigh_gid_numba(self.lgi)

    def find_neigh_gid_V1(self, verbose=False, verbosity=4):
        """
        gstslice.find_neigh_gid()
        """
        print("Calculating neigh_gid database.")
        _neigh_gid_ = self.find_neigh_gid_numba(self.lgi)
        self.neigh_gid = {key: value for key, value in _neigh_gid_.items()}

    def find_neigh_gid(self, verbose=False, verbosity=4):
        """
        Set neighbouring gids of all grains (optimized).
        """
        print('Calculating 1st order neighbours.')

        lgi = self.lgi
        neigh_gid = {}
        unique_grains = np.unique(lgi)

        dxdydz = np.array([
            (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0),
            (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, -1),
            (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 1), (0, 1, -1),
            (0, 1, 0), (0, 1, 1), (1, -1, -1), (1, -1, 0), (1, -1, 1),
            (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)
        ])

        nvox = unique_grains.size
        verbosity = np.round(nvox/verbosity)
        i = 1
        for grain_id in unique_grains:
            neighbours = np.unique(self.find_neigh_gid_core_numba(lgi, grain_id, dxdydz))
            neigh_gid[grain_id] = list(np.unique(neighbours))
            if i%verbosity == 0:
                print(f'    {np.round(i*100/nvox)}% complete.')
            i += 1

        self.neigh_gid = neigh_gid

    @staticmethod
    @njit
    def find_neigh_gid_core_numba(lgi, gid, dxdydz):
        shape_x, shape_y, shape_z = lgi.shape
        max_neighbors = 26  # Maximum possible neighbors in a 3D grid

        # Preallocate a NumPy array for neighbor storage
        neighbors = np.full(max_neighbors, -1, dtype=int32)  # -1 means empty
        neighbor_count = 0

        for dx, dy, dz in dxdydz:
            for x in range(shape_x):
                for y in range(shape_y):
                    for z in range(shape_z):
                        if lgi[x, y, z] == gid:  # Only process grain voxels
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if 0 <= nx < shape_x and 0 <= ny < shape_y and 0 <= nz < shape_z:
                                neighbor_id = lgi[nx, ny, nz]

                                if neighbor_id != gid:  # Ignore same grain
                                    # Check if the neighbor_id is already in the list
                                    found = False
                                    for i in range(neighbor_count):
                                        if neighbors[i] == neighbor_id:
                                            found = True
                                            break

                                    if not found and neighbor_count < max_neighbors:
                                        neighbors[neighbor_count] = neighbor_id
                                        neighbor_count += 1

        # Return only the valid part of the array
        return neighbors[:neighbor_count]

    def find_neigh_fdb(self, feature_name='base', instance_name='lgi',
                       disp_msg=False, verbosity=500, save_to_neigh_gid=False):
        """
        Set neighbouring fids of all features.
        """
        print(f'Calculating 1st order neighbours: {feature_name}: {instance_name}')

        if feature_name in ('base', 'lgi'):
            FDB = self.lgi
        elif feature_name[:4] in ('twin', 'twins'):
            if instance_name not in self.fdb.keys():
                raise ValueError("Invalid instance name specified")
            if instance_name[:4] in ('twin'):
                FDB = self.fdb[instance_name]['data']['fid']
            else:
                raise ValueError("Invalid instance name specified")

        neigh_fid = {}
        unique_grains = np.unique(FDB)

        dxdydz = np.array([
            (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0),
            (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, -1),
            (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 1), (0, 1, -1),
            (0, 1, 0), (0, 1, 1), (1, -1, -1), (1, -1, 0), (1, -1, 1),
            (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)
        ])

        nvox = unique_grains.size
        verbosity = max(int(np.round(nvox / max(int(verbosity), 1))), 1)
        i = 1
        for grain_id in unique_grains:
            neighbours = np.unique(self.find_neigh_gid_core_numba(FDB, grain_id, dxdydz))
            neigh_fid[grain_id] = list(np.unique(neighbours))
            if i%verbosity == 0:
                print(f'    {np.round(i*100/nvox)}% complete.')
            i += 1

        self.neigh_gid = neigh_fid
        if save_to_neigh_gid:
            self.neigh_gid = {int(fid): np.array(neigh_fid[fid], dtype=np.int32) for fid in neigh_fid}
        if instance_name in self.fdb and 'data' in self.fdb[instance_name]:
            self.fdb[instance_name]['data']['neigh_fid'] = neigh_fid
        

    def find_neigh_gid_01(self, verbose=False, verbosity=4):
        """
        Set neighbouring gids of all grains (optimized).
        """
        print('Calculating 1st order neighbours.')

        lgi = self.lgi
        neigh_gid = {}
        unique_grains = np.unique(lgi)

        dxdydz = np.array([
            (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0),
            (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, -1),
            (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 1), (0, 1, -1),
            (0, 1, 0), (0, 1, 1), (1, -1, -1), (1, -1, 0), (1, -1, 1),
            (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)
        ])

        nvox = unique_grains.size
        verbosity = np.round(nvox/verbosity)
        i = 1
        for grain_id in unique_grains:
            # neighbors = set()
            neighbors = []
            grain_mask = (lgi == grain_id)
            if i%verbosity == 0:
                print(f'    {np.round(i*100/nvox)}% complete.')
            for dx, dy, dz in dxdydz:
                shifted_lgi = np.roll(lgi, shift=(dx, dy, dz), axis=(0, 1, 2))
                neighbor_mask = (shifted_lgi != grain_id) & grain_mask
                neighbors.extend(shifted_lgi[neighbor_mask])
            neigh_gid[grain_id] = list(np.unique(neighbors))
            i += 1

        self.neigh_gid = neigh_gid

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

    def get_upto_nth_order_neighbors(self, grain_id, neigh_order,
                                     recalculate=False, include_parent=True,
                                     output_type='list'):
        """
        Calculates 0th till nth order neighbors for a given gid.

        Parameters
        ----------
        grain_id : int
            The ID of the cell for which to find neighbors.
        neigh_order : int
            The order of neighbors to calculate (1st order, 2nd order, etc.).
        include_parent : bool
            If True, user provided grain_id will also be included in the list
            of neighbours, as a grain is its 0th order neightbour, that is, its
            own neighrbour. DEfaults value is True.
        output_type : str
            Specify the desired neighbour data type. Options include the
            following:
                * list
                * nparray
                * set

        saa
        ---
        None

        Returns
        -------
        A set containing the nth order neighbors.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent')
        pxtal.simulate()
        pxtal.detect_grains()
        gid = 10
        np.unique(pxtal.gs[16].find_extended_bounding_box(gid))
        # pxtal.gs[16].find_neigh_gid_fast_all_grains(include_parent=False)
        pxtal.gs[16].find_neigh_gid()

        neigh_order = 3
        pxtal.gs[16].get_upto_nth_order_neighbors(gid,
                                                  neigh_order,
                                                  recalculate=False,
                                                  include_parent=True,
                                                  output_type='list')
        """
        if neigh_order == 0:
            return grain_id
        if recalculate or not self.neigh_gid:
            self.find_neigh_gid()
            # self.find_neigh_gid_fast_all_grains(include_parent=False)
        # Start with 1st-order neighbors
        neighbors = set(self.neigh_gid.get(grain_id, []))
        # ---------------------------
        for _ in range(neigh_order - 1):
            new_neighbors = set()
            for neighbor in neighbors:
                new_neighbors.update(self.neigh_gid.get(neighbor, []))
            neighbors.update(new_neighbors)
        # ---------------------------
        if not include_parent:
            neighbors.discard(grain_id)
        if output_type == 'list':
            return list(neighbors)
        if output_type == 'nparray':
            return np.array(list(neighbors))
        elif output_type == 'set':
            return neighbors

    def get_nth_order_neighbors(self, grain_id, neigh_order, recalculate=False,
                                include_parent=True):
        """
        Calculate only the nth order neighbors for a given gid.

        Parameters
        ----------
        grain_id : int
            The ID of the cell for which to find neighbors.
        neigh_order : int
            The order of neighbors to calculate (1st order, 2nd order, etc.).
        include_parent : bool
            If True, user provided grain_id will also be included in the list
            of neighbours, as a grain is its 0th order neightbour, that is, its
            own neighrbour. DEfaults value is True.
        output_type : str
            Specify the desired neighbour data type. Options include the
            following:
                * list
                * nparray
                * set

        saa
        ---
        None

        Returns
        -------
        A set containing the nth order neighbors.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        fname = 'input_dashboard_for_testing_50x50_alg202.xls'
        pxtal = mcgs(study='independent',
                     input_dashboard=fname)
        pxtal.simulate()
        pxtal.detect_grains()
        gid = 10
        np.unique(pxtal.gs[16].find_extended_bounding_box(gid))
        # pxtal.gs[16].find_neigh_gid_fast_all_grains(include_parent=False)
        pxtal.gs[16].find_neigh_gid()
        neigh_order = 2
        pxtal.gs[16].get_nth_order_neighbors(gid, neigh_order,
                                             recalculate=False,
                                             include_parent=True)
        """
        fx = self.get_upto_nth_order_neighbors
        neigh_upto_n_minus_1 = fx(grain_id, neigh_order-1,
                                  recalculate=recalculate,
                                  include_parent=include_parent,
                                  output_type='set')
        # --------------------------------
        if type(neigh_upto_n_minus_1) in dth.dt.NUMBERS:
            neigh_upto_n_minus_1 = set([neigh_upto_n_minus_1])
        # --------------------------------
        fx = self.get_upto_nth_order_neighbors
        neigh_upto_n = fx(grain_id, neigh_order, recalculate=recalculate,
                          include_parent=include_parent, output_type='set')
        # --------------------------------
        if type(neigh_upto_n) in dth.dt.NUMBERS:
            neigh_upto_n = set([neigh_upto_n])
        return list(neigh_upto_n.difference(neigh_upto_n_minus_1))

    def get_upto_nth_order_neighbors_all_grains(self, neigh_order,
                                                recalculate=False,
                                                include_parent=True,
                                                output_type='list'):
        """
        Calculate 0th till nth order neighbors of all gids in grain structure.

        Parameters
        ----------
        neigh_order : int
            The order of neighbors to calculate (1st order, 2nd order, etc.).

        recalculate : bool, optional
            Defaults to False

        include_parent : bool, optional
            Defaults to True

        output_type : str, optional
            Defaults to 'list'.

        Returns
        -------
        A set containing the nth order neighbors.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()

        neigh_order = 3
        pxtal.gs[16].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,
                                                             output_type='list')
        """
        fx = self.get_upto_nth_order_neighbors
        neighs_upto_nth_order = {gid: fx(gid, neigh_order, output_type='list',
                                         recalculate=recalculate,
                                         include_parent=include_parent)
                                 for gid in self.gid}
        return neighs_upto_nth_order

    def get_nth_order_neighbors_all_grains(self, neigh_order,
                                           recalculate=False,
                                           include_parent=True):
        """
        Calculate only the nth order neighbors of all gids in grain structure.

        Parameters
        ----------
        neigh_order : int
            The order of neighbors to be calculated.

        recalculate : bool, optional
            Defaults to False

        include_parent : bool, optional
            Defaults to True

        Returns
        -------
        neighs_nth_order

        saa
        ---
        None

        Exaplanations
        -------------

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        fname = 'input_dashboard_for_testing_50x50_alg202.xls'
        pxtal = mcgs(study='independent',
                     input_dashboard=fname)
        pxtal.simulate()
        pxtal.detect_grains()
        neigh_order = 2
        A = pxtal.gs[16].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                             include_parent=True,
                                                             output_type='list')
        B = pxtal.gs[16].get_nth_order_neighbors_all_grains(neigh_order,
                                                            recalculate=False,
                                                            include_parent=True)

        """
        fx = self.get_nth_order_neighbors
        neighs_nth_order = {gid: fx(gid, neigh_order, recalculate=False,
                                    include_parent=include_parent)
                            for gid in self.gid}
        return neighs_nth_order

    def get_upto_nth_order_neighbors_all_grains_prob(self, neigh_order,
                                                     recalculate=False,
                                                     include_parent=False,
                                                     print_msg=False):
        """
        Calculate 0th till nth order neigh of all gids with a probability.

        Parameters
        ----------
        neigh_order : int
            The order of neighbors to be calculated.

        recalculate : bool, optional
            Defaults to False

        include_parent : bool, optional
            Defaults to True

        print_msg : bool, optional
            Display messages if True, dont if False. Defaults to False.

        Returns
        -------
        neighs_nth_order

        saa
        ---
        None

        Exaplanations
        -------------

        Pre-example calculations
        ------------------------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 10
        fx = pxt.gs[tslice].get_upto_nth_order_neighbors_all_grains_prob

        Example-1
        ---------
        neigh0 = fx(1, include_parent=True)
        neigh0[22]

        Example-2
        ---------
        neigh1 = fx(1.06, include_parent=True)
        neigh1[2][22]

        Example-3
        ---------
        neigh2 = fx(1.5, include_parent=True)
        neigh2[2][22]
        """
        no = neigh_order
        on_neigh_all_grains_upto = self.get_upto_nth_order_neighbors_all_grains
        on_neigh_all_grains_at = self.get_nth_order_neighbors_all_grains
        if isinstance(no, (int, np.int32)):
            if print_msg:
                print('neigh_order is of type int. Adopting the usual method.')
            neigh_on = on_neigh_all_grains_upto(no,
                                                include_parent=include_parent)
            return neigh_on
        elif isinstance(no, (float, np.float64)):
            if abs(no-round(no)) < 0.05:
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

    def __setup__positions__(self):
        """Setup template dict with default spatial location keys for gids."""
        self.positions = {'front_top_left': [], 'front_bottom_left': [],
                          'front_bottom_right': [], 'front_top_right': [],
                          'front_pure_right': [], 'front_pure_bottom': [],
                          'front_pure_left': [], 'front_pure_top': [],

                          'back_top_left': [], 'back_bottom_left': [],
                          'back_bottom_right': [], 'back_top_right': [],
                          'back_pure_right': [], 'back_pure_bottom': [],
                          'back_pure_left': [], 'back_pure_top': [],

                          'front_left': [], 'front_bottom': [],
                          'front_right': [], 'front_top': [],
                          'back_left': [], 'back_bottom': [],
                          'back_right': [], 'back_top': [],

                          'boundary': [], 'corner': [], 'internal': []
                          }

    def make_pvgrid(self):
        """Make pyvista grid of the lgi."""
        print(40*'-', '\nSetting PyVista grid.')
        self.pvgrid = pv.ImageData()
        self.pvgrid.dimensions = np.array(self.lgi.shape) + 1
        self.pvgrid.origin = (0, 0, 0)
        self.pvgrid.spacing = (1, 1, 1)

    def add_scalar_field_to_pvgrid(self, sf_name="lgi", sf_value=None):
        """
        Add scalar variable to Py-Vista grid.

        Parameters
        ----------
        sf_name : str, optional
            Default value is "lgi".

        sf_value : None, optional
            Default vlaue is None.

        saa
        ---
        cell data in self.pvgrid

        Explanations
        ------------
        """
        # Validations
        # ------------------------------
        _str_ = '\nAdding scalar field: {sf_name} to PyVista grid self.pvgrid.'
        print(40*'-', _str_)
        print("Access: self.pvgrid.cell_data['{sf_name}']")
        if sf_name in self.valid_scalar_fields:
            if sf_name == "lgi":
                self.pvgrid.cell_data[sf_name] = self.lgi.flatten(order="F")
            elif sf_name == "s":
                self.pvgrid.cell_data[sf_name] = self.s.flatten(order="F")
            elif sf_name[:3].lower() in ("fid"):
                self.pvgrid.cell_data[sf_name] = sf_value.flatten(order="F")
        else:
            self.pvgrid.cell_data[sf_name] = sf_value.flatten(order="F")

    def make_zero_non_gids_in_lgi(self, gids):
        """
        Return a gids masked copy of lgi.

        Paramaters
        ----------
        gids : list

        saa
        ---
        None

        Returns
        -------
        masked_lgi : np.ndarray
            Gids masked copy of lgi
        """
        _lgi_ = deepcopy(self.lgi)
        for gid in gids:
            _lgi_[_lgi_ == gid] = -1
        _lgi_[_lgi_ != -1] = 0
        masked_lgi = np.abs(np.multiply(_lgi_, self.lgi))
        return masked_lgi

    def make_zero_non_gids_in_lgisubset(self, lgi_subset, gids):
        """
        Return a gids masked copy of lgi_subset.

        Paramaters
        ----------
        gids : list

        saa
        ---
        None

        Returns
        -------
        masked_lgi : np.ndarray
            Gids masked copy of lgi_subset.
        """
        _lgi_subset_ = deepcopy(lgi_subset)
        for gid in gids:
            _lgi_subset_[_lgi_subset_ == gid] = -1
        _lgi_subset_[_lgi_subset_ != -1] = 0
        masked_lgi = np.abs(np.multiply(_lgi_subset_, lgi_subset))
        return masked_lgi

    def plot_gs_pvvox(self, alpha=1.0, title='UPXO.MCGS3D.',
                      cs_labels='user', scalar="lgi",
                      _xname_='Z: lgi[:,:,n]',
                      _yname_='Y: lgi[:,n,:]',
                      _zname_='X: lgi[n,:,:]', show_edges=False):
        """
        Plot the grain structure as pyvista voxels.

        Parameters
        ----------
        alpha : float, optional
        title : str, optional
        cs_labels : str, optional
        _xname_ : str, optional
        _yname_ : str, optional
        _zname_ : str, optional

        NOTE
        ----
        If cs_labels is not 'user':
            X on triad will be lgi[n, :, :] -- which is z of numpy array
            Y on triad will be lgi[n, :, :] -- which is z of numpy array
            X on triad will be lgi[n, :, :] -- which is z of numpy array
        """
        pvp = pv.Plotter()
        pvp.add_mesh(self.pvgrid,
                     scalars=scalar,
                     show_edges=show_edges,
                     opacity=alpha)
        pvp.add_text(f"{title}", font_size=10)
        if cs_labels == 'user':
            _ = pvp.add_axes(line_width=5, cone_radius=0.6,
                             shaft_length=0.7, tip_length=0.3,
                             ambient=0.5, label_size=(0.4, 0.16),
                             xlabel=_xname_, ylabel=_yname_, zlabel=_zname_,
                             viewport=(0, 0, 0.25, 0.25))
        else:
            _ = pvp.add_axes(line_width=5, cone_radius=0.6, shaft_length=0.7,
                             tip_length=0.3, ambient=0.5,
                             label_size=(0.4, 0.16),
                             viewport=(0, 0, 0.25, 0.25))
        # ---------------------------------
        pvp.show()

    def plot_gs_pvvox_subset(self, lgi_subset, alpha=1.0,
                             plot_points=False, points=None,
                             isolate_gid=False, gid_to_isolate=None):
        """
        Plot subset of voxellated grain strucure in pyvista.

        Parameters
        ----------
        lgi_subset : np.ndarray
            Numpy spatial field array.

        alpha : float, optional
            Transparency value. Value must be in [0.0, 1.0]. Defaults to 1.0.

        plot_points : bool, optional
            Flag to plot additional points on top of the pvgrid. DEfaults to
            False.

        points : np.ndarray, optional
            List of coordinate points to be plotted. Defaults to None.

        isolate_gid : bool, optional
            Flag to isolate a specific gid. Defaults to False.

        gid_to_isolate : int, optional
            The gid to isolate. Defaults to None.

        Examples
        --------
        gstslice.plot_gs_pvvox_subset(gstslice.find_exbounding_cube_gid(5),
                                      alpha=0.5)
        gstslice.plot_gs_pvvox_subset(gstslice.find_bounding_cube_gid(5),
                                      alpha=0.5, isolate_gid=True, gid=5)
        """
        if isolate_gid:
            lgi_subset = self.make_zero_non_gids_in_lgisubset(lgi_subset,
                                                              [gid_to_isolate])
        pvsubset = pv.UniformGrid()
        pvsubset.dimensions = np.array(lgi_subset.shape) + 1
        pvsubset.origin = (0, 0, 0)
        pvsubset.spacing = (1, 1, 1)
        pvsubset.cell_data['lgi'] = lgi_subset.flatten(order="F")
        # --------------------------------
        pvp = pv.Plotter()
        pvp.add_mesh(pvsubset, show_edges=True, opacity=alpha)
        pvp.show()

    @staticmethod
    @njit
    def _count_labels(lgi_flat, gid_to_idx):
        counts = np.zeros(gid_to_idx.max()+1, dtype=np.int64)
        for t in range(lgi_flat.size):
            g = lgi_flat[t]
            idx = gid_to_idx[g]
            if idx >= 0:
                counts[idx] += 1
        return counts

    @staticmethod
    @njit
    def _fill_coords(lgi, gid_to_idx, counts):
        nx, ny, nz = lgi.shape
        out_i = [np.empty(c, dtype=np.uint16) for c in counts]
        out_j = [np.empty(c, dtype=np.uint16) for c in counts]
        out_k = [np.empty(c, dtype=np.uint16) for c in counts]
        pos = np.zeros(len(counts), dtype=np.int64)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    g = lgi[i, j, k]
                    idx = gid_to_idx[g]
                    if idx >= 0:
                        w = pos[idx]
                        out_i[idx][w] = i
                        out_j[idx][w] = j
                        out_k[idx][w] = k
                        pos[idx] += 1
        return out_i, out_j, out_k

    def find_grain_voxel_locs_v1(self, gids=None, disp_msg=False, verbosity=100,
                                 saa=True, throw=False, dtype=np.int32, use_uint16=True):
        """
        Find voxel locations of grains in self.lgi (label grid).
        Stores a dict: grain_id -> (N,3) array of voxel indices.
        """
        print("\nFinding voxel locations of grains in lgi")
        if not gids:
            gids = np.asarray(self.gid, dtype=np.int64)

        gid_to_idx = -np.ones(int(gids.max())+1, dtype=np.int64)
        for idx, g in enumerate(gids):
            gid_to_idx[g] = idx

        lgi_flat = self.lgi.ravel(order="C")

        counts = self._count_labels(lgi_flat, gid_to_idx)
        out_i, out_j, out_k = self._fill_coords(self.lgi, gid_to_idx, counts)

        grain_locs = {}
        for idx, g in enumerate(gids):
            coords = np.column_stack((
                out_i[idx].astype(dtype, copy=False),
                out_j[idx].astype(dtype, copy=False),
                out_k[idx].astype(dtype, copy=False)
            ))
            grain_locs[int(g)] = coords
            if disp_msg and (idx % verbosity == 0):
                print(f"gid: {g} of {len(gids)} grains")

        if saa:
            self.grain_locs = grain_locs
        if throw:
            return grain_locs

    def find_feat_voxel_locs_v1(self, feature_name='base', instance_name='lgi',
                                fids=None, disp_msg=False, verbosity=100,
                                saa=True, throw=False, use_uint16=True):
        """
        Find voxel locations of features in self.lgi (label grid).
        Stores a dict: grain_id -> (N,3) array of voxel indices.

        Example-1
        ---------
        gstslice.find_feat_voxel_locs_v1(feature_name='base',
                                         instance_name='lgi',
                                         fids=None,
                                         disp_msg=False,
                                         verbosity=100,
                                         saa=True, throw=False)
        Example-2
        ---------
        feat_locs = gstslice.find_feat_voxel_locs_v1(feature_name='twin',
                                                     instance_name='twin.0',
                                                     fids=None,
                                                     disp_msg=False,
                                                     verbosity=100,
                                                     throw=True)
        Example-3
        ---------
        feature_name, instance_name = 'twin', 'twin.0'
        feat_ids = gstslice.fdb[instance_name]['data']['twin_id'][:2]
        feat_locs = gstslice.find_feat_voxel_locs_v1(feature_name=feature_name,
                                                     instance_name=instance_name,
                                                     fids=feat_ids,
                                                     disp_msg=False,
                                                     verbosity=100,
                                                     throw=True)
        """
        print("\nFinding voxel locations of features.")
        # Validations
        if feature_name in ('base', 'lgi'):
            FDB = self.lgi
            if not fids:
                fids = np.asarray(self.gid, dtype=np.int16)
        elif feature_name[:4] in ('twin'):
            if instance_name not in self.fdb.keys():
                raise ValueError("Invalid instance name specified")
            if instance_name[:4] in ('twin'):
                FDB = self.fdb[instance_name]['data']['fid']
                if type(fids) not in (dth.dt.ITERABLES, str) and fids==None:
                    fids = np.asarray(self.gid, dtype=np.int16)
                elif fids in ("feat_host_ids", "host_ids", "host"):
                    fids = self.fdb[instance_name]['data']['feat_host_ids']
                elif fids in ("notwin_gids", "notwins", "nonhosts"):
                    fids = self.fdb[instance_name]['data']['notwin_gids']
                elif fids in ("twin_id", "twin_ids", "twins"):
                    fids = self.fdb[instance_name]['data']['twin_id']
                elif type(fids) in dth.dt.ITERABLES:
                    # Validate fids
                    _fidsall_ = np.unique(self.fdb[instance_name]['data']['fid'])
                    fids = _fidsall_[np.where([fid in _fidsall_
                                               for fid in fids])[0]]
                else:
                    raise ValueError("Invalid fids specified.")
            else:
                raise ValueError("Invalid instance name specified")
        # ------------------------------------------
        gid_to_idx = -np.ones(int(fids.max())+1, dtype=np.int64)
        for idx, g in enumerate(fids):
            gid_to_idx[g] = idx
        # ------------------------------------------
        lgi_flat = FDB.ravel(order="C")
        # ------------------------------------------
        counts = self._count_labels(lgi_flat, gid_to_idx)
        out_i, out_j, out_k = self._fill_coords(FDB, gid_to_idx, counts)
        # ------------------------------------------
        dtype_idx = np.uint16 if use_uint16 else np.int64
        feat_locs = {}
        for idx, g in enumerate(fids):
            coords = np.column_stack((
                out_i[idx].astype(dtype_idx, copy=False),
                out_j[idx].astype(dtype_idx, copy=False),
                out_k[idx].astype(dtype_idx, copy=False)
            ))
            feat_locs[int(g)] = coords
            if disp_msg and (idx % verbosity == 0):
                print(f"gid: {g} of {len(fids)} grains")
        # ------------------------------------------
        if feature_name in ('base', 'lgi'):
            if saa:
                self.feat_locs = feat_locs
            if throw:
                return feat_locs
        else:
            return feat_locs

    def find_grain_voxel_locs(self, disp_msg=False, verbosity=100, saa=True, throw=False):
        """
        Find voxel locations of grains in lgi.

        saa
        ---
        grain_locs
        """
        print('\nFinding voxel locations of grains in lgi.')
        ngrains = len(self.gid)
        if disp_msg and ngrains > verbosity:
            verbosity = ngrains//verbosity

        grain_locs = {gid: None for gid in self.gid}
        for gid in self.gid:
            grain_locs[gid] = np.vstack(np.where(self.lgi == gid)).T
            """if gid % verbosity == 0:
                print(f'gid: {gid} of {ngrains} grains')"""

        if saa:
            self.grain_locs = grain_locs
        if throw:
            return grain_locs

    def find_feature_voxel_locs(self, fname='lgi', fids=None, printmsg=True,
                                verbosity=10, saa=True, throw=False):
        """
        Find voxel locations of grains in lgi.

        saa
        ---
        grain_locs
        """
        if printmsg:
            print('\nFinding voxel locations of fids.')
        if fname == 'lgi':
            fids = self.gid
            fdb = self.lgi
        else:
            fdb = self.fdb[fname]['data']['fid']

        nfeatures = len(fids)

        if nfeatures > verbosity:
            verbosity = nfeatures//verbosity

        feat_locs = {fid: None for fid in fids}
        for i, fid in enumerate(fids, start=1):
            feat_locs[fid] = np.vstack(np.where(fdb == fid)).T
            if printmsg:
                if i % verbosity == 0:
                    print(f'fid: {i} of {nfeatures} grains')

        if saa:
            if fname == 'lgi':
                self.grain_locs = feat_locs
            else:
                self.feat_locs = feat_locs
        if throw:
            return feat_locs

    def find_spatial_bounds_of_grains(self):
        """
        Find the spatial bounds of each grain in the grain structure.

        saa
        ---
        self.spbound : dict
            Keys and values are explained below:
                * xmins : np.ndarray
                    Numpy array of minimum tight bound value of every grain
                    along x.
                * ymins : np.ndarray
                    Numpy array of minimum tight bound value of every grain
                    along y.
                * zmins : np.ndarray
                    Numpy array of minimum tight bound value of every grain
                    along z.
                * xmaxs : np.ndarray
                    Numpy array of maximum tight bound value of every grain
                    along x.
                * ymaxs : np.ndarray
                    Numpy array of maximum tight bound value of every grain
                    along y.
                * zmaxs : np.ndarray
                    Numpy array of maximum tight bound value of every grain
                    along z.
        self.spboundex : dict
            Keys and values are explained below:
                * xmins : np.ndarray
                    Numpy array of minimum loose bound value of every grain
                    along x.
                * ymins : np.ndarray
                    Numpy array of minimum loose bound value of every grain
                    along y.
                * zmins : np.ndarray
                    Numpy array of minimum loose bound value of every grain
                    along z.
                * xmaxs : np.ndarray
                    Numpy array of maximum loose bound value of every grain
                    along x.
                * ymaxs : np.ndarray
                    Numpy array of maximum loose bound value of every grain
                    along y.
                * zmaxs : np.ndarray
                    Numpy array of maximum loose bound value of every grain
                    along z.

        Explanations
        ------------
        self.spbound provide tight bounds for everyt grain.

        self.spboundex provide loose bounds for every grain, where the bounds
        are extended in each direction by a unit voxel. In case of border
        grains, self.spboundex values along the corresponding directions will
        not be extended.
        """
        print('\nFinding normal and extended spatial bounds of all grains.')
        zmins = np.array([loc[:, 0].min() for loc in self.grain_locs.values()])
        zmaxs = np.array([loc[:, 0].max() for loc in self.grain_locs.values()])
        zmins_ex = zmins - (zmins > 0)*1
        zmaxs_ex = zmaxs + (zmaxs < self.lgi.shape[0]-1)*1
        # -------------------------------
        ymins = np.array([loc[:, 1].min() for loc in self.grain_locs.values()])
        ymaxs = np.array([loc[:, 1].max() for loc in self.grain_locs.values()])
        ymins_ex = ymins - (ymins > 0)*1
        ymaxs_ex = ymaxs + (ymaxs < self.lgi.shape[1]-1)*1
        # -------------------------------
        xmins = np.array([loc[:, 2].min() for loc in self.grain_locs.values()])
        xmaxs = np.array([loc[:, 2].max() for loc in self.grain_locs.values()])
        xmins_ex = xmins - (xmins > 0)*1
        xmaxs_ex = xmaxs + (xmaxs < self.lgi.shape[2]-1)*1
        # -------------------------------
        # Formulate the extended bounding cube bounds.
        self.spbound = {'xmins': xmins, 'xmaxs': xmaxs,
                        'ymins': ymins, 'ymaxs': ymaxs,
                        'zmins': zmins, 'zmaxs': zmaxs}
        self.spboundex = {'xmins': xmins_ex, 'xmaxs': xmaxs_ex,
                          'ymins': ymins_ex, 'ymaxs': ymaxs_ex,
                          'zmins': zmins_ex, 'zmaxs': zmaxs_ex}
        print('Completed', 40*'-')

    def find_bounding_cube_gid(self, gid):
        """
        Find the subset of lgi which tight binds grain gid.

        Parameters
        ----------
        gid : int
            Grain ID.

        Returns
        -------
        lgisubset_tightbound : np.ndarray
        """
        gid = gid-1
        xsl = slice(self.spbound['xmins'][gid], self.spbound['xmaxs'][gid]+1)
        ysl = slice(self.spbound['ymins'][gid], self.spbound['ymaxs'][gid]+1)
        zsl = slice(self.spbound['zmins'][gid], self.spbound['zmaxs'][gid]+1)
        return self.lgi[zsl, ysl, xsl] # lgisubset_tightbound

    def find_exbounding_cube_gid(self, gid):
        """
        Find the subset of lgi which loose binds grain gid by a voxel in each
        of the 3 axes.

        Parameters
        ----------
        gid : int
            Grain ID.

        Returns
        -------
        lgisubset_loosebound : np.ndarray

        gstslice.find_exbounding_cube_gid(1)
        """
        xsl = slice(self.spboundex['xmins'][gid-1],
                    self.spboundex['xmaxs'][gid-1]+1)
        ysl = slice(self.spboundex['ymins'][gid-1],
                    self.spboundex['ymaxs'][gid-1]+1)
        zsl = slice(self.spboundex['zmins'][gid-1],
                    self.spboundex['zmaxs'][gid-1]+1)
        lgisubset_loosebound = self.lgi[zsl, ysl, xsl]
        return lgisubset_loosebound

    def get_bounding_cube_all(self):
        """Find the subsets of lgi which tight binds grains."""
        return {gid: self.find_bounding_cube_gid(gid) for gid in self.gid}

    def get_exbounding_cube_all(self):
        """Find the subsets of lgi which loose binds grains."""
        return {gid: self.find_exbounding_cube_gid(gid) for gid in self.gid}

    def set_gbpoints_global_point_cloud(self, points=np.array([-1, -1, -1])):
        """
        Set Pyvista PolyData with global grain boundary points.

        Parameters
        ----------
        points: np.ndarray, optional
            Numpy array of coordinate points. Defaults to value
            np.array([-1, -1, -1]).

        Save as attribute
        -----------------
        self.pointclouds_pv['gbp_global']

        Returns
        -------
        None
        """
        self.pointclouds_pv['gbp_global'] = pv.PolyData(points)

    def plot_gbpoint_cloud_global(self):
        """
        Plot all the grain boundary points clouds.

        Parameters
        ----------
        None

        Save as attribute
        -----------------
        None

        Variables visualized
        --------------------
        self.pointclouds_pv['gbp_global']

        Returns
        -------
        None
        """
        self.pointclouds_pv['gbp_global'].plot(eye_dome_lighting=True)

    def validate_scalar_field_name(self, sf_name):
        """
        Validate if user input sf_name is a valid sclar field name.

        Parameters
        ----------
        sf_name

        Save as attribute
        -----------------
        None

        Returns
        -------
        None

        Explanations
        ------------
        This definition is mainly for intewrnal use.
        """
        if sf_name not in self.valid_scalar_fields:
            print('Check self.valid_scalar_fields for valid sf names.')
            raise ValueError('Invalid sf_name specification.')

    def get_scalar_field(self,
                         sf_name="lgi",
                         sf_details={'name': 'nneigh.1',
                                     'valby': 'build',  # or 'user'
                                     'norder': 1.5,
                                     'include_parent': False,
                                     'val': {1: None,
                                             2: None},
                         },
                         make_pvgrid=False
                         ):
        """
        Return the requested scalar field.

        Parameters
        ----------
        sf_name : str, optional

        Returns
        -------
        sf_value : np.ndarray / None

        Examples
        --------
        * Pre-requisites

        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 49
        gstslice = pxt.gs[tslice]

        Example - 1: We will build the neighbour of n(O), create the
        scalr field by mapping across lgi and return a pvgrid if desired, all
        in a single function call.

        sf = gstslice.get_scalar_field(sf_name='neigh',
                                       sf_details={'name': 'nneigh.1',
                                                   'valby': 'build',
                                                   'norder': 1.5,
                                                   'include_parent': False,
                                                   'subfield': 'nneigh'
                                                   },
                                       make_pvgrid=True
                                      )
        pvgrid = gstslice.make_pvgrid_v1(feature_name='user',
                                     instance_name='none',
                                     user_fid=sf_no,
                                     scalar_name='sf_no',
                                     pvgrid_origin=(0, 0, 0),
                                     pvgrid_spacing=(1, 1, 1),
                                     perform_checks=False)
        pvgrid.plot(cmap='nipy_spectral')

        """
        self.validate_scalar_field_name(sf_name)

        if sf_name == "lgi":
            sf_value = self.lgi
        elif sf_name == "neigh":
            if sf_details['valby'] == 'user':
                neigh_gid = sf_details['val']
            elif sf_details['valby'] == 'build':
                def_neigh = self.get_upto_nth_order_neighbors_all_grains_prob
                neigh_gid = def_neigh(sf_details['norder'],
                                  recalculate=False,
                                  include_parent=sf_details['include_parent'])
                if sf_details['subfield'] == 'nneigh':
                    nneigh_gid = self.get_neigh_gid_subfield(neigh_gid,
                                                             subfield='nneigh')
                    nneigh_gid = {gid: len(neighs)
                                  for gid, neighs in neigh_gid.items()}
                elif sf_details['subfield'] == 'dist.mean':
                    pass
                elif sf_details['subfield'] == 'vf':
                    pass
            sf_value = self.map_scalar_to_lgi(neigh_gid,
                                              scalar_name=sf_details['name'],
                                              saa=False,
                                              throw=True)
        elif sf_name == "s":
            pass
        elif sf_name == 'fid':
            pass
        else:
            raise ValueError('Invalid sf_name specification.')

        _def_ = self.make_pvgrid_v1
        sfdict = dict(sf=sf_value,
                      pvgrid=_def_(feature_name='user',
                                   instance_name='none',
                                   user_fid=sf_value,
                                   scalar_name=sf_name,
                                   pvgrid_origin=(0, 0, 0),
                                   pvgrid_spacing=(1, 1, 1),
                                   perform_checks=False)
                      if make_pvgrid else None)
        return sfdict

    def get_neigh_gid_subfield(self, neigh_gid, subfield='nneigh'):
        """
        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 49
        gstslice = pxt.gs[tslice]
        def_neigh = gstslice.get_upto_nth_order_neighbors_all_grains_prob
        neigh_gid = def_neigh(1.0, recalculate=False, include_parent=True)

        nneigh = gstslice.get_neigh_gid_subfield(neigh_gid, subfield='nneigh')
        nneigh_vf = gstslice.get_neigh_gid_subfield(neigh_gid, subfield='vf')

        sf_no = gstslice.map_scalar_to_lgi(nneigh_vf,
                                           default_scalar=-1,
                                           make_pvgrid=True)
        sf_no['pvgrid'].plot()
        """
        if subfield == 'nneigh':
            ngidsbfld = {gid: len(neighs)
                          for gid, neighs in neigh_gid.items()}
        elif subfield == 'dist.mean':
            pass
        elif subfield == 'vf':
            neigh_gid = self.remove_gid_from_neigh_gid(neigh_gid)
            nx = (self.uigrid.xmax - self.uigrid.xmin) / self.uigrid.xinc
            ny = (self.uigrid.ymax - self.uigrid.ymin) / self.uigrid.yinc
            nz = (self.uigrid.zmax - self.uigrid.zmin) / self.uigrid.zinc
            volnv_total = nx * ny * nz
            if not self.mprop['volnv']:
                self.set_mprop_volnv(msg='')
            data_neigh_gid = {gid: [self.mprop['volnv'][_gid_]
                                    for _gid_ in neighs]
                              for gid, neighs in neigh_gid.items()}
            ngidsbfld = {gid: np.array(dng).sum()/volnv_total
                              for gid, dng in data_neigh_gid.items()}
        return ngidsbfld

    def remove_gid_from_neigh_gid(self, neigh_gid):
        for gid, neigh in neigh_gid.items():
            if gid in neigh:
                neigh.remove(gid)
            neigh_gid[gid] = neigh
        return neigh_gid

    def map_scalar_to_lgi(self,
                          scalars_dict,
                          default_scalar=-1,
                          scalar_name='nneigh.no1.5',
                          saa=False,
                          throw=True,
                          make_pvgrid=False
                          ):
        """
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 49
        gstslice = pxt.gs[tslice]

        def_neigh = gstslice.get_upto_nth_order_neighbors_all_grains_prob
        neigh = def_neigh(3.2, recalculate=False, include_parent=True)
        nneigh = {gid: len(_nei_) for gid, _nei_ in neigh.items()}
        sf_no = gstslice.map_scalar_to_lgi(nneigh, default_scalar=-1)

        pvgrid = gstslice.make_pvgrid_v1(feature_name='user',
                                     instance_name='none',
                                     user_fid=sf_no,
                                     scalar_name='sf_no',
                                     pvgrid_origin=(0, 0, 0),
                                     pvgrid_spacing=(1, 1, 1),
                                     perform_checks=False)
        pvgrid.plot(cmap='nipy_spectral')

        slices = pvgrid.slice_orthogonal(x=25, y=25, z=25)
        slices.plot(show_edges=True)
        """
        LGI = deepcopy(self.lgi)

        for gid in self.gid:
            if gid in scalars_dict.keys():
                LGI[LGI == gid] = scalars_dict[gid]
            else:
                LGI[LGI == gid] = default_scalar

        if saa:
            self.fdb[scalar_name] = LGI

        if throw:
            sfdict = dict(sf=LGI,
                          pvgrid=self.make_pvgrid_v1(feature_name='user',
                                                  instance_name='none',
                                                  user_fid=LGI,
                                                  scalar_name=scalar_name,
                                                  pvgrid_origin=(0, 0, 0),
                                                  pvgrid_spacing=(1, 1, 1),
                                                  perform_checks=False) if make_pvgrid else None
                          )
            return sfdict

    def get_scalar_field_slice(self, sf_name='lgi', slice_normal='x',
                               slice_location=0, interpolation='nearest'):
        """
        Get scalar field values along the specified slice.

        Parameters
        ----------
        sf_name : str or dth.dt.ITERABLES, optional
            Defaults to 'lgi'.

        slice_normal : str, optional
            Defaults to 'x'.

        slice_location : int, optional
            Defaults to 0.

        interpolation : str, optional
            Defaults to 'nearest'.

        Save as attribute
        -----------------
        None

        Returns
        -------
        sf_slice : np.ndarray
            Scalar field values in the slice.
        """
        sf_value = self.get_scalar_field(sf_name=sf_name)
        if sf_value.ndim != 3:
            raise ValueError('Invalid sf_value dimensions. Expected 3.')
        # ----------------------------------
        if isinstance(slice_normal, str):
            if slice_normal in ('x', 'y', 'z'):
                if slice_normal == 'x':
                    if slice_location >= 0 and slice_location <= sf_value.shape[2]:
                        sf_slice = sf_value[:, :, slice_location]
                    else:
                        raise ValueError('Invalid slice_location specified.')
                elif slice_normal == 'y':
                    if slice_location >= 0 and slice_location <= sf_value.shape[1]:
                        sf_slice = sf_value[:, slice_location, :]
                    else:
                        raise ValueError('Invalid slice_location specified.')
                elif slice_normal == 'z':
                    if slice_location >= 0 and slice_location <= sf_value.shape[0]:
                        sf_slice = sf_value[slice_location, :, :]
                    else:
                        raise ValueError('Invalid slice_location specified.')
            elif slice_normal in ('xy', 'yx'):
                """Slice normal to the xy plane."""
                pass
            elif slice_normal in ('yz', 'zy'):
                """Slice normal to the yz plane."""
                pass
            elif slice_normal in ('zx', 'xz'):
                """Slice normal to the xz plane."""
                pass
        # ----------------------------------
        elif type(slice_normal) in dth.dt.ITERABLES:
            if len(type(slice_normal)) != 3:
                raise ValueError('Invalid slice_normal vector size specified.')
            if interpolation not in ('nearest', 'linear'):
                print('Valid interpolation options are:')
                print("        'nearest' and 'linear'.")
                raise ValueError('Invalid interpolation option specificaion.')
            slice_normal = np.array(slice_normal).norm()
            # Write codes to actually get the slice.
        else:
            raise ValueError("Invalid slice_normal specification.")
        return sf_slice

    def plot_scalar_field_slice_orthogonal(self, sf_name='lgi',
                                           x=5.0, y=5.0, z=5.0):
        """
        Plot the scalar field along three fundamental orthogonal planes.

        Parameters
        ----------
        sf_name : str, optional
            Valid name of the scalar field. Defaults to 'lgi'.

        x : float, optional
            X-coord of the origin of three orthogonal slices. Defaults to 5.

        y : float, optional
            Y-coord of the origin of three orthogonal slices. Defaults to 5.

        z : float, optional
            Z-coord of the origin of three orthogonal slices. Defaults to 5.

        Returns
        -------
        None

        Save as attribute
        -----------------
        None

        Explanations
        ------------
        None
        """
        slices = self.pvgrid.slice_orthogonal(x=x, y=y, z=z)
        slices.plot(show_edges=False)

    def plot_scalar_field_slice(self, sf_name='lgi', slice_normal='x',
                                slice_location=0, interpolation='nearest',
                                vmin=1, vmax=None):
        """
        Plot the scalar field along the specified slice plane.

        Parameters
        ----------
        sf_name : str or optional
            Name of the scalr field. Defaults to 'lgi'.

        slice_normal : str or dth.dt.ITRERABLE, optional
            Either 'x', 'y' or 'z'. Defaults to 'x'.

        slice_location : float, optional
            Defaults to 0.

        interpolation : str, optional
            Defaults to 'nearest'.

        vmin : int, optional
            Defalts to 1.

        vmax : int or None, optional
            Defalts to None.

        Return
        ------
        ax : object
            Matplotlib axis object.

        Explanations
        ------------
        * 1.
        * 2.

        Examples
        --------
        """
        sf_slice = self.get_scalar_field_slice(sf_name=sf_name,
                                               slice_normal=slice_normal,
                                               slice_location=slice_location,
                                               interpolation=interpolation)
        fig, ax = plt.subplots()
        ax.imshow(sf_slice, vmin=vmin, vmax=vmax if vmax else self.n)
        # -------------------------------------------
        if slice_normal == 'x':
            ax.set_xlabel('Y axis'), ax.set_ylabel('Z axis')
        elif slice_normal == 'y':
            ax.set_xlabel('X axis'), ax.set_ylabel('Z axis')
        elif slice_normal == 'z':
            ax.set_xlabel('X axis'), ax.set_ylabel('Y axis')
        # -------------------------------------------
        ax.set_title(f"SF: {sf_name}. SN: {slice_normal}, SL: {slice_location}",
                     fontsize=12)
        return ax

    def plot_scalar_field_slices(self, sf_name='lgi', slice_normal='x',
                                 slice_location=0, interpolation='nearest',
                                 vmin=1, vmax=None, slice_start=0, slice_end=9,
                                 slice_incr=1, nrows=2, ncols=5, ax=None):
        """
        Plot the scalar field along multiple parallel slice planes.

        Parameters
        ----------
        sf_name : str, optional
            Name of the scalr field. Defaults to 'lgi'.

        slice_normal : str or dth.dt.ITRERABLE, optional
            Either 'x', 'y' or 'z'. Defaults to 'x'.

        slice_location : float, optional
            Defaults to 0.

        interpolation : str, optional
            Defaults to 'nearest'.

        vmin : int, optional
            Defalts to 1.

        vmax : int or None, optional
            Defalts to None.

        slice_start : int, optional
            Specify the starting location of the slice plane. Defalts to 0.

        slice_end : int, optional
            Specify the ending location of the slice plane. Defalts to 9.

        slice_incr : int, optional
            Specify the constant incrementation distances of the subsequent
            slice plane. Defalts to 1.

        nrows : int, optional
            Number of subplot rows needed in the Matplotlib figure window.
            Defalts to 2.

        ncols : int, optional
            Number of subplot columns needed in the Matplotlib figure window.
            Defalts to 5.

        ax : Matplotlib axis object, optional
            Matplotlib axis object to plot over. Defalts to None.

        Return
        ------
        ax : object
            Matplotlib axis object.

        Explanations
        ------------
        * 1.
        * 2.

        Example-1
        ---------
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        # ---------------------------------
        slice_numbers = np.arange(slice_start, slice_end+1, slice_incr)
        slice_numbers = np.reshape(slice_numbers, (nrows, ncols))
        # ---------------------------------
        fx = self.get_scalar_field_slice
        for r in range(nrows):
            for c in range(ncols):
                slice_location = slice_numbers[r, c]
                # -------------------------
                sf_slice = fx(sf_name=sf_name, slice_normal=slice_normal,
                              slice_location=slice_location,
                              interpolation=interpolation)
                # -------------------------
                ax[r, c].imshow(sf_slice, vmin=vmin if vmin else 0,
                                vmax=vmax if vmax else self.n)
                # -------------------------
                if slice_normal == 'x':
                    ax[r, c].set_xlabel('Y axis')
                    ax[r, c].set_ylabel('Z axis')
                elif slice_normal == 'y':
                    ax[r, c].set_xlabel('X axis')
                    ax[r, c].set_ylabel('Z axis')
                elif slice_normal == 'z':
                    ax[r, c].set_xlabel('X axis')
                    ax[r, c].set_ylabel('Y axis')
                # -------------------------
                ts = f"SF: {sf_name}. SN: {slice_normal}, SL: {slice_location}"
                ax[r, c].set_title(ts, fontsize=12)
        return ax

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
        parent_gid : int
            Grain ID of the parent.

        other_gid : int
            Grain ID of the other grain being merged into the parent.

        check_for_neigh : bool, optional
            If True, other_gid will be checked if it can be merged to the
            parent grain. Defaults to True.

        simple_merge : True, optional
            If True, perform a simple merging operation, else open uip for
            more complex merging opertations.

        Explanations
        ------------

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
            fx_cfn = self.check_for_neigh
            if check_for_neigh and not fx_cfn(parent_gid, other_gid):
                # print('Check for neigh failed. Nothing merged.')
                merge_success = False
            # ---------------------------------------
            if any((check_for_neigh, fx_cfn(parent_gid, other_gid))):
                merge_success = MergeGrains()
                # print(f"Grain {other_gid} merged with grain {parent_gid}.")
        return merge_success

    def perform_post_grain_merge_ops(self, merged_gid):
        """
        Perform necessary operations after performing a grain merger operation.

        Parameters
        ----------
        merged_gid

        Operations done
        ---------------
        The following variables are renumbered:
            * lgi
        The following variables / databases are recalulated:
            * self.gid
            * self.n
            * neighbouring gid database
        """
        self.renumber_gid_post_grain_merge(merged_gid)
        self.recalculate_ngrains_post_grain_merge()
        self.renumber_lgi_post_grain_merge()
        # Update neigh_gid

    def renumber_gid_post_grain_merge(self, merged_gid):
        """
        Renumber the grain ID numbers after grain merger operation.

        Parameters
        ----------
        merged_gid : int
            gid vale which has been merged.

        Save as attribute
        -----------------
        gid

        Returns
        -------
        None
        """
        GID_left = self.gid[0:merged_gid-1]
        GID_right = [gid-1 for gid in self.gid[merged_gid:]]
        self.gid = GID_left + GID_right

    def recalculate_ngrains_post_grain_merge(self):
        """
        Renumber the grain ID numbers after grain merger operation.

        Parameters
        ----------
        merged_gid : int
            gid vale which has been merged.

        Save as attribute
        -----------------
        n

        Returns
        -------
        None

        Function order
        --------------
        Secondary. Involves call to a primary function.
        """
        self.calc_num_grains()

    def renumber_lgi_post_grain_merge(self, merged_gid):
        """
        Renumber the lgi array after grain merger operation.

        Parameters
        ----------
        merged_gid : int
            gid vale which has been merged.

        Save as attribute
        -----------------
        lgi

        Returns
        -------
        None
        """
        LGI_left = self.lgi[self.lgi < merged_gid]
        self.lgi[self.lgi > merged_gid] -= 1

    def plot_largest_grain(self):
        """
        A humble method to just plot the largest grain in a temporal slice
        of a grain structure

        Parameters
        ----------
        None

        Returns
        -------
        None.

        # TODO: WRAP THIS INSIDE A FIND_LARGEST_GRAIN AND HAVE IT TRHOW
        THE GID TO THE USER

        """
        gid = self.prop['area'].idxmax()+1
        # self.g[gid]['grain'].plot()  # <-- Replace by 3D plot function.

    def plot_longest_grain(self):
        """
        A humble method to just plot the longest grain in a temporal slice
        of a grain structure

        Parameters
        ----------
        None

        Returns
        -------
        None.

        # TODO: WRAP THIS INSIDE A FIND_LONGEST_GRAIN AND HAVE IT TRHOW
        THE GID TO THE USER
        """
        gid, _, _ = self.get_gid_prop_range(PROP_NAME='aspect_ratio',
                                            range_type='percentage',
                                            percentage_range=[100, 100],
                                            )
        # plt.imshow(self.g[gid[0]]['grain'].bbox_ex)
        # <-- Replace by 3D plot function.
        '''for _gid_ in gid:
            plt.figure()
            self.g[gid]['grain'].plot()
            plt.show()'''

    def mask_lgi_with_gids(self, gids, masker=-10):
        """
        Mask lgi against user input grain indices with a masker value.

        Mask the lgi (PXGS.gs[n] specific lgi array: lattice of grain IDs)
        against user input grain indices, with a default UPXO-reserved
        place-holder value of -10.

        Parameters
        ----------
        gids : int or list
            Either a single grain index number or list of them
        masker : int, optional
            An int value, preferably -10, but compulsorily less than -5.
            Default UPXO-reserved place-holder value of -10.

        Returns
        -------
        s_masked : np.ndarray(dtype=int)
            lgi masked against gid values

        Internal calls (@dev)
        ---------------------
        None
        """
        lgi_masked = deepcopy(self.lgi).astype(int)
        for gid in gids:
            if gid in self.gid:
                lgi_masked[lgi_masked == gid] = masker
            else:
                print(f"Invalid gid: {gid}. Skipped")
        # -----------------------------------------
        return lgi_masked, masker

    def mask_s_with_gids(self, gids, masker=-10, force_masker=False):
        """
        Mask the s against user input grain indices.

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
            of having different masker values, example using differnet
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

    def plot_grains(self, gids, scalar='lgi', cmap='viridis',
                    style='surface', show_edges=True, lw=1.0,
                    opacity=0.8, view=None, scalar_bar_args=None,
                    plot_coords=False, coords=None,
                    axis_labels = ['z', 'y', 'x'], explode=0.0,
                    pvp=None, throw=False):
        """
        Plot grains given some gids.

        Parameters
        ----------
        gids : int or iterable
            Grain ID number(s).

        scalar : np.array or valid string, optional
            Defaults to 'lgi'.

        cmap : str, optional
            Defaults to 'viridis'.

        style : str, optional
            Options for style: 'surface', 'wireframe', 'points' and
            'points_gaussian' Defaults to 'surface'.

        show_edges : bool, optional
            Defaults to True.

        lw : float, optional
            Line width. Defaults to 1.0.

        opacity : float on/in [0.0, 1.0], optional
            Options for opacity include foollowing:
                * int between 0 and 1
                * Opacity transfer functions: 'linear', 'linear_r', 'geom',
                    'geom_r', 'sigmoid', 'sigmoid_r'
                * Custom transfewr function: list of values between 0 and 1,
                    example: opacity = [0, 0.2, 0.9, 0.6, 0.3]. In ythis case,
                    these values will be linearly mapped to the scalr being
                    plotted.
            Defaults to 1.0.

        view : str / None, optional
            To be implemented. Defaults to None.

        scalar_bar_args : dict, optional
            To be implemented. Defaults to None.

        plot_coords : bool, optional
            Plot additional coordinate points. Defaults to False.

        coords : np.ndarray/None, optional
            Numpy array of coordinate points. Defaults to None.

        axis_labels : list, optional
            Label strings for x, y and z - axis labels. Defaults to
            ['z', 'y', 'x'].

        pvp : PyVista plotter object / None, optional
            PyVista plotter object to plot over. Defaults to None.

        throw : bool, optional
            If True, pv.Plotter() instance shall be returned without actually
            plotting visually. Defaults to False.

        Example-1
        ---------
        gids = gstslice.gpos['boundary']-gstslice.gpos['face']['top']
        gstslice.plot_grains(gids, scalar='lgi',
                             cmap='viridis',
                             style='surface', show_edges=True, lw=1.0,
                             opacity=1, view=None,
                             scalar_bar_args=None,
                             axis_labels = ['001', '010', '100'],
                             throw=False)

        Examples for extracting gids
        ----------------------------
        1. All corber grains
           gids = gstslice.gpos['corner']['all']

        2. All grains sharing atleast a pixel with bottom face.
           gids = gstslice.gpos['face']['bottom']

        3. All grains sharing atleast a pixel with all 4 edges of front face.
           gids = gstslice.gpos['edges']['front']

        4. All grains nt sharing even a single pixel with any of the 6 faces.
           gids = gstslice.gpos['internal']

        5. Grains which share atleast a pixel with any of the edges of each of
           the 6 faces.
           gids = gstslice.gpos['edges']['left'].union(
               gstslice.gpos['edges']['right'],
               gstslice.gpos['edges']['back'],
               gstslice.gpos['edges']['front'],
               gstslice.gpos['edges']['bottom'],
               gstslice.gpos['edges']['top'])

        6. Grains sharing atleast a pixel with each of the 6 face, but not
           the 'bottom_front' and 'top_front' edges.
           global_set = gstslice.gpos['boundary']
           to_remove = [gstslice.gpos['edges']['bottom_front'],
                        gstslice.gpos['edges']['top_front']]
           gids =  global_set - to_remove[0].union(to_remove[1])
        """
        if pvp is None or not isinstance(pvp, pv.Plotter):
            pvp = pv.Plotter()
        # -------------------------------------
        ngids = len(gids)
        for i, gid in enumerate(gids, start=1):
            if i % 100 == 0:
                print(f"Adding feature ID no. {i} of {ngids}")
            pvp.add_mesh(self.pvgrid.threshold([gid, gid], scalars=scalar),
                         show_edges=show_edges, line_width=lw,
                         style=style, opacity=opacity, cmap=cmap)
        # -------------------------------------
        box_bounds = (0, self.lgi.shape[2], 0, self.lgi.shape[1],
                      0, self.lgi.shape[0])
        # -------------------------------------
        pvp.add_mesh(pv.Box(bounds=box_bounds, level=0), show_edges=show_edges,
                     line_width=2.5, color='black', style='wireframe',
                     opacity=opacity, cmap=cmap)
        # -------------------------------------
        pvp.add_axes(xlabel=axis_labels[0], ylabel=axis_labels[1],
                     zlabel=axis_labels[2], label_size=(0.4, 0.16))
        # -------------------------------------
        if plot_coords and coords is not None:
            coords = np.array(coords)
            coord_pd = pv.PolyData(coords)
            pvp.add_mesh(coord_pd, point_size=12)
            _ = pvp.add_axes(line_width=5,
                             cone_radius=0.6,
                             shaft_length=0.7,
                             tip_length=0.3,
                             ambient=0.5,
                             label_size=(0.4, 0.16))
        # -------------------------------------
        # pvp.set_background('white')
        if throw:
            return pvp
        else:
            pvp.show()

    def viz_browse_grains(self, scalar='lgi', cmap='viridis',
                          style='surface', show_edges=True, lw=1.0,
                          opacity=0.8, view=None, scalar_bar_args=None,
                          plot_coords=False, name='UPXO.MCGS.3D',
                          coords=None, axis_labels = ['z', 'y', 'x'],
                          title='Grain ID', add_outline=False, pvp=None,
                          throw=False):
        """
        Browse grains in the grain structrure using a slider.

        Parameters
        ----------
        gids : int or iterable
            Grain ID number(s).

        scalar : np.array or valid string, optional
            Defaults to 'lgi'.

        cmap : str, optional
            Defaults to 'viridis'.

        style : str, optional
            Options for style: 'surface', 'wireframe', 'points' and
            'points_gaussian' Defaults to 'surface'.

        show_edges : bool, optional
            Defaults to True.

        lw : float, optional
            Line width. Defaults to 1.0.

        opacity : float on/in [0.0, 1.0], optional
            Options for opacity include foollowing:
                * int between 0 and 1
                * Opacity transfer functions: 'linear', 'linear_r', 'geom',
                    'geom_r', 'sigmoid', 'sigmoid_r'
                * Custom transfewr function: list of values between 0 and 1,
                    example: opacity = [0, 0.2, 0.9, 0.6, 0.3]. In ythis case,
                    these values will be linearly mapped to the scalr being
                    plotted.
            Defaults to 1.0.

        view : str / None, optional
            To be implemented. Defaults to None.

        scalar_bar_args : dict, optional
            To be implemented. Defaults to None.

        plot_coords : bool, optional
            Plot additional coordinate points. Defaults to False.

        coords : np.ndarray / None, optional
            Numpy array of coordinate points. Defaults to None.

        axis_labels : list, optional
            Label strings for x, y and z - axis labels. Defaults to
            ['z', 'y', 'x'].

        title : str
            Defaults to 'Grain ID'.

        add_outline : bool
            Defaults to False.

        pvp : PyVista plotter object / None, optional
            PyVista plotter object to plot over. Defaults to None.

        throw : bool, optional
            If True, pv.Plotter() instance shall be returned without actually
            plotting visually. Defaults to False.
        """
        if pvp is None or not isinstance(pvp, pv.Plotter):
            pvp = pv.Plotter()
        # -------------------------------------
        if add_outline:
            pvp.add_mesh(self.pvgrid.outline())
        # -------------------------------------
        def create_mesh(gid):
            gid = int(gid)
            pvp.add_mesh(self.pvgrid.threshold([gid, gid],
                                               scalars='lgi'),
                         name=name,
                         show_edges=True)
            return
        # -------------------------------------
        pvp.add_slider_widget(create_mesh, [1, self.n], title=title)
        # -------------------------------------
        if throw:
            return pvp
        else:
            pvp.show()

    def viz_clip_plane(self, normal='x', origin=[5.0, 5.0, 5.0], scalar='lgi',
                       cmap='viridis', invert=True, crinkle=True,
                       normal_rotation=True, add_outline=False, throw=False,
                       pvp=None):
        """
        Visualize grain structure along a clip plane.

        Parameters
        ----------
        normal : str or dth.dt.ITERABLE(float), optional
            Normal specification of clipping plane. Default value is 'x'.

        origin : dth.dt.ITERABLE(float), optional
            Specification of origin, that is clip plane centre coordinate.

        scalar : str, optional
            self.pvgrid cell_data scalar specification. Default value is 'lgi'.

        cmap : str, optional
            Colour map specification. Default value is 'viridis'.
            Recommended values:
                * viridis
                * nipy_spectral

        invert : bool, optional
            Invert clip sense if True, dont if False. Default value is True.

        crinkle : bool, optional
            Crinkle view voxels if True, section view if False. Default value
            is True.

        normal_rotation : bool, optional
            Rotation specification of normal. Default value is True.
            NOTE: To be implemented completely.

        add_outline : bool, optional
            Add an outline around the grain structure. Default value is False.

        throw : bool, optional
            Throw the pvp if True, dont if False. Default value is False.

        pvp : bool, optional
            PyVista plotter object to plot over. If no pvp has been provided,
            new pvp shall be created. Default value is None.

        Example-1
        ---------
        Example with pvp None

        Example-2
        ---------
        Example with valid pvp input.
        """
        if pvp is None or not isinstance(pvp, pv.Plotter):
            pvp = pv.Plotter()
        # -------------------------------------
        if add_outline:
            pvp.add_mesh(self.pvgrid.outline())
        # -------------------------------------
        pvp.add_mesh_clip_plane(self.pvgrid, normal=normal, origin=origin,
                                scalars=scalar, cmap=cmap, invert=invert,
                                crinkle=crinkle,
                                normal_rotation=normal_rotation, tubing=False,
                                interaction_event=self._vtk_ievnt_)
        # -------------------------------------
        if throw:
            return pvp
        else:
            pvp.show()

    def viz_mesh_slice(self, normal='x', origin=[5.0, 5.0, 5.0], scalar='lgi',
                       cmap='viridis', normal_rotation=True, add_outline=False,
                       throw=False, pvp=None):
        """
        Visualize grain structure along a slice plane.

        Parameters
        ----------
        normal : str or dth.dt.ITERABLE(float), optional
            Normal specification of clipping plane. Default value is 'x'.

        origin : dth.dt.ITERABLE(float), optional
            Specification of origin, that is clip plane centre coordinate.

        scalar : str, optional
            self.pvgrid cell_data scalar specification. Default value is 'lgi'.

        cmap : str, optional
            Colour map specification. Default value is 'viridis'.
            Recommended values:
                * viridis
                * nipy_spectral

        add_outline : bool, optional
            Add an outline around the grain structure. Default value is False.

        throw : bool, optional
            Throw the pvp if True, dont if False. Default value is False.

        pvp : bool, optional
            PyVista plotter object to plot over. If no pvp has been provided,
            new pvp shall be created. Default value is None.

        Example-1
        ---------
        Example with pvp None

        Example-2
        ---------
        Example with valid pvp input.
        """
        if pvp is None or not isinstance(pvp, pv.Plotter):
            pvp = pv.Plotter()
        # -------------------------------------
        if add_outline:
            pvp.add_mesh(self.pvgrid.outline())
        # -------------------------------------
        pvp.add_mesh_slice(self.pvgrid, scalars=scalar,
                           normal=normal, origin=origin, cmap=cmap,
                           normal_rotation=False,
                           interaction_event=self._vtk_ievnt_)
        # -------------------------------------
        if throw:
            return pvp
        else:
            pvp.show()

    def viz_mesh_slice_ortho(self, scalar='lgi', cmap='viridis',
                             style='surface', add_outline=False,
                             throw=False, pvp=None):
        """
        Viz. grain str. along three fundamental mutually orthogonal planes.

        Parameters
        ----------
        scalar : str, optional
            self.pvgrid cell_data scalar specification. Default value is 'lgi'.

        cmap : str, optional
            Colour map specification. Default value is 'viridis'.
            Recommended values:
                * viridis
                * nipy_spectral

        add_outline : bool, optional
            Add an outline around the grain structure. Default value is False.

        throw : bool, optional
            Throw the pvp if True, dont if False. Default value is False.

        pvp : bool, optional
            PyVista plotter object to plot over. If no pvp has been provided,
            new pvp shall be created. Default value is None.

        Example-1
        ---------
        Example with pvp None

        Example-2
        ---------
        Example with valid pvp input.
        """
        if pvp is None or not isinstance(pvp, pv.Plotter):
            pvp = pv.Plotter()
        # -------------------------------------
        if add_outline:
            pvp.add_mesh(self.pvgrid.outline())
        # -------------------------------------
        pvp.add_mesh_slice_orthogonal(self.pvgrid, scalars=scalar,
                                      style=style, cmap=cmap,
                                      interaction_event=self._vtk_ievnt_)
        # -------------------------------------
        if throw:
            return pvp
        else:
            pvp.show()

    def plot_grain_sets(self, data=None, scalar='lgi',
                        opacities=[1.00, 0.90, 0.75, 0.50],
                        pvp=None, cmap='viridis', style='surface',
                        show_edges=True, lw=1.0, plot_coords=False,
                        coords=None, core_grains_opacity=1, opacity=1,
                        view=None, scalar_bar_args=None,
                        axis_labels=['001', '010', '100'], throw=False,
                        validate_data=True):
        """
        Plot multiple prominant and non-prominant grains.

        Parameters
        ----------
        data : dict, optional
            keys: str
                * 'cores': list of ints
                    List of most prominant gids.
                * 'other': list of list
                    Fi4rst lisyt - gids with next lesser prominance level
                    than thosse in 'core'.
                    Second list - gids with next lesser prominance level than
                    thse in first list.

        opacities : list, optional
            First value is alpha of most prominant grains. Could represent
                alpha of core grains.
            Second value is alpha of grains with the next prominance level.
                Could represent alpha of every 1st order neighbours.
            Third value, fourth value and so on.
            Defaults to [1.00, 0.90, 0.75, 0.50]

        pvp : bool, optional
            PyVista plotter object to plot over. If no pvp has been provided,
            new pvp shall be created. Default value is None.

        cmap : str, optional
            Colour map specification. Default value is 'viridis'.
            Recommended values:
                * viridis
                * nipy_spectral

        style : str, optional
            Options for style: 'surface', 'wireframe', 'points' and
            'points_gaussian' Defaults to 'surface'.

        show_edges : bool, optional
            Defaults to True.

        lw : float, optional
            Line width. Defaults to 1.0.

        plot_coords : bool, optional
            Plot additional coordinate points. Defaults to False.

        coords : np.ndarray or None, optional
            Numpy array of coordinate points. Defaults to None.

        core_grains_opacity : float, optional
        Specify opacity of core grains. Use this to visualize cases where you
        need to visualize a group of grains surrounding core grain(s). Defaults
        to 1.0.

        opacity : float, optional
        . Defaults to 1

        view : type, optional
            Viz. view specification. Defaults to None.

        scalar_bar_args : type, optional
            To be implemented. Defaults to None.

        axis_labels : list, optional
            Coordinate axes label string specification. Defaults to
            ['001', '010', '100'].

        throw : bool, optional
            Return plotter object if True, dont if False. Defaults to False.

        validate_data : bool, optional
            Initially validate user inputs if True, skip if False. Defaults to
            True.

        Example template for data
        -------------------------
        data = {'core': [1, 2, 3, 4], 'other': [[7, 6, 5], [12, 8, 10, 11]]}

        Example-1
        ---------
        Basic example

        Example-2
        ---------
        Visualize additional coordinate points.

        Example-3
        ---------
        Visualize core and non-core grains.

        Example-4
        ---------
        Visualize core and non-core grains along with additional coordinate
        points.

        Example-5
        ---------
        Visualize a core grain along with additional coordinate point sets
        proivided as a dict input format.
        """
        # Validations
        if validate_data:
            if not all(isinstance(d, list) for d in data['others']):
                raise ValueError('Invalid data specification.')
            if not all(len(d)>0 for d in data['others']):
                raise ValueError('Invalid data specirication')
        # -----------------------------
        # Frst we will deal with most prominant grains.
        core_grains = data['cores']
        # -------------------------------------
        pvp = self.plot_grains(core_grains, scalar=scalar,
                               cmap=cmap,
                               style='surface',
                               show_edges=show_edges, lw=lw,
                               opacity=core_grains_opacity, view=None,
                               scalar_bar_args=scalar_bar_args,
                               axis_labels=axis_labels, pvp=pv.Plotter(),
                               throw=True)
        # -----------------------------
        other_grains_opacity = np.ones(len(data['others'])) * 0.5
        for i, o in enumerate(opacities[1:]):
            if i == len(data['others']):
                break
            other_grains_opacity[i] = o
        # -----------------------------
        # Add each grain set to current visualization dataset, pvp.
        if all((len(data['others']) == 1,
                type(data['others'][0]) in dth.dt.NUMBERS)):
            data['others'] = [data['others']]
        for i, gidlist in enumerate(data['others']):
            pvp = self.plot_grains(gidlist, scalar=scalar,
                                   cmap=cmap,
                                   style=style,
                                   show_edges=show_edges, lw=lw,
                                   opacity=other_grains_opacity[i],
                                   view=None,
                                   scalar_bar_args=scalar_bar_args,
                                   axis_labels=axis_labels, pvp=pvp,
                                   throw=True)
        # -----------------------------
        if plot_coords and coords is not None:
            '''If user wiushes to plot additional coordinates and also that the
            the actual coordinate point data has been provided.'''
            if type(coords) in dth.dt.ITERABLES:
                '''Validate the user provided coordinate data.'''
                # VALIDATION
                # -----------------------
                coords = np.array(coords)
                coord_pd = pv.PolyData(coords)
                pvp.add_mesh(coord_pd, point_size=12)
                _ = pvp.add_axes(line_width=5,
                                 cone_radius=0.6,
                                 shaft_length=0.7,
                                 tip_length=0.3,
                                 ambient=0.5,
                                 label_size=(0.4, 0.16))
            elif isinstance(coords, dict):
                '''Validate the user provided coordinate data.'''
                # VALIDATION
                # -----------------------
                _R_ = np.random.random
                keys = list(coords.keys())
                pvp.add_mesh(pv.PolyData(np.array(coords[keys[0]])),
                             point_size=12,
                             color=_R_(3))
                for k in keys[1:]:
                    pvp.add_mesh(pv.PolyData(np.array(coords[k])),
                                 point_size=12,
                                 color=_R_(3))
                _ = pvp.add_axes(line_width=5,
                                 cone_radius=0.6,
                                 shaft_length=0.7,
                                 tip_length=0.3,
                                 ambient=0.5,
                                 label_size=(0.4, 0.16))
        # -----------------------------
        if throw:
            return pvp
        else:
            pvp.show()

    def find_scalar_array_in_plane(self, origin=[5.0, 5.0, 5.0],
                                   normal=[1.0, 1.0, 1.0], scalar='lgi'):
        """
        Get the scalar values array in a plane.

        Parameters
        ----------
        origin : list
            Define the origin of the slicing plane as [i, j, k].

        normal : list
            Define the normal vector of the slicing plane as [u, v, w].

        Return
        ------
        lgi_array

        saa
        ---
        None

        Explanations
        ------------
        The returned array is a 1D numpy array of all scalar values. If the
        unique set of valures is preferred use np.unique over the returned
        value or use the get_scalar_array_in_plane_unique function having the
        same input arguments.

        Example
        -------
        gstslice.find_scalar_array_in_plane(origin=[5, 4, 3], normal=[1, 2, 1],
                                            scalar='lgi')
        """
        lgi_array = self.pvgrid.slice(origin=origin,
                                      normal=normal).get_array('lgi')
        return lgi_array

    def get_scalar_array_in_plane_unique(self, origin=[5.0, 5.0, 5.0],
                                         normal=[1.0, 1.0, 1.0]):
        """
        Find unique gids in a plane defined by origin and normal.

        Parameters
        ----------
        origin : list
            Define the origin of the slicing plane as [i, j, k].

        normal : list
            Define the normal vector of the slicing plane as [u, v, w].

        Return
        ------
        lgi_array

        saa
        ---
        None

        Explanations
        ------------
        The returned array is a uniqued 1D numpy array of all scalar values.
        If all valures is preferred use the get_scalar_array_in_plane_unique
        function having the same input arguments.

        Examples
        --------
        gstslice.find_scalar_array_in_plane(origin=[5, 4, 3], normal=[1, 2, 1],
                                            scalar='lgi')

        Refer to the examples provided in the documentation of definition
        plot_gids_along_plane for some applications of this function
        get_scalar_array_in_plane_unique.
        """
        gids = self.find_scalar_array_in_plane(origin=origin, normal=normal)
        gids = np.array(np.unique(gids).tolist())
        return gids

    def plot_gids_along_plane(self, origin=[5.0, 5.0, 5.0],
                              normal=[1.0, 1.0, 1.0], cmap='viridis',
                              style='surface', show_edges=True,
                              lw=1.0, opacity=0.8, view=None,
                              scalar_bar_args=None, plot_coords=False,
                              coords=None, axis_labels=['z', 'y', 'x'],
                              pvp=None, throw=False):
        """
        Plot grains which fall alomng a plane.

        Parameters
        ----------
        origin : list
            Define the origin of the slicing plane as [i, j, k].

        normal : list
            Define the normal vector of the slicing plane as [u, v, w].

        cmap : str, optional
            Defaults to 'viridis'.

        style : str, optional
            Options for style: 'surface', 'wireframe', 'points' and
            'points_gaussian' Defaults to 'surface'.

        show_edges : bool, optional
            Defaults to True.

        lw : float, optional
            Line width. Defaults to 1.0.

        opacity : float on/in [0.0, 1.0], optional
            Options for opacity include foollowing:
                * int between 0 and 1
                * Opacity transfer functions: 'linear', 'linear_r', 'geom',
                    'geom_r', 'sigmoid', 'sigmoid_r'
                * Custom transfewr function: list of values between 0 and 1,
                    example: opacity = [0, 0.2, 0.9, 0.6, 0.3]. In ythis case,
                    these values will be linearly mapped to the scalr being
                    plotted.
            Defaults to 1.0.

        view : str / None, optional
            To be implemented. Defaults to None.

        scalar_bar_args : dict, optional
            To be implemented. Defaults to None.

        plot_coords : bool, optional
            Plot additional coordinate points. Defaults to False.

        coords : np.ndarray / None, optional
            Numpy array of coordinate points. Defaults to None.

        axis_labels : list, optional
            Label strings for x, y and z - axis labels. Defaults to
            ['z', 'y', 'x'].

        title : str
            Defaults to 'Grain ID'.

        add_outline : bool
            Defaults to False.

        pvp : PyVista plotter object / None, optional
            PyVista plotter object to plot over. Defaults to None.

        throw : bool, optional
            If True, pv.Plotter() instance shall be returned without actually
            plotting visually. Defaults to False.

        Example
        -------
        gstslice.plot_gids_along_plane(origin=[5, 5, 5], normal=[1, 0, 0],
                                  cmap='viridis', style='surface',
                                  show_edges=True, lw=1.0, opacity=1.0,
                                  view=None, scalar_bar_args=None,
                                  plot_coords=False, coords=None,
                                  axis_labels=['z', 'y', 'x'], pvp=None,
                                  throw=False)

        Longer example-1
        ----------------
        '''Find the grain IDs first.'''
        gids = gstslice.get_scalar_array_in_plane_unique(origin=[25, 25, 25],
                                                         normal=[1, 1, 1])
        # .... .... .... .... ....
        # NOTE: We can go through any one of the folloiwnwg routes.

        '''Route 1: plot all these gids'''
        gids = gids
        '''Route 2: Exclude boundary grains.'''
        gids = set(gids) - gstslice.gpos['boundary']
        '''Route 3: Consider only boundary grains.'''
        gids = set(gids).intersection(gstslice.gpos['boundary'])
        # .... .... .... .... ....
        '''Now, the actual plotting procesure.'''
        gstslice.plot_grains(gids, scalar='lgi',cmap='viridis',style='surface',
                             show_edges=True, lw=1.0, opacity=1.0, view=None,
                             scalar_bar_args=None, plot_coords=False,
                             coords=None, axis_labels=['z', 'y', 'x'],
                             pvp=None, throw=False)

        Longer example-2
        ----------------
        gids1 = gstslice.get_scalar_array_in_plane_unique(origin=[25, 25, 25],
                                                          normal=[1, 1, 1])
        gids2 = gstslice.get_scalar_array_in_plane_unique(origin=[25, 25, 25],
                                                          normal=[1, -1, 1])
        gids3 = gstslice.get_scalar_array_in_plane_unique(origin=[25, 25, 25],
                                                          normal=[1, -1, -1])
        gids = set(gids1).union(gids2, gids3)
        gids = set(gids1).intersection(gids2, gids3)
        gids = set(gids1).union(gids2, gids3) - set(gids1).intersection(gids2,
                                                                        gids3)

        gids = set(gids1).union(gids2, gids3)
        gids = gids.intersection(gstslice.gpos['boundary'])

        gstslice.plot_grains(gids, scalar='lgi',cmap='viridis',style='surface',
                             show_edges=True, lw=1.0, opacity=1.0, view=None,
                             scalar_bar_args=None, plot_coords=False,
                             coords=None, axis_labels=['z', 'y', 'x'],
                             pvp=None, throw=False)
        """
        self.plot_grains(self.get_scalar_array_in_plane_unique(origin=origin,
                                                               normal=normal,
                                                               scalar='lgi'),
                         scalar='lgi', cmap=cmap, style=style,
                         show_edges=show_edges, lw=lw, opacity=opacity,
                         view=view, scalar_bar_args=scalar_bar_args,
                         plot_coords=plot_coords, coords=coords,
                         axis_labels=axis_labels, pvp=pvp, throw=throw)

    @staticmethod
    def _compute_volumes_with_bincount(lgi, n):
        """
        Calculate the volume by number of voxels using Numba and bincount.
        """
        return np.bincount(lgi.ravel(), minlength=n+1)

    def set_mprop_volnv(self, msg=None):
        """
        Calculate the volume by number of voxels.
        """
        if msg is not None:
            if isinstance(msg, int):
                msg = str(msg)
        else:
            msg = ''

        print(40*"-", "\nSetting grain volumes (metric: 'volnv') -> " + msg)
        unique_counts = self._compute_volumes_with_bincount(self.lgi, self.n)
        self.mprop['volnv'] = {gid + 1: unique_counts[gid + 1] for gid in range(self.n)}
        print("Grain volumes (metric: 'volnv') -> " + msg + ": have been set.\n",
              40*"-")

    def get_bbox_diagonal_vectors(self):
        """
        Find the vector representing doiagonal of the bounding box.
        """
        pass

    def get_bbox_aspect_ratio(self, gid):
        pass

    def get_bbox_volume(self, gid):
        pass

    def set_mprop_volnv_old(self):
        """
        Calculate the volume by number of voxels.
        TO BE NUMBAfied
        """
        print(40*"-", "\nSetting grain volumes (metric: 'volnv').")
        self.mprop['volnv'] = {gid+1: np.argwhere(self.lgi == gid+1).shape[0]
                               for gid in range(self.n)}

    def set_mprop_pernv(self):
        """Calculate the total perimeter of the grain by number of voxels."""
        print(40*"-", "\nSetting grain perimeter values (metric: 'pernv').")
        self.mprop['pernv'] = None

    def get_voxel_volume(self):
        """Return voxel volume from pvgrid data."""
        return np.prod(self.pvgrid.spacing)

    def get_voxel_surfareas(self, ret_metric='mean'):
        """
        Return voxel surface area from pvgrid data.

        Parameters
        ----------
        ret_metric:
            Stands for return. Specifies which metric 8is to be returned.

        Explanations
        ------------
        """
        sp = self.pvgrid.spacing
        if ret_metric == 'mean':
            return (sp[0]*sp[1] + sp[1]*sp[2] + sp[2]*sp[0])/3.0
        elif ret_metric == 'min':
            return min(sp[0]*sp[1], sp[1]*sp[2], sp[2]*sp[0])
        elif ret_metric == 'max':
            return max(sp[0]*sp[1], sp[1]*sp[2], sp[2]*sp[0])
        elif ret_metric == 'all':
            return [sp[0]*sp[1], sp[1]*sp[2], sp[2]*sp[0]]

    def set_mprop_eqdia(self, base_size_spec='ignore',
                        use_skimrp=True, reset_skimrp=True,
                        measure='normal'):
        """
        Calculate equivalent sphere diameter.

        Parameters
        ----------
        base_size_spec: str, optional
            Base size specification used to calculate equivalent sphere
            diameter. Allows to use either volume or surface area. Options are:
                * 'volnv': Volume by number of voxels
                * 'volsr': Volume by surface reconstruction
                * 'volch': Volume of convex hull
                * 'sanv': surface ares by number of voxels
                * 'savi': surface area by voxel interfaces
                * 'sasr': surface area by surface reconstruction
            In case of 'volnv', volume is scaled by unit voxel volume before
            calculation of equivalent sphere diameter.
            In case of 'sanv', volume is scaled by mean unit voxel face area
            before calculation of equivalent sphere diameter.
            Defaults to 'volnv'.

        use_skimrp : bool, optional
        Defaults to True.

        reset_skimrp : bool, optional
        Defaults to True.

        measure : str, optional
        Defaults to 'normal'.

        Explanations
        ------------
        If base_size_spec in ('volnv', 'volsr', 'volch'), then the following
        procedure is used to calculate the equivakent diameter.
        V = (4/3) pi r^3 ==> r^3 = 3V/(4 pi) ==> r = cbrt(3V/(4 pi))
        d = 0.5*cbrt(3V/(4 pi)), where V is the volume measure.

        If base_size_spec in ('sanv', 'savi', 'sasr'), then the following
        procedure is used to calculate the equivakent diameter.
        S = 4 pi r^2 ==> r^2 = S/(4 pi) ==> r = sqrt(S/(4 pi))
        d = 0.5*sqrt(S/(4 pi)), where S is the surface area measure.
        """
        print(40*"-", "\nSetting grain eq.sph.dia. values (metric: 'eqdia').")
        if base_size_spec not in ('volnv', 'volsr', 'volch',
                                  'sanv', 'savi', 'sasr', 'ignore'):
            raise ValueError('Invalid metric specification.')
        if use_skimrp:
            if any((self.skimrp is None, reset_skimrp)):
                self.set_skimrp()
            self.mprop['eqdia'] = {}
            self.mprop['eqdia']['skimrp_used'] = True
            self.mprop['eqdia']['measure'] = measure
            if measure == 'normal':
                self.mprop['eqdia']['values'] = np.array([self.skimrp[gid].equivalent_diameter_area
                                                 for gid in self.gid])
            elif measure == 'feret':
                self.mprop['eqdia']['values'] = np.array([self.skimrp[gid].feret_diameter_max
                                                 for gid in self.gid])
            else:
                raise ValueError('Invalid measure specification.')
        else:
            if base_size_spec in ('volnv', 'volsr', 'volch'):
                if self.mprop[base_size_spec] is None:
                    raise ValueError('Volume measure empty.')
                vols = np.array(list(self.mprop[base_size_spec]))
                vols = vols*self.get_voxel_volume()
                val = 0.5*np.cbrt(3*vols/(4*math.pi))
                self.mprop['eqdia'] = {'base_size_spec': base_size_spec,
                                       'values': val}
            if base_size_spec in ('sanv', 'savi', 'sasr'):
                if self.mprop[base_size_spec] is None:
                    raise ValueError('Surface area measure empty.')
                sareas = np.array(list(self.mprop[base_size_spec]))
                sareas = sareas*self.get_voxel_surfareas(ret='mean')
                val = 0.5*np.sqrt(sareas/(4*math.pi))
                self.mprop['eqdia'] = {'base_size_spec': base_size_spec,
                                       'values': val}
            self.mprop['eqdia']['skimrp_used'] = False

    def set_mprop_solidity(self, reset_generators=True,
                           nan_treatment='replace', inf_treatment='replace',
                           nan_replacement=-1, inf_replacement=-1):
        """
        Set solidity morphological property of all 3D grains.

        Parameters
        ----------
        reset_generators : bool, optional
            Reset the scikit image generator if True, else False. Defaults to
            True.

        nan_treatment : str, optional
            Options include the following:
                * 'replace'
                * 'remove'
            Defaults to 'replace'.

        inf_treatment : str, optional
            Options include the following:
                * 'replace'
                * 'remove'
            Defaults to 'replace'.

        nan_replacement : int, optional
            Value to replace nan with if nan_treatment is 'replace'. Defaults
            to -1.

        inf_replacement : int, optional
            Value to replace inf with if inf_treatment is 'replace'. Defaults
            to -1.
        """
        if any((self.skimrp is None, reset_generators)):
            self.set_skimrp()
        # -----------------------
        solidity = np.array([gen.solidity for gen in self.skimrp.values()])
        # -----------------------
        nanlocs = np.isnan(solidity)
        inflocs = np.isinf(solidity)
        # -----------------------
        if nan_treatment == 'replace' and any(nanlocs):
            solidity[nanlocs] = nan_replacement
        if inf_treatment == 'replace' and any(inflocs):
            solidity[inflocs] = inf_replacement
        # -----------------------
        non_nangids = np.where(~nanlocs)[0].tolist()
        non_infgids = np.where(~inflocs)[0].tolist()
        # -----------------------
        if any((nan_treatment == 'remove', inf_treatment == 'remove')):
            valid_gids = []
            if nan_treatment == 'remove' and len(non_nangids) > 0:
                valid_gids += non_nangids
            if inf_treatment == 'remove' and len(non_infgids) > 0:
                valid_gids += non_infgids
            # -----------------------
            solidity = {valgid: solidity[valgid] for valgid in valid_gids}
        else:
            solidity = {valgid: solidity[valgid-1] for valgid in self.gid}
        self.mprop['solidity'] = solidity

    def set_mprop_arbbox(self, fmt='gid_dict', normalize=True):
        """
        Calculate aspect ratio of bounding box.

        Parameters
        ----------
        fmt: str, optional
            Specification of the data format.

            Defaults to 'gid_dict' for which ar values
            of each boundring box will be stored against the corresponding gid
            valued keys in a dictionary. In this case, self.mprop['arbbox']
            will be a dictionary. Other option is 'np', for numpy.

            In case of 'np', a 2D numpy array of aspect ratio values of each
            gid's bounding box will be stored. In the case of 'np' option
            however the user will have take note that indexing would have to be
            added by 1 to match with gid numbering.

        normalize : bool, optional
            Default value is True.
        """
        print(40*"-", "\nSetting grain bbox AR (metric: 'arbbox').")
        bbox_sizes = [self.find_bounding_cube_gid(gid).shape
                      for gid in self.gid]
        if normalize:
            gcds = [math.gcd(math.gcd(*sz[:2]), sz[2]) for sz in bbox_sizes]
            ars = [[_ for _ in np.array(sz) / gcd]
                   for sz, gcd in zip(bbox_sizes, gcds)]
        else:
            ars = np.array(bbox_sizes)
        # -------------------------------
        ars = np.array(ars)
        ars = ars.max(axis=1)/ars.min(axis=1)
        self.mprop['arbbox'] = {gid: ar for gid, ar in zip(self.gid, ars)}

    def fit_ellipsoids(self, routine=1, regularize_data=False, verbosity=50):
        """
        Fit ellipsoids to all grains in the grain structure.

        Parameters
        ----------
        routine : int, optional
            Specify which routine to use to fit ellipsoids to grains. The
            default is 1.

        regularize_data : bool, optional
            Option to remove outlying grain boundary surface points from the
            point cloud data before ellipoidal fitting. The default is False
            Note 1: It is recommended that regularize_data be set to False. The
            reason for this is, grain boundary surface points are not some
            random point distribution in space, but rather define the very
            shape of the grain.
            Note 2: Applicable for routine 1.

        Saved as attributes
        -------------------
        ellfits : dict
            ellfits is a dictionary havijng followingt keys .
            * center : ellispoid or other conic center coordinates [xc; yc; zc]
            * evecs : the radii directions as columns of the 3x3 matrix
            * radii : ellipsoid or other conic radii [a; b; c]
            * v : the 10 parameters describing the ellipsoid / conic
                 algebraically: Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz +
                     2Fyz + 2Gx + 2Hy + 2Iz + J = 0
            * unfit_gid : list of gids for whcih ellipsoids could not be fit.

        Returns
        -------
        None.

        Explanations
        ------------
        Routine 1: THis uses the codes available at the below GitHub link to
            calculate ellipsoid fits to grains. As stated there, the codes
            were ports from a similar MATLAB code available on the second link
            below. Explanations of the keys of ellfits dictionary provided
            above have been taken verbatim from this MATLAB Files Exchanhge
            link, except for the key unfit_gid.
            https://github.com/aleksandrbazhin/ellipsoid_fit_python
            https://uk.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit

        Authors
        -------
        Original MATLAB codes:
            Yury Petrov, Oculus VR, September, 2015
            https://uk.mathworks.com/matlabcentral/profile/authors/5507004
        Ported Python code repo contributors:
            Aleksandr Bazhin. https://github.com/aleksandrbazhin
            Vojtěch Vrba. https://github.com/vrbadev
            George Zogopoulos. https://github.com/Georacer
        """
        # Validations
        print(40*'-')
        if routine == 1:
            # ------------------------------
            from numpy.linalg import LinAlgError
            from upxo.external.ellipsoid_fit_python.ellipsoid_fit import data_regularize
            if regularize_data:
                from upxo.external.ellipsoid_fit_python.ellipsoid_fit import ellipsoid_fit
            # ------------------------------
            self.ellfits = {'center': {gid: None for gid in self.gid},
                            'evecs': {gid: None for gid in self.gid},
                            'radii': {gid: None for gid in self.gid},
                            'v': {gid: None for gid in self.gid},
                            'unfit_gid': []}
            # ------------------------------
            for gid in self.gid:
                if gid == 1 or gid % verbosity == 0 or gid == self.n:
                    print(f'Fitting elliposid to gid: {gid}')
                ggrid = self.pvgrid.threshold([gid, gid], scalars='lgi')
                gbsp = ggrid.extract_surface().points  # gbsurf_points
                fit_error = False

                if regularize_data:
                    try:
                        gbsp = data_regularize(gbsp)
                        _center_, _evecs_, _radii_, _v_ = ellipsoid_fit(gbsp)
                    except LinAlgError:
                        print(f'Encountered LinAlgError at gid: {gid}')
                        fit_error = True
                else:
                    try:
                        _center_, _evecs_, _radii_, _v_ = ellipsoid_fit(gbsp)
                    except LinAlgError:
                        print(f'Encountered LinAlgError at gid: {gid}.')
                        fit_error = True

                if fit_error:
                    self.ellfits['unfit_gid'].append(gid)
                    _center_ = np.repeat(np.nan, 3)
                    _evecs_ = np.reshape(np.repeat(np.nan, 9), (3, 3))
                    _radii_ = np.repeat(np.nan, 3)
                    _v_ = np.repeat(np.nan, 9)

                self.ellfits['center'][gid] = _center_
                self.ellfits['evecs'][gid] = _evecs_
                self.ellfits['radii'][gid] = _radii_
                self.ellfits['v'][gid] = _v_

    def set_mprop_arellfit(self, metric='max', calculate_efits=False,
                           efit_routine=1, efit_regularize_data=True):
        """
        Calculate aspect ratio of grain using ellipsoidal fit.

        Parameters
        ----------
        metric : str, optional
            Specify which metric to use for aspect ratio calculation. Options
            include:
                * max / maximum / maximal
                * min / minimum / minimal
                * xy / yx / z
                * yz / zy / x
                * xz / yz / y

        calculate_efits : bool, optional
            Specify whether ellipsoidal fitting is to be perfoemed before
            aspect ratio calculation. The default is True.

        efit_routine : int, optional
            If calculate_efits is specified True, then specifiy the routine
            to use for aspect ratio claculation. Refer to documentation
            (explanations) of the definition fit_ellipsoids for details.

        efit_regularize_data : bool, optional
            If calculate_efits is specified True, then specifiy whether data
            regularization is to be performed before ellipsoids are fit to
            grains. Refer to documentation (explanations) of the definition
            fit_ellipsoids for details.

        Saved as attributes
        -------------------
        self.mprop['arellfit']

        Explanations
        ------------
        """
        # Validations
        # ------------------------------------
        self.mprop['arellfit'] = {'metric': metric,
                                  'values': None}
        # ------------------------------------
        print(40*"-", "\nSetting grain AR by ell. fit. (metric: 'arellfit').")
        if calculate_efits or self.ellfits is not None:
            self.fit_ellipsoids(routine=efit_routine,
                                regularize_data=efit_regularize_data)
        self.mprop['arellfit']['values'] = {gid: None for gid in self.gid}
        if metric in ('max', 'maximum', 'maximal'):
            for gid in self.gid:
                if any(np.isnan(self.ellfits['radii'][gid])):
                    self.mprop['arellfit']['values'][gid] = np.nan
                else:
                    radii = self.ellfits['radii'][gid]
                    self.mprop['arellfit']['values'][gid] = max(radii)/min(radii)
        if metric in ('min', 'minimum', 'minimal'):
            for gid in self.gid:
                if any(np.isnan(self.ellfits['radii'][gid])):
                    self.mprop['arellfit']['values'][gid] = np.nan
                else:
                    radii = self.ellfits['radii'][gid]
                    self.mprop['arellfit']['values'][gid] = max(radii)/min(radii)
        if metric in ('xy', 'yx', 'z'):
            for gid in self.gid:
                if any(np.isnan(self.ellfits['radii'][gid])):
                    self.mprop['arellfit']['values'][gid] = np.nan
                else:
                    radii = self.ellfits['radii'][gid]
                    self.mprop['arellfit']['values'][gid] = radii[0]/radii[1]
        if metric in ('yz', 'zy', 'x'):
            for gid in self.gid:
                if any(np.isnan(self.ellfits['radii'][gid])):
                    self.mprop['arellfit']['values'][gid] = np.nan
                else:
                    radii = self.ellfits['radii'][gid]
                    self.mprop['arellfit']['values'][gid] = radii[1]/radii[2]
        if metric in ('xz', 'zx', 'y'):
            for gid in self.gid:
                if any(np.isnan(self.ellfits['radii'][gid])):
                    self.mprop['arellfit']['values'][gid] = np.nan
                else:
                    radii = self.ellfits['radii'][gid]
                    self.mprop['arellfit']['values'][gid] = radii[0]/radii[2]

    def generate_bresenham_line_3d(self, i1, i2, i3, j1, j2, j3):
        """
        Generate Bresenham line in 3d between two coordinate locations.

        Parameters
        ----------
        i1 : int
            Plane index location of 1st point in 3D array.

        i2 : int
            Row index location of 1st point in 3D array.

        i3 : int
            Column index location of 1st point in 3D array.

        j1 : int
            Plane index location of 2nd point in 3D array.

        j2 : int
            Row index location of 2nd point in 3D array.

        j3 : int
            Column index location of 2nd point in 3D array.

        Return
        ------
        ListOfPoints: list of tuples
            List of tuples, each containing index locations of the point
            coordinates.

        References
        ----------
        Codes in this function is taken verbatim from ther below link:
            https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/

        Explanations
        ------------
        This is needed to extract values along a line.
        """
        x1, y1, z1, x2, y2, z2 = i1, i2, i3, j1, j2, j3
        ListOfPoints = []
        ListOfPoints.append((x1, y1, z1))
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        if (x2 > x1):
            xs = 1
        else:
            xs = -1
        if (y2 > y1):
            ys = 1
        else:
            ys = -1
        if (z2 > z1):
            zs = 1
        else:
            zs = -1

        # Driving axis is X-axis
        if (dx >= dy and dx >= dz):
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while (x1 != x2):
                x1 += xs
                if (p1 >= 0):
                    y1 += ys
                    p1 -= 2 * dx
                if (p2 >= 0):
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                ListOfPoints.append((x1, y1, z1))

        # Driving axis is Y-axis
        elif (dy >= dx and dy >= dz):
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while (y1 != y2):
                y1 += ys
                if (p1 >= 0):
                    x1 += xs
                    p1 -= 2 * dy
                if (p2 >= 0):
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                ListOfPoints.append((x1, y1, z1))

        # Driving axis is Z-axis
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while (z1 != z2):
                z1 += zs
                if (p1 >= 0):
                    y1 += ys
                    p1 -= 2 * dz
                if (p2 >= 0):
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                ListOfPoints.append((x1, y1, z1))
        return ListOfPoints

    def get_values_along_line(self, loci, locj, scalars='lgi'):
        """
        Get values in 3D array along line between loci and locj points.

        Parameters
        ----------
        loci : list
            First location in the 3D array.

        locj : list
            Second location in th4e 3D array.

        scalars : str, optional
            Specify the scalar of interest. The default is 'lgi'.

        Return
        ------
        locations : list
            List of coord locations between the two user specified locations.
        """
        # Validations
        # ----------------------------
        i1, i2, i3 = loci
        j1, j2, j3 = locj
        locs = self.generate_bresenham_line_3d(i1, i2, i3, j1, j2, j3)
        # ----------------------------
        if scalars == 'lgi':
            intermediate_locations = np.array([self.lgi[loc[0]][loc[1]][loc[2]]
                                               for loc in locs])
        # ----------------------------
        return intermediate_locations

    def get_igs_properties_along_line(self, loci, locj, scalars='lgi'):
        """
        Measure intercept properties along line b/w two specified locations.

        Parameters
        ----------
        loci : list
            First location in the 3D array.

        locj : list
            Second location in th4e 3D array.

        scalars : str, optional
            Specify the scalar of interest. The default is 'lgi'.

        Return
        ------
        igs: dict
            Dictionary with the following keys.
            * 'ng': Number of grains on the intercept line.
            * 'nv': Number of voxels in every grain on the intercept line.
            * 'igs': IDs of grain numbers on the intercept line.
            * 'sv': Values of
        """
        # Validations
        # ----------------------------
        # Get scalar values aloing tge line between two locations.
        vals_line = self.get_values_along_line(loci, locj, scalars='lgi')
        vals_line_unique = np.unique(vals_line)  # Unique sclar values on line.
        ng = vals_line_unique.size  # Np. of grains between intercepts on line.
        nv = np.array([np.argwhere(vals_line == sv).squeeze().size
                       for sv in vals_line_unique])  # np. of voxels in grains.
        # ----------------------------
        intercept_properties = {'ng': ng,
                                'nv': nv,
                                'igs': nv.mean(),
                                'igs_median': np.median(nv),
                                'igs_range': np.ptp(nv),
                                'igs_std': nv.std(),
                                'igs_var': nv.var(),
                                'sv': vals_line,
                                'sv_unique': vals_line_unique}
        return intercept_properties

    def get_igs_along_line(self, loci, locj, metric='mean', minimum=True,
                           maximum=True, std=True, variance=True,
                           verbose=True):
        """
        Measure intercept properties along line b/w two specified locations.

        Parameters
        ----------
        loci : list
            First location in the 3D array.

        locj : list
            Second location in the 3D array.

        metric : str, optional
            User specification of metric. Options include the following:
                * mean / average / avg
                * median

        minimum : bool, optional
            Flag to return minimum. Return minimum when True. Default value is
            True.

        maximum : bool, optional
            Flag to return maximum. Return maximum when True. Default is True.

        std : bool, optional
            Flag to return standard deviation. Return standard deviation when
            True. Default value is True.

        variance : bool, optional
            Flag to return variance. Return variance when True. Default is
            True.

        Return
        ------
        igs: A dictionary having the following keys:
            * igs: float
                Intercept grain size metric.
            * metric: str
                Metric specified by the user.
            * min: float
                Minimum value.
            * max: float
                Maximum value.
            * std: float
                Standard deviation.
            * var: float
                Variance.
        """
        # Validations
        if verbose:
            print(40*'-',
                  f'\nGetting intercept grain size @line: {loci}---{locj}.')
        # ----------------------------
        igs, igs_std, igs_var = None, None, None
        # ----------------------------
        # Get scalar values aloing tge line between two locations.
        vals_line = self.get_values_along_line(loci, locj, scalars='lgi')
        vals_line_unique = np.unique(vals_line)  # Unique sclar values on line.
        nv = np.array([np.argwhere(vals_line == sv).squeeze().size
                       for sv in vals_line_unique])  # np. of voxels in grains.
        # ----------------------------
        igs = nv.mean() if metric in ('mean', 'average', 'avg') else None
        if metric in ('med', 'median'):
            igs = np.median(nv)
        # ----------------------------
        igs_min = nv.min() if minimum else None
        igs_max = nv.max() if maximum else None
        igs_std = nv.std() if std else None
        igs_var = nv.var() if variance else None
        # ----------------------------
        igs = {'igs': igs, 'metric': metric, 'min': igs_min, 'max': igs_max,
               'std': igs_std, 'var': igs_var}
        return igs

    def get_opposing_points_on_gs_bound_planes(self, plane='z',
                                               start_skip1=0, start_skip2=0,
                                               incr1=2, incr2=2,
                                               inclination='none',
                                               inclination_extent=0,
                                               shift_seperately=False,
                                               shift_starts=False,
                                               shift_ends=True,
                                               start_shift=0, end_shift=0):
        """
        Get points on the opposing boundaries of the grain structure.

        Parameters
        ----------
        plane : str
            Specify which boundary of grain structure to generate points on.
            If 'x', opposing points will be generated on the two opposing yz
            planes.

        start_skip1 : int
            Starting indices to skip on dimention 1. Defaults to 0.

        start_skip2 : int
            Starting indices to skip on dimention 2. Defaults to 1.

        incr1 : int
            Increments to be used on indices locatios in dimension 1. Defaults
            to 2.

        incr2 : int
            Increments to be used on indices locatios in dimension 1. Defaults
            to 2.

        inclination : str
            Inclination specificaiton for line formed by coordinate pairs
            start_points -- end_points. Options include the following:
                * 'none': no inclination.
                    Lines from coordinate pairs will be normal to user's plane
                    specification.
                * 'constant'.
                    Most lines from coordinate pairs will have the
                    same inclnation. Lines formed by starting and ending points
                    falling on opposite sides of the grain striucture bounds
                    will have an opposite inclination. Lengths of these lines
                    will differ. LInes would have different inclinations. End
                    points will be shifted to achive inclination. Refer to
                    explanations of parameter inclination_extent for details.
                * 'random'.
                    Lines have different length and inclinations. The minimum
                    length will be the length of the normal between the two
                    planes belongijg to the user's plane specification.

        inclination_extent : int
            Applies if inclination is True. Control the extent of inclination.
            Interpretation is tricky. The actual inclination is periodic with
            extent with a period defined by the number of points along a
            particular dimension.

            For example if plane is 'z', then dimension 1 is for y (row) and
            dimension 2 (co9lumn) is for x. The definition employs np.roll, on
            the stacked coordinates. As a consequence, the actual inclination
            becomes periodic with perdiod as the number of points along x. When
            y is 0, inclination is in plane x-z. At y>0, the line starts to
            incline away from x-z plane and the line ori itself befomes 3D in
            the x-y-z coordinate system. Defaults to 0. Can be negative or
            positive. A value of 0 would mean equivivalence with inclination
            specification set to 'none'. A positive value implies points
            shifted clockwise and negative otherwise.

        shift_seperately : bool
            If inclination is True, inclination_extent is not equal to 0, if
            shift_seperately is True, then start_points are forward shifted
            and end_pointse backward shifted. Defaults to False.

        shift_starts : bool
            Applies if shift_seperately is True. If True, starting points will
            be shifted by a value specified by start_shift. Defaults to False.

        shift_ends : bool
            Applies if shift_seperately is True. If True, ending points will
            be shifted by a value specified by end_shift. Defaults to True.

        start_shift : bool
            Applied if shift_starts is True. Explanations same as that for
            inclination_extent, with respect to starting point.

        end_shift : bool
            Applied if shift_starts is True. Explanations same as that for
            inclination_extent, with respect to ending point.

        Explanations
        ------------
        Please refer to in-code explanations of this function.

        Functionality order
        -------------------
        Tertiary

        Author
        ------
        Dr. Sunil Anandatheertha, UKAEA. 24-08-2024
        """
        # Validations
        # ----------------------
        '''Let us make the grid of index locations corrwponsing to the shape of
        lgi attribute.'''
        n = np.array(self.lgi.shape)-1
        locsy, locsz, locsx = np.meshgrid(range(0, n[0]), range(0, n[1]),
                                          range(0, n[2]))
        # ----------------------
        '''We will now sub-sample from this grid to extract only those
        indices which are specifie3d by the user provided skip and increment
        values. This is done for both extremes of the each of the three
        dimensions. In locsz_start, the z actually stands for axis 0, that is
        plane. Similarly y for axis 1 and x for axis 2. Similar explanations
        apply for rest.'''
        locsz_start = locsz[0][start_skip1::incr1, start_skip2::incr2]
        locsx_start = locsx[0][start_skip1::incr1, start_skip2::incr2]
        locsy_start = locsy[0][start_skip1::incr1, start_skip2::incr2]
        locsz_end = locsz[-1][start_skip1::incr1, start_skip2::incr2]
        locsx_end = locsx[-1][start_skip1::incr1, start_skip2::incr2]
        locsy_end = locsy[-1][start_skip1::incr1, start_skip2::incr2]
        # ----------------------
        if plane == 'z':
            '''We will make z as z, y as y and x as x for starting and ending
            points.'''
            start_points = np.vstack((locsz_start.ravel(), locsy_start.ravel(),
                                      locsx_start.ravel())).T
            end_points = np.vstack((locsz_end.ravel(), locsy_end.ravel(),
                                    locsx_end.ravel())).T
        elif plane == 'y':
            '''We will make z as y, y as z and x as x for starting and ending
            points. Recheck this doc.'''
            start_points = np.vstack((locsy_start.ravel(), locsz_start.ravel(),
                                      locsx_start.ravel())).T
            end_points = np.vstack((locsy_end.ravel(), locsz_end.ravel(),
                                    locsx_end.ravel())).T
        elif plane == 'x':
            '''We will make z as x, y as z and x as y for starting and ending
            points. Recheck this doc.'''
            start_points = np.vstack((locsy_start.ravel(), locsx_start.ravel(),
                                      locsz_start.ravel())).T
            end_points = np.vstack((locsy_end.ravel(), locsx_start.ravel(),
                                    locsz_end.ravel())).T
        # ----------------------
        '''If user does not want to incline the sampling lines, we will just
        return the sampling line end point coordinates as is.'''
        if inclination == 'none':
            return start_points, end_points
        # ----------------------
        '''If user wants constant inclination factor being applied to all,
        we will do it here. Note that a constant inclination facotr does not
        necessarily mean a constant inclination anmgle for all sampling lines
        which would be produced using the end points returned. This has already
        been explainerd before in the function's parameter documentaion.'''
        if inclination == 'constant' and inclination_extent == 0:
            return start_points, end_points
        if inclination == 'constant' and inclination_extent != 0:
            if shift_seperately:
                start_points = np.roll(start_points, start_shift, axis=0)
                end_points = np.roll(end_points, end_shift, axis=0)
            else:
                start_points = np.roll(start_points, inclination_extent,
                                       axis=0)
                end_points = np.roll(end_points, -inclination_extent, axis=0)
            return start_points, end_points
        # ----------------------
        '''Depending on shift_starts and shift_ends, the starting points and
        ending points will be shuffled by unknown distances. Yeah!! I mean
        shuffled randomly...'''
        if inclination == 'random':
            np.random.shuffle(start_points)
            np.random.shuffle(end_points)
            return start_points, end_points

    def get_igs_along_lines(self, metric='mean', minimum=True, maximum=True,
                            std=True, variance=True, lines_gen_method=1,
                            lines_kwargs1={'plane': 'z',
                                           'start_skip1': 0, 'start_skip2': 0,
                                           'incr1': 0, 'incr2': 0,
                                           'inclination': 'none',
                                           'inclination_extent': 0,
                                           'shift_seperately': False,
                                           'shift_starts': False,
                                           'shift_ends': False,
                                           'start_shift': 0, 'end_shift': 0}):
        """
        Measure intercept properties along lines defined by location sets i, j.

        Parameters
        ----------
        metric : str, optional
            User specification of metric. Options include the following:
                * mean / average / avg
                * median

        minimum : bool, optional
            Flag to return minimum. Return minimum when True. Default value is
            True.

        maximum : bool, optional
            Flag to return maximum. Return maximum when True. Default is True.

        std : bool, optional
            Flag to return standard deviation. Return standard deviation when
            True. Default value is True.

        variance : bool, optional
            Flag to return variance. Return variance when True. Default value
            is True.

        lines_gen_method : int
            Specify the method of generating sampling lines. Defaults to 1.

        lines_kwargs1 : dict
            Applies when lines_gen_method is set to 1. Control parameters for
            generating sampling lines through the grain structure. Defaults to
            the followig dictionary.
            lines_kwargs1={'plane': 'z', 'start_skip1': 0, 'start_skip2': 0,
                           'incr1': 0, 'incr2': 0, 'inclination': 'none',
                           'inclination_extent': 0, 'shift_seperately': False,
                           'shift_starts': False, 'shift_ends': False,
                           'start_shift': 0, 'end_shift': 0}
            See self.get_opposing_points_on_gs_bound_planes function
            documentaiton for details.

        Return
        ------
        igs: A dictionary having the following keys:
            * igs: list
                List of intercept grain size values.
            * metric: str
                Metric specified by the user.
            * min: list
                List of minimum values.
            * max: list
                List of maximum values.
            * std: list
                List of standard deviations.
            * var: list
                List of variance values.

        Example
        -------
        gstslice.get_lgi_along_lines(locsi, locsj, metric='mean', minimum=True,
                                maximum=True, std=True, variance=True)
        """
        # Validations
        # -------------------------
        igs = {'igs': [], 'metric': metric}
        if minimum:
            igs['min'] = []
        if maximum:
            igs['max'] = []
        if std:
            igs['std'] = []
        if variance:
            igs['var'] = []
        # -------------------------
        if lines_gen_method == 1:
            fn = self.get_opposing_points_on_gs_bound_planes
            temp1 = lines_kwargs1['inclination_extent']
            temp2 = lines_kwargs1['shift_seperately']
            locsi, locsj = fn(plane=lines_kwargs1['plane'],
                              start_skip1=lines_kwargs1['start_skip1'],
                              start_skip2=lines_kwargs1['start_skip2'],
                              incr1=lines_kwargs1['incr1'],
                              incr2=lines_kwargs1['incr2'],
                              inclination=lines_kwargs1['inclination'],
                              inclination_extent=temp1,
                              shift_seperately=temp2,
                              shift_starts=lines_kwargs1['shift_starts'],
                              shift_ends=lines_kwargs1['shift_ends'],
                              start_shift=lines_kwargs1['start_shift'],
                              end_shift=lines_kwargs1['end_shift'])
        # -------------------------
        for loci, locj in zip(locsi, locsj):
            _ = self.get_igs_along_line(loci, locj, metric=metric,
                                        minimum=minimum, maximum=maximum,
                                        std=std, variance=variance,
                                        verbose=False)
            igs['igs'].append(_['igs'])
            if minimum:
                igs['min'].append(_['min'])
            if maximum:
                igs['max'].append(_['max'])
            if std:
                igs['std'].append(_['std'])
            if variance:
                igs['var'].append(_['var'])
        # -------------------------
        igs['igs_all'] = np.array(igs['igs'])
        igs['igs'] = np.array(igs['igs']).mean()
        igs['ngrains'] = self.n
        if minimum:
            igs['min'] = np.array(igs['min'])
        if maximum:
            igs['max'] = np.array(igs['max'])
        if std:
            igs['std'] = np.array(igs['std'])
        if variance:
            igs['var'] = np.array(igs['var'])
        return igs

    def get_igs_along_lines_multiple_samples(self, metric='mean',
                                             minimum=True, maximum=True,
                                             std=True, variance=True,
                                             lines_gen_method=1,
                                             lines_kwargs1={'plane': 'z',
                                                            'start_skip1': 0,
                                                            'start_skip2': 0,
                                                            'incr1': 0,
                                                            'incr2': 0,
                                                            'inclination': 'none',
                                                            'inclination_extent': 0,
                                                            'shift_seperately': False,
                                                            'shift_starts': False,
                                                            'shift_ends': False,
                                                            'start_shift': 0,
                                                            'end_shift': 0},
                                             plot=True):
        pass

    def igs_sed_ratio(self, metric='mean', lines_gen_method=1,
                      reset_grain_size=True, base_size_spec='volnv',
                      lines_kwargs1={'plane': 'z',
                                     'start_skip1': 0, 'start_skip2': 0,
                                     'incr1': 3, 'incr2': 3,
                                     'inclination': 'none',
                                     'inclination_extent': 0,
                                     'shift_seperately': False,
                                     'shift_starts': False,
                                     'shift_ends': False,
                                     'start_shift': 0, 'end_shift': 0}):
        """
        Calculate the ratio of intercept grain size to sphere eq. diameter.

        Parameters
        ----------
        metric : str
            Default value is 'mean'. Options include 'mean' and 'median'.

        lines_gen_method : int
            Default value is 1.

        reset_grain_size : bool
            Default value iis True.

        base_size_spec : str
            Default value is 'volnv'.

        lines_kwargs1 : dict
            Default value is provided below.
            {'plane': 'z', 'start_skip1': 0, 'start_skip2': 0,
             'incr1': 3, 'incr2': 3, 'inclination': 'none',
             'inclination_extent': 0, 'shift_seperately': False,
             'shift_starts': False, 'shift_ends': False,
             'start_shift': 0, 'end_shift': 0}

        Return
        ------
        cags_ratio : float
            Characteristic Average Grain Size ratio
        """
        temp1 = lines_kwargs1['inclination_extent']
        lines_kwargs1 = {'plane': lines_kwargs1['plane'],
                         'start_skip1': lines_kwargs1['start_skip1'],
                         'start_skip2': lines_kwargs1['start_skip2'],
                         'incr1': lines_kwargs1['incr1'],
                         'incr2': lines_kwargs1['incr2'],
                         'inclination': lines_kwargs1['inclination'],
                         'inclination_extent': temp1,
                         'shift_seperately': lines_kwargs1['shift_seperately'],
                         'shift_starts': lines_kwargs1['shift_starts'],
                         'shift_ends': lines_kwargs1['shift_ends'],
                         'start_shift': lines_kwargs1['start_shift'],
                         'end_shift': lines_kwargs1['end_shift']}
        # -----------------------------------
        _lgm_ = lines_gen_method
        igs = self.get_igs_along_lines(metric=metric,
                                       minimum=False,
                                       maximum=False,
                                       std=False,
                                       variance=False,
                                       lines_gen_method=_lgm_,
                                       lines_kwargs1=lines_kwargs1)
        # -----------------------------------
        if reset_grain_size or self.mprop[base_size_spec] is None:
            self.set_mprop_eqdia(base_size_spec='volnv')
        # -----------------------------------
        if metric in ('mean', 'average', 'avg'):
            eqdia = self.mprop['eqdia']['values'].mean()
        elif metric in ('med', 'median'):
            eqdia = np.median(self.mprop['eqdia']['values'])
        # -----------------------------------
        cags_ratio = igs['igs']/eqdia  # Characteristic avg. grain size ratio
        return cags_ratio

    def set_mprop_sol(self):
        """Calculate solidity of grains."""
        print(40*"-", "\nSetting grain solidity values (metric: 'sol').")
        self.mprop['sol'] = None

    def set_mprop_ecc(self):
        """Calculate eccentricity of grains."""
        print(40*"-", "\nSetting grain eccentricity values (metric: 'ecc').")
        self.mprop['ecc'] = None

    def set_mprop_com(self):
        """Calculate compactness of grains."""
        print(40*"-", "\nSetting grain compactnes values (metric: 'com').")
        self.mprop['com'] = None

    def set_mprop_sph(self):
        """Calculate sphericity of grains."""
        print(40*"-", "\nSetting grain sphericity valkues (metric: 'sph').")
        self.mprop['sph'] = None

    def set_mprop_fn(self):
        """Calculate flatness of grains."""
        print(40*"-", "\nSetting grain flatness values (metric: 'fn').")
        self.mprop['fn'] = None

    def set_mprop_rnd(self):
        """Calculate roundness of grains."""
        print(40*"-", "\nSetting grain roundness values (metric: 'rnd').")
        self.mprop['rnd'] = None

    def set_mprop_fdim(self):
        """Calculate fractal dimension of grains."""
        print(40*"-", "\nSetting grain fractal dimensions (metric: 'fdim').")
        self.mprop['fdim'] = None

    @property
    def nvoxels(self):
        return self.mprop['volnv']

    @property
    def nvoxels_values(self):
        return np.array(list(self.mprop['volnv'].values()))

    def get_largest_gids(self):
        """
        Validation
        ----------
        maxgs = gstslice.nvoxels_values.max()  # Minimum grain size
        all([gstslice.nvoxels[i]==maxgs for i in gstslice.get_smallest_gids()])
        Above returns True. Therefore, works fine.
        """
        return np.where(self.nvoxels_values == self.nvoxels_values.max())[0]+1

    def get_smallest_gids(self):
        """
        Validation
        ----------
        mings = gstslice.nvoxels_values.min()  # Minimum grain size
        all([gstslice.nvoxels[i]==mings for i in gstslice.get_smallest_gids()])
        Above returns True. Therefore, works fine.
        """
        return np.where(self.nvoxels_values == self.nvoxels_values.min())[0]+1

    def get_s_gids(self, s):
        return self.s_gid[s]

    @property
    def single_voxel_grains(self):
        return np.where(self.nvoxels_values == 1)[0]+1

    @property
    def smallest_volume(self):
        return self.nvoxels_values.min()

    @property
    def largest_volume(self):
        return self.nvoxels_values.max()

    def small_grains(self, vth=2):
        """
        vth: int, floar
            Volume threshold
        """
        return np.where(self.nvoxels_values <= vth)[0]+1

    def large_grains(self, vth=2):
        return np.where(self.nvoxels_values >= vth)[0]+1

    def find_grains_by_nvoxels(self, nvoxels=2):
        return np.where(self.nvoxels_values == nvoxels)[0]+1

    def get_volnv_gids(self, gids):
        # Validations
        return [self.mprop['volnv'][gid] for gid in gids]

    def find_grains_by_mprop_range(self, prop_name='volnv', low=10, high=15,
                                   low_ineq='ge', high_ineq='le'):
        """
        Find gids of grains by specifying property name and range.

        Properties
        ----------
        prop_name: str
            Name of the morphjological property. Dewfaults to 'volnv'.

        low: int
            Lower threshold of the property range. Defaults to 10.

        high: int
            Upper threshold of the property range. Defaults to 15.

        low_ineq: str
            String denoting inequality for low value. Defaults to 'ge'.

        high_ineq: str
            String denoting inequality for high value. Defaults to 'le'.

        Input options
        -------------
        Options for prop_name:
            * volnv, volsr, volch
            * sanv, savi, sasr
            * pernv, pervl, pergl
            * eqdia, arbbox, arellfit
            * sol, ecc, com, sph, fn, rnd, mi
            * fdim

        Options for low_ineq:
            * 'ge'
            * 'gt'

        Options for low_ineq:
            * 'le'
            * 'lt'

        Example-1
        ---------
        We will use this function to querry the gids which have their volumes
        calculated by number of voxels.

        '''We can retireve the volume by number of voxels using the property,
        nvoxels_values.'''
        gstslice.nvoxels_values
        # ------------------------
        '''We can now querry for the gids having volnv between 10 and 15 as
        below. We will include 10 and 15 in our calculations by specifying the
        appropriate inequality.'''
        gids = gstslice.find_grains_by_mprop_range(prop_name='volnv',
                                            low=10, high=15,
                                            low_ineq='ge', high_ineq='le')
        # ------------------------
        '''Since the propety used here is volume by number of voxels, we can
        directly querry the values as below. Note that we need to subtract 1
        to make sure that we are using python array indexing and not the grain
        id number, which starts from 1.'''
        gstslice.nvoxels_values[gids-1]
        # ------------------------
        '''We can also get the values using the actual morphological property
        dictionary mprop by specifyng the property name. Note this wuld need
        a small list comprehension thoguh.'''
        [gstslice.mprop['volnv'][gid] for gid in gids]
        # ------------------------
        """
        if low_ineq not in ('ge', 'gt'):
            low_ineq = 'ge'
        if high_ineq not in ('le', 'lt'):
            low_ineq = 'le'
        # -----------------------------
        prop = np.array(list(self.mprop[prop_name].values()))
        # -----------------------------
        if low_ineq == 'ge' and high_ineq == 'le':
            prop_flag = np.logical_and(prop >= low, prop <= high)
        elif low_ineq == 'ge' and high_ineq == 'lt':
            prop_flag = np.logical_and(prop >= low, prop < high)
        elif low_ineq == 'gt' and high_ineq == 'le':
            prop_flag = np.logical_and(prop > low, prop <= high)
        elif low_ineq == 'gt' and high_ineq == 'lt':
            prop_flag = np.logical_and(prop > low, prop < high)
        # -----------------------------
        gids = np.argwhere(prop_flag).squeeze() + 1
        if type(gids) in dth.dt.NUMBERS:
            gids = np.array(gids)
        if gids.ndim == 0:
            gids = np.expand_dims(gids, 0)
        return gids

    def plot_single_voxel_grains(self):
        self.plot_grains_gids(self.single_voxel_grains)

    def get_lgi_subset_around_location(self, loc):
        # Validations
        if any(loc_ < 0 or loc_ > mxsz-1
               for loc_, mxsz in zip(loc, self.lgi.shape)):
            raise ValueError('Invalid location specirfication.')
        # ------------------------------
        def get_slice(i, imax):
            if i == 0:
                return slice(0, 2)
            elif i == imax-1:
                return slice(imax-2, imax)
            else:
                return slice(i-1, i+2)
        # ------------------------------
        lgi_subset = self.lgi[get_slice(loc[0], self.lgi.shape[0]),
                              get_slice(loc[1], self.lgi.shape[1]),
                              get_slice(loc[2], self.lgi.shape[2])]
        return lgi_subset

    def get_neigh_grains_next_to_location(self, loc):
        lgi_subset = self.get_lgi_subset_around_location(loc)
        return set(np.unique(lgi_subset)) - set([self.lgi[loc[0]][loc[1]][loc[2]]])

    def export_vtk3d(self, grid: dict, grid_fields: dict, file_path: str,
                     file_name: str, add_suffix: bool = True) -> None:
            """
            Export data to .vtk format.

            Parameters
            ----------
            grid : dict
                The grid dictionary containing the grid points.
                grid = {"x": xgr, "y": ygr, "z": zgr}

            grid_fields : dict
                The grid fields dictionary containing the grid fields.
                grid_fields = {"state_matrix": state_matrix,
                  "gid_matrix": gid_matrix}

            file_path : str
                The path where the .vtk file will be saved.

            file_name : str
                The name of the .vtk file.

            add_suffix : bool, optional
                If True, the suffix '_upxo' will be added at the end of the file name.
                This is advised to enable distinguishing any .vtk files you may create using
                applications such as Dream3D etc. The default is True.

            Returns
            -------
            None.

            """
            try:
                import pyvista as pv
            except ModuleNotFoundError:
                raise ModuleNotFoundError("PyVista has not been installed.")
                return

            full_file_name = os.path.join(file_path, file_name + ("_upxo.vtk" if add_suffix else ".vtk"))

            try:
                grid = pv.StructuredGrid(grid['x'], grid['y'], grid['z'])
                grid["values"] = grid_fields['state_matrix'].flatten(order="F")
                # Flatten in Fortran order to match VTK's indexing
                grid["gid_values"] = grid_fields['gid_matrix'].flatten(order="F")
                # Flatten in Fortran order to match VTK's indexing
                grid.save(full_file_name)
            except IOError as e:
                print(f"Error saving VTK file: {e}")

    def get_slice(self, slice_plane='xy', loc=0, scalar='lgi'):
        """
        Get a slice along one of the three fundamental planes.

        Explanations
        ------------

        Examples
        --------
        scalar = gstslice.get_slice(slice_plane='xy', loc=0, scalar='lgi')
        """
        # Validations
        # ---------------------------
        if slice_plane not in ('xy', 'yx', 'yz', 'zy', 'xz', 'zx'):
            raise ValueError('Invalid axis specification.')
        # ---------------------------
        if slice_plane in ('xy', 'yx') and scalar == 'lgi':
            scalar = self.lgi[loc, :, :]
        if slice_plane in ('yz', 'zy') and scalar == 'lgi':
            scalar = self.lgi[:, :, loc]
        if slice_plane in ('xz', 'zx') and scalar == 'lgi':
            scalar = self.lgi[:, loc, :]
        # ---------------------------
        return scalar

    def reset_slice_lgi(self, scalar_slice, library='scikit-image',
                        kernel_order=2):
        """
        Identify and labels grains in a 3D grain structure's 2D slice.

        Parameters
        ----------
        library : str, optional
            The library to use for grain identification. If not specified, the
            function raises a NotImplementedError for 'upxo'.
            {'opencv', 'scikit-image'}

        kernel_order : {1, 2}, optional
            The pixel connectivity criterion for labeling grains. Use 1 for
            4-connectivity and 2 for 8-connectivity. Defaults to 2.

        Examples
        --------
        lgi = gstslice.reset_slice_lgi(scalar_slice, library='scikit-image',
                                       kernel_order=4)
        """
        # Validations
        # --------------------
        if library == 'upxo':
            warnings.warn("upxo native grain detection is deprecated and"
                          " will be removed in a future version. Use "
                          " the either opencv or sckit-image instead",
                          category=DeprecationWarning,
                          stacklevel=2)
        elif library in dth.opt.ocv_options:
            print("Use of CV2 is deprecated. Please use scikit-image")
            '''concomp = cv2.connectedComponents
            # Acceptable values for opencv: 4, 8
            if kernel_order in (4, 8):
                KO = kernel_order
            elif kernel_order in (1, 2):
                KO = 4*kernel_order
            else:
                raise ValueError("Input must be in (1, 2, 4, 8)."
                                 f" Recieved {kernel_order}")'''
            return None
        elif library in dth.opt.ski_options:
            from skimage.measure import label as skim_label
            # Acceptable values for opencv: 1, 2
            if kernel_order in (4, 8):
                KO = int(kernel_order/4)
            elif kernel_order in (1, 2):
                KO = kernel_order
            else:
                raise ValueError("Input must be in (1, 2, 4, 8)."
                                 f" Recieved {kernel_order}")
        # --------------------
        for i, _s_ in enumerate(np.unique(scalar_slice)):
            b = (scalar_slice == _s_).astype(np.uint8)
            if library in dth.opt.ocv_options:
                _, labels = concomp(b*255, connectivity=KO)
            elif library in dth.opt.ski_options:
                labels, _ = skim_label(b, return_num=True, connectivity=KO)
            if i == 0:
                lgi = labels
            else:
                labels[labels > 0] += lgi.max()
                lgi = lgi + labels
        return lgi

    def char_slice_gid_psitions(self, lgi):
        """
        Calculate the positions of g4rains in the 2D slice.

        Parameters
        ----------
        lgi : np.ndarray

        Example
        -------
        """
        # Validations
        # --------------------------
        positions = {'top_left': None, 'bottom_left': None,
                     'bottom_right': None, 'top_right': None,
                     'pure_right': None, 'pure_bottom': None,
                     'pure_left': None, 'pure_top': None,
                     'left': None, 'right': None,
                     'bottom': None, 'top': None,
                     'boundary': None, 'corner': None, 'internal': None}
        # --------------------------
        all_bottoms = set(lgi[0, :])
        all_tops = set(lgi[-1, :])
        all_lefts = set(lgi[:, 0])
        all_rights = set(lgi[:, -1])
        boundary_grains = all_bottoms.union(all_tops, all_lefts, all_rights)
        internal_grains = set(np.unique(lgi)) - boundary_grains
        # --------------------------
        positions['left'] = all_lefts
        positions['right'] = all_rights
        positions['bottom'] = all_bottoms
        positions['top'] = all_tops
        # --------------------------
        positions['bottom_left'] = {lgi[0, 0]}
        positions['top_left'] = {lgi[-1, 0]}
        positions['bottom_right'] = {lgi[0, -1]}
        positions['top_right'] = {lgi[-1, -1]}
        # --------------------------
        positions['pure_left'] = positions['left'] - positions['bottom_left'] - positions['top_left']
        positions['pure_bottom'] = positions['bottom'] - positions['bottom_left'] - positions['bottom_right']
        positions['pure_right'] = positions['right'] - positions['bottom_right'] - positions['top_right']
        positions['pure_top'] = positions['top'] - positions['top_left'] - positions['top_right']
        # --------------------------
        positions['corner'] = {lgi[0, 0], lgi[-1, 0], lgi[0, -1], lgi[-1, -1]}
        positions['boundary'] = boundary_grains
        positions['internal'] = internal_grains
        # --------------------------
        return positions

    def char_lgi_slice_morpho(self, slice_plane='xy', loc=0, reset_lgi=True,
                              kernel_order=4,
                              mprop_names=['area', 'eqdia', 'fdia',
                                           'perimeter', 'perimeter_crofton',
                                           'solidity'],
                              ignore_border_grains_2d=True):
        """
        Characterize morphology of a 2D slice of self.lgi.

        NOTE: It may seem like there is no need for an additional
        grain identification phase needed fotthr 2D slice. However, the
        unique grain morphologies in the 3D can project to 2D to become
        disconnected regions but yet having the same grain ID value. This would
        not reproduce the EBSD sectioning artefact of grains with complex
        re-entrant (concave) morphologies resulting in both error in estimation
        of some grain properties anbd also changing the very definition of a
        2D grain. The latter could result in erroneous statistocal
        interpretations. If the user prefers this, they may choose to set
        reset_grains to False.

        Examples
        --------
        gstslice.char_lgi_slice_morpho(slice_plane='xy', loc=0,
                                       reset_lgi=True,
                                       kernel_order=4,
                                       mprop_names=['area', 'eqdia', 'fdia'],
                                       ignore_border_grains_2d=True)
        gstslice.lgi_slice['mprop']['eqdia']
        gstslice.lgi_slice['mprop']['area']
        gstslice.lgi_slice['mprop']['fdia']
        """
        # Validations
        # ----------------------------
        '''Prepare am empty dictionary to populate later onl.'''
        lgi_slice = {'lgi': None, 'mprop': {}}
        # ----------------------------
        '''Extract the 2D slice as per use spcificatiopn and reset the lgi as
        per user request.'''
        scalar_slice = self.get_slice(slice_plane=slice_plane, loc=loc,
                                      scalar='lgi')
        if reset_lgi:
            lgi_slice['lgi'] = self.reset_slice_lgi(scalar_slice,
                                                    library='scikit-image',
                                                    kernel_order=2)
        else:
            lgi_slice['lgi'] = self.lgi
        # ----------------------------
        '''Form scikit-image property generators for each gid in lgi.'''
        from skimage.measure import regionprops
        lgi_slice['fx'] = regionprops(lgi_slice['lgi'])
        # ----------------------------
        '''Form a subset of gids based on whether border grains are to be
        avoided or consoidered in caluclations.'''
        if ignore_border_grains_2d:
            positions = self.char_slice_gid_psitions(lgi_slice['lgi'])
            gids = positions['internal']
        else:
            gids = set(np.unique(lgi_slice['lgi']))
        # ----------------------------
        '''Caculate the actual values of properties and store them.'''
        for mpn in mprop_names:
            if mpn == 'area':
                mprop_data = [lgi_slice['fx'][gid-1].area
                              for gid in gids]
            if mpn == 'arbbox':
                bboxes = np.array([lgi_slice['fx'][gid-1].bbox
                                   for gid in gids])
                arbbox_dims = np.vstack((abs(bboxes[:, 2]-bboxes[:, 0]),
                                         abs(bboxes[:, 3]-bboxes[:, 1]))).T
                mprop_data = arbbox_dims.max(axis=1) / arbbox_dims.min(axis=1)
            if mpn == 'eqdia':
                mprop_data = [lgi_slice['fx'][gid-1].equivalent_diameter
                              for gid in gids]
            if mpn == 'fdia':
                mprop_data = [lgi_slice['fx'][gid-1].feret_diameter_max
                              for gid in gids]
            if mpn == 'perimeter':
                mprop_data = [lgi_slice['fx'][gid-1].perimeter
                              for gid in gids]
            if mpn == 'perimeter_crofton':
                mprop_data = [lgi_slice['fx'][gid-1].perimeter_crofton
                              for gid in gids]
            if mpn == 'solidity':
                mprop_data = [lgi_slice['fx'][gid-1].solidity
                              for gid in gids]
            lgi_slice['mprop'][mpn] = np.array(mprop_data)
        # ----------------------------
        lgi_slice['gid'] = np.array(list(gids))
        # ----------------------------
        self.lgi_slice = lgi_slice

    def sss_rel_morpho(self, slice_plane='xy', loc=0, reset_lgi=True,
                       reset_generators_3d=True, slice_gschar_kernel_order=4,
                       mprop_names_2d=['eqdia'], mprop_names_3d=['eqdia'],
                       ignore_border_grains_2d=True,
                       ignore_border_grains_3d=True, reset_mprops=False,
                       kwargs_arellfit3={'metric': 'max',
                                         'calculate_efits': False,
                                         'efit_routine': 1,
                                         'efit_regularize_data': True},
                       kwargs_solidity = {'nan_treatment': 'replace',
                                          'inf_treatment': 'replace',
                                          'nan_replacement': -1,
                                          'inf_replacement': -1},
                       kdeplot=False, save_plot3d_grains=True,
                       ave_plot2d_grains=True,
                       save_plot2d_grains=False):
        """
        Carry out surface -- sub-surface relationship study.

        Parameters
        ----------
        slice_plane : str, optional
            Specifyt the parallel plane of interest. Dependinmg on the value
            of loc, the actual plane will be selected. Default value is 'xy'.

        loc : int, optional
            Location of the plae of interest along direction normal to
            slice_plane. Default value is 0.

        reset_lgi : bool, optional
            Reset the lgi numbering to ensure spatial continuity. Default value
            is True.

        kernel_order : int, optional
            Kernel order or continuity structure to use for grain
            identification in the slice grain structure. Default value is 4.

        mprop_names_2d : list, optional
            Use specification of the morphological property names of 2D slice
            in UPXO to use for studying surface - sub-surface morphological
            property relationships. Default value is ['eqdia'].

        mprop_names_3d : list, optional
            Use specification of the morphological property names of 3D MCGS
            in UPXO to use for studying surface - sub-surface morphological
            property relationships. Default value is ['eqdia'].

        ignore_border_grains_2d : bool, optional
            Ignore all the border grains in the slice whilst calculat9ion of
            the morphological properties if True. Defaults to True.

        ignore_border_grains_3d : bool, optional
            Ignore all the border grains in the 3D MCGS whilst calculat9ion of
            the morphological properties if True. Defaults to True.

        reset_mprops : bool, optional

        kwargs_arellfit3 : dict, optional

        kwargs_solidity : dict, optional

        kdeplot : bool, optional

        save_plot3d_grains : bool, optional
           Defaults to True.

        save_plot2d_grains : bool, optional
            Defaults to True.

        Return
        ------
        None

        Explanations
        ------------
        None

        Notes
        -----
        User MUST note that ignore_border_grains_2d and ignore_border_grains_3d
        values must have 1-1 correspondance. That is, if the first value of
        mprop_names_2d is 'eqdia', then so should be of mprop_names_3d. If the
        second value of mprop_names_2d is 'aspect_ratio', then the second value
        of mprop_names_3d could either be 'arbbox' or 'arellfit'.

        Examples
        --------
        gstslice.sss_rel_morpho(slice_plane='xy', loc=0, reset_lgi=True,
                                kernel_order=4, mprop_names_2d=['eqdia'],
                                mprop_names_3d=['eqdia'],
                                ignore_border_grains_2d=True,
                                ignore_border_grains_3d=True)

        # DEVELOPMENT
        # DEALING WITH THE 3D GRAIN STRUCTURE:
        gids = gstslice.get_scalar_array_in_plane_unique(origin=[12, 12, 12],
                                                         normal=[1, 0, 0])
        gstslice.set_mprop_eqdia(base_size_spec='volnv')
        # ---------------------------------------
        # DEALING WITH THE SLICE
        gstslice.char_lgi_slice_morpho(slice_plane='yz', loc=12,
                                       reset_lgi=True, kernel_order=4,
                                       mprop_names=['eqdia'],
                                       ignore_border_grains_2d=False)
        gstslice.lgi_slice['mprop']['eqdia']
        # ---------------------------------------
        plt.figure()
        sns.histplot(gstslice.mprop['eqdia']['values'][gids-1],
                     label='3D grains: ESD', kde=True)
        sns.histplot(gstslice.lgi_slice['mprop']['eqdia'],
                     label='Slice of 3D grains: ECD', kde=True)
        plt.legend()
        plt.show()
        # ---------------------------------------
        # ---------------------------------------
        This is to be moved to a different docuemntation

        gstslice.set_mprop_eqdia(base_size_spec='volnv')
        gids_all = np.array(gstslice.gid)
        gids_internal = np.array(list(gstslice.gpos['internal']))

        plt.figure()
        sns.histplot(gstslice.mprop['eqdia']['values'][gids_all-1],
                     label='All grains: ESD', kde=True)
        sns.histplot(gstslice.mprop['eqdia']['values'][gids_internal-1],
                     label='Internal grains', kde=True)
        plt.legend()
        plt.show()

        gstslice.pvgrid.plot()

        VOLS = np.array(list(gstslice.mprop['volnv'].values()))
        gids_all = np.array(gstslice.gid)
        gids_internal = np.array(list(gstslice.gpos['internal']))
        plt.figure()
        sns.histplot(VOLS[gids_all-1], label='All grains', kde=True)
        sns.histplot(VOLS[gids_internal-1], label='Internal grains', kde=True)
        plt.legend()
        plt.xlabel('Grain volume')
        plt.ylabel('Count')
        plt.show()

        gstslice.n
        """
        # Validations
        if slice_plane not in ('xy', 'yx', 'yz', 'zy', 'xz', 'zx'):
            raise ValueError('Invalue slice_plane specification.')
        if len(mprop_names_2d) != len(mprop_names_3d):
            raise ValueError('Lengths of mprop_names_2d and mprop_names_3d',
                             'must be same.')
        for mpn in mprop_names_2d:
            if mpn not in ('eqdia', 'feqdia',
                           'circ', 'circularity',
                           'arbbox', 'arellfit',
                           'sol', 'solidity',
                           'ecc', 'eccentricity',
                           'igs', 'intercept_grain_size',
                           'fdim', 'fd'):
                raise ValueError('Invalid mprop_names_2d specification',
                                 f': {mpn}.')
        for mpn in mprop_names_3d:
            if mpn not in ('eqdia', 'feqdia',
                           'sph', 'sphericity',
                           'arbbox', 'arellfit',
                           'sol', 'solidity',
                           'ecc', 'eccentricity',
                           'igs', 'intercept_grain_size',
                           'fdim', 'fd'):
                raise ValueError('Invalid mprop_names_3d specification',
                                 f': {mpn}.')
        for mpn3d, mpn2d in zip(mprop_names_3d, mprop_names_2d):
            if mpn2d not in self._mprop3d2d_[mpn3d]:
                raise ValueError('Invalid mprop_names_3d-mprop_names_2d',
                                 f'combination: {mpn3d} : {mpn2d}')
        # -------------------------
        '''Slice data characterisation.'''
        ibg2d = ignore_border_grains_2d
        self.char_lgi_slice_morpho(slice_plane=slice_plane,
                                   loc=loc,
                                   reset_lgi=reset_lgi,
                                   kernel_order=slice_gschar_kernel_order,
                                   mprop_names=mprop_names_2d,
                                   ignore_border_grains_2d=ibg2d)
        # -------------------------
        '''Data setting: morphological properties of 3D MCGS.'''
        if reset_generators_3d:
            self.set_skimrp()
        # -------------------------
        for mpn in mprop_names_3d:
            """PROPERTY: 'eqdia'."""
            additional_condition = any((self.mprop[mpn] is None, reset_mprops))
            if mpn == 'eqdia' and additional_condition:
                self.set_mprop_eqdia(base_size_spec='ignore',
                                     use_skimrp=True, measure='normal')
            # ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            """PROPERTY: 'sph' """
            if mpn in ('sph', 'sphericity') and additional_condition:
                pass
            # ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            """PROPERTY: 'arbbox' """
            if mpn == 'arbbox' and additional_condition:
                self.set_mprop_arbbox()
            # ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            """PROPERTY: 'arellfit' """
            if mpn == 'arellfit' and additional_condition:
                kwar3 = kwargs_arellfit3
                self.set_mprop_arellfit(metric=kwar3['metric'],
                                        calculate_efits=kwar3['calculate_efits'],
                                        efit_routine=kwar3['efit_routine'],
                                        efit_regularize_data=kwar3['efit_regularize_data'])
            # ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            """PROPERTY: 'sol' """
            if mpn in ('sol', 'solidity') and additional_condition:
                kwa = kwargs_solidity
                self.set_mprop_solidity(reset_generators=False,
                                        nan_treatment=kwa['nan_treatment'],
                                        inf_treatment=kwa['inf_treatment'],
                                        nan_replacement=kwa['nan_replacement'],
                                        inf_replacement=kwa['inf_replacement'])
            # ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            """PROPERTY: 'ecc' """
            if mpn in ('ecc', 'eccentricity') and additional_condition:
                pass
        # ------------------------
        '''Get gids of interest in 3D and 2D. These are the gids of grains
        which fall at the slice plane.'''
        lgishape = self.lgi.shape
        if loc <= 0:
            loc = 0.5
        if slice_plane in ('xy', 'yx'):
            loc = lgishape[0]-0.5 if loc >= lgishape[0] else loc
            origin, normal = [loc, 0.5, 0.5], [1, 0, 0]
        elif slice_plane in ('yz', 'zy'):
            loc = lgishape[2]-0.5 if loc >= lgishape[2] else loc
            origin, normal = [0.5, 0.5, loc], [0, 0, 1]
        elif slice_plane in ('xz', 'zx'):
            loc = lgishape[1]-0.5 if loc >= lgishape[1] else loc
            origin, normal = [0.5, loc, 0.5], [0, 1, 0]
        gids_3d = self.get_scalar_array_in_plane_unique(origin=origin,
                                                        normal=normal)
        gids_2d = self.lgi_slice['gid']
        self.sssr['gids_3d'], self.sssr['gids_2d'] = gids_3d, gids_2d
        # ------------------------
        '''Create data-structure of property values to be compared.'''

        '''props, below is just an empty dictionary to hold values of
        appropriate properties. The keys are tuples of 3D and 2D morphological
        property name.'''
        self.sssr['props'] = {(mpn3d, mpn2d): [None, None]
                              for mpn3d, mpn2d in zip(mprop_names_3d,
                                                      mprop_names_2d)}
        '''Populate the above props dictionary.'''
        for mpn3d, mpn2d in zip(mprop_names_3d, mprop_names_2d):
            if mpn3d in ('eqdia'):
                propvals3d = self.mprop[mpn3d]['values']
            if mpn3d in ('arellfit', 'fdim', 'fd'):
                propvals3d = np.array(list(self.mprop[mpn3d]['values'].values()))
            elif mpn3d in ('feqdia', 'sol', 'solidity', 'sph', 'sphericity'):
                propvals3d = np.array(list(self.mprop[mpn3d].values()))
            elif mpn3d in ('arbbox'):
                propvals3d = np.array(list(self.mprop['arbbox'].values()))
            # --------------------------------
            if mpn2d in ('eqdia', 'feqdia', 'arbbox', 'circ', 'circularity',
                         'eccentricity', 'ecc', 'sol', 'solidity'):
                propvals2d = self.lgi_slice['mprop'][mpn2d]
            if mpn2d in ('arellfit', 'fdim', 'fd'):
                propvals2d = None
            # --------------------------------
            self.sssr['props'][(mpn3d, mpn2d)][0] = propvals3d
            self.sssr['props'][(mpn3d, mpn2d)][1] = propvals2d
        '''Compare compatible 3D and 2D morphological properties.'''
        if kdeplot:
            for mpn3d, mpn2d in zip(mprop_names_3d, mprop_names_2d):
                propvals3d = self.sssr['props'][(mpn3d, mpn2d)][0][gids_3d-1]
                propvals2d = self.sssr['props'][(mpn3d, mpn2d)][1]
                plt.figure(figsize=(5, 5), dpi=100)
                common_norm = True
                if any((propvals3d.var() < 1E-5, propvals2d.var() < 1E-5)):
                    common_norm = False
                sns.kdeplot(propvals3d, color='red', label=f'3D:{mpn3d}',
                            clip=[0, 200], cumulative=False,
                            linestyle="-", linewidth=2,
                            marker='s', markevery=20,
                            markersize=5, mfc='w', mec='r',
                            common_norm=common_norm)
                sns.kdeplot(propvals2d, color='blue', label=f'2D:{mpn2d}',
                            clip=[0, 200], cumulative=False,
                            linestyle="--", linewidth=1,
                            marker='o', markevery=20,
                            markersize=5, mfc='w', mec='b',
                            common_norm=common_norm)
                plt.legend()
                plt.title(f'common norm applied: {common_norm}')
                plt.show()
        # -------------------------
        if save_plot3d_grains:
            viz = self.plot_grains(gids_3d, scalar='lgi', cmap='viridis',
                                   style='surface', show_edges=True, lw=1.0,
                                   opacity=1.0, view=None,
                                   scalar_bar_args=None, plot_coords=False,
                                   coords=None, axis_labels = ['z', 'y', 'x'],
                                   pvp=None, throw=True)
            self.sssr['viz3d'] = viz
        # -------------------------
        data = np.zeros_like(self.lgi_slice['lgi'])
        for gid in self.lgi_slice['gid']:
            data[self.lgi_slice['lgi'] == gid] = gid
        self.lgi_slice['lgi_masked'] = data
        # -------------------------
        if ignore_border_grains_2d:
            grid = pv.ImageData()
            grid.dimensions = np.array(self.lgi_slice['lgi_masked'].shape+(0,)) + 1
            grid.origin, grid.spacing = (0, 0, 0), (1, 1, 0)
            grid.cell_data['lgi'] = self.lgi_slice['lgi_masked'].flatten(order='f')
            self.lgi_slice['pvgrid_masked'] = grid
        # ------------------------
        grid = pv.ImageData()
        grid.dimensions = np.array(self.lgi_slice['lgi'].shape+(0,)) + 1
        grid.origin, grid.spacing = (0, 0, 0), (1, 1, 0)
        grid.cell_data['lgi'] = self.lgi_slice['lgi'].flatten(order='f')
        self.lgi_slice['pvgrid'] = grid
        # -------------------------
        if save_plot2d_grains:
            # ------------------------
            pvp = pv.Plotter()
            pvp.add_mesh(self.lgi_slice['pvgrid'],
                         cmap="viridis", show_edges=False)
            pvp.view_xy()
            self.sssr['viz2d'] = pvp

    def sss_rel_morpho_multiple(self,
                                slice_planes=['xy', 'yz', 'xz'],
                                loc_starts=[0.0, 0.0, 0.0],
                                loc_ends=[5.0, 5.0, 5.0],
                                loc_incrs=[2.0, 2.0, 2.0],
                                reset_lgi=True,
                                slice_gschar_kernel_order=4,
                                mprop_names_2d=['eqdia', 'arbbox', 'solidity'],
                                mprop_names_3d=['eqdia', 'arbbox', 'solidity'],
                                ignore_border_grains_2d=True,
                                ignore_border_grains_3d=True,
                                save_plot3d_grains=True,
                                save_plot2d_grains=True,
                                show_legends=False,
                                identify_peaks=True,
                                show_peak_location=True,
                                cmp_peak_locations=True,
                                cmp_distributions=True,
                                plot_distribution_cmp=True,
                                kde3_color='red', kde3_clip=[0, 200],
                                kde3_cumulative=False, kde3_linestyle="-",
                                kde3_linewidth=2, kde3_marker='s',
                                kde3_markevery=20, kde3_markersize=5,
                                kde3_mfc='w', kde3_mec='r',
                                kde2_color='blue', kde2_clip=[0, 200],
                                kde2_cumulative=False, kde2_linestyle="-",
                                kde2_linewidth=2, kde2_marker='s',
                                kde2_markevery=20, kde2_markersize=5,
                                kde2_mfc='w', kde2_mec='r'):
        """
        Carry out surface -- sub-surface relationship study on multiple planes.

        Note
        ----
        mul denotes multiple studies.

        Parameters
        ----------
        slice_planes : list(slice_plane : str), optional
            Specify the parallel planes of interest. Dependinmg on the values
            in location specifications, the actual plane will be selected.
            Default value is ['xy', 'yz', 'xz'].

        loc_starts : list(loc : float), optional
            Location of the plae of interest along direction normal to
            slice_plane. Default value is [0.0, 0.0, 0.0].

        loc_ends : list(loc : float), optional
            Location of the plae of interest along direction normal to
            slice_plane. Default value is [5.0, 5.0, 5.0].

        loc_incrs : list(loc : float), optional
            Location of the plae of interest along direction normal to
            slice_plane. Default value is [2.0, 2.0, 2.0].

        reset_lgi : bool, optional
            Reset the lgi numbering to ensure spatial continuity. Default value
            is True.

        slice_gschar_kernel_order : int, optional
            Kernel order or continuity structure to use for grain
            identification in the slice grain structure. Default value is 4.

        mprop_names_2d : list, optional
            User specification of the morphological property names of 2D slice
            in UPXO to use for studying surface - sub-surface morphological
            property relationships. Default value is ['eqdia']. You could try
            ['eqdia', 'arbbox', 'solidity'].

        mprop_names_3d : list, optional
            User specification of the morphological property names of 3D MCGS
            in UPXO to use for studying surface - sub-surface morphological
            property relationships. Default value is ['eqdia']. You could try
            ['eqdia', 'arbbox', 'solidity']

        ignore_border_grains_2d : bool, optional
            Ignore all the border grains in the slice whilst calculat9ion of
            the morphological properties if True. Defaults to True.

        ignore_border_grains_3d : bool, optional
            Ignore all the border grains in the 3D MCGS whilst calculat9ion of
            the morphological properties if True. Defaults to True.

        save_plot3d_grains : bool, optional
           Defaults to True.

        save_plot2d_grains : bool, optional
            Defaults to True.

        show_legends : bool, optional
            Defaults to False.

        identify_peaks : bool, optional
            Defaults to True.

        show_peak_location : bool, optional
            Defaults to True.

        cmp_peak_locations : bool, optional
            Defaults to True.

        cmp_distributions : bool, optional
            Defaults to True.

        plot_distribution_cmp : bool, optional
            Defaults to True.

        kde3_color : str, optional
            Defaults to 'red'.

        kde3_clip : list or tuple, optional
            Defaults to [0, 200].

        kde3_cumulative : str, optional
            Defaults to False.

        kde3_linestyle : str, optional
            Defaults to "-".

        kde3_linewidth : str, optional
            Defaults to 2.

        kde3_marker : str, optional
            Defaults to 's'.

        kde3_markevery : int, optional
            Defaults to 20.

        kde3_markersize : float, optional
            Defaults to 5.0.

        kde3_mfc : str, optional
            Defaults to 'w'.

        kde3_mec : str, optional
            Defaults to 'r'.

        kde2_color : str, optional
            Defaults to 'blue'.

        kde2_clip : list or tuple, optional
            Defaults to [0, 200].

        kde2_cumulative : str, optional
            Defaults to False.

        kde2_linestyle : str, optional
            Defaults to "-".

        kde2_linewidth : str, optional
            Defaults to 2.

        kde2_marker : str, optional
            Defaults to 's'.

        kde2_markevery : int, optional
            Defaults to 20.

        kde2_markersize : float, optional
            Defaults to 5.0.

        kde2_mfc : str, optional
            Defaults to 'w'.

        kde2_mec : str, optional
            Defaults to 'r'.
        """
        sssrmul = {sp: {loc: None
                        for loc in np.arange(loc_starts[i], loc_ends[i],
                                             loc_incrs[i])}
                   for i, sp in enumerate(slice_planes)}
        # ----------------------------------------------------------
        print(40*'-')
        for sp in sssrmul.keys():
            for loc in sssrmul[sp].keys():
                print(f'slice plane: {sp}, loc: {loc}')
                self.sss_rel_morpho(slice_plane=sp, loc=loc,
                                    reset_lgi=reset_lgi,
                                    slice_gschar_kernel_order=slice_gschar_kernel_order,
                                    mprop_names_2d=mprop_names_2d,
                                    mprop_names_3d=mprop_names_3d,
                                    ignore_border_grains_2d=ignore_border_grains_2d,
                                    ignore_border_grains_3d=ignore_border_grains_3d,
                                    reset_mprops=False, kdeplot=False,
                                    save_plot3d_grains=save_plot3d_grains,
                                    save_plot2d_grains=save_plot2d_grains)
                # sssrmul[sp][loc] = deepcopy(self.sssr)
                sssrmul[sp][loc] = self.sssr
                print(self.sssr['gids_2d'].size)
            print(40*'-')
        # ----------------------------------------------------------
        for mpn3d, mpn2d in zip(mprop_names_3d, mprop_names_2d):
            plt.figure(figsize=(5, 5), dpi=100)
            print(f'Saving kdeplot image screenshot for property pair: 3D: {mpn3d}, 2D: {mpn2d}')
            for loc in sssrmul[sp].keys():
                for sp in sssrmul.keys():
                    propvals3d = sssrmul[sp][loc]['props'][(mpn3d, mpn2d)][0][sssrmul[sp][loc]['gids_3d']-1]
                    propvals2d = sssrmul[sp][loc]['props'][(mpn3d, mpn2d)][1]
                    # -----------------------------------
                    common_norm = True
                    if any((propvals3d.var() < 1E-5, propvals2d.var() < 1E-5)):
                        common_norm = False
                    # -----------------------------------
                    sns.kdeplot(propvals3d, color=kde3_color,
                                label=f'slice plane: {sp}, loc: {loc}',
                                clip=kde3_clip,
                                cumulative=kde3_cumulative,
                                linestyle=kde3_linestyle,
                                linewidth=kde3_linewidth, marker=kde3_marker,
                                markevery=kde3_markevery,
                                markersize=kde3_markersize,
                                mfc=kde3_mfc, mec=kde3_mec,
                                common_norm=common_norm)
                    sns.kdeplot(propvals2d, color=kde2_color,
                                label=f'2D:{mpn2d}',
                                clip=kde2_clip,
                                cumulative=kde2_cumulative,
                                linestyle=kde2_linestyle,
                                linewidth=kde2_linewidth, marker=kde2_marker,
                                markevery=kde2_markevery,
                                markersize=kde2_markersize,
                                mfc=kde2_mfc, mec=kde2_mec,
                                common_norm=common_norm)
            if show_legends:
                plt.legend()
            plt.title(f'3D:{mpn3d}. Common norm applied: {common_norm}')
            plt.xlabel(f'Property: 3D:{mpn3d} (in red), 2D:{mpn2d} (in blue)')
            plt.show()

    def import_ctf(self,
                   filePath,
                   fileName,
                   convertUPXOgs=True):
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
        pass

    def import_vtk(self,
                   filePath,
                   fileName,
                   convertUPXOgs=True):
        pass

    def update_dream3d_ABQ_file(self):
        """
        Take Eralp's code Dream3D2Abaqus and update it to also write:
            * element sets (or make them as groups) for:
                . texture partitioned grains
                . grain area binned grains
                . aspect ratio binned grains
                . boundary grains
                . internal grains
                . grain boundary surface elements
                . grain boundary edge elements
                . grain boundary junction point elements
                .
        Returns
        -------
        None.

        """
        pass

    def set_grain_positions(self, verbose=False):
        """
        Set positions of grains relative to grain structure boundaries.

        Parameters
        ---------
        verbose : bool, optional
            Print messages if True, else False.

        Developer notes
        ---------------
        Front face is defined by y=ymax.
        Back face is defined by y=0.
        Front to back face: Slice lgi along axis = 1

        Left face is defined by x=0.
        Right face is defined by x=xmax.
        Left to right face: Slice lgi along axis = 2

        Bottom face is defined by z=0.
        Top face is defined by z=zmax.
        Bottom to top face: Slice lgi along axis = 0
        # =========================================
        all: self.gid
        # --------------------
        boundary:  x=xmin, x=xmax, y=ymin, y=ymax, z=zmin, z=zmax
        # --------------------
        internal grains: set(all) - set(boundary)
        # --------------------
        left_face: x=xmin
        right_face: x=xmax
        back_face: y=ymin
        front_face: y=ymax
        bottom_face: z=zmin
        top_face: z=zmax
        # --------------------
        left_face_internal: x=xmin,   y!=ymin, y!=ymax,   z!=zmin, z!=zmax.
            That is, x=xmin,    ymin<y<ymax,   zmin<z<zmax
        right_face_internal: x=xmax,   y!=ymin, y!=ymax,   z!=zmin, z!=zmax
            That is, x=xmax,    ymin<y<ymax,   zmin<z<zmax

        back_face_internal: y=ymin,   x!=xmin, x!=xmax,   z!=zmin, z!=zmax
            That is, y=ymin,   xmin<x<xmax,  zmin<z<zmax
        front_face_internal: y=ymax,   x!=xmin, x!=xmax,   z!=zmin, z!=zmax
            That is, y=ymax,   xmin<x<xmax,  zmin<z<zmax

        bottom_face_internal: z=zmin,   x!=xmin, x!=xmax,   y!=ymin, y!=ymax
            That is, z=zmin,   xmin<x<xmax,  ymin<y<ymax
        top_face_internal: z=zmax,   x!=xmin, x!=xmax,   y!=ymin, y!=ymax
            That is, z=zmax,   xmin<x<xmax,  ymin<y<ymax
        # --------------------
        # Edges parallel to x-axis
        front_top_edge: INTERSECTION(front_face, top_face)
        top_back_edge: INTERSECTION(top_face, back_face)
        back_bottom_edge: INTERSECTION(back_face, bottom_face)
        bottom_front_edge: INTERSECTION(bottom_face, front_face)

        # Edges parallel to y-axis
        top_right_edge: INTERSECTION(top_face, right_face)
        right_bottom_edge: INTERSECTION(right_face, bottom_face)
        bottom_left_edge: INTERSECTION(bottom_face, left_face)
        left_top_edge: INTERSECTION(left_face, top_face)

        # Edges parallel to z-axis
        front_right_edge: INTERSECTION(front_face, right_face)
        right_back_edge: INTERSECTION(right_face, back_face)
        back_left_edge: INTERSECTION(back_face, left_face)
        left_front_edge: INTERSECTION(left_face, front_face)
        # --------------------
        # Edges on each face
        left_edges = UNION(bottom_left_edge, left_top_edge,
                           back_left_edge, left_front_edge)
        right_edges = UNION(top_right_edge, right_bottom_edge,
                            front_right_edge, right_back_edge)

        back_edges = UNION(top_back_edge, back_bottom_edge,
                           right_back_edge, back_left_edge)
        front_edges = UNION(front_top_edge, bottom_front_edge,
                            front_right_edge, left_front_edge)

        bottom_edges = UNION(back_bottom_edge, bottom_front_edge,
                             right_bottom_edge, bottom_left_edge)
        top_edges = UNION(front_top_edge, top_back_edge,
                          top_right_edge, left_top_edge)
        # --------------------
        # Grains at corners
        left_back_bottom = INTERSECTION(left_face, back_face, bottom_face)
        back_right_bottom = INTERSECTION(back_face, right_face, bottom_face)
        right_front_bottom = INTERSECTION(right_face, front_face, bottom_face)
        front_left_bottom = INTERSECTION(front_face, left_face, bottom_face)

        left_back_top = INTERSECTION(left_face, back_face, top_face)
        back_right_top = INTERSECTION(back_face, right_face, top_face)
        right_front_top = INTERSECTION(right_face, front_face, top_face)
        front_left_top = INTERSECTION(front_face, left_face, top_face)
        """
        print('Associating grain position string identifiers to grains.')
        from upxo._sup.data_ops import is_a_in_b_3d as is_a_in_b
        # -------------------------------------------
        if verbose:
            print('Calculating grain locations.')
        xmin, xmax = 0, self.lgi.shape[2]-1
        ymin, ymax = 0, self.lgi.shape[1]-1
        zmin, zmax = 0, self.lgi.shape[0]-1
        # -------------------------------------------
        # Find all grains
        allgrains = set(self.gid)
        # -------------------------------------------
        # Find all boundary grains
        boundary_grains = []
        for gid, glocs in zip(self.gid, self.grain_locs.values()):
            if any(glocs[:, 0] == xmin) or any(glocs[:, 0] == xmax):
                boundary_grains.append(gid)
            if any(glocs[:, 1] == ymin) or any(glocs[:, 1] == ymax):
                boundary_grains.append(gid)
            if any(glocs[:, 2] == zmin) or any(glocs[:, 2] == zmax):
                boundary_grains.append(gid)
        boundary_grains = set(boundary_grains)
        # -------------------------------------------
        # Find all internal grains
        internal_grains = allgrains - boundary_grains
        # -------------------------------------------
        vals = {'xmin': [2, xmin, 'xmin_locs'], 'xmax': [2, xmax, 'xmax_locs'],
                'ymin': [1, ymin, 'ymin_locs'], 'ymax': [1, ymax, 'ymax_locs'],
                'zmin': [0, zmin, 'zmin_locs'], 'zmax': [0, zmax, 'zmax_locs']}
        # ----------------------------------
        gid_mappings = {gid: None for gid in boundary_grains}
        for gid in boundary_grains:
            locations = {'xmin_locs': None, 'xmax_locs': None,
                         'ymin_locs': None, 'ymax_locs': None,
                         'zmin_locs': None, 'zmax_locs': None}
            for val_key, val in vals.items():
                locs = np.argwhere(self.grain_locs[gid][:, val[0]] == val[1]).T
                locations[val[2]] = locs.squeeze().size
            gid_mappings[gid] = []
            for loc_key, loc_npxl in locations.items():
                if loc_npxl:
                    gid_mappings[gid].append(loc_key[:3])
        # ----------------------------------
        left_face, right_face, back_face, front_face = [], [], [], []
        bottom_face, top_face = [], []

        for gid, gid_maps in gid_mappings.items():
            if 'xmi' in gid_maps:
                left_face.append(gid)
            if 'xma' in gid_maps:
                right_face.append(gid)
            if 'ymi' in gid_maps:
                back_face.append(gid)
            if 'yma' in gid_maps:
                front_face.append(gid)
            if 'zmi' in gid_maps:
                bottom_face.append(gid)
            if 'zma' in gid_maps:
                top_face.append(gid)

        left_face, right_face = set(left_face), set(right_face)
        back_face, front_face = set(back_face), set(front_face)
        bottom_face, top_face = set(bottom_face), set(top_face)
        # ----------------------------------
        # Edges parallel to x-axis
        front_top_edge = front_face.intersection(top_face)
        top_back_edge = top_face.intersection(back_face)
        back_bottom_edge = back_face.intersection(bottom_face)
        bottom_front_edge = bottom_face.intersection(front_face)
        # Edges parallel to y-axis
        top_right_edge = top_face.intersection(right_face)
        right_bottom_edge = right_face.intersection(bottom_face)
        bottom_left_edge = bottom_face.intersection(left_face)
        left_top_edge = left_face.intersection(top_face)
        # Edges parallel to z-axis
        front_right_edge = front_face.intersection(right_face)
        right_back_edge = right_face.intersection(back_face)
        back_left_edge = back_face.intersection(left_face)
        left_front_edge = left_face.intersection(front_face)
        # ----------------------------------
        # Edges on the left face
        left_edges = bottom_left_edge.union(left_top_edge, back_left_edge,
                                            left_front_edge)
        # Edges on the right face
        right_edges = top_right_edge.union(right_bottom_edge, front_right_edge,
                                           right_back_edge)
        # Edges on the back face
        back_edges = top_back_edge.union(back_bottom_edge, right_back_edge,
                                         back_left_edge)
        # Edges on the front face
        front_edges = front_top_edge.union(bottom_front_edge, front_right_edge,
                                           left_front_edge)
        # Edges on the bottom face
        bottom_edges = back_bottom_edge.union(bottom_front_edge,
                                              right_bottom_edge,
                                              bottom_left_edge)
        # Edges on the top face
        top_edges = front_top_edge.union(top_back_edge, top_right_edge,
                                         left_top_edge)
        # ----------------------------------
        # Corner grains
        left_back_bottom = left_face.intersection(back_face, bottom_face)
        back_right_bottom = back_face.intersection(right_face, bottom_face)
        right_front_bottom = right_face.intersection(front_face, bottom_face)
        front_left_bottom = front_face.intersection(left_face, bottom_face)

        left_back_top = left_face.intersection(back_face, top_face)
        back_right_top = back_face.intersection(right_face, top_face)
        right_front_top = right_face.intersection(front_face, top_face)
        front_left_top = front_face.intersection(left_face, top_face)

        corner_grains = left_back_bottom.union(back_right_bottom,
                                               right_front_bottom,
                                               front_left_bottom,
                                               left_back_top, back_right_top,
                                               right_front_top, front_left_top)
        # ----------------------------------
        self.gpos['internal'] = internal_grains
        self.gpos['boundary'] = boundary_grains
        self.gpos['corner'] = {'all': corner_grains,
                               'left_back_bottom': left_back_bottom,
                               'back_right_bottom': back_right_bottom,
                               'right_front_bottom': right_front_bottom,
                               'front_left_bottom': front_left_bottom,
                               'left_back_top': left_back_top,
                               'back_right_top': back_right_top,
                               'right_front_top': right_front_top,
                               'front_left_top': front_left_top}
        self.gpos['face'] = {'left': left_face, 'right': right_face,
                             'front': front_face, 'back': back_face,
                             'top': top_face, 'bottom': bottom_face}
        self.gpos['edges'] = {'left': left_edges, 'right': right_edges,
                              'back': back_edges, 'front': front_edges,
                              'bottom': bottom_edges, 'top': top_edges,
                              'front_top': front_top_edge,
                              'top_back': top_back_edge,
                              'back_bottom': back_bottom_edge,
                              'bottom_front': bottom_front_edge,
                              'top_right': top_right_edge,
                              'right_bottom': right_bottom_edge,
                              'bottom_left': bottom_left_edge,
                              'left_top': left_top_edge,
                              'front_right': front_right_edge,
                              'right_back': right_back_edge,
                              'back_left': back_left_edge,
                              'left_front': left_front_edge,
                              'top_front': front_top_edge,
                              'back_top': top_back_edge,
                              'bottom_back': back_bottom_edge,
                              'front_bottom': bottom_front_edge,
                              'right_top': top_right_edge,
                              'bottom_right': right_bottom_edge,
                              'left_bottom': bottom_left_edge,
                              'top_left': left_top_edge,
                              'right_front': front_right_edge,
                              'back_right': right_back_edge,
                              'left_back': back_left_edge,
                              'front_left': left_front_edge}
        xmin, xmax = 0, self.lgi.shape[2]-1
        ymin, ymax = 0, self.lgi.shape[1]-1
        zmin, zmax = 0, self.lgi.shape[0]-1

        if len(self.gpos['corner']['left_back_bottom']) > 1:
            # gstslice.plot_grains(gstslice.gpos['corner']['left_back_bottom'])
            point = np.array([zmin, ymin, xmin])
            for gid in self.gpos['corner']['left_back_bottom']:
                if is_a_in_b(point, self.grain_locs[gid]):
                    self.gpos['corner']['left_back_bottom'] = {gid}
                    break

        if len(self.gpos['corner']['back_right_bottom']) > 1:
            # gstslice.plot_grains(gstslice.gpos['corner']['back_right_bottom'])
            point = [zmin, ymin, xmax]
            for gid in self.gpos['corner']['back_right_bottom']:
                if is_a_in_b(point, self.grain_locs[gid]):
                    self.gpos['corner']['back_right_bottom'] = {gid}
                    break

        if len(self.gpos['corner']['right_front_bottom']) > 1:
            # gstslice.plot_grains(gstslice.gpos['corner']['right_front_bottom'])
            point = [zmin, ymax, xmax]
            for gid in self.gpos['corner']['right_front_bottom']:
                if is_a_in_b(point, self.grain_locs[gid]):
                    self.gpos['corner']['right_front_bottom'] = {gid}
                    break

        if len(self.gpos['corner']['front_left_bottom']) > 1:
            # gstslice.plot_grains(gstslice.gpos['corner']['front_left_bottom'])
            point = [zmin, ymax, ymin]
            for gid in self.gpos['corner']['front_left_bottom']:
                if is_a_in_b(point, self.grain_locs[gid]):
                    self.gpos['corner']['front_left_bottom'] = {gid}
                    break

        if len(self.gpos['corner']['left_back_top']) > 1:
            # gstslice.plot_grains(gstslice.gpos['corner']['left_back_top'])
            point = [zmax, ymin, xmin]
            for gid in self.gpos['corner']['left_back_top']:
                if is_a_in_b(point, self.grain_locs[gid]):
                    self.gpos['corner']['left_back_top'] = {gid}
                    break

        if len(self.gpos['corner']['back_right_top']) > 1:
            # gstslice.plot_grains(gstslice.gpos['corner']['back_right_top'])
            point = [zmax, ymin, xmax]
            for gid in self.gpos['corner']['back_right_top']:
                if is_a_in_b(point, self.grain_locs[gid]):
                    self.gpos['corner']['back_right_top'] = {gid}
                    break

        if len(self.gpos['corner']['right_front_top']) > 1:
            # gstslice.plot_grains(gstslice.gpos['corner']['right_front_top'])
            point = [zmax, ymax, xmax]
            for gid in self.gpos['corner']['right_front_top']:
                if is_a_in_b(point, self.grain_locs[gid]):
                    self.gpos['corner']['right_front_top'] = {gid}
                    break

        if len(self.gpos['corner']['front_left_top']) > 1:
            # gstslice.plot_grains(gstslice.gpos['corner']['front_left_top'])
            point = [zmax, ymax, xmin]
            for gid in self.gpos['corner']['front_left_top']:
                if is_a_in_b(point, self.grain_locs[gid]):
                    self.gpos['corner']['front_left_top'] = {gid}
                    break

        self.gpos['corner']['all'] = self.gpos['corner']['left_back_bottom'].union(
            self.gpos['corner']['back_right_bottom'],
            self.gpos['corner']['right_front_bottom'],
            self.gpos['corner']['front_left_bottom'],
            self.gpos['corner']['left_back_top'],
            self.gpos['corner']['back_right_top'],
            self.gpos['corner']['right_front_top'],
            self.gpos['corner']['front_left_top'])
        # ------------------------------------------
        # self.gpos['imap']['faces'] =

    def set_gid_imap_keys(self):
        '''
        @dev
        ----
        We will now inspect gstslice.gid_imap_keys
        gstslice.gid_imap_keys.keys()

        This contains a list of foward and reverse maps of grian position names
        to their respective position IDs. The IDs are itegrers. Do inspect
        values of each of the above keys to know the ID-key pair maps. This is
        mainly to aid programming.
        '''
        self.gid_imap_keys = {'inout':{'boundary': 0, 'internal': -1,},
                              'face': {'left': 1, 'right': 2,
                                       'back': 3, 'front': 4,
                                       'bottom': 5, 'top': 6},
                              'edge': {'left': 11, 'right': 22,
                                       'back': 33, 'front': 44,
                                       'bottom': 55, 'top': 66,
                                       'front_top': 46, 'top_back': 63,
                                       'back_bottom': 35, 'bottom_front': 54,
                                       'top_right': 62, 'right_bottom': 25,
                                       'bottom_left': 51, 'left_top': 16,
                                       'front_right': 42, 'right_back': 23,
                                       'back_left': 31, 'left_front': 14,
                                       },
                              'corner': {'left_back_bottom': 135,
                                         'back_right_bottom': 325,
                                         'right_front_bottom': 245,
                                         'front_left_bottom': 415,
                                         'left_back_top': 136,
                                         'back_right_top': 326,
                                         'right_front_top': 246,
                                         'front_left_top': 416},
                              'rev': {},
                              }
        rev = {}
        for k in self.gid_imap_keys.keys():
            if k in ('face', 'edge'):
                for kk, vv in self.gid_imap_keys[k].items():
                    rev[vv] = kk + '_' + k
            else:
                for kk, vv in self.gid_imap_keys[k].items():
                    rev[vv] = kk
        self.gid_imap_keys['rev'] = rev

    def assign_gid_imap_keys(self):
        """
        Assign inverse mapping keys to grains based on relative positions.

        Parameters
        ----------
        None

        Return
        ------
        None

        Examples
        --------
        gstslice.gid_imap
        gid_imap['presence']
        """
        print('\nAssigning gid inverse map aginst their position names.')
        self.gid_imap = {gid: [] for gid in self.gid}
        # ---------------------------
        # Internal grains
        if len(self.gpos['internal']) > 0:
            _id_ = self.gid_imap_keys['inout']['internal']
            for gid in self.gpos['internal']:
                self.gid_imap[gid].append(_id_)
        # Boundary grains
        _id_ = self.gid_imap_keys['inout']['boundary']
        for gid in self.gpos['boundary']:
            self.gid_imap[gid].append(_id_)
        # Face grains
        _ids_ = self.gid_imap_keys['face']
        for pos, gids in self.gpos['face'].items():
            _id_ = _ids_[pos]
            for gid in gids:
                self.gid_imap[gid].append(_id_)
        # Edge grains
        _ids_ = self.gid_imap_keys['edge']
        _idskeys_ = list(_ids_)
        for pos, gids in self.gpos['edges'].items():
            if pos in _idskeys_:
                _id_ = _ids_[pos]
                for gid in gids:
                    self.gid_imap[gid].append(_id_)
        # Corner grains
        _ids_ = self.gid_imap_keys['corner']
        gpos_subset_keys = set(self.gpos['corner'].keys())-set(['all'])
        gpos_subset = {key: self.gpos['corner'][key]
                       for key in gpos_subset_keys}
        for pos, gids in gpos_subset.items():
            _id_ = _ids_[pos]
            for gid in gids:
                self.gid_imap[gid].append(_id_)
        # ---------------------------
        self.gid_imap['presence'] = {gid: len(self.gid_imap[gid]) for gid in self.gid}

    def get_max_presence_gids(self):
        """
        Get grain with the maximum presence.

        Examples
        --------
        gstslice.get_max_presence_gids(plot=True)
        """
        presence = np.array(list(self.gid_imap['presence'].values()))
        gids = np.array(self.gid)
        gid_max_presence = np.argwhere(presence == presence.max())
        return [int(gid+1) for gid in gid_max_presence[0]]

    def clean_gs_GMD_by_source_erosion_v1(self, prop='volnv',
                                          parameter_metric='mean',
                                          threshold=1.0,
                                          reset_pvgrid_every_iter=False,
                                          find_neigh_every_iter=False,
                                          find_grvox_every_iter=False,
                                          find_grspabnds_every_iter=False,
                                          reset_skimrp_every_iter=False):
        """
        Clean the gs using grain merger by dissolution by source grain erosion.

        Parameters
        ----------
        prop : Provides which property to use as primary propetrty for
            merging grain. Defaults to 'volnv'.

        parameter_metric

        threshold : int, optional

        reset_pvgrid_every_iter : bool, optional
            Defaults to False.

        find_neigh_every_iter : bool, optional
            Defaults to False.

        find_grvox_every_iter : bool, optional
            Defaults to False.

        find_grspabnds_every_iter : bool, optional
            Defaults to False.

        reset_skimrp_every_iter : bool, optional
            Defaults to False.

        saa
        ---
        Following attributres are automatically updated after each tnp has been
        addressed.
            * self.lgi
            * self.n
            * self.gid
            * self.neigh_gid
            * self.mprop
            * self.grain_locs
            * self.spbound
            * self.spboundex

        Return
        ------
        None

        Options for prop
        ----------------
        Morphological properties:
            * 'volnv': Volume by number of voxels
            * 'volsr': Volume after grain boundary surface reconstruction
            * 'volch': Volume of convex hull
            * 'sanv': surface area by number of voxels
            * 'savi': surface area by voxel interfaces
            * 'sasr': surface area after grain boundary surface reconstruction
            * 'pernv': perimeter by number of voxels
            * 'pervl': perimeter by voxel edge lines
            * 'pergl': perimeter by geometric grain boundary line segments
            * 'eqdia': eqvivalent diameter
            * 'arbbox': aspect ratio by bounding box
            * 'arellfit': aspect ratio by ellipsoidal fit
            * 'sol': solidity
            * 'ecc': eccentricity - how much the shape of the grain differs
                from a sphere.
            * 'com': compactness
            * 'sph': sphericity
            * 'fn': flatness
            * 'rnd': roundness
            * 'fdim': fractal dimension
        Texture properties:
            * 'mo': list
            * 'tc': texture component name
        Phase properties:
            * 'pid': phase ID

        Explanations
        ------------
        volnv, volsr, volch

        Note @ developer
        ----------------
        v1 refers to version 1. This is the most basic version. Any
        advancements to retain this and introduce the new ones as seperate
        cvapabilities and choose v2, v3, etc.

        Author
        ------
        Dr. Sunil Anandatheertha: developed and implemented the technique.
        """
        # Validations
        threshold = int(threshold)
        # ------------------------------------------------
        self.char_morphology_of_grains(label_str_order=1,
                                       find_grain_voxel_locs=False,
                                       find_neigh=[True, [1], False, '1-no'],
                                       find_spatial_bounds_of_grains=False,
                                       force_compute=True, set_mprops=True,
                                       mprops_kwargs={'set_skimrp': False,
                                                      'volnv': True,
                                                      'solidity': False,
                                                      'sanv': False,
                                                      'rat_sanv_volnv': False}
                                       )
        # ------------------------------------------------
        _mvg_flag_ = False
        _iteration_ = 1
        while not _mvg_flag_:
            print(50*'=', f'\nIteration number: {_iteration_}')
            # tnp: threshold numpy array
            for tnp in np.arange(1, threshold+1, 1):
                print('\n', 40*'+', f'\n           Threshold value: {tnp}\n', 40*'+')
                # mvg: multi-voxel grains
                # mvg = self.find_grains_by_nvoxels(nvoxels=tnp)
                mvg = self.find_grains_by_mprop_range(prop_name=prop,
                                                      low=tnp, high=tnp,
                                                      low_ineq='ge',
                                                      high_ineq='le')
                '''gstslice.find_grains_by_mprop_range(prop_name='volnv',
                                                      low=7, high=7,
                                                      low_ineq='ge',
                                                      high_ineq='le')'''
                if mvg.size == 0:
                    '''If there are no mvgs of nvoxels=tnp, just skip this.'''
                    continue
                """Break up mvg (multi-voxel grain) into multiple single voxel
                grains."""
                print(f'mvg: {mvg}')
                for gid in mvg:
                    locations = np.argwhere(self.lgi == gid)
                    vx_neigh_gids = []
                    for loc in locations:
                        neighgrains = list(self.get_neigh_grains_next_to_location(loc))
                        if len(neighgrains) > 0:
                            vx_neigh_gids.append(neighgrains)
                    # vx_neigh_gids = [list(self.get_neigh_grains_next_to_location(loc))
                    #                  for loc in locations]
                    vx_neigh_gids_nneighs = [len(_) for _ in vx_neigh_gids]
                    if prop == 'volnv':
                        vx_neigh_vols = [np.array([self.nvoxels[_gid_]
                                                   for _gid_ in vx_neigh_gid_set])
                                         for vx_neigh_gid_set in vx_neigh_gids]
                        # ----------------------
                        gid_locs_in_array = []
                        for vx_neigh_vol in vx_neigh_vols:
                            pass
                        gid_locs_in_array = [DO.find_closest_locations(vx_neigh_vol,
                                                                       parameter_metric)
                                             for vx_neigh_vol in vx_neigh_vols]
                        # ----------------------
                        sink_gids = [vx_neigh_gid[_gla_[0]]
                                     for vx_neigh_gid, _gla_ in zip(vx_neigh_gids,
                                                                    gid_locs_in_array)]
                        """ Now that we have the sink gids, for each pixel of the mvg,
                        we will now merge the respective pixels of mvg with the
                        corresponding sink gids. """
                        for location, sink_gid in zip(locations, sink_gids):
                            self.lgi[location[0], location[1], location[2]] = sink_gid
                # Re-number the lgi
                old_gids = np.unique(self.lgi)
                new_gids = np.arange(start=1, stop=np.unique(self.lgi).size+1, step=1)
                for og, ng in zip(old_gids, new_gids):
                    self.lgi[self.lgi == og] = ng
                self.set_gid()
                self.calc_num_grains()
                self.set_mprop_volnv(msg=None)
                if reset_pvgrid_every_iter:
                    self.make_pvgrid()
                    self.add_scalar_field_to_pvgrid(sf_name="lgi",
                                                    sf_value=self.lgi)
                if find_neigh_every_iter:
                    self.find_neigh_gid(verbose=False)
                if find_grvox_every_iter:
                    verbosity=int(self.n/20)
                    if self.domvol <= self.ctrls['numba_activation_nvox_threshold']:
                        self.find_grain_voxel_locs(verbosity=verbosity)
                    else:
                        self.find_grain_voxel_locs_v1(disp_msg=True,
                                                      verbosity=verbosity,
                                                      saa=True,
                                                      throw=False,
                                                      use_uint16=True)
                if find_grspabnds_every_iter:
                    # gstslice.spbound, gstslice.spboundex
                    self.find_spatial_bounds_of_grains()
                if reset_skimrp_every_iter:
                    self.set_skimrp()
            _iteration_ += 1
            _mvg_flag_ = all([self.find_grains_by_nvoxels(nvoxels=tnp).size == 0
                              for tnp in range(threshold+1)])
        if not reset_pvgrid_every_iter:
            self.make_pvgrid()
            self.add_scalar_field_to_pvgrid(sf_name="lgi", sf_value=self.lgi)
        if not find_neigh_every_iter:
            self.find_neigh_gid(verbose=False)
        if not find_grvox_every_iter:
            self.find_grain_voxel_locs()  # gstslice.grain_locs
        # --------------------------------
        self.set_grain_positions(verbose=False)
        self.set_skimrp()
        # ------------------------------------------------
        self.char_morphology_of_grains(label_str_order=1,
                                       find_grain_voxel_locs=False,
                                       find_neigh=[True, [1], False, '1-no'],
                                       find_spatial_bounds_of_grains=False,
                                       force_compute=True, set_mprops=True,
                                       mprops_kwargs={'set_skimrp': False,
                                                      'volnv': True,
                                                      'solidity': False,
                                                      'sanv': False,
                                                      'rat_sanv_volnv': False}
                                       )
        # ------------------------------------------------

    def clean_gs_GMD_by_source_erosion_v2(self,
                                          prop1='volnv',
                                          parameter_metric='mean',
                                          threshold=1.0):
        """
        Clean the gs using grain merger by dissolution by source grain erosion.

        Parameters
        ----------
        prop: Provides which property to use as primary propetrty for
            merging grain. Defaults to 'volnv'.

        prop2: Provides which property o use as secondary property for
            merging grain.
        parameter_metric:
        threshold:

        saa
        ---
        Following attributres are automatically updated after each tnp has been
        addressed.
            * self.lgi
            * self.n
            * self.gid
            * self.neigh_gid
            * self.mprop
            * self.grain_locs
            * self.spbound
            * self.spboundex

        Return
        ------
        None

        Options for prop1, prop2, prop3, prop4
        --------------------------------------
        Morphological properties:
            * 'volnv': Volume by number of voxels
            * 'volsr': Volume after grain boundary surface reconstruction
            * 'volch': Volume of convex hull
            * 'sanv': surface area by number of voxels
            * 'savi': surface area by voxel interfaces
            * 'sasr': surface area after grain boundary surface reconstruction
            * 'pernv': perimeter by number of voxels
            * 'pervl': perimeter by voxel edge lines
            * 'pergl': perimeter by geometric grain boundary line segments
            * 'eqdia': eqvivalent diameter
            * 'arbbox': aspect ratio by bounding box
            * 'arellfit': aspect ratio by ellipsoidal fit
            * 'sol': solidity
            * 'ecc': eccentricity - how much the shape of the grain differs
                from a sphere.
            * 'com': compactness
            * 'sph': sphericity
            * 'fn': flatness
            * 'rnd': roundness
            * 'fdim': fractal dimension
        Texture properties:
            * 'mo': list
            * 'tc': texture component name
        Phase properties:
            * 'pid': phase ID

        Explanations
        ------------
        v1 refers to version 1. This is the most basic version. Any
        advancements to retain this and introduce the new ones as seperate
        cvapabilities and choose v2, v3, etc.

        Author
        ------
        Dr. Sunil Anandatheertha: developed and implemented the technique.
        """
        pass

    def initiate_gbp(self):
        self.Lgbp_all = {gid: None for gid in self.gid}
        self.Ggbp_all = {gid: None for gid in self.gid}

    def set_Lgbp_gid(self, gid, saa=True, throw=False, verbose=True):
        """
        Return
        ------
        Lgbp_all: All the Local grain boudnary points. Local because, they
            are defined againsdt the xmin, ymin, zmin of the grain and not the
            grain structure. A translation would be needed for the return
            value to align with the grain in grain structure.
        """
        if verbose:
            if gid % 50 == 0:
                print(f'Findging local gbp, gid: {gid}')
        lgiss = self.find_exbounding_cube_gid(gid)
        locs = np.argwhere(lgiss == gid)
        mean_gid_loc = locs.mean(axis=0)
        gbp = np.array(find_boundaries(self.make_zero_non_gids_in_lgisubset(lgiss, [gid]),
                                       connectivity=1, mode='subpixel',
                                       background=0), dtype=int)
        # gbp[gbp > 0] = 1
        Lgbp_all = np.argwhere(gbp > 0)/2
        if saa:
            self.Lgbp_all[gid] = Lgbp_all
        else:
            return Lgbp_all

    def set_Lgbp_all(self, verbose=True):
        """
        Create a dictionary of all the local grain boundary points.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for gid in self.gid:
            self.set_Lgbp_gid(gid, saa=True, throw=False, verbose=verbose)

    def globalise_gbp(self):
        """
        Edits the local grain boundary points dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        None

        DEscripotion
        ------------
        This function uses the following steps.
            1. Form gbpltv, the gbp local translation vector. This is 0.5 on
                all sides. This is a consequence of skimage.segmentation ->
                find_boundaries operations performed in the subpixel mode,
                which has an effect of truncating the extreme sides of the
                grain locations by half a pixel.
            2. Form minextreme, the gbp global translation vector.
            3. Update each local gbp by the total trnslation vector.
        """
        # Form the gbpltv, the gbp local translation vector
        gbpltv = np.array([0.5, 0.5, 0.5])
        # Form the gbp global translation vector
        minextreme = {gid: [self.spboundex['zmins'][gid-1],
                            self.spboundex['ymins'][gid-1],
                            self.spboundex['xmins'][gid-1]]
                      for gid in self.gid}
        # Translate all the Lgbp points by total translation vector
        self.Ggbp_all = {gid: self.Lgbp_all[gid] + gbpltv + minextreme[gid]
                         for gid in self.gid}

    def create_neigh_gid_pair_ids(self):
        print('\nCreating neigh_gid_pair_ids.')
        self.gid_pair_ids = {}
        pair_id = 1
        # ----------------------------------------
        for gid, neighbors in self.neigh_gid.items():
            for neighbor in neighbors:
                # Create a sorted tuple of the pair (ensures uniqueness)
                pair = tuple(int(_) for _ in sorted((gid, neighbor)))

                # Assign a new pair ID if not seen before
                if pair not in self.gid_pair_ids:
                    self.gid_pair_ids[pair_id] = pair
                    pair_id += 1
        # ----------------------------------------
        self.gid_pair_ids_unique_lr = np.unique(np.array(list(self.gid_pair_ids.values())), axis=0)
        self.gid_pair_ids_unique_rl = np.flip(self.gid_pair_ids_unique_lr, axis=1)
        # ----------------------------------------
        print('Creating neigh_gid_pair_ids reveresed.')
        self.gid_pair_ids_rev = {v: k for k, v in self.gid_pair_ids.items()}
        # ----------------------------------------
        print(f'.... a total of {len(self.gid_pair_ids_unique_lr)} unique gid pairs exit.')

    def is_gid_pair_in_lr_or_rl(self, gid_pair):
        def is_a_in_b(a, b):
            return any((b[:, 0] == a[0]) & (b[:, 1] == a[1]))
        if is_a_in_b(gid_pair, self.gid_pair_ids_unique_lr):
            return 'lr'
        elif is_a_in_b(gid_pair, self.gid_pair_ids_unique_rl):
            return 'rl'
        else:
            raise ValueError('Invalid gid_pair or corrupt database.')

    def build_gbp_stack(self):
        """
        Stack and uniquefy all grain boundary points.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.gbpstack = np.vstack((self.Ggbp_all[1], self.Ggbp_all[2]))
        for gid in self.gid[2:]:
            if gid%250 == 0:
                print(f'... @gid: {gid}')
            self.gbpstack = np.vstack((self.gbpstack, self.Ggbp_all[gid]))
        self.gbpstack = np.unique(self.gbpstack, axis=0)

    def build_gbpids(self):
        self.gbpids = [i for i in range(self.gbpstack.shape[0])]

    def build_gbp(self, verbose=False):
        """
        Consolidate processes to identify all grain boundary points.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print(40*'-', '\n')
        print('Initiating gbp data structure.')
        self.initiate_gbp()
        print('Setting local grain boundary points.')
        self.set_Lgbp_all(verbose=verbose)
        print('Globalising the local grain boundary points.')
        self.globalise_gbp()
        print('Building grain boundary point stack and ids.')
        self.build_gbp_stack()
        self.build_gbpids()

    def build_gbp_id_mappings(self):
        """
        Create gbp ID database.

        Parameters
        ----------
        None

        Return
        ------
        None

        Explanations
        ------------
        First create {gbp coord tuple: gbp ID} dictionary --> self.gbp_id_maps
        Then use this to create a {gid: gbp IDs} dictionary --> self.gbp_ids
        """
        # Form self.gbp_id_maps
        self.gbp_id_maps, gbpids_max = {}, max(self.gbpids)
        for i, (point, pointid) in enumerate(zip(self.gbpstack, self.gbpids),
                                             start=0):
            if i % 1E5 == 0 or i == gbpids_max:
                print(f'Creating global IDs for gbpstack: gbp no.{i}/{gbpids_max}')
            self.gbp_id_maps[(int(point[0]), int(point[1]), int(point[2]))] = pointid
        # -----------------------------------------
        # From self.gbp_ids
        self.gbp_ids = {gid: None for gid in self.gid}
        for gid in self.gid:
            if gid % 500 == 0:
                print(f'Forming local-global ID maps for gid no. {gid}')
            self.gbp_ids[gid] = set([self.gbp_id_maps[(int(point[0]), int(point[1]), int(point[2]))]
                                     for point in self.Ggbp_all[gid]])

    def find_gbsp(self):
        """
        Form grain boundary surface points.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Explanations
        ------------
        Grain boundary points of every grain is:   gstslice.Ggbp_all
        All grain boundary pints are:   gstslice.gbpstack
        All grain boundary point IDs are:   gstslice.gbpids

        Grain boundaryt point coordinaes to gbp ID value mapping. Here, the
        coordinates, written a tuple of 3 numbers form the key.
        gstslice.gbp_id_maps

        # Grain boundary point IDs for every gid.
        gstslice.gbp_ids

        # Coordinates of grain boundary points of each gid
        gstslice.Ggbp_all

        # Coordinates of all grain boundary points
        gstslice.gbpstack[0]

        gstslice.gbp_id_maps[tuple(gstslice.gbpstack[0])]

        """
        print(40*'-', '\nIdentifying gbp IDs of grain neigh pairs.')
        self.gbsurf_pids_vox = {i: None for i in self.gid_pair_ids.keys()}
        # gp_id: grain pair ID
        # gp: grain pair
        for gp_id, gp in self.gid_pair_ids.items():
            # gb = self.gid_pair_ids[1]
            gbp_gpl = self.gbp_ids[gp[0]]
            gbp_gpr = self.gbp_ids[gp[1]]
            self.gbsurf_pids_vox[gp_id] = gbp_gpl.intersection(gbp_gpr)
        print('    Use self.gbpstack[list(gstslice.gbsurf_pids_vox[i])] to get coords of ')
        print('        gbp at gb^th interface surface. This surface is between')
        print('        gid = gp[0] and gid = gb[1].')
        # self.gbpstack[list(gbsurf_pids_vox[1])]

    def setup_gid_pair_gbp_IDs_DS(self):
        print(40*'-', '\nSetting up {gid pair id: gbp ID list} data structure')
        self.gid_pair_gbp_IDs = {k: None for k, v in self.gid_pair_ids.items()}

    def find_gid_pair_gbp_IDs(self, gidl, gidr):
        """
        Find the gbp coords at the interface of gidl and gidr.

        Parameters
        ----------
        gidl: gid on the left side
        gidr: gid on the right side

        Returns
        -------
        None

        Explanations
        ------------
        # Step 1: Get core grain, which is actually, gidl.
        # Step 2: Get gidr, which must be one of O(1) neighbours of gidl.
        # Step 3: Get the grain boundary points of core grain.
        # Step 3: Alternative: Get the grain boundary points of core grain.
        # Step 4: Get the interface ID
        # Step 5: Get the gid_pair interfacial grain boundary point IDs.
        # Step 6: Get the coordinates of all grain boundary points of core_gid

        ALTERNATIVELY, we can also do this manually:
            # Step 1: Get core grain
            gid_core = 1
            # Step 2: Get one of its neighbours
            gid_neigh = gstslice.neigh_gid[gid_core][2]
            # Step 3: Get the grain boundary points of core grain.
            gbp_ids_core_grain = list(gstslice.gbp_ids[gid_core])
            gbp_coords_core_grain = gstslice.gbpstack[gbp_ids_core_grain]  # gbp_coords_core_grain.shape
            # Step 3: Alternative: Get the grain boundary points of core grain.
            gbp_coords_core_grain = gstslice.Ggbp_all[gid_core]  # gbp_coords_core_grain.shape
            # Step 4: Get the interface ID
            gid_pair = (gid_core, gid_neigh)
            lrrl = gstslice.is_gid_pair_in_lr_or_rl(gid_pair)
            if lrrl == 'lr':
                # Things are correctr. Nothing more to do.
                pass
            elif lrrl == 'rl':
                # gid_pair neede to be reversed.
                gid_pair = (gid_neigh, gid_core)
            interface_id = gstslice.gid_pair_ids_rev[gid_pair]
            # Step 5: Get the gid_pair interfacial grain boundary point IDs.
            gid_pair_gbp_IDs = list(gstslice.gbsurf_pids_vox[interface_id])
            # Step 6: Get the coordinates of all grain boundary points of core_gid
            gid_pair_gbp_coords = gstslice.gbpstack[gid_pair_gbp_IDs]
            # Step 7: Plot gid pairs and all grain boundary points: Figure 1
            data = {'cores': [gid_core], 'others': [gid_neigh]}
            gstslice.plot_grain_sets(data=data, scalar='lgi', plot_coords=True,
                                     coords=gbp_coords_core_grain,
                                     opacities=[1.00, 0.90, 0.75, 0.50],
                                     pvp=None, cmap='viridis',
                                     style='wireframe', show_edges=True, lw=0.5,
                                     opacity=1, view=None, scalar_bar_args=None,
                                     axis_labels = ['001', '010', '100'], throw=False,
                                     validate_data=False)

            # Step 8: Plot gid pairs and interfacial grain boundary points: Figure 2
            data = {'cores': [gid_core], 'others': [gid_neigh]}
            gstslice.plot_grain_sets(data=data, scalar='lgi', plot_coords=True,
                                     coords=gid_pair_gbp_coords,
                                     opacities=[1.00, 0.90, 0.75, 0.50],
                                     pvp=None, cmap='viridis',
                                     style='wireframe', show_edges=True, lw=0.5,
                                     opacity=1, view=None, scalar_bar_args=None,
                                     axis_labels = ['001', '010', '100'], throw=False,
                                     validate_data=False)
        """
        # gidl = 1
        # gidr = self.neigh_gid[gidl][2]
        # -----------------------------------------
        gbp_ids_core_grain = list(self.gbp_ids[gidl])
        gbp_coords_core_grain = self.gbpstack[gbp_ids_core_grain]  # gbp_coords_core_grain.shape
        # -----------------------------------------
        gbp_coords_core_grain = self.Ggbp_all[gidl]  # gbp_coords_core_grain.shape
        # -----------------------------------------
        gid_pair = (gidl, gidr)
        lrrl = self.is_gid_pair_in_lr_or_rl(gid_pair)
        if lrrl == 'lr':
            # Things are correctr. Nothing more to do.
            pass
        elif lrrl == 'rl':
            # gid_pair neede to be reversed.
            gid_pair = (gidr, gidl)
        interface_id = self.gid_pair_ids_rev[gid_pair]
        # -----------------------------------------
        return list(self.gbsurf_pids_vox[interface_id])
        # -----------------------------------------
        # self.gid_pair_gbp_coords = self.gbpstack[self.gid_pair_gbp_IDs]

    def set_gid_pair_gbp_IDs(self, verbose=False, verbose_interval=2500):
        """
        Find the gbp IDs at the interface of all unique gidl and gidr pairs.

        Parameteras
        -----------
        verbose: bool
            True to be verbose, False to not print out any messages. Defaults
            to False.

        verbose_interval: int
            Control how many times information gets printed. A higher number
            porints less messages. Defaults to 2500.

        Return
        ------
        None

        saa
        ---
        self.gid_pair_gbp_IDs: dict
            The

        self.gid_pair_gbp_IDs[gid_pair_id] for gid_pair_id in
        self.gid_pair_ids.values()

        Developer notes by Dr. SA
        -------------------------
        gstslice.gid_pair_ids is a dictionary of gid_pair_id as keys and
        the participating (gidl, gidr) pairs as values.

        We can feed this participating gid pair elements into the definition
        self.find_gid_pair_gbp_IDs to get (as return) the grain boundaryt point
        ids which would constitute the grain boundary interface surface.

        We can then repeat this for every (gidl, gidr) pair in the dictionary
        gstslice.gid_pair_ids.
        """
        verbose_interval = int(verbose_interval)
        for i, (gid_pair_id, gid_pairs) in enumerate(self.gid_pair_ids.items(),
                                                     start=1):
            if verbose and i % verbose_interval == 0:
                print(f'    Finding gid_pair_gbp_IDs[gid_pair_id={gid_pair_id}].')
            self.gid_pair_gbp_IDs[gid_pair_id] = self.find_gid_pair_gbp_IDs(*gid_pairs)
        print(f'    Finished finding {i} gid_pair_gbp_IDs[gid_pair_id] gbp ID lists.')

    def build_gid__gid_pair_IDs(self):
        """
        Build map between gid and IDs of all gid interface pairs (gid_gpid).

        saa
        ---
        self.gid_gpid:   gid  --  gid-pair-IDs

        Explanations
        ------------
        Example, if 10 be the gid and 12, 15, 17 be its O(1) neighbours,
        then the gid pairs are (10, 12), (10, 15) and (10, 17). These pairs
        have IDs as 10A, 10B and 10C. These IDs themselves are obtained from
        gstslice.gid_pair_ids. gstslice.gid_pair_ids_rev is the reverse
        mapping. Here, gstslice.gid_pair_ids is a dictionary with neighbour
        grain interface surface ID (which is the same as gid_pair) as the keys,
        having the values as, the tuple of gid_left and gid_right, that is
        (gidl, gidr).

        To explain a bit more, I would say that the keys, here
        are the same as the O(1) neighbour grain ID pair, the fir4st value
        is understood to be the core and thye second vale is one of the O(1)
        of the core gid. That is, gidl is core gid and gidr is one of the
        O(1) gid.
        """
        self.gid_gpid = {gid: [] for gid in self.gid}
        for intid, gpid in self.gid_pair_ids.items():
            '''
            intid: Interface ID: key in gstslice.gid_pair_ids.
            gpid: grain pair ID: (gidl, gidr). Value in gstslice.gid_pair_ids.
            '''
            self.gid_gpid[gpid[0]].append(intid)
            self.gid_gpid[gpid[1]].append(intid)
        for gid in self.gid:
            self.gid_gpid[gid] = set(self.gid_gpid[gid])

    def set_neigh_gid_interaction_pairs(self, verbose=True):
        """
        Please refer to the explanations below.

        Explanations
        ------------
        Every gid has neigh_gid, accessed as gstslice.neigh_gid[gid].
        Say this is a list [gid1, gi2, gid3,..., gidn]. A gid in this list
        shares a grain boundary with atleast one of the other gids in this
        list.

        The current definition extracts this information, which is needed
        in identifying those gbp which form one of the boundaries of a given
        grain boundary interface surface. In other words, this helps extract
        the IDs (and hence coordinates) of points which form the grain
        boundary segments.

        The end points oif these grain boundary segments are then be
        used to calculate the greain boundary junction points.

        Pre-development notes by Dr. Sunil Anandatheertha
        -------------------------------------------------
        gstslice.neigh_gid can be used to do this. Pick a gid in
        gstslice.neigh_gid[GID].

        For every gid, intersect the set
        set(gstslice.neigh_gid[GID])-set(gid) with,
        set(gstslice.neigh_gid[gid]). This will give the list of all grains
        in gstslice.neigh_gid[GID] which are neighbouring grains of the
        gid in question.

        We will now have, for every gid in gstslice.neigh_gid[GID], a set of
        grain IDs which are also neighbours of this gid. Let's say the elements
        of this set are {g1, g2, g3, ..., gn}. We now have a bunch of triples
        which can be put in a tuple (GID, gid, g1), (GID, gid, g2), etc, for
        every gid in gstslice.neigh_gid[GID]. There will be as many bunches of
        these triples as there are number of grains.

        Say we have extracted a triple, T1 = (GID, gid, gn). We can use this
        to extract the coordinates of the grain boundary junction lines as
        follows.
            Get the grain boundary point IDs of all gis in a triple.
            That is, get the grain boundary poinyts IDs of GID, gid and the gn
            under concern. They are:
                gbp1 = gstslice.gbp_ids[GID]
                gbp2 = gstslice.gbp_ids[gid]
                gbp3 = gstslice.gbp_ids[gn]

            Get the intersection of the three sets, gbp123 as,
            gbp123 = gbp1.intersection(gbp2, gbp3).

            gbp123 is the set of grain boundary point IDs which
            form the grain boundary junctipon line segment, that we are after.

            Associate an ID to gbp123 and sorte the triple. ID shopuld be key
            and triple should be the value. This ID represents the grain
            boundary junction line ID.

            Associate to the same ID in another dictionary, the set gbp123.

        Repeat for the next triple (GID, gid, g(n+1)), until we have
        exhausted all the triples.

        NOTE: the coordinates of each of the point can be obtained as
        gstslice.gbpstack[gid123] for every triple.
        """
        print('Finding neighbour triples to gstslice.triples')
        triples = []
        for GID in gstslice.gid:
            # Primary neighbours level
            primeneighs = set(gstslice.neigh_gid[GID])
            # Get tge grain boundary points of GID
            gbp1 = gstslice.gbp_ids[GID]
            for gid in primeneighs:
                # Secondary neighbours level @ gid
                '''
                Now we find the neighbours of gid (secneighs).
                Some of it must share a boundary with other neighs of GID.
                '''
                secneighs = set(gstslice.neigh_gid[gid])
                secneighs_probable = primeneighs - {gid}
                '''
                We will now find those secondary neighbours of gid which are
                also primary neighbours of GID.
                '''
                secneighsprim = secneighs.intersection(secneighs_probable)
                # Get the grain boudnar7y points of gid
                gbp2 = gstslice.gbp_ids[gid]
                if len(secneighsprim) > 0:
                    for sn in list(secneighsprim):
                        if verbose:
                            print(f'    Processing gid triple: ({GID},{gid},{sn}) to identify grain boundary line segments.')
                        # Get the grain boundary points of sn
                        gbp3 = gstslice.gbp_ids[sn]
                        common_gbp_ids = gbp1.intersection(gbp2, gbp3)
                        if len(common_gbp_ids) > 0:
                            triples.append([GID, gid, sn])
        '''
        We have a numpy array of size n x 3. Lets say vaues in 1st column are
        all prefixed by a, 2nd by b and 3rd by c. For example, triples  array
        like: [ [a1, b1, c1], [a1, b1, c2], ..., [c1, b1, a1], ... ,
               [b1, a1, c1]].
        There are sub arrays where the columns have been interchanged. We need
        to keep only [a1, b1, c1] instead of also havimg sub-arrays like
        [a1, c1, b1], [b1, a1, c1], so on. We will use DO.remove_permutations
        to do this now.
        '''
        print('Pruning triple duplicates.')
        triples = DO.remove_permutations(np.array(triples))
        triples = [(t[0], t[1], t[2])
                   for t in triples[np.argsort(triples[:, 0])]]
        '''We will now recompute the common grin boundary points.'''
        print('Re-computing grain boundary segment gbp IDs.')
        gb_segments_gbp_IDs = {t: None for t in triples}
        for t in triples:
            GID, gid, sn = t
            gbp1 = gstslice.gbp_ids[GID]
            gbp2 = gstslice.gbp_ids[gid]
            gbp3 = gstslice.gbp_ids[sn]
            common_gbp_ids = gbp1.intersection(gbp2, gbp3)
            gb_segments_gbp_IDs[t] = common_gbp_ids

        def get_triples_of_gid(triples, gid):
            #gid = gstslice.get_largest_gids()[0]
            triples_of_gid = []
            for triple in triples:
                if gid in triple:
                    triples_of_gid.append(triple)
            return triples_of_gid

        gid = 1
        triples_of_gid = get_triples_of_gid(triples, gid)

        gb_segments_gbp_IDs[triples_of_gid[0]]
        gstslice.gbpstack[list(gb_segments_gbp_IDs[triples_of_gid[2]])]
        coord_sets = dict()
        for i, triple in enumerate(triples_of_gid, start=1):
            coord_sets[i] = gstslice.gbpstack[list(gb_segments_gbp_IDs[triple])]

        data = {'cores': [gid], 'others': [gstslice.neigh_gid[gid]]}

        gid_pair_ids = list(gstslice.gid_gpid[gid])
        for id_pair in gid_pair_ids:
            coord_sets[str(id_pair)] = gstslice.gbpstack[gstslice.gid_pair_gbp_IDs[id_pair]]

        gstslice.plot_grain_sets(data=data, scalar='lgi', plot_coords=True,
                                 coords=coord_sets,
                                 opacities=[1.00, 0.50, 0.25, 0.50],
                                 pvp=None, cmap='viridis',
                                 style='wireframe', show_edges=True, lw=1,
                                 opacity=1, view=None, scalar_bar_args=None,
                                 axis_labels = ['001', '010', '100'], throw=False)

    def setup_gid_set__gbsegs(self):
        """
        Development notes
        -----------------
        Every gid has neigh_gid, accessed as gstslice.neigh_gid[gid].

        NOTES
        -----
        * Every neigh gid pair is tagged in gstslice.gid_pair_ids. The key is
        the ID of the pair. The value is a tuple of gidl (gid to the left) and
        gidr (gid to the right).

        * Every grain has numerous grain boundary pairs. They are contained in
        gstslice.gid_gpid. The key is the ID of the grain. The value is the set
        of neigh-gid-pair IDs. NOTE: The neigh-gid-pair ID is the same as the
        keys in gstslice.gid_pair_ids. NOTE: The name gstslice.gid_gpid means
        grain ID and Grain pair ID.
        """
        # gstslice.gid_pair_ids: gpair ID  --  neigh gid pairs
        # gstslice.gid_gpid:   gid  --  gid-pair-IDs
        pass

    def mesh(self, morpho_clean=True, smoother='zmesh', mesher='tetgen'):
        """
        Mesh the grain structure.

        Parameters
        ----------
        morpho_clean : bool
            True if morphological cleaning has to be carried out before
            meshing.

        smoother : str
            Options include 'zmesh' and 'upxo'. Default option is 'zmesh'.

        Return
        ------
        None

        Explanations
        ------------
        """
        pass

    def get_bbox_gid_mask(self, gid):
        '''Get the bounding box lgi of this grain'''
        BBLGI = self.find_bounding_cube_gid(gid)
        '''Mask the bounding box lgi of this grain with the grain ID'''
        BBLGI_mask = BBLGI == gid
        return BBLGI, BBLGI_mask

    def set_mprop_sanv(self, N=26, verbosity=100):
        """Calculate the total surface area by number of voxels."""
        print("\nCalculating grain surface areas (metric: 'sanv').")

        sanv = [None for gid in self.gid]
        _r = np.sqrt(3)*1.00001
        for gid in self.gid:
            if gid % verbosity == 0:
                print(f"Set gstslice[{self.m}].mprop['sanv'] for gid:{gid}/{self.gid[-1]}")
            '''Get the bounding box lgi of this grain'''
            BBLGI = self.find_bounding_cube_gid(gid)
            '''Find the locations of grain voxels in the bounding box'''
            BBLGI_locs = np.argwhere(BBLGI == gid)
            '''Construct tree of the grain voxel locations'''
            BBLGI_locstree = self._ckdtree_(BBLGI_locs)
            '''Find the number of nearest neighbours of every voxel in the grain'''
            neighbor_counts = BBLGI_locstree.query_ball_point(BBLGI_locs,
                                                              r=_r,
                                                              return_length=True)
            '''Boundary coordinates are those which have less than 26 neighbours'''
            boundary_coords = BBLGI_locs[neighbor_counts < N]
            sanv[gid-1] = boundary_coords.shape[0]
        self.mprop['sanv'] = {gid: sanv for gid, sanv in zip(self.gid, sanv)}
        print("Finished setting grain surface areas (metric: 'sanv').")

    def set_mprop_rat_sanv_volnv(self,
                                 reset_volnv=False,
                                 reset_sanv=False,
                                 N=26,
                                 verbosity=100):
        # --------------------------------
        print('\nCalculating mprop metric: rat_sanv_volnv')
        if reset_volnv or self.mprop['volnv'] == None:
            print("self.mprop['volnv'] data being set or reset")
            self.set_mprop_volnv()
        # --------------------------------
        if reset_sanv or self.mprop['sanv'] == None:
            print("self.mprop['sanv'] data being set or reset using N={N}")
            self.set_mprop_sanv(N=N, verbosity=verbosity)
        # --------------------------------
        self.mprop['rat_sanv_volnv'] = {gid: s/v
                                        for gid, s, v in zip(self.gid,
                                                             self.mprop['volnv'].values(),
                                                             self.mprop['sanv'].values())}

    def get_gb_voxels(self, gid, BBLGI):

        '''Find the locations of grain voxels in the bounding box'''
        BBLGI_locs = np.argwhere(BBLGI == gid)
        '''Construct tree of the grain voxel locations'''
        BBLGI_locstree = self._ckdtree_(BBLGI_locs)
        '''Find the number of nearest neighbours of every voxel in the grain'''
        neighbor_counts = BBLGI_locstree.query_ball_point(BBLGI_locs,
                                                          r=np.sqrt(3)*1.00001,
                                                          return_length=True)
        '''Boundary coordinates are those which have less than 26 neighbours'''
        boundary_coords = BBLGI_locs[neighbor_counts < 26]
        return boundary_coords

    def sep_gbzcore_from_bbgidmask(self, boundary_coords, BBLGI_mask):
        # Update the mask to a new variable and seperate the grain boundary
        # from core
        BBLGI_mask_ = BBLGI_mask.astype(int)
        for bc in boundary_coords:
            BBLGI_mask_[bc[0], bc[1], bc[2]] = -1

        BBLGI_mask_gb = np.copy(BBLGI_mask_)
        BBLGI_mask_gb[BBLGI_mask_gb != -1] = 0
        BBLGI_mask_gb = np.abs(BBLGI_mask_gb)

        BBLGI_mask_core = np.copy(BBLGI_mask_)
        BBLGI_mask_core[BBLGI_mask_core == -1] = 0
        CORE_coords = np.argwhere(BBLGI_mask_core == 1)

        masks = (BBLGI_mask_gb, BBLGI_mask_core)
        return masks, CORE_coords

    def get_grain_coords(self, gid):
        return self.grain_locs[gid]

    def get_mp3d_ofcoords(self, coords):
        return self._upxo_mp3d_.from_coords(coords)

    def get_grain_mp3d(self, gid):
        return self.get_mp3d_ofcoords(self.get_grain_coords(gid))

    def get_points_in_feature_coord(self, feature_type='gb',
                                    selcri='random',
                                    fcoords=None,
                                    n=1,
                                    get_neigh_vox=False,
                                    kwargs_nv={'vs': 1.0,
                                               'ret_ind': False,
                                               'ret_coords': True,
                                               'ret_in_coord': False},
                                    validate_user_inputs=True
                                    ):
        """
        feature_type will be:
            1. 'gb' in case of grain boundary
            2. 'g' in case of grains
        selcri will be:
                1. 'random' when the point is to be selected at random
                2. 'centroid' when the point is to be selected at centroid
                3. 'meandistant' when the point to be selected must be at
                    approximately be at the statistical mean of the point
        fcoords will be:
            1. gstslice.grain_locs is grain coordinates if grain coordinates
               is being used
            2. boundary_coords (calculated usig gstslice.get_gb_voxels(..))
               if grain boundary c oordinates is being used
        """
        if validate_user_inputs:
            # Validate feature_type
            if feature_type not in ('gb', 'g'):
                # gb: grain boundary
                # g: grain
                print('Invalid feature type')
                return None
            # Validate selecytion criterion
            if selcri not in ('centroid', 'random'):
                print('Invalid point seldction criterion specification.')
                return None
            # Validate fcoords
            # Validate n
            # Validate get_neigh_vox
            # Validate kwargs_nv
        # ---------------------------------
        if selcri == 'centroid':
            # Select the coordinate
            selcoord = fcoords.mean(axis=0)
        elif selcri in ('random', 'random_choice'):
            selcoord_i = np.random.choice(range(fcoords.shape[0]),
                                          n,
                                          replace=False)[0]
            selcoord = fcoords[selcoord_i]
        # ---------------------------------
        if get_neigh_vox:
            __mp3d = self.get_mp3d_ofcoords(fcoords)
            __x = __mp3d.find_first_order_neigh_CUBIC
            neigh_vox = __x(selcoord,
                            kwargs_nv['vs'],
                            return_indices=kwargs_nv['ret_ind'],
                            return_coords=kwargs_nv['ret_coords'],
                            return_input_coord=kwargs_nv['ret_in_coord'])[0]
        else:
            neigh_vox = None
        # ---------------------------------
        return selcoord, neigh_vox

    def get_k_nearest_coords_from_tree(self, tree, coord, K):
        """
        nearest_coords = gstslice.get_k_nearest_coords(tree, coord, K)
        """
        _, nearest_ids = tree.query(coord, k=K)
        close_coords_core = tree.data[nearest_ids]
        return close_coords_core

    def setup_gid_twin(self, GIDS):
        self.gid_twin = {gid: None for gid in GIDS}

    def copy_lgi_1(self):
        self.lgi1 = deepcopy(self.lgi)

    def add_fdb(self, *, fname, dnames, datas, info):
        """
        Add ferature data base.

        Parameters
        ----------
        None

        Return
        ------
        None

        Example
        -------
        self.add_fdb(fname='twin_01',
                     dnames='fid',
                     datas=123,
                     info={'a': 1, 'b': 2})

        Notes
        -----
        1. Intended for internal use.
        """
        # initial Validations for fname
        # -----------------------------------------
        '''if fname in self.fdb.keys():
            raise ValueError(f'[fname: {fname}] is an existing feature.',
                             'Use gstslice.reset_fdb(..) to reset.')'''
        # -----------------------------------------
        # Validations for infokeys
        if not isinstance(info, dict):
            raise ValueError('info must be a dictionary')
        if not all([isinstance(info_, str) for info_ in info.keys()]):
            raise ValueError('infokey_list are not all strings.')
        # -----------------------------------------
        if type(dnames) not in dth.dt.ITERABLES:
            dnames = (dnames,)
        if type(datas) not in dth.dt.ITERABLES:
            datas = (datas,)
        # -----------------------------------------
        self.fdb[fname] = {}
        self.fdb[fname]['data'] = {}
        for dname, data in zip(dnames, datas):
            self.fdb[fname]['data'][dname] = data
        self.fdb[fname]['info'] = info

    def reset_fdb(self, fname, data, info, retain_info=False):
        if fname in self.fdb.keys():
            pass
        else:
            raise ValueError(f'fname: {fname} does not exist. Nothing reset.',
                             'Use gstslice.add_fdb(..) to set.')

    def find_twin_hosts(self,
                        nprops=2,
                        mprops={'volnv': {'use': True,
                                          'reset': False,
                                          'k': [.1, .8],
                                          'min_vol': 4,
                                          },
                                'rat_sanv_volnv': {'use': True,
                                                   'reset': False,
                                                   'k': [.1, .8],
                                                   'sanv_N': 26,},
                                },
                        viz_grains=False, opacity=1.0):
        """
        nprops: Number of propertiez to use
        mprop_names: Property names
        avoid_svg: Avoid single voxel grains

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs

        pxt = mcgs()
        pxt.simulate(verbose=False)
        tslice = 25
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=True,
                                           find_spatial_bounds_of_grains=True,
                                           force_compute=True)

        GIDS_masks_mprops, GIDS_mask, GIDS = gstslice.find_twin_hosts(nprops=2,
                                 mprops={'volnv': {'use': True,
                                                   'reset': False,
                                                   'k': [0.1, 1],
                                                   'min_vol': 4,
                                                   },
                                         'rat_sanv_volnv': {'use': True,
                                                            'reset': False,
                                                            'k': [.8, 1],
                                                            'sanv_N': 26},
                                         },
                                 min_vol=0, viz_grains=True,
                                 opacity=0.2
                                 )
        """
        print(40*'-', '\nFinding grains which can host twins.\n')
        # Validate mprops
        for mn in mprops.keys():
            if mn not in ('volnv', 'rat_sanv_volnv'):
                print('  Invalid mprop names specified.')
                print('  ONly volnv, rat_sanv_volnv allowed as of now.')
                return None
        print('   mprops names validation pass.')
        # -------------------------------------
        # Validata mprop data existance
        for mn in mprops.keys():
            if mn == 'volnv' and mprops['volnv']['use']:
                if mprops[mn]['reset'] or self.mprop[mn] is None:
                    print('VOLNV data being set or reset')
                    self.set_mprop_volnv()
            if mn == 'rat_sanv_volnv' and mprops['rat_sanv_volnv']['use']:
                N = mprops['rat_sanv_volnv']['sanv_N']
                verbosity = self.n // 10
                if self.mprop['sanv'] == None:
                    self.set_mprop_sanv(N=N, verbosity=verbosity)
                if mprops[mn]['reset'] or self.mprop[mn] is None:
                    print('rat_sanv_volnv data being set or reset using N={N}')
                    self.set_mprop_rat_sanv_volnv(reset_volnv=False,
                                                  reset_sanv=False,
                                                  N=N, verbosity=verbosity)
        print('\nmprops data validation pass.')
        # -------------------------------------
        # Find the actual number of properties to use based on user input
        # mprop flag
        nprops = np.sum([1 for mn in mprops.keys() if mprops[mn]['use']])
        # -------------------------------------
        GIDS_masks_mprops = np.full(nprops+1, None)
        GIDS = {mn: None for mn in mprops.keys()}
        # -------------------------------------
        mprop_i = 0
        for i, mn in enumerate(mprops.keys(), start=0):
            if mn in ('volnv', 'rat_sanv_volnv') and mprops[mn]['use']:
                print(f'Caclulaing gid_masks for mprop: {mn}.')
                d = NPA(list(self.mprop[mn].values()))  # Data
                f = mprops[mn]['k']  # User defined factors
                dmax = d.max()  # Data maximum
                dl = dmax*f[0]  # Data low
                dh = dmax*f[1]  # Data high
                GIDS_masks_mprops[i] = _npla(d >= dl, d <= dh)
                mprop_i += 1
        # -------------------------------------
        '''Identify multi-voxel grains'''
        vol = NPA(list(self.mprop['volnv'].values()))
        GIDS_masks_mprops[mprop_i] = vol >= mprops['volnv']['min_vol']
        # -------------------------------------
        GIDS_masks_mprops = np.stack(GIDS_masks_mprops, axis=1)
        GIDS_mask = np.prod(GIDS_masks_mprops, axis=1).astype(bool)
        GIDS = np.argwhere(GIDS_mask).T[0]+1
        # -------------------------------------
        if viz_grains:
            self.make_pvgrid()
            self.add_scalar_field_to_pvgrid(sf_name="lgi", sf_value=None)
            self.plot_grains(GIDS+1, opacity=opacity, show_edges=False)
        # -------------------------------------
        return GIDS_masks_mprops, GIDS_mask, GIDS

    def setup_for_twins(self, nprops=2,
                        mprops={'volnv': {'use': True,
                                          'reset': False,
                                          'k': [.1, .8],
                                          'min_vol': 4,
                                          },
                                'rat_sanv_volnv': {'use': True,
                                                   'reset': False,
                                                   'k': [.1, .8],
                                                   'sanv_N': 26
                                                   },
                                },
                        instance_name='twin.1',
                        feature_name='annealing_twin',
                        viz_grains=False,
                        opacity=1.0):
        """
        Carry out pre-requisite operations needed to establish twins
        """
        print('Finding twin host grains.')
        _gid_data_ = self.find_twin_hosts(nprops=nprops,
                                          mprops=mprops,
                                          viz_grains=viz_grains,
                                          opacity=opacity)
        GIDS_masks_mprops, GIDS_mask, GIDS = _gid_data_
        GIDS = GIDS
        # -----------------------------------------------
        print('\nSetting up the twin data structure')
        self.setup_gid_twin(GIDS)
        print(f'\nSetting Feature Data Base [--->   {instance_name}   <---]\n',
              '     for twinned grain structur4e instance.')
        self.add_fdb(fname=instance_name,
                     dnames=('fid',
                             'feat_host_gids'),
                     datas=(deepcopy(self.lgi),
                            GIDS),
                     info={'name': feature_name,
                           'mprops': mprops,
                           'vf_min': None,
                           'vf_max': None,
                           'vf_actual': None,
                           })
        # -----------------------------------------------

    def get_local_global_coord_offset(self, gid):
        gloffset = np.array([self.spbound['zmins'][gid-1],
                             self.spbound['ymins'][gid-1],
                             self.spbound['xmins'][gid-1]])
        return gloffset

    def offset_local_to_global(self, gid, local_coord):
        """
        """
        return local_coord + self.get_local_global_coord_offset(gid)

    def get_cutoff_twvol(self, gid, cutoff_twin_vf):
        return NPA(cutoff_twin_vf)*self.mprop['volnv'][gid]

    def identify_twins_gid(self, gid,
                           twspec={'n': None,
                                   'tv': None,
                                   'dlk': np.array([1.0, -1.0, 1.0]),
                                   'dnw': np.array([0.5, 0.5, 0.5]),
                                   'dno': np.array([0.5, 0.5, 0.5]),
                                   'tdis': 'normal',
                                   'tpar': {'loc': 4, 'scale': 2.5, 'val': 1},
                                   'vf': [0.05, 1.00],
                                   'sep_bzcz': False,
                                   },
                           twgenspec={'seedsel': 'random_gb',
                                      'K': 10,
                                      'bidir_tp': False,
                                      },
                           viz=False,
                           viz_flags={'gb': True,  # Boundary
                                      'gc': True,  # Grain core
                                      'tb': True,  # Twin boundary
                                      'tc': True,  # Twin core
                                      'tpvec': False,  # Twin plane vectors
                                      },
                           viz_steps={'gb': 2,  # Boundary
                                      'gc': 4,  # Grain core
                                      }
                           ):
        """
        Generate and inlude twin in gstslice.lgi.

        Parameters
        ----------
        gid: int
            grain ID number
        seed_selcri: str
            Seed point selection criterion. Options:
                'random_gb' - Random point from grain boundary coordinates
                'random_g' - Random point from grain coordinates
                'centroid_gb' - Grain boundary centroid
                'centroid_g' - Grain boundary

        Return
        ------
        None
        """
        if self.gid_twin is None:
            print('Twin data strucures not set yet. Please, ')
            print('        run self.setup_for_twins(...) first.')
            return None
        # ------------------------------------------------------------
        # gid = gstslice.get_largest_gids()[0]
        '''Get bounding box lgi and bounding box gid mask.'''
        BBLGI, BBLGI_mask = self.get_bbox_gid_mask(gid)
        '''Extract grain boundary coordinates'''
        BCOORDS = self.get_gb_voxels(gid, BBLGI)
        '''Extract grain boundary zone and the core zone using bounding box
        gid mask and boundary coordinates'''
        _sepgbz_c_bbgidmask_ = self.sep_gbzcore_from_bbgidmask
        masks, CORE_coords = _sepgbz_c_bbgidmask_(BCOORDS, BBLGI_mask)
        BBLGI_mask_gb, BBLGI_mask_core = masks
        '''Form the tree for grain core coordinates'''
        CORE_tree = self._ckdtree_(CORE_coords)
        # ------------------------------------------------------------
        BBLGI_locs = np.argwhere(BBLGI_mask)
        # ------------------------------------------------------------
        '''Decide upon the feature coordinates to use'''
        if twgenspec['seedsel'] in ('random_gb', 'centroid_gb'):
            # Grain boundary
            feature_type = 'gb'
            fcoords = BCOORDS
        elif twgenspec['seedsel'] in ('random_g', 'centroid_g'):
            # Greian
            feature_type = 'g'
            fcoords = BBLGI_locs
        # ------------------------------------------------------------
        if twgenspec['seedsel'] in ('centroid_g', 'centroid_gb'):
            seed_sel_cri = 'centroid'
        elif twgenspec['seedsel'] in ('random_g', 'random_gb'):
            seed_sel_cri = 'random'
        # ------------------------------------------------------------
        '''Select seed coordinate(s) and get neighbouring voxels if needed.'''
        kwargs_nv = {'vs': 1.0,
                     'ret_ind': False,
                     'ret_coords': True,
                     'ret_in_coord': False}
        c_nv = self.get_points_in_feature_coord(feature_type='gb',
                                                selcri=seed_sel_cri,
                                                fcoords=fcoords,
                                                n=1,
                                                get_neigh_vox=False,
                                                kwargs_nv=kwargs_nv,
                                                validate_user_inputs=False)
        selcoord, neigh_vox = c_nv
        # ------------------------------------------------------------
        if twgenspec['K'] <= 2:
            twgenspec['K'] = 5
        '''Find the nearest neighbours of selcoord in CORE_coords.'''
        close_coords_core = self.get_k_nearest_coords_from_tree(CORE_tree,
                                                                selcoord,
                                                                twgenspec['K'])
        '''Select the nearest coordinates in core at random.'''
        n_ = np.random.choice(range(twgenspec['K']), 2, replace=False)
        point1 = close_coords_core[n_[0]]
        point2 = close_coords_core[n_[1]]
        # =====================================================================
        '''Make the seed twin plane.'''
        tp = Plane.from_three_points(selcoord, point1, point2)
        # =====================================================================
        # =====================================================================
        # ------------------------------------------------------------
        '''Get the number of twins.'''
        if twgenspec['bidir_tp'] and type(twspec['n']) in dth.dt.NUMBERS:
            n_twpl = [twspec['n'], twspec['n']]
        elif twgenspec['bidir_tp'] and type(twspec['n']) in dth.dt.ITERABLES:
            n_twpl = [twspec['n'][0], twspec['n'][1]]
        elif not twgenspec['bidir_tp'] and type(twspec['n']) in dth.dt.ITERABLES:
            n_twpl = twspec['n'][0]
        elif not twgenspec['bidir_tp'] and type(twspec['n']) in dth.dt.NUMBERS:
            n_twpl = twspec['n']
        # ------------------------------------------------------------
        '''Get the twin plane translation vector.'''
        twpl_trvec = twspec['tv']
        # ------------------------------------------------------------
        '''Construct the planes which make the twins.'''
        tps = tp.create_translated_planes(twpl_trvec, n_twpl,
                                          dlk=twspec['dlk'],
                                          dnw=twspec['dnw'],
                                          dno=twspec['dno'],
                                          bidrectional=twgenspec['bidir_tp'])
        # ------------------------------------------------------------
        '''Calc. perp distances from each plane to all bounding box coords.'''
        D = [p.calc_perp_distances(BBLGI_locs, signed=False) for p in tps]
        if twspec['sep_bzcz']:
            D_gbz = [p.calc_perp_distances(BCOORDS, signed=False)
                     for p in tps]
            D_core = [p.calc_perp_distances(CORE_coords, signed=False)
                      for p in tps]
        # ------------------------------------------------------------
        '''Calculate twin thickness from user provided data.'''
        if twspec['tdis'] == 'normal':
            twth = np.random.normal(twspec['tpar']['loc'],
                                    twspec['tpar']['scale'])
        elif twspec['tdis'] in ('normal', 'value'):
            twth = twspec['tpar']['val']
        # ------------------------------------------------------------
        '''Identify BBC points which can form twins as per thickness.'''
        TWIN_COORDS = [BBLGI_locs[np.argwhere(d <= twth)].squeeze()
                       for d in D]
        if twspec['sep_bzcz']:
            TWIN_COORDS_gbz = [BCOORDS[np.argwhere(d <= twth)].squeeze()
                               for d in D_gbz]
            TWIN_COORDS_core = [CORE_coords[np.argwhere(d <= twth)].squeeze()
                                for d in D_core]
        # ------------------------------------------------------------
        '''Find the volume of twins created.'''
        vols_of_twins = np.array([tc.shape[0] for tc in TWIN_COORDS])
        # ------------------------------------------------------------
        '''Find out the cut-off twin volumes.'''
        cutoff_twvol = self.get_cutoff_twvol(gid, twspec['vf'])
        # ------------------------------------------------------------
        '''Find the twins which can pass the cut-off volume criteria.'''
        test_1 = vols_of_twins >= cutoff_twvol[0]  # Test against minimum
        test_2 = vols_of_twins <= cutoff_twvol[1]  # Test against maximum
        tested_twins = np.prod((test_1, test_2),
                               axis=0).astype(bool)  # Compile tests
        # ------------------------------------------------------------
        '''Retain the twins which have passed the cut-off volume test.'''
        TWIN_COORDS_local = [TWIN_COORDS[i]
                             for i in range(len(tps)) if tested_twins[i]]
        # ------------------------------------------------------------
        '''Tranbsfer twin coordinates from local to global.'''
        TWIN_COORDS_global = [self.offset_local_to_global(gid, tclcl)
                              for tclcl in TWIN_COORDS_local]
        # ------------------------------------------------------------
        viz_flags={'gb': True,  # Boundary
                   'gc': True,  # Grain core
                   'tb': True,  # Twin boundary
                   'tc': True,  # Twin core
                   'tpvec': False  # Twin plane vectors
                   }
        # ------------------------------------------------------------
        if viz:
            if self.mprop['volnv'][gid] <= 100:
                viz_steps['gb'], viz_steps['gc'] = 1, 1

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if viz_flags['gb']:
                ax.scatter(BCOORDS[::viz_steps['gb'], 0],
                           BCOORDS[::viz_steps['gb'], 1],
                           BCOORDS[::viz_steps['gb'], 2],
                           c='c', marker='o',
                           alpha=0.08, s=60,
                           edgecolors='none')
            if viz_flags['gc']:
                ax.scatter(CORE_coords[::viz_steps['gc'], 0],
                           CORE_coords[::viz_steps['gc'], 1],
                           CORE_coords[::viz_steps['gc'], 2],
                           c='maroon', marker='o',
                           alpha=0.05, s=40,
                           edgecolors='none')
            ax.scatter(selcoord[0], selcoord[1], selcoord[2],
                       c='b', marker='o', alpha=1.0, s=40,
                       edgecolors='black')
            ax.scatter(close_coords_core[:, 0],
                       close_coords_core[:, 1],
                       close_coords_core[:, 2],
                       c='k', marker='o',
                       alpha=1.0, s=10,
                       edgecolors='black')
            ax.scatter(point1[0], point1[1], point1[2],
                       c='k', marker='x', alpha=1.0, s=10,
                       edgecolors='black')
            ax.scatter(point2[0], point2[1], point2[2],
                       c='k', marker='+', alpha=1.0, s=10,
                       edgecolors='black')

            if viz_flags['tpvec']:
                # Starting points of vectors
                vix, viy, viz = selcoord
                vjx, vjy, vjz = close_coords_core.T
                U, V, W = vjx - vix, vjy - viy, vjz - viz
                ax.quiver(vix, viy, viz, U, V, W, color='blue')

            if twspec['sep_bzcz']:
                for tcgbz, tcc in zip(TWIN_COORDS_gbz, TWIN_COORDS_core):
                    if viz_flags['tb']:
                        ax.scatter(tcgbz[:, 0], tcgbz[:, 1], tcgbz[:, 2],
                                   c='black', marker='o',
                                   alpha=0.25, s=20,
                                   edgecolors='black')
                    if viz_flags['tc']:
                        ax.scatter(tcc[:, 0], tcc[:, 1], tcc[:, 2],
                                   c='red', marker='o',
                                   alpha=0.25, s=20,
                                   edgecolors='red')
            else:
                for tc in TWIN_COORDS_local:
                    ax.scatter(tc[:, 0], tc[:, 1], tc[:, 2],
                               c=np.random.random(3),
                               marker='o',
                               alpha=0.25, s=20,
                               edgecolors='red')

        return TWIN_COORDS_global

    def remove_overlaps_in_twins(self, gid, twins,
                                 enforce_twin_vf_check=True,
                                 cutoff_twin_vf=[0.05, 1.00]):
        """
        twins = gstslice.identify_twins_gid(gid,....)
        twins = gstslice.remove_overlaps_in_twins(gid, twins,
                                     enforce_twin_vf_check=True,
                                     cutoff_twin_vf=[0.05, 1.00])
        """
        ntwins = len(twins)
        # ------------------------------------------------------
        if ntwins == 1:
            return twins
        # ------------------------------------------------------
        removal_stats = []
        for i in range(ntwins-1):
            if len(twins[i]) > 0:
                remove = np.array([])
                for coord in twins[i]:
                    indices = np.where(np.all(twins[i+1] == coord, axis=1))[0]
                    remove = np.hstack((remove, indices))
                twins[i+1] = np.delete(twins[i+1], remove.astype(int), axis=0)
                nremove = len(remove)
                ntotal = twins[i].shape[0]
                perc_removed = np.round(nremove*100/ntotal, 0).astype(int)
                removal_stats.append(f"({i}: {nremove}, {perc_removed}%)")
        print(f"ntwins: {i+1}.", "Coord. overlaps:", ", ".join(removal_stats))
        # ------------------------------------------------------
        if enforce_twin_vf_check:
            cutoff_twvol = self.get_cutoff_twvol(gid, cutoff_twin_vf)
        # ------------------------------------------------------
        '''Find the volume of twins created.'''
        vols_of_twins = np.array([tc.shape[0] for tc in twins])
        # ------------------------------------------------------
        '''Find out the cut-off twin volumes.'''
        cutoff_twvol = self.get_cutoff_twvol(gid, cutoff_twin_vf)
        # ------------------------------------------------------
        '''Find the twins which can pass the cut-off volume criteria.'''
        test_1 = vols_of_twins >= cutoff_twvol[0]  # Test against minimum
        test_2 = vols_of_twins <= cutoff_twvol[1]  # Test against maximum
        tested_twins = np.prod((test_1, test_2),
                               axis=0).astype(bool)  # Compile tests
        # ------------------------------------------------------
        '''Retain the twins which have passed the cut-off volume test.'''
        twins = [twins[i] for i in range(ntwins) if tested_twins[i]]
        # --------------------------------
        return twins

    def identify_twins(self,
                       base_gs_name='twin.1',
                       twspec={'n': [5, 10, 3],
                               'tv': np.array([5, -3.5, 5]),
                               'dlk': np.array([1.0, -1.0, 1.0]),
                               'dnw': np.array([0.5, 0.5, 0.5]),
                               'dno': np.array([0.5, 0.5, 0.5]),
                               'tdis': 'normal',
                               'tpar': {'loc': 1.12, 'scale': 0.25, 'val': 1},
                               'vf': [0.05, 1.00],
                               'sep_bzcz': False
                               },
                       twgenspec={'seedsel': 'random_gb',
                                  'K': 10,
                                  'bidir_tp': False,
                                  'checks': [True, True],
                                  },
                       viz=False,
                       ):
        """
        Parameters
        ----------
        seed_selcri: Twin seed selection criteria
        nnp: Number of nearest neighbouring points to use
        nt: Number of twins specification: [lower, upper, iter_threshold]
        tpl_ext: Twin plane extension on either side
        tpl_vec_spec: Twin plane vector specification
        tpl_tvec: Twin plane translation vector
        tth: twin thickness value
        cutoff_twin_vf: Cut-off twin volume fraction
        sep_twin_bz_core: Seperate twin boundary zone and core
        viz: Visualize or not
        viz_flags: Specify various visualization flag values
        viz_steps: Specify7 visualization steps to help with large data

        Exzample
        --------
        from upxo.ggrowth.mcgs import mcgs


        pxt = mcgs()
        pxt.simulate(verbose=False)
        tslice = 49
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=True,
                                           find_spatial_bounds_of_grains=True,
                                           force_compute=True)


        ninstances = 10
        for inst in range(ninstances):
            print(50*'#', 5*'\n',
                  f'Creating instance: {inst} of {ninstances}',
                  5*'\n', 50*'#')
            instance_name = 'twin.'+str(inst)
            gstslice.setup_for_twins(nprops=2,
                                mprops={'volnv': {'use': True,
                                                  'reset': False,
                                                  'k': [.02, 1.0],
                                                  'min_vol': 4,
                                                  },
                                        'rat_sanv_volnv': {'use': True,
                                                           'reset': False,
                                                           'k': [0.0, .8],
                                                           'sanv_N': 26
                                                           },
                                        },
                                instance_name=instance_name,
                                viz_grains=False,
                                opacity=1.0)


            gstslice.identify_twins(base_gs_name=instance_name,
                                    twspec={'n': [5, 10, 3],
                                            'tv': np.array([5, -3.5, 5]),
                                            'dlk': np.array([1.0, -1.0, 1.0]),
                                            'dnw': np.array([0.5, 0.5, 0.5]),
                                            'dno': np.array([0.5, 0.5, 0.5]),
                                            'tdis': 'normal',
                                            'tpar': {'loc': 1.12, 'scale': 0.25, 'val': 1},
                                            'vf': [0.05, 1.00],
                                            'sep_bzcz': False
                                            },
                                    twgenspec={'seedsel': 'random_gb',
                                               'K': 10,
                                               'bidir_tp': False,
                                               'checks': [True, True],
                                               },
                                    viz=False,
                                    )


        # gid = gstslice.gid[]
        # gid = gstslice.get_largest_gids()[0]
        gid = np.random.choice(list(gstslice.fdb['twin.7']['data']['twin_map_g_t'].keys()),
                               1
                               )[0]
        fid = gstslice.fdb['twin.7']['data']['fid']
        twin_gids = gstslice.fdb['twin.7']['data']['twin_map_g_t'][gid]


        import pyvista as pv
        pvgrid = pv.UniformGrid()
        pvgrid.dimensions = np.array(gstslice.lgi.shape) + 1
        pvgrid.origin = (0, 0, 0)
        pvgrid.spacing = (1, 1, 1)
        pvgrid.cell_data['lgi'] = gstslice.lgi.flatten(order="F")
        pvgrid.plot(cmap='nipy_spectral')


        pvp = pv.Plotter()
        thresholded = pvgrid.threshold([gid, gid])
        pvp.add_mesh(thresholded, cmap='nipy_spectral', show_edges=False, opacity=0.25)
        for twin_gid in twin_gids:
            thresholded = pvgrid.threshold([twin_gid, twin_gid])
            if thresholded.cells.size > 0:
                pvp.add_mesh(thresholded, cmap='nipy_spectral', show_edges=True, opacity=1.0)
        pvp.show()



        instance_no = 2
        feat_instance_name = 'twin.'+str(instance_no)
        fid = gstslice.fdb[feat_instance_name]['data']['fid']
        pvgrid = pv.UniformGrid()
        pvgrid.dimensions = np.array(fid.shape) + 1
        pvgrid.origin = (0, 0, 0)
        pvgrid.spacing = (1, 1, 1)
        pvgrid.cell_data['fid'] = fid.flatten(order="F")
        # pvgrid.plot(cmap='nipy_spectral')


        gids_all = list(gstslice.fdb[feat_instance_name]['data']['twin_map_g_t'].keys())


        nr, nc = 6, 6


        gids = np.reshape(np.random.choice(gids_all, nr*nc, replace=False), (nr, nc))


        pvp = pv.Plotter(shape=(nr, nc))
        for gidr in range(nr):
            for gidc in range(nc):
                print(f'gidr: {gidr}, gidc: {gidc}')
                pvp.subplot(gidr, gidc)
                gid = gids[gidr][gidc]
                thresholded = pvgrid.threshold([gid, gid])
                if thresholded.cells.size == 0:
                    break
                pvp.add_mesh(thresholded, cmap='nipy_spectral', show_edges=False, opacity=0.5)

                twin_gids = gstslice.fdb[feat_instance_name]['data']['twin_map_g_t'][gid]
                for twin_gid in twin_gids:
                    thresholded = pvgrid.threshold([twin_gid, twin_gid])
                    if thresholded.cells.size > 0:
                        pvp.add_mesh(thresholded, cmap='nipy_spectral', show_edges=True, opacity=1.0)
        pvp.show()


        import pyvista as pv
        pvgrid = pv.UniformGrid()
        pvgrid.dimensions = np.array(fid.shape) + 1
        pvgrid.origin = (0, 0, 0)
        pvgrid.spacing = (1, 1, 1)
        pvgrid.cell_data['fid'] = fid.flatten(order="F")
        pvgrid.plot(cmap='nipy_spectral')



        feat_instance_name = 'twin.9'
        fid = gstslice.fdb[feat_instance_name]['data']['fid']
        pvgrid = pv.UniformGrid()
        pvgrid.dimensions = np.array(fid.shape) + 1
        pvgrid.origin = (0, 0, 0)
        pvgrid.spacing = (1, 1, 1)
        pvgrid.cell_data['fid'] = fid.flatten(order="F")



        pvp = pv.Plotter()
        gid = 131
        thresholded = pvgrid.threshold([gid, gid])
        pvp.add_mesh(thresholded, cmap='nipy_spectral', show_edges=False, opacity=0.25)
        twin_gids = gstslice.fdb[feat_instance_name]['data']['twin_map_g_t'][gid]
        for twin_gid in twin_gids:
            thresholded = pvgrid.threshold([twin_gid, twin_gid])
            if thresholded.cells.size > 0:
                pvp.add_mesh(thresholded, cmap='nipy_spectral', show_edges=True, opacity=1.0)
        pvp.show()


        """
        # VALIDATIONS
        # =====================================================================
        if base_gs_name not in self.fdb.keys():
            raise ValueError(f'base_gs_name: {base_gs_name} is invalid',
                             'It must be a key in self.fdb.')
        else:
            if len(self.fdb) == 0:
                print(30*'#', '\ngstslice.fdb has not been set.',
                      '\nSet using gstslice.add_fdb(..)')
        # =====================================================================
        nt = twspec['n']
        ngids = len(self.gid_twin.keys())
        perc_complete = np.round(np.arange(1, ngids+1, 1)*100/ngids, 0).astype(int)
        for gid_count, gid in enumerate(self.gid_twin.keys(), start=0):
            # ------------------------------------------------
            print(5*'#', 50*'-', 5*'#',
                  f'\nFinding twins in gid: {gid} ({perc_complete[gid_count]}%)',
                  f'\n  : grain no. {gid_count} of {ngids}\n')
            ntrials, ntwins, twin_set_count = 0, 0, 0
            while ntwins < nt[2]:
                twin_set_count += 1
                try:
                    print(f'Twin set number: {twin_set_count}')
                    twspec['n'] = np.random.choice(np.arange(nt[0], nt[1], 1),
                                                   replace=False,)
                    twins = self.identify_twins_gid(gid,
                                                    twspec=twspec,
                                                    twgenspec=twgenspec,
                                                    viz=viz,)
                    twins = self.remove_overlaps_in_twins(gid,
                                                          twins,
                                                          enforce_twin_vf_check=twgenspec['checks'][0],
                                                          cutoff_twin_vf=twspec['vf'],)
                    ntwins = len(twins)
                    if ntwins > 0:
                        print(f'Twin inclusion run: HIT. {ntwins} twin sets')
                except Exception as e:
                    twins = None
                    ntwins = 1000  # Just a large number to break the loop
                    print('Twin inclusion run: MISS.')
                ntrials += 1
                self.gid_twin[gid] = twins

            print(f'No. of trials: {ntrials}')

        # Global twin ID
        GTID = max(self.gid) + 1
        # Twin count number
        twin_i = []
        # Twin ID number --> in line with the GID number. Starts at GID.max()+1
        twin_id = []
        # Twin volume - in line with twin ID number
        twin_vol = []
        # Twin volkume fraction
        twin_vf = []
        # gid - twin ID map
        twin_map_g_t = {}
        # gid - Number of twins
        twin_map_g_nt = {}
        # ------------------------------------------------
        _LGI_ = deepcopy(self.fdb[base_gs_name]['data']['fid'])
        # ------------------------------------------------
        twin_i_count = 0  # Twin count number
        tvol = 0  # Total twin volume in grain structure
        for gid, twins in self.gid_twin.items():
            ntwins_gid = 0
            '''Find the current gid volume.'''
            _gid_vol_ = self.mprop['volnv'][gid]
            '''Initiate this twin_map_g_t dict for gid: []'''
            twin_map_g_t[gid] = []
            '''Iterative over all twins in this grain.'''
            if twins is not None:
                for twin in twins:
                    '''Work on the current twqin.'''
                    if twin.shape[0] > 0:
                        '''Find this twin's volume and volume fraction.'''
                        _twin_vol_ = twin.shape[0]
                        tvol += _twin_vol_
                        _twin_vf_ = _twin_vol_/_gid_vol_
                        twin_i.append(twin_i_count)
                        twin_vol.append(_twin_vol_)
                        twin_vf.append(_twin_vf_)
                        twin_id.append(GTID)
                        '''
                        As per user request, perform secondary check over Vf
                        bounds. UPdate data accordingly. Following data to be
                        updat4ed:
                            1. Local twin count il.e. twin count in this gid
                            2. Global twin count i.e. twin count across all
                            hosting gids.
                            3.
                        '''
                        if twgenspec['checks'][1]:
                            '''Secondary twin volume fraction check.
                            Select only the qualifying twins.'''
                            if _twin_vf_ >= twspec['vf'][0] and _twin_vf_ <= twspec['vf'][1]:
                                '''Update this twin coordinstes in lgi data.'''
                                for tc in twin:
                                    _LGI_[tc[0], tc[1], tc[2]] = GTID
                                '''Update the local twin counbt.'''
                                ntwins_gid += 1
                                twin_i_count += 1
                                twin_map_g_t[gid].append(GTID)
                                '''UPdate the twin ID number.'''
                                GTID += 1
                        else:
                            '''Update this twin coordinstes in lgi data.'''
                            for tc in twin:
                                _LGI_[tc[0], tc[1], tc[2]] = GTID
                            '''Update the local twin counbt.'''
                            ntwins_gid += 1
                            twin_i_count += 1
                            twin_map_g_t[gid].append(GTID)
                            '''UPdate the twin ID number.'''
                            GTID += 1
                    else:
                        twin_vf.append(0)
            # Update the number of twins value for this grain
            twin_map_g_nt[gid] = ntwins_gid
        # -------------------------------------------------
        self.fdb[base_gs_name]['data']['fid'] = deepcopy(_LGI_)
        # -------------------------------------------------
        # Convert values to Numpy arrays
        # self.fdb[base_gs_name]['data']['twin_i'] = np.array(twin_i)
        self.fdb[base_gs_name]['data']['twin_id'] = np.array(twin_id)
        self.fdb[base_gs_name]['data']['twin_vol'] = np.array(twin_vol)
        self.fdb[base_gs_name]['data']['twin_vf'] = np.array(twin_vf)
        notwin_gids = np.arange(1, self.lgi.max()+1, 1)
        self.fdb[base_gs_name]['data']['notwin_gids'] = notwin_gids
        self.fdb["twin.0"]["data"]["twin_coords"] = self.gid_twin
        self.fdb[base_gs_name]['data']['twin_map_g_t'] = twin_map_g_t
        self.fdb[base_gs_name]['data']['twin_map_g_nt'] = twin_map_g_nt
        # Reset gid_twin database as its a;lready saved as twin_coords
        self.gid_twin = None

    def instantiate_twins(self,
                          ninstances=2,
                          base_gs_name_prefix='twin.',
                          twin_setup={'nprops': 2,
                                      'mprops': {'volnv': {'use': True,
                                                           'reset': False,
                                                           'k': [.02, 1.0],
                                                           'min_vol': 4,
                                                           },
                                                 'rat_sanv_volnv': {'use': True,
                                                                    'reset': False,
                                                                    'k': [0.0, .8],
                                                                    'sanv_N': 26
                                                                    },
                                                 }
                                      },
                          twspec={'n': [5, 10, 3],
                                  'tv': np.array([5, -3.5, 5]),
                                  'dlk': np.array([1.0, -1.0, 1.0]),
                                  'dnw': np.array([0.5, 0.5, 0.5]),
                                  'dno': np.array([0.5, 0.5, 0.5]),
                                  'tdis': 'normal',
                                  'tpar': {'loc': 1.12, 'scale': 0.25, 'val': 1},
                                  'vf': [0.05, 1.00],
                                  'sep_bzcz': False
                                  },
                          twgenspec={'seedsel': 'random_gb',
                                     'K': 10,
                                     'bidir_tp': False,
                                     'checks': [True, True],
                                     },
                          orimapspec={'schemeID': 1,
                                      'easets': {1: {'name': '--',
                                                     'mean': [90, 90, 90],
                                                     'width': 15,
                                                     'ea': [[35, 45, 0],
                                                            [63, 75, 27]
                                                            ]
                                                     }
                                                 }
                                      },
                          reset_fdb=True,
                          reset_keystring='twin.',
                          make_sep_pvgrds=False,
                          save_twin_coords=False,
                          clean_verbosity_interval_1=250
                          ):
        """
        # -----> FEATURE IDS

        gstslice.fdb["twin.0"]["data"]["feat_host_gids"] This has all gids
        initialliy selected for introducing twins. For grain ids which actually
        host twins, refer gstslice.fdb["twin.0"]["data"]["feat_host_ids"]
        instead.

        gstslice.fdb["twin.0"]["data"]["feat_host_ids"] Feature (i.e. grain)
        ids which actually host twins.

        gstslice.fdb["twin.0"]["data"]["twin_map_g_t_missed"] Numpy array
        containing the list of parent feature IDs which were misses duringh
        twin instantiation. These are featre IDs initially selected for twin
        generation, but later rejected as twins could not be produced with the
        user specified algorithm control parameter values.

        gstslice.fdb["twin.0"]["data"]["notwin_gids"] Parent features i.e.
        grains, which were not selected to parent any children i.e. twins.

        gstslice.fdb["twin.0"]["data"]["twin_id"] Twin ID numbers

        gstslice.fdb["twin.0"]["data"]["twin_map_g_t"] Contains parent to
        children map. As a dictionary, keys are parent feature IDs. A value is
        a list of children feature IDs, which in this case are twin IDs.

        gstslice.fdb["twin.0"]["data"]["map_cp"] Dictiobnary having keys as
        twin IDs and values as parent IDs. This is a reverse map of twin ID
        to parent ID. len(gstslice.fdb["twin.0"]["data"]["map_cp"].keys())
        gives the total number of children, which can also abe obtained as
        sum(gstslice.fdb["twin.0"]["data"]["twin_map_g_nt"].values()). In fact,
        in  some UPXO twined grain structure,
        len(gstslice.fdb["twin.0"]["data"]["map_cp"].keys()) was 383,
        sum(gstslice.fdb["twin.0"]["data"]["twin_map_g_nt"].values()) was also
        383 and len(gstslice.fdb["twin.0"]["data"]["twin_id"]) was also 383.
        # -----------------------------------------
        # -----> DEPRECATED

        gstslice.fdb["twin.0"]["data"]["twin_i"] DEPRECATED. Commented in code.

        gstslice.fdb["twin.0"]["data"]["parent_id"] numpy array of parent IDs
        which actually host children features. DEPRECTAED. This is the same as
        gstslice.fdb["twin.0"]["data"]["feat_host_ids"]. Commented out in code.

        gstslice.fdb["twin.0"]["data"]["twin_zero_voxels"] List of twin IDs
        with zero voxels. Should be empty. To be deprecated.
        # -----------------------------------------
        # -----> COORDINATES

        gstslice.fdb["twin.0"]["data"]["twin_coords"] Coordinates of twins.
        len(gstslice.fdb["twin.0"]["data"]["twin_coords"]) is same as
        len(gstslice.fdb["twin.0"]["data"]["parent_id"]).

        gstslice.fdb["twin.0"]["data"]["twin_map_g_t_coords"] Contains voxel
        coordinates of the twinned regions. Keys are parent feature ID. Value
        is dict with keys as children feature IDs and values being
        corresponding numpy cooridnate arrays.
        # -----------------------------------------
        # -----> SIZE AND CONTENT PROPERTIES

        gstslice.fdb["twin.0"]["data"]["twin_vol"] Twin volumes

        gstslice.fdb["twin.0"]["data"]["twin_vf"] Twin volume fractions

        gstslice.fdb["twin.0"]["data"]["twin_map_g_nt"] Contains parent to
        children count map. As a dictionary, keys are parent feature IDs. A
        value is the length of the list of children feature IDs.

        gstslice.fdb["twin.0"]["data"]["twin_vol_total"] This is total volume
        of twins across the entire domain. It is numpy.int32 type.

        gstslice.fdb["twin.0"]["data"]["twin_vf_total"] This is the overall
        volume fraction of twins.

        gstslice.fdb["twin.0"]["data"]["twin_map_g_t_nvox"] Contains parent to
        children voxel count map. As a dictionary, keys are parent feature IDs.
        A value is the list of total number of voxels in each child feature.
        List size will be the total child feature count for the current parent
        feature ID.

        gstslice.fdb["twin.0"]["data"]["twin_nvox"] Total number of voxels
        across all children features (dict value) for a given parent feature
        ID (dict key).
        # -----------------------------------------
        gstslice.fdb["twin.0"]["data"]["pvgrid"] This is the PyVista grid
        having gstslice.fdb["twin.0"]["data"]["fid"] as scalar field.

        gstslice.fdb["twin.0"]["data"]["parent_feature"] String value
        containing, the name of the parent feature, which in this case is
        'grain'.


        EXAMPLE
        -------
        import time
        from upxo.ggrowth.mcgs import mcgs

        start_time = time.time()

        pxt = mcgs(input_dashboard='input_dashboard.xls')
        pxt.simulate(verbose=False)
        tslice = 49
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=True,
                                           find_spatial_bounds_of_grains=True,
                                           force_compute=True)
        mprops = {'volnv': {'use': True, 'reset': False,
                            'k': [.02, 1.0], 'min_vol': 4,},
                  'rat_sanv_volnv': {'use': True, 'reset': False,
                                     'k': [0.0, .8], 'sanv_N': 26},}
        twspec = {'n': [5, 10, 3],
                'tv': np.array([5, -3.5, 5]),
                'dlk': np.array([1.0, -1.0, 1.0]),
                'dnw': np.array([0.5, 0.5, 0.5]),
                'dno': np.array([0.5, 0.5, 0.5]),
                'tdis': 'normal',
                'tpar': {'loc': 1.12, 'scale': 0.25, 'val': 1},
                'vf': [0.05, 1.00], 'sep_bzcz': False}
        twgenspec = {'seedsel': 'random_gb', 'K': 10,
                   'bidir_tp': False, 'checks': [True, True],}
        gstslice.instantiate_twins(ninstances=10, base_gs_name_prefix='twin.',
                                   twin_setup={'nprops': 2, 'mprops': mprops},
                                   twspec=twspec,
                                   twgenspec=twgenspec,)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.6f} seconds")
        """
        if not isinstance(ninstances, int):
            raise ValueError('Invalid ninstances input.')
        if type(base_gs_name_prefix) in dth.dt.NUMBERS:
            base_gs_name_prefix = str(base_gs_name_prefix) + '.'
        if not isinstance(base_gs_name_prefix, str):
            raise ValueError('Invalid base_gs_name_prefix input.')
        instance_names = []
        # -----------------------------------------------------------
        '''Wipe the slate clean.'''
        if reset_fdb:
            for key in list(self.fdb.keys()):
                if key.startswith(reset_keystring):
                    del self.fdb[key]
        # -----------------------------------------------------------
        for inst in range(ninstances):
            print(50*'#', 5*'\n',
                  f'Creating instance: {inst+1} of {ninstances}',
                  5*'\n', 50*'#')
            instance_name = base_gs_name_prefix+str(inst)
            instance_names.append(instance_name)
            self.setup_for_twins(nprops=twin_setup['nprops'],
                                 mprops=twin_setup['mprops'],
                                 instance_name=instance_name,
                                 viz_grains=False)
            self.identify_twins(base_gs_name=instance_name,
                                twspec={'n': twspec['n'],
                                        'tv': twspec['tv'],
                                        'dlk': twspec['dlk'],
                                        'dnw': twspec['dnw'],
                                        'dno': twspec['dno'],
                                        'tdis': twspec['tdis'],
                                        'tpar': twspec['tpar'],
                                        'vf': twspec['vf'],
                                        'sep_bzcz': twspec['sep_bzcz']
                                        },
                                twgenspec=twgenspec,
                                viz=False,)
            if self.pvgrid is not None:
                self.add_scalar_field_to_pvgrid(sf_name=instance_name,
                                                sf_value=self.fdb[instance_name]['data']['fid'])
        # -----------------------------------------------------------
        self.fdb[instance_name]['info']['pvgrid'] = False
        if make_sep_pvgrds:
            pvgrid = pv.ImageData()
            pvgrid.dimensions = np.array(self.lgi.shape) + 1
            pvgrid.origin = (0, 0, 0)
            pvgrid.spacing = (1, 1, 1)
            pvgrid.cell_data['fid'] = self.fdb[instance_name]['data']['fid'].flatten(order="F")
            self.fdb[instance_name]['data']['pvgrid'] = pvgrid
            self.fdb[instance_name]['info']['pvgrid'] = True
        # -----------------------------------------------------------
        '''
        1. Clean up the self.fdb[instance_name]['data']['twin_map_g_t'] data.
        It contains lots of keys witgh [] values. Keys reprezsent parent grain
        IDs and value represents list of twin IDs. The empty bvalues means twin
        creation proicess was unseccessful for the given gid and therefore a
        missed attempt. No need to record the missed parent grain in this data.
        2. Build data for the number of twins contained by a parent grain.
        3. Calcuklate number of voxels of each non zero voxel twin.
        4. Extract twin IDs having zero voxels. I think I saw a unique case
        where one of the twin had zero voxel locations. Never happened again so
        far. To be on the safewr side, I have introduced this to record any
        such instance. May be not needed. For later deascrewtion. Will move on
        taking this along fornow. VARIABLE TO BE TRUNCATED ?
        '''
        for instance_name in instance_names:
            print(40*'#')
            print(f"Data cleaning underway for twinning instance {instance_name}.")
            '''Parent grain to twin ID map'''
            twin_map_g_t = {}
            '''Number of twins.'''
            twin_map_g_nt = {}
            '''Voxel coordinates of all twins'''
            twin_map_g_t_coords = {}
            '''Number of voxels of each twin'''
            twin_map_g_t_nvox = {}
            '''Parent gids with missed twin generation attempts.'''
            twin_map_g_t_missed = []
            '''Twin IDs with zero voxels. VARIABLE TO BE TRUNCATED ?'''
            twin_zero_voxels = []
            # Number of pids: npids
            # pid_count: iteration variable, increments ny a unit every iteration.
            npids, pid_count = self.fdb[instance_name]['data']['feat_host_gids'].size, 1
            for pid, twin_ids in self.fdb[instance_name]['data']['twin_map_g_t'].items():
                if pid_count % clean_verbosity_interval_1 == 0:
                    print(f".... parent grain number {pid_count} of {npids}")
                if len(twin_ids) > 0:
                    twcoords = [self.find_feature_voxel_locs(fname=instance_name,
                                                             fids=[twid],
                                                             verbosity=10,
                                                             printmsg=False,
                                                             saa=False,
                                                             throw=True)
                                for twid in twin_ids]
                    twin_map_g_t__pid = []
                    twin_map_g_t_coords__pid = {}
                    nvox = []
                    for twid, twc in zip(twin_ids, twcoords):
                        if twc[twid].shape[0] > 0:
                            twin_map_g_t__pid.append(twid)
                            twin_map_g_t_coords__pid[twid] = twc[twid]
                            nvox.append(twc[twid].shape[0])
                    if sum(twin_map_g_t__pid) > 0:
                        twin_map_g_t[pid] = twin_map_g_t__pid
                        twin_map_g_nt[pid] = len(twin_map_g_t__pid)
                        twin_map_g_t_coords[pid] = twin_map_g_t_coords__pid
                        twin_map_g_t_nvox[pid] = nvox
                    else:
                        twin_zero_voxels.append(twid)
                else:
                    twin_map_g_t_missed.append(pid)
                pid_count += 1
            twin_map_g_t_missed = np.array(twin_map_g_t_missed)

            # The parent grains which actually host twins are:
            hosts = np.array(list(twin_map_g_t.keys()))

            # Clean the twin _coordinates data base as it contains parent IDs
            # which were actually missed during the twin inclusion process.
            self.fdb[instance_name]["data"]["twin_coords"] = {k: v for k, v in self.fdb["twin.0"]["data"]["twin_coords"].items() if v != None}
            # Correct twin volume data. Just deprecating it through deletion,
            # I dont want to delete at source as of now. May be later. Volumes
            # already present in twin_nvox
            del self.fdb[instance_name]["data"]["twin_vol"]
            self.fdb[instance_name]['data']['feat_host_ids'] = hosts
            self.fdb[instance_name]['data']['twin_map_g_t'] = twin_map_g_t
            self.fdb[instance_name]['data']['twin_map_g_nt'] = twin_map_g_nt
            self.fdb[instance_name]['data']['twin_map_g_t_nvox'] = twin_map_g_t_nvox
            self.fdb[instance_name]['data']['twin_nvox_sum'] = np.array([sum(nv) for nv in twin_map_g_t_nvox.values()])

            D = {}
            for key, val in self.fdb[instance_name]['data']['twin_map_g_t'].items():
                for fid_count, fid_ in enumerate(val):
                    D[fid_] = self.fdb[instance_name]['data']['twin_map_g_t_nvox'][key][fid_count]
            self.fdb[instance_name]['data']['twin_nvox'] = D

            if save_twin_coords:
                self.fdb[instance_name]['data']['twin_map_g_t_coords'] = twin_map_g_t_coords
            self.fdb[instance_name]['data']['twin_map_g_t_missed'] = twin_map_g_t_missed
            self.fdb[instance_name]['data']['twin_zero_voxels'] =  twin_zero_voxels
            self.fdb[instance_name]['data']['twin_vol_total'] = sum(self.fdb[instance_name]['data']['twin_nvox'])
            parent_ids = self.fdb[instance_name]['data']['twin_map_g_t_nvox'].keys()
            # Correct twin volume fraction data and just re-writing it!!
            parent_vols = np.array([self.grain_locs[pid].shape[0] for pid in parent_ids])
            tw_vfs = np.array([twvol/pgrainvol for twvol, pgrainvol in zip(self.fdb[instance_name]['data']['twin_nvox'], parent_vols)])
            self.fdb[instance_name]['data']['twin_vf'] = tw_vfs
            self.fdb[instance_name]['data']['twin_vf_total'] = self.fdb[instance_name]['data']['twin_vol_total'] / self.domvol
            print('\n', f"\nTwin volume fraction: {self.fdb[instance_name]['data']['twin_vf_total']}")
            # -------------------------------------------------
            """
            NOTE: THE FOLLOWING TWO MUST BE UPDATED. FOR LATER WORK. I will set
            these to None for now.
            """
            self.fdb[instance_name]['data']['twin_id'] = np.hstack([twids for twids in self.fdb[instance_name]['data']['twin_map_g_t'].values()])
            # self.fdb[instance_name]['data']['twin_i'] = list(range(self.fdb[instance_name]['data']['twin_id'].size))

            # self.fdb[instance_name]['data']['parent_id'] = np.array(list(self.fdb[instance_name]['data']['twin_map_g_t'].keys()))
            self.fdb[instance_name]['data']['parent_feature'] = 'grain'
            # Set ip Child to parnet mapping
            self.fdb[instance_name]['data']['map_cp'] = {cid: None for cid in self.fdb[instance_name]['data']['twin_id']}
            for pid, cids in self.fdb[instance_name]['data']['twin_map_g_t'].items():
                for cid in cids:
                    self.fdb[instance_name]['data']['map_cp'][cid] = pid
            '''
            Start the sanity checks.
            '''
            print("\n\nPerforming sanity checks")
            fails = 0
            for i, host in enumerate(self.fdb[instance_name]['data']['feat_host_ids']):
                if i % 500 == 0:
                    print(f"host {i} of {len(self.fdb[instance_name]['data']['feat_host_ids'])}")
                twin_ids = twin_map_g_t[host]
                # Number of voxels in the parent grain id before twin introduction
                a = np.where(self.lgi==host)[0].size
                # Number of voxels in the parent grain id after twin introduction
                b = np.where(self.fdb[instance_name]['data']['fid']==host)[0].size
                # Number of voxels in individual twins
                c = [np.where(self.fdb[instance_name]['data']['fid']==twid)[0].size
                     for twid in twin_ids]
                # Residual number of voxels, MUST be zero.
                if a-(b+sum(c)) != 0:
                    # <--- This completes sanity check number 1
                    fails += 1
                # Number of parent voxels
                npvox = self.grain_locs[host].shape[0]
                # Number of twin voxels
                twsvox = twin_map_g_t_nvox[host]
                # Number of voxels in the parent grain id after twin introduction
                # thatis, the residual number of voxels
                d = npvox-sum(twsvox)
                # This must be the same as b
                if d != b:
                    fails += 1
                    # <--- This completes the sanity check number 2
            print(f"........  {2*(len(hosts)-fails)} of {2*len(hosts)} sanity checks passed.")
            print(f"Performing additional checks.")
            a = len(self.fdb[instance_name]["data"]["twin_vf"])
            b = len(self.fdb[instance_name]["data"]["twin_map_g_t"])
            c = len(self.fdb[instance_name]["data"]["twin_map_g_nt"])
            d = len(self.fdb[instance_name]["data"]["twin_map_g_t_nvox"])
            e = len(self.fdb[instance_name]["data"]["twin_map_g_t_coords"])
            f = len(self.fdb[instance_name]["data"]["twin_nvox"])
            if a == b == c == d == e == f == 1:
                print('Array length equality check 1 passed.')
            else:
                print('Array length equality check 1 failed.')
            a = len(self.fdb[instance_name]["data"]['feat_host_gids'])
            b = len(self.fdb[instance_name]["data"]["twin_map_g_t_missed"])
            c = len(self.fdb[instance_name]["data"]["feat_host_ids"])
            if a == b+c:
                print('Array length equality check 2 passed.')
            else:
                print('Array length equality check 2 failed.')
        print(40*'#')

    def get_coords_parents_minus_childrens(self, pfids=None, pfeatname='grain',
                                           instance_name='twin.0', disp_msg=True):
        """
        Example
        -------
        pfids = gstslice.fdb['twin.0']['data']['feat_host_ids']
        # Parent minus childs
        pminuscs_coords = gstslice.get_coords_parents_minus_childrens(pfids=pfids,
                                                             pfeatname='grain',
                                                             instance_name='twin.0',
                                                             disp_msg=True)
        """
        print(40*'-', "\nCalculating coordinates of parent-children voxels of parent pfids.")
        pc_rem = {pfid: -1 for pfid in pfids}
        for pfid in pfids:
            if pfid in self.fdb[instance_name]['data']['feat_host_ids']:
                pc = self.grain_locs[pfid]
                twc = self.fdb[instance_name]['data']['twin_map_g_t_coords'][pfid]
                twc_acc = np.vstack(tuple(twc.values()))
                pc = np.ascontiguousarray(pc)
                twc_acc = np.ascontiguousarray(twc_acc)
                mask = ~np.in1d(pc.view([('', pc.dtype)]*3),
                                twc_acc.view([('', twc_acc.dtype)]*3))
                pc_rem[pfid] = pc[mask]
        if disp_msg:
            print("\nNOTE:", "\n-----")
            print("A 'key:value' pair, 'pfid:-1' indicates invalid pfid due to:")
            print("    * Invalid user entry.",
                  "\n    * valid pfid hosting no children.")
        invalids = np.where([type(pc_rem[pfid])==int for pfid in pfids])[0]
        if invalids.size == 0:
            print("Currently all pfids are correct.")

        return pc_rem

    def extract_representative_voxel(self, instance_name='base', featname='grains',):
        pass

    def break_disjoint_twins(self):
        pass

    def extract_feat_coords(self,
                            instance_name='twin.0',
                            feature_name='twins',
                            use_parent_ids=True,
                            pfids=None,
                            cfids=None):
        """
        Example-1
        ---------
        pfids = gstslice.fdb['twin.0']['data']['feat_host_ids'][:2]
        gstslice.extract_feat_coords(instance_name='twin.0',
                                     feature_name='twins',
                                     use_parent_ids=True,
                                     pfids=pfids)

        Example-2
        ---------
        cfids = list(gstslice.fdb['twin.0']['data']['map_cp'])[:4]
        gstslice.extract_feat_coords(instance_name='twin.0',
                                     feature_name='twins',
                                     use_parent_ids=False,
                                     cfids=cfids)
        """
        # -----------------------------------------
        # Validations
        if use_parent_ids:
            if not pfids:
                raise ValueError("Invalid pfids specied.")
            if type(pfids) not in dth.dt.ITERABLES:
                pfids = [pfids]
        else:
            if not cfids:
                raise ValueError("Invalid cfids specied.")
            if type(cfids) not in dth.dt.ITERABLES:
                pfids = [cfids]
        # -----------------------------------------
        if instance_name[:4] == 'twin':
            if feature_name == 'twins':
                fc_global = self.fdb[instance_name]['data']['twin_map_g_t_coords']
                if use_parent_ids:
                    # idmap = gstslice.fdb[instance_name]['data']['twin_map_g_t']
                    # fids = {pfid: idmap[pfid] for pfid in pfids}
                    fc = {pfid: fc_global[pfid] for pfid in pfids}
                else:
                    # Retrieve the parent IDs first of the given cfids
                    pfids = {cid: self.fdb[instance_name]['data']['map_cp'][cid]
                             for cid in cfids}
                    # Child feature set coordinates
                    cfset_coords = {cfid: None for cfid in cfids}
                    for cfid in cfids:
                        cfset_coords[cfid] = fc_global[pfids[cfid]][cfid]
                    fc = cfset_coords
        return fc

    def calc_bounds(self, instance_name='base', featname='grains', recalc=True,
                    use_parent_ids=True, pfids=None, cfids=None,
                    find_extended_bounds=False):
        """
        EXAMPLE - 1
        -----------
        gstslice.calc_bounds(instance_name='base', featname='grains',
                             recalc=False)
        gstslice.calc_bounds(instance_name='base', featname='grains',
                             recalc=True)

        EXAMPLE - 2
        -----------
        pfids = gstslice.fdb['twin.0']['data']['feat_host_ids'][:2]
        gstslice.calc_bounds(instance_name='twin.0', featname='twins',
                             use_parent_ids=True, pfids=pfids)

        EXAMPLE - 3
        -----------
        cfids = list(gstslice.fdb['twin.0']['data']['map_cp'])[:4]
        gstslice.calc_bounds(instance_name='twin.0', featname='twins',
                             use_parent_ids=False, cfids=cfids)
        """
        # First find the voxel locations
        if instance_name[:4] == 'base':
            if featname == 'grains':
                if recalc:
                    self.find_grain_voxel_locs(disp_msg=False, verbosity=10,
                                               saa=True, throw=False)
        elif instance_name[:4] in ('twin',):
            if featname == 'twins':
                if use_parent_ids:
                    fc = self.extract_feat_coords(instance_name=instance_name,
                                                  feature_name='twins',
                                                  use_parent_ids=True,
                                                  pfids=pfids)
                else:
                    fc = self.extract_feat_coords(instance_name=instance_name,
                                                  feature_name='twins',
                                                  use_parent_ids=False,
                                                  cfids=cfids)
        # ------------------------------------------
        # Now find the featre bounds
        if instance_name[:4] == 'base':
            self.find_spatial_bounds_of_grains()
            print("Completed. Refer gstslice.spbound and gstslice.spboundex")

        # Now find the featre bounds
        if instance_name[:4] in ('twin',):
            if use_parent_ids:
                spbound = {pfid: None for pfid in fc.keys()}
                if find_extended_bounds:
                    spbound_ex = {pfid: None for pfid in fc.keys()}
                # ------------------------------------------------------
                print('\nFinding spatial bounds of all requested features.')
                for pfid in fc.keys():
                    zmins = np.array([loc[:, 0].min() for loc in fc[pfid].values()])
                    zmaxs = np.array([loc[:, 0].max() for loc in fc[pfid].values()])
                    ymins = np.array([loc[:, 1].min() for loc in fc[pfid].values()])
                    ymaxs = np.array([loc[:, 1].max() for loc in fc[pfid].values()])
                    xmins = np.array([loc[:, 2].min() for loc in fc[pfid].values()])
                    xmaxs = np.array([loc[:, 2].max() for loc in fc[pfid].values()])
                    # -------------------------------
                    spbound[pfid] = {'xmins': xmins, 'xmaxs': xmaxs,
                                     'ymins': ymins, 'ymaxs': ymaxs,
                                     'zmins': zmins, 'zmaxs': zmaxs,
                                     'cfids': list(fc[pfid].keys())}
                    if find_extended_bounds:
                        ymins_ex = ymins-(ymins>0)*1
                        zmins_ex = zmins-(zmins>0)*1
                        zmaxs_ex = zmaxs+(zmaxs<self.lgi.shape[0]-1)*1
                        ymaxs_ex = ymaxs+(ymaxs<self.lgi.shape[1]-1)*1
                        xmins_ex = xmins-(xmins>0)*1
                        xmaxs_ex = xmaxs+(xmaxs<self.lgi.shape[2]-1)*1
                        spbound_ex[pfid] = {'xmins': xmins_ex,
                                            'xmaxs': xmaxs_ex,
                                            'ymins': ymins_ex,
                                            'ymaxs': ymaxs_ex,
                                            'zmins': zmins_ex,
                                            'zmaxs': zmaxs_ex}
                self.fdb[instance_name]['data']['spbound'] = spbound
                print(40*"-", "\nCompleted.")
                print(f"Refer gstslice.fdb['{instance_name}']['data']['spbound']")
                if find_extended_bounds:
                    self.fdb[instance_name]['data']['spbound_ex'] = spbound_ex
                    print(f"Refer gstslice.fdb['{instance_name}']['data']['spboundex'] for extended spatial bounds")

    def calc_vox_info(self):
        """
        gstslice.calc_vox_info()
        """
        nx, ny = int(len(self.gridx)), int(len(self.gridy))
        nz = int(len(self.gridz))
        n = nx*ny*nz
        if n-1 <= np.iinfo(np.uint32).max:
            dtype = np.uint32
        else:
            dtype = np.uint64
        voxids_flat = np.arange(n, dtype=dtype)
        voxids = voxids_flat.reshape((nx, ny, nz), order='C')

        self.vox = {'n': n, 'nx': nx, 'ny': ny, 'nz': nz,
                    'shape': (nx, ny, nz),
                    'ids': voxids,
                    'dtype': dtype
                    }

    def extract_pvgrid_subset(self, instance_name='twin.0',
                              feature_name='twin',
                              scalar='fid', fids=None):
        """
        Exanple
        -------
        instance_name = 'twin.0'
        feature_name = 'twin'
        fids = gstslice.fdb[instance_name]['data']['twin_id']
        pvgss = gstslice.extract_pvgrid_subset(instance_name=instance_name,
                                               feature_name=feature_name,
                                               scalar='fid',
                                               fids=fids)
        pvgss.plot()
        """
        self.validate_instance_name(instance_name)
        val, fids = self.validate_fids(instance_name=instance_name,
                                       feature_name=feature_name,
                                       fids=fids)
        # -------------------------------------------
        if not val:
            raise ValueError('fids could not be valided.')
        # -------------------------------------------
        if feature_name in ('base', 'lgi'):
            grid = self.pvgrid
        elif feature_name[:4] == 'twin':
            grid = self.fdb[instance_name]['data']['pvgrid']
        # -------------------------------------------
        # Extract pvgrid subset
        pvgss = grid.extract_cells(np.isin(grid[scalar], fids))
        return pvgss

    def plot_features(self, instance_name='twin.0', feature_name='twin',
                      scalar='fid', fids=None, show_scalars=False,
                      pv_kwargs={'cmap': 'nipy_spectral',
                                 'style': 'points',
                                 'show_edges': False,
                                 'line_width': 1.0,
                                 'opacity': 1.0},
                      plot=True,
                      validate=True
                      ):
        """
        Pre-example set up
        ------------------
        # Generate grain strucure
        # Instantiate twins. Then proceed below.

        instance_name = 'twin.0'
        feature_name = 'twin'
        fids = gstslice.fdb[instance_name]['data']['twin_id']
        pv_kwargs={'cmap': 'nipy_spectral',
                   'style': 'points',
                   'show_edges': False,
                   'line_width': 1.0,
                   'opacity': 0.75}
        Example-1
        ---------
        pvgss = gstslice.plot_features(instance_name=instance_name,
                                       feature_name=feature_name,
                                       scalar='fid', fids=fids,
                                       pv_kwargs=pv_kwargs)
        Example-2
        ---------
        # Turning plot flag False, renders this function identical to
        # the function extract_pvgrid_subset!! and returns the
        # subset pvgrid. See below.
        pvgss = gstslice.plot_features(instance_name=instance_name,
                                       feature_name=feature_name,
                                       scalar='fid', fids=fids,
                                       pv_kwargs=pv_kwargs,
                                       plot=True)
        Example-3
        ---------
        if unsure of what scalars exist, then use the same function to
        know them as below. No plpotting will be done, only available
        scalars will be provided.

        pvgss = gstslice.plot_features(instance_name=instance_name,
                                       feature_name=feature_name,
                                       show_scalars=True)
        """
        # fids = gstslice.gid
        # fids = gstslice.fdb[instance_name]['data']['twin_id']

        if show_scalars:
            if feature_name in ('base', 'lgi'):
                _an_ = self.pvgrid.array_names
            elif feature_name[:4] == 'twin':
                _an_ = self.fdb[instance_name]['data']['pvgrid'].array_names
            print(f"Available sclars: {_an_}")
            return None
        # -------------------------------------------
        pvgss = self.extract_pvgrid_subset(instance_name=instance_name,
                                           feature_name=feature_name,
                                           scalar=scalar, fids=fids)
        # -------------------------------------------
        if plot:
            pvgss.plot(scalars='fid', **pv_kwargs)
        # -------------------------------------------
        return pvgss

    def validate_instance_name(self, instance_name):
        validated = False
        if instance_name in ('base', 'lgi'):
            validated = True
        elif instance_name[:4] == 'twin':
            validated = True
        return validated

    def validate_fids(self, instance_name='twin.0',
                      feature_name='twin', fids=None):
        validated = False
        if type(fids) not in dth.dt.ITERABLES:
            if type(fids) not in dth.dt.NUMBERS:
                print("Invalid fid type.")
                return validated, fids
            else:
                fids = [int(fids)]
        else:
            fids_ = np.array([int(fid)
                              for fid in fids if type(fid) in dth.dt.NUMBERS])
            if len(fids_) != len(fids):
                print("Invalid typed fids have been removed.")

        if feature_name in ('base', 'lgi'):
            reffids = self.gid()
        if feature_name == 'twin':
            reffids = self.fdb[instance_name]['data']['twin_id']

        fids = np.array([fid for fid in fids_ if fid in reffids])
        if len(fids_) != len(fids):
            print("Invalid fid values have been removed.")

        if len(fids_) == 0:
            print("No fids to process.")
        else:
            validated = True
        return validated, fids

    @property
    def gridx(self):
        return np.arange(self.uigrid.xmin, self.uigrid.xmax, self.uigrid.xinc)

    @property
    def gridy(self):
        return np.arange(self.uigrid.ymin, self.uigrid.ymax, self.uigrid.yinc)

    @property
    def gridz(self):
        return np.arange(self.uigrid.zmin, self.uigrid.zmax, self.uigrid.zinc)

    @property
    def domsizex(self):
        return self.gridx.size

    @property
    def domsizey(self):
        return self.gridy.size

    @property
    def domsizez(self):
        return self.gridz.size

    @property
    def domvol(self):
        return self.domsizex*self.domsizey*self.domsizez

    @property
    def domsa(self):
        front = self.domainx*self.domainz
        right = self.domainy*self.domainz
        top = self.domainx*self.domainy
        return 2*(front+right+top)

    def get_gs_instance_pvgrid(self, instance_name='base'):
        """
        Example
        -------

        """
        pvgrid = pv.UniformGrid()
        pvgrid.origin = (self.uigrid.xmin, self.uigrid.ymin, self.uigrid.zmin)
        pvgrid.spacing = (self.uigrid.xinc, self.uigrid.yinc, self.uigrid.zinc)
        # -------------------------------------
        if instance_name == 'base':
            _data_ = self.lgi
        elif 'twin' in instance_name:
            _data_ = self.fdb[instance_name]['data']['fid']
        # -------------------------------------
        pvgrid.dimensions = np.array(_data_.shape) + 1
        pvgrid.cell_data[instance_name] = _data_.flatten(order="F")
        # -------------------------------------
        return pvgrid

    def plot_gs_instance(self,
                         check=False,
                         instance_name='base',
                         cmap='nipy_spectral',
                         show_edges=False,
                         lighting=True,
                         show_scalar_bar=True,
                         ):
        if not check:
            pvgrid = self.get_gs_instance_pvgrid(instance_name=instance_name)
            pvgrid.plot(cmap=cmap,
                        show_edges=show_edges,
                        lighting=lighting,
                        show_scalar_bar=show_scalar_bar)
        else:
            print(f'Available instance_names are: {self.fdb.keys()}')

    def mask_fid(self,
                 feature='twins',
                 instance_name='twin.0',
                 fid_mask_value=-32,
                 non_fid_mask=False,
                 non_fid_mask_value=-31,
                 write_to_disk=False,
                 write_sparse=True,
                 throw=True):
        """
        Mask the feature ID array with the given mask value. Options apply.

        This method masks the feature ID array for a given feature and instance
        with specified mask values.

        Parameters
        ----------
        feature: str, optional
            Specification of feature name. Must be from below valid list:
                * twins
                * blocks
                * precipitates
                * laths
                * sub-grains
            DEfaults to 'twins'.

        instance_name: str, optional
            Specification of instance name. Must be either 'base' or a valid
            gstslice.fdb.keys(). Defaults to 'twin.0'.

        fid_mask_value: int, optional
            Numerical value to mask the feature ID with. Defaults to -32.
            Negative values are recommended to avoid conflicts with valid IDs.
            If a non-negative value is provided, it will be converted to its
            negative equivalent.

        non_fid_mask: bool, optional
            If True, mask elements where the feature ID is *not* equal to the
            instance's twin ID with `non_fid_mask_value`. Defaults to False.

        non_fid_mask_value: int, optional
            Numerical value to use when `non_fid_mask` is True. Defaults to
            -31. Negative values are recommended. If a non-negative value is
            provided, it will be converted to its negative equivalent.

        write_to_disk: bool, optional
            If True, write the masked data to disk.  The specific format
            (sparse or dense) is determined by `write_sparse`. Defaults to
            False.

        write_sparse: bool, optional
            If True (and `write_to_disk` is True), write the data in a sparse
            format.  If False, write in a dense format. Defaults to True.

        throw : bool, optional
            If True, raise a `ValueError` if the `instance_name` is invalid.
            If False, return None in case of an error. Defaults to True.

        Raises
        ------
        ValueError
            If `instance_name` is invalid and `throw` is True.

        Returns
        -------
        numpy.ndarray or None
            The masked feature ID array. Returns None if `write_to_disk` is
            True or if an error occurs and `throw` is False.
        """

        '''Initial validations.'''
        if instance_name == 'base':
            return None
        if instance_name not in self.fdb.keys():
            raise ValueError('Inva;lid instance name.')

        '''Ensure that fid_mask_value is negative. '''
        if fid_mask_value >= 0:
            fid_mask_value = -fid_mask_value
        if non_fid_mask_value >= 0:
            non_fid_mask_value = -non_fid_mask_value

        '''Deepcopy the data for modification.'''
        _data_ = deepcopy(self.fdb[instance_name]['data']['fid'])

        '''Mask the values with user specified values'''
        for tid in self.fdb[instance_name]['data']['twin_id']:
            _data_[np.where(_data_ == tid)] = fid_mask_value

        '''If non_fid_mask is specified as True, values which does not belong
        to the feature ID will be masked to user specifed value of
        non_fid_mask_value. If non_fid_mask is specified as False, a maskig
        value of 0 is used.'''
        if non_fid_mask:
            _data_[np.where(_data_ != fid_mask_value)] = non_fid_mask_value
        else:
            _data_[np.where(_data_ != fid_mask_value)] = 0

        '''Wri5e data to sdisk if requetsed for.'''
        if write_to_disk:
            if write_sparse:
                # sparse array write
                '''scipy.sparse.save_npz'''
                pass
            else:
                # dense array write
                '''np.save'''
                pass

        '''Return the masked data if requested for.'''
        if throw:
            return _data_
        else:
            return None

    def mask_fid_and_make_pvgrid(self,
                                 feature='twins',
                                 instance_name='twin.0',
                                 fid_mask_value=-32,
                                 non_fid_mask=False,
                                 non_fid_mask_value=-31,
                                 write_to_disk=False,
                                 write_sparse=True,
                                 throw=True):
        _data_ = self.mask_fid(feature=feature,
                               instance_name=instance_name,
                               fid_mask_value=fid_mask_value,
                               non_fid_mask=non_fid_mask,
                               non_fid_mask_value=non_fid_mask_value,
                               write_to_disk=write_to_disk,
                               write_sparse=write_sparse,
                               throw=throw)
        pvgrid = pv.ImageData()
        pvgrid.dimensions = np.array(_data_.shape) + 1
        pvgrid.origin = (0, 0, 0)
        pvgrid.spacing = (1, 1, 1)
        pvgrid.cell_data[instance_name] = _data_.flatten(order="F")
        return pvgrid

    def mask_fid_and_plot(self,
                          feature='twins',
                          instance_names=('twin.0', ),
                          fid_mask_value=-32,
                          non_fid_mask=False,
                          non_fid_mask_value=-31,
                          write_to_disk=False,
                          write_sparse=True,
                          throw=True,
                          cmap_specs=(['blue', 'yellow', 'grey', 'red'], 2),
                          show_edges=False,
                          opacity=1.0, rmax_sp=5, cmax_sp=5,
                          thresholding=True,
                          threshold_value=-32):
        """
        Example
        -------
        import time
        start_time = time.time()
        from upxo.ggrowth.mcgs import mcgs

        pxt = mcgs(input_dashboard='input_dashboard.xls')
        pxt.simulate(verbose=False)
        tslice = 49
        gstslice = pxt.gs[tslice]

        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=True,
                                           find_spatial_bounds_of_grains=True,
                                           force_compute=True)
        gstslice.set_mprops(volnv=True, eqdia=False,
                            eqdia_base_size_spec='volnv',
                            arbbox=False, arbbox_fmt='gid_dict',
                            arellfit=False, arellfit_metric='max',
                            arellfit_calculate_efits=False,
                            arellfit_efit_routine=1,
                            arellfit_efit_regularize_data=False,
                            solidity=False, sol_nan_treatment='replace',
                            sol_inf_treatment='replace',
                            sol_nan_replacement=-1, sol_inf_replacement=-1)

        gstslice.clean_gs_GMD_by_source_erosion_v1(prop='volnv',
                                                   threshold=8,
                                                   parameter_metric='mean',
                                                   reset_pvgrid_every_iter=True,
                                                   find_neigh_every_iter=False,
                                                   find_grvox_every_iter=True,
                                                   find_grspabnds_every_iter=True)

        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=True,
                                           find_spatial_bounds_of_grains=True,
                                           force_compute=True)
        gstslice.set_mprops(volnv=True, eqdia=False,
                            eqdia_base_size_spec='volnv',
                            arbbox=False, arbbox_fmt='gid_dict',
                            arellfit=False, arellfit_metric='max',
                            arellfit_calculate_efits=False,
                            arellfit_efit_routine=1,
                            arellfit_efit_regularize_data=False,
                            solidity=False, sol_nan_treatment='replace',
                            sol_inf_treatment='replace',
                            sol_nan_replacement=-1, sol_inf_replacement=-1)

        mprops = {'volnv': {'use': True, 'reset': False,
                            'k': [.02, 1.0], 'min_vol': 4,},
                  'rat_sanv_volnv': {'use': True, 'reset': False,
                                     'k': [0.0, .8], 'sanv_N': 26},}
        twspec = {'n': [5, 10, 3],
                'tv': np.array([5, -3.5, 5]),
                'dlk': np.array([1.0, -1.0, 1.0]),
                'dnw': np.array([0.5, 0.5, 0.5]),
                'dno': np.array([0.5, 0.5, 0.5]),
                'tdis': 'normal',
                'tpar': {'loc': 1.12, 'scale': 0.1, 'val': 1},
                'vf': [0.05, 1.00], 'sep_bzcz': False}
        twgenspec = {'seedsel': 'random_gb', 'K': 20,
                   'bidir_tp': False, 'checks': [True, True],}

        gstslice.instantiate_twins(ninstances=4,
                                   base_gs_name_prefix='twin.',
                                   twin_setup={'nprops': 2, 'mprops': mprops},
                                   twspec=twspec,
                                   twgenspec=twgenspec,
                                   reset_fdb=True, )
        # ----------------------------------------
        elapsed_time_simulation = time.time() - start_time
        print(f'Total time taken: {elapsed_time_simulation}')
        # ----------------------------------------
        gstslice.mask_fid_and_plot(feature='twins',
                                   instance_names=gstslice.fdb.keys(),
                                   fid_mask_value=-32,
                                   non_fid_mask=True,
                                   non_fid_mask_value=-31,
                                   write_to_disk=False,
                                   write_sparse=True,
                                   throw=True,
                                   cmap_specs=(['white', 'yellow', 'grey', 'red'], 2),
                                   show_edges=False,
                                   opacity=1.0, rmax_sp=8, cmax_sp=13,
                                   thresholding=True,
                                   threshold_value=-32)

        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
        import seaborn as sns

        total_twin_vol_fr = []
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=88)
        for key in gstslice.fdb.keys():
            grain_tvf = gstslice.fdb[key]['data']['twin_vf']

            total_tvf = gstslice.fdb[key]['data']['twin_vf_total']
            total_twin_vol_fr.append(total_tvf)

            ntwins = np.array(list(gstslice.fdb[key]['data']['twin_map_g_nt'].values()))
            ntwins = ntwins[ntwins != 0]

            sns.kdeplot(grain_tvf, ax=axes[0], common_norm=True)
            sns.kdeplot(ntwins, ax=axes[1], common_norm=True)
        axes[0].set_xlabel('Host grain wise twin Volume fraction', fontsize=14)
        axes[1].set_xlabel('Host grain wise number of twins', fontsize=14)
        axes[0].set_ylabel('Probability density function', fontsize=14)
        axes[1].set_ylabel('Probability density function', fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 5), dpi=75)
        sns.kdeplot(total_twin_vol_fr, common_norm=True)
        plt.xlabel('Total twin volume fractions in\n grain structures', fontsize=14)
        plt.ylabel('Probability density function', fontsize=14)
        plt.tight_layout()
        plt.show()


            x_grain_tvf = np.linspace(grain_tvf.min(), grain_tvf.max(), 100)
            axes.plot(x_grain_tvf, kde_grain_tvf(x_grain_tvf))
            axes.set_xlabel('Grain TVF')
            axes.set_ylabel('Density')
            x_ntwins = np.linspace(ntwins.min(), ntwins.max(), 100)
            axes.plot(x_ntwins, kde_ntwins(x_ntwins))
            axes.set_xlabel('Number of Twins')
            axes.set_ylabel('Density')
            plt.tight_layout()

            # plot kde of grain_tvf in first subplot
            # plot kde of ntwins in second subplot

        # plot kde of total_twin_vol_fr in seperate plot window


        """
        if not type(cmap_specs) in dth.dt.ITERABLES:
            raise TypeError('cmap_specs must be a tuple or list.')
        # -------------------------------------------
        if isinstance(cmap_specs[0], str):
            cmap = plt.get_cmap(cmap_specs[0], cmap_specs[1])
        else:
            cmap = cmap_specs[0]
        # -------------------------------------------
        pvgrids, ninstances = [], len(instance_names)
        _def_ = self.mask_fid_and_make_pvgrid
        for instance_count, instance_name in enumerate(instance_names):
            print(f'Creating pvgrid for instance {instance_name}: {instance_count} of {ninstances}')
            pvgrids.append(_def_(feature=feature,
                                 instance_name=instance_name,
                                 fid_mask_value=fid_mask_value,
                                 non_fid_mask=non_fid_mask,
                                 non_fid_mask_value=non_fid_mask_value,
                                 write_to_disk=write_to_disk,
                                 write_sparse=write_sparse,
                                 throw=throw)
                           )
        # ---------------------------------------------
        print(40*'-', len(pvgrids), 40*'-')
        # print(ninstances, rmax_sp, cmax_sp)
        nr, nc = arrange_subplots(ninstances, rmax_sp, cmax_sp)
        # print(nr, nc)
        # print(40*'=')
        # ---------------------------------------------
        if not thresholding:
            if nr*nc > 1:
                i, pvp = 0, pv.Plotter(shape=(nr, nc))
                for r in range(nr):
                    for c in range(nc):
                        if i < len(pvgrids):
                            print(f'rendering {i} of {ninstances} instances')
                            pvp.subplot(r, c)
                            pvp.add_mesh(pvgrids[i], cmap=cmap,
                                         show_edges=show_edges, opacity=opacity)
                        i += 1
                pvp.show()
            elif nr*nc == 1:
                pvp = pv.Plotter()
                pvp.add_mesh(pvgrids[0], cmap=cmap,
                             show_edges=show_edges,
                             opacity=opacity)
                pvp.show()
        else:
            if nr*nc > 1:
                i, pvp = 0, pv.Plotter(shape=(nr, nc))
                for r in range(nr):
                    for c in range(nc):
                        if i < len(pvgrids):
                            print(f'rendering {i} of {ninstances} instances')
                            pvp.subplot(r, c)
                            pvp.add_mesh(pvgrids[i].threshold([threshold_value,
                                                               threshold_value]),
                                         cmap=cmap, show_edges=show_edges,
                                         opacity=opacity)
                        i += 1
                pvp.show()
            elif nr*nc == 1:
                pvp.subplot(r, c)
                pvp.add_mesh(pvgrids[0].threshold([threshold_value,
                                                   threshold_value]),
                             cmap=cmap, show_edges=show_edges,
                             opacity=opacity)
                pvp.show()

    def get_np_from_dict(self, udata):
        return np.array(list(udata.values()))

    def extract_subdomains_random(self, p=5, q=5, r=5, n=2,
                                  feature_name='base',
                                  user_fid=None,
                                  make_pvgrids=False,):
        """Extracts n random sub-domains of size pxqxr from a 3D array.

        Parameters
        ----------
        p: int
            The size of the sub-domain along the first axis.

        q: int
            The size of the sub-domain along the second axis.

        r: int
            The size of the sub-domain along the third axis.

        n: int
            The number of random sub-domains to extract.

        feature_name: str
            Name of the feature. It can take the following options.
                * 's' or 'state' for Monte-Carlo state value.
                * 'base' or 'base_gs'. Here, gstslice.lgi will become the
                parent 3D np.array from which sub-domains will be extracted.
                * Any value (i.e. feature_name) in the feature data base of the
                current temporal slice. This is available in
                gstslice.fdb.keys(). Here, gstslice.feature_name will become
                the parent 3D np.array from which sub-domains will be
                extracted.
                * 'user'. If the user wishes to extract sub-domains from a
                3D np.array of their choice, then this allows the user to do
                so. This would need supplying of thr user_fid value.

        user_fid: numpy.ndarray
            User supplied 3D NumPy array.

        make_pvgrids: bool
            If True, PyVista grids will be made for each subdomain and
            returned.

        Returns
        -------
        SD: dict
            A dictionary containing the following key: value pairs.
                * 'data': list of NumPy arrays, each representing a randomly
                extracted p x q x r sub-domaimn.
                * 'pvgrids': list of Py-Vista grid objects if make_pvgrids is
                True.

        Examples
        --------
        from upxo.ggrowth.mcgs import mcgs

        pxt = mcgs(input_dashboard='input_dashboard.xls')
        pxt.simulate(verbose=False)
        tslice = 49
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=True,
                                           find_spatial_bounds_of_grains=True,
                                           force_compute=True)

        A = gstslice.extract_subdomains_random(p=5, q=5, r=5, n=2,
                                               feature_name='base',
                                               )
        """
        if feature_name in ('s', 'state'):
            _base_data_ = self.s
        if feature_name in ('base', 'base_gs'):
            _base_data_ = self.lgi
        if 'twin.' in feature_name:
            _base_data_ = self.fdb[feature_name]['data']['fid']
        # ---------------------------------------------
        print(40*'-', f'\nExtracting {n} subdomains at random.')
        P, Q, R = (PQR-pqr+1 for pqr, PQR in zip((p, q, r), _base_data_.shape))
        # ---------------------------------------------
        subdomains = []
        sdnames = []
        pvgrids = []

        for _ in range(n):
            x, y, z = np.random.randint(0, [P, Q, R])
            subdomain = _base_data_[x:x+p, y:y+q, z:z+r]
            subdomains.append(subdomain)

            if feature_name in ('s', 'state'):
                sdnames.append('s.sd')
            if feature_name in ('base', 'base_gs'):
                sdnames.append('lgi.sd')
            if 'twin.' in feature_name:
                sdnames.append(f'{feature_name}.sd')

        SD = {'sd': subdomains}

        if make_pvgrids:
            for sd_count in range(n):
                pvgrid = self.make_pvgrid_v1(feature_name='user',
                                             instance_name='none',
                                             user_fid=subdomains[sd_count],
                                             scalar_name=sdnames[sd_count],
                                             pvgrid_origin=(0, 0, 0),
                                             pvgrid_spacing=(1, 1, 1),
                                             perform_checks=False)
                pvgrids.append(pvgrid)

            SD['pvgrids'] = pvgrids

        return SD


    def make_pvgrid_v1(self, feature_name='base', instance_name='lgi',
                       user_fid=None, scalar_name='lgi', pvgrid_origin=(0,0,0),
                       pvgrid_spacing=(1,1,1), perform_checks=True):
        if perform_checks:
            if feature_name == 'base':
                if instance_name == 'lgi':
                    fid = deepcopy(self.lgi)
            elif dth.strip_str(feature_name) in ('twin', 'twins', 'tw',
                                                 'twinned', 'twinnedgs'):
                if instance_name in self.fdb.keys():
                    fid = deepcopy(self.fdb[instance_name]['data']['fid'])
                else:
                    raise ValueError("Invalid instance_name. Does'nt exist.")
            elif feature_name == 'user':
                fid = user_fid
            else:
                raise ValueError("Invalid feature_name.")
        else:
            fid = user_fid

        pvgrid = pv.UniformGrid()
        pvgrid.dimensions = np.array(fid.shape) + 1
        pvgrid.origin = pvgrid_origin
        pvgrid.spacing = pvgrid_spacing
        pvgrid.cell_data[str(scalar_name)] = fid.flatten(order="F")

        return pvgrid

    def smoothen_sds(self, k=1, feature_name='base', instance_name='lgi',
                     user_fid=None,  down_order=0, down_mode='nearest',
                     up_order=0, up_mode='nearest',
                     make_pvgrid=False, pvgrid_scalar_name='lgi',
                     pvgrid_origin=(0, 0, 0), pvgrid_spacing=(1, 1, 1),
                     ):
        """
        Smooth a fid array by scaling and descaling.

        Parameters
        ----------
        k: float
            Scaling factor >= 1. Value of 1 will return the unmodified data. A
            value near to 1 has less effect while a value close to 0 would have
            greater effect.

        feature_name: str
            Name of the feature. Valids include 'base', ('twin', 'twins', 'tw',
            'twinned', 'twinnedgs'), 'user', ('paps', 'austenitic_packets'). It
            does not matter how a string is entered as in 'twinnedgs' or
            'twinned_gs' or 'twinned.gs'. If 'user', then user_fid will be
            used instead of internally available fid datasets. Note: fid stands
            for feature id and is/must a 3D Numpy array.

        instance_name: str
            Allowed values for base grain structure, i.e. when feature_name is
            set to 'base' are 'lgi' (only as of present version.)

        user_fid: np.ndarray
            User input value of 3D image to be used. This will only be used
            when feature_name is 'user'.

        make_pvgrid: bool
            If True, a pyvista uniform grid will be returned as pvgrid, else
            None will be returned as pvgrid.

        Returns
        -------
        fid_mod: np.ndarray
            Modified fid.

        pvgrid: pv.UniformGrid() / None
            If user inputs make_pvgrid as True, then returns a pyvista
            uniform grid object, else returns None.

        Raises
        ------
        ValueError
            If k > 1. String: k must belong to (0, 1].

        ValueError
            If feature_name is not 'base' or nort in ('twin', 'twins', 'tw',
            'twinned', 'twinnedgs') or 'user'. String: Invalid feature_name.

        ValueError
            If feature_name is in ('twin', 'twins', 'tw', 'twinned',
            'twinnedgs') and instance_name is not in self.fdb.keys().
            String: Invalid instance_name. Does'nt exist.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs

        pxt = mcgs(input_dashboard='input_dashboard.xls')
        pxt.simulate(verbose=False)
        tslice = 49
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=True,
                                           find_spatial_bounds_of_grains=True,
                                           force_compute=True)

        gstslice.smoothen_sds(k=1, feature_name='base', instance_name='lgi',
                              user_fid=None,  down_order=0, down_mode='nearest',
                              up_order=0, up_mode='nearest',
                              make_pvgrid=False, pvgrid_scalar_name='lgi',
                              pvgrid_origin=(0, 0, 0),
                              pvgrid_spacing=(1, 1, 1),)
        """
        if k <=0 or k > 1 :
            raise ValueError("k must belong to (0, 1].")
        if k >= 1:
            if feature_name == 'base':
                if instance_name == 'lgi':
                    fid = deepcopy(self.lgi)
            elif dth.strip_str(feature_name) in ('twin', 'twins', 'tw',
                                                 'twinned', 'twinnedgs'):
                if instance_name in self.fdb.keys():
                    fid = deepcopy(self.fdb[instance_name]['data']['fid'])
                else:
                    raise ValueError("Invalid instance_name. Does'nt exist.")
            elif feature_name == 'user':
                fid = user_fid
            else:
                raise ValueError("Invalid feature_name.")
        '''downscaling.'''
        resampled_fid = zoom(fid, zoom=(k, k, k),
                             order=down_order, mode=down_mode)
        '''upscaling.'''
        resampled_fid = zoom(resampled_fid, zoom=(1/k, 1/k, 1/k),
                             order=up_order, mode=up_mode)
        if make_pvgrid:
            pvgrid = self.make_pvgrid_v1(feature_name=feature_name,
                                         instance_name=instance_name,
                                         user_fid=resampled_fid,
                                         scalar_name=pvgrid_scalar_name,
                                         pvgrid_origin=pvgrid_origin,
                                         pvgrid_spacing=pvgrid_spacing,
                                         perform_checks=False)
        else:
            pvgrid = None
        return resampled_fid, pvgrid

    def deform_ortho(self, kx, ky, kz,
                     feature_name='base', instance_name='lgi', user_fid=None):
        """
        kx: float
            Scaling factor along x-axis.

        ky: float
            Scaling factor along y-axis.

        kz: float
            Scaling factor along z-axis.

        feature_name: str
            Name of the feature. Valids include 'base', ('twin', 'twins', 'tw',
            'twinned', 'twinnedgs'), 'user', ('paps', 'austenitic_packets'). It
            does not matter how a string is entered as in 'twinnedgs' or
            'twinned_gs' or 'twinned.gs'. If 'user', then user_fid will be
            used instead of internally available fid datasets. Note: fid stands
            for feature id and is/must a 3D Numpy array.

        instance_name: str
            Allowed values for base grain structure, i.e. when feature_name is
            set to 'base' are 'lgi' (only as of present version.)

        user_fid: np.ndarray
            User input value of 3D image to be used. This will only be used
            when feature_name is 'user'.
        """
        pass

    @staticmethod
    def cubic_Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[ c,-s, 0],
                         [ s, c, 0],
                         [ 0, 0, 1]])

    @staticmethod
    def cubic_Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0],
                         [0, c,-s],
                         [0, s, c]])

    @classmethod
    def cubic_euler_bunge_to_matrix(cls, phi1, Phi, phi2, degrees=True):
        """Bunge (ZXZ): R = Rz(phi1) * Rx(Phi) * Rz(phi2)."""
        if degrees:
            phi1, Phi, phi2 = np.deg2rad([phi1, Phi, phi2])
        return cls.cubic_Rz(phi1) @ cls.cubic_Rx(Phi) @ cls.cubic_Rz(phi2)

    @classmethod
    def cubic_euler_bunge_to_matrix_v1(cls, phi1, Phi, phi2,
                                        degrees=True, dtype=np.float32,
                                        validate_numpy_input=False):
        """
        Computes a batch of Bunge (ZXZ) rotation matrices from arrays of
        Euler angles. R = Rz(phi1) * Rx(Phi) * Rz(phi2).

        Parameters:
        - phi1, Phi, phi2: Arrays of Euler angles.
        - degrees (bool): If True, angles are in degrees; otherwise, radians.
        - dtype (np.dtype): The data type for the output rotation matrices.
                            Defaults to np.float32 for better performance and
                            memory usage.

        EXAMPLE - 1
        -----------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='demo_3d_04.xls')
        pxt.simulate(verbose=False)
        gstslice = pxt.gs[99]

        euler_angles = np.array([[10, 20, 30], [45, 60, 75]])
        phi1 = euler_angles[:, 0]
        Phi = euler_angles[:, 1]
        phi2 = euler_angles[:, 2]
        R = gstslice.cubic_euler_bunge_to_matrix_v1(phi1, Phi, phi2)
        print(R)
        """
        if validate_numpy_input:
            phi1 = np.asarray(phi1, dtype=dtype)
            Phi = np.asarray(Phi, dtype=dtype)
            phi2 = np.asarray(phi2, dtype=dtype)

        if degrees:
            phi1 = np.deg2rad(phi1)
            Phi = np.deg2rad(Phi)
            phi2 = np.deg2rad(phi2)

        c1, s1 = np.cos(phi1), np.sin(phi1)
        c2, s2 = np.cos(Phi), np.sin(Phi)
        c3, s3 = np.cos(phi2), np.sin(phi2)

        R = np.empty((len(phi1), 3, 3), dtype=dtype)

        R[:, 0, 0] = c1*c3-s1*s3*c2
        R[:, 0, 1] = -c1*s3-s1*c3*c2
        R[:, 0, 2] = s1*s2

        R[:, 1, 0] = s1*c3+c1*s3*c2
        R[:, 1, 1] = -s1*s3+c1*c3*c2
        R[:, 1, 2] = -c1*s2

        R[:, 2, 0] = s3*s2
        R[:, 2, 1] = c3*s2
        R[:, 2, 2] = c2

        return R

    @staticmethod
    def cubic_rotation_angle(R):
        """Return angle (rad) of a proper rotation matrix R."""
        x = (np.trace(R)-1.0)/2.0
        x = np.clip(x, -1.0, 1.0)
        return np.arccos(x)

    @staticmethod
    def cubic_rotation_angle_1(R):
        """
        Return angle (rad) of a proper rotation matrix or a batch of matrices R.

        Parameters:
        - R: A 3x3 rotation matrix or a (N, 3, 3) NumPy array of rotation matrices.

        Returns:
        - angle (float or array): Rotation angle(s) in radians.

        EXAMPLE
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='demo_3d_04.xls')
        pxt.simulate(verbose=False)
        gstslice = pxt.gs[99]

        euler_angles = np.array([[10, 20, 30], [45, 60, 75]])
        phi1 = euler_angles[:, 0]
        Phi = euler_angles[:, 1]
        phi2 = euler_angles[:, 2]
        R = gstslice.cubic_euler_bunge_to_matrix_v1(phi1, Phi, phi2)

        cubic_rotation_angle_vec(R)
        """
        # np.trace can operate along the last two axes for a stack of matrices
        x = (np.trace(R, axis1=-2, axis2=-1)-1.0)/2.0
        x = np.clip(x, -1.0, 1.0)
        return np.arccos(x)

    @staticmethod
    def cubic_rotation_axis(R, angle):
        """Return unit axis for rotation R given angle (rad).
           For very small angles, returns a default axis."""
        if angle < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        A = (R-R.T)/(2.0*np.sin(angle))
        # axis components are (A32, A13, A21)
        axis = np.array([A[2, 1], A[0, 2], A[1, 0]])
        n = np.linalg.norm(axis)
        return axis/(n if n > 0 else 1.0)

    @staticmethod
    def cubic_rotation_axis_v1(rstack, angles):
        """
        Returns unit axis for a batch of rotations R given a batch of
        angles (rad). For very small angles, returns a default axis.

        Parameters:
        - rstack: A (N, 3, 3) NumPy array of rotation matrices.
        - angles: A (N,) NumPy array of rotation angles in radians.

        Returns:
        - A (N, 3) NumPy array of unit axes.

        EXAMPLE
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='demo_3d_04.xls')
        pxt.simulate(verbose=False)
        gstslice = pxt.gs[99]

        N = int(2E6)
        phi1 = np.random.randint(0, 90, N)
        Phi = np.random.randint(0, 180, N)
        phi2 = np.random.randint(0, 360, N)
        rstack = gstslice.cubic_euler_bunge_to_matrix_v1(phi1, Phi, phi2)
        angles = np.full((len(rstack),), 4, dtype=rstack.dtype)
        rotax = gstslice.cubic_rotation_axis_v1(rstack, angles)
        """
        '''Create a boolean mask for the small angles'''
        small_angle_mask = np.abs(angles) < 1e-8
        '''Pre-allocate the output array'''
        num_rotations = rstack.shape[0]
        axes = np.empty((num_rotations, 3), dtype=rstack.dtype)
        '''Handle the small-angle case first by setting the default axis'''
        axes[small_angle_mask] = np.array([1.0, 0.0, 0.0])
        '''Work on the remaining matrices (where angle is not small)'''
        large_angle_indices = ~small_angle_mask
        R_large_angles = rstack[large_angle_indices]
        angle_large_angles = angles[large_angle_indices]
        '''Calculate A = (R-R.T)/(2.0*sin(angle)) for the large angles
        The transpose needs to operate on the last two axes'''
        denom = 2.0*np.sin(angle_large_angles)
        '''Add an extra dimension to denom for correct broadcasting'''
        A = (R_large_angles-np.transpose(R_large_angles, axes=(0, 2, 1)))
        A = A/denom[:, np.newaxis, np.newaxis]
        '''Extract the axis components using vectorized slicing'''
        large_angle_axes = np.stack([A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]],
                                    axis=1)
        '''Normalize the large-angle axes'''
        norms = np.linalg.norm(large_angle_axes, axis=1)
        '''normalized_axes: na'''
        na = large_angle_axes/np.where(norms > 0, norms, 1.0)[:, np.newaxis]
        '''Place the normalized axes back into the output array'''
        axes[large_angle_indices] = na

        return axes

    @staticmethod
    def cubic_symmetry_operators():
        """
        24 proper rotations for m-3m as signed permutation matrices with det=+1.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='demo_3d_04.xls')
        pxt.simulate(verbose=False)
        gstslice = pxt.gs[99]
        gstslice.cubic_symmetry_operators()
        """
        ops = []
        for p in permutations(range(3)):  # 6 permutations
            P = np.eye(3)[list(p)]
            for signs in product([-1,1], repeat=3):  # 8 sign patterns
                S = P * np.array(signs)[None, :]
                if round(np.linalg.det(S)) == 1:
                    ops.append(S.astype(float))
        # de-duplicate
        uniq = []
        for S in ops:
            if not any(np.allclose(S, T) for T in uniq):
                uniq.append(S)
        return uniq

    def fcc_symmetrise_ori(self, bea, dtype=np.float32):
        """
        Generate symmetric equivalents of an orientation.

        Example
        -------
        bea = texture['ori_means']['brass']
        gstslice.fcc_symmetrise_ori(bea, dtype=np.float32)
        """
        g = self.cubic_euler_bunge_to_matrix(*bea, degrees=True)
        sym_ops = self.cubic_symmetry_operators()
        eq_mats = [self._proj_to_so3(S @ g) for S in sym_ops]
        eq_mats = self._unique_rotations(eq_mats, tol=1E-8)
        symm_eq = np.array([list(self._matrix_to_euler_bunge(R, degrees=True))
                            for R in eq_mats], dtype=dtype)
        return symm_eq

    @classmethod
    def _as_rotmat(cls, x, degrees=True):
        """
        Accept Euler triplet (len==3) OR 3x3 rotation matrix.
        Return a 3x3 rotation matrix (float).
        """
        arr = np.asarray(x)
        if arr.shape == (3, 3):
            return arr.astype(float)
        if arr.ndim == 1 and arr.size == 3:
            phi1, Phi, phi2 = map(float, arr.ravel())
            return cls.cubic_euler_bunge_to_matrix(phi1, Phi, phi2, degrees=degrees)
        raise ValueError(f"_as_rotmat: expected Euler (3,) or R (3,3); got shape {arr.shape}")

    @classmethod
    def _get_cubic_ops_np(cls):
        """Cached cubic symmetry operators as a (24,3,3) float array."""
        ops = getattr(cls, "CUBIC_OPS", None)
        if ops is None:
            # Build once from your existing generator
            ops_list = cls.cubic_symmetry_operators()
            ops = np.asarray(ops_list, dtype=float)  # (24,3,3)
            cls.CUBIC_OPS = ops
        elif not isinstance(ops, np.ndarray):
            ops = np.asarray(ops, dtype=float)
            cls.CUBIC_OPS = ops
        return cls.CUBIC_OPS

    @classmethod
    def cubic_misorientation(cls, EA1, EA2, unique_tol_deg=1e-4, degrees=True):
        """
        Vectorized fast misorientation (cubic, m-3m).
        Inputs: Euler triplet (phi1,Phi,phi2) OR 3x3 rotation matrix.
        Returns:
          angle_deg_min : float
          axis_min : (3,) unit vector (sample frame)
          top3_angles_deg : list of up to 3 smallest UNIQUE angles (deg), ascending

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='demo_3d_04.xls')
        pxt.simulate(verbose=False)
        gstslice = pxt.gs[99]

        N = int(10)

        phi1 = np.random.randint(0, 90, N)
        Phi = np.random.randint(0, 180, N)
        phi2 = np.random.randint(0, 360, N)
        EA1 = np.vstack((phi1, Phi, phi2)).T

        phi1 = np.random.randint(0, 90, N)
        Phi = np.random.randint(0, 180, N)
        phi2 = np.random.randint(0, 360, N)
        EA2 = np.vstack((phi1, Phi, phi2)).T

        gA = gstslice.cubic_euler_bunge_to_matrix_v1(phi1, Phi, phi2)
        """
        gA = gstslice.cubic_euler_bunge_to_matrix_v1(EA1[:, 0], EA1[:, 1],
                                                     EA1[:, 2], degrees=True,
                                                     dtype=np.float32,
                                                     validate_numpy_input=False)

        gB = gstslice.cubic_euler_bunge_to_matrix_v1(EA2[:, 0], EA2[:, 1],
                                                     EA2[:, 2], degrees=True,
                                                     dtype=np.float32,
                                                     validate_numpy_input=False)

        # #####################################################################
        # Normalize inputs to rotation matrices
        gA = cls._as_rotmat(EA1, degrees=degrees)  # (3,3)
        gB = cls._as_rotmat(EA2, degrees=degrees)  # (3,3)

        # Symmetry ops as (24,3,3)
        S = cls._get_cubic_ops_np()                # (24,3,3)

        # All symmetry-equivalent orientations
        # RA[k] = S[k] @ gA         -> (24,3,3)
        # RB[l] = S[l] @ gB         -> (24,3,3)
        RA = S @ gA
        RB = S @ gB

        # All relative rotations: dR[l,k] = RB[l] @ RA[k].T   -> (24,24,3,3)
        RtA = np.transpose(RA, (0,2,1))            # (24,3,3)
        dR = RB[:, None, :, :] @ RtA[None, :, :, :]  # (24,24,3,3)

        # Angles from trace: angle = arccos( (tr(dR)-1)/2 )
        tr = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]     # (24,24)
        x = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        ang = np.arccos(x)                                     # radians, (24,24)

        # Minimum angle and its indices
        idx_flat = np.argmin(ang)
        l_min, k_min = divmod(int(idx_flat), ang.shape[1])
        best_angle = float(ang[l_min, k_min])                  # radians

        # Axis for the min dR
        dR_min = dR[l_min, k_min]                              # (3,3)
        if best_angle < 1e-12:
            axis_min = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            sin_a = np.sin(best_angle)
            # A = (R - R^T) / (2 sin a)
            A = (dR_min - dR_min.T) / (2.0 * sin_a)
            axis_min = np.array([A[2,1], A[0,2], A[1,0]], dtype=float)
            n = np.linalg.norm(axis_min)
            if n > 0:
                axis_min /= n
            else:
                axis_min = np.array([1.0, 0.0, 0.0], dtype=float)

        # Top-3 UNIQUE angles (deg), ascending
        angles_deg = np.degrees(np.sort(ang, axis=None))       # (576,)
        uniq = []
        for a in angles_deg:
            if not uniq or abs(a - uniq[-1]) > unique_tol_deg:
                uniq.append(a)
            if len(uniq) >= 3:
                break

        return float(np.degrees(best_angle)), axis_min, uniq

    @classmethod
    def cubic_misorientation_old1(cls, EA1, EA2, unique_tol_deg=1e-4, degrees=True):
        """
        Vectorized fast misorientation (cubic, m-3m).
        Inputs: Euler triplet (phi1,Phi,phi2) OR 3x3 rotation matrix.
        Returns:
          angle_deg_min : float
          axis_min      : (3,) unit vector (sample frame)
          top3_angles_deg : list of up to 3 smallest UNIQUE angles (deg), ascending
        """
        # Normalize inputs to rotation matrices
        gA = cls._as_rotmat(EA1, degrees=degrees)  # (3,3)
        gB = cls._as_rotmat(EA2, degrees=degrees)  # (3,3)

        # Symmetry ops as (24,3,3)
        S = cls._get_cubic_ops_np()                # (24,3,3)

        # All symmetry-equivalent orientations
        # RA[k] = S[k] @ gA         -> (24,3,3)
        # RB[l] = S[l] @ gB         -> (24,3,3)
        RA = S @ gA
        RB = S @ gB

        # All relative rotations: dR[l,k] = RB[l] @ RA[k].T   -> (24,24,3,3)
        RtA = np.transpose(RA, (0,2,1))            # (24,3,3)
        dR = RB[:, None, :, :] @ RtA[None, :, :, :]  # (24,24,3,3)

        # Angles from trace: angle = arccos( (tr(dR)-1)/2 )
        tr = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]     # (24,24)
        x = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        ang = np.arccos(x)                                     # radians, (24,24)

        # Minimum angle and its indices
        idx_flat = np.argmin(ang)
        l_min, k_min = divmod(int(idx_flat), ang.shape[1])
        best_angle = float(ang[l_min, k_min])                  # radians

        # Axis for the min dR
        dR_min = dR[l_min, k_min]                              # (3,3)
        if best_angle < 1e-12:
            axis_min = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            sin_a = np.sin(best_angle)
            # A = (R - R^T) / (2 sin a)
            A = (dR_min - dR_min.T) / (2.0 * sin_a)
            axis_min = np.array([A[2,1], A[0,2], A[1,0]], dtype=float)
            n = np.linalg.norm(axis_min)
            if n > 0:
                axis_min /= n
            else:
                axis_min = np.array([1.0, 0.0, 0.0], dtype=float)

        # Top-3 UNIQUE angles (deg), ascending
        angles_deg = np.degrees(np.sort(ang, axis=None))       # (576,)
        uniq = []
        for a in angles_deg:
            if not uniq or abs(a - uniq[-1]) > unique_tol_deg:
                uniq.append(a)
            if len(uniq) >= 3:
                break

        return float(np.degrees(best_angle)), axis_min, uniq

    @classmethod
    def cubic_misorientation_old(cls, EA1, EA2, unique_tol_deg=1e-4, degrees=True):
        """
        gA, gB: 3x3 rotation matrices (Bunge convention).
        Returns:
          - angle_deg_min: smallest misorientation angle (deg)
          - axis_min: unit axis (in sample frame) for the smallest angle
          - top3_angles_deg: list of the three smallest UNIQUE angles (deg), sorted
        """
        # ----------------------------------------------------------
        '''gA = cls.cubic_euler_bunge_to_matrix(EA1[0], EA1[1], EA1[2],
                                             degrees=degrees)
        gB = cls.cubic_euler_bunge_to_matrix(EA2[0], EA2[1], EA2[2],
                                             degrees=degrees)'''
        gA = cls._as_rotmat(EA1, degrees=degrees)
        gB = cls._as_rotmat(EA2, degrees=degrees)
        # ----------------------------------------------------------
        sym_ops = getattr(cls, "CUBIC_OPS", None)
        if sym_ops is None:
            sym_ops = cls.cubic_symmetry_operators()  # fallback
            cls.CUBIC_SYMM_OPS = sym_ops  # cache
        '''sym_ops = cls.cubic_symmetry_operators()'''
        # ----------------------------------------------------------
        angles = []
        best_angle = np.inf
        best_axis = np.array([1.0, 0.0, 0.0], dtype=float)

        for SA in sym_ops:
            RA = SA @ gA
            RtA = RA.T
            for SB in sym_ops:
                RB = SB @ gB
                dR = RB @ RtA
                ang = cls.cubic_rotation_angle(dR)
                angles.append(ang)
                if ang < best_angle:
                    best_angle = ang
                    best_axis = cls.cubic_rotation_axis(dR, ang)

        # unique + sort angles (in degrees)
        angles_deg = np.rad2deg(np.array(angles))
        angles_deg = np.sort(angles_deg)

        # unique by tolerance
        uniq = []
        for a in angles_deg:
            if not uniq or abs(a - uniq[-1]) > unique_tol_deg:
                uniq.append(a)

        miso_angles_deg = list(uniq[:3])
        return float(np.rad2deg(best_angle)), best_axis, miso_angles_deg

    def assign_twins(self, instance_name='twin.0',
                     vfs3=0.75, vfs5=0.25, by='volume', seed=12345):
        """
        Randomly assign twins to Σ3 / Σ5 so the chosen fraction matches vfs3 / vfs5
        and ensure *all* twins are assigned (no leftovers).

        Creates:
          self.fdb[instance_name]['data']['twin_id_s3'] : list[int]
          self.fdb[instance_name]['data']['twin_id_s5'] : list[int]
        """
        print(40*'-', '\nASSIGNUING S3 AND S5 twins\n')
        data = self.fdb[instance_name]['data']
        twin_ids = list(map(int, data['twin_id']))  # all twin feature IDs
        twin_nvox = data['twin_nvox']  # dict: twin_id -> voxel count
        # -----------------------------------------------
        print("....performing initial volume fraction operations")
        vfs3 = float(vfs3); vfs5 = float(vfs5)
        vfs3 = max(0.0, min(1.0, vfs3))
        vfs5 = max(0.0, min(1.0, vfs5))
        if vfs3+vfs5 > 1.0:
            vfs5 = 1.0-vfs3
        # -----------------------------------------------
        rng = Random(seed)
        twins_shuffled = twin_ids[:]
        rng.shuffle(twins_shuffled)

        s3_ids, s5_ids = [], []

        if by == 'count':
            n_total = len(twins_shuffled)
            n3_target = int(round(vfs3*n_total))
            n5_target = int(round(vfs5*n_total))
            # primary selection
            s3_ids = twins_shuffled[:min(n3_target, n_total)]
            s5_ids = twins_shuffled[len(s3_ids):min(len(s3_ids)+n5_target,
                                                    n_total)]
            # remainder assignment to ensure full coverage
            assigned = set(s3_ids) | set(s5_ids)
            remainder = [tid for tid in twins_shuffled if tid not in assigned]
            for tid in remainder:
                # deficits relative to targets
                def3 = n3_target-len(s3_ids)
                def5 = n5_target-len(s5_ids)
                if def3 > def5:
                    s3_ids.append(tid)
                elif def5 > def3:
                    s5_ids.append(tid)
                else:
                    # both met or equal deficit: balance by current counts
                    (s3_ids if len(s3_ids) <= len(s5_ids) else s5_ids).append(tid)

        elif by == 'volume':
            total_vol = float(sum(twin_nvox[int(tid)] for tid in twin_ids))
            target3 = vfs3*total_vol
            target5 = vfs5*total_vol

            acc3 = 0.0
            for tid in twins_shuffled:
                if acc3 >= target3:
                    break
                s3_ids.append(tid)
                acc3 += float(twin_nvox[int(tid)])

            assigned = set(s3_ids)
            remaining = [tid for tid in twins_shuffled if tid not in assigned]

            acc5 = 0.0
            for tid in remaining:
                if acc5 >= target5:
                    break
                s5_ids.append(tid)
                acc5 += float(twin_nvox[int(tid)])

            # remainder assignment to ensure full coverage
            assigned = set(s3_ids) | set(s5_ids)
            remainder = [tid for tid in twins_shuffled if tid not in assigned]

            def3 = target3 - acc3
            def5 = target5 - acc5
            for tid in remainder:
                vol = float(twin_nvox[int(tid)])
                # Prefer filling the larger remaining deficit; if both <=0, balance by current totals
                if def3 > def5:
                    s3_ids.append(tid); acc3 += vol; def3 -= vol
                elif def5 > def3:
                    s5_ids.append(tid); acc5 += vol; def5 -= vol
                else:
                    # equal deficits (or both met): balance by accumulated volumes
                    if acc3 <= acc5:
                        s3_ids.append(tid); acc3 += vol
                    else:
                        s5_ids.append(tid); acc5 += vol
        else:
            raise ValueError("by must be 'count' or 'volume'")

        # final consistency (no duplicates, complete partition)
        s3_set, s5_set = set(s3_ids), set(s5_ids)
        # if any overlap (shouldn't happen), push overlap to the class with smaller size
        overlap = s3_set & s5_set
        if overlap:
            for tid in list(overlap):
                if len(s3_set) > len(s5_set):
                    s3_set.remove(tid)
                else:
                    s5_set.remove(tid)

        # ensure union equals all twins
        assigned_union = s3_set | s5_set
        missing = [tid for tid in twin_ids if tid not in assigned_union]
        if missing:
            # push any missing to the smaller class
            for tid in missing:
                (s3_set if len(s3_set) <= len(s5_set) else s5_set).add(tid)

        s3_ids = list(s3_set)
        s5_ids = list(s5_set)

        # write back
        self.fdb[instance_name]['data']['twin_id_s3'] = np.asarray(s3_ids, np.int32)
        self.fdb[instance_name]['data']['twin_id_s5'] = np.asarray(s5_ids, np.int32)

        assert len(s3_ids) + len(s5_ids) == len(twin_ids)
        assert set(s3_ids).isdisjoint(set(s5_ids))

    def assign_orientations_from_unordered(self, ori_p, ori_s3, ori_s5,
                                           instance_name='twin.0',
                                           tol_deg=5.0,
                                           allow_parent_reuse=True,
                                           seed=2025):
        """
        Assign unordered experimental orientations to concrete features (hosts & twins)
        such that Σ3 twins are ~60° from their host and Σ5 twins are ~36.87° or 53.13°.

        Writes:
          data['ori_parent'] : {gid: (phi1,Phi,phi2)}
          data['ori_twin']   : {tid: (phi1,Phi,phi2)}
          data['ori_assign_report'] : dict
        """
        print(1)
        rng = Random(seed)
        data = self.fdb[instance_name]['data']

        feat_host_ids = list(map(int, data['feat_host_ids']))
        twin_id_s3    = list(map(int, data['twin_id_s3']))
        twin_id_s5    = list(map(int, data['twin_id_s5']))
        twin_map_g_t  = data['twin_map_g_t']    # {parent_gid: [twin_ids,...]}
        print(2)
        # Organize twins by host
        s3_set, s5_set = set(twin_id_s3), set(twin_id_s5)
        s3_by_host = {gid: [t for t in twin_map_g_t.get(gid, []) if t in s3_set]
                      for gid in feat_host_ids}
        s5_by_host = {gid: [t for t in twin_map_g_t.get(gid, []) if t in s5_set]
                      for gid in feat_host_ids}
        # Hosts that actually have any twins to assign
        hosts = [gid for gid in feat_host_ids if (s3_by_host.get(gid) or s5_by_host.get(gid))]
        # Randomize host order but bias by demand (more twins first)
        hosts.sort(key=lambda g: -(len(s3_by_host[g])+len(s5_by_host[g])))
        print(3)
        # Pools (unordered)
        poolP = [tuple(x) for x in ori_p]
        poolS3 = [tuple(x) for x in ori_s3]
        poolS5 = [tuple(x) for x in ori_s5]
        rng.shuffle(poolP); rng.shuffle(poolS3); rng.shuffle(poolS5)
        print(4)
        # Precompute rotation matrices
        def R_of_euler(e):
            return self.cubic_euler_bunge_to_matrix(*e, degrees=True)
        RP  = [R_of_euler(e) for e in poolP]
        RS3 = [R_of_euler(e) for e in poolS3]
        RS5 = [R_of_euler(e) for e in poolS5]
        print(5)
        # Assignment outputs
        ori_parent = {}   # {gid: euler}
        ori_twin   = {}   # {tid: euler}
        host_parent_idx = {}  # {gid: idx in poolP}

        # Helper: best parent index for a given host based on how many twins it can satisfy
        def score_parent_for_host(pidx, gid):
            if pidx is None: return -1e9
            gA = RP[pidx]
            # count Σ3 matches
            s3_need = s3_by_host.get(gid, [])
            s3_hits = 0
            for eR in RS3:
                ang, _, __ = self.cubic_misorientation(gA, eR)
                if abs(ang-60.0) <= tol_deg:
                    s3_hits += 1
            # count Σ5 matches (either target)
            s5_need = s5_by_host.get(gid, [])
            s5_hits = 0
            for eR in RS5:
                ang, _, __ = self.cubic_misorientation(gA, eR)
                if min(abs(ang - 36.86989765), abs(ang - 53.13010235)) <= tol_deg:
                    s5_hits += 1
            # weight by needs to avoid overfitting one class
            return min(s3_hits, len(s3_need)) + min(s5_hits, len(s5_need))

        print(6)
        # Choose parent for each host (greedy)
        available_parents = list(range(len(poolP)))
        for gid in hosts:
            print('.. '+str(gid))
            if not available_parents and not allow_parent_reuse:
                break  # no more parents to assign
            # Evaluate scores
            best_idx, best_score = None, -1e9
            candidates = available_parents if available_parents else range(len(poolP))
            print(f"number of scorings: {len(hosts)*len(candidates)}")
            for pidx in candidates:
                sc = score_parent_for_host(pidx, gid)
                if sc > best_score:
                    best_score, best_idx = sc, pidx
            if best_idx is None:
                continue
            host_parent_idx[gid] = best_idx
            ori_parent[gid] = poolP[best_idx]
            if not allow_parent_reuse and best_idx in available_parents:
                available_parents.remove(best_idx)

        # Assign twins for each host using the chosen parent, picking closest angles first
        def take_best_matching_children(gid, child_pool, child_rotmats, target_list_deg, twin_ids_here):
            assigned = {}
            if gid not in host_parent_idx: return assigned, []
            pidx = host_parent_idx[gid]
            gA = RP[pidx]

            # Build (dev, idx) list
            devs = []
            for j, gB in enumerate(child_rotmats):
                ang, _, __ = self.cubic_misorientation(gA, gB)
                dev = min(abs(ang - t) for t in target_list_deg)
                devs.append((dev, j, ang))
            devs.sort(key=lambda x: x[0])  # smallest deviation first

            used = set()
            for tid in twin_ids_here:
                # find first unused child within tol; if none, take closest (and mark as out-of-tol)
                pick = None
                for d, j, ang in devs:
                    if j in used: continue
                    pick = (d, j, ang)
                    if d <= tol_deg:
                        break
                if pick is None: break
                d, j, ang = pick
                used.add(j)
                assigned[tid] = child_pool[j]
            # return assigned twins and the indices to remove from pools
            return assigned, sorted(used, reverse=True)

        # Σ3 twins
        print('Working on S3 twins')
        for gid in hosts:
            twins_here = s3_by_host.get(gid, [])
            if not twins_here: continue
            assigned, idxs_to_pop = take_best_matching_children(
                gid, poolS3, RS3, [60.0], twins_here
            )
            for tid, e in assigned.items():
                ori_twin[tid] = e
            # remove used children from pool (and rotmats) in descending index order
            for j in idxs_to_pop:
                poolS3.pop(j); RS3.pop(j)

        # Σ5 twins
        print('Working on S5 twins')
        for gid in hosts:
            twins_here = s5_by_host.get(gid, [])
            if not twins_here: continue
            assigned, idxs_to_pop = take_best_matching_children(
                gid, poolS5, RS5, [36.86989765, 53.13010235], twins_here
            )
            for tid, e in assigned.items():
                ori_twin[tid] = e
            for j in idxs_to_pop:
                poolS5.pop(j); RS5.pop(j)

        # Write back + report
        data['ori_parent'] = ori_parent              # {gid: euler}
        data['ori_twin']   = ori_twin                # {tid: euler}

        # Quick validation counts
        def count_pass_s3():
            ok = 0
            for tid in twin_id_s3:
                gid = next((g for g, lst in twin_map_g_t.items() if tid in lst), None)
                if gid in ori_parent and tid in ori_twin:
                    ang, _, __ = self.cubic_misorientation(
                        self.cubic_euler_bunge_to_matrix(*ori_parent[gid], degrees=True),
                        self.cubic_euler_bunge_to_matrix(*ori_twin[tid], degrees=True)
                    )
                    if abs(ang - 60.0) <= tol_deg:
                        ok += 1
            return ok

        def count_pass_s5():
            ok = 0
            for tid in twin_id_s5:
                gid = next((g for g, lst in twin_map_g_t.items() if tid in lst), None)
                if gid in ori_parent and tid in ori_twin:
                    ang, _, __ = self.cubic_misorientation(
                        self.cubic_euler_bunge_to_matrix(*ori_parent[gid], degrees=True),
                        self.cubic_euler_bunge_to_matrix(*ori_twin[tid], degrees=True)
                    )
                    if min(abs(ang - 36.86989765), abs(ang - 53.13010235)) <= tol_deg:
                        ok += 1
            return ok

        report = {
            'tol_deg': tol_deg,
            'hosts_total': len(hosts),
            'hosts_assigned_parent': len(ori_parent),
            's3_total': len(twin_id_s3),
            's3_assigned': len([t for t in twin_id_s3 if t in ori_twin]),
            's3_pass_tol': count_pass_s3(),
            's5_total': len(twin_id_s5),
            's5_assigned': len([t for t in twin_id_s5 if t in ori_twin]),
            's5_pass_tol': count_pass_s5(),
            'parent_reuse': allow_parent_reuse,
        }
        data['ori_assign_report'] = report
        return report

    @staticmethod
    def _matrix_to_euler_bunge(R, degrees=True):
        """
        Convert rotation matrix to Bunge ZXZ Euler (phi1, Phi, phi2).
        Assumes a proper rotation matrix.
        """
        # clamp values for numerical safety
        R = np.asarray(R, dtype=float)
        # Phi from R33
        c = np.clip(R[2,2], -1.0, 1.0)
        Phi = np.arccos(c)
        if abs(Phi) < 1e-12:
            # singular: Phi = 0 -> phi1 + phi2 = atan2(R[1,0], R[0,0])
            phi1 = np.arctan2(R[1,0], R[0,0])
            phi2 = 0.0
        elif abs(Phi - np.pi) < 1e-12:
            # singular: Phi = pi -> phi2 - phi1 = atan2(R[1,2], -R[0,2])
            phi1 = np.arctan2(R[1,2], R[0,2])
            phi2 = 0.0
        else:
            phi1 = np.arctan2(R[2,0], -R[2,1])  # consistent with ZXZ Bunge
            phi2 = np.arctan2(R[0,2], R[1,2])
        if degrees:
            return (np.degrees(phi1) % 360.0, np.degrees(Phi), np.degrees(phi2) % 360.0)
        return (phi1 % (2*np.pi), Phi, phi2 % (2*np.pi))

    @staticmethod
    def _axis_angle_to_R(axis, angle_rad):
        """Rodrigues formula."""
        ax = np.asarray(axis, dtype=float)
        n = np.linalg.norm(ax)
        if n < 1e-15 or abs(angle_rad) < 1e-15:
            return np.eye(3)
        u = ax / n
        ux, uy, uz = u
        K = np.array([[0,-uz, uy],
                      [uz, 0,-ux],
                      [-uy,ux, 0]], dtype=float)
        I = np.eye(3)
        return I + np.sin(angle_rad)*K + (1.0 - np.cos(angle_rad))*(K @ K)

    @staticmethod
    def _rand_unit_vector(rng):
        """Uniform random unit vector on S2."""
        # Method: normal distribution and normalize
        v = np.array([rng.gauss(0,1), rng.gauss(0,1), rng.gauss(0,1)], dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-15:
            return np.array([1.0,0.0,0.0])
        return v / n

    @staticmethod
    def _rand_uniform_SO3(rng):
        """Uniform random rotation using random unit quaternion."""
        u1, u2, u3 = rng.random(), rng.random(), rng.random()
        q1 = np.sqrt(1-u1)*np.sin(2*np.pi*u2)
        q2 = np.sqrt(1-u1)*np.cos(2*np.pi*u2)
        q3 = np.sqrt(u1)*np.sin(2*np.pi*u3)
        q4 = np.sqrt(u1)*np.cos(2*np.pi*u3)
        # quaternion to rotation
        x,y,z,w = q1,q2,q3,q4
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ], dtype=float)
        return R

    @staticmethod
    def _proj_to_so3(R):
        # project to nearest proper rotation via SVD
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn

    @staticmethod
    def _unique_rotations(rotations, tol=1e-8):
        """Deduplicate rotation matrices by Frobenius norm tolerance."""
        uniq = []
        for R in rotations:
            if not any(np.linalg.norm(R - Q, ord='fro') < tol for Q in uniq):
                uniq.append(R)
        return uniq

    def generate_fcc_texture(self, N=1000, seed=np.random.random(),
                             shuffle=True):
        """
        EXAMPLE - 1
        -----------
        N = 1000
        gstslice.tc_info = {"copper": [0.45, 8],
                             "brass": [0.30, 10],
                             "S": [0.15, 7],
                             "goss": [0.05, 6]}

        EA_p, EA_TC = gstslice.generate_fcc_texture(N)
        """
        vf_components = {key: val[0] for key, val in self.tc_info.items()}
        spread_deg = {key: val[1] for key, val in self.tc_info.items()}

        rng = Random(seed)

        # restrict to components present in your stored means
        comps_avail = set(self.fcc_tc.keys())
        comps = [c for c in vf_components.keys() if c in comps_avail]
        if not comps:
            raise ValueError("No requested components found in self.fcc_tc.")

        # fractions (allow sum<1 -> fill with random)
        vf = {c: max(0.0, float(vf_components.get(c, 0.0))) for c in comps}
        vf_sum = sum(vf.values())

        # allocate counts via largest remainder
        raw = {c: vf[c]*N for c in comps}
        base = {c: floor(raw[c]) for c in comps}
        assigned = sum(base.values())
        rema = sorted([(raw[c]-base[c], c) for c in comps], reverse=True)
        while assigned < min(N, int(round(vf_sum*N))) and rema:
            _, c = rema.pop(0)
            base[c] += 1; assigned += 1
        n_random = N - sum(base.values())
        if n_random < 0:
            # clip over-allocation from biggest components
            over = -n_random
            for c,_cnt in sorted(base.items(), key=lambda kv: kv[1], reverse=True):
                take = min(_cnt, over)
                base[c] -= take; over -= take
                if over == 0: break
            n_random = N - sum(base.values())

        comp_eulers = {c: [] for c in comps}
        if n_random > 0:
            comp_eulers["Random"] = []

        # generate clusters per component
        for c in comps:
            count = base[c]
            if count <= 0: continue
            mean_euler = self.fcc_tc[c]
            R_mean = self.cubic_euler_bunge_to_matrix(*mean_euler, degrees=True)
            halfwidth = float(spread_deg.get(c, 5.0))
            for _ in range(count):
                axis = self._rand_unit_vector(rng)
                angle = rng.random() * np.deg2rad(halfwidth)  # uniform within cap
                Rp = self._axis_angle_to_R(axis, angle) @ R_mean
                eul = self._matrix_to_euler_bunge(Rp, degrees=True)
                comp_eulers[c].append(eul)

        # fill the remainder with uniform random orientations
        for _ in range(n_random):
            Rrand = self._rand_uniform_SO3(rng)
            eul = self._matrix_to_euler_bunge(Rrand, degrees=True)
            comp_eulers["Random"].append(eul)

        # flatten + shuffle for a global list
        eulers = [e for lst in comp_eulers.values() for e in lst]
        if shuffle:
            idx = list(range(len(eulers)))
            rng.shuffle(idx)
            eulers = [eulers[i] for i in idx]

        return np.array(eulers), comp_eulers

    @staticmethod
    def standardize_tc_info(tc_info,
                            defaults={'hw_phi1': 5,
                                      'hw_Phi': 5,
                                      'hw_phi2': 5,
                                      'std_k_phi1': 3,
                                      'std_k_Phi': 3,
                                      'std_k_phi2': 3,
                                      'perctol_phi1': 5,
                                      'perctol_Phi': 5,
                                      'perctol_phi2': 5}):
        """
        Standardizes a dictionary of texture component information to a consistent format.

        Parameters:
        - tc_info (dict): The input dictionary containing texture components.
                          Each value can be a scalar or a list of varying length.

        Returns:
        - dict: A new dictionary with all values standardized to the most complete format:
                [percentage, [ang1, ang2, ang3], [std1, std2, std3], [k1, k2, k3]].
                Default values are used to fill in missing information.
        """
        print(".. User texture information standardisation.")
        standardized_info = {}

        for key, value in tc_info.items():
            # Case 1: Value is a scalar (e.g., {"copper": a})
            if not isinstance(value, list):
                standardized_info[key] = [value,
                                          [defaults['hw_phi1'],
                                           defaults['hw_Phi'],
                                           defaults['hw_phi2']],
                                          [defaults['std_k_phi1'],
                                           defaults['std_k_Phi'],
                                           defaults['std_k_phi2']],
                                          [defaults['perctol_phi1'],
                                           defaults['perctol_Phi'],
                                           defaults['perctol_phi2']]
                                          ]

            # Case 2: Value is a list of varying length
            else:
                current_value = list(value)
                percentage = current_value[0]
                # -----------------------------------------------
                # Default for [hw_phi1, hw_Phi, hw_phi2] (euler angles spread)
                if len(current_value) < 2:
                    spreads = [defaults['hw_phi1'],
                               defaults['hw_Phi'],
                               defaults['hw_phi2']]
                elif not isinstance(current_value[1], list):
                    spreads = [current_value[1],
                               current_value[1],
                               current_value[1]]
                else:
                    spreads = current_value[1]
                # -----------------------------------------------
                # Default for [std_k_phi1, std_k_Phi, std_k_phi2] (std_k spread)
                if len(current_value) < 3:
                    std_k = [defaults['std_k_phi1'],
                             defaults['std_k_Phi'],
                             defaults['std_k_phi2']]
                else:
                    std_k = current_value[2]
                # -----------------------------------------------
                # Default for [perctol_phi1, perctol_Phi, perctol_phi2] (taper)
                if len(current_value) < 4:
                    tapers = [defaults['perctol_phi1'],
                              defaults['perctol_Phi'],
                              defaults['perctol_phi2']]
                else:
                    tapers = current_value[3]
                # -----------------------------------------------
                standardized_info[key] = [percentage, spreads, std_k, tapers]

        return standardized_info


    def generate_fcc_texture_v1(self, N=1000, distr='normal',
                                validate_miso=False, calc_miso=True,
                                tc_info_std_defaults={'hw_phi1': 5,
                                                      'hw_Phi': 5,
                                                      'hw_phi2': 5,
                                                      'std_k_phi1': 3,
                                                      'std_k_Phi': 3,
                                                      'std_k_phi2': 3,
                                                      'perctol_phi1': 5,
                                                      'perctol_Phi': 5,
                                                      'perctol_phi2': 5},
                                shuffle=False,
                                ea_dtype=np.float32,
                                miso_dtype=np.float16,
                                id_dtype=np.int16,
                                rand_ori_seed = np.random.random()
                                ):
        """
        # Complete structure of gstslice.tc_info is below.
        gstslice.tc_info = {"tc1": [vf_tc,
                                    [hw_phi1, hw_Phi, hw_phi2],
                                    [std_k_phi1, std_k_Phi, std_k_phi2],
                                    [perctol_phi1, perctol_Phi, perctol_phi2]],
                            }
        In the above,
            * vf_tc. Volume freaction of texture component.
            * [hw_phi1, hw_Phi, hw_phi2]. Half widths about mean orientation.
            * [std_k_phi1, std_k_Phi, std_k_phi2]. Standard deviation factor

            * [perctol_phi1, perctol_Phi, perctol_phi2].
        The user may choose to enter it as any one of the below; or even a
        combination of below. Non-standard inputs willbe converted to standard
        input, whilst retaining valid user provided information, before use
        insdide this function.
            * Possible input for gstslice.tc_info: 1
            gstslice.tc_info = {"copper": 0.35,
                                "brass": 0.30,
                                "S": 0.15,
                                "goss": 0.05,
                                "cube": 0.05}

            * Possible input for gstslice.tc_info: 2
            gstslice.tc_info = {"copper": [0.45, 8],
                                "brass": [0.30, 10],
                                "S": [0.15, 7],
                                "goss": [0.05, 6]}

            * Possible input for gstslice.tc_info: 3
            gstslice.tc_info = {"copper": [0.45, [8, 8, 8]],
                                "brass": [0.30, [10, 10, 10],
                                "S": [0.15, [7, 7, 7]],
                                "goss": [0.05, [6, 6, 6]]}

            * Possible input for gstslice.tc_info: 4
            gstslice.tc_info = {"copper": [0.45, [8, 8, 8], [3, 3, 3]],
                                "brass": [0.30, [10, 10, 10], [3, 3, 3]],
                                "S": [0.15, [7, 7, 7], [3, 3, 3]],
                                "goss": [0.05, [6, 6, 6], [3, 3, 3]]}

            * Possible input for gstslice.tc_info: 5
            Example standard use:
            gstslice.tc_info = {"copper": [0.45, [8,8,8], [3,3,3], [5,5,5]],
                                "brass": [0.30, [10,10,10], [3,3,3], [5,5,5]],
                                "S": [0.15, [7,7,7], [3,3,3], [5,5,5]],
                                "goss": [0.05, [6,6,6], [3,3,3], [5,5,5]]}

        EXAMPLE - 1
        -----------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='demo_3d_04.xls')
        pxt.simulate(verbose=False)
        tslice = 99
        gstslice = pxt.gs[tslice]
        gstslice.tc_info = {"copper": [0.35, 8], "brass": [0.27, 10],
                            "s": [0.15, 7], "goss": [0.10, 6],
                            "cube": [0.05, 10]}
        texture = gstslice.generate_fcc_texture_v1(N=1000)
        """
        print(60*'=', '\n\nGenerating orientations for cryst. texture\n')
        self.tc_info = self.standardize_tc_info(self.tc_info,
                                                tc_info_std_defaults)
        # -----------------------------------------------------------
        '''Volume fraction'''
        VF = {key: val[0] for key, val in self.tc_info.items()}
        '''Hafl - widths representing ori spreads'''
        HW = {key: val[1] for key, val in self.tc_info.items()}
        '''Standard deviation factors to be used to scale down spread for
        generating distribtuion.'''
        SK = {key: val[2] for key, val in self.tc_info.items()}
        '''Percentage tolerance for acceptance.'''
        PT = {key: val[3] for key, val in self.tc_info.items()}
        # -----------------------------------------------------------
        comps_avail = set(self.fcc_tc.keys())
        comps = [c for c in self.tc_info.keys() if c in comps_avail]
        if not comps:
            raise ValueError("No requested components found in self.fcc_tc.")
        # -----------------------------------------------------------
        # fractions (allow sum<1 -> fill with random)
        vf = {c: max(0.0, float(self.tc_info.get(c, 0.0)[0])) for c in comps}
        vf_sum = sum(vf.values())
        # -----------------------------------------------------------
        print('.. Caclulating TC wise orientation counts')
        # allocate counts via largest remainder
        raw = {c: vf[c]*N for c in comps}
        ori_count = {c: floor(raw[c]) for c in comps}
        assigned = sum(ori_count.values())
        rema = sorted([(raw[c]-ori_count[c], c) for c in comps], reverse=True)
        while assigned < min(N, int(round(vf_sum*N))) and rema:
            _, c = rema.pop(0)
            ori_count[c] += 1; assigned += 1
        n_random = N - sum(ori_count.values())
        if n_random < 0:
            # clip over-allocation from biggest components
            over = -n_random
            for c,_cnt in sorted(ori_count.items(),
                                 key=lambda kv: kv[1],
                                 reverse=True):
                take = min(_cnt, over)
                ori_count[c] -= take; over -= take
                if over == 0: break
            n_random = N - sum(ori_count.values())
        # -----------------------------------------------------------
        print('.. Generating ori. clusters around mean texture components')
        comp_eulers = {c: [] for c in comps}
        MISORI = {c: [] for c in comps}
        # generate clusters per component
        for c in comps:
            eacount = ori_count[c]
            if eacount <= 0:
                continue
            '''get the mean Euler angle'''
            meanang = self.fcc_tc[c]
            # -----------------------------------------------
            '''Calculate the scales needed to generate normal random numbers'''
            scales = np.asarray(HW[c])/np.asarray(SK[c])
            """Generate the euler angle cluster.
            BEA: Bunge's Euler angles.
            bea: element of BEA.
            """
            BEA = np.random.normal(loc=self.fcc_tc[c], scale=scales,
                                   size=(eacount, 3))
            # -----------------------------------------------
            '''Calculate the misorientation angles'''
            for bea in BEA:
                mo, _, _ = self.cubic_misorientation_old1(self.fcc_tc[c],
                                                          bea, degrees=True)
                comp_eulers[c].append(bea.tolist())
                MISORI[c].append(mo)
            # -----------------------------------------------
            if validate_miso:
                pass
            # -----------------------------------------------
        comp_eulers = {c: np.array(comp_eulers[c], dtype=ea_dtype) for c in comps}
        MISORI = {c: np.array(MISORI[c], dtype=miso_dtype) for c in comps}
        # -----------------------------------------------
        print('.. Filling remainder with uniform random orientations')
        if n_random > 0:
            comp_eulers["random"] = []
        for _ in range(n_random):
            Rrand = self._rand_uniform_SO3(Random(rand_ori_seed))
            eul = list(self._matrix_to_euler_bunge(Rrand, degrees=True))
            comp_eulers["random"].append(eul)
        comp_eulers["random"] = np.array(comp_eulers["random"], dtype=ea_dtype)
        # -----------------------------------------------
        '''Correct for negative texture component angles'''
        # TO BE DONE
        # -----------------------------------------------
        eulers = np.array([ea for eas in comp_eulers.values()
                           for ea in eas.tolist()], dtype=ea_dtype)
        # -----------------------------------------------
        '''Prepare meta-data'''
        ORIID_MAP = {key: i for i, key in enumerate(comp_eulers.keys(),
                                                    start=1)}
        oriid_map = np.array([], dtype=id_dtype)
        for tc in comp_eulers.keys():
            oriid_map = np.append(oriid_map,
                                  np.full(comp_eulers[tc].shape[0],
                                          ORIID_MAP[tc], dtype=id_dtype))
        ori_count['random'] = n_random
        ori_count['total'] = sum(ori_count.values())
        ori_means = {k: list(self.fcc_tc[k]) for k in self.tc_info.keys()}
        # -----------------------------------------------
        '''Generate Level 1 table of misorinetations. Include misorintations
        between mean orientations only.
        '''

        # -----------------------------------------------
        if shuffle:
            idx = np.arange(len(eulers), dtype=np.int32)
            np.random.shuffle(idx)
            eulers = eulers[idx]
            oriid_map = oriid_map[idx]
        # -----------------------------------------------
        texture = {'ori': eulers,
                   'tc_ori': comp_eulers,
                   'ori_count': ori_count,
                   'ORIID_MAP': ORIID_MAP,
                   'oriid_map': oriid_map,
                   'MISORI': MISORI,
                   'tc_info': self.tc_info,
                   'ori_means': ori_means
                   }
        # -----------------------------------------------
        return texture

    @staticmethod
    def normalize_euler_bunge(ea, degrees=True, eps=1e-6):
        """
        Normalize Bunge ZXZ Euler angles (phi1, Phi, phi2) to canonical ranges.
          - phi1, phi2 in [0, 360) deg  (or [0, 2π) rad)
          - Phi in   [0, 180]  deg      (or [0, π]   rad)
        Handles BOTH Phi > 180 and Phi < 0 via ZXZ symmetry:
           Phi' = -Phi  and (phi1', phi2') = (phi1+180, phi2+180)  [mod 360]
           Phi' = 360-Phi and same 180-shift for >180 case.
        """
        A = np.asarray(ea, dtype=float)
        A2 = np.atleast_2d(A)
        phi1, Phi, phi2 = A2[:, 0], A2[:, 1], A2[:, 2]

        if degrees:
            two_pi, pi = 360.0, 180.0
        else:
            two_pi, pi = 2*np.pi, np.pi

        # Wrap phi1, phi2 into [0, 360) or [0, 2π)
        phi1[:] = np.mod(phi1, two_pi)
        phi2[:] = np.mod(phi2, two_pi)

        # First, reduce Phi to (-pi, pi] to avoid big excursions
        # (helps when jitter pushes way beyond range)
        Phi[:] = ((Phi + pi) % (2*pi)) - pi

        # Case 1: Phi < 0  -> mirror about 0 and shift (phi1,phi2)+=180°
        neg = Phi < 0.0
        if np.any(neg):
            Phi[neg] = -Phi[neg]
            phi1[neg] = np.mod(phi1[neg] + pi, two_pi)
            phi2[neg] = np.mod(phi2[neg] + pi, two_pi)

        # Case 2: Phi > 180 -> fold to 360 - Phi, shift (phi1,phi2)+=180°
        over = Phi > pi
        if np.any(over):
            Phi[over] = 2*pi - Phi[over]
            phi1[over] = np.mod(phi1[over] + pi, two_pi)
            phi2[over] = np.mod(phi2[over] + pi, two_pi)

        # Snap very small numerical negatives/endpoint jitter
        if eps is not None:
            Phi[np.abs(Phi) < eps] = 0.0
            Phi[np.abs(Phi - pi) < eps] = pi

        out = np.stack([phi1, Phi, phi2], axis=-1)
        return out if A2.shape[0] > 1 else out[0]

    def generate_fcc_texture_v2(self, N=1000, distr='normal',
                                validate_miso=False, calc_miso=True,
                                tc_info_std_defaults={'hw_phi1': 5,
                                          'hw_Phi': 5,
                                          'hw_phi2': 5,
                                          'std_k_phi1': 3,
                                          'std_k_Phi': 3,
                                          'std_k_phi2': 3,
                                          'perctol_phi1': 5,
                                          'perctol_Phi': 5,
                                          'perctol_phi2': 5},
                                shuffle=False,
                                nshuffles=2,
                                ea_dtype=np.float32,
                                miso_dtype=np.float16,
                                id_dtype=np.int16,
                                rand_ori_seed = np.random.random(),
                                rand_ori_gen_rule = 'relaxed',
                                n_tex_instances=4,
                                n_sampling_instances=4
                                ):
        """
        # Complete structure of gstslice.tc_info is below.
        gstslice.tc_info = {"tc1": [vf_tc,
                                    [hw_phi1, hw_Phi, hw_phi2],
                                    [std_k_phi1, std_k_Phi, std_k_phi2],
                                    [perctol_phi1, perctol_Phi, perctol_phi2]],
                            }
        In the above,
            * vf_tc. Volume freaction of texture component.
            * [hw_phi1, hw_Phi, hw_phi2]. Half widths about mean orientation.
            * [std_k_phi1, std_k_Phi, std_k_phi2]. Standard deviation factor

            * [perctol_phi1, perctol_Phi, perctol_phi2].
        The user may choose to enter it as any one of the below; or even a
        combination of below. Non-standard inputs willbe converted to standard
        input, whilst retaining valid user provided information, before use
        insdide this function.
            * Possible input for gstslice.tc_info: 1
            gstslice.tc_info = {"copper": 0.35,
                                "brass": 0.30,
                                "s": 0.15,
                                "goss": 0.05,
                                "cube": 0.05}

            * Possible input for gstslice.tc_info: 2
            gstslice.tc_info = {"copper": [0.45, 8],
                                "brass": [0.30, 10],
                                "s": [0.15, 7],
                                "goss": [0.05, 6]}

            * Possible input for gstslice.tc_info: 3
            gstslice.tc_info = {"copper": [0.45, [8, 8, 8]],
                                "brass": [0.30, [10, 10, 10],
                                "s": [0.15, [7, 7, 7]],
                                "goss": [0.05, [6, 6, 6]]}

            * Possible input for gstslice.tc_info: 4
            gstslice.tc_info = {"copper": [0.45, [8, 8, 8], [3, 3, 3]],
                                "brass": [0.30, [10, 10, 10], [3, 3, 3]],
                                "s": [0.15, [7, 7, 7], [3, 3, 3]],
                                "goss": [0.05, [6, 6, 6], [3, 3, 3]]}

            * Possible input for gstslice.tc_info: 5
            Example standard use:
            gstslice.tc_info = {"copper": [0.45, [8,8,8], [3,3,3], [5,5,5]],
                                "brass": [0.30, [10,10,10], [3,3,3], [5,5,5]],
                                "s": [0.15, [7,7,7], [3,3,3], [5,5,5]],
                                "goss": [0.05, [6,6,6], [3,3,3], [5,5,5]]}

        EXAMPLE - 1
        -----------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='demo_3d_04.xls')
        pxt.simulate(verbose=False)
        tslice = 99
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=False,
                                           find_neigh=[True, [1], False, '1-no'],
                                           find_spatial_bounds_of_grains=False,
                                           force_compute=True, set_mprops=False,
                                           mprops_kwargs={'set_skimrp': False,
                                                          'volnv': True,
                                                          'solidity': False,
                                                          'sanv': False,
                                                          'rat_sanv_volnv': False})
        gstslice.tc_info = {"copper": [0.35, 8], "brass": [0.27, 10],
                            "s": [0.15, 7], "goss": [0.10, 6],
                            "cube": [0.05, 10]}
        TEX = gstslice.generate_fcc_texture_v2(N=gstslice.n, distr='normal',
                                               validate_miso=False,
                                               calc_miso=True,
                                               tc_info_std_defaults={
                                                   'hw_phi1': 5,
                                                   'hw_Phi': 5,
                                                   'hw_phi2': 5,
                                                   'std_k_phi1': 3,
                                                   'std_k_Phi': 3,
                                                   'std_k_phi2': 3,
                                                   'perctol_phi1': 5,
                                                   'perctol_Phi': 5,
                                                   'perctol_phi2': 5},
                                               shuffle=False,
                                               nshuffles=2,
                                               ea_dtype=np.float32,
                                               miso_dtype=np.float16,
                                               id_dtype=np.int16,
                                               rand_ori_seed=np.random.random(),
                                               rand_ori_gen_rule='relaxed',
                                               n_tex_instances=1,
                                               n_sampling_instances=1)
        # VALIDATION
        ngrains = gstslice.n
        noritc = [len(v) for v in TEX['tex_instance.1']['sampling_instances']['ossi.1']['tc_ori_stacks'].values()]
        ngrains == sum(noritc)
        """
        # FUNCTION LEVEL (FL) | 2 (i.e. number of tabs at start). In concoise,
        # This wil be written as in below line. These are markers / flags to
        # indicate the code indentations which give a feel of where we are.
        # (FL | 2)
        print(60*'=', '\n\nGenerating orientations for cryst. texture\n')
        self.tc_info = self.standardize_tc_info(self.tc_info,
                                                tc_info_std_defaults)
        ori_means = {k: list(self.fcc_tc[k]) for k in self.tc_info.keys()}
        # =============================================================
        '''Volume fraction'''
        VF = {key: val[0] for key, val in self.tc_info.items()}
        '''Hafl - widths representing ori spreads'''
        HW = {key: val[1] for key, val in self.tc_info.items()}
        '''Standard deviation factors to be used to scale down spread for
        generating distribtuion.'''
        SK = {key: val[2] for key, val in self.tc_info.items()}
        '''Percentage tolerance for acceptance.'''
        PT = {key: val[3] for key, val in self.tc_info.items()}
        # =============================================================
        comps_avail = set(self.fcc_tc.keys())
        tc_comps = [c for c in self.tc_info.keys() if c in comps_avail]
        # =============================================================
        # fractions (allow sum<1 -> fill with random)
        vf = {c: max(0.0, float(self.tc_info.get(c, 0.0)[0])) for c in tc_comps}
        vf_sum = sum(vf.values())
        # -----------------------------------------------------------
        print('.. Caclulating TC wise orientation counts')
        # =============================================================
        # allocate counts via largest remainder
        raw = {c: vf[c]*N for c in tc_comps}
        ori_count = {c: floor(raw[c]) for c in tc_comps}
        assigned = sum(ori_count.values())
        rema = sorted([(raw[c]-ori_count[c], c) for c in tc_comps], reverse=True)
        while assigned < min(N, int(round(vf_sum*N))) and rema:
            _, c = rema.pop(0)
            ori_count[c] += 1; assigned += 1
        n_random = N - sum(ori_count.values())
        if n_random < 0:
            # clip over-allocation from biggest components
            over = -n_random
            for c,_cnt in sorted(ori_count.items(),
                                 key=lambda kv: kv[1],
                                 reverse=True):
                take = min(_cnt, over)
                ori_count[c] -= take; over -= take
                if over == 0: break
            n_random = N - sum(ori_count.values())
        ori_count['random'] = n_random
        # =============================================================
        print("Building TEX dictionary template")
        tex_instances_num = np.arange(1, n_tex_instances+1, 1)
        TEX = {f"tex_instance.{i}": {} for i in tex_instances_num}
        TEX['n'] = n_tex_instances
        TEX['tex_instances'] = n_tex_instances
        TEX['n_sampling_instances'] = n_sampling_instances
        TEX['tc_info_std_defaults'] = tc_info_std_defaults
        TEX['shuffle'] = shuffle
        TEX['tc_info'] = self.tc_info
        TEX['fcc_tc'] = self.fcc_tc
        # =============================================================
        MOFUNC = self.cubic_misorientation_old1
        sym_eq_IDs = np.arange(1, 24+1, 1)
        sampling_instances_ids = np.arange(1, n_sampling_instances+1, 1)
        # (FL | 2)
        for ti in tex_instances_num:
            # (FL | 2) > (TEX INST | 3)
            tiname = f"tex_instance.{ti}"
            print('\n', 75*'=')
            print(f".. TEXTURE INSTANCE NUMBER: {ti} of {n_tex_instances}\n")
            print('.. Generating ori. clusters around mean texture components')
            comp_eulers = {tc_comp: [] for tc_comp in tc_comps}
            MISORI = {tc_comp: [] for tc_comp in tc_comps}
            """
            BEA_SYMEQ_MO: Misorinetation angle of all orientations
            around each symmetric equivalent of the mean oreintation of
            every texcture component specified by user. Note; See descriptions
            for variable BEA_SYMEQ_MO_TC (in below codes) for more detauils.

            BEA_SYMEQ_MO.keys()
            dict_keys(['copper', 'brass', 's', 'goss', 'cube'])

            After processing, this shoudl have.
            * len(BEA_SYMEQ_MO['copper']) --> 24
            * len(BEA_SYMEQ_MO['copper'][0]) == ori_count['copper'] --> True
            """
            symeq = {tc_comp: None for tc_comp in tc_comps}
            # generate clusters per component
            for tc_comp in tc_comps:
                # (FL | 2) > (TEX INST | 3) > (TEX COMP | 4)
                eacount = ori_count[tc_comp]
                '''if eacount <= 0:
                    continue'''
                '''get the mean Euler angle of this texture component'''
                meanang = self.fcc_tc[tc_comp]
                # -------------------------------------------------------------
                '''Calculate the scales needed to generate normal random
                numbers.'''
                scales = np.asarray(HW[tc_comp])/np.asarray(SK[tc_comp])
                # -------------------------------------------------------------
                """Generate the euler angle cluster.

                BEA_STACK: Stack of Bunge's Euler angles.

                bea: element of BEA.
                """

                """
                1. Gather all symmetric equivalents of this TC EA.
                loc_symm_eq: location - symmetric equivalents in Euler
                             space
                """
                loc_symm_eq = self.fcc_symmetrise_ori(self.fcc_tc[tc_comp],
                                                      dtype=ea_dtype)
                """
                2. Generate orientations around each ea in loc_symm_eq.

                BEA_SYMEQ: Bunge's Euler angles of all orientations around
                each symmetric equivalent of the mean oreintation of this
                texcture component. list of 24 numpy arrays. Each numpy is
                array of size defined by the number of orientation samples
                defined by 'eacount'. Note that 'eacount' is obtained using
                texture volume fraction.

                BEA_SYMEQ_MO_TC: Misorinetation angle of all orientations
                around each symmetric equivalent of the mean oreintation of
                this texcture component. list of 24 numpy array. Each numpy
                is array of size defined by the number of orientation
                samples defined by 'eacount'. Note that 'eacount' is
                obtained using texture volume fraction.

                BEA_SYMEQ_MO_TC_ids: Symmetric porientaion ID number. it
                can be any valkue bwtween 1 and 24, inclusive. Needed when
                misorintation or orientations need to be reverse-mapped
                to individual orientations in 'loc_symm_eq'
                """
                BEA_SYMEQ = [np.random.normal(loc=ea, scale=scales,
                                              size=(eacount, 3)).astype(ea_dtype)
                             for ea in loc_symm_eq]
                BEA_SYMEQ = [self.normalize_euler_bunge(bea, degrees=True).astype(ea_dtype)
                             for bea in BEA_SYMEQ]
                print("Calculating misorientations across all symmetric",
                      f"equivalents of '{tc_comp}' texture component:",
                      f"{24*eacount} oris")
                BEA_SYMEQ_MO_TC = []
                BEA_SYMEQ_MO_TC_ids = []
                for i, lseq, bea_seq in zip(sym_eq_IDs, loc_symm_eq, BEA_SYMEQ):
                    # (FL | 2) > (TEX INST | 3) > (TEX COMP | 4) > (ORI SYMM | 5)
                    if i % 8 == 0:
                        print(".. TC: '{tc_comp}'. MO calculation for",
                              f" symmetric equivalent {i} of {len(loc_symm_eq)}")
                    bea_seq_mo = np.zeros(bea_seq.shape[0], dtype=ea_dtype)
                    for ii, _bea_seq_ in enumerate(bea_seq):
                        _mo_, _, _ = MOFUNC(lseq, _bea_seq_, degrees=True)
                        bea_seq_mo[ii] = _mo_
                    BEA_SYMEQ_MO_TC.append(bea_seq_mo)
                    ___ = np.asarray([i for _ in range(bea_seq.shape[0])],
                                     dtype=np.int16)
                    BEA_SYMEQ_MO_TC_ids.append(___)

                """
                3.
                Stack up all orientations in the above list.
                """
                # (FL | 2) > (TEX INST | 3) > (TEX COMP | 4)
                BEA_STACK = np.vstack(BEA_SYMEQ)
                BEA_MO_STACK = np.hstack(BEA_SYMEQ_MO_TC)
                BEA_MO_IDS_STACK = np.hstack(BEA_SYMEQ_MO_TC_ids)
                """
                4.
                Shuffle BEA_STACK 'nshuffles' bnumber of times for randomness
                """
                shuffle_ids = np.arange(BEA_STACK.shape[0])
                for _ in range(nshuffles):
                    np.random.shuffle(shuffle_ids)
                    BEA_STACK = BEA_STACK[shuffle_ids]
                    BEA_MO_STACK = BEA_MO_STACK[shuffle_ids]
                    BEA_MO_IDS_STACK = BEA_MO_IDS_STACK[shuffle_ids]
                """
                5.
                Store symmetric equivalnet angles.

                data access:
                    [A] loc_symm_eq
                    All symmetric equivalents of mean TC orientation
                    1. symeq[tc_comp]['loc_symm_eq']
                    -----------------------------------------------------------
                    [B] BEA_SYMEQ
                    Ori distr of each of above symm eq. list of 24 np arr
                    1. symeq[tc_comp]['BEA_SYMEQ']
                    2. symeq[tc_comp]['BEA_SYMEQ'][3]
                       Content: Bunge's EA triplets (BEA), centred around the
                                symeq, symeq[tc_comp]['loc_symm_eq'][3]
                    2. len(symeq[tc_comp]['BEA_SYMEQ'][3])
                       Count: ori_count[tc_comp] or eacount in this loop
                    -----------------------------------------------------------
                    [C, D] BEA_shuffled, shuffle_ids
                    BEA_shuffled: Randomly shuffled np arr,
                         X = np.vstack(symeq[tc_comp]['BEA_SYMEQ'])
                    shuffle_ids: Shuffling order IDs in reference to np arr, X
                    NOTE: BEA_shuffled = X[shuffle_ids]
                    C1. symeq[tc_comp]['BEA_shuffled']
                    C2. len(symeq[tc_comp]['BEA_shuffled'])
                    D1. symeq[tc_comp]['shuffle_ids']
                    D2. len(symeq[tc_comp]['shuffle_ids'])
                    Counts of C and D: 24 * ori_count[tc_comp]
                    -----------------------------------------------------------
                    [E, F] BEA_SYMEQ_MO_TC, BEA_SYMEQ_MO_TC_ids
                    BEA_SYMEQ_MO_TC
                    1. symeq[tc_comp]['BEA_SYMEQ_MO_TC']
                       Content: list if 24 np. arrays
                    1.1. symeq[tc_comp]['BEA_SYMEQ_MO_TC'][3]
                       Count: ori_count[tc_comp] or eacount in this loop
                       Content: misori ang b/w every angle in
                          symeq[tc_comp]['BEA_SYMEQ'][3] with
                          symeq[tc_comp]['loc_symm_eq'][3]

                    symeq[tc_comp]['BEA_SYMEQ_MO_TC_ids']
                    -----------------------------------------------------------
                    symeq[tc_comp]['BEA_MO_STACK']

                    len(symeq[tc_comp]['loc_symm_eq'])
                    len(symeq[tc_comp]['BEA_SYMEQ'])
                    len(symeq[tc_comp]['BEA_shuffled'])
                    len(symeq[tc_comp]['shuffle_ids'])
                    len(symeq[tc_comp]['BEA_SYMEQ_MO_TC'])
                    len(symeq[tc_comp]['BEA_SYMEQ_MO_TC_ids'])
                    len(symeq[tc_comp]['BEA_MO_STACK'])
                """
                # (FL | 2) > (TEX INST | 3) > (TEX COMP | 4)
                symeq[tc_comp] = {'loc_symm_eq': loc_symm_eq,
                                  'BEA_SYMEQ': BEA_SYMEQ,
                                  'BEA_shuffled': BEA_STACK,
                                  'shuffle_ids': shuffle_ids,
                                  'BEA_SYMEQ_MO_TC': BEA_SYMEQ_MO_TC,
                                  'BEA_SYMEQ_MO_TC_ids': BEA_SYMEQ_MO_TC_ids,
                                  'BEA_MO_STACK': BEA_MO_STACK,
                                  'BEA_MO_IDS_STACK': BEA_MO_IDS_STACK
                                  }
            # =============================================================
            # (FL | 2) > (TEX INST | 3)
            TEX[tiname]['symeq_full'] = symeq
            # --------------------------------
            # len(TEX[tiname]['symeq_full'][tc_comp]['BEA_SYMEQ_MO_TC_ids'])
            # tc_ori_range = range(symeq[tc_comp]['BEA_shuffled'].shape[0])
            # =============================================================
            # (FL | 2) > (TEX INST | 3).
            print('\n')
            print(f".. Generating {len(sampling_instances_ids)} SAMPLING INSTANCES\n")
            sampling_instances = {f"ossi.{si}": {} for si in sampling_instances_ids}
            sampling_instances["metadata"] = {"ossi": "Orientation sub-set instance"}
            for si in sampling_instances_ids:
                # (FL | 2) > (TEX INST | 3) > (SAMP INST | 4)
                # NOTE: SAMP INST: Sampling instances
                siname = f"ossi.{si}"
                print(f".... Sampling instance: '{siname}' @ {tc_comps}")
                sampling_instances[siname]['nsamples'] = {}
                sampling_instances[siname]['tc_ori'] = {}
                sampling_instances[siname]['tc_ori_sample_ids'] = {}
                for tc_comp in tc_comps:
                    # 0
                    # sampling_instances[siname]['tc_ori'][tc_comp] = None
                    # 1
                    nsamples = ori_count[tc_comp]
                    sampling_instances[siname]['nsamples'][tc_comp] = nsamples
                    # 2
                    # tc_ori_sample_ids = np.random.choice(tc_ori_range, nsamples)
                    # sampling_instances[siname]['tc_ori_sample_ids'][tc_comp] = tc_ori_sample_ids
                    # 3
                    """
                    Get all 24 symmetric equivalent BEAs of the curerent TC.
                    """
                    beasets = TEX[tiname]['symeq_full'][tc_comp]['BEA_SYMEQ']
                    # 4
                    """
                    Allocating the total number of samples needed for random
                    picking. These sample counts coorespond to that from each
                    of the 24 orintatipon sample sets.
                    """
                    pss = nsamples // 24 # preliminary sample size
                    fss = nsamples % 24  # final sample size
                    samplrng = range(nsamples)
                    symorirng = range(24)
                    # 5
                    if pss >= 1:
                        sel_ori = [None for _ in symorirng]
                        sel_ori_symm_id = [None for _ in symorirng]
                        for i, beas in enumerate(beasets):
                            rand_ids = np.random.choice(samplrng,
                                                        pss).astype(np.int16)
                            sel_ori[i] = beas[rand_ids]
                            sel_ori_symm_id[i] = rand_ids
                    if fss != 0:
                        for i, beas in enumerate(beasets):
                            if i+1 <= fss:
                                rid = np.random.choice(samplrng,
                                                       1).astype(np.int16)
                                sel_ori[i] = np.vstack((sel_ori[i], beas[rid]))
                                sel_ori_symm_id[i] = np.hstack((sel_ori_symm_id[i],
                                                                rid))
                            else:
                                break
                    # 6
                    sampling_instances[siname]['tc_ori'][tc_comp] = sel_ori
                    sampling_instances[siname]['tc_ori'][tc_comp+'_stack'] = np.vstack(sel_ori)
                    sampling_instances[siname]['tc_ori_sample_ids'][tc_comp] = sel_ori_symm_id
                # -------------------------------------------
                print('...... Filling remainder with uniform random orientations')
                if n_random == 0:
                    randori = np.empty((0, 3))
                else:
                    randori = [None for _ in range(n_random)]
                    if rand_ori_gen_rule == 'relaxed':
                        for i in range(n_random):
                            ro = self._rand_uniform_SO3(Random(np.random.random()))
                            ro = list(self._matrix_to_euler_bunge(ro, degrees=True))
                            randori[i] = ro
                    elif rand_ori_gen_rule == 'strict':
                        """
                        This will ensure none of the random orientations
                        generated will have a misorientation angle less than X
                        degrees from any of the symmetric equivalents of each
                        of the user prescribed texture component mean
                        orientations. value of X will be norm of the
                        half-widths specififed for the corresponding texture
                        component.
                        """
                        pass
                    randori = np.array(randori, dtype=np.float32)

                sampling_instances[siname]['tc_ori']['random'] = randori
                # -------------------------------------------
                tc_comps_full = deepcopy(tc_comps)
                tc_comps_full.append('random')
                tc_ori_stacks = {tc_comp: sampling_instances[siname]['tc_ori'][tc_comp+'_stack']
                                 if tc_comp != 'random'
                                 else sampling_instances[siname]['tc_ori'][tc_comp]
                                 for tc_comp in tc_comps_full}
                sampling_instances[siname]['tc_ori_stacks'] = tc_ori_stacks
            # =============================================================
            # (FL | 2) > (TEX INST | 3)
            TEX[tiname]['sampling_instances'] = sampling_instances
            # TEX[tiname]['sampling_instances'][siname]['tc_ori'].keys()
            # TEX[tiname]['sampling_instances'][siname]['tc_ori_stacks'].keys()
            # =============================================================
        return TEX

    def tc_ori_stack_subset(self, ori_stack=None,
                            id_dict={'copper': [], 'brass': [], 's': [],
                                     'goss': [], 'cube': [], 'random': []},
                            invert_ids=True):
        if invert_ids:
            for tc in id_dict:
                if tc.lower() in ori_stack.keys():
                    tcidset = set(list(id_dict[tc]))
                    tcidsetall = set(list(range(len(ori_stack[tc]))))
                    id_dict[tc] = np.array(list(tcidsetall-tcidset),
                                           dtype=np.int32)
                else:
                    id_dict[tc] = np.array([], dtype=np.int32)

        for tc in id_dict:
            if tc.lower() in ori_stack.keys():
                ori_stack[tc] = ori_stack[tc][id_dict[tc]]
            else:
                print(f"TC: {tc} not a key in ori_stack.")
        return ori_stack

    def choose_TC(self, tiname='tex_instance.1', siname='ossi.1',
                  tc_ori_stacks=None, ntgrains=None, ntgrains_level=1):
        """
        Choose texture components to associate with non-touching O(1) grains.

        Parameters
        ----------
        tiname: str
            Texture instance name
        siname: str
            sampling instance name
        tc_ori_stacks: dict
            Texture component orientation stacks. Stacks of orientations drawn
            uniformly randomly from the collection of all 24 symmetric
            equivalents of every orientation belonging to a texture component.
            Keys contain name of the texture component. Value contains numpy
            array of corresponding crystallographci orienrtations.
        ntgrains

        =======================================================================
        tiname = 'tex_instance.1'
        siname = 'ossi.1'

        EXAMPLE - 1
        -----------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='demo_3d_04.xls')
        pxt.simulate(verbose=False)
        tslice = 99
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=False,
                                           find_neigh=[True, [1], False, '1-no'],
                                           find_spatial_bounds_of_grains=False,
                                           force_compute=True, set_mprops=False,
                                           mprops_kwargs={'set_skimrp': False,
                                                          'volnv': True,
                                                          'solidity': False,
                                                          'sanv': False,
                                                          'rat_sanv_volnv': False})
        gstslice.tc_info = {"copper": [0.35, 8], "brass": [0.27, 10],
                            "s": [0.15, 7], "goss": [0.10, 6],
                            "cube": [0.05, 10]}
        TEX = gstslice.generate_fcc_texture_v2(N=gstslice.n, distr='normal',
                                               validate_miso=False,
                                               calc_miso=True,
                                               tc_info_std_defaults={
                                                   'hw_phi1': 5,
                                                   'hw_Phi': 5,
                                                   'hw_phi2': 5,
                                                   'std_k_phi1': 3,
                                                   'std_k_Phi': 3,
                                                   'std_k_phi2': 3,
                                                   'perctol_phi1': 5,
                                                   'perctol_Phi': 5,
                                                   'perctol_phi2': 5},
                                               shuffle=False,
                                               nshuffles=2,
                                               ea_dtype=np.float32,
                                               miso_dtype=np.float16,
                                               id_dtype=np.int16,
                                               rand_ori_seed=np.random.random(),
                                               rand_ori_gen_rule='relaxed',
                                               n_tex_instances=1,
                                               n_sampling_instances=1)
        """
        print(40*'-')
        ntgrains = np.asarray(ntgrains, dtype=np.int32)
        tc, ntc = list(tc_ori_stacks.keys()), len(tc_ori_stacks)
        tcids = {_tc_: i for i, _tc_ in enumerate(tc, start=1)}
        noritc = np.array([len(v) for v in tc_ori_stacks.values()])
        if ntgrains_level == 1:
            if sum(noritc) != self.n:
                raise ValueError("No. of grains & No. of xtal oris do not match !")
        ntg_vf = len(ntgrains)/noritc.sum()
        print(f"Vol. frac. of max set of O(1) non-touching grains: {ntg_vf}")
        alloc = np.asarray(np.floor(noritc*ntg_vf), dtype=np.int32)
        del_alloc = len(ntgrains)-alloc.sum()
        print('Allocating texture components to O(1) non-touching grains.')
        while del_alloc != 0:
            if del_alloc >= ntc:
                # ids = np.random.choice(range(ntc), ntc, replace=False)
                alloc = alloc + 1
            else:
                alloc[np.random.choice(range(ntc),
                                       del_alloc,
                                       replace=False)] += 1
            del_alloc = len(ntgrains) - alloc.sum()
        alloc_ng = {_tc_: _alloc_ for _tc_, _alloc_ in zip(tc, alloc)}
        alloc_gids = {_tc_: None for _tc_ in tc}
        alloc_gids_neg = {_tc_: None for _tc_ in tc}
        ntgrains_ = set(ntgrains)
        for _tc_ in tc:
            alloc_gids[_tc_] = np.random.choice(list(ntgrains_),
                                                alloc_ng[_tc_],
                                                replace=False)
            ntgrains_ = ntgrains_-set(alloc_gids[_tc_])
            alloc_gids_neg = np.array(list(set(self.gid)-set(alloc_gids[_tc_])),
                                      dtype=np.int32)

        if len(ntgrains_) > 0:
            raise ValueError('All gids not allocated for ntgrains !')

        tc_ntg_vf_lcl = [v.size/len(ntgrains) for v in alloc_gids.values()]
        tc_ntg_vf_gbl = [v.size/self.n for v in alloc_gids.values()]

        print('Non-Touching O(1) grains TC allotment successfull.')
        for i, (k, v) in enumerate(alloc_gids.items()):
            print(f".. {v.size} grains allocated to TC: {k}",
                  f"Vfs (local, global): ({tc_ntg_vf_lcl[i]}, {tc_ntg_vf_gbl[i]})")

        print(f"Vfs sum: (local, global): {sum(tc_ntg_vf_lcl)}, {sum(tc_ntg_vf_gbl)}")
        print(f"Vf of Non-touching grains: {len(ntgrains)/self.n}")

        ntgrains_tc_alloc = {'alloc': alloc,
                             'del_alloc': del_alloc,
                             'alloc_gids': alloc_gids,
                             'alloc_gids_neg': alloc_gids_neg,
                             'tc_ntg_vf_lcl': tc_ntg_vf_lcl,
                             'tc_ntg_vf_lcl_sum': sum(tc_ntg_vf_lcl),
                             'tc_ntg_vf_gbl': tc_ntg_vf_gbl}
        print(40*'-')

        return ntgrains_tc_alloc

    def crop_neigh_gid(self, neigh_gid='O(1)', gids_to_crop=None):
        """
        Removes guids in gids_to_crop from neighbour order dictionary.

        Both keys and appearences in values get removed.

        Parameters
        ----------
        neigh_gid: str | dict
            Neighbour gid dictionary. If a string liuke 'O(1)' is entered,
            then the corresponding will be extracted. Idf the neighbour
            does not exist, the method should rauise an error and stop. If the
            value is dictionary, then entered value will be used without any
            validations. Defaults to 'O(1)'.
        gids_to_crop: dth.dt.ITERABLES | integer
            Grain ids to be cropped from neigh_gid. Value could be any in
            dth.dt.ITERABLES or of any integrer type in dth.dt.INTEGERS.
            Defaults to None.

        Example
        -------
        gstlice.crop_neigh_gid(neigh_gid='O(1)', gids_to_crop=[1])

        '''
        Development:
        a = {1: [3, 4, 5, 6, 2], 2: [1, 3, 5, 6, 4],
             3: [2, 1, 5, 7, 8], 4: [1, 2, 9, 6]}
        gids_to_crop = [2, 3, 10]
        for gidcrop in gids_to_crop:
            if gidcrop in a.keys():
                a.pop(gidcrop)
        gids_to_crop = set(gids_to_crop)
        for gid in a.keys():
            a[gid] = set(a[gid])
            a[gid] = a[gid] - gids_to_crop
            a[gid] = list(a[gid])
        # Expected value of a:
        a = {1: [4, 5, 6], 4: [1, 9, 6]}'''
        """
        # -----------------------------------------------
        # User input validations
        if type(neigh_gid) != dict:
            raise ValueError('neigh_gid invalid')
        if type(gids_to_crop) in dth.dt.INTEGERS:
            gids_to_crop = [gids_to_crop]
        if type(gids_to_crop) not in dth.dt.ITERABLES:
            raise ValueError('gids_to_crop invalid')
        # -----------------------------------------------
        if type(neigh_gid) == str:
            if neigh_gid.lower() == 'o(1)':
                neigh_gid = self.neigh_gid
            else:
                # Other codes for later development if needed.
                # We can work on other order dictionaries here.
                pass
        # -----------------------------------------------
        # 1. Remove keys and values
        for gidcrop in gids_to_crop:
            neigh_gid.pop(gidcrop)
        # 2. Remove gids from the values of other gids.
        gids_to_crop = set(gids_to_crop)

        for gid in neigh_gid.keys():
            neigh_gid[gid] = set(neigh_gid[gid])
            neigh_gid[gid] = neigh_gid[gid] - gids_to_crop
            neigh_gid[gid] = list(neigh_gid[gid])

        return neigh_gid

    def map_ori(self, TEXdb, texture_instance=1, sampling_instance=1,
                tc_ori_stacks=None, ntex_map_instances_L0=3,
                ntex_map_instances_L1=2, shuffle_ori_stack_bf_L1Map=True):
        # =====================================================================
        # 1 --> MAP TEXRTURE TO NON-TOUCHING O(1) GRAINS
        """
        texture_instance=1
        sampling_instance=1
        ntex_map_instances_L0=3
        ntex_map_instances_L1=2
        shuffle_ori_stack_bf_L1Map=True

        tiname = 'tex_instance.' + str(texture_instance)
        siname = 'ossi.' + str(sampling_instance)
        TEXdb = TEX
        tc_ori_stacks = TEXdb[tiname]['sampling_instances'][siname]['tc_ori_stacks']

        PRE EXAMPLE
        -----------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(study='independent', input_dashboard='demo_3d_04.xls')
        pxt.simulate(verbose=False)
        tslice = 99
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=False,
                                           find_neigh=[True, [1], False, '1-no'],
                                           find_spatial_bounds_of_grains=False,
                                           force_compute=True, set_mprops=False,
                                           mprops_kwargs={'set_skimrp': False,
                                                          'volnv': True,
                                                          'solidity': False,
                                                          'sanv': False,
                                                          'rat_sanv_volnv': False})
        gstslice.tc_info = {"copper": [0.35, 8], "brass": [0.27, 10],
                            "s": [0.15, 7], "goss": [0.10, 6],
                            "cube": [0.05, 10]}
        TEX = gstslice.generate_fcc_texture_v2(N=gstslice.n, distr='normal',
                                               validate_miso=False,
                                               calc_miso=True,
                                               tc_info_std_defaults={
                                                   'hw_phi1': 5,
                                                   'hw_Phi': 5,
                                                   'hw_phi2': 5,
                                                   'std_k_phi1': 3,
                                                   'std_k_Phi': 3,
                                                   'std_k_phi2': 3,
                                                   'perctol_phi1': 5,
                                                   'perctol_Phi': 5,
                                                   'perctol_phi2': 5},
                                               shuffle=False,
                                               nshuffles=2,
                                               ea_dtype=np.float32,
                                               miso_dtype=np.float16,
                                               id_dtype=np.int16,
                                               rand_ori_seed=np.random.random(),
                                               rand_ori_gen_rule='relaxed',
                                               n_tex_instances=1,
                                               n_sampling_instances=1)

        EXAMPLE - 1
        -----------
        """
        tiname = 'tex_instance.' + str(texture_instance)
        siname = 'ossi.' + str(sampling_instance)
        _si_ = 'sampling_instances'
        tc_ori_stacks = deepcopy(TEXdb[tiname][_si_][siname]['tc_ori_stacks'])
        # -----------------------
        N_ori_samples = TEXdb[tiname][_si_][siname]['nsamples']
        N_ori_samples['random'] = tc_ori_stacks['random'].shape[0]
        # -----------------------
        TCs = list(tc_ori_stacks.keys())
        """
        The inpjut 'tc_ori_stacks' can be obtained as below.
        TEXdb['tex_instance.1']['sampling_instances']['ossi.1']['tc_ori_stacks']
        """
        G = make_gid_net_from_neighlist(gstslice.neigh_gid)
        independent_set = networkx.maximal_independent_set(G)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #### START OF ORIENTATION MAPPING FOR ALL THE NT GRAINS.
        """
        Identifying a set of grains in which no two grasins neighbour each
        other.

        ntgrains: non-touching grains - neighbour order: 1
        """
        ntgrains = list(independent_set)
        """
        neigh_gis_neg = gstslice.crop_neigh_gid(neigh_gid=gstslice.neigh_gid, gids_to_crop=ntgrains)
        G = make_gid_net_from_neighlist(neigh_gis_neg)
        G_conn_comp = networkx.connected_components(G)
        components = [set(c) for c in G_conn_comp]
        len(components)


        SG = {1: {'d': [gstslice.neigh_gid],
                  'G': [make_gid_net_from_neighlist(gstslice.neigh_gid)],
                  'nt': []
                  }
              }
        SG[1]['nt'].append(networkx.maximal_independent_set(SG[1]['G'][0]))

        for i in np.arange(2, 4, dtype=np.int16):
            SG[i]['d'] = []
            SG[i]['G'] = []
            SG[i]['nt'] = []
            for djg_i in range(len(SG[1]['G'])):

                # djg: Dis-Joint Group
                _d_ = gstslice.crop_neigh_gid(SG[i-1]['d'][djg_i],
                                              SG[i-1]['nt'][djg_i]))
                _G_ = make_gid_net_from_neighlist(_d_)
                _mis_ = networkx.maximal_independent_set(_G_)

                SG[i]['d'].append()
                SG[i]['nt'].append()
                SG[i]['G'].append()

        """
        """
        Identifying TC data needed to start mapping from.
        """
        print(f"Generating {ntex_map_instances_L0} L0 texture mapping instances")
        # tcalloc: Tex Comp Allocation for O(1) grains
        tcalloc = {'L0Map.' + str(_+1): None
                       for _ in range(ntex_map_instances_L0)}
        for tmiidl0 in range(ntex_map_instances_L0):
            # tmiidl0: Texture Mapping Instance ID - level 0
            l0_inst_name = 'L0Map.'+str(tmiidl0+1)
            print(f"Texture mapping instance L0: {tmiidl0}/{ntex_map_instances_L0}")
            tcalloc[l0_inst_name] = {'base': gstslice.choose_TC(tiname=tiname,
                                                    siname=siname,
                                                    tc_ori_stacks=tc_ori_stacks,
                                                    ntgrains=ntgrains)}
        """
        NOTES
        -----
        A tcalloc value (ex: tcalloc['L0Map.1']), contains a key
        called 'base'. Its a dict having 'alloc_gids' as one of its keys. Its
        a dict with tcnames as keys. For each key, a numpy array of IDs are
        stored. These IDs are the ntgrain numbers which have allocated to the
        corresponding textuire componenets satigfying the volume fractions of
        each texture component. random texture component is also included.
        """
        print("Successfully completed generating:",
              f"->{ntex_map_instances_L0}<- texture mapping instances",
              f"using ->'{tiname}'<-, ->'{siname}'<- from TEX database.\n",
              40*"-", '\n')
        # ---------------------------------------------------------------------
        print(f"\nGenerating ->{ntex_map_instances_L1}<- L1 tex mapping",
              f"instances in each of the ->{ntex_map_instances_L0}<- L0 tex",
              "mapping instance")
        # ---------------------------------------------------------------------
        if shuffle_ori_stack_bf_L1Map:
            for k, v in tc_ori_stacks.items():
                np.random.shuffle(tc_ori_stacks[k])
        else:
            # Nothings needs to be done.
            pass
        # ---------------------------------------------------------------------
        # tmiidl0: Texture Mapping Instance ID - level 0
        l0_inst_names = ['L0Map.'+str(i)
                         for i in np.arange(1, ntex_map_instances_L0+1)]
        l1_inst_names = ['L1Map.'+str(i)
                         for i in np.arange(1, ntex_map_instances_L1+1)]
        print(80*'-', '\n')
        for i, l0_inst_name in enumerate(l0_inst_names, start=1):
            ntgcounts = tcalloc[l0_inst_name]['base']['alloc']
            ntgcounts = {tc: ntgcount for tc, ntgcount in zip(TCs, ntgcounts)}
            for j, l1_inst_name in enumerate(l1_inst_names, start=1):
                print(f"Texture mapping instance (L0, L1):",
                      f" ({i} of {ntex_map_instances_L0},",
                      f" {j} of {ntex_map_instances_L1}), i.e. ",
                      f"({l0_inst_name}, {l1_inst_name})")
                tc_ori_ids, tc_oris = {}, {}
                for tcname, tc_nori_counts in N_ori_samples.items():
                    tc_ori_ids[tcname] = np.random.choice(range(tc_nori_counts),
                                                  ntgcounts[tcname],
                                                  replace=False)
                    tc_oris[tcname] = tc_ori_stacks[tcname][tc_ori_ids[tcname]]
                tcalloc[l0_inst_name][l1_inst_name] = {'ori_ids': tc_ori_ids,
                                                           'ori': tc_oris}
        # ---------------------------------------------------------------------
        """
        NOTE on data structure use:
        # The follwing non-touching grain IDs:
        tcalloc['L0Map.1']['base']['alloc_gids']['copper']
        # will be allocated the following orientation ids:
        tcalloc['L0Map.1']['L1Map.1']['ori_ids']['copper']
        # which correspond to the following orientations
        tcalloc['L0Map.1']['L1Map.1']['ori']['copper']

        Please see the basic validatoins below.
        """
        # A quick set of validations
        val_results = []
        for l0_inst_name in l0_inst_names:
            for l1_inst_name in l1_inst_names:
                val = np.zeros(len(TCs), dtype=np.bool_)
                for i, tc in enumerate(TCs):
                    a = tcalloc[l0_inst_name]['base']['alloc_gids'][tc].shape[0]
                    b = tcalloc[l0_inst_name][l1_inst_name]['ori_ids'][tc].shape[0]
                    c = tcalloc[l0_inst_name][l1_inst_name]['ori'][tc].shape[0]
                    result = a==b==c
                    val_results.append(result)
                    val[i] = result
                if np.all(result):
                    print(f".. {l0_inst_name} <--> {l1_inst_name}",
                          "validation successful")
        if all(val_results):
            print("All L0-L1 ori mapping for ntgrains successful")
        #### END OF ORIENTATION MAPPING FOR ALL THE NT GRAINS.
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #### START MAP TEXTURE TO O(1) NEIGHBOURS OF NON-TOUCHING GRAINS
        _N_ori_samples_ = np.array(list(N_ori_samples.values()))
        _ntgcounts_ = np.array(list(ntgcounts.values()))
        _ntg_neg_counts_ = _N_ori_samples_-_ntgcounts_
        ntg_neg_counts = {tc: n for tc, n in zip(TCs,_ntg_neg_counts_)}
        _ntg_neg_counts_selfvf_ = _ntg_neg_counts_/_ntg_neg_counts_.sum()
        selfvf_soft_ids = np.argsort(_ntg_neg_counts_selfvf_)
        TCs_sorted = [TCs[i] for i in selfvf_soft_ids]
        # ---------------------------------------------------------------------


        tcalloc[l0_inst_name][l1_inst_name]['ori_ids']['copper']


        """
        ntg_neg_counts: Number of grains not belonging tothe ntgrains.
        TCs_sorted: lowest grain count tc through to highest grain count tc.
        """
        tcalloc_neg = {}
        for l0_inst_name in l0_inst_names:
            tcalloc_neg[l0_inst_name] = {}
            for l1_inst_name in l1_inst_names:
                tcalloc_neg[l0_inst_name][l1_inst_name] = {}


        gstlice.crop_neigh_gid(neigh_gid='O(1)', gids_to_crop=[1])

        ntg_ori_ids, ntgNEG_ori_ids, G, ntgL1 = {}, {}, {}, {}
        for l0_inst_name in l0_inst_names:
            ntg_ori_ids[l0_inst_name] = {}
            ntgNEG_ori_ids[l0_inst_name] = {}
            G[l0_inst_name] = {}
            ntgL1[l0_inst_name] = {}
            for l1_inst_name in l1_inst_names:
                # Set data of non-touching grain IDs
                ntg_ori_ids_set = np.array([], dtype=np.int32)
                for tc in TCs:
                    ntg_ori_ids_set = np.append(ntg_ori_ids_set,
                                                tcalloc[l0_inst_name][l1_inst_name]['ori_ids'][tc])
                ntg_ori_ids_set = set(list(ntg_ori_ids_set))
                ntg_ori_ids[l0_inst_name][l1_inst_name] = ntg_ori_ids_set
                # Set data of grain IDs other than non-touching grain IDs
                _ntgNEG_ori_ids_ = np.array(list(set(gstslice.gid) - ntg_ori_ids_set))
                ntgNEG_ori_ids[l0_inst_name][l1_inst_name] = _ntgNEG_ori_ids_
                _G_ = make_gid_net_from_neighlist(list(_ntgNEG_ori_ids_))
                _ntgL1_ = networkx.maximal_independent_set(_G_)
                G[l0_inst_name][l1_inst_name] = _G_
                ntgL1[l0_inst_name][l1_inst_name] = list(_ntgL1_)






        A = None
        B = None
        ntg_ori_ids[l0_inst_name][l1_inst_name][tc] = A
        ntgNEG_ori_ids[l0_inst_name][l1_inst_name][tc] = B


        tcalloc['L0Map.1']['L1Map.1']['ori_ids']['copper']

        _ntg_neg_counts_sorted = _ntg_neg_counts_[selfvf_soft_ids]

        a = _ntg_neg_counts_selfvf_[selfvf_soft_ids]/_ntg_neg_counts_selfvf_.max()
        a = np.append([0], a)
        rnum_ranges = np.vstack((a, np.roll(a, -1))).T[:-1]

        rnum_ranges

        RNUM = [0 for tc in TCs_sorted]
        lefts = rnum_ranges[:, 0]
        rights = rnum_ranges[:, 1]
        for i in range(_ntg_neg_counts_.sum()):
            r = np.random.random()
            for tcnum, lr in enumerate(rnum_ranges):
                if r >= lr[0] and r <= lr[1]:
                    RNUM[tcnum] += 1

        ntg_neg_counts = {k: v for k, v in zip(TCs, _ntg_neg_counts_)}

    def get_ks_rotations(self):
        """
        Returns the 24 Kurdjumov-Sachs (K-S) orientation relationship
        operators as 3x3 rotation matrices.
        """
        # Define a single, specific K-S variant
        r1 = Rotation.from_euler('Z', 45, degrees=True)
        r2 = Rotation.from_rotvec(np.deg2rad(54.7356) * np.array([-1, 1, 0]) / np.sqrt(2))
        g_ks = (r2 * r1).as_matrix() # Shape: (3, 3)

        # Generate the 24 cubic symmetry operators
        ops = []
        for p in permutations(range(3)):
            P = np.eye(3)[list(p)]
            for signs in product([-1, 1], repeat=3):
                S = P * np.array(signs)
                if np.linalg.det(S) == 1:
                    ops.append(S)

        sym_ops = np.array(ops) # Shape: (24, 3, 3)

        # Apply each symmetry operator to the K-S variant using matrix multiplication.
        # The @ operator automatically broadcasts the operation over the stack of 24 sym_ops.
        ks_variants = sym_ops @ g_ks # Shape: (24, 3, 3)

        return ks_variants

    def assign_pag_and_grain_orientations(self,
                                          clusters_dict,
                                          neigh_clid,
                                          orientation_pool,
                                          HAGB_threshold=15.0):
        """
        Assigns parent FCC orientations to clusters (PAGs) and final BCC
        orientations to individual grains.
        """
        print("Assigning parent FCC orientations to PAGs...")

        pag_orientations = {}
        cluster_ids = list(clusters_dict.keys())
        random.shuffle(cluster_ids) # Shuffle to avoid bias

        ks_rotations = self.get_ks_rotations()

        # --- Part A: Assign Parent FCC Orientations with HAGB constraint ---
        for i, cid in enumerate(cluster_ids):
            is_valid_orientation = False
            while not is_valid_orientation:
                # Pick a random candidate orientation from the pool
                candidate_idx = np.random.randint(len(orientation_pool))
                candidate_ea = orientation_pool[candidate_idx]

                # Check against assigned neighbors
                is_valid_orientation = True
                for neighbor_cid in neigh_clid.get(cid, []):
                    if neighbor_cid in pag_orientations:
                        neighbor_ea = pag_orientations[neighbor_cid]

                        # You would call your fast misorientation function here
                        mis_deg, _, _ = self.cubic_misorientation_old1(candidate_ea, neighbor_ea, degrees=True)
                        # mis_deg, _, _ = self.cubic_misorientation(candidate_ea, neighbor_ea)

                        if mis_deg < HAGB_threshold:
                            is_valid_orientation = False
                            break # Reject candidate and try another

            pag_orientations[cid] = candidate_ea
            if (i + 1) % 100 == 0:
                print(f"  ... assigned orientations to {i+1}/{len(cluster_ids)} PAGs")

        print("\nTransforming to BCC and assigning orientations to grains...")
        grain_orientations = {}

        # --- Part B: Generate and Assign BCC Variant Orientations ---
        for cid, parent_fcc_ea in pag_orientations.items():
            # Convert parent FCC Euler angles to a rotation matrix
            # You would call your euler-to-matrix function here
            parent_fcc_R = self.cubic_euler_bunge_to_matrix_v1(
                np.array([parent_fcc_ea[0]]),
                np.array([parent_fcc_ea[1]]),
                np.array([parent_fcc_ea[2]])
            )[0]

            # Generate the 24 BCC variant rotation matrices for this PAG
            bcc_variant_Rs = np.einsum('ijk,kl->ijl', ks_rotations, parent_fcc_R)

            # Assign a random variant to each grain within the PAG
            for gid in clusters_dict[cid]:
                random_variant_R = random.choice(bcc_variant_Rs)

                # Convert the BCC matrix back to Euler angles for storage
                # You would call your matrix-to-euler function here
                bcc_ea = self._matrix_to_euler_bunge(random_variant_R)
                grain_orientations[gid] = bcc_ea

        print("Orientation mapping complete.")
        return pag_orientations, grain_orientations

    def verify_hagb_constraint(self,
                               pag_orientations,
                               neigh_clid,
                               HAGB_threshold=15.0):
        """
        Verifies that all adjacent PAGs meet the HAGB misorientation constraint.

        Parameters
        ----------
        pag_orientations : dict
            Dictionary mapping PAG ID (cluster_id) to its parent FCC orientation.
        neigh_clid : dict
            Dictionary mapping PAG ID to its list of neighboring PAG IDs.
        HAGB_threshold : float
            The minimum required misorientation angle in degrees.

        Returns
        -------
        list
            A list of violation tuples. Each tuple contains
            (pag_id_1, pag_id_2, calculated_misorientation).
            An empty list signifies success.
        """
        violations = []
        checked_pairs = set()

        print(f"Verifying HAGB constraint (>= {HAGB_threshold}°)...")

        for pag_id, neighbors in neigh_clid.items():
            for neighbor_id in neighbors:
                # Create a unique key for the pair to avoid checking twice (e.g., 1-2 and 2-1)
                pair_key = tuple(sorted((pag_id, neighbor_id)))
                if pair_key in checked_pairs:
                    continue

                # Calculate the misorientation using your provided function
                mis_deg, _, _ = self.cubic_misorientation_old1(pag_orientations[pag_id],
                                                               pag_orientations[neighbor_id], degrees=True)

                # Check for a violation
                if mis_deg < HAGB_threshold:
                    violations.append((pag_id, neighbor_id, mis_deg))

                checked_pairs.add(pair_key)

        print("Verification complete.")
        return violations

    def plot_100_pole_figure(self, euler_angles_deg, title=""):
        """
        Creates a basic {100} pole figure scatter plot.

        Parameters
        ----------
        euler_angles_deg : ndarray
            An (N, 3) array of Bunge Euler angles in degrees.
        title : str, optional
            The title for the plot.
        """
        """{100} crystal directions definition, pole definition."""
        poles = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        """Convert Euler angles to rotation matrices using your function"""
        R_stack = self.cubic_euler_bunge_to_matrix_v1(euler_angles_deg[:, 0],
                                                      euler_angles_deg[:, 1],
                                                      euler_angles_deg[:, 2],
                                                      degrees=True)

        """Transform the poles into the sample coordinate system"""
        sample_dirs = np.einsum('nij,kj->nki', R_stack, poles).reshape(-1, 3)

        """Stereographic projection for plotting. Filter out points pointing 
        down (z < 0)"""
        sample_dirs = sample_dirs[sample_dirs[:, 2] >= 0]

        """Projection."""
        X = sample_dirs[:, 0] / (1 + sample_dirs[:, 2])
        Y = sample_dirs[:, 1] / (1 + sample_dirs[:, 2])

        fig, ax = plt.subplots(figsize=(6, 6))
        circle = plt.Circle((0, 0), 1, color='black', fill=False)
        ax.add_artist(circle)
        ax.scatter(X, Y, s=2, alpha=0.1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')
        ax.set_title(title, fontsize=14)
        plt.show()

    def plot_pag_ks_verification(self,
                                 pag_id_to_check,
                                 parent_fcc_ea,
                                 child_grain_eulers,
                                 title=""):
        """
        Creates a {100} pole figure to verify the K-S relationship for one PAG.

        Parameters
        ----------
        pag_id_to_check : int
            The ID of the PAG being verified.
        parent_fcc_ea : ndarray
            The (3,) array of Bunge Euler angles for the parent PAG.
        child_grain_eulers : ndarray
            An (N, 3) array of Bunge Euler angles for the grains within the PAG.
        title : str, optional
            The title for the plot.
        """
        # --- 1. Calculate the 24 ideal BCC variant orientations ---
        parent_fcc_R = self.cubic_euler_bunge_to_matrix_v1(np.array([parent_fcc_ea[0]]),
                                                           np.array([parent_fcc_ea[1]]),
                                                           np.array([parent_fcc_ea[2]])
                                                           )[0]

        ideal_bcc_variant_Rs = self.get_ks_rotations() @ parent_fcc_R

        ideal_bcc_eulers = np.array([self._matrix_to_euler_bunge(R, degrees=True)
                                     for R in ideal_bcc_variant_Rs])

        # --- 2. Plotting ---
        fig, ax = plt.subplots(figsize=(7, 7))
        circle = plt.Circle((0, 0), 1, color='black', fill=False, zorder=1)
        ax.add_artist(circle)

        # Plot the ACTUAL grain orientations as small blue dots
        print(child_grain_eulers, '\n', 10*'-')
        self.plot_poles(ax, child_grain_eulers,  color='cornflowerblue',
                        marker='.', s=50,  alpha=0.5, label='Actual Grains')

        # Plot the IDEAL variant orientations as large red 'X' markers
        print(ideal_bcc_eulers)
        self.plot_poles(ax, ideal_bcc_eulers,  color='red', marker='x',
                        s=100, alpha=1.0, label='Ideal K-S Variants')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')
        ax.set_title(title, fontsize=14)
        ax.legend()
        plt.show()

    def plot_pag_ks_verification_v1(self,
                                 pag_id_to_check,
                                 parent_fcc_ea,
                                 child_grain_eulers,
                                 title=""):
        """
        Creates a {100} pole figure to verify the K-S relationship for one PAG.
        """
        # --- 1. Calculate the 24 ideal BCC variant orientations ---
        parent_fcc_R = self.cubic_euler_bunge_to_matrix_v1(
            np.array([parent_fcc_ea[0]]),
            np.array([parent_fcc_ea[1]]),
            np.array([parent_fcc_ea[2]])
        )[0]

        ideal_bcc_variant_Rs = self.get_ks_rotations() @ parent_fcc_R

        # --- 2. Define the {100} poles for BCC ---
        poles = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # --- 3. Project poles for IDEAL variants ---
        # Transform poles by all 24 variant rotation matrices
        ideal_dirs = np.einsum('nij,kj->nki', ideal_bcc_variant_Rs, poles).reshape(-1, 3)
        # Filter for upper hemisphere
        ideal_dirs = ideal_dirs[ideal_dirs[:, 2] >= 0]
        # Stereographic projection
        ideal_X = ideal_dirs[:, 0] / (1 + ideal_dirs[:, 2])
        ideal_Y = ideal_dirs[:, 1] / (1 + ideal_dirs[:, 2])

        # --- 4. Project poles for ACTUAL grains ---
        # Convert all child grain Euler angles to rotation matrices at once
        actual_grain_Rs = self.cubic_euler_bunge_to_matrix_v1(
            child_grain_eulers[:, 0],
            child_grain_eulers[:, 1],
            child_grain_eulers[:, 2]
        )
        # Transform poles by all actual grain rotation matrices
        actual_dirs = np.einsum('nij,kj->nki', actual_grain_Rs, poles).reshape(-1, 3)
        # Filter for upper hemisphere
        actual_dirs = actual_dirs[actual_dirs[:, 2] >= 0]
        # Stereographic projection
        actual_X = actual_dirs[:, 0] / (1 + actual_dirs[:, 2])
        actual_Y = actual_dirs[:, 1] / (1 + actual_dirs[:, 2])

        # --- 5. Plotting ---
        fig, ax = plt.subplots(figsize=(7, 7))
        circle = plt.Circle((0, 0), 1, color='black', fill=False, zorder=1)
        ax.add_artist(circle)

        # Plot the ACTUAL grain orientations as small blue dots
        ax.scatter(actual_X, actual_Y, color='cornflowerblue', marker='.',
                   s=50, alpha=0.5, label='Actual Grains')

        # Plot the IDEAL variant orientations as large red 'X' markers
        ax.scatter(ideal_X, ideal_Y, color='red', marker='x',
                   s=100, alpha=1.0, label='Ideal K-S Variants')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')
        ax.set_title(title, fontsize=14)
        ax.legend()
        plt.show()

    def plot_pag_ks_verification_v2(self,
                                     pag_id_to_check,
                                     parent_fcc_ea,
                                     child_grain_eulers,
                                     title=""):
        """
        Creates a {100} pole figure to verify the K-S relationship for one PAG,
        including the parent FCC orientation.
        """
        # --- 1. Calculate ideal BCC variants and get parent rotation matrix ---
        parent_fcc_R = self.cubic_euler_bunge_to_matrix_v1(
            np.array([parent_fcc_ea[0]]),
            np.array([parent_fcc_ea[1]]),
            np.array([parent_fcc_ea[2]])
        )[0]

        ideal_bcc_variant_Rs = self.get_ks_rotations() @ parent_fcc_R

        # --- 2. Define the {100} crystal poles ---
        poles = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # --- 3. Project poles for IDEAL BCC VARIANTS ---
        ideal_dirs = np.einsum('nij,kj->nki', ideal_bcc_variant_Rs, poles).reshape(-1, 3)
        ideal_dirs = ideal_dirs[ideal_dirs[:, 2] >= 0]
        ideal_X = ideal_dirs[:, 0] / (1 + ideal_dirs[:, 2])
        ideal_Y = ideal_dirs[:, 1] / (1 + ideal_dirs[:, 2])

        # --- 4. Project poles for ACTUAL BCC GRAINS ---
        actual_grain_Rs = self.cubic_euler_bunge_to_matrix_v1(
            child_grain_eulers[:, 0],
            child_grain_eulers[:, 1],
            child_grain_eulers[:, 2]
        )
        # The line below has been corrected from --1 to -1
        actual_dirs = np.einsum('nij,kj->nki', actual_grain_Rs, poles).reshape(-1, 3)
        actual_dirs = actual_dirs[actual_dirs[:, 2] >= 0]
        actual_X = actual_dirs[:, 0] / (1 + actual_dirs[:, 2])
        actual_Y = actual_dirs[:, 1] / (1 + actual_dirs[:, 2])

        # --- 5. Project poles for the PARENT FCC ORIENTATION ---
        parent_dirs = parent_fcc_R @ poles.T
        parent_dirs = parent_dirs.T
        parent_dirs = parent_dirs[parent_dirs[:, 2] >= 0]
        parent_X = parent_dirs[:, 0] / (1 + parent_dirs[:, 2])
        parent_Y = parent_dirs[:, 1] / (1 + parent_dirs[:, 2])

        # --- 6. Plotting ---
        fig, ax = plt.subplots(figsize=(7, 7))
        circle = plt.Circle((0, 0), 1, color='black', fill=False, zorder=1)
        ax.add_artist(circle)

        ax.scatter(actual_X, actual_Y, color='cornflowerblue', marker='.',
                   s=50, alpha=1.0, label='Actual Grains (BCC)')
        ax.scatter(ideal_X, ideal_Y, color='red', marker='x',
                   s=50, alpha=1.0, label='Ideal K-S Variants (BCC)')
        ax.scatter(parent_X, parent_Y, color='green', marker='s',
                   s=50, alpha=1.0, edgecolor='black', label='Parent PAG (FCC)')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')
        ax.set_title(title, fontsize=14)
        ax.legend()
        plt.show()

    def plot_pag_ks_verification_v3(self,
                                    pag_ids_to_plot,
                                    pag_orientations,
                                    clusters_dict,
                                    grain_orientations,
                                    grid_dims=(2, 3)):
        """
        Creates a grid of {100} pole figures to verify the K-S relationship
        for multiple PAGs, including the parent FCC orientation for each.
        """
        rows, cols = grid_dims
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = np.atleast_1d(axes).flatten() # Ensure axes is always an iterable array

        # Get the K-S rotations once
        ks_rotations = self.get_ks_rotations()

        # Define the {100} crystal poles
        poles = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Loop through the axes and the PAG IDs to plot
        for i, ax in enumerate(axes):
            if i >= len(pag_ids_to_plot):
                ax.axis('off') # Turn off unused subplots if there are more axes than PAGs
                continue

            pag_id = pag_ids_to_plot[i]
            parent_fcc_ea = pag_orientations[pag_id]
            child_gids = clusters_dict[pag_id]
            child_grain_eulers = np.array([grain_orientations[gid] for gid in child_gids])

            # --- Calculations for this subplot ---
            # Parent FCC Rotation Matrix
            parent_fcc_R = self.cubic_euler_bunge_to_matrix_v1(
                np.array([parent_fcc_ea[0]]),
                np.array([parent_fcc_ea[1]]),
                np.array([parent_fcc_ea[2]])
            )[0]

            # Ideal BCC Variant Rotation Matrices
            ideal_bcc_variant_Rs = ks_rotations @ parent_fcc_R

            # Actual Child Grain Rotation Matrices
            actual_grain_Rs = self.cubic_euler_bunge_to_matrix_v1(
                child_grain_eulers[:, 0],
                child_grain_eulers[:, 1],
                child_grain_eulers[:, 2]
            )

            # --- Projections for plotting ---
            # Ideal BCC Variants
            ideal_dirs = np.einsum('nij,kj->nki', ideal_bcc_variant_Rs, poles).reshape(-1, 3)
            ideal_dirs = ideal_dirs[ideal_dirs[:, 2] >= 0]
            ideal_X = ideal_dirs[:, 0] / (1 + ideal_dirs[:, 2])
            ideal_Y = ideal_dirs[:, 1] / (1 + ideal_dirs[:, 2])

            # Actual BCC Grains
            actual_dirs = np.einsum('nij,kj->nki', actual_grain_Rs, poles).reshape(-1, 3)
            actual_dirs = actual_dirs[actual_dirs[:, 2] >= 0]
            actual_X = actual_dirs[:, 0] / (1 + actual_dirs[:, 2])
            actual_Y = actual_dirs[:, 1] / (1 + actual_dirs[:, 2])

            # Parent FCC Orientation
            parent_dirs = parent_fcc_R @ poles.T
            parent_dirs = parent_dirs.T
            parent_dirs = parent_dirs[parent_dirs[:, 2] >= 0]
            parent_X = parent_dirs[:, 0] / (1 + parent_dirs[:, 2])
            parent_Y = parent_dirs[:, 1] / (1 + parent_dirs[:, 2])

            # --- Plotting on the current subplot (ax) ---
            circle = plt.Circle((0, 0), 1, color='black', fill=False, zorder=1)
            ax.add_artist(circle)

            # Plot Actual Grains (blue dots)
            ax.scatter(actual_X, actual_Y, color='cornflowerblue', marker='.',
                       s=30, alpha=0.7, label='Actual Grains (BCC)')

            # Plot Ideal K-S Variants (red 'x's)
            ax.scatter(ideal_X, ideal_Y, color='red', marker='x',
                       s=50, alpha=1.0, label='Ideal K-S Variants (BCC)')

            # Plot Parent FCC Orientation (green squares)
            ax.scatter(parent_X, parent_Y, color='green', marker='s',
                       s=80, alpha=1.0, edgecolor='black', label='Parent PAG (FCC)') # Reduced size for clarity in subplots

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.axis('off')
            ax.set_title(f"PAG {pag_id}", fontsize=12)

        # Add a single legend for the entire figure
        # Collect handles/labels from one of the populated axes
        handles, labels = axes[0].get_legend_handles_labels() if len(pag_ids_to_plot) > 0 else ([], [])
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for the global legend
        plt.suptitle("K-S Verification for Multiple PAGs ({100} Pole Figures)", fontsize=16, y=1.0)
        plt.show()

    def plot_poles(self, ax, euler_angles, **kwargs):
        """Helper function to plot poles on a given axis."""
        poles = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        R_stack = self.cubic_euler_bunge_to_matrix_v1(euler_angles[:, 0],
                                                      euler_angles[:, 1],
                                                      euler_angles[:, 2],
                                                      degrees=True)

        sample_dirs = np.einsum('nij,kj->nki', R_stack, poles).reshape(-1, 3)
        sample_dirs = sample_dirs[sample_dirs[:, 2] >= 0]

        X = sample_dirs[:, 0] / (1 + sample_dirs[:, 2])
        Y = sample_dirs[:, 1] / (1 + sample_dirs[:, 2])

        ax.scatter(X, Y, **kwargs)

    def euler_to_ipf_color_old(self,
                           euler_angles_deg,
                           sample_direction=[0, 0, 1]):
        """
        Calculates the IPF-Z color for a single orientation in Bunge Euler angles.

        Parameters
        ----------
        euler_angles_deg : array-like
            A (3,) array of Bunge Euler angles (phi1, Phi, phi2) in degrees.
        euler_to_matrix_func : callable
            Reference to your function that converts Euler angles to a rotation matrix.

        Returns
        -------
        tuple
            An (R, G, B) tuple with values in the range [0, 1].
        """
        # The sample direction we are coloring with respect to (Z-axis)
        sample_direction = np.asarray(sample_direction, dtype=float)

        # Convert Euler angles to rotation matrix
        R = self.cubic_euler_bunge_to_matrix_v1(np.array([euler_angles_deg[0]]),
                                                np.array([euler_angles_deg[1]]),
                                                np.array([euler_angles_deg[2]]))[0]

        # Transform the sample direction into the crystal coordinate system
        # For IPF, we need the inverse rotation (transpose of the matrix)
        crystal_direction = R.T @ sample_direction

        # Bring the direction to the fundamental stereographic triangle [001]-[101]-[111]
        # by applying cubic symmetry operators (absolute values)
        v = np.abs(crystal_direction)
        v = v / np.linalg.norm(v) # Normalize

        # Standard IPF coloring scheme
        # R component is proportional to distance from [111] corner
        # G component is proportional to distance from [101] corner
        # B component is proportional to distance from [001] corner
        r = v[0] * (1.0 - v[2])
        g = v[1] * (1.0 - v[2])
        b = v[2]

        # Normalize the RGB values to a max of 1
        rgb_sum = r + g + b
        if rgb_sum > 0:
            r /= rgb_sum
            g /= rgb_sum
            b /= rgb_sum

        return (r, g, b)

    def euler_to_ipf_color(self, euler_angles_deg,
                           sample_direction=np.array([0, 0, 1])):
        """
        Calculates the standard IPF color for a single orientation. This is a
        robust version guaranteed to produce valid RGB values.
        """
        sample_direction = np.asarray(sample_direction, dtype=float)

        R = self.cubic_euler_bunge_to_matrix_v1(
            np.array([euler_angles_deg[0]]),
            np.array([euler_angles_deg[1]]),
            np.array([euler_angles_deg[2]])
        )[0]

        # Transform the sample direction into the crystal coordinate system
        crystal_direction = R.T @ sample_direction

        # Bring to the fundamental reference triangle by taking absolute values
        v = np.abs(crystal_direction)

        # Normalize the vector's components to have a max value of 1 for color brightness
        v_max = np.max(v)
        if v_max > 0:
            v /= v_max

        # The components of this vector in the fundamental triangle directly map to RGB
        # This ensures all values are in the range [0, 1]
        return tuple(v)

    def plot_ipf_map_pyvista_v1(self,
                                grain_orientations,
                                sample_direction=[0, 0, 1],
                                downsample_factor=1.0,
                                voxel_size=1.0,
                                opacity=0.8,
                                show_edges=False):
        """
        Creates and displays a 3D IPF map with solid voxels using PyVista.

        Parameters
        ----------
        grain_orientations : dict
            {grain_id: (3,) array of Bunge Euler angles}.
        downsample_factor : float, optional
            Factor to reduce the number of voxels for faster plotting (e.g., 0.1 for 10%).
        voxel_size : float, optional
            The side length of the cube used to represent each voxel.
        opacity : float, optional
            The opacity (alpha) of the rendered grains, from 0.0 to 1.0.
        """
        plotter = pv.Plotter(window_size=[1000, 1000])

        print("Generating PyVista meshes for 3D IPF map...")

        # Create a single cube glyph source to be used for all voxels
        glyph_source = pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)

        for i, (gid, coords) in enumerate(self.grain_locs.items()):
            if gid not in grain_orientations:
                continue

            # Optional downsampling for performance
            if downsample_factor < 1.0:
                num_points = int(len(coords) * downsample_factor)
                if num_points == 0 and len(coords) > 0:
                    num_points = 1
                indices = np.random.choice(len(coords), num_points, replace=False)
                coords = coords[indices]

            if coords.shape[0] == 0:
                continue

            # Get the orientation and calculate the IPF color for this grain
            orientation = grain_orientations[gid]
            color = self.euler_to_ipf_color(orientation, sample_direction)

            # Create a PyVista point cloud from the voxel coordinates
            point_cloud = pv.PolyData(coords)

            # Create a mesh of voxels by placing a cube glyph at each point
            voxels = point_cloud.glyph(geom=glyph_source, scale=False)

            # Add the colored voxel mesh to the plotter with opacity
            plotter.add_mesh(voxels,
                             color=color,
                             opacity=opacity,
                             show_edges=show_edges)

            if (i + 1) % 100 == 0:
                print(f"  ... processed {i+1}/{len(self.grain_locs)} grains")

        print("Rendering scene... This may take a moment for large datasets.")
        # plotter.enable_parallel_projection()
        plotter.show()

    def plot_ipf_map_pyvista_v2(self,
                               grain_orientations,
                               gids_to_plot=None,
                               sample_direction=[0, 0, 1],
                               downsample_factor=1.0,
                               voxel_size=1.0,
                               opacity=0.8,
                               show_edges=False):
        """
        Creates and displays a 3D IPF map for selected grains using PyVista.

        Parameters
        ----------
        grain_orientations : dict
            {grain_id: (3,) array of Bunge Euler angles}.
        gids_to_plot : list, tuple, or np.array, optional
            A collection of grain IDs to plot. If None or empty, all grains
            in self.grain_locs will be plotted. Default is None.
        downsample_factor : float, optional
            Factor to reduce the number of voxels for faster plotting.
        voxel_size : float, optional
            The side length of the cube used to represent each voxel.
        opacity : float, optional
            The opacity (alpha) of the rendered grains, from 0.0 to 1.0.
        show_edges : bool, optional
            If True, displays the edges of each voxel. Defaults to False.
        """
        # --- Input Handling for gids_to_plot ---
        if gids_to_plot is None or len(gids_to_plot) == 0:
            # If no specific gids are provided, plot all available grains
            gids_to_iterate = self.grain_locs.keys()
            print(f"Plotting all {len(gids_to_iterate)} grains...")
        else:
            # Otherwise, only plot the specified gids
            gids_to_iterate = gids_to_plot
            print(f"Plotting selected {len(gids_to_iterate)} grains...")

        plotter = pv.Plotter(window_size=[1000, 1000])
        glyph_source = pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)

        for i, gid in enumerate(gids_to_iterate):
            if gid not in self.grain_locs or gid not in grain_orientations:
                continue # Skip if gid has no location or orientation data

            coords = self.grain_locs[gid]

            if downsample_factor < 1.0:
                num_points = int(len(coords) * downsample_factor)
                if num_points == 0 and len(coords) > 0:
                    num_points = 1
                indices = np.random.choice(len(coords), num_points, replace=False)
                coords = coords[indices]

            if coords.shape[0] == 0:
                continue

            orientation = grain_orientations[gid]
            color = self.euler_to_ipf_color(orientation, sample_direction)

            point_cloud = pv.PolyData(coords)
            voxels = point_cloud.glyph(geom=glyph_source, scale=False)

            plotter.add_mesh(
                voxels,
                color=color,
                opacity=opacity,
                show_edges=show_edges
            )

        print("Rendering scene...")
        # Check if any meshes were added before trying to set camera
        plotter.camera.ParallelProjectionOff() # Set to perspective view
        plotter.show()

    def plot_pag_map_pyvista(self,
                             clusters_dict,
                             gids_to_plot=None,
                             downsample_factor=1.0,
                             voxel_size=1.0,
                             opacity=1.0,
                             show_edges=False):
        """
        Creates and displays a 3D PAG map using PyVista, coloring grains
        by their parent PAG ID.

        Parameters
        ----------
        clusters_dict : dict
            {pag_id: [list_of_grain_ids]}.
        gids_to_plot : list, optional
            A collection of grain IDs to plot. If None, all grains are plotted.
        downsample_factor : float, optional
            Factor to reduce the number of voxels for faster plotting.
        voxel_size : float, optional
            The side length of the cube used to represent each voxel.
        opacity : float, optional
            The opacity of the rendered grains.
        show_edges : bool, optional
            If True, displays the edges of each voxel.
        """
        plotter = pv.Plotter(window_size=[1000, 1000])

        # --- 1. Create a color map for the PAGs ---
        pag_ids = list(clusters_dict.keys())
        # Generate a distinct random color for each PAG
        pag_colors = {pag_id: np.random.rand(3) for pag_id in pag_ids}

        # --- 2. Determine which grains to plot ---
        if gids_to_plot is None or len(gids_to_plot) == 0:
            gids_to_iterate = self.grain_locs.keys()
        else:
            gids_to_iterate = gids_to_plot

        # Create a reverse map for quick lookup of a grain's PAG ID
        grain_to_pag_map = {gid: pag_id for pag_id, gids in clusters_dict.items() for gid in gids}

        print(f"Generating PyVista meshes for PAG map ({len(gids_to_iterate)} grains)...")

        glyph_source = pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)

        for i, gid in enumerate(gids_to_iterate):
            if gid not in self.grain_locs:
                continue

            # Find the parent PAG for this grain
            parent_pag_id = grain_to_pag_map.get(gid)
            if parent_pag_id is None:
                continue # Skip if grain doesn't belong to any PAG

            coords = self.grain_locs[gid]

            if downsample_factor < 1.0:
                num_points = int(len(coords) * downsample_factor)
                if num_points == 0 and len(coords) > 0:
                    num_points = 1
                indices = np.random.choice(len(coords), num_points, replace=False)
                coords = coords[indices]

            if coords.shape[0] == 0:
                continue

            # Get the color assigned to this grain's parent PAG
            color = pag_colors[parent_pag_id]

            point_cloud = pv.PolyData(coords)
            voxels = point_cloud.glyph(geom=glyph_source, scale=False)

            plotter.add_mesh(
                voxels,
                color=color,
                opacity=opacity,
                show_edges=show_edges
            )

        print("Rendering scene...")
        plotter.camera.ParallelProjectionOff() # Use perspective view
        plotter.show()

    def cluster_grains(self, neigh_gid, tcs=[1, 3, 4, 6, 7],
                       tcp=[0.05, 0.25, 0.50, 0.15, 0.05]):
        """
        Partitions an existing grain structure into new clusters using networkx
        for improved readability and maintenance.

        Parameters
        ----------
        neigh_gid : dict
            Dictionary with grain IDs as keys and lists of neigh IDs as values.
        tcs : list or tuple of int
            Target Cluster Sizes.
            A list of possible grain counts of clusters (e.g., [3, 4, 6]).
        tcp : list or tuple of float
            Target Cluster probanilities.
            The probability associated with each size. Must sum to 1.0.

        Returns
        -------
        dict
            A new dictionary of cluster IDs and the grain IDs they contain.

        Example
        -------
        neigh_gid = gstslice.neigh_gid
        target_sizes = [1, 3, 4, 6, 7]
        target_probs = [0.05, 0.25, 0.50, 0.15, 0.05]

        clusters = gstslice.cluster_grains(neigh_gid=neigh_gid,
                                           tcs=target_sizes,
                                           tcp=target_probs)
        """
        # ======================================
        print(40*'-', '\nIdentifying grain clusters to form packets')
        prob_sum = sum(tcp)
        if not np.isclose(prob_sum, 1.0):
            tcp = [p / prob_sum for p in tcp]

        G = make_gid_net_from_neighlist(neigh_gid)
        unassigned_grains = set(G.nodes())

        clusters = {}
        cluster_id_counter = 1

        while unassigned_grains:
            target_size = np.random.choice(tcs, p=tcp)
            seed_grain = random.choice(list(unassigned_grains))

            queue = collections.deque([seed_grain])
            visited_in_bfs = {seed_grain}
            new_cluster = []

            while queue and len(new_cluster) < target_size:
                current_grain = queue.popleft()

                if current_grain in unassigned_grains:
                    new_cluster.append(current_grain)
                    unassigned_grains.remove(current_grain)

                    for neighbor in G.neighbors(current_grain):
                        if neighbor in unassigned_grains and neighbor not in visited_in_bfs:
                            visited_in_bfs.add(neighbor)
                            queue.append(neighbor)

            if new_cluster:
                clusters[cluster_id_counter] = new_cluster
                cluster_id_counter += 1

        # 1. Get a list of the size of every cluster that was created
        actual_sizes = [len(grains) for grains in clusters.values()]

        # 2. Count the occurrences of each unique size
        size_counts = collections.Counter(actual_sizes)

        # 3. Calculate the probability (frequency) of each size
        total_clusters_formed = len(clusters)
        actual_distribution = {
            size: count / total_clusters_formed for size, count in size_counts.items()}

        """
        Scope for further development
            Now that actual_distribution has been found out, update the
            clusters to reduce the absolute difference in probabilities of top
            'N' tcs values. NOTE: The top 'N' tcs as the tcs which have the 'N'
            largest probabilites. The difference mentioned above is the
            difference in the user specified tcp and actual tcp as seen inside
            the variable actual_distribution. Then, include it as a user input
            in the definitio0n along with a flag variable which indicates,
            whether to perform this optimization at all. if needed also
            indicate the maximumnumber of iterations tyo limit to, in order to
            avoid consuming long time in while loops.
        """

        return {'clusters': clusters,
                'n': len(clusters),
                'actual_gid_counts': size_counts,
                'actual_distribution': actual_distribution}

    def generate_neigh_clid(self, neigh_gid,
                            tcs=[1, 3, 4, 6, 7],
                            tcp=[0.05, 0.25, 0.50, 0.15, 0.05]):
        """
        Generate neighbouring cluster ID adjacency informatio.
        Generates a cluster adjacency dictionary (neigh_clid) from grain clusters
        and a grain adjacency dictionary (neigh_gid).

        Parameters
        ----------
        clusters : dict
            A dictionary mapping {cluster_id: [list_of_grain_ids]}.
            This is the `clusters['clusters']` from your previous output.
        neigh_gid : dict
            The original dictionary mapping {grain_id: [list_of_neighbor_ids]}.

        Returns
        -------
        dict
            A new dictionary, neigh_clid, mapping {cluster_id: [list_of_neighbor_cluster_ids]}.
        """
        clusters = self.cluster_grains(neigh_gid, tcs=tcs, tcp=tcs)
        # 1. Create a fast reverse map from grain_id to cluster_id
        grain_to_cluster_map = {}
        for cluster_id, grains in clusters['clusters'].items():
            for grain_id in grains:
                grain_to_cluster_map[grain_id] = cluster_id

        # 2. Initialize the result dictionary with sets to handle duplicates
        neigh_clid = {cid: set() for cid in clusters['clusters'].keys()}

        # 3. Iterate through all grain connections to find cluster connections
        for grain_id, neighbors in neigh_gid.items():
            source_cluster_id = grain_to_cluster_map.get(grain_id)
            if source_cluster_id is None:
                continue

            for neighbor_gid in neighbors:
                neighbor_cluster_id = grain_to_cluster_map.get(neighbor_gid)

                # If the neighbor is in a different cluster, a connection exists
                if neighbor_cluster_id is not None and source_cluster_id != neighbor_cluster_id:
                    neigh_clid[source_cluster_id].add(neighbor_cluster_id)
                    # Since the relationship is mutual, add it the other way too
                    neigh_clid[neighbor_cluster_id].add(source_cluster_id)

        # 4. Convert the sets of neighbors to lists for the final output
        neigh_clid = {cid: sorted(list(neighbors)) for cid, neighbors in neigh_clid.items()}

        return clusters, neigh_clid

    def generate_neigh_clid_instances(self, neigh_gid,
                                      tcs=[1, 3, 4, 6, 7],
                                      tcp=[0.05, 0.25, 0.50, 0.15, 0.05],
                                      ninstances=1):
        cluster_sets = {clsid+1: None for clsid in range(ninstances)}
        for clsid in range(ninstances):
            clusters, neigh_clid = self.generate_neigh_clid(neigh_gid, tcs=tcs, tcp=tcp)
            cluster_sets[clsid+1] = {'clusters': clusters,
                                     'neigh_clid': neigh_clid}
        return cluster_sets

    def build_cluster_adjacency(cluster_id: Mapping[int, Iterable[int]],
                                neigh_gid: Mapping[int, Iterable[int]], *,
                                enforce_undirected: bool = True,
                                as_sorted_lists: bool = True,
                                ):
        """
        Construct a PAG-level (cluster-level) adjacency map.

        Parameters
        ----------
        cluster_id : Mapping[int, Iterable[int]]
            {cluster_id: iterable_of_grain_ids}. Each grain should belong to exactly one cluster.
        neigh_gid : Mapping[int, Iterable[int]]
            {grain_id: iterable_of_neighbor_grain_ids}. Interpreted as an undirected adjacency;
            will be symmetrized if `enforce_undirected=True`.

        enforce_undirected : bool, default True
            If True, symmetrize both the grain-level adjacency (neigh_gid) and the resulting
            cluster-level adjacency so that A ∈ N(B) ⇔ B ∈ N(A).

        as_sorted_lists : bool, default True
            If True, return neighbors as sorted lists. If False, return sets.

        Returns
        -------
        cluster_adj : dict[int, list[int] | set[int]]
            {cluster_id: neighbors}, where neighbors are the IDs of adjacent clusters
            (i.e., there exists at least one grain in cluster A that is a neighbor
            of at least one grain in cluster B, B≠A).

        Notes
        -----
        - Self-adjacency is never added.
        - Grains present in `neigh_gid` but not found in `cluster_id` are ignored.
        - A cluster will appear with an empty neighbor list/set if it has no inter-cluster contacts
          (possible in degenerate cases, e.g., a single cluster covering all grains, or if the
          provided `cluster_id`/`neigh_gid` are inconsistent).
        """
        # --- sanitize: grain -> set(neigh)
        G = {g: set(vs) for g, vs in neigh_gid.items()}
        for g, vs in G.items():
            vs.discard(g)  # drop self-loops

        # symmetrize grain graph if requested
        if enforce_undirected:
            for u, vs in list(G.items()):
                for v in list(vs):
                    G.setdefault(v, set()).add(u)

        # map grain -> cluster
        grain_to_cluster: dict[int, int] = {}
        for cid, grains in cluster_id.items():
            for g in grains:
                grain_to_cluster[g] = cid

        # initialize cluster adjacency containers
        cluster_adj: dict[int, set[int]] = {cid: set() for cid in cluster_id.keys()}

        # build cluster edges by scanning inter-grain contacts
        # Using an edge set prevents double-work and ensures no self-edges.
        cluster_edges: set[tuple[int, int]] = set()

        for cid, grains in cluster_id.items():
            for g in grains:
                # neighbors of grain g
                for h in G.get(g, ()):
                    c2 = grain_to_cluster.get(h)
                    if c2 is None or c2 == cid:
                        continue  # ignore: unknown grain or same-cluster neighbor
                    # record undirected cluster edge
                    if enforce_undirected:
                        a, b = (cid, c2) if cid <= c2 else (c2, cid)
                        cluster_edges.add((a, b))
                    else:
                        cluster_edges.add((cid, c2))

        # populate adjacency from edge set
        if enforce_undirected:
            for a, b in cluster_edges:
                cluster_adj[a].add(b)
                cluster_adj[b].add(a)
        else:
            for a, b in cluster_edges:
                cluster_adj[a].add(b)

        # format output
        if as_sorted_lists:
            return {cid: sorted(nbs) for cid, nbs in cluster_adj.items()}
        else:
            return cluster_adj


    def slice_packet_into_blocks(self,
                                 packet_gid,
                                 cluster_id,
                                 voxel_coords,
                                 block_thickness,
                                 slicing_plane,
                                 global_block_id_start=0):
        """
        Slices a single packet into multiple, contiguous blocks.

        This method uses a hybrid geometric and connectivity-based approach to
        partition the voxel cloud of a single packet into smaller, contiguous
        domains representing martensitic blocks.

        Parameters
        ----------
        packet_gid : int
            The ID of the parent packet (original grain ID).
        cluster_id : int
            The ID of the parent cluster (PAG).
        voxel_coords : ndarray
            An (N, 3) array of voxel coordinates for this packet.
        block_thickness : float
            The desired thickness of the blocks in the same units as the coords.
        slicing_plane : Plane
            A Plane object from the Plane class, defining the orientation of the slices.
        global_block_id_start : int
            The starting number for the globally unique block IDs.

        Returns
        -------
        tuple
            - dict: A dictionary of new blocks: {block_id: voxel_array}.
            - int: The next available global block ID after this operation.

        Algorithm
        ---------
        1.  Define Slicing Planes: The spatial extent of the packet's voxels
            is calculated along the normal of the `slicing_plane`. A stack of
            parallel planes is then generated to span this entire extent, with
            the spacing between planes equal to `block_thickness`.

        2.  Iterate Through Slabs: The function loops through each "slab,"
            which is the region between two adjacent planes in the stack.

        3.  Isolate & Find Connected Components: For each slab, all voxels
            within its boundaries are isolated. To find contiguous groups
            within this subset, a temporary 3D boolean grid is created. The
            `scipy.ndimage.label` function is then used on this grid to find
            and number all separate, contiguous groups of voxels. Each group
            represents a block.

        4.  Assign Block IDs and Store: Each component identified by `label`
            is assigned a unique block ID according to the convention
            'B_CLUSTERID_PACKETID_LOCALID'. The new block and its voxel
            coordinates are stored.

        5.  Cleanup (Future Scope): A final step can be added after the loop
            to assign any unassigned voxels (due to edge cases) to their
            nearest block, ensuring a complete partition.
        """
        if len(voxel_coords) < 1:
            return {}, global_block_id_start

        unit_n = slicing_plane.unit_normal
        distances = np.dot(voxel_coords - slicing_plane.point, unit_n)
        min_dist, max_dist = np.min(distances), np.max(distances)

        num_slices = int(np.ceil((max_dist - min_dist) / block_thickness))
        if num_slices == 0:
            num_slices = 1

        newly_created_blocks = {}
        local_block_counter = 1
        global_block_id = global_block_id_start

        for i in range(num_slices):
            slab_min = min_dist + i * block_thickness
            slab_max = slab_min + block_thickness

            slab_mask = (distances >= slab_min) & (distances < slab_max)
            slab_voxels = voxel_coords[slab_mask]

            if len(slab_voxels) == 0:
                continue

            min_coords = np.min(slab_voxels, axis=0)
            grid_coords = np.round((slab_voxels - min_coords)).astype(int)

            grid_shape = np.max(grid_coords, axis=0) + 1
            grid_array = np.zeros(grid_shape, dtype=bool)
            grid_array[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]] = True

            labeled_array, num_features = spndimg_label(grid_array)

            for j in range(1, num_features + 1):
                block_name = f'B_{cluster_id}_{packet_gid}_{local_block_counter}'

                component_grid_coords = np.argwhere(labeled_array == j)
                component_voxels = component_grid_coords + min_coords

                newly_created_blocks[block_name] = component_voxels
                local_block_counter += 1
                global_block_id += 1

        return newly_created_blocks, global_block_id

    def test_single_packet_slicing(self, clset, test_packet_id=None):
        """
        Runs the block generation and visualization for a single packet.
        """
        # --- 1. SETUP ---
        clusters_dict = clset['clusters']['clusters']

        # If no specific packet ID is given, pick one at random
        if test_packet_id is None:
            # Create a list of all gids that are considered packets
            all_packet_ids = [gid for gids_in_pag in clusters_dict.values() for gid in gids_in_pag]
            if not all_packet_ids:
                print("No packets to test.")
                return
            test_packet_id = random.choice(all_packet_ids)

        print(f"--- Testing Block Generation for Packet ID: {test_packet_id} ---")

        # --- 2. GATHER INPUTS ---
        # Find the parent Cluster (PAG) ID for our test packet
        grain_to_pag_map = {gid: pag_id for pag_id, gids in clusters_dict.items() for gid in gids}
        parent_cluster_id = grain_to_pag_map.get(test_packet_id)
        if parent_cluster_id is None:
            print(f"Error: Could not find parent cluster for packet {test_packet_id}")
            return

        voxel_coords = self.grain_locs[test_packet_id]

        # Define a desired block thickness (in your data's units)
        block_thickness = 4.0 # Example value, adjust as needed

        # Define a simple slicing plane (e.g., perpendicular to Z-axis)
        # passing through the centroid of the packet
        packet_centroid = np.mean(voxel_coords, axis=0)
        slicing_plane = Plane(point=packet_centroid, normal=[0, 0, 1])

        # --- 3. EXECUTE THE FUNCTION ---
        newly_created_blocks, _ = self.slice_packet_into_blocks(
            packet_gid=test_packet_id,
            cluster_id=parent_cluster_id,
            voxel_coords=voxel_coords,
            block_thickness=block_thickness,
            slicing_plane=slicing_plane
        )

        # --- 4. ANALYZE THE RESULTS ---
        print(f"\nAnalysis Results:")
        print(f"  - Number of blocks created: {len(newly_created_blocks)}")
        if newly_created_blocks:
            print(f"  - Block names: {list(newly_created_blocks.keys())}")

            # Verification: Check if any voxels were lost
            original_voxel_count = len(voxel_coords)
            new_voxel_count = sum(len(voxels) for voxels in newly_created_blocks.values())

            print(f"  - Original voxel count: {original_voxel_count}")
            print(f"  - Total voxels in new blocks: {new_voxel_count}")
            if original_voxel_count == new_voxel_count:
                print("  - Voxel count verification: SUCCESS")
            else:
                print("  - Voxel count verification: FAILED")

        # --- 5. VISUALIZE THE BLOCKS ---
        if newly_created_blocks:
            print("\nLaunching PyVista visualization...")
            plotter = pv.Plotter(window_size=[1000, 1000])
            glyph_source = pv.Cube(x_length=0.95, y_length=0.95, z_length=0.95)

            for block_id, block_voxels in newly_created_blocks.items():
                point_cloud = pv.PolyData(block_voxels)
                voxels_mesh = point_cloud.glyph(geom=glyph_source, scale=False)
                # Assign a random color to each block
                plotter.add_mesh(voxels_mesh, color=np.random.rand(3), opacity=0.9)

            plotter.add_text(f"Blocks in Packet {test_packet_id}", font_size=15)
            plotter.camera.ParallelProjectionOff()
            plotter.show()

    def generate_all_packet_slicing_options(self, clusters_dict, pag_orientations):
        """
        Generates and stores all four possible slicing planes for every packet.

        This function iterates through all packets, finds their parent PAG, and
        calls a helper to calculate the four crystallographically-derived
        slicing planes.

        Returns
        -------
        dict
            A dictionary where each key is a packet_gid and the value is a
            list of four possible Plane objects for slicing.
            {packet_gid: [Plane1, Plane2, Plane3, Plane4]}
        """
        packet_slicing_options = {}

        # Create a reverse map to quickly find a packet's parent PAG
        packet_to_pag_map = {gid: pag_id for pag_id, gids in clusters_dict.items() for gid in gids}

        # Get a unique list of all packet IDs
        all_packet_ids = list(packet_to_pag_map.keys())

        print(f"Calculating slicing plane options for {len(all_packet_ids)} packets...")
        for packet_gid in all_packet_ids:
            pag_id = packet_to_pag_map[packet_gid]

            # Call the dedicated function for a single packet
            packet_slicing_options[packet_gid] = self.get_slicing_planes_for_packet(
                pag_id=pag_id,
                pag_orientations=pag_orientations
            )

        return packet_slicing_options

    def get_slicing_planes_for_packet(self, pag_id, pag_orientations):
        """
        Calculates the four possible crystallographic slicing planes for a packet
        based on its parent PAG's orientation.
        """
        # Get the Parent PAG's Orientation Matrix
        parent_fcc_ea = pag_orientations[pag_id]
        parent_fcc_R = self.cubic_euler_bunge_to_matrix_v1(
            np.array([parent_fcc_ea[0]]),
            np.array([parent_fcc_ea[1]]),
            np.array([parent_fcc_ea[2]])
        )[0]

        # Define the four {111} plane normals in the crystal frame
        crystal_frame_normals = np.array([
            [1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]
        ]) / np.sqrt(3)

        # Transform these normals into the global sample frame
        sample_frame_normals = (parent_fcc_R @ crystal_frame_normals.T).T

        # Create a list of four Plane objects from these normals
        possible_planes = []
        for normal in sample_frame_normals:
            # The point is arbitrary for now; it serves as a template
            possible_planes.append(Plane(point=[0, 0, 0], normal=normal))

        return possible_planes

    def generate_blocks_for_packet(self,
                                     packet_gid,
                                     clusters_dict,
                                     pag_orientations,
                                     block_thickness):
        """
        Generates a dictionary of blocks for a single specified packet.

        ... (Parameters section remains the same) ...

        Returns
        -------
        tuple:
            - dict: A dictionary of new blocks: {block_id: voxel_array}.
            - Plane: The Plane object that was used for slicing.
        """
        # --- 1. Find the Parent PAG and its Data ---
        try:
            parent_cluster_id = next(p_id for p_id, gids in clusters_dict.items() if packet_gid in gids)
            voxel_coords = self.grain_locs[packet_gid]
        except (StopIteration, KeyError):
            print(f"Error: Could not find data for packet ID {packet_gid}")
            return {}, None # Return None for the plane on error

        # --- 2. Determine the Slicing Plane ---
        possible_slicing_planes = self.get_slicing_planes_for_packet(
            pag_id=parent_cluster_id,
            pag_orientations=pag_orientations
        )

        chosen_plane_template = random.choice(possible_slicing_planes)

        packet_centroid = np.mean(voxel_coords, axis=0)
        slicing_plane = Plane(point=packet_centroid, normal=chosen_plane_template.normal)

        # --- 3. Execute the Slicing Function ---
        newly_created_blocks, _ = self.slice_packet_into_blocks(
            packet_gid=packet_gid,
            cluster_id=parent_cluster_id,
            voxel_coords=voxel_coords,
            block_thickness=block_thickness,
            slicing_plane=slicing_plane
        )

        # --- 4. Return both the blocks and the plane used ---
        return newly_created_blocks, slicing_plane

    def visualize_blocks_in_packet(self,
                                     blocks_dict,
                                     packet_gid=None,
                                     voxel_size=1.0,
                                     opacity=1.0):
        """
        Visualizes a dictionary of blocks using PyVista.
        """
        if not blocks_dict:
            print("Block dictionary is empty. Nothing to visualize.")
            return

        plotter = pv.Plotter(window_size=[1000, 1000])
        glyph_source = pv.Cube(x_length=voxel_size * 0.95, y_length=voxel_size * 0.95, z_length=voxel_size * 0.95)

        print(f"Visualizing {len(blocks_dict)} blocks...")
        for block_id, block_voxels in blocks_dict.items():
            point_cloud = pv.PolyData(block_voxels)
            voxels_mesh = point_cloud.glyph(geom=glyph_source, scale=False)
            plotter.add_mesh(voxels_mesh, color=np.random.rand(3), opacity=opacity)

        if packet_gid:
            plotter.add_text(f"Generated Blocks in Packet {packet_gid}", font_size=15)

        plotter.camera.ParallelProjectionOff()
        plotter.show()

    def assign_orientations_to_blocks_old(self,
                                        pag_id,
                                        blocks_dict,
                                        pag_orientations):
        """
        Assigns a physically realistic BCC orientation to each block in a dict
        and runs a verification check on the result.
        """
        # --- 1. Get Parent Data and All 24 Variants ---
        parent_fcc_ea = pag_orientations[pag_id]

        # --- FIX STARTS HERE ---
        # Explicitly unpack the Euler angle tuple into three variables first
        phi1, Phi, phi2 = parent_fcc_ea

        # Now call the vectorized function with single-element arrays
        parent_fcc_R = self.cubic_euler_bunge_to_matrix_v1(
            np.array([phi1]),
            np.array([Phi]),
            np.array([phi2])
        )[0]
        # --- FIX ENDS HERE ---

        ks_rotations = self.get_ks_rotations()
        bcc_variant_Rs = ks_rotations @ parent_fcc_R

        # --- 2. Group the 24 Variants into their 4 Packet Groups ---
        packet_groups = self._group_ks_variants_into_packets(bcc_variant_Rs)

        # --- 3. Select One Packet Group for This Entire Set of Blocks ---
        chosen_packet_group_variants = random.choice(list(packet_groups.values()))

        # --- 4. Assign an Orientation from the Chosen Group to Each Block ---
        block_orientations = {}
        for block_id in blocks_dict.keys():
            chosen_variant_R = random.choice(chosen_packet_group_variants)
            block_orientations[block_id] = self._matrix_to_euler_bunge(chosen_variant_R)

        # --- 5. Automatically run verification ---
        self._verify_block_orientations(block_orientations)

        return block_orientations

    def assign_orientations_to_blocks(self,
                                        pag_id,
                                        blocks_dict,
                                        pag_orientations):
        """
        Assigns a physically realistic BCC orientation to each block in a dict.
        """
        # --- 1. Get Parent Data and All 24 Variants ---
        parent_fcc_ea = pag_orientations[pag_id]
        parent_fcc_R = self.cubic_euler_bunge_to_matrix_v1(
            *np.array(parent_fcc_ea).reshape(3, 1)
        )[0]

        ks_rotations = self.get_ks_rotations()
        bcc_variant_Rs = ks_rotations @ parent_fcc_R

        # --- 2. Group the 24 Variants into their 4 Packet Groups ---
        packet_groups = self._group_ks_variants_into_packets(bcc_variant_Rs)

        # --- 3. Robustly Select One Packet Group ---
        # --- FIX STARTS HERE ---
        # Filter out any packet groups that may have ended up empty
        non_empty_groups = [group for group in packet_groups.values() if group]

        if not non_empty_groups:
            # This is an unlikely but critical error state
            print(f"Warning: No valid packet groups found for PAG {pag_id}. Cannot assign orientations.")
            return {}

        # Randomly choose from ONLY the non-empty groups
        chosen_packet_group_variants = random.choice(non_empty_groups)
        # --- FIX ENDS HERE ---

        # --- 4. Assign an Orientation from the Chosen Group to Each Block ---
        block_orientations = {}
        for block_id in blocks_dict.keys():
            chosen_variant_R = random.choice(chosen_packet_group_variants)
            block_orientations[block_id] = self._matrix_to_euler_bunge(chosen_variant_R)

        # --- 5. Automatically run verification ---
        self._verify_block_orientations(block_orientations)

        return block_orientations

    def _verify_block_orientations(self, block_orientations):
        """
        Picks two random blocks and prints their misorientation to verify
        that it is a physically plausible value.
        """
        print("\n--- Running Block Orientation Verification ---")
        if len(block_orientations) < 2:
            print("  - Not enough blocks to compare.")
            return

        # Get orientations for two random blocks
        random_block_ids = random.sample(list(block_orientations.keys()), 2)
        ori1 = block_orientations[random_block_ids[0]]
        ori2 = block_orientations[random_block_ids[1]]

        # Calculate the misorientation
        mis_deg, _, _ = self.cubic_misorientation_old1(ori1, ori2, unique_tol_deg=1e-4, degrees=True)

        print(f"  - Misorientation between block '{random_block_ids[0]}' and '{random_block_ids[1]}': {mis_deg:.2f}°")
        print("  - Expected: A specific high-angle value (e.g., ~60°) or 0°.")

    def _group_ks_variants_into_packets(self, variant_Rs):
        """
        Groups the 24 K-S variants into 4 packet groups.

        Each packet is defined by a common {110}_bcc || {111}_fcc plane.
        """
        fcc_111_planes = np.array([
            [1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]
        ]) / np.sqrt(3)

        bcc_110_plane = np.array([1, 1, 0]) / np.sqrt(2)

        packet_groups = {0: [], 1: [], 2: [], 3: []}

        for variant_R in variant_Rs:
            # Transform the BCC {110} plane by this variant's orientation
            # to see what it corresponds to in the FCC frame
            fcc_equivalent_plane = variant_R.T @ bcc_110_plane

            # Find which of the four {111}_fcc planes it is parallel to
            # (dot product will be close to +/- 1)
            dot_products = np.abs(np.dot(fcc_111_planes, fcc_equivalent_plane))

            packet_index = np.argmax(dot_products)
            packet_groups[packet_index].append(variant_R)

        return packet_groups

    def visualize_blocks_ipf_map(self,
                                 blocks_dict,
                                 block_orientations,
                                 packet_gid=None,
                                 voxel_size=1.0,
                                 opacity=1.0,
                                 show_edges=False,
                                 sample_direction=[0, 0, 1]):
        """
        Visualizes the IPF map of all blocks within a single packet.

        Parameters
        ----------
        blocks_dict : dict
            Dictionary of blocks for a single packet: {block_id: voxel_array}.
        block_orientations : dict
            Dictionary mapping each block_id to its (phi1, Phi, phi2) Euler angles.
        packet_gid : int, optional
            The ID of the parent packet, used for the plot title.
        voxel_size : float, optional
            The side length of the cube used to represent each voxel.
        opacity : float, optional
            The opacity of the rendered blocks.
        show_edges : bool, optional
            If True, displays the edges of each voxel.
        sample_direction : list, optional
            The sample direction for the IPF coloring (e.g., [0,0,1] for Z).

        NOTES:
            This plot reveals the internal crystallographic structure of a
            single packet. Since all blocks within a packet are assigned
            variants from the same K-S group, their orientations are similar
            but not identical.

            Expect to see the packet colored in closely related shades. For
            example, you might see several blocks colored in different shades
            of blue and purple, but you would not expect to see a bright red
            block right next to a bright green one within the same packet.
            This visual clustering of colors confirms the physical model is
            working correctly.
        """
        if not blocks_dict:
            print("Block dictionary is empty. Nothing to visualize.")
            return

        plotter = pv.Plotter(window_size=[1000, 1000])
        glyph_source = pv.Cube(x_length=voxel_size * 0.95, y_length=voxel_size * 0.95, z_length=voxel_size * 0.95)

        print(f"Visualizing IPF map for {len(blocks_dict)} blocks...")
        for block_id, block_voxels in blocks_dict.items():
            if block_id not in block_orientations:
                continue

            # Get the orientation for this specific block
            orientation = block_orientations[block_id]

            # Calculate the IPF color for this block's orientation
            color = self.euler_to_ipf_color(orientation, sample_direction=sample_direction)

            point_cloud = pv.PolyData(block_voxels)
            voxels_mesh = point_cloud.glyph(geom=glyph_source, scale=False)

            plotter.add_mesh(
                voxels_mesh,
                color=color,
                opacity=opacity,
                show_edges=show_edges
            )

        if packet_gid:
            plotter.add_text(f"Block IPF Map for Packet {packet_gid}", font_size=15)

        plotter.camera.ParallelProjectionOff()
        plotter.show()

    def _process_single_packet(self,
                               packet_gid,
                               clusters_dict,
                               packet_slicing_options,
                               pag_orientations,
                               block_thickness):
        """
        Processes a single packet to generate and orient its blocks.
        This is a helper function for the main orchestrator.
        """
        try:
            pag_id = next(p_id for p_id, gids in clusters_dict.items() if packet_gid in gids)
            voxel_coords = self.grain_locs[packet_gid]
        except (StopIteration, KeyError):
            return None, None # Return None if data is missing

        # 1. Select a slicing plane for this run
        chosen_slicing_plane = random.choice(packet_slicing_options[packet_gid])

        # 2. Generate the blocks for this packet
        blocks_for_this_packet, _ = self.slice_packet_into_blocks(
            packet_gid=packet_gid,
            cluster_id=pag_id,
            voxel_coords=voxel_coords,
            block_thickness=block_thickness,
            slicing_plane=chosen_slicing_plane
        )

        if not blocks_for_this_packet:
            return None, None

        # 3. Assign orientations to the newly created blocks
        orientations_for_these_blocks = self.assign_orientations_to_blocks(
            pag_id=pag_id,
            blocks_dict=blocks_for_this_packet,
            pag_orientations=pag_orientations
        )

        return blocks_for_this_packet, orientations_for_these_blocks

    def generate_and_orient_all_blocks(self,
                                       clset,
                                       pag_orientations,
                                       block_thickness):
        """
        Main orchestrator to generate and assign orientations to all blocks
        in an entire microstructure.
        """
        clusters_dict = clset['clusters']['clusters']

        print("Step 1/3: Calculating all possible slicing planes...")
        packet_slicing_options = self.generate_all_packet_slicing_options(
            clusters_dict=clusters_dict,
            pag_orientations=pag_orientations
        )

        all_blocks_dict = {}
        packet_to_blocks_map = {}
        all_block_orientations = {}

        all_packet_ids = list(packet_slicing_options.keys())
        print(f"Step 2/3: Generating blocks for {len(all_packet_ids)} packets...")

        for i, packet_gid in enumerate(all_packet_ids):
            # --- Call the new, dedicated helper function ---
            blocks, orientations = self._process_single_packet(
                packet_gid=packet_gid,
                clusters_dict=clusters_dict,
                packet_slicing_options=packet_slicing_options,
                pag_orientations=pag_orientations,
                block_thickness=block_thickness
            )

            # Aggregate the results
            if blocks and orientations:
                all_blocks_dict.update(blocks)
                packet_to_blocks_map[packet_gid] = list(blocks.keys())
                all_block_orientations.update(orientations)

            if (i + 1) % 100 == 0:
                print(f"  ... processed {i+1}/{len(all_packet_ids)} packets")

        print("Step 3/3: Block generation and orientation mapping complete.")
        return all_blocks_dict, packet_to_blocks_map, all_block_orientations

    def visualize_block_morphology(self,
                                   blocks_dict,
                                   title="Block Morphology",
                                   voxel_size=1.0,
                                   opacity=1.0,
                                   cmap='nipy_spectral'
                                   ):
        """
        Visualizes a dictionary of blocks using random colors to show morphology.
        """
        if not blocks_dict:
            print("Block dictionary is empty. Nothing to visualize.")
            return

        plotter = pv.Plotter(window_size=[1000, 1000])
        glyph_source = pv.Cube(x_length=voxel_size * 0.95, y_length=voxel_size * 0.95, z_length=voxel_size * 0.95)

        total_blocks = len(blocks_dict)
        print(f"Visualizing morphology of {total_blocks} blocks...")

        for i, (block_id, block_voxels) in enumerate(blocks_dict.items()):
            point_cloud = pv.PolyData(block_voxels)
            voxels_mesh = point_cloud.glyph(geom=glyph_source, scale=False)
            plotter.add_mesh(voxels_mesh, color=np.random.rand(3), opacity=opacity,
                             cmap=cmap)

            # Print a progress update every 20 blocks
            if (i + 1) % 20 == 0 or (i + 1) == total_blocks:
                print(f"  ... processed {i + 1}/{total_blocks} blocks")

        plotter.add_text(title, font_size=15)
        plotter.camera.ParallelProjectionOff()
        print("Rendering scene...")
        plotter.show()

    def visualize_block_morphology_v1(self,
                                   blocks_dict,
                                   title="Block Morphology",
                                   voxel_size=1.0,
                                   opacity=1.0,
                                   cmap='nipy_spectral'
                                   ):
        """
        Visualizes a dictionary of blocks using integer Block IDs as scalar data
        to show morphology and includes a color bar.
        """
        if not blocks_dict:
            print("Block dictionary is empty. Nothing to visualize.")
            return

        total_blocks = len(blocks_dict)
        print(f"Visualizing morphology of {total_blocks} blocks...")

        # --- 1. Batch Data Preparation (Necessary for Color Bar) ---

        # We must merge all blocks into a single mesh and use integer IDs as scalars.
        all_block_coords = []
        block_ids_scalar = []

        # We start the block IDs for the scalar array at 1
        current_id = 1

        for i, (block_id, block_voxels) in enumerate(blocks_dict.items()):
            if block_voxels.size == 0:
                continue

            all_block_coords.append(block_voxels)
            block_ids_scalar.extend([current_id] * block_voxels.shape[0])
            current_id += 1 # Increment the unique integer ID

            # Print a progress update every 20 blocks
            if (i + 1) % 20 == 0 or (i + 1) == total_blocks:
                print(f"  ... batched data for {i + 1}/{total_blocks} blocks")

        if not all_block_coords:
            print("No voxel data found to plot.")
            return

        # Consolidate into single arrays
        final_coords_array = np.concatenate(all_block_coords, axis=0)
        final_scalar_array = np.array(block_ids_scalar, dtype=np.int32)

        # --- 2. PyVista Setup ---
        plotter = pv.Plotter(window_size=[1000, 1000])
        glyph_source = pv.Cube(x_length=voxel_size * 0.95, y_length=voxel_size * 0.95, z_length=voxel_size * 0.95)

        point_cloud = pv.PolyData(final_coords_array)
        point_cloud.point_data['BlockID'] = final_scalar_array

        voxels_mesh = point_cloud.glyph(geom=glyph_source, scale=False)

        # --- 3. Add Mesh and Color Bar ---

        # Since we use scalar data, we must choose a reasonable range (RNG)
        min_id = np.min(final_scalar_array)
        max_id = np.max(final_scalar_array)

        plotter.add_mesh(
            voxels_mesh,
            scalars='BlockID',              # Use the BlockID array for coloring
            cmap=cmap,                      # Use the specified colormap
            opacity=opacity,
            show_edges=False,

            # --- COLOR BAR GENERATION ---
            show_scalar_bar=True,           # Explicitly show the scalar bar
            scalar_bar_args={'title': 'Block ID', 'vertical': True},

            # Use the range of the scalar data
            rng=[min_id, max_id]
        )

        plotter.add_text(title, font_size=15)
        plotter.camera.ParallelProjectionOff()
        print("Rendering scene...")
        plotter.show()

    def visualize_block_ipf_map(self,
                                blocks_dict,
                                block_orientations,
                                title="Block IPF Map",
                                voxel_size=1.0,
                                opacity=1.0,
                                sample_direction=[0, 0, 1]):
        """
        Visualizes the IPF map of a collection of blocks.
        """
        if not blocks_dict:
            print("Block dictionary is empty. Nothing to visualize.")
            return

        plotter = pv.Plotter(window_size=[1000, 1000])
        glyph_source = pv.Cube(x_length=voxel_size * 0.95, y_length=voxel_size * 0.95, z_length=voxel_size * 0.95)

        total_blocks = len(blocks_dict)
        print(f"Visualizing IPF map for {total_blocks} blocks...")

        for i, (block_id, block_voxels) in enumerate(blocks_dict.items()):
            if block_id not in block_orientations:
                continue

            orientation = block_orientations[block_id]
            color = self.euler_to_ipf_color(orientation, sample_direction=sample_direction)

            point_cloud = pv.PolyData(block_voxels)
            voxels_mesh = point_cloud.glyph(geom=glyph_source, scale=False)

            plotter.add_mesh(voxels_mesh, color=color, opacity=opacity)

            # Print a progress update every 20 blocks
            if (i + 1) % 20 == 0 or (i + 1) == total_blocks:
                print(f"  ... processed {i + 1}/{total_blocks} blocks")

        plotter.add_text(title, font_size=15)
        plotter.camera.ParallelProjectionOff()
        print("Rendering scene...")
        plotter.show()

    def plot_distribution(self, data, title, xlabel, bins=20):
        """
        Plots a histogram for a given dataset.
        """
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=bins, edgecolor='black')
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    def calculate_pag_sizes(self, clusters_dict, packet_to_blocks_map, all_blocks_dict):
        """
        Calculates the size of each PAG in number of voxels.
        # --- Usage ---
    # pag_sizes_dict = calculate_pag_sizes(clusters_dict, packet_map, all_blocks)
    # plot_distribution(list(pag_sizes_dict.values()),
    #                   title="PAG Size Distribution",
    #                   xlabel="Size (Number of Voxels)")
        """
        pag_sizes = {}
        for pag_id, packet_gids in clusters_dict.items():
            total_voxels = 0
            for packet_gid in packet_gids:
                if packet_gid in packet_to_blocks_map:
                    for block_id in packet_to_blocks_map[packet_gid]:
                        if block_id in all_blocks_dict:
                            total_voxels += len(all_blocks_dict[block_id])
            pag_sizes[pag_id] = total_voxels
        return pag_sizes

    def calculate_packets_per_pag(self, clusters_dict):
        """
        Calculates the number of packets contained within each PAG.
        # --- Usage ---
        # packets_per_pag_dict = calculate_packets_per_pag(clusters_dict)
        # plot_distribution(list(packets_per_pag_dict.values()),
        #                   title="Distribution of Packets per PAG",
        #                   xlabel="Number of Packets")
        """
        return {pag_id: len(packet_gids) for pag_id, packet_gids in clusters_dict.items()}

    def calculate_blocks_per_packet(self, packet_to_blocks_map):
        """
        Calculates the number of blocks contained within each packet.

        # --- Usage ---
        # blocks_per_packet_dict = calculate_blocks_per_packet(packet_map)
        # plot_distribution(list(blocks_per_packet_dict.values()),
        #                   title="Distribution of Blocks per Packet",
        #                   xlabel="Number of Blocks")
        """
        return {packet_gid: len(block_ids) for packet_gid, block_ids in packet_to_blocks_map.items()}

    def calculate_block_morphology(self, all_blocks_dict):
        """
        Calculates the approximate thickness and aspect ratio for each block.

        This is a more advanced function that uses Principal Component Analysis
        (PCA) on each block's voxel cloud to determine its principal dimensions,
        which we use for thickness and aspect ratio.

        # --- Usage ---
        # morphology_data = calculate_block_morphology(all_blocks)
        # thicknesses = [data['thickness'] for data in morphology_data.values()]
        # aspect_ratios = [data['aspect_ratio'] for data in morphology_data.values()]

        # plot_distribution(thicknesses, "Block Thickness Distribution", "Approx. Thickness (Arbitrary Units)")
        # plot_distribution(aspect_ratios, "Block Aspect Ratio Distribution", "Aspect Ratio (Longest / Shortest Axis)")
        """
        from collections import defaultdict
        block_morphology = defaultdict(dict)
        for block_id, voxels in all_blocks_dict.items():
            if len(voxels) < 3: continue # Need at least 3 points for PCA

            # Use PCA to find the principal axes of the voxel cloud
            # 1. Center the data
            centered_voxels = voxels - np.mean(voxels, axis=0)
            # 2. Compute the covariance matrix
            cov_matrix = np.cov(centered_voxels, rowvar=False)
            # 3. Find eigenvalues, which represent the variance along principal axes
            eigenvalues, _ = np.linalg.eig(cov_matrix)

            # The principal dimensions are proportional to the sqrt of the eigenvalues
            principal_dims = np.sqrt(np.abs(eigenvalues))
            principal_dims.sort() # Sort from smallest to largest

            thickness = principal_dims[0]
            aspect_ratio = principal_dims[2] / principal_dims[0] if principal_dims[0] > 0 else 0

            block_morphology[block_id]['thickness'] = thickness
            block_morphology[block_id]['aspect_ratio'] = aspect_ratio

        return block_morphology

    def generate_block_neighbors(self, all_blocks_dict):
        """
        Generates a block adjacency dictionary from the voxel data.
        """
        print("Generating voxel-to-block map...")
        voxel_to_block_map = {}
        for block_id, voxels in all_blocks_dict.items():
            for v in voxels:
                voxel_to_block_map[tuple(v)] = block_id

        neigh_blid = defaultdict(set)

        print("Finding block neighbors (this can be slow)...")
        for block_id, voxels in all_blocks_dict.items():
            for v in voxels:
                # Check the 6 neighboring voxels in a 3D grid
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    neighbor_voxel = (v[0]+dx, v[1]+dy, v[2]+dz)

                    neighbor_block_id = voxel_to_block_map.get(neighbor_voxel)

                    if neighbor_block_id and neighbor_block_id != block_id:
                        neigh_blid[block_id].add(neighbor_block_id)

        # Convert sets to lists
        return {bid: list(neighbors) for bid, neighbors in neigh_blid.items()}

    def calculate_all_block_misorientations(self, block_neighbors, block_orientations):
        """
        Calculates the misorientation angle for all adjacent block pairs.
        # --- Usage ---
        block_neighbors = generate_block_neighbors(all_blocks)
        misorientation_data = calculate_all_block_misorientations(block_neighbors, all_block_orientations, gstslice.cubic_misorientation)
        plot_distribution(misorientation_data, "Block Boundary Misorientation Distribution", "Misorientation Angle (Degrees)", bins=90)
        """
        misorientations = []
        checked_pairs = set()
        for block_id, neighbors in block_neighbors.items():
            for neighbor_id in neighbors:
                pair_key = tuple(sorted((block_id, neighbor_id)))
                if pair_key in checked_pairs:
                    continue

                mis_deg, _, _ = self.cubic_misorientation_old1(block_orientations[block_id],
                                                               block_orientations[neighbor_id],
                                                               unique_tol_deg=1e-4, degrees=True)
                misorientations.append(mis_deg)
                checked_pairs.add(pair_key)

        return misorientations

    def display_key_statistics(self, data, name):
        """
        Calculates and prints key statistics for a given dataset.
        """
        if not data:
            print(f"--- Statistics for {name} ---\n  (No data to analyze)\n")
            return

        data_arr = np.array(data)

        stats = {
            "Mean": np.mean(data_arr),
            "Median": np.median(data_arr),
            "Std. Dev.": np.std(data_arr),
            "Min": np.min(data_arr),
            "Max": np.max(data_arr),
            "Count": len(data_arr)
        }

        print(f"--- Statistics for {name} ---")
        for key, value in stats.items():
            # Format floats to 2 decimal places, but keep Count as an integer
            if isinstance(value, float):
                print(f"  - {key:<10}: {value:.2f}")
            else:
                print(f"  - {key:<10}: {value}")
        print("") # Add a blank line for spacing

    def _create_voxel_to_block_map(self, all_blocks_dict):
        """
        Creates a dictionary mapping a (x, y, z) voxel coordinate tuple
        to its corresponding block ID.
        """
        voxel_to_block_map = {}
        for block_id, voxel_coords in all_blocks_dict.items():
            # Ensure coordinates are stored as tuples for dictionary keys
            for coord in voxel_coords:
                voxel_to_block_map[tuple(coord)] = block_id

        return voxel_to_block_map

    def extract_block_interfaces_and_neighbors(self,
                                             all_blocks_dict,
                                             block_orientations):
        """
        Identifies block interfaces and generates:
        1. A detailed map of interface voxels grouped by the two adjacent blocks,
           where each value is a single NumPy array (N, 3).
        2. A single NumPy array containing all unique boundary voxel coordinates.

        Parameters
        ----------
        all_blocks_dict : dict
            {block_id: voxel_array} for all blocks.
        block_orientations : dict
            {block_id: orientation (ea)} for all blocks.

        Returns
        -------
        tuple:
            - dict: {BlockA__BlockB: ndarray(N, 3) of BlockA_interface_voxels}
            - ndarray: (M, 3) NumPy array of all unique boundary voxel coordinates.
        """
        print("Starting interface and neighbor extraction...")

        # Step 1: Create the reverse lookup map
        voxel_to_block_map = self._create_voxel_to_block_map(all_blocks_dict)

        # Define the 6 nearest neighbors offsets (face-connected)
        neighbors = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])

        # Data structures to collect results
        interface_voxels_map = defaultdict(list)
        all_unique_boundary_coords = set()

        total_blocks = len(all_blocks_dict)
        block_counter = 0

        # Step 2: Iterate through every block and every voxel
        for block_id_A, voxel_coords_A in all_blocks_dict.items():

            # --- Progress Update ---
            block_counter += 1
            update_interval = max(1, total_blocks // 20)
            if block_counter % update_interval == 0 or block_counter == total_blocks:
                progress_percent = (block_counter / total_blocks) * 100
                print(f"  ... Processing interfaces for Block: {block_counter}/{total_blocks} ({progress_percent:.0f}%)")
            # -----------------------

            orientation_A = block_orientations.get(block_id_A)
            if orientation_A is None: continue

            for coord_A in voxel_coords_A:

                # Check all 6 neighbors
                for offset in neighbors:
                    coord_B_tuple = tuple(coord_A + offset)

                    block_id_B = voxel_to_block_map.get(coord_B_tuple)

                    # Check if neighbor exists AND belongs to a DIFFERENT block
                    if block_id_B is not None and block_id_B != block_id_A:

                        orientation_B = block_orientations.get(block_id_B)

                        # Compare orientations (Only register a boundary if misoriented)
                        if orientation_B is not None:
                            mis_angle, _, _ = self.cubic_misorientation_old1(
                                orientation_A,
                                orientation_B,
                                unique_tol_deg=1e-4,
                                degrees=True
                            )

                            if mis_angle > 0.01:
                                # Found a valid misoriented interface!

                                # 1. Populate the detailed map (Block A side)
                                key_A = f"{block_id_A}__{block_id_B}"
                                interface_voxels_map[key_A].append(coord_A)

                                # 2. Collect the coordinate for the final big array
                                all_unique_boundary_coords.add(coord_B_tuple)
                                all_unique_boundary_coords.add(tuple(coord_A))

        # --- Final Processing and Return ---

        # New Step: Convert the list of coordinates for each interface key into a single NumPy array
        final_interface_map = {}
        for key, coords_list in interface_voxels_map.items():
            if coords_list:
                final_interface_map[key] = np.array(coords_list, dtype=np.int64)
            else:
                final_interface_map[key] = np.empty((0, 3), dtype=np.int64)

        # Convert the set of unique boundary coordinates into a final NumPy array
        if all_unique_boundary_coords:
            final_boundary_array = np.array(list(all_unique_boundary_coords), dtype=np.int64)
        else:
            final_boundary_array = np.empty((0, 3), dtype=np.int64)

        print(f"\nInterface extraction complete.")
        print(f"  - Unique interfaces found: {len(final_interface_map)}")
        print(f"  - Total unique boundary voxels: {len(final_boundary_array)}")

        return final_interface_map, final_boundary_array

    def visualize_all_boundaries(self, all_boundary_voxels_array, voxel_size=1.0):
        """
        Visualizes all unique block boundary voxels as a single point cloud.

        Parameters
        ----------
        all_boundary_voxels_array : ndarray
            (N, 3) NumPy array of all unique boundary voxel coordinates.
        voxel_size : float
            The size of a single voxel (for scaling the plot).
        """
        if all_boundary_voxels_array.size == 0:
            print("Boundary array is empty. Nothing to plot.")
            return

        plotter = pv.Plotter(window_size=[1000, 1000])
        plotter.add_title(f"Complete Block Boundary Network ({all_boundary_voxels_array.shape[0]} voxels)")

        # Cube for better visualization of the voxel structure
        glyph_source = pv.Cube(x_length=voxel_size * 0.95,
                               y_length=voxel_size * 0.95,
                               z_length=voxel_size * 0.95)

        print("Plotting the consolidated boundary network...")

        # Create a PyVista PolyData object from the coordinates
        point_cloud = pv.PolyData(all_boundary_voxels_array)

        # Use glyphs (small cubes) for visualization
        voxels_mesh = point_cloud.glyph(geom=glyph_source, scale=False)

        plotter.add_mesh(
            voxels_mesh,
            color='darkred', # Use a single uniform color
            show_edges=False,
            render_points_as_spheres=False,
            opacity=0.8
        )

        plotter.show()

    def visualize_interface_map_fast(self, interface_map, voxel_size=1.0):
        """
        Visualizes the block boundaries using an optimized batching approach
        (pv.MultiBlock) for much faster rendering, coloring each unique interface
        segment with a distinct color.

        Parameters
        ----------
        interface_map : dict
            {BlockA__BlockB: ndarray(N, 3) of BlockA_interface_voxels}.
        voxel_size : float
            The size of a single voxel (for scaling the plot).
        """
        if not interface_map:
            print("Interface map is empty. Nothing to plot.")
            return

        start_time = time.time()

        # --- 1. SETUP AND BATCHING ---
        total_segments = len(interface_map)
        print(f"Plotting {total_segments} unique one-way interface segments...")

        # PyVista structure to hold all meshes and their data
        multi_block = pv.MultiBlock()

        # Array to store the color data for all voxels
        # We assign a unique integer ID to each interface for color mapping
        segment_ids = []

        # List to hold the coordinates of ALL interface voxels
        all_interface_coords = []

        segment_counter = 0
        update_interval = max(1, total_segments // 20)

        # Cube for better visualization of the voxel structure
        glyph_source = pv.Cube(x_length=voxel_size * 0.95,
                               y_length=voxel_size * 0.95,
                               z_length=voxel_size * 0.95)

        # --- 2. BATCHING LOOP ---
        for coords_array in interface_map.values():

            segment_counter += 1
            if segment_counter % update_interval == 0 or segment_counter == total_segments:
                progress_percent = (segment_counter / total_segments) * 100
                print(f"  ... Batching data: {segment_counter}/{total_segments} ({progress_percent:.0f}%)")

            if coords_array.size == 0:
                continue

            # Append coordinates and their corresponding segment ID
            all_interface_coords.append(coords_array)
            segment_ids.extend([segment_counter] * coords_array.shape[0])

        # Check if any data was found
        if not all_interface_coords:
            print("No non-empty interface segments found.")
            return

        # Consolidate all coordinates into one large NumPy array
        final_coords_array = np.concatenate(all_interface_coords, axis=0)

        # Create a single PolyData object for ALL interface voxels
        point_cloud = pv.PolyData(final_coords_array)

        # Add the segment ID data as a scalar array
        point_cloud.point_data['SegmentID'] = np.array(segment_ids, dtype=np.int32)

        # Apply the glyphs (cubes) once to the entire combined dataset
        voxels_mesh = point_cloud.glyph(geom=glyph_source, scale=False)

        end_batch_time = time.time()
        print(f"Batching complete in {end_batch_time - start_time:.2f} seconds.")

        # --- 3. PLOTTING ---
        plotter = pv.Plotter(window_size=[1000, 1000])
        plotter.add_title("Block Boundary Network - Colored by Interface Segment (FAST)")

        # The key to speed: Add the entire mesh once and color it by the 'SegmentID' array
        plotter.add_mesh(
            voxels_mesh,
            scalars='SegmentID', # Color by the unique segment ID
            cmap='gist_rainbow', # Use a diverse colormap for high contrast
            show_edges=False,
            render_points_as_spheres=False,
            # Ensure the color map is continuous over the range of segment IDs
            rng=[1, total_segments]
        )

        plotter.show()

    def map_block_ids_to_integers(self, all_blocks_dict):
        """
        Assigns a unique, sequential integer ID to every string-based Block ID
        in the provided dictionary.

        Parameters
        ----------
        all_blocks_dict : dict
            {string_block_id: voxel_array} for all blocks.

        Returns
        -------
        tuple:
            - int_to_str_map (dict): {integer_ID: string_Block_ID}
            - str_to_int_map (dict): {string_Block_ID: integer_ID}
        """

        # Get all unique Block IDs (strings)
        block_id_strings = list(all_blocks_dict.keys())

        int_to_str_map = {}
        str_to_int_map = {}

        # Start assignment from integer 1
        for i, block_id_str in enumerate(block_id_strings, start=1):

            # The integer is the key in the first dictionary
            int_to_str_map[i] = block_id_str

            # The string is the key in the inverse dictionary
            str_to_int_map[block_id_str] = i

        print(f"Mapped {len(block_id_strings)} Block IDs to unique integers (1 to {len(block_id_strings)}).")

        return int_to_str_map, str_to_int_map

    def generate_integer_block_map(self, all_blocks_dict, block_name_ID_map, lgi):
        """
        Creates a 3D NumPy array of the RVE size, where each voxel contains the
        unique integer ID corresponding to the crystallographic block it belongs to.

        Parameters
        ----------
        all_blocks_dict : dict
            {string_block_id: voxel_array}. The source of all voxel coordinates.
        block_name_ID_map : dict
            {string_Block_ID: integer_ID}. Used to translate string IDs to integers.
        self_lgi_reference : ndarray
            The 3D NumPy array (self.lgi) to determine the shape (Z, Y, X) of the RVE.

        Returns
        -------
        ndarray: A 3D array (Z, Y, X) containing the integer Block IDs (1, 2, 3...).
        """

        # Determine the shape of the RVE from the reference array (self.lgi)
        RVE_shape = lgi.shape

        # Initialize the map. Use a 0 or -1 to signify voxels not belonging to any block (outside RVE).
        # We will use -1 as a distinct background value.
        lbi = np.full(RVE_shape, -1, dtype=np.int32)  # Local Block Index

        print(f"Initialized 3D map with shape {RVE_shape}. Populating with integer Block IDs...")

        total_blocks = len(all_blocks_dict)
        block_counter = 0
        update_interval = max(1, total_blocks // 20)

        # --- Populate the map by iterating through every block ---
        for block_name, voxel_coords in all_blocks_dict.items():

            block_counter += 1
            if block_counter % update_interval == 0 or block_counter == total_blocks:
                progress_percent = (block_counter / total_blocks) * 100
                print(f"  ... Processing block {block_counter}/{total_blocks} ({progress_percent:.0f}%)")

            # 1. Get the unique integer ID for the current block
            # If a block was somehow missing from the map, we skip it.
            integer_id = block_name_ID_map.get(block_name)
            if integer_id is None:
                continue

            # 2. Extract coordinates (X, Y, Z)
            # The coordinates array should be (N, 3).
            coords = voxel_coords.astype(np.int32)

            # Note: We assume the coordinates (X, Y, Z) map directly to indices
            # (Z, Y, X) in the C-order NumPy array format. We must be careful
            # with indexing order:

            # Indices for the Z-axis (depth)
            Z_indices = coords[:, 2]
            # Indices for the Y-axis (height)
            Y_indices = coords[:, 1]
            # Indices for the X-axis (width)
            X_indices = coords[:, 0]

            # 3. Assign the integer ID to the corresponding voxel positions
            # Assumption: RVE array is indexed as [Z, Y, X].
            # If your array is indexed as [X, Y, Z], swap the indices accordingly.
            lbi[Z_indices, Y_indices, X_indices] = integer_id

        print("\n3D integer Block Map generation complete.")
        nx, ny, nz = lgi.shape
        grid = pv.UniformGrid()
        grid.dimensions = np.array([nx, ny, nz]) + 1   # points = cells+1
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        grid.cell_data.clear()
        grid.cell_data["lbi"] = lbi.ravel(order="F").astype(np.int32)
        grid = {'grid': grid,
                'lbi': 'lbi'}

        return lbi, grid

    def generate_integer_cluster_map(self, clusters_dict, lgi):
        """
        Creates a 3D NumPy array of the RVE size, where each voxel contains the
        unique integer ID corresponding to the parent PAG (Cluster ID) it belongs to.

        Parameters
        ----------
        clusters_dict : dict
            The PAG-to-Packet map: {PAG_ID: [Packet_ID_1, Packet_ID_2, ...]}
        lgi : ndarray
            The 3D array (self.lgi) where values are Packet IDs.

        Returns
        -------
        ndarray: A 3D array (Z, Y, X) containing the integer PAG IDs.
        """

        # Determine the shape of the RVE
        RVE_shape = lgi.shape

        # Initialize the map. Use a 0 to denote background/unassigned voxels.
        cluster_map_3D = np.zeros(RVE_shape, dtype=np.int32)

        # --- 1. Create a fast Packet ID to PAG ID lookup table ---
        # We need a dictionary {Packet_ID: PAG_ID}
        packet_to_pag_map = {}
        for pag_id, packet_ids in clusters_dict.items():
            for packet_id in packet_ids:
                packet_to_pag_map[packet_id] = pag_id

        print(f"Initialized 3D map with shape {RVE_shape}. Populating with Cluster IDs...")

        # --- 2. Vectorized Mapping using a temporary array ---

        # Extract all unique Packet IDs present in self.lgi (excluding 0, which is background)
        unique_packet_ids = np.unique(lgi[lgi > 0])

        # For each unique packet ID, find its corresponding PAG ID
        for packet_id in unique_packet_ids:
            pag_id = packet_to_pag_map.get(packet_id, 0) # Default to 0 if not found

            # Find all locations in self.lgi that match the current packet_id
            indices = (lgi == packet_id)

            # Assign the PAG ID to those same locations in the output map
            cluster_map_3D[indices] = pag_id

        print("3D integer Cluster Map generation complete.")
        nx, ny, nz = lgi.shape
        grid = pv.UniformGrid()
        grid.dimensions = np.array([nx, ny, nz]) + 1   # points = cells+1
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        grid.cell_data.clear()
        grid.cell_data["cluster_map_3D"] = cluster_map_3D.ravel(order="F").astype(np.int32)
        grid = {'grid': grid,
                'lcli': 'cluster_map_3D'}
        return cluster_map_3D, grid

    def generate_euler_angle_3d_maps(self, all_blocks_dict, block_orientations, lgi):
        """
        Creates three 3D NumPy arrays (phi1, Phi, phi2) of the RVE size,
        mapping the Bunge Euler angles to every voxel based on block assignment.

        Parameters
        ----------
        all_blocks_dict : dict
            {string_block_id: voxel_array}. Source of all voxel coordinates.
        block_orientations : dict
            {string_block_id: orientation (ea)}. Source of all Euler angle data.
        lgi : ndarray
            The 3D NumPy array (self.lgi) to determine the shape (Z, Y, X) of the RVE.

        Returns
        -------
        tuple: (phi1_map_3D, Phi_map_3D, phi2_map_3D)
            Three 3D arrays (Z, Y, X) containing the Euler angle components.
        """

        # Determine the shape of the RVE
        RVE_shape = lgi.shape

        # Initialize the three output arrays with zeros or a neutral value
        phi1_map_3D = np.zeros(RVE_shape, dtype=np.float32)
        Phi_map_3D = np.zeros(RVE_shape, dtype=np.float32)
        phi2_map_3D = np.zeros(RVE_shape, dtype=np.float32)

        print(f"Initialized 3D Euler angle maps with shape {RVE_shape}. Populating...")

        total_blocks = len(all_blocks_dict)
        block_counter = 0
        update_interval = max(1, total_blocks // 20)

        # --- Populate the maps by iterating through every block ---
        for block_name, voxel_coords in all_blocks_dict.items():

            block_counter += 1
            if block_counter % update_interval == 0 or block_counter == total_blocks:
                progress_percent = (block_counter / total_blocks) * 100
                print(f"  ... Processing block {block_counter}/{total_blocks} ({progress_percent:.0f}%)")

            # 1. Get the Euler angles for the current block
            euler_angles = block_orientations.get(block_name)
            if euler_angles is None:
                continue

            # Ensure euler_angles is iterable and has 3 components
            try:
                phi1, Phi, phi2 = euler_angles
            except ValueError:
                print(f"Warning: Orientation for {block_name} is invalid. Skipping.")
                continue

            # 2. Extract coordinates (X, Y, Z) and cast to integer indices
            coords = voxel_coords.astype(np.int32)

            # Indices for the RVE array (assuming [Z, Y, X] indexing)
            Z_indices = coords[:, 2]
            Y_indices = coords[:, 1]
            X_indices = coords[:, 0]

            # 3. Assign the three angle values to all corresponding voxel positions
            phi1_map_3D[Z_indices, Y_indices, X_indices] = phi1
            Phi_map_3D[Z_indices, Y_indices, X_indices] = Phi
            phi2_map_3D[Z_indices, Y_indices, X_indices] = phi2

        print("\n3D Euler Angle Map generation complete.")
        nx, ny, nz = lgi.shape
        grid = pv.UniformGrid()
        grid.dimensions = np.array([nx, ny, nz]) + 1   # points = cells+1
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        grid.cell_data.clear()
        grid.cell_data["phi1_map_3D"] = phi1_map_3D.ravel(order="F").astype(np.int32)
        grid.cell_data["Phi_map_3D"] = Phi_map_3D.ravel(order="F").astype(np.int32)
        grid.cell_data["phi2_map_3D"] = phi2_map_3D.ravel(order="F").astype(np.int32)
        grid = {'grid': grid,
                'phi1': "phi1_map_3D",
                'psi': "Phi_map_3D",
                'phi2': "phi2_map_3D"}
        return phi1_map_3D, Phi_map_3D, phi2_map_3D, grid

    def plot_pvgrid(self, pvgrid, scalar,
                    show_edges=False, alpha=1.0, title='',
                    cmap='nipy_spectral', _xname_='', _yname_='', _zname_=''):
        pvp = pv.Plotter()
        pvp.add_mesh(pvgrid,
                     scalars=scalar,
                     show_edges=show_edges,
                     opacity=alpha,
                     cmap=cmap)
        pvp.add_text(f"{title}", font_size=10)
        _ = pvp.add_axes(line_width=5, cone_radius=0.6,
                         shaft_length=0.7, tip_length=0.3,
                         ambient=0.5, label_size=(0.4, 0.16),
                         xlabel=_xname_, ylabel=_yname_, zlabel=_zname_,
                         viewport=(0, 0, 0.25, 0.25))
        pvp.show()

    def mesh_C3D8_V1(self, feature_ids_array, material_data, output_filename="abaqus_voxel_mesh.inp"):
        """
        Generates node coordinates, element connectivity, and element sets
        for a C3D8 hexahedral mesh based on a 3D NumPy array of feature IDs.
        """
        if feature_ids_array.ndim != 3:
            raise ValueError("Input array must be 3-dimensional.")

        # 1. Define Array Dimensions
        D1, D2, D3 = feature_ids_array.shape
        N1, N2, N3 = D1 + 1, D2 + 1, D3 + 1

        print(f"Mesh dimensions (Elements): {D1}x{D2}x{D3}")
        print(f"Grid dimensions (Nodes): {N1}x{N2}x{N3}")

        # 2. Generate Node Coordinates (*NODE)
        x_coords = np.arange(N1, dtype=float)
        y_coords = np.arange(N2, dtype=float)
        z_coords = np.arange(N3, dtype=float)

        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        coords = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
        node_ids = np.arange(1, len(coords) + 1)
        node_data = np.hstack((node_ids.reshape(-1, 1), coords))

        # 3. Generate Element Connectivity (*ELEMENT) and Element Sets (*ELSET)
        element_id_counter = 1
        element_data = []
        elset_map = {} # Stores {Feature ID: [Element IDs]}

        # We need a list of all unique feature IDs encountered to generate materials/sections
        unique_feature_ids = set()

        c3d8_offsets = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])

        def get_node_id(i, j, k):
            return k * (N1 * N2) + j * N1 + i + 1

        for k in range(D3):
            for j in range(D2):
                for i in range(D1):
                    feature_id = feature_ids_array[i, j, k]
                    unique_feature_ids.add(feature_id) # Collect unique IDs

                    node_ids_for_element = []
                    for di, dj, dk in c3d8_offsets:
                        node_id = get_node_id(i + di, j + dj, k + dk)
                        node_ids_for_element.append(node_id)

                    element_data.append([element_id_counter] + node_ids_for_element)

                    if feature_id not in elset_map:
                        elset_map[feature_id] = []
                    elset_map[feature_id].append(element_id_counter)

                    element_id_counter += 1

        element_data = np.array(element_data, dtype=int)

        # 4. Write the Abaqus Input File
        self.write_abaqus_inp(output_filename, node_data, element_data, elset_map, unique_feature_ids, material_data)
        print(f"\nSuccessfully generated Abaqus input file: '{output_filename}'")
        print(f"Total nodes: {len(node_data)}")
        print(f"Total elements: {len(element_data)}")
        print(f"Unique feature IDs found: {list(elset_map.keys())}")

    def write_abaqus_sections_materials(self, f, unique_feature_ids, material_data):
        """Generates *MATERIAL and *SOLID SECTION definitions."""

        f.write("\n**\n** MATERIALS AND SECTIONS\n**\n")

        # --- MATERIALS ---
        f.write("** MATERIAL DEFINITIONS\n")
        for feature_id in sorted(list(unique_feature_ids)):
            mat_name = f"MAT_ID_{feature_id}"
            props = material_data[feature_id]

            f.write(f"*MATERIAL, NAME={mat_name}\n")
            f.write("*ELASTIC\n")
            # Write Young's Modulus and Poisson's Ratio
            f.write(f"{props['YOUNGS']:.3e}, {props['POISSON']:.4f}\n")

        # --- SECTIONS ---
        f.write("\n** SECTION DEFINITIONS\n")
        for feature_id in sorted(list(unique_feature_ids)):
            elset_name = f"FEATURE_{feature_id}"
            mat_name = f"MAT_ID_{feature_id}"

            # Link the element set (FEATURE_ID) to the material (MAT_ID_ID)
            f.write(f"*SOLID SECTION, ELSET={elset_name}, MATERIAL={mat_name}\n")

    def write_abaqus_inp(self, filename, node_data, element_data, elset_map, unique_feature_ids, material_data):
        """Formats and writes the data to an Abaqus .inp file."""

        try:
            with open(filename, 'w') as f:
                f.write("*HEADING\n")
                f.write("** Voxel Mesh from NumPy Array - C3D8 Elements\n")
                f.write(f"** Generated on: {np.datetime_as_string(np.datetime64('now'))}\n")
                f.write("*PREPRINT, ECHO=NO, HISTORY=NO, MODEL=NO\n\n")

                # --- NODE Coordinates ---
                f.write("**\n** NODE DEFINITIONS\n**\n")
                f.write("*NODE\n")

                for row in node_data:
                    node_id = int(row[0])
                    coords = row[1:]
                    f.write(f"{node_id:d}, {coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f}\n")

                # --- ELEMENT Connectivity ---
                f.write("\n**\n** ELEMENT DEFINITIONS (Type C3D8)\n**\n")
                f.write("*ELEMENT, TYPE=C3D8\n")

                for row in element_data:
                    f.write(f"{row[0]:d}, {', '.join(map(str, row[1:]))}\n")

                # --- ELEMENT SETS (Feature IDs) ---
                f.write("\n**\n** ELEMENT SETS (Based on Feature IDs)\n**\n")
                for feature_id, element_list in elset_map.items():
                    elset_name = f"FEATURE_{feature_id}"
                    f.write(f"*ELSET, ELSET={elset_name}, GENERATE\n")

                    for i in range(0, len(element_list), 8):
                        line_elements = element_list[i:i+8]
                        line = ", ".join(map(str, line_elements))
                        f.write(f"{line}\n")

                # --- NEW: SECTIONS and MATERIALS ---
                self.write_abaqus_sections_materials(f, unique_feature_ids, material_data)

                # --- End of File ---
                f.write("\n**\n** END OF MESH GENERATION\n**\n")

            print(f"File handle closed successfully for: {filename}")

        except IOError as e:
            print(f"ERROR: Could not write file to '{filename}'. Check file permissions or path.")
            print(f"Details: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during file writing: {e}")
            raise

    def find_interface_voxels_3D(self):
        """
        Identifies voxels belonging to Grain Boundary Surfaces (GBS) by checking
        all 6 face-sharing neighbors in the 3D array, storing the (i, j, k) coordinates.

        Args:
            lgi_array (np.ndarray): The 3D array of voxel GIDs (Grain IDs).

        Returns:
            defaultdict: A dictionary where keys are "gbs_GID_A_GID_B" and values are
                         lists of (i, j, k) tuples representing the voxel coordinates.
        """

        if self.lgi.ndim != 3:
            raise ValueError("Input array must be 3-dimensional.")

        # 1. Initialization and Setup
        interface_voxels = defaultdict(list)
        D1, D2, D3 = self.lgi.shape

        # 6 face-sharing neighbor directions to check
        neighbor_offsets = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]

        # 2. Iterate through all voxels in 3D space
        for i in range(D1):
            for j in range(D2):
                for k in range(D3):

                    # GID of the current voxel (Voxel A)
                    gid_a = self.lgi[i, j, k]

                    # Tuple of the current voxel's coordinates
                    voxel_coords = (i, j, k)

                    # 3. Check 6 neighbors
                    for di, dj, dk in neighbor_offsets:
                        ni, nj, nk = i + di, j + dj, k + dk

                        # Check array bounds
                        if 0 <= ni < D1 and 0 <= nj < D2 and 0 <= nk < D3:

                            # GID of the neighboring voxel (Voxel B)
                            gid_b = self.lgi[ni, nj, nk]

                            # Check for interface condition
                            if gid_a != gid_b:
                                # Interface found! The current voxel (GID A) neighbors GID B.
                                key = f"gbs_{gid_a}_{gid_b}"

                                # Add the voxel's coordinates only if this specific interface
                                # key hasn't recorded this voxel's coordinates yet.
                                if voxel_coords not in interface_voxels[key]:
                                    interface_voxels[key].append(voxel_coords)

        return dict(interface_voxels)

    def visualize_interface_voxels(self, interface_map, cmap_name='nipy_spectral'):
        """
        Visualizes ONLY the interface voxels, omitting the bulk material,
        highlighting them using PyVista.

        Args:
            lgi_array (np.ndarray): The 3D array of voxel GIDs (Grain IDs).
            interface_map (dict): Dictionary of interface voxels (coords).
            cmap_name (str, optional): The name of the PyVista/Matplotlib colormap to use.
                                       Defaults to 'nipy_spectral'.
        """
        if not interface_map:
            print("No interfaces found to visualize.")
            return

        lgi_array = self.lgi

        D1, D2, D3 = lgi_array.shape
        scalar_data = np.zeros(lgi_array.shape, dtype=int)

        interface_key_to_id = {}
        interface_id_counter = 1

        # Assign Unique ID to Interface Voxels
        for key, coords_list in interface_map.items():
            if key not in interface_key_to_id:
                interface_key_to_id[key] = interface_id_counter
                interface_id_counter += 1

            current_id = interface_key_to_id[key]

            for i, j, k in coords_list:
                scalar_data[i, j, k] = current_id

        # Create PyVista Structured Grid
        grid = pv.UniformGrid()
        grid.dimensions = (D1 + 1, D2 + 1, D3 + 1)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (0.0, 0.0, 0.0)

        # Add scalar data (Interface_Type) to the cells
        grid.cell_data['Interface_Type'] = scalar_data.flatten(order='C')

        # --- Filter out bulk voxels (where Interface_Type is 0) ---
        interface_grid = grid.threshold(0.5, invert=False, scalars='Interface_Type')

        if interface_grid.n_cells == 0:
            print("No interface voxels found after filtering. Nothing to display.")
            return

        # Visualization
        pl = pv.Plotter()

        pl.add_mesh(
            interface_grid, # Use the filtered grid
            scalars='Interface_Type',
            cmap=cmap_name, # <-- USING THE USER-DEFINED/DEFAULT COLORMAP
            show_edges=True,
            opacity=1.0,
            scalar_bar_args={
                'title': 'Interface Type ID',
                'vertical': True
            }
        )

        legend_text = "\n".join([f"ID {v}: {k}" for k, v in interface_key_to_id.items()])
        pl.add_text(legend_text, position='lower_left', font_size=10)

        print(f"\nDisplaying PyVista plot (Interface Voxels Only) using cmap: {cmap_name}...")
        pl.show()

    def visualize_interface_voxels_v1(self, interface_map, cmap_name='nipy_spectral', target_gids=None):
        """
        Visualizes ONLY the interface voxels, filtering by specific GIDs requested by the user.

        Args:
            lgi_array (np.ndarray): The 3D array of voxel GIDs (Grain IDs).
            interface_map (dict): Dictionary of interface voxels (coords).
            cmap_name (str, optional): The name of the PyVista/Matplotlib colormap to use.
                                       Defaults to 'nipy_spectral'.
            target_gids (list, optional): A list of integer GIDs to focus on. Only interfaces
                                          involving these GIDs will be plotted.
                                          If None, all interfaces are plotted.
        """
        if not interface_map:
            print("No interfaces found to visualize.")
            return

        lgi_array = self.lgi

        D1, D2, D3 = lgi_array.shape
        scalar_data = np.zeros(lgi_array.shape, dtype=int)

        interface_key_to_id = {}
        interface_id_counter = 1

        # --- 1. Filter Interfaces Based on target_gids ---

        # Convert target_gids to a set of strings for efficient checking
        if target_gids:
            target_gid_strings = {str(g) for g in target_gids}
            print(f"Filtering visualization to interfaces involving GIDs: {target_gids}")

        # --- 2. Assign Unique ID to Filtered Interface Voxels ---

        # Use a temporary map to build the filtered interface data and generate IDs
        filtered_interface_map = {}

        for key, coords_list in interface_map.items():
            # Key format: "gbs_GID_A_GID_B"
            parts = key.split('_')
            gid_a = parts[1]
            gid_b = parts[2]

            is_target_interface = True
            if target_gids:
                # Check if GID_A OR GID_B is in the target list
                if gid_a not in target_gid_strings and gid_b not in target_gid_strings:
                    is_target_interface = False

            if is_target_interface:
                # Assign a unique ID to this interface key
                if key not in interface_key_to_id:
                    interface_key_to_id[key] = interface_id_counter
                    interface_id_counter += 1

                current_id = interface_key_to_id[key]
                filtered_interface_map[key] = coords_list # Store for legend

                # Mark all voxels belonging to this interface set
                for i, j, k in coords_list:
                    scalar_data[i, j, k] = current_id

        if not filtered_interface_map:
            print("No matching interface voxels found for the specified target GIDs. Nothing to display.")
            return

        # --- 3. Create PyVista Structured Grid and Filter ---

        grid = pv.UniformGrid()
        grid.dimensions = (D1 + 1, D2 + 1, D3 + 1)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (0.0, 0.0, 0.0)

        grid.cell_data['Interface_Type'] = scalar_data.flatten(order='C')

        # Filter: Keep only cells where Interface_Type > 0 (i.e., not bulk)
        interface_grid = grid.threshold(0.5, invert=False, scalars='Interface_Type')

        # --- 4. Visualization ---

        pl = pv.Plotter()

        pl.add_mesh(
            interface_grid,
            scalars='Interface_Type',
            cmap=cmap_name,
            show_edges=True,
            opacity=1.0,
            scalar_bar_args={
                'title': 'Interface Type ID',
                'vertical': True
            }
        )

        legend_text = "\n".join([f"ID {v}: {k}" for k, v in interface_key_to_id.items()])
        pl.add_text(legend_text, position='lower_left', font_size=10)

        print(f"\nDisplaying {len(interface_key_to_id)} filtered interface types using cmap: {cmap_name}...")
        pl.show()

    def mesh_fm_steel_C3D8(self, lbi_array, material_properties, all_data, output_filename="fm_steel_C3D8.inp"):
        """
        Generates a C3D8 linear hexahedral mesh specifically for the FM Steel
        microstructure, including all hierarchical element sets and orientations.

        This function acts as an orchestrator for the C3D8 meshing process.

        Parameters
        ----------
        lbi_array : ndarray
            3D NumPy array where each voxel contains its unique integer Block ID.
        material_properties : dict
            Dictionary with material constants, e.g., {'YOUNGS': 2e11, 'POISSON': 0.3}.
        all_data : dict
            A dictionary containing all necessary hierarchical data from the
            microstructure generation, including 'packet_elsets', 'pag_elsets',
            and 'block_orientations'.
        output_filename : str
            The name of the .inp file to be created.
        """
        if lbi_array.ndim != 3:
            raise ValueError("Input array 'lbi_array' must be 3-dimensional.")

        # --- 1. Node Generation ---
        D1, D2, D3 = lbi_array.shape
        N1, N2, N3 = D1 + 1, D2 + 1, D3 + 1

        x = np.arange(N1, dtype=float)
        y = np.arange(N2, dtype=float)
        z = np.arange(N3, dtype=float)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
        node_ids_col = np.arange(1, len(coords) + 1).reshape(-1, 1)
        node_data = np.hstack((node_ids_col, coords))

        # --- 2. Element & Block Set Generation ---
        element_id_counter = 1
        element_data = []
        block_elset_map = defaultdict(list)

        def get_node_id(i, j, k):
            return k * (N1 * N2) + j * N1 + i + 1

        for k_el in range(D3):
            for j_el in range(D2):
                for i_el in range(D1):
                    block_id = lbi_array[i_el, j_el, k_el]
                    # Skip background voxels if they are marked (e.g., with ID 0 or -1)
                    if block_id <= 0:
                        continue

                    nodes = [
                        get_node_id(i_el,     j_el,     k_el),
                        get_node_id(i_el + 1, j_el,     k_el),
                        get_node_id(i_el + 1, j_el + 1, k_el),
                        get_node_id(i_el,     j_el + 1, k_el),
                        get_node_id(i_el,     j_el,     k_el + 1),
                        get_node_id(i_el + 1, j_el,     k_el + 1),
                        get_node_id(i_el + 1, j_el + 1, k_el + 1),
                        get_node_id(i_el,     j_el + 1, k_el + 1)
                    ]
                    element_data.append([element_id_counter] + nodes)
                    block_elset_map[block_id].append(element_id_counter)
                    element_id_counter += 1

        # --- 3. Assemble all hierarchical data for the writer ---
        data_for_inp = {
            'element_type': 'C3D8',
            'node_data': node_data,
            'element_data': np.array(element_data, dtype=int),
            'block_elsets': block_elset_map,
            'packet_elsets': all_data.get('packet_elsets', {}),
            'pag_elsets': all_data.get('pag_elsets', {}),
            'block_orientations': all_data.get('block_orientations', {}),
            'material_properties': material_properties
        }

        # --- 4. Call the specialized writer function ---
        self.write_fm_steel_abaqus_inp(output_filename, data_for_inp)
        print(f"\nC3D8 meshing complete for '{output_filename}'")


    def mesh_fm_steel_C3D20(self, lbi_array, material_properties, all_data, output_filename="fm_steel_C3D20.inp"):
        """
        Generates a C3D20 quadratic hexahedral mesh for the FM Steel
        microstructure, including all hierarchical sets and orientations.
        """
        if lbi_array.ndim != 3:
            raise ValueError("Input array 'lbi_array' must be 3-dimensional.")

        # --- 1. Node Generation (Finer Grid) ---
        D1, D2, D3 = lbi_array.shape
        N1, N2, N3 = 2 * D1 + 1, 2 * D2 + 1, 2 * D3 + 1

        x = np.linspace(0, D1, N1)
        y = np.linspace(0, D2, N2)
        z = np.linspace(0, D3, N3)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
        node_ids_col = np.arange(1, len(coords) + 1).reshape(-1, 1)
        node_data = np.hstack((node_ids_col, coords))

        # --- 2. Element Connectivity & Block Set Generation ---
        element_id_counter = 1
        element_data = []
        block_elset_map = defaultdict(list)

        def get_node_id(i, j, k):
            return k * (N1 * N2) + j * N1 + i + 1

        for k_el in range(D3):
            for j_el in range(D2):
                for i_el in range(D1):
                    block_id = lbi_array[i_el, j_el, k_el]
                    if block_id <= 0:
                        continue

                    # Base index on the finer node grid
                    i, j, k = 2 * i_el, 2 * j_el, 2 * k_el

                    # C3D20 Node Order: 8 corners, then 12 mid-edge nodes
                    nodes = [
                        # Corners (Nodes 1-8)
                        get_node_id(i,   j,   k), get_node_id(i+2, j,   k), get_node_id(i+2, j+2, k), get_node_id(i,   j+2, k),
                        get_node_id(i,   j,   k+2), get_node_id(i+2, j,   k+2), get_node_id(i+2, j+2, k+2), get_node_id(i,   j+2, k+2),
                        # Mid-edge Nodes (Nodes 9-20)
                        get_node_id(i+1, j,   k), get_node_id(i+2, j+1, k), get_node_id(i+1, j+2, k), get_node_id(i,   j+1, k),
                        get_node_id(i,   j,   k+1), get_node_id(i+2, j,   k+1), get_node_id(i+2, j+2, k+1), get_node_id(i,   j+2, k+1),
                        get_node_id(i+1, j,   k+2), get_node_id(i+2, j+1, k+2), get_node_id(i+1, j+2, k+2), get_node_id(i,   j+1, k+2)
                    ]
                    element_data.append([element_id_counter] + nodes)
                    block_elset_map[block_id].append(element_id_counter)
                    element_id_counter += 1

        # --- 3. Assemble all data for the writer ---
        data_for_inp = {
            'element_type': 'C3D20',
            'node_data': node_data,
            'element_data': np.array(element_data, dtype=int),
            'block_elsets': block_elset_map,
            'packet_elsets': all_data.get('packet_elsets', {}),
            'pag_elsets': all_data.get('pag_elsets', {}),
            'block_orientations': all_data.get('block_orientations', {}),
            'material_properties': material_properties
        }

        # --- 4. Call the specialized writer function ---
        self.write_fm_steel_abaqus_inp(output_filename, data_for_inp)
        print(f"\nC3D20 meshing complete for '{output_filename}'")


    def write_fm_steel_abaqus_inp(self, filename, data):
        """
        Writes a complete Abaqus .inp file for the FM Steel model, including
        all hierarchical sets (Blocks, Packets, PAGs) and crystal orientations.
        """
        try:
            with open(filename, 'w') as f:
                # --- HEADING ---
                f.write("*HEADING\n")
                f.write(f"** FM Steel Voxel Mesh - {data['element_type']} Elements\n")
                f.write(f"** Generated on: {np.datetime_as_string(np.datetime64('now'))}\n")
                f.write("*PREPRINT, ECHO=NO, HISTORY=NO, MODEL=NO\n\n")

                # --- NODES ---
                f.write("**\n** NODE DEFINITIONS\n**\n")
                f.write("*NODE\n")
                np.savetxt(f, data['node_data'], fmt=['%d'] + ['%.6f']*3, delimiter=', ')

                # --- ELEMENTS ---
                f.write(f"\n**\n** ELEMENT DEFINITIONS (Type {data['element_type']})\n**\n")
                f.write(f"*ELEMENT, TYPE={data['element_type']}\n")
                np.savetxt(f, data['element_data'], fmt='%d', delimiter=', ')

                # --- HIERARCHICAL ELEMENT SETS ---
                f.write("\n**\n** HIERARCHICAL ELEMENT SETS\n**\n")
                # Block Sets
                f.write("** Block Sets\n")
                for block_id, elist in sorted(data['block_elsets'].items()):
                    f.write(f"*ELSET, ELSET=BLOCK_{block_id}\n")
                    for i in range(0, len(elist), 16):
                        f.write(", ".join(map(str, elist[i:i+16])) + "\n")
                # Packet Sets
                f.write("** Packet Sets\n")
                for packet_id, elist in sorted(data['packet_elsets'].items()):
                    f.write(f"*ELSET, ELSET=PACKET_{packet_id}\n")
                    for i in range(0, len(elist), 16):
                        f.write(", ".join(map(str, elist[i:i+16])) + "\n")
                # PAG Sets
                f.write("** PAG Sets\n")
                for pag_id, elist in sorted(data['pag_elsets'].items()):
                    f.write(f"*ELSET, ELSET=PAG_{pag_id}\n")
                    for i in range(0, len(elist), 16):
                        f.write(", ".join(map(str, elist[i:i+16])) + "\n")

                # --- ORIENTATIONS ---
                f.write("\n**\n** CRYSTAL ORIENTATION DEFINITIONS\n**\n")
                for block_id, angles in sorted(data['block_orientations'].items()):
                    f.write(f"*ORIENTATION, NAME=ORI_BLOCK_{block_id}\n")
                    f.write("3, {:.4f}, {:.4f}, {:.4f}\n".format(angles[0], angles[1], angles[2]))

                # --- MATERIAL ---
                f.write("\n**\n** MATERIAL DEFINITION\n**\n")
                f.write("*MATERIAL, NAME=FM_STEEL\n")
                f.write("*ELASTIC\n")
                props = data['material_properties']
                f.write(f"{props['YOUNGS']:.3e}, {props['POISSON']:.4f}\n")

                # --- SECTIONS (linking everything together) ---
                f.write("\n**\n** SOLID SECTION DEFINITIONS\n**\n")
                for block_id in sorted(data['block_elsets'].keys()):
                    f.write(f"*SOLID SECTION, ELSET=BLOCK_{block_id}, MATERIAL=FM_STEEL, ORIENTATION=ORI_BLOCK_{block_id}\n")

                f.write("\n** END OF FILE\n")

            print(f"\nSuccessfully generated Abaqus input file: '{filename}'")
            print(f"Total Nodes: {len(data['node_data'])}")
            print(f"Total Elements: {len(data['element_data'])}")

        except IOError as e:
            print(f"ERROR: Could not write file to '{filename}'.")
            print(f"Details: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during file writing: {e}")
            raise
