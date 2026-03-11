"""
Created on Fri Jun 28 09:32:49 2024

@author: Dr. Sunil Anandatheertha
"""
import cv2
import math
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from upxo._sup.gops import att
import defdap.ebsd as defDap_ebsd
# from defdap.quat import Quat
from scipy.ndimage import generic_filter
from upxo._sup.validation_values import _validation
from scipy.ndimage import binary_dilation, generate_binary_structure
from upxo._sup import dataTypeHandlers as dth
from upxo.xtal.mcgrain2d_definitions import grain2d
from upxo._sup.data_ops import find_intersection, find_union_with_counts


class polyxtal2d():
    __slots__ = ('map', 'lgi', 'gid', 'n', 'g', 'bjp', 'neigh_gid',
                 'mprop', 'quat', 'ea', 'closs', 'gbjp', 'gbseg1',
                 'centroids', 'gb_discrete', 'flags', 'gidshuffled',
                 'bbox', 'bbox_bounds', 'bbox_ex', 'bbox_ex_bounds',
                 '__gi__', 'positions', 'valid_mprops', 'geom'
                 )
    EPS = 1e-12

    def __init__(self):
        self.mprop = {'npixels': None,
                      'perimeter': None,
                      'solidity': None,
                      }
        '''The following need not be set while initiating this polyxtal2d
        class. They will rather be set during calls to concerned defnitions.'''
        self.flags = {'gid_shuffle_lgi': False,
                      'gid_shuffle_lgi_method': 'random',
                      'rearrange_g_after_lgi_shuffle': False,
                      }
        self.__setup__positions__()
        self.valid_mprops = {'npixels': False,
                             'npixels_gb': False,
                             'area': True,
                             'eq_diameter': False,
                             'perimeter': False,
                             'perimeter_crofton': False,
                             'compactness': False,
                             'gb_length_px': False,
                             'aspect_ratio': False,
                             'solidity': False,
                             'morph_ori': False,
                             'circularity': False,
                             'eccentricity': False,
                             'feret_diameter': False,
                             'major_axis_length': False,
                             'minor_axis_length': False,
                             'euler_number': False,
                             'char_grain_positions': False,
                             'area_bbox':  False,
                             'area_convex':  False,
                             'moments':  False,
                             'orientation':  False,
                             'inertia_tensor':  False,
                             'inertia_tensor_eigvals':  False,
                             }

    def __iter__(self):
        self.__gi__ = 1
        return self

    def __next__(self):
        if self.n:
            if self.__gi__ <= self.n:
                thisgrain = self.g[self.__gi__]['grain']
                self.__gi__ += 1
                return thisgrain
            else:
                raise StopIteration

    def __repr__(self):
        return f"UPXO.MCGS2D.PXTAL_map {id(self)}"

    def __att__(self):
        return att(self)

    def setup(self, map_type='ebsd',
              path_filename_noext=None,
              apply_kuwahara=False,
              kuwahara_misori=5,
              ):
        """
        Crystal Orientation Map. EBSD dataswt is one which can be loadsed.
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validate user inputs
        from upxo.interfaces.defdap.importebsd import ebsd_data
        UPXO_gstslice_EBSDmap = ebsd_data.load_ctf(path_filename_noext)
        self.map = deepcopy(UPXO_gstslice_EBSDmap.map_raw)
        self.buildQuatArray()
        if apply_kuwahara:
            self.kuwahara_filter(misori=kuwahara_misori)

    def kuwahara_filter(self, misori=5):
        self.map.filterData(misOriTol=misori)

    def buildQuatArray(self):
        self.map.buildQuatArray()

    def find_grains_gb(self, gb_misori=10, min_grain_size=1, print_msg=True):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validate user inputs
        print_msg and print(40*'-', '\n Finding boundaries.')
        self.map.findBoundaries(boundDef=gb_misori)
        print_msg and print(40*'-', '\n Finding grains.')
        self.map.findGrains(minGrainSize=min_grain_size)
        print_msg and print(40*'-', '\n Building neighbourhood network.')
        self.map.buildNeighbourNetwork()

    def port_essentials(self, print_msg=True):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        print_msg and print(40*'-', '\n Porting essential data to UPXO.')
        self.set_lgi()
        self.set_grains()
        self.flags['gid_shuffle_lgi'] = True
        self.set_gid()
        self.set_grains()
        self.shuffle_lgi_random_gid_wise()
        self.rearrange_g_after_lgi_shuffle()
        self.set_grain_locations()
        self.charecterize_mprops()
        print('Success.')

    def charecterize_mprops(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        print(40*'-', '\n Calculating morphological properties.')
        self.set_n()
        self.set_grain_centroids(coord_source='upxo_lgi')
        self.set_bounding_boxes()
        self.set_mprop_generator(use_extended_bbox=False)
        self.set_mprops()

    def set_mprops(self):
        self.set_mprop_npixels()
        self.set_mprop_area()
        self.set_mprop_perimeter_crofton()
        self.set_mprop_major_axis_length()
        self.set_mprop_minor_axis_length()
        self.set_mprop_ar()
        self.set_mprop_eccentricity()
        self.set_mprop_area_bbox()
        self.set_mprop_area_convex()
        self.set_mprop_equivalent_diameter_area()
        self.set_mprop_euler_number()
        self.set_mprop_feret_diameter_max()
        self.set_mprop_moments()
        self.set_mprop_solidity()
        self.set_mprop_orientation()
        self.set_mprop_inertia_tensor()
        self.set_mprop_inertia_tensor_eigvals()

    def get_data_template_giddict(self, ncopies):
        # Validation
        data_tmpt = {gid: None for gid in self.gid}
        if ncopies == 1:
            return data_tmpt
        else:
            return [data_tmpt for _ in range(ncopies)]

    def set_bounding_boxes(self):
        self.bbox_bounds = {gid: None for gid in self.gid}
        self.bbox = {gid: None for gid in self.gid}
        self.bbox_ex_bounds = {gid: None for gid in self.gid}
        self.bbox_ex = {gid: None for gid in self.gid}
        # -------------------------------
        for gid in self.gid:
            _, L = cv2.connectedComponents(np.array(self.lgi == gid,
                                                    dtype=np.uint8))
            # . . . . . . . . . . .
            rmin = np.where(L == 1)[0].min()
            rmax = np.where(L == 1)[0].max()+1
            cmin = np.where(L == 1)[1].min()
            cmax = np.where(L == 1)[1].max()+1
            # . . . . . . . . . . .
            Rlab, Clab = L.shape
            rmin_ex, rmax_ex = rmin - int(rmin != 0), rmax+int(rmin != Rlab)
            cmin_ex, cmax_ex = cmin - int(cmin != 0), cmax+int(cmax != Clab)
            # . . . . . . . . . . .
            self.bbox_bounds[gid] = [rmin, rmax, cmin, cmax]
            self.bbox[gid] = np.array(L[rmin:rmax, cmin:cmax], dtype=np.uint8)
            self.bbox_ex_bounds[gid] = [rmin_ex, rmax_ex, cmin_ex, cmax_ex]
            self.bbox_ex[gid] = np.array(L[rmin_ex:rmax_ex, cmin_ex:cmax_ex],
                                         dtype=np.uint8)

    def set_mprop_generator(self, use_extended_bbox=False):
        from skimage.measure import regionprops
        if not use_extended_bbox:
            for gid in self.gid:
                self.g[gid]['grain'].skprop = regionprops(self.bbox[gid],
                                                          cache=False)[0]
        else:
            for gid in self.gid:
                self.g[gid]['grain'].skprop = regionprops(self.bbox_ex[gid],
                                                          cache=False)[0]

    def set_mprop_npixels(self):
        self.mprop['npixels'] = np.array([self.g[gid]['grain'].skprop.num_pixels
                                          for gid in self.gid])

    def set_mprop_area(self):
        self.mprop['area'] = np.array([self.g[gid]['grain'].skprop.area
                                   for gid in self.gid])

    def set_mprop_perimeter(self):
        self.mprop['perimeter'] = np.array([self.g[gid]['grain'].skprop.perimeter_crofton
                                   for gid in self.gid])

    def set_mprop_perimeter_crofton(self):
        self.mprop['perimeter_crofton'] = np.array([self.g[gid]['grain'].skprop.perimeter_crofton
                                           for gid in self.gid])

    def set_mprop_major_axis_length(self):
        self.mprop['major_axis_length'] = np.array([self.g[gid]['grain'].skprop.major_axis_length
                                                    for gid in self.gid])

    def set_mprop_minor_axis_length(self):
        self.mprop['minor_axis_length'] = np.array([self.g[gid]['grain'].skprop.minor_axis_length
                                           for gid in self.gid])

    def set_mprop_ar(self):
        aspect_ratio = []
        for gid in self.gid:
            skprop = self.g[gid]['grain'].skprop
            maj_axis = skprop.major_axis_length
            min_axis = skprop.minor_axis_length
            if min_axis <= self.EPS:
                aspect_ratio.append(np.inf)
            else:
                aspect_ratio.append(maj_axis/min_axis)
        self.mprop['aspect_ratio'] = aspect_ratio

    def set_mprop_eccentricity(self):
        self.mprop['eccentricity'] = np.array([self.g[gid]['grain'].skprop.eccentricity
                                           for gid in self.gid])

    def set_mprop_area_bbox(self):
        self.mprop['area_bbox'] = np.array([self.g[gid]['grain'].skprop.area_bbox
                                           for gid in self.gid])

    def set_mprop_area_convex(self):
        self.mprop['area_convex'] = np.array([self.g[gid]['grain'].skprop.area_convex
                                           for gid in self.gid])

    def set_mprop_equivalent_diameter_area(self):
        self.mprop['equivalent_diameter_area'] = np.array([self.g[gid]['grain'].skprop.equivalent_diameter_area
                                           for gid in self.gid])

    def set_mprop_euler_number(self):
        self.mprop['euler_number'] = np.array([self.g[gid]['grain'].skprop.euler_number
                                           for gid in self.gid])

    def set_mprop_feret_diameter_max(self):
        self.mprop['feret_diameter_max'] = np.array([self.g[gid]['grain'].skprop.feret_diameter_max
                                           for gid in self.gid])

    def set_mprop_moments(self):
        self.mprop['moments'] = [self.g[gid]['grain'].skprop.moments
                                           for gid in self.gid]

    def set_mprop_solidity(self):
        self.mprop['solidity'] = np.array([self.g[gid]['grain'].skprop.solidity
                                           for gid in self.gid])

    def set_mprop_orientation(self):
        self.mprop['orientation'] = np.array([self.g[gid]['grain'].skprop.orientation
                                           for gid in self.gid])

    def set_mprop_inertia_tensor(self):
        self.mprop['inertia_tensor'] = [self.g[gid]['grain'].skprop.inertia_tensor
                                           for gid in self.gid]

    def set_mprop_inertia_tensor_eigvals(self):
        self.mprop['inertia_tensor_eigvals'] = [self.g[gid]['grain'].skprop.inertia_tensor_eigvals
                                           for gid in self.gid]

    def set_conversion_loss(self, refn=None):
        """
        Determine the loss due to converting UPXO.MCGS2D to DefDAP map.

        Parameters
        ----------
        None

        Return
        ------
        cl_details: dict
            Conversion loss details
            Keys:
                loss: conversion_loss value
                n_pre: number of grains before conversion
                n_post: number of grains after conversion
        """
        # Validate user inputs
        # bf = refn
        af = np.unique(self.map.grains).size
        conversion_loss = (refn-af)*100/refn
        cl_details = {'loss': np.round(conversion_loss, 2),
                      'n_pre': refn,
                      'n_post': af}
        cl = cl_details['loss']
        print(40*'-', f"\n MCGS2d-XOMAP conversion loss: {cl} %")
        if conversion_loss > 0:
            print(f'Number of grains before conversion: {refn}')
            print(f'Number of grains after conversion: {af}')
        self.closs = cl_details

    def set_bjp(self):
        '''Boundary Junction Points.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        bjp = {gid: None for gid in self.gid}
        for i in self.gid:
            bjp[i] = np.argwhere(self.gbjp*(self.lgi == i))
        self.bjp = bjp

    def plot_bjp(self):
        '''Plot boundary junction points overlaid on pxtal map.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        plt.figure(figsize=(5, 5), dpi=120)
        plt.imshow(self.lgi)
        for gid in self.bjp.keys():
        	plt.plot(*(self.bjp[gid].T), 'ro', mfc='none')
        plt.colorbar()

    def get_gbpoints_grain(self, gid,
                           retrieval_method='external',
                           chain_approximation='simple'):
        """
        Example
        -------
        gid = 1
        gstslice.pxtal[1].get_gbpoints_grain(gid,
                                             retrieval_method='tree',
                                             chain_approximation='none')
        """
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """

        # Validations
        binary_image = np.where(self.lgi == gid, 255, 0).astype(np.uint8)
        # ----------------------------------------
        if retrieval_method.lower() == 'external':
            method = cv2.RETR_EXTERNAL
        elif retrieval_method.lower() == 'list':
            method = cv2.RETR_LIST
        elif retrieval_method.lower() == 'ccomp':
            method = cv2.RETR_CCOMP
        elif retrieval_method.lower() == 'tree':
            method = cv2.RETR_TREE
        else:
            raise ValueError(f'Invalid retrieval_method: {retrieval_method}')
        # ----------------------------------------
        if chain_approximation == 'none':
            contours, heirarchy = cv2.findContours(binary_image, method,
                                                   cv2.CHAIN_APPROX_NONE)
            ch = {'all': (contours, heirarchy)}
            gb_points = {'all': contours[0].T.squeeze()}
        elif chain_approximation == 'simple':
            contours, heirarchy = cv2.findContours(binary_image, method,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            ch = {'simple': (contours, heirarchy)}
            gb_points = {'simple': contours[0].T.squeeze()}
        elif chain_approximation == 'both':
            contours1, heirarchy1 = cv2.findContours(binary_image, method,
                                                     cv2.CHAIN_APPROX_NONE)
            contours2, heirarchy2 = cv2.findContours(binary_image, method,
                                                     cv2.CHAIN_APPROX_SIMPLE)
            ch = {'all': (contours1, heirarchy1),
                  'simple': (contours2, heirarchy2)
                  }
            gb_points = {'all': contours1[0].T.squeeze(),
                         'simple': contours2[0].T.squeeze()}
        else:
            raise ValueError(f'Invalid chain_approx.: {chain_approximation}')
        # ----------------------------------------

        return gb_points, ch

    def set_grains(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        self.g = {i: {'xo': None, 'grain': grain2d(), 'defdap': g}
                  for i, g in enumerate(self.map.grainList, start=1)}

    def set_lgi(self):
        '''Returns the pixel - grain ID mapping.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # ---- > Equivalent to pxt.gs[tslice].lgi
        self.lgi = deepcopy(self.map.grains)

    def shuffle_lgi_random_gid_wise(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        self.flags['gid_shuffle_lgi_method'] = 'random'
        self.set_gid(reset_lgi=False)
        gidshuffled = np.random.permutation(self.gid)
        lginew = np.zeros_like(self.lgi)
        for i in range(len(self.gid)):
            locs = self.lgi == self.gid[i]
            lginew[locs] = gidshuffled[i]
        # ----------------------------------
        self.gidshuffled = gidshuffled
        self.lgi = lginew

    def rearrange_g_after_lgi_shuffle(self):
        self.flags['rearrange_g_after_lgi_shuffle'] = True
        g = {}
        for gid in self.gid:
            g[gid] = {'xo': self.g[self.gidshuffled[gid-1]]['xo'],
                      'grain': self.g[self.gidshuffled[gid-1]]['grain'],
                      'defdap': self.g[self.gidshuffled[gid-1]]['defdap']
                      }
        self.g = g

    def reset_lgi(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        self.set_lgi()

    def set_gid(self, reset_lgi=False):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        if reset_lgi:
            self.reset_lgi()
        self.gid = np.unique(self.lgi)

    def set_grain_locations(self):
        for gid in self.gid:
            _, L = cv2.connectedComponents(np.array(self.lgi == gid,
                                                    dtype=np.uint8))
            self.g[gid]['grain'].loc = np.argwhere(L == 1)


    def set_n(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        self.n = len(self.gid)

    def find_neigh(self, update_gid=True, reset_lgi=False):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        # FIND THE NEIGHBOURING GRAIN IDs OF EVERY GRAIN
        if update_gid:
            self.set_gid(reset_lgi=reset_lgi)
        # Exclude background or border if labeled as 0 or another spec. value
        gids = self.gid[self.gid != 0]
        # Dictionary to hold the neighbors for each grain ID
        grain_neighbors = {gid: None for gid in gids}
        # Generate a binary structure for dilation (connectivity)
        # 2D connectivity, direct neighbors
        struct = generate_binary_structure(2, 1)
        for gid in gids:
            # Create a binary mask for the current grain
            mask = self.lgi == gid
            # Dilate the mask to include borders with neighbors
            dilated_mask = binary_dilation(mask, structure=struct)
            # Find unique neighboring grain IDs in the dilated area,
            # excluding the current grain ID
            neighbors = np.unique(self.lgi[dilated_mask & ~mask])
            # Update the dictionary, excluding the current grain ID from its
            # neighbors if present
            grain_neighbors[gid] = list(set(neighbors) - {gid})
        self.neigh_gid = grain_neighbors

    def find_gbseg1(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        gbseg = {gid: {} for gid in self.gid}
        # 2D connectivity, direct neighbors
        struct = generate_binary_structure(2, 1)
        for gid in self.gid:
            # Binary mask for the current grain
            gid_mask = self.lgi == gid
            for neigh in self.neigh_gid[gid]:
                # Binary mask for the neighbor
                neigh_mask = self.lgi == neigh
                # Dilate each mask
                dilated_gid_mask = binary_dilation(gid_mask,
                                                   structure=struct)
                dilated_neigh_mask = binary_dilation(neigh_mask,
                                                     structure=struct)
                # Intersection of dilated masks with the original of the other
                # to find boundary
                boundary_gid_to_neigh = np.where((dilated_gid_mask & neigh_mask))
                boundary_neigh_to_gid = np.where((gid_mask & dilated_neigh_mask))
                # Store the boundary locations as a list of tuples (y, x)
                # positions
                # Choose boundary_gid_to_neigh or boundary_neigh_to_gid
                gbseg[gid][neigh] = list(zip(boundary_gid_to_neigh[0],
                                             boundary_gid_to_neigh[1]))
        self.gbseg1 = gbseg

    def extract_gb_discrete(self, retrieval_method='external',
                            chain_approximation='simple'):
        """
        Extract grain boundaries of every grain.
        """
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        rm, ca = retrieval_method, chain_approximation
        self.gb_discrete = {gid: {} for gid in self.gid}
        for gid in self.gid:
            gb_points, ch = self.get_gbpoints_grain(gid,
                                                    retrieval_method=rm,
                                                    chain_approximation=ca)
            self.gb_discrete[gid]['gb_points'] = gb_points
            self.gb_discrete[gid]['ch'] = ch

    def set_geom(self):
        from upxo.pxtal.geometrification import polygonised_grain_structure
        self.geom = polygonised_grain_structure(self.lgi,
                                                self.gid,
                                                self.neigh_gid)

    def find_lgi_subset_neigh(self, gid,
                              plot=True,
                              plot_kwargs={'recalc_centroids': False,},
                              cmap_name='coolwarm',
                              plot_centroids=True,
                              add_gid_text=True,
                              plot_gbseg=False
                              ):
        '''
        Example
        -------
        self.find_largest_grain
        '''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        _ = self.lgi == gid  # Mask of the grain
        _ = np.logical_or(_, self.lgi == self.neigh_gid[gid][0])
        if len(self.neigh_gid[gid]) > 1:
            for neigh in self.neigh_gid[gid][1:]:
                _ = np.logical_or(_, self.lgi == neigh)
        gid_neighbourhood = self.lgi * _
        if plot:
            ax = self.plot_grains_gids(self.neigh_gid[gid] + [gid],
                                       gclr='color',
                                       title=f"Neighbourhood of gid: {gid}",
                                       cmap_name=cmap_name,
                                       plot_centroids=plot_centroids,
                                       add_gid_text=add_gid_text,
                                       plot_gbseg=plot_gbseg
                                       )
        return gid_neighbourhood

    def remove_single_pixel_grains(self):
        spg_gids = self.single_pixel_grains
        if spg_gids:
            pass
        else:
            print('There are no single pixel grains to remove.')

    @property
    def single_pixel_grains(self):
        locations = np.where(self.mprop['npixels'] == 1)[0]
        if locations.size > 0:
            return list(locations + 1)
        else:
            return []

    @property
    def plot_single_pixel_grains(self):
        self.plot_grains_gids(self.single_pixel_grains)

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
        if return_type == 'dict':
            return {mpn: mpn in self.mprop.keys() for mpn in mpnames}
        elif return_type in ('list', 'tuple'):
            return [mpn in self.mprop.keys() for mpn in mpnames]

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

    def get_mprops(self, mpnames, set_missing_mprop=False):
        """
        Get values of mpnames.

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
                    mprop_values[mpn] = self.mprop[mpn]
                else:
                    mprop_values[mpn] = None
        # ----------------------------
        if set_missing_mprop:
            set_propnames = [mpn for mpn in mpnames if not val_exists[mpn]]
            self.set_mprops()
            mprop_values = self.get_mprops(mpnames, set_missing_mprop=False)

        return mprop_values

    def get_gids_in_params_bounds(self,
                                  search_gid_source='all',
                                  search_gids=None,
                                  mpnames=['area', 'aspect_ratio',
                                           'perimeter', 'solidity'],
                                  fx_stats=[np.mean, np.mean, np.mean, np.mean],
                                  pdslh=[[50, 50], [50, 50], [50, 50], [50, 50]],
                                  param_priority=[1, 2, 3, 2],
                                  plot_mprop=False
                                  ):
        """
        pdslh: Percentages of distance from stat to minimum and stat to maximum.

        Example
        -------
        """
        # Validations
        # ---------------------------
        pname_val = self.validate_propnames(mpnames, return_type='dict')
        mprop_values = self.get_mprops(mpnames, set_missing_mprop=False)
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
        # Collate the GID related results.
        GIDs = {'intersection': intersection,
                'union': union,
                'presence': counts,
                'mpmapped': mpinds_gids}
        # Collate the Values and Indices related results.
        VALIND = {'stat': mp_stats,
                  'statmap': fx_stats,
                  'bounds': mp_bounds,
                  'indices': mpinds}
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

    def get_upto_nth_order_neighbors(self, grain_id, neigh_order,
                                     recalculate=False, include_parent=True,
                                     output_type='list', plot=False):
        """
        Calculates the nth order neighbors for a given cell ID.

        Args:
            cell_id: The ID of the cell for which to find neighbors.
            n: The order of neighbors to calculate (1st order, 2nd order, etc.).

        Returns:
            A set containing the nth order neighbors.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 20
        gstslice = pxt.gs[tslice]
        gstslice.export_ctf(r'D:\export_folder', 'sunil')
        fname = r'D:\export_folder\sunil'
        gstslice.set_pxtal(instance_no=1, path_filename_noext=fname)
        IN, gid, neigh_order = 1, 1, 2
        gstslice.pxtal[IN].get_upto_nth_order_neighbors(gid, neigh_order,
                                                       recalculate=False,
                                                       include_parent=True,
                                                       output_type='list',
                                                       plot=True)
        # NOTE: In the above, IN refers to Instance number
        """
        if neigh_order == 0:
            return grain_id
        if recalculate or not self.neigh_gid:
            self.find_neigh(update_gid=True, reset_lgi=False)
        # Start with 1st-order neighbors
        neighbors = set(self.neigh_gid.get(grain_id, []))

        for _ in range(neigh_order - 1):
            new_neighbors = set()
            for neighbor in neighbors:
                new_neighbors.update(self.neigh_gid.get(neighbor, []))
            neighbors.update(new_neighbors)

        if not include_parent:
            neighbors.discard(grain_id)

        if plot:
            self.plot_grains_gids(list(neighbors), cmap_name='coolwarm')

        if output_type == 'list':
            return list(neighbors)
        if output_type == 'nparray':
            return np.array(list(neighbors))
        elif output_type == 'set':
            return neighbors


    def get_nth_order_neighbors(self, grain_id, neigh_order,
                                recalculate=False, include_parent=True,
                                plot=False):
        """
        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 20
        gstslice = pxt.gs[tslice]
        gstslice.export_ctf(r'D:\export_folder', 'sunil')
        fname = r'D:\export_folder\sunil'
        gstslice.set_pxtal(instance_no=1, path_filename_noext=fname)
        IN, gid, neigh_order = 1, 1, 2
        gstslice.pxtal[IN].get_nth_order_neighbors(gid, neigh_order,
                                    recalculate=False, include_parent=True,
                                    plot=True)
        # NOTE: In the above, IN refers to Instance number
        """
        neigh_upto_n_minus_1 = self.get_upto_nth_order_neighbors(grain_id,
                                                                 neigh_order-1,
                                                                 include_parent=include_parent,
                                                                 output_type='set',
                                                                 plot=False)
        if type(neigh_upto_n_minus_1) in dth.dt.NUMBERS:
            neigh_upto_n_minus_1 = set([neigh_upto_n_minus_1])

        neigh_upto_n = self.get_upto_nth_order_neighbors(grain_id, neigh_order,
                                                         include_parent=include_parent,
                                                         output_type='set',
                                                         plot=False)
        if type(neigh_upto_n) in dth.dt.NUMBERS:
            neigh_upto_n = set([neigh_upto_n])

        neighbours = list(neigh_upto_n.difference(neigh_upto_n_minus_1))

        if plot:
            self.plot_grains_gids(neighbours, cmap_name='coolwarm')
        return neighbours

    def get_upto_nth_order_neighbors_all_grains(self, neigh_order,
                                                recalculate=False,
                                                include_parent=True,
                                                output_type='list'):
        """
        Calculates the nth order neighbors for a given cell ID.

        Args:
            cell_id: The ID of the cell for which to find neighbors.
            n: The order of neighbors to calculate (1st order, 2nd order, etc.).

        Returns:
            A set containing the nth order neighbors.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 20
        gstslice = pxt.gs[tslice]
        gstslice.export_ctf(r'D:\export_folder', 'sunil')
        fname = r'D:\export_folder\sunil'
        gstslice.set_pxtal(instance_no=1, path_filename_noext=fname)
        IN, neigh_order = 1, 2
        gstslice.pxtal[IN].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,
                                                             output_type='list')
        # NOTE: In the above, IN refers to Instance number
        """
        neighs_upto_nth_order = {gid: self.get_upto_nth_order_neighbors(gid,
                                                                        neigh_order,
                                                                        recalculate=recalculate,
                                                                        include_parent=include_parent,
                                                                        output_type='list',
                                                                        plot=False)
                                 for gid in self.gid}
        return neighs_upto_nth_order

    def get_nth_order_neighbors_all_grains(self, neigh_order,
                                           recalculate=False,
                                           include_parent=True):
        """
        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 20
        gstslice = pxt.gs[tslice]
        gstslice.export_ctf(r'D:\export_folder', 'sunil')
        fname = r'D:\export_folder\sunil'
        gstslice.set_pxtal(instance_no=1, path_filename_noext=fname)
        IN, neigh_order = 1, 2
        gstslice.pxtal[IN].get_nth_order_neighbors_all_grains(neigh_order,
                                                        recalculate=False,
                                                        include_parent=True)
        # NOTE: In the above, IN refers to Instance number
        """
        neighs_nth_order = {gid: self.get_nth_order_neighbors(gid,
                                                              neigh_order,
                                                              recalculate=recalculate,
                                                              include_parent=include_parent,
                                                              plot=False)
                            for gid in self.gid}
        return neighs_nth_order

    def get_upto_nth_order_neighbors_all_grains_prob(self, neigh_order,
                                                     recalculate=False,
                                                     include_parent=False,
                                                     print_msg=False):
        """
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
                neigh_at_high = on_neigh_all_grains_at(no_low + 1,
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
        if check_for_neigh and not self.check_for_neigh(parent_gid, other_gid):
            # print('Check for neigh failed. Nothing merged.')
            merge_success = False
        # ---------------------------------------
        if any((check_for_neigh, self.check_for_neigh(parent_gid, other_gid))):
            merge_success = MergeGrains()
            # print(f"Grain {other_gid} merged with grain {parent_gid}.")
        return merge_success

    def get_grain_pixel_coords(self, gid):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        locs = self.g[gid]['grain'].coordList
        return locs

    def set_grain_centroids(self, coord_source='upxo_lgi'):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        centroids = []
        if coord_source == 'defdap_grain':
            for gid in self.gid:
                locs = self.get_grain_pixel_coords(gid)
                centroids.append(np.array(locs).T.sum(axis=1)/len(locs))
        elif coord_source == 'upxo_lgi':
            for gid in self.gid:
                centroids.append(np.argwhere(self.lgi == gid).mean(axis=0))
            centroids = np.array([[c[0], c[1]] for c in centroids])
            centroids[:, [0, 1]] = centroids[:, [1, 0]]
            self.centroids = centroids
        self.centroids = np.array([[c[0], c[1]] for c in centroids])

    def get_grain_centroid(self, gid, recalculate=True):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validate gid
        if recalculate:
            locs = self.get_grain_pixel_coords(gid)
            centroid = np.array(locs).T.sum(axis=1)/len(locs)
        else:
            centroid = self.centroid[gid-1]
        return centroid

    def find_largest_grain(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return np.argmax(self.npixels) + 1

    def __setup__positions__(self):
        self.positions = {'top_left': [], 'bottom_left': [],
                          'bottom_right': [], 'top_right': [],
                          'pure_right': [], 'pure_bottom': [],
                          'pure_left': [], 'pure_top': [],
                          'left': [], 'bottom': [], 'right': [], 'top': [],
                          'boundary': [], 'corner': [], 'internal': [], }

    def plot_data_imshow_and_get_axis(self, data):
        """
        fig, im, ax = self.plot_data_imshow_and_get_axis()
        """
        fig, ax = plt.subplots(1, figsize=(5, 5), dpi=120)
        im=ax.imshow(data)
        fig.colorbar(im, ax)
        return fig, im, ax

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
            Dictionary with keys being a subset of self.gid. Each key must have
            a single numeric or bool value.
        default_scalar: int
            Defauts to -1.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 20
        pxt.char_morph_2d(tslice)
        gstslice = pxt.gs[tslice]
        gstslice.export_ctf(r'D:\export_folder', 'sunil')
        # ---------------------------
        fname = r'D:\export_folder\sunil'
        gstslice.set_pxtal(instance_no=1, path_filename_noext=fname)
        gstslice.pxtal[1].find_gbseg1()
        GIDs, VALIND = gstslice.pxtal[1].get_gids_in_params_bounds(search_gid_source='all',
                                                                   search_gids=None,
                                                                   mpnames=['area', 'solidity'],
                                                                   fx_stats=[np.mean,np.mean],
                                                                   pdslh=[[20, 20], [5, 5]],
                                                                   param_priority=[1, 2, 3, 2],
                                                                   plot_mprop=False)
        gstslice.pxtal[1].map_scalar_to_lgi(GIDs['presence'], default_scalar=-1,
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
            fig, im, ax = self.plot_data_imshow_and_get_axis(LGI)
            # VMIN, VMAX = min(scalars_dict.values()), max(scalars_dict.values())
            plt.figure(figsize=(5, 5), dpi=120)
            plt.imshow(LGI, cmap='viridis')
            # self.plot_grain_centroids(gids, ax, add_gid_text=add_gid_text)
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

    @property
    def get_csym(self):
        '''Get crystal symmetry.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.crystalSym

    @property
    def get_pphid(self):
        '''Get primary phase ID.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.primaryPhaseID

    @property
    def get_scale(self):
        '''Get scale.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.scale

    @property
    def get_shape(self):
        '''Get shape of the map.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.shape

    @property
    def get_xdim(self):
        '''Get xdim.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.xDim

    @property
    def get_ydim(self):
        '''Get ydim.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.yDim

    @property
    def get_nph(self):
        '''Get number of phases.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.numPhases

    @property
    def get_glist(self):
        '''Get grain list.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.grainList

    @property
    def get_bc(self):
        '''Get band contra6t of map.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.bandContrastArray

    @property
    def get_bs(self):
        '''Get band slope.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.bandSlopeArray

    @property
    def get_boundaries(self):
        '''Get grain boundaries.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.boundaries

    @property
    def get_boundariesX(self):
        '''Get x-grad identified grain boundatry points.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.boundariesX

    @property
    def get_boundariesY(self):
        '''Get y-grad identified grain boundatry points.'''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.boundariesY

    @property
    def get_q(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.quatArray

    @property
    def get_ea(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.eulerAngleArray

    @property
    def get_coord(self):
        # TODO: DEBUG THIS
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        return self.map.grainList[0].coordList

    def plot_grains(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        self.map.plotGrainMap()

    def plot_gb(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        self.map.plotBoundaryMap()

    def plot_phase(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        self.map.plotPhaseMap()

    def plot_bc(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        self.map.plotBandContrastMap()

    def plot_ea(self):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        self.map.plotEulerMap()

    def plotIPFMap(self, direction):
        '''
        Example
        -------
        pxt.gs[tslice].xomap_plotIPFMap([1, 0, 0])
        '''
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        self.map.plotIPFMap(direction)

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
        # Validations
        # -----------------------------------------
        lgi_masked = deepcopy(self.lgi).astype(int)
        for gid in self.gid:
            if gid not in gids:
                lgi_masked[lgi_masked == gid] = masker
        # -----------------------------------------
        return lgi_masked, masker

    def get_bbox_bounds_gids(self, gids, plot=True, cmap_name='viridis',
                             plot_centroids=True, add_gid_text=True,
                             plot_gbseg=False,):
        gids = gstslice.pxtal[1].neigh_gid[46]
        a = gstslice.pxtal[1].mask_lgi_with_gids(gids, masker=-10)
        yloc, xloc = np.argwhere(a[0] != -10).T
        xbounds = [yloc.min(), yloc.max()]
        ybounds = [xloc.min(), xloc.max()]
        # ------------------------------------
        fig, ax = plt.subplots(1, figsize=(5, 5), dpi=120)
        im = ax.imshow(a[0][xbounds[0]:xbounds[1], ybounds[0]:ybounds[1]])
        return fig, ax, im

    def polygonize_voronoi_grid(grid):
        """
        Polygonizes grains in self.lgi and returns a shapely MultiPolygon.

        Parameters
        ----------
        grid: A NumPy array representing the integer grid values.

        Returns
        -------
        rioshapes
        polygons: list of shapely polygon objects of each grain
        multi_polygon: shapely multi-polygon object
        """
        import rasterio
        from shapely.geometry import shape as ShShape
        from shapely.geometry import MultiPolygon
        rioshapes = rasterio.features.shapes
        # Create a raster dataset from the grid array
        with rasterio.Env():
            profile = rasterio.profiles.DefaultGTiffProfile()
            profile.update(width=grid.shape[1],
                           height=grid.shape[0], count=1,
                           dtype=grid.dtype,
                           transform=rasterio.transform.Affine.identity())
            with rasterio.MemoryFile() as memfile:
                with memfile.open(**profile) as dataset:
                    dataset.write(grid, 1)
                    # Find unique cell IDs; same as self.gid
                    gids = np.unique(grid)
                    # Polygonize each unique cell
                    polygons = []
                    RESULTS = []
                    for gid in gids:
                        mask = (grid == gid).astype(np.uint8)
                        results = list(rioshapes(mask, mask=mask,
                                                 transform=dataset.transform))
                        if results:
                            RESULTS.append(results)
                            # Convert to Shapely polygons and append
                            polygons.extend([ShShape(geom[0])
                                             for geom in results])

        # Create a MultiPolygon from the collected polygons
        multi_polygon = MultiPolygon(polygons)

        return RESULTS, polygons, multi_polygon


    def plot_grains_gids(self, gids, add_points=True, points=None, gclr='color', title="user grains",
                         cmap_name='viridis', plot_centroids=True,
                         add_gid_text=True, plot_gbseg=False,
                         bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                     'mec': 'black', 'ms': 2.5},
                         addpoints_kwargs={'marker': 'x', 'mfc': 'black',
                                     'mec': 'black', 'ms': 5}
                         ):
        """
        Parameters
        ----------
        gids : int/list
            Either a single grain index number or list of them
        title : TYPE, optional
            DESCRIPTION. The default is "user grains".
        gclr :

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
        # Validations
        if not dth.IS_ITER(gids):
            gids = [gids]
        # -------------------------------
        # Validtions
        # -------------------------------
        if gclr not in ('binary', 'grayscale'):
            lgi_masked, masker = self.mask_lgi_with_gids(gids)
            fig, ax = plt.subplots(1, figsize=(5, 5), dpi=120)
            im = ax.imshow(lgi_masked, cmap=cmap_name, vmin=1)
        # -------------------------------
        if gclr in ('binary', 'grayscale'):
            lgi_masked, masker = self.mask_lgi_with_gids(gids, masker=-10)
            lgi_masked[lgi_masked != 0] = 1
            fig, ax = plt.subplots(1, figsize=(5, 5), dpi=120)
            im = ax.imshow(lgi_masked, cmap='gray_r', vmin=0, vmax=1)
        # -------------------------------
        fig.colorbar(im, ax=ax)
        # -------------------------------
        if plot_centroids:
            self.plot_grain_centroids(gids, ax, add_gid_text=add_gid_text)
        # -------------------------------
        if plot_gbseg:
            self.plot_contour_grains_gids(gids,
                                          simple_all_preference='simple',
                                          new_fig=False, ax=ax,
                                          bjp_kwargs={'marker': bjp_kwargs['marker'],
                                                      'mfc': bjp_kwargs['mfc'],
                                                      'mec': bjp_kwargs['mec'],
                                                      'ms': bjp_kwargs['ms']}
                                          )
        if add_points:
            ax.plot(points[:, 0], points[:, 1],
                    marker=addpoints_kwargs['marker'],
                    mfc=addpoints_kwargs['mfc'],
                    mec=addpoints_kwargs['mec'],
                    ms=addpoints_kwargs['ms'])
        # -------------------------------
        ax.set_title(title)
        ax.set_xlabel(r"X-axis, $\mu m$", fontsize=12)
        ax.set_ylabel(r"Y-axis, $\mu m$", fontsize=12)
        return ax

    def plotgs(self, figsize=(6, 6), dpi=120,
               custom_lgi=None,
               cmap='coolwarm', plot_cbar=True,
               title='Title',
               plot_centroid=False, plot_gid_number=False, plot_bjp=True,
               centroid_kwargs={'marker': 'o',
                                'mfc': 'yellow',
                                'mec': 'black',
                                'ms': 2.5},
               gid_text_kwargs={'fontsize': 10},
               title_kwargs={'fontsize': 10},
               label_kwargs={'fontsize': 10}
               ):
        """
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
        # Validations
        if custom_lgi is None:
            LGI = self.lgi
        else:
            LGI = custom_lgi
        # ---------------------------------
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(LGI, cmap=cmap)
        # ---------------------------------
        if plot_centroid:
            _marker_, _mfc_ = centroid_kwargs['marker'], centroid_kwargs['mfc']
            _mec_, _ms_ = centroid_kwargs['mec'], centroid_kwargs['ms']
            for gid in self.gid:
                plt.plot(self.centroids[gid-1][0], self.centroids[gid-1][1],
                         linestyle='None', marker=_marker_, mfc=_mfc_,
                         mec=_mec_, ms=_ms_)
        # ---------------------------------
        if plot_gid_number:
            _fs_ = gid_text_kwargs['fontsize']
            for gid in self.gid:
                plt.text(self.centroids[gid-1][0],
                         self.centroids[gid-1][1],
                         str(gid), fontsize=_fs_)
        # ---------------------------------
        plt.xlabel(r"X-axis, $\mu m$", fontsize=label_kwargs['fontsize'])
        plt.ylabel(r"Y-axis, $\mu m$", fontsize=label_kwargs['fontsize'])
        # ---------------------------------
        if plot_cbar:
            plt.colorbar()

    def plot_gb_discrete(self, cmap='coolwarm',
                         bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                           'mec': 'black', 'ms': 2.5},
                         simple_all_preference='simple',
                         add_centroids=True, add_gid_text=True,
                         return_axis=False, ):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        _marker_, _mfc_ = bjp_kwargs['marker'], bjp_kwargs['mfc']
        _mec_, _ms_ = bjp_kwargs['mec'], bjp_kwargs['ms']
        # ---------------------------------
        fig, ax = plt.subplots(1, figsize=(5, 5), dpi=120)
        ax.imshow(self.lgi, cmap=cmap)
        self.plot_contour_grains_all(simple_all_preference=simple_all_preference,
                                 new_fig=False, ax=ax,
                                 bjp_kwargs={'marker': bjp_kwargs['marker'],
                                             'mfc': bjp_kwargs['mfc'],
                                             'mec': bjp_kwargs['mec'],
                                             'ms': bjp_kwargs['ms']}
                                 )
        # ---------------------------------
        if add_centroids:
            self.plot_grain_centroids(self.gid, ax, add_gid_text=add_gid_text)
        # ---------------------------------
        if return_axis:
            return ax

    def plot_contour_grains_all(self, simple_all_preference='simple',
                                new_fig=True, ax=None,
                                bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                            'mec': 'black', 'ms': 2.5}):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        self.plot_contour_grains_gids(self.gid,
                                      simple_all_preference=simple_all_preference,
                                      new_fig=new_fig, ax=ax,
                                      bjp_kwargs={'marker': bjp_kwargs['marker'],
                                                  'mfc': bjp_kwargs['mfc'],
                                                  'mec': bjp_kwargs['mec'],
                                                  'ms': bjp_kwargs['ms']}
                                      )

    def plot_contour_grains_gids(self, gids, simple_all_preference='simple',
                                 new_fig=True, ax=None,
                                 bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                             'mec': 'black', 'ms': 2.5}):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        for gid in gids:
            self.plot_contour_grain_gid(gid,
                                        simple_all_preference=simple_all_preference,
                                        new_fig=new_fig, ax=ax,
                                        bjp_kwargs={'marker': bjp_kwargs['marker'],
                                                    'mfc': bjp_kwargs['mfc'],
                                                    'mec': bjp_kwargs['mec'],
                                                    'ms': bjp_kwargs['ms']}
                                        )

    def plot_contour_grain_gid(self, gid, simple_all_preference='simple',
                           new_fig=True, ax=None,
                           bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                       'mec': 'black', 'ms': 2.5}):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        _marker_, _mfc_ = bjp_kwargs['marker'], bjp_kwargs['mfc']
        _mec_, _ms_ = bjp_kwargs['mec'], bjp_kwargs['ms']
        # --------------------------------
        if new_fig:
            fig, ax = plt.subplots(1, figsize=(5, 5), dpi=120)
        # --------------------------------
        KEYS = self.gb_discrete[gid]['gb_points'].keys()
        if 'simple' in KEYS and 'all' not in KEYS:
            contour = self.gb_discrete[gid]['gb_points']['simple']
        elif 'simple' not in KEYS and 'all' in KEYS:
            contour = self.gb_discrete[gid]['gb_points']['all']
        elif 'simple' in KEYS and 'all' in KEYS:
            if simple_all_preference in ('simple', 'all'):
                pref = simple_all_preference
                contour = self.gb_discrete[gid]['gb_points'][pref]
        # --------------------------------
        ax.plot(contour[0], contour[1],  linestyle='-',
                color='blue', marker=_marker_, mfc=_mfc_,
                mec=_mec_, ms=_ms_)
        # --------------------------------
        if contour.ndim == 2:
            ax.plot([contour[0][-1], contour[0][0]],
                    [contour[1][-1], contour[1][0]],
                    linestyle='-', color='blue',
                    marker=_marker_, mfc='darkred',
                    mec='darkred', ms=_ms_)

    def plot_grain_centroid(self, gid, axis, add_gid_text=True):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        axis.plot(self.centroids[gid-1][0], self.centroids[gid-1][1],
                  marker='.', mfc='w', mec='k', ms=5)
        if add_gid_text:
            axis.text(self.centroids[gid-1][0], self.centroids[gid-1][1],
                      gid, fontsize=8)

    def plot_grain_centroids(self, gids, axis, add_gid_text=True):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        for gid in gids:
            self.plot_grain_centroid(gid, axis, add_gid_text=add_gid_text)
