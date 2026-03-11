import os
import random
import matplotlib as mpl
from copy import deepcopy
from typing import Iterable
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import cv2
import vtk
import warnings
import vedo as vd
import pyvista as pv
from numba import njit
from scipy.spatial import cKDTree
from scipy.ndimage import zoom
# from skimage.measure import label as skim_label
import seaborn as sns
from functools import partial
from matplotlib.figure import Figure
from skimage.segmentation import find_boundaries
from upxo.geoEntities.plane import Plane
from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
# from upxo._sup.console_formats import print_incrementally
from upxo._sup import dataTypeHandlers as dth
from upxo._sup.gops import att
from upxo._sup.data_templates import dict_templates
# from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure
import upxo._sup.data_ops as DO
from upxo.viz.plot_artefacts import cuboid_data
from upxo.viz.helpers import arrange_subplots

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

    mprop: morphhological properties. Type: dict
    """

    __slots__ = ('dim', 'uigrid', 'uimesh', 'm', 's', 'S', 'ndimg_label_pck',
                 'binaryStructure3D', 'n', 'lgi', 'fdb',
                 '_ckdtree_', '_upxo_mp3d_', 'domain_volume',
                 'spart_flag', 'gid', 's_gid', 'gid_s', 's_n', 'g', 'gb',
                 'positions', 'mp', 'vox_size', 'gid_twin',
                 'prop_flag', 'prop', 'are_properties_available', 'prop_stat',
                 '__gi__', '__ui', 'info',
                 'pvgrid', 'ellfits', 'skimrp', 'sssr',
                 'valid_scalar_fields', 'pointclouds_pv', 'mprop', 'lgi_slice',
                 'grain_locs', 'gpos', 'spbound', 'spboundex', 'gid_imap_keys',
                 'gid_imap', 'neigh_gid', 'Lgbp_all', 'Ggbp_all',
                 'gbpstack', 'gbpids', 'gbp_id_maps', 'gbp_ids',
                 'gid_pair_ids', 'gid_pair_ids_rev',
                 'gid_pair_ids_unique_lr', 'gid_pair_ids_unique_rl',
                 'gbsurf_pids_vox', 'gid_pair_gbp_IDs', 'gid_pair_gbp_coords',
                 'gid_gpid', 'triples', 'ctrls')
    EPS, __maxGridSizeToIgnoreStoringGrids = 1e-1, 200**3
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

    def __init__(self, dim=3, m=None, uidata=None, vox_size=None, S_total=None,
                 uigrid=None, uimesh=None, ndimg_label_pck=None,
                 instantiation_route='regular',
                 user_data=None, user_data_name='s'):
        self.ndimg_label_pck = ndimg_label_pck
        self.__ui = uidata
        self.dim, self.m, self.S,    self.vox_size = dim, m, S_total, vox_size
        self.uigrid, self.uimesh = uigrid, uimesh
        self.set__spart_flag(S_total)
        self.set__s_gid(S_total)
        self.set__gid_s()
        self.set__s_n(S_total)
        self.g, self.gb, self.info, self.ctrls = {}, {}, {}, {}
        # ------------------------------------
        self._ckdtree_ = cKDTree
        self._upxo_mp3d_ = mp3d
        # ------------------------------------
        self.mp = dict_templates.mulpnt_gs3d
        # ------------------------------------
        self.are_properties_available = False
        self.__setup__positions__()
        # ------------------------------------
        self.pvgrid = None
        self.valid_scalar_fields = ["lgi", "s", "fid"]
        self.pointclouds_pv = {'gbp_global': None, 'jp_global': None,
                               'gbp_grain': None,}
        # ------------------------------------
        """
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
        self.skimrp = None
        self.mprop = {'volnv': None, 'volsr': None, 'volch': None,
                      'sanv': None, 'savi': None, 'sasr': None, 'psa': None,
                      'pernv': None, 'pervl': None, 'pergl': None,
                      'eqdia': None, 'feqdia': None,
                      'kx': None, 'ky': None, 'kz': None, 'kxyz': None,
                      'ksr': None,
                      'arbbox': None, 'arellfit': None,
                      'sol': None, 'ecc': None, 'com': None, 'sph': None,
                      'fn': None, 'rnd': None, 'mi': None, 'fdim': None,
                      'rat_sanv_volnv': None, }
        # ------------------------------------
        self.grain_locs = {}
        self.gpos = {'internal': None, 'boundary': None, 'corner': None,
                     'face': None, 'edges': None}
        self.set_gid_imap_keys()
        self.ellfits = None
        self.sssr = {}
        self.gid_twin = None
        # ------------------------------------
        self.fdb = {}
        xlength = np.arange(self.uigrid.xmin, self.uigrid.xmax, self.uigrid.xinc).size
        ylength = np.arange(self.uigrid.ymin, self.uigrid.ymax, self.uigrid.yinc).size
        zlength = np.arange(self.uigrid.zmin, self.uigrid.zmax, self.uigrid.zinc).size
        self.domain_volume = xlength * ylength * zlength
        # ------------------------------------
        if instantiation_route == 'direct':
            self.ctrls['instantiation_route'] = instantiation_route
            self.ctrls['user_data_name'] = user_data_name
            if user_data_name in ('s', 'state'):
                self.s = user_data
            elif user_data_name in ('lgi', 'fid'):
                self.lgi = user_data
        elif instantiation_route == 'regular':
            pass
        # ------------------------------------

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
    def by_data(cls, data, data_name='s', dim=3, m=0,
                xmin=0.0, xinc=1.0, xmax=100.0,
                ymin=0.0, yinc=1.0, ymax=100.0, zmin=0.0, zinc=1.0,
                zmax=100.0, S_total=-1, nvoxels_max=1.01E9):
        r"""
        Instantiate temporal slice using just 3D Monte-Carlo state value array.

        This allows you to exploit the entire module for the input s-array. You
        can start off another branch of simulations from this, enabling the
        following thisngs:
            1. switching across algorithms.
            2. Differently evolve a local subdomain and plug it back into the
            parent domain.

        Parameters
        ----------
        data: numpy.ndarray
            3D grain structue image data.

        dim: int, optional
            Dimensionality of the problem. Defaults to 3.

        m: int, optional
            A user desired value of tempoal slice number. This is to ensure
            we have a starting point for a new Monte-Carlo simula\tion to
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
            Increment step of y-axis. Defaults to 1.0

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
        tslice = 49
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=True,
                                           find_spatial_bounds_of_grains=True,
                                           force_compute=True)

        p, q, r = 5, 5, 10
        # ------------------------------------------------------
        # USING fid
        A = gstslice.extract_subdomains_random(p=p, q=q, r=r, n=2,
                                               feature_name='base',
                                               make_pvgrids=False
                                               )

        from upxo.pxtal.mcgs3_temporal_slice import mcgs3_grain_structure

        sd = mcgs3_grain_structure.by_data(A['sd'][0], data_name='fid',
                                           dim=3, m=tslice,
                                           xmin=0.0, xinc=1.0, xmax=r,
                                           ymin=0.0, yinc=1.0, ymax=q,
                                           zmin=0.0, zinc=1.0, zmax=p,
                                           S_total=gstslice.S,
                                           nvoxels_max=1.01E9,)
        sd.char_morphology_of_grains(label_str_order=1,
                                           find_grain_voxel_locs=True,
                                           find_spatial_bounds_of_grains=True,
                                           force_compute=True)

        """
        from dataclasses import dataclass
        from scipy.ndimage import label as spndimg_label
        # from upxo.pxtal.mcgs3_temporal_slice import mcgs3_grain_structure

        from upxo.misc import make_belief
        uigrid = make_belief.uigrid(dim=dim, npixels_max=nvoxels_max,
                                    xmin=xmin, xinc=xinc, xmax=xmax,
                                    ymin=ymin, yinc=yinc, ymax=ymax,
                                    zmin=zmin, zinc=zinc, zmax=zmax)

        return cls(dim=dim, m=m,
                   uidata=None,
                   vox_size=(xinc, yinc, zinc),
                   S_total=S_total,
                   uigrid=uigrid,
                   uimesh=None,
                   ndimg_label_pck=spndimg_label,
                   instantiation_route='direct',
                   user_s=data)

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


        Parameters
        ----------
        S_total : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # VALIDATIONS
        # -----------------------------
        self.s_gid = {s: None for s in range(1, S_total+1)}

    def set__gid_s(self):
        """


        Returns
        -------
        None.

        """
        self.gid_s = []

    def set__spart_flag(self, S_total,):
        """


        Parameters
        ----------
        S_total : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # VALIDATIONS
        # -----------------------------
        self.spart_flag = {_s_: False for _s_ in range(1, S_total+1)}

    def get_binaryStructure3D(self):
        return self.binaryStructure3D

    def set_binaryStructure3D(self, n=3):
        """Set value of the binary structure type for grain identification."""
        # VALIDATIONS
        # -----------------------------
        if n in (1, 2, 3):
            self.binaryStructure3D = generate_binary_structure(3, n)
        else:
             print('Invalid binary structure-3D. n must be in (1, 2, 3). Value not set')

    def char_morphology_of_grains(self,
                                  label_str_order=1,
                                  ngrains_max=1E3,
                                  make_pvgrid=False,
                                  find_neigh=[False, [1]],
                                  find_grain_voxel_locs=False,
                                  find_spatial_bounds_of_grains=False,
                                  find_grain_locations=False,
                                  force_compute=False,
                                  extra_sf={'sfname1': None,
                                            'sfname2': None,}
                                  ):
        """
        Characterize the 3D morphology of the grain structure.

        Parameters
        ----------
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
                find_neigh[1]: List of Natural numbers representing the neigh
                    order values needed. The default self.neigh variable shall
                    contain neighbouring grain informatoipm only for order 1.
                    Other order neighbouir grain datas will be contained in a
                    different dictionary.
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
        # VALIDATIONS
        # -----------------------------
        self.find_grains(label_str_order=label_str_order, pck=self.ndimg_label_pck)
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
                # Function call
            # -----------------
            if find_grain_voxel_locs:
                # gstslice.grain_locs
                self.find_grain_voxel_locs()
            # -----------------
            if find_spatial_bounds_of_grains:
                # gstslice.spbound, gstslice.spboundex
                self.find_spatial_bounds_of_grains()
            # -----------------
            if find_grain_locations:
                if not find_grain_voxel_locs:
                    self.find_grain_voxel_locs()
                self.set_grain_positions()

    def set_skimrp(self):
        """Set the region properties of the scikit image."""
        from skimage.measure import regionprops
        self.skimrp = {}
        for gid in self.gid:
            self.skimrp[gid] = regionprops(1*(self.lgi == gid))[0]

    def set_mprops(self, volnv=True, eqdia=False, eqdia_base_size_spec='volnv',
                   arbbox=False, arbbox_fmt='gid_dict',
                   arellfit=False, arellfit_metric='max',
                   arellfit_calculate_efits=True,
                   arellfit_efit_routine=1,
                   arellfit_efit_regularize_data=True,
                   solidity=True, sol_nan_treatment='replace',
                   sol_inf_treatment='replace',
                   sol_nan_replacement=-1, sol_inf_replacement=-1):
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
        # VALIDATIONS
        # -----------------------------
        self.set_skimrp()
        # ----------------
        if volnv:
            self.set_mprop_volnv()
        # ----------------
        if eqdia:
            self.set_mprop_eqdia(base_size_spec='volnv')
        # ----------------
        if solidity:
            self.set_mprop_solidity(reset_generators=False,
                                    nan_treatment=sol_nan_treatment,
                                    inf_treatment=sol_inf_treatment,
                                    nan_replacement=sol_nan_replacement,
                                    inf_replacement=sol_inf_replacement)
        # ----------------
        if arbbox:
            self.set_mprop_arbbox(fmt=arbbox_fmt)
        # ----------------
        if arellfit:
            self.set_mprop_arellfit(metric=arellfit_metric,
                                    calculate_efits=arellfit_calculate_efits,
                                    efit_routine=arellfit_efit_routine,
                                    efit_regularize_data=arellfit_efit_regularize_data)

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
        Using the library 'scikit-image'
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

            print(20*'-', '\n', _s_)

            self.s_n[_s_-1] = len(self.s_gid[_s_])
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

    def find_neigh_gid(self, verbose=False, interval=25):
        """
        Set neighbouring gids of all grains.

        Parameters
        ----------
        verbose : bool
            DEscription.
        interval : int
            Descrition.

        saa
        ---
        neigh_gid : dict
            DEscription
        """
        print('Calculating 1st order neighbours.')
        self.neigh_gid = {grain: set() for grain in np.unique(self.lgi)}
        dxdydz = [(-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1,  0, -1),
                  (-1,  0, 0), (-1,  0, 1), (-1,  1, -1), (-1,  1, 0),
                  (-1,  1, 1), (0, -1, -1), (0, -1, 0), (0, -1, 1),
                  (0,  0, -1), (0,  0, 1), (0,  1, -1), (0,  1, 0), (0,  1, 1),
                  (1, -1, -1), (1, -1, 0), (1, -1, 1), (1,  0, -1), (1,  0, 0),
                  (1,  0, 1), (1,  1, -1), (1,  1, 0), (1,  1, 1)]
        for x in range(self.lgi.shape[0]):
            if verbose and x % interval == 0:
                print(f'Finding O(1) neigh at voxel in zslice: {interval}')
            for y in range(self.lgi.shape[1]):
                for z in range(self.lgi.shape[2]):
                    grain_id, neighbors = self.lgi[x, y, z], set()
                    for dx, dy, dz in dxdydz:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        cond1 = 0 <= nx < self.lgi.shape[0]
                        cond2 = 0 <= ny < self.lgi.shape[1]
                        cond3 = 0 <= nz < self.lgi.shape[2]
                        if cond1 and cond2 and cond3:
                            neighbor_id = self.lgi[nx, ny, nz]
                            if neighbor_id != grain_id:
                                neighbors.add(neighbor_id)
                    self.neigh_gid[grain_id].update(neighbors)
        for grain in self.neigh_gid:
            self.neigh_gid[grain] = list(self.neigh_gid[grain])

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
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        gid = 10
        np.unique(pxtal.gs[16].find_extended_bounding_box(gid))
        pxtal.gs[16].find_neigh_gid_fast_all_grains(include_parent=False)
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
            self.find_neigh_gid_fast_all_grains(include_parent=False)
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
        pxtal.gs[16].find_neigh_gid_fast_all_grains(include_parent=False)
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
        self.pvgrid = pv.UniformGrid()
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
        else:
            self.pvgrid.cell_data[sf_name] = self.sf_value.flatten(order="F")

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

    def find_grain_voxel_locs(self, verbosity=10):
        """
        Find voxel locations of grains in lgi.

        saa
        ---
        grain_locs
        """
        print('\nFinding voxel locations of grains in lgi.')
        ngrains = len(self.gid)
        verbosity = ngrains//verbosity
        for gid in self.gid:
            self.grain_locs[gid] = np.argwhere(self.lgi == gid)
            if gid % verbosity == 0:
                print(f'gid: {gid} of {ngrains} grains')

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

    def get_scalar_field(self, sf_name="lgi"):
        """
        Return the requested scalar field.

        Parameters
        ----------
        sf_name : str, optional

        Returns
        -------
        sf_value : np.ndarray / None
        """
        self.validate_scalar_field_name(sf_name)
        if sf_name == "lgi":
            sf_value = self.lgi
        else:
            sf_value = None
        return sf_value

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
        for gid in gids:
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
    @njit
    def _compute_volumes_with_bincount(lgi, n):
        """
        Calculate the volume by number of voxels using Numba and bincount.
        """
        # Flatten lgi and calculate counts using np.bincount
        return np.bincount(lgi.ravel(), minlength=n + 1)

    def set_mprop_volnv(self):
        """
        Calculate the volume by number of voxels.
        """
        print(40*"-", "\nSetting grain volumes (metric: 'volnv').")
        unique_counts = self._compute_volumes_with_bincount(self.lgi, self.n)
        self.mprop['volnv'] = {gid + 1: unique_counts[gid + 1] for gid in range(self.n)}

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
            concomp = cv2.connectedComponents
            # Acceptable values for opencv: 4, 8
            if kernel_order in (4, 8):
                KO = kernel_order
            elif kernel_order in (1, 2):
                KO = 4*kernel_order
            else:
                raise ValueError("Input must be in (1, 2, 4, 8)."
                                 f" Recieved {kernel_order}")
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
                       ave_plot2d_grains=True):
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
            grid = pv.UniformGrid()
            grid.dimensions = np.array(self.lgi_slice['lgi_masked'].shape+(0,)) + 1
            grid.origin, grid.spacing = (0, 0, 0), (1, 1, 0)
            grid.cell_data['lgi'] = self.lgi_slice['lgi_masked'].flatten(order='f')
            self.lgi_slice['pvgrid_masked'] = grid
        # ------------------------
        grid = pv.UniformGrid()
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
                sssrmul[sp][loc] = deepcopy(self.sssr)
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
        return [gid+1 for gid in gid_max_presence[0]]

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
                self.set_mprop_volnv()
                if reset_pvgrid_every_iter:
                    self.make_pvgrid()
                    self.add_scalar_field_to_pvgrid(sf_name="lgi",
                                                    sf_value=self.lgi)
                if find_neigh_every_iter:
                    self.find_neigh_gid(verbose=False)
                if find_grvox_every_iter:
                    # gstslice.grain_locs
                    self.find_grain_voxel_locs()
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
                pair = tuple(sorted((gid, neighbor)))

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
            self.gbp_id_maps[tuple(point)] = pointid
        # -----------------------------------------
        # From self.gbp_ids
        self.gbp_ids = {gid: None for gid in self.gid}
        for gid in self.gid:
            if gid % 500 == 0:
                print(f'Forming local-global ID maps for gid no. {gid}')
            self.gbp_ids[gid] = set([self.gbp_id_maps[tuple(point)]
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
        '''Make the seed twin plane.'''
        tp = Plane.from_three_points(selcoord, point1, point2)
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
        self.fdb[base_gs_name]['data']['twin_i'] = np.array(twin_i)
        self.fdb[base_gs_name]['data']['twin_id'] = np.array(twin_id)
        self.fdb[base_gs_name]['data']['twin_vol'] = np.array(twin_vol)
        self.fdb[base_gs_name]['data']['twin_vf'] = np.array(twin_vf)
        notwin_gids = np.arange(1, self.lgi.max()+1, 1)
        self.fdb[base_gs_name]['data']['notwin_gids'] = notwin_gids
        self.fdb[base_gs_name]['data']['twin_coords'] = self.gid_twin
        self.fdb[base_gs_name]['data']['twin_map_g_t'] = twin_map_g_t
        self.fdb[base_gs_name]['data']['twin_map_g_nt'] = twin_map_g_nt
        # Reset gid_twin database as its a;lready saved as twin_coords
        self.gid_twin = None
        twin_vf_total = tvol / self.domain_volume
        self.fdb[base_gs_name]['data']['twin_vol_total'] = tvol
        self.fdb[base_gs_name]['data']['twin_vf_total'] = twin_vf_total
        # -------------------------------------------------
        print(40*'#', f'\nTwin volume fraction: {twin_vf_total}')

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
                          reset_fdb=True,
                          reset_keystring='twin.'
                          ):
        """
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
        pvgrid = pv.UniformGrid()
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
        nr, nc = arrange_subplots(ninstances, rmax_sp, cmax_sp)
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
        if feature_name in ('base', 'base_gs'):
            _base_data_ = self.lgi
            scalar_name = 'lgi.subdomain'
        if 'twin.' in feature_name:
            _base_data_ = self.fdb[feature_name]['data']['fid']
            scalar_name = f'lgi.{feature_name}'
        # ---------------------------------------------
        print(40*'-', f'\nExtracting {n} subdomains at random.')
        P, Q, R = (PQR-pqr+1 for pqr, PQR in zip((p, q, r), _base_data_.shape))
        # ---------------------------------------------
        subdomains, pvgrids = [], []

        for _ in range(n):
            x = np.random.randint(0, P)
            y = np.random.randint(0, Q)
            z = np.random.randint(0, R)

            subdomain = _base_data_[x:x+p, y:y+q, z:z+r]
            subdomains.append(subdomain)

        SD = {'sd': subdomains}

        if make_pvgrids:
            pvgrids = []
            for sd_count in range(n):
                pvgrid = self.make_pvgrid_v1(feature_name='user',
                                             instance_name='none',
                                             user_fid=subdomains[sd_count],
                                             scalar_name=scalar_name,
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
        pvgrid.dimensions = np.array(fid) + 1
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



'''# ====================================================
import pyvista as pv
fid = gstslice.lgi

# As is
pvgrid = pv.UniformGrid()
pvgrid.dimensions = np.array(fid.shape) + 1
pvgrid.origin = (0, 0, 0)
pvgrid.spacing = (1, 1, 1)
pvgrid.cell_data['fid'] = fid.flatten(order="F")
pvgrid.plot()

# Modofied grid
pvgrid = pv.UniformGrid()
pvgrid.dimensions = np.array(fid.shape) + 1
pvgrid.origin = (0, 0, 0)
pvgrid.spacing = (1, 1, 0.2)
pvgrid.cell_data['fid'] = fid.flatten(order="F")
pvgrid.plot()

# Use pvgrid to make another pvgrid which has the same grid domain but
# voxel size of (1, 1, 1)
grid_extent = (np.array(fid.shape)) * pvgrid.spacing
grid_extent = np.array((200, 100, 100))
new_spacing = (1, 1, 1)
new_dimensions = np.round(grid_extent / new_spacing).astype(int) + 1
new_cell_shape = (new_dimensions[0] - 1, new_dimensions[1] - 1, new_dimensions[2] - 1)

scaling_factors = (fid.shape[0] / new_cell_shape[0],
                   fid.shape[1] / new_cell_shape[1],
                   fid.shape[2] / new_cell_shape[2])

from scipy.ndimage import zoom
resampled_fid = zoom(fid, zoom=scaling_factors, order=0)

resampled_fid.shape

pvgrid_resampled = pv.UniformGrid()
pvgrid_resampled.dimensions = resampled_fid.shape
pvgrid_resampled.origin = (0, 0, 0)
pvgrid_resampled.spacing = new_spacing  # New voxel size of (1,1,1)


pvgrid_resampled.cell_data['fid'] = resampled_fid[:-1, :-1, :-1].flatten(order="F")

# Plot the resampled grid
pvgrid_resampled.plot(show_edges=True)
# ====================================================
pvgrid = pv.UniformGrid()
pvgrid.dimensions = np.array(fid.shape) + 1
pvgrid.origin = (0, 0, 0)
pvgrid.spacing = (1, 1, 1)
pvgrid.cell_data['fid'] = fid.flatten(order="F")
pvgrid.plot(show_edges=True)

fid.shape

resampled_fid = zoom(fid, zoom=(0.5, 0.5, 0.5), order=0)
resampled_fid.shape

pvgrid = pv.UniformGrid()
pvgrid.dimensions = np.array(resampled_fid.shape) + 1
pvgrid.origin = (0, 0, 0)
pvgrid.spacing = (1, 1, 1)
pvgrid.cell_data['fid'] = resampled_fid.flatten(order="F")
pvgrid.plot(show_edges=True)

resampled_fid = zoom(resampled_fid, zoom=(2, 2, 2), order=0)
pvgrid.dimensions = np.array(resampled_fid.shape) + 1
pvgrid.origin = (0, 0, 0)
pvgrid.spacing = (1, 1, 1)
pvgrid.cell_data['fid'] = resampled_fid.flatten(order="F")
pvgrid.plot(show_edges=True)

resampled_fid = zoom(resampled_fid, zoom=(0.5, 0.5, 0.5), order=0)
pvgrid.dimensions = np.array(resampled_fid.shape) + 1
pvgrid.origin = (0, 0, 0)
pvgrid.spacing = (1, 1, 1)
pvgrid.cell_data['fid'] = resampled_fid.flatten(order="F")
pvgrid.plot(show_edges=True)

resampled_fid.shape

# Removing small features
f = 1/2

resampled_fid = zoom(fid, zoom=(f, f, f), order=0, mode='nearest')
resampled_fid = zoom(resampled_fid, zoom=(1/f, 1/f, 1/f), order=0, mode='nearest')

pvgrid.dimensions = np.array(resampled_fid.shape) + 1
pvgrid.origin = (0, 0, 0)
pvgrid.spacing = (1, 1, 1)
pvgrid.cell_data['fid'] = resampled_fid.flatten(order="F")
pvgrid.plot(cmap='nipy_spectral', show_edges=True)

pvgrid.dimensions = np.array(fid.shape) + 1
pvgrid.origin = (0, 0, 0)
pvgrid.spacing = (1, 1, 1)
pvgrid.cell_data['fid'] = fid.flatten(order="F")
pvgrid.plot(cmap='nipy_spectral', show_edges=True)
# np.unique(fid).size
# np.unique(resampled_fid).size
'''
