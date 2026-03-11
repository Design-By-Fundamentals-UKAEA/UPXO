# -*- coding: utf-8 -*-
"""
Class-1: gtess2d
----------------
Instantiation routes:
    1.
    2.
-------------------------------------------------------------------------------
Created on Thu Mar  6 18:14:08 2025
@author: Dr. Sunil Anandatheertha
"""
import matplotlib.pyplot as plt
from upxo.geoEntities.mulpoint2d_old import mulpoint2d
from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
from upxo.geoEntities.sline2d import Sline2d as sl2d
from shapely.ops import voronoi_diagram
from shapely.affinity import scale as sh_scale
from upxo._sup import dataTypeHandlers as dth
from shapely.geometry import Point
from shapely.geometry import MultiPoint as ShMultiPoint
from shapely.geometry import Polygon
from shapely import geometry as shGeometry
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely.ops import unary_union
from scipy.spatial import cKDTree as ckdt
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import matplotlib.pyplot as plt
from upxo.netops.kmake import make_gid_net_from_neighlist
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import LinearRing
import networkx as nx
import random
import copy
import math
import shapely
from scipy import stats

class gtess2d():
    """
    ===========================================================================
    @ dev notes by (Dr. Sunil Anandatheertha, )
    ===========================================================================
    ---------------> Instantiation <--------------------
    from upxo.pxtal.vortess2d import gtess2d
    repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5, },
                        'consider_boundary_grains': True } }
    gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 50],
             ybound = [0, 50], sp_distr='random', gr_tech='pds',
             smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
             niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
             k_char_length_inc=0.05, k_char_length_dec=0.05,)
    .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
    --------------------> Important variables <--------------------
    * ninst: Number of instances
    * instn: range of the ninst

    * floc: Feature location. Needs update with every change in instance.
    Does not store instance wise. As calculation is quick, it must be
    re-calculated afresh for geomewtry.

    * fdb_base: feature data base link.
    * geolink: geometry link.

    * mprop: Morphological property
    * tprop: Topological property

    --------------------> Data access <--------------------

    [gset._idmap_fc_[gslevel][instance_ID][fname][c] for c in cell_IDs]

    -------------------> IMP DEFs: Feature data operations <-------------------

    gset.add_fdb(gslevels=['link'],
                 fdbs=[gset.link_geom(instance=1, saa=True, throw=True)])

    --------------------> IMP DEFs: Geometry linking <--------------------

    gset.link_geom(gslevel='base', instance=1, make_upxo_mp=True,
                   make_upxo_mp_subfeatures=False, saa=True, throw=False)

    --------------------> IMP DEFs: Data access <--------------------

    gslevel, instance_ID, fname, fid, cids = 'base', 1, 'ph', 1, [1, 5]
    gset.get_idmap_cf(gslevel, instance_ID, fname, fid, cids=cids)
    gset.get_idmap_fc(gslevel, instance_ID, fname, cids=[])
    gset.get_idmap_fc(gslevel, instance_ID, fname, cids=cids)
    gset.extract_shapely_coords_all_grains(instance=1, gslevel='base',
                                           mfname='c_rp', make_upxo_mp=True,
                                           make_upxo_mp_subfeatures=False)

    --------------------> IMP DEFs: Assign tags and IDs <--------------------

    gset.assign_loc_tags(instance=1,gslevel='base',tol=1E-6,saa=True,throw=True)


    gset.plot(instance=1)
    gset.plot_cells(instance=1, cids=[1])

    gset.sf_gbz_cell(gset.pxtals[1].geoms[0], 0.8)
    gset.sf_gbz(instance=1,  gslevel='base', cids=[], k_shr=0.1,
                assemble_cells=True, assemble_sf=True, saa=True, throw=False)
    # gset.assemble_cells(features)


    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\




    Incomplete definitions
    ----------------------

    add_supcells(self, On=.5, merge=False)
    merge_supcells(self, scids=None, conn=True)
    add_subcells(self, pfname='base', parents='by_ids', cids=None,
                         sc_type='subgrains',
                         recursive=False, recursion=1,
                         genpar={})
    partition_cells(self, instance=1, gslevel='base', cids=[],
                            partition_type='Sub-cell-0',
                            partition_kwargs={}
                            )
    voronoi_subdivision(self, xtal_object,
                                n_seed_points, seed_lattice_type,
                                combine_small_subs)
    setup_fid(self, pxnames=['twid', 'prid', 'tcid'])
    setup_fdb

    --------------------------------------------------
    find_On_neigh(self, instance=1, gslevel='base', On=1,
                          cid_exc_rules_calc=[],
                          cid_exc_rules_results=[],
                          include_central_grain=True,
                          saa=True, throw=True,
                          n_instances=1, __recall__=False
                          )
    non = gset.find_On_neigh(instance=2, gslevel='base', On=2.9,
                                     n_instances=3)
    get_stat(self, data, nan_policy='omit')
    perturb_feat(self, instance=1, gslevel='base', cid=[], )
    fe_mesh(self, instance=1, gslevel='base', tool='gmsh')
    ---------------------------------------------------
    Definityions to update
    ----------------------
    setup_mprop
    setup_tprop
    find_areas
    find_mprop_all
    find_ar
    make_upxo_gs
    reconstruct_pxtal_from_geom_link
    gset.plot(instance=1)

    plot_cells(self, instance=1, cids=[1])
    """
    _valid_gslevels_ = ('base', 'supset', 'tw', 'pap',
                        'pap.bl', 'pap.bl.sbl', 'pap.bl.lath',
                        'pap.bl.sbl.lath', 'base.bz', 'base.ppt')
    """
    Explanation of _valid_gslevels_:
    --------------------------------
        * 'base'. The basic grain structure for all other hierarchically higher
        grain structures.
        * 'tw': Twinned grain structures.
        * 'pap': Prior-austenetic packates.
        * 'pap.bl': Prior-austenitic packet blocks.
        * 'pap.bl.sbl': Prior-austenitic packets, block and sub-blocks.
        * 'pap.bl.lath': Prior-austenitic packets, blocks and laths.
        * 'pap.bl.sbl.lath': Prior-austenitic packets, block, sub-blocks and
        laths.
        * 'base.bz': boundary zones grain structures built upon the base grain
        structures. Any number of boundary zones can be contained in any grain.
        * 'base.ppt': base grain structure with precipitates.
    """
    # -------------------------------------------------------------------------
    _valid_gsnamemaps_ = {'base': 'fdb_base',
                          'supset': 'fdb_supset',
                          'pix': 'fdb_pix',
                          'pert': 'fdb_pert',
                          'subgrains': 'fdb_subgrains',
                          'paps': 'fdb_paps'
                          }
    """
    Exaplanation of _valid_gsnamemaps_. These are just simlified user input
    strings which will correspond to different feature related variables in
    this class. These are used in definitions self.add_fdb, etc, under the
    name 'gslevel'.
        * User enters: 'base'. In this casem UPXO considers self.fdb_base
        * User enters: 'supset'.  In this casem UPXO considers self.fdb_supset
        * User enters: 'pix'. Similar explanations.
        * User enters: 'pert'. Similar explanations.
        * User enters: 'subgrains'. Similar explanations.
        * User enters: 'paps'. Similar explanations.
    """
    # -------------------------------------------------------------------------
    _valid_fnames_ = ('ph', 'tc', 'bz', 'bfzones',
                      'gc', 'sg', 'tw', 'paps', 'block', 'sub-blocks', 'laths',
                      'ppt', 'void', 'fbr', 'fbr-int')
    """
    Explanation of _valid_fnames_. These are permitted names of featues.
    Values are strings and are below. Note that these are not necessarily
    morphological features. For example, 'ph' i.e. 'phase' has also been
    included for the purpose of simplicity of data structure.
        * 'ph': Phase.
        * 'tc': texture components.
        * 'bz': bounded zones. Includes both boundary zones and core zones of
        cells such as grains, twins, precipitates, etc.
        * 'bfzones': buffer zones needed to reduce meshing complexity.

        * 'gc': grain cluster.
        * 'sg': sub-grains.
        * 'tw': twins.
        * 'paps': prior austenitic packets.
        * block: blocks of hierarchical steel grain structure (GS).
        * sub-blocks: sub-blocks in blocks of hierarchical steel GS.
        * laths: laths in blocks / sub-blocks of hierarchical steel GS.

        * 'ppt': precipitates.
        * 'void': void locations.
        * 'fbr': fibre.
        * 'fbr-int': Fibre-matrix interface.
    """
    # -------------------------------------------------------------------------
    __slots__ = ('ninst', 'instn', 'sps', 'pxtals', 'fdb_base',
                 'gen_method',
                 'fid', 'floc', 'geolink', 'cids_base',
                 'idmap_c_supc', 'idmap_c_subc',
                 'fdb_pix', '_idmap_fc_', '_idmap_cf_',
                 'fdb_pert', 'fdb_subgrains', 'fdb_paps', 'fdb_supset',
                 'uinputs',
                 'neighs_cid', 'xomap', 'n',
                 'gbjp', 'xbound', 'ybound', 'mprop', 'tprop',
                 'grain_locs',
                 'gpos')

    """
    Slot variables
    --------------
    ninst: int
        Number of instances
    sps: dict
        Seed points of each pxtal instance. keys are instance numbers and
        values are UPXo multi-point objects.
    pxtals: dict
        Shapely multi-polygon objects. Keys are instance numbers and values are
        multi-polygons.
    fdb_base: dict
        Feature database link.
    nbc: int
        Number of base cells i.e. grains.
    fid: dict
        Feature ID. Keys are instance numbers (q) and values are dictionaries.
    fid[q]['bcid']: dict
        Base cid. list of cell IDs in the base grain structure. Each value
        corresponds to the grain gset.pxtals[q][]
    fid[q]['phid']: dict
        Phase ID. Keys are phase IDs.
    """
    # -------------------------------------------------------------------------
    """
                #### NOTES ON GRAIN STRUCTURE HIERARCHY ####

    The base shapely grain strucures will be stored in gset.pxtals

    Please refer to the below explanations.
        1. 'phid': Phase ID. Every cell will have a phase ID associated to
        it. It can be accessed as gset.fid[instance]['phid'].
        2. 'bcid': Every grain, at first creation will be allocated a base
        grain ID. This will be the 1st order hierarchical feature. All
        other morphologcal subfeatures will be expressed in relation to
        this. It can be accessed as gset.fid[instance]['bcid'].
        3. 'gfid': Gloabl feature ID.
            It can be accessed as gset.fid[instance]['gfid'].
        4. 'hlid': Hierarchy level ID
            It can be accessed as gset.fid[instance]['hlid'].

        * 'scid': Sub-cell win ID.
        * 'twid': Twin ID
        * 'prid': Precipitate ID
        * 'clid': Grain cluster ID. Use this to create cell clusters
            needed to represent Prior Austenitic Packets, grain cluster
            in materials such as CuCrZr, etc.
        * 'tcid': Texture component ID.
    """

    def __init__(self, sps=None, xbound=None, ybound=None, pxtals=None,
                 uinputs=None, phid=1, gen_method='from_seed_points'):
        # ------------------------------------------------------------------
        print(50*'#')
        print(50*'#')
        print(pxtals)
        print(50*'#')
        print(50*'#')
        self.gen_method = gen_method
        self.sps = sps
        self.pxtals = pxtals
        self.ninst = len(pxtals.keys())
        self.instn = range(1, self.ninst+1)
        self.uinputs = uinputs
        self.fdb_base, self.geolink = {}, {}
        self.fdb_subgrains = {}
        self.fdb_paps = {}
        # ------------------------------------------------------------------
        '''Morphological properties.'''
        self.mprop = {inst: {'area': [], 'ar': [], 'gbl': [],
                             }
                      for inst in self.instn}
        '''Topological properties.'''
        self.tprop = {inst: {'ncells': len(self.pxtals[inst].geoms),
                             'ncrange': range(len(self.pxtals[inst].geoms)),
                             'nneigh': [],
                             'ngb': [], 'jpo': [],
                             }
                      for inst in self.instn}
        # ------------------------------------------------------------------
        self._idmap_fc_ = {'base': {}}
        '''Map features to cell-IDs. Examples of features included are in the
        tuple, self._valid_fnames_.
        general data structure:

        Example acceess - 1 (Direct access, Not preferred)
        --------------------------------------------------
        gslevel = 'base'
        instance_ID = 1
        cell_ID = 15
        gset._idmap_fc_[gslevel].keys()
        gset._idmap_fc_[gslevel][1].keys()
        gset._idmap_fc_[gslevel][1]['ph'][cell_ID]

        Example access - 2 (Preferred)
        ------------------------------
        Please refer to documentation of defintion get_idmap_cf(). Code is,
        gset.get_idmap_fc('base', 1, 'ph', 1, cids=[])

        options 3 and 2
        '''
        for i in self.instn:
            _tmp_ = [phid for cid in self.tprop[i]['ncrange']]
            self._idmap_fc_['base'][i] = {'ph': _tmp_}
        # ------------------------------------------------------------------
        self._idmap_cf_ = {'base': {}}
        '''Map cell IDs to features. Examples of features included are in the
        tuple, self._valid_fnames_. By default, phase key (i.e. 'ph') will be
        created. The user may create appropriate keys from the allowed list.

        Example acceess - 1 (Direct access, Not preferred)
        --------------------------------------------------
        gslevel = 'base'
        instance_ID = 1
        fname = 'ph'
        fid = 1
        cell_ID = 15

        gset._idmap_cf_[gslevel].keys()
        gset._idmap_cf_[gslevel][instance_ID].keys()
        gset._idmap_cf_[gslevel][instance_ID][fname].keys()
        gset._idmap_cf_[gslevel][instance_ID][fname][fid]

        The above code accessess all the cells belonging to phase of phase_ID,
        in the pxtal instance specified by instance_ID.

        Example access - 2 (Preferred)
        ------------------------------
        Please refer to documentation of defintion get_idmap_cf(). Code is,

        gset.get_idmap_cf('base', 1, 'ph', 1, cids=[])
        '''
        for i in self.instn:
            _tmp_ = {'ph': {phid: [cid for cid in self.tprop[i]['ncrange']]}}
            self._idmap_cf_['base'][i] = _tmp_
        # ------------------------------------------------------------------
        # def get_idmap_c_supc()
        self.idmap_c_supc = {i: None for i in self.instn}
        # def get_idmap_c_subc
        self.idmap_c_subc = {i: None for i in self.instn}
        # ------------------------------------------------------------------
        '''Now, we use add_fdb to actually store essential feature information.
        This includes the vertex coordinstes, instance number, CID mapping list
        of list of vertex coordinates, CID mapping list of list of vertex
        coordinate IDs, list of 2D point objects, CID mapping list of
        UPXo point 2D Object list IDs and UPXO multi-point object (Optional).
        '''
        for inst in self.instn:
            self.add_fdb(gslevels=['base'],
                         fdbs=[self.link_geom(instance=inst,
                                              make_upxo_mp=True,
                                              make_upxo_mp_subfeatures=False,
                                              saa=True, throw=True)])
    def __repr__(self):
        return f"gtess2d instances. n: {self.ninst}. "

    @classmethod
    def from_shapely_mulpolygon(cls, mpol):
        pass

    @classmethod
    def from_geometrified_mcgs2d(cls, polgs):
        """
        polgs: dict: Polygonised grain structures

        Example
        -------
        from upxo.pxtal.vortess2d import gtess2d
        gset = gtess2d.from_geometrified_mcgs2d(PXTAL)
        """
        sps = None
        pxtals = polgs
        uinputs = None
        xbound = None
        ybound = None
        return cls(sps=sps, pxtals=pxtals, xbound=xbound, ybound=ybound,
                   uinputs=uinputs, gen_method='from_geometrified_mcgs2d')

    @classmethod
    def from_seed_points(cls, sp_input='gen', seed_coords=None,
                         xbound=[0, 100], ybound=[0, 100], nsp=600,
                         n_instances=1, nsp_dev_ninstances=10,
                         sp_distr='random', gr_tech='pds', smp_tech='bridson1',
                         randuni_calc='by_points', lean='veryhigh',
                         char_length=[3, 2], niter=500, ntp=25, ntrials=10,
                         k_char_length_inc=0.1, k_char_length_dec=0.1,
                         rep_gen=True, nseeds=25,
                         repr_prop={'area': {'mean': {'val': 50, 'dev': 10, },
                                             'consider_boundary_grains': True}
                                    },
                         make_point_objects=True, make_ckdtree=True,
                         char_length_mean=0.24598,
                         char_length_min=0.1111, char_length_max=0.9999,
                         nt=10, space='linear',):
        """
        Parameters
        ----------
        sp_input: str, optional
            Seed point input method. If 'load', then seed_coords must be
            specified. If 'gen', parameters concerning generating the
            seed points must be specififed. Default value is 'gen'.

        seed_coords: np.ndarray, optional
            2D coordinate numpy array. Default value is None.

        xbound: list, optional
            Spatial bound of expected pxtal along x-axis, [xmin, xmax].
            Default value is [0, 100].

        ybound: list, optional
            Spatial bound of expected pxtal along y-axis, [ymin, ymax].
            Default value is [0, 100].

        nsp: int, optional
            Number of seed points. Default value is 600.

        n_instances: int, optional
            Number of poly-xtal instances to be generated.
            Default value is 1.

        nsp_dev_ninstances: int, optional
            Allowable deviation in number of seed points across instances. The
            first instance will be used as a reference. The value to be input
            is a percentage value. If value entered is 10, this would mean,
            the second and other instances will be created ensuring that the
            parameters needed to create them are to so as to keep the number
            of seed points between -5% and +5% of that of the 1st instance.
            This is to ensure similar morphological parameter distributions
            across all instabses. Default value is 10.

        sp_distr: int, optional
            Spatial distribution of the seed points desired. Options insluce
            'random'. Default value is 'random'.

        gr_tech: str, optional
            Gridding technique. Options include 'random', 'pds'.
            Default value is 'pds'.

        smp_tech: str, optional
            Sampling technique. Options include 'uniform', 'dart', 'bridson1'.
            Default value is 'bridson1'.

        randuni_calc: str, optional
            Random uniform calculations.
            Default value is 'by_points'.

        lean: str, optional
            UPXO point lean option used for creatinhg multipoint.
            Default value is 'veryhigh'.

        char_length: list, optional
            Characteristic lengths needed for seed point creation.
            In case of 'dart' and 'bridson1' sampling tecjhnique,
            char_length[0] determines the average spatial distance between the
            points. The higher the value, the greater the distane, which means
            the lesser the number of points and greater the mean area of
            poly-xtals. Default value is [3, 2].

        niter: int, optional
            NUmber of iterations needed for seed point creation.
            Default value is 500.

        ntrials: int, optional
            Number of trials used in the creation of the 1st pxtal instance.
            The irterations will be done to ensure pxtal parameter agrees to
            as prescibed by repr_prop. Default value is -1.

        k_char_length_inc: float, optional
            Factor to increase the characteristic length. Value must be greater
            than 0. Prescribed domain [0.02, 0.25]. A very small value would
            increase the number of iterations needed to avchieve the required
            morphologycal parameter requirement. Too big a value may lead to
            oscillating iterations. Default value is 0.1.

        k_char_length_dec: float, optional
            Factor to decrease the characteristic length. Value must be greater
            than 0. Prescribed domain [0.02, 0.25]. A very small value would
            increase the number of iterations needed to avchieve the required
            morphologycal parameter requirement. Too big a value may lead to
            oscillating iterations. Default value is 0.1.

        repr_prop: dict, optional
            Representativeness requirement of properties.
            Default value is {'area': {'mean': 6,
                                       'dev': 10,
                                       'consider_boundary_grains': True
                                       }
                              }.

        make_point_objects: bool, optional
            Default value is True.
        make_ckdtree: bool, optional
            Default value is True.
        char_length_mean: float, optional
            Default value is 0.24598.
        char_length_min: float, optional
            Default value is 0.1111.
        char_length_max: float, optional
            Default value is 0.9999.
        nt: int, optional
            Default value is 10.
        space: str, optional
            Default value is 'linear'.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True } }

        # here we get Poisson disc sampling ------------>
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 100],
                 ybound = [0, 100], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 ntp=10, ntrials=100, n_instances=25, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        # Here we get dart sampling ------------>
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 100],
                 ybound = [0, 100], sp_distr='random', gr_tech='random',
                 smp_tech='dart', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        gset.plot()
        """
        if ntrials == -1:
            ntrials = 100

        pxtals = {}
        sps = {}

        pxtal_count = 1

        if sp_input == 'load':
            seed_coords = seed_coords

        if sp_input == 'gen':
            if sp_distr == 'random':
                # ----------------------------
                if gr_tech == 'random':
                    if smp_tech == 'uniform':
                        _char_length = char_length
                        # Remaining codes here to generate the pxtal
                # ----------------------------
                if gr_tech in ('random', 'pds'):
                    if smp_tech in ('dart', 'bridson1'):
                        _char_length = [char_length[0]]
                        _def = cls._make_pxtal_single_instance
                        for trial in range(ntrials):
                            print(f"Generating pxtal. Iteration {trial+1}")
                            pxt = _def(spinput=sp_input,
                                       xbound=xbound, ybound=ybound,
                                       nsp=nsp, sp_distr=sp_distr,
                                       gridding_technique=gr_tech,
                                       sampling_technique=smp_tech,
                                       randuni_calc=randuni_calc, niter=niter,
                                       ntp=ntp,
                                       lean=lean, char_length=_char_length,
                                       make_point_objects=make_point_objects,
                                       make_ckdtree=make_ckdtree, space=space,
                                       char_length_mean=char_length_mean,
                                       char_length_min=char_length_min,
                                       char_length_max=char_length_max, nt=nt)
                            print(f"No. of seed points: {pxt['sp'].npoints}")
                            if rep_gen:
                                if 'area' in repr_prop.keys():
                                    areas = np.array([g.area
                                                      for g in pxt['pxtal'].geoms])
                                    if 'mean' in repr_prop['area'].keys():
                                        amean = areas.mean()
                                        _a = repr_prop['area']['mean']['val']
                                        _adev = repr_prop['area']['mean']['dev']
                                        amin = _a*(1-_adev/100)
                                        amax = _a*(1+_adev/100)

                                        if amin <= amean <= amax:
                                            _char_length = char_length
                                            pxtals[pxtal_count] = pxt['pxtal']
                                            sps[pxtal_count] = pxt['sp']
                                            break

                                        if amean < amin:
                                            k = 1 + trial*k_char_length_inc
                                            _char_length = [char_length[0] * k]
                                            _char_length_ = _char_length

                                        if amean > amax:
                                            k = 1 - trial*k_char_length_dec
                                            _char_length = [char_length[0] * k]
                                            _char_length_ = _char_length
            if rep_gen:
                if trial == ntrials-1:
                    print(50*'#', '\n')
                    print('WARNING')
                    print('        Maximum number of iterations reached')
                    print('        Grain Structure DID NOT converge !!!\n')
                    print(50*'#')
                print(40*'-')
                print(f"Grain structure search converged in {trial} iterations")
                print('Sample set parent found.')
                print('----')
                print(f"Target mean grain area: {repr_prop['area']['mean']['val']}")
                print(f"Sample mean grain area: {np.round(areas.mean(), 6)}")
                print('----')
                print(f"Input Char. length: {char_length}")
                print(f"Final Char. length: {np.round(_char_length_, 4)}")
                print('----')
                print(f"No. of seed points: {pxt['sp'].npoints}")
                print(f"No. of grains: {len(pxt['pxtal'].geoms)}")
                print(40*'-')
            uinputs = {'sp_input': sp_input,
                       'xbound': xbound, 'ybound': ybound, 'nsp': nsp,
                       'sp_distr': sp_distr, 'gridding_technique': gr_tech,
                       'sampling_technique': smp_tech,
                       'randuni_calc': randuni_calc, 'lean': lean,
                       'char_length': _char_length, 'niter': niter,
                       'make_point_objects': make_point_objects,
                       'make_ckdtree': make_ckdtree,
                       'char_length_mean': char_length_mean,
                       'char_length_min': char_length_min,
                       'char_length_max': char_length_max,
                       'nt': nt, 'space': space}

            if n_instances > 1:
                for inst in range(n_instances-1):
                    print(f"Generating instance number: {inst+2} using sample set parent GS.")
                    if sp_input == 'gen':
                        if sp_distr == 'random':
                            # ----------------------------
                            if gr_tech == 'random':
                                if smp_tech == 'uniform':
                                    _char_length = char_length
                                    # Remaining codes here to generate the pxtal
                            # ----------------------------
                            if gr_tech in ('random', 'pds'):
                                if smp_tech in ('dart', 'bridson1'):
                                    _char_length = [char_length[0]]
                                    _def = cls._make_pxtal_single_instance
                                    pxtal_count += 1
                                    pxt = _def(spinput=sp_input,
                                               xbound=xbound,
                                               ybound=ybound,
                                               nsp=nsp, sp_distr=sp_distr,
                                               gridding_technique=gr_tech,
                                               sampling_technique=smp_tech,
                                               randuni_calc=randuni_calc,
                                               niter=niter,
                                               lean=lean,
                                               char_length=_char_length_,
                                               make_point_objects=make_point_objects,
                                               make_ckdtree=make_ckdtree,
                                               space=space,
                                               char_length_mean=char_length_mean,
                                               char_length_min=char_length_min,
                                               char_length_max=char_length_max,
                                               nt=nt)
                                    # print(pxt)
                                    pxtals[pxtal_count] = pxt['pxtal']
                                    sps[pxtal_count] = pxt['sp']

        return cls(sps=sps, pxtals=pxtals, xbound=xbound, ybound=ybound,
                   uinputs=uinputs, gen_method='from_seed_points')

    @classmethod
    def from_geometry_link(self, geo_link):
        pass

    @staticmethod
    def _make_pxtal_single_instance(spinput='gen',
                                    xbound=[0, 100], ybound=[0, 100], nsp=600,
                                    sp_distr='random',
                                    gridding_technique='pds',
                                    sampling_technique='bridson1',
                                    randuni_calc='by_points',
                                    lean='veryhigh', char_length=[3, 2],
                                    niter=500, ntp=25, make_point_objects=True,
                                    make_ckdtree=True,
                                    char_length_mean=0.24598,
                                    char_length_min=0.1111,
                                    char_length_max=0.9999,
                                    nt=10, space='linear',):
        if sp_distr == 'random':
            if gridding_technique == 'random':
                if sampling_technique == 'uniform':
                    sp = mulpoint2d(method=sp_distr,
                                    gridding_technique=gridding_technique,
                                    sampling_technique=sampling_technique,
                                    nrndpnts=nsp,
                                    randuni_calc=randuni_calc,
                                    char_length_mean=char_length_mean,
                                    char_length_min=char_length_min,
                                    char_length_max=char_length_max,
                                    n_trials=nt, n_iterations=niter,
                                    space=space, xbound=xbound,
                                    ybound=ybound, lean=lean,
                                    make_point_objects=make_point_objects,
                                    make_ckdtree=make_ckdtree,
                                    vis=False, print_summary=False)
                if sampling_technique == 'dart':
                    sp = mulpoint2d(method=sp_distr,
                                    gridding_technique=gridding_technique,
                                    sampling_technique=sampling_technique,
                                    char_length=char_length,
                                    xbound=xbound, ybound=ybound,
                                    bridson_sampling_k=niter,
                                    make_point_objects=make_point_objects,
                                    make_ckdtree=make_ckdtree,
                                    vis=False, print_summary=False)
            if gridding_technique == 'pds':
                if sampling_technique == 'bridson1':
                    sp = mulpoint2d(method=sp_distr,
                                    gridding_technique=gridding_technique,
                                    sampling_technique=sampling_technique,
                                    xbound=xbound, ybound=ybound,
                                    char_length=char_length,
                                    bridson_sampling_k=ntp,
                                    make_point_objects=make_point_objects,
                                    make_ckdtree=make_ckdtree,
                                    vis=False, print_summary=False)

            seed_point_coords = np.vstack((sp.locx, sp.locy)).T
            xmin, xmax = xbound
            ymin, ymax = ybound
            pxtal = voronoi_diagram(ShMultiPoint(seed_point_coords),
                                    tolerance=0.0, edges=False)
            PXTAL_boundary = Polygon([[xmin, ymin], [xmax, ymin],
                                      [xmax, ymax], [xmin, ymax]])

            contained, contained_cropped = [], []
            for count in range(len(pxtal.geoms)):
                if pxtal.geoms[count].intersects(PXTAL_boundary):
                    contained.append(pxtal.geoms[count])
                    intr = pxtal.geoms[count].intersection
                    contained_cropped.append(intr(PXTAL_boundary))
            PXTAL_bound = MultiPolygon(contained_cropped)
        return {'sp': sp,
                'pxtal': PXTAL_bound,
                'bp': PXTAL_boundary,
                }

    def extract_shapely_coords_all_grains(self, instance=1, gslevel='base',
                                          mfname='c_rp',
                                          make_upxo_mp=True,
                                          make_upxo_mp_subfeatures=False):
        """
        Extract
        Parameters
        ----------
        instance: int, optional
            Default value is 1.

        mfname: str, optional
            Options inslude the following:
                * 'g_rp': grain representative point. This is a point
                gfaurenteed by shapely to be INSIDE a polygon, in the present
                case, the GRAIN.
                * 'g_cp': grain centroia point.
                * 'g_vp': grain vertx points.
            Default value is 'g_rp'

        Return
        ------
        points: dict
            keys include:
                * cid: coordinates arranged in order of cid
                * all: collation of points. Different when mfname is g_vp
                * mp_all: UPXO multi-POint2D of all collated coordinates
                * mp_sf: UPXO multi-POint2D of coordintes, sub-features wise

        Notes
        -----
        Coordinates in points['all'] and points['mp_all'] for g_vp may not be
        in the same order in the poly-xtal.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 225],
                 ybound = [0, 200], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)
        gset.plot()
        points = gset.extract_shapely_coords_all_grains(instance=1,
                                                        mfname='c_rp',
                                                        make_upxo_mp=True,)
        points.keys()
        """
        if gslevel in self._valid_gslevels_:
            cells = self.pxtals[instance].geoms
        else:
            raise ValueError('Invalid gslevel specification.')
        # ------------------------------------
        if mfname == 'c_rp':
            coords_cid = np.vstack([np.hstack(c.representative_point().xy)
                                    for c in cells])
            coords_all = coords_cid
        elif mfname == 'c_cp':
            coords_cid = np.vstack([np.hstack(c.centroid.xy) for c in cells])
            coords_all = coords_cid
        elif mfname == 'c_vp':
            coords_cid = [np.vstack(c.boundary.xy).T for c in cells]
            coords_all = np.unique(np.vstack(coords_cid), axis=0)
        # ------------------------------------
        if make_upxo_mp:
            mp_all = mp2d.from_coords(coords_all)
            if mfname in ('c_vp'):
                if make_upxo_mp_subfeatures:
                    mp_sf = [mp2d.from_coords(coords) for coords in coords_cid]
                else:
                    mp_sf = None
            else:
                mp_sf = None
        else:
            mp_all, mp_sf = None, None
        # ------------------------------------
        points = {'cid': coords_cid, 'all': coords_all, 'mp_all': mp_all,
                  'mp_sf': mp_sf}
        return points

    def add_supcell_instances(self, gslevel='base', instance=1,
                              cluster_On=2, super_cell_name='grain_clusters',
                              cell_sep_distance=4, n_supcell_instances=2,
                              rand_exclude_cluster_centre_cids=0.0,
                              exclude_cluster_centre_cids=[],
                              exclude_cluster_cids=[], ccif=1.0,
                              remove_duplicate_cells=True,
                              merge=True, sample_plot=False, throw=False):
        """
        Identify multiple instances of super-cells.

        A super-cell instance is a list of super-cell structures which are
        spatially selected such that the super-cell structures are seperated
        by cell_sep_distance. Here, the distance is the number of single cells.
        A super-cell structure is accompanied by a central cell and a group of
        other cells.

        Parameters
        ----------
        gslevel: str, optional
            Default value is 'base'.

        instance: int, optional
            Default value is 1.

        cluster_On: int, optional
            Number of nearest neighbours to be used all clusters.
            Default value is 2.

        super_cell_name='grain_clusters',

        cell_sep_distance: int, optional
            Specify the maximum seperation distance to be maintained between
            any two cluster centres. Default value is 2.

        n_supcell_instances: int, optional
            Number of super-cell cluster configurations to be produced. Each
            configuration will be a list of clusters, with each cluster having
            a cluster centre cid and a list of cids inclusive of both the cid
            itself and the cids of neighbouring cells of cluster centre cell.

        exclude_cluster_centre_cids: list, optional
            Default value is [].

        exclude_cluster_cids: list, optional
            Default value is [].

        ccif: float, optional
            Cluster cell inclusion factor, a factor determining how many random
            samples shall be drawn from the list of cluster cids of a given
            cid. Bounds: (eps, 1.0], where eps is a number which results in
            having at-least one cell in the cluster. The eps should be large
            for small clusters and could be small for large clusters. Default
            value is 1.0.

        merge: bool, optional
            Specify whether cells in a certain cluster is to be merged into a
            single cell. If True, the merge operation will be executed and will
            create a new list of pxtal instances (say PX). Each instance in PX
            can then independently become a vortess2d object in its own right!
            To do this, you can make use of the class method as,
            [from_shapely_mulpolygon(px) for px in PX]. Default value is False.

        sample_plot: bool, optional
            Default value is False.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 20, 'dev': 2.5, },
                            'consider_boundary_grains': True } }
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 100],
                 ybound = [0, 200], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[3.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        # Calculate the cluster instances
        cinst = gset.add_supcell_instances(gslevel='base', instance=1,
                 cluster_On=3, cell_sep_distance=2, n_supcell_instances=4,
                 rand_exclude_cluster_centre_cids=0.0,
                 exclude_cluster_centre_cids=[],
                 exclude_cluster_cids=[], ccif=1.0,
                 remove_duplicate_cells=True, merge=True, sample_plot=False)





        base_instance = 1
        supgs_instance = 0

        plt.figure(figsize=(5, 5), dpi=150)
        '''
        plt.figure(figsize=(5, 5), dpi=150)
        pxtal = gset.pxtals[base_instance]
        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
        plt.gca().set_aspect('equal')'''

        pxtal = gset.fdb_supset['pxtals'][supgs_instance]
        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
        plt.gca().set_aspect('equal')

        gset.plot_cells(instance=1, cids=cids)
        gset.plot_cells(instance=1, cids=supercell_centres[5])

        Data access
        -----------
        gset.fdb_supset.keys()
        >> dict_keys(['name', 'scells', 'scells_coll', 'pxtals', 'info',
                      'inst', 'ninst', 'nclust', 'ncells'])

        gset.fdb_supset['scells'].keys()  # Super cell instance IDs
        >> dict_keys([1, 2, 3, 4, 5])

        gset.fdb_supset['scells'][1].keys()
        >> dict_keys(['clids', 'cids', 'clusters'])

        gset.fdb_supset['scells'][1]['clids']  # Cluster IDs
        >> [1, 2, 3, 4, 5, 6, 7]

        gset.fdb_supset['scells'][1]['cids']  # Cluster wise centre cell IDs
        >> [131, 103, 8, 136, 45, 176, 184]

        gset.fdb_supset['scells'][1]['clusters']  # Cluster's centre cell IDs
        # [[cids of cells at & around 131],.. ,[cids of cells at & around 184]]
        >> [[159, 132, ..., 131],..., [162, 196, ..., 184]]

        gset.fdb_supset['scells_coll'].keys()  # Cluster IDs
        >> [1, 2, 3, 4, 5, 6, 7]

        gset.fdb_supset['scells_coll'][1]  # All cluster's centre cell IDs
        >> [159, 132, 134, 166,...,  182, 185, 189, 184]

        gset.fdb_supset['pxtals']  # Shapely Multi-Polygon objects.
        >> [<shapely.geometry.multipolygon.MultiPolygon at 0x18e858e4910>,
            <shapely.geometry.multipolygon.MultiPolygon at 0x18e858e4580>,
            <shapely.geometry.multipolygon.MultiPolygon at 0x18e8cbf2e50>,
            <shapely.geometry.multipolygon.MultiPolygon at 0x18e8cbf2a30>,
            <shapely.geometry.multipolygon.MultiPolygon at 0x18ea7a25ee0>]

        print(gset.fdb_supset['pxtals'][1].geoms[0])  # 1st instance 1st cell
        >> POLYGON ((4.001101245601904 92.2296115091435,
                     2.633867013140424 89.03895454375312,
                     0 88.9638129954714, 0 97.34088983027945,
                     4.001101245601904 92.2296115091435))

        gset.fdb_supset['info']
        >> {'super_cell_name': 'grain_clusters', 'source.gslevel': 'base',
            'source.instance': 1, 'cluster_On': 2, 'cell_sep_distance': 2,
            'n_supcell_instances': 5, 'rand_exclude_cluster_centre_cids': 0.0,
            'exclude_cluster_centre_cids': [], 'exclude_cluster_cids': [],
            'ccif': 1.0, 'remove_duplicate_cells': True}

        gset.fdb_supset['inst']  # Instance IDs
        >> dict_keys([1, 2, 3, 4, 5])

        gset.fdb_supset['ninst']  # Number of instances
        >> 5

        gset.fdb_supset['nclust']  # Number of clusters across instances
        >> [7, 7, 8, 8, 8]

        Exaplanations
        -------------
        Note: This uses the above data access. Please refer it before reading
        further.

        gset.fdb_supset['ninst'] is 5, meaning there are 5 instances of
        grain structures. Each instance has grain clusters. Each of the 5
        pxtals can be accessed in gset.fdb_supset['pxtals']. Now, we see that
        gset.fdb_supset['nclust'][pxtal_ID = 0] is 7. This means that in the
        pxtal gset.fdb_supset['pxtals'][pxtal_ID], 7 grain clusters have been
        created as 7 unique grains. That is, in each grain cluster, the n cell
        which form n participating elements of the cluster, have all been made
        a single cell through cell union operation. The participating cells in
        a particular cluster, say the 1st cluster, i.e. the cluster
        gset.fdb_supset['scells'][supcell instance ID = 1]['clids'][0] has the
        cell ID gset.fdb_supset['scells'][supcell instance ID = 1]['cids'][0]
        as 131. This means that the centre cell of the 1st cell cluster, is
        cid = 131. Now,
        gset.fdb_supset['scells'][supcell inst ID = 1]['clusters'][0], is
        [159, 132, 134, 166, 167, 74, 111, 145, 114, 150, 119, 131]. Lets call
        it as Ci_Cells. These are all cells which cluster aroung cid of 131.
        The actual cells which have been unioned (i.e. merged) are
        [gset.pxtals[1].geoms[i] for i in Ci_Cells], where
        Ci_Cells = gset.fdb_supset['scells'][1]['clusters'][0]. Lets call this
        Ci_Cells_geom. Ci_Cells_geom is a list of Shapely polygon objects.
        Its multi-polygon can be easily created as MultiPolygon(Ci_Cells_geom).
        """
        _info_ = {'super_cell_name': super_cell_name,
                  'source.gslevel': gslevel, 'source.instance': instance,
                  'cluster_On': cluster_On,
                  'cell_sep_distance': cell_sep_distance,
                  'n_supcell_instances': n_supcell_instances,
                  'rand_exclude_cluster_centre_cids': rand_exclude_cluster_centre_cids,
                  'exclude_cluster_centre_cids': exclude_cluster_centre_cids,
                  'exclude_cluster_cids': exclude_cluster_cids, 'ccif': ccif,
                  'remove_duplicate_cells': remove_duplicate_cells}
        # -------------------------------------------------
        non = self.find_On_neigh(gslevel=gslevel, instance=instance,
                                 On=int(cluster_On)+0.000000001,
                                 n_instances=1)
        non_mod = {k-1: [_v-1 for _v in v] for k, v in non[0].items()}
        # -------------------------------------------------
        '''Make sure cid is its own neighbour as a trivial.'''
        for cid, neigh in non_mod.items():
            if cid not in neigh:
                non_mod[cid].append(cid)
        # -------------------------------------------------
        '''Implement cell exclusion rule for cluster centre cells.'''
        if len(exclude_cluster_centre_cids) != 0:
            for cid in non_mod.keys():
                if cid in exclude_cluster_centre_cids:
                    del non_mod[cid]
        # -------------------------------------------------
        '''Build a network-x graph to allow us select the optimum spatial
        distribution of cells.'''
        G = make_gid_net_from_neighlist(non_mod)
        # -------------------------------------------------
        print(40*'-', f'\nComputing {n_supcell_instances} instances of',
              'supercell arrangements.')
        supcell_centres = {}
        for inst in range(n_supcell_instances):
            if not inst % 10:
                print(f'instance no. {inst} of {n_supcell_instances}')
            selected = set()
            nodes = list(G.nodes())
            random.shuffle(nodes)
            for cell in nodes:
                ssspl = nx.single_source_shortest_path_length
                reachable = ssspl(G, cell, cutoff=cell_sep_distance)
                if not selected.intersection(reachable.keys()):
                    selected.add(cell)
            supcell_centres[inst+1] = list(selected)
        # -------------------------------------------------
        '''Exclude the cluster centre cells as per specification.'''
        if 0 < rand_exclude_cluster_centre_cids < 1:
            for inst in range(n_supcell_instances):
                # Number of cluster centres
                ncc = len(supcell_centres[inst+1])
                _k = 1-rand_exclude_cluster_centre_cids
                ncc = np.round(_k*ncc).astype(np.int16)
                _nprc_ = np.random.choice
                supcell_centres[inst+1] = _nprc_(supcell_centres[inst+1],
                                                 ncc,
                                                 replace=False)
        # -------------------------------------------------
        print('Building supercell data-structure')
        scells = {sset_i: {} for sset_i in range(1, n_supcell_instances+1)}
        scells_coll = {sset_i: None
                       for sset_i in range(1, n_supcell_instances+1)}
        pxmerged = []
        for sset_i in range(1, n_supcell_instances+1):
            # sset_i = 1
            '''Get clusters of this supercell pxtal instance.'''
            _clusters_ = [non_mod[cid] for cid in supcell_centres[sset_i]]
            # --------------------
            '''Remove duplicates if specified by user.'''
            if remove_duplicate_cells:
                _cid_ = set()
                new_clusters = []
                for _clst_ in _clusters_:
                    new_clst = []
                    for item in _clst_:
                        if item not in _cid_:
                            new_clst.append(item)
                            _cid_.add(item)
                    new_clusters.append(new_clst)
                _clusters_ = new_clusters
            # --------------------
            if len(exclude_cluster_cids) != 0:
                # <<<<<<< TO DO >>>>>>>
                pass
            # --------------------
            if 0 < ccif < 1:
                # ccif: Cluster cell Inclusion Factor
                # <<<<<<< TO DO >>>>>>>
                pass
            # --------------------
            '''handle empty _clusters_ entries.'''
            for i, _c_ in enumerate(_clusters_, start=0):
                if len(_c_) == 0:
                    _clusters_[i].append(supcell_centres[sset_i][i])
            # --------------------
            '''Collect clusters.'''
            clusters_coll = np.hstack(_clusters_).astype(np.int16).tolist()
            '''Build cluster IDs.'''
            _clids_ = list(range(1, len(supcell_centres[sset_i])+1))
            # --------------------
            '''Form the super cell dictionary. This will have the following
            keys:
                * 'clids'. Cluster IDs. Starts at 1 and increments by 1.
                * 'cids'. Cell IDs of the cluster's central cell.
                * 'clusters'. List of [list of cell IDs in each cluster].
            '''
            scells[sset_i] = {'clids': _clids_,
                              'cids': supcell_centres[sset_i],
                              'clusters': _clusters_
                              }
            scells_coll[sset_i] = clusters_coll
            # --------------------
            '''Form merged poly-xtal instance.'''
            if merge and remove_duplicate_cells:
                pxtcells = self.pxtals[instance].geoms
                old_cells = {cid: c for cid, c in enumerate(pxtcells, start=0)}
                '''Build list of cells which will not be merged.'''
                no_merges = []
                for cid in old_cells.keys():
                    if cid not in scells_coll[sset_i]:
                        no_merges.append(old_cells[cid])
                '''Build the cells which should be merged and carry out the
                cell mergers.'''
                if len(no_merges) == 0:
                    mpxinst = None
                else:
                    '''Merge the cells'''
                    merged_cells = []
                    for scell in scells[sset_i]['clusters']:
                        if len(scell) > 0:
                            '''Merge only if there are cells to merge.'''
                            cells_to_merge = [old_cells[cid] for cid in scell]
                            _merged_cell_ = unary_union(cells_to_merge)
                            merged_cells.append(_merged_cell_)
                        else:
                            '''If not, then just update cell itself as the
                            merged cell. TO DO: To check if this is really
                            needed.'''
                            merged_cells.append(_merged_cell_)
                    '''Form the merged poly-xtal instance.'''
                    mpxinst = MultiPolygon(no_merges + merged_cells)
                pxmerged.append(mpxinst)
            # --------------------
        '''Note that any pre-existing data in self.fdb_supset will be
        re-written.'''
        if not hasattr(self, 'fdb_supset'):
            self.fdb_supset = {}
        elif not isinstance(self.fdb_supset, dict):
            self.fdb_supset = {}
        # -------------------------------
        '''Create the Super-Cell Feature Database.'''
        if self.fdb_supset:
            self.fdb_supset = {}
        self.fdb_supset['name'] = super_cell_name
        self.fdb_supset['scells'] = scells
        self.fdb_supset['scells_coll'] = scells_coll
        self.fdb_supset['pxtals'] = pxmerged
        self.fdb_supset['info'] = _info_
        self.fdb_supset['inst'] = scells.keys()
        self.fdb_supset['ninst'] = len(scells.keys())
        self.fdb_supset['nclust'] = [len(scells[i]['clids'])
                                     for i in scells.keys()]
        ncells = []
        for i in scells.keys():
            _ncells_ = []
            for _cl_ in scells[i]['clusters']:
                _ncells_.append(len(_cl_))
            ncells.append(_ncells_)
        self.fdb_supset['ncells'] = ncells
        # -------------------------------
        '''Plot a sample if user requested for a plot.'''
        if sample_plot and merge and remove_duplicate_cells:
            for i in range(len(pxmerged)):
                pxtal = pxmerged[i]
                cells = pxtal.geoms
                coords_cid = [np.vstack(c.boundary.xy).T for c in cells]
                plt.figure(figsize=(5, 5), dpi=75)
                for vc in coords_cid:
                    plt.fill(vc[:, 0], vc[:, 1])
                    plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=2)
                    plt.gca().set_aspect('equal')
        # -------------------------------
        if throw:
            return self.fdb_supset

    def partition_cells(self, gslevel='base', source_gslevel_Instance_ID=1,
                        instance_ID_source=1,
                        reset=False, instance_ID_partition=1,
                        cids_sd=[],
                        partition_type='Sub-cell-0',
                        arg={'cid_offset': -1, 'spn': 150,
                             'id_format': 'from_cid_max',
                             'seed_lattice_type': 'rand_uni',
                             'combine_small_subs': False,
                             '_buffer_factor_': 0.8,
                             '_min_subcell_area_factor_': 0.04,
                             'niter_max': 1000, '_force_select_': True
                             },
                        saa=True,
                        plot_pxtal=False,
                        throw_pxtal=False,
                        ):
        """
        gslevel: str
            Specify the grain structure level. Default value is 'base'.

        instance_ID: int
            Specify the grain structure instance. Default value is 1.

        cids_sd: Iterable
            Default value if [].

        partition_type: str
            Specify the type of cell partitioning. Following options are
            available.
                1. 'Sub-cell-0'. Voronoi-subdivision
                2. 'Sub-cell-1'. Bridson's sampling
                3. 'Sub-cell-2'. Dart sampling.
                4. 'twin-1'.
                5. 'blocks-1'.
                6. 'sub-blocks-1'.
                7. 'twins-1'.

        arg: dict
            Specify additional srguments as a dictionary.
            1.  When the partition_type is 'sub-cell-0', the following
                dictionary must be used.

                {'cid_offset': -1,
                 'spn': 100,
                 'id_format': 'from_cid_max',
                 'seed_lattice_type': 'rand_uni',
                 'combine_small_subs': False,
                 '_buffer_factor_': 0.8,
                 '_min_subcell_area_factor_': 0.09,
                 'niter_max': 1000,
                 '_force_select_': True
                 }

                Explanations
                * cid_offset.
                    Specify the maximum cid
                    Default value is -1 (Recommended).
                * spn.
                    Default value is 100.
                * id_format.
                    Default value is 'from_cid_max'.
                * seed_lattice_type.
                    Default value is 'rand_uni'.
                * combine_small_subs.
                    Default value is False.
                * _buffer_factor_.
                    Default value is 0.8.
                * _min_subcell_area_factor_.
                    Default value is 0.09.
                * niter_max.
                    Default value is 1000.
                * _force_select_.
                    Default value is True.

        saa: bool, optional
            Default value is True.

        plot_pxtal: bool, optional
            Default value is False.

        throw_pxtal: bool, optional
            Default value is False.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5, },
                            'consider_boundary_grains': True } }

        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 75],
                 ybound = [0, 75], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[5.],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        '''
        instance_ID_source = 1
        pxtal = gset.pxtals[instance_ID_source]
        plt.figure(figsize=(5, 5), dpi=150)
        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
        plt.gca().set_aspect('equal')
        '''

        cids_sd = np.random.choice(range(len(gset.pxtals[1].geoms)), 30, replace=False)

        sc = gset.partition_cells(gslevel='base', instance_ID_source=1,
                                  instance_ID_partition=1,
                                  cids_sd=cids_sd,
                            partition_type='Sub-cell-0',
                            arg={'cid_offset': -1, 'spn': 10,
                                 'id_format': 'from_cid_max',
                                 'seed_lattice_type': 'rand_uni',
                                 'combine_small_subs': False,
                                 '_buffer_factor_': 0.8,
                                 '_min_subcell_area_factor_': 0.09,
                                 'niter_max': 1000, '_force_select_': True},
                            saa=True, plot_pxtal=True, throw_pxtal=True)

        lets gather all the sub-cell IDs. Lets just get all those from whose
        parent cells have atleast 2 subcells.

        cids_sd_level_1 = []
        for _sc_ in sc['fmap'].values():
            if len(_sc_) > 1:
                cids_sd_level_1 += _sc_

        sc = gset.partition_cells(gslevel='subgrains', instance_ID_source=1,
                                  instance_ID_partition=2,
                                  cids_sd=cids_sd_level_1,
                            partition_type='Sub-cell-0',
                            arg={'cid_offset': -1, 'spn': 10,
                                 'id_format': 'from_cid_max',
                                 'seed_lattice_type': 'rand_uni',
                                 'combine_small_subs': False,
                                 '_buffer_factor_': 0.8,
                                 '_min_subcell_area_factor_': 0.04,
                                 'niter_max': 1000, '_force_select_': True},
                            saa=True, plot_pxtal=True, throw_pxtal=True)


        cids_sd_level_1 = []
        for _sc_ in sc['fmap'].values():
            if len(_sc_) > 1:
                cids_sd_level_1 += _sc_

        sc = gset.partition_cells(gslevel='subgrains', instance_ID_source=2,
                                  instance_ID_partition=3,
                                  cids_sd=cids_sd_level_1,
                            partition_type='Sub-cell-0',
                            arg={'cid_offset': -1, 'spn': 10,
                                 'id_format': 'from_cid_max',
                                 'seed_lattice_type': 'rand_uni',
                                 'combine_small_subs': False,
                                 '_buffer_factor_': 0.8,
                                 '_min_subcell_area_factor_': 0.04,
                                 'niter_max': 1000, '_force_select_': True},
                            saa=True, plot_pxtal=True, throw_pxtal=True)

        """
        # cids_sd=[]
        '''Gather all cids in specified gslevel and instance_ID_source.'''
        cids_all = self.get_all_cids(gslevel=gslevel,
                                     instance_ID=instance_ID_source)
        cells_all, _ = self.get_cells(gslevel=gslevel,
                                      instance_ID=instance_ID_source,
                                      cids=cids_all, validate=True).values()
        # ---------------------------------------------------------------------
        '''cells_sd: cells to be sub-divided.
        cids_sd: cids of cells to be sub-divided.'''
        # cids_sd = cids_all[1:]
        cells_sd, cids_sd = self.get_cells(gslevel=gslevel,
                                           instance_ID=instance_ID_source,
                                           cids=cids_sd,
                                           validate=True).values()
        # ---------------------------------------------------------------------
        '''Template dictionartty to store subcells and subcell cids.'''
        subcells = {cid: {'cells': None, 'cids': None} for cid in cids_all}
        # ---------------------------------------------------------------------
        '''Lets first deal with all those which are not getting sub-divided.'''
        for cid_all in cids_all:
            if cid_all not in cids_sd:
                subcells[cid_all]['cells'] = [cells_all[cid_all]]
                subcells[cid_all]['cids'] = np.array([0])
        # ---------------------------------------------------------------------
        '''Now, sub-divide the cells in cells_sd into subcell sets and store
        each sub-cell set in reference to cid_sd of cids_sd.'''
        if partition_type == 'Sub-cell-0':
            for cid, cell in zip(cids_sd, cells_sd):
                print(f"Finding optimum subcell config for gslevel: '{gslevel}, ",
                      f"cell ID: {cid}")
                _tmp1_ = self.voronoi_subdivision
                _tmp2_ = '_min_subcell_area_factor_'
                sc = _tmp1_(cell, cid,
                            cid_offset=arg['cid_offset'],
                            spn=arg['spn'],
                            id_format=arg['id_format'],
                            seed_lattice_type=arg['seed_lattice_type'],
                            combine_small_subs=arg['combine_small_subs'],
                            _buffer_factor_=arg['_buffer_factor_'],
                            _min_subcell_area_factor_=arg[_tmp2_],
                            niter_max=arg['niter_max'],
                            _force_select_=arg['_force_select_'])
                subcells[cid]['cells'] = sc['cells']
                subcells[cid]['cids'] = np.array(sc['cids'])
                print('')
        elif partition_type == 'Sub-cell-1':
            pass
        # ---------------------------------------------------------------------
        '''Now, reset the local subcell IDs to global IDs'''
        # print(subcells)
        for i in cids_all[1:]:
            subcells[i]['cids'] += subcells[i-1]['cids'][-1] + 1
        # ---------------------------------------------------------------------
        '''We will turn cids back to list'''
        for i in cids_all:
            subcells[i]['cids'] = list(subcells[i]['cids'])
        # ---------------------------------------------------------------------
        '''Assemble all subcells to make the pxtal.'''
        subcells_all = []
        for i in cids_all:
            subcells_all += subcells[i]['cells']
        # ---------------------------------------------------------------------
        sc_cids = []
        for sg in subcells.values():
            sc_cids +=  sg['cids']
        # ---------------------------------------------------------------------
        '''Build the subcell id - parent id relationship'''
        scid_pcid = {}
        for sc in subcells.items():
            scid, pcid = sc[0], sc[1]['cids']
            for _pcid_ in pcid:
                scid_pcid[_pcid_] = scid
        # ---------------------------------------------------------------------
        print(10*'\n', cids_sd, 10*'\n')
        SC = {'gslevel_source': gslevel,
              'instance_ID_source': instance_ID_source,
              'instance_ID_partition': instance_ID_partition,
              'cids_sd': cids_sd,
              'sc': subcells,
              'scall': subcells_all,
              'pxtal': MultiPolygon(subcells_all) if throw_pxtal else None,
              'cids': sc_cids,
              'fmap': {i: sc['cids'] for i, sc in subcells.items()},
              'rmap': scid_pcid
              }
        if saa:
            self.fdb_subgrains[instance_ID_partition] = SC
        # ---------------------------------------------------------------------
        pxtal = self.pxtals[source_gslevel_Instance_ID]
        if plot_pxtal:
            plt.figure(figsize=(5, 5), dpi=150)
            coords_cid = [np.vstack(c.boundary.xy).T for c in subcells_all]
            for vc in coords_cid:
                #plt.fill(vc[:, 0], vc[:, 1])
                plt.plot(vc[:, 0], vc[:, 1], ':k', linewidth=0.5)

            coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
            for vc in coords_cid:
                plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
            plt.gca().set_aspect('equal')
        # ---------------------------------------------------------------------
        return SC

    @staticmethod
    def voronoi_subdivision(cell, cid, cid_offset=-1,
                            spn=100, id_format='from_cid_max',
                            seed_lattice_type='rand_uni',
                            combine_small_subs=False,
                            _buffer_factor_=0.8,
                            _min_subcell_area_factor_=-1,
                            niter_max=1000, _force_select_=True
                            ):
        '''
        Divide a cell in sub-cells using seed point basd Voronoi Tessellation.

        xtal_object: shapely polygon

        spn:
            number of seed points for voronoi tessellation inside this grain
            type: shapely MultiPoint
        seed_lattice_type:
            distribution of seed points
            type: str
            values: 'ru', 'hex', 'tri', 'rec', 'rn_xtal_centroid'
        combine_small_subs:
            Whether to combine small Voronoi cells with neighbours
            type: bool - True / False

        Access:
            voronoi_subdivision(pxtal[2], seed_points, combine_small_subs)

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 50],
                 ybound = [0, 50], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)
        # ----------------------------------------------
        cell, cid = gset.pxtals[1].geoms[1], 1
        sc = gset.voronoi_subdivision(cell, cid, cid_offset=-1,
                                spn=100, id_format='from_cid_max',
                                seed_lattice_type='rand_uni',
                                combine_small_subs=True,
                                _buffer_factor_=0.8,)
        len(sc['cells'])
        >> 31  # the 'cell' has been subdivided into 31 sub-cells.

        sc['pxtal']  # View the actual subcell structures on command line.
        '''
        _pi_ = np.pi
        _sqrt_ = np.sqrt
        _urd_ = np.random.uniform
        # ----------------------------------------------------------
        if spn < 11:
            min_sc_afac_MAX = 0.25
        elif 11 <= spn < 25:
            min_sc_afac_MAX = 0.17
        elif 26 <= spn < 50:
            min_sc_afac_MAX = 0.12
        elif spn >= 50:
            min_sc_afac_MAX = 0.09
        # ----------------------------------------------------------
        def _form_seed_points_(xmin, xmax, ymin, ymax, spn):
            '''Form the shapely multi-point object will be used to make the
            bounding voronoi tessellation.'''
            if seed_lattice_type == 'rand_uni':
                x = _urd_(xmin, xmax, spn)
                y = _urd_(ymin, ymax, spn)
                sp = ShMultiPoint([[x[i], y[i]] for i in range(spn)])
            elif seed_lattice_type == 'rand_nrm':
                pass
            elif seed_lattice_type == 'bridson':
                pass
            elif seed_lattice_type == 'dart':
                pass
            elif seed_lattice_type == 'lattice.sq':
                pass
            elif seed_lattice_type == 'lattice.tri':
                pass
            elif seed_lattice_type == 'lattice.hex':
                pass
            return sp
        # ----------------------------------------------------------
        def _form_subcells_(sp, cell):
            '''Break the cell into sub-cells. The seed points will be used to
            create an overlay bounding Voronoi tessellation. Intersections b/w
            input cell and the tess cells will be ca;lculated. Those which
            intersect will be kept (with necessary crops). The ones which are
            kept are the ones which are the subcells.'''

            '''Build the Bounding Voronoi tessellation.'''
            BVT = voronoi_diagram(sp, tolerance=0.0, edges=False)
            """"Filter the BVT for necessary cells."""
            '''Build the temporary subcell list.'''
            _sc_ = []
            for count in range(len(BVT.geoms)):
                if BVT.geoms[count].intersects(cell):
                    _sc_.append(BVT.geoms[count].intersection(cell))
            '''Assess the above temporary sub-cell list and sub-select
            based on user specified condition.'''
            sc = _sc_
            return sc
        # ----------------------------------------------------------
        '''Estimate buffer distance for the cell.'''
        buff_dist = _buffer_factor_*_sqrt_(cell.area/_pi_)
        '''Get the buffered cell. This cell will be bigger then the input cell
        but roughly the same shape.'''
        cell_buffered = cell.buffer(buff_dist)
        '''Build cell bounds.'''
        x_bounds = cell_buffered.boundary.xy[0]
        y_bounds = cell_buffered.boundary.xy[1]
        xmin, xmax = min(x_bounds), max(x_bounds)
        ymin, ymax = min(y_bounds), max(y_bounds)
        '''Build the seed points.'''
        if _min_subcell_area_factor_ == -1:
            sp = _form_seed_points_(xmin, xmax, ymin, ymax, spn)
            # sc = _form_subcells_(sp, cell)
        elif 0 < _min_subcell_area_factor_ <= min_sc_afac_MAX:
            '''This means, decisions will be made to select or regenerate a set
            of sub-cells, based on whether a subcell set satisfies the user
            specified condition or not. Multiple iterations willbe performed.
            if an iteration yields the optyimum solution, iterations will stop.
            the iterations can proceed for the specified maximum number. If no
            optimum solutions are reached by the end of the maximum iterations,
            then two choices are used. 1st choice is to the ub-select amongst e
            the existing set of supoints, which yields the sub-cell division
            which is closest to being optimum. The second is an error will be
            reported that best solution could not be reached, which ofcourse
            is only if the user has the necessary conditions input to this
            function'''
            spsets = []
            afac = []
            isel = -1
            for i in range(niter_max):
                _sp_ = _form_seed_points_(xmin, xmax, ymin, ymax, spn)
                _sc_ = _form_subcells_(_sp_, cell)
                _areas_ = np.array([_c_.area for _c_ in _sc_])
                _amin_, _amean_ = _areas_.min(), _areas_.mean()
                spsets.append(_sp_)
                _afac_ = _amin_/_amean_
                afac.append(_afac_)
                if _amin_ > _min_subcell_area_factor_*_amean_:
                    isel = i
                    break
            if isel == -1:
                _loc_ = np.where(np.array(afac) >= _min_subcell_area_factor_)[0]
                if _force_select_:
                    if len(_loc_) == 0:
                        afac = np.array(afac)
                        _loc_ = np.where(afac == afac.max())[0]
                else:
                    raise ValueError('No seed point config. found !',
                                     ' Please re-try agsin or try using\n',
                                     'different parameters.')
                _loc_ = _loc_[0]
                isel = niter_max
            else:
                _loc_ = isel

            sp = spsets[_loc_]
            print(10*' ', f'No. of iterations: {isel}')
        else:
            raise ValueError('Invalid _min_subcell_area_factor_ specified.')

        sc = _form_subcells_(sp, cell)
        '''Form the cell IDs.'''
        if id_format == 'from_cid_max':
            cids = list(range(cid_offset+1, cid_offset+len(sc)+1))
        elif id_format == 'from_0':
            cids = list(range(0, len(sc)))
        print(10*' ', f'No. of sub-cells: {len(cids)}')
        '''Form the multi-polygon object if requested.'''
        if combine_small_subs:
            pxtal = MultiPolygon(sc)
        else:
            pxtal = None
        '''Build the return subcells dictionary.'''
        subcells = {'cells': sc,
                    'pxtal': pxtal,
                    'cids': cids}
        return subcells

    def get_pxtal(self, gslevel='base', instance_ID=1, validate=True):
        """
        Retrieve a specific poly-xtal object.

        Parameters
        ----------
        gslevel: str, optional
        instance_ID: str, optional
        """
        if validate:
            if not self.validate_instance_ID(gslevel, instance_ID):
                raise ValueError('Invalid gslevel, instance_ID combination.')
        # ------------------------------------------
        if gslevel == 'base':
            pxtal = self.pxtals[instance_ID]
        # ------------------------------------------
        return pxtal

    def get_cells(self, gslevel='base', instance_ID=1, cids=[0],
                  validate=True):
        """
        Extract specific cells from the hierarchy.

        Parameters
        ----------
        gslevel: str, optional
            Specify the grain structure level.
            Default value is 'base'.

        instance_ID: int, optional
            Specify the ID of the grain structure instance.
            Default value is 1.

        cids: Iterable, optional
            List of cell IDs.
            Default value is [0].

        validate: bool, optional
            Specify whether data validation is to be performed.
            Default value is True.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 50],
                 ybound = [0, 50], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)
        # ----------------------------------------------
        gset.get_cells(gslevel='base', instance_ID=1, cids=[0], validate=True)
        """
        # Validations
        if type(cids) not in dth.dt.ITERABLES:
            if type(cids) not in dth.dt.NUMBERS:
                cids = [cids]
            else:
                raise TypeError('Invalid cids value type.')
        else:
            # print(10*'\n', len(cids), 10*'\n')  # sunil
            if len(cids) > 0:
                if not all([isinstance(cid, int) or isinstance(cid, np.int32)
                            for cid in cids]):
                    raise TypeError('Invalid cids value type.')
        # ----------------------------------------------
        '''Extract the cells from the pxtal database.'''
        if gslevel == 'base':
            '''get the poly-xtal using gslevel and instance id.'''
            pxtal = self.get_pxtal(gslevel=gslevel,
                                   instance_ID=instance_ID,
                                   validate=validate)
            if len(cids) == 0:
                cells = list(pxtal.geoms)
                cids = list(range(len(cells)))
            else:
                cells = [pxtal.geoms[cid] for cid in cids]
        elif gslevel == 'subgrains':
            cells = [self.fdb_subgrains[instance_ID]['scall'][_cid_]
                     for _cid_ in cids]
            cids = [self.fdb_subgrains[instance_ID]['cids'][_cid_]
                    for _cid_ in cids]
        # ----------------------------------------------
        # Build the return object
        cells = {'cells': cells, 'cids': cids}
        # ----------------------------------------------

        return cells

    def get_all_cids(self, gslevel='base', instance_ID=1):
        if gslevel == 'base':
            cids = list(range(len(self.pxtals[instance_ID].geoms)))
        elif gslevel == 'subgrains':
            cids = list(self.fdb_subgrains[instance_ID]['cids'])

        return cids

    def convert_subgrains_to_paps(self, scinfo, pap_instance_ID=1):
        """
        Translate a (L0C : L1C-SC) GS to a (PAC : PAP) GS for FM steels.

        L0C: Level 0 Cell
        L1C-SC: Level 1 Cell - Sub-cell
        PAC: Prior Austenitic Cell
        PAPC: Prior Austenitic Packets

        Parameters
        ----------
        scinfo: dict
            Sub-cell information dictionary obtained from call to function
            'partition_cells'.

        pap_instance_ID: int, optional
            Instance ID of the PAP GS to create.

        Primary variable
        ----------------
        dfb_paps: dict
            This is a slot variable in UPXO. ~~~ DO NOT ALTER ~~~
            saa=True, will set up self.sfb_paps.

        Explanations
        ------------
        dfb_paps is primarily a 2-level dictionary. The 1st level contains
        instance and the second level contains the PAP GS info.

        dfb_paps can contain any number of instances of pap configurations. To
        add a new instabce, you will have to call this function with a new
        value for pap_instance_ID. If entered pap_instance_ID exists already as
        a key in dfb_paps, the existing PAP GS will be overwritten.

        To create a dbf_PAP using this function, a subgrains GS must have been
        previously created. To create the subgrains gs, you cna use the
        function gset.partition_cells(...) with input argument 'throw' set to
        True. The returned valkue is the input argument 'scinfo' to the presnt
        function call.

        Data structure of the primary variable
        --------------------------------------
        fdb_paps would have the following keys:
            * instance_ID: Any integer number, preferably >= 0.

        fdb_paps[instance_ID] would have the following keys:
            * gslevel_source. Same as SC['gslevel_source']. The level of the
            source grain structure used to make the subgrains, which inturn is
            fed to this function call.
            * instance_ID_source. Same as SC['instance_ID_source']. The
            instance ID number of the source grain structure.
            * instance_ID_paps. Same as SC['instance_ID_partition']. The
            instance
            * base_cid. Base grain structure cells.
            * pac_cid. Prior austenitic cells. These are cells which host the PAPs.
            * npac_cid. Non prior austenetic cells. These are cells which do not host PAPs.
            * papc_cid. PAP cells. List of lists.
            * papc_cum_cid. PAP cells. Single cumulated list.
            * fmap_cid. Forward map. Map PAPs to PACs. PAP IDs for every PAC ID.
            * rmap_cid. Reverse map. Map PACs to PAPs. PAC IDs for every PAP ID.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5, },
                            'consider_boundary_grains': True } }

        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 75],
                 ybound = [0, 75], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[5.],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        '''
        pxtal = gset.pxtals[instance_ID_source]
        plt.figure(figsize=(5, 5), dpi=150)
        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
        plt.gca().set_aspect('equal')
        '''

        cids_sd = np.random.choice(range(len(gset.pxtals[1].geoms)), 30, replace=False)
        sc = gset.partition_cells(gslevel='base', instance_ID_source=1,
                            instance_ID_partition=1,
                            cids_sd=cids_sd,
                            partition_type='Sub-cell-0',
                            arg={'cid_offset': -1, 'spn': 10,
                                 'id_format': 'from_cid_max',
                                 'seed_lattice_type': 'rand_uni',
                                 'combine_small_subs': False,
                                 '_buffer_factor_': 0.8,
                                 '_min_subcell_area_factor_': 0.09,
                                 'niter_max': 1000, '_force_select_': True},
                            saa=True, plot_pxtal=False, throw_pxtal=True)
        pap = gset.convert_subgrains_to_paps(sc, pap_instance_ID=1)
        """
        papiid = pap_instance_ID
        # ---------------------------------------------------------------------
        self.fdb_paps[papiid] = {}
        # ---------------------------------------------------------------------
        '''Meta Data.'''
        self.fdb_paps[papiid]['gslevel_source'] = scinfo['gslevel_source']
        self.fdb_paps[papiid]['instance_ID_source'] = scinfo['instance_ID_source']
        self.fdb_paps[papiid]['instance_ID_paps'] = scinfo['instance_ID_partition']
        # ---------------------------------------------------------------------
        '''IDs.'''
        # Base grain structure cells.
        self.fdb_paps[papiid]['base_cid'] = np.array(list(scinfo['sc'].keys()))
        # Prior austenitic cells. These are cells which host the PAPs.
        self.fdb_paps[papiid]['pac_cid'] = scinfo['cids_sd']
        # Non prior austenetic cells. These are cells which do not host PAPs.
        npac_id = set(self.fdb_paps[papiid]['base_cid']) - set(scinfo['cids_sd'])
        self.fdb_paps[papiid]['npac_cid'] = np.array(list(npac_id))
        # Prior austenitic packet cells
        self.fdb_paps[papiid]['papc_cid'] = [scinfo['sc'][i]['cids'] for i in scinfo['cids_sd']]
        self.fdb_paps[papiid]['papc_cum_cid'] = np.hstack([scinfo['sc'][i]['cids'] for i in scinfo['cids_sd']])
        # Forward map. Map PAPs to PACs.
        self.fdb_paps[papiid]['fmap_cid'] = scinfo['fmap']
        # Reverse map
        self.fdb_paps[papiid]['rmap_cid'] = scinfo['rmap']
        # ---------------------------------------------------------------------
        '''Poly-XTAL objects.'''
        # Base grain structure
        self.fdb_paps[papiid]['base_px'] = self.pxtals[scinfo['instance_ID_source']]
        self.fdb_paps[papiid]['pac_parents'] = scinfo['pxtal']
        # ---------------------------------------------------------------------

    def make_sub_blocks_in_blocks(self, pxtal_blocks, cids=[],
                                  pac_instance_ID=1, base_gs_instance_ID=1,
                                  plot_base=True,
                                  plot_pac=True, plot_blocks=True):
        fragments_all = []
        # POLYGONS = gset.pxtals[1].geoms
        POLYGONS = [g for g in pxtal_blocks]
        for cell_count, polygon in enumerate(POLYGONS):
            print(f'partitioning cell {cell_count}')
            mrr = polygon.minimum_rotated_rectangle
            cenx, ceny = polygon.centroid.xy
            cenx, ceny = cenx[0], ceny[0]
            mb = mrr.bounds  # mrrbounds
            x, y = np.array([mb[0], mb[2]]), np.array([mb[1], mb[3]])
            dx = mb[2]-mb[0]  # xmax - xmin
            dy = mb[3]-mb[1]  # ymax - ymin
            Ldiag = np.linalg.norm([dx, dy])
            # width, height = abs(x[1] - x[0]), abs(y[1] - y[0])
            # wmin = min(width, height)
            # =====================================================================
            A_blc = 360*np.random.random()  # Any angle

            _cdef_ = sl2d.by_LFAL  # Class definition i.e. class method
            L0 = _cdef_(location=[cenx, ceny], factor=0.5, angle=A_blc,
                        length=2*Ldiag, degree=True)
            L0.mid_coord
            f = L0.generate_factors_0_and_1(dx1=0.015, dx2=0.015, dmean=0.03, k=0.2, th_res=0.05)
            points, lines, mullines = L0.divide_at_ratios(f)
            normals = L0.distribute_normal_vectors(method='by_points', points=points)
            for normal in normals:
                normal.stretch(normal.mid_coord, 10)
            # L0.plot(sl2d=normals)

            normal_coords = [normal.coords for normal in normals]

            cut_lines = [LineString([(nc[0], nc[1]), (nc[2], nc[3])])
                         for nc in normal_coords]

            fragments = [polygon]
            for line in cut_lines:
                new_fragments = []
                for frag in fragments:
                    if frag.intersects(line):
                        split_result = split(frag, line)
                        new_fragments.extend(split_result.geoms)
                    else:
                        new_fragments.append(frag)
                fragments = new_fragments
            fragments_all += fragments

        pxtal_sb = MultiPolygon(fragments_all)

        plt.figure(figsize=(5, 5), dpi=200)
        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal_sb.geoms]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=0.5)

        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal_blocks.geoms]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '--r', linewidth=1.0)

        PAC = []
        for polid in self.fdb_paps[pac_instance_ID]['pac_cid']:
            PAC.append(self.fdb_paps[pac_instance_ID]['base_px'][polid])

        PAPC = []
        for polid in self.fdb_paps[pac_instance_ID]['papc_cum_cid']:
            PAPC.append(self.fdb_paps[pac_instance_ID]['pac_parents'][polid])


        coords_cid = [np.vstack(c.boundary.xy).T
                      for c in self.pxtals[base_gs_instance_ID].geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=1.25)

        coords_cid = [np.vstack(c.boundary.xy).T for c in PAPC]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=1.5)

        coords_cid = [np.vstack(c.boundary.xy).T for c in PAC]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=2)

        plt.gca().set_aspect('equal')

    def make_blocks_in_paps(self, plot_base=True, plot_pac=True,
                            pac_instance_ID=1, base_gs_instance_ID=1,
                            plot_pap=True):
        """
        Create the blocks inside PAPs of FM steel GS.

        Parameters
        ----------

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop = {'area': {'mean': {'val': 50, 'dev': 7.5, },
                              'consider_boundary_grains': True }
                     }

        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 100],
                 ybound = [0, 50], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[10],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,
                 )
        '''
        pxtal = gset.pxtals[1]
        plt.figure(figsize=(5, 5), dpi=200)
        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=2)
        plt.gca().set_aspect('equal')
        '''

        cids_sd = np.random.choice(range(len(gset.pxtals[1].geoms)),
                                   len(gset.pxtals[1].geoms),
                                   replace=False)

        sc = gset.partition_cells(gslevel='base', instance_ID_source=1,
                            instance_ID_partition=1, cids_sd=cids_sd,
                            partition_type='Sub-cell-0',
                            arg={'cid_offset': -1, 'spn': 10,
                                 'id_format': 'from_cid_max',
                                 'seed_lattice_type': 'rand_uni',
                                 'combine_small_subs': False,
                                 '_buffer_factor_': 0.8,
                                 '_min_subcell_area_factor_': 0.09,
                                 'niter_max': 1000, '_force_select_': True},
                            saa=True, plot_pxtal=False, throw_pxtal=True)
        gset.convert_subgrains_to_paps(sc, pap_instance_ID=1)

        gset.make_blocks_in_paps(plot_base=True, plot_pac=True)
        """
        paciiid = pac_instance_ID
        # ---------------------------------------------------------------------
        if plot_base:
            pxtal = gset.pxtals[base_gs_instance_ID]
            plt.figure(figsize=(5, 5), dpi=200)
            coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
            for vc in coords_cid:
                plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=2)
            plt.gca().set_aspect('equal')
        # ---------------------------------------------------------------------
        if plot_pac:
            plt.figure(figsize=(5, 5), dpi=200)
            PAC = []
            for polid in gset.fdb_paps[paciiid]['pac_cid']:
                PAC.append(gset.fdb_paps[paciiid]['base_px'][polid])

            coords_cid = [np.vstack(c.boundary.xy).T
                          for c in gset.fdb_paps[paciiid]['pac_parents']]
            for vc in coords_cid:
                plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=1.5)

            coords_cid = [np.vstack(c.boundary.xy).T for c in PAC]
            for vc_i, vc in enumerate(coords_cid):
                if vc_i % 50 == 0:
                    print(f'Plotting cell {vc_i}')
                plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=2)

            plt.gca().set_aspect('equal')
        # ---------------------------------------------------------------------
        fragments_all = []
        # POLYGONS = gset.pxtals[1].geoms
        POLYGONS = [gset.fdb_paps[paciiid]['pac_parents'][i]
                    for i in gset.fdb_paps[paciiid]['papc_cum_cid']]
        for cell_count, polygon in enumerate(POLYGONS):
            print(f'partitioning cell {cell_count}')
            mrr = polygon.minimum_rotated_rectangle
            cenx, ceny = polygon.centroid.xy
            cenx, ceny = cenx[0], ceny[0]
            mb = mrr.bounds  # mrrbounds
            x, y = np.array([mb[0], mb[2]]), np.array([mb[1], mb[3]])
            dx = mb[2]-mb[0]  # xmax - xmin
            dy = mb[3]-mb[1]  # ymax - ymin
            Ldiag = np.linalg.norm([dx, dy])
            # width, height = abs(x[1] - x[0]), abs(y[1] - y[0])
            # wmin = min(width, height)
            # =====================================================================
            A_blc = 360*np.random.random()  # Any angle

            _cdef_ = sl2d.by_LFAL  # Class definition i.e. class method
            L0 = _cdef_(location=[cenx, ceny], factor=0.5, angle=A_blc,
                        length=2*Ldiag, degree=True)
            L0.mid_coord
            f = L0.generate_factors_0_and_1(dx1=0.065, dx2=0.065, dmean=0.085, k=0.2, th_res=0.05)
            points, lines, mullines = L0.divide_at_ratios(f)
            normals = L0.distribute_normal_vectors(method='by_points', points=points)
            for normal in normals:
                normal.stretch(normal.mid_coord, 10)
            # L0.plot(sl2d=normals)

            normal_coords = [normal.coords for normal in normals]

            cut_lines = [LineString([(nc[0], nc[1]), (nc[2], nc[3])])
                         for nc in normal_coords]

            fragments = [polygon]
            for line in cut_lines:
                new_fragments = []
                for frag in fragments:
                    if frag.intersects(line):
                        split_result = split(frag, line)
                        new_fragments.extend(split_result.geoms)
                    else:
                        new_fragments.append(frag)
                fragments = new_fragments
            fragments_all += fragments


        for polid in gset.fdb_paps[paciiid]['npac_cid']:
            fragments_all.append(gset.fdb_paps[paciiid]['base_px'][polid])


        pxtal = MultiPolygon(fragments_all)

        plt.figure(figsize=(5, 5), dpi=200)
        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '--r', linewidth=1)

        PAC = []
        for polid in gset.fdb_paps[paciiid]['pac_cid']:
            PAC.append(gset.fdb_paps[paciiid]['base_px'][polid])

        PAPC = []
        for polid in gset.fdb_paps[paciiid]['papc_cum_cid']:
            PAPC.append(gset.fdb_paps[paciiid]['pac_parents'][polid])


        coords_cid = [np.vstack(c.boundary.xy).T
                      for c in gset.pxtals[base_gs_instance_ID].geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=1.25)

        coords_cid = [np.vstack(c.boundary.xy).T for c in PAPC]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=1.5)

        coords_cid = [np.vstack(c.boundary.xy).T for c in PAC]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=2)

        plt.gca().set_aspect('equal')
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # =====================================================================
        fragments_all = []
        # POLYGONS = gset.pxtals[1].geoms
        POLYGONS = [g for g in pxtal]
        for cell_count, polygon in enumerate(POLYGONS):
            print(f'partitioning cell {cell_count}')
            mrr = polygon.minimum_rotated_rectangle
            cenx, ceny = polygon.centroid.xy
            cenx, ceny = cenx[0], ceny[0]
            mb = mrr.bounds  # mrrbounds
            x, y = np.array([mb[0], mb[2]]), np.array([mb[1], mb[3]])
            dx = mb[2]-mb[0]  # xmax - xmin
            dy = mb[3]-mb[1]  # ymax - ymin
            Ldiag = np.linalg.norm([dx, dy])
            # width, height = abs(x[1] - x[0]), abs(y[1] - y[0])
            # wmin = min(width, height)
            # =====================================================================
            A_blc = 360*np.random.random()  # Any angle

            _cdef_ = sl2d.by_LFAL  # Class definition i.e. class method
            L0 = _cdef_(location=[cenx, ceny], factor=0.5, angle=A_blc,
                        length=2*Ldiag, degree=True)
            L0.mid_coord
            f = L0.generate_factors_0_and_1(dx1=0.015, dx2=0.015, dmean=0.03, k=0.2, th_res=0.05)
            points, lines, mullines = L0.divide_at_ratios(f)
            normals = L0.distribute_normal_vectors(method='by_points', points=points)
            for normal in normals:
                normal.stretch(normal.mid_coord, 10)
            # L0.plot(sl2d=normals)

            normal_coords = [normal.coords for normal in normals]

            cut_lines = [LineString([(nc[0], nc[1]), (nc[2], nc[3])])
                         for nc in normal_coords]

            fragments = [polygon]
            for line in cut_lines:
                new_fragments = []
                for frag in fragments:
                    if frag.intersects(line):
                        split_result = split(frag, line)
                        new_fragments.extend(split_result.geoms)
                    else:
                        new_fragments.append(frag)
                fragments = new_fragments
            fragments_all += fragments

        pxtal_sb = MultiPolygon(fragments_all)

        plt.figure(figsize=(5, 5), dpi=200)
        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal_sb.geoms]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=0.5)

        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '--r', linewidth=1.0)

        PAC = []
        for polid in gset.fdb_paps[paciiid]['pac_cid']:
            PAC.append(gset.fdb_paps[paciiid]['base_px'][polid])

        PAPC = []
        for polid in gset.fdb_paps[paciiid]['papc_cum_cid']:
            PAPC.append(gset.fdb_paps[paciiid]['pac_parents'][polid])


        coords_cid = [np.vstack(c.boundary.xy).T
                      for c in gset.pxtals[base_gs_instance_ID].geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=1.25)

        coords_cid = [np.vstack(c.boundary.xy).T for c in PAPC]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=1.5)

        coords_cid = [np.vstack(c.boundary.xy).T for c in PAC]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=2)

        plt.gca().set_aspect('equal')

        # =====================================================================
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        fragments_all = []
        # POLYGONS = gset.pxtals[1].geoms
        POLYGONS = [g for g in pxtal_sb]
        for cell_count, polygon in enumerate(POLYGONS):
            print(f'partitioning cell {cell_count}')
            mrr = polygon.minimum_rotated_rectangle
            cenx, ceny = polygon.centroid.xy
            cenx, ceny = cenx[0], ceny[0]
            mb = mrr.bounds  # mrrbounds
            x, y = np.array([mb[0], mb[2]]), np.array([mb[1], mb[3]])
            dx = mb[2]-mb[0]  # xmax - xmin
            dy = mb[3]-mb[1]  # ymax - ymin
            Ldiag = np.linalg.norm([dx, dy])
            # width, height = abs(x[1] - x[0]), abs(y[1] - y[0])
            # wmin = min(width, height)
            # =====================================================================
            A_blc = 360*np.random.random()  # Any angle

            _cdef_ = sl2d.by_LFAL  # Class definition i.e. class method
            L0 = _cdef_(location=[cenx, ceny], factor=0.5, angle=A_blc,
                        length=2*Ldiag, degree=True)
            L0.mid_coord
            f = L0.generate_factors_0_and_1(dx1=0.011, dx2=0.011, dmean=0.02, k=0.2, th_res=0.05)
            points, lines, mullines = L0.divide_at_ratios(f)
            normals = L0.distribute_normal_vectors(method='by_points', points=points)
            for normal in normals:
                normal.stretch(normal.mid_coord, 10)
            # L0.plot(sl2d=normals)

            normal_coords = [normal.coords for normal in normals]

            cut_lines = [LineString([(nc[0], nc[1]), (nc[2], nc[3])])
                         for nc in normal_coords]

            fragments = [polygon]
            for line in cut_lines:
                new_fragments = []
                for frag in fragments:
                    if frag.intersects(line):
                        split_result = split(frag, line)
                        new_fragments.extend(split_result.geoms)
                    else:
                        new_fragments.append(frag)
                fragments = new_fragments
            fragments_all += fragments

        pxtal_lath = MultiPolygon(fragments_all)

        plt.figure(figsize=(5, 5), dpi=200)
        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal_lath.geoms]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], ':g', linewidth=0.5)

        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal_sb.geoms]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=0.5)

        coords_cid = [np.vstack(c.boundary.xy).T for c in pxtal.geoms]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '--r', linewidth=1.0)

        PAC = []
        for polid in gset.fdb_paps[paciiid]['pac_cid']:
            PAC.append(gset.fdb_paps[paciiid]['base_px'][polid])

        PAPC = []
        for polid in gset.fdb_paps[paciiid]['papc_cum_cid']:
            PAPC.append(gset.fdb_paps[paciiid]['pac_parents'][polid])


        coords_cid = [np.vstack(c.boundary.xy).T
                      for c in gset.pxtals[base_gs_instance_ID].geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=1.25)

        coords_cid = [np.vstack(c.boundary.xy).T for c in PAPC]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=1.5)

        coords_cid = [np.vstack(c.boundary.xy).T for c in PAC]
        for vc_i, vc in enumerate(coords_cid):
            if vc_i % 50 == 0:
                print(f'Plotting cell {vc_i}')
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=2)

        plt.gca().set_aspect('equal')

        # =====================================================================
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def assign_loc_tags(self, instance=1, gslevel='base', tol=1E-6,
                        saa=True, throw=True):
        """
        Assign location tags to each cell feature of gslevel database.

        Parameters
        ----------
        """
        # Cell tags
        ct = {'inner': None,
              'boundary': None,
              'left': [],
              'bottom': [],
              'right': [],
              'top': [],
              'corners': None,
              'tl': None,
              'tr': None,
              'bl': None,
              'br': None
              }
        geolink = self.link_geom(instance=instance, saa=True, throw=True)
        for cid, vcs in enumerate(geolink['vc_cid'], start=1):
            if any(abs(self.uinputs['xbound'][0] - vcs[:, 0]) <= tol):
                ct['left'].append(cid)
            if any(abs(self.uinputs['xbound'][1] - vcs[:, 0]) <= tol):
                ct['right'].append(cid)
            if any(abs(self.uinputs['ybound'][0] - vcs[:, 1]) <= tol):
                ct['bottom'].append(cid)
            if any(abs(self.uinputs['ybound'][1] - vcs[:, 1]) <= tol):
                ct['top'].append(cid)
        ct['left'] = set(ct['left'])
        ct['right'] = set(ct['right'])
        ct['bottom'] = set(ct['bottom'])
        ct['top'] = set(ct['top'])
        ct['boundary'] = ct['left'] | ct['right'] | ct['bottom'] | ct['top']
        all_cids = set(range(1, len(geolink['vc_cid'])+1))
        ct['inner'] = all_cids - ct['boundary']
        ct['tl'] = ct['top'] & ct['left']
        ct['tr'] = ct['top'] & ct['right']
        ct['bl'] = ct['bottom'] & ct['left']
        ct['br'] = ct['bottom'] & ct['right']
        if saa:
            self.floc = dict(gslevel=ct)
        if throw:
            return ct

    def relate_cv_cids(self, instance=1, geolink={}, technique='dist',
                       use_trees=False, tol=1.E-6):
        """
        Relate cell vertices to IDs of cells, it is shared with.

        Use this to find the cells which share a common vertex. See example.

        geolink: dict, optional
            Specify the geometry linking dictionary. If empty, it will be
            caculated and used.

        technique: str, optional
            Specify the technique to establish vertex point sharing across
            cells. Options include the follwing:
                * mid: memory ids. This is quick as it uses just the equality
                of UPXo point2d object memory IDs.
                * dist: Distance based. This would be slower as one would
                check for point sharing based on zero-distance criteria,
                wherein two points seperated by a distance less than a
                specified tolerance value are considered as essentially
                the same points.
            Default value is 'dist'.

        use_trees: bool, optional
            Specify whether to construct trees for distance checking. May be
            slow if more cells exoist and cells contain small number of
            vertices. Only suitable when the number of vertices on each
            cell is very large. For example, a voronoi cell after having been
            roughned on its boundaries by grain boundary perturbations would
            possess a lartge number of vertex points. In these cases, it would
            be useful to use the tree structure. Otherwise, it is not
            recommended. On the contrary, if specified False, then the
            distances willbe claculated using numpy vectorised operations,
            which is very fast too!

        tol: float, optional
            Specifty the tolerance value useds to asess vertex point
            coincidence if techniqyue chosen in 'dist'.

        Returns
        -------
        vc_cids: list of lists
            An inner list provides the IDs of cells that the point at its index
            in vertex coordinate list, is shared with.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 125],
                 ybound = [0, 100], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        geolink = gset.link_geom(instance=1, saa=True, throw=True)

        cv_cid = gset.relate_cv_cids(instance=1, geolink={}, technique='mid',
                           use_trees=False, tol=1.E-6)
        # Choose a vertex ID
        vertex_id = 19

        cv_cid[vertex_id]
        >> [5, 6, 23]

        This means that the vertex number 19 of coordinates
        gset.geolink['vc'][19] (>> array([ 4.0866272 , 14.17379263])), is
        sharfed by the cells (a polygonal feature. This could be a grain, twin,
        etc.) 5, 6 and 23.
        """
        if not geolink:
            geolink = self.link_geom(instance=instance, saa=True, throw=True)

        if technique == 'mid':
            pmids = [id(up) for up in geolink['p2d']]
            pmids_fids = {fid: tuple([id(up) for up in vcs])
                          for fid, vcs in enumerate(geolink['p2d_cid'],
                                                    start=1)}
            vc_cids = []
            for pmid in pmids:
                pmid_shares = []
                for fid, pmid_fid in pmids_fids.items():
                    if pmid in pmid_fid:
                        pmid_shares.append(fid)
                vc_cids.append(pmid_shares)
        elif technique == 'dist' and use_trees:
            vcs = geolink['vc']
            vccidtrees = [ckdt(vcs) for vcs in geolink['vc_cid']]
            vc_cids = []
            for vc in vcs:
                vc_cid = []
                for cid, tree in enumerate(vccidtrees, start=1):
                    indices = tree.query_ball_point(vc, tol)
                    if indices:
                        vc_cid.append(cid)
                vc_cids.append(vc_cid)

        elif technique == 'dist' and not use_trees:
            vc_cids = []
            for vc in geolink['vc']:
                vc_cid = []
                for cid, vcs in enumerate(geolink['vc_cid'], start=1):
                    if np.any(np.sum((vc-vcs)**2, axis=1) <= tol**2):
                        vc_cid.append(cid)
                vc_cids.append(vc_cid)

        return vc_cids

    def _find_O1_neigh_(self, instance=1, gslevel='base',
                        cid_exc_rules_calc=[],
                        cid_exc_rules_results=[],
                        include_central_grain=True,
                        saa=True, throw=False):
        """
        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 125],
                 ybound = [0, 100], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)
        neigh_cids_O1 = gset._find_O1_neigh_(instance=1, gslevel='base',
                            cid_exc_rules_calc=[],
                            cid_exc_rules_results=[],
                            include_central_grain=True,
                            saa=True, throw=True)
        """
        geolink = self.link_geom(instance=instance, saa=True, throw=True)
        vc_cids = self.relate_cv_cids(instance=instance, geolink={},
                                      technique='mid',
                                      use_trees=False, tol=1.E-6)
        if len(cid_exc_rules_calc) > 0:
            for cerc in cid_exc_rules_calc:
                if not isinstance(cerc, str):
                    raise TypeError(f'Invalid cid_exclusion_rule: {cerc}')

        if len(cid_exc_rules_results) > 0:
            for cerr in cid_exc_rules_results:
                if not isinstance(cerr, str):
                    raise TypeError(f'Invalid cid_exclusion_rule: {cerr}')

        if len(cid_exc_rules_calc) > 0:
            for cer in cid_exc_rules_calc:
                if cer == 'pxb_all':
                    bcids = self.assign_loc_tags(instance=instance,
                                                 gslevel='base', tol=1E-6,
                                                 saa=True, throw=True)
                    cids = list(bcids['inner'])
        else:
            cids = [cid+1 for cid in range(len(geolink['vc_cid']))]


        neighs_cid = {cid: None for cid in cids}
        for cid in cids:
            shared_cids = [vc_cids[vid] for vid in geolink['vc_cid_ind'][cid-1]]
            neigh_cid = np.unique(np.hstack(shared_cids)).tolist()
            if not include_central_grain:
                neigh_cid.remove(cid)

            neighs_cid[cid] = neigh_cid


        if len(cid_exc_rules_results) > 0:
            for cerr in cid_exc_rules_results:
                if cer == 'pxb_all':
                    if len(cid_exc_rules_calc) == 0:
                        bcids = self.assign_loc_tags(instance=instance,
                                                     gslevel='base', tol=1E-6,
                                                     saa=True, throw=True)
                        _bcids_ = bcids['boundary']
                    for cid, neighs in neighs_cid.items():
                        neighs_cid[cid] = [neigh
                                           for neigh in neighs if neigh not in _bcids_]

        if saa:
            self.neighs_cid = neighs_cid

        if throw:
            return neighs_cid

    def find_On_neigh(self, instance=1, gslevel='base', On=1,
                      cid_exc_rules_calc=[],
                      cid_exc_rules_results=[],
                      include_central_grain=True,
                      saa=True, throw=True,
                      n_instances=1, __recall__=False
                      ):
        """
        Parameters
        ----------
        instance: int, optional
            Default value is 1.

        On: float, optional
            Default value is 1.

        cumulative: bool, optional
            Default value is True.

        find_for_boundary_grains: bool, optional
            Default value is True.

        include_boundary_grains: bool, optional
            Default value is True.

        Returns
        -------
        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 35],
                 ybound = [0, 30], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        non = gset.find_On_neigh(instance=2, gslevel='base', On=0.5,
                                 n_instances=1)
        non = gset.find_On_neigh(instance=2, gslevel='base', On=1.0,
                                 n_instances=1)

        non = gset.find_On_neigh(instance=2, gslevel='base', On=2.9,
                                 n_instances=3)
        """
        if not __recall__:
            print(40*'-', f'\nComputing {On} order neighbours of all grains.')
        # -------------------------------------------
        # On = 3
        # -------------------------------------------
        _cid_erc = cid_exc_rules_calc
        _cid_err = cid_exc_rules_results
        _icg = include_central_grain
        # -------------------------------------------
        if 0 < On < 1:
            _def_ = self._find_O1_neigh_
            neigh_cids_O1 = _def_(instance=instance, gslevel=gslevel,
                                  cid_exc_rules_calc=_cid_erc,
                                  cid_exc_rules_results=_cid_err,
                                  include_central_grain=_icg,
                                  saa=saa, throw=True)
            # Calculate counts based on propbabiltiies based on neigh lengths.
            pcount = {cid: None for cid in neigh_cids_O1.keys()}
            for cid in neigh_cids_O1.keys():
                nneigh = len(neigh_cids_O1[cid])
                pcount[cid] = np.round(On * nneigh).astype(np.int32)
            # Select neighbours
            geolink = self.link_geom(instance=1, saa=True, throw=True)
            ncells = len(geolink['vc_cid_ind'])
            neighs_On = {inst: {cid: cid for cid in range(1, ncells+1)}
                         for inst in range(n_instances)}
            for ni in range(n_instances):
                for cid, neighs in neigh_cids_O1.items():
                    neighs_ = np.random.choice(neigh_cids_O1[cid],
                                               pcount[cid],
                                               replace=False)
                    neighs_On[ni][cid] = neighs_.tolist()

        elif not On % 1:
            '''This is when the user enters On, multiples of 1.'''
            if On == 0:
                geolink = self.link_geom(instance=1, saa=True, throw=True)
                ncells = len(geolink['vc_cid_ind'])
                neighs_On = {cid: cid for cid in range(1, ncells+1)}
            elif On == 1:
                _def_ = self._find_O1_neigh_
                neigh_cids_O1 = _def_(instance=instance, gslevel=gslevel,
                                      cid_exc_rules_calc=_cid_erc,
                                      cid_exc_rules_results=_cid_err,
                                      include_central_grain=_icg,
                                      saa=saa, throw=True)
                neighs_On = neigh_cids_O1
            elif On > 1:
                for i in range(2, int(On)+1):
                    if i == 2:
                        _def1_ = self._find_O1_neigh_
                        neighs_On = _def1_(instance=instance,
                                           gslevel=gslevel,
                                           cid_exc_rules_calc=_cid_erc,
                                           cid_exc_rules_results=_cid_err,
                                           include_central_grain=_icg,
                                           saa=False, throw=True)
                    _def2_ = self.find_neigh_nplus1
                    neighs_On = _def2_(neighs_On,
                                      include_central_grain=_icg)
        elif On > 1 and On % 1 != 0:
            '''This is when user is actually tryoin to get a probability value.
            '''
            _def1_ = self.find_On_neigh
            On_floored = np.floor(On).astype(np.int8)
            neighs_On = _def1_(instance=instance,
                               gslevel=gslevel,
                               On=On_floored,
                               cid_exc_rules_calc=cid_exc_rules_calc,
                               cid_exc_rules_results=cid_exc_rules_results,
                               include_central_grain=include_central_grain,
                               saa=saa, throw=throw,
                               n_instances=n_instances,
                               __recall__=True)

            neighs_Onp1 = self.find_neigh_nplus1(neighs_On,
                                                 include_central_grain=_icg)

            neighs_On_Onp1 = {cid: None for cid in neighs_On.keys()}
            for cid in neighs_On.keys():
                On_th = set(neighs_Onp1[cid]) - set(neighs_On[cid])
                neighs_On_Onp1[cid] = list(On_th)

            # Calculate counts based on propbabiltiies based on neigh lengths.
            pcount = {cid: None for cid in neighs_On_Onp1.keys()}
            delOn = On - np.floor(On)
            print(On, delOn)
            for cid in neighs_On_Onp1.keys():
                nneigh = len(neighs_On_Onp1[cid])
                pcount[cid] = np.round(delOn * nneigh).astype(np.int32)

            neighs_On = {inst: neighs_On for inst in range(n_instances)}
            for inst in neighs_On.keys():
                for cid in neighs_On[0].keys():
                    neigh_On_floor = set(neighs_On[inst][cid])
                    neigh_On_Onp1 = neighs_On_Onp1[cid]
                    _def2_ = np.random.choice
                    neigh_On_Onp1_prob = _def2_(neigh_On_Onp1, pcount[cid],
                                                replace=False)
                    a, b = neigh_On_floor, set(neigh_On_Onp1_prob)
                    neighs_On[inst][cid] = list(a | b)

        return neighs_On

    def find_neigh_nplus1(self, neighs_On, include_central_grain=True):
        """
        neighs_Onplus1 = gset.find_neigh_nplus1(neighs_On,
                                               include_central_grain=True)
        """
        neighs_Onplus1 = {cid: None for cid in neighs_On.keys()}
        for cid, neighs in neighs_On.items():
            neighs_cumulative = []
            for neigh in neighs:
                neighs_cumulative.append(neighs_On[neigh])
            neighs = np.unique(np.hstack(neighs_cumulative)).tolist()
            if include_central_grain:
                neighs.remove(cid)
            neighs_Onplus1[cid] = neighs

        return neighs_Onplus1

    def setup_fid(self, pxnames=['twid', 'prid', 'tcid']):
        """
        Below are the valid pxnames:
            * 'scid': Sub-cell ID
            * 'twid': Twin ID
            * 'prid': Precipitate ID
            * 'clid': Grain cluster ID. Use this to create cell clusters
                needed to represent Prior Austenitic Packets, grain cluster
                in materials such as CuCrZr, etc.
            * 'tcid': Texture component ID.

        The following pxname will be automateically included:
            * 'phid': Phase ID
            * 'bcid': Base cell ID. Usually represents the graoin.
            * 'gfid': Gloabl feature ID
            * 'hlid': Hierarchy level ID

        Please refer to the below explanations.
            1. 'phid': Phase ID. Every cell will have a phase ID associated to
            it. It can be accessed as gset.fid[instance]['phid'].
            2. 'bcid': Every grain, at first creation will be allocated a base
            grain ID. This will be the 1st order hierarchical feature. All
            other morphologcal subfeatures will be expressed in relation to
            this. It can be accessed as gset.fid[instance]['bcid'].
            3. 'gfid': Gloabl feature ID.
                It can be accessed as gset.fid[instance]['gfid'].
            4. 'hlid': Hierarchy level ID
                It can be accessed as gset.fid[instance]['hlid'].

            * 'scid': Sub-cell win ID.
            * 'twid': Twin ID
            * 'prid': Precipitate ID
            * 'clid': Grain cluster ID. Use this to create cell clusters
                needed to represent Prior Austenitic Packets, grain cluster
                in materials such as CuCrZr, etc.
            * 'tcid': Texture component ID.
        """
        self.fid = {inst: {'phid': None,
                           'bcid': None,
                           'gfid': None,
                           'hlid': None
                           } for inst in range(self.ninst)}

    def get_stat(self, data, nan_policy='omit'):
        statistics = stats.describe(data, axis=None, nan_policy=nan_policy)
        return {'min': statistics.minmax[0],
                'max': statistics.minmax[1],
                'mean': statistics.mean,
                'variance': statistics.variance,
                'skewness': statistics.skewness,
                'kurtosis': statistics.kurtosis,
                'std': data.std(),
                'median': np.median(data, axis=None),
                'p25': np.percentile(data, 25),
                'p75': np.percentile(data, 75)
                }

    def setup_mprop(self):
        self.mprop = {}
        self.mprop['area'] = {1: {'prop_type': None, 'val': None, 'stat': None}}
        self.mprop['ar'] = {1: {'prop_type': None, 'val': None, 'stat': None}}

    def setup_tprop(self):
        self.tprop = {}
        self.tprop['neigh'] = {1: {0.00: None,
                                   0.50: None,
                                   0.75: None}}
        self.tprop['nneigh'] = {1: {0.00: None,
                                    0.50: None,
                                    0.75: None}}

    def find_areas(self, instance=1, gslevel='base', saa=True, throw=False):
        """
        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 125],
                 ybound = [0, 100], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=20, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        ameans, astd = [], []
        for inst in range(gset.ninst):
            area = gset.find_areas(instance=inst+1, gslevel='base', throw=True)
            ameans.append(area['stat']['mean'])
            astd.append(area['stat']['std'])

        print(ameans)
        print(astd)
        print(ameans[0]-ameans)
        """
        _valid_gsnamemaps_ = tuple(self._valid_gsnamemaps_.keys())+('link',)
        # ---------------------------------------
        if gslevel not in _valid_gsnamemaps_:
            raise KeyError(f'{gslevel: {gslevel} is invalid.}')
        # ---------------------------------------
        if gslevel == 'base':
            data = np.array([g.area for g in self.pxtals[instance].geoms])
            statistics = self.get_stat(data, nan_policy='omit')
        # ---------------------------------------
        if saa:
            self.mprop[instance]['area']['val'] = data
            self.mprop[instance]['area']['stat'] = statistics
        # ---------------------------------------
        if throw:
            return {'val': data, 'stat': statistics}

    def find_mprop_all(self,
                       mprop_name='area',
                       mprop_kwargs={},
                       gslevel='base',
                       saa=True,
                       throw=False,
                       collect_stats=True,
                       collect_12i_dev=True
                       ):
        """
        Find morphological property values across all available instances.

        Each time the area will be re-calculated.

        Parameters
        ----------
        mprop_name: str, optional
            Specify the name of the morphological property.
            Default value is 'area'.

        mprop_kwargs: dict, optional
            Specify the kwargs needed to calculate specific morphological
            property values. For details refere to the function definition
            and its documentation of the concerning morphological parameter.
            Default value is {}.

        gslevel: str, optional
            Specify name of the morphological feature.
            Default value is 'base'.

        saa: bool, optional
            Specify decision flag to save as attribute(s).
            Default value is True.

        throw: bool, optional
            Specify decision flag to return results.
            Default value is False.

        collect_stats: bool, optional
            Specify decision flag to collect statistics. Will be returned by
            default whether throw is True or False.
            Default value is True.

        collect_12i_dev: bool, optional
            Specify decision flag to collect deviation of statistical values
            from 1st to ith instance. Will only work when the collect_stats is
            True. Negative value indicates that the ith instance's
            morphological property statistic value is greater than the 1st
            instance's morphological property statistic value.
            Default value is True.

        Return
        ------
        mprop_data: dict
            Morphological property data reqiested for.

        dev_12i: dict
            Deviation between 1st to ith instance.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 125],
                 ybound = [0, 100], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=4, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        mprop_data, dev_12i = gset.find_mprop_all(mprop_name='area',
                           mprop_kwargs={}, gslevel='base', saa=True,
                           throw=False, collect_stats=True,
                           collect_12i_dev=True)

        mprop_data.keys()
        >> dict_keys(['val', 'minimum', 'maximum', 'mean', 'variance',
                      'skewness', 'kurtosis', 'std', 'median', 'p25', 'p75'])

        dev_12i.keys()
        >> dict_keys(['minimum', 'maximum', 'mean', 'variance',
                      'skewness', 'kurtosis', 'std', 'median', 'p25', 'p75'])
        """
        mprop_vals = []
        # -----------------------------------
        for inst in range(self.ninst):
            mprop = self.find_areas(instance=inst+1, gslevel=gslevel,
                                    saa=saa, throw=True)
            mprop_vals.append(mprop['val'])
        # -----------------------------------
        if collect_stats:
            minimum, maximum, mean = [], [], []
            variance, skewness, kurtosis, std = [], [], [], []
            median, p25, p75 = [], [], []

            for inst in range(self.ninst):
                if saa:
                    _data_ = self.mprop[inst+1][mprop_name]['stat']
                else:
                    _data_ = mprop_vals['stat']
                minimum.append(_data_['min'])
                maximum.append(_data_['max'])
                mean.append(_data_['mean'])
                variance.append(_data_['std'])
                skewness.append(_data_['skewness'])
                kurtosis.append(_data_['kurtosis'])
                std.append(_data_['std'])
                median.append(_data_['median'])
                p25.append(_data_['p25'])
                p75.append(_data_['p75'])

            mprop_data = {'val': mprop_vals,
                          'minimum': np.array(minimum),
                          'maximum': np.array(maximum),
                          'mean': np.array(mean),
                          'variance': np.array(variance),
                          'skewness': np.array(skewness),
                          'kurtosis': np.array(kurtosis),
                          'std': np.array(std),
                          'median': np.array(median),
                          'p25': np.array(p25),
                          'p75': np.array(p75),
                          }
            if collect_12i_dev:
                dev_12i = {k: v[0]-v
                           for k, v in mprop_data.items() if k != 'val'}
        else:
            mprop_data = {'val': mprop_vals}
            dev_12i = {}

        return mprop_data, dev_12i

    def find_ar(self, instance=1, prop_type='bbox_rot', saa=True, throw=False):
        """
        Find aspect ratio of all grains in a given instance.

        Three types of aspect ratio calculations can be done, although,
        one is not truly a measure of aspect ratio. These are:
            * using bounding box. This is the fastest, but less accurate than
            using minimum rotated boundinmg box
            * using envelope. This computes the convex hull envelope and then
            calculates the aspect ratio. This is not prescibed as results are
            not validated yet.
            * using minimum rotated bounding box. This is the most prescribed.
            Although computational expensive, due to underlying C++
            implementations in Shapely, the cost is hardly noticeable. This is
            more accurate, as it considers the morphological orientation of the
            grain.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 125],
                 ybound = [0, 100], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)
        AR = gset.find_ar(instance=1, prop_type='bbox_rot', throw=True)
        AR = gset.find_ar(instance=1, prop_type='bbox', throw=True)
        AR = gset.find_ar(instance=1, prop_type='envelope', throw=True)
        gset.mprop[1]['ar']
        """
        if prop_type == 'bbox':
            print('Finding bounding box aspect ratio of grains.')
            grains = self.pxtals[instance].geoms
            ar = np.array([None for grain in grains])
            for i, grain in enumerate(grains):
                '''Get the grain bounds'''
                bounds = grain.bounds
                bbc = np.array([[bounds[0], bounds[1]],  # xmin, ymin
                                [bounds[2], bounds[1]],  # xmax, ymin
                                [bounds[2], bounds[3]],  # xmax, ymax
                                [bounds[0], bounds[3]],  # xmin, ymax
                                ])
                '''Get the edge lengths of the bounding box coordinates.'''
                L = np.sqrt(np.sum(np.square(bbc - np.roll(bbc, -1, axis=0)),
                                   axis=1))
                '''There will be four values for four edges of the bbox.'''
                perp_line_lengths = np.unique(np.round(L, decimals=8))
                '''Of these 4 values, two pairs will be there. Taking a unique
                will reveal the lengths of two perpendicular edges. This is
                then used to calculate the aspect ratio as the ratio of
                longer side to shorter side.'''
                ar[i] = perp_line_lengths.max()/perp_line_lengths.min()

        if prop_type == 'envelope':
            print('Finding envelope aspect ratio of grains.')
            grains = self.pxtals[instance].geoms
            ar = np.array([None for grain in grains])
            for i, grain in enumerate(grains):
                '''Find bounding box coordinates. Coordinates will form a closed
                loop. Starts at top left and ends and bottom left. Ordered
                clockwise.
                '''
                bbc = np.vstack(grain.boundary.envelope.boundary.xy).T[:-1]
                '''Get the edge lengths of the bounding box coordinates.'''
                L = np.sqrt(np.sum(np.square(bbc - np.roll(bbc, -1, axis=0)),
                                   axis=1))
                '''There will be four values for four edges of the bbox.'''
                perp_line_lengths = np.unique(np.round(L, decimals=8))
                '''Of these 4 values, two pairs will be there. Taking a unique
                will reveal the lengths of two perpendicular edges. This is
                then used to calculate the aspect ratio as the ratio of
                longer side to shorter side.'''
                ar[i] = perp_line_lengths.max()/perp_line_lengths.min()

        if prop_type == 'bbox_rot':
            print('Finding ascpect ratio from rotated bounding box')
            grains = self.pxtals[instance].geoms
            ar = np.array([None for grain in grains])
            for i, grain in enumerate(grains):
                bbc = np.array(list(grain.minimum_rotated_rectangle.exterior.coords))[:-1]
                '''Get the edge lengths of the bounding box coordinates.'''
                L = np.sqrt(np.sum(np.square(bbc - np.roll(bbc, -1, axis=0)),
                                   axis=1))
                '''There will be four values for four edges of the bbox.'''
                perp_line_lengths = np.unique(np.round(L, decimals=8))
                '''Of these 4 values, two pairs will be there. Taking a unique
                will reveal the lengths of two perpendicular edges. This is
                then used to calculate the aspect ratio as the ratio of
                longer side to shorter side.'''
                ar[i] = perp_line_lengths.max()/perp_line_lengths.min()

        if saa:
            self.mprop[instance]['ar']['prop_type'] = prop_type
            self.mprop[instance]['ar']['val'] = ar
        if throw:
            return ar

    def link_geom(self, gslevel='base', instance=1,
                  make_upxo_mp=True,
                  make_upxo_mp_subfeatures=False,
                  saa=True, throw=False):
        """
        Return
        ------
        link: dict
            Following are the keys:

                * vc: Vertex coordinate values: Numpy array.

                * vc_cid: cid wise arranged vc values. Numpy array of numpy
                arrays. There will be as many inner numpy arrays as the number
                of grains. Each inner numpy array corresponds to the coordinate
                values of the vertex points belonging to that particular grain.

                * vc_cid_ind: Index locations of vertex coordinates of each
                vertex in vc_cid, mapped from vc. List of lists. An nnner list
                consists of index location of the coordinate in vc numpy array.

                * mp: Muti-point object pf all p2ds. Note: This is not a newly
                constructed multi-point 2D. Rather, it is extracted from the
                poly-xtal. So, the mp and p2d are both extracted from the
                poly-xtal. Important note:

                * p2d: upxo point objects of all points in mp.

                * p2d_cid: same as vc_cid, except that this consists of
                UPXO point objects taken from p2d, using vc_cid_ind.

            Important note: A UPXO point objects bearing the same coordinate
            values and occuring in p2d, p2d_cid and mp (i.e. in mp.points), all
            have the same Memory IDs. That is, the same UPXO point 2d object is
            used in different places. This provides a major advantage. When a
            point object is moved, the pxtal construct also changes
            appropriately !.

        Examples
        --------
        link = gset.link_geom(instance=1, saa=True, throw=True)

        Understanding
        -------------
        link['vc_cid_ind'][0]
            [25, 18, 14, 15, 22, 25]
        link['vc_cid_ind'][1]
            [18, 28, 27, 13, 14, 18]

        link['vc_cid'][0]
            array([[  4.79838951,  96.20757309],
                   [  2.10309145,  92.63671454],
                   [  0.        ,  92.70126343],
                   [  0.        , 100.        ],
                   [  3.17313095, 100.        ],
                   [  4.79838951,  96.20757309]])
        link['vc_cid'][1]
            array([[ 2.10309145, 92.63671454],
                   [ 5.28912901, 87.18863161],
                   [ 5.03114206, 86.71299143],
                   [ 0.        , 86.30397593],
                   [ 0.        , 92.70126343],
                   [ 2.10309145, 92.63671454]])

        link['p2d'][18]
            uxpo-p2d (2.1030914538410945,92.63671453775025)
        link['p2d_cid'][0][1]
            uxpo-p2d (2.1030914538410945,92.63671453775025)
        link['mp2d'].points[18]
            uxpo-p2d (2.1030914538410945,92.63671453775025)

        id(link['p2d'][18])
            2113227713792
        id(link['p2d_cid'][0][1])
            2113227713792
        id(link['mp2d'].points[18])
            2113227713792

        link['p2d'][18]
            uxpo-p2d (2.1030914538410945,92.63671453775025)

        We will arbitrarily change the coordinates of this point and see it
        being reflected in all oher places. We will change below:
        link['p2d'][18].x = 0.0
        link['p2d'][18].y = 90.0

        Now, we see that this has reflected in all places.
        link['p2d'][18]
            uxpo-p2d (0.0,90.0)
        link['p2d_cid'][0][1]
            uxpo-p2d (0.0,90.0)
        link['mp2d'].points[18]
            uxpo-p2d (0.0,90.0)
        """
        make_ump, make_umpsf = make_upxo_mp, make_upxo_mp_subfeatures
        # ---------------------------------------------
        # pxtal = PXT.pxtals[instance]
        '''Get point coordinate details'''
        _get_coords_ = self.extract_shapely_coords_all_grains
        points = _get_coords_(gslevel=gslevel, instance=instance,
                              mfname='c_vp',
                              make_upxo_mp=make_ump,
                              make_upxo_mp_subfeatures=make_umpsf)
        '''Get the vertex coordinates'''
        vc = points['all']
        vc_cid = points['cid']

        vc_cid_ind = [[None for gv in gvc] for gvc in vc_cid]
        # gvc = vc_cid[0]
        for gvci, gvc in enumerate(vc_cid, start=0):
            for gvi, gv in enumerate(gvc, start=0):
                vc_cid_ind[gvci][gvi] = np.where((vc == gv).all(axis=1))[0][0]

        '''Vertex point objects'''
        vp_cid = [[points['mp_all'].points[gvi] for gvi in gvci]
                  for gvci in vc_cid_ind]

        link = {'vc': vc,
                'instance': instance,
                'vc_cid': vc_cid,
                'vc_cid_ind': vc_cid_ind,
                'p2d': points['mp_all'].points,
                'p2d_cid': vp_cid,
                'mp': points['mp_all']}

        if saa:
            self.geolink = link

        if throw:
            return link

    def make_upxo_gs(self, instance=1):
        link = self.link_geom(instance=1, saa=True, throw=True)

    def reconstruct_pxtal_from_geom_link(self, link):
        if 'p2d' not in link.keys():
            raise KeyError("Key 'p2d' is not a key in link dictionary")
        if 'p2d_cid' not in link.keys():
            raise KeyError("Key 'p2d_cid' is not a key in link dictionary")
        # -------------------------------------------

    def setup_fdb(self):
        self.fdb = {'base': {'sh': None,
                             'vc': None,
                             'vc_cid_ind': None,
                             'p2d': None,
                             'p2d_cid': None,
                             },
                    '__superset__': {1: None},
                    'pix': {'name': None},
                    'pert': {'name': None},
                    'subgrains': {1: None},
                    'paps': {1: None},
                    }

    def add_fdb(self, instance=1, gslevels=['base'], fdbs=[None]):
        """
        Add a featutre database instance.

        gslevels: list
            List of feature names.
        feat: list
            List of feature data bases.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 125],
                 ybound = [0, 100], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)
        gset.add_fdb(gslevels=['base'],
                    fdbs=[gset.link_geom(instance=1, saa=True, throw=True)])

        gset.fdb_base.keys()
        >> dict_keys([1])

        gset.fdb_base[1].keys()
        >> dict_keys(['vc', 'instance', 'vc_cid', 'vc_cid_ind',
                      'p2d', 'p2d_cid', 'mp'])
        """
        # ---------------------------------------
        if not isinstance(gslevels, list):
            raise ValueError('gslevels must be a list')
        if not isinstance(fdbs, list):
            raise ValueError('feat must be a list')
        # ---------------------------------------
        for gslevel in gslevels:
            if gslevel not in self._valid_gsnamemaps_:
                raise ValueError('invalid gslevel')
        # ---------------------------------------
        for fname_count, gslevel in enumerate(gslevels, start=0):
            if gslevel == 'base':
                fdb = fdbs[fname_count]
                self.fdb_base[fdbs[fname_count]['instance']] = fdb
            else:
                raise KeyError(f'{gslevel: {gslevel} is invalid.}')

    def plot(self, instance=1):
        """
        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True}}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 125],
                ybound = [0, 100], sp_distr='random', gr_tech='pds',
                smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                niter=10, ntrials=100, n_instances=2, repr_prop=repr_prop,
                k_char_length_inc=0.05, k_char_length_dec=0.05,)
        gset.plot()
        """
        if not self.fdb_base:
            self.add_fdb(gslevels=['link'],
                         fdbs=[self.link_geom(instance=instance,
                                              saa=True, throw=True)])
        # ---------------------------------------
        X, Y = self.fdb_base[instance]['vc'].T
        coords_ids = self.fdb_base[instance]['vc_cid_ind']
        # ---------------------------------------
        plt.figure(figsize=(5, 5), dpi=75)
        for vc_ids in coords_ids:
            plt.fill(X[vc_ids], Y[vc_ids])
            plt.plot(X[vc_ids], Y[vc_ids], '-k', linewidth=2)
            plt.gca().set_aspect('equal')

    def plot_cells(self, instance=1, cids=[1]):
        """
        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 125],
                 ybound = [0, 100], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=1, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        non = gset.find_On_neigh(instance=1, gslevel='base', On=3)
        gset.plot_cells(instance=1, cids=non[100])

        # ====================================================================
        train to be the best among the group
        train to win
        train to dominate
        # ====================================================================
        """
        if not self.fdb_base:
            self.add_fdb(gslevels=['link'],
                         fdbs=[self.link_geom(instance=instance,
                                              saa=True, throw=True)])
        # ---------------------------------------
        X, Y = self.fdb_base[instance]['vc'].T
        coords_ids = self.fdb_base[instance]['vc_cid_ind']
        # ---------------------------------------
        plt.figure(figsize=(5, 5), dpi=75)
        for cid in cids:
            cid -= 1
            plt.fill(X[coords_ids[cid]], Y[coords_ids[cid]])
            plt.plot(X[coords_ids[cid]], Y[coords_ids[cid]], '-k', linewidth=2)
            plt.gca().set_aspect('equal')

    def get_nboundaries_cell(self, cell):
        """
        Get the number of interiors of a cell.

        Parameters
        ----------
        cell: shapely.polygon
            Specify a sapely polygon.


from shapely.geometry import box, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# Step 1: Define the original rectangle (cell)
cell = box(0, 0, 10, 5)

# Step 2: Buffer inward to get the inner cell
inner = cell.buffer(-0.5)

# Step 3: Create the annular region
annulus = cell.difference(inner)

# Step 4: Choose hole centers within the annulus — near the mid-distance from the boundary
hole_centers = [Point(1, 1), Point(9, 1), Point(5, 0.3), Point(5, 4.7)]
hole_radius = 0.1
holes = [p.buffer(hole_radius) for p in hole_centers]

# Step 5: Ensure the holes are within the annulus
holes_within_annulus = [h for h in holes if annulus.contains(h)]

# Step 6: Subtract holes from the annular region
final_shape = annulus.difference(unary_union(holes_within_annulus))

# Plotting
def plot_shape(shape, color='lightblue'):
    if shape.geom_type == 'Polygon':
        x, y = shape.exterior.xy
        plt.fill(x, y, color=color)
        for interior in shape.interiors:
            x, y = interior.xy
            plt.fill(x, y, color='white')
    elif shape.geom_type == 'MultiPolygon':
        for geom in shape.geoms:
            plot_shape(geom, color=color)

plt.figure(figsize=(8, 6))
plot_shape(final_shape)
plt.gca().set_aspect('equal')
plt.title("Annular Region with Valid Holes")
plt.show()



fs = final_shape

fs.interiors[0]
fs.interiors[1]
fs.interiors[2]

        """
        return 1+len(cell.interiors)

    def get_cell_vertices_ext_from_cell(self, cell):
        vert = np.array([crd for crd in cell.exterior.coords])[:-1]
        return vert

    def get_cell_vertices_ext_from_lr(self, lr):
        '''lr: linear ring.'''
        vert = np.array([crd for crd in lr.coords])[:-1]
        return vert

    def get_cell_vertices(self, cell):
        """
        Get cell vertices.
        """
        nbound = self.get_nboundaries_cell(cell)
        ext_vert = np.array([crd for crd in cell.exterior.coords])[:-1]
        vert = {'exterior': ext_vert,
                'interior': None}
        if nbound > 1:
            int_vert = [self.get_cell_vertices_ext_from_lr(lr)
                        for lr in cell.interiors]
            vert['interior'] = int_vert
        return vert

    def get_cell_edge_lengths(self, cell):
        """
        Get edge lengths of the shapely polygon.
        """
        vrt = self.get_cell_vertices(cell)
        lengths = np.linalg.norm(vrt - np.roll(vrt, -1, axis=0), axis=1)

    def sf_gbz_cell(self, cell, k_shr, method='min_dist_centre'):
        """
        Sub-feature introduction: grain boundary zone: for a single cell

        method: str, optional
            Followig options are permitted.
                * 'min_edge_length'
                * 'min_dist_centre'
                * 'smallest_skl_length'
        """
        xs = list(cell.exterior.coords.xy[0])
        ys = list(cell.exterior.coords.xy[1])
        # -------------------------------------------------------
        if method == 'min_edge_length':
            pass
        # -------------------------------------------------------
        if method == 'min_dist_centre':
            min_corner = shGeometry.Point(min(xs), min(ys))
            # max_corner = shGeometry.Point(max(xs), max(ys))
            center = shGeometry.Point(0.5*(min(xs)+max(xs)),
                                      0.5*(min(ys)+max(ys)))
            shdist = center.distance(min_corner)*k_shr
        # -------------------------------------------------------
        if method == 'smallest_skl_length':
            pass
        # -------------------------------------------------------
        cell_core_zone = cell.buffer(-shdist)
        cell_boundary_zone = cell-cell_core_zone
        # -------------------------------------------------------
        return cell_boundary_zone, cell_core_zone

    def sf_gbz(self, instance=1,  gslevel='base', cids=[], k_shr=0.1,
               assemble_cells=True, assemble_sf=True,
               saa=True, throw=False, viz=False):
        """
        Sub-feature introduction: grain boundary zone: multiple cells.

        Parameters
        ----------
        instance: int, optional
        gslevel: str, optional
        k_shr: float, optional
        asemble_cells: bool, optional
        assemble_sf: bool, optional
        saa: bool, optional
        throw: bool, optional

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 75],
                 ybound = [0, 100], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=1, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        zones = gset.sf_gbz(instance=1, gslevel='base', k_shr=0.15,
                            assemble_cells=True, saa=True, throw=False,
                            viz=True)

        # Find the sub-features, parent cell wise
        # -----------------------------
        pcid = 0
        # -----------------------------
        gset.pxtals[1].geoms[pcid]  # 1
        '''This is the 1st parent grain, i.e. 1st base cell.'''
        # -----------------------------
        zones['cbz'][pcid]  # 2
        zones['ccz'][pcid]  # 2
        '''This is the boundary and core zones of the 1st base cell.'''
        # -----------------------------
        zones['sf_pcwise'][pcid]  # 3
        '''This is multi-cell: zones['cbz'][pcid] + zones['ccz'][pcid]'''
        # -----------------------------
        '''below shows how to access individual geometry of multi-cell'''
        zones['sf_pcwise'][pcid][0]  # 4  <-- subfeature 1
        zones['sf_pcwise'][pcid][1]  # 5  <-- subfeature 2
        '''Alternatively, you can also access the same as belw9'''
        zones['sf_pcwise'][pcid].geoms[0]  # 6  <-- subfeature 1
        zones['sf_pcwise'][pcid].geoms[1]  # 7  <-- subfeature 2
        # Note: 4 and 5 is same as 6 and 7 resepectively.
        # -----------------------------
        plt.figure(figsize=(5, 5), dpi=150)
        coords_cid = [np.vstack(c.boundary.xy).T for c in gset.pxtals[1].geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
        plt.gca().set_aspect('equal')
        # -----------------------------

        coords_cid_out = []
        coords_cid_in = []
        sh_mls = shapely.geometry.multilinestring.MultiLineString
        sh_ls = shapely.geometry.linestring.LineString
        for c in zones['pxtal'].geoms:
            cbound = c.boundary
            if isinstance(cbound, sh_mls):
                l = [cbnd.length for cbnd in cbound.geoms]
                lsort = [i for i in np.argsort(l)[::-1]]
                ls = [cbound.geoms[i] for i in lsort]  # Line strings
                lscoords = [np.vstack(_ls_.xy).T for _ls_ in ls]
                coords_cid_out.append(lscoords[0])
                coords_cid_in.append(lscoords[1])
            if isinstance(cbound, sh_ls):
                pass

        plt.figure(figsize=(5, 5), dpi=150)
        for vc in coords_cid_out:
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
        for vc in coords_cid_in:
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=0.75)
        plt.gca().set_aspect('equal')
        ------------------------------------------------------------



        import math
        hexes = generate_hex_grid_within_bounds(
                                            hex_area=10,
                                            x_min=0, y_min=0,
                                            x_max=100, y_max=100
                                        )
        rounded_hexes = [round_coords(h, ndigits=6) for h in hexes]
        combined = unary_union(rounded_hexes)



        MultiPolygon(hexes)

        from shapely.ops import unary_union, snap
        reference = hexes[0]
        snapped = [snap(h, reference, tolerance=1e-6) for h in hexes]
        combined = unary_union(snapped)


        plt.figure(figsize=(5, 5), dpi=150)
        coords_cid = [np.vstack(c.boundary.xy).T for c in unary_union(hexes).geoms]
        for vc in coords_cid:
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
        plt.gca().set_aspect('equal')
        """
        pxt = self.pxtals[instance]
        # ----------------------------------------------------
        zones = {}
        # ----------------------------------------------------
        if len(cids) == 0:
            features = pxt.geoms
        # ----------------------------------------------------
        cbz, ccz = [], []
        for cell in features:
            _cbz_, _ccz_ = self.sf_gbz_cell(cell, k_shr)
            cbz.append(_cbz_)
            ccz.append(_ccz_)
        zones['cbz'] = cbz
        zones['ccz'] = ccz
        # ----------------------------------------------------
        bzcz = []
        if assemble_sf:
            for _bzcz_ in zip(cbz, ccz):
                bzcz.append(self.assemble_cells(_bzcz_))
            zones['sf_pcwise'] = bzcz  # Sub-features parent cell wise
        # ----------------------------------------------------
        if assemble_cells:
            mcell = MultiPolygon(cbz + ccz)
            zones['pxtal'] = mcell
        # ----------------------------------------------------
        if viz:
            coords_cid_out = []
            coords_cid_in = []
            sh_mls = shapely.geometry.multilinestring.MultiLineString
            sh_ls = shapely.geometry.linestring.LineString
            for c in zones['pxtal'].geoms:
                cbound = c.boundary
                if isinstance(cbound, sh_mls):
                    l = [cbnd.length for cbnd in cbound.geoms]
                    lsort = [i for i in np.argsort(l)[::-1]]
                    ls = [cbound.geoms[i] for i in lsort]  # Line strings
                    lscoords = [np.vstack(_ls_.xy).T for _ls_ in ls]
                    coords_cid_out.append(lscoords[0])
                    coords_cid_in.append(lscoords[1])
                if isinstance(cbound, sh_ls):
                    pass

            plt.figure(figsize=(5, 5), dpi=150)
            for vc in coords_cid_out:
                plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
            for vc in coords_cid_in:
                plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=0.75)
            plt.gca().set_aspect('equal')


        return zones

    def gen_pol6_grid(self, amin, bounds):
        """
        Generate hexagonal tessellation within the bounds.

        Examples
        --------
        from upxo.pxtal.vortess2d import gtess2d
        repr_prop={'area': {'mean': {'val': 50, 'dev': 7.5,},
                            'consider_boundary_grains': True},}
        gset = gtess2d.from_seed_points(sp_input='gen', xbound=[0, 50],
                 ybound = [0, 50], sp_distr='random', gr_tech='pds',
                 smp_tech='bridson1', lean='veryhigh', char_length=[4.5],
                 niter=10, ntrials=100, n_instances=1, repr_prop=repr_prop,
                 k_char_length_inc=0.05, k_char_length_dec=0.05,)

        zones = gset.sf_gbz(instance=1, gslevel='base', k_shr=0.1,
                            assemble_cells=True, saa=True, throw=False, viz=True)

        # zones['ccz'][0].boundary
        # nedges = len(zones['ccz'][0].boundary.coords)-2

        zones['ccz_hex'] = []
        zones['cbz_new'] = []
        zones['ccz_offcut'] = []

        amin_spec = 'by_cell'  # options: 'uniform', 'by_cell'
        amin_fac = 0.03

        if amin_spec == 'uniform':
            amin = amin_fac*np.array([cell.area
                                      for cell in gset.pxtals[1].geoms]).min()

        for ci, cell in enumerate(zones['ccz']):
            if amin_spec == 'by_cell':
                amin = gset.pxtals[1].geoms[ci].area * amin_fac
            bounds = cell.bounds
            hexes = gset.gen_pol6_grid(amin, bounds)
            hexcells = []
            for i, flag in enumerate([hex.within(cell) for hex in hexes]):
                if flag:
                    hexcells.append(hexes[i])

            ccz_offcut = cell - unary_union(hexcells)
            zones['ccz_offcut'].append(ccz_offcut)

            # ccz_offcut = gset._round_coords_(ccz_offcut, ndigits=6)

            zones['cbz_new'].append(unary_union([ccz_offcut, zones['cbz'][ci]]))

            # MultiPolygon([ccz_offcut, zones['cbz'][ci]])

            # zones['cbz'][2]
            gset.pxtals[1].geoms[2] - unary_union(hexcells)
            zones['ccz_hex'].append(unary_union(hexcells))

        coords_cid_out = []
        coords_cid_in = []
        sh_mls = shapely.geometry.multilinestring.MultiLineString
        sh_ls = shapely.geometry.linestring.LineString
        for c in zones['cbz_new']:
            cbound = c.boundary
            if isinstance(cbound, sh_mls):
                ls = [_ls_ for _ls_ in cbound.geoms]  # Line strings
                lscoords = [np.vstack(_ls_.xy).T for _ls_ in ls]
                coords_cid_out.append(lscoords[0])
                coords_cid_in.append(lscoords[1])
            if isinstance(cbound, sh_ls):
                pass

        plt.figure(figsize=(5, 5), dpi=150)
        for vc in coords_cid_out:
            plt.plot(vc[:, 0], vc[:, 1], '-k', linewidth=1.25)
        for vc in coords_cid_in:
            plt.plot(vc[:, 0], vc[:, 1], '-b', linewidth=0.75)
        plt.gca().set_aspect('equal')
        """
        xmin, ymin, xmax, ymax = bounds
        a = math.sqrt((2*amin)/(3*math.sqrt(3)))
        dx = 3*a/2  # x-spacing
        dy = math.sqrt(3)*a  # y-spacing
        cols = int((xmax-xmin)/dx)+2
        rows = int((ymax-ymin)/dy)+2
        hexagons = []
        for col in range(cols):
            for row in range(rows):
                x = xmin+col*dx
                y = ymin+row*dy+(dy/2 if col % 2 else 0)
                if x-a < xmin or x+a > xmax or y-a < ymin or y+a > ymax:
                    continue
                hex_pts = [(x+a*math.cos(math.radians(angle)),
                            y+a*math.sin(math.radians(angle)))
                           for angle in range(0, 360, 60)]
                hexagons.append(Polygon(hex_pts))
        hexagons = [self._round_coords_(h, ndigits=6) for h in hexagons]
        return hexagons

    def _round_coords_(self, poly, ndigits=6):
        return Polygon([(round(x, ndigits), round(y, ndigits))
                        for x, y in poly.exterior.coords])

    def find_scells_in_tcell(self, scells, tcell):
        hexes[0].bounds
        xmin, ymin, xmax, ymax = grain.bounds
        bbc = np.array([[bounds[0], bounds[1]],  # xmin, ymin
                        [bounds[2], bounds[1]],  # xmax, ymin
                        [bounds[2], bounds[3]],  # xmax, ymax
                        [bounds[0], bounds[3]],  # xmin, ymax
                        ])
        pass

    def assemble_cells(self, features):
        return MultiPolygon(features)

    def perturb_feat(self, instance=1, gslevel='base', cid=[], ):
        pass

    def fe_mesh(self, instance=1, gslevel='base', tool='gmsh'):
        pass

    def get_idmap_cf(self, gslevel, instance_ID, fname, fid, cids=[]):
        '''
        Use this function to get cellIDs-->feature_ID mapping.

        Parameters
        ----------
        gslevel: str
            Grain structure hierarchy level name. Must be a valid name. Please
            refer to self._valid_gslevels_ for all permitted gslevel values.
        instance_ID: str
            Instance number of the pxtal.
        fname: str
            Feature name. Could be 'ph' for phase, etc.
        fid: int
            Feature ID number. If fname is 'ph', then this would indicate
            which phase is needed.
        cids: list, optional
            Cell IDs.

        Return
        ------
        mapping: list
            A list of cell IDs belonging to the instance_ID, fname and fid.

        Example
        -------
        gset.get_idmap_cf('base', 1, 'ph', 1, cids=[12])

        The above code accessess all the cells belonging to phase of phase_ID,
        in the pxtal instance specified by instance_ID.

        NOTES
        -----
        The data can also be directly accessed as below. This, however, is not
        the recommended way to access the data, as its prone to human errors
        and complex code.

        Data structure
        --------------
        gslevel = 'base'
        instance_ID = 1
        fname = 'ph'
        fid = 1
        cell_ID = 15

        gset._idmap_cf_[gslevel].keys()
        gset._idmap_cf_[gslevel][instance_ID].keys()
        gset._idmap_cf_[gslevel][instance_ID][fname].keys()
        gset._idmap_cf_[gslevel][instance_ID][fname][fid][cell_ID]

        cell_IDs = [1, 5, 20]
        [gset._idmap_cf_[gslevel][instance_ID][fname][fid][c] for c in cell_IDs]

        Author
        ------
        Dr. Sunil Anandatheertha
        '''
        if len(cids) == 0:
            cids = slice(0, self.tprop[1]['ncells'])
            mapping = self._idmap_cf_[gslevel][instance_ID][fname][fid][cids]
        else:
            mapping = [self._idmap_cf_[gslevel][instance_ID][fname][fid][c]
                       for c in cids]
        return mapping

    def get_idmap_fc(self, gslevel, instance_ID, fname, cids=[]):
        '''
        Use this function to get feature_ID-->cellIDs mapping.

        Parameters
        ----------
        gslevel: str
            Grain structure hierarchy level name. Must be a valid name. Please
            refer to self._valid_gslevels_ for all permitted gslevel values.
        instance_ID: str
            Instance number of the pxtal.
        fname: str
            Feature name. Could be 'ph' for phase, etc.
        fid: int
            Feature ID number. If fname is 'ph', then this would indicate
            which phase is needed.
        cids: list, optional
            Cell IDs.

        Return
        ------
        mapping: list
            A list of cell IDs belonging to the instance_ID, fname and fid.

        Example
        -------
        gslevel = 'base'
        instance_ID = 1
        fname = 'ph'
        cell_IDs = [15, 25]

        gset.get_idmap_fc(gslevel, instance_ID, fname, cell_IDs)

        NOTES
        -----
        The data can also be directly accessed as below. This, however, is not
        the recommended way to access the data, as its prone to human errors
        and complex code.

        Data structure
        --------------
        gset._idmap_fc_[gslevel].keys()
        gset._idmap_fc_[gslevel][instance_ID].keys()
        [gset._idmap_fc_[gslevel][instance_ID][fname][c] for c in cell_IDs]
        '''
        if len(cids) == 0:
            cids = slice(0, self.tprop[1]['ncells'])
            mapping = self._idmap_fc_[gslevel][instance_ID][fname][cids]
        else:
            mapping = [self._idmap_fc_[gslevel][instance_ID][fname][c]
                       for c in cids]
        return mapping

    def validate_instance_ID(self, gslevel, instance_ID):
        if gslevel == 'base':
            if instance_ID in self.instn:
                flag = True
            else:
                flag = False
        else:
            flag = False

        return flag
