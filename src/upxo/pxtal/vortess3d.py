# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 12:27:59 2025

@author: Dr. Sunil Anandatheertha
"""
import numpy as np
import pyvoro
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
import tetgen
import matplotlib.pyplot as plt
from upxo.geoEntities.mulpoint2d_old import mulpoint2d
from scipy.spatial import cKDTree as TREE
from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d


class gtess3d():
    """
    ===========================================================================
    @ dev notes by (Dr. Sunil Anandatheertha, )
    ===========================================================================
    ---------------> Instantiation <--------------------
    from upxo.pxtal.vortess3d import gtess3d
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
    __slots__ = ('ninst', 'instn', 'sp', 'pxtals', 'fdb_base', 'info',
                 'gen_method',
                 'fid', 'floc', 'geolink', 'cids_base',
                 'idmap_c_supc', 'idmap_c_subc',
                 'fdb_pix', '_idmap_fc_', '_idmap_cf_',
                 'fdb_pert', 'fdb_subgrains', 'fdb_paps', 'fdb_supset',
                 'uinputs',
                 'neighs_cid', 'xomap', 'n',
                 'gbjp', 'bounds', 'mprop', 'tprop',
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

    def __init__(self, pxtals=None, gen_method='from_seed_points',
                 phid=None):
        # ------------------------------------------------------------------
        if gen_method == 'seed_point_extrusion':
            self.pxtals = pxtals['pxtals']
            self.sp = {'sps': pxtals['sps'],
                       'sepdist_mean': pxtals['sp_sepdist_mean'],
                       'coords': pxtals['sp_coords'],
                       'coords_all': pxtals['sp_coords_all'],
                       'coords_mp3d': pxtals['sp_coords_mp3d'],
                       }
            self.info = {'uinputs': pxtals['uinputs']}
            self.bounds = {'xbound': pxtals['xbound'],
                           'ybound': pxtals['ybound'],
                           'zbound': pxtals['zbound'],
                           'bbox': pxtals['bounding_box']}
            self.ninst = pxtals['n_instances']
            self.gen_method = pxtals['gen_method']
        # ------------------------------------------------------------------
        self.instn = range(1, self.ninst+1)
        '''Setup the base GS feature database'''
        self.fdb_base = {}
        self.fdb_subgrains = {}
        self.fdb_paps = {}
        self.geolink = {}
        # ------------------------------------------------------------------
        '''Morphological properties.'''
        _vol_ = [[] for inst in self.instn]
        for inst in self.instn:
            for cell in self.pxtals[inst]:
                _vol_[inst-1].append(cell['volume'])
        self.mprop = {'base': {'vol': _vol_,
                               'ar': [],
                               'gbsa': [],
                               }
                      }
        # ------------------------------------------------------------------
        '''Topological properties.'''
        self.tprop = {inst: {'ncells': len(self.pxtals[inst]),
                             'ncrange': range(len(self.pxtals[inst])),
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

        # ------------------------------------------------------------------
        # def get_idmap_c_supc()
        self.idmap_c_supc = {i: None for i in self.instn}
        # def get_idmap_c_subc
        self.idmap_c_subc = {i: None for i in self.instn}
        # ------------------------------------------------------------------

        Example access - 2 (Preferred)
        ------------------------------
        Please refer to documentation of defintion get_idmap_cf(). Code is,

        gset.get_idmap_cf('base', 1, 'ph', 1, cids=[])
        '''
        for i in self.instn:
            _tmp_ = {'ph': {phid: [cid for cid in self.tprop[i]['ncrange']]}}
            self._idmap_cf_['base'][i] = _tmp_


        '''Now, we use add_fdb to actually store essential feature information.
        This includes the vertex coordinstes, instance number, CID mapping list
        of list of vertex coordinates, CID mapping list of list of vertex
        coordinate IDs, list of 2D point objects, CID mapping list of
        UPXo point 2D Object list IDs and UPXO multi-point object (Optional).
        '''
        '''for inst in self.instn:
            self.add_fdb(gslevels=['base'],
                         fdbs=[self.link_geom(instance=inst,
                                              make_upxo_mp=True,
                                              make_upxo_mp_subfeatures=False,
                                              saa=True, throw=True)])'''

    def __repr__(self):
        return f"gtess2d instances. n: {self.ninst}. Nsp={[]}"

    def plot_sp(self, inst):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.sp['coords_all'][inst-1][:, 0],
                   self.sp['coords_all'][inst-1][:, 1],
                   self.sp['coords_all'][inst-1][:, 2])

    @classmethod
    def from_seed_point_random(cls,
                               xbound=[0, 20], ybound=[0, 20], zbound=[0, 20],
                               npnt=100, delnpnt=5, niter=100,
                               repr_prop={'vol': {'mean': {'val': 10,
                                                           'dev': 5, },
                                                  'consider_boundary_grains': True}
                                         },
                               divergence_control=True,
                               ):
        """
        bounding_box = [[0, 12], [0, 12], [0, 12]]
        points = generate_random_lattice_bounded(bounding_box, 100)
        """
        nprand = np.random.random
        x_vals = nprand((npnt, 1))*(xbound[1][1]-xbound[1][0])
        y_vals = nprand((npnt, 1))*(ybound[1][1]-ybound[1][0])
        z_vals = nprand((npnt, 1))*(zbound[1][1]-zbound[1][0])
        coords = np.hstack((x_vals, y_vals, z_vals))

    @classmethod
    def from_regular_lattice(cls, bounding_box, xincr, yincr, zincr,
                             lattice='sc',lc=None, start_offset=True):
        if lattice == 'sc':
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounding_box

            # Define start and end points based on start_offset flag
            x_start, x_end = (xmin + xincr, xmax - xincr) if start_offset else (xmin, xmax)
            y_start, y_end = (ymin + yincr, ymax - yincr) if start_offset else (ymin, ymax)
            z_start, z_end = (zmin + zincr, zmax - zincr) if start_offset else (zmin, zmax)

            # Generate the grid points along each axis
            x_vals = np.arange(x_start, x_end + xincr, xincr)  # +xincr to include xmax-xincr
            y_vals = np.arange(y_start, y_end + yincr, yincr)
            z_vals = np.arange(z_start, z_end + zincr, zincr)

            # Generate the lattice points
            coords = np.array([[x, y, z] for x in x_vals for y in y_vals for z in z_vals])
        elif lattice == 'fcc':
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounding_box

            # Define start and end points based on start_offset flag
            x_start, x_end = (xmin + xincr, xmax - xincr) if start_offset else (xmin, xmax)
            y_start, y_end = (ymin + yincr, ymax - yincr) if start_offset else (ymin, ymax)
            z_start, z_end = (zmin + zincr, zmax - zincr) if start_offset else (zmin, zmax)

            # Generate the simple cubic base lattice
            x_vals = np.arange(x_start, x_end + xincr, xincr)
            y_vals = np.arange(y_start, y_end + yincr, yincr)
            z_vals = np.arange(z_start, z_end + zincr, zincr)

            # Generate the FCC basis points relative to each cubic lattice point
            fcc_basis = np.array([
                [0, 0, 0],                   # Corner atom
                [0.5 * xincr, 0.5 * yincr, 0],  # Face-centered atom in XY plane
                [0.5 * xincr, 0, 0.5 * zincr],  # Face-centered atom in XZ plane
                [0, 0.5 * yincr, 0.5 * zincr]   # Face-centered atom in YZ plane
            ])

            # Generate FCC lattice points by adding basis to each cubic point
            fcc_lattice = np.array([
                [x, y, z] + basis for x in x_vals for y in y_vals for z in z_vals for basis in fcc_basis
            ])
        elif lattice == 'bcc':
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounding_box

            # Define start and end points based on start_offset flag
            x_start, x_end = (xmin + xincr, xmax - xincr) if start_offset else (xmin, xmax)
            y_start, y_end = (ymin + yincr, ymax - yincr) if start_offset else (ymin, ymax)
            z_start, z_end = (zmin + zincr, zmax - zincr) if start_offset else (zmin, zmax)

            # Generate the simple cubic base lattice
            x_vals = np.arange(x_start, x_end + xincr, xincr)
            y_vals = np.arange(y_start, y_end + yincr, yincr)
            z_vals = np.arange(z_start, z_end + zincr, zincr)

            # Generate the BCC basis points relative to each cubic lattice point
            bcc_basis = np.array([
                [0, 0, 0],                   # Corner atom
                [0.5 * xincr, 0.5 * yincr, 0.5 * zincr]  # Body-centered atom
            ])

            # Generate BCC lattice points by adding basis to each cubic point
            bcc_lattice = np.array([
                [x, y, z] + basis for x in x_vals for y in y_vals for z in z_vals for basis in bcc_basis
            ])
        elif lattice == 'hcp':
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounding_box

            # Adjust yincr to reflect the correct hexagonal packing (a * sqrt(3))
            yincr = yincr * np.sqrt(3)

            # Define start and end points based on start_offset flag
            x_start, x_end = (xmin + xincr, xmax - xincr) if start_offset else (xmin, xmax)
            y_start, y_end = (ymin + yincr, ymax - yincr) if start_offset else (ymin, ymax)
            z_start, z_end = (zmin + zincr, zmax - zincr) if start_offset else (zmin, zmax)

            # Generate base lattice grid
            x_vals = np.arange(x_start, x_end + xincr, xincr)
            y_vals = np.arange(y_start, y_end + yincr, yincr)
            z_vals = np.arange(z_start, z_end + zincr, zincr)

            # HCP basis atoms
            hcp_basis = np.array([
                [0, 0, 0],
                [0.5 * xincr, 0.5 * yincr, 0],
                [0.5 * xincr, (1/6) * yincr, 0.5 * zincr],
                [0, (2/3) * yincr, 0.5 * zincr]
            ])

            # Generate HCP lattice points
            hcp_lattice = np.array([
                [x, y, z] + basis for x in x_vals for y in y_vals for z in z_vals for basis in hcp_basis
            ])

            # **Remove duplicates** to prevent pyvoro issues
            hcp_lattice = np.unique(hcp_lattice, axis=0)

            # **Filter points inside bounding box**
            filtered_lattice = []
            for point in hcp_lattice:
                if (xmin <= point[0] <= xmax) and (ymin <= point[1] <= ymax) and (zmin <= point[2] <= zmax):
                    filtered_lattice.append(point)
        elif lattice == 'dc':
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounding_box

            # Define start and end points based on start_offset flag
            x_start, x_end = (xmin+lc, xmax-lc) if start_offset else (xmin, xmax)
            y_start, y_end = (ymin+lc, ymax-lc) if start_offset else (ymin, ymax)
            z_start, z_end = (zmin+lc, zmax-lc) if start_offset else (zmin, zmax)

            # Generate the simple cubic base lattice
            x_vals = np.arange(x_start, x_end+lc, lc)
            y_vals = np.arange(y_start, y_end+lc, lc)
            z_vals = np.arange(z_start, z_end+lc, lc)

            # Define the Diamond Cubic basis
            diamond_basis = np.array([
                [0, 0, 0],
                [0.5*lc, 0.5*lc, 0],
                [0.5*lc, 0, 0.5*lc],
                [0, 0.5*lc, 0.5*lc],
                [0.25*lc, 0.25*lc, 0.25*lc],
                [0.75*lc, 0.75*lc, 0.25*lc],
                [0.75*lc, 0.25*lc, 0.75*lc],
                [0.25*lc, 0.75*lc, 0.75*lc]
            ])

            # Generate Diamond Cubic lattice points by adding basis to each simple cubic point
            diamond_lattice = np.array([
                [x, y, z] + basis for x in x_vals for y in y_vals for z in z_vals for basis in diamond_basis
            ])

    @classmethod
    def from_seed_point_extrusion(cls, sp_input='gen', seed_coords=None,
                                  xbound=[0, 50], ybound=[0, 100],
                                  zbound=[0, 75], nsp=600,
                                  n_instances=1, nsp_dev_ninstances=10,
                                  sp_distr='random',
                                  gr_tech='pds', smp_tech='bridson1',
                                  randuni_calc='by_points', lean='veryhigh',
                                  char_length=[3, 2], niter=500, ntrials=10,
                                  k_char_length_inc=0.1, k_char_length_dec=0.1,
                                  repr_prop={'vol': {'use': 'mean',
                                                     'mean': {'val': 10,
                                                               'dev': 5, },
                                                     'distr': {'filename': None,
                                                               'remol': [True, 25, 75]},
                                                     'consider_boundary_grains': True}
                                            },
                                  make_point_objects=True, make_ckdtree=True,
                                  char_length_mean=0.24598,
                                  char_length_min=0.1111, char_length_max=0.9999,
                                  nt=10, space='linear', nchecks=10, sdf=0.75,
                                  divergence_control=True,
                                  periodic=[False, False, False]):
        """
        Generate multi-instance multi-parameter repr 3dvtess base pxtal.

        Uses seed point distribution derived from.
            * extrusion based pesudo-Bridson sampling algorithm.
            * extrusion based dart sampling algorithm.

        Currently supported list of morphological parameters useful for
        representativeness assessment:
            * mean grain volume

        Parameters
        ----------
        sp_input='gen',
        seed_coords=None,
        xbound=[0, 50],
        ybound=[0, 100],
        zbound=[0, 75],
        nsp=600,
        n_instances=1,
        nsp_dev_ninstances=10,
        sp_distr='random',
        gr_tech='pds'
        smp_tech='bridson1',
        randuni_calc='by_points'
        lean='veryhigh',
        char_length=[3, 2]
        niter=500
        ntrials=10,
        k_char_length_inc=0.1
        k_char_length_dec=0.1,
        repr_prop={'vol': {'use': 'mean',
                           'mean': {'val': 10,
                                     'dev': 5, },
                           'distr': {'filename': None,
                                     'remol': [True, 25, 75]},
                           'consider_boundary_grains': True}
                  }
            distr: distribution data
            filepath: full path of file containing distribution data
                      Only .csv or .dat file allowed.
            remol: List for remove outliers in distr.
                remol[0]: Flag to remove or retain outliers in distribution.
                remol[1]: Lower percentile value.
                remol[2]: Upper percentile value.
        make_point_objects=True,
        make_ckdtree=True,
        char_length_mean=0.24598
        char_length_min=0.1111,
        char_length_max=0.9999,
        nt=10,
        space='linear'
        nchecks=10
        sdf=0.75,
        divergence_control=True,
        periodic=[False, False, False]

        Example
        -------
        from upxo.pxtal.vortess3d import gtess3d
        gt3d = gtess3d.from_seed_point_extrusion(sp_input='gen', seed_coords=None,
                                      xbound=[0, 15], ybound=[0, 15], nsp=200,
                                      n_instances=2, nsp_dev_ninstances=10,
                                      sp_distr='random',
                                      gr_tech='pds', smp_tech='bridson1',
                                      randuni_calc='by_points', lean='veryhigh',
                                      char_length=[3, 0], niter=500, ntrials=10,
                                      k_char_length_inc=0.1, k_char_length_dec=0.2,
                                      repr_prop={'vol': {'use': 'mean',
                                                         'mean': {'val': 10,
                                                                   'dev': 5, },
                                                         'distr': {'filename': None,
                                                                   'remol': True},
                                                         'consider_boundary_grains': True}
                                                },
                                      make_point_objects=True, make_ckdtree=True,
                                      char_length_mean=0.24598,
                                          char_length_min=0.1111, char_length_max=0.9999,
                                      nt=10, space='linear', nchecks=10, sdf=0.75,
                                      divergence_control=True,
                                      periodic=[False, False, False])

        gt3d.visualize_multiple_voronoi_cells(1, cell_indices=None, colors=None,
                                              alpha=1, edge_color='black')
        """
        ntrials = 100 if ntrials == -1 else ntrials
        pxtals, sps = {}, {}
        pxtal_count = 1
        # #####################################################################
        '''SIMPLIFY BRANCHINGS AS THERE NOW ARE SEPERATE DEFINITIONS TO DO
        INDIVIDUAL ONES.'''
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
                        CL = [char_length[0]]
                        _def = cls._make_pxtal_single_instance
                        for trial in range(ntrials):
                            print(f"Generating pxtal.. Iteration {trial+1}")
                            pxt = _def(spinput=sp_input,
                                       xbound=xbound, ybound=ybound,
                                       nsp=nsp, sp_distr=sp_distr,
                                       gridding_technique=gr_tech,
                                       sampling_technique=smp_tech,
                                       randuni_calc=randuni_calc, niter=niter,
                                       lean=lean, char_length=_char_length,
                                       make_point_objects=make_point_objects,
                                       make_ckdtree=make_ckdtree, space=space,
                                       char_length_mean=char_length_mean,
                                       char_length_min=char_length_min,
                                       char_length_max=char_length_max, nt=nt,
                                       nchecks=nchecks, sdf=sdf,
                                       periodic=periodic)
                            ncells = len(pxt['pxtal'])
                            sp_sepdist_mean = [pxt['sepdist_mean']]
                            sp_coords = [pxt['sp_coords']]
                            sp_coords_all = [pxt['sp_coords_all']]
                            sp_coords_mp3d = [pxt['sp_coords_mp3d']]
                            zbound = [pxt['zbound']]
                            bounding_box = [pxt['bounding_box']]
                            # ------------------------------------------
                            print(f"No. of seed points: {ncells}")
                            print(50*'#', pxt.keys(), 50*'#')
                            # ------------------------------------------
                            if 'vol' in repr_prop.keys():
                                # ------------------------------------------
                                # Extract volumes of grains
                                vols = np.array([cell['volume'] for cell in pxt['pxtal']])
                                # ------------------------------------------
                                print(f'Mean volume: {vols.mean()}')
                                print(f'Min volume: {vols.min()}')
                                print(f'Max volume: {vols.max()}')
                                # ######################################
                                # MAKE SEPERATE DEFINITION
                                if 'mean' in repr_prop['vol'].keys():
                                    vmean = vols.mean()
                                    _v = repr_prop['vol']['mean']['val']
                                    _vdev = repr_prop['vol']['mean']['dev']
                                    vmin = _v*(1-_vdev/100)
                                    vmax = _v*(1+_vdev/100)
                                    if vmin <= vmean <= vmax:
                                        _char_length = char_length
                                        pxtals[pxtal_count] = pxt['pxtal']
                                        sps[pxtal_count] = pxt['sp_mp2d_list']
                                        break
                                    if vmean < vmin:
                                        if divergence_control:
                                            k = 1 + k_char_length_inc
                                        else:
                                            k = 1 + trial*k_char_length_inc
                                        # CL.append(char_length[0])
                                        _char_length = [k*CL[-1]]
                                        # _char_length_ = _char_length
                                        CL.append(_char_length)
                                    if vmean > vmax:
                                        if divergence_control:
                                            k = 1 - k_char_length_inc
                                        else:
                                            k = 1 - trial*k_char_length_inc

                                        _char_length = [k*CL[-1]]
                                        # _char_length_ = _char_length
                                        CL.append(_char_length[0])
                                # ######################################
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
            print(f"Target mean grain volume: {repr_prop['vol']['mean']['val']}")
            print(f"Sample mean grain volume: {np.round(vols.mean(), 6)}")
            print('----')
            print(f"Input Char. length: {char_length}")
            print(f"Final Char. length: {np.round(CL[-1], 4)}")
            print('----')
            if type(pxt['sp_mp2d_list']) == list:
                npoints = np.array([sp.npoints for sp in pxt['sp_mp2d_list']]).sum()
            else:
                npoints = pxt['sp_mp2d_list'].npoints
            print(f"No. of seed points: {npoints}")
            print(f"No. of grains: {len(pxt['pxtal'])}")
            print(40*'-')
            uinputs = {'sp_input': sp_input,
                       'xbound': xbound, 'ybound': ybound, 'nsp': nsp,
                       'sp_distr': sp_distr, 'gridding_technique': gr_tech,
                       'sampling_technique': smp_tech,
                       'randuni_calc': randuni_calc, 'lean': lean,
                       'char_length': CL[-1], 'niter': niter,
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
                                    # _char_length = char_length
                                    # Remaining codes foir making pxtal
                                    pass
                            # ----------------------------
                            if gr_tech in ('random', 'pds'):
                                if smp_tech in ('dart', 'bridson1'):
                                    # _char_length = [char_length[0]]
                                    _def = cls._make_pxtal_single_instance
                                    pxtal_count += 1
                                    pxt = _def(spinput=sp_input,
                                               xbound=xbound, ybound=ybound,
                                               nsp=nsp, sp_distr=sp_distr,
                                               gridding_technique=gr_tech,
                                               sampling_technique=smp_tech,
                                               randuni_calc=randuni_calc,
                                               niter=niter, lean=lean,
                                               char_length=[CL[-1],
                                                            char_length[1]],
                                               make_point_objects=make_point_objects,
                                               make_ckdtree=make_ckdtree,
                                               space=space,
                                               char_length_mean=char_length_mean,
                                               char_length_min=char_length_min,
                                               char_length_max=char_length_max,
                                               nt=nt, nchecks=10, sdf=sdf,
                                               periodic=periodic)
                                    # print(pxt)
                                    pxtals[pxtal_count] = pxt['pxtal']
                                    sps[pxtal_count] = pxt['sp_mp2d_list']
                                    sp_sepdist_mean.append(pxt['sepdist_mean'])
                                    sp_coords.append(pxt['sp_coords'])
                                    sp_coords_all.append(pxt['sp_coords_all'])
                                    sp_coords_mp3d.append(pxt['sp_coords_mp3d'])
                                    zbound.append(pxt['zbound'])
                                    bounding_box.append(pxt['bounding_box'])

        uinputs = {'sp_input': sp_input, 'seed_coords': seed_coords,
                   'xbound': xbound, 'ybound': ybound, 'nsp': nsp,
                   'n_instances': n_instances,
                   'nsp_dev_ninstances': nsp_dev_ninstances,
                   'sp_distr': sp_distr, 'gr_tech': gr_tech,
                   'smp_tech': smp_tech, 'randuni_calc': randuni_calc,
                   'lean': lean, 'char_length': char_length,
                   'niter': niter, 'ntrials': ntrials,
                   'k_char_length_inc': k_char_length_inc,
                   'k_char_length_dec': k_char_length_dec,
                   'repr_prop': repr_prop,
                   'make_point_objects': make_point_objects,
                   'make_ckdtree': make_ckdtree,
                   'char_length_mean': char_length_mean,
                   'char_length_min': char_length_min,
                   'char_length_max': char_length_max,
                   'nt': nt, 'space': space, 'nchecks': nchecks,
                   'sdf': sdf, 'divergence_control': divergence_control}

        pxtals = {'pxtals': pxtals, 'sps': sps,
                  'sp_sepdist_mean': sp_sepdist_mean,
                  'sp_coords': sp_coords, 'sp_coords_all': sp_coords_all,
                  'sp_coords_mp3d': sp_coords_mp3d,
                  'n_instances': n_instances, 'gen_method': 'from_seed_points',
                  'uinputs': uinputs, 'xbound': xbound, 'ybound': ybound,
                  'zbound': zbound, 'bounding_box': bounding_box
                  }

        return cls(pxtals=pxtals, gen_method='seed_point_extrusion')

    @classmethod
    def _make_pxtal_single_instance(cls, spinput='gen',
                                    xbound=[0, 100], ybound=[0, 100],
                                    zbound=[0, 100], nsp=600,
                                    sp_distr='random',
                                    gridding_technique='pds',
                                    sampling_technique='bridson1',
                                    randuni_calc='by_points',
                                    lean='veryhigh', char_length=[3, 2],
                                    niter=500, make_point_objects=True,
                                    make_ckdtree=True,
                                    char_length_mean=0.24598,
                                    char_length_min=0.1111,
                                    char_length_max=0.9999,
                                    nt=10, space='linear', nchecks=10,
                                    sdf=0.75,
                                    periodic=[False, False, False]
                                    ):
        """
        Example
        -------
        from upxo.pxtal.vortess3d import gtess3d
        pxtal = gtess3d._make_pxtal_single_instance(spinput='gen',
                                        xbound=[0, 100], ybound=[0, 100],
                                        zbound=[0, 100], nsp=600,
                                        sp_distr='random',
                                        gridding_technique='pds',
                                        sampling_technique='bridson1',
                                        randuni_calc='by_points',
                                        lean='veryhigh', char_length=[3, 2],
                                        niter=500, make_point_objects=True,
                                        make_ckdtree=True,
                                        char_length_mean=0.24598,
                                        char_length_min=0.1111,
                                        char_length_max=0.9999,
                                        nt=10, space='linear', nchecks=10,
                                        sdf=0.75,
                                        )
        """
        print('Creating base seed points.')
        sp = cls.create_2d_base_seedpoints(spinput=spinput, xbound=xbound,
                                           ybound=ybound, nsp=nsp,
                                           sp_distr=sp_distr,
                                           gridding_technique=gridding_technique,
                                           sampling_technique=sampling_technique,
                                           randuni_calc=randuni_calc, lean=lean,
                                           char_length=char_length, niter=niter,
                                           make_point_objects=make_point_objects,
                                           make_ckdtree=make_ckdtree,
                                           char_length_mean=char_length_mean,
                                           char_length_min=char_length_min,
                                           char_length_max=char_length_max,
                                           nt=nt, space=space)
        print('Base Seed points created.')
        # --------------------------------------------------------------
        points = np.vstack((sp.locx, sp.locy)).T
        '''nchecks = 10  #<-- NOW MADE INPUT'''
        nchecks = np.round(sp.npoints/nchecks).astype(np.int32)
        tree = TREE(points)
        check_points = points[np.random.choice(range(points.shape[0]),
                                               nchecks, replace=False)]
        dist = []
        for cp in check_points:
            d, _ = tree.query(cp, k=[2, 3])
            dist.append(d.squeeze().tolist())

        sepdist = np.array(dist).mean(axis=0)/np.cbrt(3)/2
        sepdist_mean = sepdist.mean()
        sepdist_std = np.array(dist).std(axis=0)/np.cbrt(3)/2
        # sdf = 0.75  # sepdist_factor
        seed_points = [sp]
        sepdist_array = np.array([sepdist_mean]*sp.npoints).T

        # Introduce z-axis
        coords = [np.vstack((points.T, 0*sepdist_array)).T]
        bounds = cls.find_bounds_2dpoints(points)
        '''Use specified or calculate maximum domain size along z-axos'''
        maxdomsize = max([bounds['max_x']-bounds['min_x'],
                          bounds['max_y']-bounds['min_y']])
        domsize = 0
        i = 1
        print(40*'-', '\nCreating all remaining seed points.')
        _cdef_ = cls.create_2d_base_seedpoints
        while domsize <= maxdomsize-sepdist_mean:
            '''Create a new 2D multi-point object.'''
            sp = _cdef_(spinput=spinput, xbound=xbound,
                        ybound=ybound, nsp=nsp,
                        sp_distr=sp_distr,
                        gridding_technique=gridding_technique,
                        sampling_technique=sampling_technique,
                        randuni_calc=randuni_calc, lean=lean,
                        char_length=char_length, niter=niter,
                        make_point_objects=make_point_objects,
                        make_ckdtree=make_ckdtree,
                        char_length_mean=char_length_mean,
                        char_length_min=char_length_min,
                        char_length_max=char_length_max,
                        nt=nt, space=space)
            seed_points.append(sp)
            coord_m = np.vstack((sp.locx, sp.locy))
            sepdist_array = np.array([sepdist_mean]*sp.npoints).T
            # Introduce z-axis
            coords.append(np.vstack((coord_m, i*sepdist_array)).T)
            domsize += sepdist_mean
            i += 1
            print('distance along z: ', f'{i*sepdist_array[0]}')
        print(40*'-', '\nAll Seed points created.')
        # Shift x and y so everything will be positive
        dl = sepdist_mean
        # ---------------------------
        # Control dz
        dz = 0.
        # ---------------------------
        '''Add the z-coordinate'''
        coords = [coord + np.append(abs(coord.min(axis=0)[:2])+dl, dz)
                  for coord in coords]
        coords_all = np.vstack(coords)
        # --------------------------------------------------------------
        '''Construct 3D multi-point for coords_all'''
        coords_all_mp3d = mp3d.from_coords(coords_all)
        # --------------------------------------------------------------
        '''Calculate bounding box dimebnsions.'''
        bounds = cls.find_bounds_3dpoints(coords_all)
        bounding_box = [[bounds['min_x']-dl,
                         bounds['max_x']+dl],
                        [bounds['min_y']-dl,
                         bounds['max_y']+dl],
                        [bounds['min_z']-dl,
                         bounds['max_z']+dl]]
        # --------------------------------------------------------------
        '''Compute the actual voronoi tessellation in 3D using coords_all.'''
        pxtal = pyvoro.compute_voronoi(coords_all.tolist(),
                                       bounding_box, 1.0,
                                       periodic=[periodic[0],
                                                 periodic[1],
                                                 periodic[2]])
        # --------------------------------------------------------------
        pxt = {'pxtal': pxtal,
               'zbound': [0, bounding_box[2][1]],
               'sp_mp2d_list': seed_points,
               'sp_coords': coords,
               'sp_coords_all': coords_all,
               'sp_coords_mp3d': coords_all_mp3d,
               'sepdist_mean': sepdist_mean,
               'bounds': bounds,
               'bounding_box': bounding_box,
               }
        return pxt

    @staticmethod
    def create_2d_base_seedpoints(spinput='gen',
                                  xbound=[0, 100], ybound=[0, 100], nsp=600,
                                  sp_distr='random',
                                  gridding_technique='pds',
                                  sampling_technique='bridson1',
                                  randuni_calc='by_points',
                                  lean='veryhigh', char_length=[3, 2],
                                  niter=500, make_point_objects=True,
                                  make_ckdtree=True,
                                  char_length_mean=0.24598,
                                  char_length_min=0.1111,
                                  char_length_max=0.9999,
                                  nt=10, space='linear'):
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
                                    bridson_sampling_k=niter,
                                    make_point_objects=make_point_objects,
                                    make_ckdtree=make_ckdtree,
                                    vis=False, print_summary=False)
        return sp

    @staticmethod
    def find_bounds_2dpoints(points):
        if isinstance(points, np.ndarray):  # Check if it's a NumPy array
            if points.size == 0:  # Correct way to check if a NumPy array is empty
                return None
            min_x = np.min(points[:, 0]) # Efficiently get min/max of columns
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
        elif isinstance(points, list): # Check if it is a list
            if not points:
                return None
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')
            for x, y in points:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        else:
            return None  # Handle other invalid input types

        return {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}

    @staticmethod
    def find_bounds_3dpoints(points):
        if isinstance(points, np.ndarray):
            if points.size == 0:
                return None
            if points.shape[1] != 3:
                return None
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
        elif isinstance(points, list):  # Check if it is a list
            if not points:
                return None
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            min_z, max_z = float('inf'), float('-inf')
            for x, y, z in points:  # Unpack x, y, and z
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
        else:
            return None  # Handle other invalid input types

        return {'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
                'min_z': min_z, 'max_z': max_z}

    def visualize_voronoi_cell(cell, ax=None, color='cyan', alpha=0.5,
                               edge_color='black'):
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

        vertices = np.array(cell['vertices'])  # Convert vertices to NumPy array
        faces = cell['faces']  # Get face definitions

        # Plot each face
        for face in faces:
            face_vertices = [vertices[i] for i in face['vertices']]  # Get vertex coordinates
            poly = Poly3DCollection([face_vertices], alpha=alpha, edgecolor=edge_color)
            poly.set_facecolor(color)
            ax.add_collection3d(poly)

        # Plot the seed point
        seed = cell['original']
        ax.scatter(seed[0], seed[1], seed[2], color='red', s=100, label="Seed Point")

        # Set axis limits
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Voronoi Cell Visualization")

        return ax

    def visualize_multiple_voronoi_cells(self,
                                         inst,
                                         cell_indices=None,
                                         colors=None,
                                         alpha=0.5,
                                         edge_color='black'):
        tessellation = self.pxtals[inst]
        # ---------------------------------------------------
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Use all cells if no specific indices are provided
        if cell_indices is None:
            cell_indices = list(range(len(tessellation)))

        # Generate random colors if none provided
        if colors is None:
            colors = plt.cm.jet(np.linspace(0, 1, len(cell_indices)))  # Use a colormap

        for i, cell_index in enumerate(cell_indices):
            cell = tessellation[cell_index]
            color = colors[i % len(colors)]  # Assign color

            vertices = np.array(cell['vertices'])  # Convert vertices to NumPy array
            faces = cell['faces']  # Get face definitions

            # Plot each face of the cell
            for face in faces:
                face_vertices = [vertices[i] for i in face['vertices']]  # Get vertex coordinates
                poly = Poly3DCollection([face_vertices], alpha=alpha, edgecolor=edge_color)
                poly.set_facecolor(color)
                ax.add_collection3d(poly)

            # Plot the seed point of the cell
            seed = cell['original']
            ax.scatter(seed[0], seed[1], seed[2], color='red', s=50, label=f"Seed {cell_index}")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Voronoi Cells Visualization")
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.show()

    def construct_pyvista_surface(cell):


        vertices = np.array(cell["vertices"], dtype=np.float64)
        faces = []

        for face in cell["faces"]:
            face_vertices = [v for v in face["vertices"]]
            faces.append(len(face_vertices))  # First: Number of vertices in this face
            faces.extend(face_vertices)  # Then, the actual vertex indices

        # Convert faces into a NumPy array
        faces_array = np.array(faces, dtype=np.int64)

        # Create PyVista PolyData (surface representation)
        polydata = pv.PolyData(vertices, faces_array)

        return polydata

    def generate_uniform_tet_mesh(surface, resolution=5, maxvol=0.001):


        # Step 1: Extract the bounding box of the surface
        bounds = surface.bounds  # xmin, xmax, ymin, ymax, zmin, zmax

        # Step 2: Generate interior seed points in a regular grid
        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[2], bounds[3], resolution)
        z = np.linspace(bounds[4], bounds[5], resolution)
        xx, yy, zz = np.meshgrid(x, y, z)
        interior_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        # Step 3: Initialize TetGen with the surface
        tet = tetgen.TetGen(surface)

        # Step 4: Add interior points to force uniform meshing
        tet.add_points(interior_points)

        # Step 5: Perform tetrahedralization with uniform constraints
        options = f"pq1.5a{maxvol}Y"  # "Y" prevents only surface refinement
        tet.tetrahedralize(options=options)

        # Step 6: Convert back to PyVista format and visualize
        tet_mesh = tet.grid
        return tet_mesh

    # uniform_tet_mesh = generate_uniform_tet_mesh(fine_surface, resolution=10, maxvol=0.0005)


    def interactive_slice_view(self, tet_mesh):

        # Setup PyVista plotter
        plotter = pv.Plotter()

        # Add full tetrahedral mesh with transparency
        plotter.add_mesh(tet_mesh, color='lightgray', opacity=0.2, show_edges=True)

        # Define an initial slicing plane
        def update_plane(normal, origin):
            sliced_mesh = tet_mesh.slice(normal=normal, origin=origin)
            plotter.add_mesh(sliced_mesh, color='blue', show_edges=True, line_width=1.5)

        # Add interactive plane widget
        plotter.add_mesh_clip_plane(tet_mesh,
                                    crinkle=True,
                                    show_edges=True,
                                    normal=[1, 0, 0])  # Start with X-plane

        # Display the interactive visualization
        plotter.show()

    # interactive_slice_view(tet_mesh)

    def compute_cell_quality(self, tet_mesh, metric="aspect_ratio"):

        quality_mesh = tet_mesh.compute_cell_quality(quality_measure=metric)
        quality_values = quality_mesh["CellQuality"]  # Extract computed cell quality values
        tet_mesh.cell_data[metric] = quality_values
        return tet_mesh

    def interactive_slice_view_with_quality(self, tet_mesh, metric="aspect_ratio",
                                            cmap="nipy_spectral", clim=[1, 5]):


        quality_values = self.compute_cell_quality(tet_mesh, metric)


        # Setup PyVista plotter
        plotter = pv.Plotter()

        # Add full tetrahedral mesh with transparency
        plotter.add_mesh(quality_values, scalars=metric, opacity=0.1, show_edges=True,
                         clim=clim)

        # Enable interactive slicing, color by the selected metric
        plotter.add_mesh_clip_plane(quality_values,
                                    crinkle=True,
                                    scalars=metric,
                                    cmap=cmap,
                                    show_edges=True,
                                    normal=[-1, -1, 0],
                                    clim=clim)  # Start with X-plane

        # Show interactive visualization
        plotter.show()

    def compute_cell_quality(self, tet_mesh, metric="aspect_ratio"):

        quality_mesh = tet_mesh.compute_cell_quality(quality_measure=metric)
        quality_values = quality_mesh["CellQuality"]  # Extract computed cell quality values
        tet_mesh.cell_data[metric] = quality_values
        return tet_mesh

    def plot_multiple_meshes(meshes, scalars=None, cmap="viridis", clim=[0, 1], opacity=0.5, show_edges=True):

        # Setup the PyVista plotter
        plotter = pv.Plotter()

        # Iterate over all meshes and add them to the plotter
        for mesh in meshes:
            # Add each mesh with optional scalar data and properties
            plotter.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim, opacity=opacity, show_edges=show_edges)

        # Show the combined visualization
        plotter.show()

    def interactive_slice_view_with_quality_multiple_meshes(self, tet_meshes,
                                                            metric="aspect_ratio",
                                                            cmaps=["nipy_spectral",
                                                                   "viridis"],
                                                            clim=[1, 5]):

        quality_values = [self.compute_cell_quality(tet_mesh, metric)
                          for tet_mesh in tet_meshes]

        plotter = pv.Plotter()
        for i, qualval in enumerate(quality_values):
            plotter.add_mesh(qualval,
                             scalars=metric,
                             opacity=0.1,
                             show_edges=True,
                             clim=clim)

            plotter.add_mesh_clip_plane(qualval,
                                        crinkle=True,
                                        scalars=metric,
                                        cmap=cmaps[i],
                                        show_edges=True,
                                        normal=[-1, 0, -1],
                                        clim=clim)

        # Show interactive visualization
        plotter.show()


    def generate_background_mesh(bounds, resolution=20, eps=1e-6):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        grid_x, grid_y, grid_z = np.meshgrid(
            np.linspace(xmin - eps, xmax + eps, resolution),
            np.linspace(ymin - eps, ymax + eps, resolution),
            np.linspace(zmin - eps, zmax + eps, resolution),
            indexing="ij",
        )
        return pv.StructuredGrid(grid_x, grid_y, grid_z).triangulate()

    '''bg_mesh = generate_background_mesh(surface.bounds)
    bg_mesh.plot(show_edges=True)'''

    def sizing_function(points, focus_point=np.array([0, 0, 0]), max_size=1.0, min_size=0.1):
        distances = np.linalg.norm(points - focus_point, axis=1)
        return np.clip(max_size - distances, min_size, max_size)
