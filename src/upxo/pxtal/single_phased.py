import numpy as np
import pyvista as pv
from copy import deepcopy
from upxo._sup import dataTypeHandlers as dth
from upxo.pxtal.mcgs3_temporal_slice import mcgs3_grain_structure
from dataclasses import dataclass

class OFHC_Cu_vox():
    _slots = ("twin_setup", "twspec", "fdb")
    __slots__ = _slots + mcgs3_grain_structure.__slots__
    _propdtype_ = np.float32
    _iddtype_ = np.int32
    _common_mprops_ = True
    
    
    @dataclass
    class TwinSetup:
        nprops: int
        mprops: list
        twin_axis: tuple
        
    @dataclass
    class TwinMorphSpec_v1:
        n: list
        tv: np.array
        dlk: np.array
        dnw: np.array
        dno: np.array
        tdis: str
        dlk: dict
        dlk: list
        dlk: bool
        
    class FDB:
        def __init__(self):
            pass

    @dataclass
    class MPropSetup:
        volnv: dict
        rat_sanv_volnv: dict

    
    def __init__(self, lgi=None, ea=None):
        super().__init__()
        # -----------------------------------
        _fid_ = np.unique(lgi)
        self.fdb = {}
        self.fdb = {'base': {'tess': lgi, 'ea': ea, 'fid': _fid_,
                             'n': _fid_.size,
                             'mprops': None,
                             'neigh_fid': None
                             }
                    }
        del _fid_
        # -----------------------------------
        self.twin_setup = {'nprops': 2,
                           'mprops': {'volnv': {'use': True, 'reset': False,
                                                'k': [.02, 1.0], 'min_vol': 4,},
                                      'rat_sanv_volnv': {'use': True,
                                                         'reset': False,
                                                         'k': [0.0, .8],
                                                         'sanv_N': 26}, } }
        # -----------------------------------
        self.twspec = {'n': [5, 10, 3], 
                       'tv': np.array([5, -3.5, 5]),
                       'dlk': np.array([1.0, -1.0, 1.0]),
                       'dnw': np.array([0.5, 0.5, 0.5]),
                       'dno': np.array([0.5, 0.5, 0.5]), 
                       'tdis': 'normal',
                       'tpar': {'loc': 1.12, 'scale': 0.25, 'val': 1},
                       'vf': [0.05, 1.00], 
                       'sep_bzcz': False}
        # -----------------------------------
        self.twgenspec = {'seedsel': 'random_gb', 'K': 10,
                          'bidir_tp': False, 'checks': [True, True],}
        
    def add_to_fdb(self, name, value):
        """
        value = {'lgi': lgi, 
                 'ea': ea, 
                 'mprops': None,
                 'neigh_fid': None}
        pxtal.add_to_fdb('base', value)
        """
        if name == "base":
            if "lgi" in value and value["lgi"]:
                self.add_lgi(value['lgi'])
            else:
                KeyError("Value dictionary has no lgi key or no valid lgi key")
        
    def add_data_to_fdb(self, name, value, calc_flag):
        if name == "lgi":
            self.fdb['base']['lgi'] = value
        elif name == "ea":
            self.fdb['base']['lgi'] = value
        elif name == "fid":
            if self.fdb['base']['lgi']:
                if calc_flag:
                    fid = np.unique(self.fdb['base']['lgi'])
                    self.fdb['base']['fid'] = fid
                    
                else:
                    pass
            else:
                 raise ValueError("")   
            self.fdb['base']['lgi'] = value
        elif name == "n":
            self.fdb['base']['lgi'] = value
        elif name == "mprops":
            self.fdb['base']['lgi'] = value
        elif name == "neigh_gid":
            self.fdb['base']['lgi'] = value

    def setup_twins:
        add_to_fdb

    def setup_for_twins(self, nprops=2, mprops=None, instance_name='twin.0',
                        feature_name='annealing_twin', viz_grains=False, 
                        opacity=1.0):
        """ Carry out pre-requisite operations needed to establish twins """
        print('Finding twin host grains.')
        _gid_data_ = self.find_twin_hosts(nprops=nprops, mprops=mprops,
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
                     dnames=('fid', 'feat_host_gids'),
                     datas=(deepcopy(self.lgi), GIDS),
                     info={'name': feature_name, 'mprops': mprops,
                           'vf_min': None, 'vf_max': None, 'vf_actual': None})
        
    def setup_gid_twin(self, GIDS):
        self.gid_twin = {gid: None for gid in GIDS}
        
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
        self.fdb[fname]['data'] = {}
        for dname, data in zip(dnames, datas):
            self.fdb[fname]['data'][dname] = data
        self.fdb[fname]['info'] = info

    def find_twin_hosts(self, nprops=2,
                        mprops={'volnv': {'use': True, 'reset': False,
                                          'k': [.1, .8], 'min_vol': 4,},
                                'rat_sanv_volnv': {'use': True, 'reset': False,
                                                   'k': [.1, .8], 'sanv_N': 26,}, },
                        viz_grains=False, opacity=1.0):
        if self._common_mprops_:
            mprops = self.twin_setup['mprops']
        # ---------------------------------------------------------------------
        print(40*'-', '\nFinding grains which can host twins.\n')
        # ---------------------------------------------------------------------
        # Validate mprops
        for mn in mprops.keys():
            if mn not in ('volnv', 'rat_sanv_volnv', ):
                print('  Invalid mprop names specified.')
                print('  ONly volnv, rat_sanv_volnv allowed as of now.')
                return None
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
        # ---------------------------------------------------------------------
        # Find the actual number of properties to use based on user input
        # mprop flag
        nprops = np.sum([1 for mn in mprops.keys() if mprops[mn]['use']])
        # ---------------------------------------------------------------------
        GIDS_masks_mprops = np.full(nprops+1, None)
        GIDS = {mn: None for mn in mprops.keys()}
        # ---------------------------------------------------------------------
        mprop_i = 0
        for i, mn in enumerate(mprops.keys(), start=0):
            if mn in ('volnv', 'rat_sanv_volnv') and mprops[mn]['use']:
                print(f'Caclulaing gid_masks for mprop: {mn}.')
                d = np.array(list(self.mprop[mn].values()))  # Data
                f = mprops[mn]['k']  # User defined factors
                dmax = d.max()  # Data maximum
                dl = dmax*f[0]  # Data low
                dh = dmax*f[1]  # Data high
                GIDS_masks_mprops[i] = np.logical_and(d >= dl, d <= dh)
                mprop_i += 1
        # ---------------------------------------------------------------------
        '''Identify multi-voxel grains'''
        vol = np.array(list(self.mprop['volnv'].values()))
        GIDS_masks_mprops[mprop_i] = vol >= mprops['volnv']['min_vol']
        # ---------------------------------------------------------------------
        GIDS_masks_mprops = np.stack(GIDS_masks_mprops, axis=1)
        GIDS_mask = np.prod(GIDS_masks_mprops, axis=1).astype(bool)
        GIDS = np.argwhere(GIDS_mask).T[0]+1
        # ---------------------------------------------------------------------
        if viz_grains:
            self.make_pvgrid()
            self.add_scalar_field_to_pvgrid(sf_name="lgi", sf_value=None)
            self.plot_grains(GIDS+1, opacity=opacity, show_edges=False)
        # ---------------------------------------------------------------------
        return GIDS_masks_mprops, GIDS_mask, GIDS
    
    def set_mprop_volnv(self):
        """Calculate the volume by number of voxels."""
        print(40*"-", "\nSetting grain volumes (metric: 'volnv') -> ")
        unique_counts = np.bincount(self.lgi.ravel(), minlength=self.n+1)
        self.mprop['volnv'] = {gid + 1: unique_counts[gid + 1] for gid in range(self.n)}
        print("Grain volumes (metric: 'volnv') -> have been set.\n", 40*"-")

    def set_mprop_sanv(self, N=26, verbosity=100):
        """Calculate the total surface area by number of voxels."""
        print("\nCalculating feature surface areas (metric: 'sanv').")

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

    def instantiate_twins(self,
                          ninstances=2,
                          base_gs_name_prefix='twin.',
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
            self.setup_for_twins(nprops=self.twin_setup['nprops'],
                                 mprops=self.twin_setup['mprops'],
                                 instance_name=instance_name,
                                 viz_grains=False)
            self.identify_twins(base_gs_name=instance_name,
                                twspec=self.twspec,
                                twgenspec=self.twgenspec,
                                viz=False)
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
