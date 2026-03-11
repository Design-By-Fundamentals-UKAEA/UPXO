# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:11:45 2024

@author: rg5749
"""

import sys
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import namedtuple
from matplotlib import cm
from scipy import stats
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import kruskal
from scipy.stats import entropy
from upxo.ggrowth.mcgs import mcgs
from upxo.pxtalops.Characterizer import mcgs_mchar_2d
from upxo._sup import dataTypeHandlers as dth
from upxo.geoEntities.mulpoint2d import MPoint2d
from upxo.interfaces.user_inputs.excel_commons import read_excel_range
from upxo.interfaces.user_inputs.excel_commons import write_array_to_excel
from upxo._sup.data_ops import find_outliers_iqr, distance_between_two_points
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
from upxo.statops.stattests import test_rand_distr_autocorr
from upxo.statops.stattests import test_rand_distr_runs
from upxo.statops.stattests import test_rand_distr_chisquare
from upxo.statops.stattests import test_rand_distr_kolmogorovsmirnov
from upxo.statops.stattests import test_rand_distr_kullbackleibler
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.algorithms import community
from scipy.stats import wasserstein_distance, ks_2samp, energy_distance
import netlsd
from upxo._sup.data_ops import calculate_angular_distance as calc_angdist
from upxo._sup.data_ops import calculate_density_bins
from upxo._sup.data_ops import approximate_to_bin_means
from upxo.netops import kmake
from tqdm import tqdm
from upxo.netops.kmake import make_gid_net_from_neighlist

import upxo.netops.kmake as kmake
import upxo.netops.kcmp as kcmp
import upxo.netops.kchar as kchar

NUMBERS = dth.dt.NUMBERS
ITERABLES = dth.dt.ITERABLES
RNG = np.random.default_rng()
DCOPY = deepcopy

class KREPR():
    """
    Docstring.

    Import
    ------
    from upxo.repqual.grain_network_repr_assesser import KREPR

    Parameters
    ----------
    upxogs_tgt, upxogs_smp: UPXO grain structure data

    tgset: dict
        Set of target grain structures.
        keys: int. grain strucyure IDs.
        values: must be any of the UPXO grain structure type.
        note: keys are usually tslice values, but can be user defined.
        Example:
        {0: <upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure at 0x2d37fcb54f0>,
         1: <upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure at 0x2d37fcb5680>,
         2: <upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure at 0x2d37fcb5810>,
         ...}
    sgset: dict
        Set of sample grain structures.
        keys: int. grain strucyure IDs.
        values: must be any of the UPXO grain structure type.
        note: keys are usually tslice values, but can be user defined.
        Example:
        {0: <upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure at 0x2d37fcb54f0>,
         1: <upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure at 0x2d37fcb5680>,
         2: <upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure at 0x2d37fcb5810>,
         ...}

    tkset: dict
        Set of networkx graphs of target grain structure neighbour networks.
        keys: int/float. Neighbour order, ordern value.
        values: dict
            keys: int. grain (node) IDs.
            values: list. contains neighbouring grain ids (gids), ie. node ids.
            note: keys are usually tslice values, but can be user defined.
    skset: dict
        Set of networkx graphs of sample grain structure neighbour networks.
        keys: int/float. Neighbour order, ordern value.
        values: dict
            keys: int. grain (node) IDs.
            values: list. contains neighbouring grain ids (gids), ie. node ids.
            note: keys are usually tslice values, but can be user defined.

    tnset: dict
        Set of UPXO returned target grain structure neighbour mapping.
        keys: int/float. Neighbour order, ordern value.
        values: dict
            keys: int. grain IDs.
            values: list. contains neighbouring grain ids (gids).
            note: keys are usually tslice values, but can be user defined.
    snset: dict
        Set of UPXO returned sample grain structure neighbour mapping.
        keys: int/float. Neighbour order, ordern value.
        values: dict
            keys: int. grain IDs.
            values: list. contains neighbouring grain ids (gids).
            note: keys are usually tslice values, but can be user defined.

    tmpset: dict
        Target morphological property set.
        keys: names of morphological properties.
        values: dict
            keys: O(n)
            values: dict
                keys: int. gid: grain id values
                values: list: neighbour gids.
    smpset: dict
        Sample morphological property set.
        keys: names of morphological properties.
        values: dict
            keys: O(n)
            values: dict
                keys: int. gid: grain id values
                values: list: neighbour gids.

    tid: list
        Contains usable values of target grain ids. MUST belong to tgset.keys()
        and/or tkset.keys() depending on study path.
        Note: if not set by the user, all tkset.keys() will be assigned !!
    sid: list
        Contains usable values of target grain ids. MUST belong to tgset.keys()
        and/or tkset.keys() depending on study path.
        Note: if not set by the user, all skset.keys() will be assigned !!

    ordern: list
        Contains values of the ordern. If not set by the user, ordern=1
        will be assigned.

    mprop3d_flags, sprop3d_flags: all dict types

    mprop3d, sprop2d, sprop3d: all dict types

    rkf: dict
        Contains network (k) based R-Field (i.e. rkf) data.
        Thi smust be set before representativeness assessment is done.

    _cim_: str
        Class initiation method, not intended for user use.

    Author: Dr. Sunil Anandatheertha

    om jayanti man'gaLA kALi bhadrakALi kapAlini |
    durgA kshamA shivA dhAtri svAhA svadhA namOstute ||
    """

    __slots__ = ('gstype', 'upxogs_tgt', 'upxogs_smp',
                 'tgset', 'sgset', 'tkset', 'skset', 'tnset', 'snset',
                 'tmpset', 'smpset',
                 'tid', 'sid', 'ntid', 'nsid',
                 'ordern', 'dim',
                 'rkf_flags', 'rkf',
                 'mprop3d_flags', 'mprop3d', '_cim_',)

    mp_gspn_map = {'volnv': 'volnv', 'volsr': 'volsr', 'volch': 'volch',
                   'sanv': 'sanv', 'savi': 'savi', 'sasr': 'sasr',
                   'psa': 'psa',
                   'rat_sanv_volnv': 'rat_sanv_volnv',
                   'pernv': 'pernv', 'pervl': 'pervl', 'pergl': 'pergl',
                   'eqdia': 'eqdia', 'feqdia': 'feqdia',
                   'kx': 'kx', 'ky': 'ky', 'kz': 'kz', 'ksr': 'ksr',
                   'arbbox': 'arbbox', 'arellfit': 'arellfit',
                   'sol': 'sol', 'ecc': 'ecc', 'com': 'com', 'sph': 'sph',
                   'fn': 'fn', 'rnd': 'rnd', 'fdim': 'fdim',
                   'mi': 'mi',}

    def __init__(self, **kwargs):
        self._cim_ = kwargs['_cim_']
        # --------------------------------------------
        self.dim = namedtuple('dim', ['tgt', 'smp'])
        self.gstype = namedtuple('gstype', ['tgt', 'smp'])
        # ==========================================================
        # TEND TO DIFFERENT CLASS CREATOIPN METHODS
        if kwargs['_cim_'] == 'from_gs':
            # Validations
            self.gstype.tgt = kwargs['gstype_tgt']
            self.gstype.smp = kwargs['gstype_smp']
            # --------------------------------------------
            self.upxogs_tgt = kwargs['upxogs_tgt']
            self.upxogs_smp = kwargs['upxogs_smp']
            tgset, sgset = kwargs['tgset'], kwargs['sgset']
            self.tgset = {i: gs for i, gs in tgset.items()}
            self.sgset = {i: gs for i, gs in sgset.items()}
            # self.tid, self.sid = kwargs['tid'], kwargs['sid']
            self.tkset, self.skset = None, None
            self.tnset, self.snset = None, None
        elif kwargs['_cim_'] == 'from_k':
            # Validations
            self.gstype.tgt, self.gstype.smp = None, None
            self.upxogs_tgt, self.upxogs_smp = None, None
            self.tgset, self.sgset = None, None
            self.tkset, self.skset = kwargs['tkset'], kwargs['skset']
            # self.tid, self.sid = kwargs['tid'], kwargs['sid']
            self.tnset, self.snset = None, None
        elif kwargs['_cim_'] == 'from_neigh':
            # Validations
            self.gstype.tgt, self.gstype.smp = None, None
            self.tgset, self.sgset = None, None
            self.tkset, self.skset = None, None
            self.tnset, self.snset = kwargs['tnset'], kwargs['snset']
            self.tid, self.sid = kwargs['tid'], kwargs['sid'],
        elif kwargs['_cim_'] == 'from_gsgen':
            """
            Note @dev:
                This branch is as of now, identical to 1st branch 'from_gs'.
                This is expected to change with further development.
                Continue to develop identical to 1st branch 'from_gs'.
            """
            # Validations
            self.gstype.tgt = kwargs['gstype_tgt']
            self.gstype.smp = kwargs['gstype_smp']
            self.upxogs_tgt = kwargs['upxogs_tgt']
            self.upxogs_smp = kwargs['upxogs_smp']
            tgset, sgset = kwargs['tgset'], kwargs['sgset']
            self.tgset = {i: gs for i, gs in tgset.items()}
            self.sgset = {i: gs for i, gs in sgset.items()}
            # self.tid, self.sid = kwargs['tid'], kwargs['sid']
            self.tkset, self.skset = None, None
            self.tnset, self.snset = None, None
        # ==========================================================
        # Set the dimensioanlities of the problem.
        self.init_subdef_set_dim()
        self.init_subdef_set_gsid(kwargs)
        self.init_subdef_set_neighs(kwargs)
        self.init_subdef_set_networks()
        self.init_subdef_set_prop_flags()
        # ==========================================================

    @classmethod
    def from_gs(cls, *, upxogs_tgt=None, upxogs_smp=None,
                tgset=None, sgset=None, ordern=[1],
                tsid_source='from_gs', ssid_source='from_gs',
                tid=None, sid=None, _cim_='from_gs'):
        """
        Intiantiate network based repr class using UPXO grain structure.

        Parameters
        ----------
        upxogs_tgt
        upxogs_smp
        tgset: dict
            Target grain structures. Defaults to None.
        sgset: dict
            Sample grain structures. Defaults to None.
        ordern: list
            Neighbour order-n to be used. Defaults to [1].
        _cim_: str
            Class initiation method. Defaults to 'from_gs'.
            Not intended for user. Leave it alone.

        from upxo.ggrowth.mcgs import mcgs
        tgt = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        tgt.simulate()
        tgt.detect_grains()
        tgset = {i: gs for i, gs in tgt.gs.items()}

        from upxo.repqual.grain_network_repr_assesser import KREPR
        kr = KREPR.from_gs(tgset=tgset, sgset=tgset)
        kr.creation_method
        """
        # Validations
        return cls(upxogs_tgt=upxogs_tgt, upxogs_smp=upxogs_smp,
                   tgset=tgset, sgset=sgset, ordern=ordern,
                   tsid_source=tsid_source, ssid_source=ssid_source,
                   tid=tid, sid=sid,
                   _cim_=_cim_)

    @classmethod
    def from_neigh(cls, *, tnset=None, snset=None, ordern=[1],
                   tsid_source='from_neigh', ssid_source='from_neigh',
                   tid=None, sid=None, _cim_='from_neigh'):
        """
        # Assuming 20 tslices being available witrh increments of tslice=1,
        # we will go through the folloing example.

        ordern = [1, 3]

        from upxo.ggrowth.mcgs import mcgs
        tgt = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        tgt.simulate()
        tgt.detect_grains()
        tslices = np.array(list(tgt.gs.keys()))[1::10]
        tnset = {no: {tslice: None for tslice in tslices} for no in ordern}
        for no in ordern:
            for tslice in tslices:
                _ = tgt.gs[tslice].get_upto_nth_order_neighbors_all_grains
                tnn = _(no, include_parent=True, output_type='nparray')
                tnset[no][tslice] = tnn

        smp = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        smp.simulate()
        smp.detect_grains()
        tslices = np.array(list(smp.gs.keys()))[1::10]
        snset = {no: {tslice: None for tslice in tslices} for no in ordern}
        for no in ordern:
            for tslice in tslices:
                _ = smp.gs[tslice].get_upto_nth_order_neighbors_all_grains
                snn = _(no, include_parent=True, output_type='nparray')
                snset[no][tslice] = snn

        from upxo.repqual.grain_network_repr_assesser import KREPR
        kr = KREPR.from_neigh(tnset=tnset, snset=tnset,
                              tid=list(tnset.keys()),
                              sid=list(snset.keys()),
                              _cim_='from_neigh')
        kr.snset.keys()
        kr.snset[3].keys()
        kr.snset[3][11]
        kr.snset[3][11][40]  # <-- O(3) Neigh gids of gid=40 of tslice = 11
        kr.ordern
        kr.tid
        """
        return cls(tnset=tnset, snset=snset, ordern=ordern,
                   tsid_source=tsid_source, ssid_source=ssid_source,
                   tid=tid, sid=sid,
                   _cim_=_cim_)

    @classmethod
    def from_k(cls, *, tkset=None, skset=None, ordern=[1],
               tsid_source='from_k', ssid_source='from_k',
               tid=None, sid=None,_cim_='from_k'):
        """
        # Assuming 20 tslices being available witrh increments of tslice=1,
        # we will go through the folloing example.

        ordern = [1, 3]

        from upxo.ggrowth.mcgs import mcgs
        tgt = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        tgt.simulate()
        tgt.detect_grains()
        tslices = np.array(list(tgt.gs.keys()))[1::10]
        tkset = {no: {tslice: None for tslice in tslices} for no in ordern}
        for no in ordern:
            for tslice in tslices:
                _ = tgt.gs[tslice].get_upto_nth_order_neighbors_all_grains
                tnn = _(no, include_parent=True, output_type='nparray')
                tnn_k = kmake.create_grain_network_nx(tnn)
                tkset[no][tslice] = tnn_k

        smp = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        smp.simulate()
        smp.detect_grains()
        tslices = np.array(list(smp.gs.keys()))[1::10]
        skset = {no: {tslice: None for tslice in tslices} for no in ordern}
        for no in ordern:
            for tslice in tslices:
                _ = smp.gs[tslice].get_upto_nth_order_neighbors_all_grains
                snn = _(no, include_parent=True, output_type='nparray')
                snn_k = kmake.create_grain_network_nx(snn)
                skset[no][tslice] = snn_k

        from upxo.repqual.grain_network_repr_assesser import KREPR
        kr = KREPR.from_k(tkset=tkset, skset=tkset,
                          tid=list(tkset.keys()),
                          sid=list(skset.keys()),
                          _cim_='from_k')
        kr.tkset
        kr.ordern
        kr.tid
        """
        # Validations
        return cls(tkset=tkset, skset=skset, ordern=ordern,
                   tsid_source=tsid_source, ssid_source=ssid_source,
                   tid=tid, sid=sid,
                   _cim_=_cim_)

    @classmethod
    def from_gsgen(cls, gstype_tgt='mcgs2d', gstype_smp='mcgs3d',
                   is_smp_same_as_tgt=False,
                   char_tgt=True, char_smp=True,
                   set_mprops_tgt=False, set_mprops_smp=False,
                   tgt_dashboard='input_dashboard_krepr1.xls',
                   smp_dashboard='input_dashboard_krepr2.xls', ordern=[1],
                   tsid_source='from_neigh', ssid_source='from_neigh',
                   tid=None, sid=None, _cim_='from_gsgen',
                   label_str_order=1,
                   mpflags={'volnv': True, 'volsr': False, 'volch': False,
                            'sanv': True, 'savi': False, 'sasr': False,
                            'rat_sanv_volnv': True,
                            'psa': False,
                            'pernv': False, 'pervl': False, 'pergl': False,
                            'eqdia': False, 'feqdia': False,
                            'kx': False, 'ky': False, 'kz': False,
                            'ksr': False,
                            'arbbox': True, 'arellfit': False,
                            'sol': False, 'ecc': False, 'com': False,
                            'sph': False,
                            'fn': False, 'rnd': False, 'fdim': False,
                            'mi': False,
                            }):
        """
        Initiate KREPR by generating target and sample grain structure sets.

        Parameters
        ----------
        gstype: str
            Type of grain structure needed.
            Could be deprecated later on.
            Defaults to 'mcgs'.
        is_smp_same_as_tgt: bool
            Defaults to False.
        tgt_dashboard: str
            Defaults to 'input_dashboard.xls'.
        smp_dashboard: str
            Defaults to 'input_dashboard.xls'.
        _cim_: str
            Defaults to 'from_gsgen'.

        Explanations
        ------------

        Example
        -------
        import numpy as np
        from upxo.repqual.grain_network_repr_assesser_3D import KREPR
        import matplotlib.pyplot as plt

        neigh_orders=[3]

        kr = KREPR.from_gsgen(gstype_tgt='mcgs3d', gstype_smp='mcgs3d',
                              is_smp_same_as_tgt = False,
                              characterize_tgt=True, characterize_smp=True,
                              set_mprops_tgt=False, set_mprops_smp=False,
        					  tgt_dashboard='input_dashboard.xls',
        					  smp_dashboard='input_dashboard.xls',
        					  ordern=neigh_orders,
                              tsid_source='user', ssid_source='user',
        					  tid=np.arange(10, 20, 1),
                              sid=np.arange(10, 20, 1),
                              _cim_='from_gsgen')

        kr.set_mprop3d_flags(volnv=True, sanv=True, rat_sanv_volnv=True)

        kr.calculate_mprop3d(print_msg_tors=True, print_msg_prnm=True,
                             print_msg_no=True, print_msg_gsid=False,
                             print_msg_gid=False)


        kr.set_rkf(js=True, wd=True, ksp=True, ed=True, nlsd=True,
                   degcen=False, btwcen=False, clscen=False, egnvcen=False)
        kr.calculate_rkf()
        """
        '''IUf any keys are missing in mpflags, then make up the missing ones
        with False values.'''
        mpflags.update({key: False
                        for key in cls.mp_gspn_map.keys()
                        if key not in mpflags.keys()})
        # ====================================================================
        # ====================================================================
        #       TARGET GRAIN STRUCTURE
        # ====================================================================
        # ====================================================================
        print('GENERATING TARGET GRAIN STRUCTURES')
        tgt = mcgs(study='independent', input_dashboard=tgt_dashboard)
        tgt.simulate()
        tgt.detect_grains()
        if tid is None:
            tid = list(tgt.gs.keys())
        else:
            # validate user input tid
            pass
        tgset = {i: gs for i, gs in tgt.gs.items()}
        # ====================================================================
        if char_tgt:
            ngs = len(tid)
            print(50*'#',
                  '\n Characterizing target gs database.',
                  '\n', 10*'. ')
            for tid_count, _tid_ in enumerate(tid):
                print(f'Target gsid: {tid_count} of {ngs}, gs. no. {_tid_}')
                _def_ = tgt.gs[_tid_].char_morphology_of_grains
                _def_(label_str_order=label_str_order,
                      make_pvgrid=False,
                      find_neigh=[True, ordern],
                      find_grain_voxel_locs=True,
                      find_spatial_bounds_of_grains=True,
                      find_grain_locations=True,
                      force_compute=True,)
        # ====================================================================
        if set_mprops_tgt:
            print(50*'#',
                  '\n Calculating morpho props of target gs database.',
                  '\n', 10*'. ')
            ngs = len(tid)
            for tid_count, _tid_ in enumerate(tid):
                print(f'Target gsid: {tid_count} of {ngs}')
                # tgt.gs[i].find_spatial_bounds_of_grains()
                gs = tgt.gs[_tid_]
                _msg_ = f'tslice {tid_count} of {ngs}, gs. no. {_tid_}'
                gs.set_mprops(volnv=mpflags['volnv'],
                              eqdia=mpflags['eqdia'],
                              eqdia_base_size_spec='volnv',
                              arbbox=mpflags['arbbox'],
                              arbbox_fmt='gid_dict',
                              arellfit=mpflags['arellfit'],
                              arellfit_metric='max',
                              arellfit_calculate_efits=True,
                              arellfit_efit_routine=1,
                              arellfit_efit_regularize_data=True,
                              solidity=mpflags['sol'],
                              sol_nan_treatment='replace',
                              sol_inf_treatment='replace',
                              sol_nan_replacement=-1,
                              sol_inf_replacement=-1,
                              sanv=mpflags['sanv'],
                              sanv_N=26,
                              rat_sanv_volnv=mpflags['rat_sanv_volnv'],
                              sanv_verbosity=1E2,
                              disp_msg=_msg_)
            print(50*'#')
        # ====================================================================
        # ====================================================================
        #       SAMPLE GRAIN STRUCTURE
        # ====================================================================
        # ====================================================================
        if not is_smp_same_as_tgt:
            print('GENERATING SAMPLE GRAIN STRUCTURES')
            smp = mcgs(study='independent', input_dashboard=smp_dashboard)
            smp.simulate()
            smp.detect_grains()
            if sid is None:
                sid = list(smp.gs.keys())
            else:
                # validate user input sid
                pass
            sgset = {i: gs for i, gs in smp.gs.items()}
        else:
            smp = deepcopy(tgt)
            sgset = deepcopy(tgset)
        # ====================================================================
        if char_smp:
            ngs = len(sid)
            print(50*'#',
                  '\n Characterizing Sample gs database.',
                  '\n', 10*'. ')
            for sid_count, _sid_ in enumerate(sid):
                print(f'Target gsid: {sid_count} of {ngs}, gs. no. {_sid_}')
                _def_ = smp.gs[_sid_].char_morphology_of_grains
                _def_(label_str_order=label_str_order,
                      make_pvgrid=False,
                      find_neigh=[True, ordern],
                      find_grain_voxel_locs=True,
                      find_spatial_bounds_of_grains=True,
                      find_grain_locations=True,
                      force_compute=True,)
        # ====================================================================
        if set_mprops_smp:
            print(50*'#',
                  '\n Calculating morpho props of sample gs database.',
                  '\n', 10*'. ')
            ngs = len(sid)
            for sid_count, _sid_ in enumerate(sid):
                print(f'Target gsid: {sid_count} of {sid.size}')
                # tgt.gs[i].find_spatial_bounds_of_grains()
                gs = tgt.gs[_sid_]
                _msg_ = f'tslice {sid_count} of {ngs}, gs. no. {_sid_}'
                gs.set_mprops(volnv=mpflags['volnv'],
                              eqdia=mpflags['eqdia'],
                              eqdia_base_size_spec='volnv',
                              arbbox=mpflags['arbbox'],
                              arbbox_fmt='gid_dict',
                              arellfit=mpflags['arellfit'],
                              arellfit_metric='max',
                              arellfit_calculate_efits=True,
                              arellfit_efit_routine=1,
                              arellfit_efit_regularize_data=True,
                              solidity=mpflags['sol'],
                              sol_nan_treatment='replace',
                              sol_inf_treatment='replace',
                              sol_nan_replacement=-1,
                              sol_inf_replacement=-1,
                              sanv=mpflags['sanv'],
                              sanv_N=26,
                              rat_sanv_volnv=mpflags['rat_sanv_volnv'],
                              sanv_verbosity=1E2,
                              disp_msg=_msg_)
            print(50*'#')
        # ====================================================================
        return cls(gstype_tgt=gstype_tgt, gstype_smp=gstype_smp,
                   upxogs_tgt=tgt, upxogs_smp=smp,
                   tgset=tgset, sgset=sgset,
                   ordern=ordern,
                   tsid_source=tsid_source, ssid_source=ssid_source,
                   tid=tid, sid=sid,
                   _cim_=_cim_)

    @property
    def creation_method(self):
        return self._cim_

    def init_subdef_set_dim(self):
        if self.gstype.tgt and isinstance(self.gstype.tgt, str):
            if '2' in self.gstype.tgt:
                self.dim.tgt = 2
            elif '3' in self.gstype.tgt:
                self.dim.tgt = 3
            else:
                self.dim.tgt = 2.01  # Assumed to be 2D.
        else:
            self.dim.tgt = 2.01  # Assumed to be 2D.
        if self.gstype.smp and isinstance(self.gstype.smp, str):
            if '2' in self.gstype.smp:
                self.dim.smp = 2
            elif '3' in self.gstype.smp:
                self.dim.smp = 3
            else:
                self.dim.smp = 2.01  # Assumed to be 2D.
        else:
            self.dim.smp = 2.01  # Assumed to be 2D.

    def init_subdef_set_gsid(self, data):
        print('Setting grain structure IDs.')
        if data['tsid_source'] == 'from_gs':
            from_gs, from_k, from_neigh = True, False, False
        elif data['tsid_source'] == 'from_k':
            from_gs, from_k, from_neigh = False, True, False
        elif data['tsid_source'] == 'from_neigh':
            from_gs, from_k, from_neigh = False, False, True
        elif data['tsid_source'] == 'from_gsgen':
            from_gs, from_k, from_neigh = True, False, False
        elif data['tsid_source'] == 'user':
            from_gs, from_k, from_neigh = False, False, False

        if data['tsid_source'] in ('from_gs', 'from_k',
                                   'from_neigh', 'from_gsgen'):
            self.set_tid(from_gs=from_gs, from_k=from_k, from_neigh=from_neigh,
                         tid=data['tid'])
        elif data['tsid_source'] == 'user':
            self.tid = data['tid']

        if data['ssid_source'] in ('from_gs', 'from_k',
                                   'from_neigh', 'from_gsgen'):
            self.set_sid(from_gs=from_gs, from_k=from_k, from_neigh=from_neigh,
                         sid=data['sid'])
        elif data['ssid_source'] == 'user':
            self.sid = data['sid']

        self.ntid, self.nsid = len(self.tid), len(self.sid)

    def init_subdef_set_neighs(self, data):
        self.set_ordern(data['ordern'])
        self.find_neigh_order_n(saa=True, throw=False)

    def init_subdef_set_networks(self):
        self.create_tgt_smp_networks(saa=True, throw=False)

    def init_subdef_set_prop_flags(self):
        if 3 not in (self.dim.tgt, self.dim.smp):
            '''
            This means that eiythewr:
                1. target and sample grain strucures are given to be 2d, or
                2. target or sample grain strucvtuer is assumed to be 2d.
            '''
            self.set_mprop2d_flags()
        elif self.dim.tgt == self.dim.smp == 3:
            '''
            This means that bth target and samnple grain strucruers are 3D.
            '''
            self.set_mprop3d_flags()

    def set_ordern(self, ordern):
        """
        Set the n values in O(n).

        Parametyers
        -----------
        ordern: list
            O(n) values

        Return
        ------
        None
        """
        if type(ordern) in NUMBERS:
            ordern = [abs(ordern)]
        elif type(ordern) in ITERABLES:
            if dth.ALL_NUM(ordern):
                ordern = [abs(on) for on in ordern]
            else:
                raise ValueError('Invalid datatype / datatye combinations.')
        self.ordern = ordern

    def set_tid(self, from_gs=False, from_k=False, from_neigh=False, tid=None):
        if from_gs and not from_k and not from_neigh:
            self.tid = list(self.tgset.keys())
        elif not from_gs and from_k and not from_neigh:
            self.tid = list(self.tkset.keys())
        elif not from_gs and not from_k and from_neigh:
            self.tid = list(self.tnset.keys())
        else:
            self.tid = tid

    def set_sid(self, from_gs=False, from_k=False, from_neigh=False, sid=None):
        if from_gs and not from_k and not from_neigh:
            self.sid = list(self.sgset.keys())
        elif not from_gs and from_k and not from_neigh:
            self.sid = list(self.skset.keys())
        elif not from_gs and not from_k and from_neigh:
            self.sid = list(self.snset.keys())
        else:
            self.sid = sid

    def set_mprop3d_flags(self, volnv=True, volsr=False, volch=False,
                          sanv=True, savi=False, sasr=False,
                          psa=False,
                          pernv=False, pervl=False, pergl=False,
                          eqdia=False, feqdia=False,
                          kx=False, ky=False, kz=False, ksr=False,
                          arbbox=True, arellfit=False,
                          sol=False, ecc=False, com=False, sph=False,
                          fn=False, rnd=False, mi=False, fdim=False,
                          rat_sanv_volnv=True, iputs='flags',
                          mpnames=('volnv',)):
        """
        Set flags for operational 3D morpholohgical properties.

        Parameters
        ----------
        volnv : bool, optional
            Volume by number of voxels, default True
        volsr : bool, optional
            Volume after gb surf reconstruction, default False
        volch : bool, optional
            Volume of convex hull, default False
        sanv : bool, optional
            Surface area by number of voxels, default True
        savi : bool, optional
            Surface area by voxel interfaces, default False
        sasr : bool, optional
            Surface area after gb surf reconstruction, default False
        psa : bool, optional
            Projected surface area, default False
        pernv : bool, optional
            Perimeter by number of voxels, default False
        pervl : bool, optional
            Perimeter by voxel edge lines, default False
        pergl : bool, optional
            Perimeter by geom. gb line segments, default False
        eqdia : bool, optional
            Equivalent diameter, default False
        feqdia : bool, optional
            Feret equivalent diameter, default False
        kx : bool, optional
            gb voxel local curvature in yz plane, default False
        ky : bool, optional
            gb voxel local curvature in xz plane, default False
        kz : bool, optional
            gb voxel local curvature in xy plane, default False
        ksr : bool, optional
            k computed from surf reconstruction, default False
        arbbox : bool, optional
            Aspect ratio by bounding box, default True
        arellfit : bool, optional
            Aspect ratio by ellipsoidal fit, default False
        sol : bool, optional
            Solidity of the grains, default False
        ecc : bool, optional
            Eccentricity of the grains, default False
        com : bool, optional
            Compactness of the grains, default False
        sph : bool, optional
            Sphericity of the grains, default False
        fn : bool, optional
            Flatness of the grains, default False
        rnd : bool, optional
            Roundness of the grains, default False
        mi : bool, optional
            Moment of inertia tensor, default False
        fdim : bool, optional
            Fractal dimension, default False
        rat_sanv_volnv : bool, optional
            Ratio of sanv to volnv, default True

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Invalid mpnames if type(mpnames) not in dth.dt.ITERABLES or if
            len(mpnames) > 0. Only under iputs being input as 'mp_names'.
        """
        if iputs not in ('flags', 'mp_names'):
            raise ValueError()
        # Validations
        if iputs == 'flags':
            self.mprop3d_flags = {'volnv': volnv,
                                  'volsr': volsr,
                                  'volch': volch,
                                  'sanv': sanv,
                                  'savi': savi,
                                  'sasr': sasr,
                                  'psa': psa,
                                  'pernv': pernv,
                                  'pervl': pervl,
                                  'pergl': pergl,
                                  'eqdia': eqdia,
                                  'feqdia': feqdia,
                                  'kx': kx,
                                  'ky': ky,
                                  'kz': kz,
                                  'ksr': ksr,
                                  'arbbox': arbbox,
                                  'arellfit': arellfit,
                                  'sol': sol,
                                  'ecc': ecc,
                                  'com': com,
                                  'sph': sph,
                                  'fn': fn,
                                  'rnd': rnd,
                                  'mi': mi,
                                  'fdim': fdim,
                                  'rat_sanv_volnv': rat_sanv_volnv,}
        elif iputs == 'mp_names':
            if type(mpnames) in dth.dt.ITERABLES and len(mpnames) > 0:
                self.mprop3d_flags = {mpn: True for mpn in mpnames}
            else:
                raise ValueError('Invalid mpnames.')

            self.mprop3d_flags.update({key: False
                                       for key in self.mp_gspn_map.keys()
                                       if key not in mpnames})

    def set_prop_flag(self, propname, propflagvalue):
        if not isinstance(propflagvalue, bool):
            raise TypeError(f'Invalid propflagvalue (={propflagvalue}) type. ',
                             'Must be bool.')
        if not isinstance(propname, str):
            raise TypeError(f'Invalid propname (={propname}) type. ',
                             'Must be str.')
        if propname in self.mprop2d_flags.keys():
            self.mprop2d_flags[propname] = propflagvalue
        elif propname in self.mprop3d_flags.keys():
            self.mprop3d_flags[propname] = propflagvalue
        elif propname in self.sprop2d_flags.keys():
            self.sprop2d_flags[propname] = propflagvalue
        elif propname in self.sprop3d_flags.keys():
            self.sprop3d_flags[propname] = propflagvalue
        else:
            raise ValueError(f'Invalid property name: {propname}')

    def calculate_mprop3d(self,
                          print_msg_tors=True, print_msg_prnm=True,
                          print_msg_no=True, print_msg_gsid=False,
                          print_msg_gid=False
                          ):
        """
        Data structure
        --------------
        kr.mprop3d: dict
            kr.mprop3d[tors]: dict
                kr.mprop3d[tors][prnm]: dict
                    kr.mprop3d[tors][prnm][no]: dict
                        kr.mprop3d[tors][prnm][no][gsid]: dict
                            kr.mprop3d[tors][prnm][no][gsid][gid]: np.array
                                kr.mprop3d[tors][prnm][no][gsid][gid][i]: float
        Where,
            mprop3d: 3d morphology properties
            tors: either 'tgt' or 'smp'
            prnm: property name
            no: neighbour order
            gsid: grain structure ID
            gid: grain ID
            i: prnm Property value of ith neighbour of gid grain of tors gid
                for O(n) = on.

        Data access
        -----------
        kr.mprop3d['tgt']['area_pix'][O(n)][gsid][GID]. This contains a
            list of gids which are O(n) neighbours of GID grain.
        Example:
            gid = 2
            kr.mprop3d['tgt']['area_pix'][1.25][8][gid]
            The correspionding neighbour data is:
            kr.tnset[1.25][8][gid]
        Note
        ----
        len(kr.tnset[1.25][8][gid]) = kr.mprop3d['tgt']['area_pix'][1.25][8][gid].size

        @ Dev: Variables
        ----------------
        mpflags: local copy of morpho prop flag.
        reqprop: keys in mpflags with True values.
        tors: target or sample: self.mprop3d keys.
        prnm: property name in the list of values in reqprop.
        no: neighbour order in list kr.ordern.
        gsid: Grain structue ID in self.tid
        gid: Grain IDs in local neighbour network.

        Author: Dr. Sunil Anandatheertha
        """
        print(40*'#')
        mpflags = self.mprop3d_flags
        reqprop = [prnm for prnm in mpflags.keys() if mpflags[prnm]]
        # -----------------------------------------------
        if not reqprop:
            self.mprop3d['tgt'] = 'no prop names defined !!'
            self.mprop3d['smp'] = 'no prop names defined !!'
            print('No properties calculated as no prop names querried.')
            return
        # -----------------------------------------------
        # ngs = len(list(self.tgset.keys()))
        ngs = self.tid.size
        for i, gs_i in enumerate(self.tid):
            gs = self.tgset[gs_i]
            gs.set_mprops(volnv=self.mprop3d_flags['volnv'],
                          eqdia=self.mprop3d_flags['eqdia'],
                          eqdia_base_size_spec='volnv',
                          arbbox=self.mprop3d_flags['arbbox'],
                          arbbox_fmt='gid_dict',
                          arellfit=self.mprop3d_flags['arellfit'],
                          arellfit_metric='max',
                          arellfit_calculate_efits=True,
                          arellfit_efit_routine=1,
                          arellfit_efit_regularize_data=True,
                          solidity=self.mprop3d_flags['sol'],
                          sol_nan_treatment='replace',
                          sol_inf_treatment='replace',
                          sol_nan_replacement=-1,
                          sol_inf_replacement=-1,
                          sanv=self.mprop3d_flags['sanv'],
                          sanv_N=26,
                          rat_sanv_volnv=self.mprop3d_flags['rat_sanv_volnv'],
                          sanv_verbosity=1E2,
                          disp_msg=f'tslice {i} of {ngs}, gs. no. {gs_i}')
        ngs = self.sid.size
        for i, gs_i in enumerate(self.sid):
            gs = self.sgset[gs_i]
            gs.set_mprops(volnv=self.mprop3d_flags['volnv'],
                          eqdia=self.mprop3d_flags['eqdia'],
                          eqdia_base_size_spec='volnv',
                          arbbox=self.mprop3d_flags['arbbox'],
                          arbbox_fmt='gid_dict',
                          arellfit=self.mprop3d_flags['arellfit'],
                          arellfit_metric='max',
                          arellfit_calculate_efits=True,
                          arellfit_efit_routine=1,
                          arellfit_efit_regularize_data=True,
                          solidity=self.mprop3d_flags['sol'],
                          sol_nan_treatment='replace',
                          sol_inf_treatment='replace',
                          sol_nan_replacement=-1,
                          sol_inf_replacement=-1,
                          sanv_N=26,
                          rat_sanv_volnv=self.mprop3d_flags['rat_sanv_volnv'],
                          sanv_verbosity=1E2,
                          disp_msg=f'tslice {i} of {ngs}, gs. no. {gs_i}')
        # -----------------------------------------------
        self.mprop3d = {'tgt': None, 'smp': None}
        for tors in self.mprop3d.keys():
            if print_msg_tors:
                print(f'Building tors grain-netork-propety map data for {tors}')
            self.mprop3d[tors] = {}
            for prnm in reqprop:
                if print_msg_prnm:
                    print(f'Building tors grain-netork-propety map data for property: {prnm}')
                if print_msg_no:
                    print(40*'-')
                kprop_on_level = {}
                for no in self.ordern:
                    kprop_gsid_level = {}
                    if print_msg_no:
                        print(f'.... {tors}: O(n): {no}.')
                    GSID = self.tid if tors == 'tgt' else self.sid if tors == 'smp' else None
                    for igsid, gsid in enumerate(GSID):
                        if print_msg_gsid:
                            if igsid % 5 == 0:
                                print(f'.... GSID n.: {igsid}/{len(self.tid)}.')
                        mapname = self.mp_gspn_map[prnm]
                        mprops = np.array(list(self.tgset[gsid].mprop[mapname].values()))
                        kprop_gid_level = {}
                        for gid in self.tnset[no][gsid].keys():
                            ''' ngids: neighbour grain ids. '''
                            if print_msg_gid:
                                print(f'........ gid: {gid}')
                            ngids = np.array(self.tnset[no][gsid][gid])-1
                            kprop_gid_level[gid] = mprops[ngids]
                        kprop_gsid_level[gsid] = kprop_gid_level
                    kprop_on_level[no] = kprop_gsid_level
                self.mprop3d[tors][prnm] = kprop_on_level

    def estimate_upper_ordern_bycount(self, tors='tgt', gsid=1, on_start=1.0,
                                      on_max=10.0, on_incr=0.5,
                                      neigh_count_vf_max=0.8,
                                      include_parent=True,
                                      kdeplot=True,
                                      kdeplot_kwargs={'figsize': (5, 5),
                                                     'dpi': 120,
                                                     'fill': True,
                                                     'cmap': 'cividis',
                                                     'fs_xlabel': 12,
                                                     'fs_ylabel': 12,
                                                     'fs_legend': 10,
                                                     'fs_xticks': 10,
                                                     'fs_yticks': 10,
                                                     'legend_ncols': 2,
                                                     'legend_loc': 'best'
                                                     },
                                      statplot=True,
                                      statplot_kwargs={'stat': 'mean',
                                                       'figsize': (5, 5),
                                                       'dpi': 120},
                                      gsplot=True,
                                      gsplot_kwargs={'figsize': (5, 5),
                                                     'dpi': 120},
                                      ):
        """
        Estimate O(n) needed to reach neigh_count_vf_max.

        Parameters
        ----------
        tors: str
            Specify 'tgt' for Target and 'smp' for Sample. Defaults to 'tgt'.
        gsid: int
            Grain Structure ID. Defaults to 1.
        on_start: float
            Minimum O(n) value to start iterations from. on_start >= 1.
            Defaults to 1.0.
        on_max: float
            Maximum O(n) value to end iterating. on_max >= on_start. Defaults
            to 10.0.
        on_incr: float
            del(O(n)) increments to o(n) search space. on_incr >= 0.1. Defaults
            to 0.5.
        neigh_count_vf_max: float
            neigh_count_vf value to stop iterating.
            0.11 < neigh_count_vf_max < 0.99, generally, although value may
            change depending on grain structure. Note: these bounds are not
            accurate. Defaults to 0.8.
        include_parent: bool
            Include gid in the neigh list of gid if True, else exclude.
            Defaults to True.
        plot_kde: bool
            Plot kdes of a list containing total number of neighbours of
            every gid in the grain structure for each O(n). Defaults to True.

        Returns
        -------
        LON: float
            Limiting Order-n
        neighn_stats: dict
            keys: on of every iteration.
            value: dict
                (key, value):
                    'mean' neighn.min()
                    'min': neighn.min()
                    'max': neighn.max()
                    'std': neighn.std()
                    'var': neighn.var()
                    'iqr': stats.iqr(neighn): Inter-quartile range
                    'sem': stats.sem(neighn): Standard Error of the Mean
                Where,
                    neighn = np.array([len(neighs) for neighs in ngh.values()])
                    ngh: dict: {gid: gid neighbours list}
        Ng: int
            Number of grains in the provided grain structure.

        Explanations
        ------------
        As O(n) increases the number of order-n neighbours (N) for a gid
        increases. But, it cannot increase for ever. Its maximum value is
        the total number of grains in the grain structure. The ratio of N to
        total number of grains (i.e. neigh_count_vf) is then unity. However,
        for o(n) < O(n), neigh_count_vf < 1. This function helps determine
        o(n) for which neigh_count_vf < neigh_count_vf_max.

        The kde if plotted, will show the following trends:
            * Shift right as o(n) increases during iterations.
            * Peak drops initially as o(n) increases and as width increases.
            * Peak increase again as o(n) increases further and width decreases.
            *
        """
        '''
        tors='tgt'
        gsid=1
        on_start=1
        on_max=10
        on_incr=0.2
        neigh_count_vf_max=0.9
        include_parent=True
        '''
        # Validations.
        if tors == 'tgt':
            gs, GIDs = self.tgset[gsid], self.tgset[gsid].gid
        elif tors == 'smp':
            gs, GIDs = self.sgset[gsid], self.sgset[gsid].gid
        # -------------------------------------
        neighn_stats, neighn_factor, on, kde_plot_i = {}, 0.0, on_start, 0
        neighn_values = {}
        while neighn_factor < neigh_count_vf_max and on < on_max:
            print(f'O(n)={on}')
            # SOME CALCULATIONS
            ngh = self._find_neigh_order_n_(gs, ordern=on,
                                            include_parent=include_parent,
                                            output_type='list',
                                            print_msg=False)
            neighn = np.array([len(neighs) for neighs in ngh.values()])
            neighn_values[on] = neighn
            neighn_stats[on] = {'distribution': neighn,
                                'count': neighn.size,
                                'mean': neighn.mean(),
                                'min': neighn.min(),
                                'max': neighn.max(),
                                'std': neighn.std(),
                                'var': neighn.var(),
                                'iqr': stats.iqr(neighn),
                                'sem': stats.sem(neighn)}
            on += on_incr
            neighn_factor = neighn.mean()/len(GIDs)
        else:
            if neighn_factor >= neigh_count_vf_max:
                print(f'O(n) max found for neigh_count_vf_max: {neigh_count_vf_max}')
                print(f'on: {on-on_incr}. neighn_factor: {neighn_factor}')
            elif on >= on_max:
                print(f'O(n) max found for the user set, on_max criteria: {neigh_count_vf_max}')
                print(f'on: {on-on_incr}. neighn_factor: {neighn_factor}')
            LON = np.round(on, 4)
            Ng = len(self.tgset[gsid].gid)
            '''
            Following to wrap up kdeplot after all iterations have completed:
            '''
            if kdeplot:
                plt.figure(figsize=kdeplot_kwargs['figsize'],
                           dpi=kdeplot_kwargs['dpi'])
                cmap = cm.get_cmap(kdeplot_kwargs['cmap'])
                i, _neighn_max_ = 1, []
                for _on_, neighn in neighn_values.items():
                    color = cmap(i / len(neighn_values.keys()))
                    sns.kdeplot(neighn,
                                color=color,
                                fill=kdeplot_kwargs['fill'],
                                label=f'O(n): {_on_}')
                    _neighn_max_.append(max(neighn))
                    i += 1
                plt.xlabel('GID neighbour counts for O(n)',
                           fontsize=kdeplot_kwargs['fs_xlabel'])
                plt.ylabel('KDE density',
                           fontsize=kdeplot_kwargs['fs_ylabel'])
                plt.legend(fontsize=kdeplot_kwargs['fs_legend'],
                           ncols=kdeplot_kwargs['legend_ncols'],
                           loc=kdeplot_kwargs['legend_loc'])
                plt.axvline(x=max(_neighn_max_), color='gray',
                            linestyle='dashed', linewidth=0.5)
                plt.text(max(_neighn_max_)*1.02, 0.01, f'Ng: {Ng}',
                         rotation=90)
        # -------------------------------------------------
        if statplot:
            x = np.array(list(neighn_stats.keys()))
            y = np.array([neighn_stats[no]['mean'] for no in neighn_stats.keys()])/Ng
            neighn_std = np.array([neighn_stats[no]['std'] for no in neighn_stats.keys()])/Ng

            plt.figure(figsize=statplot_kwargs['figsize'],
                       dpi=statplot_kwargs['dpi'])
            if statplot_kwargs['stat'] == 'mean':
                plt.fill_between(x, y-neighn_std, y+neighn_std,
                                 color='cyan', alpha=0.5, interpolate=True)
                plt.plot([x[0], x[-1]], [1, 1], '--k', lw=1)
                plt.errorbar(x, y, yerr=neighn_std, color='k', ecolor='b', lw=1)
                plt.xlabel('Neighbour order, O(n)', fontsize=12)
                plt.ylabel("N' = No. of neigh. grains / Ng", fontsize=12)
                plt.text(x[0], 0.9, f'Ng: {Ng}',
                         bbox=dict(boxstyle="square", ec='black',
                                   fc='cyan', alpha=0.25),
                         fontsize=12)
        if gsplot:
            gs.plotgs(figsize=gsplot_kwargs['figsize'],
                      dpi=gsplot_kwargs['dpi'])
        # -------------------------------------------------
        return LON, neighn_stats, Ng
        # -------------------------------------

    def _find_neigh_order_n_(self, gs, ordern=[1],
                             include_parent=True,
                             output_type='nparray',
                             print_msg=False):
        # non = gs.get_upto_nth_order_neighbors_all_grains(ordern,
        #                                                  include_parent=True,
        #                                                  output_type='nparray')
        non = gs.get_upto_nth_order_neighbors_all_grains_prob(ordern,
                                                              recalculate=False,
                                                              include_parent=True,
                                                              print_msg=False)
        return non

    def find_neigh_order_n(self, saa=True, throw=False):
        # Validation
        ngh = self._find_neigh_order_n_
        # --------------------------------------
        print('Starting to extract neighbourhood data for target gs dataset')
        tnset = {on: {i: None for i in self.tid} for on in self.ordern}
        for on in self.ordern:
            for i in self.tid:
                if i % 10 == 0:
                    print(f'     O(n): {on}, gsID: {i}/{self.ntid}')
                tnset[on][i] = ngh(self.tgset[i], on)
        #tnset = {on: {i: ngh(self.tgset[i], on)
        #              for i in self.tid} for on in self.ordern}
        # --------------------------------------
        print('Starting to extract neighbourhood data for sample gs dataset')
        snset = {on: {i: None for i in self.sid} for on in self.ordern}
        for on in self.ordern:
            for i in self.sid:
                if i % 10 == 0:
                    print(f'     O(n): {on}, gsID: {i}/{self.nsid}')
                snset[on][i] = ngh(self.sgset[i], on)
        #snset = {on: {i: ngh(self.sgset[i], on)
        #              for i in self.sid} for on in self.ordern}
        # --------------------------------------
        if saa:
            self.tnset, self.snset = tnset, snset
        if throw:
            return tnset, snset

    def create_gid_network(self, dataid='tgt', neigh_order=1, gsid=1):
        """
        Create the network nx graph from the neighbours dictionary.

        Parameters
        ----------
        dataid: str. Options: 'tgt' (default), 'smp'.
        neigh_order: int. Order of the raw neighbours data-structure. Defaults
            to 1.
        gsid: int. ID of the grain structure. Defaults to 1.

        Return
        ------
        nxg: network nx graph.
        """
        if dataid == 'tgt':
            neighlist = self.tnset[neigh_order][gsid]
        elif dataid == 'smp':
            neighlist = self.snset[neigh_order][gsid]
        kgid = kmake.make_gid_net_from_neighlist(neighlist)
        return kgid

    def create_tgt_networks(self, saa=True, throw=False):
        """
        Create networkx graphs for all target gs neighbours database.

        Parameters
        ----------
        saa: bool.
            Save as attrbute of True. Defults to True.
        throw: bool.
            Return value if True. Defaults to False.

        Data structure
        --------------
        dict(no1: dict(gsid1: dict(gid1: [12, 1, 16,..]))),
             no2: dict(gsid2: dict(gid2: [16, 15, 8,..]))),...
             noi: dict(gsidj: dict(gidk: [2, 86, 95,..]))),...
             noN: dict(gsidM: dict(gidG: [20, 15, 196,..]))),... )
        Where,
            noi: an element of ordern list of size N.
            gsidj: jth grain structure's ID of a toytal of M grain structes.
            gidk: kth grain ID of all G grains.
            noi-gsidj-gidk: kth grain ID in the jth grain structure's
                neighbour network dictionary of the ith O(n) database.
        """
        print(40*'-')
        print('Creating networks for target grain structure dataset.')
        tkset = {on: {i: None for i in self.tid} for on in self.ordern}
        for on in self.ordern:
            for tid in self.tid:
                if tid % 10 == 0:
                    print(f'     O(n) = {on}, gsID: {tid}/{self.ntid}')
                tkset[on][tid] = self.create_gid_network(dataid='tgt',
                                                         neigh_order=on,
                                                         gsid=tid)
        if saa:
            self.tkset = tkset
        if throw:
            return tkset

    def create_smp_networks(self, saa=True, throw=False):
        """
        Create networkx graphs for all sample gs neighbours database.

        Parameters
        ----------
        saa: bool.
            Save as attrbute of True. Defults to True.
        throw: bool.
            Return value if True. Defaults to False.

        Data structure
        --------------
        dict(no1: dict(gsid1: dict(gid1: [12, 1, 16,..]))),
             no2: dict(gsid2: dict(gid2: [16, 15, 8,..]))),...
             noi: dict(gsidj: dict(gidk: [2, 86, 95,..]))),...
             noN: dict(gsidM: dict(gidG: [20, 15, 196,..]))),... )
        Where,
            noi: an element of ordern list of size N.
            gsidj: jth grain structure's ID of a toytal of M grain structes.
            gidk: kth grain ID of all G grains.
            noi-gsidj-gidk: kth grain ID in the jth grain structure's
                neighbour network dictionary of the ith O(n) database.
        """
        print(40*'-')
        print('Creating networks for sample grain structure dataset.')
        skset = {on: {i: None for i in self.tid} for on in self.ordern}
        for on in self.ordern:
            for sid in self.sid:
                if sid % 10 == 0:
                    print(f'     O(n) = {on}, gsID: {sid}/{self.nsid}')
                skset[on][sid] = self.create_gid_network(dataid='smp',
                                                         neigh_order=on,
                                                         gsid=sid)
        if saa:
            self.skset = skset
        if throw:
            return skset

    def create_tgt_smp_networks(self, saa=True, throw=False):
        """
        Create networkx graphs for all tgt and smp gs neighbours database.

        Parameters
        ----------
        saa: bool.
            Save as attrbute of True. Defults to True.
        throw: bool.
            Return value if True. Defaults to False.

        Data structure
        --------------
        dict(no1: dict(gsid1: dict(gid1: [12, 1, 16,..]))),
             no2: dict(gsid2: dict(gid2: [16, 15, 8,..]))),...
             noi: dict(gsidj: dict(gidk: [2, 86, 95,..]))),...
             noN: dict(gsidM: dict(gidG: [20, 15, 196,..]))),... )
        Where,
            noi: an element of ordern list of size N.
            gsidj: jth grain structure's ID of a toytal of M grain structes.
            gidk: kth grain ID of all G grains.
            noi-gsidj-gidk: kth grain ID in the jth grain structure's
                neighbour network dictionary of the ith O(n) database.
        """
        tkset = self.create_tgt_networks(saa=False, throw=True)
        skset = self.create_smp_networks(saa=False, throw=True)
        if saa:
            self.tkset, self.skset = tkset, skset
        if throw:
            return tkset, skset

    def set_rkf(self, js=False, wd=False, ksp=False, ed=False, nlsd=False,
                degcen=False, btwcen=False, clscen=False, egnvcen=False):
        """
        Set rkf field calculation flags and initiate rkf dict accordingly.

        Parameters
        ----------
        js: bool
            Jaccard similarity measure of representativeness.
            Defaults to True
        wd: bool
            Wasserstein distance measure of representativeness.
            Defaults to True
        ksp: bool
            K-S test P-value measure of representativeness.
            Defaults to False
        ed: bool
            Energy distance measure of representativeness.
            Defaults to True
        nlsd: bool
            NetLSD similarity measure of representativeness.
            Defaults to False
        degcen: bool
            Betweenness Centrality. How connected each grain is. Defaults to
            False.
        btwcen: bool
            Betweenness Centrality. How important a grain is in connecting
            others. Defaults to False.
        clscen: bool
            Closeness Centrality. How close a grain is to all other grains.
            Defaults to False.
        egnvcen: bool
            Eigenvector Centrality. How influential a grain is within the
            network. Defaults to False.

        Data structures
        ---------------
        kr.rkf[RMNAME] = {n: ZEROS for n in kr.ordern}
        Where,
            MNAME = Repr metric name in ('js', 'wd', 'ksp', 'ed', 'nlsd')
            ZEROS = np.zeros((len(kr.sid), len(kr.tid)))
        """
        self.rkf_flags = {'js': js,
                          'wd': wd,
                          'ksp': ksp,
                          'ed': ed,
                          'nlsd': nlsd,
                          'degcen': degcen,
                          'btwcen': btwcen,
                          'clscen': clscen,
                          'egnvcen': egnvcen}

        self.initiate_rk_dict(js=js, wd=wd, ksp=ksp, ed=ed, nlsd=nlsd,
                              degcen=degcen, btwcen=btwcen, clscen=clscen,
                              egnvcen=egnvcen)

    def initiate_rk_dict(self, js=False, wd=False, ksp=False, ed=False,
                         nlsd=False, degcen=False, btwcen=False,
                         clscen=False, egnvcen=False):
        """
        Initiate dictionaries to store representativeness measures.

        Parameters
        ----------
        js: bool
            Jaccard similarity measure of representativeness.
            Defaults to True
        wd: bool
            Wasserstein distance measure of representativeness.
            Defaults to True
        ksp: bool
            K-S test P-value measure of representativeness.
            Defaults to False
        ed: bool
            Energy distance measure of representativeness.
            Defaults to True
        nlsd: bool
            NetLSD similarity measure of representativeness.
            Defaults to False

        Data structures
        ---------------
        kr.rkf[RMNAME] = {n: ZEROS for n in kr.ordern}
        Where,
            MNAME = Repr metric name in ('js', 'wd', 'ksp', 'ed', 'nlsd')
            ZEROS = np.zeros((len(kr.sid), len(kr.tid)))
        """
        print(40*'-')
        print('Creating R-field data structures.')
        self.rkf = {}
        nrc = len(self.sid), len(self.tid)
        data_structure = {n: np.zeros(nrc) for n in self.ordern}
        self.rkf['js'] = DCOPY(data_structure) if js else None
        self.rkf['wd'] = DCOPY(data_structure) if wd else None
        self.rkf['ksp'] = DCOPY(data_structure) if ksp else None
        self.rkf['ed'] = DCOPY(data_structure) if ed else None
        self.rkf['nlsd'] = DCOPY(data_structure) if nlsd else None
        self.rkf['degcen'] = DCOPY(data_structure) if degcen else None
        self.rkf['btwcen'] = DCOPY(data_structure) if btwcen else None
        self.rkf['clscen'] = DCOPY(data_structure) if clscen else None
        self.rkf['egnvcen'] = DCOPY(data_structure) if egnvcen else None

    def calculate_kdeg(self, ktgt, ksmp):
        """
        Calculate the node degrees of target and sample gs O(n) networks.

        Paramerters
        -----------
        ktgt: target grain structure O(n) neighbour network graph.
        ksmp: sample grain structure O(n) neighbour network graph.

        Return
        ------
        kd_tgt: node degrees of target gs O(n) neigh network graph.
        kd_smp: node degrees of sample gs O(n) neigh network graph.

        Data structures
        ---------------
        ktgt: networkx graph for target gs's O(n) neighbour netwprk dict data.
        ksmp: networkx graph for sample gs's O(n) neighbour netwprk dict data.

        kd_tgt: list: nodal degres of ktgt
        kd_smp: list: nodal degres of ksmp

        Exzplanations
        -------------
        This def calls for calculate_kdegrees. Please refer to
        upxo.netops.kchar.calculate_kdegrees for complete documentaion.
        """
        # Validations
        kd_tgt, kd_smp = kchar.calculate_kdegrees([ktgt, ksmp])
        return kd_tgt, kd_smp

    def calculate_kdeg_equal_binning(self, ktgt, ksmp):
        """
        Calculate the node degrees of T and S gs O(n) k's and equally bin them.

        Paramerters
        -----------
        ktgt: target grain structure O(n) neighbour network graph.
        ksmp: sample grain structure O(n) neighbour network graph.

        Return
        ------
        kd_tgt: node degrees of target gs O(n) neigh network graph.
        kd_smp: node degrees of sample gs O(n) neigh network graph.

        Data structures
        ---------------
        ktgt: networkx graph for target gs's O(n) neighbour netwprk dict data.
        ksmp: networkx graph for sample gs's O(n) neighbour netwprk dict data.

        kd_tgt: list: nodal degres of ktgt
        kd_smp: list: nodal degres of ksmp

        Exzplanations
        -------------
        This def calls for calculate_kdegrees_equalbinning. Please refer to
        upxo.netops.kchar.calculate_kdegrees_equalbinning for complete
        documentaion.

        Data is binned as per global min and max in degree and the distribtuion
        is re-computed using histogram.
        """
        # Validations
        kd_tgt, kd_smp = kchar.calculate_kdegrees_equalbinning([ktgt, ksmp])
        return kd_tgt, kd_smp

    def calculate_rkf_js_pairwise(self, ktgt, ksmp):
        """
        Calculate Jaccard similarity between ktgt and ksmp.

        Parameters
        ----------
        ktgt: target grain structure O(n) neighbour network graph.
        ksmp: sample grain structure O(n) neighbour network graph.

        Return
        ------
        r: representativeness level.

        Explanations
        ------------
        Refer to calculate_rkfield_js for complete documentation.
        Location: upxo.netops.kcmp.calculate_rkfield_js

        Data structures
        ---------------
        ktgt: networkx graph for target gs's O(n) neighbour netwprk dict data.
        ksmp: networkx graph for sample gs's O(n) neighbour netwprk dict data.

        r: int between 0 and 1. Higher the value, greater is
            the representativeness.
        """
        r = kcmp.calculate_rkfield_js(ktgt, ksmp)
        return r

    def calculate_rkf_wd_pairwise(self, ktgt, ksmp, equal_bins=False):
        """
        Calculate Jaccard similarity between ktgt and ksmp.

        Parameters
        ----------
        ktgt: target grain structure O(n) neighbour network graph.
        ksmp: sample grain structure O(n) neighbour network graph.

        Return
        ------
        r: representativeness level.

        Explanations
        ------------
        Refer to calculate_rkfield_wd for complete documentation.
        Location: upxo.netops.kcmp.calculate_rkfield_wd

        Data structures
        ---------------
        ktgt: networkx graph for target gs's O(n) neighbour netwprk dict data.
        ksmp: networkx graph for sample gs's O(n) neighbour netwprk dict data.

        r: int between 0 and 1. Higher the value, greater is
            the representativeness.
        """
        # Validations
        if equal_bins:
            kd_tgt, kd_smp = self.calculate_kdeg_equal_binning(ktgt, ksmp)
        else:
            kd_tgt, kd_smp = self.calculate_kdeg(ktgt, ksmp)
        r = kcmp.calculate_rkfield_wd(kd_tgt, kd_smp)
        return r

    def calculate_rkf_ksp_pairwise(self, ktgt, ksmp, equal_bins=False):
        """
        Calculate Jaccard similarity between ktgt and ksmp.

        Parameters
        ----------
        ktgt: target grain structure O(n) neighbour network graph.
        ksmp: sample grain structure O(n) neighbour network graph.

        Return
        ------
        r: representativeness level.

        Explanations
        ------------
        Refer to calculate_rkfield_ksp for complete documentation.
        Location: upxo.netops.kcmp.calculate_rkfield_ksp

        Data structures
        ---------------
        ktgt: networkx graph for target gs's O(n) neighbour netwprk dict data.
        ksmp: networkx graph for sample gs's O(n) neighbour netwprk dict data.

        r: int between 0 and 1. Higher the value, greater is
            the representativeness.
        """
        # Validations
        if equal_bins:
            kd_tgt, kd_smp = self.calculate_kdeg_equal_binning(ktgt, ksmp)
        else:
            kd_tgt, kd_smp = self.calculate_kdeg(ktgt, ksmp)
        r = kcmp.calculate_rkfield_ksp(kd_tgt, kd_smp)
        return r

    def calculate_rkf_ed_pairwise(self, ktgt, ksmp, equal_bins=False):
        """
        Calculate Jaccard similarity between ktgt and ksmp.

        Parameters
        ----------
        ktgt: target grain structure O(n) neighbour network graph.
        ksmp: sample grain structure O(n) neighbour network graph.

        Return
        ------
        r: representativeness level.

        Explanations
        ------------
        Refer to calculate_rkfield_ed for complete documentation.
        Location: upxo.netops.kcmp.calculate_rkfield_ed

        Data structures
        ---------------
        ktgt: networkx graph for target gs's O(n) neighbour netwprk dict data.
        ksmp: networkx graph for sample gs's O(n) neighbour netwprk dict data.

        r: int between 0 and 1. Higher the value, greater is
            the representativeness.
        """
        # Validations
        if equal_bins:
            kd_tgt, kd_smp = self.calculate_kdeg_equal_binning(ktgt, ksmp)
        else:
            kd_tgt, kd_smp = self.calculate_kdeg(ktgt, ksmp)
        r = kcmp.calculate_rkfield_ed(kd_tgt, kd_smp)
        return r

    def calculate_rkf_nlsd_pairwise(self, ktgt, ksmp,
                                    timescales=np.logspace(-2, 2, 20),
                                    equal_bins=False):
        """
        Calculate Jaccard similarity between ktgt and ksmp.

        Parameters
        ----------
        ktgt: target grain structure O(n) neighbour network graph.
        ksmp: sample grain structure O(n) neighbour network graph.

        Return
        ------
        r: representativeness level.

        Explanations
        ------------
        Refer to calculate_rkfield_nlsd for complete documentation.
        Location: upxo.netops.kcmp.calculate_rkfield_nlsd

        Data structures
        ---------------
        ktgt: networkx graph for target gs's O(n) neighbour netwprk dict data.
        ksmp: networkx graph for sample gs's O(n) neighbour netwprk dict data.

        r: int between 0 and 1. Higher the value, greater is
            the representativeness.
        """
        # Validations
        if equal_bins:
            kd_tgt, kd_smp = self.calculate_kdeg_equal_binning(ktgt, ksmp)
        else:
            kd_tgt, kd_smp = self.calculate_kdeg(ktgt, ksmp)
        r = kcmp.calculate_rkfield_nlsd(kd_tgt, kd_smp,
                                        timescales=timescales)
        return r

    def calculate_rkf_js_on(self, neigh_order=1):
        """
        Parameters
        ----------
        notgt: neighbour order of interest for target
        nosmp: neighbour order of interest for sample
        """
        print(f'Calculating RKF-JS for neighbour order {neigh_order}')
        # Validations
        tkset = list(self.tkset[neigh_order].values())
        skset = list(self.skset[neigh_order].values())

        DEF_rkf_js = self.calculate_rkf_js_pairwise

        for idtgt, ktgt in enumerate(tkset):
            for idsmp, ksmp in enumerate(skset):
                r = DEF_rkf_js(ktgt, ksmp)
                self.rkf['js'][neigh_order][idsmp, idtgt] = r

    def calculate_rkf_wd_on(self, neigh_order=1, equal_bins=False):
        # Validations
        print(f'Calculating RKF-WD for neighbour order {neigh_order}')
        tkset = list(self.tkset[neigh_order].values())
        skset = list(self.skset[neigh_order].values())

        DEF_rkf_wd = self.calculate_rkf_wd_pairwise

        for idtgt, ktgt in enumerate(tkset):
            for idsmp, ksmp in enumerate(skset):
                r = DEF_rkf_wd(ktgt, ksmp, equal_bins=equal_bins)
                self.rkf['wd'][neigh_order][idsmp, idtgt] = r

    def calculate_rkf_wd_on_generalized(self,
                                        neigh_order_tgt=1,
                                        neigh_order_smp=1,
                                        equal_bins=False):
        # Validations
        print(f'Calculating RKF-JS for T-O({neigh_order_tgt})|S-O({neigh_order_smp})')
        tkset = list(self.tkset[neigh_order_tgt].values())
        skset = list(self.skset[neigh_order_smp].values())
        # ---------------------------
        DEF_rkf_wd = self.calculate_rkf_wd_pairwise
        rkf_wd = np.zeros((self.nsid, self.ntid))
        # ---------------------------
        for idtgt, ktgt in enumerate(tkset):
            for idsmp, ksmp in enumerate(skset):
                r = DEF_rkf_wd(ktgt, ksmp, equal_bins=equal_bins)
                rkf_wd[idsmp, idtgt] = r

    def calculate_rkf_ksp_on(self, neigh_order=1, equal_bins=False):
        # Validations
        print(f'Calculating RKF-KSP for neighbour order {neigh_order}')
        tkset = list(self.tkset[neigh_order].values())
        skset = list(self.skset[neigh_order].values())
        DEF_rkf_ksp = self.calculate_rkf_ksp_pairwise

        for idtgt, ktgt in enumerate(tkset):
            for idsmp, ksmp in enumerate(skset):
                r = DEF_rkf_ksp(ktgt, ksmp, equal_bins=equal_bins)
                self.rkf['ksp'][neigh_order][idsmp, idtgt] = r

    def calculate_rkf_ksp_on_generalized(self,
                                         neigh_order_tgt=1,
                                         neigh_order_smp=1,
                                         equal_bins=False):
        # Validations
        print(f'Calculating RKF-KSP for T-O({neigh_order_tgt})|S-O({neigh_order_smp})')
        tkset = list(self.tkset[neigh_order_tgt].values())
        skset = list(self.skset[neigh_order_smp].values())
        # ---------------------------
        DEF_rkf_ksp = self.calculate_rkf_ksp_pairwise
        rkf_ksp = np.zeros((self.nsid, self.ntid))
        # ---------------------------
        for idtgt, ktgt in enumerate(tkset):
            for idsmp, ksmp in enumerate(skset):
                r = DEF_rkf_ksp(ktgt, ksmp, equal_bins=equal_bins)
                rkf_ksp[idsmp, idtgt] = r

    def calculate_rkf_ed_on(self, neigh_order=1, equal_bins=False):
        # Validations
        print(f'Calculating RKF-ED for neighbour order {neigh_order}')
        tkset = list(self.tkset[neigh_order].values())
        skset = list(self.skset[neigh_order].values())

        DEF_rkf_ed = self.calculate_rkf_ed_pairwise

        for idtgt, ktgt in enumerate(tkset):
            for idsmp, ksmp in enumerate(skset):
                r = DEF_rkf_ed(ktgt, ksmp, equal_bins=equal_bins)
                self.rkf['ed'][neigh_order][idsmp, idtgt] = r

    def calculate_rkf_ed_on_generalized(self,
                                        neigh_order_tgt=1,
                                        neigh_order_smp=1,
                                        equal_bins=False):
        # Validations
        print(f'Calculating RKF-ED for T-O({neigh_order_tgt})|S-O({neigh_order_smp})')
        tkset = list(self.tkset[neigh_order_tgt].values())
        skset = list(self.skset[neigh_order_smp].values())
        # ---------------------------
        DEF_rkf_ed = self.calculate_rkf_ed_pairwise
        rkf_ed = np.zeros((self.nsid, self.ntid))
        # ---------------------------
        for idtgt, ktgt in enumerate(tkset):
            for idsmp, ksmp in enumerate(skset):
                r = DEF_rkf_ed(ktgt, ksmp, equal_bins=equal_bins)
                rkf_ed[idsmp, idtgt] = r

    def calculate_rkf_nlsd_on(self, neigh_order=1,
                              timescales=np.logspace(-2, 2, 20),
                              equal_bins=False):
        # Validations
        print(f'Calculating RKF-NLSD for neighbour order {neigh_order}')
        tkset = list(self.tkset[neigh_order].values())
        skset = list(self.skset[neigh_order].values())

        DEF_rkf_nlsd = self.calculate_rkf_nlsd_pairwise

        for idtgt, ktgt in enumerate(tkset):
            for idsmp, ksmp in enumerate(skset):
                r = DEF_rkf_nlsd(ktgt, ksmp,
                             timescales=timescales,
                             equal_bins=equal_bins)
                self.rkf['ed'][neigh_order][idsmp, idtgt] = r

    def calculate_rkf_ed_nlsd_generalized(self,
                                          neigh_order_tgt=1,
                                          neigh_order_smp=1,
                                          equal_bins=False):
        # Validations
        print(f'Calculating RKF-NLSD for T-O({neigh_order_tgt})|S-O({neigh_order_smp})')
        tkset = list(self.tkset[neigh_order_tgt].values())
        skset = list(self.skset[neigh_order_smp].values())
        # ---------------------------
        DEF_rkf_nlsd = self.calculate_rkf_nlsd_pairwise
        rkf_nlsd = np.zeros((self.nsid, self.ntid))
        # ---------------------------
        for idtgt, ktgt in enumerate(tkset):
            for idsmp, ksmp in enumerate(skset):
                r = DEF_rkf_nlsd(ktgt, ksmp, equal_bins=equal_bins)
                rkf_nlsd[idsmp, idtgt] = r

    def calculate_rkf_js(self):
        for no in self.ordern:
            self.calculate_rkf_js_on(neigh_order=no)

    def calculate_rkf_wd(self):
        for no in self.ordern:
            self.calculate_rkf_wd_on(neigh_order=no)

    def calculate_rkf_ksp(self):
        for no in self.ordern:
            self.calculate_rkf_ksp_on(neigh_order=no)

    def calculate_rkf_ed(self):
        for no in self.ordern:
            self.calculate_rkf_ed_on(neigh_order=no)

    def calculate_rkf_nlsd(self, timescales=np.logspace(-2, 2, 20),
                           equal_bins=False):
        # Validations
        for no in self.ordern:
            self.calculate_rkf_nlsd_on(neigh_order=no,
                                       timescales=timescales,
                                       equal_bins=equal_bins)

    def calculate_rkf_pairwise(self, neigh_order,
                               idtgt, idsmp,
                               prop='kdegree', printmsg=False):
        # Validations
        ktgt = self.tkset[neigh_order][self.tid[idtgt]]
        ksmp = self.skset[neigh_order][self.sid[idsmp]]

        if self.rkf_flags['js']:
            r = self.calculate_rkf_js_pairwise(ktgt, ksmp)
            self.rkf['js'][neigh_order][idsmp, idtgt] = r
            if printmsg:
                printstr1 = f'RKF-JS for O({neigh_order})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r}'
                print(printstr1+printstr2)

        if self.rkf_flags['wd']:
            r = self.calculate_rkf_wd_pairwise(ktgt, ksmp)
            self.rkf['wd'][neigh_order][idsmp, idtgt] = r
            if printmsg:
                printstr1 = f'RKF-WD for O({neigh_order})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r}'
                print(printstr1+printstr2)

        if self.rkf_flags['ksp']:
            r = self.calculate_rkf_ksp_pairwise(ktgt, ksmp)
            self.rkf['ksp'][neigh_order][idsmp, idtgt] = r
            if printmsg:
                printstr1 = f'RKF-KSP for O({neigh_order})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r}'
                print(printstr1+printstr2)

        if self.rkf_flags['ed']:
            r = self.calculate_rkf_ed_pairwise(ktgt, ksmp)
            self.rkf['ed'][neigh_order][idsmp, idtgt] = r
            if printmsg:
                printstr1 = f'RKF-ED for O({neigh_order})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r}'
                print(printstr1+printstr2)

        if self.rkf_flags['nlsd']:
            r = self.calculate_rkf_nlsd_pairwise(ktgt, ksmp)
            self.rkf['nlsd'][neigh_order][idsmp, idtgt] = r
            if printmsg:
                printstr1 = f'RKF-NLSD for O({neigh_order})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r}'
                print(printstr1+printstr2)

    def calculate_rkf_pairwise_generalized(self,
                                           neigh_order_tgt, neigh_order_smp,
                                           idtgt, idsmp,
                                           prop='kdegree', printmsg=False):
        # Validations
        ktgt = self.tkset[neigh_order_tgt][idtgt]
        ksmp = self.skset[neigh_order_smp][idsmp]
        r_js, r_wd, r_ksp, r_ed, r_nlsd = None, None, None, None, None
        if self.rkf_flags['js']:
            r_js = self.calculate_rkf_js_pairwise(ktgt, ksmp)
            if printmsg:
                printstr1 = f'RKF-JS for T-O({neigh_order_tgt})|S-O({neigh_order_smp})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r_js}'
                print(printstr1+printstr2)

        if self.rkf_flags['wd']:
            r_wd = self.calculate_rkf_wd_pairwise(ktgt, ksmp)
            if printmsg:
                printstr1 = f'RKF-WD for T-O({neigh_order_tgt})|S-O({neigh_order_smp})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r_wd}'
                print(printstr1+printstr2)

        if self.rkf_flags['ksp']:
            r_ksp = self.calculate_rkf_ksp_pairwise(ktgt, ksmp)
            if printmsg:
                printstr1 = f'RKF-KSP for T-O({neigh_order_tgt})|S-O({neigh_order_smp})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r_ksp}'
                print(printstr1+printstr2)

        if self.rkf_flags['ed']:
            r_ed = self.calculate_rkf_ed_pairwise(ktgt, ksmp)
            if printmsg:
                printstr1 = f'RKF-ED for T-O({neigh_order_tgt})|S-O({neigh_order_smp})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r_ed}'
                print(printstr1+printstr2)

        if self.rkf_flags['nlsd']:
            r_nlsd = self.calculate_rkf_nlsd_pairwise(ktgt, ksmp)
            if printmsg:
                printstr1 = f'RKF-NLSD for T-O({neigh_order_tgt})|S-O({neigh_order_smp})'
                printstr2 = f', tid={idtgt}, sid={idsmp} = {r_nlsd}'
                print(printstr1 + printstr2)

        return r_js, r_wd, r_ksp, r_ed, r_nlsd

    def calculate_rkf_no(self, neigh_order, prop='kdegree'):
        for I, idtgt in enumerate(self.tid, start=0):
            for J, idsmp in enumerate(self.sid, start=0):
                if idtgt % 5 == idsmp % 20 == 0:
                    print(f'     O(n): {neigh_order}, gsID pair: ({idtgt}-{idsmp})')
                self.calculate_rkf_pairwise(neigh_order, I, J, prop=prop)

    def calculate_rkf(self, prop='kdegree'):
        """
        Calculate the network R-Field values for entire tgt and smp database.

        Parmeters
        ---------
        prop: str
            property name. Defaults to 'kdegree'. Options include:
                * 'kdegree'
                * 'area_pixel'
                * 'volume_voxel'
                * 'gblength_pixel'
                * 'gblength_geom2'
                * 'gblength_voxel'
                * 'gblength_geom3'
                * 'gbarea_voxels'
                * 'gbarea_geom'
                * 'gbrough_r'
                * 'ntjp'

        Explanations
        ------------
        User specified boolean flags in rkf_flags dictate which R-field
        metrics would be calculated.
        """
        print('++++++++++++++++++++++++++++++++++++++')
        print(str(self.ordern), '-------------')
        print('++++++++++++++++++++++++++++++++++++++')
        for no in self.ordern:
            print(40*'-')
            print(f'Calculating R-field.')
            self.calculate_rkf_no(no, prop=prop)

    def calc_self_repr(self, order, recalc_neigh=True):
        pass

    def calculate_uncertainty_angdist(self,
                                      rkf_measure='js',
                                      neigh_orders=[1],
                                      n_bins=30,
                                      data_title='Jaccard sim. measure',
                                      throw=False,
                                      plot_ad=True):
        # Validations
        if rkf_measure in self.rkf_flags.keys():
            if self.rkf_flags[rkf_measure]:
                DATA = self.rkf[rkf_measure]
            else:
                print(f'rkf_measure: {rkf_measure} not calculated.')
                return
        else:
            print(f'Invalid rkf_measure: {rkf_measure}.')
            return
        # ---------------------------------------------
        ANG_DISTANCE = {i: {'bin_means': None,
                            'min': None,
                            'mean': None,
                            'max': None,
                            'std': None,
                            'nbins': n_bins} for i in neigh_orders}
        # ---------------------------------------------
        for no in neigh_orders:
            print(f'Calculating uncertainty measure: Ang. Dist. for {rkf_measure} at O(n): {no}')
            bin_means, DATA_approx = approximate_to_bin_means(DATA[no],
                                                              n_bins=n_bins)
            ang_dist_min = np.zeros_like(bin_means)
            ang_dist_mean = np.zeros_like(bin_means)
            ang_dist_max = np.zeros_like(bin_means)
            ang_dist_std = np.zeros_like(bin_means)

            for bm_i, bm in enumerate(bin_means):
                print(f'....U(RKF: {rkf_measure}) at O({no}): bin {bm_i}/{len(bin_means)} ')
                bm_locs = np.argwhere(DATA_approx == bm)
                bin_means_sparse = np.zeros((bm_locs.shape[0],
                                             bm_locs.shape[0]))
                ang_dist_sparse = np.zeros((bm_locs.shape[0],
                                                    bm_locs.shape[0]))
                for i in range(bm_locs.shape[0]):
                    for j in range(bm_locs.shape[0]):
                        if i > j:
                            # Only find the upper tri matrix, thats enough.
                            ang_dist_sparse[j, i] = calc_angdist(bm_locs[j],
                                                                 bm_locs[i])
                        else:
                            # Nothing left to do here.
                            pass
                # plt.imshow(ang_dist_sparse)
                ang_dist_sparse = np.unique(ang_dist_sparse)
                ang_dist_sparse_compact = ang_dist_sparse[np.nonzero(ang_dist_sparse)[0]]
                if ang_dist_sparse_compact.size == 0:
                    ang_dist_min[bm_i] = np.NaN
                    ang_dist_mean[bm_i] = np.NaN
                    ang_dist_max[bm_i] = np.NaN
                    ang_dist_std[bm_i] = np.NaN
                else:
                    ang_dist_min[bm_i] = ang_dist_sparse_compact.min()
                    ang_dist_mean[bm_i] = ang_dist_sparse_compact.mean()
                    ang_dist_max[bm_i] = ang_dist_sparse_compact.max()
                    ang_dist_std[bm_i] = ang_dist_sparse_compact.std()
            ANG_DISTANCE[no]['bin_means'] = bin_means
            ANG_DISTANCE[no]['min'] = ang_dist_min
            ANG_DISTANCE[no]['mean'] = ang_dist_mean
            ANG_DISTANCE[no]['max'] = ang_dist_max
            ANG_DISTANCE[no]['std'] = ang_dist_std
        AX = self.plot_ang_dist(ANG_DISTANCE, n_bins=n_bins,
                                neigh_orders=neigh_orders,
                                figsize=(5, 5), dpi=150,
                                data_title=data_title,
                                cmap='nipy_spectral') if plot_ad else None
        if throw:
            return ANG_DISTANCE, AX

    def plot_ang_dist(self, ANG_DISTANCE, n_bins, neigh_orders=[1],
                      figsize=(5, 5), dpi=150,
                      data_title='DATA TITLE',
                      cmap='nipy_spectral', throw_axis=True
                      ):
        plt.figure(figsize=figsize,
                   dpi=dpi,
                   constrained_layout=True)
        # Choose a colormap (e.g., 'viridis', 'plasma', 'tab20')
        cmap = cm.get_cmap(cmap)
        num_colors = len(neigh_orders)  # Number of colors needed
        legends, legend_names = [], []
        color_increment = 1.0 / (len(neigh_orders) + 1)  # Add 1 to avoid using the last color in the colormap, which is often too light
        for i, neigh_order in enumerate(neigh_orders):
            color = cmap(color_increment * (i + 1))  # Use color_increment to space out the colors
            line_1, = plt.plot(ANG_DISTANCE[neigh_order]['bin_means'][:-1],
                               ANG_DISTANCE[neigh_order]['mean'][:-1],
                               linestyle='-', color=color,
                               marker='s', markersize=5, markerfacecolor=color)
            fill_1 = plt.fill_between(ANG_DISTANCE[neigh_order]['bin_means'][:-1],
                                      ANG_DISTANCE[neigh_order]['mean'][:-1] - ANG_DISTANCE[neigh_order]['std'][:-1],
                                      ANG_DISTANCE[neigh_order]['mean'][:-1] + ANG_DISTANCE[neigh_order]['std'][:-1],
                                      color=color, alpha=0.2)
            legends.append((line_1, fill_1))
            legend_names.append(f'Neigh order, O({neigh_order})')
        plt.margins(x=0)
        plt.legend(legends, legend_names, facecolor='none', edgecolor='none', loc=1)
        ax=plt.gca()
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.6)
        ax.set_xlabel(data_title, fontsize=10)
        ax.set_ylabel('Uncertainty (Mean angular distance), @Iso-R-bins, radians', fontsize=10)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        plt.grid(True, linestyle=':', color='gray', alpha=0.2)
        plt.text(0.025, 1.525, f'No. of bins: {n_bins}', fontsize=10)
        return ax
        ax.set_xlim(0.94, 1.0)
        ax.set_xticks(np.arange(0.94, 1.0, 0.02))

    def plot_rkf(self, neigh_orders=[1], power=1, figsize=(7, 5), dpi=120,
                 xtick_incr=2, ytick_incr=2,
                 lfs=7, tfs=8,
                 cmap='nipy_spectral',
                 cbarticks=np.arange(0, 1.1, 0.1),
                 cbfs=10,
                 cbtitle='Measure of representativeness R(S|T)',
                 cbfraction=0.046, cbpad=0.04,
                 cbaspect=30, shrink=0.5,
                 cborientation='vertical',
                 flags={'rkf_js': False,
                        'rkf_wd': False,
                        'rkf_ksp': False,
                        'rkf_ed': False,
                        'rkf_nlsd': False,
                        'rkf_degcen': False,
                        'rkf_btwcen': False,
                        'rkf_clscen': False,
                        'rkf_egnvcen': False,
                        }
                 ):
        """
        Example
        -------
        import numpy as np
        from upxo.repqual.grain_network_repr_assesser import KREPR
        import matplotlib.pyplot as plt
        kr = KREPR.from_gsgen(gstype='mcgs',
        					  is_smp_same_as_tgt = False,
        					  tgt_dashboard='input_dashboard.xls',
        					  smp_dashboard='input_dashboard.xls',
        					  ordern=[1, 3, 5],
                              tsid_source='from_gs',
                              ssid_source='from_gs',
        					  tid=None, sid=None,
        					  _cim_='from_gsgen')
        kr.set_rkf(js=True, wd=True, ksp=False, ed=True, nlsd=False)
        kr.calculate_rkf()

        kr.plot_rkf(neigh_orders=[1, 3, 5], figsize=(7, 5), dpi=50,
                    xtick_incr=2, ytick_incr=2,
                    lfs=7, tfs=8,
                    cmap='nipy_spectral',
                    cbarticks=np.arange(0, 1.1, 0.1),
                    cbfs=10,
                    cbtitle='Measure of representativeness R(S|T)',
                    cbfraction=0.046, cbpad=0.04, cbaspect=15, shrink=0.4,
                    cborientation='vertical',
                    plot_rkf_js=False)
        """
        # Validations
        flag_js = self.rkf_flags['js'] and flags['rkf_js']
        flag_wd = self.rkf_flags['wd'] and flags['rkf_wd']
        flag_ksp = self.rkf_flags['ksp'] and flags['rkf_ksp']
        flag_ed = self.rkf_flags['ed'] and flags['rkf_ed']
        flag_nlsd = self.rkf_flags['nlsd'] and flags['rkf_nlsd']
        flag_degcen = self.rkf_flags['degcen'] and flags['rkf_degcen']
        flag_btwcen = self.rkf_flags['btwcen'] and flags['rkf_btwcen']
        flag_clscen = self.rkf_flags['clscen'] and flags['rkf_clscen']
        flag_egnvcen = self.rkf_flags['egnvcen'] and flags['rkf_egnvcen']

        flags = [flag_js, flag_wd, flag_ksp, flag_ed, flag_nlsd,
                 flag_degcen, flag_btwcen, flag_clscen, flag_egnvcen]

        if not any(flags):
            print('Nothing to plot')
            return
        # -------------------------------------------
        fig, ax = plt.subplots(nrows=len(neigh_orders),
                               ncols=np.argwhere(flags).size,
                               figsize=figsize,
                               dpi=dpi,
                               constrained_layout=True,
                               sharex=True,
                               sharey=True)
        # -------------------------------------------
        xticks = np.arange(0, len(self.tid), xtick_incr)
        yticks = np.arange(0, len(self.sid), ytick_incr)
        # -------------------------------------------
        if len(neigh_orders) == 1 and np.argwhere(flags).size == 1:
            single_plot = True
        else:
            single_plot = False
        # ---------------------
        if len(neigh_orders) > 1 and np.argwhere(flags).size == 1:
            col_plot = True
        else:
            col_plot = False
        # ---------------------
        if len(neigh_orders) == 1 and np.argwhere(flags).size > 1:
            row_plot = True
        else:
            row_plot = False
        # ---------------------
        if not single_plot and not col_plot and not row_plot:
            matrix_type_plot = True
        else:
            matrix_type_plot = False
        # -------------------------------------------
        R = 0
        for no in neigh_orders:
            C = 0
            if flag_js:
                # print(f'JS. no: {no}, R: {R}, C: {C}')
                if single_plot: AX = ax
                elif col_plot: AX = ax[R]
                elif row_plot: AX = ax[C]
                elif matrix_type_plot: AX = ax[R, C]
                data = np.power(self.rkf['js'][no], power)
                imh = AX.imshow(data, cmap=cmap, vmin=0, vmax=1)
                AX.set_xlabel('Target GS ID', fontsize=lfs)
                AX.set_ylabel('Sample GS ID', fontsize=lfs)
                ts = f'Jaccard sim. measure,\n O(n)={no}'
                AX.set_title(ts, fontsize=tfs)
                AX.invert_yaxis()
                AX.set_xticks(xticks)
                AX.set_yticks(yticks)
                C += 1

            if flag_wd:
                # print(f'WD. no: {no}, R: {R}, C: {C}')
                if single_plot: AX = ax
                elif col_plot: AX = ax[R]
                elif row_plot: AX = ax[C]
                elif matrix_type_plot: AX = ax[R, C]
                data = np.power(self.rkf['wd'][no], power)
                imh = AX.imshow(data, cmap=cmap, vmin=0, vmax=1)
                AX.set_xlabel('Target GS ID', fontsize=lfs)
                AX.set_ylabel('Sample GS ID', fontsize=lfs)
                ts = f'Wasserstein distance based sim.\n measure, O(n)={no}'
                AX.set_title(ts, fontsize=tfs)
                AX.invert_yaxis()
                AX.set_xticks(xticks)
                AX.set_yticks(yticks)
                C += 1

            if flag_ksp:
                # print(f'KSP. no: {no}, R: {R}, C: {C}')
                if single_plot: AX = ax
                elif col_plot: AX = ax[R]
                elif row_plot: AX = ax[C]
                elif matrix_type_plot: AX = ax[R, C]
                data = np.power(self.rkf['ksp'][no], power)
                imh = AX.imshow(data, cmap=cmap, vmin=0, vmax=1)
                AX.set_xlabel('Target GS ID', fontsize=lfs)
                AX.set_ylabel('Sample GS ID', fontsize=lfs)
                ts = ['Kolmogorov-Smirnov P-value based\n sim.'
                      f' measure, O(n)={no}. Inequal bins']
                AX.set_title(ts[0], fontsize=tfs)
                AX.invert_yaxis()
                AX.set_xticks(xticks)
                AX.set_yticks(yticks)
                C += 1

            if flag_ed:
                # print(f'ED. no: {no}, R: {R}, C: {C}')
                if single_plot: AX = ax
                elif col_plot: AX = ax[R]
                elif row_plot: AX = ax[C]
                elif matrix_type_plot: AX = ax[R, C]
                data = np.power(self.rkf['ed'][no], power)
                imh = AX.imshow(data, cmap=cmap, vmin=0, vmax=1)
                AX.set_xlabel('Target GS ID', fontsize=lfs)
                AX.set_ylabel('Sample GS ID', fontsize=lfs)
                ts = f'Energy distance based\n sim. measure, O(n)={no}'
                AX.set_title(ts, fontsize=tfs)
                AX.invert_yaxis()
                AX.set_xticks(xticks)
                AX.set_yticks(yticks)
                C += 1

            if flag_nlsd:
                # print(f'NLSD. no: {no}, R: {R}, C: {C}')
                if single_plot: AX = ax
                elif col_plot: AX = ax[R]
                elif row_plot: AX = ax[C]
                elif matrix_type_plot: AX = ax[R, C]
                data = np.power(self.rkf['nlsd'][no], power)
                imh = AX.imshow(data, cmap=cmap, vmin=0, vmax=1)
                AX.set_xlabel('Target GS ID', fontsize=lfs)
                AX.set_ylabel('Sample GS ID', fontsize=lfs)
                ts = f'NetLSD sim. measure,\n O(n)={no}'
                AX.set_title(ts, fontsize=tfs)
                AX.invert_yaxis()
                AX.set_xticks(xticks)
                AX.set_yticks(yticks)
                C += 1
            R += 1
        if not single_plot:
            AX = ax[:]
        else:
            AX = ax
        cbar = plt.colorbar(imh, ax=AX,
                            fraction=cbfraction, pad=cbpad,
                            orientation=cborientation,
                            aspect=cbaspect,
                            shrink=shrink,
                            ticks=cbarticks)
        cbar.set_label(cbtitle+f'. Power: {power}', fontsize=cbfs)
