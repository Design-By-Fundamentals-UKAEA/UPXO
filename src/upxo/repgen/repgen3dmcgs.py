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
from upxo.misc import make_belief
from scipy.ndimage import generate_binary_structure
from dataclasses import dataclass
from scipy.ndimage import label as spndimg_label
import upxo._sup.data_ops as DO
from upxo.viz.plot_artefacts import cuboid_data
from upxo.viz.helpers import arrange_subplots
from upxo.repqual.grain_network_repr_assesser import KREPR

warnings.simplefilter('ignore', DeprecationWarning)


class repgen3d:

    __slots__ = ('tdist', 'tstat', 'tgs', 'sgs', 
                 'iroute', 'mpflags', 'rm0tests','rm0')
    '''
    Explanation of slot variables:
    ------------------------------
    tdist: upxo distribution collection object
        Distribution data of the target grain structure.
    tstat: upxo statistics collection object
        Statistics data of the target grain structure.
    tgs: grain structure data object
        The target grain structure.
    sgs: grain structure data object
        The sample grain structure.
    iroute: str
        Route to use for the generation of the representative grain structure.
    mpflags: dict
        Control parameters for the generation and use of morphological
        properties of the target and/or sample grain structure.
    rm0tests: dict
        Specifies which r0 tests to perform.
    '''

    VALiroutes = ('tdist.sgs', 'tstat.sgs', 'tgs.sgs')
    '''
    Explanation of VALiroutes:
    -------------------------
    The valid routes for the generation of the representative grain structure.
        1. 'tdist.sgs': Use the distribution data of the target grain structure
                        and the sample grain structure.
        2. 'tstat.sgs': Use the statistics data of the target grain structure
                        and the sample grain structure.
        3. 'tgs.sgs': Use the actual target and sample grain structures.
    '''

    VALgs = ('upxo.mc2d', 'upxo.mc3d', 
             'upxo.pv2d', 'upxo.vv3d', 
             'upxo.v2d', 'upxo.v3d', 
             'image2d', 'image3d')
    '''
    Explanation of VALgs:
    --------------------------------
    The valid grain structure types for the sample grain structure.
        1. 'upxo.mc2d': 2D Monte-Carlo type.
        2. 'upxo.mc3d': 3D Monte-Carlo type.
        3. 'upxo.pv2d': 2D pixelated Voronoi type.
        4. 'upxo.vv3d': 3D voxellated Voronoi type.
        5. 'upxo.v2d': 2D Voronoi type.
        6. 'upxo.v3d': 3D Voronoi type.
        7. 'image2d': 2D image type.
        8. 'image3d': 3D image type.
    '''

    def __init__(self,
                 tdist=None, tstat=None, tgs=None,
                 sgs=None, tdim=2,
                 iroute='tgs.sgs',
                 sgstype='upxo.mc3d', tgstype='upxo.mc3d'
                 ):

        if iroute not in self.VALiroutes:
            raise ValueError('Invalid iroute')
        if sgstype not in self.VALgs:
            raise ValueError('Invalid sgstype')
        if tgstype not in self.VALgs:
            raise ValueError('Invalid tgstype')

        self.tdist = tdist
        self.tstat = tstat
        self.tgs = tgs
        self.sgs = sgs
        self.tdim = tdim
        self.iroute = iroute
        self.sgstype = sgstype
        self.tgstype = tgstype

    @classmethod
    def from_tdist_sgs(cls, tdist=None, sgs=None, tdim=2, sgstype='upxo.mc3d'):
        """
        Alternative constructor for creating a RepGen3DMCGS instance using distribution data of target grain structure and sample grain structure.

        The sample grain structure can be of the following types:
        - see description of sgstype parameter

        Parameters
        ----------
        tdist: upxo distribution collection object, optional
            Distribution data of the target grain structure. Defaults to None.
        sgs: grain structure data object, optional
            The sample grain structure. Defaults to None.
        tdim: int, optional
            Dimensionality of the target grain structure data used for tdist. Defaults to 2.
        sgstype: str
            Type of the sample grain structure. Must be in VALgs.

        Returns
        -------
        RepGen3DMCGS
            A new RepGen3DMCGS instance.
        """
        return cls(tdist=tdist, sgs=sgs, tdim=tdim,
                   iroute='tdist.sgs', sgstype='upxo.mc3d')

    @classmethod
    def from_tstat_sgs(cls, tstat=None, sgs=None, tdim=2, sgstype='upxo.mc3d'):
        """
        Alternative constructor for creating a RepGen3DMCGS instance using statistics data of target grain structure and sample grain structure.

        The sample grain structure can be of the following types:
        - see description of sgstype parameter

        Parameters
        ----------
        tstat: upxo statistics collection object, optional
            Statistics data of the target grain structure. Defaults to None.
        sgs: grain structure data object, optional
            The sample grain structure. Defaults to None.
        tdim: int, optional
            Dimensionality of the target grain structure data used for tstat. Defaults to 2.
        sgstype: str
            Type of the sample grain structure. Must be in VALgs.

        Returns
        -------
        RepGen3DMCGS
            A new RepGen3DMCGS instance.
        """
        return cls(tstat=tstat, sgs=sgs, tdim=tdim,
                   iroute='tstat.sgs', sgstype='upxo.mc3d')

    @classmethod
    def from_tgs_sgs(cls, tgs=None, sgs=None,
                     tgstype='upxo.mc3d', sgstype='upxo.mc3d'):
        """
        Alternative constructor for creating a RepGen3DMCGS instance using  actual target and sample grain structures.

        The sample grain structure can be of the following types:
        - see description of sgstype parameter

        Parameters
        ----------
        tstat: upxo statistics collection object, optional
            Statistics data of the target grain structure. Defaults to None.
        sgs: grain structure data object, optional
            The sample grain structure. Defaults to None.
        tdim: int, optional
            Dimensionality of the target grain structure data used for tstat. Defaults to 2.
        tgstype: str, optional
            Type of the target grain structure. Must be in VALgs. Defaults to 'upxo.mc3d'. 
        sgstype: str
            Type of the sample grain structure. Must be in VALgs.

        Returns
        -------
        RepGen3DMCGS
            A new RepGen3DMCGS instance.
        """
        return cls(tgs=tgs, sgs=sgs, iroute='tgs.sgs',
                   tgstype=tgstype, sgstype=sgstype)
    
    def set_mpflags(self,
                    volnv=False, volsr=False, volch=False,
                    arbbox=False, 
                    sanv=False, savi=False, sasr=False, 
                    pernv=False, pervl=False, pergl=False,
                    solidity=False, ecc=False, com=False, sph=False, fn=False, fdim=False,
                    rat_sanv_volnv=False,
                    arellfit=False,
                    eqdia=False, feqdia=False,
                    eqdia_base_size_spec='volnv',                   
                    arbbox_fmt='gid_dict', 
                    arellfit_metric='max',
                    arellfit_calculate_efits=False,
                    arellfit_efit_routine=1,
                    arellfit_efit_regularize_data=False,):
        """
        Set  parameter values to control which morphological properties are calculated and used in representativeness assessments and/or representativeness grain structure identification/generation.

        Parameters
        ----------
        volnv : bool, optional
            Volume by number of voxels. Defaults to False.
        volsr : bool, optional
            Volume after grain boundary surface reconstruction . Defaults to False.
        volch : bool, optional
            Volume of the convex hull . Defaults to False.
        arbbox : bool, optional
            Aligned bounding box . Defaults to False.
        sanv : bool, optional
            Surface area of the non-void space . Defaults to False.
        savi : bool, optional
            Surface area of the void space . Defaults to False.
        sasr : bool, optional
            Surface area of the smallest rectangular prism . Defaults to False.
        pernv : bool, optional
            perimeter by number of voxels . Defaults to False.
        pervl : bool, optional
            perimeter by voxel edge lines . Defaults to False.
        pergl : bool, optional
            perimeter by geometric grain boundary line segments . Defaults to False.
        solidity : bool, optional
            Solidity . Defaults to False.
        ecc : bool, optional
            Eccentricity . Defaults to False.
        com : bool, optional
            compactness . Defaults to False.
        sph : bool, optional
            Sphericity . Defaults to False.
        fn : bool, optional
            Flatness . Defaults to False.
        fdim : bool, optional
            Fractal dimension . Defaults to False.
        rat_sanv_volnv : bool, optional
            Ratio of sanv to volnv . Defaults to False.
        arellfit : bool, optional
            Whether to calculate the parameters of an equivalent ellipsoid . Defaults to False.
        eqdia : bool, optional
            Equivalent diameter . Defaults to False.
        feqdia : bool, optional
            Feret equivalent diameter . Defaults to False.
        eqdia_base_size_spec : str, optional
            Specification for the base size for equivalent diameter calculation,
            'volnv' (default), 'volsr', or 'volch'.
        arbbox_fmt : str, optional
            Format of the aligned bounding box, 'gid_dict' (default).
        arellfit_metric : str, optional
            Metric to use for equivalent ellipsoid fitting, 'max' (default).
        arellfit_calculate_efits : bool, optional
            Whether to calculate the efits . Defaults to False.
        arellfit_efit_routine : int, optional
            Routine to use for efit calculation (default is 1).
        arellfit_efit_regularize_data : bool, optional
            Whether to regularize data for efit calculation . Defaults to False.
        Returns
        -------
        None
        """
        self.mpflags = dict(volnv=volnv, volsr=volsr, volch=volch,
                            arbbox=arbbox, sanv=sanv, savi=savi, sasr=sasr,
                            pernv=pernv, pervl=pervl, pergl=pergl,
                            solidity=solidity, ecc=ecc, com=com, sph=sph, fn=fn, fdim=fdim,
                            rat_sanv_volnv=rat_sanv_volnv,
                            arellfit=arellfit,
                            eqdia=eqdia, feqdia=feqdia,
                            eqdia_base_size_spec=eqdia_base_size_spec,
                            arbbox_fmt=arbbox_fmt,
                            arellfit_metric=arellfit_metric,
                            arellfit_calculate_efits=arellfit_calculate_efits,
                            arellfit_efit_routine=arellfit_efit_routine,
                            arellfit_efit_regularize_data=arellfit_efit_regularize_data)

    def char_gs(self, find_spatial_bounds_of_grains=False):
        _fsb = find_spatial_bounds_of_grains
        self.sgs.char_morphology_of_grains(label_str_order=1,
                                    find_grain_voxel_locs=False,
                                    find_spatial_bounds_of_grains=_fsb,
                                    force_compute=True)
        _a = self.mpctrls['eqdia_base_size_spec']
        _b = self.mpctrls['arellfit_calculate_efits']
        _c = self.mpctrls['arellfit_efit_routine']
        _d = self.mpctrls['arellfit_efit_regularize_data']
        self.sgs.set_mprops(volnv=self.mpctrls['volnv'],
                            eqdia=self.mpctrls['eqdia'],
                            eqdia_base_size_spec=_a,
                            arbbox=self.mpctrls['arbbox'],
                            arellfit=self.mpctrls['arellfit'],
                            arbbox_fmt=self.mpctrls['arbbox_fmt'],
                            arellfit_metric=self.mpctrls['arellfit_metric'],
                            arellfit_calculate_efits=_b,
                            arellfit_efit_routine=_c,
                            arellfit_efit_regularize_data=_d,
                            solidity=self.mpctrls['solidity'],
                            sol_nan_treatment='replace',
                            sol_inf_treatment='replace',
                            sol_nan_replacement=-1,
                            sol_inf_replacement=-1)

        if 'tgs' in self.iroute:
            self.tgs.char_morphology_of_grains(label_str_order=1,
                                        find_grain_voxel_locs=False,
                                        find_spatial_bounds_of_grains=_fsb,
                                        force_compute=True)
            self.tgs.set_mprops(volnv=self.mpctrls['volnv'],
                                eqdia=self.mpctrls['eqdia'],
                                eqdia_base_size_spec=_a,
                                arbbox=self.mpctrls['arbbox'],
                                arellfit=self.mpctrls['arellfit'],
                                arbbox_fmt=self.mpctrls['arbbox_fmt'],
                                arellfit_metric=self.mpctrls['arellfit_metric'],
                                arellfit_calculate_efits=_b,
                                arellfit_efit_routine=_c,
                                arellfit_efit_regularize_data=_d,
                                solidity=self.mpctrls['solidity'],
                                sol_nan_treatment='replace',
                                sol_inf_treatment='replace',
                                sol_nan_replacement=-1,
                                sol_inf_replacement=-1)
            
    def validate_mprops(self):
        vmp = [key for key in self.mpflags.keys() if self.mpflags[key] == True]
        if len(vmp) == 0:
            raise ValueError('No morphological properties selected for validation.')
        vmp_sgs = self.sgs.validate_mprops_l0(mprops=vmp)
        if 'tgs' in self.iroute:
            vmp_tgs = self.tgs.validate_mprops_l0(mprops=vmp)

    def rm0_setup(self, kld=False, ks2=False, mhd=True):
        self.rm0tests = dict(lkd=kld, ks2=ks2, mhd=mhd)  

    def rm0_run(self):
        rm0={}
        if self.rm0tests['kld']:
            rm0['kld'] = self.sgs.rmtest_kld(self.tgs)
        if self.rm0tests['ks2']:
            rm0['ks2'] = self.sgs.rmtest_ks2(self.tgs)
        if self.rm0tests['mhd']:
            rm0['mhd'] = self.sgs.rmtest_mhd(self.tgs)

    def rm1_setup(self,
                  nslices=1,
                  avoid_bundary_slice=True,
                  ordern=[1, 3],
                  tests={'js': True, 'wd': True,
                         'ksp': True, 'ed': True,
                         'nlsd': True, 'degcen': False, 
                         'btwcen': False, 'clscen': False, 
                         'egnvcen': False},
                  ):
        if self.tdim == 2 and self.sdim == 2:
            kr = KREPR.from_gs(tgset={1: self.tgs}, 
                               sgset={1: self.sgs},
                               ordern=ordern,
                               gstype_tgt=self.tgstype,
                               gstype_smp=self.sgstype,)
            kr.calculate_mprop2d()

        kr.set_rkf(**tests)
        

        self.rm1tests = dict(ed=ed, ordern=ordern)

    def rm1_run(self):
        rm1={}
        if self.rm1tests['ed']:
            rm1['ed'] = self.sgs.rmtest_ed(self.tgs, nO=self.rm1tests['nO'])
