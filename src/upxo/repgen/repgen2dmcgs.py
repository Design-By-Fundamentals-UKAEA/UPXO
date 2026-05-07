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
import warnings
import seaborn as sns
from functools import partial
from matplotlib.figure import Figure
from skimage.segmentation import find_boundaries
from upxo.geoEntities.mulpoint2d import MPoint2d as mp2d
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
from upxo.viz.helpers import arrange_subplots
from upxo.repqual.grain_network_repr_assesser import KREPR

warnings.simplefilter('ignore', DeprecationWarning)


class repgen2d:

    __slots__ = ('tdist', 'tstat', 'tgs', 'sgs',
                 'iroute', 'mpflags', 'rm0tests', 'rm0')
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

    # Valid Grain structure types
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

    def __init__(self, tdist=None, tstat=None, tgs=None,
                 sgs=None, tdim=2, iroute='tgs.sgs',
                 sgstype='upxo.mc2d', tgstype='upxo.mc2d'):

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
    def from_tdist_sgs(cls, tdist=None, sgs=None, tdim=2, sgstype='upxo.mc2d'):
        """
        Alternative constructor for creating a RepGen2DMCGS instance using
        distribution data of target grain structure and sample grain structure.

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
        RepGen2DMCGS
            A new RepGen2DMCGS instance.
        """
        return cls(tdist=tdist, sgs=sgs, tdim=tdim,
                   iroute='tdist.sgs', sgstype='upxo.mc2d')

    @classmethod
    def from_tstat_sgs(cls, tstat=None, sgs=None, tdim=2, sgstype='upxo.mc2d'):
        """
        Alternative constructor for creating a RepGen2DMCGS instance using
        statistics data of target grain structure and sample grain structure.

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
        RepGen2DMCGS
            A new RepGen2DMCGS instance.
        """
        return cls(tstat=tstat, sgs=sgs, tdim=tdim,
                   iroute='tstat.sgs', sgstype='upxo.mc2d')

    @classmethod
    def from_tgs_sgs(cls, tgs=None, sgs=None,
                     tgstype='upxo.mc2d', sgstype='upxo.mc2d'):
        """
        Alternative constructor for creating a RepGen2DMCGS instance using
        actual target and sample grain structures.

        The sample grain structure can be of the following types:
        - see description of sgstype parameter

        Parameters
        ----------
        tgs: grain structure data object, optional
            The target grain structure. Defaults to None.
        sgs: grain structure data object, optional
            The sample grain structure. Defaults to None.
        tgstype: str, optional
            Type of the target grain structure. Must be in VALgs. Defaults to 'upxo.mc2d'.
        sgstype: str
            Type of the sample grain structure. Must be in VALgs.

        Returns
        -------
        RepGen2DMCGS
            A new RepGen2DMCGS instance.
        """
        return cls(tgs=tgs, sgs=sgs, iroute='tgs.sgs',
                   tgstype=tgstype, sgstype=sgstype)
