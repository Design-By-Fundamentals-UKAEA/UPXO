import networkx
import numpy as np
import pyvista as pv
from copy import deepcopy
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from upxo._sup import dataTypeHandlers as dth
from scipy.ndimage import label as spndimg_label

from upxo.pxtal import img_essentials_01

class IMAGE_3D:
    """
    Description of slot variables
    -----------------------------
    img: 3D Numpy array. np.int32
    fid: 0D Numpy array of list of feature IDs. np.int32
    n: Number of individual features. int
    neigh_fid: immediate neighbour feature ID database. dict
    
    vox: Information on voxels in the image. dataclass
    
    vol: Total volume
    bvox: 
    """
    __slots__ = ('base', 'img', 'vox', 'rve', 'ctrl',
                 'mops', 'tops', 'sops', 'rops',
                 'coords', 'H',
                 'fid', 'n', 'neigh_fid',                 
                 'bvox',
                 'neigh_pairs',
                 'mprop',
                 'tprop',
                 'srep'
                 )
    def __init__(self, image3d):
        self.ctrl = {}
        self.base = image3d
        # ------------------------
        self.vox = self.vox_CLS(1.0)
        self.rve = self.rve_CLS(*np.flip(self.img.shape), self.vox)
        # ------------------------
        self.ids = self.ids_CLS()
        # ------------------------
        self.coords = self.build_coords()
        # ------------------------
        self.ctrl['part'] = self.part_cntr_CLS(1)
        # ------------------------
        self.mops = self.mops_CLS(ctrl=self.ctrl['part'], fx=self.mprops_CLS())
        self.img, self.ids.c = self.mops.part(self.base)
        # ------------------------
        self.tops = self.tops_CLS(fx=self.tprops_CLS())
        self.sops = self.sops_CLS(fx=self.sprops_CLS())
        self.rops = self.rops_CLS(fx=self.rprops_CLS())
    def __repr__(self):
        return f"IMAGE_3D: shape={self.img.shape}, n_features={self.ids.c}"
    def build_coords(self):
        self.coords = np.indices((self.rve.zn, self.rve.yn,
                                  self.rve.xn)).reshape(3, -1).T
        self.ids.v = np.arange(1, self.coords.shape[0]+1)
    def part(self):
        """Partition the 3D image into below individual features.
        * closed domains (ex. grains)
        * total domain boundary (ex. all grain boundary segments)
        * domain boundary segments (ex. individual grain boundary segments)
        * domain boundary segment edges
        * domain bondarty segment edge junction points
        """
        self.fdb, N = spndimg_label(self.img, structure=self.part_cntr.bstruct)
    # =========================================================================        
    class mops_CLS:
        """Morphological operations"""
        __slots__ = ('fx', 'ctrl')
        def __init__(self, ctrl, fx):
            self.ctrl = ctrl
            self.fx = fx
        def part(self, img):
            """Partition the 3D image into below individual features.
            * closed domains (ex. grains)
            * total domain boundary (ex. all grain boundary segments)
            * domain boundary segments (ex. individual grain boundary segments)
            * domain boundary segment edges
            * domain bondarty segment edge junction points
            """
            fdb, N = spndimg_label(img, structure=self.cntr.bstruct)
            return fdb, N
        def find_CLS_mprops(self):
            pass
    class tops_CLS:
        """Topological operations
            1. handle all ops with neighbour database
        """
        __slots__ = ('DCI', 'maps', 'no', 'ntype', 'pairs',
                     'G')
        def __init__(self, def_cls_inst):
            '''def_cls_inst: Definitions class instance.'''
            self.DCI = def_cls_inst
        def find_on_neighs(self, on, prob):
            pass
        def find_onth_neighs(self, on, prob):
            pass
        def cluster(self, clustering_stats):
            pass
        def make_graph(self):
            pass
        def partition(self):
            pass
        def crop_neigh_gid(self, neigh_gid='O(1)', gids_to_crop=None):
            """
            Removes gids in gids_to_crop from neighbour order dictionary.
            Both keys and appearences in values get removed.
            Parameters
            ----------
            neigh_gid: str | dict
                Neighbour gid dictionary. If a string liuke 'O(1)' is entered,
                then the corresponding will be extracted. Idf the neighbour
                does not exist, the method should rauise an error and stop. If 
                the value is dictionary, then entered value will be used 
                without any validations. Defaults to 'O(1)'.
            gids_to_crop: dth.dt.ITERABLES | integer
                Grain ids to be cropped from neigh_gid. Value could be any in
                dth.dt.ITERABLES or of any integrer type in dth.dt.INTEGERS.
                Defaults to None.
            Example
            -------
            gstlice.crop_neigh_gid(neigh_gid='O(1)', gids_to_crop=[1])
            """
            # User input validations
            if type(neigh_gid) != dict:
                raise ValueError('neigh_gid invalid')
            if type(gids_to_crop) in dth.dt.INTEGERS:
                gids_to_crop = [gids_to_crop]
            if type(gids_to_crop) not in dth.dt.ITERABLES:
                raise ValueError('gids_to_crop invalid')
            # --------------
            if type(neigh_gid) == str:
                if neigh_gid.lower() == 'o(1)':
                    neigh_gid = self.neigh_gid
                else:
                    # Other codes for later development if needed.
                    # We can work on other order dictionaries here.
                    pass
            # Remove keys and values
            for gidcrop in gids_to_crop:
                neigh_gid.pop(gidcrop)
            # Remove gids from the values of other gids
            gids_to_crop = set(gids_to_crop)
            for gid in neigh_gid.keys():
                neigh_gid[gid] = set(neigh_gid[gid])
                neigh_gid[gid] = neigh_gid[gid] - gids_to_crop
                neigh_gid[gid] = list(neigh_gid[gid])
            return neigh_gid
        
    class sops_CLS:
        """Spatial operations"""
        __slots__ = ('fx', )
        def __init__(self, fx):
            self.fx = fx
        def find_loc(self):
            pass
    class mprops_CLS:
        """Class to find morphological propertues"""
        def __init__(self):
            pass
    class tprops_CLS:
        """Class to find topological properties"""
        def __init__(self):
            pass
        def G_max_ind_set(self, G):
            return networkx.maximal_independent_set(G)
        def G_conn_comp(self, G):
            return [set(c) for c in networkx.connected_components(G)]        
    class sprops_CLS:
        """Class to find spatial properties"""
        def __init__(self):
            pass
    class rprops_CLS:
        """Class to find representativeness properties"""
        def __init__(self):
            pass       

    class mviz_CLS:
        def __init__(self, outer):
            pass
        
    class tviz_CLS:
        def __init__(self, outer):
            pass
        
    class mpviz_CLS:
        def __init__(self):
            pass
        
    class meta_ops_CLS:
        def __init__(self):
            pass
        def subdomain(self):
            pass
        def coarse(self, k):
            pass
        def fine(self, k):
            pass
    class cell_CLS:
        __slots__ = ('type')
        def __init__(self):
            pass
    class xtal_CLS(cell_CLS):
        __slots__ = ("a")
        def __init__(self):
            pass
    class xb_CLS(cell_CLS):
        """xtal_boundary"""
        __slots__= ("a")
        def __init__(self):
            pass
    class xbseg_CLS(cell_CLS):
        """xtal_boundary segment"""
        __slots__ = ("a")
        def __init__(self):
            pass
    class xjv_CLS(cell_CLS):
        """xtal_junction_voxels"""
        __slots__ = ("a")
        def __init__(self):
            pass