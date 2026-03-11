import numpy as np
import matplotlib.pyplot as plt
# import cv2
# from skimage.measure import label as skim_label
from scipy.spatial import cKDTree as ckdt
from defdap.quat import Quat
import upxo._sup.gops as gops
import upxo._sup.dataTypeHandlers as dth
from upxo._sup.validation_values import _validation as val
from upxo.xtalphy.orientation import grainoris as go

class Grain3d():
    __slots__ = ('loc', 'npixels', 'position', 'coords', 'gbloc', 'brec',
                 'bbox_bounds', 'bbox_ex_bounds', 'bbox', 'bbox_ex', 'skprop',
                 '_px_area', 'gid', 'gind', 'gbid', 'gbind', 'gbvert',
                 'gbsegs', 'gbsegs_pre', 'gbsegs_geo', 's', 'sn', 'neigh',
                 'precipitates', 'grain_core', 'gb_zone', 'subgrains', 'paps',
                 'blocks', 'laths', 'xstruc', 'xmin', 'xmax', 'ymin', 'ymax',
                 'loctree', 'coordtree', 'control_points_mesh',
                 'ea', 'eapert', 'q', 'qref', 'earef', 'texcomp',
                 'introduce_orientation_bands', 'xgid'
                 )

    def __init__(self):
        self.loc, self.position = None, None
        self.coords, self.gbloc = None, None
        # set bounds related
        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None
        self.brec, self.bbox_bounds, self.bbox_ex_bounds = None, None, None
        self.loctree, self.coordtree = None, None
        # set masks
        self.bbox, self.bbox_ex = None, None
        # Set local neighbourhood related slots
        self.neigh = None
        # set properties
        self.npixels, self._px_area, self.skprop = None, None, None
        # set state related slots
        self.s, self.sn = None, None
        # set grain indices related slots
        self.gid, self.gind = None, None
        # Set grain boundary indices related slots
        self.gbid, self.gbind = None, None
        # Set grain boundaryt points
        self.gbvert = None
        # Set grain bounadryu segfments
        self.gbsegs, self.gbsegs_geo = None, None
        # FEATURES
        self.precipitates, self.grain_core, self.gb_zone = None, None, None
        self.subgrains = None
        self.paps, self.blocks, self.laths = None, None, None
        # MESHING RELATED DATA
        self.control_points_mesh = None
        # PHASE RELATED
        self.xstruc = 'fcc'
        # ------------------------------------------
        # ORIENTATION RELATED
        self.ea, self.q = None, None
        self.qref, self.earef = None, None
        # eapert: ((p1low, p1high), (p2low, p2high), (p3low, p3high))
        self.eapert = ((0, 2.5), (0, 2.5), (0, 2.5)),
        self.introduce_orientation_bands = None
        self.texcomp = 'unknown'

    @classmethod
    def from_mask(cls, oris=None, **kwargs):
        pass

    @classmethod
    def from_shape(cls, grid=None, ref_loc=None, shape=None, size_def=None):
        pass

    @classmethod
    def from_partition(cls,
                       grain_to_partition=None,
                       ref_loc=None,
                       normal=None,
                       ):
        pass

    @classmethod
    def from_surfaces(cls, slist=None, sconnectivity=None):
        pass

    @classmethod
    def from_point_cloud(cls, point_cloud=None):
        pass

    @classmethod
    def from_convex_hull(cls, ch=None):
        pass

    @classmethod
    def from_vef(cls, vertices=None, edges=None, faces=None):
        # From vertices, edges and faces
        pass

    @property
    def centroid(self):
        pass

    @property
    def volume(self):
        # GRain volume
        pass

    @property
    def gbsarea(self):
        # Grain boundary surface area
        pass

    @property
    def meanori(self):
        pass

    @property
    def bbox(self):
        pass

    @property
    def eqd(self):
        # Equivalent diameter
        pass

    @property
    def abc(self):
        # Ellipsoid fit axers lengths
        pass

    @property
    def surface_area_to_volume_ratio(self):
        # Surface area to volume fratio
        pass

    def extract_boundary_voxels(self):
        pass

    def identify_boundary_surface_segments(self):
        pass

    def identify_boundary_surface_vertices(self):
        pass

    def identify_boundary_surface_junction_points(self):
        pass

    def identify_boundary_surface_segment_edges(self):
        pass

    def deflate_boundary_surface(self, np=1):
        pass

    def make_boundary_zone(self, np=2):
        self.deflate_boundary_surface(np=np)
        # Make point cloud of boundary zone
        bz = Grain3d.from_point_cloud(point_cloud=None)

    def characterize_boundary_zone_inner_surface(self):
        pass

    def distirbute_grain_boundary_precipitates(self, shape=None, size=None):
        pass

    def distribute_precipitates(self, shape=None, size=None):
        pass
