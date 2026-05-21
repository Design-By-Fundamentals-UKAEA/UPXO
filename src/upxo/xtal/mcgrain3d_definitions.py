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
        """Initialise the instance."""
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
        """Construct this instance from mask."""
        raise NotImplementedError("from_mask is not yet implemented.")

    @classmethod
    def from_shape(cls, grid=None, ref_loc=None, shape=None, size_def=None):
        """Construct this instance from shape."""
        raise NotImplementedError("from_shape is not yet implemented.")

    @classmethod
    def from_partition(cls,
                       grain_to_partition=None,
                       ref_loc=None,
                       normal=None,
                       ):
        """Construct this instance from partition."""
        raise NotImplementedError("from_partition is not yet implemented.")

    @classmethod
    def from_surfaces(cls, slist=None, sconnectivity=None):
        """Construct this instance from surfaces."""
        raise NotImplementedError("from_surfaces is not yet implemented.")

    @classmethod
    def from_point_cloud(cls, point_cloud=None):
        """Construct this instance from point cloud."""
        raise NotImplementedError("from_point_cloud is not yet implemented.")

    @classmethod
    def from_convex_hull(cls, ch=None):
        """Construct this instance from convex hull."""
        raise NotImplementedError("from_convex_hull is not yet implemented.")

    @classmethod
    def from_vef(cls, vertices=None, edges=None, faces=None):
        """Construct this instance from vef."""
        # From vertices, edges and faces
        raise NotImplementedError("from_vef is not yet implemented.")

    @property
    def centroid(self):
        """Centroid."""
        raise NotImplementedError("centroid is not yet implemented.")

    @property
    def volume(self):
        """Volume."""
        # GRain volume
        raise NotImplementedError("volume is not yet implemented.")

    @property
    def gbsarea(self):
        """Gbsarea."""
        # Grain boundary surface area
        raise NotImplementedError("gbsarea is not yet implemented.")

    @property
    def meanori(self):
        """Meanori."""
        raise NotImplementedError("meanori is not yet implemented.")

    @property
    def bbox(self):
        """Bbox."""
        raise NotImplementedError("bbox is not yet implemented.")

    @property
    def eqd(self):
        """Eqd."""
        # Equivalent diameter
        raise NotImplementedError("eqd is not yet implemented.")

    @property
    def abc(self):
        """Abc."""
        # Ellipsoid fit axers lengths
        raise NotImplementedError("abc is not yet implemented.")

    @property
    def surface_area_to_volume_ratio(self):
        """Surface area to volume ratio."""
        # Surface area to volume fratio
        raise NotImplementedError("surface_area_to_volume_ratio is not yet implemented.")

    def extract_boundary_voxels(self):
        """Extract boundary voxels."""
        raise NotImplementedError("extract_boundary_voxels is not yet implemented.")

    def identify_boundary_surface_segments(self):
        """Identify boundary surface segments."""
        raise NotImplementedError("identify_boundary_surface_segments is not yet implemented.")

    def identify_boundary_surface_vertices(self):
        """Identify boundary surface vertices."""
        raise NotImplementedError("identify_boundary_surface_vertices is not yet implemented.")

    def identify_boundary_surface_junction_points(self):
        """Identify boundary surface junction points."""
        raise NotImplementedError("identify_boundary_surface_junction_points is not yet implemented.")

    def identify_boundary_surface_segment_edges(self):
        """Identify boundary surface segment edges."""
        raise NotImplementedError("identify_boundary_surface_segment_edges is not yet implemented.")

    def deflate_boundary_surface(self, np=1):
        """Deflate boundary surface."""
        raise NotImplementedError("deflate_boundary_surface is not yet implemented.")

    def make_boundary_zone(self, np=2):
        """Build and return boundary zone."""
        self.deflate_boundary_surface(np=np)
        # Make point cloud of boundary zone
        bz = Grain3d.from_point_cloud(point_cloud=None)

    def characterize_boundary_zone_inner_surface(self):
        """Characterize boundary zone inner surface."""
        raise NotImplementedError("characterize_boundary_zone_inner_surface is not yet implemented.")

    def distirbute_grain_boundary_precipitates(self, shape=None, size=None):
        """Distirbute grain boundary precipitates."""
        raise NotImplementedError("distirbute_grain_boundary_precipitates is not yet implemented.")

    def distribute_precipitates(self, shape=None, size=None):
        """Distribute precipitates."""
        raise NotImplementedError("distribute_precipitates is not yet implemented.")
