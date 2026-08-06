"""
Geometric tessellation class. This has the following components.

    1. class geotess2d for 2D voronoi tessellation grain structures.
    2. class vtess2d, inheriting from geotess2d
    3. class regtess2d, inheriting from geotess2d
    4. class semiregtess2d, inheriting from geotess2d
    5. class demiregtess2d, inheriting from geotess2d

    geotess2d: parent class for generalized geometric tessellation

    vtess2d: 2D Voronoi tessellation

    regtess2d: Tessellation of regular polygons. Can make tessellations of
        1. Equilateral triangles
        2. SQuares
        3. REgular hexagons

    semiregtess2d: Semi-regular tessellations. Also known as Archimedan
        tessellations. Each vertex in a semiregtess2d has the same arrangement
        of polygons areound it.. Can make tessellations of
            1. triangles & Squares
            2. Triangles & Squares (but a different pattern)
            3. Hexagons & Triangles
            4. Hexagons & Triangles (but a different pattern)
            5. Hexagons & Triangles & Squares
            6. Octagons & Squares
            7. Dodecagons & Triangles
            8. Dodecagons & Squares & Hexagons

    demiregtess2d: Demi-regular tessellations.

Dependencies
------------
numpy
matplotlib
pandas
shapely

Authors
-------
Dr. Sunil Anandatheertha
vaasu.anandatheertha@ukaea.uk
sunilanandatheertha@gmail.com
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from upxo.geoEntities.point2d import Point2d as point2d
from upxo.geoEntities.mulpoint2d import MPoint2d as mulpoint2d
from upxo.pxtal.polyxtal import vtpolyxtal2d as polyxtal


class geotess2d():
    """
    Base 2D geometric / Voronoi-style tessellation grain structure.

    Parent type for VTGS-style pipelines: holds seeds, junction/vertex
    points, grain-boundary edges, grain polygons (``xtals``), neighbour
    maps, and a property table. Prefer specialised constructors on
    related VT / polyxtal classes when available; many seed and
    neighbour helpers on this base remain stubs.

    Attributes
    ----------
    bounds
        Domain bounds on x and y.
    seeds
        Seed multipoint / array for the tessellation.
    grid
        Optional underlay x/y grids.
    jp, vp
        Junction and vertex points (UPXO point entities).
    gbedges, gbseg
        Grain-boundary edges and multi-edge segments.
    xtals
        Grain polygons (typically Shapely).
    gid : list
        Grain IDs.
    neigh_gid : dict
        Neighbour grain IDs per grain.
    prop
        Morphological / other property table (e.g. DataFrame).
    info : dict
        Metadata (including whether built ``from_mcgs``).
    """

    __slots__ = ('bounds', 'seeds', 'grid',
                 'jp', 'vp',
                 'gbedges', 'gbseg',
                 'xtals',
                 'gid', 'neigh_gid',
                 'prop',
                 'info',
                 )

    def __init__(self, *, from_mcgs=False):
        """Initialise the instance."""
        self.__initiate_variables(from_mcgs=from_mcgs)

    def __initiate_variables(self, from_mcgs=False):
        """Initialise instance variables."""
        self.bounds = None
        self.seeds = None
        self.xtals = []
        self.gridpoints = None
        self.gid = []
        self.neigh_gid = {}
        self.jp = []
        self.gbedges = []
        self.gbseg = []
        self.vp = []
        # ---------------------------------
        self.info = {}
        self.info['from_mcgs'] = from_mcgs
        self.info['from_mcgs'] = ''

    def __iter__(self):
        """Iterate over xtals in vtgs2d."""
        return iter(self.xtals)

    def __len__(self):
        """Return the number of items in this instance."""
        return len(self.xtals)

    def __getitem__(self):
        """Return item at the given index or key."""
        raise NotImplementedError("__getitem__ is not yet implemented.")

    def __setitem__(self):
        """Set item at the given index or key."""
        raise NotImplementedError("__setitem__ is not yet implemented.")

    def __repr__(self):
        """Return a string representation of this instance."""
        raise NotImplementedError("__repr__ is not yet implemented.")

    def set_seed_points(self, upxo_mp2d=None):
        """Set or update seed points."""
        self.seeds = upxo_mp2d

    def make_seeds_random(self):
        """Build and return seeds random."""
        raise NotImplementedError("make_seeds_random is not yet implemented.")

    def make_seeds_pdisc(self):
        """Build and return seeds pdisc."""
        raise NotImplementedError("make_seeds_pdisc is not yet implemented.")

    def make_seeds_dart(self):
        """Build and return seeds dart."""
        raise NotImplementedError("make_seeds_dart is not yet implemented.")

    def set_seeds(self):
        """Set or update seeds."""
        raise NotImplementedError("set_seeds is not yet implemented.")

    def save(self):
        """Save."""
        raise NotImplementedError("save is not yet implemented.")

    def load(self):
        """Load."""
        raise NotImplementedError("load is not yet implemented.")

    def find_neighbours(self):
        """Find neighbours."""
        raise NotImplementedError("find_neighbours is not yet implemented.")

    def find_first_nearest_neighbours(self):
        """Find first nearest neighbours."""
        raise NotImplementedError("find_first_nearest_neighbours is not yet implemented.")

    def find_second_nearest_neighbours(self):
        """Find second nearest neighbours."""
        raise NotImplementedError("find_second_nearest_neighbours is not yet implemented.")

    def filter_grains_by_prop(self):
        """Filter grains by prop."""
        raise NotImplementedError("filter_grains_by_prop is not yet implemented.")

    def filter_grains_by_loc(self):
        """Filter grains by loc."""
        raise NotImplementedError("filter_grains_by_loc is not yet implemented.")

    def _add_vertexpoint_in_grainboundaries(self):
        """ add vertexpoint in grainboundaries."""
        raise NotImplementedError("_add_vertexpoint_in_grainboundaries is not yet implemented.")

    def divide_all_edges_in_half(self):
        """Divide all edges in half."""
        self._add_vertexpoint_in_grainboundaries()

    def move_new_vertex_point(self):
        """Move new vertex point."""
        raise NotImplementedError("move_new_vertex_point is not yet implemented.")

    def perturb_grain_boundaries(self, factor):
        """Perturb grain boundaries."""
        self.divide_all_edges_in_half()
        for gbe in self.gbedges:
            pass

    def convert_to_pixels(self):
        """Convert to pixels."""
        raise NotImplementedError("convert_to_pixels is not yet implemented.")


class geoxtal2d():
    """
    Single 2D geometric grain (crystal) within a tessellation.

    Planned companion to :class:`geotess2d` for per-grain geometry.
    Not implemented yet — constructing raises ``NotImplementedError``.
    """
    def __init__(self):
        """Initialise the instance."""
        raise NotImplementedError("__init__ is not yet implemented.")
    def __repr__(self):
        """Return a string representation of this instance."""
        raise NotImplementedError("__repr__ is not yet implemented.")



class vtgs3d():
    """
    3D Voronoi / geometric tessellation grain structure (API stub).

    Intended 3D counterpart of :class:`geotess2d` with bounds, seeds,
    grains, junction topology, and property storage. Methods are not
    implemented yet — constructing raises ``NotImplementedError``.

    Attributes (planned)
    --------------------
    bounds, seeds, grid, xtals, gid, jp, gbedges, neigh_gid, prop, info
        Same roles as the 2D tessellation base, extended to 3D.
    """
    __slots__ = ('bounds', 'xtals', 'seeds', 'grid', 'gid', 'jp', 'gbedges',
                 'neigh_gid', 'prop', 'info',
                 )

    def __init__(self):
        """Initialise the instance."""
        raise NotImplementedError("__init__ is not yet implemented.")

    def __iter__(self):
        """Return an iterator over this instance."""
        raise NotImplementedError("__iter__ is not yet implemented.")

    def __len__(self):
        """Return the number of items in this instance."""
        raise NotImplementedError("__len__ is not yet implemented.")

    def __getitem__(self):
        """Return item at the given index or key."""
        raise NotImplementedError("__getitem__ is not yet implemented.")

    def __setitem__(self):
        """Set item at the given index or key."""
        raise NotImplementedError("__setitem__ is not yet implemented.")

    def set_seeds(self):
        """Set or update seeds."""
        raise NotImplementedError("set_seeds is not yet implemented.")

    def save(self):
        """Save."""
        raise NotImplementedError("save is not yet implemented.")

    def load(self):
        """Load."""
        raise NotImplementedError("load is not yet implemented.")

    def filter_boundary_grains(self):
        """Filter boundary grains."""
        raise NotImplementedError("filter_boundary_grains is not yet implemented.")

    def filter_internal_grains(self):
        """Filter internal grains."""
        raise NotImplementedError("filter_internal_grains is not yet implemented.")


