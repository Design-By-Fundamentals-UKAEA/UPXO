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

    __slots__ = ('bounds', 'seeds', 'grid', 'gridpoints',
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
        self.grid = None
        self.gridpoints = None
        self.gid = []
        self.neigh_gid = {}
        self.jp = []
        self.gbedges = []
        self.gbseg = []
        self.vp = []
        # ---------------------------------
        self.info = {'from_mcgs': from_mcgs}
        self.prop = None

    def __iter__(self):
        """Iterate over xtals in vtgs2d."""
        return iter(self.xtals)

    def __len__(self):
        """Return the number of items in this instance."""
        return len(self.xtals)

    def __getitem__(self, key):
        """Return item at the given index or key."""
        if isinstance(key, (int, slice)):
            return self.xtals[key]
        if isinstance(key, str) and isinstance(self.prop, pd.DataFrame):
            return self.prop[key]
        if key in self.neigh_gid:
            return self.neigh_gid[key]
        raise KeyError(f"Key {key} not found in geotess2d")

    def __setitem__(self, key, value):
        """Set item at the given index or key."""
        if isinstance(key, (int, slice)):
            self.xtals[key] = value
        elif isinstance(key, str) and isinstance(self.prop, pd.DataFrame):
            self.prop[key] = value
        else:
            raise KeyError(f"Cannot set key {key} in geotess2d")

    def __repr__(self):
        """Return a string representation of this instance."""
        return f"<geotess2d: {len(self.xtals)} grains, bounds={self.bounds}>"

    def set_seed_points(self, upxo_mp2d=None):
        """Set or update seed points."""
        self.seeds = upxo_mp2d

    def make_seeds_random(self, nsp=50, bounds=None):
        """Build and return seeds random."""
        if bounds is not None:
            self.bounds = bounds
        b = self.bounds or [0.0, 1.0, 0.0, 1.0]
        xmin, xmax, ymin, ymax = b[0], b[1], b[2], b[3]
        x = np.random.uniform(xmin, xmax, nsp)
        y = np.random.uniform(ymin, ymax, nsp)
        coords = np.column_stack([x, y])
        try:
            self.seeds = mulpoint2d.from_coords(coords)
        except Exception:
            self.seeds = coords
        return self.seeds

    def make_seeds_pdisc(self, char_length=0.1, bounds=None):
        """Build and return seeds pdisc using Bridson Poisson disc sampling."""
        from upxo.statops.sampling import bridson_uniform_density
        if bounds is not None:
            self.bounds = bounds
        b = self.bounds or [0.0, 1.0, 0.0, 1.0]
        xmin, xmax, ymin, ymax = b[0], b[1], b[2], b[3]
        width = xmax - xmin
        height = ymax - ymin
        pts = bridson_uniform_density(width=width, height=height, radius=char_length)
        pts = np.asarray(pts)
        if len(pts) > 0:
            pts[:, 0] += xmin
            pts[:, 1] += ymin
        try:
            self.seeds = mulpoint2d.from_coords(pts)
        except Exception:
            self.seeds = pts
        return self.seeds

    def make_seeds_dart(self):
        """Build and return seeds dart."""
        raise NotImplementedError("make_seeds_dart is not yet implemented.")

    def set_seeds(self, seeds):
        """Set or update seeds."""
        self.seeds = seeds

    def save(self):
        """Save."""
        raise NotImplementedError("save is not yet implemented.")

    def load(self):
        """Load."""
        raise NotImplementedError("load is not yet implemented.")

    def find_neighbours(self):
        """Find neighbours for all grains in xtals using boundary topology."""
        n = len(self.xtals)
        gids = self.gid if len(self.gid) == n else list(range(n))
        self.neigh_gid = {g: [] for g in gids}
        for i in range(n):
            poly_i = self.xtals[i]
            gid_i = gids[i]
            for j in range(i + 1, n):
                poly_j = self.xtals[j]
                gid_j = gids[j]
                if poly_i.touches(poly_j) or poly_i.intersects(poly_j):
                    inter = poly_i.intersection(poly_j)
                    if not inter.is_empty and (inter.length > 1e-9 or inter.geom_type in ('LineString', 'MultiLineString')):
                        self.neigh_gid[gid_i].append(gid_j)
                        self.neigh_gid[gid_j].append(gid_i)
        return self.neigh_gid

    def find_first_nearest_neighbours(self, gid=None):
        """Find first nearest neighbours."""
        if not self.neigh_gid:
            self.find_neighbours()
        if gid is not None:
            return self.neigh_gid.get(gid, [])
        return self.neigh_gid

    def find_second_nearest_neighbours(self, gid=None):
        """Find second nearest neighbours."""
        if not self.neigh_gid:
            self.find_neighbours()

        def _2nd(target_id):
            first = set(self.neigh_gid.get(target_id, []))
            second = set()
            for n1 in first:
                for n2 in self.neigh_gid.get(n1, []):
                    if n2 != target_id and n2 not in first:
                        second.add(n2)
            return sorted(list(second))

        if gid is not None:
            return _2nd(gid)
        return {g: _2nd(g) for g in self.neigh_gid}

    def filter_boundary_grains(self):
        """Return grain IDs that touch or intersect the domain boundary."""
        if not self.bounds or not self.xtals:
            return []
        from shapely.geometry import box
        xmin, xmax, ymin, ymax = self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3]
        domain_box = box(xmin, ymin, xmax, ymax)
        domain_boundary = domain_box.boundary
        gids = self.gid if len(self.gid) == len(self.xtals) else list(range(len(self.xtals)))
        boundary_grains = []
        for gid, poly in zip(gids, self.xtals):
            if poly.intersects(domain_boundary):
                boundary_grains.append(gid)
        return boundary_grains

    def filter_internal_grains(self):
        """Return grain IDs that do not touch the domain boundary."""
        b_grains = set(self.filter_boundary_grains())
        gids = self.gid if len(self.gid) == len(self.xtals) else list(range(len(self.xtals)))
        return [gid for gid in gids if gid not in b_grains]

    def filter_grains_by_loc(self, loc='internal'):
        """Filter grains by loc ('internal' or 'boundary')."""
        if loc == 'internal':
            return self.filter_internal_grains()
        elif loc == 'boundary':
            return self.filter_boundary_grains()
        else:
            raise ValueError(f"Unknown location '{loc}'; expected 'internal' or 'boundary'.")

    def filter_grains_by_prop(self, col, op, val):
        """Filter grains by property."""
        if self.prop is None or not isinstance(self.prop, pd.DataFrame):
            raise ValueError("geotess2d.prop is not a valid pandas DataFrame.")
        query_ops = {
            '>': lambda c, v: self.prop[self.prop[c] > v],
            '>=': lambda c, v: self.prop[self.prop[c] >= v],
            '<': lambda c, v: self.prop[self.prop[c] < v],
            '<=': lambda c, v: self.prop[self.prop[c] <= v],
            '==': lambda c, v: self.prop[self.prop[c] == v],
            '!=': lambda c, v: self.prop[self.prop[c] != v],
        }
        if op not in query_ops:
            raise ValueError(f"Unsupported operator '{op}'.")
        return query_ops[op](col, val)

    def _add_vertexpoint_in_grainboundaries(self):
        """ add vertexpoint in grainboundaries."""
        raise NotImplementedError("_add_vertexpoint_in_grainboundaries is not yet implemented.")

    def divide_all_edges_in_half(self):
        """Divide all edges in half."""
        self._add_vertexpoint_in_grainboundaries()

    def move_new_vertex_point(self):
        """Move new vertex point."""
        raise NotImplementedError("move_new_vertex_point is not yet implemented.")

    def perturb_grain_boundaries(self, factor=0.05, seed=None):
        """Perturb grain boundaries with controlled curvature while keeping junctions fixed."""
        from upxo.pxtal.voronoi_tessellation_2d.engine import perturb_interfaces_2d
        if not self.xtals:
            return self.xtals
        perturbed = perturb_interfaces_2d(self.xtals, bounds=self.bounds, factor=factor, seed=seed)
        self.xtals = list(perturbed.geoms) if hasattr(perturbed, 'geoms') else list(perturbed)
        return self.xtals

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


