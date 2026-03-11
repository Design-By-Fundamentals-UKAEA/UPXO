"""
Gometric tessellation class. This has the follwing components.

    1. class geotess2d for 2D voronoi tessellation grain structures.
    2. class vtess2d, inheriting from geotess2d
    3. class regtess2d, inheriting frm geotess2d
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
vaasu.anandatheertha@ukaea.com
sunilanandatheertha@gmail.com
"""
import numpy as np
import pandas as pd
from shapely import speedups
import matplotlib.pyplot as plt
from upxo.geoEntities.point2d import point2d
from upxo.geoEntities.mulpoint2d import mulpoint2d
from upxo.pxtal.polyxtal import vtpolyxtal2d as polyxtal


class geotess2d():
    """
    Voronoi Tessellation grain structure in 3D.

    Attributes
    ----------
    bounds: x-axis and y-axis bounds
    seeds: seed points: Numpy array
    grid: x and y grid underlay: 2D Numpy arrays

    jp: Unction points:  UPXO point2d
    vp: Vertex points: UPXO point2d

    gbedges: List of all grain boundary edges: list(UPXO edge2d)
    gbseg: List of all grain bounmdary segments: list(UPXO muledge2d)

    xtals: List of grains: shapely polygons

    gid: Grain indices values: list[int]
    neigh_gid: Neighbour gid values: dict(gid: list(int))

    prop: Property holder: Pandas dataframe
    info: Meta data storage: dict
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
        self.__initiate_variables(from_mcgs=from_mcgs)

    def __initiate_variables(self, from_mcgs=False):
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
        return len(self.xtals)

    def __getitem__(self):
        pass

    def __setitem__(self):
        pass

    def __repr__(self):
        pass

    def set_seed_points(self, upxo_mp2d=None):
        self.seeds = upxo_mp2d

    def make_seeds_random(self):
        pass

    def make_seeds_pdisc(self):
        pass

    def make_seeds_dart(self):
        pass

    def set_seeds(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def find_neighbours(self):
        pass

    def find_first_nearest_neighbours(self):
        pass

    def find_second_nearest_neighbours(self):
        pass

    def filter_grains_by_prop(self):
        pass

    def filter_grains_by_loc(self):
        pass

    def _add_vertexpoint_in_grainboundaries(self):
        pass

    def divide_all_edges_in_half(self):
        self._add_vertexpoint_in_grainboundaries()

    def move_new_vertex_point(self):
        pass

    def perturb_grain_boundaries(self, factor):
        self.divide_all_edges_in_half()
        for gbe in self.gbedges:
            pass

    def convert_to_pixels(self):
        pass


class geoxtal2d():
    def __init__(self):
        pass
    def __repr__(self):
        pass



class vtgs3d():
    __slots__ = ('bounds', 'xtals', 'seeds', 'grid', 'gid', 'jp', 'gbedges',
                 'neigh_gid', 'prop', 'info',
                 )

    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass

    def __setitem__(self):
        pass

    def set_seeds(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def filter_boundary_grains(self):
        pass

    def filter_internal_grains(self):
        pass





import damask
import numpy as np
size = np.ones(3)*1e-5
cells = [100, 100, 100]
N_grains = 200
seeds = damask.seeds.from_random(size, N_grains, cells, True)
grid = damask.GeomGrid.from_Voronoi_tessellation(cells,size,seeds)
grid.save(f'Polycystal_{N_grains}_{cells[0]}x{cells[1]}x{cells[2]}')
grid


import damask
import numpy as np
size = np.ones(3)*1e-5
cells = [50, 50, 50]
N_seeds = 100
mindistance = min(size)/20
seeds = damask.seeds.from_Poisson_disc(size, N_seeds, 100, mindistance, False)
grid = damask.GeomGrid.from_Voronoi_tessellation(cells, size, seeds)
grid.save(f'Polycystal_{N_seeds}_{cells[0]}x{cells[1]}x{cells[2]}')
grid
