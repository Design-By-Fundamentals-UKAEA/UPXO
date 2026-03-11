#//////////////////////////////////////////////////////////////////////////////
# Script information for the file.
__name__ = "UPXO: UKAEA Poly-XTAL Operations"
__authors__ = ["Vaasu Anandatheertha"]
__lead_developer__ = ["Vaasu Anandatheertha"]
__emails__ = ["vaasu.anandatheertha@ukaea.uk", ]
__version__ = ["0.1:@ upto.271022","0.2:@ from.281022","0.3:@ from.181122"]
__license__ = "GPL v3"
#//////////////////////////////////////////////////////////////////////////////
import colorama
import numpy as np
from copy import deepcopy
from colorama import Fore
from colorama import Back
from colorama import Style
from shapely import speedups
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque
from colorama import init as colorama_init
import numpy.random as rand
import time
import matplotlib.pyplot as plt
from random import sample

# from upxo.geoEntities.point2d import point2d
from upxo.geoEntities.point2d import Point2d as point2d
from upxo.geoEntities.mulpoint2d import MPoint2d as mulpoint2d
# from upxo.geoEntities.mulpoint2d import mulpoint2d
from upxo.statops.distr_01 import distribution
from upxo._sup.console_formats import console_seperator
from upxo.pxtal.vt import _shapely
from upxo.statops.distr_01 import distribution
#//////////////////////////////////////////////////////////////////////////////
class INSTANCES():
    '''
    # Read a excel file to load parameter values
    INSTANCE PARMATER SET:
            1. instance dimension
            2. parameter names and their values
    '''
    __slots__ = ('D1_name',
                 'D1_par1name',
                 'D1_par1values',
                 )
    def __init__(self):
        pass

    pass


#//////////////////////////////////////////////////////////////////////////////
class vtpolyxtal2d():
    # TODO: change the following names to:
        # 1. areas_polygonal_exterior: L0_x_ape
    def __init__(self,
                 gsgen_method = 'vt',
                 vt_base_tool = 'shapely',
                 pxtal = None,
                 points = None,
                 point_method = 'mulpoints',
                 point_object_deque = None,
                 mulpoint_object = None,
                 locx_list = None,
                 locy_list = None,
                 xbound = None,
                 ybound = None,
                 vis_vtgs = False,
                 lean = 'no',
                 INSTANCE = None
                 ):
        #.......................

        #.......................
        self.L0 = self._data_container(feature = 'level_0_base',)
        #.......................
        self.L0.xbound, self.L0.ybound = xbound, ybound
        #.......................
        # Calculate the total area of the 2D pxtal domain
        self.L0.area_pxtal = float(abs(self.L0.xbound[1] - self.L0.xbound[0])*abs(self.L0.ybound[1] - self.L0.ybound[0]))
        #.......................
        # Make the boundaries data from the xbound and ybound arrays
        self.L0.boundaries_cw = self.order_points_cw(np.array([[i,j] for j in self.L0.ybound for i in self.L0.xbound]))
        #.......................
        # Store the methods used in the grain structure generation
        self.L0.gsgen_method = gsgen_method
        if gsgen_method == 'vt':
            self.L0.vt_base_tool = vt_base_tool
        #.......................
        if gsgen_method.lower() in ('vt', 'vtess', 'voronoi', 'v', 'voronoi_tessellation'):
            ''' Voronoi tessellation method '''
            if vt_base_tool in ('scipy', 'spy', 'shapely', 'sha'):
                if point_method in ('mp', 'mulpoints', 'mulpoint'):
                    # If UPXO mulpoint objects are to be used as seeds, then the
                    # following apply
                    if lean in ('no'):
                        self.L0.mpo_seeds = mulpoint_object
                if vt_base_tool in ('shapely', 'sha') and speedups.available:
                    # If shapely version supports, enable speedups
                    speedups.enable()
                    #    -    -    -    -    -
                self.from_points(vt_base_tool = vt_base_tool,
                                 point_method = point_method,
                                 mulpoint_object = mulpoint_object,
                                 xbound = self.L0.xbound,
                                 ybound = self.L0.ybound,
                                 vis_vtgs = vis_vtgs
                                 )
        #.......................
        if gsgen_method.lower() in ('mc', 'm', 'montecarlo', 'monte-carlo', 'monte_carlo'):
            ''' Monte-Carlo technique'''
            if point_method in ('mp', 'mulpoints', 'mulpoint'):
                if lean in ('no'):
                    self.L0.mpo_seeds = mulpoint_object
                #    -    -    -    -    -
        #.......................
        if gsgen_method.lower() in ('shapely_pxtal_load'):
            """
            Example
            -------
            from upxo.pxtal.polyxtal import vtpolyxtal2d as vtpxtal

            gtess = vtpxtal(gsgen_method = 'shapely_pxtal_load',
                            vt_base_tool = 'shapely',
                            pxtal = pxtal,
                            points = None,
                            point_method = None,
                            point_object_deque = None,
                            mulpoint_object = None,
                            locx_list = None,
                            locy_list = None,
                            xbound = None,
                            ybound = None,
                            vis_vtgs = False,
                            lean = 'no',
                            INSTANCE = None)
            """
            self.L0.vt_base_tool = vt_base_tool
            self.L0.pxtal = pxtal
            self.L0.xtals = list(self.L0.pxtal.geoms)
            self.L0.xtals_n = len(self.L0.xtals)
        #.......................
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def __eq__(self,
               bID = 0,
               qualification_metrics_and_values = {},
               ):
        '''
        bID: behaviour ID
        @ bID = 0: Compare mean grain sizes
        '''
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def __gt__(self,
               bID = 0,
               qualification_metrics_and_values = {},
               ):
        '''
        bID: behaviour ID
        @ bID = 0: Compare mean grain sizes
        '''
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def __lt__(self,
               bID = 0,
               qualification_metrics_and_values = {},
               ):
        '''
        bID: behaviour ID
        @ bID = 0: Compare mean grain sizes
        '''
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def __iter__(self):
        #for i in pxtal:
        #    print(40*'-')
        #    print(i.centroid)
        #    print(40*'-')
        return self.L0.xtals.__iter__()
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def load_xtals(self, method = 'ctf'):
        # Method to load list of polygon objects
        if method == 'ctf':
            ''' Import CTF files '''
            pass
        if method == 'cpr/crc':
            # Import defdap to deal with crc/cpr file and then export a ctf file
            # Then import the ctf file
            pass
        if method == 'upxo':
            pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def recompute_vtess(self):
        '''
        [1] for a polyxtal made from Voronoi tessellation,
            the seed points can be altered. If underlying
            seed points are altered, then tessellation is to be
            recomputed.
            Underlying seeds can be altered as below:
                pxtal.seed_points.plot()
        '''
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def prepare_seeds(self,
                      point_method = 'points',
                      mulpoint_object = None,
                      locx_list = [],
                      locy_list = [],
                      ):
        if point_method not in ('points', 'mulpoints', 'coords'):
            return 'Please specify correct point_method'
        if point_method == 'points':
            self.L0.coord_seeds_x, self.L0.coord_seeds_y = zip(*[(_.x, _.y) for _ in point_object_deque])
        if point_method == 'mulpoints':
            self.L0.coord_seeds_x, self.L0.coord_seeds_y = zip(*[(_.x, _.y) for _ in mulpoint_object.points])
        if point_method == 'coords':
            self.L0.coord_seeds_x, self.L0.coord_seeds_y = locx_list, locy_list
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def from_points(self,
                    vt_base_tool = 'shapely',
                    point_method = 'mulpoints',
                    mulpoint_object = None,
                    xbound = None,
                    ybound = None,
                    vis_vtgs = False,
                    ):
        #.......................
        self.prepare_seeds(point_method = point_method,
                           mulpoint_object = mulpoint_object
                           )
        #.......................
        if vt_base_tool == 'scipy':
            print('I am in scipy')
            from scipy.spatial import Voronoi
            from shapely.geometry import Polygon
            from vt import _finite_vtpols
            from vt import _make_bounding_polygon
            from vt import _clip_Voronoi_Tess_BoundBox
            vo = Voronoi(np.c_[mulpoint_object.locx,
                               mulpoint_object.locy]
                         )
            cells, points = _finite_vtpols(vo)
            boundDomain = _make_bounding_polygon(vo)
            # Get the shapely representation of xtals after clipping
            self.L0.xtals = _clip_Voronoi_Tess_BoundBox([Polygon((points[cell])) for cell in cells],
                                                        boundDomain
                                                        )
            self.L0.xtals_n = len(self.L0.xtals)
            # Calculate properties
            self.extract_shapely_coords(coord_of = 'L0_xtals_centroids',
                                        save_to_attribute = True,
                                        throw = False
                                        )
            self.extract_shapely_coords(coord_of = 'L0_xtals_reppoints',
                                        save_to_attribute = True,
                                        throw = False
                                        )
            # xtal primary boundary junction points
            self.extract_shapely_coords(coord_of = 'L0_xtal_vertices_pbjp',
                                        save_to_attribute = True,
                                        make_unique = True,
                                        throw = False
                                        )
            self.calculate_areas(area_type = 'polygonal')
            self.calculate_lengths(level = 0,
                                   length_type = 'xtal.polygonal.pbjp'
                                   )
            self.calculate_lengths(level = 0,
                                   length_type = 'xtal.polygonal.perimeter',
                                   )
            # Identify 1st nearest neighbours
            # Centroids
            # Represbntative points
            # Areas and area distribution
            # Lengths and length distribution
            # make mpo for centroids
            # make mpo for representative points
            # make mpo for vertices
        #.......................
        if vt_base_tool == 'shapely':
            print('I am in shapely')
            self.L0.pxtal = _shapely(point_method = 'mulpoints',
                                     _x = self.L0.coord_seeds_x,
                                     _y = self.L0.coord_seeds_y,
                                     xbound = xbound,
                                     ybound = ybound,
                                     vis_vtgs = vis_vtgs
                                     )
            self.L0.xtals = [_xtal for _xtal in self.L0.pxtal.geoms]
            print('##########################')
            self.L0.xtals_n = len(self.L0.xtals)
            print(self.L0.xtals_n)
            print('##########################')
            self.L0.xtals_ids = list(range(self.L0.xtals_n))
            #self.make_shapely_grains_prepared()
            # Calculate properties
            self.extract_shapely_coords(coord_of = 'L0_xtals_centroids',
                                        save_to_attribute = True,
                                        throw = False
                                        )
            self.extract_shapely_coords(coord_of = 'L0_xtals_reppoints',
                                        save_to_attribute = True,
                                        throw = False
                                        )
            self.calculate_areas(area_type = 'polygonal')
            self.calculate_lengths(level = 0,
                                   length_type = 'xtal.polygonal.pbjp'
                                   )
            #pxtal.L0.xtal_ble_val
            self.calculate_lengths(level = 0,
                                   length_type = 'xtal.polygonal.perimeter',
                                   )

            # APE_dis: areas_polygonal_exterior -- values
            self.L0.xtal_ape_dstr = distribution(data_name = 'L0.xtals_ape',
                                          data = np.array(self.L0.xtal_ape_val)
                                          )

            self.L0.xtal_pe_dstr = distribution(data_name = 'L0.xtals_pe',
                                         data = np.array(self.L0.xtal_pe_val)
                                         )
            #self.make_mpo_L0_centroids() # pxtal.L0.mpo_xtals_centroids
            #self.make_mpo_L0_reppoints() # pxtal.L0.mpo_xtals_reppoints
            #self.make_mpo_L0_vertices_unique() # pxtal.L0.mpo_xtals_vertices_unique
        #.......................
        if vt_base_tool == 'vedo':
            pass
        #.......................
        if vt_base_tool == 'freud':
            pass
        #.......................
        if vt_base_tool == 'pyvoro':
            # https://github.com/joe-jordan/pyvoro
            pass
        #.......................
        if vt_base_tool == 'pysal':
            # https://pysal.org/notebooks/lib/libpysal/voronoi.html
            # https://pysal.org/
            pass
        #.......................
        if vt_base_tool == 'grass':
            # https://grass.osgeo.org/grass78/manuals/v.voronoi.html
            # https://grass.osgeo.org/grass82/manuals/libpython/index.html
            # https://gitlab.com/vpetras/r.example.plus
            # https://grasswiki.osgeo.org/wiki/GRASS_and_Python
            # https://grass.osgeo.org/
            # http://www.gdmc.nl/publications/2009/3D_Voronoi_diagram.pdf
            pass
        #.......................
        if vt_base_tool == 'MicroStructPy':
            # https://docs.microstructpy.org/en/latest/index.html
            pass
        #.......................
        if vt_base_tool == 'custom1':
            pass
    #--------------------------------------------------------------------------
    def from_edges(self):
        # Method to make pxtal from shapely edge objects
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def from_vertices(self):
        # Method to make pxtal from shapely point objects
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def clip(self):
        # Method to clip pxtal from a bounding shapely polygon
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def add_xtal(self):
        # Method to add a xtal to pxtal
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def rem_xtal(self):
        # Method to remove a xtal to pxtal
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def progress_bar(self,
                     first_call = False,
                     final_call = False,
                     first_message = 'UPXO',
                     intermediate_print = False,
                     intermediate_message = '',
                     final_message = '',
                     progress_fraction = None,
                     color = colorama.Fore.WHITE,
                     ):
        '''
        Credits:
            Basic technique is borrowed from the below youtube video.
            https://www.youtube.com/watch?v=x1eaT88vJUA&t=79s
            Current class method has been modified. Method name, being accurate, has been
            retained as is in the YT video.
        '''
        import colorama
        percent = 100 * progress_fraction
        if first_call:
            print(2*'\n' + '\n' + first_message)
        if intermediate_print:
            print('------>' + intermediate_message + '\n')
        _bar ='*' * int(percent) + '-' * (100 - int(percent))
        print(f'\r|{_bar}| {percent:.1f}%', end = '\r')
        if final_call:
            print(final_message + 2*'\n')
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def find_neighs(self):
        '''
        Find neighbouring grains
        # TODO: THERE IS NO NEED FOR DICTIONARY. JUST USE A LIST INSTEAD
        # TODO: USE MULTI-PROCESSING MODULE AND CRANK UP THE SPEED
        # NOTE: Leave this as is
        '''
        import colorama
        __neigh = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(len(self.L0.xtals))}
        self.progress_bar(first_call = True,
                          first_message = 'Identifying ALL neighbours',
                          progress_fraction = 0,
                          color = colorama.Fore.WHITE) # Initiate the progress bar
        for xtal_count, thisGrain in enumerate(range(self.L0.xtals_n)):
            thisGrain_POU = self.L0.pxtal.geoms[thisGrain]
            neighCount = 0 # Number of neighbours for this count
            for possible_neighGrain in range(len(self.L0.xtals)):
                if thisGrain != possible_neighGrain:  # Avoid self-checks
                    possible_neighGrain_POU = self.L0.pxtal.geoms[possible_neighGrain]
                    if thisGrain_POU.touches(possible_neighGrain_POU):
                        # This means these two grains are neighbours
                        if neighCount == 0:
                            __neigh[thisGrain][1] = [possible_neighGrain]
                        else:
                            __neigh[thisGrain][1].append(possible_neighGrain)
                        neighCount += 1
            self.progress_bar(progress_fraction = (xtal_count+1)/self.L0.xtals_n,
                              color = colorama.Fore.WHITE)
        self.L0.neigh_all = deepcopy(__neigh)
        # Explanation: Say, GRn = 1
        # Then, neigh[GRn] is something like: ['L0GRAIN-0000001', [1, 3, 6]]
        # This means, grains 1, 3 and 6 are neighbours of grain GRn = 1
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def order_points_cw(self,
                        pts
                        ):
        '''
        Order a group of points clockwise
        Credit: https://gist.github.com/flashlib/e8261539915426866ae910d55a3f9959
        NOTE: The original code structure has been edited for compactness
        '''
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]
        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost, rightMost = xSorted[:2, :], xSorted[2:, :]
        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        #leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        #(tl, bl) = leftMost
        (tl, bl) = leftMost[np.argsort(leftMost[:, 1]), :]
        # if use Euclidean distance, it will run in error when the object
        # is trapezoid. So we should use the same simple y-coordinates order method.
        # now, sort the right-most coordinates according to their
        # y-coordinates so we can grab the top-right and bottom-right
        # points, respectively
        # rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        # (tr, br) = rightMost
        (tr, br) = rightMost[np.argsort(rightMost[:, 1]), :]
        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="float32")
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def extract_shapely_coords(self,
                               shapely_grains_list = None,
                               coord_of = 'L0_xtals_reppoints',
                               save_to_attribute = True,
                               make_unique = True,
                               throw = True
                               ):
        '''
        make_unique: True/False: Applies only to
        '''
        if shapely_grains_list is None:
            grain_list = self.L0.xtals
        elif type(shapely_grains_list) in (list, tuple):
            grain_list = list(shapely_grains_list)
        #''''''''''''
        if coord_of == 'L0_xtals_reppoints':
            '''
            grain_list= pxtal.L0.xtals
            a, b = zip(*[_.representative_point().xy for _ in grain_list])
            _, __ = zip(*[[_a[0], _b[0]] for (_a, _b) in zip(a, b)])
            ___ = pxtal.make_xy(_, __)
            '''
            a, b = zip(*[_.representative_point().xy for _ in grain_list])
            _, __ = zip(*[[_a[0], _b[0]] for (_a, _b) in zip(a, b)])
            ___ = self.make_xy(_, __)
            if save_to_attribute:
                self.L0.xtal_coord_reppoint_x = _
                self.L0.xtal_coord_reppoint_y = __
                self.L0.xtal_coord_reppoint_xy = ___
            if throw:
                return _, __, ___
        elif coord_of == 'L0_xtals_centroids':
            a, b = zip(*[_.centroid.xy for _ in grain_list])
            _, __ = zip(*[[_a[0], _b[0]] for (_a, _b) in zip(a, b)])
            ___ = self.make_xy(_, __)
            if save_to_attribute:
                self.L0.xtal_coord_centroid_x = _
                self.L0.xtal_coord_centroid_y = __
                self.L0.xtal_coord_centroid_xy = ___
            if throw:
                return _, __, ___
        elif coord_of == 'L0_xtal_vertices_pbjp':
            a, b = zip(*[_.boundary.xy for _ in grain_list])
            _, __ = zip(*[[_a, _b] for (_a, _b) in zip(a, b)])
            ___ = self.make_xy(_, __)
            if make_unique:
                xcoord, ycoord, xycoord = self.make_coord_list_as_unique(xcoord = _, ycoord = __)
                if save_to_attribute:
                    self.L0.xtal_coord_pbjp_x = _
                    self.L0.xtal_coord_pbjp_y = __
                    self.L0.xtal_coord_pbjp_xy = ___
                if throw:
                    return xcoord, ycoord, xycoord
            else:
                if save_to_attribute:
                    self.L0.xtal_coord_pbjp_x = _
                    self.L0.xtal_coord_pbjp_y = __
                    self.L0.xtal_coord_pbjp_xy = ___
                if throw:
                    return [list(i) for i in _], [list(i) for i in __], ___
        elif coord_of == 'L0_xtal_vertices_pbjp_xtalwise':
            if throw:
                return [xtal.exterior.xy for xtal in grain_list]
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_coord_list_as_unique(self, xcoord = None,
                                  ycoord = None):
        '''
        # Build x and y coordinste lists
        # TODO: Replace the below nested for loop with nested list comprehension
        '''
        x, y = [], []
        for _x, _y in zip(xcoord, ycoord):
            for __x, __y in zip(_x, _y):
                x.append(__x)
                y.append(__y)
        # Combine x and y and make unique coordinate list, xy
        xycoord = np.unique(np.vstack((np.array(x), np.array(y))).T, axis = 0)
        # Make vertice coordinate data list
        xcoord, ycoord = zip(*[(ಕ,  ನ) for ಕ,  ನ in xycoord])
        return xcoord, ycoord, xycoord
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def build_vertices(self,
                       data_structure = 'shapely',
                       make_unique = True
                       ):
        '''
        Build the unique coordinate list for vertices of the L0 VTGS
        '''
        #if hasattr(self, 'locx_gvert_list'):
        #    print('vertices list exit')
        if data_structure == 'shapely':
             a, b, c = self.extract_shapely_coords(coord_of = 'L0_xtal_vertices_pbjp',
                                                   make_unique = make_unique
                                                   )
             self.L0.xtal_coord_vertices_x = deepcopy(a)
             self.L0.xtal_coord_vertices_y = deepcopy(b)
             self.L0.xtal_coord_vertices_xy = deepcopy(c)
        self.count_number_of_xpbjp(recalculate = False,
                                   data_structure = 'shapely',
                                   )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def count_number_of_xpbjp(self,
                              recalculate = False,
                              data_structure = 'shapely'
                              ):
        '''
        Count the total numebr of xtal primary boundary junction points
        CAUTION: only use after vertices list have been uniqued
        '''
        if recalculate:
            self.build_vertices(data_structure = data_structure)
        self.L0.n_vert = len(self.L0.xtal_coord_vertices_x)
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_seed_points_xy(self):
        '''
        Extracts the seed points coordinates as two lists for VTGS
        '''
        return self.L0.mpo_seeds.locx, self.L0.mpo_seeds.locy
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_centroid_points_xy(self,
                                make_point_objects_upxo = True,
                                throw = True
                                ):
        '''
        Documentation
        '''
        _, __ = zip(*[_.centroid.xy for _ in self.L0.xtals])
        cenx, ceny = zip(*[[_[0], __[0]] for (_, __) in zip(_, __)])
        if make_point_objects_upxo:
            cenpo = mulpoint2d(method = 'coords', coordx = cenx, coordy = ceny)
        return list(cenx), list(ceny), cenpo
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_rep_points_xy(self,
                           make_point_objects_upxo = True,
                           throw = True
                           ):
        '''
        Documentation
        '''
        self.extract_shapely_coords
        _, __ = zip(*[_.representative_point().xy for _ in self.L0.xtals])
        repx, repy = zip(*[[_[0], __[0]] for (_, __) in zip(_, __)])
        if make_point_objects_upxo:
            reppo = mulpoint2d(method = 'coords', coordx = repx, coordy = repy)
        return list(repx), list(repy), reppo
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_xy(self,
                locx,
                locy
                ):
        '''
        Documentation
        '''
        return [[_x, _y] for (_x, _y) in zip(locx, locy)]
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_tree_vt_base_seeds(self,
                                  recalculate = True
                                  ):
        '''
        Documentation
        '''
        #if hasattr(self.L0, 'tree_seeds'):
        #if not recalculate:
        #    print('The tree exists.')
        #else:
        #    self.make_tree_seed_points(recalculate = True)
        seed_points_x, seed_points_y = self.make_seed_points_xy()
        locxy = self.make_xy(seed_points_x, seed_points_y)
        from scipy.spatial import cKDTree as ckdt
        self.L0.tree_seeds = ckdt(locxy,
                                  copy_data = False,
                                  balanced_tree = True
                                  )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_mpo_L0_centroids(self):
        self.L0.mpo_xtals_centroids = mulpoint2d(mulpoint_type = 'centroids',
                                                 method = 'coords',
                                                 coordx = self.L0.xtal_coord_centroid_x,
                                                 coordy = self.L0.xtal_coord_centroid_y
                                                 )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_mpo_L0_reppoints(self):
        self.L0.mpo_xtals_reppoints = mulpoint2d(mulpoint_type = 'reppoints',
                                                 method = 'coords',
                                                 coordx = self.L0.xtal_coord_reppoint_x,
                                                 coordy = self.L0.xtal_coord_reppoint_y
                                                 )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_mpo_L0_vertices_unique(self):
        '''
        self.L0.mpo_xtals_vertices:
            MPO. Multi-Point Object
            L0. Level 0
            _xtals_vertices. vertices of xtals
        '''
        # Create point2d objects for every vertex
        if not hasattr(self.L0, 'xtal_coord_vertices_x'):
            self.build_vertices(data_structure = 'shapely',
                                make_unique = True
                                )
        PO_L0_vertices = [point2d(x = _x,
                                  y = _y,
                                  set_rid = True, rid_length = 4,
                                  set_mid = False,
                                  set_dim = False, dim = 2,
                                  set_ptype = True, ptype = 'vt2dseed',
                                  set_jn = True, jn = 3,
                                  set_loc = False, loc = 'internal',
                                  store_original_coord = False,
                                  set_phase = False, phid = 1, phname = 'ukaea',
                                  set_tcname = False, tcname = 'B',
                                  set_ea = False, earepr = 'Bunge', ea = [45, 35, 0],
                                  set_oo = False, oo = None,
                                  set_tdist = False, tdist = 0.0000000000001,
                                  store_vis_prop = False
                                  )
                          for _x, _y in zip(self.L0.xtal_coord_vertices_x,
                                            self.L0.xtal_coord_vertices_y
                                            )
                          ]
        # Assign junction type for each of the points
        # USE THIS ATTRIBUTE OF THE POINT2D: assign_junction_type(self, jtype = 3)
        self.L0.mpo_xtals_vertices_unique = mulpoint2d(mulpoint_type = 'vertices',
                                                       point_objects = PO_L0_vertices)
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def extract_coord(self,
                      base_data_structure = 'shapely',
                      feature = 'L0_xtals_reppoints',
                      ):
        '''
        self.extract_coord(feature = 'L0_xtals_reppoints')
        self.extract_coord(feature = 'L0_xtals_centroids')
        '''
        if base_data_structure == 'shapely' and feature == 'L0_xtals_reppoints':
            self.extract_shapely_coords(shapely_grains_list = None,
                                        coord_of = 'L0_xtals_reppoints',
                                        save_to_attribute = True,
                                        make_unique = True,
                                        throw = False
                                        )
        if base_data_structure == 'shapely' and feature == 'L0_xtals_centroids':
            self.extract_shapely_coords(shapely_grains_list = None,
                                        coord_of = 'L0_xtals_centroids',
                                        save_to_attribute = True,
                                        make_unique = True,
                                        throw = False
                                        )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_tree_xtals_centroids(self,
                                  recalculate = True
                                  ):
        '''
        Documentation
        '''
        #if hasattr(self.L0, 'tree_centroids'):
        #    if not recalculate:
        #        print('The tree exists.')
        #    else:
        #        self.make_tree_centroid_points(recalculate = True)
        #else:
        from scipy.spatial import cKDTree as ckdt
        #self.cenx, self.ceny, self.cenp = self.make_centroid_points_xy(make_point_objects_upxo = True,
        #                                                               throw = True
        #                                                               )
        #locxy = self.make_xy(self.cenx,
        #                     self.ceny
        #                     )
        if hasattr(self.L0, 'xtal_coord_centroid_xy'):
            pass
        else:
            self.extract_coord(feature = 'L0_xtals_centroids')
        self.L0.tree_centroids = ckdt(self.L0.xtal_coord_centroid_xy,
                                      copy_data = False,
                                      balanced_tree = True
                                      )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_tree_xtals_reppoints(self,
                             recalculate = True
                             ):
        '''
        Documentation
        '''
        #if hasattr(self.L0, 'tree_rep_points'):
        #    if not recalculate:
        #        print('The tree exists.')
        #    else:
        #        self.make_tree_rep_points(recalculate = True)
        #else:
        from scipy.spatial import cKDTree as ckdt
        if hasattr(self.L0, 'xtal_coord_reppoint_xy'):
            if len(self.L0.xtal_coord_reppoint_xy) == 0:
                self.extract_coord(feature = 'L0_xtals_reppoints')
        else:
            self.extract_coord(feature = 'L0_xtals_reppoints')
        #self.repx, self.repy, self.repp = self.make_rep_points_xy(make_point_objects_upxo = True,
        #                                                          throw = True
        #                                                          )
        #locxy = self.make_xy(self.repx,
        #                     self.repy
        #                     )
        self.L0.tree_reppoints = ckdt(self.L0.xtal_coord_reppoint_xy,
                                      copy_data = False,
                                      balanced_tree = True
                                      )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_tree_xtals_pbjp(self,
                             recalculate = True,
                             vertex_type = 'L0_xtals_pbjp',
                             ):
        '''
        Documentation
        '''
        if vertex_type == 'L0_xtals_pbjp':
            #if hasattr(self.L0, 'tree_L0_grain_vertices'):
            #    if not recalculate:
            #        print('The tree exists.')
            #    else:
            #        self.make_tree_grain_vertices(recalculate = True,
            #                                      xtal_vertex_type = 'L0_xtal_vertices_pbjp'
            #                                      )
            #else:
            from scipy.spatial import cKDTree as ckdt
            if hasattr(self.L0, 'xtal_coord_vertices_xy'):
                if len(self.L0.xtal_coord_vertices_xy) == 0:
                    self.extract_coord(feature = 'L0_xtals_pbjp')
            else:
                self.extract_coord(feature = 'L0_xtals_pbjp')

            # self.build_vertices(data_structure = 'shapely')
            self.L0.tree_pbjp = ckdt(self.L0.xtal_coord_vertices_xy,
                                     copy_data = False,
                                     balanced_tree = True
                                     )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_tree_triple_point_junctions(self):
        '''
        Documentation
        '''
        pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_tree(self,
                 tree = 'L0.xtal.centroids',
                 ):
        '''
        tree: centroids, rep_points, seed_points
        '''
        if tree == 'seed_points':
            try:
                _ = self.L0.tree_seeds
            except AttributeError:
                self.make_tree_seed_points()
        elif tree == 'L0.xtal.centroids':
            try:
                _ = self.L0.tree_centroids
            except AttributeError:
                self.make_tree_centroid_points()
        elif tree == 'L0.xtal.reppoints':
            try:
                _ = self.L0.tree_reppoints
            except AttributeError:
                self.make_tree_rep_points()
        elif tree == 'L0.xtal.pbjp':
            try:
                _ = self.L0.xtal_coord_vertices_x
            except AttributeError:
                self.make_tree_grain_vertices(recalculate = True,
                                              grain_vertex_type = 'L0_xtal_vertices_pbjp',
                                              )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def get_neigh_xtal_of_xtals(self,
                                method: str = 'from_grain_list',
                                query_grain_id_method: str = 'fromid',
                                rebuild_neigh_database = False,
                                build_neigh_using_shapely = True,
                                central_grain_ids: list = [0, 5],
                                central_grain_loc: list = [0.0, 0.0],
                                n_near_neighbours: int = 2,
                                cut_off_distance: float = 0.2
                                ):
        '''
        methods: from_grain_list, shapely_STRtree, shapely_,
                 from_centroid_tree, from_rep_point_tree,
                 from_seed_point_tree
        NOTE TO DEVELOPER: PLEASE LEAVE THIS UNTOUCHED
        '''
        # First build the neighbours database if self.L0.neigh_all does'nt exist already.
        # if isinstance('neigh', polyxtal) == False:
        if rebuild_neigh_database:
            if build_neigh_using_shapely:
                self.find_neighs()
        if method == 'from_grain_list':
            if query_grain_id_method == 'fromid':
                #.......................
                if  n_near_neighbours == 0:
                    grains_neigh_reach_0 = [central_grain_ids]
                    return grains_neigh_reach_0
                elif n_near_neighbours > 0:
                    grains_neigh_reach_0 = [central_grain_ids]
                    grains_neigh_reach_1 = []
                    if n_near_neighbours == 2:
                        grains_neigh_reach_2 = []
                    for count, central_grain_id in enumerate(central_grain_ids):
                        #.......................
                        _ = [grain for grain in self.L0.neigh_all[central_grain_id][1:][0]]
                        _.append(central_grain_ids[count])
                        _.sort()
                        grains_neigh_reach_1.append(_)
                        #.......................
                        if n_near_neighbours == 2:
                            __grains_neigh_reach_2 = [[grain for grain in self.L0.neigh_all[grain][1:][0]] for grain in grains_neigh_reach_1[count]]
                            __grains_neigh_reach_2 = list(set([_ for __ in __grains_neigh_reach_2 for _ in __]))
                            __grains_neigh_reach_2.sort()
                            grains_neigh_reach_2.append(__grains_neigh_reach_2)
                #.......................
                if n_near_neighbours == 0:
                    return grains_neigh_reach_0
                elif n_near_neighbours == 1:
                    grains_neigh_reach_1_ = deepcopy(grains_neigh_reach_1)
                    for i, n in enumerate(grains_neigh_reach_0[0]):
                        grains_neigh_reach_1_[i].remove(n)
                    return grains_neigh_reach_0, grains_neigh_reach_1, grains_neigh_reach_1_
                elif n_near_neighbours == 2:
                    grains_neigh_reach_1_ = deepcopy(grains_neigh_reach_1)
                    for i, n in enumerate(grains_neigh_reach_0[0]):
                        grains_neigh_reach_1_[i].remove(n)
                    grains_neigh_reach_2_ = deepcopy(grains_neigh_reach_2)
                    for i, n in enumerate(grains_neigh_reach_1):
                        for _, m in enumerate(n):
                            grains_neigh_reach_2_[i].remove(m)
                    return grains_neigh_reach_0, grains_neigh_reach_1, grains_neigh_reach_1_, grains_neigh_reach_2, grains_neigh_reach_2_
        if method == 'shapely_STRtree':
            pass
        if method == 'shapely_':
            pass
        if method == 'from_centoid_tree':
            # Extract centroids / retrieve centroids
            # Build ckdtree from centroids / retrieve centroids ckdtree
            pass
        if method == 'from_rep_point_tree':
            # Extract shapely representative point/ retrieve shapely representative point
            # Build ckdtree from shapely representative point / retrieve shapely representative point ckdtree
            pass
        if method == 'from_seed_point_tree':
            # retrieve seed point
            # Build ckdtree from seed point / retrieve seed point ckdtree
            pass
        if query_grain_id_method == 'fromid':
            pass
        if query_grain_id_method == 'rep_point':
            pass
        if query_grain_id_method == 'L0_xtals_centroids':
            pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def identify_corner_xtals(self,
                              save_as_attribute = True,
                              viz = True,
                              throw = False
                              ):
        '''
        Identify the grains lying on he boundary of the domain.
        '''
        if hasattr(self, 'locxy_gvert_list'):
            pass
        else:
            self.build_vertices(data_structure = 'shapely')
        _, _, pxtal_xy_vertices = self.extract_shapely_coords(coord_of = 'L0_xtal_vertices_pbjp',
                                                             save_to_attribute = False,
                                                             make_unique = False,
                                                             throw = True
                                                             )
        # Convert this pxtal_xy_vertices to pxtal_xy_vertices made of numpy arrays
        pxtal_xy_vertices = [[np.array(list(xtal_xy[0])),
                              np.array(list(xtal_xy[1]))
                              ]
                             for xtal_xy in pxtal_xy_vertices
                             ]
        # Identify corner grains
        corner_xtals, xtal_count = [], 0
        for xtal_xy_vertices in pxtal_xy_vertices:
            for _ in self.L0.boundaries_cw:
                __ = np.stack((xtal_xy_vertices[0][:-1], xtal_xy_vertices[1][:-1]),
                              axis = -1)
                if np.equal(__, _).all(1).any():
                    corner_xtals.append(xtal_count)
            xtal_count += 1
            #location = np.where(np.all(__ == _, axis = 1))
        # Save as attribute if required
        if save_as_attribute:
            self.corner_xtals = corner_xtals
        # Visualize if required
        if viz:
            self.plot(highlight_specific_grains = True,
                      specific_grains = corner_xtals,
                      highlight_specific_grains_annot = True,
                      highlight_specific_grains_colour = True,
                      highlight_specific_grains_hatch = False,
                      )
        # Return the list of corner_xtals if needed
        if throw:
            return corner_xtals
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def build_scalar_fields_from_xtal_list(self,
                                           scalar_field = 'bx_ape',
                                           xtal_list = [None],
                                           save_to_attribute = True,
                                           compute_distribution = True,
                                           throw = False
                                           ):
        '''
        CALL:
            pxtal.build_scalar_fields_from_xtal_list(scalar_field = 'areas_polygonal_exterior',
                                                     xtal_list = pxtal.boundary_xtals,
                                                     save_to_attribute = True,
                                                     throw = False
                                                     )
        '''
        if xtal_list is not None:
            if type(xtal_list) in (np.ndarray, list, tuple):
                if scalar_field == 'ape':
                    self.calculate_areas(area_type = 'polygonal')
                if scalar_field == 'bx_ape':
                    ape_val = [self.L0.xtal_ape_val[xtal_id] for xtal_id in xtal_list]
                    if save_to_attribute:
                        self.L0.xtal_ss_boundary.ape_val = ape_val
                    if throw:
                        return ape_val
                # ---------------------------------------------------
                if scalar_field == 'ix_ape':
                    ape_val = [self.L0.xtal_ape_val[xtal_id] for xtal_id in xtal_list]
                    if save_to_attribute:
                        self.L0.xtal_ss_internal.ape_val = ape_val
                    if throw:
                        return ape_val
                #---------------------------------------------------
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def _data_container(self,
                        feature = 'xtal_list',
                        ):
        '''
        Start a dataclass with required fields
        '''
        @dataclass(repr = False)
        class data_container():
            pass
        #------------------------------------------------
        if feature == 'level_0_base':
            _ = data_container

            _.pxtal_shapely = None # Shapely multi-polygon object
            _.xtals: list = [] #
            _.xtal_objects_shapely: list = [] # Shapely polygon objects in deque data structure
            _.xtal_objects_upxo: list = [] # Shapely polygon objects in deque data structure
            #_.seed_points: list = [] # To be deprecated if not needed

            _.xbound: float = None
            _.ybound: float = None

            _.area_pxtal: float = None
            _.boundaries_cw: list = []
            _.gsgen_method: str = None
            _.vt_base_tool: str = None

            _.xtal_coord_centroid_x: list = []
            _.xtal_coord_centroid_y: list = []
            _.xtal_coord_centroid_xy: list = []
            _.xtal_coord_reppoint_x: list = []
            _.xtal_coord_reppoint_y: list = []
            _.xtal_coord_reppoint_xy: list = []
            _.xtal_coord_bjp_x: list = []
            _.xtal_coord_bjp_y: list = []
            _.xtal_coord_bjp_xy: list = []

            _.neigh_all: dict = None

            _.mpo_seeds = None # Multi-point object of seed points
            _.mpo_xtals_centroids = None # Multi-point object of centroids
            _.mpo_xtals_reppoints = None # Multi-point object of representative points
            _.mpo_xtals_vertices_unique = None # Multi-point object of xtal vertices

            _.tree_seeds = None
            _.tree_centroids = None
            _.tree_reppoints = None
            _.tree_pbjp = None

            _.xtal_ids: list = [] # List of ids of the individual xtal objects
            _.xtals_n: list = [] # NUmber of xtals in the pxtal
            _.xtal_ape_val: list = [] # xtal Area polygonal exterior - values
            _.xtal_ape_dstr: list = [] # xtal Area polygonal exterior - distribution
            _.xtal_ble_val: list = [] # xtal boundary length exterior - values
            _.xtal_pe_val: list = [] # xtal perimeter exterior - values
            _.xtal_pe_dstr: list = [] # xtal perimeter exterior - distribution
            _.xtal_bjp_n: list = [] # Number of bundary junction points in each xtal
            _.xtal_tpj_n: list = [] # Number of triple point junctions
            _.xtal_jpia_val: list[list[float]] = [] # Junction point internal angles list

            # Data containers boundary and internal xtals in the pxtal
            _.xtal_ss_boundary: list = self._data_container(feature = 'xtal_list')
            _.xtal_ss_internal: list = self._data_container(feature = 'xtal_list')
            return _
        #------------------------------------------------
        if feature == 'xtal_list':
            _xtals = data_container
            #--------------------------------------
            _xtals.ids: list[int] = None  # ids: ids of boundary xtals
            _xtals.n: int = None # n: Number of boundary xtals
            _xtals.a: float = None # a: total polygonal area external
            _xtals.af: float = None # af: area frac. = (area of all boundary xtals)/pxtal_area
            _xtals.ape_val: list[float] = None # APE_val: areas_polygonal_exterior -- values
            _xtals.ble_val: list[float] = None # XBL_val: xtal bnoundary Lengths exterior
            _xtals.bp_val: list[float] = None # XBP_val: xtal boundary perimeter
            _xtals.nBE: list[int] = None  # nBE: number of boundary edges
            return _xtals
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def __identify_xtals_with_coordinates(self,
                                          xy_coords = []
                                          ):
        self.progress_bar(first_call = True,
                          first_message = 'Identifying boundary grains',
                          progress_fraction = 0,
                          color = colorama.Fore.WHITE
                          )
        _ids = []
        for i, _xy_coords in enumerate(xy_coords):
            for _x in self.L0.xbound:
                if _x in _xy_coords[0][:-1]:
                    _ids.append(i)
            for _y in self.L0.ybound:
                if _y in _xy_coords[1][:-1]:
                    _ids.append(i)
            #print(i)
            #print(self.L0.xtals_n)
            self.progress_bar(progress_fraction = (i+1)/self.L0.xtals_n,
                              color = colorama.Fore.WHITE
                              )
        _ids = list(set(_ids))
        return _ids
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def identify_L0_xtals_boundary(self,
                                   domain_shape = 'rectangular',
                                   base_data_structure_to_use = 'shapely',
                                   build_scalar_fields = True,
                                   scalar_field_names = ['bx_ape'],
                                   viz = True,
                                   vis_dpi = 150,
                                   throw = False
                                   ):
        '''
        Identify grains lying on the boundary of the domain.
        CALL:
            pxt.identify_L0_xtals_boundary(domain_shape = 'rectangular',
                                        viz = True,
                                        vis_dpi = 150,
                                        throw = False
                                        )
        '''
        if not hasattr(self.L0, 'xtal_coord_vertices_xy'):
            self.build_vertices(data_structure = base_data_structure_to_use
                                )
        #------------------------
        if base_data_structure_to_use == 'shapely':
            _, _, pxtal_xy_vertices = self.extract_shapely_coords(coord_of = 'L0_xtal_vertices_pbjp',
                                                                  save_to_attribute = False,
                                                                  make_unique = False,
                                                                  throw = True
                                                                  )
        elif base_data_structure_to_use == 'upxo':
            pass
        elif base_data_structure_to_use == 'scipy':
            pass
        #------------------------
        # Convert this pxtal_xy_vertices to pxtal_xy_vertices made of numpy arrays
        pxtal_xy_vertices = [[np.array(list(xtal_xy[0])),
                              np.array(list(xtal_xy[1]))
                              ]
                             for xtal_xy in pxtal_xy_vertices
                             ]
        #------------------------
        # Identify boundary xtals
        if domain_shape == 'rectangular':
            _boundary_xtals = self.__identify_xtals_with_coordinates(xy_coords = pxtal_xy_vertices)
        #------------------------
        # ESTABLISH and STORE PROPERTIES OF BOUNDARY_XTALS
        self.L0.xtal_ss_boundary.ids = deepcopy(_boundary_xtals)
        #L0_xtals_boundary = self._data_container(feature = 'xtal_list',)
        self.L0.xtal_ss_boundary.n = len(_boundary_xtals)
        #L0_xtals_boundary.ids = deepcopy(_boundary_xtals)
        self.L0.xtal_ss_boundary.APE_val = [self.L0.xtal_ape_val[xtal_id] for xtal_id in _boundary_xtals]
        self.L0.xtal_ss_boundary.a = np.sum(self.L0.xtal_ss_boundary.APE_val)
        self.L0.xtal_ss_boundary.af = self.L0.xtal_ss_boundary.a / self.L0.area_pxtal
        self.L0.xtal_ss_boundary.XBL_val = None
        self.L0.xtal_ss_boundary.XBP_val = None
        self.L0.xtal_ss_boundary.nBE = None
        from distr_01 import distribution as dstr
        # APE_dis: areas_polygonal_exterior -- values
        self.L0.xtal_ss_boundary.APE_distr = dstr(data_name = 'L0_xtals_ape_val',
                                                  data = np.array(self.L0.xtal_ss_boundary.APE_val)
                                                  )
        #------------------------
        # Save as attribute
        # self.L0.xtals_boundary = deepcopy(L0_xtals_boundary)
        #------------------------
        # Visualize if required
        if viz:
            self.plot(dpi = vis_dpi,
                      highlight_specific_grains = True,
                      specific_grains = self.L0.xtal_ss_boundary.ids,
                      highlight_specific_grains_annot = False,
                      highlight_specific_grains_colour = True,
                      highlight_specific_grains_hatch = False,
                      )
        #------------------------
        if build_scalar_fields:
            #all_scalar_fields = ['areas_polygonal_exterior__boundary_xtals']
            if len(scalar_field_names) != 0:
                for sfn in scalar_field_names:
                    self.build_scalar_fields_from_xtal_list(scalar_field = sfn,
                                                            xtal_list = self.L0.xtal_ss_boundary.ids,
                                                            save_to_attribute = True,
                                                            throw = False
                                                            )
                # ACCESSES:
                    # pxtal.area_polygonal_exterior__boundary_xtals
                    # pxtal.
        #------------------------
        # Return the list of corner_xtals if needed
        if throw:
            return boundary_xtals
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def identify_L0_xtals_internal(self,
                                   domain_shape = 'rectangular',
                                   base_data_structure_to_use = 'shapely',
                                   build_scalar_fields = True,
                                   scalar_field_names = ['bx_ape'],
                                   viz = False,
                                   vis_dpi = 150,
                                   throw = False
                                   ):
        '''
        pxt.identify_internal_xtals(save_as_attribute = True,
                                    viz = False,
                                    throw = False)
        '''
        #------------------------
        if hasattr(self, 'boundary_xtals'):
            pass
        else:
            self.identify_L0_xtals_boundary(domain_shape = 'rectangular',
                                            viz = False,
                                            throw = False
                                            )
        #------------------------
        # Calculate the list of all grains which are not on the boundaries of the pxtal
        self.progress_bar(first_call = True,
                          first_message = 'Identifying internal grains',
                          progress_fraction = 0,
                          color = colorama.Fore.WHITE
                          )
        _internal_xtals = list(np.sort(list(set(self.L0.xtals_ids) - set(self.L0.xtal_ss_boundary.ids)),
                                       axis = 0
                                       )
                               )
        self.progress_bar(progress_fraction = 1.0,
                          color = colorama.Fore.WHITE
                          )
        #------------------------
        #------------------------
        # ESTABLISH and STORE PROPERTIES OF INTERNAL_XTALS
        self.L0.xtal_ss_internal.ids = deepcopy(_internal_xtals)
        #L0_xtals_boundary = self._data_container(feature = 'xtal_list',)
        self.L0.xtal_ss_internal.n = len(_internal_xtals)
        #L0_xtals_boundary.ids = deepcopy(_boundary_xtals)
        self.L0.xtal_ss_internal.APE_val = [self.L0.xtal_ape_val[xtal_id] for xtal_id in _internal_xtals]
        self.L0.xtal_ss_internal.a = np.sum(self.L0.xtal_ss_internal.APE_val)
        self.L0.xtal_ss_internal.af = self.L0.xtal_ss_internal.a / self.L0.area_pxtal
        self.L0.xtal_ss_internal.XBL_val = None
        self.L0.xtal_ss_internal.XBP_val = None
        self.L0.xtal_ss_internal.nBE = None
        from distr_01 import distribution as dstr
        # APE_dis: areas_polygonal_exterior -- values
        self.L0.xtal_ss_internal.APE_distr = dstr(data_name = 'L0_xtals_ape_val',
                                                  data = np.array(self.L0.xtal_ss_internal.APE_val)
                                                  )
        ## ESTABLISH AND STORE PROPERTIES OF INTERNAL_XTALS
        #L0_xtals_internal = self._data_container(feature = 'xtal_list',)
        #L0_xtals_internal.n = len(_internal_xtals)
        #L0_xtals_internal.ids = deepcopy(_internal_xtals)
        #L0_xtals_internal.APE_val = [self.areas_polygonal_exterior[xtal_id] for xtal_id in _internal_xtals]
        #L0_xtals_internal.a = np.sum(L0_xtals_internal.APE_val)
        #L0_xtals_internal.af = L0_xtals_internal.a / self.L0.area_pxtal
        #L0_xtals_internal.XBL_val = None
        #L0_xtals_internal.XBP_val = None
        #L0_xtals_internal.nBE = None
        #from distr_01 import distribution as dstr
        ## APE_dis: areas_polygonal_exterior -- values
        #L0_xtals_internal.APE_distr = dstr(data_name = 'areas_polygonal_internal',
        #                                   data = np.array(L0_xtals_internal.APE_val)
        #                                   )
        #------------------------
        ## Save as attribute
        #self.L0.xtals_internal = deepcopy(L0_xtals_internal)
        #------------------------
        #------------------------
        # Save as attribute if required
        #if save_as_attribute:
        #    self.L0.xtals_internal = L0_xtals_internal
        #------------------------
        # Visualize if required
        if viz:
            self.plot(dpi = vis_dpi,
                      highlight_specific_grains = True,
                      specific_grains = self.L0.xtal_ss_internal.ids,
                      highlight_specific_grains_annot = False,
                      highlight_specific_grains_colour = True,
                      highlight_specific_grains_hatch = False,
                      )
        #------------------------
        if build_scalar_fields:
            #all_scalar_fields = ['areas_polygonal_exterior__boundary_xtals']
            if len(scalar_field_names) != 0:
                for sfn in scalar_field_names:
                    self.build_scalar_fields_from_xtal_list(scalar_field = sfn,
                                                            xtal_list = self.L0.xtal_ss_internal.ids,
                                                            save_to_attribute = True,
                                                            throw = False
                                                            )
        #------------------------
        #if build_scalar_fields:
        #    all_scalar_fields = ['areas_polygonal_exterior__interior_xtals']
        #    for scalar_field_name in all_scalar_fields:
        #        self.build_scalar_fields_from_xtal_list(scalar_field = scalar_field_name,
        #                                                xtal_list = self.L0.xtals_internal.ids,
        #                                                save_to_attribute = True,
        #                                                throw = False
        #                                                )
                # ACCESSES:
                    # pxtal.area_polygonal_exterior__internal_xtals
                    # pxtal.
        #------------------------
        # Return the list of corner_xtals if needed
        if throw:
            return self.L0.xtal_ss_internal
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def check_fractions(self,
                        par = 'boundary_and_internal',
                        tolerance = 10**-6
                        ):
        '''
        check_n: check on whether the number of xtal_collection objects agree
                 on the number of xtals in the pxtals
        '''
        console_seperator(seperator = '-*', repetitions = 25)
        if par == 'boundary_and_internal':
            n, a, af = 'FAIL', 'FAIL', 'FAIL'
            from colorama import init as colorama_init
            colorama_init()
            from colorama import Fore, Back, Style
            if (self.L0.xtal_ss_boundary.n, self.L0.xtal_ss_internal.n, self.L0.xtals_n)!=(None, None, None):
                if (type(self.L0.xtal_ss_boundary.n), type(self.L0.xtal_ss_internal.n), type(self.L0.xtals_n)) == (int, int, int):
                    residual = self.L0.xtals_n - self.L0.xtal_ss_boundary.n - self.L0.xtal_ss_internal.n
                    if residual == 0:
                        n = 'PASS'
                        print(Fore.GREEN + f'count check, n = {n}, residual = {residual}')
                    else:
                        print(Fore.RED + f'count check, n = {n}, residual = {residual}')
            if (self.L0.xtal_ss_boundary.a, self.L0.xtal_ss_internal.a, self.L0.area_pxtal)!=(None, None, None):
                if (type(self.L0.xtal_ss_boundary.a), type(self.L0.xtal_ss_internal.a), type(self.L0.area_pxtal)) == (np.float64, np.float64, float):
                    residual = self.L0.area_pxtal - self.L0.xtal_ss_boundary.a - self.L0.xtal_ss_internal.a
                    if residual <= tolerance:
                        a = 'PASS'
                        print(Fore.GREEN + f'count check, a = {a}, residual = {residual}')
                    else:
                        print(Fore.RED + f'count check, a = {a}, residual = {residual}')
            if (self.L0.xtal_ss_boundary.af, self.L0.xtal_ss_internal.af)!=(None, None):
                if (type(self.L0.xtal_ss_boundary.af), type(self.L0.xtal_ss_internal.af)) == (np.float64, np.float64):
                    residual = 1.0 - self.L0.xtal_ss_boundary.af - self.L0.xtal_ss_internal.af
                    if residual <= tolerance:
                        af = 'PASS'
                        print(Fore.GREEN + f'area fraction check, af = {af}, residual = {residual}')
                    else:
                        print(Fore.RED + f'area fraction check, af = {af}, residual = {residual}')
            Style.RESET_ALL
        console_seperator(seperator = '-*', repetitions = 25)
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def make_shapely_xtals_prepared(self):
        '''
        Documentation
        '''
        from shapely.prepared import prep
        return deque([prep(_xtal) for _xtal in self.L0.xtals])
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def correlate_L0_xtal_id_seed_id(self):
        '''
        # TODO: TO BE DEPREACTED? BUGGY AND INCOMPLETE.

        correlate VTGS grain ID number to the corresponding seed ID
        '''
        from shapely.geometry import Point
        #seed_points_x, seed_points_y = self.make_seed_points_xy()
        seeds_shapely_points = [Point(_x,_y) for (_x,_y) in zip(pxtal.L0.coord_seeds_x, pxtal.L0.coord_seeds_y)]
        seeds_shapely_points_id = [id(_) for _ in seeds_shapely_points]

        vgrainid_seedid = [0 for i in self.L0.xtals]
        upxo_seeds = [0 for i in self.L0.xtals]
        from point2d_04 import point2d
        for _, grain in enumerate(self.L0.xtals_shapely_prepared):
            matched_points = list(filter(grain.contains, seeds_shapely_points))
            #matched_points_id = [__ for __ in matched_points]
            matched_point = matched_points[0]
            upxo_seeds[_] = point2d(x = matched_point.x,
                                    y = matched_point.y)
        return upxo_seeds
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def write_abapy_input_coords(self,
                                 identification_point_type = 'L0_xtals_reppoints',
                                 ):
        '''
        x and y coordinstes of the voronoi tessellation are contained in:
            self.L0.xtal_coord_vertices_x   and
            self.L0.xtal_coord_vertices_y
        Creation: 131022va
        Edit history: 221022va, 271022va
        '''
        xtal_coordinates = self.extract_shapely_coords(shapely_grains_list = None,
                                                       coord_of = 'L0_xtal_vertices_pbjp_xtalwise',
                                                       save_to_attribute = False,
                                                       make_unique = False,
                                                       throw = True
                                                       )
        n = [len(_xtal_pbjp_coord[0])-1 for _xtal_pbjp_coord in xtal_coordinates]
        nmax = max(n)
        #------------------------------------
        template = np.tile(np.array([999. for i in range(0, nmax)],
                                    dtype = float
                                    ),
                           (self.L0.xtals_n, 1)
                           )
        x, y = deepcopy(template), deepcopy(template)
        #------------------------------------
        # TODO: Documentaion to come here
        # TODO: 1. Replace the values 999 by NaN when writing to disk (Check with ABAQUS if this can work)
        for i, xy in enumerate(xtal_coordinates):
            __X, __Y = np.array(list(xy[0][:-1])), np.array(list(xy[1][:-1]))
            if len(__X) < nmax:
                endat = nmax-len(__X)
                x[i][:-endat], y[i][:-endat] = __X, __Y
            elif len(__X) == nmax:
                x[i][:], y[i][:] = __X, __Y
        np.savetxt('xdata.txt', x, delimiter=',', fmt = '%3.8f')
        np.savetxt('ydata.txt', y, delimiter=',', fmt = '%3.8f')
        #------------------------------------
        # TODO: Documentaion to come here
        identification_points_xy = []
        if identification_point_type =='L0_xtals_reppoints':
            if hasattr(self.L0,'xtal_coord_reppoint_xy'):
                identification_points_xy = self.L0.xtal_coord_reppoint_xy
            else:
                self.extract_shapely_coords(shapely_grains_list = None,
                                            coord_of = 'L0_xtals_reppoints',
                                            save_to_attribute = True,
                                            throw = False
                                            )
                identification_points_xy = np.array(self.L0.xtal_coord_reppoint_xy)
        elif identification_point_type == 'L0_xtals_centroids':
            if hasattr(self.L0,'xtal_coord_centroid_xy'):
                identification_points_xy = self.L0.xtal_coord_centroid_xy
            else:
                self.extract_shapely_coords(shapely_grains_list = None,
                                            coord_of = 'L0_xtals_centroids',
                                            save_to_attribute = True,
                                            throw = False
                                            )
                identification_points_xy = np.array(self.L0.xtal_coord_centroid_xy)
        np.savetxt('identification_points_xy.txt', identification_points_xy, delimiter=',', fmt = '%3.8f')
        #------------------------------------
        # Credit for output formatting:
            # https://stackoverflow.com/questions/8924173/how-to-print-bold-text-in-python
            # NOTE:
                # THE COLOUR SPECIFICATIONS ARE NOT WORKING [HENCE COMMENTED OUT]. USING COLORAMA INSTEAD
                # HOWEVER, BOLD AND UNDERLINE SPECIFFICATIONS WORK WELL. These are retained
        #_PURPLE = '\033[95m'
        #_CYAN = '\033[96m'
        #_DARKCYAN = '\033[36m'
        #_BLUE = '\033[94m'
        #_GREEN = '\033[92m'
        #_YELLOW = '\033[93m'
        #_RED = '\033[91m'
        _BOLD = '\033[1m'
        _UNDERLINE = '\033[4m'
        _END = '\033[0m'
        colorama_init()
        console_seperator(seperator = '-*', repetitions = 25)
        print(_BOLD + Fore.MAGENTA + 'Coordinate files have been written to parent directory' + _END)
        print(_BOLD + Fore.CYAN + 'Please use [UPXO] --> [Abaqus python script] and' + _END)
        print('      ' + _BOLD + _UNDERLINE + Fore.CYAN + 'import the VTGS.2d in ABAQUS' + _END)
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def calculate_lengths(self,
                          level = 0,
                          length_type = 'xtal.polygonal.pbjp'
                          ):
        '''
        Documentation
        pxtal.L0.xtal_ble_val

        if length_type == 'xtal.polygonal.pbjp':
            lengths of all boundaries in each xtal

        if length_type == 'xtal.polygonal.perimeter':
            lengths of all boundaries in each xtal



            _.xtal_ble_val: list = [] # xtal boundary length exterior
            _.xtal_pe_val: list = [] # xtal perimeter exterior
        '''
        if level == 0:
            if length_type == 'xtal.polygonal.pbjp':
                #----------------------
                if self.L0.vt_base_tool == 'shapely':
                    xtal_coordinates = self.extract_shapely_coords(shapely_grains_list = None,
                                                                   coord_of = 'L0_xtal_vertices_pbjp_xtalwise',
                                                                   save_to_attribute = False,
                                                                   make_unique = False,
                                                                   throw = True
                                                                   )
                    _distances = []
                    for xy in xtal_coordinates:
                        x, y = np.array(xy[0]), np.array(xy[1])
                        _x, _y = np.roll(x, shift = -1, axis = 0), np.roll(y, shift = -1, axis = 0)
                        _distances.append(np.sqrt((x[:-1]-_x[:-1])**2 + (y[:-1]-_y[:-1])**2))
                        self.L0.xtal_ble_val = deepcopy(_distances)
                elif self.L0.vt_base_tool == 'scipy':
                    _distances = []
                    for _x, _y in zip(self.L0.xtal_coord_pbjp_x, self.L0.xtal_coord_pbjp_y):
                        _distances.append(np.array([np.sqrt((_x[i+1]-_x[i])**2 + (_y[i+1]-_y[i])**2) for i in range(len(_x)-1)]))
                    self.L0.xtal_ble_val = deepcopy(_distances)
                #----------------------
            if length_type == 'xtal.polygonal.perimeter':
                if hasattr(self.L0, 'xtal_ble_val'):
                    self.L0.xtal_pe_val = [_d.sum() for _d in self.L0.xtal_ble_val]
                else:
                    self.calculate_lengths(level = 0, length_type = 'xtal.polygonal.pbjp' )
                    self.calculate_lengths(level = 0, length_type = 'xtal.polygonal.perimeter' )
            if self.L0.vt_base_tool == 'shapely':
                try:
                    _ = self.lengths_gb_polygonal_exterior
                except AttributeError:
                    self.lengths_gb_polygonal_exterior = [_.exterior.length for _ in self.L0.xtals]
            elif self.L0.vt_base_tool == 'scipy':
                pass
        elif length_type == 'voxelated.gb':
            pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def calculate_areas(self,
                        area_type = 'polygonal'
                        ):
        '''
        Documentation
        '''
        if area_type == 'polygonal':
            if self.L0.vt_base_tool == 'shapely':
                if len(self.L0.xtal_ape_val) == 0:
                    self.L0.xtal_ape_val = np.array([_.area for _ in self.L0.xtals])
            if self.L0.vt_base_tool == 'scipy':
                try:
                    _ = self.locx_gvert
                except AttributeError:
                    self.build_vertices()
                finally:
                    areas = []
                    for x, y in zip(self.L0.xtal_coord_pbjp_x, self.L0.xtal_coord_pbjp_y):
                        # REF: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
                        # In the above link, refer to solution by maxb
                        # TO FIGURE OUT: Should x be closed for the following lines of codes
                        correction = x[-1]*y[0] - y[-1]*x[0]
                        main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
                        areas.append(0.5*np.abs(main_area + correction))
                    self.L0.xtal_ape_val = areas
        elif area_type == 'voxelated':
            self.areas_voxelated = None
            pass
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def calculate_perimeter(self,length_type = 'perimeter', level = 0):
        '''
        Documentation
        '''
        self.lengths_polygonal_exterior = [self.L0.xtals[0].exterior.length for xtal_count in range(self.L0.xtals_n)]
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def calc_distributions(self, data_name = 'area', ):
        '''
        Documentation
        '''
        from distr_01 import distribution
        self.areas_polygonal_exterior_distribution = distribution(data_name = data_name,
                                                                  data = pxt.areas_polygonal_exterior,
                                                                  )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def plot_histogram(self, data_name = 'area'):
        '''
        Documentation
        '''
        if data_name == 'area':
            self.areas_polygonal_exterior_distribution.plot_histogram()
   #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def identify_grains_from_field_threshold(self,
                                             field_name = 'areas_polygonal_exterior',
                                             threshold_definition = 'percentiles',
                                             threshold_limits_values = [[0.0, 0.05], [0.05, 0.10], [0.10, 0.15]],
                                             threshold_limits_percentiles = [[0, 10], [10, 90], [90, 100]],
                                             inequality_definitions = [['>=', '<='], ['>=', '<='], ['>=', '<=']],
                                             exclude_grains = None,
                                             save_as_attribute = True,
                                             throw = True
                                             ):
        '''
        This method helps extract the grain number and the actual values of field variable which
        fit the threshold specification
        ----CALL----:
        pxt.identify_grains_from_field_threshold(field_name = 'areas_polygonal_exterior',
                                                 threshold_definition = 'percentiles',
                                                 threshold_limits_values = [[0.0, 0.05], [0.08, 0.20]],
                                                 threshold_limits_percentiles = [[0, 10], [10, 90], [90, 100]],
                                                 inequality_definitions = [['>=', '<='], ['>=', '<='], ['>=', '<=']],
                                                 exclude_grains = [None]
                                                 )
        ----FIELD NAMES----
            n. NAME: DESCRIPTION: APPLICABLE LEVELS ---> see below
            1. a_pol: AREA POLYGONAL: L0, L1, L2, L3
            2. a_pix: AREA PIXELATED: L0, L1, L2, L3
            2. a_pol_ext: to replace areas_polygonal_exterior: L0, L1, L2, L3
            2. p_pol_ext: PERIMETER POLYGONAL EXTERIOR: L0 ,L1, L2, L3
            3. diag_max: MAXIMUM DIAGONAL: L0, L1, L2, L3
            4. diag_min: MINIMUM DIAGONAL: L0, L1, L2, L3
            5. eqdia_ext: EQUIVALENT DIAMETER EXTERNAL
        '''
        if field_name == 'areas_polygonal_exterior':
            _base_data = self.L0.xtal_ape_val
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        if field_name == 'ar_polygonal_exterior':
            _base_data = None # TODO
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        if field_name == 'gndd':
            _base_data = None # TODO
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        if field_name == 'sdv001':
            _base_data = None # TODO
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        if field_name == 'sdv002':
            _base_data = None # TODO
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        if field_name == 'meshdensity':
            _base_data = None # TODO
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        field_values, xtal_ids = [], []
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        if threshold_definition == 'percentiles':
            THRESHOLD = threshold_limits_percentiles
        elif threshold_definition == 'values':
            THRESHOLD = threshold_limits_values
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        for count, threshold_limits in enumerate(THRESHOLD):
            if threshold_definition == 'percentiles':
                tdef_values = [np.percentile(_base_data, _p) for _p in threshold_limits]
            elif threshold_definition == 'values':
                tdef_values = threshold_limits
            #--------------------------
            if inequality_definitions[count][0] == '>=':
                A = _base_data >= min(tdef_values)
            elif inequality_definitions[count][0] == '>':
                A = _base_data > min(tdef_values)
            #--------------------------
            if inequality_definitions[count][1] == '<=':
                B = _base_data <= max(tdef_values)
            elif inequality_definitions[count][1] == '<':
                B = _base_data < max(tdef_values)
            #--------------------------
            # TODO: REPLACE BY A LIST COMPREHENSION INSTEAD
            subset = []
            subset_ids = []
            for i, toconsider in enumerate(A & B):
                if toconsider:
                    subset.append(_base_data[i])
                    subset_ids.append(i)
            #--------------------------
            field_values.append(np.array(subset))
            xtal_ids.append(np.array(subset_ids))
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        if save_as_attribute:
            if hasattr(self.L0, 'grains_thresholded_fields'):
                self.L0.grains_thresholded_fields[field_name] = [xtal_ids, field_values]
            else:
                self.L0.grains_thresholded_fields = dict()
                self.L0.grains_thresholded_fields[field_name] = [xtal_ids, field_values]
        #  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
        if throw:
            return field_values, xtal_ids
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def define_cmap(self,
                    field = 'areas_polygonal_exterior',
                    par = {'map_type': 'polynomial',
                           'map_poly_order': 1,
                           'clr_floor': [0.0, 0, 0],
                           'clr_ceil': [1, 1, 1],
                           'clr_axis': [1, 1, 1],
                           }
                    ):
        '''
        pxt.define_cmap()
        # TODO: Includ the default cmap option available in matplotlib
        '''
        # .. .. .. .. .. .. .. .. .. .. .. .. .. .. ..
        # Determine normalized array based on the field of interest
        if field in ('areas_polygonal_exterior'):
            _ = np.array(self.L0.xtal_ape_val)
            field_min = min(_)
            field_max = max(_ - field_min)
            __ = (_ - field_min)/field_max
        else:
            # Just a dummy asignment. To change later accordingly
            __ = None
            pass
        # .. .. .. .. .. .. .. .. .. .. .. .. .. .. ..
        # Compute colours along colour axis - 1, 2, 3
        cmap = []
        for i in range(3):
            if par['clr_axis'][i] == 1:
                clr = deepcopy(__)
                if par['map_type'] == 'polynomial':
                    if par['map_poly_order'] == 1:
                        pass
                    else:
                        clr = clr**par['map_poly_order']
                for i, c in enumerate(__):
                    if c <= par['clr_floor'][0]:
                        clr[i] = par['clr_floor'][0]
                    elif c >= par['clr_ceil'][0]:
                        clr[i] = par['clr_ceil'][0]
            # Build the cmap list of numpy arrays
            cmap.append(clr)
        # .. .. .. .. .. .. .. .. .. .. .. .. .. .. ..
        return cmap
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def summary(self):
        _sep = ''.join([30*'*', '\n'])

        print(''.join([_sep,
                       'Level 0 \n',
                       f'Number of xtals: {self.L0.xtals_n} \n',
                       _sep,
                       f'Area.minimum: {round(self.L0.xtal_ape_dstr.S.minimum, 4)} \n'
                       f'Area.percentile: 10,50,90: {np.round(np.array(self.L0.xtal_ape_dstr.S.percentiles[1:-1]), 4)} \n'
                       f'Area.mean: {round(self.L0.xtal_ape_dstr.S.mean, 4)} \n',
                       f'Area.sdev: {round(self.L0.xtal_ape_dstr.S.sdev, 4)} \n',
                       f'Area.maximum: {round(self.L0.xtal_ape_dstr.S.maximum, 4)} \n',
                       _sep,
                       f''
                       ]
                      )
              )
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def _help_(self,
               topic = 'all'):
        '''
        A small cheat-sheet to help in PXTAL data access.
        '''
        if topic == 'all':
            self._help_('info')
            self._help_('top_level')
            self._help_('mpos')
            self._help_('coords')
            self._help_('ids')
            self._help_('specifics')
            self._help_('fields')
            self._help_('statistics')
        if topic == 'info':
            print(40*'/')
            print('pxtal: ukaea poly-xtal object, generally user-named as << pxtal >>.')
            print('       in all the below, this pxtal is the pxtal before L0.')
            print('       user defined name.')
        if topic == 'top_level':
            print(40*'/')
            print('Access top level data')
            print('      ......<< pxtal.L0, *L1, L2, L3 >> Data container for level-0')
            print(''.join(['      ......<<  pxtal.L0.pxtal >> PXTAL object, ..',
                           '-           ..pxtal at end: standard name']))
            print('       pxtal bounds << pxtal.L0.xbound, *ybound >>')
        if topic == 'mpos':
            print(40*'/')
            print('Access MPOs:')
            print('      ......<< pxtal.L0.mpo_seeds >>')
            print('      ......<< pxtal.L0.mpo_xtals_centroids >>')
            print('      ......<< pxtal.L0.mpo_xtals_reppoints >>')
            print('      ......<< pxtal.L0.mpo_xtals_vertices_unique >>')
        if topic == 'coords':
            print('Access coordinates of the pxtal: ')
            print('      ......<< pxtal.L0.coord_seeds_x, *y >>') # Getters done
            print('      ......<< pxtal.L0.xtal_coord_centroid_x, *y, *xy >>')# Getters done
            print('      ......<< pxtal.L0.xtal_coord_reppoint_x, *y, *xy >>')# Getters done
            print('      ......<< pxtal.L0.xtal_coord_vertices_x, *y, *xy >>')# Getters done
            print(''.join(['      ......<< boundaries_cw >> boundaries data from.. \n',
                           '            ..the xbound and ybound arrays arranged clockwise',
                           ]
                          )
                  )
            print(40*'/')
        if topic == 'ids':
            print('Access IDs of all xtals from << pxtal.L0.xtals_ids >>\n')
            print('    ~~~ pxtal.get_ids_L0_all_xtals() ~~~') # Getters done. # Hassers done
        if topic == 'specifics':
            print('Access specific data')
            print('      ...... pxtal boundary & internal xtal data << pxtal.L0.xtal_ss_boundary, *internal >>')
            print(40*'/')
        if topic == 'fields':
            print('Access fields')
            print('       pxtal area  << pxtal.area_pxtal >>')
            print('      << xtal_ape_val >> xtal Area polygonal exterior - values')
            print('      << xtal_ape_dstr >> xtal Area polygonal exterior - distribution')
            print('      << xtal_ble_val >> xtal boundary length exterior - values')
            print('      << xtal_ble_dstr >> xtal boundary length exterior - distribution')
            print('      << xtal_pe_val >> xtal perimeter exterior - values')
            print('      << xtal_pe_dstr >> xtal perimeter exterior - distribution')
            print('      << xtal_bjp_n >> Number of bundary junction points in each xtal')
            print('      << xtal_tpj_n >> Number of triple point junctions')
            print('      << xtal_jpia_val >> Junction point internal angles list')
            print(40*'/')
        if topic == 'statistics':
            print('Access statistics')
            print('      << pxtal.L0.xtal_ape_dstr >>')
            print('      << pxtal.L0.xtal_ape_dstr.H >> histogram')
            print('      << pxtal.L0.xtal_ape_dstr.S >> summary')
            print('      << pxtal.L0.xtal_ape_dstr.K >> kde')
            print('      << pxtal.L0.xtal_ape_dstr.H.be >> bin edges')
            print('      << pxtal.L0.xtal_ape_dstr.H.hv >> histogram values')
            print('      << pxtal.L0.xtal_ape_dstr.H.nbins >> number of bins')
            print('      << pxtal.L0.xtal_ape_dstr.H.data >> base data')
            print('      << pxtal.L0.xtal_ape_dstr.K.bw >> bandwidth')
            print('      << pxtal.L0.xtal_ape_dstr.K.kd >> kernel density')
            print(40*'/')
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def plot(self,
             renderer = 'matplotlib', # pyvista
             database = 'shapely',
             figsize = [3.5, 3.5],
             dpi = 100,
             level = 0,
             default_par_faces = {'clr': 'teal',
                                  'alpha': 0.2,
                                  },
             default_par_lines = {'width': 0.5,
                                  'clr': 'black',
                                  },
             default_par_points = {'marker': 'o',
                                   'size': 5,
                                   'eclr': 'black',
                                   'fclr': 'yellow'
                                   },
             xtal_marker_centroid = False,
             xtal_marker_reppoint = False,
             xtal_marker_vertex = False,
             xtal_marker_par = {'L0_xtals_centroids': ['+', 5, 0.5, 'black', 'black'], # marker, size, linewidth, ecolour, face clr,
                                'L0_xtals_reppoints': ['x', 5, 0.5, 'black', 'red'],
                                'vertex': ['o', 10, 0.5, 'black', 'yellow'],
                                },
             xtal_face_annot_count    = False,
             xtal_face_annot_centroid = False,
             xtal_face_annot_reppoint = False,
             xtal_face_annot_vertex = False,
             xtal_face_annot_par = {'count': [8, 'black', 'normal'],
                                    'L0_xtals_centroids': [3, 'red', 'normal'], # fsize, fclr, fweight
                                    'L0_xtals_reppoints': [3, 'green', 'normal'],
                                    'vertex': [3, 'blue', 'normal'],
                                    },
             xtal_line_annot_count    = False,
             xtal_line_annot_centroid = False,
             xtal_line_annot_reppoint = False,
             xtal_line_annot_vertex = False,
             xtal_line_annot_par = {'count': [10, 'black', 'bold'],
                                    'L0_xtals_centroids': [10, 'red', 'normal'], # fsize, fclr, fweight
                                    'L0_xtals_reppoints': [10, 'green', 'normal'],
                                    'vertex': [10, 'blue', 'normal'],
                                    },
             xbound_cmap_field = 'length', # length, srmax (i.e. surf.rough.max), srmean, etc..
             grain_neigh = {0: None, # neigh_0
                            1: None, # neigh_1_
                            2: None  # neigh_2_
                            },
             xtal_clr_field = False,
             field_variable = 'areas_polygonal_exterior',
             clr_map_par = {'map_type': 'polynomial',
                            'map_poly_order': 1,
                            'clr_floor': [0.0, 0, 0],
                            'clr_ceil': [1, 1, 1],
                            'clr_axis': [1, -1, 0]
                            },
             highlight_specific_grains = False,
             specific_grains = [0],
             highlight_specific_grains_annot = False,
             highlight_specific_grains_colour = False,
             highlight_specific_grains_hatch = True,
             plot_neigh_0 = False,
             plot_neigh_1 = False,
             plot_neigh_2 = False,
             colour_neigh_0 = True,
             colour_neigh_1 = True,
             colour_neigh_2 = True,
             neigh_0 = None,
             neigh_1_ = None,
             neigh_2_ = None,
             xtal_neigh_0_annot_count = False,
             xtal_neigh_1_annot_count = False,
             xtal_neigh_2_annot_count = False,
             xtal_neigh_0_par = {'clr': 'red',
                                 'alpha': 0.5,
                                 },
             xtal_neigh_1_par = {'clr': 'green',
                                 'alpha': 0.5,
                                 },
             xtal_neigh_2_par = {'clr': 'blue',
                                 'alpha': 0.5,
                                 },
             ):
        '''
        Documentation
        '''
        #------------------------------------------------------
        if xtal_clr_field:
            clr = self.define_cmap(field = field_variable,
                                   par = clr_map_par
                                   )
        #------------------------------------------------------
        import matplotlib.pyplot as plt
        plt.figure(figsize = (figsize[0], figsize[1]),
                   dpi = dpi,
                   )
        #------------------------------------------------------
        if xtal_marker_centroid:
            # Make centroid data if it does not exit
            try:
                _ = self.L0.xtal_coord_centroid_x
            except AttributeError:
                self.L0.xtal_coord_centroid_x, self.L0.xtal_coord_centroid_y = self.extract_shapely_coords(coord_of = 'L0_xtals_centroids')
        #------------------------------------------------------
        if xtal_marker_reppoint:
            # Make reppoint data if it does not exit
            try:
                _ = self.L0.xtal_coord_reppoint_x
            except AttributeError:
                self.L0.xtal_coord_reppoint_x, self.L0.xtal_coord_reppoint_y = self.extract_shapely_coords(coord_of = 'L0_xtals_reppoints')
        #------------------------------------------------------
        if xtal_marker_vertex:
            # Make reppoint data if it does not exit
            if level == 0:
                try:
                    _ = self.locx_g_vertex
                except AttributeError:
                    self.build_vertices()
        #------------------------------------------------------
        if database == 'shapely':
            # Plot each grain
            for i, g in enumerate(self.L0.xtals):
                # Figure out thew colour of the grains first
                if xtal_clr_field:
                    _color = [clr[0][i], clr[1][i], clr[2][i]]
                    _alpha = 1.0
                else:
                    _color = default_par_faces['clr']
                    _alpha = default_par_faces['alpha']
                # Do the actual plotting
                plt.fill(g.boundary.xy[0],
                         g.boundary.xy[1],
                         color = _color,
                         linewidth = default_par_lines['width'],
                         edgecolor = default_par_lines['clr'],
                         alpha = _alpha
                         )
            # .. .. .. .. .. .. .. .. .. .. .. ..
            if highlight_specific_grains:
                import random
                # Define a list of colours
                colours = ['lightgreen', 'green', 'red', 'lightblue', 'blue',
                           'orange', 'maroon', 'teal'
                           ]
                # Define a list of hatch types for grains to be highlighted
                # Courtsey: https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html
                hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*',
                           '//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**',
                           '/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
                for i in specific_grains:
                    if highlight_specific_grains_colour and highlight_specific_grains_hatch:
                        print('###############################')
                        plt.fill(self.L0.xtals[i].boundary.xy[0],
                                 self.L0.xtals[i].boundary.xy[1],
                                 hatch = random.choice(hatches),
                                 color = random.choice(colours),
                                 edgecolor = 'black',
                                 linewidth = 1.0
                                 )
                    if highlight_specific_grains_colour * highlight_specific_grains_hatch == 0:
                        if highlight_specific_grains_colour:
                            plt.fill(self.L0.xtals[i].boundary.xy[0],
                                     self.L0.xtals[i].boundary.xy[1],
                                     color = random.choice(colours),
                                     edgecolor = 'black',
                                     linewidth = 1.0
                                     )
                        if highlight_specific_grains_hatch:
                            plt.fill(self.L0.xtals[i].boundary.xy[0],
                                     self.L0.xtals[i].boundary.xy[1],
                                     hatch = random.choice(hatches),
                                     facecolor = 'none',
                                     edgecolor = 'black',
                                     linewidth = 1.0,
                                     )
                    if highlight_specific_grains_annot:
                        rp = self.L0.xtals[i].representative_point().xy
                        plt.text(rp[0][0], rp[1][0], str(i),
                                 fontsize = xtal_face_annot_par['count'][0]+4,
                                 color = 'black',
                                 fontweight = 'bold',
                                 rotation = 0.0,
                                 ha="center",
                                 va="center",
                                 bbox=dict(boxstyle = "square",
                                           ec = 'white',
                                           fc = 'white',
                                           alpha = 0.9
                                           )
                                 )
            # .. .. .. .. .. .. .. .. .. .. .. ..
            if plot_neigh_2:
                for i in neigh_2_:
                    for j in i:
                        grain = self.L0.xtals[j]
                        if colour_neigh_2:
                            plt.fill(grain.boundary.xy[0],
                                     grain.boundary.xy[1],
                                     color = xtal_neigh_2_par['clr'],
                                     linewidth = default_par_lines['width'],
                                     edgecolor = default_par_lines['clr'],
                                     alpha = xtal_neigh_2_par['alpha'],
                                     )
                        if xtal_neigh_2_annot_count:
                            rp = grain.representative_point().xy
                            plt.text(rp[0][0], rp[1][0], str(j),
                                     fontsize = xtal_face_annot_par['count'][0],
                                     color = 'darkblue',
                                     fontweight = 'bold',
                                     )
            # .. .. .. .. .. .. .. .. .. .. .. ..
            if plot_neigh_1:
                for i in neigh_1_:
                    for j in i:
                        grain = self.L0.xtals[j]
                        if colour_neigh_1:
                            plt.fill(grain.boundary.xy[0],
                                     grain.boundary.xy[1],
                                     color = xtal_neigh_1_par['clr'],
                                     linewidth = default_par_lines['width'],
                                     edgecolor = default_par_lines['clr'],
                                     alpha = xtal_neigh_1_par['alpha'],
                                     )
                        if xtal_neigh_1_annot_count:
                            rp = grain.representative_point().xy
                            plt.text(rp[0][0], rp[1][0], str(j),
                                     fontsize = xtal_face_annot_par['count'][0]+2,
                                     color = 'darkgreen',
                                     fontweight = 'bold',
                                     )
            # .. .. .. .. .. .. .. .. .. .. .. ..
            if plot_neigh_0:
                for i in neigh_0[0]:
                    if colour_neigh_0:
                        plt.fill(self.L0.xtals[i].boundary.xy[0],
                                 self.L0.xtals[i].boundary.xy[1],
                                 color = xtal_neigh_0_par['clr'],
                                 linewidth = default_par_lines['width'],
                                 edgecolor = default_par_lines['clr'],
                                 alpha = xtal_neigh_0_par['alpha'],
                                 )
                    if xtal_neigh_0_annot_count:
                        rp = self.L0.xtals[i].representative_point().xy
                        plt.text(rp[0][0], rp[1][0], str(i),
                                 fontsize = xtal_face_annot_par['count'][0]+4,
                                 color = 'darkred',
                                 fontweight = 'bold',
                                 )
        #------------------------------------------------------
        if xtal_face_annot_count:
            try:
                _ = self.L0.xtal_coord_reppoint_x
            except AttributeError:
                self.L0.xtal_coord_reppoint_x, self.L0.xtal_coord_reppoint_y, _ = self.extract_shapely_coords(coord_of = 'L0_xtals_reppoints')
            for i, (xc, yc) in enumerate(zip(self.L0.xtal_coord_reppoint_x, self.L0.xtal_coord_reppoint_y)):
                plt.text(xc, yc, str(i),
                         fontsize = xtal_face_annot_par['count'][0],
                         color = xtal_face_annot_par['count'][1],
                         fontweight = xtal_face_annot_par['count'][2],
                         )
        #------------------------------------------------------
        if xtal_marker_centroid:
            plt.scatter(self.L0.xtal_coord_centroid_x,
                        self.L0.xtal_coord_centroid_y,
                        marker = xtal_marker_par['L0_xtals_centroids'][0],
                        s = xtal_marker_par['L0_xtals_centroids'][1],
                        linewidths = xtal_marker_par['L0_xtals_centroids'][2],
                        edgecolor = xtal_marker_par['L0_xtals_centroids'][3],
                        c = xtal_marker_par['L0_xtals_centroids'][4],
                        )
        #------------------------------------------------------
        if xtal_marker_reppoint:
            plt.scatter(self.L0.xtal_coord_reppoint_x,
                        self.L0.xtal_coord_reppoint_y,
                        marker = xtal_marker_par['L0_xtals_reppoints'][0],
                        s = xtal_marker_par['L0_xtals_reppoints'][1],
                        linewidths = xtal_marker_par['L0_xtals_reppoints'][2],
                        edgecolor = xtal_marker_par['L0_xtals_reppoints'][3],
                        c = xtal_marker_par['L0_xtals_reppoints'][4],
                        )
        #------------------------------------------------------
        if xtal_marker_vertex:
            plt.scatter(self.L0.xtal_coord_vertices_x,
                        self.L0.xtal_coord_vertices_y,
                        marker = xtal_marker_par['vertex'][0],
                        s = xtal_marker_par['vertex'][1],
                        linewidths = xtal_marker_par['vertex'][2],
                        edgecolor = xtal_marker_par['vertex'][3],
                        c = xtal_marker_par['vertex'][4],
                        )
        #------------------------------------------------------
        plt.show()
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def SFV_assign_polygonal_areas_to_reppoints(self):
        for _xtal_, _point_ in zip(self.L0.xtals, self.L0.mpo_xtals_reppoints.points):
            _point_.polygonal_area = _xtal_.area
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def plot_neigh_0(self):
        '''
        Documentation
        '''
        pass
    def plot_neigh_1(self):
        '''
        Documentation
        '''
        pass
    def plot_neigh_2(self):
        '''
        Documentation
        '''
        pass
    def get_L0_ng(self):
        return len(self.L0.xtals_ids)
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # IDS OF THE FEATURES
    def get_ids_L0_all_xtals(self):
        return self.L0.xtals_ids
    def get_ids_L0_boundary_xtals(self):
        return self.L0.xtals_boundary.ids

    # COUNT OF THE FEATURES
    def get_n_L0_boundary_xtals(self):
        return self.L0.xtals_boundary.n
    def get_n_L0_internal_xtals(self):
        return self.internal_xtals.n
    def get_xnBE_L0_boundary_xtals(self):
        #xtal number of boundary edges
        return self.L0.xtals_boundary.nBE
    def get_nBE_L0_internal_xtals(self):
        return self.internal_xtals.nBE

    # SCALAR FIELD - ALL VALUES
    def get_L0x_pae(self, ):
        # L0-xtal-polygonal-area-exterior
        return self.areas_polygonal_exterior
    def get_xea_L0_boundary(self):
        return self.L0.xtals_boundary.APE_val
    def get_xbp_L0_boundary_xtals(self):
        pass
    # THRESHOLDED XTALS
        # 1. by area quartiles
        # 2. by perimeter quartiles
        # 3. by quartiles on the aspect ratios
        # 4. by quartiles on the mean number of edges
        # 5. by quartiles of the min angle
        # 6. by quartiles of the mean angle
        # 7. by quartiles of the max angle
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def get_xcoord_vtseeds(self, level = 0, instance = 0):
        '''Instance to be included later. THIS APPLIES TO ALL SIMILAR GETTERS'''
        if level == 0:
            return self.L0.coord_seeds_x
    def get_ycoord_vtseeds(self, level = 0, instance = 0):
        if level == 0:
            return self.L0.coord_seeds_y
    def get_xycoord_vtseeds(self, level = 0, instance = 0):
        if level == 0:
            if hasattr(self.L0, 'xtal_coord_centroid_xy'):
                return self.L0.coord_seeds_xy
            else:
                print('Stacked array does not exit')
                print('Returning x and y arrays seperately')
                _1 = self.get_xcoord_vtseeds(self, level = level, instance = instance)
                _2 = self.get_ycoord_vtseeds(self, level = level, instance = instance)
                return _1, _2
    def get_xcoord_centroid(self, level = 0, instance = 0):
        '''Instance to be included later. THIS APPLIES TO ALL SIMILAR GETTERS'''
        if level == 0:
            return self.L0.xtal_coord_centroid_x
    def get_ycoord_centroid(self, level = 0, instance = 0):
        if level == 0:
            return self.L0.xtal_coord_centroid_y
    def get_xycoord_centroid(self, level = 0, instance = 0):
        if level == 0:
            if hasattr(self.L0, 'xtal_coord_centroid_xy'):
                return self.L0.xtal_coord_centroid_xy
            else:
                print('Stacked array does not exit')
                print('Returning x and y arrays seperately')
                _1 = self.get_xcoord_centroid(self, level = level, instance = instance)
                _2 = self.get_ycoord_centroid(self, level = level, instance = instance)
                return _1, _2
    def get_xcoord_reppoint(self, level = 0, instance = 0):
        '''Instance to be included later. THIS APPLIES TO ALL SIMILAR GETTERS'''
        if level == 0:
            return self.L0.xtal_coord_reppoint_x
    def get_ycoord_reppoint(self, level = 0, instance = 0):
        if level == 0:
            return self.L0.xtal_coord_reppoint_y
    def get_xycoord_reppoint(self, level = 0, instance = 0):
        if level == 0:
            if hasattr(self.L0, 'xtal_coord_reppoint_xy'):
                return self.L0.xtal_coord_reppoint_xy
            else:
                print('Stacked array does not exit')
                print('Returning x and y arrays seperately')
                _1 = self.get_xcoord_reppoint(self, level = level, instance = instance)
                _2 = self.get_ycoord_reppoint(self, level = level, instance = instance)
                return _1, _2
    def get_xcoord_vertices(self, level = 0, instance = 0):
        '''Instance to be included later. THIS APPLIES TO ALL SIMILAR GETTERS'''
        if level == 0:
            return self.L0.xtal_coord_vertices_x
    def get_ycoord_vertices(self, level = 0, instance = 0):
        if level == 0:
            return self.L0.xtal_coord_vertices_y
    def get_xycoord_vertices(self, level = 0, instance = 0):
        if level == 0:
            if hasattr(self.L0, 'xtal_coord_vertices_xy'):
                return self.L0.xtal_coord_vertices_xy
            else:
                print('Stacked array does not exit')
                print('Returning x and y arrays seperately')
                _1 = self.get_xcoord_reppoint(self, level = level, instance = instance)
                _2 = self.get_ycoord_reppoint(self, level = level, instance = instance)
                return _1, _2
    def get_ids_xtals(self, level = 0, instance = 0):
        return self.L0.xtals_ids
    def has_ids_xtals(self, level = 0, instance = 0, ids = [0, 70, 71, 58, 72]):
        all_ids = get_ids_xtals(level = level, instance = instance)
        return [False if id_ not in all_ids else True for id_ in ids]
    def get_dco_xtals_boundary(self, level = 0, instance = 0):
        # dc: data container object
        if level == 0:
            return self.L0.xtal_ss_boundary
    def get_dcostat_area_xtals_boundary(self, level = 0, instance = 0):
        return self.get_dco_xtals_boundary(level = level,
                                           instance = instance).APE_distr
    def get_dcostat_area_xtals_boundary(self, level = 0, instance = 0, measure = 'ape'):
        if level == 0:
            if measure == 'ape':
                return self.get_dco_xtals_boundary(level = level,
                                                   instance = instance).APE_distr
    def get_area_xtals_boundary(self, level = 0, instance = 0, measure = 'ape'):
        if level == 0:
            if measure == 'ape':
                return self.get_dco_xtals_boundary(level = level,
                                                   instance = instance).APE_val
    def get_stat_H_area_xtals_boundary(self, level = 0, instance = 0, measure = 'ape'):
        return self.get_dcostat_area_xtals_boundary(level = level,
                                                    instance = instance,
                                                    measure = measure).H
    def get_stat_K_area_xtals_boundary(self, level = 0, instance = 0, measure = 'ape'):
        return self.get_dcostat_area_xtals_boundary(level = level,
                                                    instance = instance,
                                                    measure = measure).K
    def get_stat_S_area_xtals_boundary(self, level = 0, instance = 0, measure = 'ape'):
        return self.get_dcostat_area_xtals_boundary(level = level,
                                                    instance = instance,
                                                    measure = measure).S
    def get_stat_nbins_area_xtals_boundary(self, level = 0, instance = 0):
        return self.get_stat_H_area_xtals_boundary(level = level,
                                                   instance = instance,
                                                   measure = measure).nbins
    def get_stat_be_area_xtals_boundary(self, level = 0, instance = 0):
        return self.get_stat_H_area_xtals_boundary(level = level,
                                                   instance = instance,
                                                   measure = measure).be
    def get_stat_hv_area_xtals_boundary(self, level = 0, instance = 0):
        return self.get_stat_H_area_xtals_boundary(level = level,
                                                   instance = instance,
                                                   measure = measure).hv





#//////////////////////////////////////////////////////////////////////////////
