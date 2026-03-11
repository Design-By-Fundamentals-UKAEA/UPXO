'''
UPXO is a python package to generate, analyze, manipulate, visualize, mesh and
export heirarchical multi-instanced spatially gradient multi-phase
poly-crystalline microstructures in 2 and 3 dimensions.

This module is a core to UPXO and has the following classes:
    point2d, point3d
    point2d_lean_highest, point3d_lean_highest
    point2d_lean_highest_mc0, point3d_lean_highest_mc0
    point2d_lean_highest_mc1, point3d_lean_highest_mc1
    point_q_space, point_be_space, point_gsm_stat_space

NOTE: NOT TO BE SHARED WITH ANYONE OTHER THAN:
    *@UKAEA: Vaasu Anandatheertha, Chris Hardie, Vikram Phalke
    *@UKAEA:  Ben Poole, Allan Harte, Cori Hamelin
    *@OX,UKAEA:  Eralp Demir, Ed Tarleton
'''

# from numpy.random import uniform as nprandu
from shapely.geometry import Point
from upxo._sup import dataTypeHandlers as dth
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from math import ceil, floor
# import random
__name__ = "UPXO-geoEntity"
__lead_developers__ = ["Dr. Vaasu Anandatheertha"]
__developers__ = ["Vaasu Anandatheertha (vaasu.anandatheertha@ukaea.uk)",
                  ]
__maintainers__ = ["Vaasu Anandatheertha (vaasu.anandatheertha@ukaea.uk)",
                   ]
__version__ = ["0.1.upto.271022.git-no", "0.2.from.281022.git-no",
               "0.3.from.031122.git-no", "0.4.from.111122.git-no",
               "0.5.from.021222.git-no", "0.6.from.081222.git-no",
               "0.7.from.221222.git-no", ""
               ]
__license__ = "GPL v3"


class point2d():
    '''Class representing point2d object in physical space.
    '''
    ROUND_ZERO_DEC_PLACE = 10
    ε = 0.000000000001
    EPS = 0.000000000001
    EPS_above = EPS
    EPS_below = EPS
    EPS_left = EPS
    EPS_right = EPS
    EPS_divisor = EPS

    __slots__ = ('dim',  # Dimension of the model
                 'lean',  # Complexity branching
                 'mid',  # memory id of the point
                 'loc',  # Location of the point in pxtal
                 'ptype',  # point type
                 'jn',  # Order of the junction point

                 'x', 'y',  # x and y - coordinates
                 'angle',  # angle in degrees, anti-clockwise +
                 'angles',  # list of angles in degrees, anti-clockwise +

                 'phase_id',  # SFVM -- Phase id
                 'phase_name',  # SFVM -- Phase name. Defaults to CuCrZr
                 'tcname',  # SFVM -- texture component name
                 'tdist',  # for use in comparison operations

                 '_mulpoints_',  # list of attached multi-point objects
                 '_edges_',  # list of attached edge objects
                 '_muledges_',  # list of attached multi-edge objects
                 '_xtals_',  # list of attached partition objects
                 '_polyxtals_',  # list of attached poly-partition objects
                 '_original_location',  # original x and y

                 'sfv_pol_area',  # SFVM -- Polygonal area
                 'sfv_pol_perimeter',  # SFVM -- Polygonal length
                 'sfv_pol_aspect_ratio',  # Aspect ratio of the partiion
                 'sfv_repr_ea',  # SFVM -- Euler angle repr (ex: 'Bunge')
                 'sfv_ea',  # SFVM -- Euler angle in degrees
                 'orientation_object',  # SFVM -- UPXO Orientation object
                 'partition',  # Dataclass instance to store buffer polygons
                 'vprop',  # Visialuzation properties
                 'image_sh',  # External image: Shapely
                 'image_pv',  # External image: Py-vista
                 'image_vtk',  # External image: VTK
                 'image_gmsh'  # External image: GMSH
                 'mesh_lc',  # Target GMSH mesh size [Char. length]
                 )

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 lean: str = 'lowest',
                 set_tdist: bool = True, tdist: float = 0.0000000000001,
                 set_mid: bool = False,
                 set_dim: bool = False, dim: int = 2,
                 set_ptype: bool = False, ptype: str = 'vt2dseed',
                 set_jn: bool = False, jn: int = 3,
                 set_loc: bool = False, loc: str = 'interior',
                 store_original_coord: bool = False,
                 set_mesh_lc: bool = False, mesh_lc=1.0,
                 attach_mulpoints: bool = False, mulpoints=None,
                 attach_edges: bool = False, edges=None,
                 attach_muledges: bool = False, muledges=None,
                 attach_xtals: bool = False, xtals=None,
                 attach_polyxtals: bool = False, polyxtals=None,
                 set_phase: bool = False,
                 phase_id: int = 1, phase_name: str = 'cucrzr',
                 set_sfv_pol_area: bool = True, sfv_pol_area: float = 0.0,
                 set_sfv_pol_perimeter: bool = True,
                 sfv_pol_perimeter: float = 0.0,
                 set_sfv_pol_aspect_ratio: bool = True,
                 sfv_pol_aspect_ratio: float = 0.0,
                 set_tcname: bool = False, tcname: str = 'B',
                 set_sfv_ea: bool = True,
                 sfv_repr_ea='Bunge', sfv_ea: list = [45, 35, 0],
                 set_orientation_object: bool = True,
                 orientation_object: object = None,
                 store_vis_prop: bool = False,
                 make_image_sh: bool = True, image_sh=None,
                 make_image_pv: bool = True, image_pv=None,
                 make_image_vtk: bool = True, image_vtk=None,
                 make_image_gmsh: bool = True, image_gmsh=None,
                 ):
        '''
        # TODO: Replace rare input arguments as kwargs

        point2d object defines a UPXO geometric location in 2D physical space
        Notes - 1
        ---------
        In all the below, the following holds:
            @user: something which user has to provide as an input
            @upxo: internally done by upxo. there is no need for
                   the user to remember
            @upxo, user: something which the user may also provide.
        '''
        self.x, self.y = float(x), float(y)
        if lean.lower() in ('ignore', 0):
            '''Do everything possible'''
            self.set_lean(lean)
            self.make_mid()
            self.set_dim()
            self.set_tdist(tdist)
            self.set_ptype(ptype=ptype)
            self.set_jn(jn=jn)
            self.set_location(loc=loc)
            self.set_original_location()
            self.set_phase(phase_id=phase_id, phase_name=phase_name)
            self.set_sfv_pol_area(sfv_pol_area)
            self.set_sfv_pol_perimeter(sfv_pol_perimeter)
            self.set_tcname(tcname)
            self.set_sfv_repr_ea(sfv_repr_ea)
            self.set_sfv_ea(sfv_ea)
            self.orientation_object = orientation_object
            self.set_mesh_lc(mesh_lc)
            self.vprop = self.set_vis_prop()
            self.attach_mulpoints(mulpoints)
            self.attach_edges(edges)
            self.attach_muledges(muledges)
            self.attach_xtals(xtals)
            self.attach_polyxtals(polyxtals)
            # self.make_partition(restart=False,
            #                    saa=True,
            #                    n=partition_n,
            #                    char_lengths=partition_char_lengths,
            #                    char_length_type=partition_char_length_type,
            #                    make_polygon=partition_make_polygon,
            #                    tool=partition_tool,
            #                    rotation=partition_rotation,
            #                    feed_partition=False,
            #                    feed_partition_name='grain',
            #                    feed_partition_tool='shapely',
            #                    feed_partition_polygon=None
            #                    )

        if lean.lower() in ('no', 'notlean', 'not_lean', 'lowest'):
            self.set_lean(lean)
            if set_mesh_lc:
                self.set_mesh_lc(mesh_lc)
            if set_mid:
                self.make_mid()
            if set_tdist:
                self.set_tdist()
            if set_dim:
                self.set_dim()
            if set_ptype:
                self.set_ptype(ptype=ptype)
            if set_jn:
                self.set_jn(jn=jn)
            if set_loc:
                self.set_location(loc=loc)
            if store_original_coord:
                self.set_original_location()
            if attach_mulpoints:
                self.attach_mulpoints(mulpoints)
            if attach_edges:
                self.attach_edges(edges)
            if attach_muledges:
                self.attach_muledges(muledges)
            if attach_xtals:
                self.attach_xtals(xtals)
            if attach_polyxtals:
                self.attach_polyxtals(polyxtals)
            if set_phase:
                self.set_phase(phase_id=phase_id, phase_name=phase_name)
            if set_sfv_pol_area:
                self.set_sfv_pol_area(sfv_pol_area)
            if set_sfv_pol_perimeter:
                self.set_sfv_pol_perimeter(sfv_pol_perimeter)
            if set_tcname:
                self.set_tcname(tcname)
            if set_sfv_ea:
                self.set_sfv_repr_ea(sfv_repr_ea)
                self.set_sfv_ea(sfv_ea)
            if set_orientation_object:
                self.set_orientation_object(orientation_object)
            if store_vis_prop:
                self.vprop = self.set_vis_prop()

        if lean in ('low'):
            self.lean = lean
            if set_mid:
                self.make_mid()
            if set_dim:
                self.set_dim()
            if set_ptype:
                self.set_ptype(ptype=ptype)
            if set_jn:
                self.set_jn(jn=jn)
            if set_loc:
                self.set_location(loc=loc)
            if store_original_coord:
                self._original_location = (self.x, self.y)
            if attach_mulpoints:
                pass
            if attach_edges:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if set_phase:
                self.set_phase(phase_id=phase_id, phase_name=phase_name)
            if set_sfv_pol_area:
                self.sfv_pol_area = sfv_pol_area
            if set_tcname:
                self.set_tcname(tcname)
            if set_sfv_ea:
                self.sfv_repr_ea = sfv_repr_ea
                self.sfv_ea = [45, 35, 0]
            if set_orientation_object:
                self.orientation_object = orientation_object
            if set_tdist:
                self.tdist = tdist
            if store_vis_prop:
                self.vprop = self.set_vis_prop()

        if lean.lower() in ('medium', 'intermediate'):
            self.lean = lean

        if lean.lower() in ('high'):
            self.lean = lean

        if lean.lower() in ('veryhigh', 'highest', 'very_high',
                            'leanest', 'fuly_lean', 'leanfull'):
            pass

    def __call__(self, x, y):
        '''Call point object instances like functions'''
        self.x, self.y = x, y

    def __eq__(self,
               point_objects: list = None,
               tdist: float = 0.0,
               use_self_tdist: bool = True,
               point_type: str = 'upxo'
               ):
        '''Check if self is coincident with "point_objects" (see attr. below).

        Parameters
        ----------
        point_objects : list / tuple / deque
            A list / tuple / deque of point objects
        tdist : float
            User value for tolerance distance.
        use_self_tdist : bool
            If True, `self.tdist` will be used as tolerance distance, else
            user value for `tdist` will be used for tolerance distance

        Returns
        -------
        True if distance b/w self and points is <= tolerance distance,
        defined in "tdist"

        Examples
        --------
        >>> from upoint import point2d
        >>> p1 = point2d(x = 0.0, y = 0.0)
        '''
        if use_self_tdist:
            tdist = self.tdist
        if isinstance(point_objects, point2d):
            if self.distance(other_object_type='point2d',
                             point_data=point_objects) <= tdist:
                return True
            else:
                return False
        elif isinstance(point_objects, list):
            to_return = [False for pobjcount in range(len(point_objects))]
            for pobjcount in range(len(point_objects)):
                if self.distance(point_objects[pobjcount]) <= tdist:
                    to_return[pobjcount] = True
            return tuple(to_return)
        else:
            return 'ERROR: Only point2d instance or tuple of them allowed'

    def __ne__(self,
               point_objects=None,
               tdist=0.1,
               use_self_tdist=True,
               point_type: str = 'upxo'
               ):
        '''
        True if distance b/w self and point_object is GT tolerance distance
        '''
        if use_self_tdist:
            tdist = self.tdist
        if isinstance(point_objects, point2d):
            if self.distance(other_object_type='point2d',
                             point_data=point_objects) > tdist:
                return True
            else:
                return False
        elif isinstance(point_objects, list):
            to_return = [False for pobjcount in point_objects]
            for i, point in enumerate(point_objects):
                if self.distance(point) > tdist:
                    to_return[i] = True
            # to_return = [False for pobjcount in range(len(point_objects))]
            # for pobjcount in range(len(point_objects)):
            #    if self.distance(point_objects[pobjcount]) > tdist:
            #        to_return[pobjcount] = True
            return tuple(to_return)
        else:
            return 'ERROR: Only point2d instance or tuple of them allowed'

    def above(self, point_objects):
        """
        Returns Truth array as per elements in point_objects.
        True if other object is above, else False in any other case

        Parameters
        ----------
        point_objects : point2d / shapely point / coord_xy list
                       / coord_xy tuple
            DESCRIPTION. An input set of points to compare against

        Returns
        -------
        list of Boolean (truth values)
        """
        point_objects_coord_y = dth.point_list_to_coordxy(point_objects).T[1]
        self_y = self.y + self.EPS_above
        above_flags = list(point_objects_coord_y > self_y)
        return above_flags

    def below(self, point_objects):
        """
        Returns Truth array as per elements in point_objects.
        True if other object is below, else False in any other case

        Parameters
        ----------
        point_objects : point2d / shapely point / coord_xy list
                       / coord_xy tuple
            DESCRIPTION. An input set of points to compare against

        Returns
        -------
        list of Boolean (truth values)
        """
        point_objects_coord_y = dth.point_list_to_coordxy(point_objects).T[1]
        self_y = self.y - self.EPS_above
        below_flags = list(point_objects_coord_y < self_y)
        return below_flags

    def left(self, point_objects):
        """
        Returns Truth array as per elements in point_objects.
        True if other object is to the left, else False in any other case

        Parameters
        ----------
        point_objects : point2d / shapely point / coord_xy list
                       / coord_xy tuple
            DESCRIPTION. An input set of points to compare against

        Returns
        -------
        list of Boolean (truth values)
        """
        point_objects_coord_x = dth.point_list_to_coordxy(point_objects).T[0]
        self_x = self.x - self.EPS_above
        left_flags = list(point_objects_coord_x < self_x)
        return left_flags

    def right(self, point_objects):
        """
        Returns Truth array as per elements in point_objects.
        True if other object is to the right, else False in any other case

        Parameters
        ----------
        point_objects : point2d / shapely point / coord_xy list
                       / coord_xy tuple
            DESCRIPTION. An input set of points to compare against

        Returns
        -------
        list of Boolean (truth values)
        """
        point_objects_coord_x = dth.point_list_to_coordxy(point_objects).T[0]
        self_x = self.x + self.EPS_above
        right_flags = list(point_objects_coord_x > self_x)
        return right_flags

    def __add__(self, toadd=0.0, make_new=True):
        """
        CASE-1: if toadd is instance of point2d:
            addition over fundamental axes lengths
        CASE-2: if toadd is single scalar:
            scalar added to both fundamental axes lengths
        CASE-3: if toadd is two scalar element tuple:
        CASE-3: if toadd is tuple of scalars or point2d objects:
            scalar added to both fundamental axes lengths and a
            tuple of point2d objects is returned. Element wise check
            of scalar or point2d is carried out.
        CASE-4: if toadd is a tuple of tuples, then the inner tuple,
        in the form (x, y) is added to the x and y of self and the
        corresponding point2d object is made and appended to tuple to
        be returned.

        make_new applies only to cases 1,2 and 3 of above 4 cases

        RESTRICTIONS:
            toadd: if not cases 1 and 2, toadd MUST be in tuple format

        Parameters
        ----------
        toadd : TYPE, optional
            DESCRIPTION. The default is 0.0.
        make_new : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if isinstance(toadd, point2d):  # CASE-1
            if make_new:
                return self.make_new(self.x+toadd.x, self.y+toadd.y)
            else:
                self.x += toadd.x
                self.y += toadd.y
        if isinstance(toadd, (int, float)):
            if make_new:
                return self.make_new(self.x+toadd, self.y+toadd)
            else:
                self.x += toadd
                self.y += toadd
        if isinstance(toadd, tuple):
            to_return = ()
            # count = 0
            for toadd_ in toadd:
                if isinstance(toadd_, (int, float)):
                    to_return += (self.make_new(self.x+toadd_, self.y+toadd_),)
                elif isinstance(toadd_, tuple):
                    to_return += (self.make_new(self.x +
                                  toadd_[0], self.y+toadd_[1]),)
                elif isinstance(toadd_, point2d):
                    to_return += (self.make_new(self.x +
                                  toadd_.x, self.y+toadd_.y),)
            return to_return

    def __sub__(self, tosub=0.0, make_new=True):
        """
        CASE-1: if tosub is instance of point2d:
            addition over fundamental axes lengths
        CASE-2: if tosub is single scalar:
            scalar added to both fundamental axes lengths
        CASE-3: if tosub is two scalar element tuple:
        CASE-3: if tosub is tuple of scalars or point2d objects:
            scalar added to both fundamental axes lengths and a
            tuple of point2d objects is returned. Element wise check
            of scalar or point2d is carried out.
        CASE-4: if tosub is a tuple of tuples, then the inner tuple,
        in the form (x, y) is added to the x and y of self and the
        corresponding point2d object is made and appended to tuple to
        be returned.

        make_new applies only to cases 1,2 and 3 of above 4 cases

        RESTRICTIONS:
            tosub: if not cases 1 and 2, tosub MUST be in tuple format

        Parameters
        ----------
        tosub : TYPE, optional
            DESCRIPTION. The default is 0.0.
        make_new : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if isinstance(tosub, point2d):  # CASE-1
            if make_new:
                return self.make_new(self.x-tosub.x, self.y-tosub.y)
            else:
                self.x -= tosub.x
                self.y -= tosub.y
        if isinstance(tosub, (int, float)):
            if make_new:
                return self.make_new(self.x-tosub, self.y-tosub)
            else:
                self.x -= tosub
                self.y -= tosub
        if isinstance(tosub, (list, tuple)):
            to_return = ()
            count = 0
            for tosub_ in tosub:
                if isinstance(tosub_, (int, float)):
                    to_return += (self.make_new(self.x-tosub_, self.y-tosub_),)
                elif isinstance(tosub_, tuple):
                    to_return += (self.make_new(self.x -
                                  tosub_[0], self.y-tosub_[1]),)
                elif isinstance(tosub_, point2d):
                    to_return += (self.make_new(self.x -
                                  tosub_.x, self.y-tosub_.y),)
                count += 1
                if count == len(tosub):
                    return to_return

    def __mul__(self, multiplier, update=False, make_new=False, throw=True):
        """


        Parameters
        ----------
        multiplier : TYPE
            DESCRIPTION.
        update : TYPE, optional
            DESCRIPTION. The default is False.
        make_new : TYPE, optional
            DESCRIPTION. The default is False.
        throw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if isinstance(multiplier, (int, float)):
            if update and not make_new:
                self.x *= multiplier
                self.y *= multiplier
                if throw:
                    return self
            if not update and make_new and throw:
                return self.make_new(self.x*multiplier, self.y*multiplier)
            if not (update and make_new) and throw:
                return self.make_new(self.x*multiplier, self.y*multiplier)
            if update and make_new:
                self.x *= multiplier
                self.y *= multiplier
                return self.make_new(self.x, self.y)

        elif isinstance(multiplier, (list, tuple)):
            if update and not make_new:
                self.x *= multiplier[0]
                self.y *= multiplier[1]
                if throw:
                    return self
            if not update and make_new and throw:
                return self.make_new(self.x*multiplier[0],
                                     self.y*multiplier[1])
            if not (update and make_new) and throw:
                return self.make_new(self.x*multiplier[0],
                                     self.y*multiplier[1])
            if update and make_new:
                self.x *= multiplier[0]
                self.y *= multiplier[1]
                return self.make_new(self.x, self.y)
        elif isinstance(multiplier, (point2d, Point)):
            # point2d: upxo point object
            # Point: shapely point object
            if update and not make_new:
                self.x *= multiplier.x
                self.y *= multiplier.y
                if throw:
                    return self
            if not update and make_new and throw:
                return self.make_new(self.x*multiplier.x, self.y*multiplier.y)
            if not (update and make_new) and throw:
                return self.make_new(self.x*multiplier.x, self.y*multiplier.y)
            if update and make_new:
                self.x *= multiplier.x
                self.y *= multiplier.y
                return self.make_new(self.x, self.y)

    def __truediv__(self, divisor, update=False, make_new=False, throw=True):
        """


        Parameters
        ----------
        divisor : TYPE
            DESCRIPTION.
        update : TYPE, optional
            DESCRIPTION. The default is False.
        make_new : TYPE, optional
            DESCRIPTION. The default is False.
        throw : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if isinstance(divisor, (float, int)):
            if not divisor <= point2d.EPS_divisor:
                if update and not make_new:
                    self.x /= divisor
                    self.y /= divisor
                    if throw:
                        return self
                if not update and make_new and throw:
                    return self.make_new(self.x/divisor, self.y/divisor)
                if not (update and make_new) and throw:
                    return self.make_new(self.x/divisor, self.y/divisor)
                if update and make_new:
                    self.x /= divisor
                    self.y /= divisor
                    return self.make_new(self.x, self.y)
        elif isinstance(divisor, (list, tuple)):
            if not (divisor[0] <= point2d.EPS_divisor
                    and divisor[1] <= point2d.EPS_divisor):
                if update and not make_new:
                    self.x /= divisor[0]
                    self.y /= divisor[1]
                    if throw:
                        return self
                if not update and make_new and throw:
                    return self.make_new(self.x/divisor[0], self.y/divisor[1])
                if not (update and make_new) and throw:
                    return self.make_new(self.x/divisor[0], self.y/divisor[1])
                if update and make_new:
                    self.x /= divisor[0]
                    self.y /= divisor[1]
                    return self.make_new(self.x, self.y)
        elif isinstance(divisor, (point2d, Point)):
            # point2d: upxo point object
            # Point: shapely point object
            if not (divisor.x <= point2d.EPS_divisor
                    and divisor.y <= point2d.EPS_divisor):
                if update and not make_new:
                    self.x /= divisor.x
                    self.y /= divisor.y
                    if throw:
                        return self
                if not update and make_new and throw:
                    return self.make_new(self.x/divisor.x, self.y/divisor.y)
                if not (update and make_new) and throw:
                    return self.make_new(self.x/divisor.x, self.y/divisor.y)
                if update and make_new:
                    self.x /= divisor.x
                    self.y /= divisor.y
                    return self.make_new(self.x, self.y)

    def __abs__(self, saa=True, throw=True):
        _x, _y = abs(self.x), abs(self.y)
        if saa:
            self.x, self.y = _x, _y
        if throw:
            return self

    def __int__(self, saa=True, throw=False):
        """
        Explanations:
            if saa and throw, then update and return new point2d
            if saa and !throw, then update and don't return
            if !saa and throw, then don't update and return
            if !saa and !throw, then don't update and don't return
        CALLS:
            p2 = point2d(x = 8, y = 12)
            p2.x = 8.456136
            p2 # upxo.p2d(8.45614, 12.0)
            p2.__int__(saa = False, throw = False)
            p2.__int__(saa = False, throw = True) # upxo.p2d(8.0, 12.0)
            p2 # upxo.p2d(8.45614, 12.0)
            p2.__int__(saa = True, throw = False)
            p2 # upxo.p2d(8, 12)
            p2.x = 8.94321
            p2 # upxo.p2d(8.94321, 12)
            p2.__int__(saa = True, throw = True) # upxo.p2d(8, 12)

        Parameters
        ----------
        saa : TYPE, optional
            DESCRIPTION. The default is True.
        throw : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if saa:
            self.x, self.y = int(self.x), int(self.y)
            if throw:
                return self
        else:
            if throw:
                return point2d(x=int(self.x), y=int(self.y))

    def round_round(self, saa=False, throw=False):
        """


        Parameters
        ----------
        saa : TYPE, optional
            DESCRIPTION. The default is False.
        throw : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if saa:
            self.x, self.y = round(self.x), round(self.y)
        if throw:
            return self

    def round_ceil(self, saa=False, throw=False):
        """


        Parameters
        ----------
        saa : TYPE, optional
            DESCRIPTION. The default is False.
        throw : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if saa:
            self.x, self.y = ceil(self.x), ceil(self.y)
        if throw:
            return self

    def round_floor(self, saa=False, throw=False):
        """


        Parameters
        ----------
        saa : TYPE, optional
            DESCRIPTION. The default is False.
        throw : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if saa:
            self.x, self.y = floor(self.x), floor(self.y)
        if throw:
            return self

    def __repr__(self):
        """
        Instance representation.

        Returns
        -------
        str
            DESCRIPTION.

        """
        return f'upxo.p2d({round(self.x, 8)}, {round(self.y, 8)})'

    def make_mid(self):
        """
        assign_mid Assign the memory address id of self to self.mid

        Returns
        -------
        None.

        """
        self.mid = id(self)

    @property
    def _status_(self):
        '''
        Status update to developer. Not aimed at users
        '''
        if self.has('mid'):
            print('______________________________')
            print('POINT2D STATUS INFORMATION')
            print(f'    mid {self.mid}')
            print(f'    (x, y) ({self.x}, {self.y})')
            print(f'    lean {self.lean}')
            print('______________________________')
        else:
            print('mid does not exist. It will be made now.')
            self.make_mid()
            self._status_

    def borrow_tdist(self, point_object=None):
        """


        Parameters
        ----------
        point_object : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if hasattr(point_object, 'tdist'):
            self.tdist = point_object.tdist

    def has(self, suffix):
        """


        Parameters
        ----------
        suffix : TYPE
            DESCRIPTION.

        Returns
        -------
        has_truth : TYPE
            DESCRIPTION.

        """
        suffix = suffix.lower()
        if suffix in ('tdist', 'toldist', 'tolerance'):
            suffix = 'tdist'
        elif suffix in ('jn', 'bjn', 'bj_n', 'j_n', 'jporder', 'xvo',
                        'xtal_vertex_order'):
            suffix = 'jn'
        elif suffix in ('ptype', 'point_type'):
            suffix = 'ptype'
        elif suffix in ('loc', 'pxtal_loc'):
            suffix = 'loc'
        elif suffix in ('rid', 'randid', 'randomid', 'random_id'):
            suffix = 'rid'
        elif suffix in ('mid', 'omid', 'memory_id', 'mem_id', 'memid',
                        'object_memory_id'):
            suffix = 'mid'
        elif suffix in ('dim', 'dimensionality'):
            suffix = 'dim'
        elif suffix in ('sfv_pol_area', 'area'):
            suffix = 'sfv_pol_area'
        elif suffix in ('ar', 'aspect_ratio'):
            suffix = 'ar'
        elif suffix in ('phase_id', 'phase_id', 'phase_id_number'):
            suffix = 'phase_id'
        elif suffix in ('phase_name', 'phasename', 'phase_name', 'phase'):
            suffix = 'phase_name'
        elif suffix in ('tcname', 'texcomp', 'texture_component', 'ori_name',
                        'oriname'):
            suffix = 'tcname'
        elif suffix in ('eaunit', 'euler_angle_unit', 'eaunits',
                        'euler_angle_units'):
            suffix = 'eaunit'
        elif suffix in ('sfv_repr_ea', 'repr_eulerangle', 'euler_ea', 'eatype',
                        'ea_repr', 'ea_type', 'ea_representation'):
            suffix = 'sfv_repr_ea'
        elif suffix in ('ea', 'ea_val', 'eaval', 'ea_value', 'eulerangle',
                        'orientation'):
            suffix = 'ea'
        has_truth = False
        if hasattr(self, suffix):
            has_truth = True
        return has_truth

    def make_new(self, x=0.0, y=0.0, lean='low', set_mid=True, set_dim=True,
                 set_ptype=False, store_original_coord=True,
                 set_tdist=True, tdist=0.0000000000001, ):
        """
        Create a new point object.

        Parameters
        ----------
        x : TYPE, optional
            DESCRIPTION. The default is 0.0.
        y : TYPE, optional
            DESCRIPTION. The default is 0.0.
        lean : TYPE, optional
            DESCRIPTION. The default is 'low'.
        set_mid : TYPE, optional
            DESCRIPTION. The default is True.
        set_dim : TYPE, optional
            DESCRIPTION. The default is True.
        set_ptype : TYPE, optional
            DESCRIPTION. The default is False.
        store_original_coord : TYPE, optional
            DESCRIPTION. The default is True.
        set_tdist : TYPE, optional
            DESCRIPTION. The default is True.
        tdist : TYPE, optional
            DESCRIPTION. The default is 0.0000000000001.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return point2d(x=x, y=y, lean=lean, set_mid=set_mid,
                       set_dim=set_dim, set_ptype=set_ptype,
                       store_original_coord=store_original_coord,
                       set_tdist=set_tdist, tdist=tdist,)

    def negx(self):
        """
        Mirror about y-axis

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.x = -self.x
        return self

    def negy(self):
        """
        Mirror about x-axis

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.y = -self.y
        return self

    def negxy(self):
        """
        Mirror about y-axis and then about x-axis

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.x, self.y = -self.x, -self.y
        return self

    def displace_by(self, delx: float = 0.0, dely: float = 0.0,):
        """
        Move self by xdisp and ydisp

        Parameters
        ----------
        delx : float, optional
            DESCRIPTION. The default is 0.0.
        dely : float, optional
            DESCRIPTION. The default is 0.0.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.x, self.y = self.x + delx, self.y + dely

    def move_to(self, xlocation: float = 0.0, ylocation: float = 0.0):
        """
        Move self to (locx, locy)

        Parameters
        ----------
        xlocation : float, optional
            DESCRIPTION. The default is 0.0.
        ylocation : float, optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        None.

        """
        self.x, self.y = xlocation, ylocation

    def align_to(self, method='point2d', ref_point_object=None,
                 xlocation: float = None, ylocation: float = None):
        """
        Align self to a new point

        NOTE: This is akin to alignTo of vedo. See below.
        https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'point2d'.
        ref_point_object : TYPE, optional
            DESCRIPTION. The default is None.
        xlocation : float, optional
            DESCRIPTION. The default is None.
        ylocation : float, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if method == 'point2d':
            self.x, self.y = ref_point_object.x, ref_point_object.y
        elif method == 'coord':
            self.x, self.y = xlocation, ylocation

    def reset(self):
        """
        Resets location of self to UPXO default point2d location

        Returns
        -------
        None.

        """
        if self.store_original_coord:
            if hasattr(self, '_original_location'):
                self.x, self.y = self._original_location

    def attach_mulpoint(self,
                        mulpointObject=None
                        ):
        """
        Attach this object to a multi-point object

        Parameters
        ----------
        mulpointObject : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self._mulpoints_ = mulpointObject

    def attach_edge(self,
                    edgeObject=None
                    ):
        """
        Attach this object to an edge object

        Parameters
        ----------
        edgeObject : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self._edges_ = edgeObject

    def set_lean(self, lean):
        """


        Parameters
        ----------
        lean : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.lean = lean

    def set_original_location(self):
        """


        Returns
        -------
        None.

        """
        self._original_location = (self.x, self.y)

    def set_dim(self, dim=2):
        """


        Parameters
        ----------
        dim : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        try:
            int(dim)
        except:
            if type(dim) != int:
                print('value of dim MUST be a real number')
        else:
            _ = int(dim)
            if _ == 2:
                self.dim = _
            else:
                print('value of dim MUST be 2 @edge2d. Assigning dim = 2')

    def set_tdist(self, tdist=0.0):
        """


        Parameters
        ----------
        tdist : TYPE, optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        None.

        """
        self.tdist = tdist

    def set_jn(self,
               jn=3,
               ):
        """
        jtype: number of l0e emanating from it / terminating in it.
        """
        self.jn = jn

    def set_ptype(self,
                  ptype='jp',
                  ):
        '''
        Assign which type of point it is going to be or it is.
        CASES INCLUDE THE FOLLOWING:
            1. jp: Junction point
            2. gbp: grain boundary point
            3. # TODO: write the many more that there are.
        '''
        self.ptype = ptype

    def attach_mulpoints(self, mulpoints):
        """


        Parameters
        ----------
        mulpoints : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.mulpoints = mulpoints

    def attach_edges(self, edges):
        """


        Parameters
        ----------
        edges : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.edges = edges

    def attach_muledges(self, muledges):
        """

        :param muledges: DESCRIPTION
        :type muledges: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.muledges = muledges

    def attach_xtals(self, xtals):
        """

        :param xtals: DESCRIPTION
        :type xtals: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.xtals = xtals

    def attach_polyxtals(self, polyxtals):

        self.polyxtals = polyxtals

    def set_location(self, loc='pxtal_internal',):
        """


        Parameters
        ----------
        loc : TYPE, optional
            DESCRIPTION. The default is 'pxtal_internal'.

        Returns
        -------
        None.

        Suggested options for loc
        -------------------------
            1. pxtal_internal / pxtal_boundary
            2. on_grain_boundary
            3. particle_seed
            4. VTGS_seed
            5. voxel_centroid
            6. voxel_corners
        """
        self.loc = loc

    def set_sfv_pol_area(self, sfv_pol_area):
        """
        Set value for 'sfv_pol_area'.

        Parameters
        ----------
        sfv_pol_area : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_pol_area = sfv_pol_area

    def set_sfv_pol_perimeter(self, sfv_pol_perimeter):
        """


        Parameters
        ----------
        sfv_pol_perimeter : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_pol_perimeter = sfv_pol_perimeter

    def set_sfv_pol_aspect_ratio(self, value):
        """


        Parameters
        ----------
        value : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_pol_aspect_ratio = value

    def set_vis_prop(self,
                     mtype='o',
                     mew=1.0,
                     mec='k',
                     msz=10,
                     mfill='w',
                     malpha=1.0,
                     bfill='teal',
                     ):
        """
        set_marker_prop Set some properties of markers

        Args:
            mtype (str): marker type. Defaults to 'o'.
            mew (float): marker edge width. Defaults to 1.0.
            mec (str): marker edge colour. Defaults to 'k'.
            msz (int): marker size. Defaults to 10.
            mfill (str): marker fill. Defaults to 'w'.
            malpha (float): marker alpha. Defaults to 1.0.
            bfill (float): marker buffer fill. Defaults to 'teal'.

        Returns:
            dict: A dictionary of visualization parameters
        """
        vprop = {'mtype': mtype,
                 'mew': mew,
                 'mec': mec,
                 'msz': msz,
                 'mfill': mfill,
                 'malpha': malpha,
                 'bfill': bfill
                 }
        return vprop

    def set_mesh_lc(self,
                    mesh_lc=1.0
                    ):
        '''
        Set value of mesh_lc for use in meshing using GMSH
        '''
        self.mesh_lc = mesh_lc

    def make_shapely(self, saa=True, throw=False):
        """
        Returs an equivalent shapely point object

        Returns
        -------
        shapely.geometry.point.Point object
            Shapely point object having the same x and y as self.x and self.y
        """
        _ = Point(self.x, self.y)
        if saa:
            self.image_sh = _
        if throw:
            return _

    def make_vedo(self):
        """
        Make a vedo point object
        """
        pass

    def make_vtk(self):
        """
        vtk Make VTK representation of the point object
        """
        pass

    def make_pyvtk(self):
        """
        make a pyvtk object
        """
        pass

    def make_paraview(self):
        """
        Make a paraview point object
        """
        pass

    def make_gmsh(self):
        """
        make a gmsh point object

        Returns
        -------
        None.

        """
        pass

    def make_pyvista(self, z: float = 0.0,
                     saa: bool = True,
                     stf: bool = False,
                     filename_base: str = 'image_pv.',
                     file_format: str = 'vtk',
                     attach_fields: bool = False,
                     throw: bool = True,
                     ):
        """


        Parameters
        ----------
        z : float, optional
            DESCRIPTION. The default is 0.0.
        saa : bool, optional
            DESCRIPTION. The default is True.
        stf : bool, optional
            DESCRIPTION. The default is False.
        filename_base : str, optional
            DESCRIPTION. The default is 'image_pv.'.
        file_format : str, optional
            DESCRIPTION. The default is 'vtk'.
        attach_fields : bool, optional
            DESCRIPTION. The default is False.
        throw : bool, optional
            DESCRIPTION. The default is True.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        _pyvista_pset : TYPE
            DESCRIPTION.

        """
        import pyvista
        _pyvista_pset = pyvista.PointSet(np.array([[self.x, self.y, z]],
                                                  dtype=np.float64))
        if stf:
            filename = filename_base
            + f'.upxo.p2d_{round(self.x, 3)}_{round(self.y, 3)}'
            _pyvista_pset.save(filename=filename)
        if saa:
            self.image_pv = _pyvista_pset
        if throw:
            return _pyvista_pset

    def set_morph_curvature(self,
                            value: float = 0.0,
                            ):
        pass

    def set_gnd(self,
                value: float = 0.0,
                ):
        pass

    def set_lattice_curvature_(self,
                               value: list[list] = [[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0]
                                                    ]
                               ):
        pass

    def set_phase(self,
                  phase_id: int = 0,
                  phase_name: str = 'cucrzr'
                  ):
        """
        Assign phase id and phase name to point

        Parameters
        ----------
        phase_id : int, optional
            DESCRIPTION. The default is 0.
        phase_name : str, optional
            DESCRIPTION. The default is 'cucrzr'.

        Returns
        -------
        None.

        """
        self.phase_id = phase_id
        self.phase_name = phase_name

    def set_tcname(self,
                   name: str = 'B'
                   ):
        """
        Assign name of the texture component to self

        Parameters
        ----------
        name : str, optional
            DESCRIPTION. The default is 'B'.

        Returns
        -------
        None.

        """
        self.tcname = name

    def set_sfv_repr_ea(self, sfv_repr_ea='Bunge'):
        """
        Assign the default representation of Euler angles

        Parameters
        ----------
        sfv_repr_ea : TYPE, optional
            DESCRIPTION. The default is 'Bunge'.

        Returns
        -------
        None.

        """
        self.sfv_repr_ea = sfv_repr_ea

    def set_sfv_ea1(self, ea1):
        """
        Assign 1st Euler angle to point object

        Parameters
        ----------
        ea1 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_ea[0] = ea1

    def set_sfv_ea2(self, ea2):
        """
        Assign 2nd Euler angle to point object

        Parameters
        ----------
        ea2 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_ea[1] = ea2

    def set_sfv_ea3(self, ea3):
        """
        Assign 3rd Euler angle to point object

        Parameters
        ----------
        ea3 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_ea[2] = ea3

    def set_sfv_ea(self,
                   ea: list = [0, 0, 0]
                   ):
        """
        Assign the three Euler angles to point object

        Parameters
        ----------
        ea : list, optional
            DESCRIPTION. The default is [0, 0, 0].

        Returns
        -------
        None.

        """
        self.sfv_ea = ea

    def set_orientation_object(self, orientation_object):
        """
        Assign the UPXO orientation object to point object

        Parameters
        ----------
        orientation_object : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.orientation_object = orientation_object

    def make_partition(self,
                       restart: bool = False,
                       saa: bool = True,
                       n: int = 4,
                       char_lengths: list = [1.0],
                       char_length_type: str = 'radius',
                       make_polygon: bool = True,
                       tool: str = 'shapely',
                       rotation: float = 0,
                       feed_partition=False,
                       feed_partition_name='grain',
                       feed_partition_tool='shapely',
                       feed_partition_polygon=None,  # A shapely polygon object
                       ):
        """


        Parameters
        ----------
        restart : bool, optional
            DESCRIPTION. The default is False.
        saa : bool, optional
            DESCRIPTION. The default is True.
        n : int, optional
            DESCRIPTION. The default is 4.
        char_lengths : list, optional
            DESCRIPTION. The default is [1.0].
        char_length_type : str, optional
            DESCRIPTION. The default is 'radius'.
        make_polygon : bool, optional
            DESCRIPTION. The default is True.
        tool : str, optional
            DESCRIPTION. The default is 'shapely'.
        rotation : float, optional
            DESCRIPTION. The default is 0.
        feed_partition : TYPE, optional
            DESCRIPTION. The default is False.
        feed_partition_name : TYPE, optional
            DESCRIPTION. The default is 'grain'.
        feed_partition_tool : TYPE, optional
            DESCRIPTION. The default is 'shapely'.
        feed_partition_polygon : TYPE, optional
            DESCRIPTION. The default is None.
        # A shapely polygon object : TYPE
            DESCRIPTION.

        Returns
        -------
        xy : TYPE
            DESCRIPTION.

        NOTES
        -----
            By default, self.x and self.y will be the centroid of the name
            to be made.

        EXAMPLE CALL
        ------------
            bo = NAME.make_partition(restart = False,
                                  saa = True,
                                  n = 4,
                                  char_lengths = [1.0],
                                  char_length_type = 'radius',
                                  make_polygon = True,
                                  tool = 'mpl',
                                  rotation = 0,
                                  feed_partition = False,
                                  feed_partition_name = 'grain',
                                  feed_partition_tool = 'shapely',
                                  feed_partition_polygon = None
                                  )
        Include below in the above, if appropriate (which ever oe is)
            name (str, optional): "closed"-name of the partition operation.
            Defaults to 'square'.

            auto_size (bool, optional): _description_. Defaults to True.

            lengths (list, optional): list/tuple/numpy array of characteristic
            lengths to make the name. Defaults to [1.0].

            polygonize (bool, optional): If true, convert the name to UPXO
            crystal2d object
            If true, crystal2d will be returned
            Note: This must not be saved to attribute. Defaults to False.

            n (int): Number of points on continuous shapes like circle,
            ellipse, etc

            rotation (float):morphological rotation of the polygon, about
            (self.x, self.y)
        """
        from dataclasses import dataclass, field

        @dataclass(init=True, frozen=False, repr=True)
        class partition_container():
            '''
            x: x-coordinate of the centre
            y: y-coordinate of the centre
            ns: list having Number of vertices
            tools: list having tool requested/required/used
            ppos: list having partition polygon objects
            '''
            x: float = field(default=None,
                             repr=True,
                             metadata={'unit': 'microns',
                                       }
                             )
            y: float = field(default=None,
                             repr=True,
                             metadata={'unit': 'microns',
                                       }
                             )
            names: list['str'] = field(default_factory=list, repr=True)
            ns: list['str'] = field(default_factory=list, repr=True)
            tools: list['str'] = field(default_factory=list, repr=True)
            ppos: list['str'] = field(default_factory=list, repr=False)
        # -----------------------
        if (saa and (hasattr(self, 'partition') is False
                     or restart)) or feed_partition:
            self.partition = partition_container()
        # -----------------------
        if feed_partition is False:
            if len(char_lengths) == 2 and char_lengths[0] != char_lengths[1]:
                # RECTANGLE
                delta_x, delta_y = 0.5*char_lengths[0], 0.5*char_lengths[1]
                xminus, xplus, yminus, yplus = _x-delta_x, _x+delta_y, _y-delta_x, _y+delta_y
                xy = [[xminus, yminus], [xplus, yminus],
                      [xplus, yplus], [xminus, yplus]]
            # ................
            if len(char_lengths) == 1 or char_lengths[0] == char_lengths[1]:
                # REGULAR POLYGONS
                if char_length_type == 'side_length':
                    # REF for converting side length to radius
                    # https://www.mathopenref.com/polygonradius.html
                    r = 0.5*char_lengths[0]/np.sin(3.141592653589793238/n)
                elif char_length_type == 'radius':
                    r = char_lengths[0]
                # .. .. .. .. .. .. ..
                ''' @DEV
                n = 5
                side_length = 1
                r = 0.5*side_length/np.sin(3.141592653589793238/n)
                #=================
                from matplotlib.patches import RegularPolygon as rp
                _mpl = rp((0, 0), numVertices = n, radius = r,
                          orientation = np.radians(45))
                xy1 = np.round(_mpl.get_verts(), 10)[:-1]
                from shapely.geometry import Polygon
                a = Polygon(xy1)
                #=================
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize = (1.6, 1.6), dpi = 100)
                plt.fill(xy1.T[0],
                         xy1.T[1],
                         color = 'red',
                         alpha = 1.0,
                         edgecolor = 'black',
                         )
                plt.show()
                '''
                from matplotlib.patches import RegularPolygon as rp
                # make the matplotlib patch object and extract its vertices
                # print('#####################')
                # print(rotation)
                # print('#####################')
                _center = (self.x, self.y)
                _mpl_po = rp(_center,
                             numVertices=n,
                             radius=r,
                             orientation=np.radians(rotation)
                             )
                xy = _mpl_po.get_verts()
                xy = np.round(xy, point2d.ROUND_ZERO_DEC_PLACE)[:-1]
                # from shapely.geometry import Polygon as shpol
                # from shapely import affinity
                # rot_shpol = affinity.rotate(shpol(xy), 45, 'center')
                # xy = rot_shpol.boundary.coords.xy
                # np.c_[np.array(list(xy[0])), np.array(list(xy[1]))][:-1]
            # ................
            if make_polygon:
                # EXTRACT THE bpo: partition polygon object
                if tool == 'mpl':
                    ppo = _mpl_po
                elif tool == 'shapely':
                    from shapely.geometry import Polygon
                    ppo = Polygon(xy)  # Buffer Polygon Object
                elif tool == 'upxo':
                    pass
        # -----------------------
        if saa:
            self.partition.x = self.x
            self.partition.y = self.y
            if feed_partition is False:
                self.partition.ns.append(n)
                self.partition.names.append('')
                self.partition.tools.append(tool)
                self.partition.ppos.append(ppo)
            elif feed_partition:
                self.partition.x = self.x
                self.partition.y = self.y
                self.partition.names.append(feed_partition_name)
                if partition_tool == 'shapely':
                    # IGNORES THE HOLES IN THE POLYGON
                    # TODO: This is a to be done in the future
                    self.partition.ns.append(
                        len(feed_partition_polygon.boundary.coords.xy[0])-1)
                self.partition.tools.append(feed_partition_tool)
                self.partition.ppos.append(feed_partition_polygon)
        else:
            return xy

    def plot(self, dpi=50, point: bool = True,
             buffer: bool = True, vprop=None,):
        # -------------------------
        if vprop is None:
            vprop = self.set_vis_prop()
        # -------------------------
        if point or buffer:
            fig = plt.figure(figsize=(1.6, 1.6), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
        # -------------------------
        if point:
            ax.plot(self.x, self.y,
                    marker=vprop['mtype'],
                    markeredgewidth=vprop['mew'],
                    markeredgecolor=vprop['mec'],
                    markersize=vprop['msz'],
                    markerfacecolor=vprop['mfill'],
                    alpha=vprop['malpha']
                    )
            plt.show()
        # -------------------------
        if buffer:
            pass

    def distance(self, other_object_type='point2d', point_data=None):
        """
        Calculate distance from self to another point(s)

        In all examples below, the following are understood appropriately
            from point2d_04 import point2d
            from mulpoint2d_3 import mulpoint2d

        Parameters
        ----------
        other_object_type : TYPE, optional
            DESCRIPTION. The default is 'point2d'.
        point_data : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # ................
        if other_object_type in ('upxo_point1d', 'point1d', 'p1d'):
            pass
        # ................
        if other_object_type in ('upxo_point1d_list', 'point1d_list',
                                 'p1dlist', 'p1d_list'):
            pass
        # ................
        if other_object_type in ('upxo_point2d', 'point2d', 'p2d'):
            '''
            Explanations:
                1. Use this when distances are to be computed against
                another point2d object
                2. INPUT TYPE of "point_data": UPXO.point2d object

            Example 1: Point to point
                p1, p2 = point2d(x = 0, y = 0), point2d(x = 1, y = 1)
                p1.distance(other_object_type = 'point2d', point_data = p2)
                p2.distance(other_object_type = 'point2d', point_data = p2)

            Example 2: Point to a list of points (method - 1)
                Not preferred, as there is a simpler method available under
                case 'point2d_list'
                x, y = list(range(0, 10, 2)), list(range(0, 20, 4))
                p = [point2d(x = _x, y = _y) for _x, _y in zip(x, y)]
                d = [p1.distance(other_object_type = 'point2d',
                                 point_data = pi) for pi in p]
            '''
            return np.sqrt((self.x-point_data.x)**2
                           + (self.y-point_data.y)**2)
        # ................
        if other_object_type in ('upxo_point2d_list', 'point2d_list',
                                 'p2dlist', 'p2d_list'):
            '''
            Explanations:
                1. Use this when distances are to be computed against a list
                of point2d objects
                2. INPUT TYPE of "point_data": list

            Example 1: Point to list of points
                p1 = point2d(x = 0, y = 0)
                p2 = point2d(x = 1, y = 1)
                p1.distance(other_object_type = 'point2d_list',
                            point_data = [p1, p2])

            Example 2: Point to list of points
            '''
            _x, _y, array = deepcopy(self.x), deepcopy(self.y), np.array
            x, y = zip(*[(pi.x, pi.y) for pi in point_data])
            return np.sqrt((_x-array(x))**2 + (_y-array(y))**2)
        # ................
        if other_object_type in ('xy_coord_list', 'coord_lists',
                                 'coord_list', 'clists', 'clists'):
            '''
            Exaplantions:
                1. Use this when distances are to be computed againt a
                list of coordinates
                2. INPUT TYPE of "point_data": [[x1, x2, ..., xn],
                                                [y1, y2, ..., yn]]

            Example 1: Point to list of x and y coordinate list
                p = point2d(x = 0, y = 0)
                point_data = [[0, 1, 2], [0, 1, 2]]
                d = p.distance(other_object_type = 'xy_coord_list',
                               point_data = point_data)
            '''
            _x, _y, array = deepcopy(self.x), deepcopy(self.y), np.array
            return np.sqrt((_x-array(point_data[0]))**2
                           + (_y-array(point_data[1]))**2)
        # ................
        if other_object_type in ('xy_coord_pairs', 'coord_pairs',
                                 'coord_pair', 'cpairs', 'cpair'):
            '''
            Exaplantions:
                1. Use this when distances are to be computed against a list
                of coordinate pairs
                2. INPUT TYPE of "point_data": [ [x1, y1], [x2, y2]]

            Example 1: Point to list of [x, y] coordinate pairs
                p = point2d()
                point_data = [[0, 0], [1, 1], [2, 2]]
                d = p.distance(other_object_type = 'xy_coord_pairs',
                               point_data = point_data)
            '''
            point_data = np.array(point_data).T
            return np.sqrt((self.x-point_data[0])**2
                           + (self.y-point_data[1])**2)
        # ................
        if other_object_type == 'upxo_point3d':
            pass
        # ................
        if other_object_type == 'upxo_mulpoint1d':
            pass
        # ................
        if other_object_type in ('upxo_mulpoint2d', 'mp2d', 'mulpoint2d'):
            '''
            Explanations:
                1. Use this when computimng distance sgainst a set of point2d
                   objects contained inside a mulpoint2d object(m).
                   PSEUDO CODE:
                       distance(self: point object, m.points: list)
                2. INPUT TYPE of "point_data": [upxo point object 1,
                                                upxo point object 2,
                                                ...,
                                                upxo point object n]

            Example 1:
                # Create the reference point2d object
                    p0 = point2d()

                # Create a mulpoint2d object
                    p1 = point2d(x = 2.0, y = 1.0)
                    p2 = p1 + 1
                    p3 = p2 * 0.6498
                    p4 = p1*0.468 + p3/p2
                    m1 = mulpoint2d(method = 'points',
                                    point_objects = [p1, p2, p3, p1 + p4*p1])

                # Calculate distance
                    d = p0.distance(other_object_type = 'mulpoint2d',
                                    point_data = m1)
                '''
            return self.distance(other_object_type='point2d_list',
                                 point_data=point_data.points)
        # ................
        if other_object_type == 'upxo_mulpoint3d':
            pass
        # ................
        if other_object_type == 'upxo_edge2d':
            pass
        # ................
        if other_object_type in ('upxo_muledge2d', 'upxo_ring2d'):
            pass
        # ................
        if other_object_type == 'upxo_edge3d':
            pass
        # ................
        if other_object_type in ('upxo_muledge3d', 'upxo_ring3d'):
            pass
        # ................
        if other_object_type == 'shapely_xtal2d_centroid':
            '''
            Explanations:
                1. Use this to find distance between self and centroid
                   of the shapely polygon object
                2. INPUT TYPE of "point_data": a valid shapely polygon object
                3. Centroidal x and y of polygon object will be used as
                   point_data = [[x], [y]]

            Example 1:
                from point2d_04 import point2d
                p0 = point2d()
                from shapely.geometry import Polygon
                shapelypol = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

                p0.distance(other_object_type = 'shapely_xtal2d_centroid',
                            point_data = shapelypol)
            '''
            centroid = point_data.centroid
            return self.distance(other_object_type='coord_list',
                                 point_data=[[centroid.x], [centroid.y]])[0]
        if other_object_type == 'shapely_xtal2dlist_centroid':
            '''
            Explanations:
                1. Use this to find distance between self and centroids of a
                   list of shapely polygon objects
                2. INPUT TYPE of "point_data": list of valid shapely polygon
                   objects

            Example 1:
                from point2d_04 import point2d
                p0 = point2d()
                from shapely.geometry import Polygon
                shapelypol1 = Polygon([[0,0], [1,0], [1,1], [0,1], [0,0]])
                shapelypol2 = Polygon([[1,1], [2,1], [2,2], [1,2], [1,1]])

                point_data = [shapelypol1, shapelypol2]
                p0.distance(other_object_type = 'shapely_xtal2dlist_centroid',
                            point_data = point_data)
            '''
            centroids = [[_.centroid.x, _.centroid.y] for _ in point_data]
            return self.distance(other_object_type='coord_pairs',
                                 point_data=centroids)

        if other_object_type == 'shapely_xtal2d_reppoint':
            '''
            Explanations:
                1. Use this to find distance between self and reppoint of the
                   shapely polygon object
                2. INPUT TYPE of "point_data": a valid shapely polygon object
                3. centroidal x and y of polygon object will be used as
                   point_data = [[x], [y]]

            Example 1:
                from point2d_04 import point2d
                p0 = point2d()
                from shapely.geometry import Polygon
                shapelypol = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

                p0.distance(other_object_type = 'shapely_xtal2d_reppoint',
                            point_data = shapelypol)
            '''
            reppoint = point_data.representative_point()
            return self.distance(other_object_type='coord_list',
                                 point_data=[[reppoint.x], [reppoint.y]])[0]
        if other_object_type == 'shapely_xtal2dlist_reppoint':
            '''
            Explanations:
                1. Use this to find distance between self and reppoints of a
                   list of shapely polygon objects
                2. INPUT TYPE of "point_data": list of valid shapely polygon
                   objects

            Example 1:
                from point2d_04 import point2d
                p0 = point2d()
                from shapely.geometry import Polygon
                shapelypol1 = Polygon([[0,0], [1,0], [1,1], [0,1], [0,0]])
                shapelypol2 = Polygon([[1,1], [2,1], [2,2], [1,2], [1,1]])

                point_data = [shapelypol1, shapelypol2]
                p0.distance(other_object_type = 'shapely_xtal2dlist_reppoint',
                            point_data = point_data)
            '''
            reppoints = [[_.representative_point().x,
                          _.representative_point().y] for _ in point_data]
            return self.distance(other_object_type='coord_pairs',
                                 point_data=reppoints)
        # ................
        if other_object_type == 'upxo_xtal2d_reppoint':
            # here point_data will be the xtal containing the representative
            # point
            # representative point has to be UPXO point2d object
            return self.distance(other_object_type='upxo_point2d',
                                 point_data=point_data.reppoint)
        if other_object_type == 'upxo_xtal2dlist_reppoint':
            # This is to call self.distance operating on case
            # 'upxo_xtal2d_reppoint'
            return [self.distance(other_object_type='upxo_xtal2d_reppoint',
                                  point_data=_point_data)
                    for _point_data in point_data]
        # ................
        if other_object_type == 'upxo_xtal2d_vertices':
            # This is to call self.distance operating on case
            # 'upxo_xtal2d_reppoint'
            pass
        # ................
        if other_object_type == 'upxo_xtal3d_centroid':
            pass
        # ................
        if other_object_type == 'upxo_xtal3d_reppoint':
            pass
        # ................
        if other_object_type == 'upxo_xtal3d_vertices':
            pass
        # ................
        if other_object_type == 'shapely_point':
            pass
        # ................
        if other_object_type == 'vtk_point':
            pass
        # ................
        if other_object_type in ('scipy_ckdtree', 'scipy_tree',
                                 'tree', 'ckdrtee', 'ckdt'):
            pass
        if point_data is None:
            print('Need other object to compute distance(s)')

    def proximity(self, other_data_type='mulpoint2d',
                  point=None, point_coord=None, point_list=None,
                  locx=None, locy=None, locxy=None,
                  m=None, m_list=None,
                  tree=None, xtal_feature='reppoints',
                  xtal=None, xtal_list=None,
                  pxtal=None, pxtal_list=None,
                  ):
        """
        CORE FEATURE OF POINT2D CLASS AND POINT3D CLASS

        NOTE: This is to call self.distance with appropriate arguments

        Exaplantoins:
            1. Calculates the objects within the user specified cor
               ('Cut-Off-Radius')
            2. Objects could be:
                a. list of point2d objects
                b. list of mulpoint objects
                c. list of list of xcoordinates and list of list of
                   ycoordinates
                d. list of list of xy coordinate pairs
                e. list of trees (ckd-tree)
                f. list of xtals
                g. list of pxtals

        Parameters
        ----------
        other_data_type : TYPE, optional
            DESCRIPTION. The default is 'mulpoint2d'.
        point : TYPE, optional
            DESCRIPTION. The default is None.
        point_coord : TYPE, optional
            DESCRIPTION. The default is None.
        point_list : TYPE, optional
            DESCRIPTION. The default is None.
        locx : TYPE, optional
            DESCRIPTION. The default is None.
        locy : TYPE, optional
            DESCRIPTION. The default is None.
        locxy : TYPE, optional
            DESCRIPTION. The default is None.
        m : TYPE, optional
            DESCRIPTION. The default is None.
        m_list : TYPE, optional
            DESCRIPTION. The default is None.
        tree : TYPE, optional
            DESCRIPTION. The default is None.
        xtal_feature : TYPE, optional
            DESCRIPTION. The default is 'reppoints'.
        xtal : TYPE, optional
            DESCRIPTION. The default is None.
        xtal_list : TYPE, optional
            DESCRIPTION. The default is None.
        pxtal : TYPE, optional
            DESCRIPTION. The default is None.
        pxtal_list : TYPE, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if other_data_type in ('point2d', 'p2d', 'p2'):
            return self.distance(other_object_type='point2d',
                                 point_data=point)
        if other_data_type in ('point2d_list', 'p2d_list', 'p2dlist'):
            return self.distance(other_object_type='point2d_list',
                                 point_data=point_list)
        if other_data_type in ('point_coord', 'point_coord', 'coord', 'xy'):
            return np.sqrt((self.x-point_coord[0])**2
                           + (self.y-point_coord[1])**2)
        if other_data_type in ('xy_coord_list'):
            return self.distance(other_object_type='xy_coord_list',
                                 point_data=None)
        if other_data_type in ('xy_coord_pairs'):
            return self.distance(other_object_type='xy_coord_pairs',
                                 point_data=None)
        if other_data_type in ('mulpoint2d', 'mp2d'):
            return self.distance(other_object_type='mulpoint2d',
                                 point_data=m)
        if other_data_type in ('mulpoint2d_list', 'list_of_mp2d',
                               'mp2_list', 'mp2d_list'):
            return [self.distance(other_object_type='mulpoint2d',
                                  point_data=m) for m in m_list]
        if other_data_type in ('upxo_xtal2d', 'shapely_xtal2d',
                               'shapely_xtal', 'scipy_cell2d',
                               'scipy_cell'):
            pass
        if other_data_type in ('upxo_pxtal2d'):
            pass
        if other_data_type in ('scipy_tree', 'ckdtree'):
            pass
        if other_data_type in ('shapely_xtal2d_centroid',
                               'shapely_xtal_centroid'):
            return self.distance(other_object_type='shapely_xtal2d_centroid',
                                 point_data=xtal)

    def find_neigh_points(self, method: str = 'points',
                          points: list = None, point_type: str = 'upxo',
                          ckdtrees: list = None, ckdtree_workers=1,
                          srtrees: list = None,
                          mulpoints: list = None, mulpoint_type: str = 'upxo',
                          edges: list = None, edge_type: str = 'upxo',
                          muledges: list = None, muledge_type: str = 'upxo',
                          rings: list = None, ring_type: str = 'upxo',
                          partitions: list = None,
                          partition_type: str = 'upxo',
                          mulpartitions: list = None,
                          mulpartition_type: str = 'upxo',
                          cutoffshape: str = 'circle',
                          cut_off_radii: float = 1.0,
                          r1: float = 1.0, r2: float = 1.0,
                          a: float = 1.0, b: float = 1.0, angle: float = 0.0,
                          comparison1: str = 'le', comparison2: str = 'ge',
                          other_point_type='self',  # LEAVE UNDOCUMENTED
                          other_point=[0.0, 0.0]  # LEAVE UNDOCUMENTED
                          ):
        """
        Find neighbouring points within a circle, ellipse, square, rectangle
        and hexagon

        Parameters
        ----------
        method : str, default "points"
            Specifies the type of comparison data. Options include "points",
            "ckdtrees", "srtrees", "mulpoints", "edges", "muledges",
            "rings", "mulrings", "partitions", "mulpartitions".
        points : list
            A list of points. Data-type of all elements of `points` MUST be
            same. NO FURTHER CHECKS ARE MADE TO ENSURE THIS.
        point_type: str, default "upxo"
            Data-type of all elements of the `points` argument. Allowed values
            are "upxo", "shapely", "vtk", "coord_pairs", "coords"
        ckdtrees : list
            A list of scipy ckdtree objects. Returned object will be list of
            lists. An inner list may contain points which are members of the
            corresponding ckdtree object and also ARE neighbours to self
            point object.
        srtrees : list
            A list of shapely srtree objects. Returned object will be list of
            lists. An inner list may contain points which are members of the
            corresponding srtree object and also ARE neighbours to self
            point object.
        mulpoints : list
            A list of multi-point objects. Returned object will be list of
            lists. An inner list may contain points which are members of the
            corresponding mulpoint object and also ARE neighbours to self
            point object.
        mulpoint_type : str, default "upxo"
            Data-type of all elements of the `mulpoints` argument. Allowed
            values are "upxo", "shapely", "vtk", "coord_pairs", "coords"
        edges : list
            A list of edge objects
        edge_type: str
            Data-type of all elements of the `edges` argument. Allowed
            values are "upxo", "shapely", "vtk", "coord_pairs", "coords"
        muledges : list
            A list of muledge objects
        muledge_type: str
            Data-type of all elements of the `muledges` argument. Allowed
            values are "upxo", "shapely", "vtk", "coord_pairs", "coords"
        rings : list
            A list of ring objects
        ring_type : str
            Data-type of all elements of the `ring` argument. Allowed
            values are "upxo", "shapely", "vtk", "coord_pairs", "coords"
        partitions : list
            A list of partition objects
        partition_type : str
            Data-type of all elements of the `partition` argument. Allowed
            values are "upxo", "shapely", "vtk", "coord_pairs", "coords"
        mulpartitions : list
            A list of mulpartition objects
        mulpartition_type : str
            Data-type of all elements of the `mulpartition` argument. Allowed
            values are "upxo", "shapely", "vtk", "coord_pairs", "coords"
        cutoffshape : str, default "circle"
            Choice of the cut-off-shape
            Options include circle, ellipse, square, rectangle and hexagon
        cut_off_radii : float, default 1.0
            Characteristic length (radius) if cutoffshape == 'circle'
        r1 and r2 : float and float, default 1.0 and 1.0
            Characteristic lengths (radii) if cutoffshape == 'ellipse'
            Larger radius oriented along x-axis before applying rotation
        a : float, default 1.0
            Characteristic length if cutoffshape == 'square'
        a and b : float and float, default 1.0 and 1.0
            Characteristic lengths (side lengths) if cutoffshape == 'rectangle'
            Larger side oriented along x-axis before applying rotation
        angle : float, default 0.0 degrees
            Angle of cutoffshape's orientation with x+ axis
            Units: degrees

        Returns
        -------
        neigh_points : list/tuple/deque
            An iterable collection of neighbouring point2d objects

        Notes
        -----
        ANy notes to come here.

        Examples-1.A
        ------------
        Example-1.A.1: method = "points", point_type = "upxo",
                       cutoffshape == 'circle'
        Example-1.A.2: method = "points", point_type = "upxo",
                       cutoffshape == 'ellipse'

        Examples-1.B
        ------------
        CASE: method = "points", point_type = "shapely"
        Examples-1.C
        ------------
        CASE: method = "points", point_type = "vtk"
        Examples-1.D
        ------------
        CASE: method = "points", point_type = "coord_pairs"
        Examples-1.E
        ------------
        CASE: method = "points", point_type = "coords"

        Examples-2
        ---------
        CASE: method = "ckdtrees"

        Examples-3
        ---------
        CASE: method = "srtrees"

        Examples-4
        ---------
        CASE: method = "mulpoints", mulpoint_type = "upxo"

        Examples-5
        ---------
        CASE: method = "edges", edge_type = "upxo"

        Examples-6
        ---------
        CASE: method = "muledges", muledge_type = "upxo"

        Examples-7
        ---------
        CASE: method = "rings", ring_type = "upxo"

        Examples-8
        ---------
        CASE: method = "mulrings", mulring_type = "upxo"

        Examples-9
        ---------
        CASE: method = "partitions", partition_type = "upxo"

        Examples-10
        ---------
        CASE: method = "mulpartitions", partition_type = "upxo"

        Parameters
        ----------
        method : str, optional
            DESCRIPTION. The default is 'points'.
        points : list, optional
            DESCRIPTION. The default is None.
        point_type : str, optional
            DESCRIPTION. The default is 'upxo'.
        ckdtrees : list, optional
            DESCRIPTION. The default is None.
        ckdtree_workers : TYPE, optional
            DESCRIPTION. The default is 1.
        srtrees : list, optional
            DESCRIPTION. The default is None.
        mulpoints : list, optional
            DESCRIPTION. The default is None.
        mulpoint_type : str, optional
            DESCRIPTION. The default is 'upxo'.
        edges : list, optional
            DESCRIPTION. The default is None.
        edge_type : str, optional
            DESCRIPTION. The default is 'upxo'.
        muledges : list, optional
            DESCRIPTION. The default is None.
        muledge_type : str, optional
            DESCRIPTION. The default is 'upxo'.
        rings : list, optional
            DESCRIPTION. The default is None.
        ring_type : str, optional
            DESCRIPTION. The default is 'upxo'.
        partitions : list, optional
            DESCRIPTION. The default is None.
        partition_type : str, optional
            DESCRIPTION. The default is 'upxo'.
        mulpartitions : list, optional
            DESCRIPTION. The default is None.
        mulpartition_type : str, optional
            DESCRIPTION. The default is 'upxo'.
        cutoffshape : str, optional
            DESCRIPTION. The default is 'circle'.
        cut_off_radii : float, optional
            DESCRIPTION. The default is 1.0.
        r1 : float, optional
            DESCRIPTION. The default is 1.0.
        r2 : float, optional
            DESCRIPTION. The default is 1.0.
        a : float, optional
            DESCRIPTION. The default is 1.0.
        b : float, optional
            DESCRIPTION. The default is 1.0.
        angle : float, optional
            DESCRIPTION. The default is 0.0.
        comparison1 : str, optional
            DESCRIPTION. The default is 'le'.
        comparison2 : str, optional
            DESCRIPTION. The default is 'ge'.
        other_point_type : TYPE, optional
            DESCRIPTION. The default is 'self'.
        # LEAVE UNDOCUMENTED
        other_point : TYPE, optional
            DESCRIPTION. The default is [0.0, 0.0]  # LEAVE UNDOCUMENTED.

        Returns
        -------
        POINTS : TYPE
            DESCRIPTION.
        NPOINTS : TYPE
            DESCRIPTION.
        INDICES : TYPE
            DESCRIPTION.
        DISTANCES : TYPE
            DESCRIPTION.

        """
        if method == 'points' and point_type == 'upxo':
            """
            from point2d_04 import point2d
            import datatype_handlers as dth

            n, nsets, = 50, 2
            points = [dth.make_upxo_point2d_RANDU(n) for _ in range(nsets)]
            cut_off_radii = [0.25, 0.50]
            NP, N, I, D = point2d().find_neigh_points(method = 'points',
                                                points = points,
                                                point_type = 'upxo',
                                                cutoffshape = 'circle',
                                                cut_off_radii = cut_off_radii
                                                )
            neigh_points, npoints, indices, distances = NP, N, I, D
            print(len(neigh_points))
            print(len(indices))
            print(len(distances))
            print(neigh_points)
            print(indices)
            print(distances)
            """
            limit_usetree = 5*10**3
            # TODO: ABOVE LIMIT HAS TO BE DETERMINED AFTER TIME PROFILING !!
            POINTS, NPOINTS, INDICES, DISTANCES = [], [], [], []
            for _points, _r in zip(points, cut_off_radii):
                if len(_points) <= limit_usetree:
                    if not isinstance(_points, np.ndarray):
                        _points = np.array(_points)
                    if other_point_type == 'self':
                        distances = np.array(self.distance(other_object_type='point2d_list',
                                                           point_data=_points)
                                             )
                    else:
                        distances = np.array(other_point.distance(other_object_type='point2d_list',
                                                                  point_data=_points)
                                             )
                    indices = distances <= _r
                    if cutoffshape == 'circle':
                        if comparison1 == 'le':
                            indices = distances <= _r
                        elif comparison1 == 'lt':
                            indices = distances <= _r
                    _indices = [i for i, _ in enumerate(indices) if _]
                    POINTS.append(list(_points[indices]))
                    NPOINTS.append(len(_indices))
                    INDICES.append(_indices)
                    DISTANCES.append(list(distances[indices]))
                if len(_points) > limit_usetree:
                    if other_point_type == 'self':
                        POINTS, NPOINTS, INDICES, DISTANCES = self.find_neigh_points(method='ckdtrees',
                                                                                     points=points,
                                                                                     point_type='upxo',
                                                                                     cutoffshape='circle',
                                                                                     cut_off_radii=cut_off_radii,
                                                                                     ckdtree_workers=ckdtree_workers,
                                                                                     other_point_type=other_point_type,
                                                                                     other_point=other_point
                                                                                     )
                    else:
                        POINTS, NPOINTS, INDICES, DISTANCES = other_point.find_neigh_points(method='ckdtrees',
                                                                                            points=points,
                                                                                            point_type='upxo',
                                                                                            cutoffshape='circle',
                                                                                            cut_off_radii=cut_off_radii,
                                                                                            ckdtree_workers=ckdtree_workers,
                                                                                            other_point_type=other_point_type,
                                                                                            other_point=other_point
                                                                                            )
            return POINTS, NPOINTS, INDICES, DISTANCES
        if method == 'ckdtrees' and point_type == 'upxo':
            """
            from point2d_04 import point2d
            n, nsets, = 50, 2
            points = [dth.make_upxo_point2d_RANDU(n) for _ in range(nsets)]
            cut_off_radii = [0.25, 0.50]
            NP, N, I, D = point2d().find_neigh_points(method = 'ckdtrees',
                                                                                    points = points,
                                                                                    point_type = 'upxo',
                                                                                    cutoffshape = 'circle',
                                                                                    cut_off_radii = cut_off_radii,
                                                                                    ckdtree_workers = 1,
                                                                                    other_point_type = 'self',
                                                                                    other_point = None
                                                                                    )
            neigh_points, npoints, indices, distances = NP, N, I, D
            print(len(neigh_points))
            print(len(indices))
            print(len(distances))
            print(neigh_points)
            print(indices)
            print(distances)
            """
            from scipy.spatial import cKDTree as ckdt
            # Use the right reference point coordinates
            if other_point_type == 'self':
                _x, _y = deepcopy(self.x), deepcopy(self.y)
            else:
                _x, _y = other_point[0], other_point[1]
            POINTS, NPOINTS, INDICES, DISTANCES = [], [], [], []
            for _points_, r in zip(points, cut_off_radii):
                # Make scipy ckdtree for the current point list
                ckdt = dth.UpxoPointList_to_ckdtree(_points_)
                # Identify the neighbouring point indices
                indices = ckdt.query_ball_point(
                    [_x, _y], r, workers=ckdtree_workers)
                # Get their coordinates and calculate distances from self
                # or reference points (other_point!!!)
                __locxy = ckdt.data[indices].T
                distances = np.sqrt((_x-__locxy[0])**2+(_y-__locxy[0])**2)
                POINTS.append(list(np.array(_points_)[indices]))
                NPOINTS.append(len(distances))
                INDICES.append(indices)
                DISTANCES.append(distances)
            return POINTS, NPOINTS, INDICES, DISTANCES
        if method == 'srtrees' and point_type == 'upxo':
            '''
            Shapely tree structure
            '''
            pass
        if method in ('mulpoints', 'mulpoint', 'mps',
                      'mp', 'upxo_mulpoint', 'unpxo_mp',
                      'mp2', 'mp2d', 'mulpoints2d') and point_type == 'upxo':
            '''
            We first make the background data needed for the example.

            from point2d_04 import point2d
            from mulpoint2d_3 import mulpoint2d
            randd, n_points = np.random.uniform, 100
            points = [[point2d(x=randd(),
                               y=randd()) for _ in range(n_points)] for _ in
                      range(2)]
            mpset = [mulpoint2d(method = 'points',
                                point_objects = _points) for _points in
                     points]
            refpoint = point2d()

            # We will start with the example now.
            NP,N,I,D = refpoint.find_neigh_points(method='mulpoints',
                                                  mulpoints=mpset,
                                                  point_type='upxo',
                                                  cutoffshape='circle',
                                                  cut_off_radii=cut_off_radii,
                                                  ckdtree_workers=1,
                                                  other_point_type='self',
                                                  other_point = None)
             neigh_points, npoints, indices, distances = NP, N, I, D
            '''
            if other_point_type == 'self':
                ref_point = self
            else:
                other_point = other_point
            # mulpoints = mpset
            ckdtress = [mulpoint.maketree(treeType='ckdtree',
                                          saa=False,
                                          throw=True) for mulpoint in mulpoints]
            neigh_points, npoints, indices, distances = point2d().find_neigh_points(method='ckdtrees',
                                                                                    points=points,
                                                                                    point_type='upxo',
                                                                                    cutoffshape='circle',
                                                                                    cut_off_radii=cut_off_radii,
                                                                                    ckdtree_workers=1,
                                                                                    other_point_type='self',
                                                                                    other_point=None
                                                                                    )
            neigh_points, npoints, indices, distances = point2d().find_neigh_points(method='ckdtrees',
                                                                                    points=points,
                                                                                    point_type=point_type,
                                                                                    cutoffshape=cutoffshape,
                                                                                    cut_off_radii=cut_off_radii,
                                                                                    ckdtree_workers=ckdtree_workers,
                                                                                    other_point_type=other_point_type,
                                                                                    other_point=None
                                                                                    )
        if method == 'edges':
            pass
        if method == 'muledges':
            pass
        if method == 'rings':
            pass
        if method == 'mulrings':
            pass
        if method == 'partitions':
            pass
        if method == 'mulpartitions':
            pass

    def find_parent_mulpoints(self,
                              mulpoints: list
                              ):
        """
        find the mul-point from the list of mulpoints which contains this
        point with tol_dist = 0.0

        Parameters
        ----------
        mulpoints : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_containing_mulpoints(self,
                                  mulpoints: list
                                  ):
        """
        use convex hull polygon of the mulpoint

        Parameters
        ----------
        mulpoints : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_nearest_mulpoints(self,
                               mulpoints: list
                               ):
        """
        Use tree to find this out.

        Parameters
        ----------
        mulpoints : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_neigh_mulpoints(self,
                             mulpoints: list
                             ):
        """
        Use tree to find this out.

        Parameters
        ----------
        mulpoints : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_neigh_edges(self,
                         edges: list
                         ):
        """
        Find edges with normal distance less than or equal to tol_dist

        Parameters
        ----------
        edges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_parent_edges(self,
                          edges: list
                          ):
        """
        Find parent edges which contain this point

        Parameters
        ----------
        edges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_nearest_edges(self,
                           edges: list
                           ):
        """
        Find the edge nearest to this point

        Parameters
        ----------
        edges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_containing_muledges(self,
                                 muledges: list
                                 ):
        """
        Find the muledge, on one of whos edge, the point lies within
        the tolerance distance

        Parameters
        ----------
        muledges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_bounding_muledges(self,
                               muledges: list
                               ):
        """
        Find the muledge, who's convex hull contains the point object

        Parameters
        ----------
        muledges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_nearest_muledges(self,
                              muledges: list
                              ):
        """
        Use the hauseldroff distaince from the point and rthe vertices
        of the muledge

        Parameters
        ----------
        muledges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_parent_partitions(self,
                               partitions: list
                               ):
        pass

    def find_neigh_partitions(self,
                              partitions: list
                              ):
        pass

    def find_nearest_partitions(self,
                                partitions: list
                                ):
        pass

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point2d_lean_highest():
    '''
    UPXO core class.
    Represents a point in 2D Cartesian space.
    Leanness is highest, meaning only stores the coordinates x and y

    NOTES TO USER:
        1. x and y must be floats
    '''
    __slots__ = ('x', 'y')

    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point2d_lean_highest_mc0():
    '''
    UPXO core class.
    Represents a point in 2D Cartesian space, made for Monte-Carlo simulation
    Leanness is highest, meaning only stores the coordinates x and y and the
    integer state, s

    NOTES TO USER:
        1. x and y must be floats
        2. s must be an int
    '''
    __slots__ = ('x', 'y', 's')

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 s: int = 1
                 ):
        self.x, self.y, self.s = x, y, s
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point2d_lean_highest_mc1():
    '''
    UPXO core class.
    Represents a point in 2D Cartesian space, made for Monte-Carlo simulation
    Leanness is highest, meaning only stores the coordinates x and y and the
    integer state, s

    NOTES TO USER:
        1. x and y must be floats
        2. s must be an int
        3. h. no input necessary. Taken internally
    '''
    __slots__ = ('x', 'y', 's', 'h')

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 s: int = 1,
                 h: float = 0.0
                 ):
        self.x, self.y, self.s, self.h = x, y, s, h
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point3d():
    '''
    UPXO core class.
    Represents a point in 2D Cartesian space.
    '''
    # -----------------------------------------------------
    """Number of decimal places to round off the very small number to zero"""
    ROUND_ZERO_DEC_PLACE = 10
    # -----------------------------------------------------
    __slots__ = ('dim',  # Dimension of the model
                 'lean',  # Leanness of the point object
                 'rid',  # random id of the point
                 'mid',  # memory id of the point
                 'x',  # x-coordinate
                 'y',  # y-coordinate
                 'z',  # z-coordinate
                 'lean',  # Complexity branching
                 '_mulpoints_',  # list of multi-point object
                 '_edges_',  # list of edge object
                 '_muledges_',  # list of multi-edge object
                 '_surfaces_',  # list of surface object
                 '_mulsurfaces_',  # list of multi-surface object
                 '_xtals_',  # list of xtal object
                 '_polyxtals_',  # list of poly-xtal object
                 'ptype',  # point type
                 'jn',  # Junction order
                 'loc',  # Location of the point in pxtal
                 'store_original_coord',
                 '_original_location',  # original x and y
                 'phase_id',  # Phase id
                 'phase_name',  # Phase name. Defaults to CuCrZr
                 'tcname',  # texture component name
                 'sfv_pol_area',  # Xtal area
                 'sfv_repr_ea',  # Euler angle representation (ex: 'Bunge')
                 'ea',  # Euler angle in degrees
                 'oo',  # UPXO Orientation object
                 'tdist',  # for use in comparison operations
                 'partition',  # Dataclass instance to store buffer polygon objects
                 'vprop'  # Visialuzation properties
                 )
    # -----------------------------------------------------
    # OPTIONS for some above slotted variables: SEE END OF FILE
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF DUNDERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0,
                 lean: str = 'lowest',
                 set_rid: bool = False,
                 rid_length: int = 4,
                 set_mid: bool = False,
                 set_dim: bool = False,
                 dim: int = 2,
                 set_ptype: bool = False,
                 ptype: str = 'vt2dseed',
                 set_jn: bool = False,
                 jn: int = 3,
                 set_loc: bool = False,
                 loc: str = 'interior',
                 store_original_coord: bool = False,
                 attach_mulpoints: bool = False,
                 attach_edges: bool = False,
                 attach_muledges: bool = False,
                 attach_xtals: bool = False,
                 attach_polyxtals: bool = False,
                 set_phase: bool = False,
                 phase_id: int = 1,
                 phase_name: str = 'UPXO',
                 set_sfv_pol_area: bool = True,
                 sfv_pol_area: int = 0,
                 set_tcname: bool = False,
                 tcname: str = 'B',
                 set_ea: bool = True,
                 sfv_repr_ea='Bunge',
                 ea: list = [45, 35, 0],
                 set_oo: bool = True,
                 oo: object = None,
                 set_tdist: bool = True,
                 tdist: float = 0.0000000000001,
                 store_vis_prop: bool = False,
                 make_partition: bool = False,
                 partition_n=4,
                 partition_char_lengths=[1.0],
                 partition_char_length_type='radius',
                 partition_make_polygon=True,
                 partition_tool='mpl',
                 partition_rotation=0,
                 feed_partition=False,
                 feed_partition_name='grain',
                 feed_partition_tool='shapely',
                 feed_partition_polygon=None
                 ):
        '''
        Exaplantions:
            1.
            2.
        '''
        if lean in ('ignore'):
            self.x, self.y, self.z = float(x), float(y), float(z)
            self.lean = lean  # TODO: MAKE SETTER
            self.make_mid()
            self.dim = dim
            self.set_ptype(ptype=ptype)
            self.set_jn(jn=jn)
            self.set_location(loc=loc)
            self.store_original_coord = store_original_coord
            self._original_location = (self.x, self.y)
            self.set_phase(phase_id=phase_id, phase_name=phase_name)
            self.sfv_pol_area = sfv_pol_area
            self.tcname = tcname
            self.sfv_repr_ea = sfv_repr_ea
            self.ea = [45, 35, 0]
            self.oo = oo
            self.tdist = tdist
            self.vprop = self.set_vis_prop()
            # self.make_partition(restart = False,
            #                    saa = True,
            #                    n = partition_n,
            #                    char_lengths = partition_char_lengths,
            #                    char_length_type = partition_char_length_type,
            #                    make_polygon = partition_make_polygon,
            #                    tool = partition_tool,
            #                    rotation = partition_rotation,
            #                    feed_partition = False,
            #                    feed_partition_name = 'grain',
            #                    feed_partition_tool = 'shapely',
            #                    feed_partition_polygon = None
            #                    )
        if lean in ('lowest'):
            self.x, self.y, self.z = float(x), float(y), float(z)
            self.lean = lean
            if set_rid:
                self.make_rid(idlength=rid_length)
            if set_mid:
                self.make_mid()
            if set_dim:
                self.dim = dim
            if set_ptype:
                self.set_ptype(ptype=ptype)
            if set_jn:
                self.set_jn(jn=jn)
            if set_loc:
                self.set_location(loc=loc)
            if store_original_coord:
                self.store_original_coord = store_original_coord
                self._original_location = (self.x, self.y)
            if attach_mulpoints:
                pass
            if attach_edges:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if set_phase:
                self.set_phase(phase_id=phase_id, phase_name=phase_name)
            if set_sfv_pol_area:
                self.sfv_pol_area = sfv_pol_area
            if set_tcname:
                self.tcname = tcname
            if set_ea:
                self.sfv_repr_ea = sfv_repr_ea
                self.ea = [45, 35, 0]
            if set_oo:
                self.oo = oo
            if set_tdist:
                self.tdist = tdist
            if store_vis_prop:
                self.vprop = self.set_vis_prop()
            if make_partition:
                self.make_partition(restart=False,
                                    saa=True,
                                    n=partition_n,
                                    char_lengths=partition_char_lengths,
                                    char_length_type=partition_char_length_type,
                                    make_polygon=partition_make_polygon,
                                    tool=partition_tool,
                                    rotation=partition_rotation,
                                    feed_partition=False,
                                    feed_partition_name='grain',
                                    feed_partition_tool='shapely',
                                    feed_partition_polygon=None
                                    )
        if lean in ('low'):
            self.x, self.y, self.z = float(x), float(y), float(z)
            self.lean = lean
            if set_rid:
                self.make_rid(idlength=rid_length)
            if set_mid:
                self.make_mid()
            if set_dim:
                self.dim = dim
            if set_ptype:
                self.set_ptype(ptype=ptype)
            if set_jn:
                self.set_jn(jn=jn)
            if set_loc:
                self.set_location(loc=loc)
            if store_original_coord:
                self.store_original_coord = store_original_coord
                self._original_location = (self.x, self.y)
            if attach_mulpoints:
                pass
            if attach_edges:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if set_phase:
                self.set_phase(phase_id=phase_id, phase_name=phase_name)
            if set_sfv_pol_area:
                self.sfv_pol_area = sfv_pol_area
            if set_tcname:
                self.tcname = tcname
            if set_ea:
                self.sfv_repr_ea = sfv_repr_ea
                self.ea = [45, 35, 0]
            if set_oo:
                self.oo = oo
            if set_tdist:
                self.tdist = tdist
            if store_vis_prop:
                self.vprop = self.set_vis_prop()
        if lean in ('highest'):
            self.x, self.y = float(x), float(y)
    # -----------------------------------------------------

    def __call__(self, x, y):
        '''Call point object instances like functions'''
        self.x, self.y, self.z = x, y, z
    # -----------------------------------------------------

    def __eq__(self,
               point_objects=None,
               tdist=0.0,
               use_self_tdist=True
               ):
        '''
        True if distance b/w self and point_objects is LE tolerance distance
        '''
        if use_self_tdist:
            tdist = self.tdist
        if isinstance(point_objects, point2d):
            if self.distance(other_object_type='point3d',
                             point_data=point_objects) <= tdist:
                return True
            else:
                return False
        elif isinstance(point_objects, list):
            to_return = [False for pobjcount in range(len(point_objects))]
            for pobjcount in range(len(point_objects)):
                if self.distance(point_objects[pobjcount]) <= tdist:
                    to_return[pobjcount] = True
            return tuple(to_return)
        else:
            return f'ERROR: Only point2d instance or tuple of them allowed'
    # -----------------------------------------------------

    def __ne__(self,
               point_objects=None,
               tdist=0.0,
               use_self_tdist=True
               ):
        '''
        True if distance b/w self and point_object is GT tolerance distance
        '''
        if use_self_tdist:
            tdist = self.tdist
        if isinstance(point_objects, point2d):
            if self.distance(other_object_type='point3d',
                             point_data=point_objects) > tdist:
                return True
            else:
                return False
        elif isinstance(point_objects, list):
            to_return = [False for pobjcount in range(len(point_objects))]
            for pobjcount in range(len(point_objects)):
                if self.distance(point_objects[pobjcount]) > tdist:
                    to_return[pobjcount] = True
            return tuple(to_return)
        else:
            return f'ERROR: Only point2d instance or tuple of them allowed'
    # -----------------------------------------------------

    def __add__(self,
                toadd=0.0,
                copy_=True
                ):
        '''
        CASE-1: if toadd is instance of point2d:
            addition over fundamental axes lengths
        CASE-2: if toadd is single scalar:
            scalar added to both fundamental axes lengths
        CASE-3: if toadd is two scalar element tuple:
        CASE-3: if toadd is tuple of scalars or point2d objects:
            scalar added to both fundamental axes lengths and a
            tuple of point2d objects is returned. Element wise check
            of scalar or point2d is carried out.
        CASE-4: if toadd is a tuple of tuples, then the inner tuple,
        in the form (x, y) is added to the x and y of self and the
        corresponding point2d object is made and appended to tuple to
        be returned.

        copy_ applies only to cases 1,2 and 3 of above 4 cases

        RESTRICTIONS:
            toadd: if not cases 1 and 2, toadd MUST be in tuple format
        '''
        if isinstance(toadd, point3d):  # CASE-1
            if copy_:
                return self.make_new(self.x+toadd.x, self.y+toadd.y, self.z+toadd.z)
            else:
                self.x += toadd.x
                self.y += toadd.y
        if isinstance(toadd, int) or isinstance(toadd, float):
            if copy_:
                return self.make_new(self.x+toadd, self.y+toadd)
            else:
                self.x += toadd
                self.y += toadd
        if isinstance(toadd, tuple):
            to_return = ()
            # count = 0
            for toadd_ in toadd:
                if isinstance(toadd_, int) or isinstance(toadd_, float):
                    to_return += (self.make_new(self.x+toadd_,
                                  self.y+toadd_, self.z+toadd_),)
                elif isinstance(toadd_, tuple):
                    to_return += (self.make_new(self.x +
                                  toadd_[0], self.y+toadd_[1], self.z+toadd_[2]),)
                elif isinstance(toadd_, point2d):
                    to_return += (self.make_new(self.x+toadd_.x,
                                  self.y+toadd_.y, self.z+toadd_.z),)
            #    count+=1
            return to_return
    # -----------------------------------------------------

    def __sub__(self,
                tosub=0.0,
                copy_=True
                ):
        '''
        CASE-1: if tosub is instance of point2d:
            addition over fundamental axes lengths
        CASE-2: if tosub is single scalar:
            scalar added to both fundamental axes lengths
        CASE-3: if tosub is two scalar element tuple:
        CASE-3: if tosub is tuple of scalars or point2d objects:
            scalar added to both fundamental axes lengths and a
            tuple of point2d objects is returned. Element wise check
            of scalar or point2d is carried out.
        CASE-4: if tosub is a tuple of tuples, then the inner tuple,
        in the form (x, y) is added to the x and y of self and the
        corresponding point2d object is made and appended to tuple to
        be returned.

        copy_ applies only to cases 1,2 and 3 of above 4 cases

        RESTRICTIONS:
            tosub: if not cases 1 and 2, tosub MUST be in tuple format
        '''
        if isinstance(tosub, point3d):  # CASE-1
            if copy_:
                return self.make_new(self.x-tosub.x, self.y-tosub.y, self.z-tosub.z)
            else:
                self.x -= tosub.x
                self.y -= tosub.y
                self.z -= tosub.z
        if isinstance(tosub, int) or isinstance(tosub, float):
            if copy_:
                return self.make_new(self.x-tosub, self.y-tosub, self.z-tosub)
            else:
                self.x -= tosub
                self.y -= tosub
                self.z -= tosub
        if isinstance(tosub, tuple):
            to_return = ()
            count = 0
            for tosub_ in tosub:
                if isinstance(tosub_, int) or isinstance(tosub_, float):
                    to_return += (self.make_new(self.x-tosub_,
                                  self.y-tosub_, self.z-tosub_),)
                elif isinstance(tosub_, tuple):
                    to_return += (self.make_new(self.x -
                                  tosub_[0], self.y-tosub_[1], self.z-tosub_[2]),)
                elif isinstance(tosub_, point3d):
                    to_return += (self.make_new(self.x-tosub_.x,
                                  self.y-tosub_.y, self.z-tosub_.z),)
                count += 1
                if count == len(tosub):
                    return to_return
    # -----------------------------------------------------

    def __mul__(self,
                multiplier=1.0
                ):
        ''''''
        if isinstance(multiplier, float) or isinstance(multiplier, int):
            return self.make_new(self.x*multiplier, self.y*multiplier, self.z*multiplier)
        elif isinstance(multiplier, point3d):
            return self.make_new(self.x*multiplier.x, self.y*multiplier.y, self.z*multiplier.z)
    # -----------------------------------------------------

    def __truediv__(self,
                    divisor=1.0
                    ):
        ''''''
        if isinstance(divisor, float) or isinstance(divisor, int):
            if not divisor == 0:
                return self.make_new(self.x/divisor, self.y/divisor, self.z/divisor)
            else:
                return f'ERROR: Division by zero'
        elif isinstance(divisor, point3d):
            return self.make_new(self.x/divisor.x, self.y/divisor.y, self.z/divisor.z)
    # -----------------------------------------------------
    # -----------------------------------------------------

    def __repr__(self):
        '''
        instance representation.
        '''
        return f'upxo.p3d({round(self.x, 5)}, {round(self.y, 5)}, {round(self.z, 5)})'
    # -----------------------------------------------------

    def make_mid(self):
        '''
        assign_mid Assign the memory address id of self to self.mid
        '''
        self.mid = id(self)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF STATUS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    @property
    def _status_(self):
        '''
        Status update to developer. Not aimed at users
        '''
        if self.has('mid'):
            print(f'mid = {self.rid}')
        else:
            print(f'mid does not exist. It will be made now.')
            self.make_mid()
            self._status_
        print('___coordinates___')
        print('(x, y) = ({self.x}, {self.y})')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF STATUS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF ID AND POINT TYPE ASSIGNMENT METHODS
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # START OF BORROWERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def borrow_tdist(self,
                     point_object=None
                     ):
        if hasattr(point_object, 'tdist'):
            self.tdist = point_object.tdist
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # END OF BORROWERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # START OF HASSERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def has(self, suffix):
        suffix = suffix.lower()
        if suffix in ('tdist', 'toldist', 'tolerance'):
            suffix = 'tdist'
        elif suffix in ('jn', 'bjn', 'bj_n', 'j_n', 'jporder', 'xvo', 'xtal_vertex_order'):
            suffix = 'jn'
        elif suffix in ('ptype', 'point_type'):
            suffix = 'ptype'
        elif suffix in ('loc', 'pxtal_loc'):
            suffix = 'loc'
        elif suffix in ('rid', 'randid', 'randomid', 'random_id'):
            suffix = 'rid'
        elif suffix in ('mid', 'omid', 'memory_id', 'mem_id', 'memid', 'object_memory_id'):
            suffix = 'mid'
        elif suffix in ('dim', 'dimensionality'):
            suffix = 'dim'
        elif suffix in ('sfv_pol_area', 'area'):
            suffix = 'sfv_pol_area'
        elif suffix in ('ar', 'aspect_ratio'):
            suffix = 'ar'
        elif suffix in ('phase_id', 'phase_id', 'phase_id_number'):
            suffix = 'phase_id'
        elif suffix in ('phase_name', 'phasename', 'phase_name', 'phase'):
            suffix = 'phase_name'
        elif suffix in ('tcname', 'texcomp', 'texture_component', 'ori_name', ''):
            suffix = 'tcname'
        elif suffix in ('eaunit', 'euler_angle_unit', 'eaunits', 'euler_angle_units'):
            suffix = 'eaunit'
        elif suffix in ('sfv_repr_ea', 'eatype', 'ea_repr', 'ea_type', 'ea_representation'):
            suffix = 'sfv_repr_ea'
        elif suffix in ('ea', 'ea_val', 'eaval', 'ea_value', 'eulerangle', 'orientation'):
            suffix = 'ea'
        # ...................
        has_truth = False
        if hasattr(self, suffix):
            has_truth = True
        return has_truth
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF HASSERS
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF OBJECT RE-CREATION CALCULATION METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def make_new(self,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 lean='low',
                 set_mid=True,
                 set_dim=True,
                 set_ptype=False,
                 store_original_coord=True,
                 set_tdist=True,
                 tdist=0.0000000000001,
                 ):
        '''
        Create a new point object.
        '''
        return point3d(x=x,
                       y=y,
                       z=z,
                       lean=lean,
                       set_mid=set_mid,
                       set_dim=set_dim,
                       set_ptype=set_ptype,
                       store_original_coord=store_original_coord,
                       set_tdist=set_tdist,
                       tdist=tdist,
                       )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF OBJECT RE-CREATION CALCULATION METHODS
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF KINEMATICS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def negx(self):
        ''' Mirror about y-axis '''
        self.x = -self.x
        return self

    def negy(self):
        ''' Mirror about x-axis '''
        self.y = -self.y
        return self

    def negz(self):
        ''' Mirror about x-axis '''
        self.z = -self.z
        return self

    def negxy(self):
        ''' Mirror about y = -x axis '''
        self.x, self.y = -self.x, -self.y
        return self

    def negyz(self):
        ''' Mirror about y = -x axis '''
        self.y, self.z = -self.y, -self.z
        return self

    def negzx(self):
        ''' Mirror about y = -x axis '''
        self.z, self.x = -self.z, -self.x
        return self

    def negxyz(self):
        ''' Mirror about y = -x axis '''
        self.x, self.y, self.z = -self.x, -self.y, -self.z
        return self
    # -----------------------------------------------------

    def displace_by(self,
                    delx: float = 0.0,
                    dely: float = 0.0,
                    delz: float = 0.0
                    ):
        '''
        Move self by xdisp and ydisp
        '''
        self.x, self.y, self.z = self.x+delx, self.y+dely, self.z+delz
    # -----------------------------------------------------

    def move_to(self,
                xlocation: float = 0.0,
                ylocation: float = 0.0,
                zlocation: float = 0.0
                ):
        '''
        Move self to (locx, locy)
        '''
        self.x, self.y, self.z = xlocation, ylocation, zlocation
    # -----------------------------------------------------

    def align_to(self,
                 method='point2d',
                 ref_point_object=None,
                 xlocation: float = None,
                 ylocation: float = None,
                 zlocation: float = None
                 ):
        '''
        Align self to a new point

        NOTE: This is akin to alignTo of vedo. See below.
        https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points
        '''
        if method == 'point2d':
            self.x, self.y, self.z = ref_point_object.x, ref_point_object.y, ref_point_object.z
        elif method == 'coord':
            self.x, self.y, self.z = xlocation, ylocation, zlocation
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF KINEMATICS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#################################################################################
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF RESET METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def reset(self):
        '''
        Reset's location of self to UPXO default point2d location'
        '''
        if self.store_original_coord:
            if hasattr(self, '_original_location'):
                self.x, self.y, self.z = self._original_location
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF RESET METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF ATTACH METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def attach_mulpoint(self,
                        mulpointObject=None
                        ):
        '''
        Attach this object to a multi-point object
        '''
        self._mulpoints_ = mulpointObject
    # -----------------------------------------------------

    def attach_edge(self,
                    edgeObject=None
                    ):
        '''
        Attach this object to an edge object
        '''
        self._edges_ = edgeObject
    # -----------------------------------------------------

    def attach_surface(self,
                       surfaceObject=None
                       ):
        '''
        Attach this object to a surface object
        '''
        self._surfaces_ = surfaceObject
    # -----------------------------------------------------

    def attach_surface(self,
                       volume_object=None
                       ):
        '''
        Attach this object to the volume object
        '''
        pass
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF ATTACH METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF SETTERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def set_dim(self,
                dim=3):
        self.dim = dim
    # -----------------------------------------------------

    def set_tdist(self,
                  tdist=0.0):
        self.tdist = tdist
    # -----------------------------------------------------

    def set_jn(self,
               jn=3
               ):
        """
        jtype: number of l0e emanating from it / terminating in it.
        """
        self.jn = jn
    # -----------------------------------------------------

    def set_ptype(self,
                  ptype='jp'
                  ):
        '''
        Assign which type of point it is going to be or it is.
        CASES INCLUDE THE FOLLOWING:
            1. jp: Junction poinzt
            2. gbp: grain boundary point
            3. # TODO: write the many more that there are.
        '''
        self.ptype = ptype
    # -----------------------------------------------------

    def set_location(self,
                     loc='pxtal_internal'
                     ):
        '''
        assign a desctiptive 1-word string.
        Suggested:
            1. pxtal_internal / pxtal_boundary
            2. on_grain_boundary
            3. particle_seed
            4. VTGS_seed
            5. voxel_centroid
            6. voxel_corners
        '''
        self.loc = loc
    # -----------------------------------------------------

    def set_vis_prop(self,
                     mtype='o',
                     mew=1.0,
                     mec='k',
                     msz=10,
                     mfill='w',
                     malpha=1.0,
                     bfill='teal'
                     ):
        """
        set_marker_prop Set some properties of markers

        Args:
            mtype (str): marker type. Defaults to 'o'.
            mew (float): marker edge width. Defaults to 1.0.
            mec (str): marker edge colour. Defaults to 'k'.
            msz (int): marker size. Defaults to 10.
            mfill (str): marker fill. Defaults to 'w'.
            malpha (float): marker alpha. Defaults to 1.0.
            bfill (float): marker buffer fill. Defaults to 'teal'.

        Returns:
            dict: A dictionary of visualization parameters
        """
        vprop = {'mtype': mtype,
                 'mew': mew,
                 'mec': mec,
                 'msz': msz,
                 'mfill': mfill,
                 'malpha': malpha,
                 'bfill': bfill
                 }
        return vprop
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF SCALAR ASSIGNMENT METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def set_area(self,
                 value: float = 0.0
                 ):
        self.area = area
    # ---------------------------------------

    def set_ar(self,
               value: float = 0.0,
               ):
        self.ar = ar
    # ---------------------------------------

    def set_morph_curvature(self,
                            value: float = 0.0,
                            ):
        pass
    # ---------------------------------------

    def set_gnd(self,
                value: float = 0.0,
                ):
        pass
    # ---------------------------------------

    def set_lattice_curvature_(self,
                               value: list[list] = [[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0]
                                                    ]
                               ):
        pass
    # ---------------------------------------

    def set_phase(self,
                  phase_id: int = 0,
                  phase_name: str = 'cucrzr'
                  ):
        """
        set_phase Assign phase id and phase name to point

        Args:
            pid (int, optional): _description_. Defaults to 0.
            pname (str, optional): _description_. Defaults to 'cucrzr'.
        """
        self.phase_id = phase_id
        self.phase_name = phase_name
    # ---------------------------------------

    def set_tcname(self,
                   name: str = 'B'
                   ):
        """
        set_tcname Assign texture component name to point

        Args:
            name (str, optional): _description_. Defaults to 'B'.
        """
        self.tcname = name
    # ---------------------------------------

    def set_sfv_repr_ea(self,
                        sfv_repr_ea='Bunge'
                        ):
        """
        sfv_repr_ea Assign the default representation of Euler angles

        Args:
            repr (str, optional): _description_. Defaults to 'Bunge'.
        """
        self.sfv_repr_ea = sfv_repr_ea
    # ---------------------------------------

    def set_sfv_ea1(self,
                    ea1: float = 0,
                    ):
        """
        ea1 assign 1st Euler angle to point object

        Args:
            ea1 (float, optional): _description_. Defaults to 0.
        """
        self.ea[0] = ea1
    # ---------------------------------------

    def set_sfv_ea2(self,
                    ea2: float = 0,
                    ):
        """
        ea2 assign 2nd Euler angle to point object

        Args:
            ea2 (float, optional): _description_. Defaults to 0.
        """
        self.ea[1] = ea2
    # ---------------------------------------

    def set_sfv_ea3(self,
                    ea3: float = 0,
                    ):
        """
        ea3 assign 3rd Euler angle to point object

        Args:
            ea2 (float, optional): _description_. Defaults to 0.
        """
        self.ea[2] = ea3
    # ---------------------------------------

    def set_sfv_ea(self,
                   ea: list = [0, 0, 0]
                   ):
        """
        ea Assign the three Euler angles to point object

        Args:
            ea (list, optional): _description_. Defaults to [0, 0, 0].
        """
        self.ea = ea
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF SCALAR ASSIGNMENT METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point3d_lean_highest():
    """
    UPXO core class.
    Represents a point in 3D Cartesian space.
    Leanness is highest, meaning only stores the coordinates x, y and z

    NOTES TO USER:
        1. x, y and z must be floats
    """
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point3d_lean_highest_mc0():
    """
    UPXO core class.
    Represents a point in 3D Cartesian space, made for Monte-Carlo simulation
    Leanness is highest, meaning only stores the coordinates x and y and the
    integer state, s

    NOTES TO USER:
        1. x, y and z must be floats
        2. s must be an int
    """
    __slots__ = ('x', 'y', 'z', 's')

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0,
                 s: int = 1
                 ):
        self.x, self.y, self.z, self.s = x, y, z, s
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point3d_lean_highest_mc1():
    """
    UPXO core class.
    Represents a point in 3D Cartesian space, made for Monte-Carlo simulation
    Leanness is highest, meaning only stores the coordinates x and y and the
    integer state, s

    NOTES TO USER:
        1. x, y and z must be floats
        2. s must be an int
        3. h. no input necessary. Taken internally
    """
    __slots__ = ('x', 'y', 'z', 's', 'h')

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0,
                 s: int = 1,
                 h: float = 0.0
                 ):
        self.x, self.y, self.z = x, y, z
        self.s, self.h = s, h
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point_q_space():
    """
    Class representing Quaternion space
    """
    __slots__ = ('q1', 'q2', 'q3', 'q4')

    def __init__(self,
                 q1=0.0,
                 q2=0.0,
                 q3=0.0,
                 q4=0.0
                 ):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point_be_space():
    """
    Class representing Bunge's Euler space
    """
    __slots__ = ('phi1', 'psi', 'phi2')

    def __init__(self,
                 phi1=0.0,
                 psi=0.0,
                 phi2=0.0
                 ):
        self.phi1 = phi1
        self.psi = psi
        self.phi2 = phi2
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point_gsm_stat_space():
    """
    Class representing a sub-space of statistics in grain structure morphology
    space

    The sub-space can be used to represent grain structure parameters such as:
        area, perimeters, junction point order, grain boundary tortuosity, etc..
    """
    __slots__ = ('amin', 'amax', 'astd', 'avar',
                 'aper05', 'aper25', 'aper50', 'aper75', 'aper95',
                 'amodality', 'amodes', 'akurt', 'askew'
                 )

    def __init__(self,
                 minn=0.0,
                 maxx=100.0,
                 std=50.0,
                 var=10.0,
                 per05=5.0,
                 per25=25.0,
                 per50=50.0,
                 per75=75.0,
                 per95=95.0,
                 modality=2.0,
                 modes=[15.0, 60.0],
                 widths=[5.0, 20.0],
                 kurt=0.0,
                 skew=0.0,
                 ):
        self.minn = minn
        self.maxx = maxx
        self.std = std
        self.var = var
        self.per05 = per05
        self.per25 = per25
        self.per50 = per50
        self.per75 = per75
        self.per95 = per95
        self.modality = modality
        self.modes = modes
        self.widths = widths
        self.kurt = kurt
        self.skew = skew
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point_tc_stat_space():
    __slots__ = ('name',
                 'ghw',
                 'vfraction',
                 'intensity'
                 )

    def __init__(self,
                 name='',
                 ghw=5.0,
                 vfraction=0.1,
                 intensity=2.0,
                 ):
        self.name = name
        self.ghw = ghw
        self.vfraction = vfraction
        self.intensity = intensity
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
