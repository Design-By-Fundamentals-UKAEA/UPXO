__name__ = "UPXO-geoEntity"
__authors__ = ["Sunil Anandatheertha"]
__lead_developer__ = ["Vaasu Anandatheertha"]
__emails__ = ["vaasu.anandatheertha@ukaea.uk", ]
__version__ = ["0.1@ upto.271022",
               "0.2@ from.281022",
               "0.3@ from.091122",
               "0.4@ from.211122"]

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from upxo.geoEntities.point2d import Point2d as point2d
from upxo._sup import dataTypeHandlers as dth


class mulpoint2d():
    """
    CORE UPXO GEOMEWTRIC ENTITY CLASS.
    Author:
        Dr. Sunil Anandatheertha

    Slots
    -----
        * points.
        * mpoints.
        * mulpoint_type.
        * rid.
        * lean.
        * npoints.
        * locx.
        * locy.
        * locxy.
        * xpert.
        * ypert.
        * xbound.
        * ybound.
        * centroid.
        * duparray.
        * coupled_array.
        * tree.
        * pdom.
        * covhull.
        * __state_change. A flag variable to indicate a change in any of the
        constituent points object. Default should be False. A change should
        make this True. self.recompute_from_mids() to be implemented when True.
    """
    __POINT_TDIST__ = 0.001
    __CONST_PI__ = 3.141592653589793238
    __slots__ = ('points', 'mpoints', 'mulpoint_type', 'rid', 'lean',
                 'npoints', 'locx', 'locy', 'locxy', 'xpert', 'ypert',
                 'xbound', 'ybound', 'centroid', 'duparray',
                 'coupled_array', 'tree', 'pdom', 'covhull',
                 '__state_change',
                 )

    def __init__(self,
                 mulpoint_type: str = 'seed',
                 method: str = 'points',
                 gridding_technique: str = 'random',
                 sampling_technique: str = 'uniform',
                 nrndpnts: int = 25,
                 randuni_calc: str = 'by_points',
                 char_length_mean: float = 0.10,
                 char_length_min: float = 0.05,
                 char_length_max: float = 0.15,
                 n_trials: int = 10,
                 n_iterations: int = 10,
                 point_objects: list = [],
                 make_point_objects: bool = True,
                 mulpoint_objects: list = None,
                 coordx: list = [],
                 coordy: list = [],
                 coordxy: list = [],
                 space: str = 'linear',
                 xbound: list = [0, 1],
                 ybound: list = [0, 1],
                 char_length: list = [0.25, 0.25],
                 latvecs: list = [0.1, 0.1],
                 angles: list = [0, 60],
                 bridson_sampling_k: float = 30,
                 perturb_flag: bool = False,
                 perturb_type: str = 'local_uniform',
                 perturb_mag: list = [0.05, 0.05],
                 lean: str = 'veryhigh',
                 pdom: str = 'grain',
                 make_rid: bool = True,
                 make_ckdtree: bool = True,
                 vis: bool = False,
                 print_summary: bool = True
                 ):
        self.lean = lean
        self.xbound = xbound
        self.ybound = ybound
        self.mulpoint_type = mulpoint_type
        self.mpoints = None
        self.points = None
        if method in dth.opt.upxo_point2d:
            print(20*'>', 1)
            self.points = point_objects
            self.npoints = len(point_objects)
            self.locx, self.locy = zip(*np.array([(pobj.x, pobj.y)
                                                  for pobj in point_objects]))
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True,
                                  )
        elif method in dth.opt.coord_xlist_ylist:
            print(20*'>', 2)
            self.locx, self.locy = np.array(coordx), np.array(coordy)
            self.npoints = len(coordx)
            self.points = deque(point2d(x=_x, y=_y)
                                for (_x, _y) in zip(self.locx, self.locy))
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True,
                                  )
        elif method in dth.opt.coord_pairs_point2d_list:
            '''
            from upxo.geoEntities.mulpoint2d_old import mulpoint2d
            from upxo.geoEntities.point2d import Point2d as p2d
            import numpy as np
            xy = [[1, 9], [2, 5], [3, 7], [4, 4], [5, 3]]
            sp = mulpoint2d(method='xy_coord_pairs_list', coordxy=xy, lean='leanest')
            '''
            print(20*'>', 3)
            coordxy = np.array(coordxy)
            print(coordxy)
            self.locx, self.locy = coordxy[:, 0], coordxy[:, 1]
            self.npoints = len(self.locx)
            self.points = deque(point2d(x=_x, y=_y)
                                for (_x, _y) in zip(self.locx, self.locy))
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True,
                                  )
        elif method in dth.opt.coord_point2d_list:
            '''
            from upxo.geoEntities.mulpoint2d_old import mulpoint2d
            from upxo.geoEntities.point2d import Point2d as p2d
            import numpy as np
            xy = [[1, 9], [2, 5], [3, 7], [4, 4], [5, 3]]
            sp = mulpoint2d(method='xy_coord_pairs_list', coordxy=xy, lean='leanest')
            '''
            print(20*'>', 4)
            self.locx, self.locy = coordxy
            self.npoints = len(self.locx)
            self.points = deque(point2d(x=_x, y=_y)
                                for (_x, _y) in zip(self.locx, self.locy))
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True,
                                  )
        elif method in dth.opt.upxo_mp:
            print(20*'>', 5)
            self.mpoints = mulpoint_objects
            _points = []
            for mpobj in mulpoint_objects:
                for mpobj_point in mpobj.points:
                    _points.append(mpobj_point)
            self.points = _points
            self.npoints = len(self.points)
            self.locx, self.locy = zip(*np.array([(pobj.x, pobj.y)
                                                  for pobj in self.points]))
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True,
                                  )
        elif method in dth.opt.upxo_mp_grid_type:
            '''
            Generate triangular, rectangular and hexagonal grid.
            '''
            # Make grid
            self.makegrid(method=method,
                          gridding_technique=gridding_technique,
                          sampling_technique=sampling_technique,
                          nrndpnts=nrndpnts,
                          randuni_calc=randuni_calc,
                          char_length_mean=char_length_mean,
                          char_length_min=char_length_min,
                          char_length_max=char_length_max,
                          n_trials=n_trials,
                          n_iterations=n_iterations,
                          space=space,
                          xbound=xbound,
                          ybound=ybound,
                          char_length=char_length,
                          angles=angles,
                          bridson_sampling_k=bridson_sampling_k,
                          vis=vis
                          )
            # Make UPXO point objects
            if not self.points:
                if make_point_objects:
                    self.points = [point2d(x=__x, y=__y) for __x, __y in zip(self.locx,
                                                                 self.locy)
                                   ]
            # Update attributes
            self.npoints = len(self.locx)
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True,
                                  )
        self.__state_change = False
        # A NOTE ON NEXT LINE:
        # Keep depth = 2 for the moment until the tree method is fully
        # developed and tested
        # WHEN depth = 2, no cleaning operation happens and duplicates
        # are not removed
        self.clean(depth=0)
        # -------------------------------------------
        self.assign_rid(flag=make_rid, idlength=6)
        # -------------------------------------------
        if make_ckdtree:
            self.maketree(treeType='ckdtree')
        # -------------------------------------------
        if vis:
            self.plot()
        # -------------------------------------------
        self._print_summary(print_summary,
                            make_point_objects=make_point_objects,
                            )
        self.locx = np.array(self.locx)
        self.locy = np.array(self.locy)
    # -----------------------------------------------

    def __repr__(self, recompute_flag=False):
        if recompute_flag:
            if self.lean in ('no', 'low', 'medium'):
                self.recompute_from_mids()
                self.recompute_from_dist(method='points_normal',
                                         tolerance_distance=0.0)
            if self.lean in ('high', 'veryhigh'):
                self.recompute_from_dist(method='coord_normal',
                                         tolerance_distance=0.0)
        str1 = f'Multi-Point ({self.npoints})'
        str2 = f' Centroid:{round(self.centroid[0], 6)},'
        str3 = f' {round(self.centroid[1], 6)}'
        return str1 + str2 + str3

    def __len__(self):
        '''
        Returns [[mp1, mp2, ... mpn], [p1, p1, ..., pn]]
        Where,
              mp1 till mpn: child multi-point objects
              p1: total number of unique points in mp1, and so on
        '''
        return self.npoints

    def __iter__(self,
                 __behaviour='__iterate__over__points__'):
        '''
        RESTRICTION: operates on a single multi-point object
                     if there are child multi-point objects,
                     then user must ensure iter operating on each of them,
                     seperately !
        OPERATION:
                     Iterates through individual points and returns the
                     tuple (m_n.locx[i], m_n.locy[i]), where, n is the
                     n^th multi-point object
        '''
        # "__behaviour" options include:
        # (a) '__iterate__over__coordinates__'
        # (b) '__iterate__over__points__'
        __behaviour = '__iterate__over__points__'
        if __behaviour == '__iterate__over__coordinates__':
            return zip(self.locx, self.locy)
        elif __behaviour == '__iterate__over__points__':
            if self.lean in ('no', 'low', 'medium'):
                return self.points
            elif self.lean in ('high', 'veryhigh'):
                return zip(self.locx, self.locy)

    def __contains__(self, point_object_list):
        '''
        Returns [[mp1, mp2, ... mpn], [[ip1], [ip2], ..., [ipn]]]
        Where,
              mp1 till mpn: child multi-point objects
              [ip1]: list of indices of the
                      queried points in mp1,
                      and so on.
        INPUTS:
            point_object_list: a list of point_objects, which
                               forms the queried points
        '''
        pass

    def pop(self, remove_indices):
        self.points = [i for j, i in enumerate(self.points)
                       if j not in remove_indices]
        self.recompute_basics(npoints_flag=True,
                              locx_locy_flag=True,
                              centroid_flag=True)

    def __pow__(self, power):
        '''
        OPERATION:
            Raises all in self.locx and self.locy by power
        RETURN:
            re-calculated self.locx and self.locy
        '''
        if self.lean in ('no', 'low', 'medium'):
            for _point in self.points:
                _point.x **= power
                _point.y **= power
        if self.lean in ('high', 'veryhigh'):
            self.locx **= power
            self.locy **= power
        self.recompute_basics(npoints_flag=False,
                              locx_locy_flag=True,
                              centroid_flag=True)

    def makegrid(self,
                 method='random',
                 gridding_technique='random',
                 sampling_technique='uniform',
                 randuni_calc='by_points',
                 nrndpnts=10,
                 char_length_mean=0.10,
                 char_length_min=0.05,
                 char_length_max=0.15,
                 n_trials=10,
                 n_iterations=10,
                 space='linear',
                 xbound=[0, 1],
                 ybound=[0, 1],
                 char_length=[0.25, 0.25],
                 angles=[0, 60],
                 bridson_sampling_k=30,
                 vis=True
                 ):
        if method == 'recgrid':
            self.__grid_rec(xbound,
                            ybound,
                            char_length,
                            space,
                            throw=False)
        elif method == 'hexgrid':
            pass
        elif method == 'trigrid1':
            self.__grid_tri(method,
                            angles,
                            xbound,
                            ybound,
                            char_length,
                            space,
                            throw=False)
        elif method == 'trigrid2':
            self.__grid_tri(method,
                            angles,
                            xbound,
                            ybound,
                            char_length,
                            space,
                            False)
        elif method == 'random':
            if gridding_technique == 'random':
                if sampling_technique == 'uniform':
                    # uniformly distributed set of points
                    self.__grid_rnd(sampling_technique=sampling_technique,
                                    randuni_calc=randuni_calc,
                                    xbound=self.xbound,
                                    ybound=self.ybound,
                                    nrndpnts=nrndpnts,
                                    char_length_mean=char_length_mean,
                                    char_length_min=char_length_min,
                                    char_length_max=char_length_max,
                                    n_trials=n_trials,
                                    n_iterations=n_iterations)
                elif sampling_technique == 'normal':
                    # points distributed with a normal distribution
                    pass
                elif sampling_technique == 'custom':
                    # points distributed with a custom distribution
                    pass
                elif sampling_technique == 'dart':
                    self.__grid_rnd(sampling_technique='dart',
                                    xbound=self.xbound,
                                    ybound=self.ybound,
                                    sampling_radius=char_length[0],
                                    sampling_k=bridson_sampling_k)
            elif gridding_technique == 'pds':
                self.__grid_rnd(sampling_technique=sampling_technique,
                                xbound=self.xbound,
                                ybound=self.ybound,
                                sampling_radius=char_length[0],
                                sampling_k=bridson_sampling_k)
        elif method == 'gradient':
            pass
        # make constituent point objects as needed
        if self.lean in ('no', 'low', 'medium'):
            self.points = deque(point2d(x=_x, y=_y)
                                for (_x, _y) in zip(self.locx, self.locy))
        elif self.lean in ('high', 'veryhigh'):
            self.points = None

    def __grid_rec(self,
                   xbound,
                   ybound,
                   char_length,
                   space,
                   throw):
        _x = np.linspace(xbound[0],
                         xbound[1],
                         int(abs(xbound[1]-xbound[0])/char_length[0]))
        _y = np.linspace(ybound[0],
                         ybound[1],
                         int(abs(ybound[1]-ybound[0])/char_length[1]))
        if space == 'linear':
            x, y = np.meshgrid(_x, _y)
        elif space == 'quadratic':
            _x *= np.sign(_x) * _x
            _y *= np.sign(_y) * _y
            # Treat left and right halves seperately
            _x = (_x < 0)*_x/abs(xbound[0]) + (_x >= 0)*_x/abs(xbound[1])
            _y = (_y < 0)*_y/abs(ybound[0]) + (_y >= 0)*_y/abs(ybound[1])
            # Make a grid
            x, y = np.meshgrid(_x, _y)
        elif space == 'logorithmic':
            # https://numpy.org/doc/stable/reference/generated/numpy.logspace.html
            pass
        elif space == 'euler':
            # TO AID ORIENTATION CLASS
            pass
        __xravel, __yravel = np.ravel(x), np.ravel(y)
        # Pass it to attribute
        # self.locx, self.locy = deque(__xravel), deque(__yravel)
        self.locx, self.locy = __xravel, __yravel
        if throw:
            return __xravel, __yravel

    def __grid_tri(self,
                   method,
                   angles,
                   xbound,
                   ybound,
                   char_length,
                   space,
                   throw=False):
        '''
        Make a triangular grid
        '''
        if method == 'trigrid1':
            _rad = np.radians
            # angles = [0, 60]
            # xbound = [-1, 1]
            # ybound = [-1, 1]
            # char_length = [0.25, 0.25]
            # c, s = np.cos(_rad(angles[0])), np.sin(_rad(angles[0]))
            x1 = np.arange(xbound[0], xbound[1], char_length[0])  # Row 1 - x
            x2 = x1 + char_length[0]*np.cos(_rad(angles[1]))  # Row 2 - x
            xi = np.vstack((x1, x2))  # Row 1 and Row 2 - x
            y1 = ybound[0]*np.ones(np.shape(x1))  # Row 1 - y
            y2 = y1 + char_length[1]*np.sin(_rad(angles[1]))  # Row 2 - y
            yi = np.vstack((y1, y2))  # Row 1 and Row 2 - y
            x, y = np.copy(xi), np.copy(yi)
            _num = ybound[1]-ybound[0]
            _den = 0*char_length[1]+2*char_length[1]*np.sin(_rad(angles[1]))
            ncopies = int(_num/_den)
            __k = 2*char_length[1]*np.sin(_rad(angles[1]))
            # __k is just a constant to avoid being calculated in every
            # iteration below
            for count in range(ncopies):
                x = np.vstack((x, xi))
                y = np.vstack((y, yi+(count+1)*__k))
            __xravel, __yravel = np.ravel(x), np.ravel(y)
            # Pass it to attribute
            self.locx, self.locy = __xravel, __yravel
            if throw:
                return __xravel, __yravel
        elif method == 'trigrid2':
            x = np.linspace(xmin, xmax, nx)
            xincr = x[1] - x[0]
            y = np.linspace(ymin, ymax, ny)
            __xy = np.meshgrid(x, y)
            x, y = __xy[0], __xy[1]
            for incr_loc in np.arange(1, np.shape(x)[0], 2):
                x[incr_loc] += xincr*0.5
            x_flat = x.flatten()
            y_flat = y.flatten()

    def __grid_rnd(self,
                   sampling_technique='uniform',
                   xbound=[0, 1],
                   ybound=[0, 1],
                   randuni_calc='by_points',
                   nrndpnts=10,
                   char_length_mean=0.10,
                   char_length_min=0.05,
                   char_length_max=0.15,
                   n_trials=10,
                   n_iterations=10,
                   space='linear',
                   sampling_radius=0.025,
                   sampling_k=30,
                   throw=False):
        if sampling_technique == 'uniform':
            # initiate the numpy>random>uniform method
            npruni = np.random.uniform
            # Calculate the coordinate valeus
            _locx = list(npruni(low=xbound[0],
                                high=xbound[1],
                                size=(nrndpnts,)))
            _locy = list(npruni(low=ybound[0],
                                high=ybound[1],
                                size=(nrndpnts,)))
            if randuni_calc == 'by_points':
                # nothing more to do here.
                # _locx and _locy have already been calculated
                pass
            if randuni_calc == 'by_char_length':
                '''
                ITERATIVELY CALCULATE RANADOM UNIFORM NUMBER DISTRIBUTION
                UNTIL THE MEAN SEPERATION DISTANCE FALLS IN THE RANGE
                DEFINED BY THE 3 CHARACTERISTIC LENGTH VALUES, WHICH ARE,
                     1> char_length_mean - TO BE CONSIDERED LATER
                     2> char_length_min
                     3> char_length_max

                ALGORITHM:
                STEP 1: Calculate minimum, mean and maximum distances
                        from ckdtree of _locx and _locy
                STEP 2: If this
                '''
                # if hasattr(self, 'tree'):
                #     if len(self.tree.data) != self.npoints:
                #         self.maketree(treeType = 'ckdtree')
                # sparse_dist_matrix = m.tree.sparse_distance_matrix(m.tree,
                #                      0.0).toarray()
                npmin = np.min
                npmax = np.max
                npmean = np.mean
                # rng = np.random.default_rng()
                # points1 = rng.random((5, 2))
                # points2 = points1
                point_coords = np.vstack((_locx, _locy)).T
                from scipy.spatial import distance_matrix
                _distances = set(distance_matrix(point_coords,
                                                 point_coords).ravel())
                # INDLCUE EXCEPTION HANDLING HERE FOR VERY VERY VERY SMALL
                # VALUES
                _distances.remove(0.0)
                _distances = np.array(list(_distances), dtype=float)
                _dmin = npmin(_distances)
                _dmean = npmean(_distances)
                _dmax = npmax(_distances)

                _min_flag, _mean_flag, _max_flag = False, False, False
                if self.is_within(input_value=_dmin,
                                  reference_value=char_length_min,
                                  percentage=10,
                                  method='-+',
                                  ):
                    _min_flag = True
                if self.is_within(input_value=_dmean,
                                  reference_value=char_length_mean,
                                  percentage=10,
                                  method='-+',
                                  ):
                    _mean_flag = True
                if self.is_within(input_value=_dmax,
                                  reference_value=char_length_max,
                                  percentage=10,
                                  method='-+',
                                  ):
                    _max_flag = True
                print('==========================')
                print(_min_flag, _mean_flag, _max_flag)
                print('==========================')
                print('Qualification metric values:')
                print(char_length_min, char_length_mean, char_length_max)
                print('Metric value performance value:')
                print(_dmin, _dmean, _dmax)
                print('==========================')
            self.locx, self.locy = _locx, _locy
            if throw:
                return _locx, _locy
        if sampling_technique == 'bridson1':
            # pds: Poisson Disk Sampling
            xstart, ystart = xbound[0], ybound[0]
            # xend, yend = xbound[1], ybound[1]
            from upxo.statops.sampling import bridson_uniform_density as bud
            # bud(width=1, height=1, radius=0.2, k=20)
            _points = bud(width=xbound[1]-xbound[0],
                          height=ybound[1]-ybound[0],
                          radius=sampling_radius,
                          k=sampling_k)
            _locx = [_[0]+xstart for _ in _points]
            _locy = [_[1]+ystart for _ in _points]
            self.locx, self.locy = _locx, _locy
            if throw:
                return _locx, _locy
        if sampling_technique == 'dart':
            from upxo.statops.sampling import dart
            xstart, ystart = xbound[0], ybound[0]
            _points = dart(width=xbound[1]-xbound[0],
                           height=ybound[1]-ybound[0],
                           radius=sampling_radius,
                           k=sampling_k
                           )
            _locx = [_[0]+xstart for _ in _points]
            _locy = [_[1]+ystart for _ in _points]
            self.locx, self.locy = _locx, _locy
            if throw:
                return _locx, _locy

    def is_within(self,
                  input_value: float = None,
                  reference_value: float = None,
                  percentage: float = 10.0,
                  method: str = '-+',
                  ):
        factor = percentage/100
        if method == '-+':
            lowerbound = reference_value*(1-factor)
            upperbound = reference_value*(1+factor)
        elif method == '.+':
            lowerbound = reference_value
            upperbound = reference_value*(1+factor)
        elif method == '-.':
            lowerbound = reference_value*(1-factor)
            upperbound = reference_value
        if input_value >= lowerbound and input_value <= upperbound:
            return True
        else:
            return False

    def assign_rid(self,
                   flag=False,
                   idlength=4):
        '''
        Assign a randomly generated ID.
        Unchanging ID for the instance's life.
        # CAUTION: Do not provide something like a change_rid function
                   down the lane.
        '''
        if flag:
            try:
                print(self.rid)
            except AttributeError:
                import string
                import random
                string = ''.join([string.ascii_uppercase,
                                  string.ascii_lowercase,
                                  string.digits,
                                  string.punctuation])
                self.rid = ''.join(random.choice(string)
                                   for _ in range(idlength))
        else:
            pass

    def build_global_rid(self):
        try:
            _ = self.rid
        except AttributeError:
            try:
                _ = self.points[0].rid
            except AttributeError:
                return [self.rid+p.rid for p in self.points]

    def unpack(self):
        '''
        Behaviour should change after bringing in the capability of this class
        being able to store many multi-point objects. In which case, this
        should return a single list of all points across all its point
        constituent objects.

        NOTE: "point constituent objects" could be either point2d objects or
               mulpoint2d objects

        DEVELOPMENT DECISION @ 12 October 2022:
            It may not be necessary to do this!, given the below restriction
            to be imposed
        RESTRICTION @ 12 October 2022:
            A mupoint2d object cannot and should not hold more than one sub-
            mulpoint2d object.
        '''
        return self.points

    def add_points(self,
                   toadd,
                   recomp=True):
        '''
        OPERATION:
            Appends the point_object to self.points
        RESTRICTIONS:
            Only when self.lean in ('no', 'low', 'medium')
        RETURN:
            Does not return anything
        '''
        if self.lean in ('ignore', 'no', 'low', 'medium'):
            self.points.append(toadd)
            if recomp:
                self.recompute_from_mids()
                self.recompute_from_dist(method='points_normal',
                                         tolerance_distance=0.0)
        elif self.lean in ('high', 'veryhigh'):
            print('Cannot add when lean specification is "{self.lean}"')

    def partition(self,
                  into='xtals_by_state'):
        '''
        Break up this multi-point object into multiple
        multi-point objects based on input conditions

        If "xtals_by_state": Useful for pixellated grain structure: either
            VTGS.2d or MCGS.2d
        '''
        pass

    def neg(self,
            axis='x',
            indices=None):
        '''
        OPERATION:
            Perform -self.locx
        INPUTS:
            indices: list of indices in self.points list to be operated opon
        CALL:
            .negx()
        TRIGGERED OPERATIONS:
            (1) relavent recompute operations
        '''
        if self.lean in ('no', 'low', 'medium'):
            if not indices:
                _iterable = range(self.npoints)
            else:
                _iterable = indices

            if axis == 'x':
                for i in _iterable:
                    self.points[i].negx()
            elif axis == 'y':
                for i in _iterable:
                    self.points[i].negy()
            elif axis == 'xy':
                self.neg(axis='x', indices=indices)
                self.neg(axis='y', indices=indices)
            self.__state_change = True
            self.clean(depth=2)
        elif self.lean in ('high', 'veryhigh'):
            if axis == 'x':
                for i in indices:
                    self.locx[i] *= -1
            elif axis == 'y':
                for i in indices:
                    self.locy[i] *= -1
            elif axis == 'xy':
                self.neg(axis='x', indices=indices)
                self.neg(axis='y', indices=indices)
            self.__state_change = True
            self.clean(depth=2)
        self.recompute_basics(npoints_flag=True,
                              locx_locy_flag=True,
                              centroid_flag=True)
        self.__state_change = False

    def clean(self,
              depth=0):
        '''
        OPERATIONS:
            (1) Remove duplicates
            (2) Removes all attributes except:
                    (a) self.locx, self.locy
        '''
        if depth == 0:
            # Remove duplicates
            if self.lean in ('no', 'low', 'medium'):
                self.recompute_from_mids()

                self.recompute_from_dist(method='points_normal',
                                         tolerance_distance=mulpoint2d.__POINT_TDIST__)
            if self.lean in ('high', 'veryhigh'):
                self.recompute_from_dist(method='coord_normal',
                                         tolerance_distance=mulpoint2d.__POINT_TDIST__)
        elif depth == 1:
            # clean at depth 0
            # Remove self.npoints and self.centroid
            pass
        elif depth == 2:
            # clean at depth 0
            # clean at depth 1
            # Remove self.scalars
            pass
        elif depth == 3:
            # clean at depth 0
            # clean at depth 1
            # clean at depth 2
            # Retain only self.locx and self.locy
            pass

    def maketree(self,
                 treeType='ckdtree',
                 saa=True,
                 throw=False):
        """
        Use tree structure to deal with a very large system of points
        """
        if treeType in ('ckdtree', 'kdtree'):
            # Scipy ckdtree
            from scipy.spatial import cKDTree as ckdt
            # Make the tree data-structure
            _tree = ckdt([[_x, _y] for (_x, _y) in zip(self.locx, self.locy)],
                         copy_data=False,
                         balanced_tree=True)
            if saa:
                self.tree = _tree
            if throw:
                return _tree

    def neigh_pair_count(self,
                         other=None,
                         cor=0.1):
        '''
        Count how many nearby pairs can be formed.
        '''
        try:
            _ = self.tree
        except AttributeError:
            self.maketree(treeType='ckdtree')
        if not other:
            return self.tree.count_neighbors(other=self.tree,
                                             r=cor,
                                             p=2)

    def neigh_points(self,
                     method='coord',
                     coord_list=[[0, 0]],
                     point_object_list=None,
                     cor=[0.1],
                     mpnorm=2,
                     workers=1,
                     return_sorted=False,
                     return_length=False,
                     vis=True
                     ):
        if method in ('coord', 'point'):
            try:
                _ = self.tree
            except AttributeError:
                self.maketree(treeType='ckdtree')
        if method == 'coord':
            indices = [None for _ in range(len(point_object_list))]
            for (i, _cor) in zip(coord_list, cor):
                indices[i] = self.tree.query_ball_point(i,
                                                        r=_cor,
                                                        p=mpnorm,
                                                        workers=workers,
                                                        return_sorted=return_sorted,
                                                        return_length=return_length)
        elif method == 'point':
            indices = [None for _ in range(len(point_object_list))]
            for i in range(len(point_object_list)):
                indices.append[i] = self.tree.query_ball_point([point_object_list[i].x,
                                                                point_object_list[i].y],
                                                               r=cor,
                                                               p=mpnorm,
                                                               workers=workers,
                                                               return_sorted=return_sorted,
                                                               return_length=return_length)
        if vis:
            visopt = ('.', 0.5, 'k', 5, 'maroon', 1.0)
            fig = plt.figure(figsize=(1.6, 1.6),
                             dpi=100)
            left, right = min(self.locx), max(self.locx)
            down, up = min(self.locy), max(self.locy)
            ax = fig.add_axes([0, 0, right-left, up-down])
            ax.plot(self.locx,
                    self.locy,
                    marker=visopt[0],
                    color='none',
                    markeredgewidth=visopt[1],
                    markeredgecolor=visopt[2],
                    markersize=visopt[3],
                    markerfacecolor=visopt[4],
                    alpha=visopt[5])
            for ind in indices:
                for i in ind:
                    ax.plot(self.locx[i],
                            self.locy[i],
                            marker=visopt[0],
                            color='none',
                            markeredgewidth=visopt[1],
                            markeredgecolor=visopt[2],
                            markersize=2*visopt[3],
                            alpha=visopt[5])
            ax.set_xlim([left, right])
            ax.set_ylim([down, up])
        return indices

    def add(self,
            toadd='points',
            coord_list=[],
            point_objects_list=None,
            multi_point_objects_list=None):
        """
        mpobj.add(toadd = 'points', point_objects_list = [p1, p1, p1, p1,
                                                          p1, p1, p1])

        Adding multi_point_objects_list will recursively unpack each
        multi-point object and append each point to self.points

        TRIGGERED OPERATIONS:
            (1) Recalculation methods
        """
        if toadd == 'coord':
            for xy in coord_list:
                self.points.append(point2d(x=xy[0],
                                           y=xy[1]))
        if toadd == 'points':
            for pobj in point_objects_list:
                self.points.append(pobj)
            self.__state_change = True

        if toadd == 'multi_points':
            for mpobj in multi_point_objects_list:
                for pobj in mpobj.points:
                    self.points.append(pobj)
            self.__state_change = True

        if self.__state_change:
            self.recompute_from_mids()
            self.recompute_from_dist(tolerance_distance=0.0)
            self.__state_change = False

    def rem(self,
            method='bypoints',
            torem='points',
            point_objects_list=None,
            multi_point_objects_list=None,
            coord_list=[],
            cut_off_radius=0.0,
            vf=0.05):
        codist = 0.0
        if method == 'bynumber':
            if torem == 'points':
                # numbers corresponds to point numbers in thelist of
                # self.points
                pass
            elif torem == 'multi_points':
                # numbers corresponds to multi-point numbers
                pass
        elif method == 'bypoints':
            if torem == 'multi_points':
                point_objects_list = deque(pobj
                                           for mpobj in multi_point_objects_list
                                           for pobj in mpobj.points)
                # [ <RETURNED_VALUE>  <OUTER_LOOP1>  <INNER_LOOP2>
                #    <INNER_LOOP3> ... <OPTIONAL_IF> ]
        elif method == 'byvf':
            pass
        elif method == 'subsample':
            pass
        # Make a pseudo remove flag array and modify it
        __remove = [pobj.__eq__(_pobj,
                                tdist=codist,
                                use_global_tdist=False)
                    for _pobj in point_objects_list for pobj in self.points]
        # Remove the point
        __remove.reverse()
        for _remove_index in __remove:
            if _remove_index:
                del self.points[_remove_index]
        if True in __remove:
            self.__state_change = True
        if self.__state_change:
            self.recompute_from_mids()
            self.recompute_from_dist(tolerance_distance=0.0)
            self.__state_change = False

    def distance(self, x, y):
        return np.sqrt((np.array(self.locx)-x)**2+(np.array(self.locy)-y)**2)

    def angles(self, x, y):
        return np.arctan2(np.array(self.locy)-y, np.array(self.locx)-x)

    def translate(self, delx, dely):
        '''
        # TODO: INCLUDE OPERATIONS FOR HIGH AND VERY HIGH LEAN SPECIFICATIONS
        '''
        if self.lean in ('no', 'low', 'medium'):
            if delx != 0:
                for pobj in self.points:
                    pobj.x += delx
            if dely != 0:
                for pobj in self.points:
                    pobj.y += dely
            self.recompute_basics(npoints_flag=False,
                                  locx_locy_flag=True,
                                  centroid_flag=True)
        elif self.lean in ('high', 'veryhigh'):
            if delx != 0:
                self.locx += delx
            if dely != 0:
                self.locy += dely
            self.recompute_basics(npoints_flag=False,
                                  locx_locy_flag=False,
                                  centroid_flag=True)

    def moveto(self, newx, newy):
        self.translate(newx - self.centroid[0], newy - self.centroid[1])

    def DG(self,
           deformation='stretch',
           facx=None,
           facy=None,
           shear=None,
           F11=None,
           F12=None,
           F21=None,
           F22=None,
           angle=None):
        '''
        Returns the deformation gradient.
        '''
        if deformation == 'stretch':
            return np.array([[facx, 0.0], [0.0, facy]])
        elif deformation == 'shearx':
            return np.array([[1.0, 0.0], [shear, 1.0]])
        elif deformation == 'sheary':
            return np.array([[1.0, shear], [0.0, 1.0]])
        elif deformation == 'pureshear':
            return np.array([[1.0, shear], [shear, 1.0]])
        elif deformation == 'rotation':
            θ = np.deg2rad(angle)
            c, s = np.cos(θ), np.sin(θ)
            return np.array([[c, -s], [s, c]])  # Nothing but the rot. matrix
        elif deformation == 'general':
            return np.array([[F11, F12], [F21, F22]])

    def rotate1(self, x, y, θ):
        '''
        x, y: coordinates of the rotation centre
        ang: angle of rotation.

        NOTE: Rotation will be counter-clockwise
        _______________________________________________________
        x1 = (x0 – xc)cos(θ) – (y0 – yc)sin(θ) + xc
        y1 = (x0 – xc)sin(θ) + (y0 – yc)cos(θ) + yc
        where:
            (x0, y0)	= Point to be rotated
            (xc, yc)	= Coordinates of center of rotation
            θ	        = Angle of rotation (positive counterclockwise)
            (x1, y1)	= Coordinates of point after rotation
        # Credit for above:
            https://danceswithcode.net/engineeringnotes/rotations_in_2d/rotations_in_2d.html
        _______________________________________________________
        Lets make this faster:
            In matrix form, this would become:
                |x1|  =  |x0-xc| x |cos(theta) -sin(theta)| + |xc|
                |y1|     |y0-yc|   |sin(theta)  cos(theta)|   |yc|
            Above can be done easily through numpy
        '''
        # Form the rotation matrix
        # θ = np.deg2rad(θ)
        # c, s = np.cos(θ), np.sin(θ)
        # R = np.array([[c, -s], [s, c]])
        R = self.DG(deformation='rotation',
                    angle=-θ)
        xyc = np.array([x, y])
        if self.lean in ('no', 'low', 'medium'):
            for i, (x0, y0) in enumerate(zip(self.locx, self.locy),
                                         start=0):
                __ = np.matmul((np.array([x0, y0]) - xyc), R) + xyc
                self.points[i].x, self.points[i].y = __
            self.recompute_basics(npoints_flag=False,
                                  locx_locy_flag=True,
                                  centroid_flag=True)
        if self.lean in ('high', 'veryhigh'):
            for i, (x0, y0) in enumerate(zip(self.locx, self.locy),
                                         start=0):
                __ = np.matmul((np.array([x0, y0]) - xyc), R) + xyc
                self.locx[i], self.locy[i] = __
            self.recompute_basics(npoints_flag=False,
                                  locx_locy_flag=False,
                                  centroid_flag=True)

    def explode(self,
                x,
                y,
                factorx,
                factory):
        '''
        1. Compute distances of all point objects to (x,y)
        2. delx and dely will be stretch_factorx and stretch_factory times the
        NOTES:
            1. Any negative factors will be made positive upon execution.
        '''
        d, θ = self.distance(x, y), self.angles(x, y)
        dcosang, dsinang = d*np.cos(θ)*abs(factorx), d*np.sin(θ)*abs(factory)
        for count in range(self.npoints):
            self.points[count].x = dcosang[count]
            self.points[count].y = dsinang[count]
        if factorx == factory:
            centroid_flag = False
        self.recompute_basics(npoints_flag=False,
                              locx_locy_flag=True,
                              centroid_flag=centroid_flag)

    def shear(self,
              deformation='shearx',
              shear=None):
        '''
        shear deform the multi-point object
        OPTIONS:
            1. shearx: shear along x - axis. A +ve shear tilts left vertical
            of fundamental rectangle right, about bottom left corner
            2. sheary: shear along y - axis. A +ve shear tilts bottom
            horizontal of fundamental rectangle up, about bottom left corner
            3. pureshear: shearx + sheary
        '''
        F = self.DG(deformation=deformation, shear=shear)
        if self.lean in ('no', 'low', 'medium'):
            for i, (x0, y0) in enumerate(zip(self.locx, self.locy), start=0):
                __ = np.matmul(np.array([x0, y0]), F)
                self.points[i].x, self.points[i].y = __
            self.recompute_basics(npoints_flag=False,
                                  locx_locy_flag=True,
                                  centroid_flag=True)
        elif self.lean in ('high', 'veryhigh'):
            for i, (x0, y0) in enumerate(zip(self.locx, self.locy), start=0):
                self.locx[i], self.locy[i] = np.matmul(np.array([x0, y0]), F)
            self.recompute_basics(npoints_flag=False,
                                  locx_locy_flag=False,
                                  centroid_flag=True)

    def mirror(self,
               a: float = 1.0,
               b: float = 0.0,
               c: float = 0.0,
               keep_original: bool = True
               ):
        '''
        a, b, c: constants in the equation of line: ax + by + c = 0
        K = −2(ax0+by0+c) / sqrt(a^2 + b^2)
        x1 = aK + x0
        y1 = aK + y0

        GENERAL NOTES TO THE USER:
            If you are looking at mirroring about x-axis,
                then choose a = 0, b = 1, c = 0
            If you are looking at mirroring about line parallel to x-axis,
                at a distance of 2 units towards +ve y-axis,
                then choose a = 0, b = 1, c = 2
        '''
        x0 = 1
        y0 = 1
        a = 0
        b = 1
        c = 2
        K = - 2*(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**22)
        a*K + x0
        b*K + y0

    def merge(self,
              retain='first',
              ):
        '''
        #TODO
        Merge two or more points based on proximity or overlap condition
        '''
        pass

    def noise(self,
              perturb_flag=False,
              perturb_type='local_uniform',
              depth=None,
              ground=None,
              height=None,
              perturb_mag=None):
        if perturb_type == 'dgh':
            # Depth, Ground and Height are lengths
            depth, ground, height = -abs(depth), abs(ground), abs(height)
            noise_x = ground + np.random.uniform(depth, height,
                                                 (self.npoints,))
            noise_y = ground + np.random.uniform(depth, height,
                                                 (self.npoints,))
            if self.lean in ('no', 'low', 'medium'):
                for count in range(self.npoints):
                    self.points[count].x += noise_x[count]
                    self.points[count].y += noise_y[count]
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=True,
                                      centroid_flag=True)
            elif self.lean in ('high', 'veryhigh'):
                # for count in range(self.npoints):
                # self.locx[count] += noise_x[count]
                # self.locy[count] += noise_y[count]
                self.locx += noise_x
                self.locy += noise_y
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=False,
                                      centroid_flag=True)
        if perturb_type == 'local_uniform':
            # TODO: CONSIDER NEGATIVE POINTS AS WELL.
            xpert = perturb_mag[0]*np.random.random(self.npoints)
            ypert = perturb_mag[1]*np.random.random(self.npoints)
            try:
                _ = self.xpert
            except AttributeError:
                self.xpert, self.ypert = xpert, ypert
            else:
                self.xpert += xpert
                self.ypert += ypert
            self.locx = np.array([_x+__x for (_x, __x) in zip(self.locx,
                                                              self.xpert)])
            self.locy = np.array([_y+__y for (_y, __y) in zip(self.locy,
                                                              self.ypert)])
            # Above two lines of code will replace the following two commented
            # lines. This is because, deque does not support element wise
            # addition.
            # self.locx += self.xpert
            # self.locy += self.ypert
            if self.lean in ('no', 'low', 'medium'):
                for count in range(self.npoints):
                    self.points[count].x = self.locx[count]
                    self.points[count].y = self.locy[count]
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=False,
                                      centroid_flag=True)
            elif self.lean in ('high', 'veryhigh'):
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=False,
                                      centroid_flag=True)
        if perturb_type == 'local_normal':
            self.xpert = perturb_mag[0][0]*np.random.normal(perturb_mag[0][1],
                                                            perturb_mag[0][2],
                                                            self.npoints
                                                            )
            self.ypert = perturb_mag[1][0]*np.random.normal(perturb_mag[1][1],
                                                            perturb_mag[1][2],
                                                            self.npoints)
            self.locx = np.array([_x+__x for (_x, __x) in zip(self.locx,
                                                              self.xpert)])
            self.locy = np.array([_y+__y for (_y, __y) in zip(self.locy,
                                                              self.ypert)])
            # Above two lines of code will replace the following two commented lines
            # This is because, deque does not support element wise addition.
            # self.locx += self.xpert
            # self.locy += self.ypert
            if self.lean in ('no', 'low', 'medium'):
                for count in range(self.npoints):
                    self.points[count].x = self.locx[count]
                    self.points[count].y = self.locy[count]
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=False,
                                      centroid_flag=True)
            elif self.lean in ('high', 'veryhigh'):
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=False,
                                      centroid_flag=True)

    def relax(self,
              iterations: int = 2,
              bounds=None,
              options: str = 'Qbb Qc Qx'):
        '''
        # TODO: MUST CLEAN THIS UP AND MAKE IT UPXO COMPATIBLE
        Credit: see below link
        https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points.smoothLloyd2D
        '''
        # def smoothLloyd2D(self, iterations=2, bounds=None,
        #                   options='Qbb Qc Qx'):
        """Lloyd relaxation of a 2D pointcloud."""
        # Credits: https://hatarilabs.com/ih-en/
        # tutorial-to-create-a-geospatial-voronoi-sh-mesh-with-python-scipy-and-geopandas
        from scipy.spatial import Voronoi as scipy_voronoi

        def _constrain_points(points):
            '''Update any points that have drifted beyond the boundaries of
            this space'''
            if bounds is not None:
                for point in points:
                    if point[0] < bounds[0]:
                        point[0] = bounds[0]
                    if point[0] > bounds[1]:
                        point[0] = bounds[1]
                    if point[1] < bounds[2]:
                        point[1] = bounds[2]
                    if point[1] > bounds[3]:
                        point[1] = bounds[3]
            return points

        def _find_centroid(vertices):
            '''The equation for the method used here to find the centroid of
            a 2D polygon is given here:
                https://en.wikipedia.org/wiki/Centroid#Of_a_polygon'''
            area = 0
            centroid_x = 0
            centroid_y = 0
            for i in range(len(vertices)-1):
                step = (vertices[i, 0]*vertices[i+1, 1]) - \
                    (vertices[i+1, 0]*vertices[i, 1])
                centroid_x += (vertices[i, 0] + vertices[i+1, 0])*step
                centroid_y += (vertices[i, 1] + vertices[i+1, 1])*step
                area += step
            if area:
                centroid_x = (1.0/(3.0*area))*centroid_x
                centroid_y = (1.0/(3.0*area))*centroid_y
            # prevent centroids from escaping bounding box
            return _constrain_points([[centroid_x, centroid_y]])[0]

        def _relax(voronoi):
            '''Moves each point to the centroid of its cell in the voronoi
            map to "relax" the points (i.e. jitter the points so as
            to spread them out within the space).'''
            centroids = []
            for idx in voronoi.point_region:
                # the region is a series of indices into voronoi.vertices
                # remove point at infinity, designated by index -1
                region = [i for i in voronoi.regions[idx] if i != -1]
                # enclose the polygon
                region = region + [region[0]]
                verts = voronoi.vertices[region]
                # find the centroid of those vertices
                centroids.append(_find_centroid(verts))
            return _constrain_points(centroids)
        if bounds is None:
            bounds = self.bounds()
        pts = self.points()[:, (0, 1)]
        for i in range(iterations):
            vor = scipy_voronoi(pts, qhull_options=options)
            _constrain_points(vor.vertices)
            pts = _relax(vor)
        # return Points(pts, c='k')
        return pts

    def recompute_from_mids(self):
        """
        mid: ID of the object's memory address'
        NOTE TO DEVELOPER: LEAVE THIS AS IS
        """
        if self.lean in ('no', 'low', 'medium'):
            mids = [id(pobj) for pobj in self.points]  # object memory IDs
            if len(mids) > len(list(set(mids))):
                # make the mids unique and unsorted
                mids = list(dict.fromkeys(mids))
                self.unique_points_mids(mids)
                self.__state_change = True
            if self.__state_change:
                self.recompute_basics(npoints_flag=True,
                                      locx_locy_flag=True,
                                      centroid_flag=True)
            # CAUTION: DO NOT CHANGE EXECUTION ORDER
            self.__state_change = False
        elif self.lean in ('high', 'veryhigh'):
            pass

    def recompute_from_dist(self,
                            method='points_normal',
                            tolerance_distance=0.0):
        '''
        DESCRIPTION: recompute() using point2d.__eq__(query_point_object)
        # WHEN TO USE:
            Use when system size is small.
            Exact size threshild YET to be determined
        '''
        self.__state_change = self.unique_points_dist(method=method,
                                                      tolerance_distance=tolerance_distance)
        if self.__state_change:
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True)
        # CAUTION: DO NOT CHANGE EXECUTION ORDER
        self.__state_change = False

    def unique_points_mids(self, mids):
        '''
        In this case, update op happens by comparing memory ID of point objects
        NOTE: For the forseable future, this method is to be used until its use
        been validated against.
        '''
        # mids = [id(pobj) for pobj in self.points] # object memory IDs
        # if len(mids) > len(list(set(mids))):
        # Choose numpy if number of points are more than 'npoints = 1000'
        if self.npoints > 1000:
            mids = np.array(mids)
        # initate _points as an empty list
        __points = []
        # Append it with unique points
        for key in mids:
            for pobj in self.points:
                if id(pobj) == key:
                    break
            __points.append(pobj)
        # Replace self.points with the new list
        self.points = __points

    def unique_points_dist(self,
                           method='points_normal',
                           tolerance_distance=0.0):
        '''
        Identify unique points as the first of the list of points which
        have a zero-distance in case of repeating points
            NOTE: zero-distance is decided by the tolerance_distance set by
                  the user

        # TODO: Enable multi-processing and speed up the process

        NOTE: LEAVE THIS METHOD AS IT IS
        '''
        if method == 'points_normal':
            __points = []
            repeated = [False for _ in self.points]
            count1 = 0
            for pobj_1 in self.points:
                if not repeated[count1]:
                    count2 = 0
                    for pobj_2 in self.points:
                        if count1 != count2:
                            if self.points[count1].__eq__(pobj_2,
                                                          tdist=tolerance_distance,
                                                          use_self_tdist=False):
                                repeated[count2] = True
                        count2 += 1
                count1 += 1
            if True in repeated:
                for (_repeat, _pobj) in zip(repeated, self.points):
                    if not _repeat:
                        __points.append(_pobj)
                self.points = __points
                return True, None, None
            else:
                return False, None, None
        elif method == 'coord_normal':
            __locx, __locy = [], []
            repeated = [False for _ in self.locx]
            for count1, (_x, _y) in enumerate(zip(self.locx, self.locy),
                                              start=0):
                if not repeated[count1]:
                    for count2, (__x, __y) in enumerate(zip(self.locx,
                                                            self.locy),
                                                        start=0):
                        if count1 != count2:
                            distdiff = np.sqrt((_x - __x)**2 + (_y - __y)**2)
                            if distdiff <= tolerance_distance:
                                repeated[count2] = True
            if True in repeated:
                for (_repeat, _x, _y) in zip(repeated, self.locx, self.locy):
                    if not _repeat:
                        __locx.append(_x)
                        __locy.append(_y)
                from copy import deepcopy as dcopy
                self.locx = dcopy(__locx)
                self.locy = dcopy(__locy)
                '''TODO: After testing, replace above and this line by the
                following line.'''
                # self.locx, self.locy = dcopy(__locx), dcopy(__locy)
                return True, __locx, __locy
            else:
                return False, self.locx, self.locy
        elif method == 'points_fast':
            '''TODO: patch development of the tree method once complete'''
            pass

    def recompute_basics(self,
                         npoints_flag=True,
                         locx_locy_flag=True,
                         centroid_flag=True):
        """
        On a need to basis, recompute the number of points, x and y coordinate
        lists (deques) and the centroid
        """
        if npoints_flag:
            self.recompute_npoints()
        if locx_locy_flag:
            self.recompute_locx_locy()
        if centroid_flag:
            self.recompute_centroid()

    def recompute_npoints(self):
        """
        Recompute the number of points in this collection
        """
        if self.lean in ('ignore', 'no', 'low', 'medium'):
            try:
                self.npoints = len(self.points)
            except AttributeError:
                self.npoints = len(self.locx)
        if self.lean in ('high', 'veryhigh'):
            self.npoints = len(self.locx)

    def recompute_locx_locy(self):
        """
        Recompute the coordinate location lists (deques)
        """
        if self.lean in ('ignore', 'no', 'low', 'medium'):
            # TODO: Combine the following two lines into a single
            # list comprehension
            self.locx = np.array([pobj.x for pobj in self.points])
            self.locy = np.array([pobj.y for pobj in self.points])
        if self.lean in ('high', 'veryhigh'):
            pass

    def recompute_centroid(self, accuracy='exact'):
        """
        Recompute the centroid of the point collection

        TODO: In the case of multi-point collection, this is to return
        a list of centroids, in the order of the multi-point list
        """
        _n = self.npoints
        _sum = np.sum
        self.centroid = [_sum(self.locx)/_n, _sum(self.locy)/_n]

    def qh(self,
           package='scipy',
           save_as_attribute=True,
           throw=True):
        """
        Compute the convex hull using various known packages
        TODO: THIS REQUIRES IMMEDIATE FURTHER DEVELOPMENT
        """
        if package == 'scipy':
            from scipy.spatial import ConvexHull
            # TODO:
            _points = [[_x, _y] for (_x, _y) in zip(self.locx, self.locy)]
            qh = ConvexHull(_points)
            if save_as_attribute:
                self.qh = qh
            if throw:
                return qh

    def qh_find_area(self):
        """
        # TODO: THIS REQUIRES IMMEDIATE FURTHER DEVELOPMENT
        """
        pass

    def qh_find_perimeter(self):
        """
        # TODO: THIS REQUIRES IMMEDIATE FURTHER DEVELOPMENT
        """
        pass

    def hole(self):
        """
        Make a hole, an empty region in a region of distributed point objects
        """
        pass

    def mapsf_area(self):
        """
        Attach xtal area to the xtal point. The xtal point should be the xtal
        representative point.
        """
        pass

    def mapsf_ori(self):
        """
        Attach orientation distribution to the point distribution
        """
        pass

    def mapsf_gnd(self):
        """
        Attach GND field to the point distribution
        """
        pass

    def _print_summary(self,
                       print_summary=True,
                       make_point_objects=None,
                       make_ckdtree=False,
                       sep_n=10,
                       sep_string='-X'):
        if print_summary:
            from colorama import init as colorama_init
            from colorama import Fore
            # from colorama import Back
            from colorama import Style
            colorama_init()
            sep_n, sep_string = sep_n, '-X-'
            if self.mulpoint_type == 'seed':
                _ = '(default)'
            else:
                _ = ''
            print(''.join([Fore.RED, sep_n*sep_string, '\n',
                           Fore.WHITE, f'Summary of: {self}', '\n\n',
                           Fore.WHITE, f'Multi-point type: {self.mulpoint_type} {_}', '\n',
                           Fore.WHITE, f'No. of points = {self.npoints}', '\n',
                           Fore.WHITE, f'Point objects exit: {make_point_objects}','\n\n',
                           Fore.WHITE, f'Centroid(x,y): {self.centroid[0], self.centroid[1]}',
                           ]
                          )
                  )

            # if make_ckdtree:
            #     string = sep_string.join([' ', '\n'])
            # string = string.join([Fore.RED, sep_n*sep_string, '\n'])
            print(''.join([Fore.RED, sep_n*sep_string, '\n']))
            Style.RESET_ALL

    def plot(self, figsize=(1.6, 1.6), dpi=100,
             visopt=('.', 0.5, 'k', 10, 'maroon', 1.0)):
        """
        self.vizopt = ('o', 0.5, 'k', 10, 'w', 0.8)
        IN EXACT ORDER:
            o   : circle marker style
            0.5 : Marker edge line width
            k   : Marker edge color
            w   : marker face colour white
            10  : marker size
            0.8 : marker face colour alpha
        """
        visopt = visopt
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.gca()

        ax.plot(self.locx, self.locy,
                marker=visopt[0],
                color='none',
                markeredgewidth=visopt[1],
                markeredgecolor=visopt[2],
                markersize=visopt[3],
                markerfacecolor=visopt[4],
                alpha=visopt[5])

        ax.set_xlim([min(self.locx), max(self.locx)])
        ax.set_ylim([min(self.locy), max(self.locy)])
        plt.show()
        return ax

    def perthist(self):
        fig = plt.figure(figsize=(1.6, 1.6),
                         dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.hist(self.xpert,
                bins=25,
                label='x-pert',
                edgecolor='k',
                facecolor='teal',
                alpha=0.5)
        ax.hist(self.ypert,
                bins=25,
                label='y-pert',
                edgecolor='k',
                facecolor='orange',
                alpha=0.5)
        ax.legend(loc='upper left',
                  shadow=False,
                  fontsize='small',
                  edgecolor='k')
