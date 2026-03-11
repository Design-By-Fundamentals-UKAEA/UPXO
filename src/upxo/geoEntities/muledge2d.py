from upxo._sup import dataTypeHandlers as dth
from upxo.geoEntities.mulpoint2d import MPoint2d as mulpoint2d
# from mulpoint2d import mulpoint2d
import point2d
from edge2d import edge2d
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from upxo._sup import gops

np.seterr(divide='ignore')
# Script information for the file.
__name__ = "UPXO-muledge"
__authors__ = ["Vaasu Anandatheertha"]
__lead_developer__ = ["Vaasu Anandatheertha"]
__emails__ = ["vaasu.anandatheertha@ukaea.uk", ]
__version__ = ["0.1: from.030522.git.no", ]

__license__ = "GPL v3"


class muledge2d():
    """
    UPXO core class.
    Represents collection of connected edges in 2D Cartesian space.
    """
    ROUND_ZERO_DEC_PLACE = 10
    EPS = 0.000000000001
    __slots__ = ('dim',  # Dimensionality
                 'ppairs',  # list of UPXO point objects
                 'cpairs',  # Coordinates of the constituent points
                 'clist',  # List of coordinates of points
                 'pmids',  # UPXO point2d memory ID list
                 'pmid_pairs',  # List of UPXO point2d memory ID pairs
                 'pindices',  # pindices of the coords which make the muledge2d
                 'eindices',  # pindices of the edges which make the muledge2d
                 'points',  # Constituent UPXO point objects
                 'mpoint',  # Multi-point with all points
                 'empoints',  # Edge wise multi-points
                 'edges',  # Constituent UPXO edge2d objects
                 'ordered',  # Specifies ordering of coordinates
                 'closed',  # Specifies whether already a ring or not
                 'psense',  # Specifies whether CW or CCW point ordering
                 'esense',  # Specifies whether CW or CCW edge ordering
                 'plean',  # Lean of UPXO point objects
                 'mplean',  # Lean of UPXO  Multi-POint objects
                 'elean',  # Lean of UPXO Edge objects
                 'lean',  # Lean of the self object
                 '__rings__'  # ring objects
                 )

    def __init__(self,
                 method='cpairs_list',
                 ordered=True,
                 closed=False,
                 psense='ccw',
                 esense='ccw',
                 clist=[],
                 cpairs_list=[],
                 points=[],
                 edges=None,
                 edge_base='upxo',
                 pindices=[],
                 eindices=[],
                 make_mp=True,
                 make_emp=True,
                 lean='ignore', plean='ignore', mplean='ignore',
                 elean='ignore', melean='ignore',
                 ):
        '''
        method : str
            Enables branching.
            Option 1: 'cpairs_list': @cpairs_list: raw co-ordinate data
            Option 2: 'edges': @edges @edge_base: edge objects

        cpairs_list : list/tuple/numpy array/deque
            [[x0, y0], [x1, y1], ...]

        edges : list/tuple

        edge_base:
            Case 1: 'upxo': UPXO edge object
            Case 2: 'shapely'
            Case 3: 'vtk'
            Case 4: 'gmsh'

        pindices : list/tuple/numpy array/deque
            Point-pairing pindices needed to make edges2d objects
        '''
        # ---------------------------------
        self.dim = 2  # Dimensionality of the multi-edge object
        # ---------------------------------
        if lean in ('ignore'):
            self.ordered = ordered
            self.closed = closed
            self.psense = psense
            self.esense = esense
            self.lean = lean
            self.plean = plean
            self.mplean = mplean
            self.elean = elean
            self.__rings__ = []
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        if method == 'clist' and self.ordered:
            '''
            clist = [[0, 0], [1, 0], [0, 0], [1, 1.01], [2, 2]]
            from muledge2d import muledge2d
            me = muledge2d(method='clist',
                           ordered=True,
                           closed=False,
                           clist=clist,
                           make_mp=True, make_emp=True,
                           lean='ignore', plean='ignore', mplean='ignore',
                           elean='ignore', melean='ignore'
                           )
            '''
            # Make clist unique
            _mp_ = mulpoint2d(method='xy_pair_list', coordxy=clist)
            clist = deque([])
            for _x, _y in zip(_mp_.locx, _mp_.locy):
                clist.append([_x, _y])
            # ---------------------------
            self.clist = clist
            npoints = len(clist)
            # ---------------------------
            # Build pindices
            self.pindices = [[i, i+1] for i in range(len(self.clist)-1)]
            # ---------------------------
            # Build UPXO points
            self.points = deque([point2d(clist[i][0],
                                         clist[i][1],
                                         lean=plean)
                                 for i in range(npoints)])
            # ---------------------------
            # Build pmids
            self.pmids = deque([id(p) for p in self.points])
            # ---------------------------
            # Build pmid_pairs
            self.pmid_pairs = [[self.pmids[i[0]], self.pmids[i[1]]]
                               for i in self.pindices]
            # ---------------------------
            # AVOID USING PPAIRS AT ALL COST.
            # TO BE DEPRACATED
            self.ppairs = [[self.points[i[0]], self.points[i[1]]]
                           for i in self.pindices]
            # ---------------------------
            # Build edges and eindices
            self.edges = deque([edge2d(method='up2d',
                                       pnta=self.points[i[0]],
                                       pntb=self.points[i[1]],
                                       edge_lean=self.elean,
                                       ) for i in self.pindices])
            self.eindices = deque([i for i in range(npoints)])
            # ---------------------------
            # Build multi-point object
            self.mpoint = mulpoint2d(method='up2d_list',
                                     point_objects=self.points,
                                     lean=self.mplean)
            # ---------------------------
            # Build edge wise multi-point
            self.empoints = [mulpoint2d(method='up2d_list',
                                        point_objects=[self.points[i[0]],
                                                       self.points[i[1]]],
                                        lean=self.mplean)
                             for i in self.pindices]
            # ---------------------------
            if make_mp:
                self.dbbuild_mpoint()
            # ---------------------------
            if make_emp:
                self.dbbuild_empoint()
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        if method == 'clist' and not self.ordered:
            '''
            # EXAMPLE-1: specifying "pindices"
            clist = [[0, 0], [1, 0], [1, 1.01], [2, 2]]
            from muledge2d import muledge2d
            pindices = [ [0, 3], [0, 1], [2, 3], [1, 2]]
            me = muledge2d(method='clist',
                           ordered=False,
                           pindices = pindices,
                           closed=False,
                           clist=clist,
                           make_mp=True, make_emp=True,
                           lean='ignore', plean='ignore', mplean='ignore',
                           elean='ignore', melean='ignore'
                           )
            >> me.points
            >> Out[409]: deque([upxo.p2d(0.0, 0.0),
                                upxo.p2d(1.0, 0.0),
                                upxo.p2d(1.0, 1.01),
                                upxo.p2d(2.0, 2.0)])
            me.edges
            >> Out[410]: [upxo.e2d[(0.0, 0.0)⚯(2.0, 2.0)⊢⊣2.8284],
                          upxo.e2d[(0.0, 0.0)⚯(1.0, 0.0)⊢⊣1.0],
                          upxo.e2d[(1.0, 1.01)⚯(2.0, 2.0)⊢⊣1.4072],
                          upxo.e2d[(1.0, 0.0)⚯(1.0, 1.01)⊢⊣1.01]]
            #---------------------------------
            # EXAMPLE-2: specifying "pindices" and "eindices"
            clist = [ [0, 0], [1, 0], [1, 1.01], [2, 2] ]
            from muledge2d import muledge2d
            pindices = [ [0, 3], [0, 1], [2, 3], [1, 2] ]
            eindices = [2, 0, 1, 3]
            me = muledge2d(method='clist',
                           ordered=False,
                           pindices = pindices,
                           eindices = eindices,
                           closed=False,
                           clist=clist,
                           make_mp=True, make_emp=True,
                           lean='ignore', plean='ignore', mplean='ignore',
                           elean='ignore', melean='ignore'
                           )
            >> me.points
            >> Out[414]: deque([upxo.p2d(0.0, 0.0),
                                upxo.p2d(1.0, 0.0),
                                upxo.p2d(1.0, 1.01),
                                upxo.p2d(2.0, 2.0)])
            me.edges
            >> Out[415]: [upxo.e2d[(1.0, 1.01)⚯(2.0, 2.0)⊢⊣1.4072],
                          upxo.e2d[(0.0, 0.0)⚯(2.0, 2.0)⊢⊣2.8284],
                          upxo.e2d[(0.0, 0.0)⚯(1.0, 0.0)⊢⊣1.0],
                          upxo.e2d[(1.0, 0.0)⚯(1.0, 1.01)⊢⊣1.01]]
            #---------------------------------
            # EXAMPLE-3: This will give an error as data is incosistant
            clist = [[0, 0], [1, 0], [0, 0], [1, 1.01], [2, 2]]
            from muledge2d import muledge2d
            pindices = [ [0, 3], [0, 1], [2, 3], [1, 2]]
            me = muledge2d(method='clist',
                           ordered=False,
                           pindices = pindices,
                           closed=False,
                           clist=clist,
                           make_mp=True, make_emp=True,
                           lean='ignore', plean='ignore', mplean='ignore',
                           elean='ignore', melean='ignore'
                           )
            >> Please enter valid clist data
            #---------------------------------
            '''
            _npoints_ = len(clist)
            # Make clist unique
            _mp_ = mulpoint2d(method='xy_pair_list', coordxy=clist)
            if _mp_.npoints == _npoints_:
                clist = deque([])
                for _x, _y in zip(_mp_.locx, _mp_.locy):
                    clist.append([_x, _y])
                proceed = True
            else:
                print('Please enter valid clist data')
                proceed = False
            _pindices_flat_ = []
            for pind in pindices:
                _pindices_flat_.append(pind[0])
                _pindices_flat_.append(pind[1])
            _max_ = np.array(_pindices_flat_).max()+1
            if proceed:
                if _max_ == _npoints_:
                    proceed = True
                else:
                    print('Please enter valid pindices data')
                    proceed = False
            if proceed:
                # ---------------------------
                npoints = len(clist)
                self.clist = clist
                # ---------------------------
                # Build pindices
                self.pindices = pindices
                # ---------------------------
                # Build UPXO points
                self.points = deque([point2d(x=coord[0],
                                             y=coord[1],
                                             lean=plean)
                                     for coord in clist]
                                    )
                # ---------------------------
                # Build pmids
                self.pmids = deque([id(p) for p in self.points])
                # ---------------------------
                # Build pmid_pairs
                self.pmid_pairs = [[self.pmids[i[0]], self.pmids[i[1]]]
                                   for i in self.pindices]
                # ---------------------------
                # Build edges
                edges = [edge2d(method='up2d',
                                pnta=self.points[i[0]],
                                pntb=self.points[i[1]],
                                edge_lean=self.elean,
                                ) for i in self.pindices]
                if eindices:
                    self.edges = deque([edges[i] for i in eindices])
                    self.eindices = eindices
                else:
                    self.edges = deque(edges)
                    self.eindices = list(range(npoints))
                # ---------------------------
                # Build multi-point object
                self.mpoint = mulpoint2d(method='up2d_list',
                                         point_objects=self.points,
                                         lean=self.mplean)
                # ---------------------------
                # Build edge wise multi-point
                self.empoints = [mulpoint2d(method='up2d_list',
                                            point_objects=[self.points[i[0]],
                                                           self.points[i[1]]],
                                            lean=self.mplean)
                                 for i in self.pindices]
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        if method == 'cpairs_list' and self.ordered:
            '''
            ***USER MUST ENSURE POINTS ARE IN RIGHT ORDER
            BOTH IN THE COORDINATE PAIRS AND ACROSS THE COORDINATE
            PAIRS LIST***

            What does 'ordered' data format mean in this case?
                : Coordinate pair arrangement is ordered for:
                    (1) Sense of the edge (direction)
                    (2) Adjacent edges have relavant data which are also
                    adjcent (in cpairs_list).

            EXAMPLE
            cpairs_list = [[[1, 1], [-1, 1]],
                           [[-1, 1], [-1, -1]],
                           [[-1, -1], [1, -1]],
                           ]
            from muledge2d import muledge2d
            me = muledge2d(method='cpairs_list',
                           ordered=True,
                           closed=False,
                           cpairs_list=cpairs_list,
                           make_mp=True, make_emp=True,
                           lean='ignore', plean='ignore', mplean='ignore',
                           elean='ignore', melean='ignore'
                           )
            me.plotme()
            me.plotmp()
            '''
            self.clist = deque([])
            self.cpairs = np.array(cpairs_list)
            self.pindices = deque([])
            self.ppairs = []
            if not closed:
                pnta = point2d(self.cpairs[0][0][0],
                               self.cpairs[0][0][1],
                               lean=plean
                               )
                pntb = point2d(self.cpairs[0][1][0],
                               self.cpairs[0][1][1],
                               lean=plean
                               )
                self.ppairs.append([pnta, pntb])
                self.pindices.append([0, 1])
                self.clist.extend([list(self.cpairs[0][0]),
                                   list(self.cpairs[0][1])])
                for i, _ in enumerate(self.cpairs[1:], start=1):
                    pnta = self.ppairs[i-1][1]
                    pntb = point2d(self.cpairs[i][1][0],
                                   self.cpairs[i][1][1],
                                   lean=plean
                                   )
                    self.ppairs.append([pnta, pntb])
                    self.pindices.append([i, i+1])
                    self.clist.append(list(self.cpairs[i][1]))
            else:
                # If the mul-edge is indeed closed
                pass
            # -------------------------------------------
            # Build the DB of all poinnts in this multi-edge object
            self.dbbuild_points(method='ppairs')
            # -------------------------------------------
            self.eindices = list(range(len(self.ppairs)))
            # ---------------------------
            if make_mp:
                self.dbbuild_mpoint()
            # ---------------------------
            if make_emp:
                self.dbbuild_empoint()
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        if method == 'cpairs_list' and not self.ordered:
            '''
            ***USER MUST ENSURE POINTS ARE IN RIGHT ORDER
            BOTH IN THE COORDINATE PAIRS AND ACROSS THE COORDINATE
            PAIRS LIST***

            Sense will be assumed to be from first to the second point
            i.e. from pnta to pntb

            Ordering will done as per pindices

            EXAMPLE
            cpairs_list = [[[1, 1], [-1, 1]],
                           [[-1, -1], [1, -1]],
                           [[-1, 1], [-1, -1]],
                           ]
            eindices = [0, 2, 1]
            # NOTE: Notice that 2nd 3rd edges are swapped from the
            # previous example (under branch for ordered=True)

            from muledge2d import muledge2d
            me = muledge2d(method='cpairs_list',
                           ordered=False, closed=False,
                           psense='ccw', esense='ccw',
                           cpairs_list=cpairs_list,
                           eindices=eindices,
                           make_mp=True, make_emp=True,
                           lean='ignore', plean='ignore', mplean='ignore',
                           elean='ignore', melean='ignore'
                           )
            me.plotme()
            me.plotmp()
            '''
            self.clist = deque([])
            self.cpairs = np.array(cpairs_list)
            ppairs = []
            _pindices = []
            if not closed:
                pnta = point2d(self.cpairs[0][0][0],
                               self.cpairs[0][0][1],
                               lean=plean
                               )
                pntb = point2d(self.cpairs[0][1][0],
                               self.cpairs[0][1][1],
                               lean=plean
                               )
                ppairs.append([pnta, pntb])
                _pindices.append([0, 1])
                self.clist.extend([list(self.cpairs[0][0]),
                                   list(self.cpairs[0][1])])
                for i, _ in enumerate(self.cpairs[1:], start=1):
                    pnta = ppairs[i-1][1]
                    pntb = point2d(self.cpairs[i][1][0],
                                   self.cpairs[i][1][1],
                                   lean=plean
                                   )
                    ppairs.append([pnta, pntb])
                    _pindices.append([i, i+1])
                    self.clist.append(list(self.cpairs[i][1]))
            else:
                # If the mul-edge is indeed closed
                pass
            # -------------------------------------------
            self.ppairs = [ppairs[i] for i in eindices]
            # -------------------------------------------
            # Build the DB of all poinnts in this multi-edge object
            self.dbbuild_points(method='ppairs')
            # -------------------------------------------
            self.pindices = [_pindices[i] for i in eindices]
            self.eindices = eindices
            # -------------------------------------------
            # Build the memory ID data-bases (only for developers)
            self.dbbuild_pmid()
            self.dbbuild_pmid_pairs()
            self.dbbuild_edges()
            # ---------------------------
            if make_mp:
                self.dbbuild_mpoint()
            # ---------------------------
            if make_emp:
                self.dbbuild_empoint()
        # ===========================================
        if method == 'up2d_list' and self.ordered:
            '''
            points = [point2d(x=0, y=0, lean='ignore'),
                      point2d(x=1, y=0, lean='ignore'),
                      point2d(x=0, y=1, lean='ignore'),
                      point2d(x=1, y=1.01, lean='ignore'),
                      point2d(x=2, y=2, lean='ignore'),
                      point2d(x=2, y=2, lean='ignore'),]
            from muledge2d import muledge2d
            me = muledge2d(method='up2d_list',
                           ordered=True,
                           closed=False,
                           points=points,
                           make_mp=True, make_emp=True,
                           lean='ignore', plean='ignore', mplean='ignore',
                           elean='ignore', melean='ignore'
                           )
            '''
            # Make points unique
            _mp_ = mulpoint2d(method='up2d_list', point_objects=points)
            # ---------------------------
            # saa points
            self.points = _mp_.points
            # ---------------------------
            # Make clist
            self.clist = deque([[p.x, p.y] for p in self.points])
            # ---------------------------
            npoints = len(self.clist)
            # ---------------------------
            # Build pindices
            self.pindices = [[i, i+1] for i in range(npoints-1)]
            # ---------------------------
            # Build pmids
            self.pmids = deque([id(p) for p in self.points])
            # ---------------------------
            # Build pmid_pairs
            self.pmid_pairs = [[self.pmids[i[0]], self.pmids[i[1]]]
                               for i in self.pindices]
            # ---------------------------
            # Build edges and eindices
            self.edges = deque([edge2d(method='up2d',
                                       pnta=self.points[i[0]],
                                       pntb=self.points[i[1]],
                                       edge_lean=self.elean,
                                       ) for i in self.pindices]
                               )
            self.eindices = deque([i for i in range(npoints)])
            # ---------------------------
            # Build multi-point object
            self.mpoint = mulpoint2d(method='up2d_list',
                                     point_objects=self.points,
                                     lean=self.mplean)
            # ---------------------------
            # Build edge wise multi-point
            self.empoints = [mulpoint2d(method='up2d_list',
                                        point_objects=[self.points[i[0]],
                                                       self.points[i[1]]],
                                        lean=self.mplean)
                             for i in self.pindices]
            # ---------------------------
        if method == 'ue2d_list' and self.ordered:
            pass
        # ============================
        for p in self.points:
            p.links['me'].append(self)

    # #########################################################
    def __att__(self):
        return gops.att(self)

    def __repr__(self):
        string = []
        string.append(str(id(self)) + ', ')
        string.append(f'e-{len(self.edges)} p-{len(self.points)}, ')
        if self.closed:
            string.append('closed, ')
        else:
            string.append('not closed, ')
        string.append(f'e-{self.esense}, p-{self.psense}, ')
        if self.ordered:
            string.append('p-ordered, ')
        else:
            string.append('p-not ordered, ')
        return ''.join(string)

    def __len__(self):
        return len(self.edges)
    # #########################################################

    def dbbuild_plean(self):
        '''
        Build a list of point lean values
        '''
        self.plean = deque([p.lean for p in self.points])

    def dbbuild_elean(self):
        '''
        Build a list of edge lean values
        '''
        self.elean = deque([e.lean for e in self.edges])

    def dbbuild_pmid(self):
        '''
        Build a list of mid from me.points
        '''
        self.pmids = deque([id(p) for p in self.points])

    def dbbuild_pmid_pairs(self):
        '''
        Build a list of mid from me.ppairs
        '''
        self.pmid_pairs = [[id(pp[0]), id(pp[1])] for pp in self.ppairs]

    def dbbuild_points(self,
                       method='ppairs',
                       data=None):
        '''
        Data-Base build -- points list
        '''
        if method == 'ppairs':
            self.points = deque([self.ppairs[0][0]])
            for pp in self.ppairs:
                self.points.append(pp[1])

    def dbbuild_mpoint(self):
        '''
        Create multi-point object of all the points in the mul
        '''
        self.mpoint = mulpoint2d(method='up2d_list',
                                 point_objects=self.points,
                                 lean=self.mplean)

    def dbbuild_empoint(self):
        '''
        Create a list of edge-wise multi-point objects
        A multi-point object will be created for every edge object.
        NOTE1: This enables easy introduction of roughness to an edge.
        NOTE2: This will also be useful for GB definition in MC sim GS.
        '''
        self.empoints = [mulpoint2d(method='up2d_list',
                                    point_objects=pp,
                                    lean=self.mplean) for pp in self.ppairs]

    def dbbuild_edges(self):
        '''
        Build data-base of edges using ppairs.
        '''
        self.edges = deque([edge2d(method='up2d', pnta=pp[0], pntb=pp[1],
                                   edge_lean=self.elean)
                            for pp in self.ppairs])

    # #########################################################
    def edit_edge_add_point(self, eind, obj):
        """
        Adds point (obj) between pnta and pntb and make two edges
        Original edge gets edited, new one will be returned.
        returns the additional edge.
        If EDGE be the edge and OBJ be the point object being added, then
        the resulting edges will be:
            edge 1: EDGE.pnta  --  OBJ
            edge 2: OBJ  --  EDGE.pntb

        Parameters
        ----------
        eind : integer
            Specifies which edge to be edited

        obj : UPXO point2d type
            Specifies the UPXO point2d object to be inserted

        Returns
        -------
        None.

        """
        '''
        PROOF OF CONCEPT CODE
        pnta, pntb, obj = point2d(0,0), point2d(1,0), point2d(0.5,0)
        edge = edge2d(pnta=pnta, pntb=pntb, edge_lean='ignore')
        edge.update_point('b', obj, method='up2d')
        new_edge = edge2d(pnta=obj, pntb=pntb, edge_lean='ignore')
        [edge, new_edge]


        1. MAKE ME.EDGES --- A DEQUE OBJECT: EASIER TO WORK WITH INSERTING
        '''
        if dth.unique_of_datatypes([obj])[0] == "<class 'UPXO-point.point2d'>":
            # self.edges[eind].
            pass

    # #########################################################
    @property
    def centroid(self):
        '''
        Return the UPXO centroid point
        '''
        _x, _y = self.mpoint.centroid
        return point2d(_x, _y, lean='ignore')

    @property
    def centroids(self):
        '''
        Return centroids of constituent edges of me.
        '''
        return list(map(lambda e: e.centroid, self.edges))

    @property
    def lengths(self):
        '''
        Returns lengths of constituent edges of me.
        '''
        return list(map(lambda e: e.length, self.edges))

    @property
    def slopes(self):
        '''
        Returns slopes of constituent edges of me.
        '''
        return list(map(lambda e: e.slope, self.edges))

    @property
    def length(self):
        '''
        Returns total length of constituent edges of me.
        '''
        return np.sum(self.lengths)

    @property
    def mean_length(self):
        '''
        Returns mean length of constituent edges of me.
        '''
        return np.mean(self.lengths)

    @property
    def mean_slope(self):
        '''
        Returns mean slope of constituent edges of me.
        '''
        return np.mean(self.slopes)

    @property
    def roughness(self):
        pass

    @property
    def angles180(self):
        '''
        clist = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1],
                 [0, -1], [1, -1]]
        from muledge2d import muledge2d
        me = muledge2d(method='clist',
                       ordered=True,
                       closed=False,
                       clist=clist,
                       make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore'
                       )
        me.angles180
        '''
        _norm_ = np.linalg.norm
        angles = []
        for coord in self.clist:
            if _norm_(coord) >= self.EPS:
                dot_product = np.dot([1, 0], coord/_norm_(coord))
                angle = np.arccos(dot_product)*57.29577951308232
                if abs(coord[1]) >= self.EPS:
                    angle = angle * coord[1]/abs(coord[1])
            else:
                angle = 360.0
            angles.append(angle)
        return angles

    @property
    def angles(self):
        '''
        clist = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1],
                 [0, -1], [1, -1]]
        from muledge2d import muledge2d
        me = muledge2d(method='clist',
                       ordered=True,
                       closed=False,
                       clist=clist,
                       make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore'
                       )
        me.angles
        '''
        angles = []
        _norm_ = np.linalg.norm
        for coord in self.clist:
            if _norm_(coord) >= self.EPS:
                dot_product = np.dot([1, 0], coord/_norm_(coord))
                angle = np.arccos(dot_product)*57.29577951308232
                y = coord[1]
                if abs(y) >= self.EPS:
                    if y/abs(y) < 0:
                        angle = 360.0-angle
            else:
                angle = 360.0
            angles.append(angle)
        return angles

    # #########################################################
    def make_ring(self, saa=True, throw=False, close_index=0):
        '''
        1. If not me.closed, then make_ring closes the me object.
        2. Update operations:
            1.1. points
            1.2. edges
            1.3. cpairs
            1.4. ppairs
            1.5. empoints
            1.6. pindices
            1.7. closed

        EXAMPLE:
            cpairs_list = [[[1, 1], [-1, 1]],
                           [[-1, 1], [-1, -1]],
                           [[-1, -1], [1, -1]],
                           ]
            from muledge2d import muledge2d
            me = muledge2d(method='cpairs_list',
                           ordered=True,
                           closed=False,
                           cpairs_list=cpairs_list,
                           make_mp=True, make_emp=True,
                           lean='ignore', plean='ignore', mplean='ignore',
                           elean='ignore', melean='ignore'
                           )
            me.is_ring(fast=True)
            me.is_ring(fast=False)
            me.make_ring()
            me.is_ring()
        '''
        if saa:
            if self.ordered:
                # 1. Close the points
                self.points.append(self.points[0])
                # 2. Close the edges
                self.edges.append(self.edges[0])
                # 3. Close the cpairs
                self.cpairs = np.append(self.cpairs,
                                        np.array([[self.cpairs[-1][1],
                                                   self.cpairs[0][0]]]
                                                 ),
                                        axis=0
                                        )
                # 4. Close the ppairs
                self.ppairs.append([self.ppairs[-1][1],
                                    self.ppairs[0][0]])
                # 5. Close the empoints
                self.empoints.append(self.empoints[0])
                # 6. Close the pindices
                if not self.ordered:
                    self.pindices.append[0]
                # 7. Close the pmids
                self.pmids.append(self.pmids[0])
                # 8. Close the pmid_pairs
                self.pmid_pairs.append([self.pmid_pairs[-1][1],
                                        self.pmid_pairs[0][0]])
                # 9. Update the closed
                self.closed = True
        if saa:
            if not self.ordered:
                pass

    def is_ring(self, fast=True):
        '''
        Returns True if 1st point is same as last point, else returns False
        Requirements to become ring:
            1. All edges to have same sense
            2. Edges to move adjacently from pnta of edge0 to pntb of edge(-1)
            3. pnta of edge0 = pntb of edge(-1)
        '''
        if self.ordered:
            if fast:
                if self.pmids[0] == self.pmids[-1]:
                    to_return = True
                else:
                    to_return = False
            else:
                if self.points[0] == self.points[-1]:
                    to_return = True
                else:
                    to_return = False

        return to_return

    # #########################################################
    def pop_point_by_index(self, n):
        """
        Delete n'th point object from the mul-edge object

        Parameters
        ----------
        n : int
            Point number within list(range(len(me.points)))

        Returns
        -------
        None.

        EXAMPLE
        -------
        clist = [[0.00, 0.00],
                 [0.50, 0.00],
                 [1.00, 0.00],
                 [1.50, 0.50],
                 [1.00, 1.01],
                 [2.00, 2.00],
                 [2.50, 0.00]
                 ]
        from muledge2d import muledge2d
        me = muledge2d(method='clist',
                       ordered=True,
                       closed=False,
                       clist=clist,
                       make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore'
                       )
        ['points:==', me.points, 'mulpoint.points:==', me.mpoint.points,
         'edges:==', me.edges, 'pmids:==', me.pmids,
         'pmid_pairs:==', me.pmid_pairs, 'pindices==', me.pindices,
         'clist:', me.clist]

        me.plotme()

        me.pop_point_by_index(0)

        ['points:==', me.points, 'mulpoint.points:==', me.mpoint.points,
         'edges:==', me.edges, 'pmids:==', me.pmids,
         'pmid_pairs:==', me.pmid_pairs, 'pindices==', me.pindices,
         'clist:', me.clist]

        [[[f'emp.{i}-a', id(me.empoints[i].points[0]),
           f'e.{i}-a', id(me.edges[i].pnta)],
          [f'emp.{i}-b', id(me.empoints[i].points[1]),
           f'e.{i}-b',id(me.edges[i].pntb)]]
         for i in range(len(me.edges))]

        me.plotme()
        """
        '''
        PROOF OF CONCEPT CODE - 1
        -------------------------
        edges = [edge2d(pnta=point2d(0.00, 0.00), pntb=point2d(0.50, 0.00)),
                 edge2d(pnta=point2d(0.50, 0.00), pntb=point2d(1.00, 0.00)),
                 edge2d(pnta=point2d(1.00, 0.00), pntb=point2d(1.50, 0.50)),
                 edge2d(pnta=point2d(1.50, 0.50), pntb=point2d(1.00, 1.01)),
                 edge2d(pnta=point2d(1.00, 1.01), pntb=point2d(2.00, 2.00)),
                 edge2d(pnta=point2d(2.00, 2.00), pntb=point2d(2.50, 0.00)),
                 ]
        pmid_pairs = np.array([[111, 222],
                               [222, 333],
                               [333, 444],
                               [444, 555],
                               [555, 666],
                               [666, 777]
                               ])

        to_pop = 222

        all_pmids = np.hstack((pmid_pairs.T[0], pmid_pairs.T[1]))
        pmids_unique = np.sort(list(set(all_pmids)))
        connections = [np.count_nonzero(all_pmids==i) for i in pmids_unique]
        pmid_connections = np.array([pmids_unique, connections])

        for i in np.array(np.where(pmid_pairs==to_pop)).T:
            pmid_pairs[i[0]][i[1]] = -1

        # Get number of points to keep intact in each edge
        _where_ = np.where
        keep_n = []
        for pair in pmid_pairs:
            keep_n.append(2-list(pair).count(-1))
        print(keep_n)

        # Identify which edges have points to be edited
        edges_to_edit = [i for i,_ in enumerate(keep_n) if _!=2]

        # Identify which edges have points to be edited along with points
        edit_edge = []
        for i, pair in enumerate(pmid_pairs):
            edit_edge.append(list(np.where(pair==-1)[0]))

        # Start editing the edges
        remove = []
        if edges_to_edit:
            for en in edges_to_edit:
                if en < len(edges)-1:
                    edge = edges[en]
                    point_to_edit = edit_edge[en][0]
                    if edit_edge[en+1]:
                        next_edge = edges[en+1]
                        next_point_to_edit = edit_edge[en+1][0]
                        if point_to_edit == 0:
                            if next_point_to_edit == 0:
                                edge.pnta.x = next_edge.pntb.x
                                edge.pnta.y = next_edge.pntb.y
                            elif next_point_to_edit == 1:
                                edge.pnta.x = next_edge.pnta.x
                                edge.pnta.y = next_edge.pnta.y
                        elif point_to_edit == 1:
                            if next_point_to_edit == 0:
                                edge.pntb.x = next_edge.pntb.x
                                edge.pntb.y = next_edge.pntb.y
                            elif next_point_to_edit == 1:
                                edge.pntb.x = next_edge.pnta.x
                                edge.pntb.y = next_edge.pnta.y
                    else:
                        remove.append(en)
                else:
                    if edit_edge[en]:
                        remove.append(en)

        edges


        for i, rem_n in enumerate(remove, start=0):
            del edges[rem_n-i]

        edges

        PROOF OF CONCEPT CODE - 2
        -------------------------
        clist = [[0.00, 0.00],
                 [0.50, 0.00],
                 [1.00, 0.00],
                 [1.50, 0.50],
                 [1.00, 1.01],
                 [2.00, 2.00],
                 [2.50, 0.00]
                 ]
        from muledge2d import muledge2d
        me = muledge2d(method='clist',
                       ordered=True,
                       closed=False,
                       clist=clist,
                       make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore'
                       )
        me.edges
        n = 1
        me.points
        PMID = me.pmids[n]

        to_pop = PMID

        pmid_pairs = np.array(deepcopy(me.pmid_pairs))
        for i in np.array(np.where(pmid_pairs==to_pop)).T:
            pmid_pairs[i[0]][i[1]] = -1

        # Get number of points to keep intact in each edge
        _where_ = np.where
        keep_n = []
        for pair in pmid_pairs:
            keep_n.append(2-list(pair).count(-1))
        print(keep_n)

        # Identify which edges have points to be edited
        edges_to_edit = [i for i,_ in enumerate(keep_n) if _!=2]

        # Identify which edges have points to be edited along with points
        edit_edge = []
        for i, pair in enumerate(pmid_pairs):
            edit_edge.append(list(np.where(pair==-1)[0]))

        # Start editing the edges
        nedges = len(me.edges)
        edges = me.edges
        remove = []
        if edges_to_edit:
            for en in edges_to_edit:
                if en < nedges-1:
                    edge = me.edges[en]
                    point_to_edit = edit_edge[en][0]
                    if edit_edge[en+1]:
                        next_edge = me.edges[en+1]
                        next_point_to_edit = edit_edge[en+1][0]
                        if point_to_edit == 0:
                            if next_point_to_edit == 0:
                                edge.pnta.x = next_edge.pntb.x
                                edge.pnta.y = next_edge.pntb.y
                            elif next_point_to_edit == 1:
                                edge.pnta.x = next_edge.pnta.x
                                edge.pnta.y = next_edge.pnta.y
                        elif point_to_edit == 1:
                            if next_point_to_edit == 0:
                                edge.pntb.x = next_edge.pntb.x
                                edge.pntb.y = next_edge.pntb.y
                            elif next_point_to_edit == 1:
                                edge.pntb.x = next_edge.pnta.x
                                edge.pntb.y = next_edge.pnta.y
                    else:
                        remove.append(en)
                else:
                    if edit_edge[en]:
                        remove.append(en)

        for i, rem_n in enumerate(remove, start=0):
            del me.edges[rem_n-i]
        '''
        _where_ = np.where
        # ##############################################################
        #            EDIT --> MPOINT
        # Update mpoint
        self.mpoint.pop_points_indices(n)
        # ##############################################################
        #            EDIT --> POINTS
        del self.points[n]
        # ##############################################################
        #            EDIT --> EDGES
        # Get the mid of point to be removed
        to_pop = self.pmids[n]
        # -----------------------------------------
        # Copy pmid_pairs and mark points to be removed
        from copy import deepcopy
        pmid_pairs = np.array(deepcopy(self.pmid_pairs))
        for i in np.array(_where_(pmid_pairs == to_pop)).T:
            pmid_pairs[i[0]][i[1]] = -1
        # -----------------------------------------
        # Get number of points to keep intact in each edge
        keep_n = []
        for pair in pmid_pairs:
            keep_n.append(2-list(pair).count(-1))
        # -----------------------------------------
        # Identify which edges have points to be edited
        edges_to_edit = [i for i, _ in enumerate(keep_n) if _ != 2]
        # -----------------------------------------
        # Identify which edges have points to be edited along with points
        edit_edge = []
        for i, pair in enumerate(pmid_pairs):
            edit_edge.append(list(_where_(pair == -1)[0]))
        # -----------------------------------------
        # Start editing the edges
        remove = []
        if edges_to_edit:
            for en in edges_to_edit:
                if en < len(self.edges)-1:
                    point_to_edit = edit_edge[en][0]
                    if edit_edge[en+1]:
                        next_edge = self.edges[en+1]
                        next_point_to_edit = edit_edge[en+1][0]
                        if point_to_edit == 0:
                            this_empoint = self.empoints[en].points[0]
                            if next_point_to_edit == 0:
                                self.edges[en].pnta.x = next_edge.pntb.x
                                self.edges[en].pnta.y = next_edge.pntb.y
                                this_empoint.x = next_edge.pntb.x
                                this_empoint.y = next_edge.pntb.y
                            elif next_point_to_edit == 1:
                                self.edges[en].pnta.x = next_edge.pnta.x
                                self.edges[en].pnta.y = next_edge.pnta.y
                                this_empoint.x = next_edge.pnta.x
                                this_empoint.y = next_edge.pnta.y
                        elif point_to_edit == 1:
                            this_empoint = self.empoints[en].points[1]
                            if next_point_to_edit == 0:
                                self.edges[en].pntb.x = next_edge.pntb.x
                                self.edges[en].pntb.y = next_edge.pntb.y
                                this_empoint.x = next_edge.pntb.x
                                this_empoint.y = next_edge.pntb.y
                            elif next_point_to_edit == 1:
                                self.edges[en].pntb.x = next_edge.pnta.x
                                self.edges[en].pntb.y = next_edge.pnta.y
                                this_empoint.x = next_edge.pnta.x
                                this_empoint.y = next_edge.pnta.y
                    else:
                        remove.append(en)
                else:
                    if edit_edge[en]:
                        remove.append(en)
        # -----------------------------------------
        for i, rem_n in enumerate(remove, start=0):
            del self.edges[rem_n-i]
            del self.empoints[rem_n-i]
        for empoint in self.empoints:
            empoint.recompute(flag_mids=True,
                              flag_dist=True,
                              flag_basics=True,
                              flag_npoint=True,
                              flag_locxy=True,
                              flag_centroid=True)
        # ##############################################################
        #            RE-BUILD --> pmids
        del self.pmids[n]
        #            RE-BUILD --> clist
        del self.clist[n]
        #            RE-BUILD --> pindices
        self.pindices = [[i, i+1] for i in range(len(self.clist)-1)]
        #            RE-BUILD --> eindices
        self.eindices = deque([i for i, _ in enumerate(self.points)])
        #            RE-BUILD --> pmid_pairs
        self.pmid_pairs = [[self.pmids[i[0]], self.pmids[i[1]]]
                           for i in self.pindices]

    def pop_points_by_indices(self, n):
        """


        Parameters
        ----------
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        EXANPLE
        -------
        clist = [[0.00, 0.00],
                 [0.50, 0.00],
                 [1.00, 0.00],
                 [1.50, 0.50],
                 [1.00, 1.01],
                 [2.00, 2.00],
                 [2.50, 0.00]
                 ]
        from muledge2d import muledge2d
        me = muledge2d(method='clist',
                       ordered=True,
                       closed=False,
                       clist=clist,
                       make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore'
                       )
        me.points
        n = [0, 2, 4, 6]
        me.pop_points_by_indices(n)
        me.points
        """
        if type(n) not in dth.dt.ITERABLES:
            n = [n]
        if np.prod(np.array(n) >= 0) == 1 and max(n) <= len(self.points):
            self.pop_point_by_index(n[0])
            if len(n) > 1:
                for i, _ in enumerate(n[1:], start=1):
                    self.pop_point_by_index(n[i]-i)
        else:
            print('Enter valid indices')

    def pop_point_by_coord(self, coord=None, tdist=0.1):
        """
        Delete point object at coord from the mul-edge object

        Parameters
        ----------
        coord : list/tuple/deque/numpy array
            A single co-ordinate pair

        Returns
        -------
        None.

        EXAMPLE
        -------
        clist = [[0.00, 0.00],
                 [0.50, 0.00],
                 [1.00, 0.00],
                 [1.50, 0.50],
                 [1.00, 1.01],
                 [2.00, 2.00],
                 [2.50, 0.00]
                 ]
        from muledge2d import muledge2d
        me = muledge2d(method='clist',
                       ordered=True,
                       closed=False,
                       clist=clist,
                       make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore'
                       )

        coord = [0.1, 0.0]
        tdist = 1.0


        me.points
        me.edges

        me.pop_point_by_coord(coord=coord, tdist=0.1)

        me.points
        me.edges
        me.clist
        me.mpoint
        me.empoints
        me.slopes
        me.angles180
        me.angles
        """
        _point_ = point2d(x=coord[0], y=coord[1], lean='ignore')
        equalities = list(_point_.__eq__(self.points,
                                         tdist=tdist,
                                         use_self_tdist=False,
                                         point_types='upxo'
                                         )
                          )
        if any(equalities):
            npoints = equalities.count(True)
            point_loc = list(np.where(np.array(equalities))[0])
            if npoints >= len(self.points)-2:
                print('Too many points within tolerance distance')
                print('No points will be removed')
                print('You may try reducing tdist if not already zero')
            else:
                self.pop_points_by_indices(point_loc)
        else:
            print('Point at coord is not part of this muledge2d')
            print('No points have been removed')

    # #########################################################
    def insert_point_by_index(self,
                              obj,
                              index=None):
        '''
        Allow inserting one point at a time in index
        PRE-REQUISITE DATA
        ------------------
        clist = [[0.00, 0.00], [0.50, 0.00], [1.00, 0.00], [1.50, 0.50],
                 [1.00, 1.01], [2.00, 2.00], [2.50, 0.00]]
        from muledge2d import muledge2d
        me = muledge2d(method='clist', ordered=True, closed=False,
                       clist=clist, make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore')

        EXAMPLE-1
        ---------
        me.plotme()
        me.insert_point_by_index(point2d(0, 5), index=1)
        me.plotme()

        me.edges, me.points
        me.mpoint, me.mpoint.points, me.mpoint.locx, me.mpoint.locy
        me.pmids, me.pmid_pairs
        me.pindices, me.eindices
        me.lengths, me.slopes, me.centroid, me.centroids

        me.insert_point_by_index(point2d(-2, 2), index=1)
        me.plotme()

        me.pop_points_by_indices([1])
        me.plotme()

        me.pop_point_by_coord(coord=[0, 4.5], tdist = 1)
        me.plotme()
        '''
        # Check validity of input
        input_valid = False
        if obj.__class__.__name__ == 'point2d':
            input_valid = True
        elif dth.IS_CPAIR(obj):
            obj = point2d(obj[0],
                          obj[1],
                          lean=self.plean)
            input_valid = True
        # Proceed only if point does not coincide with existing point(s),
        # and inpuyt is valid
        if not any(obj.distance(otype='up2d_list',
                                obj=self.points) <= self.EPS) and input_valid:
            # Update the mul-point object
            self.mpoint.__add__(obj=[obj], indices=index)
            # Points would have updated automatically
            # Update the clist
            self.clist.insert(index, [obj.x, obj.y])
            # ========================
            self.dbbuild_pmid()
            self.pindices = [[i, i+1] for i in range(len(self.clist)-1)]
            self.eindices = deque([i for i, _ in enumerate(self.points)])
            # ========================
            self.pmid_pairs = [[self.pmids[i[0]], self.pmids[i[1]]]
                               for i in self.pindices]
            # ##############################################################
            # UPDATE EDGES
            io = id(obj)
            # 1. Identify edge and point to edit
            CP = []
            # Location 0 is the edge location
            # Location 1 is the point location in the edge
            for i, pmp in enumerate(self.pmid_pairs):
                if io in pmp:
                    CP.append([i,
                               [pmp[0] == io, pmp[1] == io].index(True)
                               ])
            # 2. Update edge and insert new edge
            # NOTE: ALL CASES HAVE NOT BEEN DEALT WITH. SHOULD BE OK FOR NOW
            if CP[0][1] == 1:
                #print(len(self.edges))
                #print(self.edges[CP[1][0]])
                if CP[1][0] < len(self.edges):
                    # Update edge
                    self.edges[CP[0][0]].pntb, NE = obj, edge2d(pnta=obj, pntb=self.edges[CP[1][0]].pnta)
                elif CP[1][0] == len(self.edges):
                    self.edges[CP[0][0]].pntb, NE = obj, edge2d(pnta=obj, pntb=self.edges[CP[0][0]].pntb)
                # Update properties of modified edge
                self.edges[CP[0][0]].post_deformation_updates()
            self.edges.insert(CP[1][0], NE)
            # ##############################################################
            #            RE-BUILD --> pindices
            self.pindices = [[i, i+1] for i in range(len(self.clist)-1)]
            # -------------------------------
            #            RE-BUILD --> eindices
            self.eindices = deque([i for i, _ in enumerate(self.points)])
            # -------------------------------
        else:
            if input_valid:
                print('Input point already exists in the Multi-Edge')
            else:
                print('Please enter point in either UPXO or coordinate pair format')

    # #########################################################
    def fine(self, level):
        '''
        PRE-REQUISITE DATA
        ------------------
        clist = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]

        clist = [[0.00, 0.00], [0.50, 0.00], [1.00, 0.00], [1.50, 0.50],
                 [1.00, 1.00], [2.00, 2.00], [2.50, 0.00]]

        from muledge2d import muledge2d
        me = muledge2d(method='clist', ordered=True, closed=False,
                       clist=clist, make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore')

        EXAMPLE-1
        ---------
        me.plotme()
        me.edges
        me.fine(level=1)
        me.plotme()
        '''
        if type(level) == int and level >= 0 and level <= 2:
            for _ in range(level):
                centroids = self.centroids
                insertion_locations = list(range(1, len(self.edges)+1))
                for i, loc in enumerate(insertion_locations):
                    if i == 0:
                        self.insert_point_by_index(centroids[i],
                                                   index=loc)
                    else:
                        self.insert_point_by_index(centroids[i],
                                                   index=loc+insertion_locations[i-1])
        else:
            print('Please enter valid level. Max level allowed = 2')

    def insert_point(self, obj, index=None,
                     insertion_check='edge'):
        '''
        Allow inserting one point at a time
        EXAMPLE:

        clist = [[0.00, 0.00],
                 [0.50, 0.00],
                 [1.00, 0.00],
                 [1.50, 0.50],
                 [1.00, 1.01],
                 [2.00, 2.00],
                 [2.50, 0.00]
                 ]
        from muledge2d import muledge2d
        me = muledge2d(method='clist',
                       ordered=True,
                       closed=False,
                       clist=clist,
                       make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore'
                       )

        me.mpoint.points
        me.points
        me.edges

        # Points to add
        obj = point2d(20, 15)

        # Provide the indices to insert the points at
        index = 1

        # Update the mul-point object
        me.mpoint.__add__(obj=[obj], indices=index)
        me.mpoint.points

        me.points
        me.edges
        '''
        # ---------------------------
        # Number of points in the me.mpoint
        mpoint_npoints_bf = len(me.mpoint)
        # ---------------------------
        # Update mul-point
        me.mpoint.__add__([obj], indices=[index])
        # ---------------------------
        # Number of points in the me.mpoint
        mpoint_npoints_af = len(me.mpoint)
        # ---------------------------
        # Check if all points were added
        if mpoint_npoints_af == mpoint_npoints_bf+1:
            # The point was added
            for e in me.edges:
                if not e.contains_point(obj=obj,
                                        method='parallelity',
                                        tdist=0.0)[0]:
                    # Point
                    pass
        elif mpoint_npoints_af == mpoint_npoints_bf:
            # Point coord was already in me.mpoint and was not added
            pass
        # ---------------------------
        # saa points
        me.points = me.mpoint.points
        # ---------------------------
        # Update clist
        me.clist = deque([[p.x, p.y] for p in me.points])
        # ---------------------------
        npoints = len(me.clist)
        # ---------------------------
        # RE-BUILD --> pindices
        me.pindices = [[i, i+1] for i in range(npoints-1)]
        # ---------------------------
        me.pmids = deque([id(p) for p in me.points])
        # ---------------------------
        if insertion_check == 'edge':
            # Works only for non-intersecting edges
            pass
        if insertion_check == 'index':
            pass
        # ---------------------------
        # ---------------------------
        # ---------------------------
        # Update points
        # Update pindices
        # Update ppairs
        # Update clist and Update cpairs
        # Update pmids
        # Update pmid_pairs
        # Update plean
        # Update mpoint
        # Update edges
        # Update elean
        # Update empoints
        # Update the ring object
        pass

    def insert_coord_at(self, coord, indices=[0, 1]):
        # Update clist and Update cpairs
        # Update points
        # Update pindices
        # Update ppairs
        # Update pmids
        # Update pmid_pairs
        # Update plean
        # Update mpoint
        # Update edges
        # Update elean
        # Update empoints
        # Update the ring object
        pass

    def insert_point_bw(self, k=0.5):
        # Make coord
        # Make point object
        # Update points
        # Update pindices
        # Update ppairs
        # Update clist and Update cpairs
        # Update pmids
        # Update pmid_pairs
        # Update plean
        # Update mpoint
        # Update edges
        # Update elean
        # Update empoints
        # Update the ring object
        pass

    # #########################################################
    def move_nthpoint(self, n=0, xyincr=[0, 0], overlap_action='exit'):
        '''
        clist = [[0, 0],
                 [1, 0],
                 [1.5, 0.5],
                 [1, 1.01],
                 [2, 2],
                 [2.5, 0]]
        from muledge2d import muledge2d
        me = muledge2d(method='clist',
                       ordered=True,
                       closed=False,
                       clist=clist,
                       make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore'
                       )
        me.points
        me.mpoint.points
        me.pmids
        me.pmid_pairs
        me.clist
        me.lengths
        me.slopes

        me.move_nthpoint(n=0, xyincr=[10, 0], overlap_action='exit')

        me.points
        me.mpoint.points
        me.pmids
        me.clist
        me.lengths
        me.slopes
        '''
        # Move the point object
        # Update the cpairs
        # Check if new point coincides with any in self.points
        # If so, re-compute the multi-point

        # Update points
        # Update pindices
        # Update ppairs
        # Update clist and Update cpairs
        # Update pmids
        # Update pmid_pairs
        # Update plean
        # Update mpoint
        # Update edges
        # Update elean
        # Update empoints
        # Update the ring object
        n = 0
        _point_ = me.points[n].translate(method='xyincr',
                                           xyincr=[1,0],
                                           saa=False,
                                           make_new=True,
                                           throw=True)
        overlaps = [_point_ == point for point in me.points]
        if not any(overlaps):
            self.points[n].translate(method='xyincr',
                                     xyincr=xyincr,
                                     saa=True,
                                     make_new=False,
                                     throw=False)
            self.clist[n] = [self.points[n].x, self.points[n].y]
        else:
            if overlap_action == 'exit':
                print('New position overlaps with an existing point.')
                print('Move operation rejected')
                print('Please choose overlap_action to replace, if you')
                print('        choose to continue updating the muledge')
            elif overlap_action == 'remove_edge2':
                print('Point belonging to the current edge will be edited')
                print('  Resulting hanging edge will be removed')
                # -------------------------------------
                overlap_pmids = [me.pmids[i] for i in np.where(overlaps)[0]]
                # -------------------------------------
                # pop points at np.where(overlaps)[0] locations

    def move_pointat(self, coord=[0, 0]):
        # Update points
        # Update pindices
        # Update ppairs
        # Update clist and Update cpairs
        # Update pmids
        # Update pmid_pairs
        # Update plean
        # Update mpoint
        # Update edges
        # Update elean
        # Update empoints
        # Update the ring object
        pass

    # #########################################################
    def explode(self, k, method='centroid'):
        # . Explode the multi-point
        # . Update self.cpairs form multi-point
        # . Update all multi-point properties. points need not be updated.

        # Update mpoint
        # Update points
        # Update pindices
        # Update ppairs
        # Update clist and Update cpairs
        # Update pmids
        # Update pmid_pairs
        # Update plean
        # Update edges
        # Update elean
        # Update empoints
        # Update the ring object
        pass

    def move(self):
        # Translate the multi-point
        # Update cpairs from multi-point

        # Update mpoint
        # Update points
        # Update pindices
        # Update ppairs
        # Update clist and Update cpairs
        # Update pmids
        # Update pmid_pairs
        # Update plean
        # Update edges
        # Update elean
        # Update empoints
        # Update the ring object
        pass

    def stretch(self):
        # Stretch the multi-point
        # Update cpairs from multi-point

        # Update mpoint
        # Update points
        # Update pindices
        # Update ppairs
        # Update clist and Update cpairs
        # Update pmids
        # Update pmid_pairs
        # Update plean
        # Update edges
        # Update elean
        # Update empoints
        # Update the ring object
        pass

    # #########################################################
    def plotme(self, i=None, j=None, show_coord=True):
        if not i:
            i = 0
        if not j:
            j = len(self.edges)
        if type(i) == int and i >= 0 and i < len(self.edges):
            if j and type(j) == int and j <= len(self.edges):
                from itertools import islice
                edges = list(islice(self.edges, i, j))
                for e in edges:
                    plt.plot([e.pnta.x, e.pntb.x],
                             [e.pnta.y, e.pntb.y],
                             'bo', linestyle='-')
                if show_coord:
                    for _x_, _y_ in zip(self.mpoint.locx[i:j+1],
                                        self.mpoint.locy[i:j+1]):
                        plt.text(_x_, _y_,
                                 '(%4.2f, %4.2f)' % (_x_, _y_),
                                 horizontalalignment='center',
                                 verticalalignment='bottom'
                                 )
            else:
                print('Enter valid slice indices')
        else:
            print('Enter valid slice indices')

    def plotmp(self):
        plt.plot(self.mpoint.locx, self.mpoint.locy, 'ks')
        for _x_, _y_ in zip(self.mpoint.locx, self.mpoint.locy):
            plt.text(_x_, _y_,
                     '(%4.2f, %4.2f)' % (_x_, _y_),
                     horizontalalignment='center',
                     verticalalignment='bottom'
                     )
