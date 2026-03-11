from edge2d import edge2d
from muledge2d import muledge2d
import matplotlib.pyplot as plt
from upxo._sup import dataTypeHandlers as dth
from upxo._sup import gops
from upxo.geoEntities import pops
import numpy as np
from numpy import inf
import upxo._sup.dataTypeHandlers as dth

__name__ = "UPXO-muledge"
__authors__ = ["Vaasu Anandatheertha"]
__lead_developer__ = ["Vaasu Anandatheertha"]
__emails__ = ["vaasu.anandatheertha@ukaea.uk", ]
__version__ = ["0.1", ]
__license__ = "GPL v3"


class ring2d():
    """
    Represents a ring in 2D Cartesian space.

    ENTRY POINTS TO RING OBJECT
        (A) Bottom-up
                Only from Multi-Edge object
        (B) Top-down:
            1. From Crystal object
            2. From Poly-crystal object
        (C) Cross-library:
            1. Shapely ring object
    """
    EPS = 0.000000000001
    __slots__ = ('dim',  # Dimensionality
                 'me',  # UPXO multi-edge object
                 'ce',  # Closing edge
                 'ce_pindices',  # point indices of closing edge
                 'lean',  # Lean of the self object
                 )

    def __init__(self,
                 me=None,  # multi-edge
                 dim=2,  # Dimensionality of the ring
                 lean='ignore',  # Lean specification
                 ):
        """
        PRE-REQUISITE DATA
        ------------------
        clist = [[-1, 0], [-1, 0.5], [0, 0.5], [1, 0.50],
                 [1, 0], [1, -0.5], [0, -0.5], [-0.5, -0.5]]

        from muledge2d import muledge2d
        me = muledge2d(method='clist', ordered=True, closed=False,
                       clist=clist, make_mp=True, make_emp=True,
                       lean='ignore', plean='ignore', mplean='ignore',
                       elean='ignore', melean='ignore')
       # me.plotme()

       # EXAMPLE-1
      # ---------
        from ring2d import ring2d
        ring = ring2d(me=me, lean='ignore', )
        ring.centroids
        """
        # ---------------------------------------------------------------------
        # Check if entered multi-edge is ordered and continous
        continuity, continous = [pops.CMPEQ_pnt_fast_exact(me.edges[i].pntb,
                                                           me.edges[i+1].pnta)
                                 for i in range(0, len(me.edges)-1)], True
        if continuity.count(False) == 0:
            # Multi-edge is ordered and continuous
            continous = True
        else:
            print('Multi-edge object is not ordered and continuos')
        # ---------------------------------------------------------------------
        # Check if continous multi-edge is non-self-intersecting
        self_intersects = False
        if continous:
            ne = len(me.edges)
            # Initiate Non-Self-Intersection array with False values
            NSI = np.array([[False for _ in range(ne)] for __ in range(ne)])
            # Extract upper-triangular indices, 1 unit spacing above the
            # primary diagonal
            uti = np.triu_indices(ne, k=1)
            # Get intersection points
            for r in uti[0]:
                ref_edge = me.edges[r]
                for c in uti[1]:
                    # Get point of intersection
                    _inter_ = ref_edge.edge2d_intersection(me.edges[c],
                                                           return_ratios=False,
                                                           sort=False,
                                                           print_=False)[0]
                    if len(_inter_) > 0:
                        cont = np.array(ref_edge.contains_point(_inter_[0]))
                        if np.array_equal(cont,
                                          np.array([True, False, False]),
                                          equal_nan=False):
                            NSI[r, c] = True
            # Assess non-intersecting state of the multi-edge
            self_intersects = bool(list(NSI[uti]).count(True))
        # ---------------------------------------------------------------------
        # Check for dangling edges
        # Lets assyume that the multi-edge has no dangling edges. If there are dangling edges,
        #  the ring object cannot be formed. they should be removed and then tried agains. 
        # ---------------------------------------------------------------------
        # If continous and non-self-intersecting, close the multi-edge object
        if not self_intersects:
            self.me = me
            self.dim = me.dim
            self.ce = edge2d(pnta=me.edges[-1].pntb,
                             pntb=me.edges[0].pnta,
                             edge_lean='ignore')
            self.ce_pindices = [me.pindices[-1][1], me.pindices[0][0]]
            if self.me.lean in ('ignore'):
                self.me.__rings__.append(self)
        # ---------------------------------------------------------------------
    def __att__(self):
        return gops.att(self)

    def __repr__(self):
        string = []
        string.append(str(id(self)) + ', ')
        string.append(f'e-{len(self.me.edges)+1} p-{len(self.me.points)}')
        return ''.join(string)

    def __len__(self):
        return len(self.me.edges)+1

    @property
    def centroid(self):
        return self.me.centroid

    @property
    def centroids(self):
        _ring_centroids_ = self.me.centroids
        _ring_centroids_.append(self.ce.centroid)
        return _ring_centroids_

    @property
    def lengths(self):
        _ring_lengths_ = self.me.lengths
        _ring_lengths_.append(self.ce.length)
        return _ring_lengths_

    @property
    def slopes(self):
        _ring_slopes_ = self.me.slopes
        _ring_slopes_.append(self.ce.slope)
        return _ring_slopes_

    @property
    def length(self):
        return sum(self.lengths)

    @property
    def length_mean(self):
        return self.length/len(self)

    @property
    def slope_mean(self):
        slopes = self.slopes
        return np.ma.masked_invalid(slopes).mean()

    @property
    def slope_max_hiv(self):
        slopes = self.slopes
        if np.inf in slopes or -np.inf in slopes:
            return inf
        else:
            return max(slopes)

    @property
    def slope_max_hi(self):
        slopes = self.slopes
        return np.ma.masked_values(slopes).mean()

    @property
    def slope_max_i(self):
        slopes = self.slopes
        return np.ma.masked_values(slopes).mean()

    @property
    def roughness(self):
        return self.me.roughness

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
        me.angles180
        '''
        _angles_ = [edge.angle for edge in self.me.edges]
        _angles_.append(self.ce.angle)
        return _angles_
    
    @property
    def nodes(self):
        # Returns cumulative list of all nodes of all constituent multiline segments 