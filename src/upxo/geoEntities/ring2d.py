"""
2D ring (closed loop of edges) geometric entity for UPXO.

Usage
-----
    from upxo.geoEntities.ring2d import ring2d

Classes
-------
ring2d : Closed 2D polygon represented as a multi-edge loop with a closing edge.

Notes
-----
A ``ring2d`` is formed from an ordered, continuous, non-self-intersecting
``muledge2d`` object by adding a closing edge between the last pntb and the
first pnta of the multi-edge chain.
"""
from upxo.geoEntities.edge2d import edge2d
from upxo.geoEntities.muledge2d import muledge2d
import matplotlib.pyplot as plt
from upxo._sup import dataTypeHandlers as dth
from upxo._sup import gops
from upxo.geoEntities import pops
import numpy as np
from numpy import inf
import upxo._sup.dataTypeHandlers as dth


class ring2d():
    """
    Closed 2D polygon represented as a multi-edge loop with a closing edge.

    A ``ring2d`` wraps a continuous, non-self-intersecting ``muledge2d``
    and adds a synthetic closing edge from the last endpoint back to the first.

    Entry points:

    * Bottom-up: from a ``muledge2d`` object.
    * Top-down: from a Crystal or Poly-crystal object.
    * Cross-library: from a Shapely ring object.

    Usage
    -----
        from upxo.geoEntities.ring2d import ring2d

    Notes
    -----
    The source ``muledge2d`` is expected to be ordered, continuous, and free
    from self-intersections before the closing edge is added.
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
        Parameters
        ----------
        me : muledge2d, optional
            Ordered, continuous multi-edge chain to close into a ring.
        dim : int, optional
            Dimensionality. Default is 2.
        lean : str, optional
            Lean specification for the ring object. Default is ``'ignore'``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.muledge2d import muledge2d
            from upxo.geoEntities.ring2d import ring2d
            clist = [[-1, 0], [-1, 0.5], [0, 0.5], [1, 0.5],
                     [1, 0], [1, -0.5], [0, -0.5], [-0.5, -0.5]]
            me = muledge2d(method='clist', ordered=True, closed=False,
                           clist=clist, lean='ignore')
            ring = ring2d(me=me, lean='ignore')
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
        """
        Return a string listing of all attributes of this ring.

        Returns
        -------
        str
            Attribute listing generated by ``upxo._sup.gops.att``.
        """
        return gops.att(self)

    def __repr__(self):
        """
        Return a compact ring summary string.

        Returns
        -------
        str
            Summary in the form ``id, e-<N> p-<M>`` where ``N`` includes
            the closing edge.
        """
        string = []
        string.append(str(id(self)) + ', ')
        string.append(f'e-{len(self.me.edges)+1} p-{len(self.me.points)}')
        return ''.join(string)

    def __len__(self):
        """
        Return the total number of edges.

        Returns
        -------
        int
            Number of source multi-edge edges plus the closing edge.
        """
        return len(self.me.edges)+1

    @property
    def centroid(self):
        """
        Centroid of the ring.

        Returns
        -------
        Point2d
            Centroid point delegated to the underlying ``muledge2d`` object.
        """
        return self.me.centroid

    @property
    def centroids(self):
        """
        Centroids of all ring edges.

        Returns
        -------
        list
            Edge centroids from the source multi-edge with the closing-edge
            centroid appended.
        """
        _ring_centroids_ = self.me.centroids
        _ring_centroids_.append(self.ce.centroid)
        return _ring_centroids_

    @property
    def lengths(self):
        """
        Lengths of all ring edges.

        Returns
        -------
        list of float
            Edge lengths from the source multi-edge with the closing-edge
            length appended.
        """
        _ring_lengths_ = self.me.lengths
        _ring_lengths_.append(self.ce.length)
        return _ring_lengths_

    @property
    def slopes(self):
        """
        Slopes of all ring edges.

        Returns
        -------
        list of float
            Edge slopes from the source multi-edge with the closing-edge
            slope appended.
        """
        _ring_slopes_ = self.me.slopes
        _ring_slopes_.append(self.ce.slope)
        return _ring_slopes_

    @property
    def length(self):
        """
        Total perimeter length of the ring.

        Returns
        -------
        float
            Sum of all source-edge lengths and the closing-edge length.
        """
        return sum(self.lengths)

    @property
    def length_mean(self):
        """
        Mean edge length of the ring.

        Returns
        -------
        float
            Arithmetic mean of all ring-edge lengths.
        """
        return self.length/len(self)

    @property
    def slope_mean(self):
        """
        Mean slope, ignoring NaN and infinite values.

        Returns
        -------
        float
            Mean of masked-valid edge slopes.
        """
        slopes = self.slopes
        return np.ma.masked_invalid(slopes).mean()

    @property
    def slope_max_hiv(self):
        """
        Maximum slope with vertical-edge handling.

        Returns
        -------
        float
            ``inf`` if any edge has infinite slope; otherwise the maximum
            slope value.
        """
        slopes = self.slopes
        if np.inf in slopes or -np.inf in slopes:
            return inf
        else:
            return max(slopes)

    @property
    def slope_max_hi(self):
        """
        Masked finite slope statistic.

        Returns
        -------
        float
            Mean of slope values after masking invalid values.
        """
        slopes = self.slopes
        return np.ma.masked_values(slopes).mean()

    @property
    def slope_max_i(self):
        """
        Mean of valid slopes.

        Returns
        -------
        float
            Mean of slope values after masking invalid values.
        """
        slopes = self.slopes
        return np.ma.masked_values(slopes).mean()

    @property
    def roughness(self):
        """
        Roughness measure delegated to the underlying multi-edge.

        Returns
        -------
        float
            Roughness value returned by ``self.me.roughness``.
        """
        return self.me.roughness

    @property
    def angles(self):
        """
        Orientation angles of all edges in the ring, including the closing edge.

        Returns
        -------
        list of float
            Angles in degrees for each edge, in chain order with the closing
            edge appended last.
        """
        _angles_ = [edge.angle for edge in self.me.edges]
        _angles_.append(self.ce.angle)
        return _angles_
    
    @property
    def nodes(self):
        """
        Cumulative nodes from all constituent multi-edge segments.

        Returns
        -------
        list
            Nodes collected from the underlying multi-edge representation.
        """
        # Returns cumulative list of all nodes of all constituent multiline segments
