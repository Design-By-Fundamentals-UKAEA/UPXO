"""
Orientation and misorientation utilities.

Functions
---------
eamo : Euler-angle misorientation (major work done; validations pending).
qmo  : Quaternion misorientation (major work done; validations pending).
"""

import numpy as np
from defdap.quat import Quat


def eamo(e1, e2list, symm):
    """
    Euler-angle misorientation between one orientation and a list of others.

    Parameters
    ----------
    e1 : ndarray, shape (3,)
        Bunge Euler angles of the reference orientation, in degrees.
    e2list : ndarray, shape (N, 3)
        Bunge Euler angles of N comparison orientations, in degrees.
    symm : str
        Crystal symmetry label (e.g. ``'cubic'``).

    Returns
    -------
    ndarray, shape (N,)
        Misorientation angles in degrees.

    Examples
    --------
    >>> import numpy as np
    >>> from upxo.xtalphy.orientation import eamo
    >>> e1 = np.array([0, 0, 0], dtype=float)
    >>> e2list = np.array([[0, 0, 0], [45, 0, 0], [45, 45, 0]], dtype=float)
    >>> eamo(e1, e2list, 'cubic')
    """
    # VALIDATION FOR e1 AND e2list
    # Initial set of validations and corrections
    # Convert to radians
    e1 *= 0.017453
    e2list *= 0.017453
    # ------------------------------------
    q1 = Quat.fromEulerAngles(e1[0], e1[1], e1[2])
    mo = [q1.misOri(Quat.fromEulerAngles(e2[0], e2[1], e2[2]), symm)
          for e2 in e2list]
    return 2*np.arccos(mo)*57.295779


def qmo(q1, q2list, symm):
    """
    Quaternion misorientation between one quaternion and a list of others.

    Parameters
    ----------
    q1 : defdap.quat.Quat
        Reference quaternion.
    q2list : list of defdap.quat.Quat
        Comparison quaternions.
    symm : str
        Crystal symmetry label (e.g. ``'cubic'``).

    Returns
    -------
    ndarray
        Misorientation angles in degrees.

    Examples
    --------
    >>> qmo(q1, q2list, 'cubic')
    """
    # VALIDATION FOR q1 AND q2
    # VALIDATION TO Also deal with q1 ans q2 which are not defdap types
    return 2*np.arccos([q1.misOri(q2, symm) for q2 in q2list])*57.295779


def ipf(vector, symm):
    """Ipf."""
    raise NotImplementedError("ipf is not yet implemented.")


class grainoris():
    """
    Grain orientation container: Euler angles, quaternions, and statistics.

    Examples
    --------
    >>> import numpy as np
    >>> from upxo.xtalphy.orientation import grainoris
    >>> n = 4
    >>> ea1 = np.random.randint(0, 90, n)
    >>> ea2 = np.random.randint(0, 90, n)
    >>> ea3 = np.random.randint(0, 90, n)
    >>> G = grainoris(ea=np.vstack((ea1, ea2, ea3)).T)
    >>> G.compute_quats()
    >>> G.compute_avg()
    """
    __slots__ = ('__coords', 'ea', 'pea1', 'pea2', 'pea3', 'q', 'gid', 's',
                 'qavg'
                 )

    def __init__(self, gid=None, s=None,
                 ea=np.array([0, 0, 0]),
                 pea1=[0.0, 0.0], pea2=[0.0, 0.0], pea3=[0.0, 0.0],
                 deg=False
                 ):
        """Initialise the instance."""
        self.gid, self.s = None, None
        if deg:
            self.ea = np.radians(ea)
        else:
            self.ea = ea
        self.pea1, self.pea2, self.pea2 = pea1, pea2, pea3
        self.q = None
        self.__coords = None

    def __repr__(self):
        """Return a string representation of this instance."""
        return f"grain orientation set. gid: {self.gid} @s: {self.s}"

    def compute_avg(self):
        """Return the ute avg."""
        self.qavg = Quat.calcAverageOri(self.q)

    def compute_quats(self):
        """Return the ute quats."""
        self.q = np.array([Quat.fromEulerAngles(ea1, ea2, ea3)
                           for ea1, ea2, ea3 in self.ea*np.pi/180])

    @property
    def perturb(self, pea1, pea2, pea3):
        """Perturb."""
        return (tuple(self.pea1(pea1)),
                tuple(self.pea2(pea2)),
                tuple(self.pea3(pea3)))

    @perturb.setter
    def eapert(self, pertea):
        """Eapert."""
        # VALIDATIONS NEEDED
        self.pea1 = pertea[0]
        self.pea2 = pertea[1]
        self.pea3 = pertea[2]

    @property
    def p_ea1_deg(self):
        """P ea1 deg."""
        return self.pea1

    @p_ea1_deg.setter
    def p_ea1_deg(self, pea1):
        """P ea1 deg."""
        self.pea1 = pea1

    @property
    def p_ea2_deg(self):
        """P ea2 deg."""
        return self.pea1

    @p_ea2_deg.setter
    def p_ea2_deg(self, pea1):
        """P ea2 deg."""
        self.pea1 = pea1

    @property
    def p_ea3_deg(self):
        """P ea3 deg."""
        return self.pea1

    @p_ea3_deg.setter
    def p_ea3_deg(self, pea1):
        """P ea3 deg."""
        self.pea1 = pea1

    @property
    def coords(self):
        """Coords."""
        return self.__coords

    @coords.setter
    def coords(self, value):
        """Coords."""
        self.__coords = value

    @coords.deleter
    def coords(self):
        """Coords."""
        del self.__coords
