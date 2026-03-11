"""
definitions and classes to deal with orientations and misorientations
-------------------------
def eamo -- Major work done. Validations pending.
def qmo -- Major work done. Validations pending.
-------------------------
"""

import numpy as np
from defdap.quat import Quat


def eamo(e1, e2list, symm):
    """
    Euler Angle Mis-Orientation
    ---------------------------
    ea1, ea2: Bunge's Euler angles in degrees
    symm: Symmetry: 'cubic', OTHER OPRION CHECK RFOM DEFDAP MANUAL

    EXAMPLE CALL
    ------------
    e1 = np.array([0, 0, 0], dtype=float)
    e2list = np.array([[0, 0, 0],
                       [45, 0, 0],
                       [45, 45, 0]], dtype=float)
    symm = 'cubic'
    eamo(e1, e2list, symm)

    fcc_w = np.array([0, 0, 0], dtype=float)  # Cube
    fcc_g = np.array([0, 45, 0], dtype=float)  # Goss
    fcc_b = np.array([35, 45, 0], dtype=float)  # Brass
    fcc_s = np.array([59, 37, 63], dtype=float)  # S
    fcc_cu = np.array([90, 35, 45], dtype=float)  # Copper
    fcc_rc = np.array([0, 0, 45], dtype=float)  # Rotated Cube
    fcc_rcu = np.array([0, 35, 45], dtype=float)  # Rotated Copper
    fcc_gt = np.array([90, 25, 45], dtype=float)  # Goss twin
    fcc_cut = np.array([90, 74, 45], dtype=float)  # Copper twin

    e1 = fcc_g
    e2list = np.vstack((fcc_w, fcc_g, fcc_b, fcc_s, fcc_cu))
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
    Quaternion Mis-Orientation
    --------------------------
    q1, q2: Quaternions
    symm: Symmetry: 'cubic', OTHER OPRION CHECK RFOM DEFDAP MANUAL

    EXAMPLE CALL
    ------------
    qmo(q1, q2, 'cubic')
    """
    # VALIDATION FOR q1 AND q2
    # VALIDATION TO Also deal with q1 ans q2 which are not defdap types
    return 2*np.arccos([q1.misOri(q2, symm) for q2 in q2list])*57.295779


def ipf(vector, symm):
    pass


class grainoris():
    """
    __coords: Coordinate locations defining the grain
    ea: Set of euler angles in radians
    pea1 (User Input): [min, max] of ea perturbtaion in degrees
    pea1 (Attribute): [min, max] of ea perturbtaion in radians

    EXAMPLE
    -------
    from upxo.xtalphy.orientation import grainoris

    n = 4
    ea1 = np.random.random_integers(0, 90, n)
    ea2 = np.random.random_integers(0, 90, n)
    ea3 = np.random.random_integers(0, 90, n)
    G = grainoris(ea = np.vstack((ea1, ea2, ea3)).T)
    G.compute_quats()
    G.compute_avg()

    @ DEVELOPER
    -----------
    EA = [[0*np.pi/180, 0, 0], [0*np.pi/180, 0, 0]]
    Quat.fromEulerAngles(EA)
    q1 = Quat.fromEulerAngles(0*np.pi/180, 0, 0)
    q2 = Quat.fromEulerAngles(45*np.pi/180, 0, 0)
    mori12 = round(q1.misOri(q2, 'cubic'), 5)

    round(2*np.arccos(mori12)*180/np.pi, 10)
    """
    __slots__ = ('__coords', 'ea', 'pea1', 'pea2', 'pea3', 'q', 'gid', 's',
                 'qavg'
                 )

    def __init__(self, gid=None, s=None,
                 ea=np.array([0, 0, 0]),
                 pea1=[0.0, 0.0], pea2=[0.0, 0.0], pea3=[0.0, 0.0],
                 deg=False
                 ):
        self.gid, self.s = None, None
        if deg:
            self.ea = np.radians(ea)
        else:
            self.ea = ea
        self.pea1, self.pea2, self.pea2 = pea1, pea2, pea3
        self.q = None
        self.__coords = None

    def __repr__(self):
        return f"grain orientation set. gid: {self.gid} @s: {self.s}"

    def compute_avg(self):
        self.qavg = Quat.calcAverageOri(self.q)

    def compute_quats(self):
        self.q = np.array([Quat.fromEulerAngles(ea1, ea2, ea3)
                           for ea1, ea2, ea3 in self.ea*np.pi/180])

    @property
    def perturb(self, pea1, pea2, pea3):
        return (tuple(self.pea1(pea1)),
                tuple(self.pea2(pea2)),
                tuple(self.pea3(pea3)))

    @perturb.setter
    def eapert(self, pertea):
        # VALIDATIONS NEEDED
        self.pea1 = pertea[0]
        self.pea2 = pertea[1]
        self.pea3 = pertea[2]

    @property
    def p_ea1_deg(self):
        return self.pea1

    @p_ea1_deg.setter
    def p_ea1_deg(self, pea1):
        self.pea1 = pea1

    @property
    def p_ea2_deg(self):
        return self.pea1

    @p_ea2_deg.setter
    def p_ea2_deg(self, pea1):
        self.pea1 = pea1

    @property
    def p_ea3_deg(self):
        return self.pea1

    @p_ea3_deg.setter
    def p_ea3_deg(self, pea1):
        self.pea1 = pea1

    @property
    def coords(self):
        return self.__coords

    @coords.setter
    def coords(self, value):
        self.__coords = value

    @coords.deleter
    def coords(self):
        del self.__coords
