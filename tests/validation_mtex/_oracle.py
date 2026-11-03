"""Independent reference implementations for the validation tests.

The functions here use only numpy and scipy and share no code with UPXO, so a test that compares
a UPXO result with one of them cannot pass because both sides carry the same defect.

Conventions (the same as mtex):
    R   crystal -> sample rotation matrix, R = Rz(phi1) Rx(Phi) Rz(phi2) for Bunge angles.
    g   sample -> crystal, g = R.T. DefDAP-stored quaternions are the quaternions of g.
The 24 proper cubic operators act on the crystal side: R -> R @ S.
"""
from itertools import permutations, product

import numpy as np
from scipy.spatial.transform import Rotation


def cubic_ops() -> np.ndarray:
    ops = []
    for p in permutations(range(3)):
        P = np.eye(3)[list(p)]
        for s in product([-1, 1], repeat=3):
            S = P * np.array(s)
            if abs(np.linalg.det(S) - 1) < 1e-9:
                ops.append(S)
    return np.array(ops)


OPS = cubic_ops()
_PAIR_PRODUCTS = OPS[:, None] @ OPS[None, :]            # (24, 24, 3, 3): S_a @ S_b


def bunge_R(phi1, Phi, phi2, degrees=True):
    """Crystal -> sample matrix. Accepts scalars or arrays of equal shape."""
    a = np.atleast_1d(np.asarray(phi1, float)); b = np.atleast_1d(np.asarray(Phi, float))
    c = np.atleast_1d(np.asarray(phi2, float))
    if degrees:
        a, b, c = np.radians(a), np.radians(b), np.radians(c)
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    R = np.empty(a.shape + (3, 3))
    R[..., 0, 0] = ca * cc - sa * cb * sc
    R[..., 0, 1] = -ca * sc - sa * cb * cc
    R[..., 0, 2] = sa * sb
    R[..., 1, 0] = sa * cc + ca * cb * sc
    R[..., 1, 1] = -sa * sc + ca * cb * cc
    R[..., 1, 2] = -ca * sb
    R[..., 2, 0] = sb * sc
    R[..., 2, 1] = sb * cc
    R[..., 2, 2] = cb
    return R


def euler_from_R(R, degrees=True):
    """Bunge angles of crystal -> sample matrices (intrinsic ZXZ)."""
    e = Rotation.from_matrix(np.asarray(R)).as_euler("ZXZ", degrees=degrees)
    return np.atleast_2d(e)


def quat_wxyz_from_matrix(M):
    """Unit quaternion [w, x, y, z] of a rotation matrix (scipy gives [x, y, z, w])."""
    q = Rotation.from_matrix(np.asarray(M)).as_quat()
    return np.atleast_2d(q)[:, [3, 0, 1, 2]]


def angle_of(M):
    """Rotation angle in degrees of matrices of shape (..., 3, 3)."""
    tr = np.trace(M, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def disorientation_R(Ra, Rb):
    """Cubic disorientation in degrees of N pairs of crystal -> sample matrices.

    The relative rotation in the crystal frame is Ra.T @ Rb; the operators act on both of its
    sides: min over S_a, S_b of the angle of S_a @ (Ra.T @ Rb) @ S_b.
    """
    Ra = np.asarray(Ra, float); Rb = np.asarray(Rb, float)
    if Ra.ndim == 2:
        Ra, Rb = Ra[None], Rb[None]
    dR = np.einsum("nji,njk->nik", Ra, Rb)
    tr = np.einsum("nij,abji->nab", dR, _PAIR_PRODUCTS)
    best = tr.reshape(len(dR), -1).max(axis=1)
    return np.degrees(np.arccos(np.clip((best - 1) / 2, -1, 1)))


def orientation_difference(Ra, Rb):
    """Largest element difference between two crystal -> sample matrices after the best cubic
    relabelling of the crystal axes: min over S of max |Ra - Rb @ S|.

    Zero (to rounding) when the two matrices are the same orientation of a cubic crystal. Unlike
    an angle, it has no square-root amplification of rounding near zero.
    """
    Ra = np.asarray(Ra, float); Rb = np.asarray(Rb, float)
    return float(np.abs(Ra[None] - Rb[None] @ OPS).max(axis=(1, 2)).min())


def disorientation_R_sample_side(Ra, Rb):
    """The WRONG-side reduction (relative rotation in the sample frame, Ra @ Rb.T).

    Kept so that tests can show their pairs discriminate: this differs from the correct
    disorientation for a large share of pairs.
    """
    Ra = np.asarray(Ra); Rb = np.asarray(Rb)
    if Ra.ndim == 2:
        Ra, Rb = Ra[None], Rb[None]
    dR = np.einsum("nij,nkj->nik", Ra, Rb)
    tr = np.einsum("nij,abji->nab", dR, _PAIR_PRODUCTS)
    best = tr.reshape(len(dR), -1).max(axis=1)
    return np.degrees(np.arccos(np.clip((best - 1) / 2, -1, 1)))


def disorientation_axis_family(Ra, Rb):
    """Disorientation angle and the sorted absolute components of the rotation axis.

    The axis of the reduced rotation is returned as a unit vector with its components in
    absolute value, sorted descending: a canonical label for the family of cubic-equivalent axes
    (<111> gives (.577, .577, .577), <110> gives (.707, .707, 0), <210> gives (.894, .447, 0)).
    """
    dR = np.asarray(Ra).T @ np.asarray(Rb)
    best_a, best_M = 1e9, None
    for S in OPS:
        for S2 in OPS:
            M = S @ dR @ S2
            a = angle_of(M)
            if a < best_a - 1e-10:
                best_a, best_M = a, M
    if best_a < 1e-8:
        return 0.0, np.zeros(3)
    w, v = np.linalg.eig(best_M)
    k = int(np.argmin(np.abs(w - 1)))
    ax = np.real(v[:, k]); ax /= np.linalg.norm(ax)
    return float(best_a), np.sort(np.abs(ax))[::-1]


# ---- twins ----------------------------------------------------------------------------------

FOUR_111 = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], float) / np.sqrt(3)


def twin_operator(n):
    """180 degree rotation about the unit plane normal n (the twin operation, crystal frame)."""
    n = np.asarray(n, float)
    return 2 * np.outer(n, n) - np.eye(3)


# ---- Kurdjumov-Sachs ------------------------------------------------------------------------

def ks_matrix():
    """Rotation g with (111)_g || (011)_a and [1-10]_g || [-1-11]_a, from the vector pairs.

    Built with scipy's alignment of two vector pairs, not by assembling frames, so it does not
    repeat the construction in the library. g maps austenite crystal coordinates to martensite
    crystal coordinates.
    """
    A = np.array([[1, 1, 1], [1, -1, 0]], float)
    B = np.array([[0, 1, 1], [-1, -1, 1]], float)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    rot, _ = Rotation.align_vectors(B, A)
    assert np.abs(rot.apply(A) - B).max() < 1e-12, "the two KS vector pairs are not mutually consistent"
    return rot.as_matrix()


def ks_variants():
    """The 24 variants g_i = g_KS @ S_i (operators relabel the parent axes: right side)."""
    return ks_matrix() @ OPS


ALPHA_110 = np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]], float) / np.sqrt(2)


def packet_index(g):
    """Index (0-3) of the {111}_g plane that g maps onto a {110}_a plane; asserts it is unique."""
    hits = []
    for k, n in enumerate(FOUR_111):
        v = g @ n
        if np.max(np.abs(ALPHA_110 @ v)) > 1 - 1e-9:
            hits.append(k)
    assert len(hits) == 1, f"expected one parallel close-packed plane, found {hits}"
    return hits[0]


def bain_index(g):
    """Austenite cube axis (0-2) whose image under g lies nearest an martensite cube axis."""
    best = []
    for k in range(3):
        e = np.zeros(3); e[k] = 1
        best.append(np.max(np.abs(g @ e)))
    return int(np.argmax(best))
