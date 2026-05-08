"""
crystal_orientation.py
======================
Stand-alone module for crystallographic orientation mathematics.

All definitions are extracted (by reference, not moved) from
``upxo.pxtal.mcgs3_temporal_slice`` so that the original class is
unchanged.  Callers should import from here rather than reaching into
the temporal-slice module.

Public API
----------
Constants
    FCC_TEXTURE_COMPONENTS   – dict {name: (phi1, Phi, phi2)} in degrees (Bunge)
    CUBIC_SYMM_OPS_CACHE     – module-level cache for 24 cubic SO(3) operators

Primitive rotation helpers  (pure functions, no class needed)
    Rz(a)                    – 3x3 rotation about Z axis (angle in radians)
    Rx(a)                    – 3x3 rotation about X axis (angle in radians)
    euler_bunge_to_matrix(phi1, Phi, phi2, degrees=True)   – single Bunge ZXZ R
    euler_bunge_to_matrix_batch(phi1, Phi, phi2, ...)      – vectorized version
    matrix_to_euler_bunge(R, degrees=True)                 – R -> (phi1, Phi, phi2)
    axis_angle_to_R(axis, angle_rad)                       – Rodrigues formula
    normalize_euler_bunge(ea, degrees=True)                – canonical Euler range
    proj_to_so3(R)                                         – SVD-project to SO(3)
    unique_rotations(rotations, tol=1e-8)                  – deduplicate R matrices

Symmetry helpers
    cubic_symmetry_operators()   – list of 24 proper cubic rotation matrices
    get_cubic_ops_np()           – (24,3,3) ndarray (cached)
    fcc_symmetrise_ori(bea)      – symmetric equivalents of one orientation
    rand_unit_vector(rng)        – uniform S2 sample
    rand_uniform_SO3(rng)        – uniform SO(3) sample

Misorientation
    cubic_rotation_angle(R)      – disorientation angle from single dR
    cubic_rotation_angle_batch(rstack) – vectorized version
    cubic_rotation_axis(R, angle) – axis for a single dR
    cubic_rotation_axis_batch(rstack, angles) – vectorized version
    cubic_misorientation(EA1, EA2, ...) – fast vectorized cubic misorientation
    cubic_misorientation_scalar(EA1, EA2, ...) – scalar (loop) reference version

High-level utilities
    grain_boundary_misorientation_distribution(euler_array, pairs, ...)
        Compute the GBMD (misorientation angle distribution) for a list of
        grain-boundary pairs given per-grain Bunge Euler angles.
    gbmd_from_lfi(lfi, euler_array, connectivity=4, ...)
        Convenience wrapper: auto-detects grain boundary pairs from a
        label-field image and returns the GBMD.
    ipf_color(euler_deg, sample_direction=[0,0,1])  – IPF colour (R,G,B) tuple

Special orientation relationships
    get_ks_rotations()           – 24 Kurdjumov-Sachs BCC variant matrices

Usage example
-------------
>>> from upxo.xtalphy.crystal_orientation import (
...     cubic_misorientation,
...     grain_boundary_misorientation_distribution,
...     gbmd_from_lfi,
...     FCC_TEXTURE_COMPONENTS,
... )
>>> angle, axis, top3 = cubic_misorientation([0, 0, 0], [45, 0, 0])
>>> print(f"Misorientation: {angle:.2f} deg, axis: {axis}")
"""

from __future__ import annotations

import numpy as np
from itertools import permutations, product
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FCC_TEXTURE_COMPONENTS: dict[str, tuple[float, float, float]] = {
    "copper":       (90.0,  35.0,  45.0),   # {112}<111>  Rolling
    "brass":        (35.0,  45.0,   0.0),   # {110}<112>  Rolling
    "s":            (59.0,  37.0,  63.0),   # {123}<634>  Rolling
    "goss":         (90.0,  90.0,  45.0),   # {110}<001>  Rolling
    "cube":          (0.0,   0.0,   0.0),   # {001}<100>  Annealing/RX
    "rotated_cube": (45.0,   0.0,   0.0),   # {001}<110>  Annealing/RX
    "P":            (90.0,  45.0,   0.0),   # {011}<122>  Annealing/RX
    "A1":           (35.0,  45.0,  90.0),   # {111}<110>  Shear
    "A2":           (55.0,  90.0,  45.0),   # {111}<112>  Shear
    "B":            (45.0,  90.0,  45.0),   # {112}<110>  Shear
    "C":             (0.0,  90.0,  45.0),   # {001}<110>  Shear
    "Q":            (35.0,  55.0,  45.0),   # {013}<231>  Minor
    "D":            (59.0,  37.0,  26.0),   # {4411}<1118> Minor
}

# Module-level cache for the cubic symmetry operators (numpy array)
CUBIC_SYMM_OPS_CACHE: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Primitive elementary rotation matrices
# ---------------------------------------------------------------------------

def Rz(a: float) -> np.ndarray:
    """3x3 rotation matrix about Z axis. *a* in radians."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, -s, 0.],
                     [ s,  c, 0.],
                     [0., 0., 1.]])


def Rx(a: float) -> np.ndarray:
    """3x3 rotation matrix about X axis. *a* in radians."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.,  0., 0.],
                     [0.,  c, -s],
                     [0.,  s,  c]])


# ---------------------------------------------------------------------------
# Euler ↔ rotation matrix
# ---------------------------------------------------------------------------

def euler_bunge_to_matrix(phi1: float, Phi: float, phi2: float,
                           degrees: bool = True) -> np.ndarray:
    """
    Bunge ZXZ convention: R = Rz(phi1) · Rx(Phi) · Rz(phi2).

    Parameters
    ----------
    phi1, Phi, phi2 : float
        Bunge Euler angles.
    degrees : bool
        If True, inputs are in degrees. Default True.

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    if degrees:
        phi1, Phi, phi2 = np.deg2rad([phi1, Phi, phi2])
    return Rz(phi1) @ Rx(Phi) @ Rz(phi2)


def euler_bunge_to_matrix_batch(phi1: np.ndarray,
                                 Phi: np.ndarray,
                                 phi2: np.ndarray,
                                 degrees: bool = True,
                                 dtype=np.float64) -> np.ndarray:
    """
    Vectorized Bunge ZXZ rotation matrix from arrays of Euler angles.

    Parameters
    ----------
    phi1, Phi, phi2 : array-like, shape (N,)
        Bunge Euler angles.
    degrees : bool
        Default True.
    dtype : numpy dtype
        Output dtype.

    Returns
    -------
    R : ndarray, shape (N, 3, 3)
    """
    phi1 = np.asarray(phi1, dtype=dtype)
    Phi  = np.asarray(Phi,  dtype=dtype)
    phi2 = np.asarray(phi2, dtype=dtype)
    if degrees:
        phi1, Phi, phi2 = np.deg2rad(phi1), np.deg2rad(Phi), np.deg2rad(phi2)

    c1, s1 = np.cos(phi1), np.sin(phi1)
    c,  s  = np.cos(Phi),  np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    N = phi1.shape[0]
    R = np.empty((N, 3, 3), dtype=dtype)
    R[:, 0, 0] =  c1*c2 - s1*s2*c
    R[:, 0, 1] = -c1*s2 - s1*c2*c
    R[:, 0, 2] =  s1*s
    R[:, 1, 0] =  s1*c2 + c1*s2*c
    R[:, 1, 1] = -s1*s2 + c1*c2*c
    R[:, 1, 2] = -c1*s
    R[:, 2, 0] =  s2*s
    R[:, 2, 1] =  c2*s
    R[:, 2, 2] =  c
    return R


def matrix_to_euler_bunge(R: np.ndarray,
                           degrees: bool = True) -> tuple[float, float, float]:
    """
    Convert a (3×3) rotation matrix to Bunge ZXZ Euler angles.

    Returns
    -------
    (phi1, Phi, phi2) : tuple of float
        In degrees (default) or radians.
    """
    R = np.asarray(R, dtype=float)
    c = np.clip(R[2, 2], -1.0, 1.0)
    Phi_rad = np.arccos(c)
    if abs(Phi_rad) < 1e-12:
        phi1_rad = np.arctan2(R[1, 0], R[0, 0])
        phi2_rad = 0.0
    elif abs(Phi_rad - np.pi) < 1e-12:
        phi1_rad = np.arctan2(R[1, 2], R[0, 2])
        phi2_rad = 0.0
    else:
        phi1_rad = np.arctan2(R[2, 0], -R[2, 1])
        phi2_rad = np.arctan2(R[0, 2],  R[1, 2])
    if degrees:
        return (float(np.degrees(phi1_rad) % 360.0),
                float(np.degrees(Phi_rad)),
                float(np.degrees(phi2_rad) % 360.0))
    return (float(phi1_rad % (2*np.pi)), float(Phi_rad), float(phi2_rad % (2*np.pi)))


def normalize_euler_bunge(ea: np.ndarray,
                           degrees: bool = True,
                           eps: float = 1e-6) -> np.ndarray:
    """
    Normalize Bunge ZXZ Euler angles to canonical ranges:
      phi1, phi2 in [0, 360°)   Phi in [0, 180°]

    Parameters
    ----------
    ea : array-like, shape (..., 3) or (3,)
    degrees : bool
    eps : float
        Snap tolerance near 0 and 180.

    Returns
    -------
    Normalized array, same shape as input.
    """
    A = np.asarray(ea, dtype=float)
    A2 = np.atleast_2d(A)
    phi1, Phi, phi2 = A2[:, 0].copy(), A2[:, 1].copy(), A2[:, 2].copy()

    two_pi, pi = (360.0, 180.0) if degrees else (2*np.pi, np.pi)

    phi1[:] = np.mod(phi1, two_pi)
    phi2[:] = np.mod(phi2, two_pi)
    Phi[:]  = ((Phi + pi) % (2*pi)) - pi

    neg = Phi < 0.0
    if np.any(neg):
        Phi[neg]  = -Phi[neg]
        phi1[neg] = np.mod(phi1[neg] + pi, two_pi)
        phi2[neg] = np.mod(phi2[neg] + pi, two_pi)

    over = Phi > pi
    if np.any(over):
        Phi[over]  = 2*pi - Phi[over]
        phi1[over] = np.mod(phi1[over] + pi, two_pi)
        phi2[over] = np.mod(phi2[over] + pi, two_pi)

    if eps is not None:
        Phi[np.abs(Phi) < eps]        = 0.0
        Phi[np.abs(Phi - pi) < eps]   = pi

    out = np.stack([phi1, Phi, phi2], axis=-1)
    return out if A2.shape[0] > 1 else out[0]


def axis_angle_to_R(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rodrigues' rotation formula: axis-angle → 3×3 rotation matrix.

    Parameters
    ----------
    axis : array-like, shape (3,)
    angle_rad : float

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    ax = np.asarray(axis, dtype=float)
    n  = np.linalg.norm(ax)
    if n < 1e-15 or abs(angle_rad) < 1e-15:
        return np.eye(3)
    u = ax / n
    ux, uy, uz = u
    K = np.array([[ 0,  -uz,  uy],
                  [ uz,  0,  -ux],
                  [-uy,  ux,   0]], dtype=float)
    return np.eye(3) + np.sin(angle_rad)*K + (1.0 - np.cos(angle_rad))*(K @ K)


def proj_to_so3(R: np.ndarray) -> np.ndarray:
    """Project *R* to the nearest proper rotation matrix via SVD."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def unique_rotations(rotations: list[np.ndarray],
                     tol: float = 1e-8) -> list[np.ndarray]:
    """Return deduplicated list of rotation matrices (Frobenius norm criterion)."""
    uniq: list[np.ndarray] = []
    for R in rotations:
        if not any(np.linalg.norm(R - Q, ord='fro') < tol for Q in uniq):
            uniq.append(R)
    return uniq


# ---------------------------------------------------------------------------
# Symmetry operators
# ---------------------------------------------------------------------------

def cubic_symmetry_operators() -> list[np.ndarray]:
    """
    Return the 24 proper rotation matrices for cubic m-3m symmetry.

    Uses signed-permutation matrices with determinant +1.

    Returns
    -------
    list of ndarray, each shape (3, 3)
    """
    ops: list[np.ndarray] = []
    for p in permutations(range(3)):
        P = np.eye(3)[list(p)]
        for signs in product([-1, 1], repeat=3):
            S = P * np.array(signs)[None, :]
            if round(np.linalg.det(S)) == 1:
                ops.append(S.astype(float))
    return unique_rotations(ops)


def get_cubic_ops_np() -> np.ndarray:
    """
    Return the 24 cubic symmetry operators as a cached (24, 3, 3) ndarray.
    """
    global CUBIC_SYMM_OPS_CACHE
    if CUBIC_SYMM_OPS_CACHE is None:
        CUBIC_SYMM_OPS_CACHE = np.asarray(cubic_symmetry_operators(),
                                           dtype=float)
    return CUBIC_SYMM_OPS_CACHE


def fcc_symmetrise_ori(bea: tuple[float, float, float],
                        dtype=np.float32) -> np.ndarray:
    """
    Generate all symmetrically equivalent Bunge Euler angle triplets for
    one FCC orientation.

    Parameters
    ----------
    bea : (phi1, Phi, phi2) in degrees
    dtype : numpy dtype for output

    Returns
    -------
    ndarray, shape (M, 3), M ≤ 24
    """
    g       = euler_bunge_to_matrix(*bea, degrees=True)
    sym_ops = cubic_symmetry_operators()
    eq_mats = [proj_to_so3(S @ g) for S in sym_ops]
    eq_mats = unique_rotations(eq_mats, tol=1e-8)
    return np.array([list(matrix_to_euler_bunge(R, degrees=True))
                     for R in eq_mats], dtype=dtype)


# ---------------------------------------------------------------------------
# Random orientation sampling
# ---------------------------------------------------------------------------

def rand_unit_vector(rng) -> np.ndarray:
    """Uniform random unit vector on S². *rng* must expose ``.gauss(mu, s)``."""
    v = np.array([rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n >= 1e-15 else np.array([1., 0., 0.])


def rand_uniform_SO3(rng) -> np.ndarray:
    """
    Draw a uniform random rotation from SO(3) using the
    Shoemake (1992) method.

    Parameters
    ----------
    rng : object exposing ``.random()`` (e.g. ``random.Random()``)

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q1 = np.sqrt(1 - u1) * np.sin(2*np.pi*u2)
    q2 = np.sqrt(1 - u1) * np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1)     * np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1)     * np.cos(2*np.pi*u3)
    x, y, z, w = q1, q2, q3, q4
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Rotation angle / axis  (scalar and batch)
# ---------------------------------------------------------------------------

def cubic_rotation_angle(R: np.ndarray) -> float:
    """Disorientation angle (radians) of a proper rotation matrix *R*."""
    x = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(x))


def cubic_rotation_angle_batch(rstack: np.ndarray) -> np.ndarray:
    """
    Vectorized rotation angle for a stack of matrices.

    Parameters
    ----------
    rstack : ndarray, shape (..., 3, 3)

    Returns
    -------
    angles : ndarray, same leading shape as *rstack* minus last two dims
    """
    x = np.clip((np.trace(rstack, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(x)


def cubic_rotation_axis(R: np.ndarray, angle: float) -> np.ndarray:
    """
    Return unit rotation axis for a proper rotation *R* given its angle (rad).
    Falls back to [1,0,0] for near-zero angles.
    """
    if angle < 1e-8:
        return np.array([1., 0., 0.])
    A    = (R - R.T) / (2.0 * np.sin(angle))
    axis = np.array([A[2, 1], A[0, 2], A[1, 0]])
    n    = np.linalg.norm(axis)
    return axis / n if n > 0 else np.array([1., 0., 0.])


def cubic_rotation_axis_batch(rstack: np.ndarray,
                               angles: np.ndarray) -> np.ndarray:
    """
    Vectorized rotation axis for a stack of rotation matrices.

    Parameters
    ----------
    rstack : ndarray, shape (N, 3, 3)
    angles : ndarray, shape (N,) in radians

    Returns
    -------
    axes : ndarray, shape (N, 3)
    """
    N    = rstack.shape[0]
    axes = np.empty((N, 3), dtype=rstack.dtype)
    small = np.abs(angles) < 1e-8
    axes[small] = np.array([1., 0., 0.])
    if np.any(~small):
        R_big   = rstack[~small]
        ang_big = angles[~small]
        denom   = 2.0 * np.sin(ang_big)
        A       = (R_big - np.transpose(R_big, (0, 2, 1))) / denom[:, None, None]
        ax_big  = np.stack([A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]], axis=1)
        norms   = np.linalg.norm(ax_big, axis=1, keepdims=True)
        norms   = np.where(norms > 0, norms, 1.0)
        axes[~small] = ax_big / norms
    return axes


# ---------------------------------------------------------------------------
# Misorientation  (fast vectorized + scalar reference)
# ---------------------------------------------------------------------------

def cubic_misorientation(EA1, EA2,
                          unique_tol_deg: float = 1e-4,
                          degrees: bool = True
                          ) -> tuple[float, np.ndarray, list[float]]:
    """
    Vectorized cubic (m-3m) misorientation between two orientations.

    Parameters
    ----------
    EA1, EA2 : array-like
        Each can be a Bunge Euler triplet (phi1, Phi, phi2) **or** a
        (3×3) rotation matrix.
    unique_tol_deg : float
        Tolerance (degrees) for deduplicating the top-3 angles.
    degrees : bool
        Applies to Euler inputs only. Default True.

    Returns
    -------
    angle_deg_min : float
        Minimum (fundamental) misorientation angle in degrees.
    axis_min : ndarray, shape (3,)
        Corresponding rotation axis (sample frame).
    top3_angles_deg : list[float]
        Up to 3 smallest unique misorientation angles (degrees), ascending.
    """
    def _as_R(x):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (3, 3):
            return arr
        phi1, Phi, phi2 = float(arr[0]), float(arr[1]), float(arr[2])
        return euler_bunge_to_matrix(phi1, Phi, phi2, degrees=degrees)

    gA = _as_R(EA1)
    gB = _as_R(EA2)
    S  = get_cubic_ops_np()          # (24, 3, 3)

    RA  = S @ gA                     # (24, 3, 3)
    RB  = S @ gB
    RtA = np.transpose(RA, (0, 2, 1))

    dR  = RB[:, None, :, :] @ RtA[None, :, :, :]   # (24, 24, 3, 3)
    tr  = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]
    x   = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    ang = np.arccos(x)               # (24, 24) radians

    idx_flat       = int(np.argmin(ang))
    l_min, k_min   = divmod(idx_flat, ang.shape[1])
    best_angle_rad = float(ang[l_min, k_min])
    dR_min         = dR[l_min, k_min]

    if best_angle_rad < 1e-12:
        axis_min = np.array([1., 0., 0.])
    else:
        sin_a    = np.sin(best_angle_rad)
        A        = (dR_min - dR_min.T) / (2.0 * sin_a)
        axis_min = np.array([A[2, 1], A[0, 2], A[1, 0]])
        n        = np.linalg.norm(axis_min)
        axis_min = axis_min / n if n > 0 else np.array([1., 0., 0.])

    angles_deg_flat = np.degrees(np.sort(ang, axis=None))
    top3: list[float] = []
    for a in angles_deg_flat:
        if not top3 or abs(a - top3[-1]) > unique_tol_deg:
            top3.append(float(a))
        if len(top3) >= 3:
            break

    return float(np.degrees(best_angle_rad)), axis_min, top3


def cubic_misorientation_scalar(EA1, EA2,
                                 unique_tol_deg: float = 1e-4,
                                 degrees: bool = True
                                 ) -> tuple[float, np.ndarray, list[float]]:
    """
    Scalar (double-loop) reference implementation of cubic misorientation.
    Slower but more transparent.  Same return signature as
    ``cubic_misorientation``.
    """
    def _as_R(x):
        arr = np.asarray(x, dtype=float)
        if arr.shape == (3, 3):
            return arr
        return euler_bunge_to_matrix(float(arr[0]), float(arr[1]),
                                     float(arr[2]), degrees=degrees)

    gA      = _as_R(EA1)
    gB      = _as_R(EA2)
    sym_ops = cubic_symmetry_operators()

    angles: list[float] = []
    best_angle = np.inf
    best_axis  = np.array([1., 0., 0.], dtype=float)

    for SA in sym_ops:
        RA = SA @ gA
        for SB in sym_ops:
            dR  = (SB @ gB) @ RA.T
            ang = cubic_rotation_angle(dR)
            angles.append(ang)
            if ang < best_angle:
                best_angle = ang
                best_axis  = cubic_rotation_axis(dR, ang)

    angles_deg = np.degrees(np.sort(angles))
    top3: list[float] = []
    for a in angles_deg:
        if not top3 or abs(a - top3[-1]) > unique_tol_deg:
            top3.append(float(a))
        if len(top3) >= 3:
            break

    return float(np.degrees(best_angle)), best_axis, top3


# ---------------------------------------------------------------------------
# IPF colour
# ---------------------------------------------------------------------------

def ipf_color(euler_deg, sample_direction=(0., 0., 1.)) -> tuple[float, float, float]:
    """
    Approximate IPF colour for a single cubic orientation.

    Parameters
    ----------
    euler_deg : array-like, shape (3,)
        Bunge Euler angles in degrees.
    sample_direction : array-like, shape (3,)
        Sample reference direction. Default [001].

    Returns
    -------
    (r, g, b) : tuple of float in [0, 1]
    """
    sd = np.asarray(sample_direction, dtype=float)
    R  = euler_bunge_to_matrix(*euler_deg, degrees=True)
    v  = np.abs(R.T @ sd)
    v_max = np.max(v)
    if v_max > 0:
        v /= v_max
    return tuple(v)


# ---------------------------------------------------------------------------
# Grain-boundary misorientation distribution
# ---------------------------------------------------------------------------

def grain_boundary_misorientation_distribution(
        euler_array: np.ndarray,
        pairs: np.ndarray,
        grain_ids: np.ndarray | None = None,
        degrees: bool = True,
        n_bins: int = 36,
        angle_range: tuple[float, float] = (0.0, 65.0),
) -> dict:
    """
    Compute the Grain Boundary Misorientation Distribution (GBMD) for a
    set of grain-boundary pairs.

    Parameters
    ----------
    euler_array : ndarray, shape (N_grains, 3)
        Bunge Euler angles (phi1, Phi, phi2) for each grain.
        If *grain_ids* is given, row *i* corresponds to ``grain_ids[i]``.
        Otherwise row *i* corresponds to grain ID *i*.
    pairs : ndarray, shape (N_pairs, 2), dtype int
        Each row [gid_a, gid_b] is one grain-boundary pair.
    grain_ids : ndarray, shape (N_grains,) or None
        Grain ID labels that index into *euler_array*.
        If None, grain IDs are assumed to be 0, 1, ..., N_grains-1.
    degrees : bool
        If True, *euler_array* is in degrees. Default True.
    n_bins : int
        Number of histogram bins over *angle_range*. Default 36.
    angle_range : (float, float)
        (min_deg, max_deg) for the histogram. Default (0, 65).

    Returns
    -------
    result : dict with keys
        ``'misorientation_angles'``  – ndarray of angle per pair (degrees)
        ``'misorientation_axes'``    – ndarray, shape (N_pairs, 3)
        ``'pairs'``                  – same as input *pairs*
        ``'hist_counts'``            – bin counts
        ``'hist_bin_edges'``         – bin edge values (degrees)
        ``'hist_bin_centers'``       – bin centre values (degrees)
        ``'mean_angle'``             – mean misorientation angle (degrees)
        ``'median_angle'``           – median misorientation angle (degrees)
        ``'std_angle'``              – std deviation of angles (degrees)
        ``'n_pairs'``                – total number of pairs processed

    Notes
    -----
    The computation uses the full 24×24 cubic symmetry exhaustive search
    (``cubic_misorientation``).  For very large datasets (> 10 000 pairs)
    this may be slow; consider sub-sampling or parallelising the loop.

    Examples
    --------
    >>> from upxo.xtalphy.crystal_orientation import (
    ...     grain_boundary_misorientation_distribution, gbmd_from_lfi)
    # With explicit pairs
    >>> ea = np.random.rand(100, 3) * [360, 180, 360]   # random orientations
    >>> pairs = np.array([[0, 1], [1, 2], [3, 4]])
    >>> result = grain_boundary_misorientation_distribution(ea, pairs)
    >>> import matplotlib.pyplot as plt
    >>> plt.bar(result['hist_bin_centers'], result['hist_counts'],
    ...         width=np.diff(result['hist_bin_edges'])[0])
    >>> plt.xlabel('Misorientation angle (°)'); plt.ylabel('Count')
    >>> plt.title('GBMD'); plt.show()
    """
    euler_array = np.asarray(euler_array, dtype=float)
    pairs       = np.asarray(pairs, dtype=int)

    # Build fast ID → row-index lookup
    if grain_ids is None:
        id_to_row = {i: i for i in range(len(euler_array))}
    else:
        grain_ids = np.asarray(grain_ids, dtype=int)
        id_to_row = {int(gid): i for i, gid in enumerate(grain_ids)}

    angles: list[float] = []
    axes:   list[np.ndarray] = []

    for gid_a, gid_b in pairs:
        row_a = id_to_row.get(int(gid_a))
        row_b = id_to_row.get(int(gid_b))
        if row_a is None or row_b is None:
            continue
        ea_a = euler_array[row_a]
        ea_b = euler_array[row_b]
        angle, axis, _ = cubic_misorientation(ea_a, ea_b, degrees=degrees)
        angles.append(angle)
        axes.append(axis)

    angles_arr = np.asarray(angles)
    axes_arr   = np.asarray(axes) if axes else np.empty((0, 3))

    counts, edges = np.histogram(angles_arr, bins=n_bins, range=angle_range)
    centers       = 0.5 * (edges[:-1] + edges[1:])

    return {
        'misorientation_angles':  angles_arr,
        'misorientation_axes':    axes_arr,
        'pairs':                  pairs,
        'hist_counts':            counts,
        'hist_bin_edges':         edges,
        'hist_bin_centers':       centers,
        'mean_angle':             float(np.mean(angles_arr)) if len(angles_arr) else float('nan'),
        'median_angle':           float(np.median(angles_arr)) if len(angles_arr) else float('nan'),
        'std_angle':              float(np.std(angles_arr)) if len(angles_arr) else float('nan'),
        'n_pairs':                len(angles_arr),
    }


def gbmd_from_lfi(lfi: np.ndarray,
                  euler_array: np.ndarray,
                  grain_ids: np.ndarray | None = None,
                  connectivity: int = 4,
                  degrees: bool = True,
                  n_bins: int = 36,
                  angle_range: tuple[float, float] = (0.0, 65.0),
                  ) -> dict:
    """
    Convenience wrapper: detect grain-boundary pairs from a label-field
    image *lfi* and compute the GBMD.

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx), dtype int
        Integer label field. Each unique positive integer is a grain ID.
        Pixels with value ≤ 0 are ignored (unindexed).
    euler_array : ndarray, shape (N_grains, 3)
        Bunge Euler angles for each grain (row-order matches *grain_ids*).
    grain_ids : ndarray or None
        Grain ID labels corresponding to rows of *euler_array*.
        If None, grain IDs are 0 … N_grains-1.
    connectivity : {4, 8}
        Pixel adjacency for boundary detection. Default 4.
    degrees : bool
        Default True.
    n_bins : int
        Default 36.
    angle_range : (float, float)
        Default (0, 65).

    Returns
    -------
    Same dict as ``grain_boundary_misorientation_distribution`` plus
    an extra key ``'gb_pairs'`` containing the deduplicated set of
    adjacent grain-ID pairs.

    Notes
    -----
    The pair-detection uses integer-shift neighbour comparison, which is
    O(ny·nx) and memory-efficient.
    """
    lfi = np.asarray(lfi, dtype=int)

    # --- detect all adjacent grain-boundary pairs via pixel shifts ----------
    pairs_set: set[tuple[int, int]] = set()

    def _add(a_arr, b_arr):
        mask = (a_arr > 0) & (b_arr > 0) & (a_arr != b_arr)
        for a, b in zip(a_arr[mask].ravel(), b_arr[mask].ravel()):
            pair = (int(min(a, b)), int(max(a, b)))
            pairs_set.add(pair)

    # horizontal neighbour
    _add(lfi[:, :-1], lfi[:, 1:])
    # vertical neighbour
    _add(lfi[:-1, :], lfi[1:, :])
    if connectivity == 8:
        # diagonals
        _add(lfi[:-1, :-1], lfi[1:, 1:])
        _add(lfi[:-1, 1:],  lfi[1:, :-1])

    pairs_arr = np.array(sorted(pairs_set), dtype=int) if pairs_set else np.empty((0, 2), dtype=int)

    result = grain_boundary_misorientation_distribution(
        euler_array, pairs_arr, grain_ids=grain_ids,
        degrees=degrees, n_bins=n_bins, angle_range=angle_range,
    )
    result['gb_pairs'] = pairs_arr
    return result


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quat_to_R_batch(q: np.ndarray) -> np.ndarray:
    """
    Convert a stack of unit quaternions to rotation matrices.

    Parameters
    ----------
    q : ndarray, shape (N, 4)
        Unit quaternions in [w, x, y, z] convention.

    Returns
    -------
    R : ndarray, shape (N, 3, 3)

    Examples
    --------
    from upxo.xtalphy.crystal_orientation import quat_to_R_batch
    import numpy as np
    q = np.array([[1., 0., 0., 0.]])   # identity
    quat_to_R_batch(q)                  # → [[[1,0,0],[0,1,0],[0,0,1]]]
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2*(y*y + z*z);  R[:, 0, 1] = 2*(x*y - z*w);  R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w);      R[:, 1, 1] = 1 - 2*(x*x + z*z);  R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w);      R[:, 2, 1] = 2*(y*z + x*w);      R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def grain_avg_quats(lfi: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-grain average quaternion from a pixel-wise quaternion map.

    Handles the cyclic ambiguity of Euler angles by averaging in quaternion
    space (enforcing positive-w hemisphere before accumulation).

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx), dtype int
        Integer grain label field. Positive values are grain IDs;
        values ≤ 0 are ignored.
    quat : ndarray, shape (ny, nx, 4)
        Per-pixel unit quaternions in [w, x, y, z] convention.

    Returns
    -------
    gids : ndarray, shape (N_grains,), dtype int
        Sorted array of unique positive grain IDs found in *lfi*.
    q_mean : ndarray, shape (N_grains, 4)
        Normalised mean quaternion for each grain (row order matches *gids*).

    Examples
    --------
    from upxo.xtalphy.crystal_orientation import grain_avg_quats
    gids, q_mean = grain_avg_quats(rg.lfi_ebsd, rg.quat_ebsd)
    """
    lfi  = np.asarray(lfi, dtype=int)
    quat = np.asarray(quat, dtype=np.float64)

    gids     = np.unique(lfi)
    gids     = gids[gids > 0]
    N_labels = int(gids.max()) + 1

    q_flat   = quat.reshape(-1, 4)
    lfi_flat = lfi.ravel()

    # Enforce positive-w hemisphere before averaging
    q_flat = np.where(q_flat[:, 0:1] < 0, -q_flat, q_flat)

    q_sum   = np.zeros((N_labels, 4), dtype=np.float64)
    q_count = np.zeros(N_labels, dtype=np.int64)
    valid   = lfi_flat > 0
    np.add.at(q_sum,   lfi_flat[valid], q_flat[valid])
    np.add.at(q_count, lfi_flat[valid], 1)

    q_mean = q_sum[gids] / q_count[gids, None]
    norms  = np.linalg.norm(q_mean, axis=1, keepdims=True)
    q_mean = q_mean / np.where(norms > 0, norms, 1.0)
    return gids, q_mean


def compute_mdf_from_quats(
        lfi: np.ndarray,
        quat: np.ndarray,
        neigh_gid: dict,
        n_bins: int = 65,
        angle_range: tuple[float, float] = (0.0, 65.0),
) -> dict:
    """
    Compute the grain-boundary Misorientation Distribution Function (MDF)
    from a pixel quaternion map and a neighbour-graph dict.

    Uses per-grain quaternion averaging (correct for cyclic Euler angles)
    and a fully vectorised 24×24 cubic symmetry exhaustive search.

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx), dtype int
        Integer grain label field.
    quat : ndarray, shape (ny, nx, 4)
        Per-pixel unit quaternions [w, x, y, z].
    neigh_gid : dict
        Mapping ``{grain_id: array_of_neighbour_ids}`` as produced by
        ``EBSDReader.characterise()`` or ``find_neighs2d()``.
    n_bins : int
        Number of histogram bins. Default 65 (1° per bin over 0–65°).
    angle_range : (float, float)
        Histogram range in degrees. Default (0, 65).

    Returns
    -------
    result : dict with keys
        ``'miso_deg'``          – ndarray (N_pairs,) of disorientation angles
        ``'pairs'``             – ndarray (N_pairs, 2) of grain-ID pairs used
        ``'hist_counts'``       – bin counts (int)
        ``'hist_density'``      – probability density (float, integrates to 1)
        ``'hist_bin_edges'``    – bin edges (degrees)
        ``'hist_bin_centers'``  – bin centres (degrees)
        ``'mean_angle'``        – mean disorientation (degrees)
        ``'std_angle'``         – std deviation (degrees)
        ``'n_pairs'``           – number of unique boundary pairs

    Examples
    --------
    from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats
    mdf = compute_mdf_from_quats(rg.lfi_ebsd, rg.quat_ebsd, rg.neigh_gid_ebsd)
    print(mdf['mean_angle'], '°')
    """
    # 1. Grain-average quaternions
    gids, q_mean = grain_avg_quats(lfi, quat)
    N_labels     = int(gids.max()) + 1
    gid_to_idx   = np.zeros(N_labels, dtype=int)
    gid_to_idx[gids] = np.arange(len(gids))

    # 2. Rotation matrices
    R_all = quat_to_R_batch(q_mean)   # (N_grains, 3, 3)

    # 3. Unique neighbour pairs
    pairs_set = set()
    for gA_id, neighbours in neigh_gid.items():
        for gB_id in np.asarray(neighbours).ravel():
            gB_id = int(gB_id)
            if gB_id > 0 and int(gA_id) != gB_id:
                pairs_set.add((min(int(gA_id), gB_id), max(int(gA_id), gB_id)))
    pairs = np.array(sorted(pairs_set), dtype=int)   # (P, 2)

    # 4. Vectorised disorientation (full 24×24 cubic symmetry)
    S  = get_cubic_ops_np()
    iA = gid_to_idx[pairs[:, 0]]
    iB = gid_to_idx[pairs[:, 1]]
    RA = R_all[iA]   # (P, 3, 3)
    RB = R_all[iB]

    SA = np.einsum('sij,pjk->psik', S, RA)   # (P, 24, 3, 3)
    SB = np.einsum('sij,pjk->psik', S, RB)
    dR = np.einsum('pmab,pncb->pmnac', SB, SA)   # (P, 24, 24, 3, 3)

    tr      = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]
    cos_ang = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    miso_deg = np.degrees(np.arccos(cos_ang).min(axis=(1, 2)))   # (P,)

    # 5. Histogram
    counts, edges = np.histogram(miso_deg, bins=n_bins, range=angle_range)
    centers       = 0.5 * (edges[:-1] + edges[1:])
    bin_width     = edges[1] - edges[0]
    density       = counts / (counts.sum() * bin_width) if counts.sum() > 0 else counts.astype(float)

    return {
        'miso_deg':         miso_deg,
        'pairs':            pairs,
        'hist_counts':      counts,
        'hist_density':     density,
        'hist_bin_edges':   edges,
        'hist_bin_centers': centers,
        'mean_angle':       float(np.mean(miso_deg)),
        'std_angle':        float(np.std(miso_deg)),
        'n_pairs':          len(pairs),
    }


# ---------------------------------------------------------------------------
# MDF peak detection
# ---------------------------------------------------------------------------

#: Default cubic CSL reference angles (degrees)
CUBIC_CSL: dict[str, float] = {
    'Σ3  (twin)': 60.00,   # {111}<110>
    'Σ5':         36.87,   # {210}<001>
    'Σ7':         38.21,   # {211}<111>
    'Σ9':         38.94,   # {221}<110>
    'Σ11':        50.48,
    'Σ13b':       27.80,
}


def detect_mdf_peaks(
        mdf: dict,
        prominence: float = 0.002,
        distance: int = 3,
        csl: dict[str, float] | None = None,
        csl_tol: float = 2.0,
        bw_method: str | float = 'scott',
        n_kde: int = 500,
        angle_max: float = 65.0,
) -> dict:
    """
    Detect peaks in a pre-computed MDF, match them against CSL angles, and
    compute a KDE curve over the raw disorientation angles.

    Parameters
    ----------
    mdf : dict
        Output of ``compute_mdf_from_quats()``.  Must contain keys
        ``'hist_bin_centers'``, ``'hist_density'``, and ``'miso_deg'``.
    prominence : float
        Minimum peak prominence for ``scipy.signal.find_peaks``. Default 0.002.
    distance : int
        Minimum number of bins between two peaks. Default 3.
    csl : dict or None
        Mapping ``{label: angle_degrees}``.  If None, uses ``CUBIC_CSL``.
    csl_tol : float
        Tolerance in degrees for declaring a peak "near" a CSL. Default 2.0.
    bw_method : str or float
        Bandwidth method passed to ``scipy.stats.gaussian_kde``. Default ``'scott'``.
    n_kde : int
        Number of points in the KDE evaluation grid. Default 500.
    angle_max : float
        Upper end of the KDE evaluation grid (degrees). Default 65.

    Returns
    -------
    result : dict with keys
        ``'peak_indices'``   – ndarray of bin indices of detected peaks
        ``'peak_angles'``    – list of float, angle of each detected peak
        ``'peak_labels'``    – list of human-readable label strings
        ``'csl_nearest'``    – list of (nearest_label, delta) tuples per peak
        ``'kde'``            – fitted ``gaussian_kde`` object
        ``'kde_vals'``       – ndarray (n_kde,) of KDE density values
        ``'theta_fine'``     – ndarray (n_kde,) evaluation grid (degrees)
        ``'csl'``            – the CSL dict used
        ``'csl_tol'``        – the tolerance used
    """
    from scipy.signal import find_peaks as _find_peaks
    from scipy.stats  import gaussian_kde as _gaussian_kde

    if csl is None:
        csl = CUBIC_CSL

    centers = np.asarray(mdf['hist_bin_centers'])
    density = np.asarray(mdf['hist_density'])

    peak_indices, _ = _find_peaks(density, prominence=prominence, distance=distance)

    peak_angles: list[float] = [float(centers[pi]) for pi in peak_indices]

    csl_nearest: list[tuple[str, float]] = []
    peak_labels:  list[str] = []
    for pi, angle in zip(peak_indices, peak_angles):
        nearest = min(csl, key=lambda k: abs(csl[k] - angle))
        delta   = angle - csl[nearest]
        within  = abs(delta) <= csl_tol
        csl_nearest.append((nearest, float(delta)))
        csl_tag = f'  ≈ {nearest} (Δ={delta:+.1f}°)' if within else ''
        peak_labels.append(f'{angle:.1f}°  (density={density[pi]:.4f}){csl_tag}')

    kde        = _gaussian_kde(mdf['miso_deg'], bw_method=bw_method)
    theta_fine = np.linspace(0, angle_max, n_kde)
    kde_vals   = kde(theta_fine)

    return {
        'peak_indices': peak_indices,
        'peak_angles':  peak_angles,
        'peak_labels':  peak_labels,
        'csl_nearest':  csl_nearest,
        'kde':          kde,
        'kde_vals':     kde_vals,
        'theta_fine':   theta_fine,
        'csl':          csl,
        'csl_tol':      csl_tol,
    }


def segregate_csl_pairs(
        mdf: dict,
        selected_peaks: dict,
        csl: dict[str, float] | None = None,
        csl_tol: float = 2.0,
) -> dict:
    """
    Segregate grain-boundary pairs into CSL categories based on
    user-selected MDF peaks.

    For every selected peak, the nearest CSL reference angle is found; pairs
    whose disorientation lies within *csl_tol* of that reference are collected
    under that CSL label.

    Parameters
    ----------
    mdf : dict
        Output of ``compute_mdf_from_quats()``.  Must contain
        ``'pairs'`` (N,2) and ``'miso_deg'`` (N,).
    selected_peaks : dict
        Output of ``mdf_peak_selector()``.
        Keys: ``'angles'`` (list[float]) and ``'indices'`` (list[int]).
    csl : dict or None
        ``{label: reference_angle_degrees}``. Defaults to ``CUBIC_CSL``.
    csl_tol : float
        Tolerance in degrees. Pairs with |Δθ| ≤ csl_tol are included.
        Default 2.0.

    Returns
    -------
    result : dict  keyed by CSL label, each value is a dict with
        ``'csl_angle'``   – float, reference angle (degrees)
        ``'pairs'``       – ndarray (M, 2), matching grain-ID pairs
        ``'miso_deg'``    – ndarray (M,), disorientation of those pairs
        ``'grains_A'``    – ndarray of unique grain IDs on one side
        ``'grains_B'``    – ndarray of unique grain IDs on other side
        ``'grains_all'``  – ndarray of all unique grain IDs that touch
                            at least one boundary of this type

    Notes
    -----
    A selected peak that maps to the same CSL label as another peak
    (within csl_tol) produces only one key in the output.
    Pairs can match more than one CSL category if tolerances overlap.
    """
    if csl is None:
        csl = CUBIC_CSL

    pairs    = np.asarray(mdf['pairs'],    dtype=int)
    miso_deg = np.asarray(mdf['miso_deg'], dtype=float)

    # ── Collect unique CSL targets from the selected peaks ──────────────────
    # Multiple selected peaks may map to the same CSL; deduplicate by label.
    csl_targets: dict[str, float] = {}
    for angle in selected_peaks['angles']:
        nearest = min(csl, key=lambda k: abs(csl[k] - angle))
        if abs(angle - csl[nearest]) <= csl_tol:
            csl_targets[nearest] = csl[nearest]

    # ── Segregate ────────────────────────────────────────────────────────────
    result: dict = {}
    for label, ref_ang in csl_targets.items():
        mask        = np.abs(miso_deg - ref_ang) <= csl_tol
        sel_pairs   = pairs[mask]
        sel_miso    = miso_deg[mask]
        grains_A    = np.unique(sel_pairs[:, 0]) if len(sel_pairs) else np.empty(0, int)
        grains_B    = np.unique(sel_pairs[:, 1]) if len(sel_pairs) else np.empty(0, int)
        grains_all  = np.unique(sel_pairs) if len(sel_pairs) else np.empty(0, int)
        result[label] = {
            'csl_angle':  ref_ang,
            'pairs':      sel_pairs,
            'miso_deg':   sel_miso,
            'grains_A':   grains_A,
            'grains_B':   grains_B,
            'grains_all': grains_all,
            'n_pairs':    int(mask.sum()),
            'n_grains':   int(len(grains_all)),
        }

    return result


def csl_volume_fractions(
        lfi: np.ndarray,
        csl_grains: dict,
) -> dict:
    """
    Compute the area (volume) fraction of the indexed map occupied by grains
    that participate in each CSL boundary type.

    A grain is counted if it appears in *any* boundary of that CSL type
    (i.e. it belongs to ``csl_grains[label]['grains_all']``).

    Grains can contribute to multiple CSL categories simultaneously, so
    fractions do not necessarily sum to 1.

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx)
        Integer grain label field. Positive values = grain IDs; ≤0 = unindexed.
    csl_grains : dict
        Output of ``segregate_csl_pairs()``.

    Returns
    -------
    result : dict  keyed by CSL label, each value a dict with
        ``'n_pixels'``      – int, total pixels occupied by CSL grains
        ``'vf_indexed'``    – float, fraction of *indexed* pixels
        ``'vf_total'``      – float, fraction of *all* pixels (including unindexed)
        ``'csl_angle'``     – float, reference CSL angle (degrees)
        ``'n_grains'``      – int, number of grains in this CSL type
    """
    lfi          = np.asarray(lfi, dtype=int)
    n_total      = lfi.size
    n_indexed    = int(np.sum(lfi > 0))

    result: dict = {}
    for label, info in csl_grains.items():
        gids     = np.asarray(info['grains_all'], dtype=int)
        n_pixels = int(np.isin(lfi, gids).sum())
        result[label] = {
            'n_pixels':   n_pixels,
            'vf_indexed': n_pixels / n_indexed if n_indexed > 0 else 0.0,
            'vf_total':   n_pixels / n_total   if n_total   > 0 else 0.0,
            'csl_angle':  info['csl_angle'],
            'n_grains':   info['n_grains'],
        }

    return result


def identify_parent_grains(
        csl_grains: dict,
        prop: dict,
) -> dict:
    """
    For each CSL boundary pair, identify the **parent** (larger) grain and
    the **twin / child** (smaller) grain using grain area as the discriminator.

    A grain can play both roles across different pairs (twin chain:
    A→B→C where B is twin of A but parent of C).  Grains are therefore
    classified per their *net* role:

    * ``pure_parents``    – appear as the larger grain in every one of their pairs
    * ``pure_twins``      – appear as the smaller grain in every one of their pairs
    * ``intermediates``   – appear as parent in some pairs and twin in others

    Parameters
    ----------
    csl_grains : dict
        Output of ``segregate_csl_pairs()``.
    prop : dict
        Grain property dict as returned by ``EBSDReader.characterise()['prop']``
        or stored in ``repgen2d.prop_ebsd``.
        Must contain key ``'area'`` for each grain.

    Returns
    -------
    result : dict  keyed by CSL label, each value a dict with

        ``'pairs_labeled'``   – list of ``(parent_gid, twin_gid)`` tuples
        ``'all_parents'``     – ndarray, grains that are parent in ≥1 pair
        ``'all_twins'``       – ndarray, grains that are twin in ≥1 pair
        ``'pure_parents'``    – ndarray, grains that are *only ever* a parent
        ``'pure_twins'``      – ndarray, grains that are *only ever* a twin
        ``'intermediates'``   – ndarray, grains in both roles (twin chains)
        ``'n_pure_parents'``  – int
        ``'n_pure_twins'``    – int
        ``'n_intermediates'`` – int
        ``'csl_angle'``       – float
    """
    result: dict = {}
    for label, info in csl_grains.items():
        pairs          = info['pairs']          # (N, 2) array of grain-ID pairs
        csl_angle      = info['csl_angle']

        pairs_labeled: list[tuple[int, int]] = []
        parent_set: set[int] = set()
        twin_set:   set[int] = set()

        for gA, gB in pairs:
            aA = prop.get(int(gA), {}).get('area', 0.0)
            aB = prop.get(int(gB), {}).get('area', 0.0)
            if aA >= aB:
                parent, twin = int(gA), int(gB)
            else:
                parent, twin = int(gB), int(gA)
            pairs_labeled.append((parent, twin))
            parent_set.add(parent)
            twin_set.add(twin)

        all_parents   = np.array(sorted(parent_set), dtype=int)
        all_twins     = np.array(sorted(twin_set),   dtype=int)
        pure_parents  = np.array(sorted(parent_set - twin_set),   dtype=int)
        pure_twins    = np.array(sorted(twin_set   - parent_set), dtype=int)
        intermediates = np.array(sorted(parent_set & twin_set),   dtype=int)

        result[label] = {
            'pairs_labeled':   pairs_labeled,
            'all_parents':     all_parents,
            'all_twins':       all_twins,
            'pure_parents':    pure_parents,
            'pure_twins':      pure_twins,
            'intermediates':   intermediates,
            'n_pure_parents':  len(pure_parents),
            'n_pure_twins':    len(pure_twins),
            'n_intermediates': len(intermediates),
            'csl_angle':       csl_angle,
        }

    return result


def compute_grain_role_ratios(
        lfi: 'np.ndarray',
        parent_info: dict,
) -> dict:
    """
    Compute the ratio of each grain role (pure parent, pure twin,
    intermediate, non-role) to the total number of indexed grains,
    using the same priority rules as ``plot_grain_role_property_stats``
    (intermediate > parent > twin).

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx)
        Integer grain label field (positive = grain ID).
    parent_info : dict
        Output of :func:`identify_parent_grains`.

    Returns
    -------
    dict
        Keys: ``'total'``, ``'pure_parents'``, ``'pure_twins'``,
        ``'intermediates'``, ``'non_role'``, ``'all_role'``.
        Each value is a sub-dict with ``'count'`` and ``'ratio'``.
    """
    all_pp: set = set()
    all_pt: set = set()
    all_im: set = set()
    for info in parent_info.values():
        all_pp.update(info['pure_parents'].tolist())
        all_pt.update(info['pure_twins'].tolist())
        all_im.update(info['intermediates'].tolist())

    only_pp_or_pt    = (all_pp | all_pt) - all_im
    pure_parent_set  = only_pp_or_pt & all_pp
    pure_twin_set    = (only_pp_or_pt & all_pt) - pure_parent_set
    intermediate_set = all_im
    all_role_set     = pure_parent_set | pure_twin_set | intermediate_set
    total_gids       = set(int(g) for g in np.unique(lfi[lfi > 0]))
    non_role_set     = total_gids - all_role_set
    N                = len(total_gids)

    def _entry(gset: set) -> dict:
        n = len(gset)
        return {'count': n, 'ratio': n / N if N else 0.0, 'grain_ids': gset}

    return {
        'total':        {'count': N, 'ratio': 1.0, 'grain_ids': total_gids},
        'pure_parents':  _entry(pure_parent_set),
        'pure_twins':    _entry(pure_twin_set),
        'intermediates': _entry(intermediate_set),
        'non_role':      _entry(non_role_set),
        'all_role':      _entry(all_role_set),
    }


def assign_grain_roles_by_ratio(
        lfi: 'np.ndarray',
        grain_areas: dict,
        target_ratios: dict,
) -> dict:
    """
    Assign pure-parent / pure-twin / intermediate / non-role labels to grains
    in a grain structure so that the resulting role fractions match the
    *target_ratios* (e.g. derived from an EBSD dataset via
    :func:`compute_grain_role_ratios`).

    Grains are sorted by area (descending).  The largest grains receive the
    *pure parent* label, the next tier *pure twin*, the next *intermediate*,
    and the remainder *non-role*.  This reflects the physical expectation that
    parent grains are typically larger than their twins.

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx)
        Integer grain label field (positive = grain ID).
    grain_areas : dict
        ``{grain_id: area}`` mapping.  Can be built from an EBSD
        ``prop_ebsd`` dict or from a ``mcgs2_grain_structure.prop``
        DataFrame column.
    target_ratios : dict
        Output of :func:`compute_grain_role_ratios`.  The ``'ratio'`` field
        of ``'pure_parents'``, ``'pure_twins'``, ``'intermediates'``, and
        ``'non_role'`` entries is used.

    Returns
    -------
    dict
        Same structure as :func:`compute_grain_role_ratios`:
        keys ``'total'``, ``'pure_parents'``, ``'pure_twins'``,
        ``'intermediates'``, ``'non_role'``, ``'all_role'``;
        each value has ``'count'``, ``'ratio'``, ``'grain_ids'``.
    """
    total_gids = sorted(int(g) for g in np.unique(lfi[lfi > 0]))
    N = len(total_gids)
    if N == 0:
        empty = {'count': 0, 'ratio': 0.0, 'grain_ids': set()}
        return {k: empty for k in ('total', 'pure_parents', 'pure_twins',
                                   'intermediates', 'non_role', 'all_role')}

    # Sort grain IDs by area descending
    sorted_gids = sorted(total_gids,
                         key=lambda g: grain_areas.get(g, 0.0),
                         reverse=True)

    # Compute target counts from ratios (round, ensure sum = N)
    r_pp = target_ratios['pure_parents']['ratio']
    r_pt = target_ratios['pure_twins']['ratio']
    r_im = target_ratios['intermediates']['ratio']

    n_pp = int(round(r_pp * N))
    n_pt = int(round(r_pt * N))
    n_im = int(round(r_im * N))

    # Clamp so totals don't exceed N
    n_pp = min(n_pp, N)
    n_pt = min(n_pt, N - n_pp)
    n_im = min(n_im, N - n_pp - n_pt)
    n_nr = N - n_pp - n_pt - n_im

    # Slice sorted list
    pp_set = set(sorted_gids[:n_pp])
    pt_set = set(sorted_gids[n_pp: n_pp + n_pt])
    im_set = set(sorted_gids[n_pp + n_pt: n_pp + n_pt + n_im])
    nr_set = set(sorted_gids[n_pp + n_pt + n_im:])
    all_role_set = pp_set | pt_set | im_set

    def _entry(gset: set) -> dict:
        n = len(gset)
        return {'count': n, 'ratio': n / N, 'grain_ids': gset}

    return {
        'total':         {'count': N, 'ratio': 1.0, 'grain_ids': set(total_gids)},
        'pure_parents':  _entry(pp_set),
        'pure_twins':    _entry(pt_set),
        'intermediates': _entry(im_set),
        'non_role':      _entry(nr_set),
        'all_role':      _entry(all_role_set),
    }


def introduce_twins_by_csl(
        lfi: 'np.ndarray',
        parent_grain_ids: 'set | list',
        csl_label: str,
        twin_half_width: float = 2.0,
        twin_angle_deg: float | None = None,
        n_twins_per_parent: int = 1,
        angle_perturb_deg: float = 0.0,
        width_perturb: float = 0.0,
        rng_seed: int | None = None,
) -> dict:
    """
    Introduce straight twin lamellae into selected parent grains on a pixel
    grid using :class:`~upxo.geoEntities.sline2d.Sline2d`.

    For each parent grain a base angle is chosen; all lamellae within that
    grain are parallel (same angle) but can receive small independent
    perturbations to their angle and half-width to model realistic scatter.

    Parameters
    ----------
    lfi : ndarray, shape (ny, nx)
        Integer grain label field.  **Modified in-place.**
    parent_grain_ids : set or list
        Grain IDs of the parent grains to twin.
    csl_label : str
        CSL type label (e.g. ``'Σ3 (twin)'``).  Stored in output metadata.
    twin_half_width : float
        Nominal half-width in pixels of each twin lamella.
    twin_angle_deg : float or None
        Fixed base inclination angle (degrees, from the x-axis).  If ``None``,
        a random angle in [0°, 180°) is drawn independently for each parent.
    n_twins_per_parent : int
        Number of parallel twin lamellae to insert per parent grain.
    angle_perturb_deg : float
        Maximum angular perturbation (degrees) applied independently to each
        lamella angle around the parent's base angle.  A uniform random draw
        in ``[-angle_perturb_deg, +angle_perturb_deg]`` is used.  Set to 0
        for strictly parallel lamellae.
    width_perturb : float
        Maximum half-width perturbation (pixels) applied independently to each
        lamella.  A uniform random draw in
        ``[-width_perturb, +width_perturb]`` is added to *twin_half_width*.
        The effective half-width is clamped to a minimum of 0.5 px.
    rng_seed : int or None
        Seed for the random number generator (reproducibility).

    Returns
    -------
    dict
        ``{'lfi': ndarray,
           'twin_lines': dict[parent_gid -> list[Sline2d]],
           'new_twin_gids': dict[parent_gid -> list[new_gid]],
           'csl_label': str}``
    """
    from upxo.geoEntities.sline2d import Sline2d as Sl2d

    lfi = np.asarray(lfi, dtype=int)
    rng = np.random.default_rng(rng_seed)
    next_gid = int(lfi.max()) + 1

    twin_lines: dict = {}
    new_twin_gids: dict = {}

    for parent_gid in parent_grain_ids:
        parent_gid = int(parent_gid)
        rows, cols = np.where(lfi == parent_gid)
        if rows.size == 0:
            continue

        # Bounding box of parent grain (in pixel coords)
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        cx = (c_min + c_max) / 2.0
        cy = (r_min + r_max) / 2.0
        diag = np.sqrt((r_max - r_min) ** 2 + (c_max - c_min) ** 2) + 1.0

        twin_lines[parent_gid] = []
        new_twin_gids[parent_gid] = []

        # Parent pixel coordinates as (x=col, y=row) for Sline2d
        parent_xs = cols.astype(float)
        parent_ys = rows.astype(float)

        # Base angle for this parent (all lamellae are nominally parallel)
        base_angle = (float(twin_angle_deg) if twin_angle_deg is not None
                      else float(rng.uniform(0.0, 180.0)))

        # Build base line to compute normal direction for parallel spacing
        base_line = Sl2d.by_LFAL(
            location=[cx, cy],
            factor=0.5,
            angle=base_angle,
            length=diag,
            degree=True,
        )
        nv = base_line.normal_vector(ratio=0.0, return_type='sl2d')
        nx_unit, ny_unit = nv.x0, nv.y0
        norm = np.hypot(nx_unit, ny_unit)
        if norm > 0:
            nx_unit /= norm
            ny_unit /= norm

        # Nominal spacing: 2*half_width + 1 px gap
        spacing_px = max(2.0 * twin_half_width + 1.0, 4.0)

        # Symmetric offsets along the normal
        half_n = (n_twins_per_parent - 1) / 2.0
        all_lines = []
        perturbed_widths = []
        for k in range(n_twins_per_parent):
            # Per-lamella angle perturbation
            da = (rng.uniform(-angle_perturb_deg, angle_perturb_deg)
                  if angle_perturb_deg > 0.0 else 0.0)
            lam_angle = base_angle + da

            # Per-lamella width perturbation
            dw = (rng.uniform(-width_perturb, width_perturb)
                  if width_perturb > 0.0 else 0.0)
            lam_half_width = max(0.5, twin_half_width + dw)

            offset = (k - half_n) * spacing_px
            ox = cx + offset * nx_unit
            oy = cy + offset * ny_unit
            line = Sl2d.by_LFAL(
                location=[ox, oy],
                factor=0.5,
                angle=lam_angle,
                length=diag,
                degree=True,
            )
            all_lines.append(line)
            perturbed_widths.append(lam_half_width)

        # Assign pixels for each lamella; skip already-relabelled pixels
        for line, lam_hw in zip(all_lines, perturbed_widths):
            twin_lines[parent_gid].append(line)

            still_parent = (lfi[rows, cols] == parent_gid)
            if not still_parent.any():
                continue
            sub_xs = parent_xs[still_parent]
            sub_ys = parent_ys[still_parent]
            sub_rows = rows[still_parent]
            sub_cols = cols[still_parent]

            twin_pixel_idx = line.find_neigh_point_by_perp_distance(
                (sub_xs, sub_ys),
                r=lam_hw,
                use_bounding_rec=True,
            )
            if not twin_pixel_idx:
                continue

            t_rows = sub_rows[twin_pixel_idx]
            t_cols = sub_cols[twin_pixel_idx]
            lfi[t_rows, t_cols] = next_gid
            new_twin_gids[parent_gid].append(next_gid)
            next_gid += 1

    return {
        'lfi':           lfi,
        'twin_lines':    twin_lines,
        'new_twin_gids': new_twin_gids,
        'csl_label':     csl_label,
    }


# ---------------------------------------------------------------------------
# Grain-size matching scale factor
# ---------------------------------------------------------------------------

def compute_scale_factor_grain_size(
    lfi_sim: np.ndarray,
    mean_area_ebsd_um2: float,
    ebsd_step_size_um: float,
    prop_ebsd: dict | None = None,
) -> dict:
    """
    Compute the linear scale factor (µm per sim-pixel) that makes the mean
    grain area of the synthetic label field equal to the EBSD mean grain area.

    The derivation is:

    .. math::

        s = \\sqrt{\\bar{A}_{\\text{EBSD}} / \\bar{A}_{\\text{sim}}}

    where :math:`s` is in µm/px, :math:`\\bar{A}_{\\text{EBSD}}` is in µm²,
    and :math:`\\bar{A}_{\\text{sim}}` is in px².

    Parameters
    ----------
    lfi_sim : ndarray (ny, nx)
        Synthetic label field (integer grain IDs, background ≤ 0).
    mean_area_ebsd_um2 : float
        Mean grain area of the EBSD target in µm².
        Typically ``rg.stat_ebsd['area']['mean']``.
    ebsd_step_size_um : float
        EBSD step size in µm (used for reporting only).
    prop_ebsd : dict or None
        ``{grain_id: {'area': ..., ...}}`` dict from EBSD characterisation.
        When provided, per-grain EBSD areas are stored in the result for
        downstream plotting.

    Returns
    -------
    dict with keys
        ``'scale_factor'``          : float — µm per sim-pixel
        ``'mean_area_sim_px2'``     : float — mean sim grain area (px²)
        ``'mean_area_ebsd_um2'``    : float — target EBSD mean area (µm²)
        ``'mean_area_sim_um2'``     : float — sim mean area after scaling (µm²)
        ``'ebsd_step_size_um'``     : float
        ``'sim_domain_px'``         : tuple (ny, nx)
        ``'sim_domain_um'``         : ndarray (ny_um, nx_um) — physical size
        ``'gids_sim'``              : ndarray — grain IDs from lfi_sim (> 0)
        ``'counts_sim'``            : ndarray — pixel counts per grain
        ``'areas_sim_um2'``         : ndarray — per-grain areas after scaling (µm²)
        ``'areas_ebsd_um2'``        : ndarray or None — per-grain EBSD areas (µm²)
        ``'ratio_pixel_sizes'``     : float — scale_factor / ebsd_step_size_um
    """
    gids, counts = np.unique(lfi_sim, return_counts=True)
    valid         = gids > 0
    gids, counts  = gids[valid], counts[valid]

    mean_area_sim_px2   = counts.mean()
    scale_factor        = np.sqrt(mean_area_ebsd_um2 / mean_area_sim_px2)
    areas_sim_um2       = counts * scale_factor**2
    sim_domain_um       = np.array(lfi_sim.shape, dtype=float) * scale_factor

    areas_ebsd_um2 = None
    if prop_ebsd is not None:
        areas_ebsd_um2 = np.array([v['area'] for v in prop_ebsd.values()],
                                   dtype=float)

    return {
        'scale_factor':       float(scale_factor),
        'mean_area_sim_px2':  float(mean_area_sim_px2),
        'mean_area_ebsd_um2': float(mean_area_ebsd_um2),
        'mean_area_sim_um2':  float(areas_sim_um2.mean()),
        'ebsd_step_size_um':  float(ebsd_step_size_um),
        'sim_domain_px':      tuple(lfi_sim.shape),
        'sim_domain_um':      sim_domain_um,
        'gids_sim':           gids,
        'counts_sim':         counts,
        'areas_sim_um2':      areas_sim_um2,
        'areas_ebsd_um2':     areas_ebsd_um2,
        'ratio_pixel_sizes':  float(scale_factor / ebsd_step_size_um),
    }


# ---------------------------------------------------------------------------
# Twin thickness statistics
# ---------------------------------------------------------------------------

def compute_twin_thickness_stats(
    parent_info: dict,
    prop_ebsd: dict,
    step_size_um: float,
    thickness_key: str | None = None,
) -> dict:
    """
    Compute twin lamella thickness statistics from EBSD grain properties.

    Parameters
    ----------
    parent_info : dict
        Output of ``identify_parent_grains``.  Each value must expose
        ``'pure_twins'`` and ``'intermediates'`` arrays of EBSD grain IDs.
    prop_ebsd : dict
        Per-grain property dict ``{grain_id: {'minor_axis_length': ..., ...}}``.
    step_size_um : float
        EBSD step size in µm/pixel used to convert pixel lengths to µm.
    thickness_key : str or None
        Key to use as the thickness proxy.  If *None* (default), uses
        ``'minor_axis_length'`` when present, otherwise ``'eq_diameter'``.

    Returns
    -------
    dict with keys
        ``'gids'``      : list[int]   — twin grain IDs with data
        ``'thick_px'``  : ndarray(N,) — thickness in pixels
        ``'thick_um'``  : ndarray(N,) — thickness in µm
        ``'col'``       : str         — property key used
        ``'step_um'``   : float       — step size used
        ``'mean'``      : float
        ``'median'``    : float
        ``'std'``       : float
        ``'min'``       : float
        ``'max'``       : float
        ``'pct10'``     : float  (10th percentile)
        ``'pct90'``     : float  (90th percentile)
    """
    # 1. Collect twin + intermediate grain IDs
    twin_gids: set[int] = set()
    for info in parent_info.values():
        twin_gids.update(int(g) for g in np.asarray(info['pure_twins']).ravel())
        twin_gids.update(int(g) for g in np.asarray(info['intermediates']).ravel())

    gids_in_prop    = set(prop_ebsd.keys())
    twin_gids_found = sorted(twin_gids & gids_in_prop)

    # 2. Choose property key
    sample_keys = list(prop_ebsd[next(iter(prop_ebsd))].keys())
    if thickness_key is None:
        if 'minor_axis_length' in sample_keys:
            thickness_key = 'minor_axis_length'
        elif 'eq_diameter' in sample_keys:
            thickness_key = 'eq_diameter'
        else:
            raise RuntimeError(
                f"No suitable thickness key found.  Available: {sample_keys}"
            )

    # 3. Extract and filter
    thick_px = np.array(
        [prop_ebsd[g][thickness_key] for g in twin_gids_found
         if prop_ebsd[g][thickness_key] is not None
         and prop_ebsd[g][thickness_key] > 0],
        dtype=np.float64,
    )
    thick_um = thick_px * float(step_size_um)

    return {
        'gids':     twin_gids_found,
        'thick_px': thick_px,
        'thick_um': thick_um,
        'col':      thickness_key,
        'step_um':  float(step_size_um),
        'mean':     float(thick_um.mean()),
        'median':   float(np.median(thick_um)),
        'std':      float(thick_um.std(ddof=1)),
        'min':      float(thick_um.min()),
        'max':      float(thick_um.max()),
        'pct10':    float(np.percentile(thick_um, 10)),
        'pct90':    float(np.percentile(thick_um, 90)),
    }


# ---------------------------------------------------------------------------
# Orientation assignment – MDF-matched simulation
# ---------------------------------------------------------------------------

# Σ3 twinning rotation: 60° about [111]/√3  (w, x, y, z) convention
_SIGMA3_Q = np.array([
    np.sqrt(3.0) / 2.0,
    1.0 / (2.0 * np.sqrt(3.0)),
    1.0 / (2.0 * np.sqrt(3.0)),
    1.0 / (2.0 * np.sqrt(3.0)),
], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions stored as (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def _positive_w(q: np.ndarray) -> np.ndarray:
    """Return *q* in the canonical hemisphere (w ≥ 0)."""
    return q if q[0] >= 0.0 else -q


def assign_orientations_mdf_matched(
    lfi_after: np.ndarray,
    gs_roles: dict,
    twin_result: dict,
    ebsd_lfi: np.ndarray,
    ebsd_quat: np.ndarray,
    ebsd_parent_info: dict,
    rng_seed=None,
) -> dict:
    """
    Assign grain-averaged quaternion orientations to every domain in the
    twinned simulated label field *lfi_after* so that the grain-boundary
    MDF of the simulation approximates the EBSD target.

    Assignment strategy
    -------------------
    Pure-parent sim grains
        Sampled randomly from the pool of EBSD pure-parent grain-averaged
        quaternions.
    Geometrically-introduced twin grains  (new IDs in *twin_result*)
        Derived from the parent's assigned quaternion via the Σ3 twinning
        rotation:  q_twin = Σ3_Q ⊗ q_parent
    Pre-labelled pure-twin sim grains  (from Section 15, not in twin_result)
        Sampled from the EBSD pure-twin pool.
    Intermediate sim grains
        Sampled from the EBSD intermediate pool.
    Non-role sim grains
        Sampled from the EBSD non-role pool.

    Pools are sampled *with replacement* when the pool is smaller than the
    number of grains needing assignment.  The neighbour graph of *lfi_after*
    (4-connected pixel contacts) is returned so the caller can immediately
    compute the simulated MDF and compare it with the EBSD reference.

    Parameters
    ----------
    lfi_after : ndarray (ny, nx)
        Label field after twin introduction (integer grain IDs, 0 = background).
    gs_roles : dict
        Output of ``assign_grain_roles_by_ratio``.  Must contain keys
        ``'pure_parents'``, ``'pure_twins'``, ``'intermediates'``, ``'non_role'``
        each with a ``'grain_ids'`` entry that is a set of integer grain IDs.
    twin_result : dict
        Output of ``introduce_twins_by_csl``.  Must contain key
        ``'new_twin_gids'`` : {parent_gid : [twin_gid, ...]}.
    ebsd_lfi : ndarray (ny_e, nx_e)
        EBSD label field (grain IDs, 0 = boundary / unindexed).
    ebsd_quat : ndarray (ny_e, nx_e, 4)
        Per-pixel quaternions from EBSD in (w, x, y, z) convention.
    ebsd_parent_info : dict
        Output of ``find_csl_grains`` (keyed by CSL label).  Each value
        must expose ``'pure_parents'``, ``'pure_twins'``, and
        ``'intermediates'`` arrays of EBSD grain IDs.
    rng_seed : int or None
        Seed for the random-number generator (reproducibility).

    Returns
    -------
    dict with keys
        ``'grain_quats'``   : {grain_id : ndarray(4,)}
            Grain-averaged quaternion for every non-background grain in
            *lfi_after*.
        ``'quat_pixel'``    : ndarray(ny, nx, 4)
            Per-pixel quaternion map built by flooding each grain with its
            assigned quaternion.
        ``'neigh_gid_sim'`` : {grain_id : list[int]}
            4-connected contact neighbour graph of *lfi_after*.
    """
    rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # 1. EBSD grain-averaged quaternions
    # ------------------------------------------------------------------
    gids_ebsd, q_mean_ebsd = grain_avg_quats(ebsd_lfi, ebsd_quat)
    gid_to_q_ebsd = {int(g): q_mean_ebsd[i] for i, g in enumerate(gids_ebsd)}

    # ------------------------------------------------------------------
    # 2. Build EBSD role-stratified pools
    #    Priority: intermediate > (parent + twin) to match ratio logic.
    # ------------------------------------------------------------------
    all_pp, all_pt, all_im = set(), set(), set()
    for info in ebsd_parent_info.values():
        all_pp.update(int(g) for g in np.asarray(info['pure_parents']).ravel())
        all_pt.update(int(g) for g in np.asarray(info['pure_twins']).ravel())
        all_im.update(int(g) for g in np.asarray(info['intermediates']).ravel())

    all_role      = all_pp | all_pt | all_im
    all_gids_ebsd = set(gid_to_q_ebsd.keys())
    all_nr        = all_gids_ebsd - all_role

    # Remove intermediates from pure sets (same priority rule as compute_grain_role_ratios)
    pure_pp_set = all_pp - all_im
    pure_pt_set = (all_pt - all_im) - pure_pp_set
    im_set      = all_im

    def _build_pool(gid_set):
        qs = [gid_to_q_ebsd[g] for g in gid_set if g in gid_to_q_ebsd]
        return np.array(qs, dtype=np.float64) if qs else np.empty((0, 4), dtype=np.float64)

    pool_pp = _build_pool(pure_pp_set)
    pool_pt = _build_pool(pure_pt_set)
    pool_im = _build_pool(im_set)
    pool_nr = _build_pool(all_nr)

    def _sample_pool(pool, n):
        """Draw *n* quaternions from *pool* with replacement; fallback to
        random unit quaternions when pool is empty."""
        if n == 0:
            return []
        if len(pool) == 0:
            q = rng.standard_normal((n, 4))
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            return [_positive_w(q[i]) for i in range(n)]
        idx = rng.integers(0, len(pool), size=n)
        return [_positive_w(pool[i].copy()) for i in idx]

    # ------------------------------------------------------------------
    # 3. Classify sim grains
    # ------------------------------------------------------------------
    new_twin_map = twin_result.get('new_twin_gids', {})   # parent → [twins]
    introduced_twin_gids: set[int] = set()
    parent_of_twin: dict[int, int] = {}
    for pg, tg_list in new_twin_map.items():
        for tg in tg_list:
            introduced_twin_gids.add(int(tg))
            parent_of_twin[int(tg)] = int(pg)

    # Remove introduced twin IDs from pre-labelled role sets
    sim_pp_gids = gs_roles['pure_parents']['grain_ids']  - introduced_twin_gids
    sim_pt_gids = gs_roles['pure_twins']['grain_ids']    - introduced_twin_gids
    sim_im_gids = gs_roles['intermediates']['grain_ids'] - introduced_twin_gids
    sim_nr_gids = gs_roles['non_role']['grain_ids']      - introduced_twin_gids

    # ------------------------------------------------------------------
    # 4. Assign orientations by role
    # ------------------------------------------------------------------
    grain_quats: dict[int, np.ndarray] = {}

    # Parents first – introduced twins depend on them
    for gid, q in zip(sorted(sim_pp_gids), _sample_pool(pool_pp, len(sim_pp_gids))):
        grain_quats[gid] = q

    for gid, q in zip(sorted(sim_pt_gids), _sample_pool(pool_pt, len(sim_pt_gids))):
        grain_quats[gid] = q

    for gid, q in zip(sorted(sim_im_gids), _sample_pool(pool_im, len(sim_im_gids))):
        grain_quats[gid] = q

    for gid, q in zip(sorted(sim_nr_gids), _sample_pool(pool_nr, len(sim_nr_gids))):
        grain_quats[gid] = q

    # Introduced twin grains: q_twin = Σ3_Q ⊗ q_parent
    for tg, pg in parent_of_twin.items():
        if pg not in grain_quats:
            # Parent might not have been assigned if it was classified differently;
            # assign it now from the parent pool as fallback.
            q_fb = _sample_pool(pool_pp, 1)
            grain_quats[pg] = q_fb[0] if q_fb else np.array([1., 0., 0., 0.])
        grain_quats[tg] = _positive_w(_quat_mul(_SIGMA3_Q, grain_quats[pg]))

    # ------------------------------------------------------------------
    # 5. Per-pixel quaternion array
    # ------------------------------------------------------------------
    ny, nx = lfi_after.shape
    quat_pixel = np.zeros((ny, nx, 4), dtype=np.float64)
    quat_pixel[..., 0] = 1.0   # identity as default (background)
    for gid, q in grain_quats.items():
        quat_pixel[lfi_after == gid] = q

    # ------------------------------------------------------------------
    # 6. 4-connected neighbour graph
    # ------------------------------------------------------------------
    all_sim_ids = np.unique(lfi_after)
    all_sim_ids = all_sim_ids[all_sim_ids > 0]
    neigh_gid_sim: dict[int, set] = {int(g): set() for g in all_sim_ids}

    left, right = lfi_after[:, :-1], lfi_after[:, 1:]
    mask_h = (left > 0) & (right > 0) & (left != right)
    for a, b in zip(left[mask_h].ravel(), right[mask_h].ravel()):
        a, b = int(a), int(b)
        neigh_gid_sim[a].add(b)
        neigh_gid_sim[b].add(a)

    top, bot = lfi_after[:-1, :], lfi_after[1:, :]
    mask_v = (top > 0) & (bot > 0) & (top != bot)
    for a, b in zip(top[mask_v].ravel(), bot[mask_v].ravel()):
        a, b = int(a), int(b)
        neigh_gid_sim[a].add(b)
        neigh_gid_sim[b].add(a)

    neigh_gid_sim_lists = {k: list(v) for k, v in neigh_gid_sim.items()}

    return {
        'grain_quats':   grain_quats,
        'quat_pixel':    quat_pixel,
        'neigh_gid_sim': neigh_gid_sim_lists,
    }


# ---------------------------------------------------------------------------
# Special orientation relationships
# ---------------------------------------------------------------------------

def get_ks_rotations() -> np.ndarray:
    """
    Return the 24 Kurdjumov-Sachs (K-S) orientation relationship operators
    as (24, 3, 3) rotation matrices.

    The K-S OR is defined by the relationship between FCC austenite and
    BCC martensite/ferrite:
        {111}γ ∥ {110}α,   <110>γ ∥ <111>α

    Returns
    -------
    ks_variants : ndarray, shape (24, 3, 3)
    """
    r1    = Rotation.from_euler('Z', 45, degrees=True)
    r2    = Rotation.from_rotvec(
        np.deg2rad(54.7356) * np.array([-1., 1., 0.]) / np.sqrt(2)
    )
    g_ks  = (r2 * r1).as_matrix()

    ops: list[np.ndarray] = []
    for p in permutations(range(3)):
        P = np.eye(3)[list(p)]
        for signs in product([-1, 1], repeat=3):
            S = P * np.array(signs)
            if round(np.linalg.det(S)) == 1:
                ops.append(S)
    sym_ops     = np.array(ops)       # (48, 3, 3)
    ks_variants = sym_ops @ g_ks      # (48, 3, 3)  — degenerate variants included
    return ks_variants


# ---------------------------------------------------------------------------
# Self-sufficient examples  (run individually or via exmp_all())
# Each function is fully standalone: imports nothing from outer scope beyond
# this module and standard library.
# ---------------------------------------------------------------------------

def exmp_cubic_misorientation():
    """
    Demonstrate cubic_misorientation() for a single grain-boundary pair.

    Computes the fundamental misorientation angle (degrees), the rotation
    axis, and the three smallest unique equivalent angles between two grains
    whose orientations are given as Bunge Euler angles.
    """
    from upxo.xtalphy.crystal_orientation import cubic_misorientation

    ea_grain_A = [0.0,  35.0, 45.0]   # Copper component
    ea_grain_B = [90.0, 45.0,  0.0]   # Goss component

    angle, axis, top3 = cubic_misorientation(ea_grain_A, ea_grain_B, degrees=True)

    print("=== exmp_cubic_misorientation ===")
    print(f"  Grain A (Copper): phi1={ea_grain_A[0]}, Phi={ea_grain_A[1]}, phi2={ea_grain_A[2]} deg")
    print(f"  Grain B (Goss):   phi1={ea_grain_B[0]}, Phi={ea_grain_B[1]}, phi2={ea_grain_B[2]} deg")
    print(f"  Fundamental misorientation angle : {angle:.4f} deg")
    print(f"  Rotation axis (sample frame)     : [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
    print(f"  Top-3 unique equivalent angles   : {[round(a, 4) for a in top3]} deg")


def exmp_grain_boundary_misorientation_distribution():
    """
    Demonstrate grain_boundary_misorientation_distribution() with explicit
    grain-boundary pairs and random per-grain Euler angles.

    Prints histogram statistics and plots the GBMD bar chart.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from upxo.xtalphy.crystal_orientation import (
        grain_boundary_misorientation_distribution,
    )

    rng = np.random.default_rng(seed=42)
    N_grains = 30
    grain_ids = np.arange(1, N_grains + 1)

    # Random Bunge Euler angles in degrees
    euler = rng.uniform([0., 0., 0.], [360., 180., 360.], size=(N_grains, 3))

    # Build a simple chain of adjacent pairs: 1-2, 2-3, ..., (N-1)-N
    pairs = np.column_stack([grain_ids[:-1], grain_ids[1:]])

    result = grain_boundary_misorientation_distribution(
        euler, pairs,
        grain_ids=grain_ids,
        degrees=True,
        n_bins=18,
        angle_range=(0.0, 65.0),
    )

    print("=== exmp_grain_boundary_misorientation_distribution ===")
    print(f"  Pairs processed : {result['n_pairs']}")
    print(f"  Mean angle      : {result['mean_angle']:.3f} deg")
    print(f"  Median angle    : {result['median_angle']:.3f} deg")
    print(f"  Std deviation   : {result['std_angle']:.3f} deg")

    fig, ax = plt.subplots(figsize=(7, 4))
    width = result['hist_bin_edges'][1] - result['hist_bin_edges'][0]
    ax.bar(result['hist_bin_centers'], result['hist_counts'], width=width,
           edgecolor='k', color='steelblue', alpha=0.8)
    ax.set_xlabel('Misorientation angle (°)')
    ax.set_ylabel('Number of grain boundaries')
    ax.set_title(f"GBMD  (n={result['n_pairs']} pairs, "
                 f"mean={result['mean_angle']:.1f}°, "
                 f"std={result['std_angle']:.1f}°)")
    ax.set_xlim(0, 65)
    plt.tight_layout()
    plt.show()


def exmp_gbmd_from_lfi():
    """
    Demonstrate gbmd_from_lfi() on a small synthetic label-field image.

    A 20×20 pixel domain is partitioned into a regular 4×4 grid of grains
    (16 grains total) each given a random Bunge Euler angle.  Grain
    boundaries are detected automatically from the label field.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from upxo.xtalphy.crystal_orientation import gbmd_from_lfi

    rng = np.random.default_rng(seed=7)
    ny, nx  = 20, 20
    n_cols  = 4
    n_rows  = 4
    N_grains = n_rows * n_cols

    # Build label field: grain IDs 1..16 tiled in a 4×4 block pattern
    lfi = np.zeros((ny, nx), dtype=int)
    tile_h = ny // n_rows
    tile_w = nx // n_cols
    grain_ids = []
    for row in range(n_rows):
        for col in range(n_cols):
            gid = row * n_cols + col + 1
            grain_ids.append(gid)
            r0, r1 = row*tile_h, (row+1)*tile_h
            c0, c1 = col*tile_w, (col+1)*tile_w
            lfi[r0:r1, c0:c1] = gid

    grain_ids = np.asarray(grain_ids)
    euler     = rng.uniform([0., 0., 0.], [360., 180., 360.],
                             size=(N_grains, 3))

    result = gbmd_from_lfi(
        lfi, euler,
        grain_ids=grain_ids,
        connectivity=4,
        n_bins=13,
        angle_range=(0.0, 65.0),
    )

    print("=== exmp_gbmd_from_lfi ===")
    print(f"  GB pairs detected : {len(result['gb_pairs'])}")
    print(f"  Pairs processed   : {result['n_pairs']}")
    print(f"  Mean angle        : {result['mean_angle']:.3f} deg")
    print(f"  Median angle      : {result['median_angle']:.3f} deg")
    print(f"  Std deviation     : {result['std_angle']:.3f} deg")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: label field
    axes[0].imshow(lfi, cmap='tab20', interpolation='nearest')
    axes[0].set_title('Label field (grain IDs)')
    axes[0].set_xlabel('x (pixels)'); axes[0].set_ylabel('y (pixels)')

    # Right: GBMD
    width = result['hist_bin_edges'][1] - result['hist_bin_edges'][0]
    axes[1].bar(result['hist_bin_centers'], result['hist_counts'], width=width,
                edgecolor='k', color='darkorange', alpha=0.85)
    axes[1].set_xlabel('Misorientation angle (°)')
    axes[1].set_ylabel('Number of grain boundaries')
    axes[1].set_title(f"GBMD  (n={result['n_pairs']} pairs, "
                      f"mean={result['mean_angle']:.1f}°)")
    axes[1].set_xlim(0, 65)
    plt.tight_layout()
    plt.show()


def exmp_euler_bunge_to_matrix():
    """
    Demonstrate euler_bunge_to_matrix() for well-known FCC texture components
    and verify R is orthogonal and det(R)=+1.
    """
    import numpy as np
    from upxo.xtalphy.crystal_orientation import (
        euler_bunge_to_matrix,
        FCC_TEXTURE_COMPONENTS,
    )

    print("=== exmp_euler_bunge_to_matrix ===")
    for name, (phi1, Phi, phi2) in FCC_TEXTURE_COMPONENTS.items():
        R   = euler_bunge_to_matrix(phi1, Phi, phi2, degrees=True)
        det = np.linalg.det(R)
        err = np.linalg.norm(R @ R.T - np.eye(3), ord='fro')
        print(f"  {name:<14s}  phi1={phi1:5.1f}  Phi={Phi:5.1f}  phi2={phi2:5.1f} "
              f"  det={det:.6f}  ||RR^T - I||_F={err:.2e}")


def exmp_normalize_euler_bunge():
    """
    Demonstrate normalize_euler_bunge() on edge-case inputs including
    negative Phi, Phi > 180, and out-of-range phi1/phi2.
    """
    import numpy as np
    from upxo.xtalphy.crystal_orientation import normalize_euler_bunge

    raw = np.array([
        [ 370.,  -20.,  400.],   # phi1 > 360, Phi < 0, phi2 > 360
        [ -10.,  200.,  -5.],    # phi1 < 0,   Phi > 180
        [  45.,   90.,  90.],    # already canonical
        [   0.,    0.,   0.],    # identity
    ])

    norm = normalize_euler_bunge(raw, degrees=True)

    print("=== exmp_normalize_euler_bunge ===")
    print(f"  {'Raw':>30s}   {'Normalised':>30s}")
    for r, n in zip(raw, norm):
        rs = f"({r[0]:6.1f}, {r[1]:6.1f}, {r[2]:6.1f})"
        ns = f"({n[0]:6.1f}, {n[1]:6.1f}, {n[2]:6.1f})"
        print(f"  {rs}  ->  {ns}")


def exmp_fcc_symmetrise_ori():
    """
    Demonstrate fcc_symmetrise_ori() for the Copper texture component.
    Prints the number of distinct symmetrically equivalent orientations
    and the first few triplets.
    """
    import numpy as np
    from upxo.xtalphy.crystal_orientation import fcc_symmetrise_ori

    bea_copper = (90.0, 35.0, 45.0)   # Copper: phi1, Phi, phi2 in degrees
    equiv      = fcc_symmetrise_ori(bea_copper)

    print("=== exmp_fcc_symmetrise_ori ===")
    print(f"  Input orientation (Copper): phi1={bea_copper[0]}, "
          f"Phi={bea_copper[1]}, phi2={bea_copper[2]} deg")
    print(f"  Number of distinct equivalent orientations: {len(equiv)}")
    print("  First 5 equivalents (phi1, Phi, phi2):")
    for row in equiv[:5]:
        print(f"    ({row[0]:7.3f}, {row[1]:7.3f}, {row[2]:7.3f})")


def exmp_ipf_color():
    """
    Demonstrate ipf_color() for standard FCC texture components along [001].
    """
    from upxo.xtalphy.crystal_orientation import ipf_color, FCC_TEXTURE_COMPONENTS

    print("=== exmp_ipf_color ===")
    print(f"  {'Component':<14s}  {'R':>6s}  {'G':>6s}  {'B':>6s}")
    for name, ea in FCC_TEXTURE_COMPONENTS.items():
        r, g, b = ipf_color(ea, sample_direction=[0., 0., 1.])
        print(f"  {name:<14s}  {r:.4f}  {g:.4f}  {b:.4f}")


def exmp_get_ks_rotations():
    """
    Demonstrate get_ks_rotations().
    Prints shape, verifies all returned matrices are proper rotations.
    """
    import numpy as np
    from upxo.xtalphy.crystal_orientation import get_ks_rotations

    ks = get_ks_rotations()

    dets  = np.linalg.det(ks)
    errs  = np.array([np.linalg.norm(R @ R.T - np.eye(3), ord='fro') for R in ks])

    print("=== exmp_get_ks_rotations ===")
    print(f"  Output shape            : {ks.shape}")
    print(f"  All det(R) ≈ +1         : {np.allclose(dets,  1.0, atol=1e-10)}")
    print(f"  All ||RR^T - I||_F < 1e-10 : {np.all(errs < 1e-10)}")
    print(f"  Max orthogonality error : {errs.max():.2e}")


def exmp_all():
    """
    Run every self-sufficient example in sequence.
    Suitable for a quick sanity-check of the module.
    """
    exmp_euler_bunge_to_matrix()
    print()
    exmp_normalize_euler_bunge()
    print()
    exmp_fcc_symmetrise_ori()
    print()
    exmp_ipf_color()
    print()
    exmp_get_ks_rotations()
    print()
    exmp_cubic_misorientation()
    print()
    exmp_grain_boundary_misorientation_distribution()
    print()
    exmp_gbmd_from_lfi()
