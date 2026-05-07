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
