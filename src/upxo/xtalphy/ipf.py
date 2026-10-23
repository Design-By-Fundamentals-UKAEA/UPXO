"""
ipf.py
======
Inverse-pole-figure colours for cubic (m-3m) crystals, reproducing the default key of mtex.

The key is white at the centre of the fundamental sector and carries the primary colours at its
vertices. A crystal direction is first brought into the sector ``z >= x >= y >= 0`` (vertices
[001], [101], [111]) by the 24 proper cubic operators and the inversion. Its polar coordinates
about the sector centre then give the colour: the radius (1 at the centre, 0 on the boundary,
measured along the great circle through the centre) sets the lightness, and the azimuth, equalised
so that the three vertices sit at equal angles, sets the hue, with mtex's hue correction. Two
crystallographically equivalent directions therefore have one colour.

The constants and the arithmetic follow mtex 1.0.0 (``ipfHSVKey`` / ``HSVDirectionKey``); the
tests in ``tests/xtalphy/test_ipf_vs_mtex.py`` carry colours computed by mtex.

Public API
----------
    reduce_to_cubic_sector(directions)   -> directions in the fundamental sector
    cubic_ipf_rgb(crystal_directions)    -> RGB in [0, 1]
    ipf_crystal_direction(R_or_quats, sample_direction, ...)
    CUBIC_SECTOR_VERTICES, equal_area_xy(), sector_mask()
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# The cubic sector and its key (mtex HSVDirectionKey on the Laue group m-3m)
# ---------------------------------------------------------------------------

_S2 = np.sqrt(2.0)
_S3 = np.sqrt(3.0)

#: Vertices of the fundamental sector, [001], [101], [111] (unit vectors).
CUBIC_SECTOR_VERTICES = np.array([[0.0, 0.0, 1.0], [1.0 / _S2, 0.0, 1.0 / _S2],
                                  [1.0 / _S3, 1.0 / _S3, 1.0 / _S3]])

#: Inward normals of the three boundary planes: x >= y, z >= x, y >= 0.
_NORMALS = np.array([[1.0 / _S2, -1.0 / _S2, 0.0], [-1.0 / _S2, 0.0, 1.0 / _S2], [0.0, 1.0, 0.0]])

_CENTER = CUBIC_SECTOR_VERTICES.sum(axis=0)
_CENTER = _CENTER / np.linalg.norm(_CENTER)          # the white centre of the key
_POLE = np.array([0.0, 0.0, 1.0])                    # azimuth zero is taken from here
_GRAY_VALUE = (0.2, 0.5)                             # gray of the white and of the black centre
_GRAY_GRADIENT = (0.5, 0.5)


def _angle(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), np.sum(u * v, axis=-1))


def reduce_to_cubic_sector(directions):
    """
    Bring crystal directions into the fundamental sector ``z >= x >= y >= 0``.

    Parameters
    ----------
    directions : array-like, shape (..., 3)

    Returns
    -------
    ndarray, shape (..., 3)
        Unit vectors; the absolute values of the components sorted so that the largest is z and
        the smallest is y.
    """
    d = np.asarray(directions, dtype=float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    a = np.sort(np.abs(d), axis=-1)[..., ::-1]
    return np.stack([a[..., 1], a[..., 2], a[..., 0]], axis=-1)


def sector_mask(directions, tol=1e-12):
    """True where a direction lies inside the fundamental sector."""
    d = np.asarray(directions, dtype=float)
    return np.all(d @ _NORMALS.T >= -tol, axis=-1)


def equal_area_xy(directions):
    """Equal-area projection of upper-hemisphere directions, radius 1 on the equator."""
    d = np.asarray(directions, dtype=float)
    f = np.sqrt(1.0 / (1.0 + d[..., 2]))
    return d[..., 0] * f, d[..., 1] * f


# ---------------------------------------------------------------------------
# Polar coordinates of the sector about its centre
# ---------------------------------------------------------------------------

def _polar_radius(xyz):
    """1 at the centre and 0 on the boundary, along the great circle through the centre."""
    w = np.cross(xyz, _CENTER)
    nw = np.linalg.norm(w, axis=-1)
    at_center = nw < 1e-10
    vxc = w / np.where(at_center, 1.0, nw)[:, None]
    r = np.full(len(xyz), np.inf)
    for n in _NORMALS:
        b = np.cross(vxc, n)
        nb = np.linalg.norm(b, axis=-1)
        ok = nb > 1e-10
        if not np.any(ok):
            continue
        bc = b[ok] / nb[ok][:, None]
        d = (np.pi - _angle(xyz[ok], bc)) / (np.pi - _angle(_CENTER, bc))
        d = np.where(np.isnan(d), 1.0, d)
        r[ok] = np.minimum(r[ok], d)
    r = np.where(np.isfinite(r), r, 1.0)
    return np.where(at_center, 1.0, r)


def _tangent_at_center():
    t = _POLE - (_POLE @ _CENTER) * _CENTER
    return t / np.linalg.norm(t)


def _azimuth_about_center(xyz):
    rx = _tangent_at_center()
    ry = np.cross(_CENTER, rx)
    dv = xyz - _CENTER
    small = np.linalg.norm(dv, axis=-1) < 1e-12
    a = np.mod(np.arctan2(dv @ ry, dv @ rx), 2.0 * np.pi)
    return np.where(small, 0.0, a)


@lru_cache(maxsize=1)
def _equalisation_table(steps=1000):
    """Azimuth as it turns about the centre against its equalised value (vertices at 0, 120, 240 deg)."""
    rx = _tangent_at_center()
    at = np.mod(_azimuth_about_center(CUBIC_SECTOR_VERTICES), 2.0 * np.pi)
    phi = np.unique(np.concatenate([np.linspace(0.0, 2.0 * np.pi, steps + 1), at]))
    n1 = np.cross(rx, _CENTER)
    n1 = n1 / np.linalg.norm(n1)
    # n1 turned about the centre by phi (Rodrigues)
    c = _CENTER
    cosp, sinp = np.cos(phi)[:, None], np.sin(phi)[:, None]
    n = n1 * cosp + np.cross(c, n1) * sinp + c * (c @ n1) * (1.0 - cosp)
    omega = np.full(len(phi), np.pi)
    for m in _NORMALS:
        cr = np.cross(m, n)
        ncr = np.linalg.norm(cr, axis=-1)
        ok = ncr > 1e-12
        omega[ok] = np.minimum(omega[ok], _angle(cr[ok] / ncr[ok][:, None], c))
    seg = (omega[:-1] + omega[1:]) / 2.0 * np.diff(phi)
    arc = np.cumsum(np.isin(phi[:-1], at)) % 3          # an interval belongs to the arc of the vertices passed
    for k in range(3):
        w = np.sum(seg[arc == k])
        if w > 0:
            seg[arc == k] /= 3.0 * w
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / np.sum(seg) * 2.0 * np.pi
    return phi, cum


# ---------------------------------------------------------------------------
# Colour arithmetic (mtex plotting/directionkeys.py)
# ---------------------------------------------------------------------------

def _hsl2hsv(h, s, l):
    ll = 2.0 * np.asarray(l, dtype=float)
    ss = np.where(ll <= 1, s * ll, s * (2.0 - ll))
    v = (ll + ss) / 2.0
    sv = np.where(ll + ss < 1e-12, 0.0, 2.0 * ss / np.where(ll + ss < 1e-12, 1.0, ll + ss))
    return h, sv, v


def _hsv2rgb(h, s, v):
    h, s, v = np.broadcast_arrays(np.asarray(h, dtype=float), np.asarray(s, dtype=float),
                                  np.asarray(v, dtype=float))
    h6 = 6.0 * np.mod(h, 1.0)
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


@lru_cache(maxsize=1)
def _hue_correction(steps=1000):
    """Plain HSV gives yellow and cyan more of the circle than red, green and blue; the hue is
    passed through the cumulative of three Gaussians on the primaries."""
    z = np.linspace(0.0, 1.0, steps)
    f = 0.5 + sum(np.exp(-200.0 * (np.mod(z - c + 0.5, 1.0) - 0.5) ** 2) for c in (0.0, 1.0 / 3.0, 2.0 / 3.0))
    return np.cumsum(f / np.sum(f))


def _corrected_hue(h):
    table = _hue_correction()
    n = len(table)
    t = np.mod(np.asarray(h, dtype=float), 1.0) * (n - 1)
    i = np.clip(np.floor(t).astype(int), 0, n - 2)
    return table[i] + (t - i) * (table[i + 1] - table[i])


def _angle_radius_color(hue, radius, gray):
    hue, radius, gray = np.broadcast_arrays(np.asarray(hue, dtype=float), np.asarray(radius, dtype=float),
                                            np.asarray(gray, dtype=float))
    L = (radius - 0.5) * gray + 0.5
    d = 1.0 - np.abs(2.0 * L - 1.0)
    S = np.where(d < 1e-12, 0.0, gray * (1.0 - np.abs(2.0 * radius - 1.0)) / np.where(d < 1e-12, 1.0, d))
    h, s, v = _hsl2hsv(hue, S, L)
    return _hsv2rgb(_corrected_hue(h), s, v)


def _sphere_color(radius, rho):
    """Colour of a point of the colour sphere: radius 1 at the white centre, 0 at the black one."""
    radius, rho = np.broadcast_arrays(np.asarray(radius, dtype=float), np.asarray(rho, dtype=float))
    theta = radius * np.pi
    white = theta > np.pi / 2.0
    g = np.where(white, _GRAY_GRADIENT[0], _GRAY_GRADIENT[1])
    t = (2.0 * g * theta / np.pi + (1.0 - g) * (1.0 - np.cos(theta))) / 2.0
    gv = np.where(white, _GRAY_VALUE[0], _GRAY_VALUE[1])
    gr = 1.0 - 2.0 * gv * np.abs(t - 0.5)
    return _angle_radius_color(np.mod(rho / (2.0 * np.pi), 1.0), t, gr)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def cubic_ipf_rgb(crystal_directions):
    """
    mtex-default IPF colour of crystal directions (cubic m-3m).

    Parameters
    ----------
    crystal_directions : array-like, shape (..., 3)
        The direction of the chosen sample axis in crystal coordinates. Signs and the 24 cubic
        operators do not change the colour. Non-finite or zero vectors give the colour of the
        black centre.

    Returns
    -------
    ndarray, shape (..., 3), float64 in [0, 1]
    """
    d = np.asarray(crystal_directions, dtype=float)
    shape = d.shape[:-1]
    flat = d.reshape(-1, 3)
    norm = np.linalg.norm(flat, axis=-1)
    finite = np.all(np.isfinite(flat), axis=-1) & (norm > 0)
    unit = np.where(finite[:, None], flat / np.where(finite, norm, 1.0)[:, None], _CENTER)
    hs = reduce_to_cubic_sector(unit)
    r = _polar_radius(hs)
    rho = _azimuth_about_center(hs)
    phi, cum = _equalisation_table()
    rho_eq = np.interp(np.mod(rho, 2.0 * np.pi), phi, cum)
    radius = 0.5 + r / 2.0                              # no fold for m-3m: the whole sector is the white half
    rgb = _sphere_color(np.where(finite, radius, 0.0), np.where(finite, rho_eq, 0.0))
    return np.clip(rgb, 0.0, 1.0).reshape(shape + (3,))


def ipf_crystal_direction(orientation, sample_direction=(0.0, 0.0, 1.0), convention="defdap"):
    """
    Crystal coordinates of a sample direction for a stack of orientations.

    Parameters
    ----------
    orientation : ndarray
        Quaternions (N, 4) as [w, x, y, z], or rotation matrices (N, 3, 3) with
        ``convention='matrix'`` (the crystal -> sample matrix, ``euler_bunge_to_matrix``).
    sample_direction : array-like, shape (3,)
    convention : {'defdap', 'crystal_to_sample', 'matrix'}
        ``'defdap'``: quaternions as DefDAP stores them (``EBSDReader.quat_ebsd``,
        ``grain_avg_quats``), the quaternion of the Bunge g matrix, whose rotation matrix maps
        sample coordinates to crystal coordinates. ``'crystal_to_sample'``: the quaternion of the
        crystal -> sample rotation. ``'matrix'``: crystal -> sample matrices.

    Returns
    -------
    ndarray, shape (N, 3)
    """
    sd = np.asarray(sample_direction, dtype=float)
    sd = sd / np.linalg.norm(sd)
    o = np.asarray(orientation, dtype=float)
    if convention == "matrix":
        R = o.reshape(-1, 3, 3)
        return np.einsum("nji,j->ni", R, sd)             # R.T @ sd
    q = np.atleast_2d(o)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    M = np.empty((len(q), 3, 3))
    M[:, 0, 0] = 1 - 2 * (y * y + z * z)
    M[:, 0, 1] = 2 * (x * y - z * w)
    M[:, 0, 2] = 2 * (x * z + y * w)
    M[:, 1, 0] = 2 * (x * y + z * w)
    M[:, 1, 1] = 1 - 2 * (x * x + z * z)
    M[:, 1, 2] = 2 * (y * z - x * w)
    M[:, 2, 0] = 2 * (x * z - y * w)
    M[:, 2, 1] = 2 * (y * z + x * w)
    M[:, 2, 2] = 1 - 2 * (x * x + y * y)
    if convention == "defdap":
        return np.einsum("nij,j->ni", M, sd)              # M is the Bunge g matrix
    if convention == "crystal_to_sample":
        return np.einsum("nji,j->ni", M, sd)              # M.T @ sd
    raise ValueError("convention must be 'defdap', 'crystal_to_sample' or 'matrix'")
