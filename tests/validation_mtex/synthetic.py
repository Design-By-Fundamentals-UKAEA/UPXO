"""Synthetic EBSD-like grain structures with known ground truth.

No real EBSD data, file path or measured orientation appears here: every grain and lamella is
placed by this module from a random seed. Orientations are built and returned in UPXO's stored
convention (the quaternion of g = R.T, the sample -> crystal matrix), the same convention
``EBSDReader`` output carries, so the arrays here can be passed directly to
``upxo.xtalphy.crystal_orientation`` functions.

This gives the mtex-validation suite a structure with a known answer: known grain count and
areas, and known parent/twin identity and area for a Sigma3 lamella, so the comparison harness
can be checked before it is trusted on real, noisy EBSD maps (where there is no known answer).
"""
import numpy as np
from scipy import ndimage

from . import _oracle as ref


def voronoi_label_field(nx: int, ny: int, n_seeds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A 2D grain label field (ids 1..n_seeds) by nearest-seed partition. Pure numpy.

    Returns (lfi, seed_positions). Pixel centres are at half-integer coordinates, matching the
    pixel convention used elsewhere in the library (pixel (0, 0) centred at (0.5, 0.5)).
    """
    rng = np.random.default_rng(seed)
    seeds = rng.uniform([0, 0], [nx, ny], size=(n_seeds, 2))
    xs, ys = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    d2 = ((pts[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    lfi = (d2.argmin(axis=1) + 1).reshape(ny, nx)
    return lfi.astype(np.int32), seeds


def random_cubic_orientations(n: int, seed: int, min_pair_disorientation_deg: float | None = None,
                               max_tries_per_grain: int = 400) -> np.ndarray:
    """n random Bunge Euler angle triples (degrees), shape (n, 3).

    With ``min_pair_disorientation_deg`` set, each new orientation is rejected and redrawn until
    it is at least that far (cubic disorientation) from every orientation already placed. This
    keeps neighbouring synthetic grains from landing within a grain-boundary detection threshold
    of one another by accident, which would merge them and break the "known grain count" premise.
    """
    rng = np.random.default_rng(seed)
    out, Rs = [], []
    tries = 0
    while len(out) < n and tries < max_tries_per_grain * n:
        tries += 1
        e = rng.uniform([0, 0, 0], [360, 180, 360])
        if min_pair_disorientation_deg is not None and Rs:
            R = ref.bunge_R(*e)[0]
            d = ref.disorientation_R(np.array(Rs), np.repeat(R[None], len(Rs), axis=0))
            if d.min() < min_pair_disorientation_deg:
                continue
            Rs.append(R)
        elif min_pair_disorientation_deg is not None:
            Rs.append(ref.bunge_R(*e)[0])
        out.append(e)
    if len(out) < n:
        raise RuntimeError(f"placed only {len(out)}/{n} grains >= {min_pair_disorientation_deg} "
                            f"deg apart in {max_tries_per_grain * n} tries")
    return np.array(out)


def per_pixel_stored_quats(lfi: np.ndarray, euler_by_id: np.ndarray) -> np.ndarray:
    """Per-pixel stored quaternion array (ny, nx, 4): the quaternion of g = R.T.

    ``euler_by_id`` is 1-indexed by grain id: row 0 is grain id 1. Pixels with id <= 0 get the
    identity quaternion.
    """
    R = ref.bunge_R(*euler_by_id.T)                      # (n, 3, 3), crystal -> sample
    g = R.transpose(0, 2, 1)
    q = ref.quat_wxyz_from_matrix(g)                      # (n, 4)
    q_with_pad = np.concatenate([[[1.0, 0.0, 0.0, 0.0]], q], axis=0)
    idx = np.clip(lfi, 0, len(q))
    return q_with_pad[idx]


def neighbour_graph(lfi: np.ndarray, connectivity: int = 4) -> dict[int, np.ndarray]:
    """{grain_id: ndarray of neighbouring grain ids}, from 4- or 8-connected pixel adjacency."""
    pairs: set[tuple[int, int]] = set()

    def add(a, b):
        m = (a > 0) & (b > 0) & (a != b)
        for x, y in zip(a[m].ravel(), b[m].ravel()):
            pairs.add((int(x), int(y)))
            pairs.add((int(y), int(x)))

    add(lfi[:, :-1], lfi[:, 1:])
    add(lfi[:-1, :], lfi[1:, :])
    if connectivity == 8:
        add(lfi[:-1, :-1], lfi[1:, 1:])
        add(lfi[:-1, 1:], lfi[1:, :-1])
    out: dict[int, set[int]] = {}
    for a, b in pairs:
        out.setdefault(a, set()).add(b)
    return {k: np.array(sorted(v), dtype=int) for k, v in out.items()}


def grain_areas(lfi: np.ndarray) -> dict[int, int]:
    """{grain_id: pixel count} over positive ids."""
    ids, counts = np.unique(lfi[lfi > 0], return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts)}


def insert_sigma3_twins(lfi: np.ndarray, euler_by_id: np.ndarray, host_ids, thickness_px: float,
                         seed: int, variant: str | int = "random"):
    """Carve one Sigma3 twin lamella out of each grain in ``host_ids``.

    The lamella is a slab of the given pixel width, perpendicular to the sample-frame projection
    of a {111} plane normal of the host, centred on the host's own pixel centroid and clipped to
    the host's own mask -- so the twin area is exact and known by construction, and the pipeline
    under test is never asked to agree with itself.

    Returns (lfi2, euler_by_id2, truth). ``truth`` maps each new twin id to a dict with
    ``parent`` (the host id), ``variant`` (0-3, the {111} plane used), ``n_pixels`` and
    ``trace_deg``. A host is skipped (no entry in ``truth``) if its chosen plane normal has no
    in-plane projection (no 2D trace), or the slab misses or swallows the whole grain.

    The host's pixel mask is a Voronoi cell and so convex, and a straight slab through a convex
    region is itself connected -- but it also splits what is left of the host into two separate
    pieces, on either side of the slab, with no path between them. Both pieces keep the host's
    id and orientation: this is deliberate, not a defect. A real twin lamella commonly crosses a
    grain from one boundary to the other, and a connectivity-based grain detector (the real kind,
    DefDAP's or mtex's) genuinely reports such a host as two separate grains of one orientation.
    Ground truth here is by label, not by connected region, so the host stays one entry in
    ``truth``'s parent field; a test that runs an actual grain detector must expect, and allow
    for, that split.
    """
    rng = np.random.default_rng(seed)
    lfi2 = lfi.copy()
    rows = list(euler_by_id)
    truth: dict[int, dict] = {}
    next_id = int(lfi.max()) + 1
    for host in host_ids:
        mask = lfi == host
        if not mask.any():
            continue
        e = euler_by_id[host - 1]
        Rp = ref.bunge_R(*e)[0]
        k = int(rng.integers(4)) if variant == "random" else int(variant)
        n_s = Rp @ ref.FOUR_111[k]
        d = np.array([n_s[0], n_s[1]])
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            continue
        d = d / norm
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        s = (xs - cx) * d[0] + (ys - cy) * d[1]
        band = np.abs(s) <= thickness_px / 2.0
        if not band.any() or band.all():
            continue
        sel = np.zeros_like(mask)
        sel[ys[band], xs[band]] = True

        # Defensive, not expected to trigger on a convex Voronoi cell: if the slab were ever
        # not itself connected, keep only its largest piece rather than record a twin with a
        # detached fragment of pixels.
        sel_labels, n_sel = ndimage.label(sel, structure=np.ones((3, 3)))
        if n_sel > 1:
            sizes = ndimage.sum(sel, sel_labels, index=range(1, n_sel + 1))
            sel = sel_labels == (1 + int(np.argmax(sizes)))
        n_px = int(sel.sum())
        if n_px == 0 or n_px == int(mask.sum()):
            continue
        lfi2[sel] = next_id
        Rt = Rp @ ref.twin_operator(ref.FOUR_111[k])
        rows.append(ref.euler_from_R(Rt)[0])
        phi = float(np.degrees(np.arctan2(n_s[0], -n_s[1])) % 180.0)
        truth[next_id] = {"parent": int(host), "variant": k, "n_pixels": n_px, "trace_deg": phi}
        next_id += 1
    return lfi2, np.array(rows), truth
