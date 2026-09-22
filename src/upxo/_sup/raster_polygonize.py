"""Pure NumPy/Shapely polygonisation of labelled 2D rasters (no rasterio).

Pixel (row, col) occupies the square [col, col+1] x [row, row+1]; x is the
column index and y is the row index (identity transform), so vertices lie on
pixel corners. Connectivity is 4-connected: pixels touching only at a corner
belong to separate polygons.
"""
import numpy as np
from scipy.ndimage import find_objects
from shapely.geometry import Polygon


def _directed_edges(mask):
    """Boundary edges of a boolean mask, interior on the right (y down)."""
    p = np.pad(mask, 1, constant_values=False)
    m = p[1:-1, 1:-1]
    edges = []
    for cond, (ox0, oy0, ox1, oy1) in (
            (~p[:-2, 1:-1], (0, 0, 1, 0)),    # top:    (c, r) -> (c+1, r)
            (~p[1:-1, 2:], (1, 0, 1, 1)),     # right:  (c+1, r) -> (c+1, r+1)
            (~p[2:, 1:-1], (1, 1, 0, 1)),     # bottom: (c+1, r+1) -> (c, r+1)
            (~p[1:-1, :-2], (0, 1, 0, 0))):   # left:   (c, r+1) -> (c, r)
        r, c = np.nonzero(m & cond)
        edges.append(np.stack([c + ox0, r + oy0, c + ox1, r + oy1], axis=1))
    return np.concatenate(edges)


def _trace_rings(edges):
    """Chain directed edges into closed rings with collinear vertices dropped."""
    out = {}
    for x0, y0, x1, y1 in edges.tolist():
        out.setdefault((x0, y0), []).append((x1, y1))
    rings = []
    while out:
        start = next(iter(out))
        ring, cur, prev_dir = [start], start, None
        while True:
            nxt = out[cur]
            if len(nxt) == 1 or prev_dir is None:
                to = nxt[0]
            else:
                # Ambiguous vertex: turn right, which keeps diagonal
                # neighbours apart (4-connectivity).
                right = (-prev_dir[1], prev_dir[0])
                to = next((t for t in nxt if (t[0] - cur[0], t[1] - cur[1])
                           == right), nxt[0])
            nxt.remove(to)
            if not nxt:
                del out[cur]
            prev_dir = (to[0] - cur[0], to[1] - cur[1])
            cur = to
            if cur == start:
                break
            ring.append(cur)
        rings.extend(_drop_collinear(r) for r in _split_pinched(ring))
    return rings


def _split_pinched(ring):
    """Split a ring that revisits a vertex into simple rings.

    Where a grain's own diagonal pixels pinch a background region off, the
    right-turn trace passes through the pinch vertex twice; splitting there
    yields the exterior and a properly oriented hole.
    """
    path, pos, loops = [], {}, []
    for v in ring:
        if v in pos:
            i = pos[v]
            loop = path[i:]
            for u in loop[1:]:
                del pos[u]
            path = path[:i + 1]
            loops.append(loop)
        else:
            pos[v] = len(path)
            path.append(v)
    loops.append(path)
    return [lp for lp in loops if len(lp) >= 4]


def _drop_collinear(ring):
    """Remove vertices lying on a straight run; ring is open (not closed)."""
    n = len(ring)
    keep = []
    for i in range(n):
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            keep.append(b)
    return keep


def _signed_area(ring):
    x = np.array([p[0] for p in ring], dtype=float)
    y = np.array([p[1] for p in ring], dtype=float)
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def polygonize_mask(mask):
    """
    Polygonise a boolean mask.

    Returns
    -------
    list of dict
        GeoJSON-like ``{'type': 'Polygon', 'coordinates': [exterior, *holes]}``,
        one per 4-connected component, rings closed.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    rings = _trace_rings(_directed_edges(mask))
    # Exteriors have positive area, holes negative (interior on the right).
    exteriors = [r for r in rings if _signed_area(r) > 0]
    holes = [r for r in rings if _signed_area(r) < 0]
    ext_polys = [Polygon(r) for r in exteriors]
    ext_area = [p.area for p in ext_polys]
    ext_holes = [[] for _ in exteriors]
    for h in holes:
        pt = Polygon(h).representative_point()
        best = None
        for k, ep in enumerate(ext_polys):
            if ext_area[k] > abs(_signed_area(h)) and ep.contains(pt):
                if best is None or ext_area[k] < ext_area[best]:
                    best = k
        ext_holes[best].append(h)
    return [{'type': 'Polygon',
             'coordinates': [[*map(tuple, r), tuple(r[0])]
                             for r in [e, *hs]]}
            for e, hs in zip(exteriors, ext_holes)]


def _shift(geom, dx, dy):
    geom['coordinates'] = [[(x + dx, y + dy) for x, y in ring]
                           for ring in geom['coordinates']]
    return geom


def polygonize_labels(lgi, labels):
    """Return ``{label: [(geojson_dict, 1.0), ...]}`` for each label.

    Each label is polygonised inside its own bounding box, so the cost scales
    with grain size rather than image size. Same per-label structure as
    ``list(rasterio.features.shapes(mask, mask=mask))``.
    """
    lgi = np.asarray(lgi)
    slices = None
    if lgi.size and np.issubdtype(lgi.dtype, np.integer) and lgi.min() >= 0:
        slices = find_objects(lgi)
    out = {}
    for label in labels:
        if slices is not None and 0 < label <= len(slices) \
                and slices[label - 1] is not None:
            sr, sc = slices[label - 1]
            geoms = polygonize_mask(lgi[sr, sc] == label)
            out[label] = [(_shift(g, sc.start, sr.start), 1.0) for g in geoms]
        else:
            out[label] = [(g, 1.0) for g in polygonize_mask(lgi == label)]
    return out


def polygonize_label(lgi, label):
    """Single-label form of :func:`polygonize_labels`."""
    return polygonize_labels(lgi, [label])[label]
