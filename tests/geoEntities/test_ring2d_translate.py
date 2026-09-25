"""ring2d.translate (in place) and ring2d.translated_copy (a shifted, independent copy)."""
import numpy as np

from upxo.geoEntities.mulsline2d import MSline2d, ring2d
from upxo.geoEntities.sline2d import Sline2d as sl2d


def _msline(coords):
    lines = [sl2d(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
             for i in range(len(coords) - 1)]
    return MSline2d.from_lines(lines, close=False)


def _two_rings_sharing_a_wall():
    shared = _msline([(2, 0), (2, 1), (2, 2)])
    ring_a = ring2d([_msline([(0, 0), (2, 0)]), shared, _msline([(2, 2), (0, 2), (0, 0)])],
                    segids=[0, 1, 2], segflips=[False, False, False])
    ring_b = ring2d([_msline([(2, 0), (4, 0)]), _msline([(4, 0), (4, 2), (2, 2)]), shared],
                    segids=[0, 1, 2], segflips=[False, False, False])
    return ring_a, ring_b, shared


def _all_points(ring):
    return {id(p) for seg in ring.segments for ln in seg.lines for p in (ln.pnta, ln.pntb)}


def _stack(ring):
    return np.vstack([s.get_node_coords() for s in ring.segments])


def test_translate_moves_every_point_and_syncs_the_line_coordinates():
    ring_a, _, _ = _two_rings_sharing_a_wall()
    before = _stack(ring_a)
    ring_a.translate(3.0, -1.0)
    np.testing.assert_allclose(_stack(ring_a), before + [3.0, -1.0])
    for seg in ring_a.segments:
        for ln in seg.lines:
            assert (ln.x0, ln.y0, ln.x1, ln.y1) == (ln.pnta.x, ln.pnta.y, ln.pntb.x, ln.pntb.y)


def test_translate_moves_a_shared_point_once_when_given_a_common_seen_set():
    ring_a, ring_b, shared = _two_rings_sharing_a_wall()
    seen = ring_a.translate(1.0, 0.0)
    ring_b.translate(1.0, 0.0, seen)
    np.testing.assert_allclose(shared.get_node_coords(), [[3, 0], [3, 1], [3, 2]])


def test_translated_copy_shifts_the_copy_and_leaves_the_original_untouched():
    ring_a, _, _ = _two_rings_sharing_a_wall()
    before = _stack(ring_a).copy()
    copy = ring_a.translated_copy(10.0, 5.0, {})
    np.testing.assert_allclose(_stack(copy), before + [10.0, 5.0])
    np.testing.assert_array_equal(_stack(ring_a), before)


def test_translated_copy_shares_no_point_or_segment_with_the_original():
    ring_a, _, _ = _two_rings_sharing_a_wall()
    copy = ring_a.translated_copy(1.0, 1.0, {})
    assert _all_points(copy).isdisjoint(_all_points(ring_a))
    assert all(c is not o for c, o in zip(copy.segments, ring_a.segments))
    assert copy.segflips == ring_a.segflips and copy.segflips is not ring_a.segflips


def test_translated_copy_keeps_a_shared_wall_shared_between_copies():
    ring_a, ring_b, shared = _two_rings_sharing_a_wall()
    copies = {}
    copy_a = ring_a.translated_copy(2.0, 0.0, copies)
    copy_b = ring_b.translated_copy(2.0, 0.0, copies)
    shared_copy = copy_a.segments[1]
    assert copy_b.segments[2] is shared_copy
    assert shared_copy is not shared
    np.testing.assert_allclose(shared_copy.get_node_coords(), [[4, 0], [4, 1], [4, 2]])
    np.testing.assert_allclose(shared.get_node_coords(), [[2, 0], [2, 1], [2, 2]])


def test_translated_copy_of_a_closed_loop_stays_closed():
    loop = MSline2d.by_coords([(0, 0), (2, 0), (2, 2), (0, 2)], close=True)
    ring = ring2d([loop], segids=[0], segflips=[False])
    copy = ring.translated_copy(5.0, 5.0, {})
    coords = copy.segments[0].get_node_coords()
    assert np.allclose(coords[0], coords[-1]) and np.allclose(coords[0], [5, 5])
