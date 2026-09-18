"""Tests for UPXO 2D Voronoi tessellation and high-fidelity features.

Tests cover:
- voronoi_tessellation_2d engine (Voronoi, Laguerre/Power diagrams, PBC, CVT, perturbation)
- gtess2d factory methods: from_seed_points, from_mpoint2d, from_shapely_mulpolygon
- geotess2d topology, neighbor discovery, boundary filtering, and seed generation
- Preservation of legacy call patterns and backward compatibility
"""
import numpy as np
import pytest
from shapely.geometry import Polygon, MultiPolygon, box

from upxo.pxtal.vortess2d import gtess2d
from upxo.pxtal.geotess import geotess2d
from upxo.geoEntities.mulpoint2d import MPoint2d
from upxo.pxtal.voronoi_tessellation_2d import (
    coerce_bounds_2d,
    compute_standard_voronoi_2d,
    compute_power_diagram_2d,
    cvt_relax_2d,
    perturb_interfaces_2d,
    generate_voronoi_2d,
)


# ==============================================================================
# Engine Tests: voronoi_tessellation_2d
# ==============================================================================

def test_coerce_bounds_2d_formats():
    """Test standardizing various bounds formats."""
    # 4-element list/tuple [xmin, xmax, ymin, ymax]
    b1 = coerce_bounds_2d([0, 10, 0, 20])
    assert b1 == [[0.0, 10.0], [0.0, 20.0]]

    # 2x2 list [[xmin, xmax], [ymin, ymax]]
    b2 = coerce_bounds_2d([[0, 10], [0, 20]])
    assert b2 == [[0.0, 10.0], [0.0, 20.0]]

    # Dict format {'xbound': [0, 10], 'ybound': [0, 20]}
    b3 = coerce_bounds_2d({'xbound': [0, 10], 'ybound': [0, 20]})
    assert b3 == [[0.0, 10.0], [0.0, 20.0]]

    # Inferred from seeds
    seeds = np.array([[2, 3], [8, 17]], dtype=float)
    b4 = coerce_bounds_2d(None, seeds=seeds)
    assert b4[0][0] <= 2.0 and b4[0][1] >= 8.0
    assert b4[1][0] <= 3.0 and b4[1][1] >= 17.0


def test_compute_standard_voronoi_2d_area_coverage():
    """Verify that Voronoi cells partition the domain without gap or overlap."""
    bounds = (0.0, 100.0, 0.0, 100.0)
    np.random.seed(42)
    seeds = np.random.uniform(10, 90, size=(16, 2))

    cells = compute_standard_voronoi_2d(seeds, bounds=bounds, periodic=(False, False))
    assert len(cells) == len(seeds)

    total_area = sum(p.area for p in cells.values())
    expected_area = 100.0 * 100.0
    assert np.isclose(total_area, expected_area, rtol=1e-3)


def test_compute_standard_voronoi_2d_periodic():
    """Verify Voronoi generation with Periodic Boundary Conditions."""
    bounds = (0.0, 100.0, 0.0, 100.0)
    # Put seeds near boundaries
    seeds = np.array([
        [1.0, 50.0],
        [99.0, 50.0],
        [50.0, 50.0],
        [50.0, 1.0],
        [50.0, 99.0],
    ], dtype=float)

    cells = compute_standard_voronoi_2d(seeds, bounds=bounds, periodic=(True, True))
    assert len(cells) == len(seeds)
    total_area = sum(p.area for p in cells.values())
    assert np.isclose(total_area, 10000.0, rtol=1e-3)


def test_compute_power_diagram_2d_weights():
    """Verify Laguerre tessellation with seed weights."""
    bounds = (0.0, 50.0, 0.0, 50.0)
    seeds = np.array([
        [20.0, 25.0],
        [30.0, 25.0],
    ], dtype=float)

    # First equal weights -> equal split at x = 25
    cells_equal = compute_power_diagram_2d(seeds, weights=[1.0, 1.0], bounds=bounds)
    assert np.isclose(cells_equal[0].area, cells_equal[1].area, rtol=1e-2)

    # Second seed has much larger weight -> cell 1 expands, cell 0 shrinks
    cells_weighted = compute_power_diagram_2d(seeds, weights=[1.0, 100.0], bounds=bounds)
    assert cells_weighted[1].area > cells_weighted[0].area


def test_cvt_relaxation_2d():
    """Verify Centroidal Voronoi Tessellation via Lloyd's relaxation."""
    bounds = (0.0, 100.0, 0.0, 100.0)
    np.random.seed(123)
    initial_seeds = np.random.uniform(5, 95, size=(12, 2))

    relaxed_seeds = cvt_relax_2d(initial_seeds, bounds=bounds, max_iter=5)
    assert relaxed_seeds.shape == initial_seeds.shape
    # Seeds should have moved towards more uniform distribution
    assert not np.allclose(relaxed_seeds, initial_seeds)
    assert np.all(relaxed_seeds[:, 0] >= bounds[0])
    assert np.all(relaxed_seeds[:, 0] <= bounds[1])
    assert np.all(relaxed_seeds[:, 1] >= bounds[2])
    assert np.all(relaxed_seeds[:, 1] <= bounds[3])


def test_perturb_interfaces_2d():
    """Verify non-linear grain boundary perturbation preserves total area."""
    bounds = (0.0, 50.0, 0.0, 50.0)
    seeds = np.array([
        [15.0, 15.0],
        [35.0, 15.0],
        [15.0, 35.0],
        [35.0, 35.0],
    ], dtype=float)
    cells = compute_standard_voronoi_2d(seeds, bounds=bounds)
    total_orig_area = sum(p.area for p in cells.values())

    # Test dict input/output
    perturbed_dict = perturb_interfaces_2d(cells, bounds=bounds, factor=0.04, seed=42)
    assert isinstance(perturbed_dict, dict)
    assert len(perturbed_dict) == len(seeds)
    total_perturbed_area = sum(p.area for p in perturbed_dict.values())
    assert np.isclose(total_perturbed_area, total_orig_area, rtol=1e-2)

    # Test MultiPolygon input/output
    mpoly_in = MultiPolygon(list(cells.values()))
    mpoly_out = perturb_interfaces_2d(mpoly_in, bounds=bounds, factor=0.04, seed=42)
    assert isinstance(mpoly_out, MultiPolygon)
    assert len(mpoly_out.geoms) == len(seeds)
    assert np.isclose(mpoly_out.area, total_orig_area, rtol=1e-2)


# ==============================================================================
# gtess2d Tests
# ==============================================================================

def test_gtess2d_from_seed_points_basic():
    """Create gtess2d from raw coordinate array."""
    seeds = np.array([
        [10.0, 10.0],
        [30.0, 30.0],
        [50.0, 50.0],
    ], dtype=float)
    bounds = [[0.0, 60.0], [0.0, 60.0]]

    tess = gtess2d.from_seed_points(seeds, bounds=bounds)
    assert isinstance(tess, gtess2d)
    assert tess.ninst == 1
    assert tess.seeds is not None
    assert 1 in tess.pxtals
    assert len(tess.pxtals[1].geoms) == len(seeds)


def test_gtess2d_from_mpoint2d():
    """Create gtess2d from MPoint2d instance."""
    coords = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 15.0]])
    mpt = MPoint2d.from_coords(coords)
    bounds = [0.0, 50.0, 0.0, 50.0]

    tess = gtess2d.from_mpoint2d(mpt, bounds=bounds)
    assert isinstance(tess, gtess2d)
    assert len(tess.pxtals[1].geoms) == 3


def test_gtess2d_periodic():
    """Create gtess2d with periodic boundary conditions."""
    seeds = np.array([
        [5.0, 5.0],
        [25.0, 25.0],
    ])
    bounds = [0.0, 30.0, 0.0, 30.0]

    tess = gtess2d.from_seed_points(seeds, bounds=bounds, periodic=(True, True))
    assert tess.info['uinputs']['periodic'] == (True, True)
    assert tess.seeds.n == 2
    # Total domain area is preserved exactly (30 x 30 = 900)
    assert np.isclose(tess.pxtals[1].area, 900.0, rtol=1e-3)


def test_gtess2d_weighted_laguerre():
    """Create gtess2d with Laguerre weights."""
    seeds = np.array([
        [15.0, 20.0],
        [25.0, 20.0],
    ])
    bounds = [0.0, 40.0, 0.0, 40.0]
    weights = [2.0, 50.0]

    tess = gtess2d.from_seed_points(seeds, bounds=bounds, weights=weights)
    assert tess.info['uinputs']['weights'] == weights
    poly_0 = tess.pxtals[1].geoms[0]
    poly_1 = tess.pxtals[1].geoms[1]
    assert poly_1.area > poly_0.area


def test_gtess2d_with_cvt():
    """Create gtess2d with Lloyd CVT relaxation."""
    seeds = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [40.0, 40.0],
    ])
    bounds = [0.0, 50.0, 0.0, 50.0]

    tess = gtess2d.from_seed_points(seeds, bounds=bounds, cvt_iterations=3)
    assert tess.info['uinputs']['cvt_iterations'] == 3
    assert len(tess.pxtals[1].geoms) == 3


def test_gtess2d_with_perturbation():
    """Create gtess2d with boundary perturbation."""
    seeds = np.array([
        [10.0, 10.0],
        [30.0, 10.0],
        [20.0, 30.0],
    ])
    bounds = [0.0, 40.0, 0.0, 40.0]

    tess = gtess2d.from_seed_points(seeds, bounds=bounds, perturb_factor=0.03)
    assert tess.info['uinputs']['perturb_factor'] == 0.03
    assert len(tess.pxtals[1].geoms) == 3


def test_gtess2d_from_shapely_mulpolygon():
    """Create gtess2d directly from a Shapely MultiPolygon."""
    p1 = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    p2 = Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])
    mpoly = MultiPolygon([p1, p2])

    tess = gtess2d.from_shapely_mulpolygon(mpoly, bounds=[0, 10, 0, 10])
    assert isinstance(tess, gtess2d)
    assert len(tess.pxtals[1].geoms) == 2
    assert tess.bounds['xbound'] == [0.0, 10.0]
    assert tess.bounds['ybound'] == [0.0, 10.0]


def test_gtess2d_legacy_instantiation():
    """Ensure existing legacy instantiation (sp_input='gen', gr_tech='pds') works seamlessly."""
    tess = gtess2d.from_seed_points(nsp=8, xbound=[0, 50], ybound=[0, 50], gr_tech='pds', smp_tech='bridson1')
    assert isinstance(tess, gtess2d)
    assert 1 in tess.pxtals
    assert len(tess.pxtals[1].geoms) > 0


def test_gtess2d_legacy_load_sp():
    """Test sp_input='load' path with pre-generated seeds."""
    seeds = np.array([[10, 10], [20, 20], [30, 30]], dtype=float)
    tess = gtess2d.from_seed_points(sp_input='load', sp_in=seeds, xbound=[0, 40], ybound=[0, 40])
    assert isinstance(tess, gtess2d)
    assert len(tess.pxtals[1].geoms) == 3


# ==============================================================================
# geotess2d Tests
# ==============================================================================

def test_geotess2d_container_protocol():
    """Test container methods of geotess2d (__len__, __iter__, __getitem__, __repr__)."""
    geo = geotess2d()
    assert len(geo) == 0
    assert repr(geo).startswith("<geotess2d:")

    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    geo.xtals = [p1, p2]
    geo.gid = [101, 102]
    assert len(geo) == 2
    assert geo[0] == p1
    assert geo[1] == p2

    # iteration
    polys = list(geo)
    assert len(polys) == 2


def test_geotess2d_make_seeds():
    """Test random and Poisson-disc seed generation on geotess2d."""
    geo = geotess2d()
    bounds = [0.0, 50.0, 0.0, 50.0]

    # Random seeds
    s_rand = geo.make_seeds_random(nsp=20, bounds=bounds)
    assert isinstance(s_rand, MPoint2d)
    assert s_rand.n == 20

    # Poisson disc seeds
    s_pds = geo.make_seeds_pdisc(char_length=10.0, bounds=bounds)
    assert isinstance(s_pds, MPoint2d)
    assert s_pds.n > 0


def test_geotess2d_neighbour_topology():
    """Test neighbour connectivity analysis on geotess2d."""
    geo = geotess2d()
    geo.bounds = [0.0, 3.0, 0.0, 1.0]
    p0 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p1 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    p2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])

    geo.xtals = [p0, p1, p2]
    geo.gid = [0, 1, 2]

    adj = geo.find_neighbours()
    assert 1 in adj[0]
    assert 2 not in adj[0]
    assert 0 in adj[1] and 2 in adj[1]
    assert 1 in adj[2] and 0 not in adj[2]

    # First and second nearest neighbours
    n1 = geo.find_first_nearest_neighbours(gid=0)
    assert n1 == [1]
    n2 = geo.find_second_nearest_neighbours(gid=0)
    assert n2 == [2]


def test_geotess2d_filter_boundary_and_internal_grains():
    """Test boundary vs internal grain identification."""
    geo = geotess2d()
    geo.bounds = [0.0, 3.0, 0.0, 3.0]
    # Center grain (internal)
    p_center = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    # Corner grain (boundary)
    p_corner = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    geo.xtals = [p_center, p_corner]
    geo.gid = [0, 1]

    b_grains = geo.filter_boundary_grains()
    assert 1 in b_grains
    assert 0 not in b_grains

    i_grains = geo.filter_internal_grains()
    assert 0 in i_grains
    assert 1 not in i_grains


def test_gtess2d_conformal_meshing_compatibility():
    """Verify that a gtess2d polycrystal can be directly meshed by confMesh2dGMSH."""
    pytest.importorskip("gmsh")
    from upxo.meshing.conformal_mesher2d import confMesh2dGMSH

    seeds = np.array([
        [15.0, 15.0],
        [35.0, 15.0],
        [25.0, 35.0],
    ])
    bounds = [0.0, 50.0, 0.0, 50.0]
    tess = gtess2d.from_seed_points(seeds, bounds=bounds)

    m = confMesh2dGMSH.from_geometric_pxtal(
        pxtal=tess.pxtals[1], xbound=(0, 50), ybound=(0, 50)
    )
    m.femesh_gmsh(mesh_size_gb=5.0, mesh_size_bulk=8.0, recombine_to_quads=False)
    m.form_elsets_gmsh()
    assert len(m.elsets) == 3
    pts, lines, triangles, quads = m.get_mesh_geometry()
    assert triangles is not None and len(triangles) > 0

