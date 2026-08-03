import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from upxo.viz.xphy.pole_figure import (
    PoleFigure,
    cubic_symmetry_operators,
    matrix_to_quat,
    plot_pole_figure_from_3d
)


@pytest.fixture(autouse=True)
def mock_show(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)


def test_cubic_symmetry_operators():
    ops = cubic_symmetry_operators()
    assert ops.shape == (24, 3, 3)
    # Check proper rotations (determinant is +1)
    for op in ops:
        assert np.isclose(np.linalg.det(op), 1.0)
        assert np.allclose(op @ op.T, np.eye(3))


def test_matrix_to_quat():
    # Identity matrix should map to [1, 0, 0, 0]
    id_mat = np.eye(3)[None, :, :]
    q = matrix_to_quat(id_mat)
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])

    # Rotation of 90 degrees about Z axis: Bunge (90, 0, 0)
    # R_stack from Bunge (90, 0, 0) in degrees:
    # Rz1 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rot_z = np.array([[[0.0, -1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]]])
    q_rot = matrix_to_quat(rot_z)
    # 90 degrees about Z has quaternion [cos(45), 0, 0, sin(45)] = [0.7071, 0, 0, 0.7071]
    assert np.allclose(np.abs(q_rot), [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])


def test_orientation_standardization():
    # Define an orientation as Euler angles (90, 35, 45) Bunge
    euler = np.array([[90.0, 35.0, 45.0]])
    pf_euler = PoleFigure(euler, convention='euler_deg')

    # Get the quaternion matching it
    q = pf_euler.quats
    pf_quat = PoleFigure(q, convention='quaternion')

    # Verify rotation matrices match
    assert np.allclose(pf_euler.R_stack, pf_quat.R_stack)


def test_symmetric_poles():
    pf = PoleFigure(np.array([[1.0, 0.0, 0.0, 0.0]]), convention='quaternion')
    
    # 100 should have 3 poles (antipodal deduplicated)
    poles_100 = pf._get_symmetric_poles('100')
    assert poles_100.shape == (3, 3)
    
    # 110 should have 6 poles (antipodal deduplicated)
    poles_110 = pf._get_symmetric_poles('110')
    assert poles_110.shape == (6, 3)

    # 111 should have 4 poles (antipodal deduplicated)
    poles_111 = pf._get_symmetric_poles('111')
    assert poles_111.shape == (4, 3)

    # Custom hkl seed [1, 2, 3] should generate 24 poles (proper rotations only, no antipodals in 24 ops)
    custom_poles = pf._get_symmetric_poles([1, 2, 3])
    assert custom_poles.shape == (24, 3)

    # Custom hkl seed as string '123' should behave exactly like the list [1, 2, 3]
    custom_poles_str = pf._get_symmetric_poles('123')
    assert custom_poles_str.shape == (24, 3)
    assert np.allclose(custom_poles, custom_poles_str)




def test_projection_stereographic_and_equal_area():
    # Identity orientation: poles projected down Z axis [0, 0, 1]
    pf = PoleFigure(np.array([[1.0, 0.0, 0.0, 0.0]]), convention='quaternion')
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    # Both mapped onto upper hemisphere
    X_stereo, Y_stereo, _ = pf._calculate_projection(poles, projection='stereographic', hemisphere='both_mapped')
    # Since z is mapped to 1.0, X, Y should be 0.0
    assert np.allclose(X_stereo, 0.0)
    assert np.allclose(Y_stereo, 0.0)

    # Lateral vector [1, 0, 0] should project to radius 1.0 (since z=0, d=1.0)
    poles_lat = np.array([[1.0, 0.0, 0.0]])
    X_lat, Y_lat, _ = pf._calculate_projection(poles_lat, projection='stereographic', hemisphere='both_mapped')
    assert np.isclose(np.sqrt(X_lat**2 + Y_lat**2), 1.0)

    # Equal area projection check: lateral vector [1, 0, 0] must project to exactly radius 1.0 (normalized)
    X_ea, Y_ea, _ = pf._calculate_projection(poles_lat, projection='equal_area', hemisphere='both_mapped')
    assert np.isclose(np.sqrt(X_ea**2 + Y_ea**2), 1.0)

    # Both separate hemisphere check: should project both poles inside the unit circle (radius <= 1.0)
    X_sep, Y_sep, _ = pf._calculate_projection(poles, projection='stereographic', hemisphere='both_separate')
    assert np.all(np.sqrt(X_sep**2 + Y_sep**2) <= 1.0)


def test_ipf_colors():
    # Test random quaternion IPF color generation
    q = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    pf = PoleFigure(q, convention='quaternion')
    colors = pf._calculate_ipf_colors()
    assert colors.shape == (2, 3)
    assert np.all(colors >= 0.0) and np.all(colors <= 1.0)


def test_plotting_smoke(tmp_path):
    # Standard smoke test to verify plot_scatter and plot_density methods complete without exception.
    euler = np.array([[0.0, 0.0, 0.0], [45.0, 45.0, 45.0]])
    gids = np.array([1, 2])
    pf = PoleFigure(euler, convention='euler_deg', gids=gids)

    # 1. Hemisphere coloring
    fig, ax = pf.plot_scatter(pole_family='100', color_by='hemisphere')
    plt.close(fig)

    # 2. Role coloring
    role_map = {1: 'host', 2: 'primary_twin'}
    fig, ax = pf.plot_scatter(pole_family='110', color_by='role', color_data=role_map)
    plt.close(fig)

    # 3. Size coloring
    size_map = {1: 1500, 2: 3000}
    fig, ax = pf.plot_scatter(pole_family='111', color_by='size', color_data=size_map)
    plt.close(fig)

    # 4. IPF coloring
    fig, ax = pf.plot_scatter(pole_family='100', color_by='ipf')
    plt.close(fig)

    # 5. Density plot
    fig, ax = pf.plot_density(pole_family='100', half_width_deg=10.0, grid_points=30)
    plt.close(fig)


def test_plot_from_3d_smoke():
    # Mock 3D label field
    lgi_3d = np.zeros((5, 5, 5), dtype=np.int32)
    lgi_3d[1:3, 1:3, 1:3] = 1
    lgi_3d[3:5, 3:5, 3:5] = 2

    # Mock quaternions dict
    quats_dict = {
        1: np.array([1.0, 0.0, 0.0, 0.0]),
        2: np.array([0.7071, 0.0, 0.0, 0.7071])
    }

    # Slice along Z axis (axis=0)
    fig, ax = plot_pole_figure_from_3d(lgi_3d, quats_dict, axis=0, slice_idx=2, pole_family='100')
    assert ax is not None
    plt.close(fig)


def test_mud_density_estimation():
    # Use random orientations (representing uniform background)
    from upxo.texOps.fcc import rand_uniform_so3
    quats = rand_uniform_so3(200)
    pf = PoleFigure(quats, convention='quaternion')
    
    # Calculate density for uniform texture with very small kappa (large half_width)
    fig, ax = pf.plot_density(pole_family='100', half_width_deg=90.0, grid_points=20)
    assert ax is not None
    plt.close(fig)
    
    # Test sharp texture with high density values
    quats_sharp = np.tile([1.0, 0.0, 0.0, 0.0], (50, 1))
    pf_sharp = PoleFigure(quats_sharp, convention='quaternion')
    fig_sharp, ax_sharp = pf_sharp.plot_density(pole_family='100', half_width_deg=5.0, grid_points=20)
    assert ax_sharp is not None
    plt.close(fig_sharp)
    
    # Test adaptive grid resolution ('auto')
    fig_auto, ax_auto = pf_sharp.plot_density(pole_family='100', half_width_deg=5.0, grid_points='auto')
    assert ax_auto is not None
    plt.close(fig_auto)
    
    # Test chunked calculation path (large dataset triggering the 10^7 threshold)
    quats_large = np.tile([1.0, 0.0, 0.0, 0.0], (4500, 1))
    pf_large = PoleFigure(quats_large, convention='quaternion')
    fig_chunk, ax_chunk = pf_large.plot_density(pole_family='111', half_width_deg=7.5, grid_points=30)
    assert ax_chunk is not None
    plt.close(fig_chunk)

    # Test scaling, clabel, and scatter overlay configurations
    fig_scale_log, ax_scale_log = pf_sharp.plot_density(pole_family='100', half_width_deg=5.0, grid_points=20, contour_scale='log', clabel=True)
    assert ax_scale_log is not None
    plt.close(fig_scale_log)

    fig_scale_sqrt, ax_scale_sqrt = pf_sharp.plot_density(pole_family='100', half_width_deg=5.0, grid_points=20, contour_scale='sqrt')
    assert ax_scale_sqrt is not None
    plt.close(fig_scale_sqrt)

    fig_overlay, ax_overlay = pf_sharp.plot_density(pole_family='100', half_width_deg=5.0, grid_points=20, overlay_scatter=True, scatter_color_by='ipf')
    assert ax_overlay is not None
    plt.close(fig_overlay)

    fig_overlay_hemi, ax_overlay_hemi = pf_sharp.plot_density(pole_family='100', half_width_deg=5.0, grid_points=20, overlay_scatter=True, scatter_color_by='hemisphere')
    assert ax_overlay_hemi is not None
    plt.close(fig_overlay_hemi)

    # Test 3D density plot with matplotlib backend
    fig_3d, ax_3d = pf_sharp.plot_density_3d(pole_family='100', grid_points=10, half_width_deg=5.0, backend='matplotlib')
    assert ax_3d is not None
    plt.close(fig_3d)

    # Test 3D density plot with pyvista backend
    res_3d = pf_sharp.plot_density_3d(pole_family='100', grid_points=10, half_width_deg=5.0, backend='pyvista')
    assert res_3d is not None

    # Test generalized symmetries in PoleFigure
    quats_sym = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    
    # Hexagonal
    pf_hex = PoleFigure(quats_sym, convention='quaternion', symmetry='hexagonal')
    assert pf_hex.symmetry_ops.shape == (12, 3, 3)
    poles_hex = pf_hex._get_symmetric_poles('100')
    assert len(poles_hex) > 0

    # Orthorhombic
    pf_orth = PoleFigure(quats_sym, convention='quaternion', symmetry='orthorhombic')
    assert pf_orth.symmetry_ops.shape == (4, 3, 3)

    # Triclinic
    pf_tri_sym = PoleFigure(quats_sym, convention='quaternion', symmetry='triclinic')
    assert pf_tri_sym.symmetry_ops.shape == (1, 3, 3)

    # Custom symmetry array
    custom_ops = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    pf_custom = PoleFigure(quats_sym, convention='quaternion', symmetry=custom_ops)
    assert pf_custom.symmetry_ops.shape == (1, 3, 3)

    # Test tops wrapper methods in upxo.xtalphy.texops
    from upxo.xtalphy.texops import tops
    t = tops()
    euler_angles_deg = np.array([[10.0, 20.0, 30.0], [45.0, 0.0, 45.0]])
    
    fig_tops, ax_tops = t.plot_pole_figure_density(euler_angles_deg, pole_family='100', grid_points=10, half_width_deg=10.0, symmetry='hexagonal')
    assert ax_tops is not None
    plt.close(fig_tops)

    fig_tops_3d, ax_tops_3d = t.plot_pole_figure_density_3d(euler_angles_deg, pole_family='100', grid_points=10, half_width_deg=10.0, backend='matplotlib', symmetry='orthorhombic')
    assert ax_tops_3d is not None
    plt.close(fig_tops_3d)





