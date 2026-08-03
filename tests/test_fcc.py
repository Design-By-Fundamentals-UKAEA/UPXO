import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from upxo.texOps.fcc import FCCTexture, fcc_symmetrise_ori, rand_uniform_so3
from upxo.viz.xphy.pole_figure import PoleFigure


@pytest.fixture(autouse=True)
def mock_show(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)



def test_fcc_texture_init():
    # Valid init
    tc = FCCTexture({"copper": 0.4, "brass": 0.3})
    assert tc.tc_fractions["copper"] == 0.4
    assert tc.tc_fractions["brass"] == 0.3
    
    # Sum of fractions exceeds 1.0 -> raise ValueError
    with pytest.raises(ValueError):
        FCCTexture({"copper": 0.6, "brass": 0.5})


def test_alloc_counts():
    tc = FCCTexture({"copper": 0.45, "brass": 0.30})
    counts = tc._allocate_counts(100)
    
    # 45% Copper -> 45 orientations
    assert counts["copper"] == 45
    # 30% Brass -> 30 orientations
    assert counts["brass"] == 30
    # Remaining 25% allocated to random background
    assert counts["random"] == 25
    
    # Total count sum should equal exactly N
    assert sum(counts.values()) == 100


def test_fcc_symmetrise_ori():
    # Goss orientation (90, 90, 45) Bunge
    goss_equiv = fcc_symmetrise_ori((90.0, 90.0, 45.0))
    # Goss component should generate exactly 24 distinct symmetric orientations (proper rotations)
    assert goss_equiv.shape == (24, 3)

    # Cube orientation (0, 0, 0) should also generate 24 distinct symmetric orientations
    cube_equiv = fcc_symmetrise_ori((0.0, 0.0, 0.0))
    assert cube_equiv.shape == (24, 3)



def test_generate_euler():
    tc = FCCTexture({"copper": 0.45, "brass": 0.30}, spread=5.0)
    N = 200
    eulers = tc.generate_euler(N)
    
    assert eulers.shape == (N, 3)
    # Check canonical ranges
    assert np.all(eulers[:, 0] >= 0.0) and np.all(eulers[:, 0] <= 360.0)
    assert np.all(eulers[:, 1] >= 0.0) and np.all(eulers[:, 1] <= 180.0)
    assert np.all(eulers[:, 2] >= 0.0) and np.all(eulers[:, 2] <= 360.0)


def test_generate_quaternions():
    tc = FCCTexture({"goss": 0.5}, spread=3.0)
    N = 100
    quats = tc.generate_quaternions(N)
    
    assert quats.shape == (N, 4)
    # Check w >= 0
    assert np.all(quats[:, 0] >= -1e-8)
    # Check normalized
    norms = np.linalg.norm(quats, axis=1)
    assert np.allclose(norms, 1.0)


def test_generate_matrices():
    tc = FCCTexture({"cube": 0.25, "s": 0.25})
    N = 50
    mats = tc.generate_matrices(N)
    
    assert mats.shape == (N, 3, 3)
    for R in mats:
        # Check determinant is +1
        assert np.isclose(np.linalg.det(R), 1.0)
        # Check orthogonality
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_rand_uniform_so3():
    q = rand_uniform_so3(200)
    assert q.shape == (200, 4)
    norms = np.linalg.norm(q, axis=1)
    assert np.allclose(norms, 1.0)


def test_fcc_texture_pole_figure_integration():
    # Integrate generated orientations with visual xphy PoleFigure
    tc = FCCTexture({"copper": 0.5, "brass": 0.3}, spread=4.0)
    quats = tc.generate_quaternions(300)
    
    pf = PoleFigure(quats, convention='quaternion')
    fig, ax = pf.plot_scatter(pole_family='111', color_by='ipf')
    import matplotlib.pyplot as plt
    assert ax is not None
    plt.close(fig)


def test_sample_symmetry_group_closure():
    # Master switch False -> group size 1 (Identity only)
    tc_none = FCCTexture({"copper": 1.0}, apply_sample_symmetries=False)
    gp_none = tc_none._compute_sample_symmetry_group()
    assert len(gp_none) == 1
    assert np.allclose(gp_none[0], np.eye(3))
    
    # Only TD generator active -> group size 2 (Identity, TD)
    tc_td = FCCTexture({"copper": 1.0}, apply_sample_symmetries=True, use_rd=False, use_td=True, use_nd=False)
    gp_td = tc_td._compute_sample_symmetry_group()
    assert len(gp_td) == 2
    
    # RD and TD generators active -> group size 4 (Identity, RD, TD, and implicitly ND by closure)
    tc_rd_td = FCCTexture({"copper": 1.0}, apply_sample_symmetries=True, use_rd=True, use_td=True, use_nd=False)
    gp_rd_td = tc_rd_td._compute_sample_symmetry_group()
    assert len(gp_rd_td) == 4
    # Verify ND is in the group by closure
    assert any(np.allclose(item, np.diag([-1.0, -1.0, 1.0])) for item in gp_rd_td)

    # All three generators active -> group size 4
    tc_all = FCCTexture({"copper": 1.0}, apply_sample_symmetries=True, use_rd=True, use_td=True, use_nd=True)
    gp_all = tc_all._compute_sample_symmetry_group()
    assert len(gp_all) == 4


def test_combined_symmetry_variants_count():
    # Full sample symmetry active
    tc = FCCTexture({"s": 0.3, "goss": 0.3, "cube": 0.4}, apply_sample_symmetries=True)
    
    # S component (general orientation) -> 24 crystal * 4 sample = 96 unique orientations
    s_eqs = tc._get_combined_symmetry_equivalents(tc.components["s"])
    assert len(s_eqs) == 96
    
    # Goss component -> collapses to 24 unique orientations due to symmetry overlap (RD/TD/ND are crystal axes)
    goss_eqs = tc._get_combined_symmetry_equivalents(tc.components["goss"])
    assert len(goss_eqs) == 24

    # Cube component (aligned with RD, TD, ND) -> collapses to 24 unique orientations
    cube_eqs = tc._get_combined_symmetry_equivalents(tc.components["cube"])
    assert len(cube_eqs) == 24


def test_detailed_output():
    tc = FCCTexture({"copper": 0.5, "brass": 0.3}, spread=4.0)
    N = 300
    
    # Test Euler convention
    res_euler = tc.generate_detailed(N, convention='euler')
    assert "components" in res_euler
    assert "final_shuffled" in res_euler
    assert res_euler["final_shuffled"].shape == (300, 3)
    assert "copper" in res_euler["components"]
    assert "variants" in res_euler["components"]["copper"]
    assert "mean_eulers" in res_euler["components"]["copper"]
    assert "combined_shuffled" in res_euler["components"]["copper"]
    assert "random" in res_euler["components"]
    
    # Test Quaternion convention
    res_quat = tc.generate_detailed(N, convention='quaternion')
    assert res_quat["final_shuffled"].shape == (300, 4)
    # Check that variants are also converted
    assert res_quat["components"]["copper"]["variants"][0].shape[1] == 4
    
    # Test Matrix convention
    res_mat = tc.generate_detailed(N, convention='matrix')
    assert res_mat["final_shuffled"].shape == (300, 3, 3)
    assert res_mat["components"]["copper"]["variants"][0].shape[1:] == (3, 3)


def test_plotting_detailed_smoke():
    from upxo.viz.xphy import plot_components, plot_variants
    tc = FCCTexture({"copper": 0.5, "brass": 0.3}, spread=4.0)
    res = tc.generate_detailed(200, convention='quaternion')
    
    # Test plot_components smoke test
    fig_comp, ax_comp = plot_components(res["components"], convention='quaternion', pole_family='111')
    assert ax_comp is not None
    plt.close(fig_comp)
    
    # Test plot_variants smoke test
    copper_data = res["components"]["copper"]
    fig_var, ax_var = plot_variants(
        copper_data["variants"],
        copper_data["mean_eulers"],
        convention='quaternion',
        pole_family='100'
    )
    assert ax_var is not None
    plt.close(fig_var)



