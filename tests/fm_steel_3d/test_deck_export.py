"""Abaqus deck export with chosen set names, optional sets and a chosen level."""
import re
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.deck_export import (
    ConfigurableMeshExporter3D, as_bool, export_deck,
    find_isolated_grain_orientations, next_instance_directory, parse_integer_list,
    parse_positive_int)
from upxo.pxtal.fm_steel_3d.mesh_exporter_3d import MeshExporter3D
from upxo.pxtal.fm_steel_3d.voxel_interface_masks import (
    grain_boundary_voxels, junction_line_voxels, junction_point_voxels)

from .deck_state import build_state

LEVELS = ('pag', 'block', 'subblock')


def _export(exporter, state, base, element='C3D8', **kwargs):
    base.mkdir(parents=True, exist_ok=True)
    exporter.set_output_base_path(str(base))
    exporter.set_output_units('microns')
    return getattr(exporter, f'export_{element.lower()}')(state, 'run', **kwargs)


def _files(directory):
    return {p.name: p.read_text(errors='replace') for p in sorted(directory.iterdir()) if p.is_file()}


def _elsets_defined(files):
    names = set()
    for name, text in files.items():
        if name.startswith('03'):
            names |= set(re.findall(r'\*Elset, elset=(\S+)', text))
    return names


def _section_elsets(files):
    return set(re.findall(r'\*Solid Section, elset=([^,\s]+)', files['06_sections.inp']))


def _drop_comment_lines(text):
    return '\n'.join(line for line in text.splitlines() if not line.startswith('** Isolated grain elsets'))


# ── without options the base exporter's output is reproduced ─────────────────

@pytest.mark.parametrize('level', LEVELS)
@pytest.mark.parametrize('element', ['C3D8', 'C3D4'])
def test_no_options_reproduces_the_base_exporter(tmp_path, level, element):
    state = build_state(level)
    base = _files(_export(MeshExporter3D(), state, tmp_path / 'base', element))
    new = _files(_export(ConfigurableMeshExporter3D(), state, tmp_path / 'new', element))
    assert base.keys() == new.keys()
    for name in base:
        assert _drop_comment_lines(base[name]) == _drop_comment_lines(new[name]), name


# ── element sets ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('level', LEVELS)
def test_prefixes_name_the_sets_and_the_sections_that_use_them(tmp_path, level):
    options = {'prefix_pag': 'P_', 'prefix_pck': 'K_', 'prefix_blk': 'B_',
               'prefix_subblk': 'S_', 'prefix_iso': 'I_'}
    files = _files(_export(ConfigurableMeshExporter3D(options), build_state(level), tmp_path))
    defined = _elsets_defined(files)
    used = _section_elsets(files)
    assert used and used <= defined
    assert not any(name.startswith('es_') for name in defined)
    assert {'P_1', 'K_1_2', 'I_5'} <= defined
    assert ('B_1_1_1' in defined) == (level != 'pag')
    assert ('S_1_1_1_1' in defined) == (level == 'subblock')


def test_an_empty_prefix_falls_back_to_the_default(tmp_path):
    files = _files(_export(ConfigurableMeshExporter3D({'prefix_pag': ''}), build_state('pag'), tmp_path))
    assert 'es_pag_1' in _elsets_defined(files)


def test_switched_off_levels_leave_their_files_empty_and_out_of_the_master(tmp_path):
    options = {'elset_pag': False, 'elset_pck': False}
    files = _files(_export(ConfigurableMeshExporter3D(options), build_state('subblock'), tmp_path))
    assert files['03a_elsets_pag.inp'] == '' and files['03b_elsets_pck.inp'] == ''
    master = files['model_master.inp']
    assert '03a_elsets_pag.inp' not in master and '03b_elsets_pck.inp' not in master
    assert '03c_elsets_blk.inp' in master and '03d_elsets_subblk.inp' in master
    assert not any(name.startswith(('es_pag_', 'es_pck_')) for name in _elsets_defined(files))
    assert _section_elsets(files) and _section_elsets(files) <= _elsets_defined(files)


def test_switched_off_subblock_sets_take_their_sections_with_them(tmp_path):
    files = _files(_export(ConfigurableMeshExporter3D({'elset_subblk': False}),
                           build_state('subblock'), tmp_path))
    assert files['03d_elsets_subblk.inp'] == ''
    assert '03d_elsets_subblk.inp' not in files['model_master.inp']
    assert not any('subblk' in name for name in _section_elsets(files))
    assert _section_elsets(files) <= _elsets_defined(files)


def test_master_includes_every_level_by_default(tmp_path):
    master = _files(_export(ConfigurableMeshExporter3D(), build_state('subblock'), tmp_path))['model_master.inp']
    for name in ('03a_elsets_pag.inp', '03b_elsets_pck.inp', '03c_elsets_blk.inp', '03d_elsets_subblk.inp'):
        assert f'*INCLUDE, INPUT={name}' in master


def test_isolated_grain_sets_can_be_switched_off(tmp_path):
    files = _files(_export(ConfigurableMeshExporter3D({'elset_iso': False}), build_state('block'), tmp_path))
    assert not any(name.startswith('es_iso_') for name in _elsets_defined(files))
    assert 'MAT_ISO_5' not in files['05_materials.inp']


# ── grain-boundary, junction-line and junction-point sets ────────────────────

def _eids(text, name):
    block = re.search(rf'\*Elset, elset={name}\n((?:[\d, ]+\n)+)', text)
    return {int(v) for v in block.group(1).replace('\n', ',').split(',') if v.strip()}


def test_voxel_feature_sets_hold_the_marked_active_voxels(tmp_path):
    state = build_state('block')
    options = {'elset_gb': True, 'elset_jl': True, 'elset_jp': True,
               'prefix_gb': 'GB_', 'prefix_jl': 'JL_', 'prefix_jp': 'JP_'}
    files = _files(_export(ConfigurableMeshExporter3D(options), state, tmp_path))
    text = files['03c_elsets_blk.inp']
    active = state.lgi > 0
    for prefix, mask in (('GB_', grain_boundary_voxels), ('JL_', junction_line_voxels),
                         ('JP_', junction_point_voxels)):
        ii, jj, kk = np.where(mask(state.lgi) & active)
        expected = set((ii * 36 + jj * 6 + kk + 1).tolist())
        assert expected and _eids(text, f'{prefix}all') == expected


def test_voxel_feature_sets_are_off_by_default(tmp_path):
    text = _files(_export(ConfigurableMeshExporter3D(), build_state('block'), tmp_path))['03c_elsets_blk.inp']
    assert '_all' not in text and 'es_gb_' not in text


# ── node sets and the step ────────────────────────────────────────────────────

def _node_sets(exporter, nx=1, ny=1, nz=1, mask=None):
    mask = np.ones((nx + 1, ny + 1, nz + 1), dtype=bool) if mask is None else mask
    out = StringIO()
    exporter._write_nsets_bc(out, nx, ny, nz, mask)
    return out.getvalue()


def test_face_set_prefixes_are_used_by_sets_and_step(tmp_path):
    exporter = ConfigurableMeshExporter3D({
        'prefix_z_max': 'my_loaded_face', 'prefix_x_min': 'my_x_support',
        'nset_z_max': False,                        # the step still needs this set
    })
    sets = _node_sets(exporter)
    steps = StringIO()
    exporter._write_steps(steps, exporter._boundary_nset_config())
    assert '*Nset, nset=my_loaded_face, instance=FM_STEEL_RVE-1\n' in sets
    assert '*Nset, nset=my_x_support, instance=FM_STEEL_RVE-1\n' in sets
    assert 'my_loaded_face, 3, 3, 0.2\n' in steps.getvalue()
    assert 'my_x_support, 1, 1\n' in steps.getvalue()


def test_optional_face_sets_follow_the_switches():
    exporter = ConfigurableMeshExporter3D({'nset_x_max': False, 'nset_y_max': False})
    sets = _node_sets(exporter)
    assert 'ns_xplus' not in sets and 'ns_yplus' not in sets
    assert 'ns_zplus' in sets and 'ns_xminus' in sets

    without_step = ConfigurableMeshExporter3D({'nset_z_max': False})
    without_step.configure_analysis(write_step=False)
    assert 'ns_zplus' not in _node_sets(without_step)


def test_the_loaded_face_follows_the_load_axis():
    exporter = ConfigurableMeshExporter3D({'nset_x_max': False, 'nset_z_max': False})
    exporter.configure_analysis(load_axis='x')
    sets = _node_sets(exporter)
    assert 'ns_xplus' in sets and 'ns_zplus' not in sets


def test_switches_given_as_text_are_read_as_booleans():
    exporter = ConfigurableMeshExporter3D({'nset_y_max': 'false', 'elset_gb': 'off', 'elset_pag': 'no'})
    assert 'ns_yplus' not in _node_sets(exporter)
    assert exporter._elset_enabled('pag') is False
    assert exporter._flag('elset_gb', True) is False


def test_origin_node_set():
    sets = _node_sets(ConfigurableMeshExporter3D({'nset_origins': True, 'prefix_origins': 'at_origin'}))
    assert '*Nset, nset=at_origin, instance=FM_STEEL_RVE-1\n1\n' in sets
    assert 'ns_origin' not in _node_sets(ConfigurableMeshExporter3D())


def _brute_force_edge_nodes(mask, nx, ny, nz):
    ids = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                if not mask[i, j, k]:
                    continue
                on = (i in (0, nx)) + (j in (0, ny)) + (k in (0, nz))
                if on >= 2:
                    ids.append(i * (ny + 1) * (nz + 1) + j * (nz + 1) + k + 1)
    return ids


@pytest.mark.parametrize('shape', [(1, 1, 1), (3, 2, 4), (2, 5, 1)])
def test_domain_edge_node_set_matches_a_node_by_node_count(shape):
    nx, ny, nz = shape
    rng = np.random.default_rng(3)
    mask = rng.random((nx + 1, ny + 1, nz + 1)) > 0.2
    exporter = ConfigurableMeshExporter3D({'nset_rve_edges': True, 'prefix_rve_edges': 'edges'})
    (name, ids), = exporter._extra_node_sets(nx, ny, nz, mask)
    assert name == 'edges'
    assert ids == _brute_force_edge_nodes(mask, nx, ny, nz)


# ── level ─────────────────────────────────────────────────────────────────────

def test_packet_level_gives_one_material_and_section_per_packet(tmp_path):
    files = _files(_export(ConfigurableMeshExporter3D({'export_level': 'packet'}),
                           build_state('subblock'), tmp_path))
    materials, sections = files['05_materials.inp'], files['06_sections.inp']
    packets = {(1, 1), (1, 2), (2, 1)}
    for pag, pkt in packets:
        assert f'*Material, name=MAT_PCK_{pag}_{pkt}\n' in materials
        assert f'*Solid Section, elset=es_pck_{pag}_{pkt}, material=MAT_PCK_{pag}_{pkt}\n' in sections
    assert materials.count('*Material, name=MAT_PCK_') == len(packets)
    assert 'MAT_BLK_' not in materials and 'MAT_SUBBLK_' not in materials
    assert '1.000000, 2.000000, 3.000000,' in materials     # isolated grain orientation
    assert _section_elsets(files) <= _elsets_defined(files)


def test_packet_sections_follow_the_prefixes(tmp_path):
    files = _files(_export(ConfigurableMeshExporter3D(
        {'export_level': 'packet', 'prefix_pck': 'K_', 'prefix_iso': 'I_'}), build_state('block'), tmp_path))
    assert _section_elsets(files) == {'K_1_1', 'K_1_2', 'K_2_1', 'I_5'}
    assert _section_elsets(files) <= _elsets_defined(files)


def test_a_missing_isolated_orientation_is_a_placeholder(tmp_path):
    state = build_state('pag')
    state.isolated_grain_orientations = {}
    materials = _files(_export(ConfigurableMeshExporter3D({'export_level': 'packet'}), state, tmp_path))['05_materials.inp']
    assert 'no orientation assigned' in materials and '0.000000, 0.000000, 0.000000,' in materials


@pytest.mark.parametrize('requested, state_level, expected', [
    ('pag', 'subblock', 'pag'),
    ('block', 'subblock', 'block'),
    ('subblock', 'subblock', 'subblock'),
    ('packet', 'pag', 'packet'),
    ('auto', 'subblock', 'subblock'),
    ('auto', 'block', 'block'),
    ('auto', 'pag', 'pag'),
])
def test_requested_level(requested, state_level, expected):
    exporter = ConfigurableMeshExporter3D({'export_level': requested})
    assert exporter._detect_lowest_level(build_state(state_level)) == expected


@pytest.mark.parametrize('requested, state_level', [('subblock', 'block'), ('subblock', 'pag'), ('block', 'pag')])
def test_an_unavailable_level_falls_back_to_the_detected_one(capsys, requested, state_level):
    exporter = ConfigurableMeshExporter3D({'export_level': requested})
    assert exporter._detect_lowest_level(build_state(state_level)) == state_level
    assert f'export level {requested!r} is not available' in capsys.readouterr().out


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match='export_level'):
        ConfigurableMeshExporter3D({'export_level': 'grain'})


# ── output location ───────────────────────────────────────────────────────────

def test_flat_output_writes_into_the_base_directory(tmp_path):
    out = _export(ConfigurableMeshExporter3D(flat_output=True), build_state('pag'), tmp_path / 'here')
    assert out == tmp_path / 'here'
    assert (tmp_path / 'here' / 'model_master.inp').exists()


def test_without_flat_output_a_numbered_folder_is_made(tmp_path):
    out = _export(ConfigurableMeshExporter3D(), build_state('pag'), tmp_path / 'here')
    assert out == tmp_path / 'here' / 'run1'


# ── export_deck ───────────────────────────────────────────────────────────────

def test_export_deck_matches_a_direct_export(tmp_path):
    state = build_state('block')
    (tmp_path / 'a').mkdir()
    direct = _files(_export(ConfigurableMeshExporter3D({'prefix_blk': 'B_'}, flat_output=True),
                            state, tmp_path / 'direct', 'C3D4', custom_message='note'))
    out = export_deck(state, 'C3D4', tmp_path / 'a', output_unit='microns',
                      options={'prefix_blk': 'B_'}, custom_message='note', verbosity=0)
    assert out == tmp_path / 'a'
    assert _files(out) == direct


def test_export_deck_rescales_a_copy_of_the_grid(tmp_path):
    state = build_state('pag')
    out = export_deck(state, 'C3D8', tmp_path, scale_factor=2.0, output_unit='microns', verbosity=0)
    assert state.lgi.shape == (6, 6, 6) and state.voxel_size == 2.0
    master = (out / 'model_master.inp').read_text(errors='replace')
    assert 'Grid: 12 x 12 x 12 voxels' in master
    assert 'Voxel size: 1 microns' in master


def test_export_deck_passes_analysis_and_ignored_grains(tmp_path):
    state = build_state('pag')
    out = export_deck(state, 'C3D8', tmp_path, output_unit='microns', verbosity=0,
                      ignore_ids=[5], analysis={'load_axis': 'x', 'write_step': 'true',
                                                'applied_displacement': 0.5})
    steps = (out / '08_steps_output.inp').read_text(errors='replace')
    assert 'ns_xplus, 1, 1, 0.5\n' in steps
    assert 'es_iso_5' not in (out / '03c_elsets_blk.inp').read_text(errors='replace')

    (tmp_path / 'kept').mkdir()
    kept = export_deck(state, 'C3D8', tmp_path / 'kept', output_unit='microns', verbosity=0)
    assert 'es_iso_5' in (kept / '03c_elsets_blk.inp').read_text(errors='replace')


def test_export_deck_without_a_step(tmp_path):
    out = export_deck(build_state('pag'), 'C3D8', tmp_path, verbosity=0, analysis={'write_step': 'off'})
    assert 'No step written' in (out / '08_steps_output.inp').read_text(errors='replace')


@pytest.mark.parametrize('element_type, scale_factor, message', [
    ('C3D6', 1.0, 'element_type'),
    ('C3D8', 0, 'finite number greater than zero'),
    ('C3D8', 0.01, 'at least one voxel'),
])
def test_export_deck_rejects_bad_arguments(tmp_path, element_type, scale_factor, message):
    with pytest.raises(ValueError, match=message):
        export_deck(build_state('pag'), element_type, tmp_path, scale_factor=scale_factor, verbosity=0)


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('value, default, expected', [
    (True, False, True), (False, True, False), (1, False, True), (0, True, False),
    ('true', False, True), (' On ', False, True), ('YES', False, True), ('1', False, True),
    ('false', True, False), ('off', True, False), ('0', True, False), ('', True, False),
    ('maybe', True, True), ('maybe', False, False),
])
def test_as_bool(value, default, expected):
    assert as_bool(value, default) is expected


def test_parse_integer_list():
    assert parse_integer_list('1, 2,,3 ,') == [1, 2, 3]
    assert parse_integer_list('') == []
    assert parse_integer_list(' -1 ') == [-1]
    with pytest.raises(ValueError):
        parse_integer_list('1, x')


def test_parse_positive_int():
    assert parse_positive_int('250') == 250
    assert parse_positive_int(' 7 ') == 7
    assert parse_positive_int(12) == 12
    assert parse_positive_int('') is None and parse_positive_int(None) is None
    assert parse_positive_int('0') is None and parse_positive_int(-3) is None
    with pytest.raises(ValueError):
        parse_positive_int('lots')


def test_next_instance_directory(tmp_path):
    base = tmp_path / 'out'
    first = next_instance_directory(base)
    assert first.endswith('instance1') and (base / 'instance1').is_dir()
    (base / 'instance7').mkdir()
    (base / 'Instance3').mkdir()
    (base / 'instance9').write_text('a file, not a folder')
    (base / 'instanceX').mkdir()
    assert next_instance_directory(base).endswith('instance8')


def test_isolated_orientations_are_found_along_the_parent_chain():
    inner = SimpleNamespace(_parent=None)
    middle = SimpleNamespace(_parent=SimpleNamespace(isolated_grain_orientations={4: (1, 2, 3)}))
    outer = SimpleNamespace(_parent=middle)
    assert find_isolated_grain_orientations(middle) == {4: (1, 2, 3)}
    assert find_isolated_grain_orientations(outer) == {4: (1, 2, 3)}
    assert find_isolated_grain_orientations(inner) == {}
