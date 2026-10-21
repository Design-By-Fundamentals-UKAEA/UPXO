"""Ensemble seed-config files: writing, loading, migrating and listing."""
import json

import pytest

from upxo.pxtal.fm_steel_3d import ensemble_seed_config as esc
from upxo.pxtal.fm_steel_3d.steps import steps_ensemble


def _write(tmp_path, payload, name='seed.json'):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _valid(**extra):
    payload = {'schema_version': esc.CURRENT_SCHEMA_VERSION, 'generated_at': 'now', 'seed_name': 's',
               'notes': '', 'pipeline_mode': 'block', 'config': {}, 'injection_points': {}}
    payload.update(extra)
    return payload


# ── writing ───────────────────────────────────────────────────────────────────

def test_round_trip(tmp_path):
    config = {'LX': 10.0, 'N_SEEDS': 50, 'RANDOM_SEED': 7, 'pag_algorithm': 'x', 'unlisted_key': [1, 2],
              'pipeline_mode': 'subblock', 'MESH_ELEM_TYPE': 'C3D8'}
    path = esc.write_ensemble_seed_config(config, tmp_path / 'out', seed_name='My run', notes='n')
    assert path.name == 'My_run_0.json'
    loaded = esc.load_ensemble_seed_config(path)
    assert loaded['config'] == config
    assert loaded['pipeline_mode'] == 'subblock'
    assert loaded['notes'] == 'n' and loaded['seed_name'] == 'My_run'
    assert loaded['schema_version'] == esc.CURRENT_SCHEMA_VERSION
    assert loaded['injection_points']['base_grain'] == {'pipeline_mode': 'subblock', 'LX': 10.0,
                                                          'N_SEEDS': 50, 'RANDOM_SEED': 7}
    assert loaded['injection_points']['pag_orientation'] == {'pag_algorithm': 'x'}
    assert loaded['injection_points']['mesh'] == {'MESH_ELEM_TYPE': 'C3D8'}
    assert loaded['injection_points']['subblock'] == {}
    assert 'unlisted_key' not in json.dumps(loaded['injection_points'])


def test_every_stage_is_present_even_when_empty(tmp_path):
    path = esc.write_ensemble_seed_config({}, tmp_path)
    assert set(esc.load_ensemble_seed_config(path)['injection_points']) == set(esc.INJECTION_POINT_KEYS)


def test_the_index_counts_up_and_fills_the_first_gap(tmp_path):
    names = [esc.write_ensemble_seed_config({}, tmp_path, 'a').name for _ in range(3)]
    assert names == ['a_0.json', 'a_1.json', 'a_2.json']
    (tmp_path / 'a_1.json').unlink()
    assert esc.write_ensemble_seed_config({}, tmp_path, 'a').name == 'a_1.json'
    assert esc.write_ensemble_seed_config({}, tmp_path, 'b').name == 'b_0.json'


def test_names_are_made_safe_and_blank_means_seed(tmp_path):
    assert esc.write_ensemble_seed_config({}, tmp_path, 'a b/c:d').name == 'a_b_c_d_0.json'
    assert esc.write_ensemble_seed_config({}, tmp_path, '   ').name == 'seed_0.json'


def test_the_directory_is_created(tmp_path):
    path = esc.write_ensemble_seed_config({}, tmp_path / 'x' / 'y')
    assert path.parent == tmp_path / 'x' / 'y' and path.exists()


def test_explicit_mode_and_injection_points_win(tmp_path):
    path = esc.write_ensemble_seed_config({'pipeline_mode': 'block'}, tmp_path, pipeline_mode='subblock',
                                          injection_points={'custom': {'k': 1}})
    loaded = esc.load_ensemble_seed_config(path)
    assert loaded['pipeline_mode'] == 'subblock' and loaded['injection_points'] == {'custom': {'k': 1}}


def test_values_that_are_not_json_are_written_as_text(tmp_path):
    from pathlib import PurePosixPath
    path = esc.write_ensemble_seed_config({'where': PurePosixPath('/a/b')}, tmp_path)
    assert esc.load_ensemble_seed_config(path)['config']['where'] == '/a/b'


def test_an_unwritable_location_raises_oserror(tmp_path):
    blocker = tmp_path / 'file'
    blocker.write_text('x')
    with pytest.raises(OSError):
        esc.write_ensemble_seed_config({}, blocker / 'sub')


def test_the_notebook_step_writes_loadable_files(tmp_path):
    path = steps_ensemble.write_seed_config({'LX': 5.0, 'pipeline_mode': 'block'}, tmp_path, 'nb', 'note')
    loaded = esc.load_ensemble_seed_config(path)
    assert loaded['notes'] == 'note' and loaded['config']['LX'] == 5.0
    assert steps_ensemble.CURRENT_SCHEMA_VERSION == esc.CURRENT_SCHEMA_VERSION


# ── loading ───────────────────────────────────────────────────────────────────

def test_missing_file(tmp_path):
    with pytest.raises(esc.EnsembleSeedConfigError, match='not found'):
        esc.load_ensemble_seed_config(tmp_path / 'nope.json')


def test_unparseable_file(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{not json')
    with pytest.raises(esc.EnsembleSeedConfigError, match='Could not read/parse'):
        esc.load_ensemble_seed_config(path)


def test_json_that_is_not_an_object(tmp_path):
    with pytest.raises(esc.EnsembleSeedConfigError, match='does not contain a JSON object'):
        esc.load_ensemble_seed_config(_write(tmp_path, [1, 2]))


def test_no_schema_version(tmp_path):
    with pytest.raises(esc.EnsembleSeedConfigError, match="no 'schema_version'"):
        esc.load_ensemble_seed_config(_write(tmp_path, {'config': {}}))


def test_newer_schema(tmp_path):
    payload = _valid(schema_version=esc.CURRENT_SCHEMA_VERSION + 1)
    with pytest.raises(esc.EnsembleSeedConfigError, match='only understands up to'):
        esc.load_ensemble_seed_config(_write(tmp_path, payload))


@pytest.mark.parametrize('missing', esc.REQUIRED_KEYS)
def test_required_keys(tmp_path, missing):
    payload = _valid()
    del payload[missing]
    with pytest.raises(esc.EnsembleSeedConfigError, match=f'missing required key.*{missing}'):
        esc.load_ensemble_seed_config(_write(tmp_path, payload))


def test_older_files_are_migrated(tmp_path, monkeypatch):
    def v1_to_v2(payload):
        payload = dict(payload)
        payload['injection_points'] = {**payload.get('injection_points', {}), 'void': {}}
        payload['schema_version'] = 2
        return payload

    monkeypatch.setattr(esc, 'CURRENT_SCHEMA_VERSION', 2)
    monkeypatch.setattr(esc, '_MIGRATIONS', {1: v1_to_v2})
    loaded = esc.load_ensemble_seed_config(_write(tmp_path, _valid(schema_version=1)))
    assert loaded['schema_version'] == 2 and loaded['injection_points'] == {'void': {}}


def test_a_missing_migration_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(esc, 'CURRENT_SCHEMA_VERSION', 3)
    monkeypatch.setattr(esc, '_MIGRATIONS', {})
    with pytest.raises(esc.EnsembleSeedConfigError, match='No migration registered from schema_version 1'):
        esc.load_ensemble_seed_config(_write(tmp_path, _valid(schema_version=1)))


def test_a_migration_that_does_not_advance_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(esc, 'CURRENT_SCHEMA_VERSION', 2)
    monkeypatch.setattr(esc, '_MIGRATIONS', {1: lambda payload: payload})
    with pytest.raises(esc.EnsembleSeedConfigError, match='did not advance'):
        esc.load_ensemble_seed_config(_write(tmp_path, _valid(schema_version=1)))


# ── listing ───────────────────────────────────────────────────────────────────

def test_listing_keeps_only_seed_configs(tmp_path):
    good_b = esc.write_ensemble_seed_config({}, tmp_path, 'b')
    good_a = esc.write_ensemble_seed_config({}, tmp_path, 'a')
    (tmp_path / 'notes.txt').write_text('x')
    (tmp_path / 'broken.json').write_text('{')
    _write(tmp_path, {'something': 'else'}, 'other.json')
    assert esc.list_ensemble_seed_configs(tmp_path) == [good_a, good_b]


def test_listing_a_missing_directory(tmp_path):
    assert esc.list_ensemble_seed_configs(tmp_path / 'absent') == []
