"""A minimal set provider for the tests: one element set, one node set and one surface."""
import numpy as np

from upxo.meshing.gbconformant.fm_steel_abaqus.extra_sets import ExtraSets, SetFile
from upxo.meshing.gbconformant.pipeline.settings import Setting

SETTINGS = (Setting('FAKE_EXTRA_NAME', 'es_extra', stages=('export_abaqus', 'export_abaqus_voxel')),)


def options_from_config(config):
    return {'name': str(config.get('FAKE_EXTRA_NAME', 'es_extra')), 'file': str(config.get('FAKE_EXTRA_FILE', ''))}


def settings_problems(settings):
    return [] if settings.provider_options.get('name') != 'refuse' else ['the fake provider refuses this name']


def extra_sets(request):
    opts = request.settings.provider_options
    name = opts.get('name', 'es_extra')
    file = opts.get('file') or '03x_elsets_extra.inp'
    first = np.arange(min(3, len(request.elements)))
    return ExtraSets(
        files=[SetFile(file, 'Elset', 'Extra element set', 'extra', [(name, first)]),
               SetFile('03y_surfaces_extra.inp', 'Surface', 'Extra surface', 'extra',
                       [('surf_extra', (first, np.ones(len(first), dtype=np.int64)))]),
               SetFile('04x_nsets_extra.inp', 'Nset', 'Extra node set', 'extra',
                       [('ns_extra', np.unique(request.elements[first].ravel()))])],
        skipped=[{'name': 'es_never', 'family': 'extra', 'reason': 'not asked for'}],
        summary={'calls': 1, 'identity_given': request.identity is not None})
