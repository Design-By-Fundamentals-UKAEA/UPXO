"""Portable files for pipeline results: a JSON skeleton plus an npz of arrays.

Arrays, numbers, strings, lists, tuples, sets, dictionaries (any hashable
key, including tuples), paths and the result dataclasses of this package are
written without pickle, so the files can be read by other tools and are safe
to open. A file records a format name and version; a reader refuses a version
it does not know. Only dataclasses from ``upxo.meshing.gbconformant`` are
rebuilt when reading.
"""
from dataclasses import fields, is_dataclass
import importlib
import json
from pathlib import Path

import numpy as np

FORMAT = 'upxo.gbconformant.checkpoint'
VERSION = 1
ALLOWED_PREFIXES = ('upxo.meshing.gbconformant.',)


class SerializeError(TypeError):
    """A value cannot be written in the portable form."""


def _plain_key(key):
    return isinstance(key, str) and not key.startswith('__')


def encode(value, arrays=None, where='value'):
    """Return ``(skeleton, arrays)``: JSON-able structure and the arrays it refers to."""
    arrays = {} if arrays is None else arrays
    return _encode(value, arrays, where), arrays


def _encode(value, arrays, where):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'O':
            raise SerializeError(f'{where}: object arrays cannot be stored')
        name = f'a{len(arrays)}'
        arrays[name] = value
        return {'__array__': name}
    if isinstance(value, tuple):
        return {'__tuple__': [_encode(v, arrays, f'{where}[{i}]') for i, v in enumerate(value)]}
    if isinstance(value, list):
        return [_encode(v, arrays, f'{where}[{i}]') for i, v in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        return {'__set__': [_encode(v, arrays, where) for v in sorted(value, key=repr)]}
    if isinstance(value, dict):
        if all(_plain_key(k) for k in value):
            return {k: _encode(v, arrays, f'{where}.{k}') for k, v in value.items()}
        return {'__dict__': [[_encode(k, arrays, where), _encode(v, arrays, f'{where}[{k!r}]')]
                             for k, v in value.items()]}
    if isinstance(value, Path):
        return {'__path__': str(value)}
    if type(value).__name__ == 'FrozenGeometry' and type(value).__module__.endswith('thin_freeze'):
        return {'__frozen_geometry__': {
            'labels': _encode(value.labels, arrays, where + '.labels'),
            'ids': sorted(int(i) for i in value.ids),
            'spacing': _encode(value.spacing, arrays, where + '.spacing')}}
    if is_dataclass(value) and not isinstance(value, type):
        module = type(value).__module__
        if not module.startswith(ALLOWED_PREFIXES):
            raise SerializeError(f'{where}: {module}.{type(value).__name__} is not a result class')
        return {'__dataclass__': f'{module}:{type(value).__qualname__}',
                'fields': {f.name: _encode(getattr(value, f.name), arrays, f'{where}.{f.name}')
                           for f in fields(value)}}
    raise SerializeError(f'{where}: cannot store a {type(value).__name__}')


def decode(skeleton, arrays):
    """Rebuild a value from :func:`encode` output (``arrays`` maps names to arrays)."""
    if isinstance(skeleton, list):
        return [decode(v, arrays) for v in skeleton]
    if not isinstance(skeleton, dict):
        return skeleton
    if '__array__' in skeleton:
        return np.asarray(arrays[skeleton['__array__']])
    if '__tuple__' in skeleton:
        return tuple(decode(v, arrays) for v in skeleton['__tuple__'])
    if '__set__' in skeleton:
        return {decode(v, arrays) for v in skeleton['__set__']}
    if '__dict__' in skeleton:
        return {decode(k, arrays): decode(v, arrays) for k, v in skeleton['__dict__']}
    if '__path__' in skeleton:
        return Path(skeleton['__path__'])
    if '__frozen_geometry__' in skeleton:
        from upxo.meshing.gbconformant.d3v2p0.thin_freeze import FrozenGeometry
        spec = skeleton['__frozen_geometry__']
        return FrozenGeometry(decode(spec['labels'], arrays), spec['ids'], decode(spec['spacing'], arrays))
    if '__dataclass__' in skeleton:
        module, _, qualname = skeleton['__dataclass__'].partition(':')
        if not module.startswith(ALLOWED_PREFIXES):
            raise SerializeError(f'{module} is not an allowed result module')
        cls = importlib.import_module(module)
        for part in qualname.split('.'):
            cls = getattr(cls, part)
        if not is_dataclass(cls):
            raise SerializeError(f'{skeleton["__dataclass__"]} is not a dataclass')
        return cls(**{k: decode(v, arrays) for k, v in skeleton['fields'].items()})
    return {k: decode(v, arrays) for k, v in skeleton.items()}


def safe_name(key):
    """File-system safe stem for an artefact key."""
    return ''.join(c if c.isalnum() or c in '_.-' else '_' for c in str(key))


def write_value(stem, value, where='value'):
    """Write ``value`` as ``<stem>.json`` (+ ``<stem>.npz`` when it holds arrays).

    Returns ``{'json': name, 'npz': name or None}``; raises :class:`SerializeError`.
    """
    stem = Path(stem)
    skeleton, arrays = encode(value, where=where)
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': FORMAT, 'version': VERSION, 'value': skeleton}
    json_path = stem.with_name(stem.name + '.json')
    json_path.write_text(json.dumps(payload), encoding='utf8')
    npz_name = None
    if arrays:
        npz_path = stem.with_name(stem.name + '.npz')
        np.savez(npz_path, **arrays)
        npz_name = npz_path.name
    return {'json': json_path.name, 'npz': npz_name}


def read_value(stem):
    """Read what :func:`write_value` wrote; refuses an unknown format or version."""
    stem = Path(stem)
    payload = json.loads(stem.with_name(stem.name + '.json').read_text(encoding='utf8'))
    if payload.get('format') != FORMAT or payload.get('version') != VERSION:
        raise SerializeError(
            f"unsupported file ({payload.get('format')} v{payload.get('version')}); "
            f'this code reads {FORMAT} v{VERSION}')
    npz_path = stem.with_name(stem.name + '.npz')
    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    else:
        arrays = {}
    return decode(payload['value'], arrays)
