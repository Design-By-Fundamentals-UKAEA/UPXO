"""An optional provider of further element sets, node sets and surfaces for a deck.

A deck always gets the sets of :mod:`.element_sets`. A *provider* is a module, named by
``DeckSettings.set_provider`` (empty: none), that can add more. The deck writer checks the names and the
shape of what a provider returns, writes the sets in the deck's include files, and records the provider's name
and options in ``deck_info.json``, so a deck says how it was made.

A provider module has

* ``extra_sets(request) -> ExtraSets`` (required),
* ``SETTINGS``: a tuple of :class:`~upxo.meshing.gbconformant.pipeline.settings.Setting` that it adds to the
  settings table (optional),
* ``options_from_config(config) -> dict``: its options, read from a settings dictionary (optional),
* ``settings_problems(settings) -> list[str]``: what is wrong with the options in
  :class:`~.export.DeckSettings` (optional).

Element and node indices a provider returns are 0-based; a surface is ``(element indices, Abaqus face numbers)``.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field

import numpy as np

KEYWORDS = ('Elset', 'Nset', 'Surface')
_FILE_NAME = {'Elset': re.compile(r'03[a-z0-9]*_[A-Za-z0-9_]+\.inp\Z'),
              'Surface': re.compile(r'03[a-z0-9]*_[A-Za-z0-9_]+\.inp\Z'),
              'Nset': re.compile(r'04[a-z0-9]*_[A-Za-z0-9_]+\.inp\Z')}


@dataclass
class SetFile:
    """One include file: ``keyword`` is ``Elset``, ``Nset`` or ``Surface``; ``entries`` ``[(name, ids)]``.

    For a surface the ids are ``(element indices, Abaqus face numbers)``. Element sets and surfaces live in
    the Part (file names start with ``03``), node sets in the Assembly (``04``).
    """
    file: str
    keyword: str
    header: str
    family: str
    entries: list = field(default_factory=list)

    @property
    def scope(self):
        return 'Assembly' if self.keyword == 'Nset' else 'Part'


@dataclass
class ExtraSets:
    """What a provider returns: ``files`` (:class:`SetFile`), ``skipped`` (``{'name', 'family', 'reason'}``
    for each set asked for and not written) and a ``summary`` dictionary stored with the deck."""
    files: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


@dataclass
class SetRequest:
    """What a provider is given: the deck's mesh (compacted node numbering), the feature of every element, the
    hierarchy and the :class:`~.export.DeckSettings`. ``mesh_voxel_length`` is the length of one raw voxel in
    mesh units, ``face_nodes`` the nodes on each box face and ``families`` the families wanted (``None``: all).
    ``identity`` is the :class:`~.identity.IdentityMap` of the meshed grains (``None`` when not at hand).
    ``cache`` is free for the provider's use during one request."""
    points: np.ndarray
    elements: np.ndarray
    assigned: np.ndarray
    hierarchy: object
    settings: object
    mesh_voxel_length: float = 1.0
    face_nodes: dict = field(default_factory=dict)
    families: object = None
    cache: dict = field(default_factory=dict)
    identity: object = None


def load_provider(name):
    """The provider module named ``name`` (a dotted module path), checked for ``extra_sets``."""
    text = str(name or '').strip()
    if not text:
        raise ValueError('No set provider is named.')
    try:
        module = importlib.import_module(text)
    except ImportError as error:
        raise ValueError(f'The set provider {text!r} cannot be imported: {error}') from error
    if not callable(getattr(module, 'extra_sets', None)):
        raise ValueError(f'The set provider {text!r} has no extra_sets(request) function.')
    return module


def provider_options(name, config):
    """The options the provider reads from ``config`` (``{}`` when it has none)."""
    module = load_provider(name)
    reader = getattr(module, 'options_from_config', None)
    return dict(reader(config)) if callable(reader) else {}


def provider_problems(settings):
    """What the provider finds wrong with the options in ``settings`` (empty when there is no provider)."""
    if not getattr(settings, 'set_provider', ''):
        return []
    check = getattr(load_provider(settings.set_provider), 'settings_problems', None)
    return list(check(settings)) if callable(check) else []


def check_files(provider_name, result):
    """Raise ``ValueError`` unless ``result`` is an :class:`ExtraSets` whose files the writer can write."""
    if not isinstance(result, ExtraSets):
        raise ValueError(f'The set provider {provider_name!r} must return ExtraSets, not {type(result).__name__}.')
    seen = set()
    for item in result.files:
        where = f'set provider {provider_name!r}, file {getattr(item, "file", item)!r}'
        if not isinstance(item, SetFile) or item.keyword not in KEYWORDS:
            raise ValueError(f'{where}: not a SetFile of {", ".join(KEYWORDS)}.')
        if not _FILE_NAME[item.keyword].match(str(item.file)):
            raise ValueError(f'{where}: element sets and surfaces go in files named 03<letter>_<name>.inp, '
                             'node sets in 04<letter>_<name>.inp.')
        if item.file in seen:
            raise ValueError(f'{where}: the file name is used twice.')
        seen.add(item.file)
        for entry in item.entries:
            if len(entry) != 2:
                raise ValueError(f'{where}: an entry must be (name, ids).')
    return result
