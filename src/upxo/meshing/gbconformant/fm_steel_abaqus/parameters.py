"""The quantities the before/after comparison can show, read from a :class:`~.snapshots.Snapshot`.

Each :class:`Parameter` says what is plotted (one value per grain, per junction line or per boundary), which views
make sense and what a snapshot must hold. :func:`read` returns the values of one parameter in one snapshot as a
:class:`Series`; :func:`matched_items` pairs two series by grain (or junction set) for the matched-change view.
Values stay in the snapshot's output units; :func:`unit_factor` and :func:`choose_units` scale them for display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import constants as C
from .comparison import sphere_equivalent_diameter
from .snapshots import Snapshot

DISTRIBUTION, MATCHED, POLE = 'distribution', 'matched', 'pole'
AREA_FRACTION, AREA_PER_VOLUME, AREA_ABSOLUTE = 'fraction of the boundary area', 'area per unit volume', 'area'
NORMALISATIONS = (AREA_FRACTION, AREA_PER_VOLUME, AREA_ABSOLUTE)
BY_GRAIN, BY_VOLUME = 'number of grains', 'grain volume'
GRAIN_WEIGHTS = (BY_GRAIN, BY_VOLUME)
DISPLAY_UNITS = ('auto',) + C.UNITS
_SYMBOL = {'microns': r'$\mu$m', 'mm': 'mm', 'm': 'm'}
_PLAIN = {'microns': 'microns', 'mm': 'mm', 'm': 'm'}


class Unavailable(ValueError):
    """The parameter cannot be read from this snapshot; the message says why."""


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    group: str
    dimension: str
    """``length``, ``area``, ``volume``, ``count``, ``ratio`` or ``angle`` (what one value measures)."""
    noun: str
    """What one value belongs to (``grains``, ``lines``, ``boundaries``)."""
    views: tuple = (DISTRIBUTION, MATCHED)
    needs_volume: bool = False
    description: str = ''


PARAMETERS = (
    Parameter('grain_volume', 'Grain volume', 'Size and shape', 'volume', 'grains', needs_volume=True,
              description='Volume of every grain.'),
    Parameter('diameter', 'Sphere-equivalent diameter', 'Size and shape', 'length', 'grains', needs_volume=True,
              description='Diameter of the sphere with the volume of the grain.'),
    Parameter('aspect', 'Grain aspect ratio', 'Size and shape', 'ratio', 'grains', needs_volume=True,
              description='Longest over shortest principal semi-axis of the grain.'),
    Parameter('coordination', 'Number of neighbours (coordination number)', 'Size and shape', 'count', 'grains',
              description='Number of grains that share a boundary with the grain.'),
    Parameter('grain_area', 'Grain boundary area per grain', 'Size and shape', 'area', 'grains',
              description='Area of the interior boundaries of the grain (the faces of the box are not counted).'),
    Parameter('line_length', 'Junction line length (per line)', 'Junctions', 'length', 'lines',
              description='Length of every junction line; the matched view compares the total per set of grains.'),
    Parameter('line_length_per_grain', 'Junction line length per grain', 'Junctions', 'length', 'grains',
              description='Total length of the junction lines that border the grain.'),
    Parameter('misorientation_boundary', 'Grain boundary misorientation', 'Orientation', 'angle', 'boundaries',
              views=(DISTRIBUTION,),
              description='Misorientation across every boundary between two grains, weighted by the boundary area.'),
    Parameter('misorientation_grain', 'Grain mean boundary misorientation', 'Orientation', 'angle', 'grains',
              description='Area-weighted mean of the misorientations across all the boundaries of the grain.'),
    Parameter('pole_figures', 'Pole figures', 'Orientation', 'angle', 'grains', views=(POLE,),
              description='Pole figures of the grains on each side, and their difference.'),
)
BY_KEY = {p.key: p for p in PARAMETERS}
LABELS = tuple(p.label for p in PARAMETERS)


def parameter(key_or_label) -> Parameter:
    text = str(key_or_label)
    if text in BY_KEY:
        return BY_KEY[text]
    for p in PARAMETERS:
        if p.label == text:
            return p
    raise KeyError(f"{key_or_label!r} is not a parameter; the parameters are {', '.join(BY_KEY)}.")


@dataclass
class Series:
    """The values of one parameter in one snapshot."""

    values: np.ndarray
    weights: Optional[np.ndarray] = None
    items: Optional[dict] = None
    """``{grain label or junction set: value}``, for the matched-change view."""
    noun: str = 'grains'
    dimension: str = 'length'
    y_kind: str = 'count'
    """``count`` (unweighted), ``fraction``, ``area`` or ``per_length``: what the weights add up to per unit of value."""
    total: Optional[float] = None
    note: str = ''
    integer: bool = False

    @property
    def count(self) -> int:
        return int(len(self.values))


def _finite(values, keys=None):
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    return (values[ok], None if keys is None else np.asarray(keys)[ok])


def _per_grain(snapshot: Snapshot, values, parameter_, **extra) -> Series:
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    items = {int(k): float(v) for k, v in zip(snapshot.labels[keep], values[keep])}
    return Series(values[keep], items=items, noun=parameter_.noun, dimension=parameter_.dimension, **extra)


def read(parameter_, snapshot: Snapshot, *, normalisation=AREA_FRACTION, grain_weight=BY_GRAIN) -> Series:
    """The values of ``parameter_`` in ``snapshot``; raises :class:`Unavailable` when it has none."""
    p = parameter(parameter_) if not isinstance(parameter_, Parameter) else parameter_
    if p.needs_volume and not snapshot.has_volume:
        raise Unavailable(f'{snapshot.title} has no grain volume: it holds the interior boundaries only.')
    key = p.key
    if key == 'grain_volume':
        return _per_grain(snapshot, snapshot.volume, p)
    if key == 'diameter':
        return _per_grain(snapshot, sphere_equivalent_diameter(snapshot.volume), p)
    if key == 'aspect':
        return _per_grain(snapshot, snapshot.aspect, p)
    if key == 'coordination':
        return _per_grain(snapshot, snapshot.neighbour_count(), p, integer=True)
    if key == 'grain_area':
        area = snapshot.grain_area()
        keep = area > 0
        items = {int(k): float(v) for k, v in zip(snapshot.labels[keep], area[keep])}
        return Series(area[keep], items=items, noun=p.noun, dimension=p.dimension)
    if key == 'line_length':
        length = np.asarray(snapshot.lines['length'], dtype=float)
        items = {k: float(v) for k, v in snapshot.lines['per_set'].items() if all(label >= 0 for label in k)}
        return Series(length, items=items, noun=p.noun, dimension=p.dimension)
    if key == 'line_length_per_grain':
        items = {int(k): float(v) for k, v in snapshot.lines['per_grain'].items()}
        return Series(np.array(list(items.values()), dtype=float), items=items, noun=p.noun, dimension=p.dimension)
    if key == 'misorientation_boundary':
        angle = snapshot.pair_angle
        ok = np.isfinite(angle) & (snapshot.pair_area > 0)
        if not ok.any():
            raise Unavailable('The misorientation is only available for cubic crystals with an orientation for every grain.')
        area = snapshot.pair_area[ok]
        if normalisation == AREA_FRACTION:
            weights, y_kind = area / area.sum(), 'fraction'
        elif normalisation == AREA_PER_VOLUME:
            if not snapshot.rve_volume > 0:
                raise Unavailable('The volume of the structure is not known.')
            weights, y_kind = area / snapshot.rve_volume, 'per_length'
        elif normalisation == AREA_ABSOLUTE:
            weights, y_kind = area, 'area'
        else:
            raise ValueError(f'normalisation must be one of {", ".join(NORMALISATIONS)}, not {normalisation!r}.')
        return Series(angle[ok], weights=weights, noun=p.noun, dimension=p.dimension, y_kind=y_kind,
                      total=float(area.sum()))
    if key == 'misorientation_grain':
        mean = snapshot.mean_misorientation()
        if not np.isfinite(mean).any():
            raise Unavailable('The misorientation is only available for cubic crystals with an orientation for every grain.')
        series = _per_grain(snapshot, mean, p)
        if grain_weight == BY_VOLUME:
            if snapshot.has_volume:
                keep = np.isfinite(mean)
                volume = snapshot.volume[keep]
                series.weights, series.y_kind, series.total = volume / volume.sum(), 'fraction', float(volume.sum())
            else:
                series.note = f'{snapshot.title} has no grain volume, so the grains are counted, not weighted by volume.'
        elif grain_weight != BY_GRAIN:
            raise ValueError(f'grain_weight must be one of {", ".join(GRAIN_WEIGHTS)}, not {grain_weight!r}.')
        return series
    raise Unavailable(f'{p.label} has no values to plot.')


def available(parameter_, snapshot: Snapshot) -> Optional[str]:
    """Why ``parameter_`` cannot be read from ``snapshot``, or ``None`` when it can."""
    try:
        read(parameter_, snapshot)
    except Unavailable as problem:
        return str(problem)
    return None


def matched_items(before: Series, after: Series):
    """``(keys, before values, after values)`` for the items present on both sides, in key order."""
    if before.items is None or after.items is None:
        return [], np.zeros(0), np.zeros(0)
    keys = sorted(set(before.items) & set(after.items), key=lambda k: (str(type(k)), k))
    return keys, np.array([before.items[k] for k in keys], dtype=float), np.array([after.items[k] for k in keys], dtype=float)


# --- units ----------------------------------------------------------------------------------------------------
_POWER = {'length': 1, 'area': 2, 'volume': 3}


def unit_power(dimension: str, y_kind: str = 'count') -> int:
    """The power of a length that ``dimension`` is (0 for a number, an angle or a ratio)."""
    return _POWER.get(dimension, 0)


def unit_factor(power: int, source: str, target: str) -> float:
    """Factor converting a quantity of dimension length**power from ``source`` to ``target`` units."""
    if power == 0 or source == target:
        return 1.0
    return C.unit_scale(source, target) ** power


def choose_units(power: int, source: str, arrays) -> str:
    """The unit (of microns, mm, m) that gives the numbers of ``arrays`` the fewest digits, around 10."""
    if power == 0:
        return source
    pooled = np.concatenate([np.abs(np.asarray(a, dtype=float).ravel()) for a in arrays if np.size(a)] or [np.zeros(0)])
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    if not pooled.size:
        return source
    median = float(np.median(pooled))
    best = min(C.UNITS, key=lambda u: abs(np.log10(median * unit_factor(power, source, u)) - 1.0))
    return best


def unit_text(power: int, units: str, *, plain: bool = False) -> str:
    """``um``-style text of a quantity's unit: ``mm``, ``mm^2`` or, for a plot, mathtext such as ``$\\mu$m$^2$``."""
    if power == 0:
        return ''
    base = (_PLAIN if plain else _SYMBOL)[units]
    if power == 1:
        return base
    return f'{base}^{power}' if plain else f'{base}$^{power}$'


def count_noun(noun: str, n: int) -> str:
    """``271 grains``, ``1 grain``."""
    return f'{n} {noun[:-1] if n == 1 and noun.endswith("s") else noun}'
