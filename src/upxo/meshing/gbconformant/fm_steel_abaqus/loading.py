"""The load history of the analysis: from a displacement, or from a strain and a strain rate.

Modes
-----
``displacement``: one step that moves the loaded face by ``applied_displacement`` in ``step_time``.

``strain``: the history is given as strains. The step time follows from the strain and the strain rate; the
displacement is the strain times the domain length along the load axis (output units).

* ``monotonic``: from zero to ``target_strain``, as tension or compression.
* ``cyclic``: from zero to the maximum strain, then ``cycles`` times down to the minimum and up to the maximum
  again, at the strain rate. The limits are given as minimum and maximum strain, or as a strain amplitude and
  the ratio R = minimum / maximum (maximum = 2 amplitude / (1 - R), minimum = R maximum). The history is a
  tabular amplitude on the loaded face; the longest increment is a tenth of the shortest segment, so the peaks
  are reached.

Either mode can end with a hold: a further step that keeps the loaded face where it is for ``hold_time``.
"""
from dataclasses import dataclass, field
import math

MODES = ('displacement', 'strain')
CASES = ('monotonic', 'cyclic')
SENSES = ('tension', 'compression')
DEFINITIONS = ('min_max', 'amplitude_r')
AMPLITUDE_NAME = 'LOAD-HISTORY'
PEAK_RESOLUTION = 10          # increments per shortest segment, at most


@dataclass
class LoadStep:
    """One step: its time, increments, the displacement on the loaded face and the amplitude that scales it."""
    name: str
    time: float
    initial_inc: float
    min_inc: float
    max_inc: float
    displacement: float            # output units, before the length scale; signed
    amplitude: str | None = None
    kind: str = 'load'             # 'load' or 'hold'
    output_interval: float = 0.0


@dataclass
class LoadPlan:
    steps: list
    amplitudes: dict = field(default_factory=dict)      # {name: [(time, fraction), ...]}
    summary: dict = field(default_factory=dict)


def validate_load(settings):
    """Raise ``ValueError`` when the load settings cannot make a history."""
    if settings.load_mode not in MODES:
        raise ValueError(f'load_mode must be one of {MODES}.')
    if not float(settings.hold_time) >= 0:
        raise ValueError('hold_time must be >= 0.')
    if settings.load_mode == 'displacement':
        return
    if settings.load_case not in CASES:
        raise ValueError(f'load_case must be one of {CASES}.')
    if not float(settings.strain_rate) > 0:
        raise ValueError('strain_rate must be > 0.')
    if settings.load_case == 'monotonic':
        if settings.load_sense not in SENSES:
            raise ValueError(f'load_sense must be one of {SENSES}.')
        if not float(settings.target_strain) > 0:
            raise ValueError('target_strain must be > 0 (the sense says tension or compression).')
        return
    if settings.cycle_definition not in DEFINITIONS:
        raise ValueError(f'cycle_definition must be one of {DEFINITIONS}.')
    if int(settings.cycles) < 1:
        raise ValueError('cycles must be >= 1.')
    low, high = cycle_limits(settings)
    if not low < high:
        raise ValueError('The minimum strain must be below the maximum strain.')
    if high == 0 and low == 0:
        raise ValueError('The cycle must not be zero.')


def cycle_limits(settings):
    """``(minimum strain, maximum strain)`` of the cycle."""
    if settings.cycle_definition == 'amplitude_r':
        amplitude, ratio = float(settings.strain_amplitude), float(settings.r_ratio)
        if not amplitude > 0:
            raise ValueError('strain_amplitude must be > 0.')
        if ratio == 1:
            raise ValueError('r_ratio must not be 1 (no cycle).')
        high = 2.0 * amplitude / (1.0 - ratio)
        return ratio * high, high
    return float(settings.strain_min), float(settings.strain_max)


def strain_vertices(settings):
    """The strain history as ``[(time, strain)]`` corner points, from ``(0, 0)`` (the hold is not included)."""
    rate = float(settings.strain_rate)
    if settings.load_case == 'monotonic':
        strain = float(settings.target_strain) * (1.0 if settings.load_sense == 'tension' else -1.0)
        return [(0.0, 0.0), (abs(strain) / rate, strain)]
    low, high = cycle_limits(settings)
    points, time = [(0.0, 0.0)], 0.0
    if high != 0:
        time = abs(high) / rate
        points.append((time, high))
    for _ in range(int(settings.cycles)):
        for strain in (low, high):
            time += abs(strain - points[-1][1]) / rate
            points.append((time, strain))
    return points


def total_time(settings):
    """Time of the whole analysis, including the hold."""
    hold = float(settings.hold_time)
    if settings.load_mode == 'displacement':
        return float(settings.step_time) + hold
    return strain_vertices(settings)[-1][0] + hold


def controls(settings):
    """Increments and output interval to use, with the first step's time.

    In displacement mode these are the settings as given. In strain mode the step time is not known to the
    person who set the increments, so they are cut down to what fits: no increment is longer than the step,
    a cyclic history has none longer than a tenth of its shortest segment, and output is not rarer than once a step.
    """
    initial, low, high = float(settings.initial_inc), float(settings.min_inc), float(settings.max_inc)
    interval = float(settings.output_interval)
    if settings.load_mode == 'displacement':
        return {'time': float(settings.step_time), 'initial_inc': initial, 'min_inc': low, 'max_inc': high,
                'output_interval': interval, 'notes': []}
    vertices = strain_vertices(settings)
    first = vertices[-1][0] if settings.load_case == 'monotonic' else total_time(settings) - float(settings.hold_time)
    notes = []
    longest = high
    if settings.load_case == 'cyclic':
        segments = [b[0] - a[0] for a, b in zip(vertices, vertices[1:]) if b[0] > a[0]]
        longest = min(high, min(segments) / PEAK_RESOLUTION)
        if longest < high:
            notes.append(f'longest increment cut from {high:g} to {longest:g} so the peaks are reached')
    longest = min(longest, first)
    initial = min(initial, longest)
    low = min(low, initial)
    if interval > first:
        notes.append(f'output interval cut from {interval:g} to {first:g}')
        interval = first
    return {'time': first, 'initial_inc': initial, 'min_inc': low, 'max_inc': longest,
            'output_interval': interval, 'notes': notes}


def plan_load(settings, axis_length):
    """The steps of the analysis; ``axis_length`` is the domain length along the load axis in output units."""
    validate_load(settings)
    c = controls(settings)
    hold = float(settings.hold_time)
    steps, amplitudes, summary = [], {}, {'mode': settings.load_mode, 'notes': list(c['notes'])}
    if settings.load_mode == 'displacement':
        displacement = float(settings.applied_displacement)
        steps.append(LoadStep('Step-1', c['time'], c['initial_inc'], c['min_inc'], c['max_inc'], displacement,
                              output_interval=c['output_interval']))
        final = displacement
        summary.update(case='displacement', time=c['time'], displacement=displacement,
                       strain=displacement / axis_length if axis_length else None)
    elif settings.load_case == 'monotonic':
        strain = float(settings.target_strain) * (1.0 if settings.load_sense == 'tension' else -1.0)
        final = strain * axis_length
        steps.append(LoadStep('Step-1', c['time'], c['initial_inc'], c['min_inc'], c['max_inc'], final,
                              output_interval=c['output_interval']))
        summary.update(case='monotonic', sense=settings.load_sense, strain=strain, strain_rate=float(settings.strain_rate),
                       time=c['time'], displacement=final)
    else:
        vertices = strain_vertices(settings)
        peak = max(abs(s) for _t, s in vertices)
        amplitudes[AMPLITUDE_NAME] = [(t, s / peak) for t, s in vertices]
        low, high = cycle_limits(settings)
        final = vertices[-1][1] * axis_length
        steps.append(LoadStep('Step-1', c['time'], c['initial_inc'], c['min_inc'], c['max_inc'],
                              peak * axis_length, AMPLITUDE_NAME, output_interval=c['output_interval']))
        summary.update(case='cyclic', strain_min=low, strain_max=high, strain_rate=float(settings.strain_rate),
                       cycles=int(settings.cycles), definition=settings.cycle_definition, time=c['time'],
                       displacement=peak * axis_length)
        if settings.cycle_definition == 'amplitude_r':
            summary.update(strain_amplitude=float(settings.strain_amplitude), r_ratio=float(settings.r_ratio))
    if hold > 0:
        steps.append(LoadStep('Hold', hold, min(c['initial_inc'], hold), min(c['min_inc'], hold),
                              min(c['max_inc'], hold), final, None, 'hold', min(c['output_interval'], hold)))
        summary['hold_time'] = hold
    summary['total_time'] = sum(step.time for step in steps)
    summary['steps'] = [{'name': s.name, 'kind': s.kind, 'time': s.time} for s in steps]
    summary['axis_length'] = float(axis_length)
    return LoadPlan(steps=steps, amplitudes=amplitudes, summary=summary)


def describe(plan, units='', axis='z'):
    """The plan (or its ``summary``, as stored in a deck's report) as short text lines for a readout."""
    s = getattr(plan, 'summary', plan)
    unit = f' {units}' if units else ''
    lines = []
    if s['case'] == 'displacement':
        lines.append(f"One step: the loaded face moves {s['displacement']:g}{unit} along {axis} in {s['time']:g} time units")
        if s.get('strain') is not None:
            lines.append(f"engineering strain at the end {s['strain']:.6g} ({100 * s['strain']:.4g} %)")
    elif s['case'] == 'monotonic':
        lines.append(f"Monotonic {s['sense']} to a strain of {abs(s['strain']):g} ({100 * abs(s['strain']):.4g} %) at "
                     f"{s['strain_rate']:g} per time unit: {s['time']:g} time units, the loaded face moves "
                     f"{abs(s['displacement']):.6g}{unit} along {axis}")
    else:
        lines.append(f"Cyclic between strains {s['strain_min']:g} and {s['strain_max']:g} at {s['strain_rate']:g} per time "
                     f"unit, {s['cycles']} cycle(s): {s['time']:g} time units, peak displacement "
                     f"{s['displacement']:.6g}{unit} along {axis}")
        if s['definition'] == 'amplitude_r':
            lines.append(f"(strain amplitude {s['strain_amplitude']:g}, R = {s['r_ratio']:g})")
    if 'hold_time' in s:
        lines.append(f"then a hold of {s['hold_time']:g} time units")
    lines.append(f"total time {s['total_time']:g}")
    lines.extend(s['notes'])
    return lines
