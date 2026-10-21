"""How big the analysis output will be: degrees of freedom, frames and an estimate of the ODB.

The estimate is rough. Every requested variable is stored per frame for every node or element it covers
(4-byte values, one component per column of the variable, and a 4-byte label per entry); C3D4 has one
integration point. History output, the restart file and the compression of the file are not counted, and
the factor ``ODB_SCALE`` is where a measured ODB would be used to tune it.
"""
import math

NODE_COMPONENTS = {'U': 3, 'RF': 3, 'V': 3, 'A': 3, 'CF': 3, 'COORD': 3, 'NT': 1, 'RT': 1}
ELEMENT_COMPONENTS = {'S': 6, 'E': 6, 'LE': 6, 'NE': 6, 'EE': 6, 'PE': 6, 'THE': 6, 'PEEQ': 1, 'IVOL': 1, 'EVOL': 1,
                      'ENER': 1, 'ESEDEN': 1, 'SENER': 1, 'MISES': 1}
BYTES_PER_VALUE = 4
BYTES_PER_LABEL = 4
ODB_SCALE = 1.0


def frame_count(step_times, intervals):
    """Frames the field output requests give: the first frame, then one per output interval of each step."""
    return 1 + sum(int(math.ceil(time / interval - 1e-9)) for time, interval in zip(step_times, intervals))


def estimate_output(element_outputs, node_outputs, n_nodes, n_elements, n_depvar, step_times, intervals,
                    n_output_elements=None):
    """Size figures of the field output; ``n_output_elements`` is the number of elements the output covers."""
    covered = n_elements if n_output_elements is None else int(n_output_elements)
    unknown = []

    def components(name, table):
        if name in table:
            return table[name]
        if name == 'SDV':
            return int(n_depvar)
        unknown.append(name)
        return 1

    per_frame = 0
    for name in node_outputs:
        per_frame += n_nodes * (BYTES_PER_VALUE * components(name, NODE_COMPONENTS) + BYTES_PER_LABEL)
    for name in element_outputs:
        per_frame += covered * (BYTES_PER_VALUE * components(name, ELEMENT_COMPONENTS) + BYTES_PER_LABEL)
    frames = frame_count(step_times, intervals)
    total = per_frame * frames * ODB_SCALE
    return {'dofs': int(3 * n_nodes), 'frames': int(frames), 'bytes_per_frame': float(per_frame),
            'megabytes': float(total / 2 ** 20), 'elements_in_output': int(covered), 'rough': True,
            'unknown_variables': sorted(set(unknown))}


def estimate_for_plan(settings, plan, n_nodes, n_elements, n_output_elements=None):
    """:func:`estimate_output` for a deck's settings and its :class:`~.loading.LoadPlan`."""
    return estimate_output(settings.element_outputs, settings.node_outputs, n_nodes, n_elements, settings.n_depvar,
                           [s.time for s in plan.steps], [s.output_interval for s in plan.steps], n_output_elements)


def text(figures):
    """The figures as one short line."""
    size = figures['megabytes']
    shown = f"{size / 1024:.1f} GB" if size >= 1024 else f"{size:.0f} MB"
    return (f"{figures['dofs']:,} degrees of freedom, {figures['frames']:,} output frames, "
            f"about {shown} of field output (rough)")
