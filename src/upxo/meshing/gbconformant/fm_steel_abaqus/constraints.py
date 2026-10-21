"""Constraint template for the deck: what the face nodes allow, and the lines to adapt.

No constraint is active in the deck. The template reports, for each axis, how many nodes of the
low face have a partner on the high face at the same position in the other two coordinates (the
condition for a periodic constraint node by node), and shows the lines of a periodic, a tie and a
contact definition with the deck's own set names. Opposite faces of a conformal mesh need not carry
matching nodes; when none match, periodic constraints need a mesh made with matching faces.
"""
import numpy as np
from scipy.spatial import cKDTree

AXES = 'xyz'


def periodic_pairs(base_points, faces, tolerance):
    """{axis letter: {'pairs': (k, 2) 1-based node numbers [low, high], 'low': n, 'high': n, 'max_offset': d}}.

    ``faces`` maps ``XMIN``..``ZMAX`` to 1-based node numbers of ``base_points``. A pair is the nearest
    high-face node within ``tolerance`` in the two other coordinates; each node is used at most once.
    """
    out = {}
    for axis, letter in enumerate(AXES):
        low = np.asarray(faces[f'{letter.upper()}MIN'], dtype=np.int64)
        high = np.asarray(faces[f'{letter.upper()}MAX'], dtype=np.int64)
        other = [a for a in range(3) if a != axis]
        pairs = np.zeros((0, 2), dtype=np.int64)
        worst = 0.0
        if len(low) and len(high):
            tree = cKDTree(base_points[high - 1][:, other])
            distance, index = tree.query(base_points[low - 1][:, other], distance_upper_bound=tolerance)
            found = np.isfinite(distance)
            if found.any():
                candidates = np.column_stack((low[found], high[index[found]]))
                _, first = np.unique(candidates[:, 1], return_index=True)     # one low node per high node
                pairs = candidates[np.sort(first)]
                worst = float(distance[found].max())
        out[letter] = {'pairs': pairs, 'low': int(len(low)), 'high': int(len(high)), 'max_offset': worst}
    return out


def summary(pairs):
    """The counts of :func:`periodic_pairs` without the node numbers."""
    return {letter: {'matched_pairs': int(len(item['pairs'])), 'low_face_nodes': item['low'],
                     'high_face_nodes': item['high'], 'max_offset_mesh_units': item['max_offset']}
            for letter, item in pairs.items()}


def template_text(pairs, nset_names, instance, load_axis):
    """The content of ``07_interactions.inp``: comments only, so the deck defines no constraint."""
    lines = ['** Interactions / constraints: template, no constraint is active in this deck.',
             '**',
             '** Periodic pairs: a node of a low face and the node of the opposite face at the same position in',
             '** the other two coordinates (counts are for this mesh).']
    for letter, item in pairs.items():
        lines.append(f'**   {letter}: {len(item["pairs"])} matched pairs, {item["low"]} low-face nodes, '
                     f'{item["high"]} high-face nodes')
    if not any(len(item['pairs']) for item in pairs.values()):
        lines += ['**   No pair matches: the faces carry different nodes, so a periodic constraint node by node',
                  '**   is not possible on this mesh.']
    axis = load_axis if load_axis in pairs else 'z'
    example = pairs[axis]['pairs']
    node_low, node_high = (int(example[0][0]), int(example[0][1])) if len(example) else ('<low node>', '<high node>')
    lines += [
        '**',
        '** Periodic constraint of one pair along the load axis: high node = low node + the relative motion',
        '** carried by a reference node, repeated for each degree of freedom and each pair.',
        '** The reference node and its degrees of freedom are yours to define.',
        '**   *Equation',
        '**   3',
        f'**   {instance}.{node_high}, 1, 1.',
        f'**   {instance}.{node_low}, 1, -1.',
        '**   <reference node set>, 1, -1.',
        '**',
        '** Tie of two surfaces (needs surfaces built from element faces, not written by this deck):',
        '**   *Tie, name=tie-1, adjust=no',
        '**   <slave surface>, <master surface>',
        '**',
        '** Face node sets available to constraints and boundary conditions: '
        + ', '.join(nset_names[key] for key in ('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX')),
        '**',
    ]
    return '\n'.join(lines) + '\n'
