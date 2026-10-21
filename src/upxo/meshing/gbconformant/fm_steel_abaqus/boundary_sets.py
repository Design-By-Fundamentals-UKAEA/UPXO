"""Grain-boundary and junction nodes of a tetrahedral mesh, by feature.

A node belongs to every feature whose tetrahedra use it. A node shared by two or more features lies
on a grain boundary; one shared by three or more lies on a junction (a triple line or a corner of
several features). The features are the ones the deck's element sets stand for, so two meshed grains
that were merged into one feature have no boundary between them.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BoundaryNodes:
    """Node indices (0-based, as in the tetrahedra array) of the boundary sets."""
    grain_boundary: np.ndarray
    junction: np.ndarray
    by_feature: dict = field(default_factory=dict)      # {feature label: boundary nodes of that feature}
    n_nodes: int = 0

    def counts(self):
        return {'grain_boundary_nodes': int(len(self.grain_boundary)),
                'junction_nodes': int(len(self.junction)),
                'features_with_boundary_nodes': len(self.by_feature),
                'nodes': int(self.n_nodes)}


def boundary_nodes(tetrahedra, tet_feature, n_nodes=None):
    """Boundary nodes from the tetrahedra (0-based node indices) and one feature label per tetrahedron."""
    t = np.asarray(tetrahedra, dtype=np.int64)
    f = np.asarray(tet_feature, dtype=np.int64)
    if t.ndim != 2 or t.shape[1] != 4 or f.shape != (len(t),):
        raise ValueError('Expected Mx4 tetrahedra and one feature label per tetrahedron')
    n_nodes = int(t.max()) + 1 if n_nodes is None else int(n_nodes)
    labels, f_index = np.unique(f, return_inverse=True)
    n_labels = len(labels)
    pair = np.unique(t.ravel() * n_labels + np.repeat(f_index, 4))
    node, feat = pair // n_labels, pair % n_labels
    shared = np.bincount(node, minlength=n_nodes)
    on_boundary = shared[node] >= 2
    chosen = np.flatnonzero(on_boundary)
    order = chosen[np.argsort(feat[chosen], kind='stable')]
    cuts = np.flatnonzero(np.diff(feat[order])) + 1
    starts = np.concatenate(([0], cuts)) if len(order) else np.zeros(0, dtype=np.int64)
    stops = np.concatenate((cuts, [len(order)])) if len(order) else np.zeros(0, dtype=np.int64)
    by_feature = {int(labels[feat[order[a]]]): np.sort(node[order[a:b]]) for a, b in zip(starts, stops)}
    return BoundaryNodes(grain_boundary=np.flatnonzero(shared >= 2), junction=np.flatnonzero(shared >= 3),
                         by_feature=by_feature, n_nodes=n_nodes)
