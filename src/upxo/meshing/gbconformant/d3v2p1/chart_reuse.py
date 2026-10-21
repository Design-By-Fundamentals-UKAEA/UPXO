"""Reuse of meshed charts across the retry passes of remesh_interfaces_gmsh (opt-in).

A retry pass rebuilds the whole Gmsh model with the failed charts split into smaller disks. Every other chart is
unchanged: the same source triangles (so the same retained boundary edges, which are source edges) and the same
points. Meshing it again gives the mesh it already had. With ``ChartCache`` the charts that came out of a pass
with a valid mesh are kept as arrays. In the next pass those charts are hidden from ``generate(2)`` (and not
parametrized), so only the new and the failed charts are meshed, and their kept meshes are added to the assembled
surface as arrays. They are not put into Gmsh: adding meshes to a model entity by entity gets slower with every
call as the model grows (about 1 ms per call at 2,500 charts, quadratic in all).

A chart is identified by the sorted ids of its source triangles. Its stored mesh is the coordinates of its
interior nodes and its triangles, whose corners are either retained boundary node tags (the same in every pass:
they are source node tags) or interior nodes. Interior nodes are numbered after Gmsh's, so node numbering
differs from a run without reuse; the triangles of every chart do not.

A chart is kept only if Gmsh logged no warning that names it. Everything that makes a pass acceptable is checked
afterwards on the assembled surface exactly as without reuse (boundary conformity, orientation, intersections,
openings, volumes).
"""
import hashlib
import re

import numpy as np


def chart_key(ids):
    """Identity of a chart: its source triangle ids, order-free."""
    return hashlib.blake2b(np.sort(np.asarray(ids, dtype=np.int64)).tobytes(), digest_size=16).digest()


def warned_surfaces(messages):
    """Gmsh surface tags that any warning or error message in ``messages`` names."""
    tags = set()
    for message in messages:
        if message.lower().startswith(('warning', 'error')):
            tags.update(int(m) for m in re.findall(r'surface (\d+)', message, flags=re.IGNORECASE))
    return tags


class ChartCache:
    """Meshed charts kept between the passes of one remeshing call (and its retries)."""

    def __init__(self):
        self._charts = {}
        self.passes = []

    def __contains__(self, key):
        return key in self._charts

    def __len__(self):
        return len(self._charts)

    def harvest(self, gmsh, charts, exclude=()):
        """Keep the 2D mesh of every chart (tag -> source triangle ids) not in ``exclude`` that has one.
        Returns the number of charts newly kept."""
        added = 0
        for tag, ids in charts.items():
            if tag in exclude:
                continue
            key = chart_key(ids)
            if key in self._charts:
                continue
            _, conn = gmsh.model.mesh.getElementsByType(2, tag)
            if not len(conn):
                continue
            conn = np.asarray(conn, dtype=np.int64)
            node_tags, coords, _ = gmsh.model.mesh.getNodes(2, tag, False, False)
            node_tags = np.asarray(node_tags, dtype=np.int64)
            coords = np.asarray(coords, dtype=float).reshape(-1, 3).copy()
            if len(node_tags):
                order = np.argsort(node_tags)
                where = np.searchsorted(node_tags[order], conn)
                where = np.minimum(where, len(order) - 1)
                interior = node_tags[order][where] == conn
                local = np.where(interior, -(order[where] + 1), conn)         # interior: -(index + 1)
            else:
                local = conn.copy()
            self._charts[key] = (coords, local.reshape(-1, 3))
            added += 1
        return added

    def tags_with_meshes(self, charts):
        """Tags of the charts (tag -> source triangle ids) this cache holds a mesh for."""
        return {tag for tag, ids in charts.items() if chart_key(ids) in self._charts}

    def meshes(self, charts, tags, first_tag):
        """The kept meshes of the charts in ``tags`` as arrays, in the numbering of a model whose largest node tag
        is ``first_tag - 1``: (tag -> (m, 3) triangle corners as node tags, tags of the interior nodes, their
        coordinates). Corners that are retained boundary nodes keep their tag; interior nodes get ``first_tag``,
        ``first_tag + 1``, ... Nothing is put into Gmsh: adding meshes to a model entity by entity takes longer
        with every call as the model grows."""
        connections, node_tags, node_coords = {}, [], []
        used = first_tag
        for tag in sorted(tags):
            coords, local = self._charts[chart_key(charts[tag])]
            connections[tag] = np.where(local < 0, used - local - 1, local)
            if len(coords):
                node_tags.append(np.arange(used, used + len(coords), dtype=np.int64))
                node_coords.append(coords)
            used += len(coords)
        if node_tags:
            return connections, np.concatenate(node_tags), np.vstack(node_coords)
        return connections, np.zeros(0, dtype=np.int64), np.zeros((0, 3))

    def note_pass(self, depth, charts, reused, kept):
        self.passes.append(dict(depth=int(depth), charts=int(charts), reused=int(reused),
                                meshed=int(charts - reused), kept_after_pass=int(kept)))

    def summary(self):
        return dict(enabled=True, passes=list(self.passes), charts_kept=len(self._charts),
                    charts_reused_total=int(sum(p['reused'] for p in self.passes)),
                    charts_meshed_total=int(sum(p['meshed'] for p in self.passes)))
