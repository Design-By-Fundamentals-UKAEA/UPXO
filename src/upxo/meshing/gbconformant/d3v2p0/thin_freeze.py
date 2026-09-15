"""Whole-grain freezing from pre-refinement directional voxel thickness."""
from copy import deepcopy
from numbers import Integral
import numpy as np


def detect_thin_grains(labels, *, enabled=True, automatic=True, thickness=1,
                       explicit_ids=()):
    """Flag IDs with any axial same-label run <= thickness voxels.

    Conservative: tips and staircase corners can flag an otherwise large grain.
    This is an axial voxel criterion, not rotation-invariant medial thickness.
    All nonnegative IDs, including zero, are grains; the whole ID is frozen.
    """
    a = np.asarray(labels)
    if a.ndim != 3 or not a.size or a.dtype.kind not in 'iu' or np.any(a < 0):
        raise ValueError('Expected nonnegative 3D integer labels')
    if isinstance(thickness, (bool, np.bool_)) or not isinstance(thickness, Integral) or thickness < 1:
        raise ValueError('thickness must be a positive integer')
    if not all(isinstance(v, (bool, np.bool_)) for v in (enabled, automatic)):
        raise ValueError('Switches must be boolean')
    ids = set(map(int, np.unique(a)))
    explicit = set(explicit_ids)
    if not explicit <= ids: raise ValueError('Unknown explicit grain IDs')
    reasons = {int(g): ['explicit'] for g in explicit} if enabled else {}
    if enabled and automatic:
        for axis in range(3):
            lines = np.moveaxis(a, axis, -1).reshape(-1, a.shape[axis])
            found = set()
            for line in lines:
                cuts = np.r_[0, np.flatnonzero(line[1:] != line[:-1]) + 1, len(line)]
                found.update(map(int, line[cuts[:-1][np.diff(cuts) <= thickness]]))
            for g in found: reasons.setdefault(g, []).append('thin_run_' + 'xyz'[axis])
    return sorted(reasons), dict(enabled=bool(enabled), threshold_voxels=int(thickness),
        frozen_ids=sorted(reasons), reasons=reasons, total_grains=len(ids))


def _clip(poly, axis, bound, lower):
    result = []
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        ip = p[axis] >= bound if lower else p[axis] <= bound
        iq = q[axis] >= bound if lower else q[axis] <= bound
        if ip: result.append(p)
        if ip != iq:
            result.append(p + (q-p) * ((bound-p[axis])/(q[axis]-p[axis])))
    return np.asarray(result)


class FrozenGeometry:
    """Reject changes to protected voxel-face geometry, allowing retriangulation.

    Triangles must be axis aligned and cover the same unit voxel faces by area.
    Polygon clipping checks their entire footprint, not just node samples.
    Existing manifold/intersection validators remain required (area agreement
    alone does not rule out overlapping coverage compensated by a hole).
    """
    def __init__(self, labels, grain_ids, spacing=(1., 1., 1.)):
        self.labels = np.asarray(labels).copy()
        self.ids = set(map(int, grain_ids))
        if not self.ids <= set(np.unique(self.labels)): raise ValueError('Unknown frozen IDs')
        self.spacing = np.broadcast_to(np.asarray(spacing, float), (3,)).copy()
        if not np.isfinite(self.spacing).all() or np.any(self.spacing <= 0): raise ValueError('Invalid spacing')
        self.faces = {}
        a = self.labels
        for gid in self.ids:
            for xyz in np.argwhere(a == gid):
                for axis in range(3):
                    uv = [k for k in range(3) if k != axis]
                    for sign in (-1, 1):
                        other = xyz.copy(); other[axis] += sign
                        outside = np.any(other < 0) or np.any(other >= a.shape)
                        if outside or a[tuple(other)] != gid:
                            key = (gid, axis, int(xyz[axis] + (sign == 1)), int(xyz[uv[0]]), int(xyz[uv[1]]))
                            self.faces[key] = outside

    def check(self, surface):
        if not self.ids: return dict(frozen_grains=0)
        covered = {}
        closed = hasattr(surface, 'rve_face')
        for tri, pair in zip(surface.triangles, surface.grain_pairs):
            grains = self.ids.intersection(map(int, pair))
            if not grains: continue
            p = surface.points[tri] / self.spacing
            axes = np.flatnonzero(np.ptp(p, axis=0) < 1e-7)
            if len(axes) != 1: raise RuntimeError('Frozen grain boundary is no longer a voxel plane')
            axis = int(axes[0]); plane = int(round(p[0, axis]))
            if abs(p[0, axis] - plane) > 1e-7: raise RuntimeError('Frozen voxel plane moved')
            q = p[:, [k for k in range(3) if k != axis]]
            for i in range(int(np.floor(q[:, 0].min())), int(np.ceil(q[:, 0].max()))):
                for j in range(int(np.floor(q[:, 1].min())), int(np.ceil(q[:, 1].max()))):
                    poly = q
                    for k, bound, lower in ((0,i,True),(0,i+1,False),(1,j,True),(1,j+1,False)):
                        if len(poly): poly = _clip(poly, k, bound, lower)
                    if len(poly) < 3: continue
                    # Translate before shoelace to avoid cancellation at large coordinates.
                    poly = poly - poly[0]
                    area = abs(np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1)))/2
                    if area < 1e-10: continue
                    for gid in grains:
                        key = (gid,axis,plane,i,j)
                        if key not in self.faces: raise RuntimeError('Triangle leaves protected voxel boundary')
                        covered[key] = covered.get(key, 0.) + area
        expected = {k for k, exterior in self.faces.items() if closed or not exterior}
        if set(covered) != expected or any(abs(covered[k]-1.) > 1e-6 for k in expected):
            raise RuntimeError('Frozen boundary coverage changed or a frozen grain was lost')
        return dict(frozen_grains=len(self.ids), checked_voxel_faces=len(expected))

    def run(self, function, surface, *args, **kwargs):
        """Run transactionally; no invalid result is assigned by the notebook."""
        if not self.ids:
            return function(surface, *args, **kwargs)
        self.check(surface)
        if function.__name__ in ('remesh_interfaces_gmsh', 'remesh_closed_rve_gmsh'):
            kwargs['frozen_grain_ids'] = sorted(self.ids)
        result = function(deepcopy(surface), *args, **kwargs)
        output = result[0] if isinstance(result, tuple) else result
        try:
            self.check(output)
        except RuntimeError as exc:
            raise RuntimeError(f'{function.__name__} conflicts with frozen grains; input retained. {exc}') from exc
        return result
