"""Rescaled views of a pipeline state, for meshing at another resolution.

``mesh_state_for_scale`` returns the state itself for a factor of 1 and a
read-through ``RescaledMeshState`` otherwise. The source state is never
modified, so a series of exports at different factors always starts from the
same stored grid.
"""
import numpy as np

import upxo.gsdataops.grid_ops as gridOps

__all__ = ['RescaledMeshState', 'scaled_grid_shape', 'mesh_state_for_scale']


def scaled_grid_shape(shape, scale_factor: float) -> tuple:
    """Grid shape after rescaling by ``scale_factor``.

    Raises ValueError if the factor is not a finite number greater than zero
    or if any dimension would hold less than one voxel.
    """
    factor = float(scale_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("Voxel scale factor must be a finite number greater than zero.")
    target_shape = tuple(int(round(n * factor)) for n in shape)
    if any(n < 1 for n in target_shape):
        raise ValueError(
            f"Voxel scale factor {factor:g} makes the export grid invalid: "
            f"{target_shape}. Every dimension must contain at least one voxel."
        )
    return target_shape


class RescaledMeshState:
    """Read-through view of a pipeline state with rescaled spatial data.

    The source state is never modified. Non-spatial hierarchy/orientation
    attributes are delegated to it, while the voxel grid and every voxel-
    location dictionary consumed by the mesh exporter are rebuilt at the
    requested resolution.
    """

    def __init__(self, source, scale_factor: float):
        self._source = source
        self.scale_factor = float(scale_factor)
        self.lgi = gridOps.rescale_grid_3d(
            source.lgi, self.scale_factor, method="nearest")
        self.voxel_size = source.voxel_size / self.scale_factor
        self.grain_locs = self._locations_from_labels(
            self.lgi, getattr(source, "grain_locs", {}).keys())

        for attr in ("all_blocks", "all_subblocks"):
            if hasattr(source, attr):
                setattr(
                    self,
                    attr,
                    self._rescale_location_groups(
                        getattr(source, attr), source.lgi.shape,
                        self.scale_factor),
                )

    def __getattr__(self, name):
        return getattr(self._source, name)

    @staticmethod
    def _locations_from_labels(label_grid, labels):
        """Extract all label locations with one grid scan, not one per label."""
        labels = list(labels)
        empty = np.empty((0, 3), dtype=np.intp)
        if not labels:
            return {}

        # FM-steel label grids reserve zero for background/void. Sorting the
        # occupied coordinates once avoids an O(n_labels * n_voxels) sequence
        # of full-grid comparisons on large (e.g. 200^3) exports.
        coordinates = np.argwhere(label_grid != 0)
        if not len(coordinates):
            return {label: empty for label in labels}
        values = label_grid[tuple(coordinates.T)]
        order = np.argsort(values, kind="stable")
        coordinates = coordinates[order]
        values = values[order]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        ends = np.r_[starts[1:], len(values)]
        by_label = {
            value.item() if hasattr(value, "item") else value: coordinates[start:end]
            for value, start, end in zip(values[starts], starts, ends)
        }
        return {label: by_label.get(label, empty) for label in labels}

    @staticmethod
    def _rescale_location_groups(groups, source_shape, scale_factor):
        """Resample a mutually-exclusive voxel-location mapping as labels."""
        if not groups:
            return {}

        keys = list(groups)
        group_grid = np.zeros(source_shape, dtype=np.int32)
        for group_index, key in enumerate(keys, start=1):
            locations = np.asarray(groups[key])
            if locations.size:
                group_grid[tuple(locations.T)] = group_index

        scaled_grid = gridOps.rescale_grid_3d(
            group_grid, scale_factor, method="nearest")
        indexed_locations = RescaledMeshState._locations_from_labels(
            scaled_grid, range(1, len(keys) + 1))
        return {
            key: indexed_locations[group_index]
            for group_index, key in enumerate(keys, start=1)
        }


def mesh_state_for_scale(fm_state, scale_factor: float):
    """Return an export state derived afresh from the stored pipeline state."""
    factor = float(scale_factor)
    if factor == 1.0:
        return fm_state
    scaled_grid_shape(fm_state.lgi.shape, factor)       # validates the factor and the grid
    return RescaledMeshState(fm_state, factor)
