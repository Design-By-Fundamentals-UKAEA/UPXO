"""Feature tracking across stacked 2D MC slices assembled into 3D volumes."""

import numpy as np
from upxo.ggrowth.make3d import voxel_from_pixel


class freature_tracker(voxel_from_pixel):
    """
    Track features across a stack of 2D slices reconstructed as a 3D volume.

    Subclasses :class:`~upxo.ggrowth.make3d.voxel_from_pixel` so temporal
    or spatial 2D GS stacks become ``s`` / ``lfi`` volumes for
    cross-slice feature correspondence. Class name keeps the historical
    spelling ``freature_tracker``.
    """
    def __init__(self, STACK, meta_dict={'creation': 'from_sstack'},
                 ):
        """Initialise the instance."""
        super().__init__(STACK, meta_dict=meta_dict)