import numpy as np
from upxo.ggrowth.make3d import voxel_from_pixel


class freature_tracker(voxel_from_pixel):
    def __init__(self, STACK, meta_dict={'creation': 'from_sstack'},
                 ):
        """Initialise the instance."""
        super().__init__(STACK, meta_dict=meta_dict)