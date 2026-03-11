import numpy as np
import matplotlib.pyplot as plt

class transTemporalGSSLICEVoids:
    """
    IDEA: In a two step process, firstly, map grains of a temporal GS
      image slice onto a different GS image slice and in the second step,
      set the value to void state.
    
    APPLICATIONS:
      (1) Include non-geometric voids in the Finite Element Model enabling 
      structure property relationship studies for the efect of void 
      size and shape distributions on the mechanical behaviour of the CPFEM
      model.

    Import
    ------
    from upxo.imageOps.voids import transTemporalGSSLICEVoids as gsvoids
    """
    valid_imgCATEGORY = ('sources', 's', 'targets', 't')
    __slots__ = ('img', 'nvoids', 'void_centers', 'void_radii',
                 'dim'
                 )
    def __init__(self):
        pass

    def load_images(self, imgCATEGORY, imgs):
        if imgCATEGORY not in self.valid_imgCATEGORY:
            raise ValueError(f"imgCATEGORY: {imgCATEGORY} invalid.")
        if isinstance(imgs, dict):
            _enum_ = enumerate(imgs.values(), start=0)
            self.img[imgCATEGORY] = {i: img for i, img in _enum_}
            self.dim[imgCATEGORY] = {i: len(img.shape) for i, img in _enum_}
        elif isinstance(imgs, np.ndarray):
            self.img[imgCATEGORY] = {0: imgs}
            self.dim[imgCATEGORY] = {0: len(imgs.shape)}
        else:
            raise ValueError("targetLFI must be a dict or numpy ndarray")
    
    def load_k(self):
        pass