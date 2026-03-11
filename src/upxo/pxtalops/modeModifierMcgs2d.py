from upxo.ggrowth.mcgs import mcgs
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from skimage.segmentation import find_boundaries
import scipy.spatial.ckdtree as ckdtree
from copy import deepcopy
import pandas as pd

class _modeModifier_():
    __slots__ = ('targetTSlice', 'NSeedGrains', 'meanNeighCount', 'size_threshold',
                 'neighCounts', 'neigh_gid', 'gsTRG', 'sf', 'mods', 'G',
                 'prop', 'nrealizations', 'realizations', 
                 '_misAreas_',
                 'seedGids', 'seedGid_neighs', 'seedGid_coords'
                 )

    def __init__(self, **kwargs):
        self.NSeedGrains = kwargs.get('NSeedGrains', 5)
        self.meanNeighCount = kwargs.get('meanNeighCount', 2)
        self.size_threshold = kwargs.get('size_threshold', 10)
        self.nrealizations = kwargs.get('nrealizations', 1)

        self.mods = {}
        self.prop = pd.DataFrame({'originalSizes': [0], })

        self.realizations = {r: None for r in range(self.nrealizations)}

    def find_neighCounts(self, meanCountOffset=0, deltaPar=1.0):
        self.neighCounts = np.abs(self.meanNeighCount+np.asarray((self.meanNeighCount-meanCountOffset)*(np.random.random(self.NSeedGrains)-deltaPar), 
                                                                 dtype=np.int16))
        print(f"Neighbour counts are: \n{self.neighCounts}")

    def load_neigh(self, neigh_gid):
        self.neigh_gid = neigh_gid

    def load_UPXO_GSTRG(self, gsTRG: dict):
        # gsTRG: Target gs. This is what we intend to modify
        # Key: Tslice ID
        # Value: UPXO Grain structure object
        self.targetTSlice = list(gsTRG.keys())[0]
        self.gsTRG = gsTRG

    def load_UPXO_GSSRC(self, gsSRC):
        # gsTRG: Target gs. This is what we intend to modify
        # Key: Tslice ID
        # Value: UPXO Grain structure object
        self.gsTRG = gsTRG

    def load_scalar_filds(self, sf: dict):
        if 'lfi' not in sf.keys():
            raise ValueError("Input sf sict must contain 'lfi' key.")
        self.sf = sf

    def find_sizeThresholdedNeighs(self):
        self.seedGid_neighs = {gid: np.array([neigh for neigh in neighs if self.prop['originalAreas'][neigh] >= self.size_threshold])
                          for gid, neighs in self.seedGid_neighs.items()}

class modeModifier_mcgs2d(_modeModifier_):

    def find_areas(self, lfi):
        self.prop['originalAreas'] = np.bincount(self.sf['lfi'].ravel())

    def find_seedGrains(self):
        import networkx as nx
        from upxo.netops.kmake import make_gid_net_from_neighlist
        G = make_gid_net_from_neighlist(self.neigh_gid)
        mis = np.array(nx.maximal_independent_set(G), dtype=np.int32)
        misAreas = self.prop['originalAreas'][mis]
        mis = mis[misAreas >= self.size_threshold]
        self.seedGids = np.random.choice(mis, self.NSeedGrains, replace=False)

        self._misAreas_ = misAreas[misAreas >= self.size_threshold]

        _fx = self.gsTRG[self.targetTSlice].get_upto_nth_order_neighbors
        self.seedGid_neighs = {int(gid): np.asarray(_fx(gid, self.neighCounts[gidCount],
                    fast_estimate=False,
                    recalculate=False, include_parent=True,
                    output_type='nparray'), dtype=int) 
                    for gidCount, gid in enumerate(self.seedGids, start=0)}
        self.seedGids = np.array(list(set(np.hstack((self.seedGids,
                                        np.hstack(list(self.seedGid_neighs.values())))))))

    def find_coords_allSeedGids(self):
        self.seedGid_coords = {int(gid): self.gsTRG[self.targetTSlice].g[self.seedGids[gidCount]]['grain'].loc 
                        for gidCount, gid in enumerate(self.seedGids, start=0)}

    def see_seedGids(self, **kwargs):
        if hasattr(self, 'gs'):
            self.gsTRG[self.targetTSlice].plot_grains(gids=self.seedGids,
                            figsize=kwargs.get('figsize', (5, 5)),
                            dpi=kwargs.get('dpi', 75),
                            title=kwargs.get('title', 'Non-touching grains'),
                            )
        else:
            raise ValueError("UPXO grain structure not available. Please input 'gs' attribute")


class modeModifier_mcgs3d(_modeModifier_):
    pass