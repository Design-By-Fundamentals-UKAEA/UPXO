from copy import deepcopy
import numpy as np
import pandas as pd
import defdap.ebsd as defDap_ebsd
# from defdap.quat import Quat
from scipy.ndimage import generic_filter
from upxo._sup.validation_values import _validation

class ebsd_data():
    """
    """
    __slots__ = ('map_raw',
                 'map', 'gbjp',
                 'gid',
                 'ea_avg',
                 'prop',
                 'n',
                 'quat_avg',
                 'fileName',
                 'val',
                 )

    def __init__(self, ebsdmapdata_raw):
        self.map_raw = ebsdmapdata_raw
        self.gbjp = None
        self.val = None

    @classmethod
    def load_ctf(cls, fileName=None):
        """
        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 20  # Temporal slice number
        pxt.char_morph_2d(tslice)
        pxt.gs[tslice].export_ctf(r'D:\export_folder', 'sunil')

        from upxo.interfaces.defdap.importebsd import ebsd_data
        fileName = r'D:\export_folder\sunil'
        ebsd_data.load_ctf(fileName)
        """
        # VALIDATE IF FILE EXISTS
        map_raw = defDap_ebsd.Map(fileName, dataType="OxfordText")
        return cls(map_raw)

    def _port_defdap_to_upxo(self):
        # FIRST UNPACK THE DEFDAP DATASETS TO UPXO NATIVE
        self._unpack_defdap()

    def _unpack_defdap(self):
        '''
        Following defdap data-sets will be unpacked to UPXO
        ---------------------------------------------------
        defdep_map.grains
        defdep_map.eulerAngleArray
        defdep_map.quatArray
        defdep_grainList[0].coordList
        '''
        self.lgi = self.map.grains
        self.ea = self.map.eulerAngleArray
        self.quats = self.quatArray
        self.g_ref_orientation

    def build_quatarray(self):
        self.map.buildQuatArray()

    def detect_grains(self, size_min=10):
        self.map.findGrains(minGrainSize=size_min)

    def calc_grain_ori_avg(self):
        self.map.calcGrainMisOri()
