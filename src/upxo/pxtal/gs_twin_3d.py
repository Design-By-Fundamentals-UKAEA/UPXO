# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 11:47:44 2025

@author: rg5749
"""

from upxo.pxtal.mcgs3_temporal_slice import mcgs3_grain_structure

class OFHC_Cu(mcgs3_grain_structure):
    __slots__ = ('a',) + mcgs3_grain_structure.__slots__
    def __init__(self, a=None, *args, **kwargs):
        """Initialise OFHC copper grain structure, delegating to ``mcgs3_grain_structure``."""
        super().__init__(*args, **kwargs)