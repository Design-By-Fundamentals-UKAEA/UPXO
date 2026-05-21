# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 22:16:24 2025

@author: rg5749
"""

class fmsgs():
    __slots__ = ('gs', 'tx', 'clset', 'lpaci', 'lpagi', '')

    @classmethod
    def from_mcgs(self):
        """Construct an ``fmsgs`` instance from an MC grain structure. Not yet implemented."""
        raise NotImplementedError("from_mcgs is not yet implemented.")

    def find_neighbours(self):
        """Find neighbouring grains. Not yet implemented."""
        raise NotImplementedError("find_neighbours is not yet implemented.")
