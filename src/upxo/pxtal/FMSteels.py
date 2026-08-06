"""Ferritic-martensitic steel grain structures for UPXO."""

class fmsgs():
    """
    Ferritic–martensitic steel grain-structure container (API stub).

    Planned holder for hierarchical FM steel data (base GS, texture,
    cluster sets, PAG indices). Prefer the active
    :mod:`upxo.pxtal.fm_steel_3d` pipeline for production work.
    Methods raise ``NotImplementedError`` until implemented.
    """
    __slots__ = ('gs', 'tx', 'clset', 'lpaci', 'lpagi', '')

    @classmethod
    def from_mcgs(self):
        """Construct an ``fmsgs`` instance from an MC grain structure. Not yet implemented."""
        raise NotImplementedError("from_mcgs is not yet implemented.")

    def find_neighbours(self):
        """Find neighbouring grains. Not yet implemented."""
        raise NotImplementedError("find_neighbours is not yet implemented.")
