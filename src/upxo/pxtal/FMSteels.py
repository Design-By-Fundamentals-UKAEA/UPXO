"""Ferritic-martensitic steel grain structures for UPXO."""

class fmsgs():
    __slots__ = ('gs', 'tx', 'clset', 'lpaci', 'lpagi', '')

    @classmethod
    def from_mcgs(self):
        """Construct an ``fmsgs`` instance from an MC grain structure. Not yet implemented."""
        raise NotImplementedError("from_mcgs is not yet implemented.")

    def find_neighbours(self):
        """Find neighbouring grains. Not yet implemented."""
        raise NotImplementedError("find_neighbours is not yet implemented.")
