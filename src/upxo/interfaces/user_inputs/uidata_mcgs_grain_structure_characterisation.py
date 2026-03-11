class _uidata_mcgs_grain_structure_characterisation_:
    """
    CALL:
        from mcgs import _uidata_mcgs_grain_structure_characterisation_
        uidata_grain_identification = _uidata_mcgs_grain_structure_characterisation_(uidata)
    """
    DEV = True
    __slots__ = ('grain_identification_library', '__uisim_lock__'
                 )

    def __init__(self, uidata):
        self.grain_identification_library = uidata['grain_identification_library']

    def __repr__(self):
        str0 = 'Attributes of grain structure analysis parameters: \n'
        str1 = '    grain_identification_library'
        return str0 + str1