from termcolor import colored
class _uidata_mcgs_generate_geom_reprs_():
    """
    make_mp_grain_centoids: bool :: Make MP of grain_centroids
    make_mp_grain_points: bool :: Grains as multi-point
    make_ring_grain_boundaries: bool :: GB as UPXO ring
    make_xtal_grain: bool :: Grains as UPXO XTAL object
    make_chull_grain: bool :: Make convex hull for each grain
    create_gbz: bool :: create_grain_boundary_zone

    CALL:
        from mcgs import _uidata_mcgs_generate_geom_reprs_
        uidata_georepr = _uidata_mcgs_generate_geom_reprs_(uidata)
    """
    DEV = True
    __slots__ = ('make_mp_grain_centoids',
                 'make_mp_grain_points',
                 'make_ring_grain_boundaries',
                 'make_xtal_grain',
                 'make_chull_grain',
                 'create_gbz', '__uigeomrepr_lock__'
                 )

    def __init__(self, uidata):
        self.make_mp_grain_centoids = bool(uidata['make_mp_grain_centoids'])
        self.make_mp_grain_points = bool(uidata['make_mp_grain_points'])
        self.make_ring_grain_boundaries = bool(uidata['make_ring_grain_boundaries'])
        self.make_xtal_grain = bool(uidata['make_xtal_grain'])
        self.make_chull_grain = bool(uidata['make_chull_grain'])
        self.create_gbz = bool(uidata['create_gbz'])

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of geometric representation: \n"
        retstr += f"{colored('MAKE_MP_GRAIN_CENTOIDS', 'red')}: {colored(self.make_mp_grain_centoids, 'green')}\n"
        retstr += f"{colored('MAKE_MP_GRAIN_POINTS', 'red')}: {colored(self.make_mp_grain_points, 'green')}\n"
        retstr += f"{colored('MAKE_RING_GRAIN_BOUNDARIES', 'red')}: {colored(self.make_ring_grain_boundaries, 'green')}\n"
        retstr += f"{colored('MAKE_XTAL_GRAIN', 'red')}: {colored(self.make_xtal_grain, 'green')}\n"
        retstr += f"{colored('MAKE_CHULL_GRAIN', 'red')}: {colored(self.make_chull_grain, 'green')}\n"
        retstr += f"{colored('CREATE_GBZ', 'red')}: {colored(self.create_gbz, 'green')}\n"
        return retstr