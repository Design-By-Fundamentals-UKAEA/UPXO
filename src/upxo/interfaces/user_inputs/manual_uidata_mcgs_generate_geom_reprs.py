from termcolor import colored
class _manual_uidata_mcgs_generate_geom_reprs_():
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
    __lock__ = {'mp': False,
                'ring': False,
                'xtal': False,
                'chull': False,
                'gbz': False,
                '_': True
                }
    __slots__ = ('make_mp_grain_centoids', 'make_mp_grain_points',
                 'make_ring_grain_boundaries', 'make_xtal_grain',
                 'make_chull_grain', 'create_gbz', 'gbz_thickness',
                 )

    def __init__(self,
                 make_mp_grain_centoids=True,
                 make_mp_grain_points=True,
                 make_ring_grain_boundaries=True,
                 make_xtal_grain=True,
                 make_chull_grain=True,
                 create_gbz=True,
                 gbz_thickness=0.1,
                 read_from_file=False, filename=None
                 ):
        self.make_mp_grain_centoids = make_mp_grain_centoids
        self.make_mp_grain_points = make_mp_grain_points
        self.make_ring_grain_boundaries = make_ring_grain_boundaries
        self.make_xtal_grain = make_xtal_grain
        self.make_chull_grain = make_chull_grain
        self.create_gbz = create_gbz
        self.gbz_thickness = gbz_thickness
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = "Geometric representation parameters: \n"
        retstr += _ + f"{colored('MAKE_MP_GRAIN_CENTOIDS', 'red', attrs=['bold'])}: {colored(self.make_mp_grain_centoids, 'cyan')}\n"
        retstr += _ + f"{colored('MAKE_MP_GRAIN_POINTS', 'red', attrs=['bold'])}: {colored(self.make_mp_grain_points, 'cyan')}\n"
        retstr += _ + f"{colored('MAKE_RING_GRAIN_BOUNDARIES', 'red', attrs=['bold'])}: {colored(self.make_ring_grain_boundaries, 'cyan')}\n"
        retstr += _ + f"{colored('MAKE_XTAL_GRAIN', 'red', attrs=['bold'])}: {colored(self.make_xtal_grain, 'cyan')}\n"
        retstr += _ + f"{colored('MAKE_CHULL_GRAIN', 'red', attrs=['bold'])}: {colored(self.make_chull_grain, 'cyan')}\n"
        retstr += _ + f"{colored('CREATE_GBZ', 'red', attrs=['bold'])}: {colored(self.create_gbz, 'cyan')}\n"
        return retstr

    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist', 'red', attrs=['bold'])}")
        return self.__lock__['_']

    @property
    def locks(self):
        return self.__lock__

    @property
    def lock_update(self):
        """
        Updates the lock. Outcome could be leave lock either in open or
        locked state. Returns None.
        """
        keys = list(self.locks.keys())
        keys[list(self.locks.keys()).index('_')] = -1
        keys.remove(-1)
        sublocks = [self.locks[key] for key in keys]
        if True in set(sublocks):
            self.__lock__['_'] = True
        else:
            self.__lock__['_'] = False
        return None