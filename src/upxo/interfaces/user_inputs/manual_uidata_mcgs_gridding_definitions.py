import numpy as np
from termcolor import colored

class _manual_uidata_mcgs_gridding_definitions_:
    """
    type : str :: Type of underlying grid
    dim: int :: Physical dimensionality of the domain
    xmin : float :: X-coordinate of the start of the simulation domain
    xmax : float :: X-coordinate of the end of the simulation domain
    xinc : float :: X-coordinate increments in the simulation domain
    ymin : float :: Y-coordinate of the start of the simulation domain
    ymax : float :: Y-coordinate of the end of the simulation domain
    yinc : float :: Y-coordinate increments in the simulation domain
    zmin : float :: Z-coordinate of the start of the simulation domain
    zmax : float :: Z-coordinate of the end of the simulation domain
    zinc : float :: Z-coordinate increments in the simulation domain
    px_size: float :: Pixel size in the grid
    transformation: str :: Geometric transformation operation for the grid
    __lock__: dict :: Sub-locks (type, npixx, npixy, npix) and summary lock (_)

    CALL:
        from mcgs import _manual_uidata_mcgs_gridding_definitions_ as imname
        uidata_gridpar = imname(domainsize=Value,
                                read_from_file=Value,
                                filename=Value)
    """
    DEV = True
    __slots__ = ('type', 'dim',
                 'xmin', 'xmax', 'xinc',
                 'ymin', 'ymax', 'yinc',
                 'zmin', 'zmax', 'zinc',
                 'px_size',
                 'transformation')
    __npixx__ = 500  # Number of pixels along x-axis
    __npixy__ = 500  # Number of pixels along y-axis
    __npixels_max__ = __npixx__*__npixy__
    __lock__ = {'type': False,  # True if invalid type
                'npixx': False,  # True if npixx > self.npixx
                'npixy': False,  # True if npixy > self.npixy
                'npix': False,  # True if npix > self.npixx*self.npixy
                '_': False,  # True if any of above is True
                }

    def __init__(self,
                 domain_size=None,
                 read_from_file=False, filename=None
                 ):
        self.__lock__['_'] = False
        # ----------------------------------
        if not read_from_file:
            self.type = 'square'
            self.xmin, self.xmax = domain_size[0][0], domain_size[0][1]
            self.xinc = domain_size[3]
            self.ymin, self.ymax = domain_size[1][0], domain_size[1][1]
            self.yinc = domain_size[3]
            self.zmin, self.zmax = domain_size[2][0], domain_size[2][1]
            self.zinc = domain_size[3]
            self.px_size = self.xinc*self.yinc*self.zinc
            # ----------------------------------
            if 2 in (len(domain_size[0]),
                     len(domain_size[1]),
                     len(domain_size[2])):
                self.dim = 2
            elif len(domain_size[0])==3 and len(domain_size[1])==3 and len(domain_size[2])==3:
                self.dim = 3
            else:
                print('Invalid grid specification')
                self.__lock__ = 'locked'
            # ----------------------------------
            self.transformation = None
            # ----------------------------------
            _, __, npixels = self.grid
            if npixels >= self.__npixels_max__:
                self.__lock__['_'] = True
        else:
            pass
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = 'Attribues of gridding definitions: \n'
        retstr += _ + f"{colored('TYPE', 'red', attrs=['bold'])}: {colored(self.type, 'cyan')}\n"
        retstr += _ + f"{colored('DIMENSIONALITY', 'red', attrs=['bold'])}: {colored(self.dim, 'cyan')}\n"
        retstr += _ + f"{colored('X', 'red', attrs=['bold'])}: ({colored(self.xmin, 'cyan')}, {colored(self.xmax, 'cyan')}, {colored(self.xinc, 'cyan')})\n"
        retstr += _ + f"{colored('Y', 'red', attrs=['bold'])}: ({colored(self.ymin, 'cyan')}, {colored(self.ymax, 'cyan')}, {colored(self.yinc, 'cyan')})\n"
        retstr += _ + f"{colored('Z', 'red', attrs=['bold'])}: ({colored(self.zmin, 'cyan')}, {colored(self.zmax, 'cyan')}, {colored(self.zinc, 'cyan')})\n"
        retstr += _ + f"{colored('PIXEL SIZE', 'red', attrs=['bold'])}: {colored(self.px_size, 'cyan')}\n"
        retstr += _ + f"{colored('TRANSFORMATION', 'red', attrs=['bold'])}: {colored(self.transformation, 'cyan')}"
        return retstr

    @property
    def xbound(self):
        return (self.xmin, self.xmax, self.xinc)

    @property
    def ybound(self):
        return (self.ymin, self.ymax, self.yinc)

    @property
    def zbound(self):
        return (self.zmin, self.zmax, self.zinc)

    @property
    def xls(self):
        # Make the linear space for x
        return np.linspace(self.xmin,
                           self.xmax,
                           int((self.xmax-self.xmin)/self.xinc+1))

    @property
    def yls(self):
        # Make the linear space for y
        return np.linspace(self.ymin,
                           self.ymax,
                           int((self.ymax-self.ymin)/self.yinc+1))

    @property
    def zls(self):
        pass

    @property
    def grid(self):
        if self.dim == 2:
            # make the grid from linear spaces of x and y
            x, y = np.meshgrid(self.xls, self.yls)
            npixels = x.size*y.size
        return x, y, npixels

    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist','red',attrs=['bold'])}")
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
