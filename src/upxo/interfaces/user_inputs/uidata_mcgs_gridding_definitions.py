import numpy as np
from termcolor import colored

class _uidata_mcgs_gridding_definitions_:
    """
    This class represents the gridding definitions for the MCGS simulation.

    Attributes:
        type (str): Type of underlying grid.
        dim (int): Physical dimensionality of the domain.
        xmin (float): X-coordinate of the start of the simulation domain.
        xmax (float): X-coordinate of the end of the simulation domain.
        xinc (float): X-coordinate increments in the simulation domain.
        ymin (float): Y-coordinate of the start of the simulation domain.
        ymax (float): Y-coordinate of the end of the simulation domain.
        yinc (float): Y-coordinate increments in the simulation domain.
        zmin (float): Z-coordinate of the start of the simulation domain.
        zmax (float): Z-coordinate of the end of the simulation domain.
        zinc (float): Z-coordinate increments in the simulation domain.
        transformation (str): Geometric transformation operation for the grid.

    Methods:
        __init__(self, uidata): Initializes the gridding definitions object.
        __repr__(self): Returns a string representation of the gridding definitions.
        xbound(self): Returns the x-coordinate bounds as a tuple.
        ybound(self): Returns the y-coordinate bounds as a tuple.
        zbound(self): Returns the z-coordinate bounds as a tuple.
        xls(self): Returns a linear space for the x-coordinate.
        yls(self): Returns a linear space for the y-coordinate.
        zls(self): Returns a linear space for the z-coordinate.
        grid(self): Returns the grid coordinates and the number of pixels.

    Usage:
        from mcgs import _uidata_mcgs_gridding_definitions_
        uidata_gridpar = _uidata_mcgs_gridding_definitions_(uidata)
    """

    DEV = True
    __slots__ = ('type', 'dim', 'xmin', 'xmax', 'xinc',
                 'ymin', 'ymax', 'yinc', 'zmin', 'zmax', 'zinc', 'px_size',
                 'transformation', '__npixles_lock__', '__type_lock__')

    npixels_max = 500*500

    def __init__(self, uidata):
        """
        Initializes the gridding definitions object.

        Args:
            uidata (dict): Dictionary containing the gridding parameters.

        Returns:
            None
        """
        self.type, self.dim = uidata['type'], int(uidata['dim'])
        self.xmin, self.xmax = uidata['xmin'], uidata['xmax']
        self.xinc = uidata['xinc']
        self.ymin, self.ymax = uidata['ymin'], uidata['ymax']
        self.yinc = uidata['yinc']
        self.zmin, self.zmax = uidata['zmin'], uidata['zmax']
        self.zinc = uidata['zinc']
        self.px_size = self.xinc*self.yinc*self.zinc
        self.transformation = uidata['transformation']
        _, __, npixels = self.grid
        self.__npixles_lock__ = False
        if npixels >= self.npixels_max:
            self.__npixles_lock__ = True

    def __repr__(self):
        """
        Returns a string representation of the gridding definitions.

        Returns:
            str: String representation of the gridding definitions.
        """
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
        """
        Returns the x-coordinate bounds as a tuple.

        Returns:
            tuple: Tuple containing the x-coordinate bounds.
        """
        return (self.xmin, self.xmax, self.xinc)

    @property
    def ybound(self):
        """
        Returns the y-coordinate bounds as a tuple.

        Returns:
            tuple: Tuple containing the y-coordinate bounds.
        """
        return (self.ymin, self.ymax, self.yinc)

    @property
    def zbound(self):
        """
        Returns the z-coordinate bounds as a tuple.

        Returns:
            tuple: Tuple containing the z-coordinate bounds.
        """
        return (self.zmin, self.zmax, self.zinc)

    @property
    def xls(self):
        """
        Returns a linear space for the x-coordinate.

        Returns:
            numpy.ndarray: Linear space for the x-coordinate.
        """
        return np.linspace(self.xmin,
                           self.xmax,
                           int((self.xmax-self.xmin)/self.xinc+1))

    @property
    def yls(self):
        """
        Returns a linear space for the y-coordinate.

        Returns:
            numpy.ndarray: Linear space for the y-coordinate.
        """
        return np.linspace(self.ymin,
                           self.ymax,
                           int((self.ymax-self.ymin)/self.yinc+1))

    @property
    def zls(self):
        """
        Returns a linear space for the z-coordinate.

        Returns:
            numpy.ndarray: Linear space for the z-coordinate.
        """
        pass

    @property
    def grid(self):
        """
        Returns the grid coordinates and the number of pixels.

        Returns:
            tuple: Tuple containing the grid coordinates and the number of pixels.
        """
        if self.dim == 2:
            # make the grid from linear spaces of x and y
            x, y = np.meshgrid(self.xls, self.yls)
            npixels = x.size*y.size
        elif self.dim == 3:
            x, y, z = np.meshgrid(self.xls, self.yls, self.zls)
            npixels = x.size*y.size*z.size
        return x, y, npixels
