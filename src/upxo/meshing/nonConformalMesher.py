import numpy as np
import time

class nonConformalMesher():
    """
    A class to generate non-conformal meshes for 2D and 3D geometries.
    Supports element types: Quad4, Quad8, Tri3, Tri6, Hex8, Hex20, Tet5.
    2D Element Types: Quad4, Quad8, Tri3, Tri6
    3D Element Types: Hex8, Hex20, Tet5

    Import
    ------
    from upxo.meshing.nonConformalMesher import nonConformalMesher as ncm

    Example
    -------
    lfi = np.array([[1, 3, 3, 3], [4, 1, 3, 2]], dtype=np.int32)
    gids = np.unique(lfi)
    m = ncm.quad4(lfi=lfi, gids=gids, nelx=4, nely=2, 
                  xstart=0.0, ystart=0.0, ellx=0.2, elly=0.1)
    m.mesh(analysis_package='abaqus', elsetNamePrefix='gid_')
    m.see_mesh()
    """

    _valid_analysis_packages = ('abaqus', 'moose', 'damask')

    valid_image_data = ('lfi', 'phmap', 'custom')

    valid_xoriTypes = ('bunge', 'zxz', 'matthies', 'mtex', 
                        'zyz', 'roe', 'kocks', 'canova', 
                        'miller', 'rot', 'quat', 'axisangle')

    ABQ_ENAMES = {"cps": ["CPS3", "CPS4", "CPS4R", "CPS6", "CPS8", "CPS8R",],
                  "cpe": ["CPE3", "CPE4", "CPE4R", "CPE6", "CPE8", "CPE8R",],
                  "cax": ["CAX3", "CAX4", "CAX4R", "CAX6", "CAX8", "CAX8R",],
                  "c3d_tet": ["C3D4", "C3D10", "C3D10M",],
                  "c3d_hex": ["C3D8", "C3D8R", "C3D8I", "C3D20", "C3D20R",],
                  "c3d_wedge": ["C3D6", "C3D15",],
                  "c3d_pyramid": ["C3D5",],}

    __slots__ = ('dim', 'nelx', 'nely', 'nelz', 'xstart', 'ystart', 
                 'zstart',  'ellx', 'elly', 'ellz',  'xincr', 
                 'yincr', 'zincr','xbase', 'ybase', 'zbase',
                 'xndgrid', 'yndgrid', 'zndgrid', 'lfi',
                 'gids', 'ignored_gids', 'phmap', 'nnum', 'NODES', 
                 'etype', 'enum', 'elLoc', 'ELEMENTS', 'el_ids', 
                 'nidStart', 'elifStart', 'vertexNodeIds', 'midSideNodeIds',
                 'boundaryNodeIds', 'boundaryElIds', 'nn', 'nel', 
                 'nnInExRatio', 'nelInExRatio',
                 'ABQ_ELEMENTS', 'elsets', 'connectivity', 'elname',
                 'el_node_coords',  'elCentroids', 'floatDType',
                 'temp_container', 'analysis_package',
                 'xori', 'xoriType', 'xoriUnits', 
                 'plane_stress', 'reduced_integration', 'modifiedFormulation',
                 '_threshold_nel_for_large_mesh_elEdgePlot_',
                 'axiSymmElements'
                 )

    def __init__(self, dim=2, etype='quad4',
                 lfi=np.random.randint(0, 100, size=(10, 10), 
                            dtype=np.int32), gids=None,
                 nelx=10, nely=10, nelz=10,
                 xstart=0, ystart=0, zstart=0,
                 ellx=1, elly=1, ellz=1,
                 plane_stress=True, reduced_integration=False,
                 modifiedFormulation=False, axiSymmElements=False
                 ):
        self.dim = dim
        self.etype = etype
        self.lfi = lfi
        self.set_gids(gids=gids)
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        self.xstart, self.ystart, self.zstart = xstart, ystart, zstart
        self.ellx, self.elly, self.ellz = ellx, elly, ellz
        self.plane_stress = plane_stress
        self.reduced_integration = reduced_integration
        self.axiSymmElements = axiSymmElements
        self.modifiedFormulation = modifiedFormulation
        # IDs of all vertex and mid-side nodes of all elements
        self.vertexNodeIds, self.midSideNodeIds = [], []
        self.boundaryNodeIds = {'x-': [], 'x+': [], 'y-': [], 'y+': [], 'z-': [], 'z+': [],
                                'x-y-': [], 'x-y+': [], 'x+y-': [], 'x+y+': [],
                                'y-z-': [], 'y-z+': [], 'y+z-': [], 'y+z+': [],
                                'x-z-': [], 'x-z+': [], 'x+z-': [], 'x+z+': [],
                                'x-y-z-': [], 'x-y-z+': [], 'x-y+z-': [], 'x-y+z+': [],
                                'x+y-z-': [], 'x+y-z+': [], 'x+y+z-': [], 'x+y+z+': [], }
        # Number of nodes and number of elements
        self.nn, self.nel = 0, 0
        # Ratio of number of nodes inside to no. of nodes on surface of RVE
        # Ratio of number of elements inside to no. of elements on surface of RVE
        self.nnInExRatio, self.nelInExRatio = None, None

        self.floatDType = np.float32
        self.temp_container = {}  # Just a helper variable, noithing more.
        # Orientati0on variables
        self.xori, self.xoriType, self.xoriUnits = None, None, None
        # Set up for visualization limits
        self._threshold_nel_for_large_mesh_elEdgePlot_ = 100

    def set_gids(self, gids=None):
        if gids is None:
            self.gids = np.unique(self.lfi)
        else:
            self.gids = np.array(gids, dtype=self.lfi.dtype)

    def ignore_gids(self, gids_to_ignore):
        # Capability under development
        gids_to_ignore = np.array(gids_to_ignore, dtype=self.lfi.dtype)
        self.ignored_gids = gids_to_ignore
        self.gids = np.array([gid for gid in self.gids if gid not in gids_to_ignore],
                             dtype=self.lfi.dtype)

    @classmethod
    def quad(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
            gids=None, nelx=10, nely=10, nelz=0, xstart=0, ystart=0, zstart=0, ellx=1, elly=1, ellz=1,
            nnodes=4, plane_stress=True, reduced_integration=False, modifiedFormulation=False,
            axiSymmElements=False):
        """
        Class method to create a nonConformalMesher instance with Quad elements.

        Example
        -------
        m = nonConformalMesher.quad(lfi=my_lfi, gids=my_gids, nelx=20, nely=20, 
                        xstart=0.0, ystart=0.0, ellx=1.0, elly=1.0)
        """
        if nnodes not in (4, 8):
            raise ValueError("nnodes must be 4 or 8 for Quad elements.")
        etype = 'quad4' if nnodes == 4 else 'quad8'
        return cls(dim=2, etype=etype, lfi=lfi, gids=gids, nelx=nelx, nely=nely, nelz=nelz,
                xstart=xstart, ystart=ystart, zstart=zstart, ellx=ellx, elly=elly, ellz=ellz,
                plane_stress=plane_stress, reduced_integration=reduced_integration, 
                modifiedFormulation=modifiedFormulation, axiSymmElements=axiSymmElements)

    @classmethod
    def tri(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
            gids=None, nelx=10, nely=10, nelz=0, xstart=0, ystart=0, zstart=0, ellx=1, elly=1, ellz=1,
            nnodes=3, plane_stress=True, reduced_integration=False, modifiedFormulation=False,
            axiSymmElements=False):
        """
        Class method to create a nonConformalMesher instance with Tri elements.

        Example
        -------
        m = nonConformalMesher.tri(lfi=my_lfi, gids=my_gids, nelx=20, nely=20, 
                        xstart=0.0, ystart=0.0, ellx=1.0, elly=1.0)
        """
        if nnodes not in (3, 6):
            raise ValueError("nnodes must be 3 or 6 for Tri elements.")
        etype = 'tri3' if nnodes == 3 else 'tri6'
        return cls(dim=2, etype=etype, lfi=lfi, gids=gids, nelx=nelx, nely=nely, nelz=nelz,
                xstart=xstart, ystart=ystart, zstart=zstart, ellx=ellx, elly=elly, ellz=ellz,
                plane_stress=plane_stress, reduced_integration=reduced_integration, 
                modifiedFormulation=modifiedFormulation, axiSymmElements=axiSymmElements)

    @classmethod
    def hex(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
            gids=None, nelx=10, nely=10, nelz=10, xstart=0, ystart=0, zstart=0, ellx=1, elly=1, ellz=1,
            nnodes=8, reduced_integration=False, modifiedFormulation=False):
        """
        Class method to create a nonConformalMesher instance with Hex8 elements.

        Example
        -------
        m = nonConformalMesher.hex8(lfi=my_lfi, gids=my_gids, nelx=20, nely=20, nelz=20,
                        xstart=0.0, ystart=0.0, zstart=0.0, ellx=1.0, elly=1.0, ellz=1.0)
        """
        if nnodes not in (8, 20):
            raise ValueError("nnodes must be 8 or 20 for Hex elements.")
        etype = 'hex8' if nnodes == 8 else 'hex20'
        return cls(dim=3, etype=etype, lfi=lfi, gids=gids, nelx=nelx, nely=nely, nelz=nelz,
                xstart=xstart, ystart=ystart, zstart=zstart, ellx=ellx, elly=elly, ellz=ellz,
                reduced_integration=reduced_integration, modifiedFormulation=modifiedFormulation)

    @classmethod
    def tet(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
            gids=None, nelx=10, nely=10, nelz=10, xstart=0, ystart=0, zstart=0, ellx=1, elly=1, ellz=1,
            nnodes=4, reduced_integration=False, modifiedFormulation=False):
        """
        Example
        -------
        m = nonConformalMesher.tet5(lfi=my_lfi, gids=my_gids, nelx=20, nely=20, nelz=20,
                        xstart=0.0, ystart=0.0, zstart=0.0, ellx=1.0, elly=1.0, ellz=1.0)
        """
        if nnodes not in (5, 10):
            raise ValueError("nnodes must be 5 or 10 for Tet elements.")
        etype = 'tet5' if nnodes == 5 else 'tet10'
        return cls(dim=3, etype=etype, lfi=lfi, gids=gids, nelx=nelx, nely=nely, nelz=nelz,
                xstart=xstart, ystart=ystart, zstart=zstart, ellx=ellx, elly=elly, ellz=ellz,
                reduced_integration=reduced_integration, modifiedFormulation=modifiedFormulation)

    def mesh(self, analysis_package='abaqus', elsetNamePrefixBasic='gid_'):
        """
        Core orchastrator function for non-conformal meshing in this class.

        The function branches execution as per the element types and
        the analysis package. All parameters are updated to `self` object.
        """
        self.set_analysis_package(analysis_package=analysis_package)
        self.set_element_name()
        if self.analysis_package in ('abaqus', 'moose'):
            if self.etype in ('quad4', 'quad8'):
                self._mesh_quad4_quad8_ABQ_(elsetNamePrefixBasic=elsetNamePrefixBasic)

    def set_element_name(self, elname: str = None):
        if not elname and self.analysis_package == 'abaqus':
            self._set_element_names_ABQ_()

    def _set_element_names_ABQ_(self):
        etype_map = {"tri3": 3, "tri6": 6, "quad4": 4, "quad8": 8,
                     "tet4": 4, "tet10": 10, "hex8": 8, "hex20": 20,}
        nn = etype_map[self.etype]

        if self.axiSymmElements:
            prefix = "CAX"
        elif self.etype.startswith(("tri", "quad")):
            prefix = "CPS" if self.plane_stress else "CPE"
        else:
            prefix = "C3D"

        suffix = ""
        if self.modifiedFormulation and prefix == "C3D" and nn == 10:
            suffix = "M"
        elif self.reduced_integration:
            suffix = "R"

        self.elname = f"{prefix}{nn}{suffix}"


    def set_analysis_package(self, analysis_package: str):
        if analysis_package in self._valid_analysis_packages:
            self.analysis_package = analysis_package
            self.nidStart, self.elifStart = 1, 1
        else:
            raise ValueError("Invalid analysis package name..")

    def _mesh_quad4_quad8_ABQ_(self, makeELSetsBasic=True, elsetNamePrefixBasic='gid_'):
        self.make_base_coordinates_ABQ()
        self.cross_check_nel()
        self.define_nodeNumbers_ABQ()
        self.define_nodes_ABQ()
        self.define_elementNumbers_ABQ()
        self.define_element_locations_ABQ()
        self.define_connectivity_ABQ()
        self.get_element_node_coordinates_ABQ()
        self.calculate_element_centroids_ABQ()
        self.define_elements_ABQ()
        self.define_ABQ_elements()
        if makeELSetsBasic:
            self.make_elsets_quad4_ABQ_basic(elsetNamePrefixBasic=elsetNamePrefixBasic)

    def set_pxtal_orientations(self, xori=None, xoriType='bunge', xoriUnits='degree'):
        if not isinstance(xori, np.ndarray):
            xori = np.array(xori)  # retain the dtype of the incoming data
        if xori.shape[0] != len(self.gids):
            raise ValueError("Length of xori must match number of gids.")
        self.xoriType = xoriType
        if self.xoriType not in self.valid_xoriTypes:
            raise ValueError("Invalid xoriType provided.")
        if self.xoriType in ('bunge', 'zxz', 'matthies', 'mtex', 'zyz', 'roe', 'kocks', 'canova'):
            if xori.shape[1] != 3:
                raise ValueError(f"xori must have shape ({self.gids.size}, 3).")
        if self.xoriType in ('miller', ):
            if xori.shape[1] != 6:
                raise ValueError(f"xori must have shape ({self.gids.size}, 6).")
        if self.xoriType in ('quat', ):
            if xori.shape[1] != 4:
                raise ValueError(f"xori must have shape ({self.gids.size}, 4).")
        if self.xoriType in ('rot', ):
            if xori.shape[1] != 9:
                raise ValueError(f"xori must have shape ({self.gids.size}, 9).")
        if not all(isinstance(angle, (int, float, np.integer, np.floating)) 
                   for angle in xori.ravel()):
            raise ValueError("All orientation angles must be numeric (int or float).")

        self.xori = xori
        self.xoriUnits = xoriUnits

    def map_pxtal_orientations_to_elements(self):
        pass

    def make_base_coordinates_ABQ(self, **kwargs):
        if self.etype == 'quad4':
            self._make_base_coordinates_quad4_ABQ_(**kwargs)
        elif self.etype == 'quad8':
            self._make_base_coordinates_quad8_ABQ_(**kwargs)
        elif self.etype == 'tri3':
            pass
        elif self.etype == 'tri6':
            pass

        self.xndgrid, self.yndgrid = np.meshgrid(self.ybase, self.xbase)
    
    def cross_check_nel(self):
        if self.etype in ('quad4', 'quad8'):
            expected_nelx = self.lfi.shape[1]
            expected_nely = self.lfi.shape[0]
        elif self.etype in ('tri3', 'tri6'):
            expected_nelx = 2*self.lfi.shape[1]
            expected_nely = 2*self.lfi.shape[0]
        elif self.etype in ('hex8', 'hex20'):
            expected_nelx = self.lfi.shape[2]
            expected_nely = self.lfi.shape[1]
            expected_nelz = self.lfi.shape[0]
        elif self.etype in ('tet5', 'tet10'):
            expected_nelx = 2*self.lfi.shape[2]
            expected_nely = 2*self.lfi.shape[1]
            expected_nelz = 2*self.lfi.shape[0]
        # ----------------------------------------------
        if self.nelx != expected_nelx:
            print(f"Warning: nelx ({self.nelx}) does not match lfi shape ({expected_nelx}). Updating nelx.")
            self.nelx = expected_nelx
        if self.nely != expected_nely:
            print(f"Warning: nely ({self.nely}) does not match lfi shape ({expected_nely}). Updating nely.")
            self.nely = expected_nely
        if self.etype in ('hex8', 'hex20', 'tet5', 'tet10'):
            if self.nelz != expected_nelz:
                print(f"Warning: nely ({self.nely}) does not match lfi shape ({expected_nely}). Updating nely.")
                self.nely = expected_nely
    
    def define_nodeNumbers_ABQ(self, **kwargs):
        """
        Example
        -------
        define_nodeNumbers()  # For 'quad4'
        num_valid_nodes, valid_mask_T = define_nodeNumbers()  # For 'quad8'
        """
        if self.etype == 'quad4':
            self._define_nodeNumbers_quad4_ABQ_(**kwargs)
        if self.etype == 'quad8':
            self._define_nodeNumbers_quad8_ABQ_(**kwargs)
    
    def define_nodes_ABQ(self, **kwargs):
        if self.etype == 'quad4':
            self._define_nodes_quad4_ABQ_(**kwargs)
        elif self.etype == 'quad8':
            self._define_nodes_quad8_ABQ_(**kwargs)

    def define_elementNumbers_ABQ(self, **kwargs):
        if self.etype == 'quad4':
            self._define_elementNumbers_quad4_ABQ_(**kwargs)
        elif self.etype == 'quad8':
            self._define_elementNumbers_quad8_ABQ_(**kwargs)

    def define_element_locations_ABQ(self, **kwargs):
        if self.etype == 'quad4':
            self._define_element_locations_quad4_ABQ_(**kwargs)
        elif self.etype == 'quad8':
            self._define_element_locations_quad8_ABQ_(**kwargs)

    def define_connectivity_ABQ(self, **kwargs):
        """
        Example
        -------
        el_ids, connectivity = define_connectivity(elLoc, nnum)
        """
        if self.etype == 'quad4':
            self._define_connectivity_quad4_ABQ_(**kwargs)
        elif self.etype == 'quad8':
            self._define_connectivity_quad8_ABQ_(**kwargs)

    def get_element_node_coordinates_ABQ(self, **kwargs):
        '''Extract Element Node Coordinates'''
        if self.etype == 'quad4':
            self._get_element_node_coordinates_quad4_ABQ_(**kwargs)
        elif self.etype == 'quad8':
            self._get_element_node_coordinates_quad8_ABQ_(**kwargs)

    def calculate_element_centroids_ABQ(self, **kwargs):
        '''Centroid Calculation'''
        if self.etype == 'quad4':
            self._calculate_element_centroids_quad4_ABQ_(**kwargs)
        elif self.etype == 'quad8':
            self._calculate_element_centroids_quad8_ABQ_(**kwargs)

    def define_elements_ABQ(self, **kwargs):
        '''Create ELEMENTS dict'''
        if self.etype in ('quad4', 'quad8',):
            self.ELEMENTS = dict(zip(map(int, self.el_ids), self.connectivity.tolist()))

    def define_ABQ_elements(self):
        '''Create ELEMENTS dict'''
        if self.etype in ('quad4', 'quad8',):
            self.ABQ_ELEMENTS = np.hstack((np.expand_dims([i+1 
                                    for i in self.ELEMENTS.keys()], axis=1),
                                np.vstack(list(self.ELEMENTS.values()))))

    def make_elsets_quad4_ABQ_basic(self, elsetNamePrefixBasic='gid_'):
        elsets = {f'{elsetNamePrefixBasic}{gid}': [] for gid in self.gids}
        for elloc in self.elLoc:
            gid = self.lfi[elloc[1], elloc[2]]
            if not gid or gid not in self.gids:
                continue
            elsets[f'{elsetNamePrefixBasic}{gid}'].append(int(elloc[0]))
        self.elsets = dict(basic = elsets)

    def _make_base_coordinates_quad4_ABQ_(self, **kwargs):
        self.xbase = np.linspace(start=self.xstart, stop=self.xstart+self.ellx*self.nelx,
                    num=self.nelx+1).astype(self.floatDType)
        self.ybase = np.linspace(start=self.ystart, stop=self.ystart+self.elly*self.nely,
                    num=self.nely+1).astype(self.floatDType)

    def _make_base_coordinates_quad8_ABQ_(self, **kwargs):
        self.xbase = np.linspace(start=self.xstart, stop=self.xstart+self.ellx*self.nelx,
                    num=2*self.nelx+1).astype(self.floatDType)
        self.ybase = np.linspace(start=self.ystart, stop=self.ystart+self.elly*self.nely,
                    num=2*self.nely+1).astype(self.floatDType)

    def _define_nodeNumbers_quad4_ABQ_(self, **kwargs):
        self.nnum = np.reshape(np.arange(0, self.xndgrid.size), self.xndgrid.shape).T

    def _define_nodeNumbers_quad8_ABQ_(self, **kwargs):
        # Create grid indices (0, 1, 2...) to identify Odd vs Even rows/cols
        # We use the same shape as xndgrid
        i_grid, j_grid = np.meshgrid(np.arange(self.ybase.size),
                                        np.arange(self.xbase.size))
        # MASK: Identify valid nodes (Corners and Mid-sides)
        # A node is valid if it is NOT a center.
        # Center nodes occur when BOTH row (i) and col (j) indices are ODD.
        is_center = (i_grid % 2 != 0) & (j_grid % 2 != 0)
        is_valid = ~is_center
        # Create the ID map (initialized to -1 for safety, though never accessed)
        # We strictly transpose (.T) to match your original column-major ordering preference
        self.nnum = np.full(self.xndgrid.shape, -1, dtype=int).T
        valid_mask_T = is_valid.T
        # Assign sequential IDs ONLY to valid spots
        # This packs the IDs contiguously (0, 1, 2, 3...) skipping the centers
        num_valid_nodes = np.sum(valid_mask_T)
        self.nnum[valid_mask_T] = np.arange(num_valid_nodes)
        self.temp_container['num_valid_nodes'] = num_valid_nodes
        self.temp_container['valid_mask_T'] = valid_mask_T

    def _define_nodes_quad4_ABQ_(self, **kwargs):
        self.NODES = np.vstack((self.nnum.ravel(order='F'),
                        self.yndgrid.T.ravel(order='F'),
                        self.xndgrid.T.ravel(order='F'))).T.astype(self.floatDType)

    def _define_nodes_quad8_ABQ_(self, **kwargs):
        x_coords = self.xndgrid.T[self.temp_container['valid_mask_T']]
        y_coords = self.yndgrid.T[self.temp_container['valid_mask_T']]
        node_ids = np.arange(self.temp_container['num_valid_nodes'])
        self.temp_container = {}  # Lets reset this
        self.NODES = np.column_stack((node_ids, y_coords,
                                      x_coords)).astype(self.floatDType)

    def _define_elementNumbers_quad4_ABQ_(self, **kwargs):
        self.enum = np.reshape(np.arange(0, self.nelx*self.nely), (self.nelx, self.nely)).T

    def _define_elementNumbers_quad8_ABQ_(self, **kwargs):
        self.enum = np.reshape(np.arange(0, self.nelx*self.nely), (self.nelx, self.nely)).T

    def _define_element_locations_quad4_ABQ_(self, **kwargs):
        elLoc = np.meshgrid(np.arange(0, self.enum.shape[1]),
                    np.arange(0, self.enum.shape[0]))
        self.elLoc = np.vstack((self.enum.ravel(order='F'),
                    elLoc[1].ravel(), elLoc[0].ravel())).T

    def _define_element_locations_quad8_ABQ_(self, **kwargs):
        elLoc = np.meshgrid(np.arange(0, self.enum.shape[1]),
                    np.arange(0, self.enum.shape[0]))
        self.elLoc = np.vstack((self.enum.ravel(order='F'),
                    elLoc[1].ravel(), elLoc[0].ravel())).T

    def _define_connectivity_quad4_ABQ_(self):
        '''Extract Row and Col indices as vectors
        Assuming elLoc is shape (N_elements, 3) -> [ID, row_idx, col_idx]'''
        self.el_ids = self.elLoc[:, 0].astype(int)
        rows = self.elLoc[:, 1].astype(int)
        cols = self.elLoc[:, 2].astype(int)
        """We grab all 4 corners for ALL elements at once using grid offsets.
        ABAQUS Quad4 Order: Bottom-Left -> Bottom-Right -> Top-Right -> Top-Left (Counter-Clockwise)
        Note: Based on your previous code's logic of "flipping the last two", 
        the resulting order was: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
        # Quad4 Connectivity (CCW -- as per Abaqus convention)
        ''' REF: https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/
                    gsa/default.htm?startat=ch04s01.html '''
        n_tl = self.nnum[rows, cols]  # TL
        n_tr = self.nnum[rows, cols+1]  # TR
        n_br = self.nnum[rows+1, cols+1]  # Bottom-Right (The flipped one)
        n_bl = self.nnum[rows+1, cols]  # Bottom-Left
        self.connectivity = np.column_stack((n_tl, n_tr, n_br, n_bl))

    def _define_connectivity_quad8_ABQ_(self):
        self.el_ids = self.elLoc[:, 0].astype(int)
        rows = self.elLoc[:, 1].astype(int)
        cols = self.elLoc[:, 2].astype(int)
        """We use the nnum map we created earlier.
        Even though nnum has "holes" (indices with -1), the sampling logic
        for Quad8 ONLY accesses even rows/cols and mid-sides.
        It NEVER touches the (Odd, Odd) center spots, so we shoudl be safe."""
        rows, cols = rows*2, cols*2
        # Quad8 Connectivity (CCW -- as per Abaqus convention)
        ''' REF: https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/
                    gsa/default.htm?startat=ch04s01.html '''
        n1 = self.nnum[rows, cols] # BL
        n2 = self.nnum[rows, cols+2]  # BR
        n3 = self.nnum[rows+2, cols+2]  # TR
        n4 = self.nnum[rows+2, cols]  # TL
        n5 = self.nnum[rows, cols+1]  # Bottom-Mid
        n6 = self.nnum[rows+1, cols+2]  # Right-Mid
        n7 = self.nnum[rows+2, cols+1]  # Top-Mid
        n8 = self.nnum[rows+1, cols]  # Left-Mid
        self.connectivity = np.column_stack((n1, n2, n3, n4, n5, n6, n7, n8))

    def _get_element_node_coordinates_quad4_ABQ_(self, **kwargs):
        self.el_node_coords = self.NODES[self.connectivity, 1:]

    def _get_element_node_coordinates_quad8_ABQ_(self, **kwargs):
        self.el_node_coords = self.NODES[self.connectivity, 1:]

    def _calculate_element_centroids_quad4_ABQ_(self, **kwargs):
        self.elCentroids = self.el_node_coords.mean(axis=1)

    def _calculate_element_centroids_quad8_ABQ_(self, **kwargs):
        self.elCentroids = self.el_node_coords.mean(axis=1)

    def set_threshold_nel_for_large_mesh_elEdgePlot_(self, value: int):
        self._threshold_nel_for_large_mesh_elEdgePlot_ = value

    def see_mesh(self, figsize=(10, 5), dpi=150,
                 imageDataType='lfi', imageData=None,
                 imageDataTitle='LFI', featureName='grains', elsetType='basic',
                 overlapOnIMAGE=True, cmap='nipy_spectral', 
                 showColorBar=True, IMAGEalpha=0.3,
                 nodes=True, nodeMarker='s', nodeMarkerFaceColor='grey',
                 nodeMarkerEdgeColor='black', nodeMarkerSize=8,
                 nodeNumbers=True, nodeNumberTextColour='blue',
                 nodeNumberTextLocOffsetFactor=20, nodeNumberTextFontSize=16,
                 elementNumbers=True, elementNumberTextColour='red',
                 elementNumberTextLocOffsetFactor=20, elementNumberTextFontSize=16,
                 elementNumberTextFontWeight='bold',
                 elementEdges=True, elementEdgeLineColor='grey',
                 elementEdgeLineStyle='--',
                 elementEdgeLineWidth=0.5,
                 elementCentroids=True, elementCentroidMarker='o',
                 elementCentroidMarkerSize=2,
                 returnFigAx=False, includeLabels=False, xlabelText='', ylabelText='',
                 xlabelTextFontSize=14, ylabelTextFontSize=14, 
                 xlabelTextFontWeight='normal', ylabelTextFontWeight='normal', 
                 includeTitle=False, titleText='', 
                 titleTextFontSize=14, titleTextFontWeight='normal'):

        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # ===============================
        # Process imageDataType
        if overlapOnIMAGE and type(imageDataType) == 'str':
            if imageDataType not in self.valid_image_data:
                raise ValueError("Invalid imageData string input.")
            imageDataType = getattr(self, imageDataType)
        # -------------------------------
        if overlapOnIMAGE and imageDataType == 'lfi':
            imageData = self.lfi
        elif overlapOnIMAGE and imageDataType == 'phmap':
            imageData = self.phmap
        elif overlapOnIMAGE and imageDataType == 'custom':
            if imageData is None:
                raise ValueError("For 'custom' imageDataType, please provide 'imageData' array.")
            else:
                imageData = imageData
        elif overlapOnIMAGE and imageDataType not in ('lfi', 'phmap', 'custom'):
            raise ValueError("Invalid imageDataType provided.")
        # -------------------------------
        if overlapOnIMAGE:
            im = ax.imshow(imageData, cmap=cmap, 
                           extent=(self.xstart, self.xstart+self.ellx*self.nelx,
                                   self.ystart, self.ystart+self.elly*self.nely),
                           origin='lower', alpha=IMAGEalpha)
        # -------------------------------
        if overlapOnIMAGE and showColorBar:
                cbar = fig.colorbar(im, ax=ax)
                if featureName == 'grains':
                    cbar.set_label(f'{imageDataTitle} values (Grain IDs)',
                                   rotation=270, labelpad=15)
                else:
                    cbar.set_label(f'{imageDataTitle} Values',
                                rotation=270, labelpad=15)
        # ===============================
        if nodes and nodeNumbers and self.NODES.shape[0] <= 100:
            for i, coords in enumerate(self.NODES):
                if nodes:
                    ax.plot(coords[1], coords[2], nodeMarker, 
                            markersize=nodeMarkerSize,
                            markerfacecolor=nodeMarkerFaceColor,
                            markeredgecolor=nodeMarkerEdgeColor)
                if nodeNumbers:
                    ax.text(coords[1]+self.ellx/nodeNumberTextLocOffsetFactor,
                            coords[2]+self.elly/nodeNumberTextLocOffsetFactor,
                            str(i), fontdict={'size': nodeNumberTextFontSize},
                            color=nodeNumberTextColour)
        if nodes and nodeNumbers and self.NODES.shape[0] > 100:
            print("Node numbers and nodes plotting skipped for large meshes (>100 nodes).")
        # ===============================
        if elementEdges and self.el_node_coords.shape[0] <= self._threshold_nel_for_large_mesh_elEdgePlot_:
            for el_node_coord in self.el_node_coords:
                el_node_coord = np.vstack((el_node_coord, el_node_coord[0]))  # Close the loop
                ax.plot(np.hstack((el_node_coord[0:4, 0], [el_node_coord[0, 0]])),
                        np.hstack((el_node_coord[0:4, 1], [el_node_coord[0, 1]])),
                        elementEdgeLineStyle, 
                        color=elementEdgeLineColor,
                        linewidth=elementEdgeLineWidth)
        if elementEdges and self.el_node_coords.shape[0] > 100:
            print("Element edges plotting skipped for large meshes (>100 elements).")
        # ===============================
        if elementNumbers and self.NODES.shape[0] <= 50:
            for i, elcent in enumerate(self.elCentroids):
                ax.text(elcent[0]+self.ellx/elementNumberTextLocOffsetFactor,
                        elcent[1]+self.elly/elementNumberTextLocOffsetFactor,
                        str(i), color=elementNumberTextColour, 
                        fontdict={'size': elementNumberTextFontSize, 
                                  'weight': elementNumberTextFontWeight})
        if elementNumbers and self.NODES.shape[0] > 50:
            print("Element numbers plotting skipped for large meshes (>50 elements).")
        # ===============================
        if elementCentroids and self.NODES.shape[0] <= 1000:
            cmap = plt.cm.get_cmap(cmap)
            for i, (elids_key, elids) in enumerate(self.elsets[elsetType].items()):
                clr = cmap(self.gids[i]/len(self.gids))
                for elid in elids:
                    ax.plot(self.elCentroids[elid, 0],
                            self.elCentroids[elid, 1], 
                            elementCentroidMarker,
                            markerfacecolor=clr,
                            markersize=elementCentroidMarkerSize,
                            markeredgecolor=clr)
        if elementCentroids and self.NODES.shape[0] > 1000:
            print("Element centroids plotting skipped for large meshes (>1000 elements).")
        # ===============================
        if includeLabels:
            xlabelText = xlabelText if xlabelText else "X Coordinate"
            ylabelText = ylabelText if ylabelText else "Y Coordinate"
            ax.set_xlabel(xlabelText, fontdict={'size': xlabelTextFontSize, 'weight': xlabelTextFontWeight})
            ax.set_ylabel(ylabelText, fontdict={'size': ylabelTextFontSize, 'weight': ylabelTextFontWeight})
        if includeTitle:
            titleText = titleText if titleText else f"FE Mesh. ({self.etype}. {imageDataTitle} overlay"
            ax.set_title(titleText, fontdict={'size': titleTextFontSize, 'weight': titleTextFontWeight})
        if returnFigAx:
            return fig, ax
        
    def find_element_ids_to_remove(self, lfi_locs_to_remove):
        el_ids_remove = np.zeros(lfi_locs_to_remove.shape[0], dtype=np.int32)
        for i, remLoc in enumerate(lfi_locs_to_remove):
            remLoc = np.expand_dims(remLoc, axis=0)
            el_ids_remove[i] = self.el_ids[np.where(np.all(self.elLoc[:, 1:] == remLoc, axis=1))[0][0]]
        return el_ids_remove

    def find_boundary_features(self, boundary_offsets):
        fxm = np.unique(self.lfi[:, :boundary_offsets[0]])  # Features at x-
        fxp = np.unique(self.lfi[:, -boundary_offsets[1]:])  # Features at x+
        fym = np.unique(self.lfi[:boundary_offsets[2], :])  # Features at y-
        fyp = np.unique(self.lfi[-boundary_offsets[3]:, :])  # Features at y+
        bf = np.hstack((fxm, fxp, fym, fyp)) # Boundary features
        bf = {'fxm': fxm, 'fxp': fxp, 'fym': fym, 'fyp': fyp, 'bf': bf}
        return bf

    def find_pruning_locations(self, pharr=None, dim=2, 
                            method='cellSize', measure='npixels', unitSize=1.0, 
                            threshold=[0, 0.5], 
                            phaseValue = 1,
                            thresholdingRule='quantile', 
                            exclude_boundary_features=True,
                            boundary_offsets=[1, 1, 1, 1]
                            ):
        if method == 'cellSize':
            valueDistribution = np.bincount(self.lfi.ravel())
            if measure in ('npixels', 'npix', 'nvoxels', 'nvox'):
                pass
            elif measure in ('area', 'volume'):
                valueDistribution = valueDistribution*unitSize
            else:
                pass
        # ----------------------------------
        if method in ('cellSize', 'aspectRatio'):
            if thresholdingRule in ('q', 'quantile'):
                if min(threshold) < 0 or min(threshold) > 1:
                    raise ValueError(f"Minimum quantile cannot be < 0 or > 1.")
                if max(threshold) < 0 or max(threshold) > 1:
                    raise ValueError(f"Maximum quantile cannot be < 0 or > 1.")
                thresholdingRule, _fn_ = 'q', np.quantile

            if thresholdingRule in ('p', 'percentile', 'percentage'):
                if min(threshold) < 0 or min(threshold) > 100:
                    raise ValueError(f"Minimum percentile cannot be < 0 or > 100.")
                if max(threshold) < 0 or max(threshold) > 100:
                    raise ValueError(f"Maximum percentile cannot be < 0 or > 100.")
                thresholdingRule, _fn_ = 'p', np.percentile

            if thresholdingRule in ('v', 'value'):
                if min(threshold) < 0:
                    raise ValueError(f"Minimum percentile cannot be < 0 or > 100.")
                thresholdingRule = 'v'
            
            if thresholdingRule in ('q', 'p'):
                pruning_fids_low = np.where(valueDistribution >= _fn_(valueDistribution, min(threshold)))[0]
                pruning_fids_high = np.where(valueDistribution <= _fn_(valueDistribution, max(threshold)))[0]

            if thresholdingRule == 'v':
                pruning_fids_low = np.where(valueDistribution >= min(threshold))[0]
                pruning_fids_high = np.where(valueDistribution <= max(threshold))[0]
            fidsToPrune = np.intersect1d(pruning_fids_low, pruning_fids_high, assume_unique=False)
        else:
            raise ValueError(f"Invalid method={method}")
        # ----------------------------------
        if exclude_boundary_features:
            bf = self.find_boundary_features(boundary_offsets)
            fidsToPrune = np.array([fid for fid in fidsToPrune if fid not in bf['bf']])
        # ----------------------------------
        if pharr != None:
            # Use pharr: Phase array alongside lfi
            if type(phaseValue) != int:
                raise ValueError(f"phaseValue={phaseValue} must be of type int.")
            # To be developed
            pass
        # ----------------------------------
        lfi_locs_to_remove = []
        for pruneLoc in fidsToPrune:
            lfi_locs_to_remove.append(np.column_stack(np.where(self.lfi == pruneLoc)))
        lfi_locs_to_remove = np.vstack(lfi_locs_to_remove)
        # ----------------------------------
        return fidsToPrune, lfi_locs_to_remove

    def prune(self, lfi_locs_to_remove):
        print('Determining elements to prune and retain.')
        el_ids_remove = self.find_element_ids_to_remove(lfi_locs_to_remove)
        EltoRemove = {int(i): self.ELEMENTS[i] for i in el_ids_remove}
        el_ids_retain = np.array([eid for eid in self.ELEMENTS.keys() if eid not in el_ids_remove],
                                dtype=np.int32)
        # ---------------------
        print('Remapping nodes and elements.')
        nodeEl = {int(i): [] for i in self.NODES[:, 0]}
        for el, nodes in self.ELEMENTS.items():
            for node in nodes:
                nodeEl[node].append(el)
        # ---------------------
        print('Determining nodes to prune and retain')
        EltoRemove = list(EltoRemove.values())
        if len(EltoRemove) == 0:
            print('There are no elements to prune.')
            return None, None, None

        nodeLineUp = np.unique(np.hstack( EltoRemove ))
        # ---------------------
        elements = {int(elid): self.ELEMENTS[elid] for elid in el_ids_retain}
        elements_abq = self.ABQ_ELEMENTS[el_ids_retain]
        element_node_coordinates = self.el_node_coords[el_ids_retain]

        elCentroids = element_node_coordinates.mean(axis=1)
        # ---------------------
        node_ids_remove = []
        nodes_of_retained_elements = np.unique(np.hstack(list(elements.values())))
        for node in nodeLineUp:
            if node not in nodes_of_retained_elements:
                node_ids_remove.append(node)

        node_ids_retain = [int(i) for i in list(set(self.NODES[:, 0]) - set(node_ids_remove))]

        nodes = self.NODES[node_ids_retain]
        # ---------------------
        print('Wrapping up.')
        IDs = {'node_ids_remove': np.array(node_ids_remove, dtype=np.int32),
            'node_ids_retain': np.array(node_ids_retain, dtype=np.int32),
            'el_ids_remove': el_ids_remove,
            'el_ids_retain': el_ids_retain,
            }
        elements = {'elements': elements,
                    'elements_abq': elements_abq,
                    'element_node_coordinates': element_node_coordinates,
                    'element_centroids': elCentroids
                    }
        return nodes, elements, IDs