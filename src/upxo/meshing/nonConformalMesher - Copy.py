import numpy as np

class nonConformalMesher():
    """
    A class to generate non-conformal meshes for 2D and 3D geometries.
    Supports element types: Quad4, Quad8, Tri3, Tri6, Hex8, Hex20, Tet5.
    2D Element Types: Quad4, Quad8, Tri3, Tri6
    3D Element Types: Hex8, Hex20, Tet5

    Import
    ------
    from upxo.meshing.nonConformalMesher import nonConformalMesher as ncm
    """
    valid_analysis_packages = ('abaqus', 'moose', 'damask')
    valid_image_data = ('lfi', 'phmap', 'custom')
    __slots__ = ('nelx', 'nely', 'nelz',
                 'xstart', 'ystart',  'zstart', 
                 'ellx', 'elly', 'ellz', 
                 'etype', 'lfi', 'gids', 'phmap',
                 'xincr', 'yincr', 'zincr',
                 'xbase', 'ybase', 'xbase',
                 'xndgrid', 'yndgrid', 'zndgrid',
                 'nnum', 'NODES', 
                 'enum', 'elLoc', 'ELEMENTS', 'ABQ_ELEMENTS',
                 'elsets', 
                 'el_ids', 'analysis_package', 
                 'connectivity', 'el_node_coords', 
                 'elCentroids', 'floatDType',
                 'temp_container'
                 )

    def __init__(self, etype='quad4',
                 lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
                 gids=None,
                 nelx=10, nely=10, nelz=10,
                 xstart=0, ystart=0, zstart=0,
                 ellx=1, elly=1, ellz=1
                 ):
        self.etype = etype
        self.lfi = lfi
        self.gids = gids
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.xstart = xstart
        self.ystart = ystart
        self.zstart = zstart
        self.ellx = ellx
        self.elly = elly
        self.ellz = ellz
        self.floatDType = np.float32
        self.temp_container = {}

    def set_analysis_package(self, analysis_package: str):
        if analysis_package in self.valid_analysis_packages:
            self.analysis_package = analysis_package
        else:
            raise ValueError("Invalid analysis pacakge name..")

    @classmethod
    def quad4(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
              gids=None, nelx=10, nely=10,
              xstart=0, ystart=0, ellx=1, elly=1):
        return cls(etype='quad4', lfi=lfi, gids=gids,
                   nelx=nelx, nely=nely, xstart=xstart, ystart=ystart, 
                   ellx=ellx, elly=elly)
    @classmethod
    def quad8(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
              gids=None, nelx=10, nely=10,
              xstart=0, ystart=0, ellx=1, elly=1):
        return cls(etype='quad4', lfi=lfi, gids=gids,
                   nelx=nelx, nely=nely, xstart=xstart, ystart=ystart, 
                   ellx=ellx, elly=elly)
    @classmethod
    def tri3(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
             gids=None, nelx=10, nely=10,
             xstart=0, ystart=0, ellx=1, elly=1):
        return cls(etype='quad4', lfi=lfi, gids=gids,
                   nelx=nelx, nely=nely, xstart=xstart, ystart=ystart, 
                   ellx=ellx, elly=elly)
    @classmethod
    def tri6(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
             gids=None, nelx=10, nely=10,
             xstart=0, ystart=0, ellx=1, elly=1):
        return cls(etype='quad4', lfi=lfi, gids=gids,
                   nelx=nelx, nely=nely, xstart=xstart, ystart=ystart, 
                   ellx=ellx, elly=elly)
    @classmethod
    def hex8(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
             gids=None, nelx=10, nely=10, nelz=10,
             xstart=0, ystart=0, zstart=0, ellx=1, elly=1, ellz=1):
        return cls(etype='quad4', lfi=lfi, gids=gids,
                   nelx=nelx, nely=nely, 
                   xstart=xstart, ystart=ystart, zstart=zstart,
                   ellx=ellx, elly=elly, ellz=ellz)
    @classmethod
    def hex20(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
              gids=None, nelx=10, nely=10, nelz=10,
              xstart=0, ystart=0, zstart=0, ellx=1, elly=1, ellz=1):
        return cls(etype='quad4', lfi=lfi, gids=gids,
                   nelx=nelx, nely=nely, 
                   xstart=xstart, ystart=ystart, zstart=zstart,
                   ellx=ellx, elly=elly, ellz=ellz)
    @classmethod
    def tet5(cls, lfi=np.random.randint(0, 100, size=(10, 10), dtype=np.int32),
             gids=None, nelx=10, nely=10, nelz=10,
             xstart=0, ystart=0, zstart=0, ellx=1, elly=1, ellz=1):
        return cls(etype='quad4', lfi=lfi, gids=gids,
                   nelx=nelx, nely=nely, 
                   xstart=xstart, ystart=ystart, zstart=zstart,
                   ellx=ellx, elly=elly, ellz=ellz)
    
    def mesh(self):
        """
        Core orchastrator function for non-conformal meshing in this class.

        The function branches execution as per the element types and
        the analysis package. All parameters are updated to `self` object.
        """
        if self.etype in ('quad4', 'quad8') and self.analysis_package == 'abaqus':
            self._mesh_quad4_quad8_ABQ_()

    def _mesh_quad4_quad8_ABQ_(self):
        self.make_base_coordinates_ABQ()
        self.define_nodeNumbers_ABQ()
        self.define_nodes_ABQ()
        self.define_elementNumbers_ABQ()
        self.define_element_locations_ABQ()
        self.define_connectivity_ABQ()
        self.get_element_node_coordinates_ABQ()
        self.calculate_element_centroids_ABQ()
        self.define_elements_ABQ()
        self.make_elsets_quad4_ABQ_basic()

    def make_base_coordinates_ABQ(self):
        if self.etype == 'quad4':
            self.xbase = np.linspace(start=self.xstart, 
                        stop=self.xstart+self.ellx*self.nelx,
                        num=self.nelx+1).astype(self.floatDType)
            self.ybase = np.linspace(start=self.ystart,
                        stop=self.ystart+self.elly*self.nely,
                        num=self.nely+1).astype(self.floatDType)
        elif self.etype == 'quad8':
            self.xbase = np.linspace(start=self.xstart,
                        stop=self.xstart+self.ellx*self.nelx,
                        num=2*self.nelx+1).astype(self.floatDType)
            self.ybase = np.linspace(start=self.ystart,
                        stop=self.ystart+self.elly*self.nely,
                        num=2*self.nely+1).astype(self.floatDType)
        elif self.etype == 'tri3':
            pass
        elif self.etype == 'tri6':
            pass
        # ----------------------
        self.xndgrid, self.yndgrid = np.meshgrid(self.ybase, self.xbase)
    
    def define_nodeNumbers_ABQ(self, **kwargs):
        """
        Example
        -------
        define_nodeNumbers()  # For 'quad4'
        num_valid_nodes, valid_mask_T = define_nodeNumbers()  # For 'quad8'
        """
        if self.etype == 'quad4':
            self.nnum = np.reshape(np.arange(0, self.xndgrid.size),
                            self.xndgrid.size).T
        if self.etype == 'quad8':
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
    
    def define_nodes_ABQ(self, **kwargs):
        if self.etype == 'quad4':
            self.NODES = np.vstack((self.nnum.ravel(order='F'),
                            self.yndgrid.T.ravel(order='F'),
                            self.xndgrid.T.ravel(order='F'))).T.astype(self.floatDType)
        elif self.etype == 'quad4':
            x_coords = self.xndgrid.T[self.temp_container['valid_mask_T']]
            y_coords = self.yndgrid.T[self.temp_container['valid_mask_T']]
            node_ids = np.arange(self.temp_container['num_valid_nodes'])
            self.temp_container = {}  # Lets reset this
            self.NODES = np.column_stack((node_ids, y_coords, x_coords)).astype(self.floatDType)

    def define_elementNumbers_ABQ(self):
        if self.etype in ('quad4', 'quad8',):
            self.enum = np.reshape(np.arange(0, self.nelx*self.nely),
                            (self.nelx, self.nely)).T

    def define_element_locations_ABQ(self):
        if self.etype in ('quad4', 'quad8',):
            elLoc = np.meshgrid(np.arange(0, self.enum.shape[1]),
                        np.arange(0, self.enum.shape[0]))
            self.elLoc = np.vstack((self.enum.ravel(order='F'),
                        elLoc[1].ravel(), elLoc[0].ravel())).T

    def define_connectivity_ABQ(self):
        """
        Example
        -------
        el_ids, connectivity = define_connectivity(elLoc, nnum)
        """
        if self.etype == 'quad4':
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
            n_tl = self.nnum[rows, cols]  # Top-Left
            n_tr = self.nnum[rows, cols+1]  # Top-Right
            n_br = self.nnum[rows+1, cols+1]  # Bottom-Right (The flipped one)
            n_bl = self.nnum[rows+1, cols]  # Bottom-Left
            self.connectivity = np.column_stack((n_tl, n_tr, n_br, n_bl))
        elif self.etype == 'quad8':
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

    def get_element_node_coordinates_ABQ(self):
        '''Extract Element Node Coordinates'''
        if self.etype in ('quad4', 'quad8',):
            self.el_node_coords = self.NODES[self.connectivity, 1:]

    def calculate_element_centroids_ABQ(self):
        '''Centroid Calculation'''
        if self.etype in ('quad4', 'quad8',):
            self.elCentroids = self.el_node_coords.mean(axis=1)

    def define_elements_ABQ(self):
        '''Create ELEMENTS dict'''
        if self.etype in ('quad4', 'quad8',):
            self.ELEMENTS = dict(zip(map(int, self.el_ids),
                                    self.connectivity.tolist()))
            self.ABQ_ELEMENTS = np.hstack((np.expand_dims([i+1 
                                    for i in self.ELEMENTS.keys()], axis=1),
                                np.vstack(list(self.ELEMENTS.values()))))

    def make_elsets_quad4_ABQ_basic(self):
        elsets = {f'ELSET_GID_{gid}': [] for gid in self.gids}
        for elloc in self.elLoc:
            gid = self.lfi[elloc[1], elloc[2]]
            if not gid:
                continue
            elsets[f'ELSET_GID_{gid}'].append(int(elloc[0]))
        self.elsets = dict(basic = elsets)

    def see_mesh(self, figsize=(10, 5), dpi=150,
                 imageDataType='lfi', imageData=None,
                 imageDataTitle='LFI',
                 overlapOnIMAGE=False, cmap='nipy_spectral', 
                 showColorBar=True, IMAGEalpha=0.3,
                 nodes=True, nodeMarker='s', nodeMarkerFaceColor='grey',
                 nodeMarkerEdgeColor='black', nodeMarkerSize=8,
                 nodeNumbers=False, nodeNumberTextColour='blue',
                 nodeNumberTextLocOffsetFactor=20, nodeNumberTextFontSize=16,
                 elementEdges=True, elementEdgeLineColor='grey',
                 elementEdgeLineStyle='--',
                 elementEdgeLineWidth=0.5,
                 elementCentroids=True, elementCentroidMarker='o',
                 elementCentroidMarkerSize=2,
                 returnFigAx=False, elsetType='basic'
                 ):

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
        else:
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
                cbar.set_label(f'{imageDataTitle} Values',
                               rotation=270, labelpad=15)
        # ===============================
        if nodes and nodeNumbers:
            for i, coords in enumerate(self.NODES):
                if nodes:
                    ax.plot(coords[1], coords[2], nodeMarker, 
                            markersize=nodeMarkerSize,
                            markerfacecolor=nodeMarkerFaceColor,
                            markeredgecolor=nodeMarkerEdgeColor)
                if nodeNumbers:
                    _ = nodeNumberTextLocOffsetFactor
                    ax.text(coords[1]+self.ellx/_, coords[2]+self.elly/_,
                            str(i), fontdict={'size': nodeNumberTextFontSize},
                            color=nodeNumberTextColour)
        if elementEdges:
            for el_node_coord in self.el_node_coords:
                el_node_coord = np.vstack((el_node_coord, el_node_coord[0]))  # Close the loop
                ax.plot(np.hstack((el_node_coord[0:4, 0], [el_node_coord[0, 0]])),
                        np.hstack((el_node_coord[0:4, 1], [el_node_coord[0, 1]])),
                        elementEdgeLineStyle, 
                        color=elementEdgeLineColor,
                        linewidth=elementEdgeLineWidth)

        for i, elcent in enumerate(self.elCentroids):
            ax.text(elcent[0], elcent[1], str(i), color='red', 
                    fontdict={'size': 16, 'weight': 'bold'})

        if elementCentroids:
            cmap = plt.cm.get_cmap(cmap)
            for i, elids in enumerate(self.elsets[elsetType].items()):
                clr = cmap(self.gids[i]/self.gids.size)
                for elid in elids:
                    ax.plot(self.elCentroids[elid, 0],
                            self.elCentroids[elid, 1], 
                            elementCentroidMarker,
                            markerfacecolor=clr,
                            markersize=elementCentroidMarkerSize,
                            markeredgecolor=clr)
        if returnFigAx:
            return fig, ax