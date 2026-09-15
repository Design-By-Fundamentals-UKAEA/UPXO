"""
2D grain-boundary-conformal meshing for UPXO tessellations.

``confMesh2d`` is the legacy pygmsh path and is **deprecated**. New work
must use ``confMesh2dGMSH`` (raw gmsh API) or ``gsmesh2d.mesh_gs``.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from upxo.meshing import elemOps
from upxo.viz import meshviz
import pyvista as pv

_CONFMESH2D_REMOVED_MSG = (
    "upxo.meshing.conformal_mesher2d.confMesh2d (pygmsh) is deprecated "
    "and will be removed in a future UPXO release. Use "
    "confMesh2dGMSH.femesh_gmsh or gsmesh2d.mesh_gs instead."
)


class confMesh2d():
    """
    Deprecated pygmsh orchestrator for 2D conformal grain-structure meshing.

    .. deprecated::
        Use :class:`confMesh2dGMSH` or :func:`upxo.meshing.gsmesh2d.mesh_gs`.
        Instantiating this class emits :class:`DeprecationWarning`.

    Builds conformal FE meshes on geometric polycrystals (typically
    :class:`~upxo.pxtal.polyxtal.vtpolyxtal2d` / VTGS), with triangle or
    quad elements via pygmsh-oriented paths.

    Usage (legacy)
    --------------
    >>> from upxo.meshing.conformal_mesher2d import confMesh2d as cm2d
    >>> m = confMesh2d.from_geometric_pxtal(pxtal=..., xbound=..., ybound=...)
    >>> m.femesh_pygmsh(elementShape='tri', elementOrder=1, ...)

    Attributes
    ----------
    gtess
        Geometric tessellation / polycrystal source.
    nodes, elConn, elsets, elsets_eltype
        Mesh connectivity and element-set bookkeeping after meshing.
    elementShape, elementOrder, meshingAlgorithmID
        Element and algorithm settings.
    """
    ValidElTypesOptions =('triangle', 'quad')
    __slots__ = ('gtess', 'fids', 'pxtal_mesh', 'grid', 'mesherTool', 'elementShape',
                 'elementOrder',  'meshingAlgorithmID', '_intermediate_mesh_filename_',
                 'filtered_cells', 'filtered_mesh', 'nodes', 'elConn',
                 'meshCellFeatTypes', 'lineFeatLocation', 'GBlines',
                 'availableFeatures', 'availableElTypes', 'availableElTypeID',
                 'elsets_eltype', 'elsets', 'elID_ranges')
    def __init__(self, gtess=None):
        """Initialise the instance."""
        warnings.warn(_CONFMESH2D_REMOVED_MSG, DeprecationWarning, stacklevel=2)
        self.gtess = gtess

    @classmethod
    def from_geometric_pxtal(cls,
                gsgen_method='shapely_pxtal_load',
                pxtal=None, xbound=None, ybound=None):
        """
        Class method to mesh geometrified grain struture.

        Deprecated: prefer building Shapely ``flat_cells`` and calling
        :meth:`confMesh2dGMSH.femesh_gmsh` or :func:`upxo.meshing.gsmesh2d.mesh_gs`.
        """
        from upxo.pxtal.polyxtal import vtpolyxtal2d as vtpxtal
        gtess = vtpxtal(gsgen_method=gsgen_method, vt_base_tool='shapely', pxtal=pxtal,
                    points=None, point_method=None, point_object_deque=None,
                    mulpoint_object=None, locx_list=None, locy_list=None, xbound=xbound,
                    ybound=ybound, vis_vtgs=False, lean='no', INSTANCE=None)
        return cls(gtess=gtess)
    
    def femesh_pygmsh(self, elementShape='tri', elementOrder=1, 
                    meshingAlgorithmID=4, elsize_global=[1.0],
                    intermediateFilename='femesh', intermediateFileformat='vtk'):
        """Femesh via pygmsh.

        Deprecated: use :meth:`confMesh2dGMSH.femesh_gmsh`.
        """
        self.mesherTool = 'pygmsh'
        self.elementShape = elementShape
        self.elementOrder = elementOrder
        self.meshingAlgorithmID = meshingAlgorithmID

        from upxo.meshing.pxtalmesh_01 import geo_pxtal_mesh
        pxtal_mesh = geo_pxtal_mesh(mesher='pygmsh', pxtal=self.gtess, level=0,
                    elshape=elementShape, elorder=elementOrder,
                    algorithm=meshingAlgorithmID, elsize_global=elsize_global, 
                    optimize=True, sta=True, wtfs=True, ff=['vtk', 'inp'], throw=False)

        self.pxtal_mesh = pxtal_mesh
        self.gtess.mesh = pxtal_mesh.mesh[3]

        fn = intermediateFilename
        ff = intermediateFileformat
        self._intermediate_mesh_filename_ = f'{fn}.{ff}'

        self._write_intermediate_mesh_file_()
        self._read_intermediate_mesh_file_()

        self.filter_mesh()

    def set_fids(self, geomObject):
        """
        Assign each cell feature, a unique ID (fid) starting from 1.

        Parameters
        ---------- 
        geomObject: list of shapely geometries Geometries that will be used to form feature IDs. 

        Example
        -------
        gsConfMesh.set_fids(gsConfMesh.gtess.L0.pxtal.geoms)
        """
        self.fids = np.arange(1, len(self.gtess.L0.pxtal.geoms)+1)

    def filter_mesh(self):
        """Filter mesh."""
        import meshio
        gmsh_mesh = meshio.read(self._intermediate_mesh_filename_)
        supported_cell_types = ["line", "triangle", "quad"]
        self.filtered_cells = {cell_type: cells 
                          for cell_type, cells in gmsh_mesh.cells_dict.items() 
                          if cell_type in supported_cell_types}
        self.filtered_mesh = meshio.Mesh(points=gmsh_mesh.points,
                    cells=self.filtered_cells, point_data=gmsh_mesh.point_data,
                    cell_data=gmsh_mesh.cell_data, field_data=gmsh_mesh.field_data,)
        
    def get_mesh_geometry(self):
        """Return the mesh geometry."""
        cellTypes = [c.type for c in self.filtered_mesh.cells]
        cellTypeLoc = {cellType: i for i, cellType in enumerate(cellTypes)}

        points = self.filtered_mesh.points[:, :2]
        lines = self.filtered_mesh.cells[cellTypeLoc['line']].data if 'line' in cellTypes else None
        triangles = self.filtered_mesh.cells[cellTypeLoc['triangle']].data if 'triangle' in cellTypes else None
        quads = self.filtered_mesh.cells[cellTypeLoc['quad']].data if 'quad' in cellTypes else None
        return points, lines, triangles, quads

    def _write_intermediate_mesh_file_(self, fileName=None):
        """ write intermediate mesh file ."""
        if fileName == None:
            self.gtess.mesh.write(self._intermediate_mesh_filename_)
        else:
            if type(fileName) == str:
                self.gtess.mesh.write(fileName)

    def _read_intermediate_mesh_file_(self, fileName=None):
        """ read intermediate mesh file ."""
        if fileName == None:
            self.grid = pv.read(self._intermediate_mesh_filename_)
        else:
            if type(fileName) == str:
                self.grid = pv.read(fileName)

    def assess_quality(self, qualityMeasures=['aspect_ratio', 'skew', 'min_angle', 'area']):
        """Assess quality."""
        mqm_data, mqm_dataframe = self.pxtal_mesh.assess_pygmsh(grid=self.grid,
                    mesh_quality_measures=qualityMeasures, elshape=self.elementShape,
                    elorder=self.elementOrder, algorithm=self.meshingAlgorithmID)
        return mqm_data, mqm_dataframe
        
    def see_mesh_quality(self, mqm_data, mqm_dataframe, data_to_vis='mesh > quality > field',
                         qualityMeasures=['aspect_ratio', 'skew', 'min_angle', 'area'] ):
        """See mesh quality."""
        # DEPRECATED
        clims = [[1.0, 2.5], [-1.0, 1.0], [0.0, 90.0],
                 [0.0*mqm_dataframe['area'].max(), mqm_dataframe['area'].max()]]
        self.pxtal_mesh.vis_pyvista(data_to_vis=data_to_vis,
                    mesh_qual_fields=mqm_data,
                    mesh_qual_field_vis_par={'mesh_quality_measures': qualityMeasures,
                            'cpos': 'xy', 'scalars': 'CellQuality',
                            'show_edges': False, 'cmap': 'viridis',
                            'clims': clims, 'below_color': 'white',
                            'above_color': 'black', } )
        
    def see_femesh(self, *args, **kwargs):
        """See femesh."""
        from upxo.viz.meshviz import see_femesh
        fig, ax = see_femesh(*args, **kwargs)
        return fig, ax
    
    def find_meshCellFeatTypes(self):
        """Find meshCellFeatTypes."""
        self.meshCellFeatTypes = list(self.filtered_mesh.cells_dict.keys())

    def find_lineFeatLocation(self):
        """Find lineFeatLocation."""
        self.lineFeatLocation = int(np.argwhere([ft == 'line' for ft in self.meshCellFeatTypes])[0][0]) 

    def find_GBlines(self):
        """Find GBlines."""
        self.GBlines = self.filtered_mesh.cells[self.lineFeatLocation].data

    def find_availableFeatures(self):
        """Find availableFeatures."""
        self.availableFeatures = list(self.filtered_cells.keys())

    def find_availableElTypes(self):
        """Find availableElTypes."""
        self.availableElTypes = [eltype for eltype in self.availableFeatures if eltype in self.ValidElTypesOptions]

    def find_availableElTypeID(self):
        """Find availableElTypeID."""
        self.availableElTypeID = {eltype: int(np.argwhere(np.array(self.availableFeatures) == eltype)[0][0]) 
                                  for eltype in self.availableElTypes}

    def build_nodes(self):
        """Build and return  nodes."""
        self.nodes = self.filtered_mesh.points

    def rebuild_elConnectivity(self):
        """Rebuild elconnectivity."""
        self.elConn = elemOps.rebuild_elConnectivity(availableElTypes=self.availableElTypes,
                        availableFeatures=self.availableFeatures, filtered_mesh_cells=self.filtered_mesh.cells)

    def get_elCentroids_singleElType(self, nodes, elConn):
        """Return the elCentroids singleElType."""
        from shapely.geometry import Point
        from shapely.strtree import STRtree
        elCentroids_coords = nodes[elConn].mean(axis=1)[:, :2]
        elCentroids_shPoints = [Point(x, y) for x, y, in elCentroids_coords]
        elCentroids_Tree = STRtree(elCentroids_shPoints)
        return elCentroids_coords, elCentroids_shPoints, elCentroids_Tree

    def _form_elset_(self, cell, elCentroids, elCentroidsTree):
        """ form elset ."""
        #polygon = gsConfMesh.gtess.L0.pxtal.geoms[cid]
        candidates = elCentroidsTree.query(cell)
        containedElIds = np.array([int(pt) for pt in candidates if elCentroids[pt].within(cell)])
        return containedElIds

    def _form_elsets_elType_(self, fids, nodes, elConn_eTypeSubSet, prefix='grain.'):
        """ form elsets eltype ."""
        centroids_coords, elCentroids, elCentroidsTree = self.get_elCentroids_singleElType(nodes, elConn_eTypeSubSet)
        elsets = {}
        for fid in fids:
            cell = self.gtess.L0.pxtal.geoms[fid-1]
            elsets[prefix+str(fid)] = self._form_elset_(cell, elCentroids, elCentroidsTree)
        return elsets

    def form_elsets_elType(self):
        """Form elsets eltype."""
        fids = np.arange(1, len(self.gtess.L0.pxtal.geoms)+1)
        self.elsets_eltype = {}
        for eltype in self.availableElTypes:
            print(f"Working with element type: {eltype}")
            elConn_eTypeSubSet = self.elConn[eltype]
            self.elsets_eltype[eltype] = self._form_elsets_elType_(fids, self.nodes, elConn_eTypeSubSet, prefix='grain.')

    def get_elCentroids(self):
        """Return the elCentroids."""
        return elemOps.get_elCentroids_2d(self.nodes, self.elConn, self.availableElTypes)
    
    def find_elID_ranges(self):
        """Find elID ranges."""
        self.elID_ranges = {}
        for i, eltype in enumerate(self.availableElTypes):
            self.elID_ranges[eltype] = [1, self.filtered_mesh.cells[self.availableElTypeID[eltype]].data.shape[0]]
            if i > 0:
                self.elID_ranges[eltype][0] = self.elID_ranges[self.availableElTypes[i-1]][1] + 1

    def form_elsets(self):
        """
        Consilidates elment sets of all element types into one single dictionary.
        """
        # elsets_eltype = self.form_elsets_elType(self.availableElTypes)
        # availableElTypes = list(elsets_eltype.keys())
        # ------------
        print(f"Available element types are: {self.availableElTypes}")
        # ------------
        # self.elsets_eltype
        if len(self.availableElTypes) == 1:
            self.elsets = self.elsets_eltype[self.availableElTypes[0]]
            return
        # ------------
        self.elsets = {elsetName: elset.tolist() for elsetName, elset in self.elsets_eltype[self.availableElTypes[0]].items()}
        nel0 = self.elID_ranges[self.availableElTypes[0]][1]

        for elsetName, elset in self.elsets_eltype[self.availableElTypes[1]].items():
            elset = elset + nel0
            self.elsets[elsetName].extend(elset.tolist())
            self.elsets[elsetName] = np.array(self.elsets[elsetName], dtype=np.int32)

    def extract_gblines_for_grain(self, grain_name):
        """Extract gblines for grain."""
        grain_nodes = np.array([], dtype=int)

        for eltype, conn in self.elConn.items():
            if eltype not in self.elsets_eltype:
                continue
            elem_ids = self.elsets_eltype[eltype].get(grain_name, np.array([], dtype=int))
            if elem_ids.size == 0:
                continue
            grain_nodes = np.unique(np.concatenate((grain_nodes, conn[elem_ids].ravel())))

        if grain_nodes.size == 0:
            return np.empty((0, 2), dtype=int)

        mask = np.isin(self.GBlines, grain_nodes).all(axis=1)
        return self.GBlines[mask]

    def extract_gblines_grains(self):
        """Extract gblines grains."""
        firstKey = list(self.elsets_eltype.keys())[0]
        gblines_by_grain = {}
        for grain_name in self.elsets_eltype[firstKey].keys():
            gblines_by_grain[grain_name] = self.extract_gblines_for_grain(grain_name)
        return gblines_by_grain
    
    def extract_gbnodes_grains(self):
        """Extract gbnodes grains."""
        nodeIDs = {gname: np.unique(gblines) for gname, gblines in self.extract_gblines_grains().items()}
        return nodeIDs
    
    def extract_gbnodeCoords_grains(self):
        """Extract gbnodecoords grains."""
        nodeIDs= self.extract_gbnodes_grains()
        return {gname: self.nodes[nodeIDs] for gname, nodeIDs in nodeIDs.items()}

    def extract_gbCoords_ckdTrees(self, preserve_ZDim=False):
        """Extract gbcoords ckdtrees."""
        gbCoords = self.extract_gbnodeCoords_grains()
        from scipy.spatial import cKDTree
        gbCoords_ckdTrees = {gname: cKDTree(coords[:, :2] if not preserve_ZDim else coords) for gname, coords in gbCoords.items()}
        return gbCoords_ckdTrees
    
    def calc_element_qualities(self, ar=True, saa=False, throw=False):
        """Return the  element qualities."""
        elQual = {}
        if ar:
            elQual['ar'] = elemOps.compute_elementQuality_AR_2d(self.nodes, self.elConn)
        if saa: 
            self.elQual = elQual
        if throw:
            return elQual

    def find_elIDs_by_quality(self, **kwargs):
        """Find elIDs by quality."""
        return elemOps.find_elIDs_by_quality(**kwargs)

    def plot_elements_geometric_grain(self, grain_name, **kwargs):
        """Visualise elements geometric grain using Matplotlib or PyVista."""
        meshviz.plot_elements_geometric_grain(grain_name, **kwargs)

    def plot_elements_by_elIDs(self, element_ids, **kwargs):
        """Visualise elements by elIDs using Matplotlib or PyVista."""
        meshviz.plot_elements_by_elIDs(element_ids, **kwargs)
    
    def plot_elements_geometric_grains(self, **kwargs):
        """Visualise elements geometric grains using Matplotlib or PyVista."""
        meshviz.plot_elements_geometric_grains(**kwargs)

    def extract_gb_elements_for_grain(self, grain_name):
        """Extract gb elements for grain."""
        gb_nodes = np.unique(self.GBlines)
        gb_elements = {}
        for eltype, conn in self.elConn.items():
            elem_ids = self.elsets_eltype.get(eltype, {}).get(grain_name, np.array([], dtype=int))
            if elem_ids.size == 0:
                gb_elements[eltype] = {
                    "elem_ids": np.array([], dtype=int),
                    "conn": np.empty((0, conn.shape[1]), dtype=int),
                }
                continue
            elem_ids = elem_ids.astype(int)
            elems = conn[elem_ids]
            mask = np.isin(elems, gb_nodes).any(axis=1)
            gb_elements[eltype] = {"elem_ids": elem_ids[mask], "conn": elems[mask]}
        return gb_elements

    def collect_gb_elements_for_grains(self, grain_ids=None, grain_names=None, prefix="grain"):
        """
        Orchestrator to extract GB elements for multiple grains using extract_gb_elements_for_grain.
        Accepts either grain_ids or grain_names. If grain_ids are provided, uses
        prefix + '.' + str(grainid) to form grain names.
        """
        if grain_ids is None and grain_names is None:
            raise ValueError("Provide grain_ids or grain_names.")
        if grain_names is None:
            grain_names = [f"{prefix}.{gid}" for gid in grain_ids]

        gb_elements_by_grain = {}
        for grain_name in grain_names:
            gb_elements_by_grain[grain_name] = self.extract_gb_elements_for_grain(grain_name)
        return gb_elements_by_grain
    
    def see_gbElements_grains(self, grain_name, **kwargs):
        """See gbelements grains."""
        gb_elements_grain = self.extract_gb_elements_for_grain(grain_name)
        meshviz.see_gbElements_grains(gb_elements_grain, **kwargs)

    def build_global_element_numbering(self):
        """Build and return  global element numbering."""
        glb_elm_num = elemOps.build_global_element_numbering(self.elConn, elID_ranges=self.elID_ranges)
        return glb_elm_num
    
    def find_el_neigh(self, element_ids, n_order=1, eltype=None, include_self=False):
        """Find el neigh."""
        return elemOps.find_el_neigh(element_ids, self.elConn, n_order=n_order, 
                                    eltype=eltype, include_self=include_self)

    def find_nthOrderNeigh(self, n_order, el_subset, include_self=False):
        """Find nthOrderNeigh."""
        return elemOps.find_nthOrderNeigh(n_order, el_subset, self.availableElTypes, self.elConn, include_self=include_self)

    def resolve_eltypes(self, grain_name, grainElements, grainCoordinates, eltypes=None):
        """Resolve eltypes."""
        return elemOps.resolve_eltypes(
            grain_name, grainElements, grainCoordinates, eltypes=eltypes
        )

    def build_element_ids_by_band(self, grain_name, bands, eltypes, grainElements, nearest_dist_by_type):
        """Build and return  element ids by band."""
        return elemOps.build_element_ids_by_band(
            grain_name, bands, eltypes, grainElements, nearest_dist_by_type
        )

    def pick_contrasting_colours_from_cmap(self, n_colours, cmap_name='nipy_spectral'):
        """Pick contrasting colours from cmap."""
        return meshviz.pick_contrasting_colours_from_cmap(n_colours, cmap_name=cmap_name)

    def resolve_band_colours(self, bands, band_colours=None, auto_cmap='nipy_spectral'):
        """Resolve band colours."""
        return meshviz.resolve_band_colours(
            bands, band_colours=band_colours, auto_cmap=auto_cmap
        )

    def resolve_plot_eltype(self, eltypes, plot_eltype=None):
        """Resolve plot eltype."""
        return meshviz.resolve_plot_eltype(eltypes, plot_eltype=plot_eltype)

    def plot_band_elements(self, element_ids_by_band, bands, plot_eltype, gbcoords,
                           gblines_by_grain=None, colours_to_use=None,
                           title='Selected elements by band', band_facecolors=False):
        """Visualise band elements using Matplotlib or PyVista."""
        return meshviz.plot_band_elements(
            element_ids_by_band, bands, plot_eltype, gbcoords,
            self, gblines_by_grain=gblines_by_grain,
            colours_to_use=colours_to_use, title=title,
            band_facecolors=band_facecolors
        )

    def select_elements_in_bands(self, grain_name, bands, gbnodeCoords,
                                 grainCoordinates, grainElements, eltypes=None,
                                 plot_eltype='quad', gblines_by_grain=None,
                                 plot=True, band_colours=None,
                                 auto_cmap='nipy_spectral',
                                 band_facecolors=False):
        """Select elements in bands."""
        element_ids_by_band, nearest_dist_by_type, selected_eltypes = elemOps.select_elements_in_bands(
            grain_name, bands, gbnodeCoords, grainCoordinates, grainElements, eltypes=eltypes
        )

        fig = ax = None
        if plot:
            colours_to_use = self.resolve_band_colours(
                bands, band_colours=band_colours, auto_cmap=auto_cmap
            )
            if eltypes is None:
                plot_eltype_resolved = selected_eltypes
            else:
                plot_eltype_resolved = self.resolve_plot_eltype(
                    selected_eltypes, plot_eltype=plot_eltype
                )
            gbcoords = gbnodeCoords[grain_name][:, :2]
            title = 'Selected elements by band'
            if isinstance(plot_eltype_resolved, str):
                title = f'Selected elements by band ({plot_eltype_resolved})'
            elif isinstance(plot_eltype_resolved, (list, tuple, set)) and len(plot_eltype_resolved) > 0:
                title = f"Selected elements by band ({', '.join(plot_eltype_resolved)})"
            fig, ax = self.plot_band_elements(
                element_ids_by_band, bands, plot_eltype_resolved, gbcoords,
                gblines_by_grain=gblines_by_grain,
                colours_to_use=colours_to_use, title=title,
                band_facecolors=band_facecolors
            )

        return element_ids_by_band, nearest_dist_by_type, (fig, ax), selected_eltypes


class confMesh2dGMSH():
    """
    Conformal 2D FE mesher via the raw gmsh API (no pygmsh).

    Accepts Shapely grain polygons (``flat_cells``) and optional
    ``gid_map``, then meshes with GB-refined / bulk size controls.
    Mirrors post-processing from :class:`confMesh2d` (ELSETs, boundary /
    grain / GB NSETs) so pygmsh can be phased out without breaking
    workflows.

    Typical workflow
    ----------------
    >>> from upxo.meshing.conformal_mesher2d import confMesh2dGMSH as cm2dg
    >>> m = confMesh2dGMSH()
    >>> m.femesh_gmsh(flat_cells, gid_map,
    ...               mesh_size_gb=0.75, mesh_size_bulk=4.5,
    ...               mesh_algo=8, mesh_order=1, recombine_to_quads=True)
    >>> m.form_elsets_gmsh()
    >>> m.build_boundary_nsets(); m.build_grain_nsets(); m.build_gb_nset()

    Geometric polycrystals (the old ``confMesh2d.from_geometric_pxtal``
    path) use :meth:`from_geometric_pxtal` then :meth:`femesh_gmsh`.
    """

    ValidElTypesOptions = ('triangle', 'quad')

    __slots__ = (
        # geometry / topology inputs
        'gtess', 'fids', 'gid_map', 'flat_cells',
        # gmsh internal bookkeeping
        'point_registry', 'surface_tags', 'physical_surface_tags',
        # mesh parameters
        'mesherTool', 'elementShape', 'elementOrder',
        'meshingAlgorithmID', 'recombine_to_quads',
        # extracted mesh data
        'nodes', 'elConn', 'GBlines',
        # element set / classification data
        'availableFeatures', 'availableElTypes', 'availableElTypeID',
        'elsets_eltype', 'elsets', 'elID_ranges',
        '_physical_elsets',
        '_elem_owners',
        # node sets
        'nsets',
        # pyvista grid (kept for optional pyvista workflows)
        'grid',
        # export / validation
        '_exported',
        'validation_report',
        'fidelity_report',
        'quality_report',
    )

    def __init__(self):
        """Initialise the instance."""
        self.gtess = None
        self.fids = None
        self.gid_map = None
        self.flat_cells = None
        self.point_registry = {}
        self.surface_tags = {}
        self.physical_surface_tags = {}
        self.mesherTool = 'gmsh'
        self.elementShape = 'tri'
        self.elementOrder = 1
        self.meshingAlgorithmID = 6
        self.recombine_to_quads = False
        self.nodes = None
        self.elConn = {}
        self.GBlines = None
        self.availableFeatures = []
        self.availableElTypes = []
        self.availableElTypeID = {}
        self.elsets_eltype = {}
        self.elsets = {}
        self.elID_ranges = {}
        self._physical_elsets = {}
        self._elem_owners = {}
        self.nsets = {}
        self.grid = None
        self._exported = []
        self.validation_report = None
        self.fidelity_report = None
        self.quality_report = None

    @classmethod
    def from_geometric_pxtal(cls,
                             gsgen_method='shapely_pxtal_load',
                             pxtal=None, xbound=None, ybound=None):
        """Build a ``confMesh2dGMSH`` from a geometric polycrystal.

        Drop-in replacement for the deprecated
        ``confMesh2d.from_geometric_pxtal``. Call :meth:`femesh_gmsh`
        afterwards (``flat_cells`` may be omitted; they are taken from
        the tessellation).
        """
        from upxo.pxtal.polyxtal import vtpolyxtal2d as vtpxtal
        from upxo.meshing.gsmesh2d import _flatten_cells

        gtess = vtpxtal(
            gsgen_method=gsgen_method, vt_base_tool='shapely', pxtal=pxtal,
            points=None, point_method=None, point_object_deque=None,
            mulpoint_object=None, locx_list=None, locy_list=None,
            xbound=xbound, ybound=ybound, vis_vtgs=False, lean='no',
            INSTANCE=None)
        geoms = list(gtess.L0.pxtal.geoms)
        cells = {i + 1: geom for i, geom in enumerate(geoms)}
        flat_cells, gid_map = _flatten_cells(cells)
        inst = cls()
        inst.gtess = gtess
        inst.flat_cells = flat_cells
        inst.gid_map = gid_map
        inst.fids = np.arange(1, len(geoms) + 1)
        return inst

    def set_fids(self, geomObject):
        """Assign 1-based feature IDs from a sequence of grain geometries."""
        self.fids = np.arange(1, len(geomObject) + 1)

    # ------------------------------------------------------------------
    # Main meshing entry point
    # ------------------------------------------------------------------

    def femesh_gmsh(self, flat_cells=None, gid_map=None,
                    mesh_size_gb=1.0, mesh_size_bulk=4.0,
                    mesh_algo=6, mesh_order=1,
                    recombine_to_quads=False,
                    out_dir=None, basename='gs_mesh', formats=None,
                    field_sampling=None, n_threads=None,
                    snap_tol=None, island_cover_frac=0.95,
                    validate=True, verbose=False,
                    dist_min=None, dist_max=None,
                    clean_geometry=True, unify_winding=True,
                    optimize=False, mesh_size_from_curvature=False,
                    fidelity_tol=None):
        """
        Full gmsh meshing pipeline.

        Parameters
        ----------
        flat_cells : dict or None
            {flat_id: shapely.geometry.Polygon}.  If None, uses cells stored
            by :meth:`from_geometric_pxtal`.
        gid_map : dict, optional
            {flat_id: original_grain_id}.  If None, flat_id is used as-is.
        mesh_size_gb : float
            Target element size on grain boundaries.
        mesh_size_bulk : float
            Target element size in grain interiors.
        mesh_algo : int
            gmsh mesh algorithm ID (e.g. 6=Frontal, 8=Frontal-Delaunay quads).
        mesh_order : int
            Element order (1=linear, 2=quadratic).
        recombine_to_quads : bool
            Whether to recombine triangles into quads after meshing.
        out_dir : str or None
            Directory to write exported mesh files.  No export when None.
        basename : str
            Filename stem for exported files (extension appended per format).
        formats : list of str or None
            File format extensions to export, e.g. ``['msh', 'inp', 'vtk']``.
        field_sampling : int or None
            Distance-field samples per curve.  None chooses from curve count.
        n_threads : int or None
            Gmsh ``Mesh.MaxNumThreads``.  None uses ``os.cpu_count()``.
        snap_tol : float or None
            Point-merge quantum.  None is 1e-9 of the domain diagonal.
        island_cover_frac : float
            Fraction of an inner polygon's area that must lie in an outer
            polygon to treat it as an island (strict ``contains`` is too
            brittle after smoothing).
        validate : bool
            Raise if a grain has no elements or inverted elements are found.
        verbose : bool
            Let Gmsh print to the terminal.
        dist_min, dist_max : float or None
            Threshold-field distances.  Defaults: ``mesh_size_gb`` and
            ``2 * mesh_size_bulk``.
        clean_geometry : bool
            Snap near-duplicate vertices, drop collapsed edges, and orient
            rings CCW/CW before Gmsh.
        unify_winding : bool
            Reverse clockwise elements after extract so signed areas are >= 0.
        optimize : bool
            Run Gmsh ``Laplace2D`` (then ``Relocate2D`` if available) after
            generate.  Leftover triangles after recombination are kept.
        mesh_size_from_curvature : bool
            Let Gmsh grade size from GB curvature in addition to the
            distance Threshold field.
        fidelity_tol : float or None
            If set, raise when max grain-area relative error exceeds this.
        """
        import os
        try:
            import gmsh
        except ImportError as exc:
            raise ImportError(
                "confMesh2dGMSH requires the gmsh Python package. "
                "Install with: pip install gmsh"
            ) from exc

        if flat_cells is None:
            if not self.flat_cells:
                raise ValueError(
                    "flat_cells is required unless from_geometric_pxtal() "
                    "was used.")
            flat_cells = self.flat_cells
        if gid_map is None:
            gid_map = self.gid_map if self.gid_map is not None else {
                k: k for k in flat_cells}

        if snap_tol is None:
            xs, ys = [], []
            for polygon in flat_cells.values():
                x, y = polygon.exterior.xy
                xs.extend(x)
                ys.extend(y)
            dx = (max(xs) - min(xs)) if xs else 1.0
            dy = (max(ys) - min(ys)) if ys else 1.0
            snap_tol = max(1e-12, 1e-9 * max(dx, dy, 1.0))
        if clean_geometry:
            flat_cells = self.prepare_grain_polygons(
                flat_cells, min_edge=None, snap_tol=snap_tol)

        self.flat_cells = flat_cells
        self.elementShape = 'quad' if recombine_to_quads else 'tri'
        self.elementOrder = mesh_order
        self.meshingAlgorithmID = mesh_algo
        self.recombine_to_quads = recombine_to_quads
        self.gid_map = gid_map
        if dist_min is None:
            dist_min = mesh_size_gb
        if dist_max is None:
            dist_max = mesh_size_bulk * 2

        initialized = False
        try:
            gmsh.initialize()
            initialized = True
            gmsh.model.add("upxo_confmesh")
            gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
            threads = n_threads if n_threads is not None else (os.cpu_count() or 1)
            threads = int(max(1, threads))
            for opt in ("General.NumThreads", "Mesh.MaxNumThreads"):
                try:
                    gmsh.option.setNumber(opt, threads)
                except Exception:
                    pass

            self._build_geometry(
                flat_cells, mesh_size_gb, gmsh,
                snap_tol=snap_tol, island_cover_frac=island_cover_frac)
            self._assign_physical_groups(flat_cells, gmsh)
            self._set_mesh_options_and_generate(
                mesh_size_gb, mesh_size_bulk, mesh_algo, mesh_order,
                recombine_to_quads, gmsh, field_sampling=field_sampling,
                dist_min=dist_min, dist_max=dist_max,
                optimize=optimize,
                mesh_size_from_curvature=mesh_size_from_curvature)
            self._extract_nodes_and_elements(gmsh, unify_winding=unify_winding)
            self._extract_gblines(gmsh)

            self._exported = []
            if out_dir and formats:
                os.makedirs(out_dir, exist_ok=True)
                for fmt in formats:
                    path = os.path.join(out_dir, f'{basename}.{fmt}')
                    gmsh.write(path)
                    self._exported.append(path)
        except Exception as exc:
            raise RuntimeError(
                f"Gmsh conformal 2D meshing failed: {exc}"
            ) from exc
        finally:
            if initialized:
                try:
                    gmsh.finalize()
                except Exception:
                    pass

        self._detect_available_eltypes()
        self._build_physical_elsets()
        self._validate_mesh(flat_cells, raise_on_fail=validate)
        self.report_fidelity(flat_cells)
        self.report_quality()
        if fidelity_tol is not None:
            err = self.fidelity_report.get('max_area_rel_error', 0.0)
            if err > fidelity_tol:
                raise RuntimeError(
                    f"Conformal 2D mesh grain-area relative error "
                    f"{err:.4g} exceeds fidelity_tol={fidelity_tol}")

    # ------------------------------------------------------------------
    # Geometry building
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_grain_polygons(flat_cells, min_edge=None, snap_tol=1e-9):
        """Snap near-duplicate vertices, drop collapsed edges, orient rings.

        Exterior rings are forced CCW and interiors CW (Shapely ``orient``).
        Invalid results fall back to the input polygon.
        """
        from shapely.geometry import Polygon
        from shapely.geometry.polygon import orient

        inv = 1.0 / max(float(snap_tol), 1e-18)

        def _clean_ring(coords):
            snapped = []
            for x, y in coords:
                key = (int(round(x * inv)), int(round(y * inv)))
                pt = (key[0] / inv, key[1] / inv)
                if not snapped or pt != snapped[-1]:
                    snapped.append(pt)
            if len(snapped) > 1 and snapped[0] == snapped[-1]:
                snapped = snapped[:-1]
            if min_edge is not None and min_edge > 0 and len(snapped) >= 3:
                kept = [snapped[0]]
                for pt in snapped[1:]:
                    dx = pt[0] - kept[-1][0]
                    dy = pt[1] - kept[-1][1]
                    if (dx * dx + dy * dy) ** 0.5 >= min_edge:
                        kept.append(pt)
                if len(kept) >= 2:
                    dx = kept[0][0] - kept[-1][0]
                    dy = kept[0][1] - kept[-1][1]
                    if (dx * dx + dy * dy) ** 0.5 < min_edge:
                        kept = kept[:-1]
                snapped = kept
            return snapped

        cleaned = {}
        for fid, poly in flat_cells.items():
            try:
                ext = _clean_ring(list(poly.exterior.coords)[:-1])
                holes = [_clean_ring(list(r.coords)[:-1]) for r in poly.interiors]
                holes = [h for h in holes if len(h) >= 3]
                if len(ext) < 3:
                    cleaned[fid] = poly
                    continue
                newp = Polygon(ext, holes)
                if not newp.is_valid or newp.area <= 0:
                    cleaned[fid] = poly
                    continue
                cleaned[fid] = orient(newp, sign=1.0)
            except Exception:
                cleaned[fid] = poly
        return cleaned

    @staticmethod
    def _polygon_cover_frac(outer, inner):
        """Fraction of ``inner`` area that lies in ``outer``."""
        if inner is None or inner.area <= 0:
            return 0.0
        try:
            return float(outer.intersection(inner).area / inner.area)
        except Exception:
            return 0.0

    def _build_geometry(self, flat_cells, mesh_size_gb, gmsh,
                        snap_tol=None, island_cover_frac=0.95):
        """Register shared points/curves and build one surface per grain.

        Shared grain-boundary edges get a single Gmsh line (the neighbour
        reuses the reverse tag). Shapely interior rings become holes.
        Island grains (coverage ≥ ``island_cover_frac`` of a larger parent)
        are also carved from the parent unless they already match a parent
        interior ring.
        """
        from collections import defaultdict
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree

        self.point_registry = {}
        self.surface_tags = {}

        xs, ys = [], []
        for polygon in flat_cells.values():
            x, y = polygon.exterior.xy
            xs.extend(x)
            ys.extend(y)
        dx = (max(xs) - min(xs)) if xs else 1.0
        dy = (max(ys) - min(ys)) if ys else 1.0
        if snap_tol is None:
            snap_tol = max(1e-12, 1e-9 * max(dx, dy, 1.0))
        inv_snap = 1.0 / snap_tol

        def _get_or_add_point(x, y):
            key = (int(round(x * inv_snap)), int(round(y * inv_snap)))
            if key not in self.point_registry:
                tag = gmsh.model.geo.addPoint(x, y, 0.0, mesh_size_gb)
                self.point_registry[key] = tag
            return self.point_registry[key]

        line_registry = {}

        def _get_or_add_line(a, b):
            if a == b:
                raise ValueError("Degenerate Gmsh line: coincident endpoints.")
            key, rev = (a, b), (b, a)
            if key in line_registry:
                return line_registry[key]
            if rev in line_registry:
                return -line_registry[rev]
            tag = gmsh.model.geo.addLine(a, b)
            line_registry[key] = tag
            return tag

        def _loop_from_coords(coords):
            pts = [_get_or_add_point(x, y) for x, y in coords]
            # Drop consecutive duplicates after snapping
            dedup = [pts[0]]
            for p in pts[1:]:
                if p != dedup[-1]:
                    dedup.append(p)
            if len(dedup) > 1 and dedup[0] == dedup[-1]:
                dedup = dedup[:-1]
            if len(dedup) < 3:
                raise ValueError("Polygon ring collapsed to fewer than 3 points.")
            n = len(dedup)
            lines = [_get_or_add_line(dedup[i], dedup[(i + 1) % n]) for i in range(n)]
            return gmsh.model.geo.addCurveLoop(lines)

        flat_id_list = list(flat_cells.keys())
        polygon_list = [flat_cells[fid] for fid in flat_id_list]

        loop_tags = {}
        interior_loops = defaultdict(list)
        for flat_id, polygon in flat_cells.items():
            loop_tags[flat_id] = _loop_from_coords(list(polygon.exterior.coords)[:-1])
            for ring in polygon.interiors:
                interior_loops[flat_id].append(
                    _loop_from_coords(list(ring.coords)[:-1]))

        outer_to_island_holes = defaultdict(list)
        tree = STRtree(polygon_list)
        for i, fid_inner in enumerate(flat_id_list):
            inner = polygon_list[i]
            candidates = tree.query(inner)
            container_indices = [
                j for j in candidates
                if j != i
                and polygon_list[j].area > inner.area
                and self._polygon_cover_frac(polygon_list[j], inner) >= island_cover_frac
            ]
            if not container_indices:
                continue
            parent_idx = min(container_indices, key=lambda j: polygon_list[j].area)
            parent = polygon_list[parent_idx]
            already_interior = any(
                self._polygon_cover_frac(Polygon(ring), inner) >= island_cover_frac
                for ring in parent.interiors
            )
            if not already_interior:
                outer_to_island_holes[flat_id_list[parent_idx]].append(
                    loop_tags[fid_inner])

        for flat_id in flat_cells:
            holes = list(interior_loops.get(flat_id, []))
            holes.extend(outer_to_island_holes.get(flat_id, []))
            surf_tag = gmsh.model.geo.addPlaneSurface([loop_tags[flat_id]] + holes)
            self.surface_tags[flat_id] = surf_tag

        gmsh.model.geo.synchronize()

    def _assign_physical_groups(self, flat_cells, gmsh):
        """Create one physical surface per original grain ID.
        MultiPolygon parts sharing the same orig_gid are merged into one group."""
        from collections import defaultdict
        orig_to_surfs = defaultdict(list)
        for flat_id in flat_cells:
            orig_gid = self.gid_map.get(flat_id, flat_id)
            orig_to_surfs[orig_gid].append(self.surface_tags[flat_id])
        self.physical_surface_tags = {}
        for orig_gid, surfs in orig_to_surfs.items():
            tag = gmsh.model.addPhysicalGroup(2, surfs, name=f"grain.{orig_gid}")
            self.physical_surface_tags[orig_gid] = tag

    def _set_mesh_options_and_generate(self, mesh_size_gb, mesh_size_bulk,
                                       mesh_algo, mesh_order,
                                       recombine_to_quads, gmsh,
                                       field_sampling=None,
                                       dist_min=None, dist_max=None,
                                       optimize=False,
                                       mesh_size_from_curvature=False):
        """Apply size fields, algorithm, and generate the mesh."""
        all_curves = [t for (_, t) in gmsh.model.getEntities(1)]
        if field_sampling is None:
            field_sampling = max(20, min(100, 2000 // max(len(all_curves), 1)))
        if dist_min is None:
            dist_min = mesh_size_gb
        if dist_max is None:
            dist_max = mesh_size_bulk * 2
        dist_tag = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_tag, "CurvesList", all_curves)
        gmsh.model.mesh.field.setNumber(dist_tag, "Sampling", int(field_sampling))

        thresh_tag = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_tag, "InField", dist_tag)
        gmsh.model.mesh.field.setNumber(thresh_tag, "SizeMin", mesh_size_gb)
        gmsh.model.mesh.field.setNumber(thresh_tag, "SizeMax", mesh_size_bulk)
        gmsh.model.mesh.field.setNumber(thresh_tag, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(thresh_tag, "DistMax", dist_max)

        gmsh.model.mesh.field.setAsBackgroundMesh(thresh_tag)

        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        curv = 12.0 if mesh_size_from_curvature else 0.0
        for opt in ("Mesh.MeshSizeFromCurvature",
                    "Mesh.CharacteristicLengthFromCurvature"):
            try:
                gmsh.option.setNumber(opt, curv)
            except Exception:
                pass
        gmsh.option.setNumber("Mesh.Algorithm", mesh_algo)
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_order)

        if recombine_to_quads:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

        gmsh.model.mesh.generate(2)

        if mesh_order > 1:
            gmsh.model.mesh.setOrder(mesh_order)

        if optimize:
            for method in ("Laplace2D", "Relocate2D"):
                try:
                    gmsh.model.mesh.optimize(method)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Mesh extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _flip_winding(conn, n_corners):
        """Reverse orientation; keep mid-side nodes on the same edges."""
        conn = np.array(conn, copy=True)
        n = conn.shape[1]
        if n_corners == 3:
            if n >= 6:
                conn[:, [1, 2, 3, 5]] = conn[:, [2, 1, 5, 3]]
            else:
                conn[:, [1, 2]] = conn[:, [2, 1]]
        elif n_corners == 4:
            if n >= 8:
                conn[:, [1, 3, 4, 5, 6, 7]] = conn[:, [3, 1, 7, 6, 5, 4]]
            else:
                conn[:, [1, 3]] = conn[:, [3, 1]]
        return conn

    def _extract_nodes_and_elements(self, gmsh, unify_winding=True):
        """Populate self.nodes and self.elConn from gmsh mesh data.

        Elements are stored in ``surface_tags`` order so physical-group
        ELSETs can be filled in the same pass (see ``_build_physical_elsets``).
        """
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = node_coords.reshape(-1, 3)
        tags = np.asarray(node_tags, dtype=np.int64)
        node_array = np.full((int(tags.max()) + 1, 3), np.nan)
        node_array[tags] = coords
        self.nodes = node_array

        TRI_TYPES = {2, 9}
        QUAD_TYPES = {3, 16}

        tri_conn = []
        quad_conn = []
        tri_owners = []  # orig_gid per triangle row
        quad_owners = []

        for flat_id, surf_tag in self.surface_tags.items():
            orig_gid = self.gid_map.get(flat_id, flat_id)
            el_types, el_tags, el_node_tags = gmsh.model.mesh.getElements(2, surf_tag)
            for etype, etags, entags in zip(el_types, el_tags, el_node_tags):
                n_nodes = gmsh.model.mesh.getElementProperties(etype)[3]
                connectivity = np.asarray(entags, dtype=int).reshape(-1, n_nodes)
                if etype in TRI_TYPES:
                    tri_conn.append(connectivity)
                    tri_owners.append(np.full(len(connectivity), orig_gid, dtype=int))
                elif etype in QUAD_TYPES:
                    quad_conn.append(connectivity)
                    quad_owners.append(np.full(len(connectivity), orig_gid, dtype=int))

        self.elConn = {}
        self._elem_owners = {}
        if tri_conn:
            tri = np.vstack(tri_conn)
            if unify_winding:
                areas = self._signed_areas(self.nodes, tri, 3)
                flip = areas < -1e-18
                if np.any(flip):
                    tri = tri.copy()
                    tri[flip] = self._flip_winding(tri[flip], 3)
            self.elConn['triangle'] = tri
            self._elem_owners['triangle'] = np.concatenate(tri_owners)
        if quad_conn:
            quad = np.vstack(quad_conn)
            if unify_winding:
                areas = self._signed_areas(self.nodes, quad, 4)
                flip = areas < -1e-18
                if np.any(flip):
                    quad = quad.copy()
                    quad[flip] = self._flip_winding(quad[flip], 4)
            self.elConn['quad'] = quad
            self._elem_owners['quad'] = np.concatenate(quad_owners)

    def _extract_gblines(self, gmsh):
        """Extract boundary (curve) line segments into self.GBlines."""
        line_segs = []
        for dim, tag in gmsh.model.getEntities(1):
            el_types, el_tags, el_node_tags = gmsh.model.mesh.getElements(dim, tag)
            for etype, etags, entags in zip(el_types, el_tags, el_node_tags):
                props = gmsh.model.mesh.getElementProperties(etype)
                n_nodes = props[3]
                connectivity = entags.reshape(-1, n_nodes).astype(int)
                line_segs.append(connectivity[:, :2])  # keep only endpoints
        if line_segs:
            self.GBlines = np.vstack(line_segs)
        else:
            self.GBlines = np.empty((0, 2), dtype=int)

    def _detect_available_eltypes(self):
        """ detect available eltypes."""
        self.availableElTypes = [et for et in ('triangle', 'quad') if et in self.elConn]
        self.availableElTypeID = {et: i for i, et in enumerate(self.availableElTypes)}
        self.availableFeatures = self.availableElTypes.copy()

    def _build_physical_elsets(self, prefix='grain.'):
        """ELSETs from Gmsh surface ownership (physical groups / surface tags).

        MultiPolygon parts that share ``orig_gid`` land in the same set.
        """
        from collections import defaultdict
        self._physical_elsets = {}
        for eltype, owners in self._elem_owners.items():
            mapping = defaultdict(list)
            for i, gid in enumerate(owners):
                mapping[f"{prefix}{gid}"].append(i)
            self._physical_elsets[eltype] = {
                name: np.asarray(idx, dtype=int) for name, idx in mapping.items()
            }

    @staticmethod
    def _signed_areas(nodes, conn, n_corners):
        """Shoelace signed area of the first ``n_corners`` nodes of each element."""
        xy = nodes[conn[:, :n_corners], :2]
        x, y = xy[:, :, 0], xy[:, :, 1]
        return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)

    def _validate_mesh(self, flat_cells, raise_on_fail=True):
        """Check every grain has elements and 2D signed areas are non-negative."""
        orig_gids = sorted({self.gid_map.get(fid, fid) for fid in flat_cells})
        missing = []
        counts = {}
        for gid in orig_gids:
            key = f"grain.{gid}"
            n = 0
            for mapping in self._physical_elsets.values():
                n += len(mapping.get(key, ()))
            counts[gid] = n
            if n == 0:
                missing.append(gid)

        clockwise = 0
        degenerate = 0
        for etype, n_corners in (('triangle', 3), ('quad', 4)):
            if etype not in self.elConn:
                continue
            areas = self._signed_areas(self.nodes, self.elConn[etype], n_corners)
            clockwise += int(np.count_nonzero(areas < -1e-18))
            degenerate += int(np.count_nonzero(np.abs(areas) < 1e-18))

        n_tri = 0 if 'triangle' not in self.elConn else len(self.elConn['triangle'])
        n_quad = 0 if 'quad' not in self.elConn else len(self.elConn['quad'])
        self.validation_report = {
            'n_tri': n_tri,
            'n_quad': n_quad,
            'grain_element_counts': counts,
            'grains_missing_elements': missing,
            'clockwise_elements': clockwise,
            'degenerate_elements': degenerate,
        }
        if not raise_on_fail:
            return self.validation_report
        if missing:
            raise RuntimeError(
                f"Conformal 2D mesh is missing elements for grain IDs: {missing}")
        if degenerate:
            raise RuntimeError(
                f"Conformal 2D mesh has {degenerate} degenerate element(s).")
        return self.validation_report

    def report_fidelity(self, flat_cells=None):
        """Compare mesh grain area and GB length to the input Shapely polygons.

        Relative error is ``|mesh - shapely| / shapely`` per original grain id.
        Shared GB edges are counted once in the mesh length and match
        ``polygon.boundary.length`` per grain (each shared edge appears on
        two grain boundaries).
        """
        if flat_cells is None:
            flat_cells = self.flat_cells or {}
        gid_map = self.gid_map or {k: k for k in flat_cells}

        shapely_area = {}
        shapely_gb = {}
        for fid, poly in flat_cells.items():
            gid = gid_map.get(fid, fid)
            shapely_area[gid] = shapely_area.get(gid, 0.0) + float(poly.area)
            shapely_gb[gid] = shapely_gb.get(gid, 0.0) + float(poly.boundary.length)

        mesh_area = {gid: 0.0 for gid in shapely_area}
        for etype, ncorn in (('triangle', 3), ('quad', 4)):
            conn = self.elConn.get(etype)
            owners = self._elem_owners.get(etype)
            if conn is None or owners is None or len(conn) == 0:
                continue
            areas = np.abs(self._signed_areas(self.nodes, conn, ncorn))
            for gid in shapely_area:
                mesh_area[gid] += float(areas[owners == gid].sum())

        mesh_gb = {gid: 0.0 for gid in shapely_area}
        if self.GBlines is not None and len(self.GBlines):
            pts = self.nodes[:, :2]
            seg = np.asarray(self.GBlines)
            lengths = np.linalg.norm(pts[seg[:, 1]] - pts[seg[:, 0]], axis=1)
            # Map each GB node to grains that use it, then assign a segment
            # to a grain if both endpoints belong to that grain.
            node_to_gids = {}
            for etype, conn in self.elConn.items():
                owners = self._elem_owners.get(etype)
                if conn is None or owners is None:
                    continue
                for i, nds in enumerate(conn):
                    gid = int(owners[i])
                    for n in np.unique(nds):
                        node_to_gids.setdefault(int(n), set()).add(gid)
            for (a, b), L in zip(seg, lengths):
                ga = node_to_gids.get(int(a), set())
                gb = node_to_gids.get(int(b), set())
                for gid in ga.intersection(gb):
                    if gid in mesh_gb:
                        mesh_gb[gid] += float(L)

        def _rel(mesh_v, shp_v):
            if shp_v <= 0:
                return 0.0 if mesh_v == 0 else 1.0
            return abs(mesh_v - shp_v) / shp_v

        area_err = {gid: _rel(mesh_area[gid], shapely_area[gid])
                    for gid in shapely_area}
        gb_err = {gid: _rel(mesh_gb[gid], shapely_gb[gid])
                  for gid in shapely_gb}
        self.fidelity_report = {
            'shapely_area': shapely_area,
            'mesh_area': mesh_area,
            'grain_area_rel_error': area_err,
            'shapely_gb_length': shapely_gb,
            'mesh_gb_length': mesh_gb,
            'grain_gb_length_rel_error': gb_err,
            'max_area_rel_error': max(area_err.values()) if area_err else 0.0,
            'max_gb_length_rel_error': max(gb_err.values()) if gb_err else 0.0,
        }
        return self.fidelity_report

    def report_quality(self):
        """Aspect-ratio and min-angle statistics (corners only on quadratic)."""
        ar = elemOps.compute_elementQuality_AR_2d(self.nodes, self.elConn)
        ang = elemOps.compute_min_angle_deg_2d(self.nodes, self.elConn)

        def _stats(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return {'n': 0, 'min': None, 'max': None, 'mean': None, 'p95': None}
            return {
                'n': int(arr.size),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'mean': float(arr.mean()),
                'p95': float(np.percentile(arr, 95)),
            }

        leftover_tri = 0
        if self.recombine_to_quads and 'triangle' in self.elConn:
            leftover_tri = len(self.elConn['triangle'])
        self.quality_report = {
            'aspect_ratio': {et: _stats(v) for et, v in ar.items()},
            'min_angle_deg': {et: _stats(v) for et, v in ang.items()},
            'leftover_triangles_after_recombine': leftover_tri,
        }
        return self.quality_report

    # ------------------------------------------------------------------
    # Element sets
    # ------------------------------------------------------------------

    def get_elCentroids_singleElType(self, nodes, elConn):
        """Return the elCentroids singleElType."""
        from shapely.geometry import Point
        from shapely.strtree import STRtree
        elCentroids_coords = nodes[elConn].mean(axis=1)[:, :2]
        elCentroids_shPoints = [Point(x, y) for x, y in elCentroids_coords]
        elCentroids_Tree = STRtree(elCentroids_shPoints)
        return elCentroids_coords, elCentroids_shPoints, elCentroids_Tree

    def _form_elset_(self, cell, elCentroids, elCentroidsTree):
        """ form elset ."""
        candidates = elCentroidsTree.query(cell)
        containedElIds = np.array([int(pt) for pt in candidates if elCentroids[pt].within(cell)])
        return containedElIds

    def _form_elsets_elType_(self, flat_cells, nodes, elConn_eTypeSubSet, prefix='grain.'):
        """ form elsets eltype ."""
        centroids_coords, elCentroids, elCentroidsTree = self.get_elCentroids_singleElType(nodes, elConn_eTypeSubSet)
        elsets = {}
        for flat_id, polygon in flat_cells.items():
            orig_gid = self.gid_map.get(flat_id, flat_id)
            key = f"{prefix}{orig_gid}"
            ids = self._form_elset_(polygon, elCentroids, elCentroidsTree)
            if key in elsets and len(elsets[key]) and len(ids):
                elsets[key] = np.unique(np.concatenate((elsets[key], ids)))
            elif key in elsets and len(elsets[key]) and not len(ids):
                pass
            else:
                elsets[key] = ids
        return elsets

    def form_elsets_gmsh(self, flat_cells=None, prefix='grain.',
                         verbose=False, verify_shapely=False):
        """
        Build per-grain element sets.

        Default source is Gmsh physical-group / surface ownership recorded
        at extract time. The old centroid-in-polygon walk is kept as an
        optional check (``verify_shapely=True``) or as a fallback if
        physical ELSETs were not built.

        Parameters
        ----------
        flat_cells : dict or None
            {flat_id: shapely.geometry.Polygon}.  Defaults to the dict
            used in :meth:`femesh_gmsh`.
        prefix : str
            Prefix for element set names (default 'grain.').
        verbose : bool
            Print per-element-type progress.
        verify_shapely : bool
            Also run the centroid-in-polygon assignment and warn if the
            grain membership counts disagree.
        """
        if flat_cells is None:
            flat_cells = self.flat_cells
        if verbose:
            print(f"Available element types: {self.availableElTypes}")

        self.elsets_eltype = {}
        if self._physical_elsets:
            for eltype in self.availableElTypes:
                mapping = self._physical_elsets.get(eltype, {})
                if prefix == 'grain.':
                    self.elsets_eltype[eltype] = dict(mapping)
                else:
                    self.elsets_eltype[eltype] = {
                        f"{prefix}{name.split('.', 1)[-1]}": idx
                        for name, idx in mapping.items()
                    }
        else:
            if flat_cells is None:
                raise RuntimeError(
                    "form_elsets_gmsh needs flat_cells or a prior femesh_gmsh.")
            for eltype in self.availableElTypes:
                if verbose:
                    print(f"  Forming elsets for element type: {eltype}")
                self.elsets_eltype[eltype] = self._form_elsets_elType_(
                    flat_cells, self.nodes, self.elConn[eltype], prefix=prefix
                )

        if verify_shapely and flat_cells is not None and self._physical_elsets:
            for eltype in self.availableElTypes:
                shapely_sets = self._form_elsets_elType_(
                    flat_cells, self.nodes, self.elConn[eltype], prefix=prefix)
                phys = self.elsets_eltype[eltype]
                for name, idx in phys.items():
                    other = shapely_sets.get(name, np.array([], dtype=int))
                    if len(idx) != len(other):
                        warnings.warn(
                            f"ELSET {name!r} ({eltype}): physical-tag count "
                            f"{len(idx)} != centroid-in-polygon count {len(other)}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

        self.find_elID_ranges()

        if len(self.availableElTypes) == 1:
            self.elsets = self.elsets_eltype[self.availableElTypes[0]]
            return

        self.elsets = {name: elset.tolist()
                       for name, elset in self.elsets_eltype[self.availableElTypes[0]].items()}
        nel0 = self.elID_ranges[self.availableElTypes[0]][1]
        for name, elset in self.elsets_eltype[self.availableElTypes[1]].items():
            elset_offset = np.asarray(elset) + nel0
            self.elsets.setdefault(name, [])
            self.elsets[name].extend(elset_offset.tolist())
            self.elsets[name] = np.array(self.elsets[name], dtype=np.int32)

    def find_elID_ranges(self):
        """Find elID ranges."""
        self.elID_ranges = {}
        for i, eltype in enumerate(self.availableElTypes):
            n = self.elConn[eltype].shape[0]
            if i == 0:
                self.elID_ranges[eltype] = [1, n]
            else:
                prev = self.elID_ranges[self.availableElTypes[i - 1]][1]
                self.elID_ranges[eltype] = [prev + 1, prev + n]

    def get_elCentroids(self):
        """Return the elCentroids."""
        return elemOps.get_elCentroids_2d(self.nodes, self.elConn, self.availableElTypes)

    # ------------------------------------------------------------------
    # Node sets
    # ------------------------------------------------------------------

    def build_boundary_nsets(self, tol=1e-6):
        """
        Build node sets for domain boundaries and corners.

        Populates self.nsets with keys:
          'LEFT', 'RIGHT', 'BOTTOM', 'TOP',
          'BOTTOM_LEFT', 'BOTTOM_RIGHT', 'TOP_LEFT', 'TOP_RIGHT'
        using node coordinates from self.nodes.
        """
        pts = self.nodes[:, :2]
        valid = ~np.isnan(pts[:, 0])
        node_ids = np.where(valid)[0]
        xy = pts[valid]

        xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
        ymin, ymax = xy[:, 1].min(), xy[:, 1].max()

        def _ids(mask):
            """ ids."""
            return node_ids[mask]

        left   = _ids(np.abs(xy[:, 0] - xmin) < tol)
        right  = _ids(np.abs(xy[:, 0] - xmax) < tol)
        bottom = _ids(np.abs(xy[:, 1] - ymin) < tol)
        top    = _ids(np.abs(xy[:, 1] - ymax) < tol)

        self.nsets['LEFT']         = left
        self.nsets['RIGHT']        = right
        self.nsets['BOTTOM']       = bottom
        self.nsets['TOP']          = top
        self.nsets['BOTTOM_LEFT']  = np.intersect1d(bottom, left)
        self.nsets['BOTTOM_RIGHT'] = np.intersect1d(bottom, right)
        self.nsets['TOP_LEFT']     = np.intersect1d(top, left)
        self.nsets['TOP_RIGHT']    = np.intersect1d(top, right)

    def build_grain_nsets(self):
        """
        Build per-grain node sets from self.elsets_eltype.
        Populates self.nsets with keys matching elset names (e.g. 'grain.1').
        """
        if not self.elsets_eltype:
            raise RuntimeError("Call form_elsets_gmsh() before build_grain_nsets().")
        first_eltype = self.availableElTypes[0]
        for grain_name in self.elsets_eltype[first_eltype]:
            grain_nodes = np.array([], dtype=int)
            for eltype, conn in self.elConn.items():
                elem_ids = self.elsets_eltype.get(eltype, {}).get(grain_name, np.array([], dtype=int))
                if elem_ids.size > 0:
                    grain_nodes = np.unique(np.concatenate((grain_nodes, conn[elem_ids].ravel())))
            self.nsets[grain_name] = grain_nodes

    def build_gb_nset(self):
        """
        Build a node set containing all grain-boundary nodes.
        Populates self.nsets['GB'].
        """
        if self.GBlines is None or self.GBlines.size == 0:
            self.nsets['GB'] = np.array([], dtype=int)
        else:
            self.nsets['GB'] = np.unique(self.GBlines)

    # ------------------------------------------------------------------
    # GB line / coordinate extraction (mirrors confMesh2d)
    # ------------------------------------------------------------------

    def extract_gblines_for_grain(self, grain_name):
        """Extract gblines for grain."""
        grain_nodes = np.array([], dtype=int)
        for eltype, conn in self.elConn.items():
            if eltype not in self.elsets_eltype:
                continue
            elem_ids = self.elsets_eltype[eltype].get(grain_name, np.array([], dtype=int))
            if elem_ids.size == 0:
                continue
            grain_nodes = np.unique(np.concatenate((grain_nodes, conn[elem_ids].ravel())))
        if grain_nodes.size == 0:
            return np.empty((0, 2), dtype=int)
        mask = np.isin(self.GBlines, grain_nodes).all(axis=1)
        return self.GBlines[mask]

    def extract_gblines_grains(self):
        """Extract gblines grains."""
        first_key = self.availableElTypes[0]
        gblines_by_grain = {}
        for grain_name in self.elsets_eltype[first_key]:
            gblines_by_grain[grain_name] = self.extract_gblines_for_grain(grain_name)
        return gblines_by_grain

    def extract_gbnodes_grains(self):
        """Extract gbnodes grains."""
        return {gname: np.unique(gblines)
                for gname, gblines in self.extract_gblines_grains().items()}

    def extract_gbnodeCoords_grains(self):
        """Extract gbnodecoords grains."""
        nodeIDs = self.extract_gbnodes_grains()
        return {gname: self.nodes[ids] for gname, ids in nodeIDs.items()}

    def extract_gbCoords_ckdTrees(self, preserve_ZDim=False):
        """Extract gbcoords ckdtrees."""
        from scipy.spatial import cKDTree
        gbCoords = self.extract_gbnodeCoords_grains()
        return {gname: cKDTree(coords[:, :2] if not preserve_ZDim else coords)
                for gname, coords in gbCoords.items()}

    def extract_gb_elements_for_grain(self, grain_name):
        """Extract gb elements for grain."""
        gb_nodes = np.unique(self.GBlines)
        gb_elements = {}
        for eltype, conn in self.elConn.items():
            elem_ids = self.elsets_eltype.get(eltype, {}).get(grain_name, np.array([], dtype=int))
            if elem_ids.size == 0:
                gb_elements[eltype] = {
                    "elem_ids": np.array([], dtype=int),
                    "conn": np.empty((0, conn.shape[1]), dtype=int),
                }
                continue
            elem_ids = elem_ids.astype(int)
            elems = conn[elem_ids]
            mask = np.isin(elems, gb_nodes).any(axis=1)
            gb_elements[eltype] = {"elem_ids": elem_ids[mask], "conn": elems[mask]}
        return gb_elements

    def collect_gb_elements_for_grains(self, grain_ids=None, grain_names=None, prefix="grain"):
        """Collect gb elements for grains."""
        if grain_ids is None and grain_names is None:
            raise ValueError("Provide grain_ids or grain_names.")
        if grain_names is None:
            grain_names = [f"{prefix}.{gid}" for gid in grain_ids]
        return {gname: self.extract_gb_elements_for_grain(gname) for gname in grain_names}

    # ------------------------------------------------------------------
    # Mesh visualisation
    # ------------------------------------------------------------------

    def get_mesh_geometry(self):
        """Return (points_2d, lines, triangles, quads) arrays for plotting."""
        pts = self.nodes[:, :2]
        lines = self.GBlines if (self.GBlines is not None and self.GBlines.size > 0) else None
        triangles = self.elConn.get('triangle', None)
        quads = self.elConn.get('quad', None)
        return pts, lines, triangles, quads

    def see_femesh(self, *args, **kwargs):
        """See femesh."""
        from upxo.viz.meshviz import see_femesh
        return see_femesh(*args, **kwargs)

    def plot_by_grain(self, **kwargs):
        """Grain-coloured 2D mesh plot. See ``upxo.viz.meshviz.plot_conformal_2d_by_grain``."""
        from upxo.viz.meshviz import plot_conformal_2d_by_grain
        if not self.elsets_eltype:
            self.form_elsets_gmsh()
        return plot_conformal_2d_by_grain(
            self.nodes, self.elConn, self.elsets_eltype,
            GBlines=self.GBlines, nsets=self.nsets, **kwargs)

    def export_abaqus_inp(self, path, **kwargs):
        """Abaqus ``.inp`` export. See ``upxo.meshing.writer_ABQ.export_confmesh2d_inp``."""
        from upxo.meshing.writer_ABQ import export_confmesh2d_inp
        if not self.elsets_eltype:
            self.form_elsets_gmsh()
        if not any(k in self.nsets for k in ('LEFT', 'RIGHT', 'TOP', 'BOTTOM')):
            self.build_boundary_nsets()
            self.build_gb_nset()
        return export_confmesh2d_inp(
            path, self.nodes, self.elConn, self.elsets_eltype,
            nsets=self.nsets, **kwargs)

    def see_gbElements_grains(self, grain_name, **kwargs):
        """See gbelements grains."""
        gb_elements_grain = self.extract_gb_elements_for_grain(grain_name)
        meshviz.see_gbElements_grains(gb_elements_grain, **kwargs)

    # ------------------------------------------------------------------
    # Element quality
    # ------------------------------------------------------------------

    def calc_element_qualities(self, ar=True, saa=False, throw=False):
        """Return the  element qualities."""
        elQual = {}
        if ar:
            elQual['ar'] = elemOps.compute_elementQuality_AR_2d(self.nodes, self.elConn)
        if saa:
            self.elQual = elQual
        if throw:
            return elQual
        return elQual

    def find_elIDs_by_quality(self, **kwargs):
        """Find elIDs by quality."""
        return elemOps.find_elIDs_by_quality(**kwargs)

    # ------------------------------------------------------------------
    # Neighbour / band analysis
    # ------------------------------------------------------------------

    def find_el_neigh(self, element_ids, n_order=1, eltype=None, include_self=False):
        """Find el neigh."""
        return elemOps.find_el_neigh(element_ids, self.elConn,
                                     n_order=n_order, eltype=eltype,
                                     include_self=include_self)

    def find_nthOrderNeigh(self, n_order, el_subset, include_self=False):
        """Find nthOrderNeigh."""
        return elemOps.find_nthOrderNeigh(n_order, el_subset,
                                          self.availableElTypes, self.elConn,
                                          include_self=include_self)

    def build_global_element_numbering(self):
        """Build and return  global element numbering."""
        return elemOps.build_global_element_numbering(self.elConn,
                                                      elID_ranges=self.elID_ranges)

    def resolve_eltypes(self, grain_name, grainElements, grainCoordinates, eltypes=None):
        """Resolve eltypes."""
        return elemOps.resolve_eltypes(grain_name, grainElements,
                                       grainCoordinates, eltypes=eltypes)

    def build_element_ids_by_band(self, grain_name, bands, eltypes,
                                  grainElements, nearest_dist_by_type):
        """Build and return  element ids by band."""
        return elemOps.build_element_ids_by_band(grain_name, bands, eltypes,
                                                  grainElements, nearest_dist_by_type)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def pick_contrasting_colours_from_cmap(self, n_colours, cmap_name='nipy_spectral'):
        """Pick contrasting colours from cmap."""
        return meshviz.pick_contrasting_colours_from_cmap(n_colours, cmap_name=cmap_name)

    def resolve_band_colours(self, bands, band_colours=None, auto_cmap='nipy_spectral'):
        """Resolve band colours."""
        return meshviz.resolve_band_colours(bands, band_colours=band_colours,
                                            auto_cmap=auto_cmap)

    def resolve_plot_eltype(self, eltypes, plot_eltype=None):
        """Resolve plot eltype."""
        return meshviz.resolve_plot_eltype(eltypes, plot_eltype=plot_eltype)

    def plot_elements_geometric_grain(self, grain_name, **kwargs):
        """Visualise elements geometric grain using Matplotlib or PyVista."""
        meshviz.plot_elements_geometric_grain(grain_name, **kwargs)

    def plot_elements_by_elIDs(self, element_ids, **kwargs):
        """Visualise elements by elIDs using Matplotlib or PyVista."""
        meshviz.plot_elements_by_elIDs(element_ids, **kwargs)

    def plot_elements_geometric_grains(self, **kwargs):
        """Visualise elements geometric grains using Matplotlib or PyVista."""
        meshviz.plot_elements_geometric_grains(**kwargs)

    def plot_band_elements(self, element_ids_by_band, bands, plot_eltype, gbcoords,
                           gblines_by_grain=None, colours_to_use=None,
                           title='Selected elements by band', band_facecolors=False):
        """Visualise band elements using Matplotlib or PyVista."""
        return meshviz.plot_band_elements(
            element_ids_by_band, bands, plot_eltype, gbcoords, self,
            gblines_by_grain=gblines_by_grain, colours_to_use=colours_to_use,
            title=title, band_facecolors=band_facecolors
        )

    def select_elements_in_bands(self, grain_name, bands, gbnodeCoords,
                                 grainCoordinates, grainElements, eltypes=None,
                                 plot_eltype='quad', gblines_by_grain=None,
                                 plot=True, band_colours=None,
                                 auto_cmap='nipy_spectral', band_facecolors=False):
        """Select elements in bands."""
        element_ids_by_band, nearest_dist_by_type, selected_eltypes = elemOps.select_elements_in_bands(
            grain_name, bands, gbnodeCoords, grainCoordinates, grainElements, eltypes=eltypes
        )
        fig = ax = None
        if plot:
            colours_to_use = self.resolve_band_colours(bands, band_colours=band_colours,
                                                        auto_cmap=auto_cmap)
            if eltypes is None:
                plot_eltype_resolved = selected_eltypes
            else:
                plot_eltype_resolved = self.resolve_plot_eltype(selected_eltypes,
                                                                 plot_eltype=plot_eltype)
            gbcoords = gbnodeCoords[grain_name][:, :2]
            title = 'Selected elements by band'
            if isinstance(plot_eltype_resolved, str):
                title = f'Selected elements by band ({plot_eltype_resolved})'
            elif isinstance(plot_eltype_resolved, (list, tuple, set)) and len(plot_eltype_resolved) > 0:
                title = f"Selected elements by band ({', '.join(plot_eltype_resolved)})"
            fig, ax = self.plot_band_elements(
                element_ids_by_band, bands, plot_eltype_resolved, gbcoords,
                gblines_by_grain=gblines_by_grain, colours_to_use=colours_to_use,
                title=title, band_facecolors=band_facecolors
            )
        return element_ids_by_band, nearest_dist_by_type, (fig, ax), selected_eltypes
