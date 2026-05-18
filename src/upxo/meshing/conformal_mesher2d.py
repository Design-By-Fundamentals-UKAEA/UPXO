import numpy as np
import matplotlib.pyplot as plt
from upxo.meshing import elemOps
from upxo.viz import meshviz
import pyvista as pv

class confMesh2d():
    """
    An orchastrator class capable of multiple 2D conformal mehsing pipelines.

    Import
    ------
    from upxo.meshing.conformal_mesher2d import confMesh2d as cm2d
    """
    ValidElTypesOptions =('triangle', 'quad')
    __slots__ = ('gtess', 'fids', 'pxtal_mesh', 'grid', 'mesherTool', 'elementShape', 
                 'elementOrder',  'meshingAlgorithmID', '_intermediate_mesh_filename_', 
                 'filtered_cells', 'filtered_mesh', 'nodes', 'elConn', 
                 'meshCellFeatTypes', 'lineFeatLocation', 'GBlines',
                 'availableFeatures', 'availableElTypes', 'availableElTypeID',
                 'elsets_eltype', 'elsets', 'elID_ranges')
    def __init__(self, gtess=None):
        self.gtess = gtess

    @classmethod
    def from_geometric_pxtal(cls,
                gsgen_method='shapely_pxtal_load',
                pxtal=None, xbound=None, ybound=None):
        """
        Class method to mesh geometrified grain struture
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
        cellTypes = [c.type for c in self.filtered_mesh.cells]
        cellTypeLoc = {cellType: i for i, cellType in enumerate(cellTypes)}

        points = self.filtered_mesh.points[:, :2]
        lines = self.filtered_mesh.cells[cellTypeLoc['line']].data if 'line' in cellTypes else None
        triangles = self.filtered_mesh.cells[cellTypeLoc['triangle']].data if 'triangle' in cellTypes else None
        quads = self.filtered_mesh.cells[cellTypeLoc['quad']].data if 'quad' in cellTypes else None
        return points, lines, triangles, quads

    def _write_intermediate_mesh_file_(self, fileName=None):
        if fileName == None:
            self.gtess.mesh.write(self._intermediate_mesh_filename_)
        else:
            if type(fileName) == str:
                self.gtess.mesh.write(fileName)

    def _read_intermediate_mesh_file_(self, fileName=None):
        if fileName == None:
            self.grid = pv.read(self._intermediate_mesh_filename_)
        else:
            if type(fileName) == str:
                self.grid = pv.read(fileName)

    def assess_quality(self, qualityMeasures=['aspect_ratio', 'skew', 'min_angle', 'area']):
        mqm_data, mqm_dataframe = self.pxtal_mesh.assess_pygmsh(grid=self.grid,
                    mesh_quality_measures=qualityMeasures, elshape=self.elementShape,
                    elorder=self.elementOrder, algorithm=self.meshingAlgorithmID)
        return mqm_data, mqm_dataframe
        
    def see_mesh_quality(self, mqm_data, mqm_dataframe, data_to_vis='mesh > quality > field',
                         qualityMeasures=['aspect_ratio', 'skew', 'min_angle', 'area'] ):
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
        from upxo.viz.meshviz import see_femesh
        fig, ax = see_femesh(*args, **kwargs)
        return fig, ax
    
    def find_meshCellFeatTypes(self):
        self.meshCellFeatTypes = list(self.filtered_mesh.cells_dict.keys())

    def find_lineFeatLocation(self):
        self.lineFeatLocation = int(np.argwhere([ft == 'line' for ft in self.meshCellFeatTypes])[0][0]) 

    def find_GBlines(self):
        self.GBlines = self.filtered_mesh.cells[self.lineFeatLocation].data

    def find_availableFeatures(self):
        self.availableFeatures = list(self.filtered_cells.keys())

    def find_availableElTypes(self):
        self.availableElTypes = [eltype for eltype in self.availableFeatures if eltype in self.ValidElTypesOptions]

    def find_availableElTypeID(self):
        self.availableElTypeID = {eltype: int(np.argwhere(np.array(self.availableFeatures) == eltype)[0][0]) 
                                  for eltype in self.availableElTypes}

    def build_nodes(self):
        self.nodes = self.filtered_mesh.points

    def rebuild_elConnectivity(self):
        self.elConn = elemOps.rebuild_elConnectivity(availableElTypes=self.availableElTypes,
                        availableFeatures=self.availableFeatures, filtered_mesh_cells=self.filtered_mesh.cells)

    def get_elCentroids_singleElType(self, nodes, elConn):
        from shapely.geometry import Point
        from shapely.strtree import STRtree
        elCentroids_coords = nodes[elConn].mean(axis=1)[:, :2]
        elCentroids_shPoints = [Point(x, y) for x, y, in elCentroids_coords]
        elCentroids_Tree = STRtree(elCentroids_shPoints)
        return elCentroids_coords, elCentroids_shPoints, elCentroids_Tree

    def _form_elset_(self, cell, elCentroids, elCentroidsTree):
        #polygon = gsConfMesh.gtess.L0.pxtal.geoms[cid]
        candidates = elCentroidsTree.query(cell)
        containedElIds = np.array([int(pt) for pt in candidates if elCentroids[pt].within(cell)])
        return containedElIds

    def _form_elsets_elType_(self, fids, nodes, elConn_eTypeSubSet, prefix='grain.'):
        centroids_coords, elCentroids, elCentroidsTree = self.get_elCentroids_singleElType(nodes, elConn_eTypeSubSet)
        elsets = {}
        for fid in fids:
            cell = self.gtess.L0.pxtal.geoms[fid-1]
            elsets[prefix+str(fid)] = self._form_elset_(cell, elCentroids, elCentroidsTree)
        return elsets

    def form_elsets_elType(self):
        fids = np.arange(1, len(self.gtess.L0.pxtal.geoms)+1)
        self.elsets_eltype = {}
        for eltype in self.availableElTypes:
            print(f"Working with element type: {eltype}")
            elConn_eTypeSubSet = self.elConn[eltype]
            self.elsets_eltype[eltype] = self._form_elsets_elType_(fids, self.nodes, elConn_eTypeSubSet, prefix='grain.')

    def get_elCentroids(self):
        return elemOps.get_elCentroids_2d(self.nodes, self.elConn, self.availableElTypes)
    
    def find_elID_ranges(self):
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
        firstKey = list(self.elsets_eltype.keys())[0]
        gblines_by_grain = {}
        for grain_name in self.elsets_eltype[firstKey].keys():
            gblines_by_grain[grain_name] = self.extract_gblines_for_grain(grain_name)
        return gblines_by_grain
    
    def extract_gbnodes_grains(self):
        nodeIDs = {gname: np.unique(gblines) for gname, gblines in self.extract_gblines_grains().items()}
        return nodeIDs
    
    def extract_gbnodeCoords_grains(self):
        nodeIDs= self.extract_gbnodes_grains()
        return {gname: self.nodes[nodeIDs] for gname, nodeIDs in nodeIDs.items()}

    def extract_gbCoords_ckdTrees(self, preserve_ZDim=False):
        gbCoords = self.extract_gbnodeCoords_grains()
        from scipy.spatial import cKDTree
        gbCoords_ckdTrees = {gname: cKDTree(coords[:, :2] if not preserve_ZDim else coords) for gname, coords in gbCoords.items()}
        return gbCoords_ckdTrees
    
    def calc_element_qualities(self, ar=True, saa=False, throw=False):
        elQual = {}
        if ar:
            elQual['ar'] = elemOps.compute_elementQuality_AR_2d(self.nodes, self.elConn)
        if saa: 
            self.elQual = elQual
        if throw:
            return elQual

    def find_elIDs_by_quality(self, **kwargs):
        return elemOps.find_elIDs_by_quality(**kwargs)

    def plot_elements_geometric_grain(self, grain_name, **kwargs):
        meshviz.plot_elements_geometric_grain(grain_name, **kwargs)

    def plot_elements_by_elIDs(self, element_ids, **kwargs):
        meshviz.plot_elements_by_elIDs(element_ids, **kwargs)
    
    def plot_elements_geometric_grains(self, **kwargs):
        meshviz.plot_elements_geometric_grains(**kwargs)

    def extract_gb_elements_for_grain(self, grain_name):
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
        gb_elements_grain = self.extract_gb_elements_for_grain(grain_name)
        meshviz.see_gbElements_grains(gb_elements_grain, **kwargs)

    def build_global_element_numbering(self):
        glb_elm_num = elemOps.build_global_element_numbering(self.elConn, elID_ranges=self.elID_ranges)
        return glb_elm_num
    
    def find_el_neigh(self, element_ids, n_order=1, eltype=None, include_self=False):
        return elemOps.find_el_neigh(element_ids, self.elConn, n_order=n_order, 
                                    eltype=eltype, include_self=include_self)

    def find_nthOrderNeigh(self, n_order, el_subset, include_self=False):
        return elemOps.find_nthOrderNeigh(n_order, el_subset, self.availableElTypes, self.elConn, include_self=include_self)

    def resolve_eltypes(self, grain_name, grainElements, grainCoordinates, eltypes=None):
        return elemOps.resolve_eltypes(
            grain_name, grainElements, grainCoordinates, eltypes=eltypes
        )

    def build_element_ids_by_band(self, grain_name, bands, eltypes, grainElements, nearest_dist_by_type):
        return elemOps.build_element_ids_by_band(
            grain_name, bands, eltypes, grainElements, nearest_dist_by_type
        )

    def pick_contrasting_colours_from_cmap(self, n_colours, cmap_name='nipy_spectral'):
        return meshviz.pick_contrasting_colours_from_cmap(n_colours, cmap_name=cmap_name)

    def resolve_band_colours(self, bands, band_colours=None, auto_cmap='nipy_spectral'):
        return meshviz.resolve_band_colours(
            bands, band_colours=band_colours, auto_cmap=auto_cmap
        )

    def resolve_plot_eltype(self, eltypes, plot_eltype=None):
        return meshviz.resolve_plot_eltype(eltypes, plot_eltype=plot_eltype)

    def plot_band_elements(self, element_ids_by_band, bands, plot_eltype, gbcoords,
                           gblines_by_grain=None, colours_to_use=None,
                           title='Selected elements by band', band_facecolors=False):
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
    Conformal 2D mesher using the raw gmsh API (no pygmsh dependency).

    Designed to work directly with a dict of Shapely polygons (flat_cells)
    and an optional grain-id mapping (gid_map).  This class duplicates
    analytical / post-processing methods from confMesh2d so that pygmsh can
    be phased out without breaking existing workflows.

    Import
    ------
    from upxo.meshing.conformal_mesher2d import confMesh2dGMSH as cm2dg

    Typical workflow
    ----------------
    m = confMesh2dGMSH()
    m.femesh_gmsh(flat_cells, gid_map,
                  mesh_size_gb=0.75, mesh_size_bulk=4.5,
                  mesh_algo=8, mesh_order=1,
                  recombine_to_quads=True)
    m.form_elsets_gmsh()
    m.build_boundary_nsets()
    m.build_grain_nsets()
    m.build_gb_nset()
    """

    ValidElTypesOptions = ('triangle', 'quad')

    __slots__ = (
        # geometry / topology inputs
        'gtess', 'fids', 'gid_map',
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
        # node sets
        'nsets',
        # pyvista grid (kept for optional pyvista workflows)
        'grid',
        # export record
        '_exported',
    )

    def __init__(self):
        self.gtess = None
        self.fids = None
        self.gid_map = None
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
        self.nsets = {}
        self.grid = None
        self._exported = []

    # ------------------------------------------------------------------
    # Main meshing entry point
    # ------------------------------------------------------------------

    def femesh_gmsh(self, flat_cells, gid_map=None,
                    mesh_size_gb=1.0, mesh_size_bulk=4.0,
                    mesh_algo=6, mesh_order=1,
                    recombine_to_quads=False,
                    out_dir=None, basename='gs_mesh', formats=None):
        """
        Full gmsh meshing pipeline.

        Parameters
        ----------
        flat_cells : dict
            {flat_id: shapely.geometry.Polygon}  — grain geometries.
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
        """
        import gmsh
        import os
        self.elementShape = 'quad' if recombine_to_quads else 'tri'
        self.elementOrder = mesh_order
        self.meshingAlgorithmID = mesh_algo
        self.recombine_to_quads = recombine_to_quads
        self.gid_map = gid_map if gid_map is not None else {k: k for k in flat_cells}

        gmsh.initialize()
        gmsh.model.add("upxo_confmesh")
        gmsh.option.setNumber("General.Terminal", 0)

        self._build_geometry(flat_cells, mesh_size_gb, gmsh)
        self._assign_physical_groups(flat_cells, gmsh)
        self._set_mesh_options_and_generate(mesh_size_gb, mesh_size_bulk,
                                            mesh_algo, mesh_order,
                                            recombine_to_quads, gmsh)
        self._extract_nodes_and_elements(gmsh)
        self._extract_gblines(gmsh)

        self._exported = []
        if out_dir and formats:
            os.makedirs(out_dir, exist_ok=True)
            for fmt in formats:
                path = os.path.join(out_dir, f'{basename}.{fmt}')
                gmsh.write(path)
                self._exported.append(path)

        gmsh.finalize()

        self._detect_available_eltypes()

    # ------------------------------------------------------------------
    # Geometry building
    # ------------------------------------------------------------------

    def _build_geometry(self, flat_cells, mesh_size_gb, gmsh):
        """Register points and build surfaces for every grain polygon.

        Island grains (polygons completely contained within another grain) are
        registered as *holes* in the parent grain's Gmsh plane surface via a
        three-pass approach:
          Pass 1 — build all shared points + curve loops.
          Pass 2 — detect containment (direct parent per island grain) using
                   a Shapely STRtree for performance.
          Pass 3 — create plane surfaces; island curve loops are passed as
                   additional arguments to addPlaneSurface so Gmsh carves
                   them out of the parent surface instead of overlapping.
        """
        from collections import defaultdict
        from shapely.strtree import STRtree

        self.point_registry = {}
        self.surface_tags = {}

        round_digits = 10

        def _get_or_add_point(x, y):
            key = (round(x, round_digits), round(y, round_digits))
            if key not in self.point_registry:
                tag = gmsh.model.geo.addPoint(x, y, 0.0, mesh_size_gb)
                self.point_registry[key] = tag
            return self.point_registry[key]

        flat_id_list = list(flat_cells.keys())
        polygon_list  = [flat_cells[fid] for fid in flat_id_list]

        # ── Pass 1: register all shared points and build curve loops ─────────
        loop_tags: dict = {}
        for flat_id, polygon in flat_cells.items():
            coords   = list(polygon.exterior.coords)[:-1]
            pt_tags  = [_get_or_add_point(x, y) for x, y in coords]
            n        = len(pt_tags)
            line_tags = [
                gmsh.model.geo.addLine(pt_tags[i], pt_tags[(i + 1) % n])
                for i in range(n)
            ]
            loop_tags[flat_id] = gmsh.model.geo.addCurveLoop(line_tags)

        # ── Pass 2: detect island grains — find direct parent for each ───────
        # For each polygon, find which OTHER polygons contain it (i.e., it is
        # an island/inclusion inside them).  We do a bounding-box pre-filter
        # via STRtree then an explicit .contains() check so the result is
        # version-independent and numerically robust after Taubin smoothing.
        outer_to_holes: dict = defaultdict(list)
        tree = STRtree(polygon_list)
        for i, fid_inner in enumerate(flat_id_list):
            # tree.query without predicate returns bounding-box candidates only
            candidates = tree.query(polygon_list[i])
            container_indices = [
                j for j in candidates
                if j != i and polygon_list[j].contains(polygon_list[i])
            ]
            if container_indices:
                # Direct parent = smallest-area container (handles nesting)
                direct_parent_idx = min(container_indices,
                                        key=lambda j: polygon_list[j].area)
                outer_to_holes[flat_id_list[direct_parent_idx]].append(
                    loop_tags[fid_inner]
                )

        # ── Pass 3: create plane surfaces; holes carved from parent ──────────
        for flat_id in flat_cells:
            outer_loop = loop_tags[flat_id]
            hole_loops  = outer_to_holes.get(flat_id, [])
            surf_tag    = gmsh.model.geo.addPlaneSurface([outer_loop] + hole_loops)
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
                                       recombine_to_quads, gmsh):
        """Apply size fields, algorithm, and generate the mesh."""
        # Distance field on all curves (grain boundaries)
        all_curves = [t for (_, t) in gmsh.model.getEntities(1)]
        dist_tag = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_tag, "CurvesList", all_curves)
        gmsh.model.mesh.field.setNumber(dist_tag, "Sampling", 100)

        thresh_tag = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_tag, "InField", dist_tag)
        gmsh.model.mesh.field.setNumber(thresh_tag, "SizeMin", mesh_size_gb)
        gmsh.model.mesh.field.setNumber(thresh_tag, "SizeMax", mesh_size_bulk)
        gmsh.model.mesh.field.setNumber(thresh_tag, "DistMin", mesh_size_gb)
        gmsh.model.mesh.field.setNumber(thresh_tag, "DistMax", mesh_size_bulk * 2)

        gmsh.model.mesh.field.setAsBackgroundMesh(thresh_tag)

        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", mesh_algo)
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_order)

        if recombine_to_quads:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

        gmsh.model.mesh.generate(2)

        if mesh_order > 1:
            gmsh.model.mesh.setOrder(mesh_order)

    # ------------------------------------------------------------------
    # Mesh extraction
    # ------------------------------------------------------------------

    def _extract_nodes_and_elements(self, gmsh):
        """Populate self.nodes and self.elConn from gmsh mesh data."""
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = node_coords.reshape(-1, 3)
        # Build zero-based node array indexed by tag
        max_tag = int(node_tags.max())
        node_array = np.full((max_tag + 1, 3), np.nan)
        for tag, coord in zip(node_tags, coords):
            node_array[int(tag)] = coord
        self.nodes = node_array

        tri_conn = []
        quad_conn = []

        # element type codes: 2=3-node tri, 3=4-node quad,
        #                      9=6-node tri2, 16=8-node quad2
        TRI_TYPES = {2, 9}
        QUAD_TYPES = {3, 16}

        for dim, tag in gmsh.model.getEntities(2):
            el_types, el_tags, el_node_tags = gmsh.model.mesh.getElements(dim, tag)
            for etype, etags, entags in zip(el_types, el_tags, el_node_tags):
                props = gmsh.model.mesh.getElementProperties(etype)
                n_nodes = props[3]
                connectivity = entags.reshape(-1, n_nodes).astype(int)
                if etype in TRI_TYPES:
                    tri_conn.append(connectivity)
                elif etype in QUAD_TYPES:
                    quad_conn.append(connectivity)

        self.elConn = {}
        if tri_conn:
            self.elConn['triangle'] = np.vstack(tri_conn)
        if quad_conn:
            self.elConn['quad'] = np.vstack(quad_conn)

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
        self.availableElTypes = [et for et in ('triangle', 'quad') if et in self.elConn]
        self.availableElTypeID = {et: i for i, et in enumerate(self.availableElTypes)}
        self.availableFeatures = self.availableElTypes.copy()

    # ------------------------------------------------------------------
    # Element sets
    # ------------------------------------------------------------------

    def get_elCentroids_singleElType(self, nodes, elConn):
        from shapely.geometry import Point
        from shapely.strtree import STRtree
        elCentroids_coords = nodes[elConn].mean(axis=1)[:, :2]
        elCentroids_shPoints = [Point(x, y) for x, y in elCentroids_coords]
        elCentroids_Tree = STRtree(elCentroids_shPoints)
        return elCentroids_coords, elCentroids_shPoints, elCentroids_Tree

    def _form_elset_(self, cell, elCentroids, elCentroidsTree):
        candidates = elCentroidsTree.query(cell)
        containedElIds = np.array([int(pt) for pt in candidates if elCentroids[pt].within(cell)])
        return containedElIds

    def _form_elsets_elType_(self, flat_cells, nodes, elConn_eTypeSubSet, prefix='grain.'):
        centroids_coords, elCentroids, elCentroidsTree = self.get_elCentroids_singleElType(nodes, elConn_eTypeSubSet)
        elsets = {}
        for flat_id, polygon in flat_cells.items():
            orig_gid = self.gid_map.get(flat_id, flat_id)
            key = f"{prefix}{orig_gid}"
            elsets[key] = self._form_elset_(polygon, elCentroids, elCentroidsTree)
        return elsets

    def form_elsets_gmsh(self, flat_cells, prefix='grain.'):
        """
        Build per-grain element sets from flat_cells polygons.

        Parameters
        ----------
        flat_cells : dict
            {flat_id: shapely.geometry.Polygon} — same dict used in femesh_gmsh.
        prefix : str
            Prefix for element set names (default 'grain.').
        """
        print(f"Available element types: {self.availableElTypes}")
        self.elsets_eltype = {}
        for eltype in self.availableElTypes:
            print(f"  Forming elsets for element type: {eltype}")
            self.elsets_eltype[eltype] = self._form_elsets_elType_(
                flat_cells, self.nodes, self.elConn[eltype], prefix=prefix
            )

        self.find_elID_ranges()

        if len(self.availableElTypes) == 1:
            self.elsets = self.elsets_eltype[self.availableElTypes[0]]
            return

        # Merge multiple element types with offset global numbering
        self.elsets = {name: elset.tolist()
                       for name, elset in self.elsets_eltype[self.availableElTypes[0]].items()}
        nel0 = self.elID_ranges[self.availableElTypes[0]][1]
        for name, elset in self.elsets_eltype[self.availableElTypes[1]].items():
            elset_offset = elset + nel0
            self.elsets.setdefault(name, [])
            self.elsets[name].extend(elset_offset.tolist())
            self.elsets[name] = np.array(self.elsets[name], dtype=np.int32)

    def find_elID_ranges(self):
        self.elID_ranges = {}
        for i, eltype in enumerate(self.availableElTypes):
            n = self.elConn[eltype].shape[0]
            if i == 0:
                self.elID_ranges[eltype] = [1, n]
            else:
                prev = self.elID_ranges[self.availableElTypes[i - 1]][1]
                self.elID_ranges[eltype] = [prev + 1, prev + n]

    def get_elCentroids(self):
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
        first_key = self.availableElTypes[0]
        gblines_by_grain = {}
        for grain_name in self.elsets_eltype[first_key]:
            gblines_by_grain[grain_name] = self.extract_gblines_for_grain(grain_name)
        return gblines_by_grain

    def extract_gbnodes_grains(self):
        return {gname: np.unique(gblines)
                for gname, gblines in self.extract_gblines_grains().items()}

    def extract_gbnodeCoords_grains(self):
        nodeIDs = self.extract_gbnodes_grains()
        return {gname: self.nodes[ids] for gname, ids in nodeIDs.items()}

    def extract_gbCoords_ckdTrees(self, preserve_ZDim=False):
        from scipy.spatial import cKDTree
        gbCoords = self.extract_gbnodeCoords_grains()
        return {gname: cKDTree(coords[:, :2] if not preserve_ZDim else coords)
                for gname, coords in gbCoords.items()}

    def extract_gb_elements_for_grain(self, grain_name):
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
        from upxo.viz.meshviz import see_femesh
        return see_femesh(*args, **kwargs)

    def see_gbElements_grains(self, grain_name, **kwargs):
        gb_elements_grain = self.extract_gb_elements_for_grain(grain_name)
        meshviz.see_gbElements_grains(gb_elements_grain, **kwargs)

    # ------------------------------------------------------------------
    # Element quality
    # ------------------------------------------------------------------

    def calc_element_qualities(self, ar=True, saa=False, throw=False):
        elQual = {}
        if ar:
            elQual['ar'] = elemOps.compute_elementQuality_AR_2d(self.nodes, self.elConn)
        if saa:
            self.elQual = elQual
        if throw:
            return elQual

    def find_elIDs_by_quality(self, **kwargs):
        return elemOps.find_elIDs_by_quality(**kwargs)

    # ------------------------------------------------------------------
    # Neighbour / band analysis
    # ------------------------------------------------------------------

    def find_el_neigh(self, element_ids, n_order=1, eltype=None, include_self=False):
        return elemOps.find_el_neigh(element_ids, self.elConn,
                                     n_order=n_order, eltype=eltype,
                                     include_self=include_self)

    def find_nthOrderNeigh(self, n_order, el_subset, include_self=False):
        return elemOps.find_nthOrderNeigh(n_order, el_subset,
                                          self.availableElTypes, self.elConn,
                                          include_self=include_self)

    def build_global_element_numbering(self):
        return elemOps.build_global_element_numbering(self.elConn,
                                                      elID_ranges=self.elID_ranges)

    def resolve_eltypes(self, grain_name, grainElements, grainCoordinates, eltypes=None):
        return elemOps.resolve_eltypes(grain_name, grainElements,
                                       grainCoordinates, eltypes=eltypes)

    def build_element_ids_by_band(self, grain_name, bands, eltypes,
                                  grainElements, nearest_dist_by_type):
        return elemOps.build_element_ids_by_band(grain_name, bands, eltypes,
                                                  grainElements, nearest_dist_by_type)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def pick_contrasting_colours_from_cmap(self, n_colours, cmap_name='nipy_spectral'):
        return meshviz.pick_contrasting_colours_from_cmap(n_colours, cmap_name=cmap_name)

    def resolve_band_colours(self, bands, band_colours=None, auto_cmap='nipy_spectral'):
        return meshviz.resolve_band_colours(bands, band_colours=band_colours,
                                            auto_cmap=auto_cmap)

    def resolve_plot_eltype(self, eltypes, plot_eltype=None):
        return meshviz.resolve_plot_eltype(eltypes, plot_eltype=plot_eltype)

    def plot_elements_geometric_grain(self, grain_name, **kwargs):
        meshviz.plot_elements_geometric_grain(grain_name, **kwargs)

    def plot_elements_by_elIDs(self, element_ids, **kwargs):
        meshviz.plot_elements_by_elIDs(element_ids, **kwargs)

    def plot_elements_geometric_grains(self, **kwargs):
        meshviz.plot_elements_geometric_grains(**kwargs)

    def plot_band_elements(self, element_ids_by_band, bands, plot_eltype, gbcoords,
                           gblines_by_grain=None, colours_to_use=None,
                           title='Selected elements by band', band_facecolors=False):
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
