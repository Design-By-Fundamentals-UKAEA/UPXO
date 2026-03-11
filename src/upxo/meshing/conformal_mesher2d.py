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
    
