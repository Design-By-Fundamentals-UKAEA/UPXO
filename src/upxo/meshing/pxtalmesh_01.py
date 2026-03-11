# //////////////////////////////////////////////////////////////////////////////
# Script information for the file.
__name__ = "UPXO.femesh"
__authors__ = ["Vaasu Anandatheertha"]
__lead_developer__ = ["Vaasu Anandatheertha"]
__emails__ = ["vaasu.anandatheertha@ukaea.uk", ]
__version__ = ["0.1:@ upto.171122",
               ]
__license__ = "GPL v3"
# //////////////////////////////////////////////////////////////////////////////
import meshio
import pygmsh
import numpy as np
import pyvista as pv
# //////////////////////////////////////////////////////////////////////////////
class geo_pxtal_mesh():
    '''
    ##########################################################
    GENERAL NOTES AND REFERENCES: PYGMSH RELATED
    ##########################################################
    *
    ##########################################################
    GENERAL NOTES AND REFERENCES: GMSH RELATED
    ##########################################################
    TO KNOW MORE ON THESE AND IMPLEMENT THEM IN UPXO:
        # TODO: ~~ Commands to be ported form GMSH API ~~

        1. How do I set the Mesh.RecombinationAlgirithm value [a]
           [a] See the gmsh pdf manual @ page 197

        2. Mesh.RefineSteps
            Number of refinment steps in MeshAdapt-based 2D algirithms [a]
            Default: 10 [a]
            [a]: See the gmsh pdf manual @ page 198

        3. Mesh.SubdivisionAlgorithm
            [a]: See the gmsh pdf manual @ page 200


    algorithm = 1: MeshAdapt

    algorithm = 2: Automatic [DEFAULT]

    algorithm = 3: -----

    algorithm = 4: -----

    algorithm = 5: Delaunay
                   Can handle complex mesh size fields better - in particular size fields
                   with large element size gradients [a]
                   [a] https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorials/python/t10.py

    algorithm = 6: Frontal-Delaunay: 2D
                   Usually leads to highest quality meshes [a]
                   Sometimes just called as Frontal algorithm [b]
                   [a] https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorials/python/t10.py
                   [b] See the gmsh pdf manual @ page 145

    algorithm = 7: There is a mention here [a], but no explanation.
                   Referred to as BAMG [b]
                   [a] https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorials/python/t17.py
                   [b] See the gmsh pdf manual @ page 187

    algorithm = 8: basically, an experimental Frontal-Delaunay for quads: 2D. [a]
                   For tri, however, right triangles are created almost everywhere
                   It uses the DelQuad algorithm [b]

                   NOTE: IMPORTANT:
                       To get all quads, recombination algorithm has to be changed. [a]
                       gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2) # or 3

                   [a] https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorials/python/t11.py
                   [b] See the gmsh pdf manual @ page 143

    NOTE - 1: @ GMSH
        REF https://www.manpagez.com/info/gmsh/gmsh-2.4.0/gmsh_76.php
        Mesh.RefineSteps
        Number of refinement steps in the MeshAdapt-based 2D algorithms
        Default value: 10
        Saved in: General.OptionsFileName

    NOTE - 2: @ GMSH
        REF https://www.manpagez.com/info/gmsh/gmsh-2.4.0/gmsh_76.php
        Mesh.RandomFactor
        Random factor used in the 2D meshing algorithm (should be increased if RandomFactor * size(triangle)/size(model) approaches machine accuracy)
        Default value: 1e-09
        Saved in: General.OptionsFileName

    REFERENCES:
        https://www.manpagez.com/info/gmsh/gmsh-2.4.0/gmsh_76.php
        https://docs.salome-platform.org/latest/gui/GMSHPLUGIN/gmsh_2d_3d_hypo_page.html
        http://diposit.ub.edu/dspace/bitstream/2445/49313/1/GMSH-Guide_for_mesh_generation.pdf
    ##########################################################
    GENERAL NOTES AND REFERENCES: DATA STRUCTURE
    ##########################################################
    *
    '''
    __slots__ = ('pxtal', # Poly-xtal object
                 'mesher', # Meshing tool used
                 'elsize_global', # Global element size
                 'elsize_xtals', # Element sizes for each xtal
                 'mesh', # mesh object(s): could be from pygmsh or others
                 'uxtals', # UPXO xtal objects
                 'quadv', # Quad vertices IDs
                 'quadid', # Quad element IDs
                 'triv', # Tri vertices IDs
                 'triid', # Tri element IDs
                 'nfeatures',  # NUmber of mesh features
                 'pointidxy', # Point IDs and xy: np.ndarray: size(Np,3)
                 'modelo', # model object(s): could be from pygmsh or others: list
                 'abq_ges_pxtal', # abaqus global element size for the entire pxtal: float
                 'abq_ges_xtals', # abaqus global element sizes of each xtals: list: list
                 'abq_edge_seed_global' # abaqus global edge seeding specification: dict
                 'abq_sdv_001', # abaqus solution dependent variable no. 001
                 )
    # //////////////////////////////////////////////////////////////////////////////
    def __init__(self,
                 mesher = 'pygmsh',
                 pxtal = None,
                 level = 0,
                 elshape = 'quad',
                 elorder = 1,
                 algorithm = 8,
                 elsize_global = [0.02, 0.02, 0.02], # Min, mean, max mesh size
                 optimize = True,
                 sta = True,
                 wtfs = True,
                 ff = ['vtk', 'inp'],
                 throw = False
                 ):
        # ----------------------------------
        self.pxtal = pxtal
        self.mesher = mesher
        # ----------------------------------
        # Number of featues
        self.nfeatures = {'vertex': None,
                          'line': None,
                          'triangle': None,
                          'quadrilateral': None}
        # ----------------------------------
        if pxtal is not None:
            xtals = pxtal.L0.xtals
        # ----------------------------------
        # TODO: REFACTOR INTO A DEFINTIION
        self.elsize_xtals = [elsize_global[0] for _ in range(self.pxtal.L0.xtals_n)]
        # ----------------------------------
        print(f'meshing pxtal.get_L0_ng()')
        _mesh = [f'Element shape: {elshape}',
                 f'Algorithm: {algorithm}',
                 f'Element order: {elorder}',
                 self.mesh_pygmsh(elshape = elshape,
                                  algorithm = algorithm,
                                  elorder = elorder,
                                  ),
                 ]
        self.mesh = _mesh
        # ----------------------------------
        if mesher == 'abaqus':
            self.mesh_abaqus()
        # ----------------------------------
    # //////////////////////////////////////////////////////////////////////////////
    def mesh_pygmsh(self,
                    elshape = 'quad',
                    elorder = 1,
                    algorithm = 8,
                    elsize_xtals: list = [],
                    sta = True,
                    wtf = True,
                    throw = False,
                    ):
        # ---------------------------------------------------
        import pygmsh
        # ---------------------------------------------------
        # Extract vertices data for each upxo xtal object
        if self.pxtal.L0.vt_base_tool == 'shapely':
            x, y, _ = self.pxtal.extract_shapely_coords(shapely_grains_list=self.pxtal.L0.xtals,
                                                        coord_of='L0_xtal_vertices_pbjp',
                                                        save_to_attribute=False,
                                                        make_unique=False,
                                                        throw=True
                                                        )
            xy = [np.c_[np.array(_x[:-1]),
                        np.array(_y[:-1])] for _x, _y in zip(x, y)]
        # ---------------------------------------------------
        # Initialize empty geometry using the build in kernel in GMSH
        geometry = pygmsh.geo.Geometry()
        # ---------------------------------------------------
        # Fetch model we would like to add data to
        model = geometry.__enter__()
        # ---------------------------------------------------
        # Make pygmsh polygon objects from each upxo xtal vertices
        elsizes = self.elsize_xtals
        xtals = [model.add_polygon(_xy,
                                   mesh_size=elsizes[i])
                 for i, _xy in enumerate(xy)]
        #---------------------------------------------------
        model.synchronize()
        #---------------------------------------------------
        # Recombine triangles if quad elements are needed
        if elshape == 'quad':
            geometry.set_recombined_surfaces([xtal.surface for xtal in xtals])
        #---------------------------------------------------
        # Generate the mesh
        dim = 2
        mesh = geometry.generate_mesh(dim = dim, order = elorder, algorithm = algorithm)
        #---------------------------------------------------
        # Display message upon conclusion
        print(''.join([40*'-', '\n',
                       f'PXTAL with {self.pxtal.L0.xtals_n} xtals: meshing success \n',
                       f'     Dimensionality = {dim} \n',
                       f'     Element order = {elorder} \n',
                       f'     Algorithm = {algorithm} \n',
                       40*'-',
                       ]
                      )
              )
        # ---------------------------------------------------
        return mesh
    # //////////////////////////////////////////////////////////////////////////////
    def assign_pygmsh_elsets(self):
        '''
        Assign element sets for each xtal in the pxtal
        '''
        print('Assigning element sets')
        pass
    #//////////////////////////////////////////////////////////////////////////////
    def map_pygmsh_cell_colours(self,
                                mesh = None,
                                cmap_name = 'random'
                                ):
        '''
        Assign colours to each xtal
        '''
        if cmap_name == 'random':
            mesh.cell_data['color'] = [list(np.random.randint(0, 255, 3)) for _ in range(self.pxtal.L0.xtals_n)]
        # ----------------------------------
        print('Colours to each XTALs assigned')
        # ----------------------------------
        return mesh
    #//////////////////////////////////////////////////////////////////////////////
    def get_pygmsh_qt_elements(self):
        '''
        Extract the triangular and quadrilateral finite elements
        '''
        mesh_cells = self.mesh[3].cells
        tel, qel = mesh_cells[1].data, mesh_cells[2].data
        # ----------------------------------
        print('TRI and QUAD elements extracted')
        # -------------------------------
        return tel, qel
    #//////////////////////////////////////////////////////////////////////////////
    def get_pygmsh_mesh_feature_count(self):
        '''
        Extract the count of the number of mesh features
        '''

        for idx, feature_set in enumerate(self.mesh[-1].cells):
            feature_type = feature_set.type
            if feature_type in self.nfeatures.keys():
                self.nfeatures[feature_type] = len(feature_set)
        print(f'Features and feature counts are: {self.nfeatures}')
    #//////////////////////////////////////////////////////////////////////////////
    def get_pygmsh_tq_subsets(self,
                              grid = None,
                              ):
        '''
        Get triangular elements and quadrilateral elements as subsets
        '''
        ELEMENT_ID_TRI = 3
        tri_cells = [i for i in range(grid.n_cells) if grid.cell_n_points(i) == ELEMENT_ID_TRI]
        grid_subset_TRI = grid.extract_cells(tri_cells)
        # -------------------------------
        # extract only quadrilateral elements
        # TO DO
        # -------------------------------
        print('Element type based grid subsets extracted')
        #-------------------------------
        return grid_subset_TRI
    #//////////////////////////////////////////////////////////////////////////////
    def get_pygmsh_cells_along_line(self, line_points = [[0.0, 0.0, 0.0],
                                                         [0.04, 0.04, 0.0],
                                                         ]
                                    ):
        index = grid.find_cells_along_line(line_points[0], line_points[1])
        subset = grid.extract_cells(index)
        print('Mesh cells identified along given line segments')
        return subset
    #//////////////////////////////////////////////////////////////////////////////
    def getSizeOfNestedList(self,
                            listOfElem):
        '''
        Get number of elements in a nested list
        Credit and ref: see below link
        https://thispointer.com/python-get-number-of-elements-in-a-list-lists-of-lists-or-nested-list/
        '''
        count = 0
        # Iterate over the list
        for elem in listOfElem:
            # Check if type of element is list
            if type(elem) == list:
                # Again call this function to get the size of this element
                count += self.getSizeOfNestedList(elem)
            else:
                count += 1
        return count
    #//////////////////////////////////////////////////////////////////////////////
    def mesh_pygmsh_optimize(self,
                             grid = None,
                             tool = 'pygmsh',
                             ):
        '''
        grid: The mesh read from vtk file
        tool: options:
                    1. 'pygmsh'
                    2. 'optimesh'
        '''
        grid_opt = pygmsh.optimize(grid, method = '')
    #//////////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////////
    def assess_pygmsh(self,
                      grid = None, # Input pyvista grid data
                      rfa = True, # Read From Attribute
                      rff = False, # Read From File
                      filename = 'pxtal_mesh.vtk',
                      mesh_quality_measures = ['aspect_ratio','min_angle'],
                      elshape = None,
                      elorder = None,
                      algorithm = None
                      ):
        '''
        mesh_quality_measure options:
            1. min_angle
            2. max_angle
            3. jacobian
            4. aspect_ratio
            5. aspect_frobenius
            6. distortion
        References:

        '''
        #---------------------------------------------------
        import pyvista as pv
        import pandas as pd
        #---------------------------------------------------
        mqm_data = [[f'Element shape: {elshape}',
                     f'Element order: {elorder}',
                     f'Meshing algorithm: {algorithm}',
                     f'Mesh Quality Measure (mqm): {mqm}',
                     grid.compute_cell_quality(quality_measure = mqm)] for mqm in mesh_quality_measures]
        #---------------------------------------------------
        mqmd_list = []
        for i, mqmd in enumerate(mqm_data):
            mqmd_ = mqmd[4]
            if mesh_quality_measures[i] in ('aspect_ratio', 'min_angle', 'max_angle', 'area'):
                __ = np.delete(mqmd_.cell_data['CellQuality'], mqmd_.cell_data['CellQuality'] == -1)
                if len(__) > 10:
                    __ = np.delete(__, __ == __.max())
            elif mesh_quality_measures[i] in ('skew'):
                # This might need a sepertate treatment.
                # Retaining same codes for other measures for now.
                __ = np.delete(mqmd_.cell_data['CellQuality'], mqmd_.cell_data['CellQuality'] == -1)
                if len(__) > 10:
                    __ = np.delete(__, __ == __.max())
            elif mesh_quality_measures[i] in ('GeuzaineRemacle', 'GR', 'gamma'):
                '''
                DEFINED BY: Christophe Geuzaine and Jean-François Remacle
                            Creators of GMSH

                MESH QUALITY PARAMETER:
                    gamma: between 0 and 1.
                        gamma = 0 for a flat triangle
                        gamma = 1 for an equilateral triangle

                PARAMETER EXPRESSION:
                    gamma = 4*(sin(a)*sin(b)*sin(c))/(sin(a)+sin(b)+sin(c))

                REF1:
                    Jan Nytra, Libor Cˇerma´k and Miroslav Jı´cha
                      Applications of the discontinous Galerkin method to propagating..
                          .. acoustic wave problems
                      Advances in Mechanical Engineering. 2017, Vol. 9(6) 1–14
                      https://journals.sagepub.com/doi/full/10.1177/1687814017703631
                      https://www.researchgate.net/publication/317633186_Applications_of_the_discontinuous_Galerkin_method_to_propagating_acoustic_wave_problems

                '''
                # TODO: To be coded for speed using numpy vectorised operations only or using cython
                pass
            mqmd_list.append(np.array(__))
        #---------------------------------------------------
        mqm_dataframe = pd.concat([pd.Series(_, name = mesh_quality_measures[i]) for i, _ in enumerate(mqmd_list)],
                                  axis = 1)
        #---------------------------------------------------
        #qual_field_cleanarray =
        #qual_field_cleanarray = np.delete(qual_field_cleanarray,
        #                                  qual_field_cleanarray == qual_field_cleanarray.max()
        #                                  )
        #---------------------------------------------------
        return mqm_data, mqm_dataframe
        #---------------------------------------------------
    #//////////////////////////////////////////////////////////////////////////////
    def read_pygmsh_mesh(self):
        pass
    #//////////////////////////////////////////////////////////////////////////////
    def write_pygmsh_mesh(self):
        pass
    #//////////////////////////////////////////////////////////////////////////////
    def mesh_abaqus(self):
        pass
    #//////////////////////////////////////////////////////////////////////////////
    def read_abaqus_mesh(self):
        pass
    #//////////////////////////////////////////////////////////////////////////////
    def write_abaqus_mesh(self,
                          inputmesh_format = 'pygmsh'
                          ):
        pass
    #//////////////////////////////////////////////////////////////////////////////
    def assess_abaqus_mesh(self):
        pass
    #//////////////////////////////////////////////////////////////////////////////
    def vis_pyvista(self,
                    data_to_vis = 'mesh > mesh > all',
                    rff = False, # Read from file
                    rfia = True, # Read from argument, i.e. input argument i.e. grid
                    filename = None,
                    grid  = None,
                    field = None,
                    clr_background = 'white',
                    wnd_size = (800, 800),
                    triad_show = True,
                    triad_par = {'line_width': 4,
                                 'ambient': 0.0,
                                 'x_color': 'red',
                                 'y_color': 'green',
                                 'z_color': 'blue',
                                 'cone_radius': 0.3,
                                 'shaft_length': 0.9,
                                 'tip_length': 0.1,
                                 'xlabel': 'X Axis',
                                 'ylabel': 'Y Axis',
                                 'zlabel': 'Z Axis',
                                 'label_size': (0.08, 0.08)
                                 },
                    xtal_vert_point_size = 10,
                    el_edge_width = 2,
                    el_point_size = 2,
                    mesh_opacity = 1.0,
                    mesh_display_style = 'wireframe',
                    mesh_qual_fields = None,
                    mesh_qual_field_vis_par = {'mesh_quality_measure': 'Quality Measure name',
                                               'cpos': 'xy',
                                               'scalars': 'CellQuality',
                                               'show_edges': False,
                                               'cmap': 'jet',
                                               'clim': [-1, 1],
                                               'below_color': 'white',
                                               'above_color': 'black',
                                               },
                    stat_distr_data = None,
                    stat_distr_data_par = {'mesh_quality_measure': 'Quality Measure name',
                                           'density': True,
                                           'figsize': (1.6, 1.6),
                                           'dpi': 200,
                                           'nbins': 100,
                                           'xlabel_text': 'use_from_data',
                                           'ylabel_text': 'Count', # stat_distr_data_par['ylabel_text']
                                           'xlabel_fontsize': 8,
                                           'ylabel_fontsize': 8,
                                           'hist_xlim': [1, 2.5],
                                           'hist_ylim': [0, 100]
                                           },
                    throw_hist = False
                    ):
        '''
        DESCRIPTION
        '''
        #----------------------------------------------
        import pyvista as pv
        #----------------------------------------------
        if data_to_vis == 'mesh > mesh > all':
            pv.set_plot_theme('document')
            #....................
            if rff:
                grid = pv.read(f'{filename}')
            if rfia:
                grid = grid
            #....................
            pv.global_theme.background = clr_background
            #....................
            plotter = pv.Plotter(window_size = wnd_size,
                                 border = True,
                                 line_smoothing = True,
                                 lighting = 'three_lights'
                                 )
            #....................
            if triad_show:
                marker = pv.create_axes_marker(line_width = triad_par['line_width'],
                                               ambient = triad_par['ambient'],
                                               x_color = triad_par['x_color'],
                                               y_color = triad_par['y_color'],
                                               z_color = triad_par['z_color'],
                                               cone_radius = triad_par['cone_radius'],
                                               shaft_length = triad_par['shaft_length'],
                                               tip_length = triad_par['tip_length'],
                                               xlabel = triad_par['xlabel'],
                                               ylabel = triad_par['ylabel'],
                                               zlabel = triad_par['zlabel'],
                                               label_size = triad_par['label_size'],
                                               )
                #....................
            _ = plotter.add_actor(marker)
            #....................
            # import numpy as np
            #_ = plotter.add_points(np.array([0,0,0]),
            #                        render_points_as_spheres = True,
            #                        point_size = 20, color = 'cyan')
            #....................
            # for xtal in xtals:
            #    _ = plotter.add_lines(add each of the pxtal edge objects)
            #....................
            _ = plotter.add_points(grid.points,
                                   render_points_as_spheres = False,
                                   point_size = el_point_size,
                                   color = 'cyan'
                                   )
            #....................
            #_ = plotter.add_bounding_box(line_width=2, color='black')
            #....................
            _ = plotter.add_mesh(grid,
                                 #color = 'black',
                                 show_edges = True,
                                 edge_color = 'black',
                                 line_width = el_edge_width,
                                 render_points_as_spheres = True,
                                 point_size = xtal_vert_point_size,
                                 style = mesh_display_style,
                                 opacity = mesh_opacity,
                                 )
            #....................
            plotter.view_xy()
            plotter.camera.zoom(1.0)
            plotter.show()
        #----------------------------------------------
        if data_to_vis == 'mesh > mesh > tri':
            'plot only tri elements'
            pass
        #----------------------------------------------
        if data_to_vis == 'mesh > mesh > quad':
            'plot only quad elements'
            pass
        #----------------------------------------------
        if data_to_vis == 'mesh > quality > field':
            for i, field in enumerate(mesh_qual_fields):
                field[4].plot(cpos = mesh_qual_field_vis_par['cpos'],
                              scalars = mesh_qual_field_vis_par['scalars'],
                              show_edges = mesh_qual_field_vis_par['show_edges'],
                              cmap = mesh_qual_field_vis_par['cmap'],
                              clim = mesh_qual_field_vis_par['clims'][i],
                              below_color = mesh_qual_field_vis_par['below_color'],
                              above_color = mesh_qual_field_vis_par['above_color'],
                              scalar_bar_args = {'title': mesh_qual_field_vis_par['mesh_quality_measures'][i],
                                                 },
                              )
        #----------------------------------------------
        if data_to_vis == 'mesh > quality > histogram':
            '''
            EXAMPLE CALL:

                # Calculate the number of bins to use in the histogram
                hist_nbins = allel_n/200
                if 0 <= hist_nbins <= 10:
                    hist_nbins = int(round(hist_nbins, -2))
                elif 10 < hist_nbins <= 100:
                    hist_nbins = 100
                else:
                    hist_nbins = int(round(hist_nbins, -2))
                if int(round(hist_nbins, -2)) == 0:
                    hist_nbins = 10


            # Get the statistical distribution of mesh quality from the cleanarray
            hist_data = pxtal_mesh.vis_pyvista(data_to_vis = 'mesh > quality > histogram',
                                               stat_distr_data = qual_field_cleanarray,
                                               stat_distr_data_par = {'mesh_quality_measure': mesh_quality_measure,
                                                                      'density': True,
                                                                      'figsize': (1.6, 1.6),
                                                                      'dpi': 200,
                                                                      'nbins': hist_nbins,
                                                                      'xlabel_text': 'use_from_data',
                                                                      'ylabel_text': 'Count', # stat_distr_data_par['ylabel_text']
                                                                      'xlabel_fontsize': 8,
                                                                      'ylabel_fontsize': 8,
                                                                      'hist_xlim': hist_xlim,
                                                                      'hist_ylim': hist_ylim
                                                                      },
                                               throw_hist = True
                                               )
            '''

            #print(stat_distr_data_par['mesh_quality_measure'])

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize = stat_distr_data_par['figsize'],
                             dpi = stat_distr_data_par['dpi']
                             )
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_facecolor("white")
            weights = np.ones_like(stat_distr_data) / len(stat_distr_data)
            print('@@@@@@@@@@@@@@@@@@@@@')
            print(stat_distr_data_par['nbins'])
            print('@@@@@@@@@@@@@@@@@@@@@')
            hist_data = ax.hist(stat_distr_data,
                                bins = stat_distr_data_par['nbins'],
                                facecolor = 'cyan',
                                edgecolor = 'black',
                                linewidth = 0.5,
                                weights = weights,
                                )
            print('@@@@@@@@@@@@@@@@@@@@@')
            print(stat_distr_data)
            print('@@@@@@@@@@@@@@@@@@@@@')
            # Set xlabel
            if stat_distr_data_par['xlabel_text'] == 'use_from_data':
                _xlabel_text = stat_distr_data_par['mesh_quality_measure']
            else:
                _xlabel_text = stat_distr_data_par['ylabel_text']
            ax.set_xlabel(_xlabel_text,
                          fontsize = stat_distr_data_par['xlabel_fontsize']
                          )
            # Set y-label
            ax.set_ylabel(stat_distr_data_par['ylabel_text'],
                          fontsize = stat_distr_data_par['ylabel_fontsize'])
            ax.tick_params(axis = 'x', labelsize = 7)
            ax.tick_params(axis = 'y', labelsize = 7)
            plt.xlim(stat_distr_data_par['hist_xlim'])
            if hist_data[0].max() >= 0.75:
                #plt.ylim(stat_distr_data_par['hist_ylim'])
                plt.ylim([0.0, 1.0])
            elif hist_data[0].max() >= 0.50 and hist_data[0].max() < 0.75:
                plt.ylim([0.0, 0.75])
            elif hist_data[0].max() >= 0.25 and hist_data[0].max() < 0.50:
                plt.ylim([0.0, 0.50])
            elif hist_data[0].max() >= 0.05 and hist_data[0].max() < 0.25:
                plt.ylim([0.0, 0.25])
            elif hist_data[0].max() >= 0.00 and hist_data[0].max() < 0.05:
                plt.ylim([0.0, 0.10])
            plt.show()
            if throw_hist:
                return hist_data
        #----------------------------------------------
    def vis_kde(self,
                data = None,
                datatype = 'pandas_df',
                df_names = None,
                clips = None,
                cumulative = False,
                band_widths = None,
                colors = None
                ):
        import seaborn as sns
        from matplotlib import pyplot as plt

        for i, (df_name, clip, bw, color) in enumerate(zip(df_names, clips, band_widths, colors)):
            fig, axes = plt.subplots(figsize = (5, 5), dpi = 100)

            sns.kdeplot(data = data[df_name],
                        cut = 0,
                        fill = True,
                        bw_adjust = bw,
                        color = color,
                        clip = clip,
                        cumulative = cumulative
                        ).set(title = f'mqm: {df_name} distribution')
            plt.show()
#//////////////////////////////////////////////////////////////////////////////
# =============================================================================
# elsize = 0.1
# #---------------------------------------------------
# # Initialize empty geometry using the build in kernel in GMSH
# geometry = pygmsh.geo.Geometry()
# # Fetch model we would like to add data to
# model = geometry.__enter__()
# GCore = model.add_polygon([[0.1, 0.1],
#                            [0.9, 0.1],
#                            [0.9, 0.9],
#                            [0.1, 0.9],],
#                           mesh_size = 0.20*elsize)
#
# field0 = geometry.add_boundary_layer(edges_list = [GCore.curves[0]],
#                                      lcmin = 0.2,
#                                      lcmax = 0.2,
#                                      distmin = 0.1,
#                                      distmax = 0.1,
#                                     )
# field1 = geom.add_boundary_layer(edges_list = [GCore.curves[1]],
#                                  lcmin = 0.2,
#                                  lcmax = 0.5,
#                                  distmin = 0.0,
#                                  distmax = 0.5,
#                                 )
# # geom.set_recombined_surfaces([poly.surface])
# geom.set_background_mesh([field0, field1], operator = "Min")
#
# GBZone = model.add_polygon([[0.0, 0.0],
#                             [1.0, 0.0],
#                             [1.0, 1.0],
#                             [0.0, 1.0],],
#                            holes = [GCore], mesh_size = 0.50*elsize)
# polygon_ = model.add_polygon([[1.0, 0.0],
#                               [2.0, 0.5],
#                               [1.0, 1.0],],
#                              mesh_size = 0.50*elsize)
# # model.remove(GCore)
# # Call gmsh kernel before add physical entities
# model.synchronize()
# # Recombine triangles if quad elements are needed
# geometry.set_recombined_surfaces([GBZone.surface, GCore.surface, polygon_.surface])
# # Generate the mesh
# mesh1 = geometry.generate_mesh(dim=2, order = 1, algorithm = 8)
#
#
# #import meshio
# #mesh = meshio.read("sunil.vtk")
# #optimized_mesh = pygmsh.optimize(mesh1, method="")
# #optimized_mesh.write("sunil.vtk")
#
# mesh1.write("sunil.vtk")
#
# grid = pv.read("sunil.vtk")
# pv.global_theme.background='maroon'
# plotter = pv.Plotter(window_size = (1400, 800))
# _ = plotter.add_axes_at_origin(x_color = 'red', y_color = 'green', z_color = 'blue',
#                                 line_width = 1,
#                                 xlabel = 'x', ylabel = 'y', zlabel = 'z',
#                                 labels_off = True)
# _ = plotter.add_points(np.array([0,0,0]),
#                         render_points_as_spheres = True,
#                         point_size = 25)
# _ = plotter.add_points(grid.points,
#                         render_points_as_spheres = True,
#                         point_size = 2)
# #_ = plotter.add_bounding_box(line_width=2, color='black')
# _ = plotter.add_mesh(grid,
#                       show_edges = True,
#                       edge_color = 'black',
#                       line_width = 1,
#                       render_points_as_spheres = True,
#                       point_size = 10,
#                       style = 'wireframe')
# plotter.view_xy()
# plotter.camera.zoom(1.5)
# plotter.show()
# =============================================================================
