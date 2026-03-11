import os
from copy import deepcopy
import numpy as np
from upxo._sup.gops import att

class mesh_mcgs2d():
    """
    This is a core mcgs.meshing class
    Intended only for Monte-Carlo simulation grain structures
    Works only on square lattice.
    For non-square lattices, interpolate to a square lattice and use

    meshing_package:
        # UPXO, GMSH, PYGMSH
    gb_conformity:
        # Conformal or Non-conformal
    element_type:
        # Type of finite element
    target_fe_software:
        # Abaqus or Moose
    reduced_integration:
        # BOOL: Flag for reduced integration
    abaqus_prop:
        # Abaqus mesh properties
    m:
        # Temporal slice
    coords_element_nodes:
        # Coordinates of FE nodes
    nodal_connectivity:
        DEscription
    n_nodes:
        # Number of nodes
    n_elements:
        # Number of elements
    ABQ_NODES:
        # Nodes contained in ABAQUS friendly format
    ABQ_ELEMENTS:
        # Elements contained in ABAQUS friendly format
    __uimesh:
        # Private copy of pxtal.uimesh
    __uigrid:
        # Private copy of pxtal.uigrid
    xgrid:
        DEscription
    ygrid:
        DEscription
    lgi:
        DEscription
    dim:
        DEscription
    elsets_grains:
        Element sets for grains
    elsets_gbz:
        Element sets for grain boundary zones
    elsets_gcr:
        Element sets for grain core
    ndsets_xl:
        Nodal set for left extreme edge boundary condition
    ndsets_xer:
        Nodal set for right extreme edge boundary condition
    ndsets_yl:
        Nodal set for bottom extreme edge boundary condition
    ndsets_yr:
        Nodal set for top extreme edge boundary condition
    """
    __slots__ = ('meshing_package',
                 'gb_conformity',
                 'element_type',
                 'target_fe_software',
                 'reduced_integration',
                 'abaqus_prop',
                 'm',
                 'coords_element_nodes',
                 'nodal_connectivity',
                 'n_nodes',
                 'n_elements',
                 'ABQ_NODES',
                 'ABQ_ELEMENTS',
                 '__uimesh',
                 '__uigrid',
                 'xgrid',
                 'ygrid',
                 'lgi',
                 'dim',
                 'elsets_grains',
                 'elsets_gbz',
                 'elsets_gcr',
                 'ndsets_xl',
                 'ndsets_xr',
                 'ndsets_yl',
                 'ndsets_yr',
                 )

    def __init__(self, uimesh, uigrid, dim, m, lgi):
        self.__uimesh = uimesh
        self.__uigrid = uigrid
        self.target_fe_software = uimesh.mesh_target_fe_software
        self.meshing_package = uimesh.mesh_meshing_package
        self.gb_conformity = uimesh.mesh_gb_conformity
        self.element_type = uimesh.mesh_element_type
        self.reduced_integration = uimesh.mesh_reduced_integration
        self.m = m
        self.lgi = lgi
        self.dim = dim
        self.mesher_brancher()

    def __att__(self):
        return att(self)

    def mesher_brancher(self):
        if self.target_fe_software == 'Abaqus':
            print('--------- 1 ---------')
            self.abaqus_mesher_brancher()
        elif self.target_fe_software == 'Moose':
            self.moose_mesher_brancher()
        elif self.target_fe_software == 'DAMASK':
            self.damask_mesher_brancher()

    def abaqus_mesher_brancher(self):
        print(self.meshing_package)
        print(self.gb_conformity)
        if self.meshing_package == 'UPXO' and self.gb_conformity == 'Non-Conformal':
            print('--------- 2 ---------')
            self.upxo_abaqus_nonconformal_mesher()
        elif self.meshing_package in ('UPXO',
                                      'GMSH') and self.gb_conformity == 'Conformal':
            self.gmsh_abaqus_conformal_mesher()  # <-------- MAKE NEW DEF
        elif self.meshing_package == 'PyGMSH' and self.gb_conformity == 'Conformal':
            self.pygmsh_abaqus_conformal_mesher()  # <-------- MAKE NEW DEF
        elif self.meshing_package == 'Abaqus' and self.gb_conformity == 'Conformal':
            self.abaqus_abaqus_conformal_mesher()  # <-------- MAKE NEW DEF
        pass

    def moose_mesher_brancher(self):
        pass

    def damask_mesher_brancher(self):
        pass

    def upxo_abaqus_nonconformal_mesher(self):
        if self.dim == 2 and self.element_type == 'quad4':
            print('--------- 3 ---------')
            self.mesh_abaqus_upxo_nonconformal_quad4()
        elif self.dim == 2 and self.element_type == 'quad8':
            self.mesh_abaqus_upxo_nonconformal_quad8()  # <-------- MAKE NEW DEF
        elif self.dim == 3 and self.element_type == 'hex8':
            self.mesh_abaqus_upxo_nonconformal_hex8()  # <-------- MAKE NEW DEF
        elif self.dim == 3 and self.element_type == 'hex20':
            self.mesh_abaqus_upxo_nonconformal_hex20()  # <-------- MAKE NEW DEF

    def upxo_abaqus_conformal_mesher(self):
        if self.dim == 2 and self.element_type == 'quad4':
            self.mesh_abaqus_upxo_conformal_quad4()  # <-------- MAKE NEW DEF
        elif self.dim == 2 and self.element_type == 'quad8':
            self.mesh_abaqus_upxo_conformal_quad8()  # <-------- MAKE NEW DEF
        elif self.dim == 3 and self.element_type == 'hex8':
            self.mesh_abaqus_upxo_conformal_tri3()  # <-------- MAKE NEW DEF
        elif self.dim == 3 and self.element_type == 'hex20':
            self.mesh_abaqus_upxo_conformal_tri6()  # <-------- MAKE NEW DEF

    def mesh_abaqus_upxo_nonconformal_quad4(self):
        """
        Generates ABAQUS compatible mesh data-structure inside UPXO
        targetted at non-conformal mesh with 4 noded quadrilateral elements.
        """
        # ------------------------------------------------------
        Xlat, Ylat = np.meshgrid(np.arange(self.__uigrid.xmin,
                                           self.__uigrid.xmax+1,
                                           self.__uigrid.xinc),
                                 np.arange(self.__uigrid.ymin,
                                           self.__uigrid.ymax+1,
                                           self.__uigrid.yinc),
                                 indexing='xy')
        self.xgrid, self.ygrid = deepcopy(Xlat), deepcopy(Ylat)
        Xlat, Ylat = np.concatenate(Xlat), np.concatenate(Ylat)
        # ------------------------------------------------------
        X, Y = np.meshgrid(np.arange(self.__uigrid.xmin-0.5*self.__uigrid.xinc,
                                     self.__uigrid.xmax+1.0*self.__uigrid.xinc,
                                     self.__uigrid.xinc),
                           np.arange(self.__uigrid.ymin-0.5*self.__uigrid.yinc,
                                     self.__uigrid.ymax+1.0*self.__uigrid.yinc,
                                     self.__uigrid.yinc),
                           indexing='xy')
        # ------------------------------------------------------
        dx, dy = 1.1*0.5*self.__uigrid.xinc, 1.1*0.5*self.__uigrid.xinc
        # ------------------------------------------------------
        xbl, ybl = list(np.concatenate(X)), list(np.concatenate(Y))
        xtl, ytl = list(np.concatenate(X)), list(np.concatenate(Y))
        xtr, ytr = list(np.concatenate(X)), list(np.concatenate(Y))
        xbr, ybr = list(np.concatenate(X)), list(np.concatenate(Y))
        _x_, _y_ = [], []
        for _ in xbl:
            _x_.append(_)
        for _ in xtl:
            _x_.append(_)
        for _ in xtr:
            _x_.append(_)
        for _ in xbr:
            _x_.append(_)
        for _ in ybl:
            _y_.append(_)
        for _ in ytl:
            _y_.append(_)
        for _ in ytr:
            _y_.append(_)
        for _ in ybr:
            _y_.append(_)
        _x_, _y_ = np.array(_x_), np.array(_y_)
        # ------------------------------------------------------
        _xy_ = np.vstack((_x_, _y_)).T
        u, indices = np.unique(_xy_, axis=0, return_index=True)
        X_vec, Y_vec = np.array([_xy_[i] for i in sorted(indices)]).T
        # ------------------------------------------------------
        self.coords_element_nodes, self.nodal_connectivity = [], []
        # ------------------------------------------------------
        for xl in np.arange(self.__uigrid.xmin,
                            self.__uigrid.xmax+1,
                            self.__uigrid.xinc):
            for yl in np.arange(self.__uigrid.ymin,
                                self.__uigrid.ymax+1,
                                self.__uigrid.yinc):
                # Calculate the expanded nodal bounds of the current elemewnt
                xmin = xl-dx
                xmax = xl+dx
                ymin = yl-dy
                ymax = yl+dy
                # Find nodes within these bounds
                locs_x_possibilities = np.logical_and(X_vec >= xmin, X_vec <= xmax)
                locs_y_possibilities = np.logical_and(Y_vec >= ymin, Y_vec <= ymax)
                nodes_locs = np.logical_and(locs_x_possibilities, locs_y_possibilities)
                # Get the numbers  nodes inside this bound and
                # take it nodal connectivity array
                el_nodes = list(np.where(nodes_locs)[0])
                el_nodes = [el_nodes[0]+1, el_nodes[1]+1, el_nodes[3]+1, el_nodes[2]+1]
                self.nodal_connectivity.append(el_nodes)
                # Get the x and y coordinates of the bounded nodes
                a, b, c, d = X_vec[nodes_locs]
                nodal_coords_x = [a, b, d, c]
                a, b, c, d = Y_vec[nodes_locs]
                nodal_coords_y = [a, b, d, c]
                # Get coords of all nodes in the presente element
                self.coords_element_nodes.append(np.vstack((nodal_coords_x,
                                                            nodal_coords_y)))
        # Get nodes and the corresponding coordinates
        nodes_and_coords = []
        for ncon, ncoord in zip(self.nodal_connectivity,
                                self.coords_element_nodes):
            for _nc_, _ncoord_x_, _ncoord_y_ in zip(ncon,
                                                    ncoord[0],
                                                    ncoord[1]):
                nodes_and_coords.append([_nc_, _ncoord_x_, _ncoord_y_])
        # Build nodes data suitable for ABAQUS input file
        self.ABQ_NODES = np.unique(np.array(nodes_and_coords), axis=0)
        ## Add the 0's for the z-dimension
        #self.ABQ_NODES = np.hstack((self.ABQ_NODES,
        #                            np.zeros((self.ABQ_NODES.shape[0], 1))))
        # Build elements data suitable for ABAQUS input file
        # i.e. nodal connectivity table
        self.ABQ_ELEMENTS = np.array([[i+1]+self.nodal_connectivity[i]
                                      for i in range(len(self.nodal_connectivity))],
                                     dtype=int)
        # Store the number of nodes
        self.n_nodes = len(self.ABQ_NODES)
        # Store the number of elements
        self.n_elements = len(self.ABQ_ELEMENTS)

    def map_elements_grainids(self):
        '''
        This makes a map of grain ids and the pixel ids. Map is stored
        in a dictionary. grain id forms the keys and pixel ids for the
        values

        This is akin to the element-set defined in ABAQUS
        '''
        # Make global element ids array
        eids = np.reshape(self.ABQ_ELEMENTS.T[0], self.xgrid.shape)
        # Find locations in lgi, where values equal different grain id numbers
        a = [(np.where(self.lgi == gid)) for gid in range(1, self.lgi.max()+1)]
        a = [[list(_a[0]), list(_a[1])] for _a in a]  # Reformat data structure
        # Construct element sets. There will be as many elmewnts in
        # grain_element_sets, as the number of grains in the domain.
        grain_element_sets = {}
        for gid in range(0, self.lgi.max()):
            grain_element_sets[gid] = []
            for r, c in zip(a[gid][0], a[gid][1]):
                grain_element_sets[gid].append(eids[r, c])
        return grain_element_sets

    def abaqus_make_element_sets(self):
        return self.map_elements_grainids()

    def info(self):
        print(f"Number of elements, nodes: {self.n_elements}, {self.n_nodes}")
        print(f"Element type: {self.element_type}")
        print(f"Grain boundary conformity: {self.gb_conformity}")
        print(f"Temporal slice number of the mcgs: {self.m}")
        print(f"Target FE software: {self.target_fe_software}")

    def export_abaqus_inp_file(self,
                               folder="DATABASE_FEMESH",
                               file="UPXO_ABQ_MESH.inp",
                               ):
        os.makedirs(folder, exist_ok=True)
        file = file
        file_path = os.path.join(folder, file)
        with open(file_path, 'w') as f:
            f.write('*Heading\n')
            f.write('** Job name: UPXO-ABAQUS Model name: Model-1\n')
            f.write('** Generated by: Abaqus/CAE 6.14-1\n')
            # f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
            f.write('**\n')
            f.write('** PARTS\n')
            f.write('**\n')
            f.write('*Part, name=Part-1\n')
            f.write('*Node\n')

            np.savetxt(f,
                       self.ABQ_NODES,
                       fmt='    %d, %4.8f, %4.8f',
                       delimiter=', ',
                       header='',
                       comments='')

            f.write('*Element, type=CPS4\n')

            np.savetxt(f,
                       self.ABQ_ELEMENTS,
                       fmt='    %d, %d, %d, %d, %d',
                       delimiter=', ',
                       header='',
                       comments='')

            elsets = self.abaqus_make_element_sets().values()
            n_elsets = len(elsets)
            elset_names = []
            section_names = []
            material_names = []

            for esetnum, _ in enumerate(elsets):
                if esetnum < 10:
                    elset_names.append(f'Grain-000{esetnum}')
                    section_names.append(f'Sec-Grain-000{esetnum}')
                    material_names.append(f'Mat-Grain-000{esetnum}')
                elif esetnum >= 10 and 1 < 100:
                    elset_names.append(f'Grain-00{esetnum}')
                    section_names.append(f'Sec-Grain-00{esetnum}')
                    material_names.append(f'Mat-Grain-00{esetnum}')
                elif esetnum >= 100 and 1 < 1000:
                    elset_names.append(f'Grain-0{esetnum}')
                    section_names.append(f'Sec-Grain-0{esetnum}')
                    material_names.append(f'Mat-Grain-0{esetnum}')
                else:
                    elset_names.append(f'Grain-{esetnum}')
                    section_names.append(f'Sec-Grain-{esetnum}')
                    material_names.append(f'Mat-Grain-{esetnum}')

            # Write element sets
            for esetnum, eset in enumerate(elsets):
                f.write(f'*Elset, elset={elset_names[esetnum]}\n')
                line = ''
                for i, number in enumerate(eset):
                    line += str(number)
                    if i % 9 != 8 and i != len(eset) - 1:
                        line += ', '
                    elif i == len(eset) - 1:
                        f.write(line + '\n')
                    else:
                        f.write(line + ',\n')
                        line = ''

            # Wirte sections
            for section, elset, material in zip(section_names,
                                                elset_names,
                                                material_names):
                f.write(f"**Section: {section}\n")
                f.write(f"*Solid Section, elset={elset}, material={material}\n")
                f.write(",\n\n")

            f.write("*End Part\n")
            f.write("**\n")

            # Write materials
            for n_mat, mat in enumerate(material_names):
                f.write(f'*Material, name={mat}\n')
                f.write('*Depvar\n')
                f.write('12,\n')
                f.write('*User Material, constants=6\n')
                ea1 = np.random.randint(0, 359)
                ea2 = np.random.randint(0, 359)
                ea3 = np.random.randint(0, 180)
                f.write(f'{ea1},{ea2},{ea3},{n_mat},2,0\n')

            f.write('\n')
            f.write('**\n')
            f.write('** STEP: Loading\n')
            f.write('**\n')
            f.write('*Step, name=Loading, nlgeom=YES, inc=10000\n')
            f.write('*Static\n')
            f.write('0.01, 10., 1e-05, 1.\n')
            f.write('**\n')
            f.write('** OUTPUT REQUESTS\n')
            f.write('**\n')
            f.write('*Restart, write, frequency=0\n')
            f.write('**\n')
            f.write('** FIELD OUTPUT: F-Output-1\n')
            f.write('**\n')
            f.write('*Output, field, variable=PRESELECT\n')
            f.write('**\n')
            f.write('** FIELD OUTPUT: F-Output-2\n')
            f.write('**\n')
            f.write('*Element Output, directions=YES\n')
            f.write('SDV,\n')
            f.write('**\n')
            f.write('** HISTORY OUTPUT: H-Output-1\n')
            f.write('**\n')
            f.write('*Output, history, variable=PRESELECT\n')
            f.write('**\n')
            f.write('*End Step\n')
            print('\n')
            self.info()
            print('..............')
            print(f"Number of grain element sets: {len(elset_names)}")
            print(f"Numebr of materials: {len(material_names)}")
            print('..............')
            print('ABAQUS input file has been successfully written')
            print('\n')
            print('-------------------------------------------')
        f.close()
