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
    m:
        # Temporal slice
    coords_element_nodes:
        # Coordinates of FE nodes
    nodal_connectivity:
        DEscription
    meshInfo: dict
        General mesh information
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
    lfi:
        DEscription
    dim:
        DEscription
    elsets:
        Element sets of features
    ndsets:
        Nodal sets
    """
    __slots__ = ('meshing_package',
                 'element_type',
                 'target_fe_software',
                 'reduced_integration',
                 'coords_element_nodes',
                 'nodal_connectivity',
                 'meshInfo', 'gsInfo', 'gridInfo',
                 'ABQ_NODES',
                 'ABQ_ELEMENTS',
                 'xgrid',
                 'ygrid',
                 'lfi',
                 'dim',
                 'elsets',
                 'ndsets',
                 )
    def __init__(self,
                 getFEControlsFromUPXO=True,
                 getGridControlsFromUPXO=True,
                 meshInfo=dict(elementType='quad4',
                               FESoftware='abaqus',
                               mesher='UPXO',
                               reduced_integration=False, ),
                 gridInfo=dict(xmin=0, xinc=1, xmax=10,
                               ymin=0, yinc=1, ymax=5, ),
                 gsInfo=dict(m=10, grainAreas=np.random.random(10), ),
                 lfi=np.random.randint(low=1, high=10, size=(10, 5), dtype=np.int32),
                 ):
        # -----------------------------------------------
        if getFEControlsFromUPXO:
            self.meshInfo = dict(nNodes=np.nan, nElements=np.nan,
                elementType = meshInfo.mesh_element_type,
                FESoftware = meshInfo.mesh_target_fe_software,
                mesher = meshInfo.mesh_meshing_package,
                reducedIntegration = meshInfo.mesh_reduced_integration,
                )
        else:
            self.meshInfo = dict(nNodes=np.nan, nElements=np.nan,
                elementType = meshInfo['elementType'],
                FESoftware = meshInfo['FESoftware'],
                mesher = meshInfo['mesher'],
                reducedIntegration = meshInfo['reducedIntegration'],
                )
        # -----------------------------------------------
        if getGridControlsFromUPXO:
            self.gridInfo = dict(xmin=gridInfo.xmin,
                                 xinc=gridInfo.xinc,
                                 xmax=gridInfo.xmax,
                                 ymin=gridInfo.ymin,
                                 yinc=gridInfo.yinc,
                                 ymax=gridInfo.ymax,)
        else:
            self.gridInfo = dict(xmin=gridInfo['xmin'],
                                 xinc=gridInfo['xinc'],
                                 xmax=gridInfo['xmax'],
                                 ymin=gridInfo['ymin'],
                                 yinc=gridInfo['yinc'],
                                 ymax=gridInfo['ymax'], )
        # -----------------------------------------------
        self.gsInfo = dict(m=gsInfo.get('m', 1), dim=2,)
        self.lfi = lfi
        self.mesher_brancher()

    def __att__(self):
        return att(self)

    def mesher_brancher(self):
        if self.meshInfo['FESoftware'] == 'Abaqus':
            self.abaqus_mesher_brancher()

    def abaqus_mesher_brancher(self):
        if self.meshInfo['mesher'] == 'UPXO':
            self.upxo_abaqus_nonconformal_mesher()

    def upxo_abaqus_nonconformal_mesher(self):
        if self.meshInfo['elementType'] == 'quad4':
            self.mesh_abaqus_upxo_nonconformal_quad4()
        if self.meshInfo['elementType'] == 'quad8':
            pass

    def mesh_abaqus_upxo_nonconformal_quad4(self):
        """
        Generates ABAQUS compatible mesh data-structure inside UPXO
        targetted at non-conformal mesh with 4 noded quadrilateral elements.
        """
        # ------------------------------------------------------
        Xlat, Ylat = np.meshgrid(np.arange(self.gridInfo['xmin'],
                                           self.gridInfo['xmax']+1,
                                           self.gridInfo['xinc']),
                                 np.arange(self.gridInfo['ymin'],
                                           self.gridInfo['ymax']+1,
                                           self.gridInfo['yinc']),
                                 indexing='xy')
        self.xgrid, self.ygrid = deepcopy(Xlat), deepcopy(Ylat)
        Xlat, Ylat = np.concatenate(Xlat), np.concatenate(Ylat)
        # ------------------------------------------------------
        X, Y = np.meshgrid(np.arange(self.gridInfo['xmin']-0.5*self.gridInfo['xinc'],
                                     self.gridInfo['xmax']+1.0*self.gridInfo['xinc'],
                                     self.gridInfo['xinc']),
                           np.arange(self.gridInfo['ymin']-0.5*self.gridInfo['yinc'],
                                     self.gridInfo['ymax']+1.0*self.gridInfo['yinc'],
                                     self.gridInfo['yinc']),
                           indexing='xy')
        # ------------------------------------------------------
        dx, dy = 1.1*0.5*self.gridInfo['xinc'], 1.1*0.5*self.gridInfo['yinc']
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
        for xl in np.arange(self.gridInfo['xmin'],
                            self.gridInfo['xmax']+1,
                            self.gridInfo['xinc']):
            for yl in np.arange(self.gridInfo['ymin'],
                                self.gridInfo['ymax']+1,
                                self.gridInfo['yinc']):
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
        self.meshInfo['nNodes'] = len(self.ABQ_NODES)
        # Store the number of elements
        self.meshInfo['nElements'] = len(self.ABQ_ELEMENTS)

    def map_elements_grainids(self):
        '''
        This makes a map of grain ids and the pixel ids. Map is stored
        in a dictionary. grain id forms the keys and pixel ids for the
        values

        This is akin to the element-set defined in ABAQUS
        '''
        # Make global element ids array
        eids = np.reshape(self.ABQ_ELEMENTS.T[0], self.xgrid.shape)
        # Find locations in lfi, where values equal different grain id numbers
        a = [(np.where(self.lfi == gid)) for gid in range(1, self.lfi.max()+1)]
        a = [[list(_a[0]), list(_a[1])] for _a in a]  # Reformat data structure
        # Construct element sets. There will be as many elmewnts in
        # grain_element_sets, as the number of grains in the domain.
        grain_element_sets = {}
        for gid in range(0, self.lfi.max()):
            grain_element_sets[gid] = []
            for r, c in zip(a[gid][0], a[gid][1]):
                grain_element_sets[gid].append(eids[r, c])
        return grain_element_sets

    def abaqus_make_element_sets(self):
        return self.map_elements_grainids()

    def info(self):
        print(f"Number of elements, nodes: {self.meshInfo['nElements']}, {self.meshInfo['nNodes']}")
        print(f"Element type: {self.meshInfo['elementType']}")
        print(f"Temporal slice number of the mcgs: {self.gsInfo['m']}")
        print(f"Target FE software: {self.meshInfo['FESoftware']}")

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
