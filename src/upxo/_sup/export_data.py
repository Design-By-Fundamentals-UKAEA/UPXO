"""
This module stores data used to export grain structures to different
formats.

HISTORY
-------
V.0.1: 06-03-2024. V.0.2: Operational. V.0.3: CTF numerical data format
achieved. Docuementations added. V.0.4: Add misori template added
"""
__name__ = "UPXO"
__author__ = "Dr. Sunil Anandatheertha"
__version__ = "0.0.1"
__doc__ = "Store data used to export grain structures"


import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from upxo.interfaces.os.osops import get_file_path
from upxo._sup.validation_values import _validation
from upxo.interfaces.os.osops import get_path_UPXOwriterdataDIR

def exlwrite(self, data):
    # Validations
    """
    data must be of type dictionary.
    Key names will become the sheet names, so name them properly as needed by
    excel requiremets.

    Import
    ------
    """
    wb = Workbook()
    for i, name, value in enumerate(data.items(), start=0):
        if i == 0:
            ws = wb.active
            ws.title = name
        else:
            ws = wb.create_sheet(title=name)
        # --------------------------
        for r in range(value.shape[0]):
            for c in range(value.shape[1]):
                ws.cell(row=r+1, column=c+1, value=data[r, c])


    def write(self):
        # Create a sample numpy array
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Create a new workbook and add three sheets
        wb = Workbook()
        ws1 = wb.active
        ws1.title = "Sheet1"

        # Populate Sheet1
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ws1.cell(row=r+1, column=c+1, value=data[r, c])

        # Add and populate Sheet2
        ws2 = wb.create_sheet(title="Sheet2")
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ws2.cell(row=r+1, column=c+1, value=data[r, c])

        # Add and populate Sheet3
        ws3 = wb.create_sheet(title="Sheet3")
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ws3.cell(row=r+1, column=c+1, value=data[r, c])

        # Save the workbook
        wb.save("example.xlsx")

        # Read the data back from the Excel file
        wb = load_workbook("example.xlsx")
        data_from_sheets = {}

        for sheet_name in ["Sheet1", "Sheet2", "Sheet3"]:
            ws = wb[sheet_name]
            data = np.array([[ws.cell(row=r+1, column=c+1).value for c in range(ws.max_column)] for r in range(ws.max_row)])
            data_from_sheets[sheet_name] = data

        # Print the data read from each sheet
        for sheet_name, data in data_from_sheets.items():
            print(f"Data from {sheet_name}:\n{data}\n")


class ctf():
    """
    UPXO core class to work with/using EBSD data

    SLot variables:
        root: path of this 'export_data' module
        path: path containing the required file
        header: ctf header information
        header_lines: ctf header information as lines in list
        data_format: TO BE DEPREVCATED
        val: UPXO validation instance
        phase_name: name of the phase in the CTF file
        hgrid: x-grid
        vgrid: y-grid
        nstates: Number of monte-carlo states
        mcstates: Monte-Carlo state numpy array
        euler1: 1st Bunge's Euler angle
        euler2: 2nd Bunge's Euler angle
        euler3: 3rd Bunge's Euler angle
        bands: Number of Kikuchi bands detected
        error: Acquisiton error
        mad: Mean angular deviation, deg
        bc: Band contrast [0, 255]
        bs: Band slope [0, 255]

    from upxo._sup.export_data import ctf
    ctf = ctf()
    ctf.load_header_file(hfile='_ctf_header_CuCrZr_1.txt')
    ctf.set_metadata(projectname='UKAEA: UPXO grain structure: CPFEM',
                     username='Dr. Sunil Anandatheertha')

    Example
    ---------------
    from upxo.ggrowth.mcgs import monte_carlo_grain_structure as mcgs
    pxt = mcgs()
    pxt.simulate()
    pxt.detect_grains()

    pxt.char_morph_2d(8)

    hgrid = pxt.gs[8].xgr
    vgrid = pxt.gs[8].ygr
    mcstates = pxt.gs[8].s

    from upxo._sup.export_data import ctf
    ctf = ctf()
    ctf.load_header_file()
    ctf.make_header_from_lines()
    ctf.set_phase_name(phase_name='PHNAME')

    ctf.set_grid(hgrid, vgrid)
    ctf.set_state(pxt.uisim.S, mcstates)
    ctf.set_grid_data(field_data='rand')
    ctf.plot_ea_map(term=[1, 0, 0])
    ctf.plot_ea_map(term=[0, 1, 0])
    ctf.plot_ea_map(term=[0, 0, 1])
    """
    __slots__ = ('root', 'path', 'header', 'header_lines', 'data_format',
                 'val', 'phase_name', 'phase', 'hgrid', 'vgrid', 'nstates',
                 'mcstates', 'euler1', 'euler2', 'euler3', 'bands', 'error',
                 'mad', 'bc', 'bs', 'ea1', 'ea2', 'ea3', 'data_format',
                 'data_format_ORIX', 'H', 'H_ORIX')

    def __init__(self):
        self.root = os.getcwd()
        self.phase = self.ea1 = self.ea2 = self.ea3 = None
        self.path = self.header = self.header_lines = self.data_format = None
        self.hgrid = self.vgrid = self.nstates = self.mcstates = None
        self.euler1 = self.euler2 = self.euler3 = None
        self.bands = self.error = self.mad = self.bc = self.bs = None
        self.val = _validation()
        self.data_format = "{phase}\t{x:.3f}\t{y:.3f}\t{bands}\t{error}\t{euler1:.5f}\t{euler2:.5f}\t{euler3:.5f}\t{mad:.5f}\t{bc}\t{bs}"
        self.data_format_ORIX = "{phase}\t{x:.3f}\t{y:.3f}\t{bands}\t{euler1:.5f}\t{euler2:.5f}\t{euler3:.5f}\t{error}\t{mad:.5f}\t{bc}\t{bs}"
        self.H = """Channel Text File
Prj\tUPXO_Synthetic_Grain_Structure
Author\tDr. Sunil Anandatheertha
JobMode\tGrid
XCells\t{}
YCells\t{}
XStep\t1.0
YStep\t1.0
AcqE1\t0
AcqE2\t0
AcqE3\t0
Euler angles refer to Sample Coordinate system (CS0)!	Mag	0.0000	Coverage	0	Device	0	KV	0.0000	TiltAngle	0.0000	TiltAxis	0	DetectorOrientationE1	0.0000	DetectorOrientationE2	0.0000	DetectorOrientationE3	0.0000	WorkingDistance	0.0000	InsertionDistance	0.0000
Phases\t1
3.614;3.614;3.614	90.000;90.000;90.000	Copper	11	0			Created from UPXO
Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS"""
        self.H_ORIX = """Channel Text File
Prj\tUPXO_Synthetic_Grain_Structure
Author\tDr. Sunil Anandatheertha
JobMode\tGrid
XCells\t{}
YCells\t{}
XStep\t1.0
YStep\t1.0
AcqE1\t0
AcqE2\t0
AcqE3\t0
Euler angles refer to Sample Coordinate system (CS0)!	Mag	0.0000	Coverage	0	Device	0	KV	0.0000	TiltAngle	0.0000	TiltAxis	0	DetectorOrientationE1	0.0000	DetectorOrientationE2	0.0000	DetectorOrientationE3	0.0000	WorkingDistance	0.0000	InsertionDistance	0.0000
Phases\t1
3.614;3.614;3.614	90.000;90.000;90.000	Copper	11	0			Created from UPXO
Phase	X	Y	Bands	Euler1	Euler2	Euler3	Error	MAD	BC	BS"""

    def __repr__(self):
        return "UPXO.ctf"

    def load_header_file(self,
                         pathFinding='direct',
                         hfile='_ctf_header_CuCrZr_1.txt',
                         filePath='default',
                         headerFileLocation='',
                         ):
        """
        hfile: HEader filename, including the extention.
        Author: Dr. Sunil Anandatheertha
        """
        if pathFinding == 'find':
            # Validate file Path
            if filePath == 'default':
                try:
                    self.path = get_path_UPXOwriterdataDIR()
                except Exception as e:
                    print(f'Attempting to locate {hfile} using '
                        f'alternate method due to error: {e}')
                    self.path = get_file_path(target_file_name=hfile)
                    if not self.path:
                        raise FileNotFoundError(f"{hfile} could not be found.")
            else:
                self.path = self.val.val_path_exists(filePath, throw_path=True)

            # Validate file existence
            self.val.val_file_exists(self.path, hfile)
            # Load the file
            try:
                with open(self.path/hfile, 'r', encoding='utf-8') as file:
                    self.header_lines = file.readlines()
                    print("CTF header read successful")
            except FileNotFoundError:
                print(f"The file {self.path} does not exist.")
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")
        if pathFinding == 'direct':
            self.path = headerFileLocation
            # Load the file
            try:
                with open(self.path, 'r', encoding='utf-8') as file:
                    self.header_lines = file.readlines()
                    print("CTF header read successful")
            except FileNotFoundError:
                print(f"The file {self.path} does not exist.")
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")

    def set_phase_name(self, phase_name='PHNAME'):
        '''Set phase_name attributes'''
        self.val.valstrs((phase_name, ))
        self.phase_name = phase_name
        print(f'Phase name set to {phase_name}.')

    def set_metadata(self, **kwargs):
        """
        Permitted input arguments:
            1. projectname
            2. username
        EXAMPLE:
            ctf.set_metadata(projectname='UKAEA: UPXO grain structure: CPFEM',
                             username='YOUR NAME')
        Author: Dr. Sunil Anandatheertha
        """
        # Permitted key: (line starts in header info lines, str to replace)
        _permitted_keys_starts_ = {'projectname': ('Prj',
                                                   '__UPXO_Grain_Structure__'),
                                   'username': ('Author',
                                                '__USERNAME__')
                                   }
        # Validate if all inputs are permitted inputs
        if set(kwargs.keys())-set(_permitted_keys_starts_.keys()):
            raise TypeError('One or more inputs invalid.'
                            f'permitted arguments: {_permitted_keys_starts_}')
        # Validate if all input argument values atr strings
        self.val.valstrs(kwargs.values())
        # Replace the values in ctf header infgormation
        for idx, _ in enumerate(self.header_lines):
            for k, v in _permitted_keys_starts_.items():
                if v[0] in self.header_lines[idx]:
                    self.header_lines[idx] =\
                        self.header_lines[idx].replace(v[1],
                                                       kwargs[k])

    def set_grid(self, hgrid, vgrid):
        '''Set hgrid and vgrid attributes'''
        self.val.valnparrs_nelem(hgrid, vgrid)
        self.hgrid, self.vgrid = hgrid, vgrid
        print('Coordinate grid setup successfull.')

    def set_phase(self, phase):
        '''Set phase attributes'''
        self.val.valnparrs_nelem(self.hgrid, phase)
        self.phase = phase

    def set_state(self, nstates, mcstates):
        '''Set nstates and mcstates attributes'''
        self.val.valnparrs_nelem(self.hgrid, mcstates)
        self.nstates = nstates
        self.mcstates = mcstates
        print('State set successfull.')

    def set_bands(self, bands):
        '''Set bands attributes'''
        self.val.valnparrs_nelem(self.hgrid, bands)
        self.bands = bands

    def set_error(self, error):
        '''Set error attributes'''
        self.val.valnparrs_nelem(self.hgrid, error)
        self.error = error

    def set_ea(self, euler1, euler2, euler3):
        '''Set euler1, euler2 and euler3 attributes'''
        self.val.valnparrs_nelem(self.hgrid, euler1, euler2, euler3)
        self.euler1 = euler1
        self.euler2 = euler2
        self.euler3 = euler3

    def make_header_from_lines(self):
        self.header = ''.join(self.header_lines)
        print('Header lines creation successful.')

    def add_misori(self,
                   glb_pert_min_ea1=0, glb_pert_max_ea1=7.5,
                   glb_pert_min_ea2=0, glb_pert_max_ea2=7.5,
                   glb_pert_min_ea3=0, glb_pert_max_ea3=7.5,
                   lcl_pert_min_ea1=0, lcl_pert_max_ea1=2.5,
                   lcl_pert_min_ea2=0, lcl_pert_max_ea2=2.5,
                   lcl_pert_min_ea3=0, lcl_pert_max_ea3=2.5,
                   distance_measure='cartesian'):
        """
        Add a perturnation to euler angles.
        glb (i.e. global) values apply across Monte-Carlo state wise
        loc (i.e. local) values apply to each Monte-Carlo state wise

        removing loc would be akin to having a grain averaged orientation
        removing glb would be akin to having a tex comp averaged
        orientation.

        Adding glb perturbations:
        -------------------------
        Initially, a unique Euler angle would be asscoaited to a unique
        mcstate value. Hence, many grains having the same state would be
        assigned the same Euler angle. To overcome this, a glb pert bounded
        by [glb_pert_min, glb_pert_max] is introduced. HEnce a uniquie
        Euler angle nearby to the previous oreintation is introduced to
        every grain id beloging to a certain state. This is then repeated
        for all available states, thus covering all the grains.

        I provide a pseudo code and implementation below:



        STEP 1 ---->
        * Build s list
        * for s in range(pxtal.uisim.S): sgid = {s: [gid1, gid2, ...]}
        The above may just as well be achieved by takint it from gs instance
        sgid = pxtal.gs[n].sgid
        -------:: IMPLEMENTATION of STEP 1 -- tested
        mcstep = 8
        S = np.arange(1, pxt.uisim.S+1)
        sgid = pxt.gs[mcstep].s_gid
        euler1 = np.zeros_like(hgrid)
        euler2 = np.zeros_like(hgrid)
        euler3 = np.zeros_like(hgrid)



        STEP 2 ---->
        Build eas, representing state wise Bunge's Euler angle mapping
        # Build Euler1, Euler2 and Euler3
        -------:: IMPLEMENTATION of STEP 2 -- tested
        eas = {s: None for s in S}
        for s in S:
            if sgid[s]:
                # GEt the number of grains belonging to this s
                ngrains = len(sgid[s])
                # Build EA for all grains in this s
                _ea_ = np.tile(np.array([ctf.ea1[s-1],
                                         ctf.ea2[s-1],
                                         ctf.ea3[s-1]]), (ngrains, 1))

                # Build a random plus or minus array
                _pm_ = np.random.choice([-1, 1], (ngrains, 3))
                # Build EA perturations skeleton
                _del_ = np.random.random((len(sgid[s]), 3))
                # Find maximum perturbation distance
                _delmax_ = glb_pert_max_ea1 - glb_pert_min_ea1
                # Build EA perturations
                _del_ = glb_pert_min_ea1 + _del_*_delmax_
                # Instroduce perturbation
                _ea_ += _pm_*_del_
                eas[s] = _ea_

eas = {s: None for s in S}
for s in S:
    if sgid[s]:
        # GEt the number of grains belonging to this s
        ngrains = len(sgid[s])
        # Build EA for all grains in this s
        _ea_ = np.tile(np.array([ctf.ea1[s-1],
                                 ctf.ea2[s-1],
                                 ctf.ea3[s-1]]), (ngrains, 1))
        eas[s] = _ea_

for s in S:
    if sgid[s]:
        for i, gid in enumerate(sgid[s], start=0):
            locs = pxt.gs[mcstep].g[gid]['grain'].loc
            for loc in locs:
                ea = eas[s][i]
                euler1[loc[0], loc[1]] = ea[0]
                euler2[loc[0], loc[1]] = ea[1]
                euler3[loc[0], loc[1]] = ea[2]

plt.imshow((euler1+euler2+euler3)/3, cmap = 'nipy_spectral')

for s in S:
    if sgid[s]:
        # GEt the number of grains belonging to this s
        ngrains = len(sgid[s])
        # Build a random plus or minus array
        _pm_ = np.random.choice([-1, 1], (ngrains, 3))
        # Build EA perturations skeleton
        _del_ = np.random.random((len(sgid[s]), 3))
        # Find maximum perturbation distance
        _delmax_ = glb_pert_max_ea1 - glb_pert_min_ea1
        # Build EA perturations
        _del_ = glb_pert_min_ea1 + _del_*_delmax_
        # Instroduce perturbation
        eas[s] += _pm_*_del_


        STEP 3 ---->
        for s in S:
            if sgid[s]:
                for i, gid in enumerate(sgid[s], start=0):
                    locs = pxt.gs[mcstep].g[gid]['grain'].loc
                    for loc in locs:
                        ea = eas[s][i]
                        euler1[loc[0], loc[1]] = ea[0]
                        euler2[loc[0], loc[1]] = ea[1]
                        euler3[loc[0], loc[1]] = ea[2]

        plt.imshow(euler1)
        plt.imshow(euler3, cmap = 'gist_ncar')
        plt.imshow(euler3, cmap = 'nipy_spectral')
        plt.imshow((euler1+euler2+euler3)/3, cmap = 'nipy_spectral')

        for s in S:
            if sgid[s]:
                ngrains = len(sgid[s])
                for i, gid in enumerate(sgid[s], start=0):
                    locs = pxt.gs[mcstep].g[gid]['grain'].loc
                    npxl = locs.shape[0]
                    pm = np.random.choice([-1, 1], (npxl, 3))
                    ea1 = euler1[locs[:, 0], locs[:, 1]]
                    ea2 = euler2[locs[:, 0], locs[:, 1]]
                    ea3 = euler3[locs[:, 0], locs[:, 1]]
                    _del_ = np.random.random((npxl, 3))
                    _delmax_ = lcl_pert_max_ea1 - lcl_pert_min_ea1
                    _del_ = lcl_pert_min_ea1 + _del_*_delmax_
                    ea1 += pm[:, 0]*_del_[:, 0]
                    ea2 += pm[:, 1]*_del_[:, 1]
                    ea3 += pm[:, 2]*_del_[:, 2]
                    euler1[locs[:, 0], locs[:, 1]] = ea1
                    euler2[locs[:, 0], locs[:, 1]] = ea2
                    euler3[locs[:, 0], locs[:, 1]] = ea3



# Build EA perturations skeleton
_del_ = np.random.random((len(sgid[s]), 3))
# Find maximum perturbation distance
_delmax_ = glb_pert_max_ea1 - glb_pert_min_ea1
# Build EA perturations
_del_ = glb_pert_min_ea1 + _del_*_delmax_
# Instroduce perturbation
_ea_ += _pm_*_del_


                    for loc in locs:
                        euler1[loc[0], loc[1]] += ea[0]
                        euler2[loc[0], loc[1]] += ea[1]
                        euler3[loc[0], loc[1]] += ea[2]



        Adding lcl perturbations:
        -------------------------
        The above operation would then produce no orientation gradients
        inside a grain. lcl_pert is used to introduce such gradients.

        The approach to introduce glb and loc perturbations may change in the
        future, but the intent is established.

        Author: Dr. Sunil Anandatheertha
        pxt.gs[8].g[1]['grain'].neigh -- lists the neighbouring grains
        """
        pass

    def gen_rand_grid_field_data(self):
        """
        Generate random grid field data. Though random, pixels of same state
        value wqill have the same euler angles, band contrast, band slope,
        mean angular deviation, nu8mber of bands and error.

        Author: Dr. Sunil Anandatheertha
        """
        # ------------------------------------
        # INITIATE THE PHASE ARRAY
        '''
        # TODO: This could be a nice entry point to introduce phases to grains
        Possible pseudo code for later implementaytion.

        if 0 in mcstate:
            1. Introduce a phase ID'ed 2 in self.phase for the corresponding
            grain id.
            2. Update the CTF header information about the new phase
        '''
        print('Setting state-mapped random field data specification.')
        self.phase = np.ones_like(self.hgrid, dtype=float)
        # ------------------------------------
        # Initiate all oher arrays
        self.euler1 = np.ones_like(self.hgrid, dtype=float)
        self.euler2 = np.ones_like(self.hgrid, dtype=float)
        self.euler3 = np.ones_like(self.hgrid, dtype=float)
        self.bc = np.ones_like(self.hgrid, dtype=float)
        self.bs = np.ones_like(self.hgrid, dtype=float)
        # ------------------------------------
        # Create the Euler angle - mcstate value mapper arrays
        self.ea1, self.ea2, self.ea3 = np.random.uniform([0, 0, 0],
                                                         [360, 180, 360],
                                                         (self.nstates, 3)).T
        # ------------------------------------
        # Create the bc and bs - mcstate value mapper arrays
        _bc_ = np.random.randint(0, 255, (self.nstates, ))
        _bs_ = np.random.randint(0, 255, (self.nstates, ))
        # ------------------------------------
        # Map the euler angles, bc and bs
        for i, __ in enumerate(self.ea1, start=0):
            _ = self.mcstates == i+1
            self.euler1[_] = self.ea1[i]
            self.euler2[_] = self.ea2[i]
            self.euler3[_] = self.ea3[i]
            self.bc[_] = _bc_[i]
            self.bs[_] = _bs_[i]
        # ------------------------------------
        # Create error array
        self.error = np.zeros(self.hgrid.shape, dtype=int)
        # ------------------------------------
        # Create MAD values b/w 0.3 and 0.4
        self.mad = 0.3+0.1*np.random.random(self.hgrid.shape)
        # ------------------------------------
        # Create bands arrays
        self.bands = np.random.randint(7, 10, self.hgrid.shape)
        print('Field data specification succeffsul.')

    def format_header(self):
        '''Format the header to required size'''
        self.header = self.header.format(self.hgrid.size, self.vgrid.size)

    def set_grid_data(self, field_data='rand',
                      bands=None, error=None,
                      euler1=None, euler2=None, euler3=None,
                      mad=None, bc=None, bs=None):
        '''Set bands, error, euler1, euler2, euler3, mad, bc and bs
        attributes'''
        if field_data != 'rand':
            # Validate for existance, type and shape of bands, error,...
            self.val.valnparrs_shapes(bands, error, euler1, euler2, euler3,
                                      mad, bc, bs)
            self.val.valnparrs_nelem(self.hgrid, self.vgrid, bands, error,
                                     euler1, euler2, euler3, mad, bc, bs)
        else:
            self.gen_rand_grid_field_data()

    def assemble_grid_data(self):
        ctf_numerical_data = np.vstack((self.phase.ravel(),
                                        self.hgrid.ravel(),
                                        self.vgrid.ravel(),
                                        self.bands.ravel(),
                                        self.error.ravel(),
                                        self.euler1.ravel(),
                                        self.euler2.ravel(),
                                        self.euler3.ravel(),
                                        self.mad.ravel(),
                                        self.bc.ravel(),
                                        self.bs.ravel(),
                                        )
                                       )
        print('CTF coordinate-field data assembly successfull.')
        return ctf_numerical_data

    def write_ctf_file(self, folder, fileName):
        nc, nr = self.hgrid.shape
        # data_lines = [self.header.format(nc, nr)]
        data_lines = [self.H.format(nc, nr)]
        for r in range(nr):
            for c in range(nc):
                # Preparing data line
                data_line = self.data_format.format(phase=self.phase[r][c],
                                                    x=self.hgrid[r][c],
                                                    y=self.vgrid[r][c],
                                                    bands=self.bands[r][c],
                                                    error=self.error[r][c],
                                                    euler1=self.euler1[r][c],
                                                    euler2=self.euler2[r][c],
                                                    euler3=self.euler3[r][c],
                                                    mad=self.mad[r][c],
                                                    bc=self.bc[r][c],
                                                    bs=self.bs[r][c])
                data_lines.append(data_line)
        # Join all data
        ctf_data = "\n".join(data_lines)
        filePathBase = Path(folder)
        # filePathBase = Path(r"C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\_written_data\_ctf_export_2dmcgs")
        file_path = filePathBase / f"{fileName}.ctf"

        # Writing the data to a .ctf file
        with open(file_path, 'w') as file:
            file.write(ctf_data)
            print(file_path)

    def write_ctf_file_ORIX(self, folder, fileName):
        nr, nc = self.hgrid.shape
        # data_lines = [self.header.format(nc, nr)]
        data_lines = [self.H_ORIX.format(nc, nr)]
        for r in range(nr):
            for c in range(nc):
                # Preparing data line
                data_line = self.data_format_ORIX.format(phase=self.phase[r][c],
                                                    x=self.hgrid[r][c],
                                                    y=self.vgrid[r][c],
                                                    bands=self.bands[r][c],
                                                    euler1=self.euler1[r][c],
                                                    euler2=self.euler2[r][c],
                                                    euler3=self.euler3[r][c],
                                                    error=self.error[r][c],
                                                    mad=self.mad[r][c],
                                                    bc=self.bc[r][c],
                                                    bs=self.bs[r][c])
                data_lines.append(data_line)
        # Join all data
        ctf_data = "\n".join(data_lines)
        filePathBase = Path(folder)
        # filePathBase = Path(r"C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\_written_data\_ctf_export_2dmcgs")
        file_path = filePathBase / f"{fileName}.ctf"

        # Writing the data to a .ctf file
        with open(file_path, 'w') as file:
            file.write(ctf_data)
            print(file_path)

    def plot_ea_map(self, term=[1, 0, 0]):
        """
        Quickly plot the Euler angle map
        """
        term = [int(_/max(term)) for _ in term]
        if all([_ == 0 for _ in term]):
            raise ValueError(f'Input arg term: {term} is invalid')
        ea = term[0]*self.euler1 + term[1]*self.euler2 + term[2]*self.euler3
        ea = ea/term.count(1)
        plt.imshow(ea)
