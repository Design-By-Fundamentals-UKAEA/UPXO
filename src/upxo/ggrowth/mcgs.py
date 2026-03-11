
import numpy as np
from upxo._sup import gops
from upxo._sup import dataTypeHandlers as dth
import upxo._sup.decorators as decorators

class grid():
    """
    Description
    -----------
    This is a core UPXO > mcgs class.

    Dependencies
    ------------
    Parent class for:
        mcgs class

    Slots
    -----
        __ui: DICT: User input (ui) dict
        uigrid:  CLASS: ui: gridding parameters
        uisim:  CLASS: ui: simulation par
        uigsc:  CLASS: ui: grain strucure characterisation par
        uiint:  CLASS: ui: intervals
        uigsprop:  CLASS: ui: grain str property calculation par
        uigeorep:  CLASS: ui: geometric representations cacl par
        _mcsteps_:  LIST: stores history of mcsteps
        __g__:  DICT: base dict template for grains
        __gprop__:  DICT: base dict template for grain properties
        __gb__:  DICT: base dict template for grain boundaries
        __gbprop__:  DICT: base dict template for grain boundary properties
        g:  DICT: Grains @latest mcstep
        m: LIST: available temporal slices
        xgr: np.ndarray:
        ygr: np.ndarray:
        zgr: np.ndarray:
        NL_dict: dict: Specifies Non-Locality detasils
        px_length: Iterable: Side lengths of the pixel
        px_size:, Area or volume of the pixel
        S:  np.ndarray: State matrix
        sa: State martix modified enable fast consideration of
            Wrapped Boundary Condition
        vis: Stores instant of awrtwork class
        AIA: np.ndarray: Appended Index Array (@dev)
        AIA0: np.ndarray: Appended Index Array (@dev)
        AIA1: np.ndarray: Appended Index Array (@dev)
        xind: np.ndarray: xindices (3D only)
        yind: np.ndarray: yindices (3D only)
        zind: np.ndarray: zindices (3D only)
        xinda: np.ndarray: appended xindices (3D only)
        yinda: np.ndarray: appended yindices (3D only)
        zinda: np.ndarray: appended zindices (3D only)
        NLM_nd: np.ndarray: Non-Locality matrix
        NLM: np.ndarray: Non-locality matrix
        EAPGLB: PRIMARY GLOBAL Euler angle Definition -- state wise.
        EASGLB: SSECONDARY GLOBAL Euler angle definition -- state wise.
                Different from EAPGLB in that adjustments that happen to
                EAPGLB are carried out here and not in the proimary list.
                This is NOT available at the grid level but at the grain
                structure level.
        #####################################################################
        #####################################################################
    """
    characterization_ID = 0
    characterization_settings = None
    __slots__ = ('uigrid', 'uisim', 'uigsc', 'uiint', 'study',
                 'uigsprop', 'uimesh', 'uigeomrepr' '_mcsteps_',
                 'uidata_all', 'index', 'ndimg_label_pck',
                 '__ui', '__g__', '__gprop__', '__gb__', '__gbprop__',
                 'gs', 'xgr', 'ygr', 'zgr',
                 'NL_dict', 'px_length', 'px_size',
                 'S', 's', 'sa', 'AIA', 'AIA0', 'AIA1',
                 'xind' 'yind', 'zind', 'xinda', 'yinda', 'zinda',
                 'NLM_nd', 'NLM', 'EAPGLB', 'tex',
                 'tslices_with_prop', 'vis', 'vizstyles', 'display_messages',
                 '__info_message_display_level__', 'dt_dict'
                 )

    def __init__(self, study='independent', input_dashboard='input_dashboard.xls',
                 consider_NLM_b=False, consider_NLM_d=False, AR_teevrate=0, AR_GrainAxis="-45",
                 display_messages=True):
        self.study = study
        if study == 'independent':
            from upxo.interfaces.user_inputs.gather_user_inputs import load_uidata
            from upxo._sup.data_templates import dict_templates
            uidata_all = load_uidata(input_dashboard)
            self.uigrid = uidata_all['uigrid']
            self.uisim = uidata_all['uisim']
            self.uigsc = uidata_all['uigsc']
            self.uiint = uidata_all['uiint']
            self.uigsprop = uidata_all['uigsprop']
            self.uigeomrepr = uidata_all['uigeorep']
            self.uimesh = uidata_all['uimesh']
            self.__ui = uidata_all
            self.uidata_all = uidata_all
            self.dt_dict = dict_templates()
            self.initiate(consider_NLM_b=consider_NLM_b,
                          consider_NLM_d=consider_NLM_d,
                          AR_teevrate=AR_teevrate,
                          AR_GrainAxis=AR_GrainAxis,
                          display_messages=display_messages)
        elif study in ('para_sweep'):
            # Parameters to be manually set
            pass
        elif study == 'restart':
            pass

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        """
        Iterator to loop over grain structures at different mcsteps.

        Returns
        -------
        tuple
            A tuple containing the material and grain structure pair.
        """
        if self.index < len(self.gs.keys()):
            _m_grain_str_pair_ = list(self.gs.values())[self.index]
            self.index += 1
            return _m_grain_str_pair_
        else:
            raise StopIteration

    def __repr__(self):
        """
        Representation of the grain structure object.

        Returns
        -------
        str
            A string representation of the grain structure.
        """
        if self.uigrid.dim == 2:
            sep1 = 'UPXO 2D.MCGS\n'
        elif self.uigrid.dim == 3:
            sep1 = 'UPXO 3D.MCGS\n'
        GRID = "(A: GRID):: "
        if hasattr(self, 'uigrid'):
            GRID += f"  x:({self.uigrid.xmin},{self.uigrid.xmax},{self.uigrid.xinc}), "
            GRID += f"  y:({self.uigrid.ymin},{self.uigrid.ymax},{self.uigrid.yinc}), "
            GRID += f"  z:({self.uigrid.zmin},{self.uigrid.zmax},{self.uigrid.zinc})\n"
        else:
            GRID += 'Grid parameters not set.\n'
        # --------------------------------------
        SIMPAR = "(B: SIMPAR):: "
        if hasattr(self, 'uisim'):
            SIMPAR += f"  nstates: {self.uisim.S}  mcsteps: {self.uisim.mcsteps}"
            SIMPAR += f"  algorithms: {self.uisim.algo_hops}\n"
        else:
            SIMPAR += 'Simulation parameters not set.\n'
        # --------------------------------------
        MESHPAR = "(C: MESHPAR):: "
        if hasattr(self, 'uimesh'):
            MESHPAR += f"  GB Conformity: {self.uimesh.mesh_gb_conformity}\n"
            MESHPAR += ' '*15+f"Target FE Software: {self.uimesh.mesh_target_fe_software}"
            MESHPAR += f"  Element type: {self.uimesh.mesh_element_type}"
        else:
            MESHPAR += 'Mesh parameters not set yet.\n'
        # --------------------------------------
        return sep1 + GRID + SIMPAR + MESHPAR + '\n' + '-'*60

    def initiate(self, consider_NLM_b=False, consider_NLM_d=False,
                 AR_teevrate = 0, AR_GrainAxis = "-45", display_messages=True):
        """
        Initiate the grain structure object.

        Parameters
        ----------
        consider_NLM_b : bool, optional
            Whether to consider Non-Local Matrix of Booleans. The default is False.
        consider_NLM_d : bool, optional
            Whether to consider Non-Local Matrix of Distance measures. The default is False.
        AR_teevrate : float, optional
            Aspect ratio teev rate. The default is 0.
        AR_GrainAxis : str, optional
            Aspect ratio grain axis. The default is "-45".
        display_messages : bool, optional
            Whether to display messages. The default is True.
        """
        self._mcsteps_ = [self.uisim.S]
        # -------------------------------------------
        if self.uigrid.dim == 2:
            self.px_size = self.uigrid.xinc*self.uigrid.yinc
            self.px_length = (self.uigrid.xinc+self.uigrid.yinc)/2
        elif self.uigrid.dim == 3:
            self.vox_size = self.uigrid.xinc * self.uigrid.yinc * self.uigrid.xinc
            self.vox_length = (self.uigrid.xinc + self.uigrid.yinc + self.uigrid.zinc)/3
        # ----------------------------------------
        # Build original co-ordinate grSid
        self.build_original_coordinate_grid()
        # ----------------------------------------
        # Build original orientation state matrices
        self.build_original_state_matrix()
        self.m = list(np.arange(0, self.uisim.mcsteps, self.uiint.mcint_save_at_mcstep_interval, dtype='int'))
        # Temporarily initiate the tslices. It may get updated in case,
        # the grain growqth reaches fully_annealed codition !!
        self.tslices = list(np.arange(0, self.uisim.mcsteps, self.uiint.mcint_save_at_mcstep_interval, dtype='int'))
        # ----------------------------------------
        self.build_ea()
        # ----------------------------------------
        consider_NLM_b_flag, consider_NLM_d_flag = 'no', 'no'
        if consider_NLM_b:
            consider_NLM_b_flag = "yes"
        if consider_NLM_b:
            consider_NLM_d_flag = "yes"

        self.NL_dict = dict(NLM_b_dict=dict(flag=consider_NLM_b_flag),
                            NLM_d_dict=dict(flag=consider_NLM_d_flag,
                                            func="mexpan",
                                            par=(5., 5., 5., 5.,
                                                 0., 0., 0., 0.,
                                                 0., 0., 0., 0.),
                                            normflag="yes",),
                            ARdetails=dict(teevrate=AR_teevrate,
                                           GrainAxis=AR_GrainAxis),
                            )
        # ----------------------------------------
        # Calculate Non-Local Martix
        self.build_non_locality_matrix()
        # ----------------------------------------
        # Build appended index array
        self.AppIndArray()
        # ----------------------------------------
        # Square subset matrix
        # ssub = self.SquareSubsetMatrix()
        # ----------------------------------------
        from ..viz.artwork_definitions import artwork
        self.vis = artwork()
        self.vis.q_Col_Mat(self.uisim.S)
        # ----------------------------------------
        self.setup_transition_probability_rules()
        # self.vis.s_partitioned_tranition_probabilities(self.uisim.S, self.uisim.s_boltz_prob)
        # ----------------------------------------
        self.tslices_with_prop = []
        # ----------------------------------------
        self.vizstyles = self.dt_dict.vizstyles_mcgs()
        self.display_messages = display_messages

    def build_ea(self):
        """
        Build the Euler angle orientation data.

        Returns
        -------
        None.
        """
        ea1, ea2, ea3 = np.random.uniform([0, 0, 0], [360, 180, 360], (self.uisim.S, 3)).T
        self.EAPGLB = (ea1, ea2, ea3)

    def build_original_coordinate_grid(self):
        """
        This sets up the original coordinate grid.
        Original Coordinate Grid: DESCRIPTION
        OCG inputs
            1. dim  : Dimensionality of the grid
            2. xmin : Minimum x co-ordinate
            3. xmax : Maximum x co-ordinate
            4. xinc : x co-ordinate increment
            5. ymin : Minimum y co-ordinate
            6. ymax : Maximum y co-ordinate
            7. yinc : y co-ordinate increment
            8. zmin : Minimum z co-ordinate
            9. zmax : Maximum z co-ordinate
            10.zinc : z co-ordinate increment
        OCG outputs
            1. xgr  : x co-ordinate grid
            2. ygr  : y co-ordinate grid
            3. zgr  : z co-ordinate grid
        Returns
        -------
        None.
        """
        if self.uigrid.dim == 2:
            cogrid = np.meshgrid(np.arange(self.uigrid.xmin, self.uigrid.xmax+1, float(self.uigrid.xinc)),
                                 np.arange(self.uigrid.ymin, self.uigrid.ymax+1, float(self.uigrid.yinc)),
                                 copy=True, sparse=False, indexing='xy')
            self.xgr, self.ygr, self.zgr = cogrid[0], cogrid[1], 0
        # ----------------------------------------
        if self.uigrid.dim == 3:
            xmin, xmax = self.uigrid.xmin, self.uigrid.xmax
            ymin, ymax = self.uigrid.ymin, self.uigrid.ymax
            zmin, zmax = self.uigrid.zmin, self.uigrid.zmax
            xinc = self.uigrid.xinc
            yinc = self.uigrid.yinc
            zinc = self.uigrid.zinc
            xarr = np.arange(xmin, xmax, xinc)
            yarr = np.arange(ymin, ymax, yinc)
            zarr = np.arange(zmin, zmax, zinc)
            #nx = np.floor_divide(xmax-xmin, xinc)
            #ny = np.floor_divide(ymax-ymin, yinc)
            #nz = np.floor_divide(zmax-zmin, zinc)
            #self.xgr, self.ygr, self.zgr = np.mgrid[xmin:xmax:nx*1j,
            #                                        ymin:ymax:ny*1j,
            #                                        zmin:zmax:nz*1j]
            self.xgr, self.ygr, self.zgr = np.meshgrid(xarr, yarr, zarr)

    def build_original_state_matrix(self):
        """
        This sets up the Q-state matrix.

        Original State Matrix: DESCRIPTION
        OSM inputs
            1. S        : No. of orientation states
            2. OCG_Size : Size of the original
                          coordinate grid: a 3 element list
        OSM outputs
            1. S        : orientation state matrix
            2. S_sz0    : dim0 len of S
            3. S_sz1    : dim1 len of S
            4. S_sz2    : dim2 len of S
            5. Svec     : S in single row format. IS THIS STILL NEEDED???


        Returns
        -------
        None.

        """

        if self.uigrid.dim == 2:
            OCG_size = (self.xgr.shape[0], self.xgr.shape[1])
            # @ 2D grain structure
            if self.uisim.mcalg[0] not in ('4', '5'):
                self.S = np.random.randint(1, self.uisim.S+1, size=(OCG_size[0], OCG_size[1])).astype(int)
            else:
                self.S = np.random.randint(1, self.uisim.S+1, size=(OCG_size[0], OCG_size[1])).astype(np.float64)
        elif self.uigrid.dim == 3:
            OCG_size = (self.xgr.shape[0], self.xgr.shape[1], self.xgr.shape[2])
            # @ 3D grain structure
            self.S = np.random.randint(1, self.uisim.S+1, size=(OCG_size[0], OCG_size[1], OCG_size[2])).astype(int)

    def build_non_locality_matrix(self):
        """
        Construct the non-locality matrix used in some Monte-Carlo simulation.

        Returns
        -------
        None.
        """
        if self.uigrid.dim == 2:  # 2D GRAIN STRUCTURE
            # Calculate the size of non-local matrix
            NLM_sz0 = 2*self.uisim.NL+1
            NLM_sz1 = 2*self.uisim.NL+1
            # Calculate Non-Local Matrix of Booleans
            if self.NL_dict["NLM_b_dict"]['flag'] in set(['yes', 'y', '1']):
                NLM_bw = self.IntRules()
            else:
                NLM_bw = np.repeat([np.repeat([1.], NLM_sz0, 0)], NLM_sz1, 0)
            # Calculate Non-Local Matrix of Distance measures
            if self.NL_dict["NLM_d_dict"]['flag'] in set(['yes', 'y', '1']):
                NLM_d = self.NLM_dist()
            else:
                NLM_d = np.repeat([np.repeat([1.], NLM_sz0, 0)], NLM_sz1, 0)
            # Calculate the overall Non-Local Matrix
            self.NLM_nd = NLM_bw * NLM_d
            self.NLM = np.concatenate(self.NLM_nd)
        elif self.uigrid.dim == 3:  # 3D GRAIN STRUCTURE
            # Calculate the size of non-local matrix
            NLM_sz0 = 2*self.uisim.NL+1
            NLM_sz1 = 2*self.uisim.NL+1
            NLM_sz2 = 2*self.uisim.NL+1
            # Calculate Non-Local Matrix of Booleans
            if self.NL_dict["NLM_b_dict"]['flag'] in set(['yes', 'y', '1']):
                NLM_bw = self.IntRules()
            else:
                NLM_bw = np.repeat([np.repeat([np.repeat([1.], NLM_sz0, 0)], NLM_sz1, 0)],
                                   NLM_sz2, 0)
            # Calculate Non-Local Matrix of Distance measures
            if self.NL_dict["NLM_d_dict"]['flag'] in set(['yes', 'y', '1']):
                NLM_d = self.NLM_dist()
            else:
                NLM_d = np.repeat([np.repeat([np.repeat([1.], NLM_sz0, 0)],
                                             NLM_sz1, 0)], NLM_sz2, 0)
            # Calculate the overall Non-Local Matrix
            self.NLM_nd = NLM_bw * NLM_d
            self.NLM = np.concatenate(np.concatenate(self.NLM_nd))

    def IntRules(self):
        """
        Input arguments
        [*] ~~Kineticity~~
               Nature of partition evolution in Euclidean space
                   OPTIONS: all lower case
                       ("static", "s")
                       ("kinetic0", "k0") -- Default kinetic

        [*] ~~Dimensionality~~
            Number of fundamental axes of simulation Euclidean space
                OPTIONS: all lower case
                    (1d, 1)
                    (2d, 2)
                    (3d, 3)

        [*] ~~ARteevrate~~
            To describe the strength of partition aspect ratio
                OPTIONS: all float (following are examples)
                    0: Aims for equi-axed grains
                    1: Aims for non unit AR
                    2: Aims for a higher AR than 1
                    n: AR_(n) > AR_(n-1) > AR_0
                NOTE: Max value limited by no. of pxls across min Gr thickness

        [*] ~~NL~~
            Non-Locality parameter


        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        ARteevrate = self.NL_dict['ARdetails']['teevrate']
        GrainAxis = self.NL_dict['ARdetails']['GrainAxis']
        if self.uisim.kineticity in set(["static", "s",
                                         "sta", "kinetic",
                                         "k", "kin"]):
            if self.uisim.kineticity == "static":
                NLM_sz0 = 2*self.uisim.NL+1
                NLM_sz1 = 2*self.uisim.NL+1
                NLM_sz2 = 2*self.uisim.NL+1
                NLM_sz0 = 3
                NLM_sz1 = 3
                NLM_sz2 = 3
                if self.uigrid.dim == 2:
                    ones = np.repeat([np.repeat([1.], NLM_sz0, 0)], NLM_sz1, 0)
                    arts = np.repeat([np.repeat([float(ARteevrate)],
                                                NLM_sz0, 0)], NLM_sz1, 0)
                    if self.uisim.NL == 1:
                        # if ARteevrate == 0:
                        NLM_bw = ones + arts*np.array([[+1., +1., +1.],
                                                       [+1., +1., +1.],
                                                       [+1., +1., +1.]])
                        # elif ARteevrate == 1:
                        _vert_ = ['90', '270', 'V', 'vert', 'vertical']
                        _hor_ = ['0', '180', 'H', 'hor', 'horizontal']
                        if GrainAxis.lower() in set([string.lower()
                                                     for string in _vert_]):
                            NLM_bw = (ones + arts)*np.array([[+0., +0., +0.],
                                                             [+1., +0., +1.],
                                                             [+0., +0., +0.]])
                            print('========== 1 ==========')
                        elif GrainAxis.lower() in set([string.lower()
                                                       for string in ['+-45',
                                                                      'x']]):
                            NLM_bw = (ones + arts)*np.array([[+1., +0., +1.],
                                                             [+0., +0., +0.],
                                                             [+1., +0., +1.]])
                            print('========== 2 ==========')
                        elif GrainAxis.lower() in set([string.lower()
                                                       for string in ['+45',
                                                                      '45']]):
                            NLM_bw = (ones + arts)*np.array([[+1., +0., +0.],
                                                             [+0., +1., +0.],
                                                             [+0., +0., +1.]])
                            print('========== 3 ==========')
                        elif GrainAxis.lower() in set([string.lower()
                                                       for string in ['-45']]):
                            NLM_bw = (ones + arts)*np.array([[+1, +4., +1.],
                                                             [+4., +0, +4.],
                                                             [+1., +4., +1]])
                            print('========== 4 ==========')
                        elif GrainAxis.lower() in set([string.lower()
                                                       for string in _hor_]):
                            NLM_bw = (ones + arts)*np.array([[+0.0, +1., +0.0],
                                                             [+1.0, +0., +1.0],
                                                             [+0.0, +1., +0.0]])
                            print('========== 5 ==========')
                    elif self.uisim.NL == 2:
                        NLM_bw = ones+arts*np.array([[+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.]
                                                     ])
                    else:
                        None
                elif self.uigrid.dim == 3:
                    ones = np.ones((NLM_sz0, NLM_sz1, NLM_sz2))
                    arts = float(ARteevrate)*ones
                    arnlm = np.ones((NLM_sz0, NLM_sz1, NLM_sz2))
                    NLM_bw = (ones + arts)*arnlm
            elif self.uisim.kineticity == "kinetic":
                None
            return NLM_bw
        else:
            return print("Please input Kin")

    def AppIndArray(self):
        """
        This sets up appended index array. 

        Appended Index Array: DESCRIPTION
        AppIndArray inputs
            1. NL       : Non-Locality
            2. OCG_size : List of 3 values. Each is dim along axes 0, 1 and 2 respectively.      ~~~~TO BE DONE~~~~
        AppIndArray outputs
            1. AIA      : Appended Index Array (Matlab type array element numbers!) TO KEEP FOR THE MOMENT
            2. AIA0     : Appended dim0 Index Array
            3. AIA1     : Appended dim1 Index Array
            4. AIA2     : Appended dim2 Index Array     ~~~~TO BE DONE~~~~

        Returns
        -------
        None.

        Notes
        -----
        [*] Appended Index Array: An array that helps in quick lookup of
            indices of pixels/cells/voxels in the state matrix S when
            considering Non-Locality and Wrapped Boundary Conditions.

        [*] The Appended Index Array is built by appending rows and columns
            (and layers in 3D) to the original index array of the state matrix S
            based on the Non-Locality parameter NL. This allows for easy
            access to neighboring indices without complex boundary checks.
        
        [*] In 2D, the AIA is a 2D array where each element corresponds to
            the linear index of the state matrix S. The AIA0 and AIA1 are
            derived from the AIA to provide direct access to indices along
            each dimension.

        [*] In 3D, the AIA is replaced by a 3D state matrix 'sa' that includes
            additional layers to account for Non-Locality in all three dimensions, 
            and xinda, yinda, zinda standing for appended index arrays along each
            dimension.

        [*] This method is crucial for efficient simulation of grain growth
            processes, especially when considering interactions beyond immediate
            neighbors.

        [*] The implementation currently supports 2D and 3D grids, with the 3D
            implementation being more complex due to the additional dimension.

        [*] The method assumes that the state matrix S has already been initialized
            and that the Non-Locality parameter NL is defined in the simulation parameters.

        [*] This is crucial for performance optimization in Monte Carlo grain growth 
            simulations and is inspired by similar implementations in MATLAB from the 
            author's previous work PXO standing for 'Poly-Xtal Operations'. It is partly
            responsible for the speed of UPXO grain growth simulations, which is later 
            further enhanced using Numba JIT compilation in other parts of the code.

        Notes
        -----
        [*] Original implementation inspiration: MATLAB code from PXO (Poly-Xtal Operations)
        [*] Performance optimization: Numba JIT compilation in other parts of UPXO
        [*] Essential for efficient Monte Carlo grain growth simulations
        [*] Supports 2D and 3D grids with Non-Locality considerations
        [*] Facilitates quick index lookups in state matrix S
        [*] Enhances simulation speed and efficiency
        [*] Critical for handling Wrapped Boundary Conditions
        [*] DO NOT TOUCH WITHOUT DEEP UNDERSTANDING OF THE METHOD
        """
        if self.uigrid.dim == 2:
            OCG_size = (self.xgr.shape[0], self.xgr.shape[1])
            # Appended Index Array
            self.AIA = np.arange(np.prod(OCG_size)).reshape((OCG_size[0], OCG_size[1]))
            dim0 = np.arange(OCG_size[0])
            dim1 = np.arange(OCG_size[1])
            # Appended Index Array along dimensions 1 & 0:
            DIM1, DIM0 = np.meshgrid(dim0, dim1, copy=True, sparse=False, indexing='xy')
            for NLcount in range(0,  self.uisim.NL):
                # Left Edge
                LE = self.AIA[:, [2*NLcount]]
                # Right edge
                RE = self.AIA[:, [len(self.AIA[0]) - 1 - 2*NLcount]]
                self.AIA = np.concatenate((self.AIA, LE), axis=1)
                self.AIA = np.concatenate((RE, self.AIA), axis=1)
                # Top edge
                TE = self.AIA[[2*NLcount], :]
                # Bottom edge
                BE = self.AIA[[len(self.AIA) - 1 - 2*NLcount], :]
                self.AIA = np.concatenate((BE, self.AIA), axis=0)
                self.AIA = np.concatenate((self.AIA, TE), axis=0)
                # Left Edge
                LE_dim0 = DIM0[:, [2*NLcount]]
                # Right edge
                RE_dim0 = DIM0[:, [len(DIM0[0]) - 1 - 2*NLcount]]
                DIM0 = np.concatenate((DIM0, LE_dim0), axis=1)
                DIM0 = np.concatenate((RE_dim0, DIM0), axis=1)
                # Top edge
                TE_dim0 = DIM0[[2*NLcount], :]
                # Bottom edge
                BE_dim0 = DIM0[[len(DIM0) - 1 - 2*NLcount], :]
                DIM0 = np.concatenate((BE_dim0, DIM0), axis=0)
                DIM0 = np.concatenate((DIM0, TE_dim0), axis=0)
                # Left Edge
                LE_dim1 = DIM1[:, [2*NLcount]]
                # Right edge
                RE_dim1 = DIM1[:, [len(DIM1[0]) - 1 - 2*NLcount]]
                DIM1 = np.concatenate((DIM1, LE_dim1), axis=1)
                DIM1 = np.concatenate((RE_dim1, DIM1), axis=1)
                # Top edge
                TE_dim1 = DIM1[[2*NLcount], :]
                # Bottom edge
                BE_dim1 = DIM1[[len(DIM1) - 1 - 2*NLcount], :]
                DIM1 = np.concatenate((BE_dim1, DIM1), axis=0)
                DIM1 = np.concatenate((DIM1, TE_dim1), axis=0)
            self.AIA1 = DIM0.T
            self.AIA0 = DIM1.T
        elif self.uigrid.dim == 3:
            OCG_size = (self.xgr.shape[0], self.xgr.shape[1], self.xgr.shape[2])
            self.sa = np.zeros((OCG_size[0]+2, OCG_size[1]+2, OCG_size[2]+2))
            self.sa[1:-1, 1:-1, 1:-1] = self.S
            # ------------------------------------------------------------
            # FRONT FACE
            self.sa[0][1:-1, 1:-1] = self.S[-1]
            # BACK FACE
            self.sa[-1][1:-1, 1:-1] = self.S[0]
            # TOP FACE
            self.sa[1:-1, 0, 1:-1] = self.S[:, -1, :]
            # BOTTOM FACE
            self.sa[1:-1, -1, 1:-1] = self.S[:, 0, :]
            # LEFT FACE
            self.sa[1:-1, 1:-1, 0] = self.S[:, :, -1]
            # RIGHT FACE
            self.sa[1:-1, 1:-1, -1] = self.S[:, :, 0]
            # ------------------------------------------------------------
            # EDGE @FRONT and TOP
            self.sa[0, 0, 1:-1] = self.S[-1, -1, :]
            # EDGE @FRONT and BOTTOM
            self.sa[0, -1, 1:-1] = self.S[-1, 0, :]
            # EDGE @FRONT and LEFT
            self.sa[0, 1:-1, 0] = self.S[-1, :, -1]
            # EDGE @FRONT and RIGHT
            self.sa[0, 1:-1, -1] = self.S[-1, :, 0]
            # ------------------------------------------------------------
            # EDGE @BACK and TOP
            self.sa[-1, 0, 1:-1] = self.S[0, -1, :]
            # EDGE @BACK and BOTTOM
            self.sa[-1, -1, 1:-1] = self.S[0, 0, :]
            # EDGE @BACK and LEFT
            self.sa[-1, 1:-1, 0] = self.S[0, :, -1]
            # EDGE @BACK and RIGHT
            self.sa[-1, 1:-1, -1] = self.S[0, :, 0]
            # ------------------------------------------------------------
            # EDGE @TOP and LEFT
            self.sa[1:-1, 0, 0] = self.S[:, -1, -1]
            # EDGE @TOP and RIGHT
            self.sa[1:-1, 0, 0] = self.S[:, -1, -1]
            # EDGE @BOTTOM and LEFT
            self.sa[1:-1, -1, 0] = self.S[:, 0, -1]
            # EDGE @BOTTOM and RIGHT
            self.sa[1:-1, -1, -1] = self.S[:, 0, 0]
            # ------------------------------------------------------------
            # VERTEX @FRONT-LEFT-TOP FACES
            self.sa[0, 0, 0] = self.S[-1, -1, -1]
            # VERTEX @FRONT-LEFT-BOTTOM FACES
            self.sa[0, -1, 0] = self.S[-1, 0, -1]
            # VERTEX @FRONT-RIGHT-BOTTOM FACES
            self.sa[0, -1, -1] = self.S[-1, 0, 0]
            # VERTEX @FRONT-RIGHT-TOP FACES
            self.sa[0, 0, -1] = self.S[-1, -1, 0]
            # ------------------------------------------------------------
            # VERTEX @BACK-LEFT-TOP FACES
            self.sa[-1, 0, 0] = self.S[0, -1, -1]
            # VERTEX @BACK-LEFT-BOTTOM FACES
            self.sa[-1, -1, 0] = self.S[0, 0, -1]
            # VERTEX @BACK-RIGHT-BOTTOM FACES
            self.sa[-1, -1, -1] = self.S[0, 0, 0]
            # VERTEX @FRONT-RIGHT-TOP FACES
            self.sa[-1, 0, -1] = self.S[0, -1, 0]
            # ------------------------------------------------------------
            self.xind = np.zeros((OCG_size[0], OCG_size[1], OCG_size[2]), dtype=int)
            self.yind = np.zeros((OCG_size[0], OCG_size[1], OCG_size[2]), dtype=int)
            self.zind = np.ones((OCG_size[0], OCG_size[1], OCG_size[2]), dtype=int)
            # ------------------------------------------------------------
            tempx = np.tile(np.arange(OCG_size[0]), (OCG_size[0], 1))
            for xaxiscount in range(OCG_size[0]):
                self.xind[xaxiscount] = tempx
            tempy = np.tile(np.array([np.arange(OCG_size[1])]).T,
                            (1, OCG_size[1]))
            for yaxiscount in range(OCG_size[1]):
                self.yind[yaxiscount] = tempy
            for zaxiscount in range(OCG_size[2]):
                self.zind[zaxiscount] = float(zaxiscount)*self.zind[zaxiscount]
            # ------------------------------------------------------------
            self.xinda = np.zeros((OCG_size[0]+2, OCG_size[1]+2, OCG_size[2]+2), dtype=int)
            self.yinda = np.zeros((OCG_size[0]+2, OCG_size[1]+2, OCG_size[2]+2), dtype=int)
            self.zinda = np.zeros((OCG_size[0]+2, OCG_size[1]+2, OCG_size[2]+2), dtype=int)
            # ------------------------------------------------------------
            self.xinda[1:-1, 1:-1, 1:-1] = self.xind
            # FRONT FACE
            self.xinda[0][1:-1, 1:-1] = self.xind[-1]
            # BACK FACE
            self.xinda[-1][1:-1, 1:-1] = self.xind[0]
            # TOP FACE
            self.xinda[1:-1, 0, 1:-1] = self.xind[:, -1, :]
            # BOTTOM FACE
            self.xinda[1:-1, -1, 1:-1] = self.xind[:, 0, :]
            # LEFT FACE
            self.xinda[1:-1, 1:-1, 0] = self.xind[:, :, -1]
            # RIGHT FACE
            self.xinda[1:-1, 1:-1, -1] = self.xind[:, :, 0]
            # EDGE @FRONT and TOP
            self.xinda[0, 0, 1:-1] = self.xind[-1, -1, :]
            # EDGE @FRONT and BOTTOM
            self.xinda[0, -1, 1:-1] = self.xind[-1, 0, :]
            # EDGE @FRONT and LEFT
            self.xinda[0, 1:-1, 0] = self.xind[-1, :, -1]
            # EDGE @FRONT and RIGHT
            self.xinda[0, 1:-1, -1] = self.xind[-1, :, 0]
            # EDGE @BACK and TOP
            self.xinda[-1, 0, 1:-1] = self.xind[0, -1, :]
            # EDGE @BACK and BOTTOM
            self.xinda[-1, -1, 1:-1] = self.xind[0, 0, :]
            # EDGE @BACK and LEFT
            self.xinda[-1, 1:-1, 0] = self.xind[0, :, -1]
            # EDGE @BACK and RIGHT
            self.xinda[-1, 1:-1, -1] = self.xind[0, :, 0]
            # EDGE @TOP and LEFT
            self.xinda[1:-1, 0, 0] = self.xind[:, -1, -1]
            # EDGE @TOP and RIGHT
            self.xinda[1:-1, 0, 0] = self.xind[:, -1, -1]
            # EDGE @BOTTOM and LEFT
            self.xinda[1:-1, -1, 0] = self.xind[:, 0, -1]
            # EDGE @BOTTOM and RIGHT
            self.xinda[1:-1, -1, -1] = self.xind[:, 0, 0]
            # VERTEX @FRONT-LEFT-TOPFACES
            self.xinda[0, 0, 0] = self.xind[-1, -1, -1]
            # VERTEX @FRONT-LEFT-BOTFACES
            self.xinda[0, -1, 0] = self.xind[-1, 0, -1]
            # VERTEX @FRONT-RIGHT-BOTFACES
            self.xinda[0, -1, -1] = self.xind[-1, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.xinda[0, 0, -1] = self.xind[-1, -1, 0]
            # VERTEX @BACK-LEFT-TOP FACES
            self.xinda[-1, 0, 0] = self.xind[0, -1, -1]
            # VERTEX @BACK-LEFT-BOTFACES
            self.xinda[-1, -1, 0] = self.xind[0, 0, -1]
            # VERTEX @BACK-RIGHT-BOTFACES
            self.xinda[-1, -1, -1] = self.xind[0, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.xinda[-1, 0, -1] = self.xind[0, -1, 0]
            # ------------------------------------------------------------
            self.yinda[1:-1, 1:-1, 1:-1] = self.yind
            # FRONT FACE
            self.yinda[0][1:-1, 1:-1] = self.yind[-1]
            # BACK FACE
            self.yinda[-1][1:-1, 1:-1] = self.yind[0]
            # TOP FACE
            self.yinda[1:-1, 0, 1:-1] = self.yind[:, -1, :]
            # BOTTOM FACE
            self.yinda[1:-1, -1, 1:-1] = self.yind[:, 0, :]
            # LEFT FACE
            self.yinda[1:-1, 1:-1, 0] = self.yind[:, :, -1]
            # RIGHT FACE
            self.yinda[1:-1, 1:-1, -1] = self.yind[:, :, 0]
            # EDGE @FRONT and TOP
            self.yinda[0, 0, 1:-1] = self.yind[-1, -1, :]
            # EDGE @FRONT and BOTTOM
            self.yinda[0, -1, 1:-1] = self.yind[-1, 0, :]
            # EDGE @FRONT and LEFT
            self.yinda[0, 1:-1, 0] = self.yind[-1, :, -1]
            # EDGE @FRONT and RIGHT
            self.yinda[0, 1:-1, -1] = self.yind[-1, :, 0]
            # EDGE @BACK and TOP
            self.yinda[-1, 0, 1:-1] = self.yind[0, -1, :]
            # EDGE @BACK and BOTTOM
            self.yinda[-1, -1, 1:-1] = self.yind[0, 0, :]
            # EDGE @BACK and LEFT
            self.yinda[-1, 1:-1, 0] = self.yind[0, :, -1]
            # EDGE @BACK and RIGHT
            self.yinda[-1, 1:-1, -1] = self.yind[0, :, 0]
            # EDGE @TOP and LEFT
            self.yinda[1:-1, 0, 0] = self.yind[:, -1, -1]
            # EDGE @TOP and RIGHT
            self.yinda[1:-1, 0, 0] = self.yind[:, -1, -1]
            # EDGE @BOTTOM and LEFT
            self.yinda[1:-1, -1, 0] = self.yind[:, 0, -1]
            # EDGE @BOTTOM and RIGHT
            self.yinda[1:-1, -1, -1] = self.yind[:, 0, 0]
            # VERTEX @FRONT-LEFT-TOP FACES
            self.yinda[0, 0, 0] = self.yind[-1, -1, -1]
            # VERTEX @FRONT-LEFT-BOTFACES
            self.yinda[0, -1, 0] = self.yind[-1, 0, -1]
            # VERTEX @FRONT-RIGHT-BOTFACES
            self.yinda[0, -1, -1] = self.yind[-1, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.yinda[0, 0, -1] = self.yind[-1, -1, 0]
            # VERTEX @BACK-LEFT-TOP FACES
            self.yinda[-1, 0, 0] = self.yind[0, -1, -1]
            # VERTEX @BACK-LEFT-BOTFACES
            self.yinda[-1, -1, 0] = self.yind[0, 0, -1]
            # VERTEX @BACK-RIGHT-BOTFACES
            self.yinda[-1, -1, -1] = self.yind[0, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.yinda[-1, 0, -1] = self.yind[0, -1, 0]
            # ------------------------------------------------------------
            self.zinda[1:-1, 1:-1, 1:-1] = self.zind
            # FRONT FACE
            self.zinda[0][1:-1, 1:-1] = self.zind[-1]
            # BACK FACE
            self.zinda[-1][1:-1, 1:-1] = self.zind[0]
            # TOP FACE
            self.zinda[1:-1, 0, 1:-1] = self.zind[:, -1, :]
            # BOTTOM FACE
            self.zinda[1:-1, -1, 1:-1] = self.zind[:, 0, :]
            # LEFT FACE
            self.zinda[1:-1, 1:-1, 0] = self.zind[:, :, -1]
            # RIGHT FACE
            self.zinda[1:-1, 1:-1, -1] = self.zind[:, :, 0]
            # EDGE @FRONT and TOP
            self.zinda[0, 0, 1:-1] = self.zind[-1, -1, :]
            # EDGE @FRONT and BOTTOM
            self.zinda[0, -1, 1:-1] = self.zind[-1, 0, :]
            # EDGE @FRONT and LEFT
            self.zinda[0, 1:-1, 0] = self.zind[-1, :, -1]
            # EDGE @FRONT and RIGHT
            self.zinda[0, 1:-1, -1] = self.zind[-1, :, 0]
            # EDGE @BACK and TOP
            self.zinda[-1, 0, 1:-1] = self.zind[0, -1, :]
            # EDGE @BACK and BOTTOM
            self.zinda[-1, -1, 1:-1] = self.zind[0, 0, :]
            # EDGE @BACK and LEFT
            self.zinda[-1, 1:-1, 0] = self.zind[0, :, -1]
            # EDGE @BACK and RIGHT
            self.zinda[-1, 1:-1, -1] = self.zind[0, :, 0]
            # EDGE @TOP and LEFT
            self.zinda[1:-1, 0, 0] = self.zind[:, -1, -1]
            # EDGE @TOP and RIGHT
            self.zinda[1:-1, 0, 0] = self.zind[:, -1, -1]
            # EDGE @BOTTOM and LEFT
            self.zinda[1:-1, -1, 0] = self.zind[:, 0, -1]
            # EDGE @BOTTOM and RIGHT
            self.zinda[1:-1, -1, -1] = self.zind[:, 0, 0]
            # VERTEX @FRONT-LEFT-TOP FACES
            self.zinda[0, 0, 0] = self.zind[-1, -1, -1]
            # VERTEX @FRONT-LEFT-BOTFACES
            self.zinda[0, -1, 0] = self.zind[-1, 0, -1]
            # VERTEX @FRONT-RIGHT-BOTFACES
            self.zinda[0, -1, -1] = self.zind[-1, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.zinda[0, 0, -1] = self.zind[-1, -1, 0]
            # VERTEX @BACK-LEFT-TOP FACES
            self.zinda[-1, 0, 0] = self.zind[0, -1, -1]
            # VERTEX @BACK-LEFT-BOTFACES
            self.zinda[-1, -1, 0] = self.zind[0, 0, -1]
            # VERTEX @BACK-RIGHT-BOTFACES
            self.zinda[-1, -1, -1] = self.zind[0, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.zinda[-1, 0, -1] = self.zind[0, -1, 0]

    def SquareSubsetMatrix(self):
        """
        This returns the square subset matrix

        SquareSubsetMatrix: DESCRIPTION
        SquareSubsetMatrix inputs:
            1. NL: Non-Locality parameter
        SquareSubsetMatrix outputs:
            1. ssub


        Returns
        -------
        ssub : TYPE
            DESCRIPTION.

        """
        ss_sz0 = 2*self.uisim.NL+1  # S matrix Subset SiZe axis 0, row
        ss_sz1 = 2*self.uisim.NL+1  # S matrix Subset SiZe axis 1, col
        ssub = np.zeros((ss_sz0, ss_sz1), dtype=float)
        return ssub

    def add_gs_data_structure_template(self, m=None, dim=None, study='independent'):
        """
        Add grain statistics data structure template.

        Parameters
        ----------
        m : int, optional
            State number. The default is None.
        dim : int, optional
            Dimension of the microstructure. The default is None.
        study : str, optional
            Type of study. The default is 'independent'.
        """
        from upxo.pxtal.mcgs2_temporal_slice import mcgs2_grain_structure as _GS_
        if study == 'independent':
            if m == 0:
                self.gs = {m: _GS_(m=m, dim=dim,
                                   px_size=self.px_size if dim==2 else self.vox_size,
                                   S_total=self.uisim.S, xgr=self.xgr, ygr=self.ygr,
                                   uidata=self.__ui, uigrid=self.uigrid,
                                   uimesh=self.uimesh, EAPGLB=self.EAPGLB)}
            else:
                # THIS BRNACH NEVER REACHED ANYWAYS. TO BE DEPRECATED !!
                self.gs[m] = _GS_(m=m, dim=dim,
                                  px_size=self.px_size,
                                  S_total=self.uisim.S, xgr=self.xgr, ygr=self.ygr,
                                  uidata=self.__ui, uigrid=self.uigrid,
                                  uimesh=self.uimesh, EAPGLB=self.EAPGLB)
        elif study == 'parameter_sweep':
            xgr, ygr, npixels = self.uigrid.grid
            if m == 0:
                self.gs = {m: _GS_(m=m, dim=self.uigrid.dim,
                                   uidata=self.__ui, px_size=self.uigrid.px_size,
                                   S_total=self.uisim.S, xgr=self.xgr, ygr=self.ygr,
                                   uigrid=self.uigrid, EAPGLB=self.EAPGLB)}
            else:
                self.gs[m] = _GS_(m=m, dim=self.uigrid.dim,
                                  uidata=self.__ui, px_size=self.px_size,
                                  S_total=self.uisim.S, xgr=xgr, ygr=ygr,
                                  uigrid=self.uigrid, EAPGLB=self.EAPGLB)

    def _setup_grain_properties_dict_(self):
        """
        Store grain properties
            * areas: Areas (pixels) of all grains: s partitioned
            * Centroids of all grains: s partitioned
            * Neighbouring grain IDs (immediate ones)
            * IDs of immediate neighbours and IDs of neighbours of
              neighbouring grains

        Returns
        -------
        None.
        """
        self.__gprop__ = dict(areas=None, centroids=None, neigh_1=[], neigh_2=[[]],)
        self.gprop = {0: self.__gprop__}

    def _setup_grainboundaries_dict_(self):
        """
        Store grain boundaries
            * Grain boundary numbers used as ID
            * Compulsory: list of list of IDs of all grain boundaries
            * State wise partitioning
            * Grain boundary vertices (NOT grain boundary points)

        Returns
        -------
        None.
        """
        self.__gb__ = dict(ids=None, ind=None, spart=None, vert=None,)
        self.gb = {0: self.__gb__}

    def _setup_grainboundaries_properties_dict_(self):
        """
        Store grain boundary propeties
            * Non-pixel form of total length
            * Total length calculated from pixel side lengths
            * Total lengths of all straight lines between grain
              boundary vertices
            * Total lengths of all boundary segments between grain
              boundary vertices
            * IDs of shared grains
            * Grain boundary zone

        Returns
        -------
        None.
        """
        self.__gbprop__ = dict(length_curve=[], length_pixels=[], lengths_straight=[],
                               lengths=[], shared_grains=[], gbz=None, )
        self.gbprop = {0: self.__gbprop__}

    def setup_transition_probability_rules(self):
        """
        Set up transition probability rules and estimate T.P

        Returns
        -------
        None.
        """
        if self.uisim.s_boltz_prob == 'q_unrelated':
            '''
            Generate Boltzmann probabilities unrelated to Q
            Using: P = exp(-kbf*a)
            where a is random number between 0 and 1 for each state in S
            and kbf is boltzmann_temp_factor_max
            0 < a < 1
            0 < kbf < inf
            Thus, 0 < P < 1
            '''
            _a_ = np.random.random(size=self.simpar.S)
            kbf = self.uisim.boltzmann_temp_factor_max
            self.uisim.s_boltz_prob = np.exp(-kbf*_a_)
        elif self.uisim.s_boltz_prob == 'q_related':
            '''
            Generate Boltzmann probabilities related to Q
            Using: P = exp(-kbf*a)
            where a is scaled array of state numbers in S
            and kbf is boltzmann_temp_factor_max
            0 < a < boltzmann_temp_factor_max
            Thus, 0 < P < 1
            '''
            _a_ = np.arange(self.uisim.S)
            _a_ = self.uisim.boltzmann_temp_factor_max*_a_/_a_.max()
            _ = np.random.random(size=self.uisim.S)
            self.uisim.s_boltz_prob = np.exp(-_a_*_)

    def detect_grains(self, mcsteps=None, kernel_order=2, library='scikit-image',
                      connectivity=26, store_state_ng=True,
                      process_individual_states=False,
                      delta=0, lfiDtype=np.int32,
                      verbose=False):
        '''
        Detect grains in microstructure images using specified image processing
        library.
        This method identifies and segments grains in two-dimensional (2D) or
        three-dimensional (3D) microstructure images based on the provided
        temporal slices (mcsteps), using either OpenCV or scikit-image
        libraries for 2D images, and a SciLab-based approach for 3D images.

        Parameters
        ----------
        mcsteps : int or iterable of int, optional
            Specifies the temporal slices to analyze. If not provided, all available
            temporal slices are used.
            Each temporal slice corresponds to a unique microstructure state.
        kernel_order : int, optional
            Specifies the connectivity for grain identification. A higher order
            increases the connectivity considered during grain detection.
            Defaults to 2, indicating 8-connectivity in 2D and 26-connectivity in 3D.
        store_state_ng : bool, optional
            If True, stores the state of newly generated grains. Defaults to True.
        library : str, optional
            Specifies the library to use for grain identification in 2D. Supported
            libraries are 'opencv' and 'scikit-image'. If not specified, the default
            library set in the user's ImageGrainSize Configuration (uigsc) is used.
            UPDATE: 27/12/2025. Use of 'cc3d' for library is implemented.
        connectivity : int, optional
            Specifies the connectivity for 2D and 3D grain identification when using
            'scikit-image' or 'cc3d' libraries. Defaults to 26 for 3D images.
            NOTE: This parameter is placed to replace kernel_order in future updates.
            It is operational only when library is 'cc3d'.

        Raises
        ------
        TypeError
            If `mcsteps` is neither an int nor an iterable of int.
        ValueError
            If `mcsteps` contains values not available in the temporal slices or if
            the specified `library` is not supported for the operation.

        Returns
        -------
        None
            Modifies the object's state by updating the grain segmentation
            information for the specified temporal slices.
        '''
        mcsteps_available = list(self.tslices)  # All available temporal slices
        # ---------------------------------------------
        if not mcsteps:
            # no value entered, use all available values in tslices
            mcsteps = mcsteps_available
        # ---------------------------------------------
        if not isinstance(mcsteps, int) and not hasattr(mcsteps, '__iter__'):
            raise TypeError("mcsteps must be an int or an iterable.")
        # ---------------------------------------------
        if type(mcsteps) == int:
            if mcsteps in mcsteps_available:
                # Single value has been entered for mcsteps
                mcsteps = [mcsteps]
            else:
                # Single value entered and is not one of the available slices
                raise ValueError(f"mcsteps={mcsteps} not avaialble."
                                 f"Value must be from {self.tslices}")
        # ---------------------------------------------
        for _ in mcsteps:
            if _ not in self.tslices:
                ValueError(f"mcstep={_} not avaialble."
                           f"Value must be from {self.tslices}")
        # ---------------------------------------------
        if not library:
            library = self.uigsc.grain_identification_library
        # ---------------------------------------------
        if self.uigrid.dim == 2:
            if library == 'upxo':
                print('upxo grain detection has been deprecated..')
            elif library in dth.opt.ocv_options + dth.opt.ski_options + dth.opt.cc3d_options:
                print(f"Using {library} for grain identification")
                from upxo.pxtalops import detect_grains_from_mcstates as get_grains
                self.gs, state_ng = get_grains.mcgs2d(library=library, gs_dict=self.gs,
                                        msteps=mcsteps, kernel_order=kernel_order,
                                        store_state_ng=store_state_ng,
                                        connectivity=connectivity,
                                        process_individual_states=process_individual_states,
                                        delta=delta, lfiDtype=lfiDtype,
                                        verbose=verbose)
            else:
                raise ValueError("Required library should be in, "
                                 f"{dth.opt.ocv_options + dth.opt.ski_options + dth.opt.cc3d_options}"
                                 f". Receeived: {library}")
        # ---------------------------------------------
        if self.uigrid.dim == 3:
            for m in mcsteps:
                if m in mcsteps_available:
                    self.find_grains_scilab_ndimage_3d(m)
                else:
                    print(f'MC temporal slice no {m} invalid. Skipped')

    def detect_grains_v2(self):
        pass

    def set_characterization_settings_2d(self, setid=-1):
        """
        Set predefined morphological characterization settings for 2D microstructure
        images.

        Parameters
        ----------
        setid : int, optional
            Predefined setting ID. The default is -1, which corresponds to no
            characterization. Other options are:
                - -1: Very basic characterization with only npixels calculated.
                - 1: Basic characterization with number of pixels and
                     neighboring grain identification.
                - 2: Standard characterization with bounding box, area,
                     equivalent diameter, compactness, solidity, and
                     neighboring grain identification.
                - 3: Comprehensive characterization with bounding box,
                     equivalent diameter, compactness, aspect ratio, solidity,
                     morphological orientation, circularity, eccentricity,
                     major and minor axis lengths, Euler number, grain positions,
                     neighboring grain identification, and skim properties.

        Raises
        ------
        ValueError
            If `setid` is not -1, 1, 2, or 3.
        """
        self.characterization_flag = setid
        if setid == -1:
            characterization_settings = dict(bbox=False, bbox_ex=False, npixels=True,
                                             npixels_gb=False, area=False, eq_diameter=False,
                                             perimeter=False, perimeter_crofton=False,
                                             compactness=False, gb_length_px=False, aspect_ratio=False,
                                             solidity=False, morph_ori=False, circularity=False,
                                             eccentricity=False, feret_diameter=False,
                                             major_axis_length=False, minor_axis_length=False,
                                             euler_number=False,
                                             char_grain_positions=False, find_neigh=False,
                                             char_gb=False, make_skim_prop=False,
                                             get_grain_coords=False)
        if setid == 1:
            characterization_settings = dict(bbox=False, bbox_ex=False, npixels=True,
                                             npixels_gb=False, area=False, eq_diameter=False,
                                             perimeter=False, perimeter_crofton=False,
                                             compactness=False, gb_length_px=False, aspect_ratio=False,
                                             solidity=False, morph_ori=False, circularity=False,
                                             eccentricity=False, feret_diameter=False,
                                             major_axis_length=False, minor_axis_length=False,
                                             euler_number=False,
                                             char_grain_positions=False, find_neigh=True,
                                             char_gb=False, make_skim_prop=False,
                                             get_grain_coords=False)
        elif setid == 2:
            characterization_settings = dict(bbox=True, bbox_ex=True, npixels=False,
                                             npixels_gb=False, area=True, eq_diameter=True,
                                             perimeter=False, perimeter_crofton=False,
                                             compactness=True, gb_length_px=False, aspect_ratio=False,
                                             solidity=True, morph_ori=False, circularity=False,
                                             eccentricity=False, feret_diameter=False,
                                             major_axis_length=False, minor_axis_length=False,
                                             euler_number=False,
                                             char_grain_positions=False, find_neigh=True,
                                             char_gb=False, make_skim_prop=True,
                                             get_grain_coords=True)
        elif setid == 3:
            characterization_settings = dict(bbox=True, bbox_ex=True, npixels=False,
                                             npixels_gb=False, area=False, eq_diameter=True,
                                             perimeter=False, perimeter_crofton=False,
                                             compactness=True, gb_length_px=False, aspect_ratio=True,
                                             solidity=True, morph_ori=True, circularity=True,
                                             eccentricity=True, feret_diameter=False,
                                             major_axis_length=True, minor_axis_length=True,
                                             euler_number=True,
                                             char_grain_positions=True, find_neigh=True,
                                             char_gb=False, make_skim_prop=True,
                                             get_grain_coords=True)
        else:
            raise ValueError('setid must be -1, 1, 2, or 3.')
        self.characterization_settings = characterization_settings

    def char_morph_2d(self, M, use_characterization_settings=False, use_version=1,
                      bbox=True, bbox_ex=True, npixels=False,
                      npixels_gb=False, area=True, eq_diameter=True,
                      perimeter=True, perimeter_crofton=False,
                      compactness=True, gb_length_px=False, aspect_ratio=False,
                      solidity=True, morph_ori=False, circularity=False,
                      eccentricity=False, feret_diameter=True,
                      major_axis_length=False, minor_axis_length=False,
                      euler_number=False, append=False, saa=True, throw=False,
                      char_grain_positions=False, find_neigh=True,
                      char_gb=False, make_skim_prop=True,
                      get_grain_coords=True):
        """
        Perform morphological characterization of grains in 2D microstructure images.

        Parameters
        ----------
        M : int or iterable of int
            Temporal slice(s) to characterize.
        use_characterization_settings : bool, optional
            If True, use predefined characterization settings. The default is False.
        bbox : bool, optional
            Calculate bounding box. The default is True.
        bbox_ex : bool, optional
            Calculate extended bounding box. The default is True.
        npixels : bool, optional
            Calculate number of pixels in each grain. The default is False.
        npixels_gb : bool, optional
            Calculate number of pixels in each grain boundary. The default is False.
        area : bool, optional
            Calculate area of each grain. The default is True.
        eq_diameter : bool, optional
            Calculate equivalent diameter of each grain. The default is True.
        perimeter : bool, optional
            Calculate perimeter of each grain. The default is True.
        perimeter_crofton : bool, optional
            Calculate perimeter using Crofton method. The default is False.
        compactness : bool, optional
            Calculate compactness of each grain. The default is True.
        gb_length_px : bool, optional
            Calculate grain boundary length in pixels. The default is False.
        aspect_ratio : bool, optional
            Calculate aspect ratio of each grain. The default is False.
        solidity : bool, optional
            Calculate solidity of each grain. The default is True.
        morph_ori : bool, optional
            Calculate morphological orientation of each grain. The default is False.
        circularity : bool, optional
            Calculate circularity of each grain. The default is False.
        eccentricity : bool, optional
            Calculate eccentricity of each grain. The default is False.
            The default is False.
        feret_diameter : bool, optional
            Calculate Feret diameter of each grain. The default is True.
        major_axis_length : bool, optional
            Calculate major axis length of each grain. The default is False.
        minor_axis_length : bool, optional
            Calculate minor axis length of each grain. The default is False.
        euler_number : bool, optional
            Calculate Euler number of each grain. The default is False.
        append : bool, optional
            If True, append new properties to existing ones. The default is False.
        saa : bool, optional
            If True, use spatially aware algorithms. The default is True.
        throw : bool, optional
            If True, throw an error if characterization fails. The default is False.
        char_grain_positions : bool, optional
            If True, characterize grain positions. The default is False.
        find_neigh : bool, optional
            If True, find neighboring grains. The default is True.
        char_gb : bool, optional
            If True, characterize grain boundaries. The default is False.
        make_skim_prop : bool, optional
            If True, create skim properties. The default is True.
        get_grain_coords : bool, optional
            If True, get grain coordinates. The default is True.

        Returns
        -------
        None
            Modifies the object's state by updating the morphological properties
            of grains for the specified temporal slice(s).
        """

        if type(M) == int and M in self.m:
            print(40*'=')
            message = f'Characterizing tslice: {M}. Characteriser version {use_version}'
            print(f'|----- {message} -----|')
            self.gs[M].char_morph_2d(use_characterization_settings=False, use_version=use_version,
                                     bbox=bbox, bbox_ex=bbox_ex, npixels=npixels,
                                     npixels_gb=npixels_gb, area=area,
                                     eq_diameter=eq_diameter, perimeter=perimeter,
                                     perimeter_crofton=perimeter_crofton,
                                     compactness=compactness, gb_length_px=gb_length_px,
                                     aspect_ratio=aspect_ratio, solidity=solidity,
                                     morph_ori=morph_ori, circularity=circularity,
                                     eccentricity=eccentricity, feret_diameter=feret_diameter,
                                     major_axis_length=major_axis_length,
                                     minor_axis_length=minor_axis_length,
                                     euler_number=euler_number, append=append, saa=saa,
                                     throw=throw, char_grain_positions=char_grain_positions,
                                     find_neigh=find_neigh, char_gb=char_gb,
                                     make_skim_prop=make_skim_prop,
                                     get_grain_coords=get_grain_coords)
            self.tslices_with_prop.append(M)
            print(f'....... |-|  tslice: {M}: COMPLETE |-|')
            print(40*'=')
        elif type(M) == int and M not in self.m:
            print('Please enter valid temporal slice from PXGS.m')
        elif type(M) in dth.dt.ITERABLES:
            for tslice in M:
                if tslice in self.m:
                    message = f'Characterizing tslice: {tslice}. Characteriser version {use_version}'
                    print(f'|----- {message} -----|')
                    self.gs[tslice].char_morph_2d(use_characterization_settings=False, use_version=use_version,
                                                  bbox=bbox, bbox_ex=bbox_ex, npixels=npixels,
                                                  npixels_gb=npixels_gb, area=area,
                                                  eq_diameter=eq_diameter, perimeter=perimeter,
                                                  perimeter_crofton=perimeter_crofton,
                                                  compactness=compactness, gb_length_px=gb_length_px,
                                                  aspect_ratio=aspect_ratio, solidity=solidity,
                                                  morph_ori=morph_ori, circularity=circularity,
                                                  eccentricity=eccentricity, feret_diameter=feret_diameter,
                                                  major_axis_length=major_axis_length,
                                                  minor_axis_length=minor_axis_length,
                                                  euler_number=euler_number, append=append, saa=saa,
                                                  throw=throw, char_grain_positions=char_grain_positions,
                                                  find_neigh=find_neigh, char_gb=char_gb,
                                                  make_skim_prop=make_skim_prop,
                                                  get_grain_coords=get_grain_coords
                                                  )
                    self.tslices_with_prop.append(tslice)
                else:
                    print(f"tslice = {tslice} not in `PXGS.m`. |-| Ignored |-|")

    def find_grain_areas_fast(self, tslices):
        """
        Quickly find the grain areas without doing anything else.

        Explanations
        ------------
        Order of grain_areas is that of pxtal.gs[m].gid

        Parameters
        ----------
        tslices : int or iterable of int
            Temporal slice(s) to process.

        Return
        ------
        grain_areas : dict
            Dictionary with tslice as key and numpy array of grain areas as value.
        """
        grain_areas = {tslice: None for tslice in tslices}
        print(40*'#')
        for tslice in tslices:
            print(40*'-', f'\n Extracting grain areas. tslice={tslice}')
            grain_areas_tslice = []
            for gid in self.gs[tslice].gid:
                if gid % 100 == 0:
                    print(f'{gid} grains completed.')
                grain_areas_tslice.append(np.where(self.gs[tslice].lgi == gid)[0].size)
            grain_areas[tslice] = np.array(grain_areas_tslice)
        print(40*'#')
        return grain_areas

    def find_npixels_border_grains_fast(self, tslices):
        """
        Quickly find the number of pixels in border grains without doing anything else.

        Parameters
        ----------
        tslices : int or iterable of int
            Temporal slice(s) to process.

        Return
        ------
        border_grain_npixels : dict
            Dictionary with tslice as key and numpy array of number of pixels in border grains as value
        """
        if type(tslices) not in dth.ITERABLES:
            if type(tslices) in dth.NUMBERS:
                tslices = [tslices]
            else:
                raise TypeError('Invalid tslices type specification.')
        border_grain_npixels = {tslice: None for tslice in tslices}
        for tslice in tslices:
            border_grain_npixels[tslice] = self.gs[tslice].find_npixels_border_grains_fast()
        return border_grain_npixels

    def find_npixels_internal_grains_fast(self, tslices):
        """
        Quickly find the number of pixels in internal grains without doing anything else.

        Parameters
        ----------
        tslices : int or iterable of int
            Temporal slice(s) to process.

        Return
        ------
        internal_grain_npixels : dict
            Dictionary with tslice as key and numpy array of number of pixels in internal grains as value
        """
        if type(tslices) not in dth.ITERABLES:
            if type(tslices) in dth.NUMBERS:
                tslices = [tslices]
            else:
                raise TypeError('Invalid tslices type specification.')
        internal_grain_npixels = {tslice: None for tslice in tslices}
        for tslice in tslices:
            internal_grain_npixels[tslice] = self.gs[tslice].find_npixels_internal_grains_fast()
        return internal_grain_npixels

    def find_border_internal_grains_fast(self, tslices):
        """
        Quickly find the border and internal grains without doing anything else.
        Parameters
        ----------
        tslices : int or iterable of int
            Temporal slice(s) to process.

        Return
        ------
        border_gids : dict
            Dictionary with tslice as key and numpy array of border grain IDs as value
        internal_gids : dict
            Dictionary with tslice as key and numpy array of internal grain IDs as value
        lgi_border : dict
            Dictionary with tslice as key and numpy array of lgi of border grains as value
        lgi_internal : dict
            Dictionary with tslice as key and numpy array of lgi of internal grains as value
        """
        if type(tslices) not in dth.ITERABLES:
            tslices = [tslices]
        border_gids = {tslice: None for tslice in tslices}
        internal_gids = {tslice: None for tslice in tslices}
        lgi_border = {tslice: None for tslice in tslices}
        lgi_internal = {tslice: None for tslice in tslices}
        for tslice in tslices:
             a, b, c, d = self.gs[tslice].find_border_internal_grains_fast()
             border_gids[tslice], internal_gids[tslice] = a, b
             lgi_border[tslice], lgi_internal[tslice] = c, d
        return border_gids, internal_gids, lgi_border, lgi_internal

    def hist(self, tslices=None, PROP_NAMES=None, bins=None, kdes=None,
             kdes_bw=None, stats=None, peaks=False, height=0, prominance=0.2,
             auto_xbounds=True, auto_ybounds=True, xbounds=[0, 50], ybounds=[0, 0.2]):
        """
        Plot histograms for specified temporal slices and grain properties.

        Parameters
        ----------
        tslices : int or iterable of int, optional
            Temporal slice(s) to plot histograms for. If not provided, histograms
            will be plotted for all temporal slices with available properties.
        PROP_NAMES : str or iterable of str, optional
            Grain property/properties to plot histograms for. If not provided,
            'npixels' will be used by default.
        bins : int or iterable of int, optional
            Number of bins for the histogram. If not provided, default bin size
            from vizstyles will be used.
        kdes : bool or iterable of bool, optional
            Whether to plot kernel density estimates (KDEs). If not provided,
            KDEs will not be plotted by default.
        kdes_bw : float or iterable of float, optional
            Bandwidth adjustment for KDEs. If not provided, default bandwidth
            will be used.
        stats : str or iterable of str, optional
            Statistic to plot ('density', 'count', etc.). If not provided,
            'density' will be used by default.
        peaks : bool, optional
            Whether to mark peaks on the histogram. Default is False.
        height : float, optional
            Height threshold for peak detection. Default is 0.
        prominance : float, optional
            Prominence threshold for peak detection. Default is 0.2.
        auto_xbounds : bool, optional
            Whether to automatically determine x-axis bounds. Default is True.
        auto_ybounds : bool, optional
            Whether to automatically determine y-axis bounds. Default is True.
        xbounds : list of float, optional
            Manual x-axis bounds if auto_xbounds is False. Default is [0, 50].
        ybounds : list of float, optional
            Manual y-axis bounds if auto_ybounds is False. Default is [0, 0.2].

        Returns
        -------
        None
            Plots histograms for the specified temporal slices and grain properties.
        """
        # Validate and normalize inputs
        tslices = self._validate_tslices(tslices)
        if tslices is None:
            print("Invalid inputs. No histogram computed. Skipped")
            return
        
        PROP_NAMES = self._normalize_prop_names(PROP_NAMES, len(tslices))
        bins = self._normalize_bins(bins, len(tslices))
        kdes, kdes_bw = self._normalize_kdes(kdes, kdes_bw, len(tslices))
        stats = self._normalize_stats(stats, len(tslices))
        
        # Plot histograms for each temporal slice
        self._plot_histograms(
            tslices, PROP_NAMES, bins, kdes, kdes_bw, stats,
            peaks, height, prominance, auto_xbounds, auto_ybounds, xbounds, ybounds
        )
    
    def _validate_tslices(self, tslices):
        """Validate and return list of temporal slices with available properties."""
        if tslices is None:
            tslices_available = [gs.m for gs in self if gs.are_properties_available]
            return tslices_available if tslices_available else None
        
        if isinstance(tslices, int):
            return [tslices]
        
        if type(tslices) in dth.dt.ITERABLES:
            return tslices
        
        print("Invalid tslice datatype. Skipped")
        return None
    
    def _normalize_prop_names(self, prop_names, n_tslices):
        """Normalize property names to list format."""
        if prop_names is None:
            return ['npixels']
        
        if isinstance(prop_names, str):
            return [prop_names for _ in range(n_tslices)]
        
        if type(prop_names) not in dth.dt.ITERABLES:
            print("Invalid PROP_NAME. Considering npixels by default")
            return ['npixels']
        
        return prop_names
    
    def _normalize_bins(self, bins, n_tslices):
        """Normalize bins parameter to list of integers."""
        default_bins = self.vizstyles['bins']
        
        if bins is None:
            return [default_bins for _ in range(n_tslices)]
        
        if type(bins) in dth.dt.NUMBERS:
            return [bins for _ in range(n_tslices)]
        
        if type(bins) in dth.dt.ITERABLES:
            if len(bins) == n_tslices:
                return [
                    bin_val if type(bin_val) in dth.dt.NUMBERS else default_bins
                    for bin_val in bins
                ]
        
        return [default_bins for _ in range(n_tslices)]
    
    def _normalize_kdes(self, kdes, kdes_bw, n_tslices):
        """Normalize KDE parameters to lists."""
        if kdes is None:
            return [False] * n_tslices, [None] * n_tslices
        
        if isinstance(kdes, bool):
            kdes_list = [kdes] * n_tslices
            kdes_bw_list = self._normalize_kdes_bw(kdes_bw, n_tslices)
            return kdes_list, kdes_bw_list
        
        return kdes, kdes_bw
    
    def _normalize_kdes_bw(self, kdes_bw, n_tslices):
        """Normalize KDE bandwidth parameter to list."""
        if kdes_bw is None or (type(kdes_bw) not in dth.dt.NUMBERS and 
                                type(kdes_bw) not in dth.dt.ITERABLES):
            return [None] * n_tslices
        
        if type(kdes_bw) in dth.dt.NUMBERS:
            return [kdes_bw] * n_tslices
        
        if type(kdes_bw) in dth.dt.ITERABLES:
            if len(kdes_bw) == n_tslices:
                return [
                    bw if type(bw) in dth.dt.NUMBERS else None
                    for bw in kdes_bw
                ]
            else:
                print("len(kdes_bw) must be same as len(kdes)")
                print("Current set of kdes_bw will result in an error!")
                print("Please enter valid data.")
        
        return [None] * n_tslices
    
    def _normalize_stats(self, stats, n_tslices):
        """Normalize stats parameter to list."""
        if stats is None:
            return ['density'] * n_tslices
        
        if isinstance(stats, str):
            return [stats] * n_tslices
        
        if type(stats) not in dth.dt.ITERABLES:
            print("Invalid stats datatype. Considering density by default")
            return ['density'] * n_tslices
        
        return stats
    
    def _plot_histograms(self, tslices, prop_names, bins, kdes, kdes_bw, stats,
                         peaks, height, prominance, auto_xbounds, auto_ybounds, 
                         xbounds, ybounds):
        """Plot histograms for validated inputs."""
        for idx, tslice in enumerate(tslices):
            if tslice not in self.tslices:
                print(f"Invalid tslice: {tslice}. Skipped")
                continue
            
            if not self.gs[tslice].are_properties_available:
                print(f"Properties have not been calculated for tslice: {tslice}. Skipped")
                continue
            
            for prop_name in prop_names:
                if prop_name not in self.gs[tslice].prop.columns:
                    print(f"PROP_NAME: {prop_name} does not exist at tslice: {tslice}. Skipped")
                    continue
                
                self.gs[tslice].hist(
                    PROP_NAME=prop_name,
                    bins=bins[idx],
                    kde=kdes[idx],
                    stat=stats[idx],
                    color=self.vizstyles['hist_colors_fill'],
                    edgecolor=self.vizstyles['hist_colors_edge'],
                    alpha=self.vizstyles['hist_colors_fill_alpha'],
                    bw_adjust=kdes_bw[idx],
                    line_kws={
                        'color': self.vizstyles['kde_color'],
                        'lw': self.vizstyles['kde_thickness'],
                        'ls': '-',
                    },
                    peaks=peaks,
                    height=height,
                    auto_xbounds=auto_xbounds,
                    auto_ybounds=auto_ybounds,
                    xbounds=xbounds,
                    ybounds=ybounds,
                    prominance=prominance,
                    __stack_call__=True,
                    __tslice__=tslice
                )


    def find_grains_scilab_ndimage_3d(self, m):
        """
        Find grains in 3D microstructure images using SciPy's ndimage.label function.

        Parameters
        ----------
        m : int
            Temporal slice number to process.
        """
        GrStruct = self.gs[m]
        from scipy.ndimage import label as spndimg_label
        _S_ = GrStruct.s
        for i, _s_ in enumerate(np.unique(_S_)):
            # Mark the presence of this state
            GrStruct.spart_flag[_s_] = True
            # Recognize the grains belonging to this state
            bin_img = (_S_ == _s_).astype(np.uint8)
            labels, num_labels = spndimg_label(bin_img)
            if i == 0:
                GrStruct.lgi = labels
            else:
                labels[labels > 0] += GrStruct.lgi.max()
                GrStruct.lgi = GrStruct.lgi + labels
            GrStruct.s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))
            GrStruct.s_n[_s_-1] = len(GrStruct.s_gid[_s_])
        # Get the total number of grains
        GrStruct.n = np.unique(GrStruct.lgi).size
        # Generate and store the gid-s mapping
        GrStruct.gid = list(range(1, GrStruct.n+1))
        _gid_s_ = []
        for _gs_, _gid_ in zip(GrStruct.s_gid.keys(), GrStruct.s_gid.values()):
            if _gid_:
                for __gid__ in _gid_:
                    _gid_s_.append(_gs_)
            else:
                pass
                # _gid_s_.append(0)  # Splcing this temporarily. Retain if fully successfull.
        GrStruct.gid_s = _gid_s_
        # Make the output string to print on promnt
        optput_string_01 = f'Temporal slice number = {m}.'
        optput_string_02 = f' |||| No. of grains detected = {GrStruct.n}'
        print(optput_string_01 + optput_string_02)

    @decorators.port_doc('upxo.viz.gsviz', 'see_mcgs2d_features')
    def plotgs(self, M=[0, 4, 8, 12, 16], cmap='jet', figsize=(5,5),
               cbtick_incr=2, mbar=True, mbar_length=10, mbar_loc='bot_left'):
        """
        Plot 2D grain structure features.
        """
        from upxo.viz.gsviz import see_mcgs2d_features
        see_mcgs2d_features(M=M, mcgs2d_upxo=self, cmap=cmap, figsize=figsize,
                           cbtick_incr=cbtick_incr, mbar=mbar, mbar_length=mbar_length,
                           mbar_loc=mbar_loc)

    @property
    def pxtal_length(self):
        # Length of the pixelated crystal in the x-direction
        return self.uigrid.xmax-self.uigrid.xmin

    @property
    def pxtal_height(self):
        # Height of the pixelated crystal in the y-direction
        return self.uigrid.ymax-self.uigrid.ymin

    @property
    def pxtal_area(self):
        # Area of the pixelated crystal
        return self.pxtal_length*self.pxtal_height
# ---------------------------------------------------------------------


class mcgs(grid):
    """
    Monte-Carlo Grain Structure class
    """
    def __init__(self, study='independent', input_dashboard='input_dashboard.xls',
                 info_message_display_level='detailed',
                 consider_NLM_b=False, consider_NLM_d=False, AR_factor=0,
                 AR_GrainAxis="-45", display_messages=True):
        super().__init__(study=study, input_dashboard=input_dashboard,
                         consider_NLM_b=consider_NLM_b, consider_NLM_d=consider_NLM_d,
                         AR_teevrate=AR_factor, AR_GrainAxis=AR_GrainAxis,
                         display_messages=display_messages)

    def __str__(self):
        """
        String representation of the Monte-Carlo Grain Structure object.
        """
        str_1 = f"x:({self.uigrid.xmin},{self.uigrid.xmax},{self.uigrid.xinc}), "
        str_2 = f"y:({self.uigrid.ymin},{self.uigrid.ymax},{self.uigrid.yinc}), "
        str_3 = f"z:({self.uigrid.zmin},{self.uigrid.zmax},{self.uigrid.zinc})."
        return str_1+str_2+str_3

    def __att__(self):
        """
        Attribute representation of the Monte-Carlo Grain Structure object.
        """
        return gops.att(self)

    def simulate(self, rsfso=2, user_LIWM=False, LIWM=np.ones((3,3), dtype=np.float32), verbose=True):
        """
        Perform Monte-Carlo simulation to generate grain structures.
        
        Parameters
        ----------
        rsfso : int, optional
            Radius scaling factor for second order neighbors. Default is 2.
        user_LIWM : bool, optional
            Flag to indicate if user-defined LIWM is provided. Default is False.
        LIWM : numpy array, optional
            Local interaction weight matrix. Default is a 3x3 matrix of ones.
        verbose : bool, optional
            Flag to control verbosity of output. Default is True.
        """
        print('\n','Initiating Monte-Carlo simulation')
        print(f'     xmin, xmax, xinc: {self.uigrid.xmin}, {self.uigrid.xmax}, {self.uigrid.xinc}')
        print(f'     ymin, ymax, yinc: {self.uigrid.ymin}, {self.uigrid.ymax}, {self.uigrid.yinc}')
        print(f'     zmin, zmax, zinc: {self.uigrid.zmin}, {self.uigrid.zmax}, {self.uigrid.zinc}')
        print(f'     No. of states: {self.uisim.S}')
        print(f'     Dimensionality: {self.uigrid.dim}')
        # print(f'     Algorithm: {self.uisim.mcalg}', '\n')
        self.algo_hop = False
        # Initiate the grain-structure data-structure
        self.add_gs_data_structure_template(m=0,
                                            dim=self.uigrid.dim,
                                            study=self.study
                                            )
        # START THE MONTE-CARLO SIMULATIONS
        if self.uigrid.dim == 2 and len(self.uisim.algo_hops) == 1:
            self.algo_hop = False
            self.start_algo2d_without_hops(rsfso=rsfso, user_LIWM=user_LIWM, LIWM=LIWM)
        elif self.uigrid.dim == 2 and len(self.uisim.algo_hops) > 1:
            if self.algo_hop:
                self.start_algo2d_with_hops(verbose=verbose, user_LIWM=user_LIWM, LIWM=LIWM)
            else:
                self.start_algo2d_without_hops(rsfso=rsfso, user_LIWM=user_LIWM, LIWM=LIWM)
        elif self.uigrid.dim == 3:
            from scipy.ndimage import label as spndimg_label
            self.ndimg_label_pck = spndimg_label
            if len(self.uisim.algo_hops) == 1:
                self.algo_hop = False
                self.start_algo3d_without_hops(rsfso=rsfso, verbose=verbose, user_LIWM=user_LIWM, LIWM=LIWM)
            elif len(self.uisim.algo_hops) > 1:
                if self.algo_hop:
                    self.start_algo3d_with_hops(verbose=verbose, user_LIWM=user_LIWM, LIWM=LIWM)
                else:
                    self.start_algo3d_without_hops(rsfso=rsfso, verbose=verbose, user_LIWM=user_LIWM, LIWM=LIWM)

    def start_algo2d_without_hops(self, rsfso=2, user_LIWM=False, LIWM=np.ones((3,3), dtype=np.float32)):
        """
        Start the 2D Monte-Carlo simulation algorithm without hops.
        
        Parameters
        ----------
        rsfso : int, optional
            Radius scaling factor for second order neighbors. Default is 2.
        user_LIWM : bool, optional
            Flag to indicate if user-defined LIWM is provided. Default is False.
        LIWM : numpy array, optional
            Local interaction weight matrix. Default is a 3x3 matrix of ones.
        """
        _a, _b, _c = self.build_NLM()  # Unpack 3 rows of NLM
        # print('-----------------------')
        # print(_a)
        # print(_b)
        # print(_c)
        # print('-----------------------')
        _dm_ = self.display_messages
        # print(f'UPXO MC Algorithm ID: {self.uisim.mcalg}')
        # print(type(self.uisim.mcalg))
        if self.uisim.mcalg in ('200', '200.0'):
            import upxo.algorithms.alg200 as alg200
            self.gs, fully_annealed = alg200.run(xgr=self.xgr, ygr=self.ygr, zgr=self.zgr,
                                                 px_size=self.px_size, S=self.S,
                                                 _a=_a, _b=_b, _c=_c,
                                                 user_LIWM=user_LIWM, LIWM=LIWM,
                                                 AIA0=self.AIA0, AIA1=self.AIA1,
                                                 uisim=self.uisim, uiint=self.uiint, uidata=self.uidata_all,
                                                 uigrid=self.uigrid, display_messages=_dm_, )
        elif self.uisim.mcalg in ('201', '201.0'):
            import upxo.algorithms.alg201 as alg201
            self.gs, fully_annealed = alg201.run(xgr=self.xgr, ygr=self.ygr, zgr=self.zgr,
                                                 rsfso=rsfso, px_size=self.px_size, S=self.S,
                                                 _a=_a, _b=_b, _c=_c,
                                                 user_LIWM=user_LIWM, LIWM=LIWM,
                                                 AIA0=self.AIA0, AIA1=self.AIA1,
                                                 uisim=self.uisim, uiint=self.uiint, uidata=self.uidata_all,
                                                 uigrid=self.uigrid, display_messages=_dm_, )
        elif self.uisim.mcalg in ('202', '202.0'):
            print("    weighted (: ALG-202)")
            import upxo.algorithms.alg202 as alg202
            self.gs, fully_annealed = alg202.run(xgr=self.xgr, ygr=self.ygr, zgr=self.zgr,
                                                 px_size=self.px_size, S=self.S,
                                                 _a=_a, _b=_b, _c=_c,
                                                 user_LIWM=user_LIWM, LIWM=LIWM,
                                                 AIA0=self.AIA0, AIA1=self.AIA1,
                                                 uisim=self.uisim, uiint=self.uiint, uidata=self.uidata_all,
                                                 uigrid=self.uigrid, display_messages=_dm_, )
        # Update the tslices to accommodate the fully_annealed condition, if
        # it is achieved during simulation
        if fully_annealed['fully_annealed']:
            self.tslices = list(self.gs.keys())

    def start_algo2d_with_hops(self, user_LIWM=False, LIWM=np.ones((3,3), dtype=np.float32)):
        pass

    def start_algo3d_without_hops(self, rsfso=3, user_LIWM=False, LIWM=np.ones((3,3,3), dtype=np.float32), verbose=True):
        """
        Start the 3D Monte-Carlo simulation algorithm without hops.
        
        Parameters
        ----------
        rsfso : int, optional
            Radius scaling factor for second order neighbors. Default is 3.
        user_LIWM : bool, optional
            Flag to indicate if user-defined LIWM is provided. Default is False.
        LIWM : numpy array, optional
            Local interaction weight matrix. Default is a 3x3x3 matrix of ones.
        verbose : bool, optional
            Flag to control verbosity of output. Default is True.
        """
        if self.uisim.mcalg == '300a':
            print("Using ALG-300a")
            import upxo.algorithms.alg300a as alg300a
            print('////////////////////////////////')
            _alg300a_ = alg300a.mc_iterations_3d_alg300a
            self.gs, fully_annealed = _alg300a_(S=self.S, xinda=self.xinda, yinda=self.yinda, zinda=self.zinda,
                                                vox_size=self.vox_size, uidata=self.uidata_all, uigrid=self.uigrid,
                                                uisim=self.uisim, uiint=self.uiint, verbose=verbose,
                                                ndimg_label_pck=self.ndimg_label_pck)
        elif self.uisim.mcalg == '300b':
            print("Using ALG-300b")
            import upxo.algorithms.alg300b as alg300b
            print('////////////////////////////////')
            _alg300b_ = alg300b.mc_iterations_3d_alg300b
            self.gs, fully_annealed = _alg300b_(S=self.S, xinda=self.xinda, yinda=self.yinda, zinda=self.zinda,
                                                vox_size=self.vox_size, uidata=self.uidata_all, uigrid=self.uigrid,
                                                uisim=self.uisim, uiint=self.uiint, verbose=verbose,
                                                ndimg_label_pck=self.ndimg_label_pck)
        elif self.uisim.mcalg == '301.0':
            print("Using ALG-301")
            import upxo.algorithms.alg301 as alg301
            print('////////////////////////////////')
            _alg301_ = alg301.mc_iterations_3d_alg301
            self.gs, fully_annealed = _alg301_(S=self.S, xinda=self.xinda, yinda=self.yinda, zinda=self.zinda,
                                               vox_size=self.vox_size, uidata=self.uidata_all,
                                               uigrid=self.uigrid, uisim=self.uisim, uiint=self.uiint,
                                               verbose=verbose, ndimg_label_pck=self.ndimg_label_pck)
        elif self.uisim.mcalg == '302.0':
            print("Using ALG-302")
            import upxo.algorithms.alg302 as alg302
            print('////////////////////////////////')
            _alg302_ = alg302.mc_iterations_3d_alg302
            self.gs, fully_annealed = _alg302_(S=self.S, xinda=self.xinda, yinda=self.yinda, zinda=self.zinda,
                                               rsfso=rsfso, vox_size=self.vox_size, uidata=self.uidata_all,
                                               uigrid=self.uigrid, uisim=self.uisim, uiint=self.uiint,
                                               verbose=verbose, ndimg_label_pck=self.ndimg_label_pck)

    def build_NLM(self):
        """
        Build the Non-Locality Matrix (NLM) based on the non-locality level.

        Returns
        -------
        NLM : numpy array
            The constructed Non-Locality Matrix.
        """
        if self.uisim.NL == 1:
            NLM_00 = self.NLM_nd[0, 0]
            NLM_01 = self.NLM_nd[0, 1]
            NLM_02 = self.NLM_nd[0, 2]

            NLM_10 = self.NLM_nd[1, 0]
            NLM_11 = self.NLM_nd[1, 1]
            NLM_12 = self.NLM_nd[1, 2]

            NLM_20 = self.NLM_nd[2, 0]
            NLM_21 = self.NLM_nd[2, 1]
            NLM_22 = self.NLM_nd[2, 2]

            NLM = np.array([[NLM_00, NLM_01, NLM_02],
                            [NLM_10, NLM_11, NLM_12],
                            [NLM_20, NLM_21, NLM_22]])
        elif self.uisim.NL == 2:
            NLM_00 = self.NLM_nd[0, 0]
            NLM_01 = self.NLM_nd[0, 1]
            NLM_02 = self.NLM_nd[0, 2]
            NLM_03 = self.NLM_nd[0, 3]
            NLM_04 = self.NLM_nd[0, 4]

            NLM_10 = self.NLM_nd[1, 0]
            NLM_11 = self.NLM_nd[1, 1]
            NLM_12 = self.NLM_nd[1, 2]
            NLM_13 = self.NLM_nd[1, 3]
            NLM_14 = self.NLM_nd[1, 4]

            NLM_20 = self.NLM_nd[2, 0]
            NLM_21 = self.NLM_nd[2, 1]
            NLM_22 = self.NLM_nd[2, 2]
            NLM_23 = self.NLM_nd[2, 3]
            NLM_24 = self.NLM_nd[2, 4]

            NLM_30 = self.NLM_nd[3, 0]
            NLM_31 = self.NLM_nd[3, 1]
            NLM_32 = self.NLM_nd[3, 2]
            NLM_33 = self.NLM_nd[3, 3]
            NLM_34 = self.NLM_nd[3, 4]

            NLM_40 = self.NLM_nd[4, 0]
            NLM_41 = self.NLM_nd[4, 1]
            NLM_42 = self.NLM_nd[4, 2]
            NLM_43 = self.NLM_nd[4, 3]
            NLM_44 = self.NLM_nd[4, 4]

            NLM = np.array([[NLM_00, NLM_01, NLM_02, NLM_03, NLM_04],
                            [NLM_10, NLM_11, NLM_12, NLM_13, NLM_14],
                            [NLM_20, NLM_21, NLM_22, NLM_23, NLM_24],
                            [NLM_30, NLM_31, NLM_32, NLM_33, NLM_34],
                            [NLM_40, NLM_41, NLM_42, NLM_43, NLM_44]])
        return NLM

    def NLM_elements(self):
        # Build the Non-Locality Matrix
        _a, _b, _c = self.build_NLM()  # Unpack 3 rows of NLM
        NLM_00, NLM_01, NLM_02 = _a  # Unpack 3 colms of 1st row
        NLM_10, NLM_11, NLM_12 = _b  # Unpack 3 colms of 2nd row
        NLM_20, NLM_21, NLM_22 = _c  # Unpack 3 colms of 3rd row
        return NLM_00, NLM_01, NLM_02, NLM_10, NLM_11, NLM_12, NLM_20, NLM_21, NLM_22
