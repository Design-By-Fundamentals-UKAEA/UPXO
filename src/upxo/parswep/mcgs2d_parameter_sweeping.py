class parameter_sweep():
    """
    This is a core UPXO class. Use to generate and work with multiple mcgs
    using a combination of various generating and controlling parameters.

    Targetted at:
    -------------
        * research: understanding algorithm sensitivity
        * research: understanding parameter sensitivity
        * research: understanding grain growth evolution
        * development: UPXO

    Data structures:
    ----------------
        * Each PXTAl object is stored in a DICT under the instance number key.
        * Most properties are stored as Pandas dataframes.
        * The user input parameters are preserved under self.uiAAAAA,
        where, AAAAA is the additional name string of the corresponsing
        parameter set.
        * Mesh instances are stored seperately as a DICT under the instance
        number key, followed by a nested DICT under the mcstep key.

    Data attributes:
    ----------------
    domain: Size of the domain
    dim: dimensionality
    gmp: global morphological parameters
    qmp: Q-partitioned morphological parameters
    purge_previous: Purge previous study objects
    save_sims: BOOL: flag to pickle raw databases of simulations
    __GENLOCK__: str: flag to lock simulation capability of UPXO

    __GENLOCK__:
    ------------
        If 'locked', no simulations will be performed and grain structures
        will not be developed. Otherwise, if 'open'. Status set to 'locked'
        whenever following cases:
            * when user input validation fails.
            * when computations branch towards any lock imposed by developer
            in UPXO internals.
    """
    __slots__ = ('N', 'nstates', 'domain', 'dim',
                 'algo_hop', 'algo_hops',
                 'purge_previous', '_save_sims_',
                 'gsi',
                 'gmp', 'qmp',
                 'mesh_instances',
                 )
    __paramater_gsi_mapping_behaviour__ = 'one-many'
    __default_mcalg__ = '200'
    # ------------------------------------------------------------------
    '''
    NOTE: @ all locks :: Locked if True, Open if False

    __GS_GEN_LOCK__: GS generation lock.

    __GR_IDNT_LOCK__: Grain identification lock
    __GSMORPH_CHAR_LOCK__: Morphology characterisation lock
    __GS_ANALYSIS_LOCK__: Grain structure analysis lock
    __TEX_SAM_LOCK__: Texture sampling lock
    __TEX_MAP_LOCK__: Texture mapping lock
    __FE_GEN_LOCK__: Finite element mesh generation lock
    __FE_EXP_LOCK__: Finite element mesh export lock
    __REP_QUAL_LOCK__: Representativeness qualification lock
    __FERES_MAP_LOCK__: Finite element result map lock
    __FERES_ANALYSIS_LOCK__: Finite element result analysis lock
    '''
    __GS_GEN_LOCK__, __GR_IDNT_LOCK__ = [True, True], [True, True]
    __GSMORPH_CHAR_LOCK__, __GS_ANALYSIS_LOCK__ = [True, True], [True, True]
    __TEX_SAM_LOCK__, __TEX_MAP_LOCK__ = [True, True], [True, True]
    __FE_GEN_LOCK__, __FE_EXP_LOCK__ = [True, True], [True, True]
    __REP_QUAL_LOCK__ = [True, True]
    __FERES_MAP_LOCK__, __FERES_ANALYSIS_LOCK__ = [True, True], [True, True]
    # ------------------------------------------------------------------
    __GS_GEN_LOCK__ = [{'uigrid_type': True, 'uigrid_range': True,
                        'uisim_type': True, 'uisim_range': True,
                        'uigsc_type': True, 'uigsc_range': True,
                        'uimesh_type': True, 'uimesh_range': True,
                        'uigeomrepr_type': True, 'uigeomrepr_range': True, },
                       {'uigrid_type': True, 'uigrid_range': True,
                        'uisim_type': True, 'uisim_range': True,
                        'uigsc_type': True, 'uigsc_range': True,
                        'uimesh_type': True, 'uimesh_range': True,
                        'uigeomrepr_type': True, 'uigeomrepr_range': True, },
                       ]
    # ------------------------------------------------------------------
    def __init__(self,
                 use_default_values=False,
                 study='gs_analysis'):
        """
        Instantiates parameter sweep data-structure and perform user
        requested tasks

        Parameters
        ----------
        use_default_values : BOOL, optional
            If True, data-structure will be instantiated with default values
            for parameters of grid, simulation, grain structure analysis,
            meshing, geometric representations, etc. The default is False.
            If False, empty data-structure will be created. In which case,
            the user will have to seperately use the set_param_{ABCD} methods.

        study : srt, optional
            - If 'gs_analysis': All functionalities necessary for
            gs_analysis shall remain unlocked by default. All higher
            functionalities remain locked by default.
            - If 'gs_mesh': All functionalities necessary for grain
            structure meshing including the mesh export remain uncloked by
            default.
            - If 'gs_growth': All functionalities necessary for grain growth
            study remain unlocked by default.
            - If 'cpfe_data_analysis': All functionalities necessary for
            gs_analysis remain unlocked by default.
            - If '_development_': All functionalities necessary for
            gs_analysis remain unlocked by default.
            NOTE-1:
                User input validity may change the corrsponding sub-lock
            states.
            NOTE-2:
                Available options are contained in dth.opt.ps_studies

        Returns
        -------
        None.

        """
        # Initiate gsi=None to enable __repr__ when not assign_default_values.
        __gsi_index__ = None
        if use_default_values:
            self.initialize(N=2)
            # ----------------------------------------------------------------
            self.set_param_grid(domain_size=((0, 100),
                                             (0, 100),
                                             (0, 0),
                                             1),
                                read_from_file=False,
                                filename=None)
            for i, _ in enumerate(self.N):
                a = self.gsi[i].uisim.__uigrid_type_lock__
                b = self.gsi[i].uisim.__uigrid_range_lock__
                self.__GS_GEN_LOCK__[i]['uigrid_type'] = a
                self.__GS_GEN_LOCK__[i]['uigrid_range'] = b
            # ----------------------------------------------------------------
            self.set_param_sim(mcsteps=20,
                               nstates=32,
                               solver='python',
                               tgrad=None,
                               algo_hop=False,
                               algo_hops=[(200, 10),
                                          (201, 40),
                                          (202, 100)],
                               default_mcalg=self.default_mcalg,
                               save_at_mcsteps=np.linspace(0, 20, 5),
                               purge_previous=False,
                               read_from_file=False,
                               filename=None)
            self.set_param_gsc(char_grains=True,
                               char_stage='postsim',
                               library='scikit-image',
                               parallel=True,
                               find_gbseg=True,
                               g_area=True,
                               gb_length=True,
                               ofton=True,
                               gb_njp_order=True,
                               g_eq_dia=True,
                               g_feq_dia=True,
                               g_solidity=True,
                               g_circularity=True,
                               g_mjaxis=True,
                               g_mnaxis=True,
                               g_morph_ori=True,
                               g_el=True,
                               g_ecc=True,
                               read_from_file=False,
                               filename=None)
            self.set_param_geomrepr(make_mp_grain_centoids=True,
                                    make_mp_grain_points=True,
                                    make_ring_grain_boundaries=True,
                                    make_xtal_grain=True,
                                    make_chull_grain=True,
                                    create_gbz=[False, True],
                                    gbz_thickness=[0.1, 0.2],
                                    read_from_file=False,
                                    filename=None)
            self.set_param_mesh(generate_mesh=False,
                                target_fe_software='abaqus',
                                par_treatment='global',
                                mesher='upxo',
                                gb_conformities=('conformal',
                                                 'non_conformal',
                                                 ),
                                global_elsizes=(0.5,
                                                1.0,
                                                ),
                                mesh_algos=(4,
                                            6,
                                            ),
                                grain_internal_el_gradient=('constant',
                                                            'constant',
                                                            ),
                                grain_internal_el_gradient_par=(('automin',
                                                                 'automax'),
                                                                ('automin',
                                                                 'automax'),
                                                                ),
                                target_eltypes=('CSP4',
                                                'CSP8',
                                                ),
                                elsets=('grains',
                                        'grains',
                                        ),
                                nsets=('x-', 'x+', 'y-', 'y+', ),
                                optimize=(False,
                                          False,
                                          ),
                                opt_par=('min_angle',
                                         [45, 60],
                                         'jacobian',
                                         [0.45, 0.6],
                                         ),
                                read_from_file=False,
                                filename=None)

    def __repr__(self):
        print('/'*60)
        print("+ + + + + UPXO MCGS PARAMETER SWEEP + + + + +")
        print('/'*60)
        print(f"Number of parameter datasets: {len(self.N)}")
        print('/'*60)
        for n in self.N:
            self.info_attributes(n)
            if n != self.N[-1]:
                print('= '*30)
        print('/'*60)
        return ''

    @property
    def info_message_display_level(self):
        return [self.gsi[n].info_message_display_level for n in self.N]

    @info_message_display_level.setter
    def info_message_display_level(self, level):
        if level == 'simple':
            self.set_gsi_info_message_display_level_simple
        elif level == 'detailed':
            self.set_gsi_info_message_display_level_detailed

    @property
    def info_message_display_level_simple(self):
        for n in self.N:
            self.gsi[n].info_message_display_level_simple

    @property
    def info_message_display_level_detailed(self):
        for n in self.N:
            self.gsi[n].info_message_display_level_detailed

    def __iter__(self):
        self.__gsi_index__ = 1
        return self

    def __next__(self):
        if self.__gsi_index__ <= len(self.N):
            _gsi_ = self.gsi[self.__gsi_index__]
            self.__gsi_index__ += 1
            return _gsi_
        else:
            raise StopIteration

    def limit_check(self):
        pass

    def generate_mcgs2d(self):
        pass

    def characterize_gs(self):
        for n in self.N:
            if self.gsi[n].uigsc.parallel:
                pass

    def initialize(self, N=2):
        if type(N) == int and N != 0 and N < 25:
            self.N = [n+1 for n in range(N)]
            from mcgs import monte_carlo_grain_structure as mcgs
            self.gsi = {}
            for n in range(N):
                self.gsi[n+1] = mcgs(study='para_sweep')
        else:
            print(f'Invalid N: {N}. None initialized. Skipped')

    def run(self,
            char_post_sim=True,
            parallel_char=False,
            ):
        '''
        Sweeps across pipelines to achieve as per user request, across
        all allowed and possible combination of individual and sets of
        various parameters. Will only be consideref if defaults is True.
        The default is False.

        char_post_sim:
            Characterize during sims or post sims
        parallel_char:
            Parallelise characterization.
            Only if char_post_sim=True
            If parchar=False, characterization GS will not be characterized,
            in which case, use the characterize method explicitly.
        '''
        pass

    def assemble_locks(self):
        for n in self.N:
            self.__GENLOCK__['uisim'] = self.gsi[n].uisim.__uisim_lock__

    def set_param_grid(self,
                       domain_size=((0, 100), (0, 100), (0, 0), 1),
                       read_from_file=False,
                       filename=None
                       ):
        """
        Set up the parameters pertaining to grid. Targetted at
        ps.gsi[:].uigrid

        Parameters
        ----------
        domain_size : list/tuple/deque/np.ndarray, optional
            ((x bounds), (y bounds), (z bounds), pixel size or increment).
            A bound is specified by two values, minimum and maximum. First
            value MUST be the numerically lower number. As no checks are made
            to validate this user entry, user must take care to enter these
            values.
            NOTE: pixel size will be uniform across x, y and z axes.
            The default is ((0, 100), (0, 100), (0, 0), 1).

        Returns
        -------
        None.

        """
        for n in self.N:
            self.gsi[n].set_uigrid(domain_size,
                                   read_from_file=False,
                                   filename=None
                                   )

    def set_param_sim(self,
                      mcsteps=20,
                      nstates=32,
                      solver='python',
                      tgrad=None,
                      algo_hop=False,
                      default_mcalg='200',
                      algo_hops=[(200, 10), (201, 40), (202, 100)],
                      save_at_mcsteps=np.linspace(0, 20, 5),
                      state_sampling_scheme='rejection',
                      consider_boltzmann_probability=False,
                      s_boltz_prob='q_related',
                      boltzmann_temp_factor_max=0.1,
                      boundary_condition_type='wrapped',
                      NL=1,
                      kineticity='static',
                      purge_previous=False,
                      read_from_file=False,
                      filename=None
                      ):
        """
        Explanation
        -----------
        This is part of parameter setting methods for
        parameter sweep studies. Helps set grain structure simulation
        parameters like. The followig parameters are set by set_param_sim:
            * mcsteps
            * nstates
            * solver
            * tgrad
            * algo_hop
            * algo_hops
            * save_at_mcsteps
            * purge_previous

        Usage
        -----
        UPXO internal and user. Will be used if the user prefers quick
        parameter sweep with default values. If user, user wishes to have
        specirfic values, which would be most often the case.

        Parameters
        ----------
        mcsteps : int, optional
            Number of Monte-Carlo steps. To restrict comparisons over
            different temporal scales, all grain structure instances will
            be simulated upto equal mcsteps.
            NOTE: However, if in a case, the grain structure temporally
            saturates during simulation before the total mcsteps for that
            specific gsi is reaced, then the total number of mcsteps actyually
            covered would be smaller than that specified by the user. This is
            becvause, all core-solver algorithms in UPXO are designed to
            STOP when the grain structure reaces temporal saturation. This
            saturation happens when there is a single grain in the GS, that
            is all state values in the gs lattice become same.
            The default is 20.
        nstates : int, optional
            The total number of unique state values in the simulation. A value
            of 2 would basically generate an Ising type lattice.
            The default is 32.
        solver : str, optional
            Specifies whether the solver is to be from python or C. The choice
            depens on the following parameters:
                * Size of the spatial domain
                * Total number of mcsteps
                * Computational cost of the algorithm
            The decision between 'python' and 'c' should happen based on
            simulation domain largness, which depend on the largeness of the
            spatial domain, largeness of the temporal dimension and finally
            largeness of the computational cost.
            Input is case-independent.
            Choosing 'C' will force UPXO to use
            the C-executatable for core solver. This option is only ava8ilable
            fo0r some algorithms. To know supported algorithms, please refer
            to info on algorithms.
            Choosing 'python' will make UPXO to decide between 'python' and
            'c', based on practicality of using 'python' for large simulation
            domains.
            The default option is 'python'
        tgrad : np.ndarray, optional
            Temperature gradient field of size. Size same as that of grid of
            lattice or that of the state value matrix. Each lattice point must
            accompanied by a temperature value.
            The default is None.
        algo_hop : bool/str, optional
            This helps decide whether to use a single algorithm for the
            entire temporal domain or whether you would need UPXO
            to hop across algorithms.
            If False: disallow algorithm hopping
            If True: allow algorithm hopping
            If str: only allowed value as of now is 'auto'.
            If 'auto': UPXO will decide which algorithms to use upon need of
            a algorithm hopping.
            The default is False.
        algo_hops : list/str/int, optional
            1. If options pertaining to algorithm hopping has
            been provided by the user, then the first available
            option pertaining to algorithm ID will be used to
            set the algorithm. For example, if algo_hops is
            [(200, 10), (201, 40), (202, 100)], then mcalg will be
            set to '200'.
            2. If a numerical entry has been made (in a case where the
            user has done through direct access through set_param_sim),
            then if it is valid, then str(value) will be set for mcalg.
            If invalid, mcalg will default to '200' for each hop.
            3. If a string entry has been made (in a case where the
            user has done through direct access through set_param_sim),
            then if it is valid, then it will be set for mcalg.
            If invalid, mcalg will default to '200'

            The default is [(200, 10), (201, 40), (202, 100)], meaning:
                algo200, upto 10% sim time
                algo201, upto 40% sim time
                algo202, upto 100% sim time
        save_at_mcsteps : ITERABLE, optional
            DESCRIPTION. The default is np.linspace(0, 20, 5).

        Returns
        -------
        None.

        """
        for n in self.N:
            sim_parameters = {'mcsteps': mcsteps,
                              'nstates': nstates,
                              'solver': solver,
                              'tgrad': tgrad,
                              'default_mcalg': default_mcalg,
                              'algo_hop': algo_hop,
                              'algo_hops':  algo_hops,
                              'save_at_mcsteps': save_at_mcsteps,
                              'state_sampling_scheme': state_sampling_scheme,
                              'consider_boltzmann_probability': consider_boltzmann_probability,
                              's_boltz_prob': s_boltz_prob,
                              'boltzmann_temp_factor_max': boltzmann_temp_factor_max,
                              'boundary_condition_type': boundary_condition_type,
                              'NL': NL,
                              'kineticity': kineticity,
                              'purge_previous': purge_previous,
                              }
            self.gsi[n].set_uisim(n=n,
                                  sim_parameters=sim_parameters,
                                  read_from_file=False,
                                  filename=None
                                  )

    def set_param_gsc(self,
                      char_grains=True, char_stage='postsim',
                      library='scikit-image', parallel=True,
                      find_gbseg=True, g_area=True, gb_length=True,
                      gb_length_crofton=True, gb_njp_order=True,
                      g_eq_dia=True, g_feq_dia=True, g_solidity=True,
                      g_circularity=True, g_mjaxis=True, g_mnaxis=True,
                      g_morph_ori=True, g_el=True, g_ecc=True,
                      read_from_file=False, filename=None
                      ):
        """
        Set flags for grain structure characterisaiton and analysis.

        Parameters
        ----------
        char_grains : bool, optional
            Flag to charatcterize grains. If True, grains will be
            characterised and not if False. If True, grain boundaries will also
            be characterised for basic properties. Once the grains have been
            identifies, the characterisation will be done using scikit-image
            by default, at this version of UPXO.
            The default is False.
        char_stage : str, optional
            Choose when to characterize the grains. Options include:
                * 'postsim'
                * 'insim'
            If 'postsim', grain structure will be characterised after all
            temporal slices have been extracted i.e. after all monte-carlo
            iterations have been completed. If 'insim', grain structure
            will be charactersed at the end of each monte-carlo iteration.
            The default is 'postsim'.
        library : str, optional
            Choose which library to identifying the grains. Options include:
                * scikit-image: 2D and 3D
                * opencv: 2D only
                * upxo: 2d only (deprecated)
            The default is 'scikit-image'.
        parallel : bool, optional
            Decides whether grain structure characterisation should be done
            using parallel execution. Following combinations of options are
            permitted:
                * If True and char_stage is 'post_sim', then grain structyure
                characterisation will be done with parallel computation.
                * If True and char_stage is 'in-sim', then the combination is
                invalid. The grain characterisaion will be done at the end of
                each mc iteration.
                * If False and char_stage is 'post_sim', then grain structure
                characterisation will done after all mc iterations are
                completed, but one temporal slice after the other. However,
                the calculation of individual morphological parameters will
                be done using pooling when possible. When this option is not
                possible, behaviour will be similar to that of the combination
                False and 'in-sim'.
                *  If False and char_stage is 'in-sim', then grain structure
                characterisation will be carried out at the end of each
                mc iteration. No part of the process will be threaded or
                pooled or  executed in parallel.
            The default is False.
        find_gbseg : bool, optional
            Flag to identify the grain boundary segments. If True, the grain
            boundary segments will be identified and not if False. GB segments
            will be identified by UPXO and no other oprtion is needed.
            The following behaviours should be kept in mind:
                * Will only work if char_grains is True. Assuming it is True,
                the following further points hold.
                * If find_gbseg is False, but gb_njp_order is True, then, the
                grain boundary segments will still be identified to allow the
                calculation of njp order.
            The default is False.
        g_area : bool, optional
            Flag to calculate grain area. The calculaion takes into
            consideration, the pixel area of the underlying grid. The default
            is False.
        gb_length : bool, optional
            Falg to calculate the grain boundary length. The calculation takjes
            into considertation, side length of the pixel of the underlying
            grid. The following behaviour should be noted:
                * If char_grains is False, and gb_length is True, then
                grain boundary lengths will not be calculated.
                * if grain boundary segments have been identified and gb_length
                is True, then along with calculating grain boundary lengths,
                the lengths of gerain boundaryu segments will also be
                calculated.
            The default is False.
        gb_length_crofton : bool, optional
            Flag to calculate the Crofton perimeter of the grain boundary.
            For more information, please refer to: https://scikit-image.org/
            docs/stable/auto_examples/segmentation/plot_perimeters.html
            The default is False.
        gb_njp_order : bool, optional
            Flag to calculate the 'n' of junction points, that is the value of
            grain boundary junction point order. Its value is the number of
            grains a grain boundary junction point is being shared with. If 3,
            we have a triple point junction, if 4, we have a quadruple point
            junction and so on.
            The following behaviours must be noted:
                * Will only be calculated if char_grains is True and grain
                boundary segments have been identified.
            The default is True.
        g_eq_dia : bool, optional
            Flag to calculate the equivalent diameter of the grain. If True,
            equiavelnt diamater of the grains will be caclculated, not if
            False. The following behaviours must be noted:
                * If grain_area is False, and g_eq_dia is True, then the
                grain area will still be calculated to allow claculation
                of  grain equivalent diameter. However, only the grain
                equivalent diameter will be saved as an attribute and not the
                grain area, which was not requested.
                * Equiavalent diameter caluclation will consider the area
                of the pixel in the grid. Infact, it gets carried from the
                grain_area claculation.
            The default is True.
        g_feq_dia : bool, optional
            Flag to calculate the Feret equivalent diameter. If True, the
            Feret equivalent will be calculated, not if False. Behaviours
            are similar to that of g_eq_dia.
            The default is True.
        g_solidity : bool, optional
            Flag to calculate the solidity of grain. The default is True.
        g_circularity : bool, optional
            Flag to calculate grain circularity. The default is True.
        g_mjaxis : bool, optional
            Flag to calculate the major axis of the grain. The default is True.
        g_mnaxis : bool, optional
            Flag to calculate the ninor axis of the grain. The default is True.
        g_morph_ori : bool, optional
            Flag to calculate the morphological orientation of the grains.
            Bounded in [-90, 90] degrees. The default is True.
        g_el : TYPE, optional
            DESCRIPTION. The default is True.
        g_ecc : bool, optional
            Flag to calculate the eccentricity of the grains. The default is
            True.

        Returns
        -------
        None.

        """
        for n in self.N:
            self.gsi[n].set_uigsc(char_grains=char_grains,
                                  char_stage=char_stage,
                                  library=library, parallel=parallel,
                                  find_gbseg=find_gbseg,
                                  g_area=g_area, gb_length=gb_length,
                                  gb_length_crofton=gb_length_crofton,
                                  gb_njp_order=gb_njp_order,
                                  g_eq_dia=g_eq_dia, g_feq_dia=g_feq_dia,
                                  g_solidity=g_solidity,
                                  g_circularity=g_circularity,
                                  g_mjaxis=g_mjaxis, g_mnaxis=g_mnaxis,
                                  g_morph_ori=g_morph_ori, g_el=g_el,
                                  g_ecc=g_ecc, read_from_file=read_from_file,
                                  filename=filename
                                  )

    def set_param_geomrepr(self,
                           make_mp_grain_centoids=True,
                           make_mp_grain_points=True,
                           make_ring_grain_boundaries=True,
                           make_xtal_grain=True, make_chull_grain=True,
                           create_gbz=True, gbz_thickness = 0.1,
                           read_from_file=False, filename=None
                           ):
        """
        Set parametwers needed to generate geometrical representations of the
        Monte-Carlo Grain Structure.

        Parameters
        ----------
        make_mp_grain_centoids : bool, optional
            Make UPXO multi-point object grom the grain centroids
            The default is True.
        make_mp_grain_points : bool, optional
            Make multi-point objects of all pixel cenrtoids in grains.
            NOTE: Not recommended for large domains.
            The default is False.
        make_ring_grain_boundaries : bool, optional
            Make UPXO multi-point object from all points on the grain boundary
            of a grains. Number of objects made will equal to the number
            of grains. The default is True.
        make_xtal_grain : bool, optional
            Make UPXO XTAL object for the grain. The default is True.
        make_chull_grain : bool, optional
            Flag to create convex hull object of the grain.
            The default is True.
        create_gbz : bool, optional
            Flag to create grain boundary zone. This operation will also make
            the grain core zone. Both of these will be available to be
            turned into element sets for FE mesh export.
            The default is True.
        gbz_thickness : float/int, optional
            Control the thickness of the grain boundary zone. Value must be
            between 0 and 1 and is the fraction of actual grain boundary
            thickness in grid units to minor axis length of the grain.
            NOTE: For grains, where grain boundary zones cannot be created
            due to morphological restrictions, data for the speciric grain
            will be kept at None. Default value is 0.1.

        Returns
        -------
        None.

        """
        for n in self.N:
            self.gsi[n].set_uigeomrepr(make_mp_grain_centoids=make_mp_grain_centoids,
                                       make_mp_grain_points=make_mp_grain_points,
                                       make_ring_grain_boundaries=make_ring_grain_boundaries,
                                       make_xtal_grain=make_xtal_grain,
                                       make_chull_grain=make_chull_grain,
                                       create_gbz=create_gbz,
                                       gbz_thickness=gbz_thickness,
                                       read_from_file=read_from_file,
                                       filename=filename)

    def set_param_mesh(self, generate_mesh=False, target_fe_software='abaqus',
                       par_treatment='global', mesher='upxo',
                       gb_conformities=('conformal', 'non_conformal'),
                       global_elsizes=(0.5, 1.0), mesh_algos=(4, 6),
                       grain_internal_el_gradient=('constant', 'constant'),
                       grain_internal_el_gradient_par=(('automin', 'automax'),
                                                       ('automin', 'automax'),
                                                       ),
                       target_eltypes=('CSP4', 'CSP8'),
                       elsets=('grains', 'grains'),
                       nsets=('x-', 'x+', 'y-', 'y+', ),
                       optimize=(False, False),
                       opt_par=('min_angle', [45, 60],
                                'jacobian', [0.45, 0.6]),
                       read_from_file=False, filename=None
                       ):
        """
        Set the meshing parameters for parameter sweep studies

        Parameters
        ----------
        generate_mesh : BOOL, optional
            Flag to mesh the grain structure.
            The default is False.
        target_fe_software : STR, optional
            The FE software for which the mesh is targetted at.
            Current options include 'abaqus'.
            Future options shall be 'moose', 'damask'
            The default is 'abaqus'.
        par_treatment : STR, optional
            Specifies whether some (see below list) are to apply for
            all instances in the parameter sweep dataset, OR, whether,
            a unique parameter is to be used for a unique instance. This
            applies for the following user input parameters:
                * gb_conformities
                * global_elsizes
                * mesh_algos
                * grain_internal_el_gradient
                * target_eltypes
                * optimize
            If 'local', then 'n' values for each of the above parameters
            must be provided. 'n' is the number of parameter sweeps, which is
            len(ps.N).
            The default is 'global'.
        mesher : STR, optional
            Specify the mesher. Options are 'upxo', 'pygmsh', 'gmsh', 'abaqus'
            -'upxo': applies only to pizellated mesh (non-conformal) of the 2D,
            3D MCGS.
            -'pygmsh', 'gmsh': Applies to geometrised 2D MCGS, 3D MCGS, 2D VTGS
            and 3D VTGS
            -'abaqus': applies to 2D VTGS and geometrised 2D MCGS
            This will write data to disk. UPXO-ABAQUS python scripts are then
            to be used to construct and mesh the grain structure in ABAQUS
            The default is 'upxo'.
        gb_conformities : MIXED: STR/ITERABLE, optional
            Individual value options: 'conformal', 'non_conformal'
            If STR and 'conformal', then all instances will conformally meshed.
            if STR and 'non_conformal', then all instances will non-conformally
            meshed.
            if ITERABLE and of the right size, then each instance will be
            meshed as per the value in the location in gb_conformities
            corresponding to the instance.
            if ITERABLE and of the wrong size, parameter sweep study stops.
            The default is ('conformal', 'non_conformal').
        global_elsizes : MIXED: FLOAT/ITERABLE, optional
            If FLOAT, then it will be mapped to all instances
            If ITERABLE and of the right size, then each instance will be
            meshed with the corresponding element size.
            If ITERABLE and of the wrong size, then parameter sweep study
            stops.
            The default is (0.5, 1.0).
        mesh_algos : MIXED: INT/ITERABLE, optional
            If INT, then it will be mapped to all instances
            If ITERABLE and of the right size, then each instance will be
            meshed with the corresponding specified algirithm
            If ITERABLE and of the wrong size, then parameter sweep
            study will stop.
            The default is (4, 6).
        grain_internal_el_gradient : MIXED: STR/ITERABLE, optional
            If STR, then all instances will be meshed
            with the same element gradient specification
            If ITERABLE and of the right size, then all instances will be
            meshed using correpsoning values of element gradients
            If ITERABLE and of the wrong size, then parameter sweep study
            stops.
            Options are 'constant', 'linear_gb_to_centroid',
            'linear_centroid_to_gb', 'linear_gb_to_core', 'linear_core_to_gb'
            - For value other than 'constant', then global_elsizes will not
            be used. Instead values provided by grain_internal_el_gradient_par
            will be used.
            - For value 'linear_gb_to_centroid', min size will be near gb and
            max size will be at centroid. Variation will be linear.
            - For value 'linear_centroid_to_gb', max size will be near gb and
            min size will be at centroid. Variation will be linear.
            - For value 'linear_gb_to_core', min size will be near gb and
            size increases linearly towards the max size along vectors
            normal to the local gb edge. Vector will be directed towards
            inner region of the grain.
            - For value 'linear_core_to_gb', max size will be near gb and
            size decreases linearly towards the min size along vectors
            normal to the local gb edge. Vector will be directed towards
            inner region of the grain
            The default is ('constant', 'constant').
        grain_internal_el_gradient_par : MIXED: ITERABLE(STR/FLOAT)/ITERABLE,
        optional.
            If STR/FLOAT, same action will be mapped onto all instances.
            If (STR, STR), only allowed non-interchangeable values is 'automin'
            and 'automax'. If ('automin', 'automax'), then element sizes
            will be calculated using a combination of grain boundary
            properties, maximum intercept along the curve normal, grain
            shape factor, etc. The procedure is described in theoretical
            manual.
            If (FLOAT, FLOAT), then values will be chosen accordingly and
            maps accordingly to all instances.
            NOTE @ dev: RETAIN THIS TO BE ('automin', 'automax') and not
            replace with just 'auto', for reason of conformity to a standard
            user data specification format.
            The default is (('automin', 'automax'),).
        target_eltypes : MIXED: STR/ITERABLE, optional
            If STR, value is correct and allowed, then same element types
            will be mapped to all instances.
            If ITERABLE, all values are STR, correct and allowed, then values
            get mapped to each instance seperately and accordingly.
            The default is ('CSP4', 'CSP8').
        elsets : MIXED: STR/ITERABLE, optional
            If STR, valid and allowed, the resuested elment set will be
            mapped to all instances.
            If ITERABLE, values are STR, valid and allowed, then values will
            be mapped to corresponding instances.
            The default is ('grains', 'grains').
        nsets : ITERABLE, optional
            Nodal sets to make. Used to impose boundary conditions.
            Options: 'x-', 'x+', 'y-', 'y+', 'gb', 'rp_random_none_10'
            - Option 'gb': grain boundary nodes. A 'gn' nodal set will be
            created for each grain. Naming will be based on parent grain name.
            - Option 'rp_random_10': representative points, 10 in number.
            These points are points fully inside the grain. None of these
            points would lie on the grain boundary of the grain. 'random'
            denotes random positioning of representative points. Following
            'none' indicates completely randomised. If in place of 'none', we
            have a number (INT/FLOAT), then this number specifies the minimum
            distance of seperation between all representative points inside
            the grain. The following number 10, necessiates that there should
            be 10 coordinate positions (as ITERABLES), if 'random' locationing.
            If these input data-format rules are not conformed to, then
            parameter sweep study stops.
            The default is ('x-', 'x+', 'y-', 'y+', ).
        optimize : MIXED: BOOL/ITERABLE, optional
            If BOOL, then this optimization flag will be mapped to all
            instances.
            If ITERABLE and of right size, then each optimization flag will be
            mapped to each instance accordingly.
            If ITERABLE and of wrong size, then parameter sweep study will
            stop.
            Options: True, False
            The default is (False, False).
        opt_par : MIXED: STR/ITERABLE, optional
            Specifies the element quality metric to optimize the mesh for.
            The default is ('min_angle', [40, 60], 'jacobian', [0.45, 0.6])
            'min_angle' is the minimum angle in the distribution of
            minimum angles of all finite elements. [40, 60] denotes the
            bounds of acceptance. Note that if objectives are not met,
            UPXO will enable recursive mesh refinement near places
            where these minimum angle falls outside the specified bounds.
            'jacobian': similar explanations apply.

        Returns
        -------
        None.


        """
        for n in self.N:
            self.gsi[n].set_uimesh(generate_mesh=generate_mesh,
                                   target_fe_software=target_fe_software,
                                   par_treatment=par_treatment,
                                   mesher=mesher,
                                   gb_conformities=gb_conformities,
                                   global_elsizes=global_elsizes,
                                   mesh_algos=mesh_algos,
                                   grain_internal_el_gradient=grain_internal_el_gradient,
                                   grain_internal_el_gradient_par=grain_internal_el_gradient_par,
                                   target_eltypes=target_eltypes,
                                   elsets=elsets, nsets=nsets,
                                   optimize=optimize, opt_par=opt_par,
                                   read_from_file=read_from_file,
                                   filename=filename
                                   )

    def save(self):
        '''
        Pickle the dataset
        '''
        pass

    def update_gmp(self):
        pass

    def update_qmp(self):
        pass

    def plot(self,
             defaults=False,
             docformat='pdf',
             xax='time',
             yax='area',
             zax='sim',
             xaxpar='',
             yaxpar='',
             zaxpar='',
             plot_type='best',
             ):
        '''
        if defaults=True:
            It generates a set of plots to enable getting a quick
            overview of the data. Plots will be exported to a
            PDF [ref 1].

            Following plots are made.
                * Grain structure plots of the final MC step of all sims.
                * Grain size evolution for all sims.

            [ref 1]: Currently, only PDF is supported. It is planned to enable
            writing data to MS Word, MS PPT, Google Doc, Google presentation,
            MS Excel and Google Spreadsheet.
        --------------------------------------------------
        xax: x-axis
            Options: time, alg, temperature
        yax: y-axis
            Options: time, {morph. parameter}
        zax: z-axis
            Options:
        xaxpar: Parameter for the x-axis
        yaxpar: Parameter for the y-axis
        zaxpar: Parameter for the z-axis
        NOTE:
            if all in (xaxpar, yaxpar, zaxpar) is provided,
            UPXO will override xax, yax, zax
        --------------------------------------------------
        plot_type: Type of visualization
        --------------------------------------------------
        '''
        pass

    @property
    def uigrid(self):
        """
        This is an imitation method. It imitates the instantiated class
        which makes uigrid attribute. Imitation is of the 1st grain
        structure instance. The reason for making an imitation is:
            * All grain structure instances WILL have grids having the
            same dimensionality, bounds and increments.
            * Hence, it is not needed to make a seperate grid availabale
            to the paramater sweep object.
            * Instead, when this property method is called, it just
            returns the uigrid of the first grain structure isntance, if
            it exists.
            * No big deal here.
        NOTE: This documentation applies also to uisim, uigsc, uimesh,
        uigeomrepr
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uigrid'):
                    print('ps.uigrid:: taken from: ps.gsi[1].uigrid')
                    _ = self.gsi[1].uigrid
                else:
                    print('ps.gsi[1].uigrid has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    @property
    def uisim(self):
        """
        This is an imitation method. Refer to ps.uigrid for more documentation.
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uisim'):
                    print('ps.uisim:: taken from: ps.gsi[1].uisim')
                    _ = self.gsi[1].uisim
                else:
                    print('ps.gsi[1].uisim has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    @property
    def uigsc(self):
        """
        This is an imitation method. Refer to ps.uigrid for more documentation.
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uigsc'):
                    print('ps.uigsc:: taken from: ps.gsi[1].uigsc')
                    _ = self.gsi[1].uigsc
                else:
                    print('ps.gsi[1].uigsc has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    @property
    def uimesh(self):
        """
        This is an imitation method. Refer to ps.uigrid for more documentation.
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uimesh'):
                    print('ps.uimesh:: taken from: ps.gsi[1].uimesh')
                    _ = self.gsi[1].uimesh
                else:
                    print('ps.gsi[1].uimesh has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    @property
    def uigeomrepr(self):
        """
        This is an imitation method. Refer to ps.uigrid for more documentation.
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uigeomrepr'):
                    print('ps.uigeomrepr:: taken from: ps.gsi[1].uigeomrepr')
                    _ = self.gsi[1].uigeomrepr
                else:
                    print('ps.gsi[1].uigeomrepr has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    def info_attributes(self, n, throw=False):
        if n not in self.N:
            print('N not set. Skipped')
        else:
            str1 = f"~~~ Parameter sweep dataset: {n} ~~~"
            print(str1 + '\n')
            # ----------------------------------------------------------------
            str2 = "(A. GRID):: mc simulation grid: "
            _ = ' '*12
            if hasattr(self.gsi[n], 'uigrid'):
                str2 += f"{self.gsi[n].uigrid.dim}D."
                str2 += f"{ self.gsi[n].uigrid.type[:2]}."
                strxa = f" x:({self.gsi[n].uigrid.xmin}"
                strxb = f",{self.gsi[n].uigrid.xmax}"
                strxc = f",{self.gsi[n].uigrid.xinc})"
                strx = strxa + strxb + strxc
                strya = f" y:({self.gsi[n].uigrid.ymin}"
                stryb = f",{self.gsi[n].uigrid.ymax}"
                stryc = f",{self.gsi[n].uigrid.yinc})"
                stry = strya + stryb + stryc
                if self.gsi[n].uigrid.dim == 2:
                    str2 += strx + stry
                elif self.gsi[n].uigrid.dim == 3:
                    strza = f" z:({self.gsi[n].uigrid.zmin}"
                    strzb = f",{self.gsi[n].uigrid.zmax}"
                    strzc = f",{self.gsi[n].uigrid.zinc})"
                    strz = strza + strzb + strzc
                    str2 += strx + stry + strz
            else:
                str2 += '\n' + _
                str2 += "Grid parameters not set.\n"
                str2 += _ + "Use set_param_grid(..)"
            print(str2)
            # ----------------------------------------------------------------
            str3 = "(B: SIMPAR):: mc simulation: "
            _ = ' '*14
            if hasattr(self.gsi[n], 'uisim'):
                str3 += f"{self.gsi[n].uisim.mcsteps}"
                str3 += f"  SOLVER: {self.gsi[n].uisim.solver}"
                if len(self.gsi[n].uisim.save_at_mcsteps) > 2:
                    _0 = self.gsi[n].uisim.save_at_mcsteps[0]
                    _1 = self.gsi[n].uisim.save_at_mcsteps[1]
                    str3 += '  IN-SIM SAVES: every ' + str(int(_1 - _0))
                    str3 += ' mcsteps'
                else:
                    str3 += '  IN-SIM SAVES at 0 mcstep'
            else:
                str3 += '\n' + _
                str3 += "Simulation parameters not set.\n"
                str3 += _ + "Use set_param_sim(..)"
            print(str3)
            # ----------------------------------------------------------------
            str4 = "(C: GSCPAR):: gs characterisation: "
            _ = ' '*14
            if hasattr(self.gsi[n], 'uigsc'):
                str4 += f"{self.gsi[n].uigsc.char_grains} \n"
                if self.gsi[n].uigsc.char_grains:
                    str4 += _ + "CHARACTERIZATION STAGE: "
                    str4 += f"{self.gsi[n].uigsc.char_stage}\n"
                    str4 += _ + "PARALLEL CHARACTERIZATION: "
                    str4 += f"{self.gsi[n].uigsc.parallel}"
            else:
                str4 += '\n' + _
                str4 += 'Grain str. characterisation parameters not set.\n'
                str4 += ' '*14 + "Use set_param_uigsc(..)"
            print(str4)
            # ----------------------------------------------------------------
            str5 = "(D: MESHPAR):: fe mesh: "
            _ = ' '*15
            if hasattr(self.gsi[n], 'uimesh'):
                str5 += f"{self.gsi[n].uimesh.generate_mesh}"
                if self.gsi[n].uimesh.generate_mesh:
                    str5 += "\n" + _
                    str5 += "TARGET SOFTWARE: "
                    str5 += f"{self.gsi[n].uimesh.target_fe_software}\n"
                    str5 += _ + "PARAMETER TREATMENT: "
                    str5 += f"{self.gsi[n].uimesh.par_treatment}\n"
                    str5 += _ + "MESHER: {self.gsi[n].uimesh.mesher}\n"
                    str5 += _ + "GB CONFORMITIES: Use: "
                    str5 += f"{self.gsi[n].uimesh.gb_conformities}\n"
                    str5 += _ + "GLOBAL ELEMENT SIZES: Use: "
                    str5 += "ps.gsi[n].uimesh.global_elsizes\n"
                    str5 += _ + "MESH ALGORITHMS: Use: "
                    str5 += "ps.gsi[n].uimesh.mesh_algos\n"
                    str5 += _ + "GRAIN INTERNAL ELEMENT GRADIENT "
                    str5 += "SPECIFICATION: Use:"
                    str5 += "ps.gsi[n].uimesh.grain_internal_el_gradient\n"
                    str5 += _ + "GRAIN INTERNAL ELEMENT GRADIENT VALUES: "
                    str5 += f"{self.gsi[n].uimesh.grain_internal_el_gradient_par}\n"
                    str5 += _ + "ELMENT TARGET TYPES: Use: "
                    str5 += f"{self.gsi[n].uimesh.target_eltypes}\n"
                    str5 += _ + "ELEMENT SETS: Use: "
                    str5 += f"{self.gsi[n].uimesh.elsets}\n"
                    str5 += _ + "NODAL SETS: "
                    str5 += f"Use: {self.gsi[n].uimesh.nsets}\n"
                    str5 += _ + "OPTIMIZE MESH: Use: "
                    str5 += f"{self.gsi[n].uimesh.optimize}\n"
                    str5 += _ + "MESH OPTIMIZATION PARAMETERS: Use: "
                    str5 += f"{self.gsi[n].uimesh.opt_par}\n"
            else:
                str5 += '\n' + _
                str5 += 'FE Mesh parameters not set.\n'
                str5 += _ + "Use set_param_uimesh(..)"
            print(str5)
            # ----------------------------------------------------------------
            str6 = "(E: GEOMREPR):: PXTAL geometric repr(s).: "
            _ = ' '*16
            if hasattr(self.gsi[n], 'uigeomrepr'):
                str6 += "MAKE MULPOINT OF GRAIN CENTROIDS: "
                str6 += f"{self.gsi[n].uigeomrepr.make_mp_grain_centoids}\n"
                str6 += _ + "MAKE MULPOINT OF GRAIN POINTS: "
                str6 += f"{self.gsi[n].uigeomrepr.make_mp_grain_points}\n"
                str6 += _ + "MAKE RING OBJECTS OF GRAIN BOUNDARIES: "
                str6 += f"{self.gsi[n].uigeomrepr.make_ring_grain_boundaries}\n"
                str6 += _ + "MAKE XTAL OBJECTS OF GRAINS: "
                str6 += f"{self.gsi[n].uigeomrepr.make_xtal_grain}\n"
                str6 += _ + "MAKE CONVEX HULL OBJECTS OF GRAINS: "
                str6 += f"{self.gsi[n].uigeomrepr.make_chull_grain}\n"
                str6 += _ + "MAKE GRAIN BOUNDARY ZONE: "
                str6 += f"{self.gsi[n].uigeomrepr.create_gbz}\n"
                str6 += _ + "MAKE 2D.VTGS PXTAL FROM GRAIN CENTROIDS: "
                str6 += "Use: PS.gsi[n].uigeomrepr.make_2dvtgs"
            else:
                str6 += 'Geoemtric repr. parameters not set.\n'
                str6 += _ + "Use set_param_uigeomrepr(..)"
            print(str6)
            # ----------------------------------------------------------------

    def generate_report(self,
                        docformat='pdf'):
        """


        Parameters
        ----------
        docformat : TYPE, optional
            DESCRIPTION. The default is 'pdf'.

        Returns
        -------
        None.

        """
        pass

    def to_excel(self):
        pass

    def model(self):
        pass

    @property
    def default_mcalg(self):
        return self.__default_mcalg__
