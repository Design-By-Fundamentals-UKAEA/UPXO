from termcolor import colored
class _manual_uidata_mcgs_simpar_:
    """
    * gsi: Grain structure index
    * S: number of states - TO BE DEPRECATED
    * mcsteps: Number of mcsteps
    * nstates: Number of states
    * solver: Python / C
    * tgrad: temperature gradient grid
    * default_mcalg: The default fall-back monte-carlo algorithm
    * algo_hop: Bool flag whether to consider algorithm hopping
    * algo_hops: User provided algorithm hopping specification
    * __epoch_hops_mcsteps_pct__: percentage values of mcsteps of all epoch
        hops
    * epoch_hops_mcsteps: mcstep values of all epock hops
    * epoch_hops_algos: algorithms for each of the epoch hops
    * __mcstep_hop_locks__: lock for each of the epoch hops on validity of
        mcstep values of corresponding epoch hops
    * mcalg: monte-carlo algorithm - TO BE DEPRECATED by phasing out usage
    * save_at_mcsteps - mcstep intervals where temporal grain structure
        instancesto be stored
    * state_sampling_scheme: sampling scheme for use in state flipping.
        NOTE: TO BE GENERALIZED for suite of applicable algorithms
    * consider_boltzmann_probability: Bool flag indicvating whether an
        algorithm should use Boltzmann (i.e. transition) probability
    * s_boltz_prob: Provides choice to select state dependent kbT or
        state independent kbT
        NOTE: An accompanying variable is to be made to provide the
        exact dependency behaviour in case state dependent kbT is needed
    * boltzmann_temp_factor_max: Provides the maximum kbT factor. refer
        to theory manual for more information. In simple terms, use this
        to control grain boundary roughness.
    * boundary_condition_type: Specify type of boundary condition. Currently
        only valid entry is 'wrapped'.
        Option 'closed' is TO BE IMPLEMENTED to consider boundary effects.
        Some of the practical cases where this will be needed include
        the following of grain structures, where gradients in grain structure
        are natural due to boundary effects:
            1. chilled cast grain structure
            2. welded grain strcutures
            3. etc.
    * NL: Non-locality in lattice energy calculation. RESTRICTED to 1.
    * kineticity: str flag to allow UPXO to select algorithms which result in
        an spatially unbalanced skewed energetics in the pixel to pixel
        Hamiltonian estimation.
    * __sp__: A private copy of all user input simulation parameter values

    CALL:
        from mcgs import _uidata_mcgs_simpar_
        uisim = _uidata_mcgs_simpar_(sim_parameters)
    """
    DEV = True
    __lock__ = {'mcsteps': False,  # True for invalid mcsteps
                'nstates': False,  # True for invalid nstates
                'tgrad': False,  # True for invalid temperature gradient
                'algo_hop': False,
                'algo_hops': False,
                'mcstep_hops': False,
                'save_at_mcsteps': False,
                'algo_prop_compatability': False,
                '_': True
                }
    __slots__ = ('gsi', 'S', 'mcsteps', 'nstates', 'solver', 'tgrad',
                 'default_mcalg',
                 'algo_hop', 'algo_hops',
                 '__epoch_hops_mcsteps_pct__', 'epoch_hops_mcsteps',
                 'epoch_hops_algos',
                 '__mcstep_hop_locks__',
                 'mcalg', 'save_at_mcsteps',
                 'state_sampling_scheme', 'consider_boltzmann_probability',
                 's_boltz_prob', 'boltzmann_temp_factor_max',
                 'boundary_condition_type',
                 'NL', 'kineticity', '__वृत्ति१__',  '__sp__'
                 )

    def __init__(self,
                 n,
                 sim_parameters=None,
                 read_from_file=False,
                 filename=None
                 ):
        # Port all incoming simulation parameters to a convineient
        # private variable
        self.__sp__ = sim_parameters
        '''
        Set the grain structure index, which could be:
            * Index of the current grain structure in the parameter sweep
            study. This is the same as the key value of the present grain
            structure in ps.gsi dictionary.
            * Slightly Perturbed Grain Structure. FUTURE FEATURE.
            NOTE: Work needed in on
            this in: (1) concept development (2) procedure development
            (3) interface development (4) code development (5) implementation
        '''
        self.gsi = n
        self.set_s()
        self.set_algo_details(read_from_file)
        self.set_kbt()
        self.set_non_locality()
        self.boundary_condition_type = self.__sp__['boundary_condition_type']
        if any(self.__lock__.values()):
            self.__lock__['_'] = True
        self.__वृत्ति१__ = tuple([_[0] for _ in self.algo_hops])
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of Simulation parameters:\n"
        retstr += _ + f"{colored('GSI', 'red', attrs=['bold'])}:  {colored(self.gsi, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('Num of MCSTEPS', 'red', attrs=['bold'])}:  {colored(self.mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('S', 'red', attrs=['bold'])}:  {colored(self.S, 'cyan')} - will be deprecated.\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('ALGO_HOP', 'red', attrs=['bold'])}:  {colored(self.algo_hop, 'cyan')}\n"
        retstr += _ + f"{colored('ALGO_HOPS', 'red', attrs=['bold'])}:  {colored(self.algo_hops, 'cyan')} - {colored('TO BE MADE PRIVATE', 'red')}\n"
        retstr += _ + f"{colored('EPOCH HOPS: MCSTEPS (%)', 'red', attrs=['bold'])}:  {colored(self.__epoch_hops_mcsteps_pct__, 'cyan')}\n"
        retstr += _ + f"{colored('EPOCH HOPS: MCSTEPS', 'red', attrs=['bold'])}:  {colored(self.epoch_hops_mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('EPOCH HOPS: ALGORITHMS', 'red', attrs=['bold'])}:  {colored(self.epoch_hops_algos, 'cyan')}\n"
        retstr += _ + f"{colored('SAVE_AT_MCSTEPS', 'red', attrs=['bold'])}:  {colored(self.save_at_mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('STATE_SAMPLING_SCHEME', 'red', attrs=['bold'])}:  {colored(self.state_sampling_scheme, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('CONSIDER BOLTZMANN PROBABILITY', 'red', attrs=['bold'])}:  {colored(self.consider_boltzmann_probability, 'cyan')}\n"
        retstr += _ + f"{colored('S BOLTZMANN PROBABILITY', 'red', attrs=['bold'])}:  {colored(self.s_boltz_prob, 'cyan')}\n"
        retstr += _ + f"{colored('BOLTZMANN TEMPERATURE FACTOR MAXIMUM', 'red', attrs=['bold'])}:  {colored(self.boltzmann_temp_factor_max, 'cyan')}\n"
        retstr += _ + f"{colored('BOUNDARY CONDITION TYPE', 'red', attrs=['bold'])}:  {colored(self.boundary_condition_type, 'cyan')}\n"
        retstr += _ + f"{colored('TGRAD', 'red', attrs=['bold'])}:  {colored(self.tgrad, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('NON LOCALITY', 'red', attrs=['bold'])}:  {colored(self.NL, 'cyan')}\n"
        retstr += _ + f"{colored('KINETICITY', 'red', attrs=['bold'])}:  {colored(self.kineticity, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('DEFAULT_MCALG', 'red', attrs=['bold'])}:  {colored(self.default_mcalg, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('SOLVER', 'red', attrs=['bold'])}:  {colored(self.solver, 'cyan')}\n"
        return retstr

    def set_algo_details(self, read_from_file):
        self.default_mcalg = self.__sp__['default_mcalg']
        self.kineticity = self.__sp__['kineticity']
        # --------------------------------------------
        self.set_algorithm_hopping(read_from_file)
        self.set_mcstep_hop_lock_and_epoch_hops_mcsteps_pct()
        self.set_epoch_hops_algos()

    def set_algorithm_hopping(self, read_from_file):

        if not read_from_file:
            if type(self.__sp__['mcsteps']) == int:
                self.mcsteps = self.__sp__['mcsteps']
            else:
                self.__lock__['mcsteps'] = True
            # ----------------------------------------
            # ----------------------------------------
            if type(self.__sp__['solver']) == str:
                self.solver = self.__sp__['solver']
            else:
                self.__lock__['solver'] = True
            # ----------------------------------------
            if type(self.__sp__['tgrad']) == np.ndarray:
                self.tgrad = self.__sp__['tgrad']
            else:
                self.tgrad = 'invalid'
                self.__lock__['tgrad'] = False
            # =================================================================
            self.algo_hop = self.__sp__['algo_hop']
            algo_hops = self.__sp__['algo_hops']
            # ----------------------------------------
            if not self.algo_hop:
                """If algorithm hopping is off (algo_hop=False), then, this
                branching helps set the algorithm (ps.gsi[:].uisim.mcalg) using
                the options provided in the algo_hops.
                """
                if type(algo_hops) in dth.dt.ITERABLES:
                    print(1)
                    """if options pertaining to algorithm hopping has
                    been provided by the user, then the first available
                    option pertaining to algorithm ID will be used to
                    set the algorithm. For example, if algo_hops is
                    [(200, 10), (201, 40), (202, 100)], then mcalg will be
                    set to '200'.
                    """
                    if algo_hops[0][0] in dth.valg.mc2d:
                        self.algo_hops = ((str(algo_hops[0][0]), 100), )
                    else:
                        self.algo_hops = ((self.default_mcalg, 100), )
                elif type(algo_hops) in dth.dt.NUMBERS or type(algo_hops) == str:
                    print(2)
                    """If a numerical entry has been made (in a case where the
                    user has done through direct access through set_param_sim),
                    then if it is valid, then str(value) will be set for mcalg.
                    If invalid, mcalg will default to '200'.
                    """
                    if algo_hops in dth.valg.mc2d:
                        self.algo_hops = ((str(int(algo_hops)), 100), )
                    else:
                        self.algo_hops = ((self.default_mcalg, 100), )
                else:
                    print(3)
                    """ This branch when user input could not be validated
                    or corrected.
                    """
                    print("MCALG could not be validated and/or corrected. Skipped")
                    self.__lock__['algo_hops'] = True
                    self.algo_hops = ((self.default_mcalg, 100), )
                # -------------------------------------
            else:
                """
                This involves two steps. First, validated mcalg array will be
                built. Then the mcsteps breakup will be validated based on
                values in algo_hops. Invalidities will be attempted to be
                corrected to enable simulation completion.
                """
                # STEP 1: Build validated mcalg array
                mcalg = ['invalid' for _ in self.N]
                self.algo_hops = [(None, None) for _ in self.N]
                if type(algo_hops) in dth.dt.ITERABLES:
                    print(5)
                    for n in self.N:
                        if algo_hops[n][0] in dth.valg.mc2d:
                            self.algo_hops[n][0] = str(algo_hops[n][0])
                        else:
                            self.algo_hops[n][0] = self.__default_mcalg__
                        mcalg[n-1] = self.algo_hops[n][0]
                elif type(algo_hops) in dth.dt.NUMBERS or type(algo_hops) == str:
                    print(6)
                    self.algo_hops = [('invalid', ) for _ in self.N]
                    for n in self.N:
                        if algo_hops in dth.valg.mc2d:
                            self.algo_hops[n][0] = str(int(algo_hops))
                        else:
                            self.algo_hops[n][0] = self.__default_mcalg__
                        mcalg[n-1] = self.algo_hops[n][0]
                else:
                    print(7)
                    mcalg = ['invalid' for _ in self.N]
                    self.__lock__['algo_hops'] = True
                    print('MCALG could not be validated and/or corrected. Skipped')
                self.gsi[n+1].uisim.mcalg = self.algo_hops[n][0]
                # STEP 2: Validate mcsteps breakup based on values in algo_hops
            # =================================================================
            # =================================================================
            self.save_at_mcsteps = [int(_) for _ in self.__sp__['save_at_mcsteps']]
        else:
            pass

    def set_mcstep_hop_lock_and_epoch_hops_mcsteps_pct(self):
        """
        Takes in the info from validated algo_hops and build up
        two secondary lists which are easier to handle in algorithm
        selection. In the process, it may decide to perform additional
        validations which may be required.
        """
        # --------------------------------------------------
        ''' STEP 1    GENERATE MONTE-CARLO EPOCHS
        Each pair of adjacent entries denotes a hop range, which is basically
        the range of monte-carlo iterations within which an algorithm works.
        As to which algorithm is it, which would be working, is isolated
        later on in this method space, but the data is already present in the
        self.algo_hops. Example: If epochs is [0, 20, 100], then there would be
        two epoch hops, which are [0, 20] and [20, 100]. As you can see, its
        actually a range of iteration numbers. Note; To be accurate, the epoch
        hops would actually be [0, 20] and [21, 100].
        '''
        epoch = [_[1] for _ in self.algo_hops]
        epoch[0], epoch[-1] = 0, 100
        # --------------------------------------------------
        '''STEP 2    GENERATE TEMPORARY LOCKS FOR EACH EPOCH HOP.
        If the epoch hops contains startings and endings, which are invalid,
        then the mcsteps_lock will be set to True for the specific epoch hop.
        Obviously, there will be as many elements in it as there are
        epoch hops.
        '''
        self.__mcstep_hop_locks__ = [False for _ in epoch]
        for i in range(len(epoch)):
            if i > 0:
                if epoch[i] < epoch[i-1]:
                    self.__mcstep_hop_locks__[i] = True
        print(f"__mcstep_hop_locks__: {self.__mcstep_hop_locks__}")
        # --------------------------------------------------
        '''STEP 3: generate __epoch_hops_mcsteps_pct__ list for each algorithm.
        The outer list contains lists of all epoxh hops. Each inner list
        contains ther start and end of the monte-carlo iteration number. At
        mc iterartion number of srtart, the corresponding algorithm will be
        switched to, from the previous algorithm, if there was one active.
        If any locks in __mcstep_hop_locks__ have been locked in STEP 2, then
        'mcstep_hops' sublock of the global uisim lock gets locked and
        everything stops.
        '''
        if any(self.__mcstep_hop_locks__):
            # Branch: Locked if True in it.
            self.__lock__['mcstep_hops'] = True
            self.__epoch_hops_mcsteps_pct__ = ['invalid' for _ in self.algo_hops]
            self.epoch_hops_mcsteps = ['invalid' for _ in self.algo_hops]
            print(f"{colored('Invalid algorithm hopping specification. LOCKED. ','red')}")
        else:
            # Branch: Open if True not in it.
            self.__epoch_hops_mcsteps_pct__ = []
            for i, ah in enumerate(self.algo_hops):
                if i == 0 and len(self.algo_hops) == 1:
                    self.__epoch_hops_mcsteps_pct__.append([0, 100])
                elif i == 0 and len(self.algo_hops) > 1:
                    self.__epoch_hops_mcsteps_pct__.append([0, epoch[i]])
                elif i > 0:
                    self.__epoch_hops_mcsteps_pct__.append([epoch[i-1]+1, epoch[i]])
            '''Set the actual mcstep start end values including validity check.
            Validity check includes:
                If self.mcsteps is too small to accomodate the algo_hops,
                then uisim parameters get LOCKED.
            '''
            self.epoch_hops_mcsteps = []
            for _mcsteps_ in self.__epoch_hops_mcsteps_pct__:
                starting_mcstep = int(_mcsteps_[0]*self.mcsteps/100)
                c = int(_mcsteps_[1]*self.mcsteps/100)
                self.epoch_hops_mcsteps.append([starting_mcstep,
                                                starting_mcstep])
            self.epoch_hops_mcsteps[-1][1] = self.mcsteps
        print(f"__epoch_hops_mcsteps_pct__: {self.__epoch_hops_mcsteps_pct__}")

    def set_epoch_hops_algos(self):
        if not self.__lock__['mcstep_hops']:
            self.epoch_hops_algos = [am[0] for am in self.algo_hops]
        else:
            self.epoch_hops_algos = ['invalid' for _ in self.algo_hops]
        print(f"epoch_hops_algos:  {self.epoch_hops_algos}")

    def set_kbt(self):
        self.consider_boltzmann_probability = bool(self.__sp__['consider_boltzmann_probability'])
        self.s_boltz_prob = self.__sp__['s_boltz_prob']
        self.boltzmann_temp_factor_max = self.__sp__['boltzmann_temp_factor_max']

    def set_non_locality(self):
        self.NL = int(self.__sp__['NL'])

    def set_s(self):
        if type(self.__sp__['nstates']) == int:
            self.S = self.__sp__['nstates']
            self.nstates = self.S
        else:
            self.__lock__['nstates'] = True
        self.state_sampling_scheme = self.__sp__['state_sampling_scheme']

    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist', 'red', attrs=['bold'])}")
        return self.__lock__['_']

    @property
    def locks(self):
        return self.__lock__

    @property
    def lock_update(self):
        """
        Updates the lock. Outcome could be leave lock either in open or
        locked state. Returns None.
        """
        keys = list(self.locks.keys())
        keys[list(self.locks.keys()).index('_')] = -1
        keys.remove(-1)
        sublocks = [self.locks[key] for key in keys]
        if True in set(sublocks):
            self.__lock__['_'] = True
        else:
            self.__lock__['_'] = False
        return None
