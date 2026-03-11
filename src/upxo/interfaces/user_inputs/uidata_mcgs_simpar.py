from termcolor import colored
class _uidata_mcgs_simpar_:
    """
    Class to port user inputs on simulation parameters into UPXO.
    --------------------------------------------------
    Following list of attributes are available:
        * PUBLIC:
            - S: int: Number of individual state values
            - mcsteps: int: Number of Monte-Carlo iterations
            - state_sampling_scheme: str: Sampling scheme to use
            - mcstep_hops: list[list]: List of mcstep ranges for each algo
            - consider_boltzmann_probability: bool: Flag to consider
            Boltzmann prob.
            - s_boltz_prob: str: Type of state dependent B.Probabilty
            - boltzmann_temp_factor_max: float: Multiplication factor
            - boundary_condition_type: str: Type of boundary condition to use
            - NL: int: times unit pixel dist of non-locality for energy
            calculation. Value supported in current UPXO version: 1
                @ value = 2: algorithm needs debugging in math and
                implementation
            - kineticity: str: mobility of temporally evolving state partitions
        * PRIVATE:
            - __lock__: dict
                lock on simulation parameters. Locked if any is True. Has the
                following sublocks:
                    * __lock__['mcstep_hops']:
                        > True for invalid mc-step ranges.
                    * __lock__['non_locality']:
                        > True for invalid Non-Locality value.
                    * __lock__['kineticity']:
                        > True for invalid kineticity value
                        > True for kineticity value incompatible with other
                        atttributes
    --------------------------------------------------
    CALL: INTERNAL
        from mcgs import _uidata_mcgs_simpar_
        _ = _uidata_mcgs_simpar_(uidata)
    --------------------------------------------------
    DEPRECATIONS @DEVELOPER:
        > mcsteps
    --------------------------------------------------
    TODO set: Behaviour overrides:
        * mcalg should now contain the right order of algorityhms to use fior
        each algo_hop. This is internally constructed.
        * mcalg should be made fully private. Impose source accerss only by
        overriding using __वृत्ति१__ uinstead of mcalg. This makes access only
        possible through algo_hops and nothing else.Removes source for
        ambiguities and simplifies user-code-interface. This also forces
        developer to adhere to source more often than making a chain of
        variables all pointing to ther same source.

    TODO set: Following to suit data-structure of algo_hops
        1. state_sampling_scheme
        2. consider_boltzmann_probability: auto based on values in algo_hops
        3. s_boltz_prob: auto based on values in algo_hops
        4. boltzmann_temp_factor_max: auto based on values in algo_hops
        5. NL: auto based on values in algo_hops
        6. kineticity: auto based on values in algo_hops
    """
    DEV = True
    __lock__ = {'mcstep_hops': False,
                'non_locality': False,
                '_': False
                }
    __slots__ = ('S', 'mcsteps',
                 'mcalg', 'algo_hop', 'algo_hops', 'mcstep_hops',
                 'state_sampling_scheme', 'consider_boltzmann_probability',
                 's_boltz_prob', 'boltzmann_temp_factor_max',
                 'boundary_condition_type',
                 'NL', 'kineticity',
                                 '__वृत्ति१__',
                 )

    def __init__(self, uidata=None):
        # ------------------------------------------------------
        self.set_algorithm_hopping(uidata)
        self.set_s(uidata)
        self.set_kbt(uidata)
        self.boundary_condition_type = uidata['boundary_condition_type']
        self.NL = int(uidata['NL'])
        self.kineticity = uidata['kineticity']
        if any(self.__lock__.values()):
            self.__lock__['_'] = True

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of Simulation parameters:\n"
        retstr += _ + f"{colored('MCSTEPS', 'red', attrs=['bold'])}: {colored(self.mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('S', 'red', attrs=['bold'])}: {colored(self.S, 'cyan')} - will be deprecated.\n"
        retstr += _ + f"{colored('STATE SAMPLING SCHEME', 'red', attrs=['bold'])}: {colored(self.state_sampling_scheme, 'cyan')}\n"
        retstr += _ + f"{colored('CONSIDER BOLTZMANN PROBABILITY', 'red', attrs=['bold'])}: {colored(self.consider_boltzmann_probability, 'cyan')}\n"
        retstr += _ + f"{colored('S BOLTZAMNN PROBABILITY', 'red', attrs=['bold'])}: {colored(self.s_boltz_prob, 'cyan')}\n"
        retstr += _ + f"{colored('MAXIMUM BOLTZMANN TEMPERATURE FACTOR', 'red', attrs=['bold'])}: {colored(self.boltzmann_temp_factor_max, 'cyan')}\n"
        retstr += _ + f"{colored('BOUNDARY CONDITION TYPE', 'red', attrs=['bold'])}: {colored(self.boundary_condition_type, 'cyan')}\n"
        retstr += _ + f"{colored('NON LOCALITY', 'red', attrs=['bold'])}: {colored(self.NL, 'cyan')}\n"
        retstr += _ + f"{colored('KINETICITY', 'red', attrs=['bold'])}: {colored(self.kineticity, 'cyan')}\n"
        return retstr

    def set_algorithm_hopping(self, uidata):
        self.mcsteps = int(uidata['mcsteps'])
        self.mcstep_hops = []
        self.mcalg = str(uidata['mcalg'])
        self.algo_hop = True
        self.algo_hops = ((str(uidata['mcalg']),
                           100
                           ),
                          )
        self.__वृत्ति१__ = tuple([_[0] for _ in self.algo_hops])
        self.validate_algo_hops(uidata)

    def set_s(self, uidata):
        self.S = int(uidata['S'])
        self.state_sampling_scheme = uidata['state_sampling_scheme']

    def set_kbt(self, uidata):
        self.consider_boltzmann_probability = bool(uidata['consider_boltzmann_probability'])
        self.s_boltz_prob = uidata['s_boltz_prob']
        self.boltzmann_temp_factor_max = uidata['boltzmann_temp_factor_max']

    def validate_algo_hops(self, uidata):
        # --------------------------------------------------
        # mcalg_hops = self.algo_hops[0][0]
        # mcstep_hops = [[0, self.algo_hops[0][1]]]
        # AAA = self.algo_hops
        # AAA = (('200', 20), ('200', 40), ('200', 41))
        # AAA = (('200', 20), )
        # --------------------------------------------------
        PRINT_algo_hops = lambda: print(self.algo_hops) if self.DEV else None
        PRINT_mcsteps_lock = lambda: print(mcsteps_lock) if self.DEV else None
        # --------------------------------------------------
        print('Algo_hops details')
        PRINT_algo_hops()
        t = [_[1] for _ in self.algo_hops]
        t[0], t[-1] = 0, 100
        mcsteps_lock = [False for _ in t]

        for i in range(len(t)):
            if i > 0:
                if t[i] < t[i-1]:
                    mcsteps_lock[i] = True

        PRINT_mcsteps_lock()

        if any(mcsteps_lock):
            # locked as True in it.
            self.__lock__['mcstep_hops'] = True
            print(f"{colored('Invalid algorithm hopping specification. LOCKED. ','red')}")
        else:
            # open as True not in it.
            mcstep_hops = []
            for i, ah in enumerate(self.algo_hops):
                if i == 0 and len(self.algo_hops) == 1:
                    mcstep_hops = [0, 100]
                elif i == 0 and len(self.algo_hops) > 1:
                    mcstep_hops.append([0, t[i]])
                elif i > 0:
                    mcstep_hops.append([t[i-1]+1, t[i]])
            self.mcstep_hops = mcstep_hops

    @property
    def lock_status(self):
        return self.__lock__['_']

    @property
    def locks(self):
        return self.__lock__
