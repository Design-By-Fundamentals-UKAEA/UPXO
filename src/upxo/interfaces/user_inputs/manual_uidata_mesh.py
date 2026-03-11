from termcolor import colored
class _manual_uidata_mesh_():
    DEV = True
    __lock__ = {'mesh': False,
                'conformal': False,
                'elgradient': False,
                'optimize': False,
                'cps3': True,
                'cps4': False,
                'cps6': True,
                'cps8': True,
                'c3d4': True,
                'c3d6': True,
                'c3d8': True,
                '_': True
                }
    __slots__ = ('generate_mesh',
                 'target_fe_software',
                 'par_treatment',
                 'mesher',
                 'gb_conformities',
                 'global_elsizes',
                 'mesh_algos',
                 'grain_internal_el_gradient',
                 'grain_internal_el_gradient_par',
                 'target_eltypes',
                 'elsets',
                 'nsets',
                 'optimize',
                 'opt_par',
                 )

    def __init__(self,
                 generate_mesh=False,
                 target_fe_software='abaqus',
                 par_treatment='global',
                 mesher='upxo',
                 gb_conformities=('conformal', 'non_conformal'),
                 global_elsizes=(0.5, 1.0),
                 mesh_algos=(4, 6),
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
        Please refer documentation of parameter_sweep.set_param_mesh()
        """
        if not read_from_file:
            self.generate_mesh = generate_mesh
            self.target_fe_software = target_fe_software
            self.par_treatment = par_treatment
            self.mesher = mesher
            self.gb_conformities = gb_conformities
            self.global_elsizes = global_elsizes
            self.mesh_algos = mesh_algos
            self.grain_internal_el_gradient = grain_internal_el_gradient
            self.grain_internal_el_gradient_par = grain_internal_el_gradient_par
            self.target_eltypes = target_eltypes
            self.elsets = elsets
            self.nsets = nsets
            self.optimize = optimize
            self.opt_par = opt_par
        else:
            pass
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of meshing: \n"
        retstr += _ + f"{colored('GENERATE_MESH', 'red', attrs=['bold'])}: {colored(self.generate_mesh, 'cyan')}\n"
        retstr += _ + f"{colored('TARGET_FE_SOFTWARE', 'red', attrs=['bold'])}: {colored(self.target_fe_software, 'cyan')}\n"
        retstr += _ + f"{colored('PAR_TREATMENT', 'red', attrs=['bold'])}: {colored(self.par_treatment, 'cyan')}\n"
        retstr += _ + f"{colored('MESHER', 'red', attrs=['bold'])}: {colored(self.mesher, 'cyan')}\n"
        retstr += _ + f"{colored('GB_CONFORMITIES', 'red', attrs=['bold'])}: {colored(self.gb_conformities, 'cyan')}\n"
        retstr += _ + f"{colored('GLOBAL_ELSIZES', 'red', attrs=['bold'])}: {colored(self.global_elsizes, 'cyan')}\n"
        retstr += _ + f"{colored('MESH_ALGOS', 'red', attrs=['bold'])}: {colored(self.mesh_algos, 'cyan')}\n"
        retstr += _ + f"{colored('GRAIN_INTERNAL_EL_GRADIENT', 'red', attrs=['bold'])}: {colored(self.grain_internal_el_gradient, 'cyan')}\n"
        retstr += _ + f"{colored('GRAIN_INTERNAL_EL_GRADIENT_PAR', 'red', attrs=['bold'])}: {colored(self.grain_internal_el_gradient_par, 'cyan')}\n"
        retstr += _ + f"{colored('TARGET_ELTYPES', 'red', attrs=['bold'])}: {colored(self.target_eltypes, 'cyan')}\n"
        retstr += _ + f"{colored('ELSETS', 'red', attrs=['bold'])}: {colored(self.elsets, 'cyan')}\n"
        retstr += _ + f"{colored('NSETS', 'red', attrs=['bold'])}: {colored(self.nsets, 'cyan')}\n"
        retstr += _ + f"{colored('OPTIMIZE', 'red', attrs=['bold'])}: {colored(self.optimize, 'cyan')}\n"
        retstr += _ + f"{colored('OPT_PAR', 'red', attrs=['bold'])}: {colored(self.opt_par, 'cyan')}\n"
        return retstr

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