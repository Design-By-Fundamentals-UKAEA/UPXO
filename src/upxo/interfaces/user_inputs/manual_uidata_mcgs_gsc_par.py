from termcolor import colored
class _manual_uidata_mcgs_gsc_par_:
    """
    Parameters for grain strcuture characterisation
    """
    DEV = True
    __lock__ = {'char_grains': False,
                'char_stage': False,
                'library': False,
                'parallel': False,
                'find_gbseg': False,
                '_': True
                }
    __slots__ = ('char_grains', 'char_stage', 'library', 'parallel',
                 'g_area', 'gb_length', 'find_gbseg',
                 'gb_length_crofton', 'gb_njp_order', 'g_eq_dia',
                 'g_feq_dia', 'g_solidity', 'g_circularity',
                 'g_mjaxis', 'g_mnaxis', 'g_morph_ori',
                 'g_el', 'g_ecc',
                 )

    def __init__(self,
                 char_grains=True, char_stage='postsim',
                 library='scikit-image', parallel=True,
                 find_gbseg=True, g_area=True, gb_length=True,
                 gb_length_crofton=True, gb_njp_order=True,
                 g_eq_dia=True, g_feq_dia=True, g_solidity=True,
                 g_circularity=True, g_mjaxis=True, g_mnaxis=True,
                 g_morph_ori=True, g_el=True, g_ecc=True,
                 read_from_file=False, filename=None
                 ):
        if not read_from_file:
            self.char_grains, self.char_stage = char_grains, char_stage
            self.library, self.parallel = library, parallel
            self.find_gbseg = find_gbseg
            self.g_area, self.gb_length = g_area, gb_length
            self.gb_length_crofton = gb_length_crofton
            self.gb_njp_order = gb_njp_order
            self.g_eq_dia, self.g_feq_dia = g_eq_dia, g_feq_dia
            self.g_solidity, self.g_circularity = g_solidity, g_circularity
            self.g_mjaxis, self.g_mnaxis = g_mjaxis, g_mnaxis
            self.g_morph_ori = g_morph_ori
            self.g_el, self.g_ecc = g_el, g_ecc
        else:
            pass
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = "Grain structure characterisation parameters: \n"
        retstr += _ + f"{colored('CHAR_GRAINS', 'red', attrs=['bold'])}: {colored(self.char_grains, 'cyan')}\n"
        retstr += _ + f"{colored('CHAR_STAGE', 'red', attrs=['bold'])}: {colored(self.char_stage, 'cyan')}\n"
        retstr += _ + f"{colored('LIBRARY', 'red', attrs=['bold'])}: {colored(self.library, 'cyan')}\n"
        retstr += _ + f"{colored('PARALLEL', 'red', attrs=['bold'])}: {colored(self.parallel, 'cyan')}\n"
        retstr += _ + f"{colored('G_AREA', 'red', attrs=['bold'])}: {colored(self.g_area, 'cyan')}\n"
        retstr += _ + f"{colored('GB_LENGTH', 'red', attrs=['bold'])}: {colored(self.gb_length, 'cyan')}\n"
        retstr += _ + f"{colored('FIND_GBSEG', 'red', attrs=['bold'])}: {colored(self.find_gbseg, 'cyan')}\n"
        retstr += _ + f"{colored('GB_LENGTH_CROFTON', 'red', attrs=['bold'])}: {colored(self.gb_length_crofton, 'cyan')}\n"
        retstr += _ + f"{colored('GB_NJP_ORDER', 'red', attrs=['bold'])}: {colored(self.gb_njp_order, 'cyan')}\n"
        retstr += _ + f"{colored('G_EQ_DIA', 'red', attrs=['bold'])}: {colored(self.g_eq_dia, 'cyan')}\n"
        retstr += _ + f"{colored('G_FEQ_DIA', 'red', attrs=['bold'])}: {colored(self.g_feq_dia, 'cyan')}\n"
        retstr += _ + f"{colored('G_SOLIDITY', 'red', attrs=['bold'])}: {colored(self.g_solidity, 'cyan')}\n"
        retstr += _ + f"{colored('G_CIRCULARITY', 'red', attrs=['bold'])}: {colored(self.g_circularity, 'cyan')}\n"
        retstr += _ + f"{colored('G_MJAXIS', 'red', attrs=['bold'])}: {colored(self.g_mjaxis, 'cyan')}\n"
        retstr += _ + f"{colored('G_MNAXIS', 'red', attrs=['bold'])}: {colored(self.g_mnaxis, 'cyan')}\n"
        retstr += _ + f"{colored('G_MORPH_ORI', 'red', attrs=['bold'])}: {colored(self.g_morph_ori, 'cyan')}\n"
        retstr += _ + f"{colored('G_EL', 'red', attrs=['bold'])}: {colored(self.g_el, 'cyan')}\n"
        retstr += _ + f"{colored('G_ECC', 'red', attrs=['bold'])}: {colored(self.g_ecc, 'cyan')}\n"
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