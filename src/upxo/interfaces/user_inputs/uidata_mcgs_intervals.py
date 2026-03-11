from termcolor import colored
class _uidata_mcgs_intervals_:
    """
    mcint_grain_size_par_estim: int  ::
    mcint_gb_par_estimation: int  ::
    mcint_grain_shape_par_estim: int  ::
    mcint_save_at_mcstep_interval: int  ::
    save_final_S_only: bool  ::
    mcint_promt_display: int  ::
    mcint_plot_gs: int  ::

    CALL:
        from mcgsa import _uidata_mcgs_intervals_
        uidata_intervals = _uidata_mcgs_intervals_(uidata)
    """
    DEV = True
    __slots__ = ('mcint_grain_size_par_estim',
                 'mcint_gb_par_estimation',
                 'mcint_grain_shape_par_estim',
                 'mcint_save_at_mcstep_interval',
                 'save_final_S_only',
                 'mcint_promt_display',
                 'mcint_plot_grain_structure', '__uiint_lock__'
                 )

    def __init__(self, uidata):
        self.mcint_grain_size_par_estim = uidata['mcint_grain_size_par_estim']
        self.mcint_gb_par_estimation = uidata['mcint_gb_par_estimation']
        self.mcint_grain_shape_par_estim = uidata['mcint_grain_shape_par_estim']
        self.mcint_save_at_mcstep_interval = uidata['mcint_save_at_mcstep_interval']
        self.save_final_S_only = bool(uidata['save_final_S_only'])
        self.mcint_promt_display = uidata['mcint_promt_display']
        self.mcint_plot_grain_structure = bool(uidata['mcint_plot_grain_structure'])

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of mcgs intervals related parameters: \n"
        retstr += f"{colored('MCINT_GRAIN_SIZE_PAR_ESTIM', 'red')}: {colored(self.mcint_grain_size_par_estim, 'green')}\n"
        retstr += f"{colored('MCINT_GB_PAR_ESTIMATION', 'red')}: {colored(self.mcint_gb_par_estimation, 'green')}\n"
        retstr += f"{colored('MCINT_GRAIN_SHAPE_PAR_ESTIM', 'red')}: {colored(self.mcint_grain_shape_par_estim, 'green')}\n"
        retstr += f"{colored('MCINT_SAVE_AT_MCSTEP_INTERVAL', 'red')}: {colored(self.mcint_save_at_mcstep_interval, 'green')}\n"
        retstr += f"{colored('SAVE_FINAL_S_ONLY', 'red')}: {colored(self.save_final_S_only, 'green')}\n"
        retstr += f"{colored('MCINT_PROMT_DISPLAY', 'red')}: {colored(self.mcint_promt_display, 'green')}\n"
        retstr += f"{colored('MCINT_PLOT_GRAIN_STRUCTURE', 'red')}: {colored(self.mcint_plot_grain_structure, 'green')}\n"
        return retstr
