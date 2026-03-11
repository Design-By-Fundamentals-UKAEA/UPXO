from termcolor import colored
class _uidata_mcgs_property_calc_:
    """
    compute_grain_area_pol: bool :: Flag to compute polygonal grain area
    compute_grain_area_pix: bool :: Flag to compute pixelated grain area
    compute_gb_length_pol: bool :: Flag to compute grain boundayr length polygonal
    compute_gb_length_pxl: bool :: Flag to compute grain boundary length pixelated
    compute_grain_moments: bool :: Flag to compute grain moments
    grain_area_type_to_consider: str :: Flag to select type of area to calculate
    compute_grain_area_distr: bool :: Flag to compute
    compute_grain_area_distr_kde: bool :: Flag to compute
    compute_grain_area_distr_prop: bool :: Flag to select type of grain boundary length to calculate
    gb_length_type_to_consider: str :: Flag to compute
    compute_gb_length_distr: bool :: Flag to compute
    compute_gb_length_distr_kde: bool :: Flag to compute
    compute_gb_length_distr_prop: bool :: Flag to compute

    CALL:
        from mcgs import _uidata_mcgs_property_calc_
        uidata_propcalc = _uidata_mcgs_property_calc_(uidata)
    """
    DEV = True
    __slots__ = ('compute_grain_area_pol',
                 'compute_grain_area_pix',
                 'compute_gb_length_pol',
                 'compute_gb_length_pxl',
                 'compute_grain_moments',
                 'grain_area_type_to_consider',
                 'compute_grain_area_distr',
                 'compute_grain_area_distr_kde',
                 'compute_grain_area_distr_prop',
                 'gb_length_type_to_consider',
                 'compute_gb_length_distr',
                 'compute_gb_length_distr_kde',
                 'compute_gb_length_distr_prop',
                 '__uiprop_lock__'
                 )
    def __init__(self, uidata):
        self.compute_grain_area_pix = bool(uidata['compute_grain_area_pix'])
        if self.compute_grain_area_pix:
            self.compute_grain_area_pol = False
        else:
            self.compute_grain_area_pol = bool(uidata['compute_grain_area_pol'])
        self.compute_gb_length_pol = bool(uidata['compute_gb_length_pol'])
        self.compute_gb_length_pxl = bool(uidata['compute_gb_length_pxl'])
        self.compute_grain_moments = bool(uidata['compute_grain_moments'])
        self.grain_area_type_to_consider = bool(uidata['grain_area_type_to_consider'])
        self.compute_grain_area_distr = bool(uidata['compute_grain_area_distr'])
        self.compute_grain_area_distr_kde = bool(uidata['compute_grain_area_distr_kde'])
        self.compute_grain_area_distr_prop = bool(uidata['compute_grain_area_distr_prop'])
        self.gb_length_type_to_consider = bool(uidata['gb_length_type_to_consider'])
        self.compute_gb_length_distr = bool(uidata['compute_gb_length_distr'])
        self.compute_gb_length_distr_kde = bool(uidata['compute_gb_length_distr_kde'])
        self.compute_gb_length_distr_prop = bool(uidata['compute_gb_length_distr_prop'])

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of grain structure property estimation: \n"
        retstr += f"{colored('COMPUTE_GRAIN_AREA_POL', 'red')}: {colored(self.compute_grain_area_pol, 'green')}\n"
        retstr += f"{colored('COMPUTE_GRAIN_AREA_PIX', 'red')}: {colored(self.compute_grain_area_pix, 'green')}\n"
        retstr += f"{colored('COMPUTE_GB_LENGTH_POL', 'red')}: {colored(self.compute_gb_length_pol, 'green')}\n"
        retstr += f"{colored('COMPUTE_GB_LENGTH_PXL', 'red')}: {colored(self.compute_gb_length_pxl, 'green')}\n"
        retstr += f"{colored('COMPUTE_GRAIN_MOMENTS', 'red')}: {colored(self.compute_grain_moments, 'green')}\n"
        #retstr += f"{colored('COMPUTE_GRAIN_CENTROIDS', 'red')}: {colored(self.compute_grain_centroids, 'green')}\n"
        #retstr += f"{colored('CREATE_GRAIN_BOUNDARY_ZONE', 'red')}: {colored(self.create_grain_boundary_zone, 'green')}\n"
        return retstr
