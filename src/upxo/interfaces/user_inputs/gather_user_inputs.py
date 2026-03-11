import xlrd
import os
# Extract gridding parameters
import upxo.interfaces.user_inputs.uidata_mcgs_gridding_definitions as ip_griddef
# Exrtact simulation parametrs
import upxo.interfaces.user_inputs.uidata_mcgs_simpar as ip_simpar
# Extract parameters for grain structure analysis
import upxo.interfaces.user_inputs.uidata_mcgs_grain_structure_characterisation as ip_gschar
# Extract interval counts which trigger speciric operations
import upxo.interfaces.user_inputs.uidata_mcgs_intervals as ip_mcintervals
# Extract grain structrue property calculation parameters (bools)
import upxo.interfaces.user_inputs.uidata_mcgs_property_calc as ip_propcalc
# Extract grain geometric representation flags
import upxo.interfaces.user_inputs.uidata_mcgs_generate_geom_reprs as ip_gengeorepr
# Extract the user input data on meshing
import upxo.interfaces.user_inputs.uidata_mcgs_mesh as ip_mesh
# Load user input data
# import upxo.interfaces.user_inputs.mcgsudata as _load_user_input_data_

def _load_user_input_data_(xl_fname='input_dashboard.xls'):
    """
    CALL:
        from upxo.interfaces.user_inputs.gather_user_inputs import _load_user_input_data_
        uidata = _load_user_input_data_(xl_fname='input_dashboard.xls')
    Load user input data from an excel file
    """
    # get the current fil's path
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    print(current_file_path)
    # get the path to the input_dashboard.xls file
    xl_fname = os.path.join(current_file_path, xl_fname)
    print(xl_fname)
    workbook = xlrd.open_workbook(xl_fname)
    _sheet_, uidata = workbook.sheet_by_index(0), {}
    for r in range(_sheet_.nrows):
        cellname = _sheet_.cell_value(r, 0)
        cellvalue = _sheet_.cell_value(r, 1)
        uidata[cellname] = cellvalue
    return uidata

def load_uidata(input_dashboard):
    """
    CALL:
        from upxo.interfaces.user_inputs.gather_user_inputs import load_uidata
        uidata = load_uidata(input_dashboard='input_dashboard.xls')
    """
    # Load user input data
    __ui = _load_user_input_data_(xl_fname=input_dashboard)
    # Extract gridding parameters
    uigrid = ip_griddef._uidata_mcgs_gridding_definitions_(__ui)
    # Exrtact simulation parametrs
    uisim = ip_simpar._uidata_mcgs_simpar_(__ui)
    # Extract parameters for grain structure analysis
    uigsc = ip_gschar._uidata_mcgs_grain_structure_characterisation_(__ui)
    # Extract interval counts which trigger speciric operations
    uiint = ip_mcintervals._uidata_mcgs_intervals_(__ui)
    # Extract grain structrue property calculation parameters (bools)
    uigsprop = ip_propcalc._uidata_mcgs_property_calc_(__ui)
    # Extract grain geometric representation flags
    uigeorep = ip_gengeorepr._uidata_mcgs_generate_geom_reprs_(__ui)
    # Extract the user input data on meshing
    uimesh = ip_mesh._uidata_mcgs_mesh_(__ui)

    data = {
        'uigrid': uigrid,
        'uisim': uisim,
        'uigsc': uigsc,
        'uiint': uiint,
        'uigsprop': uigsprop,
        'uigeorep': uigeorep,
        'uimesh': uimesh
    }
    return data
