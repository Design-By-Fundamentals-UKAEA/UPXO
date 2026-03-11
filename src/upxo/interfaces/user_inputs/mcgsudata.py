import xlrd

def _load_user_input_data_(xl_fname='input_dashboard.xls'):
    """
    CALL:
        from upxo.interfaces.user_inputs.gather_user_inputs import _load_user_input_data_
        uidata = _load_user_input_data_(xl_fname='input_dashboard.xls')
    Load user input data from an excel file
    """
    workbook = xlrd.open_workbook(xl_fname)
    _sheet_, uidata = workbook.sheet_by_index(0), {}
    for r in range(_sheet_.nrows):
        cellname = _sheet_.cell_value(r, 0)
        cellvalue = _sheet_.cell_value(r, 1)
        uidata[cellname] = cellvalue
    return uidata
