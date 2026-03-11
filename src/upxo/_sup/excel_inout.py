from openpyxl import Workbook, load_workbook

def write_to_excel(wb, dictionary, filename):

    for i, (sheet_name, variable) in enumerate(dictionary.items()):
        if i == 0:
            ws = wb.active
            ws.title = sheet_name
        else:
            ws = wb.create_sheet(title=sheet_name)
        if isinstance(variable, np.ndarray):
            if variable.ndim == 1:
                for r in range(variable.shape[0]):
                    ws.cell(row=r+1, column=1, value=variable[r])
            else:
                for r in range(variable.shape[0]):
                    for c in range(variable.shape[1]):
                        ws.cell(row=r+1, column=c+1, value=variable[r, c])
        else:
            ws.cell(row=1, column=1, value=variable)

    wb.save(filename)

def read_from_excel(filename, sheet_names):
    wb = load_workbook(filename)
    data_from_sheets = {}
    for sheet_name in sheet_names:
        ws = wb[sheet_name]
        if ws.max_row == 1 and ws.max_column == 1:
            data_from_sheets[sheet_name] = ws.cell(row=1, column=1).value
        else:
            if ws.max_column == 1:
                data = np.array([ws.cell(row=r+1, column=1).value for r in range(ws.max_row)])
            else:
                data = np.array([[ws.cell(row=r+1, column=c+1).value for c in range(ws.max_column)] for r in range(ws.max_row)])
            data_from_sheets[sheet_name] = data
    return data_from_sheets
# =========================================
# DATA FROM UPXO
variables = {'xc': xc,
             'yc': yc,
             'phi1': phi1,
             'theta': theta,
             'phi2': phi2,
             'scale': scale,
             'grains': grains,
             'xmax': xmax,
             'ymax': ymax,
             'x_map': x_map,
             'y_map': y_map,
             'BC': BC,
             }
# =========================================
# WRITING UPXO DATA TO FILE
filename = r"C:\Development\M2MatMod\upxo_packaged\upxo_private\data\excel_files\data_for_conf_meshing.xlsx"
# sheet_names = list(variables.keys())
sheet_names = ['xc', 'yc', 'phi1', 'theta', 'phi2', 'scale', 'grains', 'xmax',
               'ymax', 'x_map', 'y_map', 'BC']

sheet_names


write_to_excel(Workbook(), variables, filename)
# =========================================
# READING UPXO DATA TO FILE
read_variables = read_from_excel(filename, sheet_names)
# =========================================
