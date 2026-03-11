import os
# import shutil
import pandas as pd
# import h5py
# import csv
# import openpyxl
from pathlib import Path
from upxo._sup.validation_values import _validation
from upxo._sup.dataTypeHandlers import strip_str as ss


def get_file_path(target_file_name=None):
    """
    its subdirectories.

    Parameters:
    - root_dir_path (str): The path to the directory where the search should
    start. It should be an absolute path.
    - target_file_name (str): The name of the file to search for, including the
    extension.

    Returns:
    - str: The full path to the found file if the file exists within
    the directory tree; otherwise, None.

    This function walks through all directories and subdirectories starting
    from root_dir_path, searching for target_file. If the file is found,
    the function immediately returns the full path to the file. If the search
    completes without finding the file, the function returns None.

    Example:
        get_file_path(search_for_file=True,
                      target_file_name='_ctf_header_CuCrZr_1.txt')
    """
    search_for_file = True  # Retain for later development
    if search_for_file:
        for dirpath, dirnames, filenames in os.walk(get_path_UPXOsrcDIR()):
            if target_file_name in filenames:
                return os.path.join(dirpath, target_file_name)
    return None


def get_path_UPXOsrcDIR():
    return get_path_UPXODIR_L1(dirname='src')


def get_path_UPXOdataDIR():
    return get_path_UPXODIR_L1(dirname='data')


def get_path_UPXODIR_L1(dirname='src'):
    """
    Permitted options for dirname are currently:
        1. data
        2. demos
        3. dev_scripts
        4. docs
        5. external_contributions
        6. gallery
        7. logs
        8. profiling
        9. src
        10. tests
        11. tutorials
    """
    _permitted_ = ('data', 'demos', 'dev_scripts', 'docs',
                   'external_contributions', 'gallery', 'logs',
                   'profiling', 'src', 'tests', 'tutorials')
    # User input validation
    val = _validation()
    val.valstrs(dirname)
    if dirname not in _permitted_:
        raise ValueError(f'dirname: {dirname} not permitted.')
    # Get root and build path
    cwd = Path(os.getcwd())
    n = find_n_to_targetDIR(cwd, 'upxo_private')
    # return cwd.parents[n]/dirname
    return cwd.parents[n]


def get_path_UPXOwriterdataDIR():

    src_path = get_path_UPXOsrcDIR()
    dp = src_path/'upxo'/'_writer_data'
    if dp.exists():
        return dp
    else:
        raise FileNotFoundError(f'Directory {dp} does not exist.')


def load_file(valobj,
              folder_path,
              file_name_with_ext,
              datatype='string',
              encoding='utf-8',
              loadas='line_by_line',
              datatype_return='default',
              isCTFheader=False,
              isCTF=False,
              isCRC=False,
              isEAlist=False,
              seperator_inline='tab',
              validate_before=True):
    """
    Load a text file

    Parameters
    ----------
    valobj:
        upxo validation class instance
    folder_path:
        COmplete path to the directory having the file
    file_name_with_ext:
        Complete filenamre with the extension
    datatype:
        Tyepe of data in the text file
    encoding:
        Encoding to use while reading the file contents
    loadas:
        Specifies how the data is to be loaded or read.
        Options include:
            * line_by_line: Data will be read line by line and each line would
            be an element of a list.
    datatype_return:
        Specifies if thw read data is to be converted into any other format
        before being returned to the calling function scope. Options:
            * 'default': No conversion happens
            * 'np': Read data will be validated for numerical type and
            converted to a numpy array.
            * 'numlist': Read data will be validated foe numerical type and
            converted to a list
    isCTFheader:
        Specifies whether the file being read is a CTF header file
    isCTF:
        Specifies whether the file being read is a EBSD CTF file. If True,
        returned will be a dictionary having keys 'header' and 'values'
    isCRC:
        Specifies whether the file being read is a EBSD CRC file. If True,
        returned will be a dictionary having keys 'header' and 'values'
    seperator_inline:
        Specifies the seperator in a line. If 'default', none will be
        specified
    validate_before:
        Specifies whether path and filenames should be validated beforehand.

    Examples
    --------------------
    from upxo._sup.validation_values import _validation
    from upxo.interfaces.os.osops import load_file

    valobj = _validation()
    folder_path = get_path_UPXOsrcDIR()
    file_name_with_ext = '_ctf_header_CuCrZr_1.txt'
    get_file_path(target_file_name=file_name_with_ext)

    load_file(valobj,
              folder_path=None,
              file_name_with_ext=None,
              datatype='string',
              encoding='utf-8',
              loadas='line_by_line',
              datatype_return='default',
              isCTFheader=False,
              isCTF=False,
              isCRC=False,
              seperator_inline='default',
              validate_before=True)
    """
    pass


def load_files(folder_path=None,
               locinfo='defloc',
               fileContent='ctf_header',
               filename_full='_ctf_header_CuCrZr_1.txt',
               prefix_string='UPXO_mcgs_CTF_slice_',
               suffix_start=0,
               suffix_incr=1,
               suffix_end=49,
               ext='.txt',
               encoding='utf-8',
               readMode='line_by_line'):
    """
    Loads loadable un-encrypted files. Loadable files include:
        1. .txt, .dat, .ctf, .crc
        2. .h5df, .dream3d

    Examples
    --------------
    LOAD A CTF FILE HEADER
    data = load_files(locinfo='defloc',
                      fileContent='ctf_header',
                      filename_full='_ctf_header_CuCrZr_1.txt',
                      )
    """
    valobj = _validation()

    # Validate locinfo for str
    # If locinfo not 'defloc', validate user provided folder_path
    valobj.valstrs(locinfo)
    if locinfo != 'defloc':
        folder_path = valobj.val_path_exists(folder_path, throw_path=True)
        folder_path = Path(folder_path)
    # Validate fileContent str type and see if filename_full is to be built
    # NOTE: If filename_full, then it indicates that, user intends to
    # load a bunch of files by using filename prefix stra dn suffix str!!
    valobj.valstrs(fileContent)
    if filename_full != 'build':
        # Validate if filename_full has extention, if not attach extension.
        try:
            valobj.val_filename_has_ext(filename_full)
        except Exception as e:
            print(f'File extension not provided. Exception: {e}')
            if ss(fileContent) in valobj.fileConOpt_ctf_headers:
                print('   using the default .txt for file extention')
                filename_full += '.txt'
            elif ss(fileContent) in valobj.fileConOpt_ctf_files:
                print('   using the default .ctf for file extention')
                filename_full += '.ctf'
        # If file exists and loadable, load data in the file.
        if fileContent == 'ctf_header':
            # Get the UPXO source directory
            if locinfo == 'defloc':
                folder_path = get_path_UPXOsrcDIR()/'UPXO'/'_writer_data'
            # Validate if file exists
            valobj.val_file_exists(folder_path, filename_full)
            # READ CONTENTS OF THE CTF HEADER FILE
            data = __load_file_1(__folder_path=folder_path,
                                 __file_name_full=filename_full,
                                 encoding='utf-8',
                                 readmode='linebyline')
            return {'data': data,
                    'path': str(folder_path/filename_full)}
    elif filename_full == 'build':
        # Validate prefix_string, suffix_start, suffix_incr and suffix_end
        valobj.valstrs(prefix_string)
        valobj.val_filename_ext_permitted(ext)
        valobj.valnums((suffix_start, suffix_incr, suffix_end))
        # Build fname suffixes
        sfxs = [str(sfxnum) for sfxnum in list(range(suffix_start,
                                                     suffix_end+1,
                                                     suffix_incr))]
        # Build fnames
        fnames = [prefix_string + sfx + ext for sfx in sfxs]
        # Build filepaths
        if locinfo == 'defloc' and fileContent == 'ctf_header':
            folder_path = get_path_UPXOsrcDIR()/'UPXO'/'_writer_data'
            valobj.val_path_exists(folder_path, throw_path=True)
        filepaths = [folder_path/fname for fname in fnames]
        # Initiate data array
        data = {fname: None for fname in fnames}
        fileexists = [True for fname in fnames]
        loadable = [True for fname in fnames]
        # Load data from files
        for fname in fnames:
            data[fname] = __load_file_1(__folder_path=folder_path,
                                        __file_name_full=fname,
                                        encoding='utf-8',
                                        readmode='linebyline')
        return {'data': data,
                'path': str(folder_path/filename_full),
                'fileexists': fileexists,
                'filepaths': filepaths,
                'loadable': loadable}


def __load_file_1(__folder_path=None,
                  __file_name_full=None,
                  encoding='utf-8',
                  readmode='full'):
    # Build filepath
    path = Path(__folder_path)/__file_name_full
    try:
        with open(path, 'r', encoding=encoding) as file:
            loadable = False
            if readmode in ('linebyline', 'line_by_line'):
                data = file.readlines()
                loadable = True
            elif readmode in ('full', 'atonce'):
                data = file.read()
                loadable = True
            return {'data': data,
                    'loadable': loadable}
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {'data': None,
                'loadable': False}


def find_n_to_targetDIR(cwd: Path, target_dir_name: str) -> int:
    """
    Finds the depth (n) to reach a specified target directory from the current
    working directory (cwd).

    Parameters:
    - cwd (Path): The current working directory as a Path object.
    - target_dir_name (str): The name of the target directory to reach.

    # Example usage
    cwd = Path('C:/Development/M2MatMod/upxo_packaged/upxo_private/src/upxo/meshing')
    target_dir_name = 'src'
    n = find_n_to_targetDIR(cwd, target_dir_name)

    # Accessing the src directory if found
    if n != -1:
        src_dir = cwd.parents[n]
        print(f"The 'src' directory path: {src_dir}")
    """
    for n, parent in enumerate(cwd.parents):
        if parent.name == target_dir_name:
            return n
    raise ValueError(f'Search dir {target_dir_name} does not exist',
                     f'PWD is {cwd}',
                     'Please make sure UPXO is installed properly',
                     'If installed properly, just path it from repo')


def error_handler(func):
    """Decorator to handle exceptions and log file operations."""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            print(f"Operation '{func.__name__}' completed successfully.")
            return result
        except FileNotFoundError as e:
            print(f"Error: {e}. File or directory not found in '{func.__name__}'.")
        except Exception as e:
            print(f"An unexpected error occurred in '{func.__name__}': {e}")
    return wrapper


def data_importer(func):
    """Decorator to abstract the data importing functionality."""
    @error_handler  # Use the error handling decorator
    def wrapper(file_path, *args, **kwargs):
        print(f"Importing data from {file_path}")
        return func(file_path, *args, **kwargs)
    return wrapper


def data_exporter(func):
    """Decorator to abstract the data exporting functionality."""
    @error_handler  # Use the error handling decorator
    def wrapper(data, file_path, *args, **kwargs):
        print(f"Exporting data to {file_path}")
        func(data, file_path, *args, **kwargs)
        print("Data export completed successfully.")
    return wrapper


class FileManager:
    @error_handler
    def create_folder(self, path):
        """Create a new folder at the specified path."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder created at: {path}")
        else:
            print(f"Folder already exists at: {path}")

    @error_handler
    def delete_folder(self, path):
        """Delete the folder at the specified path."""
        if os.path.exists(path) and os.path.isdir(path):
            os.rmdir(path)
            print(f"Folder deleted at: {path}")
        else:
            print(f"Folder does not exist at: {path}")

    @error_handler
    def list_folder_contents(self, path):
        """List the contents of the folder at the specified path."""
        if os.path.exists(path) and os.path.isdir(path):
            return os.listdir(path)
        else:
            print(f"Path does not exist or is not a directory: {path}")

    @error_handler
    def write_to_file(self, file_path, data):
        """Write data to a file."""
        with open(file_path, 'w') as file:
            file.write(data)
            print(f"Data written to file: {file_path}")

    @error_handler
    def read_from_file(self, file_path):
        """Read data from a file."""
        with open(file_path, 'r') as file:
            return file.read()

    @data_importer
    def import_txt(self, file_path):
        """Import data from a TXT file."""
        with open(file_path, 'r') as file:
            data = file.read()
        return data

    @data_importer
    def import_csv(self, file_path):
        """Import data from a CSV file."""
        return pd.read_csv(file_path)

    @data_exporter
    def export_csv(self, data, file_path):
        """Export data to a CSV file."""
        data.to_csv(file_path, index=False)
"""
# Example Usage
file_manager = FileManager()
file_manager.create_folder('new_folder')
file_manager.delete_folder('new_folder')
contents = file_manager.list_folder_contents('.')
file_manager.write_to_file('example.txt', 'Hello, FileManager!')
data = file_manager.read_from_file('example.txt')
print(data)
"""
