"""UPXO: Validation & Conversion Utilities (NumPy Arrays)

This module provides functions and a class for validating and manipulating
NumPy arrays within the UPXO framework:

- `_validation` (internal): Class for validation targets and type checks.
- `ensure_ndarr_depth2`: Converts elements within iterables (up to depth 2) to
ndarrays.
- `chk_obj_type`: Checks if an object belongs to a specific type.
- `contains_nparray`: Checks for array containment within a target list.

These functionalities enhance data integrity and streamline NumPy array
handling in UPXO workflows.
"""
import numpy as np
import os
from pathlib import Path
from typing import Iterable
from upxo._sup.dataTypeHandlers import dth
import shapely

NUMBERS = dth.dt.NUMBERS
ITERABLES = dth.dt.ITERABLES
_name_point2d_specs_ = dth.opt.name_point2d_specs
_name_point3d_specs_ = dth.opt.name_point3d_specs
_name_point_specs_ = dth.opt.name_point_specs

class _validation():
    """_validation (internal UPXO): Validation & Array Functions

    Provides internal functionalities for:

    - Storing pre-defined validation data (e.g., kernels).
    - Verifying object types using `chk_obj_type`.
    - Array validation and conversion:
        - `ensure_ndarr_depth2`: Converts elements up to depth 2 into ndarrays.
        - `contains_nparray`: Checks for array containment within a target
        list (ndarray).

    Intended for internal use within the UPXO framework only.

    Import
    ------
    from upxo._sup.validation_values import validation

    Author
    ------
    Dr. Sunil Anandatheertha
    """
    # ------------------------------------------------
    # CLASS VARIABLE DEFINITION
    """ Exaplanation of gbjp_iden_kernels """
    gbjp_kernels2d: list = np.array([np.array([[1, 1, 1],
                                               [1, 0, 1],
                                               [1, 1, 1]]),
                                     np.array([[0, 1, 0],
                                               [1, 0, 1],
                                               [0, 1, 0]]),
                                     np.array([[1, 0, 1],
                                               [0, 0, 0],
                                               [1, 0, 1]])
                                     ])
    # ------------------------------------------------
    # Do not use any speacial characters or spaces in any elements here
    # Any special characters get stripped of the string and would fail
    # validations !!
    fileContent_options_all = ('ctfheader', 'ebsdctf', 'ctffile', 'ctf',
                               'temperatures', 'states', 'grid2d', 'grid3d',
                               'upxoinstance', 'femesh', 'orientations')
    fileConOpt_ctf_headers = ('ctfheader', 'ebsdheader')
    fileConOpt_ctf_files = ('ebsdctf', 'ctf', 'ctffile')
    # ------------------------------------------------
    valid_extensions = ('.txt', '.dat', '.ctf', '.crc', '.h5df', '.dream3d')
    # ------------------------------------------------

    def __init__(self):
        pass

    def __repr__(self):
        return 'UPXO.Validations'

    def ensure_ndarr_depth2(self, array, var_name='VARIABLE'):
        """
        Updates all elements within an iterable (up to depth 2) to NumPy
        ndarrays.

        Parameters
        ----------
        array : Iterable
          The input iterable containing potentially nested iterables and
          elements to be converted.
        var_name : str, optional
          A variable name used for error messages. Defaults to 'VARIABLE'.

        Returns
        -------
        ndarray or list of ndarrays
          The converted iterable with all elements (up to depth 2) as NumPy
          ndarrays.

        Raises
        -------
        ValueError
          If the input `array` has a dimension (ndim) less than 2.
        TypeError
          If the input `array` is not an iterable at depth 1 or if the elements
          after conversion at depth 2 are not valid types.

        Notes
        -----
        This function recursively iterates through an iterable, converting
        elements that are not already NumPy ndarrays to ndarrays. It supports
        up to a maximum depth of 2 (i.e., nested iterables with a maximum
        depth of 2).
        """
        # ----------------------------------
        if array.ndim < 2:
            raise ValueError(f'np.array({var_name})',
                             '.ndim must be >= 2.')
        # ----------------------------------
        if not isinstance(var_name, str):
            var_name = 'VARIABLE'
        # ----------------------------------
        # VALIDATE array: Depth 1
        if not isinstance(array, Iterable):
            raise TypeError(f'{var_name} must be an Iterable.')
        else:
            if not isinstance(array, np.ndarray):
                array = np.array(array)
        # ----------------------------------
        # VALIDATE array: Depth 2
        if all([isinstance(_, Iterable) for _ in array]):
            if not all([isinstance(_, np.array) for _ in array]):
                array = np.array([np.array(_) for _ in array])
        else:
            raise TypeError('Invalid type(s) of input field array')
        # ----------------------------------
        return array

    def chk_obj_type(self, obj, expected_type):
        """
        Checks if an object's type matches the expected type.

        Args:
            obj: The object to check the type of.
            expected_type: The expected type of the object (as a string).

        Returns:
            True if the object's type matches the expected type, False
            otherwise.

        Example
        -------
        from upxo._sup.validation_values import _validation
        val = _validation()
        val.chk_obj_type(gs, expected_type)
        """
        return obj.__class__.__name__ == expected_type

    def isiter(self, _iter):
        if not isinstance(_iter, Iterable):
            raise TypeError('INput not iterable.')

    def valstrs(self, strings):
        if not isinstance(strings, Iterable):
            strings = (strings,)
        for string in strings:
            if isinstance(string, str) or string.__class__.__name__ == 'WindowsPath':
                pass
            else:
                raise TypeError(f'Invalid type({string}). Expected: {str}',
                                f' Receieved: {type(string)}')

    def valnums(self, numbers):
        if not isinstance(numbers, Iterable):
            numbers = (numbers,)
        if not all([type(_) in dth.NUMBERS for _ in numbers]):
            raise TypeError(f'Invalid types in({numbers})'
                            f'Expected: type in {dth.Numbers}')

    def val_data_exist(self, *args, **kwargs):
        if args:
            for arg in args:
                if arg is None:
                    raise ValueError('One of inputs is empty')
        if kwargs:
            for kwarg_key, kwarg_val in kwargs.items():
                if not kwarg_val:
                    raise ValueError(f'{kwarg_key} value is empty.')

    def valnparr_types(self, arr1, arr2):
        '''
        from upxo._sup.validation_values import _validation
        val = _validation()
        val.valnparr_types(arr1, arr2)
        '''
        if not type(arr1) == type(arr2):
            raise TypeError('The two arguments are not of same type.'
                            'Expected: both must be numpy.ndarray')

    def valnparr_shape(self, arr1, arr2):
        '''
        val.valnparr_shape(arr1, arr2)
        '''
        # Validate existance
        self.val_data_exist(array1=arr1,
                            array2=arr2)
        # Validate numpy array type
        self.valnparr_types(arr1, arr2)
        if not arr1.shape == arr2.shape:
            raise ValueError('Entered np arrays must have same shape.')

    def valnparrs_types(self, *args):
        '''
        Validate numpy arrays for same type

        from upxo._sup.validation_values import _validation
        val = _validation()
        val.valnparrs_types(*args)
        '''
        # Validate existence
        self.val_data_exist(*args)
        # Validate types
        if len(args) == 1:
            if not isinstance(args[0], np.ndarray):
                raise TypeError('arg no.1 is not a numpy array')
        elif len(args) > 1:
            for i, arg in enumerate(args[1:], start=1):
                if not isinstance(arg, np.ndarray):
                    raise TypeError(f'arg no.{i} is not a numpy array')

    def valnparrs_shapes(self, *args):
        '''
        Validate numpy arrays for same shape

        from upxo._sup.validation_values import _validation
        val = _validation()
        a = np.random.random((3, 3))
        b = np.random.random((3, 3))
        c = np.random.random((3, 3))
        d = np.random.random((3, 4))
        val.valnparrs_shapes(a, b, c, d)
        '''
        # Validate types. This also valkidates existance by default
        self.valnparrs_types(*args)
        # Validate shapes
        if len(args) > 1:
            for i, arg in enumerate(args[1:], start=1):
                if not arg.shape == args[0].shape:
                    raise TypeError(f'Arg no.{i}.shape is not same'
                                    ' as Arg no.0.shape')

    def valnparrs_nelem(self, *args):
        '''
        Validate the total number of elemnets

        from upxo._sup.validation_values import _validation
        val = _validation()
        a = np.random.random((3, 3))
        b = np.random.random((9, 1))
        c = np.random.random((1, 9))
        val.valnparrs_nelem(a, b, c)
        '''
        # Validate if all are iterables
        for ia in args:
            self.isiter(ia)
        # Validate types
        self.valnparrs_types(*args)
        # Validate nuimber of elemenrs
        if len(set([arg.size for arg in args])) > 1:
            raise ValueError('The np arrays have unequal sizes')

    def DEC_validate_samples(comparison_func):
        '''
        Decorator to validate if all elements in an dth.dt.ITERABLE type
        object are of thye same type
        '''

        def wrapper(self, samples=None, *args, **kwargs):
            samples = dth.inlist(samples)
            _ = set(type(_) for _ in samples)
            if len(_) != 1:
                raise TypeError('All samples must be of the same type.')

            return comparison_func(self, samples, _, *args, **kwargs)

        return wrapper

    def contains_nparray(self,
                         ttype: str = 'gbjp_kernels2d',
                         target: Iterable = None,
                         sample: Iterable = None,
                         target_depth: int = 2,
                         ) -> bool:
        """
        Checks if a sample NumPy array is contained in a list of target NumPy
        arrays.

        This function validates if a given `sample` NumPy array exists within
        a collection of `target` NumPy arrays. The `ttype` argument specifies
        the source of the `target` arrays:

        - **'user'**: Requires you to provide a list or tuple of NumPy arrays
        in the `target` argument. This function will convert these
        user-provided arrays to NumPy ndarrays if necessary.
        - **'gbjp_kernels2d' (or any other valid CLASS_VARIBLE name)**: Uses
        a predefined set of NumPy arrays stored in the `self.gbjp_kernels2d`
        class variable for comparison.

        Parameters:
        ttype (str, optional): The type of target arrays to use for validation.
                               Defaults to 'gbjp_kernels2d'.
        target (Iterable, optional): The list or tuple of target NumPy arrays
                                     for validation (only used if `ttype`
                                                     is 'user').
        sample (Iterable, optional): The NumPy array to check for containment.

        Returns:
        bool: True if the `sample` array is found in one of the `target`
        arrays, False otherwise.

        Raises:
        TypeError: If `ttype` is not a string or if `target` and `sample`
        are not iterables.
        ValueError: If `ttype` is not a valid option.

        Notes:
        - This function supports nested iterables up to depth 2
        (i.e., lists containing lists) when using the `'user'` option for
        `ttype`.

        """
        # VALIDATE ttype
        if not isinstance(ttype, str):
            raise TypeError(f'Invalid ttype. Expected {str}')
        # ----------------------------------
        # Prepare the target
        _v_, containment = False, False
        if ttype == 'user':
            if target_depth == 2:
                target, _v_ = self.ensure_ndarr_depth2(target,
                                                       var_name='target'),
            True
        elif ttype == 'gbjp_kernels2d':
            target, _v_ = self.gbjp_kernels2d, True
        # ----------------------------------
        # Check for containment of sample in target
        if _v_:
            containment = any(np.array_equal(sample, valid_kernel)
                              for valid_kernel in target)
        # ----------------------------------
        return containment

    def val_path_exists(self, path, throw_path=True):
        if not path:
            raise ValueError('Path cannot be empty')
        self.valstrs(path)
        path = Path(path)
        if path.exists():
            if throw_path:
                return path
        else:
            raise FileNotFoundError(f"Path: {path} does not exist.")

    def val_filename_has_ext(self, file_name):
        self.valstrs(file_name)
        root, ext = os.path.splitext(file_name)
        if not ext:
            raise ValueError(f"{file_name} has no extention.")

    def val_file_exists(self, path, file_name_with_ext):
        path = self.val_path_exists(path, throw_path=True)
        self.valstrs(file_name_with_ext)
        if path.__class__.__name__ != 'WindowsPath':
            path = Path(path)
        if not Path(path/file_name_with_ext).exists():
            raise FileNotFoundError(f'File: {file_name_with_ext} does not'
                                    f'exist at Path: {path}')

    def val_filename_ext_permitted(self, ext):
        self.valstrs(ext)
        if ext not in self.valid_extensions:
            raise ValueError(f'{ext} is not a permitted extensions')


def isinstance_many(tocheck, dtype):
    """
    Check if all elements of tocheck belongs to a valid dtype.

    Import
    ------
    from upxo._sup.validation_values import isinstance_many

    Parameters
    ----------
    tocheck: An iterable of data.
    dtype: Valid datatype, in dth.dt.ITERABLES

    Return
    ------
    list of bools. True indicates element belonging to dtype

    Example
    -------
    from upxo.geoEntities.point2d import p2d_leanest, p3d_leanest
    a = [p2d_leanest(1, 2), p3d_leanest(1, 2, 1)]
    isinstance_many(a, p3d_leanest)

    Author
    ------
    Dr. Sunil Anandatheertha
    """
    if type(tocheck) not in dth.dt.ITERABLES:
        tocheck = (tocheck, )
    return [isinstance(tc, dtype) for tc in tocheck]


def find_spec_of_points(data):
    """
    Find coordinate point specification format. See examples for details.

    Import
    ------
    from upxo._sup.validation_values import find_spec_of_points

    Examples
    --------
    from upxo.geoEntities.point2d import Point2d as p2d
    from upxo.geoEntities.point2d import p2d_leanest
    from upxo.geoEntities.point3d import Point3d as p3d
    from upxo.geoEntities.point3d import p3d_leanest

    Example-01
    ----------
    >> find_spec_of_points(p2d(1, 2))
    >> 'Point2d'

    Example-02
    ----------
    >> find_spec_of_points([p2d(1, 2), p2d(3, 3)])
    >> '[Point2d]'

    Example-03
    ----------
    >> find_spec_of_points(p2d_leanest(1, 2))
    >> 'p2d_leanest'

    Example-04
    ----------
    >> find_spec_of_points([p2d_leanest(1, 2), p2d_leanest(1, 2)])
    >> '[p2d_leanest]'

    Example-05
    ----------
    >> find_spec_of_points([1, 2])
    >> 'type-[1,2]'

    Example-06
    ----------
    >> find_spec_of_points([[1, 2]])
    >> 'type-[[1,2]]'

    Example-07
    ----------
    >> find_spec_of_points([[1,2],[3,4],[5,6]])
    >> 'type-[[1,2],[3,4],[5,6]]'

    Example-08
    ----------
    >> find_spec_of_points([[2,1,1,2],[3,4,5,6]])
    >> type-[[1,2,3,4],[5,6,7,8]]'

    Example-09
    ----------
    >> find_spec_of_points(p3d(1, 2, 1))
    >> 'Point3d'

    Example-10
    ----------
    >> find_spec_of_points(p3d(1, 2))
    >> 'Point3d'

    Example-11
    ----------
    >> find_spec_of_points([p3d(1, 2), p3d(3, 3)])
    >> '[Point3d]'

    Example-12
    ----------
    >> find_spec_of_points(p3d_leanest(1,2,3))
    >> 'p3d_leanest'

    Example-13
    ----------
    >> find_spec_of_points([p3d_leanest(1,2,3), p3d_leanest(1,2,3)])
    >> '[p3d_leanest]'

    Example-14
    ----------
    >> find_spec_of_points([1,2,3])
    >> 'type-[1,2,3]'

    Example-15
    ----------
    >> find_spec_of_points([[1,2,3]])
    >> 'type-[[1,2,3]]'

    Example-16
    ----------
    >> find_spec_of_points([[1,2,3],[4,5,6],[7,8,9]])
    >> 'type-[[1,2,3],[4,5,6],[7,8,9]]'

    Example-17
    ----------
    >> find_spec_of_points([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    >> 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]'

    Author
    ------
    Dr. Sunil Anandatheertha

    Scope for further development
    -----------------------------
    Extend to consider: Shapely, vtk, gmsh and pyvtk point specifications.
    """
    NUMBERS, ITERABLES = dth.dt.NUMBERS, dth.dt.ITERABLES
    # -------------------------------------
    spec_found, specification = False, 'unknown'
    # -------------------------------------
    if data.__class__.__name__ == 'p2d_leanest' and not spec_found:
        spec_found, specification = True, 'p2d_leanest'
    if data.__class__.__name__ == 'p3d_leanest' and not spec_found:
        spec_found, specification = True, 'p3d_leanest'
    # -------------------------------------
    if data.__class__.__name__ == 'Point2d' and not spec_found:
        spec_found, specification = True, 'Point2d'
    if data.__class__.__name__ == 'Point3d' and not spec_found:
        spec_found, specification = True, 'Point3d'
    # -------------------------------------
    if isinstance(data, ITERABLES) and not spec_found:
        if all(_.__class__.__name__ == 'p2d_leanest' for _ in data):
            spec_found, specification = True, '[p2d_leanest]'
        if all(_.__class__.__name__ == 'p3d_leanest' for _ in data) and not spec_found:
            spec_found, specification = True, '[p3d_leanest]'
    # -------------------------------------
    if isinstance(data, ITERABLES) and not spec_found:
        if all(_.__class__.__name__ == 'Point2d' for _ in data):
            spec_found, specification = True, '[Point2d]'
        if all(_.__class__.__name__ == 'Point3d' for _ in data) and not spec_found:
            spec_found, specification = True, '[Point3d]'
    # -------------------------------------
    if isinstance(data, ITERABLES) and all(isinstance_many(data, NUMBERS)) and not spec_found:
        if len(data) == 2:  # p = [1, 2]
            spec_found, specification = True, 'type-[1,2]'
        if len(data) == 3:  # p = [1, 2, 3]
            spec_found, specification = True, 'type-[1,2,3]'
    # -------------------------------------
    if isinstance(data, ITERABLES) and len(data) == 1 and isinstance(data[0], ITERABLES) and all(isinstance_many(data[0], NUMBERS)) and not spec_found:
        if len(data[0]) == 2:
            spec_found, specification = True, 'type-[[1,2]]'
        if len(data[0]) == 3:
            spec_found, specification = True, 'type-[[1,2,3]]'
    # -------------------------------------
    if isinstance(data, ITERABLES) and all(isinstance_many(data, ITERABLES)) and not spec_found:
        if all(len(_) == 2 for _ in data):  # p = [[2, 1], [10, 20], [31, 49]]
            spec_found, specification = True, 'type-[[1,2],[3,4],[5,6]]'
        if all(len(_) == 3 for _ in data):
            spec_found, specification = True, 'type-[[1,2,3],[4,5,6],[7,8,9]]'
    # -------------------------------------
    if isinstance(data, ITERABLES) and len(data) == 2 and all(isinstance_many(data, ITERABLES)) and not spec_found:
        if (all(len(_) == 1 for _ in data) or all(len(_) >= 2 for _ in data)):
            if all(isinstance_many(data[0], NUMBERS)) and all(isinstance_many(data[1], NUMBERS)):
                # p = [[2, 1, 1, 2], [3, 4, 5, 6]]
                spec_found, specification = True, 'type-[[1,2,3,4],[5,6,7,8]]'
    if isinstance(data, ITERABLES) and len(data) == 3 and all(isinstance_many(data, ITERABLES)) and not spec_found:
        if (all(len(_) == 1 for _ in data) or all(len(_) > 3 for _ in data)):
            if all(isinstance_many(data[0], NUMBERS)) and all(isinstance_many(data[1], NUMBERS)):
                # p = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
                spec_found, specification = True, 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]'
    # -------------------------------------
    return specification


def find_spec_points_mixed_datatypes(points):
    """
    Return specifications of all in a list of coordinate points.

    Import
    ------
    from upxo._sup.validation_values import find_spec_points_mixed_datatypes

    Examples
    --------
    from upxo.geoEntities.point2d import Point2d as p2d
    from upxo.geoEntities.point2d import p2d_leanest
    from upxo.geoEntities.point3d import Point3d as p3d
    from upxo.geoEntities.point3d import p3d_leanest

    points = [p2d(1, 2),
              [p2d(1, 2), p2d(3, 3)],
              p2d_leanest(1, 2),
              [p2d_leanest(1, 2), p2d_leanest(1, 2)],
              [1, 2],
              [[1, 2]],
              [[1,2],[3,4],[5,6]],
              [[2,1,1,2],[3,4,5,6]],
              p3d(1, 2),
              [p3d(1, 2), p3d(3, 3)],
              p3d_leanest(1,2,3),
              [p3d_leanest(1,2,3), p3d_leanest(1,2,3)],
              [1,2,3],
              [[1,2,3]],
              [[1,2,3],[4,5,6],[7,8,9]],
              [[1,2,3,4],[1,2,3,4],[1,2,3,4]],
             ]

    find_spec_points_mixed_datatypes(points)
    >>['Point2d',
     '[Point2d]',
     'p2d_leanest',
     '[p2d_leanest]',
     'type-[1,2]',
     'type-[[1,2]]',
     'type-[[1,2],[3,4],[5,6]]',
     'type-[[1,2,3,4],[5,6,7,8]]',
     'Point3d',
     '[Point3d]',
     'p3d_leanest',
     '[p3d_leanest]',
     'type-[1,2,3]',
     'type-[[1,2,3]]',
     'type-[[1,2,3],[4,5,6],[7,8,9]]',
     'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]']
    """
    class_names = [pnt.__class__.__name__ for pnt in points]
    # -------------------------------------
    if len(set(class_names)) == 1:
        spec = find_spec_of_points(points[0])
        specification = [spec for cn in class_names]
    else:
        specification = [find_spec_of_points(pnt) for pnt in points]
    # -------------------------------------
    return specification

def val_point_and_get_coord(point, return_type='coord', safe_exit=True):
    """
    from upxo.geoEntities.point2d import Point2d
    from upxo.geoEntities.point3d import Point3d
    from upxo.geoEntities.point2d import p2d_leanest
    from upxo.geoEntities.point3d import p3d_leanest

    val_point_and_get_coord(Point2d(0, 0), return_type='coord')
    val_point_and_get_coord(Point2d(0, 0), return_type='upxo')

    val_point_and_get_coord(Point2d(0, 0))
    val_point_and_get_coord([Point2d(0, 0)])
    val_point_and_get_coord([Point2d(0, 0), Point2d(0, 0)])
    val_point_and_get_coord(p2d_leanest(0, 0))
    val_point_and_get_coord([p2d_leanest(0, 0)])
    val_point_and_get_coord([p2d_leanest(0, 0), p2d_leanest(0, 0)])
    val_point_and_get_coord([0, 1])
    val_point_and_get_coord([[0, 1]])
    val_point_and_get_coord([[0], [1]])
    val_point_and_get_coord([[1,2],[3,4],[5,6]])
    val_point_and_get_coord([[2,1,1,2],[3,4,5,6]])

    val_point_and_get_coord(Point3d(1, 2, 1))
    val_point_and_get_coord(Point3d(1, 2, 1))
    val_point_and_get_coord([Point3d(1, 2, 1)])
    val_point_and_get_coord([Point3d(1, 2, 1), Point3d(3, 3, 1)])
    val_point_and_get_coord(p3d_leanest(1, 2, 1))
    val_point_and_get_coord([p3d_leanest(1, 2, 1)])
    val_point_and_get_coord([p3d_leanest(1, 2, 1), p3d_leanest(3, 3, 1)])
    val_point_and_get_coord([1,2,3])
    val_point_and_get_coord([[1,2,3]])
    val_point_and_get_coord([[1,2,3],[4,5,6],[7,8,9]])
    val_point_and_get_coord([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    """
    if point is None:
        raise ValueError('Point OR coord not provided.')
    # ------------------------------------------------------------------
    spec = find_spec_of_points(point)
    # ------------------------------------------------------------------
    if spec not in _name_point_specs_:
        raise ValueError('Invalid point specification.')
    # ------------------------------------------------------------------
    if spec in _name_point2d_specs_:
        from upxo.geoEntities.point2d import Point2d
        if spec == 'Point2d':
            trglocx, trglocy = point.x, point.y
        elif spec == '[Point2d]':
            trglocx, trglocy = point[0].x, point[0].y
        elif spec == 'p2d_leanest':
            trglocx, trglocy = point._x, point._y
        elif spec == '[p2d_leanest]':
            trglocx, trglocy = point[0]._x, point[0]._y
        elif spec == 'type-[1,2]':
            trglocx, trglocy = point[0], point[1]
        elif spec == 'type-[[1,2]]':
            trglocx, trglocy = point[0][0], point[0][1]
        elif spec == 'type-[[1,2],[3,4],[5,6]]':
            trglocx, trglocy = point[0][0], point[0][1]
        elif spec == 'type-[[1,2,3,4],[5,6,7,8]]':
            trglocx, trglocy = point[0][0], point[1][0]
        else:
            if safe_exit:
                trglocx, trglocy = None, None
            else:
                raise ValueError('Invalid point input.')
        # . . . . . . . . . . . . .
        if return_type == 'coord':
            return trglocx, trglocy
        elif return_type == 'upxo':
            return Point2d(trglocx, trglocy)
    # ------------------------------------------------------------------
    if spec in _name_point3d_specs_:
        from upxo.geoEntities.point3d import Point3d
        if spec == 'Point3d':
            trglocx, trglocy, trglocz = point.x, point.y, point.z
        elif spec == '[Point3d]':
            trglocx, trglocy, trglocz = point[0].x, point[0].y, point[0].z
        elif spec == 'p3d_leanest':
            trglocx, trglocy, trglocz = point._x, point._y, point._z
        elif spec == '[p3d_leanest]':
            trglocx, trglocy, trglocz = point[0]._x, point[0]._y, point[0]._z
        elif spec == 'type-[1,2,3]':
            trglocx, trglocy, trglocz = point[0], point[1], point[2]
        elif spec == 'type-[[1,2,3]]':
            trglocx, trglocy, trglocz = point[0][0], point[0][1], point[0][2]
        elif spec == 'type-[[1,2,3],[4,5,6],[7,8,9]]':
            trglocx, trglocy, trglocz = point[0][0], point[0][1], point[0][2]
        elif spec == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]':
            trglocx, trglocy, trglocz = point[0][0], point[1][0], point[2][0]
        else:
            if safe_exit:
                trglocx, trglocy, trglocx = None, None, None
            else:
                raise ValueError('Invalid point input.')
        # . . . . . . . . . . . . .
        if return_type == 'coord':
            return trglocx, trglocy, trglocz
        elif return_type == 'upxo':
            return Point3d(trglocx, trglocy, trglocz)

def val_points_and_get_coords(points, return_type='coord', safe_exit=True):
    """
    Returns coordinates or UPXO point list for a given points list.

    Parameters
    ----------
    points: A single point or a list of points in any of the following
        acceptable formats.

    Acceptable points specifications
    --------------------------------
    1. p2d(1,2)
    2. [p2d(1,2), p2d(3,3)]
    3. p2d_leanest(1,2)
    4. [p2d_leanest(1,2), p2d_leanest(1,2)]
    5. [1,2]
    6. [[1,2]]
    7. [[1,2],[3,4],[5,6]]
    8. [[2,1,1,2],[3,4,5,6]]
    9. p3d(1,2)
    10. [p3d(1,2), p3d(3,3)]
    11. p3d_leanest(1,2,3)
    12. [p3d_leanest(1,2,3), p3d_leanest(1,2,3)]
    13. [1,2,3]
    14. [[1,2,3]]
    15. [[1,2,3],[4,5,6],[7,8,9]]
    16. [[1,2,3,4],[1,2,3,4],[1,2,3,4]]

    Examples
    --------
    from upxo.geoEntities.point2d import Point2d
    points = [Point2d(0, 0) for _ in range(10)]
    val_points_and_get_coords(points)
    val_points_and_get_coords(points, return_type='upxo')
    val_points_and_get_coords(points, return_type='numpy')

    from upxo.geoEntities.point3d import Point3d
    points = [Point3d(0, 0, 0) for _ in range(10)]
    val_points_and_get_coords(points)
    val_points_and_get_coords(points, return_type='upxo')
    val_points_and_get_coords(points, return_type='numpy')

    from upxo.geoEntities.point2d import Point2d
    from upxo.geoEntities.point3d import Point3d
    from upxo.geoEntities.point2d import p2d_leanest
    from upxo.geoEntities.point3d import p3d_leanest

    val_points_and_get_coords(Point2d(0, 0), return_type='coord')
    val_points_and_get_coords(Point2d(0, 0), return_type='upxo')

    val_points_and_get_coords(Point2d(0, 0))
    val_points_and_get_coords([Point2d(0, 0)])
    val_points_and_get_coords([Point2d(0, 0), Point2d(0, 0)])
    val_points_and_get_coords(p2d_leanest(0, 0))
    val_points_and_get_coords([p2d_leanest(0, 0)])
    val_points_and_get_coords([p2d_leanest(0, 0), p2d_leanest(0, 0)])
    val_points_and_get_coords([0, 1])
    val_points_and_get_coords([[0, 1]])
    val_points_and_get_coords([[0], [1]])
    val_points_and_get_coords([[1,2],[3,4],[5,6]])
    val_points_and_get_coords([[2,1,1,2],[3,4,5,6]])

    val_points_and_get_coords(Point3d(1, 2, 1))
    val_points_and_get_coords(Point3d(1, 2, 1))
    val_points_and_get_coords([Point3d(1, 2, 1)])
    val_points_and_get_coords([Point3d(1, 2, 1), Point3d(3, 3, 1)])
    val_points_and_get_coords(p3d_leanest(1, 2, 1))
    val_points_and_get_coords([p3d_leanest(1, 2, 1)])
    val_points_and_get_coords([p3d_leanest(1, 2, 1), p3d_leanest(3, 3, 1)])
    val_points_and_get_coords([1,2,3])
    val_points_and_get_coords([[1,2,3]])
    val_points_and_get_coords([[1,2,3],[4,5,6],[7,8,9]])
    val_points_and_get_coords([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    val_points_and_get_coords([[1,2,3,4],[1,2,3,4],[1,2,3,4]], return_type='upxo')

    RAND = np.random.random
    val_points_and_get_coords([RAND(10), RAND(10), RAND(10)], return_type='coord')
    val_points_and_get_coords([RAND(10), RAND(10), RAND(10)], return_type='npcoord')
    val_points_and_get_coords([RAND(10), RAND(10), RAND(10)], return_type='upxo')
    """
    # --------------------------------------------------
    spec = find_spec_of_points(points)
    # --------------------------------------------------
    spec_found = True
    if spec == 'Point2d':
        coords = [points.x, points.y]
    elif spec == '[Point2d]':
        coords = [[pnt.x, pnt.y] for pnt in points]
    elif spec == 'p2d_leanest':
        coords = [points._x, points._y]
    elif spec == '[p2d_leanest]':
        coords = [[pnt._x, pnt._y] for pnt in points]
    elif spec == 'type-[1,2]':
        coords = list(points)
    elif spec == 'type-[[1,2]]':
        coords = list(points[0])
    elif spec == 'type-[[1,2],[3,4],[5,6]]':
        coords = [[pnt[0], pnt[1]] for pnt in points]
    elif spec == 'type-[[1,2,3,4],[5,6,7,8]]':
        coords = [[x, y] for x, y in zip(points[0], points[1])]
    elif spec == 'Point3d':
        coords = [points.x, points.y, points.z]
    elif spec == '[Point3d]':
        coords = [[pnt.x, pnt.y, pnt.z] for pnt in points]
    elif spec == 'p3d_leanest':
        coords = [points._x, points._y, points._z]
    elif spec == '[p3d_leanest]':
        coords = [[pnt._x, pnt._y, pnt._z] for pnt in points]
    elif spec == 'type-[1,2,3]':
        coords = list(points)
    elif spec == 'type-[[1,2,3]]':
        coords = list(points[0])
    elif spec == 'type-[[1,2,3],[4,5,6],[7,8,9]]':
        coords = [[pnt[0], pnt[1], pnt[2]] for pnt in points]
    elif spec == 'type-[[1,2,3,4],[1,2,3,4],[1,2,3,4]]':
        coords = [[x, y, z] for x, y, z in zip(points[0], points[1], points[2])]
    else:
        spec_found = False
        if safe_exit:
            coords = [None, None, None]
        else:
            raise ValueError('Invalid point input.')
        # . . . . . . . . . . . . .
    if return_type == 'coord':
        return coords
    elif return_type == 'npcoord':
        return np.array(coords)
    elif return_type == 'upxo':
        if spec in _name_point2d_specs_:
            from upxo.geoEntities.point2d import Point2d
            return [Point2d(coord[0], coord[1]) for coord in coords]
        elif spec in _name_point3d_specs_:
            from upxo.geoEntities.point3d import Point3d
            return [Point3d(coord[0], coord[1], coord[2]) for coord in coords]
