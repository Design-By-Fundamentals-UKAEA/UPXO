"""
upxo.interfaces.defdap.ebsd_reader
===================================
Thin, UPXO-native wrapper around DefDAP for loading 2D EBSD maps from
Oxford Instruments files (.ctf / .crc) and exposing the data as plain
NumPy arrays that the rest of UPXO (repgen2d, gsan2d, etc.) can consume
directly.

Supported file formats
----------------------
.ctf   Oxford Instruments text   -> data_type='OxfordText'
.crc   Oxford Instruments binary -> data_type='OxfordBinary'

Extracted arrays (all stored as plain NumPy — no DefDAP objects leak out)
-------------------------------------------------------------------------
lfi_ebsd   : np.ndarray, int,   shape (ny, nx)
    Grain label field.  Values >= 1 are grain IDs.
    -1  = remnant grain-boundary pixel (not assigned to any grain).
    -2  = pixel belonging to a grain smaller than min_grain_size.
    0   = non-indexed point.

euler_ebsd : np.ndarray, float, shape (ny, nx, 3)
    Bunge Euler angles (phi1, Phi, phi2) in **radians**.
    Transposed from DefDAP's native (3, ny, nx) to image-convention
    (ny, nx, 3) so that euler_ebsd[row, col] gives a 3-vector.

quat_ebsd  : np.ndarray, float, shape (ny, nx, 4)
    Unit quaternion per pixel (q0, q1, q2, q3) extracted from DefDAP's
    object array.  Positive-hemisphere convention (q0 >= 0).

step_size  : float
    Pixel step size in microns, read directly from the file metadata.

Usage
-----
from upxo.interfaces.defdap.ebsd_reader import EBSDReader

rdr = EBSDReader.from_file('scan.ctf', min_grain_size=10)
# or
rdr = EBSDReader.from_file('scan.crc', min_grain_size=10)

rdr.lfi_ebsd    # (ny, nx) int array
rdr.euler_ebsd  # (ny, nx, 3) float array, radians
rdr.quat_ebsd   # (ny, nx, 4) float array
rdr.step_size   # float, microns
"""

import os
import pathlib
import numpy as np

# DefDAP is an optional dependency; import lazily so that the rest of UPXO
# does not break if it is not installed.
try:
    import defdap.ebsd as _defdap_ebsd
    _DEFDAP_AVAILABLE = True
except ImportError:
    _DEFDAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# File-format registry
# ---------------------------------------------------------------------------
_EXT_TO_DATATYPE = {
    '.ctf': 'OxfordText',
    '.crc': 'OxfordBinary',
}


class EBSDReader:
    """
    UPXO-native EBSD file reader backed by DefDAP.

    Parameters are not passed directly; use the class-methods as
    constructors.

    Attributes
    ----------
    lfi_ebsd : np.ndarray, int, shape (ny, nx)
        Grain label field (see module docstring for value conventions).
    euler_ebsd : np.ndarray, float, shape (ny, nx, 3)
        Bunge Euler angles phi1, Phi, phi2 in radians.
    quat_ebsd : np.ndarray, float, shape (ny, nx, 4)
        Unit quaternion coefficients q0, q1, q2, q3 per pixel.
    step_size : float
        Step size in microns.
    file_path : str
        Absolute path to the source EBSD file.
    shape : tuple(int, int)
        Map shape (ny, nx).
    """

    __slots__ = ('lfi_ebsd', 'euler_ebsd', 'quat_ebsd',
                 'step_size', 'file_path', 'shape')

    def __init__(self):
        # Slots are populated by the class-methods; direct construction
        # is intentionally left empty.
        pass

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, file_path, min_grain_size=10,
                  misori_tol=10, data_type=None):
        """
        Load an EBSD map from a .ctf or .crc file and extract all
        UPXO-relevant arrays.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the EBSD file.  Extension must be .ctf or .crc
            unless data_type is supplied explicitly.
        min_grain_size : int, optional
            Minimum grain size in pixels passed to DefDAP's
            ``find_grains()``.  Grains smaller than this are labelled
            -2 in ``lfi_ebsd``.  Default 10.
        misori_tol : float, optional
            Misorientation tolerance in degrees for grain boundary
            detection inside DefDAP.  Default 10.
        data_type : str or None, optional
            Override the DefDAP data_type string.  When None (default)
            the format is inferred from the file extension:
            '.ctf'  -> 'OxfordText'
            '.crc'  -> 'OxfordBinary'

        Returns
        -------
        EBSDReader
            Populated instance with ``lfi_ebsd``, ``euler_ebsd``,
            ``quat_ebsd``, ``step_size``, ``file_path``, ``shape``.

        Raises
        ------
        ImportError
            If DefDAP is not installed.
        FileNotFoundError
            If ``file_path`` does not exist.
        ValueError
            If the file extension is not recognised and ``data_type``
            is not provided.
        """
        _require_defdap()

        file_path = pathlib.Path(file_path).resolve()
        _check_file(file_path)

        if data_type is None:
            data_type = _infer_data_type(file_path)

        # Load via DefDAP
        ebsd_map = _defdap_ebsd.Map(str(file_path), data_type=data_type)

        # Detect grain boundaries and grains
        ebsd_map.find_boundaries(misori_tol=misori_tol)
        ebsd_map.find_grains(min_grain_size=min_grain_size)

        obj = cls()
        obj.file_path = str(file_path)
        obj.step_size = float(ebsd_map.step_size)
        obj.shape = tuple(ebsd_map.shape)       # (ny, nx)

        obj.lfi_ebsd = _extract_lfi(ebsd_map)
        obj.euler_ebsd = _extract_euler(ebsd_map)
        obj.quat_ebsd = _extract_quat(ebsd_map)

        return obj

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def ny(self):
        """Number of rows (y pixels)."""
        return self.shape[0]

    @property
    def nx(self):
        """Number of columns (x pixels)."""
        return self.shape[1]

    @property
    def n_grains(self):
        """Number of grains detected (label values >= 1)."""
        return int(self.lfi_ebsd.max())

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        try:
            return (f"EBSDReader(file='{os.path.basename(self.file_path)}', "
                    f"shape={self.shape}, n_grains={self.n_grains}, "
                    f"step_size={self.step_size} um)")
        except AttributeError:
            return "EBSDReader(uninitialised)"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_defdap():
    if not _DEFDAP_AVAILABLE:
        raise ImportError(
            "DefDAP is required to read EBSD files. "
            "Install it with:  pip install defdap"
        )


def _check_file(path):
    if not path.exists():
        raise FileNotFoundError(f"EBSD file not found: {path}")


def _infer_data_type(path):
    ext = path.suffix.lower()
    if ext not in _EXT_TO_DATATYPE:
        raise ValueError(
            f"Unrecognised file extension '{ext}'. "
            f"Supported: {list(_EXT_TO_DATATYPE.keys())}. "
            f"Pass data_type explicitly to override."
        )
    return _EXT_TO_DATATYPE[ext]


def _extract_lfi(ebsd_map):
    """
    Return grain label field as int32 array of shape (ny, nx).

    DefDAP's grains map uses 1-based IDs for proper grains; -1 for
    remnant boundary pixels; -2 for sub-minimum-size regions; 0 for
    non-indexed points.
    """
    return np.array(ebsd_map.data.grains, dtype=np.int32)


def _extract_euler(ebsd_map):
    """
    Return Bunge Euler angles (phi1, Phi, phi2) in radians.

    DefDAP stores as (3, ny, nx); transpose to (ny, nx, 3) for
    image-convention access: euler_ebsd[row, col] -> [phi1, Phi, phi2].
    """
    # shape: (3, ny, nx) -> (ny, nx, 3)
    return np.asarray(ebsd_map.data.euler_angle,
                      dtype=np.float64).transpose(1, 2, 0)


def _extract_quat(ebsd_map):
    """
    Return quaternion coefficients (q0, q1, q2, q3) per pixel.

    DefDAP stores an object array of Quat instances (shape ny, nx).
    Each Quat exposes .quat_coef as a (4,) array in positive-hemisphere
    convention.

    Returns shape (ny, nx, 4), float64.
    """
    ori = ebsd_map.data.orientation   # (ny, nx) object array of Quat
    ny, nx = ori.shape
    quat_arr = np.empty((ny, nx, 4), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            quat_arr[i, j] = ori[i, j].quat_coef
    return quat_arr
