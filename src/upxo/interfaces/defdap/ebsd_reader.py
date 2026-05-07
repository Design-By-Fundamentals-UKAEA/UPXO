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

    # ------------------------------------------------------------------
    # LFI re-characterisation
    # ------------------------------------------------------------------

    def rechar_lfi(self, connectivity=4):
        """
        Re-characterise ``lfi_ebsd`` by filling every non-positive pixel
        (values <= 0, including DefDAP's -1 remnant-boundary and -2
        sub-minimum-size codes, and non-indexed 0) with the label of the
        spatially largest grain that borders that connected region.
        ``euler_ebsd`` and ``quat_ebsd`` for the filled pixels are then
        updated with the grain-average orientation of the assigned grain.

        Algorithm
        ---------
        1. Identify all non-positive pixels as the *unknown* mask.
        2. Use cc3d to label connected components of the unknown mask
           (connectivity 4 or 8, matching the supplied parameter).
        3. Assign each connected component a temporary unique label
           ``n_grains + cc_id`` in a working copy of ``lfi_ebsd``.
        4. Call ``find_neighs2d`` on the augmented label field to obtain
           the adjacency list for every label (standard + temporary).
        5. For each temporary CC label, find the neighbouring valid grain
           with the greatest pixel count and assign that grain's ID to all
           pixels of the CC.
        6. Compute per-grain mean Euler angles and mean (normalised)
           quaternion from the *original* valid pixels of each grain.
        7. Write those averages into ``euler_ebsd`` and ``quat_ebsd`` at
           the newly filled pixel positions.
        8. Store the updated arrays back to ``self``.

        Parameters
        ----------
        connectivity : int
            cc3d connectivity for labelling the unknown region.
            Valid 2D values: 4 (edge-only) or 8 (edge+corner). Default 4.

        Raises
        ------
        ValueError
            If ``connectivity`` is not 4 or 8.

        Notes
        -----
        After calling this method ``lfi_ebsd`` should contain only
        positive grain labels (>= 1).  ``euler_ebsd`` and ``quat_ebsd``
        are updated in-place on ``self``.  The grain-average orientation
        used for filling is the arithmetic mean of the *original* valid
        pixels — adequate for intra-grain spread; it is not a proper
        quaternion mean but sufficient for neighbourhood-assignment.
        """
        import cc3d
        from upxo.gsdataops.gid_ops import find_neighs2d

        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8 for 2D cc3d.")

        lfi = self.lfi_ebsd.astype(np.int32)
        unknown_mask = lfi <= 0

        if not unknown_mask.any():
            return  # nothing to do

        # ------------------------------------------------------------------
        # Step 1-3: label connected components of unknown pixels
        # ------------------------------------------------------------------
        n_valid = int(lfi.max())   # largest valid grain ID

        cc_labels = cc3d.connected_components(
            unknown_mask.astype(np.uint8), connectivity=connectivity
        )                              # shape (ny, nx), values 1..n_cc, 0 elsewhere
        n_cc = int(cc_labels.max())

        # Build augmented lfi: valid pixels unchanged; unknown pixels get
        # a unique temporary label n_valid + cc_id
        lfi_aug = lfi.copy()
        lfi_aug[unknown_mask] = (n_valid + cc_labels[unknown_mask]).astype(np.int32)

        # ------------------------------------------------------------------
        # Step 4: adjacency of every label in the augmented field
        # ------------------------------------------------------------------
        neigh = find_neighs2d(lfi_aug, conn=connectivity)

        # ------------------------------------------------------------------
        # Step 5: for each CC, assign the largest bordering valid grain
        # ------------------------------------------------------------------
        # Pre-compute grain sizes from original valid pixels
        valid_pixels = lfi[~unknown_mask]
        ids, counts = np.unique(valid_pixels, return_counts=True)
        grain_size = dict(zip(ids.tolist(), counts.tolist()))

        lfi_filled = lfi.copy()
        cc_assignment = {}   # cc_id -> assigned grain ID

        for cc_id in range(1, n_cc + 1):
            temp_label = n_valid + cc_id
            valid_neighbours = [
                nb for nb in neigh.get(temp_label, [])
                if 1 <= nb <= n_valid
            ]
            if not valid_neighbours:
                # Isolated island with no valid neighbour — leave as-is
                continue
            largest = max(valid_neighbours,
                          key=lambda g: grain_size.get(g, 0))
            cc_mask = cc_labels == cc_id
            lfi_filled[cc_mask] = largest
            cc_assignment[cc_id] = largest

        # ------------------------------------------------------------------
        # Step 6-7: compute grain-average orientations and fill
        # ------------------------------------------------------------------
        euler = self.euler_ebsd.copy()   # (ny, nx, 3)
        quat = self.quat_ebsd.copy()     # (ny, nx, 4)

        # Cache grain-average euler and quat (computed once per grain)
        grain_avg_euler = {}
        grain_avg_quat = {}

        assigned_grains = set(cc_assignment.values())
        for gid in assigned_grains:
            orig_mask = (lfi == gid)
            if not orig_mask.any():
                continue
            avg_e = euler[orig_mask].mean(axis=0)
            avg_q = quat[orig_mask].mean(axis=0)
            norm = np.linalg.norm(avg_q)
            if norm > 0:
                avg_q = avg_q / norm
            if avg_q[0] < 0:
                avg_q = -avg_q
            grain_avg_euler[gid] = avg_e
            grain_avg_quat[gid] = avg_q

        for cc_id, gid in cc_assignment.items():
            cc_mask = cc_labels == cc_id
            if gid in grain_avg_euler:
                euler[cc_mask] = grain_avg_euler[gid]
                quat[cc_mask] = grain_avg_quat[gid]

        # ------------------------------------------------------------------
        # Step 8: store back
        # ------------------------------------------------------------------
        self.lfi_ebsd = lfi_filled
        self.euler_ebsd = euler
        self.quat_ebsd = quat


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
