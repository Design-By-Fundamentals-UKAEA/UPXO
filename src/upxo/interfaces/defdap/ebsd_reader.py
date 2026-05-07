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

        # DefDAP's OxfordTextLoader appends the extension (.ctf/.crc) itself,
        # so we must pass the path WITHOUT the extension (stem only).
        ebsd_map = _defdap_ebsd.Map(str(file_path.with_suffix('')), dataType=data_type)

        # buildQuatArray() must be called before findBoundaries() — DefDAP
        # does not build it automatically on load (0.93.x behaviour).
        ebsd_map.buildQuatArray()

        # Detect grain boundaries and grains
        # findBoundaries uses 'boundDef' (not misori_tol) in 0.93.x
        ebsd_map.findBoundaries(boundDef=misori_tol)
        ebsd_map.findGrains(minGrainSize=min_grain_size)

        obj = cls()
        obj.file_path = str(file_path)
        obj.step_size = float(ebsd_map.stepSize)   # stepSize in 0.93.x
        obj.shape = tuple(ebsd_map.shape)           # (ny, nx)

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
    # Spatial crop
    # ------------------------------------------------------------------

    def crop(self, region, inplace=False):
        """
        Crop the EBSD map to a rectangular sub-region specified as
        **percentage** extents of the full map.

        Parameters
        ----------
        region : sequence of 4 numbers
            ``[xstart%, ystart%, xend%, yend%]`` — all values in the
            range [0, 100].

            * ``xstart%`` / ``xend%`` are measured along the *column*
              (x) axis (i.e. nx dimension).
            * ``ystart%`` / ``yend%`` are measured along the *row* (y)
              axis (i.e. ny dimension).

            Example: ``[10, 10, 80, 80]`` keeps the central 70 % of
            the map in both directions, discarding a 10 % border on
            each side.

        inplace : bool, optional
            If ``True``, modify ``self`` and return ``None``.
            If ``False`` (default), return a new :class:`EBSDReader`
            with the cropped arrays; ``self`` is left unchanged.

        Returns
        -------
        EBSDReader or None
            Cropped reader (when ``inplace=False``) or ``None``
            (when ``inplace=True``).

        Raises
        ------
        ValueError
            If *region* does not have exactly 4 elements, any value is
            outside [0, 100], or start >= end in either axis.

        Notes
        -----
        DefDAP (0.93.x) has no built-in crop, so the crop is performed
        entirely on the extracted NumPy arrays.  ``step_size`` is
        unchanged (cropping does not affect pixel pitch).
        Grain IDs in ``lfi_ebsd`` are preserved as-is — they are
        *not* re-numbered after cropping.
        """
        xs, ys, xe, ye = region
        if not (len(region) == 4):
            raise ValueError("region must have exactly 4 elements "
                             "[xstart%, ystart%, xend%, yend%]")
        for v in (xs, ys, xe, ye):
            if not (0 <= v <= 100):
                raise ValueError(f"All region values must be in [0, 100]; got {v}")
        if xs >= xe:
            raise ValueError(f"xstart% ({xs}) must be less than xend% ({xe})")
        if ys >= ye:
            raise ValueError(f"ystart% ({ys}) must be less than yend% ({ye})")

        ny, nx = self.shape

        # Convert percentage to pixel indices (inclusive on both ends)
        col0 = int(round(xs / 100.0 * nx))
        col1 = int(round(xe / 100.0 * nx))
        row0 = int(round(ys / 100.0 * ny))
        row1 = int(round(ye / 100.0 * ny))

        # Clamp to valid range
        col0 = max(0, min(col0, nx))
        col1 = max(0, min(col1, nx))
        row0 = max(0, min(row0, ny))
        row1 = max(0, min(row1, ny))

        lfi_c   = self.lfi_ebsd[row0:row1, col0:col1].copy()
        euler_c = self.euler_ebsd[row0:row1, col0:col1, :].copy()
        quat_c  = self.quat_ebsd[row0:row1, col0:col1, :].copy()
        new_shape = lfi_c.shape   # (ny_crop, nx_crop)

        print(f"[crop] region=[{xs}, {ys}, {xe}, {ye}]%  →  "
              f"rows {row0}:{row1}, cols {col0}:{col1}  |  "
              f"original {ny}×{nx}  →  cropped {new_shape[0]}×{new_shape[1]}",
              flush=True)

        if inplace:
            self.lfi_ebsd   = lfi_c
            self.euler_ebsd = euler_c
            self.quat_ebsd  = quat_c
            self.shape      = new_shape
            return None
        else:
            obj = EBSDReader.__new__(EBSDReader)
            obj.lfi_ebsd   = lfi_c
            obj.euler_ebsd = euler_c
            obj.quat_ebsd  = quat_c
            obj.step_size  = self.step_size
            obj.file_path  = self.file_path
            obj.shape      = new_shape
            return obj

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
        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:
            def _tqdm(it, **kw):          # silent fallback if tqdm not installed
                print(f"[rechar_lfi] {kw.get('desc', '')} ...", flush=True)
                return it

        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8 for 2D cc3d.")

        lfi = self.lfi_ebsd.astype(np.int32)
        unknown_mask = lfi <= 0

        if not unknown_mask.any():
            print("[rechar_lfi] No non-positive pixels found — nothing to do.")
            return

        n_unknown = int(unknown_mask.sum())
        ny, nx = lfi.shape
        print(f"[rechar_lfi] Map {ny}×{nx}  |  unknown pixels: {n_unknown} "
              f"({100*n_unknown/(ny*nx):.2f}%)  |  connectivity={connectivity}",
              flush=True)

        # ------------------------------------------------------------------
        # Step 1-3: connected-component labelling of unknown pixels
        # ------------------------------------------------------------------
        print("[rechar_lfi] Step 1/5 — labelling unknown connected components …",
              flush=True)
        n_valid = int(lfi.max())

        cc_labels = cc3d.connected_components(
            unknown_mask.astype(np.uint8), connectivity=connectivity
        )                              # (ny, nx), values 1..n_cc, 0 = valid
        n_cc = int(cc_labels.max())
        print(f"[rechar_lfi]           {n_cc} connected components found.", flush=True)

        lfi_aug = lfi.copy()
        lfi_aug[unknown_mask] = (n_valid + cc_labels[unknown_mask]).astype(np.int32)

        # ------------------------------------------------------------------
        # Step 4: adjacency
        # ------------------------------------------------------------------
        print("[rechar_lfi] Step 2/5 — computing grain adjacency …", flush=True)
        neigh = find_neighs2d(lfi_aug, conn=connectivity)

        # ------------------------------------------------------------------
        # Step 5: assign each CC to largest neighbouring valid grain
        # ------------------------------------------------------------------
        print("[rechar_lfi] Step 3/5 — assigning CC pixels to grains …", flush=True)
        valid_pixels = lfi[~unknown_mask]
        ids, counts  = np.unique(valid_pixels, return_counts=True)
        grain_size   = dict(zip(ids.tolist(), counts.tolist()))

        lfi_filled   = lfi.copy()
        cc_assignment = {}

        for cc_id in _tqdm(range(1, n_cc + 1),
                           desc="  assigning CCs", unit="CC", leave=False):
            temp_label       = n_valid + cc_id
            valid_neighbours = [
                nb for nb in neigh.get(temp_label, [])
                if 1 <= nb <= n_valid
            ]
            if not valid_neighbours:
                continue
            largest = max(valid_neighbours,
                          key=lambda g: grain_size.get(g, 0))
            cc_mask = cc_labels == cc_id
            lfi_filled[cc_mask] = largest
            cc_assignment[cc_id] = largest

        n_filled = sum((cc_labels == cc_id).sum()
                       for cc_id in cc_assignment)
        print(f"[rechar_lfi]           {len(cc_assignment)}/{n_cc} CCs assigned "
              f"({n_filled} pixels filled).", flush=True)

        # ------------------------------------------------------------------
        # Step 6-7: vectorized grain-average orientation fill
        # Avoid per-grain Python loops by using scipy.ndimage.mean over the
        # full array at once — O(ny*nx) regardless of grain count.
        # ------------------------------------------------------------------
        print("[rechar_lfi] Step 4/5 — computing grain-average orientations "
              "(vectorized) …", flush=True)
        from scipy.ndimage import mean as _ndmean

        euler = self.euler_ebsd.copy()   # (ny, nx, 3)
        quat  = self.quat_ebsd.copy()    # (ny, nx, 4)

        assigned_grains = sorted(set(cc_assignment.values()))
        n_ag = len(assigned_grains)
        print(f"[rechar_lfi]           {n_ag} unique grains to average over.",
              flush=True)

        # Compute per-grain mean for every channel in one ndimage.mean call.
        # Labels array is the *original* lfi (valid pixels only).
        valid_lfi   = lfi.copy()
        valid_lfi[unknown_mask] = 0          # neutralize unknowns for ndimage

        # euler: 3 channels
        print("[rechar_lfi]           averaging Euler channels (3) …", flush=True)
        avg_euler_arr = np.stack(
            [_ndmean(euler[:, :, ch], labels=valid_lfi, index=assigned_grains)
             for ch in range(3)], axis=1
        )                                    # (n_assigned_grains, 3)
        print("[rechar_lfi]           averaging quaternion channels (4) …",
              flush=True)
        # quat: 4 channels
        avg_quat_arr = np.stack(
            [_ndmean(quat[:, :, ch], labels=valid_lfi, index=assigned_grains)
             for ch in range(4)], axis=1
        )                                    # (n_assigned_grains, 4)

        # Normalise quaternions and enforce positive hemisphere
        norms = np.linalg.norm(avg_quat_arr, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        avg_quat_arr /= norms
        neg = avg_quat_arr[:, 0] < 0
        avg_quat_arr[neg] *= -1
        print("[rechar_lfi]           orientation averages ready.", flush=True)

        # ------------------------------------------------------------------
        # Step 5/5 — vectorised orientation write (O(n_unknown), no loop)
        # ------------------------------------------------------------------
        # lfi_filled already stores the assigned grain ID for every pixel
        # (including the ones that were unknown). Build a compact lookup:
        #   grain_ids[i]   → assigned_grains[i]
        #   gid_to_idx[gid] → row index in avg_euler_arr / avg_quat_arr
        # Then fancy-index the two arrays for all unknown pixels at once.
        # ------------------------------------------------------------------
        n_unk = int(unknown_mask.sum())
        print(f"[rechar_lfi] Step 5/5 — writing orientations for {n_unk} "
              "unknown pixels (vectorised, no loop) …", flush=True)

        grain_ids  = np.array(assigned_grains, dtype=np.intp)
        max_gid    = int(grain_ids.max()) + 1
        gid_to_idx = np.full(max_gid, -1, dtype=np.intp)
        gid_to_idx[grain_ids] = np.arange(len(grain_ids), dtype=np.intp)

        unk_y, unk_x   = np.where(unknown_mask)
        assigned_gids  = lfi_filled[unk_y, unk_x]   # grain ID per unknown px
        row_idx        = gid_to_idx[assigned_gids]   # row in avg arrays

        # Only write pixels that were actually assigned (row_idx >= 0)
        valid_write    = row_idx >= 0
        vy, vx, vr     = unk_y[valid_write], unk_x[valid_write], row_idx[valid_write]

        euler[vy, vx] = avg_euler_arr[vr]   # (n_written, 3)
        quat[vy, vx]  = avg_quat_arr[vr]    # (n_written, 4)

        n_written = int(valid_write.sum())
        print(f"[rechar_lfi]           {n_written} pixels written "
              f"({n_unk - n_written} isolated islands left unfilled).",
              flush=True)

        # ------------------------------------------------------------------
        # Step 8: store back
        # ------------------------------------------------------------------
        self.lfi_ebsd  = lfi_filled
        self.euler_ebsd = euler
        self.quat_ebsd  = quat
        print("[rechar_lfi] Done.", flush=True)


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

    DefDAP 0.93.x stores the label field in ``ebsd_map.grains`` (not
    ``ebsd_map.data.grains``).  Values: >=1 proper grain IDs, -1
    remnant boundary, -2 sub-minimum-size, 0 non-indexed.
    """
    return np.array(ebsd_map.grains, dtype=np.int32)


def _extract_euler(ebsd_map):
    """
    Return Bunge Euler angles (phi1, Phi, phi2) in radians.

    DefDAP 0.93.x stores as ``eulerAngleArray`` with shape (3, ny, nx).
    We transpose to (ny, nx, 3) for image-convention access:
    euler_ebsd[row, col] -> [phi1, Phi, phi2].
    """
    # eulerAngleArray shape: (3, ny, nx) -> transpose to (ny, nx, 3)
    return np.asarray(ebsd_map.eulerAngleArray,
                      dtype=np.float64).transpose(1, 2, 0)


def _extract_quat(ebsd_map):
    """
    Return quaternion coefficients (q0, q1, q2, q3) per pixel.

    DefDAP 0.93.x stores an object array of Quat instances in
    ``ebsd_map.quatArray`` with shape (ny, nx).  Each Quat exposes
    ``.quatCoef`` (camelCase) as a (4,) float array.
    Positive-hemisphere convention applied (q0 >= 0).

    Returns shape (ny, nx, 4), float64.
    """
    # Ensure quatArray is populated (it may need buildQuatArray() first)
    if ebsd_map.quatArray is None:
        ebsd_map.buildQuatArray()

    # DefDAP 0.93.x builds quatComps internally during buildQuatArray().
    # That array has shape (4, ny, nx) and lives at ebsd_map._quatComps
    # (private) — however the public path is to re-derive it from quatArray
    # using np.vectorize, which is ~50× faster than a pure Python double loop.
    ori    = ebsd_map.quatArray              # (ny, nx) object array of Quat
    ny, nx = ori.shape

    # Vectorized extraction: stack all .quatCoef arrays along a new axis.
    # np.frompyfunc avoids Python-level loop overhead.
    get_coef = np.frompyfunc(lambda q: q.quatCoef, 1, 1)
    coef_obj = get_coef(ori)                 # (ny, nx) object array of (4,) arrays
    # Stack into (ny, nx, 4) — np.array on an object array of equal-shape
    # sub-arrays triggers a single C-level copy.
    quat_arr = np.array(coef_obj.tolist(), dtype=np.float64)  # (ny, nx, 4)

    # Positive-hemisphere convention: flip rows where q0 < 0
    neg = quat_arr[:, :, 0] < 0
    quat_arr[neg] *= -1
    return quat_arr
