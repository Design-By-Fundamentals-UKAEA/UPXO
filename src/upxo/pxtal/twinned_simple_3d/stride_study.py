"""
stride_study.py
================
EBSD step-size (stride) sensitivity study for the twinned simple 3D
pipeline: how grain-morphology statistics respond to subsampling the
raw EBSD map at increasing strides, all other detection/crop/clean
settings held fixed.

Ported from ``pxtal/twinned_simple_3d/gui/pages_ebsd.py`` (previously
only reachable through the GUI's ``EBSDInputPage.on_run_stride_study``
-- the module-level ``_run_stride_study``/``_area_fraction_below``
functions themselves had no GUI dependency and are unchanged here).
"""

import numpy as np


def run_stride_study(ctf_path, strides, min_grain_size_detect, misori_tol,
                      crop_region, min_grain_size_clean, connectivity,
                      reuse_subsampled=True, verbose=False):
    """
    Runs Import -> Detect -> Crop -> Clean once per stride value, holding
    every other setting fixed -- isolates how step size alone affects
    computed grain morphology.

    Parameters
    ----------
    ctf_path : str
    strides : iterable of int
        Deduplicated internally; 1 means load the original file directly
        (no subsampling), matching the "no subsampling" baseline case.
    min_grain_size_detect, misori_tol : grain-detection settings.
    crop_region : (xstart_pct, ystart_pct, xend_pct, yend_pct) or None
        Crop settings; None skips cropping entirely.
    min_grain_size_clean, connectivity : cleaning settings.
    reuse_subsampled : bool
        Reuse an existing on-disk subsampled .ctf for a given stride
        instead of regenerating it.
    verbose : bool
        Forwarded to ``clean_and_rechar_from_rdr``.

    Returns
    -------
    list of dict, one per stride, sorted by stride ascending:
        {'stride': int, 'step_size': float, 'n_grains': int,
         'shape': (ny, nx), 'prop_ebsd': dict, 'stat_ebsd': dict}
    """
    from pathlib import Path
    from upxo.interfaces.defdap.ebsd_reader import EBSDReader, write_subsampled_ctf
    from upxo.repgen.repgen2dmcgs import repgen2d

    results = []
    for stride in sorted(set(int(s) for s in strides)):
        load_path = ctf_path
        if stride > 1:
            suffix = f"_s{stride}"
            src = Path(ctf_path)
            dst_path = src.with_name(src.stem + suffix + src.suffix)
            if not (reuse_subsampled and dst_path.exists()):
                write_subsampled_ctf(ctf_path, str(dst_path), stride_x=stride, stride_y=stride)
            load_path = str(dst_path)

        rdr = EBSDReader.load(load_path)
        rdr.detect_grains(min_grain_size=min_grain_size_detect, misori_tol=misori_tol)

        if crop_region is not None:
            xs, ys, xe, ye = crop_region
            rdr = rdr.crop([xs, ys, xe, ye], inplace=False)

        rg = repgen2d.from_tgs(tgs=None, tgstype='ebsd2d', ebsd_file=ctf_path)
        rg.set_ebsd_step(rdr.step_size)
        rg.clean_and_rechar_from_rdr(
            rdr, connectivity=connectivity, min_grain_size=min_grain_size_clean, verbose=verbose)
        rg.compute_ebsd_stats()

        ny, nx = rdr.shape
        results.append({
            'stride': stride,
            'step_size': rdr.step_size,
            'n_grains': len(rg.prop_ebsd),
            'shape': (ny, nx),
            'prop_ebsd': rg.prop_ebsd,
            'stat_ebsd': rg.stat_ebsd,
        })
    return results


def area_fraction_below(prop_ebsd, prop_key, threshold):
    """
    Fraction of TOTAL grain area occupied by grains whose ``prop_key``
    value is below ``threshold`` -- the "volume fraction" (area fraction,
    for a 2D EBSD map) of grains below a given property threshold (e.g.
    the property's own 25th percentile or mean), area-weighted rather
    than a plain grain-count fraction.
    """
    areas, vals = [], []
    for g in prop_ebsd.values():
        if 'area' in g and prop_key in g:
            areas.append(g['area'])
            vals.append(g[prop_key])
    areas = np.asarray(areas, dtype=float)
    vals = np.asarray(vals, dtype=float)
    mask = np.isfinite(areas) & np.isfinite(vals)
    areas, vals = areas[mask], vals[mask]
    total = areas.sum()
    if total <= 0:
        return float('nan')
    return float(areas[vals < threshold].sum() / total)


def bin_width_slider_bounds(data):
    """Slider bounds derived from the same "richness" (N, range) the
    default Freedman-Diaconis bin width itself depends on: more data
    points allow a finer minimum bin width. Returns (lo, hi)."""
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return 0.01, 1.0
    data_range = float(np.max(data) - np.min(data))
    if data_range <= 0:
        return 0.01, 1.0
    max_bins = min(data.size, 100)
    min_bins = 3
    lo = data_range / max_bins
    hi = data_range / min_bins
    if lo <= 0:
        lo = hi / 100.0
    return lo, hi
