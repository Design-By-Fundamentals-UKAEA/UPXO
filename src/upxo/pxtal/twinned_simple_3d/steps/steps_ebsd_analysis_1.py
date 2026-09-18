"""EBSD Analysis-1 -- Part B of the Twinned FCC walkthrough.

Subsample -> import -> detect grains -> crop -> clean & characterize, in
that order. The optional "Step size effect" study (exploring how EBSD step
size distorts downstream measurements) is not wrapped here -- see
stride_study.py directly if you want to explore that.
"""
from pathlib import Path

from upxo.interfaces.defdap.ebsd_reader import EBSDReader, write_subsampled_ctf
from upxo.repgen.repgen2dmcgs import repgen2d

DEFAULT_CTF_FILE = (
    r"C:\Development\EBSD datasets\UKAEA__OFHCCu\OFHC_Cu_dataset\EBSD_pre\warp_out_s2.ctf")


def subsample_and_load(ctf_file=DEFAULT_CTF_FILE, subsample=True,
                        stride_x=5, stride_y=5, reuse_subsampled=False):
    """Step 1 -- optionally subsample (recommended for large maps), then
    load the map (grain detection is a separate step, matching
    EBSDReader.load's own docstring on why load()/detect_grains() are
    split).

    Returns
    -------
    EBSDReader
    """
    load_path = ctf_file
    if subsample:
        src = Path(ctf_file)
        suffix = f"_s{stride_x}" if stride_x == stride_y else f"_s{stride_x}x{stride_y}"
        dst_path = src.with_name(src.stem + suffix + src.suffix)
        if not (reuse_subsampled and dst_path.exists()):
            write_subsampled_ctf(ctf_file, str(dst_path), stride_x=stride_x, stride_y=stride_y)
        load_path = str(dst_path)
    return EBSDReader.load(load_path)


def detect_grains(rdr, min_grain_size=10, misori_tol=10.0):
    """Step 2 -- (re-)detects grains on the already-loaded map in
    place; does not re-read the file from disk. Returns `rdr` for
    convenient chaining."""
    rdr.detect_grains(min_grain_size=min_grain_size, misori_tol=misori_tol)
    return rdr


def crop(rdr, xstart_pct=1.0, ystart_pct=1.0, xend_pct=99.0, yend_pct=99.0):
    """Step 3 -- crops to a percentage sub-region of the full map
    (default trims a 1% border all round, avoiding edge artefacts).
    Returns a NEW EBSDReader (crop is never in-place)."""
    return rdr.crop([xstart_pct, ystart_pct, xend_pct, yend_pct], inplace=False)


def clean_and_characterize(rdr, ctf_file=DEFAULT_CTF_FILE, connectivity=4,
                            min_grain_size=0, verbose=True):
    """Step 4 -- builds the repgen2d object the rest of the pipeline
    works with (`rg`), re-characterizing the cropped/detected map.

    Returns
    -------
    upxo.repgen.repgen2dmcgs.repgen2d
    """
    rg = repgen2d.from_tgs(tgs=None, tgstype='ebsd2d', ebsd_file=ctf_file)
    rg.set_ebsd_step(rdr.step_size)
    rg.clean_and_rechar_from_rdr(
        rdr, connectivity=connectivity, min_grain_size=min_grain_size, verbose=verbose)
    rg.compute_ebsd_stats()
    return rg


def import_and_clean(ctf_file=DEFAULT_CTF_FILE, subsample=True, stride_x=5, stride_y=5,
                      reuse_subsampled=False, min_grain_size_detect=10, misori_tol=10.0,
                      xstart_pct=1.0, ystart_pct=1.0, xend_pct=99.0, yend_pct=99.0,
                      connectivity=4, min_grain_size_clean=0, verbose=True):
    """Convenience one-shot: runs Steps 1-4 in sequence. Prefer the
    individual functions above in the notebook itself (one cell per step,
    so each step's result/log is visible before the next runs); this is
    here for completeness and for the other steps_*.py modules' own tests.

    Returns
    -------
    (rdr, rg) : the final cropped EBSDReader and the built repgen2d object.
    """
    rdr = subsample_and_load(ctf_file, subsample, stride_x, stride_y, reuse_subsampled)
    rdr = detect_grains(rdr, min_grain_size_detect, misori_tol)
    rdr = crop(rdr, xstart_pct, ystart_pct, xend_pct, yend_pct)
    rg = clean_and_characterize(rdr, ctf_file, connectivity, min_grain_size_clean, verbose)
    return rdr, rg
