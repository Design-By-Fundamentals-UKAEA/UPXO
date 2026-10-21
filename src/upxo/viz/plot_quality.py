"""Plot-quality presets: one switch for fonts, line widths, palette and DPI.

Three presets are provided:

* ``"quick"``        small fonts, low DPI; for fast iteration.
* ``"high"``         clean defaults for everyday inspection.
* ``"publication"``  serif fonts, inward ticks, a colour-blind-safe palette
                     and 300 DPI export; editable text in SVG/PDF.

Every preset defines the same set of matplotlib rcParams, so switching from
one preset to another fully replaces the previous one.

matplotlib reads most rcParams when an artist is *created*, so a preset
affects figures drawn after it is applied, not ones already drawn. Settings
passed explicitly to a plotting call (``fontsize=``, ``dpi=``, ...) are
unaffected and take precedence.

Use ``apply_preset`` to change the process-wide default, or
``quality_context`` to scope a preset to a block::

    from upxo.viz import plot_quality
    with plot_quality.quality_context("publication"):
        fig = make_the_figure()
        fig.savefig("fig1.pdf", dpi=plot_quality.savefig_dpi("publication"))
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

import matplotlib
from matplotlib.rcsetup import cycler       # matplotlib's own validated wrapper; no separate cycler dependency

__all__ = [
    "PRESETS", "DEFAULT_PRESET", "OKABE_ITO", "TAB10", "EBSD_COLOR", "SGS_COLOR",
    "preset_names", "get_preset", "apply_preset", "quality_context",
    "current_preset", "savefig_dpi", "describe",
]

# Okabe and Ito (2008) colour-blind-safe palette.
OKABE_ITO: List[str] = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                        "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
TAB10: List[str] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Colours for the two data sources compared throughout the pipeline.
EBSD_COLOR: str = OKABE_ITO[0]
SGS_COLOR: str = OKABE_ITO[3]

DEFAULT_PRESET = "high"


def _preset(*, dpi, savefig_dpi, family, mathfont, size, title, label, tick,
            legend, line_w, axes_w, spines_off, tick_dir, legend_frame,
            colors, savefig_bbox, svg_fonttype, pdf_fonttype) -> Dict:
    return {
        "figure.dpi": dpi,
        "savefig.dpi": savefig_dpi,
        "savefig.bbox": savefig_bbox,
        "font.family": family,
        "mathtext.fontset": mathfont,
        "font.size": size,
        "axes.titlesize": title,
        "axes.labelsize": label,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
        "lines.linewidth": line_w,
        "axes.linewidth": axes_w,
        "axes.spines.top": not spines_off,
        "axes.spines.right": not spines_off,
        "xtick.direction": tick_dir,
        "ytick.direction": tick_dir,
        "legend.frameon": legend_frame,
        "axes.prop_cycle": cycler(color=colors),
        "svg.fonttype": svg_fonttype,
        "pdf.fonttype": pdf_fonttype,
    }


PRESETS: Dict[str, Dict] = {
    "quick": _preset(
        dpi=90, savefig_dpi=120, family=["DejaVu Sans"], mathfont="dejavusans",
        size=9, title=10, label=9, tick=8, legend=8, line_w=1.2, axes_w=0.8,
        spines_off=False, tick_dir="out", legend_frame=True, colors=TAB10,
        savefig_bbox=None, svg_fonttype="path", pdf_fonttype=3),
    "high": _preset(
        dpi=110, savefig_dpi=200, family=["DejaVu Sans"], mathfont="dejavusans",
        size=10.5, title=11.5, label=10.5, tick=9.5, legend=9.5, line_w=1.6,
        axes_w=1.0, spines_off=True, tick_dir="out", legend_frame=False,
        colors=TAB10, savefig_bbox="tight", svg_fonttype="path", pdf_fonttype=3),
    "publication": _preset(
        dpi=150, savefig_dpi=300, family=["STIXGeneral", "DejaVu Serif"],
        mathfont="stix", size=11, title=12, label=11, tick=10, legend=10,
        line_w=1.8, axes_w=1.1, spines_off=True, tick_dir="in",
        legend_frame=False, colors=OKABE_ITO, savefig_bbox="tight",
        svg_fonttype="none", pdf_fonttype=42),
}

_DESCRIPTIONS = {
    "quick": "Small fonts and low DPI. Fastest to draw; for exploring.",
    "high": "Clean look at moderate DPI. Suitable for everyday inspection.",
    "publication": ("Serif fonts, inward ticks, colour-blind-safe palette, "
                    "300 DPI export, editable text in SVG/PDF."),
}

_current: Optional[str] = None


def preset_names() -> List[str]:
    """Names of the available presets, in order of increasing quality."""
    return list(PRESETS)


def describe(name: str) -> str:
    """One-line description of a preset."""
    return _DESCRIPTIONS[_key(name)]


def _key(name: str) -> str:
    key = str(name).strip().lower()
    if key not in PRESETS:
        raise KeyError(f"unknown plot-quality preset {name!r}; choose from {preset_names()}")
    return key


def get_preset(name: str) -> Dict:
    """A copy of the rcParams dict of a preset."""
    return dict(PRESETS[_key(name)])


def apply_preset(name: str) -> None:
    """Make ``name`` the process-wide default for figures created from now on."""
    global _current
    matplotlib.rcParams.update(get_preset(name))
    _current = _key(name)


@contextmanager
def quality_context(name: str):
    """Apply ``name`` inside the block only; rcParams are restored on exit."""
    global _current
    previous = _current
    with matplotlib.rc_context(get_preset(name)):
        _current = _key(name)
        try:
            yield
        finally:
            _current = previous


def current_preset() -> Optional[str]:
    """Name of the preset last applied with :func:`apply_preset` (or active
    inside :func:`quality_context`); None if none has been applied."""
    return _current


def savefig_dpi(name: Optional[str] = None) -> int:
    """DPI to use when saving a figure: that of ``name``, else of the current
    preset, else of the default preset."""
    key = _key(name) if name is not None else (_current or DEFAULT_PRESET)
    return int(PRESETS[key]["savefig.dpi"])
