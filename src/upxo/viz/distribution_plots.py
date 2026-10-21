"""Density plots of one or several samples on a matplotlib axes.

``plot_kde`` draws the Gaussian kernel density estimate of one sample and
``plot_kde_overlay`` draws a density histogram with a kernel density line for
each of several samples. Neither calls ``tight_layout`` or draws the canvas,
so that several axes of a figure can be updated before one redraw. Both
return the values that were plotted, so that a caller can keep their range.
"""
from typing import Dict, Optional

import numpy as np

__all__ = ['DEFAULT_FONT_SIZES', 'histogram_bin_count', 'plot_kde', 'plot_kde_overlay']

DEFAULT_FONT_SIZES: Dict[str, float] = {"tick": 7, "title": 9, "axis": 8, "legend": 8}

_LINE_COLOR = "#2B6CB0"
_NO_DATA_COLOR = "#718096"


def histogram_bin_count(values, bin_width: Optional[float] = None) -> int:
    """Number of histogram bins for ``values``.

    Without ``bin_width``, one bin per 5 values, between 5 and 30. With a
    bin width (in the units of ``values``), the range divided by it, between
    3 and 200; 5 when all values are equal.
    """
    if not bin_width:
        return min(30, max(5, len(values) // 5))
    span = float(np.ptp(values))
    return max(3, min(200, int(round(span / bin_width)))) if span > 0 else 5


def _decorate(ax, xlabel, title, ylabel, sizes, xlim, ylim):
    ax.set_xlabel(xlabel, fontsize=sizes["axis"])
    ax.set_ylabel(ylabel, fontsize=sizes["axis"])
    ax.set_title(title, fontsize=sizes["title"])
    ax.tick_params(labelsize=sizes["tick"])
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_kde(ax, values, xlabel: str, title: str, ylabel: str = "Density",
             bin_width: Optional[float] = None, xlim=None, ylim=None,
             font_sizes: Optional[dict] = None) -> Optional[np.ndarray]:
    """Clear ``ax`` and plot the kernel density estimate of ``values``.

    Non-finite values are dropped. With fewer than two values, or all values
    equal, a single value is drawn as a vertical line and no value as a "No
    data available" note; if the estimate fails a histogram is drawn instead.

    Parameters
    ----------
    bin_width : float, optional
        In the units of ``values``. Sets the histogram bin count of the
        fallback and the kernel bandwidth, ``bin_width / std(values)``.
        By default both are chosen automatically.
    xlim, ylim : (low, high), optional
        Axis limits applied after plotting.
    font_sizes : dict, optional
        Sizes for "tick", "title", "axis" and "legend";
        ``DEFAULT_FONT_SIZES`` by default.

    Returns
    -------
    ndarray or None
        The finite values plotted; None if there are none.
    """
    ax.clear()
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) >= 2 and np.ptp(values) > 1e-12:
        try:
            from scipy.stats import gaussian_kde
            bw_method = (bin_width / np.std(values)) if bin_width else None
            kde = gaussian_kde(values, bw_method=bw_method)
            xs = np.linspace(values.min(), values.max(), 200)
            ax.plot(xs, kde(xs), color=_LINE_COLOR)
            ax.fill_between(xs, kde(xs), color=_LINE_COLOR, alpha=0.2)
        except Exception:
            ax.hist(values, bins=histogram_bin_count(values, bin_width), color=_LINE_COLOR, alpha=0.7)
    elif len(values) == 1:
        ax.axvline(values[0], color=_LINE_COLOR)
    else:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes, color=_NO_DATA_COLOR)

    _decorate(ax, xlabel, title, ylabel, font_sizes or DEFAULT_FONT_SIZES, xlim, ylim)
    return values if len(values) else None


def plot_kde_overlay(ax, series, xlabel: str, title: str, ylabel: str = "Density",
                     bin_width: Optional[float] = None, xlim=None, ylim=None,
                     font_sizes: Optional[dict] = None) -> Optional[np.ndarray]:
    """Clear ``ax`` and overlay, for each ``(label, values, color)`` in
    ``series``, a density histogram and a kernel density line.

    Series without finite values are skipped. The line is drawn when scipy is
    available and the series has more than three distinct values. Parameters
    as for ``plot_kde``; the legend uses the "legend" size, at least 6.

    Returns
    -------
    ndarray or None
        The finite values of all series, pooled; None if there are none.
    """
    ax.clear()
    try:
        from scipy.stats import gaussian_kde
        has_scipy = True
    except ImportError:
        has_scipy = False

    pooled = []
    for label, values, color in series:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        pooled.append(values)
        ax.hist(values, bins=histogram_bin_count(values, bin_width), density=True, alpha=0.55,
                color=color, edgecolor='none', label=f"{label} (n={len(values)})")
        if has_scipy and len(values) >= 2 and len(np.unique(values)) > 3:
            try:
                bw_method = (bin_width / np.std(values)) if bin_width else None
                xs = np.linspace(values.min(), values.max(), 300)
                ax.plot(xs, gaussian_kde(values, bw_method=bw_method)(xs), color=color, linewidth=1.8)
            except Exception:
                pass

    sizes = font_sizes or DEFAULT_FONT_SIZES
    if pooled:
        ax.legend(fontsize=max(6, sizes.get("legend", sizes["tick"])), framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes, color=_NO_DATA_COLOR)
    _decorate(ax, xlabel, title, ylabel, sizes, xlim, ylim)
    return np.concatenate(pooled) if pooled else None
