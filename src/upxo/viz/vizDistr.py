"""
vizDistr.py — Distribution visualisation for UPXO grain structure analyses.

Provides the DistrViz class for plotting scalar grain property distributions
(area, perimeter, aspect ratio, …) and angular misorientation distributions
(MDF). Designed to complement ebsdviz.plot_mdf — use DistrViz.plot_mdf when
peaks are not yet computed; use ebsdviz.plot_mdf for fully annotated MDF with
peak labels and KDE from the peaks dict.

Typical usage
-------------
Grain size:
    dv = DistrViz(areas, label='Grain area', units='µm²')
    fig, ax = dv.plot_hist(bins=40, show_kde=True, step_size=rdr.step_size)
    plt.show()
    dv.print_stats()

MDF (lightweight, no peaks dict required):
    dv = DistrViz.from_mdf(mdf)
    fig, ax = dv.plot_mdf(mdf)
    plt.show()

Multiple properties:
    fig, axes = DistrViz.multi(
        {'Grain area': areas, 'Aspect ratio': ar, 'Perimeter': perim},
        units_dict={'Grain area': 'µm²', 'Aspect ratio': '', 'Perimeter': 'µm'},
        step_size=rdr.step_size,
    )
    plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# Physical units for common grain morphological properties.
# Imported by EBSDReader.see_distr and repgen2d.see_distr to avoid duplication.
PROP_UNITS = {
    'area':               'µm²',
    'perimeter':          'µm',
    'eq_diameter':        'µm',
    'major_axis_length':  'µm',
    'minor_axis_length':  'µm',
    'aspect_ratio':       '',
    'eccentricity':       '',
    'solidity':           '',
    'npixels':            'px',
}

# CSL reference angles for cubic symmetry (Σ label → disorientation angle °)
_CSL_ANGLES = {
    'S3':   60.00,
    'S5':   36.87,
    'S7':   38.21,
    'S9':   38.94,
    'S11':  50.48,
    'S13a': 22.62,
    'S13b': 27.80,
}


class DistrViz:
    """
    Distribution visualiser for scalar grain properties and MDF data.

    Parameters
    ----------
    data : array-like
        1-D array of values. NaN/Inf are stripped automatically.
    label : str
        Property name — used in axis labels and titles.
    units : str
        Unit string (e.g. 'µm²', '°'). Appended to x-label when non-empty.
    """

    def __init__(self, data, label='value', units=''):
        arr = np.asarray(data, dtype=float).ravel()
        self.data = arr[np.isfinite(arr)]
        self.label = label
        self.units = units

    # ── Alternate constructors ─────────────────────────────────────────────────

    @classmethod
    def from_mdf(cls, mdf):
        """Build from an mdf dict (output of compute_mdf_from_quats)."""
        return cls(mdf['miso_deg'], label='Misorientation angle', units='°')

    # ── Statistics ─────────────────────────────────────────────────────────────

    @property
    def stats(self):
        """Dict of descriptive statistics computed from self.data."""
        d = self.data
        return {
            'n':      len(d),
            'min':    float(d.min()),
            'max':    float(d.max()),
            'mean':   float(d.mean()),
            'median': float(np.median(d)),
            'std':    float(d.std()),
            'skew':   float(sp_stats.skew(d)),
            'kurt':   float(sp_stats.kurtosis(d)),
            'p10':    float(np.percentile(d, 10)),
            'p90':    float(np.percentile(d, 90)),
        }

    def print_stats(self):
        """Print a compact statistics summary to stdout."""
        s = self.stats
        u = f' ({self.units})' if self.units else ''
        print(f"{self.label}{u}  [n={s['n']}]")
        print(f"  min={s['min']:.3f}  max={s['max']:.3f}  "
              f"mean={s['mean']:.3f}  median={s['median']:.3f}")
        print(f"  std={s['std']:.3f}  skew={s['skew']:.3f}  "
              f"kurt={s['kurt']:.3f}")
        print(f"  P10={s['p10']:.3f}  P90={s['p90']:.3f}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _xlabel(self, step_size=None):
        parts = [self.label]
        if self.units or step_size is not None:
            inner = self.units
            if step_size is not None:
                sep = ',  ' if inner else ''
                inner += f'{sep}step={step_size} µm'
            parts.append(f'({inner})')
        return '  '.join(parts)

    def _stat_title(self):
        s = self.stats
        return (f'{self.label} distribution  '
                f'(n={s["n"]},  mean={s["mean"]:.2f},  std={s["std"]:.2f})')

    def _draw_stat_lines(self, ax):
        s = self.stats
        ax.axvline(s['mean'],   color='k',      ls='--', lw=1.2,
                   label=f'mean = {s["mean"]:.2f}')
        ax.axvline(s['median'], color='darkorange', ls=':',  lw=1.2,
                   label=f'median = {s["median"]:.2f}')

    # ── Unified dispatcher ─────────────────────────────────────────────────────

    def plot(self, vis='hist', bins=40, show_kde=True, show_stats=True,
             color='steelblue', figsize=(7, 4), log_scale=False,
             step_size=None, bw_method='scott', fill=True, ax=None):
        """
        Unified plot dispatcher — routes to plot_hist, plot_kde, or
        plot_hist_kde based on *vis*.

        Parameters
        ----------
        vis : str
            ``'hist'``, ``'kde'``, or ``'hist_kde'``.
        bins : int
            Histogram bin count (used by ``'hist'`` and ``'hist_kde'``).
        show_kde : bool
            KDE overlay on histogram (``'hist'`` only).
        show_stats : bool
            Annotate mean / median lines.
        color : str
        figsize : tuple
        log_scale : bool
            Log x-axis (``'hist'`` only).
        step_size : float or None
            Appended to x-label when provided.
        bw_method : str or float
            KDE bandwidth selector (``'kde'`` only).
        fill : bool
            Fill KDE area (``'kde'`` only).
        ax : Axes or None

        Returns
        -------
        fig, ax
        """
        if vis == 'hist':
            return self.plot_hist(bins=bins, show_kde=show_kde,
                                  show_stats=show_stats, color=color,
                                  figsize=figsize, log_scale=log_scale,
                                  step_size=step_size, ax=ax)
        elif vis == 'kde':
            return self.plot_kde(bw_method=bw_method, fill=fill,
                                 color=color, show_stats=show_stats,
                                 figsize=figsize, step_size=step_size, ax=ax)
        elif vis == 'hist_kde':
            return self.plot_hist_kde(bins=bins, color=color,
                                      show_stats=show_stats, figsize=figsize,
                                      step_size=step_size, ax=ax)
        else:
            raise ValueError(
                f"vis must be 'hist', 'kde', or 'hist_kde'; got '{vis!r}'"
            )

    # ── Scalar distribution plots ──────────────────────────────────────────────

    def plot_hist(self, bins=40, show_kde=True, show_stats=True,
                  color='steelblue', figsize=(7, 4), log_scale=False,
                  step_size=None, ax=None):
        """
        Histogram with optional KDE overlay and mean/median annotations.

        Parameters
        ----------
        bins : int
        show_kde : bool
            KDE curve scaled to match histogram counts.
        show_stats : bool
            Draw vertical mean and median lines.
        color : str
        figsize : tuple
        log_scale : bool
            Log x-axis.
        step_size : float or None
            EBSD step size — appended to x-label when provided.
        ax : Axes or None

        Returns
        -------
        fig, ax
        """
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        counts, edges, _ = ax.hist(self.data, bins=bins,
                                   color=color, edgecolor='k',
                                   alpha=0.75, label='histogram')
        if show_kde:
            kde = sp_stats.gaussian_kde(self.data)
            x = np.linspace(self.data.min(), self.data.max(), 400)
            bw = edges[1] - edges[0]
            ax.plot(x, kde(x) * len(self.data) * bw,
                    color='crimson', lw=1.8, label='KDE')

        if show_stats:
            self._draw_stat_lines(ax)
            ax.legend(fontsize=8, framealpha=0.7)

        ax.set_xlabel(self._xlabel(step_size))
        ax.set_ylabel('Count')
        ax.set_title(self._stat_title())
        if log_scale:
            ax.set_xscale('log')
        if own_fig:
            plt.tight_layout()
        return fig, ax

    def plot_kde(self, bw_method='scott', fill=True, color='steelblue',
                 show_stats=True, figsize=(7, 4), step_size=None, ax=None):
        """
        Pure KDE plot (probability density).

        Parameters
        ----------
        bw_method : str or float
            Bandwidth selector passed to scipy.stats.gaussian_kde.
        fill : bool
            Fill area under the KDE curve.
        color, figsize, step_size, ax
            Standard plot options.

        Returns
        -------
        fig, ax
        """
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        kde = sp_stats.gaussian_kde(self.data, bw_method=bw_method)
        x = np.linspace(self.data.min(), self.data.max(), 400)
        y = kde(x)
        if fill:
            ax.fill_between(x, y, alpha=0.3, color=color)
        ax.plot(x, y, color=color, lw=2)

        if show_stats:
            self._draw_stat_lines(ax)
            ax.legend(fontsize=8, framealpha=0.7)

        ax.set_xlabel(self._xlabel(step_size))
        ax.set_ylabel('Density')
        ax.set_title(self._stat_title())
        if own_fig:
            plt.tight_layout()
        return fig, ax

    def plot_hist_kde(self, bins=40, color='steelblue', show_stats=True,
                     figsize=(7, 4), step_size=None, ax=None):
        """
        Density-normalised histogram with KDE overlay.

        Returns
        -------
        fig, ax
        """
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.hist(self.data, bins=bins, density=True,
                color=color, edgecolor='k', alpha=0.5, label='histogram')
        kde = sp_stats.gaussian_kde(self.data)
        x = np.linspace(self.data.min(), self.data.max(), 400)
        ax.plot(x, kde(x), color='crimson', lw=2, label='KDE')

        if show_stats:
            self._draw_stat_lines(ax)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.set_xlabel(self._xlabel(step_size))
        ax.set_ylabel('Density')
        ax.set_title(self._stat_title())
        if own_fig:
            plt.tight_layout()
        return fig, ax

    # ── MDF plot ───────────────────────────────────────────────────────────────

    def plot_mdf(self, mdf, show_csl=True, show_stats=True,
                 angle_max=65.0, figsize=(8, 4), ax=None):
        """
        Bar-chart MDF from a pre-computed mdf dict with optional CSL markers.

        Lighter alternative to ebsdviz.plot_mdf — does not require the peaks
        dict. Use ebsdviz.plot_mdf when peak labels and KDE are needed.

        Parameters
        ----------
        mdf : dict
            Output of compute_mdf_from_quats. Required keys:
            'hist_bin_centers', 'hist_density', 'hist_bin_edges',
            'n_pairs', 'mean_angle', 'std_angle'.
        show_csl : bool
            Draw dashed vertical lines at common cubic CSL angles.
        show_stats : bool
            Annotate mean ± std in the legend.
        angle_max : float
            X-axis upper limit (degrees).
        figsize : tuple
        ax : Axes or None

        Returns
        -------
        fig, ax
        """
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        bw = float(mdf['hist_bin_edges'][1] - mdf['hist_bin_edges'][0])
        ax.bar(mdf['hist_bin_centers'], mdf['hist_density'],
               width=bw, color='steelblue', edgecolor='k',
               linewidth=0.4, alpha=0.85)

        if show_csl:
            ymax = float(np.max(mdf['hist_density']))
            for lbl, angle in _CSL_ANGLES.items():
                if angle <= angle_max:
                    ax.axvline(angle, color='firebrick',
                               lw=0.9, ls='--', alpha=0.75)
                    ax.text(angle + 0.3, ymax * 0.93, lbl,
                            color='firebrick', fontsize=7,
                            va='top', rotation=90)

        if show_stats:
            mean_a = mdf['mean_angle']
            std_a  = mdf['std_angle']
            ax.axvline(mean_a, color='k', ls='--', lw=1.2,
                       label=f'mean = {mean_a:.1f}°  (σ = {std_a:.1f}°)')
            ax.legend(fontsize=8, framealpha=0.7)

        ax.set_xlabel('Misorientation angle (°)')
        ax.set_ylabel('Probability density (°⁻¹)')
        ax.set_title(f'Grain-boundary MDF  '
                     f'(n={mdf["n_pairs"]} pairs,  cubic symmetry)')
        ax.set_xlim(0, angle_max)
        if own_fig:
            plt.tight_layout()
        return fig, ax

    # ── Multi-property grid ────────────────────────────────────────────────────

    @classmethod
    def multi(cls, data_dict, units_dict=None, step_size=None,
              bins=40, show_kde=True, show_stats=True,
              ncolumns=2, figsize_per=(5, 3.5), color='steelblue',
              log_scale=False):
        """
        Plot distributions for multiple grain properties in a subplot grid.

        Parameters
        ----------
        data_dict : dict
            {label: array-like} of grain properties to plot.
        units_dict : dict or None
            {label: units_str}. Missing keys default to no units.
        step_size : float or None
            Passed to each subplot for x-label annotation.
        bins : int
        show_kde : bool
        show_stats : bool
        ncolumns : int
        figsize_per : tuple
            (width, height) per panel in inches.
        color : str
        log_scale : bool

        Returns
        -------
        fig, axes  (axes is a flat ndarray)
        """
        labels = list(data_dict.keys())
        n = len(labels)
        nrows = (n + ncolumns - 1) // ncolumns
        figsize = (figsize_per[0] * ncolumns, figsize_per[1] * nrows)
        fig, axes = plt.subplots(nrows, ncolumns, figsize=figsize)
        axes_flat = np.array(axes).flatten()
        units_dict = units_dict or {}

        for ax, label in zip(axes_flat, labels):
            dv = cls(data_dict[label], label=label,
                     units=units_dict.get(label, ''))
            dv.plot_hist(bins=bins, show_kde=show_kde, show_stats=show_stats,
                         color=color, log_scale=log_scale,
                         step_size=step_size, ax=ax)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig, axes_flat


# ── Multi-group overlaid distribution plot ─────────────────────────────────────

def plot_grouped_distributions(
        data,
        prop_labels=None,
        group_colors=None,
        group_labels=None,
        bins=40,
        bw_method='scott',
        peak_prominence=0.01,
        figsize_per=(5, 4),
        dpi=110,
        suptitle='Property distributions by group',
        ncols=None,
        fontsize=9.0,
        show_hist=True,
        show_peaks=True,
        show_legend=True,
        x_margin=0.03,
        do_tight_layout=True,
):
    """
    Overlaid histogram + KDE + peak markers for multiple properties and groups.

    Generic plotting function — no knowledge of grain structures or UPXO data
    formats.  Data must be pre-extracted into plain arrays before calling.

    Parameters
    ----------
    data : dict
        ``{prop_name: {group_name: array-like}}`` — one entry per property,
        each containing one array per group.  Arrays may be empty; empty/size-1
        groups are silently skipped.
    prop_labels : dict or None
        ``{prop_name: display_label}`` for axis / title text.  Missing keys
        fall back to the prop_name itself.
    group_colors : dict or None
        ``{group_name: colour_string}``.  Missing keys cycle through a default
        palette.
    group_labels : dict or None
        ``{group_name: display_label}`` for legend entries.  Missing keys fall
        back to the group_name itself.
    bins : int
        Number of histogram bins (shared x-range across groups per property).
    bw_method : str or float
        Bandwidth selector passed to ``scipy.stats.gaussian_kde``.
    peak_prominence : float
        Fraction of KDE maximum used as minimum prominence for ``find_peaks``.
    figsize_per : tuple
        ``(width, height)`` in inches per subplot panel.
    dpi : int
        Figure resolution.
    suptitle : str
        Figure-level title.
    ncols : int or None
        Subplot grid columns.  ``None`` places all panels in a single row.
    fontsize : float
        Base font size; tick labels use ``fontsize-2``, legend ``fontsize-2``,
        peak annotations ``fontsize-3``, suptitle ``fontsize+1``.
    show_hist : bool
        Draw histogram bars behind the KDE curves.  Default ``True``.
    show_peaks : bool
        Draw vertical dashed lines and value annotations at KDE peaks.
        Default ``True``.
    show_legend : bool
        Draw a per-group legend on each subplot.  Default ``True``.
    x_margin : float
        Fractional padding added to both sides of the x-axis so that tick
        labels are never clipped at the axis boundary.  Default ``0.03``.
    do_tight_layout : bool
        Call ``plt.tight_layout()`` before returning.  Set to ``False`` when
        the caller needs to adjust the figure (e.g. to add a colorbar) before
        finalising the layout.  Default ``True``.

    Returns
    -------
    fig, axes : Figure and 2-D axes array (shape ``(nrows, ncols_used)``).
    """
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    _DEFAULT_PALETTE = [
        '#4878CF', '#D65F5F', '#59A14F', '#888888',
        '#F28E2B', '#76B7B2', '#E15759', '#B07AA1',
    ]

    prop_labels  = prop_labels  or {}
    group_colors = group_colors or {}
    group_labels = group_labels or {}

    prop_names = list(data.keys())
    n_props    = len(prop_names)

    # Assign default colours to any group not in group_colors
    all_groups = []
    for gd in data.values():
        for g in gd:
            if g not in all_groups:
                all_groups.append(g)
    for i, g in enumerate(all_groups):
        group_colors.setdefault(g, _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)])

    _ncols = n_props if ncols is None else max(1, min(ncols, n_props))
    _nrows = int(np.ceil(n_props / _ncols))
    fig, axes = plt.subplots(
        _nrows, _ncols,
        figsize=(_ncols * figsize_per[0], _nrows * figsize_per[1]),
        dpi=dpi, squeeze=False,
    )

    for spare in range(n_props, _nrows * _ncols):
        axes[spare // _ncols, spare % _ncols].set_visible(False)

    for idx, pname in enumerate(prop_names):
        ax     = axes[idx // _ncols, idx % _ncols]
        groups = data[pname]

        arrays = {g: np.asarray(v, dtype=float) for g, v in groups.items()}
        arrays = {g: a[np.isfinite(a)] for g, a in arrays.items() if len(a) > 1}

        if not arrays:
            ax.set_visible(False)
            continue

        combined  = np.concatenate(list(arrays.values()))
        vmin, vmax = combined.min(), combined.max()
        if vmin == vmax:
            ax.set_visible(False)
            continue

        rng       = vmax - vmin
        pad       = x_margin * rng
        bin_edges = np.linspace(vmin, vmax, bins + 1)
        bin_w     = bin_edges[1] - bin_edges[0]
        xs        = np.linspace(vmin, vmax, 600)

        for grp, vals in arrays.items():
            colour = group_colors.get(grp, '#333333')

            if show_hist:
                counts, _ = np.histogram(vals, bins=bin_edges, density=True)
                ax.bar(bin_edges[:-1], counts, width=bin_w,
                       color=colour, alpha=0.28, edgecolor='none', align='edge')

            kde = gaussian_kde(vals, bw_method=bw_method)
            ys  = kde(xs)
            ax.plot(xs, ys, color=colour, linewidth=1.8)

            if show_peaks:
                peak_idx, _ = find_peaks(ys, prominence=peak_prominence * ys.max())
                for pi in peak_idx:
                    ax.axvline(xs[pi], color=colour, linewidth=0.8,
                               linestyle='--', alpha=0.7)
                    ax.text(xs[pi], ys[pi] * 1.03, f'{xs[pi]:.3g}',
                            fontsize=fontsize - 3, color=colour,
                            ha='center', va='bottom', rotation=90)

            if show_legend:
                mn, mx = vals.min(), vals.max()
                mu, sd = vals.mean(), vals.std()
                disp = group_labels.get(grp, grp)
                lbl  = (f'{disp} (n={len(vals)})\n'
                        f'  µ={mu:.3g}  σ={sd:.3g}  [{mn:.3g}, {mx:.3g}]')
                ax.plot([], [], color=colour, linewidth=2.5, label=lbl)

        xlabel = prop_labels.get(pname, pname)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel('Density', fontsize=fontsize)
        ax.set_title(xlabel, fontsize=fontsize)
        ax.set_xlim(vmin - pad, vmax + pad)
        if show_legend:
            ax.legend(fontsize=fontsize - 2, loc='upper right', framealpha=0.85,
                      handlelength=1.2)
        ax.tick_params(labelsize=fontsize - 2)

    fig.suptitle(suptitle, fontsize=fontsize + 1, y=1.02)
    if do_tight_layout:
        plt.tight_layout()
    return fig, axes


def plot_repr_rank(
        repr_rank_ng: dict,
        figsize=None,
        dpi: int = 100,
        fontsize_annot: float = 8.0,
        fontsize_tick: float = 9.0,
        fontsize_title: float = 9.0,
        fontsize_suptitle: float = 11.0,
) -> None:
    """
    Five vertically stacked heatmaps showing the per-property rank of every
    MC time slice under each representativeness metric (ratio, Wasserstein,
    energy distance, KS statistic, Anderson–Darling statistic).

    Colour encodes rank within each column independently:
    green = best (rank 1), red = worst (rank N).  Cell text shows the raw
    numeric score.  Rows are ordered best-to-worst by the aggregate score
    (inherited from the DataFrame sort order in ``repr_rank_ng``).

    Ranking rule per column:
    - ratio, property columns  : rank by ``|value − 1|`` ascending
      (closest to 1.0 = best)
    - ratio, aggregate column  : rank by value ascending (lowest = best)
    - wasserstein / energy     : rank by value ascending (lowest = best)

    Parameters
    ----------
    repr_rank_ng : dict
        ``{'ratio': df, 'wasserstein': df, 'energy': df}`` — as stored in
        ``repgen2d.repr_rank_ng`` after calling ``find_repr_mcgs_props``.
    figsize : tuple or None
        Override default figure size.  Default auto-computes from data shape.
    dpi : int
        Figure resolution.
    fontsize_annot : float
        Font size for the numeric value printed in each cell.
    fontsize_tick : float
        Font size for axis tick labels (slice keys on y-axis, column names
        on x-axis).
    fontsize_title : float
        Font size for each panel title.
    fontsize_suptitle : float
        Font size for the overall figure title.
    """
    metrics = ('ratio', 'wasserstein', 'energy', 'ks', 'ad')
    titles = {
        'ratio':       'Ratio  (mean offset)\n1.0 = perfect  |  green = closest to 1.0',
        'wasserstein': 'Wasserstein  (shape distance)\n0 = identical  |  green = smallest',
        'energy':      'Energy  (shape distance)\n0 = identical  |  green = smallest',
        'ks':          'KS statistic  (max CDF gap)\n0 = identical  |  green = smallest',
        'ad':          'Anderson–Darling  (tail-sensitive CDF)\n0 = identical  |  green = smallest',
    }
    fmt = {'ratio': '{:.3f}', 'wasserstein': '{:.4f}', 'energy': '{:.4f}',
           'ks': '{:.4f}', 'ad': '{:.4f}'}

    sample_df = repr_rank_ng['wasserstein']
    n_slices, n_cols = sample_df.shape
    if figsize is None:
        figsize = (max(10, n_cols * 1.8), max(20, n_slices * 0.65 * 5))

    fig, axes = plt.subplots(5, 1, figsize=figsize, dpi=dpi)

    for ax, metric in zip(axes, metrics):
        df   = repr_rank_ng[metric]
        vals = df.values.astype(float)
        cols = list(df.columns)
        rows = [str(k) for k in df.index]
        nr, nc = vals.shape

        rank_mat = np.empty_like(vals)
        for j, col in enumerate(cols):
            col_vals = vals[:, j]
            if metric == 'ratio' and col != 'aggregate':
                order = np.argsort(np.abs(col_vals - 1.0))
            else:
                order = np.argsort(col_vals)
            ranks = np.empty(nr, dtype=float)
            ranks[order] = np.arange(nr)
            rank_mat[:, j] = ranks

        norm_rank = rank_mat / max(nr - 1, 1)   # 0 = best, 1 = worst

        ax.imshow(norm_rank, cmap='RdYlGn_r', vmin=0, vmax=1,
                  aspect='auto', interpolation='nearest')
        for i in range(nr):
            for j in range(nc):
                ax.text(j, i, fmt[metric].format(vals[i, j]),
                        ha='center', va='center',
                        fontsize=fontsize_annot, color='black')

        ax.set_xticks(range(nc))
        ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=fontsize_tick)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(rows, fontsize=fontsize_tick)
        ax.set_ylabel('MC time slice  (top = best aggregate)',
                      fontsize=fontsize_tick)
        ax.set_title(titles[metric], fontsize=fontsize_title, pad=8)
        ax.axvline(nc - 1.5, color='white', linewidth=2)

    fig.suptitle('MC–EBSD representativeness ranking',
                 fontsize=fontsize_suptitle, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_normalized_prop_distributions(
        ebsd_data: dict,
        mc_data: dict,
        props: list,
        scores: dict | None = None,
        prop_labels: dict | None = None,
        bins: int = 40,
        bw_method='scott',
        figsize_per: tuple = (5, 4),
        dpi: int = 100,
        ncols: int | None = None,
        fontsize: float = 9.0,
        show_hist: bool = True,
        show_peaks: bool = True,
        legend_loc: str = 'upper right',
        legend_ncol: int = 1,
        legend_fontsize: float | None = None,
) -> None:
    """
    Overlaid normalised property distributions for EBSD (merged) and MC slices.

    Each distribution is normalised by its own mean before plotting, matching
    the normalisation used in ``find_repr_mcgs_props``.  All curves are therefore
    centred near 1.0 on the x-axis and are directly shape-comparable.

    Wasserstein and energy distances are annotated in each subplot legend when
    ``scores`` is provided.

    Parameters
    ----------
    ebsd_data : dict
        ``{prop: array}`` of EBSD-merged property values, each already divided
        by its own mean.
    mc_data : dict
        ``{slice_key: {prop: array}}`` of MC property values, each already
        divided by its own mean.
    props : list of str
        Ordered list of property names to plot.
    scores : dict or None
        ``{slice_key: {prop: {'wasserstein': v, 'energy': v}}}`` extracted from
        ``repr_rank_ng``.  When supplied, each MC curve's legend entry is
        annotated with ``W=...  E=...`` for the per-property distance.
    prop_labels : dict or None
        ``{prop: display_label}``.  Defaults to ``f'{prop}  (mean normalized)'``.
    bins, bw_method, figsize_per, dpi, ncols, fontsize, show_hist, show_peaks
        Forwarded to :func:`plot_grouped_distributions`.
    legend_loc : str
        Legend location string passed to ``ax.legend(loc=...)``.
        Examples: ``'upper right'``, ``'upper left'``, ``'lower right'``,
        ``'center left'``, ``'best'``.  Default ``'upper right'``.
    legend_ncol : int
        Number of columns in the legend.  Values > 1 split entries side-by-side,
        reducing legend height and — when entries are uniform in width — the
        overall legend footprint.  Default ``1`` (single column).
    legend_fontsize : float or None
        Font size for legend text.  Reducing this is the most direct way to
        shrink the legend box since box width is driven by label text length.
        Defaults to ``fontsize - 2`` when None.
    """
    _MC_PALETTE = [
        '#4878CF', '#D65F5F', '#59A14F', '#F28E2B',
        '#76B7B2', '#E15759', '#B07AA1', '#FF9DA7',
    ]

    if prop_labels is None:
        prop_labels = {p: f'{p}  (mean normalized)' for p in props}

    group_colors = {'EBSD (merged)': '#222222'}
    for i, k in enumerate(mc_data):
        group_colors[f'MC  t={k}'] = _MC_PALETTE[i % len(_MC_PALETTE)]

    data = {}
    for p in props:
        groups = {'EBSD (merged)': ebsd_data[p]}
        for k, mc_props in mc_data.items():
            groups[f'MC  t={k}'] = mc_props[p]
        data[p] = groups

    # Always defer layout so we can post-process legends uniformly.
    fig, axes = plot_grouped_distributions(
        data,
        prop_labels=prop_labels,
        group_colors=group_colors,
        bins=bins, bw_method=bw_method,
        figsize_per=figsize_per, dpi=dpi, ncols=ncols, fontsize=fontsize,
        show_hist=show_hist, show_peaks=show_peaks,
        suptitle='Normalised property distributions — EBSD (merged) vs MC slices',
        do_tight_layout=False,
    )

    # Append score annotations and re-apply legend with user-controlled style.
    _MC_PALETTE_LIST = list(_MC_PALETTE)
    for idx, p in enumerate(props):
        ax = axes.flat[idx]
        if scores is not None:
            for i, k in enumerate(mc_data):
                if k in scores and p in scores[k]:
                    sc = scores[k][p]
                    w = sc.get('wasserstein', float('nan'))
                    e = sc.get('energy', float('nan'))
                    colour = _MC_PALETTE_LIST[i % len(_MC_PALETTE_LIST)]
                    ax.plot([], [], color=colour, lw=0,
                            label=f'  → W={w:.4f}  E={e:.4f}')
        ax.legend(fontsize=legend_fontsize if legend_fontsize is not None else fontsize - 2,
                  loc=legend_loc, framealpha=0.85,
                  ncol=legend_ncol)

    plt.tight_layout()
    plt.show()


def plot_qq_comparison(
        ebsd_data: dict,
        mc_data: dict,
        props: list,
        prop_labels: dict | None = None,
        figsize_per: tuple = (4, 4),
        dpi: int = 100,
        ncols: int | None = None,
        fontsize: float = 9.0,
) -> None:
    """
    Quantile–Quantile (Q-Q) comparison of EBSD vs MC grain property distributions.

    A Q-Q plot maps the quantiles of one distribution against the quantiles of
    another at the same probability levels (0 % to 100 %).  Both distributions
    are normalised by their own mean before comparison, so the x- and y-axes
    share the same dimensionless scale centred near 1.0.

    Interpretation
    --------------
    - Points on the diagonal (y = x) — the two distributions have identical
      shape at that quantile.  Perfect agreement.
    - Points **above** the diagonal — the MC distribution has *larger* values
      than EBSD at that quantile (heavier upper tail or higher spread in MC).
    - Points **below** the diagonal — the MC distribution has *smaller* values
      than EBSD at that quantile.
    - Deviations concentrated in the **lower-left** — fine/small grains differ.
    - Deviations concentrated in the **upper-right** — large/coarse grains differ.

    One subplot is drawn per property; each MC slice is a separate line.
    The dashed black diagonal marks perfect distributional agreement.

    Parameters
    ----------
    ebsd_data : dict
        ``{prop: array}`` of EBSD-merged values, each normalised by own mean.
    mc_data : dict
        ``{slice_key: {prop: array}}`` of MC values, each normalised by own mean.
    props : list of str
        Properties to plot.
    prop_labels : dict or None
        ``{prop: display_label}``.  Defaults to ``f'{prop}  (mean normalized)'``.
    figsize_per : tuple
        ``(width, height)`` per subplot in inches.
    dpi : int
    ncols : int or None
        Subplot grid columns.  ``None`` places all panels in a single row.
    fontsize : float
    """
    _MC_PALETTE = [
        '#4878CF', '#D65F5F', '#59A14F', '#F28E2B',
        '#76B7B2', '#E15759', '#B07AA1', '#FF9DA7',
    ]

    if prop_labels is None:
        prop_labels = {p: f'{p}  (mean normalized)' for p in props}

    n_props = len(props)
    _ncols = n_props if ncols is None else max(1, min(ncols, n_props))
    _nrows = int(np.ceil(n_props / _ncols))
    fig, axes = plt.subplots(
        _nrows, _ncols,
        figsize=(_ncols * figsize_per[0], _nrows * figsize_per[1]),
        dpi=dpi, squeeze=False,
    )

    q = np.linspace(0, 100, 300)

    for idx, p in enumerate(props):
        ax = axes[idx // _ncols, idx % _ncols]
        ebsd_q = np.percentile(ebsd_data[p], q)

        all_vals = list(ebsd_q)
        for i, (k, mc_props) in enumerate(mc_data.items()):
            mc_q = np.percentile(mc_props[p], q)
            all_vals.extend(mc_q)
            colour = _MC_PALETTE[i % len(_MC_PALETTE)]
            ax.plot(ebsd_q, mc_q, color=colour, lw=1.5, label=f'MC  t={k}')

        vmin, vmax = min(all_vals), max(all_vals)
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.0, label='perfect match')

        ax.set_xlabel(f'EBSD  {prop_labels[p]}', fontsize=fontsize)
        ax.set_ylabel(f'MC  {prop_labels[p]}', fontsize=fontsize)
        ax.set_title(prop_labels[p], fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1)
        ax.legend(fontsize=fontsize - 2, framealpha=0.8)

    for spare in range(n_props, _nrows * _ncols):
        axes[spare // _ncols, spare % _ncols].set_visible(False)

    fig.suptitle('Q-Q plots — EBSD (merged) vs MC slices  (mean-normalised)',
                 fontsize=fontsize + 1, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_ebsd_tvf(
        tvf_result: dict,
        figsize: tuple = (7, 4),
        dpi: int = 100,
        fontsize: float = 9.0,
        title: str = 'EBSD grain-role area fractions',
) -> None:
    """
    Horizontal bar chart of EBSD twin area fraction broken down by grain role.

    Bars are drawn for each of the four grain-role categories:

    - **Pure parents** — matrix grains; never a twin of any grain.
    - **Primary twins** — first-generation twins whose parent is a pure parent.
    - **Secondary twins** — twins whose parent is itself an intermediate
      (twin-of-a-twin, 2nd generation).
    - **Intermediate twins** — grains that are simultaneously a twin of one
      grain and a parent of another (twin chains).

    The overall twin area fraction (primary + secondary + intermediate) is
    annotated on the figure.

    Parameters
    ----------
    tvf_result : dict
        Output of ``repgen2d.compute_ebsd_tvf``.  Must contain keys
        ``'pure_parent_frac'``, ``'primary_twin_frac'``,
        ``'secondary_twin_frac'``, ``'intermediate_frac'``,
        ``'overall_twin_frac'``.
    figsize : tuple
        Figure size ``(width, height)`` in inches.
    dpi : int
        Figure resolution.
    fontsize : float
        Base font size for labels and tick marks.
    title : str
        Figure title.
    """
    categories = [
        ('Pure parents',     tvf_result['pure_parent_frac'],    '#555555'),
        ('Primary twins',    tvf_result['primary_twin_frac'],   '#4878CF'),
        ('Secondary twins',  tvf_result['secondary_twin_frac'], '#F28E2B'),
        ('Intermediate twins', tvf_result['intermediate_frac'], '#59A14F'),
    ]
    labels = [c[0] for c in categories]
    values = [c[1] for c in categories]
    colors = [c[2] for c in categories]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.5)

    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=fontsize - 1)

    ax.set_xlabel('Area fraction', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1)
    ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.invert_yaxis()

    overall = tvf_result['overall_twin_frac']
    ax.text(0.98, 0.04, f'Overall TVF = {overall:.4f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=fontsize, color='#222222',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                      edgecolor='#cccccc'))

    plt.tight_layout()
    plt.show()
