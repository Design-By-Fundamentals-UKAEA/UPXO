"""Matplotlib figures for the before/after comparison (no pyplot state is used)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .texture import density_difference, pole_figure, resolved_grid_points

DISTRIBUTION_KINDS = ('histogram', 'kde', 'histogram + kde', 'cdf', 'box', 'violin')
WEIGHTED_KINDS = ('histogram', 'kde', 'histogram + kde', 'cdf')
COLORS = {'before': '#4878CF', 'after': '#D65F5F'}
MAX_BINS = 400


def colour(value, default):
    """``value`` as a hex colour; ``default`` when it is empty. Raises ``ValueError`` for text that is no colour."""
    text = str(value or '').strip()
    if not text:
        return default
    try:
        return to_hex(to_rgba(text))
    except ValueError:
        raise ValueError(f'{value!r} is not a colour: use a name such as "tab:blue" or a hex code such as "#4878CF".') from None


@dataclass
class PlotStyle:
    """How a distribution is drawn."""

    kind: str = 'histogram'
    bins: int = 40
    bin_width: Optional[float] = None
    """Width of a histogram bin in the units of the plotted values; ``None`` uses ``bins`` bins."""
    bandwidth: Optional[float] = None
    """Standard deviation of the Gaussian of the KDE, in the units of the values; ``None`` uses Silverman's rule."""
    log: bool = False
    density: bool = False
    colors: tuple = (COLORS['before'], COLORS['after'])
    show_mean: bool = False
    show_std: bool = False
    show_median: bool = False


@dataclass
class DrawSeries:
    """One distribution to draw: the legend text, the values, their weights (``None`` for equal weights) and the colour."""

    label: str
    values: np.ndarray
    weights: Optional[np.ndarray] = None
    color: str = COLORS['before']
    integer: bool = False


def weighted_stats(values, weights=None) -> dict:
    """Weighted mean, standard deviation and median (the median of the weighted distribution)."""
    v = np.asarray(values, dtype=float)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)
    if not len(v) or not w.sum() > 0:
        return {'mean': np.nan, 'std': np.nan, 'median': np.nan}
    mean = float(np.average(v, weights=w))
    std = float(np.sqrt(np.average((v - mean) ** 2, weights=w)))
    order = np.argsort(v, kind='stable')
    cumulative = np.cumsum(w[order])
    median = float(v[order][min(np.searchsorted(cumulative, 0.5 * cumulative[-1]), len(v) - 1)])
    return {'mean': mean, 'std': std, 'median': median}


def silverman_bandwidth(values, weights=None) -> float:
    """Silverman's rule of thumb for the bandwidth of a Gaussian KDE (weighted)."""
    v = np.asarray(values, dtype=float)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)
    if len(v) < 2:
        return 1.0
    std = weighted_stats(v, w)['std']
    order = np.argsort(v, kind='stable')
    cumulative = np.cumsum(w[order]) / w.sum()
    last = len(v) - 1
    q75 = v[order][min(np.searchsorted(cumulative, 0.75), last)]
    q25 = v[order][min(np.searchsorted(cumulative, 0.25), last)]
    spread = min(std, (q75 - q25) / 1.34) if q75 > q25 else std
    effective = w.sum() ** 2 / np.sum(w ** 2)
    width = 0.9 * spread * effective ** (-0.2)
    return float(width) if width > 0 else float(max(np.ptp(v), 1.0) * 0.05)


def kde_curve(values, weights, bandwidth, grid) -> np.ndarray:
    """Gaussian kernel density of weighted values on ``grid``; it adds up to the total weight (1 without weights)."""
    v = np.asarray(values, dtype=float)
    w = np.ones(len(v)) / max(len(v), 1) if weights is None else np.asarray(weights, dtype=float)
    out = np.zeros(len(grid))
    step = max(1, int(2_000_000 // max(len(v), 1)))
    for start in range(0, len(grid), step):
        x = grid[start:start + step, None]
        out[start:start + step] = (np.exp(-0.5 * ((x - v[None, :]) / bandwidth) ** 2) @ w) / (bandwidth * np.sqrt(2 * np.pi))
    return out


def bin_edges(series, style: PlotStyle):
    """Common histogram edges for ``series`` and a note: whole numbers for integer data, else ``bin_width`` or ``bins`` bins."""
    filled = [s.values for s in series if len(s.values)]
    pooled = np.concatenate(filled) if filled else np.zeros(1)
    low, high = float(pooled.min()), float(pooled.max())
    note = ''
    if filled and all(s.integer for s in series if len(s.values)):
        return np.arange(np.floor(low) - 0.5, np.ceil(high) + 1.5), note
    if high <= low:
        high = low + (abs(low) * 0.1 or 1.0)
    if style.bin_width:
        n = int(np.ceil((high - low) / style.bin_width))
        if 2 <= n <= MAX_BINS:
            return low + style.bin_width * np.arange(n + 1), note
        note = f'The bin width {style.bin_width:g} gives {n} bins here, so {int(style.bins)} bins were used.'
    return np.linspace(low, high, int(style.bins) + 1), note


def _draw_marks(ax, series, style, row):
    """Vertical lines at the mean (and median) and a bar of one standard deviation either side of the mean."""
    stats = weighted_stats(series.values, series.weights)
    if not np.isfinite(stats['mean']):
        return
    height = 0.93 - 0.07 * row
    transform = ax.get_xaxis_transform()
    if style.show_mean:
        ax.axvline(stats['mean'], ymax=height, color=series.color, ls='--', lw=1.4)
    if style.show_median:
        ax.axvline(stats['median'], ymax=height - 0.12, color=series.color, ls=':', lw=1.6)
    if style.show_std:
        lo, hi = stats['mean'] - stats['std'], stats['mean'] + stats['std']
        ax.plot([lo, hi], [height, height], transform=transform, color=series.color, lw=1.6, solid_capstyle='butt')
        for end in (lo, hi):
            ax.plot([end, end], [height - 0.02, height + 0.02], transform=transform, color=series.color, lw=1.6)
        ax.plot([stats['mean']], [height], transform=transform, color=series.color, marker='o', ms=4)


def stats_text(series, unit='') -> str:
    """``mean 12.3 +- 4.5, median 11.9 um`` for a legend (mathtext plus-minus)."""
    stats = weighted_stats(series.values, series.weights)
    suffix = f' {unit}' if unit else ''
    return f"mean {stats['mean']:.4g} $\\pm$ {stats['std']:.4g}, median {stats['median']:.4g}{suffix}"


def compare_figure(series, style: PlotStyle, *, xlabel='', ylabel='', title='', subtitle='', footnote='', unit='',
                   note_unit=None):
    """Distributions of one quantity on the two sides: histogram, KDE, both, CDF, box or violin plot.

    ``series`` are :class:`DrawSeries`. A box or violin plot cannot show weights, so a weighted quantity falls back to
    the histogram with the KDE. The legend carries the text of each series, with its mean and standard deviation
    when those are marked. ``figure.compare_notes`` lists what was chosen automatically (bin width, bandwidth).
    """
    if style.kind not in DISTRIBUTION_KINDS:
        raise ValueError(f'kind must be one of {DISTRIBUTION_KINDS}')
    kind, notes = style.kind, []
    shown = [s for s in series if len(s.values)]
    weighted = any(s.weights is not None for s in shown)
    if weighted and kind not in WEIGHTED_KINDS:
        kind = 'histogram + kde'
        notes.append(f'A {style.kind} plot cannot show weights; a histogram with the KDE is drawn.')
    fig = Figure(figsize=(7.4, 4.8), layout='constrained')
    ax = fig.add_subplot(111)
    marks = style.show_mean or style.show_std or style.show_median
    if kind in ('box', 'violin'):
        if shown:
            positions = list(range(1, len(shown) + 1))
            values = [s.values for s in shown]
            if kind == 'box':
                parts = ax.boxplot(values, positions=positions, patch_artist=True)
                for patch, s in zip(parts['boxes'], shown):
                    patch.set_facecolor(s.color)
                    patch.set_alpha(0.6)
            else:
                parts = ax.violinplot(values, positions=positions, showmedians=True)
                for body, s in zip(parts['bodies'], shown):
                    body.set_facecolor(s.color)
                    body.set_alpha(0.6)
            ax.set_xticks(positions, [s.label for s in shown], fontsize=8)
        ax.set_ylabel(xlabel)
        if style.log:
            ax.set_yscale('log')
    else:
        with_hist = kind in ('histogram', 'histogram + kde')
        with_kde = kind in ('kde', 'histogram + kde')
        edges, note = bin_edges(shown, style) if with_hist else (None, '')
        if note:
            notes.append(note)
        density = weighted or with_kde or style.density
        pooled = np.concatenate([s.values for s in shown]) if shown else np.zeros(1)
        for s in shown:
            label = s.label + (f'\n{stats_text(s, unit)}' if marks else '')
            if kind == 'cdf':
                order = np.argsort(s.values, kind='stable')
                w = np.ones(len(s.values)) if s.weights is None else s.weights
                ax.step(s.values[order], np.cumsum(w[order]) / w.sum(), where='post', color=s.color, label=label)
                continue
            curve_label = label
            if with_hist:
                counts, _ = np.histogram(s.values, bins=edges, weights=s.weights)
                height = counts / np.diff(edges) if density else counts
                if density and s.weights is None:
                    height = height / len(s.values)
                ax.stairs(height, edges, fill=True, alpha=0.5, color=s.color, label=label)
                ax.stairs(height, edges, color=s.color, lw=1.0)
                curve_label = None
            if with_kde:
                h = style.bandwidth or silverman_bandwidth(s.values, s.weights)
                low = float(pooled.min()) - 3 * h
                low = max(low, 0.0) if float(pooled.min()) >= 0 else low
                grid = np.linspace(low, float(pooled.max()) + 3 * h, 500)
                ax.plot(grid, kde_curve(s.values, s.weights, h, grid), color=s.color, lw=2.0, label=curve_label)
        ax.set_xlabel(xlabel)
        if kind == 'cdf':
            ax.set_ylabel('cumulative fraction')
        elif density:
            ax.set_ylabel(ylabel or 'density')
        else:
            ax.set_ylabel(ylabel or 'count')
        if style.log:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(axis='x', useOffset=False, style='plain')
        if marks and kind != 'cdf':
            for row, s in enumerate(shown):
                _draw_marks(ax, s, style, row)
        where = f' {unit if note_unit is None else note_unit}'.rstrip()
        if with_kde and style.bandwidth is None and shown:
            notes.append('KDE bandwidth ' + ', '.join(f'{silverman_bandwidth(s.values, s.weights):.3g}{where}' for s in shown)
                         + " (Silverman's rule)")
        if with_hist and style.bin_width is None and edges is not None and len(edges) > 1:
            notes.append(f'bin width {edges[1] - edges[0]:.3g}{where}')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc='best')
    fig.suptitle(title, fontweight='bold')
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color='0.3')
    text = '; '.join(x for x in ([footnote] if footnote else []) + notes if x)
    if text:
        fig.supxlabel(text, fontsize=7.5, color='0.35')
    fig.compare_notes = notes
    return fig


def distribution_figure(before, after, kind='histogram', *, bins=40, log=False,
                        density=False, xlabel='', title='', labels=('before', 'after')):
    """Before/after distribution of one quantity as a histogram, KDE, CDF, box or violin plot."""
    if kind not in DISTRIBUTION_KINDS:
        raise ValueError(f'kind must be one of {DISTRIBUTION_KINDS}')
    series = [DrawSeries(f'{name} ({len(np.asarray(d))})', np.asarray(d, dtype=float), color=color)
              for d, name, color in zip((before, after), labels, (COLORS['before'], COLORS['after']))]
    fig = compare_figure(series, PlotStyle(kind=kind, bins=int(bins), log=log, density=density), xlabel=xlabel, title=title)
    return fig


def matched_change_figure(rows, *, quantity='value', units='', kind='scatter', bins=40, log=False,
                          labels=('before', 'after'), colors=None, subtitle='', footnote='', noun='items'):
    """Per-item change for items present on both sides: before-vs-after or ratio histogram."""
    colors = colors or (COLORS['before'], COLORS['after'])
    fig = Figure(figsize=(6.8, 5.2), layout='constrained')
    ax = fig.add_subplot(111)
    before = np.array([r['before'] for r in rows], dtype=float)
    after = np.array([r['after'] for r in rows], dtype=float)
    suffix = f' ({units})' if units else ''
    if not rows:
        ax.text(0.5, 0.5, 'No items on both sides', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig
    if kind == 'scatter':
        ax.scatter(before, after, s=14, color=colors[1], alpha=0.7)
        lo, hi = min(before.min(), after.min()), max(before.max(), after.max())
        ax.plot([lo, hi], [lo, hi], color='gray', linewidth=1, label='no change')
        ax.set_xlabel(f'{quantity}: {labels[0]}{suffix}')
        ax.set_ylabel(f'{quantity}: {labels[1]}{suffix}')
        ax.legend()
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
    elif kind == 'ratio':
        ratio = after[before > 0] / before[before > 0]
        ax.hist(ratio, bins=int(bins), color=colors[1], alpha=0.7)
        ax.axvline(1.0, color='gray', linewidth=1)
        ax.set_xlabel(f'{quantity}: {labels[1]} / {labels[0]}')
        ax.set_ylabel(f'number of {noun}')
        if log:
            ax.set_yscale('log')
    else:
        raise ValueError("kind must be 'scatter' or 'ratio'")
    fig.suptitle(f'{quantity}: {len(rows)} {noun} on both sides', fontweight='bold')
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color='0.3')
    if footnote:
        fig.supxlabel(footnote, fontsize=7.5, color='0.35')
    return fig


def texture_figure(eulers_before, eulers_after, settings, titles=('Before', 'After'), noun='features'):
    """Pole figures before and after, and (density modes) the after-minus-before difference."""
    pf_before = pole_figure(eulers_before, settings)
    pf_after = pole_figure(eulers_after, settings)
    density_modes = settings.plot_mode in ('density', 'hybrid')
    fig = Figure(figsize=(17 if density_modes else 11.5, 5.6), layout='constrained')
    axes = fig.subplots(1, 3 if density_modes else 2)
    x_label, y_label, z_label = settings.axis_labels
    labels = dict(x_label=x_label, y_label=y_label, z_label=z_label)
    grid = resolved_grid_points(settings)
    family = settings.pole_family
    counts = (len(np.asarray(eulers_before)), len(np.asarray(eulers_after)))
    for ax, pf, name, n in zip(axes[:2], (pf_before, pf_after), titles, counts):
        title = f'{name} ({n} {noun})'
        if settings.plot_mode == 'scatter':
            pf.plot_scatter(pole_family=family, ax=ax, projection=settings.projection,
                            hemisphere=settings.hemisphere, title=title, **labels)
        else:
            pf.plot_density(
                pole_family=family, ax=ax, projection=settings.projection, grid_points=grid,
                half_width_deg=settings.half_width_deg, contour_scale=settings.contour_scale,
                cmap=settings.cmap, levels=settings.levels,
                colorbar_decimals=settings.colorbar_decimals, title=title,
                overlay_scatter=settings.plot_mode == 'hybrid',
                scatter_hemisphere=settings.hemisphere, **labels)
    diff = None
    if density_modes:
        diff = density_difference(eulers_before, eulers_after, settings)
        ax = axes[2]
        zi = diff['zi_diff']
        peak = np.nanmax(np.abs(zi))
        peak = peak if np.isfinite(peak) and peak > 0 else 1.0
        levels = np.linspace(-peak, peak, 2 * settings.levels + 1)
        mappable = ax.contourf(diff['Xi'], diff['Yi'], zi, levels=levels, cmap=settings.difference_cmap)
        ax.contour(diff['Xi'], diff['Yi'], zi, levels=levels, colors='black', linewidths=0.3, alpha=0.25)
        fmt = f'%.{settings.colorbar_decimals}f' if settings.colorbar_decimals is not None else None
        label = f'({titles[1]})/max - ({titles[0]})/max' if settings.unit_normalize else f'MUD {titles[1]} - MUD {titles[0]}'
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, format=fmt, label=label)
        ax.add_artist(Circle((0, 0), 1.0, color='black', fill=False, linewidth=1.5))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')
        ax.set_title(f"Difference (IQR {diff['iqr']:.4g})", fontweight='bold')
    return fig, diff


def subset_preview_figure(labels, lo, hi, *, title='Subset box'):
    """Three mid-box slices of the parent labels with the box drawn on them."""
    from matplotlib.patches import Rectangle

    a = np.asarray(labels)
    lo, hi = np.asarray(lo), np.asarray(hi)
    centre = ((lo + hi) // 2).clip(0, np.asarray(a.shape) - 1)
    palette = np.random.default_rng(0).permutation(int(a.max()) + 1)
    fig = Figure(figsize=(13, 4.6), layout='constrained')
    axes = fig.subplots(1, 3)
    for ax, (u, v, w) in zip(axes, ((0, 1, 2), (0, 2, 1), (1, 2, 0))):
        section = np.take(a, int(centre[w]), axis=w)
        ax.imshow(palette[section].T, origin='lower', cmap='nipy_spectral',
                  interpolation='nearest', aspect='equal')
        ax.add_patch(Rectangle((lo[u] - 0.5, lo[v] - 0.5), hi[u] - lo[u], hi[v] - lo[v],
                               fill=False, edgecolor='black', linewidth=2.2))
        ax.set_xlabel('xyz'[u] + ' (voxels)')
        ax.set_ylabel('xyz'[v] + ' (voxels)')
        ax.set_title('xyz'[w] + ' = ' + str(int(centre[w])), fontsize=11)
    fig.suptitle(title, fontweight='bold')
    return fig
