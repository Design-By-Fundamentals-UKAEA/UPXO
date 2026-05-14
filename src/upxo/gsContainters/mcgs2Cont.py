"""
mcgs2Cont.py — Container for 2-D Monte-Carlo grain-structure temporal datasets.

Default import::

    import upxo.gsContainters.mcgs2Cont as gs_cntnr2_mc

Typical usage::

    cntr = gs_cntnr2_mc.MC_GS_Container2d.by_upxoMCSIM_gsset_GEN(
        indb='path/to/dashboard.xls',
        mctimeStart=2, mctimeStep=1, mctimeEnd=-1,
    )
    fig, axes = cntr.plot_temporal_distributions(props=['area', 'aspect_ratio'])
    plt.show()
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy


class MC_GS_Container2d:
    """
    Container for a set of characterised 2-D MC grain structures spanning
    a range of Monte-Carlo time slices.

    Attributes
    ----------
    indb : str
        Path to the UPXO input-dashboard Excel file.
    gsset : dict
        ``{tslice: mcgs2_grain_structure}`` — characterised grain structures
        keyed by MC time-step index.
    mctimeStart : int
        First MC time-slice index included.
    mctimeStep : int
        Stride between included time slices.
    mctimeEnd : int
        Last MC time-slice index (exclusive).  ``-1`` collects to the
        penultimate slice.
    char_kwargs : dict
        Keyword arguments passed to :meth:`char_morph_2d` for every slice.

    Notes
    -----
    Do **not** call ``__init__`` directly.  Use the class-method constructor
    :meth:`by_upxoMCSIM_gsset_GEN`.
    """

    __slots__ = (
        'indb',
        'gsset',
        'mctimeStart',
        'mctimeStep',
        'mctimeEnd',
        'char_kwargs',
        'stats_table',
    )

    @staticmethod
    def _prop_vals(gs, p: str) -> 'np.ndarray':
        """Return a 1-D finite array of values for property *p* from *gs.prop*.

        Handles both DataFrame (rows = grains, columns = properties) and
        dict-of-dicts (``{grain_id: {prop_name: value}}``) structures.
        Returns an empty array when the property is absent or gs has no prop.
        """
        prop = getattr(gs, 'prop', None)
        if prop is None:
            return np.array([], dtype=float)
        try:
            import pandas as _pd
            if isinstance(prop, _pd.DataFrame):
                if p not in prop.columns:
                    return np.array([], dtype=float)
                vals = prop[p].dropna().to_numpy(dtype=float)
            else:
                vals = np.array(
                    [prop[gid][p] for gid in prop
                     if isinstance(prop[gid], dict) and p in prop[gid]],
                    dtype=float,
                )
        except Exception:
            return np.array([], dtype=float)
        return vals[np.isfinite(vals)]

    # ── Display ────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return f'UPXO. MCS2D.Container. ID({id(self)})'

    def __repr__(self) -> str:
        n = len(self.gsset) if self.gsset else 0
        return (
            f'MC_GS_Container2d('
            f'tslices={n}, '
            f'mctimeStart={self.mctimeStart}, '
            f'mctimeStep={self.mctimeStep}, '
            f'mctimeEnd={self.mctimeEnd})'
        )

    # ── Constructor ────────────────────────────────────────────────────────────

    @classmethod
    def by_upxoMCSIM_gsset_GEN(
        cls,
        indb: str,
        upxoMCSIM_gsset=None,
        mctimeStart: int = 2,
        mctimeStep: int = 1,
        mctimeEnd: int = -1,
        **kwargs,
    ) -> 'MC_GS_Container2d':
        """
        Build a container by running (or reusing) a 2-D MC grain-growth
        simulation and characterising every selected temporal slice.

        Parameters
        ----------
        indb : str
            Path to the UPXO input-dashboard Excel file used to drive the
            MC simulation.
        upxoMCSIM_gsset : :class:`~upxo.ggrowth.mcgs.mcgs` or None, optional
            A pre-existing ``mcgs`` object whose simulation has already been
            run and whose grains have been detected.  When ``None`` (default)
            a fresh simulation is run using *indb*.
        mctimeStart : int, optional
            Index into ``pxt.m`` from which to start collecting slices.
            Default ``2`` (skips the initial, unpopulated slices).
        mctimeStep : int, optional
            Stride between collected time slices.  Default ``1``.
        mctimeEnd : int, optional
            Index into ``pxt.m`` at which to stop (exclusive).
            ``-1`` (default) collects up to but not including the last slice.
        **kwargs
            Overrides for :meth:`char_morph_2d` keyword arguments.  Any key
            not supplied falls back to the defaults listed below.

            Default ``char_morph_2d`` settings
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            .. code-block:: python

                use_version=2,
                bbox=True,                   bbox_ex=False,
                npixels=True,                identify_pixel_locations=True,
                npixels_gb=False,
                area=True,                   aspect_ratio=True,
                solidity=True,               major_axis_length=True,
                minor_axis_length=True,
                circularity=False,           eccentricity=False,
                euler_number=False,          moments_hu=False,
                append=False,                saa=True,
                throw=False,                 char_grain_positions=False,
                find_neigh=False,            char_gb=False,
                make_skim_prop=True,         get_grain_coords=True,

        Returns
        -------
        MC_GS_Container2d
            Populated container with :attr:`gsset` keyed by MC time-step
            index.

        Notes
        -----
        The underlying ``mcgs`` simulation object is deleted after
        characterisation to free memory.

        Examples
        --------
        >>> import upxo.gsContainters.mcgs2Cont as gs_cntnr2_mc
        >>> cntr = gs_cntnr2_mc.MC_GS_Container2d.by_upxoMCSIM_gsset_GEN(
        ...     indb='path/to/dashboard.xls',
        ...     mctimeStart=2, mctimeStep=2, mctimeEnd=-1,
        ...     circularity=True,
        ... )
        """
        from upxo.ggrowth.mcgs import mcgs as _mcgs

        _CHAR_DEFAULTS: dict = dict(
            use_version=2,
            bbox=True,                   bbox_ex=False,
            npixels=True,                identify_pixel_locations=True,
            npixels_gb=False,
            area=True,                   aspect_ratio=True,
            solidity=True,               major_axis_length=True,
            minor_axis_length=True,
            circularity=False,           eccentricity=False,
            euler_number=False,          moments_hu=False,
            append=False,                saa=True,
            throw=False,                 char_grain_positions=False,
            find_neigh=False,            char_gb=False,
            make_skim_prop=True,         get_grain_coords=True,
        )
        char_kw = {**_CHAR_DEFAULTS, **kwargs}

        _SCALAR_PROPS = (
            'area', 'aspect_ratio', 'solidity',
            'major_axis_length', 'minor_axis_length',
            'circularity', 'eccentricity',
        )
        active_props = [p for p in _SCALAR_PROPS if char_kw.get(p, False)]

        # ── Run or reuse simulation ──────────────────────────────────────────
        if upxoMCSIM_gsset is None:
            pxt = _mcgs(input_dashboard=indb)
            pxt.simulate(verbose=False)
            pxt.detect_grains()
        else:
            pxt = upxoMCSIM_gsset

        print(f'Temporal slices available: {pxt.m}')

        # ── Characterise each selected temporal slice ────────────────────────
        gsset: dict = {}
        for tslice in pxt.m[mctimeStart:mctimeEnd:mctimeStep]:
            gs = deepcopy(pxt.gs[tslice])
            gs.char_morph_2d(**char_kw)
            gsset[tslice] = gs

            # (min, mean, max) summary per active property
            triplets = []
            for p in active_props:
                vals = cls._prop_vals(gs, p)
                if vals.size > 0:
                    triplets.append(
                        f'{p}=({vals.min():.3g}, '
                        f'{vals.mean():.3g}, '
                        f'{vals.max():.3g})'
                    )
            summary = '  '.join(triplets)
            print(f'  tslice={tslice:4d}  grains={gs.n}  {summary}')

        del pxt  # free simulation memory

        obj = cls.__new__(cls)
        obj.indb         = indb
        obj.gsset        = gsset
        obj.mctimeStart  = mctimeStart
        obj.mctimeStep   = mctimeStep
        obj.mctimeEnd    = mctimeEnd
        obj.char_kwargs  = char_kw
        obj.stats_table  = None
        return obj

    # ── Temporal distribution visualisation ───────────────────────────────────

    def plot_temporal_distributions(
        self,
        props: list[str] | None = None,
        ncols: int = 2,
        figsize_per: tuple[float, float] = (5, 4),
        dpi: int = 110,
        bins: int = 40,
        bw_method: str | float = 'scott',
        peak_prominence: float = 0.01,
        fontsize: float = 10.0,
        suptitle: str = 'Temporal evolution of grain-property distributions',
        cmap: str = 'nipy_spectral',
        show_stats_table: bool = True,
    ):
        """
        Plot KDE distributions — one curve per temporal slice — overlaid for
        each selected morphological property.  No histograms; no on-graph
        peak markers.  A single vertical colorbar maps line colour to MC
        time-slice index.  After the figure a summary statistics table is
        printed and stored in :attr:`stats_table`.

        Parameters
        ----------
        props : list of str, optional
            Property names to plot.  Defaults to all properties that were
            enabled at characterisation time (via :attr:`char_kwargs`).
            Valid values: ``'area'``, ``'aspect_ratio'``, ``'solidity'``,
            ``'major_axis_length'``, ``'minor_axis_length'``,
            ``'circularity'``, ``'eccentricity'``.
        ncols : int, optional
            Number of subplot columns.  Default ``2``.
        figsize_per : tuple of float, optional
            ``(width, height)`` in inches per panel.  Default ``(5, 4)``.
        dpi : int, optional
            Figure resolution.  Default ``110``.
        bins : int, optional
            Bin count used only for computing the shared x-range.
            Default ``40``.
        bw_method : str or float, optional
            Bandwidth selector for ``scipy.stats.gaussian_kde``.
            Default ``'scott'``.
        peak_prominence : float, optional
            Fraction of KDE maximum used as minimum peak prominence when
            computing the dominant-peak value for the statistics table.
            Default ``0.01``.
        fontsize : float, optional
            Base font size.  Default ``10.0``.
        suptitle : str, optional
            Figure-level title.
        cmap : str, optional
            Matplotlib colormap name.  Default ``'nipy_spectral'``.
        show_stats_table : bool, optional
            If ``True`` (default), display the combined statistics table
            after the figure using ``IPython.display`` (or ``print`` as
            fallback).  Set to ``False`` to suppress the table; the data
            is still computed and stored in :attr:`stats_table`.

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
        axes : ndarray of :class:`matplotlib.axes.Axes`

        Side-effects
        ------------
        Sets :attr:`stats_table` — ``{prop_name: pandas.DataFrame}`` where
        each DataFrame has columns
        ``['tslice', 'n', 'mean', 'std', 'CI_lower_95', 'CI_upper_95', 'peak_max']``.
        The full combined table is also printed to stdout.

        Examples
        --------
        >>> fig, axes = cntr.plot_temporal_distributions(
        ...     props=['area', 'aspect_ratio'],
        ...     ncols=2,
        ...     cmap='nipy_spectral',
        ... )
        >>> import matplotlib.pyplot as plt
        >>> plt.show()
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import pandas as pd
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks
        from upxo.viz.vizDistr import plot_grouped_distributions

        _PROP_LABELS: dict = {
            'area':               'Area (px²)',
            'aspect_ratio':       'Aspect ratio',
            'solidity':           'Solidity',
            'major_axis_length':  'Major axis length (px)',
            'minor_axis_length':  'Minor axis length (px)',
            'circularity':        'Circularity',
            'eccentricity':       'Eccentricity',
        }

        if props is None:
            props = [p for p in _PROP_LABELS if self.char_kwargs.get(p, False)]

        tslices  = sorted(self.gsset.keys())
        norm     = mpl.colors.Normalize(vmin=tslices[0], vmax=tslices[-1])
        cmap_obj = cm.get_cmap(cmap)
        group_colors = {str(t): cmap_obj(norm(t)) for t in tslices}
        group_labels = {str(t): f't = {t}' for t in tslices}

        # ── Build data dict ──────────────────────────────────────────────────
        data: dict = {}
        for p in props:
            prop_groups: dict = {}
            for t in tslices:
                vals = self._prop_vals(self.gsset[t], p)
                if vals.size > 1:
                    prop_groups[str(t)] = vals
            if prop_groups:
                data[p] = prop_groups

        if not data:
            print(
                'No data to plot. Verify that props are valid and that '
                'the container was built with those properties enabled.'
            )
            return None, None

        # ── KDE-only plot (no hist, no peak markers, no per-group legend) ────
        fig, axes = plot_grouped_distributions(
            data            = data,
            prop_labels     = _PROP_LABELS,
            group_colors    = group_colors,
            group_labels    = group_labels,
            bins            = bins,
            bw_method       = bw_method,
            peak_prominence = peak_prominence,
            figsize_per     = figsize_per,
            dpi             = dpi,
            suptitle        = suptitle,
            ncols           = ncols,
            fontsize        = fontsize,
            show_hist       = False,
            show_peaks      = False,
            show_legend     = False,
            x_margin        = 0.04,
            do_tight_layout = False,
        )

        # ── Single vertical colorbar on the right ────────────────────────────
        fig.subplots_adjust(right=0.84)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.025, 0.70])
        sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label('MC time slice', fontsize=fontsize)
        cb.set_ticks(tslices)
        cb.set_ticklabels([str(t) for t in tslices])
        cb.ax.tick_params(labelsize=fontsize - 2)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # ── Statistics table ─────────────────────────────────────────────────
        rows = []
        for p in props:
            label_p = _PROP_LABELS.get(p, p)
            for t in tslices:
                vals = self._prop_vals(self.gsset[t], p)
                if vals.size < 2:
                    continue
                n    = len(vals)
                mu   = float(vals.mean())
                sd   = float(vals.std())
                ci_lo = mu - 1.96 * sd / np.sqrt(n)
                ci_hi = mu + 1.96 * sd / np.sqrt(n)

                # Dominant KDE peak (highest peak; fallback to KDE mode)
                kde  = gaussian_kde(vals, bw_method=bw_method)
                xs   = np.linspace(vals.min(), vals.max(), 600)
                ys   = kde(xs)
                pidx, _ = find_peaks(ys, prominence=peak_prominence * ys.max())
                peak_max = float(
                    xs[pidx[np.argmax(ys[pidx])]] if len(pidx) > 0
                    else xs[np.argmax(ys)]
                )

                rows.append({
                    'Property':    label_p,
                    'tslice':      t,
                    'n':           n,
                    'mean':        round(mu, 4),
                    'std':         round(sd, 4),
                    'CI_lower_95': round(ci_lo, 4),
                    'CI_upper_95': round(ci_hi, 4),
                    'peak_max':    round(peak_max, 4),
                })

        df_all = pd.DataFrame(rows, columns=[
            'Property', 'tslice', 'n', 'mean', 'std',
            'CI_lower_95', 'CI_upper_95', 'peak_max',
        ])

        # Store per-property and display
        self.stats_table = {
            p: df_all[df_all['Property'] == _PROP_LABELS.get(p, p)]
               .reset_index(drop=True)
            for p in props
        }

        if show_stats_table:
            try:
                from IPython.display import display as _display
                _display(df_all)
            except Exception:
                print(df_all.to_string(index=False))

        return fig, axes
