"""
nbWidgets.py — Interactive ipywidgets controls for UPXO Jupyter notebooks.

All functions in this module are designed to be called inside a Jupyter cell.
They render a widget panel immediately on call and return a state dict that
can be read in the next cell after the user has interacted with the controls.

Functions
---------
mdf_peak_selector          Peak checklist for MDF downstream filtering.
make_property_stats_widgets  Control panel for grain-role property plots.
read_property_stats_widgets  Read current widget values into a plain dict.
"""

from __future__ import annotations


def mdf_peak_selector(peaks: dict) -> dict:
    """
    Display an interactive ipywidgets checklist so the user can pick which
    MDF peaks to retain for downstream analysis.

    All peaks are pre-ticked.  Click **Confirm selection** to update
    ``selected_peaks``.

    Parameters
    ----------
    peaks : dict
        Output of ``crystal_orientation.detect_mdf_peaks()``.

    Returns
    -------
    selected_peaks : dict
        Pre-populated with all peaks; updated in-place on confirmation.
        Keys: ``'angles'`` (list of float), ``'indices'`` (list of int).
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    peak_indices = peaks['peak_indices']
    peak_labels  = peaks['peak_labels']
    peak_angles  = peaks['peak_angles']

    checkboxes = [
        widgets.Checkbox(value=True, description=lbl,
                         layout=widgets.Layout(width='480px'))
        for lbl in peak_labels
    ]
    confirm_btn = widgets.Button(description='Confirm selection',
                                 button_style='success', icon='check')
    output_box  = widgets.Output()

    selected_peaks: dict = {
        'angles':  list(peak_angles),
        'indices': list(peak_indices),
    }

    def _on_confirm(_):
        selected_peaks['angles']  = [peak_angles[i]  for i, cb in enumerate(checkboxes) if cb.value]
        selected_peaks['indices'] = [peak_indices[i] for i, cb in enumerate(checkboxes) if cb.value]
        with output_box:
            clear_output()
            print('selected_peaks updated')
            print(f"   angles : {selected_peaks['angles']}")

    confirm_btn.on_click(_on_confirm)

    print('Select which MDF peaks to retain for downstream analysis:')
    display(widgets.VBox(checkboxes + [confirm_btn, output_box]))
    return selected_peaks


def selectProps_twinGS(
        props: list[str] | None = None,
        groups: list[str] | None = None,
        default_ncols: int = 2,
        default_fontsize: float = 12.0,
) -> dict:
    """
    Build and display the ipywidgets control panel for
    ``plot_grain_role_property_stats``.

    Returns a dict with keys ``prop_checkboxes``, ``group_checkboxes``,
    ``ncols_slider``, and ``fontsize_slider``.  Pass the returned dict
    directly to :func:`read_property_stats_widgets` to extract current
    values before calling the plot function.

    Parameters
    ----------
    props : list of str, optional
        Property names to show.  Subset of
        ``['area', 'aspect_ratio', 'perimeter', 'solidity', 'n_neighbours']``.
        Only ``'area'`` is ticked by default.
    groups : list of str, optional
        Group names to show (all ticked by default).  Subset of
        ``['pure_parents', 'pure_twins', 'intermediates', 'non_role']``.
    default_ncols : int
        Initial value of the columns slider (0 = single row).
    default_fontsize : float
        Initial font size value.

    Returns
    -------
    dict
        ``{'prop_checkboxes': dict, 'group_checkboxes': dict,
           'ncols_slider': IntSlider, 'fontsize_slider': FloatSlider}``
    """
    import ipywidgets as widgets
    from IPython.display import display

    ALL_PROPS  = ['area', 'aspect_ratio', 'perimeter', 'solidity', 'n_neighbours']
    ALL_GROUPS = ['pure_parents', 'pure_twins', 'intermediates', 'non_role']
    PROP_LABELS_UI = {
        'area':         'Area (µm²)',
        'aspect_ratio': 'Aspect ratio',
        'perimeter':    'Perimeter (µm)',
        'solidity':     'Solidity',
        'n_neighbours': 'N neighbours',
    }
    GROUP_LABELS_UI = {
        'pure_parents':  'Pure parents',
        'pure_twins':    'Pure twins',
        'intermediates': 'Intermediates',
        'non_role':      'Non-role grains',
    }

    if props is None:
        props = ALL_PROPS
    if groups is None:
        groups = ALL_GROUPS

    prop_checkboxes = {
        p: widgets.Checkbox(value=(p == 'area'), description=PROP_LABELS_UI[p],
                            layout=widgets.Layout(width='200px'))
        for p in props
    }
    group_checkboxes = {
        g: widgets.Checkbox(value=True, description=GROUP_LABELS_UI[g],
                            layout=widgets.Layout(width='200px'))
        for g in groups
    }
    ncols_slider = widgets.IntSlider(
        value=default_ncols, min=0, max=5, step=1,
        description='Columns:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px'),
        readout=True,
    )
    fontsize_slider = widgets.FloatSlider(
        value=default_fontsize, min=6.0, max=20.0, step=0.5,
        description='Font size:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='350px'),
        readout=True,
        readout_format='.1f',
    )

    display(widgets.VBox([
        widgets.HTML('<b style="font-size:13px">Morphological / topological properties:</b>'),
        widgets.HBox(list(prop_checkboxes.values())),
        widgets.HTML('<b style="font-size:13px">Grain groups:</b>'),
        widgets.HBox(list(group_checkboxes.values())),
        widgets.HTML('<b style="font-size:13px">Subplot columns (0 = single row):</b>'),
        ncols_slider,
        widgets.HTML('<b style="font-size:13px">Font size:</b>'),
        fontsize_slider,
    ]))

    return {
        'prop_checkboxes':  prop_checkboxes,
        'group_checkboxes': group_checkboxes,
        'ncols_slider':     ncols_slider,
        'fontsize_slider':  fontsize_slider,
    }


def readProps_twinGS(widgets_dict: dict) -> dict:
    """
    Read current values from the dict returned by :func:`selectProps_twinGS`.

    Returns
    -------
    dict
        ``{'selected_props': list, 'selected_groups': list,
           'ncols': int | None, 'fontsize': float}``
    """
    selected_props  = [k for k, cb in widgets_dict['prop_checkboxes'].items()  if cb.value]
    selected_groups = [k for k, cb in widgets_dict['group_checkboxes'].items() if cb.value]
    ncols_val       = widgets_dict['ncols_slider'].value
    print(f'Properties : {selected_props}')
    print(f'Groups     : {selected_groups}')
    return {
        'selected_props':  selected_props,
        'selected_groups': selected_groups,
        'ncols':           ncols_val if ncols_val > 0 else None,
        'fontsize':        widgets_dict['fontsize_slider'].value,
    }
