from upxo._sup import dataTypeHandlers as dth
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def see_distr(self, gsdim=2, vis='hist', 
            prop_data_format='dataframe', prop_df=None, 
            prop_names=['area', 'perimeter', 'orientation', 'solidity'],
            props={'area': [], 'perimeter': [], 'orientation': [], 'solidity': []},
            prop_units={'area': 'μm²', 'perimeter': 'μm', 'orientation': 'degrees', 'solidity': ''},
            probability_density=False, 
            nbins_values={'area': 30, 'perimeter': 30, 'orientation': 30, 'solidity': 30},
            bw_adjust_values={'area': None, 'perimeter': None, 'orientation': None, 'solidity': None},
            alpha_values={'area': 0.7, 'perimeter': 0.7, 'orientation': 0.7, 'solidity': 0.7},
            color_values={'area': 'blue', 'perimeter': 'blue', 'orientation': 'blue', 'solidity': 'blue'},
            edgecolor_values={'area': 'black', 'perimeter': 'black', 'orientation': 'black', 'solidity': 'black'},
            binsize=30, alpha=0.7, color='blue', edgecolor='black',
            ncolumns=3, ylabel='count'):
    '''
    Plot distributions of multiple grain properties in subplots

    Parameters
    ----------
    gsdim : int, optional
        Dimensionality of the grain structure data (2 for 2D, 3 for 3D). Default is 2.
    vis : str, optional
        Visualization type: 'hist' for histogram, 'kde' for kernel density estimate,
        or 'hist_kde' for both overlaid. Default is 'hist'.
    prop_data_format : str, optional
        Format of the property data source. Options are 'dataframe' or 'dict'. Default
        is 'dataframe'.
    prop_df : pandas.DataFrame, optional
        DataFrame containing grain properties if prop_data_format is 'dataframe'. Default is None.
    prop_names : list of str, optional
        List of grain property names to plot distributions for. Default includes
        'area', 'perimeter', 'orientation', and 'solidity'.
    props : dict, optional
        Dictionary of grain properties if prop_data_format is 'dict'. Default is empty dict.
    prop_units : dict, optional
        Dictionary mapping property names to their units for labeling axes. Default units
        are provided for common properties.
    probability_density : bool, optional
        If True, normalize distributions to form a probability density. Default is False.
    nbins_values : dict, optional
        Dictionary specifying number of bins for each property. Default is 30 bins for each.
    bw_adjust_values : dict, optional
        Dictionary specifying bandwidth adjustment for KDE plots. If None for a property,
        optimal bandwidth is calculated automatically using Scott's rule. Default is None for all.
    alpha_values : dict, optional
        Dictionary specifying transparency (alpha) for each property distribution. Default is 0.7.
    color_values : dict, optional
        Dictionary specifying fill color for each property distribution. Default is 'blue'.
    edgecolor_values : dict, optional
        Dictionary specifying edge color for each property distribution. Default is 'black'.
    binsize : int, optional
        Default number of bins to use if not specified in nbins_values. Default is 30
    alpha : float, optional
        Default transparency (alpha) to use if not specified in alpha_values. Default is 0
    color : str, optional
        Default fill color to use if not specified in color_values. Default is 'blue'.
    edgecolor : str, optional
        Default edge color to use if not specified in edgecolor_values. Default is 'black'.
    ncolumns : int, optional
        Number of columns in the subplot grid. Default is 3.
    ylabel : str, optional
        Label for the y-axis. Default is 'count'.

    Returns
    -------
    None

    Notes
    -----
    This function creates distributions for specified grain properties in a grid of subplots.
    It supports data input as either a pandas DataFrame or a dictionary of properties.

    Import
    ------
    from upxo.viz.dataviz import see_distr
    '''
    if gsdim != 2:
        return
    
    # Select data source based on format
    if prop_data_format in ('dataframe', 'pd.DataFrame', 'df', 'pd', 'pdf'):
        if prop_df is None or not isinstance(prop_df, pd.DataFrame):
            return
        data_source = prop_df
    elif prop_data_format == 'dict':
        if not isinstance(props, dict):
            return
        data_source = props
    else:
        return
    
    if type(prop_names) != str and not isinstance(prop_names, (list, tuple, set)):
        raise ValueError("prop_names must be a string or an iterable of strings")

    if type(prop_names) == str:
        prop_names = [prop_names]
    
    # Filter valid properties
    properties = [prop for prop in prop_names 
                  if prop in dth.valid_region_properties.scikitimage_region_properties2d]
    if not properties:
        return
    
    nbins = {prop: nbins_values.get(prop, binsize) for prop in properties}
    alpha_prop = {prop: alpha_values.get(prop, alpha) for prop in properties}
    color_prop = {prop: color_values.get(prop, color) for prop in properties}
    edgecolor_prop = {prop: edgecolor_values.get(prop, edgecolor) for prop in properties}
    
    # Bandwidth adjustment: calculate optimal if None
    bw_adjust = {}
    for prop in properties:
        bw_val = bw_adjust_values.get(prop)
        if bw_val is None:
            # Use Scott's rule via seaborn default (bw_adjust=1)
            bw_adjust[prop] = 1.0
        else:
            bw_adjust[prop] = bw_val
    
    # Calculate subplot layout
    num_of_subplots = len(properties)
    nrows = (num_of_subplots + ncolumns - 1) // ncolumns  # Ceiling division
    
    # Create plots
    fig = plt.figure(1)
    for position, prop in enumerate(properties, start=1):
        ax = fig.add_subplot(nrows, ncolumns, position)
        
        data = data_source[prop]
        
        if vis == 'hist':
            sns.histplot(data,
                        stat='density' if probability_density else 'count',
                        bins=nbins[prop],
                        alpha=alpha_prop[prop],
                        color=color_prop[prop],
                        edgecolor=edgecolor_prop[prop],
                        ax=ax)
        elif vis == 'kde':
            sns.kdeplot(data,
                       bw_adjust=bw_adjust[prop],
                       color=color_prop[prop],
                       fill=True,
                       alpha=alpha_prop[prop],
                       linewidth=2,
                       ax=ax)
        elif vis == 'hist_kde':
            sns.histplot(data,
                        stat='density',
                        bins=nbins[prop],
                        alpha=alpha_prop[prop] * 0.6,
                        color=color_prop[prop],
                        edgecolor=edgecolor_prop[prop],
                        ax=ax)
            sns.kdeplot(data,
                       bw_adjust=bw_adjust[prop],
                       color='red',
                       linewidth=2,
                       ax=ax)
        
        ax.set_xlabel(f'{prop} ({prop_units.get(prop, "")})')
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()
