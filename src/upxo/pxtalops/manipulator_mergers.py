import numpy as np
import pandas as pd

"""
Core merger functions
=====================
List of dfunctions designed to merge grais based on doiffernet criteria. 

Merger assistant functions
==========================
List of defined functions to select sink grains based on various statistical metrics of neighboring 
grain mprop values for metrics such as minimum, mean, maximum, median, and quantile. The functions 
are designed to facilitate grain merging operations in microstructural simulations.

Functions
---------
- select_sinks_min_mprop: Select sink grains based on minimum neigh grain mprop.
- select_sinks_mean_mprop: Select sink grains based on mean neigh grain mprop.
- select_sinks_max_mprop: Select sink grains based on maximum neigh grain mprop.
- select_sinks_median_mprop: Select sink grains based on median neigh grain mprop.
- select_sinks_quantile_mprop: Select sink grains based on quantile neigh grain mprop.
"""

def select_sinks_min_mprop(NeighGidSubset={},
                           NeighGidsMprop={},
                           mprop_range_for_sink_selection=[-5, 5],
                           NG_sinks={},
                           force_select_criteria='distance_based'):
    '''
    Select sink grains based on minimum neigh grain mprop.

    Explanation
    -----------
    For each central grain (cgid), we look at the mprops of its neighboring grains.
    We define a proportional threshold range around the minimum mprop found among
    these neighbors, based on the specified uncertainty percentages (mprop_range_for_sink_selection).
    We then identify all neighboring grains whose mprops fall within this range as
    candidate sinks. If there are multiple candidate sinks, we randomly select one
    to be the sink for the candidate grain.

    Parameters
    ----------
    NeighGidSubset : dict
        Dictionary mapping candidate grain IDs to lists of neighboring grain IDs.
    NeighGidsMprop : dict
        Dictionary mapping candidate grain IDs to arrays of neighboring grain mprops.
    mprop_range_for_sink_selection : list or tuple
        List or tuple containing two elements representing the lower and upper uncertainty percentages for sink selection.
    NG_sinks : dict
        Dictionary to store the selected sink grain IDs for each candidate grain.
    force_select_criteria : str, optional
        Criteria to force sink selection, default is 'distance_based'. 
        Options include 'distance_based', 'max_mprop', etc.

    Returns
    -------
    dict
        Updated NG_sinks dictionary with selected sink grain IDs for each candidate grain.

    Example
    -------
    mprop_range_for_sink_selection = [-5, 5]
    prop_thresh_low = min_area
    prop_thresh_up  = min_area * (1 + mprop_range_for_sink_selection[1]/100)
    candidate_sinks = neigh_areas within [prop_thresh_low, prop_thresh_up]

    Only upper uncertainty is considered here since we are looking for minima.
    '''
    for cgid, neigh_areas in NeighGidsMprop.items():
        if len(neigh_areas) == 1:
            # Only one neighbor, assign it directly
            NG_sinks[cgid] = NeighGidSubset[cgid][0]
            continue
        elif len(neigh_areas) == 2:
            # Two neighbors, assign the one with smaller area directly
            if neigh_areas[0] < neigh_areas[1]:
                NG_sinks[cgid] = NeighGidSubset[cgid][0]
            else:
                NG_sinks[cgid] = NeighGidSubset[cgid][1]
            continue
        else:
            neigh_areas_min = np.min(neigh_areas)
            # Identify candidate sinks within uncertainty range
            prop_thresh_low = neigh_areas_min*(1-np.abs(mprop_range_for_sink_selection[0])/100)
            prop_thresh_up = neigh_areas_min*(1+np.abs(mprop_range_for_sink_selection[1])/100)
            candidate_sinks = np.where((neigh_areas >= prop_thresh_low) & (neigh_areas <= prop_thresh_up))[0]
            if len(candidate_sinks) == 0:
                if force_select_criteria == 'max_mprop':
                    candidate_sinks = [int(np.argmax(neigh_areas))]
                elif force_select_criteria == 'distance_based':
                    candidate_sinks = [int(np.argmin(np.abs(neigh_areas-neigh_areas_min)))]
            # In case of multiple minima, choose one at random
            sink = np.random.choice(candidate_sinks, size=1, replace=False)[0]
            NG_sinks[cgid] = NeighGidSubset[cgid][sink]
    return NG_sinks

def select_sinks_mean_mprop(NeighGidSubset={}, 
                            NeighGidsMprop={},
                            mprop_range_for_sink_selection=[-5, 5],
                            NG_sinks={},
                            force_select_criteria='distance_based'):
    '''
    Select sink grains based on mean neigh grain mprop.

    Explanation
    -----------
    For each central grain (cgid), we look at the mprop of its neighboring grains.
    We calculate the mean mprop of these neighbors and define a proportional threshold
    range around this mean based on the specified uncertainty percentages (mprop_range_for_sink_selection).
    We then identify all neighboring grains whose mprops fall within this range as
    candidate sinks. If there are multiple candidate sinks, we randomly select one
    to be the sink for the candidate grain.

    Example
    -------
    mprop_range_for_sink_selection = [-5, 5]
    prop_thresh_low = mean_area * (1 - mprop_range_for_sink_selection[0]/100)
    prop_thresh_up  = mean_area * (1 + mprop_range_for_sink_selection[1]/100)
    candidate_sinks = neigh_areas within [prop_thresh_low, prop_thresh_up]

    This approach considers both lower and upper uncertainties around the mean.
    '''
    for cgid, neigh_areas in NeighGidsMprop.items():
        if len(neigh_areas) == 1:
            # Only one neighbor, assign it directly
            NG_sinks[cgid] = NeighGidSubset[cgid][0]
            continue
        elif len(neigh_areas) == 2:
            # Two neighbors, assign the one closest to mean directly
            mean_area = np.mean(neigh_areas)
            if np.abs(neigh_areas[0]-mean_area) < np.abs(neigh_areas[1]-mean_area):
                NG_sinks[cgid] = NeighGidSubset[cgid][0]
            else:
                NG_sinks[cgid] = NeighGidSubset[cgid][1]
            continue
        else:
            neigh_areas_mean = np.mean(neigh_areas)
            # Identify candidate sinks within uncertainty range
            prop_thresh_low = neigh_areas_mean*(1-np.abs(mprop_range_for_sink_selection[0])/100)
            prop_thresh_up = neigh_areas_mean*(1+np.abs(mprop_range_for_sink_selection[1])/100)
            candidate_sinks = np.where((neigh_areas >= prop_thresh_low) & (neigh_areas <= prop_thresh_up))[0]
            if len(candidate_sinks) == 0:
                candidate_sinks = [int(np.argmin(np.abs(neigh_areas-neigh_areas_mean)))]
            # In case of multiple candidates, choose one at random
            sink = np.random.choice(candidate_sinks, size=1, replace=False)[0]
            NG_sinks[cgid] = NeighGidSubset[cgid][sink]
    return NG_sinks

def select_sinks_max_mprop(NeighGidSubset={},
                           NeighGidsMprop={},
                           mprop_range_for_sink_selection=[-5, 5],
                           NG_sinks={},
                           force_select_criteria='distance_based'):
    '''
    Select sink grains based on maximum neigh grain mprop.

    Explanation
    -----------
    For each central grain (cgid), we look at the mprops of its neighboring grains.
    We define a proportional threshold range around the maximum mprop found among
    these neighbors, based on the specified uncertainty percentages (mprop_range_for_sink_selection).
    We then identify all neighboring grains whose mprops fall within this range as
    candidate sinks. If there are multiple candidate sinks, we randomly select one
    to be the sink for the candidate grain.

    Example
    -------
    mprop_range_for_sink_selection = [-5, 5]
    prop_thresh_low = max_area * (1 - mprop_range_for_sink_selection[0]/100)
    prop_thresh_up  = max_area
    candidate_sinks = neigh_areas within [prop_thresh_low, prop_thresh_up]

    Only lower uncertainty is considered here since we are looking for maxima.
    '''
    for cgid, neigh_areas in NeighGidsMprop.items():
        if len(neigh_areas) == 1:
            # Only one neighbor, assign it directly
            NG_sinks[cgid] = NeighGidSubset[cgid][0]
            continue
        elif len(neigh_areas) == 2:
            # Two neighbors, assign the one with larger area directly
            if neigh_areas[0] > neigh_areas[1]:
                NG_sinks[cgid] = NeighGidSubset[cgid][0]
            else:
                NG_sinks[cgid] = NeighGidSubset[cgid][1]
            continue
        else:
            neigh_areas_max = np.max(neigh_areas)
            # Identify candidate sinks within uncertainty range
            prop_thresh_low = neigh_areas_max*(1-np.abs(mprop_range_for_sink_selection[0])/100)
            prop_thresh_up = neigh_areas_max*(1+np.abs(mprop_range_for_sink_selection[1])/100)
            candidate_sinks = np.where((neigh_areas <= prop_thresh_up) & (neigh_areas >= prop_thresh_low))[0]
            if len(candidate_sinks) == 0:
                candidate_sinks = [int(np.argmin(np.abs(neigh_areas-neigh_areas_max)))]
            # In case of multiple maxima, choose one at random
            sink = np.random.choice(candidate_sinks, size=1, replace=False)[0]
            NG_sinks[cgid] = NeighGidSubset[cgid][sink]
    return NG_sinks

def select_sinks_median_mprop(NeighGidSubset={},
                              NeighGidsMprop={},
                              mprop_range_for_sink_selection=[-5, 5],
                              NG_sinks={},
                              force_select_criteria='distance_based'):
    '''
    Select sink grains based on median neigh grain mprop.

    Explanation
    -----------
    For each central grain (cgid), we look at the mprops of its neighboring grains.
    We calculate the median mprop of these neighbors and define a proportional threshold
    range around this median based on the specified uncertainty percentages (mprop_range_for_sink_selection).
    We then identify all neighboring grains whose mprops fall within this range as
    candidate sinks. If there are multiple candidate sinks, we randomly select one
    to be the sink for the candidate grain.

    Example
    -------
    mprop_range_for_sink_selection = [-5, 5]
    prop_thresh_low = median_area * (1 - mprop_range_for_sink_selection[0]/100)
    prop_thresh_up  = median_area * (1 + mprop_range_for_sink_selection[1]/100)
    candidate_sinks = neigh_areas within [prop_thresh_low, prop_thresh_up]

    This approach considers both lower and upper uncertainties around the median.
    '''
    for cgid, neigh_areas in NeighGidsMprop.items():
        if len(neigh_areas) == 1:
            # Only one neighbor, assign it directly
            NG_sinks[cgid] = NeighGidSubset[cgid][0]
            continue
        elif len(neigh_areas) == 2:
            # Two neighbors, assign the one closest to median directly
            median_area = np.median(neigh_areas)
            if np.abs(neigh_areas[0]-median_area) < np.abs(neigh_areas[1]-median_area):
                NG_sinks[cgid] = NeighGidSubset[cgid][0]
            else:
                NG_sinks[cgid] = NeighGidSubset[cgid][1]
            continue
        else:
            neigh_areas_median = np.median(neigh_areas)
            # Identify candidate sinks within uncertainty range
            prop_thresh_low = neigh_areas_median*(1-np.abs(mprop_range_for_sink_selection[0])/100)
            prop_thresh_up = neigh_areas_median*(1+np.abs(mprop_range_for_sink_selection[1])/100)
            candidate_sinks = np.where((neigh_areas <= prop_thresh_up) & (neigh_areas >= prop_thresh_low))[0]
            if len(candidate_sinks) == 0:
                candidate_sinks = [int(np.argmin(np.abs(neigh_areas-neigh_areas_median)))]
            # In case of multiple candidates, choose one at random
            sink = np.random.choice(candidate_sinks, size=1, replace=False)[0]
            NG_sinks[cgid] = NeighGidSubset[cgid][sink]
    return NG_sinks

def select_sinks_quantile_mprop(NeighGidSubset={},
                                NeighGidsMprop={},
                                mprop_range_for_sink_selection=[-20, 20],
                                NG_sinks={},
                                sink_metric='q25',
                                force_select_criteria='distance_based'):
    '''
    Select sink grains based on quantile neigh grain mprop.

    Explanation
    -----------
    For each central grain (cgid), we look at the mprops of its neighboring grains.
    We calculate the specified quantile mprop of these neighbors and define a proportional
    threshold range around this quantile based on the specified uncertainty percentages (ssu).
    We then identify all neighboring grains whose mprops fall within this range as
    candidate sinks. If there are multiple candidate sinks, we randomly select one
    to be the sink for the candidate grain.

    Example
    -------
    mprop_range_for_sink_selection = [-5, 5]
    prop_thresh_low = quantile_area * (1 - mprop_range_for_sink_selection[0]/100)
    prop_thresh_up  = quantile_area * (1 + mprop_range_for_sink_selection[1]/100)
    candidate_sinks = neigh_areas within [prop_thresh_low, prop_thresh_up]

    This approach considers both lower and upper uncertainties around the specified quantile.
    '''
    # Mandatory quantile number extraction and validation
    if sink_metric[0] == 'q' and sink_metric[1:].isnumeric():
        qnum = int(sink_metric[1:])
        qnum = qnum*25 if qnum in (0, 1, 2, 3, 4) else qnum
        if not (0 <= qnum <= 100):
            raise ValueError(f"Invalid quantile number in sink_metric: {sink_metric}. Must be between 0 and 100.")

    for cgid, neigh_areas in NeighGidsMprop.items():
        if len(neigh_areas) == 1:
            # Only one neighbor, assign it directly
            NG_sinks[cgid] = NeighGidSubset[cgid][0]
            continue
        elif len(neigh_areas) == 2:
            # Two neighbors, assign the one closest to quantile directly
            neigh_areas_quantile = np.percentile(neigh_areas, qnum)
            if np.abs(neigh_areas[0]-neigh_areas_quantile) < np.abs(neigh_areas[1]-neigh_areas_quantile):
                NG_sinks[cgid] = NeighGidSubset[cgid][0]
            else:
                NG_sinks[cgid] = NeighGidSubset[cgid][1]
            continue
        else:
            neigh_areas_quantile = np.percentile(neigh_areas, qnum)
            # Identify candidate sinks within uncertainty range
            prop_thresh_low = neigh_areas_quantile*(1-np.abs(mprop_range_for_sink_selection[0])/100)
            prop_thresh_up = neigh_areas_quantile*(1+np.abs(mprop_range_for_sink_selection[1])/100)
            candidate_sinks = np.where((neigh_areas <= prop_thresh_up) & (neigh_areas >= prop_thresh_low))[0]
            if len(candidate_sinks) == 0:
                candidate_sinks = [int(np.argmin(np.abs(neigh_areas-neigh_areas_quantile)))]
            # In case of multiple candidates, choose one at random
            sink = np.random.choice(candidate_sinks, size=1, replace=False)[0]
            NG_sinks[cgid] = NeighGidSubset[cgid][sink]
    return NG_sinks1