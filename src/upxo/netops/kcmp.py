# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:35:01 2024

@author: rg5749
"""
import numpy as np
import networkx as nx
import netlsd
from networkx.algorithms import community
from upxo._sup.data_ops import calculate_angular_distance
from upxo._sup.data_ops import calculate_density_bins
from upxo._sup.data_ops import approximate_to_bin_means
from scipy.stats import wasserstein_distance, ks_2samp, energy_distance


def calculate_rkfield_js(G1, G2):
    """
    Calculate Jaccard similarity between the node sets of two graphs.

    Import
    ------
    from upxo.netops.kcmp import calculate_rkfield_js
    """
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    return len(nodes1 & nodes2) / len(nodes1 | nodes2)

def calculate_rkfield_wd(kd_tgt, kd_smp):
    """
    Calculate Wasserstein distance based R-Field.

    Parameters
    ----------
    kd_tgt: Netowrk Degrees of the target grain structure
    kd_smp: Netowrk Degrees of the sample grain structure

    Return
    ------
    r

    Import
    ------
    from upxo.netops.kcmp import calculate_rkfield_wd
    """
    distance = wasserstein_distance(kd_tgt, kd_smp)
    """

    """
    r = 1 / (1 + distance)
    return r

def calculate_rkfield_ksp(kd_tgt, kd_smp):
    """
    kd: Netowrk Degree

    Import
    ------
    from upxo.netops.kcmp import calculate_rkfield_ksp
    """
    # Higher p-value indicates more similar distributions
    _, p_value = ks_2samp(kd_tgt, kd_smp)
    r = p_value
    return r

def calculate_rkfield_ed(kd_tgt, kd_smp):
    """
    kd: Netowrk Degree

    Import
    ------
    from upxo.netops.kcmp import calculate_rkfield_ed
    """
    # Energy Distance (a non-parametric measure of statistical distance)
    distance = energy_distance(kd_tgt, kd_smp)
    r = 1 / (1 + distance)
    return r

def calculate_rkfield_nlsd(kd_tgt, kd_smp,
                           timescales=np.logspace(-2, 2, 20)):
    """
    Calculates NetLSD similarity between two networks.

    Args:
        G1 (nx.Graph): The first network.
        G2 (nx.Graph): The second network.
        timescales (int, optional): Number of timescales to use (default: 10).

    Returns:
        float: The NetLSD distance (lower is more similar).

    Import
    ------
    from upxo.netops.kcmp import calculate_rkfield_nlsd
    """
    descriptor1 = netlsd.heat(kd_tgt, timescales=timescales)
    descriptor2 = netlsd.heat(kd_smp, timescales=timescales)
    distance = np.linalg.norm(descriptor1 - descriptor2)
    # Convert distance to similarity (optional)
    r = 1 / (1 + distance)
    return r
