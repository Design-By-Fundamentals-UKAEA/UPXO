"""
Contains a list of definitions to enable charactyerizaion of networkx data.

Created on Fri Jun  7 11:38:09 2024

@author: Dr. Sunil Anandatheertha
"""
import numpy as np
from upxo._sup.dataTypeHandlers import IS_ITER


def calculate_kdegrees(networks):
    """
    Get degree distribution of every network graph in networks.

    Parameters
    ----------
    networks: list
        List of networkx network graphs.

    Return
    ------
    kd: list
        List of list of degree of every node in each network graph.

    Data structures
    ---------------
    networks = [k1, k2, ..., kp, ... kP]
        Where, kp: pth network graph

    ks = [k1d, k2d, ..., kid, ..., knd]
        Where, kid: degree distribution of ith network.

    kid = [dn1, dn2, ..., dnj,... dnJ]
        Where, dnj: degree of node j.
    """
    kd = []
    for k in networks:
        kd.append(np.array([k.degree(node) for node in k.nodes()]))
    return kd


def calculate_kdegrees_equalbinning(networks):
    """
    Calculate the pairwise degree distributions, and equally bin them.

    Parameters
    ----------
    tgt_graph: networkx graph
        Target graon structure O(n) neighbour network graph.
    smp_graph: networkx graph
        Sample graon structure O(n) neighbour network graph.

    Return
    ------
    kd_tgt: List of degree of every node in target network.
    kd_smp: List of degree of every node in sample network.

    Data structures
    ---------------
    networks = [k1, k2, ..., kp, ... kP]
        Where, kp: pth network graph

    ks = [k1d, k2d, ..., kid, ..., knd]
        Where, kid: degree distribution of ith network.

    kid = [dn1, dn2, ..., dnj,... dnJ]
        Where, dnj: degree of node j.

    Explanations
    ------------
    Some tests are sensitive to differences in the cumulative
    distribution functions (CDFs) of the data. Directly using the raw
    degree sequences can be misleading if the sample sizes
    (number of nodes) differ significantly between the two networks.
    Histograms normalize the data, providing a better representation of
    the underlying probability distributions.
    """
    if not IS_ITER(networks):
        networks = [networks]
    # Calculate network degree distributions
    kd = calculate_kdegrees(networks)
    # Calculate histograms with the same bins for fair comparison
    min_degree = min([min(_kd_) for _kd_ in kd])
    max_degree = max([max(_kd_) for _kd_ in kd])
    # min_degree = min(np.min(kd_tgt), np.min(kd_smp))
    # max_degree = max(np.max(kd_tgt), np.max(kd_smp))
    bins = np.arange(min_degree, max_degree + 2)
    # In the above line, +2 for inclusive range
    kdeg = []
    for _kd_ in kd:
        _kdeg_, _ = np.histogram(_kd_, bins=bins, density=True)
        kdeg.append(_kdeg_)

    return kdeg

    def calculate_degree_centrality(networks):
        degree_centrality = [nx.degree_centrality(k) for k in networks]
        return degree_centrality

    def calculate_betweenness_centrality(networks):
        btw_centrality = [nx.betweenness_centrality(k) for k in networks]
        return btw_centrality

    def calculate_closeness_centrality(networks):
        closeness_centrality = [nx.closeness_centrality(k) for k in networks]
        return closeness_centrality

    def calculate_eigenvector_centrality(networks):
        eigenvector_centrality = [nx.eigenvector_centrality(k) for k in networks]
        return eigenvector_centrality

    def calculate_centrality_measures(self):
       # Node Properties
       pass
