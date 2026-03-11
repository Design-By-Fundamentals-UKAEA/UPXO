# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:32:32 2024

@author: rg5749
"""
import networkx as nx

def make_gid_net_from_neighlist(neighbor_dict):
    """
    Creates a networkx graph from the entire neighbor_dict.

    Import
    ------
    from upxo.netops.kmake import make_gid_net_from_neighlist
    """
    G = nx.Graph()
    G.add_edges_from([(gid, neighbor) for gid, neighbors in neighbor_dict.items() for neighbor in neighbors])
    return G
