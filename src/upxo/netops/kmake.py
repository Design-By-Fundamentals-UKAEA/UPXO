"""
Grain-network construction helpers for UPXO.

Builds NetworkX graphs from neighbour dictionaries (gid → neighbour gids)
used by KREPR, gsan2d, and mode-modifier seed selection.

Import::

    from upxo.netops.kmake import make_gid_net_from_neighlist
"""
import networkx as nx


def make_gid_net_from_neighlist(neighbor_dict):
    """
    Create an undirected NetworkX graph from a neighbour map.

    Parameters
    ----------
    neighbor_dict : dict
        Mapping ``gid → iterable of neighbour gids``. Self-loops and
        direction are not special-cased; each (gid, neighbour) pair becomes
        an undirected edge.

    Returns
    -------
    networkx.Graph
        Graph with grain IDs as nodes and neighbour relations as edges.

    Example
    -------
    >>> from upxo.netops.kmake import make_gid_net_from_neighlist
    >>> G = make_gid_net_from_neighlist({1: [2, 3], 2: [1], 3: [1]})
    """
    G = nx.Graph()
    G.add_edges_from(
        [(gid, neighbor)
         for gid, neighbors in neighbor_dict.items()
         for neighbor in neighbors]
    )
    return G
