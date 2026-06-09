import numpy as np
import pandas as pd
from copy import deepcopy
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from upxo._sup import dataTypeHandlers as dth
from skimage.measure import label as skim_label
from upxo.pxtalops import detect_grains_from_mcstates
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Optional, Tuple, List

@dataclass
class metaa:
    # Imaging and sample metadata to standardize downstream analysis
    pixel_size: float = 1.0 # length per pixel (e.g., microns)
    unit: str = "px" # 'px' or physical unit
    frame_dt: Optional[float] = None # time per frame, if temporal
    material: str = ""
    sample_id: str = ""
    source: str = "" # file or instrument
    notes: str = ""
    seed: int = 42 # We will use this for reproducibility

@dataclass
class principle_component_analysis:
    pass

class kmodel():
    __slots__ = ('G', 'gprop', 'mprop', 'pathlengths')

    def __init__(self, G):
        """Initialise the instance."""
        self.G = G
        self.gprop = {}

    def characterize_graph(self, printout=True, k_char_level=None):
        """
        Compute and optionally print summary graph characteristics, with optional
        distance-based metrics for smaller, connected graphs.
        This method populates the `self.gprop` dictionary with core structural
        properties of the graph (nodes, edges, density, clustering, assortativity).
        If `k_char_level` is set to "full" or "advanced", and the graph is both
        small (< 5000 nodes) and connected, it additionally computes eccentricity-
        derived metrics (radius, diameter, center, periphery).
        Parameters
        ----------
        printout : bool, default True
            If True, prints the computed metrics to stdout.
        k_char_level : {None, 'full', 'advanced'}, optional
            Controls whether to compute additional distance-based metrics.
            - None (default): Compute only core properties.
            - 'full' or 'advanced': Also compute eccentricity, radius, diameter,
              center, and periphery, subject to graph size and connectivity.
        Returns
        None
            Results are stored in `self.gprop` and optionally printed.
        Notes: 
        -----------
        self.gprop : dict
            Populated/updated with the following keys:
            - 'num_nodes' : int
            - 'num_edges' : int
            - 'density' : float
            - 'avg_clustering_coeff' : float
            - 'degree_assortativity' : float
            Additionally, when `k_char_level` in {'full', 'advanced'} and conditions
            are met (|V| < 5000 and graph is connected):
            - 'eccentricity' : dict[node, int]
            - 'radius' : int
            - 'diameter' : int
            - 'center' : list[node]
            - 'periphery' : list[node]
        Notes
        -----
        - Advanced metrics are skipped for graphs with 5000 or more nodes to avoid
          excessive runtime.
        - Eccentricity-based metrics require the graph to be connected. If the graph
          is not connected, those metrics are skipped and a message is printed.
        - The method assumes `self.G` is a NetworkX graph object.
        """
        self.mprop = None
        # Core structural properties (always computed)
        self.gprop['num_nodes'] = self.G.number_of_nodes()
        self.gprop['num_edges'] = self.G.number_of_edges()
        self.gprop['density'] = nx.density(self.G)
        self.gprop['avg_clustering_coeff'] = nx.average_clustering(self.G)
        self.gprop['degree_assortativity'] = nx.degree_assortativity_coefficient(self.G)
        if printout:
            print(f"Number of nodes: {self.gprop['num_nodes']}")
            print(f"Number of edges: {self.gprop['num_edges']}")
            print(f"Density: {self.gprop['density']}")
            print(f"Average clustering coefficient: {self.gprop['avg_clustering_coeff']}")
            print(f"Degree assortativity coefficient: {self.gprop['degree_assortativity']}")
        if k_char_level in ('full', 'advanced'):
            # Check if graph is large; if so, skip this entirely.
            if self.G.number_of_nodes() < 5000:
                try:
                    # 1. Calculate Eccentricity ONCE (The heavy lifting)
                    # Note: This fails if the graph is not connected.
                    ecc = {int(k): v for k, v in nx.eccentricity(self.G).items() }
                    # 2. Derive everything else from the dictionary instantly
                    self.gprop['eccentricity'] = ecc
                    self.gprop['radius'] = min(ecc.values())
                    self.gprop['diameter'] = max(ecc.values())
                    self.gprop['center'] = [n for n, e in ecc.items() if e == self.gprop['radius']]
                    self.gprop['periphery'] = [n for n, e in ecc.items() if e == self.gprop['diameter']]
                    if printout:
                        print(f"Eccentricity: {self.gprop['eccentricity']}")
                        print(f"Radius: {self.gprop['radius']}")
                        print(f"Diameter: {self.gprop['diameter']}")
                        print(f"Center: {self.gprop['center']}")
                        print(f"Periphery: {self.gprop['periphery']}")
                except nx.NetworkXError:
                    print("Skipping distance metrics: Graph is not connected.")

    def load_mprop(self, mprop_df):
        """Load a morphological property DataFrame into the model.

        Parameters
        ----------
        mprop_df : pandas.DataFrame
            DataFrame whose rows correspond to grains and columns to morphological
            properties (e.g. area, aspect_ratio, eccentricity).  Stored as
            ``self.mprop`` for downstream correlation and PCA methods.
        """
        self.mprop = mprop_df

    def summary(self):
        """Print a one-line description of the graph topology and return key flags.

        Returns
        -------
        dict
            ``{'directed': bool, 'multigraph': bool}`` indicating whether
            ``self.G`` is directed and/or a multigraph.
        """
        D = self.G.is_directed()
        M = self.G.is_multigraph()
        Dflag = 'directed' if D else 'undirected'
        Mflag = 'multigraph' if M else 'simple'
        print(f"This grain structure is a, {Dflag} {Mflag} graph \n with"
               f" {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        return {'directed': D, 'multigraph': M,}
         
    def shortest_path_length(self, see_distribution=True, figsize=(3, 2), kde=True):
        """Compute all-pairs shortest path lengths and optionally plot their distribution.

        Iterates over every source node and records the shortest-path distance
        to every reachable node.  Results are stored in ``self.pathlengths``.

        Parameters
        ----------
        see_distribution : bool, default True
            If True, displays a histogram (with optional KDE) of path lengths.
        figsize : tuple of float, default (3, 2)
            Width and height of the matplotlib figure in inches.
        kde : bool, default True
            Overlay a kernel density estimate on the histogram when
            ``see_distribution`` is True.

        Returns
        -------
        list of int
            Flat list of all pairwise shortest path lengths collected across
            every source node.
        """
        pathlengths = []
        for v in self.G.nodes():
            spl = nx.shortest_path_length(self.G, source=v)
            for p in spl:
                pathlengths.append(spl[p])
        self.pathlengths = pathlengths
        print(f"average shortest path length {sum(pathlengths) / len(pathlengths)}")
        if see_distribution:
            plt.figure(figsize=figsize)
            histdata = sns.histplot(pathlengths, bins=30, kde=kde)
            plt.xlabel('Path Length')
            plt.ylabel('Frequency')
            plt.title('Shortest Path Length Distribution')
            plt.show()
        return pathlengths
    
    def shortest_path_between_two_nodes(self, source_node, target_node, weight='weight'):
        """Return the shortest path between two nodes (grain IDs) in the graph.

        Uses Dijkstra's algorithm via NetworkX, traversing grain-boundary edges
        weighted by ``weight``.

        Parameters
        ----------
        source_node : int
            Starting grain ID (node) in ``self.G``.
        target_node : int
            Destination grain ID (node) in ``self.G``.
        weight : str, default ``'weight'``
            Edge attribute to use as the path cost.  Pass ``None`` to treat all
            edges as having unit weight.

        Returns
        -------
        list of int
            Ordered list of node IDs forming the shortest path from
            ``source_node`` to ``target_node``, inclusive.

        Example
        -------
        >>> path = kmod.shortest_path_between_two_nodes(source_node=1, target_node=10)
        """
        gids_shortest_path = nx.shortest_path(self.G, source=source_node, target=target_node, 
                                              weight=weight)
        return gids_shortest_path

    def average_shortest_path_length(self, recalulate=False):
        """Return the mean of all pairwise shortest path lengths.

        Re-uses ``self.pathlengths`` if already computed, unless ``recalulate``
        is True.

        Parameters
        ----------
        recalulate : bool, default False
            Force recomputation of path lengths even if ``self.pathlengths``
            already exists.

        Returns
        -------
        float
            Mean shortest path length across all node pairs.
        """
        if hasattr(self, 'pathlengths') and not recalulate:
            pathlengths = self.pathlengths
        else:
            pathlengths = self.shortest_path_length()
        average_path_length = sum(pathlengths) / len(pathlengths)
        print(f"average shortest path length {average_path_length}")
        return average_path_length

    def see_pathlength_distribution(self, recalulate=False, figsize=(3, 2), kde=True, throw_hist=False):
        """Plot the distribution of all-pairs shortest path lengths.

        Parameters
        ----------
        recalulate : bool, default False
            Force recomputation of path lengths even if ``self.pathlengths``
            already exists.
        figsize : tuple of float, default (3, 2)
            Width and height of the matplotlib figure in inches.
        kde : bool, default True
            Overlay a kernel density estimate on the histogram.
        throw_hist : bool, default False
            If True, return the seaborn ``AxesSubplot`` object for further
            customisation; otherwise return ``None``.

        Returns
        -------
        seaborn.axisgrid.FacetGrid or None
            The histogram axes object when ``throw_hist`` is True, else None.
        """
        if hasattr(self, 'pathlengths') and not recalulate:
            pathlengths = self.pathlengths
        else:
            pathlengths = self.shortest_path_length()
        plt.figure(figsize=figsize)
        histdata = sns.histplot(pathlengths, bins=30, kde=kde)
        plt.xlabel('Path Length')
        plt.ylabel('Frequency')
        plt.title('Shortest Path Length Distribution')
        plt.show()
        if throw_hist:
            return histdata
        else:            
            return None

    def extract_subgraphs(self, method='connected_neighbors',
                          nids=1, radii=5, include_central_node=True,
                          treat_undirected=True, validate=True,
                          see_on_map=False, upxo_gs_object=None,
                          figsize=(6, 6), dpi=100, throw_plt_object=False):
        """Extract one or more subgraphs from ``self.G`` using the chosen strategy.

        Parameters
        ----------
        method : {'connected_neighbors', 'largest_connected_component'}, default 'connected_neighbors'
            Extraction strategy.

            * ``'connected_neighbors'`` — ego-graph extraction: returns a
              neighbourhood subgraph of radius ``r`` centred on each node in
              ``nids``.
            * ``'largest_connected_component'`` — returns the single largest
              connected component of ``self.G``.
        nids : int or list of int, default 1
            Central node ID(s) for ``'connected_neighbors'``.  Scalars are
            wrapped in a list automatically.
        radii : int or list of int, default 5
            Neighbourhood radius for each entry in ``nids``.  A scalar is
            broadcast to all nodes.
        include_central_node : bool, default True
            Whether to include the centre node in each ego subgraph.
        treat_undirected : bool, default True
            Treat ``self.G`` as undirected during ego-graph extraction even if
            it is directed.
        validate : bool, default True
            Raise ``ValueError`` for unrecognised ``method`` strings.
        see_on_map : bool, default False
            Visualise each subgraph overlaid on the grain structure.  Requires
            ``upxo_gs_object`` to be provided.
        upxo_gs_object : upxo grain-structure object or None
            UPXO grain-structure instance used for map visualisation when
            ``see_on_map`` is True.
        figsize : tuple of float, default (6, 6)
            Figure size for map visualisation.
        dpi : int, default 100
            DPI for map visualisation.
        throw_plt_object : bool, default False
            Return matplotlib figure objects alongside subgraphs.

        Returns
        -------
        sg : dict
            Mapping of subgraph index → ``networkx.Graph`` subgraph.
        plt_objects : dict or None
            Mapping of subgraph index → matplotlib figure when
            ``see_on_map`` and ``throw_plt_object`` are both True; else None.
        """
        if validate:
            if method not in ('connected_neighbors',
                              'maximal_independent_set',
                              'largest_connected_component'
                              ):
                raise ValueError(f"method '{method}' not recognized.")
        if method == 'connected_neighbors':
            if type(nids) not in dth.dt.ITERABLES and type(nids) in dth.dt.NUMBERS:
                nids = [int(nids)]
            if type(radii) not in dth.dt.ITERABLES and type(radii) in dth.dt.NUMBERS:
                radii = [int(radii)]*len(nids)
        # -------------------------------------------
        # Extract connected neighbours subgraphs
        '''
        This is akin to extracting nth order neighboring grains of a given grain in a grain 
        structure. The method uses NetworkX's ego_graph function to create subgraphs centered
        around specified node IDs (nids) with a defined radius (radii). The subgraphs can include
        or exclude the central node based on the include_central_node parameter. Additionally,
        the treat_undirected parameter allows for treating the graph as undirected during extraction.
        Example
        -------
        sg = kmod.extract_subgraphs(method='connected_neighbors',
                                    nids=[1, 5, 10],
                                    radii=[2, 3, 4],
                                    include_central_node=True,
                                    treat_undirected=True)
        '''
        if method == 'connected_neighbors':
            sgcc_kwargs = {'nids': nids, 'radii': radii, 'include_central_node': include_central_node,
                           'treat_undirected': treat_undirected}
            sg = self.extract_subgraph_connected_neighbors(**sgcc_kwargs)
        # -------------------------------------------
        # Extract maximum independent set subgraphs
        # TODO: Implement this method
        # -------------------------------------------
        if method == 'largest_connected_component':
            '''
            This is akin to extracting the largest cluster of interconnected grains in a grain structure.
            This identifies and extracts the largest connected component from the graph G.
            '''
            # All oother inputs are ignored
            sg = self.extract_largest_connected_component()
        # -------------------------------------------
        plt_objects, plots_exist = {}, False
        if see_on_map and upxo_gs_object is not None:
            for sgcount, sg_ in sg.items():
                sg_nodes = list(sg_.nodes)
                plt_objects[sgcount] = upxo_gs_object.plot_grains(sg_nodes, hide_non_actors=True,
                                          default_cmap='jet',
                                          title=f"gid: {nids[sgcount]} | radius: {radii[sgcount]}",
                                          figsize=figsize, dpi=dpi, 
                                          throw_plt_object=throw_plt_object)
            plots_exist = True
        # -------------------------------------------
        if plots_exist and throw_plt_object:
            return sg, plt_objects
        else:
            return sg, None
        
    def extract_subgraph_connected_neighbors(self, **kwargs):
            """Build ego-graph subgraphs centred on specified nodes.

            Accepts keyword arguments forwarded from :meth:`extract_subgraphs`:

            Parameters (via kwargs)
            -----------------------
            nids : list of int
                Centre node IDs.
            radii : list of int
                Neighbourhood radius for each centre node.
            include_central_node : bool
                Include the centre node in the returned subgraph.
            treat_undirected : bool
                Ignore edge direction during traversal.

            Returns
            -------
            dict
                Mapping of sequential index (0-based) → ``networkx.Graph``
                ego subgraph.
            """
            sg = {}
            for nid_count, (nid, r) in enumerate(zip(kwargs['nids'], kwargs['radii']), start=0):
                sg[nid_count] = nx.ego_graph(self.G, nid, radius=r,
                                             center=kwargs.get('include_central_node', True),
                                             undirected=kwargs.get('treat_undirected', True),
                                             distance=None)
                print('Sub-Graph @ node id:', nid,
                      f'with radius: {r} has {sg[nid_count].number_of_nodes()} nodes '
                      f' and {sg[nid_count].number_of_edges()} edges.')
            return sg

    def extract_largest_connected_component(self):
        """Return a copy of the largest connected component of ``self.G``.

        Returns
        -------
        networkx.Graph
            Subgraph induced by the node set that forms the largest connected
            component.  Isolated nodes and smaller components are excluded.
        """
        largest_cc = max(nx.connected_components(self.G), key=len)
        lcc_subgraph = self.G.subgraph(largest_cc).copy()
        return lcc_subgraph
    
    def GET_connected_components(self, G):
        """Return all connected components of ``G`` as a list of subgraphs.

        Parameters
        ----------
        G : networkx.Graph
            Graph to decompose.

        Returns
        -------
        list of networkx.Graph
            One frozen subgraph per connected component, ordered arbitrarily.
        """
        return [G.subgraph(c).copy() for c in nx.connected_components(G)]

    def GET_maximal_independent_set(self, G):
        """Return a maximal independent set of nodes from ``G``.

        A maximal independent set (MIS) is a set of nodes such that no two are
        adjacent, and no additional node can be added without violating that
        property.  The result is non-deterministic because NetworkX uses a
        random greedy algorithm.

        Parameters
        ----------
        G : networkx.Graph
            Graph from which to compute the MIS.

        Returns
        -------
        list of int
            Node IDs forming a maximal independent set.
        """
        return nx.maximal_independent_set(G)

    def PRUNE_connected_component(self, cc, mis_nodes):
        """Remove a set of nodes from a connected component and return the result.

        Parameters
        ----------
        cc : networkx.Graph
            Connected-component subgraph to prune (copied internally; the
            original is not modified).
        mis_nodes : iterable of int
            Node IDs to remove.  Typically the MIS returned by
            :meth:`GET_maximal_independent_set`.

        Returns
        -------
        networkx.Graph
            Copy of ``cc`` with ``mis_nodes`` removed.
        """
        cc_pruned = cc.copy()
        cc_pruned.remove_nodes_from(mis_nodes)
        return cc_pruned
    
    def partition_into_nonconnected_sets_mis(self, see_results=True, verbose=False):
        """
        Iteratively decompose a graph by:
        1) Splitting into connected components
        2) Computing a maximal independent set (MIS) per component
        3) Removing MIS nodes
        4) Repeating until no nodes remain

        Returns a dict: round_index -> sorted list of MIS nodes removed that round.
        """
        G_working = self.G.copy()
        decomposition_layers, round_counter = {}, 0
        if verbose:
            print(f"Starting decomposition for a graph with {self.G.number_of_nodes()} nodes.")
        # Loop until graph is empty
        while G_working.number_of_nodes() > 0:
            round_counter += 1
            if verbose:
                print(f"\n--- Round {round_counter} ---")
            # Collect MIS nodes for this round
            mis_nodes_this_round = set()
            # 1) Components
            components = self.GET_connected_components(G_working)
            if verbose:
                print(f"Found {len(components)} connected component(s).")
            # 2) MIS per component
            for C in components:
                if C.number_of_nodes() == 1:
                    # Single node is trivially an MIS
                    node = next(iter(C.nodes()))
                    if verbose:
                        print(f"  Component (size 1): Node {node} selected.")
                    mis_nodes_this_round.add(node)
                else:
                    mis = self.GET_maximal_independent_set(C)
                    if verbose:
                        print(f"  Component (size {C.number_of_nodes()}): MIS of size {len(mis)} found.")
                    mis_nodes_this_round.update(mis)
            # 3) Record round
            decomposition_layers[round_counter] = sorted(list(mis_nodes_this_round))
            if verbose:
                print(f"Total MIS nodes removed in Round {round_counter}: {len(mis_nodes_this_round)}")
            # 4) Remove MIS
            G_working.remove_nodes_from(mis_nodes_this_round)
            if verbose:
                print(f"Remaining nodes in graph: {G_working.number_of_nodes()}")

        if see_results:
            self.see_nnodes_vs_peeldepth(decomposition_layers)
            
        return decomposition_layers
    
    def see_nnodes_vs_peeldepth(self, decomposition_layers):
        """Plot the number of MIS nodes removed at each decomposition round (peel depth).

        Parameters
        ----------
        decomposition_layers : dict
            Mapping of round index (1-based int) → sorted list of node IDs
            removed in that round, as returned by
            :meth:`partition_into_nonconnected_sets_mis`.
        """
        nnodes = [len(r) for r in decomposition_layers.values()]
        plt.figure(figsize=(6,4), dpi=120)
        plt.plot(list(range(1, len(nnodes)+1)), nnodes, marker='o', linestyle='-', color='b', markersize=6, 
                markerfacecolor='white', markeredgewidth=1, markeredgecolor='black')
        plt.xlabel('Decomposition Round')
        plt.ylabel('Number of cells in MIS')
        plt.title('MIS Size per Decomposition Round')

    def partition_into_nonconnected_sets_mis_nrealizations(self, n, throw_pd=False,
                                                           see_results=True,
                                                           see_types=['heatmap', 'mean_std'],
                                                           _disp_n_decimals=1,
                                                           figsize=(6,4), dpi=120,
                                                           save_partitions=False,
                                                           normalize_ng=False,
                                                           vmax=0.5
                                                           ):
        """Run MIS-based graph peeling ``n`` times and summarise the statistics.

        Because :meth:`GET_maximal_independent_set` is non-deterministic, each
        realisation may yield a different peel-depth profile.  This method
        aggregates ``n`` independent runs into a DataFrame and optionally
        visualises the spread.

        Parameters
        ----------
        n : int
            Number of independent decomposition realisations to run.
        throw_pd : bool, default False
            If True, return the pandas DataFrame of per-run node counts.
        see_results : bool, default True
            Display plots if True.
        see_types : list of str, default ['heatmap', 'mean_std']
            Which plots to show.  Recognised values:
            ``'boxplot'``, ``'violinplot'``, ``'heatmap'``, ``'mean_std'``.
        _disp_n_decimals : int, default 1
            Decimal places used when printing the descriptive statistics table.
        figsize : tuple of float, default (6, 4)
            Width and height of each figure in inches.
        dpi : int, default 120
            Resolution of each figure.
        save_partitions : bool, default False
            Store a deep copy of every realisation's decomposition dict.
        normalize_ng : bool, default False
            Divide node counts by the total number of graph nodes so that
            values are fractions rather than absolute counts.
        vmax : float, default 0.5
            Colour-scale maximum for the heatmap when ``normalize_ng`` is True.

        Returns
        -------
        n_decomposition_layers_np : numpy.ndarray, shape (n, max_depth)
            Node counts (or fractions) per realisation and peel depth.
            Shorter realisations are zero-padded on the right.
        n_decomposition_layers_pd : pandas.DataFrame or None
            Same data as a DataFrame with columns ``PD1, PD2, …``.
            ``None`` when ``throw_pd`` is False.
        partitions : list of dict or None
            Deep copies of each realisation's decomposition dict when
            ``save_partitions`` is True; else None.
        """
        # Run decomposition 100 times and collect results
        num_runs, n_decomposition_layers = n, []
        if save_partitions:
            partitions = []
        if normalize_ng:
            ng = self.G.number_of_nodes()
        print(f"Progress: ")
        for run_idx in range(num_runs):
            ntf = self.partition_into_nonconnected_sets_mis(see_results=False, verbose=False)
            if save_partitions:
                partitions.append(deepcopy(ntf))
            # Convert to list of node counts per round
            if normalize_ng:
                node_counts = [len(nodes)/ng for nodes in ntf.values()]
            else:
                node_counts = [len(nodes) for nodes in ntf.values()]
            n_decomposition_layers.append(node_counts)
            if (run_idx+1) % 100 == 0:
                print(f"{(run_idx + 1)*100/(num_runs)}%", end=', ', flush=False)
            if (run_idx+1) % 1000 == 0:
                print('\n')
        print(f"  completed.")
        # Some partitions may have different peel depths, so we will pad lesser ones with 0 at end
        max_length = max(len(r) for r in n_decomposition_layers)
        n_decomposition_layers_np = np.zeros((num_runs, max_length))
        for i, r in enumerate(n_decomposition_layers):
            n_decomposition_layers_np[i, :len(r)] = r
        n_decomposition_layers_pd = pd.DataFrame(n_decomposition_layers_np, 
                                                columns=[f"PD{j+1}" for j in range(max_length)])
        # Fill NaNs for shorter runs with 0
        n_decomposition_layers_pd = n_decomposition_layers_pd.fillna(0)
        print(n_decomposition_layers_pd.describe().round(_disp_n_decimals))
        print('\n', 'Note: PDn indicates peel depth n.')

        if see_results and 'boxplot' in see_types:
            plt.figure(figsize=figsize, dpi=dpi)
            sns.boxplot(data=n_decomposition_layers_pd, palette="Set3")
            plt.xlabel('Decomposition Round (Peel Depth)')
            plt.ylabel('Number of cells in MIS')
            plt.title('MIS Size Distribution per Decomposition Round')
            plt.show()

        if see_results and 'violinplot' in see_types:
            plt.figure(figsize=figsize, dpi=dpi)
            sns.violinplot(data=n_decomposition_layers_pd, palette="Set2", inner="quartile")
            plt.xlabel('Decomposition Round (Peel Depth)')
            plt.ylabel('Number of cells in MIS')
            plt.title('MIS Size Distribution per Decomposition Round')
            plt.show()

        if see_results and 'heatmap' in see_types:
            plt.figure(figsize=figsize, dpi=dpi)
            sns.heatmap(n_decomposition_layers_pd.transpose(), cmap="YlGnBu", cbar_kws={'label': 'Number of cells in MIS'}, 
                        vmax=vmax if normalize_ng else None)
            plt.xlabel('Decomposition Round (Peel Depth)')
            plt.ylabel('Number of cells in MIS')
            plt.title('MIS Size Distribution Heatmap per Decomposition Round')
            plt.show()

        if see_results and 'mean_std' in see_types:
            plt.figure(figsize=figsize, dpi=dpi)
            mean_values = n_decomposition_layers_pd.mean()
            std_values = n_decomposition_layers_pd.std()
            x_axis = range(1, len(mean_values) + 1)
            lower_bound = mean_values - std_values
            upper_bound = mean_values + std_values
            plt.plot(x_axis, mean_values, marker='o', linestyle='-', color='b', label='Mean MIS Size')
            plt.fill_between(x_axis, lower_bound, upper_bound, color='b', alpha=0.2, label='±1 Std Dev')
            plt.xlabel('Decomposition Round (Peel Depth)')
            plt.ylabel('Number of cells in MIS')
            plt.title('Mean MIS Size with Standard Deviation per Decomposition Round')
            plt.legend()
            plt.show()

        if not throw_pd:
            n_decomposition_layers_pd = None
        if not save_partitions:
            partitions = None
        return n_decomposition_layers_np, n_decomposition_layers_pd, partitions
    
    def fit_regr_lin_mis_partitions(self, n_decomposition_layers_np):
        """Fit a linear regression to each MIS decomposition realisation.

        For each row in ``n_decomposition_layers_np`` a degree-1 polynomial is
        fitted over the peel-depth axis.  Trailing zeros (shorter realisations)
        are stripped before fitting.  The 95 % confidence interval on each
        coefficient is computed from the covariance matrix returned by
        ``numpy.polyfit``.

        Parameters
        ----------
        n_decomposition_layers_np : numpy.ndarray, shape (n_runs, max_depth)
            Array of node counts per realisation and peel depth, as returned by
            :meth:`partition_into_nonconnected_sets_mis_nrealizations`.

        Returns
        -------
        regression_coeffs : numpy.ndarray, shape (n_runs, 2)
            ``[slope, intercept]`` for each realisation.
        confidence_bounds : numpy.ndarray, shape (n_runs, 2, 2)
            95 % CI lower/upper bounds on ``[slope, intercept]`` for each run.
            Rows with fewer than 2 valid data points are filled with NaN.
        gradients : numpy.ndarray of object
            Array of 1-D arrays, one per realisation, containing the
            element-wise finite differences (``numpy.diff``) of the node-count
            profile.
        """
        data = np.array(n_decomposition_layers_np, dtype=float)
        regression_coeffs = []
        confidence_bounds = []
        gradients = []
        z_score = 1.96  # 95% CI

        for row in data:
            y = row.copy()
            if y.size > 1 and y[-1] == 0:
                y = y[:-1]
            x = np.arange(y.size)
            if y.size < 2:
                regression_coeffs.append(np.full(2, np.nan))
                confidence_bounds.append(np.full((2, 2), np.nan))
                gradients.append(np.array([], dtype=float))
                continue

            coeffs, cov = np.polyfit(x, y, 1, cov=True)
            std_err = np.sqrt(np.diag(cov))
            bounds = np.column_stack((coeffs-z_score*std_err, coeffs+z_score*std_err))

            regression_coeffs.append(coeffs)
            confidence_bounds.append(bounds)
            gradients.append(np.diff(y))

        regression_coeffs = np.vstack(regression_coeffs)
        confidence_bounds = np.stack(confidence_bounds)
        gradients = np.array(gradients, dtype=object)

        return regression_coeffs, confidence_bounds, gradients

    def community(self, method='louvain', comprops=['modularity']):
        """Detect communities in the graph and compute community properties.

        Parameters
        ----------
        method : {'louvain', 'girvan_newman', 'label_propagation', 'greedy_modularity'}, default 'louvain'
            Community detection algorithm to apply.  Note that ``'louvain'`` is
            currently a placeholder (``pass``) and does not yet produce a result.
        comprops : list of str, default ['modularity']
            Community properties to compute after detection.  Currently
            ``'modularity'`` is the only supported value; it measures the
            strength of the community partition.

        Returns
        -------
        comm : community partition object
            The raw partition returned by the chosen NetworkX algorithm.
        modularity : pandas.DataFrame
            DataFrame with columns ``['k', 'modularity']`` giving the modularity
            of each community ``k`` when ``'modularity'`` is in ``comprops``.

        Notes
        -----
        Additional algorithms are listed at
        https://networkx.org/documentation/stable/reference/algorithms/community.html
        """
        if method == 'girvan_newman':
            comm = nx.community.girvan_newman(self.G)
        elif method == 'label_propagation':
            comm = nx.community.label_propagation_communities(self.G)
        elif method == 'louvain':
            pass
        elif method == 'greedy_modularity':
            comm = nx.community.greedy_modularity_communities(self.G)
        else:
            pass
        if comprops:
            community_properties = {}
            if 'modularity' in comprops:
                '''Modularity: measures the strength of 
                division of a network into modules'''
                print("Calculating modularity of components...")
                modularity = pd.DataFrame([[k+1, nx.community.modularity(self.G, comm[k])]
                                        for k in range(len(comm))], 
                                        columns=["k", "modularity"],)
        else:
            community_properties = {}
        return comm, modularity
    
    def _create_community_node_colors_(self, communities):
        """Map each node in ``self.G`` to a colour determined by its community.

        Parameters
        ----------
        communities : list of set
            Each element is a set of node IDs belonging to one community, as
            returned by a NetworkX community detection algorithm.

        Returns
        -------
        list of str
            Hex colour string for every node in ``self.G``, in the same
            iteration order as ``self.G``.  Up to five distinct colours are
            cycled across communities.
        """
        # function to create node colour list
        number_of_colors = len(communities)
        colors = ["#D4FCB1", "#CDC5FC", "#FFC2C4", "#F2D140", "#BCC6C8"][:number_of_colors]
        node_colors = []
        for node in self.G:
            current_community_index = 0
            for community in communities:
                if node in community:
                    node_colors.append(colors[current_community_index])
                    break
                current_community_index += 1
        return node_colors
    
    def visualize_communities(self, communities, i):
        '''
        Function to plot graph with node colouring based on communities
        Example
        -------
        G = nx.petersen_graph()
        communities = list(nx.community.girvan_newman(G))
        fig, ax = plt.subplots(len(communities)+1, figsize=(15, 20))
        for comm_count, comm in enumerate(communities):
            visualize_communities(G, comm, comm_count)
        modularity_df.plot.bar(
            x="k",
            ax=ax[2],
            color="#F2D140",
            title="Modularity Trend for Girvan-Newman Community Detection",
        )
        plt.show()
        '''
        node_colors = self._create_community_node_colors_(communities)
        modularity = round(nx.community.modularity(self.G, communities), 6)
        title = f"Community Visualization of {len(communities)} communities with modularity of {modularity}"
        pos = nx.spring_layout(self.G, k=0.3, iterations=50, seed=2)
        plt.subplot(3, 1, i)
        plt.title(title)
        nx.draw(self.G, pos=pos, node_size=1000, node_color=node_colors,
                with_labels=True, font_size=20, font_color="black",)
        
    def see_graph(self, plot_type='edges', seed=1):
        """Draw ``self.G`` using a spring layout.

        Parameters
        ----------
        plot_type : {'edges', 'nodes', 'numbered nodes'}, default 'edges'
            What to render.

            * ``'edges'`` — draw edges only (no node markers or labels).
            * ``'nodes'`` — draw nodes only (no edges or labels).
            * ``'numbered nodes'`` — draw the full graph with node-ID labels.
        seed : int, default 1
            Random seed passed to ``networkx.spring_layout`` for reproducible
            node positioning.
        """
        pos = nx.spring_layout(self.G, seed=seed)  # Seed layout for reproducibility
        if plot_type == 'numbered nodes':
            plt.figure(figsize=(4, 4))
            nx.draw(self.G, pos=pos, with_labels=True)
        if plot_type == 'nodes':
            plt.figure(figsize=(4, 4))
            nx.draw_networkx_nodes(self.G, pos, node_size=700)
        if plot_type == 'edges':
            plt.figure(figsize=(4, 4))
            nx.draw_networkx_edges(self.G, pos, width=1)

class gsan2d():
    __slots__ = ('gsstack', 'pnames', 'gsid', 'dfs', 'stts', 'corr', 'pca', 'K')

    defmp={'npixels': False, 'npixels_gb': False,
           'area': True, 'aspect_ratio': True,
           'eq_diameter': False, 'feret_diameter': False,
           'perimeter': False, 'perimeter_crofton': False,
           'gb_length_px': False,
           'compactness': False, 'solidity': True, 'circularity': True,
           'eccentricity': True, 'euler_number': True,
           'moments_hu': True, 'morph_ori': False,
           'major_axis_length': True, 'minor_axis_length': True,
           }

    chctrl={'char_grain_positions': True,
            'find_neigh': True,
            'find_neigh_p': 1.0,
            'find_neigh_include_central_feat': False,
            'find_neigh_throw_numba_dict': False,
            'char_gb': False,
            'get_grain_coords': False,
            'make_skim_prop': True
            }

    metaa = metaa()

    def __init__(self,
                 creation='distr_single',
                 stack={},
                 pnames=None):
        """Initialise the instance."""
        self.metaa.creation = creation

        if creation == 'pxtal_single':
            self.gsstack = stack
            self.pnames = pnames
            if 'aspect_ratio' in self.pnames:
                self.pnames
            self.gsid = [1]
            self.dfs = {self.gsid[0]: None}
            self.stts = {self.gsid[0]: None}

        if creation == 'pxtal_tmp':
            self.gsstack = stack
            self.pnames = pnames
            self.gsid = list(stack.keys())
            self.dfs = {gsid: None for gsid in self.gsid}
            self.stts = {gsid: None for gsid in self.gsid}

        if creation == 'pxtal_varied':
            pass

        if creation == 'distr_tmp':
            pass

        if creation == 'distr_varied':
            pass

        if creation == 'distr_single':
            pass
        
    @classmethod
    def from_mcgs2d_single(cls, gstslice, detect_grains=False, prechar=False,
                           find_neigh=True, find_neigh_p=1.0, 
                           find_neigh_include_central_feat=False, 
                           find_neigh_throw_numba_dict=False,
                           npixels=False, npixels_gb=False, gb_length_px=False,
                           eq_diameter=False, feret_diameter=False,
                           perimeter=False, perimeter_crofton=False, aspect_ratio=True,
                           compactness=False, solidity=True, morph_ori=False, circularity=False,
                           eccentricity=True, euler_number=True, moments_hu=True,
                           char_grain_positions=False, char_gb=False, get_grain_coords=True, connectivity=2):
        """Characterise a single 2-D grain-structure slice and wrap it for analysis.

        Calls ``gstslice.char_morph_2d`` (and optionally ``find_neigh_v2``)
        with the requested property flags, then returns a ``gsan2d`` instance
        whose ``gsstack`` contains this one slice under key ``1``.

        Parameters
        ----------
        gstslice : mcgs2_temporal_slice
            A single grain-structure time-slice object to characterise.
        detect_grains : bool, default False
            Run grain detection on ``gstslice`` before characterisation.
        prechar : bool, default False
            If True, skip ``char_morph_2d`` (assume the slice is already
            characterised).
        find_neigh : bool, default True
            Find grain neighbours via ``find_neigh_v2`` after characterisation.
        find_neigh_p : float, default 1.0
            Sampling probability for neighbour search; must be in ``[0, 1]``.
        find_neigh_include_central_feat : bool, default False
            Include the grain itself in its own neighbour list.
        find_neigh_throw_numba_dict : bool, default False
            Return a Numba-typed dict from the neighbour search.
        npixels : bool, default False
            Characterise grain area in pixels.
        npixels_gb : bool, default False
            Characterise number of grain-boundary pixels per grain.
        gb_length_px : bool, default False
            Characterise grain-boundary arc length in pixels.
        eq_diameter : bool, default False
            Characterise equivalent circular diameter.
        feret_diameter : bool, default False
            Characterise maximum Feret (calliper) diameter.
        perimeter : bool, default False
            Characterise grain perimeter length.
        perimeter_crofton : bool, default False
            Characterise perimeter using the Crofton formula.
        aspect_ratio : bool, default True
            Characterise aspect ratio; also enables ``major_axis_length``
            and ``minor_axis_length`` automatically.
        compactness : bool, default False
            Characterise compactness (4π·area / perimeter²).
        solidity : bool, default True
            Characterise solidity (area / convex-hull area).
        morph_ori : bool, default False
            Characterise morphological orientation in degrees.
        circularity : bool, default False
            Characterise circularity.
        eccentricity : bool, default True
            Characterise eccentricity of the best-fit ellipse.
        euler_number : bool, default True
            Characterise the Euler characteristic.
        moments_hu : bool, default True
            Characterise the seven Hu invariant moments.
        char_grain_positions : bool, default False
            Classify each grain as corner, edge, or internal.
        char_gb : bool, default False
            Characterise grain-boundary pixel locations.
        get_grain_coords : bool, default True
            Extract physical pixel coordinates for each grain.
        connectivity : int, default 2
            Connectivity for feature labelling (1 = 4-connected,
            2 = 8-connected in 2-D).

        Returns
        -------
        gsan2d
            Instance with ``creation='pxtal_single'``,
            ``gsstack={1: gstslice}``, and ``pnames`` set to every
            property flag that was True.

        Example
        -------
        >>> from upxo.ggrowth.mcgs import mcgs
        >>> from upxo.analysis.analysis2d import gsan2d
        >>> pxt = mcgs(input_dashboard='path/to/input_dashboard.xls')
        >>> pxt.simulate()
        >>> gsan = gsan2d.from_mcgs2d_single(pxt.gs[10],
        ...     solidity=True, eccentricity=True, euler_number=True,
        ...     moments_hu=True, get_grain_coords=False)
        """
        cls.defmp['npixels'], cls.defmp['npixels_gb'] = npixels, npixels_gb
        cls.defmp['gb_length_px'] = gb_length_px
        cls.defmp['eq_diameter'], cls.defmp['feret_diameter'] = eq_diameter, feret_diameter
        cls.defmp['perimeter'], cls.defmp['perimeter_crofton'] = perimeter, perimeter_crofton
        cls.defmp['compactness'], cls.defmp['solidity'] = compactness, solidity
        cls.defmp['morph_ori'], cls.defmp['circularity'] = morph_ori, circularity
        cls.defmp['eccentricity'], cls.defmp['euler_number'] = eccentricity, euler_number
        cls.defmp['moments_hu'], cls.chctrl['char_gb'] = moments_hu, char_gb

        cls.chctrl['char_grain_positions'] = char_grain_positions
        cls.chctrl['get_grain_coords'] = get_grain_coords

        cls.chctrl['find_neigh'] = find_neigh
        cls.chctrl['find_neigh_p'] = find_neigh_p
        cls.chctrl['find_neigh_include_central_feat'] = find_neigh_include_central_feat
        cls.chctrl['find_neigh_throw_numba_dict'] = find_neigh_throw_numba_dict
        if aspect_ratio:
            # Caution: This will re-write the default definition in class variable 'defmp'
            cls.defmp['major_axis_length'] = True
            cls.defmp['minor_axis_length'] = True
        if detect_grains:
            gstslice, state_ng = detect_grains_from_mcstates.mcgs2d(library='scikit-image', gs_dict={1: gstslice},
                                                             msteps=[1], kernel_order=2, store_state_ng=True)
        if not prechar:
            gstslice.char_morph_2d(bbox=True, bbox_ex=True, area=True,
                                   npixels= cls.defmp['npixels'],
                                   eq_diameter=cls.defmp['eq_diameter'],
                                   feret_diameter=cls.defmp['feret_diameter'],
                                   perimeter=cls.defmp['perimeter'], 
                                   perimeter_crofton=cls.defmp['perimeter_crofton'],
                                   npixels_gb=cls.defmp['npixels_gb'],
                                   gb_length_px=cls.defmp['gb_length_px'],
                                   major_axis_length=cls.defmp['major_axis_length'],
                                   minor_axis_length=cls.defmp['minor_axis_length'],
                                   aspect_ratio=cls.defmp['aspect_ratio'],
                                   compactness=cls.defmp['compactness'],
                                   solidity=cls.defmp['solidity'], 
                                   morph_ori=cls.defmp['morph_ori'],
                                   circularity=cls.defmp['circularity'],
                                   eccentricity=cls.defmp['eccentricity'],
                                   euler_number=cls.defmp['euler_number'],
                                   moments_hu=cls.defmp['moments_hu'],
                                   char_grain_positions=cls.chctrl['char_grain_positions'],
                                   append=False, saa=True, throw=False,
                                   find_neigh=False,  # Retain False until numba - Jupyter Kernel crash issue resolved
                                   char_gb=cls.chctrl['char_gb'],
                                   make_skim_prop=cls.chctrl['make_skim_prop'],
                                   get_grain_coords=cls.chctrl['get_grain_coords'])
            if cls.chctrl['find_neigh']:
                # Calling the numba accelerated function seperatelyto avoid Jupyter Kernel crash issue
                gstslice.find_neigh_v2(p=cls.chctrl['find_neigh_p'], 
                                       include_central_grain=cls.chctrl['find_neigh_include_central_feat'],
                                       throw_numba_dict=cls.chctrl['find_neigh_throw_numba_dict'],
                                       verbosity_nfids=1000)
        obj = cls(creation='pxtal_single',
                  stack={1: gstslice},
                  pnames=[k for k, i in cls.defmp.items() if i])
        return obj

    @classmethod
    def from_gsstack_varied(cls, gsstack):
        """Construct this instance from gsstack varied."""
        obj = cls(temporal=False, stack_type='varied', gsstack=gsstack)
        obj.gsstack = gsstack
        return obj

    @classmethod
    def from_gsstack_temporal(cls, gsstack, gsids=[],
                              detect_grains=False, ispxtal=False, prechar=False,
                              find_neigh=False, find_neigh_p=1.0, 
                              find_neigh_include_central_feat=False, 
                              find_neigh_throw_numba_dict=False,
                              npixels=False, npixels_gb=False, gb_length_px=False,
                              eq_diameter=False, feret_diameter=False,
                              perimeter=False, perimeter_crofton=False, aspect_ratio=True,
                              compactness=False, solidity=True, morph_ori=False,
                              circularity=False, eccentricity=True,
                              euler_number=True, moments_hu=True,
                              char_gb=False, get_grain_coords=False):
        """Characterise every slice in a temporal grain-structure stack and wrap for analysis.

        Iterates over the stack, calls ``char_morph_2d`` (and optionally
        ``find_neigh_v2``) on each slice with the requested property flags,
        then returns a ``gsan2d`` instance whose ``gsstack`` maps each
        grain-structure ID to its characterised slice.

        Parameters
        ----------
        gsstack : dict[int, mcgs2_temporal_slice] or pxtal
            Temporal stack of grain-structure slices. If ``ispxtal`` is
            True, this must be a pxtal object exposing a ``.gs`` attribute.
        gsids : list of int, default []
            Subset of grain-structure IDs to include. An empty list
            uses all IDs present in ``gsstack``.
        detect_grains : bool, default False
            Run grain detection on each slice before characterisation.
        ispxtal : bool, default False
            If True, treat ``gsstack`` as a pxtal object and extract its
            ``.gs`` dict, filtered to ``gsids`` when provided.
        prechar : bool, default False
            If True, skip ``char_morph_2d`` on all slices (assume already
            characterised).
        find_neigh : bool, default False
            Find grain neighbours in each slice via ``find_neigh_v2``.
        find_neigh_p : float, default 1.0
            Sampling probability for neighbour search; must be in ``[0, 1]``.
        find_neigh_include_central_feat : bool, default False
            Include the grain itself in its own neighbour list.
        find_neigh_throw_numba_dict : bool, default False
            Return a Numba-typed dict from the neighbour search.
        npixels : bool, default False
            Characterise grain area in pixels.
        npixels_gb : bool, default False
            Characterise number of grain-boundary pixels per grain.
        gb_length_px : bool, default False
            Characterise grain-boundary arc length in pixels.
        eq_diameter : bool, default False
            Characterise equivalent circular diameter.
        feret_diameter : bool, default False
            Characterise maximum Feret (calliper) diameter.
        perimeter : bool, default False
            Characterise grain perimeter length.
        perimeter_crofton : bool, default False
            Characterise perimeter using the Crofton formula.
        aspect_ratio : bool, default True
            Characterise aspect ratio; also enables ``major_axis_length``
            and ``minor_axis_length`` automatically.
        compactness : bool, default False
            Characterise compactness (4π·area / perimeter²).
        solidity : bool, default True
            Characterise solidity (area / convex-hull area).
        morph_ori : bool, default False
            Characterise morphological orientation in degrees.
        circularity : bool, default False
            Characterise circularity.
        eccentricity : bool, default True
            Characterise eccentricity of the best-fit ellipse.
        euler_number : bool, default True
            Characterise the Euler characteristic.
        moments_hu : bool, default True
            Characterise the seven Hu invariant moments.
        char_gb : bool, default False
            Characterise grain-boundary pixel locations.
        get_grain_coords : bool, default False
            Extract physical pixel coordinates for each grain.

        Returns
        -------
        gsan2d
            Instance with ``creation='pxtal_tmp'``, ``gsstack`` mapping
            each included grain-structure ID to its characterised slice,
            and ``pnames`` set to every property flag that was True.

        Example
        -------
        >>> from upxo.ggrowth.mcgs import mcgs
        >>> from upxo.analysis.analysis2d import gsan2d
        >>> pxt = mcgs(input_dashboard='path/to/input_dashboard.xls')
        >>> pxt.simulate()
        >>> gsan = gsan2d.from_gsstack_temporal(pxt, ispxtal=True,
        ...     solidity=True, eccentricity=True, euler_number=True)
        """
        cls.defmp['npixels'], cls.defmp['npixels_gb'] = npixels, npixels_gb
        cls.defmp['gb_length_px'] = gb_length_px
        cls.defmp['eq_diameter'], cls.defmp['feret_diameter'] = eq_diameter, feret_diameter
        cls.defmp['perimeter'], cls.defmp['perimeter_crofton'] = perimeter, perimeter_crofton
        cls.defmp['compactness'], cls.defmp['solidity'] = compactness, solidity
        cls.defmp['morph_ori'], cls.defmp['circularity'] = morph_ori, circularity
        cls.defmp['eccentricity'], cls.defmp['euler_number'] = eccentricity, euler_number
        cls.defmp['moments_hu'], cls.chctrl['char_gb'] = moments_hu, char_gb

        cls.chctrl['get_grain_coords'] = get_grain_coords

        cls.chctrl['find_neigh'] = find_neigh
        cls.chctrl['find_neigh_p'] = find_neigh_p
        cls.chctrl['find_neigh_include_central_feat'] = find_neigh_include_central_feat
        cls.chctrl['find_neigh_throw_numba_dict'] = find_neigh_throw_numba_dict

        if aspect_ratio:
            # Caution: This will re-write the default definition in class variable 'defmp'
            cls.defmp['major_axis_length'] = True
            cls.defmp['minor_axis_length'] = True

        if ispxtal:
            val_ = list(set([gsid in gsstack.gs.keys() for gsid in gsids]))
            if len(gsids) > 0 and False in val_:
                raise ValueError('One or more specified gsids not found in gsstack.')
            if len(gsids) > 0 and len(val_) == 1 and val_[0]:
                gsstack = {i: gsstack.gs[i] for i in gsids}
            if len(gsids) == 0:
                gsstack = {m: gs for m, gs in gsstack.gs.items()}

        if detect_grains:
            for gscount in gsstack.keys():
                print(40*'=', '\n', f"Detecting grains in grain structure slice {gscount}", '\n', 40*'=')
                gsstack[gscount].detect_grains()

        if not prechar:
            for gscount in gsstack.keys():
                print(40*'=', '\n', f"Characterizing grain structure slice {gscount}", '\n', 40*'=')
                gsstack[gscount].char_morph_2d(bbox=True, bbox_ex=True, area=True,
                                               npixels= cls.defmp['npixels'],
                                               eq_diameter=cls.defmp['eq_diameter'],
                                               feret_diameter=cls.defmp['feret_diameter'],
                                               perimeter=cls.defmp['perimeter'], 
                                               perimeter_crofton=cls.defmp['perimeter_crofton'],
                                               npixels_gb=cls.defmp['npixels_gb'],
                                               gb_length_px=cls.defmp['gb_length_px'],
                                               major_axis_length=cls.defmp['major_axis_length'],
                                               minor_axis_length=cls.defmp['minor_axis_length'],
                                               aspect_ratio=cls.defmp['aspect_ratio'],
                                               compactness=cls.defmp['compactness'],
                                               solidity=cls.defmp['solidity'], 
                                               morph_ori=cls.defmp['morph_ori'],
                                               circularity=cls.defmp['circularity'],
                                               eccentricity=cls.defmp['eccentricity'],
                                               euler_number=cls.defmp['euler_number'],
                                               moments_hu=cls.defmp['moments_hu'],
                                               append=False, saa=True, throw=False,
                                               char_grain_positions=True,
                                               find_neigh=False,  # Retain False until numba - Jupyter Kernel crash issue resolved
                                               char_gb=cls.chctrl['char_gb'],
                                               make_skim_prop=cls.chctrl['make_skim_prop'],
                                               get_grain_coords=cls.chctrl['get_grain_coords'])
                if cls.chctrl['find_neigh']:
                    print(10*f"{40*'-'}")
                    # Calling the numba accelerated function seperatelyto avoid Jupyter Kernel crash issue
                    gsstack[gscount].find_neigh_v2(p=cls.chctrl['find_neigh_p'], 
                                                   include_central_grain=cls.chctrl['find_neigh_include_central_feat'],
                                                   throw_numba_dict=cls.chctrl['find_neigh_throw_numba_dict'],
                                                   verbosity_nfids=1000)

        obj = cls(creation='pxtal_tmp',
                  stack=gsstack,
                  pnames=[k for k, i in cls.defmp.items() if i])
        return obj

    @classmethod
    def from_distr(cls, distributions):
        """Construct a gsan2d instance from pre-computed property distributions.

        Parameters
        ----------
        distributions : dict[str, array-like]
            Mapping of property name → array of values, one per grain.

        Notes
        -----
        Not yet implemented.
        """
        raise NotImplementedError('from_distr is not yet implemented.')
    
    def find_neigh(self, gsids=None, p=1.0,
                   include_central_feat=False,
                   throw_numba_dict=False,
                   verbosity_nfids=1000):
        """Find neigh."""
        # -------------------------------------------
        # Validations
        if not gsids:
            gsids = self.gsid
        if type(gsids) not in dth.dt.ITERABLES and type(gsids) in dth.dt.NUMBERS:
            gsids = [int(gsids)]
        if 0 <= p <= 1.0:
            p = float(p)
        else:
            raise ValueError('p must be in the range [0, 1]')
        # -------------------------------------------
        for gsid in gsids:
            self.gsstack[gsid].find_neigh_v2(p=p,
                                             include_central_grain=include_central_feat,
                                             throw_numba_dict=throw_numba_dict,
                                             verbosity_nfids=verbosity_nfids)

    def find_neigh_variable_settings(self, gsids=None, p=1.0,
                                     include_central_feat=[False],
                                     throw_numba_dict=False,
                                     verbosity_nfids=1000):
        """Find neigh variable settings."""
        # -------------------------------------------
        # Validations
        if not gsids:
            gsids = self.gsid
        if type(gsids) not in dth.dt.ITERABLES and type(gsids) in dth.dt.NUMBERS:
            gsids = [int(gsids)]
        # Check p
        if type(p) not in dth.dt.ITERABLES and type(p) in dth.dt.NUMBERS:
            if 0 <= p <= 1.0:
                p = [float(p)]*len(gsids)
            else:
                raise ValueError('p must be in the range [0, 1]')
        # Check each p value
        if type(p) in dth.dt.ITERABLES:
            if len(p) != len(gsids):
                raise ValueError('Length of p must match length of gsids')
            for pi in p:
                if not (0 <= pi <= 1.0):
                    raise ValueError('Each p value must be in the range [0, 1]')
        # -------------------------------------------
        for gsid_count, gsid in enumerate(gsids, start=0):
            self.gsstack[gsid].find_neigh_v2(p=p[gsid_count],
                                             include_central_grain=include_central_feat,
                                             throw_numba_dict=throw_numba_dict,
                                             verbosity_nfids=verbosity_nfids)

    def extract_props(self):
        """Extract props."""
        if self.metaa.creation == 'pxtal_single':
            self.extract_props_pxtal_single()
        if self.metaa.creation == 'pxtal_tmp':
            for gsid in self.gsid:
                self.extract_props_pxtal_single(gsid=gsid)
            self.dfs['temporal'] = pd.concat(self.dfs, names=['time_slice']).reset_index(level=0)

        if self.metaa.creation == 'pxtal_varied':
            pass
        if self.metaa.creation == 'distr_single':
            pass

    def extract_props_pxtal_single(self, gsid=None):
        """Extract props pxtal single."""
        if not gsid:
            gsid = self.gsid[0]
        gsobj = self.gsstack[gsid]
        data = {pname: [] for pname in self.pnames if pname not in ('aspect_ratio',)}
        for gcount in range(len(gsobj.g.keys())):
            grain = gsobj.g[gcount+1]['grain'].skprop
            for prop in self.pnames:
                if prop not in ('aspect_ratio',):
                    data[prop].append(getattr(grain, prop))
        data = {k: np.array(v) for k, v in data.items()}
        
        if 'moments_hu' in self.pnames:
            for mhu_i, mhu_name in enumerate([f'mhu_{i+1}' for i in range(data['moments_hu'].shape[1])], start=0):
                data[mhu_name] = data['moments_hu'][:, mhu_i]
            del data['moments_hu']
        # for k in data.keys():
        #     print(f"{k}: {data[k].shape}")
        self.dfs[gsid] = self._create_dfs_(data)

    def _create_dfs_(self, data_dict):
        """ create dfs ."""
        df = pd.DataFrame({pname: data_dict[pname] for pname in data_dict.keys()})
        if 'orientation' in df.columns:
            df['orientation'] = df['orientation']*(180/np.pi)

        if 'aspect_ratio' in self.pnames:
            df['aspect_ratio'] = df['major_axis_length']/df['minor_axis_length']
            df['aspect_ratio'] = df['aspect_ratio'].replace([np.inf, -np.inf], np.nan, inplace=False)
        
        return df

    def compute_temporal_dfs(self):
        """Return the ute temporal dfs."""
        # Placeholder for method to compute temporal dataframes
        pass

    def compute_statistics(self):
        """Return the ute statistics."""
        for gsid in self.gsid:
            self.stts[gsid] = self.dfs[gsid].describe()
            self.stts[gsid].loc['skew'] = self.dfs[gsid].skew()
            self.stts[gsid].loc['kurt'] = self.dfs[gsid].kurt()
        if len(self.gsid) > 1:
            self.stts['temporal'] = pd.concat(self.stts, names=['time_slice']).reset_index(level=0)

    def correlate(self, gsids=[1], pnames=['area', 'major_axis_length', 'minor_axis_length', 'eccentricity'],
                  saa=True, throw=False):
        """Correlate."""
        if len(gsids) == 0:
            gsids = self.gsid
        if len(pnames) == 0:
            corr = {gsid: self.dfs[gsid].corr() for gsid in gsids}
            corr['pnames'] = pnames
        elif len(pnames) == 1:
            raise ValueError('Cannot correlate with single parameter')
        else:
            corr = {gsid: self.dfs[gsid][pnames].corr() for gsid in gsids}
            corr['pnames'] = pnames
        if saa:
            self.corr = corr
        if throw:
            return corr

    def correlate_temporal(self, pnames=['area', 'major_axis_length', 'minor_axis_length', 'eccentricity']):
        """Correlate temporal."""
        slices = sorted(self.dfs['temporal']['time_slice'].unique())
        num_slices = len(slices)
        corr_volume = np.zeros((num_slices, len(pnames), len(pnames)))
        for i, t in enumerate(slices):
            corr_matrix = self.dfs['temporal'][self.dfs['temporal']['time_slice'] == t][pnames].corr().values
            corr_volume[i] = corr_matrix
        self.corr['temporal'] = corr_volume

    def pca_analysis(self, gsids=[1], gids=[],
                     pnames=['area', 'major_axis_length', 'minor_axis_length', 'eccentricity'],
                     auto_ncomp=True, ncomp_method='mle', svd_solver='auto', saa=True, throw=False,
                     see_scree=True, annotate=True, see_exvar=True, see_cum_exvar=False,
                     figsize=(8, 3)):
        """Run principal component analysis on grain property data.

        Standardises the selected properties for each requested grain-structure
        ID, fits a PCA model, and optionally plots explained-variance curves.
        Results are stored in ``self.pca[gsid]`` when ``saa`` is True.

        Parameters
        ----------
        gsids : list of int, default [1]
            Grain-structure IDs to analyse. An empty list selects all
            integer keys present in ``self.dfs``.
        gids : list of int, default []
            Specific grain IDs (1-indexed rows) to include. An empty list
            uses all grains in the dataframe.
        pnames : list of str, default ['area', 'major_axis_length', 'minor_axis_length', 'eccentricity']
            Property names used as PCA features; must be columns in
            ``self.dfs[gsid]``. An empty list uses all available columns.
        auto_ncomp : bool, default True
            If True, fit with ``n_components=len(pnames)`` (all components).
            If False, ``ncomp_method`` is passed as ``n_components``.
        ncomp_method : str or int, default 'mle'
            Passed as ``n_components`` to ``sklearn.decomposition.PCA``
            when ``auto_ncomp`` is False. Common values: ``'mle'`` or an
            integer.
        svd_solver : str, default 'auto'
            SVD solver forwarded to ``PCA``; see scikit-learn documentation.
        saa : bool, default True
            Save-and-apply: store fitted results in ``self.pca[gsid]``.
        throw : bool, default False
            If True, return ``(pca_, scores_, exvar_)``; otherwise return
            ``(None, None, None)``.
        see_scree : bool, default True
            Reserved for a future scree plot; not yet implemented.
        annotate : bool, default True
            Reserved for annotation of variance plots; not yet implemented.
        see_exvar : bool, default True
            Plot per-component explained variance (%) for each gsid.
        see_cum_exvar : bool, default False
            Plot cumulative explained variance (%) for each gsid.
        figsize : tuple of float, default (8, 3)
            Figure size ``(width, height)`` in inches for variance plots.

        Returns
        -------
        pca_ : dict[int, sklearn.decomposition.PCA] or None
            gsid → fitted PCA object. ``None`` when ``throw`` is False.
        scores_ : dict[int, numpy.ndarray] or None
            gsid → score array of shape ``(n_grains, n_components)``.
            ``None`` when ``throw`` is False.
        exvar_ : dict[int, numpy.ndarray] or None
            gsid → explained-variance-ratio array of length
            ``n_components``. ``None`` when ``throw`` is False.

        Notes
        -----
        Rows containing NaN values are dropped before fitting.
        ``self.pca`` is populated with :class:`principle_component_analysis`
        objects keyed by gsid.
        """
        
        if len(gsids) == 0:
            gsids = [i for i in list(self.dfs.keys()) if type(i)==int]

        colors = plt.cm.tab10.colors  # 10 distinct colors
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']  # variety of markers
        linewidths = [1.0, 1.0, 1.0, 1.0]  # cycle through thickness
        lw_last = sum(linewidths)*1.1

        self.pca = {gsid: principle_component_analysis() for gsid in gsids}
        
        pca_, scores_, exvar_ = {}, {}, {}
        ngids = len(gsids)
        if ngids > 3:
            alphas = np.linspace(0, 1.0, ngids)
            alphas[0] = alphas[1]*0.5
            alphas[-2] = 1.0
            alphas[-1] = 0.5
        else:
            alphas = [0.5]*ngids
        for gscount, gsid in enumerate(gsids, start=0):
            if gsid != 'temporal':
                if len(pnames) == 0:
                    pnames = self.dfs[gsid].columns.to_list()
                    nprops = len(pnames)
                prop_value = self.dfs[gsid][pnames]
                # If gids specified, filter prop_value
                if len(gids) > 0:
                    prop_value = prop_value.iloc[[gid-1 for gid in gids], :]
                prop_value = prop_value.dropna()
                scaled_data = StandardScaler().fit_transform(prop_value)
                pca = PCA(n_components=len(pnames) if auto_ncomp else ncomp_method)
                scores = pca.fit_transform(scaled_data)
                exvar = pca.explained_variance_ratio_

                '''color = colors[gscount % len(colors)]
                marker = markers[gscount % len(markers)]
                lw = lw_last if gsid == gsids[-1] else linewidths[gscount % len(linewidths)]'''

                '''plt.plot(range(1, len(exvar)+1), exvar*100,
                        marker=marker, color=color, linewidth=lw,
                        label=f'GSID {gsid}', alpha=alphas[gscount])'''
                pca_[gsid], scores_[gsid], exvar_[gsid] = pca, scores, exvar
        if saa:
            for gsid in gsids:
                if gsid != 'temporal':
                    self.pca[gsid].pca = pca_[gsid]
                    self.pca[gsid].scores = scores_[gsid]
                    self.pca[gsid].exvar = exvar_[gsid]

        if see_exvar:
            plt.figure(figsize=figsize)
            for gscount, gsid in enumerate(gsids, start=0):
                color = colors[gscount % len(colors)]
                marker = markers[gscount % len(markers)]
                lw = lw_last if gsid == gsids[-1] else linewidths[gscount % len(linewidths)]
                plt.plot(range(1, len(exvar_[gsid])+1), exvar_[gsid]*100,
                            marker=marker, color=color, linewidth=lw,
                            label=f'GSID {gsid}', alpha=alphas[gscount])
            # plt.title('Scree plot', fontsize=14)
            plt.xlabel('Principal component', fontsize=12)
            plt.ylabel('Variance explained (%)', fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(title='', loc='best', ncol=5,)
            plt.tight_layout()
            plt.show()

        if see_cum_exvar:
            plt.figure(figsize=figsize)
            for gscount, gsid in enumerate(gsids, start=0):
                color = colors[gscount % len(colors)]
                marker = markers[gscount % len(markers)]
                lw = lw_last if gsid == gsids[-1] else linewidths[gscount % len(linewidths)]
                plt.plot(range(1, len(exvar_[gsid])+1), np.cumsum(exvar_[gsid])*100,
                            marker=marker, color=color, linewidth=lw,
                            label=f'GSID {gsid}', alpha=alphas[gscount])
            # plt.title('Scree plot', fontsize=14)
            plt.xlabel('Cumulative principal component', fontsize=11)
            plt.ylabel('Variance explained (%)', fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(title='', loc='best', ncol=5,)
            plt.tight_layout()
            plt.show()
        if throw:
            return pca_, scores_, exvar_
        else:
            return None, None, None

    def initiate_kmodel(self, gsids=[1], k_char_level='none',
                        recalculate_neighbours=True,
                        include_central_grain=False):
        """Initiate kmodel."""
        if type(gsids) in dth.dt.NUMBERS and int(gsids) in self.gsid:
            gsids = [int(gsids)]
        elif len(gsids) == 0:
            gsids = self.gsid
        else:
            raise ValueError('Invalid specification of gsids.')
        self.K = {gsid: None for gsid in gsids}
        for gsid in gsids:
            if recalculate_neighbours:
                self.gsstack[gsid].find_neigh(include_central_grain=include_central_grain)
            self.K[gsid] = kmodel(G=self.gsstack[gsid].make_graph(self.gsstack[gsid].neigh_gid))
            if k_char_level in ('basic', 'simple', 'full', 'advanced'):
                self.K[gsid].characterize_graph(k_char_level=k_char_level)

    def see_stats(self, gsid=[1], pname='area', metric='mean'):
        """Plot a summary statistic of one property across specified time slices.

        Parameters
        ----------
        gsid : int or list of int, default [1]
            Time-slice ID(s) to include. Each must have been characterised
            via :meth:`compute_statistics` before calling this method.
        pname : str, default 'area'
            Property name; must be a column in ``self.dfs[gsid]``.
        metric : str, default 'mean'
            Row label in the statistics table produced by
            ``DataFrame.describe()``. Valid values include ``'mean'``,
            ``'std'``, ``'min'``, ``'25%'``, ``'50%'``, ``'75%'``,
            ``'max'``, ``'skew'``, ``'kurt'``.
        """
        gsids = gsid if isinstance(gsid, list) else [gsid]
        values = [self.stts[g].loc[metric, pname] for g in gsids]
        plt.figure(figsize=(4, 3))
        plt.plot(gsids, values, marker='o', linestyle='-', color='purple')
        plt.title(f'{metric.capitalize()} of {pname}', fontsize=14)
        plt.xlabel('Time slice')
        plt.ylabel(f'{metric.capitalize()} of {pname}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def see_dstr_univariate(self, gsid=1, pnames=['area'], 
                            bw_adjust=[0.75], kde_clr=['blue'],
                            title_fsz=14, xmax_mult=1.1, grid_alpha=0.3, multiple='stack', 
                            kind='kde', fill=True, 
                            ):
        """See dstr univariate."""
        # Input validations
        for pname in pnames:
            if pname not in self.dfs[gsid].columns.to_list():
                print(f"Property name '{pname}' not found in characterized properties.")
                return

        if type(bw_adjust) not in dth.dt.ITERABLES:
            bw_adjust = [bw_adjust]
        if len(bw_adjust) < len(pnames):
            bw_adjust = bw_adjust*len(pnames)
        if len(bw_adjust) > len(pnames):
            bw_adjust = [bw_adjust[i] for i in range(len(pnames))]

        if type(kde_clr) not in dth.dt.ITERABLES and type(kde_clr) == str:
            kde_clr = [kde_clr]
        else: 
            kde_clr = ['blue']
        if len(kde_clr) < len(pnames):
            kde_clr = kde_clr*len(pnames)
        if len(kde_clr) > len(pnames):
            kde_clr = [kde_clr[i] for i in range(len(pnames))]

        for pcount, pname in enumerate(pnames):
            if kind == 'kde':
                sns.displot(self.dfs[gsid][pname], bw_adjust=bw_adjust[pcount],
                            kind='kde', multiple=multiple, fill=fill, color=kde_clr[pcount], cut=0)
            elif kind == 'ecdf':
                sns.displot(self.dfs[gsid][pname], kind='ecdf')
        
    def see_dstr_bivariate(self, gsid=1, pnames=['area', 'aspect_ratio'], jointplot=False, levels=5):
        """
        Example
        -------
        gsan.see_dstr_bivariate(gsid=1, pnames=['area', 'aspect_ratio'])
        """
        if len(pnames) != 2:
            raise ValueError('Invalid pnames specification.')
        if not jointplot:
            sns.displot(self.dfs[gsid][pnames], x=pnames[0], y=pnames[1], kind='kde', levels=levels)
        else:
            sns.jointplot(self.dfs[gsid][pnames], x=pnames[0], y=pnames[1], kind='kde', levels=levels)

    def see_pairgrid(self, gsid=1, pnames=['area', 'aspect_ratio']):
        """See pairgrid."""
        sns.PairGrid(self.dfs[gsid][pnames])  # To be debugged

    def see_correlation(self, gsids=[1], pnames=['area', 'perimeter'], recorrelate=True):
        """See correlation."""
        if recorrelate:
            corr = self.correlate(gsids=gsids, pnames=pnames, saa=False, throw=True)
        else:
            corr = self.corr

        if len(gsids) == 1:
            sns.heatmap(corr[gsids[0]], annot=True, cmap='nipy_spectral')
        else:
            pass
    
    def see_correlation_temporal(self):
        """See correlation temporal."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for this method. "
                "Install it with: pip install plotly"
            )
        slices = sorted(self.dfs['temporal']['time_slice'].unique())
        num_slices = self.corr['temporal'].shape[0]
        corr_volume = self.corr['temporal']
        pnames = self.corr['pnames']
        frames = []
        for i in range(num_slices):
            frames.append(go.Frame(
                data=[go.Heatmap(
                    z=corr_volume[i],
                    x=pnames,
                    y=pnames,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title='Correlation')
                )],
                name=f'Slice {slices[i]}'
            ))

        fig = go.Figure(
            data=[go.Heatmap(
                z=corr_volume[0],
                x=pnames,
                y=pnames,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                colorbar=dict(title='Correlation')
            )],
            layout=go.Layout(
                title='Correlation Heatmap Across Time Slices',
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {'label': 'Play', 'method': 'animate',
                        'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}]},
                        {'label': 'Pause', 'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
                    ]
                }],
                sliders=[{
                    'active': 0,
                    'steps': [
                        {'label': f'Slice {slices[i]}', 'method': 'animate',
                        'args': [[f'Slice {slices[i]}'], {'mode': 'immediate', 'frame': {'duration': 500, 'redraw': True}}]}
                        for i in range(num_slices)
                    ]
                }]
            ),
            frames=frames
        )
        fig.show()

    def see_dstr_stack(self, pname='area', metric='mean'):
        """See dstr stack."""
        values = []
        for tslice in range(len(self.stts)):
            values.append(self.stts[tslice].loc[metric, pname])
        plt.figure(figsize=(4, 3))
        plt.plot(range(len(self.stts)), values, marker='o', linestyle='-', color='purple')
        plt.title(f'{metric.capitalize()} of {pname} over Time Slices', fontsize=14)
        plt.xlabel('Time Slice')
        plt.ylabel(f'{metric.capitalize()} of {pname}')
        plt.grid(alpha=0.3)
        plt.tight_layout()

    def see_stats_stack(self, pname='', metric=''):
        """See stats stack."""
        # Placeholder for method to plot statistics
        pass

    def see_evol(self, pname='area', plottype='basic', metric='mean'):
        """See evol."""
        plt.figure(figsize=(4, 3))
        if plottype == 'basic':
            pvals = []  # Parameter values
            for gsid in self.gsid:
                pvals.append(self.stts[gsid].loc[metric, pname])
            plt.plot(self.gsid, pvals, marker='o', linestyle='-', color='purple')
            plt.title(f'{metric.capitalize()} of {pname} over m: {self.gsid[0]}:{self.gsid[-1]}', fontsize=14)
            plt.xlabel('Time Slice')
            plt.ylabel(f'{metric.capitalize()} of {pname}')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        elif plottype == 'a':
            sns.lineplot(data=self.dfs['temporal'], x='time_slice', y=pname)


