"""
Created on Thu Jul  4 16:45:03 2024

@author: Dr. Sunil Anandatheertha
"""
import numpy as np
import rasterio
from abc import ABC, abstractmethod
from copy import deepcopy
import matplotlib.pyplot as plt
from rasterio.features import shapes
from shapely.strtree import STRtree
from shapely.geometry import Point
import upxo._sup.data_ops as DO
from shapely import affinity as ShAff
from upxo._sup import dataTypeHandlers as dth
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import shape as ShShape
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection
from shapely.geometry import Polygon, MultiPoint
from upxo.geoEntities.point2d import Point2d
from upxo.geoEntities.mulpoint2d import MPoint2d
from shapely.geometry import Point as ShPoint2d
from upxo.geoEntities.mulsline2d import MSline2d
from upxo.geoEntities.mulsline2d import ring2d
from upxo._sup.data_ops import find_common_coordinates
# from meshpy.triangle import MeshInfo, build
from scipy.ndimage import generic_filter


class polygonised_grain_structure():

    __slots__ = ('lgi', 'gid', 'n', 'polygons', 'pxtal', 'neigh_gid',
                 'gsmp', 'gbsegments_raw', 'gbsegments_mls', 'gbseg_smoothed',
                 'neigh_gid_pxtal', 'GBSEG', 'GBPOINTS',
                 'raster_img_polygonisation_results', 'polygons_raw_exteriors',
                 'polygons_raw_holes', 'neigh_pols', 'centroids_raw',
                 'centroids', 'xyoffset', 'grain_loc_ids',
                 'JNP', 'GBP', 'GBP_pure', 'GBP_at_boundary', 'jnp_all_coords',
                 'quality', 'sorted_segs',
                 'jnp_all_upxo', 'jnp_all_shapely', 'gbp_all_coords',
                 'gbp_all_upxo', 'gbp_all_shapely', 'jnp_grain_wise_indices',
                 'gbp_grain_wise_coords', 'gbp_grain_wise_indices',
                 'gbp_grain_wise_points', 'gbmullines_grain_wise',
                 'junction_points_coord', 'jnp_all_sorted_coords',
                 'jnp_all_sorted_upxo', 'gbsegments', 'gid_pair_ids',
                 'gid_pair_ids_unique_lr', 'gid_pair_ids_unique_rl',
                 'nconn', 'GBSEGMENTS', 'consolidated_segments',
                 'GB', 'GBCoords', 'GRAINS', 'POLYXTAL', 'mids_all_gbsegs',
                 'sgseg_obj_list', 'smoothed')

    EPS_coord_coincide = 1E-8

    def __init__(self, lgi, gids, neigh_gid_pxtal):
        self.lgi, self.gid, self.neigh_gid_pxtal = lgi, gids, neigh_gid_pxtal
        self.n = len(np.unique(self.gid))
        self.GBSEG = {gid: [] for gid in self.gid}
        # mp = self.polygonize(verbose=True)
        self.polygons = None
        self.polygons_raw_exteriors = {gid: [] for gid in self.gid}
        self.polygons_raw_holes = {gid: [] for gid in self.gid}
        self.gsmp = None
        self.smoothed = {}

    def geometrify(self, verbose=True):
        self.polygonize(self.lgi, self.gid, verbose=verbose)

    def polygonize(self, user_lgi=False, lgi=None, user_gids=False, gids=None,
                   verbose=True):
        """
        Polygonizes grains in self.lgi
        Parameters
        ----------
        Returns
        -------
        """
        if verbose:
            print("Polygonizing the raster image of the grain structure.")
        if not user_lgi:
            lgi, gid = self.lgi, self.gid
        # ----------------------------------------
        # Validations
        # ----------------------------------------
        rioshapes = rasterio.features.shapes
        with rasterio.Env():
            profile = rasterio.profiles.DefaultGTiffProfile()
            profile.update(width=lgi.shape[1], height=lgi.shape[0],
                           count=1, dtype=lgi.dtype,
                           transform=rasterio.transform.Affine.identity())
            # ----------------------------------------
            with rasterio.MemoryFile() as memfile:
                with memfile.open(**profile) as dataset:
                    dataset.write(lgi, 1)
                    self.raster_img_polygonisation_results = []
                    for _gid_ in gid:
                        mask = (lgi == _gid_).astype(np.uint8)
                        rs = list(rioshapes(mask, mask=mask,
                                            transform=dataset.transform))
                        self.raster_img_polygonisation_results.append(rs)

    def setup_gsmp_datastructure(self):
        self.gsmp = {'raw': None, 'smoothed': None}

    def make_gsmp(self, verbose=True):
        if verbose:
            print("Setting up grain structure multi-polygon data structure.")
        if not self.gsmp:
            self.setup_gsmp_datastructure()
        # --------------------------------
        if not self.holes_exist:
            self.gsmp['raw'] = MultiPolygon(self.allpol)
        else:
            print('Not implemented yet.')

    def make_polygonal_grains_raw(self, verbose=True, perform_xtal_dict_check=True):
        if verbose:
            print("Setting up raw polygon data structure for the grain structure.")
        for i, rpolrs in enumerate(self.raster_img_polygonisation_results,
                                   start=1):
            if len(rpolrs) == 0:
                print(40*'-', f"\n No geometry at index {i}.")
            elif len(rpolrs) == 1:
                geometry = rpolrs[0]
                # geometry[1] is the raster value and is not useful anymore.
                featcoords = geometry[0]['coordinates']
                nfeatures = len(featcoords)  # Number of features found
                self.polygons_raw_exteriors[i] = Polygon(featcoords[0])
                if nfeatures > 1:
                    if nfeatures == 2:
                        self.polygons_raw_holes[i] = Polygon(featcoords[1])
                    else:
                        self.polygons_raw_holes[i] = {j: Polygon(fc)
                                                      for j, fc in enumerate(featcoords[1:],
                                                                         start=1)}
            elif len(rpolrs) > 1:
                print(40*'-', f"\n More than one anticipated geometry at index {i}.")

        if perform_xtal_dict_check:
            VALUES = list(self.polygons_raw_exteriors.values())
            A = {i: v for i, v in self.polygons_raw_exteriors.items() if type(v)!=list}
            self.polygons_raw_exteriors = A

    @property
    def expol(self):
        return list(self.polygons_raw_exteriors.values())

    @property
    def hlpol(self):
        # Return all holes in the grain structure. Searches 3 levels deep.
        if not self.holes_exist:
            return []
        holes = []
        for hp1 in self.polygons_raw_holes.values():
            if isinstance(hp1, Polygon):
                # Level 1 hole feature. Hole inside a grain
                # Example: isand grain
                holes.append(hp1)
            elif isinstance(hp1, dict):
                # Level 2 hole feature
                # hole inside a hole which is inside a grain
                # Example: particle inside an island grain insie a grain.
                for hp2 in hp1.values():
                    if isinstance(hp2, Polygon):
                        holes.append(hp2)
                    elif isinstance(hp2, dict):
                        # Level 3 hole feature
                        # whatever could be the application!! Just having it.
                        for hp3 in hp2.values():
                            if isinstance(hp2, Polygon):
                                holes.append(hp3)
                            else:
                                # We will end here and not have anymore holes!
                                pass
        return holes

    @property
    def holes_exist(self):
        return len(self.polygons_raw_holes.keys()) == 0

    @property
    def allpol(self):
        if not self.holes_exist:
            return self.expol
        else:
            return self.expol + self.hlpol

    def set_polygons(self, verbose=True):
        """
        Set self.polygons list equal to all external polygon
        """
        if verbose:
            print("Setting up raw polygon data structure for the grain structure.")
        self.polygons = self.allpol

    def find_neighbors(self, verbose=True):
        """Calculates neighboring polygon IDs for a list of polygons.

        Args:
            polygons: A list of Shapely Polygon objects.

        Returns:
            A dictionary where keys are polygon IDs (0-based index) and values are lists of
            neighboring polygon IDs.
        """
        if verbose:
            print("Finding neighboring grains based on polygon intersections.")
        if self.gid.size == 1:
            self.neigh_gid = {int(self.gid[0]): [int(self.gid[0])]}
            return
        polygons, self.neigh_gid = self.allpol, {}
        for i, polygon in enumerate(polygons, start=1):
            self.neigh_gid[i] = []
            for j, other_polygon in enumerate(polygons, start=1):
                if i != j and polygon.touches(other_polygon):
                    self.neigh_gid[i].append(j)

    def val_neigh_gid(self, pixl_gs_neigh_gid):
        """
        Parameters
        ----------
        pixl_gs_neigh_gid: neigh_gid dict of pixellated grain structure (i.e. mcgs)
        Return
        ------
        bool, bool: True if validation, else False.
        """
        diffid, neighcount = [], []
        for n1, n2 in zip(pixl_gs_neigh_gid.values(), self.neigh_gid.values()):
            count_diff = abs(len(n1)-len(n2))
            neighcount.append(count_diff)
            if count_diff == 0:
                n1, n2 = np.sort(n1), np.sort(n2)
                diffid.append(np.argwhere(n1 != n2).squeeze().size)
            else:
                diffid.append(-1)
        return not(any(neighcount)), not(any(diffid))

    def make_neighpols(self):
        polygons = self.allpol
        self.neigh_pols = {pid: [polygons[i-1] for i in self.neigh_gid[pid]]
                           for pid in self.gid}

    def extract_gbsegments_raw(self):
        self.gbsegments_raw = {}
        gptl_fx = self.get_polygon_touch_lines
        polygons = self.allpol
        for pid in self.gid:
            self.gbsegments_raw[pid] = [gptl_fx(polygons[pid-1], npols)
                                        for npols in self.neigh_pols[pid]]

    def extract_gbsegments_mls_raw(self):
        MLS = MultiLineString
        self.gbsegments_mls = {}
        for pid in self.gid:
            self.gbsegments_mls[pid] = [MLS(gbsegln)
                                        for gbsegln in self.gbsegments_raw[pid]]

    def get_polygon_touch_lines(self, poly1, poly2):
        """Calculates the lines or points where two Shapely polygons touch.

        Args:
            poly1, poly2: The Shapely Polygon objects.

        Returns:
            A list of Shapely LineString objects (if the polygons share lines)
            or a list of Shapely Point objects (if they touch at a single point).
        """
        intersection = poly1.intersection(poly2)
        # -------------------------
        if intersection.is_empty:
            return []
        # -------------------------
        if isinstance(intersection, Point):
            # Single point contact
            return [intersection]
        elif isinstance(intersection, LineString):
            # Single line contact
            return [intersection]
        else:
            # MultiLineString or GeometryCollection
            # Multiple lines or points
            return list(intersection.geoms)

    def get_multilinestring_touch_points(mls1, mls2):
        """Calculates the point(s) where two Shapely MultiLineStrings touch.

        Args:
            mls1, mls2: The Shapely MultiLineString objects.

        Returns:
            A list of Shapely Point objects representing the touch points, or an empty list if there are no touch points.
        """
        touch_points = []
        for line1 in mls1.geoms:  # Iterate through linestrings in mls1
            for line2 in mls2.geoms:  # Iterate through linestrings in mls2
                intersection = line1.intersection(line2)  # Find intersection
                if isinstance(intersection, Point):
                    touch_points.append(intersection)

        return touch_points

    def area_gid(self, gid, gsrepr='raw'):
        """
        Return area of gid grain. Valid values for gsrepr are self.gsmp.keys().
        In anycase, the returned area will be geometric and not pixellated.

        Example
        -------
        self.area_gid(1, gsrepr='raw')
        """
        return self.gsmp[gsrepr].geoms[gid-1].area

    def plot_linestrings(linestrings, ax=None, color='blue', linewidth=1, **kwargs):
        """Plots a list of Shapely LineStrings and returns the axis.

        Args:
            linestrings: A list of LineString or MultiLineString objects.
            ax (optional): A Matplotlib Axes object to plot on. If None, a new figure and axis will be created.
            color (optional): Color of the lines (default: 'blue').
            linewidth (optional): Width of the lines (default: 1).
            **kwargs: Additional keyword arguments to pass to plt.plot().

        Returns:
            The Matplotlib Axes object on which the lines were plotted.
        """
        if ax is None:
            fig, ax = plt.subplots()  # Create a figure and axis if not provided

        for linestring in linestrings:
            if isinstance(linestring, MultiLineString):
                for geom in linestring.geoms:  # Plot each LineString in a MultiLineString
                    x, y = geom.xy
                    ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)
            else:
                x, y = linestring.xy
                ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)

        ax.set_aspect('equal')  # Ensure equal aspect ratio for accurate representation
        #plt.title('Plot of LineStrings')  # Optional title

        plt.show()  # Show the plot (optional)

        return ax  # Return the axis object

    def plot_gsmp(self, raw=True, overlay_on_lgi=False,
                  xoffset=0.5, yoffset=0.5, ax=None):
        """
        Plot multi-polygon form of the grain structure.

        Args:
            gsmp: A Shapely MultiPolygon object.
            grid_array (optional): The original NumPy array of grid values for background visualization.
        """
        if ax is None:
            fig, ax = plt.subplots()
        if overlay_on_lgi:
            ax.imshow(self.lgi, cmap='viridis', origin='lower')
        # -----------------------------------
        '''Access the grain structure multi-polygon object.'''
        if raw or self.gsmp['smooth_1'] is not None:
            GSMP = self.gsmp['raw']
        if not raw and isinstance(self.gsmp['smooth_1'], MultiPolygon):
            GSMP = self.gsmp['smooth_1']
        # -----------------------------------
        for i, polygon in enumerate(GSMP.geoms, start=1):
            x, y = polygon.exterior.xy
            x, y = np.array(list(x))-xoffset, np.array(list(y))-yoffset
            ax.plot(x, y, color='black', lw=1, ls='-', marker='.')
            pcx, pcy = polygon.centroid.coords.xy
            pcx, pcy = pcx[0], pcy[0]
            ax.plot(pcx, pcy, 'ko')
            ax.text(pcx, pcy, str(i), color='white', fontsize=12,
                    fontweight='bold')
        ax.set_aspect('equal')
        return ax

    def plot_grains_gids(self, gids, add_points=True, points=None, gclr='color', title="user grains",
                         cmap_name='viridis', plot_centroids=True,
                         add_gid_text=True, plot_gbseg=False,
                         bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                     'mec': 'black', 'ms': 2.5},
                         addpoints_kwargs={'marker': 'x', 'mfc': 'black',
                                     'mec': 'black', 'ms': 5}
                         ):
        """
        Parameters
        ----------
        gids : int/list
            Either a single grain index number or list of them
        title : TYPE, optional
            DESCRIPTION. The default is "user grains".
        gclr :

        Returns
        -------
        None.

        Example-1
        ---------
            After acquiring gids for aspect_ratio between ranks 80 and 100,
            we will visualize those grains.
            . . . . . . . . . . . . . . . . . . . . . . . . . .
            As we are only interested in gid, we will not use the other
            two values returned by PXGS.gs[n].get_gid_prop_range() method:

            gid, _, __ = PXGS.gs[8].get_gid_prop_range(PROP_NAME='aspect_ratio',
                                                       range_type='rank',
                                                       rank_range=[80, 100]
                                                       )
            . . . . . . . . . . . . . . . . . . . . . . . . . .
            Now, pass gid as input for the PXGS.gs[n].plot_grains_gids(),
            which will then plot the grain strucure with only these values:

            PXGS.gs[8].plot_grains_gids(gid, cmap_name='CMRmap_r')
        """
        # Validations
        if not dth.IS_ITER(gids):
            gids = [gids]
        # -------------------------------
        # Validtions
        # -------------------------------
        if gclr not in ('binary', 'grayscale'):
            lgi_masked, masker = self.mask_lgi_with_gids(gids)
            fig, ax = plt.subplots(1, figsize=(5, 5), dpi=120)
            im = ax.imshow(lgi_masked, cmap=cmap_name, vmin=1)
        # -------------------------------
        if gclr in ('binary', 'grayscale'):
            lgi_masked, masker = self.mask_lgi_with_gids(gids, masker=-10)
            lgi_masked[lgi_masked != 0] = 1
            fig, ax = plt.subplots(1, figsize=(5, 5), dpi=120)
            im = ax.imshow(lgi_masked, cmap='gray_r', vmin=0, vmax=1)
        # -------------------------------
        fig.colorbar(im, ax=ax)
        # -------------------------------
        if plot_centroids:
            self.plot_grain_centroids(gids, ax, add_gid_text=add_gid_text)
        # -------------------------------
        if plot_gbseg:
            self.plot_contour_grains_gids(gids,
                                          simple_all_preference='simple',
                                          new_fig=False, ax=ax,
                                          bjp_kwargs={'marker': bjp_kwargs['marker'],
                                                      'mfc': bjp_kwargs['mfc'],
                                                      'mec': bjp_kwargs['mec'],
                                                      'ms': bjp_kwargs['ms']}
                                          )
        if add_points:
            ax.plot(points[:, 0], points[:, 1],
                    marker=addpoints_kwargs['marker'],
                    mfc=addpoints_kwargs['mfc'],
                    mec=addpoints_kwargs['mec'],
                    ms=addpoints_kwargs['ms'])
        # -------------------------------
        ax.set_title(title)
        ax.set_xlabel(r"X-axis, $\mu m$", fontsize=12)
        ax.set_ylabel(r"Y-axis, $\mu m$", fontsize=12)
        return ax

    def set_minimum_nnodes_per_gbseg(self):
        pass

    def subdivide_gbsegments(self, method=1):
        """
        """
        pass

    def smooth_moving_avg(self, n=3):
        pass

    def smooth_polynomial(self):
        pass

    def extract_gids_for_smoothing(self, verbose=True):
        pass

    def heal_edges(self, verbose=True):
        pass

    def plot_gbseg(self, gid, segid=[]):
        pass

    def link_polygons(self):
        pass

    def heal_polygons(self):
        pass

    def write_abq_script_data(self):
        pass

    def set_mesh_properties(self):
        pass

    def mesh(self):
        pass

    def assess_mesh_quality(self):
        pass

    def export_mesh(self):
        pass

    def set_grain_centroids_raw(self, verbose=True):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        if verbose:
            print("Extracting grain boundary segments based on polygon intersections.")
        centroids_raw = []
        for gid in self.gid:
            centroid = list(np.argwhere(self.lgi == gid).mean(axis=0)-0.5)
            #coordinates = self.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0][:-1]
            #centroids_raw.append(list(np.array(coordinates).mean(axis=0)))
            centroids_raw.append(centroid)
        self.centroids_raw = np.array(centroids_raw)


    def set_grain_centroids(self, verbose=True):
        """
        Parameters ---------- Returns ------- Example ------- Explanations
        """
        # Validations
        self.centroids = []
        # To do

    def set_polygonization_xyoffset(self, xyoffset, verbose=True):
        # Validations
        if verbose:
            print(f"Setting up polygonization xy-offset to {xyoffset}.")
        self.xyoffset = xyoffset

    def pix_to_geom(self, polygonisation_offset=0.5, verbose=True, perform_xtal_dict_check=True):
        if verbose:
            print(40*'-', "\n", "Starting geometrification of the grain structure.")
        if verbose:
            print("\n \n ------ > Phase-1 <----- \n \n")
        self.set_up_quality_measures(verbose=verbose)
        self.polygonize(verbose=verbose)
        self.set_polygonization_xyoffset(polygonisation_offset,
                                    verbose=verbose)
        self.make_polygonal_grains_raw(verbose=verbose, perform_xtal_dict_check=perform_xtal_dict_check)
        self.set_polygons(verbose=verbose)
        self.make_gsmp(verbose=verbose)
        self.find_neighbors(verbose=verbose)
        self.set_grain_centroids_raw(verbose=verbose)
        self.set_grain_loc_ids(verbose=verbose)
        self.get_junction_points_from_grain_intersections(verbose=verbose)
        self.extract_GBP(verbose=verbose)
        self.find_GBP_at_boundary(verbose=verbose)
        self.update_JNP_from_GBP_at_boundary(verbose=verbose)

        if verbose:
            print("\n \n ------ > Phase-2 <----- \n \n")
        self.build_jnp_objects(verbose=verbose)
        self.build_all_gbp_objects(verbose=verbose)
        self.get_gbp_grain_wise_coords(verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-3 <----- \n \n")
        self.build_jnp_grain_wise_indices(verbose=verbose)
        self.build_gbp_grain_wise_indices_geometric(verbose=verbose)
        self.build_gbp_grain_wise_indices_coordbased(verbose=verbose)
        self.build_gbp_grain_wise_indices_pointsbased(verbose=verbose)
        self.build_gbmullines_grain_wise(verbose=verbose)
        self.build_sorted_jnp_objects(plot=False, verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-4 <----- \n \n")
        self.align_gbmullines_start_to_jnp_start(plot_bf=False, 
                                    plot_af=False,  verbose=verbose)
        self.splice_grain_boundary_segments_at_junction_points(verbose=verbose)
        self.find_quality_of_grain_boundary_segmentation(verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-5 <----- \n \n")
        self.create_neigh_gid_pair_ids(self.neigh_gid, verbose=verbose)
        self.setup_neigh_connectivity_flags_DS(verbose=verbose)
        self.set_neigh_connectivity_flags_DS(centroid_eq_EPS=1E-8, 
                                    verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-6 <----- \n \n")
        self.gather_grain_boundary_segments_of_all_pairs(verbose=verbose)
        self.consolidate_gbsegments(squeeze_segment_data_structure=True, 
                                    verbose=verbose)
        # problematic_grains = self.get_problematic_grains()
        if verbose:
            print("\n \n ------ > Phase-7 <----- \n \n")
        self.update_consolidated_segments_with_boundary_grain_gids(plot=False, 
                                    verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-8 <----- \n \n")
        self.sort_gbsegments_by_original_order(verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-9 <----- \n \n")
        self.flip_segments_to_reorder_GBS(plot_each_grain_details=False, 
                                    verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-10 <----- \n \n")
        self.update_segflip_requirements(verbose=verbose)
        self.calculate_grain_boundary_coordinates_after_gbseg_reordering(verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-11 <----- \n \n")
        self.AssembleGBSEGS(self.GB, saa=True, throw=False, 
                                    verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-12 <----- \n \n")
        self.construct_geometric_polyxtal_from_gbcoords(self.GBCoords, dtype='shapely',
                                    saa=True, throw=False, verbose=verbose)
        if verbose:
            print("\n \n ------ > Phase-13 <----- \n \n")
        self.set_pure_gbpoints(verbose=verbose)
        if verbose:
            print(40*'-', "\n", "Completed geometrification of the grain structure.")

    def get_bounds_from_grain_boundary_points(self):
        """
        Get x and y bounds from grain boudnary points.

        Parameters
        ----------
        xyoffset: value to subtract from polygonization end coordinates in order to
            align with MCGS 2d regults.

        Explanation
        -----------
        Works on grain boudnary points obtained from poliygonization of the raster
        image.

        Use
        ---
        xmin, xmax, ymin, ymax = self.get_bounds_from_grain_boundary_points()
        """
        for gid in self.gid:
            _ripolres_ = self.raster_img_polygonisation_results[gid-1][0][0]
            if gid == 1:
                gbpoints = np.array(_ripolres_['coordinates'][0][:-1])-self.xyoffset
            else:
                new_gbpoints = np.array(_ripolres_['coordinates'][0][:-1])-self.xyoffset
                gbpoints = np.array(list(gbpoints) + list(new_gbpoints))
        gbpoints = np.unique(gbpoints, axis=1)
        xmin, xmax = gbpoints[:, 0].min(), gbpoints[:, 0].max()
        ymin, ymax = gbpoints[:, 1].min(), gbpoints[:, 1].max()
        return xmin, xmax, ymin, ymax

    def set_grain_loc_ids(self, verbose=True):
        """
        Identify the location of grains and place their IDs in self.grain_loc_ids
        dictionary.
        """
        if verbose:
            print("Setting up grain location IDs based on grain centroids.")
        if self.gid.size == 1:
            self.grain_loc_ids = {'internal': [int(self.gid[0])],
                              'boundary': [int(self.gid[0])],
                              'left': [int(self.gid[0])],
                              'bottom': [int(self.gid[0])],
                              'right': [int(self.gid[0])],
                              'top': [int(self.gid[0])],
                              'pure_left': [int(self.gid[0])],
                              'pure_bottom': [int(self.gid[0])],
                              'pure_right': [int(self.gid[0])],
                              'pure_top': [int(self.gid[0])],
                              'corner': [int(self.gid[0])],
                              'bottom_left_corner': [int(self.gid[0])],
                              'bottom_right_corner': [int(self.gid[0])],
                              'top_right_corner': [int(self.gid[0])],
                              'top_left_corner': [int(self.gid[0])],
                              }
            return
        # ----------------------------------
        xmin, xmax, ymin, ymax = self.get_bounds_from_grain_boundary_points()
        # ----------------------------------
        '''LABEL GRAINS AS PER THEIR LOCATION IN THR GRAIN STRUCTUER'''
        border_grain_flags = [False for gid in self.gid]
        internal_grain_flags = [False for gid in self.gid]
        corner_grain_flags = [False for gid in self.gid]
        # ----------------------------------
        bl_grain_flags = [False for gid in self.gid]
        tl_grain_flags = [False for gid in self.gid]
        br_grain_flags = [False for gid in self.gid]
        tr_grain_flags = [False for gid in self.gid]
        # ----------------------------------
        for gid in self.gid:
            _ripolres_ = self.raster_img_polygonisation_results[gid-1][0][0]
            coords = np.array(_ripolres_['coordinates'][0])-self.xyoffset
            c1 = xmin in coords[:, 0]
            c2 = xmax in coords[:, 0]
            c3 = ymin in coords[:, 1]
            c4 = ymax in coords[:, 1]
            if any((c1, c2, c3, c4)):
                border_grain_flags[gid-1] = True
                if any((coords[:, 0] == xmin) & (coords[:, 1] == ymin)):
                    corner_grain_flags[gid-1] = True
                    bl_grain_flags[gid-1] = True  # Bottom left
                elif any((coords[:, 0] == xmin) & (coords[:, 1] == ymax)):
                    corner_grain_flags[gid-1] = True
                    tl_grain_flags[gid-1] = True  # Top left
                elif any((coords[:, 0] == xmax) & (coords[:, 1] == ymin)):
                    corner_grain_flags[gid-1] = True
                    br_grain_flags[gid-1] = True  # Bottom right
                elif any((coords[:, 0] == xmax) & (coords[:, 1] == ymax)):
                    corner_grain_flags[gid-1] = True
                    tr_grain_flags[gid-1] = True  # Top right
            else:
                internal_grain_flags[gid-1] = True
        # ----------------------------------
        _border_grain_gids_ = np.argwhere(border_grain_flags).squeeze()+1
        _internal_grain_gids_ = np.argwhere(internal_grain_flags).squeeze()+1
        _corner_grain_gids_ = np.argwhere(corner_grain_flags).squeeze()+1

        if isinstance(_border_grain_gids_, np.integer):
            border_grain_gids = [int(_border_grain_gids_)]
        else:
            border_grain_gids = _border_grain_gids_.tolist()

        if isinstance(_internal_grain_gids_, np.integer):
            internal_grain_gids = [int(_internal_grain_gids_)]
        else:
            internal_grain_gids = _internal_grain_gids_.tolist()

        if isinstance(_corner_grain_gids_, np.integer):
            corner_grain_gids = [int(_corner_grain_gids_)]
        else:
            corner_grain_gids = _corner_grain_gids_.tolist()
        # ----------------------------------
        _bl_grain_gids_ = np.argwhere(bl_grain_flags).squeeze()+1
        _tl_grain_gids_ = np.argwhere(tl_grain_flags).squeeze()+1
        _br_grain_gids_ = np.argwhere(br_grain_flags).squeeze()+1
        _tr_grain_gids_ = np.argwhere(tr_grain_flags).squeeze()+1

        if isinstance(_bl_grain_gids_, np.integer):
            bl_grain_gids = [int(_bl_grain_gids_)]
        else:
            bl_grain_gids = _bl_grain_gids_.tolist()

        if isinstance(_tl_grain_gids_, np.integer):
            tl_grain_gids = [int(_tl_grain_gids_)]
        else:
            tl_grain_gids = _tl_grain_gids_.tolist()

        if isinstance(_br_grain_gids_, np.integer):
            br_grain_gids = [int(_br_grain_gids_)]
        else:
            br_grain_gids = _br_grain_gids_.tolist()

        if isinstance(_tr_grain_gids_, np.integer):
            tr_grain_gids = [int(_tr_grain_gids_)]
        else:
            tr_grain_gids = _tr_grain_gids_.tolist()
        # ----------------------------------
        # Identify grains which are pure edge grains. These grains are not on corner.
        l_grain_flags = [False for gid in self.gid]
        r_grain_flags = [False for gid in self.gid]
        b_grain_flags = [False for gid in self.gid]
        t_grain_flags = [False for gid in self.gid]
        # ----------------------------------
        for gid in self.gid:
            _ripolres_ = self.raster_img_polygonisation_results[gid-1][0][0]
            coords = np.array(_ripolres_['coordinates'][0])-self.xyoffset
            if any(coords[:, 0] == xmin):
                   l_grain_flags[gid-1] = True
            if any(coords[:, 0] == xmax):
                   r_grain_flags[gid-1] = True
            if any(coords[:, 1] == ymin):
                   b_grain_flags[gid-1] = True
            if any(coords[:, 1] == ymax):
                   t_grain_flags[gid-1] = True
        # ----------------------------------
        _l_grain_gids_ = np.argwhere(l_grain_flags).squeeze()+1
        _r_grain_gids_ = np.argwhere(r_grain_flags).squeeze()+1
        _b_grain_gids_ = np.argwhere(b_grain_flags).squeeze()+1
        _t_grain_gids_ = np.argwhere(t_grain_flags).squeeze()+1

        if isinstance(_l_grain_gids_, np.integer):
            l_grain_gids = [int(_l_grain_gids_)]
        else:
            l_grain_gids = _l_grain_gids_.tolist()

        if isinstance(_r_grain_gids_, np.integer):
            r_grain_gids = [int(_r_grain_gids_)]
        else:
            r_grain_gids = _r_grain_gids_.tolist()

        if isinstance(_b_grain_gids_, np.integer):
            b_grain_gids = [int(_b_grain_gids_)]
        else:
            b_grain_gids = _b_grain_gids_.tolist()

        if isinstance(_t_grain_gids_, np.integer):
            t_grain_gids = [int(_t_grain_gids_)]
        else:
            t_grain_gids = _t_grain_gids_.tolist()
        # ----------------------------------
        pl_grain_gids = list(set(l_grain_gids) - set(bl_grain_gids) - set(tl_grain_gids))
        pr_grain_gids = list(set(r_grain_gids) - set(br_grain_gids) - set(tr_grain_gids))
        pb_grain_gids = list(set(b_grain_gids) - set(bl_grain_gids) - set(br_grain_gids))
        pt_grain_gids = list(set(t_grain_gids) - set(tl_grain_gids) - set(tr_grain_gids))
        # ----------------------------------
        self.grain_loc_ids = {'internal': internal_grain_gids,
                              'boundary': border_grain_gids,
                              'left': l_grain_gids,
                              'bottom': b_grain_gids,
                              'right': r_grain_gids,
                              'top': t_grain_gids,
                              'pure_left': pl_grain_gids,
                              'pure_bottom': pb_grain_gids,
                              'pure_right': pr_grain_gids,
                              'pure_top': pt_grain_gids,
                              'corner': corner_grain_gids,
                              'bottom_left_corner': bl_grain_gids,
                              'bottom_right_corner': br_grain_gids,
                              'top_right_corner': tr_grain_gids,
                              'top_left_corner': tl_grain_gids,
                              }

    def get_junction_points_from_grain_intersections(self, verbose=True):
        if verbose:
            print("Extracting grain boundary segments and junction points")
        self.JNP = []
        for gid1 in self.gid:
            gid1_xy = []
            if gid1 % 50 == 0:
                _s_ = "Extracting Junction points"
                print(f'{_s_} {np.round(gid1*100/self.n, 2)} % complete.')
            for gid2 in self.gid:
                if gid2 <= gid1:
                    continue
                # print(gid1, gid2)
                self.polygons[gid1-1]
                self.polygons[gid2-1]
                intersec = self.polygons[gid1-1].intersection(self.polygons[gid2-1])
                # intersec = self.polygons[gid1-1].boundary.intersection(self.polygons[gid2-1].boundary)
                # print(type(intersec))
                # print(isinstance(intersec, MultiLineString))
                # print(isinstance(intersec, LineString))
                if isinstance(intersec, LineString):
                    # print(intersec.coords.self.JNP)
                    self.JNP.append([intersec.coords.xy[0][0], intersec.coords.xy[1][0]])
                    pass
                elif isinstance(intersec, MultiLineString):
                    # print(dir(intersec))
                    # print(intersec.boundary)#.coords.xy)
                    if isinstance(intersec.boundary, MultiPoint):
                        for pnt in intersec.boundary.geoms:
                            self.JNP.append([pnt.x, pnt.y])
                    if isinstance(intersec.boundary, tuple):
                        self.JNP.append([intersec.boundary[0][0], intersec.boundary[1][0]])
                        self.JNP.append([intersec.boundary[0][1], intersec.boundary[1][1]])
                    #for line in intersec.geoms:
                    #    self.JNP.append([line.coords.self.JNP[0][0], line.coords.self.JNP[1][0]])
                    #    self.JNP.append([line.coords.self.JNP[0][1], line.coords.self.JNP[1][1]])
                elif isinstance(intersec, Point):
                    self.JNP.append([intersec.x, intersec.y])
        self.JNP = np.unique(self.JNP, axis = 0)-self.xyoffset

    def extract_GBP(self, verbose=True):
        """
        Extarct grain boundary points data fdrom polygonization coord result.
        """
        if verbose:
            print("Extracting grain boundary segments and junction points at the boundary of the grain structure.")
        postpolcoords = self.raster_img_polygonisation_results
        self.GBP = []
        for gid in self.gid:
            gbp = np.array(postpolcoords[gid-1][0][0]['coordinates'][0][:-1])
            self.GBP.extend(gbp.tolist())
        self.GBP = np.array(self.GBP) - self.xyoffset

    def set_up_quality_measures(self, verbose=True):
        """
        Initiate the quality dictionary to hold results on operation quality.
        """
        if verbose:
            print("Setting up initial data structures and data")
        self.quality = {'gb_segmentation': [],
                        'GBS_reordering_success': None, }

    def get_bounds_from_GBP(self):
        """Calculate the bounds of the po;lygonized grain structure."""
        xmin = self.GBP[:, 0].min()
        xmax = self.GBP[:, 0].max()
        ymin = self.GBP[:, 1].min()
        ymax = self.GBP[:, 1].max()
        return xmin, xmax, ymin, ymax

    def find_GBP_at_boundary(self, verbose=True):
        """
        Indeitify grain boundary points which are on the grain.
        """
        if verbose:
            print("Finding grain boundary points at the boundary of the grain structure.")
        xmin, xmax, ymin, ymax = self.get_bounds_from_GBP()
        GBPx, GBPy = self.GBP[:, 0], self.GBP[:, 1]
        GBP_left, GBP_right = self.GBP[GBPx == xmin], self.GBP[GBPx == xmax]
        GBP_bot, GBP_top = self.GBP[GBPy == ymin], self.GBP[GBPy == ymax]
        self.GBP_at_boundary = np.unique(np.vstack((GBP_left, GBP_bot,
                                               GBP_right, GBP_top)), axis=0)
        # End
        ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
        # Add missing points to the junction points.
        if self.gid.size > 1:
            GBP_at_boundary_flag_to_remove = [False for _ in self.GBP_at_boundary]
            for i, gbpboundary in enumerate(self.GBP_at_boundary):
                x, y = gbpboundary
                if not any((self.JNP[:, 0] == x) & (self.JNP[:, 1] == y)):
                    GBP_at_boundary_flag_to_remove[i] = True
            self.GBP_at_boundary[GBP_at_boundary_flag_to_remove]

    def update_JNP_from_GBP_at_boundary(self, verbose=True):
        """
        Update junctionm points with the grain boundary points which are on the
        boundaries of the poly-xtal.
        """
        if verbose:
            print("Updating junction point objects based on grain boundary segment coordinates.")
        if self.JNP.size == 0:
            self.JNP = np.unique(self.GBP_at_boundary, axis=0)
            return
        self.JNP = np.append(self.JNP, self.GBP_at_boundary, axis=0)
        self.JNP = np.unique(self.JNP, axis=0)

    def build_jnp_objects(self, verbose=True):
        """
        Build coordinates, UPXO Point2d objects and shapely point objects of all
        junction points.
        """
        if verbose:
            print("Building grain boundary point objects based on grain boundary segment coordinates.")
        self.jnp_all_coords = self.JNP
        self.jnp_all_upxo = np.array([Point2d(jnp[0], jnp[1]) for jnp in self.JNP])
        self.jnp_all_shapely = [ShPoint2d(jnp[0], jnp[1]) for jnp in self.JNP]
        # jnp_all_upxo_mp = MPoint2d.from_xy(jnp_all_coords.T)
        # jnp_all_upxo_mp = MPoint2d.from_upxo_points2d(jnp_all_upxo, zloc=0.0)

    def build_all_gbp_objects(self, verbose=True):
        """Build coordinates of all grain boundary points. Global."""
        if verbose:
            print("Building grain boundary point objects based on grain boundary segment coordinates.")
        self.gbp_all_coords = []
        postpolcoords = self.raster_img_polygonisation_results
        # len(gbp_all_coords)
        for gid in self.gid:
            postpolcoords_gid = postpolcoords[gid-1][0][0]['coordinates']
            _ = np.array(postpolcoords_gid[0][:-1]) - self.xyoffset
            self.gbp_all_coords.extend(_.tolist())
        self.gbp_all_coords = np.unique(self.gbp_all_coords, axis=0)
        """Build UPXO points from gbp_all_coords data """
        self.gbp_all_upxo = np.array([Point2d(gbp[0], gbp[1])
                                      for gbp in self.gbp_all_coords])
        self.gbp_all_shapely = [ShPoint2d(gbp[0], gbp[1])
                                for gbp in self.gbp_all_coords]
        # gbp_all_upxo_mp = MPoint2d.from_upxo_points2d(gbp_all_upxo, zloc=0.0)

    def get_gbp_grain_wise_coords(self, verbose=True):
        """Build coordinates of all points of grian boundary points, grain wese."""
        if verbose:
            print("Building grain boundary point objects based on grain-wise coordinates.")
        self.gbp_grain_wise_coords = {}
        postpolcoords = self.raster_img_polygonisation_results
        for gid in self.gid:
            _ = np.array(postpolcoords[gid-1][0][0]['coordinates'][0][:-1]) - self.xyoffset
            self.gbp_grain_wise_coords[gid] = _

    def build_jnp_grain_wise_indices(self, verbose=True):
        """
        Build indices of jnp for every grain. Indices will be from jnp_all_coords.
        These indices relate to:
            * jnp_all_coords, jnp_all_upxo, jnp_all_shapely
            * jnp_all_upxo_mp.points, jnp_all_upxo_mp.coortds

        Use
        ---
        Data access:
            jnp_all_coords[jnp_grain_wise_indices[gid]]
            jnp_all_upxo[jnp_grain_wise_indices[gid]]
            jnp_all_upxo_mp.points[jnp_grain_wise_indices[gid]]

        Verification
        ------------
        gid = 10
        plt.imshow(geom.lgi)
        coord = jnp_all_coords[jnp_grain_wise_indices[gid]]
        plt.plot(coord[:, 0], coord[:, 1], 'ko')

        Note
        ----
        mid1 = id(jnp_all_upxo[jnp_grain_wise_indices[gid]][0])
        mid2 = id(jnp_all_upxo_mp.points[jnp_grain_wise_indices[gid]][0])
        mid1 == mid2

        Note@dev
        --------
        gid = 1
        jnp_all_coords[jnp_grain_wise_indices[gid]]
        jnp_all_upxo[jnp_grain_wise_indices[gid]]
        gbp_grain_wise_coords[gid]

        Build indices of gbp for every grain. Indices will be from gbp_all_coords.
        These indices relate to:
            * gbp_all_coords, gbp_all_upxo, gbp_all_shapely
            * gbp_all_upxo_mp.points, gbp_all_upxo_mp.coortds

        id(gbp_all_upxo[0])
        id(gbp_all_upxo_mp.points[0])
        """
        if verbose:
            print("Building grain boundary point objects based on grain-wise coordinates.")
        self.jnp_grain_wise_indices = {}
        for gid in self.gid:
            self.jnp_grain_wise_indices[gid] = []
            pol_shapely = ShAff.translate(self.polygons[gid-1],
                                          xoff=-self.xyoffset,
                                          yoff=-self.xyoffset)
            for i, jnp_shapely in enumerate(self.jnp_all_shapely, start=0):
                if pol_shapely.touches(jnp_shapely):
                    self.jnp_grain_wise_indices[gid].append(i)

    def build_gbp_grain_wise_indices_geometric(self, verbose=True):
        """
        This is to correct inconsistencies in build_jnp_grain_wise_indices method.
        There are cases which lead to mulline constityutent lines intersecting each other. 
        This step is crucial to avoid it.
        """
        if verbose:
            print("Building grain boundary point objects based on grain-wise coordinates.")
        self.gbp_grain_wise_indices = {}
        for gid in self.gid:
            self.gbp_grain_wise_indices[gid] = []
            pol_shapely = ShAff.translate(self.polygons[gid-1],
                                          xoff=-self.xyoffset,
                                          yoff=-self.xyoffset)
            for i, gbp_shapely in enumerate(self.gbp_all_shapely, start=0):
                if pol_shapely.touches(gbp_shapely):
                    self.gbp_grain_wise_indices[gid].append(i)

    def build_gbp_grain_wise_indices_coordbased(self, verbose=True):
        if verbose:
            print("Building grain boundary point objects based on grain-wise indices and geometric coordinates.")
        findloc = DO.find_coorda_loc_in_coords_arrayb
        self.gbp_grain_wise_indices = {gid: None for gid in self.gid}
        for gid in self.gid:
            plist = self.gbp_grain_wise_coords[gid]
            self.gbp_grain_wise_indices[gid] = [findloc(p, self.gbp_all_coords) for p in plist]

    def build_gbp_grain_wise_indices_pointsbased(self, verbose=True):
        if verbose:
            print("Building grain boundary point objects based on grain-wise indices.")
        self.gbp_grain_wise_points = {gid: None for gid in self.gid}
        for gid in self.gid:
            plist = self.gbp_grain_wise_coords[gid]
            self.gbp_grain_wise_points[gid] = self.gbp_all_upxo[self.gbp_grain_wise_indices[gid]]

    def build_gbmullines_grain_wise(self, verbose=True):
        '''We will start by first creating UPXO line objects for grain
        boundaries.'''
        if verbose:
            print("Building grain boundary multi-linestrings")
        # Build lines from gbp_grain_wise_coordinates.
        self.gbmullines_grain_wise = {gid: [] for gid in self.gid}
        for gid in self.gid:
            nodes = self.gbp_grain_wise_points[gid].tolist()
            self.gbmullines_grain_wise[gid] = MSline2d.by_nodes(nodes, close=False)
            # ax = gbmullines_grain_wise[gid].plot()

        for gid in self.gid:
            self.gbmullines_grain_wise[gid].close(reclose=False)

        for gid in self.gid:
            # Build the jnp coordinates which are not there in gbp array.
            jnp_indices_toinsert = []
            for i, jnp in enumerate(self.jnp_all_coords[self.jnp_grain_wise_indices[gid]]):
                if not DO.is_a_in_b(jnp, self.gbp_grain_wise_coords[gid]):
                    '''This means jnp should be inserted into
                    self.gbp_grain_wise_coords[gid] at the right place.'''
                    jnp_indices_toinsert.append(i)
            jnps_to_insert = self.jnp_all_upxo[self.jnp_grain_wise_indices[gid]][jnp_indices_toinsert]
            # --------------------------
            #if len(jnps_to_insert) > 0:
            self.gbmullines_grain_wise[gid].add_nodes(jnps_to_insert)

    def arrange_junction_point_coords_new(self, gbcoords_thisgrain, junction_points_coord):
        # Create a dictionary to map coordinates to their indices in gbcoords_thisgrain
        coord_index_map = {tuple(coord): idx for idx, coord in enumerate(gbcoords_thisgrain)}
        # Generate a list of indices for sorting
        sorted_indices = sorted(range(len(junction_points_coord)), key=lambda i: coord_index_map[tuple(junction_points_coord[i])])
        # Sort junction_points_coord based on the sorted indices
        sorted_junction_points = np.array([junction_points_coord[i] for i in sorted_indices])
        return sorted_junction_points, sorted_indices

    def build_sorted_jnp_objects(self, plot=False, verbose=True):
        if verbose:
            print("Building sorted junction point objects")
        # find_coord_loc = DO.find_coorda_loc_in_coords_arrayb
        self.jnp_all_sorted_coords = {gid: None for gid in self.gid}
        self.jnp_all_sorted_upxo = {gid: None for gid in self.gid}
        for gid in self.gid:
            # gbpoints_thisgrain = self.gbmullines_grain_wise[gid].nodes
            gbcoords_thisgrain = self.gbmullines_grain_wise[gid].get_node_coords()
            junction_points_upxo, junction_points_coord = [], []
            for i, jnpcoord in enumerate(self.jnp_all_coords):
                if DO.is_a_in_b(jnpcoord, gbcoords_thisgrain):
                    junction_points_coord.append(jnpcoord)
                    junction_points_upxo.append(self.jnp_all_upxo[i])
            # junction_points_coord = arrange_junction_point_coords(gbcoords_thisgrain, junction_points_coord)
            # self.arrange_junction_point_coords_new(self.gbcoords_thisgrain, self.junction_points_coord)
            junction_points_coord, _ = self.arrange_junction_point_coords_new(gbcoords_thisgrain, junction_points_coord)
            junction_points_upxo = list(np.array(junction_points_upxo)[_])

            junction_points_coord, sorted_indices = self.arrange_junction_point_coords_new(gbcoords_thisgrain,
                                                                                           junction_points_coord)
            junction_points_upxo = list(np.array(junction_points_upxo)[sorted_indices])
            # junction_points_upxo, _ = arrange_junction_points_upxo(gbpoints_thisgrain, junction_points_upxo)
            self.jnp_all_sorted_coords[gid] = junction_points_coord
            self.jnp_all_sorted_upxo[gid] = junction_points_upxo

        if plot:
            for gid in self.gid:
                plt.figure()
                coord = self.gbmullines_grain_wise[gid].get_node_coords()
                plt.plot(coord[:, 0], coord[:, 1], '-k.')
                for i, c in enumerate(coord[:-1,:], 0):
                    plt.text(c[0]+0.15, c[1]+0.15, i)

                jnp = self.jnp_all_sorted_coords[gid]
                plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)
                for i, j in enumerate(jnp):
                    plt.text(j[0]+0.15, j[1], i, color='red')

    def align_gbmullines_start_to_jnp_start(self, plot_bf=False, plot_af=False, verbose=True):
        if verbose:
            print("Aligning grain boundary multi-linestrings with junction points.")
        if plot_bf:
            for gid in self.gid:
                plt.figure()
                coord = self.gbmullines_grain_wise[gid].get_node_coords()
                plt.plot(coord[:, 0], coord[:, 1], '-k.')
                for i, c in enumerate(coord[:-1,:], 0):
                    plt.text(c[0]+0.15, c[1]+0.15, i)

                jnp = self.jnp_all_sorted_coords[gid]
                plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)
                for i, j in enumerate(jnp):
                    plt.text(j[0]+0.15, j[1], i, color='red')
        # ---------------------------------------------------
        for gid in self.gid:
            roll_distance = DO.find_coorda_loc_in_coords_arrayb(self.jnp_all_sorted_coords[gid][0],
                                                                self.gbmullines_grain_wise[gid].get_node_coords())
            self.gbmullines_grain_wise[gid].roll(roll_distance)
        # ---------------------------------------------------
        if plot_af:
            for gid in self.gid:
                plt.figure()
                coord = self.gbmullines_grain_wise[gid].get_node_coords()
                plt.plot(coord[:, 0], coord[:, 1], '-k.')
                for i, c in enumerate(coord[:-1,:], 0):
                    plt.text(c[0]+0.15, c[1]+0.15, i)

                jnp = self.jnp_all_sorted_coords[gid]
                plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)
                for i, j in enumerate(jnp):
                    plt.text(j[0]+0.15, j[1], i, color='red')

    def splice_grain_boundary_segments_at_junction_points(self, plot=False, verbose=True):
        """
        Splice the grain boundary into grain boundary segments using jnp point data
        """
        if verbose:
            print("Finding junction points at grain boundary segment intersections and splicing grain boundary segments at junction points.")
        self.gbsegments = {gid: [] for gid in self.gid}
        for gid in self.gid:
            if len(self.jnp_all_sorted_upxo[gid]) == 1:
                segment = self.gbmullines_grain_wise[gid].lines
            elif len(self.jnp_all_sorted_upxo[gid]) > 1:
                choplocs = []
                for point in self.jnp_all_sorted_upxo[gid]:
                    location = point.eq_fast(self.gbmullines_grain_wise[gid].nodes[:-1], point_spec=2)
                    choplocs.append(np.argwhere(location).squeeze().tolist())

                if choplocs[0] != 0:
                    choplocs = [0] + choplocs
                if choplocs[-1] == len(self.gbmullines_grain_wise[gid].lines):
                    choplocs = choplocs[:-1]

                ranges = []
                for i in range(1, len(choplocs)):
                    ranges.append([choplocs[i-1], choplocs[i]])

                for r in ranges:
                    lines = self.gbmullines_grain_wise[gid].lines[r[0]:r[1]]
                    self.gbsegments[gid].append(MSline2d.from_lines(lines, close=False))
                rem_lines = self.gbmullines_grain_wise[gid].lines[r[1]:len(self.gbmullines_grain_wise[gid].lines)]
                self.gbsegments[gid].append(MSline2d.from_lines(rem_lines, close=False))

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(self.lgi)
            for gid in self.gid:
                for gbseg in self.gbsegments[gid]:
                    coords = gbseg.get_node_coords()
                    ax.plot(coords[:, 0], coords[:, 1], '-o', ms=5)

                ax.plot(self.jnp_all_sorted_coords[gid][:, 0],
                        self.jnp_all_sorted_coords[gid][:, 1],
                        'k*', ms = 7)
                centroid = self.gbmullines_grain_wise[gid].get_node_coords()[:-1].mean(axis=0)
                plt.text(centroid[0], centroid[1], gid, color='white', fontsize=12)

    def find_quality_of_grain_boundary_segmentation(self, verbose=True):
        """ Now, we will collect all grain boundary segments. """
        if verbose:
            print("Building sorted grain boundary segment objects after splicing at junction points.")
        GBSEG = []
        gbseg_map_indices = {gid: [] for gid in self.gid}
        i = 0
        for gid in self.gid:
            for gbseg in self.gbsegments[gid]:
                GBSEG.append(gbseg)
                gbseg_map_indices[gid].append(i)
                i += 1

        quality = []
        for gid in self.gid:
            quality.append(int(len(gbseg_map_indices[gid]) == len(self.neigh_gid)))
        quality = (self.n-sum(quality))*100/self.n
        self.quality['gb_segmentation'] = quality
        if verbose:
            print(f'Grain boundary segmentation quality measure 1: {quality} %')

    def create_neigh_gid_pair_ids(self, neigh_gid, verbose=True):
        """Creates a dictionary mapping unique grain pairs to integer IDs.

        Args:
            neigh_gid (dict): A dictionary where keys are grain IDs and values are lists
                              of neighboring grain IDs.

        Returns:
            dict: A dictionary where keys are integer pair IDs and values are tuples of grain IDs.
        """
        if verbose:
            print("Setting up neighbor connectivity flags and consolidating grain boundary segments.")
        self.gid_pair_ids = {}
        pair_id = 1  # Start with pair ID 1
        # ----------------------------------------
        for gid, neighbors in neigh_gid.items():
            for neighbor in neighbors:
                # Create a sorted tuple of the pair (ensures uniqueness)
                pair = tuple(sorted((gid, neighbor)))

                # Assign a new pair ID if not seen before
                if pair not in self.gid_pair_ids:
                    self.gid_pair_ids[pair_id] = list(pair)
                    pair_id += 1
        # ----------------------------------------
        self.gid_pair_ids_unique_lr = np.unique(np.array(list(self.gid_pair_ids.values())), axis=0)
        self.gid_pair_ids_unique_rl = np.flip(self.gid_pair_ids_unique_lr, axis=1)

    def get_random_gbpoint_between_jnpoints(self, gbcoords, jnpcoords):
        """
        jnpcoords = jnp_all_sorted_coords[46][0:2]
        gbcoords = gbmullines_grain_wise[46].get_node_coords()
        get_random_gbpoint_between_jnpoints(gbcoords, jnpcoords)
        """
        first = DO.find_coorda_loc_in_coords_arrayb(jnpcoords[0], gbcoords)
        last = DO.find_coorda_loc_in_coords_arrayb(jnpcoords[1], gbcoords)
        if last-first >= 2:
            return gbcoords[np.random.randint(first+1, last)]
        else:
            return None

    def get_random_gbpoints_between_jnpoints(self, gid):
        """
        jnpcoords = jnp_all_sorted_coords[46][0:2]
        gbcoords = gbmullines_grain_wise[46].get_node_coords()
        get_random_gbpoint_between_jnpoints(gbcoords, jnpcoords)
        """
        gbcoords = self.gbmullines_grain_wise[gid].get_node_coords()
        seg_ends = self.extract_end_coordinates_of_grain_boundary_segments_grain_wise(gid)
        random_gb_points = {i: None for i in seg_ends.keys()}
        # ------------------------------------------
        for key, seg_end in seg_ends.items():
            first = DO.find_coorda_loc_in_coords_arrayb(seg_end[0], gbcoords)
            if sum(seg_end[1] - seg_ends[0][0]) == 0.0:
                last = len(gbcoords)
            else:
                last = DO.find_coorda_loc_in_coords_arrayb(seg_end[1], gbcoords)
            # ------------------------------------------
            if last-first >= 2:
                random_gb_points[key] = gbcoords[np.random.randint(first+1, last)]
            else:
                random_gb_points[key] = None
        return random_gb_points

    def setup_neigh_connectivity_flags_DS(self, neigh_sense='lr',
                                          field_names=['gbseg',
                                                       'nnodes_eq',
                                                       'length_eq', 'n',
                                                       'areas_raw',
                                                       'uniquified'],
                                        verbose=True):
        if verbose:
            print("Setting up neighbor connectivity flags for grain boundary segments.")
        if neigh_sense == 'lr':
            unique_pair_ids = self.gid_pair_ids_unique_lr
        elif neigh_sense == 'rl':
            unique_pair_ids = self.gid_pair_ids_unique_rl
        self.nconn = {tuple(neighpair): {fn: [] for fn in field_names}
                      for neighpair in unique_pair_ids}

    def extract_end_coordinates_of_grain_boundary_segments_grain_wise(self, gid):
        # Extract segment end coordinates
        seg_ends = {}
        for count in range(len(self.jnp_all_sorted_coords[gid])):
            seg_ends[count] = self.jnp_all_sorted_coords[gid][count:count+2]
        seg_ends[count] = np.vstack((seg_ends[count],
                                     self.gbmullines_grain_wise[gid].get_node_coords()[-1]))
        return seg_ends

    def set_neigh_connectivity_flags_DS(self, centroid_eq_EPS=1E-8, verbose=True):
        if verbose:
            print("Setting up neighbor connectivity flags for grain boundary segments based on centroid proximity.")
        for i, pair in enumerate(self.gid_pair_ids_unique_lr):
            # flag = nconn[tuple(pair)]['gbseg']
            pair_rl = tuple((pair[1], pair[0]))
            self.nconn[tuple(pair)]['areas_raw'].append(self.area_gid(pair[0], gsrepr='raw'))
            self.nconn[tuple(pair)]['areas_raw'].append(self.area_gid(pair[1], gsrepr='raw'))
            # ====================================
            for gbseg1 in self.gbsegments[pair[0]]:
                """iterating through all grian boundary segments of the centre
                grain, which is pair[0].
                """
                # gbseg1 = gbsegments[pair[0]][0]
                gbseg1_nnodes = gbseg1.nnodes  # Number of nodes
                gbseg1_centroid = gbseg1.centroid_p2dl  # Centroidal point object
                gbseg1_length = gbseg1.length  # Total lemngth
                # ====================================
                for gbseg2 in self.gbsegments[pair[1]]:
                    """iterating through all grian boundary segments current
                    neighbour grain, which is pair[1].
                    """
                    gbseg2_nnodes = gbseg2.nnodes
                    proceed = False
                    '''Prepare for the nnodes equality test.'''
                    nnodes_equality = gbseg1_nnodes == gbseg2_nnodes
                    if nnodes_equality:
                        '''nnodes equality test passed.'''
                        '''Prepare for the next test.'''
                        _fx_ = gbseg1_centroid.is_p2dl_within_cor
                        centroid_equality = _fx_(gbseg2.centroid_p2dl,
                                                 centroid_eq_EPS)
                    else:
                        continue
                    # ----------------------------
                    if centroid_equality:
                        '''centroid equality test passed.'''
                        '''Prepare for the next test.'''
                        ldiff = abs(gbseg1_length - gbseg2.length)
                        length_equality = ldiff <= centroid_eq_EPS
                    else:
                        continue
                    # ----------------------------
                    self.nconn[tuple(pair)]['gbseg'].append(gbseg1)
                    self.nconn[tuple(pair)]['gbseg'].append(gbseg2)
                    # ----------------------------
                    self.nconn[tuple(pair)]['nnodes_eq'].append(gbseg1.nnodes == gbseg2.nnodes)
                    # ----------------------------
                    self.nconn[tuple(pair)]['length_eq'].append(gbseg1.length == gbseg2.length)
                    # ----------------------------
                    self.nconn[tuple(pair)]['n'].append(len(self.nconn[tuple(pair)]['gbseg']))
                    self.nconn[tuple(pair)]['uniquified'] = False

    def get_unique_object_indices(self, nnodes, lengths, centroids, tol=1e-8):
        """
        Finds indices of unique objects based on nnodes, lengths, and centroids.

        Args:
            nnodes (np.ndarray): 1D array of nnode values.
            lengths (np.ndarray): 1D array of length values.
            centroids (np.ndarray): 2D array of centroid coordinates.
            tol (float, optional): Tolerance for centroid coordinate comparison. Defaults to 1e-6.

        Returns:
            np.ndarray: 1D array of indices corresponding to unique objects.
        """
        # Create a structured array for combined properties
        dtype = [('nnodes', nnodes.dtype),
                 ('lengths', lengths.dtype),
                 ('centroids', centroids.dtype, (2,))]
        data = np.empty(nnodes.shape[0], dtype=dtype)
        data['nnodes'] = nnodes
        data['lengths'] = lengths
        data['centroids'] = centroids
        # Round centroids to handle floating-point errors
        data['centroids'] = np.round(data['centroids'], decimals=int(-np.log10(tol)))
        # Find unique entries
        _, unique_indices = np.unique(data, return_index=True)
        return unique_indices

    def gather_grain_boundary_segments_of_all_pairs(self, verbose=True):
        # Gather grain boundary segments of all pairs
        if verbose:
            print("Gathering grain boundary segments of all pairs and consolidating them.")
            print("Constructing grain neigh pair ID keyed and gbseg valued dictionary")
        self.GBSEGMENTS = {tuple(pair): None for pair in self.gid_pair_ids_unique_lr}
        for pair in self.gid_pair_ids_unique_lr:
            _gbseg_ = self.nconn[tuple(pair)]['gbseg']
            if len(_gbseg_) > 0:
                ###########################################
                # MAKE UNIQUE THE LIST OF gbsegments in _gbseg_.
                nnodes = np.array([seg.nnodes for seg in self.nconn[tuple(pair)]['gbseg']])
                lengths = np.array([seg.length for seg in self.nconn[tuple(pair)]['gbseg']])
                centroids = np.array([seg.centroid for seg in self.nconn[tuple(pair)]['gbseg']])
                ui = self.get_unique_object_indices(nnodes, lengths, centroids, tol=1e-8)
                ###########################################
                self.GBSEGMENTS[tuple(pair)] = [self.nconn[tuple(pair)]['gbseg'][_ui_] for _ui_ in ui]
                '''As we no longer need any duplicates, we will override the repeated
                ones in ncoon as well.'''
                self.nconn[tuple(pair)]['gbseg'] = [self.nconn[tuple(pair)]['gbseg'][_ui_] for _ui_ in ui]
                ###########################################
                self.nconn[tuple(pair)]['uniquified'] = True
                # GBSEGMENTS[tuple(pair)] = _gbseg_[0]
                ###########################################
                # Access
                # [nconn[tuple((32, 36))]['gbseg'][_ui_] for _ui_ in ui]

    def consolidate_gbsegments(self, squeeze_segment_data_structure=False, verbose=True):
        """Consolidates grain boundary segments by grain ID.
        Args:
            GBSEGMENTS (dict): A dictionary where keys are tuples of grain IDs (gid1, gid2)
                               and values are grain boundary segments.
        Returns:
            dict: A dictionary where keys are grain IDs and values are lists of
                  grain boundary segments associated with that grain.
        """
        if verbose:
            print("Consolidating grain boundary segments and updating them with boundary grain IDs.")

        if self.gid.size == 1:
            self.consolidated_segments = self.gbsegments
            return
        grain_segments = {}  # To store the consolidated segments
        for pair, segment in self.GBSEGMENTS.items():
            gid1, gid2 = pair
            grain_segments.setdefault(gid1, []).append(segment)  # Add to gid1
            grain_segments.setdefault(gid2, []).append(segment)  # Add to gid2
        # -------------------------------------
        self.consolidated_segments = {}
        for index in np.unique(list(grain_segments.keys())):
            self.consolidated_segments[index] = grain_segments[index]
        # -------------------------------------
        if squeeze_segment_data_structure:
            for index in np.unique(list(grain_segments.keys())):
                squeezed = []
                if self.consolidated_segments[index] is not None:
                    for a in self.consolidated_segments[index]:
                        if a is not None:
                            squeezed.extend(a)
                self.consolidated_segments[index] = squeezed

    def get_problematic_grains(self, plot=False):
        """Def name must be changed."""
        areas, nidentical_seg, nnodes_eq, length_eq = [], [], [], []
        for nconn_val in self.nconn.values():
            areas.append(nconn_val['areas_raw'])
            nidentical_seg.append(nconn_val['n'])
            nnodes_eq.append(nconn_val['nnodes_eq'])
            length_eq.append(nconn_val['length_eq'])

        areas = np.array(areas)
        # Identify the problematic grains
        areas_min = np.array(areas).min(axis=1)
        areas_min_pair_locations = np.argwhere(areas_min == 1)
        pairids = self.gid_pair_ids_unique_lr[areas_min_pair_locations].squeeze()
        pairids_areas = areas[areas_min_pair_locations.squeeze()]
        # End
        ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## =
        # Start
        problematic_grains = {gid: self.polygons[gid-1] for gid in
                              np.unique(np.hstack((pairids[:, 0][pairids_areas[:, 0] == 1],
                                                   pairids[:, 1][pairids_areas[:, 1] == 1])))}
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(self.lgi)
            for prbgrain in problematic_grains.values():
                cenx, ceny = prbgrain.centroid.xy
                cenx, ceny = cenx[0]-0.5, ceny[0]-0.5
                ax.plot(cenx, ceny, 'kx')
            # problematic_grains
        return problematic_grains

    def find_segs_at_loc(self, gbsegs, axis='y', location=-0.5):
        gb_indices, gbsegs_at_location = [], []
        column = 0 if axis == 'x' else 1 if axis == 'y' else None
        for i, gbseg in enumerate(gbsegs, start=0):
            if all(gbseg.get_node_coords()[:, column] == location):
                gb_indices.append(i)
                gbsegs_at_location.append(gbseg)
        return gb_indices, gbsegs_at_location

    def update_consolidated_segments_with_boundary_grain_gids(self, plot=False, verbose=True):
        if verbose:
            print("Updating consolidated grain boundary segments with boundary grain IDs.")
        if self.gid.size == 1:
            return
        xmin, xmax, ymin, ymax = self.get_bounds_from_GBP()

        for fid in self.grain_loc_ids['bottom_left_corner']:
            for seg in self.gbsegments[fid]:
                if seg.has_coord([xmin, ymin]):
                    self.consolidated_segments[fid].append(seg)

        for fid in self.grain_loc_ids['bottom_right_corner']:
            for seg in self.gbsegments[fid]:
                if seg.has_coord([xmax, ymin]):
                    self.consolidated_segments[fid].append(seg)

        for fid in self.grain_loc_ids['top_right_corner']:
            for seg in self.gbsegments[fid]:
                if seg.has_coord([xmax, ymax]):
                    self.consolidated_segments[fid].append(seg)

        for fid in self.grain_loc_ids['top_left_corner']:
            for seg in self.gbsegments[fid]:
                if seg.has_coord([xmin, ymax]):
                    self.consolidated_segments[fid].append(seg)

        for gid in self.grain_loc_ids['pure_bottom']:
            gbind, gbs = self.find_segs_at_loc(self.gbsegments[gid], 
                                               axis='y', location=ymin)
            if len(gbs) > 0:
                for _gbs_ in gbs:
                    self.consolidated_segments[gid].append(_gbs_)

        for gid in self.grain_loc_ids['pure_right']:
            gbind, gbs = self.find_segs_at_loc(self.gbsegments[gid], 
                                               axis='x', location=xmax)
            if len(gbs) > 0:
                for _gbs_ in gbs:
                    self.consolidated_segments[gid].append(_gbs_)

        for gid in self.grain_loc_ids['pure_top']:
            gbind, gbs = self.find_segs_at_loc(self.gbsegments[gid], 
                                               axis='y', location=ymax)
            if len(gbs) > 0:
                for _gbs_ in gbs:
                    self.consolidated_segments[gid].append(_gbs_)

        for gid in self.grain_loc_ids['pure_left']:
            gbind, gbs = self.find_segs_at_loc(self.gbsegments[gid],
                                               axis='x', location=xmin)
            if len(gbs) > 0:
                for _gbs_ in gbs:
                    self.consolidated_segments[gid].append(_gbs_)

        if plot:
            self.plot_consolidated_segments()


    def plot_consolidated_segments(self):
        fig, ax = plt.subplots()
        ax.imshow(self.lgi)
        for gid in self.gid:
            for _ in self.consolidated_segments[gid]:
                _.plot(ax=ax)
            coord = self.gbmullines_grain_wise[gid].get_node_coords()
            plt.plot(coord[:, 0], coord[:, 1], 'k.')
            jnp = self.jnp_all_sorted_coords[gid]
            plt.plot(jnp[:, 0], jnp[:, 1], 'ks', mfc='c', ms=10, alpha=0.25)

    def check_if_all_gbsegs_can_form_closed_rings(self, GBSEGS,
                                                  _print_individual_excoord_order_=True,
                                                  _print_statement_=True
                                                  ):
        jnp_unique_counts_grain_wise = []
        for gid in GBSEGS.keys():
            extreme_coords_unique = []
            for seg in GBSEGS[gid]:
                extreme_coords_unique.extend(seg.get_node_coords()[[0, -1], :])
            extreme_coords_unique = np.unique(extreme_coords_unique, axis=0)
            extreme_coords_unique_count = [0 for _ in extreme_coords_unique]
            for i, excoord in enumerate(extreme_coords_unique):
                for seg in GBSEGS[gid]:
                    # seg = GBSEGS[gid][0]
                    segcoords = seg.get_node_coords()
                    if DO.is_a_in_b(excoord, segcoords):
                        extreme_coords_unique_count[i] += 1
            if _print_individual_excoord_order_:
                print(extreme_coords_unique_count)
            jnp_unique_counts_grain_wise.append(all(np.array(extreme_coords_unique_count) == 2))
        if all(jnp_unique_counts_grain_wise):
            if _print_statement_:
                print(40*'-', '\n All gid mapped gbsegs can form closed ring structure. \n', 40*'-')
            return True, jnp_unique_counts_grain_wise
        else:
            if _print_statement_:
                print(40*'-', '\n Some gbsegs cannot form closed ring structure. \n', 40*'-')
            return False, jnp_unique_counts_grain_wise

    def flip_segments_to_reorder_GBS(self, plot_each_grain_details=False, verbose=True):
        if verbose:
            print("Flipping grain boundary segments to reorder them in a consistent manner.")
        self.GB = {gid: None for gid in self.gid}
        self.quality['GBS_reordering_success'] = {gid: None for gid in self.gid}
        niterations = {gid: 0 for gid in self.gid}
        from upxo.geoEntities.mulsline2d import ring2d
        for gid in self.gid:
            gbsegs = self.consolidated_segments[gid]
            if verbose:
                print(40*'-')
            _gbseg_ring_ = ring2d(segments=gbsegs,
                                  segids=list(range(len(gbsegs))),
                                  segflips=[False for _ in gbsegs])
            # segments=None, segids=None, segflips=None
            # _gbseg_ring_.segments
            if plot_each_grain_details:
                _gbseg_ring_.plot_segs(plot_centroid=True, centroid_text=gid,
                                       plot_coord_order=True, visualize_flip_req=False)
            # _gbseg_ring_.create_polygon_from_coords()
            continuity, flip_needed, i_precede_chain = _gbseg_ring_.assess_spatial_continuity()
            if verbose:
                print(f'gid={gid}: ', 'gbsegs continuous' if continuity else 'gbsegs not continuous. Attempting reorder.')
            # print(40*'-', '\n Extreme coordinates of gbsegs are:\n')
            # for i, gbseg in enumerate(gbsegs, start = 0):
                # print(f'Segment {i}: ', gbseg.nodes[0], gbseg.nodes[-1])
            # print(40*'-')
            # plt.imshow(geom.lgi==gid)
            if continuity:
                '''If the segments are indeed continous, ring formation is
                stright-forward.'''
                # -------------
                """segments=None, segids=None, segflips=None"""
                # -------------
                NSEG_rng = range(len(gbsegs))
                self.GB[gid] = ring2d(gbsegs, list(NSEG_rng), [False for _ in NSEG_rng])
                # GB[gid].segments
                # GB[gid].segids
                # GB[gid].segflips
                self.quality['GBS_reordering_success'][gid] = True
                if plot_each_grain_details:
                    self.GB[gid].plot_segs(plot_centroid=True, centroid_text=gid,
                                           plot_coord_order=True, visualize_flip_req=True)
            if not continuity:
                '''If the segments are not found to be continous, ring formation
                requires the calculation of exact segids and segflip values. segids provide
                the spatial order of segments in ring.segments and segflips provide the
                boolean value indicating the need to flip a segment to ensure spatial
                continuity of ending nodes of adjacent multi-line-segments.
                '''
                self.quality['GBS_reordering_success'][gid] = False
                # ------------------------
                segids = list(range(len(gbsegs)))
                '''segstart_flip to be set to True if current gbsegs[0] goes
                counter-clockwise. Setting to False for now. Needs seperate
                assessment.'''
                segstart_flip = False
                self.GB[gid] = ring2d([gbsegs[0]], [0], [segstart_flip])
                '''
                GB[gid].segments
                GB[gid].segids
                GB[gid].segflips
                '''
                seg_num = 0
                used_segids = [seg_num]
                search_segids = set(segids)
                max_iterations = 10*len(gbsegs)
                itcount = 1  # iteration_count
                while len(search_segids) > 0:
                    current_seg = self.GB[gid].segments[-1]
                    flip_req_previous = self.GB[gid].segflips[-1]
                    # print(40*'-', '\n Current segment end nodes:\n')
                    # print(current_seg.nodes[0], current_seg.nodes[-1], '\n', 40*'-')
                    search_segids = set(segids) - set(used_segids)
                    adj = []
                    for candidate_segid in search_segids:
                        # candidate_segid = 2
                        # print(40*'-', '\n Current segment end nodes:\n')
                        # print(current_seg.nodes[0], current_seg.nodes[-1])
                        candidate_seg = gbsegs[candidate_segid]
                        if flip_req_previous:
                            adjacency = current_seg.do_i_proceed(candidate_seg)
                        else:
                            adjacency = current_seg.do_i_precede(candidate_seg)
                        adj.append(adjacency)
                        # print(40*'-', f'\n Cand. seg. {candidate_segid} end nodes:')
                        # print(candidate_seg.nodes[0], candidate_seg.nodes[-1])
                        # print(40*'-', f'\n Current precede candidate: {adjacency}\n', 40*'-')
                        # adjacency = current_seg.do_i_proceed(candidate_seg)
                        # print(adjacency)
                        if adjacency[0]:
                            used_segids.append(candidate_segid)
                            self.GB[gid].add_segment_unsafe(candidate_seg)
                            self.GB[gid].add_segid(candidate_segid)
                            self.GB[gid].add_segflip(adjacency[1])
                            '''
                            GB[gid].segments
                            GB[gid].segids
                            GB[gid].segflips
                            GB[gid].plot_segs()

                            for i, gbseg in enumerate(GB[gid].segments, start = 0):
                                print(f'Segment {i}: ', gbseg.nodes[0], gbseg.nodes[-1])
                            '''
                            break
                    if itcount >= max_iterations:
                        'Prevent infinite loop.'
                        break
                    itcount += 1
                if len(search_segids) == 0:
                    # --------------------------------
                    # Ensure complete connectivity
                    if not self.GB[gid].segments[0].nodes[0].eq_fast(self.GB[gid].segments[-1].nodes[-1]):
                        self.GB[gid].segflips[-1] = True
                    # --------------------------------
                    self.quality['GBS_reordering_success'][gid] = True
                    niterations[gid] = itcount
                    if verbose:
                        print(f'Re-ordering success. gbsegs are continous. N.Segs={len(gbsegs)}. N.Iterations={itcount}')
                else:
                    # self.quality['GBS_reordering_success'][gid] = False < -- BY default.
                    # So, nothinhg more to do here.
                    pass
                if plot_each_grain_details:
                    self.GB[gid].plot_segs(plot_centroid=True, centroid_text=gid,
                                           plot_coord_order=True, visualize_flip_req=True)
                # GB[gid].segflips
        if verbose:
            print(40*'-', f'\nTotal number of iterations: {sum(list(niterations.values()))}')

    def update_segflip_requirements(self, verbose=True):
        # Re-assessment - segflips of the last segment.
        if verbose:
            print("Updating segment flip requirements to ensure spatial continuity of grain boundary segments.")

        for gid in self.gid:
            self.GB[gid]
            self.GB[gid].segments[0]
            start = self.GB[gid].segments[0].nodes[0]
            end0 = self.GB[gid].segments[-1].nodes[0]
            end1 = self.GB[gid].segments[-1].nodes[-1]
            condition1 = start.eq_fast(end0)[0]
            condition2 = start.eq_fast(end1)[0]
            if condition1:
                self.GB[gid].segflips[-1] = True

    def sort_subsets_by_original_order(self, CA, subsets):
        """
        Sorts subsets of coordinates based on their original order in the CA array,
        and returns the sorted subsets along with their indices.

        Args:
            CA (np.ndarray): The original 2D coordinate array (N x 2).
            subsets (list): A list of np.ndarrays, each representing a subset of coordinates.

        Returns:
            tuple: A tuple containing two elements:
                - list: A list of np.ndarrays, where each subset is sorted according to the
                        order of its points in the CA array.
                - np.ndarray: A 1D array containing the original indices of the subsets
                              in the input list, corresponding to the order of sorted subsets.
        """

        coord_to_index = {tuple(coord): idx for idx, coord in enumerate(CA)}

        sorted_subsets = []
        subset_indices = []  # To track the original indices of subsets
        for i, subset in enumerate(subsets):
            sorted_indices = np.argsort([coord_to_index[tuple(coord)] for coord in subset])
            sorted_subsets.append(subset[sorted_indices])
            subset_indices.append(i)  # Record the original index

        # Sort subset indices based on the first element of each sorted subset
        sort_order = np.argsort([coord_to_index[tuple(s[0])] for s in sorted_subsets])
        sorted_subsets = [sorted_subsets[i] for i in sort_order]
        subset_indices = np.array(subset_indices)[sort_order]  # Convert to NumPy array and reorder

        return sorted_subsets, subset_indices

    def sort_gbsegments_by_original_order(self, verbose=True):
        if verbose:
            print("Sorting grain boundary segments by original order.")
        self.sorted_segs = {gid: None for gid in self.gid}
        _sorter_ = self.sort_subsets_by_original_order
        for gid in self.gid:
            _, subset_indices = _sorter_(self.gbmullines_grain_wise[gid].get_node_coords(),
                                         [seg.get_node_coords() for seg in self.consolidated_segments[gid]])
            self.sorted_segs[gid] = [self.consolidated_segments[gid][ssind] for ssind in subset_indices]

    def calculate_grain_boundary_coordinates_after_gbseg_reordering(self, verbose=True):
        if verbose:
            print("Calculating grain boundary coordinates after grain boundary segment reordering.")
        self.GBCoords = {gid: None for gid in self.gid}
        for gid in self.gid:
            if gid in self.grain_loc_ids['boundary']:
                coord = self.GB[gid].create_coords_from_segments(force_close=True)
            else:
                coord = self.GB[gid].create_coords_from_segments(force_close=False)
            self.GBCoords[gid] = coord

    def are_grains_closed_usenodes(self):
        return [gb.check_closed() for gb in self.GB.values()]

    def are_grains_closed_usecoords(self):
        coords = self.GBCoords
        flags = []
        for gid in self.gid:
            flag = np.abs(coords[gid][0] - coords[gid][-1]).sum() <= self.EPS_coord_coincide
            flags.append(flag)
        return flags

    def set_pure_gbpoints(self, verbose=True):
        if verbose:
            print("Setting up pure grain boundary points.")
        GBP_pure = []
        for gbp in np.unique(self.GBP, axis=0):
            if DO.is_a_in_b(gbp, np.unique(self.JNP, axis=0)):
                pass
            else:
                GBP_pure.append(gbp)
        self.GBP_pure = np.array(GBP_pure)

    def plot_reordered_GBCoords(self, gid, force_close=True):
        # gid = 43
        # GB[gid].segments
        # GB[gid].segflips
        coords = self.GB[gid].segments[0].get_node_coords()
        for i, seg in enumerate(self.GB[gid].segments[1:], start=1):
            if self.GB[gid].segflips[i]:
                thissegcoords = np.flip(seg.get_node_coords(), axis=0)
                coords = np.vstack((coords, thissegcoords[1:]))
            else:
                coords = np.vstack((coords, seg.get_node_coords()[1:]))
        if force_close:
            coords = self.force_close_coordinates(coords, assess_first=True)
        plt.figure()
        for gid in self.gid:
            c = self.GBCoords[gid]
            plt.plot(c[:, 0], c[:, 1])

    def plot_user_gbcoords(self, gbcoords, lw=1.5):
        fig, ax = plt.subplots()
        for gid in self.gid:
            plt.plot(gbcoords[gid][:, 0], gbcoords[gid][:, 1],
                     ls='solid', lw=lw)

    def plot_user_gbcoords1(self, gbcoords, lw=1.5):
        fig, ax = plt.subplots()
        for gid in self.gid:
            gbc = gbcoords[gid]
            if not DO.is_a_in_b(gbc[0], gbc[1:]):
                gbc = np.vstack((gbc, gbc[-1]))
            plt.plot(gbc[:, 0], gbc[:, 1],
                     ls='solid', lw=lw)

    def construct_geometric_xtals_from_gbcoords(self,
                                                coord_loop_dict,
                                                dtype='shapely',
                                                saa=True, throw=False):
        """ self.construct_geometric_xtals_from_gbcoords(GBCoords)."""
        if dtype == 'shapely':
            GRAINS = {gid: Polygon(coord_loop_dict[gid]) for gid in self.gid}
        if saa:
            self.GRAINS = GRAINS
        if throw:
            return GRAINS

    def construct_geometric_polyxtal_from_xtals(self,
                                                xtal_list,
                                                dtype='shapely',
                                                saa=True, throw=False):
        """
        self.construct_geometric_polyxtal_from_xtals(self.GRAINS.values(), dtype='shapely')
        """
        if dtype == 'shapely':
            POLYXTAL = MultiPolygon(xtal_list)
        if saa:
            self.POLYXTAL = POLYXTAL
        if throw:
            return POLYXTAL

    def construct_geometric_polyxtal_from_gbcoords(self,
                                                   coord_loop_dict,
                                                   dtype='shapely',
                                                   saa=True, throw=False,
                                                   plot_polyxtal=False,
                                                   verbose=True
                                                   ):
        if verbose:
            print("Constructing geometric polyxtal from grain boundary coordinates.")
        if dtype == 'shapely':
            if saa:
                self.construct_geometric_xtals_from_gbcoords(coord_loop_dict,
                                                             saa=True,
                                                             throw=False)
                self.POLYXTAL = MultiPolygon(self.GRAINS.values())
                if plot_polyxtal:
                    self.plot_multipolygon(self.POLYXTAL, invert_y=True)
                if throw:
                    return self.GRAINS, self.POLYXTAL

            if not saa and throw:
                GRAINS = self.construct_geometric_xtals_from_gbcoords(coord_loop_dict,
                                                                      saa=False,
                                                                      throw=True)
                POLYXTAL = MultiPolygon(GRAINS.values())
                if plot_polyxtal:
                    self.plot_multipolygon(POLYXTAL, invert_y=True)
                return GRAINS, POLYXTAL


    def AssembleGBSEGS(self, GB, saa=True, throw=False, verbose=True):
        if verbose:
            print("Assembling grain boundary segments into grain boundary multi-linestrings")
        mids_all_gbsegs = []
        for gb in GB.values():
            mids_all_gbsegs.extend([id(seg) for seg in gb.segments])
        mids_all_gbsegs = np.unique(mids_all_gbsegs)
        sgseg_obj_list = [None for mid in mids_all_gbsegs]
        # --------------------------
        for i, mid in enumerate(mids_all_gbsegs, start=0):
            for gid in self.gid:
                for gb in GB[gid].segments:
                    if mid == id(gb):
                        sgseg_obj_list[i] = gb
        sgseg_obj_list = np.array(sgseg_obj_list)
        # --------------------------
        if saa:
            self.mids_all_gbsegs = mids_all_gbsegs
            self.sgseg_obj_list = sgseg_obj_list
        if throw:
            return mids_all_gbsegs, sgseg_obj_list

    def get_mids_gbsegs(self, gid):
        return [id(seg) for seg in self.GB[gid].segments]

    def get_gbmid_indices_at_gid(self, gid, all_mids):
        segmids = self.get_mids_gbsegs(gid)
        locs = [np.argwhere(all_mids == segmid)[0][0] for segmid in segmids]
        # for segmid in segmids:
        #     locs.append(np.argwhere(all_mids == segmid)[0][0])
        return locs

    def smooth_gbsegs(self, GB, npasses=2, max_smooth_levels=[3, 3], plot=True,
                      name='kali'):
        # Validations
        if type(max_smooth_levels) in dth.dt.NUMBERS:
            max_smooth_levels = [max_smooth_levels]
        # -----------------------------------------------
        GB_smooth = deepcopy(GB)
        all_mids, sgseg_list = self.AssembleGBSEGS(GB_smooth, saa=False, throw=True)
        for np in range(npasses):
            print(f"Carrying out smoothing pass: {np+1}")
            slevel = max_smooth_levels[np]
            for seg in sgseg_list:
                seg.smooth(max_smooth_level=slevel)
        # -------------------------------------
        if plot:
            for gid in self.gid:
                GB_smooth[gid].plot_segs()
            for gid in self.gid:
                self.GB[gid].plot_segs()
        # -------------------------------------
        GBCoords_smoothed = {gid: None for gid in self.gid}
        # -------------------------------------
        for gid in self.gid:
            if gid in self.grain_loc_ids['boundary']:
                coord = GB_smooth[gid].create_coords_from_segments(force_close=True)
            else:
                coord = GB_smooth[gid].create_coords_from_segments(force_close=False)
            GBCoords_smoothed[gid] = coord
        # -------------------------------------
        if plot:
            plt.figure()
            for gid in self.gid:
                c = GBCoords_smoothed[gid]
                plt.plot(c[:, 0], c[:, 1])
        # -------------------------------------
        _fn_ = self.construct_geometric_xtals_from_gbcoords
        GRAINS = _fn_(GBCoords_smoothed, saa=False, throw=True)
        POLYXTAL = MultiPolygon(GRAINS.values())
        # -------------------------------------
        self.smoothed[name] = {'GB': GB_smooth,
                               'GBCoords': GBCoords_smoothed,
                               'GRAINS': GRAINS,
                               'POLYXTAL': POLYXTAL,
                               }

    def plotgs(self, gs_geometric, fig=None, ax=None, cmap='tab20', edgecolor='black', 
               alpha=0.7, lw=1, figsize=(10, 10), dpi=100):
        from upxo.viz.gsviz import plot_multipolygon_geometric
        fig, ax = plot_multipolygon_geometric(gs_geometric, fig=fig, ax=ax, cmap=cmap,
                        edgecolor=edgecolor, alpha=alpha, lw=lw,
                        figsize=figsize, dpi=dpi)
        return fig, ax

# =====================================================================================
# =====================================================================================
# =====================================================================================

class VoronoiMasking(ABC):
    def __init__(self, lfi):
        self.lfi = lfi  # n x m (x o)
        self.seeds = None         # Seed coordinates
        self.cell_to_id = {}   # Mapping: {cell_idx: grain_id}
        self.lfi_field = None  # Optional: Specific field to sample from
        self.cells = {}       # {cell_id: geometric object}

    def map_seeds_to_lfi(self, channel=0):
        """
        Samples the LFI at seed coordinates.
        Correctly handles the conversion from Cartesian (x, y) seeds 
        to array indices (row, col) for sampling.
        """
        from scipy.ndimage import map_coordinates

        # Handle multi-channel vs single-channel indexing
        # self.seeds.shape[1] is the spatial dimension (2 for 2D, 3 for 3D)
        if self.lfi.ndim > self.seeds.shape[1]: 
            target_data = self.lfi[..., channel]
        else:
            target_data = self.lfi

        # CRITICAL FIX: 
        # map_coordinates requires coordinates in (row, col, depth) order.
        # Our seeds are in (x, y, z) order.
        # For 2D: we swap [x, y] -> [y, x] to get [row, col]
        # For 3D: we swap [x, y, z] -> [z, y, x] to get [plane, row, col]
        coords_for_sampling = self.seeds[:, ::-1].T 

        # order=0 is essential for labeled images to avoid interpolating between IDs
        ids = map_coordinates(target_data, coords_for_sampling, order=0, mode='nearest')
        
        # Store mapping: {cell_index: entity_id}
        self.cell_to_id = {i: int(val) for i, val in enumerate(ids)}

    @classmethod
    @abstractmethod
    def by_tessellation(cls, lfi, seeds):
        """Factory method to create instance via new Voronoi computation."""
        pass

    @classmethod
    @abstractmethod
    def load_tessellation(cls, lfi, filepath):
        """Factory method to create instance from a saved geometry file."""
        pass

    @abstractmethod
    def assemble_cells(self):
        """Dimension-specific: Shapely UnaryUnion vs PyVista Merge."""
        pass

# =====================================================================================
# =====================================================================================
# =====================================================================================

from shapely.ops import unary_union
from collections import defaultdict

class GrainManifold2D(VoronoiMasking):
    @classmethod
    def by_tessellation(cls, lfi, seeds, channel=0):
        instance = cls(lfi)
        instance.seeds = seeds
        # 1. Map seeds
        instance.map_seeds_to_lfi(channel=channel)
        # 2. Generate clipped polygons (your internal logic)
        polygons = instance._generate_clipped_polygons(seeds)
        # 3. Assemble into 'cells' (Grains/Twins/etc)
        instance.assemble_cells(polygons)
        return instance

    def assemble_cells(self, polygons):
        """Steps 5 & 6: Groups by ID and performs manifold union."""
        groups = defaultdict(list)
        for cell_idx, entity_id in self.cell_to_id.items():
            groups[entity_id].append(polygons[cell_idx])
        for entity_id, poly_list in groups.items():
            # Merges all Voronoi cells into a single manifold entity
            self.cells[entity_id] = unary_union(poly_list)

    def _generate_clipped_polygons(self, seeds):
        """
        Step 3: Internal helper to generate finite, clipped Voronoi polygons.
        """
        from scipy.spatial import Voronoi
        from shapely.geometry import Polygon, box
        # 1. Define RVE boundaries based on LFI shape
        height, width = self.lfi.shape[:2]
        boundary_box = box(0, 0, width, height)
        
        # 2. Ghost Seed Padding (Reflect boundary points to trap infinite rays)
        # Using a 10% buffer based on RVE size
        pad = max(width, height) * 0.1
        
        left = seeds[seeds[:, 0] < pad].copy()
        left[:, 0] = -left[:, 0]
        
        right = seeds[seeds[:, 0] > (width - pad)].copy()
        right[:, 0] = 2 * width - right[:, 0]
        
        bottom = seeds[seeds[:, 1] < pad].copy()
        bottom[:, 1] = -bottom[:, 1]
        
        top = seeds[seeds[:, 1] > (height - pad)].copy()
        top[:, 1] = 2 * height - top[:, 1]
        
        # Stack original seeds with ghosts
        all_seeds = np.vstack([seeds, left, right, bottom, top])
        
        # 3. Generate Voronoi
        vor = Voronoi(all_seeds)
        
        # 4. Extract and Clip Polygons
        polygons = []
        self.cell_vertices_raw = [] # Initialize storage for reconstruction
        # Only iterate through the original seeds (first len(seeds) points)
        for i in range(len(seeds)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            # In a padded setup, regions for original seeds are guaranteed to be finite
            # but we clip with the bounding box to ensure perfect RVE edges
            verts = vor.vertices[region]
            # Store the raw vertex coordinates for this cell
            # We store them as a list of tuples to be hashable for the map
            self.cell_vertices_raw.append([tuple(v) for v in verts])

            poly = Polygon(verts)
            # Intersection ensures the cell stays within [0,0] to [width, height]
            clipped_poly = poly.intersection(boundary_box)
            polygons.append(clipped_poly)
            
        return polygons # This prevents the TypeError

    @classmethod
    def load_tessellation(cls, lfi, filepath):
        """Placeholder implementation to satisfy abstract requirement."""
        instance = cls(lfi)
        # Add logic to load shapely geometries from file here
        return instance
    
    def smooth_interfaces(self, iterations=10, lmbda=0.5, mu=-0.53):
        # Initial adjacency and coordinate state
        adj = self._get_vertex_adjacency()
        # Initial map: p -> current_position
        coords = {p: p for p in adj.keys()}
        
        for _ in range(iterations):
            # Step A: Shrink (lambda)
            coords = self._laplacian_step(lmbda, coords_map=coords)
            # Step B: Inflate (mu)
            coords = self._laplacian_step(mu, coords_map=coords)
        
        # Final step: update the Shapely objects
        self._reconstruct_from_coords(coords)

    def _laplacian_step(self, factor, coords_map=None):
        """
        Smoothing step. Calculates displacement for each vertex toward the average of its neighbors.
        """
        # Use the current state of vertices if no intermediate map is provided
        if coords_map is None:
            # We use tuple keys to map original coordinates to current positions
            coords_map = {p: p for p in self._get_vertex_adjacency().keys()}
        adj = self._get_vertex_adjacency()
        height, width = self.lfi.shape[:2]
        new_coords = {}
        for p, neighbors in adj.items():
            x, y = p
            # --- TOPOLOGICAL PINNING ---
            # 1. Freeze RVE boundaries (x=0, x=width, y=0, y=height)
            is_on_boundary = (x <= 0 or x >= width or y <= 0 or y >= height)
            # 2. Freeze Triple/Quadruple Points (> 2 neighbors)
            is_junction = len(neighbors) > 2
            # 3. Handle Endpoints (isolated segments)
            is_endpoint = len(neighbors) < 2
            if is_on_boundary or is_junction or is_endpoint:
                # Anchor these points to maintain manifold integrity
                new_coords[p] = coords_map[p]
            else:
                # Discrete Laplacian: v_new = v_old + factor * (Average(neighbors) - v_old)
                neighbor_list = list(neighbors)
                # Pull coordinates from the current map
                p1_coords = coords_map[neighbor_list[0]]
                p2_coords = coords_map[neighbor_list[1]]
                current_p = coords_map[p]
                # Calculate centroid of neighbors
                avg_x = (p1_coords[0]+p2_coords[0]) / 2
                avg_y = (p1_coords[1]+p2_coords[1]) / 2
                # Apply displacement
                dx, dy = avg_x - current_p[0], avg_y-current_p[1]
                new_coords[p] = (current_p[0]+factor*dx, current_p[1]+factor*dy)
        return new_coords

    def _get_vertex_adjacency_old(self):
        """
        Builds a map of shared vertices and their neighbors.
        Triple points and RVE corners are identified to be pinned.
        """
        from collections import defaultdict
        adj = defaultdict(set)
        
        for gid, poly in self.cells.items():
            # Get exterior coordinates (x, y)
            coords = list(poly.exterior.coords)
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i+1]
                adj[p1].add(p2)
                adj[p2].add(p1)
                
        return adj

    def _get_vertex_adjacenc_old2(self):
        """
        Builds a map of shared vertices and their neighbors.
        Robustly handles MultiPolygons (islands/fragmented grains).
        """
        from collections import defaultdict
        from shapely.geometry import MultiPolygon
        
        adj = defaultdict(set)
        
        for gid, geom in self.cells.items():
            # Recursive check for multi-part geometries
            if isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)
            else:
                polys = [geom]
                
            for poly in polys:
                # Extract exterior ring coordinates
                coords = list(poly.exterior.coords)
                for i in range(len(coords) - 1):
                    p1, p2 = coords[i], coords[i+1]
                    # Map adjacency bidirectionally
                    adj[p1].add(p2)
                    adj[p2].add(p1)
                    
        return adj

    def _get_vertex_adjacency(self):
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        from collections import defaultdict
        
        adj = defaultdict(set)
        
        for entity_id, geom in self.cells.items():
            # Helper to extract polygons from any geometry type
            def get_polys(g):
                if isinstance(g, Polygon):
                    return [g]
                elif isinstance(g, (MultiPolygon, GeometryCollection)):
                    # Recursively find polygons inside collections
                    res = []
                    for part in g.geoms:
                        res.extend(get_polys(part))
                    return res
                return [] # Ignore Points and LineStrings

            polys = get_polys(geom)
            
            for poly in polys:
                # Now safe to access .exterior
                coords = list(poly.exterior.coords)
                for i in range(len(coords) - 1):
                    p1, p2 = tuple(coords[i]), tuple(coords[i+1])
                    adj[p1].add(p2)
                    adj[p2].add(p1)
                    
        return adj

    def gb_smooth(self, ma_window=3, taubin_iter=10, lmbda=0.5, mu=-0.5, ma_smoother_version=2):
        # 1. Stage 1: Local Moving Average
        if ma_smoother_version == 1:
            self._apply_moving_average_v1(window_size=ma_window)
        else:
            self._apply_moving_average_v2(window_size=ma_window)
        
        # 2. CRITICAL SYNC: Re-build adjacency based on the MOVED vertices
        # This prevents Taubin from using the old jagged voxel positions.
        self.vertex_adj = self._get_vertex_adjacency() 
        
        # 3. Stage 2: Global Taubin
        self.apply_taubin_smoothing(iterations=taubin_iter, lmbda=lmbda, mu=mu)

    def _apply_moving_average_v1(self, window_size=3):
        """
        Improved Moving Average that preserves topological anchors.
        """
        from shapely.geometry import Polygon
        import numpy as np

        adj = self._get_vertex_adjacency()
        height, width = self.lfi.shape[:2]
        
        # Identify anchor points (Junctions and RVE boundaries)
        anchors = {p for p, neighbors in adj.items() 
                if len(neighbors) > 2 or p[0] <= 0 or p[0] >= width or p[1] <= 0 or p[1] >= height}

        new_polygons = []
        for i in range(len(self.seeds)):
            verts = np.array(self.cell_vertices_raw[i])
            smoothed_verts = np.copy(verts)

            # Apply smoothing only to non-anchor vertices
            for j in range(1, len(verts) - 1):
                p = tuple(verts[j])
                if p not in anchors:
                    # Average with immediate neighbors in the vertex list
                    prev_p = verts[j-1]
                    next_p = verts[j+1]
                    smoothed_verts[j] = (prev_p + verts[j] + next_p) / 3

            # Ensure closing vertex matches starting vertex
            smoothed_verts[-1] = smoothed_verts[0]
            
            # Heal with buffer(0) to prevent GEOS errors
            new_polygons.append(Polygon(smoothed_verts).buffer(0))

        self._re_assemble_from_polygons(new_polygons)

        return new_polygons

    def _apply_moving_average_v2(self, window_size=3):
        """
        Applies segment-based smoothing using the user's mean_coordinates logic.
        Each boundary segment between anchor points is smoothed independently.
        """
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        import numpy as np

        # Internal helper to compute moving average (mode='valid')
        def local_ma(data, w):
            if len(data) < w: return data
            return np.convolve(data, np.ones(w) / w, mode='valid')

        # Internal helper to apply smoothing and re-cap endpoints
        def local_smooth(coords, w):
            if len(coords) < w: return coords
            x, y = coords[:, 0], coords[:, 1]
            x_smooth = local_ma(x, w)
            y_smooth = local_ma(y, w)
            
            if len(x_smooth) > 0:
                return np.vstack([
                    [x[0], y[0]], # Keep start anchor
                    np.column_stack([x_smooth, y_smooth]),
                    [x[-1], y[-1]] # Keep end anchor
                ])
            return coords

        # 1. Identify Anchor Points (Triple points and RVE boundaries)
        adj = self._get_vertex_adjacency()
        height, width = self.lfi.shape[:2]
        anchors = {p for p, neighbors in adj.items() 
                if len(neighbors) > 2 or p[0] <= 0 or p[0] >= width or p[1] <= 0 or p[1] >= height}

        new_cells = {}
        for gid, geom in self.cells.items():
            parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
            smoothed_parts = []
            
            for part in parts:
                coords = np.array(part.exterior.coords)
                anchor_indices = sorted(list(set([i for i, p in enumerate(coords) if tuple(p) in anchors])))
                
                if not anchor_indices:
                    # Handle closed loops with no anchors (Islands)
                    smoothed_ring = local_smooth(coords, window_size)
                else:
                    # Ensure the loop is fully covered
                    if anchor_indices[0] != 0: anchor_indices = [0] + anchor_indices
                    if anchor_indices[-1] != len(coords) - 1: anchor_indices = anchor_indices + [len(coords) - 1]
                    
                    segments = []
                    for i in range(len(anchor_indices) - 1):
                        seg = coords[anchor_indices[i] : anchor_indices[i+1] + 1]
                        smoothed_seg = local_smooth(seg, window_size)
                        segments.append(smoothed_seg[:-1])
                    
                    segments.append(smoothed_seg[-1:])
                    smoothed_ring = np.vstack(segments)

                new_poly = Polygon(smoothed_ring)
                smoothed_parts.append(new_poly.buffer(0))

            new_cells[gid] = unary_union(smoothed_parts) if len(smoothed_parts) > 1 else smoothed_parts[0]

        self.cells = new_cells
        # Update raw vertices so Taubin uses the smoothed segment nodes
        for gid, geom in self.cells.items():
            if isinstance(geom, Polygon):
                self.cell_vertices_raw[gid] = list(geom.exterior.coords)
        return self.cells

    def _taubin_step(self, adj, coords_map, factor):
        """Pass of Taubin filter with RVE boundary freezing."""
        height, width = self.lfi.shape[:2]
        new_coords = {}
        
        for p, neighbors in adj.items():
            x, y = p
            
            # --- FREEZE LOGIC ---
            # 1. Freeze RVE boundaries (x=0, x=width, y=0, y=height)
            is_on_boundary = (x <= 0 or x >= width or y <= 0 or y >= height)
            
            # 2. Freeze Triple Points (more than 2 neighbors)
            is_triple_point = len(neighbors) > 2
            
            if is_on_boundary or is_triple_point:
                new_coords[p] = p # Keep original coordinates exactly
            else:
                # Standard Taubin smoothing for internal interface points
                neighbor_list = list(neighbors)
                avg_x = sum(coords_map[n][0] for n in neighbor_list) / 2
                avg_y = sum(coords_map[n][1] for n in neighbor_list) / 2
                
                dx = avg_x - coords_map[p][0]
                dy = avg_y - coords_map[p][1]
                new_coords[p] = (coords_map[p][0] + factor * dx, 
                                coords_map[p][1] + factor * dy)
        return new_coords

    def apply_taubin_smoothing(self, iterations=10, lmbda=0.5, mu=-0.53):
        """
        Step 8: Smooths interfaces while keeping triple points pinned.
        """
        adj = self._get_vertex_adjacency()
        # Current state of all vertices in the system
        coords = {p: p for p in adj.keys()}
        
        for _ in range(iterations):
            # Step 1: Shrink
            coords = self._taubin_step(adj, coords, lmbda)
            # Step 2: Inflate
            coords = self._taubin_step(adj, coords, mu)
            
        # Update the Shapely geometries with smoothed coordinates
        self._reconstruct_from_coords(coords)

    def _reconstruct_from_coords(self, smoothed_coords_map):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from collections import defaultdict
        import shapely

        updated_polygons = []

        for i in range(len(self.seeds)):
            original_verts = self.cell_vertices_raw[i] 
            new_verts = [smoothed_coords_map.get(v, v) for v in original_verts]
            
            # Create polygon and ensure it is valid
            poly = Polygon(new_verts)
            if not poly.is_valid:
                poly = poly.buffer(0) # Standard fix for self-intersections
            
            updated_polygons.append(poly)

        self.cells = {}
        groups = defaultdict(list)
        for cell_idx, entity_id in self.cell_to_id.items():
            groups[entity_id].append(updated_polygons[cell_idx])
            
        for entity_id, poly_list in groups.items():
            try:
                # Attempt union
                self.cells[entity_id] = unary_union(poly_list)
            except Exception:
                # If union fails, clean geometries further and try again
                clean_polys = [p.buffer(0.0001) for p in poly_list] # Tiny expansion to force overlap
                self.cells[entity_id] = unary_union(clean_polys)

    def _re_assemble_from_polygons_old(self, updated_polygons):
        """
        Groups Voronoi cells by their Grain/Entity ID and merges them.
        Includes robustness checks to handle GEOS TopologyExceptions.
        Stores individual constituent cells in self.constituent_cells.
        """
        from shapely.ops import unary_union
        from collections import defaultdict
        # import shapely

        # 1. Initialize the storage for constituent cells {entity_id: [poly1, poly2, ...]}
        self.constituent_cells = defaultdict(list)

        # Group individual smoothed Voronoi polygons by their assigned ID
        groups = defaultdict(list)
        for cell_idx, entity_id in self.cell_to_id.items():
            poly = updated_polygons[cell_idx]
            groups[entity_id].append(poly)
            # Store in the persistent attribute for sub-grain analysis
            self.constituent_cells[entity_id].append(poly)

        # 2. Clear current cells and perform the manifold union
        self.cells = {}
        for entity_id, poly_list in groups.items():
            try:
                # Standard union for well-behaved smoothed cells
                union_geom = unary_union(poly_list)
                # Ensure the result is valid (fixes self-intersections from MA)
                if not union_geom.is_valid:
                    union_geom = union_geom.buffer(0)
                self.cells[entity_id] = union_geom
            except Exception as e:
                # Fallback: If MA caused vertices to overlap exactly, 
                # a tiny buffer helps GEOS resolve the 'side location conflict'.
                print(f"Heuristic repair triggered for Grain {entity_id} due to: {e}")
                cleaned_polys = [p.buffer(1e-7) if not p.is_valid else p for p in poly_list]
                self.cells[entity_id] = unary_union(cleaned_polys).buffer(-1e-7)

        return self.cells

    def _re_assemble_from_polygons_old1(self, updated_polygons):
        """
        Groups Voronoi cells by their ID and merges them into a clean manifold.
        Prunes GeometryCollections to ensure only Polygons remain.
        """
        from shapely.ops import unary_union
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        from collections import defaultdict
        import shapely
        # 1. Group and capture constituents
        self.constituent_cells = defaultdict(list)
        groups = defaultdict(list)
        for cell_idx, entity_id in self.cell_to_id.items():
            poly = updated_polygons[cell_idx]
            groups[entity_id].append(poly)
            self.constituent_cells[entity_id].append(poly)
        # 2. Perform the manifold union with artifact pruning
        self.cells = {}
        for entity_id, poly_list in groups.items():
            try:
                # Stage 1: Initial Union
                union_geom = unary_union(poly_list)
                # Stage 2: Precision Snapping (Fixes sliver/topology errors)
                # This prevents GEOS from seeing 'nearly identical' lines as separate
                union_geom = shapely.set_precision(union_geom, grid_size=1e-6)
                # Stage 3: GeometryCollection Pruning
                if isinstance(union_geom, GeometryCollection):
                    # Recursively extract only the polygonal parts
                    polys = [g for g in union_geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
                    union_geom = unary_union(polys)
                # Final validation and healing
                if not union_geom.is_valid:
                    union_geom = union_geom.buffer(0)
                # Ensure Grain 1 doesn't store that stray POINT
                self.cells[entity_id] = union_geom
            except Exception as e:
                print(f"Heuristic repair triggered for Grain {entity_id}: {e}")
                cleaned_polys = [p.buffer(1e-7) if not p.is_valid else p for p in poly_list]
                self.cells[entity_id] = unary_union(cleaned_polys).buffer(-1e-7)
        return self.cells

    def _re_assemble_from_polygons(self, updated_polygons):
        """
        Groups Voronoi cells by Grain ID and merges them.
        Explicitly ignores guard seeds (ID == -1) to ensure clean RVE boundaries.
        """
        from shapely.ops import unary_union
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        from collections import defaultdict
        import shapely

        # 1. Initialize storage
        self.constituent_cells = defaultdict(list)
        groups = defaultdict(list)
        
        # 2. Group individual polygons by their assigned ID
        for cell_idx, entity_id in self.cell_to_id.items():
            # --- IGNORE GUARD SEEDS ---
            # Guard seeds have ID -1; they are only used to 'shape' the boundary
            if entity_id == -1:
                continue
                
            poly = updated_polygons[cell_idx]
            groups[entity_id].append(poly)
            self.constituent_cells[entity_id].append(poly)

        # 3. Perform the manifold union with artifact pruning
        self.cells = {}
        for entity_id, poly_list in groups.items():
            try:
                # Merge individual cells into a single Grain geometry
                union_geom = unary_union(poly_list)
                
                # Snap to precision to fix sliver/topology errors
                union_geom = shapely.set_precision(union_geom, grid_size=1e-6)

                # Prune GeometryCollections to ensure only Polygons remain
                if isinstance(union_geom, GeometryCollection):
                    polys = [g for g in union_geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
                    union_geom = unary_union(polys)

                # Final 'healing' pass to fix self-intersections from smoothing
                if not union_geom.is_valid:
                    union_geom = union_geom.buffer(0)
                
                self.cells[entity_id] = union_geom
                
            except Exception as e:
                print(f"Heuristic repair triggered for Grain {entity_id}: {e}")
                cleaned_polys = [p.buffer(1e-7) if not p.is_valid else p for p in poly_list]
                self.cells[entity_id] = unary_union(cleaned_polys).buffer(-1e-7)

        return self.cells

    def _capture_grain_constituents(self):
        """
        Groups individual Voronoi cells by their assigned Grain ID
        and stores them in an attribute for later analysis or retrieval.
        """
        from collections import defaultdict
        
        # Initialize the container: {grain_id: [list_of_shapely_polygons]}
        self.grain_to_cells = defaultdict(list)

        # Iterate through all generated Voronoi cells
        # Note: self.cell_to_id maps the index of the seed to the Grain ID from the LFI
        for i in range(len(self.seeds)):
            grain_id = self.cell_to_id[i]

            # Retrieve the individual Voronoi polygon for this seed
            # Assuming they are stored in a temporary list or self.cell_vertices_raw
            from shapely.geometry import Polygon
            cell_polygon = Polygon(self.cell_vertices_raw[i])
            
            # Store the individual cell against the final grain ID key
            self.grain_to_cells[grain_id].append(cell_polygon)

        return self.grain_to_cells

    def trim_to_rve(self, bounds=(0, 0, 200, 200)):
        """
        Trims all grain geometries to the exact RVE bounding box.
        This ensures perfectly straight, rectangular external faces.
        """
        from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
        import shapely

        # 1. Define the 'Cookie Cutter' (The RVE Box)
        minx, miny, maxx, maxy = bounds
        rve_box = box(minx, miny, maxx, maxy)

        trimmed_cells = {}

        for gid, geom in self.cells.items():
            # 2. Perform Intersection with the RVE Box
            # This cuts off any 'leakage' outside the 200x200 domain
            cut_geom = geom.intersection(rve_box)

            # 3. Artifact Pruning (Similar to our re_assemble logic)
            if isinstance(cut_geom, GeometryCollection):
                polys = [g for g in cut_geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
                cut_geom = shapely.ops.unary_union(polys)

            # 4. Snap to precision to ensure edges are exactly on the boundary
            # This prevents floating point errors like 199.99999999
            cut_geom = shapely.set_precision(cut_geom, grid_size=1e-6)

            # Final healing pass
            if not cut_geom.is_valid:
                cut_geom = cut_geom.buffer(0)

            trimmed_cells[gid] = cut_geom

        # Update the internal manifold state
        self.cells = trimmed_cells
        return self.cells

# =====================================================================================
# =====================================================================================
# =====================================================================================

class GrainManifold3D(VoronoiMasking):
    @classmethod
    def by_tessellation(cls, lfi, seeds, channel=0):
        instance = cls(lfi)
        instance.seeds = seeds
        instance.map_seeds_to_lfi(channel=channel)
        # 3D specific tessellation (returns list of PyVista UnstructuredGrids)
        solids = instance._generate_clipped_solids(seeds)
        instance.assemble_cells(solids)
        return instance

    def assemble_cells(self, solids):
        # Use PyVista/VTK boolean_union or append filters here
        pass