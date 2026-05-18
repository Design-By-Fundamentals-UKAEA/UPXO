import os
import warnings
import numpy as np
from copy import deepcopy
from upxo._sup import dataTypeHandlers as dth
from upxo.repqual.grain_network_repr_assesser import KREPR
from upxo.analysis.analysis2d import gsan2d
import matplotlib.pyplot as plt
import upxo.viz.ebsdviz as ebsdviz

warnings.simplefilter('ignore', DeprecationWarning)


class repgen2d:

    __slots__ = ('tdist', 'tstat', 'tgs', 'sgs',
                 'iroute', 'mpflags', 'rm0tests', 'rm0',
                 'sgstype', 'tgstype', 'tdim', 'px_size',
                 'sdim', 'gsan_sgs', 'gsan_tgs', 'ebsd_file', 'ebsd_step',
                 'lfi_ebsd', 'euler_ebsd', 'quat_ebsd', 'neigh_gid_ebsd',
                 'prop_ebsd', 'prop_ebsd_df', 'stat_ebsd',
                 'lfi_ebsd_merged', 'prop_ebsd_merged', 'prop_ebsd_merged_df',
                 'repr_rank_ng', 'grain_count_rank_ng',
                 'merge_info',
                 'mc_host_orientations',
                 'mc_twin_geom',
                 'mc_smooth_geom',
                 'mc_smooth_quats',
                 'mc_smooth_mesh')
    '''
    Explanation of slot variables:
    ------------------------------
    tdist: upxo distribution collection object
        Distribution data of the target grain structure.
    tstat: upxo statistics collection object
        Statistics data of the target grain structure.
    tgs: grain structure data object
        The target grain structure.
    sgs: grain structure data object
        The sample grain structure.
    iroute: str
        Route to use for the generation of the representative grain structure.
    mpflags: dict
        Control parameters for the generation and use of morphological
        properties of the target and/or sample grain structure.
    rm0tests: dict
        Specifies which r0 tests to perform.
    '''

    VALiroutes = ('tdist.sgs', 'tstat.sgs', 'tgs.sgs')
    '''
    Explanation of VALiroutes:
    -------------------------
    The valid routes for the generation of the representative grain structure.
        1. 'tdist.sgs': Use the distribution data of the target grain structure
                        and the sample grain structure.
        2. 'tstat.sgs': Use the statistics data of the target grain structure
                        and the sample grain structure.
        3. 'tgs.sgs': Use the actual target and sample grain structures.
    '''

    # Valid Grain structure types
    VALgs = ('upxo.mc2d', 'upxo.mc3d',
             'upxo.pv2d', 'upxo.vv3d',
             'upxo.v2d', 'upxo.v3d',
             'image2d', 'image3d',
             'ebsd2d')
    '''
    Explanation of VALgs:
    --------------------------------
    The valid grain structure types for the sample grain structure.
        1. 'upxo.mc2d': 2D Monte-Carlo type.
        2. 'upxo.mc3d': 3D Monte-Carlo type.
        3. 'upxo.pv2d': 2D pixelated Voronoi type.
        4. 'upxo.vv3d': 3D voxellated Voronoi type.
        5. 'upxo.v2d': 2D Voronoi type.
        6. 'upxo.v3d': 3D Voronoi type.
        7. 'image2d': 2D image type.
        8. 'image3d': 3D image type.
        9. 'ebsd2d': 2D EBSD-derived grain structure.
    '''

    def __init__(self, tdist=None, tstat=None, tgs=None,
                 sgs=None, tdim=2, iroute='tgs.sgs',
                 sgstype='upxo.mc2d', tgstype='upxo.mc2d'):
        """
        Initialise a repgen2d instance.

        Prefer the class-method constructors (``from_tgs_sgs``,
        ``from_tdist_sgs``, ``from_tstat_sgs``, ``from_tgs``) over calling
        this directly; they set ``iroute`` and the type strings correctly.

        Parameters
        ----------
        tdist : object, optional
            Distribution data of the target grain structure.
        tstat : object, optional
            Statistics data of the target grain structure.
        tgs : object, optional
            Target grain structure object.
        sgs : object, optional
            Sample grain structure object.
        tdim : int, optional
            Spatial dimensionality of the target data.  Default 2.
        iroute : str, optional
            Representativeness route.  Must be one of ``VALiroutes``
            (``'tdist.sgs'``, ``'tstat.sgs'``, ``'tgs.sgs'``).
            Default ``'tgs.sgs'``.
        sgstype : str, optional
            Type identifier for the sample grain structure.  Must be one
            of ``VALgs``.  Default ``'upxo.mc2d'``.
        tgstype : str, optional
            Type identifier for the target grain structure.  Must be one
            of ``VALgs``.  Default ``'upxo.mc2d'``.

        Raises
        ------
        ValueError
            If *iroute*, *sgstype*, or *tgstype* is not a recognised value.
        """
        if iroute not in self.VALiroutes:
            raise ValueError('Invalid iroute')
        if sgstype not in self.VALgs:
            raise ValueError('Invalid sgstype')
        if tgstype not in self.VALgs:
            raise ValueError('Invalid tgstype')

        self.tdist = tdist
        self.tstat = tstat
        self.tgs = tgs
        self.sgs = sgs
        self.tdim = tdim
        self.iroute = iroute
        self.sgstype = sgstype
        self.tgstype = tgstype

    @classmethod
    def from_tdist_sgs(cls, tdist=None, sgs=None, tdim=2, sgstype='upxo.mc2d'):
        """
        Alternative constructor for creating a RepGen2DMCGS instance using
        distribution data of target grain structure and sample grain structure.

        The sample grain structure can be of the following types:
        - see description of sgstype parameter

        Parameters
        ----------
        tdist: upxo distribution collection object, optional
            Distribution data of the target grain structure. Defaults to None.
        sgs: grain structure data object, optional
            The sample grain structure. Defaults to None.
        tdim: int, optional
            Dimensionality of the target grain structure data used for tdist. Defaults to 2.
        sgstype: str
            Type of the sample grain structure. Must be one of:
            ``'upxo.mc2d'``, ``'upxo.mc3d'``, ``'upxo.pv2d'``,
            ``'upxo.vv3d'``, ``'upxo.v2d'``, ``'upxo.v3d'``,
            ``'image2d'``, ``'image3d'``, ``'ebsd2d'``.

        Returns
        -------
        RepGen2DMCGS
            A new RepGen2DMCGS instance.
        """
        return cls(tdist=tdist, sgs=sgs, tdim=tdim,
                   iroute='tdist.sgs', sgstype='upxo.mc2d')

    @classmethod
    def from_tstat_sgs(cls, tstat=None, sgs=None, tdim=2, sgstype='upxo.mc2d'):
        """
        Alternative constructor for creating a RepGen2DMCGS instance using
        statistics data of target grain structure and sample grain structure.

        The sample grain structure can be of the following types:
        - see description of sgstype parameter

        Parameters
        ----------
        tstat: upxo statistics collection object, optional
            Statistics data of the target grain structure. Defaults to None.
        sgs: grain structure data object, optional
            The sample grain structure. Defaults to None.
        tdim: int, optional
            Dimensionality of the target grain structure data used for tstat. Defaults to 2.
        sgstype: str
            Type of the sample grain structure. Must be one of:
            ``'upxo.mc2d'``, ``'upxo.mc3d'``, ``'upxo.pv2d'``,
            ``'upxo.vv3d'``, ``'upxo.v2d'``, ``'upxo.v3d'``,
            ``'image2d'``, ``'image3d'``, ``'ebsd2d'``.

        Returns
        -------
        RepGen2DMCGS
            A new RepGen2DMCGS instance.
        """
        return cls(tstat=tstat, sgs=sgs, tdim=tdim,
                   iroute='tstat.sgs', sgstype='upxo.mc2d')

    @classmethod
    def from_tgs_sgs(cls, tgs=None, sgs=None,
                     tgstype='upxo.mc2d', sgstype='upxo.mc2d'):
        """
        Alternative constructor for creating a RepGen2DMCGS instance using
        actual target and sample grain structures.

        The sample grain structure can be of the following types:
        - see description of sgstype parameter

        Parameters
        ----------
        tgs: grain structure data object, optional
            The target grain structure. Defaults to None.
        sgs: grain structure data object, optional
            The sample grain structure. Defaults to None.
        tgstype: str, optional
            Type of the target grain structure. Must be one of:
            ``'upxo.mc2d'``, ``'upxo.mc3d'``, ``'upxo.pv2d'``,
            ``'upxo.vv3d'``, ``'upxo.v2d'``, ``'upxo.v3d'``,
            ``'image2d'``, ``'image3d'``, ``'ebsd2d'``.
            Defaults to ``'upxo.mc2d'``.
        sgstype: str
            Type of the sample grain structure. Must be one of:
            ``'upxo.mc2d'``, ``'upxo.mc3d'``, ``'upxo.pv2d'``,
            ``'upxo.vv3d'``, ``'upxo.v2d'``, ``'upxo.v3d'``,
            ``'image2d'``, ``'image3d'``, ``'ebsd2d'``.

        Returns
        -------
        RepGen2DMCGS
            A new RepGen2DMCGS instance.
        """
        return cls(tgs=tgs, sgs=sgs, iroute='tgs.sgs',
                   tgstype=tgstype, sgstype=sgstype)

    @classmethod
    def from_tgs(cls, tgs=None, tgstype='upxo.mc2d', ebsd_file=None):
        """
        Alternative constructor for creating a RepGen2DMCGS instance using
        only a target grain structure. A sample grain structure will be
        generated internally during the representativeness workflow.

        Parameters
        ----------
        tgs : grain structure data object, optional
            The target grain structure. Defaults to None.
        tgstype : str, optional
            Type of the target grain structure. Must be one of:
            ``'upxo.mc2d'``, ``'upxo.mc3d'``, ``'upxo.pv2d'``,
            ``'upxo.vv3d'``, ``'upxo.v2d'``, ``'upxo.v3d'``,
            ``'image2d'``, ``'image3d'``, ``'ebsd2d'``.
            Defaults to ``'upxo.mc2d'``.
        ebsd_file : str or None, optional
            Path to an EBSD file (.ctf, .ang, .h5oina, etc.) associated with
            the target grain structure. Stored for future use only; no parsing
            is performed at construction time. Defaults to None.

        Returns
        -------
        RepGen2DMCGS
            A new RepGen2DMCGS instance with sgs=None.

        Notes
        -----
        ``self.tgs`` holds the target grain structure; ``self.sgs`` is None
        until a sample is generated and assigned.
        ``self.ebsd_file`` stores the EBSD file path (not yet processed).
        """
        obj = cls(tgs=tgs, sgs=None, iroute='tgs.sgs',
                  tgstype=tgstype, sgstype='upxo.mc2d')
        obj.ebsd_file = ebsd_file
        return obj

    # ------------------------------------------------------------------
    # Pixel size
    # ------------------------------------------------------------------

    def set_px_size(self, px_size):
        """
        Set the physical pixel size to use for area and length calculations.
        This value will overwrite the px_size stored inside every grain
        structure object before characterisation runs.

        Parameters
        ----------
        px_size : float
            Physical size of one pixel (same units as the simulation domain).

        Notes
        -----
        Stored in ``self.px_size``.
        """
        if not isinstance(px_size, (int, float)) or px_size <= 0:
            raise ValueError('px_size must be a positive number.')
        self.px_size = float(px_size)

    def _check_px_size(self):
        """Raise if px_size has not been set."""
        try:
            _ = self.px_size
        except AttributeError:
            raise RuntimeError(
                'px_size has not been set. Call set_px_size(px_size) before '
                'running any characterisation.'
            )

    def _apply_px_size_to_gs(self, gs_obj):
        """Overwrite the px_size attribute on a grain structure object."""
        gs_obj.px_size = self.px_size

    def set_ebsd_step(self, step):
        """
        Set the step size to use for EBSD-derived grain structures when
        calculating the ENSD metric. This value will be used to determine the
        neighborhood size for ENSD calculations.

        Parameters
        ----------
        step : float
            Step size (in physical units) to use for EBSD-derived grain structures.

        Notes
        -----
        Stored in ``self.ebsd_step``.
        """
        if not isinstance(step, (int, float)) or step <= 0:
            raise ValueError('step must be a positive number.')
        self.ebsd_step = float(step)

    # ------------------------------------------------------------------
    # Morphological property flags
    # ------------------------------------------------------------------

    def set_mpflags(self, area=True, aspect_ratio=True, perimeter=False,
            perimeter_crofton=False, eq_diameter=False, feret_diameter=False,
            compactness=False, solidity=False, circularity=False, eccentricity=False,
            euler_number=False, moments_hu=False, morph_ori=False, npixels=False,
            npixels_gb=False, gb_length_px=False, major_axis_length=False,
            minor_axis_length=False, bbox=True, bbox_ex=True, char_gb=False,
            char_grain_positions=False, get_grain_coords=True,
            identify_pixel_locations=True, make_skim_prop=True, saa=True,
            use_version=2):
        """
        Set flags controlling which 2D morphological properties are computed
        during characterisation.

        Parameters
        ----------
        area : bool
            Grain area (in px_size² units). Default True.
        aspect_ratio : bool
            Aspect ratio (major / minor axis). Default True.
            Automatically enables major_axis_length and minor_axis_length.
        perimeter : bool
            Grain perimeter. Default False.
        perimeter_crofton : bool
            Crofton perimeter estimate. Default False.
        eq_diameter : bool
            Equivalent circular diameter. Default False.
        feret_diameter : bool
            Feret (maximum caliper) diameter. Default False.
        compactness : bool
            Compactness (4π·area / perimeter²). Default False.
        solidity : bool
            Solidity (area / convex hull area). Default False.
        circularity : bool
            Circularity. Default False.
        eccentricity : bool
            Eccentricity of best-fit ellipse. Default False.
        euler_number : bool
            Euler number (topology). Default False.
        moments_hu : bool
            Hu moments (7 invariants). Default False.
        morph_ori : bool
            Morphological orientation angle. Default False.
        npixels : bool
            Number of pixels per grain. Default False.
        npixels_gb : bool
            Number of grain-boundary pixels. Default False.
        gb_length_px : bool
            Grain boundary length in pixels. Default False.
        major_axis_length : bool
            Major axis length of best-fit ellipse. Default False.
        minor_axis_length : bool
            Minor axis length of best-fit ellipse. Default False.
        bbox : bool
            Axis-aligned bounding box. Default True.
        bbox_ex : bool
            Extended bounding box. Default True.
        char_gb : bool
            Characterise grain boundaries. Default False.
        char_grain_positions : bool
            Characterise grain centroid positions. Default False.
        get_grain_coords : bool
            Store pixel coordinates per grain. Default True.
        identify_pixel_locations : bool
            Identify pixel locations. Default True.
        make_skim_prop : bool
            Store scikit-image region properties. Default True.
        saa : bool
            Store skimage attribute access helper. Default True.
        use_version : int
            Characterisation version (1 or 2). Default 2.

        Notes
        -----
        Stored in ``self.mpflags`` as a flat dict of flag name → bool/int.
        """
        # aspect_ratio requires both axis lengths
        if aspect_ratio:
            major_axis_length = True
            minor_axis_length = True
        self.mpflags = dict(area=area, aspect_ratio=aspect_ratio,
            perimeter=perimeter, perimeter_crofton=perimeter_crofton,
            eq_diameter=eq_diameter, feret_diameter=feret_diameter,
            compactness=compactness, solidity=solidity, circularity=circularity,
            eccentricity=eccentricity, euler_number=euler_number,
            moments_hu=moments_hu, morph_ori=morph_ori,
            npixels=npixels, npixels_gb=npixels_gb, gb_length_px=gb_length_px,
            major_axis_length=major_axis_length, minor_axis_length=minor_axis_length,
            bbox=bbox, bbox_ex=bbox_ex, char_gb=char_gb,
            char_grain_positions=char_grain_positions, get_grain_coords=get_grain_coords,
            identify_pixel_locations=identify_pixel_locations,
            make_skim_prop=make_skim_prop, saa=saa, use_version=use_version,)

    def _char_single_gs(self, gs_obj):
        """
        Run char_morph_2d on a single grain structure object using the flags
        stored in self.mpflags, after applying the authoritative px_size.
        """
        self._apply_px_size_to_gs(gs_obj)
        f = self.mpflags
        gs_obj.char_morph_2d(
            use_version=f['use_version'],
            bbox=f['bbox'],
            bbox_ex=f['bbox_ex'],
            npixels=f['npixels'],
            npixels_gb=f['npixels_gb'],
            identify_pixel_locations=f['identify_pixel_locations'],
            area=f['area'],
            eq_diameter=f['eq_diameter'],
            perimeter=f['perimeter'],
            perimeter_crofton=f['perimeter_crofton'],
            compactness=f['compactness'],
            gb_length_px=f['gb_length_px'],
            aspect_ratio=f['aspect_ratio'],
            solidity=f['solidity'],
            morph_ori=f['morph_ori'],
            circularity=f['circularity'],
            eccentricity=f['eccentricity'],
            feret_diameter=f['feret_diameter'],
            major_axis_length=f['major_axis_length'],
            minor_axis_length=f['minor_axis_length'],
            euler_number=f['euler_number'],
            moments_hu=f['moments_hu'],
            saa=f['saa'],
            char_grain_positions=f['char_grain_positions'],
            char_gb=f['char_gb'],
            make_skim_prop=f['make_skim_prop'],
            get_grain_coords=f['get_grain_coords'],
        )

    # ------------------------------------------------------------------
    # Morphological characterisation orchestrator
    # ------------------------------------------------------------------

    def char_gs(self):
        """
        Characterise the morphological properties of the available grain
        structure objects according to the current iroute:

        - 'tdist.sgs' / 'tstat.sgs': only the sample grain structure (sgs)
          is characterised (tgs is not an UPXO object in these routes).
        - 'tgs.sgs': both tgs and sgs are characterised, provided their
          type is in the UPXO image family ('upxo.mc2d', 'upxo.pv2d',
          'image2d').

        Raises
        ------
        RuntimeError
            If px_size has not been set via set_px_size().
        RuntimeError
            If mpflags have not been set via set_mpflags().

        Notes
        -----
        Results are stored directly on the grain structure objects:
        ``sgs.prop`` (DataFrame), ``sgs.prop_flag``, ``sgs.g[gid].skprop``.
        Same for ``tgs`` when iroute='tgs.sgs'.
        """
        self._check_px_size()
        try:
            _ = self.mpflags
        except AttributeError:
            raise RuntimeError(
                'mpflags not set. Call set_mpflags() before char_gs().'
            )

        UPXO_STANDARD_TYPES = ('upxo.mc2d', 'upxo.pv2d', 'image2d')

        # Always characterise the sample grain structure
        if self.sgs is not None and self.sgstype in UPXO_STANDARD_TYPES:
            self._char_single_gs(self.sgs)
        else:
            warnings.warn('sgs is None or not a supported UPXO 2D type; skipping sgs characterisation.')

        # Characterise target only for the tgs.sgs route
        if self.iroute == 'tgs.sgs':
            if self.tgstype == 'ebsd2d':
                # EBSD route: morphology comes from rechar() -> prop_ebsd.
                # Call compute_ebsd_stats() if prop_ebsd is already populated,
                # otherwise remind the user to call rechar() first.
                try:
                    _ = self.prop_ebsd
                    self.compute_ebsd_stats()
                except AttributeError:
                    warnings.warn(
                        "tgstype='ebsd2d': call rechar() first to populate "
                        "prop_ebsd, then char_gs() will compute stat_ebsd."
                    )
            elif self.tgs is not None and self.tgstype in UPXO_STANDARD_TYPES:
                self._char_single_gs(self.tgs)
            else:
                warnings.warn('tgs is None or not a supported UPXO 2D type; skipping tgs characterisation.')
        else:
            # tdist.sgs / tstat.sgs: tgs is a distribution/statistics object
            warnings.warn(
                f"iroute='{self.iroute}': tgs is not an UPXO grain structure; "
                f"tgs characterisation skipped."
            )

    # ------------------------------------------------------------------
    # Topological (neighbourhood) characterisation orchestrator
    # ------------------------------------------------------------------

    def find_neighbours(self, p=1.0,
                        include_central_grain=False,
                        throw_numba_dict=False,
                        verbosity_nfids=1000):
        """
        Find first-order grain neighbours for all characterised grain
        structure objects using find_neigh_v2().

        Populates gs_obj.neigh_gid on each object.

        Parameters
        ----------
        p : float
            Dilation probability for neighbour detection. Default 1.0.
        include_central_grain : bool
            Include the grain itself in its neighbour list. Default False.
        throw_numba_dict : bool
            Return a numba-typed dict instead of a plain Python dict.
            Default False.
        verbosity_nfids : int
            Print progress every N grains. Default 1000.

        Notes
        -----
        Results stored in ``sgs.neigh_gid`` (and ``tgs.neigh_gid`` when
        iroute='tgs.sgs') as dicts mapping grain ID → list of neighbour IDs.
        """
        UPXO_STANDARD_TYPES = ('upxo.mc2d', 'upxo.pv2d', 'image2d')
        kwargs = dict(p=p, include_central_grain=include_central_grain,
                      throw_numba_dict=throw_numba_dict,
                      verbosity_nfids=verbosity_nfids)

        if self.sgs is not None and self.sgstype in UPXO_STANDARD_TYPES:
            self.sgs.find_neigh_v2(**kwargs)
        else:
            warnings.warn('sgs is None or not a supported UPXO 2D type; skipping neighbour detection for sgs.')

        if self.iroute == 'tgs.sgs':
            if self.tgstype == 'ebsd2d':
                # Neighbourhood for EBSD is already populated by rechar() in
                # self.neigh_gid_ebsd; nothing to do here.
                try:
                    _ = self.neigh_gid_ebsd
                except AttributeError:
                    warnings.warn(
                        "tgstype='ebsd2d': call rechar() first to populate "
                        "neigh_gid_ebsd before using find_neighbours()."
                    )
            elif self.tgs is not None and self.tgstype in UPXO_STANDARD_TYPES:
                self.tgs.find_neigh_v2(**kwargs)
            else:
                warnings.warn('tgs is None or not a supported UPXO 2D type; skipping neighbour detection for tgs.')

    # ------------------------------------------------------------------
    # Network (graph) characterisation orchestrator
    # ------------------------------------------------------------------

    def char_network(self, gsids=None, k_char_level='basic',
                     recalculate_neighbours=False,
                     include_central_grain=False):
        """
        Build and characterise the grain network (graph) for the available
        grain structure objects using the UPXO gsan2d / kmodel pipeline.

        Results are stored in self.gsan_sgs and (if iroute='tgs.sgs')
        self.gsan_tgs as gsan2d objects whose .K dict contains the kmodel.

        Parameters
        ----------
        gsids : list or None
            List of grain structure IDs to pass to initiate_kmodel().
            Defaults to [1].
        k_char_level : str
            Level of graph characterisation: 'none', 'basic', 'simple',
            'full', or 'advanced'. Default 'basic'.
        recalculate_neighbours : bool
            Re-run find_neigh() inside initiate_kmodel(). Default False
            (assumes find_neighbours() has already been called).
        include_central_grain : bool
            Include central grain when recalculating neighbours. Default False.

        Notes
        -----
        Graph objects stored in ``self.gsan_sgs`` and ``self.gsan_tgs``
        (``gsan2d`` instances). The NetworkX graph and metrics are accessible
        via ``self.gsan_sgs.K[gsid]`` (a ``kmodel`` object).
        """
        if gsids is None:
            gsids = [1]

        UPXO_STANDARD_TYPES = ('upxo.mc2d', 'upxo.pv2d', 'image2d')
        kw = dict(gsids=gsids,
                  k_char_level=k_char_level,
                  recalculate_neighbours=recalculate_neighbours,
                  include_central_grain=include_central_grain)

        if self.sgs is not None and self.sgstype in UPXO_STANDARD_TYPES:
            self.gsan_sgs = gsan2d.from_mcgs2d_single(
                self.sgs,
                prechar=True,
                find_neigh=recalculate_neighbours,
                find_neigh_p=1.0,
                find_neigh_include_central_feat=include_central_grain,
            )
            self.gsan_sgs.initiate_kmodel(**kw)
        else:
            warnings.warn('sgs is None or not a supported UPXO 2D type; skipping network characterisation for sgs.')

        if self.iroute == 'tgs.sgs':
            if self.tgstype == 'ebsd2d':
                raise NotImplementedError(
                    "char_network() is not yet supported for tgstype='ebsd2d'. "
                    "Use rechar() which populates neigh_gid_ebsd, then use "
                    "the NetworkX graph tools directly on neigh_gid_ebsd."
                )
            elif self.tgs is not None and self.tgstype in UPXO_STANDARD_TYPES:
                self.gsan_tgs = gsan2d.from_mcgs2d_single(
                    self.tgs,
                    prechar=True,
                    find_neigh=recalculate_neighbours,
                    find_neigh_p=1.0,
                    find_neigh_include_central_feat=include_central_grain,
                )
                self.gsan_tgs.initiate_kmodel(**kw)
            else:
                warnings.warn('tgs is None or not a supported UPXO 2D type; skipping network characterisation for tgs.')

    # ------------------------------------------------------------------
    # EBSD statistics
    # ------------------------------------------------------------------

    def compute_ebsd_stats(self):
        """
        Compute per-property descriptive statistics across all grains in
        ``prop_ebsd`` and store the result in ``stat_ebsd``.

        Statistics computed for every scalar property in ``prop_ebsd``:
        mean, std, min, max, median, 25th percentile (q25), 75th
        percentile (q75), and count (number of grains).

        Non-scalar properties (``centroid``, ``bbox``) are skipped.

        Raises
        ------
        RuntimeError
            If ``prop_ebsd`` has not been populated yet (call ``rechar()``
            first).

        Notes
        -----
        Stored in ``self.stat_ebsd`` as a dict:
        ``{property_name: {'mean': ..., 'std': ..., 'min': ...,
                           'max': ..., 'median': ..., 'q25': ...,
                           'q75': ..., 'count': ...}}``
        """
        try:
            prop = self.prop_ebsd
        except AttributeError:
            raise RuntimeError(
                "prop_ebsd is not set. Call rechar() before compute_ebsd_stats()."
            )

        if not prop:
            self.stat_ebsd = {}
            return

        # Identify scalar property keys from the first grain entry
        first = next(iter(prop.values()))
        scalar_keys = [k for k, v in first.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)]

        stat = {}
        for key in scalar_keys:
            vals = np.array(
                [g[key] for g in prop.values()
                 if isinstance(g[key], (int, float)) and not np.isnan(float(g[key]))],
                dtype=np.float64
            )
            if vals.size == 0:
                stat[key] = {s: float('nan') for s in
                             ('mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'count')}
                continue
            stat[key] = {
                'mean':   float(np.mean(vals)),
                'std':    float(np.std(vals, ddof=1) if vals.size > 1 else 0.0),
                'min':    float(np.min(vals)),
                'max':    float(np.max(vals)),
                'median': float(np.median(vals)),
                'q25':    float(np.percentile(vals, 25)),
                'q75':    float(np.percentile(vals, 75)),
                'count':  int(vals.size),
            }
        self.stat_ebsd = stat
        self._print_ebsd_stats_table()

    def _print_ebsd_stats_table(self):
        """Print ``stat_ebsd`` as a formatted table to stdout."""
        stat = getattr(self, 'stat_ebsd', {})
        if not stat:
            return
        print(f"{'Property':<22}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}  {'count':>6}")
        print('-' * 75)
        for prop, s in stat.items():
            print(f"{prop:<22}  {s['mean']:>10.3f}  {s['std']:>10.3f}  "
                  f"{s['min']:>10.3f}  {s['max']:>10.3f}  {s['count']:>6d}")

    # ------------------------------------------------------------------
    # Distribution visualisation
    # ------------------------------------------------------------------

    def see_distr(self, prop='area', source='ebsd', nbins=40, vis='hist',
                  show_kde=True, show_stats=True, color='steelblue',
                  figsize=(7, 4), log_scale=False, step_size=None):
        """
        Visualise the distribution of a grain morphological property.

        Parameters
        ----------
        prop : str
            Grain property name, e.g. ``'area'``, ``'perimeter'``,
            ``'aspect_ratio'``, ``'eq_diameter'``, ``'solidity'``,
            ``'eccentricity'``, ``'major_axis_length'``,
            ``'minor_axis_length'``, ``'npixels'``.
        source : str
            Which grain structure to draw data from:

            ``'ebsd'``  — ``self.prop_ebsd`` (EBSD target, dict of dicts).
            Requires ``rechar()`` or ``characterise()`` to have been called.

            ``'sgs'``   — ``self.sgs.prop`` (simulated sample grain structure,
            pandas DataFrame). Requires ``char_gs()`` to have been called.

            ``'tgs'``   — ``self.tgs.prop`` (non-EBSD target grain structure,
            pandas DataFrame). Requires ``char_gs()`` to have been called.

        nbins : int
            Number of histogram bins. Default 40.
        vis : str
            Plot style: ``'hist'``, ``'kde'``, or ``'hist_kde'``. Default
            ``'hist'``.
        show_kde : bool
            Overlay KDE on the histogram (``vis='hist'`` only). Default True.
        show_stats : bool
            Annotate mean and median lines. Default True.
        color : str
            Histogram / KDE fill colour. Default ``'steelblue'``.
        figsize : tuple
            Figure (width, height) in inches. Default ``(7, 4)``.
        log_scale : bool
            Log x-axis. Default False.
        step_size : float or None
            Physical pixel size (µm) for x-label annotation. When ``None``
            and ``source='ebsd'``, the value is not shown in the label
            (it is already embedded in the physical-unit values stored in
            ``prop_ebsd``). Default None.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes

        Raises
        ------
        RuntimeError
            If the requested source has not been populated yet.
        KeyError
            If *prop* is not present in the property data.
        ValueError
            If *source* or *vis* is not one of the accepted values.

        Examples
        --------
        >>> fig, ax = rg.see_distr(prop='area', source='ebsd', nbins=40)
        >>> plt.show()
        >>> fig, ax = rg.see_distr(prop='aspect_ratio', source='sgs',
        ...                        vis='hist_kde')
        >>> plt.show()
        """
        from upxo.viz.vizDistr import DistrViz, PROP_UNITS

        if source == 'ebsd':
            try:
                prop_data = self.prop_ebsd
            except AttributeError:
                raise RuntimeError(
                    "prop_ebsd is not populated. "
                    "Call rechar(target='tgs') or rdr.characterise() first."
                )
            data = np.array([v[prop] for v in prop_data.values()])

        elif source in ('sgs', 'tgs'):
            gs_obj = self.sgs if source == 'sgs' else self.tgs
            if gs_obj is None:
                raise RuntimeError(
                    f"self.{source} is None — no grain structure available."
                )
            try:
                df = gs_obj.prop
            except AttributeError:
                raise RuntimeError(
                    f"self.{source}.prop is not populated. "
                    "Call char_gs() first."
                )
            data = df[prop].to_numpy()

        else:
            raise ValueError(
                f"source must be 'ebsd', 'sgs', or 'tgs'; got '{source!r}'"
            )

        label = prop.replace('_', ' ').title()
        dv = DistrViz(data, label=label, units=PROP_UNITS.get(prop, ''))
        return dv.plot(vis=vis, bins=nbins, show_kde=show_kde,
                       show_stats=show_stats, color=color, figsize=figsize,
                       log_scale=log_scale, step_size=step_size)

    # ------------------------------------------------------------------
    # EBSD prop DataFrame builder
    # ------------------------------------------------------------------

    def _build_prop_ebsd_df(self):
        """Build prop_ebsd_df from the current prop_ebsd dict."""
        import pandas as pd
        _EXCLUDE = {'centroid', 'bbox'}
        rows = {}
        for gid, props in self.prop_ebsd.items():
            rows[gid] = {k: v for k, v in props.items() if k not in _EXCLUDE}
        self.prop_ebsd_df = pd.DataFrame.from_dict(rows, orient='index')
        self.prop_ebsd_df.index.name = 'grain_id'

    # ------------------------------------------------------------------
    # Twin-merge: de-twinned EBSD label field and properties
    # ------------------------------------------------------------------

    def build_merged_ebsd_lfi(self, parent_info: dict) -> None:
        """
        Merge twin grains into their parents in a deepcopy of ``lfi_ebsd``,
        re-characterise, store results, and display a side-by-side comparison.

        Twin chains (A→B→C) are resolved so every twin maps to its ultimate
        root parent before the remapping is applied.

        Parameters
        ----------
        parent_info : dict
            Output of :meth:`identify_parent_grains`.  Keyed by CSL label;
            each value must have a ``'pairs_labeled'`` key containing a list
            of ``(parent_gid, twin_gid)`` tuples.

        Populates
        ---------
        lfi_ebsd_merged : np.ndarray
            Copy of ``lfi_ebsd`` with twin pixels relabelled to their parent
            grain ID and then relabelled sequentially to 1…N.
        prop_ebsd_merged : dict
            Per-grain morphological property dict for the merged structure.
        prop_ebsd_merged_df : pd.DataFrame
            DataFrame version of ``prop_ebsd_merged`` (centroid and bbox
            excluded).  Index name: ``'grain_id'``.
        """
        import pandas as pd
        from copy import deepcopy
        from skimage.segmentation import relabel_sequential
        from upxo.interfaces.defdap.ebsd_reader import _char_lfi

        # Step 1 — build resolved twin→parent mapping (handles chains)
        raw_map = {}
        for info in parent_info.values():
            for parent_gid, twin_gid in info['pairs_labeled']:
                raw_map[twin_gid] = parent_gid

        def _resolve(gid, mapping, depth=0):
            if depth > 200 or gid not in mapping:
                return gid
            return _resolve(mapping[gid], mapping, depth + 1)

        resolved_map = {twin: _resolve(parent, raw_map)
                        for twin, parent in raw_map.items()}

        # Step 2 — apply via LUT (Look-Up Table: 1-D array indexed by old grain
        # ID; lut[old_id] = new_id; applied in one vectorised pass O(N_pixels))
        lfi_copy = deepcopy(self.lfi_ebsd)
        max_id   = int(lfi_copy.max())
        lut      = np.arange(max_id + 1, dtype=lfi_copy.dtype)
        for twin_id, parent_id in resolved_map.items():
            if 0 < twin_id <= max_id:
                lut[twin_id] = parent_id
        lfi_copy = lut[lfi_copy]

        # ── provenance: which original grains had twins merged into them ──────
        _parent_gids_with_twins: set[int] = set()
        for lbl_info in parent_info.values():
            for parent, _ in lbl_info['pairs_labeled']:
                _parent_gids_with_twins.add(parent)
        _surviving_gids = set(int(g) for g in np.unique(lfi_copy[lfi_copy > 0]))
        _hosting_gids   = _parent_gids_with_twins & _surviving_gids
        _n_total        = len(_surviving_gids)
        _n_hosting      = len(_hosting_gids)
        self.merge_info = {
            'n_merged_total':        _n_total,
            'n_hosting_grains':      _n_hosting,
            'twin_hosting_fraction': _n_hosting / _n_total if _n_total else float('nan'),
            'hosting_original_gids': _hosting_gids,
        }
        if _n_total:
            print(f'[merge_info]  merged grains: {_n_total}  |  '
                  f'with twins: {_n_hosting}  |  '
                  f'twin-hosting fraction: {_n_hosting / _n_total:.4f}')

        # Step 3 — relabel sequentially and re-characterise
        lfi_copy, _, _ = relabel_sequential(lfi_copy)
        self.lfi_ebsd_merged = lfi_copy

        px = getattr(self, 'px_size', getattr(self, 'ebsd_step', 1.0))
        self.prop_ebsd_merged = _char_lfi(lfi_copy, px_size=px, min_grain_size=0)

        # Step 4 — build DataFrame (exclude centroid and bbox)
        _EXCLUDE = {'centroid', 'bbox'}
        rows = {gid: {k: v for k, v in props.items() if k not in _EXCLUDE}
                for gid, props in self.prop_ebsd_merged.items()}
        self.prop_ebsd_merged_df = pd.DataFrame.from_dict(rows, orient='index')
        self.prop_ebsd_merged_df.index.name = 'grain_id'

        # Step 5 — automatic side-by-side visualisation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        axes[0].imshow(self.lfi_ebsd,        origin='lower', cmap='nipy_spectral')
        axes[0].set_title(f'Original EBSD  ({int(self.lfi_ebsd.max())} grains)')
        axes[0].axis('off')
        axes[1].imshow(self.lfi_ebsd_merged, origin='lower', cmap='nipy_spectral')
        axes[1].set_title(f'Twins merged ({int(self.lfi_ebsd_merged.max())} grains)')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # EBSD re-characterisation from a pre-loaded reader
    # ------------------------------------------------------------------

    def clean_and_rechar_from_rdr(self, rdr, connectivity=4,
                                   min_grain_size=0, verbose=True):
        """
        Clean and re-characterise the EBSD target grain structure from an
        already-loaded (and optionally cropped) ``EBSDReader``, then assign
        all results to the corresponding ``rg.*_ebsd`` slots.

        Internally calls ``rdr.characterise(connectivity, min_grain_size)``
        which performs:

        1. Pixel filling — non-indexed / boundary pixels (values ≤ 0) are
           assigned to the spatially largest neighbouring grain and their
           orientations are updated (cleaning step).
        2. Morphological characterisation — per-grain area, perimeter,
           aspect ratio, etc. via ``skimage.measure.regionprops``.
           Grains smaller than ``min_grain_size`` pixels are excluded from
           ``prop_ebsd``.
        3. Neighbourhood graph — first-order grain adjacency via cc3d.

        This is the preferred path when ``rdr`` has been built and cropped
        outside ``rechar()`` — it avoids reloading the file from disk.

        Parameters
        ----------
        rdr : EBSDReader
            A populated (and optionally cropped) ``EBSDReader`` instance.
        connectivity : int
            Pixel connectivity for cleaning and neighbour detection.
            4 (edge-only) or 8 (edge + corner). Default 4.
        min_grain_size : int, optional
            Minimum grain size in pixels.  Grains with fewer pixels are
            excluded from ``prop_ebsd`` after characterisation. Default 0
            (all grains included).
        verbose : bool
            Print a timing and grain-count summary on completion. Default True.

        Notes
        -----
        Populated slots after return: ``lfi_ebsd``, ``euler_ebsd``,
        ``quat_ebsd``, ``neigh_gid_ebsd``, ``prop_ebsd``.
        """
        import time, warnings
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            result = rdr.characterise(connectivity=connectivity,
                                      min_grain_size=min_grain_size)
        elapsed = time.perf_counter() - t0

        self.lfi_ebsd       = result['lfi']
        self.euler_ebsd     = result['euler']
        self.quat_ebsd      = result['quat']
        self.neigh_gid_ebsd = result['neigh_gid']
        self.prop_ebsd      = result['prop']
        self._build_prop_ebsd_df()

        if verbose:
            print(f'clean_and_rechar_from_rdr() completed in {elapsed:.1f} s')
            print(f'Non-positive pixels remaining : {int(np.sum(self.lfi_ebsd <= 0))}')
            print(f'Grains after cleaning         : {int(self.lfi_ebsd.max())}')
            print(f'Neighbour entries             : {len(self.neigh_gid_ebsd)}')
            print(f'prop_ebsd grain count         : {len(self.prop_ebsd)}')

    # ------------------------------------------------------------------
    # Quick re-characterisation (cc3d)
    # ------------------------------------------------------------------

    def rechar(self, target='tgs', connectivity=4,
               k_char_level='basic', gsids=None,
               min_grain_size=10, misori_tol=10):
        """
        Quickly re-detect grains, re-compute first-order neighbourhood, and
        build the grain network model.

        For standard UPXO grain structure types (``'upxo.mc2d'``,
        ``'upxo.pv2d'``, ``'image2d'``) this delegates to cc3d:
        - ``upxo.gsdataops.grid_ops.detect_grains_cc3d`` — grain labelling
        - ``upxo.gsdataops.gid_ops.find_neighs2d`` — neighbourhood
        - ``upxo.analysis.analysis2d.gsan2d`` + ``initiate_kmodel`` — network

        For ``'ebsd2d'`` (when ``tgstype='ebsd2d'`` and ``target`` includes
        ``'tgs'``) the EBSD file stored in ``self.ebsd_file`` is loaded via
        ``EBSDReader.from_file()``.  The extracted arrays are stored on
        ``self`` and the neighbourhood is computed from ``lfi_ebsd``.
        Full ``gsan2d`` network characterisation is not yet supported for
        the ``'ebsd2d'`` route.

        Parameters
        ----------
        target : str
            Which grain structure(s) to re-characterise:
            ``'sgs'``, ``'tgs'``, or ``'both'``. Default ``'tgs'``.
        connectivity : int
            cc3d connectivity. Valid 2D values: 4 (edge-only) or 8
            (edge+corner). Default 4.
        k_char_level : str
            Level of graph characterisation passed to ``initiate_kmodel``:
            ``'none'``, ``'basic'``, ``'simple'``, ``'full'``, or
            ``'advanced'``. Default ``'basic'``.
        gsids : list or None
            Grain structure IDs passed to ``initiate_kmodel``. Default [1].
        min_grain_size : int, optional
            Minimum grain size in pixels for EBSD grain detection.
            Passed to ``EBSDReader.from_file()``. Default 10.
        misori_tol : float, optional
            Misorientation tolerance in degrees for EBSD grain boundary
            detection. Passed to ``EBSDReader.from_file()``. Default 10.

        Raises
        ------
        ValueError
            If ``target`` is not one of the accepted values.
        ValueError
            If ``connectivity`` is not 4 or 8.
        RuntimeError
            If ``tgstype='ebsd2d'`` but ``self.ebsd_file`` is not set.

        Notes
        -----
        **Standard route** — results written onto each grain structure object:
        ``gs_obj.lgi``, ``gs_obj.n_grains``, ``gs_obj.neigh_gid``.
        Network models in ``self.gsan_sgs`` / ``self.gsan_tgs``.

        **EBSD route** — arrays stored on ``self``:
        ``self.lfi_ebsd`` (int32, ny×nx),
        ``self.euler_ebsd`` (float64, ny×nx×3, radians),
        ``self.quat_ebsd`` (float64, ny×nx×4),
        ``self.neigh_gid_ebsd`` (dict grain_id → list of neighbour IDs).
        """
        from upxo.gsdataops import grid_ops as gridOps
        from upxo.gsdataops.gid_ops import find_neighs2d

        VALID_TARGETS = ('sgs', 'tgs', 'both')
        if target not in VALID_TARGETS:
            raise ValueError(f"target must be one of {VALID_TARGETS}, got '{target}'.")
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8 for 2D cc3d.")

        if gsids is None:
            gsids = [1]

        UPXO_STANDARD_TYPES = ('upxo.mc2d', 'upxo.pv2d', 'image2d')
        UPXO_2D_TYPES = UPXO_STANDARD_TYPES + ('ebsd2d',)

        def _rechar_one(gs_obj):
            """Re-detect grains and build a gsan2d network for one GS object."""
            lfi, N, _ = gridOps.detect_grains_cc3d(
                gs_obj.s, connectivity=connectivity, delta=0,
                return_num_grains=True)
            gs_obj.lgi = lfi
            gs_obj.n_grains = N
            gs_obj.neigh_gid = find_neighs2d(lfi, conn=connectivity)
            gsan = gsan2d.from_mcgs2d_single(
                gs_obj,
                prechar=True,
                find_neigh=False,
            )
            gsan.initiate_kmodel(gsids=gsids, k_char_level=k_char_level,
                                 recalculate_neighbours=False)
            return gsan

        def _rechar_ebsd():
            """Load and characterise the EBSD file, populating the ebsd slots."""
            from upxo.interfaces.defdap.ebsd_reader import EBSDReader
            try:
                ebsd_file = self.ebsd_file
            except AttributeError:
                ebsd_file = None
            if not ebsd_file:
                raise RuntimeError(
                    "tgstype='ebsd2d' but ebsd_file is not set. "
                    "Pass ebsd_file when constructing via from_tgs()."
                )
            rdr = EBSDReader.from_file(
                ebsd_file,
                min_grain_size=min_grain_size,
                misori_tol=misori_tol,
            )
            # Delegate the full pipeline (fill pixels + morphology + neighbours)
            # to EBSDReader.characterise() — all EBSD logic lives there.
            result = rdr.characterise(connectivity=connectivity)
            self.lfi_ebsd      = result['lfi']
            self.euler_ebsd    = result['euler']
            self.quat_ebsd     = result['quat']
            self.neigh_gid_ebsd = result['neigh_gid']
            self.prop_ebsd     = result['prop']
            self._build_prop_ebsd_df()
            warnings.warn(
                "ebsd2d route: lfi_ebsd, euler_ebsd, quat_ebsd, "
                "neigh_gid_ebsd and prop_ebsd have been populated. "
                "gsan2d network characterisation is not yet supported "
                "for the ebsd2d route."
            )

        do_sgs = target in ('sgs', 'both')
        do_tgs = target in ('tgs', 'both')

        if do_sgs:
            if self.sgs is not None and self.sgstype in UPXO_STANDARD_TYPES:
                self.gsan_sgs = _rechar_one(self.sgs)
            else:
                warnings.warn('sgs is None or not a supported UPXO 2D type; skipping rechar for sgs.')

        if do_tgs and self.iroute == 'tgs.sgs':
            if self.tgstype == 'ebsd2d':
                _rechar_ebsd()
            elif self.tgs is not None and self.tgstype in UPXO_STANDARD_TYPES:
                self.gsan_tgs = _rechar_one(self.tgs)
            else:
                warnings.warn('tgs is None or not a supported UPXO 2D type; skipping rechar for tgs.')

    def compute_mdf_ebsd(self):
        """
        Compute the misorientation distribution function (MDF) from the EBSD
        dataset attached to this repgen2d instance and display it.

        Prerequisites
        -------------
        ``clean_and_rechar_from_rdr`` (or ``rechar``) must have been called
        with ``tgstype='ebsd2d'`` beforehand so that ``self.lfi_ebsd``,
        ``self.quat_ebsd``, and ``self.neigh_gid_ebsd`` are populated.

        Returns
        -------
        mdf : np.ndarray
            1-D array of grain-boundary misorientation angles (degrees) for
            every unique neighbour pair found in the EBSD map.
        peaks : dict
            Peak-detection results with keys ``'peak_indices'``,
            ``'peak_labels'``, and ``'peak_angles'``.  Pass directly to
            ``nbWidgets.mdf_peak_selector`` for interactive peak selection.

        Side Effects
        ------------
        Calls ``ebsdviz.plot_mdf`` and ``plt.show()``, rendering the MDF
        histogram with detected peaks in the current matplotlib backend.
        """
        from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats
        mdf = compute_mdf_from_quats(self.lfi_ebsd, self.quat_ebsd, self.neigh_gid_ebsd)
        # -------------------------------------------------
        from upxo.xtalphy.crystal_orientation import detect_mdf_peaks
        peaks = detect_mdf_peaks(mdf)
        print(f"{'Detected peaks':─^55}")
        for label, (csl_name, delta) in zip(peaks['peak_labels'], peaks['csl_nearest']):
            print(f'  {label}')
        # ------------------------------------------------
        ebsdviz.plot_mdf(mdf, peaks)
        plt.show()
        # -------------------------------------------------
        return mdf, peaks

    def segregate_csl_pairs(self, mdf, selected_peaks, csl, csl_tol):
        """
        Segregate grain-boundary pairs by coincidence site lattice (CSL) type.

        Classifies every unique grain-boundary pair in the EBSD dataset into
        its nearest CSL relationship based on the peaks the user retained via
        the interactive widget.  Delegates to
        ``crystal_orientation.segregate_csl_pairs`` and prints a summary table.

        Parameters
        ----------
        mdf : dict
            Output of ``compute_mdf_ebsd()`` (or directly from
            ``crystal_orientation.compute_mdf_from_quats()``).  Must contain
            ``'pairs'`` (N, 2) and ``'miso_deg'`` (N,).
        selected_peaks : dict
            Output of ``select_mdf_peaks()`` after the user has confirmed
            their selection.  Keys: ``'angles'`` (list of float) and
            ``'indices'`` (list of int).
        csl : dict or None
            ``{label: reference_angle_degrees}`` mapping.  Enter
            ``peaks['csl']`` (the CSL dict used during MDF peak detection).
            Pass ``None`` to fall back to the built-in ``CUBIC_CSL`` table.
        csl_tol : float
            Tolerance in degrees; pairs within this distance of a CSL
            reference angle are included in that category.  Input
            ``peaks['csl_tol']`` from the peak-detection step.

        Returns
        -------
        csl_grains : dict
            A dict keyed by CSL label (e.g. ``'Σ3'``); each value is a dict
            with:

            - ``csl_angle``  : float — reference misorientation angle (degrees)
            - ``pairs``      : ndarray (M, 2) — grain-ID pairs at this boundary type
            - ``miso_deg``   : ndarray (M,) — disorientation of those pairs (degrees)
            - ``grains_A``   : ndarray — unique grain IDs on one side of the boundary
            - ``grains_B``   : ndarray — unique grain IDs on the other side
            - ``grains_all`` : ndarray — all unique grain IDs touching this boundary type

        Side Effects
        ------------
        Prints a summary table with columns ``CSL type``, ``ref °``,
        ``pairs``, and ``grains`` for every CSL category found.
        """
        from upxo.xtalphy.crystal_orientation import segregate_csl_pairs

        # Segregate using selected_peaks from the widget cell
        csl_grains = segregate_csl_pairs(mdf, selected_peaks, csl=csl, csl_tol=csl_tol)

        # Summary table
        print(f"{'CSL type':<18} {'ref °':>6}  {'pairs':>6}  {'grains':>7}")
        print('─' * 42)
        for lbl, info in csl_grains.items():
            print(f"{lbl:<18} {info['csl_angle']:>6.2f}  "
                f"{len(info['pairs']):>6}  {len(info['grains_all']):>7}")
            
        return csl_grains

    def select_mdf_peaks(self, peaks):
        """
        Launch the interactive ipywidgets checklist for selecting MDF peaks.

        Wraps ``nbWidgets.mdf_peak_selector`` (re-exported via
        ``ebsdviz``) to display a checkbox panel in Jupyter so the user
        can choose which detected MDF peaks to retain for downstream CSL
        segregation.

        Parameters
        ----------
        peaks : dict
            Output of ``compute_mdf_ebsd()`` (or directly from
            ``crystal_orientation.detect_mdf_peaks()``).  Must contain
            ``'peak_indices'``, ``'peak_labels'``, and ``'peak_angles'``.

        Returns
        -------
        selected_peaks : dict
            Live state dict with keys ``'angles'`` (list of float) and
            ``'indices'`` (list of int), pre-populated with all peaks and
            updated in-place when the user clicks **Confirm selection**.
            Pass this dict to ``segregate_csl_pairs()`` or
            ``self.segregate_csl_pairs()``.

        Notes
        -----
        Must be called inside a Jupyter cell.  The returned dict is
        updated asynchronously on button click; read it in a subsequent
        cell after confirming the selection.
        """
        selected_peaks = ebsdviz.mdf_peak_selector(peaks)
        return selected_peaks
    
    def plot_mdf_selected(self, mdf, peaks, selected_peaks):
        """
        Replot the MDF histogram and KDE highlighting only the user-selected
        peaks; unselected histogram bins are greyed out.
    
        Wraps ``ebsdviz.plot_mdf_selected``, passing the ``mdf`` and
        ``peaks`` dicts that must exist in the calling notebook scope
        (produced by ``compute_mdf_ebsd()``).

        Parameters
        ----------
        mdf : ndarray
            The MDF histogram data.
        peaks : dict
            Output of ``compute_mdf_ebsd()`` (or directly from
            ``crystal_orientation.detect_mdf_peaks()``).  Must contain
            ``'peak_indices'``, ``'peak_labels'``, and ``'peak_angles'``.
        selected_peaks : dict
            Output of ``select_mdf_peaks()`` after the user has confirmed
            their selection via the widget.  Keys: ``'angles'`` (list of
            float) and ``'indices'`` (list of int).

        Returns
        -------
        None
            The figure is rendered via ``plt.show()``.  Selected peak
            angles are printed to stdout.

        Notes
        -----
        ``mdf`` and ``peaks`` must be defined in the calling scope
        (typically the notebook cell that called ``compute_mdf_ebsd()``).
        """
        ebsdviz.plot_mdf_selected(mdf, peaks, selected_peaks)
        plt.show()
        print(f'Selected peak angles : {selected_peaks["angles"]}')

    def see_csl_grain_map(self, csl_grains, **kwargs):
        """
        Render a colour-coded grain map showing CSL boundary participation
        for the EBSD target grain structure.

        Each CSL type present in *csl_grains* is assigned a distinct colour.
        Grains touching boundaries of more than one CSL type are shown with
        blended colours; non-CSL grains are displayed in semi-transparent
        light grey.  Wraps ``ebsdviz.plot_csl_grain_map`` using
        ``self.lfi_ebsd`` as the grain label field.

        Parameters
        ----------
        csl_grains : dict
            Output of ``segregate_csl_pairs()``.  Keys are CSL labels
            (e.g. ``'Σ3'``); each value must contain ``'grains_all'``,
            ``'n_pairs'``, and ``'n_grains'``.

        Returns
        -------
        None
            The figure is rendered via ``plt.show()``.

        Prerequisites
        -------------
        ``clean_and_rechar_from_rdr()`` must have been called so that
        ``self.lfi_ebsd`` is populated, and ``segregate_csl_pairs()``
        must have been called to produce *csl_grains*.
        """
        ebsdviz.plot_csl_grain_map(self.lfi_ebsd, csl_grains,
            figsize=kwargs.get('figsize', (6, 6)),
            dpi=kwargs.get('dpi', 140),
            suptitle=kwargs.get('suptitle', 'CSL grain participation map — EBSD target'),)
        plt.show()

    def compute_csl_volume_fractions(self, csl_grains):
        """
        Compute and display the area fraction of the EBSD map occupied by
        grains participating in each CSL boundary type.

        A grain is counted for a CSL category if it appears in at least one
        boundary of that type (i.e. it is in ``csl_grains[label]['grains_all']``).
        Grains can contribute to multiple categories, so fractions need not
        sum to 1.  Delegates to
        ``crystal_orientation.csl_volume_fractions``, then prints a summary
        table and renders a bar chart of volume fractions.

        Parameters
        ----------
        csl_grains : dict
            Output of ``segregate_csl_pairs()``.  Keys are CSL labels
            (e.g. ``'Σ3'``); each value must contain ``'grains_all'``,
            ``'csl_angle'``, ``'n_grains'``, and ``'n_pairs'``.

        Returns
        -------
        None
            Results are printed as a table and displayed as a bar chart
            via ``plt.show()``.  The ``vf`` dict (keyed by CSL label)
            contains per-label dicts with:

            - ``n_pixels``   : int — pixels occupied by CSL grains
            - ``vf_indexed`` : float — fraction of indexed pixels
            - ``vf_total``   : float — fraction of all pixels (incl. unindexed)
            - ``csl_angle``  : float — reference CSL angle (degrees)
            - ``n_grains``   : int — number of grains in this CSL type

        Prerequisites
        -------------
        ``clean_and_rechar_from_rdr()`` must have been called so that
        ``self.lfi_ebsd`` is populated, and ``segregate_csl_pairs()``
        must have been called to produce *csl_grains*.
        """
        from upxo.xtalphy.crystal_orientation import csl_volume_fractions
        vf = csl_volume_fractions(self.lfi_ebsd, csl_grains)
        # ── Table ─────────────────────────────────────────────────────────────────────
        print(f"{'CSL type':<18} {'ref °':>6}  {'grains':>7}  "
            f"{'pixels':>8}  {'VF (indexed)':>13}  {'VF (total)':>11}")
        print('─' * 70)
        for lbl, v in vf.items():
            print(f"{lbl:<18} {v['csl_angle']:>6.2f}  {v['n_grains']:>7}  "
                f"{v['n_pixels']:>8}  {v['vf_indexed']:>12.2%}  {v['vf_total']:>10.2%}")
        # ── Bar chart ─────────────────────────────────────────────────────────────────
        import matplotlib.colors as mcolors
        palette   = list(mcolors.TABLEAU_COLORS.values())
        labels    = list(vf.keys())
        vf_vals   = [vf[l]['vf_indexed'] * 100 for l in labels]
        colors    = [palette[i % len(palette)] for i in range(len(labels))]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(labels, vf_vals, color=colors, edgecolor='k', linewidth=0.5)
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
        ax.set_ylabel('Volume fraction of indexed area (%)')
        ax.set_title('CSL twinned volume fractions — EBSD target')
        ax.set_ylim(0, max(vf_vals) * 1.25)
        plt.tight_layout()
        plt.show()
        # ── Return the raw volume fraction data for any downstream use ─────────────────
        return vf
    
    def identify_parent_grains(self, csl_grains, **kwargs):
        """
        Identify parent, twin, and intermediate grains for each CSL boundary
        type in the EBSD target and display a summary.

        Uses grain area (from ``self.prop_ebsd``) to label each pair in
        *csl_grains* as parent (larger grain) vs twin/child (smaller grain).
        A grain's net role is determined across all pairs it participates in:

        - **pure_parents** — larger grain in every one of their pairs
        - **pure_twins** — smaller grain in every one of their pairs
        - **intermediates** — parent in some pairs, twin in others (twin chains)

        Delegates computation to
        ``crystal_orientation.identify_parent_grains``, then calls
        ``ebsdviz.print_parent_grain_summary`` and
        ``ebsdviz.plot_parent_grain_summary``.  Optionally overlays a
        spatial parent/twin map on ``self.lfi_ebsd``.

        Parameters
        ----------
        csl_grains : dict
            Output of ``segregate_csl_pairs()``.  Keys are CSL labels;
            each value must contain ``'pairs'`` and ``'csl_angle'``.
        **kwargs
            Optional display controls:

            - ``figsize`` : tuple — figure size for the summary bar chart
              (default ``(8, 4)``)
            - ``dpi`` : int — DPI for the summary chart (default ``120``)
            - ``title`` : str — title for the summary chart
            - ``plot_parent_twin_map`` : bool — if ``True``, also render a
              spatial grain map coloured by role (default ``False``)
            - ``map_figsize`` : tuple — figure size for the spatial map
              (default ``(6, 6)``)
            - ``map_dpi`` : int — DPI for the spatial map (default ``140``)
            - ``map_suptitle`` : str — super-title for the spatial map

        Returns
        -------
        parent_info : dict
            Keyed by CSL label; each value is a dict with:

            - ``pairs_labeled``   : list of ``(parent_gid, twin_gid)`` tuples
            - ``all_parents``     : ndarray — grains that are parent in ≥1 pair
            - ``all_twins``       : ndarray — grains that are twin in ≥1 pair
            - ``pure_parents``    : ndarray — grains that are only ever a parent
            - ``pure_twins``      : ndarray — grains that are only ever a twin
            - ``intermediates``   : ndarray — grains in both roles (twin chains)
            - ``n_pure_parents``  : int
            - ``n_pure_twins``    : int
            - ``n_intermediates`` : int
            - ``csl_angle``       : float — reference CSL angle (degrees)

        Prerequisites
        -------------
        ``clean_and_rechar_from_rdr()`` must have been called so that
        ``self.prop_ebsd`` (grain area data) and ``self.lfi_ebsd``
        (if ``plot_parent_twin_map=True``) are populated.
        """
        from upxo.xtalphy.crystal_orientation import identify_parent_grains
        parent_info = identify_parent_grains(csl_grains, self.prop_ebsd)

        from upxo.viz.ebsdviz import print_parent_grain_summary, plot_parent_grain_summary
        print_parent_grain_summary(parent_info, csl_grains)
        fig, ax = plot_parent_grain_summary(parent_info,
                        figsize=kwargs.get('figsize', (8, 4)), dpi=kwargs.get('dpi', 120),
            title=kwargs.get('title', 'Parent / twin / intermediate grain counts per CSL type'),)
        
        if kwargs.get('plot_parent_twin_maps', False):
            self.plot_parent_twin_map(parent_info, **kwargs)

        if kwargs.get('plot_combined_parent_twin_map', False):
            self.plot_combined_parent_twin_map(parent_info, **kwargs)

        return parent_info

    def compute_ebsd_tvf(
            self,
            parent_info: dict,
            csl_label: str | None = None,
    ) -> dict:
        """
        Compute EBSD twin area fraction broken down by grain role.

        Uses :func:`~upxo.xtalphy.crystal_orientation.classify_grain_roles_extended`
        to split ``pure_twins`` into **primary twins** (1st generation) and
        **secondary twins** (2nd generation / twins-of-twins).

        Parameters
        ----------
        parent_info : dict
            Output of :meth:`identify_parent_grains`.
        csl_label : str or None
            CSL type to analyse (e.g. ``'Σ3'``).  When ``None`` (default),
            the first key in ``parent_info`` is used automatically.  Pass
            ``list(parent_info.keys())`` to see available labels.

        Returns
        -------
        dict
            ``'csl_label'``          — the CSL label actually used
            ``'total_area'``         — sum of all grain areas in ``prop_ebsd``
            ``'pure_parent_frac'``   — area fraction of pure-parent grains
            ``'primary_twin_frac'``  — area fraction of 1st-gen twin grains
            ``'secondary_twin_frac'``— area fraction of 2nd-gen twin grains
            ``'intermediate_frac'``  — area fraction of intermediate grains
            ``'overall_twin_frac'``  — total twin area fraction
                                       (primary + secondary + intermediates)
            ``'extended_info'``      — full output of
                                       ``classify_grain_roles_extended``
        """
        from upxo.xtalphy.crystal_orientation import classify_grain_roles_extended

        ext = classify_grain_roles_extended(parent_info)

        if csl_label is None:
            csl_label = next(iter(ext))
            print(f'compute_ebsd_tvf: auto-selected CSL label  → "{csl_label}"')
            print(f'  Available labels: {list(ext.keys())}')
        elif csl_label not in ext:
            raise KeyError(
                f'csl_label "{csl_label}" not found in parent_info.  '
                f'Available: {list(ext.keys())}'
            )

        info = ext[csl_label]

        def _area(gids):
            return sum(self.prop_ebsd[g]['area']
                       for g in gids if g in self.prop_ebsd)

        total   = sum(g['area'] for g in self.prop_ebsd.values())
        pp_area = _area(info['pure_parents'])
        pt_area = _area(info['primary_twins'])
        st_area = _area(info['secondary_twins'])
        im_area = _area(info['intermediates'])

        role_area = pp_area + pt_area + st_area + im_area
        return {
            'csl_label':            csl_label,
            'total_area':           total,
            'pure_parent_frac':     pp_area / total if total else float('nan'),
            'primary_twin_frac':    pt_area / total if total else float('nan'),
            'secondary_twin_frac':  st_area / total if total else float('nan'),
            'intermediate_frac':    im_area / total if total else float('nan'),
            'overall_twin_frac':   (pt_area + st_area + im_area) / total
                                    if total else float('nan'),
            'non_role_frac':       (total - role_area) / total
                                    if total else float('nan'),
            'extended_info':        ext,
        }

    def allocate_mc_twin_hosts(
            self,
            cntr,
            mc_slices: list,
            target_fraction: float | None = None,
            size_criterion: str = 'area',
            min_host_px: int = 0,
    ) -> dict:
        """
        Designate which MC grains will host twin regions, matching the EBSD
        twin-hosting fraction.

        The target fraction is taken from
        ``self.merge_info['twin_hosting_fraction']`` (computed automatically
        by :meth:`build_merged_ebsd_lfi`) unless overridden by
        ``target_fraction``.

        Grains are ranked by ``size_criterion`` in descending order — the
        largest grains are designated hosts first, reflecting that larger
        grains are more likely to nucleate and retain twins.

        Parameters
        ----------
        cntr : MC_GS_Container2d
        mc_slices : list
            MC time-slice keys to allocate.
        target_fraction : float or None
            Fraction of MC grains to designate as twin hosts.  Defaults to
            ``self.merge_info['twin_hosting_fraction']``.
        size_criterion : str
            Column in ``cntr.gsset[k].prop`` used to rank grains.
            Default ``'area'``.
        min_host_px : int
            Grains with fewer pixels than this are excluded from host
            consideration before the ranking/selection step.  Default 0
            (no filtering, backward-compatible).

        Returns
        -------
        dict
            ``{slice_key: {
                'host_gids':       np.ndarray,
                'non_host_gids':   np.ndarray,
                'n_total':         int,
                'n_hosts':         int,
                'actual_fraction': float,
                'target_fraction': float,
            }}``
        """
        import numpy as np

        if target_fraction is None:
            if not hasattr(self, 'merge_info') or self.merge_info is None:
                raise RuntimeError(
                    'merge_info not set — call build_merged_ebsd_lfi() first, '
                    'or supply target_fraction explicitly.'
                )
            target_fraction = self.merge_info['twin_hosting_fraction']

        result = {}
        print(f'Twin-hosting target fraction : {target_fraction:.4f}')
        hdr = f'{"Slice":>10}  {"n_total":>8}  {"n_hosts":>8}  {"actual_frac":>12}  {"excluded":>9}'
        print(hdr)
        print('─' * len(hdr))

        for k in mc_slices:
            gs = cntr.gsset.get(k)
            if gs is None:
                continue
            prop = gs.prop
            if size_criterion not in prop.columns:
                raise KeyError(
                    f"'{size_criterion}' not in prop columns for slice {k}. "
                    f"Available: {list(prop.columns)}"
                )
            if min_host_px > 0 and 'npixels' in prop.columns:
                prop_elig  = prop[prop['npixels'] >= min_host_px]
                n_excluded = len(prop) - len(prop_elig)
            else:
                prop_elig  = prop
                n_excluded = 0
            sorted_gids = prop_elig[size_criterion].sort_values(ascending=False).index.to_numpy()
            n_total     = len(sorted_gids)
            n_hosts     = max(1, round(n_total * target_fraction))
            host_gids   = sorted_gids[:n_hosts]
            non_host    = sorted_gids[n_hosts:]
            actual_frac = n_hosts / n_total
            result[k] = {
                'host_gids':       host_gids,
                'non_host_gids':   non_host,
                'n_total':         n_total,
                'n_hosts':         n_hosts,
                'n_excluded':      n_excluded,
                'actual_fraction': actual_frac,
                'target_fraction': target_fraction,
            }
            print(f'{k:>10}  {n_total:>8}  {n_hosts:>8}  {actual_frac:>12.4f}  {n_excluded:>9}')

        return result

    def plot_mc_host_properties(
            self,
            cntr,
            mc_twin_hosts: dict,
            parent_info: dict,
            props: list | None = None,
            csl_label: str | None = None,
            ncols_spatial: int = 2,
            figsize_spatial: tuple = (10, 4),
            figsize_prop: tuple = (10, 4),
            dpi: int = 100,
            fontsize: float = 9.0,
    ) -> None:
        """
        Visualise MC host grains and compare their morphological property
        distributions against EBSD pure-parent grain properties.

        Produces two figures:

        1. **Spatial maps** — one subplot per selected MC slice; host grains
           in steel-blue, non-host grains in light-grey, background white.
        2. **Property distributions** — one subplot per property; KDE of MC
           host grain values (one coloured line per slice) overlaid on the
           EBSD pure-parent KDE (dashed black line).

        Parameters
        ----------
        cntr : MC_GS_Container2d
            Container with ``.gsset`` dict (same object passed to
            :meth:`sgs_mcgs2Gen`).
        mc_twin_hosts : dict
            Output of :meth:`allocate_mc_twin_hosts`.
        parent_info : dict
            Output of :meth:`identify_parent_grains`.
        props : list of str or None
            Properties to compare.  Defaults to
            ``['area', 'aspect_ratio', 'eq_diameter']``.
        csl_label : str or None
            Which CSL key to use when extracting EBSD pure parents.
            Auto-selects first key when *None*.
        ncols_spatial : int
            Columns in the spatial-map figure.
        figsize_spatial, figsize_prop : tuple
            Figure sizes in inches.
        dpi : int
        fontsize : float
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        from scipy.stats import gaussian_kde

        if props is None:
            props = ['area', 'aspect_ratio', 'eq_diameter']

        csl_key = csl_label or next(iter(parent_info))
        ebsd_pp_gids = {int(g) for g in parent_info[csl_key]['pure_parents']}
        ebsd_pp_df = self.prop_ebsd_df.loc[
            self.prop_ebsd_df.index.isin(ebsd_pp_gids)
        ]

        slices = sorted(mc_twin_hosts.keys())
        n_slices = len(slices)

        # ── Figure 1: Spatial maps ─────────────────────────────────────────
        nrows_s = max(1, (n_slices + ncols_spatial - 1) // ncols_spatial)
        fig_s, axs_s = plt.subplots(
            nrows_s, ncols_spatial,
            figsize=figsize_spatial, dpi=dpi, squeeze=False,
        )
        cmap_host = mcolors.ListedColormap(['white', '#b0b0b0', '#4682b4'])

        for idx, sk in enumerate(slices):
            row, col = divmod(idx, ncols_spatial)
            ax = axs_s[row][col]
            gs = cntr.gsset[sk]
            lfi = gs.lfi
            host_set = {int(g) for g in mc_twin_hosts[sk]['host_gids']}
            vis = np.zeros_like(lfi, dtype=np.int8)
            vis[lfi > 0] = 1
            for gid in host_set:
                vis[lfi == gid] = 2
            ax.imshow(vis, cmap=cmap_host, vmin=0, vmax=2,
                      interpolation='nearest')
            ax.set_title(
                f'Slice {sk}  |  hosts={len(host_set)}',
                fontsize=fontsize,
            )
            ax.axis('off')

        for idx in range(n_slices, nrows_s * ncols_spatial):
            row, col = divmod(idx, ncols_spatial)
            axs_s[row][col].axis('off')

        fig_s.suptitle(
            'MC host grains (blue) vs non-host (grey)', fontsize=fontsize + 1
        )
        fig_s.tight_layout()
        plt.show()

        # ── Figure 2: Property distributions ──────────────────────────────
        valid_props = [p for p in props if p in ebsd_pp_df.columns]
        if not valid_props:
            return
        ncols_p = min(3, len(valid_props))
        nrows_p = max(1, (len(valid_props) + ncols_p - 1) // ncols_p)
        fig_p, axs_p = plt.subplots(
            nrows_p, ncols_p,
            figsize=figsize_prop, dpi=dpi, squeeze=False,
        )
        colours = plt.cm.tab10(range(n_slices))

        for pidx, p in enumerate(valid_props):
            row_p, col_p = divmod(pidx, ncols_p)
            ax = axs_p[row_p][col_p]

            ebsd_vals = ebsd_pp_df[p].dropna().values
            all_vals = list(ebsd_vals)
            for sk in slices:
                gs = cntr.gsset[sk]
                if p not in gs.prop.columns:
                    continue
                # gs.prop uses 0-based RangeIndex; host_gids are 1-based grain IDs
                host_idx = {int(g) - 1 for g in mc_twin_hosts[sk]['host_gids']}
                mc_vals = gs.prop.loc[gs.prop.index.isin(host_idx), p].dropna().values
                all_vals.extend(mc_vals)

            if len(all_vals) < 4:
                ax.set_title(p, fontsize=fontsize)
                continue

            x_min, x_max = np.min(all_vals), np.max(all_vals)
            x = np.linspace(x_min, x_max, 300)

            if len(ebsd_vals) >= 2:
                ax.plot(x, gaussian_kde(ebsd_vals)(x),
                        'k--', lw=1.8, label='EBSD pure parents', zorder=5)

            for cidx, sk in enumerate(slices):
                gs = cntr.gsset[sk]
                if p not in gs.prop.columns:
                    continue
                host_idx = {int(g) - 1 for g in mc_twin_hosts[sk]['host_gids']}
                mc_vals = gs.prop.loc[gs.prop.index.isin(host_idx), p].dropna().values
                if len(mc_vals) >= 2:
                    ax.plot(x, gaussian_kde(mc_vals)(x),
                            color=colours[cidx], lw=1.2, label=f'MC {sk}')

            ax.set_xlabel(p, fontsize=fontsize)
            ax.set_ylabel('Density', fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 1)
            ax.legend(fontsize=max(6, fontsize - 2), framealpha=0.7)

        for idx in range(len(valid_props), nrows_p * ncols_p):
            row_p, col_p = divmod(idx, ncols_p)
            axs_p[row_p][col_p].axis('off')

        fig_p.suptitle(
            'MC host grain properties vs EBSD pure parents',
            fontsize=fontsize + 1,
        )
        fig_p.tight_layout()
        plt.show()

    def assign_mc_parent_orientations(
            self,
            cntr,
            mc_twin_hosts: dict,
            parent_info: dict,
            csl_label: str | None = None,
            mdf: dict | None = None,
            csl_grains: dict | None = None,
            s5_tol_deg: float = 1.0,
            ang_scatter_gaussian_deg: float = 2.0,
            rng_seed=None,
    ) -> dict:
        """
        Assign EBSD pure-parent quaternions to the MC host grains identified
        by :meth:`allocate_mc_twin_hosts`.

        Uses a neighbour-conflict-free sampling strategy: no two adjacent host
        grains receive the same quaternion (which would give zero misorientation
        — physically impossible for a grain boundary).  If the EBSD parent pool
        is smaller than required, additional FCC-texture orientations are
        generated via ``tops.synth_fcc_quats`` as a fallback.

        When *mdf* is supplied, a post-assignment S5 seeding pass seeds Σ5
        (36.87° /<100>) relationships between a fraction of adjacent host-grain
        pairs matching the EBSD S5 boundary fraction.  The replacement
        quaternions are **synthetic** (derived analytically from the pair
        partner's orientation, not sampled from the EBSD pool); the extent of
        this approximation is printed to the console.

        Parameters
        ----------
        cntr : MC_GS_Container2d
            Container with ``.gsset`` dict.
        mc_twin_hosts : dict
            Output of :meth:`allocate_mc_twin_hosts`.
        parent_info : dict
            Output of :meth:`identify_parent_grains`.
        csl_label : str or None
            CSL key to select pure parents from *parent_info*.
            Auto-selects first key when *None*.
        mdf : dict or None
            EBSD MDF dict from :meth:`compute_mdf_ebsd`.  When provided,
            enables S5 boundary seeding.
        csl_grains : dict or None
            Output of :meth:`segregate_csl_pairs` (§9).  When provided, the
            EBSD Σ5 fraction is taken directly from the genuine Σ5 boundary
            count in ``csl_grains``, which is far more accurate than the
            histogram background-subtraction fallback.
        s5_tol_deg : float
            Half-width (°) of the S5 peak window used **only** when
            *csl_grains* is ``None`` (histogram fallback, default 1°).
        ang_scatter_gaussian_deg : float
            Gaussian σ (°) of the small random rotation added to each seeded
            S5 quaternion to reproduce the angular spread seen in EBSD
            (default 2.0°).  Set to 0 for exact 36.87° misorientations.
        rng_seed : int or None
            Seed for reproducibility.

        Returns
        -------
        dict
            Keyed by MC time-slice.  Each value is the dict returned by
            ``assign_parent_orientations``:
            ``{'host_quats': {gid: ndarray(4,)}, 'pool_size': int,
               'n_hosts': int, 'n_fallback': int}``.
            Also stored in :attr:`mc_host_orientations`.
        """
        import numpy as np
        import cc3d
        from upxo.xtalphy.crystal_orientation import assign_parent_orientations

        csl_key = csl_label or next(iter(parent_info))
        ebsd_parent_gids = np.asarray(
            parent_info[csl_key]['pure_parents'], dtype=int
        )

        result = {}
        print(f'CSL label : {csl_key}')
        print(f'  {"Slice":>6}  {"n_hosts":>8}  {"pool_size":>10}  {"n_fallback":>10}')
        print('  ' + '─' * 42)

        for slice_key, info in mc_twin_hosts.items():
            gs = cntr.gsset[slice_key]
            r = assign_parent_orientations(
                host_gids=info['host_gids'],
                sim_lfi=gs.lfi,
                ebsd_parent_gids=ebsd_parent_gids,
                ebsd_lfi=self.lfi_ebsd,
                ebsd_quat=self.quat_ebsd,
                rng_seed=rng_seed,
            )
            result[slice_key] = r
            print(
                f'  {slice_key!s:>6}  {r["n_hosts"]:>8}  '
                f'{r["pool_size"]:>10}  {r["n_fallback"]:>10}'
            )

        # ── S5 boundary seeding (optional) ────────────────────────────────────
        if mdf is not None:
            from upxo.xtalphy.crystal_orientation import (
                _SIGMA5_Q_VARIANTS, _quat_mul, _positive_w,
            )

            rng_s5 = np.random.default_rng(rng_seed)

            miso = mdf['miso_deg']

            if csl_grains is not None:
                # Use true Σ5 fraction from CSL analysis (most accurate).
                _s5_key = min(
                    csl_grains,
                    key=lambda k: abs(csl_grains[k].get('csl_angle', 999.0) - 36.87),
                )
                _s5_data  = csl_grains[_s5_key]
                _s5_angle = _s5_data.get('csl_angle', 0.0)
                if abs(_s5_angle - 36.87) <= 5.0:
                    n_ebsd_s5    = len(_s5_data['pairs'])
                    ebsd_s5_frac = n_ebsd_s5 / len(miso) if len(miso) > 0 else 0.0
                    _src_label   = f'CSL ({_s5_key})'
                else:
                    ebsd_s5_frac = 0.0
                    _src_label   = 'CSL (no Σ5 entry found — skipping)'
            else:
                # Fallback: histogram background subtraction with narrow window.
                _n_hist_bins = 65
                _hist, _edges = np.histogram(miso, bins=_n_hist_bins, range=(0.0, 65.0))
                _centers = (_edges[:-1] + _edges[1:]) / 2.0
                _s5_lo, _s5_hi = 36.87 - s5_tol_deg, 36.87 + s5_tol_deg
                _s5_mask  = (_centers >= _s5_lo) & (_centers <= _s5_hi)
                _bg_lo    = (_centers >= _s5_lo - 10.0) & (_centers < _s5_lo)
                _bg_hi    = (_centers >  _s5_hi)         & (_centers <= _s5_hi + 10.0)
                _bg_bins  = np.concatenate([_hist[_bg_lo], _hist[_bg_hi]])
                _bg_level = _bg_bins.mean() if len(_bg_bins) > 0 else 0.0
                _s5_excess = max(0.0, float(_hist[_s5_mask].sum()) - _bg_level * _s5_mask.sum())
                ebsd_s5_frac = _s5_excess / len(miso) if len(miso) > 0 else 0.0
                _src_label   = f'histogram fallback (±{s5_tol_deg}°)'

            hdr2 = (f'{"Slice":>6}  {"ebsd_s5%":>9}  {"host_pairs":>11}'
                    f'  {"n_seeded":>9}  {"n_synthetic":>12}  {"synth%":>7}')
            print(f'\n  S5 boundary seeding  (source: {_src_label},'
                  f' EBSD S5 fraction = {ebsd_s5_frac:.3%},'
                  f' scatter σ={ang_scatter_gaussian_deg}°,'
                  f' synthetic orientations used)')
            print('  ' + hdr2)
            print('  ' + '─' * len(hdr2))

            _do_scatter = ang_scatter_gaussian_deg > 0.0

            for sk, orientation_result in result.items():
                gs         = cntr.gsset[sk]
                host_quats = orientation_result['host_quats']
                host_arr   = np.array(list(host_quats.keys()), dtype=np.int32)

                lfi_host = np.where(
                    np.isin(gs.lfi, host_arr), gs.lfi, 0
                ).astype(np.int32)
                edges    = list(cc3d.region_graph(lfi_host, connectivity=4))
                host_set = set(int(g) for g in host_arr)
                host_pairs = [(int(a), int(b)) for a, b in edges
                              if int(a) in host_set and int(b) in host_set]

                n_target = max(0, round(len(host_pairs) * ebsd_s5_frac))
                if n_target == 0 or not host_pairs:
                    print(f'  {sk:>6}  {ebsd_s5_frac:>9.3%}  {len(host_pairs):>11}'
                          f'  {"0":>9}  {"0":>12}  {"0.0%":>7}')
                    continue

                rng_s5.shuffle(host_pairs)
                modified: set = set()
                seeded = []
                for a, b in host_pairs:
                    if len(seeded) >= n_target:
                        break
                    if b in modified:
                        a, b = b, a
                    if a in modified:
                        continue
                    seeded.append((a, b))
                    modified.add(b)

                n_synthetic = 0
                for a, b in seeded:
                    q_a     = host_quats[a]
                    variant = _SIGMA5_Q_VARIANTS[int(rng_s5.integers(0, 3))]
                    q_b_new = _quat_mul(variant, q_a)

                    # Add Gaussian angular scatter so seeded boundaries spread
                    # around 36.87° rather than forming a delta-function spike.
                    if _do_scatter:
                        _scat_rad = rng_s5.normal(0.0, np.radians(ang_scatter_gaussian_deg))
                        _ax = rng_s5.standard_normal(3)
                        _ax /= np.linalg.norm(_ax)
                        _pert = np.empty(4, dtype=np.float64)
                        _pert[0] = np.cos(_scat_rad / 2.0)
                        _pert[1:] = np.sin(_scat_rad / 2.0) * _ax
                        q_b_new = _quat_mul(_pert, q_b_new)

                    host_quats[b] = _positive_w(q_b_new)
                    n_synthetic  += 1

                synth_frac = n_synthetic / len(host_quats)
                print(f'  {sk:>6}  {ebsd_s5_frac:>9.3%}  {len(host_pairs):>11}'
                      f'  {len(seeded):>9}  {n_synthetic:>12}  {synth_frac:>7.2%}')

        self.mc_host_orientations = result
        return result

    def plot_mc_parent_mdf(
            self,
            cntr,
            mc_host_orientations: dict,
            mdf: dict,
            slice_key=None,
            n_bins: int = 65,
            angle_range: tuple = (0.0, 65.0),
            figsize: tuple = (6, 4),
            dpi: int = 100,
            fontsize: float = 9.0,
    ) -> dict:
        """
        Compare the neighbour-misorientation distribution of the assigned MC
        host-grain orientations against the reference EBSD MDF.

        Parameters
        ----------
        cntr : MC_GS_Container2d
            Container with ``.gsset`` dict.
        mc_host_orientations : dict
            Output of :meth:`assign_mc_parent_orientations`.
        mdf : dict
            EBSD MDF dict returned by :meth:`compute_mdf_ebsd` (§5).
        slice_key : int, list, or None
            ``None`` → all keys in *mc_host_orientations*;
            ``int`` → that one slice; ``list`` → those slices (accepts
            the output of :meth:`best_match_mcgs_key` directly).
        n_bins : int
            Histogram bins for each MC MDF.
        angle_range : (float, float)
            Histogram range in degrees.
        figsize, dpi, fontsize : plot parameters.

        Returns
        -------
        dict
            ``{slice_key: mc_mdf_dict, ...}`` — one entry per plotted slice.
        """
        import cc3d
        import matplotlib.pyplot as plt
        import numpy as np
        from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats

        if slice_key is None:
            keys = list(mc_host_orientations.keys())
        elif isinstance(slice_key, (int, np.integer)):
            keys = [int(slice_key)]
        else:
            keys = list(slice_key)

        colors = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(mdf['hist_bin_centers'], mdf['hist_density'],
                'k-', lw=2.0, label='EBSD (all boundaries)')

        results = {}
        for i, sk in enumerate(keys):
            gs         = cntr.gsset[sk]
            host_quats = mc_host_orientations[sk]['host_quats']
            host_set   = set(host_quats.keys())

            ny, nx    = gs.lfi.shape
            quat_host = np.zeros((ny, nx, 4), dtype=np.float64)
            quat_host[..., 0] = 1.0
            for gid, q in host_quats.items():
                quat_host[gs.lfi == gid] = q

            host_arr = np.array(list(host_set), dtype=np.int32)
            lfi_host = np.where(np.isin(gs.lfi, host_arr), gs.lfi, 0).astype(np.int32)

            edges      = cc3d.region_graph(lfi_host, connectivity=4)
            neigh_host = {g: [] for g in host_set}
            for a, b in edges:
                a, b = int(a), int(b)
                if a in host_set and b in host_set:
                    neigh_host[a].append(b)
                    neigh_host[b].append(a)

            if not any(neigh_host.values()):
                print(f'[plot_mc_parent_mdf] slice {sk}: no adjacent host pairs — skipped.')
                continue

            mc_mdf = compute_mdf_from_quats(
                lfi_host, quat_host, neigh_host,
                n_bins=n_bins, angle_range=angle_range,
            )
            results[sk] = mc_mdf

            color = colors[i % len(colors)]
            ax.plot(mc_mdf['hist_bin_centers'], mc_mdf['hist_density'],
                    '--', lw=1.5, color=color,
                    label=f'MC slice {sk}  ({mc_mdf["n_pairs"]} pairs)')

        ax.set_xlabel('Misorientation angle (°)', fontsize=fontsize)
        ax.set_ylabel('Probability density',      fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1)
        ax.legend(fontsize=fontsize - 1, framealpha=0.7)
        ax.set_title(
            f'MC host-grain MDF  |  EBSD mean: {mdf["mean_angle"]:.1f}°',
            fontsize=fontsize,
        )
        fig.tight_layout()
        plt.show()

        return results

    # -----------------------------------------------------------------------
    # Part G — twin lamella geometry introduction
    # -----------------------------------------------------------------------

    def compute_mc_twin_thickness(
            self,
            parent_info: dict,
            abrupt_threshold: float = 0.8,
    ) -> dict:
        """
        Compute EBSD twin lamella thickness and intercept-length statistics.

        Calls :func:`~upxo.gsdataops.grid_ops.compute_twin_thickness_stats`
        (with abrupt-twin detection) and pools per-grain intercept lengths
        across all EBSD twin grains to derive Q1/Q2/Q3 quantiles.

        Parameters
        ----------
        parent_info : dict
            Output of :meth:`identify_parent_grains`.
        abrupt_threshold : float
            Twin extent / parent extent ratio below which a twin is counted
            as abruptly-ending.  Default 0.8.

        Returns
        -------
        dict
            Combined thickness + intercept stats dict.  Keys include
            ``'thick_px'``, ``'intercept_px'``, ``'intercept_q1'``,
            ``'intercept_q2'``, ``'intercept_q3'``, ``'n_abrupt_ebsd'``,
            ``'abrupt_frac_ebsd'``, and all keys from
            :func:`compute_twin_thickness_stats`.
        """
        import numpy as np
        from upxo.gsdataops.grid_ops import (
            compute_twin_thickness_stats,
            compute_grain_intercept_lengths,
        )
        from skimage.measure import regionprops as _skrp

        stats = compute_twin_thickness_stats(
            parent_info, self.prop_ebsd, self.ebsd_step,
            lfi=self.lfi_ebsd, abrupt_threshold=abrupt_threshold,
        )

        # Pool intercept lengths across all EBSD twin grains
        gid2props = {p.label: p for p in _skrp(self.lfi_ebsd.astype(np.int32))}
        all_intercepts = []
        for gid in stats['gids']:
            rp = gid2props.get(gid)
            if rp is not None:
                all_intercepts.append(compute_grain_intercept_lengths(rp, self.lfi_ebsd))

        if all_intercepts:
            intercept_px = np.concatenate(all_intercepts)
        else:
            intercept_px = np.array([], dtype=np.float64)

        stats['intercept_px'] = intercept_px
        if intercept_px.size > 0:
            stats['intercept_q1'] = float(np.percentile(intercept_px, 25))
            stats['intercept_q2'] = float(np.percentile(intercept_px, 50))
            stats['intercept_q3'] = float(np.percentile(intercept_px, 75))
        else:
            stats['intercept_q1'] = stats['intercept_q2'] = stats['intercept_q3'] = float('nan')

        print(f'  Twin thickness ({stats["col"]}) and intercept statistics (EBSD):')
        header = f'  {"Metric":<28}  {"Q1":>8}  {"Q2 (med)":>10}  {"Q3":>8}  {"IQR":>8}'
        print(header)
        print('  ' + '─' * (len(header) - 2))
        thick_um = stats['thick_um']
        if thick_um.size > 0:
            tq1, tq2, tq3 = (float(np.percentile(thick_um, p)) for p in (25, 50, 75))
            print(f'  {"Thickness (µm)":<28}  {tq1:>8.2f}  {tq2:>10.2f}  {tq3:>8.2f}  {(tq3-tq1):>8.2f}')
        if intercept_px.size > 0:
            iq1, iq2, iq3 = stats['intercept_q1'], stats['intercept_q2'], stats['intercept_q3']
            print(f'  {"Intercept length (px)":<28}  {iq1:>8.2f}  {iq2:>10.2f}  {iq3:>8.2f}  {(iq3-iq1):>8.2f}')
        print(f'  Abrupt twins: {stats["n_abrupt_ebsd"]} / '
              f'{len(stats["gids"])}  '
              f'(fraction = {stats["abrupt_frac_ebsd"]:.3f})')

        return stats

    def introduce_mc_twin_lamellae(
            self,
            cntr,
            mc_twin_hosts: dict,
            mc_host_orientations: dict,
            twin_thickness: dict,
            tvf: dict,
            csl_label: str = 'S3  (twin)',
            n_twins_per_parent: int = 1,
            ang_scatter_gaussian_deg: float = 3.0,
            twin_orient_scatter_deg: float = 1.5,
            rng_seed=None,
    ) -> dict:
        """
        Introduce S3 twin lamellae into MC host grains.

        For each slice in *mc_twin_hosts*:

        * Primary twins are introduced into each host grain using a
          lamella angle derived from the host's crystal orientation
          (projection of the active {111} trace onto XY) and a
          half-width sampled from the EBSD twin-thickness distribution.
        * A fraction ``twin_thickness['abrupt_frac_ebsd']`` of lamellae
          are truncated to end abruptly inside the host grain.
        * Secondary twins are introduced into a subset of primary-twin
          grains when ``tvf['secondary_twin_frac'] > 0``.

        Returns
        -------
        dict  (also stored in ``self.mc_twin_geom``)
            ``{sk: {'twin_result_agg': ..., 'sec_twin_result_agg': ...,
                    'all_quats': ..., 'n_abrupt_mc': int,
                    'abrupt_frac_mc': float}}``
        """
        import numpy as np
        import cc3d
        from tqdm.auto import tqdm
        from upxo.xtalphy.crystal_orientation import (
            introduce_twins_by_csl,
            compute_s3_lamella_angle_2d,
            _SIGMA3_Q, _quat_mul, _positive_w,
        )

        self.mc_twin_geom = {}
        rng = np.random.default_rng(rng_seed)
        thick_px = twin_thickness['thick_px']
        abrupt_frac_ebsd = twin_thickness.get('abrupt_frac_ebsd', 0.0)

        def _truncate_lamella(lgi, twin_gid, angle_deg, parent_gid):
            """Zero out the farther half of a twin lamella's pixels."""
            angle_rad = np.radians(angle_deg)
            d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            rows, cols = np.where(lgi == twin_gid)
            if rows.size == 0:
                return
            projs = rows * d[0] + cols * d[1]
            cut = float(np.median(projs))
            mask = projs > cut
            lgi[rows[mask], cols[mask]] = parent_gid

        sec_twin_frac = tvf.get('secondary_twin_frac', 0.0)
        prim_twin_frac = tvf.get('primary_twin_frac', 1.0)

        if thick_px.size == 0:
            thick_px = np.array([2.0])

        hdr = (f'{"Slice":>6}  {"n_hosts":>8}  {"n_prim_twins":>13}  '
               f'{"n_abrupt":>9}  {"abrupt_frac":>12}  {"n_sec_twins":>12}')
        print(hdr)
        print('─' * len(hdr))

        for sk, host_info in mc_twin_hosts.items():
            gs         = cntr.gsset[sk]
            host_gids  = host_info['host_gids']
            host_quats = mc_host_orientations[sk]['host_quats']

            twin_result_agg = {'new_twin_gids': {}, 'twin_lines': {}}
            primary_twin_quats: dict = {}

            for parent_gid in tqdm(host_gids, desc=f'Slice {sk} — primary twins', unit='grain', leave=True):
                q_par = host_quats.get(int(parent_gid))
                if q_par is None:
                    continue
                angle = compute_s3_lamella_angle_2d(q_par)
                hw = max(1.0, float(rng.choice(thick_px)) / 2.0)
                is_abrupt = rng.random() < abrupt_frac_ebsd

                res = introduce_twins_by_csl(
                    gs.lgi, [parent_gid], csl_label,
                    twin_half_width=hw,
                    twin_angle_deg=angle,
                    n_twins_per_parent=n_twins_per_parent,
                    angle_perturb_deg=ang_scatter_gaussian_deg,
                    rng_seed=None,
                )
                gs.lgi[:] = res['lfi']

                for tgid in res['new_twin_gids'].get(int(parent_gid), []):
                    if is_abrupt:
                        _truncate_lamella(gs.lgi, tgid, angle, int(parent_gid))
                    q_twin = _quat_mul(_SIGMA3_Q, q_par)
                    if twin_orient_scatter_deg > 0.0:
                        _sc = rng.normal(0.0, np.radians(twin_orient_scatter_deg))
                        _ax = rng.standard_normal(3); _ax /= np.linalg.norm(_ax)
                        _pt = np.array([np.cos(_sc / 2.), *np.sin(_sc / 2.) * _ax],
                                       dtype=np.float64)
                        q_twin = _quat_mul(_pt, q_twin)
                    primary_twin_quats[tgid] = _positive_w(q_twin)

                twin_result_agg['new_twin_gids'].update(res['new_twin_gids'])
                twin_result_agg['twin_lines'].update(res['twin_lines'])

            # Verify abrupt twins in MC via regionprops + adjacency
            n_abrupt_mc   = 0
            n_checked_mc  = 0
            try:
                from skimage.measure import regionprops as _skrp
                gid2props_mc = {p.label: p for p in _skrp(gs.lgi.astype(np.int32))}
                prim_twin_set = set(primary_twin_quats.keys())
                edges_mc = cc3d.region_graph(gs.lgi.astype(np.int32), connectivity=4)
                twin_to_par_mc = {}
                host_set = set(int(g) for g in host_gids)
                for a, b in edges_mc:
                    a, b = int(a), int(b)
                    if a in prim_twin_set and b in host_set and a not in twin_to_par_mc:
                        twin_to_par_mc[a] = b
                    elif b in prim_twin_set and a in host_set and b not in twin_to_par_mc:
                        twin_to_par_mc[b] = a
                for tgid, pgid in twin_to_par_mc.items():
                    rp_t = gid2props_mc.get(tgid)
                    rp_p = gid2props_mc.get(pgid)
                    if rp_t is None or rp_p is None:
                        continue
                    d = np.array([-np.sin(rp_p.orientation), np.cos(rp_p.orientation)])
                    proj_p = rp_p.coords @ d
                    proj_t = rp_t.coords @ d
                    ps = float(proj_p.max() - proj_p.min())
                    ts = float(proj_t.max() - proj_t.min())
                    if ps > 0 and ts / ps < 0.8:
                        n_abrupt_mc += 1
                    n_checked_mc += 1
            except Exception:
                pass
            abrupt_frac_mc = (n_abrupt_mc / n_checked_mc) if n_checked_mc > 0 else 0.0

            # Secondary twins
            sec_twin_result_agg: dict = {}
            secondary_twin_quats: dict = {}
            n_sec_twins = 0
            if sec_twin_frac > 0 and prim_twin_frac > 0 and primary_twin_quats:
                sec_ratio  = sec_twin_frac / max(prim_twin_frac, 1e-9)
                n_sec_host = max(1, round(len(primary_twin_quats) * sec_ratio))
                try:
                    _px_lut = np.bincount(gs.lgi.ravel())
                    gid2px  = {g: int(_px_lut[g]) if g < len(_px_lut) else 0
                               for g in primary_twin_quats}
                    sec_hosts = sorted(gid2px, key=gid2px.get, reverse=True)[:n_sec_host]
                except Exception:
                    sec_hosts = list(primary_twin_quats.keys())[:n_sec_host]

                sec_result_agg = {'new_twin_gids': {}, 'twin_lines': {}}
                for sec_parent in tqdm(sec_hosts, desc=f'Slice {sk} — secondary twins', unit='grain', leave=True):
                    q_sec_par = primary_twin_quats[sec_parent]
                    angle_s   = compute_s3_lamella_angle_2d(q_sec_par)
                    hw_s      = max(1.0, float(rng.choice(thick_px)) / 2.0)
                    res_s     = introduce_twins_by_csl(
                        gs.lgi, [sec_parent], csl_label,
                        twin_half_width=hw_s,
                        twin_angle_deg=angle_s,
                        n_twins_per_parent=1,
                        angle_perturb_deg=ang_scatter_gaussian_deg,
                        rng_seed=None,
                    )
                    gs.lgi[:] = res_s['lfi']
                    for stgid in res_s['new_twin_gids'].get(int(sec_parent), []):
                        q_sec = _quat_mul(_SIGMA3_Q, q_sec_par)
                        if twin_orient_scatter_deg > 0.0:
                            _sc = rng.normal(0.0, np.radians(twin_orient_scatter_deg))
                            _ax = rng.standard_normal(3); _ax /= np.linalg.norm(_ax)
                            _pt = np.array([np.cos(_sc / 2.), *np.sin(_sc / 2.) * _ax],
                                           dtype=np.float64)
                            q_sec = _quat_mul(_pt, q_sec)
                        secondary_twin_quats[stgid] = _positive_w(q_sec)
                    sec_result_agg['new_twin_gids'].update(res_s['new_twin_gids'])
                    sec_result_agg['twin_lines'].update(res_s['twin_lines'])
                n_sec_twins = sum(len(v) for v in sec_result_agg['new_twin_gids'].values())
                sec_twin_result_agg = sec_result_agg

            n_prim_twins = sum(len(v) for v in twin_result_agg['new_twin_gids'].values())
            print(f'{sk:>6}  {len(host_gids):>8}  {n_prim_twins:>13}  '
                  f'{n_abrupt_mc:>9}  {abrupt_frac_mc:>12.3f}  {n_sec_twins:>12}')

            self.mc_twin_geom[sk] = {
                'twin_result_agg':     twin_result_agg,
                'sec_twin_result_agg': sec_twin_result_agg,
                'all_quats': {
                    **host_quats,
                    **primary_twin_quats,
                    **secondary_twin_quats,
                },
                'n_abrupt_mc':    n_abrupt_mc,
                'abrupt_frac_mc': abrupt_frac_mc,
            }

        return self.mc_twin_geom

    def plot_mc_twin_mdf(
            self,
            cntr,
            mc_twin_geom: dict,
            mc_twin_hosts: dict,
            mdf: dict,
            tvf: dict,
            n_bins: int = 65,
            angle_range: tuple = (0.0, 65.0),
            figsize: tuple = (6, 4),
            dpi: int = 100,
            fontsize: float = 9.0,
            kde: bool = False,
            kde_bw: str | float = 0.1,
            kde_n_points: int = 500,
            show_peaks: bool = False,
            peak_prominence: float = 0.002,
    ) -> dict:
        """
        MDF of all oriented grain-boundary pairs in the post-twin MC structure
        overlaid on the reference EBSD MDF. Volume fractions are annotated in
        the legend so the user can gauge twin-area accuracy at a glance.

        Parameters
        ----------
        kde : bool
            If True, plot smooth KDE curves instead of histogram density.
            When enabled only KDE curves are drawn (no histogram).
        kde_bw : str or float
            KDE bandwidth passed to ``scipy.stats.gaussian_kde``.
            ``0.1`` (default, ≈1.5° effective bandwidth), ``'scott'``,
            ``'silverman'``, or any scalar factor.
        kde_n_points : int
            Number of evaluation points for the KDE grid.
        show_peaks : bool
            When ``kde=True``, annotate detected peaks. EBSD peaks are
            marked with vertical dotted lines; MC peaks similarly as
            thin coloured dotted vertical lines.
        peak_prominence : float
            Minimum prominence for ``scipy.signal.find_peaks``.

        Returns
        -------
        dict  ``{sk: mc_mdf_dict}``
        """
        import numpy as np
        import cc3d
        import matplotlib.pyplot as plt
        from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats

        if kde:
            from upxo.xtalphy.crystal_orientation import detect_mdf_peaks

        colors = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ebsd_host_frac = self.merge_info.get('twin_hosting_fraction', float('nan'))
        ebsd_label = (
            f'EBSD  host={ebsd_host_frac:.1%}'
            f'  prim={tvf["primary_twin_frac"]:.1%}'
            f'  sec={tvf["secondary_twin_frac"]:.1%}'
        )
        if kde:
            _ebsd_pk = detect_mdf_peaks(
                mdf, bw_method=kde_bw, n_kde=kde_n_points,
                prominence=peak_prominence,
            )
            ax.plot(_ebsd_pk['theta_fine'], _ebsd_pk['kde_vals'],
                    'k-', lw=2.5, label=ebsd_label)
            if show_peaks:
                for _ang in _ebsd_pk['peak_angles']:
                    ax.axvline(_ang, color='k', lw=0.8, ls=':', alpha=0.5)
        else:
            ax.plot(mdf['hist_bin_centers'], mdf['hist_density'],
                    'k-', lw=2.0, label=ebsd_label)

        results = {}
        for i, (sk, geom) in enumerate(mc_twin_geom.items()):
            gs  = cntr.gsset[sk]
            lgi = gs.lgi

            all_quats = geom['all_quats']
            if not all_quats:
                continue

            all_gid_arr = np.array(list(all_quats.keys()), dtype=np.int32)
            ny, nx = lgi.shape
            quat_map = np.zeros((ny, nx, 4), dtype=np.float64)
            quat_map[..., 0] = 1.0
            for gid, q in all_quats.items():
                quat_map[lgi == gid] = q

            lfi_masked = np.where(np.isin(lgi, all_gid_arr), lgi, 0).astype(np.int32)

            edges = cc3d.region_graph(lfi_masked, connectivity=4)
            neigh = {int(g): [] for g in all_gid_arr}
            all_set = set(int(g) for g in all_gid_arr)
            for a, b in edges:
                a, b = int(a), int(b)
                if a in all_set and b in all_set:
                    neigh[a].append(b)
                    neigh[b].append(a)

            if not any(neigh.values()):
                continue

            mc_mdf = compute_mdf_from_quats(
                lfi_masked, quat_map, neigh,
                n_bins=n_bins, angle_range=angle_range,
            )
            results[sk] = mc_mdf

            # Volume fractions via LUT — O(N_pixels) single pass
            host_gids = set(int(g) for g in mc_twin_hosts[sk]['host_gids'])
            prim_gids = {int(g) for v in geom['twin_result_agg']['new_twin_gids'].values() for g in v}
            sec_gids  = {int(g) for v in geom.get('sec_twin_result_agg', {}).get('new_twin_gids', {}).values() for g in v}
            role_lut = np.zeros(int(lgi.max()) + 1, dtype=np.uint8)
            remaining_host = host_gids - prim_gids - sec_gids
            if remaining_host:
                role_lut[list(remaining_host)] = 1
            if prim_gids:
                role_lut[list(prim_gids)] = 2
            if sec_gids:
                role_lut[list(sec_gids)] = 3
            roles    = role_lut[lgi]
            total_px = lgi.size
            vf_host  = float((roles == 1).sum()) / total_px
            vf_prim  = float((roles == 2).sum()) / total_px
            vf_sec   = float((roles == 3).sum()) / total_px

            label = (
                f'MC {sk}  host={vf_host:.1%}'
                f'  prim={vf_prim:.1%}'
                f'  sec={vf_sec:.1%}'
                f'  ({mc_mdf["n_pairs"]} pairs)'
            )
            color = colors[i % len(colors)]
            if kde:
                _mc_pk = detect_mdf_peaks(
                    mc_mdf, bw_method=kde_bw, n_kde=kde_n_points,
                    prominence=peak_prominence,
                )
                ax.plot(_mc_pk['theta_fine'], _mc_pk['kde_vals'],
                        '--', lw=1.5, color=color, label=label)
                if show_peaks:
                    for _ang in _mc_pk['peak_angles']:
                        ax.axvline(_ang, color=color, lw=0.8, ls=':', alpha=0.5)
            else:
                ax.plot(mc_mdf['hist_bin_centers'], mc_mdf['hist_density'],
                        '--', lw=1.5, color=color, label=label)

        ax.set_xlabel('Misorientation angle (°)', fontsize=fontsize)
        ax.set_ylabel('Probability density',      fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1)
        ax.legend(fontsize=fontsize - 1.5, framealpha=0.7)
        ax.set_title(
            'Post-twin MDF',
            fontsize=fontsize,
        )
        fig.tight_layout()
        plt.show()

        return results

    def visualize_mc_twin_lamellae(
            self,
            cntr,
            mc_twin_geom: dict,
            mc_twin_hosts: dict,
            parent_info: dict,
            tvf: dict,
            figsize_per_slice: tuple = (5, 4),
            dpi: int = 100,
            ncols: int = None,
            show_ebsd: bool = True,
    ) -> None:
        """
        Spatial maps of introduced twin lamellae: EBSD reference + one MC slice
        per subplot.

        Pixel colours (all subplots):
        - grey  ``#b0b0b0`` — non-participating grains
        - blue  ``#4682b4`` — host / pure-parent grains
        - orange ``#ff8c00`` — primary twin grains
        - red   ``#cc2222`` — secondary twin grains

        Subplot titles carry pixel-based volume fractions. The first subplot is
        the EBSD reference; subsequent subplots are the MC slices.

        Parameters
        ----------
        parent_info : dict
            Output of ``identify_parent_grains`` — used to colour EBSD grains.
        tvf : dict
            Output of ``compute_ebsd_tvf`` — supplies ``csl_label``,
            ``extended_info``, and EBSD volume fractions.
        ncols : int, optional
            Subplot grid columns. Default = all in a single row.
        show_ebsd : bool
            Include the EBSD reference panel. Default ``True``.
            Set to ``False`` to plot only the MC slice panels.
        """
        import math
        import numpy as np
        import matplotlib.pyplot as plt

        col_nonhost = np.array([0.69,   0.69,   0.69  ])  # #b0b0b0
        col_host    = np.array([0.275,  0.510,  0.706 ])  # #4682b4
        col_prim    = np.array([1.0,    0.549,  0.0   ])  # #ff8c00
        col_sec     = np.array([0.80,   0.133,  0.133 ])  # #cc2222

        def _lut_rgb(lgi, set1, set2, set3):
            """Build RGB via look-up table — O(N_grains) fill + O(N_pixels) index."""
            lut = np.tile(col_nonhost, (int(lgi.max()) + 1, 1))
            if set1: lut[list(set1)] = col_host
            if set2: lut[list(set2)] = col_prim
            if set3: lut[list(set3)] = col_sec
            return lut[lgi]  # (ny, nx, 3)

        def _role_vf(lgi, host_set, prim_set, sec_set):
            """Pixel-based volume fractions via LUT — O(N_pixels) single pass."""
            role_lut = np.zeros(int(lgi.max()) + 1, dtype=np.uint8)
            remaining = host_set - prim_set - sec_set
            if remaining: role_lut[list(remaining)] = 1
            if prim_set:  role_lut[list(prim_set)]  = 2
            if sec_set:   role_lut[list(sec_set)]   = 3
            roles = role_lut[lgi]
            t = lgi.size
            return (roles == 1).sum() / t, (roles == 2).sum() / t, (roles == 3).sum() / t

        keys  = list(mc_twin_geom.keys())
        n_mc  = len(keys)
        n_tot = (1 if show_ebsd else 0) + n_mc
        nc    = min(n_tot, ncols) if ncols is not None else n_tot
        nr    = math.ceil(n_tot / nc)
        fig, axes = plt.subplots(nr, nc,
                                 figsize=(figsize_per_slice[0] * nc,
                                          figsize_per_slice[1] * nr),
                                 dpi=dpi)
        axes = np.array(axes).flatten()

        mc_start = 0
        if show_ebsd:
            # ── EBSD reference subplot ────────────────────────────────────────
            csl_label = tvf['csl_label']
            ext       = tvf['extended_info'][csl_label]
            ebsd_ppar = set(int(g) for g in ext['pure_parents'])
            ebsd_prim = set(int(g) for g in ext['primary_twins'])
            ebsd_sec  = set(int(g) for g in ext['secondary_twins'])
            rgb_ebsd  = _lut_rgb(self.lfi_ebsd, ebsd_ppar, ebsd_prim, ebsd_sec)
            axes[0].imshow(rgb_ebsd, origin='upper')
            ebsd_host_frac = self.merge_info.get('twin_hosting_fraction', float('nan'))
            ebsd_title = (
                f'EBSD  host={ebsd_host_frac:.1%}'
                f'  prim={tvf["primary_twin_frac"]:.1%}'
                f'  sec={tvf["secondary_twin_frac"]:.1%}'
            )
            axes[0].set_title(ebsd_title, fontsize=8)
            axes[0].axis('off')
            mc_start = 1

        # ── MC slice subplots ─────────────────────────────────────────────────
        for ax, sk in zip(axes[mc_start:], keys):
            gs   = cntr.gsset[sk]
            lgi  = gs.lgi
            geom = mc_twin_geom[sk]
            host_set = set(int(g) for g in mc_twin_hosts[sk]['host_gids'])
            prim_set = {int(g) for v in geom['twin_result_agg']['new_twin_gids'].values() for g in v}
            sec_raw  = geom.get('sec_twin_result_agg', {}).get('new_twin_gids', {})
            sec_set  = {int(g) for v in sec_raw.values() for g in v}

            rgb = _lut_rgb(lgi, host_set - prim_set - sec_set, prim_set, sec_set)
            vf_host, vf_prim, vf_sec = _role_vf(lgi, host_set, prim_set, sec_set)

            ax.imshow(rgb, origin='upper')
            ax.set_title(
                f'Slice {sk}  host={vf_host:.1%}  prim={vf_prim:.1%}  sec={vf_sec:.1%}',
                fontsize=8,
            )
            ax.axis('off')

        for ax in axes[n_tot:]:
            ax.axis('off')

        fig.suptitle(
            'Twin lamellae map — blue: host/parent  |  orange: primary  |  red: secondary',
            fontsize=9,
        )
        fig.tight_layout()
        plt.show()

    def compare_twin_intercepts(
            self,
            cntr,
            mc_twin_geom: dict,
            twin_thickness: dict,
            figsize: tuple = (6, 4),
            dpi: int = 100,
            fontsize: float = 9.0,
    ) -> dict:
        """
        Overlay EBSD vs MC twin intercept-length distributions.

        For each MC slice the intercept lengths of introduced twin grains are
        pooled and plotted as a KDE curve against the EBSD reference pool
        (already stored in ``twin_thickness['intercept_px']``).

        Returns
        -------
        dict  ``{sk: intercept_px_array}``
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
        from upxo.gsdataops.grid_ops import compute_grain_intercept_lengths
        from skimage.measure import regionprops as _skrp

        ebsd_intercepts = twin_thickness.get('intercept_px', np.array([]))
        colors = plt.cm.tab10.colors

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if ebsd_intercepts.size > 0:
            kde_e = gaussian_kde(ebsd_intercepts)
            xg    = np.linspace(ebsd_intercepts.min(), ebsd_intercepts.max(), 300)
            eq1   = twin_thickness.get('intercept_q1', float('nan'))
            eq2   = twin_thickness.get('intercept_q2', float('nan'))
            eq3   = twin_thickness.get('intercept_q3', float('nan'))
            ax.plot(xg, kde_e(xg), 'k-', lw=2.0,
                    label=f'EBSD  Q1={eq1:.1f}  Q2={eq2:.1f}  Q3={eq3:.1f} px')

        results = {}
        for i, (sk, geom) in enumerate(mc_twin_geom.items()):
            gs = cntr.gsset[sk]
            prim_gids = [g for lst in geom['twin_result_agg']['new_twin_gids'].values() for g in lst]
            sec_gids  = [g for lst in geom.get('sec_twin_result_agg', {}).get('new_twin_gids', {}).values() for g in lst]
            all_twin_gids = prim_gids + sec_gids
            if not all_twin_gids:
                continue
            gid2props_mc = {p.label: p for p in _skrp(gs.lgi.astype(np.int32))}
            mc_intercepts_list = []
            for tgid in all_twin_gids:
                rp = gid2props_mc.get(tgid)
                if rp is not None:
                    mc_intercepts_list.append(compute_grain_intercept_lengths(rp, gs.lgi))
            if not mc_intercepts_list:
                continue
            mc_ints = np.concatenate(mc_intercepts_list)
            results[sk] = mc_ints
            if mc_ints.size > 1:
                kde_m = gaussian_kde(mc_ints)
                xg_m  = np.linspace(mc_ints.min(), mc_ints.max(), 300)
                mq1, mq2, mq3 = (float(np.percentile(mc_ints, p)) for p in (25, 50, 75))
                ax.plot(xg_m, kde_m(xg_m), '--', lw=1.5, color=colors[i % len(colors)],
                        label=f'MC slice {sk}  Q1={mq1:.1f}  Q2={mq2:.1f}  Q3={mq3:.1f} px')

        ax.set_xlabel('Intercept length (px)', fontsize=fontsize)
        ax.set_ylabel('Probability density',   fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1)
        ax.legend(fontsize=fontsize - 1, framealpha=0.7)
        ax.set_title('Twin intercept lengths — EBSD reference vs MC introduced', fontsize=fontsize)
        fig.tight_layout()
        plt.show()

        return results

    def plot_ipf_maps(
            self,
            cntr,
            mc_twin_geom: dict,
            sample_direction: tuple = (0., 0., 1.),
            figsize_per_panel: tuple = (4, 3),
            dpi: int = 100,
            fontsize: float = 9.0,
            ncols: int | None = None,
            show_ebsd: bool = True,
    ) -> None:
        """
        IPF orientation maps for the EBSD reference and every MC post-twin
        slice in *mc_twin_geom*, arranged in a grid.

        Colours encode the crystal direction parallel to *sample_direction*
        using the standard ``|R(q).T @ sd|`` formula (same as
        ``crystal_orientation.ipf_color``).  Pixels whose grain has no
        assigned orientation are rendered grey.

        Parameters
        ----------
        cntr : MC_GS_Container2d
            Container with ``.gsset`` dict.
        mc_twin_geom : dict
            Output of :meth:`introduce_mc_twin_lamellae`; each value must
            contain ``'all_quats'`` ``{gid: ndarray(4,)}``.
        sample_direction : tuple of 3 floats
            Reference direction (crystal direction to project onto).
            ``(0,0,1)`` = ND (normal direction, default),
            ``(1,0,0)`` = RD (rolling direction).
        figsize_per_panel : (float, float)
            Width × height (inches) of each subplot.
        dpi, fontsize : plot parameters.
        ncols : int or None
            Number of columns in the subplot grid.  ``None`` (default) puts
            all panels in a single row.
        show_ebsd : bool
            Include the EBSD reference panel. Default ``True``.
            Set to ``False`` to plot only the MC slice panels.
        """
        import math
        import matplotlib.pyplot as plt
        import upxo.viz.ebsdviz as _ebsdviz
        from upxo.xtalphy.crystal_orientation import grain_avg_quats

        mc_keys = list(mc_twin_geom.keys())
        n_total = (1 if show_ebsd else 0) + len(mc_keys)
        nc = n_total if ncols is None else min(int(ncols), n_total)
        nr = math.ceil(n_total / nc)
        fw = figsize_per_panel[0] * nc
        fh = figsize_per_panel[1] * nr
        fig, axes = plt.subplots(nr, nc, figsize=(fw, fh), dpi=dpi,
                                 squeeze=False)
        axes_flat = axes.ravel()

        mc_start = 0
        if show_ebsd:
            ebsd_lfi = self.lfi_ebsd
            _gids, _q_mean = grain_avg_quats(ebsd_lfi, self.quat_ebsd)
            ebsd_quats = {int(g): q for g, q in zip(_gids, _q_mean)}
            axes_flat[0].imshow(
                _ebsdviz.build_ipf_rgb(ebsd_lfi, ebsd_quats, sample_direction),
                origin='upper', interpolation='nearest',
            )
            axes_flat[0].set_title('EBSD', fontsize=fontsize)
            axes_flat[0].axis('off')
            mc_start = 1

        for i, sk in enumerate(mc_keys):
            gs  = cntr.gsset[sk]
            rgb = _ebsdviz.build_ipf_rgb(
                gs.lgi, mc_twin_geom[sk]['all_quats'], sample_direction,
            )
            axes_flat[mc_start + i].imshow(rgb, origin='upper', interpolation='nearest')
            axes_flat[mc_start + i].set_title(f'MC {sk}', fontsize=fontsize)
            axes_flat[mc_start + i].axis('off')

        for ax in axes_flat[n_total:]:
            ax.set_visible(False)

        fig.suptitle(
            f'IPF maps  —  sample direction {list(sample_direction)}',
            fontsize=fontsize,
        )
        fig.tight_layout()
        plt.show()

    def assign_mc_nonhost_orientations(
            self,
            cntr,
            mc_twin_geom: dict,
            rng_seed=None,
    ) -> None:
        """
        Assign orientations to all MC grains not already in *mc_twin_geom*
        ``'all_quats'`` by sampling from the EBSD grain-average quaternion pool.

        Mutates *mc_twin_geom* in place.  Host and twin grains already present
        in ``'all_quats'`` are not overwritten.

        Parameters
        ----------
        cntr : MC_GS_Container2d
            Container with ``.gsset`` dict.
        mc_twin_geom : dict
            Output of :meth:`introduce_mc_twin_lamellae`; ``'all_quats'`` is
            extended for each slice.
        rng_seed : int or None
            Seed for the random number generator.
        """
        import numpy as np
        from upxo.xtalphy.crystal_orientation import grain_avg_quats

        rng = np.random.default_rng(rng_seed)

        # Build EBSD grain-average quaternion pool once
        _gids_ebsd, _q_pool = grain_avg_quats(self.lfi_ebsd, self.quat_ebsd)

        for sk, geom in mc_twin_geom.items():
            gs = cntr.gsset[sk]
            all_gids = set(int(g) for g in np.unique(gs.lgi) if g > 0)
            unoriented = all_gids - set(geom['all_quats'].keys())
            if not unoriented:
                continue
            unoriented_list = sorted(unoriented)
            idx = rng.integers(0, len(_q_pool), size=len(unoriented_list))
            geom['all_quats'].update({
                gid: _q_pool[idx[i]]
                for i, gid in enumerate(unoriented_list)
            })

    def smooth_mc_slices(
            self,
            cntr,
            slice_keys: list | None = None,
            area_threshold: int = 1,
            smooth_iter: int = 10,
            smooth_lambda: float = 0.5,
            smooth_mu: float = -0.53,
            trim_bounds=(5, 5, 95, 95),
            coord_decimals: int = 6,
            verbose: bool = True,
            method: str = 'taubin',
            ma_window: int = 3,
            corner_angle_deg: float = 30.0,
            upscale_factor: int = 1,
            thin_grain_px: float = 0.0,
            fix_diagonal: bool = True,
            merge_enclosed: bool = True,
            close_staircase: bool = True,
            **seed_kwargs,
    ) -> dict:
        """
        Smooth the grain geometry for one or more MC slices and store the
        result in ``self.mc_smooth_geom``.

        Each slice is processed by
        :func:`upxo.pxtalops.gssmooth2d.smooth_gs_slice` (small-grain merge
        → tessellation → Taubin smooth → trim → neighbour graph → GID
        renumbering → validation → interface extraction → junction
        extraction).

        Parameters
        ----------
        cntr : MC_GS_Container2d
            Container with ``.gsset {sk: gs}``.
        slice_keys : list or None
            Keys to process.  ``None`` processes all keys in ``cntr.gsset``.
        area_threshold : int
            Grains with <= this many pixels are merged before tessellation.
        smooth_iter : int
            Taubin smoothing iterations.
        smooth_lambda : float
            Taubin forward-pass weight.
        smooth_mu : float
            Taubin backward-pass weight (negative).
        trim_bounds : tuple or dict
            ``(xmin%, ymin%, xmax%, ymax%)`` as percentages (0–100) of each
            slice's LFI dimensions, applied uniformly, or
            ``{sk: (xmin%, ymin%, xmax%, ymax%)}`` for per-slice values.
            Converted to pixel coordinates internally using ``gs.lgi.shape``.
        coord_decimals : int
            Decimal places for junction-point coordinate rounding.
        verbose : bool
            Print per-slice progress.
        **seed_kwargs
            Forwarded to ``generate_constrained_hybrid_seeds`` when seeds
            are generated internally (e.g. ``target_spacing``, ``bulk_spacing``).

        Returns
        -------
        dict
            ``{sk: smooth_gs_slice_result_dict}`` — also stored in
            ``self.mc_smooth_geom``.
        """
        from upxo.pxtalops.gssmooth2d import smooth_gs_slice

        if slice_keys is None:
            slice_keys = list(cntr.gsset.keys())

        bounds_map = (trim_bounds if isinstance(trim_bounds, dict)
                      else {sk: trim_bounds for sk in slice_keys})

        results: dict = {}
        for sk in slice_keys:
            if verbose:
                print(f'[smooth_mc_slices] processing slice {sk}')
            gs = cntr.gsset[sk]
            ny, nx = gs.lgi.shape
            pct = bounds_map[sk]
            trim_px = (
                round(pct[0] / 100 * nx),
                round(pct[1] / 100 * ny),
                round(pct[2] / 100 * nx),
                round(pct[3] / 100 * ny),
            )
            results[sk] = smooth_gs_slice(
                gs.lgi,
                seeds=None,
                area_threshold=area_threshold,
                smooth_iter=smooth_iter,
                smooth_lambda=smooth_lambda,
                smooth_mu=smooth_mu,
                trim_bounds=trim_px,
                coord_decimals=coord_decimals,
                verbose=verbose,
                method=method,
                ma_window=ma_window,
                corner_angle_deg=corner_angle_deg,
                upscale_factor=upscale_factor,
                thin_grain_px=thin_grain_px,
                fix_diagonal=fix_diagonal,
                merge_enclosed=merge_enclosed,
                close_staircase=close_staircase,
                **seed_kwargs,
            )

        self.mc_smooth_geom = results
        return self.mc_smooth_geom

    def assign_smooth_orientations(
            self,
            cntr,
            mc_twin_geom: dict,
            slice_keys: list | None = None,
    ) -> dict:
        """
        Map crystallographic orientations from the pixellated twinned MC grain
        structure onto the smoothed polygon grain structure stored in
        ``self.mc_smooth_geom``.

        For each smoothed polygon, Shapely's ``representative_point()``
        (guaranteed strictly inside the polygon) is queried against a
        ``cKDTree`` of pixel centres built from the original pixellated LFI.
        The grain ID at the nearest pixel is looked up in
        ``mc_twin_geom[sk]['all_quats']``.

        This approach is robust to three failure modes that break centroid-
        or ``old_to_new_gid``-based mapping:
        (1) ``cc3d`` GID relabeling inside ``_merge_small_grains``;
        (2) ``trim_to_rve`` clipping of boundary polygons;
        (3) thin twin lamellae fragmenting into multiple globular polygons.

        Result stored in ``self.mc_smooth_quats = {sk: {new_gid: quaternion}}``.

        Parameters
        ----------
        cntr : MC_GS_Container2d
            Container with ``.gsset {sk: gs}``; ``gs.lgi`` is the pixellated LFI.
        mc_twin_geom : dict
            Output of :meth:`introduce_mc_twin_lamellae` — provides
            ``'all_quats' {orig_gid: np.ndarray(4,)}`` per slice.
        slice_keys : list or None
            Slices to process. ``None`` processes all keys in
            ``self.mc_smooth_geom``.

        Returns
        -------
        dict
            ``{sk: {new_gid: np.ndarray(4,)}}`` — also stored in
            ``self.mc_smooth_quats``.
        """
        from scipy.spatial import cKDTree
        import numpy as np

        if slice_keys is None:
            slice_keys = list(self.mc_smooth_geom.keys())

        result_quats = {}
        for sk in slice_keys:
            lfi_px = cntr.gsset[sk].lgi          # original pixellated GIDs
            all_quats = mc_twin_geom[sk]['all_quats']
            result = self.mc_smooth_geom[sk]

            ny, nx = lfi_px.shape
            rows, cols = np.indices((ny, nx))
            centers = np.column_stack([cols.ravel().astype(float),
                                       rows.ravel().astype(float)])
            gid_flat = lfi_px.ravel()
            tree = cKDTree(centers)

            gid_quats = {}
            for new_gid, geom in result['cells'].items():
                rp = geom.representative_point()   # (rp.x=col, rp.y=row)
                _, idx = tree.query([rp.x, rp.y])
                orig_gid = int(gid_flat[idx])
                if orig_gid in all_quats:
                    gid_quats[new_gid] = all_quats[orig_gid]

            result_quats[sk] = gid_quats

        self.mc_smooth_quats = result_quats
        return self.mc_smooth_quats

    @staticmethod
    def _poly_ipf_colors(quats_dict: dict, sample_direction=(0., 0., 1.)) -> dict:
        """
        Return ``{gid: ndarray(3,)}`` IPF RGB colour for each grain.
        Uses the same ``|R(q)ᵀ · sd|`` formula as ``build_ipf_rgb``.
        Grains absent from *quats_dict* are not included — caller supplies grey.
        """
        import numpy as np
        sd = np.asarray(sample_direction, dtype=np.float64)
        sd /= np.linalg.norm(sd) + 1e-12
        colors = {}
        for gid, q in quats_dict.items():
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            R = np.array([
                [1 - 2*(y*y + z*z),  2*(x*y - z*w),  2*(x*z + y*w)],
                [    2*(x*y + z*w),  1 - 2*(x*x + z*z),  2*(y*z - x*w)],
                [    2*(x*z - y*w),  2*(y*z + x*w),  1 - 2*(x*x + y*y)],
            ])
            cd = np.abs(R.T @ sd)
            colors[int(gid)] = cd / (cd.max() + 1e-12)
        return colors

    def plot_smooth_ipf_comparison(
            self,
            cntr,
            mc_twin_geom: dict,
            slice_keys: list | None = None,
            sample_direction: tuple = (0., 0., 1.),
            show_pixellated: bool = True,
            ncols: int | None = None,
            figsize_per_panel: tuple = (4, 4),
            dpi: int = 150,
            fontsize: float = 9.0,
            lw_poly: float = 0.3,
            ec_poly: str = 'k',
    ) -> None:
        """
        IPF map comparison: pixellated grain structure vs smoothed polygon
        grain structure, one row per MC slice.

        Parameters
        ----------
        cntr : MC_GS_Container2d
        mc_twin_geom : dict
            Output of :meth:`introduce_mc_twin_lamellae` — provides
            ``'all_quats'`` orientation dict per slice.
        slice_keys : list or None
            Slices to plot. Defaults to all keys in ``self.mc_smooth_geom``.
        sample_direction : tuple of 3 floats
            IPF projection direction (ND = (0,0,1)).
        show_pixellated : bool
            ``True`` — two-column layout: pixellated (left) | smoothed (right).
            ``False`` — smoothed only; ``ncols`` controls the grid.
        ncols : int or None
            Column count when ``show_pixellated=False``. ``None`` → one row.
        figsize_per_panel : (float, float)
            Width × height (inches) of each panel.
        dpi, fontsize, lw_poly, ec_poly
            Appearance parameters.
        """
        import math
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
        import upxo.viz.ebsdviz as _ebsdviz

        if slice_keys is None:
            slice_keys = list(self.mc_smooth_geom.keys())

        n = len(slice_keys)
        grey = np.array([0.75, 0.75, 0.75])

        if show_pixellated:
            nr, nc = n, 2
        else:
            nc = ncols or n
            nr = math.ceil(n / nc)

        fw = figsize_per_panel[0] * nc
        fh = figsize_per_panel[1] * nr
        fig, axes = plt.subplots(nr, nc, figsize=(fw, fh), dpi=dpi,
                                 squeeze=False)

        for i, sk in enumerate(slice_keys):
            result = self.mc_smooth_geom[sk]
            all_quats = mc_twin_geom[sk]['all_quats']

            if show_pixellated:
                # ── left: pixellated IPF ─────────────────────────────────────
                ax_px = axes[i, 0]
                rgb = _ebsdviz.build_ipf_rgb(
                    cntr.gsset[sk].lgi, all_quats, sample_direction)
                ax_px.imshow(rgb, origin='upper', interpolation='nearest')
                ax_px.set_title(f'MC {sk} · px', fontsize=fontsize)
                ax_px.axis('off')
                ax_smooth = axes[i, 1]
            else:
                row, col = divmod(i, nc)
                ax_smooth = axes[row, col]

            # ── smoothed polygon IPF ──────────────────────────────────────────
            ipf_colors = self._poly_ipf_colors(
                self.mc_smooth_quats.get(sk, {}), sample_direction
            )

            patches, facecolors = [], []
            for new_gid, geom in result['cells'].items():
                color = ipf_colors.get(new_gid, grey)
                polys = ([geom] if isinstance(geom, ShapelyPolygon)
                         else list(geom.geoms) if isinstance(geom, MultiPolygon)
                         else [])
                for poly in polys:
                    coords = np.array(poly.exterior.coords)
                    patches.append(MplPolygon(coords, closed=True))
                    facecolors.append(color)

            pc = PatchCollection(patches, facecolors=facecolors,
                                 edgecolors=ec_poly, linewidths=lw_poly)
            ax_smooth.add_collection(pc)
            ny_lfi, nx_lfi = cntr.gsset[sk].lgi.shape
            ax_smooth.set_xlim(0, nx_lfi)
            ax_smooth.set_ylim(ny_lfi, 0)   # match imshow origin='upper'
            ax_smooth.set_aspect('equal')
            title = (f'MC {sk} · smooth' if show_pixellated else f'MC {sk}')
            ax_smooth.set_title(title, fontsize=fontsize)
            ax_smooth.axis('off')

        # hide any unused axes when show_pixellated=False
        if not show_pixellated:
            for j in range(n, nr * nc):
                row, col = divmod(j, nc)
                axes[row, col].axis('off')

        fig.tight_layout()
        plt.show()

    def plot_ebsd_tvf(
            self,
            tvf_result: dict,
            figsize: tuple = (7, 4),
            dpi: int = 100,
            fontsize: float = 9.0,
            title: str = 'EBSD grain-role area fractions',
    ) -> None:
        """
        Horizontal bar chart of EBSD grain-role area fractions.

        Parameters
        ----------
        tvf_result : dict
            Output of :meth:`compute_ebsd_tvf`.
        figsize, dpi, fontsize, title
            Forwarded to the core vizDistr function.
        """
        from upxo.viz.vizDistr import plot_ebsd_tvf as _plot
        _plot(tvf_result, figsize=figsize, dpi=dpi,
              fontsize=fontsize, title=title)

    def plot_parent_twin_map(self, parent_info, **kwargs):
        """
        Render a per-CSL spatial grain map coloured by parent / twin /
        intermediate role on the EBSD label field.

        One subplot is produced per CSL type found in *parent_info*.  Within
        each subplot grains are coloured as: pure parents (blue), pure twins
        (coral), intermediates (green), non-role grains (semi-transparent grey).
        Wraps ``ebsdviz.plot_parent_twin_map`` using ``self.lfi_ebsd``.

        Parameters
        ----------
        parent_info : dict
            Output of ``identify_parent_grains()``.  Keys are CSL labels;
            each value must contain ``'pure_parents'``, ``'pure_twins'``,
            ``'intermediates'``, ``'n_pure_parents'``, ``'n_pure_twins'``,
            and ``'n_intermediates'``.
        **kwargs
            - ``map_figsize``  : tuple — figure size (default ``(6, 6)``)
            - ``map_dpi``      : int — DPI (default ``140``)
            - ``map_suptitle`` : str — super-title for the figure

        Returns
        -------
        None
            The figure is rendered via ``plt.show()``.
        """
        from upxo.viz.ebsdviz import plot_parent_twin_map
        plot_parent_twin_map(self.lfi_ebsd, parent_info,
            figsize=kwargs.get('map_figsize', (6, 6)), dpi=kwargs.get('map_dpi', 140),
            suptitle=kwargs.get('map_suptitle', 'Parent / twin grain map — EBSD target'),)
        plt.show()

    def plot_combined_parent_twin_map(self, parent_info, **kwargs):
        """
        Render a single spatial grain map with pure-parent, pure-twin, and
        intermediate grains aggregated across all CSL types in *parent_info*.

        Unlike ``plot_parent_twin_map`` which produces one subplot per CSL
        type, this produces a single combined map.  Priority when a grain
        appears in multiple roles: intermediate > parent > twin.
        Wraps ``ebsdviz.plot_combined_parent_twin_map`` using ``self.lfi_ebsd``.

        Parameters
        ----------
        parent_info : dict
            Output of ``identify_parent_grains()``.  Keys are CSL labels;
            each value must contain ``'pure_parents'``, ``'pure_twins'``,
            and ``'intermediates'``.
        **kwargs
            - ``combmap_figsize``   : tuple — figure size (default ``(6, 6)``)
            - ``combmap_dpi``       : int — DPI (default ``140``)
            - ``combmap_alpha_bg``  : float — opacity of non-role grains
              (default ``0.08``)
            - ``combmap_suptitle``  : str — super-title for the figure

        Returns
        -------
        None
            The figure is rendered via ``plt.show()``.
        """
        from upxo.viz.ebsdviz import plot_combined_parent_twin_map
        plot_combined_parent_twin_map(self.lfi_ebsd, parent_info,
            figsize=kwargs.get('combmap_figsize', (6, 6)), dpi=kwargs.get('combmap_dpi', 140),
            alpha_bg=kwargs.get('combmap_alpha_bg', 0.08),
            suptitle=kwargs.get('combmap_suptitle', 'Parent / twin grain map — EBSD target'),)
        plt.show()

    def selectProps_twinGS(self):
        """
        Launch the ipywidgets control panel for selecting grain-role
        property plots.

        Thin wrapper around ``nbWidgets.selectProps_twinGS``.  Displays
        checkboxes for morphological properties and grain groups, plus
        sliders for subplot columns and font size.  Must be called inside
        a Jupyter cell.

        Returns
        -------
        dict
            Widget state dict with keys ``'prop_checkboxes'``,
            ``'group_checkboxes'``, ``'ncols_slider'``, and
            ``'fontsize_slider'``.  Pass to ``readProps_twinGS`` to
            extract current values before plotting.
        """
        from upxo.interfaces.user_inputs.nbWidgets import selectProps_twinGS
        _widgets = selectProps_twinGS()
        return _widgets

    def readProps_twinGS(self, _widgets):
        """
        Read the current widget values from the panel returned by
        ``selectProps_twinGS``.

        Thin wrapper around ``nbWidgets.readProps_twinGS``.  Prints the
        selected properties and groups to stdout and returns them as a
        plain dict.

        Parameters
        ----------
        _widgets : dict
            The dict returned by ``selectProps_twinGS()``.

        Returns
        -------
        dict
            ``{'selected_props': list, 'selected_groups': list,
               'ncols': int | None, 'fontsize': float}``
        """
        from upxo.interfaces.user_inputs.nbWidgets import readProps_twinGS
        return readProps_twinGS(_widgets)

    def selectProps_reprComp(self, props=None):
        """
        Launch the property-selection checklist for MC–EBSD comparison.

        Thin wrapper around ``nbWidgets.selectProps_reprComp``.  Displays
        only morphological property checkboxes (no grain-group controls or
        layout sliders).  Must be called inside a Jupyter cell.

        Parameters
        ----------
        props : list of str, optional
            Property names to show.  Defaults to the standard shared set.

        Returns
        -------
        dict
            ``{'prop_checkboxes': dict}`` — pass to ``readProps_reprComp``.
        """
        from upxo.interfaces.user_inputs.nbWidgets import selectProps_reprComp as _w
        return _w(props=props)

    def readProps_reprComp(self, _widgets):
        """
        Read selected property names from the panel returned by
        ``selectProps_reprComp``.

        Thin wrapper around ``nbWidgets.readProps_reprComp``.

        Parameters
        ----------
        _widgets : dict
            The dict returned by ``selectProps_reprComp()``.

        Returns
        -------
        list of str
            Names of the ticked properties.
        """
        from upxo.interfaces.user_inputs.nbWidgets import readProps_reprComp as _r
        return _r(_widgets)

    def see_grain_role_property_stats(self, parent_info, twinProps, **kwargs):
        """
        Plot morphological and topological property distributions split by
        grain role (pure parents, pure twins, intermediates, non-role) for
        the EBSD target.

        Delegates to ``ebsdviz.plot_grain_role_property_stats`` using
        ``self.lfi_ebsd``, ``self.prop_ebsd``, and ``self.neigh_gid_ebsd``.

        Parameters
        ----------
        parent_info : dict
            Output of ``identify_parent_grains()``.  Keys are CSL labels;
            values contain ``'pure_parents'``, ``'pure_twins'``, and
            ``'intermediates'`` grain-ID arrays.
        twinProps : dict
            Output of ``readProps_twinGS()``.  Must contain
            ``'selected_props'``, ``'selected_groups'``, ``'ncols'``,
            and ``'fontsize'``.
        **kwargs
            Optional overrides forwarded to ``plot_grain_role_property_stats``:

            - ``nbins``            : int   — histogram bins (default 40)
            - ``bw_method``        : str   — KDE bandwidth (default ``'scott'``)
            - ``peak_prominence``  : float — peak detection threshold (default 0.02)
            - ``figsize_per``      : tuple — per-subplot figure size (default ``(6, 4)``)
            - ``dpi``              : int   — figure DPI (default 100)
            - ``suptitle``         : str   — figure super-title

        Returns
        -------
        None
            The figure is rendered inline via matplotlib.

        Examples
        --------
        Typical Jupyter notebook workflow (one cell per step):

        .. code-block:: python

            # Cell 1 — build the widget panel and confirm selection
            _widgets = rgen.selectProps_twinGS()

            # Cell 2 — read widget values after confirming
            twinProps = rgen.readProps_twinGS(_widgets)

            # Cell 3 — plot with defaults
            rgen.see_grain_role_property_stats(parent_info, twinProps)

            # Cell 4 — override specific display options
            rgen.see_grain_role_property_stats(
                parent_info, twinProps,
                nbins=60,
                figsize_per=(7, 4),
                dpi=130,
                suptitle='Cu OFHC — grain role distributions',
            )
        """
        from upxo.viz.ebsdviz import plot_grain_role_property_stats
        fig, axes = plot_grain_role_property_stats(
            lfi         = self.lfi_ebsd,
            parent_info = parent_info,
            prop        = self.prop_ebsd,
            neigh_gid   = self.neigh_gid_ebsd,
            selected_props  = twinProps['selected_props'],
            selected_groups = twinProps['selected_groups'],
            step_size   = self.ebsd_step,
            bins        = kwargs.get('nbins', 40),
            bw_method   = kwargs.get('bw_method', 'scott'),
            peak_prominence = kwargs.get('peak_prominence', 0.02),
            figsize_per = kwargs.get('figsize_per', (6, 4)),
            dpi         = kwargs.get('dpi', 100),
            suptitle    = kwargs.get('suptitle', 'Morphological & topological statistics by grain role — EBSD target'),
            ncols       = twinProps['ncols'],
            fontsize    = twinProps['fontsize'],
        )

    def sgs_mcgs2Gen(self, INPUT_DASHBOARD, MC_TIME_START=2, MC_TIME_END=-1,
                     MC_TIME_STEP=2, show_stats_table=False,
                     figsize=(5, 4), dpi=110, **kwargs):
        """
        Generate a sample Monte-Carlo grain structure set, characterise it
        across temporal slices, and plot the temporal property distributions.

        Parameters
        ----------
        INPUT_DASHBOARD : str or path
            Path to the UPXO MC simulation input dashboard.
        MC_TIME_START : int, optional
            First MC time slice index to characterise.  Default 2.
        MC_TIME_END : int, optional
            Last MC time slice index (exclusive, ``-1`` = all).  Default -1.
        MC_TIME_STEP : int, optional
            Step between characterised time slices.  Default 2.
        show_stats_table : bool, optional
            If ``True``, print the per-property statistics table after the
            figure.  Default ``False`` — suppresses the table while still
            storing it in ``cntr.stats_table``.
        figsize_per : tuple of (float, float), optional
            ``(width, height)`` in inches for each subplot panel.  Default
            ``(5, 4)``.  Forwarded to
            :meth:`MC_GS_Container2d.plot_temporal_distributions`.
        dpi : int, optional
            Figure resolution in dots-per-inch.  Default ``110``.  Forwarded
            to :meth:`MC_GS_Container2d.plot_temporal_distributions`.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`MC_GS_Container2d.plot_temporal_distributions`.
            Accepted keys include ``props``, ``ncols``, ``bins``,
            ``bw_method``, ``peak_prominence``, ``fontsize``, ``suptitle``,
            ``cmap``.  Any key supplied here overrides the corresponding
            default set inside this method.

        Returns
        -------
        cntr : MC_GS_Container2d
            Populated container with ``gsset`` and ``stats_table``.
        """
        import upxo.gsContainters.mcgs2Cont as gs_cntnr2_mc
        MC_GS_Container2d = gs_cntnr2_mc.MC_GS_Container2d

        cntr = MC_GS_Container2d.by_upxoMCSIM_gsset_GEN(
            indb=INPUT_DASHBOARD,
            mctimeStart=MC_TIME_START,
            mctimeStep=MC_TIME_STEP,
            mctimeEnd=MC_TIME_END,
        )

        plot_kwargs = dict(
            props=['area', 'aspect_ratio', 'solidity',
                   'major_axis_length', 'minor_axis_length'],
            ncols=2, cmap='plasma', fontsize=12.0,
            suptitle='Grain-property evolution across MC time slices',
            figsize_per=figsize,
            dpi=dpi,
            show_stats_table=show_stats_table,
        )
        plot_kwargs.update(kwargs)

        fig, axes = cntr.plot_temporal_distributions(**plot_kwargs)
        plt.show()
        return cntr

    # ------------------------------------------------------------------
    # MC–EBSD representativeness ranking (twin-aware)
    # ------------------------------------------------------------------

    def rank_mcgs_by_n(self, cntr, P: float = 10.0):
        """
        Rank all MC grain structures in ``cntr.gsset`` by grain count
        relative to the de-twinned EBSD map.

        Requires :meth:`build_merged_ebsd_lfi` to have been called first.

        Parameters
        ----------
        cntr : MC_GS_Container2d
            Container with ``gsset`` dict.
        P : float, optional
            Tolerance in % below the de-twinned EBSD grain count used to
            determine eligibility.  A slice is **ineligible** (0) when
            ``n_mc < (1 - P/100) * n_ebsd_merged``; eligible (1) otherwise.
            Default 10.

        Returns
        -------
        pd.DataFrame
            Indexed by ``'mc_time_slice'``, columns:

            ``n_mc``          — grain count of the MC slice
            ``n_ebsd_merged`` — grain count of the de-twinned EBSD (constant)
            ``ratio``         — ``n_mc / n_ebsd_merged``; >1 = more grains
                                than EBSD; <1 = fewer grains
            ``eligible``      — 1 if ``ratio >= 1 - P/100``, else 0
            ``nPixQ1SimGS``   — Q1 (25th percentile) of the per-grain pixel
                                count distribution across all grains in the MC
                                slice.  A grain with ``npixels`` at or below
                                this value is among the smallest 25 % of grains
                                in that slice.

                                *Relevance to twin introduction*

                                Twin lamellae are introduced into individual
                                parent grains.  The smallest grains (lower
                                quartile) are the hardest cases: a grain with
                                very few pixels can only accommodate a 1–2
                                pixel-wide twin regardless of the intended
                                physical thickness, collapsing the twin to a
                                featureless stripe with no resolvable area,
                                aspect ratio, or boundary orientation.

                                ``nPixQ1SimGS`` captures this worst-case pixel
                                budget.  Higher values mean even the smallest
                                quartile of grains is pixel-rich, so twin
                                lamellae can span several pixels in their
                                thickness dimension — yielding realistic aspect
                                ratios, well-defined boundaries, and volume
                                fractions that faithfully reflect the intended
                                twin thickness parameter.  In short, higher
                                values make twin introduction more accurate by
                                reducing discretisation error in the twin
                                geometry.

                                Note: this is a purely discretisation argument.
                                Physical accuracy also depends on the pixel step
                                size; both should be considered together when
                                choosing a representative MC time slice.

            Sorted ascending by ``ratio`` deviation from 1.  Also stored in
            :attr:`grain_count_rank_ng`.
        """
        import pandas as pd
        from IPython.display import display as _display

        n_ebsd    = len(self.prop_ebsd_merged_df)
        threshold = 1.0 - P / 100.0
        rows = {}
        for k, gs in cntr.gsset.items():
            ratio = gs.n / n_ebsd
            # Prefer the explicit npixels column (enabled by default from
            # mcgs2Cont._CHAR_DEFAULTS npixels=True).  Fall back to area:
            # for MC simulations px_size=1 so area == npixels numerically.
            if 'npixels' in gs.prop.columns:
                npix_q1 = float(gs.prop['npixels'].quantile(0.25))
            elif 'area' in gs.prop.columns:
                npix_q1 = float(gs.prop['area'].quantile(0.25))
            else:
                npix_q1 = float('nan')
            rows[k] = {
                'n_mc':          gs.n,
                'n_ebsd_merged': n_ebsd,
                'ratio':         ratio,
                'eligible':      int(ratio >= threshold),
                'nPixQ1SimGS':   npix_q1,
            }
        df = pd.DataFrame.from_dict(rows, orient='index')
        df.index.name = 'mc_time_slice'
        df['eligible'] = df['eligible'].astype(int)
        df = df.sort_values('ratio', key=lambda s: (s - 1).abs())
        self.grain_count_rank_ng = df
        _display(df)
        return df

    def selectSlices_reprComp(self):
        """
        Show an ipywidgets checklist of MC time slices from
        ``grain_count_rank_ng`` for the user to select which slices to carry
        forward into property-distribution comparison.

        Requires :meth:`rank_mcgs_by_n` to have been called first.

        Returns
        -------
        dict
            ``{'slice_checkboxes': dict}`` — pass to
            :meth:`readSlices_reprComp`.
        """
        from upxo.interfaces.user_inputs.nbWidgets import selectSlices_reprComp as _w
        return _w(self.grain_count_rank_ng)

    def readSlices_reprComp(self, _widgets):
        """
        Read selected MC time-slice keys from the panel returned by
        :meth:`selectSlices_reprComp`.

        Parameters
        ----------
        _widgets : dict
            The dict returned by ``selectSlices_reprComp()``.

        Returns
        -------
        list
            MC time-slice keys for the ticked entries.
        """
        from upxo.interfaces.user_inputs.nbWidgets import readSlices_reprComp as _r
        return _r(_widgets)

    def find_repr_mcgs_props(
            self,
            cntr,
            mc_slices: list,
            props=None,
    ):
        """
        Score user-selected MC grain structures against ``prop_ebsd_merged_df``
        using three distribution-similarity metrics.

        Call :meth:`rank_mcgs_by_n` first, inspect the table, select slices
        via :meth:`selectSlices_reprComp` / :meth:`readSlices_reprComp`, then
        pass the result here.

        Parameters
        ----------
        cntr : MC_GS_Container2d
            Container with ``gsset`` dict.
        mc_slices : list
            Explicit list of ``mc_time_slice`` keys (from
            ``grain_count_rank_ng.index``) chosen by the user.
        props : list of str or None
            Properties to compare.  If None, the intersection of float columns
            shared between ``prop_ebsd_merged_df`` and the first candidate's
            ``prop`` DataFrame is used automatically.  Supply the list from
            :meth:`readProps_reprComp` to use a user-selected subset.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys ``'ratio'``, ``'wasserstein'``, ``'energy'`` — each a
            DataFrame of per-property scores sorted ascending by
            ``'aggregate'`` (best match first).  Also stored in
            :attr:`repr_rank_ng`.
        """
        import pandas as pd
        from scipy.stats import (wasserstein_distance, energy_distance,
                                  ks_2samp, anderson_ksamp)

        candidates = {k: cntr.gsset[k] for k in mc_slices if k in cntr.gsset}
        if not candidates:
            raise RuntimeError(
                f"None of the requested mc_slices {mc_slices} found in "
                "cntr.gsset.  Check the keys against grain_count_rank_ng.index."
            )

        # Step B — resolve shared float properties
        # Always intersect with MC columns regardless of whether props was
        # user-supplied or auto-detected; some EBSD properties (e.g.
        # eccentricity) may not exist in the MC prop DataFrame.
        sample_gs = next(iter(candidates.values()))
        mc_cols   = set(sample_gs.prop.columns)
        if props is None:
            props = [c for c in self.prop_ebsd_merged_df.columns
                     if c in mc_cols
                     and self.prop_ebsd_merged_df[c].dtype.kind == 'f']
        else:
            props = [p for p in props
                     if p in mc_cols
                     and p in self.prop_ebsd_merged_df.columns]
        if not props:
            raise RuntimeError(
                "No properties in common between prop_ebsd_merged_df and "
                "the MC grain structure prop DataFrame. Check prop column names."
            )
        print(f'Comparing on properties: {props}')

        # Step C — normalisers
        # ratio uses EBSD mean so the cell value = MC_mean / EBSD_mean.
        # Wasserstein / energy each normalise by their own distribution mean so
        # location (mean offset) is removed and only shape is compared.
        ebsd_means = {p: self.prop_ebsd_merged_df[p].dropna().mean()
                      for p in props}

        def _norm_by_ebsd(series, p):
            mu = ebsd_means[p]
            vals = series.dropna().values
            return vals / mu if mu != 0 else vals

        def _norm_by_own(series):
            vals = series.dropna().values
            mu = vals.mean()
            return vals / mu if mu != 0 else vals

        # Step D — score each candidate with five metrics
        ratio_rows, wd_rows, ed_rows, ks_rows, ad_rows = {}, {}, {}, {}, {}
        for k, gs in candidates.items():
            r_row, wd_row, ed_row, ks_row, ad_row = {}, {}, {}, {}, {}
            for p in props:
                r_row[p]  = float(_norm_by_ebsd(gs.prop[p], p).mean())
                a = _norm_by_own(self.prop_ebsd_merged_df[p])
                b = _norm_by_own(gs.prop[p])
                wd_row[p] = float(wasserstein_distance(a, b))
                ed_row[p] = float(energy_distance(a, b))
                ks_row[p] = float(ks_2samp(a, b).statistic)
                ad_row[p] = float(anderson_ksamp([a, b]).statistic)
            r_row['aggregate']  = float(np.nanmean([abs(v - 1) for v in r_row.values()
                                                     if not np.isnan(v)]))
            wd_row['aggregate'] = float(np.nanmean(list(wd_row.values())))
            ed_row['aggregate'] = float(np.nanmean(list(ed_row.values())))
            ks_row['aggregate'] = float(np.nanmean(list(ks_row.values())))
            ad_row['aggregate'] = float(np.nanmean(list(ad_row.values())))
            ratio_rows[k] = r_row
            wd_rows[k]    = wd_row
            ed_rows[k]    = ed_row
            ks_rows[k]    = ks_row
            ad_rows[k]    = ad_row

        # Step E — build, store, return
        def _to_df(rows):
            df = pd.DataFrame.from_dict(rows, orient='index')
            df.index.name = 'mc_time_slice'
            return df.sort_values('aggregate')

        result = {
            'ratio':       _to_df(ratio_rows),
            'wasserstein': _to_df(wd_rows),
            'energy':      _to_df(ed_rows),
            'ks':          _to_df(ks_rows),
            'ad':          _to_df(ad_rows),
        }
        self.repr_rank_ng = result
        return result

    def show_repr_rank_ng(self, metric: str = 'wasserstein', n_top=None):
        """
        Display the MC–EBSD ranking table for the given metric.

        Prints the available metric options (with the active one bracketed),
        a note on how the aggregate is computed and what each cell value means
        for the chosen metric, then IPython-displays the DataFrame.

        Parameters
        ----------
        metric : str, optional
            One of ``'ratio'``, ``'wasserstein'``, ``'energy'``.
            Default ``'wasserstein'``.
        n_top : int or None, optional
            If given, show only the top N rows.  Default None (all rows).
        """
        from IPython.display import display as _display

        _METRICS = ('ratio', 'wasserstein', 'energy', 'ks', 'ad')
        _NOTES = {
            'ratio': (
                "  Cell value : mean(MC property) / mean(EBSD property) — "
                "dimensionless mean offset.\n"
                "              1.0 = perfect mean match; "
                ">1.0 = MC mean higher; <1.0 = MC mean lower.\n"
                "  Aggregate  : mean of |cell − 1| across all properties. "
                "0 = perfect mean match. Lower is better."
            ),
            'wasserstein': (
                "  Cell value : Wasserstein (earth-mover) distance between "
                "shape-only distributions\n"
                "              (each normalised by its own mean, so mean offset "
                "is removed; only spread, skewness, and tails are compared).\n"
                "              0 = identical shapes. Lower is better.\n"
                "  Aggregate  : mean of per-property distances. Lower is better."
            ),
            'energy': (
                "  Cell value : Energy distance between shape-only distributions\n"
                "              (each normalised by its own mean, so mean offset "
                "is removed; more tail-sensitive than Wasserstein).\n"
                "              0 = identical shapes. Lower is better.\n"
                "  Aggregate  : mean of per-property distances. Lower is better."
            ),
            'ks': (
                "  Cell value : Kolmogorov–Smirnov statistic between shape-only "
                "distributions\n"
                "              (each normalised by its own mean).\n"
                "              Maximum absolute difference between the two CDFs.\n"
                "              0 = identical shapes. Lower is better.\n"
                "  Aggregate  : mean of per-property statistics. Lower is better."
            ),
            'ad': (
                "  Cell value : Anderson–Darling statistic between shape-only "
                "distributions\n"
                "              (each normalised by its own mean).\n"
                "              More sensitive than KS to differences in the tails.\n"
                "              0 = identical shapes. Lower is better.\n"
                "  Aggregate  : mean of per-property statistics. Lower is better."
            ),
        }

        available = '  |  '.join(
            f'[{m}]' if m == metric else m for m in _METRICS
        )
        print(f'Metric  : {available}')
        print(_NOTES[metric])
        print()

        df = self.repr_rank_ng[metric]
        _display(df.head(n_top) if n_top else df)

    def plot_repr_rank(
            self,
            figsize=None,
            dpi: int = 100,
            fontsize_annot: float = 8.0,
            fontsize_tick: float = 9.0,
            fontsize_title: float = 9.0,
            fontsize_suptitle: float = 11.0,
    ) -> None:
        """
        Three vertically stacked heatmaps of per-property rankings for all
        compared MC slices.

        Green = best-ranked within a column, red = worst.  Cell text shows the
        raw score.  Call after :meth:`find_repr_mcgs_props` has populated
        :attr:`repr_rank_ng`.  Delegates to
        :func:`upxo.viz.vizDistr.plot_repr_rank`.

        Parameters
        ----------
        figsize : tuple or None
        dpi : int
        fontsize_annot : float
            Font size for the numeric value in each cell.
        fontsize_tick : float
            Font size for tick labels (slice keys and column names).
        fontsize_title : float
            Font size for each panel title.
        fontsize_suptitle : float
            Font size for the overall figure title.
        """
        from upxo.viz.vizDistr import plot_repr_rank as _plot
        _plot(self.repr_rank_ng, figsize=figsize, dpi=dpi,
              fontsize_annot=fontsize_annot, fontsize_tick=fontsize_tick,
              fontsize_title=fontsize_title, fontsize_suptitle=fontsize_suptitle)

    def plot_normalized_prop_distributions(
            self,
            cntr,
            mc_slices: list,
            props=None,
            annotate_scores: bool = True,
            bins: int = 40,
            bw_method='scott',
            figsize_per: tuple = (5, 4),
            dpi: int = 100,
            ncols=None,
            fontsize: float = 9.0,
            show_hist: bool = True,
            show_peaks: bool = True,
            legend_loc: str = 'upper right',
            legend_ncol: int = 1,
            legend_fontsize: float | None = None,
    ) -> None:
        """
        Overlaid normalised property distributions for EBSD (merged) vs MC slices.

        Each property distribution is divided by its own mean before plotting,
        matching the normalisation used in :meth:`find_repr_mcgs_props`, so all
        curves are centred near 1.0 and are directly shape-comparable.

        Parameters
        ----------
        cntr : MC_GS_Container2d
        mc_slices : list
            MC time-slice keys to plot (subset of ``cntr.gsset`` keys).
        props : list of str or None
            Properties to plot.  Auto-detected from shared float columns when None.
        annotate_scores : bool
            If True and :attr:`repr_rank_ng` is populated, annotate each MC curve
            with its Wasserstein and energy distance for that property.
        bins, bw_method, figsize_per, dpi, ncols, fontsize, show_hist, show_peaks
            Forwarded to the core vizDistr function.
        legend_loc : str
            Legend location.  Default ``'upper right'``.
        legend_ncol : int
            Number of legend columns.  Use ``2`` or more to split entries
            side-by-side and reduce legend height.  Default ``1``.
        legend_fontsize : float or None
            Legend text size.  Reducing this is the most direct way to shrink the
            legend box.  Defaults to ``fontsize - 2`` when None.
        """
        from upxo.viz.vizDistr import plot_normalized_prop_distributions as _plot

        sample_gs = cntr.gsset[mc_slices[0]]
        mc_cols = set(sample_gs.prop.columns)
        if props is None:
            props = [c for c in self.prop_ebsd_merged_df.columns
                     if c in mc_cols
                     and self.prop_ebsd_merged_df[c].dtype.kind == 'f']
        else:
            props = [p for p in props
                     if p in mc_cols and p in self.prop_ebsd_merged_df.columns]

        def _norm(series):
            vals = series.dropna().values
            mu = vals.mean()
            return vals / mu if mu != 0 else vals

        ebsd_data = {p: _norm(self.prop_ebsd_merged_df[p]) for p in props}
        mc_data   = {k: {p: _norm(cntr.gsset[k].prop[p]) for p in props}
                     for k in mc_slices if k in cntr.gsset}

        scores = None
        if annotate_scores and hasattr(self, 'repr_rank_ng') and self.repr_rank_ng:
            scores = {}
            for k in mc_slices:
                scores[k] = {}
                for p in props:
                    scores[k][p] = {
                        'wasserstein': float(self.repr_rank_ng['wasserstein'].loc[k, p])
                        if k in self.repr_rank_ng['wasserstein'].index and p in self.repr_rank_ng['wasserstein'].columns else float('nan'),
                        'energy': float(self.repr_rank_ng['energy'].loc[k, p])
                        if k in self.repr_rank_ng['energy'].index and p in self.repr_rank_ng['energy'].columns else float('nan'),
                    }

        _plot(ebsd_data, mc_data, props, scores=scores,
              bins=bins, bw_method=bw_method, figsize_per=figsize_per,
              dpi=dpi, ncols=ncols, fontsize=fontsize,
              show_hist=show_hist, show_peaks=show_peaks,
              legend_loc=legend_loc, legend_ncol=legend_ncol,
              legend_fontsize=legend_fontsize)

    def plot_qq(
            self,
            cntr,
            mc_slices: list,
            props=None,
            figsize_per: tuple = (4, 4),
            dpi: int = 100,
            ncols=None,
            fontsize: float = 9.0,
    ) -> None:
        """
        Quantile–Quantile plots of EBSD (merged) vs MC slices per property.

        Both distributions are normalised by their own mean so both axes share
        a dimensionless scale centred near 1.0.  Points on the diagonal indicate
        identical shapes at that quantile; deviations reveal where and how the
        distributions differ.  See :func:`upxo.viz.vizDistr.plot_qq_comparison`
        for full interpretation guidance.

        Parameters
        ----------
        cntr : MC_GS_Container2d
        mc_slices : list
            MC time-slice keys to compare.
        props : list of str or None
            Properties to plot.  Auto-detected when None.
        figsize_per, dpi, ncols, fontsize
            Forwarded to the core vizDistr function.
        """
        from upxo.viz.vizDistr import plot_qq_comparison as _plot

        sample_gs = cntr.gsset[mc_slices[0]]
        mc_cols = set(sample_gs.prop.columns)
        if props is None:
            props = [c for c in self.prop_ebsd_merged_df.columns
                     if c in mc_cols
                     and self.prop_ebsd_merged_df[c].dtype.kind == 'f']
        else:
            props = [p for p in props
                     if p in mc_cols and p in self.prop_ebsd_merged_df.columns]

        def _norm(series):
            vals = series.dropna().values
            mu = vals.mean()
            return vals / mu if mu != 0 else vals

        ebsd_data = {p: _norm(self.prop_ebsd_merged_df[p]) for p in props}
        mc_data   = {k: {p: _norm(cntr.gsset[k].prop[p]) for p in props}
                     for k in mc_slices if k in cntr.gsset}

        _plot(ebsd_data, mc_data, props,
              figsize_per=figsize_per, dpi=dpi, ncols=ncols, fontsize=fontsize)

    def best_match_mcgs_key(self, metric: str = 'wasserstein', n: int = 1, by: str = 'aggregate',):
        """
        Return the best-matching MC time-slice key(s) ranked by a chosen column.

        Parameters
        ----------
        metric : str, optional
            Which metric's table to consult.  One of ``'ratio'``,
            ``'wasserstein'``, ``'energy'``.  Default ``'wasserstein'``.
        n : int, optional
            Number of top matches to return.
            - ``n=1`` (default) — returns a single key (scalar).
            - ``n>1``           — returns a list of the top-``n`` keys,
              ordered best-to-worst.
        by : str, optional
            Column to rank by.  Any property name present in
            ``repr_rank_ng[metric].columns``, or ``'aggregate'`` (default).

            Ranking direction is metric-aware:

            - *wasserstein / energy* — sort ``by`` column ascending; lower
              distance = better match regardless of which column is chosen.
            - *ratio* with a property column — sort by ``|value − 1|``
              ascending; closest to 1.0 (perfect mean match) = best.
            - *ratio* with ``by='aggregate'`` — sort ascending; the aggregate
              column already stores ``mean(|ratio − 1|)``, so lower = better.

        Returns
        -------
        key or list of keys
            MC time-slice key(s) from ``repr_rank_ng[metric].index``.

        Raises
        ------
        ValueError
            If ``by`` is not a column in ``repr_rank_ng[metric]``.

        Examples
        --------
        >>> rg.best_match_mcgs_key()                         # top-1, Wasserstein aggregate
        >>> rg.best_match_mcgs_key('energy', n=3)            # top-3 by energy aggregate
        >>> rg.best_match_mcgs_key('wasserstein', by='area') # top-1 by area shape distance
        >>> rg.best_match_mcgs_key('ratio', by='area')       # top-1 by area mean offset
        """
        _METRICS = ('ratio', 'wasserstein', 'energy', 'ks', 'ad')
        available_metrics = '  |  '.join(
            f'[{m}]' if m == metric else m for m in _METRICS
        )
        df = self.repr_rank_ng[metric]
        available_by = '  |  '.join(
            f'[{col}]' if col == by else col for col in df.columns
        )
        print(f'Metric  : {available_metrics}')
        print(f'Rank by : {available_by}')
        if by not in df.columns:
            raise ValueError(
                f"'{by}' is not a column in repr_rank_ng['{metric}']. "
                f"Available: {list(df.columns)}"
            )
        if metric == 'ratio' and by != 'aggregate':
            sort_series = (df[by] - 1.0).abs()
        else:
            sort_series = df[by]
        sorted_idx = sort_series.sort_values().index
        if n == 1:
            return sorted_idx[0]
        return list(sorted_idx[:n])

    def mesh_smooth_slices(
            self,
            slice_keys: list | None = None,
            mesh_size_gb: float = 0.75,
            mesh_size_bulk: float = 4.5,
            mesh_order: int = 1,
            mesh_algo: int = 8,
            recombine_to_quads: bool = True,
            dist_min: float = 0.5,
            dist_max: float = 5.0,
            out_dir: str | None = None,
            basename: str = 'repgen_gs_mesh',
            formats: list | None = None,
            verbose: bool = True,
    ) -> dict:
        """
        Generate conformal FE meshes for smoothed grain structure slices.

        Reads smoothed polygon geometry from ``self.mc_smooth_geom`` and calls
        :func:`upxo.meshing.gsmesh2d.mesh_gs` for each slice.

        Result stored in ``self.mc_smooth_mesh = {sk: mesh_result_dict}``.

        Parameters
        ----------
        slice_keys       : Slice keys to process.  Defaults to all keys in
                           ``self.mc_smooth_geom``.
        mesh_size_gb     : Target element size on grain boundaries.
        mesh_size_bulk   : Target element size in grain interiors.
        mesh_order       : Element order (1=linear, 2=quadratic).
        mesh_algo        : Gmsh algorithm ID (8=Frontal-Delaunay quads, 6=Frontal).
        recombine_to_quads: Recombine triangles into quads.
        dist_min         : Distance field DistMin.
        dist_max         : Distance field DistMax.
        out_dir          : Directory for exported mesh files.
        basename         : Filename stem (per-slice suffix ``_sk<sk>`` appended).
        formats          : List of format extensions, e.g. ``['msh', 'inp']``.
        verbose          : Print per-slice progress.

        Returns
        -------
        dict  {sk: mesh_result_dict}  — also stored in ``self.mc_smooth_mesh``.
        """
        from upxo.meshing.gsmesh2d import mesh_gs
        if slice_keys is None:
            slice_keys = list(self.mc_smooth_geom.keys())
        results: dict = {}
        for sk in slice_keys:
            if verbose:
                print(f'[mesh_smooth_slices] meshing slice {sk}')
            cells = self.mc_smooth_geom[sk]['cells']
            sfx = f'{basename}_sk{sk}' if basename else f'gs_mesh_sk{sk}'
            results[sk] = mesh_gs(
                cells,
                method='conformal',
                mesh_size_gb=mesh_size_gb,
                mesh_size_bulk=mesh_size_bulk,
                mesh_order=mesh_order,
                mesh_algo=mesh_algo,
                recombine_to_quads=recombine_to_quads,
                dist_min=dist_min,
                dist_max=dist_max,
                out_dir=out_dir,
                basename=sfx,
                formats=formats,
                verbose=verbose,
            )
            if verbose:
                r = results[sk]
                print(f'  → {r["n_nodes"]:,} nodes, {r["n_tri"]:,} tri, '
                      f'{r["n_quad"]:,} quad  ({r["elapsed"]:.2f}s)')
        self.mc_smooth_mesh = results
        return self.mc_smooth_mesh

    def visualize_smooth_meshes(
            self,
            slice_keys: list | None = None,
            figsize: tuple = (12, 9),
            dpi: int = 150,
            **kwargs,
    ) -> dict:
        """
        Visualize FE meshes for smoothed slices stored in ``self.mc_smooth_mesh``.

        Parameters
        ----------
        slice_keys : Slice keys to visualize.  Defaults to all keys in
                     ``self.mc_smooth_mesh``.
        figsize    : Figure size in inches.
        dpi        : Figure DPI.
        **kwargs   : Forwarded to :func:`upxo.meshing.gsmesh2d.visualize_gs_mesh`.

        Returns
        -------
        dict  {sk: (fig, ax)}
        """
        from upxo.meshing.gsmesh2d import visualize_gs_mesh
        if slice_keys is None:
            slice_keys = list(self.mc_smooth_mesh.keys())
        figs: dict = {}
        for sk in slice_keys:
            figs[sk] = visualize_gs_mesh(
                self.mc_smooth_mesh[sk],
                figsize=figsize,
                dpi=dpi,
                **kwargs,
            )
        return figs


# _char_lfi has moved to upxo.interfaces.defdap.ebsd_reader as a
# module-level helper.  Import it here for any legacy internal callers.
from upxo.interfaces.defdap.ebsd_reader import _char_lfi  # noqa: F401
