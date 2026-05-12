import os
import warnings
import numpy as np
from copy import deepcopy
from upxo._sup import dataTypeHandlers as dth
from upxo.repqual.grain_network_repr_assesser import KREPR
from upxo.analysis.analysis2d import gsan2d
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', DeprecationWarning)


class repgen2d:

    __slots__ = ('tdist', 'tstat', 'tgs', 'sgs',
                 'iroute', 'mpflags', 'rm0tests', 'rm0',
                 'sgstype', 'tgstype', 'tdim', 'px_size',
                 'sdim', 'gsan_sgs', 'gsan_tgs', 'ebsd_file',
                 'lfi_ebsd', 'euler_ebsd', 'quat_ebsd', 'neigh_gid_ebsd',
                 'prop_ebsd', 'stat_ebsd')
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
        # -------------------------------------------------
        import upxo.viz.ebsdviz as ebsdviz
        ebsdviz.plot_mdf(mdf, peaks)
        plt.show()
        # -------------------------------------------------
        return mdf, peaks
    
    def segregate_csl_pairs(self):
        """
        Segregate grain boundary pairs by coincidence site lattice (CSL) type.

        Uses the MDF peaks previously selected via the interactive widget
        (``nbWidgets.mdf_peak_selector``) together with the CSL information
        returned by ``compute_mdf_ebsd()`` to classify every unique
        grain-boundary pair into its nearest CSL relationship.

        Prerequisites
        -------------
        ``compute_mdf_ebsd()`` must have been called first so that ``mdf``
        and ``peaks`` are available in the calling scope, and the user must
        have selected the desired peaks via ``nbWidgets.mdf_peak_selector``
        (which populates ``selected_peaks``).

        Parameters
        ----------
        None

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
        Prints a table with columns ``CSL type``, ``ref °``, ``pairs``, and
        ``grains`` for every CSL relationship found among the selected peaks.

        Notes
        -----
        ``selected_peaks`` and ``peaks`` must be defined in the scope where
        this method is invoked (typically a Jupyter notebook cell).
        """
        from upxo.xtalphy.crystal_orientation import segregate_csl_pairs
        # Segregate using selected_peaks from the widget cell
        csl_grains = segregate_csl_pairs(mdf, selected_peaks, csl=peaks['csl'], csl_tol=peaks['csl_tol'])
        # Summary table
        print(f"{'CSL type':<18} {'ref °':>6}  {'pairs':>6}  {'grains':>7}")
        print('─' * 42)
        for lbl, info in csl_grains.items():
            print(f"{lbl:<18} {info['csl_angle']:>6.2f}  "
                f"{len(info['pairs']):>6}  {len(info['grains_all']):>7}")
            
        return csl_grains

# _char_lfi has moved to upxo.interfaces.defdap.ebsd_reader as a
# module-level helper.  Import it here for any legacy internal callers.
from upxo.interfaces.defdap.ebsd_reader import _char_lfi  # noqa: F401
