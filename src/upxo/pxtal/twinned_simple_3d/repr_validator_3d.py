"""
repr_validator_3d.py
====================
Multi-axis 2D slice representativeness validation for the twinned
simple 3D pipeline.
"""

import numpy as np
from typing import Optional, Dict


class RepresentativenessValidator3D:
    """
    Multi-axis 2D slice representativeness validator.

    Two modes:
    - **morphological** (pre-twin): Wasserstein distance on normalised
      grain-size distributions.
    - **crystallographic** (post-twin): Wasserstein distance on
      misorientation angle distributions.

    A 3D structure is accepted when at least ``p_percentage`` percent of
    slices pass on every tested axis.
    """

    __slots__ = (
        'n_slices_x', 'n_slices_y', 'n_slices_z',
        'test_along_x', 'test_along_y', 'test_along_z',
        'p_percentage', 'p_percentage_x', 'p_percentage_y', 'p_percentage_z',
        'wasserstein_threshold',
        'slice_results', 'axis_acceptance', 'overall_accepted',
    )

    def __init__(
            self,
            n_slices_x: int = 10,
            n_slices_y: int = 10,
            n_slices_z: int = 10,
            test_along_x: bool = True,
            test_along_y: bool = True,
            test_along_z: bool = True,
            p_percentage: float = 60.0,
            p_percentage_x: Optional[float] = None,
            p_percentage_y: Optional[float] = None,
            p_percentage_z: Optional[float] = None,
            wasserstein_threshold: float = 0.5,
    ):
        self.n_slices_x = n_slices_x
        self.n_slices_y = n_slices_y
        self.n_slices_z = n_slices_z
        self.test_along_x = test_along_x
        self.test_along_y = test_along_y
        self.test_along_z = test_along_z
        self.p_percentage = p_percentage
        # p_percentage_x/y/z: per-axis pass thresholds. Any left unset (the
        # common case for the post-twin crystallographic caller, which only
        # ever passes p_percentage) falls back to the single p_percentage
        # value, applied uniformly to that axis exactly as before.
        self.p_percentage_x = p_percentage_x if p_percentage_x is not None else p_percentage
        self.p_percentage_y = p_percentage_y if p_percentage_y is not None else p_percentage
        self.p_percentage_z = p_percentage_z if p_percentage_z is not None else p_percentage
        self.wasserstein_threshold = wasserstein_threshold
        self.slice_results: Dict = {}
        self.axis_acceptance: Dict = {}
        self.overall_accepted: Optional[bool] = None

    def validate_morphological(
            self,
            lgi_3d: np.ndarray,
            ebsd_areas: np.ndarray,
    ):
        """
        Pre-twin morphological validation against EBSD grain-size distribution.

        Calls ``section_from_3d`` and
        ``get_grain_size_distribution_from_slice`` per slice.
        """
        from scipy.stats import wasserstein_distance
        from upxo.gsdataops.grid_ops import section_from_3d
        from upxo.charops.mchar import get_grain_size_distribution_from_slice

        ebsd_norm = ebsd_areas / np.mean(ebsd_areas) if np.mean(ebsd_areas) > 0 else ebsd_areas

        # lgi_3d carries the raw MC array's native (nz, ny, nx) axis order
        # (axis0=Z, axis2=X -- see TwinnedSimple3DBase.plot_temporal_slice_3d's
        # P/R/C convention comment); only the axis-index-to-label mapping
        # is corrected here so 'X'/'Z' resolve to the physically correct
        # index -- the array itself is left in its native order.
        axes_config = [
            (2, self.n_slices_x, 'X', self.test_along_x),
            (1, self.n_slices_y, 'Y', self.test_along_y),
            (0, self.n_slices_z, 'Z', self.test_along_z),
        ]

        for axis, n_slices, axis_name, enabled in axes_config:
            if not enabled:
                continue
            results = []
            domain_size = lgi_3d.shape[axis]
            for slice_idx in np.linspace(0, domain_size - 1, n_slices, dtype=int):
                lgi_2d = section_from_3d(lgi_3d, axis=axis, location=int(slice_idx))
                sgc_areas = get_grain_size_distribution_from_slice(lgi_2d)
                if len(sgc_areas) > 0:
                    sgc_norm = sgc_areas / np.mean(sgc_areas)
                    w = float(wasserstein_distance(ebsd_norm, sgc_norm))
                else:
                    w = np.inf
                results.append({
                    'slice_idx': int(slice_idx),
                    'wasserstein': w,
                    'n_grains': int(len(sgc_areas)),
                    'passes': w < self.wasserstein_threshold,
                })
            self.slice_results[axis_name] = results

        self.aggregate()

    def validate_crystallographic(
            self,
            lgi_3d: np.ndarray,
            quat_3d: np.ndarray,
            ebsd_miso_deg: np.ndarray,
    ):
        """
        Post-twin crystallographic validation against EBSD MDF.

        Calls ``extract_2d_slice_pair``, ``find_neighs2d``, and
        ``compute_mdf_from_quats`` per slice.
        """
        from scipy.stats import wasserstein_distance
        from upxo.gsdataops.grid_ops import extract_2d_slice_pair
        from upxo.gsdataops.gid_ops import find_neighs2d
        from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats

        # lgi_3d carries the raw MC array's native (nz, ny, nx) axis order
        # (axis0=Z, axis2=X -- see TwinnedSimple3DBase.plot_temporal_slice_3d's
        # P/R/C convention comment); only the axis-index-to-label mapping
        # is corrected here so 'X'/'Z' resolve to the physically correct
        # index -- the array itself is left in its native order.
        axes_config = [
            (2, self.n_slices_x, 'X', self.test_along_x),
            (1, self.n_slices_y, 'Y', self.test_along_y),
            (0, self.n_slices_z, 'Z', self.test_along_z),
        ]

        for axis, n_slices, axis_name, enabled in axes_config:
            if not enabled:
                continue
            results = []
            domain_size = lgi_3d.shape[axis]
            for slice_idx in np.linspace(0, domain_size - 1, n_slices, dtype=int):
                lgi_2d, quat_2d = extract_2d_slice_pair(
                    lgi_3d, quat_3d, axis, int(slice_idx))
                if lgi_2d.max() <= 0:
                    results.append({'slice_idx': int(slice_idx),
                                    'wasserstein': np.inf, 'passes': False})
                    continue
                try:
                    neigh_raw = find_neighs2d(lgi_2d.astype(np.int32), conn=4)
                    neigh = {int(g): [int(n) for n in ns] for g, ns in neigh_raw.items()}
                    slice_mdf = compute_mdf_from_quats(
                        lgi_2d, quat_2d, neigh, n_bins=65, angle_range=(0.0, 65.0))
                    w = float(wasserstein_distance(
                        ebsd_miso_deg, slice_mdf['miso_deg'])) \
                        if slice_mdf['miso_deg'].size > 0 else np.inf
                except Exception:
                    w = np.inf
                results.append({
                    'slice_idx': int(slice_idx),
                    'wasserstein': w,
                    'passes': w < self.wasserstein_threshold,
                })
            self.slice_results[axis_name] = results

        self.aggregate()

    def aggregate(self):
        """Aggregate slice results into per-axis acceptance."""
        per_axis_threshold = {
            'X': self.p_percentage_x, 'Y': self.p_percentage_y, 'Z': self.p_percentage_z,
        }
        for axis_name, results in self.slice_results.items():
            n_pass = sum(1 for r in results if r['passes'])
            n_total = len(results)
            pct = 100.0 * n_pass / n_total if n_total > 0 else 0.0
            self.axis_acceptance[axis_name] = {
                'n_pass': n_pass,
                'n_total': n_total,
                'pct_pass': pct,
                'accepted': pct >= per_axis_threshold[axis_name],
            }
        self.overall_accepted = all(
            info['accepted'] for info in self.axis_acceptance.values()
        )

    def report(self) -> str:
        """Return a formatted text summary of the validation results."""
        per_axis_threshold = {
            'X': self.p_percentage_x, 'Y': self.p_percentage_y, 'Z': self.p_percentage_z,
        }
        lines = ['3D Representativeness Validation', '=' * 50]
        for axis_name, info in self.axis_acceptance.items():
            status = 'ACCEPTED' if info['accepted'] else 'REJECTED'
            lines.append(
                f'{axis_name}-axis: {info["n_pass"]}/{info["n_total"]} '
                f'({info["pct_pass"]:.1f}%) [{status}, threshold '
                f'{per_axis_threshold[axis_name]:.0f}%]')
        overall = 'YES' if self.overall_accepted else 'NO'
        lines.append(f'\nOverall 3D acceptance: {overall}')
        if self.p_percentage_x == self.p_percentage_y == self.p_percentage_z:
            lines.append(f'Threshold: {self.p_percentage_x}% per axis')
        else:
            lines.append(f'Threshold: X={self.p_percentage_x}%  '
                          f'Y={self.p_percentage_y}%  Z={self.p_percentage_z}%')
        return '\n'.join(lines)


def compute_representative_slice_mdfs(validator, cleaner, quat_3d_clean, axes=None):
    """
    Per-axis misorientation distribution (MDF) for every PASSING slice
    already identified by ``validator.validate_crystallographic``, plus
    their mean -- the representative-slice MDF a passing 2D section
    should reproduce, for comparison against the full EBSD MDF.

    Parameters
    ----------
    validator : RepresentativenessValidator3D
        Already run via ``validate_crystallographic`` -- ``slice_results``
        supplies the passing/failing slice positions per axis.
    cleaner
        Post-cleaning structure (``lgi_clean`` -- duck-typed, matching
        StructureCleaner3D and its subset variants).
    quat_3d_clean : ndarray, shape lgi_clean.shape + (4,)
        Per-voxel quaternion field (see
        ``crystal_orientation.expand_grain_quats_to_voxels``).
    axes : iterable of str, optional
        Which axes ('X'/'Y'/'Z') to compute for. Defaults to every axis
        with at least one slice result recorded.

    Returns
    -------
    dict {axis_name: {'n_passing': int, 'slice_mdfs': list of dict,
                       'mean_bin_centers': ndarray or None,
                       'mean_density': ndarray or None}}
        Each ``slice_mdfs`` entry is one passing slice's
        ``compute_mdf_from_quats`` result (skipped if the slice has no
        grains or its MDF computation fails). ``mean_bin_centers``/
        ``mean_density`` are None when no slice yielded a usable MDF.
    """
    from upxo.gsdataops.grid_ops import extract_2d_slice_pair
    from upxo.gsdataops.gid_ops import find_neighs2d
    from upxo.xtalphy.crystal_orientation import compute_mdf_from_quats

    # lgi_clean/quat_3d_clean carry the pipeline's native (nz, ny, nx)
    # axis order -- same corrected label mapping
    # RepresentativenessValidator3D itself uses internally.
    axis_index = {'X': 2, 'Y': 1, 'Z': 0}
    if axes is None:
        axes = [a for a in validator.slice_results if validator.slice_results[a]]

    result = {}
    for axis_name in axes:
        passing = [r for r in validator.slice_results.get(axis_name, []) if r['passes']]
        slice_mdfs = []
        for r in passing:
            lgi_2d, quat_2d = extract_2d_slice_pair(
                cleaner.lgi_clean, quat_3d_clean, axis_index[axis_name], r['slice_idx'])
            if lgi_2d.max() <= 0:
                continue
            neigh_raw = find_neighs2d(lgi_2d.astype(np.int32), conn=4)
            neigh = {int(g): [int(n) for n in ns] for g, ns in neigh_raw.items()}
            try:
                slice_mdf = compute_mdf_from_quats(
                    lgi_2d, quat_2d, neigh, n_bins=65, angle_range=(0.0, 65.0))
            except Exception:
                continue
            if slice_mdf['miso_deg'].size > 0:
                slice_mdfs.append(slice_mdf)

        mean_bin_centers = mean_density = None
        if slice_mdfs:
            mean_bin_centers = slice_mdfs[0]['hist_bin_centers']
            mean_density = np.mean([m['hist_density'] for m in slice_mdfs], axis=0)

        result[axis_name] = {
            'n_passing': len(passing),
            'slice_mdfs': slice_mdfs,
            'mean_bin_centers': mean_bin_centers,
            'mean_density': mean_density,
        }
    return result
