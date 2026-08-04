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
        'p_percentage', 'wasserstein_threshold',
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
            wasserstein_threshold: float = 0.5,
    ):
        self.n_slices_x = n_slices_x
        self.n_slices_y = n_slices_y
        self.n_slices_z = n_slices_z
        self.test_along_x = test_along_x
        self.test_along_y = test_along_y
        self.test_along_z = test_along_z
        self.p_percentage = p_percentage
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
        for axis_name, results in self.slice_results.items():
            n_pass = sum(1 for r in results if r['passes'])
            n_total = len(results)
            pct = 100.0 * n_pass / n_total if n_total > 0 else 0.0
            self.axis_acceptance[axis_name] = {
                'n_pass': n_pass,
                'n_total': n_total,
                'pct_pass': pct,
                'accepted': pct >= self.p_percentage,
            }
        self.overall_accepted = all(
            info['accepted'] for info in self.axis_acceptance.values()
        )

    def report(self) -> str:
        """Return a formatted text summary of the validation results."""
        lines = ['3D Representativeness Validation', '=' * 50]
        for axis_name, info in self.axis_acceptance.items():
            status = 'ACCEPTED' if info['accepted'] else 'REJECTED'
            lines.append(
                f'{axis_name}-axis: {info["n_pass"]}/{info["n_total"]} '
                f'({info["pct_pass"]:.1f}%) [{status}]')
        overall = 'YES' if self.overall_accepted else 'NO'
        lines.append(f'\nOverall 3D acceptance: {overall}')
        lines.append(f'Threshold: {self.p_percentage}% per axis')
        return '\n'.join(lines)
