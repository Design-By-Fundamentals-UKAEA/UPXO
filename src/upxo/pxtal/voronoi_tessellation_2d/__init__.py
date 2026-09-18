"""
2D Voronoi tessellation package for UPXO.

Exposes high-fidelity 2D Voronoi geometry generation routines:
- Periodic Boundary Conditions (PBC)
- Laguerre tessellation / Power diagrams (weighted Voronoi)
- Centroidal Voronoi Tessellation (CVT) via Lloyd relaxation
- Interface perturbation and boundary curvature modeling
"""

from upxo.pxtal.voronoi_tessellation_2d.engine import (
    coerce_bounds_2d,
    compute_standard_voronoi_2d,
    compute_power_diagram_2d,
    cvt_relax_2d,
    perturb_interfaces_2d,
    generate_voronoi_2d,
)

__all__ = [
    'coerce_bounds_2d',
    'compute_standard_voronoi_2d',
    'compute_power_diagram_2d',
    'cvt_relax_2d',
    'perturb_interfaces_2d',
    'generate_voronoi_2d',
]
