"""Tolerances of the mtex validation suite, in one place.

DRAFT for approval. Each value states what it protects. A test that needs a looser value than
the one listed here is a finding to report, not a reason to widen the table silently.
"""

# Quantities that are closed-form in float64 (matrix algebra, trigonometry): any difference is
# rounding. 1e-6 degree is well above the rounding of a 24 x 24 reduction and far below any
# physically meaningful angle.
EXACT_DEG = 1e-6

# Matrix and vector components of closed-form quantities.
EXACT_ELEMENT = 1e-9

# Orientations compared as matrices (same orientation of a cubic crystal): element difference
# after the best cubic relabelling. Use this, not an angle, to decide that two orientations are
# equal: arccos amplifies rounding near zero (a 1e-16 error in the trace appears as about 1e-6
# degree), so an angle of exactly zero is never reported below that level.
SAME_ORIENTATION_ELEMENT = 1e-9

# UPXO against a value frozen from mtex 1.0.0. The frozen JSON is stored at full float64
# precision, so this is the same order as EXACT_DEG.
FROZEN_MTEX_DEG = 1e-6

# UPXO against mtex computed live on the same input (different code paths, both float64).
LIVE_MTEX_DEG = 1e-4

# Comparison with a tabulated angle printed to two decimals in the literature or on the wiki.
TABULATED_DEG = 5e-3

# Statistical checks (Monte Carlo): relative error allowed on a mean.
MONTE_CARLO_REL = 0.03

# Angle between two directions considered "parallel" (1 - |cos|).
PARALLEL = 1e-9

# Minimum separation between candidate trace angles for a trace to identify its plane (degrees),
# and the measurement noise applied to the observed trace in that test.
TRACE_SEPARATION_DEG = 3.0
TRACE_NOISE_DEG = 0.5

# The largest disorientation between two cubic orientations (Mackenzie), degrees.
CUBIC_MAX_DISORIENTATION_DEG = 62.7994
