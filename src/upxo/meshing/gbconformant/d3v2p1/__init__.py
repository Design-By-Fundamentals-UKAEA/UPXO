"""Faster stages of the d3v2p0 voxel-to-conformal-tet pipeline.

Only rewritten stages live here; every function keeps the name and
arguments of its d3v2p0 counterpart and adds ``backend`` ('auto') and
``n_workers`` (None: automatic). Without worker processes or numba every
stage runs serially in pure numpy (see backend.py). Unchanged stages are
imported from d3v2p0 directly.
"""
