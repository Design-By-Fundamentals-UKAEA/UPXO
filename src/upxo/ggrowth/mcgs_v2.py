"""mcgs_v2 -- Monte-Carlo grain-growth driver configured from plain Python
values (no Excel input dashboard), for 2D and 3D.

Independent of ``upxo.ggrowth.mcgs.mcgs`` (the Excel-dashboard driver).
Supersedes ``upxo.ggrowth.mcgsV1_1``, which now re-exports this module's
``mcgs_v2`` and ``MCGSConfig`` under the old names.

Usage
-----
3D (default, ``dim=3``)::

    cfg = MCGSConfig(xmin=0, xmax=49, xinc=1, ymin=0, ymax=49, yinc=1,
                     zmin=0, zmax=49, zinc=1, Q=32, mcalg='300a',
                     mcsteps=60, save_interval=10, consider_boltzmann=False)
    pxt = mcgs_v2(cfg); pxt.simulate()

2D (``dim=2``; z-axis fields are optional and ignored)::

    cfg = MCGSConfig(dim=2, xmin=0, xmax=199, xinc=1, ymin=0, ymax=199,
                     yinc=1, Q=32, mcalg='200', mcsteps=100,
                     save_interval=10, consider_boltzmann=False)
    pxt = mcgs_v2(cfg); pxt.simulate()
    pxt.detect_grains()

After ``simulate()``: ``pxt.m`` is the sorted list of saved temporal slices,
``pxt.gs[t].s`` the state array of slice ``t`` -- shape ``(ny, nx)`` in 2D,
``(nz, ny, nx)`` in 3D -- and ``pxt.fully_annealed`` the dict returned by the
algorithm.

Algorithms
----------
2D: '200' (unweighted Potts), '201' (second-order-neighbour weighted, uses
``rsfso``), '202' (dominant-neighbour variant).
3D: '300a', '300b'.

Note on '201': alg201.py ranks the neighbour states by count but then
indexes the value-sorted state list with the random draw instead of the
ranking, so the candidate is always among the lowest-numbered neighbouring
states. The structure coarsens to a single crystal within a few tens of
steps. It is run here exactly as implemented there.

State-dependent Boltzmann acceptance
------------------------------------
``q_unrelated``: one temperature factor shared by every state.
``q_related``: Q independent per-state temperature factors.
In both cases P_q = exp(-kbf_q * a_q), a_q ~ Uniform(0, 1).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from upxo.interfaces.user_inputs.uidata_mcgs_gridding_definitions import (
    _uidata_mcgs_gridding_definitions_)
from upxo.interfaces.user_inputs.uidata_mcgs_simpar import _uidata_mcgs_simpar_
from upxo.interfaces.user_inputs.uidata_mcgs_grain_structure_characterisation import (
    _uidata_mcgs_grain_structure_characterisation_)
from upxo.interfaces.user_inputs.uidata_mcgs_intervals import _uidata_mcgs_intervals_
from upxo.interfaces.user_inputs.uidata_mcgs_property_calc import (
    _uidata_mcgs_property_calc_)
from upxo.interfaces.user_inputs.uidata_mcgs_generate_geom_reprs import (
    _uidata_mcgs_generate_geom_reprs_)
from upxo.interfaces.user_inputs.uidata_mcgs_mesh import _uidata_mcgs_mesh_

VALID_DIMS = (2, 3)
VALID_ALGORITHMS = {2: ('200', '201', '202'), 3: ('300a', '300b')}
VALID_BOLTZMANN_MODES = ('q_unrelated', 'q_related')


@dataclass
class MCGSConfig:
    """Everything mcgs_v2 needs, in plain Python values.

    ``dim`` defaults to 3. For ``dim=2`` the z-axis fields may be omitted
    and are ignored.

    boltzmann_temp_factor is used when boltzmann_mode == 'q_unrelated'.
    boltzmann_temp_factors (length Q) is used when 'q_related'.
    rsfso is only used by 2D algorithm '201'.
    """
    xmin: float
    xmax: float
    xinc: float
    ymin: float
    ymax: float
    yinc: float
    Q: int
    mcalg: str
    mcsteps: int
    save_interval: int
    consider_boltzmann: bool
    zmin: float = 0.0
    zmax: float = 0.0
    zinc: float = 1.0
    dim: int = 3
    boltzmann_mode: str = 'q_unrelated'
    boltzmann_temp_factor: Optional[float] = None
    boltzmann_temp_factors: Optional[Sequence[float]] = None
    rsfso: int = 2
    print_interval: int = 10
    rng_seed: Optional[int] = None

    def validate(self):
        if self.dim not in VALID_DIMS:
            raise ValueError(f"dim must be one of {VALID_DIMS}, got {self.dim!r}.")
        if self.mcalg not in VALID_ALGORITHMS[self.dim]:
            raise ValueError(
                f"For dim={self.dim}, mcalg must be one of "
                f"{VALID_ALGORITHMS[self.dim]}, got {self.mcalg!r}.")
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("Each axis max must be greater than its min.")
        if self.xinc <= 0 or self.yinc <= 0:
            raise ValueError("Axis increments must be greater than zero.")
        if self.dim == 3:
            if self.zmax <= self.zmin:
                raise ValueError("Each axis max must be greater than its min.")
            if self.zinc <= 0:
                raise ValueError("Axis increments must be greater than zero.")
        if self.Q < 1:
            raise ValueError("Number of Monte-Carlo states (Q) must be >= 1.")
        if self.mcsteps < 1:
            raise ValueError("mcsteps must be >= 1.")
        if self.save_interval < 1:
            raise ValueError("save_interval must be >= 1.")
        if self.print_interval < 1:
            raise ValueError("print_interval must be >= 1.")
        if self.rsfso < 1:
            raise ValueError("rsfso must be >= 1.")
        if self.consider_boltzmann:
            if self.boltzmann_mode not in VALID_BOLTZMANN_MODES:
                raise ValueError(
                    f"boltzmann_mode must be one of {VALID_BOLTZMANN_MODES}, "
                    f"got {self.boltzmann_mode!r}.")
            if self.boltzmann_mode == 'q_unrelated':
                if self.boltzmann_temp_factor is None:
                    raise ValueError(
                        "boltzmann_temp_factor is required when "
                        "boltzmann_mode == 'q_unrelated'.")
            else:
                if self.boltzmann_temp_factors is None:
                    raise ValueError(
                        "boltzmann_temp_factors is required when "
                        "boltzmann_mode == 'q_related'.")
                if len(self.boltzmann_temp_factors) != self.Q:
                    raise ValueError(
                        f"boltzmann_temp_factors must have exactly Q={self.Q} "
                        f"entries (one per state), got "
                        f"{len(self.boltzmann_temp_factors)}.")


def _build_state_boltzmann_probabilities(config: MCGSConfig, rng: np.random.Generator):
    """Per-state Boltzmann acceptance-probability array (length Q).

    Always a float64 array, even when config.consider_boltzmann is False
    (the values are then never read, but numba compiles against the
    argument type, so a string placeholder must not be passed).
    """
    Q = config.Q
    a = rng.random(size=Q)
    if config.boltzmann_mode == 'q_unrelated':
        kbf = config.boltzmann_temp_factor if config.boltzmann_temp_factor is not None else 1.0
        return np.exp(-kbf * a)
    factors = config.boltzmann_temp_factors if config.boltzmann_temp_factors is not None else [1.0] * Q
    return np.exp(-np.asarray(factors, dtype=float) * a)


def _appended_index_arrays_2d(ny, nx):
    """AIA0/AIA1 for the wrapped 3x3 neighbourhood of an (ny, nx) array.

    Equivalent to mcgs.AppIndArray for dim=2, NL=1: both are padded by one
    cell on every side with periodic wrapping, and indexed as
    ``S[AIA0[i, j], AIA1[i, j]]`` by the 2D kernels. AIA0 holds the row
    index and AIA1 the column index, both of shape (ny + 2, nx + 2).
    """
    rows, cols = np.indices((ny, nx), dtype=int)
    return np.pad(rows, 1, mode='wrap'), np.pad(cols, 1, mode='wrap')


class mcgs_v2:
    """Monte-Carlo grain-growth driver for 2D (dim=2) and 3D (dim=3)."""

    def __init__(self, config: MCGSConfig, verbose=True):
        config.validate()
        self.config = config
        self.verbose = verbose
        self.dim = config.dim

        rng = np.random.default_rng(config.rng_seed)
        self._rng = rng

        flat = self._build_flat_uidata(config)
        self.uigrid = _uidata_mcgs_gridding_definitions_(flat)
        self.uisim = _uidata_mcgs_simpar_(flat)
        self.uigsc = _uidata_mcgs_grain_structure_characterisation_(flat)
        self.uiint = _uidata_mcgs_intervals_(flat)
        self.uigsprop = _uidata_mcgs_property_calc_(flat)
        self.uigeorep = _uidata_mcgs_generate_geom_reprs_(flat)
        self.uimesh = _uidata_mcgs_mesh_(flat)
        self.uisim.s_boltz_prob = _build_state_boltzmann_probabilities(config, rng)

        # 3D slices read the flat dict; 2D slices read the dict-of-objects
        # form that mcgs.load_uidata produces (mesh export indexes
        # uinputs['uimesh']).
        if self.dim == 3:
            self.uidata_all = flat
        else:
            self.uidata_all = {
                'uigrid': self.uigrid, 'uisim': self.uisim,
                'uigsc': self.uigsc, 'uiint': self.uiint,
                'uigsprop': self.uigsprop, 'uigeorep': self.uigeorep,
                'uimesh': self.uimesh,
            }

        nx = int(round((config.xmax - config.xmin) / config.xinc)) + 1
        ny = int(round((config.ymax - config.ymin) / config.yinc)) + 1
        if self.dim == 3:
            self._init_3d(config, rng, nx, ny)
        else:
            self._init_2d(config, rng, nx, ny)

        self.gs = {}
        self.m = []
        self.tslices = []
        self.fully_annealed = None

    # ── construction ────────────────────────────────────────────────────

    def _init_3d(self, config, rng, nx, ny):
        self.vox_size = (config.xinc, config.yinc, config.zinc)
        nz = int(round((config.zmax - config.zmin) / config.zinc)) + 1
        shape = (nz, ny, nx)  # axis0=z, axis1=y, axis2=x
        self.S = rng.integers(1, config.Q + 1, size=shape)
        zind, yind, xind = np.indices(shape, dtype=int)
        self.xinda = np.pad(xind, 1, mode='wrap')
        self.yinda = np.pad(yind, 1, mode='wrap')
        self.zinda = np.pad(zind, 1, mode='wrap')

    def _init_2d(self, config, rng, nx, ny):
        self.px_size = config.xinc * config.yinc
        self.px_length = (config.xinc + config.yinc) / 2
        xarr = config.xmin + config.xinc * np.arange(nx)
        yarr = config.ymin + config.yinc * np.arange(ny)
        self.xgr, self.ygr = np.meshgrid(xarr, yarr, indexing='xy')
        self.zgr = 0
        self.S = rng.integers(1, config.Q + 1, size=(ny, nx))
        self.AIA0, self.AIA1 = _appended_index_arrays_2d(ny, nx)
        # Unweighted nearest-neighbour interaction (NL=1, no distance or
        # boolean weighting): all-ones 3x3 matrix, as the dashboard default.
        self.NLM = np.ones((3, 3), dtype=float)
        # Euler angles per state, as mcgs.build_ea
        ea = rng.uniform([0, 0, 0], [360, 180, 360], (config.Q, 3)).T
        self.EAPGLB = (ea[0], ea[1], ea[2])

    def _build_flat_uidata(self, config: MCGSConfig) -> dict:
        """Flat {name: value} dict the upxo.interfaces.user_inputs wrapper
        classes expect, built directly from config."""
        return {
            # -- uigrid --
            'type': 'square',
            'dim': config.dim,
            'xmin': config.xmin, 'xmax': config.xmax, 'xinc': config.xinc,
            'ymin': config.ymin, 'ymax': config.ymax, 'yinc': config.yinc,
            'zmin': config.zmin, 'zmax': config.zmax, 'zinc': config.zinc,
            'transformation': 'none',
            # -- uisim --
            'mcsteps': config.mcsteps,
            'mcalg': config.mcalg,
            'S': config.Q,
            'state_sampling_scheme': 'rejection',
            'consider_boltzmann_probability': config.consider_boltzmann,
            's_boltz_prob': config.boltzmann_mode,
            'boltzmann_temp_factor_max': (
                config.boltzmann_temp_factor if config.boltzmann_mode == 'q_unrelated'
                else (max(config.boltzmann_temp_factors) if config.boltzmann_temp_factors else 0.0)),
            'boundary_condition_type': 'wrapped',
            'NL': 1,
            'kineticity': 'static',
            # -- uiint --
            'mcint_grain_size_par_estim': False,
            'mcint_gb_par_estimation': False,
            'mcint_grain_shape_par_estim': False,
            'mcint_save_at_mcstep_interval': config.save_interval,
            'save_final_S_only': False,
            'mcint_promt_display': config.print_interval,
            'mcint_plot_grain_structure': False,
            # -- uigsc --
            'grain_identification_library': 'scikit-image',
            # -- uigsprop -- (inert; characterisation is a later stage)
            'compute_grain_area_pix': False,
            'compute_grain_area_pol': False,
            'compute_gb_length_pol': False,
            'compute_gb_length_pxl': False,
            'compute_grain_moments': False,
            'grain_area_type_to_consider': False,
            'compute_grain_area_distr': False,
            'compute_grain_area_distr_kde': False,
            'compute_grain_area_distr_prop': False,
            'gb_length_type_to_consider': False,
            'compute_gb_length_distr': False,
            'compute_gb_length_distr_kde': False,
            'compute_gb_length_distr_prop': False,
            # -- uigeorep -- (inert)
            'make_mp_grain_centoids': False,
            'make_mp_grain_points': False,
            'make_ring_grain_boundaries': False,
            'make_xtal_grain': False,
            'make_chull_grain': False,
            'create_gbz': False,
            # -- uimesh -- (belongs to a later export stage)
            'mesh_gb_conformity': None,
            'mesh_target_fe_software': None,
            'mesh_meshing_package': None,
            'mesh_reduced_integration': False,
            'mesh_element_type': None,
        }

    # ── simulation ──────────────────────────────────────────────────────

    def simulate(self, verbose=None):
        """Run the simulation and populate self.gs (temporal slice index ->
        grain-structure object) and self.m (sorted saved slice indices)."""
        if verbose is None:
            verbose = self.verbose
        if self.dim == 3:
            self._simulate_3d(verbose)
        else:
            self._simulate_2d(verbose)
        self.m = sorted(self.gs.keys())
        self.tslices = list(self.m)
        return self.gs, self.fully_annealed

    def _simulate_3d(self, verbose):
        from scipy.ndimage import label as ndimg_label_pck
        if self.config.mcalg == '300a':
            from upxo.algorithms.alg300a import mc_iterations_3d_alg300a as _run
        else:
            from upxo.algorithms.alg300b import mc_iterations_3d_alg300b as _run
        self.gs, self.fully_annealed = _run(
            S=self.S, vox_size=self.vox_size,
            xinda=self.xinda, yinda=self.yinda, zinda=self.zinda,
            uidata=self.uidata_all, uigrid=self.uigrid, uisim=self.uisim,
            uiint=self.uiint, uimesh=self.uimesh, verbose=verbose,
            ndimg_label_pck=ndimg_label_pck,
        )

    def _simulate_2d(self, verbose):
        _a, _b, _c = self.NLM[0], self.NLM[1], self.NLM[2]
        common = dict(xgr=self.xgr, ygr=self.ygr, zgr=self.zgr,
                      px_size=self.px_size, S=self.S, _a=_a, _b=_b, _c=_c,
                      AIA0=self.AIA0, AIA1=self.AIA1,
                      uisim=self.uisim, uiint=self.uiint,
                      uidata=self.uidata_all, uigrid=self.uigrid,
                      display_messages=verbose)
        liwm = dict(user_LIWM=False, LIWM=self.NLM)
        alg = self.config.mcalg
        if alg == '200':
            import upxo.algorithms.alg200 as mod
            self.gs, self.fully_annealed = mod.run(**common, **liwm)
        elif alg == '201':
            import upxo.algorithms.alg201 as mod
            self.gs, self.fully_annealed = mod.run(
                rsfso=self.config.rsfso, **common)
        else:  # '202'
            import upxo.algorithms.alg202 as mod
            self.gs, self.fully_annealed = mod.run(**common, **liwm)

    # ── grain detection (2D) ────────────────────────────────────────────

    def detect_grains(self, mcsteps=None, kernel_order=2,
                      library='scikit-image', connectivity=8,
                      store_state_ng=True, process_individual_states=False,
                      delta=0, lfiDtype=np.int32, verbose=False):
        """Label grains on 2D temporal slices (fills .lgi, .n, .gid, ...).

        mcsteps: int or iterable of saved slice indices; default all.
        3D labelling is not done here.
        """
        if self.dim != 2:
            raise NotImplementedError(
                "detect_grains is the 2D labelling path; for dim=3 label "
                "slices with the 3D grain-detection tools.")
        if not self.m:
            raise RuntimeError("Call simulate() before detect_grains().")
        if mcsteps is None:
            mcsteps = list(self.m)
        elif isinstance(mcsteps, (int, np.integer)):
            mcsteps = [int(mcsteps)]
        else:
            mcsteps = list(mcsteps)
        missing = [t for t in mcsteps if t not in self.gs]
        if missing:
            raise ValueError(
                f"Temporal slice(s) {missing} not available. "
                f"Saved slices: {self.m}")
        from upxo.pxtalops import detect_grains_from_mcstates as get_grains
        self.gs, state_ng = get_grains.mcgs2d(
            library=library, gs_dict=self.gs, msteps=mcsteps,
            kernel_order=kernel_order, store_state_ng=store_state_ng,
            connectivity=connectivity,
            process_individual_states=process_individual_states,
            delta=delta, lfiDtype=lfiDtype, verbose=verbose)
        return state_ng
