"""mcgsV1_1 -- a lean, GUI-driven Monte-Carlo grain-growth entry point for
the 3D twinned-FCC pipeline (3D, algorithms 300a/300b only).

Built alongside -- deliberately NOT replacing -- upxo.ggrowth.mcgs.mcgs.
That class is depended on by many other modules/notebooks and is not being
touched here; this is a separate, additive module for the twinned_simple_3d
GUI's Monte-Carlo stage.

Why a new module instead of wiring the GUI straight to mcgs.mcgs
------------------------------------------------------------------
mcgs.mcgs reads its entire configuration from an Excel "input_dashboard"
file through a multi-layer ingestion path (xlrd -> one flat dict -> 7 typed
wrapper objects, one file each under upxo.interfaces.user_inputs). Of the
dashboard's ~141 rows, only a double-digit subset ever reaches the 3D
algorithm functions (alg300a/alg300b); roughly 65 are read into the flat
dict and simply discarded even inside mcgs.py itself. Wiring the GUI to
that path would mean writing a throwaway Excel file on every run just to
satisfy an xlrd read.

mcgsV1_1 builds the same flat dict directly from Python values (no Excel
round-trip) and constructs the SAME wrapper classes mcgs.py uses
(upxo.interfaces.user_inputs.uidata_mcgs_*), so the uigrid/uisim/uiint/
uimesh/uigsc/uigsprop/uigeorep objects built here remain attribute-compatible
with mcgs3_temporal_slice.mcgs3_grain_structure and anything else in the
codebase that expects a real instance of one of those classes -- only the
ingestion path changes, not the object shapes downstream code relies on.

It dispatches straight to the existing, unmodified
upxo.algorithms.alg300a.mc_iterations_3d_alg300a /
upxo.algorithms.alg300b.mc_iterations_3d_alg300b functions -- the actual
simulation core, already decoupled from mcgs.py's Excel/wrapper machinery --
so this module carries no duplicated physics, only argument construction.

Deliberately NOT built here (confirmed unused by the 3D algorithms while
tracing this):
  * Algorithm hopping -- dead in mcgs.py itself: mcgs.simulate() always
    forces algo_hop=False right before checking it, and the "with hops" 3D
    functions are literally `raise NotImplementedError`.
  * The non-locality matrix / NL / kineticity -- mcgs.py's initiate() builds
    this unconditionally for dim==3, but neither alg300a nor alg300b ever
    reads self.NLM/self.NLM_nd; they only take the appended-index arrays
    (xinda/yinda/zinda) below.
  * The ~230-line hand-written face/edge/vertex block mcgs.py uses to build
    those appended-index arrays (mcgs.py ~600-839) -- it gets unconditionally
    overwritten right after by a single `np.pad(..., mode='wrap')` call, so
    mcgsV1_1 just does the np.pad directly.

State-dependent Boltzmann acceptance ("q_related" mode)
---------------------------------------------------------
mcgs.py's own q_related formula derives a single automatic linear ramp
across state indices from one scalar (boltzmann_temp_factor_max) -- and its
q_unrelated branch has a bug (mcgs.py:976, `self.simpar.S` does not exist;
should be `self.uisim.S`) that crashes the first time it's exercised.

mcgsV1_1 implements this fresh, correctly, and more flexibly: q_unrelated
still takes one scalar temperature factor (uniform across all states);
q_related takes Q independent per-state temperature factors, chosen by the
user rather than auto-derived -- letting some states grow faster/slower
than others by direct design instead of a fixed ramp, e.g. to introduce
natural multi-modality in the resulting grain-size distribution.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

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

VALID_ALGORITHMS = ('300a', '300b')
VALID_BOLTZMANN_MODES = ('q_unrelated', 'q_related')


@dataclass
class MCGSConfig:
    """Everything mcgsV1_1 needs, in plain Python values -- built directly
    from GUI shared_state, no Excel dashboard involved.

    boltzmann_temp_factor is used when boltzmann_mode == 'q_unrelated' (one
    value shared by every state). boltzmann_temp_factors is used when
    boltzmann_mode == 'q_related' (one value per state, length must equal Q)
    -- see the module docstring for why this replaced mcgs.py's automatic
    single-factor ramp.
    """
    xmin: float
    xmax: float
    xinc: float
    ymin: float
    ymax: float
    yinc: float
    zmin: float
    zmax: float
    zinc: float
    Q: int
    mcalg: str
    mcsteps: int
    save_interval: int
    consider_boltzmann: bool
    boltzmann_mode: str = 'q_unrelated'
    boltzmann_temp_factor: Optional[float] = None
    boltzmann_temp_factors: Optional[Sequence[float]] = None
    print_interval: int = 10
    rng_seed: Optional[int] = None

    def validate(self):
        if self.mcalg not in VALID_ALGORITHMS:
            raise ValueError(
                f"mcalg must be one of {VALID_ALGORITHMS}, got {self.mcalg!r}.")
        if self.xmax <= self.xmin or self.ymax <= self.ymin or self.zmax <= self.zmin:
            raise ValueError("Each axis max must be greater than its min.")
        if self.xinc <= 0 or self.yinc <= 0 or self.zinc <= 0:
            raise ValueError("Axis increments must be greater than zero.")
        if self.Q < 1:
            raise ValueError("Number of Monte-Carlo states (Q) must be >= 1.")
        if self.mcsteps < 1:
            raise ValueError("mcsteps must be >= 1.")
        if self.save_interval < 1:
            raise ValueError("save_interval must be >= 1.")
        if self.print_interval < 1:
            raise ValueError("print_interval must be >= 1.")
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
            else:  # q_related
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
    """Per-state Boltzmann acceptance-probability array (length Q), the
    array form uisim.s_boltz_prob must hold by the time it reaches
    alg300a/alg300b (see mcgs.py's setup_transition_probability_rules for
    the original, single-scalar-derived version this replaces).

    q_unrelated: P_q = exp(-kbf * a_q), a_q ~ Uniform(0, 1) -- one shared kbf.
    q_related:   P_q = exp(-kbf_q * a_q), a_q ~ Uniform(0, 1) -- one kbf per
                 state, so growth mobility can be tuned per state directly.

    Computed unconditionally, even when config.consider_boltzmann is False
    -- see the comment at the call site in mcgsV1_1.__init__ for why (the
    values are simply never read by mcloop_* in that case, but they still
    need to exist as a proper float64 array for numba to compile against).
    When consider_boltzmann is False the factor(s) may not have been
    supplied at all; fall back to 1.0 per state since the values are inert.
    """
    Q = config.Q
    a = rng.random(size=Q)
    if config.boltzmann_mode == 'q_unrelated':
        kbf = config.boltzmann_temp_factor if config.boltzmann_temp_factor is not None else 1.0
        return np.exp(-kbf * a)
    else:  # q_related
        factors = config.boltzmann_temp_factors if config.boltzmann_temp_factors is not None else [1.0] * Q
        kbf = np.asarray(factors, dtype=float)
        return np.exp(-kbf * a)


class mcgsV1_1:
    """Lean 3D Monte-Carlo grain-growth driver. Usage mirrors mcgs.mcgs's
    notebook usage pattern (pxt.simulate(); pxt.m[-1]; pxt.gs[tslice]) so it
    can be swapped in for GUI wiring without reshaping the rest of the
    twinned_simple_3d pipeline.
    """

    def __init__(self, config: MCGSConfig, verbose=True):
        config.validate()
        self.config = config
        self.verbose = verbose

        rng = np.random.default_rng(config.rng_seed)

        uidata = self._build_flat_uidata(config)
        self.uidata_all = uidata

        self.uigrid = _uidata_mcgs_gridding_definitions_(uidata)
        self.uisim = _uidata_mcgs_simpar_(uidata)
        self.uigsc = _uidata_mcgs_grain_structure_characterisation_(uidata)
        self.uiint = _uidata_mcgs_intervals_(uidata)
        self.uigsprop = _uidata_mcgs_property_calc_(uidata)
        self.uigeorep = _uidata_mcgs_generate_geom_reprs_(uidata)
        self.uimesh = _uidata_mcgs_mesh_(uidata)

        # Always a float64 array, never the raw 'q_unrelated'/'q_related'
        # string placeholder -- alg300a/alg300b's mcloop_* are @njit, and
        # numba compiles both branches of `elif cbp: ... sbp[...] < ...`
        # for the argument *types* it's called with, regardless of cbp's
        # runtime value. Leaving sbp as a string here made the no-Boltzmann
        # case fail to JIT-compile at all (comparing a unicode char to a
        # float). mcgs.py avoids this by always computing the array in
        # initiate(), unconditionally -- same fix here.
        self.uisim.s_boltz_prob = _build_state_boltzmann_probabilities(config, rng)

        self.vox_size = (config.xinc, config.yinc, config.zinc)

        nx = int(round((config.xmax - config.xmin) / config.xinc)) + 1
        ny = int(round((config.ymax - config.ymin) / config.yinc)) + 1
        nz = int(round((config.zmax - config.zmin) / config.zinc)) + 1
        shape = (nz, ny, nx)  # axis0=z/plane, axis1=y/row, axis2=x/column,
                                # matching alg300a/alg300b's P/R/C convention.

        self.S = rng.integers(1, config.Q + 1, size=shape)
        zind, yind, xind = np.indices(shape, dtype=int)
        self.xinda = np.pad(xind, 1, mode='wrap')
        self.yinda = np.pad(yind, 1, mode='wrap')
        self.zinda = np.pad(zind, 1, mode='wrap')

        self.gs = {}
        self.m = []
        self.fully_annealed = None

    def _build_flat_uidata(self, config: MCGSConfig) -> dict:
        """The flat {name: value} dict the upxo.interfaces.user_inputs
        wrapper classes expect -- built directly from config instead of an
        Excel read. Only includes what each wrapper class's __init__
        actually reads (see the module docstring for the ~65 dashboard
        fields this deliberately drops). Fields hardcoded below (type,
        transformation, boundary_condition_type, NL, kineticity) match the
        dashboard's own documented restrictions ("square only", "none
        only", NL "choose 1, 2 is buggy") and aren't read by alg300a/
        alg300b anyway -- see the module docstring.
        """
        return {
            # -- uigrid --
            'type': 'square',
            'dim': 3,
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
            # -- uigsprop -- (property-calc toggles belong to a later
            # characterization stage, not simulation; left inert here)
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
            # -- uigeorep -- (2D legacy polygon/ring/XTAL toggles, vestigial
            # for the 3D voxel pipeline; left inert here)
            'make_mp_grain_centoids': False,
            'make_mp_grain_points': False,
            'make_ring_grain_boundaries': False,
            'make_xtal_grain': False,
            'make_chull_grain': False,
            'create_gbz': False,
            # -- uimesh -- (belongs to the Export/meshing stage, not MC config)
            'mesh_gb_conformity': None,
            'mesh_target_fe_software': None,
            'mesh_meshing_package': None,
            'mesh_reduced_integration': False,
            'mesh_element_type': None,
        }

    def simulate(self, verbose=None):
        """Run the 3D Monte-Carlo grain-growth simulation and populate
        self.gs (temporal slice index -> mcgs3_grain_structure) and self.m
        (sorted list of the temporal slice indices that were saved)."""
        if verbose is None:
            verbose = self.verbose

        from scipy.ndimage import label as ndimg_label_pck

        if self.config.mcalg == '300a':
            from upxo.algorithms.alg300a import mc_iterations_3d_alg300a as _run
        else:  # '300b'
            from upxo.algorithms.alg300b import mc_iterations_3d_alg300b as _run

        self.gs, self.fully_annealed = _run(
            S=self.S, vox_size=self.vox_size,
            xinda=self.xinda, yinda=self.yinda, zinda=self.zinda,
            uidata=self.uidata_all, uigrid=self.uigrid, uisim=self.uisim,
            uiint=self.uiint, uimesh=self.uimesh, verbose=verbose,
            ndimg_label_pck=ndimg_label_pck,
        )
        self.m = sorted(self.gs.keys())
        return self.gs, self.fully_annealed
