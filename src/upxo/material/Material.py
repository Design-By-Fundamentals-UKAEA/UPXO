"""
Material.py
============
Aggregates a material's data across many categories -- identity,
processing route, crystal structure, texture, mechanical/physical
properties, irradiation condition, EBSD-derived statistics, etc. -- into
one ``MaterialRegistry`` (see ``registry.py``), replacing the old
``ObjectDataDictionary``-based ``build()``, which flattened every category
to a plain dict on write (discarding type information immediately) and
offered no validation or extensibility beyond editing this function
directly every time a category was added.

See ``admin/material/scoping.md`` and
``admin/material/implementation_plan.md`` for the full redesign rationale.
Categories that used to live directly in this file --
``MaterialIdentity``, ``ProcessingCondition``, ``TexCompVolFracFCC``,
``TexFibreVolFracFCC``, ``TexCompWidth`` -- have moved to their own files
(``identity.py``, ``processing.py``, ``texture.py``) or been superseded
outright (``ProcessingCondition`` -> ``ProcessingRoute``; the three
``Tex*`` classes -> ``TextureComponentProfile``). Everything below is
carried over unchanged in content -- only the container it's registered
into has changed.

There is no ``CrossCheckAndAppend`` equivalent here (the old, broken,
never-exercised update-in-place path -- see
``admin/material/scoping.md`` §2, item 5): calling
``MaterialRegistry.ingest()`` again for an already-registered category
simply replaces its stored instance and provenance, which covers the same
"update on a segment-by-segment basis" use case without a second method.
"""

from dataclasses import dataclass, field
import numpy as np

# Soft-validation whitelist for CrystalFamily.xtal_family -- the values the
# texture work (admin/twinnedFccApp/texIntegration/) actually cares about.
# Deliberately does NOT include this class's own pre-existing default
# ('mmm', a crystallographic point-group symbol, not a crystal-family
# name -- a different classification system entirely): build()'s default
# CrystalFamily() correctly triggers one soft-validation warning as a
# result, same as MaterialIdentity()'s default 'cu' does against
# KNOWN_MATERIALS below -- both are informal legacy placeholders, not real
# entries in their respective curated sets, and flagging that is soft
# validation doing exactly its job, not a bug to special-case away.
KNOWN_CRYSTAL_FAMILIES = {'FCC', 'BCC', 'HCP'}


def build():
    """
    Build and return the material data base using classes in this module.

    When no inputs are provided, defaults prescribed in the class data are used.

    Returns
    -------
    MaterialRegistry
        Typed, extensible, provenance-tracking material data container
        (see ``upxo.material.registry``). Access a category's instance via
        ``matdata.get("CategoryName")``, and its provenance via
        ``matdata.get_provenance("CategoryName")``.

    Examples
    --------
    >>> from upxo.material.Material import build
    >>> matdata = build()
    >>> matdata.categories()
    """
    from upxo.material.registry import MaterialRegistry
    from upxo.material.identity import MaterialIdentity, KNOWN_MATERIALS
    from upxo.material.processing import ProcessingRoute
    from upxo.material.texture import TextureComponentProfile

    matdata = MaterialRegistry()

    matdata.register_category("MaterialIdentity", MaterialIdentity,
                               known_values={"name": KNOWN_MATERIALS})
    matdata.register_category("ProcessingRoute", ProcessingRoute)
    matdata.register_category("IrradiationCondition", IrradiationCondition)
    matdata.register_category("CrystalFamily", CrystalFamily,
                               known_values={"xtal_family": KNOWN_CRYSTAL_FAMILIES})
    matdata.register_category("Phases", Phases)
    matdata.register_category("PhysicalProperty", PhysicalProperty)
    matdata.register_category("ElasticProperty", ElasticProperty)
    matdata.register_category("TensileStressStrain", TensileStressStrain)
    matdata.register_category("PlasticProperty", PlasticProperty)
    matdata.register_category("ExpDataAvailability", ExpDataAvailability)
    matdata.register_category("GrainEqDiaEbsd", GrainEqDiaEbsd)
    matdata.register_category("TextureComponentProfile", TextureComponentProfile)
    matdata.register_category("EBSDParameters", EBSDParameters)
    matdata.register_category("TensileTestParameters", TensileTestParameters)

    default_source = "Material.py build() default"
    matdata.ingest("MaterialIdentity", MaterialIdentity(), source=default_source)
    matdata.ingest("ProcessingRoute", ProcessingRoute(), source=default_source)
    matdata.ingest("IrradiationCondition", IrradiationCondition(), source=default_source)
    matdata.ingest("CrystalFamily", CrystalFamily(), source=default_source)
    matdata.ingest("Phases", Phases(), source=default_source)
    matdata.ingest("PhysicalProperty", PhysicalProperty(), source=default_source)
    matdata.ingest("ElasticProperty", ElasticProperty(), source=default_source)
    matdata.ingest("TensileStressStrain", TensileStressStrain(), source=default_source)
    matdata.ingest("PlasticProperty", PlasticProperty(), source=default_source)
    matdata.ingest("ExpDataAvailability", ExpDataAvailability(), source=default_source)
    matdata.ingest("GrainEqDiaEbsd", GrainEqDiaEbsd(), source=default_source)
    matdata.ingest("TextureComponentProfile", TextureComponentProfile(crystal_family="FCC"),
                   source=default_source)
    matdata.ingest("EBSDParameters", EBSDParameters(), source=default_source)
    matdata.ingest("TensileTestParameters", TensileTestParameters(), source=default_source)

    return matdata


@dataclass(frozen = True, repr = True)
class IrradiationCondition:
    irr     : str = field(default = 'neutron') # Type of irradiation
    irr_temp: float = field(default = 400) # Temperature of irradiation in Kelvin
    irr_dpa : float = field(default = 1E-5) # displacements per atom

@dataclass(frozen = True, repr = True)
class CrystalFamily:
    xtal_family: str = field(default = 'mmm') # Crystal family: mmm, etc

@dataclass(frozen = True, repr = True)
class Phases:
    nphases: int = field(default = 2) # Number of phases
    namesPhases: np.ndarray = field(default_factory=lambda: np.ndarray([], dtype=str))
    phaseFractions: np.ndarray = field(default_factory=lambda: np.ndarray([], dtype=float))

@dataclass(frozen = True, repr = True)
class PhysicalProperty:
    density: float = field(default = 2700.0) # in kg m^-3

@dataclass(frozen = True, repr = True)
class ElasticProperty:
    E: float = field(default = 70E3) # Young's modulus in MPa

@dataclass(frozen = True, repr = True)
class TensileStressStrain:
    strain: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    stress: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

@dataclass(frozen = True, repr = True)
class PlasticProperty:
    Sy001: float = field(default = 135) # 0.1% proof strength, MPa
    Sy002: float = field(default = 150) # 0.2% proof strength, MPa
    Sy003: float = field(default = 155) # 0.3% proof strength, MPa
    HV0005: float = field(default = 50) # Vicker's hardness number @ 0.005 kg-f
    HV0010: float = field(default = 50) # Vicker's hardness number @ 0.010 kg-f
    HV0020: float = field(default = 50) # Vicker's hardness number @ 0.020 kg-f
    K: float = field(default = 1234) # Fracture toughness

@dataclass(frozen = True, repr = True)
class ExpDataAvailability:
    tt          : bool = field(default = True) # tensile test. True if available else False
    fatigue_low : bool = field(default = True)
    fatigue_high: bool = field(default = True)
    ebsd        : bool = field(default = True)
    tem         : bool = field(default = True)

@dataclass(frozen = True, repr = True)
class GrainEqDiaEbsd:
    modality: int   = field(default = 2)
    skewness: float = field(default = -1.02)
    kurtosis: float = field(default = 1.24)
    dist_grain_size : np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    dist_grain_count: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    dist_grain_prob : np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

@dataclass(frozen = True, repr = True)
class EBSDParameters:
    zero_fraction_uncorrected : float = field(default = 0.0)# 0 to 1
    zero_fraction_corrected : float = field(default = 0.0)# 0 to 1
    phase_fraction: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))# data for every phase

@dataclass(frozen = True, repr = True)
class TensileTestParameters:
    sample_type     : str   = field(default = 'value')# 'dogbonegrip','dogboneshoulder'
    strain_rate     : float = field(default = 0.0)
    test_temperature: float = field(default = 0.0)
#______________________________________________________________________________
# HELPER METHODS:
def TempKelvin(temp_celcius):
    """Tempkelvin."""
    # This is only to demonstrate a way of setting value in a dataclass.
    # Value could infact be taken directly in Kelvin, inside the "IrradiationCondition" class
    return 273.0 + temp_celcius
#______________________________________________________________________________
# CALLER METHODS
def generate():
    '''
    '''
    return build()
#______________________________________________________________________________
