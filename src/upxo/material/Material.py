from dataclasses import dataclass, field#, asdict
import numpy as np
'''
    Module introduction
'''

def build():
    '''
    Sumamry:
        This def uses classes in this module to build the material data base
    
    User data input type:
        Standard def input type supported
        When no inputs are provided, defaults prescribed in class data is used
        
    Call:
        > from Material import build
        > matdata = build()
    
    Return:
        matdata: Object Data Dictionary
        
    Access:
        matdata.data
        matdata.data.keys()

    Suggestions:
        instantiate with 'variable' name: "matdata"
    
    Data access:
        > matdata.data # Displays the entire data structure
        > matdata.data.keys() # Displays outermost keys
        > matdata.data.values() # Displays values of outermost keys
    
    Developers:
        > keys should be class names
        > whilst importing classes, use three class names together
            For examnple,
            > from Material import class1, class2, class3
            > from Material import class 4, class 5, class 6
    '''
    from ODDict import ObjectDataDictionary
    matdata = ObjectDataDictionary()
    
    from Material import MaterialIdentity, ProcessingCondition, IrradiationCondition
    from Material import CrystalFamily, Phases, PhysicalProperty
    from Material import ElasticProperty, TensileStressStrain, PlasticProperty
    from Material import ExpDataAvailability, GrainEqDiaEbsd, TexCompVolFracFCC
    from Material import TexFibreVolFracFCC, TexCompWidth, EBSDParameters
    from Material import TensileTestParameters
    
    # Append for the first time
    matdata.append(MaterialIdentity())
    matdata.append(ProcessingCondition())
    matdata.append(IrradiationCondition())
    matdata.append(CrystalFamily())
    matdata.append(Phases())
    matdata.append(PhysicalProperty())
    matdata.append(ElasticProperty())
    matdata.append(TensileStressStrain())
    matdata.append(PlasticProperty())
    matdata.append(ExpDataAvailability())
    matdata.append(GrainEqDiaEbsd())
    matdata.append(TexCompVolFracFCC())
    matdata.append(TexFibreVolFracFCC())
    matdata.append(TexCompWidth())
    matdata.append(EBSDParameters())
    matdata.append(TensileTestParameters())
    
    # Retain the following lines.  These are to be used, in case, the 
    # user wishes to update the already existing data on a segment by
    # segment basis
    
    #matdata.CrossCheckAndAppend(MaterialIdentity())
    #matdata.CrossCheckAndAppend(ProcessingCondition())
    #matdata.CrossCheckAndAppend(IrradiationCondition())
    #matdata.CrossCheckAndAppend(CrystalFamily())
    #matdata.CrossCheckAndAppend(Phases())
    #matdata.CrossCheckAndAppend(PhysicalProperty())
    #matdata.CrossCheckAndAppend(ElasticProperty())
    #matdata.CrossCheckAndAppend(TensileStressStrain())
    #matdata.CrossCheckAndAppend(PlasticProperty())
    #matdata.CrossCheckAndAppend(ExpDataAvailability())
    #matdata.CrossCheckAndAppend(GrainEqDiaEbsd())
    #matdata.CrossCheckAndAppend(TexCompVolFracFCC())
    #matdata.CrossCheckAndAppend(TexFibreVolFracFCC())
    #matdata.CrossCheckAndAppend(TexCompWidth())
    #matdata.CrossCheckAndAppend(EBSDParameters())
    #matdata.CrossCheckAndAppend(TensileTestParameters())
    
    return matdata

@dataclass(frozen=False, repr=True)
class MaterialIdentity:
    '''
    MaterialIdentity()
    '''
    # Now, we assign the default arguments
    name : str = field(default = 'cu') # Name of the material
    alloy: str = field(default = 'value') # Alloy grade
    comp : str = field(default = 'value', compare = False) # Composition

@dataclass(frozen = True, repr = True)
class ProcessingCondition:
    ht    : str = field(default = 'heat treatment') # Heat treatment
    pro   : str = field(default = 'extruded') # Processing
    app   : str = field(default = 'cooling pipe') # Application
    appLoc: str = field(default = 'W-Ci unterface') # Application location

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
    namesPhases: str = field(default = np.ndarray([], dtype = str))
    phaseFractions: np.ndarray = field(default = np.ndarray([], dtype = float))

@dataclass(frozen = True, repr = True)
class PhysicalProperty:
    density: float = field(default = '2700') # in kg m^-3

@dataclass(frozen = True, repr = True)
class ElasticProperty:
    E: float = field(default = 70E3) # Young's modulus in MPa

@dataclass(frozen = True, repr = True)
class TensileStressStrain:
    strain: np.ndarray= field(default = np.array([], dtype = float))
    stress: np.ndarray= field(default = np.array([], dtype = float))

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
    dist_grain_size : np.ndarray = field(default = np.array([], dtype = float))
    dist_grain_count: np.ndarray = field(default = np.array([], dtype = int))
    dist_grain_prob : np.ndarray = field(default = np.array([], dtype = float))

@dataclass(frozen = True, repr = True)
class TexCompVolFracFCC:
    import random
    cube_tc_vf  : float = field(default = random.random()/10)
    ndcube_tc_vf: float = field(default = random.random()/10) 
    rdcube_tc_vf: float = field(default = random.random()/10)
    goss_tc_vf  : float = field(default = random.random()/10)
    brass_tc_vf : float = field(default = random.random()/10)
    copper_tc_vf: float = field(default = random.random()/10)
    s_tc_vf     : float = field(default = random.random()/10)
    t1_tc_vf    : float = field(default = random.random()/10)
    t2_tc_vf    : float = field(default = random.random()/10)
    p_tc_vf     : float = field(default = random.random()/10)

@dataclass(frozen = True, repr = True)
class TexFibreVolFracFCC:
    import random
    cube_tf_vf : float = field(default = random.random()/10)
    alpha_tf_vf: float = field(default = random.random()/10)
    beta_tf_vf : float = field(default = random.random()/10)

@dataclass(frozen = True, repr = True)
class TexCompWidth:
    import random
    cube_tc_w  : float = field(default = random.random()*10)
    ndcube_tc_w: float = field(default = random.random()*10)
    rdcube_tc_w: float = field(default = random.random()*10)
    goss_tc_w  : float = field(default = random.random()*10)
    brass_tc_w : float = field(default = random.random()*10)
    copper_tc_w: float = field(default = random.random()*10)
    s_tc_w     : float = field(default = random.random()*10)
    t1_tc_w    : float = field(default = random.random()*10)
    t2_tc_w    : float = field(default = random.random()*10)
    p_tc_w     : float = field(default = random.random()*10)

@dataclass(frozen = True, repr = True)
class EBSDParameters:
    zero_fraction_uncorrected : float = field(default = 'value')# 0 to 1
    zero_fraction_corrected : float = field(default = 'value')# 0 to 1
    phase_fraction: np.ndarray = field(default = 'value')# data for every phase

@dataclass(frozen = True, repr = True)
class TensileTestParameters:
    sample_type     : str   = field(default = 'value')# 'dogbonegrip','dogboneshoulder'
    strain_rate     : float = field(default = 'value')
    test_temperature: float = field(default = 'value')
#______________________________________________________________________________
# HELPER METHODS:    
def TempKelvin(temp_celcius):
    # This is only to demonstrate a way of setting value in a dataclass.
    # Value could infact be taken directly in Kelvin, inside the "IrradiationCondition" class
    return 273.0 + temp_celcius
#______________________________________________________________________________
# CALLER METHODS
def generate():
    '''
    '''
    import Material
    return Material.build()
#______________________________________________________________________________