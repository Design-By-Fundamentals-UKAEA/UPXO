from dataclasses import dataclass, field
import pandas as pandas # pd will be reserved for Physical Domain in UPXO
import numpy as np
import copy
###############################################################################
@dataclass(repr = False)
class pdomInstances():
    '''
    Summary:
        * Physical Domain instances
        * Each instance has the instance object and condensed object data

    Data fields:
        N: Number of physical domain instances to create
        O: object list of physical domain instances
        B: bounds of the physical domain
    '''
    count: int = -1
    print(40*'*')
    NPDI    = int(input('Number of Physical Domain instances:  '))
    print(''.join(['\n', 40*'*']))
    O       = [[i, 0] for i in range(NPDI)]
    B       = copy.deepcopy(O)
    GSSM    = copy.deepcopy(O)
    matdata = copy.deepcopy(O)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __repr__(self):
        if self.count < self.NPDI+1:
            repr1 = ''.join([40*'-', '\n We can hold ' + str(self.NPDI) + ' instances of the physical domain for you\n'])
            repr2 = ''.join([40*'-'])
        elif self.count == self.NPDI+1:
            repr1 = ''
            repr2 = ''.join(['Physical domain instance box full\n', 40*'-'])
        return repr1 + repr2
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __add__(self,
              *, 
              PDOM_info_instance_OBJECT,
              PDOM_bounds_instance_OBJECT,
              gstack_instance_OBJECT,
              material_instance_OBJECT):
        '''
        Adds a physical domain instance to self.
        '''
        try:
            self.O[self.count][1] = PDOM_info_instance_OBJECT
            self.B[self.count][1] = PDOM_bounds_instance_OBJECT
            self.GSSM[self.count][1] = gstack_instance_OBJECT
            self.matdata[self.count][1] = material_instance_OBJECT
        except:
            print('Maximum allowed instances = ', str(self.NPDI))
            print(''.join(['Physical domain instance box full \n', 40*'-']))
        else:
            print(''.join(['There are now ' + str(self.count + 1
                                                  ) + ' instances of the physical domain']))
        if self.count <= self.NPDI:
            self.count += 1
        else:
            pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    if __name__ == '__main__':
        pass
    else:
        count += 1
###############################################################################
@dataclass(frozen = False, repr = True)
class pdomain_info():
    '''
    Summary:
        Makes a basic dataclass needed to construct physical domain
    Call:
        > pdom(dimen=2, technique='vt', level=0, ninst0=2, ninst1=1, ninst2=0, ninst_tex=1)
        > pdom(dimen=3) # Rest of the fields will have default assignments
        > pdom() # All fields will be default assignments
    Suggestions:
        Instantiate as pdom, to avoid conflict with pandas, normally imported as pd
    Returns:
        InstanceName.data
    '''
    dimen    : int = field(default = 2)
    technique: str = field(default = 'vt') # or: 'mc'
    """
    vt_tool: Tool to be used in making Voronoi tessellation
    OPTIONS: scipy,  shapely,  freud, vtess, pyvoro++
    """
    vt_tool: str = field(default = 'scipy')
    """
    vt_seed_01: SEED POINT DISTRIBUTION
        'ur': uniform random | 'nr': normal random | 'rec': rectangular
        'hex': hexagonal | 'tri': triangular
        'hex_tri': combination of hex and tri: interleaved
        'hex_sq': combination of hex and square: interleaved
        'hex_ur': combination of hex and uniform random: interleaved
        'sc': simple cubic | 'fcc': face centred cubic | 'bcc': body centred cubic
        'hcp': hexagonal closed packing
        
        NOTE: trailing _p indicates purturbed coordinates
        'ur_p': uniform random | 'nr_p': normal random | 'rec_p': rectangular
        'hex_p': hexagonal | 'tri_p': triangular
        'hex_tri_p': a combin of hex and tri: interleaved
        'hex_sq_p': a combin of hex and square: interleaved
        'hex_ur_p': a combin of hex and uniform random: interleaved
        'fcc_p': face centred cubic | 'bcc_p': body centred cubic | 'hcp_p': hexagonal closed packing
        'import': import from file
    """
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    vt_seed_01: str = field(default = 'hex')
    """ vt_seed_01_A: PERTURBATION FACTORS for x, y and z """
    vt_seed_01_A: np.ndarray = field(default = np.array([0.1, 0.1, 0.1], dtype = int))
    """ vt_seed_01_B: STRETCH FACTORS FOR x, y and z """
    vt_seed_01_B: np.ndarray = field(default = np.array([1.0, 1.0, 1.0], dtype = int))
    """ vt_seed_02: SEED POINT DISTRIBUTION OPTIMIZATION FLAG"""
    vt_seed_02: str = field(default = 'dont_optimize') # or: 'dont_optimize'
    """vt_seed_02_A: OPTIMIZATION TECNIQUE"""
    vt_seed_02_A: str = field(default = 'equidistance')
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    """ some parameters governing the voronoi tessellation lattice geometry"""
    a1:     float = field(default = 0.10, repr=True) # Unit length no 1 in terms of geometric lattice
    a2:     float = field(default = 0.10, repr=True) # Unit length no 2 in terms of geometric lattice
    angle1: float = field(default = 60.0, repr=True) # Angle in degrees
    angle2: float = field(default = 60.0, repr=True) # Angle in degrees
    pertFlag: str = field(default = True, repr=True) # Should the lattice be perturbed?
    pertMethod: str = field(default = 'ru', repr = True) # perturbation rule
    px: float = field(default = 0.02, repr=True) # x-perturbation factor
    py: float = field(default = 0.02, repr=True) # y-perturbation factor
    pz: float = field(default = 0.02, repr=True) # y-perturbation factor
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    level    : int = field(default = 3) # Level of the GS in this PD instance
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    ninst0   : int = field(default = 2) # Number of instances of level 0 GS
    ninst1   : int = field(default = 3) # Number of instances of level 1 GS
    ninst2   : int = field(default = 2) # Number of instances of level 2 GS
    ninst3   : int = field(default = 1) # Number of instances of level 3 GS
    # This is in physical space
    ninst_tex: int = field(default = 1)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __repr__(self):
        reprstr0a = '\n--------------------------------------------------\n'
        reprstr0b = 'PHYSICAL DOMAIN INSTANCE:\n\n'
        reprstr1  = 'Dimensionality = ' + str(self.dimen) + '\n'
        reprstr2  = 'GS made using: ' + self.technique + ' technique' + '\n'
        reprstr3  = 'GS level = ' + str(self.level) + '\n'
        reprstr4  = 'GS Instance list is ' + \
            str([self.ninst0, self.ninst1, self.ninst2, self.ninst3]) + '\n'
        reprstr5  = 'Number of texture instances in physical domain = ' + \
            str(self.ninst_tex) + '\n'
        reprstr6  = '--------------------------------------------------\n'
        return reprstr0a + reprstr0b + reprstr1 + \
               reprstr2 + reprstr3 + reprstr4 + \
               reprstr5 + reprstr6
###############################################################################
@dataclass(frozen = False, repr = True)
class pdomain_bounds():
    '''
    A simple data class containing the bound information of the poly-crystal,
    be it Voronoi teseellated or Monte-Carlo simulation based grain structured
    '''
    # Coordinates of origin
    orix: float = field(default = 0.0)
    oriy: float = field(default = 0.0)
    oriz: float = field(default = 0.0)
    # domain length
    lenx: float = field(default = 1.0)
    leny: float = field(default = 1.0)
    lenz: float = field(default = 1.0)
    # domain pixel/coordinate step length
    delx: float = field(default = 0.1)
    dely: float = field(default = 0.1)
    delz: float = field(default = 0.1)
    # Coordinates of the end
    endx: float = field(default = 1.0)
    endy: float = field(default = 1.0)
    endz: float = field(default = 1.0)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __repr__(self):
        str1 = 'Domain starts at: '+ ''.join(str([self.orix,
                                                  self.oriy,
                                                  self.oriz]))
        str2 = 'Domain ends at: '  + ''.join(str([self.endx,
                                                  self.endy,
                                                  self.endz]))
        return str1 + '\n' + str2
###############################################################################
@dataclass(frozen = False, repr = True)
class gstack():
    '''
    Summary:
        A stack of _______

    Fields:
        Staks the instances
        gstack: Grain structure stack data
        GSS: Grain Structure Stack
        GSSM: Grain Structure Stack Matrix
            IDLevelN_GSS : IDs in GSS data of Nth level GS : Unlinked across levels
            IDLevelN_GSSM: IDs in GSSM data of Nth level GS:   Linked across levels
        List holding counts of grain strucure instances
    '''
    # Grain Structure Stack Matrix: Linked ID list of grain structure instances
    IDLevel0_GSSM: np.ndarray = field(default = np.array([], dtype = int))
    IDLevel1_GSSM: np.ndarray = field(default = np.array([], dtype = int))
    IDLevel2_GSSM: np.ndarray = field(default = np.array([], dtype = int))
    IDLevel3_GSSM: np.ndarray = field(default = np.array([], dtype = int))
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __post_init__(self):
        pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def build_IA(self, pdo_info):
        '''
        Construct the Instance Array. 
        At a shallow level, instance array stores the number of instances 
        across available levels as an array
        '''
        # Initiate Instance Array
        self.IA = [0, 0, 0, 0]
        # Make acceptable IA from the User input data / default
        if pdo_info.level == 0:
            # Level - 0 grain structure
            if pdo_info.ninst0 == 0:
                self.IA[0] = 1
            else:
                self.IA[0] = pdo_info.ninst0
        elif pdo_info.level == 1:
            # Level - 1 grain structure
            if pdo_info.ninst0 == 0:
                self.IA[0] = 1 
            else:
                self.IA[0] = pdo_info.ninst0
            if pdo_info.ninst1 == 0:
                self.IA[1] = 1
            else:
                self.IA[1] = pdo_info.ninst1
        elif pdo_info.level == 2:
            # Level - 2 grain structure
            if pdo_info.ninst0 == 0:
                self.IA[0] = 1
            else:
                self.IA[0] = pdo_info.ninst0
            if pdo_info.ninst1 == 0:
                self.IA[1] = 1
            else:
                self.IA[1] = pdo_info.ninst1
            if pdo_info.ninst2 == 0:
                self.IA[2] = 1
            else:
                self.IA[2] = pdo_info.ninst2
        elif pdo_info.level == 3:
            # Level - 3 grain structure
            if pdo_info.ninst0 == 0:
                self.IA[0] = 1
            else:
                self.IA[0] = pdo_info.ninst0
            if pdo_info.ninst1 == 0:
                self.IA[1] = 1
            else:
                self.IA[1] = pdo_info.ninst1
            if pdo_info.ninst2 == 0:
                self.IA[2] = 1
            else:
                self.IA[2] = pdo_info.ninst2
            if pdo_info.ninst3 == 0:
                self.IA[3] = 1
            else:
                self.IA[3] = pdo_info.ninst3
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def Level0_ID_GSS(self):
        '''
        Establish the level - 0 grain structure stack array - unlinked
        '''
        # Number of instances of Level-0 grain structure
        L0 = self.IA[0]
        # All instance IDs of Level 0 grain structure, continuous, unlinked
        self.l0i = [1+count for count in range(L0)]
        # Total number of instances of Level 0 grain structure
        self.p0 = len(self.l0i)
        self.lNi = [self.l0i]
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def Level1_ID_GSS(self):
        '''
        Establish the level - 1 grain structure stack array - unlinked
        '''
        # Call method for lower Level-0 grain structure
        self.Level0_ID_GSS()
        # Number of instances of Level-0, 1 grain structure
        L0, L1 = self.IA[0], self.IA[1]
        # All instance IDs of Level 1 grain structure, continuous, unlinked
        self.l1i = [1+L0+count for count in range(L0*L1)]
        # Total number of instances of Level 1 grain structure
        self.p1 = len(self.l1i)
        self.lNi.append(self.l1i)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def Level2_ID_GSS(self):
        '''
        Establish the level - 2 grain structure stack array - unlinked
        '''
        # Call method for lower Level-0 grain structure
        self.Level0_ID_GSS()
        # Call method for lower Level-1 grain structure
        self.Level1_ID_GSS()
        # Number of instances of Level-0, 1, 2 grain structure
        L0, L1, L2 = self.IA[0], self.IA[1], self.IA[2]
        # All instance IDs of Level 2 grain structure, continuous, unlinked
        self.l2i = [1+L0+L0*L1+count for count in range(L0*L1*L2)]
        # Total number of instances of Level 2 grain structure
        self.p2 = len(self.l2i)
        self.lNi.append(self.l2i)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def Level3_ID_GSS(self):
        '''
        Establish the level - 3 grain structure stack array - unlinked
        '''
        # Call method for lower Level-0 grain structure
        self.Level0_ID_GSS()
        # Call method for lower Level-1 grain structure
        self.Level1_ID_GSS()
        # Call method for lower Level-2 grain structure
        self.Level2_ID_GSS()
        # Number of instances of Level-0, 1, 2, 3 grain structure
        L0, L1, L2, L3 = self.IA[0], self.IA[1], self.IA[2], self.IA[3]
        # All instance IDs of Level 3 grain structure, continuous, unlinked
        self.l3i = [1+L0+L0*L1+L0*L1*L2+count for count in range(L0*L1*L2*L3)]
        # Total number of instances of Level 3 grain structure
        self.p3 = len(self.l3i)
        self.lNi.append(self.l3i)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def set_pmax(self, pdo_info):
        '''
        Calculate the maximum number of grain structure instances 
        amongst all available levels for the physical domain under 
        consideration
        '''
        # Max. no. of instances amonsgt level 0, 1, 2 and 3 grain structures
        if pdo_info.level == 0:
            self.pmax = np.array([self.p0],
                                 dtype = 'int').max()
        elif pdo_info.level == 1:
            self.pmax = np.array([self.p0, self.p1],
                                 dtype = 'int').max()
        elif pdo_info.level == 2:
            self.pmax = np.array([self.p0, self.p1, self.p2],
                                 dtype = 'int').max()
        elif pdo_info.level == 3:
            self.pmax = np.array([self.p0, self.p1, self.p2, self.p3],
                                 dtype = 'int').max()
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def Level0_ID_GSSM(self):
        '''
        Establish the level-0 grain struture stack matrix
        '''
        # Set up a template
        template = np.ones((self.pmax//self.p0), dtype = 'int')
        # Empty GS Stack Matrix
        self.IDLevel0_GSSM = np.array([], dtype = 'int')
        #
        for i in range(self.p0):
            self.IDLevel0_GSSM = np.append(self.IDLevel0_GSSM,
                                           template + self.l0i[i] - 1)
            if i == self.p0-1:
                self.IDLevel0_GSSM = self.IDLevel0_GSSM[np.newaxis].transpose()
        self.IDLevel_0_GSSM = self.IDLevel0_GSSM
        self.IDLevel_pdi_GSSM = [list(np.squeeze((self.IDLevel0_GSSM)))]
        self.GSStackMatrix = copy.deepcopy(self.IDLevel_0_GSSM)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def Level1_ID_GSSM(self):
        '''
        Establish the level-1 grain struture stack matrix
        '''
        # Set up a template
        self.Level0_ID_GSSM()
        template = np.ones((self.pmax//self.p1), dtype = 'int')
        self.IDLevel1_GSSM = np.array([], dtype = 'int')
        for i in range(self.p1):
            self.IDLevel1_GSSM = np.append(self.IDLevel1_GSSM,
                                           template + self.l1i[i] - 1)
            if i == self.p1-1:
                self.IDLevel1_GSSM = self.IDLevel1_GSSM[np.newaxis].transpose()
        self.IDLevel_01_GSSM = np.hstack((self.IDLevel0_GSSM,
                                          self.IDLevel1_GSSM)
                                         )
        self.IDLevel_pdi_GSSM.append(list(np.squeeze((self.IDLevel1_GSSM))))
        self.GSStackMatrix = copy.deepcopy(self.IDLevel_01_GSSM)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def Level2_ID_GSSM(self):
        '''
        Establish the level-2 grain struture stack matrix
        '''
        # Set up a template
        self.Level0_ID_GSSM()
        self.Level1_ID_GSSM()
        template = np.ones((self.pmax//self.p2), dtype = 'int')
        self.IDLevel2_GSSM = np.array([], dtype = 'int')
        for i in range(self.p2):
            self.IDLevel2_GSSM = np.append(self.IDLevel2_GSSM,
                                           template + self.l2i[i] - 1)
            if i == self.p2-1:
                self.IDLevel2_GSSM = self.IDLevel2_GSSM[np.newaxis].transpose()
        self.IDLevel_012_GSSM = np.hstack((self.IDLevel0_GSSM,
                                           self.IDLevel1_GSSM,
                                           self.IDLevel2_GSSM)
                                          )
        self.IDLevel_pdi_GSSM.append(list(np.squeeze((self.IDLevel2_GSSM))))
        self.GSStackMatrix = copy.deepcopy(self.IDLevel_012_GSSM)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def Level3_ID_GSSM(self):
        '''
        Establish the level-3 grain struture stack matrix
        '''
        # Set up a template
        self.Level0_ID_GSSM()
        self.Level1_ID_GSSM()
        self.Level2_ID_GSSM()
        template = np.ones((self.pmax//self.p3), dtype = 'int')
        self.IDLevel3_GSSM = np.array([], dtype = 'int')
        for i in range(self.p3):
            self.IDLevel3_GSSM = np.append(self.IDLevel3_GSSM,
                                           template + self.l3i[i] - 1)
            if i == self.p3-1:
                self.IDLevel3_GSSM = self.IDLevel3_GSSM[np.newaxis].transpose()
        self.IDLevel_0123_GSSM = np.hstack((self.IDLevel0_GSSM,
                                            self.IDLevel1_GSSM,
                                            self.IDLevel2_GSSM,
                                            self.IDLevel3_GSSM)
                                           )
        self.IDLevel_pdi_GSSM.append(list(np.squeeze((self.IDLevel3_GSSM))))
        self.GSStackMatrix = copy.deepcopy(self.IDLevel_0123_GSSM)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def build(self, pdo_info):
        # DEF SUMMARY: Build the instance map variables. These are:
            # IA
            # p0, p1, p2, p3
            # l0i, l1i, l2i, l3i
        # Step 1
        # Build the Instance arrays (IA) as per the dimentionality
        # ACCESS: gstack.IA, where, gstack is instance of gstack OR gstack()
        self.build_IA(pdo_info)
        if   pdo_info.level == 0: self.Level0_ID_GSS()
        elif pdo_info.level == 1: self.Level1_ID_GSS()
        elif pdo_info.level == 2: self.Level2_ID_GSS()
        elif pdo_info.level == 3: self.Level3_ID_GSS()
        self.set_pmax(pdo_info)
        if   pdo_info.level == 0:
            self.Level0_ID_GSSM()
            self.IDLevel_01_GSSM   = np.array([], dtype = int)
            self.IDLevel_012_GSSM  = np.array([], dtype = int)
            self.IDLevel_0123_GSSM = np.array([], dtype = int)
        elif pdo_info.level == 1:
            self.Level1_ID_GSSM()
            self.IDLevel_012_GSSM  = np.array([], dtype = int)
            self.IDLevel_0123_GSSM = np.array([], dtype = int)
        elif pdo_info.level == 2:
            self.Level2_ID_GSSM()
            self.IDLevel_0123_GSSM = np.array([], dtype = int)
        elif pdo_info.level == 3:
            self.Level3_ID_GSSM()
        def squeeze_unlinked_GSSM():
            self.IDLevel0_GSSM = np.squeeze(self.IDLevel0_GSSM)
            self.IDLevel1_GSSM = np.squeeze(self.IDLevel1_GSSM)
            self.IDLevel2_GSSM = np.squeeze(self.IDLevel2_GSSM)
            self.IDLevel3_GSSM = np.squeeze(self.IDLevel3_GSSM)
        squeeze_unlinked_GSSM()
###############################################################################
def make_PDI():
    '''
    make Physical Domain Instances
    NPDI: number of physical domain instances needed
    '''
    #  .  .  .  .  .  .  .  .  .  .  .  .
    import Material
    #  .  .  .  .  .  .  .  .  .  .  .  .
    pdoms = pdomInstances()
    #--------------------------------------------------
    # MAKE PHYSICAL DOMAIN
    for pdi_count in range(pdoms.NPDI):
        #--------------------------------------------------
        # Instantiate material
        matdata = Material.generate()
        #--------------------------------------------------
        # Generate physical domain info and bound objects
        pdo_info = pdomain_info()
        pdo_bounds = pdomain_bounds()
        #--------------------------------------------------
        # Instantiate material GS Stack data structure
        #  .  .  .  .  .  .  .  .  .  .  .  .
        from pdomain_2 import gstack
        #  .  .  .  .  .  .  .  .  .  .  .  .
        gstack = gstack()
        gstack.build(pdo_info)
        #--------------------------------------------------
        # Add physical domain instance
        pdoms.__add__(PDOM_info_instance_OBJECT = pdo_info,
                     PDOM_bounds_instance_OBJECT = pdo_bounds,
                     gstack_instance_OBJECT = gstack,
                     material_instance_OBJECT = matdata)
    return pdoms
###############################################################################
def make_LNI(PDs):
    '''
    LNI: Nth Level Instances, spanning ALL instances of physical domains
    PDs: To follow PEP8 guideline of code line length, Physical_Domains has
    been shortened to PDs. It is the Physical Domain Object Collection
    '''
    LNI = {gana_PDI: PDs.GSSM[gana_PDI][1].lNi for gana_PDI in range(PDs.NPDI)}
    return LNI
###############################################################################
def clean_querry_PDI_IDs(querry_PDI_IDs, Physical_Domains):
    '''
    Get the user querry into the queerY-- arg and use Physical_Domaoins format to 
    clean and format data before use
    '''
    # Data cleaning started
    # If input is float
    if isinstance(querry_PDI_IDs, str):
        if querry_PDI_IDs != 'all':
            print('Incorrect querry. I wwill consider all instances')
            querry_PDI_IDs = 'all'
    if querry_PDI_IDs == 'all':
        querry_PDI_IDs = range(Physical_Domains.NPDI)
    if isinstance(querry_PDI_IDs, float):
        querry_PDI_IDs = int(querry_PDI_IDs)
    # Convert to list
    if isinstance(querry_PDI_IDs, int):
        querry_PDI_IDs = [querry_PDI_IDs]
    if isinstance(querry_PDI_IDs, list):
        # Remove duplicates
        querry_PDI_IDs = list(set(querry_PDI_IDs))
        # Remove elements greater then the maximum allowed value
        if max(querry_PDI_IDs) > Physical_Domains.NPDI:
            print('Incorrect querry. I am correcting it to:')
            querry_PDI_IDs = querry_PDI_IDs.remove(Physical_Domains.NPDI)
    # Data cleaning ended
    return querry_PDI_IDs
###############################################################################
def print_UGSS(querry_PDI_IDs, LNI, Physical_Domains):
    '''
    print_UGSS: print Unlinked Grain Structure Stack
    querry_PDI_IDs: ID numbers of PDI being queried by user
    LNI: Nth Level Instances, spanning ALL instances of physical domains
    Physical_Domains: It is the Physical Domain Object Collection
    '''
    # Prepare reusable strings
    P, L, vvl1a1, vvl1a2, vvl1a3 = 'PD-', 'Level-', 80*'-', 40*'-', 80*'#'
    vvl1b = 'Unique instance IDs are: '
    # Prepare the user input on query data
    #  .  .  .  .  .  .  .  .  .  .  .  .
    from pdomain_2 import clean_querry_PDI_IDs
    #  .  .  .  .  .  .  .  .  .  .  .  .
    querry_PDI_IDs = clean_querry_PDI_IDs(querry_PDI_IDs, Physical_Domains)
    # Print user query information
    print(f'{vvl1a3} \n You are querying instances, {querry_PDI_IDs} of the physical domain')
    print("\n \n - - - - - UNLINKED GRAIN STRUCTURE STACK - - - - -\n \n")
    for gana_PD in querry_PDI_IDs:
        # For each physical domain instance
        print(vvl1a1) # A seperator
        for gana_level in range(Physical_Domains.O[gana_PD][1].level+1):
            # For each level of the current physical domain instance
            print(f'{P}{gana_PD}, {L}{gana_level}: {vvl1b}{LNI[gana_PD][gana_level]}')
    print(vvl1a1) # Final seperator
###############################################################################
def print_LGSS(querry_PDI_IDs, LNI, Physical_Domains):
    '''
    print_LGSS: print Linked Grain Structure Stack
    querry_PDI_IDs: ID numbers of PDI being queried by user
    LNI: Nth Level Instances, spanning ALL instances of physical domains
    Physical_Domains: It is the Physical Domain Object Collection
    '''
    # Prepare reusable strings
    P, L, vvl1a1, vvl1a2, vvl1a3 = 'PD-', 'Level-', 80*'-', 40*'-', 80*'#'
    vvl1b = 'Unique instance IDs are: '
    # Prepare the user input on query data
    #  .  .  .  .  .  .  .  .  .  .  .  .
    from pdomain_2 import clean_querry_PDI_IDs
    #  .  .  .  .  .  .  .  .  .  .  .  .
    querry_PDI_IDs = clean_querry_PDI_IDs(querry_PDI_IDs, Physical_Domains)
    # Print user query information
    print(f'{vvl1a3} \n You are querying instances, {querry_PDI_IDs} of the physical domain')
    print("\n \n - - - - - LINKED GRAIN STRUCTURE STACK - - - - -\n \n")
    for gana_PD in querry_PDI_IDs:
        # LOOP: For each physical domain instance
        print(vvl1a1) # A seperator
        for gana_level in range(Physical_Domains.O[gana_PD][1].level+1):
            # LOOP: For each level of the current physical domain instance
            # Extracct linked instances data
            linked_instances = Physical_Domains.GSSM[gana_PD][1].IDLevel_pdi_GSSM[gana_level]
            # Prepare the instances data
            linked_instances = list(np.squeeze(np.array(linked_instances)))
            # Present the instances data
            print(f'{P}{gana_PD}, {L}{gana_level}: {vvl1b}{linked_instances}')
    print(vvl1a1) # Final seperator
###############################################################################
def make_pd_df_gstack(querry_PDI_IDs, Physical_Domains):
    '''
    make_pd_df_gstack: Make pandas dataframe of gstack
    Physical_Domains: It is the Physical Domain Object Collection
    '''
    # Prepare reusable strings
    P, L, vvl1a1, vvl1a2, vvl1a3 = 'PD-', 'Level-', 80*'-', 40*'-', 80*'#'
    vvl1b = 'Unique instance IDs are: '
    # Prepare the user input on query data
    #  .  .  .  .  .  .  .  .  .  .  .  .
    from pdomain_2 import clean_querry_PDI_IDs
    #  .  .  .  .  .  .  .  .  .  .  .  .
    querry_PDI_IDs = clean_querry_PDI_IDs(querry_PDI_IDs, Physical_Domains)
    # Print user query information
    print(f'{vvl1a3} \n Instances, {querry_PDI_IDs} of the physical domain have been considered')
    gstack_pd_df = {}
    for gana_PD in querry_PDI_IDs:
        # LOOP: For each physical domain instance
        Dictionary = {}
        for gana_level in range(Physical_Domains.O[gana_PD][1].level+1):
            # LOOP: For each level of the current physical domain instance
            linked_instances = Physical_Domains.GSSM[gana_PD][1].IDLevel_pdi_GSSM[gana_level]
            Dictionary[''.join([L, str(gana_level)])] = list(np.squeeze(np.array(linked_instances)))
        gstack_pd_df[gana_PD] = pandas.DataFrame(Dictionary)
    print('Pandas dataframe(s) have be constructed for GSSM of each of the PD Instance(s)')
    return gstack_pd_df
###############################################################################
def print_instances_table(querry_PDI_IDs, Physical_Domains, gstack_pd_df):
    '''
    Prepare reusable strings
    '''
    P, L, vvl1a1, vvl1a2, vvl1a3 = 'PD-', 'Level-', 80*'-', 40*'-', 80*'#'
    vvl1b = 'Unique instance IDs are: '
    # Prepare the user input on query data
    #  .  .  .  .  .  .  .  .  .  .  .  .
    from pdomain_2 import clean_querry_PDI_IDs
    #  .  .  .  .  .  .  .  .  .  .  .  .
    querry_PDI_IDs = clean_querry_PDI_IDs(querry_PDI_IDs, Physical_Domains)
    # Print user query information
    print(f'Pandas DF for -- {P} Instances in {querry_PDI_IDs} of the PD:')
    from tabulate import tabulate
    print(vvl1a3)
    for gana_PD in querry_PDI_IDs:
        # LOOP: For each physical domain instance
        print(f'\n - - - - - PANDAS DATAFRAME "gstack_pd_df" for PD-{gana_PD} - - - - -\n ')
        gstack_pd_df[gana_PD].index.name = 'PD Chain #'
        print(tabulate(gstack_pd_df[gana_PD], headers = 'keys', tablefmt = 'fancy_grid'))
    print(vvl1a3)
###############################################################################
def get_level_instance_chains(gstack_pd_df,
                              Physical_Domains,
                              pullLink_pdi,
                              pullLink_lev,
                              Instance_Number,
                              NumberType):
    '''
    Get the entire Instance-Dependency-Chain (IDC) for a given GS instance
    '''
    if pullLink_lev > Physical_Domains.O[pullLink_pdi][1].level:
        print(''.join(['Max. available link number = ',
                       str(Physical_Domains.O[pullLink_pdi][1].level)]))
        print('No pull link set. Please repeat with correct input')
    else:
        df = gstack_pd_df[pullLink_pdi]
        if NumberType == 'gs.absolute':
            df_subset = df.loc[df[''.join(['Level-',
                                           str(pullLink_lev)])]==Instance_Number]
        elif NumberType == 'gs.relative':
            df_subset = df[''.join(['Level-',
                                    str(pullLink_lev)])][Instance_Number]
        from tabulate import tabulate
        # print(df_subset)
        print(tabulate(df_subset,
                       headers = 'keys',
                       tablefmt = 'fancy_grid'))
        return df_subset
###############################################################################
def write_instances(self):
    # Write Unlinked instances Pandas dataframe to file
    # REFER https://xlsxwriter.readthedocs.io/working_with_pandas.html
    pass
###############################################################################
def vis_all_gs_level_instances(Physical_Domains,
                               pullLink_pdi):
    '''
    Visualize the GS stack dependency
    '''
    import networkx as nx
    #  .  .  .  .  .  .  .  .  .  .  .  .
    import matplotlib.pyplot as plt
    #  .  .  .  .  .  .  .  .  .  .  .  .
    # BUG pullLink_pdi NOT IMPLEMENT CORRECTLY.
    # EFFECT OF THE ABOVE BUG: gives same graph irrespective of the valid input
    # for pullLink_pdi
    class vis_instances:
        def __init__(self):
            self.EdgeEnds = []
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        def addEdge(self, EdgeEnd1, EdgeEnd2):
            self.EdgeEnds.append([EdgeEnd1, EdgeEnd2])
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        def visualize(self):
            G = nx.Graph()
            G.add_edges_from(self.EdgeEnds)
            nx.draw_networkx(G, node_size = 200,
                             node_color = 'cyan',
                             alpha = 1.0,
                             node_shape = 'o',
                             linewidths = 5,
                             edge_color = 'black',
                             style = ':')
            plt.show()
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    pdo_info = Physical_Domains.O[pullLink_pdi][1]
    if pdo_info.level > 0:
        Graph = vis_instances()
    gstack = Physical_Domains.GSSM[pullLink_pdi][1]
    if pdo_info.level == 0:
        print('No graph visualization as Level = 0')
    elif pdo_info.level == 1:
        levs01 = np.hstack((gstack.IDLevel0_GSSM[np.newaxis].transpose(),
                            gstack.IDLevel1_GSSM[np.newaxis].transpose()))
        for count in levs01:
            Graph.addEdge(levs01[count][0], levs01[count][1])
    elif pdo_info.level == 2:
        levs01 = np.hstack((gstack.IDLevel0_GSSM[np.newaxis].transpose(),
                            gstack.IDLevel1_GSSM[np.newaxis].transpose()))
        levs12 = np.hstack((gstack.IDLevel1_GSSM[np.newaxis].transpose(),
                            gstack.IDLevel2_GSSM[np.newaxis].transpose()))
        for count in levs01:
            Graph.addEdge(levs01[count][0], levs01[count][1])
            Graph.addEdge(levs12[count][0], levs12[count][1])
    elif pdo_info.level == 3:
        levs01 = np.hstack((gstack.IDLevel0_GSSM[np.newaxis].transpose(),
                            gstack.IDLevel1_GSSM[np.newaxis].transpose()))
        levs12 = np.hstack((gstack.IDLevel1_GSSM[np.newaxis].transpose(),
                            gstack.IDLevel2_GSSM[np.newaxis].transpose()))
        levs23 = np.hstack((gstack.IDLevel2_GSSM[np.newaxis].transpose(),
                            gstack.IDLevel3_GSSM[np.newaxis].transpose()))
        for count in range(np.shape(levs01)[0]):
            Graph.addEdge(levs01[count][0], levs01[count][1])
            Graph.addEdge(levs12[count][0], levs12[count][1])
            Graph.addEdge(levs23[count][0], levs23[count][1])
    if pdo_info.level > 0:
        Graph.visualize()
###############################################################################
class maths:
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def cosd(angle):
        # cosine of angle in degree
        return np.cos(np.deg2rad(angle))
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def sind(angle):
        # sine of angle in degree
        return np.sin(np.deg2rad(angle))
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def alike(sourceArray, arrayType = 'ru'):
        # array like: make a new array (of specified type), the same shape as the sourceArray
        if arrayType == 'ru': # Random Uniform
            return np.random.random(np.shape(sourceArray))
        elif arrayType == 'z': # Random Uniform
            return np.zeros(np.shape(sourceArray))
        else:
            print('No match found for array like')
            pass
###############################################################################
class lattice(pdomInstances, maths):
    '''
    Purpose:
        1. generate 2D and 3D coordinates of all lattice instances across all PDI
        2. store 2D and 3D coordinates of all lattice instances across all PDI
        3. store lattice metadata 
        4. generate statistical data of all lattice instances
        5. store statistical data of all lattice instances

    Note:
        1. There will be **ninst0** number of unique base Level-0 lattices for 
            every instance of the physical domain

    Data fields:
        1. Lattice object
        1: Raw data: Coordinates
        2: Processed data: ckdtree implementation of VT coordinate data
        2: Processed data:
        2: Statistical data: statistics data of the lattice

    Design requirements:
        1. fast data access
        2. simple data structure for storage, access and write operations

    Design: User-inputs:
        1. a. For now, let us not ask user anything explicitly. Instead:
            the user will have to manually change parameter values internally

    Design: Internals:
        1. data centric
        2. both gridded and ungridded coordinate data for VT
        3. gridded coordinate data for MC

    Design: Outputs:
        1. object data: lattice data objects of all Level-0 lattices
        2. figure: a console output of lattice offering a sneak peak of 
            generated base lattice point distribution
        3. other data: coordinates, perturbations, statistics, metadata

    Rules:
        1. Arrays must be numpy
        2. Array structure must render internal use by xarray and dask

    TODO:
        1. use slots to enable faster data access and less memory usage
        2. make ckdTree mirrors of VT coordinate data
        3. segment coordinate space into rectangular partitions and return to 
            suitable containers
        4. with figure initialization inside a __post_init__,
            use def _repr_png_ to print out lattice on IPython console
    '''
    #//////////////////////////////////////////////////////////////////////////
    def __init__(self):
        '''
        Make __LAT_XU as data container for unperturbed x-coordinates
        It will conatain this data, of all physical domain instances
        '''
        gana_pdi = 0
        self.__allowed_add_methods = ('x-','x+','left', 'right',
                                    'y-','y+','bottom','top',
                                    'z-', 'z+', 'behind', 'back', 'front')
    #//////////////////////////////////////////////////////////////////////////
    def __len__(self, gana_pdi):
        length = pdomInstances.B[gana_pdi][1].endx - pdomInstances.B[gana_pdi][1].orix
        breadth = pdomInstances.B[gana_pdi][1].endy - pdomInstances.B[gana_pdi][1].oriy
        size = length * breadth
        if pdomInstances.O[gana_pdi][1].dimen == 3:
            thickness = pdomInstances.B[gana_pdi][1].endz - pdomInstances.B[gana_pdi][1].oriz
            size = size * thickness
        return f'Domain size of PDI = {gana_pdi} is {size} units'
    #//////////////////////////////////////////////////////////////////////////
    def __repr__(self):
        z = "Proper repr to appear in near future for lattice object(s)"
        return z
    #//////////////////////////////////////////////////////////////////////////
    def __add__(self, pdi_list, method):
        '''
        Defines addition tules and arithmetic for adding multiple lattices
        Lattices to be added should be at the PDI level
        Lattices must be of the same dimension
        Addition happens recursively from extreme right of the pdi_list, i.e.
            pdi_list[-1] towards instance at start position, i.e. pdi_list[0]

        ACCESS:
            lat.__add__([0, 1], 'x+')
            lat.__add__([1], 'front')
        
        EXAMPLES:
            lat.__add__([0, 1], 'x+'): 1 gets added to right of 0
            lat.__add__([0, 1], 'x-'): 1 gets added to left of 0
            lat.__add__([0, 1, 2], 'x-'): 2 gets added to left of 1 and,
                                        this (2+1) gets added to left of 0
        '''
        if not isinstance(pdi_list, list):
            raise TypeError("pdi_list must be a list. Please try again.")
        else:
            if not method in self.__allowed_add_methods:
                raise ValueError(''.join(['method must be one of allowed methods from',
                                          '\n              [lat.__allowed_add_methods]'
                                          ]
                                         )
                                 )
            elif method == 'x-' or method == 'left':
                # Define rules how to add lattice objects to left
                # Then define addition maths
                pass
            elif method == 'x+' or method == 'right':
                pass
            elif method == 'y-' or method == 'bottom':
                pass
            elif method == 'y+' or method == 'top':
                pass
            elif method == 'z-' or method == 'behind' or method == 'back':
                pass
            elif method == 'z+' or method == 'front':
                pass
    #//////////////////////////////////////////////////////////////////////////
    def __mul__(self, pdi, level_0_lattice_object_id, scaling_factor, scale_about):
        '''
        __mul__ performs lattice scaling
        INPUT RULES:
            pdi: int+: physical domain instance number
            level_0_lattice_object_id: ninst0 count number in this pdi
            scaling_factor
            scale_about: tuple: (facx, facy, facz)
                fac(x,y,z): 0-1:
                    if all 0.0, then scales about (orix, oriy, oriz)
                    if all 1.0, then scales about (endx, endy, endz)
                    if all 0.5, then scales about 0.5*[(endx, endy, endz)-
                                                       (orix, oriy, oriz)]
                    if (0.5, 0.6, 0.8), then scales about [0.5*(endx-orix),
                                                           0.6*(endy-oriy),
                                                           0.8*(endz-oriz)]
        ACCESS:
            lat.__mul__(0, 0, 0.5, (0, 0, 0))
        '''
        pass
    #//////////////////////////////////////////////////////////////////////////
    def container(self, display_structure = False):
        '''
        Create container data structure to host objects, raw data and metadata
        '''
        if display_structure:
            print(40*'#')
            print('\n STRUCTURE OF THE DATA MIRRORS: \n')
        LAT_unlinked = list([[] for i in range(self.NPDI)])
        LAT_linked = copy.deepcopy(LAT_unlinked)
        for gana_pdi in range(self.NPDI):
            # EEGA pdi parikramadolage
            # GSSM'nantha aakrutiyonda maadona: "deepcopy" haage irali
            level_max = self.O[gana_pdi][1].level
            LAT_unlinked[gana_pdi] = copy.deepcopy(self.GSSM[gana_pdi][1].lNi)
            LAT_linked[gana_pdi] = copy.deepcopy(self.GSSM[gana_pdi][1].GSStackMatrix)
            if display_structure:
                print(''.join([40*'>', '\n Object template (Unlinked) for lattice:']))
                print(LAT_unlinked[gana_pdi])
                print(''.join([10*'   >', '\n Object template (Linked) for lattice:']))
                print(LAT_linked[gana_pdi])
                print(40*'#')
            #######   #######   #######   #######   #######
            #DO NOT DELETE THIS BELOW COMMENTED SECTION !!!!!!!!!!!!!!!
            #for gana_level in range(level_max+1):
            #    # EEGA level parikramadolage
            #    for gana_inst in range(len(self.GSSM[gana_pdi][1].lNi[gana_level])):
            #        # EEGA instance parikramadolage
            #        print(f'PD.{gana_pdi} ~~ Level.{gana_level} ~~ Instance.{gana_inst}')
            #    print(40*'.')
            #print(40*'#')
            #######   #######   #######   #######   #######
        return LAT_unlinked, LAT_linked
    #//////////////////////////////////////////////////////////////////////////
    def make_metadata(self):
        '''
        Contains metadata about lattice instances across all levels of 
        all physical domain instances
        '''
        # NOTE: In all the booled in below, if not True, value to be False
        # NOTE: In all the valued in below, only unlinked data to be stored
            # Use lNi:gssm correspondance to associate this data to all 
            # instance values in the linked dataset
            # Storage format:
                # One dictionary for one instance
                # Example key:value pairs are below:
                    # {'mindistance': value of minimum distance,
                    #  'meandistance': value of mean distance,
                    #  'distance': 2D numpy array. Each entry to have
                    #              pointA ID - pointB ID - distance
                    # }
            # Access example:
                # ..statval_unlinked[gana_pdi][gana_level][gana_instance]
        # NOTE: conditional availabilities:
            # opttech_xx: Available only if lattice has been optimized
        # status_xx to hold True if raw lattice constructed
        self.status_flag_unlinked, self.status_flag_linked = self.container()
        # stat_xx to hold True if all statistics have been computed
        self.stat_flag_unlinked, self.stat_flag_linked = self.container()
        # statval_xx to hold statistics values
        self.stat_value_unlinked, _ = self.container()
        # opt_xx to hold True if lattice is optimized
        self.opt_flag_unlinked, self.opt_flag_linked = self.container()
        # opttech_xx to hold string about which optimization routine is used
        self.opttech_value_unlinked, self.opttech_value_linked = self.container()
        # pointcountseed_xx to hold number of seed points
        self.pointcloudseed_N_unlinked, self.pointcloudseed_N_linked = self.container()
        # pointcounttrimmed_xx to hold number of points in truncated/trimmed lattice
        self.pointcounttrimmed_N_unlinked, self.pointcounttrimmed_N_linked = self.container()
    #//////////////////////////////////////////////////////////////////////////
    def lattice_container(self, display_structure = False):
        '''
        Create container data structure to host objects of the lattice class, raw data 
        and related metadata
        
        NOTE: For memory contrainsts, only unlinked data will be stored
            It can easily be used for linked data by looking up lNi elements against 
            GSSM elements.
            They have been commented out (see #LAT_linked for example)
            Uncomment them and use appropriately
        '''
        if display_structure:
            print(40*'#')
            print('\n STRUCTURE OF THE DATA MIRRORS: \n')
        LAT_unlinked = list([[] for i in range(self.NPDI)])
        #LAT_linked = copy.deepcopy(LAT_unlinked)
        for gana_pdi in range(self.NPDI):
            # EEGA pdi parikramadolage
            # GSSM'nantha aakrutiyonda maadona: "deepcopy" haage irali
            level_max = self.O[gana_pdi][1].level
            LAT_unlinked[gana_pdi] = copy.deepcopy(self.GSSM[gana_pdi][1].lNi)
            #LAT_linked[gana_pdi] = copy.deepcopy(self.GSSM[gana_pdi][1].GSStackMatrix)
            if display_structure:
                print(''.join([40*'>', '\n Object template (Unlinked) for lattice objects:']))
                print(LAT_unlinked[gana_pdi])
                #print(''.join([10*'   >', '\n Object template (Linked) for lattice objects:']))
                #print(LAT_linked[gana_pdi])
                print(40*'#')
            #######   #######   #######   #######   #######
            #DO NOT DELETE THIS BELOW COMMENTED SECTION !!!!!!!!!!!!!!!
            #for gana_level in range(level_max+1):
            #    # EEGA level parikramadolage
            #    for gana_inst in range(len(self.GSSM[gana_pdi][1].lNi[gana_level])):
            #        # EEGA instance parikramadolage
            #        print(f'PD.{gana_pdi} ~~ Level.{gana_level} ~~ Instance.{gana_inst}')
            #    print(40*'.')
            #print(40*'#')
            #######   #######   #######   #######   #######
        return LAT_unlinked#, LAT_linked        
    #//////////////////////////////////////////////////////////////////////////
    def make_lattice_container(self):
        '''
        Build unlinked data structure for lattice object container
        '''
        self.LC = self.lattice_container()
        #print(self.LC)
    #//////////////////////////////////////////////////////////////////////////
    def status(self, *args):
        '''
        pdi_list: list: list of pdi for which status is to be displayed
        '''
        if len(args) == 0:
            # If no inputs are provided - a default execution
            status_flag_unlinked = self.status_flag_unlinked
            status_flag_linked = self.status_flag_linked
        else:
            # If user provides some input. Only 1st user input will be considered
            # Inputs not at 1st arg loc will be disregarded
            pdi_list = list(set(args[0])) # convert to list from unique of dict values
            if len(pdi_list) > 0: # If the actual list has elements in it
                if not all(isinstance(pdi, int) for pdi in pdi_list):
                    # Check if each element is an integrer
                    raise TypeError('Every pdi in pdi_list must be an integer. Please re-try')
                else:
                    if not max(pdi_list) < pdomInstances.NPDI:
                        # Check if maximum value is less than NPDI
                        raise ValueError('0 <= max(pdi_list) < lat.NPDI . Please re-try')
                    else:
                        # Initiate variables
                        status_flag_unlinked, status_flag_linked = [], []
                        # Iterate over pdi_list and update accrodingly
                        for pdi in pdi_list:
                            status_flag_unlinked.append(self.status_flag_unlinked[pdi])
                            status_flag_linked.append(self.status_flag_linked[pdi])
            else:
                status_flag_unlinked = self.status_flag_unlinked
                status_flag_linked = self.status_flag_linked                
        return status_flag_unlinked, status_flag_linked
    @property
    def statstat(self):
        return self.statstat_unlinked, self.statstat_linked
    @property
    def statval(self):
        return self.statval_unlinked, self.statval_linked
    @property
    def opt_flag(self):
        return self.opt_flag_unlinked, self.opt_flag_linked
    @property
    def opt_technique(self):
        return self.opttech_value_unlinked, self.opttech_value_linked
    @property
    def pointcloudseed_N(self):
        return self.pointcloudseed_N_unlinked, self.pointcloudseed_N_linked
    @property
    def pointcount_N_trimmed(self):
        return self.pointcounttrimmed_N_unlinked, self.pointcounttrimmed_N_linked
    #//////////////////////////////////////////////////////////////////////////
    def coord_container(self):
        '''
        Generates containers for base coordinate related data

        Variables not ending with _linked store actual raw coordinate related data
        Variables ending with _linked store object references only.

        Values already in _linked can still be useful even if not replaced by objects

        Coordinate related data are:
            [1] Unperturbed coordinates (identified by U, example: __LAT_XU)
            [2] Perturbed coordinates (identified by P, example: LAT_XP)
            [3] Perturbattion amount (identified as in example: LAT_PX)

        Objects could be: 
            [1] dataclass instances storing statistical data, or
            [2] dataclass instances storing perturbation parameters, etc.
        '''
        # See if all pdi have the same dimen and get it out
        dimensions = '3d' # Assume 3D
        for gana_pdi in range(self.NPDI):
            if gana_pdi>0:
                if self.O[gana_pdi-1][1].dimen != self.O[gana_pdi][1].dimen:
                    dimensions = '2d'
        """
        No linked data will be made. Consequently is ignored in returns.
        This decision is taken to save memory.

        However, this will put an overhead on the user to go a few steps more
        to take care of instance ID mapping, which should be straighforward
        To achieve this mapping, GSStackMatrix IDs can be looked up against
        their presence in lNi.
        """
        print(''.join([40*'*', '\n', 'Building coordinate data containers']))
        # Unperturbed coordinates. # Linked data not necessary
        self.__LAT_XU, _ = self.container()
        self.__LAT_YU, _ = self.container()
        if dimensions == '3d': self.__LAT_ZU, _ = self.container()
        # Perturbed coordinates. # Linked data not necessary
        self.LAT_XP, _ = self.container()
        self.LAT_YP, _ = self.container()
        if dimensions == '3d': self.LAT_ZP, _ = self.container()
        # Perturbation in coordinates. # Linked data not necessary
        self.LAT_PX, _ = self.container()
        self.LAT_PY, _ = self.container()
        if dimensions == '3d': self.LAT_PZ, _ = self.container()
        # Extracted metadata. # Linked data not necessary
        self.coord_emd_names, _ = self.container()
        self.coord_emd_data, _ = self.container()
        print(''.join(['Coordinate data containers build complete\n', 40*'*']))
    #//////////////////////////////////////////////////////////////////////////
    @property
    def xu(self): return self.__LAT_XU
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def yu(self): return self.__LAT_YU
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def zu(self): return self.__LAT_ZU
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def cu(self): return self.__LAT_XU, self.__LAT_YU, self.__LAT_ZU
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def xp(self): return self.LAT_XP
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def yp(self): return self.LAT_YP
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def get_zp(self): return self.LAT_ZP
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def get_cp(self): return self.LAT_XP, self.LAT_YP, self.LAT_ZP
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def get_px(self): return self.LAT_PX
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def get_py(self): return self.LAT_PY
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def get_pz(self): return self.LAT_PZ
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    @property
    def get_pc(self): return self.LAT_PX, self.LAT_PY, self.LAT_PZ
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __random_uniform(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __random_normal(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __random_clustered(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __rectangular(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __triangular(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __hexagonal(self, origin, length, end, a1, angle1):
        '''
        Build database of base seed points for hexagonal lattice
        
        Usage flexibility:
            By changing angle1 in the appropriate PDI user input,
            to other than 60 degrees, hexagons can also be stretched, either 
            along armchair or zigzag chiral axis of the lattice
        '''
        x1 = np.arange(origin[0], end[0], a1) # Row 1-x
        x2 = x1 + a1 * maths.cosd(angle1) # Row 2-x
        xi = np.vstack((x1, x2)) # Row 1 and Row 2-x
        y1 = origin[1] * np.ones(np.shape(x1)) # Row 1-y
        y2 = y1 + a1 * maths.sind(angle1) # Row 2-y
        yi = np.vstack((y1, y2)) # Row 1 and Row 2-y
        x_temp, y_temp = np.copy(xi), np.copy(yi)
        for count in range(int(end[1]/(a1 + 2 * a1 * maths.cosd(angle1)))):
            x_temp = np.vstack((x_temp, xi))
            y_temp = np.vstack((y_temp, yi+(count+1) * 2 * a1 * maths.sind(angle1)))
        #self.xbase = np.squeeze(np.reshape(x_temp, np.size(x_temp))[np.newaxis].T)
        #self.ybase = np.squeeze(np.reshape(y_temp, np.size(y_temp))[np.newaxis].T)
        self.xbase = np.reshape(x_temp, np.size(x_temp))[np.newaxis].T
        self.ybase = np.reshape(y_temp, np.size(y_temp))[np.newaxis].T
        self.zbase = maths.alike(self.xbase, 'z')
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __hexagonal_random_interleaved(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __hexagonal_rectangular_interleaved(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __hexagonal_triangular_interleaved(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __rectangular_triangular_interleaved(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __rectangular_random_interleaved(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __penrose_tiling(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __rangoli_tiling(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __simple_cubic(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __face_centred_cubic(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __body_centric_cubic(self): pass
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - -
    def __import_lattice(self): pass
    #//////////////////////////////////////////////////////////////////////////
    def base_coord_generator_vt(self, gana_pdi):
        '''
        This def is confined only to 'vt' type of grain structures as in intself
        if quite a big method. There is a seperate similar method for 'mc' type
        grain structures
        
        This method generates the lattice coordinates of the seed points needed
        for Voronoi tessellation.
        
        Input arguments:
            gana_pdi: physical domain instance count number: int
        
        Returns:
            All returns are directed towards self.
            1. xbase
            2. ybase
            3. zbase
        '''
        #gana_pdi = 0
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # In the below, domain is common across all pdi instances.
        # They are hence, also common for all instances across all levels
        domain = copy.deepcopy(pdomInstances.B[gana_pdi][1])
        origin = (domain.orix, domain.oriy, domain.oriz)
        length = (domain.lenx, domain.leny, domain.lenz)
        end    = (domain.endx, domain.endy, domain.endz)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if pdomInstances.O[gana_pdi][1].technique == 'vt':
            vtSeedDistribution = pdomInstances.O[gana_pdi][1].vt_seed_01
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Calculate unperturbed coordinates
            baseDistr = vtSeedDistribution.split('_')[0]
            if baseDistr == 'ur': pass
            elif baseDistr == 'nr': pass
            elif baseDistr == 'rec': pass
            elif baseDistr == 'tri': pass
            elif baseDistr == 'hex':
                a1 = pdomInstances.O[gana_pdi][1].a1
                angle1 = pdomInstances.O[gana_pdi][1].angle1
                self.__hexagonal(origin, length, end, a1, angle1)
            elif baseDistr == 'sc': pass
            elif baseDistr == 'fcc': pass
            elif baseDistr == 'bcc': pass
            elif baseDistr == 'hcp': pass
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Perturbed the coordinates if needed
            if len(vtSeedDistribution.split('_')) == 2:
                if vtSeedDistribution.split('_')[1] == 'p':
                        self.__perturb()
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if vtSeedDistribution == 'import':
                pass
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #//////////////////////////////////////////////////////////////////////////
    def base_coord_generator_mc(self, gana_pdi):
        pass
    #//////////////////////////////////////////////////////////////////////////
    def perturb(self, gana_pdi):
        '''
        NOTE: PERTURBATION TO BE APPLIED AFTER THE VORONOI TESSELLATION HAS BEEN 
            TRUNCATED. ELSE, THERE WILL BE GOOD CHANCES WHERE THE 
            DIFFERENT LEVBELS OF THE SAME SYTARTING MICROSTRUCRUE, HAVING 
            DIFFERENT PERTIURBATION EXTWENTS CAN HAVE DIFFWERENT NUMBER OF GRAINS, 
            MDEPENDING ON WHERE SOME OF THE POINTS MIGHT BE DURING TRUNCATION 
            PROCESS
        '''
        if pdomInstances.O[gana_pdi][1].pertMethod == 'ru':
            self.xpert = lat.px * maths.alike(self.xbase, 'ru')
            self.ypert = lat.py * maths.alike(self.ybase, 'ru')
            self.x = self.xbase + self.xpert
            self.y = self.ybase + self.ypert
            # Create z-coordinate of zeros if pdi dimensionality if 2
            if pdomInstances.O[gana_pdi][1].dimen == 3:
                self.zpert = lat.pz * maths.alike(self.ybase, 'ru')
                self.z = self.zbase + self.zpert
    #//////////////////////////////////////////////////////////////////////////
    def optimize(self):
        pass
    #//////////////////////////////////////////////////////////////////////////
    def stretch(self):
        pass
    #//////////////////////////////////////////////////////////////////////////
    def make_xyz(self):
        '''
        This makes the traditional 3-column coordinate array of x, y and z
        
        Note: 
            [1] A return is intentionallly done here to reduce lattice obj size
            If needed in future, this can easily be internalized by removing
            return and makeing self.xyz = np.hstack(....)
        '''
        xyz = np.hstack((self.xbase, self.ybase, self.zbase))
        return xyz
    #//////////////////////////////////////////////////////////////////////////
    def make_xyz_matrix(self, xyz):
        '''
        This makes the traditional 3-column coordinate array of x, y and z
        in the form of a numpy matrix
        
        Note: 
            [1] A return is intentionallly done here to reduce lattice obj size
            If needed in future, this can easily be internalized by removing
            return and chaing the code appropriately
        '''
        xyz_matrix = np.matrix(xyz)
        return xyz_matrix
    #//////////////////////////////////////////////////////////////////////////
    def point_cloud_mean(self, xyz):
        '''
        Calcuaklte the geometric mean of the point cloud
        '''
        geo_mean = np.mean(xyz, axis = 0)
        return geo_mean
    #//////////////////////////////////////////////////////////////////////////
    def squeeze_base_coord(self, xbase, ybase, zbase):
        '''
        TO DEPRECATE
        '''
        xbase_squeezed = np.squeeze(self.xbase)
        ybase_squeezed = np.squeeze(self.ybase)
        zbase_squeezed = np.squeeze(self.zbase)
        return xbase_squeezed, ybase_squeezed, zbase_squeezed
    #//////////////////////////////////////////////////////////////////////////
    @property
    def setDPI(self):
        '''
        a setter to set the figure dpi value
        '''
        self.dpi = 100
    #//////////////////////////////////////////////////////////////////////////
    def visualize_seeds(self, xyz, axis_obj, gana_pdi, gana_l0):
        '''
        Enables simple and quick visualixzation of the lattice seed points
        '''
        # -- -- -- -- -- -- -- -- --
        # Make the traditional 3-column coordinate array of x, y and z
        # xyz_matrix = self.make_xyz_matrix(self.make_xyz())
        # -- -- -- -- -- -- -- -- --
        # Calculate geometric mean of the point cloud
        geo_mean = self.point_cloud_mean(xyz)
        # -- -- -- -- -- -- -- -- --
        # Make the scatter plot
        axis_obj.scatter(xyz[:,0], xyz[:,1], marker = 'o', color = 'black')
        axis_obj.scatter(geo_mean[0], geo_mean[1], marker = 'x', color = 'red')
        # Set the labels
        axis_obj.set_xlabel('X axis')
        axis_obj.set_ylabel('Y axis')
        #axis_obj.set_zlabel('Z axis')
        axis_obj.set_title(''.join(['PDi = ', str(gana_pdi), ', L0i = ', str(gana_l0)]))
        # -- -- -- -- -- -- -- -- --
    #//////////////////////////////////////////////////////////////////////////
    def finalize(self, gana_pdi):
        '''
        A couple lines of codes to finalize creation of lattice
        '''
        if pdomInstances.O[gana_pdi][1].dimen == 2:
            self.zbase = maths.alike(self.xbase, 'z')
            self.zpert = maths.alike(self.xbase, 'z')
            self.z = maths.alike(self.x, 'z')
    #//////////////////////////////////////////////////////////////////////////
###############################################################################
def make_lattice(Physical_Domain_Object):
    '''
    Definition to build all instances of the lattice using lattice class and 
    the pdomInstances class
    '''
    from pdomain_2 import lattice
    # Instanctiate the lattice class
    lat = lattice()
    # Make containers for metadatas
    #TODO: can this be made to work automatically upin instantiation of the calass
    # this way, we could as well sacve a couple lines of code and introduce code simiplicitity
    lat.make_metadata()
    # Build unlinked data structure for lattice object container
    # TODO: this could as well be made to initialise automatiically
    # upon the initiatialization of the lattice class. This would also be useful thing to do
    lat.make_lattice_container()
    #//////////////////////////////////////////////////////////////////////////
    use_sub_plots = False
    def prepare_a_figure(plt, dpi, gana_pdi, ninst0, dimen):
        '''
        Make the axes objects for plotting lattice seed pooint distribution
        These objects will be taeken in by thwe plotting function
        '''
        fig = plt.figure(dpi = dpi)
        axis_obj = list(range(0, ninst0))
        #ninst0 = Physical_Domain_Object.O[gana_pdi][1].ninst0
        if not use_sub_plots:
            axis_obj = fig.add_subplot(1, 1, 1)
        else:
            for axcount in range(0, ninst0):
                if dimen == 2:
                    axis_obj[axcount] = fig.add_subplot(gana_pdi+1,
                                                        ninst0,
                                                        axcount + 1 )
                else:
                    axis_obj[axcount] = fig.add_subplot(gana_pdi+1,
                                                        ninst0,
                                                        axcount + 1,
                                                        projection = '3d' )
        return axis_obj
    #//////////////////////////////////////////////////////////////////////////
    for gana_pdi in range(Physical_Domain_Object.NPDI):
        #-----------------------------------------------------------
        ninst0 = Physical_Domain_Object.O[gana_pdi][1].ninst0
        dimen  = Physical_Domain_Object.O[gana_pdi][1].dimen
        #-----------------------------------------------------------
        print(''.join([40*'=', '\n Physical domain instance number >>> ', str(gana_pdi),]))
        #-----------------------------------------------------------
        # NOW BRANCH AS NECESSARY
        op_in_this_instance = 'optimizeprevious'
        if op_in_this_instance == 'optimizeprevious':
            pass
        elif op_in_this_instance == 'perturbprevious':
            pass
        elif op_in_this_instance == 'makenew':
            pass
        #-----------------------------------------------------------
# =============================================================================
#         for gana_l0 in range(ninst0):
#             lat.base_coord_generator_vt(gana_pdi)
#             xyz = lat.make_xyz()
#             lat.LC[gana_pdi][0][gana_l0] = xyz
#             lat.coord_container()
# =============================================================================
        #-----------------------------------------------------------
        import matplotlib.pyplot as plt
        if use_sub_plots:
            axis_obj = prepare_a_figure(plt, lat.setDPI, gana_pdi, ninst0, dimen)

        for gana_l0 in range(ninst0):
            print(''.join([40*'=', '\n    - Level 0 instance num : ', str(gana_l0),]))
            lat.coord_container()
            # Generate base lattice coordinates of the Voronoi tesselaltion
            lat.base_coord_generator_vt(gana_pdi)
            xyz = lat.make_xyz()
            lat.LC[gana_pdi][0][gana_l0] = xyz
            if not use_sub_plots:
                axis_obj = prepare_a_figure(plt, lat.setDPI, gana_pdi, ninst0, dimen)
                # Visualize the point cloud
                lat.visualize_seeds(xyz, axis_obj, gana_pdi, gana_l0)
            else:
                lat.visualize_seeds(xyz, axis_obj[gana_l0], gana_pdi, gana_l0)
        plt.show()
        del(axis_obj)
    #//////////////////////////////////////////////////////////////////////////
    return lat
###############################################################################
class opt_ckdtree:
    def __init__(self):
        pass
    def __coordinates_vertices_vt(self):
        pass
###############################################################################
class hdf5(lattice):
    def __init__(self):
        pass
    def write(self):
        pass
###############################################################################