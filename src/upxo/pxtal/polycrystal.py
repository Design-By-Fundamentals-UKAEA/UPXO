import random
import numpy as np
import string

# from upxo.material.Material import *  # Unused

class PolyXTAL():
    """
    Top-level polycrystal data container for UPXO.

    Holds grain structure data (GSD), material property data (MPD), feature ID
    databases, flag arrays, and geometric parameter tables for multi-level
    (L0/L1/L2) hierarchical grain structures including twins, grain-boundary
    zones, prior-austenitic packets, and laths.
    """
    #**************************
    PXID     = None # Few high level ID of the poly-crystal
    MPD      = None # Material property data
    GSD      = None # Grain Structure Data
    #**************************
    origin   = None # Co-ordinates of the origin point
    xbase, ybase, zbase = None, None, None # Base coordinates
    x, y, z = None, None, None # Perturbed coordinate x
    xpert, ypert, zpert = None, None, None # Perturbed coordinate x
    #**************************
    # FUNDAMENTAL FEATURES @LEVEL-0
    L0GS_PolyXTAL   = 1
    L0GS_NGrains = None
    G0  = None # Grains
    GE0 = None # Grain edges
    GV0 = None # Grain vertices
    # PROPERTIES OF LEVEL-0
    Ng0  = None # Number of grains
    Nge0 = None # Number of grain edges
    Ngv0 = None # Number of grain vertices
    #**************************
    Precipitates = None
    #**************************
    ID0_base = None # Feature ID values of L0GS
    ID0_pair = None # Pair values of ID for L0-L0 and L0-L0 (inverted) one-one feature correspondance
    ID1_base = None # Feature ID values of L1GS
    ID1_pair = None # Pair values of ID for L0-L1, L1-L0, L1-L1 and L1-L1 (inverted), one-one feature correspondance
    ID_ctex  = None # Feature ID values of crystallographic texture
    #**************************
    PX_flags = None # Poly-crystal flag values
    GR_flags = None # Grain flag values (mostly morphological)
    #**************************
    GGP2A    = None # Grain Geometric Parameter 2D Area
    GGP2L    = None # Grain Geometric Parameter 2D Edge length
    #------------------------------------------------------------------
    def __init__(self):
        """Initialise the instance."""
        self.ustrs = []
    #------------------------------------------------------------------
    @property
    def setcmdlHouseKeepingRules(self):
        """House keeping rules related values."""
        self.cmdSep      = '_'
        self.cmdSepCount = 40
    #------------------------------------------------------------------
    @property
    def template_PXID(self):
        """Template pxid."""
        # WORKING
        base       = {'gs_UID': None, 'ct_UID': None,}
        PolyXTAL.PXID = {instCount: base.copy() for instCount in range(PolyXTAL.GSD['N__lev0_i'])}
        #PolyXTAL.PXID = {instCount: base.copy() for instCount in range(2)}
        #return PolyXTAL.PXID
    @property
    def setPXID(self):
        """Setpxid."""
        self.template_PXID
        self.make_USTRs()
        self.set_gs_UID()
        self.set_ct_UID()
    def make_USTRs(self):
        """Build and return USTRs."""
        for i0 in range(PolyXTAL.GSD['N__lev0_i']):
            #self.ustrs.append('{}'.format(''.join(random.choice(string.ascii_letters + string.digits+string.punctuation) for i in range(12))))
            self.ustrs.append('{}'.format(''.join(random.choice(string.ascii_letters) for i in range(12))))

    def set_gs_UID(self):
        """Set or update gs UID."""
        for i0 in range(PolyXTAL.GSD['N__lev0_i']):
            PolyXTAL.PXID[i0]['gs_UID'] = 'gs_' + self.ustrs[i0]

    def set_ct_UID(self):
        """Set or update ct UID."""
        for i0 in range(PolyXTAL.GSD['N__lev0_i']):
            PolyXTAL.PXID[i0]['ct_UID'] = 'ct_' + self.ustrs[i0]

    #------------------------------------------------------------------
# =============================================================================
#     @property
#     def setupL0GS(self):
#         """Set details of the Level 0 poly-xtal."""
#         """Usage:
#                  setupL0GS()
#         Dev. history: 
#             09-05-2022 - Working
#         """
#         print(self.cmdSepCount*self.cmdSep)
#         print('SET L0-GS DETAILS')
#         PolyXTAL.nPhases        = int(input("Number of phases (default 1) >>> ") or 1)
#         PolyXTAL.namesPhases    = input("Phase name array [pname1, pname2, etc]. default cu >>> ") or ['cu']
#         PolyXTAL.phaseFractions = np.asfarray(input("Phase fraction array [VfP1, VfP2, etc]. default [1.0] >>> " ) or [1.0])
# =============================================================================
    #------------------------------------------------------------------
    @property
    def setupL1GS(self):
        """Set details of the Level 1 poly-xtal."""
        if int(PolyXTAL.GSD['gslevel']) >= 1:
            print(self.cmdSepCount*self.cmdSep)
            print('SET L1-GS DETAILS')
            
            if PolyXTAL.flag_level1gs['gbz']:
                self.setGBZData()
            if PolyXTAL.flag_level1gs['twin']:
                self.setTwinData()
            if PolyXTAL.flag_level1gs['pap']:
                self.setPAPData()
            if PolyXTAL.flag_level1gs['subgrain']:
                self.setPAPData()
            if PolyXTAL.flag_level1gs['lath']:
                self.setLathData()

        self.set_flag_level1gs()
        #self.setGBZData()
        #self.setTwinData()
        #self.setPAPData()
        #self.setLathData()

    @property
    def template_flag_level1gs(self):
        """Template flag level1gs."""
        PolyXTAL.flag_level1gs = {'gbz'     : None,
                               'twin'    : None,
                               'pap'     : None,
                               'subgrain': None,
                               'lath'    : None,
                               }

    def set_flag_level1gs(self):
        """Set or update flag level1gs."""
        self.template_flag_level1gs
        PolyXTAL.flag_level1gs['gbz']      = True
        PolyXTAL.flag_level1gs['twin']     = True
        PolyXTAL.flag_level1gs['pap']      = True
        PolyXTAL.flag_level1gs['subgrain'] = True
        PolyXTAL.flag_level1gs['lath']     = True

    @property
    def templateGBZData(self):
        """Templategbzdata."""
        PolyXTAL.gbzSpecs = {'gbzDistribution': None,
                          'gbzVfperGrain'  : None,
                          'gbzType'        : None,
                          'gbzThickness'   : None,
                          }

    def setGBZData(self):
        """Setgbzdata."""
        self.templateGBZData
        PolyXTAL.gbzSpecs['gbzDistribution'] = 'allGrains' # Options: allGrains, byGrainID, byGrainCount, byGrainPercentage
        PolyXTAL.gbzSpecs['gbzVfperGrain']   = 0.10 # 0 <= Vf < 0.5 (upper limit is assumed). If 0.10, then 10% of a GBZ hosting grain will be occupied by GBZ
        PolyXTAL.gbzSpecs['gbzType']         = 'uniform' # option: non-uniform     &    uniform
        PolyXTAL.gbzSpecs['gbzThickness']    = 'deriveFromGbzVf'
        
    @property
    def templateTwinData(self):
        """Templatetwindata."""
        #PolyXTAL.twinVolumeFrac = np.asfarray(input("Phase wise twin volume fraction array [TVfP1, TVfP2, etc]. default [0.05]") or [0.05])

        PolyXTAL.twinSpecs = {'twinVolumeFrac'         : None, # ACCESS: PolyXTAL.twinSpecs['twinVolumeFrac']
                           'twinWidth_min_fraction' : None, # ACCESS: PolyXTAL.twinSpecs['twinWidth_min_fraction']
                           'twinWidth_mean_fraction': None, # ACCESS: PolyXTAL.twinSpecs['twinWidth_mean_fraction']
                           'twinWidth_max_fraction' : None, # ACCESS: PolyXTAL.twinSpecs['twinWidth_max_fraction']
                           'twinGrain_ori_relation' : None, # ACCESS: PolyXTAL.twinSpecs['twinGrain_ori_relation']
                           }

    def setTwinData(self):
        """Settwindata."""
        self.templateTwinData
        if PolyXTAL.flag_level1gs['twin']:
            PolyXTAL.twinSpecs['twinVolumeFrac'] = input("Twin volume fraction (default: 0.05) >>>") or 0.05
            if PolyXTAL.twinSpecs['twinVolumeFrac'] == 0:
                print("Ok, removing twin requirement")
            if PolyXTAL.twinSpecs['twinVolumeFrac'] > 0:
                PolyXTAL.twinSpecs['twinWidth_min_fraction']  = input("Min.  width of twin as Frac. of host grain mean edge length (default 0.05) >>> ") or '0.05'
                PolyXTAL.twinSpecs['twinWidth_mean_fraction'] = input("Mean. width of twin as Frac. of host grain mean edge length (default 0.20) >>> ") or '0.20'
                PolyXTAL.twinSpecs['twinWidth_max_fraction']  = input("Max.  width of twin as Frac. of host grain mean edge length (default 0.30) >>> ") or '0.30'
                PolyXTAL.twinSpecs['twinGrain_ori_relation']  = input("Type of twin orientation relationship (default: ks) small case >>> ") or 'ks'
        
    @property
    def templatePAPData(self):
        """Templatepapdata."""
        PolyXTAL.papSpecs = {'n_min_PriorAustPockets' : None,
                          'n_mean_PriorAustPockets': None,
                          'n_max_PriorAustPockets' : None,
                          'dividerLineType'        : None,
                          'jaggedLineTypeAngleDev'    : None,
                          }
    def setPAPData(self):
        """Setpapdata."""
        self.templatePAPData
        PolyXTAL.papSpecs['n_min_PriorAustPockets' ] = input("Min num. of prior autenite pockets (default 3) >>> ")  or '3'
        PolyXTAL.papSpecs['n_mean_PriorAustPockets'] = input("Mean num. of prior autenite pockets (default 3) >>> ") or '3'
        PolyXTAL.papSpecs['n_max_PriorAustPockets' ] = input("Max num. of prior autenite pockets (default 5) >>> ")  or '5'
        PolyXTAL.papSpecs['dividerLineType' ]        = input("Grain divider line type (default: straight) OPTIONS: jagged>>> ") or 'straight'
        if PolyXTAL.papSpecs['dividerLineType' ] == 'jagged':
            PolyXTAL.papSpecs['jaggedLineTypeAngleDev'] = [-5, 5]
    #------------------------------------------------------------------
    @property
    def templateSubGrainData(self):
        """Templatesubgraindata."""
        self.subgrainSpecs = {'NOTE': 'this should be done only after tex calculations has been finished',
                              'minMisAng'      : None,
                              'maxMisAng'      : None,
                              'reorientSceheme': None,
                              'NumDivisions'   : None,
                              }
    #------------------------------------------------------------------
    @property
    def setupL2GS(self):
        """
        Set details of the Level 2 poly-xtal
        """
        if int(PolyXTAL.GSD['gslevel']) >= 2:
            self.setLathData()

    @property
    def templateLathData(self):
        """Templatelathdata."""
        PolyXTAL.lathSpecs = {'width_min_fraction' : None,
                           'width_mean_fraction': None,
                           'width_max_fraction' : None,
                           }
    def setLathData(self):
        """Setlathdata."""
        self.templateLathData
        PolyXTAL.lathSpecs['width_min_fraction' ] = input("Min.  width of lathe as Frac. of host grain mean pocket edge length (default 0.05) >>> ") or '0.05'
        PolyXTAL.lathSpecs['width_mean_fraction'] = input("Mean. width of lathe as Frac. of host grain mean pocket edge length (default 0.20) >>> ") or '0.20'
        PolyXTAL.lathSpecs['width_max_fraction' ] = input("Max.  width of lathe as Frac. of host grain mean pocket edge length (default 0.30) >>> ") or '0.30'
        
    @property
    def templateParticleData(self):
        """Templateparticledata."""
        PolyXTAL.ParticleSpecs = {'material': None,
                               'shape'   : None, # "circular" or "circulargn" (Gaussian noise) or "circularpn" (perlin noise)
                               'sizeType': None, # "value" or "fraction" # Value in um or fraction of grain size
                               'minSize' : None, # if sizeType is fraction, then this value < meanSize
                               'meanSize': None, # if sizeType is fraction, then this value < maxSize
                               'maxSize' : None, # if sizeType is fraction, then this value < grainSize/10
                               }
    def setParticleData(self):
        """Setparticledata."""
        PolyXTAL.ParticleSpecs['material'] = input("Particle material name >>> ") or 'particleName'
        PolyXTAL.ParticleSpecs['shape']    = input("Particle shape >>> ")   or 'circular'
        PolyXTAL.ParticleSpecs['sizeType'] = input("Particle type of size specification (default: fraction) >>> ") or 'fraction'
        PolyXTAL.ParticleSpecs['minSize']  = input("Particle minSize >>> ")  or ''
        PolyXTAL.ParticleSpecs['meanSize'] = input("Particle meanSize >>> ") or ''
        PolyXTAL.ParticleSpecs['maxSize']  = input("Particle maxSize >>> ")  or ''
    
    @property
    def templateParticleClusterData(self):
        """Templateparticleclusterdata."""
        PolyXTAL.ParticleClusterSpecs = {'icvf'               : None, # Individual cluster volume fraction
                                      'distributionType'   : None, # Type of the distribution. Options: random, Gaussian
                                      'makeClusterEnvelope': None, # 
                                      }
    def setParticleClusterData(self):
        """Setparticleclusterdata."""
        PolyXTAL.ParticleClusterSpecs['icvf']                = input("Particle cluster name >>> ") or 'clusterName'
        PolyXTAL.ParticleClusterSpecs['distributionType']    = input("Particle cluster distirbution type (random: default)>>> ") or 'random'
        PolyXTAL.ParticleClusterSpecs['makeClusterEnvelope'] = input("Make particle cluster envelope? (yes: default) >>> ") or 'yes'
    #--------------------------------------------------------------------------------------------------------------------------------
    @property
    def template_ID0_base(self):
        """
        CALL: px.make_ID0_base
        """
        base = {'GRAIN'  : None, # List of ID numbers of all grains
                'SURF'   : None, # List of ID numbers of all grain boundary surfaces
                'EDGE'   : None, # List of ID numbers of all grain boundary edges
                'JPOINT' : None, # List of ID numbers of all grain boundary junction points
                } # Key 1 refers to base grain structure (@Level-0) instance number 1. GS instance keys will have unit increment.
        PolyXTAL.ID0_base = {instCount: base.copy() for instCount in range(PolyXTAL.GSD['N__lev0_i'])}

        # IDs of polycrystal features: Level 0: FUNDAMENTAL FEATURES OF PolyXTAL INSTANCES: DICTIONARY
        # DATA CONSTRUCTION AND ACCESS:
            # id0_grains  : #1D NUMPY INT ARRAY. #ACCESS: ID0_base['grain']  << grains: grain cells. INTEGER # Parent Cell id. 2d and 3d
            # id0_surfaces: #1D NUMPY INT ARRAY. #ACCESS: ID0_base['surf']   << surfaces: grain surfaces # Parent Cell surface id. 3d only 
            # id0_bedges  : #1D NUMPY INT ARRAY. #ACCESS: ID0_base['edge']   << bedges: grain boundary edges # grain boundary edges ID. 2d and 3d 
            # id0_jpoints : #1D NUMPY INT ARRAY. #ACCESS: ID0_base['jpoint'] << jpoints: junction points # Junction points ID
    #--------------------------------------------------------------------------------------------------------------------------------
    @property
    def template_ID0_pair(self):
        """
        CALL: px.make_ID0_pair
        """
        base = {'GRAIN'       : None, # Parent Grain ID. Retain here for quick ref purposes.
                'GRAIN_SURF'  : None, # List of parent grain IDs, each against an ID list of grain boundary surfaces attached to it
                'GRAIN_EDGE'  : None, # List of parent grain IDs, each against an ID list of grain boundary edges attached to its surfaces
                'GRAIN_JPOINT': None, # List of parent grain IDs, each against an ID list of grain boundary junction points attached to its surfaces
                'SURF_GRAIN'  : None, # List of grain boundary surface IDs, each against a list of IDs of parent grains that it shares with
                'EDGE_GRAIN'  : None, # List of grain boundary edge IDs, each against a list of IDs of parent grains that it shares with
                'JPOINT_GRAIN': None, # List of grain boundary junction point IDs, each against a list of IDs of parent grains that it shares with
                
                'SURF_EDGE'   : None, # List of grain boundary surface IDs, each against an ID list of grain boundary edges attached to it
                'SURF_JPOINT' : None, # List of grain boundary surface IDs, each against an ID list of grain boundary junction points attached to it
                'EDGE_SURF'   : None, # List of grain boundary edge IDs, each against a list of IDs of grain boundary surfaces that it shares with
                'JPOINT_SURF' : None, # List of grain boundary junction point IDs, each against a list of IDs of grain boundary surfaces that it shares with
                
                'EDGE_JPOINT' : None, # List of grain boundary edge IDs, each against an ID list of grain boundary junction points attached to it
                'JPOINT_EDGE' : None, # List of grain boundary junction point IDs, each against a list of IDs of grain boundary edges that it shares with
                }
        PolyXTAL.ID0_pair = {instCount: base.copy() for instCount in range(PolyXTAL.GSD['N__lev0_i'])}
        # @ above line: Each data set dictionatry refers to one instance of base grain structure (@Level-0).

        # ID pairs of polycrystal features: Level 0: FUNDAMENTAL FEATURE PAIRS OF PolyXTAL INSTANCES: DICTIONARY
    
        # DATA CONSTRUCTION AND ACCESS:
        # NOTE1: px_i_count: grain structure instance count
            # GRAIN       : id0_grains #1D NUMPY INT ARRAY. #ACCESS: ID0_base['GRAIN']  << grains: grain cells. INTEGER # Parent Cell id. 2d and 3d
            # GRAIN_SURF  : from   id0_grain    & id0_surfaces    #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['GRAIN_SURF']
            # GRAIN_EDGE  : from   id0_grains   & id0_bedges      #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['GRAIN_EDGE']
            # GRAIN_JPOINT: from   id0_grains   & id0_jpoints     #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['GRAIN_JPOINT']
            # SURF_GRAIN  : from   id0_surfaces & id0_grains      #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['SURF_GRAIN']
            # EDGE_GRAIN  : from   id0_bedges   & id0_grains      #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['EDGE_GRAIN']
            # JPOINT_GRAIN: from   id0_jpoints  & id0_grains      #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['JPOINT_GRAIN']
            # SURF_EDGE   : from   id0_surfaces & id0_bedges      #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['SURF_EDGE']
            # SURF_JPOINT : from   id0_surfaces & id0_jpoints     #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['SURF_JPOINT']
            # EDGE_SURF   : from   id0_bedges   & id0_surfaces    #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['EDGE_SURF']
            # JPOINT_SURF : from   id0_jpoints  & id0_surfaces    #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['JPOINT_SURF']
            # EDGE_JPOINT : from   id0_bedges   & id0_jpoints     #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['EDGE_JPOINT']
            # JPOINT_EDGE : from   id0_jpoints  & id0_bedges      #2D NUMPY INT ARRAY    #ACCESS: ID0_pair[px_i_count]['JPOINT_EDGE']
    #-------------------------------------------------------------------------------------------------------------------------------
    @property
    def template_ID1_base(self):
        """
        CALL: px.make_ID1_base
        """
        base = {'GRAIN'     : None, # Parent Grain ID. Retain here for quick ref purposes
                
                'GBZ'       : None, # List of ID numbers of all grain boundary zones
                'GBZ_EB_ED' : None, # List of ID numbers of all external boundary edges for grain boundary zones
                'GBZ_IB_ED' : None, # List of ID numbers of all internal boundary edges for grain boundary zones
                'GBZ_EB_JP' : None, # List of ID numbers of all external boundary junction points for grain boundary zones
                'GBZ_IB_JP' : None, # List of ID numbers of all internal boundary junction points for grain boundary zones
                
                'GC'        : None, # List of ID numbers of all grain cores
                'GC_EB_ED'  : None, # List of ID numbers of all external boundaries for grain boundary cores
                'GC_IB_ED'  : None, # List of ID numbers of all internal boundaries for grain boundary cores
                'GC_EB_JP'  : None, # List of ID numbers of all external boundary junction points
                'GC_IB_JP'  : None, # List of ID numbers of all internal boundary junction points
                
                'TWIN'      : None, # List of ID numbers of all twin zones
                'TWIN_BED'  : None, # List of ID numbers of all twin boundary edges
                'TWIN_BJP'  : None, # List of ID numbers of all twin boundary junction points
                
                'APCKT'     : None, # List of ID numbers of all prior-austenitic packets
                'APCKT_BE'  : None, # List of ID of each prior-austenitic packet of all prior-austenitic packet boundary edges
                'APCKT_BJP' : None, # List of ID numbers of all prior-austenitic packet boundary junction points
                }

        base1 = {instCount: base.copy() for instCount in range(PolyXTAL.GSD['N__lev0_i'])} # Key 1 refers to 1st instance of Level1 GS. Level-1 GS keys will have unit increment.
        PolyXTAL.ID1_base = {instCount: base1.copy() for instCount in range(PolyXTAL.GSD['N__lev1_i'])}# Key 1 refers to base grain structure (@Level-0) instance number 1. GS instance keys will have unit increment.

        # return PolyXTAL.ID1_base
        # IDs of polycrystal features: Level 1: DICTIONARY
        
        # DATA CONSTRUCTION AND ACCESS:
        # NOTE1: px_i_count: poly-xtal instance count
        # NOTE2: lev0_i_count: Level 0 grain structure instance count
            #  GRAIN    : id0_grains      # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['grain']
    
            #  GBZ      : id1_gbz         # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gbz']
            #  GBZ_EB_ED: id1_gbz_eb_ed   # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gbz_eb_ed']
            #  GBZ_IB_ED: id1_gbz_ib_ed   # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gbz_ib_ed']
            #  GBZ_EB_JP: id1_gbz_eb_jp   # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gbz_eb_jp']
            #  GBZ_IB_JP: id1_gbz_ib_jp   # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gbz_ib_jp']
            
            #  GC       : id1_gc          # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gc']
            #  GC_EB_ED : id1_gc_eb_ed    # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gc_eb_ed']
            #  GC_IB_ED : id1_gc_ib_ed    # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gc_ib_ed']
            #  GC_EB_JP : id1_gc_eb_jp    # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gc_eb_jp']
            #  GC_IB_JP : id1_gc_ib_jp    # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['gc_ib_jp']
            
            #  TWIN     : id1_twin        # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['twin']
            #  TWIN_BED : id1_twinbe      # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['twin_bed']
            #  TWIN_BJP : id1_twinbjp     # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['twin_bjp']
            
            #  APCKT    : id1_apckt       # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['apckt']
            #  APCKT_BE : id1_apcktbe     # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['apckt_be']
            #  APCKT_BJP: id1_apcktbjp    # 1D NUMPY INT ARRAY    #ACCESS:  ID1_base[ni0]['apckt_bjp']
    #-------------------------------------------------------------------------------------------------------------------------------
    @property
    def template_ID1_pair(self):
        """
        CALL: px.make_ID1_pair
        """
        base = {'GRAIN'           : None, # Parent Grain ID. Retain here for quick ref purposes
                'GC'              : None, # List of ID numbers of all grain cores. Retain here for quick ref purposes
                'TWIN'            : None, # List of ID numbers of all twin zones. Retain here for quick ref purposes
                'APCKT'           : None, # List of ID numbers of all prior-austenitic packets. Retain here for quick ref purposes
  				    						   
                'GRAIN__GBZ'      : None, # List of ID numbers of grains having a gbz and list of corresponding gbz id numbers
                'GBZ__GRAIN'      : None, # List of ID numbers of gbz, against the list of hosting grain IDs
                'GRAIN__GC'       : None, # List of ID numbers of grains having a grain core
                'GC__GRAIN'       : None, # List of ID numbers of grain cores, against the list of hosting grain
                'GRAIN__TWIN'     : None, # List of grain IDs having twins and twin IDs in each of these grains
                'TWIN__GRAIN'     : None, # List of all twin IDs, each against the ID of grain hosting it
                'GRAIN__APCKT'    : None, # List of grain ID numbers hosting prior-austenitic packets, each against the list of ID numbers of all prior-austenitic packets
                'APCKT__GRAIN'    : None, # List of ID of all prior-austenitic packets, each against the list of ID numbers of hosting grains
  				    						   
                'GRAIN__GBZ_EB_ED': None, # Grain ID list, each against the list of edge IDs of the external boundaries of its grain boundary zone
                'GRAIN__GBZ_IB_ED': None, # Grain ID list, each against the list of edge IDs of the internal boundaries of its grain boundary zone
                'GRAIN__GBZ_EB_JP': None, # Grain ID list, each against the list of junction point IDs of the external boundaries of its grain boundary zone
                'GRAIN__GBZ_IB_JP': None, # Grain ID list, each against the list of junction point IDs of the internal boundaries of its grain boundary zone
                'GRAIN__GC_EB_ED' : None, # Grain ID list, each against the list of edge IDs of the external boundaries of its grain core
                'GRAIN__GC_IB_ED' : None, # Grain ID list, each against the list of edge IDs of the internal boundaries of its grain core
                'GRAIN__GC_EB_JP' : None, # Grain ID list, each against the list of junction point IDs of the external boundaries of its grain core
                'GRAIN__GC_IB_JP' : None, # Grain ID list, each against the list of junction point IDs of the internal boundaries of its grain core
                'GRAIN__TWIN_BED' : None, # Grain ID list, each against the list of edge IDs of the boundaries of each of its twin zones. 3D NUMPY ARRAY
                'GRAIN__TWIN_BJP' : None, # Grain ID list, each against the list of junction point IDs of the boundaries of each of its twin zones. 3D NUMPY ARRAY
                'GRAIN__APCKT_BE' : None, # Grain ID list, each against the list of edge IDs of the boundaries of each of its prior-austenitic packets. 3D NUMPY ARRAY
                'GRAIN__APCKT_BJP': None, # Grain ID list, each against the list of junction point IDs of the boundaries of each of its prior-austenitic packets. 3D NUMPY ARRAY
  				    						   
                'GBZ_EB_ED__GRAIN': None, # List of IDs of all GBZ external boundary edges, each against the ID of the grains it shares
                'GBZ_IB_ED__GRAIN': None, # List of IDs of all GBZ internal boundary edges, each against the ID of the grains it shares
                'GBZ_EB_JP__GRAIN': None, # List of IDs of all GBZ external boundary edge junction points, each against the ID of the grains it shares
                'GBZ_IB_JP__GRAIN': None, # List of IDs of all GBZ internal boundary edge junction points, each against the ID of the grains it shares
                'GC_EB_ED__GRAIN' : None, # List of IDs of all GC external boundary edges, each against the ID of the grains it shares
                'GC_IB_ED__GRAIN' : None, # List of IDs of all GC internal boundary edges, each against the ID of the grains it shares
                'GC_EB_JP__GRAIN' : None, # List of IDs of all GC external boundary edge junction points, each against the ID of the grains it shares
                'GC_IB_JP__GRAIN' : None, # List of IDs of all GC internal boundary edge junction points, each against the ID of the grains it shares
                'TWIN_BED__GRAIN' : None, # List of IDs of all Twin boundary edges, each against the ID of the grains it shares
                'TWIN_BJP__GRAIN' : None, # List of IDs of all Twin boundary edge junction point, each against the ID of the grains it shares
                'APCKT_BE__GRAIN' : None, # List of IDs of all Prior-Austenitic Packet external boundary edges, each against the ID of the grains it shares
                'APCKT_BJP__GRAIN': None, # List of IDs of all Prior-Austenitic Packet external boundary edge junction point, each against the ID of the grains it shares
                }
        # IDs of polycrystal feature pairs: Level 1: DICTIONARY
        base1 = {instCount: base.copy() for instCount in range(PolyXTAL.GSD['N__lev0_i'])} # Key 1 refers to 1st instance of Level1 GS. Level-1 GS keys will have unit increment.
        PolyXTAL.ID1_pair = {instCount: base1.copy() for instCount in range(PolyXTAL.GSD['N__lev1_i'])}# Key 1 refers to base grain structure (@Level-0) instance number 1. GS instance keys will have unit increment.

        # DATA CONSTRUCTION AND ACCESS:
            # GRAIN           : from id0_grains                     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN']
            # GC              : from id1_gc                         ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GC']
            # TWIN            : from id1_twin                       ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['TWIN']
            # APCKT           : from id1_apckt                      ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['APCKT']
            
            # GRAIN__GBZ      : from id0_grains    & id1_gbz        ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GBZ']
            # GBZ__GRAIN      : from id1_gbz       & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GBZ__GRAIN']
            # GRAIN__GC       : from id0_grains    & id1_gc         ## 2D NUMPY ARRAY # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GC']
            # GC__GRAIN       : from id1_gc        & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GC__GRAIN']
            # GRAIN__TWIN     : from id0_grains    & id1_twin       ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__TWIN']
            # TWIN__GRAIN     : from id1_twin      & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['TWIN__GRAIN']
            # GRAIN__APCKT    : from id0_grains    & id1_apckt      ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__APCKT']
            # APCKT__GRAIN    : from id1_apckt     & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['APCKT__GRAIN']
            
            # GRAIN__GBZ_EB_ED: from id0_grains    & id1_gbz_eb_ed  ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GBZ_EB_ED']
            # GRAIN__GBZ_IB_ED: from id0_grains    & id1_gbz_ib_ed  ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GBZ_IB_ED']
            # GRAIN__GBZ_EB_JP: from id0_grains    & id1_gbz_eb_jp  ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GBZ_EB_JP']
            # GRAIN__GBZ_IB_JP: from id0_grains    & id1_gbz_ib_jp  ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GBZ_IB_JP']
            # GRAIN__GC_EB_ED : from id0_grains    & id1_gc_eb_ed   ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GC_EB_ED']
            # GRAIN__GC_IB_ED : from id0_grains    & id1_gc_ib_ed   ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GC_IB_ED']
            # GRAIN__GC_EB_JP : from id0_grains    & id1_gc_eb_jp   ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GC_EB_JP']
            # GRAIN__GC_IB_JP : from id0_grains    & id1_gc_ib_jp   ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__GC_IB_JP']
            # GRAIN__TWIN_BED : from id0_grains    & id1_twinbe     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__TWIN_BED']
            # GRAIN__TWIN_BJP : from id0_grains    & id1_twinbjp    ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__TWIN_BJP']
            # GRAIN__APCKT_BE : from id0_grains    & id1_apcktbe    ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__APCKT_BE']
            # GRAIN__APCKT_BJP: from id0_grains    & id1_apcktbjp   ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GRAIN__APCKT_BJP']
            
            # GBZ_EB_ED__GRAIN: from id1_gbz_eb_ed & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GBZ_EB_ED__GRAIN']
            # GBZ_IB_ED__GRAIN: from id1_gbz_ib_ed & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GBZ_IB_ED__GRAIN']
            # GBZ_EB_JP__GRAIN: from id1_gbz_eb_jp & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GBZ_EB_JP__GRAIN']
            # GBZ_IB_JP__GRAIN: from id1_gbz_ib_jp & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GBZ_IB_JP__GRAIN']
            # GC_EB_ED__GRAIN : from id1_gc_eb_ed  & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GC_EB_ED__GRAIN']
            # GC_IB_ED__GRAIN : from id1_gc_ib_ed  & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GC_IB_ED__GRAIN']
            # GC_EB_JP__GRAIN : from id1_gc_eb_jp  & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GC_EB_JP__GRAIN']
            # GC_IB_JP__GRAIN : from id1_gc_ib_jp  & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['GC_IB_JP__GRAIN']
            # TWIN_BED__GRAIN : from id1_twinbe    & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['TWIN_BED__GRAIN']
            # TWIN_BJP__GRAIN : from id1_twinbjp   & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['TWIN_BJP__GRAIN']
            # APCKT_BE__GRAIN : from id1_apcktbe   & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['APCKT_BE__GRAIN']
            # APCKT_BJP__GRAIN: from id1_apcktbjp  & id0_grains     ## 2D NUMPY ARRAY  # ACCESS: ID1_pair[px_i_count][lev0_i_count]['APCKT_BJP__GRAIN']
    #--------------------------------------------------------------------------------------------------------------------------------
    @property
    def template_ID_ctex(self):
        """
        CALL: px.make_ID_ctex
        """
        base = {'ORI_SAMPLING_TYPE'  : None, # Type of orientation sampling to employ
                'ORI_SAMPLING_RULE'  : None, # Rule for sampling to be employed
                'ORIID'              : None, # ID list of sampled orientations
                }
        base1 = {instCount: base.copy() for instCount in range(PolyXTAL.GSD['N__lev0_i'])} # Key 1 refers to texture (@Level-0) instance number 1. Tex instance keys will have unit increment.
        PolyXTAL.ID_ctex = {instCount: base1.copy() for instCount in range(PolyXTAL.GSD['N__lev1_i'])}# Key 1 refers to grain structure (@Level-0) instance number 1. GS instance keys will have unit increment.
    
    # IDs of polycrystal features: Level 0: CRYSTALLOGRAPHIC TEXTURE DICTIONARY

    # DATA CONSTRUCTION
        # ORI_SAMPLING_TYPE: Type of orientation sampling to employ
            # 'exp_ebsd_gsa' if directly sampled from a list of orientations from grain structure analysis
            # 'exp_ebsd_ctf' if directly sampled from EBSD map in ctf file format
            # 'eos_list' if directly sampled from discretised Euler orientation space
            # 'texModel_existing' if to be sampled from an existing texture model
            # 'texModel_build' if to be sampled after building texture model
            # ACCESS:   ID_ctex['ori_sampling_type']
        
        # ORI_SAMPLING_RULE: Rule for sampling to be employed
            # None if no make randomize selection without any rules
            # 'md' if mackenzie misorientation distribution is to be respected
            # ACCESS:   ID_ctex['ori_sampling_rule']
        
        # ORIID: ID list of sampled orientations
            # 1D NUMPY INT ARRAY
            # ACCESS:   ID_ctex['oriid']
    #--------------------------------------------------------------------------------------------------------------------------------
    @property
    def template_PX_flags(self):
        """
        CALL: px.make_PX_flags
        """
        PolyXTAL.PX_flags = {'f01': None, # 1D NP array: specifies meshing tool to employ             : STR : abaqus OR gmsh OR pygmsh
                          }
    #--------------------------------------------------------------------------------------------------------------------------------
    @property
    def template_GR_flags(self):
        """
        CALL: px.make_GR_flags
        """
        base = {'ID' : None, # 1D NP array: actual grain id list                         : INT : id0_grains
                'f01': None, # 1D NP array: specifies grain location                     : STR : bgrain     OR igrain
                'f02': None, # 1D NP array: identifies gbz hoster                        : STR : gbzyes     OR gbzno
                'f03': None, # 1D NP array: identifies twin hoster                       : STR : twinyes    OR twinno
                'f04': None, # 1D NP array: twin-grain orientation relationship type     : STR : ks         OR         ?? OR ??
                'f05': None, # 1D NP array: identifies sub-division hoster               : STR : sdivyes    OR sdivno
                'f06': None, # 1D NP array: type of grain sub-divisions                  : STR : voronoi    OR geometric  OR rolled
                'f07': None, # 1D NP array: identifies if sub-divisions are PriorAusPack : STR : sdivpapyes OR sdivpapno --- Only when f06 is geometric
                'f08': None, # 1D NP array: identify precipitate/particle cluster hoster : STR : pclustyes  OR pclustno
                'f09': None, # 1D NP array: specifies particle cluster distribution      : STR : circ       OR oval       OR gaussian OR uniformingrain
                'f10': None, # 1D NP array: specifies particle shape                     : STR : circ       OR oval       OR circpn (i.e: circle with perlin noise)
                'f11': None, # 1D NP array: empty field (for cracks?)                    : ??? : 
                'f12': None, # 1D NP array: empty field (for cohesive zones?)            : ??? : 
                }
        base1          = {instCount: base.copy()  for instCount in range(PolyXTAL.GSD['N__lev0_i'])} # Key 1 refers to texture (@Level-0) instance number 1. Tex instance keys will have unit increment.
        PolyXTAL.GR_flags = {instCount: base1.copy() for instCount in range(PolyXTAL.GSD['N__lev1_i'])}# Key 1 refers to grain structure (@Level-0) instance number 1. GS instance keys will have unit increment.
    #--------------------------------------------------------------------------------------------------------------------------------
    @property
    def template_GGP2A(self):
        """
        Grain Geometry Parameter - area
        CALL: px.make_GGP2A
        """
        base = {'full' : {'gID'   : None, 'values': None}, # DICTIONARY: id data and geometric area of level 0 grains
                'gbz'  : {'gbzID' : None, 'values': None}, # DICTIONARY: id data and geometric area of grain boundary zones
                'gc'   : {'gbcID' : None, 'values': None}, # DICTIONARY: id data and geometric area of grain cores
                'twin' : {'twinID': None, 'values': None}, # DICTIONARY: id data and geometric area of twin zones
                'pap'  : {'papID' : None, 'values': None}, # DICTIONARY: id data and geometric area of prior-austenitic packets
                'lath' : {'lathID': None, 'values': None}, # DICTIONARY: id data and geometric area of laths
                }
        base1       = {instCount: base.copy()  for instCount in range(PolyXTAL.GSD['N__lev0_i'])} # Key 1 refers to texture (@Level-0) instance number 1. Tex instance keys will have unit increment.
        base2       = {instCount: base1.copy() for instCount in range(PolyXTAL.GSD['N__lev1_i'])} # Key 1 refers to grain structure (@Level-0) instance number 1. GS instance keys will have unit increment.
        PolyXTAL.GGP2A = {instCount: base2.copy() for instCount in range(PolyXTAL.GSD['N__lev2_i'])} # Key 1 refers to grain structure (@Level-0) instance number 1. GS instance keys will have unit increment.
    #--------------------------------------------------------------------------------------------------------------------------------
    @property
    def template_GGP2L(self):
        """
        Grain geometry parameter - length (grain boundary length data).

        Stores all raw data related to grain boundary length. Dictionary key layout::

            all_ungrouped_l0   : id-length pairs for all edges in L0 PolyXTAL
            all_ungrouped_l1   : id-length pairs for all edges in L1 PolyXTAL
            all_ungrouped_l2   : id-length pairs for all edges in L2 PolyXTAL
            all_ungrouped_l3   : id-length pairs for all edges in L3 PolyXTAL (future)
            all_ungrouped_gbz  : id-length pairs for grain-boundary-zone edges
            all_ungrouped_twin : id-length pairs for twin edges
            all_ungrouped_papa : id-length pairs for pap edges
            all_ungrouped_lath : id-length pairs for lath edges
            all_ungrouped_part : id-length pairs for particle edges
            all_grouped_l0_g   : per grainID -> [[edge_ID, length], ...]  (L0)
            all_grouped_l1_g   : per grainID -> [[edge_ID, length], ...]  (L1)
            all_grouped_l2_g   : per grainID -> [[edge_ID, length], ...]  (L2)
            all_grouped_gbz    : per gbzID   -> [[edge_ID, length], ...]
            all_grouped_gebz   : per gbzID   -> [[EXT_edge_ID, length], ...]
            all_grouped_gibz   : per gbzID   -> [[INT_edge_ID, length], ...]
            all_grouped_gc     : per gcID    -> [[edge_ID, length], ...]
            all_grouped_twin   : per twinID  -> [[edge_ID, length], ...]
            all_grouped_pap    : per papID   -> [[edge_ID, length], ...]
            all_grouped_lath   : per lathID  -> [[edge_ID, length], ...]

        CALL: px.make_GGP2L
        """
        base = {'all_ungrouped_l0'  : {'gbeID' : None, 'values': None}, # DICTIONARY: id data and geometric length data of level 0, all grain boundary edges
                'all_ungrouped_gbz' : {'gbeID' : None, 'values': None}, # DICTIONARY: id data and geometric length data of grain boundary zones. 1st None: 1st of gbzID list, 2nd None: 1st of edge length value
                'all_ungrouped_twin': {'gbeID' : None, 'values': None}, # DICTIONARY: id data and geometric length data of grain boundary zones. , all grain boundary edges
                'all_grouped_l0_g'  : {'gID'   : None, 'values': [[None, None],]}, # DICTIONARY: id data and geometric length data of grain boundaries of level 0. 1st None: 1st of gbeID list, 2nd None: 1st of edge length value
                'all_grouped_gbz'   : {'gbzID' : None, 'values': [[None, None],]}, # DICTIONARY: id data and geometric length data of grain boundary zones. 1st None: 1st of gbeID list, 2nd None: 1st of edge length value
                'all_grouped_gebz'  : {'gbzID' : None, 'values': [[None, None],]}, # DICTIONARY: id data and geometric length data of grain boundary zones. 1st None: 1st of gbeID (external) list, 2nd None: 1st of edge length value
                'all_grouped_gibz'  : {'gbzID' : None, 'values': [[None, None],]}, # DICTIONARY: id data and geometric length data of grain boundary zones. 1st None: 1st of gbeID (internal) list, 2nd None: 1st of edge length value
                'all_grouped_gc'    : {'gcID'  : None, 'values': [[None, None],]}, # DICTIONARY: id data and geometric length data of grain core. 1st None: 1st of gbeID list, 2nd None: 1st of edge length value
                'all_grouped_twin'  : {'twinID': None, 'values': [[None, None],]}, # DICTIONARY: id data and geometric length data of twin zones. 1st None: 1st of gbeID list, 2nd None: 1st of edge length value
                'all_grouped_pap'   : {'papID' : None, 'values': [[None, None],]}, # DICTIONARY: id data and geometric length data of pap. 1st None: 1st of gbeID list, 2nd None: 1st of edge length value
                'all_grouped_lath'  : {'lathID': None, 'values': [[None, None],]}, # DICTIONARY: id data and geometric length data of laths. 1st None: 1st of gbeID list, 2nd None: 1st of edge length value
                }
        base1       = {instCount: base.copy()  for instCount in range(PolyXTAL.GSD['N__lev0_i'])} # Key 1 refers to texture (@Level-0) instance number 1. Tex instance keys will have unit increment.
        base2       = {instCount: base1.copy() for instCount in range(PolyXTAL.GSD['N__lev1_i'])} # Key 1 refers to grain structure (@Level-0) instance number 1. GS instance keys will have unit increment.
        PolyXTAL.GGP2L = {instCount: base2.copy() for instCount in range(PolyXTAL.GSD['N__lev2_i'])} # Key 1 refers to grain structure (@Level-0) instance number 1. GS instance keys will have unit increment.
    #--------------------------------------------------------------------------------------------------------------------------------
    @property
    def setPolyXTAL(self):
        """Setpolyxtal."""
        self.setMD
        # self.setMPD  # Not implemented/defined in codebase
        self.setGSD
        self.setPXID
        self.setOrigin()
        self.setLengths()
        self.setDomain
        self.setupL0GS
        self.setupL1GS
        self.setupL2GS
    #------------------------------------------------------------------------------------------------------------------------------------------
    # Setter definitions
    @property
    def get_gsd(self):      
        """Return ``gsd``."""
        return PolyXTAL.GSD
    @property
    def get_id0_base(self): 
        """Return ``id0_base``."""
        return PolyXTAL.ID0_base
    @property
    def get_id0_pair(self): 
        """Return ``id0_pair``."""
        return PolyXTAL.ID0_pair
    @property
    def get_id1_base(self): 
        """Return ``id1_base``."""
        return PolyXTAL.ID1_base
    @property
    def get_id1_pair(self): 
        """Return ``id1_pair``."""
        return PolyXTAL.ID1_pair
    @property
    def get_id_ctex(self):  
        """Return ``id_ctex``."""
        return PolyXTAL.ID_ctex
    @property
    def get_flags_px(self): 
        """Return ``flags_px``."""
        return PolyXTAL.PX_flags
    @property
    def get_flags_gr(self): 
        """Return ``flags_gr``."""
        return PolyXTAL.GR_flags
    @property
    def get_ggp2a(self):    
        """Return ``ggp2a``."""
        return PolyXTAL.GGP2A
    @property
    def get_ggp2l(self):    
        """Return ``ggp2l``."""
        return PolyXTAL.GGP2L
    #--------------------------------------------------------------------------
    # From PolyXTAL.GSD
    @property
    def get_dimensionality(self): 
        """Return ``dimensionality``."""
        return PolyXTAL.GSD['dimen']
    @property
    def get_MorphGenTech(self): 
        """Return ``MorphGenTech``."""
        return PolyXTAL.GSD['GSMorph_gentech']
    @property
    def get_ni_level0(self): 
        """Return ``ni_level0``."""
        return PolyXTAL.GSD['N__lev0_i']
    @property
    def get_ni_level1(self): 
        """Return ``ni_level1``."""
        return PolyXTAL.GSD['N__lev1_i']
    @property
    def get_ni_level2(self): 
        """Return ``ni_level2``."""
        return PolyXTAL.GSD['N__lev2_i']
    @property
    def get_ni_texos(self): 
        """Return ``ni_texos``."""
        return PolyXTAL.GSD['N__tex_i_os']
    @property
    def get_ni_texps(self): 
        """Return ``ni_texps``."""
        return PolyXTAL.GSD['N__tex_i_ps']
    #--------------------------------------------------------------------------
    # From PolyXTAL.ID0_base
    # ni0: number of instances of level 0 grain structure
    @property
    def get_id0_g(self, ni0): 
        """Return ``id0_g``."""
        return PolyXTAL.ID0_base[ni0]['GRAIN']
    @property
    def get_id0_s(self, ni0): 
        """Return ``id0_s``."""
        return PolyXTAL.ID0_base[ni0]['SURF']
    @property
    def get_id0_e(self, ni0): 
        """Return ``id0_e``."""
        return PolyXTAL.ID0_base[ni0]['EDGE']
    @property
    def get_id0_j(self, ni0): 
        """Return ``id0_j``."""
        return PolyXTAL.ID0_base[ni0]['JPOINT']
    #--------------------------------------------------------------------------
    # From: PolyXTAL.ID0_pair
    @property 
    def get_id0_gs(self, ni0): 
        """Return ``id0_gs``."""
        return PolyXTAL.ID0_pair[ni0]['GRAIN_SURF']
    @property 
    def get_id0_ge(self, ni0): 
        """Return ``id0_ge``."""
        return PolyXTAL.ID0_pair[ni0]['GRAIN_EDGE']
    @property 
    def get_id0_gj(self, ni0): 
        """Return ``id0_gj``."""
        return PolyXTAL.ID0_pair[ni0]['GRAIN_JPOINT']
    @property 
    def get_id0_sg(self, ni0): 
        """Return ``id0_sg``."""
        return PolyXTAL.ID0_pair[ni0]['SURF_GRAIN']
    @property 
    def get_id0_eg(self, ni0): 
        """Return ``id0_eg``."""
        return PolyXTAL.ID0_pair[ni0]['EDGE_GRAIN']
    @property 
    def get_id0_jg(self, ni0): 
        """Return ``id0_jg``."""
        return PolyXTAL.ID0_pair[ni0]['JPOINT_GRAIN']
    @property 
    def get_id0_se(self, ni0): 
        """Return ``id0_se``."""
        return PolyXTAL.ID0_pair[ni0]['SURF_EDGE']
    @property
    def get_id0_sj(self, ni0): 
        """Return ``id0_sj``."""
        return PolyXTAL.ID0_pair[ni0]['SURF_JPOINT']
    @property
    def get_id0_es(self, ni0): 
        """Return ``id0_es``."""
        return PolyXTAL.ID0_pair[ni0]['EDGE_SURF']
    @property
    def get_id0_js(self, ni0): 
        """Return ``id0_js``."""
        return PolyXTAL.ID0_pair[ni0]['JPOINT_SURF']
    @property 
    def get_id0_ej(self, ni0): 
        """Return ``id0_ej``."""
        return PolyXTAL.ID0_pair[ni0]['EDGE_JPOINT']
    @property 
    def get_id0_je(self, ni0): 
        """Return ``id0_je``."""
        return PolyXTAL.ID0_pair[ni0]['JPOINT_EDGE']
    #--------------------------------------------------------------------------
    # From PolyXTAL.ID1_base
    @property 
    def get_gbz(self, ni0, ni1): 
        """Return ``gbz``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gbz']
    @property 
    def get_gbz_ebe(self, ni0, ni1): 
        """Return ``gbz_ebe``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gbz_eb_ed']
    @property 
    def get_gbz_ibe(self, ni0, ni1): 
        """Return ``gbz_ibe``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gbz_ib_ed']
    @property 
    def get_gbz_ebj(self, ni0, ni1): 
        """Return ``gbz_ebj``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gbz_eb_jp']
    @property 
    def get_gbz_ibj(self, ni0, ni1): 
        """Return ``gbz_ibj``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gbz_ib_jp']
    @property
    def get_gc(self, ni0, ni1): 
        """Return ``gc``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gc']
    @property 
    def get_gc_ebe(self, ni0, ni1): 
        """Return ``gc_ebe``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gc_eb_ed']
    @property 
    def get_gc_ibe(self, ni0, ni1): 
        """Return ``gc_ibe``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gc_ib_ed']
    @property 
    def get_gc_ebj(self, ni0, ni1): 
        """Return ``gc_ebj``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gc_eb_jp']
    @property
    def get_gc_ibj(self, ni0, ni1): 
        """Return ``gc_ibj``."""
        return PolyXTAL.ID1_base[ni0][ni1]['gc_ib_jp']
    @property
    def get_twin(self, ni0, ni1): 
        """Return ``twin``."""
        return PolyXTAL.ID1_base[ni0][ni1]['twin']
    @property
    def get_twinbed(self, ni0, ni1): 
        """Return ``twinbed``."""
        return PolyXTAL.ID1_base[ni0][ni1]['twin_bed']
    @property 
    def get_twinbjp(self, ni0, ni1): 
        """Return ``twinbjp``."""
        return PolyXTAL.ID1_base[ni0][ni1]['twin_bjp']
    @property 
    def get_apckt(self, ni0, ni1): 
        """Return ``apckt``."""
        return PolyXTAL.ID1_base[ni0][ni1]['apckt']
    @property 
    def get_apcktbe(self, ni0, ni1): 
        """Return ``apcktbe``."""
        return PolyXTAL.ID1_base[ni0][ni1]['apckt_be']
    @property 
    def get_apcktbjp(self, ni0, ni1):  
        """Return ``apcktbjp``."""
        return PolyXTAL.ID1_base[ni0][ni1]['apckt_bjp']
    # From PolyXTAL.ID1_pair
    
    #---------------------------------------------------dir---------------------------------------------------------------------------------------
    def GSvisualise(self):
        """Gsvisualise."""
        raise NotImplementedError("GSvisualise is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------
    def pf(self):
        """Pf."""
        raise NotImplementedError("pf is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------
    def ipf(self):
        """Ipf."""
        raise NotImplementedError("ipf is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------
    def odfSec(self):
        """Odfsec."""
        raise NotImplementedError("odfSec is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------
    def ipfMap(self):
        """Ipfmap."""
        raise NotImplementedError("ipfMap is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------
    def makeFillerDataCTF(self):
        """Makefillerdatactf."""
        raise NotImplementedError("makeFillerDataCTF is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------
    def writeCTF(self):
        """Writectf."""
        raise NotImplementedError("writeCTF is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------
    def genFEMesh(self):
        """Genfemesh."""
        raise NotImplementedError("genFEMesh is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------
    def writeINP(self):
        """Writeinp."""
        raise NotImplementedError("writeINP is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------
    def schmidTensor(self):
        """Schmidtensor."""
        raise NotImplementedError("schmidTensor is not yet implemented.")
    #------------------------------------------------------------------------------------------------------------------------------------------