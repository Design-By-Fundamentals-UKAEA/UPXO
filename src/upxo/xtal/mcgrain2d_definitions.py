import numpy as np
import matplotlib.pyplot as plt
# import cv2
# from skimage.measure import label as skim_label
# from defdap.quat import Quat
import upxo._sup.gops as gops
import upxo._sup.dataTypeHandlers as dth
from upxo._sup.validation_values import _validation as val
# from upxo.xtalphy.orientation import grainoris as go

class grain2d():
    """
    pxt.gs[2].g[1]['grain']
    ------------------------------
    ATTRIBUTES AND THEIR ACCESS
    ------------------------------
    loc:
        Grain pixel loca5tions.
        pxt.gs[t].g[gid]['grain'].loc, pxt.gs[t].g[gid]['grain'].coords

    npixels:
        Number of pixels in the grain
        pxt.gs[t].g[gid]['grain'].npixels

    position:
        Grain relative poistioning in the poly-crystal.
        pxt.gs[t].g[gid]['grain'].position
        initial 2 values provide x and y coord of the centroid
        last string value provides the relative positioning of the grain

    coords:
        Explanation here

    gbloc:
        grain boundary pixel locations

    bbox_bounds:
        Bounds of the bounding box
        pxt.gs[t].g[gid]['grain'].bbox_bounds

    bbox_ex_bounds:
        Bounds of the extended bounding box. Extended by unit pixels on all
        sides. If grain is a boundary grain, only possible directions  will
        be extended by unit pixel.
        pxt.gs[t].g[gid]['grain'].bbox_ex_bounds
    bbox:
        Mask on bounding box.
        pxt.gs[t].g[gid]['grain'].bbox
    bbox_ex:
        Mask on extended bouning box.
        pxt.gs[t].g[gid]['grain'].bbox_ex

    skprop:
        Scikit image property generator
        pxt.gs[t].g[gid]['grain'].skprop

    _px_area:
        Area of a single pixel
        pxt.gs[t].g[gid]['grain']._px_area

    gid:
        This grain ID. Same as the lgi number of this grain.
        pxt.gs[t].g[gid]['grain'].gid

    gind:
        Indices of grain pixel in the parent state matrix
        pxt.gs[t].g[gid]['grain'].gind

    gbid:
        Grain boundary ID
        pxt.gs[t].g[gid]['grain'].gbid

    gbind:
        Indices of grain boundary pixel in the parent state matrix
        pxt.gs[t].g[gid]['grain'].gbind

    gbvert:
        Grain boundary vertices
        pxt.gs[t].g[gid]['grain'].gbvert

    gbsegs:
        grain boundary segments
        pxt.gs[t].g[gid]['grain'].gbsegs

    gbsegs_geo:
        grain boundary segments: dict with below keys:
        'info': OPTIONS:
            'spg.menp': Single Pixel Grain. Multi-edge not possible
            'slg.menp': Straight line Grain. Multi-edge not possible
        'me': Multi-edge
        pxt.gs[t].g[gid]['grain'].gbsegs_geo

    s:
        Monte-Car;o State value of the current grain
        pxt.gs[t].g[gid]['grain'].s

    sn:
        pxt.gs[t].g[gid]['grain'].sn

    neigh:
        GIDs of the neighbouring grains
        pxt.gs[t].g[gid]['grain'].neigh

    grain_core:
        Grain core region
        pxt.gs[t].g[gid]['grain'].grain_core

    gb_zone:
        Grain boundary zone
        pxt.gs[t].g[gid]['grain'].gb_zone

    subgrains:
        sub-grains inside grain. It shall also encompass any
        island_grains
        pxt.gs[t].g[gid]['grain'].subgrains

    paps:
        prior-austenite packats
        pxt.gs[t].g[gid]['grain'].paps

    blocks:
        block structuers in paps
        pxt.gs[t].g[gid]['grain'].blocks

    laths:
        Lath structures inside blocks
        pxt.gs[t].g[gid]['grain'].laths

    xstruc: Crystal Structure: 'fcc', 'bcc', 'hcp'

    xmin, xmax, ymin, ymax:
        pxt.gs[t].g[gid]['grain'].(xmin, xmax, ymin, ymax)

    control_points_mesh:
        Points places at strategic locations inside the grain to control
        mesh density of a conformal mesh
        pxt.gs[t].g[gid]['grain'].control_points_mesh

    ea_pixels:
        Euler snagles of all pixels

    quats_pixels:
        Quaternions of all the pixesl int he grain

    ref_quat:
        Reference quaternion value of the all pixes in the grain

    ref_ea:
        Referecne Euler angle of all the pixesl in the grain

    texcomp:
        T3exture component name to which the pixel would belong to.
        Defined at each of the pixesl in the grain

    glb_pert_min_ea1,  glb_pert_max_ea1:
        Global minimum and maximum allowed perturbation to euler angle 1
    glb_pert_min_ea2,  glb_pert_max_ea2:
        Global minimum and maximum allowed perturbation to euler angle 2
    glb_pert_min_ea3,  glb_pert_max_ea3
        Global minimum and maximum allowed perturbation to euler angle 3
    lcl_pert_min_ea1, lcl_pert_max_ea1:
        Local minimum and maximum allowed perturbation to euler angle 1
    lcl_pert_min_ea2, lcl_pert_max_ea2:
        Local minimum and maximum allowed perturbation to euler angle 2
    lcl_pert_min_ea3, lcl_pert_max_ea3:
        Local minimum and maximum allowed perturbation to euler angle 3
    --------------------------------------------------------------------------
    _xgr_min_: Minimum value of the xgr of the parent grain structure
    _xgr_max_: Maximum value of the xgr of the parent grain structure
    _xgr_incr_: Increment value of the xgr of the parent grain structure

    _ygr_min_: Minimum value of the ygr of the parent grain structure
    _ygr_max_: Maximum value of the ygr of the parent grain structure
    _xgr_incr_: Increment value of the ygr of the parent grain structure
    """

    __slots__ = ('loc', 'npixels', 'position', 'coords', 'gbloc', 'brec', 'm',
                 'bbox_bounds', 'bbox_ex_bounds', 'bbox', 'bbox_ex', 'skprop',
                 '_px_area', 'gid', 'gind', 'gbid', 'gbind', 'gbvert',
                 'gbsegs', 'gbsegs_geo', 's', 'sn', 'neigh',
                 'grain_core', 'gb_zone', 'xmin', 'xmax', 'ymin', 'ymax',
                 'loctree', 'coordtree', 'ea', 'xgid', 'bbox_bz', 'bbox_core',
                 '_lfi_gbseg_empties_'
                 )

    __get_item_behaviour = 'locs_away_from_centroid'
    rtol = 1e-6

    def __init__(self):
        # Set position/location related slots
        self.loc, self.position = None, None
        self.coords, self.gbloc = None, None
        # set bounds related
        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None
        self.brec, self.bbox_bounds, self.bbox_ex_bounds = None, None, None
        self.loctree, self.coordtree = None, None
        # set masks
        self.bbox, self.bbox_ex = None, None
        # Set local neighbourhood related slots
        self.neigh = None
        # set properties
        self.npixels, self._px_area, self.skprop = None, None, None
        # set state related slots
        self.s, self.sn = None, None
        # set grain indices related slots
        self.gid, self.gind = None, None
        # Set grain boundary indices related slots
        self.gbid, self.gbind = None, None
        # Set grain boundaryt points
        # self.gbvert = None
        # Set grain bounadryu segfments
        # self.gbsegs, self.gbsegs_geo = None, None
        # FEATURES
        self.grain_core, self.gb_zone = None, None
        # ------------------------------------------
        # ORIENTATION RELATED
        self.ea  = None
        # -----------------------------------------------------------------------
        '''
        lfi_gbseg_empties: Default values of non-grain boundary segment coordinates
        inside the bounding box. You may change this value np.nan if necessary. 
        refer to demos/neighOps/neighOps-1.ipynb for usage example. use the 
        appropriate setter and getter methods to operate on this attribute.
        '''
        self._lfi_gbseg_empties_ = 0
        # -----------------------------------------------------------------------

    #def __repr__(self):
    #    _repr_ = f'UPXO [{self.position[2]}] grain2d. GID:{self.gid}'
    #    _repr_ += f'-S:{self.s}'
    #    _repr_ += f'-Sn:{self.sn}'
    #    _repr_ += f'-Centroid:[{self.position[0]:.4f},{self.position[1]:.4f}]'
    #    return _repr_

    def __len__(self):
        return len(self.loc)

    @property
    def lfi_gbseg_empties(self):
        return self._lfi_gbseg_empties_

    @lfi_gbseg_empties.setter
    def lfi_gbseg_empties(self, value):
        self._lfi_gbseg_empties_ = value

    @lfi_gbseg_empties.deleter
    def lfi_gbseg_empties(self):
        del self._lfi_gbseg_empties_

    @val.DEC_validate_samples
    def __eq__(self, samples=None, types=None):
        '''
        Allowed input datatypes:
            A number
            A UPXO grain2D object
            pxt.gs[tslice].xomap.map.grainList[0].__class__.__name__
        '''
        # VAL: See if sampls are numbers
        if list(types)[0] in dth.dt.NUMBERS:
            cmp = [self.npixels == _ for _ in samples]
        elif samples[0].__class__.__name__ == 'grain2d':  # Testing 1 is enough
            '''UPXO grain object'''
            cmp = [self.npixels == _.npixels for _ in samples]
        elif samples[0].__class__.__name__ == 'Grain':  # Testing 1 is enough
            '''DefDap grain object'''
            cmp = [self.npixels == len(_.coordList) for _ in samples]
        return cmp

    def __ne__(self, samples=None):
        return [not _ for _ in self.__eq__(samples=samples)]

    @val.DEC_validate_samples
    def __lt__(self, samples=None, types=None):
        '''
        Allowed input datatypes: number, A UPXO grain2D object
            pxt.gs[tslice].xomap.map.grainList[0].__class__.__name__
        '''
        # VAL: See if samples are numbers
        if list(types)[0] in dth.dt.NUMBERS:
            cmp = [self.npixels < _ for _ in samples]
        elif samples[0].__class__.__name__ == 'grain2d':  # Testing 1 is enough
            '''UPXO grain object'''
            cmp = [self.npixels < _.npixels for _ in samples]
        elif samples[0].__class__.__name__ == 'Grain':  # Testing 1 is enough
            '''DefDap grain object'''
            cmp = [self.npixels < len(_.coordList) for _ in samples]
        return cmp

    @val.DEC_validate_samples
    def __gt__(self, samples=None, types=None):
        '''
        Allowed input datatypes: number, A UPXO grain2D object
            pxt.gs[tslice].xomap.map.grainList[0].__class__.__name__
        '''
        # VAL: See if samples are numbers
        if list(types)[0] in dth.dt.NUMBERS:
            cmp = [self.npixels > _ for _ in samples]
        elif samples[0].__class__.__name__ == 'grain2d':  # Testing 1 is enough
            '''UPXO grain object'''
            cmp = [self.npixels > _.npixels for _ in samples]
        elif samples[0].__class__.__name__ == 'Grain':  # Testing 1 is enough
            '''DefDap grain object'''
            cmp = [self.npixels > len(_.coordList) for _ in samples]
        return cmp

    def __le__(self, samples=None, types=None):
        return [lt or eq for lt, eq in zip(self.__lt__(samples=samples),
                                           self.__eq__(samples=samples))]

    def __ge__(self, samples=None, types=None):
        return [gt or eq for gt, eq in zip(self.__gt__(samples=samples),
                                           self.__eq__(samples=samples))]

    def __getitem__(self, keys):
        """
        Provides the location subset of all
        keys --> start_percentage : end_percentage
        """
        pass

    def __mul__(self, k):
        # INCLUDE VALIDATIONS
        self._px_area *= k

    def __att__(self):
        return gops.att(self)

    def __iter__(self):
        return iter(self.loc)

    def __contains__(self, points):
        '''
        Currently accepted points data structure
        points = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
        '''
        if type(points[0]) != list:
            raise TypeError(f"Points should be a list of lists. Currently provided {type(points)}")
        if self.loctree is None:
            self.make_loctree()
        contained = [None]*len(points[0])
        for i, xy in enumerate(zip(points[0], points[1])):
            dist, idx = self.loctree.query(xy, k=1)
            contained[i] = dist <= self.rtol
        return contained

    @property
    def lfi_gbsegs(self):
        '''Convert the sparse storage into a full grain boundary segment lfi array.'''
        # Convert sparse storage into full gbseg lfi array
        _gbseg_ = np.zeros(np.prod(self.gbsegs['shape']), dtype=self.gbsegs['dtype'])
        _gbseg_[self.gbsegs['NZI']] = self.gbsegs['NZV']
        _gbseg_ = np.reshape(_gbseg_, self.gbsegs['shape'])
        # Handle empty values
        condition1 = np.isnan(self.lfi_gbseg_empties)
        condition2 = isinstance(self.lfi_gbseg_empties, np.floating)
        condition3 = isinstance(self.lfi_gbseg_empties, float)
        if condition1 or condition2 or condition3:
            _gbseg_ = np.asarray(_gbseg_, dtype=np.float32)
        _gbseg_[_gbseg_ == 0] = self.lfi_gbseg_empties
        return _gbseg_

    def set_quat(self):
        self.quats_pixels
        self.ref_quat

    def find_misori(self, angles):
        pass

    def set_glb_ea_pert(self, pert_ea1, pert_ea2, pert_ea3):
        self.glb_pert_min_ea1, self.glb_pert_max_ea1 = pert_ea1[0], pert_ea1[1]
        self.glb_pert_min_ea2, self.glb_pert_max_ea2 = pert_ea2[0], pert_ea2[1]
        self.glb_pert_min_ea3, self.glb_pert_max_ea3 = pert_ea3[0], pert_ea3[1]

    def make_loctree(self):
        from scipy.spatial import cKDTree as ckdt
        self.loctree = ckdt(self.loc, copy_data=False, balanced_tree=True)

    @property
    def mean_ea(self):
        # return the mean orientatipon: euler angle in degrees
        pass

    def make_coordtree(self):
        from scipy.spatial import cKDTree as ckdt
        self.loctree = ckdt(self.coords, copy_data=False, balanced_tree=True)

    def set_skprop(self):
        from skimage.measure import regionprops as skim_regionprops
        self.make_prop(skim_regionprops, skprop=True)

    def make_prop(self, generator, skprop=True):
        if skprop:
            #print('=======================')
            #print(generator)
            #print('=======================')
            self.skprop = generator(self.bbox_ex, cache=False)[0]

    @property
    def centroid(self):
        coords = self.coords.T
        return (coords[0].mean(), coords[1].mean())

    def plot(self, hold_on=False):
        if not hold_on:
            plt.figure()
        plt.imshow(self.bbox_ex)
        plt.title(f"Grain plot \n Grain: {self.gid}. Area: {round(self.skprop.area*100)/100} mu m^2")
        if not hold_on:
            plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
            plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
            plt.show()

    def plotgb(self, hold_on=False):
        z = np.zeros_like(self.bbox_ex)
        rmin = self.bbox_ex_bounds[0]
        cmin = self.bbox_ex_bounds[2]
        for rc in self.gbloc:
            z[rc[0]-rmin, rc[1]-cmin] = 1
        # ------------------------------------
        if not hold_on:
            plt.figure()
        plt.imshow(z)
        if not hold_on:
            plt.title(f"Grain boundary plot \n Grain: {self.gid}. Area: {round(self.skprop.area*100)/100} mu m^2. Perimeter: {round(self.skprop.perimeter*10000)/10000} mu m")
            plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
            plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
            plt.show()

    def plotgbseg(self, hold_on=False):
        if not hold_on:
            plt.figure()
        plt.imshow(self.gbsegs)
        if not hold_on:
            plt.title(f"Grain boundary segment plot \n Grain: {self.gid}. Area: {round(self.skprop.area*100)/100} mu m^2. Perimeter: {round(self.skprop.perimeter*10000)/10000} mu m")
            plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
            plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
            plt.show()
