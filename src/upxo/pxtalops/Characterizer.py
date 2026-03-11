from upxo._sup import dataTypeHandlers as dth
from skimage.measure import label as skim_label
import numpy as np
import cv2
import itertools
import matplotlib.pyplot as plt

NUMBERS, ITERABLES = dth.dt.NUMBERS, dth.dt.ITERABLES

class mcgs_mchar_2d():
    """
    Root1: ..\\src\\upxo\\scripts\\MCGS2d_characterization

    Example-1
    ---------
    This example:
        a. builds a synthetic 2D grain structure
        b. extracts subsets from a temporal slice
        c. characterizes the grain structure subsets

    Refer to Root1\\subset_characterization_1.py
    """

    __slots__ = ('xgr', 'ygr',
                 'fmat', 'fmin', 'fmax',
                 'lgi',
                 'ngrains')

    def __init__(self, xgr=None, ygr=None, mcstates=None):
        self.xgr, self.ygr = xgr, ygr

    def __repr__(self):
        return 'Characterized pxt.'

    def set_fmat(self, fmat, fmin, fmax):
        self.fmat = fmat
        self.fmin = fmin
        self.fmax = fmax

    def make_fmat_subsets(self, hsize=None, vsize=None):
        get_subsets = np.lib.stride_tricks.sliding_window_view
        fmats = get_subsets(self.fmat, (hsize, vsize))[::hsize, ::vsize]
        return fmats

    def find_grains(self,
                    library='opencv',
                    fmat=None,  # Field matrix: state values
                    kernel_order=2):
        if library in dth.opt.ocv_options:
            # Acceptable values for opencv: 4, 8
            if kernel_order in (4, 8):
                KO = kernel_order
            elif kernel_order in (1, 2):
                KO = 4*kernel_order
            else:
                raise ValueError("Input must be in (1, 2, 4, 8)."
                                 f" Recieved {kernel_order}")
        elif library in dth.opt.ski_options:
            # Acceptable values for opencv: 1, 2
            if kernel_order in (4, 8):
                KO = int(kernel_order/4)
            elif kernel_order in (1, 2):
                KO = kernel_order
            else:
                raise ValueError("Input must be in (1, 2, 4, 8)."
                                 f" Recieved {kernel_order}")
        # -----------------------------------------
        # Detect grains and store necessary data
        _S_ = fmat
        # fmat = pxt.gs[8].s
        q_values = np.arange(self.fmin, self.fmax+1)
        nq = q_values.size
        s_gid = {i: None for i in q_values}
        s_n = [i for i in q_values]
        for i, _s_ in enumerate(np.unique(_S_)):
            # -----------------------------------------
            # Identify the grains belonging to this state
            BI = (_S_ == _s_).astype(np.uint8)  # Binary image
            if library in dth.opt.ocv_options:
                _, labels = cv2.connectedComponents(BI*255,
                                                              connectivity=KO)
            elif library in dth.opt.ski_options:
                labels, _ = skim_label(BI, return_num=True,
                                                 connectivity=KO)
            # -----------------------------------------
            if i == 0:
                lgi = labels
            else:
                labels[labels > 0] += lgi.max()
                lgi = lgi + labels
            # -----------------------------------------
            s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))
            s_n[_s_-1] = len(s_gid[_s_])
            print(f"MC state = {_s_}:  Num grains = {s_n[_s_-1]}")

        # Get the total number of grains
        Ngrains = np.unique(lgi).size
        # Generate and store the gid-s mapping
        gid = list(range(1, Ngrains+1))

        _gid_s_ = []
        for _gs_, _gid_ in s_gid.items():
            if _gid_:
                for __gid__ in _gid_:
                    _gid_s_.append(_gs_)
            else:
                _gid_s_.append(0)
        gid_s = _gid_s_

        # Make the output string to print on promnt
        print(f"No. of grains detected = {Ngrains}")
        gid_npxl = [np.count_nonzero(lgi == _gid_) for _gid_ in gid]
        char_gs = {'lgi': lgi,
                   'Ngrains': Ngrains,
                   'gid': gid,
                   'gid_s': gid_s,
                   's_gid': s_gid,
                   's_n': s_n,
                   'gid_npxl': gid_npxl}
        return char_gs

    def characterize_all_subsets(self, fmats):
        characterized_subsets_all = [[None for h in range(fmats.shape[0])] for v in range(fmats.shape[1])]
        for v in range(fmats.shape[1]):
            for h in range(fmats.shape[0]):
                print(40*'-')
                print(f'Characterizing subset: ({h}, {v})')
                print('')
                char_gs = self.find_grains(library='opencv', fmat=fmats[h][v])
                characterized_subsets_all[h][v] = char_gs
        return characterized_subsets_all

    def characterize_subsets(self, fmats, alongh, alongv):
        if type(alongh) not in ITERABLES+NUMBERS:
            raise TypeError('Invalid alongh type')
        if type(alongv) not in ITERABLES+NUMBERS:
            raise TypeError('Invalid alongv type')
        if type(alongh) not in ITERABLES:
            alongh = [alongh]
        if type(alongv) not in ITERABLES:
            alongv = [alongv]
        # --------------------------------------
        characterized_subsets_indices = [hv
                                         for hv in itertools.product(alongh,
                                                                     alongv)]
        characterized_subsets = [[None for v in alongv] for h in alongh]
        for v in alongv:
            for h in alongh:
                char_gs = self.find_grains(library='opencv', fmat=fmats[h][v])
                characterized_subsets[h][v] = char_gs
        return characterized_subsets_indices, characterized_subsets
