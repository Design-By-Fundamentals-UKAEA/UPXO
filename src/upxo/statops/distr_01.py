import numpy as np
from dataclasses import dataclass
from collections import deque
from scipy import stats
import matplotlib.pyplot as plt
from colorama import init as colorama_init
from colorama import Fore, Back, Style
from upxo._sup.console_formats import console_seperator
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class distribution():
    '''
    RULES:
        An instance should not store than more one data
        If data is updated, then update operations must be perfoemed
    VARIABLES:
        data_name: Data Name
        data: Data
        nbins: Number of bins
        hist: histograms
        bin_edges: edges of the bins
    CONVENTIONS:
        nbins: contained in a list
        H: contained in a list
        bin_edges: contained in a list
    '''
    def __init__(self,
                 data_name = None,
                 data = None,
                 nbins = [None],
                 be_estimator = 'auto',
                 ):
        colorama_init()
        #*******************************
        if isinstance(data_name, list):
            data_name = np.array(data_name)
        #from distr_01 import KDE
        self.S = SUMMARY
        #*******************************
        #from distr_01 import KDE
        self.K = KDE
        #*******************************
        #from distr_01 import HISTOGRAM
        self.H = HISTOGRAM
        self.H.data = data
        #*******************************
        self.data_name = data_name
        self.data = data
        #*******************************
        self.calc_histogram(be_estimator = be_estimator)
        #*******************************
        self.update_summary()
        #*******************************

        console_seperator(seperator = '-*', repetitions = 25)

        #*******************************
    #-----------------------------------------
    def update_summary(self):
        self.find_min()
        self.find_mean()
        self.find_median(axis = None)
        self.find_max()
        self.find_total()
        self.find_variance()
        self.find_skewness()
        self.find_kurtosis()
        self.find_std_dev(axis = 0)
        self.find_variance(limits = None,
                           inclusive = (True, True),
                           axis = 0
                           )
        self.find_percentiles(percentile_list = [0, 10, 50, 90, 100],
                              throw_format = 'list',
                              see = False
                              )
    #-----------------------------------------
    def find_min(self):
        self.S.minimum = self.data.min()
    def find_mean(self):
        self.S.mean = self.data.mean()
    def find_median(self,
                    axis = None
                    ):
        self.S.median = np.median(a = self.data,
                                  axis = axis
                                  )
    def find_max(self):
        self.S.maximum = self.data.max()
    def find_total(self):
        self.S.total = self.data.sum()
    def find_std_dev(self,
                     axis = 0):
        self.S.sdev = self.data.std()
    def find_skewness(self):
        self.S.skew = stats.skew(self.data, bias = True)
    def find_kurtosis(self):
        self.S.kurt = stats.kurtosis(self.data, bias = True)
    def find_variance(self,
                      limits = None,
                      inclusive = (True, True),
                      axis = 0
                      ):
        '''
        REF: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tvar.html#scipy.stats.tvar
        '''
        if limits == None:
            self.S.variance = stats.tvar(self.data
                                    )
        else:
            self.S.variance = stats.tvar(a = self.data,
                                         limits = limits,
                                         inclusive = inclusive
                                         )
    def find_percentiles(self,
                         percentile_list = [0, 10, 50, 90, 100],
                         throw_format = 'list',
                         see = False
                         ):
        _ = [np.percentile(self.data, _p) for _p in percentile_list]
        if throw_format == 'dict':
            self.S.percentiles = dict(zip(percentile_list, _))
        elif throw_format == 'list':
            self.S.percentiles = _
        if see:
            print(_)
    #-----------------------------------------
    def calc_histogram(self,
                       be_estimator = 'auto'
                       ):
        '''
        "be_estimator" options:
            1. 'auto'
            2. 'fd' (Freedman Diaconis Estimator)
            3. 'doane'
            4. For more, refer: https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
        '''
        self.H.be = np.histogram_bin_edges(self.data, bins = be_estimator)
        self.H.hv, _ = np.histogram(self.data, bins = self.H.be)
    #-----------------------------------------
    def calc_rv_histogram(self):
        # Resample from existing histogram to yield a similar histogram
        pass
    #-----------------------------------------
    def plot_histogram(self,
                       be_estimator = 'auto'):
        # First calculate the histogram
        self.calc_histogram(be_estimator = be_estimator)
        # Then plot using "plt.bar"
        _be = self.H.be[:-1]
        _hv = self.H.hv
        plt.bar(_be,
                _hv,
                width = _be.min()/2,
                facecolor = 'gray',
                edgecolor = 'black',
                linewidth = 1)
        plt.xlabel(self.data_name)
        plt.ylabel('Count')
    #-----------------------------------------
    #-----------------------------------------
    #-----------------------------------------
    #-----------------------------------------
###############################################################################
@dataclass(repr = False, frozen = True)
class SUMMARY():
    minimum = None
    percentiles = None
    maximum = None
    total = None
    mean = None
    median = None
    variance = None
    skew = None
    kurt = None
# .. .. .. .. .. .. .. .. .. ..
@dataclass(repr = False)
class KDE():
    bw = None
    kd = None
# .. .. .. .. .. .. .. .. .. ..
@dataclass(repr = False)
class HISTOGRAM():
    hv = None # histogram values
    be = None # Bin edges
    data = None
    nbins = None
