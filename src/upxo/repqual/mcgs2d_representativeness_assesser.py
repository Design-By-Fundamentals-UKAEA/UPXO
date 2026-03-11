from .._sup import dataTypeHandlers as dth
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.spatial.distance import jensenshannon

class mc2repr():
    """
    Representativeness qualificartion
    ----------------------------------
    target_type: str
        Source of targer data. Options:
            1. ebsd0 - un-processed 2D EBSD map: DefDAP object.
            2. ebsd1 - processed DefDAP data. Remapped with avg. ori.
            3. umc2 - UPXO Monte-Carlo Grain structure 2D.
            4. umc3 - UPXO Monte-Carlo Grain structure 3D.
            5. uvt2 - UPXO Voronoi-Tessellation Grain Structure 2D.
            6. stats - Data samples across grain morphology par. Needs xori.
                       Could be in the form of dictionary or panadas dataframe.
                       If dict or pandas dataframe, key or column name
                       respectively, must be name of the parameter.
                       Examples of parameter names include:
                           1. area, perimeter
                           2. aspecrt ratio, morphologhical orientation
    ----------------------------------
    target: object
        Target grain structure data. Details:
            1. `MCGS.gs[tslice]` for umc2 and umc3
            2. `VTGS` for uvt2
            3. ddap_ebsd - for un-processed or processed DefDAP data
    ----------------------------------
    samples: dict
        Samples to match against the target.
        Keys should be sample_names
        Values should contain either:
            grain structure objects, or
            flag-string, 'make'
        If a value is a grain strucutre object, then it will be used as
        samples. It can be of types (a) umc2, (b) umc3 and (c) uvt2
        If a value is 'make', then the following will be performanceormed:
            1. read the excel file for grain structure generation parameters
            2. simulate the grain structure evolution
            3. Pull out specified slices at specified temporal slice intervals
            4. Characterize the temporal slices
    ----------------------------------
    par_bounds: dict
        DESCRIPTION:
            For each parameter in the key, value must be a list of:
                [match bounds for peak locations in percentage,
                 match bounds for peak location density in percentage,
                 J-S test bounds
                 ]
        KEYS:
            area, perimeter, aspect ratio
        VALUES:
            bounds: [ [5, 5], [5, 5], [0.1, 0.1]]
    ----------------------------------
    metrics: list
        DESCRIPTION:
            List of metrics to use to enable representativeness qualification
            Examples include:
                1. modes_n
                2. modes_loc
                3. modes_width
                4. distr_type
                5. skewness
                6. kurtosis
    ----------------------------------
    kde_options: dict
        DESCRIPTION:
            key: bw_method
            value: choose from 'scott', 'silverman' or a scalar value
    ----------------------------------
    """
    __slots__ = ('target_type',
                 'target',
                 'samples',
                 'par_bounds',
                 'metrics',
                 'kde_options',
                 'stat_tests',
                 'test_threshold',
                 'stest',
                 'test_metrics',
                 'parameters',
                 'distr_type',
                 'performance'
                 )

    def __init__(self,
                 target_type=None,
                 target=None,
                 samples=None,
                 par_bounds=None,
                 metrics=None,
                 kde_options=None,
                 stest={'tests': ['correlation',
                                  'kldiv',
                                  'ks',
                                  'jsdiv',
                                  'mannwhitneyu',
                                  'kruskalwallis',
                                  ],
                        'mw_p_threshold': 0.90,
                        'kw_p_threshold': 0.90,
                        'ks_p_threshold': 0.90,
                        },
                 test_metrics=['mode0_location',
                               'mode0_count',
                               'mode1_location',
                               'mode1_count',
                               'mean',
                               ],
                 parameters=['area',
                             ],
                 ):
        """
        This is a core UPXO class and has the following functions:

            * Caclulate type of statistical distribution of the specified
              morphological properties of the target grain structure
              and sample grain structures.

            * Estimate statistical similarity between the target grain
              structure and each of the "samples" grain structures

            * Provide an acceptance flag for each samples grain structures
        """
        self.target_type = target_type
        self.target = target
        self.samples = samples
        self.par_bounds = par_bounds
        self.metrics = metrics
        self.kde_options = kde_options
        self.stest = stest
        self.test_metrics = test_metrics
        self.parameters = parameters
        self.performance = {}
        # from scipy.stats import gaussian_kde

    def load_target(self,
                    target=None,
                    target_type=None):
        self.target = target
        self.target_type = target_type

    def load_samples(self,
                     samples=None):
        if type(samples) in dth.dt.ITERABLES:
            self.samples = samples
        else:
            print('samples must be of the type list.')

    def add_sample(self,
                   sample=None):
        if sample:
            self.samples.append(sample)

    def set_stests(self,
                   tests):
        self.stest['tests'] = tests

    def set_cor_thresh(self,
                       cor_threshold):
        while cor_threshold < 0 or cor_threshold > 1:
            self.stest['cor_threshold'] = float(input("cor_threshold [0, 1]: "))

    def set_kldiv_thresh(self,
                         kldiv_thresh):
        while kldiv_thresh < 0 or kldiv_thresh > 1:
            self.stest['kldiv_thresh'] = float(input("kldiv_thresh [0, 1]: "))

    def set_ks_thresh(self,
                      ks_thresh_D,
                      ks_thresh_P):
        while ks_thresh_D < 0 or ks_thresh_D > 1:
            self.stest['ks_thresh_D'] = float(input("ks_thresh_D [0, 1]: "))
        while ks_thresh_P < 0 or ks_thresh_P > 1:
            self.stest['ks_thresh_P'] = float(input("ks_thresh_P [0, 1]: "))

    def set_jsdiv_thresh(self,
                         jsdiv_thresh):
        while jsdiv_thresh < 0 or jsdiv_thresh > 1:
            self.stest['jsdiv_thresh'] = float(input("jsdiv_thresh [0, 1]: "))

    def prop_to_excel(self,
                      filename="pxtal_properties",
                      ):
        with pd.ExcelWriter(f"{filename}.xlsx") as writer:
            self.target.prop.to_excel(writer,
                                      sheet_name='target',
                                      index=False)
            for i, sample in enumerate(self.samples.values(), start=1):
                sample.prop.to_excel(writer,
                                     sheet_name=f"sample{i}",
                                     index=False
                                     )

    def build_distribution_dataset(self):
        self.distr_type = {'target': {}}
        for sample_name in self.samples.keys():
            self.distr_type[sample_name] = {}
        for key in self.distr_type.keys():
            for parameter in self.parameters:
                self.distr_type[key][parameter] = {'right_skewed': None,
                                                   'left_skewed': None,
                                                   'leptokurtic': None,
                                                   'platykurtic': None,
                                                   'normal': None,
                                                   'kurtosis': None,
                                                   'skewness': None
                                                   }

    def determine_distr_type(self):
        self.build_distribution_dataset()
        for parameter_name in self.parameters:
            target_skewness = skew(self.target.prop[parameter_name])
            target_kurt = kurtosis(self.target.prop[parameter_name])
            shapiro_stat, shapiro_p = shapiro(self.target.prop[parameter_name])
            self.distr_type['target'][parameter_name]['skewness'] = target_skewness
            self.distr_type['target'][parameter_name]['kurtosis'] = target_kurt
            if target_skewness > 0:
                self.distr_type['target'][parameter_name]['right_skewed'] = True
                if target_kurt > 0:
                    self.distr_type['target'][parameter_name]['leptokurtic'] = True
                else:
                    self.distr_type['target'][parameter_name]['platykurtic'] = True
            else:
                self.distr_type['target'][parameter_name]['left_skewed'] = True
                if target_kurt > 0:
                    self.distr_type['target'][parameter_name]['leptokurtic'] = True
                else:
                    self.distr_type['target'][parameter_name]['platykurtic'] = True
            if abs(target_skewness) < 0.5 and abs(target_kurt) < 1 and shapiro_p > 0.05:
                self.distr_type['target'][parameter_name]['normal'] = True
            else:
                self.distr_type['target'][parameter_name]['normal'] = False

        for sample_name, sample in self.samples.items():
            for parameter_name in self.parameters:
                sample_skewness = skew(sample.prop[parameter_name])
                sample_kurt = kurtosis(sample.prop[parameter_name])
                stat, p = shapiro(sample.prop[parameter_name])
                self.distr_type[sample_name][parameter_name]['skewness'] = target_skewness
                self.distr_type[sample_name][parameter_name]['kurtosis'] = target_kurt
                if sample_skewness > 0:
                    self.distr_type[sample_name][parameter_name]['right_skewed'] = True
                    if sample_kurt > 0:
                        self.distr_type[sample_name][parameter_name]['leptokurtic'] = True
                    else:
                        self.distr_type[sample_name][parameter_name]['platykurtic'] = True
                else:
                    self.distr_type[sample_name][parameter_name]['left_skewed'] = True
                    if sample_kurt > 0:
                        self.distr_type[sample_name][parameter_name]['leptokurtic'] = True
                    else:
                        self.distr_type[sample_name][parameter_name]['platykurtic'] = True
                if abs(sample_skewness) < 0.5 and abs(sample_kurt) < 1 and shapiro_p > 0.05:
                        self.distr_type[sample_name][parameter_name]['normal'] = True
                else:
                    self.distr_type[sample_name][parameter_name]['normal'] = False

    def test(self):
        """
        TEST 1: correlation: For two datasets, it is a measure of the linear
        relationship between them. If correlation is close to 1 then, the
        distributions are very similar.

        TEST 2: kldiv:

        TEST 3: ks: Kolmogorov-Smirnov test: Determines of the two distribution
        samples differ significantly. It uses cumulative distributions of the
        two datasets. Retyurns D-statistic and P-value.
            * D-statistic: maximum absolute difference of the cumulative
            distributions (absolute max distance (supremum) b/w the CDFs
            of the two samples). A smaller D-static value is indicative of
            similar distributions.
            * P-value: probability that thwe tywo distributions are similar. If
            p-value is low (<= 0.05), distributions are different. If p-value
            is high (> 0.05), we cannot reject the null-hypothesis that the
            two distributions are the same.
            * Note: if P <= 0.05: the null hypothesis that the two samples are
            drawn from tyhe sample sample can be rejected, indicating that the
            samples are not representative of the target

        TEST 4: jsdiv: P value will allways be between 0 and 1.
        @ 0: Distributions are identical. @ 1: Distributions are completely
        different

        TEST 5: mannwhitneyu: Mann-Whitney test: Used to determine if two '
        distribution samples are drawn from a population having the same
        population. If P-value is less than or equal to 0.05, then different
        distributiopns. If P-value is > 0.05, then the two disrtirbutions
        are similar.

        TEST 6: kruskalwallis: Kruskal-wallis test. Used to determine if there
        are statistically significant differences between two distributions.
        """
        if 'kldiv' in self.stest['tests']:
            from scipy.stats import entropy
        if 'jsdiv' in self.stest['tests']:
            from scipy.spatial.distance import jensenshannon
        if 'ks' in self.stest['tests']:
            from scipy.stats import ks_2samp
        if 'mannwhitneyu' in self.stest['tests']:
            from scipy.stats import mannwhitneyu
        if 'kruskalwallis' in self.stest['tests']:
            from scipy.stats import kruskal
        if self.stest['tests']:
            # Iterate through each of the sample object
            for sample_name, sample in self.samples.items():
                print('-----------sample-----------')
                self.performance[sample_name] = {}
                for ipar, par in enumerate(self.parameters, start=1):
                    self.performance[sample_name][par] = {}
                    for test in self.stest['tests']:
                        self.performance[sample_name][par][test] = None
                        if test == 'correlation':
                            correlation = self.target.prop[par].corr(sample.prop[par])
                            self.performance[sample_name][par][test] = correlation
                        # -------------------------------------
                        if test == 'kldiv':
                            print('kldiv test not available')
                        # -------------------------------------
                        if test == 'ks':
                            ks_D, ks_P = ks_2samp(self.target.prop[par],
                                                  sample.prop[par])
                            self.performance[sample_name][par][test] = (ks_D,
                                                                         ks_P)
                        # -------------------------------------
                        if test == 'jsdiv':
                            # TODO: DEBUG the length mismatch
                            # SOLn: Make KDE and resample data iteratively
                            # based on user satisfaction of number of bins in
                            # histogram and bandwidth in KDE calculation
                            pass
                            #js_P = jensenshannon(self.target.prop[par],
                            #                     sample.prop[par])
                            #self.performance[sample_name][par][test] = js_P
                        # -------------------------------------
                        if test == 'mannwhitneyu':
                            mwu_D, mwu_P = mannwhitneyu(self.target.prop[par].dropna(),
                                                        sample.prop[par].dropna())
                            self.performance[sample_name][par][test] = (mwu_D,
                                                                        mwu_P)
                        # -------------------------------------
                        if test == 'kruskalwallis':
                            kw_D, kw_P = kruskal(self.target.prop[par].dropna(),
                                                 sample.prop[par].dropna())
                            self.performance[sample_name][par][test] = (kw_D,
                                                                        kw_P)
                        # -------------------------------------
