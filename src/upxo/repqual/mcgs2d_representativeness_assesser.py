"""
Statistical representativeness assessment for 2D MCGS vs a target structure.

Class ``mc2repr`` runs distribution tests (e.g. KS, Mann–Whitney, Kruskal–Wallis,
Pearson) on morphological properties of target vs sample grain structures.
Targets may be EBSD-derived or UPXO MC/Voronoi GS. Results are stored for
inspection; there is no single automatic accept/reject score in this module.

Import::

    from upxo.repqual.mcgs2d_representativeness_assesser import mc2repr
"""

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
    Statistical representativeness of 2D sample GS vs a target.

    Runs distribution tests (KS, Mann–Whitney, Kruskal–Wallis, correlation,
    KL/JS divergence, etc.) on morphological properties of a **target**
    structure against one or more **samples**. Targets may be EBSD-derived
    or UPXO MC/Voronoi GS. Results are stored for inspection — there is no
    single automatic accept/reject score in this class.

    Attributes
    ----------
    target_type : str
        Target source code:

        * ``ebsd0`` — unprocessed 2D EBSD (DefDAP)
        * ``ebsd1`` — processed DefDAP (e.g. remapped avg. orientation)
        * ``umc2`` / ``umc3`` — UPXO Monte-Carlo 2D / 3D
        * ``uvt2`` — UPXO Voronoi tessellation 2D
        * ``stats`` — morphology samples as dict or DataFrame columns
    target
        Target data: ``MCGS.gs[tslice]``, VTGS, DefDAP EBSD, or stats table.
    samples : dict
        Sample name → grain-structure object, or ``'make'`` to generate from
        an Excel dashboard (simulate → temporal slices → characterise).
    par_bounds : dict
        Per-property bounds, e.g. area/perimeter/aspect ratio →
        ``[[peak_loc_%], [peak_density_%], [JS_bounds]]``.
    metrics : list
        Qualification metrics (e.g. ``modes_n``, ``modes_loc``, ``skewness``).
    kde_options : dict
        KDE options (``bw_method``: ``'scott'``, ``'silverman'``, or scalar).
    stest, test_metrics, performance
        Configured statistical tests and computed outcomes.
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
        """Load or import target."""
        self.target = target
        self.target_type = target_type

    def load_samples(self,
                     samples=None):
        """Load or import samples."""
        if type(samples) in dth.dt.ITERABLES:
            self.samples = samples
        else:
            print('samples must be of the type list.')

    def add_sample(self,
                   sample=None):
        """Add or insert sample."""
        if sample:
            self.samples.append(sample)

    def set_stests(self,
                   tests):
        """Set or update stests."""
        self.stest['tests'] = tests

    def set_cor_thresh(self,
                       cor_threshold):
        """Set or update cor thresh."""
        while cor_threshold < 0 or cor_threshold > 1:
            self.stest['cor_threshold'] = float(input("cor_threshold [0, 1]: "))

    def set_kldiv_thresh(self,
                         kldiv_thresh):
        """Set or update kldiv thresh."""
        while kldiv_thresh < 0 or kldiv_thresh > 1:
            self.stest['kldiv_thresh'] = float(input("kldiv_thresh [0, 1]: "))

    def set_ks_thresh(self,
                      ks_thresh_D,
                      ks_thresh_P):
        """Set or update ks thresh."""
        while ks_thresh_D < 0 or ks_thresh_D > 1:
            self.stest['ks_thresh_D'] = float(input("ks_thresh_D [0, 1]: "))
        while ks_thresh_P < 0 or ks_thresh_P > 1:
            self.stest['ks_thresh_P'] = float(input("ks_thresh_P [0, 1]: "))

    def set_jsdiv_thresh(self,
                         jsdiv_thresh):
        """Set or update jsdiv thresh."""
        while jsdiv_thresh < 0 or jsdiv_thresh > 1:
            self.stest['jsdiv_thresh'] = float(input("jsdiv_thresh [0, 1]: "))

    def prop_to_excel(self,
                      filename="pxtal_properties",
                      ):
        """Prop to excel."""
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
        """Build and return  distribution dataset."""
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
        """Determine distr type."""
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
