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

_PROBABILITY_FLOOR = 1e-12


def shared_bin_probabilities(target, sample, bins='auto'):
    """Probabilities of two data sets on the same histogram bins.

    The bin edges come from the pooled data (numpy's ``bins`` rule, ``'auto'`` by default), so both sets are
    counted over the same range. Missing values are dropped. Returns ``(p_target, p_sample)``, each summing to 1,
    or ``None`` when either set has no value.
    """
    t = np.asarray(target, dtype=float)
    s = np.asarray(sample, dtype=float)
    t, s = t[np.isfinite(t)], s[np.isfinite(s)]
    if not t.size or not s.size:
        return None
    edges = np.histogram_bin_edges(np.concatenate((t, s)), bins=bins)
    p = np.histogram(t, bins=edges)[0].astype(float)
    q = np.histogram(s, bins=edges)[0].astype(float)
    return p / p.sum(), q / q.sum()


def kl_divergence(target, sample, bins='auto'):
    """KL divergence D(target || sample), in nats, on shared bins; 0 for identical distributions.

    A bin one set fills and the other leaves empty would make the value infinite, so both probability vectors get
    a floor of 1e-12 and are renormalised. ``nan`` when either set has no value.
    """
    pq = shared_bin_probabilities(target, sample, bins)
    if pq is None:
        return np.nan
    p, q = (np.maximum(x, _PROBABILITY_FLOOR) for x in pq)
    return float(entropy(p / p.sum(), q / q.sum()))


def js_distance(target, sample, bins='auto'):
    """Jensen-Shannon distance (square root of the divergence, base 2) on shared bins.

    0 for identical distributions, 1 for distributions that share no bin. ``nan`` when either set has no value.
    """
    pq = shared_bin_probabilities(target, sample, bins)
    if pq is None:
        return np.nan
    return float(jensenshannon(pq[0], pq[1], base=2))


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

    @staticmethod
    def _unit_interval(name, value):
        """Return value if in [0, 1], else prompt until a valid one is given."""
        while value < 0 or value > 1:
            value = float(input(f"{name} [0, 1]: "))
        return value

    def set_cor_thresh(self,
                       cor_threshold):
        """Set or update cor thresh."""
        self.stest['cor_threshold'] = self._unit_interval(
            'cor_threshold', cor_threshold)

    def set_kldiv_thresh(self,
                         kldiv_thresh):
        """Set or update kldiv thresh."""
        self.stest['kldiv_thresh'] = self._unit_interval(
            'kldiv_thresh', kldiv_thresh)

    def set_ks_thresh(self,
                      ks_thresh_D,
                      ks_thresh_P):
        """Set or update ks thresh."""
        self.stest['ks_thresh_D'] = self._unit_interval(
            'ks_thresh_D', ks_thresh_D)
        self.stest['ks_thresh_P'] = self._unit_interval(
            'ks_thresh_P', ks_thresh_P)

    def set_jsdiv_thresh(self,
                         jsdiv_thresh):
        """Set or update jsdiv thresh."""
        self.stest['jsdiv_thresh'] = self._unit_interval(
            'jsdiv_thresh', jsdiv_thresh)

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
                self.distr_type[sample_name][parameter_name]['skewness'] = sample_skewness
                self.distr_type[sample_name][parameter_name]['kurtosis'] = sample_kurt
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
                if abs(sample_skewness) < 0.5 and abs(sample_kurt) < 1 and p > 0.05:
                        self.distr_type[sample_name][parameter_name]['normal'] = True
                else:
                    self.distr_type[sample_name][parameter_name]['normal'] = False

    def test(self):
        """
        TEST 1: correlation: For two datasets, it is a measure of the linear
        relationship between them. If correlation is close to 1 then, the
        distributions are very similar.

        TEST 2: kldiv: Kullback-Leibler divergence D(target || sample), in
        nats, of the two property distributions on shared histogram bins (see
        kl_divergence). 0 for identical distributions; it grows without bound
        as the sample puts weight where the target has little.

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

        TEST 4: jsdiv: Jensen-Shannon distance (base 2) of the two property
        distributions on shared histogram bins (see js_distance). Always
        between 0 and 1. @ 0: Distributions are identical. @ 1: Distributions
        are completely different

        TEST 5: mannwhitneyu: Mann-Whitney test: Used to determine if two '
        distribution samples are drawn from a population having the same
        population. If P-value is less than or equal to 0.05, then different
        distributiopns. If P-value is > 0.05, then the two disrtirbutions
        are similar.

        TEST 6: kruskalwallis: Kruskal-wallis test. Used to determine if there
        are statistically significant differences between two distributions.
        """
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
                            self.performance[sample_name][par][test] = kl_divergence(
                                self.target.prop[par], sample.prop[par])
                        # -------------------------------------
                        if test == 'ks':
                            ks_D, ks_P = ks_2samp(self.target.prop[par],
                                                  sample.prop[par])
                            self.performance[sample_name][par][test] = (ks_D,
                                                                         ks_P)
                        # -------------------------------------
                        if test == 'jsdiv':
                            self.performance[sample_name][par][test] = js_distance(
                                self.target.prop[par], sample.prop[par])
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
