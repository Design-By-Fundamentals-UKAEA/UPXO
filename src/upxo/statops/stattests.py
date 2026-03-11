"""
Created on Fri May 31 13:32:03 2024

@author: Dr. Sunil Anandatheertha

Imports
-------
from upxo.statops.stattests import test_rand_distr_autocorr
from upxo.statops.stattests import test_rand_distr_runs
from upxo.statops.stattests import test_rand_distr_chisquare
from upxo.statops.stattests import test_rand_distr_kolmogorovsmirnov
from upxo.statops.stattests import test_rand_distr_kullbackleibler
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import kstest
from scipy.stats import entropy
from scipy.stats import chisquare
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.spatial.distance import pdist, squareform
import upxo._sup.dataTypeHandlers as dth

def check_coord_distr_for_randomness(coords, method='by_distance', cor=1):
    """
    Parameters
    ----------
    coords: .
    method: method to choose. We can either use distance or the count. The
        valid options are 'by_distance' and 'by_count'.
    cor: cut-off radius. Used only when method = 'by_count'.

    Example
    -------
    from scipy.spatial.distance import pdist, squareform
    import seaborn as sns

    coords = np.random.random((10000,2))
    #####################################
    from scipy.spatial import cKDTree
    coordtree = cKDTree(coords, leafsize=16, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
    cut_off_radius = 0.25
    npoints = [coordtree.query_ball_point(coord, cut_off_radius, p=2., return_length=True) for coord in coords]
    sns.histplot(npoints, kde=True, color='gray', kde_kws={'linecolor': 'black'})

    test_results = test_rand_distr_autocorr(npoints,
                                            apply_random_shuffle=True,
                                            alpha=0.05,
                                            plot_acf=False,
                                            print_msg=False)
    test_results['random']
    As expected, the distribution of npoints is non-randpom.
    Now, lets check for normality.
    # 1. Visual Inspection
        # Histogram
        plt.hist(npoints, bins='auto', density=True, alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Histogram of Data')
        plt.show()

        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=plt)  # Compare to standard normal
        plt.title('Q-Q Plot (Normal)')
        plt.show()
    #####################################

    tests = {'test_rand_distr_autocorr': False,
             'test_rand_distr_runs': True,
             'test_rand_distr_chisquare': True,
             'test_rand_distr_kolmogorovsmirnov': True,
             'test_rand_distr_kullbackleibler': True}

    distances_matrix = squareform(pdist(centroids))
    triu_indices = np.triu_indices_from(distances_matrix, k=1)
    distances = distances_matrix[triu_indices]
    test_results = check_coord_distr_for_randomness(coords)
    test_results['random']
    """
    pass

def test_rand_distr_autocorr(ARRAY,
                             alpha=0.05,
                             apply_random_shuffle=True,
                             _min_array_size_=10,
                             plot_acf=False, print_msg=False):
    """
    Import
    ------
    from upxo.statops.stattests import test_rand_distr_autocorr

    Example
    -------
    from scipy.spatial.distance import pdist, squareform
    centroids = np.random.random((100,2))
    distances_matrix = squareform(pdist(centroids))
    triu_indices = np.triu_indices_from(distances_matrix, k=1)
    distances = distances_matrix[triu_indices]
    test_results = test_rand_distr_autocorr(distances,
                                            apply_random_shuffle=True,
                                            alpha=0.05,
                                            plot_acf=False,
                                            print_msg=False)
    test_results['random']

    Explanations
    ------------
    # AUTO-CORRELATION TEST TO CHECK FOR RANDOMNESS
    Checks for correlation between the values at different lags.
    """
    # Validations
    if type(ARRAY) in dth.dt.ITERABLES:
        if isinstance(ARRAY, np.ndarray):
            if ARRAY.ndim != 1:
                raise ValueError("Input must be a 1D NumPy array.")
            else:
                # nothing to do. User input is correct.
                pass
        else:
            return test_rand_distr_autocorr(np.array(ARRAY),
                                            alpha=alpha,
                                            plot_acf=plot_acf,
                                            print_msg=print_msg)
    else:
        raise ValueError("Input must be or convertable to 1D numpy array.")

    if len(ARRAY) < _min_array_size_:
        warnings.warn(f"Autocorrelation analysis may be unreliable with less than {_min_array_size_} data points.",
                      UserWarning,)

    if apply_random_shuffle:
        np.random.shuffle(ARRAY)
    # Calculate autocorrelation for specific lags
    autocorrelation_values = acf(ARRAY, fft=False)
    # significance_level adjusted for alpha
    significance_level = 1.96 * np.sqrt(1 / (len(ARRAY) - 1)) * (1 - alpha/2)
    random = all(abs(autocorrelation_values[1:]) < significance_level)

    # --------------------------------
    if print_msg:
        print('Result of Auto Correlation test for randomness.')
        print("...Autocorrelation values:", autocorrelation_values)
        print(f"...Random: {random}")
    if plot_acf:
        plt.figure(figsize=(10, 5))
        plt.stem(np.arange(len(autocorrelation_values)), autocorrelation_values, use_line_collection=True)
        plt.axhline(significance_level, color='r', linestyle='--', label=f'Significance level ({(1 - alpha)*100:.0f}%)')
        plt.axhline(-significance_level, color='r', linestyle='--')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation Plot')
        plt.legend()
        plt.show()
    return {"random": random, "autocorrelation_values": autocorrelation_values, "significance_level": significance_level}


def test_rand_distr_runs(ARRAY, alpha=0.05, print_msg=False):
    """
    Import
    ------
    from upxo.statops.stattests import test_rand_distr_runs

    Example
    -------
    from scipy.spatial.distance import pdist, squareform
    centroids = [(1, 2), (3, 4), (5, 6), (7, 8)]
    distances_matrix = squareform(pdist(centroids))
    triu_indices = np.triu_indices_from(distances_matrix, k=1)
    distances = distances_matrix[triu_indices]
    random = test_rand_distr_runs(distances)
    random

    Explanations
    ------------
    # RUNS TEST FOR RABNDOMNESS
    Checks for randomness in the sequence of values.
    """

    def runs_test(arr):
        median = np.median(arr)
        runs = np.sum(np.diff(arr > median) != 0) + 1
        n1 = np.sum(arr > median)
        n2 = np.sum(arr <= median)
        expected_runs = 1 + 2*n1*n2 / (n1 + n2)
        std_runs = np.sqrt(2*n1*n2 * (2*n1*n2 - n1 - n2) / ((n1 + n2)**2 * (n1 + n2 - 1)))
        z = (runs - expected_runs) / std_runs
        p_value = 2 * (1 - norm.cdf(abs(z)))
        return runs, expected_runs, std_runs, z, p_value
    runs, expected_runs, std_runs, z, p_value = runs_test(ARRAY)
    if print_msg:
        print(f"Runs: {runs}")
        print(f"Expected Runs: {expected_runs}")
        print(f"Standard Deviation of Runs: {std_runs}")
        print(f"Z-value: {z}")
        print(f"P-value: {p_value}")
        if p_value > alpha:
            print("The ARRAY array is consistent with being random (runs test).")
        else:
            print("The ARRAY array is not consistent with being random (runs test).")
    return True if p_value > alpha else False


def test_rand_distr_chisquare(ARRAY, alpha=0.05, print_msg=False):
    """
    Import
    ------
    from upxo.statops.stattests import test_rand_distr_chisquare

    Example
    -------
    from scipy.spatial.distance import pdist, squareform
    centroids = [(1, 2), (3, 4), (5, 6), (7, 8)]
    distances_matrix = squareform(pdist(centroids))
    triu_indices = np.triu_indices_from(distances_matrix, k=1)
    distances = distances_matrix[triu_indices]
    random = test_rand_distr_chisquare(distances)
    random

    Explanations
    ------------
    # CHI-SQUARE TEST FOR RANDOMNESS
    Checks if the data follows a uniform distribution.
    """
    # Bin the ARRAY
    num_bins = int(np.sqrt(len(ARRAY)))  # Rule of thumb for number of bins
    hist, bin_edges = np.histogram(ARRAY, bins=num_bins, density=False)
    # Calculate the expected frequency for a uniform distribution
    expected_freq = np.full_like(hist, len(ARRAY) / num_bins)
    # Perform the Chi-Square Test
    chi2_stat, p_value = chisquare(hist, expected_freq)

    if print_msg:
        print(f"Chi2 Statistic: {chi2_stat}")
        print(f"P-value: {p_value}")
    if p_value > alpha:
        random = True
        if print_msg:
            print("The ARRAY array is consistent with being random (uniformly distributed).")
    else:
        random = False
        if print_msg:
            print("The ARRAY array is not consistent with being random (not uniformly distributed).")
    return {'random': random,
            'test_statistic': chi2_stat,
            'p_value': p_value}


def test_rand_distr_kolmogorovsmirnov(ARRAY, alpha=0.05, print_msg=False):
    """
    Import
    ------
    from upxo.statops.stattests import test_rand_distr_kolmogorovsmirnov

    Example
    -------
    from scipy.spatial.distance import pdist, squareform
    centroids = [(1, 2), (3, 4), (5, 6), (7, 8)]
    distances_matrix = squareform(pdist(centroids))
    triu_indices = np.triu_indices_from(distances_matrix, k=1)
    distances = distances_matrix[triu_indices]
    random = test_rand_distr_kolmogorovsmirnov(distances)
    random

    Explanations
    ------------
    # Kolmogorov-Smirnov Test
    Compares the empirical distribution of your data to a reference
    distribution, often used for hypothesis testing.

    The K-S test compares the empirical distribution function of your data
    with a reference probability distribution (e.g., uniform distribution). It
    is useful for testing if a sample comes from a specific distribution.
    """
    # Perform the K-S Test
    ks_stat, p_value = kstest(ARRAY, 'uniform',
                              args=(np.min(ARRAY),
                                    np.max(ARRAY)-np.min(ARRAY)))
    if print_msg:
        print(f"K-S Statistic: {ks_stat}")
        print(f"P-value: {p_value}")
    if p_value > alpha:
        random = True
        if print_msg:
            print("The ARRAY array is consistent with being random (uniform distribution).")
    else:
        random = False
        if print_msg:
            print("The ARRAY array is not consistent with being random (uniform distribution).")
    return {'random': random,
            'test_statistic': ks_stat,
            'p_value': p_value}


def test_rand_distr_kullbackleibler(ARRAY, bin_method='auto', alpha=0.5,
                                    print_msg=False):
    """
    Options for bin_method
    ----------------------
    * 'auto': maximum of the ‘sturges’ and ‘fd’ estimators
    * 'fd': Freedman Diaconis Estimator. Can be too conservative for small
            datasets, but is quite good for large datasets.
    * 'scott': Can be too conservative for small datasets, but is quite good
               for large datasets. The standard deviation is not very robust
               to outliers. Values are very similar to the Freedman-Diaconis
               estimator in the absence of outliers.
    * 'rice': It tends to overestimate the number of bins and it does not take
              into account data variability
    * 'sturges': This estimator assumes normality of data and is too
                 conservative for larger, non-normal datasets.
    * 'doane': An improved version of Sturges’ formula that produces better
               estimates for non-normal datasets. This estimator attempts to
               account for the skew of the data.
    * 'sqrt': The simplest and fastest estimator. Only takes into account the
              data size.
    NOTE: The above explanations for bin_method options are taken frm the
          below reference.
    https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges

    Import
    ------
    from upxo.statops.stattests import test_rand_distr_kullbackleibler

    Example
    -------
    from scipy.spatial.distance import pdist, squareform
    centroids = [(1, 2), (3, 4), (5, 6), (7, 8)]
    distances_matrix = squareform(pdist(centroids))
    triu_indices = np.triu_indices_from(distances_matrix, k=1)
    distances = distances_matrix[triu_indices]
    random = test_rand_distr_kullbackleibler(distances, bin_method='auto')
    random

    Explanations
    ------------
    # Kullback-Leibler Divergence
    Measures how one probability distribution diverges from a reference
    distribution, providing a sense of how "random" your data is compared to a
    uniform distribution.

    The KL divergence measures how one probability distribution diverges from
    a second, expected probability distribution. It is useful to compare the
    observed distribution with an expected random distribution.
    """
    hist_obs, bin_edges = np.histogram(ARRAY, bins=bin_method, density=True)
    # Calculate histograms
    num_bins = len(bin_edges)-1
    # Expected uniform distribution
    hist_exp = np.full_like(hist_obs, 1 / num_bins)
    # Add small value to avoid division by zero
    hist_obs += 1e-10
    hist_exp += 1e-10
    # Calculate KL Divergence
    kl_div = entropy(hist_obs, hist_exp)
    if print_msg:
        print(f"KL Divergence: {kl_div}")
    if kl_div < alpha:
        random = True
        if print_msg:
            print("The ARRAY array is consistent with being random (similar to uniform distribution).")
    else:
        random = False
        if print_msg:
            print("The ARRAY array is not consistent with being random (not similar to uniform distribution).")
    return {'random': random,
            'kl_div': kl_div}
