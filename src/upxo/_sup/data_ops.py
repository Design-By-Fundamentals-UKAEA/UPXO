"""
Created on Thu May 23 09:39:31 2024

@author: rg5749

Import
------
from upxo._sup.data_ops import NAME
"""
import math
from collections import Counter
import numpy as np
import upxo._sup.dataTypeHandlers as dth
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import NearestNDInterpolator

NUMBERS = dth.dt.NUMBERS
ITERABLES = dth.dt.ITERABLES


def find_outliers_iqr(data, mode='both'):
    """
    Find outliers in data using Inter-Quartile Range.

    Import
    ------
    from upxo._sup.data_ops import find_outliers_iqr

    Parameters
    ----------
    data: input data: Iterable
    mode: specify in which direction data is to be extracted
        Options:
            'both': above and below the IQR
            'below':
            'above':
            'in-between':

    Return
    ------
    outlier_indices: indices of outliers in data.
    """
    if type(data) not in ITERABLES:
        raise TypeError('Invalid data type specified.')
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q3-q1
    lower_bound = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    # outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]

    if mode == 'both':
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    elif mode == 'below':
        outlier_indices = np.where(data < lower_bound)[0]
    elif mode == 'above':
        outlier_indices = np.where(data > upper_bound)[0]
    elif mode == 'in-between':
        outlier_indices = np.where((data >= lower_bound) & (data <= upper_bound))[0]
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'both', 'below', 'above', or 'in-between'.")

    return outlier_indices

def distance_between_two_points(point1, point2):
    return math.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)


def calculate_angular_distance(coord1, coord2):
    """
    Calculates the angle in radians between two position vectors
    formed from the origin to the given coordinates.

    Args:
        coord1 (tuple or list): The (x, y) or (x, y, z) coordinates of the first point.
        coord2 (tuple or list): The (x, y) or (x, y, z) coordinates of the second point.

    Returns:
        float: The angle between the position vectors in radians (0 to pi).

    Import
    ------
    from upxo._sup.data_ops import calculate_angular_distance
    """
    # Convert coordinates to NumPy arrays (if not already)
    vec1 = np.array(coord1)
    vec2 = np.array(coord2)

    # Input validation (optional)
    if vec1.shape != vec2.shape:
        raise ValueError("Input coordinates must have the same dimensions.")

    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitudes
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    # Handle zero magnitudes (avoid division by zero)
    if mag1 == 0 or mag2 == 0:
        return 0.0  # Angle is 0 if either vector is the zero vector

    # Calculate the cosine of the angle
    cos_theta = dot_product / (mag1 * mag2)

    # Calculate the angle (arccos) and ensure it's within the valid range
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle_rad

def calculate_density_bins(A, n_bins=10):
    """
    Calculates non-uniform bins for a matrix A of random numbers between 0 and 1,
    based on the density distribution of values.

    Args:
        A (np.ndarray): A 2D array of random numbers between 0 and 1.
        n_bins (int, optional): The desired number of bins (default: 10).

    Returns:
        np.ndarray: An array of bin edges.

    Import
    ------
    from upxo._sup.data_ops import calculate_density_bins

    Example
    -------
    A = np.random.randn(100, 100)
    bin_edges = calculate_density_bins(A, n_bins=15)

    plt.hist(A.ravel(), bins=bin_edges, density=True)
    plt.title('Histogram with Density-Based Bins')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    """

    # Flatten the array for analysis
    values = A.ravel()

    # Calculate the cumulative distribution function (CDF)
    values_sorted = np.sort(values)
    cdf = np.arange(1, len(values) + 1) / len(values)

    # Determine bin edges based on equal spacing in the CDF
    bin_edges = np.interp(np.linspace(0, 1, n_bins + 1), cdf, values_sorted)

    return bin_edges

def approximate_to_bin_means(A, n_bins=50):
    """
    Approximates each element in array A to the mean of its corresponding bin edges.

    Args:
        A (np.ndarray): A 2D array of random numbers between 0 and 1.
        bin_edges (np.ndarray): An array of bin edges calculated using `calculate_density_bins`.

    Returns:
        np.ndarray: A new 2D array with elements approximated to their bin means.

    Import
    ------
    from upxo._sup.data_ops import approximate_to_bin_means
    """
    bin_edges = calculate_density_bins(A, n_bins=n_bins)
    # Create a copy of A to avoid modifying the original array
    A_approx = A.copy()

    # Digitize to find the bin index for each value
    bin_indices = np.digitize(A_approx, bin_edges) - 1
    # Calculate bin means
    bin_means = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Replace values with their corresponding bin means
    for i in range(A_approx.shape[0]):
        for j in range(A_approx.shape[1]):
            bin_index = bin_indices[i, j]-1
            A_approx[i, j] = bin_means[bin_index]

    return bin_means, A_approx

def make_equal_bins(data1, data2):
    # Calculate histograms with the same bins for fair comparison
    min_degree = min(np.min(data1), np.min(data2))
    max_degree = max(np.max(data1), np.max(data2))
    bins = np.arange(min_degree, max_degree + 2)  # +2 for inclusive range
    data1_eq_bin, _ = np.histogram(data1, bins=bins, density=True)
    data2_eq_bin, _ = np.histogram(data2, bins=bins, density=True)
    return data1_eq_bin, data2_eq_bin

def find_intersection(arrays):
    # Filter out empty arrays
    non_empty_arrays = [arr for arr in arrays if arr.size > 0]

    # If there are no non-empty arrays, return an empty array
    if not non_empty_arrays:
        return np.array([], dtype=int)

    # Compute the intersection of the non-empty arrays
    intersection = non_empty_arrays[0]
    for arr in non_empty_arrays[1:]:
        intersection = np.intersect1d(intersection, arr)

    return intersection

def find_union_with_counts(arrays):
    # Filter out empty arrays
    non_empty_arrays = [arr for arr in arrays if arr.size > 0]

    # If there are no non-empty arrays, return an empty array and an empty dictionary
    if not non_empty_arrays:
        return np.array([], dtype=int), {}

    # Compute the union of all non-empty arrays
    union = np.unique(np.concatenate(non_empty_arrays))

    # Count the number of arrays in which each element of the union is present
    element_count = Counter()
    for arr in non_empty_arrays:
        unique_elements = np.unique(arr)
        element_count.update(unique_elements)

    # Convert the Counter object to a dictionary
    element_count_dict = dict(element_count)

    return union, element_count_dict


def increase_grid_resolution(Xgrid, Ygrid, Zvalues, factor, method='linear'):
    """Increases the resolution of a grid defined by Xgrid and Ygrid coordinates and Zvalues.

    Args:
        Xgrid: A 2D NumPy array of x-coordinates.
        Ygrid: A 2D NumPy array of y-coordinates.
        Zvalues: A 2D NumPy array of values on the grid.
        factor: The refinement factor (must be > 1).
        method: Interpolation method for x and y coordinates ('linear', 'cubic', 'nearest'). Defaults to 'linear'.

    Returns:
        Three 2D NumPy arrays:
            new_Xgrid: Refined x-coordinates.
            new_Ygrid: Refined y-coordinates.
            new_Zvalues: Interpolated z-values.

    Raises:
        ValueError: If an invalid method is specified or factor is <= 1.
    """
    if factor <= 1:
        raise ValueError("Factor must be greater than 1 for refinement.")

    # New dimensions
    new_rows = int(np.round(Xgrid.shape[0] * factor))
    new_cols = int(np.round(Xgrid.shape[1] * factor))

    # Create new, finer coordinate grids
    new_x = np.linspace(Xgrid.min(), Xgrid.max(), new_cols)
    new_y = np.linspace(Ygrid.min(), Ygrid.max(), new_rows)
    new_Xgrid, new_Ygrid = np.meshgrid(new_x, new_y)

    # Create interpolator for Xgrid and Ygrid
    xy_interpolator = RegularGridInterpolator(
        (np.unique(Xgrid), np.unique(Ygrid)), Zvalues, method=method, bounds_error=False, fill_value=None
    )

    # Create NearestNDInterpolator for Zvalues
    z_interpolator = NearestNDInterpolator(
        list(zip(Xgrid.ravel(), Ygrid.ravel())), Zvalues.ravel()
    )

    # Interpolate
    new_Zvalues = z_interpolator(new_Xgrid, new_Ygrid)

    return new_Xgrid, new_Ygrid, new_Zvalues

def decrease_grid_resolution(Xgrid, Ygrid, Zvalues, factor):
    """Decreases the resolution of a grid defined by Xgrid and Ygrid coordinates and Zvalues.

    Args:
        Xgrid: A 2D NumPy array of x-coordinates.
        Ygrid: A 2D NumPy array of y-coordinates.
        Zvalues: A 2D NumPy array of values on the grid.
        factor: The coarsening factor (must be between 0 and 1).

    Returns:
        Three 2D NumPy arrays:
            new_Xgrid: Coarsened x-coordinates.
            new_Ygrid: Coarsened y-coordinates.
            new_Zvalues: Decimated z-values.

    Raises:
        ValueError: If the factor is not between 0 and 1.
    """

    if not (0 < factor < 1):
        raise ValueError("Factor must be between 0 and 1 for coarsening.")

    # Adjust the factor for slicing
    factor = int(1 / factor)

    # Calculate new shape after coarsening
    new_shape = (int(np.round(Xgrid.shape[0] * (1/factor))), int(np.round(Xgrid.shape[1] * (1/factor))))

    # Determine the new x and y ranges based on the original grid and new shape
    new_x = np.linspace(Xgrid.min(), Xgrid.max(), new_shape[1])
    new_y = np.linspace(Ygrid.min(), Ygrid.max(), new_shape[0])

    # Create meshgrids with 'ij' indexing to match the interpolator's grid
    new_Xgrid, new_Ygrid = np.meshgrid(new_x, new_y, indexing='ij')

    # Create NearestNDInterpolator for Zvalues
    z_interpolator = NearestNDInterpolator(
        list(zip(Xgrid.ravel(), Ygrid.ravel())), Zvalues.ravel()
    )

    # Interpolate using the decimated grids
    new_Zvalues = z_interpolator(new_Xgrid, new_Ygrid)

    return new_Xgrid.T, new_Ygrid.T, new_Zvalues.T


def find_common_coordinates(array1, array2):
    """
    Finds the coordinates in array2 which are also present in array1.

    :param array1: N x 2 numpy array representing the first set of coordinates.
    :param array2: N x 2 numpy array representing the second set of coordinates.
    :return: A numpy array containing the common coordinates.
    """
    # Ensure arrays are contiguous
    array1 = np.ascontiguousarray(array1)
    array2 = np.ascontiguousarray(array2)

    # Convert to structured arrays
    dtype = np.dtype([('x', array1.dtype), ('y', array1.dtype)])
    structured_array1 = array1.view(dtype).reshape(-1)
    structured_array2 = array2.view(dtype).reshape(-1)

    # Find intersection
    common_structured = np.intersect1d(structured_array1, structured_array2)

    # Convert back to regular numpy array
    common_coordinates = common_structured.view(array1.dtype).reshape(-1, 2)

    return common_coordinates

def moving_average(data, window_size):
    """Compute the moving average of the given data with the specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def mean_coordinates(coords, window_size):
    """
    Smooths the given 2D numpy array of coordinates using a moving average.

    Parameters:
    coords (numpy.ndarray): A 2D numpy array of shape (n, 2) where n is the number of points.
    window_size (int): The window size for the moving average.

    Returns:
    numpy.ndarray: A 2D numpy array of the smoothed coordinates.
    """
    # Check if there are enough points for the moving average
    if len(coords) < window_size:
        return coords  # Return the original coordinates if not enough points

    # Separate the coordinates into x and y components
    x = coords[:, 0]
    y = coords[:, 1]

    # Apply moving average to the x and y components separately
    x_smooth = moving_average(x, window_size)
    y_smooth = moving_average(y, window_size)

    # Add the original end points to the smoothed coordinates if there are enough points
    if len(x_smooth) > 0 and len(y_smooth) > 0:
        smoothed_coords = np.vstack([
            [x[0], y[0]],  # Start point
            np.column_stack([x_smooth, y_smooth]),
            [x[-1], y[-1]]  # End point
        ])
    else:
        smoothed_coords = coords  # If not enough points, use original coordinates

    return smoothed_coords

def is_a_in_b(a, b):
    return any((b[:, 0] == a[0]) & (b[:, 1] == a[1]))

def is_a_in_b_3d(a, b):
    return any((b[:, 0] == a[0]) & (b[:, 1] == a[1]) & (b[:, 2] == a[2]))

def find_coorda_loc_in_coords_arrayb(a, b):
    # find_coorda_loc_in_coords_arrayb(neigh_points[1], sinkarray)
    return np.argwhere((b[:, 0] == a[0]) & (b[:, 1] == a[1]))[0][0]

def remove_2d_child_array_from_2d_parent_array(parent_array, child_array):
    """
    Import
    ------
    from upxo._sup.data_ops import remove_2d_child_array_from_2d_parent_array

    Example
    -------
    parent = np.array([[1, 2],
                       [3, 4],
                       [5, 6],
                       [7, 8]])
    child = np.array([[3, 4],
                      [1, 2]
                      ])
    remove_2d_child_array_from_2d_parent_array(parent, child)
    """
    return parent_array[~np.isin(parent_array, child_array).all(axis=1)]

def remove_permutations(arr):
    """
    arr = np.array([[1, 2, 3], [3, 2, 1], [1, 3, 2], [2, 1, 3],
                    [4, 3, 2], [4, 3, 2], [2, 3, 4], [4, 2, 3]
                    ])
    unique_arr = remove_permutations(arr)
    unique_arr
    """
    # Sort each row and convert to a tuple to make them hashable
    sorted_rows = [tuple(sorted(row)) for row in arr]
    # Use a set to remove duplicates
    unique_rows = list(set(sorted_rows))
    # Convert back to NumPy array
    result = np.array(unique_rows)

    return result

def remove_permutations1(arr, N):
    # Ensure each column contains numbers from 1 to N
    for col in range(arr.shape[1]):
        if set(arr[:, col]) != set(range(1, N+1)):
            raise ValueError(f"Column {col+1} does not contain all numbers from 1 to {N}")

    # Sort each row and convert to a tuple to make them hashable
    sorted_rows = [tuple(sorted(row)) for row in arr]

    # Use a set to remove duplicates
    unique_rows = list(set(sorted_rows))

    # Convert back to NumPy array
    result = np.array(unique_rows)

    return result

def find_closest_locations(array, parameter_metric):
    # Calculate the desired metric based on the input control
    if parameter_metric == 'mean':
        metric_value = np.mean(array)
    elif parameter_metric == 'minimum':
        metric_value = np.min(array)
    elif parameter_metric == 'median':
        metric_value = np.median(array)
    elif parameter_metric == 'maximum':
        metric_value = np.max(array)
    else:
        raise ValueError("Invalid parameter_metric. Choose from 'minimum', 'mean', 'median', 'maximum'.")
    # Find the locations in the array that are closest to the metric value
    diff = np.abs(array - metric_value)
    try:
        diff_min = np.min(diff)
        closest_locations = np.where(diff == diff_min)[0]
    except ValueError:
        closest_locations = array

    return closest_locations
