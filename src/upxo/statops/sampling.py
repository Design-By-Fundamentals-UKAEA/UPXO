'''
https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/

https://pypi.org/project/poissonDiskSampling/

bridson1 offers consant density Poisson disk sampling
IMPLEMENTED
    https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/Bridson_sampling.py

bridson2 also offers consant density Poisson disk sampling
TO BE IMPLEMENTED
    https://github.com/diregoblin/poisson_disc_sampling

bridson3 also offers consant density Poisson disk sampling
TO BE IMPLEMENTED
https://github.com/emulbreh/bridson

bridson4 offers variable density Poisson disk sampling
TO BE IMPLEMENTED
https://pypi.org/project/poissonDiskSampling/

dart1 offers constant density dart sampling
    https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/DART_sampling_numpy.py
'''
###############################################################################
import numpy as np
import numba
from numba import njit, prange
from scipy.spatial import KDTree

@njit
def random_points_around(p, num, radius):
    """ Generate `num` random points around `p` within the annular region (radius, 2*radius). """
    R = np.random.uniform(radius, 2 * radius, num)
    T = np.random.uniform(0, 2 * np.pi, num)
    return np.column_stack((p[0] + R * np.sin(T), p[1] + R * np.cos(T)))

@njit
def in_limits(p, width, height):
    """ Check if points are within bounds (vectorized for Numba). """
    return (p[:, 0] >= 0) & (p[:, 0] < width) & (p[:, 1] >= 0) & (p[:, 1] < height)

@njit
def in_neighborhood(p, P, M, cellsize, rows, cols, squared_radius):
    """ Vectorized check to see if a point is too close to any existing points. """
    i, j = (p[:, 0] / cellsize).astype(np.int32), (p[:, 1] / cellsize).astype(np.int32)
    valid = np.ones(len(p), dtype=np.bool_)

    for idx in prange(len(p)):  # Parallel loop
        ii, jj = i[idx], j[idx]

        i_min, i_max = max(ii - 2, 0), min(ii + 3, rows)
        j_min, j_max = max(jj - 2, 0), min(jj + 3, cols)

        occupied_indices = np.argwhere(M[i_min:i_max, j_min:j_max])  # Get nonzero indices
        if occupied_indices.shape[0] > 0:
            neighbor_points = np.empty((occupied_indices.shape[0], 2), dtype=np.float32)
            for k in range(occupied_indices.shape[0]):  # Extract points manually
                ni, nj = occupied_indices[k]
                neighbor_points[k] = P[i_min + ni, j_min + nj]

            distances = np.sum((neighbor_points - p[idx]) ** 2, axis=1)
            if np.any(distances < squared_radius):
                valid[idx] = False  # Reject this point

    return valid

@njit
def add_point(p, P, M, active_points, active_count, cellsize):
    """ Add a valid point to the active list and update the grid. """
    i, j = int(p[0] / cellsize), int(p[1] / cellsize)
    P[i, j], M[i, j] = p, True
    active_points[active_count] = p
    return active_count + 1  # Increment active count

def bridson_uniform_density(width=1.0, height=1.0, radius=0.01, k=10):
    """
    Fully optimized Bridson's Poisson Disk Sampling using Numba for acceleration.
    """
    cellsize = radius / np.sqrt(2)
    rows, cols = int(np.ceil(width / cellsize)), int(np.ceil(height / cellsize))
    squared_radius = radius ** 2  # Precompute squared radius for distance checks

    # Initialize grid and storage (Preallocate space)
    P = np.zeros((rows, cols, 2), dtype=np.float32)  # Grid of point positions
    M = np.zeros((rows, cols), dtype=bool)  # Boolean mask for occupied cells

    # Preallocate active points list (fixed-size array)
    max_active_points = int((width * height) / (radius ** 2))  # Upper bound estimate
    active_points = np.zeros((max_active_points, 2), dtype=np.float32)
    active_count = 0  # Track number of active points

    # Start with one random point
    first_point = np.random.uniform([0, 0], [width, height])
    active_count = add_point(first_point, P, M, active_points, active_count, cellsize)

    # Sampling loop
    while active_count > 0:
        idx = np.random.randint(active_count)  # Random index from active points
        p = active_points[idx]

        # Generate candidates
        Q = random_points_around(p, k, radius)

        # Filter valid points (vectorized)
        valid_mask = in_limits(Q, width, height) & in_neighborhood(Q, P, M, cellsize, rows, cols, squared_radius)
        valid_points = Q[valid_mask]

        # Add valid points in bulk
        for q in valid_points:
            if active_count < max_active_points:
                active_count = add_point(q, P, M, active_points, active_count, cellsize)

        # Remove used point by swapping with last active element (fast deletion)
        active_count -= 1
        active_points[idx] = active_points[active_count]

    return P[M]  # Return only valid points



def bridson_uniform_density_old2(width=1.0, height=1.0, radius=0.01, k=10):
    """
    Highly optimized version of Bridson's Poisson Disk Sampling.
    """
    print("Highly optimized version of Bridson's Poisson Disk Sampling.")

    def random_points_around(p, num):
        """ Generate `num` random points in the annular region (radius, 2*radius). """
        R = np.random.uniform(radius, 2 * radius, num)
        T = np.random.uniform(0, 2 * np.pi, num)
        return np.column_stack((p[0] + R * np.sin(T), p[1] + R * np.cos(T)))

    def in_limits(p):
        """ Check if points are within bounds (vectorized). """
        return (p[:, 0] >= 0) & (p[:, 0] < width) & (p[:, 1] >= 0) & (p[:, 1] < height)

    def in_neighborhood(p):
        """ Vectorized check to see if any nearby cells contain points too close. """
        i, j = (p[:, 0] / cellsize).astype(int), (p[:, 1] / cellsize).astype(int)
        valid = np.ones(len(p), dtype=bool)

        for idx, (ii, jj) in enumerate(zip(i, j)):
            i_min, i_max = max(ii - 2, 0), min(ii + 3, rows)
            j_min, j_max = max(jj - 2, 0), min(jj + 3, cols)

            # Extract occupied neighbor cells
            occupied_cells = M[i_min:i_max, j_min:j_max]
            neighbor_points = P[i_min:i_max, j_min:j_max][occupied_cells]

            if neighbor_points.size > 0:
                distances = np.sum((neighbor_points - p[idx]) ** 2, axis=1)
                if np.any(distances < squared_radius):
                    valid[idx] = False  # Reject this point

        return valid

    def add_point(p, count):
        """ Add a point to the valid points list and update the grid. """
        active_points[count] = p
        i, j = int(p[0] / cellsize), int(p[1] / cellsize)
        P[i, j], M[i, j] = p, True
        return count + 1  # Update active count

    # Grid setup
    cellsize = radius / np.sqrt(2)
    rows, cols = int(np.ceil(width / cellsize)), int(np.ceil(height / cellsize))
    squared_radius = radius ** 2  # Precompute squared radius for distance checks

    # Initialize grid and storage (Preallocate space)
    P = np.zeros((rows, cols, 2), dtype=np.float32)  # Grid of point positions
    M = np.zeros((rows, cols), dtype=bool)  # Boolean mask for occupied cells

    # Preallocate active points list (instead of dynamically appending)
    max_active_points = int((width * height) / (radius ** 2))  # Upper bound estimate
    active_points = np.zeros((max_active_points, 2), dtype=np.float32)
    active_count = 0  # Track number of active points

    # Start with one random point
    first_point = np.random.uniform([0, 0], [width, height])
    active_count = add_point(first_point, active_count)

    # Sampling loop
    while active_count > 0:
        idx = np.random.randint(active_count)  # Random index from active points
        p = active_points[idx]

        # Generate candidates
        Q = random_points_around(p, k)

        # Filter valid points (vectorized)
        valid_mask = in_limits(Q) & in_neighborhood(Q)
        valid_points = Q[valid_mask]

        # Add valid points in bulk
        for q in valid_points:
            if active_count < max_active_points:
                active_count = add_point(q, active_count)

        # Remove used point by swapping with last active element (fast deletion)
        active_count -= 1
        active_points[idx] = active_points[active_count]

    return P[M]  # Return only valid points


def bridson_uniform_density_old1(width=1.0, height=1.0, radius=0.01, k=10):
    """
    Bridson's Poisson Disk Sampling for uniform distribution of points.
    Optimized version with NumPy-based distance calculations.
    """
    def random_point_around(p, k=1):
        """ Generate `k` random points around `p` within the annular region (radius, 2*radius). """
        R = np.random.uniform(radius, 2 * radius, k)
        T = np.random.uniform(0, 2 * np.pi, k)
        return np.column_stack((p[0] + R * np.sin(T), p[1] + R * np.cos(T)))

    def in_limits(p):
        """ Check if point is within bounds. """
        return (0 <= p[0] < width) and (0 <= p[1] < height)

    def in_neighborhood(p):
        """ Check if `p` is in the neighborhood of any existing points. """
        i, j = int(p[0] / cellsize), int(p[1] / cellsize)
        if M[i, j]:  # Direct hit (fast path)
            return True

        # Extract local neighbors efficiently using NumPy
        i_min, i_max = max(i-2, 0), min(i+3, rows)
        j_min, j_max = max(j-2, 0), min(j+3, cols)

        occupied_cells = M[i_min:i_max, j_min:j_max]  # Boolean mask
        neighbor_points = P[i_min:i_max, j_min:j_max][occupied_cells]  # Get stored points

        if neighbor_points.size > 0:
            distances = np.sum((neighbor_points - p) ** 2, axis=1)  # Squared Euclidean distance
            return np.any(distances < squared_radius)  # Check if any are too close
        return False

    def add_point(p):
        """ Add a valid point to the sample list and update the grid. """
        points.append(p)
        i, j = int(p[0] / cellsize), int(p[1] / cellsize)
        P[i, j] = p
        M[i, j] = True

    # Grid setup
    cellsize = radius / np.sqrt(2)
    rows, cols = int(np.ceil(width / cellsize)), int(np.ceil(height / cellsize))
    squared_radius = radius ** 2  # Precompute squared radius for distance checks

    # Initialize grid and storage
    P = np.zeros((rows, cols, 2), dtype=np.float32)  # Stores point positions
    M = np.zeros((rows, cols), dtype=bool)  # Mask indicating occupied cells

    # Active points list (for generating new points)
    points = []
    add_point((np.random.uniform(width), np.random.uniform(height)))  # Start with 1 point

    # Sampling loop
    while points:
        idx = np.random.randint(len(points))  # Randomly select an active point
        p = points[idx]
        del points[idx]  # Remove selected point (avoid rechecking)

        # Generate candidate points
        Q = random_point_around(p, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)

    return P[M]  # Return only valid points


def bridson_uniform_density_old(width=1.0,
                            height=1.0,
                            radius=0.01,
                            k=10):
    # Credit: https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/Bridson_sampling.py
    # -----------------------------------------------------------------------------
    # From Numpy to Python
    # Copyright (2017) Nicolas P. Rougier - BSD license
    # More information at https://github.com/rougier/numpy-book
    # -----------------------------------------------------------------------------
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007
    def squared_distance(p0, p1):
        return (p0[0]-p1[0])**2 + (p0[1]-p1[1])**2
    def random_point_around(p, k=1):
        # WARNING: This is not uniform around p but we can live with it
        R = np.random.uniform(radius, 2*radius, k)
        T = np.random.uniform(0, 2*np.pi, k)
        P = np.empty((k, 2))
        P[:, 0] = p[0]+R*np.sin(T)
        P[:, 1] = p[1]+R*np.cos(T)
        return P
    def in_limits(p):
        return 0 <= p[0] < width and 0 <= p[1] < height
    def neighborhood(shape, index, n=2):
        row, col = index
        row0, row1 = max(row-n, 0), min(row+n+1, shape[0])
        col0, col1 = max(col-n, 0), min(col+n+1, shape[1])
        I = np.dstack(np.mgrid[row0:row1, col0:col1])
        I = I.reshape(I.size//2, 2).tolist()
        I.remove([row, col])
        return I
    def in_neighborhood(p):
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        if M[i, j]:
            return True
        for (i, j) in N[(i, j)]:
            if M[i, j] and squared_distance(p, P[i, j]) < squared_radius:
                return True
        return False
    def add_point(p):
        points.append(p)
        i, j = int(p[0]/cellsize), int(p[1]/cellsize)
        P[i, j], M[i, j] = p, True
    # Here `2` corresponds to the number of dimension
    cellsize = radius/np.sqrt(2)
    rows = int(np.ceil(width/cellsize))
    cols = int(np.ceil(height/cellsize))
    # Squared radius because we'll compare squared distance
    squared_radius = radius*radius
    # Positions cells
    P = np.zeros((rows, cols, 2), dtype=np.float32)
    M = np.zeros((rows, cols), dtype=bool)
    # Cache generation for neighborhood
    N = {}
    for i in range(rows):
        for j in range(cols):
            N[(i, j)] = neighborhood(M.shape, (i, j), 2)
    points = []
    add_point((np.random.uniform(width), np.random.uniform(height)))
    while len(points):
        i = np.random.randint(len(points))
        p = points[i]
        del points[i]
        Q = random_point_around(p, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
    return P[M]
# =============================================================================
#     if __name__ == '__main__':
#         plt.figure()
#         plt.subplot(1, 1, 1, aspect=1)
#         points = Bridson_sampling()
#         X = [x for (x, y) in points]
#         Y = [y for (x, y) in points]
#         plt.scatter(X, Y, s=2)
#         plt.xlim(0, 1)
#         plt.ylim(0, 1)
#         plt.show()
# =============================================================================
###############################################################################
def bridson_variable_density():
    # https://pypi.org/project/poissonDiskSampling/
    pass
###############################################################################
@njit
def dart(width=1.0, height=1.0, radius=0.025, k=30):
    """
    Optimized Dart Throwing with Numba (manual distance checks).
    Uses a pre-allocated NumPy array for 'points' instead of a list.
    """
    n = 5 * int((width + radius) * (height + radius) / (0.5 * radius * radius * 1.73205080757)) + 1
    P = np.random.uniform(0, 1, size=n * 2).reshape(n, 2)
    P[:, 0] *= width
    P[:, 1] *= height

    points = np.empty((n, 2), dtype=np.float64)  # Pre-allocate NumPy array
    points[0] = P[0]  # Initialize with the first point
    num_points = 1   # Keep track of the number of valid points
    active_list = [0]
    two_pi = 2 * np.pi

    i = 1
    while active_list and i < n:
        rand_index = active_list[np.random.randint(0, len(active_list))]
        found = False

        for _ in range(k):
            theta = np.random.uniform(0, two_pi)
            r = np.random.uniform(radius, 2 * radius)
            candidate = P[rand_index] + np.array([r * np.cos(theta), r * np.sin(theta)])

            if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                is_close = False
                for j in range(num_points):  # Iterate up to num_points
                    dx = candidate[0] - points[j, 0]
                    dy = candidate[1] - points[j, 1]
                    if dx * dx + dy * dy <= radius * radius:
                        is_close = True
                        break

                if not is_close:
                    points[num_points] = candidate  # Add to the array
                    num_points += 1              # Increment the count
                    active_list.append(i)
                    found = True
                    i += 1
                    break

        if not found:
            del active_list[active_list.index(rand_index)]

    return points[:num_points]  # Return only the valid points

def dart_old1(width=1.0, height=1.0, radius=0.025, k=30):
    """
    Optimized Dart Throwing algorithm for generating a Poisson disk sampling.

    Uses a KDTree for efficient nearest-neighbor searches, and avoids the
    large distance matrix calculation of the original.

    Args:
        width (float): Width of the sampling area.
        height (float): Height of the sampling area.
        radius (float): Minimum distance between points.
        k (int): Maximum number of attempts to find a valid point before giving up
               on adding a new point from the candidate list.

    Returns:
        list: A list of [x, y] coordinates representing the sampled points.
    """
    print(40*'#')

    # 5 times the Theoretical limit
    n = 5 * int((width + radius) * (height + radius) / (0.5 * radius * radius * 1.73205080757)) + 1

    # Generate all random points at once (simplified)
    P = np.random.uniform(low=(0, 0), high=(width, height), size=(n, 2))

    points = [P[0]]  # Initialize with the first point
    active_list = [0]  # Indices of points to consider for near neighbors
    kdtree = KDTree([P[0]]) #Initliase the KDTree with first point

    i = 1  # Index for iterating through the random points P
    while active_list and i < n:
        # Randomly choose a point from the active list
        rand_index = np.random.choice(active_list)
        found = False  # Flag to indicate if a suitable point is found

        for _ in range(k):  # Try 'k' times to find a valid point
            #Generate a point in an annulus.
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(radius, 2*radius) #r must lie between radius and 2*radius.
            candidate = P[rand_index] + [r*np.cos(theta), r*np.sin(theta)]

            # Check boundaries
            if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                # Check distance using KDTree (efficient!)
                distances, _ = kdtree.query(candidate, k=1)
                if distances > radius:
                    points.append(candidate)
                    active_list.append(i) #Append the index.
                    kdtree = KDTree(points) #Rebuild the tree with the new point.
                    found = True
                    i+=1 #Only if found is True
                    break  # Exit the inner loop

        if not found:
            active_list.remove(rand_index) #Remove if not useful.

    return points

def dart_old(width = 1.0, height = 1.0, radius = 0.025, k = 30):
    # -----------------------------------------------------------------------------
    # Credit: https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/DART_sampling_numpy.py
    # -----------------------------------------------------------------------------
    # From Numpy to Python
    # Copyright (2017) Nicolas P. Rougier - BSD license
    # More information at https://github.com/rougier/numpy-book
    # -----------------------------------------------------------------------------
    # Return array type modified by Sunil Anandatheertha, Material Engineer, UKAEA
    # Also, some simplications to calculations were introduced by Sunil Anandatheertha
    # -----------------------------------------------------------------------------
    import numpy as np
    #import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    # 5 times the Theoretical limit
    # n = 5*int((width+radius)*(height+radius) / (2*(radius/2)*(radius/2)*np.sqrt(3))) + 1
    # Simplication of the above commented out expression
    n = 5*int((width+radius)*(height+radius) / (0.5*radius*radius*1.73205080757)) + 1
    # Compute n random points
    P = np.zeros((n, 2))
    P[:, 0] = np.random.uniform(0, width, n)
    P[:, 1] = np.random.uniform(0, height, n)
    # TODO: Simplify the above three lines of codes to  single line
    # Computes respective distances at once
    D = cdist(P, P)
    # Cancel null distances on the diagonal
    D[range(n), range(n)] = 1e10
    points, indices = [P[0]], [0]
    i = 1
    last_success = 0
    while i < n and i - last_success < k:
        if D[i, indices].min() > radius:
            indices.append(i)
            points.append(P[i])
            last_success = i
        i += 1
    return [[_[0],_[1]] for _ in points]
# =============================================================================
#     if __name__ == '__main__':
#
#         plt.figure()
#         plt.subplot(1, 1, 1, aspect=1)
#
#         points = DART_sampling_numpy()
#         X = [x for (x, y) in points]
#         Y = [y for (x, y) in points]
#         plt.scatter(X, Y, s=10)
#         plt.xlim(0, 1)
#         plt.ylim(0, 1)
#         plt.show()
# =============================================================================
###############################################################################
