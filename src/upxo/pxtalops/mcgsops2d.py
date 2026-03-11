import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from upxo.pxtal.mcgs2_temporal_slice import mcgs2_grain_structure
from upxo._sup.validation_values import _validation
from upxo._sup.upxo_types import utypes


def find_gb_jp(gs=None,
               kernel=np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]]),
               sortjp='ignore',
               write_output=True,
               plot=False
               ):
    """
    DOC-STRING HERE

    Author
    ------
    Dr. Sunil Anandatheertha
    Example pre-requisites
    ----------------------
    from upxo.pxtal.mcgs import monte_carlo_grain_structure as mcgs
    PXGS = mcgs()
    PXGS.simulate()
    PXGS.detect_grains()
    from upxo.pxtalops.mcgsops2d import find_gb_jp

    Examples
    --------
    gb, jp = find_gb_jp(PXGS.gs[8])
    """
    # --------------------------------------------------
    #          START VALIDATION OF INPUTS
    val = _validation()  # Initiate the validation class
    # VALIDATE gs
    if not val.chk_obj_type(gs, utypes.expec_gs2d_tslice):
        raise TypeError('Invalid grain structure object type.',
                        f'Expected: {utypes.expec_gs2d_tslice}.',
                        f'Received: {gs.__class__.__name__}'
                        )
    # VALIDATE kernel
    if not val.contains_nparray(ttype='gbjp_kernels2d',
                                target=None,
                                sample=kernel):
        raise ValueError('Invalid kernel')
    # VALIDATE sortjp
    if not isinstance(sortjp, str) and sortjp in ('ignore',
                                                  'cw', 'clockwise',
                                                  'acw', 'anticlockwise',
                                                  'ccw', 'counterclockwise',
                                                  ):
        raise ValueError('Invalid sortjp')
    # VALIDATE write_output
    if not isinstance(write_output, bool):
        raise TypeError('Invalid write_output object type',
                        f'Expected: {bool}. ',
                        f'Received: {type(write_output)}')
    # VALIDATE plot
    if not isinstance(plot, bool):
        raise TypeError('Invalid plot object type',
                        f'Expected: {bool}. ',
                        f'Received: {type(write_output)}')

    S = gs.s
    kernel = kernel
    # --------------------------------------------------
    # Label the grains
    # TODO: connectivity information to be stopred as a slot in gs. This is
    # to be automatically retireved here !! <<<---------------
    labeled_grains = label(S, connectivity=2)
    unique_grains = np.unique(labeled_grains)
    # --------------------------------------------------
    # Prepare to collect boundary info for each grain. Exclude background
    grain_boundaries = {grain: [] for grain in unique_grains if grain != 0}
    junction_points = []
    # --------------------------------------------------
    for grain in unique_grains:
        if grain == 0:  # Skip background
            continue

        # Create a mask for the current grain
        grain_mask = labeled_grains == grain

        # Identify the boundary of the grain
        grain_boundary = convolve(grain_mask.astype(int),
                                  kernel,
                                  mode='constant',
                                  cval=0) < kernel.sum()
        grain_boundary = np.logical_and(grain_boundary, grain_mask)

        # Store the boundary pixels for the grain
        boundary_pixels = np.argwhere(grain_boundary)
        grain_boundaries[grain] = boundary_pixels

        # Identify potential junction points (for simplicity, here we
        # consider any boundary pixel with less than 8 neighbors)
        for y, x in boundary_pixels:
            neighbors = labeled_grains[max(y-1, 0):y+2, max(x-1, 0):x+2]
            unique_neighbors = np.unique(neighbors)
            if len(unique_neighbors) - (0 in unique_neighbors) > 2:
                # More than two grains meeting
                junction_points.append((y, x))
    # --------------------------------------------------
    # Unique the junction_points
    junction_points = np.array([list(_) for _ in set(junction_points)])
    # --------------------------------------------------
    """ Find missing junction points on boundary grains """
    # --------------------------------------------------
    # Extract x- and y- coordinates of all juncti0on points
    _x_all, _y_all = junction_points.T[:2]
    jp_xy_all = np.flipud(np.vstack((np.array(junction_points).T[:2])))
    # --------------------------------------------------
    # Output information
    # TODO: INvestigate the empty boundaries !! <<<---------------
    if write_output:
        for grain, boundaries in grain_boundaries.items():
            print(f"Grain {grain} has {len(boundaries)} boundary pixels")
    # --------------------------------------------------
    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(S, cmap='viridis', interpolation='nearest')
        plt.plot(jp_xy_all[0],
                 jp_xy_all[1],
                 'k.',
                 markersize=2,
                 markerfacecolor=None,
                 markeredgecolor='k')
        plt.title('Grain Structure with Boundaries and Junction Points')
        plt.axis('off')
    # --------------------------------------------------
    # Create supporting variables
    length = len(grain_boundaries.keys())
    if length <= 100:
        print_gid = [False if i % 5 != 0 else True for i in range(length)]
    elif length <= 500:
        print_gid = [False if i % 25 != 0 else True for i in range(length)]
    elif length <= 1000:
        print_gid = [False if i % 50 != 0 else True for i in range(length)]
    elif length <= 5000:
        print_gid = [False if i % 100 != 0 else True for i in range(length)]
    else:
        print_gid = [False if i % 500 != 0 else True for i in range(length)]
    # --------------------------------------------------
    # Identify junction points of every grain
    print(20*'-')
    print('Identifying junction points of every grain')
    jp_grainwise = {}
    for gid in grain_boundaries.keys():
        gb = grain_boundaries[gid]
        # Identify those junction point locations which belong to this gb
        gb_jp_loc = list(np.where((gb[:, None] == junction_points).all(-1).any(0))[0])
        # Get the junction point cooerdinates
        if gb_jp_loc:
            gb_jp = [junction_points[i] for i in gb_jp_loc]
            j_y, j_x = zip(*gb_jp)
            jp_grainwise[gid] = np.vstack((j_x, j_y))
        if print_gid[gid-1]:
            print(f'GB JP identified for grain {gid}. N_jp: {len(gb_jp_loc)}')
    # --------------------------------------------------
    # Sort the grain boundary junction points CW or CCW
    if sortjp != 'ignore':
        from upxo.gbops.mcgb2dops import sort_gb_junction_points
        jp_sorted, jp_sorti = sort_gb_junction_points(jp_grainwise, sortjp)
    # --------------------------------------------------
    return grain_boundaries, jp_grainwise
