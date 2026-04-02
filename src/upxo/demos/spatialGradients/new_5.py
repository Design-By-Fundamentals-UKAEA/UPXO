import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import pyvista as pv
from skimage.measure import label
from math import floor
# ====================================================================
# Small grid dimensions for demonstration
nx, ny, nz = 40, 30, 30
NSTATES = 8 # Number of states
NITERATIONS = 50  # Number of iterations
# ====================================================================
TFIELD = 0.00001*np.ones((nx, ny, nz))
tspecs_x = {0: [[0, 10], [0.00001, 0.00001] ],
		    1: [[11, 15], [0.0001, 0.0001] ],
            2: [[16, 20], [20, 20] ],
            3: [[21, nx], [0.001, 0.00001] ], }
tspecs_y = {0: [[0, 5], [0.00001, 0.0001] ],
		    1: [[6, 12], [0.001, 5] ],
            2: [[13, ny], [3, 0.00001] ], }
tspecs_z = {0: [[0, 10], [0.00001, 0.0001] ],
		    1: [[11, 18], [0.001, 2] ],
            2: [[19, nz], [3, 0.00001] ], }



sfreezing = {3: [5, 6, 7, 8]}

for __ in tspecs_x.keys():
    loc_bounds = tspecs_x[__][0]
    temp_bounds = tspecs_x[__][1]
    locations = np.arange(loc_bounds[0], loc_bounds[1])
    temp_1d = np.interp(locations, loc_bounds, temp_bounds)
    for loc, temp in zip(locations, temp_1d):
        TFIELD[loc, :, :] += temp
for __ in tspecs_y.keys():
    loc_bounds = tspecs_y[__][0]
    temp_bounds = tspecs_y[__][1]
    locations = np.arange(loc_bounds[0], loc_bounds[1])
    temp_1d = np.interp(locations, loc_bounds, temp_bounds)
    for loc, temp in zip(locations, temp_1d):
        TFIELD[:, loc, :] += temp
for __ in tspecs_z.keys():
    loc_bounds = tspecs_z[__][0]
    temp_bounds = tspecs_z[__][1]
    locations = np.arange(loc_bounds[0], loc_bounds[1])
    temp_1d = np.interp(locations, loc_bounds, temp_bounds)
    for loc, temp in zip(locations, temp_1d):
        TFIELD[:, :, loc] += temp

x, y, z = np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1)
grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing='ij'))
scalar_data = TFIELD.flatten(order="F")[:]  # Flatten and adjust size if necessary
grid.cell_data["TFIELD"] = scalar_data  # Use 'cell_data' instead of 'cell_arrays'

plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="viridis", show_edges=True)
plotter.show_axes()
plotter.show()
# ====================================================================

# ====================================================================

# Initialize the state matrix with random states
state_matrix = np.random.randint(1, NSTATES + 1, size=(nx, ny, nz))
hamiltonians = np.zeros(NITERATIONS)  # Array to store Hamiltonian values for all iterations
STRUCTURE = np.array(list(product([-1, 0, 1], repeat=3)))
# Remove the cenrtal pixel as it is not needed anywhere
STRUCTURE = STRUCTURE[~np.all(STRUCTURE == [0, 0, 0], axis=1)]
# ===========================================================
wm = np.ones((3, 3, 3))
# ===========================================================
# BUILD THE SKIP-ITERATION-SF TEMPORAL ARRAY
SkipIter_sf = np.zeros((NSTATES, NITERATIONS), dtype=bool)
for s in sfreezing.keys():
    if sfreezing[s]:
        for t in sfreezing[s]:
            SkipIter_sf[s, t] = True
# ===========================================================
skip_iteration_ssn = [False for _ in STRUCTURE]
# ===========================================================
# Main simulation loop
for iteration in range(NITERATIONS):
    HAM = np.zeros((nx, ny, nz))
    print(f'Iteration: {iteration}')
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # --------------------------------
                skip_iteration_sf = False
                skip_iteration_ssn_ = False
                # --------------------------------
                current_state = state_matrix[x, y, z]
                # --------------------------------
                # iteration skip due to Frozen State
                if SkipIter_sf[current_state-1, iteration]:
                    skip_iteration_sf = True
                    HAM[x, y, z] = np.nan
                # --------------------------------
                # iteration skip due to Same State Neighbours
                for i, (dx, dy, dz) in enumerate(STRUCTURE):
                    if current_state == state_matrix[(x + dx) % nx,
                                                     (y + dy) % ny,
                                                     (z + dz) % nz]:
                        skip_iteration_ssn[i] = True
                if all(skip_iteration_ssn):
                    skip_iteration_ssn_ = True
                # --------------------------------
                if skip_iteration_sf or skip_iteration_ssn_:
                    # Calculate energy at site
                    E1, E2 = [], []
                    for dx, dy, dz in STRUCTURE:
                        x2, y2, z2 = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                        E1.append(wm[dx+1, dy+1, dz+1]*(current_state == state_matrix[x2, y2, z2]))
                    # --------------------------------
                    # Flip the state
                    new_state = np.random.randint(1, NSTATES + 1)
                    while new_state == current_state:
                        new_state = np.random.randint(1, NSTATES + 1)
                    # --------------------------------
                    # Calculate new energy at site
                    for dx, dy, dz in STRUCTURE:
                        x2, y2, z2 = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                        E2.append(wm[dx+1, dy+1, dz+1]*(new_state == state_matrix[x2, y2, z2]))
                    # --------------------------------
                    # Compare the energies
                    E1_total, E2_total = 26-sum(E1), 26-sum(E2)
                    delE = E2_total - E1_total
                    # --------------------------------
                    # Accept or reject the new state
                    temperature = TFIELD[x, y, z]
                    if delE < 0 or np.random.rand() < np.exp(-delE / temperature):
                        state_matrix[x, y, z] = new_state
                        HAM[x, y, z] = E2_total
                    else:
                        HAM[x, y, z] = E1_total
    hamiltonians[iteration] = HAM.sum()


# Adjust the scalar data to match the number of cells
scalar_data = state_matrix.flatten(order="F")[:]  # Flatten and adjust size if necessary
# Assign the scalar values to the grid
grid.cell_data["mcstates"] = scalar_data  # Use 'cell_data' instead of 'cell_arrays'

'''
# Adjust the scalar data to match the number of cells
scalar_data = labeled_grains.flatten(order="F")[:]  # Flatten and adjust size if necessary
# Assign the scalar values to the grid
grid.cell_data["gid"] = scalar_data  # Use 'cell_data' instead of 'cell_arrays'
'''

# Make sure to tell PyVista what the active scalars are for volume rendering
grid.set_active_scalars("mcstates")


plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="cividis", show_edges=True)
plotter.show()



plotter = pv.Plotter(window_size=[400, 400])
plotter.add_mesh(pv.Box(bounds=[-1, nx, -1, ny, -1, nz]),
                 color='red',
                 style='wireframe',
                 line_width=5)
plotter.show_bounds(grid="front", location="outer", all_edges=True)
gsslice=grid.slice_orthogonal(x=3, y=12, z=12)
plotter.add_mesh(gsslice)
plotter.show()
