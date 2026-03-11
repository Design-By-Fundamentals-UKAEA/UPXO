import numpy as np
L = grain_boundaries[370]


def find_extreme_points(points, loc):
    if loc == 'tl':  # Top-Left Point
        point = points[np.lexsort((points[:, 0], points[:, 1]))][0]
    elif loc == 'tr':  # Top-Right Point
        point = points[np.lexsort((-points[:, 0], points[:, 1]))][0]
    elif loc == 'br':  # Bottom-Right Point
        point = points[np.lexsort((-points[:, 0], -points[:, 1]))][0]
    elif loc == 'bl':  # Bottom-Left Point
        point = points[np.lexsort((points[:, 0], -points[:, 1]))][0]
    return point


expoint = find_extreme_points(L, 'tl')


x_coordinates, y_coordinates = L[:, 1], L[:, 0]

plt.figure(figsize=(6, 6))  # Adjust figure size as needed
plt.scatter(x_coordinates, y_coordinates)
plt.plot(expoint[0], expoint[1], 'rx')
for i, (x, y) in enumerate(zip(x_coordinates, y_coordinates)):
    plt.annotate(str(i + 1), (x, y),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Scatter Plot with Point Numbers")
plt.grid(True)
plt.show()
