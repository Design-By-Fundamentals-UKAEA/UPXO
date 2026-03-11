import numpy as np

# Given list of points
L = np.array([[5, 2], [4, 1], [3.5, 1], [1, 2],
              [2, 1], [3, 1], [3, 3], [4, 3],
              [2, 3], [1, 1.25], [1, 2.25], [1.2, 2]
              ])


L = grain_boundaries[370]

# Calculate the centroid of the points
centroid = np.mean(L, axis=0)

# Function to calculate angle and distance from centroid
def angle_and_distance(point, centroid, clockwise=False):
    angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
    if clockwise:
        # To sort clockwise, we can invert the angle
        angle = -angle
    distance = np.linalg.norm(point - centroid)
    return angle, distance

# Function to sort points with an option for direction
def sort_points(points, clockwise=False):
    # Sorting indices based on angles (and distances for collinear points)
    sorted_indices = sorted(range(len(points)), key=lambda i: angle_and_distance(points[i], centroid, clockwise))
    # Sorted points using the sorted indices
    sorted_points = points[sorted_indices]
    return sorted_points, sorted_indices

# Example usage
clockwise_sorted_points, clockwise_sorted_indices = sort_points(L, clockwise=True)
counterclockwise_sorted_points, counterclockwise_sorted_indices = sort_points(L, clockwise=False)

print("Clockwise sorted points:")
print(clockwise_sorted_points)
print("Indices of the clockwise sorted points in the original array:")
print(clockwise_sorted_indices)

print("\nCounterclockwise sorted points:")
print(counterclockwise_sorted_points)
print("Indices of the counterclockwise sorted points in the original array:")
print(counterclockwise_sorted_indices)


sorted_points = clockwise_sorted_points
# Create plot
x_coordinates = sorted_points[:, 0]
y_coordinates = sorted_points[:, 1]

plt.figure(figsize=(6, 6))  # Adjust figure size as needed
plt.scatter(x_coordinates, y_coordinates)
# Add point numbers
for i, (x, y) in enumerate(zip(x_coordinates, y_coordinates)):
    plt.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
# Customize plot
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Scatter Plot with Point Numbers")
plt.grid(True)

# Show plot
plt.show()
