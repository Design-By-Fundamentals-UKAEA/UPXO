# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:10:12 2024

@author: rg5749
"""

import numpy as np

def calculate_schmid_factor(loading_axis, K1, eta1):
    """Calculates the Schmid factor for a given twin variant.

    Args:
        loading_axis: The loading axis as a 3D unit vector (e.g., [1, 0, 0] for [100] direction).
        K1: The twin plane normal as a 3D unit vector.
        eta1: The twinning shear direction as a 3D unit vector.

    Returns:
        The Schmid factor (a float value).
    """
    cos_phi = np.abs(np.dot(loading_axis, K1))
    cos_lambda = np.abs(np.dot(loading_axis, eta1))
    return cos_phi * cos_lambda

def calculate_crss(stress, schmid_factor):
    """Calculates the critical resolved shear stress (CRSS).

    Args:
        stress: The applied stress at twinning initiation.
        schmid_factor: The Schmid factor of the twin variant.

    Returns:
        The CRSS (a float value).
    """
    return stress * schmid_factor


"""#
Example Usage:
loading_axis = np.array([1, 0, 0])  # [100] loading direction
K1 = np.array([1, 1, 1]) / np.sqrt(3)  # {111} plane normal
eta1 = np.array([1, 1, -2]) / np.sqrt(6)  # [11-2] twinning direction

schmid_factor = calculate_schmid_factor(loading_axis, K1, eta1)
print(f"Schmid factor: {schmid_factor:.3f}")

# Assume experimentally determined stress at twinning initiation
stress = 100  # MPa

crss = calculate_crss(stress, schmid_factor)
print(f"CRSS: {crss:.3f} MPa")
"""

def calculate_all_crss(loading_axis, twin_variants, stress):
    """Calculates CRSS for all twin variants and returns a dictionary.

    Args:
        loading_axis: The loading axis as a 3D unit vector (e.g., [1, 0, 0] for [100] direction).
        twin_variants: A list of tuples, each containing (K1, eta1) for a twin variant,
                       where K1 and eta1 are 3D unit vectors.
        stress: The applied stress at twinning initiation.

    Returns:
        A dictionary where keys are twin variant indices (1-based) and values are dictionaries
        with keys 'K1', 'eta1', and 'CRSS', containing the corresponding values.
    """
    crss_dict = {}
    for i, (K1, eta1) in enumerate(twin_variants, start=1):  # 1-based indexing
        schmid_factor = calculate_schmid_factor(loading_axis, K1, eta1)
        crss = calculate_crss(stress, schmid_factor)
        crss_dict[i] = {'K1': K1.tolist(), 'eta1': eta1.tolist(), 'CRSS': crss}
    return crss_dict

"""
# Example Usage (FCC {111} Twinning System)
loading_axis = np.array([1, 0, 0])  # [100] loading direction
import numpy as np

# Define unique K1 planes
k1_planes = [np.array([1, 1, 1]), np.array([1, -1, 1]),
             np.array([-1, 1, 1]), np.array([-1, -1, 1])]
eta1_directions = [np.array([1, 1, -2]),
                   np.array([-1, -2, 1]),
                   np.array([-2, -1, 1])]
# Generate all twin variants by combining K1 planes with eta1 directions
twin_variants = [(k1 / np.sqrt(3), eta1 / np.sqrt(6))
                 for k1 in k1_planes
                 for eta1 in eta1_directions]

stress = 100  # MPa (replace with experimental value)

result_dict = calculate_all_crss(loading_axis, twin_variants, stress)
print(result_dict)
"""


def euler_to_rotation_matrix(phi1, Phi, phi2, degrees=False):
    """Converts Bunge Euler angles to a rotation matrix.

    Args:
        phi1, Phi, phi2: Bunge Euler angles (in radians by default).
        degrees: If True, input angles are interpreted as degrees.

    Returns:
        A 3x3 NumPy array representing the rotation matrix.
    """
    if degrees:
        phi1, Phi, phi2 = np.radians([phi1, Phi, phi2])  # Convert to radians

    c1, s1 = np.cos(phi1), np.sin(phi1)
    c2, s2 = np.cos(Phi), np.sin(Phi)
    c3, s3 = np.cos(phi2), np.sin(phi2)

    return np.array([
        [c1*c3 - s1*c2*s3, s1*c3 + c1*c2*s3, s2*s3],
        [-c1*s3 - s1*c2*c3, -s1*s3 + c1*c2*c3, s2*c3],
        [s1*s2, -c1*s2, c2]
    ])


def miller_to_cartesian(miller_indices, R):
    """Converts Miller indices to a Cartesian vector in the grain's frame.

    Args:
        miller_indices: A list or tuple of Miller indices (h, k, l).
        R: The rotation matrix representing the grain's orientation.

    Returns:
        A NumPy array representing the Cartesian vector.
    """
    B = np.linalg.inv(R) @ np.array(miller_indices)  # Reciprocal lattice vector in grain frame
    return B / np.linalg.norm(B)  # Normalize to unit vector


def calculate_activated_twins(loading_axis_miller, euler_angles, twin_variants, stress):
    """Calculates activated twin variants and their morphological orientations."""

    R = euler_to_rotation_matrix(*euler_angles)
    loading_axis_grain = miller_to_cartesian(loading_axis_miller, R)

    activated_twins = []
    for i, (K1, eta1) in enumerate(twin_variants, start=1):
        K1_grain = np.dot(R, K1)  # Transform K1 to grain's frame
        eta1_grain = np.dot(R, eta1)  # Transform eta1 to grain's frame

        schmid_factor = calculate_schmid_factor(loading_axis_grain, K1_grain, eta1_grain)
        crss = calculate_crss(stress, schmid_factor)

        if schmid_factor >= 0.408:  # Threshold for FCC {111} system
            activated_twins.append(
                {
                    "variant": i,
                    "K1": K1_grain.tolist(),
                    "eta1": eta1_grain.tolist(),
                    "CRSS": crss,
                }
            )

    return activated_twins


def calculate_twin_plane_normal(K1_grain, euler_angles):
    """Calculates the normal unit vector of a twin plane and its projections.

    Args:
        K1_grain: The twin plane normal in the grain's frame (a 3D NumPy array).
        euler_angles: The Bunge Euler angles of the grain (phi1, Phi, phi2).

    Returns:
        A dictionary containing:
            'normal': The twin plane normal unit vector in the grain structure's frame.
            'proj_xy': Projection of the normal onto the xy plane.
            'proj_yz': Projection of the normal onto the yz plane.
            'proj_zx': Projection of the normal onto the zx plane.
    """
    R_grain_to_sample = euler_to_rotation_matrix(*euler_angles, degrees=True)
    normal_sample = np.dot(R_grain_to_sample, K1_grain)  # Transform to sample frame

    # Calculate projections onto planes
    proj_xy = normal_sample.copy()
    proj_xy[2] = 0  # Set z-component to zero
    proj_xy /= np.linalg.norm(proj_xy)  # Normalize

    proj_yz = normal_sample.copy()
    proj_yz[0] = 0  # Set x-component to zero
    proj_yz /= np.linalg.norm(proj_yz)

    proj_zx = normal_sample.copy()
    proj_zx[1] = 0  # Set y-component to zero
    proj_zx /= np.linalg.norm(proj_zx)

    return {
        'normal': normal_sample,
        'proj_xy': proj_xy,
        'proj_yz': proj_yz,
        'proj_zx': proj_zx
    }


k1_planes = [np.array([1, 1, 1]), np.array([1, -1, 1]),
             np.array([-1, 1, 1]), np.array([-1, -1, 1])]
eta1_directions = [np.array([1, 1, -2]),
                   np.array([-1, -2, 1]),
                   np.array([-2, -1, 1])]
twin_variants = [(k1 / np.sqrt(3), eta1 / np.sqrt(6))
                 for k1 in k1_planes
                 for eta1 in eta1_directions]


loading_axis_miller = [1, 0, 0]   # [100] loading direction
euler_angles = (0, 0, 0)       # Example Euler angles for the grain
activated_twins = calculate_activated_twins(
    loading_axis_miller, euler_angles, twin_variants, stress=100  # MPa
)
print(activated_twins)
activated_twins[0]


calculate_twin_plane_normal(activated_twins[0]['K1'], euler_angles)
