"""
pole_figure.py
==============
Self-sufficient advanced pole figure plotting and analysis module for UPXO.
Supports stereographic and equal-area projections, custom coloring (hemisphere,
role, size, IPF), density contouring (MUD), and 3D slice integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, Tuple, List, Any


# Default role colors (compatible with twinned_simple_3d)
_DEFAULT_ROLE_COLOR = {
    'non_host':      'lightgray',
    'host':          'steelblue',
    'primary_twin':   'darkorange',
    'secondary_twin': 'crimson',
}

_NUMBA_VMF_KERNEL = None



def cubic_symmetry_operators() -> np.ndarray:
    """
    Generate the 24 proper rotations for cubic symmetry (m-3m) as 3x3 matrices.
    """
    from itertools import permutations, product
    ops = []
    for p in permutations(range(3)):
        P = np.eye(3)[list(p)]
        for signs in product([-1, 1], repeat=3):
            S = P * np.array(signs)[None, :]
            if round(np.linalg.det(S)) == 1:
                ops.append(S.astype(np.float64))
    # Deduplicate
    unique_ops = []
    for op in ops:
        if not any(np.allclose(op, u) for u in unique_ops):
            unique_ops.append(op)
    return np.array(unique_ops)


def hexagonal_symmetry_operators() -> np.ndarray:
    """
    Generate the 12 proper rotations for hexagonal symmetry (6/mmm) as 3x3 matrices.
    """
    ops = []
    # 6-fold rotation axis about Z
    for i in range(6):
        angle = i * np.pi / 3
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        ops.append(R)
    # 2-fold rotation axes perpendicular to Z
    for i in range(6):
        angle = i * np.pi / 6
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [2*c*c - 1, 2*c*s, 0],
            [2*c*s, 2*s*s - 1, 0],
            [0, 0, -1]
        ])
        ops.append(R)
    # Deduplicate
    unique_ops = []
    for op in ops:
        if not any(np.allclose(op, u, atol=1e-6) for u in unique_ops):
            unique_ops.append(op)
    return np.array(unique_ops)


def tetragonal_symmetry_operators() -> np.ndarray:
    """
    Generate the 8 proper rotations for tetragonal symmetry (4/mmm) as 3x3 matrices.
    """
    ops = []
    # 4-fold rotation axis about Z
    for i in range(4):
        angle = i * np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        ops.append(R)
    # 2-fold rotation axes perpendicular to Z
    for i in range(4):
        angle = i * np.pi / 4
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [2*c*c - 1, 2*c*s, 0],
            [2*c*s, 2*s*s - 1, 0],
            [0, 0, -1]
        ])
        ops.append(R)
    # Deduplicate
    unique_ops = []
    for op in ops:
        if not any(np.allclose(op, u, atol=1e-6) for u in unique_ops):
            unique_ops.append(op)
    return np.array(unique_ops)


def trigonal_symmetry_operators() -> np.ndarray:
    """
    Generate the 6 proper rotations for trigonal symmetry (32) as 3x3 matrices.
    """
    ops = []
    # 3-fold rotation axis about Z
    for i in range(3):
        angle = i * 2 * np.pi / 3
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        ops.append(R)
    # 2-fold rotation axes perpendicular to Z
    for i in range(3):
        angle = i * np.pi / 3
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [2*c*c - 1, 2*c*s, 0],
            [2*c*s, 2*s*s - 1, 0],
            [0, 0, -1]
        ])
        ops.append(R)
    # Deduplicate
    unique_ops = []
    for op in ops:
        if not any(np.allclose(op, u, atol=1e-6) for u in unique_ops):
            unique_ops.append(op)
    return np.array(unique_ops)


def orthorhombic_symmetry_operators() -> np.ndarray:
    """
    Generate the 4 proper rotations for orthorhombic symmetry (222) as 3x3 matrices.
    """
    return np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ], dtype=np.float64)


def triclinic_symmetry_operators() -> np.ndarray:
    """
    Generate the identity rotation for triclinic symmetry (1) as 3x3 matrices.
    """
    return np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float64)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert a stack of 3x3 rotation matrices to unit quaternions [w, x, y, z] with w >= 0.
    """
    R = np.atleast_3d(R)
    N = R.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    for i in range(N):
        t = tr[i]
        mat = R[i]
        if t > 0:
            S = np.sqrt(t + 1.0) * 2.0
            qw = 0.25 * S
            qx = (mat[2, 1] - mat[1, 2]) / S
            qy = (mat[0, 2] - mat[2, 0]) / S
            qz = (mat[1, 0] - mat[0, 1]) / S
        elif (mat[0, 0] > mat[1, 1]) and (mat[0, 0] > mat[2, 2]):
            S = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
            qw = (mat[2, 1] - mat[1, 2]) / S
            qx = 0.25 * S
            qy = (mat[0, 1] + mat[1, 0]) / S
            qz = (mat[0, 2] + mat[2, 0]) / S
        elif mat[1, 1] > mat[2, 2]:
            S = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
            qw = (mat[0, 2] - mat[2, 0]) / S
            qx = (mat[0, 1] + mat[1, 0]) / S
            qy = 0.25 * S
            qz = (mat[1, 2] + mat[2, 1]) / S
        else:
            S = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
            qw = (mat[1, 0] - mat[0, 1]) / S
            qx = (mat[0, 2] + mat[2, 0]) / S
            qy = (mat[1, 2] + mat[2, 1]) / S
            qz = 0.25 * S
        q[i] = [qw, qx, qy, qz]

    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / (norm + 1e-12)
    neg_w = q[:, 0] < 0
    q[neg_w] *= -1
    return q.squeeze()


def compute_sample_symmetry_group(
    apply_sample_symmetries: bool = True,
    use_rd: bool = True,
    use_td: bool = True,
    use_nd: bool = True,
) -> List[np.ndarray]:
    """
    Compute the mathematically closed group of the selected macroscopic
    sample-symmetry generators: 180-degree rotations about the Rolling (RD),
    Transverse (TD), and Normal (ND) sample directions. Returns a list of
    3x3 matrices (always includes the identity).
    """
    generators = []
    if apply_sample_symmetries:
        if use_rd:
            generators.append(np.diag([1.0, -1.0, -1.0]))
        if use_td:
            generators.append(np.diag([-1.0, 1.0, -1.0]))
        if use_nd:
            generators.append(np.diag([-1.0, -1.0, 1.0]))

    group = [np.eye(3)]
    for gen in generators:
        current_elements = list(group)
        for elem in current_elements:
            product = gen @ elem
            if not any(np.allclose(product, item) for item in group):
                group.append(product)
    return group


def _euler_deg_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """Bunge ZXZ (phi1, Phi, phi2) in degrees -> (N, 3, 3) rotation matrices."""
    data = np.atleast_2d(np.asarray(euler_deg, dtype=np.float64))
    phi1, Phi, phi2 = np.deg2rad(data[:, 0]), np.deg2rad(data[:, 1]), np.deg2rad(data[:, 2])
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    zeros, ones = np.zeros_like(c1), np.ones_like(c1)
    Rz1 = np.array([[c1, -s1, zeros], [s1, c1, zeros], [zeros, zeros, ones]]).transpose(2, 0, 1)
    Rx = np.array([[ones, zeros, zeros], [zeros, c, -s], [zeros, s, c]]).transpose(2, 0, 1)
    Rz2 = np.array([[c2, -s2, zeros], [s2, c2, zeros], [zeros, zeros, ones]]).transpose(2, 0, 1)
    return Rz1 @ Rx @ Rz2


def matrix_to_euler_bunge(R: np.ndarray) -> np.ndarray:
    """
    Convert a stack of 3x3 rotation matrices (Bunge ZXZ convention, i.e. the
    R = Rz1 @ Rx @ Rz2 form used throughout this module) to Bunge Euler angles
    (phi1, Phi, phi2) in degrees, shape (N, 3) (or (3,) for a single matrix).
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        R = R[None, :, :]
    N = R.shape[0]

    Phi = np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0))
    sin_Phi = np.sin(Phi)
    phi1 = np.zeros(N)
    phi2 = np.zeros(N)

    reg = sin_Phi > 1e-7
    phi1[reg] = np.arctan2(R[reg, 0, 2], -R[reg, 1, 2])
    phi2[reg] = np.arctan2(R[reg, 2, 0], R[reg, 2, 1])

    deg = ~reg
    if np.any(deg):
        # Gimbal lock (Phi == 0 or pi): only (phi1 +/- phi2) is determined;
        # follow the same convention as the rest of this module -- fix phi2=0.
        phi2[deg] = 0.0
        phi1[deg] = np.arctan2(R[deg, 1, 0], R[deg, 0, 0])

    euler = np.rad2deg(np.stack([phi1, Phi, phi2], axis=-1))
    euler[:, 0] = np.mod(euler[:, 0], 360.0)
    euler[:, 2] = np.mod(euler[:, 2], 360.0)
    return euler.squeeze()


def apply_sample_symmetry(
    orientations: np.ndarray,
    convention: str,
    group_ops: List[np.ndarray],
) -> np.ndarray:
    """
    Replicate each orientation through every element of a closed sample-symmetry
    group (see compute_sample_symmetry_group), returning the concatenated set of
    replicas in the same convention as the input.

    This is the standard way real (already-fixed) orientation data is
    symmetrized for sample-symmetric pole figures/densities: each measured
    orientation g is left-multiplied by every sample-symmetry operator M
    (g' = M @ g), since M acts on the *sample* frame the orientation maps into
    -- this is distinct from FCCTexture's synthetic-generation use of sample
    symmetry, which instead samples new points around the symmetric-equivalent
    *cluster centers* of an ideal texture component.
    """
    data = np.asarray(orientations)
    if convention == 'matrix':
        R = data if data.ndim == 3 else data[None, :, :]
    elif convention in ('euler_deg', 'euler_rad'):
        euler_deg_data = np.rad2deg(np.atleast_2d(data)) if convention == 'euler_rad' \
            else np.atleast_2d(data)
        R = _euler_deg_to_matrix(euler_deg_data)
    elif convention == 'quaternion':
        q = np.atleast_2d(data)
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / (norm + 1e-12)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.zeros((q.shape[0], 3, 3), dtype=np.float64)
        R[:, 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
        R[:, 0, 1] = 2.0 * (x*y - w*z)
        R[:, 0, 2] = 2.0 * (x*z + w*y)
        R[:, 1, 0] = 2.0 * (x*y + w*z)
        R[:, 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
        R[:, 1, 2] = 2.0 * (y*z - w*x)
        R[:, 2, 0] = 2.0 * (x*z - w*y)
        R[:, 2, 1] = 2.0 * (y*z + w*x)
        R[:, 2, 2] = 1.0 - 2.0 * (x**2 + y**2)
    else:
        raise ValueError(f"Invalid convention: {convention}")

    replicas = [np.einsum('ij,njk->nik', op, R) for op in group_ops]
    R_all = np.concatenate(replicas, axis=0)

    if convention == 'matrix':
        return R_all
    elif convention == 'quaternion':
        return matrix_to_quat(R_all)
    elif convention == 'euler_rad':
        return np.deg2rad(matrix_to_euler_bunge(R_all))
    else:  # euler_deg
        return matrix_to_euler_bunge(R_all)


class PoleFigure:
    """
    Advanced self-sufficient pole figure generator and density analyzer.
    """
    
    FCC_POLES = {
        '100': np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]),
        '110': np.array([[1,1,0], [-1,-1,0], [1,-1,0], [-1,1,0], [1,0,1], [-1,0,-1],
                         [1,0,-1], [-1,0,1], [0,1,1], [0,-1,-1], [0,1,-1], [0,-1,1]]),
        '111': np.array([[1,1,1], [-1,-1,-1], [1,-1,-1], [-1,1,1], [1,1,-1], [-1,-1,1],
                         [-1,1,-1], [1,-1,1]])
    }
    
    SYM_OPS = cubic_symmetry_operators()

    def __init__(
        self,
        orientations: np.ndarray,
        convention: str = 'quaternion',  # 'quaternion' | 'euler_deg' | 'euler_rad' | 'matrix'
        gids: Optional[np.ndarray] = None,
        symmetry: Union[str, np.ndarray] = 'cubic'
    ):
        """
        Initialize the PoleFigure object.

        Parameters
        ----------
        orientations : np.ndarray
            Orientation data. Shapes:
            - 'quaternion': (N, 4)
            - 'euler_deg' or 'euler_rad': (N, 3)
            - 'matrix': (N, 3, 3)
        convention : str
            Convention type of orientations.
        gids : np.ndarray, optional
            Grain IDs corresponding to the orientations, used for metadata-based coloring.
        symmetry : str or np.ndarray
            Crystal symmetry to use. Either a string ('cubic', 'hexagonal', 'tetragonal', 
            'trigonal', 'orthorhombic', 'triclinic') or a custom array of shape (M, 3, 3).
        """
        self.gids = np.asarray(gids) if gids is not None else None
        
        # Store or build symmetry operators stack
        if isinstance(symmetry, str):
            sym_key = symmetry.lower()
            if sym_key == 'cubic':
                self.symmetry_ops = cubic_symmetry_operators()
            elif sym_key == 'hexagonal':
                self.symmetry_ops = hexagonal_symmetry_operators()
            elif sym_key == 'tetragonal':
                self.symmetry_ops = tetragonal_symmetry_operators()
            elif sym_key == 'trigonal':
                self.symmetry_ops = trigonal_symmetry_operators()
            elif sym_key == 'orthorhombic':
                self.symmetry_ops = orthorhombic_symmetry_operators()
            elif sym_key == 'triclinic':
                self.symmetry_ops = triclinic_symmetry_operators()
            else:
                raise ValueError(
                    f"Unknown symmetry key '{symmetry}'. Choose from: "
                    "'cubic', 'hexagonal', 'tetragonal', 'trigonal', 'orthorhombic', 'triclinic'."
                )
        else:
            self.symmetry_ops = np.asarray(symmetry, dtype=np.float64)
            if self.symmetry_ops.ndim != 3 or self.symmetry_ops.shape[1:] != (3, 3):
                raise ValueError("Custom symmetry operators must be a stack of matrices of shape (M, 3, 3).")

        # Convert orientation inputs to standardized rotation matrix stacks (N, 3, 3)
        # representing g^T (crystal-to-sample rotation).
        self.R_stack = self._standardize_orientations(orientations, convention)
        
        # Keep quaternions for IPF coloring
        if convention == 'quaternion':
            self.quats = orientations
        else:
            self.quats = matrix_to_quat(self.R_stack)


    def _standardize_orientations(self, data: np.ndarray, convention: str) -> np.ndarray:
        """Standardize input orientations into (N, 3, 3) rotation matrices."""
        data = np.asarray(data)
        if convention == 'matrix':
            if data.ndim == 2 and data.shape == (3, 3):
                return data[None, :, :]
            return data
        
        elif convention in ('euler_deg', 'euler_rad'):
            if data.ndim == 1:
                data = data[None, :]
            phi1 = data[:, 0]
            Phi = data[:, 1]
            phi2 = data[:, 2]
            if convention == 'euler_deg':
                phi1, Phi, phi2 = np.deg2rad([phi1, Phi, phi2])
                
            c1, s1 = np.cos(phi1), np.sin(phi1)
            c, s = np.cos(Phi), np.sin(Phi)
            c2, s2 = np.cos(phi2), np.sin(phi2)
            
            zeros = np.zeros_like(c1)
            ones = np.ones_like(c1)
            
            Rz1 = np.array([[c1, -s1, zeros], [s1, c1, zeros], [zeros, zeros, ones]]).transpose(2, 0, 1)
            Rx = np.array([[ones, zeros, zeros], [zeros, c, -s], [zeros, s, c]]).transpose(2, 0, 1)
            Rz2 = np.array([[c2, -s2, zeros], [s2, c2, zeros], [zeros, zeros, ones]]).transpose(2, 0, 1)
            
            return Rz1 @ Rx @ Rz2
            
        elif convention == 'quaternion':
            if data.ndim == 1:
                data = data[None, :]
            # Normalize quaternions just in case
            norm = np.linalg.norm(data, axis=1, keepdims=True)
            q = data / (norm + 1e-12)
            
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            
            # Construct R^T (crystal-to-sample)
            R_T = np.zeros((q.shape[0], 3, 3), dtype=np.float64)
            R_T[:, 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
            R_T[:, 0, 1] = 2.0 * (x*y - w*z)
            R_T[:, 0, 2] = 2.0 * (x*z + w*y)
            
            R_T[:, 1, 0] = 2.0 * (x*y + w*z)
            R_T[:, 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
            R_T[:, 1, 2] = 2.0 * (y*z - w*x)
            
            R_T[:, 2, 0] = 2.0 * (x*z - w*y)
            R_T[:, 2, 1] = 2.0 * (y*z + w*x)
            R_T[:, 2, 2] = 1.0 - 2.0 * (x**2 + y**2)
            return R_T
        else:
            raise ValueError(f"Invalid convention: {convention}")

    def _get_symmetric_poles(self, hkl: Union[str, List[int], Tuple[int, int, int]]) -> np.ndarray:
        """Returns the normalized coordinates for the symmetric family of poles (antipodal-deduplicated)."""
        if isinstance(hkl, str):
            if hkl in self.FCC_POLES:
                raw_poles = self.FCC_POLES[hkl]
            elif len(hkl) == 3 and hkl.isdigit():
                seed = np.array([float(hkl[0]), float(hkl[1]), float(hkl[2])])
                seed /= np.linalg.norm(seed) + 1e-12
                raw_poles = np.einsum('nij,j->ni', self.symmetry_ops, seed)
            else:
                raise ValueError(
                    f"Unknown pole family '{hkl}'. Choose from '100', '110', '111', "
                    "or provide a 3-digit numeric string (e.g. '123') or a list/tuple."
                )
        else:
            # Build custom pole family dynamically by applying proper symmetries to a seed vector
            seed = np.asarray(hkl, dtype=np.float64)
            seed /= np.linalg.norm(seed) + 1e-12
            # Apply the proper rotations to seed vector
            raw_poles = np.einsum('nij,j->ni', self.symmetry_ops, seed)


        
        # Deduplicate vectors (treating v and -v as duplicate axes to avoid overlapping points in projection)
        unique_poles = []
        for p in raw_poles:
            p_norm = p / (np.linalg.norm(p) + 1e-12)
            if not any(np.allclose(p_norm, u, atol=1e-5) or np.allclose(-p_norm, u, atol=1e-5) for u in unique_poles):
                unique_poles.append(p_norm)
        return np.array(unique_poles)


    def _calculate_projection(
        self,
        poles: np.ndarray,
        projection: str = 'stereographic',
        hemisphere: str = 'both_mapped'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms poles into the sample coordinate system and projects them.
        Returns X, Y coordinates, and original orientation indices.
        """
        # Transform poles to sample system: sample_dirs shape (N_orientations, N_poles, 3)
        sample_dirs = np.einsum('nij,kj->nki', self.R_stack, poles)
        N_ori, N_poles, _ = sample_dirs.shape
        
        # Flatten orientations and poles together
        sample_dirs = sample_dirs.reshape(-1, 3)
        ori_indices = np.repeat(np.arange(N_ori), N_poles)
        
        # Hemisphere logic
        if hemisphere == 'upper':
            mask = sample_dirs[:, 2] >= 0
            dirs = sample_dirs[mask]
            ori_indices = ori_indices[mask]
        elif hemisphere == 'lower':
            mask = sample_dirs[:, 2] < 0
            dirs = -sample_dirs[mask]
            ori_indices = ori_indices[mask]
        elif hemisphere in ('both_mapped', 'both_separate'):
            # Map all points to upper hemisphere: invert vectors pointing down
            neg_z = sample_dirs[:, 2] < 0
            dirs = sample_dirs.copy()
            dirs[neg_z] *= -1.0
        else:
            raise ValueError(f"Invalid hemisphere configuration: {hemisphere}")
            
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        
        if projection == 'stereographic':
            d = 1.0 / (1.0 + z)
        elif projection == 'equal_area':
            # Normalized Lambert Equal-Area projection (scaled by 1/sqrt(2) to fit unit circle)
            d = np.sqrt(1.0 / (1.0 + z))
        else:
            raise ValueError(f"Unknown projection: {projection}")
            
        X = x * d
        Y = y * d
        return X, Y, ori_indices

    def _calculate_ipf_colors(
        self,
        ipf_direction: Tuple[float, float, float] = (0., 0., 1.)
    ) -> np.ndarray:
        """
        Calculate Inverse Pole Figure (IPF) RGB colors for each orientation.
        """
        sd = np.asarray(ipf_direction, dtype=np.float64)
        sd /= np.linalg.norm(sd) + 1e-12
        
        w = self.quats[:, 0]
        x = self.quats[:, 1]
        y = self.quats[:, 2]
        z = self.quats[:, 3]
        
        s0, s1, s2 = sd
        v0 = (1 - 2*(y*y + z*z))*s0 + 2*(x*y + w*z)*s1 + 2*(x*z - w*y)*s2
        v1 = 2*(x*y - w*z)*s0 + (1 - 2*(x*x + z*z))*s1 + 2*(y*z + w*x)*s2
        v2 = 2*(x*z + w*y)*s0 + 2*(y*z - w*x)*s1 + (1 - 2*(x*x + y*y))*s2
        
        v = np.stack([np.abs(v0), np.abs(v1), np.abs(v2)], axis=-1)
        vmx = v.max(axis=-1, keepdims=True)
        vmx = np.where(vmx == 0, 1.0, vmx)
        rgb = np.clip(v / vmx, 0.0, 1.0)
        return rgb

    def _draw_axis_crosshairs(
        self,
        ax: plt.Axes,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        z_label: Optional[str] = None,
    ) -> None:
        """
        Optional reference markers for the sample X/Y/Z directions. X and Y
        are in-plane (this pole figure's horizontal/vertical screen axes map
        directly onto sample +X/+Y -- an unflipped, right-handed
        correspondence), so each gets a full crosshair line through the
        circle with its label at the positive end. Z is the out-of-plane
        / viewing axis -- it has no in-plane position to mark, so it is
        noted as a small corner label instead. All three default to None
        (nothing drawn), so callers that don't pass them see no change.
        """
        line_kwargs = dict(color='dimgray', linewidth=0.8, alpha=0.7, zorder=4)
        label_kwargs = dict(fontsize=10, fontweight='bold', color='dimgray',
                             ha='center', va='center', zorder=5)
        if x_label:
            ax.plot([-1.0, 1.0], [0.0, 0.0], **line_kwargs)
            ax.text(1.08, 0.0, x_label, **label_kwargs)
        if y_label:
            ax.plot([0.0, 0.0], [-1.0, 1.0], **line_kwargs)
            ax.text(0.0, 1.08, y_label, **label_kwargs)
        if z_label:
            ax.text(0.02, 0.02, f"{z_label} (out-of-page)", transform=ax.transAxes,
                    fontsize=8, style='italic', color='dimgray', ha='left', va='bottom')

    def plot_scatter(
        self,
        pole_family: Union[str, List[int], Tuple[int, int, int]] = '100',
        ax: Optional[plt.Axes] = None,
        projection: str = 'stereographic',
        hemisphere: str = 'both_mapped',
        color_by: str = 'hemisphere',       # 'hemisphere' | 'role' | 'size' | 'ipf' | 'custom'
        color_data: Optional[Union[np.ndarray, Dict]] = None,
        ipf_direction: Tuple[float, float, float] = (0., 0., 1.),
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        z_label: Optional[str] = None,
        title: Optional[str] = None,
        **scatter_kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the pole figure as a 2D scatter plot.

        x_label / y_label / z_label : optional labels for the sample X/Y/Z
        directions (e.g. "RD"/"TD"/"ND"). X/Y are drawn as crosshair lines
        through the circle; Z is noted in a corner (it is the out-of-plane
        axis). Default None/None/None: no labels drawn, unchanged from
        prior behaviour.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()
            
        poles = self._get_symmetric_poles(pole_family)
        X, Y, ori_indices = self._calculate_projection(poles, projection, hemisphere)
        
        # Add basic reference circle
        circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=1.5, zorder=2)
        ax.add_artist(circle)
        
        # Setup scatter arguments
        s_args = {'s': 10, 'alpha': 0.6, 'edgecolors': 'none'}
        s_args.update(scatter_kwargs)
        
        # ── Color mapping logic ──
        if color_by == 'hemisphere':
            # Check original z hemisphere coordinate to distinguish color
            sample_dirs_all = np.einsum('nij,kj->nki', self.R_stack, poles)
            z_original = sample_dirs_all.reshape(-1, 3)[:, 2]
            
            # Readjust indices depending on the projected hemisphere mask
            # For simplicity, calculate sign of original z for the matching projected coordinates
            if hemisphere == 'upper':
                z_proj = z_original[z_original >= 0]
            elif hemisphere == 'lower':
                z_proj = z_original[z_original < 0]
            else: # both_mapped / both_separate
                z_proj = z_original
                
            upper_mask = z_proj >= 0

            # Always split into upper/lower scatter calls with a legend --
            # 'both_separate' already did this; unified here so
            # 'both_mapped' (and the trivially one-sided 'upper'/'lower')
            # also get a legend explaining what blue/red mean, instead of
            # leaving the viewer to guess. Point positions/colors are
            # unchanged from before for every hemisphere mode.
            if upper_mask.any():
                ax.scatter(X[upper_mask], Y[upper_mask], c='blue', label='Upper Hemisphere', **s_args)
            if (~upper_mask).any():
                ax.scatter(X[~upper_mask], Y[~upper_mask], c='red', label='Lower Hemisphere', **s_args)
            ax.legend(loc='upper right')
                
        elif color_by == 'role':
            if not isinstance(color_data, dict):
                raise ValueError("color_data must be a dict mapping GID -> role for color_by='role'")
            if self.gids is None:
                raise ValueError("gids must be provided to PoleFigure on init for color_by='role'")
                
            # Map each projected point's grain ID to its corresponding role and color
            point_gids = self.gids[ori_indices]
            colors = []
            for gid in point_gids:
                role = color_data.get(int(gid), 'non_host')
                colors.append(_DEFAULT_ROLE_COLOR.get(role, 'gray'))
            ax.scatter(X, Y, c=colors, **s_args)
            
        elif color_by == 'size':
            if not isinstance(color_data, dict):
                raise ValueError("color_data must be a dict mapping GID -> size/voxel count for color_by='size'")
            if self.gids is None:
                raise ValueError("gids must be provided to PoleFigure on init for color_by='size'")
                
            point_gids = self.gids[ori_indices]
            sizes = np.array([color_data.get(int(gid), 0.0) for gid in point_gids])
            
            # Map size to colormap
            im = ax.scatter(X, Y, c=sizes, cmap=scatter_kwargs.get('cmap', 'viridis'), **s_args)
            ax.figure.colorbar(im, ax=ax, label='Grain Size (voxel count)', fraction=0.046, pad=0.04)
            
        elif color_by == 'ipf':
            # Calculate IPF RGB mapping for all orientations
            ipf_colors = self._calculate_ipf_colors(ipf_direction)
            point_colors = ipf_colors[ori_indices]
            ax.scatter(X, Y, c=point_colors, **s_args)
            
        elif color_by == 'custom':
            if color_data is None:
                raise ValueError("color_data cannot be None for color_by='custom'")
            point_colors = np.asarray(color_data)[ori_indices]
            im = ax.scatter(X, Y, c=point_colors, cmap=scatter_kwargs.get('cmap', 'viridis'), **s_args)
            if 'cmap' in scatter_kwargs or not isinstance(color_data[0], (str, tuple, list)):
                ax.figure.colorbar(im, ax=ax, label='Custom Value', fraction=0.046, pad=0.04)
        elif color_by == 'fixed':
            # No computed coloring at all -- s_args already carries
            # whatever 'color'/'label'/'marker' the caller passed via
            # **scatter_kwargs (or matplotlib's own defaults otherwise).
            # Meant for overlaying two independent PoleFigure datasets on
            # one ax (e.g. EBSD vs synthetic) with a marker-shape
            # distinction instead of a color-based one, so per-point
            # coloring doesn't fight the marker-shape legend.
            ax.scatter(X, Y, **s_args)
        else:
            raise ValueError(f"Unknown color_by mode: {color_by}")

        self._draw_axis_crosshairs(ax, x_label, y_label, z_label)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')

        default_title = f"{{{pole_family}}} Pole Figure" if isinstance(pole_family, str) else "Pole Figure"
        ax.set_title(title or default_title, fontsize=12, fontweight='bold')
        
        return fig, ax

    @staticmethod
    def auto_grid_points(half_width_deg: float = 7.5) -> int:
        """The grid resolution grid_points='auto' resolves to inside
        _compute_mud_grid, exposed standalone so callers that need to know
        the number ahead of time (e.g. to build resolution presets as
        fractions of it) don't have to duplicate the formula or construct
        a real PoleFigure just to ask. Depends only on half_width_deg (the
        vMF kernel width), not on sample size -- a fixed number for a
        given half-width, clipped to [80, 250]."""
        omega = np.deg2rad(half_width_deg)
        kappa = 1000.0 if omega < 1e-6 else min(np.log(2.0) / (1.0 - np.cos(omega)), 1000.0)
        return int(np.clip(50 + 6 * np.sqrt(kappa), 80, 250))

    def _compute_mud_grid(
        self,
        poles: np.ndarray,
        grid_points: Union[int, str] = 'auto',
        half_width_deg: float = 7.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the MUD (Multiples of Uniform Distribution) density field
        on a regular grid inside the projection unit circle, using a
        numerically stable, proper antipodal-wrapped spherical von
        Mises-Fisher (vMF) kernel -- the shared core of plot_density(),
        factored out so a second PoleFigure's MUD field can be computed
        on an IDENTICAL grid for a difference plot
        (see plot_density_difference()).

        Note: the grid's back-projection (step 4 below) is always the
        equal-area one, independent of any 'projection' setting elsewhere
        -- matching plot_density()'s own pre-existing behaviour, which
        this refactor preserves unchanged rather than alters.

        Returns (Xi, Yi, zi_mud, mask): Xi/Yi are the full (grid_points,
        grid_points) coordinate meshes, zi_mud is the density field over
        that same shape (NaN outside the projection circle), and mask is
        the boolean circle mask (True = inside).
        """
        # Project all stored orientations, mapped to the upper hemisphere
        # (antipodal symmetry).
        sample_dirs = np.einsum('nij,kj->nki', self.R_stack, poles)
        sample_dirs = sample_dirs.reshape(-1, 3)
        neg_z = sample_dirs[:, 2] < 0
        dirs = sample_dirs.copy()
        dirs[neg_z] *= -1.0

        # ── 1. Expose and Calculate vMF concentration parameter kappa ──
        omega = np.deg2rad(half_width_deg)
        if omega < 1e-6:
            kappa = 1000.0
        else:
            kappa = np.log(2.0) / (1.0 - np.cos(omega))
        kappa = min(kappa, 1000.0)

        # ── 2. Determine Grid Resolution (Adaptive or User-Specified) ──
        if grid_points == 'auto':
            grid_points_val = self.auto_grid_points(half_width_deg)
        else:
            grid_points_val = int(grid_points)

        # ── 3. Construct uniform evaluation grid inside the projection unit circle ──
        xi = np.linspace(-1.05, 1.05, grid_points_val)
        yi = np.linspace(-1.05, 1.05, grid_points_val)
        Xi, Yi = np.meshgrid(xi, yi)
        R2 = Xi**2 + Yi**2

        # Mask points inside the projection circle
        mask = R2 <= 1.0

        # ── 4. Back-project to 3D Cartesian coordinates on the sphere ──
        X_valid = Xi[mask]
        Y_valid = Yi[mask]
        R2_valid = R2[mask]

        z = 1.0 - R2_valid
        x = X_valid * np.sqrt(2.0 - R2_valid)
        y = Y_valid * np.sqrt(2.0 - R2_valid)
        grid_vectors = np.stack([x, y, z], axis=-1)  # shape (N_valid, 3)

        # ── 5. Spherical Density Calculation (Stable vMF with JIT & Chunking) ──
        N_valid = grid_vectors.shape[0]
        N_all_poles = dirs.shape[0]

        if kappa < 1e-4:
            # Uniform distribution limit
            mud_valid = np.ones(N_valid, dtype=np.float64)
        else:
            # Threshold to trigger JIT compilation (e.g. 20,000 poles / ~5000 grains)
            # Above this threshold, compile JIT or use the cached JIT function if numba is available.
            numba_success = False
            if N_all_poles >= 20000:
                global _NUMBA_VMF_KERNEL
                if _NUMBA_VMF_KERNEL is None:
                    try:
                        import numba
                        from numba import njit, prange

                        @njit(parallel=True, fastmath=True)
                        def numba_vmf_kernel(grid, poles, k):
                            N_g = grid.shape[0]
                            N_p = poles.shape[0]
                            out = np.zeros(N_g, dtype=np.float64)
                            denom = 1.0 - np.exp(-2.0 * k)
                            for i in prange(N_g):
                                val_sum = 0.0
                                for j in range(N_p):
                                    dot = grid[i, 0]*poles[j, 0] + grid[i, 1]*poles[j, 1] + grid[i, 2]*poles[j, 2]
                                    numerator = np.exp(k * (dot - 1.0)) + np.exp(-k * (dot + 1.0))
                                    val_sum += k * numerator / denom
                                out[i] = val_sum / N_p
                            return out

                        _NUMBA_VMF_KERNEL = numba_vmf_kernel
                    except ImportError:
                        _NUMBA_VMF_KERNEL = False  # Mark as False to avoid re-attempting imports on later calls

                if _NUMBA_VMF_KERNEL:
                    try:
                        mud_valid = _NUMBA_VMF_KERNEL(grid_vectors, dirs, kappa)
                        numba_success = True
                    except Exception:
                        pass

            if not numba_success:
                # Memory-safety limit: if total operations footprint exceeds 10^7 entries, evaluate in chunks
                if N_valid * N_all_poles > 10000000:
                    chunk_size = 5000
                    mud_valid = np.zeros(N_valid, dtype=np.float64)
                    for start_idx in range(0, N_valid, chunk_size):
                        end_idx = min(start_idx + chunk_size, N_valid)
                        grid_chunk = grid_vectors[start_idx:end_idx]

                        D_chunk = np.dot(grid_chunk, dirs.T)
                        numerator = np.exp(kappa * (D_chunk - 1.0)) + np.exp(-kappa * (D_chunk + 1.0))
                        denominator = 1.0 - np.exp(-2.0 * kappa)
                        mud_valid[start_idx:end_idx] = np.mean(kappa * numerator / denominator, axis=1)
                else:
                    # Standard vectorized calculation
                    D = np.dot(grid_vectors, dirs.T)
                    numerator = np.exp(kappa * (D - 1.0)) + np.exp(-kappa * (D + 1.0))
                    denominator = 1.0 - np.exp(-2.0 * kappa)
                    mud_valid = np.mean(kappa * numerator / denominator, axis=1)

        # Reconstruct full grid
        zi_mud = np.full_like(Xi, np.nan)
        zi_mud[mask] = mud_valid
        return Xi, Yi, zi_mud, mask

    def plot_density(
        self,
        pole_family: Union[str, List[int], Tuple[int, int, int]] = '100',
        ax: Optional[plt.Axes] = None,
        projection: str = 'equal_area',
        grid_points: Union[int, str] = 'auto',
        half_width_deg: float = 7.5,
        contour_scale: str = 'linear',  # 'linear' | 'log' | 'sqrt'
        clabel: bool = False,
        overlay_scatter: bool = False,
        scatter_color_by: str = 'ipf',
        scatter_ipf_direction: np.ndarray = np.array([0, 0, 1]),
        scatter_hemisphere: str = 'both_mapped',
        cmap: str = 'viridis',
        levels: int = 10,
        mud_clip_min: Optional[float] = None,
        mud_clip_max: Optional[float] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        z_label: Optional[str] = None,
        colorbar_decimals: Optional[int] = None,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the pole figure as continuous density contours in Multiples of Uniform Distribution (MUD)
        using a numerically stable, proper antipodal-wrapped spherical von Mises-Fisher (vMF) kernel.

        colorbar_decimals : optional number of decimal places for the
        colour-bar's tick labels (e.g. 2 -> "1.23"). Default None: leave
        matplotlib's own automatic tick formatting unchanged.

        mud_clip_min / mud_clip_max : optional lower/upper bounds to clip the
        computed MUD values to before building the color scale -- either can
        be given alone. Useful for pinning a consistent color scale across
        multiple plots, or compressing outlier density spikes so the rest of
        the distribution stays visible. Default None/None: auto-scale to the
        data's own min/max, unchanged from prior behaviour.

        scatter_hemisphere : hemisphere selection for the optional scatter
        overlay (only used when overlay_scatter=True) -- 'upper' | 'lower' |
        'both_mapped' | 'both_separate', same semantics as plot_scatter's
        hemisphere parameter. The density surface itself always folds both
        hemispheres onto the upper one (antipodal crystallographic symmetry),
        so this only affects which points the overlay draws.

        x_label / y_label / z_label : optional labels for the sample X/Y/Z
        directions, same semantics as plot_scatter's parameters of the same
        name. Default None/None/None: no labels drawn, unchanged from prior
        behaviour.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        # Get raw symmetric 3D pole vectors (unit vectors)
        poles = self._get_symmetric_poles(pole_family)
        if poles.size == 0:
            raise ValueError("No poles generated for this family.")

        Xi, Yi, zi_mud, mask = self._compute_mud_grid(poles, grid_points, half_width_deg)

        # Optional clip of the MUD color scale's lower/upper bounds. The
        # data itself is clipped (so values beyond a requested bound are
        # compressed into the end color -- standard "extend" behaviour,
        # useful when the requested bound is *more* restrictive than the
        # actual data). NaN cells (outside the projection circle) pass
        # through np.clip unaffected.
        if mud_clip_min is not None or mud_clip_max is not None:
            zi_mud = np.clip(zi_mud, mud_clip_min, mud_clip_max)

        # ── 6. Setup levels based on contour_scale ──
        # Bounds come directly from the user's requested clip value
        # whenever given, NOT re-derived from the (possibly still-smaller)
        # clipped data -- this matters when a requested bound is *less*
        # restrictive than the actual data (e.g. mud_clip_max above the
        # true max): clipping the data alone has no effect in that case,
        # so without this the scale would silently stay at the data's own
        # (smaller) max instead of stretching to match what was asked for.
        mud_min = mud_clip_min if mud_clip_min is not None else np.nanmin(zi_mud)
        mud_max = mud_clip_max if mud_clip_max is not None else np.nanmax(zi_mud)
        if mud_min == mud_max or np.isnan(mud_min) or np.isnan(mud_max):
            mud_min = 0.0
            mud_max = 1.0
            
        if contour_scale == 'log':
            min_val = max(0.1, mud_min)
            levels_arr = np.logspace(np.log10(min_val), np.log10(mud_max), levels)
        elif contour_scale == 'sqrt':
            levels_arr = np.square(np.linspace(np.sqrt(mud_min), np.sqrt(mud_max), levels))
        else:
            levels_arr = np.linspace(mud_min, mud_max, levels)
            
        # Draw boundary reference circle
        circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=1.5, zorder=2)
        ax.add_artist(circle)
        
        # Draw density contours
        alpha_val = 0.6 if overlay_scatter else 1.0
        im = ax.contourf(Xi, Yi, zi_mud, levels=levels_arr, cmap=cmap, alpha=alpha_val, zorder=1)
        
        # Conditionally draw contour line labels
        contours = ax.contour(Xi, Yi, zi_mud, levels=levels_arr, colors='black', linewidths=0.5, alpha=0.3, zorder=1)
        if clabel:
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

        cbar_format = f'%.{colorbar_decimals}f' if colorbar_decimals is not None else None
        ax.figure.colorbar(im, ax=ax, label='MUD (Multiples of Uniform Distribution)',
                           fraction=0.046, pad=0.04, format=cbar_format)
        
        # ── 7. Overlay scatter points ──
        if overlay_scatter:
            X, Y, ori_indices = self._calculate_projection(poles, projection, hemisphere=scatter_hemisphere)
            s_args = {'s': 8, 'alpha': 0.8, 'edgecolors': 'none', 'zorder': 3}
            if scatter_color_by == 'ipf':
                ipf_colors = self._calculate_ipf_colors(scatter_ipf_direction)
                point_colors = ipf_colors[ori_indices]
                ax.scatter(X, Y, c=point_colors, **s_args)
            elif scatter_color_by == 'hemisphere':
                # Mirrors plot_scatter's own hemisphere-vs-color_by logic:
                # z_original must be filtered down the same way _calculate_
                # projection filtered X/Y, otherwise the color array's length
                # mismatches the point count whenever scatter_hemisphere
                # actually excludes points (e.g. 'upper'/'lower', as opposed
                # to the previously-hardcoded 'both_mapped' which never did).
                sample_dirs_all = np.einsum('nij,kj->nki', self.R_stack, poles)
                z_original = sample_dirs_all.reshape(-1, 3)[:, 2]
                if scatter_hemisphere == 'upper':
                    z_proj = z_original[z_original >= 0]
                elif scatter_hemisphere == 'lower':
                    z_proj = z_original[z_original < 0]
                else:  # both_mapped / both_separate
                    z_proj = z_original
                upper_mask = z_proj >= 0
                colors = np.where(upper_mask, 'blue', 'red')
                ax.scatter(X, Y, c=colors, **s_args)
            else:
                ax.scatter(X, Y, c='black', **s_args)

        self._draw_axis_crosshairs(ax, x_label, y_label, z_label)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')

        default_title = f"{{{pole_family}}} Density Pole Figure" if isinstance(pole_family, str) else "Density Pole Figure"
        ax.set_title(title or default_title, fontsize=12, fontweight='bold')
        
        if standalone:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def plot_density_difference(
        self,
        other: "PoleFigure",
        pole_family: Union[str, List[int], Tuple[int, int, int]] = '100',
        ax: Optional[plt.Axes] = None,
        grid_points: Union[int, str] = 'auto',
        half_width_deg: float = 7.5,
        unit_normalize: bool = False,
        cmap: str = 'RdBu_r',
        levels: int = 10,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        z_label: Optional[str] = None,
        colorbar_decimals: Optional[int] = None,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the difference between this PoleFigure's MUD density field
        and `other`'s (e.g. an EBSD reference vs a synthetic structure),
        evaluated on an identical grid (same pole_family / grid_points /
        half_width_deg on both sides) so the two are directly comparable
        point-for-point. self is treated as the reference/"real" side and
        `other` as the "synthetic" side for the colorbar title only -- the
        underlying math is symmetric up to an overall sign flip.

        unit_normalize=False (default): plots the raw difference,
            MUD.PF-REAL - MUD.PF-SYNTH.
        unit_normalize=True: each field is divided by its own max before
        differencing, so the comparison is of *shape* rather than
        absolute magnitude,
            (MUD.PF-REAL)/max - (MUD'.PF-SYNTH.)/max

        A diverging colormap centered at zero is used since the
        difference can be positive or negative.

        colorbar_decimals : optional number of decimal places for the
        colour-bar's tick labels. Default None: matplotlib's own
        automatic tick formatting.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        poles_self = self._get_symmetric_poles(pole_family)
        if poles_self.size == 0:
            raise ValueError("No poles generated for this family.")
        _, _, zi_self, _ = self._compute_mud_grid(poles_self, grid_points, half_width_deg)

        poles_other = other._get_symmetric_poles(pole_family)
        if poles_other.size == 0:
            raise ValueError("No poles generated for this family (other).")
        Xi, Yi, zi_other, _ = other._compute_mud_grid(poles_other, grid_points, half_width_deg)

        if zi_self.shape != zi_other.shape:
            raise ValueError(
                "Grid shape mismatch between the two datasets -- pass the "
                "same grid_points to both (e.g. an explicit integer rather "
                "than 'auto', which can resolve to a different resolution "
                "per dataset).")

        if unit_normalize:
            max_self = np.nanmax(zi_self)
            max_other = np.nanmax(zi_other)
            zi_self_n = zi_self / max_self if max_self and np.isfinite(max_self) else zi_self
            zi_other_n = zi_other / max_other if max_other and np.isfinite(max_other) else zi_other
            zi_diff = zi_self_n - zi_other_n
            cbar_label = "[(MUD.PF-REAL)/max - (MUD'.PF-SYNTH.)/max]"
        else:
            zi_diff = zi_self - zi_other
            cbar_label = "[MUD.PF-REAL - MUD.PF-SYNTH.]"

        abs_max = np.nanmax(np.abs(zi_diff))
        if not np.isfinite(abs_max) or abs_max == 0:
            abs_max = 1.0
        levels_arr = np.linspace(-abs_max, abs_max, 2 * levels + 1)

        circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=1.5, zorder=2)
        ax.add_artist(circle)

        im = ax.contourf(Xi, Yi, zi_diff, levels=levels_arr, cmap=cmap, zorder=1)
        ax.contour(Xi, Yi, zi_diff, levels=levels_arr, colors='black',
                   linewidths=0.3, alpha=0.25, zorder=1)
        cbar_format = f'%.{colorbar_decimals}f' if colorbar_decimals is not None else None
        ax.figure.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04, format=cbar_format)

        self._draw_axis_crosshairs(ax, x_label, y_label, z_label)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')

        default_title = (f"{{{pole_family}}} MUD Density Difference" if isinstance(pole_family, str)
                         else "MUD Density Difference")
        ax.set_title(title or default_title, fontsize=12, fontweight='bold')

        if standalone:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def plot_density_3d(
        self,
        pole_family: Union[str, List[int], Tuple[int, int, int]] = '100',
        grid_points: int = 60,
        half_width_deg: float = 7.5,
        cmap: str = 'viridis',
        backend: str = 'pyvista',  # 'pyvista' | 'matplotlib'
        title: Optional[str] = None
    ) -> Any:
        """
        Plot the MUD density as a 3D spherical dome surface using PyVista (default) or Matplotlib.
        """
        # Resolve pyvista fallback if backend is pyvista
        if backend == 'pyvista':
            try:
                import pyvista as pv
            except ImportError:
                import warnings
                warnings.warn("PyVista is not installed. Falling back to Matplotlib backend.")
                backend = 'matplotlib'

        poles = self._get_symmetric_poles(pole_family)
        if poles.size == 0:
            raise ValueError("No poles generated for this family.")
            
        sample_dirs = np.einsum('nij,kj->nki', self.R_stack, poles)
        sample_dirs = sample_dirs.reshape(-1, 3)
        
        neg_z = sample_dirs[:, 2] < 0
        dirs = sample_dirs.copy()
        dirs[neg_z] *= -1.0
        
        omega = np.deg2rad(half_width_deg)
        if omega < 1e-6:
            kappa = 1000.0
        else:
            kappa = np.log(2.0) / (1.0 - np.cos(omega))
        kappa = min(kappa, 1000.0)
        
        # Create a regular grid in spherical coordinates for 3D hemisphere surface
        theta = np.linspace(0, np.pi/2, grid_points)  # Co-latitude: 0 to 90 degrees
        phi = np.linspace(0, 2*np.pi, grid_points)    # Azimuth: 0 to 360 degrees
        
        Theta, Phi = np.meshgrid(theta, phi)
        
        # Spherical coordinates to Cartesian unit vectors
        x = np.sin(Theta) * np.cos(Phi)
        y = np.sin(Theta) * np.sin(Phi)
        z = np.cos(Theta)
        
        # Reshape to list of vectors
        grid_vectors = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
        
        # Evaluate density
        N_valid = grid_vectors.shape[0]
        N_all_poles = dirs.shape[0]
        
        if kappa < 1e-4:
            mud_valid = np.ones(N_valid, dtype=np.float64)
        else:
            if N_valid * N_all_poles > 10000000:
                chunk_size = 5000
                mud_valid = np.zeros(N_valid, dtype=np.float64)
                for start_idx in range(0, N_valid, chunk_size):
                    end_idx = min(start_idx + chunk_size, N_valid)
                    grid_chunk = grid_vectors[start_idx:end_idx]
                    
                    D_chunk = np.dot(grid_chunk, dirs.T)
                    numerator = np.exp(kappa * (D_chunk - 1.0)) + np.exp(-kappa * (D_chunk + 1.0))
                    denominator = 1.0 - np.exp(-2.0 * kappa)
                    mud_valid[start_idx:end_idx] = np.mean(kappa * numerator / denominator, axis=1)
            else:
                D = np.dot(grid_vectors, dirs.T)
                numerator = np.exp(kappa * (D - 1.0)) + np.exp(-kappa * (D + 1.0))
                denominator = 1.0 - np.exp(-2.0 * kappa)
                mud_valid = np.mean(kappa * numerator / denominator, axis=1)
                
        # Reshape back to grid shape
        mud_grid = mud_valid.reshape(Theta.shape)
        
        default_title = f"{{{pole_family}}} 3D Spherical Density" if isinstance(pole_family, str) else "3D Spherical Density"
        plot_title = title or default_title

        if backend == 'pyvista':
            import pyvista as pv
            # Generate a structured grid in pyvista
            grid = pv.StructuredGrid(x, y, z)
            grid["MUD"] = mud_grid.ravel(order='F')

            
            # Setup Plotter
            plotter = pv.Plotter(notebook=True)
            plotter.add_mesh(grid, scalars="MUD", cmap=cmap, smooth_shading=True, show_scalar_bar=True, scalar_bar_args={"title": "MUD"})
            plotter.add_text(plot_title, font_size=12)
            plotter.show_grid(color='gray', xtitle='RD', ytitle='TD', ztitle='ND')
            return plotter
        else:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Normalize mud_grid values to [0, 1] for colormap coloring
            mud_min, mud_max = np.min(mud_grid), np.max(mud_grid)
            if mud_min == mud_max:
                norm_mud = np.zeros_like(mud_grid)
            else:
                norm_mud = (mud_grid - mud_min) / (mud_max - mud_min + 1e-12)
                
            colormap = plt.get_cmap(cmap)
            facecolors = colormap(norm_mud)
            
            # Plot the hemisphere surface in 3D
            surf = ax.plot_surface(
                x, y, z, 
                facecolors=facecolors, 
                rstride=1, cstride=1, 
                linewidth=0, antialiased=True, 
                shade=False
            )
            
            # Add colorbar
            m = cm.ScalarMappable(cmap=colormap)
            m.set_array(mud_grid)
            ax.figure.colorbar(m, ax=ax, label='MUD (Multiples of Uniform Distribution)', fraction=0.046, pad=0.08)
            
            # Set aspect and labels
            ax.set_xlabel('RD (X)')
            ax.set_ylabel('TD (Y)')
            ax.set_zlabel('ND (Z)')
            
            # Set axis limits to make it look spherical
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(0.0, 1.1)
            ax.set_title(plot_title, fontsize=12, fontweight='bold')
            
            return fig, ax




def plot_pole_figure_from_3d(
    lgi_3d: np.ndarray,
    quats_dict: Dict[int, np.ndarray],
    axis: int = 2,
    slice_idx: Optional[int] = None,
    pole_family: Union[str, List[int], Tuple[int, int, int]] = '100',
    ax: Optional[plt.Axes] = None,
    plot_type: str = 'scatter',  # 'scatter' | 'density'
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Slices a 3D grain label grid, extracts active grain orientations,
    and plots their pole figure.

    Parameters
    ----------
    lgi_3d : np.ndarray
        3D grain label field (nz, ny, nx).
    quats_dict : dict
        Dict mapping grain ID -> quaternion [w, x, y, z].
    axis : int
        Slicing direction normal: 0=X, 1=Y, 2=Z.
    slice_idx : int, optional
        Coordinate index along the sliced axis. Defaults to mid-slice.
    pole_family : str or List[int]
        Target pole family to project.
    ax : plt.Axes, optional
        Matplotlib axis.
    plot_type : str
        Type of plot: 'scatter' or 'density'.
    **kwargs :
        Additional arguments passed to PoleFigure.plot_scatter or plot_density.
    """
    from upxo.gsdataops.grid_ops import section_from_3d
    
    if slice_idx is None:
        slice_idx = lgi_3d.shape[axis] // 2
        
    lgi_2d = section_from_3d(lgi_3d, axis=axis, location=slice_idx)
    unique_gids = np.unique(lgi_2d[lgi_2d > 0])
    
    # Collate active orientations in the slice
    slice_quats = []
    slice_gids = []
    for gid in unique_gids:
        g = int(gid)
        if g in quats_dict:
            slice_quats.append(quats_dict[g])
            slice_gids.append(g)
            
    if not slice_quats:
        raise ValueError(f"No grains with orientations in quats_dict found in slice {slice_idx} along axis {axis}.")
        
    pf = PoleFigure(np.array(slice_quats), convention='quaternion', gids=np.array(slice_gids))
    
    if plot_type == 'scatter':
        return pf.plot_scatter(pole_family=pole_family, ax=ax, **kwargs)
    elif plot_type == 'density':
        return pf.plot_density(pole_family=pole_family, ax=ax, **kwargs)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


def plot_components(
    components_dict: Dict[str, Any],
    convention: str = 'quaternion',
    pole_family: Union[str, List[int], Tuple[int, int, int]] = '111',
    projection: str = 'stereographic',
    hemisphere: str = 'both_mapped',
    ax: Optional[plt.Axes] = None,
    cmap: str = 'Set1',
    scatter_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot scatter pole figure where each texture component group is colored uniquely
    and listed in the legend.
    """
    import matplotlib.pyplot as plt
    
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
        
    # Draw boundary circle
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=1.5, zorder=2)
    ax.add_artist(circle)
    
    # Extract active components in order
    comps = [k for k in components_dict.keys() if k != 'random']
    # Map colors using the colormap
    colormap = plt.get_cmap(cmap)
    colors_list = [colormap(i) for i in np.linspace(0, 0.9, len(comps))]
    
    s_args = {'s': 10, 'alpha': 0.6, 'edgecolors': 'none'}
    if scatter_kwargs:
        s_args.update(scatter_kwargs)
        
    # Plot components
    for idx, comp in enumerate(comps):
        data = components_dict[comp]
        oris = data["combined_shuffled"]
        if oris.size == 0:
            continue
            
        pf_temp = PoleFigure(oris, convention=convention)
        poles = pf_temp._get_symmetric_poles(pole_family)
        X, Y, _ = pf_temp._calculate_projection(poles, projection, hemisphere)
        
        pct = oris.shape[0]
        ax.scatter(X, Y, color=colors_list[idx], label=f"{comp} ({pct} grains)", **s_args)
        
    # Plot random background if present
    if 'random' in components_dict:
        rand_oris = components_dict['random']
        if rand_oris.size > 0:
            pf_temp = PoleFigure(rand_oris, convention=convention)
            poles = pf_temp._get_symmetric_poles(pole_family)
            X, Y, _ = pf_temp._calculate_projection(poles, projection, hemisphere)
            pct = rand_oris.shape[0]
            ax.scatter(X, Y, color='lightgray', label=f"random ({pct} grains)", **s_args)
            
    ax.legend(loc='upper right', framealpha=0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    
    title = f"{{{pole_family}}} Components Scatter" if isinstance(pole_family, str) else "Components Scatter"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    if standalone:
        plt.tight_layout()
        plt.show()
        
    return fig, ax


def plot_variants(
    variants_list: List[np.ndarray],
    mean_eulers: np.ndarray,
    convention: str = 'quaternion',
    pole_family: Union[str, List[int], Tuple[int, int, int]] = '111',
    projection: str = 'stereographic',
    hemisphere: str = 'both_mapped',
    ax: Optional[plt.Axes] = None,
    cmap: str = 'tab20',
    scatter_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot scatter pole figure for a single component where each symmetric variant
    is colored uniquely, with its mean Euler angles listed in the legend.
    """
    import matplotlib.pyplot as plt
    
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()
        
    # Draw boundary circle
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=1.5, zorder=2)
    ax.add_artist(circle)
    
    non_empty_indices = [i for i, v in enumerate(variants_list) if v.size > 0]
    num_vars = len(non_empty_indices)
    
    colormap = plt.get_cmap(cmap)
    colors_list = [colormap(i) for i in np.linspace(0, 0.9, num_vars)]
    
    s_args = {'s': 12, 'alpha': 0.7, 'edgecolors': 'none'}
    if scatter_kwargs:
        s_args.update(scatter_kwargs)
        
    for c_idx, v_idx in enumerate(non_empty_indices):
        oris = variants_list[v_idx]
        mean_ea = mean_eulers[v_idx]
        
        pf_temp = PoleFigure(oris, convention=convention)
        poles = pf_temp._get_symmetric_poles(pole_family)
        X, Y, _ = pf_temp._calculate_projection(poles, projection, hemisphere)
        
        # Format Bunge Euler angles in degrees for label
        label_str = f"Var {v_idx+1}: ({mean_ea[0]:.1f}°, {mean_ea[1]:.1f}°, {mean_ea[2]:.1f}°)"
        ax.scatter(X, Y, color=colors_list[c_idx], label=label_str, **s_args)
        
    # Place legend to the right of the plot to avoid overlapping scatter points
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0, framealpha=0.8, fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    
    title = f"{{{pole_family}}} Variants Scatter" if isinstance(pole_family, str) else "Variants Scatter"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    if standalone:
        plt.tight_layout()
        plt.show()
        
    return fig, ax

