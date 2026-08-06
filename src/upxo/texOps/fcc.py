"""
fcc.py
======
Crystallographic texture generator for Face-Centered Cubic (FCC) materials.
Generates model textures (combinations of Copper, Brass, Goss, S, Cube, etc.)
with customizable volume fractions, spreads, and background random components.
"""

import numpy as np
from itertools import permutations, product
from typing import Dict, Union, Tuple, List, Optional, Any



# Standard FCC texture component Euler angles (Bunge ZXZ convention, in degrees)
STD_COMPONENTS = {
    "copper": (90.0, 35.0, 45.0),
    "brass":  (35.0, 45.0, 0.0),
    "s":      (59.0, 37.0, 63.0),
    "goss":   (90.0, 90.0, 45.0),
    "cube":   (0.0,  0.0,  0.0),
    "rotated_cube": (45.0, 0.0, 0.0),
    "P":  (90.0, 45.0, 0.0),
    "A1": (35.0, 45.0, 90.0),
    "A2": (55.0, 90.0, 45.0),
    "B":  (45.0, 90.0, 45.0),
    "C":  (0.0,  90.0, 45.0),
    "Q":  (35.0, 55.0, 45.0),
    "D":  (59.0, 37.0, 26.0),
}


def cubic_symmetry_operators() -> np.ndarray:
    """Generate the 24 proper rotation matrices for cubic m-3m symmetry."""
    ops = []
    for p in permutations(range(3)):
        P = np.eye(3)[list(p)]
        for signs in product([-1, 1], repeat=3):
            S = P * np.array(signs)[None, :]
            if round(np.linalg.det(S)) == 1:
                ops.append(S.astype(np.float64))
    
    unique_ops = []
    for op in ops:
        if not any(np.allclose(op, u) for u in unique_ops):
            unique_ops.append(op)
    return np.array(unique_ops)


def euler_bunge_to_matrix(phi1: float, Phi: float, phi2: float, degrees: bool = True) -> np.ndarray:
    """Convert Bunge ZXZ Euler angles to a 3x3 rotation matrix."""
    if degrees:
        phi1, Phi, phi2 = np.deg2rad([phi1, Phi, phi2])
    
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    
    Rz1 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    Rz2 = np.array([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
    
    return Rz1 @ Rx @ Rz2


def matrix_to_euler_bunge(R: np.ndarray, degrees: bool = True) -> Tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to Bunge ZXZ Euler angles."""
    R = np.asarray(R, dtype=float)
    c = np.clip(R[2, 2], -1.0, 1.0)
    Phi_rad = np.arccos(c)
    
    if abs(Phi_rad) < 1e-12:
        phi1_rad = np.arctan2(R[1, 0], R[0, 0])
        phi2_rad = 0.0
    elif abs(Phi_rad - np.pi) < 1e-12:
        phi1_rad = np.arctan2(R[1, 2], R[0, 2])
        phi2_rad = 0.0
    else:
        phi1_rad = np.arctan2(R[2, 0], -R[2, 1])
        phi2_rad = np.arctan2(R[0, 2], R[1, 2])
        
    if degrees:
        return (float(np.degrees(phi1_rad) % 360.0),
                float(np.degrees(Phi_rad)),
                float(np.degrees(phi2_rad) % 360.0))
    return (float(phi1_rad % (2 * np.pi)), float(Phi_rad), float(phi2_rad % (2 * np.pi)))


def normalize_euler_bunge(ea: np.ndarray, degrees: bool = True, eps: float = 1e-6) -> np.ndarray:
    """Normalize Bunge ZXZ Euler angles to canonical ranges."""
    A = np.asarray(ea, dtype=float)
    A2 = np.atleast_2d(A)
    phi1, Phi, phi2 = A2[:, 0].copy(), A2[:, 1].copy(), A2[:, 2].copy()
    
    two_pi, pi = (360.0, 180.0) if degrees else (2 * np.pi, np.pi)
    
    phi1[:] = np.mod(phi1, two_pi)
    phi2[:] = np.mod(phi2, two_pi)
    Phi[:] = ((Phi + pi) % (2 * pi)) - pi
    
    neg = Phi < 0.0
    if np.any(neg):
        Phi[neg] = -Phi[neg]
        phi1[neg] = np.mod(phi1[neg] + pi, two_pi)
        phi2[neg] = np.mod(phi2[neg] + pi, two_pi)
        
    over = Phi > pi
    if np.any(over):
        Phi[over] = 2 * pi - Phi[over]
        phi1[over] = np.mod(phi1[over] + pi, two_pi)
        phi2[over] = np.mod(phi2[over] + pi, two_pi)
        
    if eps is not None:
        Phi[np.abs(Phi) < eps] = 0.0
        Phi[np.abs(Phi - pi) < eps] = pi
        
    out = np.stack([phi1, Phi, phi2], axis=-1)
    return out if A2.shape[0] > 1 else out[0]


def proj_to_so3(R: np.ndarray) -> np.ndarray:
    """Project 3x3 matrix R to the nearest proper rotation matrix via SVD."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def unique_rotations(rotations: List[np.ndarray], tol: float = 1e-8) -> List[np.ndarray]:
    """Deduplicate list of rotation matrices."""
    uniq = []
    for R in rotations:
        if not any(np.linalg.norm(R - Q, ord='fro') < tol for Q in uniq):
            uniq.append(R)
    return uniq


def fcc_symmetrise_ori(bea: Tuple[float, float, float]) -> np.ndarray:
    """Generate all symmetrically equivalent Bunge Euler angle triplets for one FCC orientation."""
    g = euler_bunge_to_matrix(*bea, degrees=True)
    sym_ops = cubic_symmetry_operators()
    eq_mats = [proj_to_so3(S @ g) for S in sym_ops]
    eq_mats = unique_rotations(eq_mats, tol=1e-8)
    return np.array([list(matrix_to_euler_bunge(R, degrees=True))
                     for R in eq_mats], dtype=np.float32)


def rand_uniform_so3(size: int = 1) -> np.ndarray:
    """Draw uniform random rotations from SO(3) using Shoemake (1992) method."""
    u1 = np.random.random(size)
    u2 = np.random.random(size)
    u3 = np.random.random(size)
    
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    
    q = np.column_stack([q4, q1, q2, q3])  # [w, x, y, z]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0
    return q


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a stack of quaternions [w, x, y, z] to (N, 3, 3) rotation matrices."""
    q = np.atleast_2d(q)
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
    return R


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a stack of 3x3 rotation matrices to unit quaternions [w, x, y, z]."""
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
    q /= norm + 1e-12
    q[q[:, 0] < 0] *= -1.0
    return q.squeeze()


class FCCTexture:
    """
    Synthetic FCC crystallographic texture from named ideal components.

    Samples orientations around ideal Bunge Euler positions (copper,
    brass, Goss, cube, …) with per-component volume fractions and
    angular spreads. Remainder of the volume fraction is random
    background. Optional sample symmetries (RD/TD/ND 180° rotations)
    for rolling-style ODFs. Complements :class:`~upxo.xtalphy.texops.tops`
    and registry :class:`~upxo.material.texture.TextureComponentProfile`.
    """

    def __init__(
        self,
        tc_fractions: Dict[str, float],
        spread: Union[float, Dict[str, float]] = 5.0,
        apply_sample_symmetries: bool = True,
        use_rd: bool = True,
        use_td: bool = True,
        use_nd: bool = True,
        custom_components: Optional[Dict[str, Tuple[float, float, float]]] = None
    ):
        """
        Initialize the texture model with volume fractions and spreads.

        Parameters
        ----------
        tc_fractions : dict
            Texture component names mapping to their volume fractions (0.0 to 1.0).
            The remainder (1.0 - sum(fractions)) is automatically allocated to a random texture background.
        spread : float or dict
            The half-width dispersion (standard deviation) in degrees for each component.
        apply_sample_symmetries : bool
            Whether to apply macroscopic sample symmetry (e.g. rolling symmetry).
        use_rd : bool
            Apply 180° rotation symmetry about the Rolling Direction (sample X-axis).
        use_td : bool
            Apply 180° rotation symmetry about the Transverse Direction (sample Y-axis).
        use_nd : bool
            Apply 180° rotation symmetry about the Normal Direction (sample Z-axis).
        custom_components : dict, optional
            Custom texture component names mapping to their Bunge Euler angles in degrees (phi1, Phi, phi2).
        """
        # Validate fractions
        total_frac = sum(tc_fractions.values())
        if total_frac > 1.0001:
            raise ValueError(f"Sum of volume fractions cannot exceed 1.0 (got {total_frac:.4f})")
            
        self.tc_fractions = {k: max(0.0, float(v)) for k, v in tc_fractions.items()}
        
        # Build components library
        self.components = dict(STD_COMPONENTS)
        if custom_components:
            self.components.update(custom_components)
            
        # Standardize spreads
        if isinstance(spread, (int, float)):
            self.spreads = {k: float(spread) for k in self.tc_fractions}
        else:
            self.spreads = {k: float(spread.get(k, 5.0)) for k in self.tc_fractions}
            
        self.apply_sample_symmetries = apply_sample_symmetries
        self.use_rd = use_rd
        self.use_td = use_td
        self.use_nd = use_nd

    def _compute_sample_symmetry_group(self) -> List[np.ndarray]:
        """Compute the mathematically closed group of selected sample symmetry generators."""
        generators = []
        if self.apply_sample_symmetries:
            if self.use_rd:
                generators.append(np.diag([1.0, -1.0, -1.0]))
            if self.use_td:
                generators.append(np.diag([-1.0, 1.0, -1.0]))
            if self.use_nd:
                generators.append(np.diag([-1.0, -1.0, 1.0]))
                
        # Group closure computation (generators commute and are self-inverse)
        group = [np.eye(3)]
        for gen in generators:
            current_elements = list(group)
            for elem in current_elements:
                product = gen @ elem
                if not any(np.allclose(product, item) for item in group):
                    group.append(product)
        return group

    def _get_combined_symmetry_equivalents(self, mean_euler: Tuple[float, float, float]) -> np.ndarray:
        """Apply crystal (left) and sample (right) symmetries and return unique canonical Euler angles."""
        g = euler_bunge_to_matrix(*mean_euler, degrees=True)
        sym_ops = cubic_symmetry_operators()
        sample_ops = self._compute_sample_symmetry_group()
        
        eq_mats = []
        for S in sym_ops:
            for M in sample_ops:
                eq_mats.append(proj_to_so3(S @ g @ M))
                
        eq_mats = unique_rotations(eq_mats, tol=1e-8)
        return np.array([list(matrix_to_euler_bunge(R, degrees=True))
                         for R in eq_mats], dtype=np.float32)

    def _allocate_counts(self, N: int) -> Dict[str, int]:
        """Allocate orientation counts to components using largest remainder method."""
        raw_counts = {k: self.tc_fractions[k] * N for k in self.tc_fractions}
        counts = {k: int(np.floor(raw_counts[k])) for k in self.tc_fractions}
        
        assigned = sum(counts.values())
        vf_sum = sum(self.tc_fractions.values())
        target_assigned = min(N, int(round(vf_sum * N)))
        
        remainders = sorted(
            [(raw_counts[k] - counts[k], k) for k in self.tc_fractions],
            reverse=True
        )
        
        while assigned < target_assigned and remainders:
            _, k = remainders.pop(0)
            counts[k] += 1
            assigned += 1
            
        # Handle random remainder allocation
        n_random = N - sum(counts.values())
        if n_random < 0:
            # Over-allocation clip from largest components
            over = -n_random
            for k, val in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                take = min(val, over)
                counts[k] -= take
                over -= take
                if over == 0:
                    break
            n_random = N - sum(counts.values())
            
        counts['random'] = n_random
        return counts

    def _generate_detailed_euler_dict(self, N: int) -> Dict[str, Any]:
        """Generate detailed texture orientation data in Bunge Euler angles (degrees)."""
        counts = self._allocate_counts(N)
        
        components_data = {}
        global_list = []
        
        for tc, eacount in counts.items():
            if tc == 'random' or eacount <= 0:
                continue
                
            if tc not in self.components:
                raise ValueError(f"Unknown texture component '{tc}'. Choose from {list(self.components.keys())}")
                
            mean_euler = self.components[tc]
            spread = self.spreads[tc]
            
            # Symmetrize the component mean
            sym_eqs = self._get_combined_symmetry_equivalents(mean_euler)
            num_sym = len(sym_eqs)
            
            # Allocate orientations among the symmetric equivalents
            if eacount >= num_sym:
                base_alloc = eacount // num_sym
                rem_alloc = eacount % num_sym
                allocs = np.full(num_sym, base_alloc, dtype=int)
                if rem_alloc > 0:
                    chosen_indices = np.random.choice(num_sym, size=rem_alloc, replace=False)
                    allocs[chosen_indices] += 1
            else:
                allocs = np.zeros(num_sym, dtype=int)
                chosen_indices = np.random.choice(num_sym, size=eacount, replace=False)
                allocs[chosen_indices] = 1
                
            variants = []
            for i, ea in enumerate(sym_eqs):
                alloc = allocs[i]
                if alloc <= 0:
                    variants.append(np.empty((0, 3), dtype=np.float64))
                    continue
                # Sample normally in Euler space around the symmetric equivalent
                sampled = np.random.normal(loc=ea, scale=spread, size=(alloc, 3))
                sampled_normalized = np.atleast_2d(normalize_euler_bunge(sampled, degrees=True))
                variants.append(sampled_normalized)
                
            # Filter empty arrays when combining to avoid vstack errors
            non_empty_variants = [v for v in variants if v.shape[0] > 0]
            if non_empty_variants:
                combined = np.vstack(non_empty_variants)
                combined_shuffled = combined.copy()
                np.random.shuffle(combined_shuffled)
            else:
                combined_shuffled = np.empty((0, 3), dtype=np.float64)
                
            components_data[tc] = {
                "variants": variants,
                "mean_eulers": sym_eqs,
                "combined_shuffled": combined_shuffled
            }
            if combined_shuffled.size > 0:
                global_list.append(combined_shuffled)
                
        # Generate random background orientations
        n_random = counts.get('random', 0)
        if n_random > 0:
            rand_quats = rand_uniform_so3(n_random)
            rand_mats = quat_to_matrix(rand_quats)
            rand_eulers = np.array([matrix_to_euler_bunge(R, degrees=True) for R in rand_mats])
            components_data["random"] = rand_eulers
            global_list.append(rand_eulers)
        else:
            components_data["random"] = np.empty((0, 3), dtype=np.float64)
            
        if global_list:
            final_shuffled = np.vstack(global_list)
            np.random.shuffle(final_shuffled)
        else:
            final_shuffled = np.empty((0, 3), dtype=np.float64)
            
        return {
            "components": components_data,
            "final_shuffled": final_shuffled
        }

    def generate_euler(self, N: int) -> np.ndarray:
        """Generate N orientations as Bunge Euler angles in degrees (phi1, Phi, phi2)."""
        res = self._generate_detailed_euler_dict(N)
        return res["final_shuffled"]

    def generate_detailed(self, N: int, convention: str = 'euler') -> Dict[str, Any]:
        """
        Generate detailed texture orientations, organized by component and variant clusters.

        Parameters
        ----------
        N : int
            Total number of orientations to generate.
        convention : str
            Output orientation convention: 'euler' (Bunge ZXZ degrees), 'quaternion' (w, x, y, z), or 'matrix' (3x3).

        Returns
        -------
        dict
            Structured dictionary of generated orientations.
        """
        if convention not in ('euler', 'quaternion', 'matrix'):
            raise ValueError(f"Invalid convention '{convention}'. Choose from 'euler', 'quaternion', 'matrix'.")
            
        res = self._generate_detailed_euler_dict(N)
        
        if convention == 'euler':
            return res
            
        # Convert all arrays in the dict
        def convert_arr(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                if convention == 'quaternion':
                    return np.empty((0, 4), dtype=np.float64)
                elif convention == 'matrix':
                    return np.empty((0, 3, 3), dtype=np.float64)
            
            arr_2d = np.atleast_2d(arr)
            mats = np.array([euler_bunge_to_matrix(*ea, degrees=True) for ea in arr_2d])
            if convention == 'matrix':
                return mats
            elif convention == 'quaternion':
                return np.atleast_2d(matrix_to_quat(mats))
            return arr_2d


        converted_components = {}
        for tc, data in res["components"].items():
            if tc == 'random':
                converted_components[tc] = convert_arr(data)
            else:
                converted_components[tc] = {
                    "variants": [convert_arr(v) for v in data["variants"]],
                    "mean_eulers": data["mean_eulers"],  # Keep mean_eulers in Bunge Euler degrees!
                    "combined_shuffled": convert_arr(data["combined_shuffled"])
                }
                
        return {
            "components": converted_components,
            "final_shuffled": convert_arr(res["final_shuffled"])
        }

    def generate_quaternions(self, N: int) -> np.ndarray:
        """Generate N orientations as unit quaternions [w, x, y, z]."""
        res = self.generate_detailed(N, convention='quaternion')
        return res["final_shuffled"]

    def generate_matrices(self, N: int) -> np.ndarray:
        """Generate N orientations as 3x3 rotation matrices."""
        res = self.generate_detailed(N, convention='matrix')
        return res["final_shuffled"]

