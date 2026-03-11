from copy import deepcopy
import numpy as np
from math import floor
from random import Random
from itertools import permutations, product, combinations
import matplotlib.pyplot as plt

class tops:

    __slots__ = ('tc_info', 'ori_count', 'N', 'n_texture_instances',
                 'n_sampling_instances', 'tex', 'sym_ops')

    fcc_tc_std = {"copper": (90.0, 35.0, 45.0),  # {112}<111> Rolling
                  "brass": (35.0, 45.0, 0.0),  # {110}<112> Rolling
                  "s": (59.0, 37.0, 63.0),  # {123}<634> Rolling
                  "goss": (90.0, 90.0, 45.0),  # {110}<001> Rolling
                  "cube": (0.0, 0.0, 0.0),  # {001}<100> Annealing / RX
                  "rotated_cube": (45.0, 0.0, 0.0),  # {001}<110> Annealing / RX
                  "P": (90.0, 45.0, 0.0),   # {011}<122> Annealing / RX
                  "A1": (35.0, 45.0, 90.0),  # {111}<110> Shear
                  "A2": (55.0, 90.0, 45.0),  # {111}<112> Shear
                  "B": (45.0, 90.0, 45.0),  # {112}<110> Shear
                  "C": (0.0, 90.0, 45.0),  # {001}<110> Shear
                  "Q": (35.0, 55.0, 45.0),  # {013}<231> transitional
                  "D": (59.0, 37.0, 26.0),  # {4411}<1118> approx. transitional
                  }

    ea_dtype = np.int32

    FCC_POLES = {
        '100': np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]),
        '110': np.array([[1,1,0], [-1,-1,0], [1,-1,0], [-1,1,0], [1,0,1], [-1,0,-1],
                         [1,0,-1], [-1,0,1], [0,1,1], [0,-1,-1], [0,1,-1], [0,-1,1]]),
        '111': np.array([[1,1,1], [-1,-1,-1], [1,-1,-1], [-1,1,1], [1,1,-1], [-1,-1,1],
                         [-1,1,-1], [1,-1,1]])
        }

    def __init__(self):
        self.sym_ops = self.cubic_symmetry_operators()

    @classmethod
    def synth_fcc(cls, N=1000,
                  tc_info={"copper": [0.45, [90.0, 35.0, 45.0]],
                           "brass": [0.30, [35.0, 45.0, 0.0]],
                           "goss": [0.10, [0.0, 45.0, 0.0]],
                           "rotated_cube": [0.15, [45.0, 0.0, 0.0]],
                           },
                  n_tex_instances=2,
                  n_sampling_instances=50):
        """
        Example
        -------
        from upxo.xtalphy.texops import tops
        tex = tops.synth_fcc(N=500,
                      tc_info={"cube": [0.05, [0.0, 0.0, 0.0]],
                               "rotated_cube": [0.05, [45.0, 0.0, 0.0]],
                               "brass": [0.25, [35.0, 45.0, 0.0]],
                               "goss": [0.10, [0.0, 45.0, 0.0]],
                               "copper": [0.35, [90.0, 35.0, 45.0]],
                               "s": [0.2, [59.0, 37.0, 63.0]]
                               },
                      n_tex_instances=1,
                      n_sampling_instances=5)

        b = tex.tex['tex_instance.1']['sampling_instances']['ossi.1']['tc_ori_stacks']
        ea = np.vstack(list(b.values()))
        tex.plot_pole_figure(ea, pole_family='111', title="")

        b1 = [tex.tex['tex_instance.1']['sampling_instances'][k]['tc_ori_stacks']
             for k in tex.tex['tex_instance.1']['sampling_instances'].keys() if k[:5] == 'ossi.']
        b2 = [np.vstack(list(_b1_.values())) for _b1_ in b1]
        b3 = np.vstack(b2)
        b3.shape
        tex.plot_pole_figure(b3, pole_family='111')
        """
        tex = cls()
        tex.set_tc_info(tc_info)
        tex.gen_tex_fcc_synthetic(N=N,
                                  n_tex_instances=n_tex_instances,
                                  n_sampling_instances=n_sampling_instances)
        return tex

    def plot_pole_figure(self, euler_angles_deg, pole_family='100', title=""):
        """
        Creates a complete pole figure scatter plot for a specified pole family.

        Args:
            euler_angles_deg (np.ndarray): An (N, 3) array of Bunge Euler angles.
            pole_family (str): The pole family to plot: '100', '110', or '111'.
            title (str, optional): Custom title for the plot.
        """
        # 1. Validate and retrieve the correct set of poles
        if pole_family not in self.FCC_POLES:
            raise ValueError(f"Invalid pole_family. Choose from {list(self.FCC_POLES.keys())}")
        poles = self.FCC_POLES[pole_family]

        # Normalize the pole vectors (especially for {111} and {110})
        poles = poles / np.linalg.norm(poles, axis=1, keepdims=True)

        # Convert Euler angles to rotation matrices
        R_stack = self.cubic_euler_bunge_to_matrix_v1(
            euler_angles_deg[:, 0],
            euler_angles_deg[:, 1],
            euler_angles_deg[:, 2],
            degrees=True
        )

        # Transform all poles into the sample coordinate system
        sample_dirs = np.einsum('nij,kj->nki', R_stack, poles).reshape(-1, 3)

        # 2. Separate points and project both hemispheres
        upper = sample_dirs[sample_dirs[:, 2] >= 0]
        lower = -sample_dirs[sample_dirs[:, 2] < 0] # Invert for projection

        X_upper = upper[:, 0] / (1 + upper[:, 2])
        Y_upper = upper[:, 1] / (1 + upper[:, 2])
        X_lower = lower[:, 0] / (1 + lower[:, 2])
        Y_lower = lower[:, 1] / (1 + lower[:, 2])

        # 3. Create the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        circle = plt.Circle((0, 0), 1, color='black', fill=False)
        ax.add_artist(circle)

        ax.scatter(X_upper, Y_upper, s=5, c='blue', alpha=0.3)
        ax.scatter(X_lower, Y_lower, s=5, c='red', alpha=0.3)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.axis('off')

        # Set a descriptive title if one isn't provided
        default_title = f"{{{pole_family}}} Pole Figure"
        ax.set_title(title or default_title, fontsize=14)
        plt.show()

    def cubic_Rz(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[ c,-s, 0], [ s, c, 0], [ 0, 0, 1]])

    def cubic_Rx(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c,-s], [0, s, c]])

    def cubic_euler_bunge_to_matrix(self, phi1, Phi, phi2, degrees=True):
        """
        Bunge (ZXZ): R = Rz(phi1) * Rx(Phi) * Rz(phi2).

        EXAMPLE - 1
        -----------
        from upxo.xtalphy.texops import tops as TO
        ea = np.array([10, 20, 30])
        tex = TO()
        R = tex.cubic_euler_bunge_to_matrix(*ea)
        print(R)
        """
        if degrees:
            phi1, Phi, phi2 = np.deg2rad([phi1, Phi, phi2])
        return self.cubic_Rz(phi1) @ self.cubic_Rx(Phi) @ self.cubic_Rz(phi2)

    def cubic_euler_bunge_to_matrix_v1(self, phi1, Phi, phi2, degrees=True):
        if degrees:
            phi1, Phi, phi2 = np.deg2rad([phi1, Phi, phi2])

        # Ensure phi1, Phi, phi2 are arrays
        phi1 = np.asarray(phi1)
        Phi = np.asarray(Phi)
        phi2 = np.asarray(phi2)

        c1, s1 = np.cos(phi1), np.sin(phi1)
        c, s = np.cos(Phi), np.sin(Phi)
        c2, s2 = np.cos(phi2), np.sin(phi2)

        # Create helper arrays of zeros and ones with the correct shape
        zeros = np.zeros_like(c1)
        ones = np.ones_like(c1)

        # Build the stacks of rotation matrices
        Rz1 = np.array([[c1, -s1, zeros], [s1, c1, zeros], [zeros, zeros, ones]]).transpose(2, 0, 1)
        Rx = np.array([[ones, zeros, zeros], [zeros, c, -s], [zeros, s, c]]).transpose(2, 0, 1)
        Rz2 = np.array([[c2, -s2, zeros], [s2, c2, zeros], [zeros, zeros, ones]]).transpose(2, 0, 1)

        # Perform the batch matrix multiplication
        return Rz1 @ Rx @ Rz2

    def cubic_rotation_angle(self, R):
        """Return angle (rad) of a proper rotation matrix R."""
        x = (np.trace(R)-1.0)/2.0
        x = np.clip(x, -1.0, 1.0)
        return np.arccos(x)

    def cubic_rotation_axis(self, R, angle):
        """Return unit axis for rotation R given angle (rad).
           For very small angles, returns a default axis."""
        if angle < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        A = (R-R.T)/(2.0*np.sin(angle))
        # axis components are (A32, A13, A21)
        axis = np.array([A[2, 1], A[0, 2], A[1, 0]])
        n = np.linalg.norm(axis)
        return axis/(n if n > 0 else 1.0)

    def cubic_symmetry_operators(self):
        """
        24 proper rotations for m-3m as signed permutation matrices with det=+1.

        Example
        -------
        from upxo.xtalphy.texops import tops as TO
        tex = TO()
        # ----------
        tex.cubic_symmetry_operators()
        """
        ops = []
        for p in permutations(range(3)):  # 6 permutations
            P = np.eye(3)[list(p)]
            for signs in product([-1,1], repeat=3):  # 8 sign patterns
                S = P * np.array(signs)[None, :]
                if round(np.linalg.det(S)) == 1:
                    ops.append(S.astype(float))
        # de-duplicate
        uniq = []
        for S in ops:
            if not any(np.allclose(S, T) for T in uniq):
                uniq.append(S)
        return uniq

    def fcc_symmetrise_ori(self, ea_bunge, dtype=np.float32):
        """
        Generate symmetric equivalents of an orientation.

        Example
        -------
        from upxo.xtalphy.texops import tops as TO
        tex = TO()
        # ----------
        ea_bunge = np.array([10, 20, 30])
        tex.fcc_symmetrise_ori(ea_bunge, dtype=np.float32)
        """
        g = self.cubic_euler_bunge_to_matrix(*ea_bunge, degrees=True)
        # sym_ops = self.cubic_symmetry_operators()
        eq_mats = [self._proj_to_so3(S @ g) for S in self.sym_ops]
        eq_mats = self._unique_rotations(eq_mats, tol=1E-8)
        symm_eq = np.array([list(self._matrix_to_euler_bunge(R, degrees=True))
                            for R in eq_mats], dtype=dtype)
        return symm_eq

    def fcc_symmetrise_ori_V1(self, ea_bunge, dtype=np.float32):
        """
        Generate symmetric equivalents of an orientation inside Fund. Zone.

        Example
        -------
        from upxo.xtalphy.texops import tops as TO
        tex = TO()
        # ----------
        fcc_tc_std = {"copper": (90.0, 35.0, 45.0),  # {112}<111> Rolling
                      "brass": (35.0, 45.0, 0.0),  # {110}<112> Rolling
                      "s": (59.0, 37.0, 63.0),  # {123}<634> Rolling
                      "goss": (0.0, 45.0, 0.0),  # {110}<001> Rolling
                      "cube": (0.0, 0.0, 0.0),  # {001}<100> Annealing / RX
                      "rotated_cube": (45.0, 0.0, 0.0),  # {001}<110> Annealing / RX
                      "P": (90.0, 45.0, 0.0),   # {011}<122> Annealing / RX
                      "A1": (35.0, 45.0, 90.0),  # {111}<110> Shear
                      "A2": (55.0, 90.0, 45.0),  # {111}<112> Shear
                      "B": (45.0, 90.0, 45.0),  # {112}<110> Shear
                      "C": (0.0, 90.0, 45.0),  # {001}<110> Shear
                      "Q": (35.0, 55.0, 45.0),  # {013}<231> transitional
                      "D": (59.0, 37.0, 26.0),  # {4411}<1118> approx. transitional
                      }
        EA_fund_zone = {}
        for tc, tcea in fcc_tc_std.items():
            ea_bunge = np.array(tcea)
            tex.fcc_symmetrise_ori(ea_bunge, dtype=np.float32)

            g = tex.cubic_euler_bunge_to_matrix(*ea_bunge, degrees=True)
            eq_mats = [tex._proj_to_so3(S @ g) for S in tex.sym_ops]
            eq_mats = tex._unique_rotations(eq_mats, tol=1E-8)
            symm_eq = np.array([list(tex._matrix_to_euler_bunge(R, degrees=True))
                                for R in eq_mats], dtype=np.float32)
            EA_fund_zone[tc] = symm_eq[np.where(np.prod(symm_eq <= 90, axis=1))[0]]
        """
        g = self.cubic_euler_bunge_to_matrix(*ea_bunge, degrees=True)
        # sym_ops = self.cubic_symmetry_operators()
        eq_mats = [self._proj_to_so3(S @ g) for S in self.sym_ops]
        eq_mats = self._unique_rotations(eq_mats, tol=1E-8)
        symm_eq = np.array([list(self._matrix_to_euler_bunge(R, degrees=True))
                            for R in eq_mats], dtype=dtype)
        return symm_eq

    @staticmethod
    def _matrix_to_euler_bunge(R, degrees=True):
        """
        Convert rotation matrix to Bunge ZXZ Euler (phi1, Phi, phi2).
        Assumes a proper rotation matrix.
        """
        # clamp values for numerical safety
        R = np.asarray(R, dtype=float)
        # Phi from R33
        c = np.clip(R[2,2], -1.0, 1.0)
        Phi = np.arccos(c)
        if abs(Phi) < 1e-12:
            # singular: Phi = 0 -> phi1 + phi2 = atan2(R[1,0], R[0,0])
            phi1 = np.arctan2(R[1,0], R[0,0])
            phi2 = 0.0
        elif abs(Phi - np.pi) < 1e-12:
            # singular: Phi = pi -> phi2 - phi1 = atan2(R[1,2], -R[0,2])
            phi1 = np.arctan2(R[1,2], R[0,2])
            phi2 = 0.0
        else:
            phi1 = np.arctan2(R[2,0], -R[2,1])  # consistent with ZXZ Bunge
            phi2 = np.arctan2(R[0,2], R[1,2])
        if degrees:
            return (np.degrees(phi1) % 360.0, np.degrees(Phi), np.degrees(phi2) % 360.0)
        return (phi1 % (2*np.pi), Phi, phi2 % (2*np.pi))

    @staticmethod
    def _axis_angle_to_R(axis, angle_rad):
        """Rodrigues formula."""
        ax = np.asarray(axis, dtype=float)
        n = np.linalg.norm(ax)
        if n < 1e-15 or abs(angle_rad) < 1e-15:
            return np.eye(3)
        u = ax / n
        ux, uy, uz = u
        K = np.array([[0,-uz, uy],
                      [uz, 0,-ux],
                      [-uy,ux, 0]], dtype=float)
        I = np.eye(3)
        return I + np.sin(angle_rad)*K + (1.0 - np.cos(angle_rad))*(K @ K)

    @staticmethod
    def _rand_unit_vector(rng):
        """Uniform random unit vector on S2."""
        # Method: normal distribution and normalize
        v = np.array([rng.gauss(0,1), rng.gauss(0,1), rng.gauss(0,1)], dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-15:
            return np.array([1.0,0.0,0.0])
        return v / n

    @staticmethod
    def _rand_uniform_SO3(rng):
        """Uniform random rotation using random unit quaternion."""
        u1, u2, u3 = rng.random(), rng.random(), rng.random()
        q1 = np.sqrt(1-u1)*np.sin(2*np.pi*u2)
        q2 = np.sqrt(1-u1)*np.cos(2*np.pi*u2)
        q3 = np.sqrt(u1)*np.sin(2*np.pi*u3)
        q4 = np.sqrt(u1)*np.cos(2*np.pi*u3)
        # quaternion to rotation
        x,y,z,w = q1,q2,q3,q4
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ], dtype=float)
        return R

    @staticmethod
    def _proj_to_so3(R):
        # project to nearest proper rotation via SVD
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn

    @staticmethod
    def _unique_rotations(rotations, tol=1e-8):
        """Deduplicate rotation matrices by Frobenius norm tolerance."""
        uniq = []
        for R in rotations:
            if not any(np.linalg.norm(R - Q, ord='fro') < tol for Q in uniq):
                uniq.append(R)
        return uniq

    def _as_rotmat(self, x, degrees=True):
        """
        Accept Euler triplet (len==3) OR 3x3 rotation matrix.
        Return a 3x3 rotation matrix (float).
        """
        arr = np.asarray(x)
        if arr.shape == (3, 3):
            return arr.astype(float)
        if arr.ndim == 1 and arr.size == 3:
            phi1, Phi, phi2 = map(float, arr.ravel())
            return self.cubic_euler_bunge_to_matrix(phi1, Phi, phi2, degrees=degrees)
        raise ValueError(f"_as_rotmat: expected Euler (3,) or R (3,3); got shape {arr.shape}")

    def _get_cubic_ops_np(self):
        """Cached cubic symmetry operators as a (24,3,3) float array."""
        '''ops = getattr(self, "CUBIC_OPS", None)
        if ops is None:
            # Build once from your existing generator
            # ops_list = self.cubic_symmetry_operators()
            ops = np.asarray(self.sym_ops, dtype=float)  # (24,3,3)
            CUBIC_OPS = ops
        elif not isinstance(ops, np.ndarray):
            ops = np.asarray(ops, dtype=float)
            CUBIC_OPS = ops
        return CUBIC_OPS'''
        return np.asarray(self.sym_ops, dtype=float)

    def normalize_euler_bunge(self, ea, degrees=True, eps=1e-6):
        """
        Normalize Bunge ZXZ Euler angles (phi1, Phi, phi2) to canonical ranges.
          - phi1, phi2 in [0, 360) deg  (or [0, 2π) rad)
          - Phi in   [0, 180]  deg      (or [0, π]   rad)
        Handles BOTH Phi > 180 and Phi < 0 via ZXZ symmetry:
           Phi' = -Phi  and (phi1', phi2') = (phi1+180, phi2+180)  [mod 360]
           Phi' = 360-Phi and same 180-shift for >180 case.
        """
        A = np.asarray(ea, dtype=float)
        A2 = np.atleast_2d(A)
        phi1, Phi, phi2 = A2[:, 0], A2[:, 1], A2[:, 2]

        if degrees:
            two_pi, pi = 360.0, 180.0
        else:
            two_pi, pi = 2*np.pi, np.pi

        # Wrap phi1, phi2 into [0, 360) or [0, 2π)
        phi1[:] = np.mod(phi1, two_pi)
        phi2[:] = np.mod(phi2, two_pi)

        # First, reduce Phi to (-pi, pi] to avoid big excursions
        # (helps when jitter pushes way beyond range)
        Phi[:] = ((Phi + pi) % (2*pi)) - pi

        # Case 1: Phi < 0  -> mirror about 0 and shift (phi1,phi2)+=180°
        neg = Phi < 0.0
        if np.any(neg):
            Phi[neg] = -Phi[neg]
            phi1[neg] = np.mod(phi1[neg] + pi, two_pi)
            phi2[neg] = np.mod(phi2[neg] + pi, two_pi)

        # Case 2: Phi > 180 -> fold to 360 - Phi, shift (phi1,phi2)+=180°
        over = Phi > pi
        if np.any(over):
            Phi[over] = 2*pi - Phi[over]
            phi1[over] = np.mod(phi1[over] + pi, two_pi)
            phi2[over] = np.mod(phi2[over] + pi, two_pi)

        # Snap very small numerical negatives/endpoint jitter
        if eps is not None:
            Phi[np.abs(Phi) < eps] = 0.0
            Phi[np.abs(Phi - pi) < eps] = pi

        out = np.stack([phi1, Phi, phi2], axis=-1)
        return out if A2.shape[0] > 1 else out[0]

    def cubic_misorientation_V1(self, EA1, EA2, unique_tol_deg=1e-4, degrees=True):
        """
        Vectorized fast misorientation (cubic, m-3m).
        Inputs: Euler triplet (phi1,Phi,phi2) OR 3x3 rotation matrix.
        Returns:
          angle_deg_min : float
          axis_min      : (3,) unit vector (sample frame)
          top3_angles_deg : list of up to 3 smallest UNIQUE angles (deg), ascending

         Example
         -------
         from upxo.xtalphy.texops import tops as TO
         tex = TO()
         # ----------
         tex.cubic_misorientation_V1([0, 0, 0], [35, 45, 90])
        """
        # Normalize inputs to rotation matrices
        gA = self._as_rotmat(EA1, degrees=degrees)  # (3,3)
        gB = self._as_rotmat(EA2, degrees=degrees)  # (3,3)

        # Symmetry ops as (24,3,3)
        S = self._get_cubic_ops_np()  # (24,3,3)

        # All symmetry-equivalent orientations
        # RA[k] = S[k] @ gA -> (24,3,3)
        # RB[l] = S[l] @ gB -> (24,3,3)
        RA = S @ gA
        RB = S @ gB

        # All relative rotations: dR[l,k] = RB[l] @ RA[k].T   -> (24,24,3,3)
        RtA = np.transpose(RA, (0,2,1))  # (24,3,3)
        dR = RB[:, None, :, :] @ RtA[None, :, :, :]  # (24,24,3,3)

        # Angles from trace: angle = arccos( (tr(dR)-1)/2 )
        # tr = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]  # (24,24)
        tr = np.einsum('iijj->ij', dR)
        x = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        ang = np.arccos(x)  # radians, (24,24)

        # Minimum angle and its indices
        idx_flat = np.argmin(ang)
        l_min, k_min = divmod(int(idx_flat), ang.shape[1])
        best_angle = float(ang[l_min, k_min])  # radians

        # Axis for the min dR
        dR_min = dR[l_min, k_min]  # (3,3)
        if best_angle < 1e-12:
            axis_min = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            sin_a = np.sin(best_angle)
            # A = (R - R^T) / (2 sin a)
            A = (dR_min - dR_min.T) / (2.0 * sin_a)
            axis_min = np.array([A[2,1], A[0,2], A[1,0]], dtype=float)
            n = np.linalg.norm(axis_min)
            if n > 0:
                axis_min /= n
            else:
                axis_min = np.array([1.0, 0.0, 0.0], dtype=float)

        # Top-3 UNIQUE angles (deg), ascending
        angles_deg = np.degrees(np.sort(ang, axis=None))  # (576,)
        uniq = []
        for a in angles_deg:
            if not uniq or abs(a - uniq[-1]) > unique_tol_deg:
                uniq.append(a)
            if len(uniq) >= 3:
                break

        return float(np.degrees(best_angle)), axis_min, uniq

    def cubic_misorientation_V2(self, EA1, EA2, unique_tol_deg=1e-4, degrees=True):
        """
        Vectorized fast misorientation (cubic, m-3m).
        Inputs: Euler triplet (phi1,Phi,phi2) OR 3x3 rotation matrix.
        Returns:
          angle_deg_min : float
          axis_min      : (3,) unit vector (sample frame)
          top3_angles_deg : list of up to 3 smallest UNIQUE angles (deg), ascending

         Example
         -------
         from upxo.xtalphy.texops import tops as TO
         tex = TO()
         # ----------
         tex.cubic_misorientation_V2([0, 0, 0], [35, 45, 90])
        """
        # Normalize inputs to rotation matrices
        gA = self._as_rotmat(EA1, degrees=degrees)  # (3,3)
        gB = self._as_rotmat(EA2, degrees=degrees)  # (3,3)

        # Symmetry ops as (24,3,3)
        S = self._get_cubic_ops_np()  # (24,3,3)

        # All symmetry-equivalent orientations
        # RA[k] = S[k] @ gA -> (24,3,3)
        # RB[l] = S[l] @ gB -> (24,3,3)
        RA = S @ gA
        RB = S @ gB

        # All relative rotations: dR[l,k] = RB[l] @ RA[k].T   -> (24,24,3,3)
        RtA = np.transpose(RA, (0,2,1))  # (24,3,3)
        dR = RB[:, None, :, :] @ RtA[None, :, :, :]  # (24,24,3,3)

        # Angles from trace: angle = arccos( (tr(dR)-1)/2 )
        # tr = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]  # (24,24)
        tr = np.einsum('iijj->ij', dR)
        x = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        ang = np.arccos(x)  # radians, (24,24)

        # Minimum angle and its indices
        idx_flat = np.argmin(ang)
        l_min, k_min = divmod(int(idx_flat), ang.shape[1])
        best_angle = float(ang[l_min, k_min])  # radians

        # Axis for the min dR
        dR_min = dR[l_min, k_min]  # (3,3)
        if best_angle < 1e-12:
            axis_min = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            sin_a = np.sin(best_angle)
            # A = (R - R^T) / (2 sin a)
            A = (dR_min - dR_min.T) / (2.0 * sin_a)
            axis_min = np.array([A[2,1], A[0,2], A[1,0]], dtype=float)
            n = np.linalg.norm(axis_min)
            if n > 0:
                axis_min /= n
            else:
                axis_min = np.array([1.0, 0.0, 0.0], dtype=float)

        return float(np.degrees(best_angle)), axis_min

    def cubic_misorientation_V3(self, EA1, EA2, degrees=True):
        """
        Vectorized fast misorientation (cubic, m-3m).
        Inputs: Euler triplet (phi1,Phi,phi2) OR 3x3 rotation matrix.
        Returns:
          angle_deg_min : float
          axis_min      : (3,) unit vector (sample frame)
          top3_angles_deg : list of up to 3 smallest UNIQUE angles (deg), ascending

         Example
         -------
         from upxo.xtalphy.texops import tops as TO
         tex = TO()
         # ----------
         tex.cubic_misorientation_V2([0, 0, 0], [35, 45, 90])
        """
        # Normalize inputs to rotation matrices
        gA = self._as_rotmat(EA1, degrees=degrees)  # (3,3)
        gB = self._as_rotmat(EA2, degrees=degrees)  # (3,3)

        # Symmetry ops as (24,3,3)
        S = self._get_cubic_ops_np()  # (24,3,3)

        # All symmetry-equivalent orientations
        # RA[k] = S[k] @ gA -> (24,3,3)
        # RB[l] = S[l] @ gB -> (24,3,3)
        RA = S @ gA
        RB = S @ gB

        # All relative rotations: dR[l,k] = RB[l] @ RA[k].T   -> (24,24,3,3)
        RtA = np.transpose(RA, (0,2,1))  # (24,3,3)
        dR = RB[:, None, :, :] @ RtA[None, :, :, :]  # (24,24,3,3)

        # Angles from trace: angle = arccos( (tr(dR)-1)/2 )
        # tr = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]  # (24,24)
        tr = np.einsum('iijj->ij', dR)
        x = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        ang = np.arccos(x)  # radians, (24,24)

        # Minimum angle and its indices
        idx_flat = np.argmin(ang)
        l_min, k_min = divmod(int(idx_flat), ang.shape[1])
        best_angle = float(ang[l_min, k_min])  # radians

        return float(np.degrees(best_angle))

    def set_tc_info(self, tc_info,
                    defaults={'hw_phi1': 5,
                              'hw_Phi': 5,
                              'hw_phi2': 5,
                              'std_k_phi1': 3,
                              'std_k_Phi': 3,
                              'std_k_phi2': 3,
                              'perctol_phi1': 5,
                              'perctol_Phi': 5,
                              'perctol_phi2': 5}):
        """
        Standardizes a dictionary of texture component information to a consistent format.

        Parameters:
        - tc_info (dict): The input dictionary containing texture components.
                          Each value can be a scalar or a list of varying length.

        Returns:
        - dict: A new dictionary with all values standardized to the most complete format:
                [percentage, [ang1, ang2, ang3], [std1, std2, std3], [k1, k2, k3]].
                Default values are used to fill in missing information.

        Examples
        --------
        from upxo.xtalphy.texops import tops as TO
        tex = TO()
        # ----------
        # Possible input for gstslice.tc_info: 1
        tex.set_tc_info({"copper": [0.45, [90.0, 35.0, 45.0]],
                         "brass": [0.30, [35.0, 45.0, 0.0]],
                         "goss": [0.20, [90.0, 90.0, 45.0]],
                         "rotated_cube": [0.01, [45.0, 0.0, 0.0]]
                         })

        # Possible input for gstslice.tc_info: 2
        tex.set_tc_info({"copper": [0.45, [90.0, 35,.0 45.0], 8.0],
                         "s": [0.15, [59.0, 37.0, 63.0], 7.0],
                         "goss": [0.20, [90.0, 90.0, 45.0], 5.5],
                         "rotated_cube": [0.01, [45.0, 0.0, 0.0], 9.8]
                         })

        # Possible input for gstslice.tc_info: 3
        tex.set_tc_info({"copper": [0.45, [90.0, 35.0, 45.0], [8.0, 8.0, 8.0]],
                         "goss": [0.05, [90.0, 90.0, 45.0], [6.0, 6.0, 6.0]],
                         "brass": [0.30, [35.0, 45.0, 0.0], [6.0, 6.0, 6.0]],
                         "rotated_cube": [0.01, [45.0, 0.0, 0.0], [8.0, 8.0, 8.0]]
                         })

        # Possible input for gstslice.tc_info: 4
        tex.set_tc_info({"copper": [0.45, [90.0, 35.0, 45.0], [8.0, 8.0, 8.0], [3.0, 3.0, 3.0]],
                         "brass": [0.30, [35.0, 45.0, 0.0], [10.0, 10.0, 10.0], [3.0, 3.0, 3.0]],
                         "goss": [0.05, [90.0, 90.0, 45.0], [6.0, 6.0, 6.0], [3.0, 3.0, 3.0]],
                         "rotated_cube": [0.01, [45.0, 0.0, 0.0], [8.0, 8.0, 8.0], [3.0, 3.0, 3.0]]
                         })

        # Possible input for gstslice.tc_info: 5
        tex.set_tc_info({"copper": [0.45, [90.0, 35.0, 45.0], [8.0, 8.0, 8.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]],
                         "s": [0.15, [59.0, 37.0, 63.0], [7.0, 7.0, 7.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]],
                         "brass": [0.30, [35.0, 45.0, 0.0], [10.0, 10.0, 10.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]],
                         "goss": [0.05, [90.0, 90.0, 45.0], [6.0, 6.0, 6.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]],
                         "rotated_cube": [0.01, [45.0, 0.0, 0.0], [8.0, 8.0, 8.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]
                         })

        print(tex.tc_info)

        NOTE
        ----
        fcc_tc = {"copper": (90.0, 35.0, 45.0),  # {112}<111> # Rolling texture
                  "brass": (35.0, 45.0, 0.0),  # {110}<112> # Rolling texture
                  "s": (59.0, 37.0, 63.0),  # {123}<634> # Rolling texture
                  "goss": (90.0, 90.0, 45.0),  # {110}<001> # Rolling texture
                  "cube": (0.0, 0.0, 0.0),  # {001}<100> # Annealing / RX
                  "rotated_cube": (45.0, 0.0, 0.0),  # {001}<110> # Annealing / RX
                  "P": (90.0, 45.0, 0.0),   # {011}<122> # Annealing / RX
                  "A1": (35.0, 45.0, 90.0),  # {111}<110> # Shear texture
                  "A2": (55.0, 90.0, 45.0),  # {111}<112> # Shear texture
                  "B": (45.0, 90.0, 45.0),  # {112}<110> # Shear texture
                  "C": (0.0, 90.0, 45.0),  # {001}<110> # Shear texture
                  "Q": (35.0, 55.0, 45.0),  # {013}<231> # Minor / transitional
                  "D": (59.0, 37.0, 26.0),  # {4411}<1118> approx. # Minor / transitional
                  }
        """
        print(".. User texture information standardisation.")
        standardized_info = {}

        for key, value in tc_info.items():
            if not isinstance(value, list) or len(value) < 2:
                raise ValueError(
                    f"Invalid format for '{key}'. Input must be a list of at least two "
                    f"elements: [percentage, [phi1, Phi, phi2], ...]."
                )
            if not isinstance(value[1], list) or len(value[1]) != 3:
                raise ValueError(
                    f"Invalid format for '{key}'. The second element must be a list "
                    f"of three Euler angles."
                )

            percentage = value[0]
            euler_angles = value[1]

            current_value = list(value)

            if len(current_value) < 3:
                spreads = [defaults['hw_phi1'], defaults['hw_Phi'], defaults['hw_phi2']]
            elif not isinstance(current_value[2], list):
                # Handle scalar spread value
                spreads = [current_value[2], current_value[2], current_value[2]]
            else:
                spreads = current_value[2]

            if len(current_value) < 4:
                std_k = [defaults['std_k_phi1'], defaults['std_k_Phi'], defaults['std_k_phi2']]
            else:
                std_k = current_value[3]

            if len(current_value) < 5:
                perctol = [defaults['perctol_phi1'], defaults['perctol_Phi'], defaults['perctol_phi2']]
            else:
                perctol = current_value[4]

            # --- 4. Assemble the standardized entry ---
            standardized_info[key] = [percentage, euler_angles, spreads, std_k, perctol]

        self.tc_info = standardized_info

    def get_tcinfo(self):
        """
        Examples
        --------
        from upxo.xtalphy.texops import tops as TO
        tex = TO()
        # ----------
        tex.set_tc_info({"copper": [0.45, [90.0, 35.0, 45.0]],
                         "brass": [0.30, [35.0, 45.0, 0.0]],
                         "goss": [0.20, [90.0, 90.0, 45.0]],
                         "rotated_cube": [0.01, [45.0, 0.0, 0.0]]
                         })
        VF, HW, SK, PT, ori_means = tex.get_tcinfo()
        """
        # =============================================================
        '''Volume fraction'''
        VF = {key: val[0] for key, val in self.tc_info.items()}
        '''Hafl - widths representing ori spreads'''
        HW = {key: val[2] for key, val in self.tc_info.items()}
        '''Standard deviation factors to be used to scale down spread for
        generating distribtuion.'''
        SK = {key: val[3] for key, val in self.tc_info.items()}
        '''Percentage tolerance for acceptance.'''
        PT = {key: val[4] for key, val in self.tc_info.items()}
        '''Mean orientations as specified by the user.'''
        ori_means = {key: val[1] for key, val in self.tc_info.items()}

        return VF, HW, SK, PT, ori_means

    def alloc_ori_counts_to_tc(self, N):
        """
        from upxo.xtalphy.texops import tops as TO
        tex = TO()
        tex.set_tc_info({"copper": [0.45, [90.0, 35.0, 45.0]],
                         "brass": [0.30, [35.0, 45.0, 0.0]],
                         "goss": [0.20, [90.0, 90.0, 45.0]],
                         "rotated_cube": [0.01, [45.0, 0.0, 0.0]]
                         })
        ori_count = tex.alloc_ori_counts_to_tc(100)
        print(ori_count)
        """
        tc_comps = set(self.tc_info.keys())
        # fractions (allow sum<1 -> fill with random)
        vf = {c: max(0.0, float(self.tc_info.get(c, 0.0)[0]))
              for c in tc_comps}
        vf_sum = sum(vf.values())
        # -----------------------------------------------------------
        print('.. Caclulating TC wise orientation counts')
        # =============================================================
        # allocate counts via largest remainder
        raw = {c: vf[c]*N for c in tc_comps}
        ori_count = {c: floor(raw[c]) for c in tc_comps}
        assigned = sum(ori_count.values())
        rema = sorted([(raw[c]-ori_count[c], c) for c in tc_comps],
                      reverse=True)
        while assigned < min(N, int(round(vf_sum*N))) and rema:
            _, c = rema.pop(0)
            ori_count[c] += 1
            assigned += 1
        n_random = N-sum(ori_count.values())
        if n_random < 0:
            # clip over-allocation from biggest components
            over = -n_random
            for c, ocount in sorted(ori_count.items(),
                                    key=lambda kv: kv[1],
                                    reverse=True):
                take = min(ocount, over)
                ori_count[c] -= take
                over -= take
                if over == 0:
                    break
            n_random = N-sum(ori_count.values())
        ori_count['random'] = n_random
        self.ori_count = ori_count

    def build_tex_dict_template(self):
        print("Building TEX dictionary template")
        self.tex = {f"tex_instance.{i}": {}
               for i in np.arange(1, self.n_texture_instances+1, 1)}
        self.tex['tc_info'] = self.tc_info
        self.tex['shuffle'] = True
        self.tex['tc_info'] = self.tc_info
        self.tex['fcc_tc_std'] = self.fcc_tc_std

    def gen_tex_fcc_synthetic(self, N=1000,
                              distr='normal',
                              n_tex_instances=1,
                              n_sampling_instances=1,
                              rand_ori_gen_rule='relaxed',
                              ):
        """
        from upxo.xtalphy.texops import tops as TO
        tex = TO()
        tex.set_tc_info({"copper": [0.45, [90.0, 35.0, 45.0]],
                         "brass": [0.30, [35.0, 45.0, 0.0]],
                         "goss": [0.10, [90.0, 90.0, 45.0]],
                         "rotated_cube": [0.15, [45.0, 0.0, 0.0]]
                         })
        tex.gen_tex_fcc_synthetic(N=500, n_tex_instances=2,
                                  n_sampling_instances=500)

        tex.tex.keys()
        tex.n_texture_instances
        tex.n_sampling_instances
        tex.tex['tc_info']
        tex.tex['fcc_tc_std']
        tex.tex['N']

        tex.tex['tex_instance.1'].keys()
        >> dict_keys(['symeq_full', 'sampling_instances'])

        tex.tex['tex_instance.1']['symeq_full']['goss'].keys()
        >> dict_keys(['loc_symm_eq', 'BEA_SYMEQ', 'BEA_shuffled',
                      'shuffle_ids', 'BEA_SYMEQ_MO_TC', 'BEA_SYMEQ_MO_TC_ids',
                      'BEA_MO_STACK', 'BEA_MO_IDS_STACK'])

        len(tex.tex['tex_instance.1']['symeq_full']['goss']['BEA_SYMEQ'])
        >> 24

        tex.tex['tex_instance.1']['sampling_instances'].keys()
        >> dict_keys(['ossi.1', 'metadata'])

        tex.tex['tex_instance.1']['sampling_instances']['ossi.1'].keys()
        >> dict_keys(['nsamples', 'tc_ori', 'tc_ori_sample_ids', 'tc_ori_stacks'])

        a = tex.tex['tex_instance.1']['sampling_instances']['ossi.1']['nsamples']
        a
        >> {'copper': 45, 'brass': 30, 'goss': 10, 'rotated_cube': 15}
        sum(list(a.values()))

        b = tex.tex['tex_instance.1']['sampling_instances']['ossi.1']['tc_ori_stacks']
        b.keys()
        sum([_b_.shape[0] for _b_ in b.values()])
        """
        # FUNCTION LEVEL (FL) | 2 (i.e. number of tabs at start). In concoise,
        # This wil be written as in below line. These are markers / flags to
        # indicate the code indentations which give a feel of where we are.
        # (FL | 2)
        print(60*'=', '\n\nGenerating orientations for cryst. texture\n')
        VF, HW, SK, PT, ori_means = self.get_tcinfo()
        self.alloc_ori_counts_to_tc(N)
        tc_comps = list(self.tc_info.keys())
        # =============================================================
        self.N = N
        self.n_texture_instances = n_tex_instances
        self.n_sampling_instances = n_sampling_instances
        # =============================================================
        self.build_tex_dict_template()
        # (FL | 2)
        # =============================================================
        for ti in np.arange(1, self.n_texture_instances+1, 1):
            print('\n', 75*'=')
            print(f".. TEXTURE INSTANCE NUMBER: {ti} of {n_tex_instances}\n")
            tiname = f"tex_instance.{ti}"
            # --------------------------------------
            self.gen_tex_fcc_synthetic_single_instance(VF, HW, SK, PT,
                                                       ori_means, tiname)
            # --------------------------------------
            _fn_ = self.gen_tex_fcc_sampling_instances
            SI = _fn_(tiname,
                      np.arange(1, self.n_sampling_instances+1, 1),
                      tc_comps,
                      rand_ori_gen_rule=rand_ori_gen_rule)
            # --------------------------------------
            self.tex[tiname]['sampling_instances'] = SI

    def gen_tex_fcc_synthetic_single_instance(self, VF, HW, SK, PT,
                                              ori_means, tiname):
        tc_comps = set(self.ori_count.keys())
        # (FL | 2) > (TEX INST | 3)
        print('.. Generating ori. clusters around mean texture components')
        comp_eulers = {tc_comp: [] for tc_comp in tc_comps}
        MISORI = {tc_comp: [] for tc_comp in tc_comps}
        """
        BEA_SYMEQ_MO: Misorinetation angle of all orientations
        around each symmetric equivalent of the mean oreintation of
        every texcture component specified by user. Note; See descriptions
        for variable BEA_SYMEQ_MO_TC (in below codes) for more detauils.

        BEA_SYMEQ_MO.keys()
        dict_keys(['copper', 'brass', 's', 'goss', 'cube'])

        After processing, this shoudl have.
        * len(BEA_SYMEQ_MO['copper']) --> 24
        * len(BEA_SYMEQ_MO['copper'][0]) == ori_count['copper'] --> True
        """
        symeq = {tc_comp: None for tc_comp in tc_comps}
        _fn_ = self.gen_symmetric_equivalents_fcc_ori
        tex_symmeq = _fn_(VF, HW, SK, PT, ori_means)
        # ---------------------------------------------------------------------
        self.tex[tiname]['symeq_full'] = tex_symmeq
        # len(TEX[tiname]['symeq_full'][tc_comp]['BEA_SYMEQ_MO_TC_ids'])
        # tc_ori_range = range(symeq[tc_comp]['BEA_shuffled'].shape[0])

    def gen_tex_fcc_sampling_instances(self, tiname, sampling_instances_ids,
                                       tc_comps, rand_ori_gen_rule='relaxed'):
        print('\n')
        print(f".. Generating {len(sampling_instances_ids)} SAMPLING INSTANCES\n")
        SI = {f"ossi.{si}": {} for si in sampling_instances_ids}
        SI["metadata"] = {"ossi": "Orientation sub-set instance"}
        for si in sampling_instances_ids:
            # (FL | 2) > (TEX INST | 3) > (SAMP INST | 4)
            # NOTE: SAMP INST: Sampling instances
            siname = f"ossi.{si}"
            print(f".... Sampling instance: '{siname}' @ {tc_comps}")
            SI[siname]['nsamples'] = {}
            SI[siname]['tc_ori'] = {}
            SI[siname]['tc_ori_sample_ids'] = {}
            for tc_comp in tc_comps:
                # 0
                # SI[siname]['tc_ori'][tc_comp] = None
                # 1
                nsamples = self.ori_count[tc_comp]
                SI[siname]['nsamples'][tc_comp] = nsamples
                # 2
                # tc_ori_sample_ids = np.random.choice(tc_ori_range, nsamples)
                # SI[siname]['tc_ori_sample_ids'][tc_comp] = tc_ori_sample_ids
                # 3
                """
                Get all 24 symmetric equivalent BEAs of the curerent TC.
                """
                beasets = self.tex[tiname]['symeq_full'][tc_comp]['BEA_SYMEQ']
                # 4
                """
                Allocating the total number of samples needed for random
                picking. These sample counts coorespond to that from each
                of the 24 orintatipon sample sets.
                """
                pss = nsamples // 24  # preliminary sample size
                fss = nsamples % 24  # final sample size
                samplrng, symorirng = range(nsamples), range(24)

                sel_ori = [np.empty((0, 3), dtype=self.ea_dtype) for _ in symorirng]
                sel_ori_symm_id = [np.empty(0, dtype=np.int16) for _ in symorirng]
                # 5
                if pss >= 1:
                    # sel_ori = [None for _ in symorirng]
                    # sel_ori_symm_id = [None for _ in symorirng]
                    for i, beas in enumerate(beasets):
                        rand_ids = np.random.choice(samplrng,
                                                    pss).astype(np.int16)
                        sel_ori[i] = beas[rand_ids]
                        sel_ori_symm_id[i] = rand_ids
                if fss != 0:
                    for i, beas in enumerate(beasets):
                        if i+1 <= fss:
                            rid = np.random.choice(samplrng,
                                                   1).astype(np.int16)
                            sel_ori[i] = np.vstack((sel_ori[i], beas[rid]))
                            sel_ori_symm_id[i] = np.hstack((sel_ori_symm_id[i],
                                                            rid))
                        else:
                            break
                # 6
                SI[siname]['tc_ori'][tc_comp] = sel_ori
                SI[siname]['tc_ori'][tc_comp+'_stack'] = np.vstack(sel_ori)
                SI[siname]['tc_ori_sample_ids'][tc_comp] = sel_ori_symm_id
            # -------------------------------------------
            print('...... Filling remainder with uniform random orientations')
            if self.ori_count['random'] == 0:
                randori = np.empty((0, 3))
            else:
                randori = [None for _ in range(self.ori_count['random'])]
                if rand_ori_gen_rule == 'relaxed':
                    for i in range(self.ori_count['random']):
                        ro = self._rand_uniform_SO3(Random(np.random.random()))
                        ro = list(self._matrix_to_euler_bunge(ro, degrees=True))
                        randori[i] = ro
                elif rand_ori_gen_rule == 'strict':
                    """
                    This will ensure none of the random orientations
                    generated will have a misorientation angle less than X
                    degrees from any of the symmetric equivalents of each
                    of the user prescribed texture component mean
                    orientations. value of X will be norm of the
                    half-widths specififed for the corresponding texture
                    component.
                    """
                    pass
                randori = np.array(randori, dtype=np.float32)

            SI[siname]['tc_ori']['random'] = randori
            # -------------------------------------------
            tc_comps_full = deepcopy(tc_comps)
            tc_comps_full.append('random')
            tc_ori_stacks = {tc_comp: SI[siname]['tc_ori'][tc_comp+'_stack']
                             if tc_comp != 'random'
                             else SI[siname]['tc_ori'][tc_comp]
                             for tc_comp in tc_comps_full}
            SI[siname]['tc_ori_stacks'] = tc_ori_stacks
        return SI

    def gen_symmetric_equivalents_fcc_ori(self, VF, HW, SK, PT,
                                          ori_means):
        """
        BEA_SYMEQ_MO: Misorinetation angle of all orientations
        around each symmetric equivalent of the mean oreintation of
        every texcture component specified by user. Note; See descriptions
        for variable BEA_SYMEQ_MO_TC (in below codes) for more detauils.

        BEA_SYMEQ_MO.keys()
        dict_keys(['copper', 'brass', 's', 'goss', 'cube'])

        After processing, this shoudl have.
        * len(BEA_SYMEQ_MO['copper']) --> 24
        * len(BEA_SYMEQ_MO['copper'][0]) == ori_count['copper'] --> True
        """
        tc_comps = set(self.tc_info.keys())
        symeq = {tc_comp: None for tc_comp in tc_comps}
        sym_eq_IDs = np.arange(1, 24+1, 1)
        nshuffles = 2
        # generate clusters per component
        for tc_comp in tc_comps:
            # (FL | 2) > (TEX INST | 3) > (TEX COMP | 4)
            eacount = self.ori_count[tc_comp]
            if eacount <= 0:
                symeq[tc_comp] = {'loc_symm_eq': [],
                                  'BEA_SYMEQ': [],
                                  'BEA_shuffled': np.empty((0, 3)),
                                  'shuffle_ids': [],
                                  'BEA_SYMEQ_MO_TC': [],
                                  'BEA_SYMEQ_MO_TC_ids': [],
                                  'BEA_MO_STACK': np.empty(0),
                                  'BEA_MO_IDS_STACK': np.empty(0)}
                continue
            '''get the mean Euler angle of this texture component'''
            # meanang = self.tc_info[tc_comp][1]
            # -------------------------------------------------------------
            '''Calculate the scales needed to generate normal random
            numbers.'''
            scales = np.asarray(HW[tc_comp])/np.asarray(SK[tc_comp])
            # -------------------------------------------------------------
            """Generate the euler angle cluster.

            BEA_STACK: Stack of Bunge's Euler angles.
            bea: element of BEA.
            """
            # ---------------------
            """
            1. Gather all symmetric equivalents of this TC EA.
            loc_symm_eq: location - symmetric equivalents in Euler
                         space
            """
            loc_symm_eq = self.fcc_symmetrise_ori(self.tc_info[tc_comp][1],
                                                  dtype=self.ea_dtype)
            """
            2. Generate orientations around each ea in loc_symm_eq.

            BEA_SYMEQ: Bunge's Euler angles of all orientations around
            each symmetric equivalent of the mean oreintation of this
            texcture component. list of 24 numpy arrays. Each numpy is
            array of size defined by the number of orientation samples
            defined by 'eacount'. Note that 'eacount' is obtained using
            texture volume fraction.

            BEA_SYMEQ_MO_TC: Misorinetation angle of all orientations
            around each symmetric equivalent of the mean oreintation of
            this texcture component. list of 24 numpy array. Each numpy
            is array of size defined by the number of orientation
            samples defined by 'eacount'. Note that 'eacount' is
            obtained using texture volume fraction.

            BEA_SYMEQ_MO_TC_ids: Symmetric porientaion ID number. it
            can be any valkue bwtween 1 and 24, inclusive. Needed when
            misorintation or orientations need to be reverse-mapped
            to individual orientations in 'loc_symm_eq'
            """
            BEA_SYMEQ = [np.random.normal(loc=ea, scale=scales,
                                          size=(eacount, 3)).astype(self.ea_dtype)
                         for ea in loc_symm_eq]
            BEA_SYMEQ = [self.normalize_euler_bunge(bea, degrees=True).astype(self.ea_dtype)
                         for bea in BEA_SYMEQ]
            # print('++++++++++++++++++++++++++++++')
            # print(BEA_SYMEQ)
            # print('++++++++++++++++++++++++++++++')
            print("Calculating misorientations across all symmetric",
                  f"equivalents of '{tc_comp}' texture component:",
                  f"{24*eacount} oris")
            BEA_SYMEQ_MO_TC = []
            BEA_SYMEQ_MO_TC_ids = []
            for i, lseq, bea_seq in zip(sym_eq_IDs, loc_symm_eq, BEA_SYMEQ):
                # print('@@@@@', bea_seq, '####')
                # (FL | 2) > (TEX INST | 3) > (TEX COMP | 4) > (ORI SYMM | 5)
                if i % 3 == 0:
                    print(".. TC: '{tc_comp}'. MO calculation for",
                          f" symmetric equivalent {i} of {len(loc_symm_eq)}")
                bea_seq_2d = np.atleast_2d(bea_seq)
                bea_seq_mo = np.zeros(bea_seq.shape[0], dtype=self.ea_dtype)
                for ii, _bea_seq_ in enumerate(bea_seq):
                    _mo_ = self.cubic_misorientation_V3(lseq, _bea_seq_,
                                                        degrees=True)
                    bea_seq_mo[ii] = _mo_
                BEA_SYMEQ_MO_TC.append(bea_seq_mo)
                ___ = np.asarray([i for _ in range(bea_seq.shape[0])],
                                 dtype=np.int16)
                BEA_SYMEQ_MO_TC_ids.append(___)

            """
            3.
            Stack up all orientations in the above list.
            """
            # (FL | 2) > (TEX INST | 3) > (TEX COMP | 4)
            BEA_STACK = np.vstack(BEA_SYMEQ)
            BEA_MO_STACK = np.hstack(BEA_SYMEQ_MO_TC)
            BEA_MO_IDS_STACK = np.hstack(BEA_SYMEQ_MO_TC_ids)
            """
            4.
            Shuffle BEA_STACK 'nshuffles' bnumber of times for randomness
            """
            shuffle_ids = np.arange(BEA_STACK.shape[0])
            for _ in range(nshuffles):
                np.random.shuffle(shuffle_ids)
                BEA_STACK = BEA_STACK[shuffle_ids]
                BEA_MO_STACK = BEA_MO_STACK[shuffle_ids]
                BEA_MO_IDS_STACK = BEA_MO_IDS_STACK[shuffle_ids]
            """
            5.
            Store symmetric equivalnet angles.

            data access:
                [A] loc_symm_eq
                All symmetric equivalents of mean TC orientation
                1. symeq[tc_comp]['loc_symm_eq']
                -----------------------------------------------------------
                [B] BEA_SYMEQ
                Ori distr of each of above symm eq. list of 24 np arr
                1. symeq[tc_comp]['BEA_SYMEQ']
                2. symeq[tc_comp]['BEA_SYMEQ'][3]
                   Content: Bunge's EA triplets (BEA), centred around the
                            symeq, symeq[tc_comp]['loc_symm_eq'][3]
                2. len(symeq[tc_comp]['BEA_SYMEQ'][3])
                   Count: ori_count[tc_comp] or eacount in this loop
                -----------------------------------------------------------
                [C, D] BEA_shuffled, shuffle_ids
                BEA_shuffled: Randomly shuffled np arr,
                     X = np.vstack(symeq[tc_comp]['BEA_SYMEQ'])
                shuffle_ids: Shuffling order IDs in reference to np arr, X
                NOTE: BEA_shuffled = X[shuffle_ids]
                C1. symeq[tc_comp]['BEA_shuffled']
                C2. len(symeq[tc_comp]['BEA_shuffled'])
                D1. symeq[tc_comp]['shuffle_ids']
                D2. len(symeq[tc_comp]['shuffle_ids'])
                Counts of C and D: 24 * ori_count[tc_comp]
                -----------------------------------------------------------
                [E, F] BEA_SYMEQ_MO_TC, BEA_SYMEQ_MO_TC_ids
                BEA_SYMEQ_MO_TC
                1. symeq[tc_comp]['BEA_SYMEQ_MO_TC']
                   Content: list if 24 np. arrays
                1.1. symeq[tc_comp]['BEA_SYMEQ_MO_TC'][3]
                   Count: ori_count[tc_comp] or eacount in this loop
                   Content: misori ang b/w every angle in
                      symeq[tc_comp]['BEA_SYMEQ'][3] with
                      symeq[tc_comp]['loc_symm_eq'][3]

                symeq[tc_comp]['BEA_SYMEQ_MO_TC_ids']
                -----------------------------------------------------------
                symeq[tc_comp]['BEA_MO_STACK']

                len(symeq[tc_comp]['loc_symm_eq'])
                len(symeq[tc_comp]['BEA_SYMEQ'])
                len(symeq[tc_comp]['BEA_shuffled'])
                len(symeq[tc_comp]['shuffle_ids'])
                len(symeq[tc_comp]['BEA_SYMEQ_MO_TC'])
                len(symeq[tc_comp]['BEA_SYMEQ_MO_TC_ids'])
                len(symeq[tc_comp]['BEA_MO_STACK'])
            """
            # (FL | 2) > (TEX INST | 3) > (TEX COMP | 4)
            symeq[tc_comp] = {'loc_symm_eq': loc_symm_eq,
                              'BEA_SYMEQ': BEA_SYMEQ,
                              'BEA_shuffled': BEA_STACK,
                              'shuffle_ids': shuffle_ids,
                              'BEA_SYMEQ_MO_TC': BEA_SYMEQ_MO_TC,
                              'BEA_SYMEQ_MO_TC_ids': BEA_SYMEQ_MO_TC_ids,
                              'BEA_MO_STACK': BEA_MO_STACK,
                              'BEA_MO_IDS_STACK': BEA_MO_IDS_STACK
                              }
        return symeq

    def cubic_Ry(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])

    def sheet_D2_sample_ops(self):
        """Orthorhombic rolling-sheet sample symmetry (RD, TD, ND 180°)."""
        return [
            np.eye(3, dtype=float),
            self.cubic_Rx(np.pi),  # 180° about RD
            self.cubic_Ry(np.pi),  # 180° about TD
            self.cubic_Rz(np.pi),  # 180° about ND
        ]

    @staticmethod
    def _wrap_half_open(x, hi, eps=1e-12):
        """Wrap to [0, hi) with numerical guard."""
        x = np.mod(x, hi)
        # map exact/near hi back to 0 to keep half-open convention
        x[np.abs(x - hi) < eps] = 0.0
        return x

    def _canonicalize_on_boundaries(self, ea, eps=1e-8):
        """
        Strong Bunge FZ canonicalization for cubic + D2:
          - phi1, phi2 ∈ [0, 90)   (half-open)
          - Phi ∈ [0, 90]          (closed)
        Tie-breaks:
          * If Phi≈0: set phi2=0, wrap phi1 to [0,90)
          * If Phi≈90: SHIFT AMBIGUITY away from phi2: set phi1←phi1+phi2, phi2=0, wrap phi1 to [0,90)
          * Otherwise: wrap phi1, phi2 to [0,90); clamp tiny Phi jitters.
        """
        phi1, Phi, phi2 = map(float, ea)

        def wrap90(x):
            y = x % 90.0
            # map nearly-90 back to 0 for half-open
            if abs(y - 90.0) < eps:
                y = 0.0
            return y

        # snap Phi to [0,90] with small jitter tolerance
        if abs(Phi) < eps:
            Phi = 0.0
            phi2 = 0.0
            phi1 = wrap90(phi1)
            return np.array([phi1, Phi, phi2], float)

        if abs(Phi - 90.0) < eps:
            Phi = 90.0
            # resolve ZXZ degeneracy at Phi=90 by absorbing phi2 into phi1
            phi1 = wrap90(phi1 + phi2)
            phi2 = 0.0
            return np.array([phi1, Phi, phi2], float)

        # general case
        Phi = max(0.0, min(90.0, Phi))
        phi1 = wrap90(phi1)
        phi2 = wrap90(phi2)
        return np.array([phi1, Phi, phi2], float)

    def _in_FZ(self, ea, eps=1e-8):
        """Check Bunge FZ: phi1,phi2 in [0,90), Phi in [0,90] (with eps)."""
        phi1, Phi, phi2 = ea
        return (
            (-eps <= phi1 < 90.0 - eps + 1e-15) and
            (-eps <= phi2 < 90.0 - eps + 1e-15) and
            (-eps <= Phi <= 90.0 + eps)
        )

    # ---------- CORE: variants in FZ for a single Euler triple ----------

    def fcc_variants_in_FZ_old(self, ea_bunge, *, degrees=True, tol=1e-8):
        """
        Return ALL unique variants of an orientation inside the Bunge FZ
        for cubic (24) crystal + orthorhombic-sheet D2 (4) sample symmetry.
        Uniqueness is by rotation (not angles), and angles are canonicalized.

        Parameters
        ----------
        ea_bunge : (3,) iterable
          (phi1, Phi, phi2) in degrees by default.
        degrees : bool
        tol : float
          Frobenius tolerance for rotation de-duplication.

        Returns
        -------
        variants_FZ : (k,3) float ndarray (degrees)
        """
        g = self._as_rotmat(ea_bunge, degrees=degrees)  # 3x3
        S = self._get_cubic_ops_np()                    # (24,3,3)
        R_sample = self.sheet_D2_sample_ops()           # list of 4

        # Full orbit: S @ g @ R
        mats = []
        for Si in S:
            L = Si @ g
            for Rr in R_sample:
                mats.append(self._proj_to_so3(L @ Rr))

        # Deduplicate rotations first (avoid near-identical images)
        mats = self._unique_rotations(mats, tol=tol)

        # Map to Euler, canonicalize to FZ, keep only those inside
        eulers = []
        rot_keep = []
        for M in mats:
            ea = np.array(self._matrix_to_euler_bunge(M, degrees=True), dtype=float)
            # reduce to global canonical ranges first
            ea = self.normalize_euler_bunge(ea, degrees=True, eps=1e-10)
            # apply FZ boundary tie-breaks
            ea = self._canonicalize_on_boundaries(ea, eps=1e-8)
            if self._in_FZ(ea, eps=1e-8):
                eulers.append(ea)
                rot_keep.append(M)

        if not eulers:
            return np.empty((0, 3), dtype=float)

        # Deduplicate again by rotation to ensure uniqueness within FZ
        uniq = []
        uniq_ea = []
        for M, ea in zip(rot_keep, eulers):
            if not any(np.linalg.norm(M - Q, ord='fro') < tol for Q in uniq):
                uniq.append(M)
                uniq_ea.append(ea)

        return np.asarray(uniq_ea, dtype=float)

    # ---------- BATCH: variants per TC in self.tc_info ----------

    def tc_variants_in_FZ(self):
        """
        For every TC mean orientation in self.tc_info, compute the
        unique FZ variants (cubic + D2). Returns dict: name -> (k,3) array.
        """
        if not hasattr(self, 'tc_info'):
            raise RuntimeError("tc_info not set. Call set_tc_info(...) first.")

        out = {}
        for name, vals in self.tc_info.items():
            ea = vals[1]  # [phi1, Phi, phi2]
            out[name] = self.fcc_variants_in_FZ(ea, degrees=True)
        return out

    def _unique_rows_ea(self, ea_list, tol=1e-8):
        if not ea_list:
            return np.empty((0,3), float)
        A = np.asarray(ea_list, float)
        G = np.round(A / tol) * tol
        # stable lexicographic unique
        keys = G[:,0] * 1e6 + G[:,1] * 1e3 + G[:,2]
        order = np.argsort(keys)
        G = G[order]
        keep = [0]
        for i in range(1, G.shape[0]):
            if not np.allclose(G[i], G[keep[-1]], atol=tol, rtol=0):
                keep.append(i)
        return G[keep]

    def fcc_variants_in_FZ(self, ea_bunge, *, degrees=True, tol=1e-8):
        """
        Unique FZ variants for cubic crystal (24 left ops).
        Sample symmetry (D2, 4 right ops) is used ONLY to fold into the FZ.
        """
        g = self._as_rotmat(ea_bunge, degrees=degrees)   # 3x3
        S = self._get_cubic_ops_np()                      # (24,3,3)
        R_sample = self.sheet_D2_sample_ops()             # 4 right ops

        canon_ea = []
        for Si in S:
            L = Si @ g  # crystal orbit element

            # fold to FZ: try all 4 sample right-ops, pick canonical rep
            candidates = []
            for Rr in R_sample:
                M = self._proj_to_so3(L @ Rr)
                ea = np.array(self._matrix_to_euler_bunge(M, degrees=True), float)
                ea = self.normalize_euler_bunge(ea, degrees=True, eps=1e-10)
                ea = self._canonicalize_on_boundaries(ea, eps=1e-8)
                if self._in_FZ(ea, eps=1e-8):
                    candidates.append(ea)

            if not candidates:
                # If none land exactly in the FZ due to tiny jitter, keep the
                # lexicographically smallest after clamping Phi to [0,90]
                continue

            # choose a single canonical representative for this crystal image
            C = np.asarray(candidates)
            # lexicographic min after small rounding grid to be stable
            key = np.round(C / tol) * tol
            idx = np.lexsort((key[:,2], key[:,1], key[:,0]))[0]
            canon_ea.append(C[idx])

        # final dedupe by canonical Euler rows
        variants_FZ = self._unique_rows_ea(canon_ea, tol=1e-8)
        return variants_FZ
