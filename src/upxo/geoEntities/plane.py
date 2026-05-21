"""3D plane geometric entity for UPXO.

This module provides a lightweight :class:`Plane` class with helpers for
construction, projection, distance calculations, parallel stacks, and
visualisation in 3D.

Classes
-------
Plane
    A 3D plane defined by a point and a normal vector.

Usage
-----
::

    from upxo.geoEntities.plane import Plane

@author: Dr. Sunil Anandatheertha
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plane:
    """Represent a 3D plane using a point and a normal vector."""

    def __init__(self, point, normal):
        """
        Initialize a plane from a point and a normal vector.

        Parameters
        ----------
        point : array-like
            Any 3-component point on the plane.
        normal : array-like
            3-component normal vector of the plane.
        """
        self.point = np.array(point).astype(float)
        self.normal = np.array(normal).astype(float)

    def __repr__(self):
        """Return a concise string representation of the plane."""
        return f"Plane(point={[round(xyz, 6) for xyz in self.point]}, normal={[round(x, 6) for x in self.normal]})"

    @classmethod
    def from_three_points(cls, point1, point2, point3):
        """
        Construct a plane from three non-collinear points.

        Parameters
        ----------
        point1 : array-like
            First point on the plane.
        point2 : array-like
            Second point on the plane.
        point3 : array-like
            Third point on the plane.

        Returns
        -------
        Plane
            Plane instance passing through the three points.

        Raises
        ------
        ValueError
            If any input point is invalid, if any two input points coincide,
            or if the three points are collinear.

        Notes
        -----
        The normal is computed as ``cross(point2 - point1, point3 - point1)``.
        Input points must not be collinear.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            point1 = np.array([1, 0, 2])
            point2 = np.array([0, 2, 1])
            point3 = np.array([3, 1, -1])
            plane = Plane.from_three_points(point1, point2, point3)
            print(plane)
        """
        from upxo._sup.validation_values import val_point_and_get_coord
        point1 = np.array(val_point_and_get_coord(point=point1, return_type='coord', safe_exit=False), dtype=float,)
        point2 = np.array(val_point_and_get_coord(point=point2, return_type='coord', safe_exit=False), dtype=float,)
        point3 = np.array(val_point_and_get_coord(point=point3, return_type='coord', safe_exit=False), dtype=float,)
        if np.allclose(point1, point2) or np.allclose(point1, point3) or np.allclose(point2, point3):
            raise ValueError('Input points must be distinct.')
        vector1, vector2 = point2-point1, point3-point1
        normal = np.cross(vector1, vector2)
        if np.allclose(normal, 0.0):
            raise ValueError('Input points must not be collinear.')
        return cls(point1, normal)

    @classmethod
    def from_edge(cls, point1, point2, f=0.5):
        """
        Create a plane normal to an edge and passing through a point on it.

        Parameters
        ----------
        point1 : array-like
            Start point of the edge.
        point2 : array-like
            End point of the edge.
        f : float, optional
            Fraction along the edge in ``[0, 1]`` where the plane passes.

        Returns
        -------
        Plane
            Plane with normal parallel to ``point2 - point1``.

        Notes
        -----
        ``f`` is clipped into ``[0, 1]`` before use.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            point1 = np.array([1, 0, 2])
            point2 = np.array([3, 2, 0])

            midpoint_plane = Plane.from_edge(point1, point2)
            three_quarter_plane = Plane.from_edge(point1, point2, f=0.75)
        """
        f = np.clip(f, 0, 1)
        point_on_edge = point1+f*(point2-point1)
        edge_vector = point2-point1
        normal = edge_vector
        return cls(point=point_on_edge, normal=normal)

    @classmethod
    def from_euler_angles(cls, euler_angles, point_on_plane, ea_format='rpy', degree=False):
        """
        Construct plane from Euler-angle orientation and a point.

        Parameters
        ----------
        euler_angles : tuple or list
            Euler angles corresponding to ``ea_format``.
        point_on_plane : tuple or list
            Coordinates ``(x, y, z)`` of a point on the plane.
        ea_format : {'rpy', 'bunge', 'roe'}, optional
            Convention used for ``euler_angles``.
        degree : bool, optional
            If True, interpret Euler angles in degrees.

        Returns
        -------
        Plane
            Plane instance whose normal is derived from Euler rotations.

        Notes
        -----
        The method rotates the reference normal ``[0, 0, 1]`` using the
        convention-specific rotation mapping.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            Plane.from_euler_angles((0, np.pi/2, np.pi/2), (1, 1, 1), ea_format='rpy', degree=False)
            Plane.from_euler_angles((0, np.pi/2, np.pi/2), (1, 1, 1), ea_format='roe', degree=False)
            Plane.from_euler_angles((0, np.pi/2, np.pi/2), (1, 1, 1), ea_format='bunge', degree=False)
            Plane.from_euler_angles((0, np.pi/7, np.pi/2), (1, 1, 1), ea_format='bunge', degree=False)
        """
        if ea_format == 'rpy':
            roll, pitch, yaw = euler_angles
        elif ea_format == 'bunge':
            phi1, Phi, phi2 = euler_angles
            roll = phi1
            pitch = np.arccos(np.cos(Phi)*np.cos(phi2))
            yaw = np.arcsin(np.sin(Phi)*np.sin(phi2))
        elif ea_format == 'roe':
            alpha, beta, gamma = euler_angles
            phi = alpha + np.arctan2((sinbeta := np.sin(beta)), (cosbeta := np.cos(beta)))+gamma
            theta = np.arccos(np.cos(alpha)*cosbeta)
            psi = -alpha + np.arctan2(sinbeta, cosbeta)-gamma
            roll, pitch, yaw = phi, theta, psi
        else:
            raise ValueError('Invalid ea_format specification.')

        if degree:
            roll, pitch, yaw = np.radians([roll, pitch, yaw])

        cosroll, sinroll = np.cos(roll), np.sin(roll)
        cospitch, sinpitch = np.cos(pitch), np.sin(pitch)
        cosyaw, sinyaw = np.cos(yaw), np.sin(yaw)

        R_roll = np.array([[1, 0, 0], [0, cosroll, -sinroll], [0, sinroll, cosroll]])
        R_pitch = np.array([[cospitch, 0, sinpitch], [0, 1, 0], [-sinpitch, 0, cospitch]])
        R_yaw = np.array([[cosyaw, -sinyaw, 0], [sinyaw, cosyaw, 0], [0, 0, 1]])

        R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        normal_vector = np.dot(R, np.array([0, 0, 1]))

        return cls(point_on_plane, normal_vector)

    @property
    def unit_normal(self):
        """
        Return the unit normal vector of the plane.

        Returns
        -------
        numpy.ndarray
            Unit vector in the direction of the plane normal.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane

            Plane(point=(1, 1, 1), normal=(2, -1, 3)).unit_normal
        """
        return self.normal / np.linalg.norm(self.normal)

    def distance_to_point(self, point):
        """
        Calculate signed distance from the plane to a point.

        Positive when the point is on the same side as the normal, negative
        otherwise.

        Parameters
        ----------
        point : array-like
            3D point coordinates.

        Returns
        -------
        float
            Signed perpendicular distance from the plane to ``point``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            plane = Plane(point=(0, 0, 0), normal=(0, 0, 1))
            plane.distance_to_point(np.array([1, 2, 3]))  # returns 3.0
        """
        vector_to_point = point - self.point
        return np.dot(vector_to_point, self.normal) / np.linalg.norm(self.normal)

    def calc_perp_distances(self, points, signed=True):
        """
        Compute perpendicular distances from plane to one or more points.

        Parameters
        ----------
        points : array-like
            Single 3D point or array of shape ``(n, 3)``.
        signed : bool, optional
            If True, return signed distances. If False, return absolute values.

        Returns
        -------
        numpy.ndarray
            Distance(s) from the plane to the input point set.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            plane = Plane(point=(1, 1, 0), normal=(1, 1, 2))
            points = [np.array([0, 2, 1]), np.array([3, 0, -1]),
                      np.array([2, 2, 2])]
            plane.calc_perp_distances(points)
        """
        points = np.reshape(points, (-1, 3))
        vector_to_point = points-self.point
        distances = np.dot(vector_to_point, self.normal)/np.linalg.norm(self.normal)
        if not signed:
            distances = abs(distances)
        return distances

    def find_close_points(self, point_coords, cod=0.25):
        """
        Find points within a cutoff distance from the plane.

        Parameters
        ----------
        point_coords : array-like
            Candidate points of shape ``(n, 3)``.
        cod : float, optional
            Cutoff distance threshold.

        Returns
        -------
        None
            Current implementation computes distances but does not yet return
            selected points.

        Notes
        -----
        To be developed.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            import numpy as np

            plane = Plane(point=(0.5, 0.5, 0.5), normal=(1, 1, 1))
            xspec, yspec, zspec = [0, 1, 0.2], [0, 1, 0.2], [0, 1, 0.2]
            mulpoint3d = mp3d.from_xyz_grid(
                xspec=xspec, yspec=yspec, zspec=zspec,
                dxyz=[0.0, 0.0, 0.0], translate_ref=[0.5, 0.5, 0.5],
                rot=[0.0, 0.0, 0.0], rot_ref=[0.5, 0.5, 0.5], degree=True)
            D = plane.calc_perp_distances(mulpoint3d.coords, signed=False)
            cod = 0.1
            coords_within_cod = mulpoint3d.coords[np.argwhere(D <= cod)].squeeze()
            mulpoint3d.plot(coords_within_cod, primary_ms=25, primary_alpha=0.0,
                    secondary_alpha=0.25, xbound=xspec[:2], ybound=yspec[:2],
                    zbound=zspec[:2])
        """
        D = self.calc_perp_distances(point_coords, signed=False)

    def create_parallel_stack(self, spacing, num_planes):
        """
        Create a stack of planes parallel to this one.

        Parameters
        ----------
        spacing : float
            The perpendicular distance between each parallel plane.
        num_planes : int
            The total number of planes to generate in the stack.

        Returns
        -------
        list
            A list of new ``Plane`` objects.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane

            base = Plane(point=(0, 0, 0), normal=(0, 0, 1))
            stack = base.create_parallel_stack(spacing=0.5, num_planes=5)
            for p in stack:
                print(p)
        """
        unit_n = self.unit_normal
        new_planes = []
        for i in range(num_planes):
            new_point = self.point+(i*spacing*unit_n)
            new_planes.append(Plane(point=new_point, normal=self.normal))
        return new_planes

    def project_point(self, point):
        """
        Project a point onto the plane.

        Parameters
        ----------
        point : array-like
            3D point to project.

        Returns
        -------
        numpy.ndarray
            Projection of ``point`` onto the plane surface.

        Notes
        -----
        The signed distance from the plane to the point is computed first
        (via :meth:`distance_to_point`), then the point is translated along
        the normal by that distance to reach its closest position on the plane.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane

            Plane(point=(1, 1, 1), normal=(1, 1, 1)).project_point((0, 0, 0))
            Plane(point=(1, 1, 1), normal=(-1, -1, -1)).project_point((0, 0, 0))
        """
        distance = self.distance_to_point(point)
        projection_vector = distance * self.normal
        return point - projection_vector

    def generate_random_points(self, num_points=3):
        """
        Generate random points lying on the plane.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate.

        Returns
        -------
        list
            Random points represented as coordinate tuples.

        Notes
        -----
        Two in-plane orthogonal directions are constructed from the normal,
        then random linear combinations are sampled.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane

            plane = Plane(point=(0, 0, 0), normal=(0, 0, 1))
            plane.generate_random_points(3)
        """
        if np.abs(self.normal[0]) > np.abs(self.normal[1]):
            tangent = np.cross(self.normal, [0, 1, 0])
        else:
            tangent = np.cross(self.normal, [1, 0, 0])
        tangent = tangent / np.linalg.norm(tangent)
        perp_tangent = np.cross(self.normal, tangent)
        s, t = np.random.random((num_points, 2)).T
        rpoints = [tuple(self.point+_s_*tangent+_t_*perp_tangent) for _s_, _t_ in zip(s, t)]
        return rpoints

    def flip_normal(self):
        """
        Flip the direction of the plane normal.

        Notes
        -----
        Flipping the normal changes the side of the plane that is considered
        the "front." Be mindful of how this affects signed distance calculations
        and geometric operations that depend on normal orientation.

        To be developed.
        """
        pass

    def is_parallel(self, other_plane):
        """
        Check whether this plane is parallel to another plane.

        Parameters
        ----------
        other_plane : Plane
            Plane to compare against.

        Returns
        -------
        bool
            True if normals are parallel (or anti-parallel), else False.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane

            plane1 = Plane(point=(1, 1, 1), normal=(+2, -1, +3))
            plane2 = Plane(point=(0, 0, 0), normal=(+2, -1, +3))
            plane3 = Plane(point=(1, 1, 1), normal=(-2, +1, -3))
            plane4 = Plane(point=(1, 1, 1), normal=(-2, -1, -3))

            plane1.is_parallel(plane1)  # True
            plane1.is_parallel(plane2)  # True
            plane1.is_parallel(plane3)  # True (anti-parallel normals)
            plane1.is_parallel(plane4)  # False
        """
        return np.all(np.cross(self.normal, other_plane.normal) == 0)

    def find_intersection_vector(self, plane):
        """Find direction vector of intersection line between two planes.

        Parameters
        ----------
        plane : Plane
            Second plane.

        Returns
        -------
        numpy.ndarray or None
            Direction vector of line of intersection, or ``None`` for parallel
            planes.

        Notes
        -----
        This method returns direction only; computing a point on the
        intersection line requires additional constraints.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane

            plane1 = Plane(point=(1, 1, 0), normal=(1, 1, 2))
            plane2 = Plane(point=(0, 2, 1), normal=(-2, -1, 1))
            intersection_vector = plane1.find_intersection_vector(plane2)
            print(intersection_vector)
        """
        if np.all(np.cross(self.normal, plane.normal) == 0):
            return None
        direction_vector = np.cross(self.normal, plane.normal)
        return direction_vector

    def create_translated_planes(self, translation_vector, num_planes,
            dlk=np.array([1.0, -1.0, 1.0]), dnw=np.array([0.5, 0.5, 0.5]),
            dno=np.array([0.5, 0.5, 0.5]), bidrectional=False,):
        """Create translated copies of the current plane.

        Parameters
        ----------
        translation_vector : array-like
            Base translation increment applied between planes.
        num_planes : int
            Number of planes to generate (including self in one-sided mode).
        dlk : numpy.ndarray, optional
            Random translational perturbation scale.
        dnw : numpy.ndarray, optional
            Random normal perturbation scale.
        dno : numpy.ndarray, optional
            Offset term used with normal perturbation.
        bidrectional : bool, optional
            If True, generate planes on both positive and negative directions.

        Returns
        -------
        numpy.ndarray
            Array of ``Plane`` objects.

        Examples
        --------
        **Example 1** — simple translated stack:

        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            plane = Plane(point=(0, 0, 0), normal=(1, 1, 1))
            translated = plane.create_translated_planes(
                np.array([2, 3, -1]), num_planes=4)
            print(translated)

        **Example 2** — translated planes with point filtering and plotting:

        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
            import numpy as np
            import matplotlib.pyplot as plt

            xspec, yspec, zspec = [0, 1, 0.05], [0, 1, 0.05], [0, 1, 0.05]
            mpnt3d = mp3d.from_xyz_grid(
                xspec=xspec, yspec=yspec, zspec=zspec,
                dxyz=[0.0, 0.0, 0.0], translate_ref=[0.5, 0.5, 0.5],
                rot=[0.0, 0.0, 0.0], rot_ref=[0.5, 0.5, 0.5], degree=True)
            plane = Plane(point=(0.0, 0.0, 0.0), normal=(1, 1, 1))
            planes = plane.create_translated_planes(np.array([0.2, 0.2, 0.2]),
                                                    num_planes=8)
            D = [p.calc_perp_distances(mpnt3d.coords, signed=False) for p in planes]
            cod = 0.125
            coords = [mpnt3d.coords[np.argwhere(d <= cod)].squeeze() for d in D]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(mpnt3d.coords[:, 0], mpnt3d.coords[:, 1],
                       mpnt3d.coords[:, 2], c='b', marker='o', alpha=0.01,
                       s=100, edgecolors='black')
            for coord in coords:
                if coord is not None:
                    ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2],
                               c=np.random.random(3), marker='o', alpha=0.8,
                               s=50, edgecolors='black')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            plt.show()

        **Example 3** — with perturbation parameters:

        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            plane1 = Plane(point=(0, 0, 0), normal=(0, 1, 2))
            planes = plane1.create_translated_planes(
                np.array([1, -1, 1]), num_planes=10,
                dlk=np.array([0.0, 0.0, 0.0]),
                dnw=np.array([0.1, 0.0001, 0.0]),
                dno=np.array([0.5, 0.5, 0.5]))
            print(planes)
        """
        if bidrectional:
            tv = translation_vector
            tps1 = self.create_translated_planes(tv, num_planes, dlk=dlk, dnw=dnw, dno=dno)
            tps2 = self.create_translated_planes(-tv, num_planes, dlk=dlk, dnw=dnw, dno=dno)
            tr_planes = tuple(tps1)+tuple(tps2)
        else:
            npl = num_planes-1
            dl = dlk * np.random.random((npl, 3))
            TV = np.arange(1, npl+1)[:, np.newaxis]*np.tile(translation_vector, (npl, 1)) + dl
            TV = TV + self.point
            NR = np.tile(self.normal, (npl, 1)) + dnw*(dno-np.random.random((npl, 3)))
            tr_planes = [self]
            for tv, nr in zip(TV, NR):
                tr_planes.append(Plane(point=tv, normal=nr))

        return np.array(tr_planes)

    def offset_point_on_plane(self, point, offset_vector):
        """Offset a point on the plane by an in-plane displacement vector.

        Parameters
        ----------
        point : array-like
            Point constrained to lie on the plane.
        offset_vector : array-like
            Desired offset vector (possibly containing out-of-plane component).

        Returns
        -------
        numpy.ndarray
            New point after applying in-plane component of offset.

        Raises
        ------
        ValueError
            If ``point`` does not lie on the plane.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            my_plane = Plane(point=(2, -1, 0), normal=(0, 1, 1))
            point_on_plane = np.array([2, -1, 0])
            offset_vector = np.array([1, -1, 2])
            offset_point = my_plane.offset_point_on_plane(point_on_plane, offset_vector)
            print(offset_point)
        """
        distance_to_plane = self.distance_to_point(point)
        if not np.isclose(distance_to_plane, 0):
            raise ValueError("The input point does not lie on the plane.")
        projection_onto_plane = self.project_point(offset_vector)
        in_plane_offset = offset_vector - projection_onto_plane
        return point + in_plane_offset

    def calculate_inclined_circle(self, center, radius, angle_A, angle_B, angle_C):
        """Compute points of an inclined circle associated with the plane.

        Parameters
        ----------
        center : array-like
            Circle center.
        radius : float
            Circle radius.
        angle_A : float
            Inclination rotation about x-axis (radians).
        angle_B : float
            Inclination rotation about y-axis (radians).
        angle_C : float
            Inclination rotation about z-axis (radians).

        Returns
        -------
        numpy.ndarray
            Rotated circle points of shape ``(num_points, 3)``.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            my_plane = Plane(point=(1, 0, 2), normal=(0, 1, -1))
            center = np.array([1, 2, 2])
            radius = 1.5
            angle_A = np.radians(30)
            angle_B = np.radians(-20)
            angle_C = np.radians(60)
            circle_points = my_plane.calculate_inclined_circle(
                center, radius, angle_A, angle_B, angle_C)
            my_plane.visualize(circle_points)
        """
        Rx = self._rotation_matrix_x(angle_A)
        Ry = self._rotation_matrix_y(angle_B)
        Rz = self._rotation_matrix_z(angle_C)
        circle_x_axis = self._find_orthogonal_vector(self.normal)
        circle_points = self._generate_circle_points(center, radius, circle_x_axis)
        rotated_circle = np.dot(Rx @ Ry @ Rz, circle_points.T).T
        return rotated_circle

    def _generate_circle_points(self, center, radius, axis):
        """
        Generate sample points for a circle centered at ``center``.

        Parameters
        ----------
        center : array-like
            Circle center.
        radius : float
            Circle radius.
        axis : array-like
            Reference axis for circle construction.

        Returns
        -------
        numpy.ndarray
            Array of sampled circle points.
        """
        num_points = 20
        theta = np.linspace(0, 2*np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(theta)
        circle_points = np.stack([x, y, z], axis=-1)+center
        return circle_points

    def _rotation_matrix_x(self, angle):
        """Return 3x3 rotation matrix about x-axis for ``angle`` radians."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _rotation_matrix_y(self, angle):
        """Return 3x3 rotation matrix about y-axis for ``angle`` radians."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _rotation_matrix_z(self, angle):
        """Return 3x3 rotation matrix about z-axis for ``angle`` radians."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _find_orthogonal_vector(self, vector):
        """
        Find a non-zero vector orthogonal to input vector.

        Parameters
        ----------
        vector : array-like
            Input 3D vector.

        Returns
        -------
        numpy.ndarray
            A vector orthogonal to ``vector``.
        """
        if np.abs(vector[0]) > np.abs(vector[1]):
            return np.cross(vector, [0, 1, 0])
        else:
            return np.cross(vector, [1, 0, 0])

    def visualize(self, circle_points=None):
        """Visualize plane and optional circle in a 3D matplotlib figure.

        Parameters
        ----------
        circle_points : array-like, optional
            Optional ``(n, 3)`` points to overlay on the plane.

        Returns
        -------
        None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
        z = (-self.normal[0]*x-self.normal[1]*y-self.point[2])/self.normal[2]
        ax.plot_surface(x, y, z, alpha=0.5)
        if circle_points is not None:
            ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def visualize1(self, circle_points=None, other_planes=None):
        """
        Visualize current plane, optional circle, and optional extra planes.

        Parameters
        ----------
        circle_points : array-like, optional
            Optional ``(n, 3)`` circle points to plot.
        other_planes : list, optional
            Additional ``Plane`` objects to render.

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            plane1 = Plane(point=(1, 0, 2), normal=(0, 1, -1))
            plane2 = Plane(point=(-1, 1, 0), normal=(1, 1, 1))
            circle_points = plane1.calculate_inclined_circle(
                center=[1, 2, 2], radius=1.5,
                angle_A=np.radians(30), angle_B=np.radians(-20),
                angle_C=np.radians(60))
            plane1.visualize1(circle_points, other_planes=[plane2])
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        if other_planes:
            for plane in other_planes:
                x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
                z = (-plane.normal[0]*x-plane.normal[1]*y-plane.point[2])/plane.normal[2]
                ax.plot_surface(x, y, z, alpha=0.3)
        midpoint = self.point+0.5*self.normal
        ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.normal[0], self.normal[1], self.normal[2], color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def angle_between_planes(self, other_plane):
        """Calculate angle between this plane and another plane.

        Parameters
        ----------
        other_plane : Plane
            Plane to compare.

        Returns
        -------
        float
            Angle between plane normals in radians.

        Examples
        --------
        .. code-block:: python

            from upxo.geoEntities.plane import Plane
            import numpy as np

            plane1 = Plane(point=(0, 0, 0), normal=(1, 1, 0))
            plane2 = Plane(point=(0, 0, 0), normal=(1, 0, 0))
            intersection_angle = plane1.angle_between_planes(plane2)
            print(f"Angle (radians): {intersection_angle}")
            print(f"Angle (degrees): {np.degrees(intersection_angle)}")
        """
        angle_cos = np.dot(self.normal, other_plane.normal) / \
                    (np.linalg.norm(self.normal)*np.linalg.norm(other_plane.normal))
        angle = np.arccos(np.clip(angle_cos, -1.0, 1.0))
        return angle
