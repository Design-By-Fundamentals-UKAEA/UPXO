import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plane:
    def __init__(self, point, normal):
        self.point = np.array(point).astype(float)
        self.normal = np.array(normal).astype(float)

    def __repr__(self):
        return f"Plane(point={[round(xyz, 6) for xyz in self.point]}, normal={[round(x, 6) for x in self.normal]})"

    @classmethod
    def from_three_points(cls, point1, point2, point3):
        """
        Constructs a plane from three non-collinear points.

        Explanation
        -----------
        Class Method:
            We use @classmethod to indicate that this method works with the
            Plane class itself, not just specific instances.
        Vectors in the Plane:
            We calculate two vectors lying within the plane by subtracting the
            coordinates of the three provided points.
        Normal Vector:
            The cross product of two non-parallel vectors within the plane
            gives us a normal vector to the plane.
        Plane Creation:
            We now have a normal vector and a point on the plane (point1). We
            can directly use the existing __init__ method of the Plane class to
            create the plane object.

        Note
        ----
        Make sure the three points are not collinear (on the same line).
        Otherwise, you cannot uniquely define a plane.

        Example
        -------
        from upxo.geoEntities.plane import Plane
        point1 = np.array([1, 0, 2])
        point2 = np.array([0, 2, 1])
        point3 = np.array([3, 1, -1])

        plane_from_points = Plane.from_three_points(point1, point2, point3)
        print(plane_from_points)
        """
        # Calculate two vectors lying within the plane
        vector1 = point2 - point1
        vector2 = point3 - point1
        # Normal vector is the cross product of the vectors within the plane
        normal = np.cross(vector1, vector2)
        # Use any of the three points on the plane
        return cls(point1, normal)

    @classmethod
    def from_edge(cls, point1, point2, f=0.5):
        """
        Creates a plane that's normal to an edge and passes through a
        point on the edge at a factor f from the starting point.

        Args:
            point1: A NumPy array representing the starting point of the edge.
            point2: A NumPy array representing the ending point of the edge.
            f: A factor between 0 and 1, determining the point on the edge
               where the plane passes through (default is 0.5, the midpoint).

        Returns:
            A Plane object that's normal to the edge and passes through the
            specified point on the edge.

        Explanation

        Class Method: @classmethod allows for a flexible way to construct a
        plane object.

        Explanations
        ------------
        The method takes the edge's starting point (point1), ending point
        (point2), and a factor f. It computes the point on the edge where the
        plane will pass through based on the factor f.
        Normal Vector: The direction vector of the edge (point2 - point1) is
        the normal vector of the plane we want to create
        (since we need the plane  perpendicular to the edge).
        Plane Creation: We use the computed point and normal to directly create
        a Plane object.

        Example
        -------
        from upxo.geoEntities.plane import Plane
        point1 = np.array([1, 0, 2])
        point2 = np.array([3, 2, 0])

        # Plane passing through the midpoint of the edge
        midpoint_plane = Plane.from_edge(point1, point2)

        # Plane passing through a point 3/4 of the way along the edge
        three_quarter_plane = Plane.from_edge(point1, point2, f=0.75)
        """
        # Ensure f is between 0 and 1
        f = np.clip(f, 0, 1)
        # Calculate the point on the edge
        point_on_edge = point1 + f * (point2 - point1)
        # Direction vector of the edge is the normal of the plane
        edge_vector = point2 - point1
        normal = edge_vector
        # Create the plane
        return cls(point=point_on_edge, normal=normal)

    @classmethod
    def from_euler_angles(cls, euler_angles, point_on_plane, ea_format='rpy', degree=False):
        """
        Calculate the point and normal vector of a plane using Euler angles.

        Parameters:
            euler_angles (tuple or list): Euler angles in radians (roll, pitch,
                                                                   yaw).
            point_on_plane (tuple or list): Coordinates of a point on the plane
            (x, y, z).

        Returns:
            tuple: A tuple containing the point on the plane (x, y, z) and the
            normal vector (nx, ny, nz).
        Examples
        --------
        from upxo.geoEntities.plane import Plane
        Plane.from_euler_angles((0,np.pi/2,np.pi/2), (1, 1, 1),
                                ea_format='rpy', degree=False)
        Plane.from_euler_angles((0,np.pi/2,np.pi/2), (1, 1, 1),
                                ea_format='roe', degree=False)
        Plane.from_euler_angles((0,np.pi/2,np.pi/2), (1, 1, 1),
                                ea_format='bunge', degree=False)
        Plane.from_euler_angles((0,np.pi/7,np.pi/2), (1, 1, 1),
                                ea_format='bunge', degree=False)
        """
        # Validations
        # ---------------------------------------------
        if ea_format == 'rpy':
            # Roll Pitch Yaw. Do nothing.
            roll, pitch, yaw = euler_angles
        elif ea_format == 'bunge':
            # Convert Bunge's Euler angles to Euler angles (roll, pitch, yaw).
            phi1, Phi, phi2 = euler_angles
            roll = phi1
            pitch = np.arccos(np.cos(Phi) * np.cos(phi2))
            yaw = np.arcsin(np.sin(Phi) * np.sin(phi2))
        elif ea_format == 'roe':
            # Convert Rotation of Axes (RoE) Euler angles to standard Euler
            # angles.
            alpha, beta, gamma = euler_angles
            phi = alpha + np.arctan2((sinbeta := np.sin(beta)),
                                     (cosbeta := np.cos(beta))) + gamma
            theta = np.arccos(np.cos(alpha) * cosbeta)
            psi = -alpha + np.arctan2(sinbeta, cosbeta) - gamma
            roll, pitch, yaw = phi, theta, psi
            """
            # Original set6 of codes before I learnt the := operator :D
            alpha, beta, gamma = euler_angles
            cosbeta, sinbeta = np.cos(beta), np.sin(beta)
            phi = alpha + np.arctan2(sinbeta, cosbeta) + gamma
            theta = np.arccos(np.cos(alpha) * np.cos(beta))
            psi = -alpha + np.arctan2(sinbeta, cosbeta) - gamma
            roll, pitch, yaw = phi, theta, psi
            """
        else:
            raise ValueError('Invalid ea_format specification.')
        # ---------------------------------------------
        if degree:
            roll, pitch, yaw = np.radians([roll, pitch, yaw])
        # ---------------------------------------------
        cosroll, sinroll = np.cos(roll), np.sin(roll)
        cospitch, sinpitch = np.cos(pitch), np.sin(pitch)
        cosyaw, sinyaw = np.cos(yaw), np.sin(yaw)
        # ---------------------------------------------
        # Rotation matrices
        R_roll = np.array([[1, 0, 0],
                           [0, cosroll, -sinroll],
                           [0, sinroll, cosroll]])
        R_pitch = np.array([[cospitch, 0, sinpitch],
                            [0, 1, 0],
                            [-sinpitch, 0, cospitch]])
        R_yaw = np.array([[cosyaw, -sinyaw, 0],
                          [sinyaw, cosyaw, 0],
                          [0, 0, 1]])
        # ---------------------------------------------
        # Combined rotation matrix
        R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        # ---------------------------------------------
        # Normal vector along the z-axis
        normal_vector = np.dot(R, np.array([0, 0, 1]))
        # ---------------------------------------------
        return cls(point_on_plane, normal_vector)

    @property
    def unit_normal(self):
        """
        Return the unit normal vector of the plane.

        Example
        -------
        from upxo.geoEntities.plane import Plane
        Plane(point=(1, 1, 1), normal=(2, -1, 3)).unit_normal
        """
        return self.normal / np.linalg.norm(self.normal)

    def distance_to_point(self, point):
        """
        Calculates the signed distance between a point and the plane.

        (Positive if the point is on the same side of the plane as the normal
         vector, negative otherwise).
        """
        vector_to_point = point - self.point
        return np.dot(vector_to_point,
                      self.normal) / np.linalg.norm(self.normal)

    def calc_perp_distances(self, points, signed=True):
        """
        Calculates the signed perpendicular distances from the plane to a
        list of 3D points.

        If the distance is positive, then the point lies on the same side of
        the plane as the normal vector points towards. IF the distrance is
        negative, the point lies on the opposite side of the plane from the
        direction of the normal vector. If zero distance, the point lies
        directly on the plane.

        Args:
            points: A list of NumPy arrays, where each array represents a 3D
            point.

        Returns:
            A NumPy array containing the perpendicular distances from the plane
            to each point.

        Example-1
        ---------
        from upxo.geoEntities.plane import Plane
        plane = Plane(point=(1, 1, 0), normal=(1, 1, 2))
        points = [np.array([0, 2, 1]), np.array([3, 0, -1]),
                  np.array([2, 2, 2])]
        plane.calc_perp_distances(points)
        """
        '''Reshape points to ensure it's a 2D array (n x 3) where n is the
        number of points.'''
        points = np.reshape(points, (-1, 3))
        # Vector from points to the plane's point
        vector_to_point = points - self.point
        # Distances using dot product and normalization
        distances = np.dot(vector_to_point,
                           self.normal) / np.linalg.norm(self.normal)
        if not signed:
            distances = abs(distances)
        return distances

    def find_close_points(self, point_coords, cod=0.25):
        """
        Example-1
        ---------
        from upxo.geoEntities.plane import Plane
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d

        plane = Plane(point=(0.5, 0.5, 0.5), normal=(1, 1, 1))
        xspec, yspec, zspec = [0, 1, 0.2], [0, 1, 0.2], [0, 1, 0.2]
        mulpoint3d = mp3d.from_xyz_grid(xspec=xspec, yspec=yspec, zspec=zspec,
                                        dxyz=[0.0, 0.0, 0.0],
                                        translate_ref=[0.5, 0.5, 0.5],
                                        rot=[0.0, 0.0, 0.0],
                                        rot_ref=[0.5, 0.5, 0.5],
                                        degree=True)
        D = plane.calc_perp_distances(mulpoint3d.coords, signed=False)
        cod = 0.1
        coords_within_cod = mulpoint3d.coords[np.argwhere(D <= cod)].squeeze()
        mulpoint3d.plot(coords_within_cod, primary_ms=25, primary_alpha=0.0,
                        secondary_alpha=0.25, xbound=xspec[:2],
                        ybound=yspec[:2], zbound=zspec[:2])
        """
        # Validations
        # --------------------------------
        D = self.calc_perp_distances(point_coords,
                                                   signed=False)

        # --------------------------------

    def create_parallel_stack(self, spacing, num_planes):
        """
        Creates a stack of planes parallel to this one.

        Parameters
        ----------
        spacing : float
            The perpendicular distance between each parallel plane.
        num_planes : int
            The total number of planes to generate in the stack.

        Returns
        -------
        list
            A list of new Plane objects.
        """
        # Use the unit normal to ensure correct spacing
        unit_n = self.unit_normal
        new_planes = []

        for i in range(num_planes):
            # Calculate the new point by moving along the normal vector
            new_point = self.point + (i * spacing * unit_n)

            # Create a new plane with the new point but the SAME normal vector
            # Assuming the class is named 'Plane' for the constructor call
            new_planes.append(Plane(point=new_point, normal=self.normal))

        return new_planes

    def project_point(self, point):
        """
        Projects a point onto the plane.

        Examples
        --------
        from upxo.geoEntities.plane import Plane
        Plane(point=(1, 1, 1), normal=(1, 1, 1)).project_point((0,0,0))
        Plane(point=(1, 1, 1), normal=(-1, -1, -1)).project_point((0,0,0))
        """
        # We calculate a vector pointing from a known point on the plane (
        # self.point) to the point we are trying to project (point). This
        # happens inside distance_to_point. This distance tells us how far we
        # need to move along the normal direction.
        distance = self.distance_to_point(point)
        ''' We scale the plane's normal vector (self.normal) by the calculated
        distance. This gives us the vector that represents the exact movement
        needed to go from the original point to its projection on the plane.
        '''
        projection_vector = distance * self.normal
        '''
        Subtract the projection_vector from the original point. This
        subtraction takes us from the original point, moves us directly towards
        the plane, and places us at the point of projection.
        '''
        return point - projection_vector

    def generate_random_points(self, num_points=3):
        """
        Generate random points lying on the plane.

        Explanation
        -----------
        Finding Tangent Vectors:
            We find two vectors lying within the plane that are orthogonal
            (perpendicular) to the normal vector. This is done
            using the cross product and some conditional logic to ensure we
            non-zero vectors.
        Generating Random Points:
            We use random scalars (s and t) between 0 and 1.
            A random point on the plane is formed by taking a linear
            combination of the point on the plane (self.point), the tangent
            vector, and the vector perpendicular to the tangent.

        Examples
        --------
        from upxo.geoEntities.plane import Plane
        plane = Plane(point=(0, 0, 0), normal=(0, 0, 1))
        plane.generate_random_points(3)
        """
        # Find two orthogonal vectors in the plane (using the normal vector)
        if np.abs(self.normal[0]) > np.abs(self.normal[1]):
            tangent = np.cross(self.normal, [0, 1, 0])
        else:
            tangent = np.cross(self.normal, [1, 0, 0])
        tangent = tangent / np.linalg.norm(tangent)  # Normalize

        perp_tangent = np.cross(self.normal, tangent)
        # Generate random points
        s, t = np.random.random((num_points, 2)).T
        rpoints = [tuple(self.point + _s_*tangent + _t_*perp_tangent)
                   for _s_, _t_ in zip(s, t)]
        return rpoints

    def flip_normal(self):
        """
        Flips the normal of ht plane.

        Important Considerations
        ------------------------
        "Flipping" is Visual:
            Flipping the normal changes the side of the plane that's considered
            the "front."
        Calculations:
            Be mindful of how flipping the normal might affect any calculations
            or geometric operations you perform with the plane.
        """
        pass

    def is_parallel(self, other_plane):
        """
        Checks if this plane is parallel to another plane.

        Explanation
        -----------
        Cross Product Logic: Two planes are parallel if and only if their
        normal vectors are parallel. The cross product of two parallel
        vectors is the zero vector.

        np.all and np.cross: The np.cross function calculates the cross product
        of our normal vectors. The np.all function checks if all elements of
        the resulting cross product vector are zero.

        Examples
        --------
        from upxo.geoEntities.plane import Plane

        plane1 = Plane(point=(1, 1, 1), normal=(+2, -1, +3))
        plane2 = Plane(point=(0, 0, 0), normal=(+2, -1, +3))
        plane3 = Plane(point=(1, 1, 1), normal=(-2, +1, -3))
        plane4 = Plane(point=(1, 1, 1), normal=(-2, -1, -3))

        plane1.is_parallel(plane1)
        plane1.is_parallel(plane2)
        plane1.is_parallel(plane3)
        plane1.is_parallel(plane4)
        """
        return np.all(np.cross(self.normal, other_plane.normal) == 0)

    def find_intersection_vector(self, plane):
        """Finds the dir. vector of the line of intersection b/w two planes.

        Args:
            self: The self object.
            plane: Another Plane object.

        Returns:
            A NumPy array representing the direction vector of the intersection
            line. None if the planes are parallel.

        Explanation
        -----------
        Check for Parallelism:
            Two planes are parallel if their normal vectors are parallel. In
            this case, there's no intersection.
        Intersection Is Perpendicular to Normals:
            The line of intersection of two planes is perpendicular to both of
            their normal vectors.
        Cross Product:
            The cross product of the normal vectors gives us a vector with the
            direction of the line of intersection.

        Important Notes
        ---------------
        This function only gives the direction vector. To fully describe the
        line of intersection, you also need a point that lies on the line.
        Here's one way to find such a point:
        https://math.stackexchange.com/questions/475953/how-to-calculate-the-intersection-of-two-planes

        Numerical Issues
        ----------------
        In practice, due to floating-point calculations, you might want to
        check if the result of the cross product is very close to zero instead
        of exactly zero to handle cases where the planes are almost parallel.

        Example
        -------
        from upxo.geoEntities.plane import Plane
        plane1 = Plane(point=(1, 1, 0), normal=(1, 1, 2))
        plane2 = Plane(point=(0, 2, 1), normal=(-2, -1, 1))

        intersection_vector = plane1.find_intersection_vector(plane2)
        print(intersection_vector)
        """

        # Check for parallelism (no intersection if parallel)
        if np.all(np.cross(self.normal, plane.normal) == 0):
            return None

        # The direction of the intersection line is perpendicular to both plane
        # normals:
        direction_vector = np.cross(self.normal, plane.normal)

        return direction_vector

    def create_translated_planes(self,
                                 translation_vector,
                                 num_planes,
                                 dlk=np.array([1.0, -1.0, 1.0]),
                                 dnw=np.array([0.5, 0.5, 0.5]),
                                 dno=np.array([0.5, 0.5, 0.5]),
                                 bidrectional=False,
                                 ):
        """Creates an array of planes translated from the original plane by a
        given vector.

        Args:
            translation_vector: A NumPy array representing the translation
            vector.
            num_planes: The number of translated planes to generate.

        Returns:
            A NumPy array of Plane objects, each representing a translated
            plane.

        Explanation

        Iterative Translation: The method loops num_planes times.
        New Point Calculation: In each iteration, it calculates the new point
            for the translated plane by adding the translation_vector to the
            original plane's point (self.point).
        Creating New Plane: A new Plane object is created using the translated
            point (new_point) and the original plane's normal vector
            (self.normal) since the plane's orientation remains the same.
        Storing and Returning: The newly created Plane object is appended to a
            list, and finally, the list is converted to and returned as a NumPy
            array.

        Example-1
        ---------
        from upxo.geoEntities.plane import Plane
        plane = Plane(point=(0, 0, 0), normal=(1, 1, 1))
        translation_vector = np.array([2, 3, -1])
        num_planes = 4
        translated_planes_array = plane.create_translated_planes(translation_vector, num_planes)
        print(translated_planes_array)  # Array containing 4 Plane objects

        Example-2
        ---------
        from upxo.geoEntities.plane import Plane
        from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
        xspec, yspec, zspec = [0, 1, 0.05], [0, 1, 0.05], [0, 1, 0.05]
        mpnt3d = mp3d.from_xyz_grid(xspec=xspec, yspec=yspec, zspec=zspec,
                                    dxyz=[0.0, 0.0, 0.0],
                                    translate_ref=[0.5, 0.5, 0.5],
                                    rot=[0.0, 0.0, 0.0],
                                    rot_ref=[0.5, 0.5, 0.5],
                                    degree=True)
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(1, 1, 1))
        num_planes, translation_vector = 8, np.array([0.2, 0.2, 0.2])

        planes = plane.create_translated_planes(translation_vector, num_planes)
        D = [plane.calc_perp_distances(mpnt3d.coords, signed=False)
             for plane in planes]

        cod = 0.125
        coords = [mpnt3d.coords[np.argwhere(d <= cod)].squeeze() for d in D]
        # --------------------------------------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mpnt3d.coords[:, 0], mpnt3d.coords[:, 1],
                   mpnt3d.coords[:, 2], c='b', marker='o', alpha=0.01, s=100,
                   edgecolors='black')

        for coord in coords:
            if coord is not None:
                ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2],
                           c=np.random.random(3), marker='o', alpha=0.8, s=50,
                           edgecolors='black')

        xbound, ybound, zbound = xspec[:2], yspec[:2], zspec[:2]
        vertices = np.array([[xbound[0], ybound[0], zbound[0]],  # 0
                             [xbound[1], ybound[0], zbound[0]],  # 1
                             [xbound[1], ybound[1], zbound[0]],  # 2
                             [xbound[0], ybound[1], zbound[0]],  # 3
                             [xbound[0], ybound[0], zbound[1]],  # 4
                             [xbound[1], ybound[0], zbound[1]],  # 5
                             [xbound[1], ybound[1], zbound[1]],  # 6
                             [xbound[0], ybound[1], zbound[1]]])  # 7
        # Define the edges of the cuboid
        edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        for edge in edges:
            ax.plot(*zip(*vertices[edge]), color='k', linewidth=2.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        # --------------------------------------

        mpnt3d.plot(coords_within_cod, primary_ms=25, primary_alpha=0.0,
                        secondary_alpha=0.25, xbound=xspec[:2],
                        ybound=yspec[:2], zbound=zspec[:2])
        # ===================================================================
        from upxo.geoEntities.plane import Plane
        plane1 = Plane(point=(0, 0, 0), normal=(0, 1, 2))
        num_planes, translation_vector = 10, np.array([1, -1, 1])
        dlk = np.array([0.1, 0.0, 0.0])
        dnw = np.array([0.0, 0.0, 0.0])  # Width values
        dno = np.array([0.5, 0.5, 0.5])  # Offset values

        planes = plane1.create_translated_planes(translation_vector, num_planes,
                                                 dlk=np.array([0.0, 0.0, 0.0]),
                                                 dnw=np.array([0.1, 0.0001, 0.0]),
                                                 dno=np.array([0.5, 0.5, 0.5])
                                                 )

        print(planes)
        """
        if bidrectional:
            '''If planes are to be bi-directionally generated.'''
            tps1 = self.create_translated_planes(translation_vector,
                                                 num_planes,
                                                 dlk=dlk,
                                                 dnw=dnw,
                                                 dno=dno)
            tps2 = self.create_translated_planes(-translation_vector,
                                                 num_planes,
                                                 dlk=dlk,
                                                 dnw=dnw,
                                                 dno=dno)
            tr_planes = tuple(tps1) + tuple(tps2)
        else:
            '''If planes are to be on one side only.'''
            npl =  num_planes-1
            dl = dlk * np.random.random((npl, 3))
            TV = np.arange(1, npl+1)[:, np.newaxis]*np.tile(translation_vector, (npl, 1)) + dl
            TV = TV + self.point
            NR = np.tile(self.normal, (npl, 1)) + dnw*(dno-np.random.random((npl, 3)))
            tr_planes = [self]
            for tv, nr in zip(TV, NR):
                tr_planes.append(Plane(point=tv, normal=nr))

        return np.array(tr_planes)

    def offset_point_on_plane(self, point, offset_vector):
        """Offsets a point on the plane by a given pos. vec. within the plane.

        Args:
            point: A NumPy array representing the point on the plane.
            offset_vector: A NumPy array representing the offset vector within
            the plane.

        Returns:
            A NumPy array representing the new offset point.

        Explanation

        Projecting the Offset Vector: We use the project_point method to
        project the offset_vector onto the plane. This is necessary because the
        given offset_vector might have a component perpendicular to the plane,
        which we need to remove.
        Calculating In-Plane Offset: Subtracting the projection from the
        original offset_vector gives us an in_plane_offset that lies entirely
        within the plane.
        Offsetting the Point: We add this in_plane_offset to the given point
        to get the final offset point.

        Example
        -------
        from upxo.geoEntities.plane import Plane

        # Invalid try:
        my_plane = Plane(point=(2, -1, 0), normal=(0, 1, 1))
        point_on_plane = np.array([2, 3, 1])
        offset_vector = np.array([1, -1, 2])  # May not lie entirely within the plane
        offset_point = my_plane.offset_point_on_plane(point_on_plane,
                                                      offset_vector)
        print(offset_point)

        # Valid try:
        my_plane = Plane(point=(2, -1, 0), normal=(0, 1, 1))
        point_on_plane = np.array([2, -1, 0])
        offset_vector = np.array([1, -1, 2])  # May not lie entirely within the plane
        offset_point = my_plane.offset_point_on_plane(point_on_plane,
                                                      offset_vector)
        print(offset_point)
        """
        # Check if the point lies on the plane
        distance_to_plane = self.distance_to_point(point)
        # Tolerance check for floating point values
        if not np.isclose(distance_to_plane, 0):
            raise ValueError("The input point does not lie on the plane.")
        projection_onto_plane = self.project_point(offset_vector)
        in_plane_offset = offset_vector - projection_onto_plane

        # Offset the point by the in-plane offset vector
        return point + in_plane_offset


    def calculate_inclined_circle(self, center, radius, angle_A, angle_B, angle_C):
        """Calculates points on a circle centered at a point on the plane,
        inclined at angles A, B, C to the XY, YZ, and XZ planes respectively.

        Args:
            center: A NumPy array representing the center of the circle.
            radius: The radius of the circle.
            angle_A: Angle of inclination to the XY plane (radians).
            angle_B: Angle of inclination to the YZ plane (radians).
            angle_C: Angle of inclination to the XZ plane (radians).

        Returns:
            A NumPy array of points representing the circle.

        Examples
        --------
        from upxo.geoEntities.plane import Plane

        my_plane = Plane(point=(1, 0, 2), normal=(0, 1, -1))
        center = np.array([1, 2, 2])
        radius = 1.5
        angle_A = np.radians(30)
        angle_B = np.radians(-20)
        angle_C = np.radians(60)

        circle_points = my_plane.calculate_inclined_circle(center, radius,
                                                           angle_A, angle_B,
                                                           angle_C)


        Visulizations
        -------------
        my_plane.visualize(circle_points)
        """

        # Construct rotation matrices based on your preferred angle
        # representation
        Rx = self._rotation_matrix_x(angle_A)
        Ry = self._rotation_matrix_y(angle_B)
        Rz = self._rotation_matrix_z(angle_C)

        # Find a vector orthogonal to the plane's normal (lies within the
        # plane)
        circle_x_axis = self._find_orthogonal_vector(self.normal)

        # Use this vector to create a standard circle in the plane
        circle_points = self._generate_circle_points(center,
                                                     radius,
                                                     circle_x_axis)

        # Rotate the circle using combined rotation matrix
        rotated_circle = np.dot(Rx @ Ry @ Rz, circle_points.T).T

        return rotated_circle

    def _generate_circle_points(self, center, radius, axis):
        num_points = 20  # Number of points on the circle
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(theta)  # Circle initially lies in XY plane
        circle_points = np.stack([x, y, z], axis=-1) + center
        return circle_points

    def _rotation_matrix_x(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    def _rotation_matrix_y(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    def _rotation_matrix_z(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    def _find_orthogonal_vector(self, vector):
        """
        A reliable way to find a vector orthogonal to the normal, ensuring a
        non-zero result.
        """
        if np.abs(vector[0]) > np.abs(vector[1]):
            return np.cross(vector, [0, 1, 0])
        else:
            return np.cross(vector, [1, 0, 0])

    def visualize(self, circle_points=None):
        """Visualizes the plane and an optional circle on the plane."""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the plane (using a rectangular region)
        x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
        z = (-self.normal[0] * x - self.normal[1] * y - self.point[2]) / self.normal[2]
        ax.plot_surface(x, y, z, alpha=0.5)

        # Plot the circle (if provided)
        if circle_points is not None:
            ax.plot(circle_points[:, 0],
                    circle_points[:, 1],
                    circle_points[:, 2],
                    color='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def visualize1(self, circle_points=None, other_planes=None):
        """
        Visualizes the plane, optional circle, and other planes.

        from upxo.geoEntities.plane import Plane
        plane1 = Plane(point=(1, 0, 2), normal=(0, 1, -1))
        plane2 = Plane(point=(-1, 1, 0), normal=(1, 1, 1))

        circle_points = plane1.calculate_inclined_circle(center=[1, 2, 2],
                                                         radius=1.5,
                                                         angle_A=np.radians(30),
                                                         angle_B=np.radians(-20),
                                                         angle_C=np.radians(60))

        # Visualize plane1 with the circle and plane2
        plane1.visualize1(circle_points, other_planes=[plane2])
        """

        fig = plt.figure(figsize=(8, 6))  # Adjust figure size if needed
        ax = fig.add_subplot(111, projection='3d')

        # ... (Plane and circle plotting from previous code) ...

        # Plot other planes
        if other_planes:
            for plane in other_planes:
                x, y = np.meshgrid(np.linspace(-5, 5, 10),
                                   np.linspace(-5, 5, 10))
                z = (-plane.normal[0] * x - plane.normal[1] * y - plane.point[2]) / plane.normal[2]
                ax.plot_surface(x, y, z, alpha=0.3)  # Slightly more transparent

        # Plot normal vector
        midpoint = self.point + 0.5 * self.normal  # Point along the normal
        ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.normal[0], self.normal[1], self.normal[2],
                  color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def angle_between_planes(self, other_plane):
        """Calculates the angle between this plane and another plane.

        Args
        -----
        other_plane: The other Plane object.

        Returns
        -------
        The angle between the two planes in radians.


        Explanation
        -----------
        Dot Product of Normals:
            The angle between two planes is equal to the angle between
            their normal vectors. The dot product gives us the cosine of
            this angle.
        Arccosine: We use np.arccos to find the angle whose cosine is the
            calculated value.
        np.clip: This ensures the value passed to np.arccos stays within
            [-1, 1] due to potential floating-point inaccuracies.

        Example
        -------
        from upxo.geoEntities.plane import Plane
        plane1 = Plane(point=(0, 0, 0), normal=(1, 1, 0))
        plane2 = Plane(point=(0, 0, 0), normal=(1, 0, 0))

        intersection_angle = plane1.angle_between_planes(plane2)
        print(f"Angle of intersection (in radians): {intersection_angle}")
        print(f"Angle of intersection (in degrees): {np.degrees(intersection_angle)}")
        """

        angle_cos = np.dot(self.normal, other_plane.normal) / \
                    (np.linalg.norm(self.normal) * np.linalg.norm(other_plane.normal))
        angle = np.arccos(np.clip(angle_cos, -1.0, 1.0))  # Clip to handle numerical edge cases
        return angle
