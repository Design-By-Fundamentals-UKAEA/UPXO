class Surface():
    __slots__ = ('x', 'y', 'z')

7    def __init__(self):
        pass

    def __repr__(self):
        return "UPXO surface."

    @classmethod
    def from_points(self):
        pass

    @classmethod
    def from_vertices(self):
        pass

    def compute_normals(self):
        pass

    def shortest_path(self, point):
        pass

    def triangulate(self, point):
        pass

    def distribute_points(self, n, min_distance=-1):
        pass

    def pyvista_mesh(self):
        pass

    def smooth_laplace(self, niterations):
        pass

    def smooth_taubin(self, niterations):
        pass
