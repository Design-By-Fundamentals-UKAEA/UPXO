'''
https://stackoverflow.com/questions/34968838/python-finite-boundary-voronoi-cells
https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python?noredirect=1&lq=1
https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647
'''
#//////////////////////////////////////////////////////////////////////////////
def _clip_Voronoi_Tess_BoundBox(xtals = None,
                                boundary_object = None):
    #import random
    #from shapely.ops import voronoi_diagram
    #from shapely.geometry import LineString, MultiPolygon, Polygon, Point, MultiPoint, LinearRing
    #from shapely.ops import polygonize, split, SplitOp, voronoi_diagram
    #from shapely import affinity
    #import numpy as np
    #import matplotlib.pyplot as plt

    # FUNCTION IN TESS_SCIPY
    #from shapely.ops import voronoi_diagram
    #from shapely.geometry import LineString, MultiPolygon, Polygon, Point, MultiPoint, LinearRing
    #from shapely.ops import polygonize, split, SplitOp, voronoi_diagram
    _xtals = []
    for xtal in xtals:
        # Clip this polygon with the boundary of the bounding box
        clipped_xtal = xtal.intersection(boundary_object)
        if clipped_xtal.area > 0.0:
            _xtals.append(clipped_xtal)
    return _xtals
#//////////////////////////////////////////////////////////////////////////////
def _shapely(point_method = 'mulpoints',
             _x = None,
             _y = None,
             xbound = None,
             ybound = None,
             vis_vtgs = True
             ):
    #******************
    #import random
    from shapely.ops import voronoi_diagram
    #from shapely.geometry import LineString
    from shapely.geometry import MultiPolygon
    from shapely.geometry import Polygon
    #from shapely.geometry import Point
    from shapely.geometry import MultiPoint
    #from shapely.geometry import LinearRing
    #from shapely.ops import polygonize
    #from shapely.ops import split
    #from shapely.ops import SplitOp
    #from shapely import affinity
    #import numpy as np
    #******************
    points_list = [[__x, __y] for (__x, __y) in zip(_x, _y)]
    #******************
    seeds_mp = MultiPoint(points_list)
    #******************
    pxtal = voronoi_diagram(seeds_mp)
    #******************
    vgrains = [grain for grain in pxtal.geoms]
    #******************
    bound_rect = Polygon([[xbound[0], ybound[0]],
                          [xbound[1], ybound[0]],
                          [xbound[1], ybound[1]],
                          [xbound[0], ybound[1]]]
                         )
    #******************
    # from vt import _clip_Voronoi_Tess_BoundBox
    xtal_list_xtal = _clip_Voronoi_Tess_BoundBox(vgrains, bound_rect)
    #******************
    pxtal = MultiPolygon(xtal_list_xtal)
    #******************
    #print(vis_vtgs)
    #******************
    if vis_vtgs:
        import matplotlib.pyplot as plt
        plt.figure(figsize = (3.5, 3.5), dpi = 100)
        gcount = 0
        for xtal in xtal_list_xtal:
            plt.fill(xtal.boundary.xy[0],
                     xtal.boundary.xy[1],
                     color = 'white',
                     edgecolor = 'black',
                     linewidth = 1)
            xc = xtal.centroid.x
            yc = xtal.centroid.y
            plt.text(xc, yc, str(gcount), fontsize = 10, fontweight = 'bold', color = 'red')
            gcount += 1
        #plt.plot(x_flat, y_flat, '+')
        maximum = max([xbound[1], ybound[1]])
        #print(maximum)
        #plt.xlim([xbound[0], xbound[0] + maximum])
        #plt.ylim([ybound[0], ybound[0] + maximum])
        #plt.set_aspect('equal', 'box')
    #***********************
    #print(vgrains)
    return pxtal#, vgrains
#//////////////////////////////////////////////////////////////////////////////
def _finite_vtpols(vo):
    import numpy as np
    from shapely.geometry import Polygon
    from scipy.spatial import Voronoi
    if vo.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vo.vertices.tolist()
    center = vo.points.mean(axis=0)
    radius = None
    if radius is None:
        radius = vo.points.ptp().max()*2
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vo.ridge_points, vo.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vo.point_region):
        vertices = vo.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an infinite ridge
            t = vo.points[p2] - vo.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vo.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vo.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        # finish
        new_regions.append(new_region.tolist())
        cells = new_regions
        #Ng0 = len(new_regions)
        points = np.asarray(new_vertices)
    return cells, points
#//////////////////////////////////////////////////////////////////////////////
def _make_bounding_polygon(data = None, method = 'auto_from_scipy_vto'):
    if method == 'auto_from_scipy_vto':
        min_x = data.min_bound[0] - 0.00
        max_x = data.max_bound[0] + 0.00
        min_y = data.min_bound[1] - 0.00
        max_y = data.max_bound[1] + 0.00
        # Polygonal bounding box for Voronoi tessellation
        from shapely.geometry import Polygon
        boundDomain = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
        return boundDomain
#//////////////////////////////////////////////////////////////////////////////
