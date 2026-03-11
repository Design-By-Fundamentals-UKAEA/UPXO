import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries

from shapely.ops import voronoi_diagram
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import LinearRing



def bounding_rectangle_pxtal(pxtal):
    # Calculate the grid on the poly-xtal
    rec = pxtal.envelope
    recx, recy = rec.boundary.xy[0][:-1], rec.boundary.xy[1][:-1]
    xlimits = [min(recx), max(recx)]
    ylimits = [min(recy), max(recy)]
    return recx, recy, xlimits, ylimits
