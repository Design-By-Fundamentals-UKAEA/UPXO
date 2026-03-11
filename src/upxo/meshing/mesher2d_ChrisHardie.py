"""
This code is to port UPXO data into a format amenable for Chris Hardie's
module to geometrify the non-geometric MCGS2D UPXO grain structure and to
export geometry for conformal meshing in ABAQUS.

Created on Fri Jul 12 10:54:32 2024

@authors:
    Chris Hardie (primary author), UKAEA
    Sunil Anandatheertha, UKAEA
"""
import pickle
import numpy as np

class Region:
    def __init__(self, xc, yc, phi1, theta, phi2,
                 scale, phase, grains, xmax, ymax, BC):
        """
        Import
        ------
        from upxo.meshing.mesher2d_ChrisHardie import Region

        Example
        -------
        # All grain structure genereation codes here frst.
        # ..... Remaining codes here to convert the data-structure
        # --------------------------------
        import cv2
        import numpy as np
        import gmsh
        import pyvista as pv
        import upxo._sup.data_ops as DO
        from copy import deepcopy
        from shapely import affinity

        from upxo.geoEntities.mulsline2d import MSline2d
        import matplotlib.pyplot as plt
        from scipy.ndimage import generic_filter
        from meshpy.triangle import MeshInfo, build
        from upxo.ggrowth.mcgs import mcgs
        from upxo.geoEntities.mulsline2d import MSline2d
        from shapely.geometry import Point
        from shapely.geometry import MultiPolygon, Point
        from scipy.spatial import cKDTree
        from shapely.geometry import Point
        from upxo.geoEntities.mulsline2d import ring2d
        from upxo._sup.data_ops import find_common_coordinates
        from shapely.geometry import LineString, MultiLineString
        from upxo.geoEntities.point2d import Point2d
        from upxo.geoEntities.mulpoint2d import MPoint2d
        from shapely.geometry import Point as ShPoint2d
        from upxo._sup.data_ops import remove_2d_child_array_from_2d_parent_array
        # ---------------------------
        AUTO_PLOT_EVERYTHING = False
        # ---------------------------
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()

        tslice = 9
        pxt.char_morph_2d(tslice)
        gstslice = pxt.gs[tslice]
        gstslice.find_neigh()
        folder, fileName = r'D:\export_folder', 'sunil'
        gstslice.export_ctf(folder, fileName, factor=1, method='nearest')
        fname = r'D:\export_folder\sunil'
        gstslice.set_pxtal(instance_no=1, path_filename_noext=fname)
        gstslice.pxtal[1].find_gbseg1()
        pxtalmap = gstslice.pxtal[1]

        # --------------------------------
        # We now create data for creatring the current class, Region.
        xc = np.arange(0, pxtalmap.xDim, pxtalmap.stepSize)
        yc = np.arange(0, pxtalmap.yDim, pxtalmap.stepSize)
        phi1, theta, phi2 = pxtalmap.eulerAngleArray
        scale = pxtalmap.stepSize
        phase = pxtalmap.phaseArray
        grains = pxtalmap.grains
        xmax, ymax = pxtalmap.xDim, pxtalmap.yDim
        BC = pxtalmap.bandContrastArray

        x_map = np.zeros_like(grains)
        y_map = np.zeros_like(grains)
        BC = np.zeros_like(grains)
        # --------------------------------
        wb = Workbook()
        for i, (name, value) in enumerate(data.items(), start=0):
            if i == 0:
                ws = wb.active
                ws.title = name
            else:
                ws = wb.create_sheet(title=name)
            # --------------------------
            for r in range(value.shape[0]):
                for c in range(value.shape[1]):
                    ws.cell(row=r+1, column=c+1, value=data[r, c])

        np.savetxt("utoe2s_xc.csv", xc)
        np.savetxt("utoe2s_yc.csv", yc)
        np.savetxt("utoe2s_phi1.csv", phi1)
        np.savetxt("utoe2s_theta.csv", theta)
        np.savetxt("utoe2s_phi2.csv", phi2)
        np.savetxt("utoe2s_scale.csv", np.array([scale]))
        np.savetxt("utoe2s_grains.csv", grains)
        np.savetxt("utoe2s_xmax.csv", np.array([xmax]))
        np.savetxt("utoe2s_ymax.csv", np.array([ymax]))
        np.savetxt("utoe2s_BC.csv", BC)

        data = {'xc': xc,
                'yc': yc,
                'phi1': phi1,
                'theta': theta,
                'phi2': phi2,
                'scale': [scale],
                'grains': grains,
                'xmax': [xmax],
                'ymax': [ymax]}

        for name, value in data.items():
            print(name)
            np.savetxt(f"utoe2s_{name}.csv", value)

        import_data = {'xc': None, 'yc': None, 'phi1': None, 'theta': None,
                       'phi2': None, 'scale': None, 'grains': None,
                       'xmax': None, 'ymax': None}
        from numpy import genfromtxt
        for varname in variables.keys():
            filename = f"utoe2s_{varname}.csv"
            import_data[varname] = genfromtxt(filename, delimiter=',')

        my_data = genfromtxt('utoe2s_xc.csv', delimiter=',')
        # --------------------------------
        from upxo.meshing.mesher2d_ChrisHardie import Region
        R = Region(xc, yc, phi1, theta, phi2, scale, phase, grains,
                   xmax, ymax, BC)
        filename = r"D:\export_folder\aaa"
        R.save_pickle(filename)
        """
        self.xc, self.yc = xc, yc
        self.phi1, self.theta, self.phi2 = phi1, theta, phi2
        self.scale, self.phase, self.grains = scale, phase, grains

        self.boundaries = np.zeros((self.xc.size, self.xc.size))
        self.BC = np.zeros((self.xc.size, self.xc.size))
        self.x_map = np.zeros((self.xc.size, self.xc.size))
        self.y_map = np.zeros((self.xc.size, self.xc.size))

    def write_data(self):

        pass

    def save_pickle(self, file_name):
        with open(file_name+'.pickle', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
            # pickle.dump(self, file)


with open(filename+'.pickle', 'wb') as file:
    A = pickle.load(file)


file = open(filename+'.pickle', 'r', encoding="utf8")
data = pickle.load(file)
