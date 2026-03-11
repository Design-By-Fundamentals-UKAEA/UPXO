import pandas as pd

class dict_templates():
    # -----------------------------------------
    '''
    gc: Grain Centroids
    gcpos: Grain Centroids for position segregated grains
    rp: Representative Points
    jp2: Double Junction Points
    jp3: Triple Junction Points
    jp4: Qadruple Point Junctions
    '''

    def __init__(self):
        pass

    def mulpnt_gs2d(self):
        return {'gc': None,
                'gcpos': {'in': None, 'boundary': None,
                          'corner': None,
                          'left': None, 'bottom': None,
                          'right': None, 'top': None,
                          'pure_left': None, 'pure_bottom': None,
                          'pure_right': None, 'pure_top': None, },
                'rp': None, 'jp2': None, 'jp3': None, }

    def mulpnt_gs3d(self):
        return {'gc': None,
                'gcpos': {'in': None, 'boundary': None,
                          'corner': None,

                          'front_left': None, 'front_bottom': None,
                          'front_right': None, 'front_top': None,
                          'front_pure_left': None, 'front_pure_bottom': None,
                          'front_pure_right': None, 'front_pure_top': None,

                          'back_left': None, 'back_bottom': None,
                          'back_right': None, 'back_top': None,
                          'back_pure_left': None, 'back_pure_bottom': None,
                          'back_pure_right': None, 'back_pure_top': None,
                          },
                'rp': None, 'jp2': None, 'jp3': None, }

    def vizstyles_mcgs(self):
        return {'hist_colors_fill': "#4CC9F0",
                'hist_colors_edge': 'black',
                'hist_colors_fill_alpha': 0.5,
                'kde_color': 'crimson',
                'kde_thickness': 1,
                'bins': 25,
                'hist_area_xbounds': [0, 100],
                'hist_area_ybounds_density': [0, 0.2],
                'hist_area_ybounds_freq': [0, 50],
                'hist_area_ybounds_counts': [0, 50],
                'hist_peri_xbounds': [0, 100],
                'hist_peri_ybounds_density': [0, 0.2],
                'hist_peri_ybounds_freq': [0, 50],
                'hist_peri_ybounds_counts': [0, 50],
                }


class pd_templates():

    def __init__(self):
        pass

    def make_prop2d_df(self, bbox=True, bbox_ex=True, npixels=True,
                       npixels_gb=True, area=True, eq_diameter=True,
                       perimeter=True, perimeter_crofton=True,
                       compactness=True, gb_length_px=True, aspect_ratio=True,
                       solidity=True, morph_ori=True, circularity=True,
                       eccentricity=True, feret_diameter=True,
                       major_axis_length=True, minor_axis_length=True,
                       euler_number=True, moments_hu=False, append=False, ):
        """
        Construct empty pandas dataframe of properties

        Parameters
        ----------
        bbox : bool
            Bounding rectangle
        bbox_ex : bool
            Extended bounding rectangle
        npixels : bool
            Number of pixels in the grain.
        npixels_gb : bool
            Number of pixels on the grain boundary.
        area : bool
            Area of the grain: number of pixels in the
            grain * pixel area.
        eq_diameter : bool
            Equivalent circle diameter.
        perimeter : bool
            Perimeter of the grain boundary.
            DEF: `Total length of all lines  passing through the centres
            of grain boundary pixels taken in order`
        perimeter_crofton : bool
            Crofton type perimeter of the grain boundary.
        compactness : bool
            Compactness of the grain.
            DEF: `(pixel area) / (Area of circle with perimeter equal to
                                  grain perimeter)`.
        gb_length_px : bool
            Deprecated use. Not recommended.
        aspect_ratio : bool
            Aspect ratio of the grain.
            Calculated as the ratio of major axis length to minor axis
            length of a ellipse fit to the grain.
        solidity : bool
            Solidity of the grain.
            DEF: `(npixels) / (number of pixels falling inside
                               the convex hull computed on the grain)`
        morph_ori : bool
            Morphological orientation of the grain (in deg).
            DEF: `-pi/2 to +pi/2. Counter-clockwise from x-axis`
        circularity : bool
            Indicate how close the grain shape is to being circular.
        eccentricity : bool
            Eccentricity of the grain.
            DEF: `(distance between focal points) / (major axis length)`
        feret_diameter : bool
            Average Feret diameter of the grain. Also called Caliper
            diameter
            NOTE: The name Caliper diameter is not explicitly used inside
            UPXO.
            DEF: Feret, or Caliper diameter is essentially the perpendicular
            distance between the two parallel lines running parallel to the
            grain boundary. Consequently, it is bounded by a minimum and a
            maximum value. `<Df> = Df_max / Df_min`, where `Df_max` and
            `Df_min` are the maximum and minimum Feret diamater.
        major_axis_length : bool
            Major axis length of the ellipse fit to the grain.
        minor_axis_length : bool
            Minor axis length of the ellipse fit to the grain.
        euler_number : bool
            Euler number of the grain.
            Will be 1 for grains without island grains.
        append : bool
            DESCRIPTION


        Returns
        -------
        None.

        """
        # Make new Pandas dataframe
        prop_flag = {'npixels': npixels,
                     'npixels_gb': npixels_gb,
                     'area': area,
                     'eq_diameter': eq_diameter,
                     'perimeter': perimeter,
                     'perimeter_crofton': perimeter_crofton,
                     'compactness': compactness,
                     'gb_length_px': gb_length_px,
                     'aspect_ratio': aspect_ratio,
                     'solidity': solidity,
                     'morph_ori': morph_ori,
                     'circularity': circularity,
                     'eccentricity': eccentricity,
                     'feret_diameter': feret_diameter,
                     'major_axis_length': major_axis_length,
                     'minor_axis_length': minor_axis_length,
                     'euler_number': euler_number,
                     'moments_hu': moments_hu,
                     }
        _columns = [key for key in prop_flag.keys()
                    if prop_flag[key]]
        prop = pd.DataFrame(columns=_columns)
        prop_stat = pd.DataFrame(columns=_columns)

        return prop_flag, prop, prop_stat

    def make_prop3d_df(self, npixels=True):
        # Make new Pandas dataframe
        prop_flag = {'npixels': npixels}
        _columns = [key for key in prop_flag.keys()
                    if prop_flag[key]]
        prop = pd.DataFrame(columns=_columns)
        prop_stat = pd.DataFrame(columns=_columns)

        return prop_flag, prop, prop_stat
