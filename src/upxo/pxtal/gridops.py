import numpy as np
import shapely
def make_grid_pxtal(distribution_type, **kwargs):
    if distribution_type == 'rectgrid':
        method_bounds = kwargs['method_bounds']
        usefactor     = kwargs['usefactor']
        xlimits       = kwargs['xlimits']
        ylimits       = kwargs['ylimits']
        if method_bounds == 'frombounds':
            xmin, xmax = xlimits
            ymin, ymax = ylimits
            if usefactor == True:
                grid_mul_factor = kwargs['grid_mul_factor']
                x = np.linspace(xmin, xmax, int((xmax - xmin)*grid_mul_factor))
                y = np.linspace(ymin, ymax, int((ymax - ymin)*grid_mul_factor))
            else:
                x = np.linspace(xmin, xmax, int(xmax - xmin))
                y = np.linspace(ymin, ymax, int(ymax - ymin))
            grid_spacing_x = min([x[1] - x[0], y[1] - y[0]])
            grid_spacing_y = grid_spacing_x
            x, y = np.meshgrid(x, y)
        return x, y, grid_spacing_x
    elif distribution_type == 'random':
        # arguments::: method_bounds: 'bounded_by_pxtal', 'user_data'
        if method_bounds == 'bounded_by_shmulpol':
            # arguments::: shapely object of type "MultiPolygon", like pxtal
            # arg method_bounds MUST be followed by keyword args mulpol having the 
            # shapely multi-polygon object
            import shapely
            xmin, xmax = min(mulpol.envelope.boundary.xy[0]), max(mulpol.envelope.boundary.xy[0])
            ymin, ymax = min(mulpol.envelope.boundary.xy[1]), max(mulpol.envelope.boundary.xy[1])
        elif method_bounds == 'user_data':
            # arguments: xlimits = [xmin, xmax], ylimits = [ymin, ymax]
            # arg method_bounds MUST be followed by keyword args xlimits and ylimits
            xmin, xmax = xlimits
            ylim, ymax = ylimits
            #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        if usefactor == True:
            pass
        elif usefactor == False:
            if distribution_subtype == 'uniform':
                # Access: make_grid_pxtal('random', **kwargs)
                # RANDOM RANDOM
                x = np.random.random((domain_size_count, domain_size_count))
                y = np.random.random((domain_size_count, domain_size_count))
            elif distribution_subtype == 'power':
                # RANDOM POWER
                exponent = 3
                x = np.reshape(np.random.power(exponent, size = domain_size_count**2), (domain_size_count, domain_size_count))
                y = np.reshape(np.random.power(exponent, size = domain_size_count**2), (domain_size_count, domain_size_count))
            elif distribution_subtype == 'exponential':
                # RANDOM EXPONENTIAL
                exponent = 1
                x = np.reshape(np.random.exponential(exponent, size = domain_size_count**2), (domain_size_count, domain_size_count))
                y = np.reshape(np.random.exponential(exponent, size = domain_size_count**2), (domain_size_count, domain_size_count))
        x = x/x.max()
        y = y/y.max()
        x = x*(xmax-xmin) + xmin
        y = y*(ymax-ymin) + ymin
