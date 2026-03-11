def att(obj):
    import inspect
    methods = []
    slotvariables = []
    dunders = []
    for name, member in inspect.getmembers(obj):
        if inspect.ismethod(member):
            methods.append(name)
        if name == '__slots__':
            for memb in member:
                slotvariables.append(memb)
        if '__' in name:
            dunders.append(name)
    if '__init__' in methods:
        del methods[methods.index('__init__')]
    to_return = []
    for _ in [methods, slotvariables, dunders]:
        for __ in _:
            to_return.append(__)
        to_return.append('---------------------')
    return to_return


def slots(obj):
    import inspect
    slotvariables = []
    for name, member in inspect.getmembers(obj):
        if name == '__slots__':
            for memb in member:
                slotvariables.append(memb)
    to_return = []
    to_return.append('---------------------')
    for _ in slotvariables:
        to_return.append(_)
    to_return.append('---------------------')
    return to_return


def methods(obj):
    import inspect
    methods = []
    for name, member in inspect.getmembers(obj):
        if inspect.ismethod(member):
            methods.append(name)
    if '__init__' in methods:
        del methods[methods.index('__init__')]
    to_return = []
    to_return.append('---------------------')
    for _ in methods:
        to_return.append(_)
    to_return.append('---------------------')
    return to_return


def dunders(obj):
    import inspect
    dunders = []
    for name, member in inspect.getmembers(obj):
        if '__' in name:
            dunders.append(name)
    to_return = []
    to_return.append('---------------------')
    for _ in dunders:
        to_return.append(_)
    to_return.append('---------------------')
    return to_return


def PROFILE_this_method(method, n, method_inputs, sort='cumulative'):
    """
    Profile the method

    Parameters
    ----------
    method : method object
        Method object to be profiled
    n : int
        Number of runs.
    method_inputs : tuple
        Inputs needed for the method. Must be entered as a tuple
    sort : str, optional
        Sort option field value for print_stats of profiler.
        The default is 'cumulative'.

    Returns
    -------
    None.

    """
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    # Instantiate the class multiple times
    for i in range(n):
        _ = method(*method_inputs)

    profiler.disable()
    profiler.print_stats(sort=sort)


def PROFILE_up2d_INST(n, lean):
    """
    PROFILE_up2d_INST(10000, 'ignore')
    """
    import cProfile
    from point2d import point2d
    profiler = cProfile.Profile()
    profiler.enable()

    # Instantiate the class multiple times
    for i in range(n):
        _ = point2d(x=0, y=1, lean=lean)

    profiler.disable()
    profiler.print_stats(sort='cumulative')
