import importlib

def port_doc(module_path, func_name):
    """
    Decorator to port the docstring from a function in another module.
    Parameters
    ----------
    module_path : str
        The dot-separated path to the module containing the source function.
    func_name : str
        The name of the function whose docstring is to be ported.

    Returns
    -------
    decorator : function
        A decorator that adds the source function's docstring to the target function.

    Notes
    -----
    This decorator imports the specified module and retrieves the docstring
    from the specified function. It appends this docstring to the target function's
    existing docstring. If the import fails, it leaves the target function's
    docstring unchanged.

    Import
    ------
    import upxo._sup.decorators as decorators
    """
    def decorator(target_func):
        try:
            # Import the module only when the decorator is executed
            module = importlib.import_module(module_path)
            source_func = getattr(module, func_name)
            
            # Combine the source docstring with the local one
            source_doc = source_func.__doc__ or ""
            local_doc = target_func.__doc__ or ""
            target_func.__doc__ = f"{local_doc}\n\nFrom {module_path}.{func_name}:\n{source_doc}"
        except Exception:
            # If the import fails for any reason, exit silently
            pass
        return target_func
    return decorator