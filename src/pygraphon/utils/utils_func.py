"""Random utils to deal with python functions."""
import functools
import types


def copy_func(f):
    """Copy a function.

    Will create a completely new function with the same code, globals, defaults, closure,
    unrelated to the original function.
    Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)

    Parameters
    ----------
    f : Callable
        function to copy

    Returns
    -------
    Callable
        copy of the function
    """
    g = types.FunctionType(
        f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__
    )
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
