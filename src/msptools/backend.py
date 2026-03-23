from numpy.typing import ArrayLike
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

def get_backend(array: ArrayLike) -> None:
    _backend = None
    if isinstance(array, np.ndarray):
        _backend = np
    elif isinstance(array, cp.ndarray) and cp is not None:
        _backend = cp
    else:
        raise ValueError("Unsupported array type. Only NumPy and CuPy arrays are supported.")
    return _backend



    


