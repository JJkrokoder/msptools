from msptools.backend import get_backend
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
    
def test_backend():
    # Test CPU backend
    cpu_backend = get_backend(np.zeros(1))
    a_cpu = cpu_backend.array([1, 2, 3])
    assert isinstance(a_cpu, np.ndarray)
    
    if cp is not None:
        # Test GPU backend
        gpu_backend = get_backend(cp.zeros(1))
        a_gpu = gpu_backend.array([1, 2, 3])
        assert isinstance(a_gpu, cp.ndarray)