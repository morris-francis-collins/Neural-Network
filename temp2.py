import time
import numpy as np
import cupy as cu

for power in range(5):
    dim = 10 ** power
    print(f"matrix dim, {dim}x{dim}")

    start = time.time()
    X = cu.random.randn(dim, dim)
    Y = cu.random.randn(dim, dim)
    Z = X @ Y
    end = time.time()
    print("gpu", end - start)

    start = time.time()
    X = np.random.randn(dim, dim)
    Y = np.random.randn(dim, dim)
    Z = X @ Y
    end = time.time()
    print("cpu", end - start)