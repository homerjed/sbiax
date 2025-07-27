import time
import jax.numpy as jnp
import numpy as np
import os

def benchmark_matmul(n=3000, runs=5):
    print("Benchmarking matrix multiplication...")
    print(f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}")
    print(f"MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS')}")
    print(f"OPENBLAS_NUM_THREADS={os.getenv('OPENBLAS_NUM_THREADS')}")
    print()

    # Initialize large matrices
    A = jnp.ones((n, n)).astype(np.float64)
    B = jnp.ones((n, n)).astype(np.float64)

    times = []
    for i in range(runs):
        t0 = time.time()
        C = A @ B  # matrix multiplication
        t1 = time.time()
        dt = t1 - t0
        times.append(dt)
        print(f"Run {i+1}: {dt:.3f} s")

    avg_time = sum(times) / len(times)
    print(f"\nAverage time over {runs} runs: {avg_time:.3f} seconds")

if __name__ == "__main__":
    benchmark_matmul()
