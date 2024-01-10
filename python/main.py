import numpy as np
import timeit

def matmul(A, B):
    """Matrix multiplication of two numpy arrays"""
    return np.dot(A, B)

# 2048 x 2048 matrix
A = np.random.rand(2048, 2048)
B = np.random.rand(2048, 2048)

# print(A)

# Time the function
execution_time = timeit.timeit('matmul(A, B)', globals=globals(), number=1)
print(f"Execution time: {execution_time} seconds")
