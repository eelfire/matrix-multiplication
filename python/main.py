import numpy as np
import timeit
import time

# A = BC + BD
def matmul(B, C, D):
    """Matrix multiplication of two numpy arrays"""
    return np.dot(B, (C + D))
    # return np.matmul(B, (C + D))
    # return np.multiply(B, (C + D))

# 2048 x 2048 matrix
# B = np.random.rand(2048, 2048)
# C = np.random.rand(2048, 2048)
# B = np.random.randint(0, 100, (2048, 2048), dtype='uint')
# C = np.random.randint(0, 100, (2048, 2048), dtype='uint')
# B = np.fromfunction(lambda i, j: i + j, (2048, 2048), dtype=int)
# C = np.fromfunction(lambda i, j: i + j, (2048, 2048), dtype=int)
B = np.ones((2048, 2048), dtype='uint')
C = np.ones((2048, 2048), dtype='uint')

# 5 x 5 matrix
# B5 = np.random.rand(5, 5)
# C5 = np.random.rand(5, 5)
B5 = np.ones((5, 5), dtype='uint')
C5 = np.ones((5, 5), dtype='uint')

# Time the function
# execution_time = timeit.timeit('matmul(B, (C + D))', globals=globals(), number=1)
# print(f"Execution time: {execution_time} seconds")

def print_head_tail(m):
    print(m[0])
    print(m[-1])

def matmul_sequential(B, C, D):
    start = time.time()
    A = matmul(B, C, D)
    end = time.time()
    print(f"Execution time: {end - start} seconds")

    print_head_tail(A)


# Introduce multithreaded generation of matrix
import multiprocessing as mp

# def matmul_parallel(B, C, D, partition, n_processes=4):
#     A = np.zeros((2048, 2048), dtype='uint')
#     # partition A into n_processes parts
#     # calculate the start and end indices for each part in A using partition and partition size
#     # use numpy to calculate the dot product of B and (C + D) for that partition
#     # store the result in the corresponding part of A


#     return A

# def matmul_mp(B, C, D):
#     start = time.time()

#     # Create a pool of workers
#     pool = mp.Pool(16)

#     # Create a list of tasks
#     tasks = [(B, C, D)] * 16

#     # Run the tasks
#     results = pool.starmap(matmul, tasks)

#     # Close the pool
#     pool.close()

#     # Combine the results
#     # A = sum(results)
#     A = results

#     end = time.time()
#     print(f"Execution time: {end - start} seconds")

#     print_head_tail(A)

from concurrent.futures import ProcessPoolExecutor

def matmul_partition(B, C, D, partition):
    start, end = partition
    result = np.matmul(B, (C + D)[:, start:end])
    return result

def matmul_parallelize(B, C, D, partition, n_processes=4):
    A = np.zeros((2048, 2048), dtype='uint')

    # Create a ProcessPoolExecutor with the specified number of processes
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Create a list of futures for parallel execution
        futures = []

        # Split A into partitions and submit tasks for parallel processing
        partition_size = A.shape[1] // n_processes
        for i in range(n_processes):
            start = i * partition_size
            end = (i + 1) * partition_size if i < n_processes - 1 else A.shape[1]
            partition_range = (start, end)

            # Submit a task to the executor for each partition
            future = executor.submit(matmul_partition, B, C, D, partition_range)
            futures.append((future, partition_range))

        # Wait for all tasks to complete and update the corresponding part of A
        for future, partition_range in futures:
            result = future.result()
            start, end = partition_range
            A[:, start:end] = result

    return A

def matmul_parallel(B, C, D, partition, n_processes=4):
    start = time.time()
    A = matmul_parallelize(B, C, D, partition, n_processes)
    end = time.time()
    print(f"Execution time: {end - start} seconds")

    print_head_tail(A)


# matmul_sequential(B, C, D)
# matmul_mp(B, C, C)
matmul_parallel(B, C, C, (0, 2048), n_processes=4)
# matmul_mp(B5, C5, C5)
