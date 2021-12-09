from multiprocessing import Pool, shared_memory
import numpy as np
from pathos.multiprocessing import ProcessingPool
import time
from functools import wraps


def parallel_gen(f, array_shape, array_dtype, nproc=16, shared_object=None) -> np.ndarray:
    
    target = np.ndarray(array_shape, dtype=array_dtype)
    shm_a = shared_memory.SharedMemory(create=True, size=target.nbytes)
    shm_b = shared_memory
    shared = np.ndarray(array_shape, dtype=array_dtype, buffer=shm_a.buf)
    pool = Pool(nproc)
    pool.map() # TODO
    pool.close()
    pool.join()

    target = shared.copy()

    shm_a.close()
    shm_a.unlink()
    return target




def f(args):
    idx, shape, dtype, name = args
    time.sleep(5)
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr[idx] = np.array(list(range(idx, idx+5)))


if __name__ == '__main__':

    a = np.ndarray((5, 5))
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    pool = ProcessingPool(nodes=5)
    inputs = [0, 1, 2, 3, 4]
    results = pool.uimap(f, [(i, a.shape, a.dtype, shm.name) for i in inputs])
    # while not results.ready():
    #     print(".", end='')
    #     time.sleep(1)
    pool.close()
    pool.join()
    print(b)
    shm.close()
    shm.unlink()
