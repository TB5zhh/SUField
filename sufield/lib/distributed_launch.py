import random
from functools import wraps
import torch.multiprocessing
import torch.distributed as dist


def _distributed_init(id, world_size, init_method, fn, args):
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=id)
    # dist.all_reduce_multigpu(torch.zeros(1).to(f"cuda:{id}"))
    try:
        fn(*args)
    finally:
        dist.destroy_process_group()


def distributed(world_size):
    raise RuntimeError
    if world_size > 1:

        def decorator(fn):

            @wraps(_distributed_init)
            def wrapper(*args):
                port = random.randint(10000, 20000)
                init_method = f'tcp://localhost:{port}'
                torch.multiprocessing.spawn(fn=_distributed_init, args=(world_size, init_method, fn, args), nprocs=world_size)

            return wrapper
    else:

        def decorator(fn):
            return fn

    return decorator


def run_distributed(world_size, fn, *args):
    port = random.randint(10000, 20000)
    init_method = f'tcp://localhost:{port}'
    if world_size > 1:
        torch.multiprocessing.spawn(fn=_distributed_init, args=(world_size, init_method, fn, args), nprocs=world_size)
    else:
        fn(*args)
