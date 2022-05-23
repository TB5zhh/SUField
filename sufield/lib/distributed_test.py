import random
import torch
import torch.multiprocessing as mp 
import torch.distributed as dist
from ..datasets.dataset import ToyDataset
from ..datasets.sampler import DistributedInfSampler, InfSampler
def main():
    port = random.randint(10000, 20000)
    distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
    world_size = 1
    if world_size > 1:
        mp.spawn(
            fn=runner, 
            args=(distributed_init_method, world_size),
            nprocs = world_size
        )
    else:
        runner(0, None, 1)

def runner(id, init_method, world_size):
    print(f'hi. I\'m #{id}')
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=id)
    if id == 0:
        print('init ok')
    # dist.all_reduce_multigpu(torch.zeros(1).to(f'cuda:{id}'))
    # if id == 0:
    #     print('reduce ok')

    dataset = ToyDataset(16)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=InfSampler(dataset))
    if id == 0:
        print(len(dataloader))
    for idx, i in enumerate(dataloader):
        print(f"rank {id}: round#{idx} {i}")
        if idx >= 16:
            break

if __name__ == '__main__':
    main()