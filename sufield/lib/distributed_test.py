import torch

from .utils.distributed import get_rank

from .distributed_launch import run_distributed
from ..datasets.dataset import ToyDataset
from ..datasets.sampler import DistributedInfSampler, InfSampler


# @distributed(8)
def test():
    id = get_rank()
    print(f'hi. I\'m #{id}')

    dataset = ToyDataset(16)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=InfSampler(dataset))
    if id == 0:
        print(len(dataloader))
    for idx, i in enumerate(dataloader):
        print(f"rank {id}: round#{idx} {i}")
        if idx >= 15:
            break

if __name__ == '__main__':
    run_distributed(1, test)
